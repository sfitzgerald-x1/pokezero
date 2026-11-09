"""Profile marks' phase site (gpu_engine.v2e_j2c, profile=True only); CPU only: generation, no CUDA build.

Non-profile generation must stay byte-identical: transition.step on a tiny table, in both modes, at the default and at
small chunk sizes, against the generator as of BASE (the commit the site was added on; the test fails, naming the
commit, when git or that commit is unavailable). A profile build must be the plain build plus its mark lines, chunk for
chunk: mark lines do not count toward the chunk size, so profile marks and plain-build kernels name the same chunks.
"""
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from gpu_engine import v2e_gen, v2e_j2c

BASE = '5bc2fe8c2b2588ca8aae4e44cc7ab8f32ad25ea4'
ROOT = Path(__file__).resolve().parents[1]
# A stand-in oracle: its file is .../gpu_engine/transition.py and its function step_with_chance, as the real one.
# The jitted helper is called twice with the same avals, so JAX reuses one cached sub-jaxpr whose inner equations
# carry no outer frames: their phase line must come from the calling (top-level) equation.
SOURCE = '''import jax
import jax.numpy as jnp

@jax.jit
def helper(x):
    return x * 3 + 1


def step_with_chance(x, y):
    a = helper(x) + y
    b = helper(a) * y
    return jnp.where(b > 0, b, -b) + a
'''
LINES = dict(first=10, second=11, last=12)


def stand_in():
    folder = Path(tempfile.mkdtemp(prefix='v2e-marks-')) / 'gpu_engine';folder.mkdir()
    path = folder / 'transition.py';path.write_text(SOURCE)
    spec = importlib.util.spec_from_file_location('v2e_marks_stand_in', path);module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def base_generator():
    """gpu_engine/v2e_j2c.py as of BASE, imported under another name; AssertionError naming the commit without git or it."""
    try:text = subprocess.run(['git', '-C', str(ROOT), 'show', f'{BASE}:gpu_engine/v2e_j2c.py'], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise AssertionError(f'the byte-identity check needs commit {BASE} (the generator before the profile-mark site) in this '
            f'checkout: fetch it (git fetch origin {BASE}) or unshallow the clone ({error})') from None
    name = 'v2e_j2c_at_base';spec = importlib.util.spec_from_loader(name, loader=None);module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:exec(compile(text, f'{name}.py', 'exec'), module.__dict__)
    finally:sys.modules.pop(name, None)
    return module


def tiny_step():
    """(closed jaxpr, avals) of transition.step on a tiny synthetic table, traced as the generator traces (substitutions on)."""
    import jax.numpy as j
    from gpu_engine import transition as tr, v2e_modules
    from gpu_engine.state import empty_state
    types = ['_none', 'normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting', 'poison', 'ground', 'flying', 'psychic', 'bug', 'rock',
        'ghost', 'dragon', 'dark', 'steel']
    md = {'vocabulary': dict(moves=['_none', 'struggle', 'tackle'], abilities=['_none', 'pressure'], items=['_none', 'leftovers'],
            species=['_none', 'pikachu'], types=types),
        'moves': [None, dict(id='struggle', base_power=50, type=1, source_type='Normal', category='Physical', pp=1, accuracy=True, priority=0, flags={}),
            dict(id='tackle', base_power=35, type=1, source_type='Normal', category='Physical', pp=35, accuracy=95, priority=0, flags={'contact': 1})],
        'type_chart': np.ones((17, 17)).tolist()}
    table = tr.tables_from_metadata(md);b = 4
    state = empty_state(b)._replace(species=j.ones((b, 2, 6), j.int32), hp=j.full((b, 2, 6), 100, j.int32), max_hp=j.full((b, 2, 6), 100, j.int32),
        level=j.full((b, 2, 6), 80, j.int32), done=j.zeros(b, bool))
    with v2e_modules.trace_substitutions():
        closed, avals, _, _ = v2e_gen.trace(lambda s, a: tr.step(s, a, table), (state, j.zeros((b, 2), j.int32)))
    return closed, avals


MARK = re.compile(r'^\s*if \((b == 0 && t == 0|b == 0)\) err\[1 \+ \d+\] = \(uint32_t\)\(clock64\(\) - T0\);$')


def strip_marks(files):
    return {name: '\n'.join(line for line in text.split('\n') if not MARK.match(line)) for name, text in files.items()}


class ProfileMarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle = stand_in()
        x = np.arange(4, dtype=np.int32);cls.example = (x, x + 2)
        cls.closed, cls.avals, _, _ = v2e_gen.trace(cls.oracle.step_with_chance, cls.example)

    def generate(self, generator=v2e_j2c, mode='thread', profile=False):
        return generator.generate('t', self.closed, self.avals, profile=profile, mode=mode)

    def test_marks_record_the_step_with_chance_line_and_module(self):
        m = self.generate(profile=True)
        self.assertTrue(m.marks)
        for mark in m.marks:
            self.assertEqual(len(mark), 5)
            self.assertIn(mark[3], LINES.values())
            self.assertEqual(mark[4], 'transition')
        lines = [mark[3] for mark in m.marks]
        self.assertEqual(lines, sorted(lines))   # program order
        # Both helper calls emit a multiply; the second one's inner equation takes the second call's line, not the first's.
        muls = [mark[3] for mark in m.marks if mark[0] == 'mul']
        self.assertIn(LINES['first'], muls);self.assertIn(LINES['second'], muls)

    def test_marks_without_a_phase_frame_record_none(self):
        closed, avals, _, _ = v2e_gen.trace(lambda x: x * 3 + 1, (np.arange(4, dtype=np.int32),))
        m = v2e_j2c.generate('u', closed, avals, profile=True, mode='thread')
        self.assertTrue(m.marks)
        self.assertTrue(all(mark[3] is None for mark in m.marks))

    def test_stand_in_profile_build_is_the_plain_build_plus_mark_lines(self):
        for mode in ('thread', 'warp'):
            plain, profiled = self.generate(mode=mode), self.generate(mode=mode, profile=True)
            self.assertEqual(strip_marks(profiled.files), plain.files, mode)
            self.assertFalse(plain.marks)


def chunked(chunk_lines):
    """os.environ with V2E_CHUNK_LINES set (None: the generator's default chunk size)."""
    env = {k: v for k, v in os.environ.items() if k != 'V2E_CHUNK_LINES'}
    if chunk_lines is not None:env['V2E_CHUNK_LINES'] = chunk_lines
    return mock.patch.dict(os.environ, env, clear=True)


class StepBuildTest(unittest.TestCase):
    """transition.step on a tiny table: several chunks at both sizes, both modes."""
    @classmethod
    def setUpClass(cls):
        cls.closed, cls.avals = tiny_step()

    def generate(self, generator=v2e_j2c, mode='thread', profile=False):
        return generator.generate('s', self.closed, self.avals, profile=profile, mode=mode)

    def test_profile_build_is_the_plain_build_plus_mark_lines_chunk_for_chunk(self):
        for chunk_lines in (None, '400'):
            with chunked(chunk_lines):
                for mode in ('thread', 'warp'):
                    plain, profiled = self.generate(mode=mode), self.generate(mode=mode, profile=True)
                    self.assertGreater(plain.stats['chunks'], 1, (mode, chunk_lines))
                    self.assertEqual(profiled.stats['chunks'], plain.stats['chunks'], (mode, chunk_lines))
                    self.assertEqual(strip_marks(profiled.files), plain.files, (mode, chunk_lines))

    def test_plain_build_is_byte_identical_to_the_base_generator(self):
        base = base_generator()
        for chunk_lines in (None, '400'):
            with chunked(chunk_lines):
                for mode in ('thread', 'warp'):
                    now, then = self.generate(mode=mode), self.generate(base, mode=mode)
                    self.assertGreater(now.stats['chunks'], 1)
                    self.assertEqual(now.files, then.files, (mode, chunk_lines))
                    self.assertEqual(now.digest, then.digest)


if __name__ == '__main__':
    unittest.main()
