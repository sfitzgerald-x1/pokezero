"""The engine's reviewed source pins are current at HEAD (scripts/refresh_source_pins.py --check), in the layout that lets
concurrent refreshes of different files merge, with the counts in one place; pure Python."""
import contextlib
import io
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import refresh_source_pins as refresh  # noqa: E402
from tests import _source_pins as pins  # noqa: E402


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = refresh.main(list(argv))
    return code, out.getvalue()


def scratch_engine(scratch):
    root = Path(scratch)
    (root/'gpu_engine').mkdir()
    for path in (ROOT/'gpu_engine').glob('*.py'):
        shutil.copy2(path, root/'gpu_engine'/path.name)
    return root


class SourcePinRefreshTests(unittest.TestCase):
    def test_check_passes_at_head(self):
        code, out = run('--root', str(ROOT), '--check')
        self.assertEqual(code, 0, out)
        self.assertIn('pins current', out)

    def test_every_table_is_counted_here_and_is_what_the_refresher_reads(self):
        """A pinned file added or dropped is a deliberate act: it changes PIN_COUNTS (tests/_source_pins.py) and nothing
        else. The counted tables are exactly the ones the refresher finds in its files, each as the module would load
        it (the same literal the refresher rewrites), naming existing gpu_engine files; a per-module table names its
        module list in order; and no aggregate DOMAIN_SHA256 / KERNEL_SHA256 literal is left."""
        found = {(name, var) for name in refresh.FILES for var in refresh.PIN_DICTS
                 if refresh._pins((ROOT/'gpu_engine'/name).read_text(), var)}
        self.assertEqual(found, set(pins.PIN_COUNTS))
        self.assertEqual(set(pins.DOMAIN_TABLES), set(refresh.LISTED))
        for (name, var), count in pins.PIN_COUNTS.items():
            with self.subTest(table=f'{name}:{var}'):
                text = (ROOT/'gpu_engine'/name).read_text()
                loaded = pins.table(name, var)
                self.assertEqual(len(loaded), count)
                self.assertEqual(loaded, refresh.ast.literal_eval(refresh._pins(text, var).group(1)))
                self.assertTrue(all((ROOT/'gpu_engine'/file).is_file() for file in loaded))
                if (name, var) in pins.DOMAIN_TABLES:
                    holder, listing = pins.DOMAIN_TABLES[(name, var)]
                    self.assertEqual(list(loaded), [m + '.py' for m in pins.modules(holder, listing)])
        for name in refresh.FILES:
            self.assertIsNone(re.search(r'^(DOMAIN|KERNEL)_SHA256 = ', (ROOT/'gpu_engine'/name).read_text(), re.M), name)

    def test_a_stale_pin_fails_check_and_the_refresh_reaches_a_fixed_point(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = scratch_engine(scratch)
            residual = root/'gpu_engine'/'residual_order.py'
            residual.write_text(residual.read_text() + '\n# a reviewed edit\n')
            code, out = run('--root', str(root), '--check')
            self.assertEqual(code, 1)
            stale = dict(line[len('STALE '):].split(': ', 1) for line in out.splitlines() if line.startswith('STALE '))
            self.assertEqual(stale['engine_source_frame_witness.py'], "SOURCE_PINS['residual_order.py']")
            for name in ('chance_consumers_pins.py', 'phase_d_chance.py'):
                self.assertEqual(stale[name], "SOURCE_PINS['residual_order.py'], DOMAIN_PINS['residual_order.py']")
            self.assertEqual(stale['conditional_law.py'], "KERNEL_PINS['residual_order.py']")
            # chance_consumers.py holds no pins any more, so a domain edit no longer re-pins it, and through it
            # ordinary_law's MAPPING_PINS: the cascade that made every two domain edits conflict is gone.
            self.assertEqual(stale['ordinary_law.py'], "DOMAIN_PINS['residual_order.py']")
            code, out = run('--root', str(root))
            self.assertEqual(code, 0, out)
            self.assertEqual(run('--root', str(root), '--check')[0], 0)

    def test_an_old_layout_is_format_drift_and_a_refresh_rewrites_it(self):
        """A single-line or unsorted table (e.g. from a branch older than this layout) fails --check; a refresh writes
        the canonical layout without changing any value."""
        with tempfile.TemporaryDirectory() as scratch:
            root = scratch_engine(scratch)
            witness = root/'gpu_engine'/'engine_source_frame_witness.py'
            text = witness.read_text()
            match = refresh._pins(text, 'SOURCE_PINS')
            values = refresh.ast.literal_eval(match.group(1))
            witness.write_text(text[:match.start(1)] + repr(values) + text[match.end(1):])   # the old one-line layout
            code, out = run('--root', str(root), '--check')
            self.assertEqual(code, 1)
            self.assertIn('FORMAT engine_source_frame_witness.py: SOURCE_PINS', out)
            self.assertEqual(run('--root', str(root))[0], 0)
            self.assertEqual(witness.read_text(), text)
            law = root/'gpu_engine'/'conditional_law.py'
            text = law.read_text()
            match = refresh._pins(text, 'KERNEL_PINS')
            entries = list(refresh.ast.literal_eval(match.group(1)).items())
            swapped = dict([entries[1], entries[0]] + entries[2:])   # out of KERNEL_MODULES order
            law.write_text(text[:match.start(1)] + refresh.canonical(swapped, [k[:-3] for k in swapped]) + text[match.end(1):])
            code, out = run('--root', str(root), '--check')
            self.assertEqual(code, 1)
            self.assertIn('FORMAT conditional_law.py: KERNEL_PINS', out)
            self.assertEqual(run('--root', str(root))[0], 0)
            self.assertEqual(law.read_text(), text)

    def test_noncanonical_assignment_spacing_cannot_hide_a_table(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = scratch_engine(scratch)
            path = root/'gpu_engine'/'chance_consumers_pins.py'
            text = path.read_text()
            match = refresh._pins(text, 'DOMAIN_PINS')
            path.write_text(text[:match.start()] + 'DOMAIN_PINS=' + repr(pins.table(path.name, 'DOMAIN_PINS', text))
                            + text[match.end():])
            code, out = run('--root', str(root), '--check')
            self.assertEqual(code, 1)
            self.assertIn('FORMAT chance_consumers_pins.py: DOMAIN_PINS', out)
            self.assertEqual(run('--root', str(root))[0], 0)
            self.assertEqual(path.read_text(), text)

    def test_missing_or_duplicate_table_definition_refuses_without_writing(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = scratch_engine(scratch)
            path = root/'gpu_engine'/'chance_consumers_pins.py'
            text = path.read_text()
            match = refresh._pins(text, 'DOMAIN_PINS')
            for broken in (text[:match.start()] + text[match.end():], text + '\n' + match[0] + '\n'):
                path.write_text(broken)
                for flag in ((), ('--check',)):
                    code, out = run('--root', str(root), *flag)
                    self.assertEqual(code, 1)
                    self.assertIn('TABLE chance_consumers_pins.py: DOMAIN_PINS', out)
                    self.assertEqual(path.read_text(), broken)

    def test_duplicate_domain_module_refuses_without_writing(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = scratch_engine(scratch)
            path = root/'gpu_engine'/'conditional_law.py'
            text = path.read_text()
            broken = text.replace('KERNEL_MODULES = tuple("', 'KERNEL_MODULES = tuple("absorption ', 1)
            self.assertNotEqual(broken, text)
            path.write_text(broken)
            for flag in ((), ('--check',)):
                code, out = run('--root', str(root), *flag)
                self.assertEqual(code, 1)
                self.assertIn('LIST conditional_law.py: KERNEL_PINS', out)
                self.assertEqual(path.read_text(), broken)

    def test_an_aggregate_literal_or_a_list_mismatch_is_refused_not_written(self):
        """Both are review decisions: a DOMAIN_SHA256 literal back from an older branch, or a per-module table that no
        longer names its module list (a module added or dropped on one side). --check names them; a refresh writes
        nothing."""
        with tempfile.TemporaryDirectory() as scratch:
            root = scratch_engine(scratch)
            law = root/'gpu_engine'/'ordinary_law.py'
            before = law.read_text()
            law.write_text(before.replace('DOMAIN_PINS = {', "DOMAIN_SHA256 = '" + '0' * 64 + "'\nDOMAIN_PINS = {", 1))
            code, out = run('--root', str(root), '--check')
            self.assertEqual(code, 1)
            self.assertIn('AGGREGATE ordinary_law.py: DOMAIN_SHA256', out)
            stamp = law.read_text()
            code, out = run('--root', str(root))
            self.assertEqual(code, 1)
            self.assertIn('REFUSED AGGREGATE ordinary_law.py', out)
            self.assertEqual(law.read_text(), stamp)
            law.write_text(before)
            phase = root/'gpu_engine'/'phase_d_chance.py'
            text = phase.read_text()
            match = refresh._pins(text, 'DOMAIN_PINS')
            shortened = re.sub(r"^    'weather\.py': '[0-9a-f]{64}',\n", '', match.group(1), count=1, flags=re.M)
            phase.write_text(text[:match.start(1)] + shortened + text[match.end(1):])
            self.assertNotEqual(phase.read_text(), text)
            code, out = run('--root', str(root), '--check')
            self.assertEqual(code, 1)
            self.assertIn('LIST phase_d_chance.py: DOMAIN_PINS does not name DOMAIN_MODULES exactly', out)
            stamp = phase.read_text()
            self.assertEqual(run('--root', str(root))[0], 1)
            self.assertEqual(phase.read_text(), stamp)


if __name__ == '__main__':
    unittest.main()
