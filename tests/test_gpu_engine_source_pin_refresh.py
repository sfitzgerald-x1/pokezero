"""The engine's reviewed source pins are current at HEAD (scripts/refresh_source_pins.py --check); pure Python."""
import contextlib
import io
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import refresh_source_pins as refresh  # noqa: E402


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = refresh.main(list(argv))
    return code, out.getvalue()


class SourcePinRefreshTests(unittest.TestCase):
    def test_check_passes_at_head(self):
        code, out = run('--root', str(ROOT), '--check')
        self.assertEqual(code, 0, out)
        self.assertIn('pins current', out)

    def test_a_stale_pin_fails_check_and_the_refresh_reaches_a_fixed_point(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            (root/'gpu_engine').mkdir()
            for path in (ROOT/'gpu_engine').glob('*.py'):
                shutil.copy2(path, root/'gpu_engine'/path.name)
            residual = root/'gpu_engine'/'residual_order.py'
            residual.write_text(residual.read_text() + '\n# a reviewed edit\n')
            code, out = run('--root', str(root), '--check')
            self.assertEqual(code, 1)
            stale = dict(line[len('STALE '):].split(': ', 1) for line in out.splitlines() if line.startswith('STALE '))
            for name in ('engine_source_frame_witness.py', 'chance_consumers.py', 'phase_d_chance.py'):
                self.assertEqual(stale[name].split(', ')[0], "SOURCE_PINS['residual_order.py']")
            self.assertEqual(stale['conditional_law.py'], 'KERNEL_SHA256')
            # chance_consumers.py itself changes in the refresh, so ordinary_law's mapping pin of it moves too: the
            # fixed point covers pins of pinned files.
            self.assertEqual(stale['ordinary_law.py'], "MAPPING_PINS['chance_consumers.py'], DOMAIN_SHA256")
            code, out = run('--root', str(root))
            self.assertEqual(code, 0, out)
            self.assertEqual(run('--root', str(root), '--check')[0], 0)


if __name__ == '__main__':
    unittest.main()
