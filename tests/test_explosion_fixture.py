"""Self-validating checks for the seed-148 explosion fixture.

The fixture is a gzipped gen3 randbats protocol log chosen for its dense early
history: a turn-7 Explosion double-faint (double cold-replacement phase), plus
status, Spikes, and a public berry eat all inside turn 16. Reused by the token
format documentation and the turn-merged-token tests; the manifest carries the
deterministic regeneration recipe (seed 148 at the recorded commit).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import unittest
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "showdown"
LOG_GZ = FIXTURE_DIR / "explosion-seed148.log.gz"
MANIFEST = FIXTURE_DIR / "explosion-seed148.manifest.json"


def load_explosion_fixture_lines() -> list[str]:
    """Decompress the fixture and return its protocol lines (shared helper)."""
    return gzip.open(LOG_GZ, "rb").read().decode().split("\n")


class ExplosionFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lines = load_explosion_fixture_lines()
        cls.manifest = json.loads(MANIFEST.read_text())

    def test_integrity_matches_manifest(self) -> None:
        clean = "\n".join(self.lines).encode()
        self.assertEqual(
            hashlib.sha256(clean).hexdigest(), self.manifest["sha256_clean_log"]
        )
        self.assertFalse([l for l in self.lines if l.startswith("|t:|")])

    def _turn_block(self, turn: int) -> list[str]:
        start = self.lines.index(f"|turn|{turn}")
        try:
            end = self.lines.index(f"|turn|{turn + 1}")
        except ValueError:
            end = len(self.lines)
        return self.lines[start:end]

    def test_turn7_explosion_double_faint(self) -> None:
        block = self._turn_block(7)
        self.assertIn("|move|p2a: Weezing|Explosion|p1a: Gligar", block)
        faints = [l for l in block if l.startswith("|faint|")]
        self.assertEqual(len(faints), 2, faints)
        # both sides replace cold after the double faint
        switches_after = [
            l for l in block[block.index(faints[-1]) :] if l.startswith("|switch|")
        ]
        self.assertEqual(len(switches_after), 2, switches_after)

    def test_richness_inside_turn_16(self) -> None:
        upto_16 = self.lines[: self.lines.index("|turn|17")]
        self.assertTrue(any("|-status|" in l and "tox" in l for l in upto_16))
        self.assertTrue(any(l.startswith("|-sidestart|") and "Spikes" in l for l in upto_16))
        self.assertTrue(any("|-enditem|" in l and "Liechi" in l and "eat" in l for l in upto_16))

    def test_game_reaches_recorded_length(self) -> None:
        self.assertIn("|turn|48", self.lines)
        self.assertTrue(any(l.startswith("|win|") for l in self.lines))


if __name__ == "__main__":
    unittest.main()
