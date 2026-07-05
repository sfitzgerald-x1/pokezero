"""Self-validation for the v2.2 token-format documentation (docs/token-format/).

The doc pipeline is: extract_turn16.py -> turn16-token-dump.json ->
generate_token_doc.py -> token-format-v2_2.html, with the JSON and HTML committed so
the reference is browsable from the repo. This test is the anti-drift latch:

- it REGENERATES the dump from the committed seed-148 explosion fixture through the
  live v2.2 encoder and asserts byte-identity with the committed JSON — any encoder
  change that touches the observation surface fails here, forcing a doc regeneration
  instead of letting the documentation silently rot;
- it asserts the committed HTML carries sentinel values from the dump (the Explosion
  turn token index, the merged cold-replacement pair token index, and the exact
  encoded Explosion damage fraction), so the HTML cannot be stale relative to the
  JSON it claims to render;
- plus a tag-balance sanity check on the HTML.

Requires a built Gen 3 Showdown checkout (same skip gate as the corpus tests).
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC_DIR = REPO / "docs" / "token-format"
DUMP_PATH = DOC_DIR / "turn16-token-dump.json"
HTML_PATH = DOC_DIR / "token-format-v2_2.html"
SHOWDOWN_ROOT = Path(
    os.environ.get("POKEZERO_SHOWDOWN_ROOT", "/Users/scott/workspace/pokerena/vendor/pokemon-showdown")
)


@unittest.skipUnless(
    (SHOWDOWN_ROOT / "data" / "random-battles" / "gen3" / "sets.json").exists(),
    "requires a local Gen 3 Pokemon Showdown checkout",
)
class TokenFormatDocSelfValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # The extractor records this env value into the dump's provenance section;
        # pin it to the documented setting regardless of the ambient environment —
        # and RESTORE it afterwards (other suites, e.g. the harness set-source
        # gates, are sensitive to this variable being set).
        previous = os.environ.get("POKEZERO_BELIEF_SET_SOURCE")

        def _restore() -> None:
            if previous is None:
                os.environ.pop("POKEZERO_BELIEF_SET_SOURCE", None)
            else:
                os.environ["POKEZERO_BELIEF_SET_SOURCE"] = previous

        cls.addClassCleanup(_restore)
        os.environ["POKEZERO_BELIEF_SET_SOURCE"] = "1"
        if str(DOC_DIR) not in sys.path:
            sys.path.insert(0, str(DOC_DIR))
        cls.extract = importlib.import_module("extract_turn16")
        cls.committed_text = DUMP_PATH.read_text(encoding="utf-8")
        cls.committed = json.loads(cls.committed_text)
        cls.html = HTML_PATH.read_text(encoding="utf-8")

    def test_committed_dump_matches_live_regeneration_byte_for_byte(self) -> None:
        regenerated = json.dumps(self.extract.dump, indent=1) + "\n"
        if regenerated != self.committed_text:
            # Give a structured diff hint before failing on the raw text.
            self.assertEqual(
                self.extract.dump,
                self.committed,
                "docs/token-format/turn16-token-dump.json no longer matches the live "
                "v2.2 encoder output. The observation surface changed — regenerate the "
                "doc: uv run python docs/token-format/extract_turn16.py && "
                "uv run python docs/token-format/generate_token_doc.py",
            )
        self.assertEqual(
            regenerated,
            self.committed_text,
            "dump content matches but serialization differs — regenerate with "
            "extract_turn16.py (json.dumps indent=1 + trailing newline)",
        )

    def test_dump_pins_the_documented_boundary_shape(self) -> None:
        layout = self.committed["layout"]
        self.assertEqual(layout["schema_version"], "pokezero.observation.v2.2")
        self.assertEqual(layout["token_count"], 151)
        # 155 = the post-#519 census (153 + the per-sub-block SELF_HP_COST pair).
        self.assertEqual(layout["numeric_feature_count"], 155)
        self.assertEqual(layout["categorical_feature_count"], 51)
        self.assertEqual(layout["attention_mask_at_boundary"]["transition_attended"], 18)
        self.assertEqual(layout["per_action_token_count_same_boundary"], 35)
        # The Explosion pair: the worked example is the turn-7 PHASE_TURN token with
        # the cold-replacement pair as its companion.
        explosion = self.committed["line_to_token_examples"][0]
        self.assertEqual(explosion["decoded_fields"]["turn"], 7)
        self.assertEqual(explosion["decoded_fields"]["phase"], "turn")
        second = explosion["decoded_fields"]["second"]
        self.assertEqual(second["action"], "explosion")
        self.assertTrue(second["ko"] and second["crit"])
        pair = explosion["companion_tokens"][0]["decoded_fields"]
        self.assertEqual(pair["phase"], "replacement")
        self.assertEqual(pair["first"]["status"], "action")
        self.assertEqual(pair["second"]["status"], "action")
        # The #519 self-cost surface, on real lines: Explosion's terminal 1.0 plus the
        # turn-1 self-cost anatomy example with BOTH sub-blocks nonzero (Substitute's
        # exact quarter on A vs Double-Edge recoil on B).
        self.assertEqual(
            explosion["encoded_token"]["numerics"]["NUMERIC_TM2_SELF_HP_COST"], 1.0
        )
        self_cost = self.committed["line_to_token_examples"][1]
        self.assertEqual(self_cost["decoded_fields"]["turn"], 1)
        self.assertEqual(
            self_cost["encoded_token"]["numerics"]["NUMERIC_TT_SELF_HP_COST"], 0.25
        )
        self.assertEqual(
            self_cost["encoded_token"]["numerics"]["NUMERIC_TM2_SELF_HP_COST"], 0.072785
        )

    def test_committed_html_carries_dump_sentinels(self) -> None:
        explosion = self.committed["line_to_token_examples"][0]
        turn_token_index = explosion["token_index"]
        pair_token_index = explosion["companion_tokens"][0]["token_index"]
        damage = explosion["decoded_fields"]["second"]["damage_fraction"]
        for sentinel in (
            f"token {turn_token_index}",  # the Explosion turn token card
            f"token {pair_token_index}",  # the merged cold-replacement pair card
            f"{damage:.6g}",  # the encoded Explosion damage fraction (0.514403)
            "tt2_move:explosion",
            "tt_phase:replacement",
            "tt2_status:absent",
            "NUMERIC_TM2_SELF_HP_COST",  # the #519 self-cost surface must be featured
            self.committed["masks"]["belief_set_source"]["source_hash"],
        ):
            self.assertIn(
                str(sentinel),
                self.html,
                f"committed HTML is stale: sentinel {sentinel!r} from the dump is "
                "missing — regenerate with generate_token_doc.py",
            )

    def test_committed_html_tag_balance(self) -> None:
        counts: dict[str, list[int]] = {}
        for match in re.finditer(r"<(/?)([a-zA-Z0-9]+)", self.html):
            closing, tag = match.group(1) == "/", match.group(2).lower()
            counts.setdefault(tag, [0, 0])[1 if closing else 0] += 1
        unbalanced = {
            tag: pair
            for tag, pair in counts.items()
            if pair[0] != pair[1] and tag not in ("br", "hr", "meta", "link")
        }
        self.assertEqual(unbalanced, {}, "unbalanced HTML tags in the committed doc")


if __name__ == "__main__":
    unittest.main()
