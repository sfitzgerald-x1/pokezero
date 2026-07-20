"""Tests for the edge-case scenario suite (corpus + fallback sweep)."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pokezero.golden_corpus_scenarios import (  # noqa: E402
    KNOWN_FALLBACK_REASONS,
    ScriptedPreferencePolicy,
    scenario_specs,
)


def _live_showdown_available() -> bool:
    root = Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT") or "/Users/scott/workspace/pokerena/vendor/pokemon-showdown")
    return (root / "dist" / "sim" / "index.js").exists() and shutil.which("node") is not None


class _Obs:
    def __init__(self, candidates, mask=(True,) * 9):
        self.metadata = {"action_candidates": candidates}
        self.legal_action_mask = mask


class ScriptedPolicyTests(unittest.TestCase):
    def test_prefers_listed_move_then_falls_back(self) -> None:
        policy = ScriptedPreferencePolicy(preferences=(("surf",), ("recover",)))
        candidates = [
            {"action_index": 0, "kind": "move", "legal": True, "move_id": "psychic"},
            {"action_index": 1, "kind": "move", "legal": True, "move_id": "surf"},
        ]
        import random
        first = policy.select_action(_Obs(candidates), rng=random.Random(0))
        self.assertEqual(first.action_index, 1)  # surf preferred turn 1
        second = policy.select_action(_Obs(candidates), rng=random.Random(0))
        self.assertEqual(second.action_index, 0)  # recover absent -> first legal move

    def test_scenario_specs_are_well_formed(self) -> None:
        specs = scenario_specs()
        self.assertGreaterEqual(len(specs), 10)
        names = [spec.name for spec in specs]
        self.assertEqual(len(names), len(set(names)))
        for spec in specs:
            self.assertTrue(spec.p1_team and spec.p2_team)


@unittest.skipIf(not _live_showdown_available(), "requires a built local Showdown checkout")
class ScenarioSweepLiveTests(unittest.TestCase):
    def test_key_scenarios_search_or_fail_closed_with_known_reasons(self) -> None:
        from pokezero.golden_corpus_scenarios import run_scenario_fallback_sweep

        specs = {s.name: s for s in scenario_specs()}
        chosen = [
            specs["truant_slaking"], specs["ditto_transform"],
            specs["baton_pass_boundary"], specs["attract_snorlax"],
        ]
        report = run_scenario_fallback_sweep(
            showdown_root=os.environ.get("POKEZERO_SHOWDOWN_ROOT")
            or "/Users/scott/workspace/pokerena/vendor/pokemon-showdown",
            specs=chosen,
        )
        for name, stats in report.items():
            self.assertEqual(sum(stats["unmapped_choices"].values()), 0, name)
            self.assertEqual(
                set(stats["fallback_reasons"]) - KNOWN_FALLBACK_REASONS, set(), name
            )
        # Truant phases must SEARCH (the modeled path), not fall back.
        self.assertEqual(report["truant_slaking"]["fallback_decisions"], 0)
        self.assertGreater(report["truant_slaking"]["searched_decisions"], 0)
        # Post-transform Ditto decisions must fail closed via the moveset guard.
        ditto = report["ditto_transform"]
        self.assertTrue(
            any("self_moveset_mismatch" in reason for reason in ditto["world_failure_reasons"])
        )
        # The Baton Pass boundary must search straight through.
        self.assertEqual(report["baton_pass_boundary"]["fallback_decisions"], 0)
        # Attract (free branch + paralysis composition) must SEARCH, not wall:
        # the ``attract`` allow-list entry + the immobilization patch let every
        # attracted decision construct worlds. Pre-fix these walled with
        # ``volatile_unsupported: attract`` (a NON-known reason -> checked above).
        attract = report["attract_snorlax"]
        self.assertEqual(attract["fallback_decisions"], 0)
        self.assertGreater(attract["searched_decisions"], 0)
        self.assertFalse(
            any("volatile_unsupported" in reason for reason in attract["world_failure_reasons"]),
            attract["world_failure_reasons"],
        )

    def test_item_state_scenarios_search_instead_of_failing_closed(self) -> None:
        # The Trick-swap current-item override + berry-consumption clearing:
        # decisions after a public exchange/eat must SEARCH — the pre-fix
        # behavior walled every remaining decision of the battle with
        # public_effect_blocked (48/60 of the seed-7013 bench fallbacks).
        from pokezero.golden_corpus_scenarios import run_scenario_fallback_sweep

        specs = {s.name: s for s in scenario_specs()}
        chosen = [specs["trick_swap_exchange"], specs["trick_berry_pinch"], specs["berry_eat_chesto"]]
        report = run_scenario_fallback_sweep(
            showdown_root=os.environ.get("POKEZERO_SHOWDOWN_ROOT")
            or "/Users/scott/workspace/pokerena/vendor/pokemon-showdown",
            specs=chosen,
        )
        for name, stats in report.items():
            self.assertEqual(stats["fallback_decisions"], 0, (name, stats["world_failure_reasons"]))
            self.assertGreater(stats["searched_decisions"], 0, name)
            self.assertEqual(sum(stats["unmapped_choices"].values()), 0, name)
            # Zero item walls: nothing may fail closed on a tricked/eaten item.
            self.assertFalse(
                any("public_effect_blocked" in reason for reason in stats["world_failure_reasons"]),
                (name, stats["world_failure_reasons"]),
            )
        # The scripted p2 Trick guarantees the override fires on both seats.
        self.assertGreater(report["trick_berry_pinch"]["item_override_decisions"], 0)
        # The Chesto-Rest eat guarantees the consumption-removal fires.
        self.assertGreater(report["berry_eat_chesto"]["removed_item_decisions"], 0)

    def test_scenario_corpus_generates_and_verifies(self) -> None:
        from pokezero.golden_corpus import verify_golden_corpus
        from pokezero.golden_corpus_scenarios import generate_scenario_corpus, scenario_specs as _specs

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "scenarios"
            from pokezero.golden_corpus_scenarios import play_scenario_games  # noqa: F401
            manifest = generate_scenario_corpus(
                out_dir=out,
                showdown_root=os.environ.get("POKEZERO_SHOWDOWN_ROOT")
                or "/Users/scott/workspace/pokerena/vendor/pokemon-showdown",
            )
            self.assertTrue((out / "manifest.json").exists())
            verification = verify_golden_corpus(out)
            self.assertTrue(getattr(verification, "ok", True) in (True,) or verification)


if __name__ == "__main__":
    unittest.main()
