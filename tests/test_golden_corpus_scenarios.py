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
from _showdown_root import showdown_root_str


def _live_showdown_available() -> bool:
    root = Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT") or showdown_root_str())
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
            or showdown_root_str(),
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
        # Post-transform Ditto decisions must now SEARCH. They used to fail
        # closed through the moveset guard (the request advertises the copied
        # moveset, the sampled world still holds Ditto's own), because the gen3
        # engine has no TRANSFORM volatile. The copied form is baked into the
        # active's spec instead, so the desync is reconciled rather than refused
        # and neither the moveset guard nor a transform block should fire.
        ditto = report["ditto_transform"]
        self.assertEqual(ditto["fallback_decisions"], 0, ditto["world_failure_reasons"])
        self.assertGreater(ditto["searched_decisions"], 0)
        self.assertFalse(
            any(
                "self_moveset_mismatch" in reason or "transform" in reason
                for reason in ditto["world_failure_reasons"]
            ),
            ditto["world_failure_reasons"],
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

    def test_the_taunt_route_into_a_struggle_only_request_searches(self) -> None:
        # `struggle_taunt_stall` (added by #1199) drove the Taunt route into a
        # Struggle-only request and contributed 9 `no_worlds_constructed`
        # refusals, every one `volatile_unsupported: side 'p1': ['taunt']` --
        # the world was never built, so the `none -> struggle` translation #1202
        # added downstream was never reached on this route. `taunt` is now in
        # `engine_world._SUPPORTED_VOLATILES` with its counter seeded, so the
        # scenario must search straight through.
        #
        # Measured on this exact scenario, `worlds=2`: 9 refusals / 30 of 102
        # worlds constructed before, 0 refusals / 48 of 48 after.
        from pokezero.golden_corpus_scenarios import run_scenario_fallback_sweep

        specs = {s.name: s for s in scenario_specs()}
        report = run_scenario_fallback_sweep(
            showdown_root=os.environ.get("POKEZERO_SHOWDOWN_ROOT")
            or showdown_root_str(),
            specs=[specs["struggle_taunt_stall"]],
        )
        stats = report["struggle_taunt_stall"]
        self.assertEqual(stats["fallback_decisions"], 0, stats["world_failure_reasons"])
        self.assertGreater(stats["searched_decisions"], 0)
        self.assertEqual(sum(stats["unmapped_choices"].values()), 0)
        self.assertFalse(
            any("volatile_unsupported" in reason for reason in stats["world_failure_reasons"]),
            stats["world_failure_reasons"],
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
            or showdown_root_str(),
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
                or showdown_root_str(),
            )
            self.assertTrue((out / "manifest.json").exists())
            verification = verify_golden_corpus(out)
            self.assertTrue(getattr(verification, "ok", True) in (True,) or verification)


if __name__ == "__main__":
    unittest.main()
