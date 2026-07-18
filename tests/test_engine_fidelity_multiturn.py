"""Unit tests for the multi-turn fidelity differential's pure parts + a live smoke."""

from __future__ import annotations

import os
import shutil
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pokezero.engine_fidelity import TurnFeatures  # noqa: E402
from pokezero.engine_fidelity_multiturn import (  # noqa: E402
    check_expected_traces,
    cumulative_features,
    drift_adjusted,
    engine_step_choices,
    match_step_branch,
    observed_boost_deltas,
    step_changed_active,
)
from pokezero.showdown_fixture import (  # noqa: E402
    FixturePokemon,
    FixtureStep,
    MultiTurnFixtureResult,
    request_requires_action,
)

try:
    import poke_engine
except ImportError:  # pragma: no cover - native wheel absent
    poke_engine = None


def integration_config():
    """Mirror tests/test_showdown_fixture.py: only run when a built checkout + node exist."""

    from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT, LocalShowdownConfig

    root = Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT)
    if not (root / "dist" / "sim" / "index.js").exists():
        return None
    if shutil.which("node") is None:
        return None
    return LocalShowdownConfig(showdown_root=root, read_timeout_seconds=15.0)


def _features(**kwargs) -> TurnFeatures:
    base = dict(p1_hp=200, p2_hp=300, p1_status="NONE", p2_status="NONE",
                fainted=frozenset(), weather="NONE", side_conditions={})
    base.update(kwargs)
    return TurnFeatures(**base)


def _result(initial_lines, step_lines_list) -> MultiTurnFixtureResult:
    return MultiTurnFixtureResult(
        format_id="gen3customgame", seed=1,
        initial_protocol_lines=tuple(initial_lines),
        initial_requests={},
        steps=tuple(
            FixtureStep(choices={}, protocol_lines=tuple(lines), requests={}, terminal=False)
            for lines in step_lines_list
        ),
        terminal=False,
    )


class RequestRequiresActionTests(unittest.TestCase):
    def test_wait_and_missing_requests_need_no_choice(self) -> None:
        self.assertFalse(request_requires_action(None))
        self.assertFalse(request_requires_action({"wait": True, "side": {"id": "p2"}}))
        self.assertFalse(request_requires_action({"teamPreview": True}))

    def test_force_switch_and_active_requests_need_a_choice(self) -> None:
        self.assertTrue(request_requires_action({"forceSwitch": [True], "side": {}}))
        self.assertTrue(request_requires_action({"active": [{"moves": []}], "side": {}}))
        self.assertFalse(request_requires_action({"forceSwitch": [False], "side": {}}))


class BoostDeltaTests(unittest.TestCase):
    def test_boost_and_unboost_lines_accumulate_per_side(self) -> None:
        deltas = observed_boost_deltas([
            "|move|p2a: Snorlax|Curse|p2a: Snorlax",
            "|-boost|p2a: Snorlax|atk|1",
            "|-boost|p2a: Snorlax|def|1",
            "|-unboost|p2a: Snorlax|spe|1",
            "|-unboost|p1a: Politoed|atk|1",
            "|-damage|p1a: Politoed|150/200",
        ])
        self.assertEqual(deltas["p2"], {"attack": 1, "defense": 1, "speed": -1})
        self.assertEqual(deltas["p1"], {"attack": -1})

    def test_no_boost_lines_is_empty_both_sides(self) -> None:
        self.assertEqual(observed_boost_deltas(["|upkeep", "|turn|3"]), {"p1": {}, "p2": {}})


class DriftAdjustTests(unittest.TestCase):
    def test_observed_hp_shifts_by_accumulated_engine_offset(self) -> None:
        adjusted = drift_adjusted(
            _features(p1_hp=180, p2_hp=250),
            showdown_prev=_features(p1_hp=190, p2_hp=280),
            engine_prev=_features(p1_hp=195, p2_hp=270),
            p1_active_changed=False, p2_active_changed=False,
        )
        self.assertEqual(adjusted.p1_hp, 185)  # +(195-190)
        self.assertEqual(adjusted.p2_hp, 240)  # +(270-280)

    def test_faints_and_switched_actives_are_never_shifted(self) -> None:
        adjusted = drift_adjusted(
            _features(p1_hp=0, p2_hp=250),
            showdown_prev=_features(p1_hp=190, p2_hp=280),
            engine_prev=_features(p1_hp=195, p2_hp=270),
            p1_active_changed=False, p2_active_changed=True,
        )
        self.assertEqual(adjusted.p1_hp, 0)
        self.assertEqual(adjusted.p2_hp, 250)

    def test_step_changed_active_detects_switch_lines(self) -> None:
        lines = ["|switch|p1a: Starmie|Starmie, L80|227/227", "|-damage|p2a: Snorlax|300/387"]
        self.assertTrue(step_changed_active(lines, "p1"))
        self.assertFalse(step_changed_active(lines, "p2"))


class CumulativeFeatureTests(unittest.TestCase):
    def test_hp_and_status_persist_across_steps_while_faints_fold_per_step(self) -> None:
        initial = [
            "|switch|p1a: Swampert|Swampert, L80|301/301",
            "|switch|p2a: Starmie|Starmie, L80|227/227",
        ]
        step1 = ["|-damage|p2a: Starmie|150/227", "|-status|p2a: Starmie|tox"]
        step2 = ["|upkeep", "|turn|3"]  # p2 untouched this step
        result = _result(initial, [step1, step2])
        features = cumulative_features(result, initial + step1 + step2, step2)
        self.assertEqual(features.p2_hp, 150)  # carried, not unknown
        self.assertEqual(features.p2_status, "TOXIC")
        self.assertEqual(features.fainted, frozenset())

        step3 = ["|-damage|p2a: Starmie|0 fnt", "|faint|p2a: Starmie"]
        features = cumulative_features(result, initial + step1 + step2 + step3, step3)
        self.assertEqual(features.fainted, frozenset({"p2"}))


class EngineStepChoiceTests(unittest.TestCase):
    _GRASS_IVS = {"hp": 31, "atk": 30, "def": 31, "spa": 30, "spd": 31, "spe": 31}

    def _state(self, *, p1_force_switch=False, p2_saved="NONE"):
        return SimpleNamespace(
            side_one=SimpleNamespace(
                active_index="0", force_switch=p1_force_switch,
                switch_out_move_second_saved_move="NONE",
            ),
            side_two=SimpleNamespace(
                active_index="0", force_switch=False,
                switch_out_move_second_saved_move=p2_saved,
            ),
        )

    def _teams(self):
        celebi = FixturePokemon(species="Celebi", moves=("Baton Pass",))
        starmie = FixturePokemon(species="Starmie", moves=("Surf",), ivs=self._GRASS_IVS)
        snorlax = FixturePokemon(species="Snorlax", moves=("Curse",))
        return [celebi, starmie], [snorlax]

    def test_moves_normalize_and_hidden_power_gets_typed(self) -> None:
        p1_team, p2_team = self._teams()
        p1_team = [FixturePokemon(species="Starmie", moves=("Hidden Power",), ivs=self._GRASS_IVS)]
        moves = engine_step_choices(
            self._state(), {"p1": "move hiddenpower", "p2": "move curse"},
            p1_team=p1_team, p2_team=p2_team,
        )
        self.assertEqual(moves, ("hiddenpowergrass70", "curse"))

    def test_forced_switch_is_bare_species_and_waiting_seat_resupplies_saved_move(self) -> None:
        p1_team, p2_team = self._teams()
        moves = engine_step_choices(
            self._state(p1_force_switch=True, p2_saved="CURSE"),
            {"p1": "switch 2"},  # p2 absent: waiting on the mid-turn boundary
            p1_team=p1_team, p2_team=p2_team,
        )
        self.assertEqual(moves, ("starmie", "curse"))

    def test_waiting_seat_without_saved_move_is_none(self) -> None:
        p1_team, p2_team = self._teams()
        moves = engine_step_choices(
            self._state(p1_force_switch=True), {"p1": "switch 2"},
            p1_team=p1_team, p2_team=p2_team,
        )
        self.assertEqual(moves, ("starmie", "none"))

    def test_voluntary_switch_keeps_switch_prefix(self) -> None:
        p1_team, p2_team = self._teams()
        moves = engine_step_choices(
            self._state(), {"p1": "switch 2", "p2": "move curse"},
            p1_team=p1_team, p2_team=p2_team,
        )
        self.assertEqual(moves, ("switch starmie", "curse"))


class MatchStepBranchTests(unittest.TestCase):
    _NO_BOOSTS = {"p1": {}, "p2": {}}

    def _branch(self, boost_deltas=None, **kwargs):
        return {
            "percentage": 50.0,
            "features": _features(**kwargs),
            "boost_deltas": self._NO_BOOSTS if boost_deltas is None else boost_deltas,
            "applied": object(),
            "raw": "",
        }

    def test_boost_deltas_disambiguate_observationally_identical_branches(self) -> None:
        # Sleep Talk calling Curse vs calling Rest (no-op): same TurnFeatures.
        curse = self._branch(boost_deltas={"p1": {}, "p2": {"attack": 1, "defense": 1, "speed": -1}})
        rest_noop = self._branch()
        observed = _features()
        row, variant, misses = match_step_branch(
            observed, observed, [curse, rest_noop],
            observed_boosts={"p1": {}, "p2": {"attack": 1, "defense": 1, "speed": -1}},
            p1_start_hp=200, p2_start_hp=300,
        )
        self.assertIs(row, curse)
        self.assertEqual(variant, "drift_adjusted")
        self.assertEqual(misses, [])

    def test_no_branch_with_observed_boosts_is_a_divergence(self) -> None:
        row, variant, misses = match_step_branch(
            _features(), _features(), [self._branch()],
            observed_boosts={"p1": {"attack": -1}, "p2": {}},
            p1_start_hp=200, p2_start_hp=300,
        )
        self.assertIsNone(row)
        self.assertIn("not in branch support", misses[0])

    def test_raw_observation_is_a_fallback_for_heal_to_full_sync_steps(self) -> None:
        # Recover clamps both sims to max HP: the raw observation matches the
        # branch exactly while the stale drift offset pushes it out of band.
        branch = self._branch(p2_hp=213)
        observed = _features(p2_hp=213)
        adjusted = _features(p2_hp=220)
        row, variant, _ = match_step_branch(
            observed, adjusted, [branch],
            observed_boosts=self._NO_BOOSTS, p1_start_hp=200, p2_start_hp=209,
        )
        self.assertIs(row, branch)
        self.assertEqual(variant, "raw")


class ExpectedTraceTests(unittest.TestCase):
    def test_trace_mismatch_reports_expected_and_actual(self) -> None:
        steps = [
            {"status": "matched", "telemetry": {"p1.reflect": 4}},
            {"status": "matched", "telemetry": {"p1.reflect": 4}},
        ]
        mismatches = check_expected_traces({"p1.reflect": (4, 3)}, steps)
        self.assertEqual(len(mismatches), 1)
        self.assertIn("expected (4, 3), engine (4, 4)", mismatches[0])
        self.assertEqual(check_expected_traces({"p1.reflect": (4, 4)}, steps), [])


@unittest.skipIf(poke_engine is None, "poke-engine wheel not installed")
class LiveMultiTurnSmokeTest(unittest.TestCase):
    """One short case end-to-end against the real sim + engine (skipped without both)."""

    def test_toxic_escalation_single_seed_matches(self) -> None:
        config = integration_config()
        if config is None:
            self.skipTest("Pokemon Showdown checkout or node runtime unavailable")
        import dataclasses

        from pokezero.dex import load_showdown_dex
        from pokezero.engine_fidelity_multiturn import curated_multiturn_cases, run_multiturn_case

        dex = load_showdown_dex(config.resolved_showdown_root())
        case = next(c for c in curated_multiturn_cases() if c.name == "toxic_escalation")
        case = dataclasses.replace(case, seeds=(22,))
        row = run_multiturn_case(case, dex=dex, module=poke_engine, config=config)
        self.assertEqual(row["status"], "ok", row)
        telemetry = [step["telemetry"]["p2.toxic_count"] for step in row["seeds"][0]["steps"]]
        self.assertEqual(telemetry, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
