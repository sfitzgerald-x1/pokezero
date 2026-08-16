"""Unit tests for the engine-MCTS POC policy (fake engine module; no native dep)."""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import pathlib
import random
import sys
import tempfile
import warnings
from types import SimpleNamespace
import unittest
from collections import Counter
import dataclasses
from dataclasses import replace
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pokezero.engine_search import (  # noqa: E402
    EngineMctsConfig,
    EngineMctsPolicy,
    EngineMctsStats,
    EngineSearchFallbackError,
    _ABORT_LOSSY_SUBCASES_ATTR,
    _FALLBACK_SAMPLE_KEY_CEILING,
    _OVERRIDE_DISAGREEMENT_ADDRESSES,
    _FALLBACK_SAMPLES_PER_CLASS,
    _REASON_DETAIL_LIMIT,
    _bounded_reason_detail,
    free_decision_features,
    _latch_encoder_tables_to_model_config,
    _locked_aggregate_choice,
    _world_visit_shares,
    native_search_args,
)


class _FakeObservation:
    def __init__(self, mask, candidates):
        self.legal_action_mask = mask
        self.metadata = {"action_candidates": candidates}


class _FakeContext:
    def __init__(self, observation, public_state=object(), player_id="p1"):
        self.observation = observation
        self.public_materialization_state = public_state
        self.player_id = player_id


def _candidates():
    return [
        {"action_index": 0, "kind": "move", "legal": True, "move_id": "earthquake"},
        {"action_index": 1, "kind": "move", "legal": True, "move_id": "hiddenpower"},
        {"action_index": 2, "kind": "move", "legal": False, "move_id": "protect"},
        {"action_index": 4, "kind": "switch", "legal": True, "pokemon": {"species": "Starmie"}},
        {"action_index": 5, "kind": "switch", "legal": False, "pokemon": {"species": "Snorlax"}},
    ]


def _policy():
    # module is never touched by the mapping/fallback tests
    return EngineMctsPolicy(dex=None, set_source=None, module=object(), config=EngineMctsConfig())


class ChoiceMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = _policy()
        mask = (True, True, False, False, True, False, False, False, False)
        self.context = _FakeContext(_FakeObservation(mask, _candidates()))

    def test_moves_switches_and_hidden_power_map(self) -> None:
        mapped = self.policy._map_choices(
            self.context,
            {"earthquake": 0.2, "switch starmie": 0.5, "hiddenpowergrass70": 0.1},
        )
        self.assertEqual(mapped, 4)  # highest-weight legal choice

    def test_hidden_power_engine_id_maps_to_plain_request_slot(self) -> None:
        mapped = self.policy._map_choices(self.context, {"hiddenpowergrass70": 1.0})
        self.assertEqual(mapped, 1)

    def test_illegal_candidates_never_selected(self) -> None:
        mapped = self.policy._map_choices(
            self.context, {"protect": 1.0, "switch snorlax": 0.9, "earthquake": 0.1}
        )
        self.assertEqual(mapped, 0)
        self.assertEqual(
            set(self.policy.stats.unmapped_choices), {"protect", "switch snorlax"}
        )

    def test_no_mappable_choice_returns_none(self) -> None:
        self.assertIsNone(self.policy._map_choices(self.context, {"surf": 1.0}))

    def test_cosmetic_forme_switch_maps_canonically(self) -> None:
        # The engine displays the collapsed base id ("switch unown") while the
        # request candidate carries the lettered forme ("Unown-C") — the
        # seed-7001 bench repro's mapping half.
        candidates = _candidates() + [
            {"action_index": 6, "kind": "switch", "legal": True, "pokemon": {"species": "Unown-C"}},
        ]
        mask = (True, True, False, False, True, False, True, False, False)
        context = _FakeContext(_FakeObservation(mask, candidates))
        mapped = self.policy._map_choices(context, {"switch unown": 1.0})
        self.assertEqual(mapped, 6)
        self.assertEqual(dict(self.policy.stats.unmapped_choices), {})


class OwnSideSelectionTests(unittest.TestCase):
    """The policy must read ITS OWN seat's visit distribution (p2 included)."""

    class _Entry:
        def __init__(self, move_choice, visits):
            self.move_choice = move_choice
            self.visits = visits

    class _Result:
        def __init__(self):
            self.total_visits = 100
            # side_one (p1) prefers earthquake; side_two (p2) prefers surf.
            self.side_one = [OwnSideSelectionTests._Entry("earthquake", 90)]
            self.side_two = [OwnSideSelectionTests._Entry("surf", 90)]

    def _run_seat(self, player_id):
        import unittest.mock as mock
        from pokezero.engine_search import EngineMctsConfig, EngineMctsPolicy
        from pokezero.engine_world import EngineWorld
        from pokezero.poke_engine_adapter import BattleSpec, SideSpec, PokemonSpec, MoveSpec

        module = mock.Mock()
        module.monte_carlo_tree_search.return_value = self._Result()
        policy = EngineMctsPolicy(
            dex=None, set_source=None, module=module,
            config=EngineMctsConfig(worlds=1, sample_retry_factor=1),
        )
        candidates = [
            {"action_index": 0, "kind": "move", "legal": True, "move_id": "earthquake"},
            {"action_index": 1, "kind": "move", "legal": True, "move_id": "surf"},
        ]
        mask = (True, True, False, False, False, False, False, False, False)
        context = _FakeContext(_FakeObservation(mask, candidates), player_id=player_id)
        world = EngineWorld(
            spec=None,
            slot_sides={"p1": "side_one", "p2": "side_two"},
            party_species={"p1": (), "p2": ()},
        )
        with mock.patch("pokezero.engine_search._gen3_randbat_belief_start_override_result",
                        return_value=(object(), None)), \
             mock.patch("pokezero.engine_search.world_battle_spec", return_value=world), \
             mock.patch("pokezero.engine_search.build_poke_engine_state", return_value=object()):
            decision = policy.select_action_with_context(context, rng=random.Random(0))
        return decision.action_index

    def test_p1_reads_side_one(self) -> None:
        self.assertEqual(self._run_seat("p1"), 0)  # earthquake

    def test_p2_reads_side_two(self) -> None:
        self.assertEqual(self._run_seat("p2"), 1)  # surf, NOT p1's earthquake


class AttractPatchFallbackTests(unittest.TestCase):
    """A missing local patch must degrade safely, never search a no-op state."""

    def test_missing_attract_patch_is_an_attributed_fallback(self) -> None:
        import unittest.mock as mock

        from pokezero.engine_world import EngineWorld
        from pokezero.poke_engine_adapter import PokeEngineAttractUnsupportedError

        module = mock.Mock()
        policy = EngineMctsPolicy(
            dex=None,
            set_source=None,
            module=module,
            config=EngineMctsConfig(worlds=1, sample_retry_factor=1),
        )
        mask = (True, False, False, False, False, False, False, False, False)
        context = _FakeContext(_FakeObservation(mask, _candidates()))
        world = EngineWorld(
            spec=None,
            slot_sides={"p1": "side_one", "p2": "side_two"},
            party_species={"p1": (), "p2": ()},
        )
        with mock.patch(
            "pokezero.engine_search._gen3_randbat_belief_start_override_result",
            return_value=(object(), None),
        ), mock.patch("pokezero.engine_search.world_battle_spec", return_value=world), mock.patch(
            "pokezero.engine_search.build_poke_engine_state",
            side_effect=PokeEngineAttractUnsupportedError("missing patch"),
        ):
            decision = policy.select_action_with_context(context, rng=random.Random(0))

        self.assertEqual(decision.metadata["engine_mcts"]["fallback"], "no_worlds_constructed")
        self.assertEqual(
            policy.stats.world_failure_reasons,
            Counter({"attract_patch_unavailable": 1}),
        )


class FallbackTests(unittest.TestCase):
    def test_missing_public_state_falls_back_uniform_legal(self) -> None:
        policy = _policy()
        mask = (False, True, False, False, False, False, False, False, False)
        context = _FakeContext(_FakeObservation(mask, _candidates()), public_state=None)
        decision = policy.select_action_with_context(context, rng=random.Random(1))
        self.assertEqual(decision.action_index, 1)
        self.assertEqual(policy.stats.fallback_decisions, 1)
        self.assertEqual(policy.stats.fallback_reasons, Counter({"no_public_state": 1}))
        self.assertEqual(decision.metadata["engine_mcts"]["fallback"], "no_public_state")

    def test_stats_report_shape(self) -> None:
        policy = _policy()
        payload = policy.stats.to_dict()
        self.assertEqual(payload["decisions"], 0)
        self.assertEqual(payload["fallback_rate"], 0.0)


class RechargeSignalTests(unittest.TestCase):
    """Unit tests for the risk-bearing recharge signal (review finding)."""

    class _Action:
        kind = "move"
        move_id = "hyperbeam"

    class _Round:
        def __init__(self, actions):
            self.actions = actions

    def _context(self, *, events, prev_action="hyperbeam", active="Slaking"):
        import unittest.mock as mock

        action = None
        if prev_action is not None:
            action = self._Action()
            action.move_id = prev_action
        rounds = {4: self._Round({"p2": action} if action else {})}
        context = type("Ctx", (), {
            "player_id": "p1",
            "decision_round_index": 5,
            "trajectory": object(),
            "observation": type("Obs", (), {
                "metadata": {
                    "belief_view": {"opponent_pokemon": [
                        {"species": active, "active": True},
                    ]},
                    "recent_public_events": events,
                },
            })(),
        })()
        return context, rounds

    def _slots(self, context, rounds):
        import unittest.mock as mock

        policy = _policy()
        with mock.patch(
            "pokezero.engine_search.public_action_rounds_from_trajectory_metadata",
            return_value=rounds,
        ):
            return policy._recharging_slots(context)

    def test_clean_hit_with_visible_anchor_locks(self) -> None:
        context, rounds = self._context(events=[
            "|move|p2a: Slaking|Hyper Beam|p1a: Blissey",
            "|-damage|p1a: Blissey|100/300",
        ])
        self.assertEqual(self._slots(context, rounds), ("p2",))

    def test_visible_miss_suppresses_lock(self) -> None:
        context, rounds = self._context(events=[
            "|move|p2a: Slaking|Hyper Beam|p1a: Blissey",
            "|-miss|p2a: Slaking|p1a: Blissey",
        ])
        self.assertEqual(self._slots(context, rounds), ())

    def test_scrolled_out_anchor_fails_open(self) -> None:
        # Round record says hyperbeam, but the move line is gone from the
        # window: cannot verify hit -> NO lock (the confirmed wrong-lock fix).
        context, rounds = self._context(events=[
            "|-weather|Sandstorm|[upkeep]",
            "|-damage|p2a: Slaking|300/400",
        ])
        self.assertEqual(self._slots(context, rounds), ())

    def test_species_continuity_guard(self) -> None:
        # The HB user fainted; a replacement is active -> no lock.
        context, rounds = self._context(active="Blissey", events=[
            "|move|p2a: Slaking|Hyper Beam|p1a: Blissey",
            "|-damage|p1a: Blissey|100/300",
        ])
        self.assertEqual(self._slots(context, rounds), ())

    def test_non_recharge_previous_action_no_lock(self) -> None:
        context, rounds = self._context(prev_action="bodyslam", events=[
            "|move|p2a: Slaking|Body Slam|p1a: Blissey",
        ])
        self.assertEqual(self._slots(context, rounds), ())

    def test_parser_tracker_is_preferred_over_the_reconstruction(self) -> None:
        # ONE PARSER TRUTH, TWO CONSUMERS (spec v4 pack A1): when the observation carries the
        # parser's ``must_recharge`` tracker, the world seeds from IT, not from the round-record
        # reconstruction. The tracker reads the ``-mustrecharge`` line the sim emits only on a
        # LANDED recharge move, so it is strictly stronger evidence.
        context, rounds = self._context(prev_action=None, events=[])
        context.observation.metadata["opponent_must_recharge"] = True
        self.assertEqual(self._slots(context, rounds), ("p2",))

    def test_parser_tracker_false_is_a_proof_not_an_absent_signal(self) -> None:
        # An explicit False must not be overridden by the weaker fallback: the tracker saw the
        # whole stream and there was no lock. (Here the reconstruction WOULD have locked.)
        context, rounds = self._context(events=[
            "|move|p2a: Slaking|Hyper Beam|p1a: Blissey",
            "|-damage|p1a: Blissey|100/300",
        ])
        context.observation.metadata["opponent_must_recharge"] = False
        self.assertEqual(self._slots(context, rounds), ())

    def test_missing_tracker_key_still_falls_back(self) -> None:
        # Cached rollouts and hand-built contexts predate the pack; they keep the old behaviour.
        context, rounds = self._context(events=[
            "|move|p2a: Slaking|Hyper Beam|p1a: Blissey",
            "|-damage|p1a: Blissey|100/300",
        ])
        self.assertNotIn("opponent_must_recharge", context.observation.metadata)
        self.assertEqual(self._slots(context, rounds), ("p2",))


class PublicEffectSignalTests(unittest.TestCase):
    """The item-mutation split: removals/consumptions clear, confirmed swaps
    substitute the current item, unconfirmed mutations fail closed."""

    def _signals(self, opponent_pokemon, self_pokemon=None):
        belief_view = {"opponent_pokemon": opponent_pokemon}
        if self_pokemon is not None:
            belief_view["self_pokemon"] = self_pokemon
        context = type("Ctx", (), {
            "player_id": "p1",
            "observation": type("Obs", (), {
                "metadata": {
                    "belief_view": belief_view,
                    "recent_public_events": [],
                },
            })(),
        })()
        blocked, encored, removed, overridden, _transformed = (
            _policy()._public_effect_signals(context)
        )
        return blocked, encored, removed, overridden

    def _transform_signal(self, opponent_pokemon, self_pokemon=None):
        """The transform half of the same signal bundle."""
        belief_view = {"opponent_pokemon": opponent_pokemon}
        if self_pokemon is not None:
            belief_view["self_pokemon"] = self_pokemon
        context = type("Ctx", (), {
            "player_id": "p1",
            "observation": type("Obs", (), {
                "metadata": {"belief_view": belief_view, "recent_public_events": []},
            })(),
        })()
        blocked, _encored, _removed, _overridden, transformed = (
            _policy()._public_effect_signals(context)
        )
        return blocked, transformed

    def test_knock_off_removal_is_not_blocked(self) -> None:
        blocked, _encored, removed, overridden = self._signals([
            {"species": "Blissey", "active": True, "item_mutated": True, "item_removed": True},
        ])
        self.assertEqual(blocked, {})
        self.assertEqual(removed, {"p2": ("blissey",)})
        self.assertEqual(overridden, {})

    def test_trick_swap_with_confirmed_current_item_overrides(self) -> None:
        # The post-swap CURRENT item is protocol-confirmed (the |-item| line):
        # worlds substitute it instead of failing closed.
        blocked, _encored, removed, overridden = self._signals([
            {"species": "Furret", "active": True, "item_mutated": True,
             "item_removed": False, "current_public_item": "Petaya Berry"},
        ])
        self.assertEqual(blocked, {})
        self.assertEqual(removed, {})
        self.assertEqual(overridden, {"p2": {"furret": "petayaberry"}})

    def test_mutation_without_confirmed_current_item_stays_fail_closed(self) -> None:
        # No protocol-confirmed current item (unaudited mutation source, or a
        # pre-override serialized payload): never guess — fail closed.
        blocked, _encored, removed, overridden = self._signals([
            {"species": "Blissey", "active": True, "item_mutated": True, "item_removed": False},
        ])
        self.assertEqual(blocked, {"p2": "item mutated on Blissey with unconfirmed current item"})
        self.assertEqual(removed, {})
        self.assertEqual(overridden, {})

    def test_consumed_item_routes_to_removed_without_mutation(self) -> None:
        # A publicly-eaten berry: item_removed without item_mutated (the eaten
        # item still pins the original assignment). The removal signal must
        # not require the mutation flag.
        blocked, _encored, removed, overridden = self._signals([
            {"species": "Furret", "active": True, "item_mutated": False, "item_removed": True},
        ])
        self.assertEqual(blocked, {})
        self.assertEqual(removed, {"p2": ("furret",)})
        self.assertEqual(overridden, {})

    def test_removal_beats_stale_current_item(self) -> None:
        # Trick gave the mon an item, then it was stripped/eaten: item_removed
        # wins over any leftover current_public_item value.
        blocked, _encored, removed, overridden = self._signals([
            {"species": "Furret", "active": True, "item_mutated": True,
             "item_removed": True, "current_public_item": "Petaya Berry"},
        ])
        self.assertEqual(blocked, {})
        self.assertEqual(removed, {"p2": ("furret",)})
        self.assertEqual(overridden, {})

    def test_self_side_item_signals_use_the_self_slot(self) -> None:
        # The self side's world team is the battle-START assignment too: after
        # the opponent Tricks OUR mon (or our berry is eaten) the same signals
        # apply, keyed to the self slot. The self seat never walled here — it
        # was silently stale.
        blocked, _encored, removed, overridden = self._signals(
            [
                {"species": "Alakazam", "active": True, "item_mutated": True,
                 "item_removed": False, "current_public_item": "Leftovers"},
            ],
            self_pokemon=[
                {"species": "Furret", "active": True, "item_mutated": True,
                 "item_removed": False, "current_public_item": "Petaya Berry"},
                {"species": "Snorlax", "active": False, "item_removed": True},
            ],
        )
        self.assertEqual(blocked, {})
        self.assertEqual(overridden, {
            "p2": {"alakazam": "leftovers"},
            "p1": {"furret": "petayaberry"},
        })
        self.assertEqual(removed, {"p1": ("snorlax",)})

    def test_benched_removal_still_collected(self) -> None:
        # The mutation lives on the mon, not the active slot: a knocked-off
        # mon on the bench still needs its sampled item cleared.
        blocked, _encored, removed, _overridden = self._signals([
            {"species": "Snorlax", "active": True},
            {"species": "Blissey", "active": False, "item_mutated": True, "item_removed": True},
        ])
        self.assertEqual(blocked, {})
        self.assertEqual(removed, {"p2": ("blissey",)})

    def test_multiple_removals_accumulate(self) -> None:
        blocked, _encored, removed, _overridden = self._signals([
            {"species": "Blissey", "active": False, "item_mutated": True, "item_removed": True},
            {"species": "Snorlax", "active": True, "item_mutated": True, "item_removed": True},
        ])
        self.assertEqual(blocked, {})
        self.assertEqual(removed, {"p2": ("blissey", "snorlax")})

    def test_removal_plus_unconfirmed_mutation_still_blocks_the_slot(self) -> None:
        # One mon knocked off (representable), another mutated with no
        # confirmed current item: the slot must still fail closed.
        blocked, _encored, removed, _overridden = self._signals([
            {"species": "Blissey", "active": False, "item_mutated": True, "item_removed": True},
            {"species": "Kecleon", "active": True, "item_mutated": True, "item_removed": False},
        ])
        self.assertEqual(blocked, {"p2": "item mutated on Kecleon with unconfirmed current item"})
        self.assertEqual(removed, {"p2": ("blissey",)})


class PhaseTelemetryTests(unittest.TestCase):
    """Crate-measured per-phase walls (encode/model/tree) reach the stats payload.

    The depth study attributes decision wall to phases and explicitly forbids
    deriving a missing phase by subtraction, so the transport must carry all
    three and must not fabricate a value when an older crate omits them.
    """

    def test_phase_walls_accumulate_from_reports(self) -> None:
        from pokezero.engine_search import EngineMctsStats

        stats = EngineMctsStats()
        for report in (
            {"encode_s": 0.25, "model_s": 1.5, "tree_s": 0.125},
            {"encode_s": 0.25, "model_s": 0.5, "tree_s": 0.125},
        ):
            stats.encode_wall_seconds += float(report.get("encode_s") or 0.0)
            stats.model_wall_seconds += float(report.get("model_s") or 0.0)
            stats.tree_wall_seconds += float(report.get("tree_s") or 0.0)
        payload = stats.to_dict()
        self.assertAlmostEqual(payload["encode_wall_seconds"], 0.5)
        self.assertAlmostEqual(payload["model_wall_seconds"], 2.0)
        self.assertAlmostEqual(payload["tree_wall_seconds"], 0.25)

    def test_missing_phase_fields_default_to_zero_not_inferred(self) -> None:
        from pokezero.engine_search import EngineMctsStats

        stats = EngineMctsStats()
        payload = stats.to_dict()
        for key in ("encode_wall_seconds", "model_wall_seconds", "tree_wall_seconds"):
            self.assertIn(key, payload)
            self.assertEqual(payload[key], 0.0)


class ModelConfigValidationTests(unittest.TestCase):
    def test_model_mode_requires_artifacts(self) -> None:
        with self.assertRaises(ValueError):
            EngineMctsConfig(leaf_eval="model")
        with self.assertRaises(ValueError):
            EngineMctsConfig(leaf_eval="model", model_path="x.pt")  # tables missing

    def test_unknown_leaf_eval_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EngineMctsConfig(leaf_eval="foulplay")

    def test_batch_must_not_exceed_sims(self) -> None:
        with self.assertRaises(ValueError):
            EngineMctsConfig(
                leaf_eval="model",
                model_path="x.pt",
                checkpoint_path="checkpoint.pt",
                tables_path="t.json",
                search_sims=8,
                search_batch=16,
            )

    def test_early_stop_is_model_only_and_floor_is_bounded(self) -> None:
        with self.assertRaisesRegex(ValueError, "only with leaf_eval='model'"):
            EngineMctsConfig(early_stop=True)
        with self.assertRaisesRegex(ValueError, "early_stop_min_sims"):
            EngineMctsConfig(
                leaf_eval="model",
                model_path="x.pt",
                checkpoint_path="checkpoint.pt",
                tables_path="t.json",
                search_sims=8,
                search_batch=8,
                early_stop=True,
                early_stop_min_sims=9,
            )

    def test_missing_model_artifact_fails_at_init(self) -> None:
        with self.assertRaises(ValueError):
            EngineMctsPolicy(
                dex=None,
                set_source=None,
                module=object(),
                config=EngineMctsConfig(
                    leaf_eval="model",
                    model_path="/nonexistent/model_ts.pt",
                    checkpoint_path="/nonexistent/checkpoint.pt",
                    tables_path="/nonexistent/tables.json",
                ),
            )


class ModelObservationContractTests(unittest.TestCase):
    @staticmethod
    def _model_config(*, budget: int = 32):
        return SimpleNamespace(
            observation_schema_version="pokezero.observation.v3",
            token_count=87,
            categorical_feature_count=51,
            numeric_feature_count=155,
            stats_block_enabled=True,
            exact_state_enabled=True,
            transition_token_budget=budget,
            transition_token_count=64,
            tier2_residuals=True,
            tier2_investment=False,
        )

    @staticmethod
    def _tables(*, schema: str = "pokezero.observation.v3") -> dict:
        return {
            "layout": {
                "schema_version": schema,
                "token_count": 87,
                "categorical_feature_count": 51,
                "numeric_feature_count": 155,
                "default_feature_masks": {
                    "stats_block": True,
                    "exact_state": True,
                    "transition_token_budget": 64,
                    "tier2_residuals": True,
                    "tier2_investment": False,
                },
            }
        }

    def test_tables_history_budget_is_latched_to_checkpoint(self) -> None:
        encoded = _latch_encoder_tables_to_model_config(
            json.dumps(self._tables()), self._model_config(budget=32)
        )

        masks = json.loads(encoded)["layout"]["default_feature_masks"]
        self.assertEqual(masks["transition_token_budget"], 32)

    def test_tables_schema_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "observation contract"):
            _latch_encoder_tables_to_model_config(
                json.dumps(self._tables(schema="pokezero.observation.v2.2")),
                self._model_config(),
            )

    def test_policy_init_latches_real_table_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "model_ts.pt"
            checkpoint_path = root / "checkpoint.pt"
            tables_path = root / "tables.json"
            model_path.touch()
            checkpoint_path.touch()
            tables_path.write_text(json.dumps(self._tables()), encoding="utf-8")
            # Stub the whole checkpoint read, not one field of it: policy init now reads the
            # model config AND `value_calibration_transform` from a single payload, so a stub
            # that supplied only the config left the calibration fence parsing an empty file.
            with patch(
                "pokezero.neural_policy.load_transformer_checkpoint_payload",
                return_value={"value_calibration_transform": None},
            ), patch(
                "pokezero.neural_policy.parse_transformer_model_config",
                return_value=self._model_config(budget=32),
            ):
                policy = EngineMctsPolicy(
                    dex=None,
                    set_source=None,
                    module=object(),
                    config=EngineMctsConfig(
                        leaf_eval="model",
                        model_path=str(model_path),
                        checkpoint_path=str(checkpoint_path),
                        tables_path=str(tables_path),
                    ),
                )

        masks = json.loads(policy._tables_json)["layout"]["default_feature_masks"]
        self.assertEqual(masks["transition_token_budget"], 32)

    def test_root_history_wider_than_checkpoint_fails_closed(self) -> None:
        policy = object.__new__(EngineMctsPolicy)
        policy._model_config = self._model_config(budget=32)
        prefix = (True,) * 23
        observation = SimpleNamespace(
            schema_version="pokezero.observation.v3",
            attention_mask=prefix + (True,) * 33 + (False,) * 31,
            categorical_ids=tuple((0,) * 51 for _ in range(87)),
            numeric_features=tuple((0.0,) * 155 for _ in range(87)),
        )

        with self.assertRaisesRegex(EngineSearchFallbackError, "exceeding checkpoint budget 32"):
            policy._validate_model_root_observation(observation)

    def test_root_history_at_checkpoint_budget_is_valid(self) -> None:
        policy = object.__new__(EngineMctsPolicy)
        policy._model_config = self._model_config(budget=32)
        prefix = (True,) * 23
        observation = SimpleNamespace(
            schema_version="pokezero.observation.v3",
            attention_mask=prefix + (True,) * 32 + (False,) * 32,
            categorical_ids=tuple((0,) * 51 for _ in range(87)),
            numeric_features=tuple((0.0,) * 155 for _ in range(87)),
        )

        policy._validate_model_root_observation(observation)


class EarlyStopAggregateTests(unittest.TestCase):
    @staticmethod
    def _report(
        *,
        requested: int,
        completed: int,
        visits: tuple[tuple[str, int], ...],
    ) -> dict:
        return {
            "requested_iterations": requested,
            "iterations": completed,
            "side_one": [
                {"move": move, "visits": count, "q": 0.5} for move, count in visits
            ],
        }

    def test_locked_choice_requires_strict_full_budget_separation(self) -> None:
        locked = self._report(
            requested=100,
            completed=60,
            visits=(("switch persian", 56), ("thunderwave", 4)),
        )
        tied_upper_bound = self._report(
            requested=100,
            completed=60,
            visits=(("switch persian", 50), ("thunderwave", 10)),
        )

        self.assertEqual(
            _locked_aggregate_choice([("side_one", locked)]),
            "switch persian",
        )
        self.assertIsNone(
            _locked_aggregate_choice([("side_one", tied_upper_bound)])
        )

    def test_multi_world_bound_includes_every_world(self) -> None:
        stopped = self._report(
            requested=100,
            completed=60,
            visits=(("switch persian", 56), ("thunderwave", 4)),
        )
        full = self._report(
            requested=100,
            completed=100,
            visits=(("switch persian", 80), ("thunderwave", 20)),
        )
        opposing = self._report(
            requested=100,
            completed=60,
            visits=(("switch persian", 4), ("thunderwave", 56)),
        )

        self.assertEqual(
            _locked_aggregate_choice(
                [("side_one", stopped), ("side_one", full)]
            ),
            "switch persian",
        )
        self.assertIsNone(
            _locked_aggregate_choice(
                [("side_one", stopped), ("side_one", opposing)]
            )
        )

    def test_malformed_visit_conservation_cannot_lock(self) -> None:
        malformed = self._report(
            requested=100,
            completed=60,
            visits=(("switch persian", 55), ("thunderwave", 4)),
        )
        self.assertIsNone(_locked_aggregate_choice([("side_one", malformed)]))

    def test_side_two_reports_use_the_requested_side(self) -> None:
        report = self._report(
            requested=100,
            completed=60,
            visits=(("switch persian", 56), ("thunderwave", 4)),
        )
        report["side_two"] = report.pop("side_one")
        self.assertEqual(
            _locked_aggregate_choice([("side_two", report)]),
            "switch persian",
        )


class EarlyStopPolicyIntegrationTests(unittest.TestCase):
    class _Native:
        def __init__(self, responses):
            self.responses = list(responses)
            self.calls = []

        def search_batched_multi_encoded(self, *args):
            self.calls.append(args)
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return json.dumps(response)

    @staticmethod
    def _report(
        alpha: int,
        beta: int,
        *,
        requested: int = 100,
        stopped: bool,
        max_depth_reached: int | None = None,
    ) -> dict:
        completed = alpha + beta
        report = {
            "iterations": completed,
            "requested_iterations": requested,
            "remaining_iterations": requested - completed,
            "early_stopped": stopped,
            "model_evals": completed,
            "lossy_renders": 0,
            "attribution_unsafe_renders": 0,
            "prior_fallbacks": 0,
            "side_one": [
                {"move": "alpha", "visits": alpha, "q": 0.5},
                {"move": "beta", "visits": beta, "q": 0.5},
            ],
        }
        if max_depth_reached is not None:
            # Omitted by default so every pre-existing caller keeps the shape it
            # had; the ladder's saturation test skips reports without it.
            report["max_depth_reached"] = max_depth_reached
        return report

    @staticmethod
    def _context():
        observation = _FakeObservation(
            (True, True, False, False, False, False, False, False, False),
            [
                {"action_index": 0, "kind": "move", "legal": True, "move_id": "alpha"},
                {"action_index": 1, "kind": "move", "legal": True, "move_id": "beta"},
            ],
        )
        return SimpleNamespace(
            observation=observation,
            public_materialization_state=SimpleNamespace(
                replay=SimpleNamespace(turn_number=1)
            ),
            player_id="p1",
            battle_id="early-stop-test",
            decision_round_index=0,
        )

    @staticmethod
    def _world(label: str):
        return (
            SimpleNamespace(
                party_species={"p1": ("rattata",), "p2": ("chansey",)},
                slot_sides={"p1": "side_one"},
            ),
            SimpleNamespace(to_string=lambda: label),
        )

    def _policy(self, *, early_stop: bool, strict: bool = False):
        policy = object.__new__(EngineMctsPolicy)
        policy.policy_id = "early-stop-test"
        policy._config = EngineMctsConfig(
            worlds=2,
            leaf_eval="model",
            model_path="model.pt",
            checkpoint_path="checkpoint.pt",
            tables_path="tables.json",
            search_sims=100,
            search_batch=10,
            early_stop=early_stop,
            early_stop_min_sims=20,
            strict_fallbacks=strict,
        )
        policy._tables_json = "{}"
        policy.stats = EngineMctsStats()
        policy._world_failures_before = {}
        return policy

    def _run(self, policy, native, worlds):
        fake_module = SimpleNamespace(
            FoldState=SimpleNamespace(from_payload=lambda _payload: object())
        )
        live_fold = SimpleNamespace(to_payload=lambda: {})
        with (
            patch.dict(sys.modules, {"pokezero_search": fake_module}),
            patch.object(EngineMctsPolicy, "_native", return_value=native),
            patch.object(
                EngineMctsPolicy, "_validate_model_root_observation", return_value=None
            ),
            patch.object(EngineMctsPolicy, "_root_inputs_json", return_value="{}"),
        ):
            return policy._search_model(
                self._context(), worlds, live_fold, random.Random(7)
            )

    def test_locked_multi_world_stop_is_accepted(self) -> None:
        native = self._Native(
            [
                self._report(56, 4, stopped=True),
                self._report(56, 4, stopped=True),
            ]
        )
        policy = self._policy(early_stop=True)

        decision = self._run(
            policy,
            native,
            [self._world("world-a"), self._world("world-b")],
        )

        self.assertEqual(decision.action_index, 0)
        self.assertTrue(decision.metadata["engine_mcts"]["early_stop"]["aggregate_locked"])
        self.assertEqual(decision.metadata["engine_mcts"]["early_stop"]["simulations_saved"], 80)
        self.assertEqual(policy.stats.early_stop_accepted_decisions, 1)
        self.assertTrue(all(len(call) == 14 for call in native.calls))

    def test_ambiguous_multi_world_stop_replays_full_budget(self) -> None:
        native = self._Native(
            [
                self._report(56, 4, stopped=True),
                self._report(4, 56, stopped=True),
                self._report(60, 40, stopped=False),
                self._report(55, 45, stopped=False),
            ]
        )
        policy = self._policy(early_stop=True)

        decision = self._run(
            policy,
            native,
            [self._world("world-a"), self._world("world-b")],
        )

        self.assertEqual(decision.action_index, 0)
        stop = decision.metadata["engine_mcts"]["early_stop"]
        self.assertFalse(stop["aggregate_locked"])
        self.assertEqual(stop["full_budget_replays"], 2)
        self.assertEqual(policy.stats.total_iterations, 320)
        self.assertEqual(policy.stats.early_stop_full_budget_replays, 2)

    def test_failed_required_replay_fails_the_decision_closed(self) -> None:
        native = self._Native(
            [
                self._report(56, 4, stopped=True),
                self._report(4, 56, stopped=True),
                RuntimeError("replay failed"),
            ]
        )
        policy = self._policy(early_stop=True, strict=True)

        with self.assertRaisesRegex(
            EngineSearchFallbackError, "reason=early_stop_replay_failed"
        ):
            self._run(
                policy,
                native,
                [self._world("world-a"), self._world("world-b")],
            )

    def test_unmappable_locked_choice_replays_instead_of_falling_back(self) -> None:
        native = self._Native(
            [
                self._report(56, 4, stopped=True),
                self._report(40, 60, stopped=False),
            ]
        )
        policy = self._policy(early_stop=True)
        context = self._context()
        context.observation.metadata["action_candidates"][0]["legal"] = False
        context.observation.legal_action_mask = (
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        )
        fake_module = SimpleNamespace(
            FoldState=SimpleNamespace(from_payload=lambda _payload: object())
        )
        with (
            patch.dict(sys.modules, {"pokezero_search": fake_module}),
            patch.object(EngineMctsPolicy, "_native", return_value=native),
            patch.object(
                EngineMctsPolicy, "_validate_model_root_observation", return_value=None
            ),
            patch.object(EngineMctsPolicy, "_root_inputs_json", return_value="{}"),
        ):
            decision = policy._search_model(
                context,
                [self._world("world-a")],
                SimpleNamespace(to_payload=lambda: {}),
                random.Random(7),
            )

        self.assertEqual(decision.action_index, 1)
        self.assertEqual(
            decision.metadata["engine_mcts"]["early_stop"]["full_budget_replays"],
            1,
        )

    def test_disabled_feature_preserves_old_native_call_shape(self) -> None:
        report = self._report(70, 30, stopped=False)
        report.pop("requested_iterations")
        report.pop("remaining_iterations")
        report.pop("early_stopped")
        native = self._Native([report])
        policy = self._policy(early_stop=False)

        decision = self._run(policy, native, [self._world("world-a")])

        self.assertEqual(decision.action_index, 0)
        self.assertEqual(len(native.calls[0]), 12)

    def test_attribution_unsafe_native_branch_fails_the_world_closed(self) -> None:
        native = self._Native(
            [ValueError("attribution-unsafe renderer branch rejected before tree/model fold: sleeptalk_called_unidentified")]
        )
        policy = self._policy(early_stop=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            decision = self._run(policy, native, [self._world("unsafe-world")])
        self.assertEqual(decision.metadata["engine_mcts"]["fallback"], "crate_search_failed")
        self.assertEqual(policy.stats.worlds_searched, 0)
        self.assertIn(
            "crate_search: attribution-unsafe renderer branch rejected before tree/model fold: sleeptalk_called_unidentified",
            policy.stats.world_failure_reasons,
        )
        self.assertEqual(policy.stats.attribution_unsafe_renders, 1)

    # --- diagnostics from ABORTED worlds ------------------------------------------
    #
    # Every abort used to discard every diagnostic the world had accumulated:
    # `model.rs` returns Err before the report string exists, and this seam salvaged
    # exactly one number (`attribution_unsafe_renders`). So `lossy_subcase_renders`
    # described only the clean-completion subset -- the subset that does not need
    # diagnosing. (What SHARE of worlds abort is not measured; an earlier revision of
    # this comment said "~92% of the fallback residue" and that figure was withdrawn as
    # unsourceable. See the `abort_telemetry` module header.) #1158 paid
    # for it: its Protect-marker counter reads zero both when the fix never fires and
    # when the fix fires but the world dies at its NEXT unsafe branch.

    @staticmethod
    def _aborting_error(message: str, payload) -> Exception:
        error = ValueError(message)
        # The attribute the crate attaches (abort_telemetry.rs, ABORT_PAYLOAD_ATTR).
        # Set through the module constant, so a rename cannot leave this fixture
        # testing a name nothing produces.
        setattr(error, _ABORT_LOSSY_SUBCASES_ATTR, payload)
        return error

    def _abort(self, policy, error) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._run(policy, self._Native([error]), [self._world("aborting-world")])

    def test_an_aborted_world_still_reports_the_subcases_it_observed(self) -> None:
        """THE POINT. Without this the change is unpinned.

        A world renders the Protect marker, then dies at a LATER unsafe branch. That is
        the case #1158's counter was built to distinguish from "the fix never fired", and
        the case that read zero.
        """
        policy = self._policy(early_stop=False)
        message = (
            "attribution-unsafe renderer branch rejected before tree/model fold: "
            "sleeptalk_called_unidentified"
        )
        self._abort(
            policy,
            self._aborting_error(
                message,
                {
                    "sleeptalk_called_unidentified:protect_marker_rendered": 2,
                    "attract_immobilization_source_unknown": 1,
                },
            ),
        )

        self.assertEqual(
            policy.stats.lossy_subcase_renders[
                "sleeptalk_called_unidentified:protect_marker_rendered"
            ],
            2,
        )
        self.assertEqual(
            policy.stats.lossy_subcase_renders["attract_immobilization_source_unknown"], 1
        )
        # Through to_dict(), because the in-memory Counter is not what the shard report
        # carries and a counter that stops at the object is still invisible.
        emitted = policy.stats.to_dict()["lossy_subcase_renders"]
        self.assertEqual(
            emitted["sleeptalk_called_unidentified:protect_marker_rendered"], 2
        )
        # The reason key must be BYTE-IDENTICAL: it is a measurement contract compared
        # across eras, so the counts may not ride inside the message.
        self.assertEqual(
            list(policy.stats.world_failure_reasons), [f"crate_search: {message}"]
        )
        # The old salvaged counter still fires -- this adds a channel, it does not
        # replace one.
        self.assertEqual(policy.stats.attribution_unsafe_renders, 1)

    def test_an_aborted_world_reports_each_observation_exactly_once(self) -> None:
        """A world that both ACCUMULATES and ABORTS must not be counted twice.

        The report field and the exception payload are exclusive outcomes of one native
        invocation. Absorbing both for one world -- or absorbing the payload once per
        handler in a chain -- would inflate the class silently, which is the same
        category of defect as losing it.
        """
        policy = self._policy(early_stop=False)
        self._abort(
            policy,
            self._aborting_error(
                "attribution-unsafe renderer branch rejected before tree/model fold: x",
                {"sleeptalk_called_unidentified:ambiguous": 5},
            ),
        )

        self.assertEqual(
            policy.stats.lossy_subcase_renders["sleeptalk_called_unidentified:ambiguous"],
            5,
        )
        # Nothing else was invented, and 5 was not doubled to 10.
        self.assertEqual(sum(policy.stats.lossy_subcase_renders.values()), 5)

        # Now a CLEAN world reporting the SAME key in the same decision: the two
        # channels ADD, one per invocation, exactly like lossy_renders.
        report = self._report(70, 30, stopped=False)
        report["lossy_subcases"] = {"sleeptalk_called_unidentified:ambiguous": 3}
        aborted = self._aborting_error(
            "attribution-unsafe renderer branch rejected before tree/model fold: x",
            {"sleeptalk_called_unidentified:ambiguous": 5},
        )
        both = self._policy(early_stop=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._run(
                both,
                self._Native([report, aborted]),
                [self._world("clean"), self._world("aborted")],
            )
        self.assertEqual(
            both.stats.lossy_subcase_renders["sleeptalk_called_unidentified:ambiguous"], 8
        )
        self.assertEqual(sum(both.stats.lossy_subcase_renders.values()), 8)

    def test_a_failure_carrying_no_payload_is_a_silent_no_op(self) -> None:
        """Older wheels attach nothing, and most failures here are not native at all.

        A crash inside the handler whose entire job is to keep the other worlds alive
        would be a strictly worse defect than the one being fixed, so the shapes that
        cannot be counted must be ignored rather than raised.

        `uncountable values` is the shape an `isinstance(payload, Mapping)` check alone
        does NOT cover: the mapping is well-formed and the VALUES are not counts. Review
        measured all three raising out of `_absorb_lossy_subcases`'s `int(count)`, out of
        the `except Exception` in `run_world`, and -- there being no outer try around
        `_search_model` -- out of `decide()` entirely. `True` is in the list because
        `bool` is an `int` subclass and would otherwise be silently counted as 1.
        """
        for label, error in (
            ("no attribute", ValueError("stale wheel: no payload")),
            ("not a mapping", self._aborting_error("junk payload", "not a mapping")),
            ("payload is None", self._aborting_error("no counts", None)),
            ("non-native failure", RuntimeError("something else entirely")),
            (
                "uncountable values",
                self._aborting_error(
                    "attribution-unsafe renderer branch rejected before tree/model fold: x",
                    {"a": "many", "b": None, "c": {"nested": 1}, "d": True},
                ),
            ),
        ):
            with self.subTest(shape=label):
                policy = self._policy(early_stop=False)
                self._abort(policy, error)
                self.assertEqual(policy.stats.lossy_subcase_renders, Counter())
                # The world was still counted as a failure -- silence about the
                # sub-cases is not silence about the abort.
                self.assertEqual(sum(policy.stats.world_failure_reasons.values()), 1)

    def test_countable_entries_survive_alongside_uncountable_ones(self) -> None:
        """Dropping the junk must not drop the data next to it.

        The obvious over-correction for the shape above is to reject the whole payload
        when any value is uncountable, which would let one bad entry erase a world's
        entire observation -- the same class of silent loss this change exists to fix,
        at a smaller scale.
        """
        policy = self._policy(early_stop=False)
        self._abort(
            policy,
            self._aborting_error(
                "attribution-unsafe renderer branch rejected before tree/model fold: x",
                {
                    "sleeptalk_called_unidentified:protect_marker_rendered": 4,
                    "junk": "many",
                },
            ),
        )
        self.assertEqual(
            policy.stats.lossy_subcase_renders[
                "sleeptalk_called_unidentified:protect_marker_rendered"
            ],
            4,
        )
        self.assertEqual(sum(policy.stats.lossy_subcase_renders.values()), 4)

    def test_the_failure_reason_is_recorded_before_the_diagnostics_are_absorbed(
        self,
    ) -> None:
        """ORDERING, and the reason it is an ordering and not a preference.

        `_absorb_aborted_lossy_subcases` is written to swallow everything a malformed
        payload can throw, and the tests above exercise the shapes that were measured to
        throw. But "written to" is not "proven to": the payload comes off an arbitrary
        caught exception, so the set of shapes is open, and the method's own docstring
        records that an escape propagates straight out of `decide()`.

        If the absorb runs FIRST, an escape takes the world's `world_failure_reasons`
        entry with it. Those keys are a measurement contract compared across eras, so the
        fallback would be UNDERCOUNTED -- a wrong number -- rather than merely
        undiagnosed. Recording the reason first makes the worst case a missing diagnostic.

        Driven by an absorb that is FORCED to raise, because a test over a payload shape
        that happens to be handled would pass under either ordering and pin nothing.
        """

        policy = self._policy(early_stop=False)
        message = "attribution-unsafe renderer branch rejected before tree/model fold: x"

        def _explode(_error: BaseException) -> None:
            raise RuntimeError("absorb blew up on an unforeseen payload shape")

        policy._absorb_aborted_lossy_subcases = _explode  # type: ignore[method-assign]

        with self.assertRaises(RuntimeError):
            self._abort(policy, self._aborting_error(message, {"whatever": 1}))

        # The reason survived the escape. Under the other ordering this is empty.
        self.assertEqual(
            list(policy.stats.world_failure_reasons), [f"crate_search: {message}"]
        )
        self.assertEqual(policy.stats.attribution_unsafe_renders, 1)

    # --- telemetry counting (model path) ------------------------------------------
    #
    # The model path used to count every world TWICE — once above the aggregation loop
    # and once per record inside it — and to add the search interval to
    # ``search_wall_seconds`` twice, both times measured from the same
    # ``search_started``. So model-mode ``worlds_searched`` and the derived
    # ``search_wall_per_searched_decision`` were ~2x inflated. Scores were never
    # affected: none of it feeds ``aggregated``. The hp_fraction paths always counted
    # once and are the reference shape.

    def _reports(self, count: int) -> list[dict]:
        return [self._report(70, 30, stopped=False) for _ in range(count)]

    def test_each_world_is_counted_exactly_once(self) -> None:
        worlds = [self._world(f"world-{index}") for index in range(3)]
        policy = self._policy(early_stop=False)

        decision = self._run(policy, self._Native(self._reports(len(worlds))), worlds)

        # Three worlds in, three counted -- not six, which is what the double
        # increment produced and what the metadata handed to every consumer.
        self.assertEqual(policy.stats.worlds_searched, len(worlds))
        self.assertEqual(
            decision.metadata["engine_mcts"]["worlds_searched"], len(worlds)
        )

    def test_renderer_counters_are_preserved_from_native_reports(self) -> None:
        report = self._report(70, 30, stopped=False)
        report["lossy_renders"] = 3
        report["attribution_unsafe_renders"] = 0
        policy = self._policy(early_stop=False)
        self._run(policy, self._Native([report]), [self._world("world-a")])
        self.assertEqual(policy.stats.lossy_renders, 3)
        self.assertEqual(policy.stats.attribution_unsafe_renders, 0)

    def test_world_count_is_linear_in_the_number_of_worlds(self) -> None:
        # Guards the shape as well as one value: a per-record increment would keep the
        # ratio at 2 for every N, so checking a single N could be read as an off-by-one.
        for count in (1, 2, 4):
            with self.subTest(worlds=count):
                policy = self._policy(early_stop=False)
                worlds = [self._world(f"world-{index}") for index in range(count)]
                self._run(policy, self._Native(self._reports(count)), worlds)
                self.assertEqual(policy.stats.worlds_searched, count)

    def test_search_wall_is_accumulated_once_per_decision(self) -> None:
        # A stepped clock: the first reading is ``search_started``, every later reading
        # is a fixed interval after it. One accumulation therefore records exactly
        # STEP; the old double accumulation recorded 2 * STEP. Robust to how many
        # times perf_counter is called, only to how many times the wall is added to.
        step = 7.5
        readings = iter([100.0])

        def clock() -> float:
            return next(readings, 100.0 + step)

        policy = self._policy(early_stop=False)
        worlds = [self._world(f"world-{index}") for index in range(3)]
        with patch("pokezero.engine_search.time.perf_counter", clock):
            self._run(policy, self._Native(self._reports(len(worlds))), worlds)

        self.assertAlmostEqual(policy.stats.search_wall_seconds, step)

    def test_reported_seconds_per_decision_is_not_inflated(self) -> None:
        # The number the depth study actually reads
        # (``search_wall_per_searched_decision``), end to end through the stats payload.
        step = 4.0
        readings = iter([0.0])

        def clock() -> float:
            return next(readings, step)

        policy = self._policy(early_stop=False)
        with patch("pokezero.engine_search.time.perf_counter", clock):
            self._run(policy, self._Native(self._reports(2)), [self._world("a"), self._world("b")])

        payload = policy.stats.to_dict()
        self.assertEqual(payload["worlds_searched"], 2)
        self.assertAlmostEqual(payload["search_wall_per_searched_decision"], step)


class _FakeEvent:
    def __init__(self, raw_line):
        self.raw_line = raw_line


class _FakeReplay:
    def __init__(self, lines):
        self.public_events = tuple(_FakeEvent(line) for line in lines)
        self.turn_number = 1


class _FakePublicState:
    def __init__(self, lines):
        self.replay = _FakeReplay(lines)


class LiveFoldAdvanceTests(unittest.TestCase):
    """The incremental per-battle root fold (ledger: live root-fold export)."""

    LEAD = [
        "|switch|p1a: Rattata|Rattata, L88|100/100",
        "|switch|p2a: Chansey|Chansey, L80|100/100",
        "|turn|1",
    ]
    ROUND2 = [
        "|move|p1a: Rattata|Tackle|p2a: Chansey",
        "|-damage|p2a: Chansey|468/641",
        "|upkeep",
        "|turn|2",
    ]

    def _context(self, lines, battle_id="fold-test", round_index=0):
        context = _FakeContext(
            _FakeObservation((True,) * 9, _candidates()),
            public_state=_FakePublicState(lines),
        )
        context.battle_id = battle_id
        context.decision_round_index = round_index
        return context

    def test_incremental_advance_consumes_only_new_lines(self) -> None:
        policy = _policy()
        fold = policy._advance_live_fold(self._context(self.LEAD))
        self.assertIsNotNone(fold)
        self.assertEqual(policy.stats.fold_advanced_lines, len(self.LEAD))
        lead_total = fold.products().transition_token_total
        # Second decision: only the four new lines fold (not a whole-log refold).
        fold2 = policy._advance_live_fold(
            self._context(self.LEAD + self.ROUND2, round_index=1)
        )
        self.assertIs(fold2, fold)  # same per-battle state, advanced in place
        self.assertEqual(
            policy.stats.fold_advanced_lines, len(self.LEAD) + len(self.ROUND2)
        )
        # Exactly one new token: the tackle (lead lines fold only once).
        self.assertEqual(fold2.products().transition_token_total, lead_total + 1)

    def test_rewound_stream_breaks_the_fold_loudly(self) -> None:
        policy = _policy()
        self.assertIsNotNone(policy._advance_live_fold(self._context(self.LEAD)))
        import warnings as _warnings

        from pokezero.engine_search import EngineSearchFoldMismatchWarning

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            result = policy._advance_live_fold(
                self._context(self.LEAD[:1], round_index=1)
            )
        self.assertIsNone(result)
        self.assertTrue(
            any(issubclass(w.category, EngineSearchFoldMismatchWarning) for w in caught)
        )
        # Broken stays broken for the battle (no silent resync).
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            self.assertIsNone(
                policy._advance_live_fold(
                    self._context(self.LEAD + self.ROUND2, round_index=2)
                )
            )

    def test_new_battle_resets_fold_state(self) -> None:
        policy = _policy()
        policy._advance_live_fold(self._context(self.LEAD, battle_id="battle-a"))
        fold_b = policy._advance_live_fold(
            self._context(self.LEAD, battle_id="battle-b")
        )
        self.assertIsNotNone(fold_b)
        self.assertEqual(
            [key[0] for key in policy._live_folds], ["battle-b"]
        )  # battle-a state dropped

    def test_perspective_follows_the_acting_seat(self) -> None:
        policy = _policy()
        context = self._context(self.LEAD + self.ROUND2)
        context.player_id = "p2"
        fold = policy._advance_live_fold(context)
        self.assertEqual(fold.perspective_slot, "p2")


class _FakeAnnotationToken:
    def __init__(self, residual=None, residual_valid=False, cb_bit=False, investment=0.0):
        self.residual = residual
        self.residual_valid = residual_valid
        self.cb_bit = cb_bit
        self.investment = investment


class _FakeAnnotationState:
    def __init__(self, tokens):
        self.transition_tokens = tuple(tokens)


class _FakeAnnotationSource:
    """EnvTier2AnnotationSource-shaped stub over a fixed annotated stream."""

    def __init__(self, tokens, active=True):
        self._state = _FakeAnnotationState(tokens)
        self._active = active
        self.overlay_calls = 0

    def active(self):
        return self._active

    def boundary_state(self, player_id):
        return self._state

    def overlay_for(self, player_id):
        self.overlay_calls += 1
        return {
            index: (t.residual, t.residual_valid, t.cb_bit, t.investment)
            for index, t in enumerate(self._state.transition_tokens)
            if t.residual is not None or t.residual_valid or t.cb_bit or t.investment
        }


class Tier2OverlayTests(unittest.TestCase):
    """The live fold must carry the env trackers' Tier-2 conclusions
    (annotated products at search leaves == what the env encodes)."""

    LEAD = LiveFoldAdvanceTests.LEAD
    ROUND2 = LiveFoldAdvanceTests.ROUND2
    _context = LiveFoldAdvanceTests._context

    def _annotated_policy(self, tokens, active=True):
        source = _FakeAnnotationSource(tokens, active=active)
        policy = EngineMctsPolicy(
            dex=None, set_source=None, module=object(),
            config=EngineMctsConfig(), annotation_source=source,
        )
        return policy, source

    def test_overlay_applies_to_the_live_fold(self) -> None:
        # Boundary 1: two unannotated lead tokens. Boundary 2: the tackle
        # token (index 2), which the env tracker assessed with a residual —
        # the per-boundary arrival shape of real tracker conclusions.
        policy, source = self._annotated_policy(
            [_FakeAnnotationToken(), _FakeAnnotationToken()]
        )
        policy._advance_live_fold(self._context(self.LEAD))
        self.assertEqual(policy.stats.fold_annotations_applied, 0)
        source._state = _FakeAnnotationState(
            [
                _FakeAnnotationToken(),
                _FakeAnnotationToken(),
                _FakeAnnotationToken(residual=0.25, residual_valid=True),
            ]
        )
        fold = policy._advance_live_fold(
            self._context(self.LEAD + self.ROUND2, round_index=1)
        )
        self.assertIsNotNone(fold)
        self.assertEqual(policy.stats.fold_annotations_applied, 1)
        annotated = fold.products().transition_tokens[2]
        self.assertEqual(annotated.residual, 0.25)
        self.assertTrue(annotated.residual_valid)
        # Re-application at the next boundary is an idempotent equality check.
        policy._advance_live_fold(
            self._context(self.LEAD + self.ROUND2, round_index=2)
        )
        self.assertEqual(policy.stats.fold_annotations_applied, 1)

    def test_inactive_source_applies_nothing(self) -> None:
        tokens = [_FakeAnnotationToken(residual=0.5, residual_valid=True)]
        policy, source = self._annotated_policy(tokens, active=False)
        fold = policy._advance_live_fold(self._context(self.LEAD))
        self.assertIsNotNone(fold)
        self.assertEqual(source.overlay_calls, 0)
        self.assertEqual(policy.stats.fold_annotations_applied, 0)

    def test_changed_conclusion_breaks_the_fold_loudly(self) -> None:
        # Tracker conclusions are per-index immutable; a changed value is a
        # regression and must fail closed, not silently re-annotate.
        tokens = [
            _FakeAnnotationToken(residual=0.25, residual_valid=True),
            _FakeAnnotationToken(),
        ]
        policy, source = self._annotated_policy(tokens)
        self.assertIsNotNone(policy._advance_live_fold(self._context(self.LEAD)))
        self.assertEqual(policy.stats.fold_annotations_applied, 1)
        source._state.transition_tokens[0].residual = 0.75  # mutate in place
        import warnings as _warnings

        from pokezero.engine_search import EngineSearchFoldMismatchWarning

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            result = policy._advance_live_fold(
                self._context(self.LEAD + self.ROUND2, round_index=1)
            )
        self.assertIsNone(result)
        self.assertTrue(
            any(issubclass(w.category, EngineSearchFoldMismatchWarning) for w in caught)
        )

    def test_stale_annotation_breaks_the_fold_loudly(self) -> None:
        policy, source = self._annotated_policy([])
        # A delayed tracker conclusion outside the fold's identifiable tail
        # must fail closed before it can be applied to the wrong token.
        source.overlay_for = lambda _player_id: {-1: (0.25, True, False, 0.0)}
        import warnings as _warnings

        from pokezero.engine_search import EngineSearchFoldMismatchWarning

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            result = policy._advance_live_fold(self._context(self.LEAD))
        self.assertIsNone(result)
        self.assertTrue(
            any(issubclass(w.category, EngineSearchFoldMismatchWarning) for w in caught)
        )

    def test_cross_check_binds_against_env_surfaces(self) -> None:
        # With an active source, the cross-check reference is the env's own
        # encoder state (production binding): a reference whose surfaces ARE
        # the fold's products passes; a corrupted stream fails loudly.
        policy, source = self._annotated_policy([])
        context = self._context(self.LEAD)
        fold = policy._advance_live_fold(context)
        products = fold.products()

        class _Perspective:
            showdown_slot = "p1"
            opponent_showdown_slot = "p2"

        class _BoundState:
            transition_tokens = tuple(products.transition_tokens)
            turn_merged_tokens = tuple(products.turn_merged_tokens)
            tendency_stats = products.tendency_stats
            perspective = _Perspective()

        source._state = _BoundState()
        import warnings as _warnings

        with _warnings.catch_warnings(record=True):
            _warnings.simplefilter("always")
            policy._fold_cross_check(
                context, fold, context.public_materialization_state.replay
            )
        self.assertEqual(policy.stats.fold_cross_check_failures, 0)
        # Corrupt the reference stream: the mismatch must be loud.
        source._state.transition_tokens = tuple(products.transition_tokens[:-1])
        from pokezero.engine_search import EngineSearchFoldMismatchWarning

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            policy._fold_cross_check(
                context, fold, context.public_materialization_state.replay
            )
        self.assertEqual(policy.stats.fold_cross_check_failures, 1)
        self.assertTrue(
            any(issubclass(w.category, EngineSearchFoldMismatchWarning) for w in caught)
        )

    def test_cross_check_binds_tier2_pins_against_env_surfaces(self) -> None:
        def check_pin(player_id, annotate):
            policy, source = self._annotated_policy([])
            context = self._context(self.LEAD + self.ROUND2)
            context.player_id = player_id
            initial = policy._advance_live_fold(context)
            self.assertIsNotNone(initial)
            annotated_tokens = tuple(annotate(token) for token in initial.products().transition_tokens)
            source._state = _FakeAnnotationState(annotated_tokens)
            refreshed = self._context(self.LEAD + self.ROUND2, round_index=1)
            refreshed.player_id = player_id
            fold = policy._advance_live_fold(refreshed)
            self.assertIsNotNone(fold)
            products = fold.products()
            perspective = type(
                "Perspective",
                (),
                {
                    "showdown_slot": player_id,
                    "opponent_showdown_slot": "p2" if player_id == "p1" else "p1",
                },
            )()
            bound_state = type("BoundState", (), {})()
            bound_state.transition_tokens = annotated_tokens
            bound_state.turn_merged_tokens = tuple(products.turn_merged_tokens)
            bound_state.tendency_stats = products.tendency_stats
            bound_state.perspective = perspective
            source._state = bound_state
            import warnings as _warnings

            with _warnings.catch_warnings(record=True):
                _warnings.simplefilter("always")
                policy._fold_cross_check(
                    refreshed, fold, refreshed.public_materialization_state.replay
                )
            self.assertEqual(policy.stats.fold_cross_check_failures, 0)
            return products

        cb_products = check_pin(
            "p2",
            lambda token: replace(
                token,
                cb_bit=token.kind == "move" and token.actor_slot == "p1",
            ),
        )
        self.assertTrue(cb_products.cb_pinned_species)
        investment_products = check_pin(
            "p1",
            lambda token: replace(
                token,
                investment=(
                    0.5
                    if token.kind == "move"
                    and token.actor_slot == "p1"
                    and token.defender_species
                    else 0.0
                ),
            ),
        )
        self.assertTrue(investment_products.investment_pinned)


class _CumulativeAnnotationSource:
    """`EnvTier2AnnotationSource`-shaped stub with the real cumulative contract.

    `overlay_for` is "the env trackers' per-index conclusions, CUMULATIVE from
    battle start" -- it never stops offering an index, which is exactly why the
    fold's pruning window and the source's overlay drift apart.
    """

    def __init__(self):
        self.overlay: dict[int, tuple] = {}
        self.enabled = True
        self.calls = 0

    def offer(self, index, values):
        self.overlay[index] = values

    def active(self):
        return self.enabled

    def boundary_state(self, player_id):  # pragma: no cover - cross-check only
        raise NotImplementedError

    def overlay_for(self, player_id):
        self.calls += 1
        return dict(self.overlay)


class _WindowedAnnotationSource:
    """A hypothetical source that offers only what the fold can still identify.

    Not a real shape -- the control arm for the no-op proof. Against it the
    adapter never re-seats anything, so any state difference from the
    cumulative arm is caused by re-seating.

    Deliberately NOT a subclass of `_CumulativeAnnotationSource`:
    `test_unreachable_readjudication.EveryWorkflowTestCountGuardMatchesItsModuleTests`
    forbids same-module inheritance here, because it would break the AST
    derivation behind this module's exact test-count pin.
    """

    fold = None

    def __init__(self):
        self.overlay: dict[int, tuple] = {}
        self.enabled = True
        self.calls = 0

    def offer(self, index, values):
        self.overlay[index] = values

    def active(self):
        return self.enabled

    def boundary_state(self, player_id):  # pragma: no cover - cross-check only
        raise NotImplementedError

    def overlay_for(self, player_id):
        self.calls += 1
        overlay = dict(self.overlay)
        if self.fold is None:
            return overlay
        tail_start = self.fold.action_total - len(self.fold.action_tail)
        return {
            index: values
            for index, values in overlay.items()
            if index in self.fold.annotations
            or tail_start <= index <= self.fold.action_total
        }


class _PerSeatAnnotationSource:
    """`EnvTier2AnnotationSource`-shaped stub whose overlay is PER SEAT.

    The real `overlay_for` takes `player_id` and reads that seat's own tracker
    state, so the two seats of one battle can hold different conclusions at
    different indices. `_CumulativeAnnotationSource` is deliberately seat-blind
    (the adversarial shape for the READ side of the per-seat record); this is the
    honest shape, and it is what the WRITE side needs -- a mutant that writes
    every seat's conclusions under one key is invisible unless both seats
    actually apply something.

    Deliberately NOT a subclass of `_CumulativeAnnotationSource`, for the reason
    given on `_WindowedAnnotationSource`: the AST derivation behind this module's
    exact test-count pin forbids same-module inheritance here.
    """

    def __init__(self):
        self.overlays: dict[str, dict[int, tuple]] = {}
        self.enabled = True
        self.calls = 0

    def offer(self, player_id, index, values):
        self.overlays.setdefault(player_id, {})[index] = values

    def active(self):
        return self.enabled

    def boundary_state(self, player_id):  # pragma: no cover - cross-check only
        raise NotImplementedError

    def overlay_for(self, player_id):
        self.calls += 1
        return dict(self.overlays.get(player_id, {}))


class Tier2PrunedConclusionTests(unittest.TestCase):
    """The cumulative source vs. the windowed fold: `fallback:live_fold_broken`.

    Measured mechanism (control block, seed 9900080, both seats, round 248):
    ``tracker annotations for indices [2] arrived outside the identifiable range
    [4, 516]``. Index 2's conclusion did NOT arrive late — it was applied at an
    early boundary and then evicted by ``FoldState._prune_annotations`` once the
    512-entry action tail slid past it, while ``EnvTier2AnnotationSource`` kept
    offering the whole cumulative overlay. The adapter read
    ``index not in fold.annotations`` as "never applied" and refused the truth.

    These tests reproduce that at the same shape with a 4-entry tail, and pin
    the two directions the guard still has to hold in. (One test widens the tail
    to 64 instead, because it needs the OPPOSITE fixture -- an index the fold
    keeps rather than evicts.)

    ⚠ **SAMPLE SIZE OF THE MEASURED RESULT: n=1 in root events.** The whole class
    was measured on ONE battle -- 2 root events, 4 refused decisions (``_fold_broken``
    is sticky per ``(battle, seat)``, so 2 events counted as 4), one seed
    (9900080), both seats. #1216's headline "live_fold_broken 4 → 0" is that one
    battle, and it is reported here next to the number rather than only as a
    figure, because "4 → 0" reads like a population result and is not one. The
    fix is minimal and its no-op-ness is proved directly (see
    ``test_resettling_a_pruned_conclusion_changes_no_fold_state``), so this is a
    disclosure about GENERALITY, not a doubt about the four. What it does mean:
    the trigger (>512 per-action tokens in one battle) has been exercised in
    production exactly once, so the frequency of the class, and whether long
    battles have further variants of it, are UNMEASURED. The census block cannot
    close that -- no census game reaches 512 per-action tokens -- so a
    deliberately-long-games block is the measurement that would.
    """

    LEAD = LiveFoldAdvanceTests.LEAD
    _context = LiveFoldAdvanceTests._context
    # Index 2 is the first ACTION token after the two lead switches -- the same
    # index the production failure names.
    ANNOTATED_INDEX = 2
    VALUES = (0.25, True, False, 0.0)

    @staticmethod
    def _round(turn: int) -> list[str]:
        return [
            "|move|p1a: Rattata|Tackle|p2a: Chansey",
            f"|-damage|p2a: Chansey|{641 - turn}/641",
            "|move|p2a: Chansey|Pound|p1a: Rattata",
            f"|-damage|p1a: Rattata|{300 - turn}/300",
            "|upkeep",
            f"|turn|{turn + 1}",
        ]

    def _policy_with_small_tail(self, source, battle_id="fold-prune", action_tail=4):
        """A policy whose fold has a 4-token action tail and a 1-token merged tail.

        Production's limits are 512/128; the eviction the bug rides on needs a
        battle longer than the action tail, which is 249 rounds at the real
        limits and four here. Nothing else about the path changes: this is the
        SAME ``_advance_live_fold`` -> ``FoldState.advance_in_place`` ->
        ``_prune_annotations`` -> ``_apply_tier2_overlay`` chain.

        ``action_tail`` is widened by exactly one test, which needs the OPPOSITE
        of eviction: an index the fold KEEPS across many boundaries, the way
        production's 512-entry tail keeps one.
        """
        policy = EngineMctsPolicy(
            dex=None, set_source=None, module=object(),
            config=EngineMctsConfig(), annotation_source=source,
        )
        return policy, self._seat_fold(policy, battle_id, "p1", action_tail=action_tail)

    @staticmethod
    def _seat_fold(policy, battle_id, seat, action_tail=4):
        """Seat one small-tailed fold on ``policy`` and return its record key."""
        from pokezero.transitions_fold import FoldState

        key = (battle_id, seat)
        policy._live_folds[key] = FoldState.initial(
            perspective_slot=seat, action_tail_limit=action_tail, merged_tail_limit=1
        )
        policy._fold_consumed[key] = 0
        return key

    def _lines_through(self, rounds):
        """The public log after ``rounds`` further rounds past the lead."""
        return list(self.LEAD) + [
            line for turn in range(1, rounds + 1) for line in self._round(turn)
        ]

    def _drive(self, policy, rounds, battle_id="fold-prune", offer_from=1,
               seat="p1", values=None):
        """Advance ``rounds`` boundaries; the source starts offering index 2 at
        ``offer_from``. Returns (last fold or None, list of per-round folds)."""
        lines = list(self.LEAD)
        folds = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for index in range(rounds):
                if index:
                    lines = lines + self._round(index)
                context = self._context(lines, battle_id=battle_id, round_index=index)
                context.player_id = seat
                fold = policy._advance_live_fold(context)
                folds.append(fold)
                if fold is None:
                    break
                if index >= offer_from:
                    policy._annotation_source.offer(
                        self.ANNOTATED_INDEX,
                        self.VALUES if values is None else values,
                    )
        self._caught = list(caught)
        return folds

    def _assert_no_fold_break(self, folds):
        from pokezero.engine_search import EngineSearchFoldMismatchWarning

        self.assertNotIn(None, folds, "the fold broke")
        self.assertFalse(
            [w for w in self._caught if issubclass(w.category, EngineSearchFoldMismatchWarning)],
            "a fold-mismatch warning was raised",
        )

    def test_a_pruned_conclusion_the_source_keeps_offering_does_not_break_the_fold(self):
        """THE regression. 8 boundaries drives index 2 out of the 4-token tail."""
        source = _CumulativeAnnotationSource()
        policy, key = self._policy_with_small_tail(source)
        folds = self._drive(policy, 8)
        self._assert_no_fold_break(folds)
        final = folds[-1]
        tail_start = final.action_total - len(final.action_tail)
        # The premise of the test: index 2 really did leave the identifiable
        # range and really was pruned. Without this the test passes vacuously.
        self.assertGreater(tail_start, self.ANNOTATED_INDEX)
        self.assertNotIn(self.ANNOTATED_INDEX, final.annotations)
        self.assertNotIn(self.ANNOTATED_INDEX, final.rep_index_map)
        self.assertIn(self.ANNOTATED_INDEX, source.overlay)
        # Applied exactly once, then re-seated on every later boundary.
        self.assertEqual(policy.stats.fold_annotations_applied, 1)

    def test_the_resettled_conclusion_is_counted_not_silent(self):
        """A class that stops refusing must not stop being visible."""
        source = _CumulativeAnnotationSource()
        policy, _key = self._policy_with_small_tail(source)
        self._assert_no_fold_break(self._drive(policy, 8))
        self.assertGreater(policy.stats.fold_annotations_resettled, 0)
        self.assertGreater(policy.stats.fold_annotation_resettle_boundaries, 0)
        payload = policy.stats.to_dict()
        self.assertEqual(
            payload["fold_annotations_resettled"], policy.stats.fold_annotations_resettled
        )
        self.assertEqual(
            payload["fold_annotation_resettle_boundaries"],
            policy.stats.fold_annotation_resettle_boundaries,
        )

    def test_resettling_a_pruned_conclusion_changes_no_fold_state(self):
        """The no-op claim, proved rather than argued.

        Arm A gets the real CUMULATIVE overlay (index 2 offered forever). Arm B
        gets a hypothetical windowed source that stops offering index 2 the
        moment it is pruned, so arm B never re-seats anything. If re-seating
        touched the fold at all the two serialized states would differ.
        """
        source_a = _CumulativeAnnotationSource()
        policy_a, key_a = self._policy_with_small_tail(source_a, battle_id="arm-a")
        self._assert_no_fold_break(self._drive(policy_a, 8, battle_id="arm-a"))

        source_b = _WindowedAnnotationSource()
        policy_b, key_b = self._policy_with_small_tail(source_b, battle_id="arm-b")
        source_b.fold = policy_b._live_folds[key_b]
        self._assert_no_fold_break(self._drive(policy_b, 8, battle_id="arm-b"))

        self.assertGreater(policy_a.stats.fold_annotations_resettled, 0)
        self.assertEqual(policy_b.stats.fold_annotations_resettled, 0)
        self.assertEqual(
            policy_a._live_folds[key_a].to_payload(),
            policy_b._live_folds[key_b].to_payload(),
        )
        self.assertEqual(
            policy_a.stats.fold_annotations_applied,
            policy_b.stats.fold_annotations_applied,
        )

    def test_a_changed_conclusion_on_a_pruned_index_still_breaks_the_fold(self):
        """The fail-open direction. Per-index immutability must survive pruning.

        Dropping pruned indices from the overlay instead of re-seating them
        would make this silent -- the fold would never see the changed value.
        """
        from pokezero.engine_search import EngineSearchFoldMismatchWarning

        source = _CumulativeAnnotationSource()
        policy, _key = self._policy_with_small_tail(source)
        self._assert_no_fold_break(self._drive(policy, 8))
        # A tracker conclusion that CHANGED after the fold pruned its index.
        source.offer(self.ANNOTATED_INDEX, (0.75, True, False, 0.0))
        lines = list(self.LEAD) + [
            line for turn in range(1, 8) for line in self._round(turn)
        ] + self._round(8)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = policy._advance_live_fold(
                self._context(lines, battle_id="fold-prune", round_index=8)
            )
        self.assertIsNone(result)
        messages = [str(w.message) for w in caught]
        self.assertTrue(
            any(issubclass(w.category, EngineSearchFoldMismatchWarning) for w in caught),
            messages,
        )
        # The RIGHT diagnosis: immutability, not a bogus "arrived outside the
        # identifiable range". A mutant that reports the wrong cause is a
        # mutant that survives on the count alone.
        self.assertTrue(
            any("per-index immutable" in message for message in messages), messages
        )

    def test_a_never_applied_conclusion_past_the_tail_still_breaks_the_fold(self):
        """The guard's motivating case, bounded rather than deleted.

        Same shape as the regression -- a positive index below ``tail_start`` --
        but this one was never applied, so the fold has no banked contribution
        for it and applying it now is impossible. Still loud.
        """
        from pokezero.engine_search import EngineSearchFoldMismatchWarning

        source = _CumulativeAnnotationSource()
        policy, _key = self._policy_with_small_tail(source)
        # Never offer anything, so nothing is ever applied.
        self._assert_no_fold_break(self._drive(policy, 8, offer_from=99))
        self.assertEqual(policy.stats.fold_annotations_applied, 0)
        source.offer(self.ANNOTATED_INDEX, self.VALUES)
        lines = list(self.LEAD) + [
            line for turn in range(1, 8) for line in self._round(turn)
        ] + self._round(8)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = policy._advance_live_fold(
                self._context(lines, battle_id="fold-prune", round_index=8)
            )
        self.assertIsNone(result)
        messages = [str(w.message) for w in caught]
        self.assertTrue(
            any(issubclass(w.category, EngineSearchFoldMismatchWarning) for w in caught),
            messages,
        )
        self.assertTrue(
            any("outside the identifiable range" in message for message in messages),
            messages,
        )

    def test_a_conclusion_on_the_open_window_index_is_identifiable(self):
        """The upper edge of the identifiable range, which nothing else pinned.

        ``_token_identity`` accepts ``index == action_total`` while a window is
        open, so the adapter's range test has to be inclusive there. Found by a
        SAFER-direction mutant (`<=` -> `<`) surviving the suite: a strictly
        more conservative guard that nothing could see.
        """
        source = _CumulativeAnnotationSource()
        policy, key = self._policy_with_small_tail(source, battle_id="open-window")
        # A mid-turn boundary: the tackle has been emitted, the turn has not
        # closed, so index 2 is the OPEN window's virtual token.
        lines = list(self.LEAD) + [
            "|move|p1a: Rattata|Tackle|p2a: Chansey",
            "|-damage|p2a: Chansey|468/641",
        ]
        source.offer(self.ANNOTATED_INDEX, self.VALUES)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fold = policy._advance_live_fold(
                self._context(lines, battle_id="open-window", round_index=0)
            )
        self._caught = list(caught)
        self._assert_no_fold_break([fold])
        self.assertEqual(fold.action_total, self.ANNOTATED_INDEX)  # the edge itself
        self.assertIsNotNone(fold.current_window)
        self.assertEqual(policy.stats.fold_annotations_applied, 1)
        self.assertEqual(
            fold.products().transition_tokens[-1].residual, self.VALUES[0]
        )

    def test_the_applied_record_does_not_outlive_its_fold(self):
        """The other fail-open: a record kept past its fold re-seats a
        conclusion that fold never applied. Both drop sites are pinned."""
        source = _CumulativeAnnotationSource()
        policy, key = self._policy_with_small_tail(source, battle_id="battle-a")
        self._assert_no_fold_break(self._drive(policy, 8, battle_id="battle-a"))
        self.assertIn(self.ANNOTATED_INDEX, policy._fold_annotations_seen[key])

        # 1. another battle drops it (drivers run one battle at a time).
        source.enabled = False
        policy._advance_live_fold(
            self._context(self.LEAD, battle_id="battle-b", round_index=0)
        )
        self.assertNotIn(key, policy._fold_annotations_seen)

        # 2. a fold rebuilt from scratch for the SAME key starts empty.
        fresh = ("battle-b", "p1")
        policy._live_folds.pop(fresh, None)
        policy._fold_consumed.pop(fresh, None)
        policy._fold_annotations_seen[fresh] = {self.ANNOTATED_INDEX: self.VALUES}
        policy._advance_live_fold(
            self._context(self.LEAD, battle_id="battle-b", round_index=1)
        )
        self.assertEqual(policy._fold_annotations_seen.get(fresh, {}), {})

    def test_the_applied_record_is_not_shared_across_SEATS(self):
        """Cross-BATTLE isolation is pinned above; cross-SEAT was pinned by
        nothing, and a single record shared by every `(battle, seat)` key
        survived the whole suite.

        The record key is `(battle_id, player_id)` because the two seats' folds
        are independent objects with independent annotation histories: the
        per-action index space is the shared public stream, but whether a
        tracker has concluded anything for a given index is perspective-
        dependent (`EnvTier2AnnotationSource.overlay_for` takes `player_id`). A
        record shared across seats re-seats p1's conclusion into a p2 fold that
        never applied it and has no banked contribution for it -- silently
        desynchronizing p2's encoder-visible surface, which is exactly the
        fail-open the guard exists to prevent.

        The source here is deliberately seat-blind, which is the strongest form
        of the hazard: both seats are offered the same cumulative overlay, so
        only the per-seat RECORD can tell p1's applied index from p2's late one.
        """
        from pokezero.engine_search import EngineSearchFoldMismatchWarning

        source = _CumulativeAnnotationSource()
        policy, key_p1 = self._policy_with_small_tail(source, battle_id="two-seats")
        key_p2 = self._seat_fold(policy, "two-seats", "p2")

        # p2 drives first, while the source offers nothing, so p2's fold applies
        # NOTHING and index 2 leaves its identifiable range unrecorded.
        self._assert_no_fold_break(
            self._drive(policy, 8, battle_id="two-seats", offer_from=99, seat="p2")
        )
        self.assertNotIn(
            self.ANNOTATED_INDEX, policy._fold_annotations_seen.get(key_p2, {})
        )
        # p1 then applies index 2 and keeps it past the prune -- the regression
        # path, on the OTHER seat of the same battle.
        self._assert_no_fold_break(
            self._drive(policy, 8, battle_id="two-seats", seat="p1")
        )

        # One more p2 boundary. Index 2 is now in the overlay, below p2's
        # tail_start, and absent from p2's OWN record: p2 must refuse.
        context = self._context(
            self._lines_through(8), battle_id="two-seats", round_index=8
        )
        context.player_id = "p2"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = policy._advance_live_fold(context)
        self.assertIsNone(result, "p2 re-seated a conclusion only p1 ever applied")
        messages = [str(w.message) for w in caught]
        self.assertTrue(
            any(issubclass(w.category, EngineSearchFoldMismatchWarning) for w in caught),
            messages,
        )
        # The right diagnosis, so a mutant cannot die of the wrong cause.
        self.assertTrue(
            any("outside the identifiable range" in message for message in messages),
            messages,
        )
        self.assertIn(
            self.ANNOTATED_INDEX, policy._fold_annotations_seen.get(key_p1, {})
        )
        self.assertNotIn(
            self.ANNOTATED_INDEX, policy._fold_annotations_seen.get(key_p2, {})
        )

    def test_the_applied_record_holds_what_the_fold_APPLIED_not_what_was_OFFERED(self):
        """Recording from `overlay` instead of `fold.annotations` also survived
        the whole suite, and it turns "applied once" into "offered once" -- the
        exact property `_apply_tier2_overlay`'s monotonicity argument rests on.

        `apply_annotations_in_place` CANONICALIZES every value it applies
        (`transitions_fold.py:320-327`: float / bool / bool / float), so that a
        loosely-typed overlay -- int-flag trackers, or a `to_payload` round-trip,
        which emits `list(values)` at `transitions_fold.py:1215` -- lands
        byte-identically. Record the raw OFFERED tuple instead and the later
        re-seat hands the fold a value the fold never applied, which the fold's
        own equality check then rejects: the class comes back as a fold break on
        a loosely-typed source.

        `Decimal` is the loose type here because plain int flags CANNOT show it:
        `(1, 1, 0, 0) == (1.0, True, False, 0.0)` is True in Python, so an
        equality assertion is silent on exactly the substitution being ruled out.
        """
        from decimal import Decimal

        loose = (Decimal("0.1"), 1, 0, 0)
        canonical = (0.1, True, False, 0.0)
        # The premise, asserted rather than assumed: raw != canonical here.
        self.assertNotEqual(loose, canonical)
        self.assertEqual(float(loose[0]), canonical[0])

        source = _CumulativeAnnotationSource()
        policy, key = self._policy_with_small_tail(source, battle_id="loose-types")
        self._assert_no_fold_break(
            self._drive(policy, 8, battle_id="loose-types", values=loose)
        )
        self.assertGreater(policy.stats.fold_annotations_resettled, 0)
        record = policy._fold_annotations_seen.get(key, {})
        self.assertEqual(record.get(self.ANNOTATED_INDEX), canonical)
        # Types, not just `==`: see the docstring. This is what dies when the
        # record is filled from the overlay.
        self.assertEqual(
            [type(value) for value in record[self.ANNOTATED_INDEX]],
            [float, bool, bool, float],
        )

    def test_a_record_disagreeing_with_the_fold_fails_LOUDLY(self):
        """The monotonicity invariant's live guard, forced.

        `_apply_tier2_overlay`'s docstring argues this is unreachable: a pruned
        index cannot re-enter `[tail_start, action_total]` because `tail_start`
        is monotonic non-decreasing, so no index is ever applied FRESH twice.
        Per-index immutability does NOT cover it -- that check only runs for
        indices the fold currently holds. The line was `record.setdefault(...)`,
        which keeps the first value silently and can therefore never report the
        invariant breaking; it is now an explicit raise, and this test is what
        makes that raise falsifiable.

        Unreachable in operation means it has to be forced, so the record is
        corrupted directly, with index 2 still HELD -- the fresh-apply shape a
        `tail_start` regression would produce, not the re-seat shape.

        THE CORRUPTION DIFFERS FROM `VALUES` IN THE LAST FIELD ONLY, deliberately.
        It used to differ in the FIRST (`0.75` vs `0.25`), and #1220's review
        measured what that costs: `tuple(previous) != tuple(values)` coarsened to
        `tuple(previous)[:1] != tuple(values)[:1]` -- a guard that compares only
        the residual and ignores the valid flag, the counter-bit and the
        investment -- SURVIVED all 141 tests on `6af47d25`, because the one
        forced corruption happened to move field 0. Moving the LAST field instead
        kills every prefix-truncation of the comparison at no extra cost, and
        loses nothing: the full-tuple guard fires either way.
        """
        from pokezero.engine_search import EngineSearchFoldMismatchWarning

        source = _CumulativeAnnotationSource()
        policy, key = self._policy_with_small_tail(source, battle_id="record-guard")
        self._assert_no_fold_break(self._drive(policy, 3, battle_id="record-guard"))
        fold = policy._live_folds[key]
        # Still held, so the loop below takes the `index in fold.annotations`
        # branch and the failure can only come from the record refresh.
        self.assertIn(self.ANNOTATED_INDEX, fold.annotations)
        self.assertEqual(
            policy._fold_annotations_seen[key][self.ANNOTATED_INDEX], self.VALUES
        )
        corrupted = (0.25, True, False, 0.75)
        self.assertNotEqual(corrupted, self.VALUES, "the corruption corrupts nothing")
        self.assertEqual(
            corrupted[:-1],
            self.VALUES[:-1],
            "the corruption must differ in the LAST field ONLY, or a guard that "
            "compares a prefix of the tuple passes this test unchanged",
        )
        policy._fold_annotations_seen[key][self.ANNOTATED_INDEX] = corrupted

        context = self._context(
            self._lines_through(3), battle_id="record-guard", round_index=3
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            applied = policy._apply_tier2_overlay(context, key, fold)
        self.assertFalse(applied, "the record guard did not fire")
        messages = [str(w.message) for w in caught]
        self.assertTrue(
            any(issubclass(w.category, EngineSearchFoldMismatchWarning) for w in caught),
            messages,
        )
        self.assertTrue(
            any("AssertionError" in message for message in messages), messages
        )
        self.assertTrue(
            any(
                f"applied-conclusion record for token index {self.ANNOTATED_INDEX}"
                in message
                for message in messages
            ),
            messages,
        )
        self.assertTrue(
            any("tail_start regressed" in message for message in messages), messages
        )
        # And the fold is broken for the seat, not silently carried on.
        self.assertIn(key, policy._fold_broken)

    def test_a_held_index_is_re_recorded_across_boundaries_without_complaint(self):
        """The SAFER side of that guard's boundary, and why it is `!=` and not
        `is not None`.

        Found by a safer-direction mutant: raising on ANY re-record, equal value
        included, passed all 138 tests. It would break every production battle,
        because production's action tail is 512, so an applied index is
        re-recorded with the SAME value at each of the ~512 boundaries it stays
        held for. Nothing saw it because the 4-token fixture the rest of this
        class uses prunes index 2 one boundary AFTER it is applied and
        `apply_annotations_in_place` re-prunes every re-seat before the record
        refresh reaches it -- so no test in the class ever re-recorded anything
        at all.

        Same chain, one knob different: a tail long enough to keep index 2.
        """
        source = _CumulativeAnnotationSource()
        policy, key = self._policy_with_small_tail(
            source, battle_id="held-index", action_tail=64
        )
        self._assert_no_fold_break(
            self._drive(policy, 8, battle_id="held-index")
        )
        fold = policy._live_folds[key]
        # The premise, so this cannot pass vacuously: the index is HELD, never
        # pruned, never re-seated -- the opposite fixture to the regression above.
        self.assertEqual(fold.action_total - len(fold.action_tail), 0)
        self.assertIn(self.ANNOTATED_INDEX, fold.annotations)
        self.assertEqual(policy.stats.fold_annotations_resettled, 0)
        # Offered at every one of the 8 boundaries, applied at exactly one --
        # so the record refresh saw an already-recorded index on 5 of them
        # (boundaries 0 and 1 have an empty overlay; boundary 2 applies).
        self.assertEqual(source.calls, 8)
        self.assertEqual(policy.stats.fold_annotations_applied, 1)
        self.assertEqual(policy.stats.fold_annotation_boundaries, 1)
        self.assertEqual(
            policy._fold_annotations_seen[key][self.ANNOTATED_INDEX], self.VALUES
        )

    def test_two_conclusions_with_DIFFERENT_values_both_land_in_the_record(self):
        """The CARDINALITY axis. Every other test in this class offers exactly
        one index, so no test ever put a second entry in the record -- and three
        mutants of that axis survive all 139, one of them a plausible typo on a
        line this PR introduces.

        The blocking one is `previous = record.get(index)` ->
        `previous = next(iter(record.values()), None)`: a wrong-index lookup that
        compares the NEW index's value against some OTHER index's recorded value.
        With two differing conclusions it raises on a completely benign path and
        breaks the seat for the rest of the battle -- the same failure mode as
        the safer-direction mutant that produced
        `test_a_held_index_is_re_recorded_across_boundaries_without_complaint`,
        one index over. Its blast radius is every battle where two Tier-2
        conclusions differ, i.e. the normal case. The other two are
        `fold.annotations.items()` -> `[:1]` and -> `[-1:]`, which record only the
        first / only the last held index and so re-open #1216's bug for every
        other index.

        ⚠ AND THE FIRST VERSION OF THIS TEST HAD TO STAGGER ITS TWO OFFERS, WHICH
        LEFT THE MIRROR MUTANT ALIVE. #1220's review measured it:
        `fold.annotations.items()` -> `list(...)[-1:]` -- record only the LAST
        held index -- SURVIVED all 141 tests on `6af47d25`, while `[:1]` died
        here. The stagger is why, and the stagger is necessary: the record must
        already hold index 2 when index 5 arrives, which is exactly what kills the
        wrong-index-lookup mutant. But under a stagger every boundary applies at
        most one new index, so "last held" and "all newly held" coincide, and TWO
        CONCLUSIONS AT THE SAME BOUNDARY was untested. The cumulative overlay
        makes that ordinary, not exotic: `overlay_for` re-offers the whole
        cumulative map at every boundary, so any two trackers concluding within
        one boundary of each other arrive together.

        Consequence of the survivor, which is why it is not cosmetic: every index
        but the last at such a boundary never enters the record. It is then pruned
        with nothing in `seen`, so the next boundary classifies it `stale`, which
        raises, which is `_mark_fold_broken` -> `fallback:live_fold_broken` -- the
        refusal #1216 closed, re-opened loudly for the ordinary case.

        So the fixture now does both: index 2 alone at one boundary (the stagger,
        kept), then indices 5 AND 6 together at the next (the same-boundary case).
        Three conclusions over two application boundaries, asserted as such below
        so the same-boundary half cannot silently degrade back into a stagger.

        Tail 64, so nothing is pruned and nothing is re-seated: this is purely
        about the record holding more than one thing at a time.
        """
        second_index, second_values = 5, (0.5, True, True, 0.25)
        third_index, third_values = 6, (0.125, False, True, 0.5)
        # The premise: the values must all DIFFER, or a cross-index comparison
        # would be satisfied by accident and this test would pass vacuously.
        self.assertEqual(
            3, len({self.VALUES, second_values, third_values}), "values collide"
        )
        self.assertEqual(
            3, len({self.ANNOTATED_INDEX, second_index, third_index}), "indices collide"
        )

        source = _CumulativeAnnotationSource()
        policy, key = self._policy_with_small_tail(
            source, battle_id="two-indices", action_tail=64
        )
        lines = list(self.LEAD)
        folds = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for boundary in range(8):
                if boundary:
                    lines = lines + self._round(boundary)
                fold = policy._advance_live_fold(
                    self._context(lines, battle_id="two-indices", round_index=boundary)
                )
                folds.append(fold)
                if fold is None:
                    break
                if boundary >= 1:
                    source.offer(self.ANNOTATED_INDEX, self.VALUES)
                if boundary >= 2:
                    # Offered a boundary later, so it is applied a boundary later
                    # and the record already holds index 2 when it arrives.
                    source.offer(second_index, second_values)
                    # SAME boundary as the one above, which is the case the
                    # stagger cannot reach: both are fresh at the next
                    # application, so "the last held index" is no longer "every
                    # index this boundary applied".
                    source.offer(third_index, third_values)
        self._caught = list(caught)
        self._assert_no_fold_break(folds)

        fold = policy._live_folds[key]
        # Premises, so the assertions below cannot pass for the wrong reason.
        self.assertEqual(fold.action_total - len(fold.action_tail), 0)
        self.assertEqual(policy.stats.fold_annotations_resettled, 0)
        self.assertEqual(
            sorted(fold.annotations),
            [self.ANNOTATED_INDEX, second_index, third_index],
        )
        # ALL THREE, with their OWN values. Exact dict, not three `assertIn`s: a
        # mutant that drops any entry has to fail here.
        self.assertEqual(
            policy._fold_annotations_seen[key],
            {
                self.ANNOTATED_INDEX: self.VALUES,
                second_index: second_values,
                third_index: third_values,
            },
        )
        # THE SHAPE PREMISE, and the whole point of the third offer: 3
        # conclusions over 2 application boundaries, so one boundary applied two
        # of them. Without this, a future edit could re-stagger the offers, keep
        # the dict assertion green, and silently give `[-1:]` its life back.
        self.assertEqual(policy.stats.fold_annotations_applied, 3)
        self.assertEqual(policy.stats.fold_annotation_boundaries, 2)
        self.assertNotIn(key, policy._fold_broken)

    def test_each_seats_conclusions_are_WRITTEN_under_its_own_key(self):
        """The WRITE half of the per-seat record.

        `test_the_applied_record_is_not_shared_across_SEATS` pins the READ half
        only, and review measured the gap: collapsing the write key to a constant
        seat SURVIVES it, because there the p2 record is legitimately empty, so
        "absent" and "correctly isolated" are indistinguishable. The write key is
        only observable when BOTH seats apply something, which needs a seat-aware
        source -- the real `overlay_for(player_id)` shape.

        Disjoint indices with different values, tail 64 so nothing prunes: each
        seat's record must hold exactly its own conclusion.
        """
        source = _PerSeatAnnotationSource()
        policy, key_p1 = self._policy_with_small_tail(
            source, battle_id="write-key", action_tail=64
        )
        key_p2 = self._seat_fold(policy, "write-key", "p2", action_tail=64)
        p1_values = self.VALUES
        p2_index, p2_values = 5, (0.5, True, True, 0.25)
        self.assertNotEqual(p2_values, p1_values)

        lines = list(self.LEAD)
        folds = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for boundary in range(6):
                if boundary:
                    lines = lines + self._round(boundary)
                for seat in ("p1", "p2"):
                    context = self._context(
                        lines, battle_id="write-key", round_index=boundary
                    )
                    context.player_id = seat
                    folds.append(policy._advance_live_fold(context))
                if boundary >= 1:
                    source.offer("p1", self.ANNOTATED_INDEX, p1_values)
                if boundary >= 2:
                    source.offer("p2", p2_index, p2_values)
        self._caught = list(caught)
        self._assert_no_fold_break(folds)

        # Each seat applied its own and only its own.
        self.assertEqual(
            policy._fold_annotations_seen[key_p1], {self.ANNOTATED_INDEX: p1_values}
        )
        self.assertEqual(policy._fold_annotations_seen[key_p2], {p2_index: p2_values})
        self.assertEqual(
            sorted(policy._fold_annotations_seen), sorted([key_p1, key_p2])
        )
        # And the folds themselves stayed apart, so the records are not agreeing
        # by way of both seats having been given everything.
        self.assertEqual(
            sorted(policy._live_folds[key_p1].annotations), [self.ANNOTATED_INDEX]
        )
        self.assertEqual(sorted(policy._live_folds[key_p2].annotations), [p2_index])
        self.assertEqual(policy.stats.fold_annotations_applied, 2)


class FallbackAlertTests(unittest.TestCase):
    """Every fallback must be LOUD: warning + logger; strict mode raises."""

    def _fallback_context(self):
        mask = (False, True, False, False, False, False, False, False, False)
        context = _FakeContext(_FakeObservation(mask, _candidates()), public_state=None)
        context.battle_id = "alert-test"
        context.decision_round_index = 7
        return context

    def test_fallback_emits_warning_with_context(self) -> None:
        from pokezero.engine_search import EngineSearchFallbackWarning

        policy = _policy()
        with self.assertWarns(EngineSearchFallbackWarning) as caught:
            policy.select_action_with_context(self._fallback_context(), rng=random.Random(1))
        message = str(caught.warning)
        self.assertIn("FALLBACK", message)
        self.assertIn("battle=alert-test", message)
        self.assertIn("round=7", message)
        self.assertIn("reason=no_public_state", message)

    def test_fallback_logs_on_stable_logger(self) -> None:
        import logging

        policy = _policy()
        with self.assertLogs("pokezero.engine_search.fallback", level=logging.WARNING) as logs:
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                policy.select_action_with_context(self._fallback_context(), rng=random.Random(1))
        self.assertTrue(any("FALLBACK" in line for line in logs.output))

    def test_strict_mode_raises_instead_of_falling_back(self) -> None:
        from pokezero.engine_search import EngineMctsConfig, EngineMctsPolicy, EngineSearchFallbackError

        policy = EngineMctsPolicy(
            dex=None, set_source=None, module=object(),
            config=EngineMctsConfig(strict_fallbacks=True),
        )
        with self.assertRaises(EngineSearchFallbackError) as caught:
            policy.select_action_with_context(self._fallback_context(), rng=random.Random(1))
        self.assertIn("reason=no_public_state", str(caught.exception))


class _FakeState:
    def __init__(self, serialized: str = "state") -> None:
        self._serialized = serialized

    def to_string(self) -> str:
        return self._serialized


class _FakeWorld:
    def __init__(self, side: str = "side_one") -> None:
        self.slot_sides = {"p1": side, "p2": "side_two" if side == "side_one" else "side_one"}


class _FakeCrate:
    """Stand-in for the native module; records the call and returns a report."""

    def __init__(self, reports, *, error: Exception | None = None) -> None:
        self._reports = list(reports)
        self._error = error
        self.calls: list[dict] = []

    def puct_search_multi(self, state_str, iterations, **kwargs):
        self.calls.append({"state_str": state_str, "iterations": iterations, **kwargs})
        if self._error is not None:
            raise self._error
        return json.dumps(self._reports.pop(0))


def _crate_report(side_one, side_two, *, max_depth_reached, iterations=1024):
    return {
        "iterations": iterations,
        "max_depth_reached": max_depth_reached,
        "side_one": side_one,
        "side_two": side_two,
    }


class HandcraftedCrateSearchTests(unittest.TestCase):
    """leaf_eval='hp_fraction_crate': the crate tree with handcrafted leaves.

    The depth-decay study's control arm (docs/mcts_handcrafted_leaf_depth_findings.md)
    depends on this path pricing leaves WITHOUT a model while keeping every other
    knob identical to model mode, and on its depth-reached telemetry being real
    rather than the configured cap echoed back.
    """

    def setUp(self) -> None:
        self.policy = EngineMctsPolicy(
            dex=None,
            set_source=None,
            module=object(),
            config=EngineMctsConfig(
                leaf_eval="hp_fraction_crate", search_sims=64, search_depth=4, worlds=2
            ),
        )
        mask = (True, True, False, False, True, False, False, False, False)
        self.context = _FakeContext(_FakeObservation(mask, _candidates()))

    def _run(self, crate, worlds):
        with patch.dict(sys.modules, {"pokezero_search": crate}):
            return self.policy._search_hp_fraction_crate(
                self.context, worlds, random.Random(7)
            )

    def test_config_needs_no_model_artifacts_but_needs_a_budget(self) -> None:
        EngineMctsConfig(leaf_eval="hp_fraction_crate")  # no model/tables required
        with self.assertRaises(ValueError):
            EngineMctsConfig(leaf_eval="hp_fraction_crate", search_depth=0)
        with self.assertRaises(ValueError):
            EngineMctsConfig(leaf_eval="hp_fraction_crate", search_sims=0)

    def test_search_config_reaches_the_crate_unchanged(self) -> None:
        crate = _FakeCrate(
            [
                _crate_report(
                    [{"move": "earthquake", "visits": 64, "q": 0.6}], [], max_depth_reached=3
                )
            ]
        )
        self._run(crate, [(_FakeWorld(), _FakeState("STATE-1"))])
        call = crate.calls[0]
        self.assertEqual(call["state_str"], "STATE-1")
        self.assertEqual(call["iterations"], 64)
        self.assertEqual(call["max_depth"], 4)
        self.assertEqual(call["c_puct"], 1.4)
        self.assertTrue(call["deep_ko_split"])

    def test_visit_shares_aggregate_across_worlds(self) -> None:
        # World A prefers the switch 3:1; world B prefers earthquake 3:1 but the
        # switch still carries a quarter — the normalized sum decides.
        crate = _FakeCrate(
            [
                _crate_report(
                    [
                        {"move": "switch starmie", "visits": 30, "q": 0.7},
                        {"move": "earthquake", "visits": 10, "q": 0.4},
                    ],
                    [],
                    max_depth_reached=2,
                ),
                _crate_report(
                    [
                        {"move": "earthquake", "visits": 30, "q": 0.6},
                        {"move": "switch starmie", "visits": 10, "q": 0.5},
                    ],
                    [],
                    max_depth_reached=3,
                ),
            ]
        )
        decision = self._run(
            crate, [(_FakeWorld(), _FakeState()), (_FakeWorld(), _FakeState())]
        )
        aggregated = decision.metadata["engine_mcts"]["aggregated_choices"]
        self.assertAlmostEqual(aggregated["earthquake"], 1.0, places=4)
        self.assertAlmostEqual(aggregated["switch starmie"], 1.0, places=4)
        self.assertEqual(decision.metadata["engine_mcts"]["leaf_eval"], "hp_fraction_crate")
        self.assertEqual(decision.metadata["engine_mcts"]["worlds_searched"], 2)

    def test_p2_reads_side_two(self) -> None:
        context = _FakeContext(
            _FakeObservation(
                (True, True, False, False, True, False, False, False, False), _candidates()
            ),
            player_id="p2",
        )
        crate = _FakeCrate(
            [
                _crate_report(
                    [{"move": "hiddenpowergrass70", "visits": 64, "q": 0.9}],
                    [{"move": "earthquake", "visits": 64, "q": 0.2}],
                    max_depth_reached=1,
                )
            ]
        )
        with patch.dict(sys.modules, {"pokezero_search": crate}):
            decision = self.policy._search_hp_fraction_crate(
                context, [(_FakeWorld(), _FakeState())], random.Random(7)
            )
        # p2 sits on side_two in this world, so the side_two entry decides.
        self.assertEqual(decision.action_index, 0)

    def test_depth_reached_is_measured_not_the_configured_cap(self) -> None:
        crate = _FakeCrate(
            [
                _crate_report(
                    [{"move": "earthquake", "visits": 64, "q": 0.5}], [], max_depth_reached=1
                ),
                _crate_report(
                    [{"move": "earthquake", "visits": 64, "q": 0.5}], [], max_depth_reached=3
                ),
            ]
        )
        decision = self._run(
            crate, [(_FakeWorld(), _FakeState()), (_FakeWorld(), _FakeState())]
        )
        stats = self.policy.stats.to_dict()
        self.assertEqual(stats["depth_reached_samples"], 2)
        self.assertEqual(stats["depth_reached_max"], 3)
        self.assertEqual(stats["depth_reached_mean"], 2.0)
        self.assertEqual(stats["depth_reached_histogram"], {"1": 1, "3": 1})
        self.assertEqual(decision.metadata["engine_mcts"]["max_depth_reached"], 3)
        self.assertEqual(decision.metadata["engine_mcts"]["depths_reached"], (1, 3))

    def test_crate_failure_is_attributed_and_falls_back(self) -> None:
        crate = _FakeCrate([], error=ValueError("battle is already over at the root"))
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            decision = self._run(crate, [(_FakeWorld(), _FakeState())])
        stats = self.policy.stats.to_dict()
        self.assertEqual(stats["worlds_searched"], 0)
        self.assertEqual(stats["fallback_decisions"], 1)
        self.assertIn(
            "crate_search_hp: battle is already over at the root",
            stats["world_failure_reasons"],
        )
        self.assertIn(decision.action_index, (0, 1, 4))


class BoundedReasonDetailTests(unittest.TestCase):
    """Overflowing a telemetry key must never make two reasons look like one.

    The old seam was a bare ``[:160]``. Independent review on #1030 showed that
    silently aliases: two DIFFERENT attract sub-case sets that share a prefix
    truncated to byte-identical `world_failure_reasons` keys, and the label that
    got cut was `+volatile` -- one of the non-downgradeable arms the sub-case
    split exists to measure. A measurement that hides its answer inside a bucket
    belonging to a different question is worse than one that is merely coarse.
    """

    def test_the_budget_itself_is_pinned(self) -> None:
        # Every other assertion here is expressed RELATIVE to the constant, so
        # they all move with it and none of them notices a re-narrowed budget.
        # The crate mirrors this number as a hardcoded literal
        # (`PY_REASON_DETAIL_LIMIT` in gen3_confusion_event_renderer.rs) and
        # cannot see this side, so without this pin the two silently
        # desynchronise. Found by independent review as a surviving mutant.
        self.assertGreaterEqual(_REASON_DETAIL_LIMIT, 512)

    def test_a_limit_smaller_than_the_suffix_still_honours_the_bound(self) -> None:
        for limit in (0, 1, 5, 18, 19):
            self.assertLessEqual(len(_bounded_reason_detail("q" * 100, limit=limit)), limit)

    def test_short_reasons_are_untouched(self) -> None:
        text = "attract_empty_tail_ambiguous:paralyzed+miss"
        self.assertEqual(_bounded_reason_detail(text), text)

    def test_boundary_is_inclusive(self) -> None:
        exact = "x" * _REASON_DETAIL_LIMIT
        self.assertEqual(_bounded_reason_detail(exact), exact)
        self.assertEqual(len(_bounded_reason_detail("x" * (_REASON_DETAIL_LIMIT + 1))),
                         _REASON_DETAIL_LIMIT)

    def test_overflow_stays_within_budget(self) -> None:
        self.assertLessEqual(
            len(_bounded_reason_detail("y" * (_REASON_DETAIL_LIMIT * 4))),
            _REASON_DETAIL_LIMIT,
        )

    def test_overflow_announces_itself(self) -> None:
        self.assertIn("~trunc:", _bounded_reason_detail("z" * (_REASON_DETAIL_LIMIT + 50)))

    def test_distinct_reasons_sharing_a_prefix_keep_distinct_keys(self) -> None:
        # The exact failure review reproduced, scaled to the current limit: the
        # shorter set is a strict prefix of the longer one, so a bare slice put
        # both in the same bucket and the trailing sub-case vanished.
        shared = "attribution-unsafe renderer branch rejected before tree/model fold: " + (
            "attract_empty_tail_ambiguous:paralyzed+cannot_act," * 12
        )
        short = shared + "attract_empty_tail_ambiguous:paralyzed+miss"
        long = short + "+volatile"

        self.assertGreater(len(short), _REASON_DETAIL_LIMIT, "fixture must overflow")
        self.assertEqual(short[:_REASON_DETAIL_LIMIT], long[:_REASON_DETAIL_LIMIT],
                         "fixture must alias under a bare slice, or it proves nothing")
        self.assertNotEqual(
            _bounded_reason_detail(short),
            _bounded_reason_detail(long),
            "two different sub-case sets collapsed into one telemetry bucket",
        )

    def test_truncation_is_deterministic_across_calls(self) -> None:
        text = "w" * (_REASON_DETAIL_LIMIT + 7)
        self.assertEqual(_bounded_reason_detail(text), _bounded_reason_detail(text))


class WorldAbortRateTests(unittest.TestCase):
    """`fallback_rate` hides the per-world abort rate behind an exponent.

    A decision falls back only when EVERY world fails, and each world is
    searched under its own seed, so the reported fallback rate is roughly the
    per-world abort rate raised to the Wth power. At W=4 a 60% per-world abort
    rate surfaces as a ~13% fallback rate, which reads as "mostly fine" while
    three quarters of the sampled belief is being discarded on every decision.

    These tests pin the denominator that makes the real rate visible.
    """

    def test_an_empty_denominator_reports_none_not_a_healthy_zero(self) -> None:
        # 0.0 would say "no worlds aborted" when the truth is "nothing was
        # measured". Broken-instrument-reads-as-healthy is the failure this whole
        # metric exists to prevent, so it must not be the metric's own behaviour.
        # `depth_reached_mean` sets the precedent for omitting instead.
        stats = EngineMctsStats().to_dict()
        self.assertEqual(stats["worlds_constructed"], 0)
        self.assertIsNone(stats["world_search_abort_rate"])
        self.assertIsNone(stats["belief_sample_rejection_rate"])

    def test_a_searched_world_with_no_denominator_does_not_read_as_healthy(self) -> None:
        # The shard-fixture bug found in review: worlds_searched populated,
        # worlds_constructed not. Must not report a clean 0.0 abort rate.
        stats = EngineMctsStats()
        stats.worlds_attempted = 160
        stats.worlds_searched = 152
        self.assertIsNone(stats.to_dict()["world_search_abort_rate"])

    def test_the_two_failure_modes_are_separated(self) -> None:
        # 10 attempts -> 8 built -> 8 searches attempted -> 6 searched. Those are two
        # DIFFERENT defects (belief sampling / world building vs. the search aborting
        # on an attribution-unsafe branch) and were previously only separable by
        # parsing the `world_failure_reasons` taxonomy.
        stats = EngineMctsStats()
        stats.worlds_attempted = 10
        stats.worlds_constructed = 8
        stats.world_search_attempts = 8
        stats.worlds_searched = 6
        payload = stats.to_dict()
        self.assertAlmostEqual(payload["belief_sample_rejection_rate"], 0.2)
        self.assertAlmostEqual(payload["world_search_abort_rate"], 0.25)

    def test_a_ladder_reuses_its_worlds_and_the_abort_rate_survives_it(self) -> None:
        # THREE denominators, and the reason there are three. A ladder searches the
        # same 4 constructed worlds again at every rung -- 4 + 3 + 2 + 1 = 10 search
        # attempts against 4 constructions -- so an abort rate computed against
        # CONSTRUCTIONS reads 1 - 10/4 = -1.5. Measured at -1.75 on a real canary
        # before review caught it.
        stats = EngineMctsStats()
        stats.worlds_attempted = 4
        stats.worlds_constructed = 4
        stats.world_search_attempts = 10
        stats.worlds_searched = 10
        payload = stats.to_dict()
        self.assertEqual(payload["world_search_abort_rate"], 0.0)
        self.assertEqual(payload["belief_sample_rejection_rate"], 0.0)
        # And it still MEASURES: two of the ten searches abort.
        stats.worlds_searched = 8
        self.assertAlmostEqual(stats.to_dict()["world_search_abort_rate"], 0.2)

    def test_a_partial_abort_is_not_a_fallback_but_is_still_visible(self) -> None:
        """The case the whole counter exists for.

        One of two worlds aborts. The decision succeeds, so `fallback_rate`
        stays 0 and every existing metric calls this healthy -- while the
        aggregate actually rests on half the sampled hypotheses.

        This test pins the per-decision metadata against a hand-seeded counter;
        the abort RATE itself is asserted by the three end-to-end `decide()`
        tests below, which are the only place the real denominator is built.
        """
        harness = EarlyStopPolicyIntegrationTests()
        policy = harness._policy(early_stop=False)
        # `decide()` owns this increment at its single dispatch point; the test
        # enters at `_search_model`, so it stands in for that here.
        #
        # Deliberately NOT 2. Setting the cumulative counter equal to
        # `len(worlds)` makes those two expressions alias, and the per-decision
        # metadata assertion below then cannot tell them apart -- independent
        # review showed the cumulative-counter mutant surviving exactly that way.
        # 7 stands in for "seven worlds already constructed on earlier
        # decisions". Entering at `_search_model` never increments, so the
        # counter stays 7 while this decision's metadata must read 2.
        policy.stats.worlds_constructed = 7
        native = harness._Native(
            [
                harness._report(56, 4, stopped=False),
                ValueError("attribution-unsafe renderer branch rejected before leaf encode"),
            ]
        )

        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            decision = harness._run(
                policy, native, [harness._world("world-a"), harness._world("world-b")]
            )

        stats = policy.stats.to_dict()
        self.assertEqual(stats["fallback_decisions"], 0, "a partial abort is not a fallback")
        self.assertEqual(stats["worlds_searched"], 1)
        self.assertEqual(stats["worlds_constructed"], 7)
        self.assertEqual(decision.metadata["engine_mcts"]["worlds_searched"], 1)
        # PER DECISION (2), never the running total (7).
        self.assertEqual(decision.metadata["engine_mcts"]["worlds_constructed"], 2)

    def test_the_denominator_is_incremented_on_the_real_decision_path(self) -> None:
        """Pins the increment itself, not just the arithmetic over it.

        The counter lives at `decide()`'s single dispatch point, which the
        `_search_model`-entry tests above bypass. Without this, deleting the
        increment leaves the whole suite green and every abort rate silently
        reads 0.0 -- a broken measurement that looks like a clean bill of health,
        which is the exact failure this metric exists to prevent.
        """
        import unittest.mock as mock

        from pokezero.engine_world import EngineWorld

        module = mock.Mock()
        module.monte_carlo_tree_search.return_value = OwnSideSelectionTests._Result()
        policy = EngineMctsPolicy(
            dex=None,
            set_source=None,
            module=module,
            config=EngineMctsConfig(worlds=3, sample_retry_factor=1),
        )
        candidates = [
            {"action_index": 0, "kind": "move", "legal": True, "move_id": "earthquake"},
            {"action_index": 1, "kind": "move", "legal": True, "move_id": "surf"},
        ]
        mask = (True, True, False, False, False, False, False, False, False)
        context = _FakeContext(_FakeObservation(mask, candidates))
        world = EngineWorld(
            spec=None,
            slot_sides={"p1": "side_one", "p2": "side_two"},
            party_species={"p1": (), "p2": ()},
        )
        with mock.patch(
            "pokezero.engine_search._gen3_randbat_belief_start_override_result",
            return_value=(object(), None),
        ), mock.patch(
            "pokezero.engine_search.world_battle_spec", return_value=world
        ), mock.patch(
            "pokezero.engine_search.build_poke_engine_state", return_value=object()
        ):
            policy.select_action_with_context(context, rng=random.Random(0))

        stats = policy.stats.to_dict()
        self.assertEqual(stats["worlds_constructed"], 3)
        self.assertEqual(stats["worlds_attempted"], 3)
        self.assertEqual(stats["belief_sample_rejection_rate"], 0.0)

    def test_partial_construction_over_two_decisions_keeps_the_two_rates_apart(self) -> None:
        """The case that separates `len(worlds)` from every look-alike.

        Found by independent review: with a sampler that ALWAYS succeeds,
        `len(worlds)` and `config.worlds` are indistinguishable, and a
        per-decision metadata value is indistinguishable from the cumulative
        counter. Both mutants survived. This drives `decide()` twice with a
        PARTIALLY failing sampler so the three quantities all differ:

          * requested 3, sampler yields 2 -> `len(worlds)` 2, `config.worlds` 3
          * after two decisions the cumulative counter is 4, the per-decision
            value is still 2

        Without this, worlds lost to CONSTRUCTION get re-attributed to
        `world_search_abort_rate`, re-merging the exact two defects this PR
        exists to separate.
        """
        import unittest.mock as mock

        from pokezero.engine_world import EngineWorld

        module = mock.Mock()
        module.monte_carlo_tree_search.return_value = OwnSideSelectionTests._Result()
        policy = EngineMctsPolicy(
            dex=None,
            set_source=None,
            module=module,
            config=EngineMctsConfig(worlds=3, sample_retry_factor=1),
        )
        candidates = [
            {"action_index": 0, "kind": "move", "legal": True, "move_id": "earthquake"},
            {"action_index": 1, "kind": "move", "legal": True, "move_id": "surf"},
        ]
        mask = (True, True, False, False, False, False, False, False, False)
        world = EngineWorld(
            spec=None,
            slot_sides={"p1": "side_one", "p2": "side_two"},
            party_species={"p1": (), "p2": ()},
        )

        # 3 attempts per decision, retry factor 1: the middle one is rejected by
        # the belief sampler, so each decision constructs exactly 2 of 3.
        attempts = {"n": 0}

        def sampler(**_kwargs):
            attempts["n"] += 1
            return (None, "rejected") if attempts["n"] % 3 == 2 else (object(), None)

        decisions = []
        with mock.patch(
            "pokezero.engine_search._gen3_randbat_belief_start_override_result",
            side_effect=sampler,
        ), mock.patch(
            "pokezero.engine_search.world_battle_spec", return_value=world
        ), mock.patch(
            "pokezero.engine_search.build_poke_engine_state", return_value=object()
        ):
            for _ in range(2):
                context = _FakeContext(_FakeObservation(mask, candidates))
                decisions.append(
                    policy.select_action_with_context(context, rng=random.Random(0))
                )

        stats = policy.stats.to_dict()
        self.assertEqual(stats["worlds_attempted"], 6)
        # 2 per decision, NOT config.worlds (3) and NOT worlds_attempted (6).
        self.assertEqual(stats["worlds_constructed"], 4)
        self.assertEqual(stats["worlds_searched"], 4)
        # Construction loss must NOT leak into the abort rate.
        self.assertEqual(stats["world_search_abort_rate"], 0.0)
        self.assertAlmostEqual(stats["belief_sample_rejection_rate"], 1.0 / 3.0)

        # Per-decision metadata is PER DECISION, not the running total: both
        # decisions report 2, not 2 then 4.
        for decision in decisions:
            self.assertEqual(
                decision.metadata["engine_mcts"]["worlds_constructed"], 2, decision.metadata
            )

    def test_a_searching_path_reports_its_own_decisions_denominator(self) -> None:
        """Same mutant, on a path that can actually abort a world.

        The two-decision test above runs the legacy `hp_fraction` path, which
        never aborts, so it cannot tell a per-decision value from the cumulative
        counter on the paths that matter. This drives `decide()` end to end on
        `hp_fraction_crate` twice, with construction losing a world each time AND
        the crate aborting one of the survivors, and asserts the metadata is
        per-decision. Substituting `self.stats.worlds_constructed` here reports 2
        then 4 and fails.
        """
        import unittest.mock as mock

        from pokezero.engine_world import EngineWorld

        policy = EngineMctsPolicy(
            dex=None,
            set_source=None,
            module=object(),
            config=EngineMctsConfig(
                leaf_eval="hp_fraction_crate", search_sims=64, search_depth=4,
                worlds=3, sample_retry_factor=1,
            ),
        )
        mask = (True, True, False, False, True, False, False, False, False)
        world = EngineWorld(
            spec=None,
            slot_sides={"p1": "side_one", "p2": "side_two"},
            party_species={"p1": (), "p2": ()},
        )

        attempts = {"n": 0}

        def sampler(**_kwargs):
            attempts["n"] += 1
            return (None, "rejected") if attempts["n"] % 3 == 2 else (object(), None)

        # Two constructed worlds per decision; the crate aborts the second.
        good = _crate_report([{"move": "earthquake", "visits": 64, "q": 0.6}], [],
                             max_depth_reached=2)
        crate = _FakeCrate([good, ValueError("boom"), good, ValueError("boom")])

        decisions = []
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            with mock.patch(
                "pokezero.engine_search._gen3_randbat_belief_start_override_result",
                side_effect=sampler,
            ), mock.patch(
                "pokezero.engine_search.world_battle_spec", return_value=world
            ), mock.patch(
                "pokezero.engine_search.build_poke_engine_state",
                side_effect=lambda *a, **k: _FakeState(),
            ), patch.dict(sys.modules, {"pokezero_search": crate}):
                for _ in range(2):
                    context = _FakeContext(_FakeObservation(mask, _candidates()))
                    decisions.append(
                        policy.select_action_with_context(context, rng=random.Random(0))
                    )

        stats = policy.stats.to_dict()
        self.assertEqual(stats["worlds_constructed"], 4)
        self.assertEqual(stats["worlds_searched"], 2)
        # Now BOTH defects are present and must stay separated.
        self.assertAlmostEqual(stats["world_search_abort_rate"], 0.5)
        self.assertAlmostEqual(stats["belief_sample_rejection_rate"], 1.0 / 3.0)
        for decision in decisions:
            engine = decision.metadata["engine_mcts"]
            self.assertEqual(engine["worlds_constructed"], 2, decision.metadata)
            self.assertEqual(engine["worlds_searched"], 1, decision.metadata)

    def test_the_increment_is_reached_on_the_model_path_the_campaign_runs(self) -> None:
        """`leaf_eval="model"` is the only path FoulPlay actually runs.

        Found by independent review as a third surviving mutant: guarding the
        increment with `if self._config.leaf_eval != "model"` left all 96 tests
        green, because every model-path test enters at `_search_model` with a
        hand-seeded counter and nothing observes whether `_search` incremented
        before dispatching. Under that mutant every FoulPlay shard reports
        `worlds_constructed: 0` and `world_search_abort_rate: null` forever --
        a dead metric on the one path anyone runs.

        `foulplay_bridge._build_policy`, `mcts_acceptance_h2h.build_policies` and
        `k0_grid_h2h.main` all pass `leaf_eval="model"` to `EngineMctsConfig`.
        """
        import unittest.mock as mock

        from pokezero.engine_world import EngineWorld

        harness = EarlyStopPolicyIntegrationTests()
        policy = harness._policy(early_stop=False)
        policy._config = replace(policy._config, worlds=3, sample_retry_factor=1)
        policy.stats = EngineMctsStats()
        # `_policy()` builds via `object.__new__` for the `_search_model`-entry
        # tests, so the attributes only `_search` touches are absent.
        policy._fixed_override = None
        policy._dex = None
        policy._set_source = None
        policy._module = object()
        policy._env_tier2_source = None

        world = EngineWorld(
            spec=None,
            slot_sides={"p1": "side_one", "p2": "side_two"},
            party_species={"p1": ("rattata",), "p2": ("chansey",)},
        )
        attempts = {"n": 0}

        def sampler(**_kwargs):
            attempts["n"] += 1
            return (None, "rejected") if attempts["n"] % 3 == 2 else (object(), None)

        # Two worlds constructed per decision; the crate aborts one of them.
        native = harness._Native(
            [harness._report(56, 4, stopped=False), ValueError("attribution-unsafe")]
        )
        fake_module = SimpleNamespace(
            FoldState=SimpleNamespace(from_payload=lambda _payload: object())
        )

        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            with mock.patch(
                "pokezero.engine_search._gen3_randbat_belief_start_override_result",
                side_effect=sampler,
            ), mock.patch(
                "pokezero.engine_search.world_battle_spec", return_value=world
            ), mock.patch(
                # DISTINCT serialized states per world. Two belief completions
                # that serialize identically are one hypothesis drawn twice and
                # are now collapsed into a single search, which would make this
                # fixture's "two worlds, one aborts" shape unreachable -- both
                # worlds would ride the one report. Production completions differ;
                # only the mock made them identical.
                "pokezero.engine_search.build_poke_engine_state",
                side_effect=lambda *a, _n=itertools.count(), **k: SimpleNamespace(
                    to_string=lambda s=f"S{next(_n)}": s
                ),
            ), patch.dict(
                sys.modules, {"pokezero_search": fake_module}
            ), patch.object(
                EngineMctsPolicy,
                "_advance_live_fold",
                return_value=SimpleNamespace(to_payload=lambda: {}),
            ), patch.object(
                EngineMctsPolicy, "_native", return_value=native
            ), patch.object(
                EngineMctsPolicy, "_validate_model_root_observation", return_value=None
            ), patch.object(
                EngineMctsPolicy, "_root_inputs_json", return_value="{}"
            ):
                decision = policy.select_action_with_context(
                    harness._context(), rng=random.Random(7)
                )

        stats = policy.stats.to_dict()
        # The increment ran: without it these are 0 and the rate is None.
        self.assertEqual(stats["worlds_constructed"], 2)
        self.assertEqual(stats["worlds_searched"], 1)
        self.assertAlmostEqual(stats["world_search_abort_rate"], 0.5)
        self.assertEqual(decision.metadata["engine_mcts"]["worlds_constructed"], 2)

    def test_the_stop_floor_is_clamped_to_the_rung_not_the_total_budget(self) -> None:
        # `early_stop_min_sims` is validated against `search_sims`, which on a ladder
        # cell is the TOTAL for the decision. A floor of 20 against a 100 total split
        # 4 ways is a floor ABOVE the 25-sim rung -- so on the cheap early rungs,
        # exactly where a stop saves the most, the rule could never fire. Found in
        # review; the floor is clamped to the rung's own per-world budget.
        harness = EarlyStopPolicyIntegrationTests()
        policy = harness._policy(early_stop=True)      # min_sims 20, search_sims 100
        policy._ladder_sims_override = 8              # a rung far below the floor
        native = harness._Native(
            [harness._report(4, 4, requested=8, stopped=True)] * 4
        )
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            harness._run(
                policy, native, [harness._world("world-a"), harness._world("world-b")]
            )
        # The floor and its side_key are the last two positionals -- and they are
        # only present at all when the floor is non-zero, so the full-budget REPLAY
        # calls are two positionals shorter. Split on that rather than on call order.
        stopping = [call for call in native.calls if len(call) == 14]
        self.assertTrue(stopping, "the first pass must carry a floor")
        self.assertEqual(
            {call[-2] for call in stopping}, {8},
            "the floor must not exceed the rung it gates",
        )
        self.assertEqual({call[1] for call in stopping}, {8}, "and the rung's sims")

    def test_the_ambiguous_replay_does_not_multiply_by_the_collapse_factor(self) -> None:
        # Review of the F3 fix. The replay loop walks RECORDS, and a collapsed
        # N-group contributes N records -- so replaying N times ALREADY spends the
        # multiplicity. Scaling each replay by it as well cost N^2 x rung_sims:
        # at a 4,096 rung and a 3-group that is 36,864 sims for one group against
        # an intended 12,288, i.e. 2.25x the whole decision's budget, and the
        # oversized trees then fed the saturation test and re-forged the very
        # licence F3 removed.
        harness = EarlyStopPolicyIntegrationTests()
        policy = harness._policy(early_stop=True)
        policy._config = replace(policy._config, worlds=3)
        policy._ladder_sims_override = 40
        policy._ladder_depth_override = 3
        # A COLLAPSED GROUP is the whole point: without one, multiplicity is 1 and
        # `rung_sims * multiplicity == rung_sims`, so the defect is invisible. Two
        # worlds share a belief completion (#1009 searches them once at 2x sims) and
        # a third disagrees, which keeps the cross-world lock ambiguous so the
        # fail-open replay actually fires.
        native = harness._Native([
            harness._report(18, 2, requested=80, stopped=True),   # the 2-group, 2x40
            harness._report(2, 18, requested=40, stopped=True),   # the odd world
            harness._report(30, 10, requested=40, stopped=False),  # 3 replays follow
            harness._report(30, 10, requested=40, stopped=False),
            harness._report(10, 30, requested=40, stopped=False),
        ])
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            harness._run(policy, native, [
                harness._world("dup"), harness._world("dup"), harness._world("other"),
            ])
        self.assertEqual(policy.stats.worlds_collapsed, 1, "the group really collapsed")
        # The REPLAY calls are the ones with no stop floor, so they are two
        # positionals shorter. Three records replay -- the 2-group contributes two --
        # and every one must ask for the rung's budget, NOT 2x it.
        replays = [call for call in native.calls if len(call) == 12]
        self.assertEqual(len(replays), 3)
        self.assertEqual({call[1] for call in replays}, {40},
                         "the rung's sims, never a multiple of them")
        self.assertEqual({call[7] for call in replays}, {3}, "and the rung's depth")
        # The INITIAL search of the group is where the multiplicity belongs, and it
        # still carries it -- the fix removes it from the replay only.
        first = [call for call in native.calls if len(call) == 14]
        self.assertEqual(sorted(call[1] for call in first), [40, 80])

    def test_the_claim_list_covers_every_per_decision_counter_named_as_one(self) -> None:
        # The list is only a generalisation if adding a counter without classifying
        # it FAILS. Every `EngineMctsStats` field whose name declares it counts
        # DECISIONS must be either rewound or deliberately exempt, and the exemptions
        # are named here so a new one cannot be added silently.
        from pokezero.engine_search import LADDER_PER_DECISION_CLAIMS

        exempt = {
            # Gross on purpose: `fallback_rate` nets `ladder_recovered_fallbacks`
            # out, so the taxonomy and the rate stay independently readable.
            "fallback_decisions",
            # The ladder's own accounting, charged once per decision BY the ladder.
            "ladder_decisions", "ladder_unsearched_decisions",
            "ladder_recovered_fallbacks",
            # Charged once per decide(), outside `_search_model` entirely.
            "decisions",
            # Per RUNG by construction and documented as such -- it is the
            # denominator that tells a reader a figure is rung-scoped.
            "searched_decisions",
            # Replay counts, i.e. work: a rung that replayed really did replay.
            "early_stop_full_budget_replays",
            # Charged in `_search` BEFORE the ladder dispatch (engine_search.py:2223,
            # :2225), so they are already once per decision. VERIFIED, not assumed --
            # that is the whole point of this guard.
            "removed_item_decisions", "item_override_decisions",
            # Charged by the BRIDGE, once per decision it submits
            # (foulplay_bridge.py:4804), and never inside a search.
            "oracle_belief_decisions",
        }
        named = {
            f.name for f in dataclasses.fields(EngineMctsStats)
            if f.name.endswith("_decisions") or f.name.endswith("decisions")
        }
        unclassified = named - set(LADDER_PER_DECISION_CLAIMS) - exempt
        self.assertEqual(
            unclassified, set(),
            "a counter of DECISIONS must be rewound to the winning rung or listed "
            "as exempt with a reason; see LADDER_PER_DECISION_CLAIMS",
        )

    def test_the_claim_list_membership_is_pinned_exactly(self) -> None:
        """The classification IS the rule, so the rule is pinned by literal.

        The rewind test above iterates `LADDER_PER_DECISION_CLAIMS`, which makes it
        self-referential: deleting an entry deletes it from the assertion too, and a
        mutant that dropped H2's Q-gap sums survived the whole suite for exactly that
        reason. The guard on `*_decisions` names does not catch it either, because
        `root_q_gap_sum` is not named for what it counts. So the membership is
        asserted as a literal -- shrinking it, or adding a counter without deciding
        which side of the rule it falls on, fails HERE.
        """
        from pokezero.engine_search import LADDER_PER_DECISION_CLAIMS

        self.assertEqual(
            LADDER_PER_DECISION_CLAIMS,
            (
                "override_measured_decisions",
                "model_override_decisions",
                "search_override_unmeasured",
                "root_arm_gap_samples",
                "root_q_gap_sum",
                "root_visit_gap_sum",
                "opponent_top_arm_decisions",
                "opponent_prior_arm_decisions",
                "early_stop_accepted_decisions",
            ),
            "every name here is a claim about ONE decision, charged once per RUNG by "
            "`_search_model`. Adding or removing one is a decision about the rule, "
            "not a refactor -- see the `LADDER_PER_DECISION_CLAIMS` docstring for "
            "what is deliberately excluded and why.",
        )
        # And each one must really exist on the stats object, or the rewind is a
        # silent no-op on a typo.
        fields = {f.name for f in dataclasses.fields(EngineMctsStats)}
        for name in LADDER_PER_DECISION_CLAIMS:
            with self.subTest(claim=name):
                self.assertIn(name, fields)

    def test_a_mutable_claim_cannot_be_put_in_the_scalar_list(self) -> None:
        # The MECHANISM defect behind the surface above, and the reason there are two
        # lists. The generic rewind snapshots with `getattr`, which for a Counter
        # returns a REFERENCE -- so `now - before` compares the object with itself and
        # the rewind is a silent no-op. Review appended a histogram to the scalar
        # tuple: no error, no effect, every test still green. So the scalar list is
        # type-checked, and the mutable ones have their own copy-and-restore path.
        from pokezero.engine_search import (
            LADDER_PER_DECISION_CLAIM_HISTOGRAMS,
            LADDER_PER_DECISION_CLAIMS,
        )

        empty = EngineMctsStats()
        for name in LADDER_PER_DECISION_CLAIMS:
            with self.subTest(claim=name):
                self.assertIsInstance(
                    getattr(empty, name), (int, float),
                    "a mutable claim belongs in LADDER_PER_DECISION_CLAIM_HISTOGRAMS; "
                    "the scalar rewind cannot express it and fails SILENTLY",
                )
        for name in LADDER_PER_DECISION_CLAIM_HISTOGRAMS:
            with self.subTest(histogram=name):
                self.assertIsInstance(getattr(empty, name), Counter)
        self.assertEqual(
            LADDER_PER_DECISION_CLAIM_HISTOGRAMS,
            ("root_q_gap_histogram", "root_visit_gap_histogram"),
            "pinned as a literal for the same reason the scalar list is",
        )

    def test_a_single_axis_cell_is_still_dynamic(self) -> None:
        # `dynamic` is `depth_min is not None OR worlds_min is not None`, and every
        # other test sets BOTH -- so `or` -> `and`, and dropping the worlds term,
        # both survived. Either makes a single-axis cell emit UNSTAMPED dynamic rows,
        # which the deploy analyzer then refuses. `_budget_rungs` fully supports
        # single-axis, so this is reachable through a cell key today.
        for kwargs in ({"depth_min": 2}, {"worlds_min": 1}):
            with self.subTest(**kwargs):
                harness = EarlyStopPolicyIntegrationTests()
                policy = harness._policy(early_stop=False)
                policy._config = replace(policy._config, search_depth=3,
                                         search_sims=100, **kwargs)
                policy.stats = EngineMctsStats()
                policy.policy_id = "engine-mcts"
                # The per-battle adaptive state, which `_search_ladder` resets on a new
                # battle_id but does not create from nothing.
                policy._ladder_battle = None
                policy._ladder_worlds = None
                policy._ladder_depth = None
                policy._ladder_depth_ceiling = {}
                policy._ladder_probing = False
                policy._ladder_pending_addresses = None

                from pokezero.policy import PolicyDecision

                def _fake(context, worlds, live_fold, rng, _p=policy):
                    _p.stats.root_decision_rows.append({"m": 0})
                    _p._ladder_saturated = False
                    _p._ladder_worlds_agree = True
                    return PolicyDecision(action_index=0, policy_id=_p.policy_id)

                policy._search_model = _fake
                policy._search_ladder(object(), [(object(), object()) for _ in range(2)],
                                      object(), random.Random(0))
                self.assertTrue(policy.stats.to_dict()["ladder_dynamic"])
                self.assertIn("ladder_superseded", policy.stats.root_decision_rows[0])

    def test_work_counters_are_deliberately_not_rewound(self) -> None:
        # The other half of the rule, or "rewind everything" would pass the test
        # above. A rung really did its iterations and its wall, and a cost analysis
        # needs the sum -- `ladder_rungs_per_decision` is how a reader divides it.
        from pokezero.engine_search import LADDER_PER_DECISION_CLAIMS

        for name in ("total_iterations", "model_evals", "search_wall_seconds",
                     "depth_reached_samples", "worlds_searched",
                     "world_search_attempts", "fallback_decisions"):
            with self.subTest(counter=name):
                self.assertNotIn(name, LADDER_PER_DECISION_CLAIMS)

    def test_a_cell_that_barely_searched_also_reports_no_wall(self) -> None:
        # The 99.99% case, which gating on `searched_decisions > 0` left open: review
        # measured ONE searched rung against 10,000 ladder decisions reporting a
        # 0.3 ms wall and passing a 20 s cap -- the same "unevaluable read as a pass"
        # the 100% case was fixed for, one decision short of it. The gate is COVERAGE:
        # most decisions the engine was asked to make must have reached a search.
        stats = EngineMctsStats()
        stats.decisions = stats.ladder_decisions = 10000
        stats.ladder_unsearched_decisions = 9999
        stats.searched_decisions = 1
        stats.search_wall_seconds = 3.0
        self.assertNotIn("search_wall_per_ladder_decision", stats.to_dict())
        # At full coverage it is emitted, or the gate would block every real cell.
        stats.ladder_unsearched_decisions = 0
        self.assertAlmostEqual(
            stats.to_dict()["search_wall_per_ladder_decision"], 0.0003
        )
        # And a 5% fallback rate -- realistic, and inside the 2% health gate's
        # neighbourhood -- must still report, or the gate is unusable.
        stats.ladder_unsearched_decisions = 500
        self.assertIn("search_wall_per_ladder_decision", stats.to_dict())

    def test_a_ladder_cell_that_never_searched_reports_no_wall_at_all(self) -> None:
        # `ladder_decisions` is charged BEFORE the first rung runs, so a cell whose
        # every decision fell back at rung 0 still had a non-zero denominator and
        # emitted a wall -- the FALLBACK wall -- which the power report's cap read as
        # a PASS at 0.30s. A cell whose latency is entirely unknown must read
        # UNEVALUABLE. Found in review.
        stats = EngineMctsStats()
        stats.decisions = stats.ladder_decisions = 1000
        stats.searched_decisions = 0
        stats.search_wall_seconds = 300.0
        payload = stats.to_dict()
        self.assertNotIn("search_wall_per_ladder_decision", payload)
        self.assertNotIn("search_wall_per_searched_decision", payload)
        # And it reappears the moment anything was actually searched.
        stats.searched_decisions = 2000
        self.assertIn("search_wall_per_ladder_decision", stats.to_dict())

    def test_the_stop_floor_is_untouched_when_there_is_no_rung(self) -> None:
        # And a fixed cell keeps the configured floor exactly, so the clamp cannot
        # change what a banked cell measured.
        harness = EarlyStopPolicyIntegrationTests()
        policy = harness._policy(early_stop=True)
        native = harness._Native([harness._report(56, 4, stopped=False)] * 2)
        harness._run(
            policy, native, [harness._world("world-a"), harness._world("world-b")]
        )
        self.assertEqual({call[-2] for call in native.calls}, {20})

    def test_a_healthy_decision_reports_a_zero_abort_rate(self) -> None:
        harness = EarlyStopPolicyIntegrationTests()
        policy = harness._policy(early_stop=False)
        # Both denominators, because they count different things: constructions
        # (once per decision) and SEARCH attempts (once per world per rung). This
        # test enters at `_search_model`, past the dispatch point that charges them.
        policy.stats.worlds_constructed = 2
        policy.stats.world_search_attempts = 2
        native = harness._Native(
            [
                harness._report(56, 4, stopped=False),
                harness._report(56, 4, stopped=False),
            ]
        )
        harness._run(
            policy, native, [harness._world("world-a"), harness._world("world-b")]
        )
        stats = policy.stats.to_dict()
        self.assertEqual(stats["worlds_searched"], 2)
        self.assertEqual(stats["world_search_abort_rate"], 0.0)


class FallbackAddressTests(unittest.TestCase):
    """A fallback must be REPLAYABLE, not just counted.

    Era 57 recorded 7,498 fallbacks and left no way to reproduce a single one: the
    addresses existed only in pod logs, and deleting the Jobs deleted them.
    `battle_id` carries the seed, so (battle_id, round, seat) is a complete address
    for one turn.
    """

    def _policy(self, **cfg):
        import unittest.mock as mock

        policy = EngineMctsPolicy(
            dex=None, set_source=None, module=mock.Mock(),
            config=EngineMctsConfig(**cfg),
        )
        return policy

    def _fallback(self, policy, *, battle="b-8220001", rnd=7, seat="p1",
                  reason="crate_search_failed", classes=()):
        ctx = SimpleNamespace(
            observation=_FakeObservation(
                (True, True, False, False, False, False, False, False, False),
                _candidates(),
            ),
            public_materialization_state=object(),
            player_id=seat,
            battle_id=battle,
            decision_round_index=rnd,
        )
        policy._world_failures_before = {}
        for cls in classes:
            policy.stats.world_failure_reasons[cls] += 1
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            policy._fallback(ctx, random.Random(0), reason)

    def test_the_address_identifies_one_turn(self) -> None:
        p = self._policy()
        self._fallback(p, battle="battle-gen3randombattle-controlled-8220024",
                       rnd=8, seat="p1", classes=("crate_search: sleeptalk",))
        samples = p.stats.to_dict()["fallback_samples"]
        self.assertIn("crate_search: sleeptalk", samples)
        entry = samples["crate_search: sleeptalk"][0]
        # The seed is IN the battle id, which is what makes this replayable.
        self.assertIn("8220024", entry["battle_id"])
        self.assertEqual(entry["round"], 8)
        self.assertEqual(entry["seat"], "p1")
        self.assertEqual(entry["reason"], "crate_search_failed")

    def test_a_rare_class_still_gets_an_address_under_a_dominant_one(self) -> None:
        """The reason the cap is PER CLASS.

        Era 57 was 49.5% one reason. A global cap fills with that class and the
        rare classes -- the ones you actually need an address for -- get none.
        """
        p = self._policy()
        for i in range(500):
            self._fallback(p, battle=f"b-{i}", rnd=i, classes=("dominant",))
        self._fallback(p, battle="b-rare", rnd=99, classes=("rare_class",))
        samples = p.stats.to_dict()["fallback_samples"]
        self.assertIn("rare_class", samples)
        self.assertEqual(samples["rare_class"][0]["battle_id"], "b-rare")

    def test_the_sample_is_bounded_per_class(self) -> None:
        p = self._policy()
        for i in range(50):
            self._fallback(p, battle=f"b-{i}", rnd=i, classes=("one",))
        self.assertEqual(len(p.stats.to_dict()["fallback_samples"]["one"]), 3)

    def test_a_decision_failing_on_several_classes_is_addressable_under_each(self) -> None:
        p = self._policy()
        self._fallback(p, classes=("class_a", "class_b"))
        samples = p.stats.to_dict()["fallback_samples"]
        self.assertIn("class_a", samples)
        self.assertIn("class_b", samples)

    def test_a_fallback_with_no_world_failures_is_still_addressable(self) -> None:
        """no_public_state / live_fold_broken produce no world-failure classes.

        Keying only by world-failure class would leave them unaddressable.
        """
        p = self._policy()
        self._fallback(p, reason="no_public_state", classes=())
        samples = p.stats.to_dict()["fallback_samples"]
        self.assertIn("fallback:no_public_state", samples)
        self.assertEqual(samples["fallback:no_public_state"][0]["reason"],
                         "no_public_state")

    def test_a_class_that_stops_refusing_does_not_stop_being_counted(self) -> None:
        """The whole reason `lossy_subcase_renders` exists.

        `sleeptalk_called_unidentified:ambiguous` appeared in `world_failure_reasons` only
        BECAUSE it aborted the world. Making it usable therefore deletes the one number
        that tracked it -- and this campaign spent two eras unable to say what had changed
        inside a class, which is exactly that failure. So the usable arm is counted on its
        own channel and that channel reaches the shard report.
        """
        stats = EngineMctsStats()
        stats.lossy_subcase_renders["sleeptalk_called_unidentified:ambiguous"] += 7
        stats.lossy_subcase_renders[
            "sleeptalk_called_unidentified:ambiguous_unrenderable"
        ] += 2
        emitted = stats.to_dict()["lossy_subcase_renders"]
        self.assertEqual(emitted["sleeptalk_called_unidentified:ambiguous"], 7)
        self.assertEqual(
            emitted["sleeptalk_called_unidentified:ambiguous_unrenderable"], 2
        )
        # And it must be a real dict in the report, not a Counter that a json.dumps
        # elsewhere might refuse or render differently.
        self.assertIsInstance(emitted, dict)
        import json as _json

        _json.dumps(stats.to_dict())

    def test_the_crate_report_lossy_subcases_reach_the_stats(self) -> None:
        """The seam, not just the container.

        The crate emits `lossy_subcases` as a sub-case -> count object in its search
        report. If nothing reads it, the counter above stays empty forever and the class is
        invisible in exactly the way this test exists to prevent.
        """
        import unittest.mock as mock

        policy = EngineMctsPolicy(
            dex=None, set_source=None, module=mock.Mock(), config=EngineMctsConfig()
        )
        policy._absorb_lossy_subcases(
            {
                "lossy_subcases": {
                    "sleeptalk_called_unidentified:ambiguous": 5,
                    "attract_immobilization_source_unknown": 1,
                },
            }
        )
        self.assertEqual(
            policy.stats.lossy_subcase_renders[
                "sleeptalk_called_unidentified:ambiguous"
            ],
            5,
        )
        self.assertEqual(
            policy.stats.lossy_subcase_renders["attract_immobilization_source_unknown"], 1
        )
        # An absent key must be a no-op, not a crash: older images emit no such field.
        policy._absorb_lossy_subcases({})
        self.assertEqual(
            policy.stats.lossy_subcase_renders[
                "sleeptalk_called_unidentified:ambiguous"
            ],
            5,
        )

    def test_the_crate_and_python_agree_on_the_lossy_subcases_key(self) -> None:
        """A rename on either side zeroes the class SILENTLY, which is the failure mode.

        `lossy_subcase_renders` is fed from one JSON key that the Rust search report emits
        and this module reads. Nothing else connects them: if either side is renamed the
        counter stays at zero forever, no test fails, and the class becomes invisible --
        precisely the outcome this counter was added to prevent. So the two spellings are
        asserted against each other.
        """
        repo = pathlib.Path(__file__).resolve().parent.parent
        model_rs = (repo / "rust" / "pokezero-search" / "src" / "model.rs").read_text()
        engine_py = (repo / "src" / "pokezero" / "engine_search.py").read_text()
        self.assertIn(
            '\\"lossy_subcases\\":{}', model_rs,
            "the crate no longer emits a `lossy_subcases` object in its search report; "
            "the Python counter below reads that exact key and would silently stay zero",
        )
        self.assertIn(
            'report.get("lossy_subcases")', engine_py,
            "engine_search no longer reads `lossy_subcases`, so the class is invisible",
        )

    def test_the_crate_and_python_agree_on_the_abort_payload_attribute(self) -> None:
        """The ABORT arm's spelling, pinned exactly like the report key's above.

        The report key covers only worlds whose search COMPLETED. Aborts carry their
        counts on an exception ATTRIBUTE instead; that name is a
        second, independent cross-language contract with the identical failure mode -- a
        rename on either side leaves the abort arm reading zero forever, no test fails,
        and the class goes back to describing the clean subset only.
        """
        repo = pathlib.Path(__file__).resolve().parent.parent
        abort_rs = (
            repo / "rust" / "pokezero-search" / "src" / "abort_telemetry.rs"
        ).read_text()
        self.assertIn(
            f'ABORT_PAYLOAD_ATTR: &str = "{_ABORT_LOSSY_SUBCASES_ATTR}"',
            abort_rs,
            "the crate attaches its abort payload under a different attribute name than "
            "engine_search reads, so every aborted world's counts are dropped silently",
        )
        # Both ends of the wire, so a half-applied removal cannot go green. The crate's
        # own suite pins the Rust side in far more detail
        # (`the_search_path_records_into_the_ledger_and_attaches_it_on_abort`); this is
        # the arm that fires when someone edits only Python, or only Rust, and runs only
        # the other language's tests.
        model_rs = (repo / "rust" / "pokezero-search" / "src" / "model.rs").read_text()
        engine_py = (repo / "src" / "pokezero" / "engine_search.py").read_text()
        self.assertIn(
            "abort_telemetry::guarded_search_with_ledger(", model_rs,
            "the crate no longer runs its search through the guard that attaches the "
            "ledger to an aborting error, so aborted worlds report nothing",
        )
        self.assertIn(
            "_absorb_aborted_lossy_subcases(error)", engine_py,
            "engine_search no longer reads the abort payload at the failure seam",
        )
        # And the counts must never be smuggled into the message: that string is the
        # world_failure_reasons key, whose bytes are compared across eras and which
        # _bounded_reason_detail truncates at 512 chars. Pinned as "exactly one call
        # site" rather than one forbidden spelling -- `!contains("json_object()}")`
        # caught a single format-string shape and sailed past
        # `format!("{} [lossy={}]", raw, ..json_object())`.
        self.assertEqual(
            model_rs.count("json_object()"), 1,
            "`json_object()` must have exactly one call site in model.rs (the search "
            "report); a second one is how the counts reach the reason key",
        )

    def test_the_subcase_key_is_present_in_the_report_even_when_empty(self) -> None:
        """Key-absent and value-zero must stay distinguishable.

        A missing key reads as a genuine zero to every downstream consumer, so "the crate
        emitted nothing" and "this build has no such counter" would collapse onto the same
        reading. That distinction was established deliberately and is relied on -- and the
        abort arm makes it load-bearing again, because an abort that observed nothing still
        attaches an EMPTY object rather than no attribute.
        """
        emitted = EngineMctsStats().to_dict()
        self.assertIn("lossy_subcase_renders", emitted)
        self.assertEqual(emitted["lossy_subcase_renders"], {})
        self.assertIsInstance(emitted["lossy_subcase_renders"], dict)

    def test_no_fallbacks_means_no_samples_not_a_stub(self) -> None:
        self.assertEqual(EngineMctsStats().to_dict()["fallback_samples"], {})

    def test_the_addresses_for_a_class_come_from_different_battles(self) -> None:
        """Three views of one incident cannot tell you whether a class generalises.

        A refusal cause typically closes worlds for the rest of the battle it appears
        in, so first-3 retention hands back rounds N, N+1, N+2 of a single battle.
        """
        p = self._policy()
        for rnd in range(30):
            self._fallback(p, battle="b-one", rnd=rnd, classes=("sticky",))
        for rnd in range(30):
            self._fallback(p, battle="b-two", rnd=rnd, classes=("sticky",))

        bucket = p.stats.to_dict()["fallback_samples"]["sticky"]
        self.assertEqual([e["battle_id"] for e in bucket], ["b-one", "b-two"])
        # A class confined to one battle still keeps an address -- replay needs one.
        q = self._policy()
        self._fallback(q, battle="b-solo", classes=("lonely",))
        self._fallback(q, battle="b-solo", rnd=99, classes=("lonely",))
        self.assertEqual(len(q.stats.fallback_samples["lonely"]), 1)

    def test_the_key_ceiling_is_enforced_and_announces_what_it_dropped(self) -> None:
        """An incomplete sample that looks complete is how a coverage claim goes wrong.

        Reason keys interpolate operands (species, turn numbers, HP values), so the key
        space is data-dependent. The ceiling makes the report size unconditional; the
        counter makes the truncation visible.
        """
        p = self._policy()
        for i in range(_FALLBACK_SAMPLE_KEY_CEILING + 25):
            self._fallback(p, battle=f"b-{i}", classes=(f"minted_class_{i}",))

        stats = p.stats.to_dict()
        self.assertEqual(len(stats["fallback_samples"]), _FALLBACK_SAMPLE_KEY_CEILING)
        self.assertGreater(stats["fallback_sample_addresses_dropped"], 0)
        # Entries stay bounded by ceiling x per-class cap, so the report cannot grow
        # without limit no matter how many classes the data mints.
        total = sum(len(v) for v in stats["fallback_samples"].values())
        self.assertLessEqual(
            total, _FALLBACK_SAMPLE_KEY_CEILING * _FALLBACK_SAMPLES_PER_CLASS
        )

    def test_a_rare_reason_is_addressable_even_past_the_class_ceiling(self) -> None:
        """The ceiling must not reintroduce the bug it was added beside.

        The class ceiling exists to bound an unbounded class space. Applying it to
        reason keys too meant that past 256 classes a rare reason lost its key and
        became unaddressable -- and since classes were iterated first they claimed the
        last slot, so the reason key was precisely what got dropped. `live_fold_broken`
        is the worst case: it has no world failures, so the reason key is its ONLY key.
        """
        p = self._policy()
        for i in range(_FALLBACK_SAMPLE_KEY_CEILING + 50):
            self._fallback(p, battle=f"b-fill-{i}", classes=(f"filler_{i}",))
        self.assertEqual(len(p.stats.fallback_samples),
                         _FALLBACK_SAMPLE_KEY_CEILING)  # ceiling is really full

        self._fallback(p, battle="b-rare", rnd=5, reason="choices_unmapped",
                       classes=("filler_0",))
        self._fallback(p, battle="b-fold", rnd=6, reason="live_fold_broken",
                       classes=())

        samples = p.stats.to_dict()["fallback_samples"]
        self.assertIn("fallback:choices_unmapped", samples)
        self.assertIn("fallback:live_fold_broken", samples)
        # The invariant from the sibling test, re-asserted PAST the ceiling -- that is
        # the only place it catches this.
        self.assertEqual(
            {k.split("fallback:")[-1] for k in samples if k.startswith("fallback:")},
            set(p.stats.fallback_reasons),
        )
        self.assertEqual(samples["fallback:live_fold_broken"][0]["battle_id"], "b-fold")

    def test_the_dropped_counter_counts_addresses_not_classes(self) -> None:
        """Named for the unit it measures.

        It increments per dropped OCCURRENCE. Read as a class count it says 1,001 where
        two classes were lost, which is the cite-a-number-for-a-different-quantity
        mistake this store's own comments were twice corrected for.
        """
        p = self._policy()
        for i in range(_FALLBACK_SAMPLE_KEY_CEILING):
            self._fallback(p, battle=f"b-fill-{i}", classes=(f"filler_{i}",))
        for i in range(200):
            self._fallback(p, battle=f"b-over-{i}", classes=("overflow_a", "overflow_b"))

        lost = {c for c in p.stats.world_failure_reasons
                if c not in p.stats.fallback_samples}
        self.assertEqual(len(lost), 3)  # filler_255, overflow_a, overflow_b
        # Measured 601 addresses against those 3 classes -- a 200x gap. Asserted as a
        # ratio rather than the literal, because the literal depends on this helper's
        # cumulative delta and the CLAIM is about the unit, not the arithmetic. My first
        # draft asserted 400 from hand-arithmetic and was simply wrong.
        self.assertGreater(p.stats.fallback_sample_addresses_dropped, 100 * len(lost))
        self.assertNotIn("overflow_a", p.stats.fallback_samples)

    def test_a_missing_round_index_is_null_not_a_question_mark(self) -> None:
        """A replay harness doing int(entry["round"]) must get a null, not ValueError."""
        import unittest.mock as mock

        p = self._policy()
        ctx = SimpleNamespace(
            observation=_FakeObservation(
                (True, True, False, False, False, False, False, False, False),
                _candidates(),
            ),
            public_materialization_state=object(),
            player_id="p1",
            battle_id="b-noround",
        )  # NO decision_round_index
        p._world_failures_before = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p._fallback(ctx, random.Random(0), "crate_search_failed")
        entry = p.stats.to_dict()["fallback_samples"]["fallback:crate_search_failed"][0]
        self.assertIsNone(entry["round"])

    def test_a_rare_reason_survives_a_dominant_class_it_co_occurs_with(self) -> None:
        """The per-class cap protects rare CLASSES. Reasons are a separate axis.

        `choices_unmapped` co-occurs with world failures, so keying only off the
        per-decision delta files its single address under a class whose three slots
        the dominant reason already filled -- and the rare reason gets no address at
        all. That is era 57's failure mode on the other axis, so the reason key is
        minted unconditionally rather than only when the delta is empty.
        """
        p = self._policy()
        for i in range(50):
            self._fallback(p, battle=f"b-{i}", rnd=i,
                           reason="crate_search_failed", classes=("dominant",))
        self.assertEqual(len(p.stats.fallback_samples["dominant"]),
                         _FALLBACK_SAMPLES_PER_CLASS)  # slots are genuinely full
        self._fallback(p, battle="b-9999001", rnd=41, seat="p2",
                       reason="choices_unmapped", classes=("dominant",))

        samples = p.stats.to_dict()["fallback_samples"]
        self.assertIn("fallback:choices_unmapped", samples)
        entry = samples["fallback:choices_unmapped"][0]
        self.assertEqual(
            (entry["battle_id"], entry["round"], entry["seat"]),
            ("b-9999001", 41, "p2"),
        )
        # Every reason the counter saw must be addressable, or the store is
        # reporting a count it cannot reproduce.
        self.assertEqual(
            {k.split("fallback:")[-1] for k in samples if k.startswith("fallback:")},
            set(p.stats.fallback_reasons),
        )

    def test_the_address_is_the_failing_decision_not_every_class_seen_so_far(self) -> None:
        """Pins the per-decision delta through the REAL seam, not a hand-set dict.

        The other tests set `_world_failures_before` themselves, so they pass just as
        well if the delta becomes cumulative. Cumulative would file each fallback
        under every class seen so far in the shard, consuming the three slots with
        addresses that did not fail on that class -- silently WRONG addresses, worse
        than none. So drive `select_action_with_context` twice and require the second
        fallback not to be filed under the first decision's class.
        """
        import unittest.mock as mock

        p = self._policy()
        ctx = SimpleNamespace(
            observation=_FakeObservation(
                (True, True, False, False, False, False, False, False, False),
                _candidates(),
            ),
            public_materialization_state=None,  # forces reason=no_public_state
            player_id="p1",
            battle_id="b-8220002",
            decision_round_index=3,
        )
        p.stats.world_failure_reasons["stale_earlier_decision_class"] += 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p.select_action_with_context(ctx, rng=random.Random(0))

        samples = p.stats.to_dict()["fallback_samples"]
        self.assertNotIn("stale_earlier_decision_class", samples)
        self.assertIn("fallback:no_public_state", samples)


class WorldCollapseTest(unittest.TestCase):
    """Duplicate belief completions must be CONCENTRATED, not skipped.

    Repeated draws of one completion are not redundant work: the per-world seed
    drives chance-node sampling, so each is an independent Monte-Carlo estimate
    and averaging them reduces that completion's variance. Searching once and
    reusing the answer stays unbiased but throws the extra samples away. So the
    duplicates are folded into ONE search at N x the sim budget instead.

    These drive the real `_search_model` through the fake native. An earlier
    version of this class reimplemented the production loop inline, and four
    mutants -- including "disable the memo entirely" -- passed all of it.
    """

    # Reuse the existing end-to-end harness WITHOUT subclassing it: subclassing a
    # TestCase re-runs every one of its tests under this name too.
    _Native = EarlyStopPolicyIntegrationTests._Native
    # staticmethod(): a bare assignment rebinds these as instance methods, so
    # `self` would arrive as the first positional argument.
    _report = staticmethod(EarlyStopPolicyIntegrationTests._report)
    _world = staticmethod(EarlyStopPolicyIntegrationTests._world)
    _context = staticmethod(EarlyStopPolicyIntegrationTests._context)

    def _policy(self, **kw):
        return EarlyStopPolicyIntegrationTests._policy(self, **kw)

    def _run(self, policy, native, worlds):
        return EarlyStopPolicyIntegrationTests._run(self, policy, native, worlds)

    def test_duplicates_are_searched_once(self) -> None:
        policy = self._policy(early_stop=False)
        native = self._Native([self._report(60, 40, stopped=False)])
        worlds = [self._world("W"), self._world("W")]
        self._run(policy, native, worlds)
        self.assertEqual(len(native.calls), 1, "two draws of one completion = one search")
        self.assertEqual(policy.stats.worlds_collapsed, 1)
        self.assertEqual(policy.stats.unique_worlds_searched, 1)

    def test_the_duplicate_budget_is_concentrated_not_discarded(self) -> None:
        """N draws buy N x the sims on one tree -- same compute, deeper search."""
        policy = self._policy(early_stop=False)
        native = self._Native([self._report(60, 40, stopped=False)])
        base = policy._config.search_sims
        self._run(policy, native, [self._world("W")] * 3)
        self.assertEqual(len(native.calls), 1)
        # arg 1 of search_batched_multi_encoded is the iteration count
        self.assertEqual(native.calls[0][1], base * 3)

    def test_distinct_worlds_are_still_searched_separately(self) -> None:
        policy = self._policy(early_stop=False)
        native = self._Native([
            self._report(60, 40, stopped=False),
            self._report(10, 90, stopped=False),
        ])
        self._run(policy, native, [self._world("A"), self._world("B")])
        self.assertEqual(len(native.calls), 2)
        self.assertEqual(policy.stats.worlds_collapsed, 0)
        self.assertEqual(policy.stats.unique_worlds_searched, 2)
        self.assertEqual(native.calls[0][1], policy._config.search_sims)

    def test_a_single_world_is_unchanged(self) -> None:
        policy = self._policy(early_stop=False)
        native = self._Native([self._report(60, 40, stopped=False)])
        self._run(policy, native, [self._world("W")])
        self.assertEqual(len(native.calls), 1)
        self.assertEqual(native.calls[0][1], policy._config.search_sims)
        self.assertEqual(policy.stats.worlds_collapsed, 0)

    def test_every_draw_still_contributes_its_weight(self) -> None:
        """The belief must not be flattened.

        Completion A drawn 3x, completion B drawn 1x. Aggregation weights every
        RECORD equally, so folding the duplicates into one record would turn a
        3:1 belief into 1:1. Three records must reach the aggregator.
        """
        policy = self._policy(early_stop=False)
        native = self._Native([
            self._report(100, 0, stopped=False),
            self._report(0, 100, stopped=False),
        ])
        worlds = [self._world("A"), self._world("A"), self._world("A"), self._world("B")]
        decision = self._run(policy, native, worlds)
        self.assertEqual(len(native.calls), 2)
        self.assertEqual(policy.stats.worlds_collapsed, 2)
        self.assertEqual(policy.stats.unique_worlds_searched, 2)

        # THE ASSERTION THAT MATTERS, and the one this test was missing: look at
        # the AGGREGATE, not the counters. Review showed that replacing
        # `for record in records:` with `records[:1]` -- appending one record per
        # group instead of N, i.e. exactly the belief flattening this change is
        # built to avoid -- passed all 112 tests. A 3:1 belief silently became
        # 1:1 and nothing noticed.
        meta = decision.metadata["engine_mcts"]
        self.assertEqual(meta["worlds_searched"], 4, "all four DRAWS must reach the aggregator")
        self.assertEqual(
            meta["aggregated_choices"],
            {"alpha": 3.0, "beta": 1.0},
            "a 3:1 belief must aggregate 3:1, not be flattened to 1:1",
        )

    def test_reports_are_not_shared_between_twins(self) -> None:
        policy = self._policy(early_stop=False)
        native = self._Native([self._report(60, 40, stopped=False)])
        self._run(policy, native, [self._world("W"), self._world("W")])
        # One search, so any aliasing would be between the twins' records.
        self.assertEqual(len(native.calls), 1)

    def test_early_stop_savings_are_not_multiplied_by_duplicates(self) -> None:
        """A stopped search must be counted once, not once per duplicate draw.

        Found by independent review and measured: three identical worlds with a
        locked early stop reported 120 simulations_saved where one search had
        actually saved 40, inflating the headline metric of a DIFFERENT feature
        by the collapse multiplicity.
        """
        policy = self._policy(early_stop=True)
        native = self._Native([self._report(60, 40, requested=100, stopped=True)])
        self._run(policy, native, [self._world("W")] * 3)
        self.assertEqual(len(native.calls), 1, "one search for three identical draws")
        # The SAVINGS are per search (one search saved 40). The WORLD COUNTER is
        # per world (three worlds stopped). Conflating them is what produced both
        # the 120-vs-40 over-count and, after the first fix, a 1-vs-3
        # under-count in the counter whose name says "worlds".
        self.assertEqual(
            policy.stats.early_stop_triggered_worlds, 3,
            "three WORLDS stopped, even though one search ran",
        )

    def test_depth_samples_stay_per_world_not_per_search(self) -> None:
        """The depth ladder reads this counter; collapsing must not redefine it.

        Concentration searches one tree per unique completion, so counting a
        depth sample per SEARCH would silently turn a per-world counter into a
        per-search one -- making the ladder incomparable to every historical run.
        Found by review; the mutant that reverts it used to survive.
        """
        policy = self._policy(early_stop=False)
        deep, shallow = self._report(60, 40, stopped=False), self._report(60, 40, stopped=False)
        # DISTINCT depths: with both at the same value the weight could attach to
        # the wrong sample and nothing would notice.
        deep["max_depth_reached"], shallow["max_depth_reached"] = 5, 2
        native = self._Native([deep, shallow])
        # 3 draws of A, 1 of B -> 2 searches, but FOUR worlds.
        self._run(policy, native, [self._world("A")] * 3 + [self._world("B")])
        self.assertEqual(len(native.calls), 2, "two unique completions")
        self.assertEqual(
            policy.stats.depth_reached_samples, 4,
            "one sample per WORLD (4), not per search (2)",
        )
        self.assertEqual(sum(policy.stats.depth_reached_histogram.values()), 4)
        # The SUM, not just the count: dropping the weight here left samples and
        # the histogram correct while the MEAN was 2.4x wrong, undetected.
        self.assertEqual(
            policy.stats.depth_reached_sum, 5 * 3 + 2 * 1,
            "the weight must reach the sum, or the ladder mean is wrong",
        )
        self.assertEqual(policy.stats.depth_reached_histogram[5], 3)
        self.assertEqual(policy.stats.depth_reached_histogram[2], 1)

    def test_the_early_stop_replay_does_not_multiply_depth_samples(self) -> None:
        """The replay path must weigh 1, not the group multiplicity.

        Found by review, and a bug my own per-world fix introduced: the replay
        reuses each stopped RECORD, so a weight smuggled through the record made
        every replay of an N-group add N again -- 12 samples where 6 was right,
        and a skewed mean, not merely a skewed count.
        """
        policy = self._policy(early_stop=True)
        # Ambiguous stop -> each stopped world is replayed at full budget.
        first = self._report(30, 30, requested=100, stopped=True)
        first["max_depth_reached"] = 2
        replays = []
        for _ in range(3):
            r = self._report(60, 40, requested=100, stopped=False)
            r["max_depth_reached"] = 2
            replays.append(r)
        native = self._Native([first] + replays)
        self._run(policy, native, [self._world("W")] * 3)
        self.assertEqual(
            policy.stats.depth_reached_samples, 6,
            "3 worlds searched once + 3 replays = 6, not 3 + 3x3 = 12",
        )

    def test_worlds_stopped_counts_worlds_like_its_name_and_its_sibling(self) -> None:
        """`worlds_stopped` and `full_budget_replays` must share a denominator.

        Deduping the early-stop SAVINGS was the real fix (120 reported where 40
        was true), but applying the same dedupe to `worlds_stopped` put it on a
        different denominator from `full_budget_replays` for the same event.
        """
        policy = self._policy(early_stop=True)
        # Decisive split so the aggregate LOCKS (leader - runner_up > remaining)
        # and no full-budget replay follows: 58 - 2 = 56 > 40 remaining.
        native = self._Native([self._report(58, 2, requested=100, stopped=True)])
        decision = self._run(policy, native, [self._world("W")] * 3)
        self.assertEqual(len(native.calls), 1, "one search for three identical draws")
        early = decision.metadata["engine_mcts"]["early_stop"]
        self.assertEqual(early["worlds_stopped"], 3, "three WORLDS stopped, one search")
        self.assertEqual(
            policy.stats.early_stop_sims_saved, 40,
            "savings are per SEARCH -- 40, not 3x40",
        )
        self.assertEqual(
            policy.stats.early_stop_triggered_worlds, 3,
            "the shard counter says WORLDS; it must not under-report by the "
            "collapse multiplicity -- the mirror of the 120-vs-40 over-count",
        )
        self.assertEqual(early["full_budget_replays"], 0, "a locked stop needs no replay")

    def test_telemetry_exposes_both_counters(self) -> None:
        from pokezero.engine_search import EngineMctsStats

        payload = EngineMctsStats().to_dict()
        self.assertIn("worlds_collapsed", payload)
        self.assertIn("unique_worlds_searched", payload)

    # --- review round 1 -------------------------------------------------------
    # Both of the following were measured against the pre-fix branch, not
    # imagined: an aborting 3-group reported `worlds_searched - worlds_collapsed`
    # = -2, and recorded its abort ONCE where main records it three times.

    def _abort(self):
        return ValueError(
            "attribution-unsafe renderer branch rejected before tree/model fold: sleeptalk"
        )

    def test_an_aborting_group_counts_its_refusal_once_per_WORLD(self) -> None:
        """`world_failure_reasons` must stay on the same unit as its denominator.

        A refusal is deterministic in the state, so every draw of an aborting
        completion aborts -- an aborting group is the COMMON case for
        duplicates, not a corner one. `worlds_attempted` still counts draws, so
        counting the abort once per SEARCH would deflate this counter by the
        collapse multiplicity while its denominator kept the old unit. The
        fallback-burndown campaign ranks classes on these counts across eras.
        """
        policy = self._policy(early_stop=False)
        native = self._Native([self._abort()])
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            self._run(policy, native, [self._world("W")] * 3)
        self.assertEqual(len(native.calls), 1, "one search for three identical draws")
        key = next(iter(policy.stats.world_failure_reasons))
        self.assertEqual(
            policy.stats.world_failure_reasons[key], 3,
            "three WORLD draws aborted; collapsing the search must not collapse "
            "the refusal count away from its per-draw denominator",
        )
        self.assertEqual(
            policy.stats.attribution_unsafe_renders, 3,
            "the abort-channel counter rides the same unit",
        )

    def test_a_failed_group_does_not_break_the_collapse_identity(self) -> None:
        """`worlds_searched - worlds_collapsed == unique_worlds_searched`, always.

        Incrementing `worlds_collapsed` before the search -- where the sim
        scaling is decided -- made this go NEGATIVE when the group aborted, since
        a failing group contributes no records to `worlds_searched`. Measured at
        -2 on the pre-fix branch.
        """
        policy = self._policy(early_stop=False)
        native = self._Native([self._report(60, 40, stopped=False), self._abort()])
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            self._run(policy, native, [self._world("A")] * 3 + [self._world("B")] * 2)
        stats = policy.stats
        self.assertEqual(len(native.calls), 2, "one search per unique completion")
        self.assertEqual(stats.worlds_searched, 3, "only A's three draws aggregate")
        self.assertEqual(
            stats.worlds_searched - stats.worlds_collapsed,
            stats.unique_worlds_searched,
            "the documented identity must hold when a collapsed group aborts",
        )
        self.assertEqual(stats.unique_worlds_searched, 1, "only A returned a report")
        self.assertGreaterEqual(
            stats.worlds_searched - stats.worlds_collapsed, 0,
            "a search count can never be negative",
        )


class RootDecisionTelemetryTest(unittest.TestCase):
    """The override / Q-gap / opponent-arm counters, on constructed reports.

    Every case here is a report whose ANSWER IS ARITHMETIC: the visit block and
    the prior block are written to disagree (or agree) by construction, so a
    counter that fires when they agree, or fails to when they differ, is a wrong
    number and not a judgement call. That is the bar this file's other telemetry
    is held to (`test_depth_reached_is_measured_not_the_configured_cap`), and the
    reason the honesty counter gets four cases of its own: an unmeasurable
    decision silently booked as agreement is indistinguishable, in a shard, from
    a search that never overrides the model.

    These drive the real `_search_model` through the fake native, so the counting
    site, the aggregation and the choice translation are all the production ones.
    """

    _Native = EarlyStopPolicyIntegrationTests._Native
    _context = staticmethod(EarlyStopPolicyIntegrationTests._context)

    @staticmethod
    def _report(
        arms: "list[tuple[str, int, float, float | None]]",
        *,
        root_priors: "list[float] | None",
        opponent: "list[tuple[str, int, float, float | None]]" = (),
    ) -> dict:
        """One world report: `(move, visits, q, prior)` per arm, per seat.

        `prior=None` omits the column, which is what a pre-`arm_priors` image
        produces; `root_priors=None` is the crate's own "the prior path did not
        resolve" signal. The two are INDEPENDENT here on purpose -- the pairing
        between them is what the honesty cases exercise.
        """

        def entries(rows):
            out = []
            for move, visits, q, prior in rows:
                entry = {"move": move, "visits": visits, "q": q}
                if prior is not None:
                    entry["prior"] = prior
                out.append(entry)
            return out

        completed = sum(row[1] for row in arms)
        return {
            "iterations": completed,
            "requested_iterations": completed,
            "remaining_iterations": 0,
            "early_stopped": False,
            "model_evals": completed,
            "lossy_renders": 0,
            "attribution_unsafe_renders": 0,
            "prior_fallbacks": 0,
            "root_priors": root_priors,
            "side_one": entries(arms),
            "side_two": entries(opponent),
        }

    @staticmethod
    def _world(label: str):
        return (
            SimpleNamespace(
                party_species={"p1": ("rattata",), "p2": ("chansey",)},
                slot_sides={"p1": "side_one"},
            ),
            SimpleNamespace(to_string=lambda: label),
        )

    def _policy(self, *, telemetry: bool = True, opponent_priors: bool = False):
        policy = object.__new__(EngineMctsPolicy)
        policy.policy_id = "override-telemetry-test"
        policy._config = EngineMctsConfig(
            worlds=2,
            leaf_eval="model",
            model_path="model.pt",
            checkpoint_path="checkpoint.pt",
            tables_path="tables.json",
            search_sims=100,
            search_batch=10,
            override_telemetry=telemetry,
            use_opponent_priors=opponent_priors,
        )
        policy._tables_json = "{}"
        policy.stats = EngineMctsStats()
        policy._world_failures_before = {}
        return policy

    def _run(self, policy, reports, worlds=None, context=None):
        native = self._Native(reports)
        fake_module = SimpleNamespace(
            FoldState=SimpleNamespace(from_payload=lambda _payload: object())
        )
        with (
            patch.dict(sys.modules, {"pokezero_search": fake_module}),
            patch.object(EngineMctsPolicy, "_native", return_value=native),
            patch.object(
                EngineMctsPolicy, "_validate_model_root_observation", return_value=None
            ),
            patch.object(EngineMctsPolicy, "_root_inputs_json", return_value="{}"),
        ):
            decision = policy._search_model(
                self._context() if context is None else context,
                [self._world("world-a")] if worlds is None else worlds,
                SimpleNamespace(to_payload=lambda: {}),
                random.Random(7),
            )
        return decision, native

    # -- the counter FIRES -----------------------------------------------------

    def test_an_override_fires_when_the_model_prefers_the_other_arm(self) -> None:
        """Visits say alpha (action 0); priors say beta (action 1). Provably a
        disagreement, so the counter must fire and name both actions."""
        policy = self._policy()
        decision, _ = self._run(
            policy,
            [self._report(
                [("alpha", 60, 0.5, 0.2), ("beta", 40, 0.5, 0.8)],
                root_priors=[0.2, 0.8],
            )],
        )
        stats = policy.stats.to_dict()
        self.assertEqual(decision.action_index, 0, "the search's own choice is alpha")
        self.assertEqual(stats["model_override_decisions"], 1)
        self.assertEqual(stats["override_measured_decisions"], 1)
        self.assertEqual(stats["search_override_unmeasured"], 0)
        self.assertEqual(stats["model_override_rate"], 1.0)
        block = decision.metadata["engine_mcts"]["override"]
        self.assertEqual((block["model_argmax"], block["search_argmax"]), (1, 0))
        self.assertIs(block["model_override"], True)
        # The forkable address, which is the whole point of retaining it.
        self.assertEqual(stats["override_disagreements"], [{
            "battle_id": "early-stop-test",
            "round": 0,
            "seat": "p1",
            "model_argmax": 1,
            "search_argmax": 0,
            "model_choice": "beta",
        }])

    def test_agreement_does_not_fire(self) -> None:
        """Same visits, priors moved onto the SAME arm: measured, not an override.

        The mirror of the case above and not a weaker version of it: a counter
        that reads 100% is as broken as one that reads 0%, and only the pair
        separates them.
        """
        policy = self._policy()
        decision, _ = self._run(
            policy,
            [self._report(
                [("alpha", 60, 0.5, 0.8), ("beta", 40, 0.5, 0.2)],
                root_priors=[0.8, 0.2],
            )],
        )
        stats = policy.stats.to_dict()
        self.assertEqual(stats["model_override_decisions"], 0)
        self.assertEqual(stats["override_measured_decisions"], 1)
        self.assertEqual(stats["search_override_unmeasured"], 0)
        self.assertEqual(stats["model_override_rate"], 0.0)
        self.assertIs(
            decision.metadata["engine_mcts"]["override"]["model_override"], False
        )
        self.assertEqual(stats["override_disagreements"], [])

    def test_the_priors_decide_the_model_argmax_not_the_visits(self) -> None:
        """A visit-ordering change alone must not move the model's argmax.

        Same priors, reversed visit block. `stats_to_json` sorts entries by
        visits, so a reader that took "the first entry" or "the entry with the
        most visits" as the model's answer would pass every test above and
        silently report the SEARCH's choice twice -- an override rate pinned at
        zero by construction.
        """
        policy = self._policy()
        self._run(
            policy,
            [self._report(
                [("beta", 90, 0.5, 0.1), ("alpha", 10, 0.5, 0.9)],
                root_priors=[0.1, 0.9],
            )],
        )
        stats = policy.stats.to_dict()
        self.assertEqual(stats["model_override_decisions"], 1, "beta played, alpha liked")
        self.assertEqual(stats["override_disagreements"][0]["model_choice"], "alpha")

    # -- the HONESTY counter fires ---------------------------------------------

    def test_a_decision_with_no_root_priors_is_unmeasured_not_agreement(self) -> None:
        """`model_priors=False`, or any root the prior path refused.

        The visit block still names an arm, so the tempting reading is "the model
        agreed". It did not express anything.
        """
        policy = self._policy()
        decision, _ = self._run(
            policy,
            [self._report([("alpha", 60, 0.5, 0.5), ("beta", 40, 0.5, 0.5)],
                          root_priors=None)],
        )
        stats = policy.stats.to_dict()
        self.assertEqual(stats["search_override_unmeasured"], 1)
        self.assertEqual(stats["override_measured_decisions"], 0)
        self.assertEqual(stats["model_override_decisions"], 0)
        self.assertNotIn("model_override_rate", stats, "no denominator, no rate")
        self.assertEqual(
            stats["search_override_unmeasured_causes"], {"no_root_priors": 1}
        )
        block = decision.metadata["engine_mcts"]["override"]
        self.assertIsNone(block["model_argmax"])
        self.assertIsNone(block["model_override"], "never False on an unmeasured read")

    def test_uniform_arm_priors_do_not_manufacture_a_model_argmax(self) -> None:
        """The defect this counter exists to avoid, made concrete.

        A refused root prior path leaves `MoveStats::prior` at the uniform `1/n`
        it was CONSTRUCTED with, and in a report that is indistinguishable from a
        model prior that happens to be flat. An implementation that read the arms
        alone would return a confident argmax here -- the first arm, on the
        tie-break -- and book the decision as measured agreement. The authority is
        `root_priors`, which is null exactly when the path did not resolve.
        """
        policy = self._policy()
        self._run(
            policy,
            [self._report(
                # Uniform, and the arm order deliberately puts a DIFFERENT arm
                # first from the one the visits chose, so a fabricated argmax
                # would even read as an override.
                [("beta", 40, 0.5, 0.5), ("alpha", 60, 0.5, 0.5)],
                root_priors=None,
            )],
        )
        stats = policy.stats.to_dict()
        self.assertEqual(stats["override_measured_decisions"], 0)
        self.assertEqual(stats["model_override_decisions"], 0)
        self.assertEqual(
            stats["search_override_unmeasured_causes"], {"no_root_priors": 1}
        )

    def test_a_stale_image_is_named_rather_than_read_as_agreement(self) -> None:
        """Priors resolved, arms unnamed: new Python against an old crate."""
        policy = self._policy()
        self._run(
            policy,
            [self._report([("alpha", 60, 0.5, None), ("beta", 40, 0.5, None)],
                          root_priors=[0.2, 0.8])],
        )
        self.assertEqual(
            policy.stats.to_dict()["search_override_unmeasured_causes"],
            {"prior_arms_absent": 1},
        )

    def test_priors_that_disagree_with_the_authority_are_refused(self) -> None:
        """Arms whose priors are not `root_priors`' multiset.

        Defensive -- the crate writes both off one stat vector -- but a pairing
        bug would file one arm's prior under another arm's name, and that is a
        WRONG argmax rather than a missing one.
        """
        policy = self._policy()
        self._run(
            policy,
            [self._report([("alpha", 60, 0.5, 0.5), ("beta", 40, 0.5, 0.5)],
                          root_priors=[0.2, 0.8])],
        )
        self.assertEqual(
            policy.stats.to_dict()["search_override_unmeasured_causes"],
            {"prior_arms_misaligned": 1},
        )

    def test_a_partly_priced_decision_is_refused_not_approximated(self) -> None:
        """Two worlds, one priced. A subset aggregate is a different quantity."""
        policy = self._policy()
        self._run(
            policy,
            [
                self._report([("alpha", 60, 0.5, 0.2), ("beta", 40, 0.5, 0.8)],
                             root_priors=[0.2, 0.8]),
                self._report([("alpha", 60, 0.5, 0.5), ("beta", 40, 0.5, 0.5)],
                             root_priors=None),
            ],
            worlds=[self._world("world-a"), self._world("world-b")],
        )
        self.assertEqual(
            policy.stats.to_dict()["search_override_unmeasured_causes"],
            {"priors_missing_in_some_worlds": 1},
        )

    def test_an_unmappable_model_arm_is_unmeasured_and_leaves_the_stop_counters_alone(
        self,
    ) -> None:
        """The model's top arm names no legal request action.

        And the counter that must NOT move: `unmapped_choices` /
        `choices_unmapped_causes` are campaign stop-condition terms, so a probe
        that ran on every searched decision and touched them would make both
        unreadable.
        """
        policy = self._policy()
        self._run(
            policy,
            [self._report(
                [("alpha", 60, 0.5, 0.2), ("nosuchmove", 40, 0.5, 0.8)],
                root_priors=[0.2, 0.8],
            )],
        )
        stats = policy.stats.to_dict()
        self.assertEqual(
            stats["search_override_unmeasured_causes"], {"model_choice_unmapped": 1}
        )
        self.assertEqual(stats["unmapped_choices"], {"nosuchmove": 1},
                         "counted once, by _map_choices, not twice")
        self.assertEqual(stats["choices_unmapped_causes"], {})

    def test_the_denominator_identity_holds_across_a_mixed_run(self) -> None:
        """`measured + unmeasured == searched_decisions`, which is what the
        plan's denominator (`overrides / (searched - unmeasured)`) rests on."""
        policy = self._policy()
        # An override, an unmeasured, an agreement, an unmeasured. The arm priors
        # and `root_priors` are moved TOGETHER: they are the same numbers in the
        # crate, and a test that varied one alone would exercise the misalignment
        # guard instead of the case it names.
        for alpha_prior, beta_prior, priced in (
            (0.2, 0.8, True), (0.5, 0.5, False), (0.8, 0.2, True), (0.5, 0.5, False),
        ):
            self._run(
                policy,
                [self._report(
                    [("alpha", 60, 0.5, alpha_prior), ("beta", 40, 0.5, beta_prior)],
                    root_priors=[alpha_prior, beta_prior] if priced else None,
                )],
            )
        stats = policy.stats.to_dict()
        self.assertEqual(stats["searched_decisions"], 4)
        self.assertEqual(
            stats["override_measured_decisions"] + stats["search_override_unmeasured"],
            stats["searched_decisions"],
        )
        self.assertEqual(stats["model_override_decisions"], 1)

    def test_a_naming_difference_is_not_an_override(self) -> None:
        """Two displays, one action index: not a change of move.

        The engine and the request name the same action differently in three
        places (typed Hidden Power vs plain `hiddenpower`, `MoveChoice::None` vs
        `recharge`/`struggle`), which is what `_ChoiceVocabulary` translates. An
        implementation comparing DISPLAYS reports this as an override. The two
        Hidden Power arms are synthetic -- a real mon carries one -- so this pins
        the translation rather than a reachable state.
        """
        policy = self._policy()
        candidates = [
            {"action_index": 0, "kind": "move", "legal": True, "move_id": "hiddenpower"},
            {"action_index": 1, "kind": "move", "legal": True, "move_id": "beta"},
        ]
        context = SimpleNamespace(
            observation=_FakeObservation(
                (True, True, False, False, False, False, False, False, False),
                candidates,
            ),
            public_materialization_state=SimpleNamespace(
                replay=SimpleNamespace(turn_number=1)
            ),
            player_id="p1",
            battle_id="naming",
            decision_round_index=2,
        )
        self._run(
            policy,
            [self._report(
                [("hiddenpowergrass70", 60, 0.5, 0.2),
                 ("hiddenpowerice70", 40, 0.5, 0.8)],
                root_priors=[0.2, 0.8],
            )],
            context=context,
        )
        stats = policy.stats.to_dict()
        self.assertEqual(stats["override_measured_decisions"], 1)
        self.assertEqual(stats["model_override_decisions"], 0,
                         "both arms are action 0; nothing was overridden")

    def test_a_tie_resolves_the_same_way_on_both_sides(self) -> None:
        """Equal visits and equal priors: not an override, under any tie-break.

        The arms are named so that an insertion-order rule and a name-order rule
        disagree, which is how a tie-break mismatch between the two aggregates
        would show up: as a floor under the override rate that nothing earned.
        """
        policy = self._policy()
        candidates = [
            {"action_index": 0, "kind": "move", "legal": True, "move_id": "alpha"},
            {"action_index": 1, "kind": "move", "legal": True, "move_id": "zeta"},
        ]
        context = SimpleNamespace(
            observation=_FakeObservation(
                (True, True, False, False, False, False, False, False, False),
                candidates,
            ),
            public_materialization_state=SimpleNamespace(
                replay=SimpleNamespace(turn_number=1)
            ),
            player_id="p1",
            battle_id="tie",
            decision_round_index=1,
        )
        self._run(
            policy,
            [self._report(
                [("alpha", 50, 0.5, 0.5), ("zeta", 50, 0.5, 0.5)],
                root_priors=[0.5, 0.5],
            )],
            context=context,
        )
        stats = policy.stats.to_dict()
        self.assertEqual(stats["override_measured_decisions"], 1)
        self.assertEqual(stats["model_override_decisions"], 0)

    # -- flag off --------------------------------------------------------------

    def test_the_flag_off_records_nothing_and_calls_the_pre_flag_arity(self) -> None:
        policy = self._policy(telemetry=False)
        decision, native = self._run(
            policy,
            [self._report([("alpha", 60, 0.5, 0.2), ("beta", 40, 0.5, 0.8)],
                          root_priors=[0.2, 0.8])],
        )
        stats = policy.stats.to_dict()
        self.assertEqual(stats["searched_decisions"], 1)
        for key in (
            "override_measured_decisions",
            "model_override_decisions",
            "search_override_unmeasured",
            "root_arm_gap_samples",
            "opponent_top_arm_decisions",
        ):
            self.assertEqual(stats[key], 0, key)
        self.assertEqual(stats["root_decision_rows"], [])
        self.assertNotIn("override", decision.metadata["engine_mcts"])
        # The 12-positional pre-flag call, byte for byte: nothing after
        # `model_priors` is materialized, so a stale image is unaffected.
        self.assertEqual(len(native.calls[0]), 12)

    def test_the_flag_on_appends_exactly_one_positional_past_fpu(self) -> None:
        policy = self._policy()
        _decision, native = self._run(
            policy,
            [self._report([("alpha", 60, 0.5, 0.2), ("beta", 40, 0.5, 0.8)],
                          root_priors=[0.2, 0.8])],
        )
        call = native.calls[0]
        self.assertEqual(len(call), 17)
        # The cascade must materialize every earlier slot at its DEFAULT, or the
        # True lands in `fpu_reduction` -- which the crate accepts as a valid
        # reduction and which changes selection.
        self.assertEqual(call[12], 0, "early_stop_min_sims")
        self.assertIs(call[13], True, "early_stop_side_one (p1 on side_one)")
        self.assertIs(call[14], False, "use_opponent_priors")
        self.assertIsNone(call[15], "fpu_reduction")
        self.assertIs(call[16], True, "arm_priors")

    def test_the_config_refuses_the_flag_where_it_cannot_be_measured(self) -> None:
        with self.assertRaisesRegex(ValueError, "leaf_eval='model'"):
            EngineMctsConfig(leaf_eval="hp_fraction_crate", override_telemetry=True)

    # -- H2: the two top-arm gaps ---------------------------------------------

    def test_the_q_and_visit_gaps_are_measured_over_the_top_two_arms(self) -> None:
        policy = self._policy()
        decision, _ = self._run(
            policy,
            [self._report(
                [("alpha", 75, 0.62, 0.5), ("beta", 25, 0.55, 0.5)],
                root_priors=[0.5, 0.5],
            )],
        )
        stats = policy.stats.to_dict()
        self.assertEqual(stats["root_arm_gap_samples"], 1)
        self.assertAlmostEqual(stats["root_q_gap_mean"], 0.07, places=6)
        self.assertAlmostEqual(stats["root_visit_gap_mean"], 0.5, places=6)
        self.assertEqual(stats["root_q_gap_histogram"], {"0.07": 1})
        self.assertEqual(stats["root_visit_gap_histogram"], {"0.50": 1})
        block = decision.metadata["engine_mcts"]["override"]
        self.assertAlmostEqual(block["root_q_gap"], 0.07, places=6)
        row = stats["root_decision_rows"][0]
        self.assertEqual([arm["move"] for arm in row["top_arms"]], ["alpha", "beta"])
        self.assertAlmostEqual(row["top_arms"][0]["q"], 0.62, places=6)

    def test_a_p2_decision_reports_q_in_the_acting_seats_frame(self) -> None:
        """`finalize` accumulates the side-ONE-absolute expectation into BOTH stat
        vectors, and `stats_to_json` prints it unreflected, so a p2 decision's
        arms arrive in the opponent's frame. Pooling seats without the flip
        averages a win probability against a loss probability."""
        policy = self._policy()
        context = self._context()
        context.player_id = "p2"
        world = (
            SimpleNamespace(
                party_species={"p1": ("rattata",), "p2": ("chansey",)},
                slot_sides={"p2": "side_two"},
            ),
            SimpleNamespace(to_string=lambda: "world-a"),
        )
        report = self._report([], root_priors=[0.5, 0.5])
        report["side_one"] = []
        report["side_two"] = [
            {"move": "alpha", "visits": 75, "q": 0.62, "prior": 0.5},
            {"move": "beta", "visits": 25, "q": 0.55, "prior": 0.5},
        ]
        self._run(policy, [report], worlds=[world], context=context)
        row = policy.stats.to_dict()["root_decision_rows"][0]
        self.assertAlmostEqual(row["top_arms"][0]["q"], 1.0 - 0.62, places=6)
        self.assertAlmostEqual(row["top_arms"][1]["q"], 1.0 - 0.55, places=6)
        # The gap is flip-invariant, which is exactly why the row carries the
        # values: the histogram alone cannot witness the frame.
        self.assertAlmostEqual(
            policy.stats.to_dict()["root_q_gap_mean"], 0.07, places=6
        )

    def test_a_single_armed_decision_contributes_no_gap(self) -> None:
        policy = self._policy()
        self._run(
            policy,
            [self._report([("alpha", 100, 0.5, 1.0)], root_priors=[1.0])],
        )
        stats = policy.stats.to_dict()
        self.assertEqual(stats["root_arm_gap_samples"], 0)
        self.assertNotIn("root_q_gap_mean", stats)
        self.assertEqual(stats["override_measured_decisions"], 1, "still measurable")

    # -- H4: the in-tree opponent's arm ---------------------------------------

    def test_the_opponent_seats_top_arm_is_absorbed(self) -> None:
        policy = self._policy()
        decision, _ = self._run(
            policy,
            [self._report(
                [("alpha", 60, 0.5, 0.8), ("beta", 40, 0.5, 0.2)],
                root_priors=[0.8, 0.2],
                opponent=[("opp-slow", 30, 0.5, None), ("opp-fast", 70, 0.5, None)],
            )],
        )
        stats = policy.stats.to_dict()
        self.assertEqual(stats["opponent_top_arm_decisions"], 1)
        self.assertEqual(
            decision.metadata["engine_mcts"]["override"]["opponent_top_arm"], "opp-fast"
        )
        self.assertEqual(stats["root_decision_rows"][0]["opponent_top_arm"], "opp-fast")
        self.assertIsNone(stats["root_decision_rows"][0]["opponent_prior_arm"])
        self.assertEqual(stats["opponent_prior_arm_decisions"], 0)

    def test_the_opponents_model_prior_arm_needs_the_flag_and_a_non_uniform_row(
        self,
    ) -> None:
        """Uniform opponent priors are REFUSED rather than argmaxed.

        The crate exports no `root_opponent_priors`, so unlike the acting seat
        there is no authority saying whether that seat was priced from the model.
        A flat row is what a refused opponent seat leaves behind, and taking its
        argmax would report a fabricated prediction for H4 to score.
        """
        uniform = [("opp-slow", 30, 0.5, 0.5), ("opp-fast", 70, 0.5, 0.5)]
        priced = [("opp-slow", 30, 0.5, 0.9), ("opp-fast", 70, 0.5, 0.1)]
        for opponent_priors, opponent, expected in (
            (False, priced, None),
            (True, uniform, None),
            (True, priced, "opp-slow"),
        ):
            with self.subTest(opponent_priors=opponent_priors):
                policy = self._policy(opponent_priors=opponent_priors)
                self._run(
                    policy,
                    [self._report(
                        [("alpha", 60, 0.5, 0.8), ("beta", 40, 0.5, 0.2)],
                        root_priors=[0.8, 0.2],
                        opponent=opponent,
                    )],
                )
                stats = policy.stats.to_dict()
                self.assertEqual(
                    stats["root_decision_rows"][0]["opponent_prior_arm"], expected
                )
                self.assertEqual(
                    stats["opponent_prior_arm_decisions"], 0 if expected is None else 1
                )

    # -- bounded stores --------------------------------------------------------

    def test_both_per_decision_stores_are_bounded_and_count_their_overflow(self) -> None:
        policy = self._policy()
        reports = [self._report(
            [("alpha", 60, 0.5, 0.2), ("beta", 40, 0.5, 0.8)], root_priors=[0.2, 0.8]
        ) for _ in range(3)]
        with (
            patch("pokezero.engine_search._OVERRIDE_DISAGREEMENT_ADDRESSES", 1),
            patch("pokezero.engine_search._ROOT_DECISION_ROWS", 2),
        ):
            for report in reports:
                self._run(policy, [report])
        stats = policy.stats.to_dict()
        self.assertEqual(stats["model_override_decisions"], 3, "the COUNT is complete")
        self.assertEqual(len(stats["override_disagreements"]), 1)
        self.assertEqual(stats["override_disagreement_addresses_dropped"], 2)
        self.assertEqual(len(stats["root_decision_rows"]), 2)
        self.assertEqual(stats["root_decision_rows_dropped"], 1)


class WorldCacheKeyTest(unittest.TestCase):
    def test_the_seed_is_excluded(self) -> None:
        from pokezero.engine_search import world_cache_key

        a = {"state_str": "S", "ctx_json": "C", "seed": 1}
        b = {"state_str": "S", "ctx_json": "C", "seed": 999}
        self.assertEqual(world_cache_key(a, "side_one"), world_cache_key(b, "side_one"))

    def test_real_differences_separate(self) -> None:
        from pokezero.engine_search import world_cache_key

        base = {"state_str": "S", "ctx_json": "C", "seed": 1}
        k = world_cache_key(base, "side_one")
        self.assertNotEqual(k, world_cache_key({**base, "state_str": "S2"}, "side_one"))
        self.assertNotEqual(k, world_cache_key({**base, "ctx_json": "C2"}, "side_one"))
        self.assertNotEqual(k, world_cache_key(base, "side_two"))


class FreeDecisionFeatureTest(unittest.TestCase):
    """The inputs a production depth rule may key on, and the line it must not cross.

    Every feature here is knowable BEFORE the search runs. That is the whole discipline:
    a rule fitted on a post-search quantity -- occupancy, the top-1 visit share, the Q
    gap, whether search overrode the model -- cannot be evaluated in production, because
    there you must choose the depth first. The `f_` prefix is that boundary, made
    mechanical so a consumer can select input columns without knowing the schema.
    """

    @staticmethod
    def _ctx(moves, switches, turn=7):
        """Build the observation the way `showdown.py` publishes it, not the way the
        reader wishes it looked.

        The first version of this fixture handed the function
        `SimpleNamespace(observation=SimpleNamespace(candidates=[...]))`. That shape does
        not exist: `PokeZeroObservationV0` has no `candidates` field -- the candidates are
        `metadata["action_candidates"]` -- so the code under test read None, counted zero
        legal actions, and marked every decision FORCED, while this test passed. It cost a
        collection run: 29 of 29 shard rows came back `f_legal_actions == 0` with
        `top_arms` listing real moves beside them.

        So the fixture now mirrors `_action_candidate_metadata`: ALL nine slots present (4
        move, 5 switch) whether legal or not, each carrying its `action_index`, and a
        `legal_action_mask` that must agree. Anything that reads a different field, or
        skips the mask cross-check, fails here.
        """
        cands, mask = [], []
        for slot in range(4):                      # move slots 0-3
            legal = slot < moves
            cands.append({"action_index": slot, "kind": "move", "legal": legal,
                          "move_slot": slot, "move_id": f"move{slot}",
                          "move_name": f"Move {slot}", "disabled": not legal})
            mask.append(legal)
        for slot in range(5):                      # switch slots 4-8
            legal = slot < switches
            cands.append({"action_index": 4 + slot, "kind": "switch", "legal": legal,
                          "switch_slot": slot, "team_index": slot,
                          "pokemon": {"species": f"mon{slot}"}})
            mask.append(legal)
        return SimpleNamespace(
            observation=SimpleNamespace(
                metadata={"action_candidates": cands},
                legal_action_mask=tuple(mask),
            ),
            public_materialization_state=SimpleNamespace(
                replay=SimpleNamespace(turn_number=turn)),
        )

    def test_the_candidates_are_read_from_metadata_not_an_attribute(self) -> None:
        """The regression pin for the shape defect above.

        An observation carrying a decoy `candidates` attribute AND the real
        `metadata["action_candidates"]` must be counted from metadata. Reading the
        attribute would give 1; reading metadata gives 6.
        """
        ctx = self._ctx(3, 3)
        ctx.observation.candidates = [{"kind": "move", "legal": True}]  # decoy
        f = free_decision_features(ctx, 4096, {"a": 1.0})
        self.assertEqual(f["f_legal_actions"], 6)
        self.assertFalse(f["f_forced"])

    def test_an_unreadable_candidate_list_reports_ABSENCE_not_zero(self) -> None:
        """A failed read must not manufacture the strongest available claim.

        `f_forced` is `actions <= 1`, so a reader that silently returns 0 asserts "there
        was nothing to decide" from no evidence. That is exactly how the metadata-path
        defect survived: it never raised, it just labelled all 29 rows of the first
        collection shard `forced` with zero legal actions, beside `top_arms` listing real
        moves. None propagates; 0 lies.
        """
        bare = SimpleNamespace(
            observation=SimpleNamespace(metadata={}, legal_action_mask=None),
            public_materialization_state=None,
        )
        f = free_decision_features(bare, 4096, {"a": 1.0})
        for key in ("f_legal_moves", "f_legal_switches", "f_legal_actions", "f_forced"):
            with self.subTest(key=key):
                self.assertIsNone(f[key])
        # The features that ARE readable from a bare context still report.
        self.assertEqual(f["f_sims_per_world"], 4096)
        self.assertEqual(f["f_turn"], 0)

    def test_admission_needs_BOTH_the_legal_flag_and_the_mask_bit(self) -> None:
        """Either filter alone must be able to reject.

        `showdown.py` derives `legal` and `legal_action_mask` from the same source, so a
        disagreement is not an observed production state -- this pins the AND as
        deliberate rather than redundant, and mirrors `_choice_vocabulary`'s admission
        rule. Without it, dropping the `legal` check is a silent no-op change here and a
        live overcount anywhere the two sources ever diverge.
        """
        ctx = self._ctx(4, 5)
        cands = ctx.observation.metadata["action_candidates"]
        cands[1] = {**cands[1], "legal": False}   # flag says no, mask still says yes
        f = free_decision_features(ctx, 4096, {"a": 1.0})
        self.assertEqual((f["f_legal_moves"], f["f_legal_switches"]), (3, 5))

    def test_the_mask_vetoes_a_candidate_that_claims_legal(self) -> None:
        """`legal` and the mask must AGREE. Admission mirrors `_choice_vocabulary`.

        A feature that counted a wider action set than the search chooses from would key
        the depth table on a branching factor the search never faced.
        """
        ctx = self._ctx(4, 5)
        vetoed = list(ctx.observation.legal_action_mask)
        vetoed[2] = False                       # mask says no; the candidate says legal
        ctx.observation.legal_action_mask = tuple(vetoed)
        f = free_decision_features(ctx, 4096, {"a": 1.0})
        self.assertEqual((f["f_legal_moves"], f["f_legal_switches"]), (3, 5))

    def test_branching_is_split_into_moves_and_switches(self) -> None:
        # Split deliberately: a switch changes the active mon and a move does not, so
        # they do not cost the same in tree width. An illegal candidate counts as neither.
        f = free_decision_features(self._ctx(4, 5), 4096, {"a": 1.0})
        self.assertEqual((f["f_legal_moves"], f["f_legal_switches"]), (4, 5))
        self.assertEqual(f["f_legal_actions"], 9)
        self.assertFalse(f["f_forced"])

    def test_a_single_legal_action_is_forced(self) -> None:
        # Nothing to decide, so no depth is worth buying. Measured at 1.6% of decisions
        # overall and 3.2% past turn 30 -- a late-game phenomenon.
        for moves, switches in ((1, 0), (0, 1)):
            with self.subTest(moves=moves, switches=switches):
                self.assertTrue(free_decision_features(
                    self._ctx(moves, switches), 4096, {"a": 1.0})["f_forced"])

    def test_the_root_prior_summarises_to_confidence_and_entropy(self) -> None:
        sharp = free_decision_features(self._ctx(4, 0), 4096, {"a": 0.97, "b": 0.03})
        flat = free_decision_features(self._ctx(4, 0), 4096, {"a": 0.5, "b": 0.5})
        self.assertAlmostEqual(sharp["f_root_prior_top1"], 0.97, places=3)
        self.assertAlmostEqual(flat["f_root_prior_top1"], 0.5, places=3)
        self.assertLess(sharp["f_root_prior_entropy"], flat["f_root_prior_entropy"],
                        "a confident prior must read as LOWER entropy")

    def test_an_absent_prior_is_None_not_a_confident_zero(self) -> None:
        # The prior is unavailable when the crate refused to price the root. Reporting
        # 0.0 would be a claim about confidence; None is an absence.
        f = free_decision_features(self._ctx(4, 0), 4096, {})
        self.assertIsNone(f["f_root_prior_top1"])
        self.assertIsNone(f["f_root_prior_entropy"])

    def test_the_allocation_and_turn_are_carried(self) -> None:
        f = free_decision_features(self._ctx(2, 1, turn=23), 1024, {"a": 1.0})
        self.assertEqual(f["f_sims_per_world"], 1024)
        self.assertEqual(f["f_turn"], 23)

    def test_no_label_can_leak_into_the_input_set(self) -> None:
        f = free_decision_features(self._ctx(4, 5), 4096, {"a": 1.0})
        self.assertTrue(all(k.startswith("f_") for k in f), sorted(f))
        for label in ("depth_occupancy", "max_depth_reached", "top_arms",
                      "model_override", "visit_share"):
            self.assertNotIn(label, f)


class TurnAllocationTest(unittest.TestCase):
    """`_turn_allocation`: what ONE turn spends, and that it is always one budget.

    `search_sims` is the TOTAL sim-equivalents for a decision, spent once. The ladder
    walks across TURNS, not within a decision. An earlier revision read the spec as a
    within-decision escalation and ran a fresh full-budget search per rung, which cost
    6.7 x 16,384 ~ 110,000 sims per decision -- measured on a canary at 36-63 s against
    the banked fixed cell's 10.06 s, a 3.6x-6.3x REGRESSION where the feature is meant
    to be compute-neutral. These tests exist to make that unbuildable again.
    """

    def _policy(self, **cfg):
        base = dict(
            leaf_eval="model", worlds=4, search_sims=16384, search_batch=16,
            search_depth=6, model_path="/tmp/m.pt", checkpoint_path="/tmp/c.pt",
            tables_path="/tmp/t.json",
        )
        base.update(cfg)
        policy = EngineMctsPolicy.__new__(EngineMctsPolicy)
        policy._config = EngineMctsConfig(**base)
        policy.stats = EngineMctsStats()
        policy._ladder_battle = None
        policy._ladder_worlds = None
        policy._ladder_depth = None
        policy._ladder_depth_ceiling = {}
        policy._ladder_probing = False
        return policy

    def test_a_fixed_cell_spends_the_configured_budget_untouched(self) -> None:
        # THE NO-OP PROPERTY, and it is about `None`, not a number: `None` means "use
        # the configured budget", which is what keeps a fixed cell's native argv
        # byte-identical to every banked shard's. A number here silently ran it at
        # budget/worlds -- a 4x compute cut at worlds=4 under an unchanged config_id.
        self.assertEqual(self._policy(search_depth=4)._turn_allocation(4), (4, None, 4))

    def test_one_turn_is_one_budget_at_every_allocation(self) -> None:
        # COMPUTE NEUTRALITY, the property the whole design rests on: 4x4096 and
        # 1x16384 are the same 16,384 spent two ways, so a dynamic cell costs the same
        # per decision as a fixed one.
        for worlds in (4, 3, 2, 1):
            with self.subTest(worlds=worlds):
                p = self._policy(depth_min=2, worlds_min=1)
                p._ladder_worlds, p._ladder_depth = worlds, 2
                got_worlds, sims, depth = p._turn_allocation(4)
                self.assertEqual(got_worlds, worlds)
                self.assertEqual((got_worlds, depth), (worlds, 2))
                self.assertLessEqual(got_worlds * sims, 16384, "never OVER the budget")
                self.assertGreaterEqual(got_worlds * sims, 16384 - worlds,
                                        "and never materially under it")

    def test_the_allocation_never_exceeds_the_worlds_actually_sampled(self) -> None:
        p = self._policy(worlds=16, worlds_min=2, depth_min=2)
        p._ladder_worlds, p._ladder_depth = 16, 2
        self.assertEqual(p._turn_allocation(3)[0], 3)

    def test_a_budget_below_the_world_cap_is_refused_not_degraded(self) -> None:
        # Refused at config time, which is what lets `_turn_allocation` divide without
        # a clamp. The clamp it replaced broke the worlds_min FLOOR (ran 2 worlds when
        # the floor said 3) and put a false 0.25 into world_search_abort_rate.
        for kwargs in ({"depth_min": 2, "worlds_min": 3}, {"worlds_min": 2}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError) as caught:
                    self._policy(search_sims=2, search_batch=1, search_depth=3, **kwargs)
                self.assertIn("search_sims must be >= worlds", str(caught.exception))


class LadderStateMachineTest(unittest.TestCase):
    """The across-TURN state machine and its SHADOW probe.

    One PLAYED search per decision, so the move is always chosen from a depth already
    known to saturate. When there is headroom, a second search runs at depth+1 purely to
    answer "would this allocation fill one more ply?" and its decision is DISCARDED.

    That shape is what makes the rule safe. Both naive rules are broken: up-only parks
    one depth PAST the ceiling (it moves to D+1 on saturating at D and has no way back),
    and up-and-down oscillates D <-> D+1 forever -- and in that case half of all turns
    PLAY a move chosen from exactly the thin tree the saturation rule exists to avoid.
    The shadow costs a doubled budget on the probing turn and nothing else, and probes
    are bounded to one per world count per battle.
    """

    def _policy(self, **cfg):
        from pokezero.policy import PolicyDecision

        base = dict(
            leaf_eval="model", worlds=4, search_sims=16384, search_batch=16,
            search_depth=8, model_path="/tmp/m.pt", checkpoint_path="/tmp/c.pt",
            tables_path="/tmp/t.json", depth_min=2, worlds_min=1,
        )
        base.update(cfg)
        policy = EngineMctsPolicy.__new__(EngineMctsPolicy)
        policy._config = EngineMctsConfig(**base)
        policy.stats = EngineMctsStats()
        policy.policy_id = "engine-mcts"
        policy._ladder_battle = None
        policy._ladder_worlds = None
        policy._ladder_depth = None
        policy._ladder_depth_ceiling = {}
        policy._ladder_probing = False
        policy._ladder_pending_addresses = None
        policy.searches = []          # every search: (worlds, depth, "played"|"shadow")
        policy.script = {}
        policy._turn = 0
        policy._calls_this_turn = 0

        def _fake_search_model(context, worlds, live_fold, rng):
            # FIRST search of a turn is the played one; any later search in the same
            # turn is a shadow. That is the contract `_search_ladder` implements.
            kind = "played" if policy._calls_this_turn == 0 else "shadow"
            policy._calls_this_turn += 1
            policy.searches.append((len(worlds), policy._ladder_depth_override, kind))
            plan = policy.script.get(policy._turn, {})
            key = "" if kind == "played" else "shadow_"
            if plan.get(key + "fallback"):
                policy.stats.fallback_decisions += 1
                return PolicyDecision(action_index=8, policy_id=policy.policy_id)
            policy._ladder_saturated = plan.get(key + "saturated", False)
            policy._ladder_worlds_agree = plan.get("agree", False)
            return PolicyDecision(action_index=0, policy_id=policy.policy_id)

        policy._search_model = _fake_search_model
        return policy

    @staticmethod
    def _ctx(battle="b1"):
        return SimpleNamespace(battle_id=battle, decision_round_index=0, player_id="p1")

    def _turns(self, policy, n, battle="b1"):
        for i in range(n):
            policy._turn = i
            policy._calls_this_turn = 0
            policy._search_ladder(self._ctx(battle),
                                  [(object(), object()) for _ in range(4)],
                                  object(), random.Random(0))
        return [(w, d) for w, d, k in policy.searches if k == "played"]

    # -- the cost property -----------------------------------------------------

    def test_a_turn_with_no_headroom_runs_exactly_one_search(self) -> None:
        # THE COST PROPERTY. The rung design ran 6.1-8.9 searches per decision, measured
        # at 36-63 s against the banked fixed cell's 10.06 s.
        p = self._policy()
        p.script = {i: {"saturated": False, "agree": True} for i in range(5)}
        self._turns(p, 5)
        self.assertEqual([k for _, _, k in p.searches], ["played"] * 5)
        self.assertEqual(p.stats.ladder_shadow_probes, 0)
        self.assertEqual(p.stats.to_dict()["ladder_rungs_per_decision"], 1.0)

    def test_a_probing_turn_runs_two_and_plays_the_shallower(self) -> None:
        p = self._policy()
        p.script = {0: {"saturated": True, "shadow_saturated": True}}
        p.script.update({i: {"saturated": False} for i in range(1, 3)})
        played = self._turns(p, 3)
        self.assertEqual([k for _, _, k in p.searches][:2], ["played", "shadow"])
        self.assertEqual(p.searches[0][1], 2, "the played move is at the KNOWN depth")
        self.assertEqual(p.searches[1][1], 3, "the shadow looks one ply deeper")
        self.assertEqual(played[0], (4, 2))
        self.assertEqual(p.stats.ladder_shadow_probes, 1)

    # -- depth moves only on shadow evidence -----------------------------------

    def test_depth_advances_when_the_shadow_saturates(self) -> None:
        p = self._policy()
        p.script = {i: {"saturated": True, "shadow_saturated": True} for i in range(3)}
        self.assertEqual([d for _, d in self._turns(p, 3)], [2, 3, 4])
        self.assertEqual(p.stats.ladder_depth_rungs, 3)

    def test_a_shadow_that_does_not_saturate_latches_and_never_oscillates(self) -> None:
        """THE POINT OF THE WHOLE DESIGN.

        Turn 0 saturates at depth 2, so a shadow tries depth 3 and fails. Depth stays at
        2 -- and no move was ever played at 3, which is what the shadow buys over
        spending a real turn there. It must then STOP probing: without the latch it
        would shadow every turn forever, doubling the budget on all of them.
        """
        p = self._policy()
        p.script = {i: {"saturated": True, "shadow_saturated": False} for i in range(6)}
        played = self._turns(p, 6)
        self.assertEqual([d for _, d in played], [2] * 6, "depth never moves")
        self.assertEqual([d for _, d, k in p.searches if k == "shadow"], [3],
                         "and it probes ONCE, not every turn")
        self.assertEqual(p.stats.ladder_depth_latches, 1)
        self.assertEqual(p.stats.ladder_shadow_probes, 1)

    def test_a_failed_shadow_latches_nothing(self) -> None:
        # A latch must record a MEASUREMENT. A shadow that fell back measured nothing,
        # so latching on it would record a ceiling the search never found -- and would
        # permanently cap the battle on one bad search.
        p = self._policy()
        p.script = {0: {"saturated": True, "shadow_fallback": True},
                    1: {"saturated": True, "shadow_saturated": True}}
        p.script.update({i: {"saturated": False} for i in range(2, 4)})
        self._turns(p, 4)
        self.assertEqual(p.stats.ladder_shadow_probe_failures, 1)
        self.assertEqual(p.stats.ladder_depth_latches, 0)
        self.assertEqual(p.stats.ladder_shadow_probes, 2, "it may probe again")

    # -- worlds ---------------------------------------------------------------

    def test_worlds_step_down_one_at_a_time_on_agreement(self) -> None:
        p = self._policy()
        p.script = {i: {"agree": True, "saturated": False} for i in range(5)}
        self.assertEqual([w for w, _ in self._turns(p, 5)], [4, 3, 2, 1, 1])
        self.assertEqual(p.stats.ladder_world_drops, 3)

    def test_worlds_do_not_drop_while_the_leaders_disagree(self) -> None:
        p = self._policy()
        p.script = {i: {"agree": False, "saturated": False} for i in range(4)}
        self.assertEqual([w for w, _ in self._turns(p, 4)], [4] * 4)
        self.assertEqual(p.stats.ladder_world_drops, 0)
        self.assertEqual(p.stats.ladder_worlds_disagree_stops, 4)

    def test_the_latch_is_scoped_to_its_world_count(self) -> None:
        # The mechanism BEHIND the test below, pinned separately because the code that
        # looked like it did the clearing was dead. The ceiling is keyed by world count
        # and worlds only ever decrease, so a ceiling latched at 4 worlds cannot gate 3 --
        # no explicit clearing step exists or is needed.
        p = self._policy()
        p._ladder_depth_ceiling = {4: 2}
        self.assertFalse(p._should_shadow_probe(4, 2, saturated=True),
                         "latched at 4 worlds: no probe")
        self.assertTrue(p._should_shadow_probe(3, 2, saturated=True),
                        "3 worlds has its own ceiling, unset, so it may probe")

    def test_dropping_a_world_clears_the_latch(self) -> None:
        # The latch describes an ALLOCATION, not a depth. Dropping a world raises
        # sims-per-world, which is the only thing that can change the saturating depth,
        # so it is the only thing that licenses probing that depth again.
        p = self._policy()
        p.script = {0: {"saturated": True, "shadow_saturated": False, "agree": True},
                    1: {"saturated": True, "shadow_saturated": True}}
        p.script.update({i: {"saturated": False} for i in range(2, 4)})
        self._turns(p, 4)
        shadows = [(w, d) for w, d, k in p.searches if k == "shadow"]
        self.assertEqual(shadows[0], (4, 3), "probed and latched at 4 worlds")
        self.assertEqual(shadows[1], (3, 3), "probed AGAIN once worlds dropped")
        self.assertEqual(p.stats.ladder_depth_latches, 1)

    # -- floors, caps, reset, fallback ----------------------------------------

    def test_the_floors_and_caps_are_respected(self) -> None:
        p = self._policy(search_depth=4, depth_min=2, worlds_min=2)
        p.script = {i: {"saturated": True, "shadow_saturated": True, "agree": True}
                    for i in range(10)}
        played = self._turns(p, 10)
        self.assertEqual(min(w for w, _ in played), 2, "worlds_min is a floor")
        self.assertEqual(max(d for _, d in played), 4, "search_depth is a cap")
        self.assertGreaterEqual(min(d for _, d in played), 2, "depth_min is a floor")
        self.assertEqual([d for _, d, k in p.searches if k == "shadow" and d > 4], [],
                         "and no shadow probes past the cap either")

    def test_the_state_resets_per_battle(self) -> None:
        p = self._policy()
        p.script = {i: {"saturated": True, "shadow_saturated": True, "agree": True}
                    for i in range(8)}
        self._turns(p, 4, battle="b1")
        climbed = [(w, d) for w, d, k in p.searches if k == "played"][-1]
        self._turns(p, 1, battle="b2")
        self.assertEqual([(w, d) for w, d, k in p.searches if k == "played"][-1], (4, 2),
                         "a new battle is a new problem: back to the caps and floors")
        self.assertNotEqual(climbed, (4, 2), "and it really had moved away from them")

    def test_a_fallback_moves_nothing(self) -> None:
        p = self._policy()
        p.script = {0: {"fallback": True}, 1: {"saturated": True, "shadow_saturated": True}}
        played = self._turns(p, 2)
        self.assertEqual(played, [(4, 2), (4, 2)], "the failed turn moved nothing")
        self.assertEqual(p.stats.ladder_unsearched_decisions, 1)
        self.assertEqual(p.stats.ladder_shadow_probes, 1, "and it did not probe either")

    def test_a_fixed_cell_neither_adapts_nor_shadows_nor_stamps(self) -> None:
        p = self._policy(depth_min=None, worlds_min=None)
        p.script = {i: {"saturated": True, "shadow_saturated": True, "agree": True}
                    for i in range(3)}
        played = self._turns(p, 3)
        self.assertEqual(played, [(4, 8)] * 3, "no adaptation at all")
        self.assertEqual(p.stats.ladder_shadow_probes, 0, "and never a doubled budget")
        self.assertFalse(p.stats.to_dict()["ladder_dynamic"])

    # -- the shadow must not vote --------------------------------------------

    def test_the_shadow_gets_no_vote_in_any_per_decision_claim(self) -> None:
        """The rewind, on the one path that still needs it.

        Four review rounds established that a second search inside one decision
        contaminates every per-decision surface -- counters, gap sums, histograms, rows,
        override addresses. The shadow is exactly that second search, so its CLAIMS are
        rewound and its WORK is not: it made no decision, so it gets no vote in any
        rate; it really did burn the sims and the wall, so a cost analysis must see them.
        """
        from pokezero.engine_search import (
            LADDER_PER_DECISION_CLAIM_HISTOGRAMS, LADDER_PER_DECISION_CLAIMS,
        )
        from pokezero.policy import PolicyDecision

        p = self._policy()
        p.script = {0: {"saturated": True, "shadow_saturated": True}}
        p.script.update({i: {"saturated": False} for i in range(1, 2)})
        orig = p._search_model

        def _charging(context, worlds, live_fold, rng):
            for name in LADDER_PER_DECISION_CLAIMS:
                setattr(p.stats, name, getattr(p.stats, name) + 1)
            for name in LADDER_PER_DECISION_CLAIM_HISTOGRAMS:
                getattr(p.stats, name)["0.00-0.05"] += 1
            p.stats.root_decision_rows.append({"marker": len(p.searches)})
            p.stats.total_iterations += 16384
            staging = getattr(p, "_ladder_pending_addresses", None)
            (staging.append if staging is not None else p._commit_override_address)(
                {"marker": len(p.searches)}
            )
            return orig(context, worlds, live_fold, rng)

        p._search_model = _charging
        self._turns(p, 1)
        self.assertEqual([k for _, _, k in p.searches], ["played", "shadow"])
        for name in LADDER_PER_DECISION_CLAIMS:
            with self.subTest(claim=name):
                self.assertEqual(getattr(p.stats, name), 1, "one decision, one vote")
        for name in LADDER_PER_DECISION_CLAIM_HISTOGRAMS:
            with self.subTest(histogram=name):
                self.assertEqual(dict(getattr(p.stats, name)), {"0.00-0.05": 1})
        self.assertEqual(len(p.stats.root_decision_rows), 1, "the shadow leaves no row")
        self.assertEqual(len(p.stats.override_disagreements), 1)
        self.assertEqual(p.stats.override_addresses_superseded, 1)
        # WORK is kept: the shadow really did spend a budget, and hiding that would
        # under-report the feature's cost.
        self.assertEqual(p.stats.total_iterations, 2 * 16384)
        self.assertEqual(p.stats.ladder_rungs_run, 2)
        self.assertEqual(p.stats.ladder_decisions, 1)
        self.assertEqual(p.stats.to_dict()["ladder_rungs_per_decision"], 2.0)

