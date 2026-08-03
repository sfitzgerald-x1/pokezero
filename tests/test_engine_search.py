"""Unit tests for the engine-MCTS POC policy (fake engine module; no native dep)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import sys
import tempfile
import warnings
from types import SimpleNamespace
import unittest
from collections import Counter
from dataclasses import replace
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pokezero.engine_search import (  # noqa: E402
    EngineMctsConfig,
    EngineMctsPolicy,
    EngineMctsStats,
    EngineSearchFallbackError,
    _REASON_DETAIL_LIMIT,
    _bounded_reason_detail,
    _latch_encoder_tables_to_model_config,
    _locked_aggregate_choice,
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
            with patch(
                "pokezero.neural_policy.load_transformer_model_config",
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
    ) -> dict:
        completed = alpha + beta
        return {
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
        # 10 attempts -> 8 built -> 6 searched. Those are two DIFFERENT defects
        # (belief sampling / world building vs. the search aborting on an
        # attribution-unsafe branch) and were previously only separable by
        # parsing the `world_failure_reasons` taxonomy.
        stats = EngineMctsStats()
        stats.worlds_attempted = 10
        stats.worlds_constructed = 8
        stats.worlds_searched = 6
        payload = stats.to_dict()
        self.assertAlmostEqual(payload["belief_sample_rejection_rate"], 0.2)
        self.assertAlmostEqual(payload["world_search_abort_rate"], 0.25)

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

        `foulplay_bridge.py:2475`, `mcts_acceptance_h2h.py:97` and
        `k0_grid_h2h.py:158` all select "model".
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
                "pokezero.engine_search.build_poke_engine_state",
                side_effect=lambda *a, **k: SimpleNamespace(to_string=lambda: "S"),
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

    def test_a_healthy_decision_reports_a_zero_abort_rate(self) -> None:
        harness = EarlyStopPolicyIntegrationTests()
        policy = harness._policy(early_stop=False)
        policy.stats.worlds_constructed = 2
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


if __name__ == "__main__":
    unittest.main()
