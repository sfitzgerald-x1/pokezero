"""Unit tests for the engine-MCTS POC policy (fake engine module; no native dep)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import sys
import tempfile
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
    EngineSearchFallbackError,
    _latch_encoder_tables_to_model_config,
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
        return _policy()._public_effect_signals(context)

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


if __name__ == "__main__":
    unittest.main()
