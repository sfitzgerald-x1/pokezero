"""Unit tests for the engine-MCTS POC policy (fake engine module; no native dep)."""

from __future__ import annotations

import os
import random
import sys
import unittest
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pokezero.engine_search import EngineMctsConfig, EngineMctsPolicy  # noqa: E402


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
                    tables_path="/nonexistent/tables.json",
                ),
            )


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