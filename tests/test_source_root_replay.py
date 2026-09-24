from dataclasses import replace
import unittest

from pokezero.actions import ACTION_COUNT
from pokezero.mcts_eval.source_root_replay import SourceRootReplayError, source_bound_replay_prefix
from pokezero.public_decision_corpus import (
    PublicActionIdentifier,
    PublicDecisionRecord,
    PublicObservation,
    PublicResolvedActionRound,
    public_decision_id,
)


def _mask(*legal: int) -> tuple[bool, ...]:
    return tuple(index in legal for index in range(ACTION_COUNT))


def _record(
    *,
    turn_index: int,
    recorded_action_index: int,
    candidates: list[dict[str, object]],
    rounds: tuple[PublicResolvedActionRound, ...],
    seed: int = 9,
    player: str = "p1",
) -> PublicDecisionRecord:
    observation = PublicObservation(
        schema_version="test",
        categorical_ids=(),
        numeric_features=(),
        token_type_ids=(),
        attention_mask=(),
        legal_action_mask=_mask(recorded_action_index),
        acting_player_state={"action_candidates": candidates},
    )
    prototype = PublicDecisionRecord(
        decision_id="pending",
        battle_id="source-root-replay",
        seed=seed,
        format_id="gen3randombattle",
        acting_player=player,
        turn_index=turn_index,
        recorded_action_index=recorded_action_index,
        observation=observation,
        history=(),
        current_legal_action_mask=_mask(recorded_action_index),
        public_resolved_action_rounds=rounds,
        public_belief_view={},
    )
    return replace(prototype, decision_id=public_decision_id(prototype))


def _round(turn_index: int, p1: PublicActionIdentifier, p2: PublicActionIdentifier) -> PublicResolvedActionRound:
    return PublicResolvedActionRound(turn_index=turn_index, actions={"p1": p1, "p2": p2})


class SourceBoundReplayPrefixTest(unittest.TestCase):
    def test_repairs_only_the_source_actors_own_unresolved_move(self) -> None:
        prior = _record(
            turn_index=0,
            recorded_action_index=4,
            candidates=[{"action_index": 4, "kind": "move", "move_id": "spore", "legal": True}],
            rounds=(),
        )
        target = _record(
            turn_index=1,
            recorded_action_index=1,
            candidates=[{"action_index": 1, "kind": "move", "move_id": "tackle", "legal": True}],
            rounds=(
                _round(
                    0,
                    PublicActionIdentifier(kind="event", event_id="unresolved-public-event"),
                    PublicActionIdentifier(kind="move", move_id="tackle"),
                ),
            ),
        )

        repaired = source_bound_replay_prefix(target, source_records=(prior, target))

        self.assertEqual(repaired.public_action_rounds[0].actions["p1"], PublicActionIdentifier(kind="move", move_id="spore"))
        self.assertEqual([repair.to_dict() for repair in repaired.repairs], [{
            "turn_index": 0,
            "player_id": "p1",
            "original_event_id": "unresolved-public-event",
            "source_action": {"kind": "move", "move_id": "spore"},
        }])

    def test_repairs_source_switch_using_canonical_species(self) -> None:
        prior = _record(
            turn_index=0,
            recorded_action_index=6,
            candidates=[{"action_index": 6, "kind": "switch", "pokemon": {"species": "Gengar"}, "legal": True}],
            rounds=(),
        )
        target = _record(
            turn_index=1,
            recorded_action_index=1,
            candidates=[{"action_index": 1, "kind": "move", "move_id": "tackle", "legal": True}],
            rounds=(_round(0, PublicActionIdentifier(kind="event", event_id="unresolved-public-event"), PublicActionIdentifier(kind="move", move_id="tackle")),),
        )

        repaired = source_bound_replay_prefix(target, source_records=(prior,))

        self.assertEqual(repaired.public_action_rounds[0].actions["p1"], PublicActionIdentifier(kind="switch", switched_species="Gengar"))

    def test_refuses_unresolved_opponent_action(self) -> None:
        target = _record(
            turn_index=1,
            recorded_action_index=1,
            candidates=[{"action_index": 1, "kind": "move", "move_id": "tackle", "legal": True}],
            rounds=(_round(0, PublicActionIdentifier(kind="move", move_id="tackle"), PublicActionIdentifier(kind="event", event_id="unresolved-public-event")),),
        )

        with self.assertRaisesRegex(SourceRootReplayError, "unresolved_public_event_for_non_source_player"):
            source_bound_replay_prefix(target, source_records=())

    def test_refuses_missing_or_noncanonical_source_action(self) -> None:
        target = _record(
            turn_index=1,
            recorded_action_index=1,
            candidates=[{"action_index": 1, "kind": "move", "move_id": "tackle", "legal": True}],
            rounds=(_round(0, PublicActionIdentifier(kind="event", event_id="unresolved-public-event"), PublicActionIdentifier(kind="move", move_id="tackle")),),
        )
        with self.assertRaisesRegex(SourceRootReplayError, "missing_source_record"):
            source_bound_replay_prefix(target, source_records=())

        noncanonical = _record(
            turn_index=0,
            recorded_action_index=3,
            candidates=[{"action_index": 3, "kind": "tera", "legal": True}],
            rounds=(),
        )
        with self.assertRaisesRegex(SourceRootReplayError, "not_canonical_move_or_switch"):
            source_bound_replay_prefix(target, source_records=(noncanonical,))

    def test_refuses_ambiguous_same_turn_source_record(self) -> None:
        target = _record(
            turn_index=1,
            recorded_action_index=1,
            candidates=[{"action_index": 1, "kind": "move", "move_id": "tackle", "legal": True}],
            rounds=(_round(0, PublicActionIdentifier(kind="event", event_id="unresolved-public-event"), PublicActionIdentifier(kind="move", move_id="tackle")),),
        )
        first = _record(turn_index=0, recorded_action_index=2, candidates=[{"action_index": 2, "kind": "move", "move_id": "spore", "legal": True}], rounds=())
        second = _record(turn_index=0, recorded_action_index=3, candidates=[{"action_index": 3, "kind": "move", "move_id": "tackle", "legal": True}], rounds=())

        with self.assertRaisesRegex(SourceRootReplayError, "ambiguous_source_record_for_turn"):
            source_bound_replay_prefix(target, source_records=(first, second))
