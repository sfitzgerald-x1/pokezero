"""Producer-side tests for the OPPONENT-MOVE JOURNAL.

Reverting ``src/pokezero/foulplay_bridge.py`` makes every test in this file FAIL
(the module cannot import ``OpponentJournalEntry``), which is a coarse signal. The
falsifiability that matters was established by MUTATION: 28 mutants introduced one
at a time into the producer, all 28 caught. Reproduce with the script recorded on
the PR.

MUTATE THE WIRING, NOT JUST THE LOGIC. The first version of this file mutated only
the pure functions (``_opponent_journal_for_result``, ``_last_addressed_round``,
``_request_digest``, the two ``to_dict``s) and scored 12/12 -- while SIX wiring
edges were completely untested, because every mode test passed ``opponent_journal=``
explicitly and nothing went through ``_config_from_args`` or ``_run_single_game``.
Two of the six were severe:

* flipping the recording gate ``!= "off"`` to ``== "full"`` makes the SHIPPED
  DEFAULT record nothing, ever -- suite still green;
* pinning ``_config_from_args`` to ``"off"`` makes ``--opponent-journal full`` a
  silent no-op in every production entry point -- suite still green.

``OpponentJournalWiringTest`` exists for that layer, and deliberately names no mode
it does not have to: the default-mode tests construct the config with no journal
argument at all.

Three tests are easy to write so that they cannot fail, and are written against that:

* ``test_stats_and_submitted_choices_are_identical_with_and_without_the_journal``
  asserts two runs are equal, which is trivially true if neither run journals. It
  therefore also asserts that the two runs DID differ in the intended way.
* ``test_journal_changes_no_address_and_no_occurrence_count`` scans a document that
  really contains ``fallback_samples``, mirrored under two paths as production
  shards are, and asserts the baseline is non-empty before comparing. Scanned
  against a journal-only document both sides are zero and it passes for free.
* ``test_addressed_ignores_malformed_and_boolean_rounds`` puts the real address at
  round 0. At round 1 the bool guard is untestable, because ``True`` coerces to 1
  and yields the same prefix -- that version of the test survived its mutant.
* ``test_run_single_game_carries_recorder_failures_to_the_result`` asserts a NON-ZERO
  failure count. Its sibling asserts 0 on a clean game, which stays true if the
  wiring drops the field entirely -- and did.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.actions import ACTION_COUNT
from pokezero.engine_search import EngineMctsStats
from pokezero.fallback_addresses import scan_corpus
from pokezero.foulplay_bridge import (
    DEFAULT_BATTLE_ID_PREFIX,
    OPPONENT_JOURNAL_SCHEMA_VERSION,
    ControlledFoulPlayBenchmarkResult,
    ControlledFoulPlayConfig,
    ControlledFoulPlayGameResult,
    OpponentJournalEntry,
    _ControlledBattleState,
    _config_from_args,
    _handle_decision_boundary,
    _last_addressed_round,
    _opponent_journal_for_result,
    _request_digest,
    _run_single_game,
    build_arg_parser,
)
from pokezero.policy import PolicyDecision
from pokezero.trajectory import BattleTrajectory
from pokezero.observation import PokeZeroObservationV0


REQUEST_P1 = '|request|{"active":[{"moves":[{"id":"icebeam"}]}],"side":{"id":"p1"},"rqid":3}'
REQUEST_P2 = '|request|{"active":[{"moves":[{"id":"thunderbolt"}]}],"side":{"id":"p2"},"rqid":3}'


class _PlayerState:
    def __init__(self, slot: str) -> None:
        self.slot = slot


class _Bridge:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send(self, payload: dict) -> None:
        self.messages.append(payload)


def _observation(slot: str) -> PokeZeroObservationV0:
    return PokeZeroObservationV0(
        categorical_ids=(),
        numeric_features=(),
        token_type_ids=(),
        attention_mask=(),
        # Fully legal: these tests exercise the journal, and a narrow mask would
        # make TrajectoryStep reject the actions under test for an unrelated reason.
        legal_action_mask=(True,) * ACTION_COUNT,
        metadata={"slot": slot},
    )


def _battle_state(battle_id: str = "battle-7") -> _ControlledBattleState:
    state = _ControlledBattleState(
        battle_id=battle_id,
        seed=7,
        format_id="gen3randombattle",
        trajectory=BattleTrajectory(battle_id=battle_id, format_id="gen3randombattle", seed=7),
    )
    state.request_lines["p1"] = REQUEST_P1
    state.request_lines["p2"] = REQUEST_P2
    return state


def _run_boundary(
    *,
    config: ControlledFoulPlayConfig,
    state: _ControlledBattleState,
    decision_round: int,
    requested_players: tuple[str, ...],
    foulplay_choice: str = "move icebeam",
    foulplay_action: int = 0,
    policy: object | None = None,
    select=None,
) -> _Bridge:
    """Drive one real ``_handle_decision_boundary`` over stubbed collaborators."""

    bridge = _Bridge()

    def default_select(_policy, _observation, _context, *, seed):
        return PolicyDecision(action_index=0, policy_id="stub")

    async def choice(**_kwargs) -> str:
        return foulplay_choice

    with (
        patch(
            "pokezero.foulplay_bridge._player_state",
            side_effect=lambda _state, slot, **_kwargs: _PlayerState(slot),
        ),
        patch(
            "pokezero.foulplay_bridge.observation_from_player_state",
            side_effect=lambda player_state, **_kwargs: _observation(player_state.slot),
        ),
        patch(
            "pokezero.foulplay_bridge._observation_with_search_metadata",
            side_effect=lambda value, _state: value,
        ),
        patch(
            "pokezero.foulplay_bridge._select_policy_decision",
            side_effect=select or default_select,
        ),
        patch(
            "pokezero.foulplay_bridge.showdown_choice_for_action",
            side_effect=lambda player_state, action: f"{player_state.slot}:{action}",
        ),
        patch("pokezero.foulplay_bridge._wait_for_foulplay_choice_or_exit", side_effect=choice),
        patch(
            "pokezero.foulplay_bridge.action_index_from_choice_string",
            side_effect=lambda _state, _choice: foulplay_action,
        ),
    ):
        asyncio.run(
            _handle_decision_boundary(
                config=config,
                bridge=bridge,  # type: ignore[arg-type]
                server=object(),
                state=state,
                policy=policy if policy is not None else object(),
                vocab=object(),
                dex=object(),
                observation_spec=SimpleNamespace(schema_version="v2.2"),
                decision_round=decision_round,
                requested_players=requested_players,
                foulplay_process=object(),
                foulplay_logs=object(),
            )
        )
    return bridge


def _config(**overrides) -> ControlledFoulPlayConfig:
    return ControlledFoulPlayConfig(
        checkpoint=Path("checkpoint.pt"),
        showdown_root=Path("/showdown"),
        **overrides,
    )


class _StubStatsPolicy:
    """A policy that owns a REAL ``EngineMctsStats``, as ``EngineMctsPolicy`` does."""

    def __init__(self) -> None:
        self.stats = EngineMctsStats()

    def address(self, battle_id: str, round_index: int, seat: str, key: str = "k") -> None:
        self.stats.fallback_samples.setdefault(key, []).append(
            {"battle_id": battle_id, "round": round_index, "seat": seat, "reason": "r"}
        )


class OpponentJournalRecordingTest(unittest.TestCase):
    def test_journal_records_the_choice_foul_play_actually_submitted(self) -> None:
        config = _config(pokezero_player="p2", opponent_journal="full")
        state = _battle_state()

        bridge = _run_boundary(
            config=config,
            state=state,
            decision_round=4,
            requested_players=("p1", "p2"),
            foulplay_choice="switch 3",
            foulplay_action=6,
        )

        self.assertEqual(len(state.opponent_journal), 1)
        entry = state.opponent_journal[0]
        self.assertEqual(entry.round, 4)
        self.assertEqual(entry.seat, "p1")
        self.assertEqual(entry.action, 6)
        # The journalled string is the SAME object that went to the BattleStream --
        # not a re-derivation from the protocol.
        self.assertEqual(entry.choice, "switch 3")
        self.assertEqual(bridge.messages[0]["choices"]["p1"], entry.choice)

    def test_journal_digests_the_raw_request_the_choice_answers(self) -> None:
        config = _config(pokezero_player="p2", opponent_journal="full")
        state = _battle_state()

        _run_boundary(
            config=config, state=state, decision_round=0, requested_players=("p1", "p2")
        )

        expected = hashlib.sha256(REQUEST_P1.encode("utf-8")).hexdigest()[:12]
        self.assertEqual(state.opponent_journal[0].request_sha256, expected)
        # The OPPONENT's request, never ours: keying on the wrong seat's request is a
        # silent mis-verification, not a crash.
        self.assertNotEqual(
            expected, hashlib.sha256(REQUEST_P2.encode("utf-8")).hexdigest()[:12]
        )

    def test_request_digest_is_empty_rather_than_fabricated_when_absent(self) -> None:
        self.assertEqual(_request_digest(None), "")
        self.assertEqual(_request_digest(""), "")
        self.assertEqual(len(_request_digest("|request|{}")), 12)

    def test_rounds_are_explicit_because_the_opponent_does_not_act_every_round(self) -> None:
        """Positional indexing desyncs on our own force-switch rounds."""
        config = _config(pokezero_player="p2", opponent_journal="full")
        state = _battle_state()

        _run_boundary(config=config, state=state, decision_round=0, requested_players=("p1", "p2"))
        # Round 1: OUR forced replacement. The opponent is not asked, so nothing is
        # journalled and round 1 is a hole in the sequence.
        _run_boundary(config=config, state=state, decision_round=1, requested_players=("p2",))
        _run_boundary(config=config, state=state, decision_round=2, requested_players=("p1", "p2"))

        self.assertEqual([entry.round for entry in state.opponent_journal], [0, 2])
        # The desync this pins: list position 1 holds round 2, so a consumer reading
        # by index would apply the round-2 move at round 1.
        self.assertNotEqual(state.opponent_journal[1].round, 1)

    def test_opponent_only_rounds_are_journalled(self) -> None:
        """A foul-play force-switch is its own decision round and must be kept."""
        config = _config(pokezero_player="p2", opponent_journal="full")
        state = _battle_state()

        _run_boundary(config=config, state=state, decision_round=0, requested_players=("p1", "p2"))
        _run_boundary(config=config, state=state, decision_round=1, requested_players=("p1",))

        self.assertEqual([entry.round for entry in state.opponent_journal], [0, 1])

    def test_seat_separates_the_same_seed_played_from_both_seats(self) -> None:
        """``foulplay_paired_eval`` runs one seed band from BOTH seats."""
        state_p1_opponent = _battle_state("battle-7800000")
        state_p2_opponent = _battle_state("battle-7800000")

        _run_boundary(
            config=_config(pokezero_player="p2", opponent_journal="full"),
            state=state_p1_opponent,
            decision_round=0,
            requested_players=("p1", "p2"),
        )
        _run_boundary(
            config=_config(pokezero_player="p1", opponent_journal="full"),
            state=state_p2_opponent,
            decision_round=0,
            requested_players=("p1", "p2"),
        )

        self.assertEqual(state_p1_opponent.opponent_journal[0].seat, "p1")
        self.assertEqual(state_p2_opponent.opponent_journal[0].seat, "p2")
        # Same battle_id, same round: only the seat tells the two apart.
        self.assertEqual(state_p1_opponent.battle_id, state_p2_opponent.battle_id)
        self.assertNotEqual(
            state_p1_opponent.opponent_journal[0], state_p2_opponent.opponent_journal[0]
        )

    def test_off_records_nothing_at_all(self) -> None:
        state = _battle_state()

        _run_boundary(
            config=_config(pokezero_player="p2", opponent_journal="off"),
            state=state,
            decision_round=0,
            requested_players=("p1", "p2"),
        )

        self.assertEqual(state.opponent_journal, [])

    def test_config_rejects_an_unknown_mode(self) -> None:
        with self.assertRaises(ValueError) as raised:
            _config(opponent_journal="sometimes")
        self.assertIn("opponent_journal", str(raised.exception))


class OpponentJournalModeTest(unittest.TestCase):
    JOURNAL = tuple(
        OpponentJournalEntry(round=index, seat="p1", choice="move 1", action=0, request_sha256="a" * 12)
        for index in range(10)
    )

    def test_full_emits_every_recorded_round(self) -> None:
        emitted = _opponent_journal_for_result(
            self.JOURNAL, mode="full", policy=_StubStatsPolicy(), battle_id="battle-7"
        )
        self.assertEqual(emitted, self.JOURNAL)

    def test_off_emits_nothing_even_with_addresses(self) -> None:
        policy = _StubStatsPolicy()
        policy.address("battle-7", 5, "p2")
        emitted = _opponent_journal_for_result(
            self.JOURNAL, mode="off", policy=policy, battle_id="battle-7"
        )
        self.assertEqual(emitted, ())

    def test_addressed_keeps_the_prefix_through_the_last_addressed_round(self) -> None:
        policy = _StubStatsPolicy()
        policy.address("battle-7", 2, "p2", key="early")
        policy.address("battle-7", 5, "p2", key="late")

        emitted = _opponent_journal_for_result(
            self.JOURNAL, mode="addressed", policy=policy, battle_id="battle-7"
        )

        self.assertEqual([entry.round for entry in emitted], [0, 1, 2, 3, 4, 5])

    def test_addressed_emits_nothing_for_a_battle_with_no_address(self) -> None:
        policy = _StubStatsPolicy()
        policy.address("battle-OTHER", 5, "p2")

        emitted = _opponent_journal_for_result(
            self.JOURNAL, mode="addressed", policy=policy, battle_id="battle-7"
        )

        self.assertEqual(emitted, ())

    def test_addressed_emits_nothing_when_the_policy_keeps_no_addresses(self) -> None:
        """The raw and root-puct arms: no address store, so nothing to replay to."""
        emitted = _opponent_journal_for_result(
            self.JOURNAL, mode="addressed", policy=object(), battle_id="battle-7"
        )
        self.assertEqual(emitted, ())

    def test_addressed_ignores_malformed_and_boolean_rounds(self) -> None:
        """The real address is at round 0 ON PURPOSE.

        A first version of this test put it at round 1, where ``True`` -- which is
        an ``int`` in Python and coerces to 1 -- produced the same prefix as the
        real address. Dropping the bool guard left the test green. The real address
        must sit BELOW every malformed value or the guard is untested.
        """
        policy = _StubStatsPolicy()
        policy.stats.fallback_samples["junk"] = [
            {"battle_id": "battle-7", "round": True, "seat": "p2"},
            {"battle_id": "battle-7", "round": "9", "seat": "p2"},
            {"battle_id": "battle-7", "round": 4.0, "seat": "p2"},
            "not-a-mapping",
        ]
        policy.address("battle-7", 0, "p2")

        emitted = _opponent_journal_for_result(
            self.JOURNAL, mode="addressed", policy=policy, battle_id="battle-7"
        )

        # `True` would read as round 1; "9" and 4.0 would either crash a max() or
        # extend the prefix. None may reach past the one real address at round 0.
        self.assertEqual([entry.round for entry in emitted], [0])


class OpponentJournalPayloadTest(unittest.TestCase):
    def _result(self, *, mode: str, games) -> dict:
        return ControlledFoulPlayBenchmarkResult(
            config=_config(games=max(len(games), 1), opponent_journal=mode),
            policy_id="stub",
            games=tuple(games),
        ).to_dict()

    def _game(
        self, battle_id: str, *, emitted: int, recorded: int, failures: int = 0
    ) -> ControlledFoulPlayGameResult:
        return ControlledFoulPlayGameResult(
            battle_id=battle_id,
            seed=1,
            winner=None,
            pokezero_won=False,
            decision_rounds=recorded,
            pokezero_decisions=recorded,
            root_puct_searches=0,
            root_puct_fallbacks=0,
            opponent_journal=tuple(
                OpponentJournalEntry(
                    round=index, seat="p2", choice="move 1", action=0, request_sha256="b" * 12
                )
                for index in range(emitted)
            ),
            opponent_journal_recorded=recorded,
            opponent_journal_failures=failures,
        )

    def test_header_reports_mode_and_the_truncation_as_numbers(self) -> None:
        """`failures` is NON-ZERO on purpose.

        With an all-zero fixture the `record_failures` assertion compares 0 against
        0 and survives the summation being replaced by a literal 0 -- which is the
        only path by which the count reaches JSON at all, since the per-game row
        omits the key when it is zero.
        """
        payload = self._result(
            mode="addressed",
            games=[
                self._game("a", emitted=3, recorded=20, failures=2),
                self._game("b", emitted=0, recorded=15, failures=1),
            ],
        )

        self.assertEqual(
            payload["opponent_journal"],
            {
                "schema_version": OPPONENT_JOURNAL_SCHEMA_VERSION,
                "mode": "addressed",
                "entries_key": "opponent_moves",
                "recorded_decisions": 35,
                "emitted_decisions": 3,
                "games_with_journal": 1,
                "record_failures": 3,
            },
        )

    def test_rows_name_the_battle_that_lost_rounds(self) -> None:
        """A header total cannot say WHICH battle is unreplayable; the row can."""
        payload = self._result(
            mode="full",
            games=[
                self._game("clean", emitted=4, recorded=4),
                self._game("lossy", emitted=3, recorded=3, failures=2),
            ],
        )

        rows = {row["battle_id"]: row for row in payload["game_results"]}
        self.assertNotIn("opponent_moves_record_failures", rows["clean"])
        self.assertEqual(rows["lossy"]["opponent_moves_record_failures"], 2)
        # The header total still agrees with the rows it summarises.
        self.assertEqual(payload["opponent_journal"]["record_failures"], 2)

    def test_header_is_present_when_journaling_is_off(self) -> None:
        """Off, on-with-no-addresses and too-old-to-know must be distinguishable."""
        payload = self._result(mode="off", games=[self._game("a", emitted=0, recorded=0)])

        self.assertEqual(payload["opponent_journal"]["mode"], "off")
        self.assertEqual(payload["opponent_journal"]["emitted_decisions"], 0)

    def test_game_rows_carry_the_journal_and_omit_it_when_empty(self) -> None:
        payload = self._result(
            mode="addressed",
            games=[self._game("a", emitted=2, recorded=9), self._game("b", emitted=0, recorded=9)],
        )

        rows = payload["game_results"]
        self.assertEqual(
            rows[0]["opponent_moves"],
            [
                {"round": 0, "seat": "p2", "choice": "move 1", "action": 0, "request_sha256": "b" * 12},
                {"round": 1, "seat": "p2", "choice": "move 1", "action": 0, "request_sha256": "b" * 12},
            ],
        )
        self.assertNotIn("opponent_moves", rows[1])
        self.assertEqual(rows[1]["opponent_moves_recorded"], 9)
        # The header key and the row key are DIFFERENT names on purpose, and the
        # header says which is which. One name with two JSON shapes (mapping at the
        # root, list in the rows) breaks the recursive-by-name lookup idiom that
        # `fallback_addresses` uses and every consumer copies.
        self.assertEqual(payload["opponent_journal"]["entries_key"], "opponent_moves")
        self.assertIsInstance(payload["opponent_journal"], dict)
        self.assertIsInstance(rows[0]["opponent_moves"], list)

    def test_the_summary_round_trips_through_json(self) -> None:
        payload = self._result(mode="full", games=[self._game("a", emitted=2, recorded=2)])
        reloaded = json.loads(json.dumps(payload, sort_keys=True))
        self.assertEqual(reloaded["game_results"][0]["opponent_moves"][1]["round"], 1)


class FallbackAddressReaderInvariantTest(unittest.TestCase):
    """The journal must be invisible to ``pokezero.fallback_addresses``.

    That reader accepts a mapping as a CUMULATIVE stats scope iff it contains
    ``fallback_samples``, and harvests addresses from every mapping so named. Adding
    a sibling block must not create a scope, an address, or an occurrence count.
    """

    def _shard(self, *, journal: bool) -> dict:
        game = ControlledFoulPlayGameResult(
            battle_id="battle-7",
            seed=7,
            winner=None,
            pokezero_won=False,
            decision_rounds=4,
            pokezero_decisions=4,
            root_puct_searches=0,
            root_puct_fallbacks=0,
            opponent_journal=tuple(
                OpponentJournalEntry(
                    round=index, seat="p2", choice="move 1", action=0, request_sha256="c" * 12
                )
                for index in range(4)
            )
            if journal
            else (),
            opponent_journal_recorded=4 if journal else 0,
        )
        stats = EngineMctsStats()
        stats.decisions = 4
        stats.fallback_decisions = 2
        stats.fallback_reasons["crate_search_failed"] = 2
        stats.world_failure_reasons["volatile_unsupported: ['perish0']"] = 6
        stats.fallback_samples["crate_search_failed"] = [
            {"battle_id": "battle-7", "round": 1, "seat": "p1", "reason": "crate_search_failed"},
            {"battle_id": "battle-7", "round": 3, "seat": "p1", "reason": "crate_search_failed"},
        ]
        payload = ControlledFoulPlayBenchmarkResult(
            config=_config(opponent_journal="full" if journal else "off"),
            policy_id="stub",
            games=(game,),
            policy_stats=stats.to_dict(),
        ).to_dict()
        # Mirror the real shard shape so the baseline is NON-EMPTY and the reader's
        # own de-duplication is exercised: production shards carry the same
        # `to_dict()` under `engine_mcts.policy_stats` and again under
        # `per_seat[seat].policy_stats`, and `_scan_document` de-duplicates whole
        # blocks by content. Two copies here means a journal that leaked into the
        # stats dict unevenly would show up as a doubled count rather than passing.
        payload["engine_mcts"] = {"policy_stats": stats.to_dict()}
        payload["per_seat"] = {"p1": {"policy_stats": stats.to_dict()}}
        return payload

    def _scan(self, document: dict, directory: Path, name: str):
        path = directory / name
        path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
        return scan_corpus([path])

    def test_journal_changes_no_address_and_no_occurrence_count(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            plain = self._scan(self._shard(journal=False), directory, "plain.json")
            journalled = self._scan(self._shard(journal=True), directory, "journalled.json")

        # Baseline is real, not vacuous.
        self.assertEqual(len(plain.addresses), 2)
        self.assertEqual(plain.decision_counts["fallback:crate_search_failed"], 2)

        self.assertEqual(
            [(a.battle_id, a.round, a.seat, a.key) for a in journalled.addresses],
            [(a.battle_id, a.round, a.seat, a.key) for a in plain.addresses],
        )
        self.assertEqual(journalled.decision_counts, plain.decision_counts)
        self.assertEqual(journalled.world_counts, plain.world_counts)
        self.assertEqual(journalled.addresses_dropped, plain.addresses_dropped)
        self.assertEqual(journalled.shards_read, plain.shards_read)
        self.assertEqual(journalled.shards_unreadable, 0)

    def test_the_journal_key_is_not_named_like_a_stats_scope(self) -> None:
        """A literal guard on the one name the reader dispatches on."""
        document = self._shard(journal=True)
        row = document["game_results"][0]
        self.assertIn("opponent_moves", row)
        self.assertNotIn("fallback_samples", row)
        self.assertNotIn("fallback_samples", document["opponent_journal"])


class SearchBehaviourUnchangedTest(unittest.TestCase):
    """Journaling on and off must produce the same run.

    The strong form -- identical ``EngineMctsStats.to_dict()`` -- holds by
    construction: the journal is written in ``foulplay_bridge`` and never touches
    ``engine_search``. This drives the real decision boundary twice to pin the
    weaker but falsifiable claims that go with it: the policy sees identical
    arguments and an identically-positioned RNG, and the same choices are submitted.
    """

    def _run(self, mode: str) -> dict:
        config = _config(pokezero_player="p2", opponent_journal=mode)
        state = _battle_state()
        policy = _StubStatsPolicy()
        seen: list[tuple] = []

        def select(_policy, _observation, context, *, seed):
            # A real, seeded draw, positioned exactly where the searcher's would be.
            rng = random.Random(f"{seed}:{context.player_id}:{context.decision_round_index}")
            draw = rng.getrandbits(63)
            seen.append((context.decision_round_index, context.player_id, seed, draw))
            policy.stats.decisions += 1
            policy.stats.total_iterations += draw % 97
            policy.stats.fallback_samples.setdefault("k", []).append(
                {
                    "battle_id": state.battle_id,
                    "round": context.decision_round_index,
                    "seat": context.player_id,
                    "reason": "r",
                }
            )
            return PolicyDecision(action_index=draw % 4, policy_id="stub")

        bridges = []
        for round_index in range(3):
            bridges.append(
                _run_boundary(
                    config=config,
                    state=state,
                    decision_round=round_index,
                    requested_players=("p1", "p2"),
                    foulplay_choice=f"move {round_index}",
                    foulplay_action=round_index % 3,
                    policy=policy,
                    select=select,
                )
            )
        return {
            "stats": policy.stats.to_dict(),
            "policy_calls": seen,
            "submitted": [bridge.messages[0]["choices"] for bridge in bridges],
            "trajectory": [
                (step.player_id, step.turn_index, step.action_index)
                for step in state.trajectory.steps
            ],
            "journal": [entry.to_dict() for entry in state.opponent_journal],
        }

    def test_stats_and_submitted_choices_are_identical_with_and_without_the_journal(self) -> None:
        off = self._run("off")
        full = self._run("full")

        self.assertEqual(off["stats"], full["stats"])
        self.assertEqual(off["policy_calls"], full["policy_calls"])
        self.assertEqual(off["submitted"], full["submitted"])
        self.assertEqual(off["trajectory"], full["trajectory"])
        # And the run really did differ in the one intended way -- otherwise the
        # equalities above are being asserted about two identical no-ops.
        self.assertEqual(off["journal"], [])
        self.assertEqual(len(full["journal"]), 3)

    def test_the_stats_serializer_gained_no_journal_field(self) -> None:
        """`EngineMctsStats.to_dict()` is a frozen consumer contract."""
        self.assertNotIn("opponent_journal", EngineMctsStats().to_dict())


class OpponentJournalWiringTest(unittest.TestCase):
    """The layer between the pure functions and production.

    Independent review broke five wiring edges one at a time and the suite stayed
    green on all five, because every mode test above passed ``opponent_journal=``
    explicitly and none of them went through ``_config_from_args`` or
    ``_run_single_game``. Two of the survivors were severe: flipping the recording
    gate from ``!= "off"`` to ``== "full"`` makes the SHIPPED DEFAULT record nothing
    ever, and pinning ``_config_from_args`` to ``"off"`` makes ``--opponent-journal
    full`` a silent no-op in every production entry point.

    So: no test in this class names a mode it does not have to. The default-mode
    tests construct ``ControlledFoulPlayConfig`` with NO journal argument on purpose.
    """

    def test_the_shipped_default_records(self) -> None:
        """The default config, untouched, must journal. Kills `!= "off"` -> `== "full"`."""
        config = _config(pokezero_player="p2")  # no opponent_journal argument
        self.assertEqual(config.opponent_journal, "addressed")
        state = _battle_state()

        _run_boundary(
            config=config, state=state, decision_round=0, requested_players=("p1", "p2")
        )

        self.assertEqual(len(state.opponent_journal), 1)
        self.assertEqual(state.opponent_journal[0].choice, "move icebeam")

    def test_config_from_args_carries_the_flag_into_production(self) -> None:
        """`_config_from_args` is the SOLE production path to a config."""
        parser = build_arg_parser()
        base = ["--checkpoint", "c.pt", "--showdown-root", "/s"]

        default = _config_from_args(parser.parse_args(base))
        explicit = _config_from_args(parser.parse_args(base + ["--opponent-journal", "full"]))
        disabled = _config_from_args(parser.parse_args(base + ["--opponent-journal", "off"]))

        self.assertEqual(default.opponent_journal, "addressed")
        self.assertEqual(explicit.opponent_journal, "full")
        self.assertEqual(disabled.opponent_journal, "off")

    def _play_one_game(self, *, config, policy, rounds: int, failing_rounds=()):
        """Drive the real `_run_single_game` over stubbed collaborators."""
        seed = 7
        battle_id = f"{DEFAULT_BATTLE_ID_PREFIX}-{seed}"
        events = [{"battleId": battle_id, "type": "ready", "requested": ["p1", "p2"]}
                  for _ in range(rounds)]
        events.append({"battleId": battle_id, "type": "terminal"})
        pending = iter(events)

        class Server:
            async def send_room_lines(self, *_args, **_kwargs) -> None:
                return None

        class Bridge:
            def __init__(self) -> None:
                self.sent: list[dict] = []

            async def send(self, payload: dict) -> None:
                self.sent.append(payload)

            async def next_event(self) -> dict:
                return next(pending)

        async def boundary(*, state, decision_round, **_kwargs):
            # Stand in for the recording site, whose own behaviour is pinned above.
            if decision_round in failing_rounds:
                state.opponent_journal_failures += 1
                return None
            state.opponent_journal.append(
                OpponentJournalEntry(round=decision_round, seat=config.foulplay_player,
                                     choice="move 1", action=0, request_sha256="d" * 12)
            )
            return None

        async def notify(**_kwargs) -> None:
            return None

        with (
            patch("pokezero.foulplay_bridge._handle_decision_boundary", side_effect=boundary),
            patch("pokezero.foulplay_bridge._notify_foulplay_terminal", side_effect=notify),
        ):
            return asyncio.run(
                _run_single_game(
                    config=config,
                    bridge=Bridge(),  # type: ignore[arg-type]
                    server=Server(),  # type: ignore[arg-type]
                    policy=policy,
                    vocab=object(),
                    dex=object(),
                    observation_spec=SimpleNamespace(schema_version="v2.2"),
                    seed=seed,
                    foulplay_process=object(),  # type: ignore[arg-type]
                    foulplay_logs=object(),
                )
            )

    def test_run_single_game_applies_the_configured_mode_to_the_real_battle_id(self) -> None:
        """Kills mode-pinned-to-full, battle_id-wrong, and recorded-count-zeroed."""
        seed = 7
        battle_id = f"{DEFAULT_BATTLE_ID_PREFIX}-{seed}"
        policy = _StubStatsPolicy()
        # Address at round 2 of THIS battle, acting seat = our seat.
        policy.address(battle_id, 2, "p1")

        result = self._play_one_game(
            config=_config(pokezero_player="p1", opponent_journal="addressed"),
            policy=policy,
            rounds=6,
        )

        self.assertEqual(result.battle_id, battle_id)
        # Truncated to the addressed prefix: mode really was read from the config...
        self.assertEqual([entry.round for entry in result.opponent_journal], [0, 1, 2])
        # ...against the real battle id (a wrong id empties this)...
        self.assertTrue(result.opponent_journal)
        # ...and the recorded count is the FULL observed total, not the emitted one,
        # so the truncation is legible rather than invisible.
        self.assertEqual(result.opponent_journal_recorded, 6)
        self.assertEqual(result.opponent_journal_failures, 0)

    def test_run_single_game_carries_recorder_failures_to_the_result(self) -> None:
        """A lost round must reach the shard. Kills `opponent_journal_failures=0`.

        The sibling test asserts this field is 0 on a clean game, which is true for
        free if the wiring drops it -- so the non-zero case is the test.
        """
        result = self._play_one_game(
            config=_config(pokezero_player="p1", opponent_journal="full"),
            policy=_StubStatsPolicy(),
            rounds=5,
            failing_rounds={1, 3},
        )

        self.assertEqual(result.opponent_journal_failures, 2)
        self.assertEqual([entry.round for entry in result.opponent_journal], [0, 2, 4])
        self.assertEqual(result.opponent_journal_recorded, 3)

    def test_run_single_game_passes_our_acting_seat_to_the_address_lookup(self) -> None:
        """Kills `seat=None` at the `_run_single_game` call site.

        The seat filter is inert in production (one `pokezero_player` per bridge
        invocation, and `battle_id` embeds a per-invocation seed), so this feeds the
        lookup an address whose acting seat is NOT ours -- a state the bridge cannot
        itself produce. That is deliberate: the point is to pin the wiring edge, not
        to claim the input is reachable. Without the seat argument the foreign
        address extends the prefix and the journal is emitted anyway.
        """
        seed = 7
        battle_id = f"{DEFAULT_BATTLE_ID_PREFIX}-{seed}"
        policy = _StubStatsPolicy()
        policy.address(battle_id, 3, "p2")  # acting seat p2; we are p1

        result = self._play_one_game(
            config=_config(pokezero_player="p1", opponent_journal="addressed"),
            policy=policy,
            rounds=6,
        )

        self.assertEqual(result.opponent_journal, ())
        # ...and the same battle WITH our own acting seat does emit, so the empty
        # result above is the seat filter and not a broken fixture.
        policy.address(battle_id, 1, "p1")
        emitting = self._play_one_game(
            config=_config(pokezero_player="p1", opponent_journal="addressed"),
            policy=policy,
            rounds=6,
        )
        self.assertEqual([entry.round for entry in emitting.opponent_journal], [0, 1])

    def test_run_single_game_honours_off(self) -> None:
        policy = _StubStatsPolicy()
        policy.address(f"{DEFAULT_BATTLE_ID_PREFIX}-7", 2, "p1")

        result = self._play_one_game(
            config=_config(pokezero_player="p1", opponent_journal="off"),
            policy=policy,
            rounds=4,
        )

        self.assertEqual(result.opponent_journal, ())

    def test_run_single_game_full_keeps_every_round(self) -> None:
        result = self._play_one_game(
            config=_config(pokezero_player="p1", opponent_journal="full"),
            policy=_StubStatsPolicy(),
            rounds=4,
        )

        self.assertEqual([entry.round for entry in result.opponent_journal], [0, 1, 2, 3])
        self.assertEqual(result.opponent_journal_recorded, 4)

    def test_a_recorder_failure_is_counted_and_never_loses_the_battle(self) -> None:
        """Telemetry on the live decision path, on by default: it must not throw."""
        config = _config(pokezero_player="p2")
        state = _battle_state()

        with patch(
            "pokezero.foulplay_bridge.OpponentJournalEntry",
            side_effect=RuntimeError("boom"),
        ):
            bridge = _run_boundary(
                config=config, state=state, decision_round=0, requested_players=("p1", "p2")
            )

        # The battle went on and the choices were still submitted.
        self.assertEqual(bridge.messages[0]["choices"]["p1"], "move icebeam")
        # And the loss is a number, not a gap.
        self.assertEqual(state.opponent_journal, [])
        self.assertEqual(state.opponent_journal_failures, 1)

    def test_addressed_ignores_an_address_from_another_acting_seat(self) -> None:
        """`_last_addressed_round` completes the locator, not half of it.

        Inert through the bridge (one `pokezero_player` per invocation), so it is
        tested directly. Documented as defence in the function's docstring.
        """
        policy = _StubStatsPolicy()
        policy.address("battle-7", 5, "p2")

        self.assertEqual(_last_addressed_round(policy, "battle-7", "p1"), None)
        self.assertEqual(_last_addressed_round(policy, "battle-7", "p2"), 5)
        # Unfiltered still works, for callers that have no seat to give.
        self.assertEqual(_last_addressed_round(policy, "battle-7"), 5)


class OpponentJournalCliTest(unittest.TestCase):
    def test_default_is_addressed(self) -> None:
        args = build_arg_parser().parse_args(
            ["--checkpoint", "c.pt", "--showdown-root", "/s"]
        )
        self.assertEqual(args.opponent_journal, "addressed")

    def test_mode_is_selectable_and_validated(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "--checkpoint", "c.pt", "--showdown-root", "/s",
                "--opponent-journal", "full",
            ]
        )
        self.assertEqual(args.opponent_journal, "full")
        with self.assertRaises(SystemExit):
            build_arg_parser().parse_args(
                [
                    "--checkpoint", "c.pt", "--showdown-root", "/s",
                    "--opponent-journal", "sometimes",
                ]
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
