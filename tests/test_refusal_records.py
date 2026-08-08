"""Producer-side tests for the always-on REFUSAL RECORDER in the bridge.

The recorder itself (``fallback_replay``) is covered by ``test_fallback_replay.py``
and by the end-to-end chain. What is NEW here is the WIRING, and this file exists
because wiring is what the last three features in this area shipped untested.

MUTATE THE WIRING, NOT JUST THE PURE FUNCTIONS. ``#1188``'s own test file records
that its first revision scored 12/12 on the pure functions while six wiring edges
were completely untested, two of them severe enough to make the shipped default a
silent no-op. The same four edges exist here and each has a test that FAILS when
the edge is broken:

* the attach call in ``_run_controlled_foulplay_games`` deleted
  -> ``test_the_shipped_default_attaches_the_recorder``;
* ``record_refusals`` default flipped to ``False``
  -> ``test_the_shipped_default_attaches_the_recorder``,
     ``test_config_from_args_carries_the_flag_into_production``;
* the records dropped from the game payload
  -> ``test_run_single_game_files_records_under_the_real_battle_id``,
     ``test_the_game_row_carries_the_records``;
* the health block dropped from the summary payload
  -> ``test_the_summary_always_carries_the_health_header``,
     ``test_instrument_errors_reach_the_summary``,
     ``test_a_partial_progress_summary_reports_health_too``.

Tests that are easy to write so that they cannot fail, and what is done about it:

* ``test_stats_are_identical_with_and_without_the_recorder`` asserts two runs are
  equal, which is trivially true if neither run records. It therefore also asserts
  the two runs DID differ in the intended way, and it compares the WHOLE
  ``EngineMctsStats.to_dict()`` (69 keys on this tree), not a chosen subset --
  ``#1180``'s own end-to-end comparison covers 7.
* ``test_records_change_no_address_and_no_occurrence_count`` scans a document that
  really contains ``fallback_samples``, mirrored under two paths as production
  shards are, and asserts the baseline is NON-EMPTY before comparing. Against a
  records-only document both sides are zero and it passes for free.
* ``test_health_is_not_trustworthy_when_it_was_never_reported`` is the null-world
  test for the health block: the wrong implementation is ``trustworthy = not
  errors``, which is True for a run that never attached, never ran, and never had
  an error channel. Every conjunct is therefore falsified on its own.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pokezero.actions import ACTION_COUNT
from pokezero.engine_search import EngineMctsStats
from pokezero.fallback_addresses import scan_corpus
from pokezero.fallback_replay import RefusalRecord, attach_refusal_recorder
from pokezero.foulplay_bridge import (
    DEFAULT_BATTLE_ID_PREFIX,
    REFUSAL_RECORDER_SCHEMA_VERSION,
    ControlledFoulPlayBenchmarkResult,
    ControlledFoulPlayConfig,
    ControlledFoulPlayGameResult,
    RefusalRecorderHealth,
    _config_from_args,
    _RefusalCapture,
    _run_controlled_foulplay_games,
    _run_single_game,
    build_arg_parser,
)
from pokezero.observation import PokeZeroObservationV0


def _config(**overrides) -> ControlledFoulPlayConfig:
    return ControlledFoulPlayConfig(
        checkpoint=Path("checkpoint.pt"), showdown_root=Path("/showdown"), **overrides
    )


TRAPPED_KEY = (
    "self_request_state_unsupported: self active request flags ['trapped'] constrain "
    "legality beyond this construction (sampled world does not trap: foe ability 'swarm')"
)


class _Context:
    """A ``PolicyContext``-shaped stand-in, with a REAL observation.

    The observation matters: ``_request_legal_choices`` reads
    ``metadata["action_candidates"]`` filtered by ``legal_action_mask``, and a bare
    ``SimpleNamespace`` would exercise the early-return rather than the mapping.
    """

    def __init__(self, battle_id: str, round_index: int, seat: str, seed: int = 7) -> None:
        self.battle_id = battle_id
        self.decision_round_index = round_index
        self.player_id = seat
        self.seed = seed
        self.observation = PokeZeroObservationV0(
            categorical_ids=(),
            numeric_features=(),
            token_type_ids=(),
            attention_mask=(),
            legal_action_mask=(True,) * ACTION_COUNT,
            metadata={
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "thunderbolt"},
                    {"action_index": 1, "kind": "move", "legal": True, "move_id": "painsplit"},
                    {
                        "action_index": 4,
                        "kind": "switch",
                        "legal": True,
                        "pokemon": {"species": "Misdreavus"},
                    },
                ]
            },
        )


class _EnginePolicyStub:
    """Shaped like ``EngineMctsPolicy`` where the recorder touches it.

    A REAL ``EngineMctsStats``, and the three methods the recorder wraps with their
    real signatures. The recorder under test is the real one; only the search is
    stubbed, so a wiring test exercises the same attach path production takes.
    """

    def __init__(self) -> None:
        self.stats = EngineMctsStats()
        self.policy_id = "engine-mcts-stub"

    def select_action_with_context(self, context, *, rng):
        return None

    def _map_choices(self, context, aggregated):
        return None

    def _fallback(self, context, rng, reason):
        self.stats.fallback_decisions += 1
        self.stats.fallback_reasons[reason] += 1
        self.stats.fallback_samples.setdefault(f"fallback:{reason}", []).append(
            {
                "battle_id": context.battle_id,
                "round": context.decision_round_index,
                "seat": context.player_id,
                "reason": reason,
            }
        )
        return None

    # -- scripted decisions, in the real order --

    def refuse(
        self,
        context,
        *,
        reason: str = "no_worlds_constructed",
        world_failures: dict | None = None,
        worlds: tuple[int, int, int] = (16, 0, 0),
        aggregated: dict | None = None,
        unmapped: dict | None = None,
    ) -> None:
        self.select_action_with_context(context, rng=None)
        for key, count in (world_failures or {TRAPPED_KEY: 16}).items():
            self.stats.world_failure_reasons[key] += count
        attempted, constructed, searched = worlds
        self.stats.worlds_attempted += attempted
        self.stats.worlds_constructed += constructed
        self.stats.worlds_searched += searched
        if aggregated is not None:
            self._map_choices(context, aggregated)
        for key, count in (unmapped or {}).items():
            self.stats.unmapped_choices[key] += count
        self._fallback(context, None, reason)

    def succeed(self, context) -> None:
        self.select_action_with_context(context, rng=None)
        self.stats.decisions += 1
        self.stats.worlds_attempted += 4
        self.stats.worlds_constructed += 4
        self.stats.worlds_searched += 4


class RefusalRecorderHealthTest(unittest.TestCase):
    """Silence must never read as OK."""

    def test_health_is_not_trustworthy_when_it_was_never_reported(self) -> None:
        # The plausible wrong implementation is `trustworthy = not errors`, which
        # this default satisfies while describing a run that never recorded.
        self.assertFalse(RefusalRecorderHealth().trustworthy)
        self.assertFalse(RefusalRecorderHealth(enabled=True).trustworthy)

    def test_every_conjunct_of_trustworthy_is_load_bearing(self) -> None:
        good = RefusalRecorderHealth(enabled=True, attached=True, health_reported=True)
        self.assertTrue(good.trustworthy)
        # ...and each one, alone, takes it away.
        self.assertFalse(
            RefusalRecorderHealth(enabled=True, attached=False, health_reported=True).trustworthy
        )
        self.assertFalse(
            RefusalRecorderHealth(enabled=True, attached=True, health_reported=False).trustworthy
        )
        self.assertFalse(
            RefusalRecorderHealth(
                enabled=True, attached=True, health_reported=True, instrument_errors=("boom",)
            ).trustworthy
        )
        self.assertFalse(
            RefusalRecorderHealth(
                enabled=True, attached=True, health_reported=True, degraded_records=1
            ).trustworthy
        )

    def test_a_failed_attach_says_why(self) -> None:
        """A bare False sends the reader to the source; the message does not."""
        capture = _RefusalCapture(SimpleNamespace(), enabled=True)
        health = capture.health()

        self.assertTrue(health.enabled)
        self.assertFalse(health.attached)
        self.assertIn("stats", health.attach_error or "")
        self.assertFalse(health.trustworthy)
        # And it did not raise -- a diagnostic on by default must never be able to
        # abort a strength benchmark on the raw or root-PUCT arm.
        self.assertEqual(capture.records_for("battle-1"), ())

    def test_disabled_is_distinguishable_from_broken(self) -> None:
        off = _RefusalCapture(_EnginePolicyStub(), enabled=False).health()
        broken = _RefusalCapture(SimpleNamespace(), enabled=True).health()

        self.assertEqual((off.enabled, off.attached, off.attach_error), (False, False, None))
        self.assertTrue(broken.enabled)
        self.assertIsNotNone(broken.attach_error)

    def test_capture_partitions_records_by_battle(self) -> None:
        policy = _EnginePolicyStub()
        capture = _RefusalCapture(policy, enabled=True)
        try:
            policy.refuse(_Context("battle-1", 3, "p1"))
            policy.refuse(_Context("battle-2", 5, "p1"))
            policy.refuse(_Context("battle-2", 6, "p1"))

            self.assertEqual([r.round for r in capture.records_for("battle-1")], [3])
            self.assertEqual([r.round for r in capture.records_for("battle-2")], [5, 6])
            self.assertEqual(capture.records_for("battle-3"), ())
            self.assertEqual(capture.health().recorded_refusals, 3)
        finally:
            capture.detach()

    def test_detach_restores_the_policy(self) -> None:
        policy = _EnginePolicyStub()
        capture = _RefusalCapture(policy, enabled=True)
        capture.detach()
        policy.refuse(_Context("battle-1", 1, "p1"))

        # Detached: the decision still happened, and nothing was recorded.
        self.assertEqual(policy.stats.fallback_decisions, 1)
        self.assertEqual(capture.records_for("battle-1"), ())
        # ...and the health it reports is of the records it DID take, honestly zero.
        self.assertEqual(capture.health().recorded_refusals, 0)


class RefusalRecordPayloadTest(unittest.TestCase):
    def _result(self, *, records=(), health=None) -> dict:
        game = ControlledFoulPlayGameResult(
            battle_id="battle-7",
            seed=7,
            winner=None,
            pokezero_won=False,
            decision_rounds=4,
            pokezero_decisions=4,
            root_puct_searches=0,
            root_puct_fallbacks=0,
            refusal_records=tuple(records),
        )
        return ControlledFoulPlayBenchmarkResult(
            config=_config(), policy_id="stub", games=(game,), refusal_recorder=health
        ).to_dict()

    def test_the_summary_always_carries_the_health_header(self) -> None:
        """Absent, empty and off are three states; a reader must tell them apart."""
        payload = self._result()
        header = payload["refusal_recorder"]

        self.assertEqual(header["schema_version"], REFUSAL_RECORDER_SCHEMA_VERSION)
        self.assertEqual(header["records_key"], "refusals")
        # No health reported -> UNKNOWN, never clean.
        self.assertFalse(header["health_reported"])
        self.assertFalse(header["trustworthy"])

    def test_the_game_row_carries_the_records(self) -> None:
        record = RefusalRecord(
            battle_id="battle-7",
            round=3,
            seat="p1",
            reason="no_worlds_constructed",
            world_failures={TRAPPED_KEY: 16},
            worlds_attempted=16,
            request_legal_choices=("thunderbolt", "painsplit"),
        )
        row = self._result(records=[record])["game_results"][0]

        self.assertEqual(len(row["refusals"]), 1)
        self.assertEqual(row["refusals"][0]["round"], 3)
        self.assertEqual(row["refusals"][0]["world_failures"], {TRAPPED_KEY: 16})
        self.assertEqual(row["refusals"][0]["request_legal_choices"], ["thunderbolt", "painsplit"])

    def test_a_row_with_no_refusals_omits_the_key(self) -> None:
        self.assertNotIn("refusals", self._result()["game_results"][0])

    def test_health_numbers_reach_the_header(self) -> None:
        health = RefusalRecorderHealth(
            enabled=True,
            attached=True,
            health_reported=True,
            instrument_errors=("TypeError: boom",),
            degraded_records=2,
            recorded_refusals=9,
        )
        header = self._result(health=health)["refusal_recorder"]

        self.assertEqual(header["instrument_errors"], ["TypeError: boom"])
        self.assertEqual(header["degraded_records"], 2)
        self.assertEqual(header["recorded_refusals"], 9)
        self.assertFalse(header["trustworthy"])

    def test_the_summary_round_trips_through_json(self) -> None:
        record = RefusalRecord(
            battle_id="battle-7",
            round=3,
            seat="p1",
            reason="choices_unmapped",
            engine_choices={"substitute": 0.7},
            map_choices_calls=({"substitute": 0.7},),
            unmapped_choices={"substitute": 1},
            choices_unmapped_causes={"all_unmapped_legality_mismatch": 1},
            decision_rng_seed="7:p1:3",
        )
        payload = self._result(
            records=[record],
            health=RefusalRecorderHealth(enabled=True, attached=True, health_reported=True),
        )
        self.assertEqual(json.loads(json.dumps(payload, sort_keys=True)), payload)


class RefusalRecordReaderInvariantTest(unittest.TestCase):
    """The records must be INVISIBLE to ``pokezero.fallback_addresses``.

    That reader accepts a mapping as a cumulative stats scope iff it CONTAINS
    ``fallback_samples``, and harvests addresses from every mapping so NAMED. A
    refusal record carries counter mappings of its own, so the risk is real: if one
    were spelled ``world_failure_reasons`` or ``fallback_reasons`` inside a block
    that also picked up a ``fallback_samples`` key, every occurrence total in the
    corpus would move.
    """

    def _shard(self, *, records: bool) -> dict:
        record = RefusalRecord(
            battle_id="battle-7",
            round=1,
            seat="p1",
            reason="crate_search_failed",
            world_failures={TRAPPED_KEY: 16},
            worlds_attempted=16,
            unmapped_choices={"substitute": 3},
            request_legal_choices=("thunderbolt",),
        )
        game = ControlledFoulPlayGameResult(
            battle_id="battle-7",
            seed=7,
            winner=None,
            pokezero_won=False,
            decision_rounds=4,
            pokezero_decisions=4,
            root_puct_searches=0,
            root_puct_fallbacks=0,
            refusal_records=(record, record) if records else (),
        )
        stats = EngineMctsStats()
        stats.decisions = 4
        stats.fallback_decisions = 2
        stats.fallback_reasons["crate_search_failed"] = 2
        stats.world_failure_reasons[TRAPPED_KEY] = 6
        stats.fallback_samples["crate_search_failed"] = [
            {"battle_id": "battle-7", "round": 1, "seat": "p1", "reason": "crate_search_failed"},
            {"battle_id": "battle-7", "round": 3, "seat": "p1", "reason": "crate_search_failed"},
        ]
        payload = ControlledFoulPlayBenchmarkResult(
            config=_config(record_refusals=records),
            policy_id="stub",
            games=(game,),
            policy_stats=stats.to_dict(),
            refusal_recorder=RefusalRecorderHealth(
                enabled=records, attached=records, health_reported=records
            ),
        ).to_dict()
        # The real shard shape: production mirrors `to_dict()` under
        # `engine_mcts.policy_stats` AND under `per_seat[seat].policy_stats`, and
        # `_scan_document` de-duplicates whole blocks by content. Two copies means a
        # leak into the stats dict would surface as a doubled count, not as a pass.
        payload["engine_mcts"] = {"policy_stats": stats.to_dict()}
        payload["per_seat"] = {"p1": {"policy_stats": stats.to_dict()}}
        return payload

    def _scan(self, document: dict, directory: Path, name: str):
        path = directory / name
        path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
        return scan_corpus([path])

    def test_records_change_no_address_and_no_occurrence_count(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            plain = self._scan(self._shard(records=False), directory, "plain.json")
            recorded = self._scan(self._shard(records=True), directory, "recorded.json")

        # The baseline is real, not vacuous.
        self.assertEqual(len(plain.addresses), 2)
        self.assertEqual(plain.decision_counts["fallback:crate_search_failed"], 2)
        self.assertTrue(plain.world_counts)

        self.assertEqual(
            [(a.battle_id, a.round, a.seat, a.key) for a in recorded.addresses],
            [(a.battle_id, a.round, a.seat, a.key) for a in plain.addresses],
        )
        self.assertEqual(recorded.decision_counts, plain.decision_counts)
        self.assertEqual(recorded.world_counts, plain.world_counts)
        self.assertEqual(recorded.addresses_dropped, plain.addresses_dropped)
        self.assertEqual(recorded.shards_read, plain.shards_read)
        self.assertEqual(recorded.shards_unreadable, 0)

    def test_no_key_the_reader_dispatches_on_appears_in_the_new_blocks(self) -> None:
        """A literal guard on the two names that steer the reader."""
        document = self._shard(records=True)
        row = document["game_results"][0]

        self.assertIn("refusals", row)
        self.assertNotIn("fallback_samples", row)
        self.assertNotIn("fallback_samples", document["refusal_recorder"])
        for record in row["refusals"]:
            self.assertNotIn("fallback_samples", record)
            # The per-decision delta is `world_failures`; spelling it
            # `world_failure_reasons` would be harmless HERE and catastrophic in any
            # block that also carried `fallback_samples`.
            self.assertNotIn("world_failure_reasons", record)
            self.assertNotIn("fallback_reasons", record)
            self.assertIn("world_failures", record)

    def test_the_stats_serializer_gained_no_refusal_field(self) -> None:
        """`EngineMctsStats.to_dict()` is a frozen consumer contract."""
        keys = EngineMctsStats().to_dict()
        self.assertNotIn("refusals", keys)
        self.assertNotIn("refusal_recorder", keys)


class SearchBehaviourUnchangedTest(unittest.TestCase):
    """Recorder attached and detached must produce the same run.

    ``#1180``'s own end-to-end check compares SEVEN keys of
    ``EngineMctsStats.to_dict()``. This compares ALL of them, and additionally the
    arguments the policy saw and the RNG draws it took -- because "the counters
    match" and "the search did the same thing" are different claims and only the
    second is the one being made.
    """

    def _run(self, *, record: bool) -> dict:
        policy = _EnginePolicyStub()
        seen: list[tuple] = []
        capture = _RefusalCapture(policy, enabled=record)
        try:
            for round_index in range(6):
                context = _Context("battle-7", round_index, "p1")
                # A real seeded draw, positioned exactly where the searcher's is.
                rng = random.Random(f"{context.seed}:{context.player_id}:{round_index}")
                draw = rng.getrandbits(63)
                seen.append((round_index, context.player_id, draw))
                if round_index % 2:
                    policy.refuse(
                        context,
                        world_failures={TRAPPED_KEY: 8 + round_index},
                        aggregated={"substitute": 0.9},
                        unmapped={"substitute": 1},
                    )
                else:
                    policy.succeed(context)
            return {
                "stats": policy.stats.to_dict(),
                "policy_calls": seen,
                "records": [r.to_dict() for r in capture.records_for("battle-7")],
            }
        finally:
            capture.detach()

    def test_stats_are_identical_with_and_without_the_recorder(self) -> None:
        detached = self._run(record=False)
        attached = self._run(record=True)

        # ALL keys, named in the failure message so a divergence says which one.
        self.assertEqual(set(detached["stats"]), set(attached["stats"]))
        self.assertGreater(len(detached["stats"]), 40)
        self.assertEqual(detached["stats"], attached["stats"])
        self.assertEqual(detached["policy_calls"], attached["policy_calls"])
        # ...and the two runs really did differ in the one intended way. Without
        # this the equalities above are asserted about two identical no-ops.
        self.assertEqual(detached["records"], [])
        self.assertEqual(len(attached["records"]), 3)

    def test_the_records_are_per_decision_deltas_not_running_totals(self) -> None:
        """The null-world test for the capture, run through the bridge's own owner.

        A recorder that reported the cumulative counter is the plausible wrong
        implementation -- it is what the shard already gives you, and it is exactly
        the resolution that made the campaign theorise. Decision 3 fires the trapped
        class 11 times after decision 1 fired it 9; a cumulative reading gives 20.
        """
        attached = self._run(record=True)
        counts = [record["world_failures"][TRAPPED_KEY] for record in attached["records"]]

        self.assertEqual(counts, [9, 11, 13])
        self.assertEqual(
            [record["unmapped_choices"] for record in attached["records"]],
            [{"substitute": 1}] * 3,
        )


class RefusalRecorderWiringTest(unittest.TestCase):
    """The layer between the pure objects and production."""

    def test_config_from_args_carries_the_flag_into_production(self) -> None:
        """`_config_from_args` is the SOLE production path to a config."""
        parser = build_arg_parser()
        base = ["--checkpoint", "c.pt", "--showdown-root", "/s"]

        default = _config_from_args(parser.parse_args(base))
        disabled = _config_from_args(parser.parse_args(base + ["--no-refusal-records"]))

        self.assertTrue(default.record_refusals)
        self.assertFalse(disabled.record_refusals)

    def _play(self, *, config, policy, games: int = 2, refuse_rounds=(1,), boundary=None):
        """Drive the REAL `_run_controlled_foulplay_games` over stubbed collaborators.

        `_run_single_game` is real too, so the whole attach -> record -> partition ->
        serialize path runs; only the websocket server, the bridge process and the
        decision boundary are stubbed.
        """
        config = replace(config, games=games)
        seeds = [config.seed_start + offset for offset in range(games)]
        pending: dict[str, list] = {}
        for seed in seeds:
            battle_id = f"{DEFAULT_BATTLE_ID_PREFIX}-{seed}"
            pending[battle_id] = [
                {"battleId": battle_id, "type": "ready", "requested": ["p1", "p2"]}
                for _ in range(4)
            ] + [{"battleId": battle_id, "type": "terminal"}]

        class Server:
            def __init__(self, **_kwargs) -> None:
                pass

            async def start(self) -> None:
                return None

            async def close(self) -> None:
                return None

            async def send_room_lines(self, *_a, **_k) -> None:
                return None

            uri = "ws://stub"

        class Bridge:
            def __init__(self, **_kwargs) -> None:
                self.current: str | None = None

            async def start(self) -> None:
                return None

            async def close(self) -> None:
                return None

            async def send(self, payload: dict) -> None:
                if payload.get("type") == "start":
                    self.current = payload["battleId"]

            async def next_event(self) -> dict:
                return pending[self.current].pop(0)

        async def default_boundary(*, state, decision_round, **_kwargs):
            context = _Context(state.battle_id, decision_round, config.pokezero_player)
            if decision_round in refuse_rounds:
                policy.refuse(context, aggregated={"substitute": 0.9})
            else:
                policy.succeed(context)
            return None

        boundary_fn = boundary or default_boundary

        async def noop(*_a, **_k) -> None:
            return None

        progress: list[ControlledFoulPlayBenchmarkResult] = []
        with (
            patch("pokezero.foulplay_bridge._FoulPlayWebsocketServer", Server),
            patch("pokezero.foulplay_bridge._BattleBridge", Bridge),
            patch("pokezero.foulplay_bridge._spawn_foulplay", side_effect=lambda *a, **k: _proc()),
            patch("pokezero.foulplay_bridge._drain_process_stream", side_effect=noop),
            patch("pokezero.foulplay_bridge._stop_foulplay_process", side_effect=noop),
            patch("pokezero.foulplay_bridge._wait_for_foulplay_challenge_or_exit", side_effect=noop),
            patch("pokezero.foulplay_bridge._handle_decision_boundary", side_effect=boundary_fn),
            patch("pokezero.foulplay_bridge._notify_foulplay_terminal", side_effect=noop),
        ):
            result = asyncio.run(
                _run_controlled_foulplay_games(
                    config,
                    policy=policy,
                    policy_id="stub",
                    vocab=object(),
                    dex=object(),
                    observation_spec=SimpleNamespace(schema_version="v2.2"),
                    feature_masks=object(),
                    checkpoint_sha256=None,
                    progress_callback=progress.append,
                )
            )
        return result, progress

    def test_the_shipped_default_attaches_the_recorder(self) -> None:
        """The default config, untouched, must record.

        Kills BOTH the deleted attach call and a default flipped to False -- neither
        of which any assertion elsewhere in this file can see, because every other
        test that needs a recorder builds `_RefusalCapture` directly.
        """
        config = _config()  # NO record_refusals argument, on purpose.
        self.assertTrue(config.record_refusals)
        policy = _EnginePolicyStub()

        result, _ = self._play(config=config, policy=policy)
        payload = result.to_dict()

        self.assertTrue(payload["refusal_recorder"]["enabled"])
        self.assertTrue(payload["refusal_recorder"]["attached"])
        self.assertTrue(payload["refusal_recorder"]["health_reported"])
        self.assertTrue(payload["refusal_recorder"]["trustworthy"])
        self.assertEqual(payload["refusal_recorder"]["recorded_refusals"], 2)
        self.assertEqual([len(row["refusals"]) for row in payload["game_results"]], [1, 1])

    def test_run_single_game_files_records_under_the_real_battle_id(self) -> None:
        """One recorder for the run; every record on the battle it happened in.

        Kills the records being dropped from the row, and kills a filter on the
        wrong battle -- which with a run-scoped recorder would put game 1's refusal
        on game 2's row as well.
        """
        policy = _EnginePolicyStub()
        result, _ = self._play(config=_config(seed_start=41), policy=policy, games=3)

        rows = {row.battle_id: row for row in result.games}
        self.assertEqual(
            sorted(rows),
            [f"{DEFAULT_BATTLE_ID_PREFIX}-{seed}" for seed in (41, 42, 43)],
        )
        for battle_id, row in rows.items():
            self.assertEqual(len(row.refusal_records), 1)
            self.assertEqual(row.refusal_records[0].battle_id, battle_id)
            self.assertEqual(row.refusal_records[0].round, 1)

    def test_the_flag_really_switches_it_off(self) -> None:
        policy = _EnginePolicyStub()
        result, _ = self._play(config=_config(record_refusals=False), policy=policy)
        payload = result.to_dict()

        self.assertEqual([row.refusal_records for row in result.games], [(), ()])
        self.assertFalse(payload["refusal_recorder"]["enabled"])
        self.assertFalse(payload["refusal_recorder"]["attached"])
        # OFF is not CLEAN.
        self.assertFalse(payload["refusal_recorder"]["trustworthy"])
        # ...and the battles really did refuse, so the empty lists above are the
        # flag and not a fixture that never refused.
        self.assertEqual(policy.stats.fallback_decisions, 2)

    def test_a_partial_progress_summary_reports_health_too(self) -> None:
        """`--summary-out` rewrites the whole document every game.

        A run killed mid-way leaves a progress write as its only artifact, so health
        reported only on the final result would make every abandoned run unknown.
        """
        _, progress = self._play(config=_config(), policy=_EnginePolicyStub(), games=3)

        self.assertEqual(len(progress), 3)
        headers = [partial.to_dict()["refusal_recorder"] for partial in progress]
        self.assertEqual([header["health_reported"] for header in headers], [True] * 3)
        self.assertEqual([header["recorded_refusals"] for header in headers], [1, 2, 3])

    def test_an_unrecordable_policy_is_reported_and_never_fatal(self) -> None:
        """The raw and root-PUCT arms have no engine searcher to attach to.

        That is the correct outcome -- those arms file no `fallback_samples` either
        -- but it must not render as a clean engine-mcts run.
        """

        class RawPolicy:
            policy_id = "raw"

        async def boundary(**_kwargs) -> None:
            return None

        result, _ = self._play(
            config=_config(policy_mode="raw"), policy=RawPolicy(), boundary=boundary
        )
        header = result.to_dict()["refusal_recorder"]

        self.assertTrue(header["enabled"])
        self.assertFalse(header["attached"])
        self.assertIsNotNone(header["attach_error"])
        self.assertFalse(header["trustworthy"])
        self.assertEqual(result.completed_games, 2)

    def test_instrument_errors_reach_the_summary(self) -> None:
        """A swallowed capture failure must not render as "no refusals".

        The recorder deliberately never raises into the search. So the ONLY way a
        reader learns the capture broke is this list, and an implementation that
        reports the records but drops the errors produces an empty, confident,
        wrong answer.
        """
        policy = _EnginePolicyStub()

        async def boundary(*, state, decision_round, **_kwargs):
            # Refusals reached WITHOUT the decision ever opening -- the real shape a
            # searcher that refuses outside `select_action_with_context` produces.
            # The recorder has no pre-decision baseline, files the record anyway
            # marked degraded, and records WHY on `errors`. An implementation that
            # forwarded the records but dropped the errors would publish deltas that
            # silently span more than one decision, as a confident measurement.
            policy._fallback(
                _Context(state.battle_id, decision_round, "p1"), None, "crate_search_failed"
            )
            return None

        result, _ = self._play(config=_config(), policy=policy, boundary=boundary)

        payload = result.to_dict()
        header = payload["refusal_recorder"]
        self.assertTrue(header["attached"])
        self.assertEqual(len(header["instrument_errors"]), 8)
        self.assertEqual(header["degraded_records"], 8)
        self.assertFalse(header["trustworthy"])
        # ...and the records are still DELIVERED, marked, rather than suppressed.
        self.assertEqual([len(row["refusals"]) for row in payload["game_results"]], [4, 4])
        self.assertTrue(
            all(record["degraded"] for row in payload["game_results"] for record in row["refusals"])
        )

    def test_a_degraded_record_is_counted_and_blocks_trust(self) -> None:
        """A record whose baseline was lost is readable, but not a measurement."""
        policy = _EnginePolicyStub()
        capture = _RefusalCapture(policy, enabled=True)
        try:
            # `_fallback` with no preceding `select_action_with_context`: no
            # pre-decision snapshot, so the deltas may span more than one decision.
            policy._fallback(_Context("battle-7", 2, "p1"), None, "crate_search_failed")
            health = capture.health()
        finally:
            capture.detach()

        self.assertEqual(health.degraded_records, 1)
        self.assertTrue(health.instrument_errors)
        self.assertFalse(health.trustworthy)
        self.assertTrue(capture.records_for("battle-7")[0].degraded)

    def test_two_runs_over_one_policy_do_not_leak_records(self) -> None:
        """`_run_controlled_foulplay_games` detaches, so the comparison runner's
        second arm starts clean rather than inheriting the first arm's refusals."""
        policy = _EnginePolicyStub()
        first, _ = self._play(config=_config(seed_start=1), policy=policy)
        second, _ = self._play(config=_config(seed_start=100), policy=policy)

        self.assertEqual(first.to_dict()["refusal_recorder"]["recorded_refusals"], 2)
        self.assertEqual(second.to_dict()["refusal_recorder"]["recorded_refusals"], 2)
        self.assertEqual(
            [row.refusal_records[0].battle_id for row in second.games],
            [f"{DEFAULT_BATTLE_ID_PREFIX}-{seed}" for seed in (100, 101)],
        )


class _proc:
    """The subprocess handle `_run_controlled_foulplay_games` carries around."""

    returncode = None
    stdout = None
    stderr = None


class RefusalRecorderCliTest(unittest.TestCase):
    def test_the_flag_exists_and_defaults_to_recording(self) -> None:
        base = ["--checkpoint", "c.pt", "--showdown-root", "/s"]
        self.assertFalse(build_arg_parser().parse_args(base).no_refusal_records)
        self.assertTrue(
            build_arg_parser().parse_args(base + ["--no-refusal-records"]).no_refusal_records
        )

    def test_the_help_text_states_the_measured_cost(self) -> None:
        """A default-on instrument must justify itself where the operator looks."""
        text = build_arg_parser().format_help()
        self.assertIn("--no-refusal-records", text)


class RecorderAttachesToTheRealPolicyShapeTest(unittest.TestCase):
    """The recorder's own validation, run against the bridge's own usage.

    `attach_refusal_recorder` validates six `stats` attributes and three methods at
    attach time. `_EnginePolicyStub` owns a REAL `EngineMctsStats`, so this pins
    that the six names still exist on the real stats object -- a rename there would
    otherwise turn every production run's header into `attached: false` while every
    other test in this file kept passing.
    """

    def test_a_real_stats_object_satisfies_the_recorder(self) -> None:
        recorder = attach_refusal_recorder(_EnginePolicyStub())
        try:
            self.assertEqual(recorder.errors, [])
        finally:
            recorder.detach()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
