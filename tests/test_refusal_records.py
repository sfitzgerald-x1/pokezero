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
     ``test_a_partial_progress_summary_reports_health_too``;
* the detach removed from the run's teardown
  -> ``test_the_run_uninstalls_the_recorder_from_the_policy``.

Tests that are easy to write so that they cannot fail, and what is done about it:

* ``test_stats_are_identical_with_and_without_the_recorder``. Two failure modes,
  both real, both found by review of the first revision:

  1. It asserts two runs are equal, which is trivially true if neither run records
     -- so it also asserts the two runs DID differ in the intended way.
  2. The FIRST revision claimed "all 69 keys of ``to_dict()``". That number was
     wrong and the claim was hollow. Measured: 46 dataclass fields, **48 keys** on a
     fresh stats object, and only **13** hold a non-default value under this
     fixture. The other 35 comparisons are ``0``/``{}``/``None`` on both sides,
     including every wall-clock key, which could not be bit-identical on a real
     policy anyway. The full-dict equality is still asserted, but the claim now
     rests on ``_STATS_KEYS_EXERCISED``, pinned BY NAME so a fixture that stops
     moving a counter fails instead of quietly shrinking the comparison.

* THE COUNTERS ARE NOT THE WHOLE CLAIM. The first revision's stub called
  ``select_action_with_context(context, rng=None)``, so the policy never touched an
  RNG, and three mutants that make the recorder ACTIVELY PERTURB the live decision
  path -- consume a draw before the fan-out, substitute ``random.Random(0)`` into
  the search, substitute ``dict(aggregated)`` into ``_map_choices`` -- passed all
  237 tests. Every counter still matched, because a different world sample produces
  the same COUNTS in a stub. ``test_the_recorder_consumes_no_draw_from_the_decision_rng``
  and ``test_the_policy_receives_the_caller_s_own_rng_and_aggregate`` are that layer.

* ``test_records_change_no_address_and_no_occurrence_count`` scans a document that
  really contains ``fallback_samples``, mirrored under two paths as production
  shards are, and asserts the baseline is NON-EMPTY before comparing. Against a
  records-only document both sides are zero and it passes for free.
* ``test_health_is_not_trustworthy_when_it_was_never_reported`` is the null-world
  test for the health block: the wrong implementation is ``trustworthy = not
  errors``, which is True for a run that never attached, never ran, and never had
  an error channel. Every conjunct is therefore falsified on its own, in a loop, so
  adding a conjunct without a case for it is visible.
* ``test_two_runs_over_one_policy_do_not_leak_records`` CANNOT see the detach
  wiring -- each run builds a fresh capture, so arm two is clean either way, and
  deleting the detach left the suite green. The leak is the ORPHANED recorder still
  subscribed to a policy that outlives the call, which is only visible on
  ``policy.__dict__``. That is what the test above it looks at.
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
    _REFUSAL_RECORDS_PER_BATTLE,
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
        # What the BASE methods actually received. The recorder's contract is that
        # it forwards these untouched; nothing else in this file can see that.
        self.seen_rng: list = []
        self.seen_aggregated: list = []
        self.seen_fallback_rng: list = []
        self.draws: list[int] = []

    def select_action_with_context(self, context, *, rng):
        # RECORDED, and the draw is TAKEN. Without this the policy never touches the
        # decision RNG and three mutants that make the recorder actively perturb the
        # live decision path -- consuming a draw, substituting a different Random,
        # substituting a copied aggregate -- all pass the whole suite.
        self.seen_rng.append(rng)
        if rng is not None:
            self.draws.append(rng.getrandbits(63))
        return None

    def _map_choices(self, context, aggregated):
        self.seen_aggregated.append(aggregated)
        return None

    def _fallback(self, context, rng, reason):
        self.seen_fallback_rng.append(rng)
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
        rng=None,
    ) -> None:
        self.select_action_with_context(context, rng=rng)
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

    def succeed(self, context, *, rng=None) -> None:
        self.select_action_with_context(context, rng=rng)
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
        base = dict(enabled=True, attached=True, health_reported=True)
        self.assertTrue(RefusalRecorderHealth(**base).trustworthy)
        # ...and each one, alone, takes it away.
        for name, broken in (
            ("enabled", dict(base, enabled=False)),
            ("attached", dict(base, attached=False)),
            ("health_reported", dict(base, health_reported=False)),
            ("instrument_errors", dict(base, instrument_errors=("boom",),
                                       instrument_errors_total=1)),
            ("degraded_records", dict(base, degraded_records=1)),
            # The reconciliation: a record that reached no game row is neither
            # emitted nor dropped, and before this conjunct existed it vanished
            # while `recorded_refusals` still counted it and this still said True.
            ("reconciled", dict(base, recorded_refusals=3, emitted_refusals=1,
                                records_dropped=1)),
        ):
            with self.subTest(conjunct=name):
                self.assertFalse(RefusalRecorderHealth(**broken).trustworthy)

    def test_a_truncated_run_is_reconciled_but_says_it_truncated(self) -> None:
        """Dropping to a ceiling is honest; dropping silently is not."""
        health = RefusalRecorderHealth(
            enabled=True, attached=True, health_reported=True,
            recorded_refusals=40, emitted_refusals=8, records_dropped=32,
        )
        self.assertTrue(health.reconciled)
        self.assertTrue(health.trustworthy)
        self.assertEqual(health.to_dict()["records_dropped"], 32)

    def test_instrument_errors_are_sampled_but_the_count_is_not(self) -> None:
        """The list is one string per failed capture and it is uncapped upstream."""
        policy = _EnginePolicyStub()
        capture = _RefusalCapture(policy, enabled=True)
        try:
            for round_index in range(60):
                # No `select_action_with_context`, so every capture files an error.
                policy._fallback(_Context("battle-7", round_index, "p1"), None, "r")
            health = capture.health()
        finally:
            capture.detach()

        self.assertEqual(health.instrument_errors_total, 60)
        self.assertEqual(len(health.instrument_errors), 20)
        self.assertFalse(health.trustworthy)

    def test_an_unattached_capture_cannot_claim_to_have_reported_health(self) -> None:
        """`health_reported` asserts an error CHANNEL, and there is none here."""
        health = _RefusalCapture(SimpleNamespace(), enabled=True).health()
        self.assertFalse(health.health_reported)
        self.assertEqual(health.instrument_errors_total, 0)
        self.assertFalse(health.trustworthy)

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


#: The `EngineMctsStats.to_dict()` keys the fixture below actually MOVES off their
#: default. Named rather than counted, because the count is what made the first
#: version of this file overclaim: `to_dict()` emits 48 keys on a fresh stats object
#: and 38 of them are `0`/`{}`/`None` in BOTH arms of the comparison, so "all 48 keys
#: are equal" is 38 assertions that two zeros are equal. Thirteen of the 38 are
#: wall-clock fields that could not be bit-identical on a real policy anyway.
#:
#: The full-dict equality below is still asserted -- a recorder that INVENTED a value
#: in one of the 38 must fail -- but this set is what carries the claim, and pinning
#: it by name means a fixture that stops exercising a counter fails loudly instead of
#: quietly shrinking the real comparison.
_STATS_KEYS_EXERCISED = {
    "belief_sample_rejection_rate",
    "decisions",
    "fallback_decisions",
    "fallback_rate",
    "fallback_reasons",
    "fallback_samples",
    "unmapped_choices",
    "world_failure_reasons",
    "worlds_attempted",
    "worlds_constructed",
    "worlds_searched",
    # Derived, and they move because their inputs did: `wall_per_decision` is
    # wall/decisions (None -> 0.0 once `decisions` is non-zero) and
    # `world_search_abort_rate` is a ratio over `worlds_attempted`.
    "wall_per_decision",
    "world_search_abort_rate",
}


class SearchBehaviourUnchangedTest(unittest.TestCase):
    """Recorder attached and detached must produce the same run.

    Three separate claims, and the first version of this class only established one
    of them:

    1. the COUNTERS agree -- full `EngineMctsStats.to_dict()`, plus the named subset
       above that actually moves;
    2. the policy receives the SAME OBJECTS -- the same `Random`, the same
       `aggregated` mapping. A recorder that substitutes a copy has changed the
       search's inputs even though every counter still matches;
    3. the policy receives them in the SAME RNG STATE -- the recorder consumes no
       draw. This is the one that matters most for a default-on change, because a
       single `getrandbits` inside the wrapper silently reseeds every world in the
       decision.

    Claims 2 and 3 were true in both worlds before, for a dull reason: the fixture
    called `select_action_with_context(context, rng=None)`, so the policy never
    touched an RNG and three mutants that make the recorder actively perturb the live
    decision path all passed the suite. The stub now takes the draw.
    """

    def _run(self, *, record: bool) -> dict:
        policy = _EnginePolicyStub()
        capture = _RefusalCapture(policy, enabled=record)
        try:
            for round_index in range(6):
                context = _Context("battle-7", round_index, "p1")
                # The bridge's own per-decision seeding rule (`foulplay_bridge`
                # `_select_policy_decision`), so the RNG is positioned exactly where
                # the searcher's is and the draws below are the draws it would take.
                rng = random.Random(f"{context.seed}:{context.player_id}:{round_index}")
                if round_index % 2:
                    policy.refuse(
                        context,
                        world_failures={TRAPPED_KEY: 8 + round_index},
                        aggregated={"substitute": 0.9},
                        unmapped={"substitute": 1},
                        rng=rng,
                    )
                else:
                    policy.succeed(context, rng=rng)
            return {
                "stats": policy.stats.to_dict(),
                "draws": list(policy.draws),
                "records": [r.to_dict() for r in capture.records_for("battle-7")],
            }
        finally:
            capture.detach()

    def test_stats_are_identical_with_and_without_the_recorder(self) -> None:
        detached = self._run(record=False)
        attached = self._run(record=True)

        self.assertEqual(set(detached["stats"]), set(attached["stats"]))
        self.assertEqual(detached["stats"], attached["stats"])
        # The comparison is not 48 assertions that zero equals zero: these keys hold
        # a value that this fixture moved, in both arms.
        empty = EngineMctsStats().to_dict()
        moved = {
            key for key, value in detached["stats"].items() if value != empty.get(key)
        }
        self.assertEqual(moved, _STATS_KEYS_EXERCISED)
        # ...and the two runs really did differ in the one intended way. Without
        # this the equalities above are asserted about two identical no-ops.
        self.assertEqual(detached["records"], [])
        self.assertEqual(len(attached["records"]), 3)

    def test_the_recorder_consumes_no_draw_from_the_decision_rng(self) -> None:
        """Every world seed in the decision comes off this RNG.

        A single `getrandbits` inside the wrapper shifts the whole stream, so the
        search samples different worlds while every counter in the test above still
        matches. Six decisions, bit-identical draw sequence.
        """
        detached = self._run(record=False)
        attached = self._run(record=True)

        self.assertEqual(len(detached["draws"]), 6)
        self.assertEqual(detached["draws"], attached["draws"])

    def test_the_policy_receives_the_caller_s_own_rng_and_aggregate(self) -> None:
        """Identity, not equality.

        A recorder that hands the search a fresh `Random(0)`, or a `dict(aggregated)`
        copy, produces a run whose counters are unchanged and whose inputs are not.
        Equality cannot see either; `assertIs` can.
        """
        policy = _EnginePolicyStub()
        capture = _RefusalCapture(policy, enabled=True)
        try:
            rng = random.Random("600016:p1:7")
            aggregated = {"substitute": 0.9, "thunderbolt": 0.1}
            context = _Context("battle-7", 7, "p1")
            policy.refuse(context, aggregated=aggregated, rng=rng)

            self.assertIs(policy.seen_rng[0], rng)
            self.assertIs(policy.seen_aggregated[0], aggregated)
            self.assertIsNot(policy.seen_aggregated[0], None)
            # ...while the RECORD holds a snapshot rather than the live object, so
            # the identity above cannot be satisfied by simply not copying anywhere.
            # `aggregated` is a Counter the search keeps mutating after this call.
            record = capture.records_for("battle-7")[0]
            aggregated["substitute"] = 99.0
            self.assertEqual(record.map_choices_calls[0]["substitute"], 0.9)
        finally:
            capture.detach()

    def test_the_recorder_leaves_the_policy_exactly_as_it_found_it(self) -> None:
        """Detach restores the instance dict, and does not freeze bound copies on it.

        `_Hook` saves `_ABSENT` for a name that was a CLASS method precisely so
        `setattr` does not park a self-referential bound copy on the instance. That
        is unobservable through behaviour and very observable through `__dict__`.
        """
        policy = _EnginePolicyStub()
        before = set(policy.__dict__)

        capture = _RefusalCapture(policy, enabled=True)
        self.assertTrue({"select_action_with_context", "_map_choices", "_fallback"}
                        <= set(policy.__dict__))
        capture.detach()

        self.assertEqual(set(policy.__dict__), before)
        self.assertNotIn("_pz_refusal_hook", policy.__dict__)

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

    def _play(
        self,
        *,
        config,
        policy,
        games: int = 2,
        rounds: int = 4,
        refuse_rounds=(1,),
        boundary=None,
        failing_close: str | None = None,
    ):
        """Drive the REAL `_run_controlled_foulplay_games` over stubbed collaborators.

        `_run_single_game` is real too, so the whole attach -> record -> partition ->
        cap -> serialize path runs; only the websocket server, the bridge process and
        the decision boundary are stubbed.
        """
        config = replace(config, games=games)
        seeds = [config.seed_start + offset for offset in range(games)]
        pending: dict[str, list] = {}
        for seed in seeds:
            battle_id = f"{DEFAULT_BATTLE_ID_PREFIX}-{seed}"
            pending[battle_id] = [
                {"battleId": battle_id, "type": "ready", "requested": ["p1", "p2"]}
                for _ in range(rounds)
            ] + [{"battleId": battle_id, "type": "terminal"}]

        self._server_closed = False
        outer = self

        class Server:
            def __init__(self, **_kwargs) -> None:
                pass

            async def start(self) -> None:
                return None

            async def close(self) -> None:
                outer._server_closed = True
                if failing_close == "server":
                    raise RuntimeError("server close failed")

            async def send_room_lines(self, *_a, **_k) -> None:
                return None

            uri = "ws://stub"

        class Bridge:
            def __init__(self, **_kwargs) -> None:
                self.current: str | None = None

            async def start(self) -> None:
                return None

            async def close(self) -> None:
                if failing_close == "bridge":
                    raise RuntimeError("bridge close failed")

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

    def test_the_run_uninstalls_the_recorder_from_the_policy(self) -> None:
        """The DETACH WIRING, which the test above cannot see.

        Each run builds a fresh `_RefusalCapture`, so arm two is clean whether or not
        arm one detached -- deleting `refusal_capture.detach()` left the whole suite
        green. What leaks is the ORPHANED recorder: `_Hook.remove` never empties
        `recorders`, the three wrappers stay installed on a policy that outlives this
        call, and every later `_fallback` fans out to N accumulating subscribers for
        the life of the process. Nothing observes that through behaviour, so this
        looks at the policy itself.
        """
        policy = _EnginePolicyStub()
        clean = set(policy.__dict__)

        self._play(config=_config(), policy=policy)

        self.assertNotIn("_pz_refusal_hook", policy.__dict__)
        self.assertEqual(set(policy.__dict__), clean)
        # And the orphan is really gone: a later refusal grows nobody's record list.
        _, progress = self._play(config=_config(seed_start=50), policy=policy)
        policy.refuse(_Context("battle-after", 0, "p1"))
        self.assertEqual(progress[-1].to_dict()["refusal_recorder"]["recorded_refusals"], 2)

    def test_a_teardown_failure_cannot_skip_the_detach(self) -> None:
        """`bridge.close()` raising used to skip `server.close()` AND the detach."""
        policy = _EnginePolicyStub()

        with self.assertRaises(RuntimeError):
            self._play(config=_config(), policy=policy, failing_close="bridge")

        self.assertNotIn("_pz_refusal_hook", policy.__dict__)
        self.assertTrue(self._server_closed)

    def test_records_are_capped_per_battle_and_the_loss_is_a_number(self) -> None:
        """A cause closes worlds for the rest of its battle; the tail is repeats.

        Measured on a real d4/s1024/w4 batch: one seed filed TEN consecutive
        identical `no_worlds_constructed` refusals. Uncapped, a high-refusal cell
        puts every one of them on the row and `_write_json` re-serializes the lot
        once per game.
        """
        policy = _EnginePolicyStub()
        result, _ = self._play(
            config=_config(), policy=policy, games=1, rounds=14, refuse_rounds=range(14)
        )
        header = result.to_dict()["refusal_recorder"]

        self.assertEqual(len(result.games[0].refusal_records), _REFUSAL_RECORDS_PER_BATTLE)
        self.assertEqual(header["recorded_refusals"], 14)
        self.assertEqual(header["emitted_refusals"], _REFUSAL_RECORDS_PER_BATTLE)
        self.assertEqual(header["records_dropped"], 14 - _REFUSAL_RECORDS_PER_BATTLE)
        # Truncation is not corruption: the identity still holds and the ceiling is
        # published so a reader can tell WHICH bound bit.
        self.assertTrue(header["reconciled"])
        self.assertTrue(header["trustworthy"])
        self.assertEqual(header["records_per_battle_ceiling"], _REFUSAL_RECORDS_PER_BATTLE)

    def test_the_run_ceiling_bounds_the_document_across_games(self) -> None:
        """The per-battle cap alone does not bound a 250-game run."""
        policy = _EnginePolicyStub()
        with patch("pokezero.foulplay_bridge._REFUSAL_RECORDS_PER_RUN", 5):
            result, _ = self._play(
                config=_config(), policy=policy, games=4, rounds=4, refuse_rounds=range(4)
            )
        header = result.to_dict()["refusal_recorder"]

        emitted = [len(row.refusal_records) for row in result.games]
        self.assertEqual(sum(emitted), 5)
        # Front-loaded, not spread: the ceiling is a hard stop, and the games past it
        # emit nothing rather than each losing a little invisibly.
        self.assertEqual(emitted, [4, 1, 0, 0])
        self.assertEqual(header["recorded_refusals"], 16)
        self.assertEqual(header["emitted_refusals"], 5)
        self.assertEqual(header["records_dropped"], 11)
        self.assertTrue(header["reconciled"])

    def test_a_record_belonging_to_no_game_row_breaks_the_identity(self) -> None:
        """The silent-loss case the reconciliation exists for.

        A refusal filed under a battle id that never produces a row -- a game that
        raised after refusing, or a battle_id filter that drifted -- is dropped from
        the document while the recorder still counted it. `records_dropped` catches
        it as a NUMBER, which is the only form in which a missing record is visible.
        """
        policy = _EnginePolicyStub()

        async def boundary(*, state, decision_round, **_kwargs):
            # Refuse under a battle the loop will never build a row for.
            policy.refuse(_Context("battle-nowhere", decision_round, "p1"))
            return None

        result, _ = self._play(config=_config(), policy=policy, boundary=boundary)
        header = result.to_dict()["refusal_recorder"]

        self.assertEqual(header["recorded_refusals"], 8)
        self.assertEqual(header["emitted_refusals"], 0)
        # Counted at end of run rather than silently absent.
        self.assertEqual(header["records_dropped"], 8)
        self.assertTrue(header["reconciled"])


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
