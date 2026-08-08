"""Tests for the refusal recorder and the replay driver.

Null-world discipline, applied per test. The recorder's plausible wrong
implementation is "report the cumulative counters" -- which is what the shard
already gives you and is exactly the resolution that made the campaign theorise.
So every recorder test runs at least TWO decisions with overlapping classes, and
asserts the second record does not carry the first's counts. A cumulative
implementation passes a one-decision test and fails all of these.

The driver's plausible wrong implementation is a two-valued verdict. Each
outcome therefore has a test that a two-valued classifier would have to get
wrong, and `SAME_BATTLE_DIFFERENT_ROUND` is asserted to be distinct from both
`REPRODUCED` and `ADDRESS_ABSENT`.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from pokezero.fallback_replay import (
    RefusalRecord,
    ReplayOutcome,
    ReplayResult,
    attach_refusal_recorder,
    format_refusal,
    replay_fallback,
    results_to_json,
)
from pokezero.fallback_replay_spec import (
    FIDELITY_EXACT,
    FIDELITY_OPPONENT_UNPINNED,
    HARNESS_FOULPLAY_BRIDGE,
    HARNESS_ROLLOUT_HC_GRID,
    ReplaySpec,
)


# --- a stand-in for EngineMctsPolicy ---------------------------------------
#
# Shaped from the real one: the same counter names, the same cumulative
# semantics, and the same three method signatures the recorder wraps. It is a
# stub of the POLICY, not of the recorder -- the recorder under test is the real
# one, and it is wired the same way it will be wired to a real policy.


class _Stats:
    def __init__(self) -> None:
        self.world_failure_reasons: Counter = Counter()
        self.unmapped_choices: Counter = Counter()
        self.choices_unmapped_causes: Counter = Counter()
        self.worlds_attempted = 0
        self.worlds_constructed = 0
        self.worlds_searched = 0
        self.fallback_decisions = 0


class _Context:
    def __init__(self, battle_id: str, round_index: int, seat: str, seed: int = 600000):
        self.battle_id = battle_id
        self.decision_round_index = round_index
        self.player_id = seat
        self.seed = seed
        self.observation = _Observation()


class _Observation:
    def __init__(self, candidates: Any = None, mask: Any = None) -> None:
        self.metadata = {"action_candidates": candidates if candidates is not None else []}
        self.legal_action_mask = mask if mask is not None else ()


class _FakePolicy:
    """Runs a scripted sequence of decisions against the real recorder."""

    def __init__(self) -> None:
        self.stats = _Stats()
        self.fallback_calls: list[str] = []

    # The three wrapped methods, with the real signatures.

    def select_action_with_context(self, context: Any, *, rng: Any) -> Any:
        # The real one runs the whole search; `_decide` below plays that role so
        # each test can script exactly which counters move on which decision.
        return None

    def _map_choices(self, context: Any, aggregated: Any) -> Any:
        return None

    def _fallback(self, context: Any, rng: Any, reason: str) -> Any:
        # The real `_fallback` mutates counters too; the stub does the same so a
        # recorder that captured AFTER delegating would visibly fold the
        # refusal's own bookkeeping into its evidence.
        self.stats.fallback_decisions += 1
        self.fallback_calls.append(reason)
        return f"fallback:{reason}"


def _decide(
    policy: _FakePolicy,
    context: _Context,
    *,
    world_failures: dict[str, int] | None = None,
    worlds: tuple[int, int, int] = (0, 0, 0),
    aggregated: dict[str, float] | None = None,
    unmapped: dict[str, int] | None = None,
    causes: dict[str, int] | None = None,
    reason: str | None = None,
) -> None:
    """Drive one decision through the wrapped methods, in the real order."""
    policy.select_action_with_context(context, rng=None)
    stats = policy.stats
    for key, count in (world_failures or {}).items():
        stats.world_failure_reasons[key] += count
    attempted, constructed, searched = worlds
    stats.worlds_attempted += attempted
    stats.worlds_constructed += constructed
    stats.worlds_searched += searched
    if aggregated is not None:
        policy._map_choices(context, aggregated)
    for key, count in (unmapped or {}).items():
        stats.unmapped_choices[key] += count
    for key, count in (causes or {}).items():
        stats.choices_unmapped_causes[key] += count
    if reason is not None:
        policy._fallback(context, None, reason)


_TRAPPED = (
    "self_request_state_unsupported: self active request flags ['trapped'] constrain "
    "legality beyond this construction (sampled world does not trap: foe ability 'swarm')"
)
_BATON = "materialization_blocker: baton-pass:substitute"


# --- the recorder -----------------------------------------------------------


class TestRecorderCapturesPerDecisionState:
    def test_world_failures_are_a_delta_not_a_running_total(self):
        # THE null-world test for this module. Decision 1 fires `_TRAPPED` x3;
        # decision 2 fires it x1 and `_BATON` x2. A recorder that reported the
        # cumulative counter would give decision 2 `_TRAPPED: 4`, which is the
        # number the shard already prints and the one that made the campaign
        # theorise.
        policy = _FakePolicy()
        recorder = attach_refusal_recorder(policy)
        _decide(
            policy,
            _Context("hcgrid-hc-d4-600016", 10, "p1"),
            world_failures={_TRAPPED: 3},
            reason="no_worlds_constructed",
        )
        _decide(
            policy,
            _Context("hcgrid-hc-d4-600016", 47, "p1"),
            world_failures={_TRAPPED: 1, _BATON: 2},
            reason="no_worlds_constructed",
        )
        first, second = recorder.records
        assert first.world_failures == {_TRAPPED: 3}
        assert second.world_failures == {_TRAPPED: 1, _BATON: 2}
        assert policy.stats.world_failure_reasons[_TRAPPED] == 4  # cumulative truth

    def test_world_budget_is_a_delta_too(self):
        policy = _FakePolicy()
        recorder = attach_refusal_recorder(policy)
        _decide(policy, _Context("b", 1, "p1"), worlds=(8, 4, 4), reason="x")
        _decide(policy, _Context("b", 2, "p1"), worlds=(16, 0, 0), reason="y")
        first, second = recorder.records
        assert (first.worlds_attempted, first.worlds_constructed) == (8, 4)
        assert (second.worlds_attempted, second.worlds_constructed) == (16, 0)

    def test_a_decision_that_does_not_refuse_records_nothing(self):
        # A recorder that emitted per decision rather than per refusal would
        # bury three real refusals in ten thousand clean ones.
        policy = _FakePolicy()
        recorder = attach_refusal_recorder(policy)
        _decide(policy, _Context("b", 1, "p1"), world_failures={_TRAPPED: 5})
        _decide(policy, _Context("b", 2, "p1"), reason="crate_search_failed")
        assert [r.round for r in recorder.records] == [2]
        # ...and the clean decision's failures are not attributed to the refusal.
        assert recorder.records[0].world_failures == {}

    def test_engine_choices_are_captured_at_refusal_time(self):
        # `aggregated` is a live Counter. A recorder holding a REFERENCE would
        # report whatever it contained when the record was read, not when the
        # decision refused.
        policy = _FakePolicy()
        recorder = attach_refusal_recorder(policy)
        aggregated = Counter({"substitute": 0.75, "protect": 0.25})
        policy.select_action_with_context(_Context("b", 9, "p1"), rng=None)
        policy._map_choices(_Context("b", 9, "p1"), aggregated)
        policy._fallback(_Context("b", 9, "p1"), None, "choices_unmapped")
        aggregated.clear()
        aggregated["earthquake"] = 1.0
        assert recorder.records[0].engine_choices == {
            "substitute": 0.75,
            "protect": 0.25,
        }

    def test_engine_choices_do_not_leak_between_decisions(self):
        policy = _FakePolicy()
        recorder = attach_refusal_recorder(policy)
        _decide(
            policy,
            _Context("b", 1, "p1"),
            aggregated={"substitute": 1.0},
            reason="choices_unmapped",
        )
        # Decision 2 never reaches the mapping step (it constructed no worlds).
        _decide(policy, _Context("b", 2, "p1"), reason="no_worlds_constructed")
        assert recorder.records[0].engine_choices == {"substitute": 1.0}
        assert recorder.records[1].engine_choices == {}

    def test_request_legal_set_mirrors_the_mapping_admission_rule(self):
        # The set must be built by `_map_choices`'s rule -- `legal` AND permitted
        # by the mask -- or every comparison against engine_choices is a
        # comparison with this helper. Three candidates, only one admissible.
        policy = _FakePolicy()
        recorder = attach_refusal_recorder(policy)
        context = _Context("b", 5, "p1")
        context.observation = _Observation(
            candidates=[
                {"kind": "move", "move_id": "surf", "legal": True, "action_index": 0},
                {"kind": "move", "move_id": "toxic", "legal": False, "action_index": 1},
                # legal, but the mask forbids it this turn
                {"kind": "move", "move_id": "rest", "legal": True, "action_index": 2},
                {
                    "kind": "switch",
                    "pokemon": {"species": "Zapdos"},
                    "legal": True,
                    "action_index": 3,
                },
            ],
            mask=(True, True, False, True),
        )
        policy.select_action_with_context(context, rng=None)
        policy._fallback(context, None, "choices_unmapped")
        assert recorder.records[0].request_legal_choices == ("surf", "switch Zapdos")

    def test_decision_rng_seed_is_reconstructed_from_the_context(self):
        policy = _FakePolicy()
        recorder = attach_refusal_recorder(policy)
        _decide(policy, _Context("b", 103, "p1", seed=8220001), reason="x")
        assert recorder.records[0].decision_rng_seed == "8220001:p1:103"

    def test_detach_restores_the_policy_exactly(self):
        # A recorder that could not be removed would make every future run
        # carry an instrument, which is how a measurement becomes a behaviour.
        policy = _FakePolicy()
        originals = (
            policy.select_action_with_context,
            policy._map_choices,
            policy._fallback,
        )
        recorder = attach_refusal_recorder(policy)
        assert policy._fallback is not originals[2]
        recorder.detach()
        assert policy.select_action_with_context == originals[0]
        assert policy._map_choices == originals[1]
        assert policy._fallback == originals[2]
        _decide(policy, _Context("b", 1, "p1"), reason="x")
        assert len(recorder.records) == 0

    def test_context_manager_detaches(self):
        policy = _FakePolicy()
        original = policy._fallback
        with attach_refusal_recorder(policy):
            pass
        assert policy._fallback == original

    def test_recording_does_not_change_what_the_policy_returns(self):
        policy = _FakePolicy()
        attach_refusal_recorder(policy)
        assert policy._fallback(_Context("b", 1, "p1"), None, "x") == "fallback:x"
        assert policy.stats.fallback_decisions == 1

    def test_at_and_reasons_locate_records(self):
        policy = _FakePolicy()
        recorder = attach_refusal_recorder(policy)
        _decide(policy, _Context("b", 4, "p1"), reason="no_worlds_constructed")
        _decide(policy, _Context("b", 9, "p2"), reason="crate_search_failed")
        assert recorder.at("b", 9, "p2").reason == "crate_search_failed"
        # Seat is part of the key: the two seats of one battle are different
        # decisions and a battle+round lookup would conflate them.
        assert recorder.at("b", 9, "p1") is None
        assert recorder.reasons() == Counter(
            {"no_worlds_constructed": 1, "crate_search_failed": 1}
        )


# --- the driver -------------------------------------------------------------


def _spec(
    *,
    battle_id: str = "hcgrid-hc-d4-600016",
    round_index: int = 47,
    seat: str = "p1",
    reason: str = "no_worlds_constructed",
    harness: str = HARNESS_ROLLOUT_HC_GRID,
    seed: int = 600016,
    fidelity: str = FIDELITY_EXACT,
) -> ReplaySpec:
    return ReplaySpec(
        battle_id=battle_id,
        round=round_index,
        seat=seat,
        reason=reason,
        key=f"fallback:{reason}",
        source="hc-d4.json",
        harness=harness,
        seed=seed,
        fidelity=fidelity,
        fidelity_notes=("note",) if fidelity != FIDELITY_EXACT else (),
    )


def _record(
    battle_id: str = "hcgrid-hc-d4-600016",
    round_index: int = 47,
    seat: str = "p1",
    reason: str = "no_worlds_constructed",
) -> RefusalRecord:
    return RefusalRecord(
        battle_id=battle_id, round=round_index, seat=seat, reason=reason
    )


class TestReplayOutcomes:
    def test_reproduced(self):
        result = replay_fallback(_spec(), lambda spec: [_record()])
        assert result.outcome is ReplayOutcome.REPRODUCED
        assert result.reproduced
        assert result.record is not None
        assert result.fidelity_caveat is None

    def test_reason_changed_is_not_a_reproduction(self):
        # Same decision, different refusal. A two-valued verdict has to call
        # this either a success (wrong: the recorded cause is gone) or a total
        # miss (wrong: the decision still refuses, and the new reason is the
        # next predicate in line -- the re-refusal the plan wants counted).
        result = replay_fallback(
            _spec(), lambda spec: [_record(reason="crate_search_failed")]
        )
        assert result.outcome is ReplayOutcome.REASON_CHANGED
        assert not result.reproduced
        assert result.record is not None
        assert result.record.reason == "crate_search_failed"

    def test_same_battle_different_round_is_its_own_outcome(self):
        result = replay_fallback(_spec(), lambda spec: [_record(round_index=12)])
        assert result.outcome is ReplayOutcome.SAME_BATTLE_DIFFERENT_ROUND
        assert result.outcome is not ReplayOutcome.ADDRESS_ABSENT
        assert result.record is None
        # ...and the refusals that DID happen are still handed back, because
        # under a diverged trajectory they are the only real ones available.
        assert len(result.all_records) == 1

    def test_wrong_seat_at_the_right_round_is_not_a_match(self):
        result = replay_fallback(_spec(seat="p1"), lambda spec: [_record(seat="p2")])
        assert result.outcome is ReplayOutcome.SAME_BATTLE_DIFFERENT_ROUND

    def test_no_refusal_at_all(self):
        result = replay_fallback(_spec(), lambda spec: [])
        assert result.outcome is ReplayOutcome.NO_REFUSAL
        assert result.all_records == ()

    def test_address_absent_when_only_other_battles_refused(self):
        result = replay_fallback(
            _spec(), lambda spec: [_record(battle_id="hcgrid-hc-d4-600021")]
        )
        assert result.outcome is ReplayOutcome.ADDRESS_ABSENT


class TestFidelityCaveat:
    def test_non_exact_fidelity_caveats_even_a_reproduction(self):
        # The load-bearing case. A foul-play address that happens to land on the
        # same round is EVIDENCE, not proof, because the trajectory that led
        # there was not pinned. Without this, a corpus gate launders the two.
        spec = _spec(
            battle_id="battle-gen3randombattle-controlled-8220001",
            round_index=103,
            harness=HARNESS_FOULPLAY_BRIDGE,
            seed=8220001,
            reason="crate_search_failed",
            fidelity=FIDELITY_OPPONENT_UNPINNED,
        )
        result = replay_fallback(
            spec,
            lambda s: [
                _record(
                    battle_id=spec.battle_id,
                    round_index=103,
                    reason="crate_search_failed",
                )
            ],
        )
        assert result.outcome is ReplayOutcome.REPRODUCED
        assert result.fidelity_caveat is not None
        assert FIDELITY_OPPONENT_UNPINNED in result.fidelity_caveat

    def test_exact_fidelity_carries_no_caveat(self):
        result = replay_fallback(_spec(), lambda spec: [_record()])
        assert result.fidelity_caveat is None


class TestRendering:
    def test_format_refusal_prints_both_sides_of_a_legality_mismatch(self):
        record = RefusalRecord(
            battle_id="b",
            round=7,
            seat="p1",
            reason="choices_unmapped",
            engine_choices={"substitute": 1.0},
            request_legal_choices=("surf", "rest"),
            unmapped_choices={"substitute": 1},
            choices_unmapped_causes={"all_unmapped_legality_mismatch": 1},
        )
        text = format_refusal(record)
        assert "engine proposed" in text and "substitute" in text
        assert "request offered: surf, rest" in text
        assert "proposed but NOT offered" in text

    def test_results_to_json_round_trips(self):
        import json

        result = replay_fallback(_spec(), lambda spec: [_record()])
        payload = json.loads(results_to_json([result]))
        assert payload[0]["outcome"] == "reproduced"
        assert payload[0]["spec"]["battle_id"] == "hcgrid-hc-d4-600016"
        assert payload[0]["record"]["round"] == 47


class TestRunnerRefusesWhatItCannotStandUp:
    def test_unsupported_harness_is_named(self):
        from _showdown_root import has_showdown, showdown_root_str

        from pokezero.fallback_replay import UnsupportedHarness, rollout_runner

        pytest.importorskip("poke_engine")
        pytest.importorskip("pokezero_search")
        if not has_showdown():
            pytest.skip(f"needs a Showdown checkout (looked in {showdown_root_str()})")
        runner = rollout_runner(showdown_root=showdown_root_str())
        with pytest.raises(UnsupportedHarness) as excinfo:
            runner(_spec(harness=HARNESS_FOULPLAY_BRIDGE))
        assert HARNESS_FOULPLAY_BRIDGE in str(excinfo.value)
