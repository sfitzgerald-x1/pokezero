#!/usr/bin/env python3
"""Measure what the always-on refusal recorder costs, on both axes.

The recorder is wired ON by default in ``foulplay_bridge``. A default-on
instrument has to justify itself with numbers, and the two numbers that matter
are not the same shape:

**Decision time.** The recorder wraps three bound methods. Two of them
(``select_action_with_context``, ``_map_choices``) run on every decision; the
third (``_fallback``) runs only when the decision refuses. So there are TWO
per-decision costs, and quoting only the cheap one would understate a run whose
fallback rate is high -- which is exactly the run the recorder exists for. Both
are measured, and blended at a caller-supplied fallback rate.

**Summary bytes.** ``foulplay_bridge._write_json`` rewrites the ENTIRE summary
after every completed game when ``--summary-out`` is set, so the document size
is re-serialized once per game and cumulative bytes written grow as
O(games^2). That is the axis the opponent-journal review found the previous
author had missed, so it is reported here as well as the flat per-shard delta.

Nothing here is synthetic if it does not have to be: pass ``--records-json``
(the ``records`` list of a real recorder dump) and the size half is measured on
real record contents rather than on a fabricated one.

Usage::

    python scripts/refusal_recorder_cost.py \\
        --decisions 20000 --records-json <dump>.json \\
        --games 8 --refusals-per-game 6.9 --per-game-row-bytes 3383 \\
        --base-summary-bytes 43921 --fallback-rate 0.01
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from pokezero.actions import ACTION_COUNT
from pokezero.engine_search import EngineMctsStats
from pokezero.fallback_replay import RefusalRecord, attach_refusal_recorder
from pokezero.observation import PokeZeroObservationV0

# A refusal class from the real era 61-64 corpus, so the record being timed and
# sized carries a real key rather than a short synthetic one.
_TRAPPED = (
    "self_request_state_unsupported: self active request flags ['trapped'] constrain "
    "legality beyond this construction (sampled world does not trap: foe ability 'insomnia')"
)


class _Context:
    def __init__(self, battle_id: str, round_index: int, seat: str = "p1") -> None:
        self.battle_id = battle_id
        self.decision_round_index = round_index
        self.player_id = seat
        self.seed = 8220000
        self.observation = PokeZeroObservationV0(
            categorical_ids=(),
            numeric_features=(),
            token_type_ids=(),
            attention_mask=(),
            legal_action_mask=(True,) * ACTION_COUNT,
            metadata={
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "thunderbolt"},
                    {"action_index": 1, "kind": "move", "legal": True, "move_id": "shadowball"},
                    {"action_index": 2, "kind": "move", "legal": True, "move_id": "thunderwave"},
                    {"action_index": 3, "kind": "move", "legal": True, "move_id": "painsplit"},
                    {"action_index": 4, "kind": "switch", "legal": True,
                     "pokemon": {"species": "Salamence"}},
                    {"action_index": 5, "kind": "switch", "legal": True,
                     "pokemon": {"species": "Snorlax"}},
                ]
            },
        )


class _Policy:
    """The three methods the recorder wraps, over a REAL ``EngineMctsStats``."""

    def __init__(self) -> None:
        self.stats = EngineMctsStats()

    def select_action_with_context(self, context: Any, *, rng: Any) -> None:
        return None

    def _map_choices(self, context: Any, aggregated: Any) -> None:
        return None

    def _fallback(self, context: Any, rng: Any, reason: str) -> None:
        self.stats.fallback_decisions += 1
        return None


def _decision(policy: _Policy, context: _Context, *, refuse: bool) -> None:
    policy.select_action_with_context(context, rng=None)
    policy.stats.worlds_attempted += 16
    if refuse:
        policy.stats.world_failure_reasons[_TRAPPED] += 16
        policy._map_choices(context, {"thunderbolt": 0.6, "shadowball": 0.4})
        policy._fallback(context, None, "no_worlds_constructed")
    else:
        policy.stats.worlds_constructed += 16
        policy.stats.worlds_searched += 16
        policy._map_choices(context, {"thunderbolt": 0.6, "shadowball": 0.4})


def _time(decisions: int, *, refuse: bool, record: bool, repeats: int) -> float:
    """Seconds per decision, median of ``repeats`` timing runs."""
    samples = []
    for _ in range(repeats):
        policy = _Policy()
        contexts = [_Context("battle-1", index % 250) for index in range(decisions)]
        recorder = attach_refusal_recorder(policy) if record else None
        try:
            started = time.perf_counter()
            for context in contexts:
                _decision(policy, context, refuse=refuse)
            samples.append((time.perf_counter() - started) / decisions)
        finally:
            if recorder is not None:
                recorder.detach()
    return statistics.median(samples)


def _sample_records(path: Path | None) -> list[dict]:
    if path is None:
        return [
            RefusalRecord(
                battle_id="battle-gen3randombattle-controlled-8220000",
                round=17,
                seat="p1",
                reason="no_worlds_constructed",
                world_failures={_TRAPPED: 10, _TRAPPED.replace("insomnia", "swarm"): 6},
                worlds_attempted=16,
                request_legal_choices=("thunderbolt", "shadowball", "thunderwave", "painsplit"),
                decision_rng_seed="8220000:p1:17",
            ).to_dict()
        ]
    document = json.loads(path.read_text(encoding="utf-8"))
    records = document.get("records") if isinstance(document, dict) else document
    if not isinstance(records, list) or not records:
        raise SystemExit(f"{path} carries no `records` list")
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--fallback-rate", type=float, default=0.01,
                        help="Blend weight for the refusing cost. Default is the "
                             "measured era-64 cell-D rate.")
    parser.add_argument("--decision-boundary-seconds", type=float, default=5.02,
                        help="Prior measured median decision boundary on our side.")
    parser.add_argument("--records-json", type=Path, default=None,
                        help="A recorder dump; its `records` are used for the size half.")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--refusals-per-game", type=float, default=6.9)
    parser.add_argument("--per-game-row-bytes", type=int, default=3383)
    parser.add_argument("--base-summary-bytes", type=int, default=43921)
    parser.add_argument("--long-run-games", type=int, default=250)
    args = parser.parse_args(argv)

    clean = _time(args.decisions, refuse=False, record=False, repeats=args.repeats)
    clean_rec = _time(args.decisions, refuse=False, record=True, repeats=args.repeats)
    refused = _time(args.decisions, refuse=True, record=False, repeats=args.repeats)
    refused_rec = _time(args.decisions, refuse=True, record=True, repeats=args.repeats)

    non_refusing = clean_rec - clean
    refusing = refused_rec - refused
    blended = (1 - args.fallback_rate) * non_refusing + args.fallback_rate * refusing

    print(f"DECISION TIME  ({args.decisions} decisions x {args.repeats} runs, median)")
    print(f"  non-refusing decision  {non_refusing * 1e6:8.2f} us "
          f"({non_refusing / args.decision_boundary_seconds:.2e} of a "
          f"{args.decision_boundary_seconds} s boundary)")
    print(f"  refusing decision      {refusing * 1e6:8.2f} us "
          f"({refusing / args.decision_boundary_seconds:.2e} of the boundary)")
    print(f"  blended at {args.fallback_rate:.2%} fallback rate  {blended * 1e6:8.2f} us")

    records = _sample_records(args.records_json)
    sizes = [len(json.dumps(record, sort_keys=True).encode("utf-8")) for record in records]
    median_record = statistics.median(sizes)
    per_game = median_record * args.refusals_per_game
    shard_delta = per_game * args.games
    base_shard = args.base_summary_bytes

    print()
    print(f"SUMMARY BYTES  ({len(records)} real records"
          f"{' from ' + str(args.records_json) if args.records_json else ' (synthetic)'})")
    print(f"  per refusal record     {median_record:8.0f} B median "
          f"(min {min(sizes)}, max {max(sizes)})")
    print(f"  per game               {per_game:8.0f} B at {args.refusals_per_game} refusals/game "
          f"(+{per_game / args.per_game_row_bytes:.1%} of a {args.per_game_row_bytes} B row)")
    print(f"  {args.games}-game shard          {shard_delta:8.0f} B on a {base_shard} B summary: "
          f"+{shard_delta / base_shard:.1%}")

    # `_write_json` rewrites the whole document per game: cumulative bytes are the
    # sum over games of the document size at that point.
    def cumulative(records_on: bool) -> float:
        total = 0.0
        for games_done in range(1, args.long_run_games + 1):
            row = args.per_game_row_bytes + (per_game if records_on else 0.0)
            total += base_shard - args.per_game_row_bytes * args.games + row * games_done
        return total

    off = cumulative(False)
    on = cumulative(True)
    print()
    print(f"O(games^2) SERIALIZATION  (--summary-out, {args.long_run_games} games)")
    print(f"  records off  {off / 1e6:8.1f} MB")
    print(f"  records on   {on / 1e6:8.1f} MB   (+{(on - off) / 1e6:.1f} MB, "
          f"+{(on - off) / off:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
