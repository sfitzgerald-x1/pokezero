#!/usr/bin/env python3
"""Merge FoulPlay power-config shards into the campaign's deliverable metric.

The campaign's whole output is a **paired delta (search − raw) per cell**, and
until this existed nothing computed one: the shards carried raw game rows and
the plan described the statistics, but no merger, Wilson interval, bootstrap or
McNemar test lived anywhere in either repo.

What "paired" means here, precisely, and why the join is strict:

    A pair is one (battle seed, seat). The search arm and the raw arm played
    that same seed from that same seat against the same FoulPlay build, so the
    two rows differ only by search config. The delta is the mean of per-pair
    differences, NOT the difference of two arms' means -- those coincide only
    when both arms cover exactly the same pairs, which is the thing most likely
    to be false in a partially-failed campaign.

Fail-closed rules, each chosen because the alternative silently produces a
plausible number:

* a pair present in one arm and missing in the other is DROPPED and COUNTED,
  never half-scored;
* shards from two different engine builds refuse to merge (`--expect-fingerprint`);
* two shards claiming the same (config_id, seed, seat) with different outcomes
  is terminal, not last-write-wins;
* a cell whose `search_wall_per_searched_decision` mean exceeds the cap is
  reported REJECTED and its delta is not eligible for adoption;
* a depth cell that does not out-reach its reference is BUDGET-STARVED and
  excluded, rather than being read as a null (the confound the depth axis
  exists to avoid). Requires `--campaign` to know each cell's reference; the
  report records whether the rule was applied.

Usage::

    python scripts/foulplay_power_report.py results/*.json \\
        --expect-fingerprint <sha256> --out report.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

SCHEMA_VERSION = "pokezero.foulplay-power-report.v1"
# Section 6, binding. Reported against the mean; p95 is reported, not gated.
LATENCY_CAP_SECONDS = 20.0
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260730
# Section 8: "a rise above ~2% means FoulPlay is steering games into
# world-construction gaps". Gates eligibility, per section 9 Phase 2 (ii).
FALLBACK_LIMIT = 0.02
# Section 8: ">= 400 pairs per cell (200 per seat)".
MIN_PAIRS = 400


def wilson(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Same estimator the acceptance report uses."""
    if n == 0:
        return (0.0, 0.0)
    phat = wins / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return ((centre - margin) / denom, (centre + margin) / denom)


def mcnemar(deltas: list[float]) -> dict:
    """Discordant-pair counts and a normal-approximation two-sided z.

    Reported alongside the bootstrap because the two answer different
    questions: the bootstrap bounds the size of the delta, McNemar asks whether
    the DISCORDANT pairs are lopsided. The July-30 baseline was quoted as
    "+0.058 (discordant 87/64)" and this reproduces that shape.
    """
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    n = wins + losses
    if n == 0:
        return {"search_better": 0, "raw_better": 0, "discordant": 0, "z": None,
                "note": "no discordant pairs"}
    z = (wins - losses) / math.sqrt(n)
    return {"search_better": wins, "raw_better": losses, "discordant": n,
            "z": round(z, 3)}


def load_shards(paths: list[str]) -> list[dict]:
    """Read the shards, REFUSING any whose witness schema this reader did not write.

    The check used to be `schema_version != "...v1"` and nothing else, which made the
    one string it validated the one thing that could not distinguish two shapes: this
    branch and `phase1/rollout-model-priors` were both widening
    `per_seat[*].policy_stats` with a rollout-leaf witness -- different key NAME,
    different UNIT, different PRESENCE rule, and a different unit again for the mode
    tally -- under a byte-identical `schema_version`. See the long note at
    `foulplay_paired_eval.SCHEMA_VERSION` for the shape that was agreed and why the
    version stamp lives inside the witness block rather than in a second envelope
    string.

    Four refusals, and NONE of them coerces:

    * an UNKNOWN envelope version. Named individually so a future v2 envelope is a
      refusal here and not a hopeful read of a shape this code cannot interpret.
    * a witness with NO `rollout_leaf_schema` stamp. That is the pre-adoption shape --
      either branch's, since neither stamped -- and it is the one case that cannot be
      resolved after the fact, because "written before the rename" and "written by a
      writer that dropped a field" are otherwise the same artifact.
    * a witness stamped at a schema this reader does not implement.
    * a witness carrying `rollout_leaf_world_records`, this branch's superseded name
      for the world counter. Renaming it on the way in would be the coercion the whole
      check exists to stop: `+= 1` and `+= weight` are different quantities the moment
      a duplicate belief draw collapses.

    ... and one positive requirement, which is what makes the PRESENCE rule real: a
    witnessed seat must carry EVERY key in `ROLLOUT_WITNESS_KEYS`. The arm-off
    readings are `{}`, `0` and `null`, all present. A seat missing them was written by
    a conditional-presence writer, under which an arm-off run and a pre-arm shard are
    indistinguishable -- and this module's own absent-is-not-false reasoning
    (`rollout_leaf_of_shard`) rests on being able to tell those apart.

    A silent re-schema of banked data is worse than a crash, so every one of these is
    `SystemExit` and none is a warning.
    """
    from foulplay_paired_eval import (  # noqa: PLC0415 — sibling script, lazy like `search_config_id`
        ROLLOUT_WITNESS_KEYS,
        ROLLOUT_WITNESS_QUOTIENT_KEYS,
        ROLLOUT_WITNESS_STAMP,
        ROLLOUT_WITNESS_SUPERSEDED_KEYS,
        SCHEMA_VERSION as SHARD_SCHEMA_VERSION,
        current_witness_schema,
    )

    expected_schema = current_witness_schema()

    shards = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        schema = payload.get("schema_version")
        if schema != SHARD_SCHEMA_VERSION:
            raise SystemExit(
                f"{path}: unexpected schema_version {schema!r}; this reader writes and "
                f"reads {SHARD_SCHEMA_VERSION!r}. Refusing rather than guessing: a "
                "shard whose envelope is unknown cannot be merged, and coercing it "
                "would re-schema banked data silently."
            )
        for seat, block in sorted((payload.get("per_seat") or {}).items()):
            stats = (block or {}).get("policy_stats")
            if not isinstance(stats, dict):
                # A raw-arm seat has no engine at all. Nothing to check, and the arm
                # is already witnessed by `arm` and `config_id`.
                continue
            superseded = [k for k in ROLLOUT_WITNESS_SUPERSEDED_KEYS if k in stats]
            if superseded:
                raise SystemExit(
                    f"{path}: seat {seat} carries {', '.join(superseded)}, the "
                    "SUPERSEDED name for the rollout world counter. That writer "
                    "accumulated `+= 1` per crate report; the current one accumulates "
                    "`+= weight` per world, and the two differ whenever a duplicate "
                    "belief draw collapsed. Refused rather than renamed -- coercing "
                    "one into the other is the silent re-schema this check exists to "
                    "stop. Migrate it (#1271 ships `migrate_rollout_leaf_shard_v1`, "
                    "which refuses unless `worlds_collapsed == 0` and "
                    "`rollout_dead_ends == 0`, the conditions under which the two "
                    "rules provably agree), then re-bank."
                )
            witness = [key for key in ROLLOUT_WITNESS_KEYS if key in stats]
            if not witness:
                # Not a rollout-era writer at all: a seat from before the arm existed.
                # Nothing to validate, and `rollout_leaf_of_shard` relies on exactly
                # this case being distinguishable from an arm-off one.
                continue
            stamp = stats.get(ROLLOUT_WITNESS_STAMP)
            if stamp is None:
                raise SystemExit(
                    f"{path}: seat {seat} carries a rollout witness "
                    f"({', '.join(witness)}) with no {ROLLOUT_WITNESS_STAMP!r}. An "
                    "unstamped block is unresolvable after the fact -- 'written before "
                    "the rename' and 'written by a writer that dropped a field' are the "
                    "same artifact. Both pre-adoption writers produced this shape."
                )
            if stamp != expected_schema:
                raise SystemExit(
                    f"{path}: seat {seat} is stamped "
                    f"{ROLLOUT_WITNESS_STAMP}={stamp!r}; this reader implements "
                    f"{expected_schema!r}. Refused rather than read on the "
                    "assumption that the fields it knows mean what they used to."
                )
            missing = [key for key in ROLLOUT_WITNESS_KEYS if key not in stats]
            # THE ONE LEGITIMATELY PARTIAL BLOCK, and the reason this exemption is
            # here rather than argued about at merge time.
            #
            # This reader REFUSED A #1271 SHARD. Measured, not predicted: with the arm
            # engaged and `rollouts_run == 0`, #1271's writer OMITS the three
            # quotients, because a quotient of an empty partition has no value -- and
            # this loop then reported them as "a writer that encodes 'off' as
            # absence" and exited. The two branches disagreed on the presence axis and
            # the disagreement survived deleting this branch's WRITER, because what
            # collides here is this branch's READER.
            #
            # Resolved in #1271's favour, which is also the direction the campaign's
            # own reasoning points once the cases are separated:
            #   * ARM OFF -> the whole block is absent, and that is caught above by
            #     `if not witness: continue` -- an arm-off seat is already
            #     distinguishable from a pre-arm one by `rollout_leaf_of_shard`, which
            #     reads the CONFIG echo rather than guessing from telemetry. "Absent is
            #     not false" is about that echo, and it still holds.
            #   * ARM ON, NO ROLLOUT LAUNCHED -> every COUNT is present and required;
            #     only the three QUOTIENTS are absent, because there is no denominator.
            #     `null` is worse than absent here, not better: a pooling reader
            #     averages a `null` as though it were a measurement, and v1's
            #     unconditional writer emitted exactly that.
            # So the counts are still required unconditionally and the exemption is
            # scoped by VALUE to `rollouts_run == 0`, matching
            # `require_rollout_leaf_shard_schema` key for key -- and #1271 additionally
            # REFUSES a quotient that is present at zero rollouts, so the two readers
            # cannot drift into disagreeing about this block again.
            if int(stats.get("rollouts_run") or 0) == 0:
                missing = [
                    key for key in missing if key not in ROLLOUT_WITNESS_QUOTIENT_KEYS
                ]
            if missing:
                raise SystemExit(
                    f"{path}: seat {seat} is a stamped rollout witness missing "
                    f"{', '.join(missing)}. Every COUNT is emitted whenever the arm "
                    "engaged, so a missing one is not an arm-off cell -- an arm-off "
                    "cell has no block at all and is handled above -- it is a writer "
                    "that dropped a field. The three QUOTIENTS are exempt only when "
                    "`rollouts_run == 0`, where they have no denominator."
                )
        payload["_path"] = path
        shards.append(payload)
    if not shards:
        raise SystemExit("no shards given")
    return shards


def assert_single_build(shards: list[dict], expect: str | None) -> str:
    """One build era for the whole campaign (the #952/#963 standard)."""
    prints = {s.get("engine_fingerprint") for s in shards}
    if len(prints) != 1:
        raise SystemExit(
            f"shards span {len(prints)} engine builds: {sorted(map(str, prints))}. "
            "A campaign is ONE build era; refusing to merge across builds."
        )
    fingerprint = prints.pop()
    if expect is not None and fingerprint != expect:
        raise SystemExit(
            f"engine fingerprint {fingerprint} does not match --expect-fingerprint {expect}"
        )
    return fingerprint


def rollout_leaf_of_shard(shard: dict) -> bool:
    """A shard's rollout-leaf flag, with absence resolved rather than defaulted.

    This was `bool(shard.get("rollout_leaf", False))`, written twice -- literally
    the construct the comment on `rollout_leaf_of` (the CAMPAIGN-CELL version, 300
    lines down) condemns: "ABSENT IS NOT FALSE, and `bool(cell.get(...))` made it
    False". The fix landed for cells and not for shards.

    It is resolved here rather than merely renamed, and the resolution is the
    WITNESS, not a default. Stated as narrowly as it is enforced:

    * a shard whose seats carry no rollout witness at all was written before the arm
      existed, so an absent flag cannot be hiding a rollout game -- which is the only
      thing absent-is-false could get wrong here. False is a conclusion from the
      witness, not a default. `load_shards` is what makes that reliable: it refuses a
      witness that is unstamped, superseded, or partial, so "no witness" really does
      mean "no rollout run" rather than "a witness this reader failed to recognise".
    * a shard whose seats DO carry the witness was written by `rollout_body_fields`,
      which emits the flag unconditionally and refuses a body disagreeing with the
      cell id. So the key is present, and its absence is a dropped field -- refused,
      not defaulted, because reading it as False files the arm under its own
      value-head control and manufactures the null.
    * a non-boolean is refused either way, because `bool("false")` is True and would
      file the value-head control under the arm.
    """
    from foulplay_paired_eval import (  # noqa: PLC0415
        ROLLOUT_WITNESS_KEYS,
        ROLLOUT_WITNESS_QUOTIENT_KEYS,
        ROLLOUT_WITNESS_STAMP,
    )

    if "rollout_leaf" not in shard:
        witnessed = any(
            ROLLOUT_WITNESS_STAMP in ((block or {}).get("policy_stats") or {})
            or any(
                key in ((block or {}).get("policy_stats") or {})
                for key in ROLLOUT_WITNESS_KEYS
            )
            for block in (shard.get("per_seat") or {}).values()
        )
        if not witnessed:
            return False
        raise SystemExit(
            f"{shard.get('_path')}: no `rollout_leaf` in a shard whose seats carry the "
            "rollout witness. Every writer that emits the witness also emits this flag "
            "unconditionally (`rollout_body_fields`), so absence is a dropped field "
            "rather than an arm-off cell -- and reading it as False files the arm "
            "under its own value-head control, which manufactures the null."
        )
    value = shard["rollout_leaf"]
    if not isinstance(value, bool):
        raise SystemExit(
            f"{shard.get('_path')}: rollout_leaf={value!r} is not a boolean. Every "
            "non-empty string is truthy, so a string here would file a value-head "
            "shard under the arm."
        )
    return value


def collect_rows(shards: list[dict]) -> tuple[dict, dict]:
    """(config_id -> {(seed, seat): score}), and config_id -> shard metadata."""
    rows: dict[str, dict[tuple[int, str], float]] = defaultdict(dict)
    meta: dict[str, dict] = {}
    for shard in shards:
        cid = shard["config_id"]
        # RESOLVED ONCE per shard, and read twice from that one value. It used to be
        # `bool(shard.get("rollout_leaf", False))` written out at both sites, which
        # is two readers of one key -- and the second refusing first made the first
        # one's coercion unobservable, i.e. a mutant that reverted only the recorded
        # value survived the suite as an equivalent. One reader, one rule.
        rollout_leaf = rollout_leaf_of_shard(shard)
        meta.setdefault(cid, {"arm": shard["arm"], "shards": [], "per_seat": [],
                              "checkpoint": shard.get("checkpoint"),
                              "opponent_priors": shard.get("opponent_priors", False),
                              # config_id CARRIES this one, so every shard of a
                              # cell must agree; disagreement is asserted below.
                              "oracle_belief": bool(shard.get("oracle_belief", False)),
                              # config_id carries this one too (`+rollout<R>p<cap>`),
                              # so the same agreement rule applies and is asserted
                              # below. Recorded as well as keyed, because the id can
                              # be recomputed wrongly and the witness cannot -- and
                              # because the merged report otherwise never states
                              # which cells priced their leaves by rollout.
                              "rollout_leaf": rollout_leaf,
                              # config_id deliberately does NOT carry this one --
                              # it is observational, so telemetry-on and
                              # telemetry-off are the same search and pool into one
                              # cell. Counted per shard rather than collapsed to a
                              # boolean, because an override rate whose denominator
                              # is a SUBSET of the cell's games has to be readable
                              # as such (see search_config_id's note).
                              "override_telemetry_shards": Counter()})
        if bool(shard.get("oracle_belief", False)) != meta[cid]["oracle_belief"]:
            # Not a warning. The oracle and sampled arms are the two halves of
            # §4a's split; pooled, the centerpiece figure is their average.
            raise SystemExit(
                f"cell {cid} pools oracle-belief and sampled-belief shards "
                f"({shard['_path']}). config_id must carry +oracle-belief -- an "
                "older driver wrote one of these shards."
            )
        if rollout_leaf != meta[cid]["rollout_leaf"]:
            # Not a warning either, and for a strictly stronger reason than the
            # oracle split above: the arbiter arm's ENTIRE reading is rollout-leaf
            # against the same config with the leaf off. Pooled, the centerpiece
            # figure is the average of the experiment and its own control, which
            # reads as "the leaf made no difference" by construction -- i.e. the
            # merge does not merely blur the answer, it manufactures the null.
            # Reachable exactly one way: a driver that predates the
            # `+rollout<R>p<cap>` fragment wrote one of these shards.
            raise SystemExit(
                f"cell {cid} pools rollout-leaf and value-head shards "
                f"({shard['_path']}). config_id must carry +rollout<R>p<cap> -- an "
                "older driver wrote one of these shards."
            )
        meta[cid]["override_telemetry_shards"][
            "on" if shard.get("override_telemetry") else "off"
        ] += 1
        meta[cid]["shards"].append(shard["_path"])
        meta[cid]["per_seat"].append(shard.get("per_seat", {}))
        for row in shard.get("rows", []):
            key = (int(row["seed"]), row["seat"])
            score = float(row["score"])
            if key in rows[cid] and rows[cid][key] != score:
                raise SystemExit(
                    f"conflicting scores for {cid} seed {key[0]} seat {key[1]}: "
                    f"{rows[cid][key]} vs {score}. Two shards disagree about the "
                    "same game; refusing to pick one."
                )
            rows[cid][key] = score
    return rows, meta


def latency_of(meta_entry: dict) -> dict:
    """Mean/p95 of the GATE field across a cell's seats, plus the other wall.

    THE GATE FIELD IS PER-DECISION WHEN THE SHARD OFFERS ONE. On a dynamic-budget
    cell `search_wall_per_searched_decision` is per-RUNG -- `searched_decisions` is
    charged once per `_search_model` call and a ladder calls it once per rung
    (measured 2,224 rungs against 1,062 decisions) -- so gating on it lets a cell
    at 2.1 rungs/decision report 5.7 s while its true per-decision wall is 12 s,
    and the 20 s/turn cap silently stops gating on exactly the cells this feature
    exists to produce. `search_wall_per_ladder_decision` is hoisted into every
    shard's seat block for this reason; prefer it, and say which one was used so a
    reader is never guessing. Found in review.
    """
    gate, p95, other, rungs = [], [], [], []
    per_decision_seats = 0
    per_rung_seats = 0
    for per_seat in meta_entry["per_seat"]:
        for seat in (per_seat or {}).values():
            ladder_wall = seat.get("search_wall_per_ladder_decision")
            if ladder_wall is not None:
                gate.append(float(ladder_wall))
                per_decision_seats += 1
            elif seat.get("search_wall_per_searched_decision") is not None:
                gate.append(float(seat["search_wall_per_searched_decision"]))
                per_rung_seats += 1
            if seat.get("ladder_rungs_per_decision") is not None:
                rungs.append(float(seat["ladder_rungs_per_decision"]))
            if seat.get("wall_per_decision_p95") is not None:
                p95.append(float(seat["wall_per_decision_p95"]))
            if seat.get("wall_per_decision_mean") is not None:
                other.append(float(seat["wall_per_decision_mean"]))
    return {
        "search_wall_per_searched_decision_mean": (sum(gate) / len(gate)) if gate else None,
        # WHICH denominator the line above is on. A cell that mixes them is a cell
        # whose shards were not all built from one image, which `assert_single_build`
        # already refuses -- but if it ever happens, the reader must see it.
        "gate_denominator": (
            None
            if not gate
            else "per_ladder_decision"
            if per_decision_seats and not per_rung_seats
            else "per_searched_decision_PER_RUNG"
            if per_rung_seats and not per_decision_seats
            else "MIXED - do not compare"
        ),
        "ladder_rungs_per_decision_mean": (sum(rungs) / len(rungs)) if rungs else None,
        "wall_per_decision_p95_max": max(p95) if p95 else None,
        "wall_per_decision_mean": (sum(other) / len(other)) if other else None,
    }


def health_of(meta_entry: dict) -> dict:
    fallback, depth = [], []
    opponent_fallback: list[float] = []
    reasons: dict[str, int] = {}
    for per_seat in meta_entry["per_seat"]:
        for seat in (per_seat or {}).values():
            if seat.get("fallback_rate") is not None:
                fallback.append(float(seat["fallback_rate"]))
            if seat.get("depth_reached_mean") is not None:
                depth.append(float(seat["depth_reached_mean"]))
            for reason, count in (seat.get("world_failure_reasons") or {}).items():
                reasons[reason] = reasons.get(reason, 0) + int(count)
            # HEAD-TO-HEAD: the opponent seat searches too, and its health is not in
            # `fallback_rate`, which describes the pokezero seat alone. A cell whose
            # OPPONENT was falling back is contaminated in exactly the way the eligibility
            # gate exists to catch, and without this it clears the gate on the other
            # seat's clean number.
            opp = seat.get("opponent_engine_mcts") or {}
            if opp.get("fallback_rate") is not None:
                opponent_fallback.append(float(opp["fallback_rate"]))
    return {
        "fallback_rate": (sum(fallback) / len(fallback)) if fallback else None,
        "opponent_fallback_rate": ((sum(opponent_fallback) / len(opponent_fallback))
                                   if opponent_fallback else None),
        "depth_reached_mean": (sum(depth) / len(depth)) if depth else None,
        "world_failure_reasons": dict(sorted(reasons.items())),
    }


def paired_improvement(candidate: dict, cand_raw: dict, anchor: dict, anchor_raw: dict):
    """CI on (candidate_delta - anchor_delta), per pair.

    Section 9 Phase 2 (iii). Each cell's per-pair score is its own delta
    against ITS OWN raw arm, and the improvement is the paired difference of
    those deltas.

    Subtracting the raw arms explicitly matters: when candidate and anchor
    share a raw arm the terms cancel and this reduces to `candidate - anchor`,
    but cell G runs on k1 against R1 while the anchor runs on k0 against R0, so
    they do NOT cancel. An earlier version computed `candidate - anchor`
    unconditionally. Measured on the committed cross-checkpoint fixture
    (`CrossCheckpointImprovementTest`), that gives **-0.475** where the true
    difference of deltas is **+0.025** -- wrong in magnitude AND SIGN, on the
    campaign's designed checkpoint contrast.
    """
    from pokezero.mcts_eval.scoring import bootstrap_mean

    shared = sorted(
        set(candidate) & set(cand_raw) & set(anchor) & set(anchor_raw)
    )
    if not shared:
        return None
    # The caller records this. A cell can clear --min-pairs on its OWN delta
    # while overlapping the anchor on far fewer pairs, and the improvement CI
    # is computed over the OVERLAP -- a 20-pair overlap can yield
    # "+1.000 [+1.000, +1.000]" beside a reported `pairs: 400`.
    deltas = [
        (candidate[k] - cand_raw[k]) - (anchor[k] - anchor_raw[k]) for k in shared
    ]
    interval = bootstrap_mean(deltas, _indices(len(shared)))
    return interval, len(shared)


def score_cell(search: dict, raw: dict, indices) -> dict:
    """Paired delta over the pairs BOTH arms actually played."""
    from pokezero.mcts_eval.scoring import bootstrap_mean, bootstrap_paired_delta

    shared = sorted(set(search) & set(raw))
    only_search = sorted(set(search) - set(raw))
    only_raw = sorted(set(raw) - set(search))
    if not shared:
        return {"pairs": 0, "error": "no shared (seed, seat) pairs between the arms"}
    t = [search[k] for k in shared]
    b = [raw[k] for k in shared]
    delta = bootstrap_paired_delta(t, b, indices)
    per_seat = {}
    for seat in ("p1", "p2"):
        keys = [k for k in shared if k[1] == seat]
        if not keys:
            continue
        st = [search[k] for k in keys]
        sb = [raw[k] for k in keys]
        seat_idx = _indices(len(keys))
        seat_delta = bootstrap_paired_delta(st, sb, seat_idx)
        per_seat[seat] = {
            "pairs": len(keys),
            "search_rate": sum(st) / len(st),
            "raw_rate": sum(sb) / len(sb),
            "paired_delta": seat_delta.to_payload(),
        }
    return {
        "pairs": len(shared),
        "dropped_unpaired": {"search_only": len(only_search), "raw_only": len(only_raw)},
        "search_rate": sum(t) / len(t),
        "search_wilson95": list(wilson(sum(t), len(t))),
        "raw_rate": sum(b) / len(b),
        "raw_wilson95": list(wilson(sum(b), len(b))),
        "paired_delta": delta.to_payload(),
        "mcnemar": mcnemar([x - y for x, y in zip(t, b)]),
        "per_seat": per_seat,
    }


def _indices(n: int):
    from pokezero.mcts_eval.scoring import bootstrap_indices

    return bootstrap_indices(sample_size=n, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)


def seat_gap_flag(cell: dict) -> str | None:
    """#937 bug class: a search-arm seat gap the raw arm does not show."""
    seats = cell.get("per_seat") or {}
    if set(seats) != {"p1", "p2"}:
        return None
    d1, d2 = seats["p1"]["paired_delta"], seats["p2"]["paired_delta"]
    disjoint = d1["low"] > d2["high"] or d2["low"] > d1["high"]
    return "STOP-AND-INVESTIGATE: per-seat paired deltas disjoint at 95%" if disjoint else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--expect-fingerprint", default=None)
    ap.add_argument("--anchor", default=None,
                    help="config_id of the anchor cell (A); winners must beat it")
    ap.add_argument("--cap-seconds", type=float, default=LATENCY_CAP_SECONDS)
    ap.add_argument("--min-pairs", type=int, default=MIN_PAIRS,
                    help="section 8 minimum per cell (default %(default)s). Lower it "
                         "only for a deliberately partial read, and say so in the write-up: "
                         "the campaign's own acceptance criterion is the default.")
    ap.add_argument("--campaign", default=None,
                    help="campaign JSON; supplies each depth cell's reads_against "
                         "reference so the section 5 non-starvation rule can be applied")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    shards = load_shards(args.shards)
    fingerprint = assert_single_build(shards, args.expect_fingerprint)
    rows, meta = collect_rows(shards)

    # config_id -> the config_id its depth evidence is read against. Empty
    # without --campaign, in which case depth cells are scored WITHOUT the
    # non-starvation rule and the report says so.
    depth_reference: dict[str, str] = {}
    if args.campaign:
        campaign = json.loads(Path(args.campaign).read_text(encoding="utf-8"))
        by_cell = {c["cell_id"]: c for c in campaign.get("cells", [])}

        # THE driver's builder, imported rather than re-implemented. The two
        # copies drifting is a silent failure of exactly the kind the tag
        # comment below records, and the selection knobs doubled the surface.
        from foulplay_paired_eval import search_config_id  # noqa: PLC0415

        # ABSENT IS NOT FALSE, and `bool(cell.get("rollout_leaf"))` made it False.
        #
        # Every other knob below loses at most a suffix to an absent key. This one
        # loses the WHOLE `+rollout<R>p<cap>` fragment, and what it renders instead
        # is not a broken id that matches no shard -- it is byte-identical to the
        # VALUE-HEAD CONTROL's id, which is a real shard. So a rollout cell whose
        # `rollout_leaf` key is missing (renamed, typo'd, dropped by a launcher that
        # predates the arm) does not fail to resolve: it resolves to the thing the
        # arm is measured against, and `depth_reference` then points the §5
        # non-starvation rule at the control while reporting clean.
        #
        # A cell that says `false` and a cell that says nothing are therefore
        # different inputs and must not produce the same answer. Two checks, because
        # neither alone is enough:
        #
        #   * here, per cell: the arm's own knobs present with no `rollout_leaf` is
        #     an incoherent cell, and a non-boolean flag is a control filed as the
        #     arm (`bool("false")` is True);
        #   * at the `depth_reference` assignment below, structurally: a cell whose
        #     reference id EQUALS its own id is reading against itself, which is what
        #     a lost fragment actually produces. That check needs no guess about
        #     WHICH key went missing, and it is the one that catches the case where
        #     the flag and all its knobs vanished together.
        #
        # What is deliberately NOT done: requiring the key on every cell once any
        # cell declares it. The fixture in
        # `tests/test_foulplay_power_report.RolloutCellReferenceIdTest` is the normal
        # shape and refutes it -- the value-head CONTROL legitimately omits the flag
        # while its rollout twin carries it, so that rule refuses correct campaigns.
        # Measured, not reasoned: it was written that way and the fixture failed.
        #: The knobs that only exist because the arm does. Any one of them present
        #: without `rollout_leaf` is an incoherent cell whichever way the campaign
        #: as a whole reads.
        _ROLLOUT_SIBLINGS = (
            "rollout_count", "rollout_max_plies", "rollout_policy",
            "rollout_seed", "rollout_threads",
        )

        def rollout_leaf_of(cell):
            """The cell's arm flag, with absent refused rather than read as False."""
            present = "rollout_leaf" in cell
            siblings = [key for key in _ROLLOUT_SIBLINGS if key in cell]
            if not present and siblings:
                raise SystemExit(
                    f"campaign cell {cell.get('cell_id')!r} carries {siblings} but no "
                    "'rollout_leaf': absent is not false. Refused rather than rendered, "
                    "because dropping the fragment renders the VALUE-HEAD CONTROL's id "
                    "-- a real shard -- so the arm would be measured against itself."
                )
            value = cell.get("rollout_leaf", False)
            if not isinstance(value, bool):
                # `"false"` is truthy. A string here would turn a control into the arm.
                raise SystemExit(
                    f"campaign cell {cell.get('cell_id')!r} has rollout_leaf="
                    f"{value!r}: a real boolean is required, because every non-empty "
                    "string is truthy and would file a value-head cell under the arm."
                )
            return value

        def cid_of(cell):
            # The campaign KEY (k0/k1), matching what the launcher passes as
            # --checkpoint-tag. Deriving it from the checkpoint path instead
            # silently produced ids that matched no shard, so depth_reference
            # was populated, `depth_rule_applied` reported true, and the §5
            # non-starvation rule never fired for any cell.
            tag = cell["checkpoint"]
            if cell["arm"] == "raw":
                return f"raw@{tag}"
            return search_config_id(
                depth=cell["depth"], sims=cell["sims"],
                batch=cell["batch"], worlds=cell["worlds"], tag=tag,
                opponent_priors=bool(cell.get("opponent_priors")),
                fpu_reduction=cell.get("fpu_reduction"),
                c_puct=cell.get("c_puct"),
                oracle_belief=bool(cell.get("oracle_belief")),
                # Lockstep with the driver's builder. The shared docstring warns that the
                # two drifting apart is a SILENT failure -- the reference matches no shard
                # -- so the opponent fragment has to land in BOTH or fixing one creates
                # exactly that drift.
                opponent_policy_mode=cell.get("opponent_policy_mode") or "foul-play",
                opponent_engine_depth=cell.get("opponent_engine_depth"),
                opponent_engine_sims=cell.get("opponent_engine_sims"),
                # Kept in lockstep with the driver's builder deliberately. The
                # shared docstring warns that these two drifting apart is a
                # SILENT failure -- the reference simply matches no shard -- and
                # it has already happened once, on the checkpoint tag.
                early_stop=bool(cell.get("early_stop")),
                early_stop_min_sims=cell.get("early_stop_min_sims"),
                depth_min=cell.get("depth_min"),
                worlds_min=cell.get("worlds_min"),
                # Lockstep again, for the third time and the same reason. A
                # rollout cell whose reference id was built without this fragment
                # would match no shard, `depth_reference` would populate from
                # nothing, and the §5 non-starvation rule would report clean --
                # the precise failure the tag comment above records having already
                # happened. `search_config_id` requires R and the cap whenever the
                # arm is on, so a campaign cell that sets `rollout_leaf` without
                # them fails here loudly rather than rendering a pooled id.
                # Through `rollout_leaf_of`, never `bool(cell.get(...))`: see the
                # note above that helper for what an ABSENT flag renders.
                rollout_leaf=rollout_leaf_of(cell),
                rollout_count=cell.get("rollout_count"),
                rollout_max_plies=cell.get("rollout_max_plies"),
                rollout_policy=cell.get("rollout_policy") or "uniform",
            )

        for cell in campaign.get("cells", []):
            ref = cell.get("reads_against")
            if ref and ref in by_cell:
                own_id, reference_id = cid_of(cell), cid_of(by_cell[ref])
                # A CELL MAY NOT READ AGAINST ITSELF, and this is what a lost id
                # fragment actually produces: two DIFFERENT campaign cells rendering
                # one id. Every prior instance of that in this campaign was silent
                # -- an absent knob drops its fragment, the id lands on the cell's
                # own reference, `depth_reference` populates, `depth_rule_applied`
                # reports true, and the section 5 non-starvation rule compares a
                # cell to itself and finds no starvation by construction.
                #
                # Checked here rather than per knob because it needs no guess about
                # WHICH knob went missing: any fragment this builder can drop
                # (`rollout_leaf`, the oracle belief, early stop, the opponent) shows
                # up as this equality when the two cells differ only by that knob.
                if own_id == reference_id and cell.get("cell_id") != ref:
                    raise SystemExit(
                        f"campaign cells {cell.get('cell_id')!r} and {ref!r} both render "
                        f"the id {own_id!r}, so {cell.get('cell_id')!r} would read "
                        "against ITSELF. Refused: the two cells differ by a knob this "
                        "id builder dropped, and the comparison would report no "
                        "difference by construction."
                    )
                depth_reference[own_id] = reference_id

    raw_arms = {c: r for c, r in rows.items() if meta[c]["arm"] == "raw"}
    if not raw_arms:
        raise SystemExit("no raw arm among the shards; a paired delta is undefined")

    cells: dict[str, dict] = {}
    for cid, search in rows.items():
        if meta[cid]["arm"] == "raw":
            continue
        # Pair against the raw arm of the SAME checkpoint. Matching on the
        # checkpoint rather than on arity is what keeps cell G (k1) off cell
        # A's (k0) denominator.
        ckpt = meta[cid]["checkpoint"]
        candidates = [c for c in raw_arms if meta[c]["checkpoint"] == ckpt]
        if len(candidates) != 1:
            cells[cid] = {"error": f"expected exactly one raw arm for checkpoint "
                                   f"{ckpt!r}, found {len(candidates)}"}
            continue
        raw = raw_arms[candidates[0]]
        shared_n = len(set(search) & set(raw))
        scored = score_cell(search, raw, _indices(shared_n)) if shared_n else {
            "pairs": 0, "error": "no shared pairs"}
        scored["raw_arm"] = candidates[0]
        scored["latency"] = latency_of(meta[cid])
        scored["health"] = health_of(meta[cid])
        scored["opponent_priors"] = meta[cid]["opponent_priors"]
        # Arm identity witnessed from the shards, not from the cell id alone --
        # and for the telemetry the cell id says nothing at all.
        scored["oracle_belief"] = meta[cid]["oracle_belief"]
        # Same standing: the arbiter arm is read as rollout-leaf against the same
        # config off, so a reader of the merged report must be able to tell which
        # side of that contrast a cell is on without re-parsing its id.
        scored["rollout_leaf"] = meta[cid]["rollout_leaf"]
        scored["override_telemetry_shards"] = dict(
            meta[cid]["override_telemetry_shards"]
        )
        if scored.get("pairs", 0) < args.min_pairs:
            scored["min_pairs_shortfall"] = (
                f"{scored.get('pairs', 0)} pairs < the required minimum of {args.min_pairs}"
            )

        gate = scored["latency"]["search_wall_per_searched_decision_mean"]
        if gate is None:
            scored["cap"] = "UNEVALUABLE - no search_wall_per_searched_decision in any shard"
        elif gate > args.cap_seconds:
            scored["cap"] = f"REJECTED - mean {gate:.2f}s exceeds {args.cap_seconds:.0f}s"
        else:
            scored["cap"] = f"PASS - mean {gate:.2f}s"
        flag = seat_gap_flag(scored)
        if flag:
            scored["seat_health"] = flag
        cells[cid] = scored

    # --- section 9 Phase 2 eligibility -------------------------------------
    # (i) cap, (ii) seat AND fallback health, (iii) depth evidence where the
    # campaign says a cell is a depth cell.
    for cid, cell in cells.items():
        reasons = []
        if not cell.get("pairs"):
            reasons.append("no shared pairs")
        if not str(cell.get("cap", "")).startswith("PASS"):
            reasons.append(cell.get("cap", "cap unknown"))
        if "seat_health" in cell:
            reasons.append("seat gap")
        fb = (cell.get("health") or {}).get("fallback_rate")
        if fb is not None and fb > FALLBACK_LIMIT:
            reasons.append(f"fallback {fb:.1%} over {FALLBACK_LIMIT:.0%}")
        # The OPPONENT seat is held to the same bar. In a head-to-head the opponent is half
        # the experiment, so a cell whose opponent was falling back is exactly as
        # contaminated as one whose pokezero seat was -- and reads as a tie rather than as
        # a fault, which is worse.
        ofb = (cell.get("health") or {}).get("opponent_fallback_rate")
        if ofb is not None and ofb > FALLBACK_LIMIT:
            reasons.append(
                f"OPPONENT fallback {ofb:.1%} over {FALLBACK_LIMIT:.0%}")
        if cell.get("min_pairs_shortfall"):
            reasons.append(cell["min_pairs_shortfall"])
        # Depth cells: a d6/d8 cell that did not out-reach its reference is
        # BUDGET-STARVED, and its flat strength is void rather than a null.
        # This is the confound section 5 exists to prevent, and it is why the
        # reference is cell H and not the anchor.
        ref = depth_reference.get(cid)
        if ref:
            mine = (cell.get("health") or {}).get("depth_reached_mean")
            theirs = (cells.get(ref, {}).get("health") or {}).get("depth_reached_mean")
            if mine is None:
                reasons.append("UNSCOREABLE - depth cell with no depth_reached evidence")
            elif theirs is None:
                reasons.append(f"UNSCOREABLE - reference {ref} has no depth_reached evidence")
            elif mine <= theirs:
                reasons.append(
                    f"BUDGET-STARVED - depth_reached_mean {mine:.2f} <= {ref}'s {theirs:.2f}; "
                    "strength is void, not a null"
                )
        cell["ineligible_because"] = reasons

    eligible = {c: v for c, v in cells.items() if not v["ineligible_because"]}
    ranked = sorted(eligible, key=lambda c: eligible[c]["paired_delta"]["point"], reverse=True)

    winner = None
    adoption = None
    if args.anchor is not None:
        if args.anchor not in cells:
            raise SystemExit(
                f"--anchor {args.anchor!r} is not among the shards' config_ids "
                f"({sorted(cells)}). Refusing to silently fall back to "
                "largest-delta-wins: cell ids are checkpoint-qualified and a typo "
                "would disable the adoption rule without a diagnostic."
            )
        if not cells[args.anchor].get("pairs"):
            raise SystemExit(
                f"anchor {args.anchor!r} has no scoreable pairs "
                f"({cells[args.anchor].get('error')}); the adoption rule is undefined."
            )
    if ranked and args.anchor:
        # Section 9 Phase 2 is FILTER-then-rank: (iii) is a per-cell condition,
        # not a test applied only to the leader. Testing just ranked[0] and
        # falling back to the anchor adopts the anchor whenever the largest
        # delta happens to be noisy, even though a slightly smaller cell
        # cleanly beats it -- measured on a fixture, that discarded a cell 5pp
        # better than the adopted one.
        anchor_rows = rows[args.anchor]
        anchor_raw = rows[cells[args.anchor]["raw_arm"]]
        beats_anchor = []
        for cid in ranked:
            if cid == args.anchor:
                beats_anchor.append(cid)
                continue
            result = paired_improvement(
                rows[cid], rows[cells[cid]["raw_arm"]], anchor_rows, anchor_raw
            )
            if result is None:
                cells[cid]["improvement_over_anchor"] = None
                continue
            imp, overlap = result
            payload = imp.to_payload()
            payload["pairs"] = overlap
            cells[cid]["improvement_over_anchor"] = payload
            if overlap < args.min_pairs:
                # The improvement is estimated over the OVERLAP with the
                # anchor, which can be far thinner than either cell's own n.
                payload["ineligible"] = (
                    f"overlap with the anchor is {overlap} pairs < {args.min_pairs}"
                )
                cells[cid]["ineligible_because"].append(payload["ineligible"])
                continue
            if imp.low > 0.0:
                beats_anchor.append(cid)
        # `ranked` is already sorted by delta, so the first survivor is the
        # largest delta among cells that pass every criterion.
        if beats_anchor and beats_anchor[0] != args.anchor:
            winner = beats_anchor[0]
            imp = cells[winner]["improvement_over_anchor"]
            adoption = (
                f"largest eligible delta whose improvement over {args.anchor} excludes 0: "
                f"{imp['point']:+.3f} [{imp['low']:+.3f}, {imp['high']:+.3f}]"
            )
        elif not cells[args.anchor]["ineligible_because"]:
            winner = args.anchor
            adoption = (
                f"no eligible cell's improvement over {args.anchor} excludes 0; "
                "adopting the anchor per section 9 Phase 2"
            )
        else:
            # The anchor is the fallback, not an exemption. Adopting a cell the
            # report itself rejected -- over the cap, seat-gapped, or short of
            # the minimum -- would publish it as the campaign's power config
            # with an adoption string that never mentions the rejection.
            winner = None
            adoption = (
                f"NO ADOPTION: no eligible cell beats the anchor, and the anchor "
                f"{args.anchor} is itself ineligible "
                f"({'; '.join(cells[args.anchor]['ineligible_because'])}). "
                "Section 9 Phase 2's fallback assumes a healthy anchor."
            )
    elif ranked:
        winner = ranked[0]
        adoption = "no --anchor given; reporting the largest eligible delta only"
    else:
        # No cell survived eligibility. Saying so explicitly matters: a null
        # winner with a null reason reads as "not computed yet" rather than as
        # "every cell was rejected".
        rejected = {c: v["ineligible_because"] for c, v in cells.items()}
        winner = None
        adoption = (
            "NO ADOPTION: no cell passed eligibility. "
            + "; ".join(f"{c}: {', '.join(r)}" for c, r in sorted(rejected.items()) if r)
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "engine_fingerprint": fingerprint,
        "cap_seconds": args.cap_seconds,
        "anchor": args.anchor,
        "cells": cells,
        "ranking_eligible": ranked,
        "winner": winner,
        "adoption_rule": adoption,
        "depth_rule_applied": sorted(
            c for c in depth_reference if c in cells
        ),
        "depth_rule_unmatched": sorted(
            c for c in depth_reference if c not in cells
        ),
        "min_pairs": args.min_pairs,
        "winner_note": (
            "Eligibility requires shared pairs, >= the section 8 minimum, a passing "
            "cap, seat and fallback health, and -- for depth cells, when --campaign "
            "is given -- reached-depth clearing their reference. A cell excluded by "
            "the cap or as budget-starved is NOT a miss for its strength prediction; "
            "it is unscored. Adoption compares the paired IMPROVEMENT over the "
            "anchor, not two independent deltas."
        ),
    }
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
