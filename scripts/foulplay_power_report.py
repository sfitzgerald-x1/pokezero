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
  report records whether the rule was applied;
* a cell whose OPPONENT was not equally strong against both arms is
  CONTENTION-CONFOUNDED and ineligible. The reference opponent is time-budgeted
  and thinks concurrently with our search on the same host, so whichever arm
  spends more CPU per decision faces a weaker opponent and is flattered by it.
  `contention_of` compares the opponent's realized work per granted
  budget-second BETWEEN the arms, stratified by foul-play's own per-decision
  schedule, and refuses -- including when it was never measured, because an
  unmeasured confound reads as a clean delta rather than as a fault.

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
sys.path.insert(0, str(REPO_ROOT / "src"))

# The reader-side schema refusal, on the POOLING reader. See
# `require_rollout_leaf_document_schema`: this is the second of the two read-path
# call sites, and it is the one that decides what gets averaged.
from pokezero.engine_search import (  # noqa: E402
    require_rollout_leaf_document_schema,
)

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
    shards = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        schema = payload.get("schema_version")
        if schema != "pokezero.foulplay-paired-shard.v1":
            raise SystemExit(f"{path}: unexpected schema_version {schema!r}")
        # THE DOCUMENT'S schema_version SAYS NOTHING ABOUT ITS ROLLOUT BLOCK. The
        # check above pins the paired-shard envelope, which was already at v1 before
        # the rollout columns existed and does not move when they are re-schemaed --
        # so this pooling reader would have averaged a v1 rollout block, or a block
        # whose world counter it reads as zero, without the envelope changing a
        # character. Refused here, on the read, in the process that does the pooling.
        require_rollout_leaf_document_schema(payload)
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


def collect_rows(shards: list[dict]) -> tuple[dict, dict]:
    """(config_id -> {(seed, seat): score}), and config_id -> shard metadata."""
    rows: dict[str, dict[tuple[int, str], float]] = defaultdict(dict)
    meta: dict[str, dict] = {}
    for shard in shards:
        cid = shard["config_id"]
        meta.setdefault(cid, {"arm": shard["arm"], "shards": [], "per_seat": [],
                              "checkpoint": shard.get("checkpoint"),
                              "opponent_priors": shard.get("opponent_priors", False),
                              # config_id CARRIES this one, so every shard of a
                              # cell must agree; disagreement is asserted below.
                              "oracle_belief": bool(shard.get("oracle_belief", False)),
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


def think_headers_of(meta_entry: dict) -> list[dict | None]:
    """Every seat's opponent-think header for one cell, absences INCLUDED as None.

    One per (shard, seat), and a seat whose block is missing contributes a None rather than
    being skipped: `pool_foulplay_think` counts those and refuses, because coverage computed
    over the shards that DID carry a block reads clean while a quarter of the arm was never
    measured. Skipping them here would launder that into a pass.
    """
    headers: list[dict | None] = []
    for per_seat in meta_entry["per_seat"]:
        for seat in (per_seat or {}).values():
            headers.append(seat.get("foulplay_think"))
    return headers


def contention_of(search_meta: dict, raw_meta: dict) -> dict:
    """THE CROSS-ARM CONTENTION GATE, run where the two arms actually meet.

    The reference opponent is TIME-budgeted and its search overlaps ours on the same host, so
    CPU taken from it shows up as a weaker opponent and reads as a strength gain for whichever
    arm spends more CPU per decision -- a confound in the flattering direction, on exactly the
    arm whose job is to arbitrate a disputed effect. The instrument that measures it
    (`foulplay_think`, see OPPONENT-THINK CONTENTION INSTRUMENT in `pokezero/foulplay_bridge.py`)
    lands per shard, and the merged shard runs it p1-against-p2 WITHIN one arm. Within-arm
    symmetry cannot see a between-arm difference: both seats of the hungry arm are equally
    starved. This report is the first place the search arm's shards and its raw arm's shards
    are both in hand, so this is where the between-arm comparison belongs.

    Imported rather than reimplemented, per this file's own rule about `search_config_id`: a
    second copy of the admissibility rules is how two files end up disagreeing about whether a
    contention number may be used.
    """
    from pokezero.foulplay_bridge import (  # noqa: PLC0415
        cross_arm_foulplay_contention,
        pool_foulplay_think,
    )

    return cross_arm_foulplay_contention(
        pool_foulplay_think(think_headers_of(search_meta), label="search"),
        pool_foulplay_think(think_headers_of(raw_meta), label="raw"),
        hungry_label="search",
        lean_label="raw",
    )


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

    # Named at module scope in the bridge so the report and the gate cannot drift apart on
    # the number the whole refusal turns on.
    from pokezero.foulplay_bridge import (  # noqa: PLC0415
        FOULPLAY_THINK_CROSS_ARM_RESOLVING_STRATUM_DECISIONS,
        FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO,
        FOULPLAY_THINK_MEASURED_DECISION_LOG_SD,
        FOULPLAY_THINK_MEASURED_RUN_LOG_SD,
    )

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
            )

        for cell in campaign.get("cells", []):
            ref = cell.get("reads_against")
            if ref and ref in by_cell:
                depth_reference[cid_of(cell)] = cid_of(by_cell[ref])

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
        # BETWEEN the arms, not within one. See contention_of.
        scored["contention"] = contention_of(meta[cid], meta[candidates[0]])
        scored["opponent_priors"] = meta[cid]["opponent_priors"]
        # Arm identity witnessed from the shards, not from the cell id alone --
        # and for the telemetry the cell id says nothing at all.
        scored["oracle_belief"] = meta[cid]["oracle_belief"]
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
        # THE CROSS-ARM CONTENTION GATE, held to the same bar as the cap: a comparison the
        # gate cannot defend is UNSCORED, not a null. Anything but `ok` makes the cell
        # ineligible, including "not measured" -- a cell whose opponent-throughput parity is
        # unknown is exactly as unbankable as one whose opponent was starved, and it reads as
        # a clean delta rather than as a fault, which is worse. Same precedent as
        # `cap: UNEVALUABLE`.
        contention = cell.get("contention") or {}
        if contention.get("status") != "ok":
            reasons.append(
                "CONTENTION-CONFOUNDED - cross-arm opponent throughput not comparable: "
                + ", ".join(contention.get("refusal_reasons") or ["contention unknown"])
            )
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
    # A CONTENTION-REFUSED ANCHOR CANNOT BE A COMPARATOR, which the ineligibility list alone
    # does not achieve. The adopted quantity is the paired IMPROVEMENT
    # (candidate_delta - anchor_delta), so a confounded anchor puts its confounded delta inside
    # the number that gets adopted -- and the adoption string never mentions it. Making the
    # anchor ineligible only removes it from `ranked`; it stays the comparator. Found by
    # independent review, which demonstrated a 3.8x-starved anchor producing
    # "largest eligible delta whose improvement over anchor@k0 excludes 0: +0.333".
    anchor_contention_refused = bool(
        args.anchor is not None
        and (cells.get(args.anchor, {}).get("contention") or {}).get("status") != "ok"
    )
    if anchor_contention_refused:
        reasons = (cells[args.anchor].get("contention") or {}).get("refusal_reasons") or [
            "contention unknown"
        ]
        winner = None
        adoption = (
            f"NO ADOPTION: the anchor {args.anchor} is CONTENTION-CONFOUNDED "
            f"({', '.join(reasons)}), and the adoption rule subtracts the anchor's delta from "
            "every candidate's -- so no improvement over it can be trusted. Section 9 Phase 2's "
            "comparator has to be a clean cell."
        )
    if ranked and args.anchor and not anchor_contention_refused:
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
    elif ranked and not anchor_contention_refused:
        winner = ranked[0]
        adoption = "no --anchor given; reporting the largest eligible delta only"
    elif not anchor_contention_refused:
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
        # THE PREREGISTRATION, in the artifact. Deliberately not a CLI flag: a threshold
        # that can be loosened at report time is not preregistered, and the direction it
        # would be loosened in is known in advance.
        "contention_gate": {
            "max_fold_ratio": FOULPLAY_THINK_MAX_CROSS_ARM_FOLD_RATIO,
            "measured_decision_log_sd": FOULPLAY_THINK_MEASURED_DECISION_LOG_SD,
            "measured_run_log_sd": FOULPLAY_THINK_MEASURED_RUN_LOG_SD,
            "note": (
                "Refuses when the opponent's realized visits per granted budget-second "
                "differ between the arms by more than the fold ratio in any compared "
                "stratum. Threshold derived from the instrument's own measured precision: "
                "six uncontended passes of real foul-play searches gave 15 matched-arm fold "
                "ratios spanning 1.0013-1.1170, so 1.25 leaves ~2x log-margin on the largest "
                "matched-arm fold observed; and the variance decomposition says that spread is "
                "nearly flat in n (a nominal z=3 bound is 1.272 at the per-stratum floor and "
                "1.246 at n=200, because the whole-run term dominates the per-decision one), so "
                "a FIXED bound is the right shape. Six passes justify a fixed constant of about "
                "this size. They do not justify more digits or a false-refusal rate: the run "
                "term carries 5 degrees of freedom, so its point-estimate floor of 1.2447 (the "
                "n->inf asymptote; 1.2506 at the calibration's own n=24, and 1.25 is 0.42% above "
                "the first and 0.05% below the second -- quote the n) has a ONE-SIDED 95% "
                "chi-square upper bound of 1.580 (1.711 at the two-sided interval's upper end) "
                "and a t-based floor of 1.47-1.49 at the two-sided tail of z=3, "
                "p=0.0027 (t=5.507 on 5 df; p=0.002 would give 1.537 instead, so quote the p). "
                "Every one of those floors is computed from the run COMPONENT 0.0516, not from "
                "the raw pass-mean SD 0.052745 = sqrt(0.0516^2 + 0.0529^2/24), on which the same "
                "quantities read 1.5080, 1.5521 (p=0.002), 1.6692 (p=0.001) and 1.7312. "
                "The false-refusal probability is bounded BY THE DATA only at <18% (0 of 15 "
                "matched pairs). A pass BOUNDS the confound; it does not show it is zero, and "
                "between 1.117 and 1.25 the gate is deliberately silent."
            ),
            "scope": (
                "The 1.1170 anchor was measured with position variance CANCELLED: the six "
                "calibration passes replayed identical positions. Real paired arms share a "
                "battle seed but diverge after their first differing choice, and position SD "
                "(0.1003) is the largest of the three terms, so 1.1170 UNDERSTATES real "
                "matched-arm variation. Measured magnitude: SD(log fold) rises 7.3% at n=24 and "
                "0.9% at n=200, because the run term dominates -- which makes the anchor ~1.126 "
                "rather than 1.117, and leaves 1.25 with 1.88x log-margin instead of 2.02x. A "
                "scope qualifier, not a threshold change. Also unmeasured: the bound was taken "
                "at 2x1000ms on an 18-core macOS box; the early-game 8x500ms stratum borrows it."
            ),
            "resolving_stratum_decisions": (
                FOULPLAY_THINK_CROSS_ARM_RESOLVING_STRATUM_DECISIONS
            ),
            "resolution_rule": (
                "A stratum holding fewer than "
                f"{FOULPLAY_THINK_CROSS_ARM_RESOLVING_STRATUM_DECISIONS} measured decisions on "
                "either arm cannot resolve the threshold (its own nominal z=3 resolution is "
                "wider than the fold ratio), so it is EXCLUDED from the compared set rather "
                "than refusing the comparison: refusing voided perfectly matched arms over a "
                "rounding-error slice of the run, at 15 of 15 matched calibration pairs on a "
                "1.0% minor stratum. Excluded strata count as UNCOVERED, are named in the "
                "verdict's strata_excluded_for_resolution with their denominators and their "
                "resolution and no rate, sized in cross_arm_share_excluded_for_resolution, and "
                "the min_cross_arm_compared_share floor decides. A coverage refusal names WHICH "
                "of the two shortfalls it was -- "
                "cross_arm_strata_excluded_for_resolution_cover_too_little or "
                "cross_arm_compared_strata_cover_too_little -- because they have different "
                "remedies and neither is contention. The floor is 27 and not the calibration's "
                "24: 27 is what the shipped decomposition computes (1.250197 at n=26, 1.249996 "
                "at n=27) and a test recomputes it from the two SDs, whereas 24 is a crossover "
                "only for a run SD in [0.051426, 0.051475]. min_stratum_decisions (5) and "
                "min_measured_decisions (20) are both below 27 and therefore inert at this "
                "layer. Under position variance NOT cancelling the same arithmetic gives 124, "
                "which is a share question and not a refusal."
            ),
            "pass_bounds": (
                "A passing cell's mix-standardized arm-level opponent-throughput SHORTFALL is "
                "at most 24.0% (a composed fold of 1.3158 = max_fold_ratio / "
                "min_cross_arm_compared_share). Stated as a shortfall because readers subtract: "
                "the fold would be misread as 31.6% and the per-stratum threshold as 25%."
            ),
        },
        # THE SCOPE LIMIT STAYS PROSE, HERE AND IN `contention_gate`. A keyed
        # `tracked_follow_ups` entry was tried and withdrawn: an artifact field asserting a
        # project's own bookkeeping tests the bookkeeping rather than the behaviour, and it rots
        # the moment the item is closed somewhere else. What has to survive is the claim a reader
        # of `winner` sees, so that is what is written and what the note tests pin.
        "winner_note": (
            "Eligibility requires shared pairs, >= the section 8 minimum, a passing "
            "cap, seat and fallback health, a cross-arm opponent-throughput comparison "
            "the contention gate accepts (which bounds the opponent-strength confound in "
            "THROUGHPUT units, not in win-rate units -- a passing cell's mix-standardized "
            "arm-level opponent-throughput shortfall is at most 24.0%, but nothing here "
            "calibrates opponent visits to opponent strength, so the residual in pp is "
            "unknown, not small. `contention: ok` is therefore NOT clearance of the strength "
            "comparison; the measurement that would convert one to the other is raw against "
            "raw with the opponent's per-battle budget cut 24%, and it has not been run), "
            "and -- for depth "
            "cells, when --campaign is given -- reached-depth clearing their reference. "
            "A cell excluded by "
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
