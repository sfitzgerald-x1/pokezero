"""Phase 0 of the checkpoint-trait-tracking plan: inventory + feasibility (no GPU).

Resolves the five lineages against shared experiment storage, merges disjoint legs onto a
single cumulative-games axis, builds the 100k milestone grid mapping each milestone to
(leg, iteration, sha256), pins the 500k checkpoints, and emits a Phase-1 data-source verdict
per lineage. Writes traits/inventory.json and checks gate G0.

Naming-era note: the v22-lr3m lineage's 500k leg is `foundation-emetamon-v2-2-lr3m-500k-belief`
(older naming), which the plan's `emeta-v2-2-lr3m-.*` pattern does not match — it is folded in
explicitly here so the lineage has its 500k checkpoint (a Phase-2 target and G0 requirement).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re

EXP = "/shared/scott-experiment"
OUT = "/shared/traits/inventory.json"
MILESTONE_STEP = 100_000

# key -> (regex over run-id, explicit extra run-ids to fold in for naming-era gaps)
LINEAGES = {
    "m50-ep7":       (r"^metamon-m50-.*-lr10m-ep7$", []),
    "l200-ep7-wu75": (r"^metamon-l200-.*-lr10m-ep7-wu75$", []),
    "v22-lr3m":      (r"^emeta-v2-2-lr3m-.*$", ["foundation-emetamon-v2-2-lr3m-500k-belief"]),
    "m50-seq":       (r"^metamon-m-50m-.*-seq-20260710$", []),
    "l200-seq":      (r"^metamon-l-200m-.*-seq-20260710$", []),
}


def summary(run):
    try:
        return json.load(open(os.path.join(EXP, run, "distributed-foundation-summary.json")))
    except Exception:
        return None


def retained_iters(run):
    out = []
    for d in glob.glob(os.path.join(EXP, run, "run", "iteration-*")):
        ck = os.path.join(d, "transformer-policy.pt")
        if os.path.isfile(ck) and os.path.getsize(ck) > 0:
            out.append(int(os.path.basename(d).split("-")[1]))
    return sorted(out)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_legs(pattern, extras):
    rx = re.compile(pattern)
    runs = set(extras)
    for d in os.listdir(EXP):
        if rx.match(d) and os.path.isdir(os.path.join(EXP, d)):
            runs.add(d)
    legs = []
    for run in runs:
        s = summary(run)
        if not s:
            continue
        offset = int(s.get("completed_games_before_run") or 0)
        gpi = int(s.get("games_per_iteration") or 1600)
        its = retained_iters(run)
        if not its:
            continue
        legs.append({"run": run, "offset": offset, "games_per_iteration": gpi,
                     "retained_iterations": its, "max_iter": max(its),
                     "terminal_games": offset + max(its) * gpi})
    legs.sort(key=lambda l: l["offset"])
    return legs


def milestone_map(legs):
    """For each 100k milestone up to the frontier, pin (leg, iteration, substitution?)."""
    if not legs:
        return [], 0
    frontier = max(l["terminal_games"] for l in legs)
    grid = []
    G = MILESTONE_STEP
    while G <= frontier:
        # find the leg covering G (largest offset <= G with terminal >= G-ish)
        leg = None
        for l in legs:
            if l["offset"] < G <= l["terminal_games"] + l["games_per_iteration"]:
                leg = l
        if leg is None:
            leg = min(legs, key=lambda l: abs(l["offset"] - G))
        gpi, off = leg["games_per_iteration"], leg["offset"]
        want_iter = round((G - off) / gpi)
        want_iter = max(1, min(want_iter, leg["max_iter"]))
        # substitute nearest retained if pruned
        if want_iter in leg["retained_iterations"]:
            it, sub = want_iter, False
        else:
            it = min(leg["retained_iterations"], key=lambda x: abs(x - want_iter))
            sub = True
        grid.append({"milestone": G, "leg": leg["run"], "iteration": it,
                     "actual_games": off + it * gpi, "substituted": sub,
                     "checkpoint": os.path.join(EXP, leg["run"], "run", f"iteration-{it:04d}", "transformer-policy.pt")})
        G += MILESTONE_STEP
    return grid, frontier


def phase1_verdict(legs):
    """Can Phase-1 self-play basics be recovered from archived collect caches?
    Conservative default REGENERATE unless a milestone leg has an intact, decodable cache dir.
    (Cache tensors decode to action indices; recovering move-name/forced-vs-voluntary needs the
    action-space map + per-turn request state. Ambiguity -> REGENERATE, per plan.)"""
    for leg in legs:
        cache_dirs = glob.glob(os.path.join(EXP, leg["run"], "cache", "iteration-*"))
        for d in cache_dirs:
            if glob.glob(os.path.join(d, "shard-*")):
                # a populated cache exists; but decodability is unproven here -> REGENERATE (safe)
                return "REGENERATE", f"populated caches exist (e.g. {os.path.basename(d)}) but decode path unproven; regenerate for correctness"
    return "REGENERATE", "no populated collect caches found for milestone checkpoints"


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    inv = {"schema": "pokezero.trait_inventory.v1", "milestone_step": MILESTONE_STEP, "lineages": {}}
    g0_ok = True
    for key, (pattern, extras) in LINEAGES.items():
        legs = resolve_legs(pattern, extras)
        grid, frontier = milestone_map(legs)
        verdict, why = phase1_verdict(legs)
        # continuity check: each leg's offset == previous leg's terminal (±1 iter)
        continuity = []
        for i in range(1, len(legs)):
            gap = legs[i]["offset"] - legs[i - 1]["terminal_games"]
            continuity.append({"between": [legs[i - 1]["run"], legs[i]["run"]], "gap_games": gap,
                               "ok": abs(gap) <= legs[i]["games_per_iteration"]})
        # pin sha for the 500k milestone (G0 requirement)
        m500 = next((m for m in grid if m["milestone"] == 500_000), None)
        if m500 and os.path.isfile(m500["checkpoint"]):
            m500["sha256"] = sha256(m500["checkpoint"])
        else:
            g0_ok = False
        inv["lineages"][key] = {
            "pattern": pattern, "folded_in": extras,
            "legs": [{k: l[k] for k in ("run", "offset", "games_per_iteration", "max_iter", "terminal_games")} for l in legs],
            "leg_continuity": continuity,
            "frontier_games": frontier,
            "phase1_source": verdict, "phase1_source_reason": why,
            "n_milestones": len(grid), "milestones": grid,
            "has_500k": m500 is not None,
        }
        print(f"{key}: legs={[l['run'] for l in legs]} frontier={frontier:,} milestones={len(grid)} 500k={'sha ' + m500['sha256'][:12] if m500 and m500.get('sha256') else 'MISSING'} src={verdict}")

    inv["gate_g0"] = {"passed": g0_ok, "requirement": "every lineage has a 500k checkpoint pinned by sha"}
    json.dump(inv, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT} | G0 {'PASS' if g0_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
