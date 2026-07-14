"""Foul-play trait capture: checkpoint vs the FoulPlay search bot, emitted in the SAME omniscient
event schema as trait_eval self-play so trait_extract.py runs unchanged.

It reuses the controlled-foulplay bridge (which drives the real Showdown battle and an external
FoulPlay process) through its public `run_controlled_foulplay_benchmark(trajectory_callback=...)`.
The bridge stashes the omniscient protocol and per-decision request snapshots into
`trajectory.metadata` (see the additive hook in foulplay_bridge.py); this driver projects that
into one events record per game. The bot is always p1 (the plan characterizes the bot only; the
opponent seat contributes just the opponent-PP metric).
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import time
from pathlib import Path

from pokezero.foulplay_bridge import ControlledFoulPlayConfig, run_controlled_foulplay_benchmark


def _resolve_winner_seat(protocol):
    """Map |player|pX|USERNAME and |win|USERNAME in the omniscient log to the winning seat."""
    name_to_seat = {}
    win_name = None
    for line in protocol:
        if line.startswith("|player|"):
            p = line.split("|")  # |player|p1|USERNAME|avatar|rating
            if len(p) > 3 and p[3]:
                name_to_seat[p[3]] = p[2]
        elif line.startswith("|win|"):
            win_name = line.split("|", 2)[2].strip() if line.count("|") >= 2 else None
        elif line.startswith("|tie"):
            return None
    if not win_name:
        return None
    return name_to_seat.get(win_name)


def _parse_request(line):
    try:
        return json.loads(line.split("|request|", 1)[1])
    except Exception:
        return None


def _active_pp(req):
    active = (req or {}).get("active") or []
    if not active or not isinstance(active[0], dict):
        return None
    return [{"id": m.get("id") or m.get("move"), "pp": m.get("pp"), "maxpp": m.get("maxpp")}
            for m in (active[0].get("moves") or []) if isinstance(m, dict)]


def _active_species(req):
    for mon in ((req or {}).get("side") or {}).get("pokemon") or []:
        if mon.get("active"):
            return (mon.get("details") or "").split(",")[0].strip() or None
    return None


def _movesets(request_history):
    """First request per seat fully reveals that seat's own team (species + move ids)."""
    out = {}
    for seat, line in request_history:
        if seat in out:
            continue
        req = _parse_request(line)
        if req is None:
            continue
        team = [{"species": (m.get("details") or "").split(",")[0].strip(),
                 "moves": list(m.get("moves") or [])}
                for m in ((req.get("side") or {}).get("pokemon") or [])]
        if team:
            out[seat] = team
    return out


def _event_from_trajectory(traj):
    md = traj.metadata or {}
    protocol = list(md.get("omniscient_protocol") or [])
    rh = [(s, l) for s, l in (md.get("request_history") or [])]
    term = traj.terminal
    winner = _resolve_winner_seat(protocol)
    if winner is None and term is not None and getattr(term, "winner", None) in ("p1", "p2"):
        winner = term.winner
    pp_track = []
    for i, (seat, line) in enumerate(rh):
        req = _parse_request(line)
        pp = _active_pp(req)
        if pp is not None:
            pp_track.append({"turn": i, "seat": seat, "mon": _active_species(req), "moves": pp})
    return {
        "seed": traj.seed,
        "opponent": "foulplay",
        "winner": winner,
        "turn_count": (term.turn_count if term else None),
        "capped": (bool(term.capped) if term else False),
        "protocol": protocol,
        "movesets": _movesets(rh),
        "pp_track": pp_track,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--lineage", default=None)
    ap.add_argument("--milestone", type=int, default=None)
    ap.add_argument("--games", type=int, required=True, help="games for THIS shard")
    ap.add_argument("--seed-start", type=int, default=71_000_000,
                    help="first seed for this shard; give each shard a disjoint block")
    ap.add_argument("--search-time-ms", type=int, default=1000)
    ap.add_argument("--showdown-root", type=Path, required=True)
    ap.add_argument("--foulplay-root", type=Path, required=True)
    ap.add_argument("--foulplay-python", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--belief-set-source", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # opt into the bridge's omniscient metadata stash (gated so it never touches the p1-only path).
    os.environ["POKEZERO_TRAIT_CAPTURE"] = "1"

    config = ControlledFoulPlayConfig(
        checkpoint=args.checkpoint,
        showdown_root=args.showdown_root,
        foulplay_root=args.foulplay_root,
        foulplay_python=args.foulplay_python,
        games=args.games,
        seed_start=args.seed_start,
        search_time_ms=args.search_time_ms,
        device=args.device,
        belief_set_source=True if args.belief_set_source else None,
        pokezero_player="p1",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    state = {"n": 0, "seen": 0}
    with gzip.open(args.out, "wt") as f:
        f.write(json.dumps({"record": "manifest", "checkpoint": str(args.checkpoint),
                            "lineage": args.lineage, "milestone": args.milestone,
                            "opponent": "foulplay", "seed_start": args.seed_start,
                            "games_requested": args.games, "search_time_ms": args.search_time_ms,
                            "capture_version": "trait_eval.v1"}) + "\n")

        f.flush()

        def on_trajectory(traj):
            state["seen"] += 1
            if traj.terminal is None:
                return
            rec = _event_from_trajectory(traj)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            state["n"] += 1
            if state["n"] % 50 == 0:
                print(f"  foulplay: {state['n']}/{args.games} ({time.time()-t0:.0f}s)", flush=True)

        print(f"benchmark starting: games={args.games} search_ms={args.search_time_ms}", flush=True)
        result = run_controlled_foulplay_benchmark(config, trajectory_callback=on_trajectory)
        result = asyncio.run(result) if hasattr(result, "__await__") else result
        n_result = len(getattr(result, "games", []) or [])
        print(f"benchmark done: result_games={n_result} callbacks_seen={state['seen']} written={state['n']}",
              flush=True)
    print(f"WROTE {args.out} games={state['n']} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
