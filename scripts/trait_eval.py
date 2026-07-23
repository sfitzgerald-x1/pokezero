"""Trait-eval game generation with full event capture (checkpoint trait-tracking plan).

Drives games with a checkpoint and captures, per game, the omniscient Showdown protocol log
plus per-decision active-mon PP snapshots and starting movesets — the sufficiency contract for
every trait metric. `LocalShowdownEnv` already exposes `protocol_lines` (the omniscient log) and
per-player requests (which carry active-mon PP, so Pressure's double-decrement is handled by the
simulator, not inferred). Works with any checkpoint; the 500k restriction is scope, not machinery.

Opponents:
  --opponent self      one policy drives both seats (self-play; both seats behavioral)
  --opponent foulplay  the checkpoint's seat vs the FoulPlay search bot (bot seat behavioral;
                       opponent seat contributes only the opponent-PP metric)

Output: events-<shard>.jsonl.gz — one JSON object per game.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import json
import sys
import time
from pathlib import Path

from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
from pokezero.online_client import build_agent, build_agent_remote
from pokezero.showdown import observation_from_player_state

MAX_STEPS = 1000
# gen3 randbats games resolve well under this; a game still going past it is a stall (a weak
# checkpoint that can't close). We end such games as timeouts (capped) rather than letting them
# run to Showdown's turn-1000 tie — which both skews avg-turns and balloons memory (~13k protocol
# lines/game). Timeouts are kept for behavioral metrics but excluded from avg_turns downstream.
TURN_CAP = 200
SEATS = ("p1", "p2")
# recreate the warm env every N games as a backstop: reset() reuse fixes the per-game subprocess
# leak; the env also grows a never-cleared stderr list (cleared per game below), and there is a
# slower persistent climb, so we recycle occasionally AND run with generous memory.
ENV_RECYCLE = 200


def _request_pp(request):
    """Active mon's move (id, pp, maxpp) list from a player request, or None."""
    if not request:
        return None
    active = request.get("active") or []
    if not active or not isinstance(active[0], dict):
        return None
    return [{"id": m.get("id") or m.get("move"), "pp": m.get("pp"), "maxpp": m.get("maxpp")}
            for m in (active[0].get("moves") or []) if isinstance(m, dict)]


def _team_movesets(request):
    """Starting species + movesets for the side that owns this request (own-team is fully revealed)."""
    side = (request or {}).get("side") or {}
    out = []
    for mon in side.get("pokemon") or []:
        details = mon.get("details") or ""
        species = details.split(",")[0].strip()
        # ability lets trait_extract gate ability-dependent traits (Intimidate, Volt/Water Absorb,
        # Flash Fire) exactly on team composition; older captures without it fall back to inference.
        ability = mon.get("baseAbility") or mon.get("ability") or ""
        out.append({"species": species, "moves": list(mon.get("moves") or []), "ability": ability})
    return out


def _rss_mb():
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        return -1


def _active_species(request):
    """Species of the side's currently-active mon (the one whose PP this request reveals)."""
    side = (request or {}).get("side") or {}
    for mon in side.get("pokemon") or []:
        if mon.get("active"):
            return (mon.get("details") or "").split(",")[0].strip() or None
    return None


def play_self_play_game(agent, env, seed):
    # The env's bridge process is a warm pool reused across battles; reset() ends the previous
    # battle, drains it, and clears protocol_lines. One env per shard (not per game) — spawning a
    # fresh env per game leaks the node subprocess and OOMs after a few hundred games.
    env.reset(seed=seed)
    pp_track = []
    movesets = {}
    timed_out = False
    for _ in range(MAX_STEPS):
        if env.terminal() is not None:
            break
        actions = {}
        cur_turn = 0
        for player in env.requested_players():
            state = env._state_for_player(player)
            if state.request is None or state.request_kind in {"wait", "none", "team_preview"}:
                continue
            if not any(state.legal_action_mask):
                continue
            cur_turn = max(cur_turn, state.turn_number or 0)
            if player not in movesets:
                movesets[player] = _team_movesets(state.request)
            obs = observation_from_player_state(
                state, category_vocab=agent.vocab, spec=agent.spec, dex=agent.dex,
                **({"feature_masks": agent.feature_masks} if agent.feature_masks is not None else {}),
            )
            idx = agent.policy.select_action(obs, rng=agent.rng).action_index
            actions[player] = idx
            pp = _request_pp(state.request)
            if pp is not None:
                pp_track.append({"turn": state.turn_number, "seat": player,
                                 "mon": _active_species(state.request), "moves": pp})
        if cur_turn > TURN_CAP:
            timed_out = True
            break
        if not actions:
            break
        env.step(actions)
    term = env.terminal()
    natural = term is not None and not timed_out
    return {
        "seed": seed,
        "opponent": "self",
        "winner": (term.winner if natural else None),
        "turn_count": (term.turn_count if natural else (term.turn_count if term else TURN_CAP)),
        "capped": bool(timed_out or (term.capped if term else False)),
        "protocol": list(env.protocol_lines),
        "movesets": movesets,
        "pp_track": pp_track,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--lineage", default=None, help="lineage key (recorded in manifest for the report)")
    ap.add_argument("--milestone", type=int, default=None, help="cumulative-games milestone this checkpoint sits at")
    ap.add_argument("--opponent", choices=["self", "foulplay"], default="self")
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--seed-start", type=int, default=61_000_000)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--search-ms", type=int, default=100)
    ap.add_argument("--showdown-root", required=True)
    ap.add_argument("--belief-set-source", action="store_true")
    ap.add_argument("--infsvc-url", default=None,
                    help="WS-L1 inference server base URL. When set, the policy forward is served "
                         "remotely (one shared GPU per checkpoint) and this shard runs CPU-only; "
                         "--checkpoint is then provenance-only (recorded in the manifest, not loaded).")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.opponent == "foulplay":
        print("foulplay opponent not yet wired in trait_eval; use --opponent self", file=sys.stderr)
        return 2

    if args.infsvc_url:
        agent = build_agent_remote(args.infsvc_url, args.showdown_root, our_name="trait", deterministic=True)
    else:
        agent = build_agent(str(args.checkpoint), args.showdown_root, our_name="trait", deterministic=True)
    env_kwargs = {"showdown_root": args.showdown_root, "set_belief_source": args.belief_set_source or None,
                  "observation_spec": agent.spec, "category_vocab": agent.vocab}
    if agent.feature_masks is not None:
        env_kwargs["feature_masks"] = agent.feature_masks

    # deterministic shard split of the seed block
    my_seeds = [args.seed_start + i for i in range(args.games) if i % args.num_shards == args.shard]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = 0
    # one env for the whole shard — its bridge process is a warm pool reused across resets.
    env = LocalShowdownEnv(LocalShowdownConfig(**env_kwargs))
    try:
        with gzip.open(args.out, "wt") as f:
            f.write(json.dumps({"record": "manifest", "checkpoint": str(args.checkpoint),
                                "lineage": args.lineage, "milestone": args.milestone,
                                "opponent": args.opponent, "shard": args.shard, "num_shards": args.num_shards,
                                "seed_start": args.seed_start, "games_requested": args.games,
                                "capture_version": "trait_eval.v1"}) + "\n")
            for i, seed in enumerate(my_seeds):
                if i > 0 and i % ENV_RECYCLE == 0:
                    env.close()
                    gc.collect()
                    env = LocalShowdownEnv(LocalShowdownConfig(**env_kwargs))
                rec = play_self_play_game(agent, env, seed)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                # the env's node-stderr reader appends to this list forever; drop it per game.
                try:
                    env._stderr_lines.clear()
                except Exception:
                    pass
                n += 1
                if n % 50 == 0:
                    print(f"  shard {args.shard}: {n}/{len(my_seeds)} ({time.time()-t0:.0f}s) rss={_rss_mb()}MB", flush=True)
    finally:
        env.close()
    print(f"WROTE {args.out} games={n} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
