"""Gate G1: independent cross-check of trait_extract against a second counting path.

trait_extract.py walks a stateful protocol parser (GameParse). This verifier re-derives a
handful of metrics from the raw protocol with deliberately different, dead-simple logic and
asserts they agree with the metrics.json. Agreement between two independent implementations is
the G1 evidence that the extraction is faithful to the captured events (not that the games are
"right" — that is behavior, not extraction).

Checks (behavioral seats only, matching trait_extract's seat set):
  - move-use totals per move id            vs move_distribution
  - mean turn_count                        vs avg_turns
  - Toxic / Substitute / Spikes use counts vs move_categories[*].total_uses
  - immunity switch-in count               vs switch_behavior.immunity_switchin.total
  - mons-alive-on-win (p1)                 vs avg_bot_mons_alive_on_win
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
from collections import Counter


def load(events_globs, metrics_path):
    files = []
    for g in events_globs:
        files.extend(sorted(glob.glob(g)) or [g])
    manifest, games = None, []
    for path in files:
        for line in gzip.open(path, "rt"):
            rec = json.loads(line)
            if rec.get("record") == "manifest":
                manifest = manifest or rec
            else:
                games.append(rec)
    return manifest, games, json.load(open(metrics_path))


def norm(name):
    return "".join(c for c in name.lower() if c.isalnum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="+", required=True)
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args()
    manifest, games, m = load(args.events, args.metrics)

    behav = tuple(m["behavioral_seats"])  # ('p1','p2') self, ('p1',) foulplay
    # --- independent tallies straight off the raw protocol ---
    move_uses = Counter()
    tox = sub = spikes = 0
    turns = []
    immunity = 0
    alive_on_win = []

    for g in games:
        turns.append(g.get("turn_count") or 0)
        faints = {"p1": 0, "p2": 0}
        pending_switch_seat = None
        cur_turn_has_switch = set()
        for line in g["protocol"]:
            if not line.startswith("|"):
                continue
            p = line.split("|")
            tag = p[1]
            a = p[2:]
            seat = a[0][:2] if a and a[0][:2] in ("p1", "p2") else None
            if tag == "turn":
                cur_turn_has_switch = set()
            elif tag == "switch" and seat:
                cur_turn_has_switch.add(seat)
            elif tag == "move" and seat in behav and len(a) > 1:
                move_uses[a[1]] += 1
                nm = norm(a[1])
                if nm == "toxic":
                    tox += 1
                elif nm == "substitute":
                    sub += 1
                elif nm == "spikes":
                    spikes += 1
            elif tag == "-immune" and seat in behav and seat in cur_turn_has_switch:
                immunity += 1
            elif tag == "faint" and seat:
                faints[seat] += 1
        if g.get("winner") == "p1":
            alive_on_win.append(6 - faints["p1"])

    checks = []

    def chk(name, mine, theirs, tol=1e-9):
        ok = abs(mine - theirs) <= tol
        checks.append((name, mine, theirs, ok))

    md = m["move_distribution"]
    # every move id agrees exactly
    all_moves = set(md) | set(move_uses)
    move_ok = all(md.get(k, 0) == move_uses.get(k, 0) for k in all_moves)
    checks.append(("move_distribution (all ids)", "match" if move_ok else "MISMATCH",
                   f"{len(all_moves)} ids", move_ok))

    chk("avg_turns", round(sum(turns) / len(turns), 4) if turns else 0.0, m["avg_turns"], tol=1e-4)
    chk("cat_toxic.total_uses", tox, m["move_categories"]["cat_toxic"]["total_uses"])
    chk("cat_substitute.total_uses", sub, m["move_categories"]["cat_substitute"]["total_uses"])
    chk("cat_spikes.total_uses", spikes, m["move_categories"]["cat_spikes"]["total_uses"])
    chk("immunity_switchin.total", immunity,
        m["switch_behavior"].get("immunity_switchin", {}).get("total", 0))
    chk("avg_bot_mons_alive_on_win",
        round(sum(alive_on_win) / len(alive_on_win), 4) if alive_on_win else 0.0,
        m["avg_bot_mons_alive_on_win"], tol=1e-4)

    print(f"G1 cross-check: {len(games)} games, behavioral seats {behav}\n")
    allok = True
    for name, mine, theirs, ok in checks:
        allok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:32s} independent={mine!s:>10}  extract={theirs!s:>10}")
    print(f"\nG1 {'PASS — extraction agrees with independent path' if allok else 'FAIL'}")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
