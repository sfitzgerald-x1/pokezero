#!/usr/bin/env python3
"""Cluster the `damage_calc:*` residue by move and mechanic.

WHY THIS EXISTS. After five cycles the non-damage residue has been driven down
to attributed mechanisms, and what remains is dominated by rows where the engine
and Showdown simply disagree about how much a move hits for. Handing the next
lane 46 raw rows invites the failure this ledger keeps recording: someone reads
one row, narrates a cause, and the narration becomes a prescription. This tool
produces structure instead — the partition that decides whether a row is real,
then clusters of the real ones by the factors a damage formula could hinge on.

THE PARTITION THAT MATTERS. `triage_roll_components` records, per damaged slot,
whether Showdown's observed damage is a member of the engine's own legal roll
set (`observed_in_legal_set`). gen3 rolls 85%-100% in 16 steps, so two honest
implementations routinely disagree on a single roll; that is not a divergence.
A row is only real when the observed value is reachable by NO engine roll.

The `unexplained_ratio_*` labels are NOT that test. Their window is 0.92-1.09,
narrower than gen3's true roll spread (a 0.85 roll against a 1.00 roll is a
legitimate 0.85 ratio), so the label over-reports. Cluster on the legal-set
membership, and use the ratio only to describe a cluster once it is real.
"""

from __future__ import annotations

import argparse
import collections
import json
from typing import Any


def _attacker_of(slot: str, choices: dict[str, str]) -> str:
    """The move that damaged `slot` is the OTHER side's choice."""

    return choices.get("p2" if slot == "p1" else "p1", "?")


def _factors(ctx: dict[str, Any], attacker: str, defender: str) -> list[str]:
    """Pre-state facts a damage formula could hinge on, as flat tags."""

    out: list[str] = []
    atk = ctx.get(attacker) or {}
    dfn = ctx.get(defender) or {}
    if atk.get("boosts"):
        out.append("attacker_boosts=" + ",".join(f"{k}{v:+d}" for k, v in sorted(atk["boosts"].items())))
    if dfn.get("boosts"):
        out.append("defender_boosts=" + ",".join(f"{k}{v:+d}" for k, v in sorted(dfn["boosts"].items())))
    if dfn.get("reflect"):
        out.append("reflect")
    if dfn.get("light_screen"):
        out.append("light_screen")
    if str(atk.get("status", "NONE")).upper() == "BURN":
        out.append("attacker_burned")
    for label, mon in (("attacker", atk), ("defender", dfn)):
        item = str(mon.get("item", "NONE")).upper()
        if item not in ("NONE", ""):
            out.append(f"{label}_item={item.lower()}")
        ability = str(mon.get("ability", "NONE")).upper()
        if ability not in ("NONE", ""):
            out.append(f"{label}_ability={ability.lower()}")
    weather = str(ctx.get("weather", "NONE")).upper()
    if weather not in ("NONE", ""):
        out.append(f"weather={weather.lower()}")
    if dfn.get("types"):
        out.append("defender_types=" + "/".join(t.lower() for t in dfn["types"] if t != "TYPELESS"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--triage", required=True)
    ap.add_argument("--json")
    args = ap.parse_args()

    triage = json.load(open(args.triage))
    rows = [r for r in triage["results"] if r["bucket"].startswith("damage_calc")]

    real: list[dict[str, Any]] = []
    reachable: list[dict[str, Any]] = []
    for row in rows:
        ctx = row.get("context") or {}
        legal = row.get("observed_in_legal_set") or []
        for index, finding in enumerate(row.get("findings") or []):
            slot, rest = finding.split(":", 1)
            pair, _, label = rest.partition(":")
            obs, _, eng = pair.partition("vs")
            try:
                obs_v, eng_v = int(obs), int(eng)
            except ValueError:
                continue
            item = {
                "seed": row["seed"],
                "step": row["step"],
                "slot": slot,
                "move": _attacker_of(slot, row.get("choices") or {}),
                "observed": obs_v,
                "engine": eng_v,
                "ratio": round(obs_v / max(eng_v, 1), 3),
                "label": label,
                "reachable": bool(legal[index]) if index < len(legal) else None,
                "factors": _factors(ctx, "p2" if slot == "p1" else "p1", slot),
            }
            (real if not (legal[index] if index < len(legal) else False) else reachable).append(item)

    print(f"damage_calc rows: {len(rows)}   damaged-slot findings: {len(real) + len(reachable)}")
    print()
    print("PARTITION — is Showdown's damage reachable by ANY engine roll?")
    print(f"  {len(reachable):4d}  reachable  -> NOT a divergence; a roll disagreement the label over-reported")
    print(f"  {len(real):4d}  unreachable -> REAL: no engine roll produces this value")
    if not real:
        return 0

    print()
    print("REAL findings clustered by move:")
    by_move = collections.defaultdict(list)
    for f in real:
        by_move[f["move"]].append(f)
    for move, group in sorted(by_move.items(), key=lambda kv: -len(kv[1])):
        ratios = sorted(g["ratio"] for g in group)
        lo, hi = ratios[0], ratios[-1]
        span = "constant" if hi - lo < 0.02 else f"{lo}-{hi}"
        print(f"  {len(group):3d}  {move:<24s} ratio {span}")
        for g in group[:2]:
            print(f"         seed {g['seed']} step {g['step']} {g['slot']}: obs {g['observed']} vs eng {g['engine']}")

    print()
    print("WINDOW-SEMANTICS WARNING — read before treating any ratio band as a lead.")
    print("  `_classify_ratio`'s low edge (0.92) treats the ENGINE value as the MEAN")
    print("  roll. If the engine reports TOP-of-range the true band is [0.85, 1.00]-")
    print("  shaped, and every finding just under 0.92 is this filter's own artifact")
    print("  packed against its floor. A spike at 0.90-0.92 is NOT a cluster.")
    print("  Constants are deliberately NOT chosen here, pending the damage lane's")
    print("  source-cited answer on engine damage-value semantics (max vs mean).")
    print("  Prefer `reachable` (legal-roll-set membership) — it needs no constant.")
    print()
    print("REAL findings clustered by ratio (DESCRIPTIVE ONLY, see warning above):")
    by_ratio = collections.Counter(f["ratio"] for f in real)
    for ratio, n in by_ratio.most_common(12):
        moves = sorted({f["move"] for f in real if f["ratio"] == ratio})
        print(f"  {n:3d}  ratio {ratio:<6} moves: {', '.join(moves[:6])}")

    print()
    print("Factor frequency among REAL findings (vs the reachable control group):")
    real_f = collections.Counter(t for f in real for t in f["factors"])
    ctrl_f = collections.Counter(t for f in reachable for t in f["factors"])
    print(f"  {'factor':<40s} {'real':>6s} {'ctrl':>6s}")
    for factor, n in real_f.most_common(18):
        print(f"  {factor:<40s} {n:6d} {ctrl_f.get(factor, 0):6d}")

    if args.json:
        json.dump({"real": real, "reachable": reachable}, open(args.json, "w"), indent=1)
        print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
