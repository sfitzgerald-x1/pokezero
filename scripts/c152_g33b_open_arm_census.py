#!/usr/bin/env python
"""C152: measure G33b's two OPEN arms — the weather arm, and the exact speed tie.

`reports/c147_g33b_residual_bucket_gate.md` shipped `leftovers_slot_truncated`
(`rust/pokezero-search/src/events.rs`) and left two arms of the family open:

* **the shared weather entry at order 8 when the WINNER is faster.** Order 8
  precedes every order-10 handler on both sides, so a fatal weather chip always
  puts the winner's order-10.4 Leftovers slot behind the truncation point --
  but the shipped predicate ends in a single speed test that gates only when the
  loser is strictly faster, so the winner-faster half is not gated. C147 left it
  "unmeasured, not believed absent", and the function's own doc comment says the
  third column of its table and the fourth disagree on exactly this row.
* **an exact speed tie.** `residual_speed_order` returns `None` on a tie and
  `add_end_of_turn_branches` forks both orders, so the shipped predicate declines
  to guess.

This script turns the stderr of a THROWAWAY instrumented build into the census
those two arms need. The instrumented build is not committed and its fingerprint
is not reproducible from any committed tree -- same discipline as
`reports/artifacts/c147_g33b_gate_reach.json`, whose instrumentation this
extends. What IS committed is this parser, the instrumentation source quoted in
the artifact, and the resulting counts.

The instrumentation emits ONE `C152_TRUNC` line per battle-ending residual
instruction found by `leftovers_slot_truncated`, BEFORE any arm returns, so the
census sees the whole family and not only the part the shipped gate acts on.

Classifying the weather arm needs no instruction classification, which is the
thing the shipped predicate's doc comment says is impossible: a fatal residual
damage equals the victim's remaining HP and carries no phase information. The
census uses a STATE predicate instead. Order 8 is the FIRST residual phase, so
the loser's HP when its weather chip fires is still its pre-residual HP, and a
gen3 sandstorm/hail chip is `maxhp / 16`. So:

    loser dies at order 8  <=>  the loser takes a chip at all
                                (`weather_chips(state, loser)` is Some)
                           AND  `loser_pre_hp <= loser_maxhp // 16`

Both operands come from the pre-residual state. `amount == loser_maxhp // 16` is
recorded as an independent corroboration, not as the test.

Usage::

    python scripts/c152_g33b_open_arm_census.py \\
        --stderr dev=/tmp/trunc_dev.txt --stderr holdout=/tmp/trunc_holdout.txt \\
        --window dev=19000000:200 --window holdout=19100000:200 \\
        --json reports/artifacts/c152_g33b_open_arm_census.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

LINE = re.compile(r"^C152_TRUNC (?P<body>.*)$")

INSTRUMENTATION = [
    "// inside `leftovers_slot_truncated` (rust/pokezero-search/src/events.rs),",
    "// immediately after `if hp[loser] > 0 || has_reserve[loser] { continue; }`",
    "// and BEFORE the Liquid Ooze / Future Sight / Perish Song / speed arms return,",
    "// so the census sees every battle-ending residual instruction the predicate",
    "// reaches and not only the ones it acts on:",
    'eprintln!("C152_TRUNC arm={} order={} loser={} winner={} idx={} seglen={} '
    "amount={} loser_pre_hp={} loser_maxhp={} loser_weather={} winner_weather={} "
    'winner_item_lefto={} winner_heals_before={} loser_speed={} winner_speed={}", ...);',
    "// `arm` is which of the five enumerated arms this instruction falls in;",
    "// `order` is residual_speed_order(state) rendered as loser_first / winner_first / tie;",
    "// `loser_weather` / `winner_weather` are weather_chips(state, side);",
    "// `loser_pre_hp` is the active's HP BEFORE the residual segment;",
    "// `winner_heals_before` counts positive Heals on the winner's side earlier in the segment;",
    "// speeds are the crate's `effective_speed` replica of the engine's get_effective_speed.",
]


def parse(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LINE.match(raw.strip())
        if not match:
            continue
        row: dict[str, Any] = {}
        for token in match.group("body").split():
            if "=" not in token:
                continue
            key, _, value = token.partition("=")
            if value in ("true", "false"):
                row[key] = value == "true"
            else:
                try:
                    row[key] = int(value)
                except ValueError:
                    row[key] = value
        rows.append(row)
    return rows


def weather_fatal(row: dict[str, Any]) -> bool:
    """Did the loser die to its own order-8 weather chip?

    STATE predicate only -- see the module docstring. Order 8 runs before every
    order-10 handler, so the loser's HP at its chip is its pre-residual HP.
    """
    if row.get("loser_weather", "none") == "none":
        return False
    chip = max(1, int(row["loser_maxhp"]) // 16)
    return int(row["loser_pre_hp"]) <= chip


def census(rows: list[dict[str, Any]]) -> dict[str, Any]:
    arms = Counter(r["arm"] for r in rows)
    orders = Counter(r["order"] for r in rows)
    by_arm_order = Counter((r["arm"], r["order"]) for r in rows)

    gates_today = [r for r in rows if r["arm"] == "order_le_10" and r["order"] == "loser_first"]
    ties = [r for r in rows if r["order"] == "tie"]
    winner_first = [r for r in rows if r["order"] == "winner_first"]

    wf = [r for r in rows if weather_fatal(r)]
    wf_open = [r for r in wf if r["order"] != "loser_first"]
    wf_open_lefto = [r for r in wf_open if r.get("winner_item_lefto")]

    return {
        "instructions_seen": len(rows),
        "by_arm": dict(arms),
        "by_order": dict(orders),
        "by_arm_and_order": {f"{a}|{o}": n for (a, o), n in sorted(by_arm_order.items())},
        "gate_fires_today": len(gates_today),
        "gate_fires_today_with_leftovers_winner": sum(
            1 for r in gates_today if r.get("winner_item_lefto")
        ),
        "weather_arm": {
            "loser_dies_to_its_own_order_8_chip": len(wf),
            "of_those_not_gated_today": len(wf_open),
            "of_those_not_gated_today_and_winner_holds_leftovers": len(wf_open_lefto),
            # THE LOAD-BEARING ZERO for C152's retirement of this arm. An
            # over-booked heal slot can only mislabel a heal that actually got
            # emitted, and the only heal the winner can emit before the order-8
            # truncation is a resolving Wish -- which `residual_heal_cause`
            # labels correctly by its `DecrementWish` adjacency test, without the
            # plan. If this is ever nonzero the retirement needs re-arguing.
            "winner_side_heals_before_the_truncation": sum(
                int(r.get("winner_heals_before", 0)) for r in wf_open
            ),
            "rows": wf_open_lefto[:25],
            "any_loser_takes_a_chip_at_all": sum(
                1 for r in rows if r.get("loser_weather", "none") != "none"
            ),
            "amount_equals_chip_corroboration": sum(
                1
                for r in wf
                if int(r["amount"]) == max(1, int(r["loser_maxhp"]) // 16)
            ),
        },
        "tie_arm": {
            "exact_speed_ties": len(ties),
            "exact_speed_ties_with_leftovers_winner": sum(
                1 for r in ties if r.get("winner_item_lefto")
            ),
            "rows": ties[:25],
        },
        "winner_first_total": len(winner_first),
        "winner_first_with_leftovers_winner_and_no_earlier_heal": sum(
            1
            for r in winner_first
            if r.get("winner_item_lefto") and int(r.get("winner_heals_before", 0)) == 0
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stderr", action="append", required=True, metavar="NAME=PATH",
        help="a captured stderr file from the instrumented build, one per window",
    )
    parser.add_argument(
        "--window", action="append", default=[], metavar="NAME=SEEDSTART:GAMES",
        help="record the window each stderr capture came from",
    )
    parser.add_argument("--json", required=True)
    args = parser.parse_args(argv)

    windows = {}
    for spec in args.window:
        name, _, rest = spec.partition("=")
        seed_start, _, games = rest.partition(":")
        windows[name] = {"seed_start": int(seed_start), "games": int(games)}

    per_window: dict[str, Any] = {}
    combined: list[dict[str, Any]] = []
    for spec in args.stderr:
        name, _, path = spec.partition("=")
        rows = parse(Path(path))
        combined.extend(rows)
        per_window[name] = {**windows.get(name, {}), **census(rows)}

    out = {
        "schema": "c152-g33b-open-arm-census/1",
        "what": (
            "How often each of G33b's two OPEN arms -- the order-8 weather entry with "
            "the winner faster, and an exact speed tie -- is reached, measured with a "
            "throwaway instrumented build. Extends reports/artifacts/c147_g33b_gate_reach.json, "
            "which counted only the arm the shipped gate acts on."
        ),
        "instrumentation": INSTRUMENTATION,
        "weather_arm_test": (
            "STATE predicate, no instruction classification: weather_chips(state, loser) "
            "is Some AND loser_pre_hp <= loser_maxhp // 16. Order 8 is the first residual "
            "phase, so the loser's HP at its chip is still its pre-residual HP."
        ),
        "per_window": per_window,
        "all_windows_combined": census(combined),
    }
    Path(args.json).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out["all_windows_combined"], indent=2, sort_keys=True))
    print(f"-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
