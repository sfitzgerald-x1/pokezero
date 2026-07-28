#!/usr/bin/env python3
"""Aggregate `neural benchmark` summaries into a head-to-head strength matrix (self-contained HTML).

Consumes the summary JSONs written by `pokezero-neural benchmark --summary-out` (one per shard) and
sums their `head_to_heads` blocks. Every ordered pair is played in BOTH seat orders by benchmark, so
"A beats B" here is over the mirrored total: A's wins as first seat in "A vs B" plus A's wins as
second seat in "B vs A". Seat advantage therefore cancels rather than being averaged away.

Scripted baselines (random-legal / simple-legal) are kept out of the matrix and reported separately —
they are a sanity floor, not arms.

Usage: h2h_matrix_report.py --summaries 'dir/*.json' --out matrix.html [--title ...]
"""
from __future__ import annotations

import argparse
import collections
import glob
import html as html_mod
import json
import os

BASELINES = {"random-legal", "simple-legal"}


def esc(x):
    return html_mod.escape(str(x))


def load_pairs(paths):
    """-> {(a, b): [a_wins, b_wins, games, ties]} with a<b canonical, plus baseline rows."""
    pairs: dict = collections.defaultdict(lambda: [0, 0, 0, 0])
    base: dict = collections.defaultdict(lambda: [0, 0, 0])  # arm -> [wins, games, ties] vs baselines
    seeds_seen: set = set()
    for path in paths:
        try:
            data = json.load(open(path))
        except Exception:
            continue
        for h in data.get("head_to_heads", []):
            first, second = h["first_policy_id"], h["second_policy_id"]
            fw, sw = int(h["first_policy_wins"]), int(h["second_policy_wins"])
            games, ties = int(h["games"]), int(h.get("ties", 0))
            if first in BASELINES or second in BASELINES:
                arm = second if first in BASELINES else first
                wins = sw if first in BASELINES else fw
                row = base[arm]
                row[0] += wins
                row[1] += games
                row[2] += ties
                continue
            a, b = sorted((first, second))
            rec = pairs[(a, b)]
            # normalize into (a's wins, b's wins)
            if first == a:
                rec[0] += fw
                rec[1] += sw
            else:
                rec[0] += sw
                rec[1] += fw
            rec[2] += games
            rec[3] += ties
        for m in data.get("matchups", []):
            seeds_seen.add((m.get("label"), m.get("seed_start")))
    return pairs, base, len(seeds_seen)


def build(pairs, base, shard_count, title):
    arms = sorted({a for pair in pairs for a in pair})
    # overall record per arm across the matrix
    tally = {a: [0, 0] for a in arms}  # wins, games
    for (a, b), (aw, bw, games, _ties) in pairs.items():
        tally[a][0] += aw
        tally[a][1] += games
        tally[b][0] += bw
        tally[b][1] += games
    order = sorted(arms, key=lambda a: -(tally[a][0] / tally[a][1] if tally[a][1] else 0))

    def cell(row, col):
        if row == col:
            return '<td class="diag">—</td>'
        a, b = sorted((row, col))
        rec = pairs.get((a, b))
        if not rec:
            return '<td class="dim">·</td>'
        aw, bw, games, _ = rec
        wins = aw if row == a else bw
        rate = wins / games if games else 0.0
        # colour by advantage; the palette is symmetric around 50%
        cls = "win" if rate >= 0.55 else ("loss" if rate <= 0.45 else "even")
        return (f'<td class="{cls}">{rate * 100:.1f}%'
                f'<span class="sub2">{wins}/{games}</span></td>')

    head = "".join(f"<th>{esc(a)}</th>" for a in order)
    rows = []
    for r in order:
        wins, games = tally[r]
        overall = wins / games if games else 0.0
        rows.append(
            f"<tr><th>{esc(r)}</th>" + "".join(cell(r, c) for c in order)
            + f'<td class="tot">{overall * 100:.1f}%<span class="sub2">{wins}/{games}</span></td></tr>')

    base_rows = "".join(
        f"<tr><th>{esc(a)}</th><td>{(w / g * 100 if g else 0):.1f}%<span class=sub2>{w}/{g}</span></td></tr>"
        for a, (w, g, _t) in sorted(base.items(), key=lambda kv: -(kv[1][0] / kv[1][1] if kv[1][1] else 0)))

    total_games = sum(v[2] for v in pairs.values())
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>
:root{{--bg:#fff;--fg:#0f172a;--dim:#64748b;--line:#e2e8f0;--card:#f8fafc;--accent:#2563eb}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0b1220;--fg:#e5edf7;--dim:#93a4bb;--line:#1e2a3d;--card:#111a2b;--accent:#60a5fa}}}}
:root[data-theme=dark]{{--bg:#0b1220;--fg:#e5edf7;--dim:#93a4bb;--line:#1e2a3d;--card:#111a2b;--accent:#60a5fa}}
:root[data-theme=light]{{--bg:#fff;--fg:#0f172a;--dim:#64748b;--line:#e2e8f0;--card:#f8fafc;--accent:#2563eb}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif}}
.wrap{{max-width:1040px;margin:0 auto;padding:28px 20px 80px}}
h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:17px;margin:30px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}}
.sub{{color:var(--dim);margin:0 0 10px}}.sub2{{display:block;font-size:11px;color:var(--dim)}}
.tablewrap{{overflow-x:auto;border:1px solid var(--line);border-radius:10px;margin:10px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{padding:8px 10px;text-align:center;border-bottom:1px solid var(--line);white-space:nowrap}}
th:first-child{{text-align:left}}thead th{{background:var(--card);color:var(--accent)}}
td.win{{background:rgba(5,150,105,.14)}}td.loss{{background:rgba(220,38,38,.12)}}td.even{{background:rgba(100,116,139,.10)}}
td.diag{{color:var(--dim)}}td.tot{{font-weight:600}}.dim{{color:var(--dim)}}
.note{{color:var(--dim);border-left:3px solid var(--accent);padding:6px 10px;margin:10px 0;font-size:12.5px;background:var(--card);border-radius:0 6px 6px 0}}
</style></head><body><div class="wrap">
<h1>{esc(title)}</h1>
<p class="sub">{len(order)} arms · {len(pairs)} pairs · {total_games} games in the matrix · aggregated from {shard_count} shard matchups</p>
<div class="note">Each cell is the ROW arm's win rate against the COLUMN arm, over games played in
<b>both seat orders</b> (benchmark mirrors every matchup), so first-seat advantage cancels rather
than being averaged away. Cells shade green above 55% and red below 45%. The right-hand column is
the arm's record across the whole matrix.</div>
<h2>Strength matrix — head-to-head win rate</h2>
<div class="tablewrap"><table>
<thead><tr><th>row vs col</th>{head}<th>overall</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<h2>Sanity floor — vs scripted baselines</h2>
<p class="sub">Not part of the matrix; a healthy arm should be at or near 100%.</p>
<div class="tablewrap"><table><thead><tr><th>arm</th><th>win rate vs random-legal + simple-legal</th></tr></thead>
<tbody>{base_rows}</tbody></table></div>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", required=True, help="glob of benchmark --summary-out JSONs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="PokeZero head-to-head strength matrix")
    args = ap.parse_args()
    paths = sorted(glob.glob(args.summaries))
    if not paths:
        raise SystemExit(f"no summaries matched {args.summaries}")
    pairs, base, shards = load_pairs(paths)
    if not pairs:
        raise SystemExit("no non-baseline head-to-head records found")
    open(args.out, "w").write(build(pairs, base, shards, args.title))
    print(f"WROTE {args.out} ({len(paths)} summaries, {len(pairs)} pairs, "
          f"{sum(v[2] for v in pairs.values())} games)")
    print(f"  (file size {os.path.getsize(args.out)} bytes)")


if __name__ == "__main__":
    main()
