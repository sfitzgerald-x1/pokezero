"""Render trait metrics into one self-contained, no-CDN HTML report.

Consumes a directory of metrics-*.json (each tagged with lineage/milestone/opponent by
trait_extract.py) and renders: Phase-1 trajectories over the cumulative-games axis per lineage
(top moves, avg turns, avg pivots) and a Phase-2 500k cross-lineage panel (move categories,
switch behavior, resource/endgame, species vector) with self-play and foul-play kept separate.
Everything is inline (SVG charts + tables); no network fetches.
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
from collections import defaultdict

# stable lineage order + display names
LINEAGE_ORDER = ["m50-ep7", "l200-ep7-wu75", "v22-lr3m", "m50-seq", "l200-seq"]
PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]


def load(metrics_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(metrics_dir, "metrics-*.json"))):
        try:
            rows.append(json.load(open(path)))
        except Exception:
            continue
    return rows


def esc(x):
    return html.escape(str(x))


def color_for(lineage):
    try:
        return PALETTE[LINEAGE_ORDER.index(lineage) % len(PALETTE)]
    except ValueError:
        return "#64748b"


def svg_lines(series, ylabel, width=560, height=220, pad=44):
    """series: {lineage: [(milestone, value), ...]}. Inline SVG multi-line chart."""
    pts = [(m, v) for s in series.values() for (m, v) in s if v is not None]
    if not pts:
        return f'<div class="empty">no data for {esc(ylabel)}</div>'
    xs = [m for m, _ in pts]
    ys = [v for _, v in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin:
        xmax = xmin + 1
    if ymax == ymin:
        ymax = ymin + 1

    def X(m):
        return pad + (m - xmin) / (xmax - xmin) * (width - 2 * pad)

    def Y(v):
        return height - pad - (v - ymin) / (ymax - ymin) * (height - 2 * pad)

    out = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{esc(ylabel)}">']
    out.append(f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" class="axis"/>')
    out.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" class="axis"/>')
    for frac in (0.0, 0.5, 1.0):
        yv = ymin + frac * (ymax - ymin)
        y = Y(yv)
        out.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{width-pad}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{pad-6}" y="{y+3:.1f}" class="ylab" text-anchor="end">{yv:.2f}</text>')
    out.append(f'<text x="{pad}" y="{height-pad+16}" class="xlab">{xmin/1000:.0f}k</text>')
    out.append(f'<text x="{width-pad}" y="{height-pad+16}" class="xlab" text-anchor="end">{xmax/1000:.0f}k</text>')
    out.append(f'<text x="6" y="{pad-8}" class="ylabel">{esc(ylabel)}</text>')
    for lineage, s in series.items():
        s = sorted((m, v) for m, v in s if v is not None)
        if not s:
            continue
        c = color_for(lineage)
        d = " ".join(f'{X(m):.1f},{Y(v):.1f}' for m, v in s)
        out.append(f'<polyline points="{d}" fill="none" stroke="{c}" stroke-width="2"/>')
        for m, v in s:
            out.append(f'<circle cx="{X(m):.1f}" cy="{Y(v):.1f}" r="2.5" fill="{c}"/>')
    out.append("</svg>")
    return "".join(out)


def legend(lineages):
    items = "".join(
        f'<span class="lg"><i style="background:{color_for(l)}"></i>{esc(l)}</span>' for l in lineages
    )
    return f'<div class="legend">{items}</div>'


def phase1_section(rows_self):
    by_lin = defaultdict(list)
    for r in rows_self:
        if r.get("milestone") is not None:
            by_lin[r.get("lineage")].append(r)
    lineages = [l for l in LINEAGE_ORDER if l in by_lin] + [l for l in by_lin if l not in LINEAGE_ORDER]
    turns = {l: [(r["milestone"], r.get("avg_turns")) for r in by_lin[l]] for l in lineages}
    pivots = {l: [(r["milestone"], r.get("avg_pivots")) for r in by_lin[l]] for l in lineages}
    # top-move stability: share of each lineage's frontier top move over milestones
    winr = {l: [(r["milestone"], r.get("bot_win_rate")) for r in by_lin[l]] for l in lineages}
    if not lineages:
        return '<section><h2>Phase 1 — basics over training</h2><div class="empty">no milestone metrics yet</div></section>'
    return f"""<section>
      <h2>Phase 1 — self-play basics over training</h2>
      {legend(lineages)}
      <div class="grid3">
        <div class="card">{svg_lines(turns, "avg turns/game")}</div>
        <div class="card">{svg_lines(pivots, "avg pivots/seat-game")}</div>
        <div class="card">{svg_lines(winr, "bot win rate")}</div>
      </div>
      {phase1_moves_table(by_lin, lineages)}
    </section>"""


def phase1_moves_table(by_lin, lineages):
    parts = ['<h3>Top-5 moves at the frontier milestone</h3><div class="tablewrap"><table>']
    parts.append("<tr><th>lineage</th><th>milestone</th><th>top-5 moves (share)</th></tr>")
    for l in lineages:
        rr = sorted(by_lin[l], key=lambda r: r["milestone"])
        if not rr:
            continue
        r = rr[-1]
        moves = ", ".join(f'{esc(m["move"])} ({m["share"]:.3f})' for m in r.get("top5_moves", []))
        parts.append(f'<tr><td>{esc(l)}</td><td>{r["milestone"]/1000:.0f}k</td><td>{moves}</td></tr>')
    parts.append("</table></div>")
    return "".join(parts)


def _fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return esc(v)


def phase2_panel(rows, opponent):
    rows = [r for r in rows if r.get("opponent") == opponent and r.get("milestone") == 500000]
    by_lin = {r.get("lineage"): r for r in rows}
    lineages = [l for l in LINEAGE_ORDER if l in by_lin] + [l for l in by_lin if l not in LINEAGE_ORDER]
    if not lineages:
        return f'<div class="empty">no {esc(opponent)} 500k metrics</div>'
    cats = sorted({c for r in rows for c in (r.get("move_categories") or {})})
    switches = sorted({k for r in rows for k in (r.get("switch_behavior") or {})})

    def header():
        return "<tr><th>metric</th>" + "".join(f"<th>{esc(l)}</th>" for l in lineages) + "</tr>"

    def row(label, fn):
        cells = "".join(f"<td>{fn(by_lin[l])}</td>" for l in lineages)
        return f"<tr><td class='mlabel'>{esc(label)}</td>{cells}</tr>"

    out = [f'<div class="tablewrap"><table><caption>opponent = {esc(opponent)}</caption>', header()]
    out.append(row("games", lambda r: r.get("n_games")))
    out.append(row("bot win rate", lambda r: _fmt(r.get("bot_win_rate"))))
    out.append(row("avg turns", lambda r: _fmt(r.get("avg_turns"), 1)))
    out.append(row("avg pivots", lambda r: _fmt(r.get("avg_pivots"), 2)))
    out.append('<tr class="grp"><td colspan="%d">move categories — uses / seat-game present (carrier rate)</td></tr>' % (len(lineages) + 1))
    for c in cats:
        def fn(r, c=c):
            mc = (r.get("move_categories") or {}).get(c)
            if not mc:
                return "—"
            return f'{mc["uses_per_seat_game_present"]:.2f} <span class="dim">({mc["carrier_rate"]:.2f})</span>'
        out.append(row(c.replace("cat_", ""), fn))

    # conditional breakdowns — the "only when it matters" splits, surfaced explicitly
    def ex(r, k):
        return (r.get("move_category_extras") or {}).get(k, 0)

    def cat_total(r, k):
        return (r.get("move_categories") or {}).get(k, {}).get("total_uses", 0)

    def cond(c, t):
        if not t:
            return f"{c}" if c else "—"
        return f'{c}/{t} <span class="dim">({c / t * 100:.0f}%)</span>'

    out.append('<tr class="grp"><td colspan="%d">conditional breakdowns — occurrences meeting the condition / category total</td></tr>' % (len(lineages) + 1))
    out.append(row("rapid spin: spikes on own side", lambda r: cond(ex(r, "cat_rapidspin_spikesdown"), cat_total(r, "cat_rapidspin_total"))))
    out.append(row("phaze: enemy boosted / behind sub", lambda r: cond(ex(r, "cat_phaze_justified"), cat_total(r, "cat_phaze"))))
    out.append(row("solar beam: in sun", lambda r: cond(ex(r, "cat_solarbeam_sun"), cat_total(r, "cat_solarbeam"))))
    out.append(row("baton pass: actual BP switches", lambda r: cond(ex(r, "bp_switch"), cat_total(r, "cat_batonpass"))))
    out.append(row("explosion / self-destruct", lambda r: f'{ex(r, "cat_boom_explosion")} / {ex(r, "cat_boom_selfdestruct")}'))
    out.append(row("focus punch: executed / disrupted", lambda r: f'{ex(r, "focuspunch_executed")} / {ex(r, "focuspunch_disrupted")}'))

    out.append('<tr class="grp"><td colspan="%d">switch behavior — per seat-game</td></tr>' % (len(lineages) + 1))
    for s in switches:
        out.append(row(s, lambda r, s=s: _fmt((r.get("switch_behavior") or {}).get(s, {}).get("per_seat_game"))))
    out.append('<tr class="grp"><td colspan="%d">resource / endgame</td></tr>' % (len(lineages) + 1))
    out.append(row("bot PP exhaust/game", lambda r: _fmt(r.get("pp_exhaustion_bot_per_game"), 2)))
    out.append(row("opp PP exhaust/game", lambda r: _fmt(r.get("pp_exhaustion_opp_per_game"), 2)))
    out.append(row("mons alive on win", lambda r: _fmt(r.get("avg_bot_mons_alive_on_win"), 2)))
    out.append(row("opp mons alive on loss", lambda r: _fmt(r.get("avg_opp_mons_alive_on_loss"), 2)))
    out.append(row("focus-punch success", lambda r: _fmt(r.get("focus_punch_success_rate"))))
    out.append(row("top closer on win", lambda r: esc((r.get("top5_last_active_on_win") or [["—"]])[0][0])))
    out.append("</table></div>")
    return "".join(out)


CSS = """
:root{--bg:#fff;--fg:#0f172a;--dim:#64748b;--line:#e2e8f0;--card:#f8fafc;--accent:#2563eb}
@media(prefers-color-scheme:dark){:root{--bg:#0b1220;--fg:#e5edf7;--dim:#93a4bb;--line:#1e2a3d;--card:#111a2b;--accent:#60a5fa}}
:root[data-theme=dark]{--bg:#0b1220;--fg:#e5edf7;--dim:#93a4bb;--line:#1e2a3d;--card:#111a2b;--accent:#60a5fa}
:root[data-theme=light]{--bg:#fff;--fg:#0f172a;--dim:#64748b;--line:#e2e8f0;--card:#f8fafc;--accent:#2563eb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1040px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 12px;border-bottom:1px solid var(--line);padding-bottom:6px}
h3{font-size:14px;margin:20px 0 8px;color:var(--dim)}
.sub{color:var(--dim);margin:0 0 8px}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}
.chart{width:100%;height:auto}.axis{stroke:var(--dim);stroke-width:1}.grid{stroke:var(--line);stroke-width:1}
.ylab,.xlab{fill:var(--dim);font-size:10px}.ylabel{fill:var(--dim);font-size:11px;font-weight:600}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:6px 0 14px}.lg{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--dim)}
.lg i{width:12px;height:12px;border-radius:3px;display:inline-block}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;margin:10px 0}
table{border-collapse:collapse;width:100%;font-size:12.5px}caption{text-align:left;padding:8px 10px;color:var(--dim);font-weight:600}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}.mlabel{color:var(--dim)}
tr.grp td{background:var(--card);color:var(--accent);font-weight:600;text-align:left}
.dim{color:var(--dim)}.empty{color:var(--dim);padding:16px;font-style:italic}
.cols2{display:grid;grid-template-columns:1fr;gap:16px}@media(min-width:900px){.cols2{grid-template-columns:1fr 1fr}}
"""


def build_html(rows):
    rows_self = [r for r in rows if r.get("opponent") == "self"]
    n_self = len(rows_self)
    n_foul = len([r for r in rows if r.get("opponent") == "foulplay"])
    body = [f'<div class="wrap"><h1>PokeZero checkpoint trait tracking</h1>',
            f'<p class="sub">{len(rows)} metric sets · {n_self} self-play · {n_foul} foul-play · '
            f'lineages: {esc(", ".join(sorted({r.get("lineage") for r in rows if r.get("lineage")})))}</p>']
    body.append(phase1_section(rows_self))
    body.append('<section><h2>Phase 2 — 500k trait panel</h2>'
                '<div class="cols2">'
                f'<div>{phase2_panel(rows, "self")}</div>'
                f'<div>{phase2_panel(rows, "foulplay")}</div>'
                '</div></section>')
    body.append("</div>")
    return f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>PokeZero trait tracking</title><style>{CSS}</style></head><body>{''.join(body)}</body></html>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = load(args.metrics_dir)
    open(args.out, "w").write(build_html(rows))
    print(f"WROTE {args.out} ({len(rows)} metric sets)")


if __name__ == "__main__":
    main()
