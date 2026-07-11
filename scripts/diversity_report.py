"""Render a self-contained HTML report for one diversity-fingerprint analysis run.

Reads pairwise.json (Stage A action/value) and, when present, style.json and
matchup.json (Stage A style + Stage B), and renders similarity heatmaps anchored to
the within-run null band, the pair x layer verdict matrix, the null band + gate panel,
and the roster provenance table. No CDN, no network: inline CSS + inline SVG only,
styled for light and dark. `--index` writes an index over sibling analysis dirs.
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os

LABEL_ORDER = ["ref10m", "clean50m", "orig200m", "seqL200m", "lr15L200m"]

CSS = """
:root { --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#e2e2e2; --card:#f7f7f8;
        --same:#2e7d32; --diverse:#c62828; --accent:#1565c0; }
:root[data-theme=dark]{ --bg:#14161a; --fg:#e6e6e6; --muted:#9aa; --line:#2a2e35; --card:#1c1f26;
        --same:#66bb6a; --diverse:#ef5350; --accent:#64b5f6; }
@media (prefers-color-scheme: dark){ :root:not([data-theme=light]){ --bg:#14161a; --fg:#e6e6e6; --muted:#9aa;
        --line:#2a2e35; --card:#1c1f26; --same:#66bb6a; --diverse:#ef5350; --accent:#64b5f6; } }
* { box-sizing:border-box; } body{ margin:0; padding:24px; background:var(--bg); color:var(--fg);
        font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
h1{ font-size:22px; margin:0 0 4px; } h2{ font-size:16px; margin:28px 0 10px; border-bottom:1px solid var(--line); padding-bottom:4px; }
.muted{ color:var(--muted); } .card{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:14px; margin:10px 0; }
table{ border-collapse:collapse; font-size:13px; } th,td{ padding:5px 9px; border:1px solid var(--line); text-align:center; }
th{ background:var(--card); font-weight:600; } td.l{ text-align:left; } code{ font-size:12px; color:var(--muted); }
.same{ color:var(--same); font-weight:600; } .diverse{ color:var(--diverse); font-weight:600; }
.grid{ display:flex; flex-wrap:wrap; gap:20px; } .hm{ overflow-x:auto; }
.tag{ display:inline-block; padding:1px 7px; border-radius:10px; font-size:11px; border:1px solid var(--line); }
"""


def esc(s):
    return html.escape(str(s))


def heatmap_svg(title, labels, mat, p95, vmax=None):
    """mat[i][j] distance; cells above p95 tinted diverse, below tinted same, anchored at p95."""
    n = len(labels)
    cell, pad, top = 46, 90, 60
    W = pad + n * cell + 20
    H = top + n * cell + 20
    vmax = vmax or max((mat[i][j] for i in range(n) for j in range(n) if i != j), default=1.0) or 1.0
    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto">']
    parts.append(f'<text x="8" y="20" fill="var(--fg)" font-size="13" font-weight="600">{esc(title)}</text>')
    parts.append(f'<text x="8" y="38" fill="var(--muted)" font-size="11">null p95 = {p95:.4f} (cells above ⇒ diverse)</text>')
    for j, lab in enumerate(labels):
        x = pad + j * cell + cell / 2
        parts.append(f'<text x="{x}" y="{top-6}" fill="var(--muted)" font-size="10" text-anchor="middle" transform="rotate(-30 {x} {top-6})">{esc(lab)}</text>')
    for i, lab in enumerate(labels):
        y = top + i * cell + cell / 2 + 4
        parts.append(f'<text x="{pad-6}" y="{y}" fill="var(--muted)" font-size="10" text-anchor="end">{esc(lab)}</text>')
        for j in range(n):
            x = pad + j * cell
            yy = top + i * cell
            if i == j:
                parts.append(f'<rect x="{x}" y="{yy}" width="{cell-2}" height="{cell-2}" fill="var(--line)"/>')
                continue
            v = mat[i][j]
            if p95 and v > p95:
                t = min(1.0, (v - p95) / max(vmax - p95, 1e-9))
                fill = f'rgba(198,40,40,{0.25 + 0.6*t})'
            else:
                t = 1.0 - (v / p95 if p95 else 0)
                fill = f'rgba(46,125,50,{0.15 + 0.45*max(t,0)})'
            parts.append(f'<rect x="{x}" y="{yy}" width="{cell-2}" height="{cell-2}" fill="{fill}"/>')
            parts.append(f'<text x="{x+cell/2-1}" y="{yy+cell/2+3}" fill="var(--fg)" font-size="9" text-anchor="middle">{v:.3f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def build_matrix(pairwise, labels, metric):
    idx = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    mat = [[0.0] * n for _ in range(n)]
    for pk, m in pairwise.items():
        a, b = pk.split("|")
        la, lb = a.split(":")[0], b.split(":")[0]
        if a.endswith(":roster") and b.endswith(":roster") and la in idx and lb in idx:
            v = m.get(metric)
            if v is None:
                continue
            mat[idx[la]][idx[lb]] = mat[idx[lb]][idx[la]] = v
    return mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", required=True)
    ap.add_argument("--index", action="store_true")
    args = ap.parse_args()

    if args.index:
        rows = []
        for d in sorted(glob.glob(os.path.join(args.analysis_dir, "diversity-*"))):
            pj = os.path.join(d, "pairwise.json")
            if os.path.isfile(pj):
                p = json.load(open(pj))
                nd = sum(1 for v in p.get("verdicts", {}).values() if v["action"] == "diverse" or v["value"] == "diverse")
                rows.append(f'<tr><td class="l"><a href="{esc(os.path.basename(d))}/report.html">{esc(os.path.basename(d))}</a></td>'
                            f'<td>{p.get("n_common_decisions")}</td><td>{len(p.get("verdicts",{}))}</td><td>{nd}</td></tr>')
        page = f"<style>{CSS}</style><h1>Diversity fingerprint analyses</h1><table><tr><th>run</th><th>decisions</th><th>pairs</th><th>diverse-on-≥1-layer</th></tr>{''.join(rows)}</table>"
        open(os.path.join(args.analysis_dir, "index.html"), "w").write(page)
        print("wrote index.html")
        return

    p = json.load(open(os.path.join(args.analysis_dir, "pairwise.json")))
    metas, pairwise, null_band = p["checkpoints"], p["pairwise"], p["null_band"]
    verdicts, gate = p["verdicts"], p["gate_shuffle"]
    labels = [l for l in LABEL_ORDER if f"{l}:roster" in metas] + [l.split(":")[0] for l in metas if l.endswith(":roster") and l.split(":")[0] not in LABEL_ORDER]

    style_path = os.path.join(args.analysis_dir, "style.json")
    style = json.load(open(style_path)) if os.path.isfile(style_path) else None
    matchup_path = os.path.join(args.analysis_dir, "matchup.json")
    matchup = json.load(open(matchup_path)) if os.path.isfile(matchup_path) else None

    def style_lookup(la, lb):
        if not style:
            return None
        for pk, v in style["verdicts"].items():
            a, b = pk.split("|")[0].split(":")[0], pk.split("|")[1].split(":")[0]
            if {a, b} == {la, lb}:
                return v
        return None

    def matchup_lookup(la, lb):
        if not matchup:
            return None
        return matchup["verdicts"].get(f"{la}|{lb}") or matchup["verdicts"].get(f"{lb}|{la}")

    out = [f"<style>{CSS}</style>", "<h1>Strategy-diversity fingerprints — Stage A</h1>",
           f'<div class="muted">common decisions: {p["n_common_decisions"]} · verdict rule: a cross-run pair is <b>diverse</b> on a layer iff a metric exceeds the within-run null p95</div>']

    # roster provenance
    out.append("<h2>Roster</h2><div class='hm'><table><tr><th>label</th><th>role</th><th>schema</th><th>census</th><th>decisions</th><th>checkpoint</th></tr>")
    for k in sorted(metas):
        m = metas[k]
        out.append(f"<tr><td>{esc(m['label'])}</td><td>{esc(m['role'])}</td><td>{esc(m['observation_schema'])}</td><td>{m['numeric_census']}</td><td>{m['n_decisions']}</td><td class='l'><code>{esc(m['checkpoint'])}</code></td></tr>")
    out.append("</table></div>")

    # heatmaps
    out.append("<h2>Similarity heatmaps (roster × roster, distance)</h2><div class='grid'>")
    for metric, title in [("top1_disagreement", "Action: top-1 disagreement"),
                          ("js_divergence", "Action: policy JS divergence"),
                          ("value_1_minus_pearson", "Value: 1 − Pearson"),
                          ("value_p95_abs", "Value: p95 |Δvalue|")]:
        mat = build_matrix(pairwise, labels, metric)
        pb = null_band[metric]["p95"] or 0.0
        out.append(f"<div class='hm'>{heatmap_svg(title, labels, mat, pb)}</div>")
    out.append("</div>")

    # verdict matrix (all layers)
    out.append("<h2>Verdict matrix (cross-run roster pairs)</h2><div class='hm'><table><tr><th>pair</th><th>action</th><th>value</th><th>style</th><th>matchup</th><th>layers diverse</th></tr>")
    for pk, v in verdicts.items():
        la, lb = pk.split("|")[0].split(":")[0], pk.split("|")[1].split(":")[0]
        st = style_lookup(la, lb)
        mu = matchup_lookup(la, lb)
        cells = {"action": v["action"], "value": v["value"],
                 "style": (st["style"] if st else "—"), "matchup": (mu["matchup"] if mu else "—")}
        ndiv = sum(1 for x in cells.values() if x == "diverse")
        row = f"<tr><td class='l'>{esc(la)} vs {esc(lb)}</td>"
        for lyr in ("action", "value", "style", "matchup"):
            c = cells[lyr]
            cls = c if c in ("same", "diverse") else "muted"
            row += f"<td class='{cls}'>{c}</td>"
        row += f"<td><b>{ndiv}/{sum(1 for x in cells.values() if x!='—')}</b></td></tr>"
        out.append(row)
    out.append("</table></div>")

    if style:
        out.append("<h2>Style layer (behavioral z-vectors)</h2><div class='grid'>")
        smat = [[0.0] * len(labels) for _ in labels]
        idx = {l: i for i, l in enumerate(labels)}
        for pk, d in style["pairwise_distance"].items():
            a, b = pk.split("|")[0], pk.split("|")[1]
            la, lb = a.split(":")[0], b.split(":")[0]
            if a.endswith(":roster") and b.endswith(":roster") and la in idx and lb in idx:
                smat[idx[la]][idx[lb]] = smat[idx[lb]][idx[la]] = d
        out.append(f"<div class='hm'>{heatmap_svg('Style: euclidean distance (z-scored features)', labels, smat, style['null_band']['p95'] or 0.0)}</div>")
        out.append(f"<div class='card muted'>Style features: {esc(', '.join(style['features']))}. Null p95={style['null_band']['p95']:.2f} (n={style['null_band']['n']}). "
                   f"NOTE: at 500 self-play games the within-run null band carries sampling noise, so style verdicts are conservative.</div></div>")

    if matchup:
        it = matchup["intransitivity"]
        out.append("<h2>Matchup layer (round-robin)</h2>")
        verdict = "NON-TRANSITIVE (rock-paper-scissors structure the strength axis can't explain)" if it["significant_cycles"] else "TRANSITIVE (one strength axis explains the win matrix)"
        cls = "diverse" if it["significant_cycles"] else "same"
        out.append("<div class='card'>Bradley-Terry strengths: " +
                   ", ".join(f"<b>{esc(l)}</b> {s:+.2f}" for l, s in sorted(matchup['bt_strengths'].items(), key=lambda x: -x[1])) +
                   f"<br>Intransitivity: observed <b>{it['observed']:.4f}</b> vs null mean {it['null_mean']:.4f} (95th {it['null_p95']:.4f}), "
                   f"p={it['p_value']:.3f} → <b class='{cls}'>{esc(verdict)}</b></div>")
        labs = matchup["labels"]
        out.append("<div class='hm'><table><tr><th>row beats col →</th>" + "".join(f"<th>{esc(l)}</th>" for l in labs) + "</tr>")
        for a in labs:
            row = f"<tr><td class='l'>{esc(a)}</td>"
            for b in labs:
                w = None if a == b else matchup["win_matrix"][a].get(b)
                row += "<td style='background:var(--line)'></td>" if a == b else (f"<td>{w:.2f}</td>" if w is not None else "<td>—</td>")
            out.append(row + "</tr>")
        out.append("</table></div>")

    # null band + gate
    out.append("<h2>Within-run null band + gates</h2><div class='card'><table><tr><th>metric</th><th>n pairs</th><th>p95 (threshold)</th><th>max</th></tr>")
    for m in ("top1_disagreement", "js_divergence", "value_1_minus_pearson", "value_p95_abs"):
        nb = null_band[m]
        out.append(f"<tr><td class='l'>{m}</td><td>{nb['n']}</td><td>{nb['p95']:.4f}</td><td>{nb['max']:.4f}</td></tr>")
    out.append("</table>")
    gp = "PASS" if gate["passes"] else "FAIL"
    out.append(f"<p>Shuffled-label control on <code>{esc(gate['pair'])}</code>: aligned top1-dis={gate['aligned']['top1_disagreement']:.3f} JS={gate['aligned']['js_divergence']:.4f} → "
               f"shuffled top1-dis={gate['shuffled']['top1_disagreement']:.3f} JS={gate['shuffled']['js_divergence']:.4f} — <b class='{'same' if gate['passes'] else 'diverse'}'>{gp}</b> "
               f"(shuffling must inflate both).</p></div>")
    out.append(f"<p class='muted'>null band n=5 (five within-run roster/null pairs); with n=5 the p95≈max, a deliberately conservative bar. Style and matchup layers are added by later stages.</p>")

    open(os.path.join(args.analysis_dir, "report.html"), "w").write("".join(out))
    print("wrote report.html")


if __name__ == "__main__":
    main()
