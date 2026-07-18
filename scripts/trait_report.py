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
# v22-flat2m is a FORK of v22-lr3m at 2M (flat-LR twin) — a separate entity, ordered next to its
# parent so the post-fork divergence is easy to read off the trajectories.
LINEAGE_ORDER = ["m50-ep7", "l200-ep7-wu75", "v22-lr3m", "v22-flat2m", "m50-seq", "l200-seq"]
PALETTE = ["#2563eb", "#dc2626", "#059669", "#0891b2", "#d97706", "#7c3aed"]

# Lineages dropped from the report entirely (every section). The seq lineages stalled at 1000k and
# are no longer being tracked. Their metrics remain on disk, so this is reversible — clear the set
# to bring them back.
REPORT_EXCLUDE_LINEAGES = {"m50-seq", "l200-seq"}

# (lineage, milestone) points dropped from the Phase-1 basics charts only. v22-lr3m@100k stalls
# ~50% of its games to the turn cap; its scale compresses every other lineage's line. The point is
# real and stays in the by-checkpoint trajectories — it is excluded here for legibility, and the
# exclusion is stated in the section rather than hidden.
BASICS_EXCLUDE = {("v22-lr3m", 100000)}

# Lineages held out of the trait <-> foul-play correlation. m50-ep7 was held out while foul-play
# existed only at 500k: it won ~0.20 against a 0.31-0.34 cluster, so at n=5 it was a single
# high-leverage point. With foul-play now also at each frontier, its two points (0.198@500k ->
# 0.353@1000k) sit on a continuum spanning 0.198-0.420 across 10 checkpoints, so it is no longer
# leverage — excluding it would just throw away the win-rate spread. Empty by design; keep the
# hook for a future genuine outlier.
CORR_EXCLUDE_LINEAGES = set()


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
    dropped = []
    for r in rows_self:
        if r.get("milestone") is None:
            continue
        if (r.get("lineage"), r.get("milestone")) in BASICS_EXCLUDE:
            dropped.append(f'{r.get("lineage")}@{r.get("milestone") // 1000}k')
            continue
        by_lin[r.get("lineage")].append(r)
    lineages = [l for l in LINEAGE_ORDER if l in by_lin] + [l for l in by_lin if l not in LINEAGE_ORDER]
    # No win-rate chart here: this section is self-play, where the bot drives both seats, so p1's
    # win rate is ~0.5 by construction. Where it does move it only echoes the tie/timeout rate,
    # which the timeout chart shows directly. Win rate is meaningful only vs foul-play (500k panel).
    turns = {l: [(r["milestone"], r.get("avg_turns")) for r in by_lin[l]] for l in lineages}
    pivots = {l: [(r["milestone"], r.get("avg_pivots")) for r in by_lin[l]] for l in lineages}
    # timeout rate: fraction of games that stalled to the turn cap (a checkpoint that can't close)
    timeout = {l: [(r["milestone"], (r.get("timeout_rate") or 0) * 100) for r in by_lin[l]] for l in lineages}
    if not lineages:
        return '<section><h2>Phase 1 — basics over training</h2><div class="empty">no milestone metrics yet</div></section>'
    drop_note = ('' if not dropped else
                 f'<p class="sub">Excluded for legibility: {esc(", ".join(sorted(set(dropped))))} '
                 f'(stalls ~50% of games to the turn cap; its scale compresses the other lineages). '
                 f'The point is retained in the by-checkpoint trajectories below.</p>')
    return f"""<section>
      <h2>Phase 1 — self-play basics over training</h2>
      {drop_note}
      {legend(lineages)}
      <div class="grid3">
        <div class="card">{svg_lines(turns, "avg turns/game (decided)")}</div>
        <div class="card">{svg_lines(pivots, "avg pivots/seat-game")}</div>
        <div class="card">{svg_lines(timeout, "timeout rate % (stalled to cap)")}</div>
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


# ---- per-checkpoint accessors (used by the trajectory charts) ----
def _catrate(r, c):
    mc = (r.get("move_categories") or {}).get(c)
    return mc.get("uses_per_seat_game_present") if mc else None


def _cattotal(r, c):
    return (r.get("move_categories") or {}).get(c, {}).get("total_uses", 0)


def _extra(r, k):
    return (r.get("move_category_extras") or {}).get(k, 0)


def _fracpct(r, num_key, denom_key):
    """Conditional rate from an UNGATED pair, both out of move_category_extras.

    Never divide an extras counter by move_categories[*].total_uses: extras count every
    occurrence while total_uses is gated on the seat's moveset carrying the move, so a move used
    but not carried (Metronome/Mimic) inflates the numerator only — that mismatch produced a
    100.4% "solar beam in sun". Denominator here is the sum of the mutually exclusive outcomes.
    """
    n = _extra(r, num_key)
    t = _extra(r, denom_key)
    return (100.0 * n / t) if t else None


def _fracpct2(r, num_key, *parts):
    """Same, where the denominator is the sum of several ungated outcome counters."""
    n = _extra(r, num_key)
    t = sum(_extra(r, x) for x in parts)
    return (100.0 * n / t) if t else None


def _switchrate(r, k):
    return (r.get("switch_behavior") or {}).get(k, {}).get("per_seat_game")


def _pct(rate):
    return rate * 100 if rate is not None else None


# (label, accessor) — each becomes a small-multiple line chart over the milestone axis
TRAJECTORY_CHARTS = [
    ("category use / seat-game", [
        ("stat-boost", lambda r: _catrate(r, "cat_stat_boost")),
        ("toxic", lambda r: _catrate(r, "cat_toxic")),
        ("substitute", lambda r: _catrate(r, "cat_substitute")),
        ("spikes", lambda r: _catrate(r, "cat_spikes")),
        ("heal (excl Rest)", lambda r: _catrate(r, "cat_heal")),
        ("phaze", lambda r: _catrate(r, "cat_phaze")),
        ("rest", lambda r: _catrate(r, "cat_rest")),
        ("sleep (excl Yawn)", lambda r: _catrate(r, "cat_sleep")),
        ("knock off", lambda r: _catrate(r, "cat_knockoff")),
    ]),
    ("status-inducing moves / seat-game", [
        ("status move (any)", lambda r: _catrate(r, "cat_status_move")),
        ("paralysis (T-Wave/Stun/Glare)", lambda r: _catrate(r, "cat_para")),
        ("burn (Will-O-Wisp)", lambda r: _catrate(r, "cat_burn")),
        ("toxic", lambda r: _catrate(r, "cat_toxic")),
        ("sleep move", lambda r: _catrate(r, "cat_sleep")),
        ("yawn", lambda r: _catrate(r, "cat_yawn")),
    ]),
    ("conditional breakdowns (%)", [
        ("phaze: enemy boosted/sub %", lambda r: _fracpct2(r, "cat_phaze_justified", "cat_phaze_justified", "cat_phaze_neutral")),
        ("rapid spin: spikes-down %", lambda r: _fracpct(r, "cat_rapidspin_spikesdown", "cat_rapidspin_total")),
        ("solar beam: in sun %", lambda r: _fracpct2(r, "cat_solarbeam_sun", "cat_solarbeam_sun", "cat_solarbeam_nosun")),
        ("BP w/ stat or sub %", lambda r: _fracpct(r, "bp_stat_or_sub", "bp_switch")),
        ("focus punch success %", lambda r: _pct(r.get("focus_punch_success_rate"))),
        ("opp focus punch disrupted %", lambda r: _pct(r.get("opp_focus_punch_disruption_rate"))),
        ("destiny bond success %", lambda r: _pct(r.get("destinybond_success_rate"))),
        ("enemy boom blocked %", lambda r: _pct(r.get("boom_block_rate"))),
    ]),
    ("switch behavior / seat-game", [
        ("immunity switch-in", lambda r: _switchrate(r, "immunity_switchin")),
        ("sleeping mon out", lambda r: _switchrate(r, "switch_out_sleeping")),
        ("frozen mon out", lambda r: _switchrate(r, "switch_out_frozen")),
    ]),
    # resource/endgame in SELF-PLAY: both seats are the same policy, so opp-PP ≈ bot-PP and
    # opp-mons-on-loss ≈ mons-on-win (the winner's margin) — the paired plots are essentially
    # identical, so only one of each is shown here. The bot-vs-opp split IS kept in the 500k panel,
    # where the foul-play column has a genuinely different opponent (FoulPlay).
    ("resource / endgame (self-play — opp mirrors bot, shown once)", [
        ("PP exhausted / game", lambda r: r.get("pp_exhaustion_bot_per_game")),
        ("mons alive at game end (winner)", lambda r: r.get("avg_bot_mons_alive_on_win")),
    ]),
    ("setup payoff", [
        # Reversal/Flail avg BP rises if the policy learns to fire them at low HP; Belly Drum avg
        # KOs measures whether the setup pays off (mons the drummer removes after using it).
        ("reversal/flail avg BP", lambda r: r.get("reversal_avg_bp")),
        ("belly drum: avg KOs after", lambda r: r.get("bellydrum_avg_kos")),
        ("belly drum: % uses w/ a KO", lambda r: _pct(r.get("bellydrum_ko_rate"))),
    ]),
    ("priority moves (Quick Attack / Extreme Speed / Mach Punch)", [
        # column 1: how often priority is used; column 2: the skilled use — fired when the opponent
        # outspeeds us (from turn order); column 3: the payoff — fraction of uses that land the KO.
        ("use / seat-game", lambda r: _catrate(r, "cat_priority")),
        ("vs faster opp %", lambda r: _pct(r.get("priority_vs_faster_rate"))),
        ("KO rate %", lambda r: _pct(r.get("priority_ko_rate"))),
    ]),
    ("intentional weather (use / seat-game when carried)", [
        # use rate conditioned on the move being in the team's pool (games-present denominator).
        ("sunny day", lambda r: _catrate(r, "cat_weather_sun")),
        ("rain dance", lambda r: _catrate(r, "cat_weather_rain")),
    ]),
    ("ability reads (per game, only games the ability is on the team)", [
        ("intimidate activations / game", lambda r: r.get("intimidate_activations_per_game")),
        ("absorb switch-in reads / game", lambda r: r.get("absorb_switchins_per_game")),
    ]),
    ("toxic management", [
        # avg peak toxic stage reached before a badly-poisoned mon leaves/cures/faints. Falling over
        # training = the policy switches toxiced mons out earlier to preserve HP.
        ("avg toxic stage reached", lambda r: r.get("avg_toxic_stage")),
    ]),
]


def _series(rows, fn):
    by = defaultdict(list)
    for r in rows:
        if r.get("milestone") is None:
            continue
        v = fn(r)
        if v is not None:
            by[r.get("lineage")].append((r["milestone"], v))
    return by


def phase2_trajectories(rows_self):
    """Per-checkpoint breakdowns: every trait as a line over the milestone axis (one line per
    lineage, one point per checkpoint) — the by-checkpoint view, not a single 500k aggregate."""
    checkpoints = {(r.get("lineage"), r.get("milestone")) for r in rows_self if r.get("milestone") is not None}
    lineages = sorted({l for l, _ in checkpoints}, key=lambda l: LINEAGE_ORDER.index(l) if l in LINEAGE_ORDER else 99)
    if not checkpoints:
        return ""
    multi = any(sum(1 for l2, m in checkpoints if l2 == l) > 1 for l in lineages)
    note = "" if multi else ('<p class="sub">Only 500k checkpoints present so far — each line is a single '
                             'point until the milestone sweep fills in the trajectory.</p>')
    blocks = [f'<section><h2>Trait breakdowns by checkpoint — self-play trajectories</h2>'
              f'<p class="sub">{len(checkpoints)} checkpoints across {len(lineages)} lineages · '
              f'x-axis = cumulative games · each point is one checkpoint (no aggregation)</p>{note}{legend(lineages)}']
    for group_title, charts in TRAJECTORY_CHARTS:
        cards = "".join(f'<div class="card">{svg_lines(_series(rows_self, fn), label)}</div>' for label, fn in charts)
        blocks.append(f'<h3>{esc(group_title)}</h3><div class="grid3">{cards}</div>')
    blocks.append("</section>")
    return "".join(blocks)


def paired_checkpoints(rows):
    """(lineage, milestone) checkpoints that have BOTH self-play and foul-play metrics. Foul-play
    is run on selected checkpoints (500k + each lineage's frontier), so this is the set the
    self-vs-foul panel and the correlation can speak to."""
    def keys(opp):
        return {(r["lineage"], r["milestone"]) for r in rows
                if r.get("opponent") == opp and r.get("milestone") is not None and r.get("lineage")}
    both = keys("self") & keys("foulplay")
    return sorted(both, key=lambda c: (LINEAGE_ORDER.index(c[0]) if c[0] in LINEAGE_ORDER else 99, c[1]))


def ck_label(c):
    return f"{c[0]}@{c[1] // 1000}k"


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sxy / (sxx * syy) ** 0.5


# extra correlation traits NOT already among the trajectory accessors (avoid duplicate bars)
_CORR_EXTRA = [
    ("timeout rate %", lambda r: _pct(r.get("timeout_rate"))),
    ("avg turns (decided)", lambda r: r.get("avg_turns")),
    ("avg pivots", lambda r: r.get("avg_pivots")),
]


def phase2_correlations(rows):
    """Pearson r of each 500k self-play trait vs the bot's foul-play win rate, across lineages.
    We have foul-play only at 500k, so this correlates the five lineages' 500k self-play behavior
    with how they fare against FoulPlay. Small n — read as directional, not precise."""
    # Trait AND outcome both come from the foul-play games — the same population. (Correlating a
    # self-play trait against a foul-play win rate would mix two different game distributions.)
    foulm = {(r["lineage"], r["milestone"]): r for r in rows
             if r.get("opponent") == "foulplay" and r.get("milestone") is not None and r.get("lineage")}
    all_cks = sorted(foulm, key=lambda c: (LINEAGE_ORDER.index(c[0]) if c[0] in LINEAGE_ORDER else 99, c[1]))
    cks = [c for c in all_cks if c[0] not in CORR_EXCLUDE_LINEAGES]
    held_out = sorted({c[0] for c in all_cks if c[0] in CORR_EXCLUDE_LINEAGES})
    if len(cks) < 3:
        return ""
    winr = {c: foulm[c].get("bot_win_rate") for c in cks}
    vals = [w for w in winr.values() if w is not None]
    spread = (max(vals) - min(vals)) if vals else 0.0
    traits = [(lbl, fn) for _, charts in TRAJECTORY_CHARTS for lbl, fn in charts] + _CORR_EXTRA
    results = []
    for lbl, fn in traits:
        xs, ys = [], []
        for c in cks:
            v, w = fn(foulm[c]), winr[c]   # trait and win rate from the same foul-play games
            if v is not None and w is not None:
                xs.append(v)
                ys.append(w)
        r = _pearson(xs, ys)
        if r is not None:
            results.append((lbl, r, len(xs)))
    results.sort(key=lambda t: -t[1])
    held = ('' if not held_out else
            f'Held out: {esc(", ".join(held_out))} (foul-play win rate far below the rest — a '
            f'high-leverage point at this n). ')
    # power warning scales with how much win-rate variance we actually have
    weak = len(cks) < 8 or spread < 0.08
    warn = (f'<p class="warn">Low power: n={len(cks)} checkpoints over a {spread:.3f} win-rate spread. '
            f'With this little variance the r values are unstable — treat as hypothesis-generating '
            f'only. Foul-play on more checkpoints widens the win-rate axis and fixes this.</p>'
            if weak else
            f'<p class="sub">n={len(cks)} checkpoints spanning a {spread:.3f} win-rate range.</p>')
    return (f'<section><h2>Trait &#8596; foul-play win-rate correlation</h2>'
            f'<p class="sub">Pearson r across {len(cks)} checkpoints. Both the trait and the win rate '
            f'are measured on the <strong>same foul-play games</strong> '
            f'({", ".join(f"{ck_label(c)} {winr[c]:.2f}" for c in cks if winr[c] is not None)}). {held}'
            f'Green = trait tracks a higher foul-play win rate, red = lower.</p>'
            f'<p class="warn">One point per <em>checkpoint</em>, not per game: each checkpoint&#39;s '
            f'1000 foul-play games collapse into a single (trait, win-rate) pair, so n is the number '
            f'of checkpoints — game volume buys precision within a point, not more points. It is also '
            f'an aggregate correlation confounded by overall checkpoint strength (better models do more '
            f'of everything effective <em>and</em> win more), so read it as association, not cause.</p>'
            f'{warn}{_svg_corr(results)}</section>')


PG_LABEL = {
    "cat_stat_boost": "stat-boost", "cat_toxic": "toxic", "cat_substitute": "substitute",
    "cat_spikes": "spikes", "cat_heal": "heal (excl Rest)", "cat_phaze": "phaze",
    "cat_rest": "rest", "cat_sleep": "sleep (excl Yawn)", "cat_para": "paralysis",
    "cat_leechseed": "leech seed", "cat_boom": "explosion/self-destruct",
    "cat_batonpass": "baton pass", "cat_solarbeam": "solar beam",
    "cat_rapidspin_total": "rapid spin", "cat_yawn": "yawn", "cat_wish": "wish",
    "cat_weather_sun": "sunny day", "cat_weather_rain": "rain dance", "cat_curse": "curse",
    "pivot": "pivot (voluntary switch)", "forced_switch": "forced switch",
    "immunity_switchin": "immunity switch-in", "switch_out_sleeping": "sleeping mon out",
    "switch_out_frozen": "frozen mon out", "cat_phaze_justified": "phaze when justified",
    "cat_rapidspin_spikesdown": "rapid spin w/ spikes down", "cat_solarbeam_sun": "solar beam in sun",
    "bp_stat_or_sub": "BP w/ stat or sub", "focuspunch_executed": "focus punch landed",
    "focuspunch_disrupted": "focus punch disrupted",
}


def per_game_corr_section(rows, opponent, heading, blurb):
    """Per-game trait->win correlation, aggregated across checkpoints. Each checkpoint contributes
    its own within-checkpoint r (n = its decided seat-games); we show the mean and the min..max
    range across checkpoints. Agreement in sign across independent checkpoints is the evidence —
    a single r is one model, but the same sign in every model is hard to get from noise."""
    ms = [r for r in rows if r.get("opponent") == opponent and r.get("per_game_correlations")]
    if len(ms) < 2:
        return ""
    traits = sorted({t for m in ms for t in m["per_game_correlations"]})
    results = []
    for t in traits:
        rs = [m["per_game_correlations"][t]["r"] for m in ms if t in m["per_game_correlations"]]
        ns = sum(m["per_game_correlations"][t]["n"] for m in ms if t in m["per_game_correlations"])
        if len(rs) < 2:
            continue
        mean = sum(rs) / len(rs)
        consistent = all(x > 0 for x in rs) or all(x < 0 for x in rs)
        results.append((PG_LABEL.get(t, t), mean, min(rs), max(rs), len(rs), ns, consistent))
    if not results:
        return ""
    results.sort(key=lambda x: -x[1])
    tot_games = sum(m.get("per_game_rows", 0) for m in ms)
    return (f'<section><h2>{esc(heading)}</h2><p class="sub">{blurb} '
            f'{len(ms)} checkpoints, {tot_games:,} decided seat-games total. Bar = mean r across '
            f'checkpoints; whisker = min..max. <strong>Bold</strong> = every checkpoint agrees in '
            f'sign (consistency across independent checkpoints is the signal, not any single r).</p>'
            f'{_svg_pg_corr(results)}</section>')


def _svg_pg_corr(results, width=700, rowh=22, pad=200):
    valw = 46
    h = rowh * len(results) + 22
    x0, x1 = pad, width - valw
    cx, half = (x0 + x1) / 2, (x1 - x0) / 2
    # per-game r values are small; scale to the observed max so the chart is readable
    lim = max(0.05, max(max(abs(m), abs(lo), abs(hi)) for _, m, lo, hi, _, _, _ in results))
    out = [f'<svg viewBox="0 0 {width} {h}" class="chart" role="img" aria-label="per-game trait correlations">']
    for frac in (-1, -0.5, 0, 0.5, 1):
        x = cx + frac * half
        out.append(f'<line x1="{x:.0f}" y1="2" x2="{x:.0f}" y2="{h-16}" class="{"axis" if frac == 0 else "grid"}"/>')
        out.append(f'<text x="{x:.0f}" y="{h-4}" class="xlab" text-anchor="middle">{frac*lim:+.2f}</text>')
    for i, (lbl, mean, lo, hi, k, ns, consistent) in enumerate(results):
        y = 8 + i * rowh
        color = "#059669" if mean >= 0 else "#dc2626"
        bw = rowh * 0.5
        mx = cx + (mean / lim) * half
        lox, hix = cx + (lo / lim) * half, cx + (hi / lim) * half
        weight = ' font-weight="700"' if consistent else ''
        out.append(f'<text x="{x0-8}" y="{y+bw*0.85:.0f}" class="ylab" text-anchor="end"{weight}>{esc(lbl)}</text>')
        out.append(f'<line x1="{lox:.1f}" y1="{y+bw/2:.1f}" x2="{hix:.1f}" y2="{y+bw/2:.1f}" stroke="{color}" stroke-width="1" opacity="0.45"/>')
        out.append(f'<rect x="{min(cx, mx):.1f}" y="{y:.0f}" width="{abs(mx-cx):.1f}" height="{bw:.0f}" rx="2" fill="{color}" opacity="0.85"/>')
        out.append(f'<text x="{width-4}" y="{y+bw*0.85:.0f}" class="ylab" text-anchor="end"{weight}>{mean:+.3f}</text>')
    out.append("</svg>")
    return "".join(out)


def _svg_corr(results, width=700, rowh=24, pad=180):
    if not results:
        return '<div class="empty">not enough lineages with both self + foul 500k</div>'
    valw = 46
    h = rowh * len(results) + 22
    x0, x1 = pad, width - valw
    cx, half = (x0 + x1) / 2, (x1 - x0) / 2
    out = [f'<svg viewBox="0 0 {width} {h}" class="chart" role="img" aria-label="trait correlations">']
    for frac in (-1, -0.5, 0, 0.5, 1):
        x = cx + frac * half
        out.append(f'<line x1="{x:.0f}" y1="2" x2="{x:.0f}" y2="{h-16}" class="{"axis" if frac == 0 else "grid"}"/>')
        if frac in (-1, 0, 1):
            out.append(f'<text x="{x:.0f}" y="{h-4}" class="xlab" text-anchor="middle">{frac:+.0f}</text>')
    for i, (lbl, r, n) in enumerate(results):
        y = 8 + i * rowh
        color = "#059669" if r >= 0 else "#dc2626"
        bx = cx + min(0.0, r) * half
        out.append(f'<text x="{x0-8}" y="{y+rowh*0.55:.0f}" class="ylab" text-anchor="end">{esc(lbl)}</text>')
        out.append(f'<rect x="{bx:.1f}" y="{y:.0f}" width="{abs(r)*half:.1f}" height="{rowh*0.55:.0f}" rx="2" fill="{color}" opacity="0.85"/>')
        out.append(f'<text x="{width-4}" y="{y+rowh*0.55:.0f}" class="ylab" text-anchor="end">{r:+.2f}</text>')
    out.append("</svg>")
    return "".join(out)


def _fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return esc(v)


def phase2_panel(rows, opponent, checkpoints):
    """One column per CHECKPOINT (lineage@milestone) — 500k plus each lineage's frontier — rather
    than a single pinned milestone, so the panel tracks the recent checkpoints too."""
    by_ck = {(r["lineage"], r["milestone"]): r for r in rows if r.get("opponent") == opponent}
    lineages = [c for c in checkpoints if c in by_ck]
    if not lineages:
        return f'<div class="empty">no {esc(opponent)} metrics</div>'
    rows = [by_ck[c] for c in lineages]
    cats = sorted({c for r in rows for c in (r.get("move_categories") or {})})
    switches = sorted({k for r in rows for k in (r.get("switch_behavior") or {})})
    by_lin = by_ck  # rows are addressed by checkpoint key below

    def header():
        return "<tr><th>metric</th>" + "".join(f"<th>{esc(ck_label(c))}</th>" for c in lineages) + "</tr>"

    def row(label, fn):
        cells = "".join(f"<td>{fn(by_lin[l])}</td>" for l in lineages)
        return f"<tr><td class='mlabel'>{esc(label)}</td>{cells}</tr>"

    out = [f'<div class="tablewrap"><table><caption>opponent = {esc(opponent)}</caption>', header()]
    out.append(row("games", lambda r: r.get("n_games")))
    out.append(row("timeout rate", lambda r: _fmt(r.get("timeout_rate"))))
    # win rate is only meaningful against a real opponent: in self-play the bot drives both seats,
    # so p1's rate is ~0.5 by construction and inviting a self-vs-foul comparison would mislead.
    if opponent != "self":
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
    out.append(row("rapid spin: spikes on own side", lambda r: cond(ex(r, "cat_rapidspin_spikesdown"), ex(r, "cat_rapidspin_total"))))
    out.append(row("phaze: enemy boosted / behind sub", lambda r: cond(ex(r, "cat_phaze_justified"), ex(r, "cat_phaze_justified") + ex(r, "cat_phaze_neutral"))))
    out.append(row("solar beam: in sun", lambda r: cond(ex(r, "cat_solarbeam_sun"), ex(r, "cat_solarbeam_sun") + ex(r, "cat_solarbeam_nosun"))))
    out.append(row("BP w/ stat or sub", lambda r: cond(ex(r, "bp_stat_or_sub"), ex(r, "bp_switch"))))
    out.append(row("focus punch success rate", lambda r: f'{_fmt(r.get("focus_punch_success_rate"))} <span class="dim">(n={ex(r, "focuspunch_attempt") or r.get("focus_punch_attempts", 0)})</span>'))
    out.append(row("opp focus punch disrupted", lambda r: f'{_fmt(r.get("opp_focus_punch_disruption_rate"))} <span class="dim">(n={r.get("opp_focus_punch_attempts", 0)})</span>'))

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
    out.append('<tr class="grp"><td colspan="%d">setup payoff (avg over uses)</td></tr>' % (len(lineages) + 1))
    out.append(row("reversal/flail avg BP", lambda r: f'{_fmt(r.get("reversal_avg_bp"), 1)} <span class="dim">(n={r.get("reversal_uses", 0)})</span>'))
    out.append(row("belly drum: avg KOs after", lambda r: f'{_fmt(r.get("bellydrum_avg_kos"), 2)} <span class="dim">(n={r.get("bellydrum_uses", 0)})</span>'))
    out.append(row("belly drum: % uses w/ a KO", lambda r: f'{_pct(r.get("bellydrum_ko_rate"))} <span class="dim">(n={r.get("bellydrum_uses", 0)})</span>'))
    out.append('<tr class="grp"><td colspan="%d">priority moves &amp; destiny bond (rate over uses)</td></tr>' % (len(lineages) + 1))
    out.append(row("priority vs faster opp", lambda r: f'{_fmt(r.get("priority_vs_faster_rate"))} <span class="dim">(n={r.get("priority_uses", 0)})</span>'))
    out.append(row("priority KO rate", lambda r: _fmt(r.get("priority_ko_rate"))))
    out.append(row("destiny bond success", lambda r: f'{_fmt(r.get("destinybond_success_rate"))} <span class="dim">(n={r.get("destinybond_uses", 0)})</span>'))
    out.append('<tr class="grp"><td colspan="%d">ability reads / toxic / boom</td></tr>' % (len(lineages) + 1))
    out.append(row("intimidate activations / game", lambda r: f'{_fmt(r.get("intimidate_activations_per_game"), 2)} <span class="dim">(g={r.get("intimidate_present_seat_games", 0)})</span>'))
    out.append(row("absorb switch-in reads / game", lambda r: f'{_fmt(r.get("absorb_switchins_per_game"), 2)} <span class="dim">(g={r.get("absorb_present_seat_games", 0)})</span>'))
    out.append(row("avg toxic stage reached", lambda r: f'{_fmt(r.get("avg_toxic_stage"), 2)} <span class="dim">(n={r.get("toxic_episodes", 0)})</span>'))
    out.append(row("enemy boom blocked", lambda r: f'{_fmt(r.get("boom_block_rate"))} <span class="dim">(n={r.get("boom_faced", 0)})</span>'))
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
.warn{color:var(--dim);border-left:3px solid #d97706;padding:6px 10px;margin:8px 0;font-size:12.5px;background:var(--card);border-radius:0 6px 6px 0}
.cols2{display:grid;grid-template-columns:1fr;gap:16px}@media(min-width:900px){.cols2{grid-template-columns:1fr 1fr}}
"""


def build_html(rows):
    # drop excluded lineages up front so every section below is consistent
    rows = [r for r in rows if r.get("lineage") not in REPORT_EXCLUDE_LINEAGES]
    rows_self = [r for r in rows if r.get("opponent") == "self"]
    n_self = len(rows_self)
    n_foul = len([r for r in rows if r.get("opponent") == "foulplay"])
    body = [f'<div class="wrap"><h1>PokeZero checkpoint trait tracking</h1>',
            f'<p class="sub">{len(rows)} metric sets · {n_self} self-play · {n_foul} foul-play · '
            f'lineages: {esc(", ".join(sorted({r.get("lineage") for r in rows if r.get("lineage")})))}</p>']
    body.append(phase1_section(rows_self))
    body.append(phase2_trajectories(rows_self))
    body.append(per_game_corr_section(
        rows, "foulplay", "Per-game trait ↔ win correlation (vs FoulPlay)",
        "Within each checkpoint&#39;s foul-play games: did the bot use the trait more in games it "
        "<em>won</em>? One row per game, so n is games rather than checkpoints — this is the "
        "powered version of the aggregate chart below. Confound to keep in mind: longer games "
        "contain more of everything, and game length is not independent of the result."))
    body.append(per_game_corr_section(
        rows, "self", "Per-game trait ↔ win correlation (self-play, paired)",
        "The best-controlled view we have. Both seats are the <em>same policy</em> playing the "
        "<em>same game</em>, so comparing the winning seat&#39;s behavior against the losing "
        "seat&#39;s holds policy strength and game length fixed by construction — a game-level "
        "quantity has no within-game variance and cannot leak in."))
    body.append(phase2_correlations(rows))
    cks = paired_checkpoints(rows)
    body.append('<section><h2>Phase 2 — detailed panel by checkpoint (self vs foul-play)</h2>'
                f'<p class="sub">Full breakdown for every checkpoint with both self-play and foul-play '
                f'({", ".join(ck_label(c) for c in cks) or "none yet"}) — 500k plus each lineage&#39;s '
                f'most recent checkpoint. Self-play and foul-play are kept separate, never merged.</p>'
                f'{phase2_panel(rows, "self", cks)}'
                f'{phase2_panel(rows, "foulplay", cks)}'
                '</section>')
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
