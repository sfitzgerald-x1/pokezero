#!/usr/bin/env python3
"""Generate the v2.2 (turn-merged) observation token-format explainer HTML.

Reads the committed turn-16 dump (extract_turn16.py) and emits
token-format-v2_2.html next to it. Same visual language as the v2 turn-10 doc
(dark, token cards with section hues, chips + numeric tables, token strip, census
groups); the structure is extended for the turn-merged schema: sub-block A/B card
anatomy, the Explosion double-replacement walkthrough, the per-mon pinned Tier-2
surface, PP-validity bits, defender identity, and the dual-schema story.

Run from anywhere:  uv run python docs/token-format/generate_token_doc.py
"""
import html
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = json.load(open(HERE / "turn16-token-dump.json"))
L, C, M = D["layout"], D["context"], D["masks"]
TOK = D["tokens"]
CAT_LAYOUT = L["categorical_slot_layout"]

HUE = {
    "field": "#7ab8d9", "self_pokemon": "#86c48e", "opponent_pokemon": "#d9a26b",
    "action_candidate": "#d98a99", "stats": "#d4c27a", "transition": "#a795d9",
}
NICE = {
    "field": "Field", "self_pokemon": "Self mon", "opponent_pokemon": "Opponent mon",
    "action_candidate": "Action candidate", "stats": "Stats", "transition": "Turn-merged transition",
}


def esc(x):
    return html.escape(str(x))


def fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


# ---------- token strip ----------
attended = {t["index"] for t in TOK}
strip_cells = []
for sec, meta in L["sections"].items():
    for i in range(meta["offset"], meta["offset"] + meta["count"]):
        on = i in attended
        strip_cells.append(
            f'<i style="background:{HUE[sec]};opacity:{1.0 if on else 0.16}" '
            f'title="{i}: {NICE[sec]}{"" if on else " (padded)"}"></i>')
strip = "".join(strip_cells)

legend = " ".join(
    f'<span class="lgd"><i style="background:{HUE[s]}"></i>{NICE[s]} <span class="mut">&times;{m["count"]}</span></span>'
    for s, m in L["sections"].items())


# ---------- token cards ----------
# Column routing for the two-sub-block anatomy of a v2.2 transition row.
def _tm_bucket(kind, name):
    if kind == "cat":
        if name.startswith("CATEGORY_TM_SECOND_"):
            return "B"
        if name in ("CATEGORY_SLOT", "CATEGORY_ROLE", "CATEGORY_MOVE_EFFECT"):
            return "shared"
        return "A"  # PRIMARY/SECONDARY/TYPE_*/MOVE_CATEGORY/MOVE_PRIORITY/TM_FIRST_*
    if name.startswith("NUMERIC_TM2_"):
        return "B"
    if name in ("NUMERIC_PRESENT", "NUMERIC_TT_OWN_SPIKES", "NUMERIC_TT_OPP_SPIKES",
                "NUMERIC_TT_ABS_TURN", "NUMERIC_TT_TURNS_AGO"):
        return "shared"
    return "A"


def _chip(slot, cv):
    label = cv["token"] if isinstance(cv, dict) else cv
    cid = cv.get("id", "?") if isinstance(cv, dict) else "?"
    short = slot.replace("CATEGORY_", "").replace("_OFFSET", "").lower()
    return f'<span class="chip" title="{esc(slot)} (id {esc(cid)})"><b>{esc(short)}</b> {esc(label)}</span>'


def _numrows(items):
    return "".join(f'<tr><td>{esc(k)}</td><td>{fmt(v)}</td></tr>' for k, v in items)


def token_card(t):
    sec = t["section"]
    header = (f'<div class="tok-h"><span class="idx">token {t["index"]}</span>'
              f'<span class="mut">{NICE[sec]} &middot; type_id {t["token_type_id"]}</span></div>')
    if sec != "transition":
        chips = "".join(_chip(s, cv) for s, cv in t["categoricals"].items())
        return (f'<div class="tok" style="border-left-color:{HUE[sec]}">{header}'
                f'<div class="chips">{chips}</div>'
                f'<table class="nums">{_numrows(t["numerics"].items())}</table></div>')
    # v2.2 transition row: shared context + sub-block A + sub-block B.
    buckets = {"shared": {"chips": [], "nums": []}, "A": {"chips": [], "nums": []}, "B": {"chips": [], "nums": []}}
    for slot, cv in t["categoricals"].items():
        buckets[_tm_bucket("cat", slot)]["chips"].append(_chip(slot, cv))
    for name, value in t["numerics"].items():
        buckets[_tm_bucket("num", name)]["nums"].append((name, value))
    dec = t.get("decoded_turn_merged_token", {})
    second_status = dec.get("second", {}).get("status", "?")
    b_title = "B &middot; second mover" if second_status == "action" else f"B &middot; {esc(second_status)} (no executed action)"
    parts = [f'<div class="tok tok-tm" style="border-left-color:{HUE[sec]}">{header}']
    parts.append(f'<div class="sb sb-shared"><h5>shared &middot; phase / context trio / position</h5>'
                 f'<div class="chips">{"".join(buckets["shared"]["chips"])}</div>'
                 f'<table class="nums">{_numrows(buckets["shared"]["nums"])}</table></div>')
    parts.append(f'<div class="sb sb-a"><h5>A &middot; first mover</h5>'
                 f'<div class="chips">{"".join(buckets["A"]["chips"])}</div>'
                 f'<table class="nums">{_numrows(buckets["A"]["nums"])}</table></div>')
    parts.append(f'<div class="sb sb-b"><h5>{b_title}</h5>'
                 f'<div class="chips">{"".join(buckets["B"]["chips"])}</div>'
                 f'<table class="nums">{_numrows(buckets["B"]["nums"])}</table></div>')
    parts.append('</div>')
    return "".join(parts)


def section_tokens(sec):
    return "".join(token_card(t) for t in TOK if t["section"] == sec)


# ---------- protocol lines ----------
def proto(lines):
    out = []
    for ln in lines:
        cls = "pl"
        if ln.startswith("|turn|"):
            cls += " pl-turn"
        elif "|faint|" in ln or "-damage" in ln or "-sidestart" in ln or "-enditem" in ln:
            cls += " pl-dmg"
        out.append(f'<span class="{cls}">{esc(ln)}</span>')
    return "<pre class='proto'>" + "\n".join(out) + "</pre>"


# ---------- decoded-fields table for worked examples ----------
def decoded_table(dec):
    rows = [("turn", dec["turn"]), ("phase", dec["phase"])]
    trio = dec.get("context_trio", {})
    rows.append(("context trio", f'own spikes {trio.get("own_spikes_layers")} / opp spikes '
                                 f'{trio.get("opp_spikes_layers")} / weather {trio.get("weather")}'))
    body = "".join(f"<tr><td>{esc(k)}</td><td>{esc(fmt(v))}</td></tr>" for k, v in rows)
    subs = ""
    for label, key in (("first (A)", "first"), ("second (B)", "second")):
        sub = dec.get(key, {})
        inner = "".join(f"<tr><td>{esc(k)}</td><td>{esc(fmt(v))}</td></tr>" for k, v in sub.items())
        subs += f'<h5>{esc(label)}</h5><table class="nums df">{inner}</table>'
    return f'<table class="nums df">{body}</table>{subs}'


def worked_example(ex, highlight=False):
    cls = "ex ex-hi" if highlight else "ex"
    companion_html = ""
    for comp in ex.get("companion_tokens", []):
        companion_html += (f'<h4>companion token (slot {esc(comp["transition_slot"])}) '
                           f'&mdash; the merged cold-replacement pair</h4>'
                           f'{token_card(comp["encoded_token"])}')
    return (
        f'<div class="{cls}"><p class="ex-note">{esc(ex["note"])}</p>'
        f'<div class="ex-grid">'
        f'<div><h4>protocol</h4>{proto(ex["protocol_lines"])}'
        f'<h4>extracted TurnMergedToken</h4>{decoded_table(ex["decoded_fields"])}</div>'
        f'<div><h4>encoded token (slot {esc(ex["transition_slot"])})</h4>'
        f'{token_card(ex["encoded_token"])}{companion_html}</div>'
        f'</div></div>')


examples_html = "".join(worked_example(ex) for ex in D["line_to_token_examples"][1:])
explosion_html = worked_example(D["line_to_token_examples"][0], highlight=True)

# ---------- numeric column census ----------
GROUPS = [
    ("Core presence & identity", r"^NUMERIC_(HP_FRACTION|ACTIVE|LEGAL|PRESENT|REVEALED|FAINTED|STATUS_|BOOST_|LEVEL)"),
    ("Raw mechanics (dex-derived)", r"^NUMERIC_(BASE_|PRIORITY|ACCURACY|EFFECT_CHANCE|SELF_HP_COST|MOVE_PP)"),
    ("Field & side conditions", r"^NUMERIC_(TURN_COUNT|WEATHER|SPIKES|SIDE_|SCREEN|REFLECT|LIGHT|SAFEGUARD|MIST|SELF_HAZARDS|OPP_HAZARDS|SELF_SCREENS|OPP_SCREENS|SELF_FUTURE|OPP_FUTURE|SELF_SLEEP_CLAUSE|OPP_SLEEP_CLAUSE|SELF_WISH|OPP_WISH|SELF_REFLECT|SELF_LIGHT|SELF_SAFEGUARD|SELF_MIST|OPP_REFLECT|OPP_LIGHT|OPP_SAFEGUARD|OPP_MIST|TOXIC_STAGE)"),
    ("Exact-state layer", r"^NUMERIC_(PP_|SLEEP|REST|WAKE|TURNS_A|TRAPPER|WISH|CANDIDATE|POSSIBLE|UNCERTAINTY|EXPECTED_|ACTUAL_|MON_)"),
    ("Opponent move PP ledger (v2)", r"^NUMERIC_OPP_MOVE_PP_OFFSET"),
    ("Stats / tendency block", r"^NUMERIC_STAT_"),
    ("Transition sub-block A / per-action fields", r"^NUMERIC_TT_"),
    ("v2.1 block: PP-validity bits", r"^NUMERIC_OPP_MOVE_PP_VALID"),
    ("v2.1 block: sub HP + per-mon pinned Tier-2", r"^NUMERIC_(SUB_HP|TIER2_)"),
    ("v2.2 block: transition sub-block B (TM2)", r"^NUMERIC_TM2_"),
    ("Everything else", r"."),
]
cols = list(L["numeric_column_names"])
seen = set()
census_html = []
for title, pat in GROUPS:
    rx = re.compile(pat)
    members = [(i, c) for i, c in enumerate(cols) if i not in seen and rx.search(c)]
    if not members:
        continue
    for i, _ in members:
        seen.add(i)
    rows = "".join(f'<tr><td class="mut">{i}</td><td>{esc(c)}</td></tr>' for i, c in members)
    census_html.append(
        f'<details class="census"><summary>{esc(title)} <span class="mut">({len(members)} columns)</span></summary>'
        f'<table class="nums census-t">{rows}</table></details>')
census = "".join(census_html)

# ---------- categorical column census ----------
CAT_GROUPS = [
    ("Fixed slots (9)", r"^CATEGORY_(PRIMARY|SECONDARY|ROLE|SLOT|TYPE_|MOVE_)"),
    ("Belief buckets + volatiles", r"^CATEGORY_(BELIEF_|VOLATILE_)"),
    ("v2.2 turn-merged columns (12)", r"^CATEGORY_TM_"),
]
cat_cols = list(CAT_LAYOUT["column_names"])
cat_seen = set()
cat_census_html = []
for title, pat in CAT_GROUPS:
    rx = re.compile(pat)
    members = [(i, c) for i, c in enumerate(cat_cols) if i not in cat_seen and rx.search(c)]
    if not members:
        continue
    for i, _ in members:
        cat_seen.add(i)
    rows = "".join(f'<tr><td class="mut">{i}</td><td>{esc(c)}</td></tr>' for i, c in members)
    cat_census_html.append(
        f'<details class="census"><summary>{esc(title)} <span class="mut">({len(members)} columns)</span></summary>'
        f'<table class="nums census-t">{rows}</table></details>')
assert len(cat_seen) == len(cat_cols), "categorical census groups must cover every column"
cat_census = "".join(cat_census_html)

# ---------- battle summary ----------
S = C["battle_state_summary"]
summary_rows = "".join(
    f"<tr><td>{esc(k.replace('_', ' '))}</td><td>{esc(json.dumps(v) if isinstance(v, (dict, list)) else v)}</td></tr>"
    for k, v in S.items())

phase_census = " &middot; ".join(f"{esc(k)} &times;{esc(v)}" for k, v in L["turn_merged_phase_census"].items())
dual = M["dual_schema_story"]
dual_rows = "".join(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in dual.items())

FM = M["feature_masks"]
NCB = L["numeric_census_boundaries"]
TM2_NUMERIC_COUNT = sum(1 for c in L["numeric_column_names"] if c.startswith("NUMERIC_TM2_"))

page = f"""<meta charset="utf-8">
<title>PokeZero observation v2.2 — turn-merged token format</title>
<style>
  :root {{
    --bg:#101318; --panel:#171c24; --panel2:#1d232d; --line:#2a323e;
    --tx:#d5dae3; --mut:#8a93a3; --acc:#7ab8d9; --tm:#a795d9; --hi:#d98a99;
  }}
  body {{ background:var(--bg); color:var(--tx); font:15px/1.55 -apple-system, "Segoe UI", sans-serif;
         margin:0; padding:32px 20px 80px; }}
  main {{ max-width:1060px; margin:0 auto; }}
  h1 {{ font-size:26px; margin:0 0 4px; letter-spacing:-0.01em; text-wrap:balance; }}
  h2 {{ font-size:19px; margin:44px 0 10px; color:var(--acc); }}
  h3 {{ font-size:15px; margin:22px 0 8px; }}
  h4 {{ font-size:12px; margin:12px 0 6px; color:var(--mut); text-transform:uppercase; letter-spacing:0.06em; }}
  h5 {{ font-size:11px; margin:8px 0 4px; color:var(--mut); text-transform:uppercase; letter-spacing:0.05em; }}
  p  {{ max-width:76ch; }}
  .mut {{ color:var(--mut); }}
  .prov {{ font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; color:var(--mut); margin:2px 0 26px; }}
  nav {{ background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:12px 18px; font-size:14px; }}
  nav a {{ color:var(--tx); text-decoration:none; margin-right:16px; }}
  nav a:hover {{ color:var(--acc); }}
  a {{ color:var(--acc); }}
  code {{ font:12px ui-monospace, Menlo, monospace; background:var(--panel2); border-radius:3px; padding:0 4px; }}
  .strip {{ display:flex; gap:1px; margin:14px 0 6px; height:34px; }}
  .strip i {{ flex:1 1 0; border-radius:1px; }}
  .lgd {{ margin-right:16px; font-size:13px; }}
  .lgd i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; }}
  .facts {{ display:flex; flex-wrap:wrap; gap:10px; margin:18px 0; }}
  .fact {{ background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:10px 16px; }}
  .fact b {{ display:block; font-size:20px; font-variant-numeric:tabular-nums; }}
  .fact span {{ font-size:12px; color:var(--mut); }}
  .proto {{ background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:12px 14px;
            font:12px/1.6 ui-monospace, SFMono-Regular, Menlo, monospace; overflow-x:auto; }}
  .pl {{ display:block; color:var(--mut); }}
  .pl-turn {{ color:var(--acc); font-weight:600; }}
  .pl-dmg {{ color:var(--tx); }}
  .tokgrid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(min(320px, 100%), 1fr)); gap:10px; margin:14px 0; }}
  .tok {{ background:var(--panel); border:1px solid var(--line); border-left:3px solid; border-radius:6px; padding:10px 12px; overflow-x:auto; }}
  .tok-h {{ display:flex; justify-content:space-between; align-items:baseline; margin-bottom:6px; }}
  .idx {{ font:600 12px ui-monospace, Menlo, monospace; color:var(--tx); }}
  .tok-h .mut {{ font-size:11px; }}
  .chips {{ display:flex; flex-wrap:wrap; gap:4px; margin-bottom:8px; }}
  .chip {{ background:var(--panel2); border:1px solid var(--line); border-radius:10px; padding:1px 8px;
           font:11px ui-monospace, Menlo, monospace; }}
  .chip b {{ color:var(--mut); font-weight:500; }}
  table.nums {{ border-collapse:collapse; width:100%; font:12px ui-monospace, Menlo, monospace;
                font-variant-numeric:tabular-nums; }}
  table.nums td {{ border-top:1px solid var(--line); padding:2px 6px 2px 0; }}
  table.nums td:last-child {{ text-align:right; color:var(--acc); }}
  .df td:last-child {{ color:var(--tx); }}
  .sb {{ border:1px solid var(--line); border-radius:5px; padding:6px 8px; margin:6px 0; overflow-x:auto; }}
  .sb-a {{ border-left:3px solid #86c48e; }}
  .sb-b {{ border-left:3px solid var(--tm); }}
  .sb-shared {{ border-left:3px solid var(--mut); }}
  .ex {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px 18px; margin:14px 0; }}
  .ex-hi {{ border:1px solid var(--hi); box-shadow:0 0 0 1px var(--hi); }}
  .ex-note {{ margin:0 0 10px; font-size:14px; max-width:110ch; }}
  .ex-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .ex-grid > div {{ min-width:0; overflow-x:auto; }}
  @media (max-width:820px) {{ .ex-grid {{ grid-template-columns:1fr; }} }}
  details.census {{ background:var(--panel); border:1px solid var(--line); border-radius:6px;
                    padding:8px 14px; margin:8px 0; overflow-x:auto; }}
  details.census summary {{ cursor:pointer; font-weight:600; }}
  .census-t {{ margin-top:8px; }}
  .census-t td.mut {{ width:44px; }}
  .note {{ border-left:3px solid var(--acc); background:var(--panel); padding:10px 14px; border-radius:0 6px 6px 0;
           font-size:14px; max-width:86ch; margin:12px 0; overflow-wrap:break-word; }}
  .note-tm {{ border-left-color:var(--tm); }}
  table.kv {{ border-collapse:collapse; font-size:13px; width:100%; }}
  table.kv td {{ border-top:1px solid var(--line); padding:4px 10px 4px 0; vertical-align:top; word-break:break-word; }}
  table.kv td:first-child {{ color:var(--mut); white-space:nowrap; width:220px; }}
</style>
<main>
<h1>PokeZero observation v2.2 — the turn-merged token format</h1>
<div class="prov">schema {esc(L["schema_version"])} &middot; {esc(L["token_count"])} tokens &times;
{esc(L["numeric_feature_count"])} numeric + {esc(L["categorical_feature_count"])} categorical columns &middot;
belief source {esc(M["belief_set_source"]["source_hash"])} &middot;
example: the seed-148 explosion fixture, turn-16 boundary, p1 perspective</div>

<nav>
  <a href="#shape">1 &middot; Shape</a><a href="#game">2 &middot; The game at turn 16</a>
  <a href="#anatomy">3 &middot; Turn-merged anatomy</a><a href="#explosion">4 &middot; Explosion walkthrough</a>
  <a href="#walk">5 &middot; Token walkthrough</a><a href="#lines">6 &middot; Line &rarr; token</a>
  <a href="#census">7 &middot; All {esc(L["numeric_feature_count"])}+{esc(L["categorical_feature_count"])} columns</a>
  <a href="#masks">8 &middot; Masks &amp; schemas</a>
</nav>

<h2 id="shape">1 &middot; The shape at a glance</h2>
<p>Every observation is a fixed grid of <b>{esc(L["token_count"])} tokens</b>. Each token has the same
{esc(L["numeric_feature_count"])} named numeric columns (most zero for any given token kind) plus
{esc(L["categorical_feature_count"])} categorical slots: {esc(CAT_LAYOUT["fixed_count"])} fixed, bucketed belief slots
({esc(CAT_LAYOUT["belief_ability_buckets"])} ability + {esc(CAT_LAYOUT["belief_item_buckets"])} item +
{esc(CAT_LAYOUT["belief_move_buckets"])} move candidates, {esc(CAT_LAYOUT["volatile_buckets"])} volatiles), and the
{esc(CAT_LAYOUT["turn_merged_extra"])} v2.2 turn-merged columns. The v2.2 change is confined to the transition
section: instead of one token per <i>declared action</i>, the block carries <b>one token per
turn/lead/replacement phase with two ordered sub-blocks</b> — speed order becomes explicit structure,
and a consumed-but-never-executed declaration becomes representable (<code>tt2_status:negated</code>).</p>
<div class="strip">{strip}</div>
<div>{legend}</div>
<p class="mut" style="font-size:13px">Each cell is one token slot at this turn-16 boundary; dimmed cells are
zero-padded and attention-masked ({esc(L["attention_mask_at_boundary"]["attended_total"])} of {esc(L["token_count"])} attended here —
{esc(L["attention_mask_at_boundary"]["transition_attended"])} of {esc(L["sections"]["transition"]["count"])} transition slots
populated after 15 completed turns: {phase_census}).</p>
<div class="facts">
  <div class="fact"><b>{esc(L["token_count"])}</b><span>tokens per observation</span></div>
  <div class="fact"><b>{esc(L["numeric_feature_count"])}</b><span>numeric columns per token ({esc(NCB["v2"])} v2 &rarr; {esc(NCB["v2.1"])} v2.1 &rarr; {esc(NCB["v2.2"])} v2.2)</span></div>
  <div class="fact"><b>{esc(L["categorical_feature_count"])}</b><span>categorical columns (39 + 12 turn-merged)</span></div>
  <div class="fact"><b>1/turn</b><span>transition tokens (phases; was 2/turn declared actions)</span></div>
  <div class="fact"><b>K in TURN units</b><span>budget counts whole-turn rows: 18 vs 35 per-action here</span></div>
  <div class="fact"><b>oldest-first</b><span>truncation when the stream exceeds K</span></div>
</div>
<div class="note note-tm"><b>K budget unit change (loud):</b> {esc(L["k_budget_unit_note"])}</div>

<h2 id="game">2 &middot; The game at turn 16</h2>
<p>Fifteen turns of switch-heavy randbats: Weezing's turn-7 Explosion crit KOs our freshly-switched Gligar
and faints itself (the double cold replacement), Cloyster lands Toxic on turn 12 and Spikes on turn 13,
and on turn 15 our badly-poisoned Pidgeot eats its Liechi Berry in the residual phase and faints to
poison, sending Regirock in over the Spikes. At the boundary our Regirock (227/259) faces Piloswine
(156/316, Choice Band still only a belief candidate) with all six opponents revealed by the
switch-heavy play — a fully-attended opponent section.</p>
<table class="kv">{summary_rows}</table>
<h3>Protocol, start &rarr; the turn-16 boundary <span class="mut">({esc(C["elided_request_lines"])} |request| lines elided)</span></h3>
{proto(C["protocol_lines_start_to_boundary"])}

<h2 id="anatomy">3 &middot; Turn-merged token anatomy</h2>
<p>A v2.2 transition row has three column families, rendered throughout this doc as the shared box plus
sub-blocks <b>A</b> and <b>B</b>:</p>
<div class="note note-tm">
<b>Shared (once per token):</b> <code>CATEGORY_SLOT</code> is re-purposed to <code>tt_phase:&lt;turn|lead|replacement|extra&gt;</code>;
the context trio (<code>NUMERIC_TT_OWN_SPIKES</code>/<code>OPP_SPIKES</code>, weather on <code>CATEGORY_MOVE_EFFECT</code>) is captured at the
FIRST sub-block's declaration; the positional pair <code>NUMERIC_TT_ABS_TURN</code>/<code>TT_TURNS_AGO</code> is stored once.<br><br>
<b>Sub-block A (first mover)</b> rides the existing per-action columns unchanged: actor species in
<code>CATEGORY_PRIMARY</code>, action label in <code>CATEGORY_SECONDARY</code>, outcome/effectiveness/side-effect on
<code>TYPE_1</code>/<code>TYPE_2</code>/<code>MOVE_CATEGORY</code>, defender identity in <code>CATEGORY_MOVE_PRIORITY</code> (exactly the v2.1
per-action semantics), numerics on <code>NUMERIC_TT_*</code>. The per-action <code>tt_kind</code> moves to the appended
<code>CATEGORY_TM_FIRST_KIND</code> because SLOT now holds the phase; RestTalk/Baton-Pass collapses ride
<code>CATEGORY_TM_FIRST_CANT</code>/<code>_BP</code>.<br><br>
<b>Sub-block B (second mover)</b> lives entirely on the appended columns: 9 categorical
(<code>CATEGORY_TM_SECOND_*</code>) + {TM2_NUMERIC_COUNT} numeric (<code>NUMERIC_TM2_*</code>). Its labels use <code>tt2_</code>-prefixed vocabulary
families so the unordered categorical bag stays sub-block-bound. <code>NUMERIC_TM2_PRESENT</code> is the
second-half-is-an-executed-action bit.</div>
<h3>PENDING / NEGATED / ABSENT — the consumption-proof rule</h3>
<p>When the second half of a token is not an executed action, <code>CATEGORY_TM_SECOND_KIND</code> carries a status
label instead of a kind, all TM2 numerics stay 0.0, and <code>CATEGORY_TM_SECOND_SPECIES</code> still names the mon
whose declaration is being described (when known):</p>
<div class="note">
<b><code>tt2_status:negated</code></b> — the side DECLARED an action and the engine provably consumed it with zero
protocol trace. Engine-verified: when a mon faints mid-turn before its opponent's declared action
executes (hazard sack, faster Explosion KO), that action emits NO <code>|move|</code> line at all — even
non-targeted moves like a declared Spikes layer fizzle. Hazard-sacking is a true free pivot, and the
negated sub-block is what makes it learnable.<br><br>
<b><code>tt2_status:pending</code></b> — the turn is still OPEN with <i>no consumption proof</i>: a replay prefix cut at
a mid-turn forceSwitch boundary (e.g. awaiting the Baton Pass completion choice). NEGATED is only
stamped on proof of consumption — the turn closed, or a mid-turn faint occurred (which the engine turns
into a full cancel of every remaining action); encoding an open turn as negated would assert the
free-pivot semantics exactly where they are false.<br><br>
<b><code>tt2_status:absent</code></b> — no declaration was ever expected: the empty half of a single replacement
token (see the Regirock example in section 6).<br><br>
This game contains {esc(len(D["negated_or_pending_sub_blocks_in_game"]))} NEGATED/PENDING sub-blocks before the boundary
(every declared action executed), so section 6 demonstrates the semantics on the ABSENT half and the
Explosion walkthrough explains where a negation WOULD have appeared.</div>
<h3>First-mover-context recoverability</h3>
<p>The context trio is stored once per merged token, captured at the first sub-block's declaration. A
second sub-block therefore reconstructs with the first mover's trio — recoverable exactly, except when
the first mover's own action changed the trio (hazard set/clear or weather set, including ability
weather on a switch-in). That is precisely the <code>side_effect &isin; {{hazard-set, hazard-clear, weather-set}}</code>
allowance the corpus bijection gate (<code>tests/test_turn_merged_corpus.py</code>) tolerates;
<code>flatten_turn_merged_tokens</code> reconstructs the per-action stream field-exactly everywhere else, and the
RestTalk click it resynthesizes is protocol-constant by construction (the merger verifies default
fields before collapsing).</p>

<h2 id="explosion">4 &middot; The Explosion double-replacement walkthrough</h2>
<p>The fixture's namesake event, and the shape the merged schema was designed around: after a
simultaneous double-faint, BOTH sides replace blind in ONE engine forceSwitch cycle — the two
<code>|switch|</code> lines are emitted back-to-back in arbitrary order, and neither player saw the other's
choice. That phase is ONE merged pair token with two switch sub-blocks (<code>tt_phase:replacement</code>,
both halves ACTION), the same cold-pair shape as the lead token. Sequential faints (move KO now,
residual faint later the same turn) are separate engine request cycles and stay two single tokens —
the cold-pair signal is &ldquo;was the OTHER side also waiting on a replacement when this switch-in was
emitted&rdquo;, not log adjacency.</p>
<p>This is also the terminal case of the <b>per-sub-block self-cost column</b>: Explosion spends the
user's whole HP, so Weezing's sub-block carries <code>NUMERIC_TM2_SELF_HP_COST = 1.0</code> — which is why one
<code>tt2_ko</code> bit accounts for two faints. The graded end of the same scale (Substitute's exact 0.25,
Double-Edge recoil at 0.073 and 0.123) is worked through in section 6's self-cost anatomy example.</p>
{explosion_html}

<h2 id="walk">5 &middot; Token walkthrough — every attended token</h2>

<h3 style="color:{HUE['field']}">Field token (index 0)</h3>
<p>Global state: whose request this is, weather, hazard/screen layers per side
(<code>NUMERIC_SELF_HAZARDS = 0.333</code> — Cloyster's one Spikes layer / 3), absolute turn (/1000).</p>
<div class="tokgrid">{section_tokens("field")}</div>

<h3 style="color:{HUE['self_pokemon']}">Self-mon tokens (indices 1&ndash;6)</h3>
<p>Our six, fully known: exact HP fractions, actual stats, status (Pidgeot fainted with its tox story),
exact-state counters, and — v2.1 blocks carried forward — the active mon's substitute HP fraction
(<code>NUMERIC_SUB_HP_FRACTION</code>, 0 here; Pidgeot's earlier subs were on prior stints). Self uncertainty is
0 — our own side carries no belief.</p>
<div class="tokgrid">{section_tokens("self_pokemon")}</div>

<h3 style="color:{HUE['opponent_pokemon']}">Opponent-mon tokens (indices 7&ndash;12)</h3>
<p>Only revealed mons are attended — all 6 of 6 here (Piloswine, Jumpluff, Weezing&dagger;, Mew, Illumise,
Cloyster), a rarity this early. Each carries the belief layer: candidate-set
size/uncertainty, candidate abilities/items/moves as categorical buckets, the exact PP ledger
(<code>NUMERIC_OPP_MOVE_PP_OFFSET+k</code>), and the v2.1 <b>PP-validity bits</b>
(<code>NUMERIC_OPP_MOVE_PP_VALID_OFFSET+k</code>): bit k = 1 iff bucket k's move is protocol-revealed, regardless
of remaining PP — closing the v2 revealed-at-0-PP collision. Expected stats come from the
variant-conditioned spread. The <b>per-mon pinned Tier-2 surface</b> also lives here:
<code>NUMERIC_TIER2_CB_PINNED</code> (the authoritative current-state Choice Band conclusion, derived from the
FULL untruncated per-action stream so it survives K-truncation) and its defender-side twin
<code>NUMERIC_TIER2_INVESTMENT_PINNED</code> — both 0.0 on this path (section 8).</p>
<div class="tokgrid">{section_tokens("opponent_pokemon")}</div>

<h3 style="color:{HUE['action_candidate']}">Action-candidate tokens (indices 13&ndash;21)</h3>
<p>The 9 legal-action slots (4 moves + 5 switches at a move request); <code>NUMERIC_LEGAL</code> masks what's
currently playable (Pidgeot's slot is dead), and each carries its move/species identity so the policy
head scores real objects, not indices.</p>
<div class="tokgrid">{section_tokens("action_candidate")}</div>

<h3 style="color:{HUE['stats']}">Stats token (index 22)</h3>
<p>The unbounded-memory channel: whole-game tendency aggregates as (count, opportunity) rates — where
behaviors older than the K-window survive truncation. Unchanged by v2.2 (it aggregates the per-action
fold, which the merged stream shares).</p>
<div class="tokgrid">{section_tokens("stats")}</div>

<h3 style="color:{HUE['transition']}">Turn-merged transition tokens (indices 23&ndash;40 populated)</h3>
<p>One token per phase, oldest first: the lead pair, fifteen <code>tt_phase:turn</code> tokens, and two
replacement phases (the turn-7 cold PAIR and the turn-15 single with its ABSENT half). Each card below
shows the shared context, sub-block A (first mover — explicit speed order), and sub-block B. Defender
identity rides <code>CATEGORY_MOVE_PRIORITY</code> (A) / <code>CATEGORY_TM_SECOND_DEFENDER</code> (B) on move sub-blocks.
The Tier-2 residual/validity/CB/investment slots exist in BOTH sub-blocks (columns 117&ndash;120 and
148&ndash;152) — all zero on this replay path.</p>
<div class="tokgrid">{section_tokens("transition")}</div>

<h2 id="lines">6 &middot; From protocol line to token — worked examples</h2>
{examples_html}

<h2 id="census">7 &middot; All {esc(L["numeric_feature_count"])} numeric + {esc(L["categorical_feature_count"])} categorical columns</h2>
<p>Grouped by layer, with the schema each group was born in. Any column can appear on any token, but in
practice each group lives on the token kinds noted in section 5. Census boundaries:
v2 ends at column {esc(L["numeric_census_boundaries"]["v2"] - 1)}, v2.1 at {esc(L["numeric_census_boundaries"]["v2.1"] - 1)},
v2.2 at {esc(L["numeric_census_boundaries"]["v2.2"] - 1)}.</p>
{census}
<h3>Categorical columns</h3>
{cat_census}

<h2 id="masks">8 &middot; Masks, gates, provenance — and the dual-schema story</h2>
<div class="note">
<b>Feature masks</b> (recorded in every checkpoint's <code>model_config</code>, latched back into every eval
encode): stats_block={esc(FM["stats_block"])}, exact_state={esc(FM["exact_state"])},
transition_token_budget={esc(FM["transition_token_budget"])} (TURN units under v2.2),
tier2_residuals={esc(FM["tier2_residuals"])}, tier2_investment={esc(FM["tier2_investment"])} (flipped ON for
this dump to document the full surface; defaults False).<br><br>
<b>Tier-2 gates</b>: {esc(M["live_feature_blocks"]["tier2_note"])}<br><br>
<b>Belief provenance</b>: candidate-set source hash <code>{esc(M["belief_set_source"]["source_hash"])}</code> — stamped in
cache metadata and checkpoints; eval paths refuse or warn on mismatch.
{esc(M["belief_set_source"]["note"])}.<br><br>
<b>Extraction</b>: {esc(D["extraction"]["replay_loop"])}. {esc(D["extraction"]["boundary_request_provenance"])}.
Zero category-vocab OOV; vocabulary built with <code>include_turn_merged=True</code> (a v2.2 encode REFUSES a
base vocabulary — the tt_phase/tt2_* families must be enumerated, never hashed into the OOV band).
</div>
<h3>The dual-schema table</h3>
<table class="kv">{dual_rows}</table>
<p class="mut" style="font-size:13px">Self-validation: <code>tests/test_token_format_doc.py</code> regenerates the
dump behind this page from the committed fixture and asserts byte-identity with
<code>turn16-token-dump.json</code>, then checks sentinel values in this HTML — if the encoder changes, the test
fails and this doc must be regenerated.</p>
</main>
"""

out = HERE / "token-format-v2_2.html"
out.write_text(page, encoding="utf-8")
print(f"wrote {out} ({len(page)} bytes)")
