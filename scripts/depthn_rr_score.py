"""RR (realized-reply re-pricing): score every searched DN2W16 decision against Foul Play's actual reply. CPU only.

Inputs: the re-pricer's rows (scripts/depthn_rr_reprice.py; one JSON line per decision sent to the server, with its 16
re-priced depth-2 worlds) and the live run directory (results/<ordinal>.json: cell.seed and cell.arm;
telemetry/<ordinal>.jsonl: search.policy_target by decision_index; owning-wire/<ordinal>.jsonl: the owner's capture).
Output: --out, a new directory holding summary.json, summary.md and decisions.jsonl. Needs numpy and scipy (bayes_hedge's
HiGHS LP and the t quantile); importing gpu_engine.search_hedge and pokezero.v5_gumbel pulls in no JAX.

Foul Play's reply b_FP (proxy (a), the lead's rules), from the owner's capture:
- The decision's request is the owner's request_count-th |request| line (phase_d_owning_capture.read_capture's alignment;
  its ROOT rows cross-check decision_index -> request_count, and a mismatch censors the decision as 'other').
- Window: the lines after that request through the end of the turn (the next |turn| line, exclusive, or the end of the
  capture). Showdown sends each turn's log BEFORE the next requests, so these are the lines that resolve the decision.
  The window runs past an owner request inside the turn (our own Baton Pass replacement, a wait request) so a slower Foul
  Play's move after it is still seen; Foul Play's faint replacements always follow its faint, which decides first.
- Forced-switch decision (the own request has forceSwitch; else request_kind 'switch'): Foul Play's |switch| before the
  window's first |move| line, next owner |request| line or Foul Play |faint| is its reply (both sides replacing);
  otherwise Foul Play was not asked: 'wait' (a later replacement after its own end-of-turn faint is not this round's).
- Otherwise the first decisive Foul Play event in the window:
  its first |move| line (move id, lowercase alphanumerics, Hidden Power collapsed; Sleep Talk/Metronome callers come
  before the called move's [from] line, and locked turns' [from]lockedmove lines are the chosen move);
  its |switch| before any move line: voluntary switch (species from the details, as an id); after a move line: censored
  'other'; |drag| is never a choice;
  |-activate|<fp>a: ...|move: Pursuit (our Pursuit hitting its switch first): censored 'other';
  its |faint| before it acted: censored 'ko_first', or the earlier |cant| reason's cause.
  A |cant| naming Foul Play's own blocked move (|cant|<fp>a: X|move: Taunt|Thunder Wave, |Disable|Body Slam,
  |move: Imprison|Body Slam, |Focus Punch|Focus Punch) shows its choice: that move is the reply. An [of] |cant| (Damp's
  holder naming the other side's move) names none. Any other |cant| waits for a later move line (Sleep Talk follows
  |cant|...|slp|); without one the decision is censored: slp, par, flinch, frz -> 'sleep_para_flinch'; anything else
  (recharge, Truant, Attract, no Foul Play event at all, a missing or invalid capture) -> 'other'.
Mapping b_FP into world w (row world facts): a move to the legal lane b in 0-3 that plays it as the re-pricer's
lane_labels reads the encoder: lane 0 is Struggle when no move is usable (peer.usable) and the charge move while one
charges, any other lane active_moves[move_targets[b]]. Struggle maps only in a world where its Foul Play must Struggle,
and nothing else maps there; the lane's recorded label (peer.lanes) must agree. A switch to the lane b in 4-8 with switch_targets[b-4] == the
unfainted party slot of that species and legal[b]; 'wait' to column 0 of a world whose opponent was not requested. A
world is also dropped when that column is unpriced for any legal action of ours, so every action's R uses the same
worlds. R(a) = the uniform mean over the mapped worlds of G_w[a][b_w]. 0 mapped worlds: 'unmapped' (not censored).

Scorings: dropped (censored and unmapped decisions excluded) and pi_filled (those decisions use averaging's own
q(a) = sum_w 1/16 sum_b pi_w(b) G_w[a][b]). A decision is scored when its replay is COMPLETE, every world is priced for
every legal action against every kept reply, and the live greedy action is legal. Operators on the same grids: greedy
(live), averaging, bayes_hedge lam 0 and 0.5, prior_regularized c, noise_gated k, world_lcb kappa, target_argmax (the
live policy_target; argmax of root_logits + q, the same target, when it is missing), tempered T (2, 4, inf: averaging with
pi_w^T = softmax(root_peer_logits / T) over the kept lanes, T = inf uniform; a waiting world keeps column 0 at mass 1).
D = R(op) - R(greedy), 0 when op is
greedy. noise_gated plays averaging's a* only when a* != greedy and mean_w gap > k sd_w(gap) / sqrt(W), gap_w =
Q_w(a*) - Q_w(greedy); world_lcb plays argmax q(a) - kappa sd_w(Q_w(a)) / sqrt(W) (sample sd; Q_w(a) = sum_b pi_w(b) G_w[a][b]).
noise_gated and per_world_q are pokezero.depthn_root_operator's (the live NOISE_GATED_K2 plays noise_gated with k 2 and, the
search not knowing the live greedy, greedy = the argmax of root_logits: greedy_argmax here).
Guards (the live actor's, through pokezero.depthn_effect_guard's own functions): every operator but greedy plays greedy where
the own-recharge guard trips on its choice (an override INTO a recharge move greedy did not choose: root.own_moves) or
UNREPRESENTED_EFFECT does (the row's effect_guard record: (b) at the root, or (c) on that operator's own choice); a root the
replay refused OPPONENT_RECHARGE_UNREPRESENTED has no priced worlds (greedy plays, never scored; counted). Every table above
is under the guards. The guard section (summary.json guard, summary.md "Effect guard") reports the (b) share of priced roots
per tier and for the union (turn-1 and turn-2-added parts apart), (c) per operator, the primary result (RR on unguarded roots,
(b) false), RR on the guarded roots with the raw choices for context, a tier-B-legal split of the unguarded roots, the
sensitivity rows ((a) holds, neither (b) nor any operator's (c)), and above_20pct when the guarded share exceeds 20%.
Live rule (screen-r2): a record of rule 'reach' trips on turn-1 (b), the reach M (below) >= 5% or (c); the primary, gate and
selection follow each record's own live rule (an older record's: the any-cell turn-2 arm). The any-cell scope is reported
only (summary guard.any_cell, summary.md "Any-cell scope": its share, RR on its unguarded roots, a cross-tab against live).
Guard reach (report only; the lead's narrower scope): per row M_s = sum_w p_w sum_(a,b) pi_own(a) pi_peer,w(b) q_(w,ab),s
(pokezero.depthn_effect_guard.reach over the row's effect_guard masses, root prior and worlds), exact when the record's turn-2
masses carry the guard's own `guarded` flag, else lower (tier A, and Psycho Boost under a revealed White Herb holder) and upper
(every tier-C mass) bounds; the
reach scope at each bound is turn-1 (b) or a turn-2 M >= 5%, (c) per operator unchanged. decisions.jsonl gains reach_turn2
{m_own_lower, m_own_upper, m_opp_lower, m_opp_upper, tripped (None: the bounds straddle 5%), exact} and
reach_scope_tripped_lower / _upper; summary guard.reach and summary.md "Guard reach" report each bound's share (union, turn 1
and reach-turn-2-added, per tier, per side), the ambiguous rows, RR on each bound's unguarded roots and a cross-tab against the
any-cell scope.
Statistics: D is summed per battle (ordinal); mean and SE over battles; a t interval with battles - 1 df at 90% for the
mean per-battle sum. The per-decision mean D is sum D / decisions, and its interval is the per-battle one divided by
the mean decisions per battle (the same t statistic, so the same signs). Parity, over every row searched live (source 'search'):
the replay is COMPLETE, averaging recomputed from the captured worlds plays the live search action, and its q equals the
replay's q_values exactly after a JSON round trip. Below 99% the verdict is PARITY_FAILED: no candidate is selected, as for
GATE_FAILED. --parity-mode replay (a replay on a modified tree, e.g. a fixed server): every replay-COMPLETE row is held to
the replay's own action and q_values instead of the live action. Also reported, never gating: every table restricted to decisions where Foul Play was requested (no 'wait').
External reply prior (--reply-prior, for a Foul Play reply model): rows {ordinal, decision_index, world, logits[9]} add
external_br (averaging with pi_w = softmax of the external logits over the kept lanes) and external_mix alpha (0.25, 0.5,
0.75: alpha * external + (1 - alpha) * the champion's pi_w), reported only, never candidates. A waiting world needs no
logits; a decision missing any requested world's logits is excluded from those operators and counted.

Run from the repository root with the venv that has numpy and scipy:
  python scripts/depthn_rr_score.py --rows ROWS/rows-*.jsonl --run-directory RUN --out NEW_DIRECTORY
"""
import argparse
import collections
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import statistics
import sys

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT), str(_ROOT / 'src')):
    if _path not in sys.path:sys.path.insert(0, _path)

from gpu_engine import search_hedge as hedge  # noqa: E402
from pokezero import v5_gumbel as original  # noqa: E402
from pokezero.phase_d_owning_capture import read_capture  # noqa: E402
from pokezero import depthn_effect_guard as effect_guard  # noqa: E402
# argmax (lowest index on ties), target_argmax, per_world_q and noise_gated: the live search's TARGET_ARGMAX and NOISE_GATED_K2
# play these very functions (one definition)
from pokezero.depthn_root_operator import argmax, noise_gated, per_world_q, target_argmax  # noqa: E402,F401

SCHEMA = 'pokezero.depthn-rr-score.v1'
LEVEL = .90
OVERRIDE_MIN = .05
PARITY_MIN = .99
PARITY_MODES = ('live', 'replay')   # live: against the live search (the default); replay: the replay against itself (a modified tree)
PRIOR_C = (.02, .05, .1, .2, .5)
NOISE_K = (1, 2)
LCB_KAPPA = (.5, 1, 2)
BANDS = (('<1%', .01), ('1-5%', .05), ('5-20%', .2), ('20-50%', .5), ('>=50%', math.inf))
CAUSES = ('ko_first', 'sleep_para_flinch', 'other')
SLEEP_PARA_FLINCH = frozenset(('slp', 'par', 'flinch', 'frz'))
SCORINGS = ('dropped', 'pi_filled')
HALVES = ('all', 'even', 'odd')
TEMPER_T = (2, 4, math.inf)
EXTERNAL_ALPHA = (.25, .5, .75)
SETTINGS = (('greedy', 'averaging', 'bayes_hedge:lam=0', 'blend:lam=0.5') + tuple(f'prior_regularized:c={c}' for c in PRIOR_C)
            + tuple(f'noise_gated:k={k}' for k in NOISE_K) + tuple(f'world_lcb:kappa={k}' for k in LCB_KAPPA) + ('target_argmax',)
            + tuple(f'tempered:T={t}' for t in TEMPER_T))
CANDIDATES = SETTINGS[2:]   # selection excludes greedy and averaging
# With --reply-prior only, and reported only (never candidates, so the gate, selection and confirmation do not depend on it).
EXTERNAL_SETTINGS = ('external_br',) + tuple(f'external_mix:alpha={a}' for a in EXTERNAL_ALPHA)
REPLY_PRIOR_FIELDS = dict(ordinal='ordinal', decision_index='decision_index', world='world', logits='logits')
WAITING = (1.,) + (0.,) * 8   # a waiting opponent: column 0 at mass 1
GUARD_SHARE_FLAG = .20   # the summary flags a guarded share of priced roots above this
OWN_RECHARGE, UNREPRESENTED = 'OWN_RECHARGE_UNREPRESENTED', effect_guard.CAUSE

# ---- the re-pricer's field names (scripts/depthn_rr_reprice.py, row schema pokezero.depthn-rr-reprice.row.v1 at f9e143ae),
# in one place: a rename is a one-line change here ----
ROW_FIELDS = dict(
    ordinal=('ordinal',), battle_id=('battle_id',), owner=('owner',), decision_index=('decision_index',),
    request_count=('request_count',),
    source=('live', 'source'), live_action=('live', 'search_action_index'), live_q=('live', 'search_q_values'),
    greedy_action=('live', 'greedy_action_index'), request_kind=('live', 'request_kind'), turn=('live', 'turn'),
    played_action=('live', 'played_action_index'),
    live_cause=('live', 'cause'), replay_status=('replay', 'status'), replay_cause=('replay', 'cause'),
    replay_action=('replay', 'action_index'), replay_q=('replay', 'q_values'),
    action_equal=('parity', 'action_equal'), q_equal=('parity', 'q_equal'),
    root_logits=('root', 'root_logits'), root_legal=('root', 'root_legal'), worlds=('worlds',),
    own_moves=('root', 'own_moves'), refusal_cause=('root', 'refusal_cause'), effect_guard=('effect_guard',))
WORLD_FIELDS = dict(
    requested=('opponent_requested',), kept=('kept',), logits=('root_peer_logits',), grid=('grid',), refusals=('refusals',),
    active_slot=('opponent', 'active_slot'), active_species=('opponent', 'active_species'), active_moves=('opponent', 'active_moves'),
    charge_move=('opponent', 'charge_move'), party=('opponent', 'party'),
    move_targets=('peer', 'move_targets'), switch_targets=('peer', 'switch_targets'), legal=('peer', 'legal'),
    usable=('peer', 'usable'), lanes=('peer', 'lanes'))
PARTY_FIELDS = dict(slot='slot', species='species', fainted='fainted')   # one party member (a dict per slot)


def _field(value, path):
    for key in path:
        if type(value) is not dict:return None
        value = value.get(key)
    return value


def _world(w):
    out = {k: _field(w, p) for k, p in WORLD_FIELDS.items()};party = out.pop('party')
    out['party_species'] = out['fainted'] = None
    if type(party) is list and all(type(m) is dict for m in party):
        slots = {m.get(PARTY_FIELDS['slot']) if type(m.get(PARTY_FIELDS['slot'])) is int else i: m for i, m in enumerate(party)}
        members = [slots.get(s) or {} for s in range(max(slots) + 1)] if slots else []
        out['party_species'] = [m.get(PARTY_FIELDS['species']) for m in members]
        out['fainted'] = [m.get(PARTY_FIELDS['fainted']) for m in members]
    return out


def normalize(row):
    """One re-pricer row in this scorer's names (ROW_FIELDS / WORLD_FIELDS); a missing field is None."""
    out = {k: _field(row, p) for k, p in ROW_FIELDS.items()}
    worlds = out['worlds']
    out['worlds'] = None if type(worlds) is not list else [None if type(w) is not dict else _world(w) for w in worlds]
    return out


def load_rows(paths):
    """The only reader of re-pricer rows: each JSONL row of `paths`, normalized."""
    for path in paths:
        with open(path, 'rb') as stream:
            for raw in stream:
                if raw.strip():yield normalize(json.loads(raw))


# ---- Foul Play's reply from the owner's capture ----

@dataclass(frozen=True)
class Reply:
    kind: str | None     # 'move', 'switch' or 'wait'; None when censored
    id: str | None       # the move or species Showdown id
    cause: str | None    # censored: one of CAUSES
    detail: str          # what decided it (audit)


def censored(cause, detail):return Reply(None, None, cause, detail)


def showdown_id(name):
    """Lowercase alphanumerics; every Hidden Power variant is 'hiddenpower'."""
    value = re.sub('[^a-z0-9]', '', str(name).lower())
    return 'hiddenpower' if value.startswith('hiddenpower') else value


def _who(ident):return ident.split(':', 1)[0].strip()


def _cant_cause(reason):return 'sleep_para_flinch' if reason in SLEEP_PARA_FLINCH else 'other'


def turn_window(lines, request_count):
    """(the owner's request_count-th |request| line, the lines after it up to the next |turn| line), or None."""
    seen = 0
    for i, line in enumerate(lines):
        if line.startswith('|request|'):
            seen += 1
            if seen == request_count:
                window = []
                for later in lines[i + 1:]:
                    if later.startswith('|turn|'):break
                    window.append(later)
                return line, window
    return None


def forced_request(request_line):
    """True when the own request is a forceSwitch, False when not, None when it cannot be read."""
    try:request = json.loads(request_line[len('|request|'):])
    except ValueError:return None
    if type(request) is not dict:return None
    force = request.get('forceSwitch')
    return bool(force) and any(force)


def cant_named_move(parts):
    """The move a |cant| line names as the Pokemon's own blocked choice (|cant|X|move: Taunt|Thunder Wave, |cant|X|Disable|Body
    Slam, |cant|X|move: Imprison|Body Slam, |cant|X|Focus Punch|Focus Punch), else None. An [of] line (Damp: the holder's
    |cant| names the OTHER side's move) and a reason without a move (slp, par, flinch, frz, recharge, Truant) name none."""
    rest = parts[3:]
    if len(rest) < 2 or not rest[1] or rest[1].startswith('[') or any(p.startswith('[of]') for p in rest):return None
    return showdown_id(rest[1])


def fp_reply(window, owner, forced):
    """Foul Play's reply to one decision from its turn window (module docstring)."""
    fp = ('p2' if owner == 'p1' else 'p1') + 'a'
    if forced:
        # Both sides replace at once, before anything else; a move, a later owner request or Foul Play's own faint (its
        # end-of-turn replacement follows) ends this decision's round.
        for line in window:
            parts = line.split('|')
            if len(parts) < 3:continue
            if parts[1] in ('move', 'request') or parts[1] == 'faint' and _who(parts[2]) == fp:break
            if parts[1] == 'switch' and _who(parts[2]) == fp:return Reply('switch', showdown_id(parts[3].split(',')[0]), None, 'replacement_switch')
        return Reply('wait', None, None, 'not_requested')
    moved, cant = False, None
    for line in window:
        parts = line.split('|')
        if len(parts) < 3:continue
        kind, who = parts[1], _who(parts[2])
        if kind == 'move':
            if who == fp:return Reply('move', showdown_id(parts[3]), None, 'after_cant:' + cant if cant is not None else 'move')
            moved = True
        elif who != fp:continue
        elif kind == 'switch':
            if moved:return censored('other', 'switch_after_move')
            return Reply('switch', showdown_id(parts[3].split(',')[0]), None, 'voluntary_switch')
        elif kind == '-activate' and len(parts) > 3 and parts[3] == 'move: Pursuit':return censored('other', 'pursuit_on_switch')
        elif kind == 'cant':
            if any(p.startswith('[of]') for p in parts[3:]):continue   # its ability blocking OUR move (Damp): not its own action
            named = cant_named_move(parts)
            if named is not None:return Reply('move', named, None, 'cant_named:' + parts[3])
            if cant is None:cant = '|'.join(parts[3:])
        elif kind == 'faint':
            if cant is not None:return censored(_cant_cause(cant), 'cant:' + cant)
            return censored('ko_first', 'faint_before_acting')
    if cant is not None:return censored(_cant_cause(cant), 'cant:' + cant)
    return censored('other', 'no_fp_action')


@dataclass(frozen=True)
class Capture:
    lines: tuple | None
    roots: dict          # decision_index -> request_count, from read_capture's ROOT rows
    owner: str | None
    error: str | None


@lru_cache(maxsize=4)
def load_capture(path):
    """read_capture validates the chain and gives the ROOT alignment; the durable WIRE rows give every line, including
    those after the last root."""
    path = Path(path)
    if not path.exists():return Capture(None, {}, None, 'capture_missing')
    try:
        roots, owner = {}, None
        for root in read_capture(path):roots[root['decision_index']] = root['request_count'];owner = root['owner']
    except (ValueError, KeyError, TypeError):return Capture(None, {}, None, 'capture_invalid')
    lines = []
    for raw in path.read_bytes().split(b'\n')[:-1]:   # an unterminated tail is never durable (read_capture's rule)
        row = json.loads(raw)
        if row.get('event') == 'BEGIN':owner = row.get('owner')
        elif row.get('event') == 'WIRE':lines.extend(row['lines'])
    return Capture(tuple(lines), roots, owner, None)


def reply_for(capture, decision):
    if capture.error is not None:return censored('other', capture.error)
    if capture.owner != decision['owner']:return censored('other', 'owner_mismatch')
    expected = capture.roots.get(decision['decision_index'])
    if expected is not None and expected != decision['request_count']:return censored('other', 'request_misaligned')
    found = turn_window(capture.lines, decision['request_count']) if type(decision['request_count']) is int else None
    if found is None:return censored('other', 'request_missing')
    line, window = found
    forced = forced_request(line)
    if forced is None:forced = decision['request_kind'] == 'switch'
    return fp_reply(window, decision['owner'], forced)


# ---- mapping b_FP into each world ----

def move_lanes(facts, legal):
    """{lane: move id} over the opponent's legal lanes 0-3, as the re-pricer's lane_labels reads its encoder: lane 0 is
    Struggle when no move is usable (peer.usable) and the charge move while one charges; any other lane is
    active_moves[move_targets[b]]. None without the facts."""
    moves, targets, usable = facts.get('active_moves'), facts.get('move_targets'), facts.get('usable')
    if type(moves) is not list or type(targets) is not list or type(usable) is not list or not usable:return None
    struggle, charge, out = not any(usable), facts.get('charge_move'), {}
    for b in range(4):
        if not legal[b]:continue
        if b == 0 and struggle:out[b] = 'struggle'
        elif b == 0 and charge is not None:out[b] = charge
        elif b < len(targets) and type(targets[b]) is int and 0 <= targets[b] < len(moves):out[b] = moves[targets[b]]
    return out


def map_world(reply, facts, world, root_legal):
    """(lane, None) or (None, why this world is dropped)."""
    if reply.kind == 'wait':
        if world.opponent_requested:return None, 'wait_but_requested'
        lane = 0
    else:
        if not world.opponent_requested:return None, 'reply_but_waiting'
        legal = facts.get('legal')
        if type(legal) is not list or len(legal) != 9:return None, 'facts_missing'
        if reply.kind == 'move':
            lanes = move_lanes(facts, legal)
            if lanes is None:return None, 'facts_missing'
            struggle = not any(facts['usable'])
            if struggle and reply.id != 'struggle':return None, 'struggle_world'   # this world's Foul Play can only Struggle
            found = [b for b, m in sorted(lanes.items()) if m == reply.id]
            if not found:
                if reply.id == 'struggle':return None, 'struggle_not_forced'   # a usable move here: lane 0 is a real slot
                return None, 'move_missing' if reply.id not in facts['active_moves'] and reply.id != facts.get('charge_move') else 'move_not_legal'
            lane, label = found[0], 'move:' + reply.id
        else:
            party, fainted, targets = facts.get('party_species'), facts.get('fainted'), facts.get('switch_targets')
            if type(party) is not list or type(fainted) is not list or type(targets) is not list:return None, 'facts_missing'
            slots = [p for p, s in enumerate(party) if s == reply.id]
            if not slots:return None, 'species_missing'
            alive = [p for p in slots if not (p < len(fainted) and fainted[p])]
            if not alive:return None, 'species_fainted'
            found = [b for b in range(4, 9) if b - 4 < len(targets) and targets[b - 4] in alive and legal[b]]
            if not found:return None, 'switch_not_legal'
            lane, label = found[0], 'switch:' + reply.id
        labels = facts.get('lanes')   # the re-pricer's own reading of each legal lane: must agree with ours
        if type(labels) is list and len(labels) == 9 and labels[lane] != label:return None, 'lane_label_mismatch'
    if any(root_legal[a] and world.joint_values[a][lane] is None for a in range(9)):return None, 'unpriced'
    return lane, None


def realized(worlds, mapped, root_legal):
    """R(a): the uniform mean over the mapped (world, lane) pairs of G_w[a][lane]; None on illegal actions."""
    return [math.fsum(worlds[w].joint_values[a][b] for w, b in mapped) / len(mapped) if root_legal[a] else None for a in range(9)]


# ---- operators ----

def prior_q(gs, priors):
    """q(a) = sum_w 1/W sum_b priors[w][b] G_w[a][b] over each grid's replies (its kept lanes, or the waiting column 0), summed
    exactly as search_hedge.averaging sums (fsum, world-major), so averaging's own priors give its q bit for bit."""
    count = len(gs)
    return {a: math.fsum(priors[w][b] * g.values[i, k] / count for w, g in enumerate(gs) for k, b in enumerate(g.replies))
            for i, a in enumerate(gs[0].actions)}


def tempered_prior(world, t):
    """pi_w^T: softmax(root_peer_logits / T) over the kept lanes, with v5_gumbel._probabilities' masking (T = inf: uniform
    over them; T = 1: averaging's own prior); a waiting world keeps column 0 at mass 1."""
    if not world.opponent_requested:return WAITING
    return original._probabilities((0.,) * 9 if t == math.inf else tuple(x / t for x in world.opponent_logits), world.opponent_legal)


def tempered(gs, worlds, t):
    """The tempered operator's q (dict over our legal actions) and its argmax, lowest index on ties."""
    q = prior_q(gs, [tempered_prior(w, t) for w in worlds])
    return q, argmax(q, sorted(q))


def external_prior(world, logits, alpha=None):
    """The external reply prior over the kept lanes (softmax of its logits, _probabilities' masking), mixed as
    alpha * external + (1 - alpha) * the champion's pi_w when alpha is given; a waiting world keeps column 0 at mass 1."""
    if not world.opponent_requested:return WAITING
    external = original._probabilities(logits, world.opponent_legal)
    if alpha is None:return external
    own = original._probabilities(world.opponent_logits, world.opponent_legal)
    return tuple(alpha * e + (1. - alpha) * o for e, o in zip(external, own))


def external_operators(gs, worlds, external):
    """({EXTERNAL_SETTINGS name: action or None}, the requested worlds without external logits). A waiting world needs none;
    any requested world without them excludes the decision from every external operator."""
    missing = [w for w, world in enumerate(worlds) if world.opponent_requested and w not in external]
    if missing:return dict.fromkeys(EXTERNAL_SETTINGS), missing
    out = {}
    for name, alpha in (('external_br', None),) + tuple((f'external_mix:alpha={a}', a) for a in EXTERNAL_ALPHA):
        q = prior_q(gs, [external_prior(world, external.get(w), alpha) for w, world in enumerate(worlds)])
        out[name] = argmax(q, sorted(q))
    return out, missing


def load_reply_prior(path):
    """{(ordinal, decision_index): {world: logits}} from a JSONL of rows {ordinal, decision_index, world (the row's block-order
    world index), logits: nine finite reals (the opponent's reply lanes; lanes outside a world's kept set are ignored)}.
    A malformed or duplicated row raises."""
    from array import array
    f = REPLY_PRIOR_FIELDS;out = {}
    with open(path, 'rb') as stream:
        for number, raw in enumerate(stream, 1):
            if not raw.strip():continue
            row = json.loads(raw)
            key = tuple(row.get(f[k]) if type(row) is dict else None for k in ('ordinal', 'decision_index', 'world'))
            logits = row.get(f['logits']) if type(row) is dict else None
            if (any(type(k) is not int for k in key) or type(logits) is not list or len(logits) != 9
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in logits)):
                raise ValueError(f'{path}:{number}: integer ordinal, decision_index and world and nine finite logits required')
            worlds = out.setdefault(key[:2], {})
            if key[2] in worlds:raise ValueError(f'{path}:{number}: duplicate (ordinal, decision_index, world) {key}')
            worlds[key[2]] = array('d', logits)
    return out


def world_lcb(q, spread, kappa):
    """argmax over the legal actions of q(a) - kappa * sd_w(Q_w(a)) / sqrt(W); q is averaging's (= mean_w Q_w(a))."""
    return argmax({a: q[a] - (kappa * s if kappa else 0.) for a, s in spread.items()}, sorted(spread))


def _hedge_top(worlds, legal, lam):
    choice = hedge.bayes_hedge(worlds, legal, lam=lam)
    return None if choice is None else hedge.top_action(choice.strategy)


def operators(worlds, gs, average, greedy, root_logits, root_legal, target, failures):
    """Every SETTINGS choice on one decision's grids (None when that operator failed; counted in `failures`)."""
    chosen = hedge.top_action(average.strategy);qw = per_world_q(gs)
    out = {'greedy': greedy, 'averaging': chosen}
    for name, lam in (('bayes_hedge:lam=0', 0.), ('blend:lam=0.5', .5)):
        try:out[name] = _hedge_top(worlds, root_legal, lam)
        except ValueError:out[name] = None;failures[name] += 1
    for c in PRIOR_C:out[f'prior_regularized:c={c}'] = hedge.top_action(hedge.prior_regularized(average, root_logits, root_legal, c).strategy)
    for k in NOISE_K:out[f'noise_gated:k={k}'] = noise_gated(qw, chosen, greedy, k)
    spread = {a: statistics.stdev([x[a] for x in qw]) / math.sqrt(len(qw)) for a in qw[0]}
    for kappa in LCB_KAPPA:out[f'world_lcb:kappa={kappa}'] = world_lcb(average.q, spread, kappa)
    out['target_argmax'], fallback = target_argmax(target, root_legal, root_logits, average.q)
    for t in TEMPER_T:out[f'tempered:T={t}'] = tempered(gs, worlds, t)[1]
    return out, fallback


def guard_choices(choices, greedy, own_moves, record, b=None):
    """Every operator's choice after the live actor's guards (pokezero.depthn_effect_guard, the same functions): greedy plays
    where the own-recharge guard (an override INTO a recharge move greedy did not choose) or UNREPRESENTED_EFFECT ((b) at the
    root, or (c) on this operator's own choice) trips. Returns ({name: played choice}, {name: [causes] or None}); greedy is
    never guarded; `record` None (a row without the guard) applies the recharge guard alone. `b` replaces the record's root-level
    (b) (the reach scopes: turn-1 (b) or the reach's turn 2); (c) stays the record's."""
    played, causes = {}, {}
    for name, choice in choices.items():
        if name == 'greedy' or choice is None:played[name], causes[name] = choice, None;continue
        why = []
        if effect_guard.recharge_guarded(own_moves, choice, greedy):why.append(OWN_RECHARGE)
        if record is not None and ((record['b'] if b is None else b) or effect_guard.trips(record, choice)['c']):why.append(UNREPRESENTED)
        played[name], causes[name] = (greedy if why else choice), (why or None)
    return played, causes


def band(p):
    for name, upper in BANDS:
        if p < upper:return name


# ---- one decision ----

class Unscorable(Exception):
    def __init__(self, reason):super().__init__(reason);self.reason = reason


def world_prices(decision):
    """The row's worlds as v5_gumbel.WorldPrices(opponent_requested, kept, root_peer_logits, grid), as depthn_backups builds
    them for search_hedge (kept is the reply set: pi_w is the softmax of root_peer_logits over it)."""
    worlds = decision['worlds']
    if not worlds or any(w is None for w in worlds):raise Unscorable('world_missing')
    if any(w['grid'] is None for w in worlds):raise Unscorable('world_unpriced')
    try:return tuple(original.WorldPrices(w['requested'], tuple(w['kept']), tuple(w['logits']), w['grid']) for w in worlds)
    except (ValueError, TypeError):raise Unscorable('world_invalid') from None


@dataclass(frozen=True)
class Base:
    """One row's root and worlds through search_hedge (the averaging the live server played)."""
    legal: tuple
    logits: list
    worlds: tuple
    grids: tuple
    prior: tuple
    average: object


def recompute(decision):
    """Base from a row whose replay is COMPLETE; raises Unscorable."""
    if decision['replay_status'] != 'COMPLETE':raise Unscorable(f"replay_{decision['replay_status']}")
    legal, logits = decision['root_legal'], decision['root_logits']
    if type(legal) is not list or len(legal) != 9 or any(type(x) is not bool for x in legal) or not any(legal):raise Unscorable('root_legal_invalid')
    if type(logits) is not list or len(logits) != 9:raise Unscorable('root_logits_invalid')
    legal = tuple(legal)
    worlds = world_prices(decision)
    gs = hedge.grids(worlds, legal)
    if gs is None:raise Unscorable('unpriced_cell')
    try:prior = original._probabilities(logits, legal)
    except ValueError:raise Unscorable('root_logits_invalid') from None
    return Base(legal, logits, worlds, gs, prior, hedge.averaging(worlds, legal))


def q_json_equal(q, replay_q):
    """Recomputed q against the replay's q_values: exact float equality after a JSON round trip; None only equals None."""
    mine = json.loads(json.dumps(list(q)))
    if type(replay_q) is not list or len(replay_q) != len(mine):return False
    return all(x is None and y is None or x is not None and type(y) in (int, float) and float(x) == float(y) for x, y in zip(mine, replay_q))


def parity_row(decision, base, reason, mode='live'):
    """A row's parity: dict(match, action_equal, q_equal, cause). mode 'live' (a live-searched row): a replay that is not
    COMPLETE, a row whose averaging cannot be recomputed, a recomputed action other than the live one, or a recomputed q other
    than the replay's q_values is a mismatch. mode 'replay' (a replay-COMPLETE row of a replay on a modified tree, whose
    decisions legitimately differ from live): the recomputed action is held to the replay's own action instead of the live one."""
    if mode not in PARITY_MODES:raise ValueError(f'parity mode in {PARITY_MODES} required')
    if decision['replay_status'] != 'COMPLETE':
        return dict(match=False, action_equal=None, q_equal=None, cause=f"replay_{decision['replay_status']}:{decision['replay_cause']}")
    if base is None:return dict(match=False, action_equal=None, q_equal=None, cause='recompute_' + reason)
    reference = decision['live_action'] if mode == 'live' else decision['replay_action']
    action_equal = type(reference) is int and reference == hedge.top_action(base.average.strategy)
    q_equal = q_json_equal(base.average.q, decision['replay_q'])
    cause = None if action_equal and q_equal else 'q_differs' if action_equal else 'action_differs' if q_equal else 'action_and_q_differ'
    return dict(match=cause is None, action_equal=action_equal, q_equal=q_equal, cause=cause)


def parity_passed(matches, total):return total > 0 and matches / total >= PARITY_MIN


def score_decision(decision, reply, target, failures, base=None, external=None):
    """The scored decision: operators' choices and R under both scorings; raises Unscorable. `external` is None without
    --reply-prior, else this decision's {world: logits} (possibly empty): the EXTERNAL_SETTINGS operators join."""
    base = recompute(decision) if base is None else base
    legal, logits, worlds, gs, prior, average = base.legal, base.logits, base.worlds, base.grids, base.prior, base.average
    greedy = decision['greedy_action']
    if type(greedy) is not int or not 0 <= greedy < 9 or not legal[greedy]:raise Unscorable('greedy_missing')
    raw, fallback = operators(worlds, gs, average, greedy, logits, legal, target, failures)
    external_missing = None
    if external is not None:
        found, external_missing = external_operators(gs, worlds, external);raw.update(found)
    record = decision.get('effect_guard') if type(decision.get('effect_guard')) is dict else None
    choices, guards = guard_choices(raw, greedy, decision.get('own_moves'), record)
    reach = None if record is None else effect_guard.reach(record, logits, legal, decision['worlds'])
    scopes = None if reach is None else {scope: bool(record.get('b_turn1')) or reach[f'tripped_{scope}'] for scope in ('lower', 'upper')}
    reach_choices = {} if scopes is None else {scope: guard_choices(raw, greedy, decision.get('own_moves'), record, b=b)[0] for scope, b in scopes.items()}
    if record is not None:   # the any-cell scope (report only): the record's any-cell (b), (c) per operator
        reach_choices['any_cell'] = guard_choices(raw, greedy, decision.get('own_moves'), record,
            b=bool(record['any_cell_b']) if record.get('rule') == effect_guard.RULE else bool(record['b']))[0]
    lanes, drops = [], {}
    if reply.cause is None:
        for w, (world, facts) in enumerate(zip(worlds, decision['worlds'])):
            lane, why = map_world(reply, facts, world, legal)
            if lane is None:drops[w] = why
            else:lanes.append((w, lane))
    observed = reply.cause is None and bool(lanes)
    r_dropped = realized(worlds, lanes, legal) if observed else None
    r_filled = r_dropped if observed else list(average.q)
    def entry_of(choice):
        entry = dict(choice=choice, prior=None, band=None, R=dict(dropped=None, pi_filled=None), D=dict(dropped=None, pi_filled=None))
        if choice is not None:
            entry.update(prior=prior[choice], band=band(prior[choice]))
            for scoring, r in (('dropped', r_dropped), ('pi_filled', r_filled)):
                if r is None:continue
                entry['R'][scoring] = r[choice];entry['D'][scoring] = 0. if choice == greedy else r[choice] - r[greedy]
        return entry
    ops, raw_ops = {}, {}
    for name, choice in choices.items():ops[name] = dict(entry_of(choice), raw_choice=raw[name], guard_causes=guards[name])
    for name, choice in raw.items():raw_ops[name] = entry_of(choice)
    reach_ops = {scope: {name: entry_of(choice) for name, choice in c.items()} for scope, c in reach_choices.items()}
    greedy_argmax = argmax(logits, [a for a in range(9) if legal[a]])
    return dict(observed=observed, unmapped=reply.cause is None and not lanes, mapped_worlds=[list(x) for x in lanes], raw_operators=raw_ops,
                guard=None if record is None else dict(guard_facts(record, raw), reach=reach, scopes=scopes), reach_operators=reach_ops,
                world_drops=drops, R=dict(dropped=r_dropped, pi_filled=r_filled), operators=ops, averaging_q=list(average.q),
                greedy_argmax=greedy_argmax, target_fallback=fallback, worlds=len(worlds), external_missing_worlds=external_missing)


def guard_facts(record, raw):
    """A root's guard facts for the report under the record's own live rule: (b) at turn 1 / turn 2, the tiers of each arm, (a),
    tier-B legality, per operator whether (c) trips on its raw choice, and the any-cell arm (report only). A reach record
    (rule 'reach', screen-r2 on): its turn-2 arm is the reach M (its tiers: a side's tier mass >= THRESHOLD), unavailable /
    error its fail-closed part; an older record's turn-2 arm is the any-cell one (its turn-2 reasons), and its any-cell scope
    is its live scope."""
    reasons = record.get('reasons') or ()
    tiers = lambda source: sorted({r['tier'] for r in reasons if r.get('source') == source and r.get('tier')})
    sides1 = sorted({r['side'] for r in reasons if r.get('source') == 'turn1' and r.get('side')})
    live_reach = record.get('rule') == effect_guard.RULE
    # the any-cell arm's hits: a reach record keeps them under any_cell_reasons (6eefb119's records still in reasons)
    any_reasons = record.get('any_cell_reasons', reasons) if live_reach else reasons
    any_turn2 = any(r.get('source') == 'turn2' for r in any_reasons)   # a priced any-cell turn-2 hit, not an unavailable capture
    masses2 = {}
    if live_reach:
        arm = record.get('reach') or {};unavailable = bool(arm.get('unavailable'))
        turn2 = bool(record.get('b_turn2')) and not unavailable
        # a trip is the tiers whose combined mass crosses THRESHOLD on a side: every contributing tier, with its own mass
        for side, per in (arm.get('tiers') or {}).items():
            if (arm.get(f'm_{side}') or 0.) >= effect_guard.THRESHOLD:masses2[side] = {t: m for t, m in per.items() if m > 0}
        tiers2 = sorted({t for per in masses2.values() for t in per})
    else:
        turn2, tiers2 = any_turn2, tiers('turn2')
        unavailable = any(r.get('source') in ('turn2_unavailable', 'guard_error') for r in reasons)
    any_b = bool(record['any_cell_b']) if live_reach else bool(record['b'])
    return dict(rule=record.get('rule', 'any_cell'), b=bool(record['b']), b_turn1=bool(record.get('b_turn1')), b_turn2=bool(record.get('b_turn2')),
                turn2=turn2, tiers_turn1=tiers('turn1'), sides_turn1=sides1, tiers_turn2=tiers2, tier_masses_turn2=masses2, unavailable=unavailable,
                m_own=record.get('m_own'), m_opp=record.get('m_opp'), arms=record.get('arms'),
                any_cell=dict(b=any_b, turn2=any_turn2, unavailable=any(r.get('source') in ('turn2_unavailable', 'guard_error') for r in list(any_reasons) + list(reasons))),
                a=bool(record.get('a')), tier_b_legal=bool(record.get('tier_b_legal')),
                c={name: effect_guard.trips(record, choice)['c'] for name, choice in raw.items() if choice is not None})


# ---- statistics ----

def cluster_ci(groups, level=LEVEL):
    """Mean D with a cluster-robust t interval: D summed per battle, then the mean and SE over battles (t, battles - 1 df).
    `mean`/`ci90` are per decision (sum D / decisions; the per-battle interval over mean decisions per battle)."""
    sums = [math.fsum(v) for v in groups.values() if v];m = len(sums);n = sum(len(v) for v in groups.values())
    out = dict(mean=None, ci90=None, battles=m, decisions=n, per_battle_mean=None, per_battle_se=None, per_battle_ci90=None)
    if not n:return out
    per_battle = math.fsum(sums) / m;scale = m / n
    out.update(mean=math.fsum(sums) / n, per_battle_mean=per_battle)
    if m >= 2:
        from scipy.stats import t
        se = statistics.stdev(sums) / math.sqrt(m);half = float(t.ppf(.5 + level / 2, m - 1)) * se
        out.update(per_battle_se=se, per_battle_ci90=[per_battle - half, per_battle + half],
                   ci90=[(per_battle - half) * scale, (per_battle + half) * scale])
    return out


def operator_metric(records, name, scoring, band_name=None):
    groups, overrides, n = collections.defaultdict(list), 0, 0
    for r in records:
        choice, d, b = r['ops'][name]
        d = d[0] if scoring == 'dropped' else d[1]
        if choice is None or d is None or band_name is not None and b != band_name:continue
        groups[r['ordinal']].append(d);n += 1;overrides += choice != r['ops']['greedy'][0]
    return dict(override=overrides / n if n else None, **cluster_ci(groups))


def population(records, scoring, half='all', kind=None):
    return [r for r in records if (half == 'all' or r['half'] == half) and (scoring == 'pi_filled' or r['observed'])
            and (kind is None or r['kind'] == kind)]


def metrics(records, settings=SETTINGS):
    """{scoring: {half: {operators, by_request_kind, by_prior_band}}} over the compact decision records."""
    kinds = sorted({r['kind'] for r in records if r['kind'] is not None})
    out = {}
    for scoring in SCORINGS:
        out[scoring] = {}
        for half in HALVES:
            pop = population(records, scoring, half)
            out[scoring][half] = dict(
                decisions=len(pop), battles=len({r['ordinal'] for r in pop}),
                operators={name: operator_metric(pop, name, scoring) for name in settings},
                by_request_kind={k: {name: operator_metric(population(pop, scoring, kind=k), name, scoring) for name in settings} for k in kinds},
                by_prior_band={b: {name: operator_metric(pop, name, scoring, b) for name in settings} for b, _ in BANDS})
    return out


def operator_table(records, settings):
    """{scoring: {operator: operator_metric}} over all of `records` (both halves)."""
    return {scoring: {name: operator_metric(population(records, scoring), name, scoring) for name in settings} for scoring in SCORINGS}


def reach_fields(guard):
    """decisions.jsonl's reach fields (the measurement session keys on these names): reach_turn2 = {m_own_lower, m_own_upper,
    m_opp_lower, m_opp_upper, tripped (None: the bounds straddle the threshold), exact}, reach_scope_tripped_lower and _upper
    (turn-1 (b) or the reach's turn 2 at that bound; (c) applies per operator). All None for a row without the guard record."""
    if guard is None or guard.get('reach') is None:
        return dict(reach_turn2=None, reach_scope_tripped_lower=None, reach_scope_tripped_upper=None)
    r = guard['reach']
    return dict(reach_turn2={k: r[k] for k in ('m_own_lower', 'm_own_upper', 'm_opp_lower', 'm_opp_upper', 'tripped', 'exact')},
                reach_scope_tripped_lower=guard['scopes']['lower'], reach_scope_tripped_upper=guard['scopes']['upper'])


def reach_report(have, settings):
    """The reach scopes (report only; module docstring) over the rows with the guard record: per bound its guarded share
    (union; turn 1 and reach-turn-2-added; per tier and per side, turn 1 or the reach), the ambiguous rows, RR on its unguarded
    roots under its own guards, and a cross-tab against the any-cell scope."""
    have = [r for r in have if r['guard'].get('reach') is not None]   # rows scored here (score_decision's reach)
    n = len(have);share = lambda k: k / n if n else None;g = lambda r: r['guard']
    out = dict(rows=n, threshold=effect_guard.THRESHOLD, ambiguous=sum(g(r)['reach']['tripped'] is None for r in have),
               exact_rows=sum(bool(g(r)['reach']['exact']) for r in have), unavailable=sum(bool(g(r)['reach']['unavailable']) for r in have))
    for scope in ('lower', 'upper'):
        reach2 = lambda r: bool(g(r)['reach'][f'tripped_{scope}'])
        # a tripping side's every contributing tier (mass > 0), so an A + C combination is not dropped
        tier = lambda r, t: (t in g(r)['tiers_turn1']) or g(r)['reach']['unavailable'] or (g(r)['reach']['tiers'] is not None and any(
            v[t] > 0 and (g(r)['reach'][f'm_{side}_{scope}'] or 0.) >= effect_guard.THRESHOLD for side, v in g(r)['reach']['tiers'][scope].items()))
        side = lambda r, s: (s in g(r)['sides_turn1']) or g(r)['reach']['unavailable'] or (
            (g(r)['reach'][f'm_{s}_{scope}'] or 0.) >= effect_guard.THRESHOLD)
        guarded = [r for r in have if g(r)['scopes'][scope]];clean = [r for r in have if not g(r)['scopes'][scope]]
        t1 = sum(g(r)['b_turn1'] for r in have);added = sum(reach2(r) and not g(r)['b_turn1'] for r in have)
        cross = collections.Counter((('any_cell' if g(r)['any_cell']['b'] else 'any_cell_clear'), ('reach' if g(r)['scopes'][scope] else 'reach_clear'))
                                    for r in have)
        out[scope] = dict(roots=len(guarded), share=share(len(guarded)), turn1=t1, turn1_share=share(t1), turn2_added=added, turn2_added_share=share(added),
            tiers={t: dict(roots=sum(tier(r, t) for r in have), share=share(sum(tier(r, t) for r in have))) for t in ('A', 'C')},
            sides={s: dict(roots=sum(side(r, s) for r in have), share=share(sum(side(r, s) for r in have))) for s in ('own', 'opp')},
            unguarded=dict(roots=len(clean), table=operator_table([dict(r, ops=r['reach_ops'][scope]) for r in clean], settings)),
            cross_tab={f'{a}/{b}': cross[(a, b)] for a in ('any_cell', 'any_cell_clear') for b in ('reach', 'reach_clear')})
    return out


def any_cell_report(have, settings):
    """The any-cell scope (report only, the mirror of the reach section): turn-1 (b) or a guarded turn-2 prior >= THRESHOLD at
    any expanded cell, (c) per operator. Its guarded share (union, turn 1 vs any-cell-turn-2-added, unavailable or error), RR
    on its unguarded roots under its own guards, and a cross-tab against the live scope (the record's rule)."""
    have = [r for r in have if 'any_cell' in (r.get('reach_ops') or {})];n = len(have);share = lambda k: k / n if n else None;g = lambda r: r['guard']
    guarded = [r for r in have if g(r)['any_cell']['b']];clean = [r for r in have if not g(r)['any_cell']['b']]
    t1 = sum(g(r)['b_turn1'] for r in have);added = sum(g(r)['any_cell']['turn2'] and not g(r)['b_turn1'] for r in have)
    error = sum(g(r)['any_cell']['b'] and not g(r)['b_turn1'] and not g(r)['any_cell']['turn2'] for r in have)
    cross = collections.Counter((('any_cell' if g(r)['any_cell']['b'] else 'any_cell_clear'), ('live' if g(r)['b'] else 'live_clear')) for r in have)
    return dict(rows=n, rules=dict(collections.Counter(g(r)['rule'] for r in have)), roots=len(guarded), share=share(len(guarded)),
        turn1=t1, turn1_share=share(t1), turn2_added=added, turn2_added_share=share(added), unavailable_or_error=error,
        unguarded=dict(roots=len(clean), table=operator_table([dict(r, ops=r['reach_ops']['any_cell']) for r in clean], settings)),
        cross_tab={f'{a}/{b}': cross[(a, b)] for a in ('any_cell', 'any_cell_clear') for b in ('live', 'live_clear')})


def guard_report(records, settings, opponent_recharge=0):
    """The effect guard over the scored (priced) roots that carry its record (module docstring): its share of them per tier
    and for the union, turn-1 and turn-2-added parts apart, (c) per operator; the primary result, RR on the unguarded roots
    ((b) false; every guard still applies per operator); RR on the guarded roots with the raw (unguarded) choices, for
    context; the unguarded roots split by tier-B legality; the sensitivity rows ((a) holds, neither (b) nor any operator's (c))."""
    have = [r for r in records if r['guard'] is not None];n = len(have);share = lambda k: k / n if n else None
    g = lambda r: r['guard'];b = [r for r in have if g(r)['b']];unguarded = [r for r in have if not g(r)['b']]
    def part(test1, test2):
        t1 = sum(bool(test1(r)) for r in have);added = sum(bool(test2(r)) and not test1(r) for r in have)
        return dict(turn1=t1, turn1_share=share(t1), turn2_added=added, turn2_added_share=share(added), total=t1 + added, share=share(t1 + added))
    tiers = {t: part(lambda r, t=t: t in g(r)['tiers_turn1'], lambda r, t=t: t in g(r)['tiers_turn2']) for t in ('A', 'C')}
    union = part(lambda r: g(r)['b_turn1'], lambda r: g(r).get('turn2', g(r)['b_turn2']))
    # (b) without a turn-1 or a priced turn-2 reason: an unavailable turn-2 capture or an error record (fail closed): its own part
    error = sum(g(r)['b'] and not g(r)['b_turn1'] and not g(r).get('turn2', g(r)['b_turn2']) for r in have)
    union.update(unavailable_or_error=error, unavailable_or_error_share=share(error), total=union['total'] + error, share=share(union['total'] + error))
    c = {name: dict(c_only=sum(g(r)['c'].get(name, False) and not g(r)['b'] for r in have),
                    union=sum(g(r)['b'] or g(r)['c'].get(name, False) for r in have)) for name in settings}
    for v in c.values():v.update(union_share=share(v['union']))
    sensitive = [r for r in have if g(r)['a'] and not g(r)['b'] and not any(g(r)['c'].values())]
    raw = lambda rs: [dict(r, ops=r['raw_ops']) for r in rs]
    return dict(priced_roots=n, without_record=len(records) - n, opponent_recharge_refused=opponent_recharge, threshold=effect_guard.THRESHOLD,
        union=union, tiers=tiers, unavailable=sum(g(r)['unavailable'] for r in have), c=c,
        guarded_share=union['share'], above_20pct=union['share'] is not None and union['share'] > GUARD_SHARE_FLAG,
        max_union_with_c_share=max((v['union_share'] for v in c.values() if v['union_share'] is not None), default=None),
        primary=dict(roots=len(unguarded), table=operator_table(unguarded, settings)),
        guarded_context=dict(roots=len(b), table=operator_table(raw(b), settings)),
        tier_b=dict(legal=dict(roots=sum(g(r)['tier_b_legal'] for r in unguarded), table=operator_table([r for r in unguarded if g(r)['tier_b_legal']], settings)),
                    not_legal=dict(roots=sum(not g(r)['tier_b_legal'] for r in unguarded),
                                   table=operator_table([r for r in unguarded if not g(r)['tier_b_legal']], settings))),
        sensitivity=dict(roots=len(sensitive), share=share(len(sensitive)), table=operator_table(sensitive, settings)),
        reach=reach_report(have, settings), any_cell=any_cell_report(have, settings),
        rules=dict(collections.Counter(r['guard']['rule'] for r in have)))


def validation_gate(table):
    m = table['pi_filled']['all']['operators']['averaging'];upper = None if m['ci90'] is None else m['ci90'][1]
    return dict(rule='averaging D under pi_filled, all decisions: upper 90% bound < 0', mean=m['mean'], ci90=m['ci90'],
                battles=m['battles'], decisions=m['decisions'], passed=upper is not None and upper < 0)


def select(table):
    """Even-seed battles under pi_filled: the highest mean D among CANDIDATES with override >= 5% (declared order on ties)."""
    even, rows, best = table['pi_filled']['even']['operators'], [], None
    for name in CANDIDATES:
        m = even[name];eligible = m['override'] is not None and m['override'] >= OVERRIDE_MIN and m['mean'] is not None
        rows.append(dict(setting=name, override=m['override'], mean=m['mean'], ci90=m['ci90'], eligible=eligible))
        if eligible and (best is None or m['mean'] > best['mean']):best = rows[-1]
    return dict(chosen=None if best is None else best['setting'], candidates=rows)


def confirm(table, setting):
    """Odd-seed battles: PASS when the lower 90% bound is > 0 under pi_filled and dropped and the pi_filled override >= 5%."""
    pi, dropped = table['pi_filled']['odd']['operators'][setting], table['dropped']['odd']['operators'][setting]
    lower = {s: None if m['ci90'] is None else m['ci90'][0] for s, m in (('pi_filled', pi), ('dropped', dropped))}
    checks = dict(pi_filled_lower_above_0=lower['pi_filled'] is not None and lower['pi_filled'] > 0,
                  dropped_lower_above_0=lower['dropped'] is not None and lower['dropped'] > 0,
                  override_at_least_5pct=pi['override'] is not None and pi['override'] >= OVERRIDE_MIN)
    return dict(setting=setting, pi_filled=pi, dropped=dropped, checks=checks, passed=all(checks.values()))


def verdict(parity_ok, gate, selection, confirmation):
    if not parity_ok:return 'PARITY_FAILED'
    if not gate['passed']:return 'GATE_FAILED'
    if selection['chosen'] is None:return 'NO_CANDIDATE'
    return 'PASS' if confirmation['passed'] else 'FAIL'


def decide(table, parity_ok):
    """Gate, selection and confirmation; no candidate is selected when parity or the gate fails."""
    gate = validation_gate(table)
    selection = select(table) if parity_ok and gate['passed'] else dict(chosen=None, candidates=None)
    confirmation = None if selection['chosen'] is None else confirm(table, selection['chosen'])
    return dict(gate=gate, selection=selection, confirmation=confirmation,
                verdict=verdict(parity_ok, gate, selection, confirmation or dict(passed=False)))


# ---- the live run directory ----

class RunDirectory:
    def __init__(self, root):self.root = Path(root);self._cells = {};self._targets = {}

    def cell(self, ordinal):
        if ordinal not in self._cells:
            path = self.root / 'results' / f'{ordinal}.json'
            cell = (json.loads(path.read_bytes()).get('cell') or {}) if path.exists() else {}
            self._cells[ordinal] = dict(seed=cell.get('seed'), arm=cell.get('arm'))
        return self._cells[ordinal]

    def policy_target(self, ordinal, decision_index):
        if ordinal not in self._targets:
            targets, path = {}, self.root / 'telemetry' / f'{ordinal}.jsonl'
            if path.exists():
                for raw in path.read_bytes().split(b'\n'):
                    try:row = json.loads(raw)
                    except ValueError:continue
                    if type(row) is dict and type(row.get('decision_index')) is int:
                        target = (row.get('search') or {}).get('policy_target')
                        targets[row['decision_index']] = tuple(target) if type(target) is list else None
            self._targets[ordinal] = targets
        target = self._targets[ordinal].get(decision_index)
        return None if target is None else list(target)

    def capture(self, ordinal):return load_capture(str(self.root / 'owning-wire' / f'{ordinal}.jsonl'))


# ---- the run ----

class Coverage:
    def __init__(self):
        self.rows = 0;self.duplicates = 0;self.arms = collections.Counter();self.sources = collections.Counter()
        self.replay_complete = 0;self.excluded = collections.Counter();self.replay_agree = [0, 0]
        self.parity = collections.Counter();self.parity_causes = collections.Counter();self.status_mismatches = collections.Counter()
        self.greedy_disagree = [];self.replies = collections.Counter();self.details = collections.Counter()
        self.kind = collections.defaultdict(lambda: collections.Counter());self.causes = collections.Counter()
        self.unmapped = 0;self.mapped_hist = collections.Counter();self.drops = collections.Counter()
        self.target_fallbacks = 0;self.failures = collections.Counter();self.no_half = 0;self.seen = set()
        self.external = None   # with --reply-prior: Counter(decisions, missing_decisions, missing_worlds)
        self.opponent_recharge = 0   # rows refused OPPONENT_RECHARGE_UNREPRESENTED: no priced worlds, greedy plays (D = 0), never scored

    def add_parity(self, check):
        self.parity['rows'] += 1;self.parity['matches'] += check['match']
        self.parity['action_equal'] += check['action_equal'] is True;self.parity['q_equal'] += check['q_equal'] is True
        if check['cause'] is not None:self.parity_causes[check['cause']] += 1

    def report(self, records):
        scored = len(records);share = lambda a, b: a / b if b else None
        battles = {h: len({r['ordinal'] for r in records if r['half'] == h}) for h in ('even', 'odd')}
        p = self.parity
        return dict(
            searched_decisions=self.rows, duplicate_rows=self.duplicates, arms=dict(self.arms), live_sources=dict(self.sources),
            replay_complete=self.replay_complete, scored=scored, excluded=dict(self.excluded),
            parity=dict(mode=getattr(self, 'parity_mode', 'live'), rule=(
                (f'every row searched live (source search): replay COMPLETE, averaging recomputed from the captured worlds '
                 f'plays the live search action_index, and its q equals the replay q_values after a JSON round trip; >= {PARITY_MIN:.0%}')
                if getattr(self, 'parity_mode', 'live') == 'live' else
                (f'replay mode (a modified tree): every replay-COMPLETE row; averaging recomputed from the captured worlds plays the '
                 f"replay's own action_index and its q equals the replay q_values after a JSON round trip; >= {PARITY_MIN:.0%}")),
                        live_searched=p['rows'], matches=p['matches'], rate=share(p['matches'], p['rows']),
                        action_equal=p['action_equal'], action_rate=share(p['action_equal'], p['rows']),
                        q_equal=p['q_equal'], q_rate=share(p['q_equal'], p['rows']), mismatches_by_cause=dict(self.parity_causes),
                        passed=parity_passed(p['matches'], p['rows'])),
            status_mismatches=dict(note='live searched but replay not COMPLETE (replay status:cause), or replay COMPLETE but not searched '
                                        'live (live source:cause)', by_cause=dict(self.status_mismatches)),
            averaging_equals_replay_action=dict(with_replay_action=self.replay_agree[1], matches=self.replay_agree[0], rate=share(*self.replay_agree)),
            greedy_check=dict(rule='live greedy == argmax of root_logits over root_legal (lowest index on ties)',
                              disagreements=len(self.greedy_disagree), examples=self.greedy_disagree[:20]),
            replies=dict(self.replies),
            censored=dict(count=sum(self.causes.values()), share=share(sum(self.causes.values()), scored),
                          by_request_kind={k: dict(decisions=c['decisions'], censored=c['censored'], share=share(c['censored'], c['decisions']))
                                           for k, c in sorted(self.kind.items(), key=lambda x: str(x[0]))},
                          by_cause={c: dict(count=self.causes[c], share=share(self.causes[c], scored)) for c in CAUSES}),
            reply_details=dict(self.details),
            unmapped=dict(count=self.unmapped, share=share(self.unmapped, scored)),
            mapped_worlds=dict(note='uncensored decisions: the number of worlds b_FP maps into',
                               histogram={str(k): self.mapped_hist[k] for k in sorted(self.mapped_hist)}),
            world_drop_reasons=dict(self.drops), target_argmax_fallbacks=self.target_fallbacks, operator_failures=dict(self.failures),
            halves=dict(even=dict(battles=battles['even'], decisions=sum(r['half'] == 'even' for r in records)),
                        odd=dict(battles=battles['odd'], decisions=sum(r['half'] == 'odd' for r in records)), no_seed=self.no_half),
            fp_requested=dict(note="decisions whose reply is not 'wait'", decisions=sum(r['fp_requested'] for r in records)),
            external_prior=None if self.external is None else dict(
                note='scored decisions; one missing a requested world\'s external logits is excluded from the external operators',
                scored=self.external['decisions'], with_logits=self.external['decisions'] - self.external['missing_decisions'],
                excluded=self.external['missing_decisions'], missing_worlds=self.external['missing_worlds']))


def half_of(seed):
    """'even' for an even battle seed (cell.seed % 2 == 0), 'odd' for an odd one, None without a seed."""
    return None if type(seed) is not int else 'even' if seed % 2 == 0 else 'odd'


def process(decision, run, coverage, prior=None, parity_mode='live'):
    """One row: (decisions.jsonl entry, compact record or None). `prior`: load_reply_prior's dict, or None."""
    coverage.rows += 1
    ordinal, index = decision['ordinal'], decision['decision_index']
    address = dict(ordinal=ordinal, battle_id=decision['battle_id'], owner=decision['owner'], decision_index=index,
                   request_count=decision['request_count'], turn=decision['turn'], request_kind=decision['request_kind'],
                   source=decision['source'])
    if (ordinal, index) in coverage.seen:coverage.duplicates += 1;return dict(address, scored=False, reason='duplicate_row'), None
    coverage.seen.add((ordinal, index))
    cell = run.cell(ordinal);seed = cell['seed'];half = half_of(seed)
    coverage.arms[str(cell['arm'])] += 1;coverage.sources[str(decision['source'])] += 1
    address.update(seed=seed, half=half, arm=cell['arm'])
    complete = decision['replay_status'] == 'COMPLETE';searched = decision['source'] == 'search'
    if decision.get('refusal_cause') == 'OPPONENT_RECHARGE_UNREPRESENTED':coverage.opponent_recharge += 1   # every operator plays greedy
    coverage.replay_complete += complete
    if searched != complete:
        coverage.status_mismatches[f"replay_{decision['replay_status']}:{decision['replay_cause']}" if searched
                                   else f"live_{decision['source']}:{decision['live_cause']}"] += 1
    base = reason = None
    try:base = recompute(decision)
    except Unscorable as error:reason = error.reason
    checked = searched if parity_mode == 'live' else complete   # replay mode: every replay-COMPLETE row, held to itself
    parity = parity_row(decision, base, reason, parity_mode) if checked else None
    if parity is not None:coverage.add_parity(parity)
    address.update(parity=parity)
    if base is None:
        coverage.excluded[reason] += 1
        return dict(address, scored=False, reason=reason), None
    reply = reply_for(run.capture(ordinal), decision)
    external = None if prior is None else prior.get((ordinal, index), {})
    try:scored = score_decision(decision, reply, run.policy_target(ordinal, index), coverage.failures, base, external)
    except Unscorable as error:
        coverage.excluded[error.reason] += 1
        return dict(address, scored=False, reason=error.reason), None
    ops = scored['operators'];chosen = ops['averaging']['choice']
    if type(decision['replay_action']) is int:coverage.replay_agree[0] += decision['replay_action'] == chosen;coverage.replay_agree[1] += 1
    if scored['greedy_argmax'] != decision['greedy_action']:
        coverage.greedy_disagree.append(dict(ordinal=ordinal, decision_index=index, live=decision['greedy_action'], argmax=scored['greedy_argmax']))
    coverage.replies[reply.kind or 'censored'] += 1;coverage.details[reply.detail] += 1
    kind = coverage.kind[decision['request_kind']];kind['decisions'] += 1;kind['censored'] += reply.cause is not None
    if reply.cause is not None:coverage.causes[reply.cause] += 1
    else:coverage.mapped_hist[len(scored['mapped_worlds'])] += 1
    coverage.unmapped += scored['unmapped'];coverage.target_fallbacks += scored['target_fallback'];coverage.no_half += half is None
    for why in scored['world_drops'].values():coverage.drops[why] += 1
    if external is not None:
        missing = scored['external_missing_worlds']
        coverage.external['decisions'] += 1;coverage.external['missing_decisions'] += bool(missing);coverage.external['missing_worlds'] += len(missing)
    entry = dict(address, scored=True, reply=dict(kind=reply.kind, id=reply.id, cause=reply.cause, detail=reply.detail),
                 censored=reply.cause is not None, unmapped=scored['unmapped'], observed=scored['observed'],
                 mapped_worlds=scored['mapped_worlds'], world_drops={str(k): v for k, v in scored['world_drops'].items()},
                 live_action=decision['live_action'], averaging_action=chosen, greedy_action=decision['greedy_action'],
                 greedy_argmax=scored['greedy_argmax'], target_fallback=scored['target_fallback'], R=scored['R'], operators=ops, guard=scored['guard'],
                 **reach_fields(scored['guard']),
                 external_missing_worlds=scored['external_missing_worlds'])
    record = dict(ordinal=ordinal, half=half, kind=decision['request_kind'], observed=scored['observed'], fp_requested=reply.kind != 'wait',
                  ops={n: (o['choice'], (o['D']['dropped'], o['D']['pi_filled']), o['band']) for n, o in ops.items()},
                  raw_ops={n: (o['choice'], (o['D']['dropped'], o['D']['pi_filled']), o['band']) for n, o in scored['raw_operators'].items()},
                  reach_ops={scope: {n: (o['choice'], (o['D']['dropped'], o['D']['pi_filled']), o['band']) for n, o in ops_.items()}
                             for scope, ops_ in scored['reach_operators'].items()},
                  guard=scored['guard'])
    return entry, record


def score_run(rows, run_directory, out, reply_prior=None, parity_mode='live'):
    """Score every row into the new directory `out`; returns the summary. `reply_prior`: an external reply prior JSONL
    (load_reply_prior) adding the reported-only EXTERNAL_SETTINGS operators."""
    prior = None if reply_prior is None else load_reply_prior(reply_prior)
    settings = SETTINGS + (EXTERNAL_SETTINGS if prior is not None else ())
    out = Path(out);out.mkdir(parents=True, exist_ok=False)
    if parity_mode not in PARITY_MODES:raise ValueError(f'parity mode in {PARITY_MODES} required')
    run, coverage, records = RunDirectory(run_directory), Coverage(), [];coverage.parity_mode = parity_mode
    if prior is not None:coverage.external = collections.Counter()
    with (out / 'decisions.jsonl').open('x') as stream:
        for decision in load_rows(rows):
            entry, record = process(decision, run, coverage, prior, parity_mode)
            stream.write(json.dumps(entry) + '\n')
            if record is not None:records.append(record)
    table = metrics(records, settings);cover = coverage.report(records)
    outcome = decide(table, cover['parity']['passed'])
    # Reported only (the gate and selection use `metrics`): without 'wait' decisions. There every mapped world is a waiting
    # world (pi_w is column 0's unit mass), so R is averaging's own per-world value over them: exactly its q, and averaging's
    # D >= 0, when every world waits.
    requested = metrics([r for r in records if r['fp_requested']], settings)
    guard = guard_report(records, settings, coverage.opponent_recharge)
    summary = dict(schema=SCHEMA, inputs=dict(rows=[str(Path(p).resolve()) for p in rows], run_directory=str(Path(run_directory).resolve()),
                                              reply_prior=None if reply_prior is None else str(Path(reply_prior).resolve())),
                   settings=list(settings), candidates=list(CANDIDATES), reported_only=[n for n in settings if n in EXTERNAL_SETTINGS], level=LEVEL, override_min=OVERRIDE_MIN, parity_min=PARITY_MIN,
                   bands=[b for b, _ in BANDS], **outcome, coverage=cover, metrics=table, metrics_fp_requested=requested, guard=guard)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    (out / 'summary.md').write_text(markdown(summary))
    return summary


# ---- summary.md ----

def _num(x, signed=True):return '-' if x is None else (f'{x:+.4f}' if signed else f'{x:.4f}')


def _pct(x):return '-' if x is None else f'{100 * x:.1f}%'


def _ci(ci):return '-' if ci is None else f'[{ci[0]:+.4f}, {ci[1]:+.4f}]'


def _metric_row(label, m):
    return f"| {label} | {_pct(m['override'])} | {_num(m['mean'])} | {_ci(m['ci90'])} | {_num(m['per_battle_mean'])} | {m['battles']} | {m['decisions']} |"


HEADER = '| setting | override | mean D | 90% CI | per-battle mean D | battles | decisions |\n|---|---|---|---|---|---|---|'


def markdown(summary):
    cover, table, gate, selection, confirmation = (summary[k] for k in ('coverage', 'metrics', 'gate', 'selection', 'confirmation'))
    settings = summary['settings']
    out = ['# RR (realized-reply re-pricing) score', '', f"**Verdict: {summary['verdict']}**", '',
           f"- Parity (rows searched live: replay COMPLETE, recomputed averaging's action == live, q == replay q): "
           f"{cover['parity']['matches']}/{cover['parity']['live_searched']} = {_pct(cover['parity']['rate'])} "
           f"(required >= {_pct(PARITY_MIN)}): {'ok' if cover['parity']['passed'] else 'PARITY_FAILED'}",
           f"- Gate (averaging D, pi_filled, all decisions, upper 90% bound < 0): mean {_num(gate['mean'])}, CI {_ci(gate['ci90'])}: "
           f"{'passed' if gate['passed'] else 'GATE_FAILED'}",
           f"- Selected (even seeds, pi_filled, override >= {_pct(OVERRIDE_MIN)}, highest mean D): {selection['chosen'] or 'none'}"]
    if confirmation is not None:
        out.append(f"- Confirmation (odd seeds): pi_filled CI {_ci(confirmation['pi_filled']['ci90'])}, dropped CI "
                   f"{_ci(confirmation['dropped']['ci90'])}, override {_pct(confirmation['pi_filled']['override'])}: "
                   f"{'PASS' if confirmation['passed'] else 'FAIL'}")
    out += ['', '## Coverage', '', '| item | value |', '|---|---|',
            f"| searched decisions (rows) | {cover['searched_decisions']} |", f"| replay COMPLETE | {cover['replay_complete']} |",
            f"| scored | {cover['scored']} |", f"| excluded | {cover['excluded']} |",
            f"| parity | {cover['parity']['matches']}/{cover['parity']['live_searched']} ({_pct(cover['parity']['rate'])}); "
            f"action {_pct(cover['parity']['action_rate'])}, q {_pct(cover['parity']['q_rate'])} |",
            f"| parity mismatches by cause | {cover['parity']['mismatches_by_cause']} |",
            f"| status mismatches by cause | {cover['status_mismatches']['by_cause']} |",
            f"| decisions with Foul Play requested (not 'wait') | {cover['fp_requested']['decisions']} |",
            f"| external reply prior (reported-only operators) | {cover['external_prior'] or 'not given'} |",
            f"| averaging == replay action | {_pct(cover['averaging_equals_replay_action']['rate'])} |",
            f"| greedy check disagreements | {cover['greedy_check']['disagreements']} |",
            f"| censored | {cover['censored']['count']} ({_pct(cover['censored']['share'])}) |",
            f"| unmapped | {cover['unmapped']['count']} ({_pct(cover['unmapped']['share'])}) |",
            f"| replies | {cover['replies']} |", f"| target_argmax fallbacks | {cover['target_argmax_fallbacks']} |",
            f"| operator failures | {cover['operator_failures']} |", f"| halves | {cover['halves']} |", '',
            '| censored by request_kind | decisions | censored | share |', '|---|---|---|---|']
    out += [f"| {k} | {c['decisions']} | {c['censored']} | {_pct(c['share'])} |" for k, c in cover['censored']['by_request_kind'].items()]
    out += ['', '| censored cause | count | share of scored |', '|---|---|---|']
    out += [f"| {c} | {v['count']} | {_pct(v['share'])} |" for c, v in cover['censored']['by_cause'].items()]
    out += ['', '| mapped worlds | decisions |', '|---|---|'] + [f'| {k} | {v} |' for k, v in cover['mapped_worlds']['histogram'].items()]
    out += ['', f"World drop reasons: {cover['world_drop_reasons']}", '']
    if selection['candidates']:
        out += ['## Selection (even seeds, pi_filled)', '', '| setting | override | mean D | 90% CI | eligible |', '|---|---|---|---|---|']
        out += [f"| {r['setting']} | {_pct(r['override'])} | {_num(r['mean'])} | {_ci(r['ci90'])} | {'yes' if r['eligible'] else 'no'} |"
                for r in selection['candidates']]
        out.append('')
    for scoring in SCORINGS:
        for half in HALVES:
            block = table[scoring][half]
            out += [f"## {scoring}, {half} ({block['battles']} battles, {block['decisions']} decisions)", '', HEADER]
            out += [_metric_row(name, block['operators'][name]) for name in settings] + ['']
        out += [f'## {scoring}, all: by request_kind', '', HEADER.replace('| setting |', '| setting / kind |')]
        out += [_metric_row(f'{name} / {k}', m[name]) for k, m in table[scoring]['all']['by_request_kind'].items() for name in settings] + ['']
        out += [f"## {scoring}, all: by prior band of the setting's action", '', HEADER.replace('| setting |', '| setting / band |')]
        out += [_metric_row(f'{name} / {b}', m[name]) for b, m in table[scoring]['all']['by_prior_band'].items() for name in settings
                if m[name]['decisions']] + ['']
    out += guard_markdown(summary['guard'], settings)
    requested = summary['metrics_fp_requested']
    out += ["# Reported view: decisions where Foul Play was requested (no 'wait' decisions; the gate and selection above are unchanged)", '']
    for scoring in SCORINGS:
        for half in HALVES:
            block = requested[scoring][half]
            out += [f"## requested only: {scoring}, {half} ({block['battles']} battles, {block['decisions']} decisions)", '', HEADER]
            out += [_metric_row(name, block['operators'][name]) for name in settings] + ['']
    return '\n'.join(out) + '\n'


def guard_markdown(guard, settings):
    """summary.md's effect-guard section."""
    if guard is None:return []
    u, t = guard['union'], guard['tiers']
    out = ['# Effect guard (UNREPRESENTED_EFFECT; every operator plays greedy where a guard trips)', '',
           f"- live rule per record: {guard.get('rules')} (reach: turn-1 (b), the reach M_own or M_opp >= 5%, or (c); any_cell: an older "
           "record's any-cell turn-2 arm). The primary, the gate and the selection follow each record's live rule.",
           f"- priced roots with the guard record: {guard['priced_roots']} (without one: {guard['without_record']}); refused "
           f"OPPONENT_RECHARGE_UNREPRESENTED (no priced worlds, greedy plays): {guard['opponent_recharge_refused']}",
           f"- (b) guarded share: {_pct(guard['guarded_share'])} (turn 1 {_pct(u['turn1_share'])}, turn 2 added {_pct(u['turn2_added_share'])}, "
           f"turn-2 capture unavailable or guard error {_pct(u['unavailable_or_error_share'])})"
           + (f"; **above {GUARD_SHARE_FLAG:.0%}**" if guard['above_20pct'] else ''),
           f"- largest (b) or (c) share over the operators: {_pct(guard['max_union_with_c_share'])}", '',
           '| tier | turn 1 | turn 2 added | total share |', '|---|---|---|---|']
    out += [f"| {k} | {v['turn1']} ({_pct(v['turn1_share'])}) | {v['turn2_added']} ({_pct(v['turn2_added_share'])}) | {_pct(v['share'])} |" for k, v in t.items()]
    out += [f"| unavailable or error | - | - | {u['unavailable_or_error']} ({_pct(u['unavailable_or_error_share'])}) |",
            f"| union | {u['turn1']} ({_pct(u['turn1_share'])}) | {u['turn2_added']} ({_pct(u['turn2_added_share'])}) | {_pct(u['share'])} |", '',
            '| setting | (c) only roots | (b) or (c) share |', '|---|---|---|']
    out += [f"| {name} | {guard['c'][name]['c_only']} | {_pct(guard['c'][name]['union_share'])} |" for name in settings if name in guard['c']]
    for title, block in ((f"Primary: RR on unguarded roots ({guard['primary']['roots']})", guard['primary']['table']),
                         (f"Guarded roots, raw choices for context ({guard['guarded_context']['roots']})", guard['guarded_context']['table']),
                         (f"Unguarded, tier-B move legal ({guard['tier_b']['legal']['roots']})", guard['tier_b']['legal']['table']),
                         (f"Unguarded, no tier-B move legal ({guard['tier_b']['not_legal']['roots']})", guard['tier_b']['not_legal']['table']),
                         (f"Sensitivity: (a) holds, neither (b) nor (c) ({guard['sensitivity']['roots']}, {_pct(guard['sensitivity']['share'])})",
                          guard['sensitivity']['table'])):
        for scoring in SCORINGS:
            out += ['', f'## {title}: {scoring}', '', HEADER] + [_metric_row(name, block[scoring][name]) for name in settings]
    return out + [''] + reach_markdown(guard.get('reach'), settings) + any_cell_markdown(guard.get('any_cell'), settings)


def any_cell_markdown(any_cell, settings):
    """summary.md's any-cell section (report only)."""
    if not any_cell or not any_cell['rows']:return []
    c = any_cell['cross_tab']
    out = ['# Any-cell scope (report only; the pre-screen-r2 turn-2 arm)', '',
           f"Rows {any_cell['rows']} ({any_cell['rules']}): guarded {any_cell['roots']} ({_pct(any_cell['share'])}): turn 1 {any_cell['turn1']} "
           f"({_pct(any_cell['turn1_share'])}), any-cell turn 2 added {any_cell['turn2_added']} ({_pct(any_cell['turn2_added_share'])}), unavailable or "
           f"error {any_cell['unavailable_or_error']}. Against the live scope: any-cell and live {c['any_cell/live']}, any-cell only "
           f"{c['any_cell/live_clear']}, live only {c['any_cell_clear/live']}, neither {c['any_cell_clear/live_clear']}."]
    block = any_cell['unguarded']
    for scoring in SCORINGS:
        out += ['', f"## Any-cell scope: RR on its unguarded roots ({block['roots']}): {scoring}", '', HEADER]
        out += [_metric_row(name, block['table'][scoring][name]) for name in settings]
    return out + ['']


def reach_markdown(reach, settings):
    """summary.md's reach section (report only)."""
    if not reach:return []
    out = ['# Guard reach (report only; the primary, gate and selection above stay on the any-cell scope)', '',
           f"Turn 2 trips when M_own or M_opp >= {reach['threshold']:g}: M_s = sum_w p_w sum_(a,b) pi_own(a) pi_peer,w(b) q_(w,ab),s. Rows "
           f"{reach['rows']}: exact {reach['exact_rows']}, bounds {reach['rows'] - reach['exact_rows']} (lower: tier A, Psycho Boost under a revealed White Herb holder; upper: every "
           f"tier-C mass guarded), ambiguous (the bounds straddle the threshold) {reach['ambiguous']}, unavailable (fail closed) {reach['unavailable']}.", '',
           '| scope | guarded | turn 1 | reach turn 2 added | tier A | tier C | own | opp | any-cell / reach | any-cell / clear | clear / reach | clear / clear |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for scope in ('lower', 'upper'):
        x = reach[scope];c = x['cross_tab']
        out.append(f"| {scope} | {x['roots']} ({_pct(x['share'])}) | {x['turn1']} ({_pct(x['turn1_share'])}) | {x['turn2_added']} ({_pct(x['turn2_added_share'])}) | "
                   f"{x['tiers']['A']['roots']} ({_pct(x['tiers']['A']['share'])}) | {x['tiers']['C']['roots']} ({_pct(x['tiers']['C']['share'])}) | "
                   f"{x['sides']['own']['roots']} ({_pct(x['sides']['own']['share'])}) | {x['sides']['opp']['roots']} ({_pct(x['sides']['opp']['share'])}) | "
                   f"{c['any_cell/reach']} | {c['any_cell/reach_clear']} | {c['any_cell_clear/reach']} | {c['any_cell_clear/reach_clear']} |")
    for scope, note in (('lower', 'the larger unguarded set'), ('upper', 'the smaller, certainly clean set')):
        block = reach[scope]['unguarded']
        for scoring in SCORINGS:
            out += ['', f"## Reach {scope} scope: RR on its unguarded roots ({block['roots']}; {note}): {scoring}", '', HEADER]
            out += [_metric_row(name, block['table'][scoring][name]) for name in settings]
    return out + ['']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument('--rows', nargs='+', required=True, help="the re-pricer's rows-*.jsonl")
    parser.add_argument('--run-directory', required=True, help='the live screen run directory (results/, telemetry/, owning-wire/)')
    parser.add_argument('--out', required=True, help='a new output directory')
    parser.add_argument('--reply-prior', help='optional JSONL of external reply logits per (ordinal, decision_index, world): adds the '
                                              'reported-only external_br and external_mix operators')
    parser.add_argument('--parity-mode', choices=PARITY_MODES, default='live',
                        help="live (default): parity against the live search; replay: the replay against itself, for a replay on a "
                             "modified tree whose decisions legitimately differ from live")
    args = parser.parse_args(argv)
    summary = score_run(args.rows, args.run_directory, args.out, args.reply_prior, args.parity_mode)
    cover = summary['coverage']
    print(json.dumps(dict(verdict=summary['verdict'], scored=cover['scored'], parity=cover['parity']['rate'],
                          gate=summary['gate']['passed'], chosen=summary['selection']['chosen'], out=str(Path(args.out).resolve()))))
    return 0


if __name__ == '__main__':sys.exit(main())
