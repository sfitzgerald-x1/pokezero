#!/usr/bin/env python3
"""Offline head-to-head search: do its worlds' observation inputs equal the real game's? (V2 offline view parity)

The question. The live DepthN search had input bugs in five classes (shown types zeroed; the root prior read from a
world's encoding; the peer's own team order; peer belief columns such as Pressure PP and hazard credits; carry fields
such as Baton Pass, Trace and recharge). This harness measures which of them reach the OFFLINE path,
`gpu_engine/v2_search_head_to_head.py`, whose worlds come from `device_public_worlds.detach_roots` (a clone of the real
tracker carry plus the owner redaction `_redact_peer_private`), `device_public_worlds.plan_starts` and
`device_public_world_state.materialize_host` (`_rebase_host`), and whose world views are encoded by D's FramePricer
(`device_public_teacher_batch.make_kernel.views`: `executor.observe(state, public, seat, executor.legal(state)[:, seat])`).
Nothing is priced; no model runs on a world.

Games. Champion against champion with the offline path's own machinery: `v2_search_head_to_head.make_step`
(decide/apply), its champion binding (`champion_setup`) and its block/swap construction, both seat orders in lockstep.
Seat A (the searching seat, the owner) is at `a_side` per row exactly as there; both seats play the raw sampled action
(`--policy sample`, the collector's) or their greedy legal argmax (`--policy argmax`). A decision is one of A's searched
rows (live, requested, input-valid: play_cell's rows); it is checked when a seeded hash of (stream, frame, owner) falls
under `--sample-rate`, until this shard's share of `--sample-decisions`.

Checks per sampled decision (owner O, foe F):
1. Owner. For each recipe world (W = 2 * --world-blocks: plan_starts world 0 and 1 of each block seed, as Search.choose),
   materialize it as the pricer does (materialize_host with the frame's world keys) and encode O's view of the world's
   root turn with the pricer's views; compare with decide's own observation of the real root (the Request's root inputs
   and root legal). Every differing (token, column) cell, numeric and categorical, the attention mask and the legal mask
   are recorded, with the pricer's would-be pre-refusal code (make_kernel: 1 own error, 2 legal, 3-7 first differing
   input field, 8 peer error). The same comparison runs on the true world below ("owner.true").
2. Peer. A TRUE world: the opponent fixtures are the real foe team (each member's source row from the constructor
   truth, verified against the real set; its original fixture spread, its real gender), materialized by
   materialize_host from O's real detached root; then re-indexed so physical indices equal the real game's (the
   assignment equals the real physical_to_public; the foe's move slots follow the real slot order) and re-based with
   `_rebase_host(root, real assignment, real active)`. F's view of that world (the reply-prior input) is compared with
   decide's observation of the real root for F. The offline path's deliberate redactions are applied to the real side
   as named tiers, cumulatively: REDACTIONS below. Remaining cells are reported raw and after aligning the bench order.
   A real member with no matching source row, an unknown gender, a materializer refusal, or a foe that is not
   requested, skips the peer check with its reason.
3. Peer order. Per decision: whether the real tracker's foe own_order equals `[active, *rest ascending]` in real
   physical terms (`differs_real_coords`: the share is the foe "back-switch" share; note the two orders also part on a
   FIRST switch-in that skips party order, 0 -> 3 gives [3, 1, 2, 0, 4, 5]); whether a revealed member sits at a
   different request position than in the offline world's actual layout (`revealed_positions_differ`: materializer
   physical order is revealed members by public slot, then the hidden ones); and whether F's true-world view differs
   from the real one ONLY by a permutation of the own (1-6) and switch (17-21) tokens and switch legal lanes.
   The true back-switch share (a member re-entering after being benched, from the actives seen at every decision:
   History) is reported apart from the order-vs-rule share, which also counts out-of-order first entries, faint
   replacements and drag-ins.
4. Five-class verdict with counts and the mechanism's file:line. Carry fields (Baton Pass, Trace, choice lock, Encore,
   traps, Substitute; per side) and peer belief columns (Pressure PP, shown hazard credit, Spikes, Toxic, sleep clause)
   get per-mechanism verdicts counted over COMPARED decisions only, with refused and foe-not-requested counts apart:
   PRESENT / ABSENT / NOT_EXERCISED; a class is ABSENT only when every mechanism was compared clean
   (ABSENT_WHERE_EXERCISED when some never were). The peer's own exact hazard credit and exact damage halves are
   EXCLUDED_BY_DESIGN (named redactions); recharge is NOT_APPLICABLE (no engine state). Legal-lane differences carry
   their cause (PP exhausted -> peer belief; trap/lock/disabled -> carry; otherwise their own 'legal' bucket). Owner
   worlds get the same per-class rollup (types, bench order, belief, carry, legal, other) and the would-be refusal codes.

5. Reply-prior effect of the peer order (with a forward, i.e. always in `run`): at each compared peer decision, one batched
   champion forward per step (the evaluator the offline pricer uses for the reply prior, on actor_inputs) on (a) the true
   world's foe view, (b) the real foe view after the same named redactions, (l) the offline-layout world's foe view
   (the true team in the offline layout: bench encounter order, source move order; not literally a searched world, whose
   hidden members are sampled) and (b2) b again at another batch row. Lanes are aligned by identity (a move
   lane = active member and move, a switch lane = its target member; align()'s mapping, lane_permutation) and compared
   as masked softmaxes over the lanes legal in both: KL(b || a), total variation, top-1 identity flip, max |dp|.
   summary (a pure function of positions.jsonl, so `report` recomputes it): the headline is the order-only group
   (order-caused cells and no residual cell), split by foe back-switch; then median/p95/max KL, flip share and mean TV for
   all compared decisions and by back-switch, order vs the rebase rule, order-caused cells, residual cells (any class)
   and a control row (KL ~ 0). Decisions whose legal sets differ are excluded from every group and shown in their own row
   with the dropped mass (per-decision dropped_b/dropped_a); non-finite values are excluded and counted; the noise floor
   comes from b against b2. Decisions still differing after alignment are dumped per token and member (species,
   ability, item, moves, last move, types on each side) and classified: color_change, forecast, transform_residue,
   conversion, other.

Self-checks: re-encoding the real root through the pricer's views equals decide's observation (both seats); root
source is structural (decide's own_inputs row is O's own observation) and re-checked by A's forward on the Request's
root inputs against decide's raw logits; the true world's foe species/levels/moves/stats/mapping equal the real ones
(notes otherwise); the materializer's own world public equals the true world's re-based public up to sigma (a
materializer public-data defect shows there).

Output (`--out`, a new directory): summary.json, summary.md, positions.jsonl (per decision: address, diffs, flags).
Shards (`--shard i --shards k`) split the block's games (games[i::k] after the `--max-games` prefix) and run as separate
processes on one GPU; `report --inputs <shard dirs> --out <dir>` merges them (the summary is a pure function of the
positions). A shard's games are a valid smaller batch whose initial states need not equal the full block's (as the
head-to-head's --games prefix). Smoke mode: `--max-games` and `--max-steps` (CPU JAX works).

    XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/v2_offline_view_parity.py run --config CONFIG \
        --champion-archive NPZ --champion-archive-sha256 SHA --champion-checkpoint-sha256 SHA \
        --max-games 256 --shard 0 --shards 4 --sample-rate 0.2 --sample-decisions 8000 --out OUT/shard-0
    python scripts/v2_offline_view_parity.py report --inputs OUT/shard-0 OUT/shard-1 OUT/shard-2 OUT/shard-3 --out OUT/merged

All paths are command-line arguments.
"""
import argparse
import collections
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / 'src'):
    if str(_path) not in sys.path:sys.path.insert(0, str(_path))

SCHEMA = 'v2-offline-view-parity.v1'
SELECT_DOMAIN = 'v2-offline-view-parity.select.v1'
CELL_CAP = 64   # cells listed per kind per comparison (counts and keys are always complete)
OBS_FIELDS = ('numeric_features', 'categorical_ids', 'token_type_ids', 'attention_mask', 'legal_action_mask')
KINDS = dict(numeric_features='numeric', categorical_ids='categorical', token_type_ids='token_type',
    attention_mask='attention', legal_action_mask='legal')
# The FramePricer's root-input comparison order (device_public_teacher_batch.SERIAL_INPUT_ORDER, history_mask always ones).
PRE_INPUT_ORDER = ('numeric_features', 'categorical_ids', 'token_type_ids', 'attention_mask', 'history_mask')
TYPE_TOKENS = tuple(range(1, 13)) + tuple(range(23, 29))   # own, opponent and belief tokens carry shown/current types
TYPE_COLUMNS = (4, 5)   # CATEGORY_TYPE_1, CATEGORY_TYPE_2 (pokezero.showdown; checked in the tests)
LANE_CONSTANT_COLUMNS = (1, 2, 3)   # action-token categories fixed per lane: SECONDARY, ROLE, SLOT
SECTIONS = (('field', 0, 1), ('own', 1, 7), ('opponent', 7, 13), ('move_actions', 13, 17), ('switch_actions', 17, 22),
    ('stats', 22, 23), ('belief', 23, 29))
# The offline path's deliberate redactions of the peer's (foe's) view, applied cumulatively to the real side.
REDACTIONS = (
    ('peer_damage_exact', 'gpu_engine/device_public_worlds.py:126-145 (_redact_peer_private): the foe-victim halves of the '
        'damage history and the exact supplements (damage_exact_taken[foe], damage_exact_dealt[owner]) are zeroed; '
        'declared in device_public_teacher.py:27-35 (own.LAST_DAMAGE_TAKEN, foe.LAST_DAMAGE_DEALT)'),
    ('peer_exact_hazard', 'gpu_engine/engine_device_public_port.py:152-155 (redact_for_owner): the foe\'s exact hazard '
        'history is zeroed (field.SELF_HAZARD_CREDIT of the reply prior; device_public_teacher.py:30)'),
    ('foe_hp_public_percent', 'gpu_engine/device_public_world_state.py:286-299 (materialize): each revealed foe member\'s HP '
        'is max(1, round(shown * maxHP / 100)), 0 when shown 0 (status cleared), from the public percent'),
)
CLASSES = ('types', 'root_source', 'peer_order', 'peer_belief_columns', 'carry_fields')
MECHANISMS = dict(
    types='gpu_engine/device_public_worlds.py:162-169 and :262-264 (world public = clone of the real carry; readout '
        'shown_types, engine_device_trackers.py:297); device_public_world_state.py:304-306 (world foe types = shown types, '
        'else the source species types)',
    root_source='gpu_engine/v2_search_head_to_head.py:138-141 (own_inputs = observations[a_side]), :342-343 (Request root_logits, '
        'root_legal, root_inputs = the actor\'s own row), :572 (logits = raw_logits[rows, a_side]); '
        'device_public_teacher_batch.py:87-94 (a world whose owner encoding differs from root_inputs is pre-refused)',
    peer_order='gpu_engine/device_public_world_state.py:105 and :153 (_rebase/_rebase_host: foe own_order = [active, *rest '
        'ascending] in world physical terms; materializer physical order = revealed by public slot, then hidden, :223 and '
        ':242) against the real tracker\'s swap-with-slot-0 per installation (engine_device_public_hooks.py:175-179; legacy '
        'engine_device_trackers.py:215-218, initial swap :189)',
    peer_belief_columns='gpu_engine/device_public_worlds.py:136-145 (the world public carry is the real carry, owner-redacted); '
        'engine_device_public_hooks.py:361-373 (Pressure charged from the receiving observer\'s own ability); '
        'device_public_world_state.py:317-332 (foe PP = max PP - public use count)',
    carry_fields='gpu_engine/device_public_worlds.py:36-49 (only owner cells are copied); device_public_world_state.py:210-211 '
        '(owner forced_switch/baton_pass/entry_pending refused), :307-309 (Trace: borrowed ability), :345-353 (public '
        'volatile bag: others refused); recharge: no engine state (gpu_engine/transition.py, mechanics.py, state.py have none)',
)


def log(message, **values):
    print(json.dumps(dict(message=message, time=time.strftime('%H:%M:%S'), **values), default=str), flush=True)


# ---------------------------------------------------------------------------------------------- names and cells

def section(token):
    for name, start, stop in SECTIONS:
        if start <= token < stop:return name
    raise ValueError('token out of range: %r' % token)


_NAMES = {}


def _names():
    if not _NAMES:
        from pokezero import showdown as reference
        numeric = {}
        for name, value in vars(reference).items():
            if name.startswith('NUMERIC_') and isinstance(value, int):
                try:numeric.setdefault(reference.v4_numeric_index(value), name[8:])
                except (ValueError, KeyError):pass
        for base, width in (('OPP_MOVE_PP_OFFSET', 16), ('OPP_MOVE_PP_VALID_OFFSET', 16), ('STAT_WEATHER_REVEAL_OFFSET', 6)):
            start = reference.v4_numeric_index(getattr(reference, 'NUMERIC_' + base))
            for i in range(width):numeric[start + i] = base.replace('_OFFSET', '') + '_' + str(i)
        r = reference;categorical = {}
        for name in ('PRIMARY', 'SECONDARY', 'ROLE', 'SLOT', 'TYPE_1', 'TYPE_2', 'MOVE_CATEGORY', 'MOVE_EFFECT', 'MOVE_PRIORITY',
                'LAST_USED_MOVE', 'TRACED_ABILITY'):
            categorical[getattr(r, 'CATEGORY_' + name)] = name
        for name, start, stop in (('BELIEF_ABILITY', r.CATEGORY_BELIEF_ABILITY_OFFSET, r.CATEGORY_BELIEF_ITEM_OFFSET),
                ('BELIEF_ITEM', r.CATEGORY_BELIEF_ITEM_OFFSET, r.CATEGORY_BELIEF_MOVE_OFFSET),
                ('BELIEF_MOVE', r.CATEGORY_BELIEF_MOVE_OFFSET, r.CATEGORY_VOLATILE_OFFSET),
                ('VOLATILE', r.CATEGORY_VOLATILE_OFFSET, r.CATEGORY_VOLATILE_OFFSET + 6)):
            for column in range(start, stop):categorical.setdefault(column, '%s_%d' % (name, column - start))
        _NAMES.update(numeric=numeric, categorical=categorical)
    return _NAMES


def column_name(token, kind, column):
    sec = section(token)
    if kind == 'numeric':
        if sec == 'belief':
            return ('posterior_weight_%d' % column if column < 64 else 'prior_weight_%d' % (column - 64) if column < 128 else
                ('prior_count', 'posterior_count', 'reason', 'pin_conflict')[column - 128])
        return _names()['numeric'].get(column, 'col%d' % column)
    if kind == 'categorical':return _names()['categorical'].get(column, 'cat%d' % column)
    return ''


def cell_key(kind, index):
    """(section, kind, name) of one differing cell; index is (token, column), (token,) or (lane,)."""
    if kind == 'legal':return ('legal', 'legal', 'lane_%d' % index[0])
    token = int(index[0])
    if kind in ('numeric', 'categorical'):return (section(token), kind, column_name(token, kind, int(index[1])))
    return (section(token), kind, '')


def _py(value):
    if isinstance(value, (np.floating, float)):return round(float(value), 6)
    if isinstance(value, (np.bool_, bool)):return int(value)
    return int(value)


def diff(world, real, *, cap=CELL_CAP):
    """Every differing cell of two observations (dicts of OBS_FIELDS arrays without a batch axis).

    Returns {kind: {count, cells: [[*index, world, real], ...] (first `cap`)}, total, keys: sorted unique
    'section:kind:name', exact: set of (kind, *index)} -- `exact` is not serialized (see `serial`)."""
    out, keys, exact, total = {}, set(), set(), 0
    for field, kind in KINDS.items():
        a, b = np.asarray(world[field]), np.asarray(real[field])
        if a.shape != b.shape:raise ValueError('observation geometry differs: ' + field)
        where = np.argwhere(a != b)
        out[kind] = dict(count=int(len(where)), cells=[[*map(int, i), _py(a[tuple(i)]), _py(b[tuple(i)])] for i in where[:cap]])
        total += len(where)
        for i in where:
            keys.add(':'.join(cell_key(kind, tuple(int(x) for x in i))));exact.add((kind,) + tuple(int(x) for x in i))
    # Zeroed shown types over the FULL arrays (not the capped listing): world 0, real nonzero.
    w, r = np.asarray(world['categorical_ids']), np.asarray(real['categorical_ids'])
    tokens, columns = list(TYPE_TOKENS), list(TYPE_COLUMNS)
    zeroed = int(np.sum((w[tokens][:, columns] == 0) & (r[tokens][:, columns] != 0)))
    out.update(total=int(total), keys=sorted(keys), exact=exact, zeroed_types=zeroed)
    return out


def serial(result):
    """A diff without its exact cell set and without empty kinds (positions.jsonl stays small)."""
    return {k: v for k, v in result.items() if k != 'exact' and not (isinstance(v, dict) and v.get('count') == 0)}


# Legal-lane causes and the class each belongs to (legal lanes are not all carry fields).
LEGAL_CAUSE_CLASS = dict(pp='peer_belief_columns', trap='carry_fields', disabled='carry_fields', lock='carry_fields',
    hp='other', move_id='other', foe_ability_trap='legal', unknown='legal')


def legal_causes(lanes, world, real, seat, real_order, active):
    """Why each differing legal lane differs: world/real are one-row host BattleStates whose `seat` side uses the same
    physical coordinates; lanes are request lanes of the REAL order (0-3 the active's move slots, 4-8 switch members)."""
    s, a, out = int(seat), int(active), {}
    members = switch_members(real_order, a) if _is_perm(real_order) and a in [int(x) for x in real_order] else None
    def active_of(state, side):return int(np.asarray(state.active)[0, side])
    for lane in lanes:
        lane = int(lane)
        if lane < 4:
            w_pp, r_pp = int(world.pp[0, s, a, lane]), int(real.pp[0, s, a, lane])
            if (w_pp > 0) != (r_pp > 0):cause = 'pp'
            elif bool(world.disabled[0, s, a, lane]) != bool(real.disabled[0, s, a, lane]):cause = 'disabled'
            elif any(int(getattr(world, n)[0, s, a]) != int(getattr(real, n)[0, s, a]) for n in ('choice_move', 'encore_move', 'charge_move')):
                cause = 'lock'
            elif int(world.moves[0, s, a, lane]) != int(real.moves[0, s, a, lane]):cause = 'move_id'
            else:cause = 'unknown'
        elif members is None:cause = 'unknown'
        else:
            p = members[lane - 4]
            linked = lambda st:(int(st.trap_source[0, s, a]) >= 0, int(st.volatile[0, s, a]) & 8 != 0, bool(st.trapped[0, s]))
            foe_ability = lambda st:int(st.ability[0, 1 - s, active_of(st, 1 - s)])
            if (int(world.hp[0, s, p]) > 0) != (int(real.hp[0, s, p]) > 0):cause = 'hp'
            elif linked(world) != linked(real):cause = 'trap'
            elif int(world.charge_move[0, s, a]) != int(real.charge_move[0, s, a]):cause = 'lock'
            elif foe_ability(world) != foe_ability(real):cause = 'foe_ability_trap'
            else:cause = 'unknown'
        out[lane] = cause
    return out


def with_causes(result, causes):
    """A diff whose legal-lane keys name their cause: 'legal:legal:lane_<k>/<cause>'."""
    keys = [k for k in result['keys'] if not k.startswith('legal:')]
    keys += ['legal:legal:lane_%d/%s' % (lane, cause) for lane, cause in sorted(causes.items())]
    out = dict(result, keys=sorted(keys))
    if 'legal' in out:out['legal'] = dict(out['legal'], causes={str(k): v for k, v in sorted(causes.items())})
    return out


def legal_lanes(result):
    return sorted({int(i[1]) for i in result['exact'] if i[0] == 'legal'})


class History:
    """Per row and side, from the actives seen at every lockstep decision: which members have been active (a member
    re-entering after being benched is a back-switch) and whether the current active arrived by Baton Pass (the side's
    baton_pass flag was set at the decision before its active changed). Switches inside one engine step are not seen."""
    def __init__(self, active):
        active = np.asarray(active);rows = len(active);self.prev = active.copy()
        self.ever = np.zeros((rows, 2, 6), bool);self.ever[np.arange(rows)[:, None], np.arange(2)[None, :], active] = True
        self.back = np.zeros((rows, 2), bool);self.pending = np.zeros((rows, 2), bool)
        self.baton_in = np.full((rows, 2), -1, np.int32);self.switch_ins = np.zeros((rows, 2), np.int32)

    def update(self, active, baton_pass):
        active = np.asarray(active);changed = active != self.prev
        for r, s in zip(*np.nonzero(changed)):
            p = int(active[r, s])
            if self.ever[r, s, p]:self.back[r, s] = True
            self.ever[r, s, p] = True;self.switch_ins[r, s] += 1
            self.baton_in[r, s] = p if self.pending[r, s] else -1
        self.pending = np.where(changed, False, self.pending) | np.asarray(baton_pass, bool)
        self.prev = active.copy()

    def flags(self, row, side):
        return dict(back_switched=bool(self.back[row, side]), baton_in=bool(self.baton_in[row, side] >= 0 and self.baton_in[row, side] == self.prev[row, side]),
            switch_ins=int(self.switch_ins[row, side]), ever_active=[int(p) for p in np.flatnonzero(self.ever[row, side])])


def numeric_max_abs(world, real):
    a, b = np.asarray(world['numeric_features'], np.float64), np.asarray(real['numeric_features'], np.float64)
    return float(np.max(np.abs(a - b))) if a.size else 0.


# ---------------------------------------------------------------------------------------------- order

def rule_order(active):
    """The offline foe own_order: [active, *rest ascending] (device_public_world_state.py:105, :153)."""
    active = int(active)
    return [active] + [i for i in range(6) if i != active]


def offline_layout_order(active, physical_to_public):
    """The offline world's actual foe request order, in real identities: the materializer puts revealed members first in
    public-slot order and hidden members after (device_public_world_state.py:223, :242), then _rebase applies the rule;
    hidden members are listed here by real physical index (a sampled world holds other species there)."""
    active = int(active);mapping = [int(x) for x in physical_to_public]
    revealed = [p for _, p in sorted((s, p) for p, s in enumerate(mapping) if s >= 0)]
    return [active] + [p for p in revealed if p != active] + [p for p in range(6) if mapping[p] < 0 and p != active]


def revealed_positions_differ(real_order, layout, physical_to_public):
    real_order, layout = [int(x) for x in real_order], [int(x) for x in layout]
    return any(real_order.index(p) != layout.index(p) for p in range(6) if int(physical_to_public[p]) >= 0)


def _is_perm(order):
    return sorted(int(x) for x in order) == list(range(6))


def switch_members(order, active):
    """The encoder's five switch lanes as physical members: request positions other than the active's
    (engine_device_observation.py:322-324)."""
    order = [int(x) for x in order];at = order.index(int(active))
    return [order[k] for k in range(6) if k != at]


def align(world_obs, world_order, real_order, active, *, move_map=None):
    """world_obs re-indexed so every own token (1-6), switch token (17-21) and switch legal lane (4-8) holds the same
    physical member as in real_order (and, with move_map {world slot: real slot}, every move token 13-16 and move legal
    lane the same move). Orders are physical member lists in the SAME coordinates (map world coordinates first)."""
    w = {k: np.array(world_obs[k], copy=True) for k in OBS_FIELDS}
    at = {int(p): k for k, p in enumerate(world_order)}
    tokens = ('numeric_features', 'categorical_ids', 'token_type_ids', 'attention_mask')
    for k, p in enumerate(real_order):
        for f in tokens:w[f][1 + k] = world_obs[f][1 + at[int(p)]]
    lanes = {p: i for i, p in enumerate(switch_members(world_order, active))}
    for i, p in enumerate(switch_members(real_order, active)):
        for f in tokens:w[f][17 + i] = world_obs[f][17 + lanes[p]]
        w['legal_action_mask'][4 + i] = world_obs['legal_action_mask'][4 + lanes[p]]
    if move_map:
        for slot, target in move_map.items():
            for f in tokens:w[f][13 + target] = world_obs[f][13 + slot]
            w['legal_action_mask'][target] = world_obs['legal_action_mask'][slot]
    # Action lanes keep their lane constants (secondary kind, role, slot: engine_device_observation.py:338-339).
    w['categorical_ids'][13:22, list(LANE_CONSTANT_COLUMNS)] = np.asarray(world_obs['categorical_ids'])[13:22, list(LANE_CONSTANT_COLUMNS)]
    return w


# ---------------------------------------------------------------------------------------------- redactions

def redaction_tiers(cloned, redacted):
    """The real foe-view carries, one per cumulative named redaction, from the real one-row clone and the same row
    through device_public_worlds._redact_peer_private (only its data redactions; the identity bookkeeping it clears,
    physical_to_public/active_physical/own_order of the foe, is what _rebase re-sets in a world)."""
    tier1 = cloned._replace(damage_history=redacted.damage_history, damage_exact_taken=redacted.damage_exact_taken,
        damage_exact_dealt=redacted.damage_exact_dealt)
    tier2 = tier1._replace(public_port=redacted.public_port)
    return dict(raw=cloned, peer_damage_exact=tier1, peer_exact_hazard=tier2)


def foe_hp_from_public_percent(hp, max_hp, status, physical_to_public, hp_percent):
    """device_public_world_state.py:286-299 on one real side: every revealed member's HP becomes
    max(1, round(shown * maxHP / 100)) if shown else 0, and a shown-0 member's status 0. Host int arrays [6]."""
    hp, status = np.array(hp, copy=True), np.array(status, copy=True)
    for p in range(6):
        slot = int(physical_to_public[p])
        if slot < 0:continue
        shown, maximum = int(hp_percent[slot]), int(max_hp[p])
        hp[p] = max(1, round(shown * maximum / 100)) if shown else 0
        if not shown:status[p] = 0
    return hp, status


def redact_state_hp(state, foe, physical_to_public, hp_percent):
    """foe_hp_from_public_percent applied to a one-row host BattleState."""
    hp, status = np.array(state.hp, copy=True), np.array(state.status, copy=True)
    hp[0, foe], status[0, foe] = foe_hp_from_public_percent(hp[0, foe], state.max_hp[0, foe], status[0, foe],
        physical_to_public, hp_percent)
    return state._replace(hp=hp, status=status)


# ---------------------------------------------------------------------------------------------- the true world

class Skip(Exception):
    """A decision the peer check cannot compare; the message is its recorded reason."""


def _hp_name(name, hp_type):
    from pokezero.dex import normalize_id
    value = normalize_id(name)
    return 'hiddenpower' + hp_type if value == 'hiddenpower' and hp_type else value


def member_variant(source, real, p):
    """(variant index, variant, {variant slot: real slot}) for real foe member p.

    `real` holds host arrays of the foe side: species/level/base_ability/item/hidden_power_type/gender [6],
    moves [6, 4], truth_probabilities [6, 64], truth_known [6]. Candidates are the constructor truth's source rows
    (v5_device_rollout.constructor_truth), each verified against the real set; without truth, every source row of the
    species that matches (level, ability, move set and a still-held item)."""
    from pokezero.dex import normalize_id
    vocab = source.vocabulary
    sid = int(real['species'][p])
    if not 0 < sid < len(vocab['species']):raise Skip('real foe species id outside the engine namespace')
    name = vocab['species'][sid]
    universe = source.original.universe_for(name)
    if universe is None or not universe.variants:raise Skip('real foe species has no source universe')
    hpt = int(real['hidden_power_type'][p])
    hp_type = normalize_id(vocab['types'][hpt]) if 0 < hpt < len(vocab['types']) else ''
    slots = [(k, _hp_name(vocab['moves'][int(m)], hp_type)) for k, m in enumerate(real['moves'][p]) if int(m) > 0]
    ability = normalize_id(vocab['abilities'][int(real['base_ability'][p])]) if int(real['base_ability'][p]) > 0 else ''
    item = normalize_id(vocab['items'][int(real['item'][p])]) if int(real['item'][p]) > 0 else None
    known = bool(real['truth_known'][p])
    candidates = np.flatnonzero(np.asarray(real['truth_probabilities'][p]) > 0) if known else range(len(universe.variants))
    def fits(v):
        return (v.level == int(real['level'][p]) and normalize_id(v.ability) == ability
            and sorted(_hp_name(m, hp_type) for m in v.moves) == sorted(n for _, n in slots)
            and (item is None or normalize_id(v.item) == item))
    matched = [int(k) for k in candidates if int(k) < len(universe.variants) and fits(universe.variants[int(k)])]
    if not matched:raise Skip('real foe set has no matching source row' + ('' if known else ' (no constructor truth)'))
    if not known and len({normalize_id(universe.variants[k].item) for k in matched}) > 1:
        raise Skip('original item ambiguous without constructor truth')
    real_names = tuple(n for _, n in slots)
    index = next((k for k in matched if tuple(_hp_name(m, hp_type) for m in universe.variants[k].moves) == real_names), matched[0])
    variant = universe.variants[index]
    where = {n: k for k, n in slots}
    move_map = {slot: where[_hp_name(m, hp_type)] for slot, m in enumerate(variant.moves)}
    return index, variant, move_map, name


def true_world_start(root, real, source, recipe):
    """The Start that materializes the REAL foe team from O's root: fixtures in the materializer's physical order
    (revealed members by public slot, then hidden ones by real physical index), each revealed one with its source row.
    Returns (start, sigma: materializer physical -> real physical, move_maps per real physical, assignment, active)."""
    from pokezero.determinization import _fixture_from_variant_payload
    from gpu_engine import device_public_worlds as worlds
    mapping = [int(x) for x in real['physical_to_public']]
    revealed = sorted((s, p) for p, s in enumerate(mapping) if s >= 0)
    present = tuple(int(x) for x in np.flatnonzero(np.asarray(root.view['present'][0])))
    if tuple(s for s, _ in revealed) != present:raise Skip('real foe mapping and public present slots differ')
    sigma = [p for _, p in revealed] + [p for p in range(6) if mapping[p] < 0]
    fixtures, indices, maps = [], {}, {}
    for p in sigma:
        index, variant, move_map, name = member_variant(source, real, p)
        code = int(real['gender'][p])
        if code not in (0, 1, 2):raise Skip('real foe gender unknown')
        fixture = _fixture_from_variant_payload(variant.to_summary(), fallback_species=name, set_source=source.original,
            gender={0: 'N', 1: 'M', 2: 'F'}[code])
        if fixture is None:raise Skip('original fixture spread unavailable for a real foe set')
        fixtures.append(fixture);indices[p] = index;maps[p] = move_map
    start = worlds.Start(0, recipe.world_seed(root.address, 0), tuple(fixtures),
        tuple((s, indices[p]) for s, p in revealed), ())
    return start, tuple(sigma), maps, {s: p for s, p in revealed}


def realign(world, sigma, move_maps, owner):
    """The materialized world (materializer physical coordinates, variant move slots) re-indexed to real physical
    indices and real move slots: a host BattleState. Value cells naming a foe physical index or move slot follow."""
    from gpu_engine.state import BattleState
    foe = 1 - owner;st = world.state;sig = np.asarray(sigma, np.int32)
    arrays = {}
    for name in BattleState._fields:
        a = np.asarray(getattr(st, name));b = np.array(a, copy=True)
        if a.ndim >= 3 and a.shape[1:3] == (2, 6):b[0, foe, sig] = a[0, foe, :]
        arrays[name] = b
    for name in ('moves', 'pp', 'disabled'):
        a = arrays[name];b = np.array(a, copy=True)
        for p in range(6):
            b[0, foe, p] = np.zeros_like(a[0, foe, p])
            for slot, target in move_maps[p].items():b[0, foe, p, target] = a[0, foe, p, slot]
        arrays[name] = b
    active = int(sig[int(st.active[0, foe])]);arrays['active'][0, foe] = active
    trap = arrays['trap_source'][0, owner]
    arrays['trap_source'][0, owner] = np.where(trap >= 0, sig[np.clip(trap, 0, 5)], trap)
    kind, index = int(st.last_action_kind[0, foe]), int(st.last_action_index[0, foe])
    if kind == 1 and index >= 0:arrays['last_action_index'][0, foe] = sig[index]
    elif kind == 2 and index >= 0:arrays['last_action_index'][0, foe] = move_maps[active].get(index, index)
    return BattleState(**arrays), active


def build_true_world(root, real, source, materializer, recipe, keys):
    """(materialized world in the offline layout, realigned state, its _rebase_host public, sigma, move_maps, notes)."""
    from gpu_engine import device_public_world_state as states
    owner = root.address.owner;foe = 1 - owner
    start, sigma, move_maps, assignment = true_world_start(root, real, source, recipe)
    world = states.materialize_host(root, start, materializer, recipe, keys=keys)
    if world.state is None:raise Skip('materializer refused: ' + ','.join(world.refusals)[:200])
    state, active = realign(world, sigma, move_maps, owner)
    public = states._rebase_host(root, assignment, active, state=state)
    notes = []
    for name in ('species', 'level', 'base_ability', 'gender'):
        if not np.array_equal(getattr(state, name)[0, foe], real[name]):notes.append('true-world %s differs from the real team' % name)
    if not np.array_equal(state.moves[0, foe], real['moves']):notes.append('true-world move ids differ from the real slots')
    for name in ('stats', 'max_hp', 'hidden_power_type', 'hidden_power_power'):
        if name in real and not np.array_equal(getattr(state, name)[0, foe], real[name]):
            notes.append('true-world %s differs from the real team (source fixture spread/items)' % name)
    if active != int(real['active']):notes.append('true-world active differs from the real active')
    if not np.array_equal(np.asarray(public.physical_to_public)[0, foe], np.asarray(real['physical_to_public'])):
        notes.append('true-world physical_to_public differs from the real mapping')
    return world, state, public, sigma, move_maps, notes


def public_differences(world_public, public, sigma, owner):
    """Fields where the materializer's own world public (materializer coordinates) differs from the true world's public
    re-based from the root in real coordinates: every tracker field must be equal, except that the foe's
    physical_to_public/active_physical must agree through sigma and its own_order is the rebase rule in each
    coordinate system (compared as the peer order, not here). A materializer public-data defect shows here."""
    foe = 1 - owner;sig = np.asarray(sigma);out = []
    for name in type(public)._fields:
        a, b = getattr(world_public, name), getattr(public, name)
        if name == 'physical_to_public':
            a, b = np.asarray(a), np.asarray(b)
            same = np.array_equal(a[0, owner], b[0, owner]) and np.array_equal(a[0, foe], b[0, foe][sig])
        elif name == 'active_physical':
            a, b = np.asarray(a), np.asarray(b)
            same = a[0, owner] == b[0, owner] and (a[0, foe] < 0 and b[0, foe] < 0 or int(sig[int(a[0, foe])]) == int(b[0, foe]))
        elif name == 'own_order':
            same = np.array_equal(np.asarray(a)[0, owner], np.asarray(b)[0, owner])
        else:
            import jax
            la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
            same = len(la) == len(lb) and all(np.asarray(x).dtype == np.asarray(y).dtype and np.array_equal(x, y) for x, y in zip(la, lb))
        if not same:out.append(name)
    return out


# ---------------------------------------------------------------------------------------------- encoding

_ENCODERS = {}


def make_encoder(executor):
    """make_encoder_uncached, one per executor (its compiled programs are reused across shards of one process)."""
    held = _ENCODERS.get(id(executor))
    if held is None or held[0] is not executor:
        held = _ENCODERS[id(executor)] = (executor, make_encoder_uncached(executor))
    return held[1]


def make_encoder_uncached(executor):
    """The FramePricer's views (device_public_teacher_batch.make_kernel.views) for one seat per world, mapped over
    worlds of one row each: executor.observe(state, public, [seat], executor.legal(state)[:, seat])."""
    import jax
    import jax.numpy as j
    def one(state, public, seat):
        masks = executor.legal(state)
        o = executor.observe(state, public, j.reshape(seat, (1,)).astype(j.int32), masks[:, seat]).observation
        return dict(numeric_features=o.numeric_features[0], categorical_ids=o.categorical_ids[0], token_type_ids=o.token_type_ids[0],
            attention_mask=o.attention_mask[0], legal_action_mask=o.legal_action_mask[0], error=o.error[0], requested=j.any(masks[0, seat]))
    run = jax.jit(jax.vmap(one))
    def encode(items):
        """items: [(state, public, seat)] with host leaves of batch 1. Returns one host dict per item."""
        if not items:return []
        count = len(items);bucket = max(64, 1 << (count - 1).bit_length())
        padded = list(items) + [items[0]] * (bucket - count)
        def stack(*xs):
            first = xs[0]
            if any(np.asarray(x).dtype != np.asarray(first).dtype or np.shape(x) != np.shape(first) for x in xs):
                raise ValueError('world/real carry leaves differ in dtype or shape')
            return np.stack([np.asarray(x) for x in xs])
        states = jax.tree.map(stack, *[x[0] for x in padded]);publics = jax.tree.map(stack, *[x[1] for x in padded])
        seats = np.asarray([x[2] for x in padded], np.int32)
        out = jax.device_get(run(states, publics, seats))
        return [{k: v[i] for k, v in out.items()} for i in range(count)]
    return encode


def pre_code(own, peer, root_inputs, root_legal):
    """make_kernel's pre-refusal code for a world (device_public_teacher_batch.py:87-95; 9/10 need a forward)."""
    if int(own['error']) != 0:return 1
    if not np.array_equal(np.asarray(own['legal_action_mask'], bool), np.asarray(root_legal, bool)):return 2
    for index, key in enumerate(PRE_INPUT_ORDER):
        if key == 'history_mask':continue   # actor_inputs' history_mask is ones on both sides
        if not np.array_equal(np.asarray(own[key]), np.asarray(root_inputs[key])):return 3 + index
    if peer is not None and int(peer['error']) != 0:return 8
    return 0


# ---------------------------------------------------------------------------------------------- selection and shards

def selected(seed, stream, frame, owner, rate):
    """A seeded, shard-independent Bernoulli choice of one decision."""
    if rate >= 1:return True
    digest = hashlib.sha256(json.dumps([SELECT_DOMAIN, int(seed), int(stream), int(frame), int(owner)]).encode()).digest()
    return int.from_bytes(digest[:8], 'big') / 2.**64 < rate


def shard_games(games, shard, shards):
    if type(shards) is not int or shards < 1 or type(shard) is not int or not 0 <= shard < shards:
        raise ValueError('shard index 0 <= i < k required')
    return tuple(games[shard::shards])


def shard_quota(total, shards):
    return int(math.ceil(total / shards))


# ---------------------------------------------------------------------------------------------- one step's checks

@dataclass
class Setup:
    """What a shard plays and checks. `fresh()` returns the initial carry (both seat orders); rows are lane-major:
    the first n rows hold A in seat 0, the next n the swapped copy."""
    executor: object
    source: object
    materializer: object
    recipes: tuple
    decide: object
    apply: object
    params: object
    a_side: object
    game: object
    seat_order: object
    fresh: object
    meta: dict
    forward: object = None   # A's forward(params, inputs) -> v5_model.Output, for the root-logit re-check (optional)


# Mechanisms whose exercise is counted per decision, by side; a class verdict is ABSENT only where they were compared.
CARRY_MECHANISMS = ('baton_pass', 'trace', 'choice_lock', 'encore', 'trapped', 'substitute')
BELIEF_MECHANISMS = ('pressure_pp', 'hazard_credit_opp', 'spikes', 'toxic', 'sleep_clause')
EXCLUDED_BY_DESIGN = dict(hazard_credit_self='peer_exact_hazard', damage_exact='peer_damage_exact')
ROOT_LOGIT_TOLERANCE = .05   # |re-forward - decide's raw logits| above this is a root-source mismatch (bf16 batch-shape noise is far below)


# ---------------------------------------------------------------------------------------------- reply-prior effect size

def lane_permutation(world_order, real_order, active, move_map=None):
    """perm[real lane] = the world lane holding the same identity: align()'s mapping for 9-lane vectors (move lanes by
    {world slot: real slot}, switch lanes by member through each side's request order)."""
    perm = list(range(9))
    for slot, target in (move_map or {}).items():perm[int(target)] = int(slot)
    lanes = {p: i for i, p in enumerate(switch_members(world_order, active))}
    for i, p in enumerate(switch_members(real_order, active)):perm[4 + i] = 4 + lanes[p]
    return perm


def masked_log_softmax(logits, legal):
    """log softmax over the legal lanes (float64); -inf elsewhere."""
    x = np.where(np.asarray(legal, bool), np.asarray(logits, np.float64), -np.inf)
    top = np.max(x)
    return x - (top + np.log(np.sum(np.exp(x - top))))


def effect(a_logits, a_legal, b_logits, b_legal):
    """How far reply prior a is from b over identity-aligned lanes: KL(b || a), total variation, top-1 flip and max |dp|,
    on the lanes legal in both (support_differs when the legal sets differ). None without a common legal lane."""
    a_legal, b_legal = np.asarray(a_legal, bool), np.asarray(b_legal, bool);support = a_legal & b_legal
    if not support.any():return None
    la, lb = masked_log_softmax(a_logits, support), masked_log_softmax(b_logits, support)
    pa, pb = np.exp(la), np.exp(lb)
    kl = float(np.sum(pb[support] * (lb[support] - la[support])))
    top_a, top_b = int(np.argmax(np.where(support, pa, -1.))), int(np.argmax(np.where(support, pb, -1.)))
    # The mass the common support drops: b's own full prior on lanes illegal in a, and a's on lanes illegal in b.
    dropped_b = float(np.sum(np.exp(masked_log_softmax(b_logits, b_legal))[b_legal & ~a_legal]))
    dropped_a = float(np.sum(np.exp(masked_log_softmax(a_logits, a_legal))[a_legal & ~b_legal]))
    return dict(kl=kl, tv=float(.5 * np.sum(np.abs(pa - pb))), top1_flip=top_a != top_b, max_abs_dp=float(np.max(np.abs(pa - pb))),
        top_a=top_a, top_b=top_b, support_differs=not np.array_equal(a_legal, b_legal), legal=int(support.sum()),
        dropped_b=dropped_b, dropped_a=dropped_a, pa=[round(float(x), 6) for x in pa], pb=[round(float(x), 6) for x in pb])


def lane_identities(state, side, order, active, vocab):
    """The identity of each real request lane: a move lane is (active member, move), a switch lane its target member."""
    name = lambda kind, i:vocab[kind][int(i)] if 0 < int(i) < len(vocab[kind]) else ''
    a = int(active);out = ['move:p%d:%s' % (a, name('moves', state.moves[0, side, a, k])) for k in range(4)]
    members = switch_members(order, a) if _is_perm(order) else [None] * 5
    out += ['switch:p%s:%s' % (p, name('species', state.species[0, side, p]) if p is not None else '') for p in members]
    return out


TYPE_CHANGE_SOURCES = (('color_change', ('colorchange',)), ('forecast', ('forecast',)))


def describe_residual(exact, real_state, world_state, clone, foe, real_order, vocab):
    """The cells still differing after the redactions and the order alignment, by token, with the member they describe
    (species, ability, item, moves, the last move that may have triggered a type change, types on each side) and a
    classification: color_change, forecast, transform_residue, conversion or other."""
    from pokezero.dex import normalize_id
    name = lambda kind, i:vocab[kind][int(i)] if 0 < int(i) < len(vocab[kind]) else ''
    types = lambda st, side, p:[name('types', x) for x in np.asarray(st.types)[0, side, p]]
    tokens = collections.defaultdict(set)
    for cell in exact:
        if cell[0] in ('numeric', 'categorical', 'attention', 'token_type'):tokens[int(cell[1])].add(':'.join(cell_key(cell[0], cell[1:])))
    owner = 1 - foe;out = []
    mapping = np.asarray(clone.physical_to_public)[0]
    for token in sorted(tokens):
        sec = section(token);entry = dict(token=token, section=sec, cells=sorted(tokens[token]))
        side = p = None
        if sec == 'own':side, p = foe, int(real_order[token - 1])
        elif sec in ('opponent', 'belief'):
            slot = token - (7 if sec == 'opponent' else 23);hit = np.flatnonzero(mapping[owner] == slot)
            if len(hit):side, p = owner, int(hit[0])
        if p is not None:
            slot = int(mapping[side, p])
            moves = [normalize_id(name('moves', m)) for m in np.asarray(real_state.moves)[0, side, p] if int(m)]
            ability, base = normalize_id(name('abilities', real_state.ability[0, side, p])), normalize_id(name('abilities', real_state.base_ability[0, side, p]))
            last = normalize_id(name('moves', real_state.last_move[0, side, p]))
            shown = [] if slot < 0 or clone.shown_types is None else [name('types', x) for x in np.asarray(clone.shown_types)[0, side, slot]]
            public_last = '' if slot < 0 or clone.last_moves is None else name('moves', np.asarray(clone.last_moves)[0, side, slot])
            kind = next((label for label, abilities in TYPE_CHANGE_SOURCES if ability in abilities or base in abilities), None)
            if kind is None:
                species = normalize_id(name('species', real_state.species[0, side, p]))
                kind = ('transform_residue' if 'transform' in moves or last == 'transform' or species == 'ditto' else
                    'conversion' if any(m in ('conversion', 'conversion2', 'camouflage') for m in moves) else 'other')
            entry.update(side='peer' if side == foe else 'owner', member=p, public_slot=slot, species=name('species', real_state.species[0, side, p]),
                ability=ability, base_ability=base, item=name('items', real_state.item[0, side, p]), moves=moves, last_move=last,
                public_last_move=public_last, real_types=types(real_state, side, p), world_types=types(world_state, side, p),
                base_types=[name('types', x) for x in np.asarray(real_state.base_types)[0, side, p]], shown_types=shown,
                classification=kind)
        out.append(entry)
    return out


def mechanisms(real_state, clone, owner, history, pressure, max_pp):
    """Which carry/belief mechanisms a decision exercises in the REAL game: per side (the active's carry state) and, for
    the peer's belief columns, the real public facts the reply prior reads. `history` = History.flags per side."""
    foe = 1 - owner
    def carry(side):
        a = int(real_state.active[0, side]);f = (history or {}).get(side) or {}
        return dict(baton_pass=bool(f.get('baton_in')), trace=bool(real_state.ability[0, side, a] != real_state.base_ability[0, side, a]),
            choice_lock=bool(real_state.choice_move[0, side, a] > 0), encore=bool(real_state.encore_move[0, side, a] > 0),
            trapped=bool(real_state.trapped[0, side]) or int(real_state.trap_source[0, side, a]) >= 0 or (int(real_state.volatile[0, side, a]) & 8) != 0,
            substitute=bool(real_state.substitute_hp[0, side, a] > 0))
    fa = int(real_state.active[0, foe])
    spent = any(int(m) > 0 and int(real_state.pp[0, foe, fa, k]) < max_pp[int(m)] for k, m in enumerate(real_state.moves[0, foe, fa]))
    seen = ((history or {}).get(owner) or {}).get('ever_active', range(6))
    port = clone.public_port
    hazards = np.zeros((2, 2)) if port is None else np.asarray(port.hazards)[0].sum(axis=-1)   # [victim side, (shown, exact)]
    belief = dict(pressure_pp=bool(pressure) and spent and any(int(real_state.ability[0, owner, p]) == pressure for p in seen),
        hazard_credit_opp=bool(hazards[owner, 0] != 0), spikes=bool(np.any(np.asarray(real_state.spikes)[0] > 0)),
        toxic=int(real_state.status[0, foe, fa]) == 4, sleep_clause=False if port is None else bool(np.any(np.asarray(port.holder)[0] >= 0)),
        hazard_credit_self=bool(hazards[foe, 1] != 0),
        damage_exact=bool(np.any(np.asarray(clone.damage_exact_taken)[0, foe] != 0) or np.any(np.asarray(clone.damage_exact_dealt)[0, owner] != 0))
            if clone.damage_exact_taken is not None else False)
    return dict(peer=dict(carry(foe), **belief), owner=carry(owner))


class Checker:
    def __init__(self, setup):
        self.setup = setup;self.encode = make_encoder(setup.executor);self.timing = collections.Counter();self._forward = None
        names = setup.materializer.namespaces['abilities'] if hasattr(setup.materializer, 'namespaces') else {}
        self.pressure = int(names.get('pressure', 0)) if names else 0
        self.max_pp = tuple(getattr(setup.materializer, 'max_pp', ()))

    def forward_rows(self, rows):
        """A's forward (the evaluator the offline pricer uses for root and reply priors) on views in actor_inputs layout,
        one batched call padded to a power-of-two bucket. rows: dicts of the four observation tensors (no batch axis)."""
        import jax
        s = self.setup
        if not rows:return np.zeros((0, 9), np.float32)
        if self._forward is None:self._forward = jax.jit(lambda params, inputs: s.forward(params, inputs).policy_logits)
        count = len(rows);bucket = max(64, 1 << (count - 1).bit_length())
        batch = {k: np.stack([np.asarray(r[k]) for r in rows] + [np.asarray(rows[0][k])] * (bucket - count))[:, None] for k in OBS_FIELDS[:4]}
        batch['history_mask'] = np.ones((bucket, 1), bool)
        return np.asarray(jax.device_get(self._forward(s.params[0], batch)), np.float32)[:count]

    def __call__(self, carry, decided, host, rows, step, history=None):
        import jax
        import jax.numpy as j
        from gpu_engine import device_public_worlds as worlds, device_public_world_state as states
        from gpu_engine.v5_device_rollout import RootContext
        s = self.setup;clock = time.perf_counter()
        rows = [int(r) for r in rows];owners = [int(s.a_side[r]) for r in rows];idx = j.asarray(rows, j.int32)
        fields = OBS_FIELDS + ('error',)
        observed = jax.device_get([{f: getattr(o, f)[idx] for f in fields} for o in decided['observations']])
        own_inputs = jax.device_get({k: v[idx] for k, v in decided['own_inputs'].items()})
        real_states = jax.device_get(jax.tree.map(lambda x: x[idx], carry.state))
        truth = jax.device_get((carry.truth.probabilities[idx], carry.truth.known[idx]))
        context = RootContext(carry.state, carry.public, carry.streams, carry.frames)
        roots = worlds.detach_roots(context, s.source, rows=tuple(rows), owners=tuple(owners), enable_device_worlds=True)
        # The real one-row clones and their owner redaction: detach_roots' own jitted program (clone, gathers, redaction).
        count = len(rows);bucket = max(64, 1 << (count - 1).bit_length())
        r = j.asarray(rows + [rows[0]] * (bucket - count), j.int32);o = j.asarray(owners + [owners[0]] * (bucket - count), j.int32)
        state_fields = {name: getattr(carry.state, name) for name in worlds.OWN_MON_FIELDS + worlds.OWN_SIDE_FIELDS}
        cloned, redacted = jax.device_get(worlds._detach_device()(carry.public, state_fields, carry.streams, carry.frames, r, o)[:2])
        self.timing['detach_seconds'] += time.perf_counter() - clock;clock = time.perf_counter()
        plans = [[worlds.plan_starts(root, s.source, recipe) for recipe in s.recipes] for root in roots]
        self.timing['plan_seconds'] += time.perf_counter() - clock;clock = time.perf_counter()
        seeds = sorted({states.kernel_seed(root, start.world, recipe) for root, per in zip(roots, plans)
            for recipe, p in zip(s.recipes, per) for start in p.starts} | {states.kernel_seed(root, 0, s.recipes[0]) for root in roots})
        keys = dict(zip(seeds, states.world_keys(seeds)))
        items, jobs = [], []   # jobs: (decision, role, extra) per encoded item, in item order
        def add(state, public, seat, role, i, extra=None):
            items.append((state, public, seat));jobs.append((i, role, extra))
        records = []
        for i, (row, owner, root) in enumerate(zip(rows, owners, roots)):
            foe = 1 - owner
            real_state = jax.tree.map(lambda x: np.asarray(x[i:i + 1]), real_states)
            clone, red = worlds._row(cloned, i), worlds._row(redacted, i)
            mapping = np.asarray(clone.physical_to_public)[0, foe]
            real = dict(species=real_state.species[0, foe], level=real_state.level[0, foe], moves=real_state.moves[0, foe],
                base_ability=real_state.base_ability[0, foe], item=real_state.item[0, foe], gender=real_state.gender[0, foe],
                hidden_power_type=real_state.hidden_power_type[0, foe], physical_to_public=mapping,
                stats=real_state.stats[0, foe], max_hp=real_state.max_hp[0, foe], hidden_power_power=real_state.hidden_power_power[0, foe],
                active=int(real_state.active[0, foe]), truth_probabilities=truth[0][i, foe], truth_known=truth[1][i, foe])
            real_order = [int(x) for x in np.asarray(clone.own_order)[0, foe]]
            owner_order = [int(x) for x in np.asarray(clone.own_order)[0, owner]]
            layout = offline_layout_order(real['active'], mapping)
            flags = None if history is None else {owner: history.flags(row, owner), foe: history.flags(row, foe)}
            record = dict(shard=s.meta.get('shard'), step=int(step), stream=int(root.address.stream), frame=int(root.address.frame),
                owner=owner, row=row, game=int(s.game[row]), seat_order=int(s.seat_order[row]), turn=int(host['turn'][row]),
                foe_requested=bool(host['requested'][row, foe]),
                order=dict(real=real_order, rule_real=rule_order(real['active']), offline_layout=layout,
                    differs_real_coords=real_order != rule_order(real['active']),
                    revealed_positions_differ=revealed_positions_differ(real_order, layout, mapping) if _is_perm(real_order) else None,
                    revealed=int(np.sum(mapping >= 0))),
                history=None if flags is None else dict(owner=flags[owner], foe=flags[foe]),
                mechanisms=mechanisms(real_state, clone, owner, flags, self.pressure, self.max_pp),
                carry=dict(owner_baton_pass=bool(real_state.baton_pass[0, owner]), foe_baton_pass=bool(real_state.baton_pass[0, foe]),
                    owner_forced_switch=bool(real_state.forced_switch[0, owner])))
            records.append(record)
            # Self checks and the real foe tiers.
            add(real_state, clone, owner, 'real_owner', i, dict(state=real_state, order=owner_order, public=clone))
            tiers = redaction_tiers(clone, red)
            add(real_state, tiers['raw'], foe, 'real_foe:raw', i)
            add(real_state, tiers['peer_damage_exact'], foe, 'real_foe:peer_damage_exact', i)
            add(real_state, tiers['peer_exact_hazard'], foe, 'real_foe:peer_exact_hazard', i)
            hp_state = redact_state_hp(real_state, foe, mapping, np.asarray(root.view['hp_percent'])[0])
            add(hp_state, tiers['peer_exact_hazard'], foe, 'real_foe:foe_hp_public_percent', i, dict(state=hp_state))
            # Recipe worlds (the pricer's own materialization).
            record['owner_check'] = dict(worlds=[])
            for b, (recipe, p) in enumerate(zip(s.recipes, plans[i])):
                for start in p.starts:
                    world = states.materialize_host(root, start, s.materializer, recipe, keys=keys)
                    label = 'b%dw%d' % (b, start.world)
                    if world.state is None:
                        record['owner_check']['worlds'].append(dict(world=label, refusal=','.join(world.refusals)[:300]));continue
                    add(world.state, world.public, owner, 'world_owner', i, dict(label=label, state=world.state,
                        order=[int(x) for x in np.asarray(world.public.own_order)[0, owner]]))
                    add(world.state, world.public, foe, 'world_peer', i, dict(label=label))
            # The true world.
            try:
                world, state, public, sigma, move_maps, notes = build_true_world(root, real, s.source, s.materializer, s.recipes[0], keys)
                record['true_world'] = dict(status='built', notes=notes, sigma=list(sigma),
                    move_order_permuted=any(k != v for m in move_maps.values() for k, v in m.items()),
                    public_differs=public_differences(world.public, public, sigma, owner))
                add(state, public, foe, 'true_foe', i, dict(order=[int(x) for x in np.asarray(public.own_order)[0, foe]], state=state))
                add(state, public, owner, 'true_owner', i, dict(label='true', state=state,
                    order=[int(x) for x in np.asarray(public.own_order)[0, owner]]))
                layout_order = [int(sigma[int(x)]) for x in np.asarray(world.public.own_order)[0, foe]]
                add(world.state, world.public, foe, 'layout_foe', i,
                    dict(order=layout_order, move_map=move_maps[int(sigma[int(world.state.active[0, foe])])]))
            except Skip as error:
                record['true_world'] = dict(status='skipped', reason=str(error))
            except (ValueError, KeyError, IndexError, TypeError) as error:   # recorded, never fatal to a long run
                record['true_world'] = dict(status='skipped', reason='true-world error: %s: %s' % (type(error).__name__, str(error)[:200]))
        self.timing['materialize_seconds'] += time.perf_counter() - clock;clock = time.perf_counter()
        encoded = self.encode(items)
        self.timing['encode_seconds'] += time.perf_counter() - clock;clock = time.perf_counter()
        views = collections.defaultdict(dict)
        for (i, role, extra), view in zip(jobs, encoded):
            if role in ('world_owner', 'world_peer'):views[i].setdefault(role, {})[extra['label']] = (view, extra)
            else:views[i][role] = (view, extra)
        logits = self._forward_all(records, views, own_inputs, host) if s.forward is not None else {}
        self.timing['forward_seconds'] += time.perf_counter() - clock;clock = time.perf_counter()
        for i, record in enumerate(records):
            self._compare(record, views[i], observed, own_inputs, host, i, logits.get(i, {}))
        self.timing['compare_seconds'] += time.perf_counter() - clock
        self.timing['decisions'] += len(records)
        return records

    def _owner_world(self, owner, label, own, extra, peer, root_view, inputs, root_legal, real):
        """One owner-world comparison: raw cells (legal lanes with causes), order-caused cells, classes, pre code."""
        real_state, real_order = real['state'], real['order']
        active = int(real_state.active[0, owner])
        d = diff(own, root_view)
        world_order = extra['order']
        aligned = (diff(align(own, world_order, real_order, active), root_view)
            if world_order != real_order and _is_perm(world_order) and _is_perm(real_order) else d)
        causes = legal_causes(legal_lanes(aligned), extra['state'], real_state, owner, real_order, active)
        aligned_keys = with_causes(aligned, causes)['keys']
        d = with_causes(d, legal_causes(legal_lanes(d), extra['state'], real_state, owner, real_order, active))
        order_keys = sorted(set(d['keys']) - set(aligned_keys))
        classes = sorted({_class_of(k) for k in aligned_keys} | ({'order'} if order_keys else set()))
        return dict(world=label, refusal=None, pre=pre_code(own, peer, inputs, root_legal), error=int(own['error']),
            max_abs=numeric_max_abs(own, root_view), order_keys=order_keys, aligned_keys=aligned_keys, classes=classes, **serial(d))

    def _forward_all(self, records, views, own_inputs, host):
        """One batched forward per step: each decision's root inputs (the root-logit re-check) and, where the peer check
        will compare, (a) the true world's foe view, (l) the offline-layout world's foe view, (b) the real foe view after the
        named redactions and (b2) that same view again at another batch row (the forward noise floor)."""
        rows, where = [], []
        for i, record in enumerate(records):
            rows.append({k: own_inputs[k][i] for k in OBS_FIELDS[:4]});where.append((i, 'root'))
            v = views[i]
            if record['foe_requested'] and 'true_foe' in v and 'layout_foe' in v:
                for role, view in (('a', v['true_foe'][0]), ('l', v['layout_foe'][0]), ('b', v['real_foe:foe_hp_public_percent'][0])):
                    rows.append(view);where.append((i, role))
        for i, record in enumerate(records):   # b2 rows after every other row: a different batch position than b
            v = views[i]
            if record['foe_requested'] and 'true_foe' in v and 'layout_foe' in v:rows.append(v['real_foe:foe_hp_public_percent'][0]);where.append((i, 'b2'))
        out = collections.defaultdict(dict)
        for (i, role), row in zip(where, self.forward_rows(rows)):out[i][role] = row
        for i, record in enumerate(records):
            raw = np.asarray(host['raw_logits'][record['row'], record['owner']], np.float32);mine = out[i]['root']
            record['root_logits'] = dict(equal=bool(np.array_equal(mine, raw)), max_abs=float(np.max(np.abs(mine.astype(np.float64) - raw))))
        return out

    def _peer_effect(self, logits, world_view, layout_view, layout_extra, final_real, final_state, foe, world_order, real_order, active):
        """Reply-prior effect of the input differences, on identity-aligned lanes (see effect())."""
        if not ('a' in logits and _is_perm(world_order) and _is_perm(real_order)):return None
        perm_a = lane_permutation(world_order, real_order, active)
        legal_b = np.asarray(final_real['legal_action_mask'], bool)
        pick = lambda vector, perm:np.asarray(vector)[perm]
        identities = lane_identities(final_state, foe, real_order, active, self.setup.source.vocabulary)
        layout = None
        if _is_perm(layout_extra['order']):
            perm_l = lane_permutation(layout_extra['order'], real_order, active, move_map=layout_extra['move_map'])
            layout = effect(pick(logits['l'], perm_l), pick(layout_view['legal_action_mask'], perm_l), logits['b'], legal_b)
        out = dict(identities=identities,
            primary=effect(pick(logits['a'], perm_a), pick(world_view['legal_action_mask'], perm_a), logits['b'], legal_b),
            layout=layout, noise=effect(logits['b2'], legal_b, logits['b'], legal_b))
        for key in ('primary', 'layout'):
            if out[key]:out[key].update(top_a_identity=identities[out[key]['top_a']], top_b_identity=identities[out[key]['top_b']])
        if out['noise']:out['noise'] = {k: out['noise'][k] for k in ('kl', 'tv', 'top1_flip', 'max_abs_dp')}
        return out

    def _compare(self, record, views, observed, own_inputs, host, i, logits=None):
        owner = record['owner'];foe = 1 - owner
        decide_owner = {f: observed[owner][f][i] for f in OBS_FIELDS + ('error',)}
        decide_foe = {f: observed[foe][f][i] for f in OBS_FIELDS + ('error',)}
        inputs = {k: np.asarray(own_inputs[k][i]) for k in own_inputs}   # decide's own row (the Request's root inputs)
        root_legal = np.asarray(host['legal'][record['row'], owner], bool)
        root_view = dict(inputs, legal_action_mask=root_legal)
        # Root source (structural): the Request's root inputs/legal are the actor's own observation row; with a forward,
        # A's forward on those inputs re-derives decide's raw logits (record['root_logits']).
        record['root_source'] = dict(inputs_equal=all(np.array_equal(inputs[k], decide_owner[k]) for k in OBS_FIELDS[:4]),
            legal_equal=bool(np.array_equal(root_legal, decide_owner['legal_action_mask'])))
        real_owner, real_extra = views['real_owner']
        record['self_check'] = dict(owner_reencode_equal=diff(real_owner, decide_owner)['total'] == 0)
        if record['foe_requested']:
            record['self_check']['foe_reencode_equal'] = diff(views['real_foe:raw'][0], decide_foe)['total'] == 0
        # Owner check: recipe worlds and the true world.
        worlds = record['owner_check']
        for label, (own, extra) in views.get('world_owner', {}).items():
            peer = views.get('world_peer', {}).get(label, (None, None))[0]
            worlds['worlds'].append(self._owner_world(owner, label, own, extra, peer, root_view, inputs, root_legal, real_extra))
        worlds['worlds'].sort(key=lambda w: w['world'])
        if 'true_owner' in views:
            own, extra = views['true_owner']
            worlds['true'] = self._owner_world(owner, 'true', own, extra, None, root_view, inputs, root_legal, real_extra)
        # Peer check.
        peer = dict(status='skipped', reason=None, kind=None)
        if not record['foe_requested']:peer.update(reason='foe not requested at this decision', kind='not_requested')
        elif record['true_world']['status'] != 'built':peer.update(reason=record['true_world']['reason'], kind='refused')
        else:
            world_view, extra = views['true_foe']
            tiers = [('raw', decide_foe)] + [(name, views['real_foe:' + name][0]) for name, _ in REDACTIONS]
            diffs = [(name, diff(world_view, real)) for name, real in tiers]
            remaining = diffs[-1][1];final_real = tiers[-1][1];final_state = views['real_foe:foe_hp_public_percent'][1]['state']
            explained, introduced = {}, {}
            for (_, before), (name, after) in zip(diffs, diffs[1:]):
                gone = before['exact'] - after['exact'];new = after['exact'] - before['exact']
                explained[name] = dict(cells=len(gone), keys=sorted({':'.join(cell_key(c[0], c[1:])) for c in gone}))
                introduced[name] = dict(cells=len(new), keys=sorted({':'.join(cell_key(c[0], c[1:])) for c in new}))
            active = int(np.asarray(record['order']['rule_real'])[0])
            world_order, real_order = extra['order'], record['order']['real']
            if _is_perm(world_order) and _is_perm(real_order) and active in world_order and active in real_order:
                aligned = diff(align(world_view, world_order, real_order, active), final_real)
            else:aligned = remaining
            aligned = with_causes(aligned, legal_causes(legal_lanes(aligned), extra['state'], final_state, foe, real_order, active))
            remaining = with_causes(remaining, legal_causes(legal_lanes(remaining), extra['state'], final_state, foe, real_order, active))
            order_keys = sorted(set(remaining['keys']) - set(aligned['keys']))
            layout_view, layout_extra = views['layout_foe']
            layout_raw = diff(layout_view, final_real)
            layout_aligned = (diff(align(layout_view, layout_extra['order'], real_order, active, move_map=layout_extra['move_map']), final_real)
                if _is_perm(layout_extra['order']) and _is_perm(real_order) else layout_raw)
            peer_effect = self._peer_effect(logits or {}, world_view, layout_view, layout_extra, final_real, final_state, foe,
                world_order, real_order, active)
            residual = (describe_residual(aligned['exact'], final_state, extra['state'], real_extra['public'], foe, real_order,
                self.setup.source.vocabulary) if aligned['total'] else None)
            peer = dict(status='compared', reason=None, kind='compared', world_order=world_order, effect=peer_effect, residual=residual,
                tiers={name: d['total'] for name, d in diffs}, explained=explained, introduced=introduced,
                remaining=serial(remaining), aligned=serial(aligned), order_keys=order_keys,
                order_only=remaining['total'] > 0 and aligned['total'] == 0,
                max_abs=numeric_max_abs(world_view, final_real), public_differs=record['true_world'].get('public_differs', []),
                offline_layout=dict(raw=layout_raw['total'], aligned=layout_aligned['total'], aligned_keys=layout_aligned['keys'],
                    order_only=layout_raw['total'] > 0 and layout_aligned['total'] == 0))
        record['peer'] = peer


# ---------------------------------------------------------------------------------------------- the shard loop

def run_shard(setup, out, *, max_steps=None, quota=None, sample_rate=1., sample_seed=0, policy='sample'):
    """Play the shard's games to the end (or max_steps), checking sampled decisions; write the shard's outputs."""
    import jax
    import jax.numpy as j
    from pokezero import v5_gumbel as original
    out = Path(out);out.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter();checker = Checker(setup);timing = collections.Counter()
    carry = setup.fresh();step = 0;checked = 0;quota_step = None;seen = 0
    history = History(jax.device_get(carry.state.active))
    a_side = setup.a_side;a_side_device = j.asarray(a_side);everyone = np.arange(len(a_side))
    log('OFFLINE_VIEW_PARITY_START', **dict(setup.meta, rows=len(a_side), quota=quota, sample_rate=sample_rate, policy=policy))
    with (out / 'positions.jsonl').open('w') as stream, patch.object(original.Recipe, 'selected', lambda self, address: True):
        while True:
            clock = time.perf_counter()
            decided = setup.decide(carry, setup.params, a_side_device)
            host = jax.device_get({k: decided[k] for k in ('live', 'requested', 'legal', 'raw_logits', 'actions', 'input_valid', 'turn')})
            timing['decide_seconds'] += time.perf_counter() - clock
            history.update(*jax.device_get((carry.state.active, carry.state.baton_pass)))
            if not host['live'].any() or (max_steps is not None and step >= max_steps):break
            if quota is not None and checked >= quota:break   # nothing left to check in this shard
            frames = np.asarray(carry.frames);streams = np.asarray(carry.streams)
            rows = np.flatnonzero(host['live'] & host['requested'][everyone, a_side] & host['input_valid'][everyone, a_side])
            seen += len(rows)
            chosen = [int(r) for r in rows if selected(sample_seed, streams[r], frames[r], a_side[r], sample_rate)]
            if quota is not None:
                chosen = chosen[:max(0, quota - checked)]
                if quota_step is None and checked + len(chosen) >= quota:quota_step = step
            if chosen:
                for record in checker(carry, decided, host, chosen, step, history):stream.write(json.dumps(record) + '\n')
                stream.flush();checked += len(chosen)
            actions = host['actions'].copy()
            if policy == 'argmax':
                logits = np.where(host['legal'], host['raw_logits'], -np.inf)
                greedy = np.argmax(logits, axis=-1).astype(np.int32)
                use = host['requested'] & host['legal'].any(axis=-1)
                actions = np.where(use, greedy, actions)
            clock = time.perf_counter()
            carry = setup.apply(carry, decided, j.asarray(actions))
            timing['apply_seconds'] += time.perf_counter() - clock
            step += 1
            if step % 25 == 0:
                log('STEP', step=step, live=int(host['live'].sum()), checked=checked, elapsed=round(time.perf_counter() - begin, 1))
    statuses = jax.device_get(carry.status)
    meta = dict(setup.meta, steps=step, owner_decisions_seen=int(seen), decisions_checked=checked, quota=quota,
        quota_reached_at_step=quota_step, sample_rate=sample_rate, sample_seed=sample_seed, policy=policy,
        dispositions={str(k): int(v) for k, v in zip(*np.unique(np.asarray(statuses), return_counts=True))},
        timing=dict(wall_seconds=time.perf_counter() - begin, **timing, **checker.timing))
    positions = [json.loads(line) for line in (out / 'positions.jsonl').read_text().splitlines() if line]
    write_summary(out, positions, [meta])
    log('OFFLINE_VIEW_PARITY', checked=checked, steps=step, wall_seconds=round(meta['timing']['wall_seconds'], 1), out=str(out))
    return meta


# ---------------------------------------------------------------------------------------------- summary

def _class_of(key):
    """Which input-bug class a differing cell belongs to (documented in summary.md). Legal lanes go by their cause
    (LEGAL_CAUSE_CLASS: PP exhausted is a peer-belief fact, traps/locks/disabled are carry fields, else 'legal')."""
    from gpu_engine import engine_device_features as features
    sec, kind, name = key.split(':')
    public = {item[0] for item in features.MON_FIELDS.values()} | {'TOXIC_STAGE', 'WAKE_KNOWN', 'LAST_DAMAGE_DEALT',
        'LAST_DAMAGE_TAKEN', 'TRUANT_LOAF', 'SUB_HP_FRACTION'}
    if kind == 'legal':return LEGAL_CAUSE_CLASS.get(name.partition('/')[2] or 'unknown', 'legal')
    if kind == 'categorical' and name in ('TYPE_1', 'TYPE_2') and sec in ('own', 'opponent', 'belief'):return 'types'
    if sec == 'field' and kind == 'categorical' and name == 'PRIMARY':return 'carry_fields'   # request kind
    if sec == 'own' and (name.startswith('BOOST_') or (kind == 'numeric' and name in ('ACTIVE', 'LEGAL'))
            or (kind == 'categorical' and (name.startswith('BELIEF_ABILITY') or name == 'TRACED_ABILITY' or name.startswith('VOLATILE')))):
        return 'carry_fields'
    if sec in ('move_actions', 'switch_actions') and name in ('LEGAL', 'ACTIVE', 'PRESENT'):return 'carry_fields'
    if sec in ('field', 'opponent', 'belief', 'stats'):return 'peer_belief_columns'
    if sec == 'own' and (name in public or (kind == 'categorical' and name == 'LAST_USED_MOVE')):return 'peer_belief_columns'
    if sec == 'move_actions' and name == 'MOVE_PP_FRACTION':return 'peer_belief_columns'
    return 'other'


OWNER_CLASSES = dict(types='types', order='order', belief='peer_belief_columns', carry='carry_fields', legal='legal', other='other')


def _rank(key_lists, denominator, top=40):
    counts = collections.Counter(k for keys in key_lists for k in set(keys))
    return [dict(cell=k, decisions=n, share=round(n / denominator, 6) if denominator else None)
        for k, n in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:top]]


def _share(numerator, denominator):
    return None if not denominator else round(numerator / denominator, 6)


def _zeroed_types(result):
    """Zeroed shown-type cells of a serialized diff, counted over the full arrays (diff's zeroed_types)."""
    return int(result.get('zeroed_types', 0))


def owner_rollup(worlds):
    """Owner-world differences per class (types, bench order, belief columns, carry fields, legal lanes, other)."""
    out = {}
    for label, name in OWNER_CLASSES.items():
        hit = [w for w in worlds if name in w.get('classes', ())]
        keys = [w.get('order_keys', []) for w in hit] if name == 'order' else [[k for k in w.get('aligned_keys', w.get('keys', [])) if _class_of(k) == name] for w in hit]
        out[label] = dict(worlds=len(hit), share=_share(len(hit), len(worlds)), cells=_rank(keys, len(worlds), top=8))
    return out


def _owner_compared(p):
    return [w for w in p['owner_check']['worlds'] if not w['refusal']] + ([p['owner_check']['true']] if p['owner_check'].get('true') else [])


def mechanism_table(P, side, names, class_name):
    """Per mechanism: sampled decisions exercising it, how many were compared, refused or had the foe not requested,
    and how many compared ones show differences of the class. NOT_EXERCISED when none was compared."""
    rows = {}
    for m in names:
        hit = [p for p in P if (p.get('mechanisms') or {}).get(side, {}).get(m)]
        if side == 'peer':
            compared = [p for p in hit if p['peer']['status'] == 'compared']
            refused = sum(p['peer'].get('kind') == 'refused' for p in hit);not_requested = sum(p['peer'].get('kind') == 'not_requested' for p in hit)
            diffs = sum(any(_class_of(k) == class_name for k in p['peer']['aligned']['keys']) for p in compared)
            any_diffs = sum(p['peer']['aligned']['total'] > 0 for p in compared)
        else:
            compared = [p for p in hit if _owner_compared(p)]
            refused, not_requested = len(hit) - len(compared), 0
            diffs = sum(any(class_name in w.get('classes', ()) for w in _owner_compared(p)) for p in compared)
            any_diffs = sum(any(w['total'] > 0 for w in _owner_compared(p)) for p in compared)
        rows[m] = dict(sampled=len(hit), compared=len(compared), refused=refused, not_requested=not_requested,
            with_class_differences=diffs, with_any_differences=any_diffs,
            verdict='PRESENT' if diffs else 'ABSENT' if compared else 'NOT_EXERCISED')
    return rows


def class_verdict(differences, table, compared):
    """PRESENT on any difference of the class; otherwise ABSENT only when every mechanism was compared clean,
    ABSENT_WHERE_EXERCISED when some were, NOT_EXERCISED when none (or nothing) was compared."""
    if differences:return 'PRESENT'
    verdicts = [r['verdict'] for r in table.values()]
    if not compared or not any(v == 'ABSENT' for v in verdicts):return 'NOT_EXERCISED'
    return 'ABSENT' if all(v == 'ABSENT' for v in verdicts) else 'ABSENT_WHERE_EXERCISED'


def _finite_effect(v):
    """Whether an effect dict's KL, TV, max |dp| and probabilities are all finite (the NaN guard)."""
    numbers = [v.get('kl'), v.get('tv'), v.get('max_abs_dp')] + list(v.get('pa', ())) + list(v.get('pb', ()))
    return all(x is not None and np.isfinite(float(x)) for x in numbers)


def effect_stats(values, *, keep_support_differs=False):
    """median/p95/max KL(b || a), top-1 flip share, mean total variation and mean/max of max |dp| over effect dicts.

    Decisions whose legal sets differ (support_differs: the statistics cover only the common lanes) are excluded
    unless keep_support_differs, and counted; so are non-finite values (the NaN guard), counted as `nonfinite`. With
    the per-decision dropped_b/dropped_a fields, the dropped mass is summarized (b's prior on lanes illegal in a, and
    the reverse); positions written before those fields count as `dropped_unavailable`."""
    values = [v for v in values if v]
    differs = [v for v in values if v.get('support_differs')]
    kept = values if keep_support_differs else [v for v in values if not v.get('support_differs')]
    finite = [v for v in kept if _finite_effect(v)]
    out = dict(n=len(finite), support_differs=len(differs), excluded_support_differs=0 if keep_support_differs else len(differs),
        nonfinite=len(kept) - len(finite))
    if finite:
        kl = np.asarray([v['kl'] for v in finite]);tv = np.asarray([v['tv'] for v in finite]);dp = np.asarray([v['max_abs_dp'] for v in finite])
        out.update(kl_median=float(np.median(kl)), kl_p95=float(np.quantile(kl, .95)), kl_max=float(kl.max()),
            top1_flip_share=round(float(np.mean([bool(v['top1_flip']) for v in finite])), 6), tv_mean=float(tv.mean()),
            max_abs_dp_mean=float(dp.mean()), max_abs_dp_max=float(dp.max()))
    dropped = [v for v in (finite if keep_support_differs else []) if v.get('support_differs')]
    if dropped:
        with_mass = [v for v in dropped if 'dropped_b' in v and 'dropped_a' in v]
        out.update(dropped_unavailable=len(dropped) - len(with_mass))
        if with_mass:
            b_mass = np.asarray([v['dropped_b'] for v in with_mass]);a_mass = np.asarray([v['dropped_a'] for v in with_mass])
            out.update(dropped_b_mean=float(b_mass.mean()), dropped_b_max=float(b_mass.max()),
                dropped_a_mean=float(a_mass.mean()), dropped_a_max=float(a_mass.max()))
    return out


def _back(p):
    return bool((p.get('history') or {}).get('foe', {}).get('back_switched'))


def _order_only(p):
    """Order-caused cells and no residual cell after the alignment (the peer view differs ONLY by bench order)."""
    return bool(p['peer'].get('order_only', bool(p['peer']['order_keys']) and p['peer']['aligned']['total'] == 0))


EFFECT_GROUPS = (
    ('order only (headline): order-caused cells, no residual cells', _order_only),
    ('order only, foe back-switched', lambda p:_order_only(p) and _back(p)),
    ('order only, no foe back-switch', lambda p:_order_only(p) and not _back(p)),
    ('all compared', lambda p:True),
    ('foe back-switched', _back),
    ('no foe back-switch', lambda p:not _back(p)),
    ('foe order != rebase rule', lambda p:p['order']['differs_real_coords']),
    ('foe order == rebase rule', lambda p:not p['order']['differs_real_coords']),
    ('order-caused cells present (with or without residual)', lambda p:bool(p['peer']['order_keys'])),
    ('no order-caused cells', lambda p:not p['peer']['order_keys']),
    ('residual cells after alignment (any class)', lambda p:p['peer']['aligned']['total'] > 0),
    ('control: no order and no residual cells', lambda p:not p['peer']['order_keys'] and p['peer']['aligned']['total'] == 0),
)
SUPPORT_GROUP = 'legal sets differ (excluded from every row above; common-lane statistics and dropped mass)'


def summarize_effect(P):
    """The reply-prior effect of the peer-order bug: (a) the true world's foe view against (b) the real foe view after
    the same named redactions, identity-aligned; (l), the true team in the offline layout, beside it; the noise floor from
    (b) forwarded twice at different batch rows, and from the control row. A pure function of positions.jsonl."""
    rows = [p for p in P if p['peer']['status'] == 'compared' and p['peer'].get('effect')]
    groups = []
    for label, keep in EFFECT_GROUPS:
        chosen = [p for p in rows if keep(p)]
        groups.append(dict(group=label, primary=effect_stats([p['peer']['effect']['primary'] for p in chosen]),
            layout=effect_stats([p['peer']['effect']['layout'] for p in chosen])))
    support = dict(group=SUPPORT_GROUP,
        primary=effect_stats([e for p in rows for e in [p['peer']['effect']['primary']] if e and e.get('support_differs')], keep_support_differs=True),
        layout=effect_stats([e for p in rows for e in [p['peer']['effect']['layout']] if e and e.get('support_differs')], keep_support_differs=True))
    control = next(g for g in groups if g['group'].startswith('control'))
    flips = [dict(stream=p['stream'], frame=p['frame'], owner=p['owner'], turn=p['turn'], kl=round(e['kl'], 6), order_only=_order_only(p),
        real_top=e['top_b_identity'], world_top=e['top_a_identity'])
        for p in rows for e in [p['peer']['effect']['primary']] if e and e['top1_flip'] and _finite_effect(e) and not e.get('support_differs')]
    return dict(decisions=len(rows), headline=groups[0], groups=groups, support_differs=support,
        noise_floor=dict(repeat_forward=effect_stats([p['peer']['effect']['noise'] for p in rows]), control_primary=control['primary']),
        top1_flips=flips[:50],
        definition='KL(b || a) over the lanes legal in both, lanes aligned by identity (a move lane = active member and move, a '
            'switch lane = its target member, through each side\'s request order); a = the true world\'s foe view (the offline '
            'reply-prior input, real physical indices), l = the true team in the offline layout (bench encounter order, source '
            'move order; not literally a searched world, whose hidden members are sampled), b = the real foe view after the named '
            'redactions; champion forward as the offline pricer\'s reply prior. Decisions whose legal sets differ are excluded '
            'from every group and reported in their own row; non-finite values are excluded and counted')


def summarize_residual(P):
    rows = [(p, p['peer']['residual']) for p in P if p['peer']['status'] == 'compared' and p['peer'].get('residual')]
    entries = [dict(stream=p['stream'], frame=p['frame'], owner=p['owner'], turn=p['turn'], **entry) for p, residual in rows for entry in residual]
    return dict(decisions=len(rows), tokens=len(entries), classification=dict(collections.Counter(e.get('classification', 'no member') for e in entries)),
        entries=entries[:50])


def summarize(positions, metas):
    P = positions;n = len(P)
    worlds = [w for p in P for w in p['owner_check']['worlds']]
    refused = [w for w in worlds if w['refusal']];compared = [w for w in worlds if not w['refusal']]
    trues = [p['owner_check']['true'] for p in P if p['owner_check'].get('true')]
    pre = collections.Counter(str(w['pre']) for w in compared)
    owner = dict(worlds=len(worlds), refused=len(refused),
        refusal_reasons=dict(collections.Counter(w['refusal'].split(':')[0][:120] for w in refused).most_common(20)),
        compared=len(compared), exact=sum(w['total'] == 0 for w in compared),
        exact_share=_share(sum(w['total'] == 0 for w in compared), len(compared)),
        pre_codes=dict(sorted(pre.items())), would_be_pre_refused=sum(v for k, v in pre.items() if k != '0'),
        classes=owner_rollup(compared), cells_ranked=_rank([w['keys'] for w in compared], len(compared)),
        true_world=dict(compared=len(trues), exact=sum(w['total'] == 0 for w in trues),
            pre_codes=dict(sorted(collections.Counter(str(w['pre']) for w in trues).items())), classes=owner_rollup(trues),
            cells_ranked=_rank([w['keys'] for w in trues], len(trues))))
    forward = [p['root_logits'] for p in P if p.get('root_logits')]
    root = dict(decisions=n, structural=dict(inputs_mismatch=sum(not p['root_source']['inputs_equal'] for p in P),
            legal_mismatch=sum(not p['root_source']['legal_equal'] for p in P)),
        forward=dict(checked=len(forward), equal=sum(x['equal'] for x in forward), tolerance=ROOT_LOGIT_TOLERANCE,
            above_tolerance=sum(x['max_abs'] > ROOT_LOGIT_TOLERANCE for x in forward),
            max_abs=max((x['max_abs'] for x in forward), default=None)))
    built = [p['true_world'] for p in P if p['true_world']['status'] == 'built']
    selfcheck = dict(owner_reencode_mismatch=sum(not p['self_check']['owner_reencode_equal'] for p in P),
        foe_reencode_mismatch=sum(p['self_check'].get('foe_reencode_equal') is False for p in P),
        true_world_notes=dict(collections.Counter(note for p in P for note in p['true_world'].get('notes', ()))),
        true_worlds_built=len(built), materializer_public_differs=sum(bool(x.get('public_differs')) for x in built),
        materializer_public_fields=dict(collections.Counter(f for x in built for f in x.get('public_differs', ()))))
    peers = [p['peer'] for p in P];cmp = [x for x in peers if x['status'] == 'compared']
    explained = {name: dict(decisions=sum(x['explained'][name]['cells'] > 0 for x in cmp), cells=sum(x['explained'][name]['cells'] for x in cmp),
        introduced_cells=sum(x['introduced'][name]['cells'] for x in cmp),
        cells_ranked=_rank([x['explained'][name]['keys'] for x in cmp], len(cmp), top=15), mechanism=text) for name, text in REDACTIONS}
    order_caused = [x for x in cmp if x['order_keys']]
    peer = dict(decisions=n, compared=len(cmp),
        refused=sum(x.get('kind') == 'refused' for x in peers), not_requested=sum(x.get('kind') == 'not_requested' for x in peers),
        skipped=dict(collections.Counter(x['reason'][:120] for x in peers if x['status'] != 'compared').most_common(20)),
        raw_differs=sum(x['tiers']['raw'] > 0 for x in cmp), redactions=explained,
        remaining_differs=sum(x['remaining']['total'] > 0 for x in cmp),
        aligned_remaining_differs=sum(x['aligned']['total'] > 0 for x in cmp),
        order_only=sum(x['order_only'] for x in cmp), order_only_share=_share(sum(x['order_only'] for x in cmp), len(cmp)),
        order_and_other=sum(bool(x['order_keys']) and x['aligned']['total'] > 0 for x in cmp),
        order_caused_cells_ranked=_rank([x['order_keys'] for x in cmp], len(cmp), top=20),
        remaining_cells_ranked=_rank([x['aligned']['keys'] for x in cmp], len(cmp)),
        move_order_permuted=sum(p['true_world'].get('move_order_permuted', False) for p in P if p['peer']['status'] == 'compared'),
        offline_layout=dict(raw_differs=sum(x['offline_layout']['raw'] > 0 for x in cmp),
            order_only=sum(x['offline_layout']['order_only'] for x in cmp),
            order_only_share=_share(sum(x['offline_layout']['order_only'] for x in cmp), len(cmp)),
            aligned_differs=sum(x['offline_layout']['aligned'] > 0 for x in cmp)))
    orders = [p['order'] for p in P];backs = [p for p in P if p.get('history')]
    back = sum(p['history']['foe']['back_switched'] for p in backs)
    order = dict(decisions=len(orders), differs_from_rule=sum(o['differs_real_coords'] for o in orders),
        differs_from_rule_share=_share(sum(o['differs_real_coords'] for o in orders), len(orders)),
        foe_back_switched=back, foe_back_switched_share=_share(back, len(backs)),
        differs_without_back_switch=sum(p['order']['differs_real_coords'] and not p['history']['foe']['back_switched'] for p in backs),
        revealed_positions_differ=sum(bool(o['revealed_positions_differ']) for o in orders),
        revealed_positions_differ_share=_share(sum(bool(o['revealed_positions_differ']) for o in orders), len(orders)),
        with_switch_history=sum(o['real'] != list(range(6)) for o in orders),
        order_only_share_of_compared=peer['order_only_share'])
    zeroed_owner = sum(_zeroed_types(w) for w in compared + trues)
    zeroed_peer = sum(_zeroed_types(x['aligned']) for x in cmp)
    by_class = collections.defaultdict(collections.Counter)
    for x in cmp:
        for c in {_class_of(k) for k in x['aligned']['keys']}:by_class[c]['decisions'] += 1
        for k in x['aligned']['keys']:by_class[_class_of(k)][k] += 1
    def evidence_cells(name, top=12):
        c = by_class.get(name, collections.Counter())
        return [dict(cell=k, decisions=v, share=_share(v, len(cmp))) for k, v in c.most_common() if k != 'decisions'][:top]
    owner_class_worlds = lambda name: sum(name in w.get('classes', ()) for w in compared + trues)
    carry_table = {'peer.' + k: v for k, v in mechanism_table(P, 'peer', CARRY_MECHANISMS, 'carry_fields').items()}
    carry_table.update({'owner.' + k: v for k, v in mechanism_table(P, 'owner', CARRY_MECHANISMS, 'carry_fields').items()})
    belief_table = mechanism_table(P, 'peer', BELIEF_MECHANISMS, 'peer_belief_columns')
    excluded = {m: dict(mechanism_table(P, 'peer', (m,), 'peer_belief_columns')[m], verdict='EXCLUDED_BY_DESIGN',
        redaction=redaction) for m, redaction in EXCLUDED_BY_DESIGN.items()}
    carry_differences = by_class['carry_fields']['decisions'] + owner_class_worlds('carry_fields')
    belief_differences = by_class['peer_belief_columns']['decisions']
    classes = dict(
        types=dict(verdict='PRESENT' if zeroed_owner + zeroed_peer else 'ABSENT' if compared or trues or cmp else 'NOT_EXERCISED',
            evidence=dict(zeroed_type_cells_owner_worlds=zeroed_owner, zeroed_type_cells_peer=zeroed_peer,
                owner_worlds_compared=len(compared) + len(trues), owner_worlds_with_type_differences=owner_class_worlds('types'),
                peer_compared=len(cmp), peer_decisions_with_any_type_difference=by_class['types']['decisions'],
                cells=evidence_cells('types')), mechanism=MECHANISMS['types'],
            rule='PRESENT when a TYPE_1/TYPE_2 category is zero in a world view and nonzero in the real view (full arrays)'),
        root_source=dict(verdict='PRESENT' if root['structural']['inputs_mismatch'] + root['structural']['legal_mismatch']
                + root['forward']['above_tolerance'] else 'ABSENT' if n else 'NOT_EXERCISED',
            evidence=dict(root, owner_world_pre_codes=owner['pre_codes']), mechanism=MECHANISMS['root_source'],
            rule="structural: the Request's root inputs and legal are decide's own row for A's seat (near-tautological); "
                "re-checked by A's forward on those inputs against decide's raw logits (|diff| <= %s)" % ROOT_LOGIT_TOLERANCE),
        peer_order=dict(verdict='PRESENT' if order_caused else 'ABSENT' if cmp else 'NOT_EXERCISED',
            evidence=dict(foe_order_differs_from_rule_share=order['differs_from_rule_share'],
                foe_back_switched_share=order['foe_back_switched_share'], differs_without_back_switch=order['differs_without_back_switch'],
                revealed_positions_differ_share=order['revealed_positions_differ_share'],
                peer_compared=len(cmp), decisions_with_order_caused_cells=len(order_caused),
                order_only=peer['order_only'], order_only_share=peer['order_only_share'], order_and_other=peer['order_and_other'],
                offline_layout_order_only_share=peer['offline_layout']['order_only_share'],
                owner_worlds_with_order_differences=owner_class_worlds('order'), cells=peer['order_caused_cells_ranked'][:12]),
            mechanism=MECHANISMS['peer_order'],
            rule='PRESENT when aligning the bench order removes differing cells from the peer view in some decision'),
        peer_belief_columns=dict(verdict=class_verdict(belief_differences, belief_table, len(cmp)),
            evidence=dict(peer_compared=len(cmp), decisions=belief_differences, share=_share(belief_differences, len(cmp)),
                owner_worlds_with_belief_differences=owner_class_worlds('peer_belief_columns'),
                mechanisms=belief_table, excluded_by_design=excluded, cells=evidence_cells('peer_belief_columns')),
            mechanism=MECHANISMS['peer_belief_columns'],
            rule='PRESENT when a field/opponent/belief/stats cell, an own public-feature column, the own move PP fraction or a '
                'PP-exhausted legal lane differs after the named redactions and the order alignment; the peer\'s own exact hazard '
                'credit and exact damage halves are excluded by design (redactions peer_exact_hazard, peer_damage_exact)'),
        carry_fields=dict(verdict=class_verdict(carry_differences, carry_table, len(cmp) + len(compared) + len(trues)),
            evidence=dict(peer_compared=len(cmp), decisions=by_class['carry_fields']['decisions'],
                share=_share(by_class['carry_fields']['decisions'], len(cmp)), owner_worlds_with_carry_differences=owner_class_worlds('carry_fields'),
                mechanisms=carry_table, cells=evidence_cells('carry_fields'),
                recharge='NOT_APPLICABLE: the GPU engine has no recharge state, so the offline games never reach it'),
            mechanism=MECHANISMS['carry_fields'],
            rule='PRESENT when a trap/lock/disabled legal lane, the request kind, an own boost/ability/volatile/active cell or an '
                'action legality cell differs (peer after redactions and alignment; owner worlds); per mechanism NOT_EXERCISED '
                'when no compared decision exercised it'))
    other = dict(decisions=by_class['other']['decisions'], cells=evidence_cells('other', top=20))
    legal = dict(decisions=by_class['legal']['decisions'], cells=evidence_cells('legal', top=12))
    runs = sorted(metas, key=lambda m: (m.get('shard') or 0))
    timing = collections.Counter()
    for m in runs:timing.update({k: v for k, v in m.get('timing', {}).items() if isinstance(v, (int, float))})
    check_seconds = sum(timing[k] for k in ('detach_seconds', 'forward_seconds', 'plan_seconds', 'materialize_seconds', 'encode_seconds', 'compare_seconds'))
    return dict(schema=SCHEMA, decisions=n, runs=runs, owner=owner, root_source=root, self_check=selfcheck, peer=peer, order=order,
        other_remaining=other, legal_remaining=legal, classes=classes, peer_order_effect=summarize_effect(P), residual=summarize_residual(P),
        timing=dict(total=dict(timing), check_seconds=check_seconds,
            check_seconds_per_100_decisions=None if not n else 100 * check_seconds / n,
            wall_seconds_per_100_decisions=None if not n else 100 * sum(m.get('timing', {}).get('wall_seconds', 0) for m in runs) / n),
        redactions=[dict(name=name, mechanism=text) for name, text in REDACTIONS],
        scope='Inputs of the offline head-to-head search worlds against the real root, on GPU-engine games; no pricing, '
            'no Foul Play result.')


def _mechanism_lines(title, table):
    lines = ['', '| %s | verdict | sampled | compared | refused | foe not requested | with class differences |' % title, '|---|---|---|---|---|---|---|']
    for name, r in table.items():
        lines.append('| %s | %s | %d | %d | %d | %d | %d |' % (name, r['verdict'], r['sampled'], r['compared'], r['refused'], r['not_requested'],
            r['with_class_differences']))
    return lines


def render(summary):
    s = summary;lines = ['# V2 offline view parity', '']
    runs = s['runs']
    if runs:
        m = runs[0]
        lines += ['config `%s` (sha256 `%s`), iteration %s, block %s; shards %s of %s; policy %s; worlds %s per decision' % (
            m.get('config'), m.get('config_sha256'), m.get('iteration'), m.get('block'), sorted(r.get('shard') for r in runs),
            m.get('shards'), m.get('policy'), 2 * (m.get('world_blocks') or 0)), '']
    lines += ['Decisions checked: %d (owner decisions seen: %s; steps per shard: %s)' % (s['decisions'],
        sum(r.get('owner_decisions_seen', 0) for r in runs), [r.get('steps') for r in runs]), '']
    lines += ['## Five-class verdict', '', '| class | verdict | evidence | mechanism |', '|---|---|---|---|']
    for name in CLASSES:
        c = s['classes'][name]
        evidence = {k: v for k, v in c['evidence'].items() if k not in ('cells', 'mechanisms', 'excluded_by_design')}
        cells = ', '.join('%s (%s)' % (x['cell'], x['decisions']) for x in c['evidence'].get('cells', [])[:6])
        lines.append('| %s | **%s** | %s%s | %s |' % (name, c['verdict'], json.dumps(evidence, default=str).replace('|', '/'),
            ('; cells: ' + cells) if cells else '', c['mechanism'].replace('|', '/')))
    sc = s['self_check']
    lines += ['', 'True-world notes (materialized real team vs the real one, e.g. stats, max HP or Hidden Power from the source '
        'spread): %s of %s true worlds; %s. Materializer world public vs the re-based public: %s differ (%s).' % (
            sum(sc['true_world_notes'].values()), sc['true_worlds_built'], json.dumps(sc['true_world_notes']) or '{}',
            sc['materializer_public_differs'], json.dumps(sc['materializer_public_fields'])), '',
        'Verdicts: PRESENT (a difference of the class), ABSENT (every listed mechanism compared, none differs), '
        'ABSENT_WHERE_EXERCISED (no difference, some mechanisms never compared), NOT_EXERCISED (none compared), '
        'NOT_APPLICABLE, EXCLUDED_BY_DESIGN (a named redaction).', '',
        'Rules: ' + '; '.join('%s: %s' % (k, s['classes'][k]['rule']) for k in CLASSES if s['classes'][k].get('rule'))]
    lines += _mechanism_lines('carry mechanism (side.name)', s['classes']['carry_fields']['evidence']['mechanisms'])
    lines += ['', 'recharge: ' + s['classes']['carry_fields']['evidence']['recharge']]
    lines += _mechanism_lines('peer belief mechanism', dict(s['classes']['peer_belief_columns']['evidence']['mechanisms'],
        **s['classes']['peer_belief_columns']['evidence']['excluded_by_design']))
    o = s['owner'];root = s['root_source']
    lines += ['', '## Owner check (recipe worlds against the real root)', '',
        '- worlds: %d; refused by the materializer: %d; compared: %d; exact: %d (share %s)' % (o['worlds'], o['refused'], o['compared'], o['exact'], o['exact_share']),
        '- the pricer\'s would-be pre-refusal codes of compared worlds: %s (0 admitted; 1 own error; 2 legal; 3-6 first differing input; 8 peer error); would be refused: %d' % (
            o['pre_codes'], o['would_be_pre_refused']),
        '- refusal reasons: %s' % o['refusal_reasons'],
        '- true world (real foe team): compared %d, exact %d, codes %s' % (o['true_world']['compared'], o['true_world']['exact'], o['true_world']['pre_codes']),
        '- root source (structural): %s; forward re-check: %s' % (json.dumps(root['structural']), json.dumps(root['forward'])), '',
        '| owner-world class | recipe worlds | share | true worlds | top cells |', '|---|---|---|---|---|']
    for label in OWNER_CLASSES:
        a, b = o['classes'][label], o['true_world']['classes'][label]
        lines.append('| %s | %d | %s | %d | %s |' % (label, a['worlds'], a['share'], b['worlds'], ', '.join(x['cell'] for x in a['cells'][:4])))
    if o['cells_ranked']:
        lines += ['', '| owner-view cell (recipe worlds) | worlds | share |', '|---|---|---|']
        lines += ['| %s | %d | %s |' % (x['cell'], x['decisions'], x['share']) for x in o['cells_ranked'][:20]]
    p = s['peer']
    lines += ['', '## Peer check (true world, foe seat)', '',
        '- compared: %d of %d; refused (true world not built): %d; foe not requested: %d; skipped reasons: %s' % (
            p['compared'], p['decisions'], p['refused'], p['not_requested'], p['skipped']),
        '- raw differs: %d; after the named redactions: %d; after the bench-order alignment: %d' % (p['raw_differs'], p['remaining_differs'], p['aligned_remaining_differs']),
        '- order only: %d (share %s); order and other: %d; move slots permuted from the variant order: %d' % (p['order_only'], p['order_only_share'], p['order_and_other'], p['move_order_permuted']),
        '- the offline world layout (materializer physical order, variant move order): raw differs %d, order only %d (share %s), still differs after alignment %d' % (
            p['offline_layout']['raw_differs'], p['offline_layout']['order_only'], p['offline_layout']['order_only_share'], p['offline_layout']['aligned_differs']), '',
        '| named redaction (real side, cumulative) | decisions | cells removed | cells introduced | mechanism |', '|---|---|---|---|---|']
    for name, _ in REDACTIONS:
        r = p['redactions'][name]
        lines.append('| %s | %d | %d | %d | %s |' % (name, r['decisions'], r['cells'], r['introduced_cells'], r['mechanism'].replace('|', '/')))
    lines += ['', '| remaining peer cell (after redactions and alignment) | class | decisions | share |', '|---|---|---|---|']
    lines += ['| %s | %s | %d | %s |' % (x['cell'], _class_of(x['cell']), x['decisions'], x['share']) for x in p['remaining_cells_ranked'][:30]]
    lines += ['', '| order-caused peer cell | decisions | share |', '|---|---|---|']
    lines += ['| %s | %d | %s |' % (x['cell'], x['decisions'], x['share']) for x in p['order_caused_cells_ranked'][:12]]
    e = s['peer_order_effect']
    fmt = lambda x:'-' if x is None else ('%.3g' % x if isinstance(x, float) else str(x))
    head = e['headline']['primary']
    lines += ['', '## Reply-prior effect of the peer order (champion forward, identity-aligned lanes)', '', e['definition'], '',
        '**Headline (order only: the peer view differs only by bench order):** n %s, KL median %s, p95 %s, max %s, top-1 flip share %s, '
        'mean TV %s (legal sets differ: %s excluded; non-finite: %s)' % (head.get('n', 0), fmt(head.get('kl_median')), fmt(head.get('kl_p95')),
            fmt(head.get('kl_max')), fmt(head.get('top1_flip_share')), fmt(head.get('tv_mean')), head.get('excluded_support_differs', 0),
            head.get('nonfinite', 0)), '',
        '| group | n | KL median | KL p95 | KL max | top-1 flip share | mean TV | mean max abs dp | excl. legal differs | non-finite '
        '| layout n | layout KL median | layout KL p95 | layout flip share |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for g in e['groups'] + [e['support_differs']]:
        a_, l_ = g['primary'], g['layout']
        lines.append('| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %d | %s | %s | %s |' % (g['group'], a_.get('n', 0), fmt(a_.get('kl_median')),
            fmt(a_.get('kl_p95')), fmt(a_.get('kl_max')), fmt(a_.get('top1_flip_share')), fmt(a_.get('tv_mean')), fmt(a_.get('max_abs_dp_mean')),
            a_.get('excluded_support_differs', 0), a_.get('nonfinite', 0), l_.get('n', 0), fmt(l_.get('kl_median')), fmt(l_.get('kl_p95')),
            fmt(l_.get('top1_flip_share'))))
    sd = e['support_differs']['primary']
    lines += ['', '- legal sets differ: %s decisions; dropped mass, b\'s prior on lanes illegal in a: mean %s, max %s; a\'s on lanes '
        'illegal in b: mean %s, max %s; positions without the dropped-mass fields: %s' % (sd.get('n', 0), fmt(sd.get('dropped_b_mean')),
            fmt(sd.get('dropped_b_max')), fmt(sd.get('dropped_a_mean')), fmt(sd.get('dropped_a_max')), sd.get('dropped_unavailable', 0))]
    nf = e['noise_floor']['repeat_forward']
    lines += ['', '- forward noise floor, (b) forwarded twice at different batch rows: n %s, KL max %s, max abs dp max %s; control row KL max %s' % (
        nf.get('n', 0), fmt(nf.get('kl_max')), fmt(nf.get('max_abs_dp_max')), fmt(e['noise_floor']['control_primary'].get('kl_max'))),
        '- top-1 flips (first %d): %s' % (len(e['top1_flips']), json.dumps(e['top1_flips'][:10]))]
    res = s['residual']
    lines += ['', '## Residual peer cells after alignment (by token and member)', '',
        '- decisions: %d; tokens: %d; classification: %s' % (res['decisions'], res['tokens'], json.dumps(res['classification'])), '',
        '| stream/frame | token | member | species | ability (base) | item | last move (public) | real types | world types | shown types | class | cells |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for x in res['entries'][:30]:
        lines.append('| %s/%s | %s | %s %s | %s | %s (%s) | %s | %s (%s) | %s | %s | %s | %s | %s |' % (x['stream'], x['frame'], x['token'],
            x.get('side', ''), x.get('member', ''), x.get('species', ''), x.get('ability', ''), x.get('base_ability', ''), x.get('item', ''),
            x.get('last_move', ''), x.get('public_last_move', ''), '/'.join(x.get('real_types', [])), '/'.join(x.get('world_types', [])),
            '/'.join(x.get('shown_types', [])), x.get('classification', ''), ', '.join(x['cells'][:4])))
    r = s['order']
    lines += ['', '## Peer order', '',
        '- foe order != rebase rule (real coordinates): %d of %d (%s); it counts first switch-ins out of party order, faint '
            'replacements and drag-ins as well as back-switches' % (r['differs_from_rule'], r['decisions'], r['differs_from_rule_share']),
        '- true back-switch share (the foe has re-entered a member benched earlier, seen at decisions): %d (%s); order differs '
            'without any back-switch: %s' % (r['foe_back_switched'], r['foe_back_switched_share'], r['differs_without_back_switch']),
        '- offline layout: a revealed foe member at another request position than in the offline world: %d (%s)' % (
            r['revealed_positions_differ'], r['revealed_positions_differ_share']),
        '- decisions whose foe order is not the initial one: %d' % r['with_switch_history'], '',
        '## Self-checks', '', '- %s' % json.dumps(s['self_check']), '',
        '## Timing', '', '- %s' % json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in s['timing'].items() if k != 'total'}),
        '- totals: %s' % json.dumps({k: round(v, 2) for k, v in s['timing']['total'].items()}), '', s['scope'], '']
    return '\n'.join(lines)


def write_summary(out, positions, metas):
    out = Path(out);summary = summarize(positions, metas)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=str) + '\n')
    (out / 'summary.md').write_text(render(summary))
    return summary


REPORT_KEYS = ('config_sha256', 'iteration', 'block', 'shards', 'max_games', 'world_blocks', 'search_seed', 'sample_seed',
    'sample_rate', 'sample_decisions', 'policy', 'champion')


def report(inputs, out):
    """Merge shard outputs: positions concatenated in shard order; the summary recomputed from them."""
    out = Path(out);out.mkdir(parents=True, exist_ok=False)
    metas, positions = [], []
    for directory in inputs:
        directory = Path(directory)
        metas.extend(json.loads((directory / 'summary.json').read_text())['runs'])
        positions.extend(json.loads(line) for line in (directory / 'positions.jsonl').read_text().splitlines() if line)
    for key in REPORT_KEYS:
        if len({json.dumps(m.get(key), sort_keys=True, default=str) for m in metas}) > 1:raise SystemExit('shards differ in ' + key)
    shards = sorted(m.get('shard') for m in metas)
    if len(set(shards)) != len(shards):raise SystemExit('a shard appears twice')
    missing = sorted(set(range(metas[0].get('shards') or 0)) - set(shards)) if metas else []
    if missing:log('REPORT_MISSING_SHARDS', missing=missing)
    positions.sort(key=lambda p: (p.get('shard') or 0, p['step'], p['row']))
    with (out / 'positions.jsonl').open('w') as stream:
        for p in positions:stream.write(json.dumps(p) + '\n')
    summary = write_summary(out, positions, metas)
    log('OFFLINE_VIEW_PARITY_REPORT', decisions=summary['decisions'], shards=shards, missing=missing,
        verdicts={k: v['verdict'] for k, v in summary['classes'].items()})
    return summary


# ---------------------------------------------------------------------------------------------- the real setup

@dataclass
class EmptyShard:
    """A shard with no games (more shards than games): its full run metadata, for report."""
    meta: dict


def build_setup(args):
    """The offline path's own preparation: v2_fused.prepare(--config), the champion binding, the block's games and its
    swapped copy (v2_search_head_to_head.main), make_step, and Search's source/materializer/executor/recipes.
    An empty shard returns EmptyShard with the same metadata."""
    import jax
    import jax.numpy as j
    from pokezero import v5_gumbel as original
    from gpu_engine import v2_search_head_to_head as h2h, device_public_worlds as worlds, device_public_world_state as states
    from gpu_engine import device_public_teacher as teacher
    from gpu_engine.v2_fused import prepare
    from gpu_engine.v2_head_to_head import SWAP_OFFSET
    from gpu_engine.v5_device_backend import RuntimeBackend
    context = prepare(argparse.Namespace(config=args.config, arm='B', iterations=2 if args.iteration else 1))
    plan, prepared = context['plan'], context['prepared']
    evaluator, record, params = h2h.champion_setup(args, prepared.catalog)
    backend = RuntimeBackend(prepared, context['pool'], gender_policy=context['gender']);backend.check_plan(plan)
    batch = plan.iterations[args.iteration][args.block]
    games, roots = batch.games, batch.gender_root_ids
    if args.max_games:games, roots = games[:args.max_games], (None if roots is None else roots[:args.max_games])
    picked = shard_games(tuple(range(len(games))), args.shard, args.shards)
    meta = dict(shard=args.shard, shards=args.shards, config=args.config, config_sha256=context['declared'].get('config_sha256'),
        iteration=args.iteration, block=args.block, max_games=args.max_games, games=list(picked), world_blocks=args.world_blocks,
        search_seed=args.search_seed, sample_decisions=getattr(args, 'sample_decisions', None), champion=record.evaluator,
        sampling_seed=batch.sampling_seed, max_requests=plan.max_requests, backend=jax.default_backend())
    if not picked:return EmptyShard(meta)
    batch = replace(batch, games=tuple(games[g] for g in picked), gender_root_ids=None if roots is None else tuple(roots[g] for g in picked))
    swapped = replace(batch, games=tuple(replace(g, stream_id=g.stream_id - SWAP_OFFSET, battle_id=g.battle_id + '-swap') for g in batch.games))
    def fresh():
        carries = [jax.block_until_ready(backend._construct(declared)) for declared in (batch, swapped)]
        return jax.tree.map(lambda x, y: j.concatenate([x, y]), *carries)
    decide, apply = h2h.make_step(prepared.engine_tables, prepared.observation_tables, plan.model, sampling_seed=batch.sampling_seed,
        max_requests=plan.max_requests, forwards=(evaluator, evaluator))
    metadata = json.loads(prepared.metadata_json)
    source = worlds.compile_source(prepared.source, metadata, expected_source_hash=prepared.source.metadata.source_hash,
        expected_source_rows_sha256=plan.model.label_vocabulary.source_binding_sha256)
    materializer = states.compile_tables(source, metadata)
    executor = teacher.make_executor(prepared.engine_tables, prepared.observation_tables, enable_device_teacher=True)
    recipes = tuple(original.Recipe(s) for s in h2h.world_block_seeds(args.search_seed, args.world_blocks))
    n = len(picked)
    return Setup(executor, source, materializer, recipes, decide, apply, (params, params), np.repeat(np.asarray([0, 1], np.int32), n),
        np.tile(np.asarray(picked, np.int32), 2), np.repeat(np.asarray([0, 1], np.int32), n), fresh, meta, forward=evaluator)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='play and check one shard')
    run.add_argument('--config', required=True, help='the diagnostic config v2_search_head_to_head takes (--config)')
    run.add_argument('--iteration', type=int, default=1);run.add_argument('--block', type=int, default=0)
    run.add_argument('--max-games', type=int, help='prefix of the block before sharding (smoke and sizing)')
    run.add_argument('--max-steps', type=int, help='stop after this many lockstep steps (smoke)')
    run.add_argument('--shard', type=int, default=0);run.add_argument('--shards', type=int, default=1)
    run.add_argument('--policy', choices=('sample', 'argmax'), default='sample', help='both seats: raw sampled action or greedy legal argmax')
    run.add_argument('--world-blocks', type=int, default=4, help='recipe worlds per decision = 2 * blocks (v2_search_head_to_head default 4)')
    run.add_argument('--search-seed', type=int, default=2026110500, help='the recipe block seed (v2_search_head_to_head --search-seed)')
    run.add_argument('--sample-decisions', type=int, default=2000, help='decisions checked in total (each shard ceil(N / shards))')
    run.add_argument('--sample-rate', type=float, default=1., help='per-decision selection probability (seeded hash of stream, frame, owner)')
    run.add_argument('--sample-seed', type=int, default=20261109)
    run.add_argument('--out', required=True, help='a new directory')
    from gpu_engine import champion_evaluator
    champion_evaluator.add_arguments(run)
    merge = sub.add_parser('report', help='merge shard outputs')
    merge.add_argument('--inputs', nargs='+', required=True);merge.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if args.command == 'report':
        report(args.inputs, args.out);return 0
    champion_evaluator.check_arguments(run, args)
    if not args.champion_archive:run.error('the champion flags are required (--champion-archive, --champion-archive-sha256, --champion-checkpoint-sha256)')
    if not 0 < args.sample_rate <= 1 or args.sample_decisions < 1 or args.world_blocks < 1:run.error('0 < --sample-rate <= 1, positive counts')
    if Path(args.out).exists():run.error('--out must be a new directory')
    setup = build_setup(args);quota = shard_quota(args.sample_decisions, args.shards)
    if isinstance(setup, EmptyShard):
        out = Path(args.out);out.mkdir(parents=True);(out / 'positions.jsonl').write_text('')
        write_summary(out, [], [dict(setup.meta, steps=0, owner_decisions_seen=0, decisions_checked=0, quota=quota, quota_reached_at_step=None,
            sample_rate=args.sample_rate, sample_seed=args.sample_seed, policy=args.policy, timing={})])
        return 0
    run_shard(setup, args.out, max_steps=args.max_steps, quota=quota, sample_rate=args.sample_rate, sample_seed=args.sample_seed,
        policy=args.policy)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
