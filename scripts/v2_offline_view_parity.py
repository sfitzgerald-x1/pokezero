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
4. Five-class verdict: PRESENT / ABSENT / NOT_APPLICABLE per class, with counts and the mechanism's file:line.

Self-checks: re-encoding the real root through the pricer's views equals decide's observation (both seats);
decide's own_inputs row equals O's own observation (the root source); the true world's foe species/levels/moves and
mapping equal the real ones (notes otherwise).

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
    out.update(total=int(total), keys=sorted(keys), exact=exact)
    return out


def serial(result):
    """A diff without its exact cell set and without empty kinds (positions.jsonl stays small)."""
    return {k: v for k, v in result.items() if k != 'exact' and not (isinstance(v, dict) and v.get('count') == 0)}


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


class Checker:
    def __init__(self, setup):
        self.setup = setup;self.encode = make_encoder(setup.executor);self.timing = collections.Counter()

    def __call__(self, carry, decided, host, rows, step):
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
            layout = offline_layout_order(real['active'], mapping)
            record = dict(shard=s.meta.get('shard'), step=int(step), stream=int(root.address.stream), frame=int(root.address.frame),
                owner=owner, row=row, game=int(s.game[row]), seat_order=int(s.seat_order[row]), turn=int(host['turn'][row]),
                foe_requested=bool(host['requested'][row, foe]),
                order=dict(real=real_order, rule_real=rule_order(real['active']), offline_layout=layout,
                    differs_real_coords=real_order != rule_order(real['active']),
                    revealed_positions_differ=revealed_positions_differ(real_order, layout, mapping) if _is_perm(real_order) else None,
                    revealed=int(np.sum(mapping >= 0))),
                carry=dict(owner_baton_pass=bool(real_state.baton_pass[0, owner]), foe_baton_pass=bool(real_state.baton_pass[0, foe]),
                    owner_traced=bool(real_state.ability[0, owner, real_state.active[0, owner]] != real_state.base_ability[0, owner, real_state.active[0, owner]]),
                    foe_traced=bool(real_state.ability[0, foe, real['active']] != real_state.base_ability[0, foe, real['active']]),
                    owner_forced_switch=bool(real_state.forced_switch[0, owner])))
            records.append(record)
            # Self checks and the real foe tiers.
            add(real_state, clone, owner, 'real_owner', i)
            tiers = redaction_tiers(clone, red)
            add(real_state, tiers['raw'], foe, 'real_foe:raw', i)
            add(real_state, tiers['peer_damage_exact'], foe, 'real_foe:peer_damage_exact', i)
            add(real_state, tiers['peer_exact_hazard'], foe, 'real_foe:peer_exact_hazard', i)
            hp_state = redact_state_hp(real_state, foe, mapping, np.asarray(root.view['hp_percent'])[0])
            add(hp_state, tiers['peer_exact_hazard'], foe, 'real_foe:foe_hp_public_percent', i)
            # Recipe worlds (the pricer's own materialization).
            record['owner_check'] = dict(worlds=[])
            for b, (recipe, p) in enumerate(zip(s.recipes, plans[i])):
                for start in p.starts:
                    world = states.materialize_host(root, start, s.materializer, recipe, keys=keys)
                    label = 'b%dw%d' % (b, start.world)
                    if world.state is None:
                        record['owner_check']['worlds'].append(dict(world=label, refusal=','.join(world.refusals)[:300]));continue
                    add(world.state, world.public, owner, 'world_owner', i, label)
                    add(world.state, world.public, foe, 'world_peer', i, label)
            # The true world.
            try:
                world, state, public, sigma, move_maps, notes = build_true_world(root, real, s.source, s.materializer, s.recipes[0], keys)
                record['true_world'] = dict(status='built', notes=notes, sigma=list(sigma),
                    move_order_permuted=any(k != v for m in move_maps.values() for k, v in m.items()))
                add(state, public, foe, 'true_foe', i, dict(order=[int(x) for x in np.asarray(public.own_order)[0, foe]]))
                add(state, public, owner, 'true_owner', i)
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
            if role in ('world_owner', 'world_peer'):views[i].setdefault(role, {})[extra] = view
            else:views[i][role] = (view, extra)
        for i, record in enumerate(records):
            self._compare(record, views[i], observed, own_inputs, host, i)
        self.timing['compare_seconds'] += time.perf_counter() - clock
        self.timing['decisions'] += len(records)
        return records

    def _compare(self, record, views, observed, own_inputs, host, i):
        owner = record['owner'];foe = 1 - owner
        decide_owner = {f: observed[owner][f][i] for f in OBS_FIELDS + ('error',)}
        decide_foe = {f: observed[foe][f][i] for f in OBS_FIELDS + ('error',)}
        inputs = {k: np.asarray(own_inputs[k][i]) for k in own_inputs}   # decide's own row (the Request's root inputs)
        root_legal = np.asarray(host['legal'][record['row'], owner], bool)
        root_view = dict(inputs, legal_action_mask=root_legal)
        # Root source: the Request's root inputs are the actor's own observation of the real root.
        record['root_source'] = dict(inputs_equal=all(np.array_equal(inputs[k], decide_owner[k]) for k in OBS_FIELDS[:4]),
            legal_equal=bool(np.array_equal(root_legal, decide_owner['legal_action_mask'])))
        real_owner = views['real_owner'][0]
        record['self_check'] = dict(owner_reencode_equal=diff(real_owner, decide_owner)['total'] == 0)
        if record['foe_requested']:
            record['self_check']['foe_reencode_equal'] = diff(views['real_foe:raw'][0], decide_foe)['total'] == 0
        # Owner check, recipe worlds.
        worlds = record['owner_check']
        for label, own in views.get('world_owner', {}).items():
            peer = views.get('world_peer', {}).get(label)
            d = diff(own, root_view)
            worlds['worlds'].append(dict(world=label, refusal=None, pre=pre_code(own, peer, inputs, root_legal),
                error=int(own['error']), max_abs=numeric_max_abs(own, root_view), **serial(d)))
        worlds['worlds'].sort(key=lambda w: w['world'])
        if 'true_owner' in views:
            own = views['true_owner'][0];d = diff(own, root_view)
            worlds['true'] = dict(world='true', refusal=None, pre=pre_code(own, None, inputs, root_legal), error=int(own['error']),
                max_abs=numeric_max_abs(own, root_view), **serial(d))
        # Peer check.
        peer = dict(status='skipped', reason=None)
        if not record['foe_requested']:peer['reason'] = 'foe not requested at this decision'
        elif record['true_world']['status'] != 'built':peer['reason'] = record['true_world']['reason']
        else:
            world_view, extra = views['true_foe']
            tiers = [('raw', decide_foe)] + [(name, views['real_foe:' + name][0]) for name, _ in REDACTIONS]
            diffs = [(name, diff(world_view, real)) for name, real in tiers]
            remaining = diffs[-1][1];final_real = tiers[-1][1]
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
            order_keys = sorted(set(remaining['keys']) - set(aligned['keys']))
            layout_view, layout_extra = views['layout_foe']
            layout_raw = diff(layout_view, final_real)
            layout_aligned = (diff(align(layout_view, layout_extra['order'], real_order, active, move_map=layout_extra['move_map']), final_real)
                if _is_perm(layout_extra['order']) and _is_perm(real_order) else layout_raw)
            peer = dict(status='compared', reason=None, world_order=world_order,
                tiers={name: d['total'] for name, d in diffs}, explained=explained, introduced=introduced,
                remaining=serial(remaining), aligned=serial(aligned), order_keys=order_keys,
                order_only=remaining['total'] > 0 and aligned['total'] == 0,
                max_abs=numeric_max_abs(world_view, final_real),
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
    a_side = setup.a_side;a_side_device = j.asarray(a_side);everyone = np.arange(len(a_side))
    log('OFFLINE_VIEW_PARITY_START', **dict(setup.meta, rows=len(a_side), quota=quota, sample_rate=sample_rate, policy=policy))
    with (out / 'positions.jsonl').open('w') as stream, patch.object(original.Recipe, 'selected', lambda self, address: True):
        while True:
            clock = time.perf_counter()
            decided = setup.decide(carry, setup.params, a_side_device)
            host = jax.device_get({k: decided[k] for k in ('live', 'requested', 'legal', 'raw_logits', 'actions', 'input_valid', 'turn')})
            timing['decide_seconds'] += time.perf_counter() - clock
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
                for record in checker(carry, decided, host, chosen, step):stream.write(json.dumps(record) + '\n')
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
    """Which input-bug class a remaining peer cell belongs to (documented in summary.md)."""
    from gpu_engine import engine_device_features as features
    sec, kind, name = key.split(':')
    public = {item[0] for item in features.MON_FIELDS.values()} | {'TOXIC_STAGE', 'WAKE_KNOWN', 'LAST_DAMAGE_DEALT',
        'LAST_DAMAGE_TAKEN', 'TRUANT_LOAF', 'SUB_HP_FRACTION'}
    if kind == 'categorical' and name in ('TYPE_1', 'TYPE_2') and sec in ('own', 'opponent', 'belief'):return 'types'
    if kind == 'legal' or (sec == 'field' and kind == 'categorical' and name == 'PRIMARY'):return 'carry_fields'
    if sec == 'own' and (name.startswith('BOOST_') or (kind == 'numeric' and name in ('ACTIVE', 'LEGAL'))
            or (kind == 'categorical' and (name.startswith('BELIEF_ABILITY') or name == 'TRACED_ABILITY' or name.startswith('VOLATILE')))):
        return 'carry_fields'
    if sec in ('move_actions', 'switch_actions') and name in ('LEGAL', 'ACTIVE', 'PRESENT'):return 'carry_fields'
    if sec in ('field', 'opponent', 'belief', 'stats'):return 'peer_belief_columns'
    if sec == 'own' and (name in public or (kind == 'categorical' and name == 'LAST_USED_MOVE')):return 'peer_belief_columns'
    if sec == 'move_actions' and name == 'MOVE_PP_FRACTION':return 'peer_belief_columns'
    return 'other'


def _rank(key_lists, denominator, top=40):
    counts = collections.Counter(k for keys in key_lists for k in set(keys))
    return [dict(cell=k, decisions=n, share=round(n / denominator, 6) if denominator else None)
        for k, n in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:top]]


def _share(numerator, denominator):
    return None if not denominator else round(numerator / denominator, 6)


def _zeroed_types(result):
    """Cells of a serialized diff where a TYPE category is zero in the world and nonzero in the real view."""
    from pokezero import showdown as reference
    columns = (reference.CATEGORY_TYPE_1, reference.CATEGORY_TYPE_2)
    return sum(1 for token, column, w, r in result.get('categorical', {}).get('cells', ()) if column in columns and w == 0 and r != 0
        and section(token) in ('own', 'opponent', 'belief'))


def summarize(positions, metas):
    P = positions;n = len(P)
    worlds = [w for p in P for w in p['owner_check']['worlds']]
    refused = [w for w in worlds if w['refusal']];compared = [w for w in worlds if not w['refusal']]
    trues = [p['owner_check']['true'] for p in P if p['owner_check'].get('true')]
    owner = dict(worlds=len(worlds), refused=len(refused),
        refusal_reasons=dict(collections.Counter(w['refusal'].split(':')[0][:120] for w in refused).most_common(20)),
        compared=len(compared), exact=sum(w['total'] == 0 for w in compared),
        exact_share=_share(sum(w['total'] == 0 for w in compared), len(compared)),
        pre_codes=dict(sorted(collections.Counter(str(w['pre']) for w in compared).items())),
        cells_ranked=_rank([w['keys'] for w in compared], len(compared)),
        true_world=dict(compared=len(trues), exact=sum(w['total'] == 0 for w in trues),
            pre_codes=dict(sorted(collections.Counter(str(w['pre']) for w in trues).items())),
            cells_ranked=_rank([w['keys'] for w in trues], len(trues))))
    root = dict(decisions=n, inputs_mismatch=sum(not p['root_source']['inputs_equal'] for p in P),
        legal_mismatch=sum(not p['root_source']['legal_equal'] for p in P))
    selfcheck = dict(owner_reencode_mismatch=sum(not p['self_check']['owner_reencode_equal'] for p in P),
        foe_reencode_mismatch=sum(p['self_check'].get('foe_reencode_equal') is False for p in P),
        true_world_notes=dict(collections.Counter(note for p in P for note in p['true_world'].get('notes', ()))))
    peers = [p['peer'] for p in P];cmp = [x for x in peers if x['status'] == 'compared']
    explained = {name: dict(decisions=sum(x['explained'][name]['cells'] > 0 for x in cmp), cells=sum(x['explained'][name]['cells'] for x in cmp),
        introduced_cells=sum(x['introduced'][name]['cells'] for x in cmp),
        cells_ranked=_rank([x['explained'][name]['keys'] for x in cmp], len(cmp), top=15), mechanism=text) for name, text in REDACTIONS}
    order_caused = [x for x in cmp if x['order_keys']]
    peer = dict(decisions=n, compared=len(cmp),
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
    orders = [p['order'] for p in P]
    order = dict(decisions=len(orders), back_switch=sum(o['differs_real_coords'] for o in orders),
        back_switch_share=_share(sum(o['differs_real_coords'] for o in orders), len(orders)),
        revealed_positions_differ=sum(bool(o['revealed_positions_differ']) for o in orders),
        revealed_positions_differ_share=_share(sum(bool(o['revealed_positions_differ']) for o in orders), len(orders)),
        with_switch_history=sum(o['real'] != list(range(6)) for o in orders),
        order_only_share_of_compared=peer['order_only_share'])
    carry = dict(owner_baton_pass=sum(p['carry']['owner_baton_pass'] for p in P), foe_baton_pass=sum(p['carry']['foe_baton_pass'] for p in P),
        owner_traced=sum(p['carry']['owner_traced'] for p in P), foe_traced=sum(p['carry']['foe_traced'] for p in P),
        owner_forced_switch=sum(p['carry']['owner_forced_switch'] for p in P))
    zeroed_owner = sum(_zeroed_types(w) for w in compared + trues)
    zeroed_peer = sum(_zeroed_types(x['aligned']) for x in cmp)
    by_class = collections.defaultdict(lambda: collections.Counter())
    for x in cmp:
        classes = {_class_of(k) for k in x['aligned']['keys']}
        for c in classes:by_class[c]['decisions'] += 1
        for k in x['aligned']['keys']:by_class[_class_of(k)][k] += 1
    def evidence_cells(name, top=12):
        c = by_class.get(name, collections.Counter())
        return [dict(cell=k, decisions=v, share=_share(v, len(cmp))) for k, v in c.most_common() if k != 'decisions'][:top]
    def verdict(present, exercised):
        return 'PRESENT' if present else 'ABSENT' if exercised else 'NOT_APPLICABLE'
    type_cells = sum(1 for x in cmp for k in x['aligned']['keys'] if _class_of(k) == 'types')
    classes = dict(
        types=dict(verdict=verdict(zeroed_owner + zeroed_peer > 0, len(compared) + len(trues) + len(cmp) > 0),
            evidence=dict(zeroed_type_cells_owner_worlds=zeroed_owner, zeroed_type_cells_peer=zeroed_peer,
                owner_worlds_compared=len(compared) + len(trues), peer_compared=len(cmp),
                peer_decisions_with_any_type_difference=by_class['types']['decisions'], peer_type_cells=type_cells,
                cells=evidence_cells('types')), mechanism=MECHANISMS['types'],
            rule='PRESENT when a TYPE_1/TYPE_2 category is zero in a world view and nonzero in the real view'),
        root_source=dict(verdict=verdict(root['inputs_mismatch'] + root['legal_mismatch'] > 0, n > 0),
            evidence=dict(root, owner_world_pre_codes=owner['pre_codes']), mechanism=MECHANISMS['root_source'],
            rule="PRESENT when the Request's root inputs or root legal are not the actor's own observation of the real root"),
        peer_order=dict(verdict=verdict(len(order_caused) > 0, len(cmp) > 0),
            evidence=dict(back_switch=order['back_switch'], back_switch_share=order['back_switch_share'],
                revealed_positions_differ_share=order['revealed_positions_differ_share'],
                peer_compared=len(cmp), decisions_with_order_caused_cells=len(order_caused),
                order_only=peer['order_only'], order_only_share=peer['order_only_share'], order_and_other=peer['order_and_other'],
                offline_layout_order_only_share=peer['offline_layout']['order_only_share'], cells=peer['order_caused_cells_ranked'][:12]),
            mechanism=MECHANISMS['peer_order'],
            rule='PRESENT when aligning the bench order removes differing cells from the peer view in some decision'),
        peer_belief_columns=dict(verdict=verdict(by_class['peer_belief_columns']['decisions'] > 0, len(cmp) > 0),
            evidence=dict(peer_compared=len(cmp), decisions=by_class['peer_belief_columns']['decisions'],
                share=_share(by_class['peer_belief_columns']['decisions'], len(cmp)), cells=evidence_cells('peer_belief_columns')),
            mechanism=MECHANISMS['peer_belief_columns'],
            rule='PRESENT when a field/opponent/belief/stats cell, an own public-feature column or the own move PP fraction '
                'differs after the named redactions and the order alignment'),
        carry_fields=dict(verdict=verdict(by_class['carry_fields']['decisions'] > 0, len(cmp) > 0),
            evidence=dict(peer_compared=len(cmp), decisions=by_class['carry_fields']['decisions'],
                share=_share(by_class['carry_fields']['decisions'], len(cmp)), cells=evidence_cells('carry_fields'), exercised=carry,
                recharge='NOT_APPLICABLE: the GPU engine has no recharge state, so the offline games never reach it'),
            mechanism=MECHANISMS['carry_fields'],
            rule='PRESENT when a legal lane, the request kind, an own boost/ability/volatile/active cell or an action '
                'legality cell differs after the named redactions and the order alignment'))
    other = dict(decisions=by_class['other']['decisions'], cells=evidence_cells('other', top=20))
    runs = sorted(metas, key=lambda m: (m.get('shard') or 0))
    timing = collections.Counter()
    for m in runs:timing.update({k: v for k, v in m.get('timing', {}).items() if isinstance(v, (int, float))})
    check_seconds = sum(timing[k] for k in ('detach_seconds', 'plan_seconds', 'materialize_seconds', 'encode_seconds', 'compare_seconds'))
    return dict(schema=SCHEMA, decisions=n, runs=runs, owner=owner, root_source=root, self_check=selfcheck, peer=peer, order=order,
        carry=carry, other_remaining=other, classes=classes,
        timing=dict(total=dict(timing), check_seconds=check_seconds,
            check_seconds_per_100_decisions=None if not n else 100 * check_seconds / n,
            wall_seconds_per_100_decisions=None if not n else 100 * sum(m.get('timing', {}).get('wall_seconds', 0) for m in runs) / n),
        redactions=[dict(name=name, mechanism=text) for name, text in REDACTIONS],
        scope='Inputs of the offline head-to-head search worlds against the real root, on GPU-engine games; no pricing, '
            'no Foul Play result.')


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
        evidence = {k: v for k, v in c['evidence'].items() if k not in ('cells',)}
        cells = ', '.join('%s (%s)' % (x['cell'], x['decisions']) for x in c['evidence'].get('cells', [])[:6])
        lines.append('| %s | **%s** | %s%s | %s |' % (name, c['verdict'], json.dumps(evidence, default=str).replace('|', '/'),
            ('; cells: ' + cells) if cells else '', c['mechanism'].replace('|', '/')))
    lines += ['', 'Rules: ' + '; '.join('%s: %s' % (k, s['classes'][k]['rule']) for k in CLASSES if s['classes'][k].get('rule')), '']
    o = s['owner']
    lines += ['## Owner check (recipe worlds against the real root)', '',
        '- worlds: %d; refused by the materializer: %d; compared: %d; exact: %d (share %s)' % (o['worlds'], o['refused'], o['compared'], o['exact'], o['exact_share']),
        '- pricer pre-refusal codes of compared worlds: %s (0 = admitted; 2 legal; 3-6 first differing input)' % o['pre_codes'],
        '- refusal reasons: %s' % o['refusal_reasons'],
        '- true world (real foe team): compared %d, exact %d, codes %s' % (o['true_world']['compared'], o['true_world']['exact'], o['true_world']['pre_codes']), '']
    if o['cells_ranked']:
        lines += ['| owner-view cell (recipe worlds) | worlds | share |', '|---|---|---|']
        lines += ['| %s | %d | %s |' % (x['cell'], x['decisions'], x['share']) for x in o['cells_ranked'][:20]] + ['']
    p = s['peer']
    lines += ['## Peer check (true world, foe seat)', '',
        '- compared: %d of %d; skipped: %s' % (p['compared'], p['decisions'], p['skipped']),
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
    r = s['order']
    lines += ['', '## Peer order', '',
        '- foe own_order != [active, *rest ascending] (real coordinates, the "back-switch" share): %d of %d (%s)' % (r['back_switch'], r['decisions'], r['back_switch_share']),
        '- a revealed foe member at another request position than in the offline world layout: %d (%s)' % (r['revealed_positions_differ'], r['revealed_positions_differ_share']),
        '- decisions whose foe order is not the initial one: %d' % r['with_switch_history'], '',
        '## Self-checks', '', '- %s' % json.dumps(s['self_check']), '- root source: %s' % json.dumps(s['root_source']), '',
        '## Timing', '', '- %s' % json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in s['timing'].items() if k != 'total'}),
        '- totals: %s' % json.dumps({k: round(v, 2) for k, v in s['timing']['total'].items()}), '', s['scope'], '']
    return '\n'.join(lines)


def write_summary(out, positions, metas):
    out = Path(out);summary = summarize(positions, metas)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=str) + '\n')
    (out / 'summary.md').write_text(render(summary))
    return summary


def report(inputs, out):
    """Merge shard outputs: positions concatenated in shard order; the summary recomputed from them."""
    out = Path(out);out.mkdir(parents=True, exist_ok=False)
    metas, positions = [], []
    for directory in inputs:
        directory = Path(directory)
        metas.extend(json.loads((directory / 'summary.json').read_text())['runs'])
        positions.extend(json.loads(line) for line in (directory / 'positions.jsonl').read_text().splitlines() if line)
    keys = ('config_sha256', 'iteration', 'block', 'shards', 'max_games', 'world_blocks', 'search_seed', 'sample_seed', 'policy')
    for key in keys:
        if len({json.dumps(m.get(key)) for m in metas}) > 1:raise SystemExit('shards differ in ' + key)
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

def build_setup(args):
    """The offline path's own preparation: v2_fused.prepare(--config), the champion binding, the block's games and its
    swapped copy (v2_search_head_to_head.main), make_step, and Search's source/materializer/executor/recipes."""
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
    if not picked:return None
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
    meta = dict(shard=args.shard, shards=args.shards, config=args.config, config_sha256=context['declared'].get('config_sha256'),
        iteration=args.iteration, block=args.block, max_games=args.max_games, games=list(picked), world_blocks=args.world_blocks,
        search_seed=args.search_seed, champion=record.evaluator, sampling_seed=batch.sampling_seed, max_requests=plan.max_requests,
        backend=jax.default_backend())
    return Setup(executor, source, materializer, recipes, decide, apply, (params, params), np.repeat(np.asarray([0, 1], np.int32), n),
        np.tile(np.asarray(picked, np.int32), 2), np.repeat(np.asarray([0, 1], np.int32), n), fresh, meta)


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
    setup = build_setup(args)
    if setup is None:
        Path(args.out).mkdir(parents=True);(Path(args.out) / 'positions.jsonl').write_text('')
        write_summary(args.out, [], [dict(shard=args.shard, shards=args.shards, config=args.config, games=[], steps=0)]);return 0
    run_shard(setup, args.out, max_steps=args.max_steps, quota=shard_quota(args.sample_decisions, args.shards),
        sample_rate=args.sample_rate, sample_seed=args.sample_seed, policy=args.policy)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
