"""Two-pass regression replayer over V2 capture shards (fidelity-patch design 5.2).

`base` runs the jitted transition.step_with_public of this tree on every
captured (state, actions, trackers) input and stores, per shard, a 64-bit
digest of every output leaf for every lane and side (leaves with a side axis
are digested per side), plus the post arrays of the faint-clear tracker leaves
on lanes where an active faints. `patched` runs a later tree on the same
shards. Both passes migrate their inputs by leaf path (existing leaves keep
shape and dtype; new state leaves take empty_state's values, new tracker
leaves their declared neutral values; the filled leaves are recorded), so a
base pass also runs on a capture that predates a leaf. In the patched pass
every changed lane-step is attributed through the per-PR plug-in table:

- involvement predicates (pre-state and actions only, a conservative superset)
  name the effects a lane may exercise and the side that acts;
- each effect declares the leaves it may change on the acting (user) side, the
  other (target) side and side-less leaves. A change outside the union of the
  involved effects' allowances is changed-involved-unexpected-fields, a
  failure, unless the leaf is waived (--waive) or it is a KO consequence on a
  lane whose set of fainting mons differs from the base (a KO cascade): HP,
  fainted, forced-switch/replacement and done/winner leaves, the TURN-line
  bookkeeping of both sides, the newly (un)fainted mon's own per-mon state (its
  PP included: a mon KO'd before its turn spends none), its side's action history
  and the step's executed-move count, a cancelled or cleared last_move (an
  active's or a fainted mon's) and a released
  trap link (a fainted or departed mon's own, or one pointing at such a foe; a
  departure is a pre-step active replaced in either tree), each checked per mon
  against stored base arrays, and the trackers of its side. A lane whose
  residual ran in exactly one tree is a cascade on both sides that also accepts
  the residual handlers' state writes and the tracker families they and the
  end-of-turn hooks write (not every tracker): a phase shift (the residual was
  deferred in exactly one tree: a KO moved between the action and residual
  phases) or a terminal step (the battle ended in either tree) whose residual
  ran in exactly one tree, which both passes record per lane. Any other terminal
  step accepts only the end-of-turn writes gated on the battle ending (the
  survivor's Truant flip and turn_end's upkeep). All are counted and flagged per
  lane. Anything else stays unexpected.

Report: unchanged, changed-and-involved per effect with its changed fields,
changed-but-uninvolved and changed-involved-unexpected-fields (both must be
0), involved-but-unchanged; every changed lane-step is listed.

Special rules (PR0): `trackers.public_order` is checked against a numpy replay
of its swap rule over the recorded install calls instead of bitwise while the
base pass lacks it (bitwise once the base has it too), and with
--faint-clear the PR0 faint-time public clear may change only the listed
tracker entries on lanes where an active faints (meanlook_trap: the actives'
entries). Leaves the faint rule explains are removed from the lane-step's
changes; any other changed leaf on it still needs an involved effect. PR0's
remaining changes (the Intimidate and weather reveal gates, the Color Change
type revert on departure, the Mean Look clear on both seats at every arrival)
are the `PR0` plug-in; its Mean Look predicate also reads the tracker pre-value.

Special rule (R): while the base pass lacks `state.effect_draws`, the leaf is
checked against an independent copy of the design 2.1 derivation (the effect
block a deferred step saves, the resumed block or zero otherwise) instead of
against its migrated default, so the appended stream itself is never involvement.

The involvement superset resolves the executed move as the engine does (slot,
charge, standing or same-turn Encore, Struggle, a resumed queued action, every
Sleep Talk-callable move, a Magic Coat reflection) and covers Effect Spore and
Yawn sleep under Sleep Clause, Rest or Effect Spore against Insomnia/Vital
Spirit (a Trace holder copying either, any install of a holder), and an incoming
White Herb holder. Known gaps, which can only cause false failures: Snatch,
Mirror Move, Metronome and Assist calls (not in the Gen 3 randbats pool); the
recharge override (M1 adds it).

Both passes must run on the same JAX version, digest scheme and lane padding.
Paths are command-line arguments.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

SCHEMA = 'pokezero.v2e_regress.v2'
DIGEST_FORMAT = 'per-lane-per-side-v2'
ROOT = Path(__file__).resolve().parents[1]
DIGEST_SEEDS = (0x243F6A88, 0x85A308D3)
# Leaves whose second axis is not a battle side.
NOT_SIDE = ('state.keys',)
# PR0's faint-time public clear (design 2.5): these tracker entries of the
# fainted mon may be cleared; meanlook_trap may be cleared on both actives.
FAINT_CLEAR_MON = ('confusion_turns', 'encore_turns', 'wrap_trap_turns', 'stall_counter', 'sub_present',
                   'item_swapped', 'choice_locked')
FAINT_CLEAR_SEATS = ('meanlook_trap',)
FAINT_CLEAR_LEAVES = tuple(['trackers.public_volatiles'] + [f"trackers.mon_features['{k}']" for k in FAINT_CLEAR_MON + FAINT_CLEAR_SEATS])
PUBLIC_ORDER = 'trackers.public_order'
# Fidelity R (design 2.1): the saved effect stream. While the base pass predates the leaf it is checked
# against an independent copy of the design's derivation (EFFECT_SALT, EFFECT_BOUNDS) instead of bitwise.
EFFECT_DRAWS = 'state.effect_draws'
EFFECT_SALT, EFFECT_BOUNDS = 1732050, (16, 16, 4, 2)
MEANLOOK = "trackers.mon_features['meanlook_trap']"
PRE_TRACKERS = (MEANLOOK, 'trackers.active_slot')
TRAP_MOVES = ('meanlook', 'spiderweb', 'block')
DEFAULT_DESTINY_BOND = 64
SLEEP_MOVES = ('spore', 'hypnosis', 'sleeppowder', 'sing', 'lovelykiss', 'grasswhistle')


def log(message, **values):
    print(json.dumps(dict(message=message, time=time.strftime('%H:%M:%S'), **values), default=str), flush=True)


# ---------------------------------------------------------------------------
# Leaf naming. Inputs (state, actions, trackers) and the PublicStepResult
# output share names for the state and tracker leaves, so a post leaf can be
# compared with its own (migrated) pre value.

def _named(prefix, tree):
    import jax
    return {prefix + jax.tree_util.keystr(path): leaf for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]}


def name_inputs(state, actions, trackers):
    return {**_named('state', state), 'actions': actions, **_named('trackers', trackers)}


def name_outputs(out):
    step = out.step
    return {**_named('state', step.state), 'step.invalid_actions': step.invalid_actions,
            'step.placeholder_moves': step.placeholder_moves, **_named('trackers', out.trackers)}


def capture_name(path):
    """capture.json leaf path ([0].x / [1] / [2].y) -> input name."""
    for prefix, name in (('[0]', 'state'), ('[2]', 'trackers')):
        if path.startswith(prefix):return name + path[len(prefix):]
    if path == '[1]':return 'actions'
    raise ValueError(f'unexpected capture leaf path {path}')


def has_side_axis(name, shape, lanes):
    return len(shape) >= 2 and shape[0] == lanes and shape[1] == 2 and name not in NOT_SIDE


# ---------------------------------------------------------------------------
# Digests: per lane (and side), two independent 32-bit sums of a bijective mix
# of every element (so any single-element change always changes both words).

def _as_u32(x):
    import jax
    import jax.numpy as j
    if x.dtype == j.bool_:return x.astype(j.uint32)
    size = x.dtype.itemsize
    width = {1: j.uint8, 2: j.uint16, 4: j.uint32, 8: j.uint32}[size]
    if size == 8:return jax.lax.bitcast_convert_type(x, j.uint32)   # [...,2]
    u = x if x.dtype == width else jax.lax.bitcast_convert_type(x, width)
    return u.astype(j.uint32)


def _fmix(h):
    import jax.numpy as j
    h = h ^ (h >> 16);h = h * j.uint32(0x85EBCA6B);h = h ^ (h >> 13);h = h * j.uint32(0xC2B2AE35)
    return h ^ (h >> 16)


def lane_digest(x, lanes):
    import jax.numpy as j
    x = j.asarray(x)
    if x.ndim == 0 or x.shape[0] != lanes:x = j.broadcast_to(x, (lanes,) + x.shape)   # a lane-global leaf
    u = _as_u32(x).reshape(lanes, -1)
    index = j.arange(u.shape[1], dtype=j.uint32)[None, :] * j.uint32(0x9E3779B9)
    return j.stack([j.sum(_fmix(u ^ (index + j.uint32(seed))), axis=1, dtype=j.uint32) for seed in DIGEST_SEEDS], axis=1)


def side_digest(name, x, lanes):
    """[L,2 sides,2 words]: per side for a side-axis leaf, else the whole digest on both sides."""
    import jax.numpy as j
    x = j.asarray(x)
    if has_side_axis(name, x.shape, lanes):return j.stack([lane_digest(x[:, s], lanes) for s in range(2)], axis=1)
    whole = lane_digest(x, lanes)
    return j.stack([whole, whole], axis=1)


def digest_named(named, lanes):
    import jax.numpy as j
    names = list(named)
    return names, j.stack([side_digest(n, named[n], lanes) for n in names], axis=1)   # [L, N, 2, 2]


# ---------------------------------------------------------------------------
# Migration by leaf path.

class MigrationError(ValueError):
    pass


def tracker_default(name, template):
    """Declared neutral values for new tracker leaves (never a constructed game's lane 0)."""
    if name == PUBLIC_ORDER:
        if template.shape[1:] != (2, 6):raise MigrationError(f'{name}: expected [B,2,6], tree has {template.shape}')
        return lambda lanes: np.broadcast_to(np.arange(6, dtype=template.dtype), (lanes, 2, 6))
    if re.match(r"^trackers\.(mon_features|field_features)\[", name):
        return lambda lanes: np.zeros((lanes,) + template.shape[1:], template.dtype)
    raise MigrationError(f'new tracker leaf {name} has no declared neutral value; add it to tracker_default()')


def migrate(capture_names, leaves, template_named, *, defaults=None):
    """Shard leaves (capture order) -> {name: array} in the template's leaf set.

    Existing leaves must keep their per-lane shape and dtype. A template leaf
    absent from the capture takes `defaults[name](lanes)` (patched_pass: empty_state
    for state leaves), else its declared tracker neutral value; the default must
    match the template's per-lane shape and dtype. A captured leaf missing from
    the template is refused.
    """
    defaults = defaults or {}
    lanes = int(leaves[0].shape[0])
    stored = dict(zip(capture_names, leaves))
    dropped = sorted(set(stored) - set(template_named))
    if dropped:raise MigrationError(f'captured leaves missing from this tree: {dropped[:8]}')
    out, filled = {}, []
    for name, template in template_named.items():
        template = np.asarray(template)
        if name in stored:
            value = np.asarray(stored[name])
            if value.shape[1:] != template.shape[1:] or value.dtype != template.dtype:
                raise MigrationError(f'{name}: captured {value.shape[1:]}{value.dtype} vs tree {template.shape[1:]}{template.dtype}')
            out[name] = value
        else:
            fill = defaults.get(name) or tracker_default(name, template)
            value = np.asarray(fill(lanes))
            if value.shape != (lanes,) + template.shape[1:] or value.dtype != template.dtype:
                raise MigrationError(f'{name}: default {value.shape}{value.dtype} vs tree {template.shape[1:]}{template.dtype}')
            out[name] = np.ascontiguousarray(value);filled.append(name)
    return out, filled


def pad_lanes(named, lanes):
    """Pad every lane-batched leaf to `lanes` with copies of lane 0 (one compile for every shard)."""
    current = int(np.asarray(named['state.hp']).shape[0])
    if lanes <= current:return named, current
    def pad(x):
        x = np.asarray(x)
        if x.ndim == 0 or x.shape[0] != current:return x
        return np.concatenate([x, np.repeat(x[:1], lanes - current, axis=0)], axis=0)
    return {k: pad(v) for k, v in named.items()}, current


# ---------------------------------------------------------------------------
# Involvement predicates (numpy, pre-state and actions only). Each returns
# [L,2]: the side(s) that exercise the effect this step.

class Pre:
    """Numpy view of a migrated pre-state plus the move resolution of the step."""
    def __init__(self, named, vocab, sleep_talk_callable, struggle_mask, *, destiny_bond=DEFAULT_DESTINY_BOND):
        self.s = {k[len('state.'):]: np.asarray(v) for k, v in named.items() if k.startswith('state.')}
        # Tracker pre-values a predicate may read (PRE_TRACKERS): a public flag can outlive its engine link.
        self.t = {k[len('trackers.'):]: np.asarray(v) for k, v in named.items() if k.startswith('trackers.')}
        self.actions = np.asarray(named['actions']);self.vocab = vocab;self.destiny_bond = destiny_bond
        self.lanes = self.actions.shape[0];self.ids = {m: i for i, m in enumerate(vocab['moves'])}
        self.sleep_talk_callable = np.asarray(sleep_talk_callable);self.struggle = np.asarray(struggle_mask)
        self.candidates = self._candidates()

    def active(self, name):
        rows = np.arange(self.lanes)[:, None];sides = np.arange(2)[None, :]
        return self.s[name][rows, sides, self.s['active']]

    def _candidates(self):
        """[L,2,K] move ids a side may execute this step (0 = none); a superset.

        Action slot, charge lock, Encore (standing, or applied this turn by a foe's
        Encore to the side's last executed slot), Struggle, the queued action on a
        resumed turn, every Sleep Talk-callable move, and a foe's move reflected
        by this side's Magic Coat."""
        s = self.s;rows = np.arange(self.lanes)[:, None];sides = np.arange(2)[None, :]
        saved = np.where(s['pending_residual'], s['queued_actor'], -1)
        acts = np.where(saved[:, None] == sides, s['queued_actions'], self.actions)
        slot = np.clip(acts, 0, 3)
        chosen = np.where(acts < 4, s['moves'][rows, sides, s['active'], slot], 0)
        charge = self.active('charge_move')
        encore = np.where(self.active('encore_counter') > 0, self.active('encore_move'), 0)
        struggle = np.where(self.struggle, self.ids.get('struggle', 0), 0)
        # Same-turn Encore: a faster foe's Encore overrides this side's queued move
        # with its last executed slot (encore.override reads action history).
        index = s['last_action_index'];valid = (s['last_action_kind'] == 2) & (index >= 0) & (index < 4)
        last = np.where(valid, s['moves'][rows, sides, s['active'], np.clip(index, 0, 3)], 0)
        talk = self.ids.get('sleeptalk', -1)
        own = s['moves'][rows, sides, s['active']]                      # [L,2,4]
        def expand(base):
            called = np.where(self.sleep_talk_callable[own] & (base == talk).any(-1, keepdims=True), own, 0)
            return np.concatenate([base, called], axis=-1)
        base = np.stack([chosen, charge, encore, struggle], axis=-1)
        foe_encore = (expand(base) == self.ids.get('encore', -1)).any(-1)[:, ::-1]
        candidates = expand(np.concatenate([base, np.where(foe_encore, last, 0)[..., None]], axis=-1))
        coat = (candidates == self.ids.get('magiccoat', -1)).any(-1)    # [L,2]
        reflected = np.where(coat[..., None], candidates[:, ::-1], 0)
        return np.concatenate([candidates, reflected], axis=-1)

    def side_uses(self, *names):
        ids = [self.ids[n] for n in names if n in self.ids]
        return np.isin(self.candidates, ids).any(axis=2) if ids else np.zeros((self.lanes, 2), bool)

    def uses(self, *names):
        return self.side_uses(*names).any(axis=1)

    def ability_ids(self, *names):
        table = self.vocab['abilities'];return [table.index(n) for n in names if n in table]

    def active_ability(self, *names):
        return np.isin(self.active('ability'), self.ability_ids(*names))


def sleepers_besides_target(p):
    """[L,2] a side holds a living non-Rest sleeper that is not its (possibly switching) active."""
    s = p.s;slp = 5
    sleepers = (s['hp'] > 0) & (s['status'] == slp) & ~s['rest_origin'].astype(bool)     # [L,2,6]
    other = np.arange(6)[None, None, :] != s['active'][..., None]
    switching = (p.actions >= 4)[..., None]          # a side switching may bring the sleeper in as target
    return (sleepers & (other | switching)).any(-1)


def sleep_clause(p):
    """A sleep attempt while the slept side has another living non-Rest sleeper: a primary
    sleep move or Yawn by the actor; Effect Spore on the actor's contact; a pending Yawn."""
    others = sleepers_besides_target(p)
    by_move = p.side_uses(*SLEEP_MOVES, 'yawn') & others[:, ::-1]
    spore = p.active_ability('effectspore')[:, ::-1] & others & (p.candidates > 0).any(-1)
    drowsy = ((p.s['yawn_counter'] > 0).any(-1) & others)[:, ::-1] if 'yawn_counter' in p.s else np.zeros_like(by_move)
    return by_move | spore | drowsy


def sleep_immunity(p):
    """A sleep move or Yawn aimed at an Insomnia/Vital Spirit side; Rest by such a holder;
    such a holder hitting an Effect Spore foe. A Trace holder counts as a holder while the
    opposing party holds one (it copies the foe's ability as it enters), and a benched holder
    is exposed by any install this step (a switch, Baton Pass, the foe's Roar/Whirlwind, a
    replacement)."""
    ids = p.ability_ids('insomnia', 'vitalspirit')
    holder = np.isin(p.s['ability'], ids)                                                # [L,2,6]
    holder = holder | (np.isin(p.s['ability'], p.ability_ids('trace')) & holder.any(-1)[:, ::-1, None])
    rows = np.arange(p.lanes)[:, None];sides = np.arange(2)[None, :]
    active = holder[rows, sides, p.s['active']]
    exposed = active | (holder.any(-1) & installs(p))                                   # per holder side
    aimed = p.side_uses(*SLEEP_MOVES, 'yawn') & exposed[:, ::-1]
    rest = p.side_uses('rest') & active
    spore = active & p.active_ability('effectspore')[:, ::-1] & (p.candidates > 0).any(-1)
    return aimed | rest | spore


def counter(name):
    def predicate(p):
        return (p.s[name] > 0).any(axis=2) if name in p.s else np.zeros((p.lanes, 2), bool)
    return predicate


def destiny_bond_bit(p):
    return ((p.s['volatile'] & p.destiny_bond) != 0).any(axis=2)


def installs(p):
    """[L,2] a side may install a new active this step (and so its active departs): a switch
    action, Baton Pass, the foe's Roar/Whirlwind, a replacement (forced switch or fainted active)."""
    rows = np.arange(p.lanes)[:, None];sides = np.arange(2)[None, :]
    active_hp = p.s['hp'][rows, sides, p.s['active']]
    return ((p.actions >= 4) | p.side_uses('batonpass') | p.side_uses('roar', 'whirlwind')[:, ::-1]
            | p.s['forced_switch'].astype(bool) | (active_hp <= 0))


def white_herb(p):
    """An active White Herb holder, or one that may come in this step (switch, Baton
    Pass, the foe's Roar/Whirlwind, a replacement)."""
    items = p.vocab['items']
    if 'whiteherb' not in items:return np.zeros((p.lanes, 2), bool)
    herb = items.index('whiteherb');holders = p.s['item'] == herb                       # [L,2,6]
    return (p.active('item') == herb) | (holders.any(-1) & installs(p))


def entering_ability(*names):
    """PR0 reveal gates: a side may install a holder of one of these abilities this step (the
    reveal happens at its switch-in; the foe's HP/Substitute or the weather is not predicted).
    A voluntary switch installs exactly its target, bench_indices(active)[action-4]; a
    replacement, a Baton target or a drag may install any benched holder."""
    def predicate(p):
        holder = np.isin(p.s['ability'], p.ability_ids(*names))                         # [L,2,6]
        rows = np.arange(p.lanes)[:, None];sides = np.arange(2)[None, :]
        active_hp = p.s['hp'][rows, sides, p.s['active']]
        forced = p.s['forced_switch'].astype(bool) | (active_hp <= 0)
        slot = np.clip(p.actions - 4, 0, 4);target = slot + (slot >= p.s['active'])   # state.bench_indices
        voluntary = (p.actions >= 4) & ~forced & holder[rows, sides, target]
        other = forced | p.side_uses('batonpass') | p.side_uses('roar', 'whirlwind')[:, ::-1]
        return voluntary | (other & holder.any(-1))
    return predicate


def meanlook_arrival(p):
    """PR0 Mean Look arrival clear (B2, Baton quirk included): any arrival clears both seats'
    public flag. Involved on both sides when a side may install this step and a flag may be set:
    one on an active's entry in the tracker pre-state (it can outlive the engine link), an
    engine trap link, or a trapping move this step."""
    held = np.zeros((p.lanes, 2), bool)
    flag, slot = p.t.get(MEANLOOK[len('trackers.'):]), p.t.get('active_slot')
    if flag is not None and slot is not None:
        # An arrival clears the actives' entries only (the incoming one is zeroed in both trees).
        rows = np.arange(p.lanes)[:, None];sides = np.arange(2)[None, :]
        held |= (slot >= 0) & (flag[rows, sides, np.clip(slot, 0, 5)] > 0)
    if 'trap_source' in p.s:held |= (p.s['trap_source'] >= 0).any(-1)
    held |= p.side_uses(*TRAP_MOVES)
    lane = installs(p).any(-1) & held.any(-1)
    return np.repeat(lane[:, None], 2, axis=1)


def type_revert(p):
    """PR0 Kecleon revert: a departing active with a live type overlay (types != base types),
    or a Color Change active that a same-step hit may change before it departs (Pursuit)."""
    changed = (p.active('types') != p.active('base_types')).any(-1) if 'base_types' in p.s else np.zeros((p.lanes, 2), bool)
    return (changed | p.active_ability('colorchange')) & installs(p)


def confusion_active(p):
    return p.active('confusion_counter') > 0


def move(*names):
    return lambda p: p.side_uses(*names)


def ooze_leech_seed(p):
    """R: a side holding Liquid Ooze in its party while Leech Seed may drain it at this step's residual: one of its
    mons carries the seed (Baton Pass moves it to the recipient), or the foe may use Leech Seed now."""
    ooze = np.isin(p.s['ability'], p.ability_ids('liquidooze')).any(-1)
    seeded = ((np.asarray(p.s['volatile']).astype(np.uint32) & 4) != 0).any(-1)
    return ooze & (seeded | p.side_uses('leechseed')[:, ::-1])


def substitute_holder(p):
    """A1 (design 13): a side whose active may faint behind a Substitute this step, whose
    Substitute the faint now clears: one up in the pre-state (a Baton recipient inherits
    the passer's), or a Substitute move this step."""
    held = p.active('substitute_hp') > 0 if 'substitute_hp' in p.s else np.zeros((p.lanes, 2), bool)
    return held | p.side_uses('substitute')


# Leaf families for the allowed-change sets (regexes over output names).
FAMILIES = {
    'hp': (r'^state\.(hp|substitute_hp)$', r'^trackers\.(hp_percent|damage_exact_(dealt|taken))$',
           r'^trackers\.(damage_history|negative_items)\.', r"^trackers\.mon_features\['last_damage_(dealt|taken)'\]$"),
    'boosts': (r'^state\.boosts$', r'^trackers\.boosts$'),
    'status': (r'^state\.(status|status_counter|sleep_skipped|rest_origin)$', r'^trackers\.status$', r'^trackers\.public_volatiles$',
               r"^trackers\.mon_features\['(sleep_turns|sleep_skipped_turns|sleep_cant_pending|sleep_usable_pending|rest_sleep|toxic_stage)'\]$"),
    'item': (r'^state\.item$', r'^trackers\.(shown_item|item_removed|belief_mask)$', r'^trackers\.(visible_belief|negative_items)\.',
             r"^trackers\.mon_features\['item_swapped'\]$"),
    'reveal': (r'^trackers\.(shown_ability|original_ability|borrowed_ability|belief_mask)$', r'^trackers\.visible_belief\.'),
    'types': (r'^trackers\.shown_types$',),
    'meanlook': (r"^trackers\.mon_features\['meanlook_trap'\]$",),
    # A1: a status write updates the sleep-clause port (holder on the other seat, victims
    # on the cured one) and refreshes the subject's visible belief basis.
    'port': (r'^trackers\.public_port\.',),
    'visible': (r'^trackers\.visible_belief\.',),
    'sub': (r'^state\.substitute_hp$',),
    # A2 review L3: a status write touches only the sleep-clause entries of the port.
    'clause': (r'^trackers\.public_port\.(holder|victims)$',),
    # A2: Knock Off's permanent flag and B2's removal credit; the end-of-turn Choice unlock; Destiny
    # Bond's volatile bit and bag; Color Change's live type (state and shown); a Protect-blocked count.
    'knock': (r'^state\.item_knocked$', r"^trackers\.field_features\['items_removed_credit'\]$"),
    'choice': (r'^state\.choice_move$',),
    'bond': (r'^state\.volatile$', r'^trackers\.public_volatiles$'),
    'typechange': (r'^state\.types$', r'^trackers\.shown_types$'),
    'blocked': (r'^trackers\.public_port\.blocked$',),
    # A2: the device's Substitute presence (feature and bag bit) when a hit now breaks one; a pending
    # Wish lands with a heal line, or at full HP stays pending, by the recipient's HP.
    'subtracker': (r"^trackers\.mon_features\['sub_present'\]$", r'^trackers\.public_volatiles$'),
    'wish': (r"^trackers\.field_features\['wish_(pending|set_turn|turns)'\]$",),
    'all': (r'.*',),
}
ALL = dict(user=('all',), target=('all',), glob=('all',))
# KO cascade: leaves any faint difference may change (both sides / side-less).
FAMILIES['ko'] = (r'^state\.(forced_switch|pending_residual|queued_actor|queued_actions|queued_stall|residual_draws|'
                  r'phazing_draws|secondary_draws|sleep_talk_draws|replacement_count|done|winner)$', r'^trackers\.fainted$',
                  # R: a mon that faints before its turn executes no move: the step's executed-move count.
                  r'^step\.placeholder_moves$')
# R: ...nor records an action: its side's action history, accepted on a side with a newly (un)fainted mon.
FAMILIES['ko_action'] = (r'^state\.(last_action_kind|last_action_index)$',)
# A1: a KO cancels the remaining queued action (that actor's last_move) and the end-of-step clear zeroes a
# fainted mon's last_move; linking.prune releases a link whose owner or target fainted, on either side.
# A2 (review L2): accepted per mon against stored base arrays (KO_LINK_LEAVES, link_ok), never as whole leaves.
KO_LINK_LEAVES = ('state.last_move', 'state.trap_source')
# A1: whether a step ends in a forced-replacement request or a new TURN line follows the faint set;
# B2's per-turn bookkeeping follows it on both sides (turn counter, the actives' stint clocks, a
# badly poisoned active's toxic stage, the post-upkeep port flag, weather upkeeps).
FAMILIES['turnline'] = (r'^trackers\.turn$', r"^trackers\.mon_features\['(turns_active|turns_active_total|toxic_stage)'\]$",
                        r'^trackers\.public_port\.post_upkeep$', r"^trackers\.field_features\['weather_upkeeps'\]$")
# A1: a phase shift. An involved effect that changes damage taken (Curse, Superpower, Charm) or a
# cure (Refresh, Aromatherapy) can move a KO between the action phase, which defers the residual
# (state.pending_residual and the saved draws), and the residual phase. The residual handlers and
# the ordinary end-of-turn clears then ran in exactly one tree: their writes may differ on both
# sides (with all trackers), even when the set of fainted mons is the same.
FAMILIES['residual'] = (r'^state\.(item|status_counter|volatile|boosts|encore_move|encore_counter|charge_move|charge_counter|'
                        r'truant_loaf|wish_counter|wish_amount|weather|weather_counter)$', r'^trackers\.(weather|field_features)')
# A2 (review L1): the tracker families the residual handlers (Wish, weather, Speed Boost, Leftovers, pinch
# berries, status, Leech Seed) and turn_end (observe, faint clear, clocks, upkeep, negative items, TURN)
# write. On a phase shift these are accepted on both seats instead of every tracker.
FAMILIES['residual_trackers'] = (
    r'^trackers\.(hp_percent|status|boosts|fainted|weather|spikes|turn|shown_item|item_removed|belief_mask|shown_ability|'
    r'original_ability|borrowed_ability|public_volatiles|damage_exact_(dealt|taken))$',
    r'^trackers\.(damage_history|negative_items|visible_belief|public_port)\.', r'^trackers\.field_features\[',
    r"^trackers\.mon_features\['(sleep_turns|rest_sleep|toxic_stage|sleep_skipped_turns|sleep_cant_pending|sleep_usable_pending|"
    r"confusion_turns|encore_turns|wrap_trap_turns|stall_counter|sub_present|meanlook_trap|item_swapped|choice_locked|"
    r"last_damage_dealt|last_damage_taken|turns_active|turns_active_total)'\]$")
# A2 review L1: a terminal step whose residual ran in both trees or in neither. Only the end-of-turn writes gated
# on `done` differ: the Truant flip of the survivor (transition, `ordinary & ~work.done`) and turn_end's upkeep
# (`residual_ready & ~work.done`: the Sleep Talk pending clocks, the port's post-upkeep flag, negative items).
FAMILIES['terminal'] = (r'^state\.truant_loaf$',
    r"^trackers\.mon_features\['(sleep_skipped_turns|sleep_cant_pending|sleep_usable_pending)'\]$",
    r'^trackers\.public_port\.post_upkeep$', r'^trackers\.negative_items\.')
# R: a Focus Punch user losing focus (state.volatile's FOCUS_LOST bit) to a hit that now reaches it.
FAMILIES['focus'] = (r'^state\.volatile$',)
# Per-mon state a faint may change on the mon that (un)fainted; checked per mon against stored base arrays.
KO_MON_LEAVES = tuple(f'state.{n}' for n in ('status', 'status_counter', 'sleep_skipped', 'rest_origin', 'volatile',
    'confusion_counter', 'boosts', 'encore_move', 'encore_counter', 'charge_move', 'charge_counter', 'choice_move',
    'protect_counter', 'truant_loaf', 'pp'))      # pp (R): a mon KO'd before its turn spends none
# effect -> leaves it may change: user (acting side), target (the other side), glob (side-less).
# A1's sets: a stat change alters damage taken later in the step (and pinch berries);
# a sleep landing or not decides whether the target acts at all, so it may change anything.
ALLOWED = {
    'curse': dict(user=('boosts', 'hp', 'item'), target=('hp',), glob=()),
    'superpower': dict(user=('boosts', 'hp', 'item'), target=('hp',), glob=()),
    'charm': dict(user=('hp',), target=('boosts', 'hp', 'item', 'reveal'), glob=()),
    'lovelykiss': ALL, 'sleep_clause': ALL, 'sleep_immunity': ALL,
    # A cure goes through party_cure/observe: port.condition and team_cure write the sleep-clause
    # port (victims on the user's seat, holder on the foe's) and _refresh the visible belief.
    'aromatherapy': dict(user=('status', 'hp', 'clause', 'visible'), target=('hp', 'clause'), glob=()),
    'refresh': dict(user=('status', 'hp', 'clause', 'visible'), target=('hp', 'clause'), glob=()),
    # The faint clears only the holder's Substitute HP (no tracker reads it).
    'sub_faint': dict(user=('sub',), target=(), glob=()),
    # PR0 (tracker only): a gated reveal drops the entrant's own ability reveal; the
    # Color Change revert rewrites the departing mon's shown types. No state, no foe field.
    'intimidate_reveal': dict(user=('reveal',), target=(), glob=()),
    'weather_reveal': dict(user=('reveal',), target=(), glob=()),
    'type_revert': dict(user=('types',), target=(), glob=()),
    # B2 clears the flag on both seats at an arrival; both sides are actors.
    'meanlook_arrival': dict(user=('meanlook',), target=('meanlook',), glob=()),
    # A2. Knock Off: the target's item, flag, credit, end-of-turn Choice unlock and Sticky Hold reveal;
    # without its item the target loses Leftovers (HP), pinch berries (stages), Lum (a status a contact
    # ability gives it later stays, with the clause port) and Choice Band (less damage to the user, whose
    # own HP-gated items follow).
    'knockoff': dict(user=('hp', 'item', 'boosts', 'clause', 'wish'),
                     target=('item', 'knock', 'choice', 'hp', 'boosts', 'status', 'clause', 'visible', 'reveal'), glob=()),
    # Giga Drain: the drainer's HP (heal, or Liquid Ooze damage) and its HP-gated items; the Ooze
    # holder's reveal; a later Leech Seed drain from the drainer reads its HP (the target's HP).
    'gigadrain': dict(user=('hp', 'item', 'boosts', 'visible', 'wish'), target=('hp', 'reveal'), glob=()),
    # Counter/Mirror Coat: the damage and its consequences on the target (HP-gated items, Color Change,
    # contact-ability reveals); contact abilities hurt or status the Counter user (Lum, clause port);
    # a failed attempt no longer reaches the foe's Protect count.
    'counter': dict(user=('hp', 'status', 'item', 'boosts', 'clause', 'visible', 'blocked'),
                    target=('hp', 'item', 'boosts', 'typechange', 'reveal', 'clause', 'visible', 'subtracker', 'wish'), glob=()),
    'mirrorcoat': dict(user=('hp', 'item', 'boosts', 'blocked'),
                       target=('hp', 'item', 'boosts', 'typechange', 'reveal', 'visible', 'subtracker', 'wish'), glob=()),
    # Destiny Bond: the user's bit and bag; its trigger faints the foe (KO cascade rules for the rest).
    'destinybond': dict(user=('bond',), target=('hp',), glob=()),
    'destinybond_trigger': dict(user=('bond', 'hp'), target=('hp',), glob=()),
    # R. Bonemerang's second hit: more damage to the target or its Substitute (the device's presence and bag
    # when the second hit breaks it), and what follows from the target's HP: its HP-gated items (pinch berry
    # and stages), a pending Wish landing by HP, Color Change and its reveal when the second hit is the first
    # to reach it, the belief refresh of a changed condition, a Focus Punch losing focus. The user's HP follows
    # from the target's reply (Counter/Mirror Coat read the second hit, a lost Focus Punch) and its HP items.
    'bonemerang': dict(user=('hp', 'item', 'boosts', 'visible'),
                       target=('hp', 'item', 'boosts', 'typechange', 'reveal', 'visible', 'subtracker', 'wish', 'focus'),
                       glob=()),
    # R (optional carry from A2): Liquid Ooze on Leech Seed. The holder (user) is revealed; the seeder (target) takes
    # damage instead of healing, and a slower seeder's later residual handlers (a pinch berry, its condition) follow.
    'ooze_seed': dict(user=('reveal',), target=('hp', 'visible', 'item', 'boosts'), glob=()),
}


# Allowances an effect earns only on a KO cascade in which the other side (its target's) has a newly (un)fainted
# mon: R: a second-hit KO of a slower target cancels its queued move, which can then no longer status the user
# (seen: a thawing Parasect's Spore; the Sleep Clause port and the public status bag follow the status).
KO_ALLOWED = {'bonemerang': dict(user=('status', 'clause'))}


def allowed_leaf(name, families):
    return any(re.search(p, name) for f in families for p in FAMILIES[f])


# Per-PR plug-in table: effect -> predicate. Choice unlock and the effect stream
# are never involvement (design 5.2). `none` is the base-against-base self-test.
PLUGINS = {
    'none': {},
    # PR0's public_order and faint-clear rules are special rules (see the docstring);
    # these are its remaining tracker changes.
    'PR0': dict(intimidate_reveal=entering_ability('intimidate'),
                weather_reveal=entering_ability('drizzle', 'drought', 'sandstream'), type_revert=type_revert,
                meanlook_arrival=meanlook_arrival),
    'A1': dict(curse=move('curse'), superpower=move('superpower'), charm=move('charm'), lovelykiss=move('lovelykiss'),
               aromatherapy=move('aromatherapy'), refresh=move('refresh'), sleep_clause=sleep_clause,
               sleep_immunity=sleep_immunity, sub_faint=substitute_holder),
    'A2': dict(knockoff=move('knockoff'), gigadrain=move('gigadrain'), counter=move('counter'), mirrorcoat=move('mirrorcoat'),
               destinybond=move('destinybond'), destinybond_trigger=destiny_bond_bit),
    'R': dict(bonemerang=move('bonemerang'), ooze_seed=ooze_leech_seed),
    'B': dict(irontail=move('irontail'), signalbeam=move('signalbeam'), steelwing=move('steelwing'),
              meteormash=move('meteormash'), headbutt=move('headbutt'), needlearm=move('needlearm'),
              confusion=confusion_active),
    'M1': dict(hyperbeam=move('hyperbeam'), trick=move('trick'), recharge=counter('recharge_counter')),
    'M2': dict(perishsong=move('perishsong'), yawn=move('yawn'), wrap=move('wrap'), perish=counter('perish_counter'),
               drowsy=counter('yawn_counter'), partial_trap=counter('partial_trap_counter')),
    'C': dict(psychoboost=move('psychoboost'), whiteherb=white_herb, sacredfire=move('sacredfire'), facade=move('facade')),
}


def predicates(prs):
    table = {}
    for pr in prs:
        if pr not in PLUGINS:raise ValueError(f'unknown plug-in {pr}; choose from {sorted(PLUGINS)}')
        table.update(PLUGINS[pr])
    return table


def lane_verdict(changed, effects, side_axis, *, cascade=None, waive=()):
    """changed: {(leaf, side)} for side-axis leaves, {(leaf, None)} for side-less ones;
    effects: {effect: [actor sides]}; cascade: None, or dict(ko_sides=[2] bool (sides whose every tracker
    may follow a KO), mon_ok=callable(leaf, side) -> a KO_MON_LEAVES leaf differs only on that side's
    (un)fainted mons, a KO_LINK_LEAVES leaf only where link_ok allows, optional families=extra leaf families
    accepted on both sides, e.g. ('residual', 'residual_trackers') on a phase shift).
    Returns (unexpected entries, waived entries, entries accepted only as KO consequences)."""
    unexpected, waived, by_ko = [], [], []
    for leaf, side in sorted(changed, key=lambda x: (x[0], -1 if x[1] is None else x[1])):
        ok = False
        for effect, actors in effects.items():
            rule = ALLOWED.get(effect, ALL)
            if side is None:ok |= allowed_leaf(leaf, rule['glob'])
            else:ok |= any(allowed_leaf(leaf, rule['user' if side == a else 'target']) for a in actors)
            if ok:break
        if not ok and cascade is not None:
            # An effect's KO-only allowance on its actor's side, when the other side has a newly (un)fainted mon.
            ko = side is not None and any(side == a and bool(cascade['ko_sides'][1 - a])
                and allowed_leaf(leaf, KO_ALLOWED.get(effect, {}).get('user', ())) for effect, actors in effects.items() for a in actors)
            ko = ko or (allowed_leaf(leaf, ('hp', 'ko', 'turnline') + tuple(cascade.get('families', ())))
                  or (side is not None and leaf in KO_MON_LEAVES + KO_LINK_LEAVES and cascade['mon_ok'](leaf, side))
                  or (side is not None and (leaf.startswith('trackers.') or allowed_leaf(leaf, ('ko_action',)))
                      and bool(cascade['ko_sides'][side])))
            if ko:by_ko.append((leaf, side));continue
        if ok:continue
        (waived if any(re.search(w, leaf) for w in waive) else unexpected).append((leaf, side))
    return unexpected, waived, by_ko


def active_mask(active):
    """[2] active party indices -> [2,6] bool."""
    active = np.asarray(active)
    return np.arange(6)[None, :] == active[:, None]


def link_ok(leaf, side, a, b, ko_mons, acting, *, departed):
    """[6] mons of `side` whose KO_LINK_LEAVES entry a KO cascade may change (a, b: patched and base values).

    last_move: a fainted mon's is cleared at step end, and a cancelled action is an active's (the actor
    at its turn: the pre-step or post-step active). trap_source (A2 review L2): only a mon that fainted or
    departed ([2,6]: the pre-step active whose post-step active differs in either tree) releases a link:
    its own, or one pointing at it from the other side (linking.prune / entry.depart release both).
    """
    own = ko_mons[side] | acting[side]
    if leaf == 'state.last_move':return own
    gone = ko_mons | departed
    points = lambda v: (np.asarray(v) >= 0) & gone[1 - side][np.clip(np.asarray(v), 0, 5)]
    return gone[side] | points(a) | points(b)


def departed_mask(pre_active, post_active, base_post_active=None):
    """[2,6] the pre-step actives that left the field in this tree or the base tree (A2 review L2)."""
    pre = np.asarray(pre_active);left = pre != np.asarray(post_active)
    if base_post_active is not None:left = left | (pre != np.asarray(base_post_active))
    return active_mask(pre) & left[:, None]


# ---------------------------------------------------------------------------
# Special rules.

def replay_public_order(pre_order, calls):
    """Numpy reference of PR0's swap rule over the recorded install calls.

    pre_order [L,2,6]; calls: (side [L], slot [L], use [L]) in call order, with
    use already excluding initial installs. Each used row moves `slot` to 0.
    """
    order = np.array(pre_order, copy=True)
    rows = np.arange(order.shape[0])
    for side, slot, use in calls:
        side, slot, use = np.asarray(side), np.asarray(slot), np.asarray(use, bool)
        current = order[rows, side]                                           # [L,6]
        position = np.argmax(current == slot[:, None], axis=1)
        found = (current == slot[:, None]).any(axis=1)
        swapped = current.copy()
        swapped[rows, position] = current[:, 0];swapped[:, 0] = slot
        order[rows, side] = np.where((use & found)[:, None], swapped, current)
    return order


def faint_clear_ok(lane_changes, post, base, fainted, physical_to_public, active_slot):
    """True when every changed faint-clear leaf differs only on allowed, cleared entries:
    the fainted mon's public slot, and for meanlook_trap the two actives' slots.

    post/base: {name: array for this lane}; fainted [2,6] physical; physical_to_public [2,6];
    active_slot [2] public slots of the actives.
    """
    for name in lane_changes:
        if name not in FAINT_CLEAR_LEAVES:return False
        a, b = np.asarray(post[name]), np.asarray(base[name])
        if a.shape != b.shape:return False
        differ = a != b
        allowed = np.zeros_like(differ)
        if name.endswith(tuple(f"['{k}']" for k in FAINT_CLEAR_SEATS)):
            for side in range(2):
                slot = int(active_slot[side])
                if 0 <= slot < 6:allowed[side, slot] = True
        else:
            for side, mon in zip(*np.nonzero(fainted)):
                slot = int(physical_to_public[side, mon])
                if 0 <= slot < 6:allowed[side, slot] = True
        if (differ & ~allowed).any() or (a[differ] != 0).any():return False
    return True


def faint_rule(entries, post, base, fainted, physical_to_public, active_slot):
    """Split a faint lane-step's changes: the faint-clear leaves the rule explains (every changed
    one, or none) leave `entries`; the rest still need an involved effect.

    entries: {(leaf, side)}; post/base: {faint leaf: this lane's array} for the changed faint leaves.
    Returns (remaining entries, explained leaves).
    """
    leaves = sorted(post)
    if leaves and faint_clear_ok(leaves, post, base, fainted, physical_to_public, active_slot):
        return {(leaf, side) for leaf, side in entries if leaf not in leaves}, leaves
    return entries, []


# ---------------------------------------------------------------------------
# Engine plumbing.

def context_and_template(args):
    import jax
    import jax.numpy as j
    from gpu_engine.v2_bench_collector import journal, prepare, actor, construct
    events = journal(args.reference_run);context = prepare(args)
    snapshot, _, _ = actor(args, context['plan'], events)
    _, roster, _ = construct(context, snapshot, events, games=1)
    first = lambda tree: jax.tree.map(lambda x: x[:1], tree)
    template = (first(roster.device.state), j.zeros((1, 2), j.int32), first(roster.device.public))
    return context, template


class Stepper:
    """The jitted step plus per-lane, per-side digests, faint data and (optionally) the install log.

    base_names: output names of the base pass; outputs outside it are new, and
    their own pre values are digested too (new leaves must keep their defaults).
    """
    def __init__(self, step_fn, treedef, *, base_names=None, faint_leaves=FAINT_CLEAR_LEAVES, record_installs=False,
                 effect_rule=False):
        import jax
        import jax.numpy as j
        names = {}
        def traced(leaves):
            state, actions, trackers = jax.tree.unflatten(treedef, leaves)
            lanes = state.hp.shape[0];calls = [];ran = []
            # A2 review L1: whether the residual ran, read at its one weather clock call (eligible is
            # transition's residual_ready there: only Wish heals between the two). None if never called.
            from gpu_engine import weather as climate
            clock = climate.residual_clock
            def clock_recorder(weather, counter, *, eligible=True, **extra):
                ran.append(j.broadcast_to(j.asarray(eligible, j.bool_), (lanes,)))
                return clock(weather, counter, eligible=eligible, **extra)
            climate.residual_clock = clock_recorder
            try:
                if record_installs:
                    from gpu_engine import engine_device_public_hooks as hooks
                    original = hooks.install
                    def recorder(c, st, side, use, *, initial=False):
                        out = original(c, st, side, use, initial=initial)
                        s = j.broadcast_to(j.asarray(side, j.int32), (lanes,))
                        slot = out.trackers.active_slot[j.arange(lanes), s]
                        calls.append((s, slot, j.broadcast_to(j.asarray(use, j.bool_), (lanes,)) & (not initial)))
                        return out
                    hooks.install = recorder
                    try:out = step_fn(state, actions, trackers)
                    finally:hooks.install = original
                else:
                    out = step_fn(state, actions, trackers)
            finally:
                climate.residual_clock = clock
            named_in = name_inputs(state, actions, trackers);named_out = name_outputs(out)
            names['out'], digests = digest_named(named_out, lanes)
            names['side_axis'] = [has_side_axis(n, named_out[n].shape, lanes) for n in names['out']]
            new = [n for n in sorted(named_out) if base_names is not None and n not in base_names and n in named_in]
            names['pre'] = new
            pre = digest_named({n: named_in[n] for n in new}, lanes)[1] if new else j.zeros((lanes, 0, 2, 2), j.uint32)
            effect_ok = None
            if effect_rule:
                # The design's derivation, copied: fold_in(split(keys,2)[1], EFFECT_SALT); a resumed turn reuses its
                # saved block; the block is saved exactly when the step defers its residual; idle lanes keep theirs.
                split = jax.vmap(lambda k: jax.random.split(k, 2))(state.keys)[:, 1]
                fresh = jax.vmap(lambda k: jax.random.randint(jax.random.fold_in(k, EFFECT_SALT), (2, 4), 0,
                    j.array(EFFECT_BOUNDS, j.int32)))(split)
                live = ~state.done & ~out.step.invalid_actions
                used = j.where((live & state.pending_residual)[:, None, None], state.effect_draws, fresh)
                saved = j.where(out.step.state.pending_residual[:, None, None], used, 0)
                reference = j.where(live[:, None, None], saved, state.effect_draws)
                effect_ok = (out.step.state.effect_draws == reference).all(-1)              # [L,2]
            return dict(digests=digests, pre_digests=pre, fainted=(state.hp > 0) & (out.step.state.hp <= 0),
                extra={n: named_out[n] for n in faint_leaves if n in named_out}, calls=calls,
                ko={n: named_out[n] for n in KO_MON_LEAVES + KO_LINK_LEAVES if n in named_out},
                post_active=out.step.state.active, post_done=out.step.state.done,
                physical_to_public=out.trackers.physical_to_public, active_slot=out.trackers.active_slot,
                public_order=named_out.get(PUBLIC_ORDER), residual_ran=ran[0] if len(ran) == 1 else None,
                effect_ok=effect_ok)
        self._jit = jax.jit(traced);self._names = names

    def __call__(self, leaves):
        import jax
        result = jax.device_get(self._jit(leaves))
        result.update(names=self._names['out'], pre_names=self._names['pre'], side_axis=self._names['side_axis'])
        return result


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)


def git_source():
    rev = git('rev-parse', 'HEAD').stdout.strip()
    dirty = git('status', '--porcelain', '--', 'gpu_engine', 'src').stdout
    return rev, bool(dirty.strip())


def load_shard(path):
    with np.load(path) as z:
        keys = sorted((k for k in z.files if k[0] == 'l' and k[1:].isdigit()), key=lambda k: int(k[1:]))
        return [z[k] for k in keys], int(z['tick']), int(z['level'])


def prefetched(paths, workers=8):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(load_shard, p) for p in paths[:workers]]
        for i, path in enumerate(paths):
            loaded = pending[i].result();pending[i] = None
            if i + workers < len(paths):pending.append(pool.submit(load_shard, paths[i + workers]))
            yield path, loaded


def shard_list(args, capture):
    shards = sorted(Path(args.shards).glob('shard-*.npz'))
    if args.limit:shards = shards[:args.limit]
    if not shards:raise SystemExit(f'no capture shards under {args.shards}')
    if not args.limit and capture.get('shards') is not None and len(shards) != capture['shards']:
        raise SystemExit(f"{len(shards)} shards under {args.shards}, capture.json records {capture['shards']}")
    return shards


def capture_record(args):
    path = Path(args.shards).parent / 'capture.json';blob = path.read_bytes()
    return json.loads(blob), hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Passes.

def base_pass(args, *, step_override=None, prepared=None):
    """step_override/prepared ((context, template)) let tests drive the pass without a V2 runtime.

    Inputs are migrated by leaf path exactly as in the patched pass, so a tree with leaves the
    capture predates (e.g. trackers.public_order) still runs on it; the filled leaves are
    recorded in the manifest (migrated_new_inputs)."""
    import jax
    from gpu_engine.v2e_diff import oracle_digest
    rev, dirty = git_source()
    if dirty and not args.allow_dirty:raise SystemExit('the source tree is dirty; commit it or pass --allow-dirty')
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    capture, capture_sha = capture_record(args)
    context, template = prepared or context_and_template(args)
    treedef = jax.tree.structure(template)
    template_named = {n: np.asarray(leaf) for n, leaf in zip(_template_names(template), jax.tree.leaves(template))}
    capture_names = [capture_name(p) for p in capture['paths']];names_order = _template_names(template)
    defaults = state_defaults(template_named);filled = None
    tables = context['prepared']
    stepper = Stepper(step_override or _engine_step(tables), treedef)
    shards = shard_list(args, capture);pad = args.pad_lanes or 0
    names = side_axis = None;lane_steps = 0;live = 0;begin = time.perf_counter();residual_recorded = False
    for index, (path, (leaves, tick, level)) in enumerate(prefetched(shards)):
        lanes = int(leaves[0].shape[0])
        named, filled = migrate(capture_names, leaves, template_named, defaults=defaults)
        named, real = pad_lanes(named, pad)
        result = stepper([named[n] for n in names_order])
        names, side_axis = result['names'], result['side_axis']
        done = np.asarray(named['state.done'])[:real]
        fainted = result['fainted'][:real]
        faint_lanes = np.flatnonzero(fainted.any(axis=(1, 2)) & ~done)
        record = dict(digests=result['digests'][:real], done=done, tick=np.int64(tick), level=np.int64(level),
            fainted=fainted, faint_lanes=faint_lanes)
        for i, n in enumerate(sorted(result['extra'])):record[f'faint_{i:02d}'] = result['extra'][n][faint_lanes]
        for i, n in enumerate(sorted(result['ko'])):record[f'ko_{i:02d}'] = result['ko'][n][:real]
        # A2 review L1/L2: whether the residual ran, and the post-step actives (a departure in this tree).
        record['post_active'] = np.asarray(result['post_active'])[:real]
        if result['residual_ran'] is not None:record['residual_ran'] = np.asarray(result['residual_ran'])[:real]
        residual_recorded = result['residual_ran'] is not None
        np.savez_compressed(out / f'digest-{path.stem}.npz', **record)
        lane_steps += real;live += int((~done).sum())
        if index % 50 == 0:log('BASE_SHARD', index=index, shard=path.name, lanes=lanes, padded=pad or lanes, seconds=time.perf_counter() - begin)
    digest, modules = oracle_digest()
    manifest = dict(schema=SCHEMA, digest_format=DIGEST_FORMAT, pass_='base', source_commit=rev, source_dirty=dirty,
        oracle_digest=digest, oracle_modules=len(modules), capture_sha256=capture_sha, capture_commit=capture.get('source_commit'),
        shards=[p.name for p in shards], partial=bool(args.limit), lane_steps=lane_steps, live_lane_steps=live,
        output_names=names, side_axis=side_axis, input_names=names_order, migrated_new_inputs=sorted(filled or ()),
        faint_leaves=sorted(n for n in FAINT_CLEAR_LEAVES if n in names),
        ko_leaves=sorted(n for n in KO_MON_LEAVES + KO_LINK_LEAVES if n in names),
        residual_ran_recorded=residual_recorded, post_active_recorded=True,
        pad_lanes=pad, digest_seeds=list(DIGEST_SEEDS),
        jax=jax.__version__, backend=jax.default_backend(), seconds=time.perf_counter() - begin)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    log('BASE_DONE', shards=len(shards), lane_steps=lane_steps, live_lane_steps=live, seconds=manifest['seconds'])
    return 0


def classify(changed, involved, rule_ok, unexpected):
    if not changed:return 'unchanged'
    if rule_ok:return 'rule'
    if not involved:return 'uninvolved'
    return 'unexpected' if unexpected else 'involved'


def check_base(args, manifest, rev):
    """Digest scheme, JAX and the PR's merge base must match the stored base pass."""
    import jax
    problems = []
    if manifest.get('schema') != SCHEMA or manifest.get('digest_format') != DIGEST_FORMAT:
        problems.append(f"base schema {manifest.get('schema')}/{manifest.get('digest_format')} != {SCHEMA}/{DIGEST_FORMAT}")
    if manifest.get('digest_seeds') != list(DIGEST_SEEDS):problems.append('digest seeds differ from the base pass')
    if jax.__version__ != manifest['jax'] or jax.default_backend() != manifest['backend']:
        problems.append(f"base pass ran on jax {manifest['jax']}/{manifest['backend']}, this is {jax.__version__}/{jax.default_backend()}")
    if problems:raise SystemExit('; '.join(problems))
    merge_base = None
    if args.integration:
        merge_base = git('merge-base', 'HEAD', args.integration).stdout.strip() or None
    matches = merge_base is not None and merge_base == manifest['source_commit']
    if not matches and not args.allow_base_mismatch:
        raise SystemExit(f"base pass source {manifest['source_commit'][:12]} is not this PR's merge base "
                         f"{(merge_base or 'unknown (give --integration REF)')[:12]}; pass --allow-base-mismatch to override")
    return dict(merge_base=merge_base, base_source_commit=manifest['source_commit'], base_matches_merge_base=matches,
        allow_base_mismatch=bool(args.allow_base_mismatch))


def patched_pass(args, *, step_override=None, prepared=None):
    """step_override: a deliberately perturbed step for self-tests; prepared as in base_pass."""
    import jax
    from gpu_engine import state as state_module
    from gpu_engine.v2e_diff import oracle_digest
    base_dir = Path(args.base);manifest = json.loads((base_dir / 'manifest.json').read_text())
    rev, dirty = git_source()
    if dirty and not args.allow_dirty:raise SystemExit('the source tree is dirty; commit it or pass --allow-dirty')
    base_check = check_base(args, manifest, rev)
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    capture, capture_sha = capture_record(args)
    if capture_sha != manifest['capture_sha256']:raise ValueError('capture differs from the base pass')
    plugins = [p for p in args.pr.split(',') if p]
    warnings = []
    if len([p for p in plugins if p != 'none']) > 1:
        warnings.append(f'plug-in superset {plugins}: changes are attributed to every listed PR\'s effects')
        log('WARNING', message_=warnings[-1])
    context, template = prepared or context_and_template(args)
    treedef = jax.tree.structure(template)
    template_named = {}
    for name, leaf in zip(_template_names(template), jax.tree.leaves(template)):template_named[name] = np.asarray(leaf)
    capture_names = [capture_name(p) for p in capture['paths']]
    new_inputs = sorted(set(template_named) - set(capture_names))
    table = predicates(plugins)
    # The PR's own plug-in (the first listed) must declare allowed fields for every effect;
    # a superset's other plug-ins only warn.
    own = PLUGINS[plugins[0]] if plugins else {}
    undeclared_own = sorted(e for e in own if e not in ALLOWED)
    if undeclared_own:
        raise SystemExit(f'plug-in {plugins[0]} has effects without declared allowed fields: {undeclared_own}; add them to ALLOWED')
    undeclared = sorted(e for e in table if e not in ALLOWED)
    if undeclared:warnings.append(f'effects of other plug-ins without declared allowed fields (any change accepted): {undeclared}')
    tables = context['prepared'];metadata = context['pool'].metadata();vocab = metadata['vocabulary']
    base_names = manifest['output_names']
    # The replay rule is for the PR that adds public_order (its base pass lacks the leaf); once
    # the base pass has it too (a migrated identity pre), the leaf is compared bitwise.
    public_order_rule = PUBLIC_ORDER in new_inputs and PUBLIC_ORDER not in base_names
    # Fidelity R: likewise the saved effect block is checked against the design's derivation while the base lacks it.
    effect_rule = EFFECT_DRAWS in new_inputs and EFFECT_DRAWS not in base_names
    stepper = Stepper(step_override or _engine_step(tables), treedef, base_names=set(base_names), record_installs=public_order_rule,
        effect_rule=effect_rule)
    shards = [p for p in shard_list(args, capture) if p.name in set(manifest['shards'])]
    if args.limit == 0 and len(shards) != len(manifest['shards']):raise ValueError('shard set differs from the base pass')
    usable = jax.jit(state_module.usable_moves)
    callable_moves = np.asarray(tables.engine_tables.sleep_talk.callable)
    destiny_bond = getattr(state_module, 'DESTINY_BOND', DEFAULT_DESTINY_BOND)
    counts = dict(lane_steps=0, done_skipped=0, unchanged=0, changed_and_involved=0, changed_but_uninvolved=0,
                  changed_involved_unexpected_fields=0, ko_cascade=0, phase_shift=0, terminal=0, terminal_residual=0, waived=0,
                  rule_explained=0, faint_rule_leaves=0, effect_draws_saved=0)
    by_effect = {k: dict(changed=0, involved_unchanged=0, changed_fields={}, unexpected_fields={}, ko_cascade=0,
                         phase_shift=0, terminal=0, terminal_residual=0, ko_cascade_non_hp_fields={}) for k in table}
    ko_leaves = manifest.get('ko_leaves')
    if ko_leaves is None:raise SystemExit('base pass predates the KO-cascade arrays; re-run the base pass')
    # last_move/trap_source are checked per mon against the base arrays; a base pass without them gets the
    # stricter whole-side fallback (only a side with a newly (un)fainted mon), recorded as a warning.
    link_rule = 'per-mon' if all(n in ko_leaves for n in KO_LINK_LEAVES) else 'ko-side'
    if link_rule != 'per-mon':
        warnings.append('base pass predates the per-mon last_move/trap_source arrays: whole-KO-side fallback rule')
    # A2 review L1: a terminal step widens to the residual families only where the residual ran in exactly one tree.
    residual_rule = 'per-lane' if manifest.get('residual_ran_recorded') else 'all-terminal'
    if residual_rule != 'per-lane':
        warnings.append('base pass predates the residual-ran record: every terminal step takes the residual families')
    if not manifest.get('post_active_recorded'):
        warnings.append('base pass predates the post-step actives: departures are read from this tree only')
    uninvolved_leaves = {};new_outputs = None
    listing = (out / 'changed.jsonl').open('x');begin = time.perf_counter()
    names_order = _template_names(template)
    defaults = state_defaults(template_named)
    try:
        for index, (path, (leaves, tick, level)) in enumerate(prefetched(shards)):
            named, filled = migrate(capture_names, leaves, template_named, defaults=defaults)
            named, real = pad_lanes(named, manifest['pad_lanes'])
            result = stepper([named[n] for n in names_order])
            names = result['names'];position = {n: i for i, n in enumerate(names)};side_axis = dict(zip(names, result['side_axis']))
            missing = [n for n in base_names if n not in position]
            if missing:raise ValueError(f'output leaves removed by this tree: {missing[:8]}')
            new_outputs = sorted(set(names) - set(base_names))
            digests = result['digests'][:real]
            done_now = np.asarray(named['state.done'])
            base = np.load(base_dir / f'digest-{path.stem}.npz')
            diff = np.zeros((real, len(names), 2), bool)
            shared = [position[n] for n in base_names]
            diff[:, shared] = (digests[:, shared] != base['digests']).any(-1)
            # New leaves must keep their default-filled pre values unless involved.
            for k, n in enumerate(result['pre_names']):
                if n == PUBLIC_ORDER and public_order_rule:continue
                diff[:, position[n]] = (digests[:, position[n]] != result['pre_digests'][:real, k]).any(-1)
                if n == EFFECT_DRAWS and effect_rule:
                    # A saved block equal to the derivation explains the change; anything else stays a change.
                    counts['effect_draws_saved'] += int((diff[:, position[n]].any(-1) & ~done_now[:real]).sum())
                    diff[:, position[n]] = ~np.asarray(result['effect_ok'])[:real]
            for n in new_outputs:
                if n not in result['pre_names']:diff[:, position[n]] = True          # a new output with no pre value
            if public_order_rule:
                calls = [(np.asarray(a)[:real], np.asarray(b)[:real], np.asarray(c)[:real]) for a, b, c in result['calls']]
                replay = replay_public_order(np.asarray(named[PUBLIC_ORDER])[:real], calls)
                bad = replay != result['public_order'][:real]                         # [L,2,6]
                diff[:, position[PUBLIC_ORDER]] = bad.any(-1)
            struggle = ~np.asarray(usable(_state_only(named, template, real))).any(-1)
            pre = Pre({n: np.asarray(named[n])[:real] for n in named if n.startswith('state.') or n == 'actions' or n in PRE_TRACKERS},
                vocab, callable_moves, struggle, destiny_bond=destiny_bond)
            involved = {k: np.asarray(f(pre), bool) for k, f in table.items()}       # [L,2] actor sides
            done = np.asarray(named['state.done'])[:real]
            ko_mons_all = result['fainted'][:real] != base['fainted']                       # [L,2,6] (un)fainted mons
            cascade_lanes = ko_mons_all.reshape(real, -1).any(-1)
            # A phase shift: the residual was deferred in exactly one tree (see FAMILIES['residual']).
            phase_lanes = diff[:, position['state.pending_residual']].any(-1) if 'state.pending_residual' in position \
                else np.zeros(real, bool)
            # A terminal step: the battle ended in either tree (post done here, or done differing from the base).
            done_diff = diff[:, position['state.done']].any(-1) if 'state.done' in position else np.zeros(real, bool)
            terminal_lanes = np.asarray(result['post_done'])[:real].astype(bool) | done_diff
            # A2 review L1: the residual ran in exactly one tree (unknown without both records: every terminal step).
            if residual_rule == 'per-lane' and 'residual_ran' in base.files and result['residual_ran'] is not None:
                residual_lanes = np.asarray(result['residual_ran'])[:real] != base['residual_ran']
            else:
                residual_lanes = np.ones(real, bool)
            base_post_active = base['post_active'] if 'post_active' in base.files else None
            base_ko = {n: base[f'ko_{i:02d}'] for i, n in enumerate(ko_leaves)}
            pre_active = np.asarray(named['state.active'])[:real];post_active = np.asarray(result['post_active'])[:real]
            faint_base = {};faint_set = set()
            if args.faint_clear:
                lanes_list = base['faint_lanes'].tolist()
                for i, n in enumerate(manifest['faint_leaves']):faint_base[n] = dict(zip(lanes_list, base[f'faint_{i:02d}']))
                faint_set = set(lanes_list)
            for lane in range(real):
                counts['lane_steps'] += 1
                if done[lane]:counts['done_skipped'] += 1;continue
                entries = set()
                for i in np.flatnonzero(diff[lane].any(-1)):
                    leaf = names[i]
                    if side_axis[leaf]:entries |= {(leaf, s) for s in (0, 1) if diff[lane, i, s]}
                    else:entries.add((leaf, None))
                changed = sorted({leaf for leaf, _ in entries})
                effects = {k: [s for s in (0, 1) if v[lane, s]] for k, v in involved.items() if v[lane].any()}
                # The faint-clear rule explains its own leaves; any other changed leaf on the
                # same lane-step still needs an involved effect (e.g. a revert on a faint lane).
                faint_explained = []
                if changed and args.faint_clear and lane in faint_set:
                    leaves = [n for n in changed if n in faint_base]
                    entries, faint_explained = faint_rule(entries, {n: result['extra'][n][lane] for n in leaves},
                        {n: faint_base[n][lane] for n in leaves}, result['fainted'][lane],
                        result['physical_to_public'][lane], result['active_slot'][lane])
                    counts['faint_rule_leaves'] += bool(faint_explained)
                rule_ok = bool(changed) and not entries
                changed_rest = sorted({leaf for leaf, _ in entries})
                phase = bool(phase_lanes[lane]);terminal = bool(terminal_lanes[lane]) and not phase
                terminal_residual = terminal and bool(residual_lanes[lane])
                cascade = bool(cascade_lanes[lane]) or phase or terminal;cascade_info = None
                if cascade:
                    ko_mons = ko_mons_all[lane]
                    acting = active_mask(pre_active[lane]) | active_mask(post_active[lane])        # [2,6]
                    departed = departed_mask(pre_active[lane], post_active[lane],
                        None if base_post_active is None else base_post_active[lane])
                    def mon_ok(leaf, side, lane=lane, ko_mons=ko_mons, acting=acting, departed=departed):
                        if leaf not in base_ko:
                            return leaf in KO_LINK_LEAVES and bool(ko_mons[side].any())         # fallback, warned
                        a = np.asarray(result['ko'][leaf][lane][side]);b = np.asarray(base_ko[leaf][lane][side])
                        differ = (a != b).reshape(6, -1).any(-1)
                        allowed = (link_ok(leaf, side, a, b, ko_mons, acting, departed=departed) if leaf in KO_LINK_LEAVES
                            else ko_mons[side])
                        return not (differ & ~allowed).any()
                    # A phase shift, or a terminal step whose residual ran in one tree only, accepts the residual
                    # handlers' writes; any other terminal step only the done-gated end-of-turn writes (L1).
                    families = (('residual', 'residual_trackers') if phase or terminal_residual
                        else ('terminal',) if terminal else ())
                    cascade_info = dict(ko_sides=ko_mons.any(-1), mon_ok=mon_ok, families=families)
                unexpected, waived, by_ko = lane_verdict(entries, effects, side_axis, cascade=cascade_info, waive=args.waive or ()) \
                    if changed_rest and effects else ([], [], [])
                kind = classify(changed, bool(effects), rule_ok, unexpected)
                if kind == 'unchanged':
                    counts['unchanged'] += 1
                    for e in effects:by_effect[e]['involved_unchanged'] += 1
                    continue
                labels = [f"{leaf}@{'glob' if side is None else side}" for leaf, side in sorted(entries, key=str)]
                if kind == 'rule':counts['rule_explained'] += 1
                elif kind in ('involved', 'unexpected'):
                    counts['changed_and_involved' if kind == 'involved' else 'changed_involved_unexpected_fields'] += 1
                    counts['ko_cascade'] += cascade;counts['phase_shift'] += phase;counts['terminal'] += terminal
                    counts['terminal_residual'] += terminal_residual
                    counts['waived'] += bool(waived)
                    for e, actors in effects.items():
                        stats = by_effect[e];stats['changed'] += 1;stats['ko_cascade'] += cascade;stats['phase_shift'] += phase
                        stats['terminal'] += terminal;stats['terminal_residual'] += terminal_residual
                        for leaf, side in by_ko:
                            if allowed_leaf(leaf, ('hp',)):continue
                            role = 'glob' if side is None else ('user' if side in actors else 'target')
                            key = f'{leaf}@{role}';stats['ko_cascade_non_hp_fields'][key] = stats['ko_cascade_non_hp_fields'].get(key, 0) + 1
                        for leaf, side in entries:
                            role = 'glob' if side is None else ('user' if side in actors else 'target')
                            key = f'{leaf}@{role}';stats['changed_fields'][key] = stats['changed_fields'].get(key, 0) + 1
                        for leaf, side in unexpected:
                            role = 'glob' if side is None else ('user' if side in actors else 'target')
                            key = f'{leaf}@{role}';stats['unexpected_fields'][key] = stats['unexpected_fields'].get(key, 0) + 1
                else:
                    counts['changed_but_uninvolved'] += 1
                    for n in changed_rest:uninvolved_leaves[n] = uninvolved_leaves.get(n, 0) + 1
                listing.write(json.dumps(dict(shard=path.name, lane=lane, tick=tick, kind=kind, ko_cascade=cascade, phase_shift=phase,
                    terminal=terminal, terminal_residual=terminal_residual,
                    involved={k: v for k, v in effects.items()}, leaves=changed, entries=labels, faint_rule=faint_explained,
                    unexpected=[f"{l}@{'glob' if s is None else s}" for l, s in unexpected],
                    ko_consequences=[f"{l}@{'glob' if s is None else s}" for l, s in by_ko],
                    waived=[f"{l}@{'glob' if s is None else s}" for l, s in waived])) + '\n')
            if index % 50 == 0:
                log('PATCHED_SHARD', index=index, shard=path.name, seconds=time.perf_counter() - begin, **counts)
    finally:
        listing.close()
    digest, modules = oracle_digest()
    failures = counts['changed_but_uninvolved'] + counts['changed_involved_unexpected_fields']
    partial = bool(args.limit) or manifest.get('partial', False)
    # A partial run never passes: passed is None without failures (exit 2), False with them (exit 1).
    report = dict(schema=SCHEMA, digest_format=DIGEST_FORMAT, pass_='patched', plug_ins=plugins, source_commit=rev,
        source_dirty=dirty, allow_dirty=bool(args.allow_dirty), oracle_digest=digest, base_oracle_digest=manifest['oracle_digest'],
        capture_sha256=capture_sha, **base_check, shards=len(shards), partial=partial, failures=failures,
        migrated_new_inputs=new_inputs, new_outputs=new_outputs, public_order_rule=public_order_rule, link_rule=link_rule,
        effect_rule=effect_rule, residual_rule=residual_rule,
        faint_clear_rule=bool(args.faint_clear), waive=list(args.waive or ()), counts=counts, by_effect=by_effect,
        allowed_fields={e: ALLOWED.get(e, 'UNDECLARED: any change') for e in table},
        ko_allowed_fields={e: KO_ALLOWED[e] for e in table if e in KO_ALLOWED},
        uninvolved_leaves=dict(sorted(uninvolved_leaves.items(), key=lambda x: -x[1])), warnings=warnings,
        passed=False if failures else (None if partial else True), seconds=time.perf_counter() - begin)
    (out / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    log('PATCHED_DONE', passed=report['passed'], partial=report['partial'], **counts)
    return 1 if failures else (2 if partial else 0)


def state_defaults(template_named):
    """New BattleState leaves default to empty_state's values (counters 0, sources -1, ...),
    shape- and dtype-checked against the tree's leaf."""
    from gpu_engine.state import empty_state
    blank = empty_state(1);defaults = {}
    for name in blank._fields:
        key = f'state.{name}';value = np.asarray(getattr(blank, name))
        if key in template_named:
            template = template_named[key]
            if value.shape[1:] != template.shape[1:] or value.dtype != template.dtype:
                raise MigrationError(f'{key}: empty_state {value.shape[1:]}{value.dtype} vs tree {template.shape[1:]}{template.dtype}')
        defaults[key] = (lambda v: lambda lanes: np.broadcast_to(v, (lanes,) + v.shape[1:]))(value)
    return defaults


def _engine_step(tables):
    from gpu_engine import v2e_modules
    return v2e_modules.step(tables.engine_tables, tables.observation_tables)


def _template_names(template):
    state, actions, trackers = template
    return list(name_inputs(state, actions, trackers))


def _state_only(named, template, real):
    import jax
    state = template[0]
    names = list(_named('state', state))
    return jax.tree.unflatten(jax.tree.structure(state), [np.asarray(named[n])[:real] for n in names])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('base', 'patched'):
        p = sub.add_parser(name)
        p.add_argument('--config', required=True)
        p.add_argument('--reference-run', required=True)
        p.add_argument('--shards', required=True, help='directory of capture shards (capture.json beside it)')
        p.add_argument('--out', required=True, help='new output directory')
        p.add_argument('--arm', default='B', choices=('A', 'B'))
        p.add_argument('--limit', type=int, default=0, help='first N shards (0: all); a limited run is partial and never passes (exit 2)')
        p.add_argument('--allow-dirty', action='store_true', help='run from a tree with uncommitted gpu_engine/src changes')
        if name == 'base':
            p.add_argument('--pad-lanes', type=int, default=2048, help='pad every shard to this many lanes (one compile); 0 keeps native counts')
        else:
            p.add_argument('--base', required=True, help='output directory of the base pass')
            p.add_argument('--pr', default='none', help=f'comma list of involvement plug-ins, the PR\'s own first: {",".join(PLUGINS)}')
            p.add_argument('--faint-clear', action='store_true', help="apply PR0's faint-time public clear rule")
            p.add_argument('--integration', default=None, help='integration ref; the base pass must be at the merge base with it')
            p.add_argument('--allow-base-mismatch', action='store_true', help='accept a base pass not at the merge base (recorded)')
            p.add_argument('--waive', action='append', default=None, help='leaf regex whose unexpected changes are accepted (recorded)')
    args = parser.parse_args(argv)
    return base_pass(args) if args.command == 'base' else patched_pass(args)


if __name__ == '__main__':
    sys.exit(main())
