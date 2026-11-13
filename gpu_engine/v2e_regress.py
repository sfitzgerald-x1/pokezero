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

Special rule (B, with --pr B): a faint's clearVolatile now zeroes the fainted
mon's Choice lock and confusion at the step end (state.choice_move,
state.confusion_counter; CLEAR_VOLATILE_LEAVES). A (leaf, side) change is
explained only when every differing mon on that side fainted this step (HP
>0 -> 0 in this tree) and is 0 in this tree, read from the per-mon KO arrays
both passes store; any other change of those leaves still needs an involved
effect. Explained lane-steps are counted as clear_volatile_leaves.

Special rules (M1, with --pr M1; post-state rules that check this tree, never
a pre-state predicate alone). The base pass stores per-lane arrays of
WITNESS_LEAVES (wit_XX) so these rules compare per mon and per bag column:
- recharge_rule: Hyper Beam's mustrecharge. A (state.recharge_counter, side)
  change is explained only on a side whose executed move may be Hyper Beam
  (involvement) AND whose move connected in this tree (recorded around the
  device hooks: HP damage its damage hook records, or the foe's Substitute
  lower at that hook than when this side's move line opened; or the foe's
  Substitute falling over the step while it stayed in) AND whose post
  counter is nonzero exactly at its living post-step active, with the value
  the residual leaves (1 when it ran, 2 when it was deferred), AND whose
  mustrecharge bag bit is set at that active's public slot.
- bag_rule: a (trackers.public_volatiles, side) change is explained only when
  every (slot, column) differing from the base's stored bag is a witnessed
  cell: mustrecharge at that recharging active's slot (or a mon that departed
  after the hit), and on a Trick lane Substitute at a mon whose Substitute
  stood before the step and stands in exactly one tree, or confusion at a mon
  whose Lum Berry cure differs (below), each bit equal to this tree's state.
- trick_rule (Trick lanes; the pool is the pre-step items of either side's
  pre- or post-step active, so a mon switched in before the Trick counts): a
  Lum Berry in the pool that was eaten here, or a holder's own Lum that left
  it, witnesses per-mon differences of status (state, public status, the
  sleep features, the clause port) and confusion (counter, feature); a pinch
  berry in the pool witnesses a one-stage difference in its own stat at an
  active (eaten here: post item 0; eaten only in the base: the holder's own
  berry), in state and public boosts; a witnessed Substitute the device's
  sub_present.
- ledger_rule: B2's item ledger on Knock Off. A change of the per-mon
  mon_features item_mutated / item_original on a side is explained only when
  they are nonzero exactly at the mons whose state.item_knocked flipped this
  step in this tree, with item_original the item each held before the step.
  Trick lanes take the ledger through the trick allowance instead.
Explained lane-steps are counted as recharge_leaves, bag_leaves,
trick_witness_leaves and ledger_leaves.

Special rule (M2, with --pr M2; post-state, per mon, never a pre-state predicate alone): m2_rule explains the new
leaves Perish Song, Yawn and Wrap set this step (the captures carry none, so any nonzero value differs from the migrated
default) from this tree's own evidence, recorded around the device hooks (the move hook's line with use, and the target
at that line; the damage hook's HP damage after a Wrap line; port_protect's Protect block; turn_end's ordinary flag):
- perish: a Perish Song line with use on some side; each side's counter nonzero only at its living post-step active,
  which is not Soundproof, with 3 after the order-12 tick or 4 without it; the bag's perish bits at that slot are
  perish{min(count,3)}.
- yawn: the other side's Yawn line had use, met no Protect and found this side's active unstatused, without a
  Substitute and alive; the counter is nonzero only at that same living active (not Insomnia/Vital Spirit), 1 after
  the residual or 2 without it; the bag's yawn bit is set there.
- wrap: the other side's Wrap line dealt HP damage (a Substitute hit traps nothing); counter and source nonzero only at
  that same living target, the source the Wrap user's party index, the counter 3+draws[f][7] (the design derivation,
  recomputed from the keys) less one after the tick; partiallytrapped set in the bag; WRAP_TRAP_TURNS 1 at that slot
  exactly when turn_end was ordinary, 0 elsewhere. A ticked counter (less one) needs the trapper held at the holder's
  handler: the Wrap user the other side's living active with activeTurns > 0 there, recorded at the HP-line cutpoint
  transition emits right after that handler (a slower trapper may faint from its own residual after the tick; the step
  end stands in only when no cutpoint was recorded).
- wrap tick (M2 review): on a side the other side's Wrap struck, state.hp is explained only when adding the tick
  max(1, maxhp//16) back at the trapped active reproduces the base's stored per-side digest exactly (mon_digest, a numpy
  copy of lane_digest) after a witnessed tick, or, for a lethal tick that cleared the trap with the trapper held, some
  base HP 1..tick there; state.substitute_hp is never explained. Where the handler cutpoint recorded the holder's HP
  after the tick, a later heal in the same residual (a Leech Seed drain from a slower trapper) is replayed against the
  base with its max-HP cap: base = pre + min(later, max - pre). Both are `must` entries: left unexplained, they are
  unexpected even on a KO cascade lane (counted as m2_forced). No allowance covers them (wrap allows the HP lines only).
- reveals (M2 review): a reveal entry on side x is explained only when x's post-step active holds Soundproof on a
  Perish Song lane, or Insomnia/Vital Spirit after the other side's unblocked Yawn line at it (no allowance covers them).
The bag cells m2_rule witnesses join M1's in bag_rule. Explained lane-steps are counted as m2_leaves.
Known limits: the pre-step allowances (perish: hp/m2/reveal on the holder's side; drowsy: status/item/clause/visible/m2;
partial_trap: everything, as a captured switch the trap refuses freezes the lane) are pre-state predicates; no capture
carries a pre-step M2 counter, so the shards never exercise them (unit tests and probes do), and the wrong-mon check
bounds their per-mon writes to the actives.
Wrong-mon check (M2, M1 re-review's benched-mon lesson): on a lane an M2 effect involves (move or pre-step counter),
a remaining entry of M2_MON_LEAVES (HP, status, status counter, the M2 counters and source) is unexpected when its
per-mon difference lies off that side's pre-step and post-step actives: against the base's per-mon KO arrays where
both passes store the leaf (status, status counter), else this tree's own pre-to-post change (HP and the M2 leaves,
which the base does not store; within one step only a mon active at its start or end changes there). An effect landing
on a benched mon is so flagged even where the leaf's family is allowed. Flagged lane-steps are counted as
m2_wrong_mon.

Special rules (WH, with --pr WH; post-state, per mon). The base pass stores per-lane witness arrays of the holder's
item, item_knocked flag and stages and of the device's item reveal, removal, narrowing, visible basis, ledger, per-stint
item flags and removal credit (witness_names). The Stepper records this tree's White Herb uses around the device hook
white_herb: a valid use is the side's living active holding White Herb with a negative stage, left with no item and exactly
max(stage, 0), no other state leaf of any mon changed by the call, and the hook's consumption reveal (shown_item White
Herb, item_removed) at its slot; any other call with use, or a change without use, is bad (forced unexpected). White
Herb stays involved only on a lane with a valid use. A Tricked-in White Herb whose living receiver has a negative stage
must already be used when the device's trick hook runs (setItem's Start inside Trick's onHit precedes the -item lines),
and the hook's herb_used must equal those uses; otherwise the lane is bad too (wh_trick_order). wh_rule then explains, per mon and per public slot against the
base's arrays, the restored stages (this tree's = the base's + the recorded raise, or a mon whose faint differs), the
consumed item (White Herb in the base, none here), a holder the base knocked off afterwards (item_knocked, credit),
and a Trick partner that got the herb only in the base; item-family slot differences must lie at an active of either
side. Facade stays involved only for a side whose own Facade line ran by its pre-step active under par/brn/psn/tox
(recorded at the move hook). A frozen user's Sacred Fire stays involved only for a side whose pre-step active was
frozen at its record_move and thawed at its own Sacred Fire line, or, at its own BeforeMove (eligible, the executed
move Sacred Fire), failed frz's roll and did not act (flinch or confusion), and never where a Sacred Fire line ran still frozen (sacred_actors; lanes kept by the stop alone: sacredfire_stopped).
A pinch berry eaten in exactly one tree on a White Herb or Facade lane (its holder's damage moved across the
quarter-HP line) is explained per mon by pinch_rule (one stage in its own stat, the item, the device's reveal at its
slot). Explained lane-steps are counted as wh_leaves (pinch_leaves for berries); malformed calls as wh_bad; lanes that
lost the involvement as wh_dropped, facade_dropped and sacredfire_dropped.

KO cascade rule for the saved effect block (fidelity R's state.effect_draws,
found by the M1 GPU pass): once the base pass has the leaf it is compared
bitwise, and a KO cascade or phase shift that moves the residual deferral
saves (or no longer saves) the block, like the other saved draws in the 'ko'
family. A (state.effect_draws, side) change on a cascade lane is accepted only
when this tree's block equals the design 2.1 derivation (the Stepper's
effect_ok: the block a deferred step saves, else zero); any other change stays
unexpected. Likewise (same pass, a Trick-given Choice Band moving a KO from the
residual into the action phase): Color Change retypes its holder only when the
hit leaves it standing (gen 3 inherits data/mods/gen4/abilities.ts:46-47,
onDamagingHit `if (!damage || !target.hp) return`), so on a cascade lane a
(state.types / trackers.shown_types, side) change is accepted only when a Color
Change holder on that side fainted this step in this tree or (un)fainted
relative to the base (RETYPE_LEAVES), and only at that holder: per mon
against the base's stored types and shown types.

The involvement superset resolves the executed move as the engine does (slot,
charge, standing or same-turn Encore, Struggle, a resumed queued action, every
Sleep Talk-callable move, a Magic Coat reflection, and no move on a recharge
turn) and covers Effect Spore and
Yawn sleep under Sleep Clause, Rest or Effect Spore against Insomnia/Vital
Spirit (a Trace holder copying either, any install of a holder), and an incoming
White Herb holder. Known gaps, which can only cause false failures: Snatch,
Mirror Move, Metronome and Assist calls (not in the Gen 3 randbats pool).

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
                   'item_swapped', 'choice_locked', 'choice_public')
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
# M2: the new leaves (the captures predate them) and the bag columns m2_rule reads.
WRAP_CLOCK = "trackers.mon_features['wrap_trap_turns']"
M2_BAG = ('perish0', 'perish1', 'perish2', 'perish3', 'yawn', 'partiallytrapped')
M2_COUNTER_LEAVES = ('state.perish_counter', 'state.yawn_counter', 'state.partial_trap_counter', 'state.partial_trap_source')


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
        # M1: a recharging active's action is the recharge pseudo-move (no move); the other overrides stay a superset.
        if 'recharge_counter' in s:chosen = np.where(self.active('recharge_counter') > 0, 0, chosen)
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
    Pass, the foe's Roar/Whirlwind, a replacement); WH: and a side that may Trick an
    active holder's herb away (it may restore the Tricker at the Trick's own AfterMove)."""
    items = p.vocab['items']
    if 'whiteherb' not in items:return np.zeros((p.lanes, 2), bool)
    herb = items.index('whiteherb');holders = p.s['item'] == herb                       # [L,2,6]
    active = p.active('item') == herb
    return active | (holders.any(-1) & installs(p)) | (p.side_uses('trick') & active[:, ::-1])


# WH (C-lite): status ids, mechanics.SLP and mechanics.FRZ (PAR 1, BRN 2, PSN 3, TOX 4, SLP 5, FRZ 6). Literal here: this
# module imports no JAX at load; test_gpu_engine_fidelity_wh pins them to mechanics.
SLP, FRZ = 5, 6


def facade_status(p):
    """WH: a side that may execute Facade while its active is neither asleep nor frozen: only a par/brn/psn/tox user at
    its getDamage doubles, and an asleep or frozen user reaches Facade only after waking or thawing (no status) or through
    Sleep Talk (slp). A status gained earlier in the step counts (pre-state superset)."""
    return p.side_uses('facade') & ~np.isin(p.active('status'), [SLP, FRZ])


def frozen_defrost(p):
    """WH: a frozen active that may execute Sacred Fire (the only defrost move with a pool user: Ho-Oh, which Gen 3 can
    freeze, the typechart giving frz immunity to Ice alone)."""
    return p.side_uses('sacredfire') & (p.active('status') == FRZ)


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
    # M2 review: the HP lines without the HP itself (state.hp and a Substitute's HP need a witness on Wrap lanes).
    'hp_lines': (r'^trackers\.(hp_percent|damage_exact_(dealt|taken))$', r'^trackers\.(damage_history|negative_items)\.',
                 r"^trackers\.mon_features\['last_damage_(dealt|taken)'\]$"),
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
    # M1: B2's per-stint Trick/Choice flags and per-mon item ledger; a Lum Berry the Trick hands over cures confusion.
    'itemflags': (r"^trackers\.mon_features\['(item_swapped|choice_public|choice_locked|item_mutated|item_original)'\]$",),
    'lum': (r'^state\.confusion_counter$', r"^trackers\.mon_features\['confusion_turns'\]$", r'^trackers\.public_volatiles$'),
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
    r"confusion_turns|encore_turns|wrap_trap_turns|stall_counter|sub_present|meanlook_trap|item_swapped|choice_locked|choice_public|"
    r"last_damage_dealt|last_damage_taken|turns_active|turns_active_total)'\]$")
# A2 review L1: a terminal step whose residual ran in both trees or in neither. Only the end-of-turn writes gated
# on `done` differ: the Truant flip of the survivor (transition, `ordinary & ~work.done`) and turn_end's upkeep
# (`residual_ready & ~work.done`: the Sleep Talk pending clocks, the port's post-upkeep flag, negative items).
FAMILIES['terminal'] = (r'^state\.truant_loaf$',
    r"^trackers\.mon_features\['(sleep_skipped_turns|sleep_cant_pending|sleep_usable_pending)'\]$",
    r'^trackers\.public_port\.post_upkeep$', r'^trackers\.negative_items\.')
# M1 GPU pass: a KO that moved (between the trees or between the action and residual phases) leaves a Color Change
# holder's type differing: it retypes only when the hit leaves it standing (data/mods/gen4/abilities.ts:46-47).
# Accepted per side on a cascade lane only where a holder on that side fainted this step in this tree or
# (un)fainted relative to the base (retype_ok).
RETYPE_LEAVES = ('state.types', 'trackers.shown_types')
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
    # B. Iron Tail's Def drop is the target's stage only (nothing later in the step reads its own Def without
    # confusion, which is its own involvement). Steel Wing's Def raise lowers the physical damage the user takes
    # later in the step: its HP, Substitute and HP-gated items and Wish, and the attacker's recoil. Meteor Mash's
    # Atk raise comes after the user's only hit of the step. A flinch or a confusion (Signal Beam, or a confused
    # active) decides whether the target acts at all, so, like a sleep landing (A1), it may change anything.
    'irontail': dict(user=(), target=('boosts',), glob=()),
    'steelwing': dict(user=('boosts', 'hp', 'item', 'visible', 'subtracker', 'wish'), target=('hp', 'item', 'boosts', 'visible'),
                      glob=()),
    'meteormash': dict(user=('boosts',), target=(), glob=()),
    'headbutt': ALL, 'needlearm': ALL, 'signalbeam': ALL, 'confusion': ALL,
    # M1. Hyper Beam's own changes (the counter and the bag bit) are explained only by recharge_rule, a post-state
    # check on an involved side; nothing else may change. A recharge turn (a pre-state counter) decides whether the
    # side acts at all, so, like a sleep landing, it may change anything (no capture carries one).
    'hyperbeam': dict(user=(), target=(), glob=()),
    'recharge': ALL,
    # Trick: the actives' items and the device ledger, flags and narrowing on both seats (with the visible belief); the
    # Choice lock follows the current item (the Tricker no longer locks, a slower receiver does); the swapped items' HP
    # effects follow on either side: Choice Band damage (HP, a Substitute's HP, Wish landing by HP) and Leftovers. A
    # swapped berry's stages (pinch), cure (Lum: status, clause port, confusion) and a Substitute's bag bit and presence
    # are never allowed here: trick_rule and bag_rule explain them per mon from post-state witnesses (M1 review).
    'trick': dict(user=('item', 'itemflags', 'choice', 'hp', 'wish'), target=('item', 'itemflags', 'choice', 'hp', 'wish'), glob=()),
    # M2. The counters, the source, the bag cells and the wrap clock are explained only by m2_rule (post-state, per mon);
    # the allowances are what follows from them. M2 review: Perish Song's Soundproof and Yawn's Insomnia/Vital Spirit
    # -immune reveals are explained only by m2_rule's holder witness (no allowance), and Wrap's tick in the target's HP
    # only by its digest witness (state.hp and a Substitute's HP are never allowed here); what remains is the tick's HP
    # lines (damage history, negative items) and the belief refresh of an HP line.
    'perishsong': dict(user=(), target=(), glob=()),
    'yawn': dict(user=(), target=(), glob=()),
    'wrap': dict(user=(), target=('hp_lines', 'visible'), glob=()),
    # Volatiles a pre-step state already carries (no capture has one: every captured counter is a migrated 0). A Perish
    # faint, a Yawn sleep (with Lum and the clause port) at the residual, and a partial trap's tick, release and switch
    # block (a captured switch the trap now refuses freezes the lane, so it may change anything, like M1's recharge).
    'perish': dict(user=('hp', 'm2', 'reveal'), target=(), glob=()),
    'drowsy': dict(user=('status', 'item', 'clause', 'visible', 'm2'), target=('clause',), glob=()),
    'partial_trap': ALL,
}
# M2: the leaves m2_rule explains, for the pre-step effects above.
FAMILIES['m2'] = tuple(f'^{re.escape(n)}$' for n in M2_COUNTER_LEAVES) + (rf'^{re.escape(WRAP_CLOCK)}$', r'^trackers\.public_volatiles$')
# WH. White Herb's own writes (the holder's item and stages, the device's consumption reveal, ledger and flags) are
# explained only by wh_rule (post-state, per mon, from this tree's recorded valid uses); the allowances are what follows
# from a restored stage: damage dealt and taken on both sides (HP, a Substitute and its device presence, a pending Wish
# landing by HP, the belief refresh of an HP line). wh_rule also drops the effect from a lane with no valid recorded use.
# Facade: its doubled damage (the target's HP and what follows, the user's HP from a Counter reply); facade_rule keeps the
# effect only for a side whose own Facade line ran under par/brn/psn/tox in this tree. Sacred Fire: whether a frozen user
# acts at all, so, like a sleep landing (A1), anything; only a frozen user is involved. pinch_rule (both effects): a pinch
# berry eaten in exactly one tree because the damage differs, per mon (M1's pinch witness).
ALLOWED['whiteherb'] = dict(user=('hp', 'visible', 'subtracker', 'wish'), target=('hp', 'visible', 'subtracker', 'wish'), glob=())
ALLOWED['facade'] = dict(user=('hp', 'visible', 'subtracker', 'wish'), target=('hp', 'visible', 'subtracker', 'wish'), glob=())
ALLOWED['sacredfire'] = ALL


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
    # WH (White Herb plus C-lite, which replaces C's draft; Psycho Boost's drop is unchanged, its restore is White Herb's).
    'WH': dict(whiteherb=white_herb, facade=facade_status, sacredfire=frozen_defrost),
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
    accepted on both sides, e.g. ('residual', 'residual_trackers') on a phase shift, optional effect_ok=[2] bool:
    this tree's saved effect block equals the design derivation per side (state.effect_draws is accepted only there),
    optional retype_ok=callable(leaf, side) -> bool: that side's RETYPE_LEAVES differ from the base only at a Color Change
    holder that fainted this step here or (un)fainted relative to the base).
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
                  or (side is not None and leaf == EFFECT_DRAWS and cascade.get('effect_ok') is not None
                      and bool(cascade['effect_ok'][side]))
                  or (side is not None and leaf in RETYPE_LEAVES and cascade.get('retype_ok') is not None
                      and bool(cascade['retype_ok'](leaf, side)))
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


# B: a faint's clearVolatile (sim/battle.ts:2576) zeroes the fainted mon's Choice lock and confusion at the step end.
CLEAR_VOLATILE_LEAVES = ('state.choice_move', 'state.confusion_counter')


def clear_volatile_rule(entries, post, base_post, fainted):
    """B special rule (a post-state rule like PR0's faint rule; not involvement): a faint's clearVolatile zeroes
    the fainted mon's Choice lock and confusion. post/base_post: {leaf: [2,6]} this lane's post values in this
    tree / the base; fainted: [2,6] this tree's mons whose HP went >0 -> 0 this step. A (leaf, side) entry is
    explained iff every differing mon on that side fainted this step and is 0 in this tree; anything else stays
    for involvement. Returns (remaining entries, explained entries)."""
    explained = set()
    for leaf, side in entries:
        if leaf in post and leaf in base_post and side is not None:
            a, b = np.asarray(post[leaf][side]), np.asarray(base_post[leaf][side])
            differ = a != b
            if differ.any() and not (differ & ~(np.asarray(fainted[side], bool) & (a == 0))).any():
                explained.add((leaf, side))
    return entries - explained, sorted(explained, key=str)


# M1 special rules (post-state; see the docstring).
LEDGER_LEAVES = ("trackers.mon_features['item_mutated']", "trackers.mon_features['item_original']")
MUSTRECHARGE = 'mustrecharge'
BAG = 'trackers.public_volatiles'
# M1 review: per-lane base arrays the per-mon witnesses compare against (stored by the base pass as wit_XX).
WITNESS_LEAVES = ('state.types', 'state.substitute_hp', BAG, 'trackers.boosts', 'trackers.shown_types', 'trackers.status')
# This tree's post values the M1 rules read besides the KO and witness arrays.
M1_POST_LEAVES = ('state.item',)
# Independent copies of the engine's Gen 3 item subset the Trick witnesses read (gpu_engine/items.py).
PINCH_STATS = {'liechiberry': 0, 'petayaberry': 2, 'salacberry': 4}
LUM_BERRY = 'lumberry'
STATUS_MON_LEAVES = ('state.status', 'state.status_counter', 'state.sleep_skipped', 'state.rest_origin')
SLEEP_FEATURES = tuple(f"trackers.mon_features['{k}']" for k in ('sleep_turns', 'rest_sleep', 'toxic_stage', 'sleep_skipped_turns',
                                                                  'sleep_cant_pending', 'sleep_usable_pending'))
CLAUSE_LEAVES = ('trackers.public_port.holder', 'trackers.public_port.victims')
CONFUSION_FEATURE = "trackers.mon_features['confusion_turns']"
SUB_FEATURE = "trackers.mon_features['sub_present']"


def _differ(a, b):
    """[6] mons (or public slots) whose value differs, over any trailing axes."""
    a, b = np.asarray(a), np.asarray(b)
    return (a != b).reshape(a.shape[0], -1).any(-1)


def _slots(mask, physical_to_public):
    """[6] physical mons -> [6] their public slots (this tree's mapping)."""
    out = np.zeros(6, bool)
    for k in np.flatnonzero(np.asarray(mask, bool)):
        slot = int(physical_to_public[k])
        if 0 <= slot < 6:out[slot] = True
    return out


def recharge_rule(entries, counter, bag, hyperbeam, evidence, residual_ran, post_active, post_hp, physical_to_public, departed):
    """M1: Hyper Beam's mustrecharge counter. counter [2,6] this tree's post recharge_counter (physical); bag [2,6] its post
    mustrecharge bag column (public slots); hyperbeam [2] involvement; evidence [2] post-state proof in this tree that the
    side's move connected (its recorded direct damage, or the foe's Substitute falling); residual_ran (None: unknown);
    post_active [2]; post_hp [2,6]; departed [2,6] pre-step actives that left the field. A (state.recharge_counter, side)
    entry is explained iff the side may have used Hyper Beam and connected, and the counter is nonzero exactly at its
    living post-step active with the value the residual leaves (1 ran, 2 deferred), with the mustrecharge bag bit set at
    that active's public slot. Returns (remaining entries, explained entries, [2,6] public slots whose mustrecharge bag
    column may differ: that recharging active's, and those of mons that departed after the hit)."""
    explained = set();counter = np.asarray(counter);bag = np.asarray(bag, bool);cells = np.zeros((2, 6), bool)
    values = (1, 2) if residual_ran is None else ((1,) if residual_ran else (2,))
    for side in (0, 1):
        if not (bool(hyperbeam[side]) and bool(evidence[side])):continue
        active = int(post_active[side]);nonzero = counter[side] != 0
        slot = int(physical_to_public[side, active]) if 0 <= active < 6 else -1
        recharging = (0 <= slot < 6 and bool(nonzero[active]) and int(nonzero.sum()) == 1 and int(post_hp[side, active]) > 0
                      and int(counter[side, active]) in values and bool(bag[side, slot]))
        if recharging:
            cells[side, slot] = True
            if ('state.recharge_counter', side) in entries:explained.add(('state.recharge_counter', side))
        cells[side] |= _slots(departed[side], physical_to_public[side])
    return entries - explained, sorted(explained, key=str), cells


def bag_rule(entries, bag, base_bag, allowed):
    """M1 (review): a (trackers.public_volatiles, side) entry is explained iff every (public slot, column) where this tree's
    bag differs from the base's stored bag is an allowed cell. bag/base_bag [2,6,C] (None: unknown, nothing explained);
    allowed [2,6,C] the witnessed cells."""
    if bag is None or base_bag is None or allowed is None:return entries, []
    explained = set()
    for leaf, side in entries:
        if leaf != BAG or side is None:continue
        diff = np.asarray(bag[side]) != np.asarray(base_bag[side])
        if diff.any() and not (diff & ~np.asarray(allowed[side], bool)).any():explained.add((leaf, side))
    return entries - explained, sorted(explained, key=str)


def item_witnesses(pre_active, post_active, pre_item, post_item, post_status, post_confusion, pre_sub, post_sub, base_sub,
                   lum, pinch):
    """M1 (review): per-mon witnesses on a Trick lane, from this tree's pre/post state and the base's stored Substitutes.
    acts [2,6]: either side's pre- or post-step active (a mon switched in before the Trick is its target). The pool is
    the pre-step items of those mons. lum [2,6]: an active whose Lum Berry outcome differs from the base's (a Lum in the
    pool, eaten here: post item 0 with no status or confusion left; or the holder's own Lum gone to the other side).
    sub [2,6]: an active whose Substitute stood before the step and stands in exactly one tree. stats: the stats of the
    pinch berries in the pool (pinch: item id -> stat)."""
    pre_item, post_item = np.asarray(pre_item), np.asarray(post_item)
    acts = np.zeros((2, 6), bool)
    for side in (0, 1):
        for active in (int(pre_active[side]), int(post_active[side])):
            if 0 <= active < 6:acts[side, active] = True
    pool = {int(pre_item[side, m]) for side, m in zip(*np.nonzero(acts))}
    lum_mons = np.zeros((2, 6), bool)
    if lum is not None and lum in pool:
        eaten = (post_item == 0) & (np.asarray(post_status) == 0) & (np.asarray(post_confusion) == 0)
        lum_mons = acts & (eaten | ((pre_item == lum) & (post_item != lum)))
    sub_mons = np.zeros((2, 6), bool)
    if base_sub is not None:
        sub_mons = acts & (np.asarray(pre_sub) > 0) & ((np.asarray(post_sub) > 0) != (np.asarray(base_sub) > 0))
    return dict(acts=acts, lum=lum_mons, sub=sub_mons, stats={pinch[i] for i in pool if i in pinch})


def pinch_boosts(boosts, base_boosts, w, pre_item, post_item, pinch, side):
    """M1 (review): ([6] differing mons, ok) for state.boosts on a side: every mon whose stages differ from the base's is an
    active whose difference is one stage in one stat of a pinch berry in the pool (w['stats']): +1 eaten here (its
    post item is 0), -1 eaten only in the base (its own berry, the item it held before the step)."""
    d = np.asarray(boosts[side]).astype(np.int64) - np.asarray(base_boosts[side]).astype(np.int64)
    mons = d.reshape(6, -1).any(-1);ok = bool(mons.any())
    for m in np.flatnonzero(mons):
        cols = np.flatnonzero(d[m])
        c = int(cols[0])
        ok &= (bool(w['acts'][side, m]) and len(cols) == 1 and c in w['stats'] and abs(int(d[m, c])) == 1
               and (int(post_item[side, m]) == 0 if d[m, c] > 0 else pinch.get(int(pre_item[side, m])) == c))
    return mons, ok


def trick_rule(entries, w, ko, base_ko, wit, base_wit, physical_to_public, pre_item, post_item, pinch):
    """M1 (review): on a Trick lane, explain status, confusion, boosts and Substitute-presence entries only by the per-mon
    witnesses (item_witnesses; ko/base_ko this tree's and the base's per-mon KO arrays, wit/base_wit the witness arrays,
    all [2,6,...] for this lane). Returns (remaining entries, explained entries, [2,6] confusion-witnessed mons)."""
    explained = set();status_side = np.zeros(2, bool);confused = np.zeros((2, 6), bool)

    def per_mon(leaf, side, mons):
        if leaf not in ko or leaf not in base_ko:return False
        diff = _differ(ko[leaf][side], base_ko[leaf][side])
        return bool(diff.any()) and not (diff & ~mons).any()

    def per_slot(leaf, side, mons):
        if leaf not in wit or leaf not in base_wit:return False
        diff = _differ(wit[leaf][side], base_wit[leaf][side])
        return bool(diff.any()) and not (diff & ~_slots(mons, physical_to_public[side])).any()
    for side in (0, 1):
        lum = w['lum'][side]
        for leaf in STATUS_MON_LEAVES:
            if per_mon(leaf, side, lum):
                status_side[side] = True
                if (leaf, side) in entries:explained.add((leaf, side))
        if per_mon('state.confusion_counter', side, lum):
            confused[side] = _differ(ko['state.confusion_counter'][side], base_ko['state.confusion_counter'][side])
            if ('state.confusion_counter', side) in entries:explained.add(('state.confusion_counter', side))
        if ('trackers.status', side) in entries and per_slot('trackers.status', side, lum):explained.add(('trackers.status', side))
        if 'state.boosts' in ko and 'state.boosts' in base_ko:
            mons, ok = pinch_boosts(ko['state.boosts'], base_ko['state.boosts'], w, pre_item, post_item, pinch, side)
            if ok:
                if ('state.boosts', side) in entries:explained.add(('state.boosts', side))
                if ('trackers.boosts', side) in entries and per_slot('trackers.boosts', side, mons):
                    explained.add(('trackers.boosts', side))
    for leaf, side in entries:
        if side is not None and leaf in SLEEP_FEATURES and status_side[side]:explained.add((leaf, side))
        elif side is not None and leaf == CONFUSION_FEATURE and confused[side].any():explained.add((leaf, side))
        elif side is not None and leaf == SUB_FEATURE and w['sub'][side].any():explained.add((leaf, side))
        elif leaf in CLAUSE_LEAVES and status_side.any():explained.add((leaf, side))
    return entries - explained, sorted(explained, key=str), confused


def ledger_rule(entries, mutated, original, knocked_pre, knocked_post, item_pre, physical_to_public, *, trick=False):
    """M1: B2's item ledger written by A2's Knock Off (public_items REMOVE: mutated, and the removed item as the
    original). mutated/original [2,6] this tree's post mon_features (public slots); knocked_pre/knocked_post [2,6]
    state.item_knocked before and after the step and item_pre [2,6] state.item before it (physical). A ledger (leaf,
    side) entry is explained iff both keys are nonzero exactly at the public slots of that side's mons knocked this
    step, with item_original their pre-step item. A Trick lane (trick=True) is left to the trick allowance."""
    explained = set()
    if trick:return entries, []
    for leaf, side in entries:
        if side is None or leaf not in LEDGER_LEAVES:continue
        expect_mutated = np.zeros(6, np.int64);expect_original = np.zeros(6, np.int64)
        newly = np.asarray(knocked_post[side], bool) & ~np.asarray(knocked_pre[side], bool)
        ok = True
        for k in np.flatnonzero(newly):
            slot = int(physical_to_public[side, k])
            if not 0 <= slot < 6:ok = False;continue
            expect_mutated[slot] = 1;expect_original[slot] = int(item_pre[side, k])
        ok = ok and bool(newly.any()) and np.array_equal(np.asarray(mutated[side]), expect_mutated) \
            and np.array_equal(np.asarray(original[side]), expect_original)
        if ok:explained.add((leaf, side))
    return entries - explained, sorted(explained, key=str)


# M2 special rules (post-state; see the docstring). The counters and the wrap clock are leaves the captures predate, so
# any nonzero value differs from its migrated default and must be explained here (or by a pre-state effect's allowance).


def _m2_values(ran, done, settled, started):
    """Counter values a volatile started this step may show at the step end: `settled` after its residual tick, `started`
    when the residual did not run (deferred) or the battle ended before its handler (a residual KO)."""
    if ran is None:return {settled, started}
    if not ran:return {started}
    return {settled, started} if done else {settled}


def m2_rule(entries, w):
    """M2: Perish Song, Yawn and Wrap set this step, from this tree's post state and hook evidence (w: this lane's arrays).

    perish (state.perish_counter on side s): Perish Song executed on some side (a move line with use); side s's counter is
    nonzero only at its living post-step active, which is not Soundproof, with 3 after the order-12 tick or 4 without it.
    yawn (side s): the other side's Yawn line had use, met no Protect, and found side s's active unstatused, without a
    Substitute and alive; the counter is nonzero only at that same, living, post-step active (not Insomnia/Vital Spirit), 1
    after the residual or 2 without it. wrap (counter and source on side s): the other side's Wrap line dealt HP damage; the
    counter is nonzero only at that same target, still active and alive, with source the Wrap user's party index and the
    value 3+draws[f][7], less one after the residual tick. The wrap clock (B2's WRAP_TRAP_TURNS) is nonzero only at that
    trapped mon's public slot, 1 when the step's turn_end was ordinary.
    M2 review: a ticked counter (3+draws[f][7]-1) also needs the trapper held at the holder's handler: the Wrap user
    the other side's living active with activeTurns > 0 there (else the handler releases the trap silently,
    data/mods/gen5/conditions.ts), read at the step's own HP-line cutpoint after that handler (tick_seen/tick_trapper/
    tick_held); a slower trapper may faint from its own residual after the tick. Without a recorded cutpoint (a
    stand-in step) the step end stands in (nothing after the residual moves an active). On a side the other side's Wrap struck,
    state.hp and state.substitute_hp must be explained here (`must`: never an allowance or a KO cascade): state.hp only
    when adding the tick, max(1, maxhp//16), back at the trapped active reproduces the base's per-side digest exactly
    (after a witnessed tick; or, for a lethal tick that cleared the trap with the trapper held, any base HP 1..tick); a
    Substitute's HP never. Where the handler cutpoint recorded the holder's HP right after its tick (tick_hp), the base is
    taken as pre + min(later, max - pre), pre = tick_hp + tick and later = this tree's change after the handler: a later
    heal (a Leech Seed drain from a slower trapper) is equal in both trees but the base's stops at max HP. A reveal entry (shown/original ability, belief) on side x is explained only when x's post-step
    active holds Soundproof on a Perish Song lane or Insomnia/Vital Spirit after the other side's unblocked Yawn line at it.
    Returns (remaining entries, explained entries, [2,6,len(M2_BAG)] bag cells the witnessed volatiles explain, each equal
    to the derived bit, the entries that must be explained here)."""
    explained = set();cells = np.zeros((2, 6, len(M2_BAG)), bool);must = set()
    turns = np.asarray(w.get('active_turns', np.ones(2, np.int32)));max_hp = w.get('max_hp');base_hp = w.get('base_hp')
    bag = w.get('bag');p2p = np.asarray(w['p2p']);active = np.asarray(w['post_active']);hp = np.asarray(w['post_hp'])
    ran, done = w.get('ran'), bool(w.get('done', False))
    def slot(side, mon):
        k = int(p2p[side, mon]) if 0 <= mon < 6 else -1
        return k if 0 <= k < 6 else -1
    def cell_ok(side, k, values):
        """The bag's M2 columns at (side, public slot k) equal `values` (a dict column -> bit) where given."""
        if bag is None or k < 0:return False
        return all(bool(bag[side, k, M2_BAG.index(c)]) == v for c, v in values.items())
    for side in (0, 1):
        f = 1 - side;a = int(active[side]);k = slot(side, a);live = 0 <= a < 6 and int(hp[side, a]) > 0
        # Perish Song.
        counter = np.asarray(w['perish'][side]);nz = counter != 0
        if ('state.perish_counter', side) in entries and nz.any():
            v = int(counter[a]) if 0 <= a < 6 else 0
            bits = {c: (v > 0 and min(v, 3) == i) for i, c in enumerate(M2_BAG[:4])}
            ok = (bool(np.asarray(w['perish_used']).any()) and live and int(nz.sum()) == 1 and bool(nz[a])
                  and int(w['ability'][side][a]) not in w['soundproof'] and v in _m2_values(ran, done, 3, 4) and cell_ok(side, k, bits))
            if ok:
                explained.add(('state.perish_counter', side))
                for i in range(4):cells[side, k, i] = True
        # Yawn.
        counter = np.asarray(w['yawn'][side]);nz = counter != 0
        if ('state.yawn_counter', side) in entries and nz.any():
            v = int(counter[a]) if 0 <= a < 6 else 0
            ok = (bool(w['yawn_by'][f]) and not bool(w['blocked'][f]) and int(w['yawn_target'][f]) == a and live
                  and int(nz.sum()) == 1 and bool(nz[a]) and int(w['ability'][side][a]) not in w['sleep_immune']
                  and v in _m2_values(ran, done, 1, 2) and cell_ok(side, k, {'yawn': True}))
            if ok:
                explained.add(('state.yawn_counter', side));cells[side, k, M2_BAG.index('yawn')] = True
        # Wrap: counter, source and the clock together; the trapper at the residual; the tick in the target's HP.
        counter = np.asarray(w['trap'][side]);source = np.asarray(w['source'][side]);nz = (counter != 0) | (source != -1)
        mine = {('state.partial_trap_counter', side), ('state.partial_trap_source', side), (WRAP_CLOCK, side)} & entries
        d = 3 + int(w['draws7'][f]);user = int(w['wrap_user'][f])
        struck = bool(w['wrap_by'][f]) and int(w['wrap_target'][f]) == a
        if w.get('tick_seen') is not None:
            # The trapper at the holder's handler (the step's own cutpoint): the Wrap user, alive, activeTurns > 0.
            held = (bool(w['tick_seen'][side]) and int(w['tick_trapper'][side]) == user and 0 <= user < 6
                    and bool(w['tick_held'][side]))
        else:
            # No handler cutpoint recorded (a stand-in step): the step end, where nothing after the residual moves an active.
            held = 0 <= user < 6 and int(active[f]) == user and int(hp[f, user]) > 0 and int(turns[f]) > 0
        ticked = False
        if mine and nz.any():
            v = int(counter[a]) if 0 <= a < 6 else 0
            ok = (struck and live and int(nz.sum()) == 1 and bool(nz[a]) and int(source[a]) == user
                  and v in _m2_values(ran, done, d - 1, d) and (v != d - 1 or held) and cell_ok(side, k, {'partiallytrapped': True}))
            # The clock is checked whenever the trap is (a missing tick of the clock is a missing producer), and must be
            # present when the clock leaf changed.
            clock = w.get('clock')
            if ok and (clock is not None or (WRAP_CLOCK, side) in mine):
                want = np.zeros(6, np.int64)
                if k >= 0 and bool(w.get('ordinary', False)):want[k] = 1
                ok = clock is not None and np.array_equal(np.asarray(clock[side]).astype(np.int64), want)
            if ok:
                explained |= mine;cells[side, k, M2_BAG.index('partiallytrapped')] = True;ticked = v == d - 1
        hp_mine = {('state.hp', side), ('state.substitute_hp', side)} & entries
        if bool(w['wrap_by'][f]) and hp_mine:
            must |= hp_mine
            if ('state.hp', side) in hp_mine and base_hp is not None and max_hp is not None and 0 <= a < 6:
                tick = max(1, int(max_hp[side][a]) // 16);now = int(hp[side, a]);top = int(max_hp[side][a])
                after = int(w['tick_hp'][side]) if w.get('tick_hp') is not None else -1
                if ticked and after >= 0:
                    # The holder had after+tick before its handler. A later change in the residual (a heal: the drain of
                    # a Leech Seed it planted on a slower trapper) is the same amount in both trees, the trapper being
                    # untouched by the tick, but the base's heal stops at max HP: the base ends at pre + min(later, max - pre).
                    pre = after + tick;later = now - after
                    candidates = (pre + (min(later, top - pre) if later >= 0 else later),)
                elif ticked:candidates = (now + tick,)
                elif (struck and now == 0 and int(counter[a]) == 0 and int(source[a]) == -1 and ran is not False and held):
                    candidates = range(1, tick + 1)                 # a lethal tick: the base kept 1..tick HP there
                else:candidates = ()
                want = np.asarray(base_hp[side], np.uint32)
                for c in candidates:
                    trial = np.array(hp[side], np.int32);trial[a] = c
                    if np.array_equal(mon_digest(trial), want):explained.add(('state.hp', side));break
    # The -immune reveals, each at a holder of the immunity the line met.
    for x in (0, 1):
        reveals = {e for e in entries if e[1] == x and allowed_leaf(e[0], ('reveal',))}
        a = int(active[x])
        if not reveals or not 0 <= a < 6:continue
        ability = int(w['ability'][x][a])
        sound = ability in w['soundproof'] and bool(np.asarray(w['perish_used']).any())
        sleepy = (ability in w['sleep_immune'] and bool(w['yawn_by'][1 - x]) and not bool(w['blocked'][1 - x])
                  and int(w['yawn_target'][1 - x]) == a)
        if sound or sleepy:explained |= reveals
    return entries - explained, sorted(explained, key=str), cells, must


def _np_fmix(h):
    h = h ^ (h >> np.uint32(16));h = h * np.uint32(0x85EBCA6B);h = h ^ (h >> np.uint32(13));h = h * np.uint32(0xC2B2AE35)
    return h ^ (h >> np.uint32(16))


def mon_digest(values):
    """numpy copy of lane_digest for one lane's per-side value (here a side's [6] HP): the two words the base pass stores
    for that (leaf, side). The sum is over elements, so this reproduces a digest of an edited side exactly."""
    u = np.ascontiguousarray(np.asarray(values, np.int32).reshape(-1)).view(np.uint32)
    index = np.arange(u.size, dtype=np.uint32) * np.uint32(0x9E3779B9)
    return np.array([np.sum(_np_fmix(u ^ (index + np.uint32(seed))), dtype=np.uint32) for seed in DIGEST_SEEDS], np.uint32)


# M2: the effects whose lanes the wrong-mon check covers, and the per-mon leaves it checks.
M2_EFFECTS = ('perishsong', 'yawn', 'wrap', 'perish', 'drowsy', 'partial_trap')
M2_MON_LEAVES = ('state.hp', 'state.status', 'state.status_counter') + M2_COUNTER_LEAVES


def m2_wrong_mon(entries, post, ref, pre_active, post_active):
    """M2: the (leaf, side) entries of M2_MON_LEAVES whose per-mon difference lies off that side's pre-step and post-step
    actives (an effect landing on a benched mon). post: this tree's per-lane [2,6] post values; ref: per leaf, the base's
    post values where both passes store it (the KO arrays), else this tree's pre-step values."""
    ok = active_mask(pre_active) | active_mask(post_active)
    bad = []
    for leaf, side in sorted(entries, key=str):
        if side is None or leaf not in M2_MON_LEAVES or leaf not in post or leaf not in ref:continue
        if (_differ(post[leaf][side], ref[leaf][side]) & ~ok[side]).any():bad.append((leaf, side))
    return bad


# WH special rules (post-state, per mon). The base pass stores these as per-lane witness arrays too (wit_XX): the holder's
# item, item flags and stages, and the device's item reveal, removal, narrowing, visible basis, ledger, flags and credit.
WH_MON_LEAVES = ('state.item', 'state.item_knocked', 'state.boosts')
WH_SLOT_LEAVES = ('trackers.shown_item', 'trackers.item_removed', 'trackers.belief_mask', 'trackers.boosts') + tuple(
    f"trackers.mon_features['{k}']" for k in ('item_swapped', 'choice_public', 'choice_locked', 'item_mutated', 'item_original'))
WH_VISIBLE = 'trackers.visible_belief.'
WH_ORIGINAL = "trackers.mon_features['item_original']"
WH_CREDIT = "trackers.field_features['items_removed_credit']"


def witness_names(names):
    """The per-lane witness arrays both passes return (sorted): M1's, and WH's per-mon/per-slot item and stage leaves."""
    wanted = set(WITNESS_LEAVES) | set(WH_MON_LEAVES) | set(WH_SLOT_LEAVES) | {WH_CREDIT}
    return sorted(n for n in names if n in wanted or n.startswith(WH_VISIBLE))


def wh_rule(entries, w):
    """WH: White Herb's own writes, from this tree's post state and its recorded uses (w: this lane's arrays).

    Evidence (Stepper, recorded around the device hook white_herb with this tree's before/after states at each trigger):
    used [2,6] the mons with a valid use (the side's active held White Herb and lived, had a negative stage, and
    afterwards had no item and exactly max(stage, 0), no other state leaf of any mon changed in that call, and the hook
    left the device's shown_item White Herb and item_removed set at its public slot: the producer), raise [2,6,7] the
    stages each use restored, bad: a call with use set that was not such a use, or one without use that changed a mon's
    item or stages. A bad lane explains nothing here (the caller forces it unexpected).
    With wit / base (this tree's and the base pass's per-lane witness arrays), p2p [2,6] (physical -> public slot) and
    acting [2,6] (each side's pre- or post-step active), an entry of side s is explained when:
    - state.boosts: every mon whose stages differ is a used mon with this tree's stages = the base's + its raise (the base
      kept the negatives; later changes in the step are the same in both trees), or one whose faint differs (ko [2,6]).
    - trackers.boosts: the same per public slot (the device copies an active's stages; ko slots follow the cascade).
    - item family (state.item, state.item_knocked, shown_item, item_removed, belief_mask, the visible basis, the ledger,
      the per-stint flags): every differing mon or slot is an active of either side (a used holder, or a Trick partner,
      the herb having moved in the base only) and the side has a used mon; state.item differs at a used mon only from
      White Herb (base) to 0, and at another active only from White Herb (the base's Trick handed it the herb) to 0;
      state.item_knocked differs only at a used mon the base knocked off (base True, here False).
    - the removal credit (field_features items_removed_credit, per side): the base's exceeds this tree's by exactly the
      used mons the base knocked off on that side.
    - credit [2] (recorded at the trick hook from the recorder's own due mask): a side whose Trick partner credit B2
      loses (a receiver's herb lines end the pairing); with no use or partner of its own there, only its active's ledger
      original may differ, unrecorded (0) here.
    Returns (remaining entries, explained entries)."""
    explained = set();wit, base = w['wit'], w['base'];used = np.asarray(w['used'], bool);herb = w['herb']
    if w.get('bad') or not used.any():return entries, []
    credit = np.asarray(w.get('credit', np.zeros(2, bool)), bool)
    raised = np.asarray(w['raise']);ko = np.asarray(w.get('ko', np.zeros((2, 6), bool)), bool)
    p2p = np.asarray(w['p2p']);acting = np.asarray(w['acting'], bool)
    def have(leaf):return leaf in wit and leaf in base
    def slots(mask, side):return _slots(mask, p2p[side])
    knocked_base = np.zeros((2, 6), bool)
    if have('state.item_knocked'):
        knocked_base = used & np.asarray(base['state.item_knocked'], bool) & ~np.asarray(wit['state.item_knocked'], bool)
    # A Trick partner: an active that holds nothing here but the herb in the base (the base's Trick handed it over).
    partner = np.zeros((2, 6), bool)
    if have('state.item'):
        partner = acting & (np.asarray(base['state.item']) == herb) & (np.asarray(wit['state.item']) == 0) & ~used
    for side in (0, 1):
        mine = used[side]
        if not (mine.any() or partner[side].any() or credit[side]):continue
        for leaf, s_ in sorted(entries, key=str):
            if s_ != side or not have(leaf):continue
            if leaf not in WH_MON_LEAVES and leaf not in WH_SLOT_LEAVES and not leaf.startswith(WH_VISIBLE):continue
            a, b = np.asarray(wit[leaf])[side], np.asarray(base[leaf])[side];diff = _differ(a, b)
            if not diff.any():continue
            if leaf == 'state.boosts':
                good = mine & np.all(a == b + raised[side], axis=-1)
                ok = not (diff & ~(good | ko[side])).any()
            elif leaf == 'trackers.boosts':
                good = np.zeros(6, bool)
                for m in np.flatnonzero(mine):
                    k = int(p2p[side, m])
                    if 0 <= k < 6 and np.array_equal(a[k], b[k] + raised[side, m]):good[k] = True
                ok = not (diff & ~(good | slots(ko[side], side))).any()
            elif leaf == 'state.item':
                ok = not (diff & ~((mine | partner[side]) & (b == herb) & (a == 0))).any()
            elif leaf == 'state.item_knocked':
                ok = not (diff & ~knocked_base[side]).any()
            elif credit[side] and not (mine.any() or partner[side].any()):
                # Only a lost Trick partner credit on this side: its active's ledger original, unrecorded here, and nothing
                # else (review round 2).
                ok = leaf == WH_ORIGINAL and not (diff & ~(slots(acting[side], side) & (np.asarray(a) == 0))).any()
            elif leaf in WH_SLOT_LEAVES or leaf.startswith(WH_VISIBLE):
                ok = not (diff & ~slots(acting[side], side)).any()
            else:
                continue
            if ok:explained.add((leaf, side))
    if have(WH_CREDIT):
        a, b = np.asarray(wit[WH_CREDIT]), np.asarray(base[WH_CREDIT])
        for side in (0, 1):
            if (WH_CREDIT, side) in entries and int(b[side]) - int(a[side]) == int(knocked_base[side].sum()) > 0:
                explained.add((WH_CREDIT, side))
    return entries - explained, sorted(explained, key=str)


def pinch_rule(entries, w):
    """WH: a pinch berry (Liechi, Petaya, Salac) eaten in exactly one tree on a White Herb or Facade lane, whose changed
    damage moved its holder across the quarter-HP line (the GPU pass at 5951098e: a doubled Facade's target ate its berry
    in this tree only). w: wit / base (this tree's and the base's witness arrays for the lane), pre_item [2,6] (this
    lane's pre-step items), pinch {item id: stat}, p2p, acting [2,6]. Per side, the berry mons are actives whose pre-step
    item is a pinch berry held at the end in exactly one tree; then state.item differs only there, state.boosts only there
    by exactly +1 (eaten here) or -1 (eaten only in the base) in the berry's own stat (trackers.boosts likewise per public
    slot), and the device's item reveal, removal, narrowing, visible basis and flags only at their slots.
    Returns (remaining entries, explained entries)."""
    wit, base, pinch = w['wit'], w['base'], w['pinch'];explained = set()
    if not pinch or 'state.item' not in wit or 'state.item' not in base:return entries, []
    pre = np.asarray(w['pre_item']);p2p = np.asarray(w['p2p']);acting = np.asarray(w['acting'], bool)
    stat = np.vectorize(lambda i: pinch.get(int(i), -1))(pre)
    here, there = np.asarray(wit['state.item']), np.asarray(base['state.item'])
    sign = np.where(acting & (stat >= 0) & (here == 0) & (there == pre), 1,
                    np.where(acting & (stat >= 0) & (there == 0) & (here == pre), -1, 0))
    for side in (0, 1):
        berry = sign[side] != 0
        if not berry.any():continue
        slots = _slots(berry, p2p[side])
        def staged(a, b, mons_stat, mons_sign):
            d = np.asarray(a).astype(np.int64) - np.asarray(b).astype(np.int64);ok = True
            for k in np.flatnonzero(d.reshape(d.shape[0], -1).any(-1)):
                want = np.zeros(d.shape[1], np.int64)
                if mons_stat[k] < 0:return False
                want[mons_stat[k]] = mons_sign[k];ok &= bool(np.array_equal(d[k], want))
            return ok
        for leaf, s_ in sorted(entries, key=str):
            if s_ != side or leaf not in wit or leaf not in base:continue
            a, b = np.asarray(wit[leaf])[side], np.asarray(base[leaf])[side]
            if a.ndim == 0:continue
            diff = _differ(a, b)
            if not diff.any():continue
            if leaf == 'state.item':ok = not (diff & ~berry).any()
            elif leaf == 'state.boosts':ok = not (diff & ~berry).any() and staged(a, b, np.where(berry, stat[side], -1), sign[side])
            elif leaf == 'trackers.boosts':
                by_slot_stat = np.full(6, -1);by_slot_sign = np.zeros(6, np.int64)
                for m in np.flatnonzero(berry):
                    k = int(p2p[side, m])
                    if 0 <= k < 6:by_slot_stat[k] = stat[side, m];by_slot_sign[k] = sign[side, m]
                ok = not (diff & ~slots).any() and staged(a, b, by_slot_stat, by_slot_sign)
            elif leaf in WH_SLOT_LEAVES or leaf.startswith(WH_VISIBLE):ok = not (diff & ~slots).any()
            else:continue
            if ok:explained.add((leaf, side))
    return entries - explained, sorted(explained, key=str)


def facade_actors(sides, doubled):
    """WH: the Facade involvement a lane keeps: its pre-state actor sides whose own Facade line ran in this tree, by the
    side's pre-step active (side and slot), while that user had par, brn, psn or tox (recorded at the move hook: the
    status Facade's onBasePower reads)."""
    return [s for s in sides if bool(doubled[s])]


def sacred_actors(sides, line, stopped, frozen_line):
    """WH: the Sacred Fire involvement a lane keeps: its pre-state frozen-user sides where this tree tied the divergence
    to that mon at its own move: line, the side's pre-step active was frozen at its record_move (BeforeMove let it act:
    it failed frz's 20% roll and the defrost flag let it pass) and is not frozen at its Sacred Fire line (frz's
    onModifyMove thawed it in useMove between the two: the mon that thaws is the frozen user); or stopped, recorded at
    the side's own BeforeMove: eligible, its pre-step active frozen, the executed move Sacred Fire, frz's roll failed and
    the user did not act (flinch or confusion stopped it after frz let it pass). A side
    whose Sacred Fire line ran with its user still frozen (frozen_line) never keeps it. The GPU pass at edfcba5d: the
    thaw planted on the foe (mutant S3) passed under the ALL allowance alone."""
    return [s for s in sides if (bool(line[s]) or bool(stopped[s])) and not bool(frozen_line[s])]


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
                 effect_rule=False, m2_ids=None, wh_ids=None):
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
            from gpu_engine import engine_device_public_hooks as hooks
            damage_hook, move_hook = hooks.damage, hooks.move;hits = [];opened = {}
            protect_hook, turn_end_hook, hp_line_hook = hooks.port_protect, hooks.turn_end, hooks.hp_line
            # M2: this tree's own evidence, recorded around the device hooks (never read from the engine's M2 writes).
            sides2 = j.arange(2)[None, :];zero2 = j.zeros((lanes, 2), j.bool_)
            m2e = dict(perish_used=zero2, yawn_by=zero2, yawn_target=j.full((lanes, 2), -1, j.int32), wrap_line=zero2,
                       wrap_by=zero2, wrap_user=j.full((lanes, 2), -1, j.int32), wrap_target=j.full((lanes, 2), -1, j.int32),
                       blocked=zero2, ordinary=None, tick_seen=zero2, tick_trapper=j.full((lanes, 2), -1, j.int32), tick_held=zero2,
                       tick_hp=j.full((lanes, 2), -1, j.int32))
            wrap_calls = []
            # WH: this tree's White Herb uses, read at the device hook from the before/after states it is given; each side's
            # own Facade line by the pre-step active under par/brn/psn/tox; and Sacred Fire's thaw: frozen at the side's
            # record_move (after BeforeMove, before useMove's ModifyMove) and thawed at its own line, a frozen user stopped
            # after frz let it pass, or a line run still frozen (wh_rule, facade_actors, sacred_actors).
            whe = dict(used=j.zeros((lanes, 2, 6), j.bool_), raise_=j.zeros((lanes, 2, 6, 7), j.int32), bad=j.zeros((lanes,), j.bool_),
                       facade=zero2, sacred_pre=zero2, sacred_line=zero2, sacred_stopped=zero2, sacred_frz=zero2,
                       trick_order=j.zeros((lanes,), j.bool_), credit_lost=zero2)
            from gpu_engine import action_history as act_history
            record_move_fn = act_history.record_move
            def record_move_recorder(st, side, slot, *, eligible, struggle=False):
                if wh_ids is not None:
                    # The side's chosen move slot, recorded once BeforeMove let it act: a Sacred Fire user still frozen
                    # here (it failed frz's 20% roll and the defrost flag let it pass) thaws in useMove just after.
                    s = j.broadcast_to(j.asarray(side, j.int32), (lanes,));r = j.arange(lanes);mon = st.active[r, s]
                    k = j.clip(j.broadcast_to(j.asarray(slot, j.int32), (lanes,)), 0, st.moves.shape[-1] - 1)
                    e = j.broadcast_to(j.asarray(eligible, j.bool_), (lanes,)) & ~j.broadcast_to(j.asarray(struggle, j.bool_), (lanes,))
                    frozen = e & (st.moves[r, s, mon, k] == wh_ids['sacredfire']) & (st.status[r, s, mon] == FRZ) & (mon == state.active[r, s])
                    whe['sacred_pre'] = whe['sacred_pre'] | at_side(frozen, s)
                return record_move_fn(st, side, slot, eligible=eligible, struggle=struggle)
            from gpu_engine import mechanics as mech
            before_move_fn, before_move_hook, trick_hook = mech.before_move, hooks.before_move, getattr(hooks, 'trick', None)
            defrost_seen = {}
            def before_move_fn_recorder(*a, **kw):
                # The step's BeforeMove result for the side whose device before_move hook follows; defrost_move is the
                # executed move's defrost flag (Sacred Fire's).
                defrost_seen['flag'] = kw.get('defrost_move', False)
                return before_move_fn(*a, **kw)
            def before_move_recorder(c, st, side, result, use, **kw):
                if wh_ids is not None:
                    # Sacred Fire "stopped" (review): at this side's BeforeMove, eligible, its pre-step active, frozen, the
                    # executed move Sacred Fire, frz's 20% roll failed (still frozen) and the user did not act (flinch or
                    # confusion stopped it after frz let it pass).
                    s = j.broadcast_to(j.asarray(side, j.int32), (lanes,));r = j.arange(lanes);mon = st.active[r, s]
                    d = j.broadcast_to(j.asarray(defrost_seen.get('flag', False), j.bool_), (lanes,))
                    u = j.broadcast_to(j.asarray(use, j.bool_), (lanes,))
                    stopped = (u & d & (mon == state.active[r, s]) & (st.status[r, s, mon] == FRZ) & (result.status == FRZ)
                               & ~j.broadcast_to(j.asarray(result.can_act, j.bool_), (lanes,)))
                    whe['sacred_stopped'] = whe['sacred_stopped'] | at_side(stopped, s)
                return before_move_hook(c, st, side, result, use, **kw)
            def trick_recorder(c, before, after, side, use, choice_items, **kw):
                if wh_ids is not None:
                    # White Herb received by Trick (review finding 1): setItem's Start runs inside Trick's onHit, before
                    # the -item lines, so at this hook a living receiver of White Herb with a negative stage has already
                    # used it here; one that has not marks the device order wrong (forced unexpected, like a bad use).
                    # `after` is the swapped state before any herb use (the trick hook's own -item operand).
                    u = j.broadcast_to(j.asarray(use, j.bool_), (lanes,));r = j.arange(lanes);due = []
                    for m_ in (0, 1):
                        mon = after.active[r, m_]
                        due.append(u & (after.item[r, m_, mon] == wh_ids['herb']) & (after.hp[r, m_, mon] > 0)
                                   & (after.boosts[r, m_, mon] < 0).any(-1))
                        whe['trick_order'] = whe['trick_order'] | (due[m_] & ~whe['used'][r, m_, mon])
                    due = j.stack(due, axis=1)                                                       # [L,2] by side
                    # The partner credits B2 loses when a receiver's herb lines end its Trick pairing, from this recorder's
                    # own due mask (never the engine's herb_used): the user's when the target used one, the target's when
                    # either did. herb_used disagreeing with it (or missing where a use is due) is bad too.
                    herb = kw.get('herb_used')
                    told = j.zeros((lanes, 2), j.bool_) if herb is None else j.broadcast_to(j.asarray(herb, j.bool_), (lanes, 2))
                    whe['trick_order'] = whe['trick_order'] | (u & (told != due).any(-1))
                    s = j.broadcast_to(j.asarray(side, j.int32), (lanes,));t = 1 - s
                    got = lambda m_: after.item[r, m_, after.active[r, m_]] > 0
                    lost_u = u & due[r, t] & got(s);lost_t = u & (due[r, t] | due[r, s]) & got(t)
                    whe['credit_lost'] = whe['credit_lost'] | at_side(lost_u, s) | at_side(lost_t, t)
                return trick_hook(c, before, after, side, use, choice_items, **kw)
            herb_hook = getattr(hooks, 'white_herb', None)
            def herb_recorder(c, before, after, side, use):
                out = herb_hook(c, before, after, side, use)
                s = j.broadcast_to(j.asarray(side, j.int32), (lanes,));u = j.broadcast_to(j.asarray(use, j.bool_), (lanes,))
                r = j.arange(lanes);holder = before.active[r, s]
                at = (j.arange(2)[None, :, None] == s[:, None, None]) & (j.arange(6)[None, None, :] == holder[:, None, None])
                moved = (before.item != after.item) | (before.boosts != after.boosts).any(-1)            # [L,2,6]
                rest = j.ones((lanes,), j.bool_)
                for name in before._fields:
                    if name in ('item', 'boosts'):continue
                    x, y = getattr(before, name), getattr(after, name)
                    rest = rest & (x == y).reshape(lanes, -1).all(-1)
                b_boost, a_boost = before.boosts[r, s, holder], after.boosts[r, s, holder]
                slot = j.clip(out.trackers.physical_to_public[r, s, holder], 0, 5)
                t = out.trackers
                produced = ((t.shown_item[r, s, slot] == wh_ids['herb']) & (t.item_removed[r, s, slot] != 0)
                            if getattr(t, 'shown_item', None) is not None and getattr(t, 'item_removed', None) is not None
                            else j.zeros((lanes,), j.bool_))
                valid = (u & (before.item[r, s, holder] == wh_ids['herb']) & (after.item[r, s, holder] == 0) & (before.hp[r, s, holder] > 0)
                         & (b_boost < 0).any(-1) & (a_boost == j.maximum(b_boost, 0)).all(-1) & ~(moved & ~at).any((1, 2)) & rest & produced)
                whe['bad'] = whe['bad'] | (u & ~valid) | (~u & (moved.any((1, 2)) | ~rest))
                whe['used'] = whe['used'] | (at & valid[:, None, None])
                whe['raise_'] = whe['raise_'] + j.where((at & valid[:, None, None])[..., None], after.boosts - before.boosts, 0)
                return out
            def at_side(mask, s):
                return (s[:, None] == sides2) & mask[:, None]
            def foe_sub(st, s):
                r = j.arange(lanes);foe = 1 - s
                return st.substitute_hp[r, foe, st.active[r, foe]]
            def move_recorder(c, st, side, *rest, **kw):
                # The foe's active Substitute when this side's move line opens (a Substitute made or remade earlier in
                # the step included); the damage hook that follows reads it again after the hit.
                s = j.broadcast_to(j.asarray(side, j.int32), (lanes,))
                opened.update(side=s, sub=foe_sub(st, s))
                if wh_ids is not None and wh_ids.get('facade') is not None and len(rest) >= 2:
                    # WH: this side's own Facade line with use by its pre-step active (side and slot), and that user's
                    # status at the line (Facade's onBasePower reads it).
                    mv = j.broadcast_to(j.asarray(rest[0], j.int32), (lanes,));use = j.broadcast_to(j.asarray(rest[1], j.bool_), (lanes,))
                    r = j.arange(lanes);mon = st.active[r, s];status = st.status[r, s, mon];own = mon == state.active[r, s]
                    doubled = use & own & (mv == wh_ids['facade']) & (status >= 1) & (status <= 4)
                    whe['facade'] = whe['facade'] | at_side(doubled, s)
                    # Sacred Fire: the pre-step active, frozen at this side's record_move, runs its line thawed (frz's
                    # onModifyMove, between the two); or a line with use still frozen (never in a correct tree). A
                    # frozen user stopped after frz let it pass is recorded at BeforeMove (before_move_recorder).
                    sf = mv == wh_ids['sacredfire']
                    whe['sacred_line'] = whe['sacred_line'] | at_side(use & sf & own & (status == 0) & whe['sacred_pre'][r, s], s)
                    whe['sacred_frz'] = whe['sacred_frz'] | at_side(use & sf & (status == FRZ), s)
                if m2_ids is not None and len(rest) >= 2:
                    # M2: a move line (the caller's, or a called child's) of Perish Song, Yawn or Wrap with use set. Yawn: the
                    # target at the line holds no status and no Substitute and lives (its onTryHit and TryPrimaryHit, read
                    # from this tree's state at the line). Wrap: the user's and the target's party indices at the line.
                    mv = j.broadcast_to(j.asarray(rest[0], j.int32), (lanes,));use = j.broadcast_to(j.asarray(rest[1], j.bool_), (lanes,))
                    r = j.arange(lanes);foe = 1 - s;target = st.active[r, foe]
                    clear = (st.status[r, foe, target] == 0) & (st.substitute_hp[r, foe, target] == 0) & (st.hp[r, foe, target] > 0)
                    m2e['perish_used'] = m2e['perish_used'] | at_side(use & (mv == m2_ids.get('perishsong', -9)), s)
                    yawn = use & (mv == m2_ids.get('yawn', -9)) & clear
                    m2e['yawn_by'] = m2e['yawn_by'] | at_side(yawn, s)
                    m2e['yawn_target'] = j.where(at_side(yawn, s), target[:, None], m2e['yawn_target'])
                    wrap = use & (mv == m2_ids.get('wrap', -9))
                    m2e['wrap_line'] = m2e['wrap_line'] | at_side(wrap, s)
                    m2e['wrap_user'] = j.where(at_side(wrap, s), st.active[r, s][:, None], m2e['wrap_user'])
                    m2e['wrap_target'] = j.where(at_side(wrap, s), target[:, None], m2e['wrap_target'])
                return move_hook(c, st, side, *rest, **kw)
            def damage_recorder(c, st, side, dealt, taken, use):
                # M1 review: this tree's hit record (recorded around the device hooks, never read from the engine's own
                # recharge write), the recharge rule's evidence that the move connected: HP damage the device records,
                # or the foe's Substitute lower than when this side's move line opened.
                s = j.broadcast_to(j.asarray(side, j.int32), (lanes,));use = j.broadcast_to(j.asarray(use, j.bool_), (lanes,))
                struck = use & (j.broadcast_to(j.asarray(taken), (lanes,)) > 0)
                # M2: Wrap's evidence is HP damage alone (a Substitute hit traps nothing), after this side's Wrap line.
                m2e['wrap_by'] = m2e['wrap_by'] | (at_side(struck, s) & m2e['wrap_line'])
                if opened:struck = struck | (use & (opened['side'] == s) & (foe_sub(st, s) < opened['sub']))
                hits.append((s, struck))
                return damage_hook(c, st, side, dealt, taken, use)
            def protect_recorder(c, side, acts, accuracy_pass, protected, protectable):
                # M2: this side's move passed accuracy and met the foe's Protect (Yawn has the protect flag).
                s = j.broadcast_to(j.asarray(side, j.int32), (lanes,))
                met = j.broadcast_to(j.asarray(acts) & j.asarray(accuracy_pass) & j.asarray(protected) & j.asarray(protectable), (lanes,))
                m2e['blocked'] = m2e['blocked'] | at_side(met, s)
                return protect_hook(c, side, acts, accuracy_pass, protected, protectable)
            def turn_end_recorder(c, before, after, ordinary, **kw):
                m2e['ordinary'] = j.broadcast_to(j.asarray(ordinary, j.bool_), (lanes,))
                return turn_end_hook(c, before, after, ordinary, **kw)
            def hp_line_recorder(c, st, side, use, *rest, **kw):
                # M2 review: the trapper at the holder's own partiallytrapped handler. Transition emits this HP-line cutpoint
                # right after the Wrap handler at each residual position a Wrap event can take (residual_order.POSITIONS
                # ['wrap'], R1), its use mask being the handler's tick; the handler
                # ran for a holder where the caller's _wrap_holder is set. Recorded there from this tree's state (the trapper
                # is the other side's active then; the holder's tick does not touch it): its index, alive, activeTurns > 0.
                caller = sys._getframe(1).f_locals
                if m2_ids is not None and use is caller.get('_wrap_tick') and '_wrap_holder' in caller:
                    wrap_calls.append(True)
                    s = j.broadcast_to(j.asarray(side, j.int32), (lanes,));r = j.arange(lanes);f = 1 - s
                    ran_here = at_side(j.broadcast_to(caller['_wrap_holder'], (lanes,)), s)
                    trapper = st.active[r, f];held = (st.hp[r, f, trapper] > 0) & (st.active_turns[r, f] > 0)
                    m2e['tick_seen'] = m2e['tick_seen'] | ran_here
                    m2e['tick_trapper'] = j.where(ran_here, trapper[:, None], m2e['tick_trapper'])
                    m2e['tick_held'] = j.where(ran_here, held[:, None], m2e['tick_held'])
                    # The holder's HP right after its handler (a later heal in the same residual, e.g. the drain of a Leech
                    # Seed the holder planted on a slower trapper, may move it again before the step ends).
                    m2e['tick_hp'] = j.where(ran_here, st.hp[r, s, st.active[r, s]][:, None], m2e['tick_hp'])
                return hp_line_hook(c, st, side, use, *rest, **kw)
            hooks.damage, hooks.move = damage_recorder, move_recorder
            hooks.port_protect, hooks.turn_end, hooks.hp_line = protect_recorder, turn_end_recorder, hp_line_recorder
            if wh_ids is not None and herb_hook is not None:hooks.white_herb = herb_recorder
            if wh_ids is not None:
                act_history.record_move = record_move_recorder
                mech.before_move, hooks.before_move = before_move_fn_recorder, before_move_recorder
                if trick_hook is not None:hooks.trick = trick_recorder
            try:
                if record_installs:
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
                climate.residual_clock = clock;hooks.damage, hooks.move = damage_hook, move_hook
                hooks.port_protect, hooks.turn_end, hooks.hp_line = protect_hook, turn_end_hook, hp_line_hook
                if herb_hook is not None:hooks.white_herb = herb_hook
                act_history.record_move = record_move_fn
                mech.before_move, hooks.before_move = before_move_fn, before_move_hook
                if trick_hook is not None:hooks.trick = trick_hook
            if not wrap_calls:
                for key in ('tick_seen', 'tick_trapper', 'tick_held', 'tick_hp'):m2e[key] = None   # no handler cutpoint: no evidence
            named_in = name_inputs(state, actions, trackers);named_out = name_outputs(out)
            names['out'], digests = digest_named(named_out, lanes)
            names['side_axis'] = [has_side_axis(n, named_out[n].shape, lanes) for n in names['out']]
            new = [n for n in sorted(named_out) if base_names is not None and n not in base_names and n in named_in]
            names['pre'] = new
            pre = digest_named({n: named_in[n] for n in new}, lanes)[1] if new else j.zeros((lanes, 0, 2, 2), j.uint32)
            effect_ok = None
            # Computed whenever the tree has the leaf: the effect rule (base lacks it) and the KO cascade rule read it.
            if effect_rule or hasattr(state, 'effect_draws'):
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
            # M1 rule arrays: this tree's post values the recharge and ledger rules read (absent leaves: None).
            m1 = {}
            if 'state.recharge_counter' in named_out:m1['counter'] = named_out['state.recharge_counter']
            from gpu_engine.public_active import NAMES as bag_names
            bag = named_out.get('trackers.public_volatiles')
            if bag is not None and MUSTRECHARGE in bag_names and bag.ndim == 4 and bag.shape[-1] == len(bag_names):
                m1['bag'] = bag[..., bag_names.index(MUSTRECHARGE)]
            hit = j.zeros((lanes, 2), j.bool_)
            for s, struck in hits:hit = hit | ((s[:, None] == j.arange(2)[None, :]) & struck[:, None])
            m1['hit'] = hit
            m1['post'] = {n: named_out[n] for n in M1_POST_LEAVES if n in named_out}
            for key, leaf in (('mutated', LEDGER_LEAVES[0]), ('original', LEDGER_LEAVES[1]), ('knocked', 'state.item_knocked')):
                if leaf in named_out:m1[key] = named_out[leaf]
            m1['post_hp'] = out.step.state.hp
            # M2 rule arrays: this tree's post counters, bag columns, wrap clock and ability, the hook evidence, and the
            # Wrap duration draw the step used (the design derivation: draws[s][7] of sample_chance_inputs on these keys,
            # or the saved block on a resumed turn).
            m2 = None
            if m2_ids is not None and hasattr(out.step.state, 'perish_counter'):
                from gpu_engine.transition import sample_chance_inputs
                post = out.step.state;live = ~state.done & ~out.step.invalid_actions
                fresh = sample_chance_inputs(state).draws
                draws = j.where((live & state.pending_residual)[:, None, None], state.residual_draws, fresh)
                m2 = dict(perish=post.perish_counter, yawn=post.yawn_counter, trap=post.partial_trap_counter,
                          source=post.partial_trap_source, ability=post.ability, draws7=draws[:, :, 7], active_turns=post.active_turns,
                          clock=named_out.get(WRAP_CLOCK), **{k: v for k, v in m2e.items() if v is not None})
                if bag is not None and bag.ndim == 4 and bag.shape[-1] == len(bag_names):
                    m2['bag'] = j.stack([bag[..., bag_names.index(n)] for n in M2_BAG], axis=-1)
            return dict(digests=digests, pre_digests=pre, fainted=(state.hp > 0) & (out.step.state.hp <= 0), m1=m1,
                extra={n: named_out[n] for n in faint_leaves if n in named_out}, calls=calls,
                ko={n: named_out[n] for n in KO_MON_LEAVES + KO_LINK_LEAVES if n in named_out},
                post_active=out.step.state.active, post_done=out.step.state.done,
                physical_to_public=out.trackers.physical_to_public, active_slot=out.trackers.active_slot,
                public_order=named_out.get(PUBLIC_ORDER), residual_ran=ran[0] if len(ran) == 1 else None,
                effect_ok=effect_ok, wit={n: named_out[n] for n in witness_names(named_out)}, m2=m2,
                wh=whe if wh_ids is not None else None)
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


def dirty_paths():
    """Uncommitted paths under gpu_engine/ and src/ (a base pass run with a newer harness records exactly which)."""
    return sorted(line[3:] for line in git('status', '--porcelain', '--', 'gpu_engine', 'src').stdout.splitlines() if line.strip())


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
    names = side_axis = None;lane_steps = 0;live = 0;begin = time.perf_counter();residual_recorded = False;witness = []
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
        # M1 review: the per-lane witness arrays the M1 rules compare per mon and per bag column.
        witness = sorted(result['wit'])
        for i, n in enumerate(witness):record[f'wit_{i:02d}'] = np.asarray(result['wit'][n])[:real]
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
        residual_ran_recorded=residual_recorded, post_active_recorded=True, witness_leaves=witness,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), dirty_paths=dirty_paths(),
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
    color_change = [vocab['abilities'].index('colorchange')] if 'colorchange' in vocab.get('abilities', ()) else []
    base_names = manifest['output_names']
    # The replay rule is for the PR that adds public_order (its base pass lacks the leaf); once
    # the base pass has it too (a migrated identity pre), the leaf is compared bitwise.
    public_order_rule = PUBLIC_ORDER in new_inputs and PUBLIC_ORDER not in base_names
    # Fidelity R: likewise the saved effect block is checked against the design's derivation while the base lacks it.
    effect_rule = EFFECT_DRAWS in new_inputs and EFFECT_DRAWS not in base_names
    m2_rules = 'M2' in plugins
    m2_ids = {n: vocab['moves'].index(n) for n in ('perishsong', 'yawn', 'wrap') if n in vocab['moves']} if m2_rules else None
    # WH: White Herb's post-state rule and recorded uses, and Facade's recorded line status.
    wh_rules = 'WH' in plugins
    wh_ids = dict(herb=vocab['items'].index('whiteherb') if 'whiteherb' in vocab.get('items', ()) else -9,
                  facade=vocab['moves'].index('facade') if 'facade' in vocab['moves'] else None,
                  sacredfire=vocab['moves'].index('sacredfire') if 'sacredfire' in vocab['moves'] else -9) if wh_rules else None
    stepper = Stepper(step_override or _engine_step(tables), treedef, base_names=set(base_names), record_installs=public_order_rule,
        effect_rule=effect_rule, m2_ids=m2_ids, wh_ids=wh_ids)
    shards = [p for p in shard_list(args, capture) if p.name in set(manifest['shards'])]
    if args.limit == 0 and len(shards) != len(manifest['shards']):raise ValueError('shard set differs from the base pass')
    usable = jax.jit(state_module.usable_moves)
    callable_moves = np.asarray(tables.engine_tables.sleep_talk.callable)
    destiny_bond = getattr(state_module, 'DESTINY_BOND', DEFAULT_DESTINY_BOND)
    counts = dict(lane_steps=0, done_skipped=0, unchanged=0, changed_and_involved=0, changed_but_uninvolved=0,
                  changed_involved_unexpected_fields=0, ko_cascade=0, phase_shift=0, terminal=0, terminal_residual=0, waived=0,
                  rule_explained=0, faint_rule_leaves=0, clear_volatile_leaves=0, effect_draws_saved=0,
                  recharge_leaves=0, bag_leaves=0, trick_witness_leaves=0, ledger_leaves=0, m2_leaves=0, m2_wrong_mon=0, m2_forced=0,
                  wh_leaves=0, wh_bad=0, wh_trick_order=0, wh_dropped=0, facade_dropped=0, sacredfire_dropped=0,
                  sacredfire_stopped=0, pinch_leaves=0)
    clear_volatile = 'B' in plugins
    m1_rules = 'M1' in plugins
    # M1 review: the base's per-lane witness arrays (wit_XX); the M1 rules compare per mon and per bag column.
    witness_leaves = manifest.get('witness_leaves')
    if (m1_rules or m2_rules or wh_rules) and witness_leaves is None:
        raise SystemExit('base pass predates the M1 witness arrays (wit_XX); re-run the base pass with this harness')
    if wh_rules and not set(WH_MON_LEAVES) <= set(witness_leaves):
        raise SystemExit('base pass predates the WH witness arrays (state.item, state.item_knocked, state.boosts); re-run it with this harness')
    ability_names = vocab.get('abilities', [])
    soundproof = {ability_names.index(n) for n in ('soundproof',) if n in ability_names}
    sleep_immune = {ability_names.index(n) for n in ('insomnia', 'vitalspirit') if n in ability_names}
    # M2 review: the base's per-side state.hp digest, which the Wrap tick witness reproduces (column in base_names order).
    hp_base = base_names.index('state.hp') if 'state.hp' in base_names else None
    # A base pass run on the base engine with a newer harness is dirty in the harness only; anything else is recorded.
    if manifest.get('source_dirty') and set(manifest.get('dirty_paths') or ['?']) != {'gpu_engine/v2e_regress.py'}:
        warnings.append(f"base pass ran on a dirty tree beyond the harness: {manifest.get('dirty_paths')}")
    if manifest.get('harness_sha256') and manifest['harness_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        warnings.append('the base pass ran with another harness version (its witness arrays may differ in kind)')
    item_names = vocab.get('items', [])
    lum_id = item_names.index(LUM_BERRY) if LUM_BERRY in item_names else None
    pinch_ids = {item_names.index(n): stat for n, stat in PINCH_STATS.items() if n in item_names}
    from gpu_engine.public_active import NAMES as bag_columns
    columns = {n: bag_columns.index(n) for n in (MUSTRECHARGE, 'substitute', 'confusion', *M2_BAG) if n in bag_columns}
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
            base_wit = {n: base[f'wit_{i:02d}'] for i, n in enumerate(witness_leaves or ())}
            wit = {n: np.asarray(v)[:real] for n, v in result['wit'].items()}
            sub_pre = np.asarray(named['state.substitute_hp'])[:real] if 'state.substitute_hp' in named else None
            pre_active = np.asarray(named['state.active'])[:real];post_active = np.asarray(result['post_active'])[:real]
            m1 = result.get('m1') or {}
            knocked_pre = np.asarray(named['state.item_knocked'])[:real] if 'state.item_knocked' in named else None
            item_pre = np.asarray(named['state.item'])[:real]
            holders = np.isin(np.asarray(named['state.ability'])[:real], color_change)       # [L,2,6] Color Change
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
                # WH: White Herb stays involved only where this tree recorded a valid use (and no malformed call); Facade only
                # for a side whose own Facade line ran under par/brn/psn/tox (post-state witnesses, never the predicate alone).
                wh_lane = None
                if wh_rules and result.get('wh') is not None:
                    wh_lane = {k: np.asarray(v)[lane] for k, v in result['wh'].items()}
                    if bool(wh_lane.get('trick_order', False)):
                        # A Tricked-in White Herb the device saw used after Trick's -item lines: bad, like a malformed use.
                        wh_lane['bad'] = True;counts['wh_trick_order'] += bool(changed)
                    if 'whiteherb' in effects and (bool(wh_lane['bad']) or not wh_lane['used'].any()):
                        effects.pop('whiteherb');counts['wh_dropped'] += bool(changed)
                    if 'facade' in effects:
                        kept = facade_actors(effects['facade'], wh_lane['facade'])
                        if kept:effects['facade'] = kept
                        else:effects.pop('facade');counts['facade_dropped'] += bool(changed)
                    if 'sacredfire' in effects and 'sacred_line' in wh_lane:
                        kept = sacred_actors(effects['sacredfire'], wh_lane['sacred_line'], wh_lane['sacred_stopped'],
                                             wh_lane['sacred_frz'])
                        if kept:
                            effects['sacredfire'] = kept
                            counts['sacredfire_stopped'] += bool(changed) and not any(bool(wh_lane['sacred_line'][s]) for s in kept)
                        else:effects.pop('sacredfire');counts['sacredfire_dropped'] += bool(changed)
                # The faint-clear rule explains its own leaves; any other changed leaf on the
                # same lane-step still needs an involved effect (e.g. a revert on a faint lane).
                faint_explained = []
                if changed and args.faint_clear and lane in faint_set:
                    leaves = [n for n in changed if n in faint_base]
                    entries, faint_explained = faint_rule(entries, {n: result['extra'][n][lane] for n in leaves},
                        {n: faint_base[n][lane] for n in leaves}, result['fainted'][lane],
                        result['physical_to_public'][lane], result['active_slot'][lane])
                    counts['faint_rule_leaves'] += bool(faint_explained)
                # B: the faint's clearVolatile of the fainted mon's Choice lock and confusion (CLEAR_VOLATILE_LEAVES).
                clear_explained = []
                if changed and clear_volatile:
                    leaves = [n for n in CLEAR_VOLATILE_LEAVES if n in result['ko'] and n in base_ko]
                    entries, clear_explained = clear_volatile_rule(entries, {n: result['ko'][n][lane] for n in leaves},
                        {n: base_ko[n][lane] for n in leaves}, result['fainted'][lane])
                    counts['clear_volatile_leaves'] += bool(clear_explained)
                # M1: Hyper Beam's counter (recharge_rule), the bag per cell (bag_rule), a Trick's berry and Substitute
                # consequences (trick_rule) and Knock Off's item ledger (ledger_rule); all post-state, per mon.
                recharge_explained, bag_explained, trick_explained, ledger_explained, m2_explained = [], [], [], [], []
                m2_must = set()
                if changed and (m1_rules or m2_rules):
                    zero2 = np.zeros(2, bool);p2p = result['physical_to_public'][lane]
                    lane_wit = {n: v[lane] for n, v in wit.items()};lane_base_wit = {n: v[lane] for n, v in base_wit.items()}
                    lane_ko = {n: v[lane] for n, v in result['ko'].items()};lane_base_ko = {n: v[lane] for n, v in base_ko.items()}
                    bag = lane_wit.get(BAG)
                    allowed = np.zeros(np.asarray(bag).shape, bool) if bag is not None else None
                if changed and m2_rules and result.get('m2') is not None:
                    # M2: Perish Song, Yawn and Wrap from this tree's post state and hook evidence (m2_rule).
                    m2 = result['m2']
                    w = {k: np.asarray(v)[lane] for k, v in m2.items() if v is not None}
                    w.update(p2p=p2p, post_active=post_active[lane], post_hp=np.asarray(m1['post_hp'])[lane],
                             done=bool(np.asarray(result['post_done'])[lane]),
                             ran=None if result['residual_ran'] is None else bool(np.asarray(result['residual_ran'])[lane]),
                             soundproof=soundproof, sleep_immune=sleep_immune, max_hp=np.asarray(named['state.max_hp'])[lane],
                             base_hp=None if hp_base is None else np.asarray(base['digests'][lane, hp_base]))
                    if 'ordinary' in w:w['ordinary'] = bool(w['ordinary'])
                    entries, m2_explained, m2_cells, m2_must = m2_rule(entries, w)
                    if allowed is not None:
                        for i, name in enumerate(M2_BAG):
                            if name in columns:allowed[:, :, columns[name]] |= m2_cells[:, :, i]
                    counts['m2_leaves'] += bool(m2_explained)
                if changed and m1_rules:
                    departed = departed_mask(pre_active[lane], post_active[lane],
                        None if base_post_active is None else base_post_active[lane])
                    if 'counter' in m1 and 'bag' in m1:
                        # Evidence the side's move connected: its recorded direct damage, or the foe's Substitute falling
                        # while that foe stayed in (nothing else lowers a Substitute within the step).
                        evidence = np.asarray(m1['hit'][lane], bool).copy()
                        if sub_pre is not None and 'state.substitute_hp' in lane_wit:
                            for side in (0, 1):
                                f = 1 - side;a = int(pre_active[lane][f])
                                evidence[side] |= (a == int(post_active[lane][f]) and 0 <= a < 6
                                    and int(lane_wit['state.substitute_hp'][f, a]) < int(sub_pre[lane][f, a]))
                        ran = None if result['residual_ran'] is None else bool(np.asarray(result['residual_ran'])[lane])
                        entries, recharge_explained, cells = recharge_rule(entries, m1['counter'][lane], m1['bag'][lane],
                            involved['hyperbeam'][lane] if 'hyperbeam' in involved else zero2, evidence, ran, post_active[lane],
                            m1['post_hp'][lane], p2p, departed)
                        if allowed is not None and MUSTRECHARGE in columns:allowed[:, :, columns[MUSTRECHARGE]] |= cells
                        counts['recharge_leaves'] += bool(recharge_explained)
                    trick_lane = bool(involved['trick'][lane].any()) if 'trick' in involved else False
                    if trick_lane and 'state.item' in m1['post'] and 'state.status' in lane_ko and 'state.confusion_counter' in lane_ko:
                        post_item = np.asarray(m1['post']['state.item'][lane])
                        w = item_witnesses(pre_active[lane], post_active[lane], item_pre[lane], post_item, lane_ko['state.status'],
                            lane_ko['state.confusion_counter'], None if sub_pre is None else sub_pre[lane],
                            lane_wit.get('state.substitute_hp'), lane_base_wit.get('state.substitute_hp'), lum_id, pinch_ids)
                        entries, trick_explained, confused = trick_rule(entries, w, lane_ko, lane_base_ko, lane_wit, lane_base_wit,
                            p2p, item_pre[lane], post_item, pinch_ids)
                        counts['trick_witness_leaves'] += bool(trick_explained)
                        if allowed is not None:
                            # A swap moves only these bag cells, each equal to this tree's own state at that mon.
                            for side in (0, 1):
                                for m in range(6):
                                    slot = int(p2p[side, m])
                                    if not 0 <= slot < 6:continue
                                    if 'substitute' in columns and w['sub'][side, m]:
                                        c = columns['substitute']
                                        allowed[side, slot, c] = bool(bag[side, slot, c]) == (int(lane_wit['state.substitute_hp'][side, m]) > 0)
                                    if 'confusion' in columns and confused[side, m]:
                                        c = columns['confusion']
                                        allowed[side, slot, c] = bool(bag[side, slot, c]) == (int(lane_ko['state.confusion_counter'][side, m]) > 0)
                    if 'mutated' in m1 and 'original' in m1 and 'knocked' in m1 and knocked_pre is not None:
                        entries, ledger_explained = ledger_rule(entries, m1['mutated'][lane], m1['original'][lane], knocked_pre[lane],
                            m1['knocked'][lane], item_pre[lane], result['physical_to_public'][lane], trick=trick_lane)
                        counts['ledger_leaves'] += bool(ledger_explained)
                if changed and (m1_rules or m2_rules):
                    entries, bag_explained = bag_rule(entries, bag, lane_base_wit.get(BAG), allowed)
                    counts['bag_leaves'] += bool(bag_explained)
                wh_explained = []
                if changed and wh_lane is not None:
                    # WH: the holder's item and stages and the device's consumption, per mon and per slot (wh_rule).
                    w = dict(used=wh_lane['used'], bad=bool(wh_lane['bad']), herb=wh_ids['herb'],
                             p2p=result['physical_to_public'][lane], ko=ko_mons_all[lane],
                             acting=active_mask(pre_active[lane]) | active_mask(post_active[lane]),
                             wit={n: v[lane] for n, v in wit.items()}, base={n: v[lane] for n, v in base_wit.items()})
                    w['raise'] = wh_lane['raise_'];w['credit'] = wh_lane.get('credit_lost', np.zeros(2, bool))
                    entries, wh_explained = wh_rule(entries, w)
                    counts['wh_leaves'] += bool(wh_explained)
                    if 'whiteherb' in effects or 'facade' in effects:
                        # A pinch berry eaten in one tree only, the changed damage having moved its holder across 25%.
                        w.update(pre_item=item_pre[lane], pinch=pinch_ids)
                        entries, pinch_explained = pinch_rule(entries, w)
                        wh_explained = wh_explained + pinch_explained;counts['pinch_leaves'] += bool(pinch_explained)
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
                    # The saved effect block follows the deferral: accepted per side where this tree's block is the derivation.
                    effect_ok = None if result['effect_ok'] is None else np.asarray(result['effect_ok'])[lane]
                    # Only a Color Change holder whose KO moved (this tree's faint this step, or a faint differing from the
                    # base) may differ in type, per mon (state) and per public slot (shown) against the base's arrays.
                    moved_holders = holders[lane] & (ko_mons | np.asarray(result['fainted'][lane], bool))
                    def retype_ok(leaf, side, lane=lane, moved=moved_holders):
                        if leaf not in wit or leaf not in base_wit:return False
                        diff = _differ(wit[leaf][lane][side], base_wit[leaf][lane][side])
                        mons = moved[side] if leaf == 'state.types' else _slots(moved[side], result['physical_to_public'][lane][side])
                        return bool(diff.any()) and not (diff & ~mons).any()
                    cascade_info = dict(ko_sides=ko_mons.any(-1), mon_ok=mon_ok, families=families, effect_ok=effect_ok,
                                        retype_ok=retype_ok)
                unexpected, waived, by_ko = lane_verdict(entries, effects, side_axis, cascade=cascade_info, waive=args.waive or ()) \
                    if changed_rest and effects else ([], [], [])
                m2_wrong = []
                if changed_rest and m2_rules and any(e in effects for e in M2_EFFECTS):
                    # M2: an M2 effect's per-mon leaves may differ only at that side's pre- or post-step active.
                    post_m = {'state.hp': np.asarray(m1['post_hp'])[lane]};ref_m = {'state.hp': np.asarray(named['state.hp'])[lane]}
                    for n in ('state.status', 'state.status_counter'):
                        if n in result['ko'] and n in base_ko:post_m[n] = result['ko'][n][lane];ref_m[n] = base_ko[n][lane]
                    for n, key in zip(M2_COUNTER_LEAVES, ('perish', 'yawn', 'trap', 'source')):
                        if result.get('m2') is not None and result['m2'].get(key) is not None and n in named:
                            post_m[n] = np.asarray(result['m2'][key])[lane];ref_m[n] = np.asarray(named[n])[lane]
                    m2_wrong = m2_wrong_mon(entries, post_m, ref_m, pre_active[lane], post_active[lane])
                    unexpected = unexpected + [e for e in m2_wrong if e not in unexpected]
                    counts['m2_wrong_mon'] += bool(m2_wrong)
                # M2 review: the struck side's HP and Substitute HP that m2_rule's witness did not explain are unexpected
                # (never an allowance or a KO cascade).
                m2_forced = [e for e in sorted(m2_must, key=str) if e in entries]
                unexpected = unexpected + [e for e in m2_forced if e not in unexpected]
                counts['m2_forced'] += bool(m2_forced)
                # WH: a malformed White Herb call (use without a valid restore, or a change without use) is unexpected.
                if changed and wh_lane is not None and bool(wh_lane['bad']):
                    unexpected = unexpected + [('wh_evidence', None)];rule_ok = False;counts['wh_bad'] += 1
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
                    clear_volatile_rule=[f'{l}@{side}' for l, side in clear_explained],
                    recharge_rule=[f'{l}@{side}' for l, side in recharge_explained],
                    bag_rule=[f'{l}@{side}' for l, side in bag_explained],
                    trick_rule=[f"{l}@{'glob' if side is None else side}" for l, side in trick_explained],
                    ledger_rule=[f'{l}@{side}' for l, side in ledger_explained],
                    m2_rule=[f'{l}@{side}' for l, side in m2_explained],
                    m2_wrong_mon=[f'{l}@{side}' for l, side in m2_wrong],
                    m2_forced=[f'{l}@{side}' for l, side in m2_forced],
                    wh_rule=[f'{l}@{side}' for l, side in wh_explained],
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
        faint_clear_rule=bool(args.faint_clear), clear_volatile_rule=clear_volatile, m1_rules=m1_rules, m2_rules=m2_rules, wh_rules=wh_rules,
        witness_leaves=witness_leaves, base_source_dirty=manifest.get('source_dirty'), base_dirty_paths=manifest.get('dirty_paths'),
        base_harness_sha256=manifest.get('harness_sha256'), harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        waive=list(args.waive or ()), counts=counts, by_effect=by_effect,
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
