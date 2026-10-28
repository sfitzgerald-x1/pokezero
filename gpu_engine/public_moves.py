"""Device move-reveal/PP and active last-move consumers for public inputs.

Running Pressure and Transform flags must come from public ability/Transform
consumers, never the sampled opponent's physical state. The raw protocol adapter
and those consumers remain required before production encoder admission. This
ports reveal/PP/last-used fields only: move-triggered choice/item inference,
sleep clocks, team cures, damage attribution and tendencies remain separate.
"""
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokezero.belief import _CALLER_MOVES, _NEVER_PRESSURED_POOL_MOVES, _called_move_source
from pokezero.category_vocab import normalize_category_value
from pokezero.dex import normalize_id
from pokezero import showdown as reference

from . import public_belief, public_mon_state


SCHEMA = "pokezero.public_moves.v1"
NONE, SLEEP_TALK, RANDOM_CALLER, LOCKED_MOVE, OTHER_FROM = range(5)
LAST_UNKNOWN, LAST_SWITCH, LAST_MOVE = range(3)
CAPACITY = 32


class MoveTables(NamedTuple):
    predicate: jax.Array            # [M], set-source predicate namespace
    max_pp: jax.Array               # [M], catalog rule floor(base PP*8/5)
    foe_targeted: jax.Array         # source's Pressure rule, not wire target
    struggle: jax.Array
    last_category: jax.Array        # exact checkpoint vocabulary rows
    switch_category: jax.Array      # ordinary / Baton Pass sentinel
    source_binding: jax.Array       # [8] source/predicate contract
    binding: jax.Array              # [8] complete move/vocabulary contract


@dataclass(frozen=True)
class MoveCatalog:
    names: tuple[str, ...]
    ids: Mapping[str, int]
    device: MoveTables

    def move_id(self, name: str) -> int:
        key = normalize_id(name)
        if not key or key not in self.ids:
            raise ValueError("move is outside the frozen public move namespace")
        return self.ids[key]


def lexical_move_tags(raw_line: str) -> tuple[int, bool]:
    """Translate source lexical tags, keeping two deliberately distinct facts.

    Python's belief caller parser scans for '[from]' anywhere, while its current
    last-move parser tests startswith on trailing pipe fields. Do not collapse
    those tests: leading whitespace can distinguish them on an off-script line.
    This maps wire strings only; the device decides reveal/charge eligibility.
    """
    if not isinstance(raw_line, str):
        raise ValueError("public move input must be a protocol string")
    parts = raw_line.split("|")
    if len(parts) < 4 or parts[:2] != ["", "move"]:
        raise ValueError("expected a move protocol line")
    caller = _called_move_source(raw_line)
    code = (NONE if caller is None else SLEEP_TALK if caller == "sleeptalk" else
            RANDOM_CALLER if caller in _CALLER_MOVES else
            LOCKED_MOVE if caller == "lockedmove" else OTHER_FROM)
    return code, any(part.startswith("[from]") for part in parts[4:])


def _static_category(vocabulary, label):
    """Pure vocabulary lookup; compiling unused OOV rows is not observed drift.

    Reuse the vocabulary's resolved alias index and exact source hash formula.
    Runtime encode() remains the authority in the independent parity tests.
    """
    key = normalize_category_value(label)
    if not key:
        return 0
    row = vocabulary._index.get(key)
    if row is not None:
        return row
    value = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")
    return 1 + len(vocabulary.tokens) + value % vocabulary.oov_buckets


def compile_moves(source, dex, vocabulary) -> MoveCatalog:
    """Freeze one move-ID namespace, derived source predicates and policy rows."""
    names = ("",) + tuple(sorted(({normalize_id(name) for name in dex.moves} |
        {normalize_id(m) for rows in source.variants for variant in rows for m in variant.moves} |
        {"struggle", "sleeptalk", "snore", "metronome", "mirrormove", "assist", "naturepower", "copycat"}) - {""}))
    ids = {name: index for index, name in enumerate(names) if index}
    predicate, max_pp, foe, struggle, categories = [], [], [], [], []
    for name in names:
        info = dex.move_info(name)
        predicate.append(source.predicate_id("move", name) if name else 0)
        max_pp.append(info.max_pp if info is not None else 0)
        foe.append(bool(name) and name not in _NEVER_PRESSURED_POOL_MOVES)
        struggle.append(name == "struggle")
        categories.append(_static_category(vocabulary, f"move:{name}") if name else 0)
    switch = [_static_category(vocabulary, label) for label in (
        reference.LAST_USED_MOVE_SWITCH_SENTINEL, reference.LAST_USED_MOVE_BATON_PASS_SENTINEL)]
    contract = dict(schema=SCHEMA, names=names, predicate=predicate, max_pp=max_pp,
        foe_targeted=foe, struggle=struggle, last_category=categories, switch_category=switch,
        source_binding=np.asarray(source.device.binding).tolist(),
        caller_moves=sorted(_CALLER_MOVES), never_pressured=sorted(_NEVER_PRESSURED_POOL_MOVES),
        vocabulary_tokens=vocabulary.tokens, oov_buckets=vocabulary.oov_buckets,
        vocabulary_aliases=dict(vocabulary.aliases), capacity=CAPACITY)
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).digest()
    binding = np.frombuffer(digest, dtype=">u4").astype(np.uint32)
    return MoveCatalog(names, MappingProxyType(ids), MoveTables(
        jnp.asarray(predicate, jnp.int32), jnp.asarray(max_pp, jnp.int32),
        jnp.asarray(foe, jnp.bool_), jnp.asarray(struggle, jnp.bool_),
        jnp.asarray(categories, jnp.int32), jnp.asarray(switch, jnp.int32),
        source.device.binding, jnp.asarray(binding)))


class MoveState(NamedTuple):
    binding: jax.Array              # [B,8], immutable across continuations
    move_ids: jax.Array             # [B,2,6,32], first-reveal order, 0 empty
    uses: jax.Array                 # same shape, persistent real-set PP charges
    last_kind: jax.Array            # [B,2], current active's last-used state
    last_id: jax.Array              # [B,2], move namespace, not vocabulary rows


class MoveUpdate(NamedTuple):
    state: MoveState
    beliefs: public_belief.BeliefState
    admitted: jax.Array
    revealed: jax.Array
    charge: jax.Array
    doubled_despite_wire: jax.Array


def initialize(batch: int, tables: MoveTables) -> MoveState:
    if type(batch) is not int or batch <= 0:
        raise ValueError("positive public move batch required")
    return MoveState(jnp.broadcast_to(tables.binding, (batch, 8)),
        jnp.zeros((batch, 2, 6, CAPACITY), jnp.int32),
        jnp.zeros((batch, 2, 6, CAPACITY), jnp.int32),
        jnp.zeros((batch, 2), jnp.int32), jnp.zeros((batch, 2), jnp.int32))


def _bool(value):
    result = jnp.asarray(value)
    if result.dtype != jnp.bool_:
        raise ValueError("public move flags must be boolean without coercion")
    return result


def arrive(state: MoveState, tables: MoveTables, *, side, valid=True):
    """Reset only the active last-move sentinel; real-set PP survives switching."""
    side = public_mon_state._int32(side)
    rows = jnp.arange(state.last_kind.shape[0])
    seat = jnp.clip(side, 0, 1)
    admitted = (side >= 0) & (side < 2) & jnp.all(state.binding == tables.binding, axis=-1) & _bool(valid)
    return state._replace(last_kind=public_mon_state._write(state.last_kind, (rows, seat), LAST_SWITCH, admitted),
        last_id=public_mon_state._write(state.last_id, (rows, seat), 0, admitted)), admitted


def record_move(state: MoveState, beliefs: public_belief.BeliefState, mons: public_mon_state.MonState,
                tables: MoveTables, source_tables: public_belief.DeviceTables, *,
                side, slot, encounter, move_id, caller=NONE, has_from=False,
                transformed=False, opposing_pressure=False, wire_target_side=-1, valid=True) -> MoveUpdate:
    """Apply source reveal/charge rules with public context supplied by its consumers.

    A full sparse ledger or int32 counter overflow is an explicit refusal, never
    a truncated reveal or a wrapped PP count. The event-fold caller must reject
    the whole chunk on refusal. No transformed/called move can become a real-set
    reveal except Sleep Talk's own-set callee, exactly as in Python.
    """
    p = public_mon_state
    rows, seat, target, address_ok = p._address(mons, side, slot)
    encounter, move_id, caller, wire_target_side = (p._int32(x) for x in
        (encounter, move_id, caller, wire_target_side))
    has_from, transformed, opposing_pressure, valid = (_bool(x) for x in
        (has_from, transformed, opposing_pressure, valid))
    known_move = (move_id > 0) & (move_id < tables.predicate.shape[0])
    safe_move = jnp.clip(move_id, 0, tables.predicate.shape[0] - 1)
    context_ok = (encounter > 0) & (mons.encounter[rows, seat, target] == encounter) & (
        mons.active[rows, seat] == target) & ~mons.fainted[rows, seat, target]
    source_ok = jnp.all(tables.source_binding == source_tables.binding) & jnp.all(
        state.binding == tables.binding, axis=-1) & (
        beliefs.species[rows, 1 - seat, target] == mons.source_species[rows, seat, target])
    reveal = ~transformed & ~tables.struggle[safe_move] & (caller != RANDOM_CALLER) & (caller != LOCKED_MOVE)
    charged = reveal & (caller != SLEEP_TALK)
    charge = jnp.where(charged, 1 + (opposing_pressure & tables.foe_targeted[safe_move]).astype(jnp.int32), 0)
    address = rows, seat, target
    ids = state.move_ids[address]
    seen = jnp.any(ids == move_id[..., None], axis=-1)
    empty = jnp.any(ids == 0, axis=-1)
    index = jnp.argmax(jnp.where(seen[..., None], ids == move_id[..., None], ids == 0), axis=-1)
    old_uses = state.uses[address][rows, index]
    room = ~reveal | seen | empty
    fits_counter = ~charged | (old_uses <= jnp.iinfo(jnp.int32).max - charge)
    admitted = address_ok & context_ok & source_ok & known_move & (caller >= NONE) & (caller <= OTHER_FROM) & (
        wire_target_side >= -1) & (wire_target_side <= 1) & room & fits_counter & valid
    reveal &= admitted
    charge = jnp.where(admitted, charge, 0)
    full_address = rows, seat, target, index
    next_state = state._replace(
        move_ids=p._write(state.move_ids, full_address, move_id, reveal),
        uses=p._write(state.uses, full_address, old_uses + charge, reveal),
        last_kind=p._write(state.last_kind, (rows, seat), LAST_MOVE, admitted & ~has_from),
        last_id=p._write(state.last_id, (rows, seat), move_id, admitted & ~has_from))
    # A move by side s narrows the other observer's belief about that encounter.
    slots = (jnp.arange(2)[None, :, None] == (1 - seat)[:, None, None]) & (
        jnp.arange(6)[None, None, :] == target[:, None, None])
    next_beliefs, _ = public_belief.add_evidence(beliefs, source_tables,
        jnp.broadcast_to(tables.predicate[safe_move], (rows.shape[0],))[:, None, None],
        valid=slots & reveal[:, None, None])
    return MoveUpdate(next_state, next_beliefs, admitted, reveal, charge,
        admitted & (charge == 2) & (wire_target_side != 1 - seat))


def bucket_pp(state: MoveState, tables: MoveTables, bucket_move_ids):
    """Return fraction, revealed-validity and per-battle contract admission.

    Padding ID0 is allowed. Malformed IDs or a changed catalog refuse the whole
    battle's projection rather than silently reinterpret the persisted ledger.
    """
    ids = public_mon_state._int32(bucket_move_ids)
    if ids.ndim != 4 or ids.shape[:3] != state.move_ids.shape[:3]:
        raise ValueError("move PP buckets require B×2×6×K move IDs")
    admitted = jnp.all(state.binding == tables.binding, axis=-1) & jnp.all(
        (ids >= 0) & (ids < tables.max_pp.shape[0]), axis=tuple(range(1, ids.ndim)))
    in_range = (ids > 0) & (ids < tables.max_pp.shape[0])
    matches = (state.move_ids[..., None, :] == ids[..., :, None]) & (state.move_ids[..., None, :] > 0)
    seen = jnp.any(matches, axis=-1) & in_range
    uses = jnp.sum(jnp.where(matches, state.uses[..., None, :], 0), axis=-1)
    maximum = tables.max_pp[jnp.clip(ids, 0, tables.max_pp.shape[0] - 1)]
    fraction = jnp.where(seen & (maximum > 0), jnp.maximum(0, maximum - uses) / jnp.maximum(maximum, 1), 0)
    mask = admitted.reshape((admitted.shape[0],) + (1,) * (ids.ndim - 1))
    return jnp.where(mask, fraction, 0).astype(jnp.float32), seen & mask, admitted


def last_move_categories(state: MoveState, tables: MoveTables, arrived_by_baton):
    """Project move versus ordinary/Baton switch sentinel through actual vocab."""
    baton = jnp.broadcast_to(_bool(arrived_by_baton), state.last_kind.shape)
    good_id = (state.last_id > 0) & (state.last_id < tables.last_category.shape[0])
    valid = (state.last_kind >= LAST_UNKNOWN) & (state.last_kind <= LAST_MOVE) & jnp.where(
        state.last_kind == LAST_MOVE, good_id, state.last_id == 0)
    value = jnp.where(state.last_kind == LAST_SWITCH, tables.switch_category[baton.astype(jnp.int32)],
        jnp.where(state.last_kind == LAST_MOVE,
                  tables.last_category[jnp.clip(state.last_id, 0, tables.last_category.shape[0] - 1)], 0))
    bound = jnp.all(state.binding == tables.binding, axis=-1) & jnp.all(valid, axis=-1)
    return jnp.where(bound[:, None], value, 0), bound
