"""Owner-only engine facts for the existing partial private-team v4 writer.

This is NOT an observation/rollout adapter. Public events, own action requests,
public overlays and all B4 runtime gates remain separate. In particular an
opponent's physical party order never becomes public encounter order here.

The caller supplies the verified constructor's engine-metadata binding and an
explicit, hash-bound request exposure contract. BattleState alone cannot prove
which ability or gender the source request exposes. No exposure is inferred
from a sampled opponent, nor from the existence of a physical value.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import NamedTuple
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from pokezero.category_vocab import normalize_category_value
from pokezero.dex import normalize_id
from . import private_team
from .state import BattleState

SCHEMA = 'pokezero.engine_private_team.v1'


@dataclass(frozen=True)
class RequestExposure:
    """Required source-request facts, never guessed from physical state.

    current/base means the request exposes that engine ability; hidden means
    neither is exposed. physical gender preserves -1 unknown/0 genderless as
    no sex label; absent means the request has no sex field at all. HP uses
    the canonical exact fraction while alive and literal ``0 fnt`` on faint.
    """
    ability: str
    gender: str
    stats: bool
    item: bool

    def __post_init__(self):
        if (self.ability not in ('current', 'base', 'hidden')
                or self.gender not in ('physical', 'absent')
                or type(self.stats) is not bool or type(self.item) is not bool):
            raise ValueError('explicit typed request exposure required')


class Tables(NamedTuple):
    """Trusted device primitive arguments; copied binding tags do not seal data."""
    engine_binding: object
    binding: object
    vocabulary_binding: object
    species_categories: object
    base_stats: object
    type_categories: object
    ability_categories: object
    item_categories: object
    status_categories: object
    role: object
    ability_mode: object
    gender_visible: object
    stats_visible: object
    item_visible: object


@dataclass(frozen=True, eq=False)
class Compiled:
    """Host handle, admitted by compiler identity and retained immutable edges.

    Constructing/copying this dataclass does not admit a new handle. Device
    Tables are exposed for inspection, not as an alternate checked API.
    """
    device: Tables

    def __getattr__(self, name):
        if name in Tables._fields:
            return getattr(self.device, name)
        raise AttributeError(name)


_COMPILED = weakref.WeakKeyDictionary()


def _checked_tables(compiled):
    record = _COMPILED.get(compiled) if type(compiled) is Compiled else None
    if (record is None or compiled.device is not record[0]
            or any(current is not original for current, original in zip(compiled.device, record[1]))):
        raise ValueError('original compiled holder and immutable array edges required')
    return compiled.device


def _words(value):
    return np.frombuffer(hashlib.sha256(value).digest(), dtype='>u4').astype(np.uint32)


def engine_metadata_binding(metadata):
    """Bind the exact static metadata snapshot supplied by the constructor."""
    return jnp.asarray(_words(json.dumps(metadata, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()))


def compile_engine_private_team(metadata, *, vocabulary, dex,
                                checkpoint_sha256, exposure):
    """Freeze engine-ID -> source label/dex mappings; OOV never hashes silently."""
    if (type(checkpoint_sha256) is not str or len(checkpoint_sha256) != 64
            or any(c not in '0123456789abcdef' for c in checkpoint_sha256)
            or type(exposure) is not RequestExposure):
        raise ValueError('exact checkpoint SHA256 and RequestExposure required')
    engine = engine_metadata_binding(metadata)
    def names(kind):
        rows = metadata['vocabulary'][kind]
        if (type(rows) is not list or not rows or rows[0] not in ('', '_none')
                or any(type(x) is not str or not normalize_id(x) for x in rows[1:])
                or len({normalize_id(x) for x in rows[1:]}) != len(rows)-1):
            raise ValueError('ordered engine namespace with explicit zero required: '+kind)
        return rows
    species, abilities, items = (names(x) for x in ('species', 'abilities', 'items'))
    def label(value):
        if not value:
            return 0
        if not vocabulary.is_enumerated(value):
            raise ValueError('unmapped checkpoint label: '+normalize_category_value(value))
        return vocabulary.encode(value)
    bases, types, categories = [], [], []
    for index, name in enumerate(species):
        # Engine IDs erase punctuation. Only the exact dex row proves the
        # request spelling; a base-form fallback cannot invent a wire forme.
        info = dex.species_info(name) if index else None
        if index and info is None:
            raise ValueError('engine species outside source dex: '+name)
        if index and (normalize_id(info.id) != normalize_id(name)
                or normalize_id(info.name) != normalize_id(name)):
            raise ValueError('source dex identity differs from engine species: '+name)
        bases.append([info.base_stats.get(k, 0) if info else 0 for k in private_team.STATS])
        types.append([label('type:'+info.types[i]) if info and i < len(info.types) else 0
            for i in (0, 1)])
        categories.append(label('species:'+info.name) if index else 0)
    base = np.asarray(bases, np.float32)
    if not np.all(np.isfinite(base)) or np.any(base < 0):
        raise ValueError('source base stats must be finite nonnegative float32')
    ability = [label('belief:possible_ability:'+normalize_id(x)) if i else 0
        for i, x in enumerate(abilities)]
    item = [label('belief:possible_item:'+normalize_id(x)) if i else 0
        for i, x in enumerate(items)]
    # Engine mechanics order differs from public_mon_state's status order.
    statuses = [label('status:'+x) for x in ('none', 'par', 'brn', 'psn', 'tox', 'slp', 'frz')]
    vb = private_team.vocabulary_binding(vocabulary)
    role = label('pokemon:self')
    payload = dict(schema=SCHEMA, checkpoint=checkpoint_sha256, exposure=asdict(exposure),
        engine=np.asarray(engine).tolist(), vocabulary=np.asarray(vb).tolist(),
        species_categories=categories, base_stats=base.tolist(), type_categories=types,
        ability_categories=ability, item_categories=item, status_categories=statuses, role=role)
    binding = _words(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
    tables = Tables(engine, jnp.asarray(binding), vb, jnp.asarray(categories, jnp.int32),
        jnp.asarray(base), jnp.asarray(types, jnp.int32), jnp.asarray(ability, jnp.int32),
        jnp.asarray(item, jnp.int32), jnp.asarray(statuses, jnp.int32), jnp.asarray(role, jnp.int32),
        jnp.asarray(('current', 'base', 'hidden').index(exposure.ability), jnp.int32),
        jnp.asarray(exposure.gender == 'physical'), jnp.asarray(exposure.stats), jnp.asarray(exposure.item))
    compiled = Compiled(tables)
    _COMPILED[compiled] = (tables, tuple(tables))
    return compiled


class OwnedTeam(NamedTuple):
    owner_side: object
    engine_binding: object
    species: object
    level: object
    gender: object
    hp: object
    max_hp: object
    status: object
    active: object
    stats: object
    item: object
    ability: object
    base_ability: object


def _require(value, shape, dtype, name):
    if getattr(value, 'shape', None) != shape or getattr(value, 'dtype', None) != jnp.dtype(dtype):
        raise ValueError('exact shape/dtype required: '+name)


def select_owned_team(state, owner_sides, *, engine_binding):
    """Gather the owner FIRST, before checking/reading any private contents.

    Shape/type checks inspect no data. Invalid owner rows gather safe zero-seat
    storage but are explicitly refused by team_facts, with no emitted facts.
    The binding must come from verified construction, not a guessed namespace.
    """
    if type(state) is not BattleState:
        raise ValueError('engine BattleState required')
    batch = state.species.shape[0]
    if not batch:
        raise ValueError('nonempty engine batch required')
    _require(owner_sides, (batch,), jnp.int32, 'owner sides')
    _require(engine_binding, (batch, 8), jnp.uint32, 'constructor engine binding')
    rows = jnp.arange(batch)
    owners = jnp.clip(owner_sides, 0, 1)
    values = []
    for field in OwnedTeam._fields[2:]:
        value = getattr(state, field)
        tail = (6, 6) if field == 'stats' else () if field == 'active' else (6,)
        _require(value, (batch, 2, *tail), jnp.int32, field)
        values.append(value[rows, owners])
    return OwnedTeam(owner_sides, engine_binding, *values)


class Result(NamedTuple):
    facts: private_team.TeamFacts
    admitted: object


def _host_call(owned):
    if type(owned) is not OwnedTeam:
        raise ValueError('typed owned team required')
    # Probe through a public JAX primitive too: a zero-argument outer JIT can
    # close over concrete owned arrays, so checking arguments alone misses it.
    if (isinstance(owned.owner_side, jax.core.Tracer)
            or isinstance(jnp.zeros((), jnp.int32), jax.core.Tracer)):
        raise ValueError('checked host boundary cannot be traced; use prepare_team_facts')


def team_facts(owned, compiled, *, expected_binding):
    """Checked host boundary for the existing partial own-team input.

    A binding word alone never authenticates replaced device arrays. For JIT,
    use prepare_team_facts, whose host guard runs before every cached call.
    """
    _host_call(owned)
    return _team_facts_device(owned, _checked_tables(compiled), expected_binding=expected_binding)


def prepare_team_facts(compiled, *, expected_binding):
    """Prepare a closed-over kernel with a mandatory guard on EVERY host call.

    Do not JIT this wrapper again: that would cache away its host guard. The
    wrapper refuses tracing; its enclosed device primitive is already jitted.
    """
    tables = _checked_tables(compiled)
    _require(expected_binding, (8,), jnp.uint32, 'expected adapter binding')
    if not np.array_equal(np.asarray(expected_binding), np.asarray(tables.binding)):
        raise ValueError('prepared kernel requires the pinned adapter binding')
    fast = jax.jit(lambda owned: _team_facts_device(owned, tables,
        expected_binding=expected_binding))
    def checked(owned):
        _host_call(owned)
        _checked_tables(compiled)
        return fast(owned)
    return checked


def _team_facts_device(owned, tables, *, expected_binding):
    """TRUSTED primitive: caller must retain the authenticated table closure.

    This function's word comparison is not content authentication. It must not
    receive independently supplied/replaced tables, even with copied tags.

    Failed rows carry owner=-1 and zero facts, so even a caller accidentally
    ignoring Result.admitted cannot admit them through project_team.
    """
    if type(owned) is not OwnedTeam or type(tables) is not Tables:
        raise ValueError('typed owned team and compiled tables required')
    batch = owned.owner_side.shape[0]
    _require(owned.owner_side, (batch,), jnp.int32, 'owned side')
    _require(owned.engine_binding, (batch, 8), jnp.uint32, 'owned engine binding')
    _require(expected_binding, (8,), jnp.uint32, 'expected adapter binding')
    for field in OwnedTeam._fields[2:]:
        tail = (6, 6) if field == 'stats' else () if field == 'active' else (6,)
        _require(getattr(owned, field), (batch, *tail), jnp.int32, field)
    present = owned.species > 0
    active = jnp.arange(6) == owned.active[:, None]
    ability = jnp.where(tables.ability_mode == 0, owned.ability, owned.base_ability)
    ability = jnp.where(tables.ability_mode == 2, 0, ability)
    item = jnp.where(tables.item_visible, owned.item, 0)
    ok = (owned.owner_side >= 0) & (owned.owner_side <= 1)
    ok &= jnp.all(owned.engine_binding == tables.engine_binding, axis=-1)
    ok &= jnp.all(tables.binding == expected_binding)
    ok &= jnp.all((owned.species >= 0) & (owned.species < tables.species_categories.shape[0]), axis=-1)
    ok &= ~jnp.any(present[:, 1:] & ~present[:, :-1], axis=-1)
    ok &= ~jnp.any(present, axis=-1) | ((owned.active >= 0) & (owned.active < 6) & jnp.any(present & active, axis=-1))
    valid = ((owned.level >= 1) & (owned.level <= 100) & (owned.hp >= 0)
        & (owned.max_hp > 0) & (owned.hp <= owned.max_hp)
        & (owned.status >= 0) & (owned.status < 7)
        & ((ability >= 0) & (ability < tables.ability_categories.shape[0]))
        & ((item >= 0) & (item < tables.item_categories.shape[0])))
    valid &= (tables.ability_mode == 2) | (ability > 0)
    valid &= ~tables.gender_visible | ((owned.gender >= -1) & (owned.gender <= 2))
    valid &= ~tables.stats_visible | jnp.all(owned.stats >= 0, axis=-1)
    ok &= jnp.all(~present | valid, axis=-1)
    visible = present & ok[:, None]
    species = jnp.clip(owned.species, 0, tables.species_categories.shape[0]-1)
    fainted = owned.hp == 0
    # _condition_features('0 fnt') has fainted=True but status='none'.
    status = jnp.where(fainted, 0, jnp.clip(owned.status, 0, 6))
    ability = jnp.clip(ability, 0, tables.ability_categories.shape[0]-1)
    item = jnp.clip(item, 0, tables.item_categories.shape[0]-1)
    categories = jnp.stack((tables.species_categories[species], tables.status_categories[status],
        jnp.broadcast_to(tables.role, present.shape), *[tables.type_categories[species, i] for i in (0, 1)],
        tables.ability_categories[ability], tables.item_categories[item]), axis=-1)
    # Source literal 0 fnt has no denominator and consequently no actual HP.
    fraction = ~fainted & present
    denominator = jnp.where(fraction, owned.max_hp, 1).astype(jnp.float32)
    actual = jnp.where(tables.stats_visible, owned.stats, 0).astype(jnp.float32)
    # Even absent stats maps still derive max HP from an alive fraction.
    actual = actual.at[..., 0].set(jnp.where(fraction, owned.max_hp, 0))
    def mask(value):
        shape = visible.shape + (1,) * (value.ndim-2)
        return jnp.where(visible.reshape(shape), value, jnp.zeros_like(value))
    facts = private_team.TeamFacts(jnp.where(ok, owned.owner_side, -1),
        jnp.broadcast_to(tables.vocabulary_binding, (batch, 8)), visible,
        mask(active), mask(fainted), mask(owned.level),
        mask(jnp.where(tables.gender_visible & (owned.gender > 0), owned.gender, 0)),
        mask(jnp.where(fraction, owned.hp, 0).astype(jnp.float32)),
        jnp.where(visible, denominator, 1), mask(tables.base_stats[species]), mask(actual),
        mask(categories), mask(ability > 0), mask(item > 0))
    return Result(facts, ok)
