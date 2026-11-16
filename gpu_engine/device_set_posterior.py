"""set_posterior's strike likelihood on the device: the producer of in-tree damage-evidence narrowing (option A).

A port of the host stage-1 set_posterior (pokezero.set_posterior at 034e979c, frozen in the narrowing scope's section 3b)
for one public damage event per row: an opponent's move hit the observer's own active, whose exact HP and stats the
observer knows. For every variant of the attacker's source list (at most 64, source order, the same bits as the public
belief's hard mask) it evaluates the engine's own 16 damage rolls and counts the rolls that fit the observed loss (a KO is
censored: rolls >= the HP before the hit), then reduces the per-variant result over the subject's hard support with
set_posterior's refusal order and returns the source-order survivor bits that narrow_pin intersects into the standing pin.

The damage arithmetic is the engine's: mechanics.gen3_damage over the 16 rolls, with the modifiers built by the engine's
own helpers (items.attack_modifier / defense_modifier, combat.attack_modifier / base_power with tier D's pinch abilities,
weather.effective / damage_modifier, solar_beam.base_power, variable_moves.resolve / facade_power, absorption's Flash
Fire phase 1, Explosion's Defense halving). Three Gen 3 modifiers the engine does not model are added exactly as tier2
applies them, so the kernel matches set_posterior on Showdown strikes: type-boost items (Silk Scarf, Linoone's Setup
Sweeper item; mods/gen3/items.ts), Marvel Scale and Plus/Minus (mods/gen3/abilities.ts:115-134 reads getAllActive, the
opposing active in singles). In-tree (guard=True) a strike where one of them is active for a hard variant is refused,
so the engine's own damage never excludes its true set.

What only public facts can decide follows tier2's public rules, precompiled per shown HP percent: a pinch ability is on
at <= 1/3 and ambiguous within the 0.04 band of 1/3 (a refused strike), Flail / Reversal take their power from the
percent and are ambiguous when +-1% straddles a breakpoint, Eruption / Water Spout below full HP are refused.

Deliberate departures from 034e979c, both counted by the parity harness:
- a hard variant whose Hidden Power type cannot affect the defender gets count 0 (set_posterior v2) instead of refusing
  the whole strike as core-unsupported:hiddenpower (v1);
- Guts boosts Atk only (data/abilities.ts guts: onModifyAtk; the engine's combat.attack_modifier), where tier2 also
  multiplies a special move's SpA.

No PRNG: the kernel reads the crit the engine already drew and the HP it already lost; it adds no chance coordinate and
reads no effect stream. Tables are compiled once on the host (compile_tables) and hang off engine_device_belief.Tables
only when set_posterior_narrowing is on, so the flag-off programs are unchanged.
"""
from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as j

from pokezero.dex import normalize_id, resolve_move_base_power
from pokezero.investment import FIXED_EXACT_DAMAGE_MOVES, UNMODELED_DAMAGE_MOVES
from . import absorption as absorb
from . import combat as traits
from . import damage_moves as special_hits
from . import items as held
from . import mechanics as m
from . import pursuit as intercepts
from . import self_destruct as exploding
from . import solar_beam as charging
from . import variable_moves as variable
from . import weather as climate

SCHEMA = 'pokezero.device_set_posterior.v1'
ROLLS = 16
LANES = 64
WORDS = 2
HAPPINESS = 255   # Gen 3 random sets carry no happiness; the Pokemon constructor's default (Return 102)

# set_posterior's refused move classes (034e979c; the parity harness asserts equality when set_posterior is importable).
FIXED_DAMAGE_MOVES = frozenset(FIXED_EXACT_DAMAGE_MOVES) | frozenset(UNMODELED_DAMAGE_MOVES) | {'bide'}
OHKO_MOVES = frozenset({'fissure', 'horndrill', 'guillotine', 'sheercold'})
CONDITIONAL_POWER_MOVES = frozenset({'weatherball', 'magnitude', 'lowkick', 'smellingsalt', 'spitup', 'present', 'beatup',
    'triplekick', 'rollout', 'iceball', 'furycutter', 'futuresight', 'doomdesire', 'frustration', 'stomp', 'rage'})
MULTI_HIT_MOVES = frozenset({'doublekick', 'bonemerang', 'twineedle', 'doubleslap', 'cometpunch', 'furyattack', 'furyswipes',
    'pinmissile', 'spikecannon', 'barrage', 'bulletseed', 'iciclespear', 'rockblast', 'armthrust', 'triplekick', 'beatup'})
FULL_HP_MOVES = frozenset({'eruption', 'waterspout'})

# Refusal codes in set_posterior's order (parse_strikes/_refusal, then the single-variant skip, then _counts), then the
# device-only guard. 0 admits. A caller passes its protocol/state refusals as a bitmask over these codes (1 << code);
# the kernel adds the static move classes and the first set bit names the strike, as _refusal's first match does.
REASONS = ('', 'called', 'struggle', 'fixed-damage', 'ohko', 'conditional-power', 'multi-hit', 'hit-substitute',
    'endure-or-focusband', 'transform', 'type-change', 'item-changed', 'ability-overridden', 'defender-semi-invulnerable',
    'unmodelled:charge', 'unmodelled:mud/water-sport', 'unvalidated:screen', 'no-hp-drop', 'single-variant', 'unknown-defender', 'no-support',
    'core-unsupported', 'ambiguous-band', 'effectiveness-disagreement',
    'engine-unmodeled:typeboost', 'engine-unmodeled:marvelscale', 'engine-unmodeled:plusminus')
CODE = {name: i for i, name in enumerate(REASONS)}
PRECONDITIONS = frozenset(REASONS[1:CODE['single-variant']])   # what a caller may pass (protocol / engine-state facts)


def bit(name):
    """The precondition bit of a refusal name (a conditional-power:<move> name maps to conditional-power)."""
    name = 'conditional-power' if name.startswith('conditional-power') else name
    if name not in PRECONDITIONS:raise ValueError('not a caller refusal: ' + name)
    return 1 << CODE[name]


class PosteriorTables(NamedTuple):
    # Per source species (public_belief.compile_source order; row 0 is the unknown species).
    count: object            # [S] int32 variants
    species_types: object    # [S,2] int32 engine type ids of the species (STAB, as tier2's dex species types)
    engine_species: object   # [S] int32 engine species id (Thick Club / Light Ball / Soul Dew species rules)
    # Per source variant, source order, padded to 64.
    valid: object            # [S,64] bool
    participant: object      # [S,64] bool, set_posterior's _variant_participant built (fixture and base stats exist)
    level: object            # [S,64] int32
    stats: object            # [S,64,6] int32 HP, Atk, Def, SpA, SpD, Spe (_computed_stats of the generator spread)
    ability: object          # [S,64] int32 engine ability id (0 unknown)
    item: object             # [S,64] int32 engine item id (0 none or unknown)
    moves: object            # [S,64,4] int32 move keys (move_key namespace; -1 outside the engine vocabulary)
    hp_type: object          # [S,64] int32 engine type id of the variant's Hidden Power (0 none)
    hp_power: object         # [S,64] int32 its base power
    # Per engine move.
    move_key: object         # [M] int32 the move's identity for set membership (every Hidden Power alias: one key)
    power: object            # [M] int32 metadata base power
    type: object             # [M] int32
    physical: object         # [M] bool
    damaging: object         # [M] bool, category Physical or Special
    move_class: object       # [M] int32 bitmask of the static refused classes
    full_hp: object          # [M] bool, Eruption / Water Spout
    revenge: object          # [M] bool, Revenge (conditional power after a hit this turn: the caller's bit)
    # Per engine ability / item.
    partner: object          # [A] int32 the Plus/Minus partner ability id (0 none)
    marvel_scale: object     # [A] bool
    boost_type: object       # [I] int32 type-boost item's type (0 none)
    boost_num: object        # [I] int32 (11, or 21 for Sea Incense)
    boost_den: object        # [I] int32
    # Per shown HP percent 0..100 (tier2's float rules on x/100, evaluated once on the host).
    pinch_on: object         # [101] bool (<= 1/3, or inside the band: tier2's conservative interpretation)
    pinch_ambiguous: object  # [101] bool
    flail_power: object      # [101] int32
    flail_ambiguous: object  # [101] bool
    # The engine's own sub-tables and type chart (from the same metadata).
    type_num: object
    type_den: object
    items: object
    combat: object
    weather: object
    absorption: object
    variable: object
    solar: object
    explosion: object        # self_destruct.SelfDestructTable
    pursuit: object          # pursuit.PursuitTable
    binding: object          # [8] uint32 sha256 of the schema, the source hash and every array above


class StrikeInputs(NamedTuple):
    """One public damage event per row; every field [B] unless noted. Engine id namespaces."""
    species: object             # source species id of the attacker (0: unknown, nothing narrows)
    move: object                # engine move id
    crit: object                # bool
    weather: object             # raw engine weather code (suppression is per variant)
    pursuit: object             # bool, a Pursuit hitting a switching target (base power 80)
    attacker_status: object     # engine status code
    attacker_boosts: object     # [B,7] Atk, Def, SpA, SpD, Spe, Acc, Eva
    attacker_percent: object    # shown HP percent 0..100
    attacker_flash_fire: object # bool
    defender_stats: object      # [B,6] the observer's own active: HP (max), Atk, Def, SpA, SpD, Spe
    defender_boosts: object     # [B,7]
    defender_types: object      # [B,2]
    defender_ability: object
    defender_item: object
    defender_species: object    # engine species id
    defender_status: object
    hp_before: object
    observed: object            # exact HP lost
    ko: object                  # bool
    effectiveness: object       # the printed line: -1 resisted, 0 none, 1 supereffective
    precondition: object        # int32 bitmask of caller refusals (bit(name))


class StrikeResult(NamedTuple):
    rolls: object        # [B,64,16] int32 the engine's rolls per variant (0 when immune or powerless)
    counts: object       # [B,64] int32 rolls that fit
    unsupported: object  # [B,64] bool: no participant, the move is not in the set, or no damage channel
    ambiguous: object    # [B,64] bool: pinch band or Flail / Reversal breakpoint
    immune: object       # [B,64] bool: the variant's resolved type cannot affect the defender (count 0)
    effdis: object       # [B,64] bool: rolls fit but the variant's own effectiveness line differs from the printed one
    typeboost: object    # [B,64] bool: an engine-unmodeled type-boost item is active
    marvel: object       # [B,64] bool: the defender's Marvel Scale is active
    plusminus: object    # [B,64] bool: Plus / Minus is active
    move_bits: object    # [B] int32 the static move-class refusals (incl. Eruption / Water Spout below full HP)


class Reduced(NamedTuple):
    reason: object       # [B] int32 code (0 admitted)
    fit: object          # [B,2] uint32 source-order survivor bits (fit & hard; zero unless admitted)


# --------------------------------------------------------------------------- host compile


def _vocab_ids(names):
    return {normalize_id(name): i for i, name in enumerate(names) if i}


def _percent_tables(flail):
    """tier2's public HP rules on the shown percent x (fraction x/100, as showdown._condition_features reads 'x/100')."""
    from pokezero.tier2 import Tier2Config
    band = Tier2Config().pinch_ambiguity_band
    on, ambiguous, power, straddle = [], [], [], []
    for x in range(101):
        fraction = max(0.0, min(1.0, float(x) / float(100)))
        amb = abs(3.0 * fraction - 1.0) <= band     # tier2._strike_damage, verbatim
        ambiguous.append(amb)
        on.append(amb or fraction <= 1.0 / 3.0)
        if flail is None:
            power.append(0);straddle.append(False);continue
        power.append(resolve_move_base_power(flail, fraction))
        low = resolve_move_base_power(flail, max(0.0, fraction - 0.01))
        high = resolve_move_base_power(flail, min(1.0, fraction + 0.01))
        straddle.append(low != high)
    return dict(pinch_on=np.asarray(on, bool), pinch_ambiguous=np.asarray(ambiguous, bool),
                flail_power=np.asarray(power, np.int32), flail_ambiguous=np.asarray(straddle, bool))


def _hidden_power_iv_power(ivs):
    """Gen 3 Hidden Power base power from the IVs (sim/dex.ts getHiddenPower, gen < 6)."""
    total = sum((((ivs[stat] >> 1) & 1) << n) for n, stat in enumerate(('hp', 'atk', 'def', 'spe', 'spa', 'spd')))
    return total * 40 // 63 + 30


def compile_tables(source, dex, metadata, *, source_tables=None):
    """Freeze the variant participants and rule tables once (host, no JAX tracing).

    source: the bound Gen3RandbatSource set_posterior reads; species order and variant order are public_belief's (the
    belief mask's bits). Each participant is set_posterior._variant_participant's recipe, call for call:
    determinization._fixture_from_variant_payload -> _base_stats_for_species -> _computed_stats."""
    from pokezero import determinization as worlds
    from pokezero import tier2
    from . import public_belief
    tables = public_belief.compile_source(source) if source_tables is None else source_tables
    vocab = metadata['vocabulary']
    moves_v, types_v, abilities_v, items_v, species_v = (vocab[k] for k in ('moves', 'types', 'abilities', 'items', 'species'))
    move_ids, type_ids, ability_ids, item_ids, species_ids = map(_vocab_ids, (moves_v, types_v, abilities_v, items_v, species_v))
    hp_key = move_key_table(metadata)[1]

    def key(move):
        move = tier2.canonical_move_id(move)
        if move.startswith('hiddenpower'):return hp_key
        return move_ids.get(move, -1)

    names = tables.species_names
    s_count = len(names)
    count = np.zeros(s_count, np.int32);species_types = np.zeros((s_count, 2), np.int32);engine_species = np.zeros(s_count, np.int32)
    valid = np.zeros((s_count, LANES), bool);participant = np.zeros((s_count, LANES), bool)
    level = np.zeros((s_count, LANES), np.int32);stats = np.zeros((s_count, LANES, 6), np.int32)
    ability = np.zeros((s_count, LANES), np.int32);item = np.zeros((s_count, LANES), np.int32)
    moves = np.full((s_count, LANES, 4), -1, np.int32);hp_type = np.zeros((s_count, LANES), np.int32)
    hp_power = np.zeros((s_count, LANES), np.int32)
    unknown = dict(ability=set(), item=set(), move=set(), type=set())
    for sid, variants in enumerate(tables.variants):
        if not sid:continue
        species = names[sid]
        skey = tier2._species_key(species)
        info = dex.species_info(skey)
        for k, name in enumerate(info.types if info else ()):
            species_types[sid, k] = type_ids.get(normalize_id(name), 0)
        engine_species[sid] = species_ids.get(normalize_id(skey), 0)
        count[sid] = len(variants)
        for v, variant in enumerate(variants):
            valid[sid, v] = True
            fixture = worlds._fixture_from_variant_payload(variant.to_summary(), fallback_species=species, set_source=source)
            base = None if fixture is None else worlds._base_stats_for_species(source, fixture.species)
            if fixture is None or base is None:continue
            computed = worlds._computed_stats(base, evs=fixture.evs, ivs=fixture.ivs, level=fixture.level)
            participant[sid, v] = True
            level[sid, v] = fixture.level
            stats[sid, v] = [computed[s] for s in ('hp', 'atk', 'def', 'spa', 'spd', 'spe')]
            name = normalize_id(fixture.ability or '')
            ability[sid, v] = ability_ids.get(name, 0)
            if name and not ability[sid, v]:unknown['ability'].add(name)
            name = normalize_id(fixture.item or '')
            item[sid, v] = item_ids.get(name, 0)
            if name and not item[sid, v]:unknown['item'].add(name)
            canonical = [tier2.canonical_move_id(x) for x in fixture.moves]
            for n, move in enumerate(canonical[:4]):
                moves[sid, v, n] = key(move)
                if moves[sid, v, n] < 0:unknown['move'].add(move)
            hidden = next((x for x in canonical if x.startswith('hiddenpower')), None)   # _participant_move_id: the first
            if hidden is not None:
                # The variant's own Hidden Power move, as tier2 resolves it (dex type and base power). Every production
                # variant names its type; a source whose sets carry the bare 'hiddenpower' (a probe's literal source, whose
                # prepared ids drop the type) gets the bare move's dex type and power, as tier2 computes it there.
                hp_info = dex.move_info(hidden)
                hp_type[sid, v] = type_ids.get(normalize_id(hp_info.type if hp_info else ''), 0)
                if hp_info is None or not hp_type[sid, v]:unknown['type'].add(hidden)
                hp_power[sid, v] = hp_info.base_power if hp_info else 0
                if (len(hidden) > len('hiddenpower') and hp_info is not None
                        and hp_info.base_power != _hidden_power_iv_power(fixture.ivs)):
                    raise ValueError(f'{variant.variant_id}: dex Hidden Power power {hp_info.base_power} differs from its IVs')
    variants = dict(count=count, species_types=species_types, engine_species=engine_species, valid=valid,
        participant=participant, level=level, stats=stats, ability=ability, item=item, moves=moves, hp_type=hp_type,
        hp_power=hp_power)
    compiled = assemble(variants, metadata, flail=dex.move_info('flail'), source_hash=source.metadata.source_hash)
    return compiled, {k: tuple(sorted(v)) for k, v in unknown.items()}


def move_key_table(metadata):
    """[M] move identity for set membership: every Hidden Power alias maps to one key (the generic id if present)."""
    names = metadata['vocabulary']['moves']
    aliases = [i for i, name in enumerate(names) if i and normalize_id(name).startswith('hiddenpower')]
    hp_key = _vocab_ids(names).get('hiddenpower', aliases[0] if aliases else -1)
    return np.asarray([hp_key if i in aliases else i for i in range(len(names))], np.int32), hp_key


def assemble(variants, metadata, *, flail, source_hash):
    """PosteriorTables from the per-species / per-variant arrays (compile_tables, or authored fixtures) and the engine
    metadata: the per-move, per-ability, per-item and per-percent rules, the engine's sub-tables and the binding."""
    from pokezero import tier2
    vocab = metadata['vocabulary']
    moves_v, types_v, abilities_v, items_v = (vocab[k] for k in ('moves', 'types', 'abilities', 'items'))
    type_ids, ability_ids, item_ids = map(_vocab_ids, (types_v, abilities_v, items_v))
    rows = metadata['moves']
    if len(rows) != len(moves_v):raise ValueError('move metadata/vocabulary mismatch')
    field = lambda name, default:[default if row is None else row[name] for row in rows]
    category = field('category', 'Status')
    multi = np.asarray(special_hits.from_metadata(metadata).hits) > 1
    move_class = np.zeros(len(moves_v), np.int32)
    for i, name in enumerate(moves_v):
        move = normalize_id(name)
        for cls, members in (('struggle', {'struggle'}), ('fixed-damage', FIXED_DAMAGE_MOVES), ('ohko', OHKO_MOVES),
                             ('conditional-power', CONDITIONAL_POWER_MOVES), ('multi-hit', MULTI_HIT_MOVES)):
            if move in members or (cls == 'multi-hit' and multi[i]):move_class[i] |= 1 << CODE[cls]
    partner = np.zeros(len(abilities_v), np.int32)
    for a, b in (('plus', 'minus'), ('minus', 'plus')):
        if a in ability_ids:partner[ability_ids[a]] = ability_ids.get(b, 0)
    boost_type = np.zeros(len(items_v), np.int32);boost_num = np.ones(len(items_v), np.int32);boost_den = np.ones(len(items_v), np.int32)
    for name, typ in tier2.TYPE_BOOST_ITEMS.items():
        if name in item_ids:
            boost_type[item_ids[name]] = type_ids[normalize_id(typ)]
            boost_num[item_ids[name]], boost_den[item_ids[name]] = (21, 20) if name == 'seaincense' else (11, 10)
    chart = np.ones((len(types_v), len(types_v)), np.float32)
    chart[1:, 1:] = np.asarray(metadata['type_chart'], np.float32)
    arrays = dict(variants, move_key=move_key_table(metadata)[0],
        power=np.asarray(field('base_power', 0), np.int32), type=np.asarray(field('type', 0), np.int32),
        physical=np.asarray([x == 'Physical' for x in category], bool), damaging=np.asarray([x in ('Physical', 'Special') for x in category], bool),
        move_class=move_class, full_hp=np.asarray([normalize_id(n) in FULL_HP_MOVES for n in moves_v], bool),
        revenge=np.asarray([normalize_id(n) == 'revenge' for n in moves_v], bool), partner=partner,
        marvel_scale=np.asarray([normalize_id(n) == 'marvelscale' for n in abilities_v], bool), boost_type=boost_type,
        boost_num=boost_num, boost_den=boost_den, **_percent_tables(flail),
        type_num=np.where(chart == .5, 1, chart).astype(np.int32), type_den=np.where(chart == .5, 2, 1).astype(np.int32))
    digest = hashlib.sha256(json.dumps([SCHEMA, source_hash, sorted(arrays)]).encode())
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        digest.update(json.dumps([name, a.shape, a.dtype.str]).encode());digest.update(a.tobytes())
    binding = np.frombuffer(digest.digest(), '>u4').astype(np.uint32)
    return PosteriorTables(**{k: j.asarray(v) for k, v in arrays.items()},
        items=held.from_metadata(metadata), combat=traits.from_metadata(metadata), weather=climate.from_metadata(metadata),
        absorption=absorb.from_metadata(metadata), variable=variable.from_metadata(metadata), solar=charging.from_metadata(metadata),
        explosion=exploding.from_metadata(metadata), pursuit=intercepts.from_metadata(metadata), binding=j.asarray(binding))


# --------------------------------------------------------------------------- device kernel


def pack(bits):
    """[..., 64] bool -> [..., 2] uint32 in the belief mask's layout (variant i: word i//32, bit i%32)."""
    words = bits.reshape(bits.shape[:-1] + (WORDS, 32)).astype(j.uint32)
    return j.sum(words << j.arange(32, dtype=j.uint32), axis=-1, dtype=j.uint32)


def unpack(words):
    index = j.arange(LANES, dtype=j.uint32)
    return ((j.take(words, index // 32, axis=-1) >> (index % 32)) & 1).astype(bool)


def _lowest(mask):
    """Index of the lowest set bit of a nonnegative int32 bitmask (0 when empty: code 0 admits)."""
    mask = mask.astype(j.uint32)
    low = mask & (~mask + j.uint32(1))
    return j.where(mask == 0, 0, jax.lax.population_count(low - j.uint32(1))).astype(j.int32)


def strike(tables, x):
    """Per-variant rolls, counts and flags for one strike per row (all 64 lanes; hard is applied by reduce)."""
    s = j.clip(x.species, 0, tables.count.shape[0] - 1)
    mv = j.clip(x.move, 0, tables.power.shape[0] - 1)
    col = lambda v:j.asarray(v)[:, None]
    pct = j.clip(x.attacker_percent, 0, 100)
    valid, ok = tables.valid[s], tables.participant[s] & tables.valid[s]
    level, stats, ability, item = tables.level[s], tables.stats[s], tables.ability[s], tables.item[s]
    key = tables.move_key[mv]
    has_move = j.any(tables.moves[s] == key[:, None, None], axis=-1) & col(key > 0)
    # Base power and type: the engine's resolve (Hidden Power per variant, Return at happiness 255); Flail / Reversal
    # from the shown percent (tier2's public rule) and a Pursuit on a switching target at 80.
    power, typ, physical = variable.resolve(tables.variable, col(mv), col(tables.power[mv]), col(tables.type[mv]),
        col(tables.physical[mv]), tables.hp_type[s], tables.hp_power[s], HAPPINESS, 1, 1)
    flail = tables.variable.hp_dependent[mv]
    power = j.where(col(flail), col(tables.flail_power[pct]), power)
    power = j.where(col(tables.pursuit.pursuit[mv] & x.pursuit), 80, power)
    defender_ability = col(x.defender_ability)
    effective = climate.effective(tables.weather, col(x.weather),
        j.stack((ability, j.broadcast_to(defender_ability, ability.shape)), axis=-1), j.ones(ability.shape + (2,), j.int32))
    power = charging.base_power(tables.solar, col(mv), power, effective)
    power = variable.facade_power(tables.variable, col(mv), power, col(x.attacker_status))
    # BasePower chain (tier D): the variant's pinch ability, decided by the shown percent, with the defender's Thick Fat.
    pinch_max = col(j.where(tables.pinch_on[pct], 3, 1))
    power = traits.base_power(tables.combat, defender_ability, typ, power, attacker_ability=ability, hp=1, max_hp=pinch_max)
    pinch_type = tables.combat.pinch_type[ability]
    ambiguous = ((pinch_type > 0) & (typ == pinch_type) & col(tables.pinch_ambiguous[pct])) | col(flail & tables.flail_ambiguous[pct])
    unsupported = ~ok | ~has_move | col(~tables.damaging[mv]) | (power <= 0)
    # Stats and stat events.
    attack = j.where(physical, stats[..., 1], stats[..., 3])
    attack_stage = j.where(physical, col(x.attacker_boosts[:, 0]), col(x.attacker_boosts[:, 2]))
    defense = j.where(physical, col(x.defender_stats[:, 2]), col(x.defender_stats[:, 4]))
    defense_stage = j.where(physical, col(x.defender_boosts[:, 1]), col(x.defender_boosts[:, 3]))
    item_num, item_den = held.attack_modifier(tables.items, item, col(tables.engine_species[s]), physical)
    typeboost = (tables.boost_type[item] > 0) & (tables.boost_type[item] == typ)
    item_num = j.where(typeboost, item_num * tables.boost_num[item], item_num)
    item_den = j.where(typeboost, item_den * tables.boost_den[item], item_den)
    attack_num, attack_den = traits.attack_modifier(tables.combat, ability, col(x.attacker_status), physical,
        item_num=item_num, item_den=item_den)
    plusminus = (tables.partner[ability] > 0) & (tables.partner[ability] == defender_ability) & ~physical
    attack_num = j.where(plusminus, attack_num * 3, attack_num);attack_den = j.where(plusminus, attack_den * 2, attack_den)
    defense_num, defense_den = held.defense_modifier(tables.items, col(x.defender_item), col(x.defender_species), physical)
    marvel = col(tables.marvel_scale[x.defender_ability] & (x.defender_status != m.NONE)) & physical
    defense_num = j.where(marvel, defense_num * 3, defense_num);defense_den = j.where(marvel, defense_den * 2, defense_den)
    weather_num, weather_den = climate.damage_modifier(tables.weather, effective, typ)
    flash = col(j.where(x.attacker_flash_fire, j.uint32(absorb.FLASH_FIRE), j.uint32(0)))
    phase1_num, phase1_den = absorb.damage_phase1(tables.absorption, typ, flash)
    types = x.defender_types
    num = tables.type_num[typ, col(types[:, 0])] * tables.type_num[typ, col(types[:, 1])]
    den = tables.type_den[typ, col(types[:, 0])] * tables.type_den[typ, col(types[:, 1])]
    immune = num == 0
    stab = j.any(tables.species_types[s][:, None, :] == typ[..., None], axis=-1) & (typ != 0)
    roll = 85 + j.arange(ROLLS, dtype=j.int32)
    e = lambda v:j.asarray(v)[..., None]
    rolls = m.gen3_damage(level=e(level), power=e(power), attack=e(attack), defense=e(defense), attack_stage=e(attack_stage),
        defense_stage=e(defense_stage), physical=e(physical), critical=e(col(x.crit)),
        stab=e(stab), typeless=e(typ == 0), burned=e(col(x.attacker_status == m.BRN)), guts=e(tables.combat.guts[ability]),
        hustle=e(tables.combat.hustle[ability]), attack_num=e(attack_num), attack_den=e(attack_den),
        defense_num=e(defense_num), defense_den=e(defense_den),
        halve_defense=e(col(tables.explosion.move[mv]) & physical), weather_num=e(weather_num), weather_den=e(weather_den),
        phase1_num=e(phase1_num), phase1_den=e(phase1_den), effectiveness_num=e(num), effectiveness_den=e(den),
        immune=e(immune), roll=roll)
    rolls = j.where(e(valid), rolls, 0)
    fit = j.where(col(x.ko)[..., None], rolls >= x.hp_before[:, None, None], rolls == x.observed[:, None, None])
    counts = j.sum(fit, axis=-1, dtype=j.int32)
    line = j.where(num > den, 1, j.where(num < den, -1, 0))   # supereffective / resisted / neither (immune: never fits)
    effdis = (counts > 0) & (line != col(x.effectiveness))
    full = tables.full_hp[mv] & (pct < 100)
    move_bits = tables.move_class[mv] | j.where(full, 1 << CODE['conditional-power'], 0)
    return StrikeResult(rolls, counts, unsupported, ambiguous, immune & ~unsupported, effdis, typeboost & ~unsupported,
        marvel & ~unsupported, plusminus & ~unsupported, move_bits.astype(j.int32))


def reduce(result, hard, precondition, *, guard):
    """set_posterior's verdict for one strike per row over its hard support ([B,2] uint32): the first caller or move-class
    refusal; else single-variant / no-support; else the first hard variant in source order that is core-unsupported,
    ambiguous or disagrees on effectiveness (in that order within the variant, as _counts checks it); else, with guard,
    an engine-unmodeled modifier active on any hard variant. An admitted strike's survivors are its fitting hard variants."""
    hard_bits = unpack(hard)
    code = _lowest(j.asarray(precondition, j.int32) | result.move_bits)
    size = j.sum(hard_bits, axis=-1)
    unsupported, ambiguous, effdis = result.unsupported & hard_bits, result.ambiguous & hard_bits, result.effdis & hard_bits
    stop = unsupported | ambiguous | effdis
    first = j.argmax(stop, axis=-1)
    take = lambda a:j.take_along_axis(a, first[:, None], axis=-1)[:, 0]
    core = j.where(take(unsupported), CODE['core-unsupported'], j.where(take(ambiguous), CODE['ambiguous-band'],
        CODE['effectiveness-disagreement']))
    code = j.where(code > 0, code, j.where(size == 1, CODE['single-variant'], j.where(size == 0, CODE['no-support'],
        j.where(j.any(stop, axis=-1), core, 0))))
    if guard:
        unmodeled = j.where(j.any(result.typeboost & hard_bits, axis=-1), CODE['engine-unmodeled:typeboost'],
            j.where(j.any(result.marvel & hard_bits, axis=-1), CODE['engine-unmodeled:marvelscale'],
            j.where(j.any(result.plusminus & hard_bits, axis=-1), CODE['engine-unmodeled:plusminus'], 0)))
        code = j.where(code > 0, code, unmodeled)
    fit = pack((result.counts > 0) & hard_bits & (code == 0)[:, None])
    return Reduced(code.astype(j.int32), fit)


def narrow(pin, pin_active, conflicts, fit, admitted):
    """public_belief.narrow_pin on the standing pin of the strike's subject (B2 installs set_posterior's support with the
    same function): intersect with the active pin, decline an empty incoming set, count a conflict on a disjoint one and
    never erase. An off-model strike (no hard variant fits) arrives empty and is declined, as set_posterior drops it.
    Returns (pin, pin_active, conflicts, changed)."""
    from . import public_belief
    state = public_belief.BeliefState(j.zeros_like(pin_active, j.int32), j.ones_like(pin_active), j.zeros_like(pin),
        pin, pin_active, j.ones_like(pin_active), conflicts)
    updated, changed = public_belief.narrow_pin(state, fit, valid=admitted)
    return updated.pin, updated.pin_active, updated.conflicts, changed


_SOURCES = {}


def register_source(source):
    """Bind a Gen3RandbatSource for engine_device_belief.compile_tables by its hash (a source the process cache does
    not hold: a copied production source, a probe's literal source)."""
    _SOURCES[source.metadata.source_hash] = source
    return source


def bound_source(source_hash, source=None):
    """The Gen3RandbatSource the encoder catalog's compiled source was built from: the explicit one, a registered one,
    else randbat's process cache (load_gen3_randbat_source_cached), matched by source hash. Refuses rather than guess."""
    if source is None:
        source = _SOURCES.get(source_hash)
    if source is None:
        from pokezero import randbat
        with randbat._SOURCE_CACHE_LOCK:
            found = [s for s in randbat._SOURCE_CACHE.values() if s.metadata.source_hash == source_hash]
        source = found[0] if found else None
    if source is None or source.metadata.source_hash != source_hash:
        raise ValueError('set_posterior_narrowing needs the bound Gen3RandbatSource of source hash ' + str(source_hash))
    return source


def enable(belief_tables, posterior):
    """Belief tables with the producer's tables and the standing pins on (what compile_tables builds when the
    checkpoint masks carry set_posterior_narrowing; tests and probes without that mask use this)."""
    return belief_tables._replace(standing_pins_enabled=j.asarray(True), posterior=posterior)


# --------------------------------------------------------------------------- in-tree producer


def in_tree(trackers, belief_tables, state, side, move, caller_move, critical, hits_sub, endured, dealt, hp_before, hp_after,
            num, den, use, *, pursuit=False, hit_by=None):
    """The producer at one damage reveal of the engine step (the hook beside device_public.damage, and the Pursuit
    intercept): side's active hit the other side's active with `move` for `dealt` HP (exact; 0 on a Substitute hit).

    Every operand is one the engine already computed (no draw): the resolved crit, the Substitute and Endure gates, the
    actual HP, and num/den of the true move type (the printed effectiveness line). The attacker's public facts come from
    the carry (its public encounter slot, its species, its hard mask and its shown HP percent); the defender is the
    observer's own active, read exactly from state. Refusals that only engine state can tell are the caller bits:
    called (a Sleep Talk callee: move != caller_move), hit-substitute, endure-or-focusband, type-change (types !=
    base_types on either side), ability-overridden (ability != base_ability: Trace), item-changed (B2's item_mutated
    ledger on either slot), Revenge after a hit this turn (hit_by) and no-hp-drop. An admitted strike's fitting hard
    variants narrow the attacker's standing pin (narrow_pin: intersect, decline empty, count a disjoint conflict) and
    that subject is refreshed, as B2 installs set_posterior's support. Returns (trackers, reduced)."""
    from . import engine_device_visible_belief as visible
    from . import lane_index as li
    tables = belief_tables.posterior
    b = state.hp.shape[0];r = j.arange(b)
    side = j.broadcast_to(j.asarray(side, j.int32), (b,));foe = 1 - side
    own, target = li.take(state.active, r, side), li.take(state.active, r, foe)
    get = lambda value, seat, physical:li.take(value, r, seat, j.clip(physical, 0, 5))
    slot = j.clip(get(trackers.physical_to_public, side, own), 0, 5)
    target_slot = j.clip(get(trackers.physical_to_public, foe, target), 0, 5)
    species = li.take(trackers.species, r, side, slot)
    source_species = belief_tables.source_species[j.clip(species, 0, belief_tables.source_species.shape[0] - 1)]
    hard = li.take(trackers.belief_mask, r, side, slot)
    mutated = trackers.mon_features.get('item_mutated') if trackers.mon_features is not None else None
    changed = j.zeros(b, bool) if mutated is None else (li.take(mutated, r, side, slot) != 0) | (li.take(mutated, r, foe, target_slot) != 0)
    retyped = j.any(get(state.types, side, own) != get(state.base_types, side, own), axis=-1) | j.any(
        get(state.types, foe, target) != get(state.base_types, foe, target), axis=-1)
    traced = (get(state.ability, side, own) != get(state.base_ability, side, own)) | (
        get(state.ability, foe, target) != get(state.base_ability, foe, target))
    pre = (j.where(move != caller_move, bit('called'), 0) | j.where(hits_sub, bit('hit-substitute'), 0)
           | j.where(endured, bit('endure-or-focusband'), 0) | j.where(retyped, bit('type-change'), 0)
           | j.where(changed, bit('item-changed'), 0) | j.where(traced, bit('ability-overridden'), 0)
           | j.where(~hits_sub & (dealt <= 0), bit('no-hp-drop'), 0))
    if hit_by is not None:
        revenge = tables.revenge[j.clip(move, 0, tables.revenge.shape[0] - 1)] & hit_by
        pre = pre | j.where(revenge, bit('conditional-power'), 0)
    x = StrikeInputs(species=source_species, move=move, crit=critical, weather=state.weather,
        pursuit=j.broadcast_to(j.asarray(pursuit, bool), (b,)), attacker_status=get(state.status, side, own),
        attacker_boosts=get(state.boosts, side, own), attacker_percent=li.take(trackers.hp_percent, r, side, slot),
        attacker_flash_fire=(get(state.volatile, side, own) & j.uint32(absorb.FLASH_FIRE)) != 0,
        defender_stats=get(state.stats, foe, target), defender_boosts=get(state.boosts, foe, target),
        defender_types=get(state.types, foe, target), defender_ability=get(state.ability, foe, target),
        defender_item=get(state.item, foe, target), defender_species=get(state.species, foe, target),
        defender_status=get(state.status, foe, target), hp_before=hp_before, observed=dealt, ko=hp_after == 0,
        effectiveness=j.where(num > den, 1, j.where(num < den, -1, 0)), precondition=pre.astype(j.int32))
    reduced = reduce(strike(tables, x), hard, x.precondition, guard=True)
    # The host resolves its own defender by species in the original request, not by physical slot or active status.
    # Identical duplicates are ambiguous too; being able to read the engine's exact active stats does not admit them.
    defender_species = get(state.species, foe, target)
    known_defender = (defender_species > 0) & (j.sum(li.take(state.species, r, foe) == defender_species[:, None], axis=1) == 1)
    unknown = ~known_defender & ((reduced.reason == 0) | (reduced.reason > CODE['unknown-defender']))
    reduced = Reduced(j.where(unknown, CODE['unknown-defender'], reduced.reason),
                      j.where(unknown[:, None], j.uint32(0), reduced.fit))
    admitted = j.broadcast_to(j.asarray(use, bool), (b,)) & (reduced.reason == 0) & (source_species > 0)
    trackers, _ = visible.register_variant_survivors(trackers, belief_tables, side, own, reduced.fit, admitted)
    trackers = visible.refresh(trackers, belief_tables, side, own, admitted)
    return trackers, reduced
