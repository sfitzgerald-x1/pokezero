"""Literal direct-hit controls at the SAME executed target HP assignment.

INTERNAL kinds21/22/23 are critical/super-effective/resisted. Every ten-word
row is [kind, actor_side, actor_physical, target_side, target_physical, move,
OLD_HP, NEW_HP, actual effectiveness numerator, actual effectiveness denominator].
Emit critical first, then at most one effectiveness control, immediately AFTER
the numerical HP assignment and BEFORE its existing MAIN_HP callback. None of
these flags or ratios is reconstructed from the amount of HP lost.

Positive nonlethal ordinary direct-target writes and the exact original compiled
Seismic Toss fixed-level row are represented. Miss,
immunity, substitute, protection, Endure and no-output writes stay UNKNOWN.
The registered caller separately dispatches literal no-damage callbacks via
engine_no_damage_callbacks; these positive-write validators are unchanged.
These APIs remain nonlethal; the separate typed KO source requires its actual
later faint callback and never relaxes this validator. Diagnostic calls retain UNKNOWN256. The separate PRIVATE registered
source seam omits only this new latch for valid literal writes in the matching
open accounting context; prior refusal bits remain unchanged. Neutral/noncritical
writes emit no control and provide no no-output certificate. No damage formula,
faint, source exhaustiveness, public
admission, original-physics equivalence or qualification is implemented here.
The actual caller must pass literal values from the same execution. Device
data/copies are trusted transport, not proof of caller origin or single use.
"""
from dataclasses import dataclass
import hashlib
import json
import subprocess
from pathlib import Path
from typing import NamedTuple
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from . import engine_public_events as public, engine_lifecycle_callbacks as life
from . import public_mon_state

CRITICAL, SUPER_EFFECTIVE, RESISTED = 21, 22, 23
UNKNOWN_DIRECT_HIT = 256
SCHEMA = 'pokezero.engine_direct_hit_controls.literal.v1'
# transition.tables_from_metadata maps each type cell to (0,1), (1,1),
# (1,2), or (2,1); the actual target has two type cells. Do not reduce (2,2).
POSITIVE_EFFECTIVENESS_RATIOS = ((1,1), (2,1), (4,1), (1,2), (1,4), (2,2))
_COMPILED = weakref.WeakKeyDictionary()


class Data(NamedTuple):
    binding: object                 # GLOBAL8, including batch8
    engine_binding: object
    source_binding: object
    move_binding: object
    producer_binding: object
    move_count: object
    seismic_toss_move: object       # exact canonical fixed-LEVEL row, or0
    ko_candidates: object          # ORIGINAL positive-power/nonrecoil rows only
    no_damage: object              # original literal miss/immunity/TryHit capability
    secondary_status: object = None # explicit original-rule plain secondary capability
    substitute_damage: object = None # exact original ordinary Substitute source
    protect: object = None # explicit verified-rule Protect-condition family
    primary_plain_status: object = None # original plain-primary PAR/BRN source
    bulk_up: object = None # original-bound ATK/DEF self-pair capability
    rough_skin: object = None # original-bound actual contact HP write
    color_change: object = None # original Gen3 DamagingHit type mutation
    pinch_item: object = None # original Gen3 residual item transaction
    self_heal: object = None # original positive Recover/Soft-Boiled/Synthesis write
    belly_drum: object = None # original literal HP payment and absolute ATK6
    sleep_blocked: object = None # original ordinary SLP BeforeMove result
    encore: object = None # original apply/residual/request lifecycle
    leech_seed: object = None # original seed assignment + paired residual source slot
    self_move: object = None # partial original self-move failures/Wish start
    return_ko_move: object = None # canonical source-bound Return; actual happiness required
    heal_bell: object = None # source-bound actual six-member cure transaction


class Context(NamedTuple):
    """Private optional caller holder, not consumer registration or proof.

    The owning host must check the original compiler before every invocation;
    this holder merely transports its SAME Data into the actual numerical core.
    """
    data: Data


class RegisteredContext(NamedTuple):
    """PRIVATE same-helper/accounting statics; not a caller acceptance flag.

    Original compiler identities must be checked by the owning host. The root
    still owns the actual BEFORE ticket and exact following HP span capture;
    static tables alone cannot attest a ticket, origin or consumer completion.
    """
    data: Data
    accounting: object


@dataclass(frozen=True, eq=False)
class Compiled:
    device: Data


class HitWrite(NamedTuple):
    """Literal BEFORE/AFTER HP and actual booleans/ratio at the kernel write."""
    power: object
    critical: object
    hit: object
    immune: object
    numerator: object
    denominator: object
    substitute: object
    protected: object
    enduring: object
    old_hp: object
    new_hp: object
    maximum: object


def _host():
    if isinstance(jnp.zeros((), jnp.int32), jax.core.Tracer):
        raise ValueError('checked direct-hit host boundary cannot be traced')


def _compile_return_ko(catalog,metadata,rules):
    """Bind existing original Return resolver; no power or HP is synthesized."""
    from . import engine_request_source as request
    origin=request._rules_origin(rules)
    script=r"""
const fs=require('fs'),path=require('path'),crypto=require('crypto'),root=process.argv[1],meta=JSON.parse(fs.readFileSync(0,'utf8'));
const verified=require(process.argv[2]).build(root,meta),d=require(path.join(root,'dist/sim/dex.js')).Dex.mod('gen3'),m=d.moves.get('return');
process.stdout.write(JSON.stringify({id:m.id,power:m.basePower,type:m.type,category:m.category,target:m.target,accuracy:m.accuracy,callback:crypto.createHash('sha256').update(m.basePowerCallback.toString()).digest('hex'),callbacks:Object.keys(m).filter(k=>(/^on/.test(k)||/Callback$/.test(k))&&m[k]!==undefined),exotic:!!(m.multihit||m.damage||m.recoil||m.drain||m.heal||m.self||m.secondary||m.secondaries||m.boosts||m.status||m.volatileStatus||m.sideCondition||m.slotCondition||m.condition||m.weather||m.selfdestruct||m.forceSwitch||m.selfSwitch||m.ohko),source_tree:verified.source_tree,compiled_tree:verified.compiled_tree}));"""
    row=json.loads(subprocess.run([origin.node,'-e',script,origin.root,str(Path(request.__file__).with_name('request_source_rules.cjs'))],input=json.dumps(metadata,allow_nan=False),capture_output=True,text=True,check=True,timeout=30).stdout)
    if (row['id'],row['power'],row['type'],row['category'],row['target'],row['accuracy'],row['callback'],row['callbacks'],row['exotic'])!=('return',0,'Normal','Physical','normal',100,'a2bce14a77c22d55d42a7e1bd68c59689ef9a08d57dbabb645afb96a6ac2ecac',['basePowerCallback'],False):raise ValueError('original canonical Return resolver required')
    if json.dumps(row['source_tree'],sort_keys=True)!=origin.source_tree_json or json.dumps(row['compiled_tree'],sort_keys=True)!=origin.compiled_tree_json:raise ValueError('Return source tree changed')
    names=metadata['vocabulary']['moves'];table=catalog.engine_table;wanted=np.asarray([n=='return'for n in names],bool)
    if table.variable_moves is None or not wanted.any():return 0,row
    if not np.array_equal(table.variable_moves.returns,wanted):raise ValueError('original installed Return resolver required')
    ids=np.flatnonzero(wanted)
    if len(ids)!=1:return 0,row
    move=int(ids[0]);normal=metadata['vocabulary']['types'].index('normal')
    if not(move>0 and int(catalog.ids[move])>0 and int(catalog.targets[move])==1 and int(table.power[move])==0 and int(table.type[move])==normal and bool(table.physical[move]) and int(table.accuracy[move])==100 and not bool(table.always_hit[move])):raise ValueError('original canonical Return numerical row required')
    return move,row


def compile_direct_hit_controls(*, engine_catalog, metadata=None, source_rules=None):
    """Bind the original immutable kernel catalog outside every owning JIT."""
    _host()
    public.check_catalog(engine_catalog, engine_catalog.engine_table,
        expected_binding=engine_catalog.binding,
        expected_engine_binding=np.asarray(engine_catalog.engine_binding, np.uint32))
    count = len(engine_catalog.engine_table.power)
    if not 1 < count < 2**31:
        raise ValueError('nonempty original move namespace required')
    num = np.asarray(engine_catalog.engine_table.type_num)
    den = np.asarray(engine_catalog.engine_table.type_den)
    if (num.shape != (18,18) or den.shape != num.shape or num.dtype != np.int32 or den.dtype != np.int32
            or not np.all(((num == 0) & (den == 1)) | ((num == 1) & ((den == 1) | (den == 2)))
                | ((num == 2) & (den == 1)))):
        raise ValueError('original finite kernel type-chart word domain required')
    # The only zero-power source capability is canonical Seismic Toss. Its
    # original LEVEL resolver returns before crit/effectiveness modifiers.
    # Other fixed families (including Night Shade) remain unsupported.
    table=engine_catalog.engine_table;seismic=0
    public_id=engine_catalog.public_moves.ids.get('seismictoss',0)
    for i,pid in enumerate(engine_catalog.ids):
        if (i>0 and public_id>0 and pid==public_id and engine_catalog.targets[i]==1
                and table.damage_moves is not None and int(table.power[i])==0
                and int(table.type[i])==2 and bool(table.physical[i])
                and int(table.accuracy[i])==100 and not bool(table.always_hit[i])
                and int(table.priority[i])==0 and bool(table.damage_moves.level_damage[i])
                and int(table.damage_moves.constant_damage[i])==0
                and int(table.damage_moves.recoil_den[i])==0 and not bool(table.damage_moves.struggle[i])):
            seismic=i
    return_ko=0;return_source=None
    if metadata is not None and source_rules is not None:return_ko,return_source=_compile_return_ko(engine_catalog,metadata,source_rules)
    ko=((np.asarray(table.power)>0)|((return_ko>0)&(np.arange(count)==return_ko)))&(np.asarray(engine_catalog.ids)>0)&(np.asarray(engine_catalog.targets)==1)
    ko &= np.arange(count)!=int(table.struggle)
    if table.damage_moves is not None:ko &= np.asarray(table.damage_moves.recoil_den)==0
    if table.self_destruct is not None:ko &= ~np.asarray(table.self_destruct.move)
    from . import engine_no_damage_callbacks as no_damage
    secondary_data=None;substitute_damage_data=None;protect_data=None;bulk_up_data=None;plain_data=None;rough_skin_data=None;color_data=None;pinch_data=None;heal_data=None;belly_data=None;sleep_block_data=None;encore_data=None;leech_data=None;self_move_data=None;heal_bell_data=None
    if metadata is not None or source_rules is not None:
        if metadata is None or source_rules is None:raise ValueError('original secondary metadata AND rules required')
        from . import engine_secondary_status_callbacks as secondary_status
        secondary_data=secondary_status.compile_data(engine_catalog,metadata,source_rules)
        from . import engine_substitute_damage_callbacks as substitute_damage
        substitute_damage_data=substitute_damage.compile_data(engine_catalog,metadata,source_rules)
        from . import engine_protect_callbacks as protect_callbacks
        protect_data=protect_callbacks.compile_data(engine_catalog,metadata,source_rules)
        from . import engine_bulk_up_callbacks as bulk_up
        bulk_up_data=bulk_up.compile_data(engine_catalog,metadata,source_rules)
        from . import engine_primary_plain_status_callbacks as plain_primary
        plain_data=plain_primary.compile_data(engine_catalog,metadata,source_rules)
        from . import engine_rough_skin_callbacks as rough_skin
        rough_skin_data=rough_skin.compile_data(engine_catalog,metadata,source_rules,substitute_damage_data)
        from . import engine_color_change_callbacks as color
        color_data=color.compile_data(engine_catalog,metadata,source_rules)
        from . import engine_pinch_item_callbacks as pinch
        pinch_data=pinch.compile_data(engine_catalog,metadata,source_rules)
        from . import engine_self_heal_callbacks as self_heal
        heal_data=self_heal.compile_data(engine_catalog,metadata,source_rules)
        from . import engine_belly_drum_callbacks as belly
        belly_data=belly.compile_data(engine_catalog,metadata,source_rules)
        from . import engine_sleep_blocked_callbacks as sleep_blocked
        sleep_block_data=sleep_blocked.compile_data(engine_catalog,metadata,source_rules)
        from . import engine_encore_callbacks as encore_callbacks
        encore_data=encore_callbacks.compile_data(engine_catalog,metadata,source_rules)
        from . import engine_leech_seed_callbacks as leech_callbacks
        leech_data=leech_callbacks.compile_data(engine_catalog,metadata,source_rules)
        from . import engine_heal_bell_callbacks as heal_bell_callbacks
        heal_bell_data=heal_bell_callbacks.compile_data(engine_catalog,metadata,source_rules)
    if metadata is not None:
        from . import engine_self_move_callbacks as self_move_callbacks
        self_move_data=self_move_callbacks.compile_data(engine_catalog,metadata,source_rules)
    no_damage_data=no_damage.compile_data(engine_catalog,metadata,source_rules)
    payload = dict(schema=SCHEMA, heal_bell_binding=None if heal_bell_data is None else np.asarray(heal_bell_data.binding).tolist(), return_ko_move=return_ko,return_source=return_source, self_move_binding=None if self_move_data is None else np.asarray(self_move_data.binding).tolist(), leech_binding=None if leech_data is None else np.asarray(leech_data.binding).tolist(), encore_binding=None if encore_data is None else np.asarray(encore_data.binding).tolist(), sleep_blocked_binding=None if sleep_block_data is None else np.asarray(sleep_block_data.binding).tolist(), belly_drum_binding=None if belly_data is None else np.asarray(belly_data.binding).tolist(), self_heal_binding=None if heal_data is None else np.asarray(heal_data.binding).tolist(), pinch_binding=None if pinch_data is None else np.asarray(pinch_data.binding).tolist(), color_change_binding=None if color_data is None else np.asarray(color_data.binding).tolist(), rough_skin_binding=None if rough_skin_data is None else np.asarray(rough_skin_data.binding).tolist(), primary_plain_binding=None if plain_data is None else np.asarray(plain_data.binding).tolist(), bulk_up_binding=None if bulk_up_data is None else np.asarray(bulk_up_data.binding).tolist(), protect_binding=None if protect_data is None else np.asarray(protect_data.binding).tolist(), substitute_damage_binding=None if substitute_damage_data is None else np.asarray(substitute_damage_data.binding).tolist(), secondary_binding=None if secondary_data is None else np.asarray(secondary_data.binding).tolist(), no_damage_binding=np.asarray(no_damage_data.binding).tolist(), producer=engine_catalog.binding,
        engine=engine_catalog.engine_binding, source=engine_catalog.source_binding,
        moves=engine_catalog.move_binding, move_count=count, seismic_toss_move=seismic,ko_candidates=ko.tolist(),
        kinds=[CRITICAL, SUPER_EFFECTIVE, RESISTED],
        positive_effectiveness_ratios=POSITIVE_EFFECTIVENESS_RATIOS,
        source_files={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('transition.py', 'critical.py', 'mechanics.py', 'damage_moves.py', 'engine_direct_hit_controls.py','engine_direct_ko_callbacks.py','engine_recoil_callbacks.py','secondary.py','engine_secondary_status_callbacks.py','engine_flash_fire_end_callbacks.py','engine_substitute_damage_callbacks.py','engine_protect_callbacks.py','engine_bulk_up_callbacks.py','engine_primary_plain_status_callbacks.py','engine_rough_skin_callbacks.py','engine_color_change_callbacks.py','engine_pinch_item_callbacks.py','engine_self_heal_callbacks.py','engine_belly_drum_callbacks.py','engine_sleep_blocked_callbacks.py','engine_encore_callbacks.py','engine_leech_seed_callbacks.py','engine_self_move_callbacks.py','engine_heal_bell_callbacks.py','engine_request_order.py','effects.py','sustain.py','encore.py','engine_request_source.py','engine_private_actions.py','items.py','state.py','entry.py')},
        scope='literal-positive-nonlethal-and-source-pending-nonterminal-ko-only-no-completeness')
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).digest()
    words = lambda x:jnp.asarray(x, jnp.uint32)
    data = Data(words(np.frombuffer(digest, dtype='>u4').astype(np.uint32)),
        words(engine_catalog.engine_binding), words(engine_catalog.source_binding),
        words(engine_catalog.move_binding),
        words(np.frombuffer(bytes.fromhex(engine_catalog.binding), dtype='>u4').astype(np.uint32)),
        jnp.array(count, jnp.int32),jnp.array(seismic,jnp.int32),jnp.asarray(ko),no_damage_data,secondary_data,substitute_damage_data,protect_data,plain_data,bulk_up_data,rough_skin_data,color_data,pinch_data,heal_data,belly_data,sleep_block_data,encore_data,leech_data,self_move_data,jnp.array(return_ko,jnp.int32),heal_bell_data)
    result = Compiled(data)
    _COMPILED[result] = (data, tuple(data), engine_catalog,source_rules)
    return result


def check_compiled(compiled):
    _host()
    record = _COMPILED.get(compiled) if type(compiled) is Compiled else None
    if (record is None or compiled.device is not record[0]
            or any(a is not b for a, b in zip(compiled.device, record[1]))
            or any(not isinstance(a, jax.Array) or isinstance(a, jax.core.Tracer) for a in jax.tree.leaves(compiled.device))):
        raise ValueError('original immutable direct-hit compiler required')
    catalog = record[2]
    if record[3] is not None:
        from . import engine_request_source as request
        request._rules_origin(record[3])
    public.check_catalog(catalog, catalog.engine_table, expected_binding=catalog.binding,
        expected_engine_binding=compiled.device.engine_binding)
    return compiled.device


def _check(tape, data):
    if type(tape) is public.CallbackTape:
        return False
    if type(tape) is not life.LifecycleTape or type(tape.cursor) is not life.Cursor:
        raise ValueError('typed actual lifecycle tape/cursor required')
    if getattr(tape.count, 'ndim', None) != 1 or not tape.count.shape[0]:
        raise ValueError('nonempty direct-hit callback batch required')
    b = tape.count.shape[0]
    if tape.words.ndim != 3 or tape.words.shape[0] != b or not 1 <= tape.words.shape[1] <= 64 or tape.words.shape[2] != 10:
        raise ValueError('bounded actual callback storage required')
    c = tape.words.shape[1]
    for value, shape, dtype in ((tape.words, (b,c,10), jnp.int32), (tape.count, (b,), jnp.int32),
            (tape.refusal, (b,), jnp.int32), (tape.turn, (b,), jnp.int32),
            (tape.stream_id, (b,), jnp.int32), (tape.event_turn, (b,c), jnp.int32),
            (tape.event_phase, (b,c), jnp.int32), (tape.event_generation, (b,c,2), jnp.int32),
            (tape.switch_details, (b,c,6), jnp.int32)):
        life._typed(value, shape, dtype)
    if type(data) is not Data:
        raise ValueError('original checked direct-hit data required')
    for value in (tape.engine_binding, tape.source_binding, tape.move_binding, tape.producer_binding,
            data.binding, data.engine_binding, data.source_binding, data.move_binding, data.producer_binding):
        life._typed(value, (8,), jnp.uint32)
    life._typed(data.move_count, (), jnp.int32)
    life._typed(data.seismic_toss_move, (), jnp.int32)
    life._typed(data.return_ko_move, (), jnp.int32)
    if data.ko_candidates.ndim!=1 or len(data.ko_candidates)<2:raise ValueError('exact original KO move domain required')
    life._typed(data.ko_candidates,(len(data.ko_candidates),),jnp.bool_)
    from . import engine_no_damage_callbacks as no_damage
    no_damage._shape(data.no_damage)
    if data.heal_bell is not None:
        from . import engine_heal_bell_callbacks as heal_bell_callbacks
        heal_bell_callbacks.shape(data.heal_bell)
    if data.secondary_status is not None:
        from . import engine_secondary_status_callbacks as secondary_status
        secondary_status.shape(data.secondary_status)
    if data.substitute_damage is not None:
        from . import engine_substitute_damage_callbacks as substitute_damage
        substitute_damage.shape(data.substitute_damage)
    if data.primary_plain_status is not None:
        from . import engine_primary_plain_status_callbacks as plain_primary
        plain_primary.shape(data.primary_plain_status)
    if data.color_change is not None:
        from . import engine_color_change_callbacks as color
        color.shape(data.color_change)
    if data.pinch_item is not None:
        from . import engine_pinch_item_callbacks as pinch
        pinch.shape(data.pinch_item)
    if data.bulk_up is not None:
        from . import engine_bulk_up_callbacks as bulk_up
        bulk_up.shape(data.bulk_up)
    if data.protect is not None:
        from . import engine_protect_callbacks as protect_callbacks
        protect_callbacks.shape(data.protect)
    if data.self_heal is not None:
        from . import engine_self_heal_callbacks as self_heal
        self_heal.shape(data.self_heal)
    if data.leech_seed is not None:
        from . import engine_leech_seed_callbacks as leech_callbacks
        leech_callbacks.shape(data.leech_seed)
    if data.self_move is not None:
        from . import engine_self_move_callbacks as self_move_callbacks
        self_move_callbacks.shape(data.self_move)
    if data.encore is not None:
        from . import engine_encore_callbacks as encore_callbacks
        encore_callbacks.shape(data.encore)
    if data.sleep_blocked is not None:
        from . import engine_sleep_blocked_callbacks as sleep_blocked
        sleep_blocked.shape(data.sleep_blocked)
    life._valid(tape.cursor)
    return True


def _boolean(value, shape):
    if type(value) is bool:
        return jnp.full(shape, value)
    if getattr(value, 'dtype', None) != jnp.dtype(bool) or getattr(value, 'shape', None) not in ((), shape):
        raise ValueError('actual direct-hit flags must be boolean scalar/B')
    return jnp.broadcast_to(jnp.asarray(value), shape)


def _word(value, shape):
    value = public_mon_state._int32(value)
    if value.shape not in ((), shape):
        raise ValueError('literal direct-hit words must be int32 scalar/B')
    return jnp.broadcast_to(value, shape)


def _unknown(tape, eligible):
    return tape._replace(refusal=tape.refusal | jnp.where(eligible, UNKNOWN_DIRECT_HIT, 0))


def _typed_write(write, b):
    if type(write) is not HitWrite:
        raise ValueError('literal typed direct-hit write required')
    return HitWrite(**{name:(_boolean(value, (b,)) if name in
        ('critical', 'hit', 'immune', 'substitute', 'protected', 'enduring')
        else _word(value, (b,))) for name, value in zip(write._fields, write)})


def fixed_level_move_device(data,move):
    """Original compiled source row only; not actual write/attempt acceptance."""
    if type(data) is not Data:raise ValueError('original direct-hit Data required')
    life._typed(data.seismic_toss_move,(),jnp.int32)
    return ((data.seismic_toss_move>0)&(data.seismic_toss_move<data.move_count)
        &(move==data.seismic_toss_move))


def annotation_flags_device(write,*,data=None,move=None):
    # Keep the literal unused crit/type fields in HitWrite/Outcome. The original
    # fixed-level source returns before the public modifier annotations.
    fixed=False if data is None or move is None else fixed_level_move_device(data,move)
    return write.critical&~jnp.asarray(fixed), (write.numerator!=write.denominator)&~jnp.asarray(fixed)


def _valid_write(w,*,data=None,move=None):
    fixed=False if data is None or move is None else fixed_level_move_device(data,move)
    power=((w.power>0)&~jnp.asarray(fixed))|((w.power==0)&jnp.asarray(fixed))
    return (power & w.hit & ~w.immune & ~w.substitute & ~w.protected & ~w.enduring
        & _positive_ratio(w.numerator, w.denominator)
        & (w.old_hp > w.new_hp) & (w.new_hp > 0) & (w.maximum >= w.old_hp))


def validate_write_device(write, b, *, data=None, move=None):
    """Pure literal write validity only; no damage formula or caller proof."""
    if type(b) is not int or b <= 0 or type(write) is not HitWrite:
        raise ValueError('positive static batch and literal HitWrite required')
    try:
        w = _typed_write(write, b)
    except (TypeError, ValueError):
        return jnp.zeros(b, bool)
    return _valid_write(w,data=data,move=move)


def _positive_ratio(numerator, denominator):
    pairs = jnp.asarray(POSITIVE_EFFECTIVENESS_RATIOS, jnp.int32)
    return jnp.any((numerator[...,None] == pairs[:,0]) & (denominator[...,None] == pairs[:,1]), axis=-1)


def validate_raw_device(raw):
    """Pure payload syntax/finite words, NOT actual callsite/consumer proof.

    Owning lower/full ports must separately bind the same actual move, actor,
    target, clocks, previous HP and immediately following matching MAIN_HP.
    Every positive finite ratio is legal for CRITICAL, including neutral2/2.
    """
    if getattr(raw, 'ndim', None) != 2 or not raw.shape[0] or raw.shape[1] != 10:
        raise ValueError('nonempty int32[B,10] direct-hit payload required')
    life._typed(raw, raw.shape, jnp.int32)
    kind, actor, physical, target, target_physical, move, old, new, num, den = raw.T
    return (((kind == CRITICAL) | ((kind == SUPER_EFFECTIVE) & (num > den))
        | ((kind == RESISTED) & (num < den)))
        & (actor >= 0) & (actor < 2) & (target == 1-actor)
        & (physical >= 0) & (physical < 6) & (target_physical >= 0) & (target_physical < 6)
        & (move > 0) & (old > new) & (new > 0) & _positive_ratio(num, den))


def _record(tape, actor_side, actor_physical, target_physical, move, write, data, *, eligible):
    """Shared actual-write body; returns valid-and-room, never latches UNKNOWN.

    Rows for one write are appended atomically: a two-control write with one
    remaining slot sets OVERFLOW/UNKNOWN without publishing a partial prefix.
    The caller must not invoke this from final state or move-selection metadata.
    """
    if not _check(tape, data):
        return tape, jnp.zeros_like(tape.count, bool)
    eligible = _boolean(eligible, tape.count.shape)
    if type(write) is not HitWrite:
        raise ValueError('literal typed direct-hit write required')
    try:
        actor_side, actor_physical, target_physical, move = (
            _word(x, tape.count.shape) for x in (actor_side, actor_physical, target_physical, move))
        w = _typed_write(write, tape.count.shape[0])
    except (TypeError, ValueError):
        return tape, jnp.zeros_like(tape.count, bool)
    target_side = 1-actor_side
    rows = jnp.arange(tape.count.shape[0]); safe = jnp.clip(actor_side, 0, 1)
    critical,effect=annotation_flags_device(w,data=data,move=move)
    need = critical.astype(jnp.int32) + effect.astype(jnp.int32)
    room = (tape.count >= 0) & (tape.count <= tape.words.shape[1]-need)
    valid = (life._valid(tape.cursor) & tape.cursor.initialized
        & (tape.stream_id > 0) & (tape.stream_id == tape.cursor.stream_id)
        & jnp.all(tape.engine_binding == data.engine_binding) & jnp.all(tape.source_binding == data.source_binding)
        & jnp.all(tape.move_binding == data.move_binding) & jnp.all(tape.producer_binding == data.producer_binding)
        & jnp.all(tape.cursor.engine_binding == data.engine_binding, axis=1)
        & (tape.turn == tape.cursor.turn) & (tape.cursor.phase == 0)
        & (actor_side >= 0) & (actor_side < 2) & (actor_physical >= 0) & (actor_physical < 6)
        & (target_physical >= 0) & (target_physical < 6)
        & (actor_physical == tape.cursor.physical_active[rows, safe])
        & (target_physical == tape.cursor.physical_active[rows, 1-safe])
        & (data.move_count > 1) & (move > 0) & (move < data.move_count)
        & (tape.count >= 0) & (tape.count <= tape.words.shape[1])
        & _valid_write(w,data=data,move=move))
    overflow = eligible & valid & ~room
    # Check only actual NEW slots. A clipped out-of-storage address would
    # inspect the previous prefix and must never hide capacity exhaustion.
    for offset in range(2):
        address = tape.count+offset
        inside = (address >= 0) & (address < tape.words.shape[1])
        index = jnp.clip(address, 0, tape.words.shape[1]-1)
        valid &= (need <= offset) | ~inside | jnp.all(tape.switch_details[rows, index] == 0, axis=1)
    use = eligible & valid & room
    facts = (move, w.old_hp, w.new_hp, w.numerator, w.denominator)
    for kind, flag in ((CRITICAL, critical), (SUPER_EFFECTIVE, effect & (w.numerator > w.denominator)),
                       (RESISTED, effect & (w.numerator < w.denominator))):
        tape = public._append(tape, kind, actor_side, actor_physical, target_side, target_physical,
            facts, use & flag)
    tape = tape._replace(refusal=tape.refusal | jnp.where(overflow, public.OVERFLOW, 0))
    return tape, valid & room


def record_direct_hit_controls_device(tape, actor_side, actor_physical, target_physical,
                                     move, write, data, *, eligible):
    """Diagnostic API: every requested observation retains UNKNOWN256."""
    result, _ = _record(tape, actor_side, actor_physical, target_physical,
        move, write, data, eligible=eligible)
    if type(tape) is public.CallbackTape:
        return result
    return _unknown(result, _boolean(eligible, tape.count.shape))


def _registered_bound(tape, context):
    # Local import avoids a cycle through the full adapter's control port.
    from . import engine_execution_completeness as accounting
    if (type(context) is not RegisteredContext or type(context.data) is not Data
            or type(context.accounting) is not accounting.Tables):
        raise ValueError('typed private original direct-hit/accounting context required')
    if type(tape) is not life.LifecycleTape:
        raise ValueError('registered direct-hit requires actual LifecycleTape')
    _check(tape, context.data)
    b = tape.count.shape[0]
    for value in context.accounting:
        life._typed(value, (8,), jnp.uint32)
    actual = accounting._context(tape)
    if actual is None:
        return jnp.zeros(b, bool)
    if type(actual.tables) is not accounting.Tables or type(actual.ledger) is not accounting.Ledger:
        raise ValueError('typed actual accounting tables/ledger required')
    for value in actual.tables:
        life._typed(value, (8,), jnp.uint32)
    ledger = actual.ledger
    for value, shape, dtype in ((ledger.stream_id,(b,),jnp.int32), (ledger.frame,(b,),jnp.int32),
            (ledger.opened,(b,),jnp.bool_), (ledger.inventory_binding,(8,),jnp.uint32)):
        life._typed(value, shape, dtype)
    bound = (accounting._validate(ledger, actual.tables) & accounting._tape(ledger, tape, actual.tables)
        & (ledger.stream_id == tape.stream_id) & (ledger.frame > 0) & ledger.opened
        & jnp.all(ledger.inventory_binding == context.accounting.inventory_binding))
    for old, new in zip(context.accounting, actual.tables):
        bound &= jnp.all(old == new)
    for name in ('engine_binding','source_binding','producer_binding'):
        bound &= jnp.all(getattr(context.accounting,name) == getattr(context.data,name))
        bound &= jnp.all(getattr(actual.tables,name) == getattr(tape,name))
    return bound


def record_registered_direct_hit_controls_device(tape, actor_side, actor_physical, target_physical,
                                                move, write, context, *, eligible):
    """PRIVATE source registration, never completeness or consumer credit.

    Only a valid literal write in the matching open actual accounting frame
    avoids this emitter's NEW UNKNOWN. Missing/bad context, unsupported writes
    and capacity loss remain UNKNOWN. No BEFORE refusal bit is removed.
    An exact root-owned ticket/span and BOTH genuine owners are still required.
    """
    bound = _registered_bound(tape, context)
    requested = _boolean(eligible, tape.count.shape)
    result, valid = _record(tape, actor_side, actor_physical, target_physical,
        move, write, context.data, eligible=requested & bound)
    return _unknown(result, requested & ~(bound & valid))


def record_direct_hit_controls(tape, actor_side, actor_physical, target_physical,
                              move, write, *, compiled, eligible):
    _host()
    return record_direct_hit_controls_device(tape, actor_side, actor_physical, target_physical,
        move, write, check_compiled(compiled), eligible=eligible)
