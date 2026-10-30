"""Causal engine receipts -> the original FULL-v4 raw-control transport.

Host compilation runs the pinned wire writer over finite lexical schemas and
every supported original move-table entry. It retains their exact control
words/order, without introducing a second move-name classifier. The trusted
device primitive fills only documented dynamic address, clock and HP fields.
It never reads a final BattleState or reconstructs a raw line.

This is a format/component adapter, not complete engine instrumentation. Missing
callbacks and UNKNOWN remain missing. Frames are consumed once by the owning
driver; this lower primitive does not attest their origin or prevent replay.
Expanded frames drain through fixed64-event chunks without truncation. A later
chunk failure refuses the WHOLE frame, including its earlier accepted chunks.
All scientific/full-encoder/B4 qualification flags remain false.
"""
from dataclasses import dataclass
import hashlib
from typing import NamedTuple
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from . import engine_lifecycle_event_device as causal
from . import engine_public_events as engine, public_context, public_inputs as inputs
from . import public_wire, public_refresh_stream, public_toxic, v4_encoder

SCHEMA = 'pokezero.engine_full_v4_raw_device.partial.v1'
_NAMES = ('source_binding', 'species_binding', 'ability_binding', 'move_binding',
    'item_binding', 'context_binding', 'field_binding', 'active_binding',
    'damage_binding', 'type_binding', 'sleep_binding', 'toxic_binding', 'tendency_binding')
Bindings = NamedTuple('Bindings', [(name, object) for name in _NAMES])


class Tables(NamedTuple):
    causal: causal.Tables
    templates: object
    length: object
    main: object
    kinds: object
    moves: object
    conditions: object
    statuses: object
    cures: object
    lum: object
    namespaces: Bindings
    binding: object


@dataclass(frozen=True, eq=False)
class Compiled:
    catalog: object
    causal: object
    device: Tables


_COMPILED = weakref.WeakKeyDictionary()


def compile_adapter(metadata, projector, *, catalog):
    """Host-only original full catalog + reviewed causal projector boundary.

    Strings exist only here, at static schema compilation. Each engine move row
    keeps the original canonical public ID and target classification. A source
    schema which cannot be represented remains unsupported, never relabelled.
    """
    v4_encoder.check_catalog(catalog)
    causal.base.check_projector(projector)
    if projector.catalog.public_moves is not catalog.moves:
        raise ValueError('original engine/public FULL catalog move edge required')
    simple = public_context.compile_context(catalog.source, catalog.abilities, catalog.moves)
    lower = causal.compile_projector(metadata, projector, species_catalog=catalog.species,
        source=catalog.source, context_tables=simple, status_events=True, items=catalog.items)
    kwargs = v4_encoder.wire_kwargs(catalog)
    name = catalog.species.names[1]
    ident = ('p1a: '+name, 'p2a: '+name)
    arrivals = [f'|switch|{ident[s]}|{name}, L80, M|100/100' for s in (0, 1)]
    prior, projections = public_wire.project_lines(public_wire.WireState(), arrivals+['|turn|1'], **kwargs)
    # Namespace words are obtained from an actual FULL writer projection, not
    # substituted on a basic chunk which lacks those consumers.
    ns = Bindings(*(jnp.array(catalog.source.device.binding if key == 'source_binding'
        else getattr(projections[0], key), jnp.uint32) for key in _NAMES))
    templates = [np.zeros((1, 16), np.int32)]
    lengths, mains = [0], [0]
    allowed = {int(k) for k in (inputs.Kind.RAW_BEGIN, inputs.Kind.SWITCH_PRELUDE,
        inputs.Kind.MUDSHOT, inputs.Kind.NONPROC, inputs.Kind.TENDENCY,
        inputs.Kind.ACTIVE, inputs.Kind.TYPES, inputs.Kind.FIELD,
        inputs.Kind.DAMAGE_CONTROL, inputs.Kind.SLEEP, inputs.Kind.TOXIC, inputs.Kind.STATUS)}

    def add(line, main_kind, before=prior):
        try:
            if isinstance(line, tuple):
                _, projections = public_wire.project_lines(before, line, **kwargs)
            else:
                _, projection = public_wire.project_line(before, line, **kwargs)
                projections = (projection,)
            words = np.array([e.words() for p in projections for e in p.events], np.int32)
            indices = np.flatnonzero(words[:, 0] == int(main_kind))
            if (len(indices) != 1 or not 1 <= len(words) <= 64
                    or any(int(k) not in allowed | {int(main_kind)} for k in words[:, 0])):
                return 0
            templates.append(words)
            lengths.append(len(words)); mains.append(int(indices[0]))
            return len(templates)-1
        except ValueError:
            return 0

    kinds = np.zeros(33, np.int32)
    kinds[inputs.Kind.SWITCH] = add(arrivals[0], inputs.Kind.SWITCH, public_wire.WireState())
    kinds[inputs.Kind.TURN] = add('|turn|2', inputs.Kind.TURN)
    kinds[inputs.Kind.ACTION_CHUNK_END] = add('|', inputs.Kind.ACTION_CHUNK_END)
    kinds[inputs.Kind.UPKEEP] = add('|upkeep', inputs.Kind.UPKEEP)
    kinds[inputs.Kind.FAINT] = add('|faint|'+ident[0], inputs.Kind.FAINT)
    conditions = np.array([add(f'|-damage|{ident[0]}|71/100'+suffix, inputs.Kind.CONDITION)
        for suffix in ('', ' slp', ' par', ' brn', ' frz', ' psn', ' tox')]
        + [add(f'|-damage|{ident[0]}|0 fnt', inputs.Kind.CONDITION)], np.int32)
    tokens=('', 'slp', 'par', 'brn', 'frz', 'psn', 'tox')
    statuses=np.array([0]+[add(f'|-status|{ident[0]}|{token}',inputs.Kind.STATUS) for token in tokens[1:]],np.int32)
    cures=np.array([0]+[add(f'|-curestatus|{ident[0]}|{token}|[msg]',inputs.Kind.STATUS) for token in tokens[1:]],np.int32)
    lum = np.zeros((7, 2), np.int32)
    if bool(lower.device.lum_events):
        # Effective Gen3 inherited eatItem/onEat emits these genuine source
        # lines in this order. Keep every raw begin/closure from the original
        # writer, including multiple semantic events in one consume receipt.
        for old_status in range(7):
            for confusion in range(2):
                if old_status == 0 and not confusion:
                    continue
                lines = [f'|-enditem|{ident[0]}|Lum Berry|[eat]']
                if old_status:
                    lines.append(f'|-curestatus|{ident[0]}|{tokens[old_status]}|[msg]')
                if confusion:
                    lines.append(f'|-end|{ident[0]}|confusion')
                lum[old_status, confusion] = add(tuple(lines), inputs.Kind.ITEM)
        if not np.all(lum[1:]) or lum[0, 1] == 0:
            raise ValueError('FULL writer lacks exact Lum raw schemas')
    moves = np.zeros(len(catalog.moves.names), np.int32)
    targets = {}
    for pid, target in zip(projector.catalog.ids, projector.catalog.targets):
        if pid:
            if pid in targets and targets[pid] != target:
                raise ValueError('canonical public move has conflicting source targets')
            targets[pid] = target
    for pid, target in targets.items():
        if target in (0, 1):
            moves[pid] = add(f'|move|{ident[0]}|{catalog.moves.names[pid]}|{ident[target]}', inputs.Kind.MOVE)
    if not all(kinds[k] for k in (1, 2, 3, 6, 12)) or not np.all(conditions) or not np.all(statuses[1:]) or not np.all(cures[1:]):
        raise ValueError('FULL writer lacks required causal lifecycle schemas')
    width = max(lengths)
    packed = np.zeros((len(templates), width, 16), np.int32)
    for i, words in enumerate(templates): packed[i, :len(words)] = words
    payload = packed.tobytes()+np.array(lengths, np.int32).tobytes()+np.array(mains, np.int32).tobytes()
    payload += kinds.tobytes()+moves.tobytes()+conditions.tobytes()+statuses.tobytes()+cures.tobytes()+lum.tobytes()+np.asarray(catalog.binding).tobytes()+SCHEMA.encode()
    binding = jnp.array(np.frombuffer(hashlib.sha256(payload).digest(), dtype='>u4').astype(np.uint32))
    tables = Tables(lower.device, jnp.array(packed), jnp.array(lengths, jnp.int32),
        jnp.array(mains, jnp.int32), jnp.array(kinds), jnp.array(moves), jnp.array(conditions),
        jnp.array(statuses), jnp.array(cures), jnp.array(lum), ns, binding)
    result = Compiled(catalog, lower, tables)
    _COMPILED[result] = (catalog, lower, tables, tuple(jax.tree.leaves(tables)))
    return result


def check_adapter(compiled):
    """Host-only guard: call on EVERY owning driver invocation, outside JIT."""
    record = _COMPILED.get(compiled) if type(compiled) is Compiled else None
    if (record is None or compiled.catalog is not record[0] or compiled.causal is not record[1]
            or compiled.device is not record[2] or type(compiled.device) is not Tables):
        raise ValueError('original compiled FULL raw adapter required')
    v4_encoder.check_catalog(compiled.catalog)
    causal.check_projector(compiled.causal)
    leaves = tuple(jax.tree.leaves(compiled.device))
    if (len(leaves) != len(record[3]) or any(a is not b or not isinstance(a, jax.Array)
            or isinstance(a, jax.core.Tracer) for a, b in zip(leaves, record[3]))):
        raise ValueError('original immutable FULL raw schema edges required')
    return compiled.device


class StreamState(NamedTuple):
    causal: causal.StreamState
    sequence: object
    live: object
    species: object               # ONLY already observed public arrival IDs
    status: object                # ONLY prior accepted public status facts


def register(stream_ids, owners, *, compiled):
    check_adapter(compiled)
    return StreamState(causal.register(stream_ids, owners, compiled=compiled.causal),
        jnp.zeros_like(stream_ids), jnp.zeros((stream_ids.shape[0], 2), bool),
        jnp.zeros((stream_ids.shape[0], 2, 6), jnp.int32),
        jnp.zeros((stream_ids.shape[0], 2, 6), jnp.int32))


class FrameChunks(NamedTuple):
    words: object               # [bounded chunks,B,64,16]
    valid: object               # [bounded chunks,B,64]
    more: object                # [bounded chunks,B]


class Projection(NamedTuple):
    state: StreamState
    chunk: inputs.EventChunk
    error: object
    refusal: object
    component_admitted: object
    full_line_consumers_complete: object
    full_encoder_admitted: object
    B4_qualified: object
    frames: FrameChunks


def _commit(ok, old, new):
    return jax.tree.map(lambda a, b: jnp.where(ok.reshape((ok.shape[0],)+(1,)*(a.ndim-1)), b, a), old, new)


def _commit_public(ok, before, after):
    # Refresh contains global source bindings alongside B-row state; use its
    # owning reducer's commit rather than treating global words as batch rows.
    refresh = public_refresh_stream.commit(ok, before.context.refresh, after.context.refresh)
    candidate = _commit(ok, before._replace(context=before.context._replace(refresh=None)),
        after._replace(context=after.context._replace(refresh=None)))
    return candidate._replace(context=candidate.context._replace(refresh=refresh))


def project_device(state, tape, before_registry, after_registry, tables):
    """Trusted, pure device closure; genuine same-core frame delivered ONCE.

    The causal scan resolves BEFORE identities. This second device scan expands
    only its accepted typed events, retaining public live/clock state. Physical
    party IDs never enter EventChunk. Dynamic inputs are trusted; static holder
    and immutable-edge checks must have run at the outer host boundary.
    """
    if type(state) is not StreamState or type(tables) is not Tables:
        raise ValueError('typed FULL raw stream and static tables required')
    lower = causal.project_device(state.causal, tape, before_registry, after_registry, tables.causal)
    b = tape.stream_id.shape[0]; rows = jnp.arange(b)
    causal.base._shape(state.sequence, (b,), jnp.int32)
    causal.base._shape(state.live, (b, 2), jnp.bool_)
    causal.base._shape(state.species, (b, 2, 6), jnp.int32)
    causal.base._shape(state.status, (b, 2, 6), jnp.int32)
    error = jnp.where((lower.error == 0) & (state.sequence >= 0), 0,
        jnp.where(lower.error != 0, lower.error, int(inputs.Error.SEQUENCE)))
    width = tables.templates.shape[1]
    # At most64 causal receipts, each expanding to at mostwidth events. This
    # is a proven finite storage bound, not a smaller roster/cap/truncation.
    capacity = 64*width

    def step(carry, index):
        words, count, sequence, turn, phase, slots, live, species, statuses, error, overflow = carry
        semantic = lower.chunk.words[:, index]; kind = semantic[:, 0]
        present = lower.chunk.valid[:, index]
        move = kind == int(inputs.Kind.MOVE); condition = kind == int(inputs.Kind.CONDITION)
        arrival = kind == int(inputs.Kind.SWITCH); faint = kind == int(inputs.Kind.FAINT)
        key = tables.kinds[jnp.clip(kind, 0, 32)]
        mid = semantic[:, 9]
        key = jnp.where(move, tables.moves[jnp.clip(mid, 0, tables.moves.shape[0]-1)], key)
        ci = jnp.where(semantic[:, 12] == 1, 7, semantic[:, 11])
        key = jnp.where(condition, tables.conditions[jnp.clip(ci, 0, 7)], key)
        status_event = kind == int(inputs.Kind.STATUS)
        lum_event = kind == int(inputs.Kind.ITEM)
        # This is a SAME-index causal receipt join, not a final-state join:
        # status-enabled lower mode forbids RAW_BEGIN and emits exactly one
        # semantic per accepted input. It never drops a receipt. The actual
        # preceding-MOVE/HP -> wake-CURE regression locks this alignment.
        old_engine_status = tape.words[:,index,5]
        old_public_status = jnp.array([0,2,3,5,6,1,4],jnp.int32)[jnp.clip(old_engine_status,0,6)]
        status_key=jnp.where(semantic[:,9]>0,tables.statuses[jnp.clip(semantic[:,9],0,6)],tables.cures[old_public_status])
        key=jnp.where(status_event,status_key,key)
        lum_old_engine = tape.words[:,index,6]
        lum_old_public = jnp.array([0,2,3,5,6,1,4],jnp.int32)[jnp.clip(lum_old_engine,0,6)]
        lum_key = tables.lum[lum_old_public,jnp.clip(tape.words[:,index,8],0,1)]
        key=jnp.where(lum_event,lum_key,key)
        size = tables.length[key]; main = tables.main[key]
        enough = (count <= capacity-size) & (sequence <= 2**31-1-size)
        overflow |= present & (error == 0) & (count > capacity-size)
        good = (key > 0) & enough
        error = jnp.where(present & (error == 0) & ~good, int(inputs.Error.PAYLOAD), error)
        schema = tables.templates[key]
        subject = jnp.where(move, semantic[:, 4], semantic[:, 6])
        safe = jnp.clip(subject, 0, 1)
        new_slot = jnp.where(move, semantic[:, 5], semantic[:, 7])
        safe_slot = jnp.clip(new_slot, 0, 5)
        prior_species = species[rows, safe, safe_slot]
        collision = jnp.any((species[rows, safe] == semantic[:, 9, None])
            & (jnp.arange(6)[None, :] != new_slot[:, None]), axis=1)
        # The engine has no public nickname receipt. Two distinct arrivals of
        # the same species would have the same species-form raw ident, which
        # the pinned writer reuses rather than assigning a new public slot.
        # Refuse that ambiguity and an unobserved forme change atomically.
        identity = ~arrival | (~collision & ((prior_species == 0) | (prior_species == semantic[:, 9])))
        # Old status is a causal receipt fact, and must agree with the BEFORE
        # public ledger. An enum-valid false WAKE/THAW cannot rewrite CURE
        # lexical metadata or silently erase a different observed condition.
        previous_status=statuses[rows,safe,safe_slot]
        identity &= ~status_event | (old_public_status==previous_status)
        identity &= ~lum_event | ((lum_old_public==previous_status) & live[rows,safe])
        error = jnp.where(present & (error == 0) & ~identity, int(inputs.Error.IDENTITY), error)
        use = present & (error == 0)
        next_species = species.at[rows, safe, safe_slot].set(
            jnp.where(arrival, semantic[:, 9], prior_species))
        next_status=jnp.where(arrival,semantic[:,15],jnp.where(condition,semantic[:,11],
            jnp.where(status_event,semantic[:,9],jnp.where(faint|lum_event,0,previous_status))))
        status_written=arrival|condition|status_event|faint|lum_event
        next_statuses=statuses.at[rows,safe,safe_slot].set(jnp.where(status_written,next_status,previous_status))
        current_slots = slots.at[rows, safe].set(jnp.where(arrival, new_slot, slots[rows, safe]))
        current_live = live.at[rows, safe].set(jnp.where(arrival, True, live[rows, safe]))

        # Schema sides are actor/victim0 and the opposite side1. Slots and
        # encounters come ONLY from this causal public ledger, never party IDs.
        def side_map(value): return jnp.where(value >= 0, jnp.where(value == 0, subject[:, None], 1-subject[:, None]), -1)
        def address(value):
            side = side_map(value)
            slot = current_slots[rows[:, None], jnp.clip(side, 0, 1)]
            return side, slot
        result = schema
        result = result.at[:, :, 1].set(sequence[:, None]+jnp.arange(width)[None, :]+1)
        post = jnp.arange(width)[None, :] >= main[:, None]
        result = result.at[:, :, 2].set(jnp.where(post, semantic[:, 2, None], turn[:, None]))
        result = result.at[:, :, 3].set(jnp.where(post, semantic[:, 3, None], phase[:, None]))
        for side_col, slot_col in ((4, 5), (6, 7)):
            side, slot = address(schema[:, :, side_col])
            result = result.at[:, :, side_col].set(side)
            result = result.at[:, :, slot_col].set(jnp.where(schema[:, :, slot_col] >= 0, slot, -1))
        kinds = schema[:, :, 0]
        target_side = result[:, :, 6]
        target_enc = current_slots[rows[:, None], jnp.clip(target_side, 0, 1)]+1
        addressed = ((kinds == int(inputs.Kind.ACTIVE)) | (kinds == int(inputs.Kind.TYPES))
            | (kinds == int(inputs.Kind.SLEEP)) | (kinds == int(inputs.Kind.STATUS))
            | (kinds == int(inputs.Kind.ITEM)))
        result = result.at[:, :, 8].set(jnp.where(addressed, target_enc, result[:, :, 8]))
        mud = kinds == int(inputs.Kind.MUDSHOT)
        ms, sl = address(schema[:, :, 9]); enc = jnp.where(ms >= 0, sl+1, 0)
        queue = schema[:, :, 12] == 1
        # QUEUE is a lexical declaration flag even without a live target. The
        # source then emits QUEUE + none-address; it does not turn it into a
        # different lexical fact or clear the packed RAW_BEGIN queue bit.
        missing_queue = queue & ~current_live[rows[:, None], jnp.clip(ms, 0, 1)]
        ms = jnp.where(missing_queue, -1, ms); sl = jnp.where((ms >= 0), sl, -1)
        enc = jnp.where(ms >= 0, enc, 0)
        for col, value in ((9, ms), (10, sl), (11, enc), (12, queue.astype(jnp.int32))):
            result = result.at[:, :, col].set(jnp.where(mud, value, result[:, :, col]))
        non = kinds == int(inputs.Kind.NONPROC)
        targeted = non & ((schema[:, :, 8] == 1) | (schema[:, :, 8] == 2))
        berry = non & ((schema[:, :, 15] & 2) != 0)
        hp = jnp.where(arrival, semantic[:, 13], semantic[:, 9])
        maximum = jnp.where(arrival, semantic[:, 14], semantic[:, 10])
        for col, value in ((9, subject), (10, new_slot), (11, semantic[:, 8])):
            result = result.at[:, :, col].set(jnp.where(targeted | berry, value[:, None], result[:, :, col]))
        # Berry NOTE addresses are targeted, but contain no condition HP data.
        for col, value in ((12, hp), (13, maximum)):
            result = result.at[:, :, col].set(jnp.where(targeted, value[:, None], result[:, :, col]))
        toxic = kinds == int(inputs.Kind.TOXIC)
        result = result.at[:, :, 9].set(jnp.where(toxic, semantic[:, 8, None], result[:, :, 9]))
        result = result.at[:, :, 11].set(jnp.where(toxic & condition[:, None],
            jnp.where(semantic[:, 12, None] == 1, -1, hp[:, None]), result[:, :, 11]))
        result = result.at[:, :, 12].set(jnp.where(toxic & condition[:, None], maximum[:, None], result[:, :, 12]))
        tendency = kinds == int(inputs.Kind.TENDENCY)
        for col, value, active in ((9, semantic[:, 9], arrival), (10, semantic[:, 9], move),
                (11, semantic[:, 6], move), (12, semantic[:, 2], kind == int(inputs.Kind.TURN))):
            result = result.at[:, :, col].set(jnp.where(tendency & active[:, None], value[:, None], result[:, :, col]))
        raw = kinds == int(inputs.Kind.RAW_BEGIN)
        for col, value in ((13, hp), (14, maximum), (15, semantic[:, 11])):
            result = result.at[:, :, col].set(jnp.where(raw & condition[:, None], value[:, None], result[:, :, col]))
        semantic = semantic.at[:, 1].set(sequence+main+1)
        result = result.at[rows, main].set(semantic)

        def write(buffer, offset):
            valid = use & (offset < size)
            at = jnp.minimum(count+offset, capacity-1)
            return buffer.at[rows, at].set(jnp.where(valid[:, None], result[:, offset], buffer[rows, at])), None
        words, _ = jax.lax.scan(write, words, jnp.arange(width))
        live_next = current_live.at[rows, safe].set(jnp.where(faint, False, current_live[rows, safe]))
        return (words, count+jnp.where(use, size, 0), sequence+jnp.where(use, size, 0),
            jnp.where(use, semantic[:, 2], turn), jnp.where(use, semantic[:, 3], phase),
            jnp.where(use[:, None], current_slots, slots), jnp.where(use[:, None], live_next, live),
            jnp.where(use[:, None, None], next_species, species),
            jnp.where(use[:, None, None],next_statuses,statuses),error, overflow), None

    init = (jnp.zeros((b, capacity, 16), jnp.int32), jnp.zeros(b, jnp.int32), state.sequence,
        state.causal.cursor.turn, state.causal.cursor.phase, state.causal.cursor.public_slot,
        state.live, state.species, state.status,error, jnp.zeros(b, bool))
    (words, count, sequence, _, _, _, live, species,statuses,error, overflow), _ = jax.lax.scan(step, init, jnp.arange(64))
    ok = error == 0
    next_state = _commit(ok, state, StreamState(lower.state, sequence, live, species,statuses))
    next_state = next_state._replace(causal=next_state.causal._replace(admitted=state.causal.admitted & ok,
        cursor=next_state.causal.cursor._replace(admitted=state.causal.cursor.admitted & ok)))
    valid = (jnp.arange(capacity)[None, :] < count[:, None]) & ok[:, None]
    no = jnp.zeros(b, bool)
    packed = jnp.where(valid[:, :, None], words, 0).reshape((b,width,64,16)).transpose(1,0,2,3)
    masks = valid.reshape((b,width,64)).transpose(1,0,2)
    more = (count[None,:] > (jnp.arange(width)[:,None]+1)*64) & ok[None,:]
    frames = FrameChunks(packed,masks,more)
    # Compatibility view of the FIRST chunk. Consumers must drain frames;
    # needs_more forbids encoding this prefix as a complete observation.
    chunk = inputs.EventChunk(tape.stream_id, packed[0], masks[0], more[0], *tables.namespaces)
    return Projection(next_state, chunk, error, lower.refusal | jnp.where(overflow, engine.OVERFLOW, 0), ok, no, no, no, frames)


def project_frame(state, frame, tables):
    if type(frame) is not causal.life.LifecycleResult:
        raise ValueError('genuine same-core lifecycle result required')
    return project_device(state, frame.callbacks, frame.before_registry, frame.after_registry, tables)


class Consumed(NamedTuple):
    projection: Projection
    public: v4_encoder.State


def project_and_fold(state, frame, tables, public, catalog):
    """Trusted device path into the ORIGINAL full retained public consumers."""
    projection = project_frame(state, frame, tables)
    def fold(carry, data):
        current,error=carry
        words,valid,more=data
        def consume(_):
            chunk=projection.chunk._replace(words=words,valid=valid,more=more)
            candidate,failed=v4_encoder.fold_chunk(current,chunk,catalog)
            return candidate,jnp.where(error!=0,error,failed)
        # Padding slots are not protocol input and must not finalize a prefix.
        return jax.lax.cond(jnp.any(valid),consume,lambda _:carry,None),None
    (candidate,fold_error),_=jax.lax.scan(fold,(public,jnp.zeros_like(projection.error)),projection.frames)
    visibility = jnp.where(jnp.arange(2)[None, :] == state.causal.owner[:, None],
        public_toxic.EXACT, public_toxic.PERCENTAGE)
    same = ((public.cursor.sequence == state.sequence) & (public.cursor.stream_id == state.causal.cursor.stream_id)
        & (public.cursor.turn == state.causal.cursor.turn) & (public.cursor.phase == state.causal.cursor.phase)
        & jnp.all(public.context.toxic.visibility == visibility, axis=1)
        & jnp.all(public.mons.species == state.species, axis=(1, 2))
        & jnp.all(public.mons.status == state.status,axis=(1,2)))
    error = jnp.where(projection.error != 0, projection.error,
        jnp.where(same, fold_error, int(inputs.Error.STREAM)))
    ok = error == 0
    next_state = _commit(ok, state, projection.state)
    next_state = next_state._replace(causal=next_state.causal._replace(admitted=state.causal.admitted & ok,
        cursor=next_state.causal.cursor._replace(admitted=state.causal.cursor.admitted & ok)))
    projection = projection._replace(state=next_state, error=error, component_admitted=ok,
        chunk=projection.chunk._replace(words=jnp.where(ok[:, None, None], projection.chunk.words, 0),
            valid=projection.chunk.valid & ok[:, None],more=projection.chunk.more & ok),
        frames=FrameChunks(jnp.where(ok[None,:,None,None],projection.frames.words,0),
            projection.frames.valid & ok[None,:,None],projection.frames.more & ok[None,:]))
    return Consumed(projection, _commit_public(ok, public, candidate))


def consume(state, frame, public, *, compiled):
    """Checked host invocation; suitable outer boundary for a cached closure."""
    check_adapter(compiled)
    return project_and_fold(state, frame, compiled.device, public, compiled.catalog)
