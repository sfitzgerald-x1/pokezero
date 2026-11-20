"""Trace-only numeric witnesses and frozen pre-encoder arithmetic for ENC.

JAX imports are deferred. Arithmetic substitutions must cover a fresh explicit
lower/compile operation; existing cached jitted closures must not be reused.
"""
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import ast
import hashlib
import json
from pathlib import Path
import sys

from gpu_engine.encoder_regression_witness import (
    ENCODER_SHAPES, EVENT_FIELDS, EXPECTED_GUARDS, HISTORY_HP,
    HISTORY_MAXIMUM, MAPPING, NEXT_SLOT, TURN,
)


LEGACY_SOURCE = '24f39abaa478fd11d27188353ccb0fe6a3f471f0'
LEGACY_BODY_SHA256 = {
    '_add': '232da7af72fcefc99dc983e2cfba982a79933435bf888027160aecc27943ce71',
    'observe': '26cef3902bc76ac204e91e882894f230d790fda6b11cded8e325866ab783c4f5',
    'direct': '6ffab46cc4ecefd60293faf2826a5ee99c05cbcbc4a6e5d456878b8a9153461b',
    'damage': '43ab5766d248ae8e886f446a12cf5dc6ad02d3e64964fa9a115912aba0452ed0',
    'spikes': '4e8f620d5aa1d11255ac9e5121b06059ac0aed2ac2fced6bb72535241521dbed',
}


def encoder_source_identity():
    root = Path(__file__).parent
    names = ('encoder_regression_trace.py', 'encoder_regression_witness.py', 'v2e_regress.py',
             'encoder_numeric.py', 'engine_device_damage_history.py',
             'engine_device_public_hooks.py', 'engine_device_public_port.py')
    return dict(legacy=legacy_source_identity(),
                files={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names})


def legacy_source_identity():
    tree = ast.parse(Path(__file__).read_text())
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_legacy_functions')
    actual = {n.name: _ast_identity(n)
              for n in factory.body if isinstance(n, ast.FunctionDef)}
    if actual != LEGACY_BODY_SHA256:
        raise ValueError('frozen ENC arithmetic differs from the recorded legacy source')
    return dict(commit=LEGACY_SOURCE, ast_format='structural-fields-v1', bodies=dict(actual))


def _ast_identity(node):
    """Serialize fields explicitly; ast.dump formatting changes across Python.

    Python 3.12 added FunctionDef.type_params. An empty type-parameter list
    has no meaning for these baseline functions; nonempty lists stay hashed.
    All other fields, including empty lists and None, remain in the proof.
    """
    def encode(value):
        if value is Ellipsis:return {'literal': 'ellipsis'}
        if isinstance(value, ast.AST):
            return [type(value).__name__, [[name, encode(field)] for name, field in ast.iter_fields(value)
                    if name != 'type_params' or field]]
        if isinstance(value, list):return [encode(item) for item in value]
        return value
    return hashlib.sha256(json.dumps(encode(node), separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


@dataclass
class TraceBuild:
    out: object
    events: tuple
    schema: tuple
    pre: dict


@dataclass
class EncoderPrograms:
    corrected: object
    legacy: object
    compare: object
    source_identity: dict
    schema: tuple = ()

    def execute(self, leaves):
        actual = self.corrected(leaves)
        legacy = self.legacy(leaves)
        a = actual.pop('_encoder');b = legacy.pop('_encoder')
        outside, inputs = self.compare(a['outputs'], b['outputs'], a['events'], b['events'], a['pre'], b['pre'])
        actual['encoder'] = dict(actual={n: a['outputs'][n] for n in ENCODER_SHAPES},
                                 pre=a['pre'], events=a['events'], legacy_digests=legacy['digests'],
                                 outside_equal=outside, event_inputs_equal=inputs)
        return actual


def compile_encoder_programs(build_program, example_leaves, *, source_identity):
    """Compile distinct, non-donating programs while arithmetic is selected.

    build_program(mode) returns a fresh function carrying Stepper's existing
    recorders. The caller separately verifies the immutable event schemas.
    No full output tree is copied to the host by execute().
    """
    import jax
    import jax.numpy as j
    programs = {}
    for mode in ('corrected', 'legacy'):
        with encoder_arithmetic(mode):
            programs[mode] = jax.jit(build_program(mode), donate_argnums=()).lower(example_leaves).compile()

    def compare(a, b, events_a, events_b, pre_a, pre_b):
        lanes = a['trackers.damage_history.taken'].shape[0]

        def equal(x, y):
            if x.shape != y.shape or x.dtype != y.dtype:
                raise ValueError('corrected/legacy ENC output shape/dtype mismatch')
            per_lane = bool(x.ndim and x.shape[0] == lanes)
            if j.issubdtype(x.dtype, j.inexact):
                x = jax.lax.bitcast_convert_type(x, j.uint8)
                y = jax.lax.bitcast_convert_type(y, j.uint8)
            same = x == y
            if per_lane:
                return same.reshape(lanes, -1).all(axis=1)
            return j.broadcast_to(same.all(), (lanes,))

        if set(a) != set(b):raise ValueError('corrected/legacy ENC output names differ')
        outside = j.ones((lanes,), j.bool_)
        for name in sorted(a):
            # Seven-leaf shapes/dtypes are validated by recorder and host fold.
            if name not in ENCODER_SHAPES:outside = outside & equal(a[name], b[name])
        inputs = j.ones((lanes,), j.bool_)
        if len(events_a) != len(events_b) or set(pre_a) != set(pre_b):
            raise ValueError('corrected/legacy ENC evidence structures differ')
        for name in sorted(pre_a):inputs = inputs & equal(pre_a[name], pre_b[name])
        for left, right in zip(events_a, events_b):
            if set(left) != set(right):raise ValueError('corrected/legacy ENC event fields differ')
            for name in sorted(left):inputs = inputs & equal(left[name], right[name])
        return outside, inputs

    return EncoderPrograms(programs['corrected'], programs['legacy'], jax.jit(compare), source_identity)


def _legacy_functions():
    """Baseline bodies, unchanged apart from their enclosing private scope."""
    import jax
    import jax.numpy as j
    from gpu_engine import engine_device_damage_history as damage_history
    from gpu_engine import engine_device_public_hooks as hooks
    from gpu_engine import engine_device_public_port as port
    _li, tr = hooks._li, hooks.tr
    _r, _get, _write, feature = hooks._r, hooks._get, hooks._write, hooks.feature
    _seat, _set = port._seat, port._set

    def _add(pair,value):
        high,low=pair[...,0],pair[...,1]
        total=jax.lax.optimization_barrier(high+value)
        recovered=jax.lax.optimization_barrier(total-high)
        error=jax.lax.optimization_barrier((high-(total-recovered))+(value-recovered))
        correction=jax.lax.optimization_barrier(low+error)
        new_high=jax.lax.optimization_barrier(total+correction)
        new_low=jax.lax.optimization_barrier(correction-(new_high-total))
        return j.where((value>0)[...,None],j.stack((new_high,new_low),-1),pair)

    def observe(history,state,use,percent,*,record_damage=True):
        r=j.arange(len(state.hp))[:,None];s=j.arange(2)[None,:]
        hp=state.hp[r,s,state.active];maximum=state.max_hp[r,s,state.active]
        exact=j.maximum(0,history.hp-hp).astype(j.float32)/j.maximum(history.maximum,1)
        shown=j.maximum(0,percent(history.hp,history.maximum)-percent(hp,maximum)).astype(j.float32)/100
        values=j.stack((shown,exact),-1)*j.asarray(record_damage,j.bool_)
        values=j.where(use[:,None,None],values,0)
        taken=_add(history.taken,values)
        return history._replace(hp=j.where(use[:,None],hp,history.hp),
            maximum=j.where(use[:,None],maximum,history.maximum),taken=taken)

    def direct(history,side,shown,exact,use):
        r=j.arange(len(use));value=j.stack((shown,exact),-1)
        old=history.dealt[r,side]
        return history._replace(dealt=history.dealt.at[r,side].set(
            j.where(use[:,None,None],_add(old,value),old)))

    def damage(c,state,side,dealt,taken,use):
        r=_r(state);foe=1-side;p=_li.take(state.active,r,foe);slot=j.clip(_li.take(c.trackers.physical_to_public,r,foe,p),0,5)
        previous=_li.take(c.trackers.hp_percent,r,foe,slot)
        current=tr._percent(_get(state,'hp',foe),_get(state,'max_hp',foe))
        shown=j.maximum(0,previous-current).astype(j.float32)/100.0
        exact=taken.astype(j.float32)/j.maximum(_get(state,'max_hp',foe),1).astype(j.float32)
        if c.trackers.damage_history is not None:
            return c._replace(trackers=c.trackers._replace(damage_history=damage_history.direct(
                c.trackers.damage_history,side,shown,exact,use & (taken>0))))
        c=c._replace(trackers=_write(c.trackers,'damage_exact_dealt',side,_li.take(state.active,r,side),exact,use))
        c=c._replace(trackers=_write(c.trackers,'damage_exact_taken',foe,p,exact,use))
        c=feature(c,side,_li.take(state.active,r,side),'last_damage_dealt',shown,use)
        return feature(c,foe,p,'last_damage_taken',shown,use)

    def spikes(t,before,after,side,use,percent):
        h=t.public_port
        if h is None:return t
        r=j.arange(len(use));s=_seat(side,len(use));p=_li.take(before.active,r,s)
        hp=_li.take(before.hp,r,s,p);maximum=_li.take(before.max_hp,r,s,p);new=_li.take(after.hp,r,s,p)
        yes=use&(maximum>0)&(hp>new)&(hp>=0)&(hp<=maximum)&(new>=0)
        shown=j.maximum(0,percent(hp,maximum)-percent(new,maximum)).astype(j.float32)/100
        exact=j.maximum(0,hp-new).astype(j.float32)/j.maximum(maximum,1)
        value=j.where(yes[:,None],j.stack((shown,exact),-1),0)
        old=_li.take(h.hazards,r,s);new=_add(old,value)
        return t._replace(public_port=h._replace(hazards=_set(h.hazards,s,new,yes)))

    return ((damage_history, 'observe', observe), (damage_history, 'direct', direct),
            (hooks, 'damage', damage), (port, 'spikes', spikes))


@contextmanager
def encoder_arithmetic(mode):
    """Serialize, substitute for one fresh compilation, and always restore."""
    if mode not in ('corrected', 'legacy'):
        raise ValueError(f'unknown encoder arithmetic mode {mode}')
    import jax
    from gpu_engine.v2e_modules import _substituting
    with _substituting:
        legacy_source_identity()
        substitutions = _legacy_functions() if mode == 'legacy' else ()
        saved = [(module, name, getattr(module, name)) for module, name, _ in substitutions]
        try:
            jax.clear_caches()
            for module, name, value in substitutions:setattr(module, name, value)
            yield
        finally:
            for module, name, value in reversed(saved):setattr(module, name, value)
            jax.clear_caches()


def encoder_pre(trackers):
    """References to immutable input arrays; never retain the mutable Context."""
    h, port = trackers.damage_history, trackers.public_port
    if h is None or port is None:
        raise ValueError('ENC requires damage history and public port')
    return {
        'trackers.damage_history.taken': h.taken,
        'trackers.damage_history.dealt': h.dealt,
        'trackers.damage_exact_taken': trackers.damage_exact_taken,
        'trackers.damage_exact_dealt': trackers.damage_exact_dealt,
        "trackers.mon_features['last_damage_taken']": trackers.mon_features['last_damage_taken'],
        "trackers.mon_features['last_damage_dealt']": trackers.mon_features['last_damage_dealt'],
        'trackers.public_port.hazards': port.hazards,
        HISTORY_HP: h.hp, HISTORY_MAXIMUM: h.maximum,
        MAPPING: trackers.physical_to_public, NEXT_SLOT: trackers.next_slot, TURN: trackers.turn,
    }


def trace_encoder_call(step_fn, state, actions, trackers, *, mode):
    """Capture one Python-unrolled step during a fresh corrected/legacy trace.

    Arithmetic must already be selected by encoder_arithmetic around lowering.
    Wrapping the currently installed hooks preserves Stepper's other recorders.
    Unknown writers, incomplete guard sequences and nested JAX traces fail.
    """
    if mode not in ('corrected', 'legacy'):
        raise ValueError(f'unknown encoder trace mode {mode}')
    import jax
    import jax.numpy as j
    from gpu_engine import engine_device_damage_history as history
    from gpu_engine import engine_device_public_hooks as hooks
    from gpu_engine import engine_device_public_port as port
    from gpu_engine.v2e_modules import _substituting
    pre = encoder_pre(trackers)
    lanes = state.hp.shape[0];r = j.arange(lanes);both = j.arange(2)[None, :]
    for name, shape in ENCODER_SHAPES.items():
        a = pre[name]
        if a.shape != (lanes,) + shape or a.dtype != j.float32:
            raise ValueError(f'{name}: unsupported ENC carry shape/dtype')
    records, events, stack, sites = [], [], [], Counter()

    def vector(value):
        return j.broadcast_to(j.asarray(value), (lanes,))

    def active_hp(st):
        return st.hp[r[:, None], both, st.active], st.max_hp[r[:, None], both, st.active]

    # A payload must belong to the same JAX trace as the step inputs. Python
    # list capture from an inner cond/scan/JIT would otherwise leak a tracer.
    root_trace = getattr(state.hp, '_trace', None)

    def record(kind, payload, invoke):
        fields = EVENT_FIELDS[kind]
        if set(payload) != set(fields):raise ValueError(f'{kind}: incomplete ENC payload')
        for name, (dtype, shape) in fields.items():
            a = j.asarray(payload[name])
            if a.shape != (lanes,) + shape or a.dtype != j.dtype(dtype):
                raise ValueError(f'{kind}.{name}: unsupported ENC shape/dtype')
            trace = getattr(a, '_trace', None)
            if isinstance(a, jax.core.Tracer) and trace is not root_trace:
                raise ValueError('ENC event inside nested JAX trace is unsupported')
            payload[name] = a
        frame = sys._getframe(2)
        site = (frame.f_globals.get('__name__', ''), frame.f_code.co_name, frame.f_lineno)
        occurrence = sites[site];sites[site] += 1
        seq = len(records);guards = Counter()
        spec = dict(seq=seq, kind=kind, parent=stack[-1] if stack else None,
                    site=site + (occurrence,), fields=tuple((n, d, s) for n, (d, s) in fields.items()), guards=guards)
        records.append(spec);events.append(payload);stack.append(seq)
        try:
            out = invoke()
            if dict(guards) != EXPECTED_GUARDS[kind]:
                raise ValueError(f'{kind}: incomplete ENC writers: {dict(guards)}')
            spec['guards'] = dict(guards)
            return out
        finally:
            stack.pop()

    def guard(name, kinds):
        if not stack or records[stack[-1]]['kind'] not in kinds:
            raise ValueError(f'ENC writer {name} outside its recorded hook')
        records[stack[-1]]['guards'][name] += 1

    with _substituting:
        saved = []

        def replace(module, name, value):
            original = getattr(module, name);saved.append((module, name, original))
            setattr(module, name, value)
            return original

        def install(c, st, side, use, *, initial=False):
            t = c.trackers;encoder_pre(t);s = vector(side);p = st.active[r, s]
            payload = dict(side=s, use=vector(use), initial=vector(initial), active=st.active,
                           hp=st.hp[r, s, p], maximum=st.max_hp[r, s, p],
                           mapping=t.physical_to_public, next_slot=t.next_slot)
            return record('ARRIVE', payload, lambda: old_install(c, st, side, use, initial=initial))

        def observe(c, st, use, *, record_damage=True, rest_side=None, rest_mask=False):
            t = c.trackers;encoder_pre(t);hp, maximum = active_hp(st)
            payload = dict(use=vector(use), record_damage=vector(record_damage), hp=hp, maximum=maximum,
                           history_hp=t.damage_history.hp, history_maximum=t.damage_history.maximum)
            return record('OBSERVE', payload, lambda: old_observe(c, st, use, record_damage=record_damage,
                                                                rest_side=rest_side, rest_mask=rest_mask))

        def damage(c, st, side, dealt, taken, use):
            t = c.trackers;encoder_pre(t);s = vector(side);victim = 1 - s;p = st.active[r, victim]
            payload = dict(side=s, use=vector(use), dealt=vector(dealt), taken=vector(taken), active=st.active,
                           hp=st.hp[r, victim, p], maximum=st.max_hp[r, victim, p], hp_percent=t.hp_percent,
                           mapping=t.physical_to_public)
            return record('DIRECT', payload, lambda: old_damage(c, st, side, dealt, taken, use))

        def spikes(t, before, after, side, use, percent):
            encoder_pre(t);s = vector(side);p = before.active[r, s]
            payload = dict(side=s, use=vector(use), physical=p, hp=before.hp[r, s, p],
                           maximum=before.max_hp[r, s, p], new_hp=after.hp[r, s, p])
            return record('SPIKES', payload, lambda: old_spikes(t, before, after, side, use, percent))

        def public_turn(c, st, use, turn):
            t = c.trackers;encoder_pre(t)
            payload = dict(use=vector(use), number=vector(turn), active=st.active, mapping=t.physical_to_public)
            return record('TURN', payload, lambda: old_turn(c, st, use, turn))

        def turn_end(c, before, after, ordinary, *, upkeep=None, leftovers=None, residual=None, upkeep_living=None):
            t = c.trackers;encoder_pre(t)
            payload = dict(before_done=before.done, after_done=after.done, pending_residual=after.pending_residual,
                           forced_switch=after.forced_switch, after_turn=after.turn, incoming_turn=t.turn,
                           ordinary=vector(ordinary))
            return record('BOUNDARY', payload, lambda: old_end(c, before, after, ordinary,
                upkeep=upkeep, leftovers=leftovers, residual=residual, upkeep_living=upkeep_living))

        def arrive_guard(h, st, side, use):
            guard('arrive', ('ARRIVE',))
            return old_arrive(h, st, side, use)

        def observe_guard(h, st, use, percent, *, record_damage=True):
            guard('observe', ('OBSERVE',))
            return old_history_observe(h, st, use, percent, record_damage=record_damage)

        def direct_guard(h, side, shown, exact, use, **kw):
            guard('direct', ('DIRECT',))
            if (kw.get('pairs', False) is not (mode == 'corrected')):
                raise ValueError('ENC direct scalar/pair mode mismatch')
            return old_direct(h, side, shown, exact, use, **kw)

        def settle_guard(h, use):
            guard('settle', ('TURN',))
            return old_settle(h, use)

        def feature_guard(c, side, physical, name, value, use, *, add=False):
            if name in ('last_damage_taken', 'last_damage_dealt'):guard('feature', ('ARRIVE', 'TURN'))
            return old_feature(c, side, physical, name, value, use, add=add)

        def write_guard(t, name, side, physical, value, use):
            if name in ('damage_exact_taken', 'damage_exact_dealt'):guard('write', ('ARRIVE', 'TURN'))
            return old_write(t, name, side, physical, value, use)

        try:
            old_install = replace(hooks, 'install', install)
            old_observe = replace(hooks, 'observe', observe)
            old_damage = replace(hooks, 'damage', damage)
            old_spikes = replace(port, 'spikes', spikes)
            old_turn = replace(hooks, 'public_turn', public_turn)
            old_end = replace(hooks, 'turn_end', turn_end)
            old_arrive = replace(history, 'arrive', arrive_guard)
            old_history_observe = replace(history, 'observe', observe_guard)
            old_direct = replace(history, 'direct', direct_guard)
            old_settle = replace(history, 'settle', settle_guard)
            old_feature = replace(hooks, 'feature', feature_guard)
            old_write = replace(hooks, '_write', write_guard)
            out = step_fn(state, actions, trackers)
        finally:
            for module, name, original in reversed(saved):setattr(module, name, original)
    return TraceBuild(out, tuple(events), tuple(records), pre)
