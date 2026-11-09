"""Phase harness of the option-A CUDA step (design note sections 2.1-2.2): taps, slicer, carry spec, generated phases.

The engine step (transition.step_with_chance on the device path, as gpu_engine.v2e_modules.step calls it) is cut at the
top-level statement boundaries b0..b8 of BOUNDARIES. Everything here works from the UNMODIFIED oracle:

- Taps. While a step is traced, sys.settrace is restricted to transition.step_with_chance.__code__: at the `line` event of
  each boundary line (occurrences counted per line: b4 and b5 are line 689 in the action loop's iterations 0 and 1) the
  frame's locals are snapshotted (every JAX tracer, alone or inside a pytree such as `work` or `chance_inputs`), plus
  `_device_public.trackers` as `trackers`. b0 is the entry (the arguments, chance included), b8 the returned
  PublicStepResult. The snapshotted tracers are the oracle's own values, returned as extra outputs of the same trace.
  Boundary lines are checked against the live transition.py source and v2e_census.PHASES/ANCHORS (BoundaryMoved).
  Diagnostic sub-boundaries (865, 1124, 1582 per action; 1796 and each residual event at 1804) are taps too, behind
  boundaries(sub=True); they are not part of the spec.
- Liveness and slicer. jax.make_jaxpr of the tapped step at v2e_j2c.LANES under v2e_modules.trace_substitutions (the
  generator's trace). Each tapped var is lane-dependent (its cone reaches a step input) or constant-derived. A
  boundary's carry C_k is the lane-dependent tapped vars the later phases read, aliases (one var under several names)
  deduplicated by var identity, preferring state > actions > trackers > chance > other locals (plain names before
  _-prefixed ones, then f_locals order). Phase k's jaxpr is a
  backward DCE from C_{k+1}'s vars that stops at C_k's vars (JAX's per-primitive DCE rules prune call bodies); reaching a
  step input outside C_k raises IncompleteCarry. Constant-derived values are never carried: their equations are
  recomputed inside the phase, so the generator's constant-folding decisions (the f32 division rewrite) stay those of
  the whole program. b8 is the full 148-leaf step output in tree_flatten_with_path order.
- Spec. `python -m gpu_engine.v2e_phases spec ...` writes v2e_phases_spec.json (contract pokezero.v2e-phases-spec.v1):
  per boundary its ordered carry (= the FFI operand order of the phase starting there; a phase's outputs are the next
  boundary's carry in order). Additive fields: an entry's `aliases`, a b8 entry's `output_path`, a boundary's
  `event_line`, a phase's equation counts, `composites` (FG = b6->b8).
- Generated phases. Each phase jaxpr compiles with gpu_engine.v2e_gen.compile_jaxpr (thread mode, as the `step` module)
  into one FFI custom call whose operands follow the spec: inputs C_k in order, then the generator's constants; outputs
  C_{k+1} in order. An output that is an input unchanged (untouched state and tracker leaves) is aliased to it
  (input_output_aliases) and never written.
- v2e_diff providers: tap_identity (the tapped oracle's step outputs against the untapped oracle's), phase:<k>
  (teacher-forced: the generated phase on the oracle's C_k against the oracle's C_{k+1}; k one of PHASES, FG, or
  '<from>..<to>' boundary names) and chain:<spec> (generated and hand phases from the oracle's C_0, every produced
  boundary compared; `gen-A-G`, `gen-A-E+hand-FG`). Hand phases come from HAND_PHASES, filled by
  gpu_engine.v2e_phases_cuda when that module exists (register_hand_phase). Every gate's oracle side is the same
  full tapped program (all boundaries), so one XLA compile per lane count serves them (persistent compilation cache).
- `timings`: each generated phase, FG and the whole chain at --lanes (default 128, 2,048, 8,192) and at the live pricer's
  engine-step lane counts (--live), 50 warm calls with donated inputs, median and p90; the monolithic generated step for
  reference. `floats`: the float-site report (every f32 equation of the step and of observe: source line and the
  lowering the generator chose).

Paths are command-line arguments; nothing cluster-specific lives here.
"""
import argparse
from dataclasses import dataclass, field
import dis
import functools
import importlib
import inspect
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

SCHEMA = 'pokezero.v2e-phases-spec.v1'
SPEC = Path(__file__).resolve().parent / 'v2e_phases_spec.json'
SOURCES = ('state', 'actions', 'trackers', 'chance', 'local')
LOCAL_SOURCE = {'state': 'state', 'actions': 'actions', 'trackers': 'trackers', 'chance_inputs': 'chance'}
DTYPES = ('bool', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'float32')


class BoundaryMoved(RuntimeError):
    """A boundary statement is no longer at its line in transition.step_with_chance (or v2e_census disagrees)."""


class IncompleteCarry(RuntimeError):
    """A phase's backward cone reached a step input that is not in its starting carry."""


@dataclass(frozen=True)
class Boundary:
    name: str
    line: object          # transition.py line whose `line` event opens the boundary; None: entry (b0) or return (b8)
    occurrence: int       # which execution of that line (loop iteration)
    statement: str        # the source text at `line` must start with this (stripped)
    phase: object = None  # v2e_census.PHASES name `line` lies in (drift check); None: not checked
    main: bool = True     # one of the spec's b0..b8


BOUNDARIES = (
    Boundary('b0', None, 0, 'entry of step_with_chance'),
    Boundary('b1', 511, 0, 'speed = _speeds(work,table)', 'B'),
    Boundary('b2', 566, 0, 'bench = bench_indices(state.active)', 'D'),
    Boundary('b3', 669, 0, 'else:work = _lum_update(work,table,live,_device_public=_device_public)', 'D'),
    Boundary('b4', 689, 0, '_faint_before_hp=work.hp', 'E1'),
    Boundary('b5', 689, 1, '_faint_before_hp=work.hp', 'E1'),
    Boundary('b6', 1689, 0, 'turn_context = live & (~replacement | state.pending_residual)', 'F1'),
    Boundary('b7', 2041, 0, 'work=linking.prune(work)', 'G'),
    Boundary('b8', None, 0, 'return of step_with_chance (PublicStepResult)'),
)
PHASES = (('A', 'b0', 'b1'), ('BC', 'b1', 'b2'), ('D', 'b2', 'b3'), ('E0', 'b3', 'b4'), ('E_order0', 'b4', 'b5'),
    ('E_order1', 'b5', 'b6'), ('F', 'b6', 'b7'), ('G', 'b7', 'b8'))
COMPOSITES = {'FG': ('b6', 'b8')}   # one generated slice (M2's teacher-forced reference) or one hand phase
LETTERS = {'A': ('A',), 'B': ('BC',), 'C': ('BC',), 'BC': ('BC',), 'D': ('D',), 'E': ('E0', 'E_order0', 'E_order1'),
    'E0': ('E0',), 'E_order0': ('E_order0',), 'E_order1': ('E_order1',), 'F': ('F',), 'G': ('G',)}


def _sub_boundaries():
    """Diagnostic sub-boundaries, in program order after their main boundary: pre-move | damage | effects | post per
    action (865, 1124, 1582; 1582 is a comment, so its tap is the next line with code), and in F the residual order
    (1796) and each of the 10 residual events (1804)."""
    out = {}
    for order in (0, 1):
        out[f'b{4 + order}'] = [Boundary(f'b{4 + order}.865', 865, order, 'attack_stats = _active(work.stats, work, side)', 'E2', False),
            Boundary(f'b{4 + order}.1124', 1124, order, 'if table.effects is not None:', 'E3', False),
            Boundary(f'b{4 + order}.1582', 1582, order, '# Gen3 applies contact DamagingHit', 'E4', False)]
    out['b6'] = [Boundary('b6.1796', 1796, 0, 'event_sides = jnp.array([0,1,0,1,0,1,0,1,0,1])', 'F2', False)] + [
        Boundary(f'b6.1804.{i}', 1804, i, '_faint_before_hp=work.hp', 'F2', False) for i in range(10)]
    return out


def boundaries(sub=False):
    """The boundaries in program order: the spec's b0..b8, with the diagnostic sub-boundaries after their main one."""
    if not sub:return BOUNDARIES
    extra = _sub_boundaries();out = []
    for b in BOUNDARIES:out += [b] + extra.get(b.name, [])
    return tuple(out)


def step_code():
    from gpu_engine import transition
    return transition.step_with_chance.__code__


def event_lines(code, bs):
    """{boundary name: the line whose `line` event taps it}: the boundary line itself, or (a comment line) the next line
    with code."""
    starts = sorted({line for _, line in dis.findlinestarts(code) if line is not None})
    out = {}
    for b in bs:
        if b.line is None:continue
        later = [l for l in starts if l >= b.line]
        if not later:raise BoundaryMoved(f'{b.name}: no code at or after line {b.line}')
        out[b.name] = later[0]
    return out


def check_boundaries(bs=BOUNDARIES):
    """Fail loudly (BoundaryMoved) unless every boundary statement is still at its line in transition.step_with_chance and
    agrees with v2e_census (the def line at SPAN, ANCHORS where it lists the line, the PHASES phase)."""
    from gpu_engine import transition, v2e_census as cz
    lines, first = inspect.getsourcelines(transition.step_with_chance)
    if first != cz.SPAN[0]:raise BoundaryMoved(f'step_with_chance starts at line {first}, v2e_census.SPAN says {cz.SPAN[0]}')
    problems = []
    for b in bs:
        if b.line is None:continue
        if not first <= b.line < first + len(lines):problems.append(f'{b.name}: line {b.line} is outside step_with_chance');continue
        text = lines[b.line - first].strip()
        if not text.startswith(b.statement):
            moved = [first + i for i, t in enumerate(lines) if t.strip().startswith(b.statement)]
            problems.append(f'{b.name}: transition.py:{b.line} reads {text[:80]!r}, expected {b.statement!r}'
                + (f' (now at line {moved})' if moved else ' (statement not found)'))
        anchor = cz.ANCHORS.get(b.line)
        if anchor is not None and not (anchor.startswith(b.statement) or b.statement.startswith(anchor)):
            problems.append(f'{b.name}: v2e_census.ANCHORS[{b.line}] is {anchor!r}, the boundary statement {b.statement!r}')
        if b.phase is not None and cz.phase_of(b.line) != b.phase:
            problems.append(f'{b.name}: line {b.line} is in census phase {cz.phase_of(b.line)}, expected {b.phase}')
    if problems:raise BoundaryMoved('boundary statements moved:\n  ' + '\n  '.join(problems))
    events = event_lines(transition.step_with_chance.__code__, bs)
    return {b.name: dict(line=b.line, event_line=events.get(b.name), text=lines[b.line - first].strip() if b.line else b.statement) for b in bs}


# ---------------------------------------------------------------------------
# Taps.

def _tracer_type():
    try:
        from jax._src.core import Tracer
    except ImportError:   # pragma: no cover
        from jax.core import Tracer
    return Tracer


def local_snapshot(frame):
    """{name: (local, path, tracer)} of every tracer leaf among the frame's locals (in f_locals order)."""
    import jax
    tracer = _tracer_type();out = {}
    for local, value in list(frame.f_locals.items()):
        try:leaves, _ = jax.tree_util.tree_flatten_with_path(value)
        except Exception:continue
        for path, leaf in leaves:
            if isinstance(leaf, tracer):out[local + jax.tree_util.keystr(path)] = (local, jax.tree_util.keystr(path), leaf)
    return out


def step_snapshot(frame):
    """local_snapshot plus the tracker carry, `_device_public.trackers`, as the local `trackers`."""
    import jax
    out = local_snapshot(frame)
    context = frame.f_locals.get('_device_public')
    if context is not None:
        if any(v[0] == 'trackers' for v in out.values()):raise RuntimeError('step_with_chance has a local named trackers; rename the tap')
        tracer = _tracer_type()
        for path, leaf in jax.tree_util.tree_flatten_with_path(context.trackers)[0]:
            if isinstance(leaf, tracer):out['trackers' + jax.tree_util.keystr(path)] = ('trackers', jax.tree_util.keystr(path), leaf)
    return out


class Tap:
    """sys.settrace restricted to one code object while a function is traced: snapshot(frame) at its entry (the boundary
    with line None that comes first) and at the `line` event of each (line, occurrence) boundary. Restores the previous
    trace function on exit; check() fails unless the code ran exactly once and every boundary was recorded."""

    def __init__(self, code, bs, snapshot, events=None):
        self.code = code;self.snapshot = snapshot;self.records = {};self.calls = 0;self.errors = []
        events = events or event_lines(code, bs)
        self.entry = next((b.name for b in bs if b.line is None), None)
        self.want = {}
        for b in bs:
            if b.line is not None:self.want.setdefault(events[b.name], {})[b.occurrence] = b.name
        self.names = [b.name for b in bs if b.line is not None] + ([self.entry] if self.entry else [])
        self.counts = {}

    def __enter__(self):
        self.previous = sys.gettrace();sys.settrace(self._global)
        return self

    def __exit__(self, *exc):
        sys.settrace(self.previous)
        return False

    def _global(self, frame, event, arg):
        if frame.f_code is not self.code:return None
        self.calls += 1
        if self.calls == 1 and self.entry is not None:self._take(self.entry, frame)
        return self._local if self.calls == 1 else None

    def _local(self, frame, event, arg):
        if event == 'line':
            names = self.want.get(frame.f_lineno)
            if names is not None:
                n = self.counts[frame.f_lineno] = self.counts.get(frame.f_lineno, -1) + 1
                name = names.get(n)
                if name is not None:self._take(name, frame)
        return self._local

    def _take(self, name, frame):
        try:self.records[name] = self.snapshot(frame)
        except Exception as error:   # a trace function must not raise; check() re-raises
            self.errors.append(f'{name}: {type(error).__name__}: {error}')

    def check(self):
        if self.errors:raise RuntimeError('tap snapshot failed: ' + '; '.join(self.errors))
        if self.calls != 1:raise RuntimeError(f'the tapped code ran {self.calls} times in one trace (expected once)')
        missing = [n for n in self.names if n not in self.records]
        if missing:raise RuntimeError(f'boundaries never reached: {missing} (line events seen: {self.counts})')


def output_entries(out, prefix=''):
    """b8: (name, source, path, value, output_path) per step output leaf in tree_flatten_with_path order. Names drop the
    keystr's leading dot ('step.state.hp', 'step.invalid_actions', 'trackers.public_volatiles')."""
    import jax
    rows = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(out)[0]:
        full = jax.tree_util.keystr(path);name = prefix + full.lstrip('.')
        if full.startswith('.step.state'):source, sub = 'state', full[len('.step.state'):]
        elif full.startswith('.trackers'):source, sub = 'trackers', full[len('.trackers'):]
        else:source, sub = 'local', ''
        rows.append((name, source, sub, leaf, full))
    return rows


def run_tapped(fn, args, bs, *, code=None, snapshot=step_snapshot):
    """fn(*args) under taps: (outputs, {boundary: {name: (local, path, tracer)}}) for every boundary with a line or the
    entry; the final boundary (the return) is the caller's outputs."""
    tap = Tap(code or step_code(), bs, snapshot)
    with tap:out = fn(*args)
    tap.check()
    return out, tap.records


def source_of(local):
    return LOCAL_SOURCE.get(local, 'local')


# ---------------------------------------------------------------------------
# Liveness and slicing.

CALLS = {'jit', 'pjit', 'closed_call', 'core_call', 'custom_jvp_call', 'custom_vjp_call', 'remat', 'checkpoint'}


def _is_literal(a):
    from jax.extend import core as jcore
    return isinstance(a, jcore.Literal)


def _call_body(e):
    """The inner open jaxpr of a call-like equation whose invars align with the equation's, else None."""
    if e.primitive.name not in CALLS:return None
    for key in ('jaxpr', 'call_jaxpr', 'fun_jaxpr'):
        body = e.params.get(key)
        if body is None:continue
        inner = getattr(body, 'jaxpr', body)
        if hasattr(inner, 'invars') and len(inner.invars) == len(e.invars) and len(inner.outvars) == len(e.outvars):return inner
    return None


def lane_dependence(jaxpr, inputs=None, memo=None):
    """{var: True when its value depends on a jaxpr input} over the top-level equations of `jaxpr` (constvars and literals
    are constant). Call bodies are analysed per output, so a call's constant-derived output stays constant."""
    memo = {} if memo is None else memo
    dep = {v: (True if inputs is None else bool(d)) for v, d in zip(jaxpr.invars, inputs or [True] * len(jaxpr.invars))}
    for v in jaxpr.constvars:dep[v] = False
    for e in jaxpr.eqns:
        ins = tuple(False if _is_literal(a) else dep.get(a, False) for a in e.invars)
        body = _call_body(e) if any(ins) else None
        if body is not None:
            key = (id(body), ins)
            if key not in memo:
                inner = lane_dependence(body, ins, memo)
                memo[key] = (body, tuple(False if _is_literal(o) else inner.get(o, False) for o in body.outvars))
            outs = memo[key][1]
        else:
            outs = (any(ins),) * len(e.outvars)
        for v, d in zip(e.outvars, outs):dep[v] = d
    return dep


def _leaf_count(jaxpr):
    from jax.extend import core as jcore
    return sum((sum(_leaf_count(s) for s in subs) if (subs := list(jcore.jaxprs_in_params(e.params))) else 1) for e in jaxpr.eqns)


@functools.lru_cache(maxsize=None)
def _rule_arity(rule):
    try:return len(inspect.signature(rule).parameters)
    except (TypeError, ValueError):return 3


def backward_slice(jaxpr, wanted, stop):
    """Equations computing `wanted` (vars or literals) from the `stop` vars: a backward DCE over the top-level equations that
    does not expand stop vars (JAX's per-primitive DCE rules prune call bodies to the used outputs). Returns (kept
    [(original, pruned) in program order], reached stop vars, leftover vars no kept equation defines: step inputs or
    constvars)."""
    from jax._src.interpreters import partial_eval as pe
    needed = set();reached = set()
    for a in wanted:
        if _is_literal(a):continue
        (reached if a in stop else needed).add(a)
    kept = []
    for e in reversed(jaxpr.eqns):
        used = [(not _is_literal(v)) and type(v).__name__ != 'DropVar' and v in needed for v in e.outvars]
        if not any(used):continue
        rule = pe.dce_rules.get(e.primitive)
        if rule is None:used_inputs, new = [True] * len(e.invars), e
        else:
            # jax 0.12: rule(used_outputs, live_ins, eqn); older jax: rule(used_outputs, eqn).
            used_inputs, new = rule(used, [True] * len(e.invars), e) if _rule_arity(rule) == 3 else rule(used, e)
            if new is None:continue
        kept.append((e, new))
        for v, u in zip(e.outvars, used):
            if u:needed.discard(v)
        for a, u in zip(e.invars, used_inputs):
            if u and not _is_literal(a):(reached if a in stop else needed).add(a)
    kept.reverse()
    return kept, reached, needed


@dataclass(eq=False)
class Entry:
    """One carry leaf: unique name, source, key path, numpy dtype name, per-lane shape, its jaxpr atom."""
    name: str
    source: str
    path: str
    dtype: str
    shape: tuple
    atom: object = None
    aliases: list = field(default_factory=list)
    output_path: object = None

    def json(self):
        out = dict(name=self.name, source=self.source, path=self.path, dtype=self.dtype, shape=[int(d) for d in self.shape])
        if self.aliases:out['aliases'] = list(self.aliases)
        if self.output_path is not None:out['output_path'] = self.output_path
        return out


def _dtype_name(aval):
    import jax
    if jax.dtypes.issubdtype(aval.dtype, jax.dtypes.prng_key):raise TypeError(f'typed PRNG key {aval.dtype} in a carry (not supported)')
    return np.dtype(aval.dtype).name


def _debug_info(name, args, results):
    """A jax DebugInfo naming a phase jaxpr's operands (jax 0.12 warns on a Jaxpr without one); None if unavailable."""
    try:
        from jax._src.core import DebugInfo
        return DebugInfo('v2e_phases', name, tuple(args), tuple(results))
    except Exception:
        return None


@dataclass(eq=False)
class PhaseSlice:
    name: str
    start: str
    end: str
    closed: object        # ClosedJaxpr over C_start (in order) -> C_end (in order)
    aliases: dict         # {output index: input index}: outputs that are an input unchanged
    stats: dict


class Plan:
    """Carries and phase slices of one tapped trace.

    closed: the tapped program's closed jaxpr (DCE'd); n_out: its first n_out outvars are the step outputs (the final
    boundary, in order). taps: {boundary: [(name, source, path, outvar index)]} for the other boundaries, in snapshot
    order. order: boundary names in program order (the final one last). input_names: a name per jaxpr invar (errors).
    """

    def __init__(self, closed, n_out, taps, order, final_rows, input_names=None, lanes=None, meta=None):
        from gpu_engine import v2e_j2c
        self.closed = closed;self.order = list(order);self.final = self.order[-1];self.meta = meta or {}
        self.lanes = lanes or v2e_j2c.LANES
        jx = closed.jaxpr;outvars = list(jx.outvars)
        self.input_names = dict(zip(jx.invars, input_names or [f'in{i}' for i in range(len(jx.invars))]))
        self.dep = lane_dependence(jx)
        self.tapped = {};self.candidates = {};self.constant = {}
        for b in self.order[:-1]:
            rows = taps[b];ranked = sorted(range(len(rows)), key=lambda i: (SOURCES.index(rows[i][1]), rows[i][0].startswith('_'), i))
            seen = {};cands = [];constant = 0
            for i in ranked:
                name, source, path, index = rows[i];atom = outvars[index]
                if _is_literal(atom) or not self.dep.get(atom, False):constant += 1;continue
                if atom in seen:seen[atom].aliases.append(name);continue
                if not (atom.aval.shape and atom.aval.shape[0] == self.lanes):
                    raise IncompleteCarry(f'{b}: lane-dependent {name} has no lane axis ({atom.aval.shape})')
                entry = Entry(name, source, path, _dtype_name(atom.aval), tuple(atom.aval.shape[1:]), atom)
                seen[atom] = entry;cands.append(entry)
            self.candidates[b] = cands;self.constant[b] = constant;self.tapped[b] = len(rows)
        final = []
        for (name, source, path, output_path), atom in zip(final_rows, outvars[:n_out]):
            shape = tuple(atom.aval.shape)
            if not (shape and shape[0] == self.lanes):raise IncompleteCarry(f'{self.final}: output {name} has no lane axis ({shape})')
            final.append(Entry(name, source, path, _dtype_name(atom.aval), shape[1:], atom, output_path=output_path))
        self.carry = {self.final: final};self.slices = {};self.lane_equations = {}
        for start, end in reversed(list(zip(self.order[:-1], self.order[1:]))):
            self.carry[start] = None
            s = self._slice(start, end, self.candidates[start])
            self.carry[start] = [e for e in self.candidates[start] if e.atom in s['reached']]
            self.slices[(start, end)] = self._assemble(start, end, s)
        names = lambda b: [e.name for e in self.carry[b]]
        for b in self.order:
            if len(set(names(b))) != len(names(b)) and b != self.final:raise RuntimeError(f'{b}: duplicate carry names')

    def _slice(self, start, end, stop_entries):
        stop = {e.atom for e in stop_entries}
        kept, reached, leftover = backward_slice(self.closed.jaxpr, [e.atom for e in self.carry[end]], stop)
        inputs = [v for v in leftover if v in self.input_names]
        if inputs:
            raise IncompleteCarry(f'phase {start}->{end}: the cone reaches step inputs outside C_{start}: '
                + ', '.join(sorted(self.input_names[v] for v in inputs)[:20]))
        others = [v for v in leftover if v not in set(self.closed.jaxpr.constvars) and v not in self.input_names]
        if others:raise RuntimeError(f'phase {start}->{end}: {len(others)} vars without a defining equation')
        return dict(kept=kept, reached=reached, leftover=leftover)

    def _assemble(self, start, end, s, name=None):
        from jax.extend import core as jcore
        from jax._src.interpreters import partial_eval as pe
        jx = self.closed.jaxpr
        invars = [e.atom for e in self.carry[start]];outvars = [e.atom for e in self.carry[end]]
        constvars = [v for v in jx.constvars if v in s['leftover']]
        consts = [c for v, c in zip(jx.constvars, self.closed.consts) if v in s['leftover']]
        eqns = [new for _, new in s['kept']]
        from jax._src import core as _core
        effects = _core.join_effects(*(e.effects for e in eqns)) if eqns else jcore.no_effects
        info = _debug_info(name or f'{start}->{end}', [e.name for e in self.carry[start]], [e.name for e in self.carry[end]])
        jaxpr = jcore.Jaxpr(constvars=constvars, invars=invars, outvars=outvars, eqns=eqns, effects=effects, **({'debug_info': info} if info else {}))
        dced, _ = pe.dce_jaxpr(jaxpr, [True] * len(outvars), instantiate=True)
        closed = jcore.ClosedJaxpr(dced, consts)
        index = {v: i for i, v in enumerate(invars)};aliases = {};taken = set()
        for o, v in enumerate(outvars):
            if not _is_literal(v) and v in index and index[v] not in taken:aliases[o] = index[v];taken.add(index[v])
        lane = [e for e, _ in s['kept'] if any(not _is_literal(v) and self.dep.get(v, False) for v in e.outvars)]
        self.lane_equations[(start, end)] = {id(e) for e in lane}
        stats = dict(top_level=len(dced.eqns), leaf=_leaf_count(dced), lane_top_level=len(lane), inputs=len(invars), outputs=len(outvars),
            aliased=len(aliases), constants=len(consts))
        return PhaseSlice(name or f'{start}->{end}', start, end, closed, aliases, stats)

    def slice(self, start, end, name=None):
        """The phase slice C_start -> C_end (consecutive or not, e.g. FG = b6 -> b8), cached."""
        key = (start, end)
        if key not in self.slices:
            if self.order.index(start) >= self.order.index(end):raise ValueError(f'{start} does not precede {end}')
            s = self._slice(start, end, self.carry[start])
            self.slices[key] = self._assemble(start, end, s, name)
        return self.slices[key]

    def recomputed(self):
        """Lane-dependent top-level equations that more than one consecutive phase slice computes (0 when every value
        crossing a boundary is held by a tapped local)."""
        seen = {};dup = 0
        for (start, end), ids in self.lane_equations.items():
            if self.order.index(end) != self.order.index(start) + 1:continue
            for i in ids:
                if i in seen:dup += 1
                seen[i] = (start, end)
        return dup

    def summary(self):
        rows = []
        for b in self.order:
            carry = self.carry[b];elements = sum(int(np.prod(e.shape)) if e.shape else 1 for e in carry)
            rows.append(dict(boundary=b, lane_vars=len(carry), elements=elements, tapped=self.tapped.get(b),
                constant=self.constant.get(b), candidates=len(self.candidates.get(b, carry)),
                by_source={s: sum(1 for e in carry if e.source == s) for s in SOURCES}))
        return dict(boundaries=rows, phases={f'{a}->{b}': s.stats for (a, b), s in self.slices.items()}, recomputed_lane_equations=self.recomputed(),
            whole_program=dict(top_level=len(self.closed.jaxpr.eqns), leaf=_leaf_count(self.closed.jaxpr)))


def trace_plan(fn, example, bs, *, snapshot=step_snapshot, code=None, lanes=None, input_names=None, meta=None):
    """Plan of fn(*example) (pytrees of arrays with a lane axis) traced at `lanes` (default v2e_j2c.LANES) under taps at
    `bs` (the last boundary is the return: its carry is the full output, in tree_flatten_with_path order)."""
    import jax
    from jax._src.interpreters import partial_eval as pe
    from jax.extend import core as jcore
    from gpu_engine import v2e_j2c
    lanes = lanes or v2e_j2c.LANES
    flat, in_tree = jax.tree.flatten(tuple(example))
    avals = [jax.ShapeDtypeStruct((lanes,) + tuple(np.shape(x)[1:]), x.dtype) for x in flat]
    layout = {}
    tapped = list(bs[:-1])   # the last boundary is the return: the outputs themselves
    def flat_fn(*xs):
        out, records = run_tapped(fn, jax.tree.unflatten(in_tree, xs), tapped, code=code, snapshot=snapshot)
        rows = output_entries(out);leaves = [r[3] for r in rows]
        layout['final'] = [(r[0], r[1], r[2], r[4]) for r in rows];layout['taps'] = {}
        for b in tapped:
            entries = []
            for name, (local, path, value) in records[b.name].items():
                entries.append((name, source_of(local), path, len(leaves)));leaves.append(value)
            layout['taps'][b.name] = entries
        return leaves
    closed = jax.make_jaxpr(flat_fn)(*avals)
    dced, _ = pe.dce_jaxpr(closed.jaxpr, [True] * len(closed.jaxpr.outvars), instantiate=True)
    closed = jcore.ClosedJaxpr(dced, closed.consts)
    names = input_names or [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(tuple(example))[0]]
    return Plan(closed, len(layout['final']), layout['taps'], [b.name for b in bs], layout['final'], input_names=names, lanes=lanes, meta=meta)


def step_plan(ctx, sub=False):
    """The step's plan: v2e_modules.step(tables) traced at v2e_j2c.LANES under the generator's trace substitutions, tapped
    at boundaries(sub). Cached per (tables, sub)."""
    key = (id(ctx['engine_tables']), id(ctx['observation_tables']), bool(sub))
    if key in _PLANS:return _PLANS[key]
    from gpu_engine import v2e_modules
    bs = boundaries(sub);check_boundaries(bs)
    fn = v2e_modules.step(ctx['engine_tables'], ctx['observation_tables']);state, actions, trackers = ctx['template']
    names = ['state' + p for p in _paths(state)] + ['actions' + p for p in _paths(actions)] + ['trackers' + p for p in _paths(trackers)]
    begin = time.perf_counter()
    with v2e_modules.trace_substitutions():
        plan = trace_plan(lambda s, a, t: fn(s, a, t), (state, actions, trackers), bs, input_names=names)
    plan.meta.update(trace_seconds=time.perf_counter() - begin, sub=bool(sub))
    _PLANS[key] = plan
    return plan


_PLANS = {}


def _paths(tree):
    import jax
    return [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(tree)[0]]


# ---------------------------------------------------------------------------
# The spec.

def spec_of(plan, bs=BOUNDARIES, oracle_digest=None):
    """The contract JSON (pokezero.v2e-phases-spec.v1) of a plan over the main boundaries."""
    info = check_boundaries(bs) if bs is BOUNDARIES else {}
    out = dict(schema=SCHEMA, oracle_digest=oracle_digest, lanes_traced=int(plan.lanes), boundaries=[], phases=[], composites=[])
    for b in bs:
        if b.name not in plan.carry:continue
        carry = plan.carry[b.name];row = info.get(b.name, {})
        out['boundaries'].append(dict(name=b.name, line=b.line, occurrence=b.occurrence, statement=row.get('text', b.statement),
            event_line=row.get('event_line'), lane_vars=len(carry), elements=sum(int(np.prod(e.shape)) if e.shape else 1 for e in carry),
            carry=[e.json() for e in carry]))
    for name, start, end in PHASES:
        if (start, end) in plan.slices:
            s = plan.slices[(start, end)];out['phases'].append(dict(name=name, **{'from': start, 'to': end}, equations=s.stats['top_level'],
                leaf_equations=s.stats['leaf'], lane_equations=s.stats['lane_top_level'], aliased_outputs=s.stats['aliased']))
    for name, (start, end) in COMPOSITES.items():out['composites'].append({'name': name, 'from': start, 'to': end})
    return out


def validate_spec(spec, *, output_leaves=None):
    """Format check of a spec dict against the contract; raises ValueError listing every problem."""
    problems = []
    if spec.get('schema') != SCHEMA:problems.append(f'schema {spec.get("schema")!r}')
    for key in ('oracle_digest', 'lanes_traced', 'boundaries', 'phases'):
        if key not in spec:problems.append(f'missing {key}')
    names = [b.get('name') for b in spec.get('boundaries', [])]
    if names != [b.name for b in BOUNDARIES]:problems.append(f'boundaries {names}, expected b0..b8')
    for b in spec.get('boundaries', []):
        for key in ('name', 'line', 'occurrence', 'statement', 'carry'):
            if key not in b:problems.append(f'{b.get("name")}: missing {key}')
        seen = set()
        for e in b.get('carry', []):
            for key in ('name', 'source', 'path', 'dtype', 'shape'):
                if key not in e:problems.append(f'{b.get("name")}: an entry lacks {key}')
            if e.get('name') in seen:problems.append(f'{b.get("name")}: duplicate name {e.get("name")}')
            seen.add(e.get('name'))
            if e.get('source') not in SOURCES:problems.append(f'{b.get("name")}/{e.get("name")}: source {e.get("source")!r}')
            if e.get('dtype') not in DTYPES:problems.append(f'{b.get("name")}/{e.get("name")}: dtype {e.get("dtype")!r}')
            if not isinstance(e.get('shape'), list) or not all(isinstance(d, int) and d > 0 for d in e.get('shape', [0])):
                problems.append(f'{b.get("name")}/{e.get("name")}: shape {e.get("shape")!r}')
    phases = [(p.get('name'), p.get('from'), p.get('to')) for p in spec.get('phases', [])]
    if phases != list(PHASES):problems.append(f'phases {phases}, expected {list(PHASES)}')
    if output_leaves is not None and spec.get('boundaries') and len(spec['boundaries'][-1].get('carry', [])) != output_leaves:
        problems.append(f"b8 carries {len(spec['boundaries'][-1]['carry'])} leaves, the step outputs {output_leaves}")
    if problems:raise ValueError('invalid phases spec: ' + '; '.join(problems))
    return spec


def load_spec(path=SPEC):
    return validate_spec(json.loads(Path(path).read_text()))


def carry_signature(rows):
    return [(e['name'], e['dtype'], list(e['shape'])) for e in rows]


def spec_differences(have, want):
    """Boundaries whose carry (names, dtypes, shapes, order) differs between two specs."""
    a = {b['name']: carry_signature(b['carry']) for b in have['boundaries']}
    b = {x['name']: carry_signature(x['carry']) for x in want['boundaries']}
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


# ---------------------------------------------------------------------------
# Generated phases and hand phases.

@dataclass
class HandPhase:
    """A hand-written phase: fn(carry: {name: array} at `start`, context) -> {name: array} at `end` (or (that, err));
    names, dtypes and shapes are the committed spec's."""
    name: str
    start: str
    end: str
    fn: object
    info: dict = field(default_factory=dict)


HAND_PHASES = {}


def register_hand_phase(name, start, end, fn, **info):
    HAND_PHASES[name] = HandPhase(name, start, end, fn, dict(info))
    return HAND_PHASES[name]


def load_hand_phases():
    """Import gpu_engine.v2e_phases_cuda (M2's hand phases) when it exists; it registers with register_hand_phase or
    exposes a HAND_PHASES dict of HandPhase."""
    try:module = importlib.import_module('gpu_engine.v2e_phases_cuda')
    except ModuleNotFoundError as error:
        if error.name != 'gpu_engine.v2e_phases_cuda':raise
        return HAND_PHASES
    for name, hp in (getattr(module, 'HAND_PHASES', None) or {}).items():
        if hp is not HAND_PHASES.get(name):HAND_PHASES[name] = hp
    return HAND_PHASES


def _label(start, end):
    for name, a, b in PHASES:
        if (a, b) == (start, end):return name
    for name, (a, b) in COMPOSITES.items():
        if (a, b) == (start, end):return name
    return f'{start}_{end}'.replace('.', '_')


def compile_phase(plan, start, end, build_dir=None, mode='thread'):
    """The generated phase start -> end: a GeneratedFlat over C_start (spec order) returning (C_end list, err)."""
    from gpu_engine import v2e_gen
    s = plan.slice(start, end);name = 'phase_' + _label(start, end)
    avals = [_aval(e.atom) for e in plan.carry[start]]
    begin = time.perf_counter()
    generated = v2e_gen.compile_jaxpr(name, s.closed, avals, build_dir=build_dir, mode=mode, aliases=s.aliases)
    generated.stats = dict(generated.module.stats, seconds=time.perf_counter() - begin, chunks=generated.module.stats.get('chunks'))
    return generated


def _aval(atom):
    import jax
    return jax.ShapeDtypeStruct(tuple(atom.aval.shape), atom.aval.dtype)


@dataclass
class Unit:
    kind: str     # 'gen' | 'hand'
    name: str
    start: str
    end: str


def parse_chain(spec, hand=None):
    """Units of a chain spec: '+'-joined segments 'gen-<X>[-<Y>]' (generated phases: X, or the range X..Y in PHASES order;
    letters A..G name groups, FG alone is the b6->b8 composite slice) and 'hand-<NAME>' (HAND_PHASES[NAME]). The units must
    run contiguously from b0 to b8."""
    hand = HAND_PHASES if hand is None else hand
    order = [p[0] for p in PHASES];units = []
    for segment in spec.split('+'):
        kind, _, rest = segment.partition('-')
        if kind == 'hand':
            if rest not in hand:raise KeyError(f'hand phase {rest!r} is not registered (HAND_PHASES: {sorted(hand)})')
            hp = hand[rest];units.append(Unit('hand', rest, hp.start, hp.end));continue
        if kind != 'gen' or not rest:raise ValueError(f'chain segment {segment!r}: expected gen-<phases> or hand-<name>')
        parts = rest.split('-')
        if len(parts) == 1 and parts[0] in COMPOSITES:
            a, b = COMPOSITES[parts[0]];units.append(Unit('gen', parts[0], a, b));continue
        if len(parts) > 2 or any(p not in LETTERS for p in parts):raise ValueError(f'chain segment {segment!r}: unknown phase in {parts}')
        first = order.index(LETTERS[parts[0]][0]);last = order.index(LETTERS[parts[-1]][-1])
        if last < first:raise ValueError(f'chain segment {segment!r}: empty range')
        for name, a, b in PHASES[first:last + 1]:units.append(Unit('gen', name, a, b))
    at = 'b0'
    for u in units:
        if u.start != at:raise ValueError(f'chain {spec!r}: {u.kind} {u.name} starts at {u.start}, the chain is at {at}')
        at = u.end
    if at != 'b8':raise ValueError(f'chain {spec!r} ends at {at}, not b8')
    return units


# ---------------------------------------------------------------------------
# The oracle side: the full tapped program.

def oracle_full(ctx, plan, bs):
    """fn(state, actions, trackers) -> {'<boundary>/<name>': value} for every carry entry of every boundary of `plan`
    (the final boundary: the step outputs), from the UNMODIFIED oracle (no trace substitutions) under taps."""
    from gpu_engine import v2e_modules
    step = v2e_modules.step(ctx['engine_tables'], ctx['observation_tables'])
    tapped = [b for b in bs if b.name != plan.final]
    want = {b: [e.name for e in plan.carry[b]] for b in plan.order}
    def fn(state, actions, trackers):
        out, records = run_tapped(step, (state, actions, trackers), tapped)
        result = {}
        for b in plan.order[:-1]:
            snap = records[b]
            for name in want[b]:
                if name not in snap:raise KeyError(f'{b}: carry entry {name} is not a tapped local of the oracle trace')
                result[f'{b}/{name}'] = snap[name][2]
        rows = output_entries(out)
        if [r[0] for r in rows] != want[plan.final]:raise RuntimeError('step output leaves differ from the plan')
        for r in rows:result[f'{plan.final}/{r[0]}'] = r[3]
        return result
    return fn


def _zero_err():
    import jax.numpy as j
    return j.zeros((1,), j.uint32)


def _context_plan(argument_boundaries, ctx):
    sub = any('.' in b for b in argument_boundaries)
    plan = step_plan(ctx, sub=sub)
    return plan, boundaries(sub)


def _spec_check(plan):
    """(matches, differing boundaries) of the live plan's main carries against the committed spec (no spec: (None, None))."""
    if not SPEC.exists():return None, None
    live = dict(boundaries=[dict(name=b.name, carry=[e.json() for e in plan.carry[b.name]]) for b in BOUNDARIES if b.name in plan.carry])
    diff = spec_differences(live, json.loads(SPEC.read_text()))
    return not diff, diff


def _build_dir(ctx):
    """The provider's build directory: ctx['build_dir'], else the directory v2e_diff loaded its CUDA library from
    (--build-dir), else None (v2e_gen's default)."""
    if ctx.get('build_dir'):return ctx['build_dir']
    from gpu_engine import v2e_cuda
    path = v2e_cuda._loaded.get('path')
    return Path(path).parent if path else None


def _phase_range(argument):
    """(start, end, label) of a phase argument: a PHASES name, a composite (FG) or '<from>..<to>' boundary names."""
    for name, a, b in PHASES:
        if argument == name:return a, b, name
    if argument in COMPOSITES:a, b = COMPOSITES[argument];return a, b, argument
    if '..' in argument:a, _, b = argument.partition('..');return a, b, argument
    raise KeyError(f'unknown phase {argument!r}: one of {[p[0] for p in PHASES] + list(COMPOSITES)} or <from>..<to>')


def diff_tap_identity(argument, ctx):
    """v2e_diff provider: the untapped oracle's step outputs (jax side) against the tapped oracle's (the full tapped
    program every phase gate uses as its oracle side; its taps stay outputs of the compiled program and are dropped on
    the host by `project`)."""
    import jax
    from gpu_engine import v2e_modules
    plan, bs = _context_plan([], ctx)
    step = v2e_modules.step(ctx['engine_tables'], ctx['observation_tables'])
    full = oracle_full(ctx, plan, bs);finals = [e.name for e in plan.carry[plan.final]]
    structure = {}
    def reference(s, a, t, c):
        out = step(s, a, t);structure['tree'] = jax.tree.structure(out)
        return out
    def tapped(s, a, t, c):
        return full(s, a, t)
    def project(got):
        return jax.tree.unflatten(structure['tree'], [got[f'{plan.final}/{n}'] for n in finals])
    tapped.project = project
    matches, differing = _spec_check(plan)
    reference.info = tapped.info = dict(provider='tap_identity', taps={b: len(plan.carry[b]) for b in plan.order}, spec_matches_committed=matches,
        spec_differing_boundaries=differing, plan_trace_seconds=plan.meta.get('trace_seconds'))
    return reference, tapped


def _boundary_of(key):
    """'b3' from a run() leaf name such as "['b3/work.hp']" or the double-quoted keystr of a name holding quotes."""
    k = key[2:-2] if key[:2] in ("['", '["') else key
    return k.split('/', 1)[0]


def _observer(order):
    """observe(diffs, lanes, path) / summary() pair: per boundary, lane-steps with any differing leaf; the first differing
    boundary in chain order, with its first (shard, leaf, lane)."""
    totals = {b: 0 for b in order};first = {};state = dict(lanes=0)
    def observe(diffs, lanes, path):
        state['lanes'] += int(lanes);per = {};leaves = {}
        for key, d in diffs.items():
            b = _boundary_of(key);d = np.asarray(d).reshape(-1)
            if d.size != lanes:d = np.full(lanes, bool(d.any()))
            per[b] = per[b] | d if b in per else d.copy()
            if d.any():leaves.setdefault(b, []).append((key, d))
        for b, d in per.items():
            totals[b] = totals.get(b, 0) + int(d.sum())
            if d.any() and b not in first:
                lane = int(np.flatnonzero(d)[0])
                first[b] = dict(shard=str(path), lane=lane, leaf=next(k for k, x in leaves[b] if x[lane]))
    def summary():
        differing = [b for b in order if totals.get(b)]
        return dict(boundaries_compared=list(order), differing_by_boundary={b: totals.get(b, 0) for b in order},
            first_differing_boundary=differing[0] if differing else None, first_difference_by_boundary=first,
            first_difference_at_first_boundary=first.get(differing[0]) if differing else None)
    return observe, summary


def diff_phase(argument, ctx):
    """v2e_diff provider, teacher-forced: the generated phase on the oracle's tapped C_start against the oracle's C_end.
    The oracle side is the full tapped program (projected to C_end plus a zero error word); the CUDA side is a `teacher`:
    it reads C_start from the oracle side's own outputs and calls the phase's custom call."""
    import jax
    start, end, label = _phase_range(argument)
    plan, bs = _context_plan([start, end], ctx)
    if start not in plan.carry or end not in plan.carry:raise KeyError(f'phase {argument}: boundaries {start}, {end} not in the plan')
    generated = compile_phase(plan, start, end, _build_dir(ctx), mode=ctx.get('mode') or 'thread')
    full = oracle_full(ctx, plan, bs)
    ins = [e.name for e in plan.carry[start]];outs = [e.name for e in plan.carry[end]]
    def reference(s, a, t, c):
        return full(s, a, t)
    def project(want):
        out = {f'{end}/{n}': want[f'{end}/{n}'] for n in outs}
        out[f'err/{label}'] = _zero_err()
        return out
    reference.project = project
    call = jax.jit(lambda *xs: generated(*xs))
    def cuda(s, a, t, c):   # unused: the teacher replaces the CUDA side's own call
        raise RuntimeError('phase: the CUDA side is teacher-forced')
    def teacher(want):
        got, err = call(*[want[f'{start}/{n}'] for n in ins])
        out = {f'{end}/{n}': x for n, x in zip(outs, got)}
        out[f'err/{label}'] = err[:1]
        return out
    cuda.teacher = teacher
    matches, differing = _spec_check(plan)
    info = dict(provider='phase', phase=label, start=start, end=end, implementation='generated', library=str(generated.path),
        digest=generated.module.digest, chunks=generated.module.stats.get('chunks'), slice=plan.slice(start, end).stats,
        aliased_outputs=len(plan.slice(start, end).aliases), spec_matches_committed=matches, spec_differing_boundaries=differing)
    reference.info = cuda.info = info
    return reference, cuda


def build_chain(plan, units, ctx, build_dir=None, mode='thread', committed=None):
    """(run, implementations): run(C_0 list) -> ({boundary: carry list}, {unit: err}) over the units, jittable; hand phases
    must match the committed spec at their boundaries."""
    import jax
    import jax.numpy as j
    names = {b: [e.name for e in plan.carry[b]] for b in plan.order}
    steps = [];impl = []
    for u in units:
        if u.kind == 'gen':
            g = compile_phase(plan, u.start, u.end, build_dir, mode=mode)
            steps.append((u, lambda xs, g=g: g(*xs)));impl.append(dict(unit=u.name, kind='gen', start=u.start, end=u.end, library=str(g.path), digest=g.module.digest))
            continue
        hp = HAND_PHASES[u.name];committed = committed if committed is not None else (load_spec() if SPEC.exists() else None)
        if committed is None:raise RuntimeError('hand phases need the committed gpu_engine/v2e_phases_spec.json')
        sig = {b['name']: carry_signature(b['carry']) for b in committed['boundaries']}
        for b in (u.start, u.end):
            live = [(e.name, e.dtype, list(e.shape)) for e in plan.carry[b]]
            if sig.get(b) != live:raise RuntimeError(f'hand phase {u.name}: the live carry at {b} differs from the committed spec (regenerate and rebuild)')
        def run_hand(xs, hp=hp, u=u):
            res = hp.fn(dict(zip(names[u.start], xs)), ctx)
            out, err = res if isinstance(res, tuple) else (res, _zero_err())
            if sorted(out) != sorted(names[u.end]):raise ValueError(f'hand phase {hp.name}: returns {len(out)} names, C_{u.end} has {len(names[u.end])}')
            return [out[n] for n in names[u.end]], j.asarray(err, j.uint32).reshape(-1)[:1]
        steps.append((u, run_hand));impl.append(dict(unit=u.name, kind='hand', start=u.start, end=u.end, **hp.info))
    def run(c0):
        carries = {'b0': list(c0)};errs = {};cur = list(c0)
        for u, fn in steps:
            cur, err = fn(cur);carries[u.end] = cur;errs[u.name] = err[:1]
        return carries, errs
    return run, impl


def diff_chain(argument, ctx):
    """v2e_diff provider: a chain of generated and hand phases from the oracle's C_0 (teacher input), every boundary the
    chain produces compared with the oracle's tapped carry; the last is the full 148-leaf step output (the tapped
    program's, which tap_identity proves equal to the unmodified oracle's). summary() reports the first differing
    boundary (then leaf, then lane) and differing lane-steps per boundary."""
    import jax
    load_hand_phases()
    units = parse_chain(argument)
    plan, bs = _context_plan([], ctx)
    run, impl = build_chain(plan, units, ctx, _build_dir(ctx), mode=ctx.get('mode') or 'thread')
    produced = [u.end for u in units];names = {b: [e.name for e in plan.carry[b]] for b in plan.order}
    full = oracle_full(ctx, plan, bs);call = jax.jit(run)
    def reference(s, a, t, c):
        return full(s, a, t)
    def project(want):
        out = {f'{b}/{n}': want[f'{b}/{n}'] for b in produced for n in names[b]}
        out.update({f'err/{u.name}': _zero_err() for u in units})
        return out
    reference.project = project
    def cuda(s, a, t, c):
        raise RuntimeError('chain: the CUDA side runs from the oracle C_0 (teacher)')
    def teacher(want):
        carries, errs = call([want[f'b0/{n}'] for n in names['b0']])
        out = {f'{b}/{n}': x for b in produced for n, x in zip(names[b], carries[b])}
        out.update({f'err/{k}': v for k, v in errs.items()})
        return out
    cuda.teacher = teacher
    observe, summary = _observer(produced + ['err'])
    reference.observe_diffs = observe;reference.summary = summary
    matches, differing = _spec_check(plan)
    reference.info = cuda.info = dict(provider='chain', chain=argument, units=impl, spec_matches_committed=matches, spec_differing_boundaries=differing)
    return reference, cuda


# ---------------------------------------------------------------------------
# Commands.

def _context(args):
    from gpu_engine import v2e_census as cz
    return cz.load_context(args)


def _example(args, ctx):
    from gpu_engine import v2e_census as cz
    if getattr(args, 'shards', None):return cz.shard_tree(cz.shard_paths(args.shards, ctx['template'])[0], ctx['template'], lanes=1)
    return ctx['template']


def log(message, **values):
    print(json.dumps(dict(message=message, time=time.strftime('%H:%M:%S'), **values), default=str), flush=True)


def plan_markdown(plan, spec):
    s = plan.summary()
    out = ['| Boundary | Line | Lane vars | Per-lane elements | state | actions | trackers | chance | local | Tapped leaves | Constant-derived |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    lines = {b['name']: b for b in spec['boundaries']}
    for row in s['boundaries']:
        b = lines.get(row['boundary'], {});src = row['by_source']
        out.append(f"| {row['boundary']} | {b.get('line') or '-'} | {row['lane_vars']} | {row['elements']:,} | {src['state']} | {src['actions']} | "
            f"{src['trackers']} | {src['chance']} | {src['local']} | {row['tapped'] or '-'} | {row['constant'] or 0} |")
    out += ['', '| Phase | From | To | Top-level equations | Leaf equations | Lane equations | Inputs | Outputs | Aliased |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, a, b in PHASES:
        st = plan.slices[(a, b)].stats
        out.append(f"| {name} | {a} | {b} | {st['top_level']:,} | {st['leaf']:,} | {st['lane_top_level']:,} | {st['inputs']} | {st['outputs']} | {st['aliased']} |")
    out += ['', f"Whole tapped program: {s['whole_program']['top_level']:,} top-level, {s['whole_program']['leaf']:,} leaf equations. "
        f"Lane equations computed by more than one phase: {s['recomputed_lane_equations']}."]
    return '\n'.join(out) + '\n'


def cmd_spec(args):
    """Trace the step's plan and write the spec (and, with --report, plan.json/plan.md)."""
    ctx = _context(args)
    from gpu_engine import v2e_diff
    if args.shards:ctx['template'] = _example(args, ctx)
    plan = step_plan(ctx)
    digest, modules = v2e_diff.oracle_digest()
    spec = validate_spec(spec_of(plan, oracle_digest=digest), output_leaves=len(plan.carry[plan.final]))
    out = Path(args.out);out.write_text(json.dumps(spec, indent=1) + '\n')
    log('SPEC', out=str(out), oracle_digest=digest, oracle_modules=len(modules), trace_seconds=plan.meta.get('trace_seconds'),
        carry={b['name']: b['lane_vars'] for b in spec['boundaries']}, recomputed=plan.recomputed())
    if args.report:
        report = Path(args.report);report.mkdir(parents=True, exist_ok=False)
        (report / 'plan.json').write_text(json.dumps(dict(plan.summary(), meta=plan.meta, oracle_digest=digest), indent=1, default=str) + '\n')
        (report / 'plan.md').write_text(plan_markdown(plan, spec))
        if args.sub:
            sub = step_plan(ctx, sub=True)
            (report / 'plan_sub.json').write_text(json.dumps(dict(sub.summary(), meta=sub.meta), indent=1, default=str) + '\n')
    return 0


# -- timings ----------------------------------------------------------------------------------------

def live_shape():
    """The live DepthN pricer's engine-step lane counts (v2e_phase_profile.LIVE; device_depthn_pricer, device_depth2_pricer):
    P1 (turn 1) steps every world's 81 children at once: capacity x 2 worlds x JOINT_LANES lanes, one step per call;
    a block program steps B = block_lanes // K nodes x K = k_own x k_opp pairs per block (a vmap over nodes of a vmap
    over pairs of executor.advance at batch 1), blocks_per_call blocks per call under jax.lax.map, so blocks_per_call
    sequential steps per call (a block without a live node is skipped by lax.cond)."""
    from gpu_engine.v2e_phase_profile import LIVE
    capacity = LIVE['bucket'] * LIVE['worlds'] // 2   # roots per call (make_pricers: capacity b x W/2)
    k = LIVE['k'] * LIVE['k'];joint = 81                # K pairs per node; device_public_teacher.JOINT_LANES children per world
    nodes = max(1, LIVE['block_lanes'] // k)          # device_depthn_pricer.block_nodes
    p1 = capacity * 2 * joint                          # turn-1 children: 2 worlds per root x 81
    blocks = -(-p1 // nodes)                           # the leaf ply over every turn-1 child at full width
    return dict(source='v2e_phase_profile.LIVE via depthn_live_search.make_pricers -> device_depthn_pricer.DepthNPricer', live=dict(LIVE),
        roots_per_call=capacity, p1_step_lanes=p1, p1_steps_per_call=1, block_nodes=nodes, pairs_per_node=k, block_step_lanes=nodes * k,
        block_steps_per_call=LIVE['blocks_per_call'], full_ply_blocks=blocks, full_ply_calls=-(-blocks // LIVE['blocks_per_call']),
        note='depth 2: P1 is one step of p1_step_lanes; the leaf ply steps every live turn-1 child x K pairs in blocks of block_step_lanes, '
            'block_steps_per_call sequential steps per device call (jax.lax.map; a block with no live node is skipped)')


def _timed(fn, make_args, *, calls, warm):
    import jax
    times = []
    for i in range(warm + calls):
        xs = make_args();jax.block_until_ready(xs)
        begin = time.perf_counter();jax.block_until_ready(fn(*xs));elapsed = 1e3 * (time.perf_counter() - begin)
        if i >= warm:times.append(elapsed)
    q = lambda p: float(np.percentile(np.asarray(times), p))
    return dict(calls=calls, warm=warm, ms_median=statistics.median(times), ms_p90=q(90), ms_min=min(times), ms_mean=statistics.fmean(times))


def cmd_timings(args):
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    ctx = _context(args)
    import jax
    import jax.numpy as j
    from gpu_engine import transition, v2e_census as cz, v2e_gen, v2e_phase_profile as pp
    ctx['build_dir'] = args.build_dir
    plan = step_plan(ctx);paths = cz.shard_paths(args.shards, ctx['template'])
    units = [Unit('gen', name, a, b) for name, a, b in PHASES]
    gens = {u.name: compile_phase(plan, u.start, u.end, args.build_dir) for u in units}
    gens['FG'] = compile_phase(plan, *COMPOSITES['FG'], args.build_dir)
    run, impl = build_chain(plan, units, ctx, args.build_dir);chain = jax.jit(run)
    names = {b: [e.name for e in plan.carry[b]] for b in plan.order}
    step_fn = v2e_gen.compile_function('step', pp.step_function(ctx), ctx['template'], build_dir=args.build_dir, mode='thread') if not args.no_step else None
    def c0_of(tree):
        s, a, t = tree;c = transition.sample_chance_inputs(s)
        values = dict({'state' + p: x for p, x in zip(_paths(s), jax.tree.leaves(s))}, **{'actions' + p: x for p, x in zip(_paths(a), jax.tree.leaves(a))},
            **{'trackers' + p: x for p, x in zip(_paths(t), jax.tree.leaves(t))}, **{'chance_inputs' + p: x for p, x in zip(_paths(c), jax.tree.leaves(c))})
        return [values[n] for n in names['b0']]
    c0 = jax.jit(c0_of)
    copy = jax.jit(lambda xs: [j.array(x, copy=True) for x in xs])   # fresh buffers to donate
    report = dict(schema='pokezero.v2e-phase-timings.v1', source_commit=os.environ.get('V2_SOURCE_COMMIT'), calls=args.calls, warm=args.warm,
        method='wall ms per call (jax.block_until_ready), inputs donated (aliased carry leaves stay in place) and copied fresh outside the timed window',
        libraries={k: dict(path=str(g.path), digest=g.module.digest, chunks=g.module.stats.get('chunks'), aliased=len(g.module.aliases)) for k, g in gens.items()},
        live_shape=live_shape(), lanes={})
    lane_counts = list(args.lanes)
    if args.live:lane_counts += [report['live_shape']['block_step_lanes'], report['live_shape']['p1_step_lanes']]
    for lanes in lane_counts:
        tree, sources = pp.batch_for(paths, ctx['template'], lanes);tree = pp.device_tree(tree)
        carries, errs = chain(c0(tree));errors = {k: int(np.asarray(v)[0]) for k, v in errs.items()}
        entry = dict(sources=sources, error_words=errors, phases={})
        live_only = lanes in (report['live_shape']['block_step_lanes'], report['live_shape']['p1_step_lanes']) and lanes not in args.lanes
        todo = ['FG'] if live_only else [u.name for u in units] + ['FG']
        for name in todo:
            a, b = (COMPOSITES[name] if name in COMPOSITES else next((x, y) for n, x, y in PHASES if n == name))
            g = gens[name];inputs = carries[a]
            f = jax.jit(lambda *xs, g=g: g(*xs), donate_argnums=tuple(range(len(inputs))))
            entry['phases'][name] = _timed(f, lambda: copy(inputs), calls=args.calls, warm=args.warm)
            log('PHASE_TIMING', lanes=lanes, phase=name, ms=entry['phases'][name]['ms_median'])
        f = jax.jit(run, donate_argnums=(0,))
        entry['chain'] = _timed(f, lambda: (copy(carries['b0']),), calls=args.calls, warm=args.warm)
        entry['chain_with_chance'] = _timed(jax.jit(lambda s, a, t: run(c0_of((s, a, t)))), lambda: tree, calls=args.calls, warm=args.warm)
        if step_fn is not None and not live_only:
            entry['monolithic_step'] = _timed(jax.jit(lambda s, a, t: step_fn(s, a, t)), lambda: tree, calls=args.calls, warm=args.warm)
        entry['sum_of_phases_ms'] = sum(entry['phases'][u.name]['ms_median'] for u in units if u.name in entry['phases']) or None
        report['lanes'][str(lanes)] = entry
        log('TIMINGS', lanes=lanes, chain_ms=entry['chain']['ms_median'], step_ms=(entry.get('monolithic_step') or {}).get('ms_median'), errors=errors)
        (out / 'timings.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    (out / 'timings.md').write_text(timings_markdown(report))
    log('DONE', out=str(out / 'timings.json'))
    return 0


def timings_markdown(report):
    lanes = list(report['lanes'])
    head = '| Phase | ' + ' | '.join(f'{l} lanes ms (median / p90)' for l in lanes) + ' |'
    out = [head, '| --- |' + ' ---: |' * len(lanes)]
    names = [p[0] for p in PHASES] + ['FG']
    for name in names:
        cells = []
        for l in lanes:
            t = report['lanes'][l]['phases'].get(name)
            cells.append(f"{t['ms_median']:.3f} / {t['ms_p90']:.3f}" if t else '-')
        out.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    for key, title in (('chain', 'chain (C_0 -> b8, 8 calls)'), ('chain_with_chance', 'chain + chance from (state, actions, trackers)'),
            ('monolithic_step', 'monolithic generated step')):
        cells = []
        for l in lanes:
            t = report['lanes'][l].get(key)
            cells.append(f"{t['ms_median']:.3f} / {t['ms_p90']:.3f}" if t else '-')
        out.append(f'| {title} | ' + ' | '.join(cells) + ' |')
    ls = report['live_shape']
    out += ['', f"Live pricer engine steps: P1 {ls['p1_step_lanes']} lanes x {ls['p1_steps_per_call']} per call; block program "
        f"{ls['block_step_lanes']} lanes ({ls['block_nodes']} nodes x {ls['pairs_per_node']} pairs) x {ls['block_steps_per_call']} sequential steps per call; "
        f"a full ply is {ls['full_ply_blocks']} blocks in {ls['full_ply_calls']} calls."]
    return '\n'.join(out) + '\n'


# -- float sites --------------------------------------------------------------------------------------

def float_summary(sites):
    """Per source line ('file:line'): sites by lowering; the distinct reciprocal constants (divisor -> reciprocal bits)."""
    rows = {}
    for s in sites:
        r = rows.setdefault(s.get('where') or '?', dict(sites=0, by_lowering={}, by_category={}, primitives={}, step_lines=set(), reciprocals={}, modules=set()))
        r['sites'] += 1;low = s['lowering'];r['by_lowering'][low] = r['by_lowering'].get(low, 0) + 1
        r['by_category'][s['category']] = r['by_category'].get(s['category'], 0) + 1
        r['primitives'][s['primitive']] = r['primitives'].get(s['primitive'], 0) + 1
        if s.get('step_line') is not None:r['step_lines'].add(s['step_line'])
        if s.get('module'):r['modules'].add(s['module'])
        if low in ('recip_mul', 'folded_recip_mul'):
            for d, q in zip(s.get('divisor_bits', []), s.get('reciprocal_bits', [])):r['reciprocals'][d] = q
    for r in rows.values():r['step_lines'] = sorted(r['step_lines']);r['modules'] = sorted(r['modules'])
    return rows


def floats_markdown(name, rows):
    out = [f'### {name}', '', '| Source line | f32 sites | arith | rn add/sub/mul | div.full | host recip x mul (C -> 1/C bits) | folded | other lowerings |',
        '| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |']
    def key(item):
        where = item[0];f, _, l = where.rpartition(':')
        return (f, int(l) if l.isdigit() else 0)
    for where, r in sorted(rows.items(), key=key):
        if not r['by_category'].get('arith'):continue
        low = r['by_lowering'];recip = ', '.join(f'{d}->{q}' for d, q in list(r['reciprocals'].items())[:4]) + (' ...' if len(r['reciprocals']) > 4 else '')
        rm = low.get('recip_mul', 0) + low.get('folded_recip_mul', 0)
        other = ', '.join(f'{k} {v}' for k, v in sorted(low.items()) if k not in ('rn', 'div.full', 'recip_mul', 'folded_recip_mul', 'folded'))
        out.append(f"| {where} | {r['sites']} | {r['by_category'].get('arith', 0)} | {low.get('rn', 0)} | {low.get('div.full', 0)} | "
            f"{rm}{(' (' + recip + ')') if recip else ''} | {low.get('folded', 0)} | {other} |")
    return '\n'.join(out) + '\n'


def cmd_floats(args):
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    ctx = _context(args)
    from gpu_engine import v2e_gen, v2e_j2c, v2e_modules
    state, actions, trackers = _example(args, ctx)
    modules = [('step', v2e_modules.step(ctx['engine_tables'], ctx['observation_tables']), (state, actions, trackers), 'thread'),
        ('observe', v2e_modules.observe(ctx['engine_tables'], ctx['observation_tables']), (state, trackers), 'warp')]
    report = dict(schema='pokezero.v2e-float-sites.v1', source_commit=os.environ.get('V2_SOURCE_COMMIT'),
        lowerings=dict(rn='an IEEE round-to-nearest helper (v2e::fadd/fsub/fmul)', **{'div.full': 'runtime division v2e::fdiv (div.full.f32)'},
            recip_mul="XLA's A / C -> A * RN(1 / C): host reciprocal (reciprocal_bits) times A at run time", folded_recip_mul='the same, folded on the host',
            folded='evaluated on the host as XLA constant-folds it', identity='an identity rewrite (x + 0, x * 1, x / 1, neg(neg(x)))',
            nan_literal='arithmetic with a NaN literal folds to that NaN, quieted', cse='reuses an earlier identical equation',
            other='another helper (fmin_x/fmax_x, conversions, ...)', convert='a conversion', compare='a comparison', data='data movement', reduce='a reduction'),
        modules={})
    text = ['# Float sites: the generator\'s lowering of every f32 equation', '']
    for name, fn, example, mode in modules:
        begin = time.perf_counter()
        with v2e_modules.trace_substitutions():
            closed, avals, _, _ = v2e_gen.trace(lambda *a: fn(*a), tuple(example))
        sites = [];v2e_j2c.generate(name, closed, avals, mode=mode, float_sites=sites)
        rows = float_summary(sites)
        totals = {}
        for s in sites:totals[s['lowering']] = totals.get(s['lowering'], 0) + 1
        report['modules'][name] = dict(mode=mode, seconds=time.perf_counter() - begin, sites=len(sites), by_lowering=totals,
            arith_sites=sum(1 for s in sites if s['category'] == 'arith'), by_line=rows)
        (out / f'float_sites_{name}.json').write_text(json.dumps(sites, indent=None, default=str) + '\n')
        log('FLOATS', module=name, sites=len(sites), by_lowering=totals)
        text += [f"{name}: {len(sites)} f32 equations ({report['modules'][name]['arith_sites']} arithmetic); by lowering: "
            + ', '.join(f'{k} {v}' for k, v in sorted(totals.items())) + '.', '', floats_markdown(name, rows)]
    (out / 'float_sites.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    (out / 'float_sites.md').write_text('\n'.join(text))
    log('DONE', out=str(out / 'float_sites.json'))
    return 0


def cmd_build(args):
    """Generate and build every phase library (and FG) into --build-dir, so later gates find them cached."""
    ctx = _context(args);plan = step_plan(ctx);rows = {}
    for name, (a, b) in [(n, (x, y)) for n, x, y in PHASES] + list(COMPOSITES.items()):
        g = compile_phase(plan, a, b, args.build_dir)
        rows[name] = dict(path=str(g.path), digest=g.module.digest, aliased=len(g.module.aliases), **{k: g.module.stats.get(k) for k in
            ('chunks', 'lines', 'consts', 'cse_hits', 'folds', 'generate_s', 'build_s', 'equations')}, arena_words=g.module.arena_words)
        log('PHASE_BUILT', phase=name, **rows[name])
    if args.out:Path(args.out).write_text(json.dumps(rows, indent=1, default=str) + '\n')
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='command', required=True)
    def common(q, shards_required=False):
        q.add_argument('--config', required=True);q.add_argument('--reference-run', required=True)
        q.add_argument('--arm', default='B', choices=('A', 'B'));q.add_argument('--games', type=int, default=1)
        q.add_argument('--shards', required=shards_required, help='capture shard directory (shards/ beside capture.json)')
    q = sub.add_parser('spec', help='trace the plan and write the spec');common(q)
    q.add_argument('--out', default=str(SPEC), help='spec path (default: the committed gpu_engine/v2e_phases_spec.json)')
    q.add_argument('--report', help='new directory for plan.json / plan.md');q.add_argument('--sub', action='store_true', help='also the sub-boundary plan')
    q = sub.add_parser('timings', help='generated phase, chain and step timings');common(q, shards_required=True)
    q.add_argument('--build-dir', required=True);q.add_argument('--out', required=True)
    q.add_argument('--lanes', type=int, nargs='+', default=[128, 2048, 8192]);q.add_argument('--calls', type=int, default=50)
    q.add_argument('--warm', type=int, default=5);q.add_argument('--live', action='store_true', help='also the live pricer\'s step lane counts')
    q.add_argument('--no-step', action='store_true', help='skip the monolithic generated step')
    q = sub.add_parser('floats', help='float-site report of the step and observe');common(q)
    q.add_argument('--out', required=True)
    q = sub.add_parser('build', help='generate and build every phase library into --build-dir');common(q)
    q.add_argument('--build-dir', required=True);q.add_argument('--out', help='JSON of the built libraries')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    return dict(spec=cmd_spec, timings=cmd_timings, floats=cmd_floats, build=cmd_build)[args.command](args)


if __name__ == '__main__':raise SystemExit(main())
