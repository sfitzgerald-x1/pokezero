"""Generate per-lane CUDA kernels from oracle jaxprs (jaxpr -> CUDA C++).

The JAX engine is a straight-line, row-independent program over a leading
lane axis. A function is traced with lane size LANES (an unusual size, so the
lane axis is recognizable: every shape whose axis 0 has LANES elements is a
per-lane value; LANES may appear nowhere else). One CUDA thread executes one
lane: each jaxpr equation becomes a loop over that lane's elements, with every
primitive implemented by the XLA-exact scalar helpers of cuda/v2e_jax.cuh.

Storage: kernel inputs and outputs are lane-major device buffers in their own
dtypes; closure constants (engine/observation tables) are extra inputs without
a lane axis; every intermediate lives in a 32-bit-word arena laid out
element-major across lanes (word w of lane b at arena[w * lanes + b], so
neighbouring threads touch neighbouring words). Arena slots are reused by
liveness; reshape/squeeze/no-op converts are views; elementwise ops and
scatters update a dying operand in place.

Reads of a per-lane operand at another lane cannot happen in this engine (all
indexing uses the lane iota); gathers/scatters that would cross lanes set an
error word instead, which callers check.
"""
from dataclasses import dataclass, field
import hashlib
import math

import numpy as np

LANES = 37

CTYPE = {np.dtype(np.bool_): 'bool', np.dtype(np.int8): 'int8_t', np.dtype(np.uint8): 'uint8_t',
    np.dtype(np.int16): 'int16_t', np.dtype(np.uint16): 'uint16_t', np.dtype(np.int32): 'int32_t',
    np.dtype(np.uint32): 'uint32_t', np.dtype(np.float32): 'float'}
SIGNED = {np.dtype(np.int8), np.dtype(np.int16), np.dtype(np.int32)}
UNSIGNED = {np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.uint32)}
F32 = np.dtype(np.float32)
BOOL = np.dtype(np.bool_)


class Unsupported(Exception):
    pass


@dataclass(eq=False)
class Slot:
    off: int
    words: int
    refs: int = 0             # pending reads of every value stored in this slot


@dataclass(eq=False)
class Val:
    dtype: np.dtype
    shape: tuple
    kind: str                 # 'arena' | 'in' | 'const' | 'lit' | 'lane_id'
    slot: object = None       # Slot for 'arena' values (shared by views and in-place results)
    ptr: str = ''             # pointer expression for 'in'/'const'
    lit: object = None        # python scalar for 'lit'

    @property
    def off(self):
        return self.slot.off

    @property
    def lane(self):
        return len(self.shape) > 0 and self.shape[0] == LANES

    @property
    def lshape(self):
        return tuple(self.shape[1:]) if self.lane else tuple(self.shape)

    @property
    def size(self):
        return int(math.prod(self.lshape))


def ctype(dtype):
    dtype = np.dtype(dtype)
    if dtype not in CTYPE:raise Unsupported(f'dtype {dtype}')
    return CTYPE[dtype]


def literal(value, dtype):
    dtype = np.dtype(dtype)
    if dtype == BOOL:return 'true' if bool(value) else 'false'
    if dtype == F32:
        bits = int(np.asarray(value, np.float32).view(np.uint32))
        return f'__uint_as_float(0x{bits:08x}u)'
    v = int(np.asarray(value, dtype))
    if dtype in UNSIGNED:return f'(({ctype(dtype)}){v}u)'
    if v == -2**31:return '((int32_t)0x80000000u)'
    return f'(({ctype(dtype)}){v})'


def word_to(dtype, w):
    dtype = np.dtype(dtype)
    if dtype == F32:return f'__uint_as_float({w})'
    if dtype == BOOL:return f'(({w}) != 0u)'
    return f'(({ctype(dtype)})({w}))'


def to_word(dtype, x):
    dtype = np.dtype(dtype)
    if dtype == F32:return f'__float_as_uint({x})'
    if dtype in SIGNED and dtype.itemsize < 4:return f'(uint32_t)(int32_t)({x})'
    return f'(uint32_t)({x})'


def strides(shape):
    out = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):out[i] = out[i + 1] * shape[i + 1]
    return out


class Kernel:
    """Code generator state for one kernel."""

    CHUNK_LINES = 1500

    def __init__(self, name):
        self.name = name;self.lines = [];self.chunks = [];self.indent = 1;self.words = 0;self.free = [];self.peak = 0
        self.smem = 0;self.chunk_smem = []
        self.inputs = [];self.consts = [];self.outputs = [];self.loop = 0;self.lane_literals = 0

    # -- emission ------------------------------------------------------------
    def emit(self, line):
        self.lines.append('  ' * self.indent + line)

    def maybe_split(self):
        """Close the current kernel chunk at an equation boundary (no C variable spans it)."""
        if len(self.lines) >= self.CHUNK_LINES and self.indent == 1:
            self.chunks.append(self.lines);self.chunk_smem.append(self.smem);self.lines = [];self.smem = 0

    def var(self, prefix='t'):
        self.loop += 1
        return f'{prefix}{self.loop}'

    # -- arena -----------------------------------------------------------------
    def alloc(self, words):
        words = max(1, int(words))
        for i, (off, n) in enumerate(self.free):
            if n >= words:
                self.free[i] = (off + words, n - words)
                if self.free[i][1] == 0:self.free.pop(i)
                return off
        off = self.words;self.words += words;self.peak = max(self.peak, self.words)
        return off

    def release(self, off, words):
        self.free.append((off, max(1, int(words))));self.free.sort()
        merged = []
        for o, n in self.free:
            if merged and merged[-1][0] + merged[-1][1] == o:merged[-1] = (merged[-1][0], merged[-1][1] + n)
            else:merged.append((o, n))
        self.free = merged

    # -- element access ----------------------------------------------------------
    def load(self, v, idx):
        """C expression for element `idx` (per-lane flat index expression) of v."""
        if v.kind == 'lit':return literal(v.lit, v.dtype)
        if v.kind == 'lane_id':return f'(({ctype(v.dtype)})b)'
        if v.kind == 'arena':return word_to(v.dtype, f'A[{v.off} + ({idx})]')
        if v.kind == 'in':
            if v.lane:return f'{v.ptr}[(size_t)b * {v.size} + ({idx})]'
            return f'{v.ptr}[{idx}]'
        if v.kind == 'const':return f'{v.ptr}[{idx}]'
        raise Unsupported(v.kind)

    def store(self, v, idx, expr):
        self.emit(f'A[{v.off} + ({idx})] = {to_word(v.dtype, expr)};')

    def loop_over(self, n, body):
        """Warp-parallel loop: element i of n is handled by thread i % 32 of the lane's warp."""
        if n == 0:return
        i = self.var('ix')
        self.emit(f'for (int {i} = t; {i} < {n}; {i} += 32) {{');self.indent += 1
        body(i)
        self.indent -= 1;self.emit('}')

    def serial_loop(self, n, body):
        """Sequential loop inside one thread."""
        if n == 0:return
        if n == 1:
            body('0');return
        i = self.var('ix')
        self.emit(f'for (int {i} = 0; {i} < {n}; ++{i}) {{');self.indent += 1
        body(i)
        self.indent -= 1;self.emit('}')

    def coords(self, i, shape):
        """Per-axis coordinate expressions of flat index i over shape."""
        st = strides(shape);out = []
        for d, (s, n) in enumerate(zip(st, shape)):
            if n == 1:out.append('0')
            elif d == 0:out.append(f'(({i}) / {s})' if s > 1 else f'({i})')
            else:out.append(f'((({i}) / {s}) % {n})' if s > 1 else f'(({i}) % {n})')
        return out


# ---------------------------------------------------------------------------
# Elementwise primitive semantics.

def binary(prim, dtype, a, b):
    t = np.dtype(dtype);c = ctype(t)
    if t == F32:
        table = dict(add=f'v2e::fadd({a}, {b})', sub=f'v2e::fsub({a}, {b})', mul=f'v2e::fmul({a}, {b})',
            div=f'v2e::fdiv({a}, {b})', rem=f'v2e::frem({a}, {b})', max=f'v2e::fmax_x({a}, {b})', min=f'v2e::fmin_x({a}, {b})')
    elif t == BOOL:
        table = dict(**{'and': f'({a} && {b})', 'or': f'({a} || {b})', 'xor': f'({a} != {b})'},
            max=f'({a} || {b})', min=f'({a} && {b})', add=f'({a} || {b})', mul=f'({a} && {b})')
    else:
        u = c if t in UNSIGNED else 'u' + c
        wrap = lambda op: f'(({c})(({u})({a}) {op} ({u})({b})))'
        table = dict(add=wrap('+'), sub=wrap('-'), mul=wrap('*'),
            max=f'(({a}) > ({b}) ? ({a}) : ({b}))', min=f'(({a}) < ({b}) ? ({a}) : ({b}))',
            **{'and': f'(({c})(({a}) & ({b})))', 'or': f'(({c})(({a}) | ({b})))', 'xor': f'(({c})(({a}) ^ ({b})))'},
            shift_left=f'v2e::shl<{c}>({a}, {b})', shift_right_logical=f'v2e::shr_logical<{c}>({a}, {b})',
            shift_right_arithmetic=f'v2e::shr_arith<{c}>({a}, {b})')
        if t == np.dtype(np.int32):table.update(div=f'v2e::idiv({a}, {b})', rem=f'v2e::irem({a}, {b})')
        elif t == np.dtype(np.uint32):table.update(div=f'v2e::udiv({a}, {b})', rem=f'v2e::urem({a}, {b})')
    if prim not in table:raise Unsupported(f'{prim} on {t}')
    return table[prim]


def compare(prim, a, b):
    return {'eq': f'({a} == {b})', 'ne': f'({a} != {b})', 'lt': f'({a} < {b})', 'le': f'({a} <= {b})',
        'gt': f'({a} > {b})', 'ge': f'({a} >= {b})'}[prim]


def unary(prim, dtype, a, params):
    t = np.dtype(dtype);c = ctype(t)
    if prim == 'not':return f'(!{a})' if t == BOOL else f'(({c})~({a}))'
    if t == F32:
        table = dict(neg=f'v2e::fneg({a})', abs=f'v2e::fabs_x({a})', sign=f'v2e::fsign({a})', floor=f'v2e::ffloor({a})',
            ceil=f'v2e::fceil({a})', is_finite=f'v2e::fisfinite({a})',
            round=f'v2e::fround_even({a})' if int(params.get('rounding_method', 0)) == 1 else f'v2e::fround_away({a})')
    else:
        table = dict(neg=f'(({c})(0u - (uint32_t)({a})))', abs=f'(({a}) < 0 ? ({c})(0u - (uint32_t)({a})) : ({a}))',
            sign=f'(({c})((({a}) > 0) - (({a}) < 0)))' if t in SIGNED else f'(({c})(({a}) != 0))',
            population_count=f'(({c})__popc((uint32_t)({a})))' if t.itemsize == 4 else f'(({c})__popc((uint32_t)(({a}) & 0x{(1 << 8 * t.itemsize) - 1:x}u)))',
            clz=f'(({c})__clz((int)({a})))')
    if prim not in table:raise Unsupported(f'{prim} on {t}')
    return table[prim]


def convert(src, dst, a):
    src, dst = np.dtype(src), np.dtype(dst)
    if src == dst:return a
    if dst == BOOL:return f'({a} != 0)' if src != F32 else f'({a} != 0.0f)'
    if src == BOOL:return f'(({ctype(dst)})({a} ? 1 : 0))' if dst != F32 else f'({a} ? 1.0f : 0.0f)'
    if dst == F32:
        if src == np.dtype(np.uint32):return f'v2e::u2f({a})'
        return f'v2e::s2f((int32_t)({a}))'
    if src == F32:
        if dst == np.dtype(np.int32):return f'v2e::f2s32({a})'
        if dst == np.dtype(np.uint32):return f'v2e::f2u32({a})'
        info = np.iinfo(dst)
        return f'(({ctype(dst)})v2e::f2s_narrow({a}, {int(info.min)}, {int(info.max)}))'
    return f'(({ctype(dst)})({a}))'  # integer <-> integer: wrap (two's complement)


ELEMENTWISE_BINARY = {'add', 'sub', 'mul', 'div', 'rem', 'max', 'min', 'and', 'or', 'xor',
    'shift_left', 'shift_right_logical', 'shift_right_arithmetic'}
COMPARE = {'eq', 'ne', 'lt', 'le', 'gt', 'ge'}
UNARY = {'not', 'neg', 'abs', 'sign', 'floor', 'ceil', 'round', 'is_finite', 'population_count', 'clz'}
VIEWS = {'reshape', 'squeeze', 'expand_dims'}


@dataclass
class Module:
    name: str
    files: dict                                   # file name -> generated source
    arena_words: int
    inputs: list = field(default_factory=list)    # [(dtype, shape)] lane inputs (traced shapes)
    consts: list = field(default_factory=list)    # numpy arrays
    outputs: list = field(default_factory=list)   # [(dtype, shape)] traced shapes
    lane_literals: int = 0                        # integer literals equal to LANES (index normalization)

    @property
    def digest(self):
        h = hashlib.sha256()
        for name in sorted(self.files):h.update(name.encode());h.update(self.files[name].encode())
        return h.hexdigest()[:16]

    @property
    def source_lines(self):
        return sum(text.count(chr(10)) for text in self.files.values())


class Generator:
    def __init__(self, name, profile=False):
        self.k = Kernel(name);self.env = {};self.profile = profile;self.marks = []

    # -- values ------------------------------------------------------------------
    def read(self, atom):
        from jax.extend import core as jcore
        if isinstance(atom, jcore.Literal):
            val = np.asarray(atom.val)
            if val.shape:raise Unsupported('non-scalar literal')
            if val.dtype.kind in 'iu' and int(val) in (LANES, -LANES, LANES - 1):self.k.lane_literals += 1
            return Val(np.dtype(atom.aval.dtype), (), 'lit', lit=val.item())
        return self.env[atom]

    def bind(self, var, val):
        self.env[var] = val

    def unref(self, slot):
        slot.refs -= 1
        if slot.refs == 0 and slot.off >= 0:
            self.k.release(slot.off, slot.words);slot.off = -1

    # -- traversal ------------------------------------------------------------------
    def run(self, jaxpr, args):
        """Generate code for jaxpr with arguments bound to args; returns output Vals."""
        from jax.extend import core as jcore
        uses = {}
        for e in jaxpr.eqns:
            for a in e.invars:
                if not isinstance(a, jcore.Literal):uses[a] = uses.get(a, 0) + 1
        for a in jaxpr.outvars:
            if not isinstance(a, jcore.Literal):uses[a] = uses.get(a, 0) + 1
        saved = self.env;self.env = {}
        for var, val in zip(jaxpr.invars, args):
            self.env[var] = val
            if val.kind == 'arena':val.slot.refs += uses.get(var, 0)
        for e in jaxpr.eqns:
            ins = [self.read(a) for a in e.invars]
            pending = {}
            for v in ins:
                if v.kind == 'arena':pending[id(v.slot)] = pending.get(id(v.slot), 0) + 1
            dying = {id(v) for v in ins if v.kind == 'arena' and v.slot.refs == pending[id(v.slot)]}
            before = len(self.k.lines)
            try:outs = self.equation(e, ins, dying)
            except Unsupported as error:
                if getattr(error, 'located', False):raise
                ex = Unsupported(f'{error} [primitive {e.primitive.name}: ' + ', '.join(f'{v.kind}{v.shape}{v.dtype}' for v in ins)
                    + ' -> ' + ', '.join(str(tuple(v.aval.shape)) for v in e.outvars) + f'; params {str(e.params)[:300]}]')
                ex.located = True
                raise ex from None
            for var, val in zip(e.outvars, outs):
                dropped = type(var).__name__ == 'DropVar'
                if not dropped:self.env[var] = val
                if val.kind == 'arena':val.slot.refs += 0 if dropped else uses.get(var, 0)
            for v in ins:
                if v.kind == 'arena':self.unref(v.slot)
            for val in outs:
                if val.kind == 'arena' and val.slot.refs == 0 and val.slot.off >= 0:
                    self.k.release(val.slot.off, val.slot.words);val.slot.off = -1
            if len(self.k.lines) != before:
                self.k.emit('__syncwarp();')
                if self.profile:
                    info = e.source_info.traceback.frames[0] if getattr(e.source_info, 'traceback', None) and e.source_info.traceback.frames else None
                    where = f'{info.file_name.rsplit("/", 1)[-1]}:{info.line_num}' if info else '?'
                    self.marks.append((e.primitive.name, where, ','.join(str(tuple(v.aval.shape)) for v in e.outvars)))
                    self.k.emit(f'if (b == 0 && t == 0) err[1 + {len(self.marks) - 1}] = (uint32_t)(clock64() - T0);')
            self.k.maybe_split()
        # Each returned arena value keeps one pending reference (its outvar use);
        # the caller consumes it (see call()).
        outs = [self.read(a) for a in jaxpr.outvars]
        self.env = saved
        return outs

    def equation(self, e, ins, dying):
        p = e.primitive.name;params = e.params
        outs = [Val(np.dtype(v.aval.dtype), tuple(v.aval.shape), 'arena') for v in e.outvars]
        for o in outs:
            if LANES in o.shape[1:]:raise Unsupported(f'{p}: lane size off axis 0 in {o.shape}')
        method = getattr(self, 'p_' + p.replace('-', '_'), None)
        if method is not None:return method(e, ins, outs, params, dying)
        if p in ELEMENTWISE_BINARY:return [self.elementwise(outs[0], ins, lambda a, b: binary(p, ins[0].dtype, a, b), dying)]
        if p in COMPARE:return [self.elementwise(outs[0], ins, lambda a, b: compare(p, a, b), dying)]
        if p in UNARY:return [self.elementwise(outs[0], ins, lambda a: unary(p, ins[0].dtype, a, params), dying)]
        raise Unsupported(f'primitive {p}')

    # -- helpers ------------------------------------------------------------------
    def result(self, out, ins=(), dying=()):
        """Arena storage for out; in place over a dying same-size operand when allowed."""
        out.kind = 'arena'
        for v in ins:
            if id(v) in dying and v.kind == 'arena' and v.size == out.size and v.slot.words == max(1, out.size):
                out.slot = v.slot;out.inplace = True
                return out
        out.slot = Slot(self.k.alloc(out.size), max(1, out.size));out.inplace = False
        return out

    def elementwise(self, out, ins, fn, dying):
        index = {}
        for n, v in enumerate(ins):
            if v.kind == 'lit':continue
            if v.lshape == out.lshape and v.lane == out.lane:continue
            if v.kind == 'lane_id' and out.lane:continue
            if not v.lane and v.shape == ():index[n] = lambda i: '0';continue  # rank-0 operand broadcasts
            full_v = v.shape;full_o = out.shape
            if len(full_v) == len(full_o) and all(a == c or a == 1 for a, c in zip(full_v, full_o)) and (v.lane == out.lane or (out.lane and full_v[0] == 1)):
                vst = strides(v.lshape);voff = 1 if v.lane else 0
                def mapping(i, v=v, vst=vst, voff=voff):
                    full = (['b'] if out.lane else []) + self.k.coords(i, out.lshape)
                    parts = [f'({full[d]}) * {vst[d - voff]}' for d in range(len(full)) if d >= voff and v.shape[d] != 1]
                    return ' + '.join(parts) or '0'
                index[n] = mapping;continue
            raise Unsupported(f'elementwise shape {v.shape} -> {out.shape}')
        self.result(out, [v for n, v in enumerate(ins) if n not in index], dying)
        self.k.loop_over(out.size, lambda i: self.k.store(out, i, fn(*[self.k.load(v, index[n](i) if n in index else i) for n, v in enumerate(ins)])))
        return out

    def copy_to(self, out, src_index, src):
        """out[i] = src[src_index(i_coords)] for every out element (src_index returns flat expr)."""
        self.result(out)
        def body(i):
            self.k.store(out, i, self.k.load(src, src_index(i)))
        self.k.loop_over(out.size, body)
        return out

    # -- primitives -----------------------------------------------------------------
    def p_select_n(self, e, ins, outs, params, dying):
        pred, cases = ins[0], ins[1:]
        def fn(p, *cs):
            if len(cs) == 2:return f'(({p}) ? ({cs[1]}) : ({cs[0]}))'
            expr = cs[-1]
            for k in range(len(cs) - 2, -1, -1):expr = f'((int)({p}) == {k} ? ({cs[k]}) : {expr})'
            return expr
        return [self.elementwise(outs[0], [pred, *cases], fn, dying)]

    def p_convert_element_type(self, e, ins, outs, params, dying):
        src = ins[0];out = outs[0]
        if src.kind == 'arena' and src.dtype == out.dtype:return [self.view(src, out)]
        return [self.elementwise(out, ins, lambda a: convert(src.dtype, out.dtype, a), dying)]

    def p_clamp(self, e, ins, outs, params, dying):
        lo, x, hi = ins;t = x.dtype
        if t == F32:fn = lambda a, v, c: f'v2e::fmin_x(v2e::fmax_x({v}, {a}), {c})'
        else:fn = lambda a, v, c: f'((({v}) < ({a}) ? ({a}) : ({v})) > ({c}) ? ({c}) : (({v}) < ({a}) ? ({a}) : ({v})))'
        return [self.elementwise(outs[0], [lo, x, hi], fn, dying)]

    def p_integer_pow(self, e, ins, outs, params, dying):
        y = int(params['y']);t = ins[0].dtype
        def fn(a):
            if y == 2:return binary('mul', t, a, a)
            raise Unsupported('integer_pow')
        return [self.elementwise(outs[0], ins, fn, dying)]

    def view(self, src, out):
        if src.kind == 'arena':
            out.kind = 'arena';out.slot = src.slot
            return out
        if src.kind in ('lit',):
            out.kind = 'lit';out.lit = src.lit;return out
        if src.kind == 'lane_id':
            out.kind = 'lane_id';return out
        if src.kind in ('in', 'const'):
            if src.lane != out.lane:raise Unsupported('view across lane axis')
            out.kind = src.kind;out.ptr = src.ptr;return out
        raise Unsupported(src.kind)

    def p_reshape(self, e, ins, outs, params, dying):
        src, out = ins[0], outs[0]
        if params.get('dimensions') not in (None, ()):raise Unsupported('reshape with dimensions')
        if src.lane != out.lane:raise Unsupported(f'reshape across lane axis {src.shape} -> {out.shape}')
        if src.kind == 'lane_id' and out.size != src.size:raise Unsupported('lane id reshape')
        if src.kind == 'lane_id':return [self.view(src, out)]
        if src.kind == 'lit':return [self.view(src, out)]
        return [self.view(src, out)]

    def p_squeeze(self, e, ins, outs, params, dying):
        src, out = ins[0], outs[0]
        if src.lane and 0 in tuple(params.get('dimensions', ())):raise Unsupported('squeeze of the lane axis')
        return [self.view(src, out)]

    p_expand_dims = p_squeeze

    def p_copy(self, e, ins, outs, params, dying):
        return [self.view(ins[0], outs[0])]

    p_copy_p = p_copy

    def p_optimization_barrier(self, e, ins, outs, params, dying):
        return [self.view(v, o) for v, o in zip(ins, outs)]

    def p_iota(self, e, ins, outs, params, dying):
        out = outs[0];dim = int(params['dimension'])
        if out.lane and dim == 0:
            out.kind = 'lane_id';return [out]
        shape = out.lshape;d = dim - 1 if out.lane else dim
        self.result(out)
        self.k.loop_over(out.size, lambda i: self.k.store(out, i, f'(({ctype(out.dtype)}){self.k.coords(i, shape)[d]})'))
        return [out]

    def p_broadcast_in_dim(self, e, ins, outs, params, dying):
        src, out = ins[0], outs[0];bd = tuple(params['broadcast_dimensions'])
        if src.kind == 'lit':
            out.kind = 'lit';out.lit = src.lit;return [out]
        if src.kind == 'lane_id':
            if not (out.lane and 0 in bd and bd.index(0) == 0):raise Unsupported('lane id broadcast')
            out.kind = 'lane_id';return [out]
        if src.lane and (not bd or bd[0] != 0):raise Unsupported('lane axis moved by broadcast')
        oshape = out.shape;ishape = src.shape
        def src_index(i):
            full = (['b'] if out.lane else []) + self.k.coords(i, out.lshape)
            parts = []
            ist = strides(src.lshape)
            for j, od in enumerate(bd):
                if src.lane and j == 0:continue
                jj = j - 1 if src.lane else j
                c = '0' if ishape[j] == 1 and oshape[od] != 1 else full[od]
                if ist[jj] != 0 and ishape[j] != 1:parts.append(f'({c}) * {ist[jj]}')
            return ' + '.join(parts) if parts else '0'
        return [self.copy_to(out, src_index, src)]

    def p_transpose(self, e, ins, outs, params, dying):
        src, out = ins[0], outs[0];perm = tuple(params['permutation'])
        if src.lane and perm[0] != 0:raise Unsupported('transpose moves lane axis')
        ist = strides(src.lshape)
        def src_index(i):
            full = (['b'] if out.lane else []) + self.k.coords(i, out.lshape)
            parts = []
            for od, idim in enumerate(perm):
                if src.lane and idim == 0:continue
                parts.append(f'({full[od]}) * {ist[idim - 1 if src.lane else idim]}')
            return ' + '.join(parts) if parts else '0'
        return [self.copy_to(out, src_index, src)]

    def p_slice(self, e, ins, outs, params, dying):
        src, out = ins[0], outs[0]
        start = tuple(params['start_indices']);stride = tuple(params['strides'] or (1,) * len(start))
        if src.lane and (start[0] != 0 or out.shape[0] != LANES):raise Unsupported('slice across lanes')
        ist = strides(src.lshape)
        def src_index(i):
            full = (['b'] if out.lane else []) + self.k.coords(i, out.lshape)
            parts = []
            for d in range(len(start)):
                if src.lane and d == 0:continue
                dd = d - 1 if src.lane else d
                parts.append(f'({start[d]} + ({full[d]}) * {stride[d]}) * {ist[dd]}')
            return ' + '.join(parts) if parts else '0'
        return [self.copy_to(out, src_index, src)]

    def p_rev(self, e, ins, outs, params, dying):
        src, out = ins[0], outs[0];dims = set(params['dimensions'])
        if src.lane and 0 in dims:raise Unsupported('rev over lanes')
        ist = strides(src.lshape);shape = src.shape
        def src_index(i):
            full = (['b'] if out.lane else []) + self.k.coords(i, out.lshape)
            parts = []
            for d in range(len(shape)):
                if src.lane and d == 0:continue
                c = f'({shape[d] - 1} - ({full[d]}))' if d in dims else full[d]
                parts.append(f'({c}) * {ist[d - 1 if src.lane else d]}')
            return ' + '.join(parts) if parts else '0'
        return [self.copy_to(out, src_index, src)]

    def p_concatenate(self, e, ins, outs, params, dying):
        out = outs[0];dim = int(params['dimension'])
        if out.lane and dim == 0:raise Unsupported('concatenate over lanes')
        self.result(out);ost = strides(out.lshape);d = dim - 1 if out.lane else dim;offset = 0
        for src in ins:
            n = src.shape[dim]
            if src.lane != out.lane and src.kind not in ('lit',):
                if not (out.lane and src.kind == 'lane_id'):raise Unsupported('concatenate mixed lanes')
            def body(i, src=src, offset=offset):
                c = self.k.coords(i, src.lshape)
                parts = [f'({c[x]} + {offset}) * {ost[x]}' if x == d else f'({c[x]}) * {ost[x]}' for x in range(len(c))]
                self.k.store(out, ' + '.join(parts) if parts else '0', self.k.load(src, i))
            self.k.loop_over(src.size, body);offset += n
        return [out]

    def p_stack(self, e, ins, outs, params, dying):
        out = outs[0];axis = int(params['axis'])
        if out.lane and axis == 0:raise Unsupported('stack over lanes')
        self.result(out);ost = strides(out.lshape);d = axis - 1 if out.lane else axis
        for k, src in enumerate(ins):
            def body(i, src=src, k=k):
                c = self.k.coords(i, src.lshape);c = c[:d] + [str(k)] + c[d:]
                self.k.store(out, ' + '.join(f'({x}) * {s}' for x, s in zip(c, ost)) or '0', self.k.load(src, i))
            self.k.loop_over(src.size, body)
        return [out]

    def p_unstack(self, e, ins, outs, params, dying):
        src = ins[0];axis = int(params['axis'])
        if src.lane and axis == 0:raise Unsupported('unstack over lanes')
        ist = strides(src.lshape);d = axis - 1 if src.lane else axis
        for k, out in enumerate(outs):
            def src_index(i, k=k, out=out):
                c = self.k.coords(i, out.lshape);c = c[:d] + [str(k)] + c[d:]
                return ' + '.join(f'({x}) * {s}' for x, s in zip(c, ist)) or '0'
            self.copy_to(out, src_index, src)
        return outs

    def p_pad(self, e, ins, outs, params, dying):
        src, padv = ins;out = outs[0];config = params['padding_config']
        if src.lane and tuple(config[0]) != (0, 0, 0):raise Unsupported('pad over lanes')
        self.result(out);ist = strides(src.lshape)
        def body(i):
            full = (['b'] if out.lane else []) + self.k.coords(i, out.lshape)
            conds = [];parts = []
            for dd, (lo, hi, interior) in enumerate(config):
                if src.lane and dd == 0:continue
                rel = f'(({full[dd]}) - ({lo}))'
                step = interior + 1
                conds.append(f'({rel} >= 0 && {rel} % {step} == 0 && {rel} / {step} < {src.shape[dd]})')
                parts.append(f'({rel} / {step}) * {ist[dd - 1 if src.lane else dd]}')
            inside = ' && '.join(conds) or 'true'
            idx = ' + '.join(parts) or '0'
            self.k.store(out, i, f'(({inside}) ? ({self.k.load(src, f"({inside}) ? ({idx}) : 0")}) : ({self.k.load(padv, "0")}))')
        self.k.loop_over(out.size, body)
        return [out]

    def reduce(self, e, ins, outs, params, dying, init, combine):
        src, out = ins[0], outs[0];axes = tuple(params['axes'])
        if src.lane and 0 in axes:raise Unsupported('reduce over lanes')
        self.result(out)
        keep = [d for d in range(len(src.shape)) if d not in axes and not (src.lane and d == 0)]
        red = [d for d in axes];ist = strides(src.lshape);off = 1 if src.lane else 0
        rshape = [src.shape[d] for d in red];rn = int(math.prod(rshape))
        c = ctype(out.dtype)
        def body(i):
            oc = self.k.coords(i, out.lshape)
            acc = self.k.var('acc')
            self.k.emit(f'{c} {acc} = {init};')
            base = ' + '.join(f'({oc[j]}) * {ist[d - off]}' for j, d in enumerate(keep)) or '0'
            def inner(r):
                rc = self.k.coords(r, rshape)
                idx = ' + '.join([base] + [f'({rc[j]}) * {ist[d - off]}' for j, d in enumerate(red)])
                self.k.emit(f'{acc} = {combine(acc, self.k.load(src, idx))};')
            self.k.serial_loop(rn, inner)
            self.k.store(out, i, acc)
        self.k.loop_over(out.size, body)
        return [out]

    def p_reduce_sum(self, e, ins, outs, params, dying):
        t = ins[0].dtype
        if t == F32:return self.reduce(e, ins, outs, params, dying, literal(0.0, F32), lambda a, x: f'v2e::fadd({a}, {x})')
        return self.reduce(e, ins, outs, params, dying, literal(0, t), lambda a, x: binary('add', t, a, x))

    def p_reduce_max(self, e, ins, outs, params, dying):
        t = ins[0].dtype
        init = literal(-np.inf, F32) if t == F32 else ('false' if t == BOOL else literal(np.iinfo(t).min, t))
        return self.reduce(e, ins, outs, params, dying, init, lambda a, x: binary('max', t, a, x))

    def p_reduce_min(self, e, ins, outs, params, dying):
        t = ins[0].dtype
        init = literal(np.inf, F32) if t == F32 else ('true' if t == BOOL else literal(np.iinfo(t).max, t))
        return self.reduce(e, ins, outs, params, dying, init, lambda a, x: binary('min', t, a, x))

    def p_reduce_or(self, e, ins, outs, params, dying):
        t = ins[0].dtype
        return self.reduce(e, ins, outs, params, dying, 'false' if t == BOOL else literal(0, t), lambda a, x: binary('or', t, a, x))

    def p_reduce_and(self, e, ins, outs, params, dying):
        t = ins[0].dtype
        return self.reduce(e, ins, outs, params, dying, 'true' if t == BOOL else literal(-1 if t in SIGNED else np.iinfo(t).max, t),
            lambda a, x: binary('and', t, a, x))

    def p_reduce_xor(self, e, ins, outs, params, dying):
        t = ins[0].dtype
        return self.reduce(e, ins, outs, params, dying, 'false' if t == BOOL else literal(0, t), lambda a, x: binary('xor', t, a, x))

    def p_argmax(self, e, ins, outs, params, dying):
        return self.arg(e, ins, outs, params, dying, 'gt')

    def p_argmin(self, e, ins, outs, params, dying):
        return self.arg(e, ins, outs, params, dying, 'lt')

    def arg(self, e, ins, outs, params, dying, cmp):
        src, out = ins[0], outs[0];axes = tuple(params['axes'])
        if len(axes) != 1 or (src.lane and axes[0] == 0):raise Unsupported('argmax axes')
        ax = axes[0];off = 1 if src.lane else 0;ist = strides(src.lshape);n = src.shape[ax]
        keep = [d for d in range(len(src.shape)) if d != ax and not (src.lane and d == 0)]
        t = src.dtype;self.result(out)
        def body(i):
            oc = self.k.coords(i, out.lshape)
            base = ' + '.join(f'({oc[j]}) * {ist[d - off]}' for j, d in enumerate(keep)) or '0'
            best, at, j = self.k.var('best'), self.k.var('at'), self.k.var('j')
            self.k.emit(f'{ctype(t)} {best} = {self.k.load(src, base)}; int {at} = 0;')
            self.k.emit(f'for (int {j} = 1; {j} < {n}; ++{j}) {{');self.k.indent += 1
            x = self.k.var('x');self.k.emit(f'{ctype(t)} {x} = {self.k.load(src, f"{base} + {j} * {ist[ax - off]}")};')
            if t == F32:
                # XLA argmax: NaN wins (first NaN), else strictly greater.
                better = f'(!v2e::fnan({best}) && (v2e::fnan({x}) || {x} {">" if cmp == "gt" else "<"} {best}))'
            else:better = f'({x} {">" if cmp == "gt" else "<"} {best})'
            self.k.emit(f'if {better} {{ {best} = {x}; {at} = {j}; }}')
            self.k.indent -= 1;self.k.emit('}')
            self.k.store(out, i, f'(({ctype(out.dtype)}){at})')
        self.k.loop_over(out.size, body)
        return [out]

    def p_cumsum(self, e, ins, outs, params, dying):
        src, out = ins[0], outs[0];ax = int(params['axis']);rev = bool(params.get('reverse', False))
        if src.lane and ax == 0:raise Unsupported('cumsum over lanes')
        off = 1 if src.lane else 0;ist = strides(src.lshape);n = src.shape[ax];t = src.dtype
        keep_shape = [src.shape[d] for d in range(len(src.shape)) if d != ax and not (src.lane and d == 0)]
        keep = [d for d in range(len(src.shape)) if d != ax and not (src.lane and d == 0)]
        self.result(out)
        def body(i):
            oc = self.k.coords(i, keep_shape)
            base = ' + '.join(f'({oc[j]}) * {ist[d - off]}' for j, d in enumerate(keep)) or '0'
            acc, j = self.k.var('acc'), self.k.var('j')
            self.k.emit(f'{ctype(t)} {acc} = {literal(0, t)};')
            loop = f'for (int {j} = {n - 1}; {j} >= 0; --{j})' if rev else f'for (int {j} = 0; {j} < {n}; ++{j})'
            self.k.emit(loop + ' {');self.k.indent += 1
            idx = f'{base} + {j} * {ist[ax - off]}'
            self.k.emit(f'{acc} = {binary("add", t, acc, self.k.load(src, idx))};')
            self.k.store(out, idx, acc)
            self.k.indent -= 1;self.k.emit('}')
        self.k.loop_over(int(math.prod(keep_shape)), body)
        return [out]

    # -- gather / scatter -----------------------------------------------------------
    def p_gather(self, e, ins, outs, params, dying):
        op, idx = ins;out = outs[0];dn = params['dimension_numbers'];slice_sizes = tuple(params['slice_sizes'])
        mode = str(params['mode']);fill = params.get('fill_value')
        offset_dims = tuple(dn.offset_dims);collapsed = tuple(dn.collapsed_slice_dims);start_map = tuple(dn.start_index_map)
        obatch = tuple(getattr(dn, 'operand_batching_dims', ()));ibatch = tuple(getattr(dn, 'start_indices_batching_dims', ()))
        oshape = op.shape;ishape = idx.shape;outshape = out.shape
        ivd = len(ishape) - 1;nvec = ishape[ivd]
        batch_out = [d for d in range(len(outshape)) if d not in offset_dims]
        idx_batch_dims = [d for d in range(len(ishape)) if d != ivd]
        window = [d for d in range(len(oshape)) if d not in collapsed and d not in obatch]
        ist = strides(idx.lshape);ost = strides(op.lshape)
        fill_mode = 'FILL' in mode.upper()
        self.result(out)
        def body(i):
            full = (['b'] if out.lane else []) + self.k.coords(i, out.lshape)
            G = [full[d] for d in batch_out]   # coordinates over idx batch dims, in order
            def idx_at(k):
                c = list(G);c.insert(ivd, str(k))
                parts = [f'({c[d]}) * {ist[d - (1 if idx.lane else 0)]}' for d in range(len(ishape)) if not (idx.lane and d == 0)]
                if idx.lane and c[0] != 'b':self.k.emit(f'if (({c[0]}) != b) atomicOr(err, 1u);')
                return ' + '.join(parts) or '0'
            starts = ['0'] * len(oshape)
            for k, d in enumerate(start_map):
                s = self.k.var('s');self.k.emit(f'int64_t {s} = (int64_t){self.k.load(idx, idx_at(k))};');starts[d] = s
            for j, d in enumerate(obatch):
                starts[d] = G[idx_batch_dims.index(ibatch[j])]
            oob = []
            for d in start_map:
                if op.lane and d == 0:continue  # the lane coordinate is checked against b, never clamped
                hi = oshape[d] - slice_sizes[d]
                if fill_mode:oob.append(f'({starts[d]} < 0 || {starts[d]} > {hi})')
                else:
                    c = self.k.var('c');self.k.emit(f'int64_t {c} = {starts[d]} < 0 ? 0 : ({starts[d]} > {hi} ? {hi} : {starts[d]});');starts[d] = c
            wcoords = {d: full[offset_dims[j]] for j, d in enumerate(window)}
            coord = [f'({starts[d]} + {wcoords[d]})' if d in wcoords else f'({starts[d]})' for d in range(len(oshape))]
            if op.lane:
                self.k.emit(f'if ({coord[0]} != b) atomicOr(err, 2u);')
                flat = ' + '.join(f'({coord[d]}) * {ost[d - 1]}' for d in range(1, len(oshape))) or '0'
            else:flat = ' + '.join(f'({coord[d]}) * {ost[d]}' for d in range(len(oshape))) or '0'
            value = self.k.load(op, f'({" || ".join(oob)}) ? 0 : ({flat})' if oob else flat)
            if oob:
                fv = literal(fill if fill is not None else (np.nan if op.dtype == F32 else (True if op.dtype == BOOL else
                    (np.iinfo(op.dtype).min if op.dtype in SIGNED else np.iinfo(op.dtype).max))), op.dtype)
                value = f'(({" || ".join(oob)}) ? {fv} : {value})'
            self.k.store(out, i, value)
        self.k.loop_over(out.size, body)
        return [out]

    def scatter(self, e, ins, outs, params, dying, combine, atomic=None):
        op, idx, upd = ins;out = outs[0];dn = params['dimension_numbers'];mode = str(params['mode'])
        # Order-independent updates may run in parallel: integer/bool combiners via
        # atomics, or replacement with statically distinct target indices.
        unique = combine is None and idx.kind == 'const' and False
        if combine is None and idx.kind in ('const', 'lit'):
            unique = True if idx.kind == 'lit' and upd.size == 1 else (idx.kind == 'const' and self.distinct_const(idx))
        parallel = (atomic is not None and op.dtype != F32) or unique
        uwd = tuple(dn.update_window_dims);inserted = tuple(dn.inserted_window_dims);sd2od = tuple(dn.scatter_dims_to_operand_dims)
        obatch = tuple(getattr(dn, 'operand_batching_dims', ()));ibatch = tuple(getattr(dn, 'scatter_indices_batching_dims', ()))
        oshape = op.shape;ishape = idx.shape;ushape = upd.shape;ivd = len(ishape) - 1
        if params.get('update_jaxpr') is not None and combine is None:raise Unsupported('scatter combiner')
        self.result(out, [op], dying)
        if not out.inplace:
            self.k.loop_over(out.size, lambda i: self.k.store(out, i, self.k.load(op, i)))
        ist = strides(idx.lshape);ost = strides(op.lshape)
        scatter_dims = [d for d in range(len(ushape)) if d not in uwd]
        idx_batch_dims = [d for d in range(len(ishape)) if d != ivd]
        window = [d for d in range(len(oshape)) if d not in inserted and d not in obatch]
        clip = 'CLIP' in mode.upper()
        def body(i):
            full = (['b'] if upd.lane else []) + self.k.coords(i, upd.lshape)
            G = [full[d] for d in scatter_dims]
            def idx_at(k):
                c = list(G);c.insert(ivd, str(k))
                if idx.lane and c[0] != 'b':self.k.emit(f'if (({c[0]}) != b) atomicOr(err, 4u);')
                return ' + '.join(f'({c[d]}) * {ist[d - (1 if idx.lane else 0)]}' for d in range(len(ishape)) if not (idx.lane and d == 0)) or '0'
            starts = ['0'] * len(oshape)
            for k, d in enumerate(sd2od):
                s = self.k.var('s');self.k.emit(f'int64_t {s} = (int64_t){self.k.load(idx, idx_at(k))};');starts[d] = s
            for j, d in enumerate(obatch):starts[d] = G[idx_batch_dims.index(ibatch[j])]
            wsize = {d: ushape[uwd[j]] for j, d in enumerate(window)}
            conds = []
            for d in range(len(oshape)):
                size = wsize.get(d, 1);hi = oshape[d] - size
                if op.lane and d == 0:continue  # lane coordinate: compared with b below
                if d in sd2od:
                    if clip:
                        c = self.k.var('c');self.k.emit(f'int64_t {c} = {starts[d]} < 0 ? 0 : ({starts[d]} > {hi} ? {hi} : {starts[d]});');starts[d] = c
                    else:conds.append(f'({starts[d]} >= 0 && {starts[d]} <= {hi})')
            wc = {d: full[uwd[j]] for j, d in enumerate(window)}
            coord = [f'({starts[d]} + {wc[d]})' if d in wc else f'({starts[d]})' for d in range(len(oshape))]
            if op.lane:
                conds.append(f'({coord[0]} == b)')
                flat = ' + '.join(f'({coord[d]}) * {ost[d - 1]}' for d in range(1, len(oshape))) or '0'
            else:flat = ' + '.join(f'({coord[d]}) * {ost[d]}' for d in range(len(oshape))) or '0'
            guard = ' && '.join(conds) or 'true'
            f = self.k.var('f');self.k.emit(f'const int64_t {f} = {flat};')
            self.k.emit(f'if ({guard}) {{');self.k.indent += 1
            u = self.k.load(upd, i)
            if parallel and atomic is not None:
                self.k.emit(atomic(f'A + ({out.off}) + ({f})', u, out.dtype))
            else:
                value = u if combine is None else combine(self.k.load(out, f), u)
                self.k.store(out, f, value)
            self.k.indent -= 1;self.k.emit('}')
        self.k.emit('__syncwarp();')
        if parallel:
            self.k.loop_over(upd.size, body)
        else:
            self.k.emit('if (t == 0) {');self.k.indent += 1
            self.k.serial_loop(upd.size, body)
            self.k.indent -= 1;self.k.emit('}')
        return [out]

    def p_scatter(self, e, ins, outs, params, dying):
        return self.scatter(e, ins, outs, dict(params, update_jaxpr=None), dying, None)

    def p_scatter_add(self, e, ins, outs, params, dying):
        t = ins[0].dtype
        return self.scatter(e, ins, outs, params, dying, lambda a, u: binary('add', t, a, u),
            atomic=None if t == F32 or t == BOOL else (lambda ptr, u, dt: f'atomicAdd({ptr}, (uint32_t)({u}));'))

    def p_scatter_min(self, e, ins, outs, params, dying):
        t = ins[0].dtype
        return self.scatter(e, ins, outs, params, dying, lambda a, u: binary('min', t, a, u), atomic=self.atomic_minmax('Min', t))

    def p_scatter_max(self, e, ins, outs, params, dying):
        t = ins[0].dtype
        return self.scatter(e, ins, outs, params, dying, lambda a, u: binary('max', t, a, u), atomic=self.atomic_minmax('Max', t))

    @staticmethod
    def atomic_minmax(which, t):
        t = np.dtype(t)
        if t == np.dtype(np.int32):return lambda ptr, u, dt: f'atomic{which}(reinterpret_cast<int*>({ptr}), (int)({u}));'
        if t == np.dtype(np.uint32):return lambda ptr, u, dt: f'atomic{which}(reinterpret_cast<unsigned int*>({ptr}), (unsigned int)({u}));'
        if t == BOOL:return lambda ptr, u, dt: (f'atomicAnd({ptr}, (uint32_t)({u}));' if which == 'Min' else f'atomicOr({ptr}, (uint32_t)({u}));')
        return None

    def distinct_const(self, idx):
        arr = self.k.consts[int(idx.ptr[2:])]
        flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr.reshape(-1, 1)
        return len({tuple(r) for r in flat.tolist()}) == len(flat)

    def p_scatter_mul(self, e, ins, outs, params, dying):
        t = ins[0].dtype
        return self.scatter(e, ins, outs, params, dying, lambda a, u: binary('mul', t, a, u))

    def p_dynamic_slice(self, e, ins, outs, params, dying):
        src, starts = ins[0], ins[1:];out = outs[0];sizes = tuple(params['slice_sizes'])
        if src.lane and sizes[0] != LANES:raise Unsupported('dynamic_slice across lanes')
        st = []
        for d, s in enumerate(starts):
            if src.lane and d == 0:st.append('0');continue
            v = self.k.var('ds');hi = src.shape[d] - sizes[d]
            self.k.emit(f'int64_t {v} = (int64_t){self.k.load(s, "0")}; {v} = {v} < 0 ? 0 : ({v} > {hi} ? {hi} : {v});');st.append(v)
        ist = strides(src.lshape)
        def src_index(i):
            full = (['b'] if out.lane else []) + self.k.coords(i, out.lshape)
            return ' + '.join(f'({st[d]} + {full[d]}) * {ist[d - (1 if src.lane else 0)]}' for d in range(len(src.shape)) if not (src.lane and d == 0)) or '0'
        return [self.copy_to(out, src_index, src)]

    def p_dynamic_update_slice(self, e, ins, outs, params, dying):
        src, upd, starts = ins[0], ins[1], ins[2:];out = outs[0]
        self.result(out, [src], dying)
        if not out.inplace:
            self.k.loop_over(out.size, lambda i: self.k.store(out, i, self.k.load(src, i)))
            self.k.emit('__syncwarp();')
        st = []
        for d, s in enumerate(starts):
            if src.lane and d == 0:st.append('0');continue
            v = self.k.var('ds');hi = src.shape[d] - upd.shape[d]
            self.k.emit(f'int64_t {v} = (int64_t){self.k.load(s, "0")}; {v} = {v} < 0 ? 0 : ({v} > {hi} ? {hi} : {v});');st.append(v)
        ost = strides(out.lshape)
        def body(i):
            full = (['b'] if upd.lane else []) + self.k.coords(i, upd.lshape)
            flat = ' + '.join(f'({st[d]} + {full[d]}) * {ost[d - (1 if out.lane else 0)]}' for d in range(len(out.shape)) if not (out.lane and d == 0)) or '0'
            self.k.store(out, flat, self.k.load(upd, i))
        self.k.loop_over(upd.size, body)
        return [out]

    def p_sort(self, e, ins, outs, params, dying):
        """Stable sort as a warp-cooperative bitonic network over (keys..., original index).

        (keys, index) is a total order with distinct entries, so any correct
        sorting network yields exactly the stable order. Rows are sorted one at
        a time by the whole warp in shared memory; padding sorts last.
        """
        dim = int(params['dimension']);nk = int(params['num_keys'])
        src = ins[0]
        if src.lane and dim == 0:raise Unsupported('sort over lanes')
        for v in ins[:nk]:
            if v.dtype == F32:raise Unsupported('float sort keys')
        off = 1 if src.lane else 0;ist = strides(src.lshape);n = src.shape[dim]
        keep = [d for d in range(len(src.shape)) if d != dim and not (src.lane and d == 0)]
        keep_shape = [src.shape[d] for d in keep];rows = int(math.prod(keep_shape))
        npad = 1 << max(0, (n - 1).bit_length())
        for o in outs:self.result(o)
        need = (nk + 1) * npad;self.k.smem = max(self.k.smem, need)
        k = self.k;step = ist[dim - off]
        key_t = ['uint32_t' if v.dtype in UNSIGNED else ('int32_t' if v.dtype != BOOL else 'int32_t') for v in ins[:nk]]
        def key_load(q, slot):
            return f'(({key_t[q]})S[{q * npad} + ({slot})])'
        row = k.var('row')
        k.emit(f'for (int {row} = 0; {row} < {rows}; ++{row}) {{');k.indent += 1
        rc = k.coords(row, keep_shape)
        base = ' + '.join(f'({rc[j]}) * {ist[d - off]}' for j, d in enumerate(keep)) or '0'
        q = k.var('q')
        k.emit(f'for (int {q} = t; {q} < {npad}; {q} += 32) {{');k.indent += 1
        k.emit(f'const bool real = {q} < {n};')
        for kk in range(nk):
            fill = '0xFFFFFFFFu' if key_t[kk] == 'uint32_t' else '0x7FFFFFFFu'
            k.emit(f'S[{kk * npad} + {q}] = real ? (uint32_t)({ins[kk] and self.k.load(ins[kk], f"{base} + ({q}) * {step}")}) : {fill};')
        k.emit(f'S[{nk * npad} + {q}] = (uint32_t){q};')
        k.indent -= 1;k.emit('}');k.emit('__syncwarp();')
        kv, jv, iv = k.var('k'), k.var('j'), k.var('i')
        k.emit(f'for (int {kv} = 2; {kv} <= {npad}; {kv} <<= 1) {{');k.indent += 1
        k.emit(f'for (int {jv} = {kv} >> 1; {jv} > 0; {jv} >>= 1) {{');k.indent += 1
        k.emit(f'for (int {iv} = t; {iv} < {npad}; {iv} += 32) {{');k.indent += 1
        k.emit(f'const int other = {iv} ^ {jv};')
        k.emit(f'if (other > {iv}) {{');k.indent += 1
        greater = f'(S[{nk * npad} + {iv}] > S[{nk * npad} + other])'
        for kk in range(nk - 1, -1, -1):
            a, c = key_load(kk, iv), key_load(kk, 'other')
            greater = f'({a} > {c} || ({a} == {c} && {greater}))'
        k.emit(f'const bool up = ({iv} & {kv}) == 0;')
        k.emit(f'if (up == {greater}) {{');k.indent += 1
        for kk in range(nk + 1):
            k.emit(f'{{ const uint32_t sw = S[{kk * npad} + {iv}]; S[{kk * npad} + {iv}] = S[{kk * npad} + other]; S[{kk * npad} + other] = sw; }}')
        k.indent -= 1;k.emit('}')
        k.indent -= 1;k.emit('}')
        k.indent -= 1;k.emit('}')
        k.emit('__syncwarp();')
        k.indent -= 1;k.emit('}')
        k.indent -= 1;k.emit('}')
        z = k.var('z')
        k.emit(f'for (int {z} = t; {z} < {n}; {z} += 32) {{');k.indent += 1
        k.emit(f'const int from = (int)S[{nk * npad} + {z}];')
        for v, o in zip(ins, outs):
            k.store(o, f'{base} + ({z}) * {step}', k.load(v, f'{base} + from * {step}'))
        k.indent -= 1;k.emit('}');k.emit('__syncwarp();')
        k.indent -= 1;k.emit('}')
        return outs

    # -- calls ----------------------------------------------------------------------
    def call(self, closed, ins, outs):
        jaxpr = closed.jaxpr
        consts = [self.const(c) for c in closed.consts]
        res = self.run_with_consts(jaxpr, consts, ins)
        for r in res:
            if r.kind == 'arena':r.slot.refs -= 1  # hand ownership back to the enclosing run()
        return res

    def run_with_consts(self, jaxpr, consts, ins):
        from jax.extend import core as jcore
        if jaxpr.constvars:
            inner = jcore.Jaxpr(constvars=(), invars=list(jaxpr.constvars) + list(jaxpr.invars), outvars=jaxpr.outvars,
                eqns=jaxpr.eqns, effects=jaxpr.effects)
            return self.run(inner, list(consts) + list(ins))
        return self.run(jaxpr, ins)

    def p_jit(self, e, ins, outs, params, dying):
        return self.call(params['jaxpr'], ins, outs)

    p_pjit = p_jit

    def p_closed_call(self, e, ins, outs, params, dying):
        return self.call(params['call_jaxpr'], ins, outs)

    def p_custom_jvp_call(self, e, ins, outs, params, dying):
        return self.call(params['call_jaxpr'], ins, outs)

    def p_custom_vjp_call(self, e, ins, outs, params, dying):
        return self.call(params['call_jaxpr'], ins, outs)

    # -- constants --------------------------------------------------------------------
    def const(self, value):
        arr = np.asarray(value)
        if arr.ndim == 0:return Val(np.dtype(arr.dtype), (), 'lit', lit=arr.item())
        if LANES in arr.shape:raise Unsupported(f'constant with lane-sized axis {arr.shape}')
        for k, c in enumerate(self.k.consts):
            if c is value or (c.shape == arr.shape and c.dtype == arr.dtype and np.array_equal(c, arr)):
                return Val(np.dtype(arr.dtype), arr.shape, 'const', ptr=f'cs{k}')
        self.k.consts.append(arr)
        return Val(np.dtype(arr.dtype), arr.shape, 'const', ptr=f'cs{len(self.k.consts) - 1}')


def generate(name, closed, input_avals, profile=False):
    """CUDA source for closed jaxpr over lane inputs (flattened avals with axis 0 = LANES)."""
    g = Generator(name, profile);k = g.k
    ins = []
    for n, aval in enumerate(input_avals):
        if not (len(aval.shape) and aval.shape[0] == LANES):raise Unsupported(f'input {n} lacks the lane axis')
        ins.append(Val(np.dtype(aval.dtype), tuple(aval.shape), 'in', ptr=f'in{n}'))
    outs = g.run_with_consts(closed.jaxpr, [g.const(c) for c in closed.consts], ins)
    # Copy results to output buffers (lane-major, native dtypes), in the last chunk.
    for n, v in enumerate(outs):
        size = v.size
        if not v.lane:raise Unsupported(f'output {n} has no lane axis')
        k.emit(f'{{ {ctype(v.dtype)}* o = reinterpret_cast<{ctype(v.dtype)}*>(p.out[{n}]);')
        k.loop_over(size, lambda i: k.emit(f'o[(size_t)b * {size} + ({i})] = {k.load(v, i)};'))
        k.emit('}')
    chunks = k.chunks + ([k.lines] if k.lines else [])
    smems = k.chunk_smem + ([k.smem] if k.lines else [])
    decl = []
    for n, v in enumerate(ins):decl.append(f'  const {ctype(v.dtype)}* __restrict__ in{n} = reinterpret_cast<const {ctype(v.dtype)}*>(p.in[{n}]);')
    for n, c in enumerate(k.consts):decl.append(f'  const {ctype(c.dtype)}* __restrict__ cs{n} = reinterpret_cast<const {ctype(c.dtype)}*>(p.in[{len(ins) + n}]);')
    words = max(1, k.peak);ns = f'v2e_gen_{name}'
    header = f'''// Generated by gpu_engine.v2e_j2c from the oracle jaxpr of `{name}`. Do not edit.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "v2e_jax.cuh"

namespace {ns} {{

constexpr int kInputs = {len(ins) + len(k.consts)};
constexpr int kOutputs = {len(outs)};
constexpr int kChunks = {len(chunks)};
constexpr int64_t kArenaWords = {words};

struct Params {{
  const void* in[kInputs > 0 ? kInputs : 1];
  void* out[kOutputs > 0 ? kOutputs : 1];
  uint32_t* arena;
  uint32_t* err;
  int64_t lanes;
}};

{chr(10).join(f'void launch_{c}(const Params& p, cudaStream_t stream);' for c in range(len(chunks)))}

}}  // namespace {ns}
'''
    files = {f'{ns}.cuh': header}
    for c, lines in enumerate(chunks):
        smem_decl = f'__shared__ uint32_t S[{smems[c]}];' if smems[c] else ''
        files[f'{ns}_{c:03d}.cu'] = f'''// Generated chunk {c} of `{name}`. Do not edit.
#include "{ns}.cuh"

namespace {ns} {{

// One warp per lane: thread t handles elements t, t + 32, ... of every value.
__global__ void __launch_bounds__(32) chunk_{c}(const Params p) {{
  const int64_t b = blockIdx.x;
  const int t = threadIdx.x;
  if (b >= p.lanes) return;
  uint32_t* __restrict__ A = p.arena + (size_t)b * kArenaWords;
  uint32_t* err = p.err;
  {smem_decl}
  const long long T0 = clock64(); (void)T0;
  (void)A; (void)err; (void)t;
{chr(10).join(decl)}
{chr(10).join(lines)}
}}

void launch_{c}(const Params& p, cudaStream_t stream) {{
  chunk_{c}<<<static_cast<unsigned>(p.lanes), 32, 0, stream>>>(p);
}}

}}  // namespace {ns}
'''
    launches = chr(10).join(f'  launch_{c}(p, stream);' for c in range(len(chunks)))
    files[f'{ns}_ffi.cu'] = f'''// Generated FFI handler of `{name}`. Do not edit.
#include "{ns}.cuh"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {ns} {{

static ffi::Error Impl(cudaStream_t stream, ffi::RemainingArgs args, ffi::RemainingRets rets) {{
  if (args.size() != static_cast<size_t>(kInputs) || rets.size() != static_cast<size_t>(kOutputs + 2))
    return ffi::Error::InvalidArgument("{name}: operand count");
  Params p{{}};
  for (int n = 0; n < kInputs; ++n) {{
    auto a = args.get<ffi::AnyBuffer>(n);
    if (!a.has_value()) return a.error();
    p.in[n] = a->untyped_data();
  }}
  for (int n = 0; n < kOutputs; ++n) {{
    auto r = rets.get<ffi::AnyBuffer>(n);
    if (!r.has_value()) return r.error();
    p.out[n] = (*r)->untyped_data();
  }}
  auto arena = rets.get<ffi::AnyBuffer>(kOutputs);
  auto err = rets.get<ffi::AnyBuffer>(kOutputs + 1);
  if (!arena.has_value()) return arena.error();
  if (!err.has_value()) return err.error();
  p.arena = static_cast<uint32_t*>((*arena)->untyped_data());
  p.err = static_cast<uint32_t*>((*err)->untyped_data());
  p.lanes = static_cast<int64_t>((*arena)->element_count() / kArenaWords);
  cudaMemsetAsync(p.err, 0, sizeof(uint32_t), stream);
{launches}
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(status));
  return ffi::Error::Success();
}}

}}  // namespace {ns}

XLA_FFI_DEFINE_HANDLER_SYMBOL(V2eGen_{name}, {ns}::Impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().RemainingArgs().RemainingRets(),
                              {{ffi::Traits::kCmdBufferCompatible}});
'''
    module = Module(name, files, words, [(np.dtype(v.dtype), v.shape) for v in ins], list(k.consts),
        [(np.dtype(v.dtype), v.shape) for v in outs], k.lane_literals)
    module.marks = g.marks
    return module
