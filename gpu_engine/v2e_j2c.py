"""Generate per-lane CUDA kernels from oracle jaxprs (jaxpr -> CUDA C++).

The JAX engine is a straight-line, row-independent program over a leading
lane axis. A function is traced with lane size LANES (an unusual size, so the
lane axis is recognizable: every shape whose axis 0 has LANES elements is a
per-lane value; LANES may appear nowhere else). One warp executes one lane.

Values of at most 32 elements per lane are register-resident: thread t holds
element t (clamped), so elementwise work is register arithmetic and
index-mapped reads are warp shuffles, always executed in warp-uniform control
flow. Larger values live in a per-lane global arena (lane-major, slots reused
by liveness) and are processed 32 elements at a time. Broadcasts to large
shapes, iotas and the lane id stay lazy (index maps or expressions, no
storage). Every primitive uses the XLA-exact scalar helpers of
cuda/v2e_jax.cuh. Scatters keep exact sequential semantics (a target element
is updated in update order); stable sorts use a warp radix sort (a stable
sort's result is unique); order-free integer reductions and scans use warp
collectives. Live register values are spilled to the arena at kernel-chunk
boundaries.

Reads of a per-lane operand at another lane cannot happen in this engine (all
indexing uses the lane iota); gathers/scatters that would cross lanes set an
error word instead, which callers check.
"""
from dataclasses import dataclass, field
import hashlib
import math
import os

import numpy as np

LANES = 37
REG = 32

# The generator mirrors XLA-GPU behavior that was pinned by probes on GB300 with exactly these
# versions (algebraic simplifier and constant-folding order, A / C -> A * (1 / C), NaN-literal folds,
# reduction combiners). Another version may differ silently, so generation refuses to run on it.
# V2E_ALLOW_UNPINNED_JAX=1 exists only to re-run the GPU conformance battery when re-pinning.
PINNED = {'jax': '0.12.0', 'jaxlib': '0.12.0', 'jax-cuda13-plugin': '0.12.0', 'jax-cuda13-pjrt': '0.12.0'}
# A CPU-only host (CPU CI: no jax-cuda* distribution at all) needs only jax/jaxlib to match; once any
# jax-cuda* distribution is installed (a GPU host, whichever CUDA), the pinned plugins are required.
CUDA_PLUGINS = ('jax-cuda13-plugin', 'jax-cuda13-pjrt')


def check_pinned_versions():
    import importlib.metadata as metadata
    found = {}
    for name in PINNED:
        try:found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:found[name] = None
    gpu = any((d.metadata['Name'] or '').lower().replace('_', '-').startswith('jax-cuda') for d in metadata.distributions())
    differ = {name: (found[name], want) for name, want in PINNED.items() if found[name] != want and (gpu or name not in CUDA_PLUGINS)}
    if differ and os.environ.get('V2E_ALLOW_UNPINNED_JAX') != '1':
        raise Unsupported('jax/jaxlib versions differ from the pinned generator semantics (found, pinned): ' + repr(differ)
            + '; re-run gpu_engine.v2e_conformance on GPU and re-pin before generating')
    return found
WARPS = 4          # lanes (warps) per thread block
PARALLEL_MIN = 64  # reduction length from which order-free reductions use the whole warp

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
class Holder:
    """Storage shared by a value and its views."""
    kind: str            # 'reg' | 'arena'
    words: int
    dtype: np.dtype
    name: str = ''       # register variable (reg)
    off: int = -1        # arena word offset (arena)
    refs: int = 0        # pending reads
    home: object = None  # thread backend: home(q) -> expression loading word q of a small value
    deps: list = field(default_factory=list)  # holders kept alive while this one lives
    chunk: int = -1      # thread backend: kernel chunk in which `name` holds the value


@dataclass(eq=False)
class Val:
    dtype: np.dtype
    shape: tuple
    src: str                  # 'store' | 'in' | 'const' | 'lit' | 'expr'
    holder: object = None     # Holder for 'store'
    ptr: str = ''
    lit: object = None
    stride: int = 0           # per-lane words of an 'in' buffer
    imap: object = None       # flat index map onto the holder (lazy broadcast view of a store)
    fn: object = None         # element expression (src 'expr')
    arr: object = None        # numpy value of a 'const' Val (or of a lane-free iota)
    cderived: bool = False    # computed from constants only (XLA would fold it on the host)
    lbc: object = None        # per-lane value equal in every lane: lbc[...] (shape lshape), from constants
    bsig: object = None       # constant-derived values as XLA sees them: None (a constant), a broadcast
                              # (source shape, broadcast_dimensions, shape), or 'uncertain'

    @property
    def kind(self):
        return self.holder.kind if self.src == 'store' else self.src

    @property
    def lane(self):
        return len(self.shape) > 0 and self.shape[0] == LANES

    @property
    def lshape(self):
        return tuple(self.shape[1:]) if self.lane else tuple(self.shape)

    @property
    def size(self):
        return int(math.prod(self.lshape))

    @property
    def dense(self):
        """Stored with element i at holder word i."""
        return self.src == 'store' and self.imap is None


def aval_of(aval):
    """(numpy dtype, shape) of a jaxpr aval; PRNG key arrays are uint32 words with a trailing axis of 2."""
    import jax
    if jax.dtypes.issubdtype(aval.dtype, jax.dtypes.prng_key):
        if str(aval.dtype) != 'key<fry>':raise Unsupported(f'PRNG implementation {aval.dtype}')
        return np.dtype(np.uint32), tuple(aval.shape) + (2,)
    return np.dtype(aval.dtype), tuple(aval.shape)


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


def sort_kind(dtype):
    dtype = np.dtype(dtype)
    return 2 if dtype == F32 else (1 if dtype in SIGNED else 0)


# ---------------------------------------------------------------------------
# Scalar semantics.

def binary(prim, dtype, a, b):
    t = np.dtype(dtype);c = ctype(t)
    if t == F32:
        table = dict(add=f'v2e::fadd({a}, {b})', sub=f'v2e::fsub({a}, {b})', mul=f'v2e::fmul({a}, {b})',
            div=f'v2e::fdiv({a}, {b})', rem=f'v2e::frem({a}, {b})', max=f'v2e::fmax_x({a}, {b})', min=f'v2e::fmin_x({a}, {b})')
    elif t == BOOL:
        table = dict(**{'and': f'({a} && {b})', 'or': f'({a} || {b})', 'xor': f'({a} != {b})'},
            max=f'({a} || {b})', min=f'({a} && {b})', add=f'({a} || {b})', mul=f'({a} && {b})', ne=f'({a} != {b})')
    else:
        # Wrapping arithmetic in 32-bit unsigned (narrow operands would otherwise promote to int and overflow).
        wrap = lambda op: f'(({c})((uint32_t)({a}) {op} (uint32_t)({b})))'
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
            ceil=f'v2e::fceil({a})', is_finite=f'v2e::fisfinite({a})', square=f'v2e::fmul({a}, {a})',
            round=f'v2e::fround_even({a})' if int(params.get('rounding_method', 0)) == 1 else f'v2e::fround_away({a})')
    else:
        table = dict(neg=f'(({c})(0u - (uint32_t)({a})))', abs=f'(({a}) < 0 ? ({c})(0u - (uint32_t)({a})) : ({a}))',
            sign=f'(({c})((({a}) > 0) - (({a}) < 0)))' if t in SIGNED else f'(({c})(({a}) != 0))',
            population_count=f'(({c})__popc((uint32_t)({a})))' if t.itemsize == 4 else f'(({c})__popc((uint32_t)(({a}) & 0x{(1 << 8 * t.itemsize) - 1:x}u)))',
            clz=f'(({c})__clz((int)({a})))' if t.itemsize == 4 else f'(({c})(__clz((int)((uint32_t)({a}) & 0x{(1 << 8 * t.itemsize) - 1:x}u)) - {32 - 8 * t.itemsize}))', square=binary('mul', t, a, a) if t != BOOL else f'({a})')
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
    return f'(({ctype(dst)})({a}))'


OPAQUE = {'optimization_barrier'}   # XLA's constant folding stops at these
CALLS = {'jit', 'pjit', 'closed_call', 'custom_jvp_call', 'custom_vjp_call'}
DATA_MOVEMENT = {'slice', 'concatenate', 'transpose', 'rev', 'pad', 'dynamic_slice', 'dynamic_update_slice', 'gather',
    'copy', 'copy_p', 'squeeze', 'expand_dims', 'reshape', 'broadcast_in_dim', 'select_n', 'stack', 'unstack'}
LANE_FOLDABLE = {'add', 'sub', 'mul', 'div', 'max', 'min', 'and', 'or', 'xor', 'not', 'neg', 'abs', 'floor', 'ceil',
    'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'select_n', 'convert_element_type', 'broadcast_in_dim'}
ELEMENTWISE_BINARY = {'add', 'sub', 'mul', 'div', 'rem', 'max', 'min', 'and', 'or', 'xor',
    'shift_left', 'shift_right_logical', 'shift_right_arithmetic'}
COMPARE = {'eq', 'ne', 'lt', 'le', 'gt', 'ge'}
UNARY = {'not', 'neg', 'abs', 'sign', 'floor', 'ceil', 'round', 'is_finite', 'population_count', 'clz', 'square'}


# ---------------------------------------------------------------------------
# Code emission state.

class Kernel:
    CHUNK_LINES = 2500

    def __init__(self, name):
        if os.environ.get('V2E_CHUNK_LINES'):self.CHUNK_LINES = int(os.environ['V2E_CHUNK_LINES'])
        self.name = name;self.lines = [];self.chunks = [];self.indent = 1;self.words = 0;self.free = [];self.peak = 0
        self.consts = [];self.counter = 0;self.lane_literals = 0;self.smem = 0;self.chunk_smem = []

    def emit(self, line):
        self.lines.append('  ' * self.indent + line)

    def open(self, head):
        self.emit(head + ' {');self.indent += 1

    def close(self):
        self.indent -= 1;self.emit('}')

    def var(self, prefix='v'):
        self.counter += 1
        return f'{prefix}{self.counter}'

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
        while self.free and self.free[-1][0] + self.free[-1][1] == self.words:
            o, n = self.free.pop();self.words = o

    def coords(self, i, shape):
        st = strides(shape);out = []
        for d, (s, n) in enumerate(zip(st, shape)):
            if n == 1:out.append('0')
            elif d == 0:out.append(f'(({i}) / {s})' if s > 1 else f'({i})')
            else:out.append(f'((({i}) / {s}) % {n})' if s > 1 else f'(({i}) % {n})')
        return out


@dataclass
class Module:
    name: str
    files: dict
    arena_words: int
    inputs: list = field(default_factory=list)
    consts: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    lane_literals: int = 0
    marks: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def digest(self):
        h = hashlib.sha256()
        for name in sorted(self.files):h.update(name.encode());h.update(self.files[name].encode())
        return h.hexdigest()[:16]

    @property
    def source_lines(self):
        return sum(text.count(chr(10)) for text in self.files.values())


def thread_index(n):
    """This thread's (clamped) element of a register value with n elements."""
    return '0' if n <= 1 else ('t' if n >= REG else f'(t < {n} ? t : {n - 1})')


class Generator:
    def __init__(self, name, profile=False):
        self.k = Kernel(name);self.env = {};self.profile = profile;self.marks = [];self.live = {};self.temps = []
        self.cse = {};self.cse_hits = 0;self.folds = 0;self.declined_runtime = False;self.fold_recip = False
        self.collective = True   # warp collectives (shuffles, warp scans) available
        self.rolled = 0          # depth of emitted loops that are not fully unrolled

    # -- storage ------------------------------------------------------------------
    def new_reg(self, out):
        h = Holder('reg', max(1, out.size), out.dtype, name=self.k.var('r'))
        out.src = 'store';out.holder = h;out.imap = None;self.live[h] = None
        return out

    def new_arena(self, out):
        h = Holder('arena', max(1, out.size), out.dtype, off=self.k.alloc(out.size))
        out.src = 'store';out.holder = h;out.imap = None;self.live[h] = None
        return out

    def unref(self, h):
        h.refs -= 1
        if h.refs == 0:self.free_holder(h)

    def free_holder(self, h):
        if h.kind == 'arena' and h.off >= 0:self.k.release(h.off, h.words);h.off = -1
        self.live.pop(h, None);self.release_deps(h)

    def release_deps(self, h):
        deps, h.deps = h.deps, []
        for d in deps:self.unref(d)

    def take_over(self, op, out):
        """out reuses dying op's arena slot in place. The slot moves to a fresh holder and op's holder
        is retired, so nothing (in particular a CSE entry) can still read op through it."""
        old = op.holder
        h = Holder('arena', old.words, out.dtype, off=old.off)
        old.off = -1;self.live.pop(old, None)
        out.src = 'store';out.holder = h;out.imap = None;self.live[h] = None

    def spill_live(self):
        """Move every live register value to the arena (kernel-chunk boundary)."""
        for h in list(self.live):
            if h.kind != 'reg' or h.refs <= 0:continue
            off = self.k.alloc(h.words)
            if h.words == 1:self.k.emit(f'if (t == 0) A[{off}] = {to_word(h.dtype, h.name)};')
            else:self.k.emit(f'if (t < {h.words}) A[{off} + t] = {to_word(h.dtype, h.name)};')
            h.kind = 'arena';h.off = off
        self.k.emit('__syncwarp();')

    def temp_arena(self, v):
        """Dense arena copy of v valid until release_temps()."""
        h = Holder('arena', max(1, v.size), v.dtype, off=self.k.alloc(v.size))
        tmp = Val(v.dtype, v.shape, 'store', holder=h);self.live[h] = None;h.refs = 1;self.temps.append(h)
        base, i = self.k.var('base'), self.k.var('i')
        self.k.open(f'for (int {base} = 0; {base} < {v.size}; {base} += 32)')
        self.k.emit(f'const int {i} = {base} + t < {v.size} ? {base} + t : {v.size - 1};')
        val = self.get(v, i)
        self.k.emit(f'if ({base} + t < {v.size}) A[{h.off} + {i}] = {to_word(v.dtype, val)};')
        self.k.close()
        self.k.emit('__syncwarp();')
        return tmp

    def dense_arena(self, v):
        return v if (v.dense and v.kind == 'arena') else self.temp_arena(v)

    def release_temps(self):
        for h in self.temps:
            h.refs = 0;self.free_holder(h)
        self.temps = []

    # -- element access -----------------------------------------------------------
    def get(self, v, idx, own=False, dyn=False):
        """Expression for element idx of v as read by this thread (may emit a shuffle).

        Must be called in warp-uniform control flow (shuffles); own=True asserts
        this thread holds element idx of a dense register value.
        """
        if v.imap is not None:idx = v.imap(idx);own = False
        kind = v.kind
        if kind == 'lit':return literal(v.lit, v.dtype)
        if kind == 'expr':return v.fn(idx)
        if kind == 'in':return f'{v.ptr}[(size_t)b * {v.stride} + ({idx})]'
        if kind == 'const':return f'{v.ptr}[{idx}]'
        h = v.holder
        if kind == 'arena':return word_to(v.dtype, f'A[{h.off} + ({idx})]')
        if h.words == 1 or own:return h.name   # size-1 registers are uniform across the warp
        s = self.k.var('s')
        self.k.emit(f'const {ctype(v.dtype)} {s} = v2e::shfl<{ctype(v.dtype)}>({h.name}, (int)({idx}));')
        return s

    def produce(self, out, expr):
        """Materialize out with element i = expr(i, own); expr may emit statements."""
        n = out.size
        if n <= REG:
            self.new_reg(out)
            i = self.k.var('i')
            self.k.emit(f'const int {i} = {thread_index(n)};')
            e = expr(i, True)
            self.k.emit(f'const {ctype(out.dtype)} {out.holder.name} = {e};')
            return out
        self.new_arena(out)
        base, i, ic = self.k.var('base'), self.k.var('i'), self.k.var('ic')
        self.k.open(f'for (int {base} = 0; {base} < {n}; {base} += 32)')
        self.k.emit(f'const int {i} = {base} + t; const int {ic} = {i} < {n} ? {i} : {n - 1};')
        e = expr(ic, False)
        self.k.emit(f'if ({i} < {n}) A[{out.holder.off} + {i}] = {to_word(out.dtype, e)};')
        self.k.close()
        self.k.emit('__syncwarp();')
        return out

    def full_coords(self, out, i):
        return (['b'] if out.lane else []) + self.k.coords(i, out.lshape)

    def flat(self, v, full):
        """Per-lane flat index of v at full coordinates (lane coordinate dropped)."""
        st = strides(v.lshape);off = 1 if v.lane else 0
        parts = [f'({full[d]}) * {st[d - off]}' for d in range(off, len(v.shape)) if v.shape[d] != 1]
        return ' + '.join(parts) or '0'

    def loop_open(self, var, n, start=0):
        """Emit a counted loop; short loops are fully unrolled (constant indices after unrolling)."""
        if n - start <= REG:self.k.emit('#pragma unroll')
        else:self.rolled += 1
        self.k.open(f'for (int {var} = {start}; {var} < {n}; ++{var})')

    def loop_close(self, n, start=0):
        self.k.close()
        if n - start > REG:self.rolled -= 1

    def mark_cond(self):
        return 'b == 0 && t == 0'

    # -- module emission -------------------------------------------------------------
    def input_value(self, v):
        if v.size > REG:return v
        # Small inputs are loaded into registers once.
        ptr, size = v.ptr, v.size
        return self.produce(Val(v.dtype, v.shape, 'store'), lambda i, own: f'{ptr}[(size_t)b * {size} + {i}]')

    def emit_output(self, v):
        k = self.k;size = v.size
        if v.kind == 'reg' and v.dense:
            k.emit(f'if (t < {size}) o[(size_t)b * {size} + t] = {v.holder.name};')
            return
        base = k.var('base')
        k.open(f'for (int {base} = 0; {base} < {size}; {base} += 32)')
        i = k.var('i');k.emit(f'const int {i} = {base} + t < {size} ? {base} + t : {size - 1};')
        val = self.get(v, i)
        k.emit(f'if ({base} + t < {size}) o[(size_t)b * {size} + {i}] = {val};')
        k.close()

    def chunk_source(self, ns, c, lines, smem, decl):
        warps = WARPS if smem * 4 * WARPS <= 40 * 1024 else 1
        smem_decl = f'__shared__ uint32_t S_all[{warps}][{smem}]; uint32_t* S = S_all[threadIdx.x >> 5]; (void)S;' if smem else ''
        return f'''// Generated chunk {c} of `{self.k.name}`. Do not edit.
#include "{ns}.cuh"

namespace {ns} {{

// One warp per lane: thread t holds element t of every register value.
__global__ void __launch_bounds__({32 * warps}) chunk_{c}(const Params p) {{
  const int64_t b = (int64_t)blockIdx.x * {warps} + (threadIdx.x >> 5);
  const int t = threadIdx.x & 31;
  if (b >= p.lanes) return;
  uint32_t* __restrict__ A = p.arena + (size_t)b * kArenaWords;
  uint32_t* err = p.err;
  {smem_decl}
  const long long T0 = clock64(); (void)T0;
  (void)A; (void)err;
{chr(10).join(decl)}
{chr(10).join(lines)}
}}

void launch_{c}(const Params& p, cudaStream_t stream) {{
  chunk_{c}<<<static_cast<unsigned>((p.lanes + {warps - 1}) / {warps}), {32 * warps}, 0, stream>>>(p);
}}

}}  // namespace {ns}
'''

    # -- traversal ------------------------------------------------------------------
    def read(self, atom):
        from jax.extend import core as jcore
        if isinstance(atom, jcore.Literal):
            val = np.asarray(atom.val)
            if val.shape:raise Unsupported('non-scalar literal')
            if val.dtype.kind in 'iu' and int(val) in (LANES, -LANES, LANES - 1):self.k.lane_literals += 1
            return Val(np.dtype(atom.aval.dtype), (), 'lit', lit=val[()] if val.dtype == np.float32 else val.item(), cderived=True)
        return self.env[atom]

    @staticmethod
    def value_key(v):
        if v.src == 'lit' and not v.shape:return ('lit', str(v.dtype), np.asarray(v.lit, v.dtype).tobytes())   # bits: NaN payloads, -0
        return id(v)

    def cse_key(self, e, ins):
        """Value-numbering key of a pure equation (None: not deduplicated)."""
        from jax.extend import core as jcore
        for value in e.params.values():
            if isinstance(value, (jcore.Jaxpr, jcore.ClosedJaxpr)) or (isinstance(value, (tuple, list)) and value and isinstance(value[0], (jcore.Jaxpr, jcore.ClosedJaxpr))):
                return None
        try:params = repr(sorted(e.params.items()))
        except Exception:return None
        return (e.primitive.name, params, tuple(self.value_key(v) for v in ins), tuple(str(v.aval) for v in e.outvars))

    def alive(self, v):
        return v.src != 'store' or v.holder in self.live

    def run(self, jaxpr, args):
        from jax.extend import core as jcore
        uses = {}
        for e in jaxpr.eqns:
            for a in e.invars:
                if not isinstance(a, jcore.Literal):uses[a] = uses.get(a, 0) + 1
        for a in jaxpr.outvars:
            if not isinstance(a, jcore.Literal):uses[a] = uses.get(a, 0) + 1
        saved = self.env;self.env = {}
        neg_fed = {a for e in jaxpr.eqns if e.primitive.name in {'neg'} | CALLS for a in e.invars if not isinstance(a, jcore.Literal)}
        neg_fed |= {a for a in jaxpr.outvars if not isinstance(a, jcore.Literal)}
        for var, val in zip(jaxpr.invars, args):
            self.env[var] = val
            if val.src == 'store':val.holder.refs += uses.get(var, 0)
        for e in jaxpr.eqns:
            ins = [self.read(a) for a in e.invars]
            pending = {}
            for v in ins:
                if v.src == 'store':pending[id(v.holder)] = pending.get(id(v.holder), 0) + 1
            dying = {id(v) for v in ins if v.src == 'store' and v.holder.refs == pending[id(v.holder)] == 1}
            before = len(self.k.lines);self.eq_start = before
            key = self.cse_key(e, ins);hit = self.cse.get(key) if key is not None else None
            self.declined_runtime = False
            folded = self.constant_fold(e, ins) if hit is None else None
            if folded is not None:
                outs = folded;self.folds += 1
            elif hit is not None and all(self.alive(v) for v in hit[1]):
                outs = hit[1];self.cse_hits += 1
            else:
                if not self.declined_runtime:self.evaluator_guard(e, ins)
                self.neg_feeds_neg = bool(e.outvars) and e.outvars[0] in neg_fed
                try:outs = self.equation(e, ins, dying)
                except Unsupported as error:
                    if getattr(error, 'located', False):raise
                    ex = Unsupported(f'{error} [primitive {e.primitive.name}: ' + ', '.join(f'{v.kind}{v.shape}{v.dtype}' for v in ins)
                        + ' -> ' + ', '.join(str(tuple(v.aval.shape)) for v in e.outvars) + f'; params {str(e.params)[:300]}]')
                    ex.located = True
                    raise ex from None
                if key is not None:self.cse[key] = (ins, outs)   # ins kept alive: their ids stay unique
            if ins and all(v.cderived for v in ins) and e.primitive.name not in OPAQUE | CALLS and not self.declined_runtime:
                for val in outs:val.cderived = True
            for var, val in zip(e.outvars, outs):
                dropped = type(var).__name__ == 'DropVar'
                if not dropped:self.env[var] = val
                if val.src == 'store':val.holder.refs += 0 if dropped else uses.get(var, 0)
            for v in ins:
                if v.src == 'store':self.unref(v.holder)
            for val in outs:
                if val.src == 'store' and val.holder.refs == 0:self.free_holder(val.holder)
            if self.profile and len(self.k.lines) != before:
                info = None
                tb = getattr(e.source_info, 'traceback', None)
                if tb is not None and tb.frames:
                    frames = [f for f in tb.frames if 'v2e_' not in f.file_name and '/jax/' not in f.file_name]
                    info = frames[0] if frames else tb.frames[0]
                where = f'{info.file_name.rsplit("/", 1)[-1]}:{info.line_num}' if info else '?'
                self.marks.append((e.primitive.name, where, ','.join(str(tuple(v.aval.shape)) for v in e.outvars)))
                self.k.emit(f'if ({self.mark_cond()}) err[1 + {len(self.marks) - 1}] = (uint32_t)(clock64() - T0);')
            self.maybe_split()
        outs = [self.read(a) for a in jaxpr.outvars]
        self.env = saved
        return outs

    def maybe_split(self):
        if len(self.k.lines) >= self.k.CHUNK_LINES and self.k.indent == 1:
            self.spill_live()
            self.k.chunks.append(self.k.lines);self.k.chunk_smem.append(self.k.smem);self.k.lines = [];self.k.smem = 0

    def equation(self, e, ins, dying):
        p = e.primitive.name;params = e.params
        outs = [Val(*aval_of(v.aval), 'store') for v in e.outvars]
        for o in outs:
            if LANES in o.shape[1:]:raise Unsupported(f'{p}: lane size off axis 0 in {o.shape}')
        method = getattr(self, 'p_' + p.replace('-', '_'), None)
        if method is not None:return method(e, ins, outs, params, dying)
        if p in ELEMENTWISE_BINARY:
            folded = self.identity_fold(p, ins, outs[0])
            if folded is not None:return [folded]
            if p == 'div' and ins[0].dtype == F32 and ins[1].bsig == 'uncertain':
                raise Unsupported('f32 division by a reshaped or sliced constant broadcast (XLA folding not reproduced)')
            if p == 'div' and ins[0].dtype == F32 and self.const_like(ins[1]) and not self.plain_const(ins[0]) and not (
                    ins[0].cderived and self.fold_signature('div', ins, e) != 'recip'):
                return [self.divide_by_constant(ins[0], ins[1], outs[0])]
            return [self.elementwise(outs[0], ins, lambda a, b: binary(p, ins[0].dtype, a, b))]
        if p in COMPARE:return [self.elementwise(outs[0], ins, lambda a, b: compare(p, a, b))]
        if p in UNARY:
            if p == 'neg' and ins[0].dtype == F32:
                inner = getattr(ins[0], 'neg_of', None)
                if inner is not None and self.alive(inner):return [self.view(inner, outs[0])]   # XLA: neg(neg(x)) -> x
                out = self.elementwise(outs[0], ins, lambda a: unary(p, ins[0].dtype, a, params));out.neg_of = ins[0]
                if ins[0].src == 'store' and out.src == 'store' and getattr(self, 'neg_feeds_neg', True):
                    # PTX neg.f32 canonicalizes NaN, so the fold (which keeps payloads) must stay available.
                    ins[0].holder.refs += 1;out.holder.deps.append(ins[0].holder)
                return [out]
            return [self.elementwise(outs[0], ins, lambda a: unary(p, ins[0].dtype, a, params))]
        raise Unsupported(f'primitive {p}')

    # -- constant folding (XLA folds all-constant subexpressions with its IEEE evaluator) -------
    FOLD_F32 = {'add': np.add, 'sub': np.subtract, 'mul': np.multiply, 'div': np.divide}
    FOLD_ANY = {'eq': np.equal, 'ne': np.not_equal, 'lt': np.less, 'le': np.less_equal, 'gt': np.greater, 'ge': np.greater_equal}
    FOLD_INT = {'add': np.add, 'sub': np.subtract, 'mul': np.multiply, 'and': np.bitwise_and, 'or': np.bitwise_or,
        'xor': np.bitwise_xor, 'max': np.maximum, 'min': np.minimum}

    @staticmethod
    def is_scalar(v):
        return v.kind == 'lit' or v.size <= 1

    @staticmethod
    def plain_const(v):
        """A constant as XLA sees it (not a broadcast of one)."""
        return v.kind == 'const' and v.arr is not None and v.bsig is None and v.size > 1

    def const_like(self, v):
        """A constant or a broadcast of a constant (XLA's m::Constant / m::Broadcast(m::Constant))."""
        if v.kind == 'lit':return True
        if not v.cderived or v.bsig == 'uncertain':return False
        return (v.kind == 'const' and v.arr is not None) or v.lbc is not None

    def reciprocal(self, b):
        """1 / b computed on the host in IEEE f32 (XLA's InvertConstant), with b's provenance."""
        with np.errstate(all='ignore'):
            if b.kind == 'lit':return Val(F32, b.shape, 'lit', lit=np.float32(1.0) / np.float32(b.lit), cderived=True)
            if b.lbc is not None:
                r = self.wrap((np.float32(1.0) / np.asarray(b.lbc, np.float32))[None], F32, b.shape, lane_mode=True)
            else:
                r = self.const(np.float32(1.0) / np.asarray(b.arr, np.float32))
        r.bsig = b.bsig;r.cderived = True
        return r

    def divide_by_constant(self, a, b, out):
        """XLA's A / C -> A * (1 / C) for a non-constant A and a constant (or broadcast constant) C. The
        product then folds by the usual rule (or runs at run time, an IEEE multiply on the device)."""
        recip = self.reciprocal(b)
        if a.cderived:
            sig = self.fold_signature('mul', [a, recip], None)
            if sig not in ('runtime', 'uncertain', 'recip'):
                lane = out.lane
                def value(v):
                    if v.kind == 'lit':return np.asarray(v.lit, np.float32)
                    if v.lbc is not None:return np.asarray(v.lbc, np.float32)[None]
                    return np.asarray(v.arr, np.float32)
                with np.errstate(all='ignore'):
                    r = np.broadcast_to(value(a) * value(recip), ((1,) + tuple(out.shape[1:])) if lane else out.shape)
                res = self.wrap(r, F32, out.shape, lane_mode=lane)
                if res is not None:
                    res.bsig = sig;res.cderived = True
                    return res
            if sig == 'uncertain':raise Unsupported('f32 division of a reshaped or sliced constant broadcast')
        res = self.elementwise(out, [a, recip], lambda x, y: binary('mul', F32, x, y))
        res.cderived = False;self.declined_runtime = True   # a runtime product, as in XLA
        return res

    def fold_signature(self, p, ins, e):
        """How XLA treats an op whose operands are all constant-derived: the provenance of the folded
        result, or 'runtime'. HloConstantFolding needs real constant operands (a broadcast of a
        constant is not one); the algebraic simplifier first sinks broadcasts through elementwise ops
        when the broadcasts match, or when there is one non-scalar operand. So an elementwise op folds
        if an operand is a plain constant, if all non-scalar operands are identical broadcasts, or if
        there is at most one non-scalar broadcast; otherwise XLA evaluates it at run time."""
        if p == 'broadcast_in_dim':
            v = ins[0];bd = tuple(e.params['broadcast_dimensions']);shape = tuple(e.outvars[0].aval.shape)
            if self.is_scalar(v):return None
            if v.bsig is None:return (tuple(v.shape), bd, shape)
            if v.bsig == 'uncertain':return 'uncertain'
            src_shape, inner, _ = v.bsig
            return (src_shape, tuple(bd[d] for d in inner), shape)
        nonscalar = [v for v in ins if not self.is_scalar(v)]
        if any(v.bsig == 'uncertain' for v in nonscalar):return 'uncertain'
        if p == 'div' and ins[0].dtype == F32:
            sigs = {v.bsig for v in nonscalar}
            if all(v.bsig is None for v in nonscalar):return None   # constants only (C/C, k/C, C/k, k/k): exact fold
            if None not in sigs and len(sigs) == 1:
                # Broadcasts sink; the inner division folds exactly unless its dividend is a broadcast scalar,
                # where the simplifier's A / C -> A * (1 / C) fires first (probed: k / bcast(C) = k * RN(1/C)).
                if self.is_scalar(ins[0]) and not self.is_scalar(ins[1]):self.fold_recip = True
                return next(iter(sigs))
            if self.plain_const(ins[0]):return None   # a constant dividend: HloConstantFolding divides exactly
            return 'recip'                            # the simplifier's A / C -> A * (1 / C) runs first
        if p in DATA_MOVEMENT:
            return None if all(v.bsig is None for v in nonscalar) else 'uncertain'
        if any(v.bsig is None for v in nonscalar):return None   # a plain constant operand: folded
        sigs = {v.bsig for v in nonscalar}
        if len(sigs) <= 1:return next(iter(sigs), None)      # matching broadcasts, or a unique one: sunk, then folded
        return 'runtime'

    def lane_value(self, src, out, bd):
        """Lane-free value of broadcast_in_dim(src) when src has no lane axis (every lane sees it)."""
        target = [1] * len(out.shape)
        for k, d in enumerate(bd):target[d] = src.shape[k]
        return np.broadcast_to(np.asarray(src.arr).reshape(target), (1,) + tuple(out.lshape))[0]

    def constant_fold(self, e, ins):
        """Evaluate an equation whose operands are all literals or closed-over constants, as XLA's
        constant folding does (IEEE f32 on the host, so e.g. a constant division is exact, not div.full).
        Values broadcast from constants along the lane axis are evaluated at the constant's own shape and
        broadcast again, as XLA sinks such broadcasts. Folds only where the evaluator's result is
        unambiguous; anything else is generated as usual (or refused by evaluator_guard)."""
        p = e.primitive.name
        if not ins or any(v.imap is not None for v in ins):return None
        out_lane = bool(e.outvars) and len(e.outvars[0].aval.shape) > 0 and e.outvars[0].aval.shape[0] == LANES
        lane_mode = any(v.lbc is not None for v in ins) or out_lane   # a lane-shaped result of constants
        def foldable(v):
            if v.kind == 'lit':return True
            if v.lbc is not None:return True
            return v.kind == 'const' and v.arr is not None
        if not all(foldable(v) for v in ins):return None
        if lane_mode and p not in LANE_FOLDABLE:return None
        self.fold_recip = False
        sig = self.fold_signature(p, ins, e)
        if sig in ('runtime', 'recip'):
            self.declined_runtime = True   # XLA computes this at run time (or rewrites it); so does the device code
            return None
        if sig == 'uncertain' and p not in DATA_MOVEMENT and any(v.dtype == F32 for v in ins):
            return None   # unknown XLA provenance: the guard refuses it
        avals = [aval_of(o.aval) for o in e.outvars]
        if any(sh != tuple(o.aval.shape) for (dt, sh), o in zip(avals, e.outvars)):return None   # PRNG keys
        if any(hasattr(a, 'aval') and aval_of(a.aval)[1] != tuple(a.aval.shape) for a in e.invars):return None
        if len(e.outvars) != 1 and p != 'unstack':return None
        dtype, shape = avals[0]
        eval_shape = ((1,) + tuple(shape[1:])) if lane_mode else shape   # a lane axis of 1 stands for "every lane"
        def value(v):
            if v.kind == 'lit':
                return np.broadcast_to(np.asarray(v.lit, v.dtype), ((1,) + tuple(v.shape[1:])) if (lane_mode and v.lane) else v.shape)
            if v.lbc is not None:return np.asarray(v.lbc)[None]
            return np.asarray(v.arr)
        try:
            xs = [value(v) for v in ins]
            with np.errstate(all='ignore'):
                t = ins[0].dtype
                if p in DATA_MOVEMENT - {'broadcast_in_dim', 'reshape', 'squeeze', 'expand_dims', 'select_n'}:
                    r = self.move(e, xs)   # bit-exact data movement, as XLA folds it
                    if r is None:return None
                    if p == 'unstack':
                        res = [self.wrap(np.asarray(x).astype(dt), dt, sh) for x, (dt, sh) in zip(r, avals)]
                        return None if any(v is None for v in res) else res
                elif p == 'broadcast_in_dim':
                    bd = tuple(e.params['broadcast_dimensions']);target = [1] * len(shape)
                    for k, d in enumerate(bd):target[d] = xs[0].shape[k]
                    r = np.broadcast_to(xs[0].reshape(target), eval_shape)
                elif p in ('reshape', 'squeeze', 'expand_dims') and len(xs) == 1 and not lane_mode:
                    if p == 'reshape' and e.params.get('dimensions') not in (None, ()):return None
                    if ins[0].kind == 'const':return [self.view(ins[0], Val(dtype, shape, 'store'))]   # same buffer
                    r = xs[0].reshape(shape)
                elif p in self.FOLD_ANY and len(xs) == 2:r = self.FOLD_ANY[p](xs[0], xs[1])
                elif p == 'select_n':
                    full = np.broadcast_shapes(*[x.shape for x in xs])
                    r = np.choose(np.broadcast_to(xs[0].astype(np.int32), full), [np.broadcast_to(x, full) for x in xs[1:]])
                elif t == F32 and p == 'div' and self.fold_recip:
                    r = xs[0].astype(np.float32) * (np.float32(1.0) / xs[1].astype(np.float32))   # k * RN(1 / C), as XLA rewrites it
                elif t == F32 and p in self.FOLD_F32:r = self.FOLD_F32[p](xs[0].astype(np.float32), xs[1].astype(np.float32))
                elif t == F32 and p in ('neg', 'abs', 'floor', 'ceil'):
                    r = {'neg': np.negative, 'abs': np.abs, 'floor': np.floor, 'ceil': np.ceil}[p](xs[0].astype(np.float32))
                elif t in SIGNED | UNSIGNED and p in self.FOLD_INT:r = self.FOLD_INT[p](xs[0], xs[1]).astype(t)
                elif t in SIGNED | UNSIGNED and p == 'neg':r = (-xs[0]).astype(t)
                elif t == BOOL and p in ('and', 'or', 'xor', 'not'):
                    r = {'and': np.logical_and, 'or': np.logical_or, 'xor': np.logical_xor}[p](xs[0], xs[1]) if p != 'not' else np.logical_not(xs[0])
                elif p == 'convert_element_type':
                    x = xs[0]
                    if x.dtype == np.float32 and dtype != F32:
                        # Truncation is unambiguous only for finite in-range values (the guard refuses the rest).
                        if dtype == BOOL:r = x != 0
                        else:
                            info = np.iinfo(dtype)
                            if not (np.isfinite(x).all() and (x > info.min - 1).all() and (x < info.max + 1).all()):return None
                            r = np.trunc(x).astype(dtype)
                    else:r = x.astype(dtype)
                else:return None
            # NaN results come from the host FPU, as XLA's evaluator computes them in this same process.
            r = np.broadcast_to(np.asarray(r).astype(dtype), eval_shape)
        except (TypeError, ValueError):
            return None
        res = self.wrap(r, dtype, shape, lane_mode)
        if res is None:return None
        res.bsig = sig
        return [res]

    def wrap(self, r, dtype, shape, lane_mode=False):
        """A folded value as a literal (uniform), a constant, or a lane broadcast of a constant."""
        flat = np.ascontiguousarray(r).reshape(-1)   # broadcast results may be zero-stride views
        if flat.size and np.all(flat.view(np.uint8).reshape(flat.size, -1) == flat[:1].view(np.uint8).reshape(1, -1)):
            return Val(dtype, shape, 'lit', lit=flat[0] if flat.dtype == np.float32 else flat[0].item())   # f32 keeps NaN bits
        if not lane_mode:
            if LANES in shape:return None
            return self.const(np.broadcast_to(r, shape).copy())
        c = self.const(np.ascontiguousarray(r[0]));out = Val(dtype, shape, 'lit')
        self.lazy(out, lambda i: self.get(c, self.flat(c, self.full_coords(out, i)[1:])));out.lbc = np.asarray(c.arr)
        return out

    @staticmethod
    def move(e, xs):
        """Bit-exact evaluation of a data-movement primitive (numpy, or XLA's CPU backend if available)."""
        p = e.primitive.name;q = e.params
        if p in ('copy', 'copy_p'):return xs[0]
        if p == 'slice':
            st = q['strides'] or (1,) * len(q['start_indices'])
            return xs[0][tuple(slice(a, b, c) for a, b, c in zip(q['start_indices'], q['limit_indices'], st))]
        if p == 'concatenate':return np.concatenate(xs, axis=int(q['dimension']))
        if p == 'transpose':return np.transpose(xs[0], q['permutation'])
        if p == 'rev':return np.flip(xs[0], axis=tuple(q['dimensions'])) if q['dimensions'] else xs[0]
        if p == 'stack':return np.stack(xs, axis=int(q['axis']))
        if p == 'unstack':return list(np.moveaxis(xs[0], int(q['axis']), 0))
        try:
            import jax
            cpu = jax.devices('cpu')[0]
        except RuntimeError:
            return None   # no CPU backend in this process: leave it to the device (the guard decides)
        with jax.default_device(cpu):
            r = e.primitive.bind(*[jax.device_put(np.array(x, order='C'), cpu) for x in xs], **q)   # keeps 0-d operands 0-d
        return np.asarray(r[0] if isinstance(r, (list, tuple)) else r)

    SENSITIVE_F32 = {'div', 'max', 'min', 'neg', 'abs', 'reduce_sum', 'reduce_max', 'reduce_min', 'reduce_prod',
        'argmax', 'argmin', 'cumsum', 'cummax', 'cummin', 'cumprod', 'sort'}

    def evaluator_guard(self, e, ins):
        """XLA folds constant-derived subexpressions on the host. Where the host evaluator and the device
        disagree (division vs div.full, NaN payloads in min/max/neg/abs, reduction order and signed zeros,
        f32 -> int) and the generator could not fold the same expression, refuse instead of emitting the
        device form."""
        if not ins or not all(v.cderived for v in ins) or e.primitive.name in OPAQUE | CALLS:return
        p = e.primitive.name;f32_in = any(v.dtype == F32 for v in ins)
        if f32_in and any(getattr(v, 'bsig', None) == 'uncertain' for v in ins) and p not in DATA_MOVEMENT:
            raise Unsupported(f'f32 {p} on a reshaped or sliced constant broadcast (XLA folding not reproduced)')
        if (p in self.SENSITIVE_F32 and f32_in) or (p == 'convert_element_type' and ins[0].dtype == F32 and
                np.dtype(e.params['new_dtype']) != F32):
            raise Unsupported(f'constant-derived f32 {p} that XLA folds on the host (not reproduced here)')

    # -- elementwise ----------------------------------------------------------------
    def identity_fold(self, p, ins, out):
        """op(x, NaN literal) -> that NaN, quieted, sign cleared;
        x + 0, 0 + x, x - 0, x * 1, 1 * x, x / 1, max(x, x), min(x, x), min(x, +inf), max(x, -inf) -> x,
        as XLA's algebraic simplifier rewrites them for floats too (pinned by the float_identities
        conformance case: -0 and NaN payloads survive)."""
        if ins[0].dtype != F32:return None
        if p in ('add', 'sub', 'mul', 'div', 'max', 'min'):
            # XLA's GPU codegen folds arithmetic with a NaN literal to that NaN, quieted and with the sign
            # cleared, whatever the other operand is (probed: 0xffc00005 -> 0x7fc00005, 0x7f800003 -> 0x7fc00003).
            for v in ins:
                if v.kind == 'lit' and np.isnan(np.float32(v.lit)):
                    bits = (int(np.asarray(v.lit, np.float32).view(np.uint32)) | 0x00400000) & 0x7fffffff
                    return Val(F32, out.shape, 'lit', lit=np.array([bits], np.uint32).view(np.float32)[0], cderived=True)
        def is_lit(v, value):
            if v.kind == 'lit':return bool(np.float32(v.lit) == np.float32(value))
            if v.kind == 'const' and v.arr is not None and v.arr.size:return bool(np.all(np.asarray(v.arr, np.float32) == np.float32(value)))
            return False
        x = None
        if p in ('max', 'min') and ins[0] is ins[1]:x = ins[0]
        elif p == 'min' and (is_lit(ins[1], np.inf) or is_lit(ins[0], np.inf)):x = ins[0] if is_lit(ins[1], np.inf) else ins[1]
        elif p == 'max' and (is_lit(ins[1], -np.inf) or is_lit(ins[0], -np.inf)):x = ins[0] if is_lit(ins[1], -np.inf) else ins[1]
        elif p == 'add':x = ins[0] if is_lit(ins[1], 0.0) else (ins[1] if is_lit(ins[0], 0.0) else None)
        elif p == 'sub':x = ins[0] if is_lit(ins[1], 0.0) else None
        elif p == 'mul':x = ins[0] if is_lit(ins[1], 1.0) else (ins[1] if is_lit(ins[0], 1.0) else None)
        elif p == 'div':x = ins[0] if is_lit(ins[1], 1.0) else None
        if x is None:return None
        if x.shape == out.shape:return self.view(x, out)
        return self.elementwise(out, [x], lambda a: a)

    def operand_index(self, v, out):
        """(index_fn(i) -> expr, identity) for reading operand v at out element i (numpy broadcasting)."""
        if v.kind == 'lit' or (not v.lane and v.shape == ()):return (lambda i: '0'), False
        if v.lshape == out.lshape and v.lane == out.lane:return (lambda i: i), True
        if len(v.shape) == len(out.shape) and all(a == c or a == 1 for a, c in zip(v.shape, out.shape)) and (v.lane == out.lane or (out.lane and v.shape[0] == 1)):
            return (lambda i: self.flat(v, self.full_coords(out, i))), False
        raise Unsupported(f'elementwise shape {v.shape} -> {out.shape}')

    def elementwise(self, out, ins, fn):
        maps = [self.operand_index(v, out) for v in ins]
        return self.produce(out, lambda i, own: fn(*[self.get(v, m(i), own and ident) for v, (m, ident) in zip(ins, maps)]))

    def p_select_n(self, e, ins, outs, params, dying):
        def fn(p, *cs):
            if len(cs) == 2:return f'(({p}) ? ({cs[1]}) : ({cs[0]}))'
            expr = cs[-1]
            for k in range(len(cs) - 2, -1, -1):expr = f'((int)({p}) == {k} ? ({cs[k]}) : {expr})'
            return expr
        return [self.elementwise(outs[0], ins, fn)]

    def p_convert_element_type(self, e, ins, outs, params, dying):
        src, out = ins[0], outs[0]
        if src.dtype == out.dtype:return [self.view(src, out)]
        return [self.elementwise(out, ins, lambda a: convert(src.dtype, out.dtype, a))]

    def p_bitcast_convert_type(self, e, ins, outs, params, dying):
        src, out = ins[0], outs[0]
        if src.dtype.itemsize != 4 or out.dtype.itemsize != 4:raise Unsupported('bitcast width')
        return [self.elementwise(out, ins, lambda a: word_to(out.dtype, to_word(src.dtype, a)))]

    def p_clamp(self, e, ins, outs, params, dying):
        if ins[1].dtype == F32:fn = lambda a, v, c: f'v2e::fmin_x(v2e::fmax_x({v}, {a}), {c})'
        else:
            def fn(a, v, c):
                m = f'(({v}) < ({a}) ? ({a}) : ({v}))'
                return f'({m} > ({c}) ? ({c}) : {m})'
        return [self.elementwise(outs[0], ins, fn)]

    def p_integer_pow(self, e, ins, outs, params, dying):
        y = int(params['y']);t = ins[0].dtype
        if y != 2:raise Unsupported('integer_pow')
        return [self.elementwise(outs[0], ins, lambda a: binary('mul', t, a, a))]

    # -- views and shape ops ----------------------------------------------------------
    def view(self, src, out):
        """out shares src's storage/expression with the same flat element order."""
        if src.lane != out.lane and src.src not in ('lit',):raise Unsupported('view across lane axis')
        out.src = src.src;out.holder = src.holder;out.ptr = src.ptr;out.lit = src.lit;out.stride = src.stride
        out.imap = src.imap;out.fn = src.fn;out.cderived = src.cderived
        out.arr = None if src.arr is None else np.asarray(src.arr).reshape(out.shape)
        out.lbc = None if src.lbc is None or not out.lane else np.asarray(src.lbc).reshape(out.lshape)
        out.bsig = src.bsig if (src.bsig is None or src.shape == out.shape) else 'uncertain'
        return out

    def p_reshape(self, e, ins, outs, params, dying):
        src, out = ins[0], outs[0]
        if params.get('dimensions') not in (None, ()):raise Unsupported('reshape with dimensions')
        if src.lane != out.lane:raise Unsupported(f'reshape across lane axis {src.shape} -> {out.shape}')
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
        res = []
        for v, o in zip(ins, outs):
            if v.src == 'store' and not v.cderived:res.append(self.view(v, o))
            else:res.append(self.produce(o, lambda i, own, v=v: self.get(v, i)))   # opaque, as XLA's barrier
            o.cderived = False
        return res

    def lazy(self, out, fn):
        out.src = 'expr';out.fn = fn;out.holder = None;out.imap = None
        return out

    def p_iota(self, e, ins, outs, params, dying):
        # Not a constant for folding purposes: XLA's constant folding leaves iota (and ops on it) to run time.
        out = outs[0];dim = int(params['dimension']);c = ctype(out.dtype)
        if out.lane and dim == 0:return [self.lazy(out, lambda i: f'(({c})b)')]
        d = dim - 1 if out.lane else dim;shape = out.lshape
        return [self.lazy(out, lambda i: f'(({c}){self.k.coords(i, shape)[d]})')]

    def p_broadcast_in_dim(self, e, ins, outs, params, dying):
        src, out = ins[0], outs[0];bd = tuple(params['broadcast_dimensions'])
        if src.kind == 'lit':
            out.src = 'lit';out.lit = src.lit;return [out]
        if src.lane and (not bd or bd[0] != 0):raise Unsupported('lane axis moved by broadcast')
        if src.size == out.size and src.lane == out.lane:return [self.view(src, out)]
        def index(i):
            full = self.full_coords(out, i);coords = [('0' if src.shape[j] == 1 and out.shape[od] != 1 else full[od]) for j, od in enumerate(bd)]
            return self.flat(src, coords)
        if src.src == 'expr' or (src.src in ('in', 'const') and out.size > REG):
            self.lazy(out, lambda i: self.get(src, index(i)))
            if src.src == 'const' and src.arr is not None and out.lane:
                out.lbc = self.lane_value(src, out, bd);out.bsig = (tuple(src.shape), bd, tuple(out.shape))
            return [out]
        if out.size > REG:
            out.src = 'store';out.holder = src.holder
            inner = src.imap
            out.imap = (lambda i: inner(index(i))) if inner is not None else index
            return [out]
        return [self.produce(out, lambda i, own: self.get(src, index(i)))]

    def p_transpose(self, e, ins, outs, params, dying):
        src, out = ins[0], outs[0];perm = tuple(params['permutation'])
        if src.lane and perm[0] != 0:raise Unsupported('transpose moves lane axis')
        def index(i):
            full = self.full_coords(out, i);coords = [None] * len(perm)
            for od, idim in enumerate(perm):coords[idim] = full[od]
            return self.flat(src, coords)
        return [self.produce(out, lambda i, own: self.get(src, index(i)))]

    def p_slice(self, e, ins, outs, params, dying):
        src, out = ins[0], outs[0]
        start = tuple(params['start_indices']);stride = tuple(params['strides'] or (1,) * len(start))
        if src.lane and (start[0] != 0 or out.shape[0] != LANES):raise Unsupported('slice across lanes')
        def index(i):
            full = self.full_coords(out, i)
            return self.flat(src, [full[0] if (src.lane and d == 0) else f'({start[d]} + ({full[d]}) * {stride[d]})' for d in range(len(start))])
        return [self.produce(out, lambda i, own: self.get(src, index(i)))]

    def p_rev(self, e, ins, outs, params, dying):
        src, out = ins[0], outs[0];dims = set(params['dimensions'])
        if src.lane and 0 in dims:raise Unsupported('rev over lanes')
        def index(i):
            full = self.full_coords(out, i)
            return self.flat(src, [f'({src.shape[d] - 1} - ({full[d]}))' if d in dims else full[d] for d in range(len(src.shape))])
        return [self.produce(out, lambda i, own: self.get(src, index(i)))]

    def selector(self, cases, default):
        """Nested ternary over (condition, value) pairs; values were computed unconditionally."""
        expr = default
        for cond, value in reversed(cases):expr = f'(({cond}) ? ({value}) : {expr})'
        return expr

    def p_concatenate(self, e, ins, outs, params, dying):
        out = outs[0];dim = int(params['dimension'])
        if out.lane and dim == 0:raise Unsupported('concatenate over lanes')
        def expr(i, own):
            full = self.full_coords(out, i);c = self.k.var('cc');cases = [];offset = 0
            self.k.emit(f'const int {c} = {full[dim]};')
            for src in ins:
                n = src.shape[dim]
                coords = list(full);coords[dim] = f'({c} - {offset} < 0 ? 0 : ({c} - {offset} >= {n} ? {n - 1} : {c} - {offset}))'
                value = self.get(src, self.flat(src, coords))
                cases.append((f'{c} < {offset + n}', value));offset += n
            return self.selector(cases[:-1], cases[-1][1])
        return [self.produce(out, expr)]

    def p_stack(self, e, ins, outs, params, dying):
        out = outs[0];axis = int(params['axis'])
        if out.lane and axis == 0:raise Unsupported('stack over lanes')
        def expr(i, own):
            full = self.full_coords(out, i);k = full[axis];rest = full[:axis] + full[axis + 1:]
            values = [self.get(src, self.flat(src, rest)) for src in ins]
            return self.selector([(f'({k}) == {n}', v) for n, v in enumerate(values[:-1])], values[-1])
        return [self.produce(out, expr)]

    def p_unstack(self, e, ins, outs, params, dying):
        src = ins[0];axis = int(params['axis'])
        if src.lane and axis == 0:raise Unsupported('unstack over lanes')
        res = []
        for k, out in enumerate(outs):
            def expr(i, own, out=out, k=k):
                full = self.full_coords(out, i);full = full[:axis] + [str(k)] + full[axis:]
                return self.get(src, self.flat(src, full))
            res.append(self.produce(out, expr))
        return res

    def p_pad(self, e, ins, outs, params, dying):
        src, padv = ins;out = outs[0];config = params['padding_config']
        if src.lane and tuple(config[0]) != (0, 0, 0):raise Unsupported('pad over lanes')
        def expr(i, own):
            full = self.full_coords(out, i);conds = [];coords = []
            for d, (lo, hi, interior) in enumerate(config):
                if src.lane and d == 0:coords.append(full[0]);continue
                rel = f'(({full[d]}) - ({lo}))';step = interior + 1
                conds.append(f'({rel} >= 0 && {rel} % {step} == 0 && {rel} / {step} < {src.shape[d]})')
                coords.append(f'(({rel} >= 0 && {rel} / {step} < {src.shape[d]}) ? {rel} / {step} : 0)')
            inside = ' && '.join(conds) or 'true'
            value = self.get(src, self.flat(src, coords));fill = self.get(padv, '0')
            return f'(({inside}) ? ({value}) : ({fill}))'
        return [self.produce(out, expr)]

    def starts(self, src, sizes, starts):
        st = []
        for d, s in enumerate(starts):
            if src.lane and d == 0:st.append('0');continue
            v = self.k.var('ds');hi = src.shape[d] - sizes[d]
            self.k.emit(f'int64_t {v} = (int64_t){self.get(s, "0")}; {v} = {v} < 0 ? 0 : ({v} > {hi} ? {hi} : {v});');st.append(v)
        return st

    def p_dynamic_slice(self, e, ins, outs, params, dying):
        src, starts = ins[0], ins[1:];out = outs[0];sizes = tuple(params['slice_sizes'])
        if src.lane and sizes[0] != LANES:raise Unsupported('dynamic_slice across lanes')
        st = self.starts(src, sizes, starts)
        def expr(i, own):
            full = self.full_coords(out, i)
            return self.get(src, self.flat(src, [full[0] if (src.lane and d == 0) else f'({st[d]} + {full[d]})' for d in range(len(src.shape))]), dyn=True)
        return [self.produce(out, expr)]

    def p_dynamic_update_slice(self, e, ins, outs, params, dying):
        src, upd, starts = ins[0], ins[1], ins[2:];out = outs[0]
        if src.lane != upd.lane:raise Unsupported('dynamic_update_slice across the lane axis')
        st = self.starts(src, upd.shape, starts)
        def expr(i, own):
            full = self.full_coords(out, i);conds = [];coords = []
            for d in range(len(src.shape)):
                if src.lane and d == 0:coords.append(full[0]);continue
                rel = f'(({full[d]}) - {st[d]})'
                conds.append(f'({rel} >= 0 && {rel} < {upd.shape[d]})')
                coords.append(f'({rel} < 0 ? 0 : ({rel} >= {upd.shape[d]} ? {upd.shape[d] - 1} : {rel}))')
            inside = ' && '.join(conds) or 'true'
            new = self.get(upd, self.flat(upd, coords), dyn=True);old = self.get(src, i, own)
            return f'(({inside}) ? ({new}) : ({old}))'
        return [self.produce(out, expr)]

    # -- reductions ---------------------------------------------------------------------
    @staticmethod
    def float_order_guard(what, dtype, n, limit):
        """XLA's float reductions follow its reduction emitter's tree order, and its reduction max/min
        combiner keeps NaN payloads and orders signed zeros by operand position (unlike elementwise
        max.NaN), so results depend on that order; only orders pinned by probes are generated."""
        if np.dtype(dtype) == F32 and n > limit:raise Unsupported(f'order-sensitive float {what} over {n} elements')

    def reduce(self, ins, outs, params, init, combine, order_free, limit=REG * 1024):
        src, out = ins[0], outs[0];axes = tuple(params['axes'])
        if src.lane and 0 in axes:raise Unsupported('reduce over lanes')
        keep = [d for d in range(len(src.shape)) if d not in axes]
        rshape = [src.shape[d] for d in axes];rn = int(math.prod(rshape));c = ctype(out.dtype)
        if rn == 1:return [self.view(src, out)]   # XLA rewrites a size-1 reduction to a reshape
        self.float_order_guard('reduction', src.dtype, rn, limit)
        def src_index(oc, r):
            rc = self.k.coords(r, rshape);coords = [None] * len(src.shape)
            for j, d in enumerate(keep):coords[d] = oc[j]
            for j, d in enumerate(axes):coords[d] = rc[j]
            return self.flat(src, coords)
        if order_free and rn >= PARALLEL_MIN and self.collective:
            # Associative and commutative combiner: the whole warp reduces each output element.
            return [self.warp_each(out, lambda oc: self.warp_reduce(c, init, combine, rn, lambda r: self.get(src, src_index(oc, r))))]
        def expr(i, own):
            oc = self.full_coords(out, i);acc = self.k.var('acc');r = self.k.var('r')
            self.k.emit(f'{c} {acc} = {init};')
            self.loop_open(r, rn)
            self.k.emit(f'{acc} = {combine(acc, self.get(src, src_index(oc, r)))};')
            self.loop_close(rn)
            return acc
        return [self.produce(out, expr)]

    def warp_each(self, out, body):
        """out[o] = body(coords of o) for each output element in turn; body is warp-collective."""
        n = out.size;c = ctype(out.dtype);o = self.k.var('o')
        if n <= REG:
            self.new_reg(out);name = out.holder.name;i = self.k.var('i')
            self.k.emit(f'const int {i} = {thread_index(n)};')
            self.k.emit(f'{c} {name}{{}};')
            self.k.open(f'for (int {o} = 0; {o} < {n}; ++{o})')
            val = body(self.full_coords(out, o))
            self.k.emit(f'if ({i} == {o}) {name} = {val};')
            self.k.close()
            return out
        self.new_arena(out)
        self.k.open(f'for (int {o} = 0; {o} < {n}; ++{o})')
        val = body(self.full_coords(out, o))
        self.k.emit(f'if (t == 0) A[{out.holder.off} + {o}] = {to_word(out.dtype, val)};')
        self.k.close()
        self.k.emit('__syncwarp();')
        return out

    def warp_reduce(self, c, init, combine, rn, element):
        acc, base, r = self.k.var('acc'), self.k.var('base'), self.k.var('r')
        self.k.emit(f'{c} {acc} = {init};')
        self.k.open(f'for (int {base} = 0; {base} < {rn}; {base} += 32)')
        self.k.emit(f'const int {r} = {base} + t < {rn} ? {base} + t : {rn - 1};')
        x = element(r)
        self.k.emit(f'if ({base} + t < {rn}) {acc} = {combine(acc, x)};')
        self.k.close()
        self.k.open('for (int s = 16; s > 0; s >>= 1)')
        y = self.k.var('y')   # shuffle hoisted: combiners may short-circuit
        self.k.emit(f'const {c} {y} = v2e::shfl_xor<{c}>({acc}, s);')
        self.k.emit(f'{acc} = {combine(acc, y)};')
        self.k.close()
        return acc

    def p_reduce_sum(self, e, ins, outs, params, dying):
        t = ins[0].dtype
        if t == F32:
            axes = tuple(params['axes']);src = ins[0];rn = int(math.prod(src.shape[a] for a in axes))
            if rn == 2:return self.pair_sum(ins[0], outs[0], axes)
            return self.reduce(ins, outs, params, literal(0.0, F32), lambda a, x: f'v2e::fadd({a}, {x})', False, limit=2)
        return self.reduce(ins, outs, params, literal(0, t) if t != BOOL else 'false', lambda a, x: binary('add', t, a, x), True)

    def pair_sum(self, src, out, axes):
        """f32 sum over exactly two elements: x0 + x1 without the +0 init (probed: (-0) + (-0) stays -0)."""
        if src.lane and 0 in axes:raise Unsupported('reduce over lanes')
        keep = [d for d in range(len(src.shape)) if d not in axes];rshape = [src.shape[d] for d in axes]
        def at(oc, r):
            rc = self.k.coords(str(r), rshape);coords = [None] * len(src.shape)
            for j, d in enumerate(keep):coords[d] = oc[j]
            for j, d in enumerate(axes):coords[d] = rc[j]
            return self.flat(src, coords)
        return [self.produce(out, lambda i, own: f'v2e::fadd({self.get(src, at(self.full_coords(out, i), 0))}, {self.get(src, at(self.full_coords(out, i), 1))})')]

    def p_reduce_max(self, e, ins, outs, params, dying):
        t = ins[0].dtype
        init = literal(-np.inf, F32) if t == F32 else ('false' if t == BOOL else literal(np.iinfo(t).min, t))
        return self.reduce(ins, outs, params, init, lambda a, x: binary('max', t, a, x), True, limit=1)

    def p_reduce_min(self, e, ins, outs, params, dying):
        t = ins[0].dtype
        init = literal(np.inf, F32) if t == F32 else ('true' if t == BOOL else literal(np.iinfo(t).max, t))
        return self.reduce(ins, outs, params, init, lambda a, x: binary('min', t, a, x), True, limit=1)

    def p_reduce_or(self, e, ins, outs, params, dying):
        t = ins[0].dtype
        return self.reduce(ins, outs, params, 'false' if t == BOOL else literal(0, t), lambda a, x: binary('or', t, a, x), True)

    def p_reduce_and(self, e, ins, outs, params, dying):
        t = ins[0].dtype
        return self.reduce(ins, outs, params, 'true' if t == BOOL else literal(-1 if t in SIGNED else np.iinfo(t).max, t),
            lambda a, x: binary('and', t, a, x), True)

    def p_reduce_xor(self, e, ins, outs, params, dying):
        t = ins[0].dtype
        return self.reduce(ins, outs, params, 'false' if t == BOOL else literal(0, t), lambda a, x: binary('xor', t, a, x), True)

    def p_argmax(self, e, ins, outs, params, dying):
        return self.arg(ins, outs, params, '>')

    def p_argmin(self, e, ins, outs, params, dying):
        return self.arg(ins, outs, params, '<')

    def arg(self, ins, outs, params, op):
        """First index of the max (min); NaN wins over numbers, as XLA's argmax comparator."""
        src, out = ins[0], outs[0];axes = tuple(params['axes'])
        if len(axes) != 1 or (src.lane and axes[0] == 0):raise Unsupported('argmax axes')
        ax = axes[0];n = src.shape[ax];keep = [d for d in range(len(src.shape)) if d != ax];t = src.dtype;c = ctype(t)
        if n == 1:
            out.src = 'lit';out.lit = 0;return [out]
        self.float_order_guard('argmax/argmin', t, n, 1)
        def at(oc, j):
            coords = [None] * len(src.shape)
            for q, d in enumerate(keep):coords[d] = oc[q]
            coords[ax] = j
            return self.flat(src, coords)
        def better(x, ix, y, iy):
            """(x, ix) is preferred over (y, iy)."""
            if t == F32:
                return (f'(v2e::fnan({x}) ? (!v2e::fnan({y}) || {ix} < {iy}) : (!v2e::fnan({y}) && ({x} {op} {y} || ({x} == {y} && {ix} < {iy}))))')
            return f'({x} {op} {y} || ({x} == {y} && {ix} < {iy}))'
        if n >= PARALLEL_MIN and self.collective:
            def body(oc):
                best, bi, base, j = self.k.var('best'), self.k.var('bi'), self.k.var('base'), self.k.var('j')
                self.k.emit(f'{c} {best} = {self.get(src, at(oc, "0"))}; int {bi} = 0;')
                self.k.open(f'for (int {base} = 0; {base} < {n}; {base} += 32)')
                self.k.emit(f'const int {j} = {base} + t < {n} ? {base} + t : {n - 1};')
                x = self.get(src, at(oc, j))
                self.k.emit(f'if ({base} + t < {n} && {better(x, j, best, bi)}) {{ {best} = {x}; {bi} = {j}; }}')
                self.k.close()
                self.k.open('for (int s = 16; s > 0; s >>= 1)')
                y, iy = self.k.var('y'), self.k.var('iy')
                self.k.emit(f'const {c} {y} = v2e::shfl_xor<{c}>({best}, s); const int {iy} = __shfl_xor_sync(v2e::kFull, {bi}, s);')
                self.k.emit(f'if ({better(y, iy, best, bi)}) {{ {best} = {y}; {bi} = {iy}; }}')
                self.k.close()
                return f'(({ctype(out.dtype)}){bi})'
            return [self.warp_each(out, body)]
        def expr(i, own):
            oc = self.full_coords(out, i)
            best, bi, j = self.k.var('best'), self.k.var('bi'), self.k.var('j')
            self.k.emit(f'{c} {best} = {self.get(src, at(oc, "0"))}; int {bi} = 0;')
            self.loop_open(j, n, 1)
            x = self.k.var('x')
            self.k.emit(f'const {c} {x} = {self.get(src, at(oc, j))};')
            self.k.emit(f'if ({better(x, j, best, bi)}) {{ {best} = {x}; {bi} = {j}; }}')
            self.loop_close(n, 1)
            return f'(({ctype(out.dtype)}){bi})'
        return [self.produce(out, expr)]

    def p_cumsum(self, e, ins, outs, params, dying):
        src, out = ins[0], outs[0];ax = int(params['axis']);rev = bool(params.get('reverse', False));t = src.dtype;c = ctype(t)
        if src.lane and ax == 0:raise Unsupported('cumsum over lanes')
        n = src.shape[ax]
        if n == 1:return [self.view(src, out)]
        self.float_order_guard('cumsum', t, n, 1)
        if out.size <= REG:
            # Uniform loop over the scanned axis; each thread keeps the prefix of its own element.
            def expr(i, own):
                full = self.full_coords(out, i);acc, j = self.k.var('acc'), self.k.var('j');pos = full[ax]
                self.k.emit(f'{c} {acc} = {literal(0, t)};')
                self.loop_open(j, n)
                coords = list(full);coords[ax] = f'{n - 1} - {j}' if rev else j
                x = self.get(src, self.flat(src, coords))
                cond = f'{n - 1} - {j} >= ({pos})' if rev else f'{j} <= ({pos})'
                self.k.emit(f'if ({cond}) {acc} = {binary("add", t, acc, x)};')
                self.loop_close(n)
                return acc
            return [self.produce(out, expr)]
        keep = [d for d in range(len(src.shape)) if d != ax and not (src.lane and d == 0)]
        keep_shape = [src.shape[d] for d in keep];rows = int(math.prod(keep_shape))
        self.new_arena(out);off = out.holder.off
        row = self.k.var('row')
        def coords_at(rc, j):
            coords = ['b'] if src.lane else []
            coords = coords + [None] * (len(src.shape) - len(coords));q = 0
            for d in keep:coords[d] = rc[q];q += 1
            coords[ax] = j
            return coords
        if not self.collective:
            # One thread per lane: sequential prefix per row.
            self.rolled += 1
            self.k.open(f'for (int {row} = 0; {row} < {rows}; ++{row})')
            rc = self.k.coords(row, keep_shape);acc, j = self.k.var('acc'), self.k.var('j')
            self.k.emit(f'{c} {acc} = {literal(0, t)};')
            self.k.open(f'for (int {j} = {n - 1}; {j} >= 0; --{j})' if rev else f'for (int {j} = 0; {j} < {n}; ++{j})')
            x = self.get(src, self.flat(src, coords_at(rc, j)))
            self.k.emit(f'{acc} = {binary("add", t, acc, x)};')
            self.k.emit(f'{self.arena_ref(off, self.flat(out, coords_at(rc, j)))} = {to_word(t, acc)};')
            self.k.close();self.k.close()
            self.rolled -= 1
            return [out]
        # Integer (wrapping) prefix sums are order-free: warp scan per row.
        self.k.open(f'for (int {row} = 0; {row} < {rows}; ++{row})')
        rc = self.k.coords(row, keep_shape);carry, base = self.k.var('carry'), self.k.var('base')
        self.k.emit(f'{c} {carry} = {literal(0, t)};')
        self.k.open(f'for (int {base} = 0; {base} < {n}; {base} += 32)')
        q, j, x = self.k.var('q'), self.k.var('j'), self.k.var('x')
        self.k.emit(f'const int {q} = {base} + t < {n} ? {base} + t : {n - 1};')
        self.k.emit(f'const int {j} = {f"{n - 1} - {q}" if rev else q};')
        val = self.get(src, self.flat(src, coords_at(rc, j)))
        self.k.emit(f'{c} {x} = {base} + t < {n} ? {val} : {literal(0, t)};')
        self.k.open('for (int s = 1; s < 32; s <<= 1)')
        y = self.k.var('y')
        self.k.emit(f'const {c} {y} = v2e::shfl_up<{c}>({x}, s);')
        self.k.emit(f'if (t >= s) {x} = {binary("add", t, x, y)};')
        self.k.close()
        self.k.emit(f'{x} = {binary("add", t, x, carry)};')
        self.k.emit(f'if ({base} + t < {n}) A[{off} + {self.flat(out, coords_at(rc, j))}] = {to_word(t, x)};')
        self.k.emit(f'{carry} = v2e::shfl<{c}>({x}, 31);')
        self.k.close();self.k.close()
        self.k.emit('__syncwarp();')
        return [out]

    # -- gather / scatter -----------------------------------------------------------------
    def p_gather(self, e, ins, outs, params, dying):
        op, idx = ins;out = outs[0];dn = params['dimension_numbers'];slice_sizes = tuple(params['slice_sizes'])
        mode = str(params['mode']);fill = params.get('fill_value')
        offset_dims = tuple(dn.offset_dims);collapsed = tuple(dn.collapsed_slice_dims);start_map = tuple(dn.start_index_map)
        obatch = tuple(getattr(dn, 'operand_batching_dims', ()));ibatch = tuple(getattr(dn, 'start_indices_batching_dims', ()))
        oshape = op.shape;ishape = idx.shape;ivd = len(ishape) - 1
        batch_out = [d for d in range(len(out.shape)) if d not in offset_dims]
        idx_batch_dims = [d for d in range(len(ishape)) if d != ivd]
        window = [d for d in range(len(oshape)) if d not in collapsed and d not in obatch]
        fill_mode = 'FILL' in mode.upper()
        def expr(i, own):
            full = self.full_coords(out, i);G = [full[d] for d in batch_out]
            def idx_coords(k):
                c = list(G);c.insert(ivd, str(k));return c
            starts = ['0'] * len(oshape)
            for k, d in enumerate(start_map):
                c = idx_coords(k)
                if idx.lane and c[0] != 'b':self.k.emit(f'if (({c[0]}) != b) atomicOr(err, 1u);')
                s = self.k.var('st');self.k.emit(f'int64_t {s} = (int64_t){self.get(idx, self.flat(idx, c))};');starts[d] = s
            for j, d in enumerate(obatch):starts[d] = G[idx_batch_dims.index(ibatch[j])]
            oob = []
            for d in start_map:
                if op.lane and d == 0:continue
                hi = oshape[d] - slice_sizes[d]
                if fill_mode:oob.append(f'({starts[d]} < 0 || {starts[d]} > {hi})')
                cl = self.k.var('cl');self.k.emit(f'const int64_t {cl} = {starts[d]} < 0 ? 0 : ({starts[d]} > {hi} ? {hi} : {starts[d]});');starts[d] = cl
            wcoords = {d: full[offset_dims[j]] for j, d in enumerate(window)}
            coord = [f'({starts[d]} + {wcoords[d]})' if d in wcoords else f'({starts[d]})' for d in range(len(oshape))]
            if op.lane:self.k.emit(f'if ({coord[0]} != b) atomicOr(err, 2u);');coord[0] = 'b'
            value = self.get(op, self.flat(op, coord), dyn=True)
            if oob:
                fv = literal(fill if fill is not None else (np.nan if op.dtype == F32 else (True if op.dtype == BOOL else
                    (np.iinfo(op.dtype).min if op.dtype in SIGNED else np.iinfo(op.dtype).max))), op.dtype)
                value = f'(({" || ".join(oob)}) ? {fv} : {value})'
            return value
        return [self.produce(out, expr)]

    def scatter_target(self, ins, params):
        """target(u) -> (guard, flat target, value) of update element u, emitting index code."""
        op, idx, upd = ins;dn = params['dimension_numbers'];mode = str(params['mode'])
        uwd = tuple(dn.update_window_dims);inserted = tuple(dn.inserted_window_dims);sd2od = tuple(dn.scatter_dims_to_operand_dims)
        obatch = tuple(getattr(dn, 'operand_batching_dims', ()));ibatch = tuple(getattr(dn, 'scatter_indices_batching_dims', ()))
        oshape = op.shape;ishape = idx.shape;ushape = upd.shape;ivd = len(ishape) - 1
        scatter_dims = [d for d in range(len(ushape)) if d not in uwd]
        idx_batch_dims = [d for d in range(len(ishape)) if d != ivd]
        window = [d for d in range(len(oshape)) if d not in inserted and d not in obatch]
        clip = 'CLIP' in mode.upper()
        def target(u):
            """Emit (guard, flat target, value) for update element u."""
            full = (['b'] if upd.lane else []) + self.k.coords(u, upd.lshape);G = [full[d] for d in scatter_dims]
            starts = ['0'] * len(oshape)
            for k, d in enumerate(sd2od):
                c = list(G);c.insert(ivd, str(k))
                if idx.lane and c[0] != 'b':self.k.emit(f'if (({c[0]}) != b) atomicOr(err, 4u);')
                s = self.k.var('st');self.k.emit(f'int64_t {s} = (int64_t){self.get(idx, self.flat(idx, c))};');starts[d] = s
            for j, d in enumerate(obatch):starts[d] = G[idx_batch_dims.index(ibatch[j])]
            wsize = {d: ushape[uwd[j]] for j, d in enumerate(window)};conds = []
            for d in range(len(oshape)):
                if op.lane and d == 0:continue
                size = wsize.get(d, 1);hi = oshape[d] - size
                if d in sd2od:
                    if clip:
                        c = self.k.var('cl');self.k.emit(f'const int64_t {c} = {starts[d]} < 0 ? 0 : ({starts[d]} > {hi} ? {hi} : {starts[d]});');starts[d] = c
                    else:conds.append(f'({starts[d]} >= 0 && {starts[d]} <= {hi})')
            wc = {d: full[uwd[j]] for j, d in enumerate(window)}
            coord = [f'({starts[d]} + {wc[d]})' if d in wc else f'({starts[d]})' for d in range(len(oshape))]
            if op.lane:
                # A write into another lane (in range, or clamped into range) cannot be reproduced per lane: flag it.
                other = f'({coord[0]}) != b' + ('' if clip else f' && ({coord[0]}) >= 0 && ({coord[0]}) < p.lanes')
                self.k.emit(f'if ({other}) atomicOr(err, 8u);')
                conds.append(f'({coord[0]} == b)');coord[0] = 'b'
            f = self.k.var('f');self.k.emit(f'const int64_t {f} = {self.flat(op, coord)};')
            value = self.get(upd, u)
            return ' && '.join(conds) or 'true', f, value
        return target

    def scatter(self, ins, outs, params, dying, combine, atomic=None):
        """Exact sequential scatter: updates of one target element apply in update order."""
        op, idx, upd = ins;out = outs[0];U = upd.size;n = out.size
        target = self.scatter_target(ins, params)
        c = ctype(out.dtype)
        if n <= REG:
            # Register target: every update in turn; the owner thread applies it.
            self.new_reg(out);name = out.holder.name;i = self.k.var('i')
            self.k.emit(f'const int {i} = {thread_index(n)};')
            self.k.emit(f'{c} {name} = {self.get(op, i, op.dense)};')
            u = self.k.var('u')
            self.k.open(f'for (int {u} = 0; {u} < {U}; ++{u})')
            guard, f, value = target(u)
            new = value if combine is None else combine(name, value)
            self.k.emit(f'if (({guard}) && (int64_t){i} == {f}) {name} = {new};')
            self.k.close()
            return [out]
        shared = sum(1 for v in ins if v.src == 'store' and v.holder is op.holder)
        if id(op) in dying and op.dense and op.kind == 'arena' and op.holder.words == n and shared == 1:
            self.take_over(op, out)
        else:
            self.produce(out, lambda i, own: self.get(op, i))
        off = out.holder.off
        base, u = self.k.var('base'), self.k.var('u')
        self.k.open(f'for (int {base} = 0; {base} < {U}; {base} += 32)')
        self.k.emit(f'const int {u} = {base} + t < {U} ? {base} + t : {U - 1};')
        guard, f, value = target(u)
        g = self.k.var('g')
        self.k.emit(f'const bool {g} = {base} + t < {U} && ({guard});')
        if atomic is not None:
            # Order-independent combiner: apply concurrently.
            self.k.emit(f'if ({g}) {atomic(f"A + {off} + {f}", value)}')
        else:
            key, peers = self.k.var('key'), self.k.var('peers')
            self.k.emit(f'const unsigned long long {key} = {g} ? (unsigned long long){f} : (0xffffffff00000000ull | (unsigned)t);')
            self.k.emit(f'const unsigned {peers} = __match_any_sync(v2e::kFull, {key});')
            if combine is None:
                # Overwrite: the last update of each target in this group of 32 wins.
                self.k.emit(f'if ({g} && 31 - __clz((int){peers}) == t) A[{off} + {f}] = {to_word(out.dtype, value)};')
            else:
                rank, rounds, k = self.k.var('rank'), self.k.var('rounds'), self.k.var('k')
                self.k.emit(f'const int {rank} = __popc({peers} & v2e::lanes_below(t));')
                self.k.emit(f'const int {rounds} = (int)__reduce_max_sync(v2e::kFull, {g} ? (unsigned)__popc({peers}) : 0u);')
                self.k.open(f'for (int {k} = 0; {k} < {rounds}; ++{k})')
                cur = word_to(out.dtype, f'A[{off} + {f}]')
                self.k.emit(f'if ({g} && {rank} == {k}) A[{off} + {f}] = {to_word(out.dtype, combine(cur, value))};')
                self.k.emit('__syncwarp();')
                self.k.close()
        self.k.emit('__syncwarp();')
        self.k.close()
        return [out]

    def p_scatter(self, e, ins, outs, params, dying):
        jx = params.get('update_jaxpr')
        if jx is not None:
            jx = getattr(jx, 'jaxpr', jx)
            if jx.eqns or len(jx.invars) != 2 or list(jx.outvars) != [jx.invars[1]]:
                raise Unsupported('scatter with a custom update function (.at[].apply)')
        return self.scatter(ins, outs, params, dying, None)

    def atomic(self, which, t):
        t = np.dtype(t)
        if t == np.dtype(np.int32):return lambda ptr, u: f'atomic{which}(reinterpret_cast<int*>({ptr}), (int)({u}));'
        if t == np.dtype(np.uint32):return lambda ptr, u: f'atomic{which}(reinterpret_cast<unsigned int*>({ptr}), (unsigned int)({u}));'
        if t == BOOL and which in ('Min', 'Max'):
            return lambda ptr, u: (f'atomicAnd({ptr}, (uint32_t)({u}));' if which == 'Min' else f'atomicOr({ptr}, (uint32_t)({u}));')
        return None

    def float_combiner_guard(self, params, ins, what):
        if ins[0].dtype == F32 and not params.get('unique_indices') and ins[2].size > 1:
            raise Unsupported(f'f32 {what} with possibly duplicate indices (XLA applies duplicates in atomic or tree order)')

    def p_scatter_add(self, e, ins, outs, params, dying):
        t = ins[0].dtype
        self.float_combiner_guard(params, ins, 'scatter_add')
        # Word atomics only for full-width integers (narrow words must stay sign/zero-extended).
        free = (lambda ptr, u: f'atomicAdd({ptr}, (uint32_t)({u}));') if t in SIGNED | UNSIGNED and t.itemsize == 4 else None
        return self.scatter(ins, outs, params, dying, lambda a, u: binary('add', t, a, u), free)

    def p_scatter_min(self, e, ins, outs, params, dying):
        t = ins[0].dtype
        return self.scatter(ins, outs, params, dying, lambda a, u: binary('min', t, a, u), self.atomic('Min', t))

    def p_scatter_max(self, e, ins, outs, params, dying):
        t = ins[0].dtype
        return self.scatter(ins, outs, params, dying, lambda a, u: binary('max', t, a, u), self.atomic('Max', t))

    def p_scatter_mul(self, e, ins, outs, params, dying):
        t = ins[0].dtype
        self.float_combiner_guard(params, ins, 'scatter_mul')
        return self.scatter(ins, outs, params, dying, lambda a, u: binary('mul', t, a, u))

    # -- sort ------------------------------------------------------------------------------
    @staticmethod
    def stability_guard(params, ins):
        """Both backends sort stably; an unstable XLA sort orders ties arbitrarily, which is only
        unobservable when the sole operand is an integer key (equal keys are identical bits)."""
        if not params.get('is_stable', True) and (len(ins) > 1 or ins[0].dtype == F32):
            raise Unsupported('unstable sort with observable tie order')

    def p_sort(self, e, ins, outs, params, dying):
        dim = int(params['dimension']);nk = int(params['num_keys']);src = ins[0]
        if src.lane and dim == 0:raise Unsupported('sort over lanes')
        self.stability_guard(params, ins)
        ins = [self.dense_arena(v) for v in ins]
        off = 1 if src.lane else 0;ist = strides(src.lshape);n = src.shape[dim]
        keep = [d for d in range(len(src.shape)) if d != dim and not (src.lane and d == 0)]
        keep_shape = [src.shape[d] for d in keep];rows = int(math.prod(keep_shape));step = ist[dim - off]
        stage = [self.new_arena(Val(o.dtype, o.shape, 'store')) for o in outs]
        self.k.smem = max(self.k.smem, 2 * n + 32)
        row = self.k.var('row')
        self.k.open(f'for (int {row} = 0; {row} < {rows}; ++{row})')
        rc = self.k.coords(row, keep_shape)
        base = ' + '.join(f'({rc[j]}) * {ist[d - off]}' for j, d in enumerate(keep)) or '0'
        keys = ', '.join(f'v2e::SortKey{{A + {v.holder.off} + ({base}), {step}, {sort_kind(v.dtype)}}}' for v in ins[:nk])
        kv = self.k.var('keys')
        self.k.emit(f'const v2e::SortKey {kv}[{nk}] = {{{keys}}};')
        self.k.emit(f'v2e::radix_argsort<{nk}>({kv}, {n}, S, S + {n}, S + {2 * n}, t);')
        z = self.k.var('z')
        self.k.open(f'for (int {z} = t; {z} < {n}; {z} += 32)')
        self.k.emit(f'const int from = (int)S[{z}];')
        for v, o in zip(ins, stage):
            self.k.emit(f'A[{o.holder.off} + ({base}) + ({z}) * {step}] = A[{v.holder.off} + ({base}) + from * {step}];')
        self.k.close()
        self.k.emit('__syncwarp();')
        self.k.close()
        self.release_temps()
        res = []
        for o, s in zip(outs, stage):
            if o.size <= REG:
                # Small results go to registers.
                self.produce(o, lambda i, own, s=s: self.get(s, i));self.free_holder(s.holder)
                res.append(o)
            else:
                res.append(self.view(s, o))
        return res

    # -- PRNG (threefry2x32, partitionable counters: element i uses counter (0, i)) ----------------
    def p_random_wrap(self, e, ins, outs, params, dying):
        return [self.view(ins[0], outs[0])]

    def p_random_unwrap(self, e, ins, outs, params, dying):
        return [self.view(ins[0], outs[0])]

    @staticmethod
    def partitionable_guard():
        import jax
        if not jax.config.jax_threefry_partitionable:raise Unsupported('PRNG counters assume jax_threefry_partitionable=True')

    def key_words(self, keys, kc):
        self.partitionable_guard()
        return self.get(keys, self.flat(keys, kc + ['0'])), self.get(keys, self.flat(keys, kc + ['1']))

    def p_random_bits(self, e, ins, outs, params, dying):
        keys, out = ins[0], outs[0];shape = tuple(params['shape'])
        if int(params['bit_width']) != 32:raise Unsupported('random_bits width')
        nk = len(keys.shape) - 1
        def expr(i, own):
            full = self.full_coords(out, i);kc, sc = full[:nk], full[nk:]
            k0, k1 = self.key_words(keys, kc)
            cnt = ' + '.join(f'({c}) * {st}' for c, st, n in zip(sc, strides(shape), shape) if n != 1) or '0'
            return f'v2e::threefry_xor({k0}, {k1}, 0u, (uint32_t)({cnt}))'
        return [self.produce(out, expr)]

    def p_random_split(self, e, ins, outs, params, dying):
        keys, out = ins[0], outs[0];shape = tuple(params['shape']);nk = len(keys.shape) - 1
        def expr(i, own):
            full = self.full_coords(out, i);kc, sc, w = full[:nk], full[nk:-1], full[-1]
            k0, k1 = self.key_words(keys, kc)
            cnt = ' + '.join(f'({c}) * {st}' for c, st, n in zip(sc, strides(shape), shape) if n != 1) or '0'
            return f'v2e::threefry_word({k0}, {k1}, 0u, (uint32_t)({cnt}), (int)({w}))'
        return [self.produce(out, expr)]

    def p_random_fold_in(self, e, ins, outs, params, dying):
        keys, msgs = ins;out = outs[0];nk = len(keys.shape) - 1
        if msgs.shape not in (tuple(keys.shape[:-1]), ()):raise Unsupported('random_fold_in broadcasting')
        def expr(i, own):
            full = self.full_coords(out, i);kc, w = full[:nk], full[-1]
            k0, k1 = self.key_words(keys, kc)
            m = self.get(msgs, self.flat(msgs, kc) if msgs.shape else '0')
            return f'v2e::threefry_word({k0}, {k1}, 0u, (uint32_t)({m}), (int)({w}))'
        return [self.produce(out, expr)]

    # -- calls ------------------------------------------------------------------------------
    def call(self, closed, ins):
        consts = [self.const(c) for c in closed.consts]
        res = self.run_with_consts(closed.jaxpr, consts, ins)
        for r in res:
            if r.src == 'store':r.holder.refs -= 1  # consume the inner outvar reference
        return res

    def run_with_consts(self, jaxpr, consts, ins):
        from jax.extend import core as jcore
        if jaxpr.constvars:
            inner = jcore.Jaxpr(constvars=(), invars=list(jaxpr.constvars) + list(jaxpr.invars), outvars=jaxpr.outvars,
                eqns=jaxpr.eqns, effects=jaxpr.effects)
            return self.run(inner, list(consts) + list(ins))
        return self.run(jaxpr, ins)

    def p_jit(self, e, ins, outs, params, dying):
        return self.call(params['jaxpr'], ins)

    p_pjit = p_jit

    def p_closed_call(self, e, ins, outs, params, dying):
        return self.call(params['call_jaxpr'], ins)

    def p_custom_jvp_call(self, e, ins, outs, params, dying):
        return self.call(params['call_jaxpr'], ins)

    def p_custom_vjp_call(self, e, ins, outs, params, dying):
        return self.call(params['call_jaxpr'], ins)

    # -- constants ---------------------------------------------------------------------------
    def const(self, value):
        arr = np.asarray(value)
        if arr.ndim == 0:return Val(np.dtype(arr.dtype), (), 'lit', lit=arr[()] if arr.dtype == np.float32 else arr.item(), cderived=True)
        if LANES in arr.shape:raise Unsupported(f'constant with lane-sized axis {arr.shape}')
        for k, c in enumerate(self.k.consts):
            if c is value or (c.shape == arr.shape and c.dtype == arr.dtype and np.array_equal(c, arr) and
                    np.array_equal(np.asarray(c).view(np.uint8), arr.view(np.uint8))):
                return Val(np.dtype(arr.dtype), arr.shape, 'const', ptr=f'cs{k}', arr=c, cderived=True)
        self.k.consts.append(arr)
        return Val(np.dtype(arr.dtype), arr.shape, 'const', ptr=f'cs{len(self.k.consts) - 1}', arr=arr, cderived=True)


class ThreadGenerator(Generator):
    """One thread per lane, for scalar-heavy modules (the engine step: ~90% of its values are per-lane scalars).

    Values of at most REG elements are register arrays filled by fully unrolled loops (constant
    indices, so nvcc keeps them in registers); data-dependent reads and writes of them are unrolled
    selects. Larger values live in a lane-interleaved global arena (element i of lane b at
    arena[i * lanes + b], coalesced across a warp) and are processed by per-thread loops. Every
    algorithm is the plain sequential one, so scatters, scans and sorts are exact by construction.
    """
    THREADS = 32   # lanes per block: one warp per block spreads small batches over SMs

    def __init__(self, name, profile=False):
        super().__init__(name, profile)
        self.collective = False;self.chunk_id = 0;self.eq_start = 0;self.reloads = 0
        self.k.CHUNK_LINES = int(os.environ.get('V2E_CHUNK_LINES') or 5000)

    def mark_cond(self):
        return 'b == 0'

    # Small values keep a home (arena slot or input buffer) once they leave registers; a chunk
    # reloads one into registers at its first use there (values are immutable, so a homed value
    # is never stored twice).
    def new_reg(self, out):
        super().new_reg(out);out.holder.chunk = self.chunk_id
        return out

    def input_value(self, v):
        if v.size > REG:return v
        ptr, size, t = v.ptr, v.size, v.dtype
        h = Holder('arena', size, t, home=lambda q: f'{ptr}[(size_t)b * {size} + {q}]')
        out = Val(t, v.shape, 'store', holder=h);self.live[h] = None
        return out

    def free_holder(self, h):
        if h.off >= 0:self.k.release(h.off, h.words);h.off = -1
        self.live.pop(h, None);self.release_deps(h)

    def dense_arena(self, v):
        return v if (v.dense and v.kind == 'arena' and v.holder.off >= 0) else self.temp_arena(v)

    def reload(self, h):
        """Load a homed small value into registers at the start of the current equation."""
        c = ctype(h.dtype);name = self.k.var('r')
        lines = [f'{c} {name};' if h.words == 1 else f'{c} {name}[{h.words}];']
        lines += [f'{name} = {h.home(0)};'] if h.words == 1 else [f'{name}[{q}] = {h.home(q)};' for q in range(h.words)]
        for line in lines:
            self.k.lines.insert(self.eq_start, '  ' + line);self.eq_start += 1
        h.kind = 'reg';h.name = name;h.chunk = self.chunk_id;self.reloads += 1

    def maybe_split(self):
        if len(self.k.lines) >= self.k.CHUNK_LINES and self.k.indent == 1:
            self.spill_live()
            self.k.chunks.append(self.k.lines);self.k.chunk_smem.append(0);self.k.lines = [];self.chunk_id += 1

    def arena_ref(self, off, idx):
        return f'Ab[(size_t)({off} + ({idx})) * L]'

    def get(self, v, idx, own=False, dyn=False):
        if v.imap is not None:idx = v.imap(idx)
        kind = v.kind
        if kind == 'lit':return literal(v.lit, v.dtype)
        if kind == 'expr':return v.fn(idx)
        if kind == 'in':return f'{v.ptr}[(size_t)b * {v.stride} + ({idx})]'
        if kind == 'const':return f'{v.ptr}[{idx}]'
        h = v.holder
        if (kind == 'arena' or h.chunk != self.chunk_id) and h.home is not None and h.words <= REG:self.reload(h)
        elif kind == 'arena':return word_to(v.dtype, self.arena_ref(h.off, idx))
        if h.words == 1:return h.name
        if dyn or self.rolled:return f'v2e::pick({h.name}, (int64_t)({idx}))'
        return f'{h.name}[{idx}]'

    def produce(self, out, expr):
        n = out.size;c = ctype(out.dtype)
        if n <= REG:
            self.new_reg(out);name = out.holder.name
            if n == 1:
                self.k.emit(f'const {c} {name} = {expr("0", True)};')
                return out
            i = self.k.var('i')
            self.k.emit(f'{c} {name}[{n}];')
            self.loop_open(i, n)
            self.k.emit(f'{name}[{i}] = {expr(i, True)};')
            self.loop_close(n)
            return out
        self.new_arena(out);i = self.k.var('i')
        self.loop_open(i, n)
        e = expr(i, False)
        self.k.emit(f'{self.arena_ref(out.holder.off, i)} = {to_word(out.dtype, e)};')
        self.loop_close(n)
        return out

    def spill_live(self):
        for h in list(self.live):
            if h.kind != 'reg' or h.refs <= 0:continue
            if h.home is None:
                off = self.k.alloc(h.words)
                if h.words == 1:self.k.emit(f'{self.arena_ref(off, 0)} = {to_word(h.dtype, h.name)};')
                else:
                    for q in range(h.words):self.k.emit(f'{self.arena_ref(off, q)} = {to_word(h.dtype, f"{h.name}[{q}]")};')
                h.off = off;h.home = (lambda q, off=off, t=h.dtype: word_to(t, self.arena_ref(off, q)))
            h.kind = 'arena'

    def temp_arena(self, v):
        h = Holder('arena', max(1, v.size), v.dtype, off=self.k.alloc(v.size))
        tmp = Val(v.dtype, v.shape, 'store', holder=h);self.live[h] = None;h.refs = 1;self.temps.append(h)
        i = self.k.var('i')
        self.loop_open(i, v.size)
        self.k.emit(f'{self.arena_ref(h.off, i)} = {to_word(v.dtype, self.get(v, i))};')
        self.loop_close(v.size)
        return tmp

    def emit_output(self, v):
        size = v.size
        if v.kind == 'reg' and v.dense:
            if size == 1:self.k.emit(f'o[(size_t)b] = {v.holder.name};')
            else:
                for q in range(size):self.k.emit(f'o[(size_t)b * {size} + {q}] = {v.holder.name}[{q}];')
            return
        i = self.k.var('i')
        self.loop_open(i, size)
        self.k.emit(f'o[(size_t)b * {size} + {i}] = {self.get(v, i)};')
        self.loop_close(size)

    def scatter(self, ins, outs, params, dying, combine, atomic=None):
        """Sequential scatter in update order (each thread owns its lane)."""
        op, idx, upd = ins;out = outs[0];U = upd.size;n = out.size;c = ctype(out.dtype)
        target = self.scatter_target(ins, params)
        if n <= REG:
            self.new_reg(out);name = out.holder.name
            if n == 1:self.k.emit(f'{c} {name} = {self.get(op, "0")};')
            else:
                k = self.k.var('k')
                self.k.emit(f'{c} {name}[{n}];')
                self.loop_open(k, n)
                self.k.emit(f'{name}[{k}] = {self.get(op, k)};')
                self.loop_close(n)
            u = self.k.var('u')
            self.loop_open(u, U)
            guard, f, value = target(u)
            x = self.k.var('x')
            self.k.emit(f'const {c} {x} = {value};')
            if n == 1:
                new = x if combine is None else combine(name, x)
                self.k.emit(f'if (({guard}) && {f} == 0) {name} = {new};')
            else:
                k = self.k.var('k')
                self.k.emit('#pragma unroll')
                self.k.open(f'for (int {k} = 0; {k} < {n}; ++{k})')
                new = x if combine is None else combine(f'{name}[{k}]', x)
                self.k.emit(f'if (({guard}) && {f} == {k}) {name}[{k}] = {new};')
                self.k.close()
            self.loop_close(U)
            return [out]
        shared = sum(1 for v in ins if v.src == 'store' and v.holder is op.holder)
        if id(op) in dying and op.dense and op.kind == 'arena' and op.holder.words == n and shared == 1:
            self.take_over(op, out)
        else:
            self.produce(out, lambda i, own: self.get(op, i))
        off = out.holder.off;u = self.k.var('u')
        self.loop_open(u, U)
        guard, f, value = target(u)
        cur = word_to(out.dtype, self.arena_ref(off, f))
        new = value if combine is None else combine(cur, value)
        self.k.emit(f'if ({guard}) {self.arena_ref(off, f)} = {to_word(out.dtype, new)};')
        self.loop_close(U)
        return [out]

    def p_sort(self, e, ins, outs, params, dying):
        dim = int(params['dimension']);nk = int(params['num_keys']);src = ins[0]
        if src.lane and dim == 0:raise Unsupported('sort over lanes')
        self.stability_guard(params, ins)
        ins = [self.dense_arena(v) for v in ins]
        off = 1 if src.lane else 0;ist = strides(src.lshape);n = src.shape[dim]
        keep = [d for d in range(len(src.shape)) if d != dim and not (src.lane and d == 0)]
        keep_shape = [src.shape[d] for d in keep];rows = int(math.prod(keep_shape));step = ist[dim - off]
        stage = [self.new_arena(Val(o.dtype, o.shape, 'store')) for o in outs]
        perm = Holder('arena', n, np.dtype(np.uint32), off=self.k.alloc(n));self.live[perm] = None;perm.refs = 1;self.temps.append(perm)
        row = self.k.var('row');self.rolled += 1
        self.k.open(f'for (int {row} = 0; {row} < {rows}; ++{row})')
        rc = self.k.coords(row, keep_shape)
        base = ' + '.join(f'({rc[j]}) * {ist[d - off]}' for j, d in enumerate(keep)) or '0'
        keys = ', '.join(f'v2e::SortKey{{Ab + (size_t)({v.holder.off} + ({base})) * L, (int64_t){step} * L, {sort_kind(v.dtype)}}}' for v in ins[:nk])
        kv = self.k.var('keys')
        self.k.emit(f'const v2e::SortKey {kv}[{nk}] = {{{keys}}};')
        self.k.emit(f'v2e::thread_argsort<{nk}>({kv}, {n}, Ab + (size_t){perm.off} * L, L);')
        z = self.k.var('z')
        self.k.open(f'for (int {z} = 0; {z} < {n}; ++{z})')
        self.k.emit(f'const int from = (int){self.arena_ref(perm.off, z)};')
        for v, o in zip(ins, stage):
            self.k.emit(f'{self.arena_ref(o.holder.off, f"({base}) + ({z}) * {step}")} = {self.arena_ref(v.holder.off, f"({base}) + from * {step}")};')
        self.k.close()
        self.k.close();self.rolled -= 1
        self.release_temps()
        res = []
        for o, st in zip(outs, stage):
            if o.size <= REG:
                self.produce(o, lambda i, own, st=st: self.get(st, i));self.free_holder(st.holder)
                res.append(o)
            else:
                res.append(self.view(st, o))
        return res

    def chunk_source(self, ns, c, lines, smem, decl):
        tpb = self.THREADS
        return f'''// Generated chunk {c} of `{self.k.name}`. Do not edit.
#include "{ns}.cuh"

namespace {ns} {{

// One thread per lane; small values are register arrays, large ones a lane-interleaved arena.
__global__ void __launch_bounds__({tpb}) chunk_{c}(const Params p) {{
  const int64_t b = (int64_t)blockIdx.x * {tpb} + threadIdx.x;
  if (b >= p.lanes) return;
  const int64_t L = p.lanes;
  uint32_t* __restrict__ Ab = p.arena + b;
  uint32_t* err = p.err;
  const long long T0 = clock64(); (void)T0;
  (void)Ab; (void)err; (void)L;
{chr(10).join(decl)}
{chr(10).join(lines)}
}}

void launch_{c}(const Params& p, cudaStream_t stream) {{
  chunk_{c}<<<static_cast<unsigned>((p.lanes + {tpb - 1}) / {tpb}), {tpb}, 0, stream>>>(p);
}}

}}  // namespace {ns}
'''


def generate(name, closed, input_avals, profile=False, mode='warp'):
    """CUDA sources for closed jaxpr over lane inputs (flattened avals with axis 0 = LANES).

    mode 'warp': one warp per lane (array-heavy modules); 'thread': one thread per lane (scalar-heavy modules).
    """
    versions = check_pinned_versions()
    g = (ThreadGenerator if mode == 'thread' else Generator)(name, profile);k = g.k
    ins = []
    for n, aval in enumerate(input_avals):
        if not (len(aval.shape) and aval.shape[0] == LANES):raise Unsupported(f'input {n} lacks the lane axis')
        v = Val(np.dtype(aval.dtype), tuple(aval.shape), 'in', ptr=f'in{n}')
        v.stride = v.size
        ins.append(g.input_value(v))
    outs = g.run_with_consts(closed.jaxpr, [g.const(c) for c in closed.consts], ins)
    g.eq_start = len(k.lines)
    for n, v in enumerate(outs):
        if not v.lane:raise Unsupported(f'output {n} has no lane axis')
        k.emit(f'{{ {ctype(v.dtype)}* o = reinterpret_cast<{ctype(v.dtype)}*>(p.out[{n}]);')
        g.emit_output(v)
        k.emit('}')
    chunks = k.chunks + ([k.lines] if k.lines else [])
    smems = k.chunk_smem + ([k.smem] if k.lines else [])
    # Constant buffers no generated code reads (intermediate fold results) are not passed to the kernels.
    import re
    pattern = re.compile(r'\bcs(\d+)\b')
    used = sorted({int(m) for lines in chunks for line in lines for m in pattern.findall(line)})
    if len(used) != len(k.consts):
        remap = {old: new for new, old in enumerate(used)}
        chunks = [[pattern.sub(lambda m: f'cs{remap[int(m.group(1))]}', line) for line in lines] for lines in chunks]
        k.consts = [k.consts[q] for q in used]
    decl = []
    for n, aval in enumerate(input_avals):
        decl.append(f'  const {ctype(aval.dtype)}* __restrict__ in{n} = reinterpret_cast<const {ctype(aval.dtype)}*>(p.in[{n}]);')
    for n, c in enumerate(k.consts):
        decl.append(f'  const {ctype(c.dtype)}* __restrict__ cs{n} = reinterpret_cast<const {ctype(c.dtype)}*>(p.in[{len(input_avals) + n}]);')
    words = max(1, k.peak);ns = f'v2e_gen_{name}'
    header = f'''// Generated by gpu_engine.v2e_j2c from the oracle jaxpr of `{name}`. Do not edit.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "v2e_jax.cuh"
#include "v2e_rng.cuh"

namespace {ns} {{

constexpr int kInputs = {len(input_avals) + len(k.consts)};
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
        files[f'{ns}_{c:03d}.cu'] = g.chunk_source(ns, c, lines, smems[c], decl)
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
    module = Module(name, files, words, [(np.dtype(a.dtype), tuple(a.shape)) for a in input_avals], list(k.consts),
        [(np.dtype(v.dtype), v.shape) for v in outs], k.lane_literals, g.marks)
    module.stats.update(versions=versions, chunks=len(chunks), cse_hits=g.cse_hits, folds=g.folds, consts=len(k.consts), lines=module.source_lines,
        mode=mode, reloads=getattr(g, 'reloads', 0))
    return module
