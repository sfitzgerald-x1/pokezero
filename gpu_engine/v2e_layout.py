"""Generated CUDA layout header for the option-A engine: gpu_engine/cuda/v2e_layout.cuh (design note section 3.3).

The header is generated from jax.tree_util.tree_flatten_with_path over three pytrees, so hand-written CUDA and JAX
cannot drift apart silently:
- (state, actions, trackers), the step inputs, with the same leaf paths as v2e_capture.leaf_paths: StateView,
  ActionsView and TrackerView structs, one typed Leaf<T, D...> member per leaf in flatten order (= FFI operand order;
  kStepOffset is each struct's first operand in the whole (state, actions, trackers) flatten order);
- the phase carries of gpu_engine/v2e_phases_spec.json (M0a's spec, contract pokezero.v2e-phases-spec.v1): one struct
  C_<boundary> per boundary, members in carry order (= the FFI operand order of the phase that starts there);
- the tables: transition.MoveTable (the engine tables), observation_tables.belief (the belief tables the hooks read,
  paths prefixed `belief`) and mechanics._ACCURACY_COUNTS exactly as Python computes it: one struct-of-arrays
  EngineTables over a single packed buffer with field names that match the pytree paths; pack_tables() builds that
  buffer on the host in the header's order and offsets.

Leaf<T, D...> is a per-lane row-major [B, D...] buffer with XLA-exact point access:
- get(b, i...)    gather: a negative index wraps once (i + dim), then clamps into [0, dim - 1];
- set(b, i..., v) scatter (FILL_OR_DROP): wrap once, drop the whole write if any index is still out of range, convert
  the update to the operand dtype exactly as JAX's _scatter_impl does (promote, scatter, convert back: integer wrap,
  float -> int toward zero saturating with NaN -> 0, anything -> bool as != 0); returns whether the write was kept;
- raw(b, i...)    unchecked reference, for proven-in-range fast paths;
- span(b, i...)   gather-clamped pointer to the sub-block below a partial index; writable(b, i...) asks whether a
  scatter at a partial index would be kept.
Under V2E_LAYOUT_DEBUG every Leaf and Table carries `uint32_t* dbg` (dbg[0] clamps, dbg[1] drops) the accessors count
into. emulate_get / emulate_set are the numpy reference of the same semantics; `probe` checks them and the device
accessors against jax.lax on a GPU for every dtype in use (negative, in-range and out-of-range indices).

Drift check: each struct has kDigest, the 64-bit prefix of sha256 over its (path, dtype, shape) lines, and kDigestHex;
digests(description, spec) returns the same values in Python, so an FFI handler compares its compiled kDigest with a
`layout_digest` attribute passed from Python and returns the error word kStaleLayout instead of touching the buffers.

Subcommands (paths are arguments; no cluster paths are built in):
  describe  dump the pytree description JSON from the real tables (GPU host; v2e_diff's loading);
  generate  header from a description JSON and a spec (CPU);
  check     regenerate and compare with the committed header (CPU);
  probe     GPU: header compiles under v2e_cuda.NVCC_FLAGS; accessors and the numpy emulation against jax.lax; the
            digest refusal through a real FFI handler. Writes probe.json.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np

SCHEMA = 'pokezero.v2e-layout-desc.v1'
SPEC_SCHEMA = 'pokezero.v2e-phases-spec.v1'
HEADER = Path(__file__).resolve().parent / 'cuda' / 'v2e_layout.cuh'
DEFAULT_SPEC = Path(__file__).resolve().parent / 'v2e_phases_spec.json'
DEFAULT_DESCRIPTION = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures' / 'v2e_layout_description.json'
ACCURACY_PATH = 'mechanics._ACCURACY_COUNTS'
TABLE_ALIGN = 16
STALE_LAYOUT = 0x4c41594f   # 'LAYO': the error word of a handler whose compiled layout digest differs from Python's
CTYPE = {'bool': 'bool', 'int8': 'int8_t', 'uint8': 'uint8_t', 'int16': 'int16_t', 'uint16': 'uint16_t', 'int32': 'int32_t',
    'uint32': 'uint32_t', 'float32': 'float'}
KEYWORDS = {'auto', 'bool', 'break', 'case', 'char', 'class', 'const', 'continue', 'default', 'delete', 'do', 'double', 'else', 'enum',
    'explicit', 'export', 'extern', 'false', 'float', 'for', 'friend', 'goto', 'if', 'inline', 'int', 'long', 'namespace', 'new',
    'operator', 'private', 'protected', 'public', 'register', 'return', 'short', 'signed', 'sizeof', 'static', 'struct', 'switch',
    'template', 'this', 'throw', 'true', 'try', 'typedef', 'typename', 'union', 'unsigned', 'using', 'virtual', 'void', 'volatile',
    'while', 'get', 'set', 'raw', 'span', 'bind', 'base', 'writable', 'lanes', 'dbg', 'p', 'offset', 'counters', 'lane'}


# ---------------------------------------------------------------------------
# Descriptions: (path, dtype, shape) per leaf.

def describe(tree, *, lane_axis=True):
    """[{path, dtype, shape}] over tree_flatten_with_path; with lane_axis the leading axis is dropped (per-lane shape)."""
    import jax
    out = []
    for path, x in jax.tree_util.tree_flatten_with_path(tree)[0]:
        shape = list(np.shape(x))
        if lane_axis:
            if not shape:raise ValueError(f'{jax.tree_util.keystr(path)}: a lane tree leaf needs a leading lane axis')
            shape = shape[1:]
        out.append(dict(path=jax.tree_util.keystr(path), dtype=str(np.dtype(x.dtype)), shape=[int(d) for d in shape]))
    return out


def description_from_context(ctx):
    """The full description of a loaded context (v2e_census.load_context): step inputs, step output, tables."""
    import jax
    from gpu_engine import mechanics, v2e_modules
    state, actions, trackers = ctx['template']
    fn = v2e_modules.step(ctx['engine_tables'], ctx['observation_tables'])
    out = jax.eval_shape(fn, state, actions, trackers)
    belief = getattr(ctx['observation_tables'], 'belief', None)
    tables = describe(ctx['engine_tables'], lane_axis=False)
    tables += [dict(e, path='belief' + e['path']) for e in (describe(belief, lane_axis=False) if belief is not None else [])]
    counts = np.asarray(mechanics._ACCURACY_COUNTS, np.int32)
    tables.append(dict(path=ACCURACY_PATH, dtype='int32', shape=list(counts.shape)))
    return dict(schema=SCHEMA, state=describe(state), actions=describe(actions), trackers=describe(trackers),
        step_output=describe(out), tables=tables)


def load_description(path):
    d = json.loads(Path(path).read_text())
    if d.get('schema') != SCHEMA:raise ValueError(f'{path}: not a {SCHEMA} description')
    return d


def load_spec(path):
    s = json.loads(Path(path).read_text())
    if s.get('schema') != SPEC_SCHEMA:raise ValueError(f'{path}: not a {SPEC_SCHEMA} spec')
    for b in s['boundaries']:
        names = [e['name'] for e in b['carry']]
        if len(set(names)) != len(names):raise ValueError(f"spec boundary {b['name']}: duplicate carry names")
        for e in b['carry']:
            if e['dtype'] not in CTYPE:raise ValueError(f"spec boundary {b['name']}: {e['name']} has unsupported dtype {e['dtype']}")
    return s


# ---------------------------------------------------------------------------
# Names and digests.

def identifier(path, taken=None):
    """C identifier for a pytree key path: .mon_features['turns_active'] -> mon_features_turns_active, state.hp -> state_hp."""
    s = re.sub(r"\['([^']*)'\]", r'_\1', path);s = re.sub(r'\["([^"]*)"\]', r'_\1', s);s = re.sub(r'\[(-?\d+)\]', r'_\1', s)
    s = re.sub(r'[^0-9A-Za-z_]', '_', s.replace('.', '_')).strip('_') or 'leaf'
    if s[0].isdigit():s = '_' + s
    if s in KEYWORDS:s += '_'
    if taken is not None:
        base = s;k = 2
        while s in taken:s = f'{base}_{k}';k += 1
        taken.add(s)
    return s


def identifiers(entries):
    taken = set()
    return [identifier(e['path'], taken) for e in entries]


def digest_lines(entries):
    return '\n'.join(f"{e['path']}\t{e['dtype']}\t{json.dumps([int(d) for d in e['shape']])}" for e in entries)


def digest_hex(entries):
    """sha256 over the (path, dtype, shape) lines of one struct's leaves, in order."""
    return hashlib.sha256(digest_lines(entries).encode()).hexdigest()


def digest64(entries):
    """The 64-bit prefix the header exposes as kDigest (compare with the layout_digest attribute)."""
    return int(digest_hex(entries)[:16], 16)


def structs(description, spec=None):
    """Ordered {struct name: [entries]} of every struct the header emits (lane structs, carries, tables)."""
    out = {'StateView': description['state'], 'ActionsView': description['actions'], 'TrackerView': description['trackers']}
    if spec is not None:
        for b in spec['boundaries']:
            out['C_' + b['name']] = [dict(path=e['name'], dtype=e['dtype'], shape=e['shape'], source=e.get('source'), leaf=e.get('path')) for e in b['carry']]
    out['EngineTables'] = description['tables']
    return out


def digests(description, spec=None):
    """{struct: (digest64, digest hex)} as compiled into the header."""
    return {name: (digest64(entries), digest_hex(entries)) for name, entries in structs(description, spec).items()}


def digest_attr(d64):
    """The `layout_digest` FFI attribute for a kDigest value: the same 64 bits as a numpy int64 (XLA FFI attributes have no
    unsigned 64-bit type); the handler compares static_cast<uint64_t>(attribute) with kDigest."""
    return np.int64(d64 - 2**64 if d64 >= 2**63 else d64)


# ---------------------------------------------------------------------------
# Table packing (host side).

def table_layout(entries):
    """([(entry, offset bytes, nbytes)], total bytes) of the packed table buffer: every array at a TABLE_ALIGN-aligned offset."""
    rows = [];offset = 0
    for e in entries:
        nbytes = int(np.prod(e['shape'], dtype=np.int64)) * np.dtype(e['dtype']).itemsize
        rows.append((e, offset, nbytes));offset += (nbytes + TABLE_ALIGN - 1) // TABLE_ALIGN * TABLE_ALIGN
    return rows, max(offset, TABLE_ALIGN)


def table_leaves(engine_tables, belief_tables=None):
    """[(path, numpy array)] of the tables in the header's order (engine tables, belief tables, the accuracy counts)."""
    import jax
    from gpu_engine import mechanics
    leaves = [(jax.tree_util.keystr(p), np.asarray(x)) for p, x in jax.tree_util.tree_flatten_with_path(engine_tables)[0]]
    if belief_tables is not None:
        leaves += [('belief' + jax.tree_util.keystr(p), np.asarray(x)) for p, x in jax.tree_util.tree_flatten_with_path(belief_tables)[0]]
    leaves.append((ACCURACY_PATH, np.asarray(mechanics._ACCURACY_COUNTS, np.int32)))
    return leaves


def pack_tables(description, engine_tables, belief_tables=None):
    """One uint8 buffer holding every table leaf at the header's offset (row-major, little-endian), checked against the
    description's (path, dtype, shape) so a stale header cannot be fed a different table."""
    entries = description['tables'];rows, total = table_layout(entries);leaves = table_leaves(engine_tables, belief_tables)
    if [p for p, _ in leaves] != [e['path'] for e in entries]:
        raise ValueError('table leaf paths differ from the layout description (regenerate the header)')
    buffer = np.zeros(total, np.uint8)
    for (e, offset, nbytes), (path, x) in zip(rows, leaves):
        if str(x.dtype) != e['dtype'] or list(x.shape) != list(e['shape']):
            raise ValueError(f'{path}: {x.dtype}{list(x.shape)} differs from the layout description {e["dtype"]}{e["shape"]}')
        x = np.ascontiguousarray(x.astype(x.dtype.newbyteorder('<'))) if x.dtype.byteorder == '>' else np.ascontiguousarray(x)
        buffer[offset:offset + nbytes] = np.frombuffer(x.tobytes(), np.uint8)
    return buffer


# ---------------------------------------------------------------------------
# numpy emulation of JAX's default indexing semantics (the reference the C++ accessors are tested against).

def gather_index(index, size):
    """XLA gather point index: wrap a negative index once, then clamp into [0, size - 1]."""
    index = np.asarray(index).astype(np.int64)
    index = np.where(index < 0, index + size, index)
    return np.clip(index, 0, size - 1)


def scatter_index(index, size):
    """(wrapped index, kept) of a FILL_OR_DROP scatter point index: wrap once, keep only if in range."""
    index = np.asarray(index).astype(np.int64)
    index = np.where(index < 0, index + size, index)
    return index, (index >= 0) & (index < size)


def convert(value, dtype):
    """lax.convert_element_type semantics on a numpy array: integer wrap, float -> int toward zero saturating with NaN -> 0,
    -> bool as != 0, bool/int -> float (round to nearest even)."""
    value = np.asarray(value);dtype = np.dtype(dtype)
    if value.dtype == dtype:return value
    if dtype == np.bool_:return value != 0
    if dtype.kind in 'iu':
        if value.dtype.kind == 'f':
            info = np.iinfo(dtype);v = np.where(np.isnan(value), 0.0, value.astype(np.float64))
            with np.errstate(invalid='ignore'):
                v = np.where(v >= float(info.max), float(info.max), np.where(v <= float(info.min), float(info.min), np.trunc(v)))
            return v.astype(dtype)
        if value.dtype == np.bool_:return value.astype(dtype)
        return value.astype(np.uint64 if value.dtype.kind == 'u' else np.int64).astype(dtype)   # two's complement wrap
    if dtype.kind == 'f':return value.astype(dtype)
    raise TypeError(f'unsupported conversion to {dtype}')


def emulate_get(x, *indices):
    """x[arange(B), *indices] with JAX's default gather semantics (indices: scalars or [B] integer arrays)."""
    x = np.asarray(x);b = x.shape[0];rows = np.arange(b)
    idx = tuple(gather_index(np.broadcast_to(np.asarray(i), (b,)), n) for i, n in zip(indices, x.shape[1:]))
    return x[(rows,) + idx]


def promoted_dtype(a, b):
    """jnp.promote_types with 64-bit results canonicalized to 32 bits (JAX with x64 disabled), as _scatter_impl promotes."""
    a = np.dtype(a);b = np.dtype(b)
    if a == b:return a
    p = np.promote_types(a, b)
    if p.itemsize == 8:p = np.dtype({'i': np.int32, 'u': np.uint32, 'f': np.float32}[p.kind])
    return p


def promotes(operand_dtype, update_dtype):
    """Whether x.at[].set(v) promotes the whole operand (jax _scatter_impl: promote_dtypes(x, y) differs from x's dtype).
    Then every element of x round-trips through the promoted type, which is lossy only for a float update into an
    integer operand with |x| > 2**24. The oracle has no such scatter (scatter-census), and Leaf::set converts the update
    alone."""
    return promoted_dtype(operand_dtype, update_dtype) != np.dtype(operand_dtype)


def emulate_set_point(x, indices, value):
    """Leaf::set semantics: drop out-of-range rows, convert the update alone to the operand dtype."""
    x = np.asarray(x);b = x.shape[0];out = x.copy();keep = np.ones(b, bool);idx = []
    for i, n in zip(indices, x.shape[1:len(indices) + 1]):
        wrapped, ok = scatter_index(np.broadcast_to(np.asarray(i), (b,)), n);keep &= ok;idx.append(wrapped)
    value = np.broadcast_to(convert(value, x.dtype), (b,) + x.shape[1 + len(indices):])
    rows = np.flatnonzero(keep)
    out[(rows,) + tuple(np.asarray(i)[rows] for i in idx)] = value[rows]
    return out


def emulate_set(x, indices, value):
    """x.at[arange(B), *indices].set(value) exactly as jax (_scatter_impl): promote operand and update to their common
    dtype, scatter with FILL_OR_DROP (drop out-of-range rows), convert the result back to the operand dtype. Equals
    emulate_set_point unless promotes(x.dtype, value.dtype)."""
    x = np.asarray(x);value = np.asarray(value);p = promoted_dtype(x.dtype, value.dtype)
    out = emulate_set_point(convert(x, p), indices, convert(value, p))
    return convert(out, x.dtype)


# ---------------------------------------------------------------------------
# Header emission.

PRELUDE = r'''// Generated by gpu_engine.v2e_layout from the oracle pytrees. Do not edit; regenerate with
//   python -m gpu_engine.v2e_layout generate --description <desc.json> --spec <v2e_phases_spec.json> --out <this file>
// and check with `python -m gpu_engine.v2e_layout check`.
//
// Per-lane buffers are row-major [B, ...] as the FFI operands are today; a Leaf<T, D...> gives XLA-exact point access:
//   get(b, i...)      gather:  a negative index wraps once (i + dim), then clamps into [0, dim - 1];
//   set(b, i..., v)   scatter: wraps once, drops the whole write if any index is out of range (returns false), converts
//                     v to T as JAX's _scatter_impl does (integer wrap; float -> int toward zero, saturating, NaN -> 0;
//                     -> bool as != 0);
//   raw(b, i...)      unchecked reference, for proven-in-range fast paths;
//   span(b, i...)     gather-clamped pointer to the sub-block below the given (partial) indices;
//   writable(b, i...) whether a scatter at the given (partial) indices is kept.
// Under V2E_LAYOUT_DEBUG every Leaf and Table carries `uint32_t* dbg` (dbg[0] clamps, dbg[1] drops) the accessors count
// into. Every struct has kDigest (64-bit prefix of sha256 over its path/dtype/shape lines) and kDigestHex; a
// hand-written FFI handler compares kDigest against the `layout_digest` attribute Python passes
// (gpu_engine.v2e_layout.digests) and returns kStaleLayout instead of touching the buffers when they differ.
#pragma once
#include <cstdint>
#include <cstddef>
#include <limits>
#include <type_traits>

#if defined(__CUDACC__)
#define V2E_LAYOUT_HD __host__ __device__ __forceinline__
#else
#define V2E_LAYOUT_HD inline
#endif

namespace v2e_layout {

constexpr uint32_t kStaleLayout = 0x%(stale)08xu;   // error word: the .so was built for another layout

V2E_LAYOUT_HD void count_event(uint32_t* w) {
#if defined(__CUDA_ARCH__)
  atomicAdd(w, 1u);
#else
  ++*w;
#endif
}

#if defined(V2E_LAYOUT_DEBUG)
#define V2E_LAYOUT_DBG_MEMBER uint32_t* dbg = nullptr;
#define V2E_LAYOUT_COUNT(dbg, slot) do { if (dbg) count_event((dbg) + (slot)); } while (0)
#else
#define V2E_LAYOUT_DBG_MEMBER
#define V2E_LAYOUT_COUNT(dbg, slot) do { (void)(dbg); } while (0)
#endif

// XLA gather point index: wrap once, clamp into [0, n - 1]. Debug slot 0 counts clamps.
V2E_LAYOUT_HD int64_t gather_index(int64_t i, int64_t n, uint32_t* dbg) {
  if (i < 0) i += n;
  if (i < 0) { V2E_LAYOUT_COUNT(dbg, 0); return 0; }
  if (i >= n) { V2E_LAYOUT_COUNT(dbg, 0); return n - 1; }
  return i;
}

// FILL_OR_DROP scatter point index: wrap once; false when still out of range (the caller drops the whole write).
V2E_LAYOUT_HD bool scatter_index(int64_t& i, int64_t n) {
  if (i < 0) i += n;
  return i >= 0 && i < n;
}

// lax.convert_element_type semantics for a scatter update (jax _scatter_impl: promote, scatter, convert back to T).
template <class T, class U>
V2E_LAYOUT_HD T convert(U v) {
  if constexpr (std::is_same<T, U>::value) {
    return v;
  } else if constexpr (std::is_same<T, bool>::value) {
    return v != U(0);
  } else if constexpr (std::is_integral<T>::value && std::is_floating_point<U>::value) {
    if (v != v) return T(0);
    const double hi = static_cast<double>(std::numeric_limits<T>::max());
    const double lo = static_cast<double>(std::numeric_limits<T>::min());
    const double d = static_cast<double>(v);
    if (d >= hi) return std::numeric_limits<T>::max();
    if (d <= lo) return std::numeric_limits<T>::min();
    return static_cast<T>(d);   // toward zero
  } else if constexpr (std::is_integral<T>::value && std::is_same<U, bool>::value) {
    return v ? T(1) : T(0);
  } else if constexpr (std::is_integral<T>::value && std::is_integral<U>::value) {
    using W = typename std::conditional<std::is_signed<U>::value, int64_t, uint64_t>::type;
    return static_cast<T>(static_cast<typename std::make_unsigned<T>::type>(static_cast<W>(v)));   // two's complement wrap
  } else if constexpr (std::is_floating_point<T>::value && std::is_same<U, bool>::value) {
    return v ? T(1) : T(0);
  } else if constexpr (std::is_floating_point<T>::value && std::is_integral<U>::value) {
#if defined(__CUDA_ARCH__)
    if constexpr (sizeof(U) >= 4 && std::is_signed<U>::value) return __int2float_rn(static_cast<int32_t>(v));
    else if constexpr (sizeof(U) >= 4) return __uint2float_rn(static_cast<uint32_t>(v));
    else return static_cast<T>(v);
#else
    return static_cast<T>(v);   // round to nearest even, as the device intrinsics
#endif
  } else {
    return static_cast<T>(v);
  }
}

// ---- per-lane leaves --------------------------------------------------------------------------
// Common storage: row-major [B, D...]; `p` points at lane 0; kPerLane elements per lane.
template <class T, int... D>
struct LeafBase {
  static constexpr int kRank = sizeof...(D);
  static constexpr int64_t kPerLane = (int64_t(1) * ... * int64_t(D));
  T* p = nullptr;
  int64_t lanes = 0;
  V2E_LAYOUT_DBG_MEMBER
  V2E_LAYOUT_HD T* lane(int64_t b) const { return p + b * kPerLane; }
  V2E_LAYOUT_HD uint32_t* counters() const {
#if defined(V2E_LAYOUT_DEBUG)
    return dbg;
#else
    return nullptr;
#endif
  }
};

template <class T, int... D> struct Leaf;

%(leaf_specializations)s
// ---- tables -----------------------------------------------------------------------------------
// A table array (no lane axis), row-major, read through the same gather semantics; constant for every lane.
template <class T, int... D>
struct Table {
  static constexpr int kRank = sizeof...(D);
  static constexpr int64_t kSize = (int64_t(1) * ... * int64_t(D));
  const T* p = nullptr;
  V2E_LAYOUT_DBG_MEMBER
  V2E_LAYOUT_HD uint32_t* counters() const {
#if defined(V2E_LAYOUT_DEBUG)
    return dbg;
#else
    return nullptr;
#endif
  }
%(table_methods)s};
'''

STRUCT_TEMPLATE = '''
// ---- %(name)s: %(doc)s
struct %(name)s {
  static constexpr uint64_t kDigest = 0x%(digest)016xull;
  static constexpr const char* kDigestHex = "%(digest_hex)s";
  static constexpr int kLeaves = %(count)d;
%(members)s
%(args)s
  // Bind from an array of kLeaves buffer pointers in leaf order (the FFI operands), every leaf with `lanes` rows.
  V2E_LAYOUT_HD static %(name)s bind(void* const* bufs, int64_t lanes, uint32_t* dbg = nullptr) {
    %(name)s v;
%(binds)s
#if defined(V2E_LAYOUT_DEBUG)
%(dbg_binds)s
#else
    (void)dbg;
#endif
    return v;
  }
};
'''

TABLES_TEMPLATE = '''
// ---- EngineTables: struct-of-arrays over one packed buffer (gpu_engine.v2e_layout.pack_tables builds it, in this order).
struct EngineTables {
  static constexpr uint64_t kDigest = 0x%(digest)016xull;
  static constexpr const char* kDigestHex = "%(digest_hex)s";
  static constexpr int kLeaves = %(count)d;
  static constexpr int64_t kBytes = %(total)d;
  const unsigned char* base = nullptr;
  V2E_LAYOUT_DBG_MEMBER
%(offsets)s
%(methods)s
  V2E_LAYOUT_HD static EngineTables bind(const void* buffer, uint32_t* dbg = nullptr) {
    EngineTables t;
    t.base = static_cast<const unsigned char*>(buffer);
#if defined(V2E_LAYOUT_DEBUG)
    t.dbg = dbg;
#else
    (void)dbg;
#endif
    return t;
  }
};
'''


def _flat(idx, dims):
    """C expression of the row-major offset of a full index within one lane."""
    terms = []
    for k, i in enumerate(idx):
        terms.append(i + ''.join(f' * {d}' for d in dims[k + 1:]))
    return ' + '.join(terms) if terms else '0'


def _leaf_specialization(rank):
    """The Leaf<T, D0..D{rank-1}> partial specialization: get/set/raw/span/writable with the index count fixed."""
    dims = [f'D{k}' for k in range(rank)];idx = [f'i{k}' for k in range(rank)]
    tparams = ', '.join(['class T'] + [f'int {d}' for d in dims]);targs = ', '.join(['T'] + dims)
    sig = ', '.join(['int64_t b'] + [f'int64_t {i}' for i in idx])
    out = [f'template <{tparams}>', f'struct Leaf<{targs}> : LeafBase<{targs}> {{', f'  using Base = LeafBase<{targs}>;',
        '  using Base::p; using Base::lanes; using Base::kPerLane; using Base::kRank;', '  using value_type = T;']
    if rank:out.append(f'  static constexpr int kDims[{rank}] = {{{", ".join(dims)}}};')
    out.append(f'  V2E_LAYOUT_HD static int64_t offset({sig}) {{ return b * kPerLane + {_flat(idx, dims)}; }}')
    out.append(f'  V2E_LAYOUT_HD T& raw({sig}) const {{ return p[offset({", ".join(["b"] + idx)})]; }}')
    clamped = [f'gather_index({i}, {d}, this->counters())' for i, d in zip(idx, dims)]
    out.append(f'  V2E_LAYOUT_HD T get({sig}) const {{ return p[offset({", ".join(["b"] + clamped)})]; }}')
    out.append('  template <class U>')
    out.append(f'  V2E_LAYOUT_HD bool set({sig}, U v) const {{')
    if rank:
        out.append('    bool ok = true;')
        out += [f'    ok = scatter_index({i}, {d}) && ok;' for i, d in zip(idx, dims)]
        out.append('    if (!ok) { V2E_LAYOUT_COUNT(this->counters(), 1); return false; }')
    out.append(f'    p[offset({", ".join(["b"] + idx)})] = convert<T>(v);')
    out.append('    return true;')
    out.append('  }')
    for m in range(rank):   # partial indices (the full index is raw/get/set)
        psig = ', '.join(['int64_t b'] + [f'int64_t {i}' for i in idx[:m]]);zeros = ['0'] * (rank - m)
        pclamped = [f'gather_index({i}, {d}, this->counters())' for i, d in zip(idx[:m], dims[:m])]
        out.append(f'  V2E_LAYOUT_HD T* span({psig}) const {{ return p + offset({", ".join(["b"] + pclamped + zeros)}); }}')
        wbody = ' && '.join([f'scatter_index({i}, {d})' for i, d in zip(idx[:m], dims[:m])]) or 'true'
        out.append(f'  V2E_LAYOUT_HD bool writable({psig}) const {{ (void)b; return {wbody}; }}')
    out.append('};')
    return '\n'.join(out) + '\n'


def _table_methods(ranks):
    out = []
    for rank in sorted(ranks):
        idx = [f'i{k}' for k in range(rank)]
        sig = ', '.join(f'int64_t {i}' for i in idx)
        if rank == 0:
            out.append('  template <int R = kRank, typename std::enable_if<R == 0, int>::type = 0>')
            out.append('  V2E_LAYOUT_HD T get() const { return p[0]; }')
            out.append('  template <int R = kRank, typename std::enable_if<R == 0, int>::type = 0>')
            out.append('  V2E_LAYOUT_HD T raw() const { return p[0]; }')
            continue
        pre = f'  template <int R = kRank, typename std::enable_if<R == {rank}, int>::type = 0>'
        dims = [f'dims[{k}]' for k in range(rank)]
        out += [pre, f'  V2E_LAYOUT_HD T get({sig}) const {{', '    constexpr int dims[kRank] = {D...};']
        out += [f'    {i} = gather_index({i}, dims[{k}], this->counters());' for k, i in enumerate(idx)]
        out += [f'    return p[{_flat(idx, dims)}];', '  }']
        out += [pre, f'  V2E_LAYOUT_HD T raw({sig}) const {{', '    constexpr int dims[kRank] = {D...};', f'    return p[{_flat(idx, dims)}];', '  }']
    return '\n'.join(out) + '\n'


def _struct(name, entries, doc, *, offset_name=None, offset=None):
    idents = identifiers(entries);members = [];args = [];binds = [];dbg = []
    for k, (e, ident) in enumerate(zip(entries, idents)):
        ct = CTYPE[e['dtype']];shape = [int(d) for d in e['shape']];targs = ', '.join([ct] + [str(d) for d in shape])
        extra = f" (source {e['source']}, {e['leaf']!r})" if e.get('source') else ''
        members.append(f"  Leaf<{targs}> {ident};   // arg {k}: {e['path']} {e['dtype']}{shape}{extra}")
        args.append(f'  static constexpr int arg_{ident} = {k};')
        binds.append(f'    v.{ident}.p = static_cast<{ct}*>(bufs[{k}]); v.{ident}.lanes = lanes;')
        dbg.append(f'    v.{ident}.dbg = dbg;')
    if offset_name is not None:
        args.insert(0, f'  static constexpr int {offset_name} = {offset};   // first operand of this struct in the (state, actions, trackers) flatten order')
    return STRUCT_TEMPLATE % dict(name=name, doc=doc, digest=digest64(entries), digest_hex=digest_hex(entries), count=len(entries),
        members='\n'.join(members), args='\n'.join(args), binds='\n'.join(binds), dbg_binds='\n'.join(dbg))


def _tables(entries):
    rows, total = table_layout(entries);idents = identifiers(entries);offsets = [];methods = []
    for (e, offset, nbytes), ident in zip(rows, idents):
        ct = CTYPE[e['dtype']];shape = [int(d) for d in e['shape']];targs = ', '.join([ct] + [str(d) for d in shape])
        offsets.append(f"  static constexpr int64_t off_{ident} = {offset};   // {e['path']} {e['dtype']}{shape}, {nbytes} bytes")
        methods.append(f'  V2E_LAYOUT_HD Table<{targs}> {ident}() const {{\n    Table<{targs}> t; t.p = reinterpret_cast<const {ct}*>(base + off_{ident});'
            '\n#if defined(V2E_LAYOUT_DEBUG)\n    t.dbg = dbg;\n#endif\n    return t;\n  }')
    return TABLES_TEMPLATE % dict(digest=digest64(entries), digest_hex=digest_hex(entries), count=len(entries), total=total,
        offsets='\n'.join(offsets), methods='\n'.join(methods))


def lane_ranks(description, spec=None):
    ranks = {len(e['shape']) for key in ('state', 'actions', 'trackers') for e in description[key]}
    ranks |= {len(e['shape']) for b in (spec['boundaries'] if spec else []) for e in b['carry']}
    return sorted(ranks | {0, 1, 2, 3, 4})


def generate(description, spec=None):
    """The header text for a description (load_description) and an optional phases spec (load_spec)."""
    for name, entries in structs(description, spec).items():
        for e in entries:
            if e['dtype'] not in CTYPE:raise ValueError(f"{name}: {e['path']} has unsupported dtype {e['dtype']}")
    table_ranks = {len(e['shape']) for e in description['tables']} | {0, 1, 2}
    n_state = len(description['state']);n_actions = len(description['actions'])
    text = PRELUDE % dict(stale=STALE_LAYOUT, leaf_specializations='\n'.join(_leaf_specialization(r) for r in lane_ranks(description, spec)),
        table_methods=_table_methods(table_ranks))
    text += '\n// ---- step inputs: (state, actions, trackers) in v2e_capture.leaf_paths order; kStepOffset is each struct\'s first operand.\n'
    text += f'constexpr int kStepLeaves = {n_state + n_actions + len(description["trackers"])};\n'
    text += f'constexpr int kStepOutputLeaves = {len(description["step_output"])};\n'
    text += _struct('StateView', description['state'], 'transition.BattleState leaves (per lane)', offset_name='kStepOffset', offset=0)
    text += _struct('ActionsView', description['actions'], 'the actions array', offset_name='kStepOffset', offset=n_state)
    text += _struct('TrackerView', description['trackers'], 'engine_device_trackers.Trackers leaves (per lane); M1 hooks take this view',
        offset_name='kStepOffset', offset=n_state + n_actions)
    if spec is not None:
        text += f"\n// ---- phase carries (v2e_phases_spec.json, oracle digest {spec.get('oracle_digest')}): C_<boundary>, members in carry order = FFI operand order.\n"
        for b in spec['boundaries']:
            entries = structs(description, spec)['C_' + b['name']]
            text += _struct('C_' + b['name'], entries, f"carry at {b['name']} (line {b.get('line')}, occurrence {b.get('occurrence')}): {b.get('statement')}")
        phases = ', '.join(f"{p['name']} {p['from']}->{p['to']}" for p in spec.get('phases', []))
        text += f'// phases: {phases}\n'
    text += _tables(description['tables'])
    text += '\n}  // namespace v2e_layout\n'
    return text


# ---------------------------------------------------------------------------
# Probe sources: one host/device function applies get then set at a per-lane index; the GPU probe wraps it in FFI
# handlers (the shape every hand-written test handler takes: operands, a layout_digest attribute, an error word) and the
# CPU test compiles it with g++.

PROBE_COMMON = r'''// Probe of v2e_layout.cuh accessors (generated by gpu_engine.v2e_layout; the CPU test and the GPU probe share it).
#pragma once
#define V2E_LAYOUT_DEBUG 1
#include "v2e_layout.cuh"

namespace v2e_layout_probe {
using namespace v2e_layout;

// Lane b: got[b] = leaf.get(b, idx[b]) on the operand, then kept[b] = leaf.set(b, idx[b], v[b]) into the same operand.
template <class T, class U, int... D>
V2E_LAYOUT_HD void point(T* x, const int32_t* idx, const U* v, T* got, uint8_t* kept, int64_t b, int64_t lanes, uint32_t* dbg) {
  Leaf<T, D...> leaf;
  leaf.p = x; leaf.lanes = lanes; leaf.dbg = dbg;
  constexpr int R = sizeof...(D);
  const int32_t* i = idx + b * (R > 0 ? R : 1);
  if constexpr (R == 0) { got[b] = leaf.get(b); kept[b] = leaf.set(b, v[b]); }
  else if constexpr (R == 1) { got[b] = leaf.get(b, i[0]); kept[b] = leaf.set(b, i[0], v[b]); }
  else if constexpr (R == 2) { got[b] = leaf.get(b, i[0], i[1]); kept[b] = leaf.set(b, i[0], i[1], v[b]); }
  else if constexpr (R == 3) { got[b] = leaf.get(b, i[0], i[1], i[2]); kept[b] = leaf.set(b, i[0], i[1], i[2], v[b]); }
  else { got[b] = leaf.get(b, i[0], i[1], i[2], i[3]); kept[b] = leaf.set(b, i[0], i[1], i[2], i[3], v[b]); }
}

}  // namespace v2e_layout_probe
'''

PROBE_CUDA_HEAD = r'''// GPU probe of v2e_layout.cuh (generated by gpu_engine.v2e_layout). XLA FFI handlers in the shape of a hand-written
// test handler: operands, a `layout_digest` attribute checked against the compiled digest, an error word output.
#include <cstdint>
#include <cuda_runtime.h>
#include "xla/ffi/api/ffi.h"
#include "v2e_layout_probe_common.cuh"

namespace ffi = xla::ffi;
using namespace v2e_layout;
using namespace v2e_layout_probe;

template <class T, class U, int... D>
__global__ void point_kernel(T* x, const int32_t* idx, const U* v, T* got, uint8_t* kept, int64_t lanes, uint32_t* dbg) {
  const int64_t b = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (b >= lanes) return;
  point<T, U, D...>(x, idx, v, got, kept, b, lanes, dbg);
}

// Every probe handler: x [B, D...] T, idx [B, R] int32, v [B] U -> out [B, D...] T (x after set), got [B] T, kept [B] u8,
// dbg [2] u32 (clamps, drops), err [1] u32 (kStaleLayout when layout_digest differs from TrackerView::kDigest, else 0).
// layout_digest arrives as int64 (gpu_engine.v2e_layout.digest_attr): XLA FFI attributes have no unsigned 64-bit type.
template <class T, class U, int... D>
static ffi::Error PointImpl(cudaStream_t stream, int64_t layout_digest, ffi::AnyBuffer x, ffi::AnyBuffer idx, ffi::AnyBuffer v,
                            ffi::Result<ffi::AnyBuffer> out, ffi::Result<ffi::AnyBuffer> got, ffi::Result<ffi::AnyBuffer> kept,
                            ffi::Result<ffi::Buffer<ffi::U32>> dbg, ffi::Result<ffi::Buffer<ffi::U32>> err) {
  constexpr int64_t per_lane = LeafBase<T, D...>::kPerLane;
  const int64_t lanes = static_cast<int64_t>(x.element_count()) / per_lane;
  cudaMemsetAsync(dbg->typed_data(), 0, 2 * sizeof(uint32_t), stream);
  // The output is the operand (a hand-written handler aliases it in place; the probe copies it through).
  cudaMemcpyAsync(out->untyped_data(), x.untyped_data(), x.size_bytes(), cudaMemcpyDeviceToDevice, stream);
  if (static_cast<uint64_t>(layout_digest) != TrackerView::kDigest) {   // the attribute is the digest as int64 (digest_attr)
    const uint32_t stale = kStaleLayout;
    cudaMemcpyAsync(err->typed_data(), &stale, sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
    return ffi::Error::Success();   // refused: the operand is left as it was
  }
  cudaMemsetAsync(err->typed_data(), 0, sizeof(uint32_t), stream);
  const int threads = 128;
  point_kernel<T, U, D...><<<static_cast<unsigned>((lanes + threads - 1) / threads), threads, 0, stream>>>(
      static_cast<T*>(out->untyped_data()), static_cast<const int32_t*>(idx.untyped_data()), static_cast<const U*>(v.untyped_data()),
      static_cast<T*>(got->untyped_data()), static_cast<uint8_t*>(kept->untyped_data()), lanes, dbg->typed_data());
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(status));
  return ffi::Error::Success();
}

#define V2E_PROBE_HANDLER(NAME, ...)                                                                                   \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(NAME, (PointImpl<__VA_ARGS__>),                                                        \
                                ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Attr<int64_t>("layout_digest") \
                                    .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()                   \
                                    .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()                   \
                                    .Ret<ffi::Buffer<ffi::U32>>().Ret<ffi::Buffer<ffi::U32>>(),                          \
                                {xla::ffi::Traits::kCmdBufferCompatible})

'''

PROBE_HOST = r"""// Host probe of v2e_layout.cuh (generated by gpu_engine.v2e_layout; compiled with g++/clang++ -x c++ in the CPU test).
// One case per stdin line: "<T> <U> <dims> <B> x... idx... v..." (numbers as text, nan/inf allowed); prints
// "got... | kept... | out... | clamps drops".
#include "v2e_layout_probe_common.cuh"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
using namespace v2e_layout;
using namespace v2e_layout_probe;

template <class T> T parse(const std::string& s) { return static_cast<T>(std::strtod(s.c_str(), nullptr)); }
template <class T> void emit(std::ostream& out, T v) { out << static_cast<double>(v) << ' '; }

template <class T, class U, int... D>
void run_case(std::istream& in, std::ostream& out, int lanes) {
  constexpr int64_t per = LeafBase<T, D...>::kPerLane;
  constexpr int R = sizeof...(D);
  const int64_t nx = per * lanes, ni = lanes * (R > 0 ? R : 1);
  std::unique_ptr<T[]> x(new T[nx]());
  std::unique_ptr<int32_t[]> idx(new int32_t[ni]());
  std::unique_ptr<U[]> v(new U[lanes]());
  std::unique_ptr<T[]> got(new T[lanes]());
  std::unique_ptr<uint8_t[]> kept(new uint8_t[lanes]());
  uint32_t dbg[2] = {0u, 0u};
  std::string s;
  for (int64_t k = 0; k < nx; ++k) { in >> s; x[k] = parse<T>(s); }
  for (int64_t k = 0; k < ni; ++k) { in >> s; idx[k] = parse<int32_t>(s); }
  for (int64_t k = 0; k < lanes; ++k) { in >> s; v[k] = parse<U>(s); }
  for (int b = 0; b < lanes; ++b) point<T, U, D...>(x.get(), idx.get(), v.get(), got.get(), kept.get(), b, lanes, dbg);
  out.precision(17);
  for (int64_t k = 0; k < lanes; ++k) emit(out, got[k]);
  out << "| ";
  for (int64_t k = 0; k < lanes; ++k) out << static_cast<int>(kept[k]) << ' ';
  out << "| ";
  for (int64_t k = 0; k < nx; ++k) emit(out, x[k]);
  out << "| " << dbg[0] << ' ' << dbg[1] << "\n";
}

int main() {
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    std::istringstream in(line);
    std::string t, u, dims; int lanes;
    in >> t >> u >> dims >> lanes;
%(dispatch)s
    std::cerr << "unknown case " << t << ' ' << u << ' ' << dims << "\n";
    return 2;
  }
  return 0;
}
"""

PROBE_DIMS = {0: (), 1: (6,), 2: (2, 6), 3: (2, 6, 4), 4: (2, 6, 6, 2)}
UPDATE_DTYPES = ('float32', 'int32', 'bool')


def probe_cases(dtypes):
    """[(T, U, dims)] exercised by the probes: every dtype in use as the operand, updates of its own type and of each
    UPDATE_DTYPES type, ranks 0..4 (the per-lane ranks of the oracle pytrees)."""
    cases = []
    for t in dtypes:
        for u in dict.fromkeys((t,) + UPDATE_DTYPES):
            for rank, dims in PROBE_DIMS.items():cases.append((t, u, dims))
    return cases


def case_name(t, u, dims):
    return f'{t}_{u}_r{len(dims)}'


def _targs(t, u, dims):
    return ', '.join([CTYPE[t], CTYPE[u]] + [str(d) for d in dims])


def probe_cuda_source(cases):
    lines = [PROBE_CUDA_HEAD]
    for t, u, dims in cases:lines.append(f'V2E_PROBE_HANDLER(V2eLayoutProbe_{case_name(t, u, dims)}, {_targs(t, u, dims)});')
    return '\n'.join(lines) + '\n'


def probe_host_source(cases):
    dispatch = []
    for t, u, dims in cases:
        dims_key = 'x'.join(map(str, dims)) or 'scalar'
        dispatch.append(f'    if (t == "{t}" && u == "{u}" && dims == "{dims_key}") {{ run_case<{_targs(t, u, dims)}>(in, std::cout, lanes); continue; }}')
    return PROBE_HOST % dict(dispatch='\n'.join(dispatch))


def probe_inputs(t, u, dims, lanes, seed):
    """(x, idx, v) numpy test vectors: indices negative (wrapping and beyond), in range and past the end; updates with
    negatives, NaN, infinities, out-of-range magnitudes and fractions."""
    rng = np.random.default_rng(seed);shape = (lanes,) + tuple(dims)
    def values(dtype, size):
        dtype = np.dtype(dtype)
        if dtype == np.bool_:return rng.integers(0, 2, size).astype(bool)
        if dtype.kind == 'f':
            pool = np.array([0.0, -0.0, 1.0, -1.0, 2.7, -2.7, 0.5, -0.5, 1e10, -1e10, 3e9, -3e9, 65535.5, -32769.0, 255.9, np.nan, np.inf, -np.inf, 2147483648.0, -2147483904.0], np.float32)
            return np.where(rng.random(size) < 0.5, rng.choice(pool, size), rng.normal(0, 1000, size).astype(np.float32)).astype(np.float32)
        info = np.iinfo(dtype)
        wide = rng.integers(-2**31, 2**31 - 1, size, dtype=np.int64) if dtype.kind == 'i' else rng.integers(0, 2**32 - 1, size, dtype=np.int64)
        small = rng.integers(max(info.min, -300), min(info.max, 300) + 1, size, dtype=np.int64)
        return np.where(rng.random(size) < 0.3, wide, small).astype(dtype) if dtype.itemsize == 4 else small.astype(dtype)
    x = values(t, int(np.prod(shape))).reshape(shape)
    rank = len(dims);idx = np.zeros((lanes, max(rank, 1)), np.int32)
    for k, n in enumerate(dims):
        pool = np.concatenate([np.arange(-2 * n - 1, 2 * n + 2), np.array([-2**31, 2**31 - 1, -n, n, -1, 0, n - 1])])
        idx[:, k] = rng.choice(pool, lanes)
    v = values(u, lanes)
    return x, idx, v


def _text(a):
    a = np.asarray(a)
    return ' '.join(repr(float(v)) if a.dtype.kind == 'f' else str(int(v)) for v in a.reshape(-1))


def _from_text(tokens, dtype):
    dtype = np.dtype(dtype);values = np.array([float(t) for t in tokens], np.float64)
    if dtype == np.bool_:return values != 0
    return values.astype(dtype)


def cxx_compiler():
    import shutil
    for name in ('g++', 'clang++', 'c++'):
        found = shutil.which(name)
        if found:return found
    return None


def host_probe(header_text, cases, lanes, *, work, compiler=None, seed_base=0):
    """Compile the host probe against `header_text` with a C++ compiler (-x c++, no CUDA) and run every case; returns
    [{case, got, kept, out, clamps, drops, x, idx, v}] with numpy arrays, or None when no compiler is available."""
    compiler = compiler or cxx_compiler()
    if compiler is None:return None
    work = Path(work);work.mkdir(parents=True, exist_ok=True)
    (work / 'v2e_layout.cuh').write_text(header_text);(work / 'v2e_layout_probe_common.cuh').write_text(PROBE_COMMON)
    (work / 'v2e_layout_probe_host.cpp').write_text(probe_host_source(cases))
    exe = work / 'v2e_layout_probe_host'
    result = subprocess.run([compiler, '-std=c++17', '-O1', '-x', 'c++', '-o', str(exe), str(work / 'v2e_layout_probe_host.cpp')], capture_output=True, text=True)
    if result.returncode:raise RuntimeError(f'{compiler} failed ({result.returncode}):\n{result.stderr[-4000:]}')
    rows = [];lines = []
    for n, (t, u, dims) in enumerate(cases):
        x, idx, v = probe_inputs(t, u, dims, lanes, seed=seed_base + n);rows.append(dict(case=case_name(t, u, dims), t=t, u=u, dims=dims, x=x, idx=idx, v=v))
        lines.append(f"{t} {u} {'x'.join(map(str, dims)) or 'scalar'} {lanes} {_text(x)} {_text(idx[:, :max(len(dims), 1)])} {_text(v)}")
    run = subprocess.run([str(exe)], input='\n'.join(lines) + '\n', capture_output=True, text=True)
    if run.returncode:raise RuntimeError(f'host probe failed ({run.returncode}): {run.stderr[-2000:]}')
    outputs = run.stdout.strip().splitlines()
    if len(outputs) != len(rows):raise RuntimeError(f'host probe wrote {len(outputs)} lines for {len(rows)} cases')
    for row, line in zip(rows, outputs):
        got, kept, out, dbg = [part.split() for part in line.split('|')]
        row.update(got=_from_text(got, row['t']), kept=np.array([int(k) for k in kept], bool), out=_from_text(out, row['t']).reshape(row['x'].shape),
            clamps=int(dbg[0]), drops=int(dbg[1]))
    return rows


def _jax_reference(x, idx, v, rank):
    import jax
    import jax.numpy as j
    xj = j.asarray(x);rows = j.arange(x.shape[0]);index = tuple(j.asarray(idx[:, k]) for k in range(rank))
    got = xj[(rows,) + index] if rank else xj
    out = xj.at[(rows,) + index].set(j.asarray(v)) if rank else xj.at[rows].set(j.asarray(v))
    return np.asarray(got), np.asarray(out)


def _bits_equal(a, b):
    a = np.asarray(a);b = np.asarray(b)
    if a.dtype != b.dtype or a.shape != b.shape:return False
    return bool(np.array_equal(a.view(np.uint8) if a.dtype != np.bool_ else a, b.view(np.uint8) if b.dtype != np.bool_ else b))


def compile_header_check(header, work, nvcc_flags, nvcc):
    """nvcc compiles a translation unit that instantiates every struct's bind() and a few accessors, under NVCC_FLAGS."""
    import jax.ffi
    text = Path(header).read_text();names = re.findall(r'^struct (\w+) \{', text, re.M)
    body = ['#include "v2e_layout.cuh"', 'using namespace v2e_layout;', '__global__ void touch(void* const* bufs, const void* tables, int64_t lanes, uint32_t* sink) {',
        '  uint32_t acc = 0u;']
    for name in names:
        if name == 'EngineTables':body.append('  { auto t = EngineTables::bind(tables); acc += static_cast<uint32_t>(reinterpret_cast<uintptr_t>(t.base)); }')
        elif name in ('LeafBase', 'Leaf', 'Table'):continue
        else:body.append(f'  {{ auto v = {name}::bind(bufs, lanes); acc += static_cast<uint32_t>(v.kLeaves); }}')
    body += ['  sink[0] = acc;', '}']
    src = Path(work) / 'v2e_layout_instantiate.cu';src.write_text('\n'.join(body) + '\n')
    flags = [f for f in nvcc_flags if f not in ('-shared',)]
    cmd = [nvcc, *flags, f'-I{jax.ffi.include_dir()}', f'-I{Path(header).parent}', '-c', '-o', str(src.with_suffix('.o')), str(src)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return dict(command=' '.join(cmd), returncode=result.returncode, stderr=result.stderr[-4000:])


def cmd_probe(args):
    import jax
    import jax.numpy as j
    from gpu_engine import v2e_cuda
    description = load_description(args.description);spec = load_spec(args.spec) if args.spec and Path(args.spec).exists() else None
    header = Path(args.header);dtypes = sorted({e['dtype'] for key in ('state', 'actions', 'trackers') for e in description[key]} | {'int16', 'uint8', 'uint16'})
    cases = probe_cases(dtypes);report = dict(header=str(header), dtypes=dtypes, cases=[], lanes=args.lanes)
    digest = hashlib.sha256(header.read_bytes() + PROBE_COMMON.encode() + probe_cuda_source(cases).encode() + repr(v2e_cuda.NVCC_FLAGS).encode()).hexdigest()[:16]
    work = Path(args.build_dir) / f'layout-probe-{digest}';work.mkdir(parents=True, exist_ok=True)
    (work / 'v2e_layout.cuh').write_text(header.read_text());(work / 'v2e_layout_probe_common.cuh').write_text(PROBE_COMMON)
    (work / 'v2e_layout_probe.cu').write_text(probe_cuda_source(cases))
    report['header_compile'] = compile_header_check(work / 'v2e_layout.cuh', work, v2e_cuda.NVCC_FLAGS, v2e_cuda.nvcc())
    target = work / 'libv2e_layout_probe.so'
    if not target.exists():
        begin = time.perf_counter()
        cmd = [v2e_cuda.nvcc(), *v2e_cuda.NVCC_FLAGS, f'-I{jax.ffi.include_dir()}', f'-I{work}', '-o', str(target), str(work / 'v2e_layout_probe.cu')]
        result = subprocess.run(cmd, capture_output=True, text=True)
        report['probe_compile'] = dict(returncode=result.returncode, seconds=time.perf_counter() - begin, stderr=result.stderr[-4000:])
        if result.returncode:
            Path(args.out).mkdir(parents=True, exist_ok=True);(Path(args.out) / 'probe.json').write_text(json.dumps(report, indent=1))
            print(result.stderr[-4000:]);return 2
    lib = ctypes.cdll.LoadLibrary(str(target))
    for t, u, dims in cases:
        name = f'V2eLayoutProbe_{case_name(t, u, dims)}'
        jax.ffi.register_ffi_target(f'v2e_layout_probe_{digest}_{name}', jax.ffi.pycapsule(getattr(lib, name)), platform='CUDA')
    want_digest = digests(description, spec)['TrackerView'][0]
    def call(t, u, dims, x, idx, v, layout_digest):
        name = f'V2eLayoutProbe_{case_name(t, u, dims)}';lanes = x.shape[0]
        shapes = (jax.ShapeDtypeStruct(x.shape, x.dtype), jax.ShapeDtypeStruct((lanes,), x.dtype), jax.ShapeDtypeStruct((lanes,), np.uint8),
            jax.ShapeDtypeStruct((2,), np.uint32), jax.ShapeDtypeStruct((1,), np.uint32))
        out = jax.ffi.ffi_call(f'v2e_layout_probe_{digest}_{name}', shapes)(j.asarray(x), j.asarray(idx), j.asarray(v), layout_digest=digest_attr(layout_digest))
        return [np.asarray(o) for o in out]
    all_equal = True
    for n, (t, u, dims) in enumerate(cases):
        x, idx, v = probe_inputs(t, u, dims, args.lanes, seed=n);rank = len(dims)
        got_j, out_j = _jax_reference(x, idx, v, rank)
        got_e = emulate_get(x, *[idx[:, k] for k in range(rank)]);out_e = emulate_set(x, [idx[:, k] for k in range(rank)], v)
        out_p = emulate_set_point(x, [idx[:, k] for k in range(rank)], v)
        out_c, got_c, kept_c, dbg_c, err_c = call(t, u, dims, x, idx, v, want_digest)
        promoting = promotes(t, u);kept = kept_c.astype(bool);rows = np.flatnonzero(kept)
        element = (rows,) + tuple(scatter_index(idx[rows, k], d)[0] for k, d in enumerate(dims))   # the written element of every kept lane
        written_j = out_j[element];written_c = out_c[element]
        row = dict(case=case_name(t, u, dims), operand=t, update=u, dims=list(dims), lanes=int(x.shape[0]), promotes_operand=promoting,
            cuda_get_equals_jax=_bits_equal(got_c, got_j), cuda_set_equals_jax=_bits_equal(out_c, out_j),
            cuda_set_equals_point_emulation=_bits_equal(out_c, out_p), cuda_written_elements_equal_jax=_bits_equal(written_c, written_j),
            emulation_get_equals_jax=_bits_equal(got_e, got_j), emulation_set_equals_jax=_bits_equal(out_e, out_j),
            clamps=int(dbg_c[0]), drops=int(dbg_c[1]), kept=int(kept.sum()), error_word=int(err_c[0]),
            negative_indices=int((idx[:, :rank] < 0).sum()), out_of_range_indices=int(sum(((idx[:, k] < -d) | (idx[:, k] >= d)).sum() for k, d in enumerate(dims))))
        # A promoting pair (float update into an integer operand) round-trips the whole operand through float32 in JAX;
        # the point accessor converts the update alone, so it must match JAX on the written elements and the point
        # emulation everywhere (the oracle has no promoting scatter: scatter-census).
        set_ok = row['cuda_set_equals_jax'] if not promoting else (row['cuda_written_elements_equal_jax'] and row['cuda_set_equals_point_emulation'])
        row['equal'] = set_ok and row['cuda_get_equals_jax'] and row['emulation_get_equals_jax'] and row['emulation_set_equals_jax'] and row['error_word'] == 0
        all_equal &= row['equal'];report['cases'].append(row)
        if not row['equal']:print('MISMATCH', row)
    report['promoting_pairs'] = sorted({f"{r['operand']}<-{r['update']}" for r in report['cases'] if r['promotes_operand']})
    # The digest refusal: a wrong layout_digest attribute returns kStaleLayout and leaves the operand untouched (out = x).
    t, u, dims = cases[0];x, idx, v = probe_inputs(t, u, dims, args.lanes, seed=0)
    out_c, got_c, kept_c, dbg_c, err_c = call(t, u, dims, x, idx, v, want_digest ^ 1)
    report['digest_refusal'] = dict(passed_digest=f'{want_digest ^ 1:016x}', compiled_digest=f'{want_digest:016x}', error_word=int(err_c[0]),
        expected_error_word=STALE_LAYOUT, refused=int(err_c[0]) == STALE_LAYOUT, operand_untouched=_bits_equal(out_c, x))
    report['digests'] = {k: dict(digest64=f'{a:016x}', sha256=b) for k, (a, b) in digests(description, spec).items()}
    report['equal'] = all_equal and report['digest_refusal']['refused'] and report['digest_refusal']['operand_untouched'] and report['header_compile']['returncode'] == 0
    report['jax'] = jax.__version__
    Path(args.out).mkdir(parents=True, exist_ok=True);(Path(args.out) / 'probe.json').write_text(json.dumps(report, indent=1) + '\n')
    print(f"probe: {len(cases)} cases, equal={report['equal']}, header_compile={report['header_compile']['returncode']}, "
        f"digest_refusal={report['digest_refusal']['refused']} -> {Path(args.out) / 'probe.json'}")
    return 0 if report['equal'] else 1


# ---------------------------------------------------------------------------
# Scatter census: the (operand, update) dtype pairs of every scatter in the oracle step, so the point semantics of
# Leaf::set (update conversion only) is proven sufficient: no pair promotes the operand.

def scatter_dtype_pairs(closed):
    """{'<operand dtype><-<update dtype>': count} over every scatter-family equation of a closed jaxpr (sub-jaxprs
    included). The operand dtype is the original one when the operand is a convert_element_type of another var (that is
    _scatter_impl's promotion), else the scatter operand's."""
    from jax.extend import core as jcore
    import collections
    pairs = collections.Counter()
    def walk(jaxpr):
        producer = {}
        for e in jaxpr.eqns:
            for v in e.outvars:producer[v] = e
            if e.primitive.name.startswith('scatter'):
                operand, _, update = e.invars[0], e.invars[1], e.invars[2]
                src = producer.get(operand)
                original = str(src.invars[0].aval.dtype) if src is not None and src.primitive.name == 'convert_element_type' and not isinstance(src.invars[0], jcore.Literal) else str(operand.aval.dtype)
                pairs[f'{original}<-{update.aval.dtype}' + ('' if original == str(operand.aval.dtype) else f' (promoted to {operand.aval.dtype})')] += 1
            for sub in jcore.jaxprs_in_params(e.params):walk(sub)
    walk(closed.jaxpr)
    return dict(pairs)


def cmd_scatter_census(args):
    import jax
    from gpu_engine import v2e_census as cz, v2e_modules
    ctx = cz.load_context(args);state, actions, trackers = ctx['template']
    fn = v2e_modules.step(ctx['engine_tables'], ctx['observation_tables'])
    begin = time.perf_counter();closed = jax.make_jaxpr(fn)(state, actions, trackers)
    pairs = scatter_dtype_pairs(closed)
    promoting = {k: v for k, v in pairs.items() if 'promoted' in k or promotes(*[d.split(' ')[0] for d in k.split('<-')])}
    report = dict(schema='pokezero.v2e-scatter-census.v1', jax=jax.__version__, seconds=time.perf_counter() - begin, pairs=pairs, promoting=promoting,
        point_semantics_sufficient=not promoting)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True);Path(args.out).write_text(json.dumps(report, indent=1) + '\n')
    print(f"scatter-census: {sum(pairs.values())} scatters, pairs {pairs}, promoting {promoting} -> {args.out}")
    return 0 if not promoting else 1


# ---------------------------------------------------------------------------
# CLI.

def cmd_describe(args):
    from gpu_engine import v2e_census as cz
    ctx = cz.load_context(args)
    d = description_from_context(ctx)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True);Path(args.out).write_text(json.dumps(d, indent=1) + '\n')
    print(f'described: {len(d["state"])} state, {len(d["actions"])} actions, {len(d["trackers"])} trackers, {len(d["step_output"])} output leaves, {len(d["tables"])} tables -> {args.out}')
    return 0


def _spec_or_none(path):
    if path and Path(path).exists():return load_spec(path)
    if path:print(f'note: spec {path} does not exist; carries omitted')
    return None


def cmd_generate(args):
    description = load_description(args.description);spec = _spec_or_none(args.spec)
    text = generate(description, spec)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True);Path(args.out).write_text(text)
    print(f'wrote {args.out}: {len(text.splitlines())} lines; digests ' + ', '.join(f'{k} {v[0]:016x}' for k, v in digests(description, spec).items()))
    return 0


def cmd_check(args):
    description = load_description(args.description);spec = _spec_or_none(args.spec)
    want = generate(description, spec);have = Path(args.header).read_text()
    if want != have:
        import difflib
        diff = list(difflib.unified_diff(have.splitlines(), want.splitlines(), 'committed', 'regenerated', lineterm='', n=1))
        print('\n'.join(diff[:60]));print(f'STALE: {args.header} differs from the regenerated header ({len(diff)} diff lines)')
        return 1
    print(f'OK: {args.header} matches the regenerated header')
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0]);sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('describe', help='dump the pytree description from the real tables (GPU host)')
    q.add_argument('--config', required=True);q.add_argument('--reference-run', required=True);q.add_argument('--arm', default='B', choices=('A', 'B'))
    q.add_argument('--out', required=True);q.set_defaults(fn=cmd_describe)
    for name, fn, help_ in (('generate', cmd_generate, 'write the header'), ('check', cmd_check, 'regenerate and compare with the committed header')):
        q = sub.add_parser(name, help=help_)
        q.add_argument('--description', default=str(DEFAULT_DESCRIPTION));q.add_argument('--spec', default=str(DEFAULT_SPEC))
        q.add_argument('--out' if name == 'generate' else '--header', default=str(HEADER));q.set_defaults(fn=fn)
    q = sub.add_parser('probe', help='GPU: header compiles under NVCC_FLAGS; accessors and numpy emulation vs jax.lax; digest refusal')
    q.add_argument('--build-dir', required=True);q.add_argument('--out', required=True);q.add_argument('--header', default=str(HEADER))
    q.add_argument('--description', default=str(DEFAULT_DESCRIPTION));q.add_argument('--spec', default=str(DEFAULT_SPEC))
    q.add_argument('--lanes', type=int, default=96);q.set_defaults(fn=cmd_probe)
    q = sub.add_parser('scatter-census', help='GPU host: (operand, update) dtype pairs of every scatter in the oracle step')
    q.add_argument('--config', required=True);q.add_argument('--reference-run', required=True);q.add_argument('--arm', default='B', choices=('A', 'B'))
    q.add_argument('--out', required=True);q.set_defaults(fn=cmd_scatter_census)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    return args.fn(args)


if __name__ == '__main__':raise SystemExit(main())
