// Scalar semantics of XLA GPU (jaxlib 0.12, sm_103) primitives, bit-exact.
//
// Pinned by probes against XLA on GB300 (see the engine acceleration write-up):
//  - f32 add/sub/mul round to nearest, never contracted into FMA (explicit _rn);
//  - f32 divide is PTX div.full.f32 (not IEEE div.rn);
//  - denormals are kept (no flush to zero);
//  - f32 -> int converts toward zero, saturating, NaN -> 0;
//  - min/max propagate NaN and order -0 < +0;
//  - integer div truncates; x/0 = -1 (signed) or all ones (unsigned);
//    INT_MIN/-1 = INT_MIN; rem: x%0 = x, INT_MIN%-1 = 0;
//  - shifts by >= bit width (or negative) give 0, arithmetic right shift sign-fills.
// Integer arithmetic wraps (computed in unsigned). Every helper is checked
// against XLA by the conformance probe in gpu_engine.v2e_conformance.
#pragma once
#include <cstdint>
#include <cmath>
#include <type_traits>

namespace v2e {

// ---- float -----------------------------------------------------------------
__device__ __forceinline__ float fadd(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float fsub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float fmul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float fdiv(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float frem(float a, float b) { return fmodf(a, b); }
__device__ __forceinline__ bool fnan(float a) { return a != a; }
// NaN-propagating min/max returning the canonical NaN, as XLA emits (PTX .NaN).
__device__ __forceinline__ float fmin_x(float a, float b) {
  float r;
  asm("min.NaN.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float fmax_x(float a, float b) {
  float r;
  asm("max.NaN.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float fneg(float a) {
  float r;
  asm("neg.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ float fabs_x(float a) {
  float r;
  asm("abs.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ float fsign(float a) {
  if (fnan(a) || a == 0.0f) return a;
  return a > 0.0f ? 1.0f : -1.0f;
}
__device__ __forceinline__ float ffloor(float a) { return floorf(a); }
__device__ __forceinline__ float fceil(float a) { return ceilf(a); }
__device__ __forceinline__ float fround_even(float a) { return rintf(a); }
__device__ __forceinline__ float fround_away(float a) { return roundf(a); }
__device__ __forceinline__ bool fisfinite(float a) { return isfinite(a); }

// ---- conversions -----------------------------------------------------------
__device__ __forceinline__ int32_t f2s32(float a) {
  if (fnan(a)) return 0;
  if (a >= 2147483648.0f) return INT32_MAX;
  if (a <= -2147483648.0f) return INT32_MIN;
  return static_cast<int32_t>(a);
}
__device__ __forceinline__ uint32_t f2u32(float a) {
  if (fnan(a) || a <= 0.0f) return 0u;
  if (a >= 4294967296.0f) return UINT32_MAX;
  return static_cast<uint32_t>(a);
}
__device__ __forceinline__ int32_t f2s_narrow(float a, int32_t lo, int32_t hi) {
  if (fnan(a)) return 0;
  if (a >= static_cast<float>(hi)) return hi;
  if (a <= static_cast<float>(lo)) return lo;
  return static_cast<int32_t>(a);
}
__device__ __forceinline__ float s2f(int32_t a) { return __int2float_rn(a); }
__device__ __forceinline__ float u2f(uint32_t a) { return __uint2float_rn(a); }

// ---- integers (wrapping) ----------------------------------------------------
__device__ __forceinline__ int32_t iadd(int32_t a, int32_t b) { return static_cast<int32_t>(static_cast<uint32_t>(a) + static_cast<uint32_t>(b)); }
__device__ __forceinline__ int32_t isub(int32_t a, int32_t b) { return static_cast<int32_t>(static_cast<uint32_t>(a) - static_cast<uint32_t>(b)); }
__device__ __forceinline__ int32_t imul(int32_t a, int32_t b) { return static_cast<int32_t>(static_cast<uint32_t>(a) * static_cast<uint32_t>(b)); }
__device__ __forceinline__ int32_t ineg(int32_t a) { return static_cast<int32_t>(0u - static_cast<uint32_t>(a)); }
__device__ __forceinline__ int32_t iabs(int32_t a) { return a < 0 ? ineg(a) : a; }
__device__ __forceinline__ int32_t isign(int32_t a) { return (a > 0) - (a < 0); }
__device__ __forceinline__ int32_t idiv(int32_t a, int32_t b) {
  if (b == 0) return -1;
  if (a == INT32_MIN && b == -1) return INT32_MIN;
  return a / b;
}
__device__ __forceinline__ int32_t irem(int32_t a, int32_t b) {
  if (b == 0) return a;
  if (a == INT32_MIN && b == -1) return 0;
  return a % b;
}
__device__ __forceinline__ uint32_t udiv(uint32_t a, uint32_t b) { return b == 0u ? UINT32_MAX : a / b; }
__device__ __forceinline__ uint32_t urem(uint32_t a, uint32_t b) { return b == 0u ? a : a % b; }
template <class T>
__device__ __forceinline__ T shl(T a, T s) {
  using U = typename std::make_unsigned<T>::type;
  const U u = static_cast<U>(s);
  return u >= sizeof(T) * 8u ? T(0) : static_cast<T>(static_cast<U>(a) << u);
}
template <class T>
__device__ __forceinline__ T shr_logical(T a, T s) {
  using U = typename std::make_unsigned<T>::type;
  const U u = static_cast<U>(s);
  return u >= sizeof(T) * 8u ? T(0) : static_cast<T>(static_cast<U>(a) >> u);
}
template <class T>
__device__ __forceinline__ T shr_arith(T a, T s) {
  using U = typename std::make_unsigned<T>::type;
  using S = typename std::make_signed<T>::type;
  const U u = static_cast<U>(s);
  const S v = static_cast<S>(a);
  if (u >= sizeof(T) * 8u) return static_cast<T>(v < 0 ? S(-1) : S(0));
  return static_cast<T>(v >> u);
}

// ---- warp helpers ----------------------------------------------------------
constexpr unsigned kFull = 0xffffffffu;

template <class T>
__device__ __forceinline__ T shfl(T x, int lane) {
  if constexpr (std::is_same<T, bool>::value) {
    return __shfl_sync(kFull, static_cast<int>(x), lane) != 0;
  } else if constexpr (std::is_same<T, float>::value) {
    return __shfl_sync(kFull, x, lane);
  } else if constexpr (sizeof(T) < 4) {
    return static_cast<T>(__shfl_sync(kFull, static_cast<int>(x), lane));
  } else {
    return __shfl_sync(kFull, x, lane);
  }
}

template <class T>
__device__ __forceinline__ T shfl_xor(T x, int mask) {
  if constexpr (std::is_same<T, bool>::value) {
    return __shfl_xor_sync(kFull, static_cast<int>(x), mask) != 0;
  } else if constexpr (std::is_same<T, float>::value) {
    return __shfl_xor_sync(kFull, x, mask);
  } else if constexpr (sizeof(T) < 4) {
    return static_cast<T>(__shfl_xor_sync(kFull, static_cast<int>(x), mask));
  } else {
    return __shfl_xor_sync(kFull, x, mask);
  }
}

template <class T>
__device__ __forceinline__ T shfl_up(T x, int delta) {
  if constexpr (std::is_same<T, bool>::value) {
    return __shfl_up_sync(kFull, static_cast<int>(x), delta) != 0;
  } else if constexpr (std::is_same<T, float>::value) {
    return __shfl_up_sync(kFull, x, delta);
  } else if constexpr (sizeof(T) < 4) {
    return static_cast<T>(__shfl_up_sync(kFull, static_cast<int>(x), delta));
  } else {
    return __shfl_up_sync(kFull, x, delta);
  }
}

struct SortKey {
  const uint32_t* base;  // element j of the row is base[j * stride]
  int64_t stride;
  int kind;              // 0 unsigned (incl. bool), 1 signed (sign-extended words), 2 f32
};

// Order-preserving unsigned image of a key word. Floats follow JAX's sort
// comparator: -0 == +0, every NaN equal and above +inf.
__device__ __forceinline__ uint32_t sort_bits(uint32_t w, int kind) {
  if (kind == 1) return w ^ 0x80000000u;
  if (kind == 2) {
    const uint32_t mag = w & 0x7fffffffu;
    if (mag == 0u) w = 0u;
    else if (mag > 0x7f800000u) w = 0x7fc00000u;
    return (w & 0x80000000u) ? ~w : (w | 0x80000000u);
  }
  return w;
}

// Stable argsort of one row of n elements by lexicographic keys (keys[0] most
// significant), as a warp-cooperative LSD radix sort (5-bit digits; only the
// bit range that actually differs is processed). Every correct stable sort
// yields the same permutation, so the result equals XLA's stable sort.
// P, Q: n shared words each; O: 32 shared words. All 32 threads must call it.
// On return P[j] is the source index of sorted position j.
template <int NK>
__device__ void radix_argsort(const SortKey (&keys)[NK], int n, uint32_t* P, uint32_t* Q, uint32_t* O, int t) {
  for (int j = t; j < n; j += 32) P[j] = static_cast<uint32_t>(j);
  __syncwarp();
  for (int kk = NK - 1; kk >= 0; --kk) {
    const SortKey key = keys[kk];
    const uint32_t first = n > 0 ? sort_bits(key.base[0], key.kind) : 0u;
    uint32_t diff = 0u;
    for (int j = t; j < n; j += 32) diff |= sort_bits(key.base[j * key.stride], key.kind) ^ first;
    for (int s = 16; s > 0; s >>= 1) diff |= __shfl_xor_sync(kFull, diff, s);
    const int bits = diff ? 32 - __clz(static_cast<int>(diff)) : 0;
    for (int shift = 0; shift < bits; shift += 5) {
      O[t] = 0u;
      __syncwarp();
      for (int j = t; j < n; j += 32)
        atomicAdd(&O[(sort_bits(key.base[static_cast<int>(P[j]) * key.stride], key.kind) >> shift) & 31u], 1u);
      __syncwarp();
      const uint32_t c = O[t];
      uint32_t x = c;
      for (int s = 1; s < 32; s <<= 1) {
        const uint32_t y = __shfl_up_sync(kFull, x, s);
        if (t >= s) x += y;
      }
      __syncwarp();
      O[t] = x - c;
      __syncwarp();
      for (int base = 0; base < n; base += 32) {
        const int j = base + t;
        const bool active = j < n;
        const uint32_t e = active ? P[j] : 0u;
        const uint32_t d = active ? ((sort_bits(key.base[static_cast<int>(e) * key.stride], key.kind) >> shift) & 31u)
                                  : (32u + static_cast<uint32_t>(t));
        const uint32_t peers = __match_any_sync(kFull, d);
        const uint32_t rank = __popc(peers & ((1u << t) - 1u));
        const uint32_t dst = active ? O[d] + rank : 0u;
        __syncwarp();
        if (active && rank == 0u) O[d] += static_cast<uint32_t>(__popc(peers));
        __syncwarp();
        if (active) Q[dst] = e;
      }
      __syncwarp();
      for (int j = t; j < n; j += 32) P[j] = Q[j];
      __syncwarp();
    }
  }
}

// Lexicographic key order of row elements a and b.
template <int NK>
__device__ __forceinline__ bool key_less(const SortKey (&keys)[NK], uint32_t a, uint32_t b) {
#pragma unroll
  for (int kk = 0; kk < NK; ++kk) {
    const uint32_t ka = sort_bits(keys[kk].base[static_cast<int64_t>(a) * keys[kk].stride], keys[kk].kind);
    const uint32_t kb = sort_bits(keys[kk].base[static_cast<int64_t>(b) * keys[kk].stride], keys[kk].kind);
    if (ka != kb) return ka < kb;
  }
  return false;
}

// One thread: stable argsort (insertion) of n row elements; P[j * ps] receives the source index of position j.
template <int NK>
__device__ void thread_argsort(const SortKey (&keys)[NK], int n, uint32_t* P, int64_t ps) {
  for (int j = 0; j < n; ++j) P[j * ps] = static_cast<uint32_t>(j);
  for (int j = 1; j < n; ++j) {
    const uint32_t x = P[j * ps];
    int i = j - 1;
    while (i >= 0 && key_less<NK>(keys, x, P[i * ps])) {
      P[(i + 1) * ps] = P[i * ps];
      --i;
    }
    P[(i + 1) * ps] = x;
  }
}

// Element k of a small register array (unrolled selects keep the array in registers).
template <class T, int N>
__device__ __forceinline__ T pick(const T (&a)[N], int64_t k) {
  T x = a[0];
#pragma unroll
  for (int j = 1; j < N; ++j) x = (k == j) ? a[j] : x;
  return x;
}

// Lanes of the warp below t.
__device__ __forceinline__ uint32_t lanes_below(int t) { return (1u << t) - 1u; }

}  // namespace v2e
