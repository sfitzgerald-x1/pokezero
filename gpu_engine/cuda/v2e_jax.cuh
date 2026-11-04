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

}  // namespace v2e
