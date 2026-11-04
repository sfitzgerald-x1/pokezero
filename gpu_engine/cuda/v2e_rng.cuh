// Exact JAX threefry2x32 RNG (jax 0.12, jax_threefry_partitionable=True).
//
// Raw keys are two uint32 words (k.a, k.b), as in old-style jax.random keys.
//   split(k, n)[i]      = threefry(k, hi(i), lo(i)) -> (x0, x1)
//   fold_in(k, d)       = threefry(k, 0, uint32(d)) -> (x0, x1)
//   random_bits32(k)[i] = x0 ^ x1 of threefry(k, hi(i), lo(i))
//   randint(k, ..., lo, hi)[i] follows jax.random._randint for int32.
// Every operation is uint32 arithmetic with wraparound; no floats.
#pragma once
#include <cstdint>

namespace v2e {

struct Key {
  uint32_t a, b;
};

__host__ __device__ inline uint32_t rotl32(uint32_t x, uint32_t d) {
  return (x << d) | (x >> (32u - d));  // d is always 6..29 here
}

__host__ __device__ inline void threefry2x32(uint32_t k0, uint32_t k1, uint32_t& x0, uint32_t& x1) {
  const uint32_t k2 = k0 ^ k1 ^ 0x1BD11BDAu;
  x0 += k0;
  x1 += k1;
#define V2E_ROUND(r) \
  x0 += x1;          \
  x1 = rotl32(x1, r); \
  x1 ^= x0;
  V2E_ROUND(13) V2E_ROUND(15) V2E_ROUND(26) V2E_ROUND(6)
  x0 += k1; x1 += k2 + 1u;
  V2E_ROUND(17) V2E_ROUND(29) V2E_ROUND(16) V2E_ROUND(24)
  x0 += k2; x1 += k0 + 2u;
  V2E_ROUND(13) V2E_ROUND(15) V2E_ROUND(26) V2E_ROUND(6)
  x0 += k0; x1 += k1 + 3u;
  V2E_ROUND(17) V2E_ROUND(29) V2E_ROUND(16) V2E_ROUND(24)
  x0 += k1; x1 += k2 + 4u;
  V2E_ROUND(13) V2E_ROUND(15) V2E_ROUND(26) V2E_ROUND(6)
  x0 += k2; x1 += k0 + 5u;
#undef V2E_ROUND
}

// split(k, n)[i] and fold_in(k, i) coincide for 32-bit i.
__host__ __device__ inline Key fold_in(Key k, uint32_t data) {
  uint32_t x0 = 0u, x1 = data;
  threefry2x32(k.a, k.b, x0, x1);
  return Key{x0, x1};
}

__host__ __device__ inline Key split_at(Key k, uint32_t index) { return fold_in(k, index); }

__host__ __device__ inline uint32_t random_bits32(Key k, uint32_t index) {
  uint32_t x0 = 0u, x1 = index;
  threefry2x32(k.a, k.b, x0, x1);
  return x0 ^ x1;
}

// jax.random.randint(k, shape, minval, maxval, int32), flattened element `index`.
__host__ __device__ inline int32_t randint_i32(Key k, uint32_t index, int32_t minval, int32_t maxval) {
  const Key k1 = split_at(k, 0u), k2 = split_at(k, 1u);
  const uint32_t higher = random_bits32(k1, index), lower = random_bits32(k2, index);
  uint32_t span = static_cast<uint32_t>(maxval) - static_cast<uint32_t>(minval);
  if (maxval <= minval) span = 1u;
  uint32_t multiplier = 65536u % span;
  multiplier = (multiplier * multiplier) % span;
  const uint32_t offset = ((higher % span) * multiplier + (lower % span)) % span;
  return static_cast<int32_t>(static_cast<uint32_t>(minval) + offset);
}

}  // namespace v2e
