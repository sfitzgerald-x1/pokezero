// Exact CPython random.Random, as a device/host function set (Modules/_randommodule.c).
//
//   random.Random(int_seed): the integer's absolute value is split into 32-bit
//     words from the right (little-endian word order), leading zero words are
//     dropped (keyused = (bit_length-1)/32+1, at least 1), and init_by_array
//     seeds the generator (init_genrand(19650218) then the two mixing passes).
//   getrandbits(k), 1 <= k <= 32: genrand_uint32() >> (32-k).
//   randrange(n) == _randbelow_with_getrandbits(n): k = n.bit_length();
//     r = getrandbits(k); while r >= n: r = getrandbits(k).
// All arithmetic is uint32 with wraparound; no floats.
#pragma once
#include <cstdint>

namespace v2e {

constexpr int kMtN = 624;
constexpr int kMtM = 397;
constexpr uint32_t kMtMatrixA = 0x9908b0dfu;
constexpr uint32_t kMtUpperMask = 0x80000000u;
constexpr uint32_t kMtLowerMask = 0x7fffffffu;

struct Mt19937 {
  uint32_t mt[kMtN];
  int index;
};

__host__ __device__ inline void mt_init_genrand(Mt19937& s, uint32_t seed) {
  s.mt[0] = seed;
  for (int i = 1; i < kMtN; ++i) s.mt[i] = 1812433253u * (s.mt[i - 1] ^ (s.mt[i - 1] >> 30)) + static_cast<uint32_t>(i);
  s.index = kMtN;
}

// init_by_array(key, key_length) exactly as CPython (key_length >= 1).
__host__ __device__ inline void mt_init_by_array(Mt19937& s, const uint32_t* key, int key_length) {
  mt_init_genrand(s, 19650218u);
  int i = 1, j = 0;
  int k = kMtN > key_length ? kMtN : key_length;
  for (; k; --k) {
    s.mt[i] = (s.mt[i] ^ ((s.mt[i - 1] ^ (s.mt[i - 1] >> 30)) * 1664525u)) + key[j] + static_cast<uint32_t>(j);
    ++i;
    ++j;
    if (i >= kMtN) {
      s.mt[0] = s.mt[kMtN - 1];
      i = 1;
    }
    if (j >= key_length) j = 0;
  }
  for (k = kMtN - 1; k; --k) {
    s.mt[i] = (s.mt[i] ^ ((s.mt[i - 1] ^ (s.mt[i - 1] >> 30)) * 1566083941u)) - static_cast<uint32_t>(i);
    ++i;
    if (i >= kMtN) {
      s.mt[0] = s.mt[kMtN - 1];
      i = 1;
    }
  }
  s.mt[0] = 0x80000000u;
  s.index = kMtN;
}

// random.Random(seed) for a non-negative integer given as `words` little-endian
// 32-bit words (any count >= 1). Leading zero words are dropped as CPython's
// _PyLong_NumBits-based keyused does; a zero seed keeps one zero word.
__host__ __device__ inline void mt_seed_words(Mt19937& s, const uint32_t* words, int count) {
  int used = count;
  while (used > 1 && words[used - 1] == 0u) --used;
  mt_init_by_array(s, words, used);
}

__host__ __device__ inline uint32_t mt_genrand_uint32(Mt19937& s) {
  if (s.index >= kMtN) {
    uint32_t y;
    int kk = 0;
    for (; kk < kMtN - kMtM; ++kk) {
      y = (s.mt[kk] & kMtUpperMask) | (s.mt[kk + 1] & kMtLowerMask);
      s.mt[kk] = s.mt[kk + kMtM] ^ (y >> 1) ^ ((y & 1u) ? kMtMatrixA : 0u);
    }
    for (; kk < kMtN - 1; ++kk) {
      y = (s.mt[kk] & kMtUpperMask) | (s.mt[kk + 1] & kMtLowerMask);
      s.mt[kk] = s.mt[kk + (kMtM - kMtN)] ^ (y >> 1) ^ ((y & 1u) ? kMtMatrixA : 0u);
    }
    y = (s.mt[kMtN - 1] & kMtUpperMask) | (s.mt[0] & kMtLowerMask);
    s.mt[kMtN - 1] = s.mt[kMtM - 1] ^ (y >> 1) ^ ((y & 1u) ? kMtMatrixA : 0u);
    s.index = 0;
  }
  uint32_t y = s.mt[s.index++];
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680u;
  y ^= (y << 15) & 0xefc60000u;
  y ^= (y >> 18);
  return y;
}

// getrandbits(k) for 1 <= k <= 32.
__host__ __device__ inline uint32_t mt_getrandbits(Mt19937& s, int k) {
  return mt_genrand_uint32(s) >> (32 - k);
}

__host__ __device__ inline int mt_bit_length(uint32_t n) {
  int k = 0;
  while (n) {
    ++k;
    n >>= 1;
  }
  return k;
}

// randrange(n) for 1 <= n <= 2**32-1 (CPython _randbelow_with_getrandbits).
__host__ __device__ inline uint32_t mt_randbelow(Mt19937& s, uint32_t n) {
  const int k = mt_bit_length(n);
  uint32_t r = mt_getrandbits(s, k);
  while (r >= n) r = mt_getrandbits(s, k);
  return r;
}

}  // namespace v2e
