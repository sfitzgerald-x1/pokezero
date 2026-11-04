// XLA FFI entry points of the V2 CUDA engine (P0 round trip, P2 chance inputs).
//
// Every handler is command-buffer compatible, so XLA keeps it inside the
// per-tick CUDA graph. Pytree leaves arrive in jax.tree flatten order with a
// leading lane axis; outputs alias inputs where state is updated in place.
#include <cstdint>
#include <string>
#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"
#include "v2e_rng.cuh"

namespace ffi = xla::ffi;

namespace v2e {

constexpr int kMaxLeaves = 256;

struct Leaves {
  uint8_t* data[kMaxLeaves];
  uint32_t lane_bytes[kMaxLeaves];
  int count;
};

static ffi::Error cuda_error(const char* where) {
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
    return ffi::Error::Internal(std::string(where) + ": " + cudaGetErrorString(status));
  return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
// P0: in-place round trip. One thread per lane reads and rewrites its own
// slice of every leaf (the access pattern of the per-lane engine kernels) and
// folds the bytes into a checksum that proves the reads happened.

__global__ void touch_kernel(Leaves leaves, int64_t lanes, uint32_t* checksum) {
  const int64_t lane = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (lane >= lanes) return;
  uint32_t sum = 0u, mix = 0u;
  for (int i = 0; i < leaves.count; ++i) {
    const uint32_t n = leaves.lane_bytes[i];
    uint8_t* base = leaves.data[i] + lane * static_cast<int64_t>(n);
    if ((n & 3u) == 0u && (reinterpret_cast<uintptr_t>(base) & 3u) == 0u) {
      uint32_t* words = reinterpret_cast<uint32_t*>(base);
      for (uint32_t k = 0; k < n / 4u; ++k) {
        const uint32_t v = words[k];
        sum += v * (2u * k + 1u);
        mix ^= v + 0x9E3779B9u * (k + 1u);
        words[k] = v;
      }
    } else {
      for (uint32_t k = 0; k < n; ++k) {
        const uint32_t v = base[k];
        sum += v * (2u * k + 1u);
        mix ^= v + 0x9E3779B9u * (k + 1u);
        base[k] = static_cast<uint8_t>(v);
      }
    }
  }
  atomicAdd(&checksum[0], sum);
  atomicXor(&checksum[1], mix);
}

static ffi::Error RoundTripImpl(cudaStream_t stream, int64_t lanes, ffi::RemainingArgs args,
                                ffi::Result<ffi::Buffer<ffi::U32>> checksum, ffi::RemainingRets rets) {
  if (args.size() != rets.size() || args.size() > static_cast<size_t>(kMaxLeaves) || lanes <= 0)
    return ffi::Error::InvalidArgument("round trip: matching leaf lists up to 256 and positive lanes required");
  Leaves leaves{};
  leaves.count = static_cast<int>(args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    auto arg = args.get<ffi::AnyBuffer>(i);
    auto ret = rets.get<ffi::AnyBuffer>(i);
    if (!arg.has_value()) return arg.error();
    if (!ret.has_value()) return ret.error();
    const size_t bytes = arg->size_bytes();
    if (bytes % static_cast<size_t>(lanes) != 0)
      return ffi::Error::InvalidArgument("round trip: every leaf needs the leading lane axis");
    void* dst = (*ret)->untyped_data();
    if (dst != arg->untyped_data())
      cudaMemcpyAsync(dst, arg->untyped_data(), bytes, cudaMemcpyDeviceToDevice, stream);
    leaves.data[i] = static_cast<uint8_t*>(dst);
    leaves.lane_bytes[i] = static_cast<uint32_t>(bytes / static_cast<size_t>(lanes));
  }
  cudaMemsetAsync(checksum->typed_data(), 0, 2 * sizeof(uint32_t), stream);
  const int threads = 128;
  touch_kernel<<<static_cast<unsigned>((lanes + threads - 1) / threads), threads, 0, stream>>>(
      leaves, lanes, checksum->typed_data());
  return cuda_error("touch_kernel");
}

// ---------------------------------------------------------------------------
// P2: transition.sample_chance_inputs plus the step's key advance,
// split(state.keys, 2)[:, 0], bit-exact.

__global__ void chance_kernel(int64_t lanes, const uint32_t* keys, const int32_t* replacement_count,
                              int32_t* draws, int32_t* phazing, int32_t* secondary, int32_t* sleep_talk,
                              int32_t* entry_ties, int32_t* initial_tie, uint32_t* next_keys) {
  const int64_t b = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (b >= lanes) return;
  constexpr int32_t limits[12] = {2, 100, 100, 100, 16, 16, 100, 4, 8, 2, 2, 2};
  const Key key{keys[2 * b], keys[2 * b + 1]};
  const Key ordinary = split_at(key, 1u);  // jax.random.split(key, 2)[1]
  for (uint32_t i = 0; i < 24u; ++i) draws[24 * b + i] = randint_i32(ordinary, i, 0, limits[i % 12u]);
  const Key kp = fold_in(ordinary, 271828u), ks = fold_in(ordinary, 1618033u), kt = fold_in(ordinary, 1414213u);
  const Key ke = fold_in(key, 982451653u + static_cast<uint32_t>(replacement_count[b]));
  for (uint32_t i = 0; i < 2u; ++i) {
    phazing[2 * b + i] = randint_i32(kp, i, 0, 60);
    secondary[2 * b + i] = randint_i32(ks, i, 0, 100);
    sleep_talk[2 * b + i] = randint_i32(kt, i, 0, 60);
    entry_ties[2 * b + i] = randint_i32(ke, i, 0, 2);
  }
  initial_tie[b] = randint_i32(fold_in(key, 314159u), 0u, 0, 2);
  const Key advanced = split_at(key, 0u);
  next_keys[2 * b] = advanced.a;
  next_keys[2 * b + 1] = advanced.b;
}

static ffi::Error ChanceImpl(cudaStream_t stream, ffi::Buffer<ffi::U32> keys, ffi::Buffer<ffi::S32> replacement_count,
                             ffi::Result<ffi::Buffer<ffi::S32>> draws, ffi::Result<ffi::Buffer<ffi::S32>> phazing,
                             ffi::Result<ffi::Buffer<ffi::S32>> secondary, ffi::Result<ffi::Buffer<ffi::S32>> sleep_talk,
                             ffi::Result<ffi::Buffer<ffi::S32>> entry_ties, ffi::Result<ffi::Buffer<ffi::S32>> initial_tie,
                             ffi::Result<ffi::Buffer<ffi::U32>> next_keys) {
  const int64_t lanes = static_cast<int64_t>(replacement_count.element_count());
  if (keys.element_count() != static_cast<size_t>(2 * lanes) || draws->element_count() != static_cast<size_t>(24 * lanes))
    return ffi::Error::InvalidArgument("chance: [B,2] keys, [B] replacement counts and [B,2,12] draws required");
  const int threads = 128;
  chance_kernel<<<static_cast<unsigned>((lanes + threads - 1) / threads), threads, 0, stream>>>(
      lanes, keys.typed_data(), replacement_count.typed_data(), draws->typed_data(), phazing->typed_data(),
      secondary->typed_data(), sleep_talk->typed_data(), entry_ties->typed_data(), initial_tie->typed_data(),
      next_keys->typed_data());
  return cuda_error("chance_kernel");
}

}  // namespace v2e

XLA_FFI_DEFINE_HANDLER_SYMBOL(V2eRoundTrip, v2e::RoundTripImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<int64_t>("lanes")
                                  .RemainingArgs()
                                  .Ret<ffi::Buffer<ffi::U32>>()
                                  .RemainingRets(),
                              {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(V2eChance, v2e::ChanceImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::U32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::U32>>(),
                              {ffi::Traits::kCmdBufferCompatible});
