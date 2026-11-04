// Probe handlers used only to pin XLA GPU float semantics against CUDA
// instructions (division variants). Not part of the engine.
#include <cstdint>
#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace v2e {

__global__ void div_variants_kernel(int64_t n, const float* a, const float* b, float* out) {
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  const float x = a[i], y = b[i];
  float r;
  asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(x), "f"(y));
  out[0 * n + i] = r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(x), "f"(y));
  out[1 * n + i] = r;
  asm("div.approx.f32 %0, %1, %2;" : "=f"(r) : "f"(x), "f"(y));
  out[2 * n + i] = r;
  asm("div.full.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(x), "f"(y));
  out[3 * n + i] = r;
  asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(x), "f"(y));
  out[4 * n + i] = r;
  float rc;
  asm("rcp.approx.f32 %0, %1;" : "=f"(rc) : "f"(y));
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(x), "f"(rc));
  out[5 * n + i] = r;
}

static ffi::Error DivVariantsImpl(cudaStream_t stream, ffi::Buffer<ffi::F32> a, ffi::Buffer<ffi::F32> b,
                                  ffi::Result<ffi::Buffer<ffi::F32>> out) {
  const int64_t n = static_cast<int64_t>(a.element_count());
  div_variants_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(n, a.typed_data(), b.typed_data(),
                                                                                    out->typed_data());
  return ffi::Error::Success();
}

}  // namespace v2e

XLA_FFI_DEFINE_HANDLER_SYMBOL(V2eDivVariants, v2e::DivVariantsImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>());
