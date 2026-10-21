"""Verify one actual JAX GPU and a synchronized JIT operation, not game speed."""
import json
import platform


def main():
    import jax
    import jax.numpy as jnp

    devices = jax.devices("gpu")
    if len(devices) != 1:
        raise RuntimeError(f"exactly one GPU required; got {len(devices)}")
    x = jax.jit(lambda x: x * x + 1)(jnp.arange(2048, dtype=jnp.float32))
    x.block_until_ready()
    expected = 2047 * 2047 + 1
    if float(x[-1]) != expected or x.devices() != {devices[0]}:
        raise RuntimeError("JIT operation did not execute correctly on the sole GPU")
    print(json.dumps(dict(status="JAX_GPU_PROBE_OK", jax=jax.__version__,
                          architecture=platform.machine(), gpu_count=len(devices),
                          device_kind=devices[0].device_kind, kernel_shape=list(x.shape),
                          games_per_second=None, scope="Device/JIT capability only"), sort_keys=True))


if __name__ == "__main__":
    main()
