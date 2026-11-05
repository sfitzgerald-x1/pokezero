"""Compare the CUDA MT19937 (gpu_engine/cuda/v2e_mt19937.cuh, through the V2eMtProbe handler) with CPython.

For every seed: random.Random(seed).getrandbits(32) and randrange(n) sequences on the host against the
device probe's words. Seeds include 0, 1, 2**31-1, 2**31, 2**32-1, 2**256-1, seeds with zero high words and
real Recipe.world_seed values (--recipe-seed / --addresses); n includes 1, 2, 3, 7, 100, 255, 256, 257,
2**31-1, 2**31, 2**32-1. Prints one JSON line per comparison and a summary; exit 1 on any difference.
"""
import argparse
import json
import random
import sys
import time

import numpy as np


def seed_words(seed, words=8):
    return np.frombuffer(int(seed).to_bytes(4 * words, 'little'), np.uint32)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draws', type=int, default=100000, help='draws per (seed, mode)')
    parser.add_argument('--seeds', type=int, default=64, help='random 256-bit seeds in addition to the edge seeds')
    parser.add_argument('--recipe-seed', type=int, default=None, help='Recipe seed for actual world seeds')
    parser.add_argument('--addresses', type=int, default=16, help='addresses (stream, frame, owner) drawn for world seeds')
    parser.add_argument('--build-dir', default=None)
    args = parser.parse_args(argv)
    import jax
    import jax.numpy as j
    from gpu_engine import v2e_cuda
    v2e_cuda.load(args.build_dir)
    rng = np.random.default_rng(20261104)
    seeds = [0, 1, 2, 2**31 - 1, 2**31, 2**32 - 1, 2**32, 2**64 - 1, 2**64, 2**255, 2**256 - 1, (2**224 - 1), 2**32 * 7]
    seeds += [int.from_bytes(rng.bytes(32), 'little') for _ in range(args.seeds)]
    if args.recipe_seed is not None:
        from pokezero import v5_contract as c
        from pokezero.v5_gumbel import Recipe
        recipe = Recipe(args.recipe_seed)
        for _ in range(args.addresses):
            address = c.DecisionAddress(int(rng.integers(0, 2**31)), int(rng.integers(0, 400)), int(rng.integers(0, 2)))
            seeds += [recipe.world_seed(address, 0), recipe.world_seed(address, 1)]
    limits = [1, 2, 3, 7, 100, 255, 256, 257, 1000, 2**31 - 1, 2**31, 2**32 - 1]
    specs = [(seed, 0, 0) for seed in seeds] + [(seed, 1, n) for seed in seeds[:16] for n in limits]
    words = np.stack([seed_words(seed) for seed, _, _ in specs])
    spec = np.asarray([(mode, np.uint32(n).view(np.int32)) for _, mode, n in specs], np.int32).reshape(-1, 2)
    out = jax.ShapeDtypeStruct((len(specs), args.draws), np.uint32)
    clock = time.perf_counter()
    device = np.asarray(jax.ffi.ffi_call('v2e_V2eMtProbe', out)(j.asarray(words), j.asarray(spec), draws=np.int64(args.draws)))
    device_seconds = time.perf_counter() - clock
    differing = 0;compared = 0
    for row, (seed, mode, n) in enumerate(specs):
        r = random.Random(seed)
        if mode == 0:
            host = np.fromiter((r.getrandbits(32) for _ in range(args.draws)), np.uint32, args.draws)
        else:
            host = np.fromiter((r.randrange(n) for _ in range(args.draws)), np.uint32, args.draws)
        same = np.array_equal(host, device[row])
        compared += args.draws;differing += int(not same)
        if not same:
            first = int(np.flatnonzero(host != device[row])[0])
            print(json.dumps(dict(seed=str(seed), mode=mode, n=n, first_difference=first, host=int(host[first]),
                device=int(device[row][first]))), flush=True)
    print(json.dumps(dict(message='MT19937_COMPARED', sequences=len(specs), draws_compared=compared,
        differing_sequences=differing, seeds=len(seeds), device_seconds=device_seconds)), flush=True)
    return 1 if differing else 0


if __name__ == '__main__':
    sys.exit(main())
