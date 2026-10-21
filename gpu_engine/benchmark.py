"""Early fixed-root GPU core-step timings, NOT the final throughput benchmark.

Repeated one-step requests do not advance or complete games. Missing encoder,
policy, end-to-end, game-length and game-rate fields stay null. This diagnostic
cannot evaluate the handoff's go/no-go thresholds or qualify move coverage.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from .state import legal_actions
from .transition import state_from_pool, step, tables_from_metadata


def sole_gpu():
    devices = jax.devices("gpu")
    if len(devices) != 1:
        raise RuntimeError(f"exactly one visible GPU required; got {len(devices)}")
    return devices[0]


def summarize_samples(samples):
    if not samples or any(not np.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("positive finite synchronized wall-time samples required")
    return dict(engine_step_ms=statistics.fmean(samples),
                median_step_ms=statistics.median(samples), samples_ms=list(samples))


def load_bound_pool(pool_path):
    """Load and hash the same immutable byte snapshot, outside the clock."""
    with Path(pool_path).open("rb") as source:
        before = os.fstat(source.fileno())
        payload = source.read()
        after = os.fstat(source.fileno())
    identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    if identity(before) != identity(after) or len(payload) != after.st_size:
        raise RuntimeError("team pool changed while reading")
    # An atomic pathname replacement cannot change this snapshot or its digest.
    # ctime is deliberately excluded: unlinking a stable open inode changes it.
    with np.load(io.BytesIO(payload),allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata_utf8"].tobytes())
        fields = ("species","level","types","stats","moves","pp","item","ability")
        pool = {name:archive[name] for name in fields}
    return pool, metadata, hashlib.sha256(payload).hexdigest()


def render_markdown(result):
    rows = ["# Partial GPU core step timings", "",
        f"Hardware: {result['device']}; JAX {result['jax']}; backend GPU.", "",
        "Fixed initial real-team roots, repeated one-step requests. Every move is still flagged a placeholder.",
        "Synchronized wall time includes host dispatch; initialization, action sampling, compilation and warm-up are excluded.",
        "No completed games, final fidelity, move coverage or go/no-go conclusion follows.", "",
        "| Batch | Core step ms | Encode ms | Forward ms | End to end ms | Game steps per second | Games per second | Mean game turns |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in result["rows"]:
        rows.append(f"| {row['batch']} | {row['engine_step_ms']:.3f} | — | — | — | — | — | — |")
    return "\n".join(rows) + "\n"


def measure(pool_path, *, batches=(256,1024,2048,8192), repeats=100, warmup=5):
    if (not batches or any(type(b) is not int or b <= 0 for b in batches)
            or type(repeats) is not int or repeats < 2 or type(warmup) is not int or warmup < 1):
        raise ValueError("positive integer batches, >=2 repeats and >=1 warm-up required")
    device = sole_gpu()
    pool, metadata, pool_sha = load_bound_pool(pool_path)
    if (metadata.get("schema") != "pokezero.gpu-team-pool.v1"
            or metadata.get("completed_teams") != len(pool["species"])
            or len(pool["species"]) < 200000 or 2 * max(batches) > len(pool["species"])):
        raise ValueError("complete >=200000-team pool with enough distinct root teams required")
    source_hashes = {}
    for name in ("state.py","mechanics.py","effects.py","transition.py","benchmark.py"):
        with Path(__file__).with_name(name).open("rb") as source:
            source_hashes[name] = hashlib.file_digest(source,"sha256").hexdigest()
    table = tables_from_metadata(metadata)
    compiled_step = jax.jit(step)
    sample_actions = jax.jit(jax.vmap(lambda key,mask: jax.random.categorical(
        jax.random.fold_in(key,271828),jnp.where(mask,0.0,-jnp.inf)).astype(jnp.int32)))
    result = dict(schema="pokezero.gpu-partial-core-benchmark.v1",
        status="PARTIAL_CORE_FIXED_ROOTS_NOT_GAME_THROUGHPUT",
        created_utc=datetime.now(timezone.utc).isoformat(),
        device=device.device_kind, gpu_count=1, jax=jax.__version__, numpy=np.__version__,
        python=platform.python_version(), team_count=len(pool["species"]), team_pool_sha256=pool_sha,
        source_files_sha256=source_hashes,
        repeats=repeats,warmup=warmup,implemented_move_coverage=0.0,
        all_moves_placeholder=True,go_no_go="NOT_EVALUATED_INCOMPLETE_ENGINE",
        workload="Same initial team prefix across batch sizes; fixed uniform legal actions; repeated one-step requests, not games",
        timing="Synchronized whole-result wall time including host dispatch; initialization/action sampling/compilation/warm-up excluded",
        rows=[])
    for batch in batches:
        state = state_from_pool(pool,np.arange(2*batch).reshape(batch,2),seed=20261021)
        actions = sample_actions(state.keys,legal_actions(state))
        jax.block_until_ready((state,actions,table))
        first = compiled_step(state,actions,table)
        jax.block_until_ready(first)
        if first.state.hp.devices() != {device} or bool(jnp.any(first.invalid_actions)):
            raise RuntimeError("compiled step was not a valid request on the sole GPU")
        for _ in range(warmup):
            jax.block_until_ready(compiled_step(state,actions,table))
        samples = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            jax.block_until_ready(compiled_step(state,actions,table))
            samples.append((time.perf_counter_ns()-start)/1e6)
        stats = device.memory_stats() or {}
        row = dict(batch=batch,**summarize_samples(samples),
            placeholder_moves_in_one_request=int(jnp.sum(first.placeholder_moves)),
            encode_ms=None,forward_ms=None,end_to_end_ms=None,
            game_steps_per_second=None,games_per_second=None,mean_turns_per_game=None,
            gpu_utilization_percent=None,
            gpu_memory_bytes=int(stats["bytes_in_use"]) if "bytes_in_use" in stats else None)
        result["rows"].append(row)
        print(f"PARTIAL_CORE_BATCH={batch} STEP_MS={row['engine_step_ms']:.6f}",flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-pool",required=True,type=Path)
    parser.add_argument("--out-prefix",required=True,type=Path)
    parser.add_argument("--batches",nargs="+",type=int,default=[256,1024,2048,8192])
    parser.add_argument("--repeats",type=int,default=100)
    parser.add_argument("--warmup",type=int,default=5)
    args = parser.parse_args()
    paths = [Path(str(args.out_prefix)+suffix) for suffix in (".json",".md")]
    if any(path.exists() for path in paths):
        raise FileExistsError("refusing to overwrite an existing benchmark result")
    result = measure(args.team_pool,batches=tuple(args.batches),repeats=args.repeats,warmup=args.warmup)
    paths[0].parent.mkdir(parents=True,exist_ok=True)
    for path,content in zip(paths,(json.dumps(result,indent=2,allow_nan=False)+"\n",render_markdown(result))):
        with path.open("x") as output:
            output.write(content)
    print(json.dumps(dict(status=result["status"],json=str(paths[0]),markdown=str(paths[1]))))


if __name__ == "__main__":
    main()
