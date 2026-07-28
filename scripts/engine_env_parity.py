#!/usr/bin/env python
"""Record-schema parity gate: engine-env vs Showdown-env training caches.

Collects the SAME games through both backends and checks that the resulting
training-cache shards are interchangeable to the trainer. Values will differ —
the two backends realize different chance outcomes and the engine backend
carries degenerate belief columns — so this gate deliberately checks structure
and ingestion, never equality.

The binding check is the last one: :func:`concat_training_caches` is the exact
compatibility contract the fleet uses to merge shards, so if a Showdown shard
and an engine shard concatenate and then read back through
:func:`iter_training_cache_batches`, the trainer cannot tell them apart.

Note on ``categorical_ids`` width: the writer stores categorical rows in
``compact-nonzero`` mode, whose stored width is the widest nonzero row IN THAT
SHARD — data-dependent, not schema. The semantic width is
``categorical_storage.original_feature_count``, and that is what must match
(``concat_training_caches`` pads the narrower part with zeros).

Usage:
    python scripts/engine_env_parity.py --games 5 --out <workdir>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pokezero.dataset import (  # noqa: E402
    _CONCAT_COMPAT_FIELDS,
    concat_training_caches,
    iter_training_cache_batches,
)
from pokezero.rollout_cli import main as rollout_main  # noqa: E402

# Per-shard compaction artifacts, not schema. See the module docstring.
_COMPACTION_KEYS = ("observation_shapes", "categorical_storage")


def _collect(backend: str, out: Path, games: int, seed_start: int, showdown_root: Path | None) -> None:
    if out.exists():
        shutil.rmtree(out)
    argv = [
        "collect-selfplay-training-cache",
        "--games", str(games),
        "--out", str(out),
        "--env", backend,
        "--transition-token-budget", "0",
        "--seed-start", str(seed_start),
        "--current-policy", "random-legal",
    ]
    if showdown_root is not None:
        argv += ["--showdown-root", str(showdown_root)]
    code = rollout_main(argv)
    if code != 0:
        raise SystemExit(f"{backend} collection failed with exit code {code}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--showdown-root", type=Path, default=None)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    engine_dir = args.out / "cache-engine"
    showdown_dir = args.out / "cache-showdown"
    print(f"collecting {args.games} games through each backend (seed-start {args.seed_start})")
    _collect("engine", engine_dir, args.games, args.seed_start, args.showdown_root)
    _collect("showdown", showdown_dir, args.games, args.seed_start, args.showdown_root)

    failures: list[str] = []

    # 1. Array set.
    engine_arrays = {p.name for p in engine_dir.glob("*.npy")}
    showdown_arrays = {p.name for p in showdown_dir.glob("*.npy")}
    if engine_arrays != showdown_arrays:
        failures.append(
            f"array set differs: engine-only={sorted(engine_arrays - showdown_arrays)} "
            f"showdown-only={sorted(showdown_arrays - engine_arrays)}"
        )
    print(f"\n[1] array set: {len(engine_arrays)} arrays, "
          f"{'identical' if engine_arrays == showdown_arrays else 'DIFFERS'}")

    # 2. Per-array dtype and trailing dimensions (leading dim = example count,
    #    which legitimately differs: the two backends play different games).
    print(f"\n[2] {'array':28s} {'dtype':10s} {'engine shape':18s} {'showdown shape':18s} verdict")
    print("    " + "-" * 92)
    for name in sorted(engine_arrays & showdown_arrays):
        a = numpy.load(engine_dir / name, mmap_mode="r")
        b = numpy.load(showdown_dir / name, mmap_mode="r")
        problems = []
        if a.dtype != b.dtype:
            problems.append(f"dtype {a.dtype} != {b.dtype}")
        if a.ndim != b.ndim:
            problems.append(f"ndim {a.ndim} != {b.ndim}")
        if name == "categorical_ids.npy":
            # Trailing dim is the compaction width; the semantic width is
            # checked via metadata below.
            if a.shape[1:-1] != b.shape[1:-1]:
                problems.append(f"token dims {a.shape[1:-1]} != {b.shape[1:-1]}")
        elif a.shape[1:] != b.shape[1:]:
            problems.append(f"trailing dims {a.shape[1:]} != {b.shape[1:]}")
        verdict = "OK" if not problems else "MISMATCH: " + "; ".join(problems)
        failures.extend([f"{name}: {p}" for p in problems])
        print(f"    {name:28s} {str(a.dtype):10s} {str(a.shape):18s} {str(b.shape):18s} {verdict}")

    # 3. Metadata: the fleet's own concat-compatibility contract.
    engine_meta = json.loads((engine_dir / "metadata.json").read_text())
    showdown_meta = json.loads((showdown_dir / "metadata.json").read_text())
    print("\n[3] metadata")
    if set(engine_meta) != set(showdown_meta):
        failures.append("metadata key set differs")
    print(f"    {'key set':34s} {'identical' if set(engine_meta) == set(showdown_meta) else 'DIFFERS'}")
    for field in (*_CONCAT_COMPAT_FIELDS, "array_dtypes"):
        same = engine_meta.get(field) == showdown_meta.get(field)
        if not same:
            failures.append(f"metadata.{field} differs")
        print(f"    {field:34s} {'identical' if same else 'DIFFERS'}")
        if not same:
            print(f"      engine  : {json.dumps(engine_meta.get(field))[:300]}")
            print(f"      showdown: {json.dumps(showdown_meta.get(field))[:300]}")
    for field in ("mode", "original_feature_count"):
        a = engine_meta.get("categorical_storage", {}).get(field)
        b = showdown_meta.get("categorical_storage", {}).get(field)
        if a != b:
            failures.append(f"categorical_storage.{field} differs ({a!r} != {b!r})")
        print(f"    categorical_storage.{field:14s} {'identical' if a == b else 'DIFFERS'} ({a!r} / {b!r})")
    for field in _COMPACTION_KEYS:
        print(f"    {field:34s} per-shard compaction (not compared)")

    # 4. Value ranges — plausibility, never equality.
    print("\n[4] value ranges")
    for name in (
        "categorical_ids", "numeric_features", "token_type_ids",
        "attention_mask", "legal_action_mask", "returns", "rewards", "value_estimates",
    ):
        path = f"{name}.npy"
        if path not in engine_arrays:
            continue
        a = numpy.load(engine_dir / path).astype("float64")
        b = numpy.load(showdown_dir / path).astype("float64")
        print(f"    {name:20s} engine[{a.min():9.4f},{a.max():9.4f}] mean={a.mean():9.4f}   "
              f"showdown[{b.min():9.4f},{b.max():9.4f}] mean={b.mean():9.4f}")
        if not numpy.isfinite(a).all():
            failures.append(f"{name}: engine shard contains non-finite values")
        # A range the Showdown shard never reaches would mean the engine
        # encoder is writing out-of-vocabulary / out-of-scale values.
        if name in {"categorical_ids", "token_type_ids"} and a.max() > b.max() * 4 + 4:
            failures.append(f"{name}: engine max {a.max()} implausible vs showdown {b.max()}")

    # 5. k = 0: the transition region must be entirely unattended in both.
    print("\n[5] k=0 masking")
    for label, directory in (("engine", engine_dir), ("showdown", showdown_dir)):
        attention = numpy.load(directory / "attention_mask.npy")
        attended = bool(attention[:, 23:].any())
        if attended:
            failures.append(f"{label}: transition region attended under --transition-token-budget 0")
        print(f"    {label:10s} transition region unattended: {not attended}")

    # 6. The binding check: do they merge, and does the merge read back?
    print("\n[6] concat + read-back (the trainer's own compatibility contract)")
    merged = args.out / "cache-merged"
    if merged.exists():
        shutil.rmtree(merged)
    try:
        concat_training_caches([engine_dir, showdown_dir], merged)
        batches = list(iter_training_cache_batches(merged, batch_size=64))
        rows = sum(int(batch.categorical_ids.shape[0]) for batch in batches)
        print(f"    concatenated OK -> {merged.name}; read back {len(batches)} batches, {rows} examples")
        if rows == 0:
            failures.append("merged cache read back zero examples")
    except Exception as exc:  # noqa: BLE001 — the gate reports, never masks
        failures.append(f"concat/read-back failed: {exc}")
        print(f"    FAILED: {exc}")

    print("\n" + "=" * 70)
    if failures:
        print("RESULT: SCHEMA PARITY FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("RESULT: SCHEMA PARITY OK")
    print("  Identical array set, dtypes, token dimensions, dataset config, feature masks")
    print("  and categorical semantics; the two shards concatenate and read back through")
    print("  the trainer's own loader. Values differ (different games, degenerate belief")
    print("  columns on the engine side) — by design.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
