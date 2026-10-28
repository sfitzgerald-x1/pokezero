#!/usr/bin/env python3
"""Freeze whole saved iterations and battle-separated A1 membership remotely.

This reads existing caches; it never collects games or fits a model. Output is a
manifest of source hashes plus a compact audit summary. Keep both beside the
remote data. Zero-return labels remain unresolved, not silently counted as draws.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pokezero.value_precision_probe import battle_split, outcome_labels, validate_cache_metadata


def digest_file(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"source changed during hashing: {path.name}")
    return digest.hexdigest()


def freeze(root: Path, output: Path, contract: dict, *, target: int, lineage: str,
           selection_salt: str, split_salt: str) -> dict:
    if target <= 0 or output.exists() or not lineage or not selection_salt or not split_salt:
        raise ValueError("positive target, new output directory, lineage and salts required")
    iterations = defaultdict(list)
    ineligible = Counter()
    ineligible_iterations = set()
    for shard_path in sorted(root.glob("iteration-*/shard-*")):
        if not shard_path.is_dir():
            continue
        metadata_path = shard_path / "metadata.json"
        if not metadata_path.is_file():
            ineligible["missing cache metadata"] += 1
            ineligible_iterations.add(shard_path.parent.name)
            continue
        metadata = json.loads(metadata_path.read_text())
        # Eligibility uses metadata only, before labels are inspected.
        try:
            validate_cache_metadata(metadata, contract)
        except ValueError as exc:
            ineligible[str(exc)] += 1
            ineligible_iterations.add(shard_path.parent.name)
            continue
        count = metadata.get("example_count")
        if type(count) is not int or count <= 0:
            raise ValueError("positive cache example count required")
        iterations[metadata_path.parent.parent.name].append((metadata_path.parent, metadata))
    for name in ineligible_iterations:
        iterations.pop(name, None)
    order = sorted(iterations, key=lambda key: hashlib.sha256(
        f"{selection_salt}\0{key}".encode()).hexdigest())
    selected = []
    count = 0
    for name in order:
        selected.append(name)
        count += sum(metadata["example_count"] for _, metadata in iterations[name])
        if count >= target:
            break
    if count < target:
        raise ValueError("insufficient saved examples")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "pokezero.value_probe_cohort.v1", "lineage": lineage,
                "target_examples": target, "selected_examples": count,
                "selection_salt": selection_salt, "split_salt": split_salt,
                "selection_unit": "whole retained iteration ordered by label-independent SHA256",
                "split_unit": "lineage plus game seed across all selected iterations and shards",
                "selected_iterations": selected, "contract": contract, "shards": [],
                "ineligible_metadata_counts": dict(ineligible),
                "ineligible_iterations": sorted(ineligible_iterations),
                "label_rule": "sign of unshaped discounted returns; caps excluded; zeros unresolved",
                "prior_exposure": "Held out from head refit only; frozen trunk may have trained on these caches.",
                "policy_distribution": "Historical selfplay outcomes, not champion-policy counterfactual values."}
    totals = Counter()
    splits = {key: Counter() for key in ["train", "validation", "test"]}
    split_cache = {}
    started = time.monotonic()
    for name in selected:
        for path, metadata in iterations[name]:
            if json.loads((path / "metadata.json").read_text()) != metadata:
                raise ValueError("metadata changed after eligibility selection")
            files = sorted(path.glob("*.npy"))
            required = {key + ".npy" for key in ["numeric_features", "categorical_ids", "token_type_ids",
                        "attention_mask", "returns", "terminal_capped", "seeds", "turn_indices", "window_indices"]}
            if not required.issubset({item.name for item in files}):
                raise ValueError("missing observation or label arrays")
            identities = {item: (item.stat().st_ino, item.stat().st_size, item.stat().st_mtime_ns)
                          for item in [path / "metadata.json", *files]}
            hashes = {item.name: digest_file(item) for item in [path / "metadata.json", *files]}
            arrays = {key: np.load(path / (key + ".npy"), mmap_mode="r", allow_pickle=False)
                      for key in ["returns", "terminal_capped", "seeds", "turn_indices", "window_indices"]}
            rows = metadata["example_count"]
            for key in ["returns", "terminal_capped", "seeds", "turn_indices"]:
                if arrays[key].shape != (rows,):
                    raise ValueError(f"cache array shape mismatch: {key}")
            if arrays["seeds"].dtype.kind not in "iu" or arrays["turn_indices"].dtype.kind not in "iu":
                raise ValueError("integer seed/turn arrays required")
            if arrays["window_indices"].shape != (rows, 1):
                raise ValueError("one window index per saved example required")
            observations = {key: np.load(path / (key + ".npy"), mmap_mode="r", allow_pickle=False)
                            for key in ["numeric_features", "categorical_ids", "token_type_ids", "attention_mask"]}
            stored_rows = observations["numeric_features"].shape[0]
            if stored_rows != rows + 1:
                raise ValueError("one stored row per example plus padding required")
            for key, array in observations.items():
                if array.shape != (stored_rows, *metadata["observation_shapes"][key]):
                    raise ValueError(f"observation array shape mismatch: {key}")
            if observations["numeric_features"].dtype.kind != "f" or not np.isfinite(observations["numeric_features"]).all():
                raise ValueError("nonfinite numeric observations")
            for key, upper in [("categorical_ids", contract["categorical_vocab_size"]),
                               ("token_type_ids", contract["token_type_vocab_size"])]:
                array = observations[key]
                if array.dtype.kind not in "iu" or (array < 0).any() or (array >= upper).any():
                    raise ValueError(f"invalid checkpoint embedding IDs: {key}")
            if observations["attention_mask"].dtype != np.bool_:
                raise ValueError("boolean observation attention required")
            indices = arrays["window_indices"]
            if indices.dtype.kind not in "iu" or (indices < 1).any() or (indices >= stored_rows).any():
                raise ValueError("invalid saved observation row indices")
            if not np.array_equal(indices[:, 0], np.arange(1, rows + 1)):
                raise ValueError("one-frame builder requires exact per-example row mapping")
            if not observations["attention_mask"][indices[:, 0]].any(axis=-1).all():
                raise ValueError("empty attended observation")
            labels, status = outcome_labels(arrays["returns"], arrays["terminal_capped"], metadata["dataset_config"])
            for seed in np.unique(arrays["seeds"]):
                seed = int(seed)
                split_cache.setdefault(seed, battle_split(seed, lineage=lineage, salt=split_salt))
            split_codes = np.array([split_cache[int(seed)] for seed in arrays["seeds"]])
            row_counts = Counter({"rows": rows, "known": int((status == 0).sum()),
                                  "capped": int((status == 1).sum()),
                                  "unresolved_zero": int((status == 2).sum())})
            totals.update(row_counts)
            for split, counts in splits.items():
                mask = split_codes == split
                counts.update({"rows": int(mask.sum()), "known": int(((status == 0) & mask).sum()),
                               "capped": int(((status == 1) & mask).sum()),
                               "unresolved_zero": int(((status == 2) & mask).sum()),
                               "wins": int(((labels == 1) & mask).sum()),
                               "losses": int(((labels == -1) & mask).sum())})
            manifest["shards"].append({"path": str(path), "iteration": name,
                                        "hashes": hashes, "counts": dict(row_counts),
                                        "metadata": metadata})
            if any((item.stat().st_ino, item.stat().st_size, item.stat().st_mtime_ns) != identity
                   for item, identity in identities.items()):
                raise ValueError("source changed during cohort audit")
        print(json.dumps({"event": "iteration_audited", "iteration": name,
                          "rows_so_far": totals["rows"]}), flush=True)
    assert totals["rows"] == count
    manifest["counts"] = dict(totals)
    manifest["split_counts"] = {key: dict(value) for key, value in splits.items()}
    manifest["game_seed_count"] = len(split_cache)
    manifest["elapsed_seconds"] = time.monotonic() - started
    payload = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    with (output / "cohort-manifest.json").open("xb") as stream:
        stream.write(payload)
    summary = {key: manifest[key] for key in ["schema", "lineage", "target_examples", "selected_examples",
               "selected_iterations", "counts", "split_counts", "game_seed_count", "elapsed_seconds",
               "label_rule", "prior_exposure", "policy_distribution"]}
    summary.update({"manifest_sha256": hashlib.sha256(payload).hexdigest(),
                    "manifest_bytes": len(payload), "status": "COHORT_FROZEN_LABEL_ZEROS_UNRESOLVED",
                    "terminal_provenance_complete": False, "new_games": 0, "model_fits": 0})
    with (output / "cohort-summary.json").open("x") as stream:
        json.dump(summary, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--target", type=int, default=500000)
    parser.add_argument("--lineage", required=True)
    parser.add_argument("--selection-salt", required=True)
    parser.add_argument("--split-salt", required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.root, args.output, json.loads(args.contract.read_text()),
                           target=args.target, lineage=args.lineage,
                           selection_salt=args.selection_salt, split_salt=args.split_salt)), flush=True)
