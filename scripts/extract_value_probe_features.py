#!/usr/bin/env python3
"""Extract frozen FP32 champion tokens from an admitted saved A1 cohort.

Run beside the remote caches and checkpoint. Full features and membership stay
there. This does not collect games, fit heads, resume partial outputs or replace
unresolved zero returns with draws. Refit holdout is not historical trunk holdout.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pokezero.value_precision_probe import (battle_split, checkpoint_contract, frozen_token_features,
                                           outcome_labels, validate_cache_metadata)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def _model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([name, array.shape, array.dtype.str]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def extract(manifest_path, checkpoint, cache_root, output, *, manifest_sha256, batch_size=512, device="cpu"):
    """Stream every selected row, including label-unknown coverage, in FP32."""
    from pokezero.neural_policy import (EntityTokenTransformerPolicy, parse_transformer_model_config,
                                       require_torch)
    torch = require_torch()
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("positive extraction batch size required")
    manifest_path, checkpoint, cache_root, output = map(Path, (manifest_path, checkpoint, cache_root, output))
    if output.exists(): raise ValueError("new feature output directory required; no partial resume")
    blob = manifest_path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != manifest_sha256:
        raise ValueError("cohort manifest binding mismatch")
    cohort = json.loads(blob)
    if cohort.get("schema") != "pokezero.value_probe_cohort.v1":
        raise ValueError("frozen value-probe cohort required")
    contract = checkpoint_contract(checkpoint)
    if cohort["contract"] != contract or contract["value_calibration_transform"] is not None:
        raise ValueError("exact uncalibrated champion/cohort contract required")
    checkpoint_bytes = checkpoint.read_bytes()
    if hashlib.sha256(checkpoint_bytes).hexdigest() != contract["checkpoint_sha256"]:
        raise ValueError("checkpoint changed before deserialization")
    payload = torch.load(io.BytesIO(checkpoint_bytes), weights_only=True, map_location="cpu")
    model = EntityTokenTransformerPolicy(parse_transformer_model_config(payload))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.float().to(device).eval()
    if str(device).startswith("cuda"):
        # FP32 extraction is distinct from A3's separately gated bf16 path.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    before_model = _model_digest(model)
    root = cache_root.resolve(strict=True)
    if output.resolve().is_relative_to(root):
        raise ValueError("feature output must be outside the read-only cache root")
    paths = [Path(s["path"]).resolve(strict=True) for s in cohort["shards"]]
    if len(paths) != len(set(paths)) or any(not p.is_relative_to(root) for p in paths):
        raise ValueError("cohort shard paths must be unique and inside admitted cache root")
    if not paths or sum(s["counts"]["rows"] for s in cohort["shards"]) != cohort["selected_examples"]:
        raise ValueError("cohort row coverage mismatch")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"schema": "pokezero.value_probe_features.v1", "cohort_manifest_sha256": manifest_sha256,
        "checkpoint_contract": contract, "source_model_state_sha256": before_model,
        "dtype": "float32", "device": str(device), "batch_size": batch_size, "shards": [],
        "lineage": cohort["lineage"], "split_salt": cohort["split_salt"],
        "label_rule": cohort["label_rule"], "prior_exposure": cohort["prior_exposure"],
        "policy_distribution": cohort["policy_distribution"], "new_games": 0, "model_fits": 0}
    totals, split_totals = Counter(), Counter()
    for index, (source, path) in enumerate(zip(cohort["shards"], paths)):
        names = ("numeric_features", "categorical_ids", "token_type_ids", "attention_mask",
                 "window_indices", "returns", "terminal_capped", "seeds", "turn_indices")
        required = {"metadata.json", *(name+".npy" for name in names)}
        if not required.issubset(source["hashes"]):
            raise ValueError("every consumed cache member requires a frozen hash")
        identities = {}
        for name, expected in source["hashes"].items():
            if Path(name).name != name: raise ValueError("unsafe cohort member filename")
            member = path / name
            stat = member.stat(); identities[name] = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if digest_file(member) != expected: raise ValueError("frozen source hash changed: "+name)
        metadata = json.loads((path / "metadata.json").read_text())
        if metadata != source["metadata"]: raise ValueError("frozen metadata mismatch")
        validate_cache_metadata(metadata, contract)
        arrays = {name: np.load(path/(name+".npy"), mmap_mode="r", allow_pickle=False) for name in names}
        rows = metadata["example_count"]
        if rows != source["counts"]["rows"] or not np.array_equal(arrays["window_indices"], np.arange(1, rows+1)[:, None]):
            raise ValueError("frozen example-to-observation mapping mismatch")
        labels, status = outcome_labels(arrays["returns"], arrays["terminal_capped"], metadata["dataset_config"])
        counts = dict(rows=rows, known=int((status==0).sum()), capped=int((status==1).sum()),
                      unresolved_zero=int((status==2).sum()))
        if counts != source["counts"]: raise ValueError("frozen label coverage mismatch")
        split = np.array([{"train":0, "validation":1, "test":2}[battle_split(int(seed),
            lineage=cohort["lineage"], salt=cohort["split_salt"])] for seed in arrays["seeds"]], np.uint8)
        destination = output/f"shard-{index:05d}"; destination.mkdir()
        features = np.lib.format.open_memmap(destination/"tokens.npy", mode="w+", dtype=np.float32,
                                             shape=(rows, 23, contract["width"]))
        baseline = np.lib.format.open_memmap(destination/"champion_values.npy", mode="w+", dtype=np.float32, shape=(rows,))
        for start in range(0, rows, batch_size):
            end = min(rows, start+batch_size)
            selected = arrays["window_indices"][start:end, 0]
            tensors = {name: torch.as_tensor(np.array(arrays[name][selected], copy=True),
                dtype=torch.bool if name=="attention_mask" else torch.float32 if name=="numeric_features" else torch.long,
                device=device)[:, None] for name in names[:4]}
            tensors["history_mask"] = torch.ones((end-start, 1), dtype=torch.bool, device=device)
            tokens, prediction = frozen_token_features(model, tensors)
            if (tokens.dtype != torch.float32 or prediction.value.dtype != torch.float32
                    or not torch.isfinite(tokens).all() or not torch.isfinite(prediction.value).all()):
                raise ValueError("finite FP32 frozen features and values required")
            features[start:end] = tokens.cpu().numpy()
            baseline[start:end] = prediction.value.cpu().numpy()
        features.flush(); baseline.flush(); del features, baseline
        extra = {"attention_mask": np.array(arrays["attention_mask"][1:], copy=True), "outcomes": labels,
            "label_status": status, "split": split, "seeds": np.array(arrays["seeds"], copy=True),
            "turn_indices": np.array(arrays["turn_indices"], copy=True)}
        for name, array in extra.items():
            with (destination/(name+".npy")).open("xb") as stream: np.save(stream, array, allow_pickle=False)
        for name, identity in identities.items():
            stat=(path/name).stat()
            if (stat.st_ino, stat.st_size, stat.st_mtime_ns) != identity:
                raise ValueError("source changed during feature extraction")
        hashes = {p.name:digest_file(p) for p in sorted(destination.glob("*.npy"))}
        result["shards"].append({"path":str(destination.resolve()), "source_path":str(path),
            "source_hashes":source["hashes"], "hashes":hashes, "counts":counts})
        totals.update(counts)
        for code, name in enumerate(("train", "validation", "test")): split_totals[name]+=int((split==code).sum())
        print(json.dumps({"event":"feature_shard_completed", "shard":index, "rows":totals["rows"]}), flush=True)
    if totals["rows"] != cohort["selected_examples"] or _model_digest(model) != before_model:
        raise ValueError("row coverage or frozen model state changed")
    result.update(counts=dict(totals), split_rows=dict(split_totals), elapsed_seconds=time.monotonic()-started,
        status="FROZEN_FP32_FEATURES_COMPLETE_LABEL_ZEROS_UNRESOLVED", trunk_unchanged=True,
        top_two_resolution="UNMEASURED_REQUIRES_ACTUAL_SIBLING_OBSERVATIONS")
    with (output/"features-manifest.json").open("x") as stream: json.dump(result,stream,indent=2,sort_keys=True);stream.write("\n")
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "checkpoint", "cache-root", "output"): parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--manifest-sha256",required=True)
    parser.add_argument("--batch-size",type=int,default=512)
    parser.add_argument("--device",default="cpu")
    args=parser.parse_args()
    result=extract(args.manifest,args.checkpoint,args.cache_root,args.output,
        manifest_sha256=args.manifest_sha256,batch_size=args.batch_size,device=args.device)
    print(json.dumps({"status":result["status"],"counts":result["counts"],"new_games":0,"model_fits":0}),flush=True)
