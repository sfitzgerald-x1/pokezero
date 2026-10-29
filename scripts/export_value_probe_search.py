#!/usr/bin/env python3
"""Export matched FP32 champion/refit Rust leaves; no games or search calls.

Select A1 architecture by validation MSE only. Preserve original checkpoint,
feature lineage and all three output meanings. Export is preparation, never A4
admission or strength evidence. Large models remain on the remote output path.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pokezero.neural_policy import load_transformer_checkpoint, require_torch
from pokezero.engine_search import _fence_calibration_seam
from pokezero.value_precision_probe import checkpoint_contract
from pokezero.value_head_export import (
    build_value_head_export, load_selected_head, sha256_file, validate_export,
)
from export_model import export_torchscript, make_random_inputs


def export_pair(checkpoint, summary, output, *, checkpoint_sha256, summary_sha256,
                device="cpu", tolerance=1e-5, real_inputs=None):
    torch = require_torch()
    if device not in {"cpu", "cuda"} or not 0 < tolerance <= 1e-5:
        raise ValueError("FP32 CPU/CUDA and tolerance at most 1e-5 required")
    contract = checkpoint_contract(checkpoint)
    if contract["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("champion checkpoint bytes mismatch")
    # Rust maps raw value to probability and cannot apply checkpoint calibration.
    _fence_calibration_seam({"value_calibration_transform": contract["value_calibration_transform"]}, "export checkpoint")
    head, selection = load_selected_head(summary, summary_sha256=summary_sha256,
                                         checkpoint_contract=contract)
    output = Path(output)
    if output.exists():
        raise ValueError("fresh export output required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, result = load_transformer_checkpoint(str(checkpoint), map_location="cpu")
    model = model.float().to(device).eval()
    head = head.float().to(device).eval()
    output.mkdir(parents=True, exist_ok=False)
    # Test distinct batch sizes after trace; validate masks on randomized inputs.
    batches = [make_random_inputs(result.model_config, size, seed=20261029+size, device=device)
               for size in (1, 7, 64)]
    real_binding = None
    if real_inputs is not None:
        import numpy as np
        # A caller freezes the saved nonholdout observation file externally.
        real_binding = {"path": str(real_inputs), "sha256": sha256_file(real_inputs)}
        with np.load(real_inputs, allow_pickle=False) as arrays:
            names = ("categorical_ids", "numeric_features", "token_type_ids", "attention_mask", "history_mask")
            if set(arrays.files) != set(names):
                raise ValueError("exact five saved observation tensors required")
            real = tuple(torch.from_numpy(arrays[name].copy()).to(device) for name in names)
        if real[4].dtype != torch.bool or not real[4].all():
            raise ValueError("one-frame saved observations require attended history")
        if real[0].shape[0] < 20:
            raise ValueError("at least 20 saved observations required")
        batches.append(real)
    manifest = {"schema": "pokezero.value_probe_search_export.v1", "checkpoint_contract": contract,
        "selection": selection, "device": device, "precision": "float32",
        "tf32": False, "tolerance": tolerance, "formats": {},
        "export_source_sha256": {str(path.name): sha256_file(path) for path in (
            Path(__file__), Path(__file__).with_name("export_model.py"),
            Path(__file__).resolve().parents[1] / "src/pokezero/value_head_export.py")},
        "real_saved_inputs": real_binding, "new_games": 0, "search_calls": 0,
        "root_and_opponent_checkpoint": "unchanged original champion",
        "native_interface": "five positional window tensors -> policy_logits,value,opponent_action_logits",
        "admission": "NOT_ADMITTED_REQUIRES_NATIVE_COST_CAPTURE_AND_INDEPENDENT_PROSPECTIVE_REVIEW"}
    for arm, selected in (("champion", None), ("refit", head)):
        module = build_value_head_export(model, selected).to(device).eval()
        path = output / (arm + "_ts_" + device + ".pt")
        traced = export_torchscript(module, make_random_inputs(result.model_config, 4,
            seed=20261029, device=device), path)
        # Reload from disk: validate actual artifact, rather than in-memory trace.
        traced = torch.jit.load(str(path), map_location=device).eval()
        parity = [validate_export(module, traced, model, selected, batch, tolerance=tolerance)
                  for batch in batches]
        manifest["formats"][arm] = {"path": str(path), "sha256": sha256_file(path),
            "parity_batches": [int(batch[0].shape[0]) for batch in batches], "parity": parity}
    if (checkpoint_contract(checkpoint) != contract or sha256_file(summary) != summary_sha256
            or sha256_file(Path(summary).parent / (selection["kind"] + ".pt")) != selection["head_sha256"]):
        raise ValueError("export inputs changed")
    with (output / "export-manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--real-inputs", type=Path)
    args = parser.parse_args()
    result = export_pair(args.checkpoint, args.summary, args.output,
        checkpoint_sha256=args.checkpoint_sha256, summary_sha256=args.summary_sha256,
        device=args.device, real_inputs=args.real_inputs)
    print(json.dumps({"status": "MATCHED_FP32_EXPORT_COMPLETE_NOT_ADMITTED", "new_games": 0,
                      "selection": result["selection"], "models": result["formats"]}), flush=True)
