#!/usr/bin/env python3
"""Export the entity-token transformer checkpoint to TorchScript / ONNX.

The native search crate (docs/test_time_search_plan_v3.md, "Integration
endgame") runs model inference outside Python — TorchScript via tch-rs or
ONNX via onnxruntime. This script proves that export path for a real
checkpoint and validates numerical parity against the eager model.

The checkpoint's forward is keyword-only and returns a dataclass;
``ExportableTransformerPolicy`` wraps it with positional args and a plain
tensor-tuple output so ``torch.jit.trace`` / ``torch.onnx.export`` can
consume it. Only the expanded-observation path (``categorical_ids`` et al.)
is exported — the row-indexed training path is not needed at inference.

Usage:
    python scripts/export_model.py --checkpoint runs/.../model.pt \
        --out-dir exports/ --formats ts,onnx --validate

Outputs (in --out-dir):
    model_ts.pt        TorchScript trace (torch.jit.trace, dynamic batch)
    model.onnx         ONNX graph (dynamic batch axis)
    export_manifest.json   config summary + parity results
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

INPUT_NAMES = (
    "categorical_ids",
    "numeric_features",
    "token_type_ids",
    "attention_mask",
    "history_mask",
)
OUTPUT_NAMES = ("policy_logits", "value", "opponent_action_logits")
DEFAULT_TOLERANCE = 1e-4
DEFAULT_VALIDATE_BATCH = 64
# Traced at a batch size distinct from every validation batch so a trace
# that silently baked the batch dimension in cannot pass validation.
TRACE_BATCH = 4


def _require_torch_nn() -> tuple[Any, Any]:
    import torch
    from torch import nn

    return torch, nn


def build_exportable_module(model: Any) -> Any:
    """Wrap the keyword-only forward in a positional-args, tuple-output shim."""

    _, nn = _require_torch_nn()

    class ExportableTransformerPolicy(nn.Module):  # type: ignore[misc]
        """Positional-arg shim over EntityTokenTransformerPolicy's expanded path.

        Input order matches INPUT_NAMES; output order matches OUTPUT_NAMES.
        """

        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(
            self,
            categorical_ids: Any,
            numeric_features: Any,
            token_type_ids: Any,
            attention_mask: Any,
            history_mask: Any,
        ) -> tuple[Any, Any, Any]:
            output = self.wrapped(
                categorical_ids=categorical_ids,
                numeric_features=numeric_features,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                history_mask=history_mask,
            )
            return output.policy_logits, output.value, output.opponent_action_logits

    return ExportableTransformerPolicy(model).eval()


def make_random_inputs(config: Any, batch_size: int, *, seed: int, device: str = "cpu") -> tuple[Any, ...]:
    """Random tensors that satisfy _validate_tensor_shapes for ``config``.

    Attention masks are random with the first token forced valid so no
    (batch, window) row is fully masked; history_mask is all-True (window=1
    checkpoints always observe the current state). Values are drawn inside
    each vocabulary so the embedding clamp path stays inactive.

    window_size > 1 checkpoints would need padded-turn (history_mask False)
    validation patterns this generator does not produce yet — refuse rather
    than under-validate.
    """
    window = getattr(config, "window_size", 1)
    if window != 1:
        raise ValueError(
            f"make_random_inputs only validates window_size == 1 checkpoints (got {window}); "
            "add padded-turn history_mask patterns before exporting window>1 models."
        )

    torch, _ = _require_torch_nn()
    generator = torch.Generator().manual_seed(seed)
    shape_prefix = (batch_size, config.window_size, config.token_count)
    categorical_ids = torch.randint(
        0,
        config.categorical_vocab_size,
        (*shape_prefix, config.categorical_feature_count),
        generator=generator,
    )
    numeric_features = torch.randn(
        (*shape_prefix, config.numeric_feature_count), generator=generator
    )
    token_type_ids = torch.randint(
        0, config.token_type_vocab_size, shape_prefix, generator=generator
    )
    attention_mask = torch.rand(shape_prefix, generator=generator) > 0.2
    attention_mask[..., 0] = True
    history_mask = torch.ones(batch_size, config.window_size, dtype=torch.bool)
    tensors = (categorical_ids, numeric_features, token_type_ids, attention_mask, history_mask)
    return tuple(tensor.to(device) for tensor in tensors)


def export_torchscript(shim: Any, example_inputs: tuple[Any, ...], out_path: Path) -> Any:
    torch, _ = _require_torch_nn()
    with torch.no_grad(), warnings.catch_warnings():
        # The trace emits TracerWarnings for the shape-validation Python
        # branches in the eager forward; they are trace-time-only checks.
        warnings.simplefilter("ignore")
        traced = torch.jit.trace(shim, example_inputs)
    traced.save(str(out_path))
    return traced


def export_onnx(
    shim: Any,
    example_inputs: tuple[Any, ...],
    out_path: Path,
    *,
    exporter: str = "auto",
    opset: int = 18,
) -> str:
    """Export to ONNX with a dynamic batch axis. Returns the exporter used.

    ``auto`` prefers the torch.export-based (dynamo) exporter and falls back
    to an explicit error when the dynamo path is
    unavailable (e.g. missing onnxscript). Both produce parity-passing
    graphs for this model (see docs/model_export_findings.md).
    """

    torch, _ = _require_torch_nn()
    last_error: Exception | None = None
    # The legacy exporter cannot lower aten::_transformer_encoder_layer_fwd,
    # so "auto" means dynamo-or-error (with guidance), never a silent fallback.
    attempts = ("dynamo",) if exporter == "auto" else (exporter,)
    for attempt in attempts:
        try:
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if attempt == "dynamo":
                    batch = torch.export.Dim("batch")
                    torch.onnx.export(
                        shim,
                        example_inputs,
                        str(out_path),
                        input_names=list(INPUT_NAMES),
                        output_names=list(OUTPUT_NAMES),
                        dynamic_shapes={name: {0: batch} for name in INPUT_NAMES},
                        dynamo=True,
                    )
                else:
                    torch.onnx.export(
                        shim,
                        example_inputs,
                        str(out_path),
                        input_names=list(INPUT_NAMES),
                        output_names=list(OUTPUT_NAMES),
                        dynamic_axes={name: {0: "batch"} for name in INPUT_NAMES},
                        opset_version=opset,
                        dynamo=False,
                    )
            return attempt
        except Exception as error:  # noqa: BLE001 - report the exporter that failed.
            last_error = error
            print(f"ONNX export via {attempt} exporter failed: {error}", file=sys.stderr)
    raise RuntimeError(f"All ONNX export attempts failed: {last_error}") from last_error


def _max_abs_diff(reference: Any, candidate: Any) -> float:
    import numpy as np

    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    return float(np.abs(reference - candidate).max())


def _eager_reference(shim: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    torch, _ = _require_torch_nn()
    with torch.no_grad():
        return tuple(tensor.detach().cpu().numpy() for tensor in shim(*inputs))


def validate_torchscript(shim: Any, traced: Any, config: Any, *, seed: int, batch_size: int) -> dict[str, float]:
    torch, _ = _require_torch_nn()
    diffs: dict[str, float] = {}
    for batch in (batch_size, 1):
        inputs = make_random_inputs(config, batch, seed=seed + batch)
        reference = _eager_reference(shim, inputs)
        with torch.no_grad():
            produced = tuple(tensor.detach().cpu().numpy() for tensor in traced(*inputs))
        for name, ref, got in zip(OUTPUT_NAMES, reference, produced):
            key = f"{name}@batch{batch}"
            diffs[key] = _max_abs_diff(ref, got)
    return diffs


def validate_onnx(shim: Any, onnx_path: Path, config: Any, *, seed: int, batch_size: int) -> dict[str, float]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    diffs: dict[str, float] = {}
    for batch in (batch_size, 1):
        inputs = make_random_inputs(config, batch, seed=seed + batch)
        reference = _eager_reference(shim, inputs)
        feeds = {name: tensor.numpy() for name, tensor in zip(INPUT_NAMES, inputs)}
        produced = session.run(None, feeds)
        for name, ref, got in zip(OUTPUT_NAMES, reference, produced):
            key = f"{name}@batch{batch}"
            diffs[key] = _max_abs_diff(ref, got)
    return diffs


def _parity_verdict(diffs: dict[str, float], tolerance: float) -> tuple[float, bool]:
    """Binding parity = policy_logits + value (opponent head is informational)."""

    binding = [
        value
        for key, value in diffs.items()
        if key.startswith(("policy_logits", "value"))
    ]
    worst = max(binding)
    return worst, worst <= tolerance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to a load_transformer_checkpoint-compatible .pt file.")
    parser.add_argument("--out-dir", required=True, help="Directory for exported artifacts.")
    parser.add_argument("--formats", default="ts,onnx", help="Comma-separated subset of {ts,onnx}.")
    parser.add_argument("--validate", action="store_true", help="Compare exports against eager on random inputs.")
    parser.add_argument("--validate-batch", type=int, default=DEFAULT_VALIDATE_BATCH, help="Random inputs used for parity (default 64).")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="Max abs diff allowed on policy logits + value.")
    parser.add_argument("--seed", type=int, default=20260718, help="Base RNG seed for trace/validation inputs.")
    parser.add_argument("--onnx-exporter", choices=("auto", "dynamo", "legacy"), default="auto")
    args = parser.parse_args(argv)

    formats = [item.strip() for item in args.formats.split(",") if item.strip()]
    unknown = sorted(set(formats) - {"ts", "onnx"})
    if unknown or not formats:
        parser.error(f"--formats must be a non-empty subset of ts,onnx (got {args.formats!r}).")

    from pokezero.neural_policy import load_transformer_checkpoint

    model, result = load_transformer_checkpoint(args.checkpoint, map_location="cpu")
    config = result.model_config
    model.eval()
    shim = build_exportable_module(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Loaded checkpoint: {parameter_count / 1e6:.1f}M params, d={config.embedding_dim}, "
          f"layers={config.transformer_layers}, window={config.window_size}, tokens={config.token_count}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    example_inputs = make_random_inputs(config, TRACE_BATCH, seed=args.seed)

    manifest: dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "parameter_count": parameter_count,
        "input_names": list(INPUT_NAMES),
        "output_names": list(OUTPUT_NAMES),
        "input_shapes": {
            "categorical_ids": ["batch", config.window_size, config.token_count, config.categorical_feature_count],
            "numeric_features": ["batch", config.window_size, config.token_count, config.numeric_feature_count],
            "token_type_ids": ["batch", config.window_size, config.token_count],
            "attention_mask": ["batch", config.window_size, config.token_count],
            "history_mask": ["batch", config.window_size],
        },
        "tolerance": args.tolerance,
        "formats": {},
    }

    failures: list[str] = []
    if "ts" in formats:
        ts_path = out_dir / "model_ts.pt"
        traced = export_torchscript(shim, example_inputs, ts_path)
        entry: dict[str, Any] = {"path": str(ts_path)}
        print(f"TorchScript written: {ts_path}")
        if args.validate:
            diffs = validate_torchscript(shim, traced, config, seed=args.seed + 1, batch_size=args.validate_batch)
            worst, passed = _parity_verdict(diffs, args.tolerance)
            entry.update({"parity": diffs, "parity_max_abs_diff": worst, "parity_pass": passed})
            print(f"TorchScript parity max abs diff (policy+value): {worst:.3e} -> {'PASS' if passed else 'FAIL'}")
            if not passed:
                failures.append(f"torchscript parity {worst:.3e} > {args.tolerance}")
        manifest["formats"]["torchscript"] = entry

    if "onnx" in formats:
        onnx_path = out_dir / "model.onnx"
        exporter_used = export_onnx(shim, example_inputs, onnx_path, exporter=args.onnx_exporter)
        entry = {"path": str(onnx_path), "exporter": exporter_used}
        print(f"ONNX written via {exporter_used} exporter: {onnx_path}")
        if args.validate:
            diffs = validate_onnx(shim, onnx_path, config, seed=args.seed + 1, batch_size=args.validate_batch)
            worst, passed = _parity_verdict(diffs, args.tolerance)
            entry.update({"parity": diffs, "parity_max_abs_diff": worst, "parity_pass": passed})
            print(f"ONNX parity max abs diff (policy+value): {worst:.3e} -> {'PASS' if passed else 'FAIL'}")
            if not passed:
                failures.append(f"onnx parity {worst:.3e} > {args.tolerance}")
        manifest["formats"]["onnx"] = entry

    manifest_path = out_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest written: {manifest_path}")

    if failures:
        print("PARITY FAILURE: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
