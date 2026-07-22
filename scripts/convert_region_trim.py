#!/usr/bin/env python
"""Region-trim converter + parity gate + forward-pass benchmark.

Physically trims a checkpoint's transition-token region (e.g. 128 rows -> 16)
after the model trained its whole life under a ``transition_token_budget``
mask at or below the target. Conversion changes ZERO weights — masked rows were never
attention keys, never pooled, and had no gradient path, so the checkpoint is
already the small-region model on a larger canvas; the only edits are the
stamped ``transition_token_count`` / ``token_count`` model-config fields.

Subcommands
  convert  IN -> OUT artifact with the trimmed region stamped. Fail-closed:
           refuses budget > target (shrink-to-budget is the only safe
           direction), target > current region, or in-place writes.
  parity   Forward ~N sampled decision states from existing training-cache
           shards through BOTH checkpoints on CPU fp32 (original at full
           region, converted on token-sliced arrays); asserts argmax-identical
           policies and max |dlogit| / |dvalue| below tolerance. Also asserts
           every dropped token row was unattended (truncation-only proof).
  bench    Timed forward passes original vs converted on synthetic batches
           with realistic attention fill; reports sec/pass, evals/sec and the
           trimmed:original throughput ratio (the "how many more checks per
           second in search" number).

Run where torch + the checkpoint live:
  python scripts/convert_region_trim.py convert --checkpoint C.pt --output T.pt --target-region 16
  python scripts/convert_region_trim.py parity --original C.pt --converted T.pt --cache-glob '.../cache/*' --samples 2000
  python scripts/convert_region_trim.py bench --original C.pt --converted T.pt --passes 200 --batch-sizes 1,32,256
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokezero.neural_policy import (  # noqa: E402
    NEURAL_POLICY_SCHEMA_VERSION,
    TransformerPolicyConfig,
)
from pokezero.showdown import observation_spec_for_schema  # noqa: E402


def trimmed_model_config_dict(model_config: dict, target_region: int) -> dict:
    """Pure config transform for the trim (torch-free; unit-tested).

    Fail-closed rules:
      - the stamped ``transition_token_budget`` must be <= ``target_region``
        (rows above the budget were trained-on if budget > target — refusing is
        the only safe direction);
      - ``target_region`` must be <= the CURRENT physical region (growing a
        canvas would fabricate untrained rows);
      - the result must re-validate through TransformerPolicyConfig (which
        enforces token_count == fixed prefix + region).
    """
    config = TransformerPolicyConfig.from_dict(model_config)
    if target_region <= 0:
        raise ValueError("target region must be positive")
    if config.transition_token_budget > target_region:
        raise ValueError(
            f"fail closed: stamped transition_token_budget {config.transition_token_budget} "
            f"exceeds target region {target_region} — this checkpoint trained on rows the "
            "trim would remove; shrink-to-budget is the only safe direction."
        )
    if target_region > config.transition_token_count:
        raise ValueError(
            f"fail closed: target region {target_region} exceeds the checkpoint's physical "
            f"region {config.transition_token_count} — growing the canvas would fabricate "
            "untrained rows."
        )
    schema_spec = observation_spec_for_schema(config.observation_schema_version)
    fixed_tokens = schema_spec.token_count - schema_spec.transition_token_count
    trimmed = dict(model_config)
    trimmed["transition_token_count"] = int(target_region)
    trimmed["token_count"] = int(fixed_tokens + target_region)
    # Round-trip through the dataclass so the cross-validation runs NOW, not at load time.
    TransformerPolicyConfig.from_dict(trimmed)
    return trimmed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_payload(path: Path):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != NEURAL_POLICY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported neural policy schema: {payload.get('schema_version')!r}")
    return payload


def cmd_convert(args: argparse.Namespace) -> None:
    import torch

    source = Path(args.checkpoint)
    output = Path(args.output)
    if output.resolve() == source.resolve():
        raise SystemExit("fail closed: refusing to convert in place — write a NEW artifact.")
    payload = _load_payload(source)
    before = TransformerPolicyConfig.from_dict(payload["model_config"])
    payload["model_config"] = trimmed_model_config_dict(dict(payload["model_config"]), args.target_region)
    after = TransformerPolicyConfig.from_dict(payload["model_config"])
    # Atomic write, same pattern as save_transformer_checkpoint.
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with open(temporary, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(json.dumps({
        "source": str(source), "source_sha256": _sha256(source),
        "output": str(output), "output_sha256": _sha256(output),
        "region": {"before": before.transition_token_count, "after": after.transition_token_count},
        "token_count": {"before": before.token_count, "after": after.token_count},
        "budget": after.transition_token_budget,
        "weights_changed": 0,
    }, indent=2))


def _load_model(path: Path):
    from pokezero.neural_policy import load_transformer_checkpoint

    model, result = load_transformer_checkpoint(path, map_location="cpu")
    model.eval()
    return model, result.model_config


def _token_slice(tensor, token_count: int, axis: int):
    index = [slice(None)] * tensor.dim()
    index[axis] = slice(0, token_count)
    return tensor[tuple(index)]


def _iter_cache_examples(cache_glob: str, samples: int):
    """Yield dense per-example tensors from training-cache shards (both formats)."""
    import torch

    from pokezero.dataset import iter_training_cache_batches

    paths = sorted(glob.glob(cache_glob))
    if not paths:
        raise SystemExit(f"no cache shards match {cache_glob!r}")
    yielded = 0
    for path in paths:
        for batch in iter_training_cache_batches(path):
            categorical = torch.as_tensor(batch.categorical_ids)
            numeric = torch.as_tensor(batch.numeric_features)
            token_types = torch.as_tensor(batch.token_type_ids)
            attention = torch.as_tensor(batch.attention_mask)
            if batch.row_categorical_ids is not None:
                # Row-indexed caches deduplicate window rows; regather to dense.
                rows_c = torch.as_tensor(batch.row_categorical_ids)
                rows_n = torch.as_tensor(batch.row_numeric_features)
                rows_t = torch.as_tensor(batch.row_token_type_ids)
                rows_a = torch.as_tensor(batch.row_attention_mask)
                idx = torch.as_tensor(batch.window_row_indices)
                categorical, numeric = rows_c[idx], rows_n[idx]
                token_types, attention = rows_t[idx], rows_a[idx]
            history = torch.as_tensor(batch.history_mask)
            legal = torch.as_tensor(batch.legal_action_mask)
            for i in range(categorical.shape[0]):
                yield (categorical[i:i + 1], numeric[i:i + 1], token_types[i:i + 1],
                       attention[i:i + 1], history[i:i + 1], legal[i:i + 1])
                yielded += 1
                if yielded >= samples:
                    return


def cmd_parity(args: argparse.Namespace) -> None:
    import torch

    original, original_config = _load_model(Path(args.original))
    converted, converted_config = _load_model(Path(args.converted))
    small = converted_config.token_count
    if original_config.transition_token_budget != converted_config.transition_token_budget:
        raise SystemExit("budget mismatch between checkpoints — not a trim pair.")
    checked = 0
    argmax_mismatch = 0
    max_dlogit = 0.0
    max_dvalue = 0.0
    dropped_attended = 0
    with torch.inference_mode():
        for cat, num, tok, att, hist, legal in _iter_cache_examples(args.cache_glob, args.samples):
            # Truncation-only proof: every dropped row must be unattended.
            if bool(att[..., small:].any()):
                dropped_attended += 1
                continue
            out_full = original(
                categorical_ids=cat, numeric_features=num.float(), token_type_ids=tok,
                attention_mask=att, history_mask=hist,
            )
            out_trim = converted(
                categorical_ids=_token_slice(cat, small, 2),
                numeric_features=_token_slice(num, small, 2).float(),
                token_type_ids=_token_slice(tok, small, 2),
                attention_mask=_token_slice(att, small, 2),
                history_mask=hist,
            )
            logits_full = out_full.policy_logits.masked_fill(~legal.bool(), float("-inf"))
            logits_trim = out_trim.policy_logits.masked_fill(~legal.bool(), float("-inf"))
            if int(logits_full.argmax(dim=-1)) != int(logits_trim.argmax(dim=-1)):
                argmax_mismatch += 1
            finite = legal.bool()
            dlogit = (out_full.policy_logits - out_trim.policy_logits)[finite].abs().max()
            dvalue = (out_full.value - out_trim.value).abs().max()
            max_dlogit = max(max_dlogit, float(dlogit))
            max_dvalue = max(max_dvalue, float(dvalue))
            checked += 1
    report = {
        "checked": checked,
        "dropped_row_attended_violations": dropped_attended,
        "argmax_mismatches": argmax_mismatch,
        "max_abs_dlogit": max_dlogit,
        "max_abs_dvalue": max_dvalue,
        "tolerance": args.tolerance,
        "pass": (dropped_attended == 0 and argmax_mismatch == 0
                 and max_dlogit < args.tolerance and max_dvalue < args.tolerance),
    }
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit("PARITY GATE FAILED")


def _synthetic_batch(config, batch_size: int):
    """Realistic-fill synthetic inputs: fixed prefix + budget rows attended."""
    import torch

    tokens = config.token_count
    fixed = tokens - config.transition_token_count
    generator = torch.Generator().manual_seed(20260722)
    cat = torch.randint(1, max(2, config.categorical_vocab_size), (batch_size, 1, tokens, config.categorical_feature_count), generator=generator)
    num = torch.rand((batch_size, 1, tokens, config.numeric_feature_count), generator=generator)
    tok = torch.randint(0, config.token_type_vocab_size, (batch_size, 1, tokens), generator=generator)
    att = torch.zeros((batch_size, 1, tokens), dtype=torch.bool)
    att[..., :fixed] = True
    att[..., fixed:fixed + config.transition_token_budget] = True
    hist = torch.ones((batch_size, 1), dtype=torch.bool)
    return {"categorical_ids": cat, "numeric_features": num, "token_type_ids": tok,
            "attention_mask": att, "history_mask": hist}


def cmd_bench(args: argparse.Namespace) -> None:
    import torch

    results = {}
    models = {
        "original": _load_model(Path(args.original)),
        "converted": _load_model(Path(args.converted)),
    }
    torch.set_num_threads(args.threads)
    for label, (model, config) in models.items():
        results[label] = {"token_count": config.token_count, "batches": {}}
        for batch_size in args.batch_sizes:
            batch = _synthetic_batch(config, batch_size)
            with torch.inference_mode():
                for _ in range(args.warmup):
                    model(**batch)
                timings = []
                for _ in range(args.passes):
                    start = time.perf_counter()
                    model(**batch)
                    timings.append(time.perf_counter() - start)
            per_pass = statistics.median(timings)
            results[label]["batches"][batch_size] = {
                "median_ms_per_pass": round(per_pass * 1e3, 3),
                "evals_per_second": round(batch_size / per_pass, 1),
            }
    summary = {"threads": args.threads, "passes": args.passes, "results": results, "speedup": {}}
    for batch_size in args.batch_sizes:
        orig = results["original"]["batches"][batch_size]["evals_per_second"]
        trim = results["converted"]["batches"][batch_size]["evals_per_second"]
        summary["speedup"][batch_size] = round(trim / orig, 2)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser("convert")
    convert.add_argument("--checkpoint", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--target-region", type=int, required=True)
    convert.set_defaults(func=cmd_convert)

    parity = sub.add_parser("parity")
    parity.add_argument("--original", required=True)
    parity.add_argument("--converted", required=True)
    parity.add_argument("--cache-glob", required=True)
    parity.add_argument("--samples", type=int, default=2000)
    parity.add_argument("--tolerance", type=float, default=1e-5)
    parity.set_defaults(func=cmd_parity)

    bench = sub.add_parser("bench")
    bench.add_argument("--original", required=True)
    bench.add_argument("--converted", required=True)
    bench.add_argument("--passes", type=int, default=200)
    bench.add_argument("--warmup", type=int, default=20)
    bench.add_argument("--threads", type=int, default=1)
    bench.add_argument("--batch-sizes", type=lambda s: [int(x) for x in s.split(",")], default=[1, 32, 256])
    bench.set_defaults(func=cmd_bench)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
