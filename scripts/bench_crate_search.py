#!/usr/bin/env python3
"""Bench model-in-the-loop batched search in the native crate (track D).

Sweeps the virtual-loss batched one-ply PUCT (`NativeLeafModel.search_batched`)
over batch sizes x sims-per-decision and reports searches/sec (decisions/sec)
plus effective model-priced sims/sec, next to the forward-only ceiling at the
same batch size. Reference points: the Python-side model-cost ladder
(docs/engine_search_poc.md: ~3.2ms/eval CPU saturated, 168 evals/s batch-1)
and the export bench (docs/model_export_findings.md).

The leaf observation is a template stub (per-leaf copy prices marshaling, not
encoding) — the Rust v2.2 encoder is a separate in-flight stream. Forward
cost is value-independent, so throughput is real; leaf CONTENT is not.

Usage:
    scripts/build_search_crate_model.sh <venv-python>   # build the crate first
    python scripts/export_model.py --checkpoint <ckpt> --out-dir exports/ --formats ts
    python scripts/bench_crate_search.py --artifact exports/model_ts.pt \
        --batch-sizes 1,16,64,256 --sims 64,256,1024 --out bench_crate.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
for path in (str(_SRC), str(_REPO / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

# Python-side sequential reference: ms per model eval on CPU at saturated
# batch (docs/engine_search_poc.md model-cost ladder, 10.2M v2.2 checkpoint).
PYTHON_LADDER_MS_PER_EVAL = 3.2


def _template_inputs(manifest: dict, seed: int):
    """Random valid v2.2-shaped observation (batch 1) at the artifact's shapes."""
    from export_model import make_random_inputs

    from pokezero.neural_policy import TransformerPolicyConfig

    shapes = manifest["input_shapes"]
    _, window, tokens, cat = shapes["categorical_ids"]
    num = shapes["numeric_features"][-1]
    config = TransformerPolicyConfig.compact_category(
        category_vocab=("a", "b"),
        category_oov_buckets=1,
        categorical_feature_count=cat,
        numeric_feature_count=num,
        embedding_dim=16,
        transformer_layers=1,
        attention_heads=2,
        feedforward_dim=32,
        dropout=0.0,
    )
    inputs = make_random_inputs(config, 1, seed=seed)
    flat = (
        inputs[0].flatten().tolist(),
        inputs[1].flatten().tolist(),
        inputs[2].flatten().tolist(),
        inputs[3].flatten().tolist(),
        inputs[4].flatten().tolist(),
    )
    return flat, (window, tokens, cat, num)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", default="exports/model_ts.pt", help="TorchScript artifact (per-device; CPU trace for --device cpu).")
    parser.add_argument("--manifest", default=None, help="export_manifest.json (default: alongside the artifact).")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"), help="tch device; the artifact must be traced for it.")
    parser.add_argument("--batch-sizes", default="1,16,64,256")
    parser.add_argument("--sims", default="64,256,1024")
    parser.add_argument("--min-time", type=float, default=3.0, help="Minimum measured seconds per cell (repeat searches until reached).")
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--out", default=None, help="Optional markdown output path.")
    args = parser.parse_args(argv)

    import pokezero_search

    if not getattr(pokezero_search, "MODEL_FEATURE_ENABLED", False):
        print("pokezero_search built without the model feature; run scripts/build_search_crate_model.sh", file=sys.stderr)
        return 1

    from pokezero.poke_engine_adapter import build_poke_engine_state, minimal_gen3_fixture

    manifest_path = Path(args.manifest) if args.manifest else Path(args.artifact).parent / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    template, (window, tokens, cat, num) = _template_inputs(manifest, args.seed)
    native = pokezero_search.NativeLeafModel(
        str(args.artifact),
        device=args.device,
        window=window,
        tokens=tokens,
        categorical_features=cat,
        numeric_features=num,
    )
    state_str = build_poke_engine_state(minimal_gen3_fixture()).to_string()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    sims_list = [int(x) for x in args.sims.split(",") if x.strip()]

    forward_rates: dict[int, float] = {}
    for batch in batch_sizes:
        forward_rates[batch] = native.bench_forward(batch, iters=max(6, int(600 / batch)), warmup=2, seed=args.seed)
        print(f"forward-only b={batch}: {forward_rates[batch]:,.0f} evals/s")

    lines = [
        f"| batch | sims/decision | searches/s | ms/decision | model-priced sims/s | forward-only evals/s (b) | python-seq est. ms/decision ({PYTHON_LADDER_MS_PER_EVAL}ms/eval) |",
        "|---|---|---|---|---|---|---|",
    ]
    for sims in sims_list:
        for batch in batch_sizes:
            searches = 0
            start = time.perf_counter()
            # One untimed warm search per cell (first call pays allocator/JIT warmup).
            native.search_batched(state_str, sims, batch, *template, seed=args.seed)
            start = time.perf_counter()
            while time.perf_counter() - start < args.min_time:
                report = json.loads(
                    native.search_batched(state_str, sims, batch, *template, seed=args.seed + searches)
                )
                searches += 1
            elapsed = time.perf_counter() - start
            per_search = elapsed / searches
            row = (
                f"| {batch} | {sims} | {1.0 / per_search:,.2f} | {per_search * 1e3:,.0f} | "
                f"{sims / per_search:,.0f} | {forward_rates[batch]:,.0f} | {sims * PYTHON_LADDER_MS_PER_EVAL:,.0f} |"
            )
            lines.append(row)
            print(
                f"batch={batch:<4} sims={sims:<5} {1.0/per_search:8.2f} searches/s  "
                f"{per_search*1e3:8.0f} ms/decision  {sims/per_search:8.0f} sims/s  "
                f"(model_evals={report['model_evals']}, terminal={report['terminal_leaves']})"
            )

    table = "\n".join(lines)
    header = (
        f"Device: {args.device}; artifact: {args.artifact}; state: minimal_gen3_fixture; "
        f"min-time {args.min_time}s/cell.\n\n"
    )
    if args.out:
        Path(args.out).write_text(header + table + "\n")
        print(f"\nwrote {args.out}")
    else:
        print("\n" + header + table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
