#!/usr/bin/env python3
"""Sanity-check a real checkpoint + its TorchScript export on golden-corpus rows.

Complements scripts/export_model.py --validate (random inputs): this feeds
REAL v2.2 observations from the golden corpus (docs/golden_corpus_notes.md)
through the eager checkpoint and the exported TorchScript artifact and checks:

  1. Export parity on real rows (max abs diff on policy_logits + value).
  2. Non-degenerate value head: the value varies across observations
     (std above --min-value-std, range not collapsed to a constant).
  3. Masked priors are a distribution: softmax over legal actions sums to 1
     with exactly zero probability mass on illegal actions.

Usage:
    python scripts/verify_real_checkpoint_outputs.py \
        --checkpoint checkpoints/curated/<ckpt>.pt \
        --ts-artifact exports/<model>/model_ts.pt \
        --corpus corpus/golden-v2 --rows 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def build_batch(corpus_dir: Path, row_count: int, device: str = "cpu"):
    """Stack the first ``row_count`` corpus decision rows into model inputs."""

    import numpy as np
    import torch

    from pokezero.golden_corpus import load_golden_corpus

    corpus = load_golden_corpus(corpus_dir)
    rows = tuple(corpus.decision_rows)[:row_count]
    if len(rows) < row_count:
        raise SystemExit(f"corpus has only {len(rows)} rows; asked for {row_count}")

    def stack(name, dtype):
        return torch.from_numpy(
            np.stack([np.asarray(getattr(r.arrays, name)) for r in rows]).astype(dtype)
        )

    categorical = stack("categorical_ids", np.int64).unsqueeze(1)
    numeric = stack("numeric_features", np.float32).unsqueeze(1)
    token_type = stack("token_type_ids", np.int64).unsqueeze(1)
    attention = stack("attention_mask", bool).unsqueeze(1)
    history = torch.ones(len(rows), 1, dtype=torch.bool)
    legal = stack("legal_action_mask", bool)
    inputs = tuple(t.to(device) for t in (categorical, numeric, token_type, attention, history))
    return inputs, legal.to(device)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ts-artifact", required=True, help="TorchScript artifact traced on --device.")
    parser.add_argument("--corpus", required=True, help="Golden corpus dir (schema v1 or v2 layout).")
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--parity-tolerance", type=float, default=0.0, help="Max abs diff allowed eager vs TS (default bit-exact).")
    parser.add_argument("--min-value-std", type=float, default=0.01)
    args = parser.parse_args(argv)

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from export_model import build_exportable_module  # noqa: E402 (scripts/ sibling)
    from pokezero.neural_policy import load_transformer_checkpoint

    model, _ = load_transformer_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()
    if args.device != "cpu":
        model = model.to(args.device)
    shim = build_exportable_module(model)
    traced = torch.jit.load(args.ts_artifact, map_location=args.device)

    inputs, legal = build_batch(Path(args.corpus), args.rows, device=args.device)

    with torch.no_grad():
        eager_logits, eager_value, _ = shim(*inputs)
        ts_logits, ts_value, _ = traced(*inputs)

    failures: list[str] = []

    parity = max(
        (eager_logits - ts_logits).abs().max().item(),
        (eager_value - ts_value).abs().max().item(),
    )
    print(f"parity eager-vs-TS on {args.rows} real rows: max abs diff {parity:.3e}")
    if parity > args.parity_tolerance:
        failures.append(f"parity {parity:.3e} > {args.parity_tolerance}")

    value = eager_value.float().cpu()
    print(
        f"value head: min {value.min():.4f} max {value.max():.4f} "
        f"mean {value.mean():.4f} std {value.std():.4f} (unique {value.unique().numel()}/{args.rows})"
    )
    if value.std().item() < args.min_value_std:
        failures.append(f"value std {value.std():.4f} < {args.min_value_std} (degenerate head?)")
    if value.abs().max().item() > 1.0 + 1e-6:
        failures.append("value outside tanh range [-1, 1]")

    masked = eager_logits.masked_fill(~legal, float("-inf"))
    priors = torch.softmax(masked, dim=-1)
    illegal_mass = priors.masked_fill(legal, 0.0).sum(dim=-1).max().item()
    sums = priors.sum(dim=-1)
    print(
        f"masked priors: row sums in [{sums.min():.6f}, {sums.max():.6f}], "
        f"max illegal mass {illegal_mass:.3e}, "
        f"mean legal actions {legal.sum(dim=-1).float().mean():.2f}, "
        f"mean max-prior {priors.max(dim=-1).values.mean():.4f}"
    )
    if illegal_mass != 0.0:
        failures.append(f"illegal prior mass {illegal_mass:.3e} != 0")
    if (sums - 1.0).abs().max().item() > 1e-5:
        failures.append("masked prior rows do not sum to 1")

    if failures:
        print("FAIL: " + "; ".join(failures), file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
