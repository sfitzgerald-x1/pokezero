#!/usr/bin/env python3
"""ECE for SEVERAL value heads on ONE calibration set, with paired intervals and deltas.

WHY THIS EXISTS. Phase 3 advances an arm on beta AND ECE. `mcts/sibling_beta.py` computes
beta and says so about its other half: "NO ECE. ... the plan's gate is 'beta AND ECE' and ECE
is the component that would catch the rescale this tool warns about ... so it is a separate
instrument and a real gap." This is that instrument.

Training already emits an ECE per cell (`--value-calibration-out`), so why more. Two reasons,
and both are needed to make an advancement decision rather than a comparison of decimals:

  1. NO INTERVAL. The training report is a point estimate. Cell ECEs here sit ~0.003 apart on
     21,171 examples, and nothing in the artifacts says whether that is a difference. A gate
     read off two point estimates is the failure mode this programme has already retracted
     claims for.
  2. NO BASELINE. The gate is a change versus the UNTUNED head, and the untuned checkpoint
     never had a calibration pass on this set at all -- training only reports for what it
     trained. So the delta the gate asks for could not be formed.

PAIRED, on purpose. Every head is evaluated on the SAME examples in the SAME order, and the
bootstrap resamples EXAMPLE INDICES once and applies them to every head. Two independently
bootstrapped ECEs would give badly conservative overlapping intervals, because the estimates
are strongly correlated by construction: they share the returns and most of the trunk.

THE BIAS DECOMPOSITION IS REPORTED BESIDE ECE, and it is not decoration. The sibling probe's
own docstring: two heads can share an ECE of 0.05, one off by a constant (which "cancels
exactly in a comparison and ranks siblings perfectly") and one scattering per position (which
"destroys ranking"). Only the second breaks search. So an ECE that is almost entirely |bias|
is measuring the component that does NOT affect the sibling ordering Phase 3 exists to fix,
and a reader must be able to see that from the output rather than infer it.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

N_BINS_DEFAULT = 10


def ece_from(preds: list[float], rets: list[float], bins: int) -> dict:
    """ECE over `bins` equal-width bins on [-1, 1], the return scale the head lives on.

    Matches `pokezero.value_calibration`: per-bin |mean_prediction - mean_return| weighted by
    bin occupancy. Recomputed here rather than called, so the bootstrap can run on resampled
    indices without a forward pass per rep -- and cross-checked against the library's own
    number by `--expect-ece`, because a reimplemented metric that silently disagrees with the
    one in the artifacts would make every delta below incomparable with the published cells.
    """
    n = len(preds)
    lo, hi = -1.0, 1.0
    width = (hi - lo) / bins
    sums_p = [0.0] * bins
    sums_r = [0.0] * bins
    counts = [0] * bins
    for p, r in zip(preds, rets):
        idx = int((p - lo) / width)
        idx = 0 if idx < 0 else (bins - 1 if idx >= bins else idx)
        sums_p[idx] += p
        sums_r[idx] += r
        counts[idx] += 1
    ece = 0.0
    for b in range(bins):
        if not counts[b]:
            continue
        ece += (counts[b] / n) * abs(sums_p[b] / counts[b] - sums_r[b] / counts[b])
    bias = st.fmean([p - r for p, r in zip(preds, rets)])
    # The scatter that a constant offset cannot explain. ECE is bounded below by |bias| only
    # loosely, so both are reported and their ratio is the reading that matters.
    return {"ece": ece, "bias": bias, "abs_bias": abs(bias),
            "bias_share_of_ece": (abs(bias) / ece) if ece > 0 else None,
            "mae": st.fmean([abs(p - r) for p, r in zip(preds, rets)]),
            "n": n}


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--ref", required=True, help="head name the deltas are taken against")
    ap.add_argument("--data", nargs="+", required=True, help="calibration shard dirs")
    ap.add_argument("--bins", type=int, default=N_BINS_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--boot-seeds", default="20260816,3,42")
    ap.add_argument("--expect-ece", action="append", default=[], metavar="NAME=VALUE",
                    help="assert a head's ECE matches a published figure (e.g. the cell's own "
                         "value-calibration.json) to 1e-6. Use it: an ECE recomputed here "
                         "that disagrees with the artifacts makes every delta meaningless.")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from pokezero.neural_policy import load_transformer_checkpoint
    from pokezero.value_calibration import evaluate_value_calibration

    heads = {}
    for spec in args.head:
        name, _, path = spec.partition("=")
        heads[name] = Path(path or name)
    if args.ref not in heads:
        raise SystemExit(f"--ref {args.ref!r} is not one of {sorted(heads)}")
    expect = {}
    for spec in args.expect_ece:
        name, _, val = spec.partition("=")
        expect[name] = float(val)

    # Per-example dumps, in one fixed example order. Recomputed per head by re-reading the
    # same shard list; the returns column is then asserted IDENTICAL across heads, which is
    # what makes the bootstrap paired rather than merely simultaneous.
    per_head: dict[str, dict] = {}
    ref_returns: list[float] | None = None
    for name, path in heads.items():
        model, result = load_transformer_checkpoint(path, map_location=args.device)
        preds, rets = _dump(model, result, args.data, args.batch_size, args.device)
        if ref_returns is None:
            ref_returns = rets
        elif rets != ref_returns:
            raise SystemExit(
                f"REFUSING: {name}'s returns column differs from the first head's "
                f"({len(rets)} vs {len(ref_returns)} values). The heads were not evaluated on "
                f"one example set, so no paired delta below is meaningful.")
        got = ece_from(preds, rets, args.bins)
        lib = evaluate_value_calibration(
            model=model, training_result=result, paths=[Path(p) for p in args.data],
            batch_size=args.batch_size, bins=args.bins, device=args.device)
        if abs(got["ece"] - lib.expected_calibration_error) > 1e-9:
            raise SystemExit(
                f"REFUSING: recomputed ECE {got['ece']!r} != library "
                f"{lib.expected_calibration_error!r} for {name}. The bootstrap would then be "
                f"resampling a metric that is not the one in the artifacts.")
        if name in expect and abs(got["ece"] - expect[name]) > 1e-6:
            raise SystemExit(
                f"REFUSING: {name} ECE {got['ece']:.8f} != expected {expect[name]:.8f}. Either "
                f"the calibration data or the checkpoint is not the one the published cell "
                f"used, so nothing here is comparable to it.")
        per_head[name] = {"preds": preds, "point": got}
        print(f"{name}: n={got['n']} ECE {got['ece']:.6f}  bias {got['bias']:+.6f}  "
              f"|bias|/ECE {got['bias_share_of_ece']:.4f}  mae {got['mae']:.4f}"
              f"{'  [matches published]' if name in expect else ''}", flush=True)

    rets = ref_returns
    n = len(rets)
    seeds = [int(s) for s in args.boot_seeds.split(",") if s.strip()]
    draws = {k: [] for k in heads}
    dif = {k: [] for k in heads}
    for sd in seeds:
        rng = random.Random(sd)
        for _ in range(args.boot):
            idx = [rng.randrange(n) for _ in range(n)]
            yb = [rets[i] for i in idx]
            vals = {k: ece_from([per_head[k]["preds"][i] for i in idx], yb, args.bins)["ece"]
                    for k in heads}
            for k in heads:
                draws[k].append(vals[k])
                dif[k].append(vals[k] - vals[args.ref])

    print(f"\n=== ECE, paired bootstrap over {len(draws[args.ref])} reps "
          f"({len(seeds)} seeds x {args.boot}) ===")
    print(f"{'head':>10s} {'ECE':>9s} {'      95% CI':>20s} {'|bias|/ECE':>11s}")
    for k in heads:
        print(f"{k:>10s} {per_head[k]['point']['ece']:9.5f} "
              f"[{pct(draws[k], .025):8.5f},{pct(draws[k], .975):8.5f}] "
              f"{per_head[k]['point']['bias_share_of_ece']:11.4f}")
    print(f"\n=== PAIRED DELTAS vs {args.ref} (lower ECE is better) ===")
    for k in heads:
        if k == args.ref:
            continue
        d = per_head[k]["point"]["ece"] - per_head[args.ref]["point"]["ece"]
        lo, hi = pct(dif[k], .025), pct(dif[k], .975)
        verdict = ("WORSE than ref" if lo > 0 else
                   "BETTER than ref" if hi < 0 else "within noise")
        print(f"  {k:>10s}  d ECE {d:+.5f}  95% CI [{lo:+.5f},{hi:+.5f}]  {verdict}")

    out = {
        "schema": "pokezero.phase3.value-head-ece.v1",
        "ref": args.ref, "bins": args.bins, "n_examples": n,
        "data_paths": [str(p) for p in args.data],
        "heads": {k: {"checkpoint": str(heads[k]), **per_head[k]["point"]} for k in heads},
        "ece_ci95": {k: [pct(draws[k], .025), pct(draws[k], .975)] for k in heads},
        "paired_delta_ece_ci95_vs_ref": {
            k: [pct(dif[k], .025), pct(dif[k], .975)] for k in heads if k != args.ref},
        "bootstrap": {"reps_per_seed": args.boot, "seeds": seeds,
                      "reps_total": len(draws[args.ref])},
        "bias_note": ("A constant offset cancels in a sibling comparison and does not harm "
                      "ranking (value_head_sibling_probe.py docstring); per-position scatter "
                      "does. Read |bias|/ECE before reading ECE as a search-relevant number."),
    }
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, sort_keys=True))
        print(f"\nwrote {args.json}")
    return 0


def _dump(model, result, data, batch_size, device):
    """Per-example (prediction, return), in shard-iteration order.

    Mirrors `value_calibration.evaluate_value_calibration`'s loop, including the calibration
    transform, so the recomputed ECE is the published one. It is mirrored rather than reused
    because the library returns only aggregates and the bootstrap needs the examples.
    """
    from pokezero.dataset import iter_training_batches
    from pokezero.value_calibration import (
        _apply_value_calibration_transform, _trajectory_dataset_config_from_training_result,
    )
    from pokezero.neural_policy import (
        model_forward_from_training_tensors, training_batch_to_torch,
    )
    import torch

    cfg = _trajectory_dataset_config_from_training_result(result)
    transform = getattr(result, "value_calibration_transform", None)
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model, "to"):
        model.to(device)
    preds: list[float] = []
    rets: list[float] = []
    with torch.no_grad():
        for batch in iter_training_batches(data, batch_size=batch_size, config=cfg,
                                          defer_cache_window_expansion=True):
            tensors = training_batch_to_torch(batch, device=device)
            output = model_forward_from_training_tensors(model, tensors)
            preds.extend(_apply_value_calibration_transform(float(v), transform)
                         for v in output.value.detach().cpu().tolist())
            rets.extend(float(v) for v in tensors["returns"].detach().cpu().tolist())
    return preds, rets


if __name__ == "__main__":
    raise SystemExit(main())
