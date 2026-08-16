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

# TWO DIFFERENT CHECKS, and the help text used to describe them as one. The first runs on
# every head on every invocation and asks "is the number I recomputed the library's number";
# the second runs only when the caller passes --expect-ece and asks "is the library's number
# the one the published cell reported", which pins the DATA and the CHECKPOINT rather than the
# arithmetic. Naming them here is what lets the help text and the comparisons drift together
# instead of apart.
LIBRARY_CROSSCHECK_TOL = 1e-9
PUBLISHED_ECE_TOL = 1e-6
EXPECT_ECE_HELP = (
    f"assert a head's ECE matches a PUBLISHED figure (e.g. the cell's own "
    f"value-calibration.json) to {PUBLISHED_ECE_TOL:g}, which pins the DATA and the CHECKPOINT "
    f"to the ones that cell used. This is a SEPARATE check from the recomputed-vs-library "
    f"cross-check, which is unconditional and runs at {LIBRARY_CROSSCHECK_TOL:g} on every head "
    f"whether or not this flag is passed.")


def bin_index(prediction: float, bins: int) -> int:
    """The bin a prediction falls in, byte-for-byte as `value_calibration` computes it.

    This is a TRANSCRIPTION, and it has to be exact. The arithmetic here is
    `int(((p + 1) / 2) * bins)`, matching `_ValueCalibrationTotals._bin_index`
    (`src/pokezero/value_calibration.py`). The obvious algebraic rearrangement
    `int((p + 1) / (2 / bins))` -- divide by a precomputed width -- is NOT the same function
    in binary floating point: `2 / 10` is not representable, so at 3 of the 9 interior bin
    edges of the default 10-bin grid (p = -0.40, 0.20, 0.40) it puts the point in the bin
    BELOW. Bins are half-open `[lower, upper)`, so the library is the correct one and an edge
    value belongs to the bin above. That rearrangement is exactly what this script shipped
    with, and it made every ECE here a different metric from the one in the artifacts, in a
    way the `--expect-ece` cross-check could only catch by luck.

    Clipping and the `p == 1.0` special case are the library's too: the top bin is CLOSED at
    +1, so a saturated head does not index past the end.
    """
    clipped = min(1.0, max(-1.0, prediction))
    if clipped == 1.0:
        return bins - 1
    return int(((clipped + 1.0) / 2.0) * bins)


def ece_from(preds: list[float], rets: list[float], bins: int) -> dict:
    """ECE over `bins` equal-width bins on [-1, 1], the return scale the head lives on.

    Matches `pokezero.value_calibration`: per-bin |mean_prediction - mean_return| weighted by
    bin occupancy, over the bins `bin_index` above assigns. Recomputed here rather than called,
    so the bootstrap can run on resampled indices without a forward pass per rep -- and
    cross-checked UNCONDITIONALLY against the library's own number in `main`, at
    `LIBRARY_CROSSCHECK_TOL`, because a reimplemented metric that silently disagrees with the
    one in the artifacts would make every delta below incomparable with the published cells.
    """
    n = len(preds)
    sums_p = [0.0] * bins
    sums_r = [0.0] * bins
    counts = [0] * bins
    for p, r in zip(preds, rets):
        idx = bin_index(p, bins)
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


def fmt_share(share: float | None, width: int = 11) -> str:
    """`bias_share_of_ece` for a fixed-width column, INCLUDING when it does not exist.

    `abs(bias)/ece` is None when ECE is exactly 0 -- a head perfectly calibrated on this set,
    which is the one head nobody would want the tool to crash on. Formatting None with
    `:.4f` raises TypeError, so both readouts went through a format spec that could not
    render the value the metric is defined to return.
    """
    return f"{'n/a':>{width}s}" if share is None else f"{share:{width}.4f}"


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
                    help=EXPECT_ECE_HELP)
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
        if abs(got["ece"] - lib.expected_calibration_error) > LIBRARY_CROSSCHECK_TOL:
            raise SystemExit(
                f"REFUSING: recomputed ECE {got['ece']!r} != library "
                f"{lib.expected_calibration_error!r} for {name}. The bootstrap would then be "
                f"resampling a metric that is not the one in the artifacts.")
        if name in expect and abs(got["ece"] - expect[name]) > PUBLISHED_ECE_TOL:
            raise SystemExit(
                f"REFUSING: {name} ECE {got['ece']:.8f} != expected {expect[name]:.8f}. Either "
                f"the calibration data or the checkpoint is not the one the published cell "
                f"used, so nothing here is comparable to it.")
        per_head[name] = {"preds": preds, "point": got}
        print(f"{name}: n={got['n']} ECE {got['ece']:.6f}  bias {got['bias']:+.6f}  "
              f"|bias|/ECE {fmt_share(got['bias_share_of_ece'], 0).strip()}  "
              f"mae {got['mae']:.4f}"
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
              f"{fmt_share(per_head[k]['point']['bias_share_of_ece'])}")
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
