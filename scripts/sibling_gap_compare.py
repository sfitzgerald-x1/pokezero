#!/usr/bin/env python3
"""PAIRED comparison of sibling-gap regressions across value-head cells.

Companion to the beta instrument, which scores ONE pairs file. Advancement is a DIFFERENCE
between cells, and this computes that difference with an interval.

Why the beta instrument alone cannot answer the advancement question. `sibling_beta.py`
bootstraps ONE pairs file, so comparing two cells means eyeballing two overlapping CIs --
which is not a test of a difference, and here it is a badly conservative one: every cell is
measured on the SAME 465 pairs against the SAME true_gap column, so the two estimates are
strongly positively correlated and the difference is far better determined than either
marginal interval suggests.

So the resampling is done ONCE over pair INDICES and applied to every cell, and the statistic
is the DIFFERENCE. That is the interval an advancement decision needs.

Also reported, because it is the mechanism test that separates verdict (a) from verdict (b):

  corr(head_gap_cell, head_gap_ref)  -- if a cell's predictions are an affine rescale of the
                                        reference's, this is 1.0 and NOTHING was learned; the
                                        beta movement is then arithmetic on the spread.
  sd(head_gap) ratio                 -- the spread factor, i.e. the size of that rescale.
  implied_beta_from_pure_rescale     -- what beta WOULD read if the cell were exactly the
                                        reference rescaled by its own sd ratio. If the measured
                                        beta lands on this, the beta move is fully explained by
                                        spread and R^2 is the only thing left to look at.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics as st
from pathlib import Path


def ols(xs, ys):
    n = len(xs)
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    beta_th = sxy / sxx          # true on head  (calibration slope)
    beta_ht = sxy / syy          # head on true  (the gate's convention)
    r2 = beta_th * beta_ht
    return beta_ht, beta_th, r2


def pearson(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / (sxx * syy) ** 0.5 if sxx > 0 and syy > 0 else float("nan")


def pct(v, q):
    return sorted(v)[min(len(v) - 1, int(q * len(v)))]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _certified_rescore(doc: dict, *, name: str, path: Path) -> dict:
    """Require a candidate to be the merger's certified replay product.

    The raw bank is a permitted reference because it is the original ground truth.  Every
    other cell must come through ``rescore_value_head --merge``: per-shard diagnostic output is
    intentionally written before a failed reproduction refusal, and comparing that output as a
    normal cell would turn a known state mismatch into a plausible beta/R² verdict.
    """
    rescore = doc.get("rescore")
    if not isinstance(rescore, dict):
        raise SystemExit(
            f"REFUSING: candidate cell {name!r} ({path}) is not a certified merged rescore. "
            "Raw pairs are allowed only for --ref; every candidate must pass replay "
            "reproduction before it can be compared.")
    if (rescore.get("schema") != "pokezero.phase3.value-head-rescore.v1"
            or rescore.get("certified_reproduction") is not True):
        raise SystemExit(
            f"REFUSING: candidate cell {name!r} ({path}) lacks the merged rescore "
            "certification. Do not score diagnostic-only or per-shard output.")
    for key in ("banked_pairs_sha256", "state_checkpoint_sha256", "head_checkpoint_sha256"):
        value = rescore.get(key)
        if (not isinstance(value, str) or len(value) != 64
                or any(ch not in "0123456789abcdef" for ch in value)):
            raise SystemExit(
                f"REFUSING: candidate cell {name!r} ({path}) has no valid {key}. "
                "Its replay and candidate-weight identity is not auditable.")
    return rescore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", required=True, metavar="NAME=FILE")
    ap.add_argument("--ref", required=True, help="cell name the deltas are taken against")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--boot-seeds", default="20260816,3,42")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cells = {}
    source_banks = {}
    for spec in args.cell:
        name, _, path = spec.partition("=")
        if not name or not path or name in cells:
            raise SystemExit(f"REFUSING: --cell must name one unique NAME=FILE, got {spec!r}")
        path_obj = Path(path)
        doc = json.loads(path_obj.read_text())
        if name == args.ref:
            source_banks[name] = sha256_file(path_obj)
        else:
            source_banks[name] = _certified_rescore(doc, name=name, path=path_obj)[
                "banked_pairs_sha256"]
        rows = {}
        for p in doc.get("pairs") or []:
            if p.get("head_gap") is None or p.get("true_gap") is None:
                continue
            key = (p["seed"], p["prefix"], p["seat"])
            if key in rows:
                raise SystemExit(f"REFUSING: {name} duplicates pair {key}.")
            rows[key] = p
        cells[name] = rows
        print(f"{name}: {len(rows)} pairs from {path}")
    if args.ref not in cells:
        raise SystemExit(f"--ref {args.ref!r} is not one of {sorted(cells)}")

    expected_bank = source_banks[args.ref]
    for name, bank in source_banks.items():
        if bank != expected_bank:
            raise SystemExit(
                f"REFUSING: {name}'s source bank SHA {bank} != {args.ref}'s "
                f"{expected_bank}. The cells do not reuse one ground truth.")

    # ALIGN on the pair identity. A cell that dropped a pair must not shift another cell's
    # rows by one; the whole design rests on every cell being read at the same states.
    keys = sorted(set.intersection(*(set(r) for r in cells.values())))
    for name, rows in cells.items():
        if len(rows) != len(keys):
            raise SystemExit(
                f"REFUSING: {name} has {len(rows)} pairs but only {len(keys)} are common to "
                "all cells. A dropped or foreign pair cannot be silently intersected away.")
    print(f"\naligned on {len(keys)} pairs common to all {len(cells)} cells")

    ys = [float(cells[args.ref][k]["true_gap"]) for k in keys]
    xs = {n: [float(cells[n][k]["head_gap"]) for k in keys] for n in cells}
    # true_gap is REUSED ground truth and must be byte-identical across cells; if it is not,
    # the cells were not scored against one truth and no delta below means anything.
    for n in cells:
        other = [float(cells[n][k]["true_gap"]) for k in keys]
        if other != ys:
            raise SystemExit(f"REFUSING: {n}'s true_gap column differs from {args.ref}'s.")
        for key in keys:
            reference = cells[args.ref][key]
            candidate = cells[n][key]
            for field in ("arm_a", "arm_b", "noise_var"):
                if candidate.get(field) != reference.get(field):
                    raise SystemExit(
                        f"REFUSING: {n}'s {field} differs from {args.ref}'s at {key}. "
                        "A value-head comparison must retain the same sibling actions and "
                        "the same ground-truth noise witness.")

    lab_var = st.mean([float(cells[args.ref][k]["noise_var"]) for k in keys])
    sd_true = st.pstdev(ys)
    atten = 1.0 - lab_var / sd_true ** 2
    print(f"label var {lab_var:.6f}  measured-gap var {sd_true ** 2:.6f}  "
          f"attenuation factor {atten:.6f} (= R^2 ceiling; SHARED by every cell because "
          f"true_gap and noise_var are the reused ground truth)")
    if atten <= 0:
        # The correction divides every beta by this number. Non-positive means the banked
        # per-pair rollout noise accounts for ALL of the observed variance in true_gap, i.e.
        # the ground-truth column carries no measurable signal to regress against. Dividing
        # by it does not degrade gracefully: it SIGN-FLIPS every beta, so the tool would
        # print a confident "MOVED" off an interval that is the negative of the one it means,
        # and exit 0. There is no reading of this output that is worth having.
        raise SystemExit(
            f"REFUSING: attenuation factor {atten:.6f} is not positive. The mean banked "
            f"noise_var ({lab_var:.6f}) is at or above the observed variance of true_gap "
            f"({sd_true ** 2:.6f}), so the reused ground truth has no variance left to "
            f"explain and every noise-corrected beta below would be a sign-flipped "
            f"artefact of dividing by a non-positive number. Most likely cause: the "
            f"noise_var column and the true_gap column do not come from the same bank.")

    point = {}
    for n in cells:
        fit = ols(xs[n], ys)
        if fit is None:
            # `ols` signals a degenerate column by returning None, and a COLLAPSED value head
            # -- constant output, hence constant head_gap -- is a live outcome of the training
            # this comparator scores. Unpacking None here raised a bare TypeError that named
            # neither the cell nor the reason.
            raise SystemExit(
                f"CANNOT RUN: cell {n!r} has no variance to regress -- its head_gap column is "
                f"constant over all {len(keys)} aligned pairs (a collapsed value head does "
                f"this), or the shared true_gap column is. No slope or R^2 is defined.")
        b_ht, b_th, r2 = fit
        point[n] = {
            "beta_head_on_true_raw": b_ht,
            "beta_head_on_true_noise_corrected": b_ht / atten,
            "compression_x_corrected": 1.0 / (b_ht / atten),
            "beta_true_on_head_calibration": b_th,
            "r2": r2,
            "sd_head_gap": st.pstdev(xs[n]),
            "sd_ratio_vs_ref": st.pstdev(xs[n]) / st.pstdev(xs[args.ref]),
            "corr_head_gap_with_ref": pearson(xs[n], xs[args.ref]),
        }
    # SECOND PASS, and it has to be one. This quantity reads the REF cell's beta, which is
    # not in `point` yet while the loop is still walking the cells that were listed before
    # the ref on the command line. Computing it inside the loop left every such cell with
    # None -- and the RESCALE TEST below formats it with `:.4f`, so the tool crashed on
    # argument ORDER alone: fine with the ref first, TypeError with the ref second, and the
    # mechanism test that separates "learned something" from "rescaled its output" simply
    # unavailable in the second case.
    ref_beta = point[args.ref]["beta_head_on_true_noise_corrected"]
    for n in cells:
        point[n]["implied_beta_corrected_from_pure_rescale"] = (
            ref_beta * point[n]["sd_ratio_vs_ref"])

    seeds = [int(s) for s in args.boot_seeds.split(",") if s.strip()]
    n = len(keys)
    draws = {n_: {"r2": [], "beta": []} for n_ in cells}
    dif = {n_: {"r2": [], "beta": []} for n_ in cells}
    for sd_ in seeds:
        rng = random.Random(sd_)
        for _ in range(args.boot):
            idx = [rng.randrange(n) for _ in range(n)]
            yb = [ys[i] for i in idx]
            got = {}
            for n_ in cells:
                r = ols([xs[n_][i] for i in idx], yb)
                if r is None:
                    got = None
                    break
                got[n_] = r
            if got is None:
                continue
            for n_ in cells:
                draws[n_]["r2"].append(got[n_][2])
                draws[n_]["beta"].append(got[n_][0] / atten)
                dif[n_]["r2"].append(got[n_][2] - got[args.ref][2])
                dif[n_]["beta"].append((got[n_][0] - got[args.ref][0]) / atten)

    print(f"\n=== per cell (paired bootstrap, {len(draws[args.ref]['r2'])} reps over "
          f"{len(seeds)} seeds) ===")
    hdr = (f"{'cell':>10s} {'beta_corr':>10s} {'  95% CI':>18s} {'x':>6s} {'R^2':>8s} "
           f"{'  95% CI':>18s} {'sd(hg)':>8s}")
    print(hdr)
    for n_ in cells:
        p = point[n_]
        print(f"{n_:>10s} {p['beta_head_on_true_noise_corrected']:10.4f} "
              f"[{pct(draws[n_]['beta'], .025):7.4f},{pct(draws[n_]['beta'], .975):7.4f}] "
              f"{p['compression_x_corrected']:6.2f} {p['r2']:8.4f} "
              f"[{pct(draws[n_]['r2'], .025):7.4f},{pct(draws[n_]['r2'], .975):7.4f}] "
              f"{p['sd_head_gap']:8.5f}")

    print(f"\n=== PAIRED DELTAS vs {args.ref} (the interval an advancement decision needs) ===")
    for n_ in cells:
        if n_ == args.ref:
            continue
        db = (point[n_]["beta_head_on_true_noise_corrected"]
              - point[args.ref]["beta_head_on_true_noise_corrected"])
        dr = point[n_]["r2"] - point[args.ref]["r2"]
        blo, bhi = pct(dif[n_]["beta"], .025), pct(dif[n_]["beta"], .975)
        rlo, rhi = pct(dif[n_]["r2"], .025), pct(dif[n_]["r2"], .975)
        print(f"  {n_:>10s}  d beta_corr {db:+.4f}  95% CI [{blo:+.4f},{bhi:+.4f}]  "
              f"{'MOVED' if (blo > 0) == (bhi > 0) else 'within noise'}")
        print(f"  {'':>10s}  d R^2       {dr:+.4f}  95% CI [{rlo:+.4f},{rhi:+.4f}]  "
              f"{'MOVED' if (rlo > 0) == (rhi > 0) else 'within noise'}")

    print(f"\n=== RESCALE TEST: is a cell just {args.ref} with a different output spread? ===")
    for n_ in cells:
        if n_ == args.ref:
            continue
        p = point[n_]
        print(f"  {n_:>10s}  corr(head_gap, {args.ref}) {p['corr_head_gap_with_ref']:.5f}   "
              f"sd ratio {p['sd_ratio_vs_ref']:.4f}   "
              f"beta_corr measured {p['beta_head_on_true_noise_corrected']:.4f} vs "
              f"{p['implied_beta_corrected_from_pure_rescale']:.4f} implied by spread alone")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "schema": "pokezero.phase3.cell-comparison.v1",
            "ref": args.ref, "n_pairs_aligned": len(keys),
            "banked_pairs_sha256": expected_bank,
            "label_variance": lab_var, "attenuation_factor_shared": atten,
            "r2_ceiling": atten,
            "point": point,
            "bootstrap": {"reps_total": len(draws[args.ref]["r2"]), "seeds": seeds,
                          "reps_per_seed": args.boot},
            "ci": {n_: {"beta_corrected": [pct(draws[n_]["beta"], .025),
                                           pct(draws[n_]["beta"], .975)],
                        "r2": [pct(draws[n_]["r2"], .025), pct(draws[n_]["r2"], .975)]}
                   for n_ in cells},
            "paired_delta_ci_vs_ref": {
                n_: {"d_beta_corrected": [pct(dif[n_]["beta"], .025),
                                          pct(dif[n_]["beta"], .975)],
                     "d_r2": [pct(dif[n_]["r2"], .025), pct(dif[n_]["r2"], .975)]}
                for n_ in cells if n_ != args.ref},
        }, indent=1, sort_keys=True))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
