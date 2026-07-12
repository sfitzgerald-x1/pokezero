"""Stage A analysis: pairwise action/value fingerprint distances, within-run null band,
and the pre-registered verdict per cross-run pair. Reads the per-checkpoint fp-*.json
emitted by diversity_fingerprint_extract.py; writes pairwise.json + verdict.json.

Layers computed here: action (top-1 disagreement + JS divergence) and value (1-Pearson
+ p95 abs disagreement). Style and matchup layers are produced by their own stages and
merged into the verdict by the report generator.

Gates: self-pair sanity (A vs A: agreement 1, JS 0, Pearson 1) and a shuffled-label
control (misaligned decisions must collapse action agreement toward chance and inflate
JS), both embedded in the output.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from itertools import combinations

import numpy as np


def load_fp(path):
    d = json.load(open(path))
    by_id = {}
    for r in d["decisions"]:
        by_id[r["decision_id"]] = r
    return d, by_id


def js_divergence(pa: dict, pb: dict) -> float:
    keys = set(pa) | set(pb)
    a = np.array([pa.get(k, 0.0) for k in keys], dtype=float)
    b = np.array([pb.get(k, 0.0) for k in keys], dtype=float)
    m = 0.5 * (a + b)
    def kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def norm_entropy(probs: dict) -> float:
    p = np.array([v for v in probs.values() if v > 0], dtype=float)
    if len(p) <= 1:
        return 0.0
    return float(-np.sum(p * np.log(p)) / math.log(len(probs)))


def pair_metrics(fa, fb, ids, contested_mask):
    top_eq, js, contested_eq = [], [], []
    va, vb = [], []
    for did in ids:
        ra, rb = fa[did], fb[did]
        # probs come back with string keys after json round-trip
        pa = {int(k): v for k, v in ra["probs"].items()}
        pb = {int(k): v for k, v in rb["probs"].items()}
        eq = 1.0 if ra["top1"] == rb["top1"] else 0.0
        top_eq.append(eq)
        js.append(js_divergence(pa, pb))
        va.append(ra["value"]); vb.append(rb["value"])
        if contested_mask[did]:
            contested_eq.append(eq)
    va, vb = np.array(va), np.array(vb)
    pear = float(np.corrcoef(va, vb)[0, 1]) if va.std() > 0 and vb.std() > 0 else float("nan")
    return {
        "n": len(ids),
        "top1_disagreement": 1.0 - float(np.mean(top_eq)),
        "top1_disagreement_contested": 1.0 - float(np.mean(contested_eq)) if contested_eq else None,
        "js_divergence": float(np.mean(js)),
        "value_1_minus_pearson": 1.0 - pear,
        "value_p95_abs": float(np.percentile(np.abs(va - vb), 95)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    metas, fps = {}, {}
    for path in sorted(glob.glob(os.path.join(args.fp_dir, "fp-*.json"))):
        meta, by_id = load_fp(path)
        key = f"{meta['label']}:{meta['role']}"
        metas[key] = {k: meta[k] for k in ("label", "role", "checkpoint", "observation_schema", "numeric_census", "n_decisions", "corpus_sha256")}
        fps[key] = by_id

    keys = sorted(fps)
    common = set.intersection(*[set(fps[k]) for k in keys])
    common = sorted(common)

    # contested = mean normalized entropy across all checkpoints above global median
    ent_by_id = {did: np.mean([norm_entropy({int(kk): vv for kk, vv in fps[k][did]["probs"].items()}) for k in keys]) for did in common}
    med = float(np.median(list(ent_by_id.values())))
    contested_mask = {did: ent_by_id[did] > med for did in common}

    # all pairwise
    pairwise = {}
    for a, b in combinations(keys, 2):
        pairwise[f"{a}|{b}"] = pair_metrics(fps[a], fps[b], common, contested_mask)

    # within-run null: same label, roster vs null
    labels = sorted({k.split(":")[0] for k in keys})
    null_pairs = []
    for lab in labels:
        rk, nk = f"{lab}:roster", f"{lab}:null"
        if rk in fps and nk in fps:
            null_pairs.append((f"{rk}|{nk}", pairwise.get(f"{rk}|{nk}") or pairwise.get(f"{nk}|{rk}")))
    metrics = ["top1_disagreement", "js_divergence", "value_1_minus_pearson", "value_p95_abs"]
    null_band = {}
    for m in metrics:
        vals = [p[m] for _, p in null_pairs if p and p[m] is not None]
        null_band[m] = {"n": len(vals), "p95": float(np.percentile(vals, 95)) if vals else None,
                        "max": float(np.max(vals)) if vals else None, "values": vals}

    # cross-run roster-roster verdicts vs null band
    roster_keys = [k for k in keys if k.endswith(":roster")]
    verdicts = {}
    for a, b in combinations(roster_keys, 2):
        pk = f"{a}|{b}" if f"{a}|{b}" in pairwise else f"{b}|{a}"
        p = pairwise[pk]
        layer = {}
        # action layer: diverse if EITHER action metric exceeds its null p95
        action_div = any(p[m] is not None and null_band[m]["p95"] is not None and p[m] > null_band[m]["p95"]
                         for m in ("top1_disagreement", "js_divergence"))
        value_div = any(p[m] is not None and null_band[m]["p95"] is not None and p[m] > null_band[m]["p95"]
                        for m in ("value_1_minus_pearson", "value_p95_abs"))
        verdicts[pk] = {"action": "diverse" if action_div else "same", "value": "diverse" if value_div else "same",
                        "metrics": p}

    # GATE: shuffled-label control (misalign decisions of one model)
    a, b = roster_keys[0], roster_keys[1]
    rng = np.random.default_rng(0)
    shuffled_ids = list(common); perm = list(common)
    rng.shuffle(perm)
    fb_shuf = {did: fps[b][perm[i]] for i, did in enumerate(shuffled_ids)}
    aligned = pair_metrics(fps[a], fps[b], common, contested_mask)
    shuffled = pair_metrics(fps[a], fb_shuf, common, contested_mask)
    gate_shuffle = {"pair": f"{a}|{b}", "aligned": aligned, "shuffled": shuffled,
                    "passes": shuffled["top1_disagreement"] > aligned["top1_disagreement"] and shuffled["js_divergence"] > aligned["js_divergence"]}

    os.makedirs(args.out_dir, exist_ok=True)
    out = {"schema": "pokezero.diversity_pairwise.v1", "checkpoints": metas, "n_common_decisions": len(common),
           "contested_entropy_median": med, "pairwise": pairwise, "null_band": null_band,
           "verdicts": verdicts, "gate_shuffle": gate_shuffle}
    json.dump(out, open(os.path.join(args.out_dir, "pairwise.json"), "w"), indent=1)
    print("n_common:", len(common), "| null_band p95:", {m: (round(null_band[m]["p95"], 4) if null_band[m]["p95"] is not None else None) for m in metrics})
    print("shuffle gate passes:", gate_shuffle["passes"])
    for pk, v in verdicts.items():
        print(f"  {pk}: action={v['action']} value={v['value']} | top1_dis={v['metrics']['top1_disagreement']:.3f} js={v['metrics']['js_divergence']:.4f} v_pear={1-v['metrics']['value_1_minus_pearson']:.3f}")


if __name__ == "__main__":
    main()
