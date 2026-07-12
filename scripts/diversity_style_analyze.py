"""Stage A style layer: build z-scored behavioral style vectors per checkpoint from the
behavior_probe outputs, compute pairwise euclidean distances, the within-run null band,
and cross-run verdicts. Writes style.json (merged into the report's verdict matrix).

Features (from behavior_probe move_class_usage + scalars): attack/status/setup/heal/
hazard/clear/phaze move-class rates, avg game length, pivot rate, pivots per game,
forced switches per game, distinct-move count. 'other' rate is dropped (redundant;
the rates sum to 1). Conversion speed is not available from behavior_probe and is
omitted (documented).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from itertools import combinations

import numpy as np

MOVE_CLASSES = ["attack", "status", "setup", "heal", "hazard", "clear", "phaze"]
SCALARS = ["avg_turns", "pivot_rate", "pivots_per_game", "forced_switches_per_game", "distinct_moves"]
FEATURES = [f"mc_{c}" for c in MOVE_CLASSES] + SCALARS


def feature_vector(ck: dict) -> list[float]:
    mcu = ck.get("move_class_usage") or {}
    v = [float((mcu.get(c) or {}).get("rate", 0.0)) for c in MOVE_CLASSES]
    v += [float(ck.get(s, 0.0)) for s in SCALARS]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    keys, raw = [], []
    meta = {}
    for path in sorted(glob.glob(os.path.join(args.style_dir, "style-*.json"))):
        d = json.load(open(path))
        ck = d["checkpoints"][0]
        base = os.path.basename(path)[len("style-"):-len(".json")]  # <label>-<role>
        label, role = base.rsplit("-", 1)
        key = f"{label}:{role}"
        keys.append(key)
        raw.append(feature_vector(ck))
        meta[key] = {"label": label, "role": role, "games": ck.get("games"),
                     "checkpoint": ck.get("checkpoint")}

    X = np.array(raw, dtype=float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd  # z-score per feature across the roster
    zidx = {k: i for i, k in enumerate(keys)}

    def dist(a, b):
        return float(np.linalg.norm(Z[zidx[a]] - Z[zidx[b]]))

    pairwise = {f"{a}|{b}": dist(a, b) for a, b in combinations(keys, 2)}

    labels = sorted({k.split(":")[0] for k in keys})
    null_vals = []
    for lab in labels:
        rk, nk = f"{lab}:roster", f"{lab}:null"
        if rk in zidx and nk in zidx:
            null_vals.append(dist(rk, nk))
    null_p95 = float(np.percentile(null_vals, 95)) if null_vals else None

    roster = [k for k in keys if k.endswith(":roster")]
    verdicts = {}
    for a, b in combinations(roster, 2):
        d = dist(a, b)
        verdicts[f"{a}|{b}"] = {"style": "diverse" if (null_p95 is not None and d > null_p95) else "same",
                                "distance": d}

    os.makedirs(args.out_dir, exist_ok=True)
    out = {"schema": "pokezero.diversity_style.v1", "features": FEATURES, "meta": meta,
           "z_vectors": {k: Z[zidx[k]].tolist() for k in keys},
           "raw_vectors": {k: raw[zidx[k]] for k in keys},
           "pairwise_distance": pairwise,
           "null_band": {"n": len(null_vals), "p95": null_p95, "max": float(np.max(null_vals)) if null_vals else None, "values": null_vals},
           "verdicts": verdicts}
    json.dump(out, open(os.path.join(args.out_dir, "style.json"), "w"), indent=1)
    print("style null p95:", round(null_p95, 3) if null_p95 else None)
    for pk, v in verdicts.items():
        a, b = pk.split("|")[0].split(":")[0], pk.split("|")[1].split(":")[0]
        print(f"  {a} vs {b}: style={v['style']} dist={v['distance']:.2f}")


if __name__ == "__main__":
    main()
