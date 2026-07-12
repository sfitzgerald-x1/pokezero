"""Stage B matchup layer: from the round-robin benchmark summaries, build the seat-
symmetric win matrix, fit Bradley-Terry (one latent strength axis), and test whether the
observed outcomes are non-transitive beyond that axis (rock-paper-scissors structure =
strategic diversity the strength axis cannot explain).

Each rr-<A>.json holds A's win rate vs every other roster checkpoint (already seat-
balanced by the benchmark). The pair (A,B) is measured from both A's and B's jobs; we
average for a seat- and job-symmetric estimate.

Outputs matchup.json: BT strengths, residual matrix, the intransitivity statistic with a
bootstrap null (binomial resample of each matchup's wins), and per-pair verdicts.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from itertools import combinations

import numpy as np


def load_matches(rr_dir):
    """Return wins[(a,b)] and games[(a,b)] for ordered pairs from all rr-*.json."""
    wins, games = {}, {}
    labels = set()
    for path in sorted(glob.glob(os.path.join(rr_dir, "rr-*.json"))):
        d = json.load(open(path))
        a = os.path.basename(path)[len("rr-"):-len(".json")]
        # normalize case to match roster labels via the head-to-head id
        for h in d.get("head_to_heads", []):
            b = h.get("second_policy_id")
            if b in ("random-legal", "simple-legal", "max-damage", "foul-play"):
                continue
            aid = h.get("first_policy_id") or a
            labels.add(aid); labels.add(b)
            wins[(aid, b)] = int(h["first_policy_wins"])
            games[(aid, b)] = int(h["games"])
    return wins, games, sorted(labels)


def sym_winrate(labels, wins, games):
    """W[i][j] = P(i beats j), averaging the (i,j) and (j,i) job measurements."""
    n = len(labels)
    idx = {l: i for i, l in enumerate(labels)}
    W = np.full((n, n), np.nan)
    N = np.zeros((n, n))
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i == j:
                continue
            ests, tot = [], 0
            if (a, b) in wins:
                ests.append(wins[(a, b)] / games[(a, b)]); tot += games[(a, b)]
            if (b, a) in wins:  # b's win rate vs a -> a's win rate = 1 - that
                ests.append(1 - wins[(b, a)] / games[(b, a)]); tot += games[(b, a)]
            if ests:
                W[i][j] = float(np.mean(ests)); N[i][j] = tot
    return W, N, idx


def bradley_terry(labels, W, N):
    """MM (Zermelo) fit from the symmetric win matrix scaled to integer-ish counts."""
    n = len(labels)
    wins_i = np.zeros(n)
    nij = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and not np.isnan(W[i][j]):
                g = N[i][j] or 1.0
                nij[i][j] = g
                wins_i[i] += W[i][j] * g
    p = np.ones(n)
    for _ in range(500):
        p_new = p.copy()
        for i in range(n):
            denom = sum(nij[i][j] / (p[i] + p[j]) for j in range(n) if j != i and (nij[i][j] + nij[j][i]) > 0)
            if denom > 0:
                p_new[i] = wins_i[i] / denom
        p_new = np.maximum(p_new, 1e-9)
        p_new /= p_new.mean()
        if np.max(np.abs(p_new - p)) < 1e-9:
            p = p_new; break
        p = p_new
    strengths = np.log(p)
    pred = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                pred[i][j] = p[i] / (p[i] + p[j])
    return strengths, pred


def intransitivity(W):
    """Sum over unordered triples of the directed-3-cycle magnitude. A triple {i,j,k} is a
    cycle iff the three directed edges around the fixed loop i->j->k->i all have the same
    sign (all wins or all losses); its magnitude is the product of the |W-0.5| margins.
    Exactly 0 for any transitive win matrix; positive only under genuine rock-paper-scissors.
    """
    n = W.shape[0]
    s = 0.0
    for i, j, k in combinations(range(n), 3):
        edges = [W[i][j] - 0.5, W[j][k] - 0.5, W[k][i] - 0.5]
        signs = [(1 if e > 0 else (-1 if e < 0 else 0)) for e in edges]
        if signs[0] != 0 and signs[0] == signs[1] == signs[2]:
            s += abs(edges[0] * edges[1] * edges[2])
    return float(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rr-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    wins, games, labels = load_matches(args.rr_dir)
    W, N, idx = sym_winrate(labels, wins, games)
    strengths, pred = bradley_terry(labels, W, N)
    resid = np.where(np.isnan(W), 0.0, W - pred)
    cyc = intransitivity(np.where(np.isnan(W), 0.5, W))

    # bootstrap null: resample each measured matchup's wins ~ Binomial(n, W), recompute cycle
    rng = np.random.default_rng(0)
    n = len(labels)
    cyc_boot = []
    for _ in range(args.bootstrap):
        Wb = W.copy()
        for i in range(n):
            for j in range(n):
                if i != j and not np.isnan(W[i][j]) and N[i][j] > 0:
                    Wb[i][j] = rng.binomial(int(N[i][j]), W[i][j]) / N[i][j]
        cyc_boot.append(intransitivity(np.where(np.isnan(Wb), 0.5, Wb)))
    cyc_boot = np.array(cyc_boot)
    # null = expected cycle under pure BT (transitive): resample from PRED
    cyc_null = []
    for _ in range(args.bootstrap):
        Wn = pred.copy()
        for i in range(n):
            for j in range(n):
                if i != j and N[i][j] > 0:
                    Wn[i][j] = rng.binomial(int(N[i][j]), pred[i][j]) / N[i][j]
        cyc_null.append(intransitivity(Wn))
    cyc_null = np.array(cyc_null)
    p_value = float(np.mean(cyc_null >= cyc))

    # per-pair verdict: matchup-diverse if the seat-symmetric residual is large AND
    # the pair sits in the non-transitive structure (|resid| beyond the null 95th pct)
    resid_null_p95 = float(np.percentile(np.abs(
        [pred[i][j] - (rng.binomial(int(N[i][j]), pred[i][j]) / N[i][j] if N[i][j] > 0 else pred[i][j])
         for i in range(n) for j in range(n) if i != j and N[i][j] > 0]), 95))
    verdicts = {}
    for i, j in combinations(range(n), 2):
        r = abs(resid[i][j])
        verdicts[f"{labels[i]}|{labels[j]}"] = {
            "matchup": "diverse" if r > resid_null_p95 else "same",
            "win_rate_i_over_j": None if np.isnan(W[i][j]) else round(float(W[i][j]), 3),
            "bt_pred": round(float(pred[i][j]), 3), "residual": round(float(resid[i][j]), 3),
            "games": int(N[i][j])}

    os.makedirs(args.out_dir, exist_ok=True)
    out = {"schema": "pokezero.diversity_matchup.v1", "labels": labels,
           "bt_strengths": {l: round(float(s), 3) for l, s in zip(labels, strengths)},
           "win_matrix": {labels[i]: {labels[j]: (None if np.isnan(W[i][j]) else round(float(W[i][j]), 3)) for j in range(n) if j != i} for i in range(n)},
           "residual_matrix": {labels[i]: {labels[j]: round(float(resid[i][j]), 3) for j in range(n) if j != i} for i in range(n)},
           "intransitivity": {"observed": round(cyc, 5), "null_mean": round(float(cyc_null.mean()), 5),
                              "null_p95": round(float(np.percentile(cyc_null, 95)), 5), "p_value": p_value,
                              "significant_cycles": p_value < 0.05},
           "resid_null_p95": round(resid_null_p95, 4), "verdicts": verdicts}
    json.dump(out, open(os.path.join(args.out_dir, "matchup.json"), "w"), indent=1)
    print("BT strengths:", out["bt_strengths"])
    print(f"intransitivity: observed={cyc:.4f} null_mean={cyc_null.mean():.4f} p={p_value:.3f} "
          f"significant_cycles={out['intransitivity']['significant_cycles']}")
    for pk, v in verdicts.items():
        a, b = pk.split("|")
        print(f"  {a} vs {b}: wr={v['win_rate_i_over_j']} bt_pred={v['bt_pred']} resid={v['residual']} -> matchup={v['matchup']}")


if __name__ == "__main__":
    main()
