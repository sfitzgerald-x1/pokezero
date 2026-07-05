"""Oracle-fit: derive candidate shaping weights FROM DATA instead of hand-tuning.

For every decision of every game in the given rollout-record JSONL(s), extract the
potential's component vector (the exact components pokezero.shaping's Phi uses: hp
differential, alive differential, per-status differentials, hazard-layer differential —
all player-relative and normalized) and the game's terminal outcome (+1/-1 from that
player's perspective; ties and capped games are skipped). Fit a logistic regression of
outcome on components. The fitted linear function IS the oracle potential: the direction
in component space most predictive of winning, i.e. the best linear guess at what a
converged value function would credit.

Reports:
  - the raw logit weight vector with per-component z-scores (Wald, from the Hessian) and
    sign sanity (own faints negative <=> alive-differential weight positive, foe status
    positive, ...),
  - McFadden pseudo-R^2 vs the intercept-only model,
  - a --shaping-weights-ready config: weights normalized into the Phi schema, anchored so
    hp_weight matches the WSE arm-1 anchor (0.5) by default, plus the implied magnitude
    scale so hand candidates can be sanity-scaled against the data.

Fitting is a pure-numpy Newton-Raphson (no sklearn in the neural extras; no new deps).

Usage:
    python scripts/shaping_oracle_fit.py --records runs/pool/records.jsonl \
      --out evals/shaping_oracle.json [--components hp,faint,status:par] [--anchor-hp 0.5]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy

from pokezero.collection import iter_rollout_records
from pokezero.shaping import (
    HP_COMPONENT,
    STATUS_COMPONENT_PREFIX,
    ShapingConfig,
    component_names,
    ground_truth_components_by_step_index,
)

RIDGE = 1e-6  # tiny L2 for numerical stability on rank-deficient corpora
MAX_NEWTON_ITERATIONS = 100
NEWTON_TOLERANCE = 1e-10


def extract_design_matrix(records_paths: list[Path], names: tuple[str, ...]):
    rows: list[list[float]] = []
    outcomes: list[float] = []
    games = 0
    skipped_games = 0
    for records_path in records_paths:
        for record in iter_rollout_records(records_path):
            terminal = record.terminal or record.trajectory.terminal
            if terminal is None or terminal.capped or terminal.winner is None:
                skipped_games += 1
                continue
            games += 1
            components_by_step = ground_truth_components_by_step_index(record)
            for step_index, step in enumerate(record.trajectory.steps):
                components = components_by_step[step_index]
                rows.append([components[name] for name in names])
                outcomes.append(1.0 if terminal.winner == step.player_id else -1.0)
    if not rows:
        raise ValueError("no decided games in the given records; cannot fit an oracle.")
    return numpy.asarray(rows, dtype=numpy.float64), numpy.asarray(outcomes, dtype=numpy.float64), games, skipped_games


def fit_logistic(features: numpy.ndarray, outcomes: numpy.ndarray):
    """Newton-Raphson logistic fit of P(win) on [1, features]; returns (weights, stderr, ll, ll_null)."""
    labels = (outcomes > 0).astype(numpy.float64)
    design = numpy.concatenate([numpy.ones((features.shape[0], 1)), features], axis=1)
    weights = numpy.zeros(design.shape[1])
    for _ in range(MAX_NEWTON_ITERATIONS):
        logits = design @ weights
        probabilities = 1.0 / (1.0 + numpy.exp(-numpy.clip(logits, -35.0, 35.0)))
        gradient = design.T @ (labels - probabilities) - RIDGE * weights
        working = numpy.clip(probabilities * (1.0 - probabilities), 1e-12, None)
        hessian = (design * working[:, None]).T @ design + RIDGE * numpy.eye(design.shape[1])
        step = numpy.linalg.solve(hessian, gradient)
        weights = weights + step
        if float(numpy.max(numpy.abs(step))) < NEWTON_TOLERANCE:
            break
    logits = design @ weights
    probabilities = 1.0 / (1.0 + numpy.exp(-numpy.clip(logits, -35.0, 35.0)))
    log_likelihood = float(
        numpy.sum(labels * numpy.log(numpy.clip(probabilities, 1e-12, None)))
        + numpy.sum((1.0 - labels) * numpy.log(numpy.clip(1.0 - probabilities, 1e-12, None)))
    )
    base_rate = float(numpy.clip(labels.mean(), 1e-12, 1.0 - 1e-12))
    log_likelihood_null = float(
        labels.sum() * math.log(base_rate) + (len(labels) - labels.sum()) * math.log(1.0 - base_rate)
    )
    working = numpy.clip(probabilities * (1.0 - probabilities), 1e-12, None)
    hessian = (design * working[:, None]).T @ design + RIDGE * numpy.eye(design.shape[1])
    standard_errors = numpy.sqrt(numpy.diag(numpy.linalg.inv(hessian)))
    return weights, standard_errors, log_likelihood, log_likelihood_null


def shaping_config_from_weights(
    weights_by_name: dict[str, float], *, anchor_hp: float | None
) -> tuple[ShapingConfig, float]:
    """Map fitted component weights into the Phi weight schema, optionally hp-anchored.

    Components are already sign-folded (foe status/hazard positive), so the schema map is
    direct. Scale: logit weights live in log-odds units; anchoring hp_weight to the WSE
    arm-1 value (0.5) preserves the fitted RATIOS at a return-compatible magnitude. The
    returned scale is (anchored / raw), the number to multiply other raw logit weights by.
    """
    raw_hp = weights_by_name.get(HP_COMPONENT, 0.0)
    if anchor_hp is None or raw_hp == 0.0:
        scale = 1.0
    else:
        scale = anchor_hp / raw_hp
    status_weights = tuple(
        (name[len(STATUS_COMPONENT_PREFIX) :], scale * weight)
        for name, weight in weights_by_name.items()
        if name.startswith(STATUS_COMPONENT_PREFIX)
    )
    config = ShapingConfig(
        hp_weight=scale * raw_hp,
        faint_weight=scale * weights_by_name.get("faint", 0.0),
        status_weights=status_weights,
        hazard_weight=scale * weights_by_name.get("hazard", 0.0),
    )
    return config, scale


EXPECTED_SIGNS = {"hp": +1, "faint": +1, "hazard": +1}


def _expected_sign(name: str) -> int:
    # Every component is folded player-relative-positive (own hp/alive up, foe status or
    # foe hazards up => better for us), so a sane oracle weight is positive across the board.
    return EXPECTED_SIGNS.get(name, +1)


def parse_component_selection(raw: str | None) -> tuple[str, ...]:
    names = component_names()
    if raw is None:
        return names
    selected: list[str] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token == "status":
            selected.extend(name for name in names if name.startswith(STATUS_COMPONENT_PREFIX))
        elif token in names:
            selected.append(token)
        else:
            raise ValueError(f"unknown component {token!r}; expected subset of {', '.join(names)} or 'status'.")
    deduped = tuple(dict.fromkeys(selected))
    if not deduped:
        raise ValueError("--components selected nothing.")
    return deduped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", action="append", required=True, type=Path, help="Rollout-record JSONL. May repeat.")
    parser.add_argument(
        "--components",
        default=None,
        help="Comma-separated component subset to fit (e.g. 'hp,faint' or 'status' or 'hp,status:par').",
    )
    parser.add_argument(
        "--anchor-hp",
        type=float,
        default=0.5,
        help="Scale the suggested config so hp_weight equals this (WSE arm-1 anchor). 0 disables anchoring.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    names = parse_component_selection(args.components)
    started = time.perf_counter()
    features, outcomes, games, skipped = extract_design_matrix(list(args.records), names)
    weights, standard_errors, log_likelihood, log_likelihood_null = fit_logistic(features, outcomes)
    elapsed = time.perf_counter() - started

    weights_by_name = {name: float(weights[1 + index]) for index, name in enumerate(names)}
    z_by_name = {
        name: (float(weights[1 + index]) / float(standard_errors[1 + index]) if standard_errors[1 + index] > 0 else 0.0)
        for index, name in enumerate(names)
    }
    pseudo_r2 = 1.0 - (log_likelihood / log_likelihood_null) if log_likelihood_null != 0.0 else None
    anchor = args.anchor_hp if args.anchor_hp else None
    suggested, scale = shaping_config_from_weights(weights_by_name, anchor_hp=anchor)

    sign_rows = []
    for name in names:
        weight = weights_by_name[name]
        z = z_by_name[name]
        expected = _expected_sign(name)
        agrees = (weight == 0.0) or (weight > 0) == (expected > 0)
        sign_rows.append(
            {"component": name, "weight": weight, "z": z, "expected_sign": "+", "sign_ok": bool(agrees)}
        )

    report = {
        "records": [str(path) for path in args.records],
        "games_used": games,
        "games_skipped_undecided": skipped,
        "examples": int(features.shape[0]),
        "components": list(names),
        "intercept": float(weights[0]),
        "fit_seconds": elapsed,
        "weights": weights_by_name,
        "z_scores": z_by_name,
        "sign_table": sign_rows,
        "mcfadden_pseudo_r2": pseudo_r2,
        "log_likelihood": log_likelihood,
        "log_likelihood_null": log_likelihood_null,
        "anchor_hp": anchor,
        "logit_to_return_scale": scale,
        "suggested_shaping_weights": suggested.to_dict(),
        "suggested_shaping_weights_json": suggested.canonical_json(),
    }

    print(f"examples: {report['examples']}  games: {games} (skipped {skipped} undecided)  fit: {elapsed:.2f}s")
    print(f"pseudo-R^2 (McFadden): {pseudo_r2:.4f}" if pseudo_r2 is not None else "pseudo-R^2: -")
    header = f"{'component':14} {'logit_w':>10} {'z':>8} {'sign':>5}"
    print(header)
    print("-" * len(header))
    for row in sign_rows:
        print(
            f"{row['component']:14} {row['weight']:10.4f} {row['z']:8.2f} "
            f"{'ok' if row['sign_ok'] else 'FLIP':>5}"
        )
    print(f"scale (logit -> return units, hp anchored at {anchor}): {scale:.5f}")
    print("suggested --shaping-weights:")
    print(f"  '{report['suggested_shaping_weights_json']}'")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
