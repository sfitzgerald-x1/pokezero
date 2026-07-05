"""Supervised shaping-candidate ranker: the no-RL evaluator for reward configs.

Rationale: a shaping config that cannot make a SUPERVISED value head price hazards/status
will never do it in RL — so rank candidates by training identical tiny models on
identical observations whose only difference is the shaped return target, then measure
what each head learned. Minutes per candidate on CPU; kills bad configs before any
micro-RL arm is spent.

Per candidate (fixed seed, fixed epochs, identical model config):
  1. retroactively compute shaped per-step returns over the frozen records corpus
     (monte-carlo shaped return-to-go via the exact pokezero.shaping + dataset machinery
     collection/training use — TransformerTrainingConfig.shaping_weights);
  2. train a tiny transformer (policy CE + value MSE; the value target is the shaped
     return) on the train split;
  3. evaluate:
     [i]  dV hazard response on injected state pairs, reusing scripts/hazard_probe.py's
          injection primitives (pokezero.checkpoint_factors.with_self_spikes/with_opp_spikes)
          over a fixed scripted-driver corpus — self-spikes 0->3 (expect V down), opp-spikes
          0->3 (expect V up);
     [ii] terminal-outcome prediction on held-out games, three statistics per candidate:
            corrected Pearson  corr(pred + Phi_cand, terminal) — the eval-facing value
                               (the shaped optimum is V' = V - Phi, so pred + Phi is what
                               search/eval would consume);
            Phi-alone Pearson  corr(Phi_cand, terminal) — the free ride. Pearson is
                               scale-invariant, so a DEAD head (pred ~ tiny noise) still
                               scores corrected ~ Phi-alone: corrected Pearson alone can
                               be carried entirely by the potential (#510 review, HIGH-1);
            head marginal      the PARTIAL correlation of the raw prediction with terminal
                               controlling for Phi_cand:
                                 r_{pt.f} = (r_pt - r_pf*r_tf)/sqrt((1-r_pf^2)(1-r_tf^2)).
                               This is the gate that measures the HEAD. Why partial
                               correlation and not the additive delta
                               (corrected - Phi-alone): the delta inherits the pred/Phi
                               VARIANCE ratio — a dead head sits at delta ~ 0 (admitted by
                               any -eps margin), while a genuinely good head under a
                               huge-|Phi| candidate is variance-diluted below Phi-alone
                               and unfairly killed. The partial correlation is scale-free
                               in both pred and Phi, reduces to the plain raw Pearson for
                               the control (Phi = 0 has no variance to control for), and
                               collapses to ~0 for a head that is dead or merely
                               re-encodes Phi. It is the standard statistic for "marginal
                               predictive contribution given a covariate".
     [iii] calibration of the corrected prediction vs terminal (bias + 10-bin ECE).

Ranking (documented composite):
    delta_v_score = (max(0, -value_self_response) + max(0, value_opp_response))
                    / max(CONTROL head value spread, 1e-6)
  (denominator is the CONTROL's spread: a saturated-target head that collapses its own
  value spread must not amplify its noise dV into a winning score.)
  "retained" requires ALL of:
    - corrected Pearson >= floor, floor = max(control - retention*|control|,
      --min-pearson-floor). The relative term uses |control| so a noisy NEGATIVE control
      can never place the floor above the control itself; the absolute minimum keeps the
      gate meaningful when the control is ~0;
    - head marginal >= --min-head-marginal (the head must beat Phi's free ride);
    - value spread >= --min-spread-frac x control spread (collapsed-head guard);
    - corrected ECE <= --max-ece (calibration sanity: an ECE of 8 next to retained=yes
      must be impossible to print).
  Retained candidates sort by delta_v_score descending (dV is the primary read); failed
  candidates sort by HEAD MARGINAL descending (least-broken first — corrected Pearson is
  the Phi free ride for failed heads, so it is only the final tiebreak).

Validity self-checks (on by default): two synthetic probes ride along — wse-arm1 x -1
(inverted signs: the correction anti-corrects) and wse-arm1 x
--validity-saturation-factor (returns saturate the +-1 clip and the head learns nothing
— the degenerate case from the #510 review). BOTH must rank strictly below the unshaped
control or the tool refuses to emit a ranking (exit 2; diagnostics still printed and
written with ranking_withheld=true). A non-positive control Pearson marks the whole run
LOW-CONFIDENCE with a loud warning: the corpus cannot support the retention comparison.

The candidate list MUST include the unshaped control ("shaping": null) — it defines the
Pearson budget and the dV spread denominator.

Usage:
    python scripts/shaping_ranker.py --records records.jsonl \
      --candidates candidates.json --showdown-root ~/pokemon-showdown \
      --out evals/shaping_ranker.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from pokezero.checkpoint_factors import build_corpus, with_opp_spikes, with_self_spikes
from pokezero.collection import read_rollout_records, write_rollout_record
from pokezero.dex import load_showdown_dex_cached
from pokezero.neural_policy import (
    TransformerPolicyConfig,
    TransformerTrainingConfig,
    evaluate_transformer_observation_value,
    require_torch,
    train_transformer_policy,
)
from pokezero.randbat_vocab import gen3_category_vocabulary
from pokezero.shaping import (
    SHAPING_PRESETS,
    ShapingConfig,
    ground_truth_components_by_step_index,
    potential_from_components,
    resolve_shaping_config,
)
from pokezero.showdown import DEFAULT_REPLAY_OBSERVATION_SPEC, observation_from_player_state

SPIKES_LAYERS_PROBE = 3
VALIDITY_INVERTED_LABEL = "validity-inverted(builtin)"
VALIDITY_SATURATING_LABEL = "validity-saturating(builtin)"


def load_candidates(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("candidates file must be a non-empty JSON list.")
    candidates = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict) or "label" not in entry or "shaping" not in entry:
            raise ValueError(f"candidate #{index} must be an object with 'label' and 'shaping' keys.")
        shaping = resolve_shaping_config(entry["shaping"]) if entry["shaping"] is not None else None
        candidates.append({"label": str(entry["label"]), "shaping": shaping, "builtin": False})
    labels = [candidate["label"] for candidate in candidates]
    if len(set(labels)) != len(labels):
        raise ValueError("candidate labels must be unique.")
    if not any(candidate["shaping"] is None for candidate in candidates):
        raise ValueError("candidates must include the unshaped control ('shaping': null).")
    return candidates


def scale_shaping_config(config: ShapingConfig, factor: float) -> ShapingConfig:
    return ShapingConfig(
        hp_weight=config.hp_weight * factor,
        faint_weight=config.faint_weight * factor,
        status_weights=tuple((status, weight * factor) for status, weight in config.status_weights),
        hazard_weight=config.hazard_weight * factor,
        terminal_mode=config.terminal_mode,
    )


def builtin_validity_candidates(saturation_factor: float) -> list[dict]:
    base = SHAPING_PRESETS["wse-arm1"]
    return [
        {"label": VALIDITY_INVERTED_LABEL, "shaping": scale_shaping_config(base, -1.0), "builtin": True},
        {
            "label": VALIDITY_SATURATING_LABEL,
            "shaping": scale_shaping_config(base, saturation_factor),
            "builtin": True,
        },
    ]


def split_records(records, heldout_fraction: float):
    if len(records) < 2:
        raise ValueError("need at least 2 games to hold out an eval split.")
    heldout_count = max(1, int(round(len(records) * heldout_fraction)))
    heldout_count = min(heldout_count, len(records) - 1)
    return records[:-heldout_count], records[-heldout_count:]


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def _partial_pearson(predictions: list[float], outcomes: list[float], phis: list[float]) -> float | None:
    """Partial correlation of prediction with outcome controlling for Phi (see module doc).

    Degenerate cases: Phi (near-)constant -> plain raw Pearson (the control's case);
    prediction (near-)constant, or perfectly collinear with Phi -> 0 (a head with no
    signal of its own contributes no marginal information).
    """
    r_pt = _pearson(predictions, outcomes)
    if r_pt is None:
        return 0.0
    r_pf = _pearson(predictions, phis)
    r_tf = _pearson(outcomes, phis)
    if r_pf is None or r_tf is None:
        # Phi has no variance (unshaped control / degenerate metadata): nothing to control for.
        return r_pt
    denominator_sq = (1.0 - r_pf * r_pf) * (1.0 - r_tf * r_tf)
    if denominator_sq <= 1e-12:
        return 0.0
    return (r_pt - r_pf * r_tf) / math.sqrt(denominator_sq)


def _ece(predictions: list[float], outcomes: list[float], bins: int = 10) -> float | None:
    if not predictions:
        return None
    binned: dict[int, list[int]] = {}
    for index, prediction in enumerate(predictions):
        clipped = min(1.0, max(-1.0, prediction))
        bin_index = min(bins - 1, int((clipped + 1.0) / 2.0 * bins))
        binned.setdefault(bin_index, []).append(index)
    total = len(predictions)
    error = 0.0
    for indices in binned.values():
        mean_prediction = sum(predictions[i] for i in indices) / len(indices)
        mean_outcome = sum(outcomes[i] for i in indices) / len(indices)
        error += (len(indices) / total) * abs(mean_prediction - mean_outcome)
    return error


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def evaluate_candidate_model(model, result, *, shaping, heldout_records, corpus, vocab, dex, value_states: int):
    def value_of(state) -> float:
        observation = observation_from_player_state(
            state, category_vocab=vocab, spec=DEFAULT_REPLAY_OBSERVATION_SPEC, dex=dex
        )
        return evaluate_transformer_observation_value(model=model, result=result, observations=[observation])

    # [i] dV hazard response on injected pairs (shared primitives with hazard_probe).
    base_values: list[float] = []
    self_deltas: list[float] = []
    opp_deltas: list[float] = []
    for entry in corpus[:value_states]:
        state = entry.state
        base = value_of(with_self_spikes(state, 0))
        base_values.append(base)
        self_deltas.append(value_of(with_self_spikes(state, SPIKES_LAYERS_PROBE)) - base)
        opp_base = value_of(with_opp_spikes(state, 0))
        opp_deltas.append(value_of(with_opp_spikes(state, SPIKES_LAYERS_PROBE)) - opp_base)
    value_spread = _std(base_values)
    value_self_response = sum(self_deltas) / len(self_deltas) if self_deltas else 0.0
    value_opp_response = sum(opp_deltas) / len(opp_deltas) if opp_deltas else 0.0

    # [ii]+[iii] terminal-outcome prediction on held-out games (see module docstring for
    # why the gate statistic is the PARTIAL correlation controlling for Phi).
    raw_predictions: list[float] = []
    corrected_predictions: list[float] = []
    phi_values: list[float] = []
    outcomes: list[float] = []
    for record in heldout_records:
        terminal = record.terminal or record.trajectory.terminal
        if terminal is None or terminal.capped or terminal.winner is None:
            continue
        components_by_step = ground_truth_components_by_step_index(record)
        for step_index, step in enumerate(record.trajectory.steps):
            prediction = evaluate_transformer_observation_value(
                model=model, result=result, observations=[step.observation]
            )
            phi = (
                potential_from_components(components_by_step[step_index], shaping)
                if shaping is not None
                else 0.0
            )
            raw_predictions.append(prediction)
            corrected_predictions.append(prediction + phi)
            phi_values.append(phi)
            outcomes.append(1.0 if terminal.winner == step.player_id else -1.0)

    return {
        "value_states": min(value_states, len(corpus)),
        "value_spread": value_spread,
        "value_self_response": value_self_response,
        "value_opp_response": value_opp_response,
        "heldout_examples": len(raw_predictions),
        "terminal_pearson": _pearson(corrected_predictions, outcomes),
        "terminal_pearson_uncorrected": _pearson(raw_predictions, outcomes),
        "phi_pearson": _pearson(phi_values, outcomes),
        "head_marginal": _partial_pearson(raw_predictions, outcomes, phi_values),
        "terminal_bias": (
            sum(p - o for p, o in zip(corrected_predictions, outcomes)) / len(corrected_predictions)
            if corrected_predictions
            else None
        ),
        "terminal_ece": _ece(corrected_predictions, outcomes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, required=True, help="Frozen rollout-record JSONL corpus.")
    parser.add_argument("--candidates", type=Path, required=True, help="JSON list of {label, shaping} candidates.")
    parser.add_argument("--showdown-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7, help="Torch seed re-applied before every candidate.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--discount", type=float, default=1.0, help="Return discount (and shaping gamma).")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=128)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--heldout-fraction", type=float, default=0.2, help="Trailing game fraction held out.")
    parser.add_argument(
        "--pearson-retention",
        type=float,
        default=0.10,
        help="Allowed drop of corrected Pearson vs control, as a fraction of |control|.",
    )
    parser.add_argument(
        "--min-pearson-floor",
        type=float,
        default=0.0,
        help=(
            "Absolute minimum retention floor; keeps the gate meaningful when the control's "
            "Pearson is near or below zero (which also marks the run LOW-CONFIDENCE)."
        ),
    )
    parser.add_argument(
        "--min-head-marginal",
        type=float,
        default=0.05,
        help=(
            "Minimum partial correlation of the head's raw prediction with terminal outcome "
            "controlling for Phi_candidate. Blocks candidates whose corrected Pearson is "
            "carried entirely by the potential (dead or Phi-parroting heads)."
        ),
    )
    parser.add_argument(
        "--min-spread-frac",
        type=float,
        default=0.25,
        help="Minimum candidate head value-spread as a fraction of the control head's spread.",
    )
    parser.add_argument(
        "--max-ece",
        type=float,
        default=2.0,
        help="Maximum corrected-prediction ECE; saturated-target heads blow far past this.",
    )
    parser.add_argument(
        "--validity-checks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run the built-in inverted-signs and saturating (wse-arm1 x factor) probes; both "
            "must rank below the control or no ranking is emitted (exit 2)."
        ),
    )
    parser.add_argument(
        "--validity-saturation-factor",
        type=float,
        default=80.0,
        help="Scale for the built-in saturating validity probe (the arm1 x N clip-saturation case).",
    )
    parser.add_argument("--corpus-games", type=int, default=8, help="Scripted-driver games for the dV state corpus.")
    parser.add_argument("--corpus-states", type=int, default=200, help="Max dV corpus states collected.")
    parser.add_argument("--value-states", type=int, default=120, help="Corpus states probed per candidate (5 value evals each).")
    parser.add_argument("--work-dir", type=Path, default=None, help="Where split JSONLs are written (default: alongside --out or CWD).")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    torch = require_torch()
    candidates = load_candidates(args.candidates)
    # Control first: its head spread and Pearson define the dV denominator and the floor.
    candidates.sort(key=lambda candidate: candidate["shaping"] is not None)
    if args.validity_checks:
        existing = {candidate["label"] for candidate in candidates}
        candidates.extend(
            candidate
            for candidate in builtin_validity_candidates(args.validity_saturation_factor)
            if candidate["label"] not in existing
        )
    records = list(read_rollout_records(args.records))
    train_records, heldout_records = split_records(records, args.heldout_fraction)
    work_dir = args.work_dir or (args.out.parent if args.out is not None else Path("."))
    work_dir.mkdir(parents=True, exist_ok=True)
    train_path = work_dir / "shaping_ranker_train_split.jsonl"
    with train_path.open("w", encoding="utf-8") as handle:
        for record in train_records:
            write_rollout_record(handle, record)

    print(f"[ranker] corpus: {len(train_records)} train games / {len(heldout_records)} heldout games")
    print(f"[ranker] building dV state corpus ({args.corpus_games} scripted games)...")
    corpus = build_corpus(
        str(args.showdown_root), num_games=args.corpus_games, max_states=args.corpus_states
    )
    vocab = gen3_category_vocabulary(args.showdown_root)
    dex = load_showdown_dex_cached(args.showdown_root)
    model_config = TransformerPolicyConfig.compact_category(
        category_vocab=vocab.tokens,
        category_oov_buckets=vocab.oov_buckets,
        policy_id="shaping-ranker-tiny",
        embedding_dim=args.embedding_dim,
        transformer_layers=args.layers,
        attention_heads=args.attention_heads,
        feedforward_dim=args.feedforward_dim,
        dropout=0.0,
    )
    print(f"[ranker] dV corpus: {len(corpus)} states; tiny model: {args.embedding_dim}d x{args.layers}")

    rows = []
    for candidate in candidates:
        label = candidate["label"]
        shaping = candidate["shaping"]
        shaping_json = shaping.canonical_json() if shaping is not None else None
        training_config = TransformerTrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            discount=args.discount,
            value_loss_weight=args.value_loss_weight,
            shaping_weights=shaping_json,
            device="cpu",
        )
        torch.manual_seed(args.seed)
        started = time.perf_counter()
        model, result = train_transformer_policy(
            train_path, model_config=model_config, training_config=training_config
        )
        train_seconds = time.perf_counter() - started
        evaluation = evaluate_candidate_model(
            model,
            result,
            shaping=shaping,
            heldout_records=heldout_records,
            corpus=corpus,
            vocab=vocab,
            dex=dex,
            value_states=args.value_states,
        )
        eval_seconds = time.perf_counter() - started - train_seconds
        rows.append(
            {
                "label": label,
                "builtin_validity_probe": bool(candidate["builtin"]),
                "shaping": shaping.to_dict() if shaping is not None else None,
                "train_seconds": train_seconds,
                "eval_seconds": eval_seconds,
                **evaluation,
            }
        )
        print(
            f"[ranker] {label}: dV_self={evaluation['value_self_response']:+.4f} "
            f"dV_opp={evaluation['value_opp_response']:+.4f} spread={evaluation['value_spread']:.4f} "
            f"pearson*={_fmt(evaluation['terminal_pearson'])} raw={_fmt(evaluation['terminal_pearson_uncorrected'])} "
            f"phi={_fmt(evaluation['phi_pearson'])} marginal={_fmt(evaluation['head_marginal'])} "
            f"({train_seconds:.1f}s train + {eval_seconds:.1f}s eval)"
        )

    control = next(row for row in rows if row["shaping"] is None)
    control_pearson = control["terminal_pearson"] or 0.0
    control_spread = control["value_spread"]
    low_confidence = control_pearson <= 0.0
    if low_confidence:
        print(
            "\nWARNING: the unshaped control's terminal Pearson is non-positive "
            f"({control_pearson:.4f}) — this corpus cannot support the retention "
            "comparison and the entire ranking is LOW-CONFIDENCE. Grow the corpus.",
            file=sys.stderr,
        )
    # The relative term uses |control| so a negative control can never place the floor
    # ABOVE the control itself (#510 review MED-1); the absolute minimum keeps the gate
    # meaningful when the control is ~0.
    pearson_floor = max(
        control_pearson - args.pearson_retention * abs(control_pearson),
        args.min_pearson_floor,
    )
    for row in rows:
        pearson = row["terminal_pearson"] or 0.0
        marginal = row["head_marginal"] or 0.0
        ece = row["terminal_ece"] if row["terminal_ece"] is not None else float("inf")
        spread_frac = row["value_spread"] / control_spread if control_spread > 0 else 0.0
        # Control-spread denominator: a collapsed head must not amplify its own noise dV.
        row["delta_v_score"] = (
            max(0.0, -row["value_self_response"]) + max(0.0, row["value_opp_response"])
        ) / max(control_spread, 1e-6)
        row["spread_frac"] = spread_frac
        row["pearson_floor"] = pearson_floor
        checks = {
            "pearson": bool(pearson >= pearson_floor),
            "marginal": bool(marginal >= args.min_head_marginal),
            "spread": bool(spread_frac >= args.min_spread_frac),
            "ece": bool(ece <= args.max_ece),
        }
        row["retention_checks"] = checks
        row["pearson_retained"] = all(checks.values())
    ranked = sorted(
        rows,
        key=lambda row: (
            not row["pearson_retained"],
            # Retained: primary read is the dV response. Failed: order by the HEAD'S OWN
            # signal (the marginal), not corrected Pearson — for failed heads corrected
            # Pearson is exactly the Phi free ride this gate exists to discount, so a
            # dead saturated head would otherwise float to the top of the failed group.
            -row["delta_v_score"] if row["pearson_retained"] else 0.0,
            0.0 if row["pearson_retained"] else -(row["head_marginal"] or -1.0),
            -(row["terminal_pearson"] or -1.0),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank

    header = (
        f"{'rank':>4} {'label':28} {'dV_self':>8} {'dV_opp':>8} {'spread':>7} {'dV_score':>9} "
        f"{'pearson*':>8} {'phi_p':>8} {'marginal':>8} {'raw_p':>8} {'ece':>7} {'sec':>6}  retained"
    )
    print()
    print(header)
    print("-" * len(header))
    for row in ranked:
        failed = [name for name, passed in row["retention_checks"].items() if not passed]
        retained_text = "yes" if row["pearson_retained"] else f"NO({','.join(failed)})"
        print(
            f"{row['rank']:>4} {row['label'][:28]:28} {row['value_self_response']:8.4f} "
            f"{row['value_opp_response']:8.4f} {row['value_spread']:7.4f} {row['delta_v_score']:9.4f} "
            f"{_fmt(row['terminal_pearson']):>8} {_fmt(row['phi_pearson']):>8} "
            f"{_fmt(row['head_marginal']):>8} {_fmt(row['terminal_pearson_uncorrected']):>8} "
            f"{(row['terminal_ece'] if row['terminal_ece'] is not None else float('nan')):7.3f} "
            f"{row['train_seconds'] + row['eval_seconds']:6.0f}  {retained_text}"
        )
    print(
        "\npearson* = corrected (pred + Phi_cand) vs terminal; phi_p = Phi alone; marginal = "
        "partial corr of raw pred vs terminal controlling for Phi (the head's own signal)."
        f"\ncomposite: retained requires pearson* >= {pearson_floor:.4f} "
        f"(control {control_pearson:.4f}, retention {args.pearson_retention:.0%}, abs floor "
        f"{args.min_pearson_floor:g}) AND marginal >= {args.min_head_marginal:g} AND spread >= "
        f"{args.min_spread_frac:g}x control AND ece <= {args.max_ece:g}; retained sort by "
        "delta_v_score (control-spread normalized) desc, failed sort by head marginal desc."
    )

    validity_failed: list[str] = []
    if args.validity_checks:
        control_rank = control["rank"]
        for row in ranked:
            if row["builtin_validity_probe"] and row["rank"] <= control_rank:
                validity_failed.append(row["label"])
        if validity_failed:
            print(
                "\nVALIDITY CHECK FAILED: built-in bad-config probe(s) "
                f"{', '.join(validity_failed)} did not rank below the unshaped control — "
                "the composite cannot be trusted on this corpus/model budget. "
                "RANKING WITHHELD.",
                file=sys.stderr,
            )

    report = {
        "records": str(args.records),
        "train_games": len(train_records),
        "heldout_games": len(heldout_records),
        "corpus_states": len(corpus),
        "seed": args.seed,
        "epochs": args.epochs,
        "model": {
            "embedding_dim": args.embedding_dim,
            "layers": args.layers,
            "attention_heads": args.attention_heads,
            "feedforward_dim": args.feedforward_dim,
        },
        "pearson_retention": args.pearson_retention,
        "min_pearson_floor": args.min_pearson_floor,
        "min_head_marginal": args.min_head_marginal,
        "min_spread_frac": args.min_spread_frac,
        "max_ece": args.max_ece,
        "control_pearson": control_pearson,
        "control_spread": control_spread,
        "pearson_floor": pearson_floor,
        "low_confidence": low_confidence,
        "validity_checks_enabled": args.validity_checks,
        "validity_failed_probes": validity_failed,
        "ranking_withheld": bool(validity_failed),
        "composite": (
            "constraints: corrected Pearson >= max(control - retention*|control|, abs floor); "
            "head marginal (partial corr of raw pred vs terminal controlling for Phi) >= min; "
            "spread >= frac*control; ece <= max. Retained sort by delta_v_score = "
            "(max(0,-dV_self)+max(0,dV_opp))/control_spread desc; failed sort by corrected "
            "head marginal desc (corrected Pearson is the Phi free ride for failed heads). "
            "Built-in inverted + saturating probes must rank below control or "
            "the ranking is withheld."
        ),
        "ranked": ranked,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 2 if validity_failed else 0


def _fmt(value) -> str:
    return "-" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
