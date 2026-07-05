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
          0->3 (expect V up), judged against the head's value spread;
     [ii] PBRS-corrected terminal Pearson on held-out games. Under potential-based
          shaping with a zero terminal potential the OPTIMAL shaped value is
          V'(s) = V(s) - Phi(s), so raw prediction-vs-terminal Pearson penalizes every
          shaped head by construction. The eval-facing quantity is the corrected value
          prediction + Phi_candidate(s): a head that actually learned its shaped targets
          recovers V and retains terminal Pearson; a head trained on broken targets
          (e.g. inverted signs, whose returns saturate the [-1,1] clip and whose
          correction then actively anti-corrects) loses it. Raw Pearson is also
          reported for reference;
     [iii] simple calibration of the corrected prediction vs terminal (bias + 10-bin ECE).

Ranking (documented composite):
    delta_v_score = (max(0, -value_self_response) + max(0, value_opp_response)) / value_spread
  - constraint: heldout CORRECTED terminal Pearson must stay within --pearson-retention
    (default 10%) of the unshaped control's Pearson; candidates failing the constraint
    sink below every passing candidate regardless of dV.
  - passing candidates sort by delta_v_score descending (dV is the primary read);
    failing candidates sort by corrected Pearson descending (least-broken first).
The candidate list MUST include the unshaped control ("shaping": null) — it defines the
Pearson budget, and a deliberately-bad config ranking above sane ones marks the tool broken.

Usage:
    python scripts/shaping_ranker.py --records records.jsonl \
      --candidates candidates.json --showdown-root ~/pokemon-showdown \
      --out evals/shaping_ranker.json
"""
from __future__ import annotations

import argparse
import json
import math
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
    ground_truth_components_by_step_index,
    potential_from_components,
    resolve_shaping_config,
)
from pokezero.showdown import DEFAULT_REPLAY_OBSERVATION_SPEC, observation_from_player_state

SPIKES_LAYERS_PROBE = 3


def load_candidates(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("candidates file must be a non-empty JSON list.")
    candidates = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict) or "label" not in entry or "shaping" not in entry:
            raise ValueError(f"candidate #{index} must be an object with 'label' and 'shaping' keys.")
        shaping = resolve_shaping_config(entry["shaping"]) if entry["shaping"] is not None else None
        candidates.append({"label": str(entry["label"]), "shaping": shaping})
    labels = [candidate["label"] for candidate in candidates]
    if len(set(labels)) != len(labels):
        raise ValueError("candidate labels must be unique.")
    if not any(candidate["shaping"] is None for candidate in candidates):
        raise ValueError("candidates must include the unshaped control ('shaping': null).")
    return candidates


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

    # [ii]+[iii] terminal-outcome prediction on held-out games. The gate metric is the
    # PBRS-corrected prediction (pred + Phi_candidate(s)): the shaped optimum is
    # V' = V - Phi, so the corrected value is what search/eval would consume.
    raw_predictions: list[float] = []
    corrected_predictions: list[float] = []
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
            correction = (
                potential_from_components(components_by_step[step_index], shaping)
                if shaping is not None
                else 0.0
            )
            raw_predictions.append(prediction)
            corrected_predictions.append(prediction + correction)
            outcomes.append(1.0 if terminal.winner == step.player_id else -1.0)

    return {
        "value_states": min(value_states, len(corpus)),
        "value_spread": value_spread,
        "value_self_response": value_self_response,
        "value_opp_response": value_opp_response,
        "delta_v_score": (
            (max(0.0, -value_self_response) + max(0.0, value_opp_response)) / max(value_spread, 1e-6)
        ),
        "heldout_examples": len(raw_predictions),
        "terminal_pearson": _pearson(corrected_predictions, outcomes),
        "terminal_pearson_uncorrected": _pearson(raw_predictions, outcomes),
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
        help="Allowed fractional drop of heldout terminal Pearson vs the unshaped control.",
    )
    parser.add_argument("--corpus-games", type=int, default=8, help="Scripted-driver games for the dV state corpus.")
    parser.add_argument("--corpus-states", type=int, default=200, help="Max dV corpus states collected.")
    parser.add_argument("--value-states", type=int, default=120, help="Corpus states probed per candidate (5 value evals each).")
    parser.add_argument("--work-dir", type=Path, default=None, help="Where split JSONLs are written (default: alongside --out or CWD).")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    torch = require_torch()
    candidates = load_candidates(args.candidates)
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
                "shaping": shaping.to_dict() if shaping is not None else None,
                "train_seconds": train_seconds,
                "eval_seconds": eval_seconds,
                **evaluation,
            }
        )
        print(
            f"[ranker] {label}: dV_self={evaluation['value_self_response']:+.4f} "
            f"dV_opp={evaluation['value_opp_response']:+.4f} spread={evaluation['value_spread']:.4f} "
            f"pearson={evaluation['terminal_pearson']:.4f} "
            f"(raw {evaluation['terminal_pearson_uncorrected']:.4f}) "
            f"({train_seconds:.1f}s train + {eval_seconds:.1f}s eval)"
        )

    control = next(row for row in rows if row["shaping"] is None)
    control_pearson = control["terminal_pearson"] or 0.0
    pearson_floor = control_pearson * (1.0 - args.pearson_retention)
    for row in rows:
        pearson = row["terminal_pearson"] or 0.0
        row["pearson_floor"] = pearson_floor
        row["pearson_retained"] = bool(pearson >= pearson_floor)
    ranked = sorted(
        rows,
        key=lambda row: (
            not row["pearson_retained"],
            # Retained: primary read is the dV response. Failed: order by how much
            # corrected-Pearson survives (least-broken first) — dV is noise once the
            # head no longer predicts outcomes.
            -row["delta_v_score"] if row["pearson_retained"] else 0.0,
            -(row["terminal_pearson"] or -1.0),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank

    header = (
        f"{'rank':>4} {'label':24} {'dV_self':>9} {'dV_opp':>8} {'spread':>8} {'dV_score':>9} "
        f"{'pearson*':>8} {'raw_p':>8} {'retained':>8} {'bias':>8} {'ece':>7} {'sec/cand':>9}"
    )
    print()
    print(header)
    print("-" * len(header))
    for row in ranked:
        print(
            f"{row['rank']:>4} {row['label'][:24]:24} {row['value_self_response']:9.4f} "
            f"{row['value_opp_response']:8.4f} {row['value_spread']:8.4f} {row['delta_v_score']:9.4f} "
            f"{(row['terminal_pearson'] if row['terminal_pearson'] is not None else float('nan')):8.4f} "
            f"{(row['terminal_pearson_uncorrected'] if row['terminal_pearson_uncorrected'] is not None else float('nan')):8.4f} "
            f"{('yes' if row['pearson_retained'] else 'NO'):>8} "
            f"{(row['terminal_bias'] if row['terminal_bias'] is not None else float('nan')):8.4f} "
            f"{(row['terminal_ece'] if row['terminal_ece'] is not None else float('nan')):7.4f} "
            f"{row['train_seconds'] + row['eval_seconds']:9.1f}"
        )
    print(
        f"\npearson* = PBRS-corrected (prediction + Phi_candidate) vs terminal; raw_p = uncorrected."
        f"\ncomposite: pass corrected Pearson >= {pearson_floor:.4f} "
        f"(control {control_pearson:.4f} - {args.pearson_retention:.0%}); retained sort by "
        "delta_v_score desc, failed sort by corrected Pearson desc."
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
        "control_pearson": control_pearson,
        "composite": (
            "constraint: PBRS-corrected terminal Pearson within retention of control; "
            "retained sort by delta_v_score = (max(0,-dV_self)+max(0,dV_opp))/spread desc; "
            "failed sort by corrected Pearson desc"
        ),
        "ranked": ranked,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
