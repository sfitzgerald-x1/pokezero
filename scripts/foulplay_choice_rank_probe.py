"""Compare checkpoint legal-action rankings on foul-play-generated decision states.

This probe is intentionally different from the fixed self-play choice sampler:
it uses captured foul-play games as the state source, then scores multiple
PokeZero checkpoints on the same decision states. That makes it useful for
checking whether a perturbed continuation changed policy preferences on an
out-of-distribution trajectory rather than only changing win-rate noise.

Input captures must be the JSONL format emitted by ``scripts/foulplay_mirror.sh``
or the same FOULPLAY_CAPTURE_PATH hook:

    {"t":"recv","msg":">battle-...\\n|player|..."}
    {"t":"send","msg":"battle-...|/choose move surf|3"}

Example:

    POKEZERO_BELIEF_SET_SOURCE=1 python scripts/foulplay_choice_rank_probe.py \\
      --showdown-root /path/to/pokemon-showdown \\
      --capture /tmp/fpmirror/capA.jsonl=FoulPlayA \\
      --checkpoint /shared/.../original/iteration-0069/transformer-policy.pt=orig-110k \\
      --checkpoint /shared/.../perturbed/iteration-0007/transformer-policy.pt=pert-110k \\
      --pair orig-110k:pert-110k \\
      --out /tmp/foulplay-choice-ranks.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pokezero.actions import ACTION_COUNT, MOVE_ACTION_COUNT, canonical_switch_action_map
from pokezero.dex import load_showdown_dex_cached
from pokezero.neural_policy import evaluate_transformer_action_priors
from pokezero.online_client import build_agent
from pokezero.opponents import require_current_family_checkpoint_paths
from pokezero.randbat import load_gen3_randbat_source_cached
from pokezero.randbat_vocab import gen3_category_vocabulary
from pokezero.showdown import (
    normalize_for_player,
    observation_from_player_state,
    parse_showdown_replay,
)
from pokezero.teacher_capture import action_index_from_choice_string, parse_capture_transcript


SCHEMA_VERSION = "pokezero.foulplay_choice_rank_probe.v1"


def _belief_set_source_enabled() -> bool:
    return os.environ.get("POKEZERO_BELIEF_SET_SOURCE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _checkpoint_specs(raw_specs: Sequence[str]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for raw in raw_specs:
        path, _, label = raw.partition("=")
        if not path:
            raise ValueError("--checkpoint requires PATH[=LABEL]")
        specs.append((label or Path(path).stem, path))
    labels = [label for label, _ in specs]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"duplicate checkpoint labels: {', '.join(duplicates)}")
    return specs


def _capture_specs(raw_specs: Sequence[str]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for raw in raw_specs:
        path, _, username = raw.partition("=")
        if not path or not username:
            raise ValueError(f"--capture requires PATH=USERNAME, got {raw!r}")
        specs.append((path, username))
    return specs


def _pair_specs(raw_specs: Sequence[str] | None, labels: set[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for raw in raw_specs or ():
        left, sep, right = raw.partition(":")
        if not sep or not left or not right:
            raise ValueError(f"--pair requires BASELINE:CANDIDATE, got {raw!r}")
        if left not in labels or right not in labels:
            raise ValueError(f"--pair references unknown checkpoint label: {raw!r}")
        pairs.append((left, right))
    return pairs


def _choice_label(state: Any, action_index: int) -> str:
    request = state.request if isinstance(state.request, Mapping) else {}
    if action_index < MOVE_ACTION_COUNT:
        active = request.get("active")
        first = active[0] if isinstance(active, (list, tuple)) and active else None
        moves = first.get("moves") if isinstance(first, Mapping) else None
        if isinstance(moves, (list, tuple)) and action_index < len(moves):
            move = moves[action_index]
            if isinstance(move, Mapping):
                return f"move:{move.get('move') or move.get('id') or action_index + 1}"
        return f"move:{action_index + 1}"

    # Dense switch action index maps to legal switch labels in team order excluding
    # the active mon. This mirrors the action encoding while avoiding private Showdown
    # slot numbers in the report.
    switch_offset = action_index - MOVE_ACTION_COUNT
    active_index = next((i for i, mon in enumerate(state.self_team) if mon.active), None)
    if active_index is not None:
        switch_targets = canonical_switch_action_map(active_index, team_size=len(state.self_team))
        if 0 <= switch_offset < len(switch_targets):
            target = state.self_team[switch_targets[switch_offset]]
            return f"switch:{target.species}"
    return f"switch:{switch_offset + 1}"


def _ranked_actions(state: Any, probs: Sequence[float]) -> list[dict[str, Any]]:
    legal = [index for index in range(min(ACTION_COUNT, len(probs))) if state.legal_action_mask[index]]
    rows = [
        {
            "action_index": index,
            "label": _choice_label(state, index),
            "probability": round(float(probs[index]), 8),
        }
        for index in legal
    ]
    rows.sort(key=lambda row: (-float(row["probability"]), int(row["action_index"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _distribution_by_action(rows: Sequence[Mapping[str, Any]]) -> dict[int, float]:
    return {int(row["action_index"]): float(row["probability"]) for row in rows}


def _rank_by_action(rows: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    return {int(row["action_index"]): int(row["rank"]) for row in rows}


def _top_label(rows: Sequence[Mapping[str, Any]]) -> str | None:
    return str(rows[0]["label"]) if rows else None


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _round_or_none(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None else None


def _js_divergence(left: Mapping[int, float], right: Mapping[int, float]) -> float:
    keys = sorted(set(left) | set(right))
    total_left = sum(max(0.0, left.get(key, 0.0)) for key in keys) or 1.0
    total_right = sum(max(0.0, right.get(key, 0.0)) for key in keys) or 1.0
    p = {key: max(0.0, left.get(key, 0.0)) / total_left for key in keys}
    q = {key: max(0.0, right.get(key, 0.0)) / total_right for key in keys}
    m = {key: 0.5 * (p[key] + q[key]) for key in keys}

    def kl(a: Mapping[int, float], b: Mapping[int, float]) -> float:
        return sum(value * math.log(value / b[key], 2) for key, value in a.items() if value > 0.0)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _spearman_rank_correlation(left: Mapping[int, int], right: Mapping[int, int]) -> float | None:
    keys = sorted(set(left) & set(right))
    n = len(keys)
    if n < 2:
        return None
    diffs = [left[key] - right[key] for key in keys]
    return 1.0 - (6.0 * sum(diff * diff for diff in diffs)) / (n * (n * n - 1))


def _active_snapshot(state: Any) -> dict[str, Any]:
    active = state.self_active
    opponent = state.opponent_active
    return {
        "active": active.species if active is not None else None,
        "active_condition": active.condition if active is not None else None,
        "opponent_active": opponent.species if opponent is not None else None,
        "opponent_condition": opponent.condition if opponent is not None else None,
        "turn": state.turn_number,
        "weather": state.weather,
        "self_side_conditions": list(state.self_side_conditions),
        "opponent_side_conditions": list(state.opponent_side_conditions),
    }


def _build_states_from_captures(
    capture_specs: Sequence[tuple[str, str]],
    *,
    set_source: Any,
    max_states: int | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    stats = {
        "decisions": 0,
        "states": 0,
        "parse_errors": 0,
        "undecoded_choices": 0,
        "no_legal_mask": 0,
    }
    for capture_path, username in capture_specs:
        for game in parse_capture_transcript(capture_path):
            for decision_index, decision in enumerate(game.decisions):
                stats["decisions"] += 1
                try:
                    replay = parse_showdown_replay(decision.protocol_lines, battle_id=decision.room)
                    state = normalize_for_player(
                        replay,
                        player_id="capture",
                        player_name=username,
                        format_id="gen3randombattle",
                        set_source=set_source,
                        include_turn_merged=True,
                    )
                except ValueError:
                    stats["parse_errors"] += 1
                    continue
                if not any(state.legal_action_mask):
                    stats["no_legal_mask"] += 1
                    continue
                teacher_action = action_index_from_choice_string(state, decision.choice)
                if teacher_action is None:
                    stats["undecoded_choices"] += 1
                records.append(
                    {
                        "source_capture": str(capture_path),
                        "username": username,
                        "room": decision.room,
                        "decision_index": decision_index,
                        "state_id": (
                            f"{Path(capture_path).name}:{decision.room}:"
                            f"{username}:decision={decision_index:04d}"
                        ),
                        "teacher_choice": decision.choice,
                        "teacher_action_index": teacher_action,
                        "teacher_action_label": (
                            _choice_label(state, teacher_action) if teacher_action is not None else None
                        ),
                        "state": state,
                        "context": _active_snapshot(state),
                    }
                )
                stats["states"] += 1
                if max_states is not None and len(records) >= max_states:
                    return records, stats
    return records, stats


def _score_checkpoints(
    states: Sequence[dict[str, Any]],
    checkpoint_specs: Sequence[tuple[str, str]],
    *,
    showdown_root: str,
    device: str | None,
    temperature: float,
) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for label, path in checkpoint_specs:
        print(f"[rank-probe] scoring {label}…", file=sys.stderr)
        agent = build_agent(path, showdown_root, our_name="foulplay-rank-probe", deterministic=True)
        result = agent.policy.result
        model_config = result.model_config
        if int(model_config.window_size) != 1:
            raise ValueError(
                f"{label} has window_size={model_config.window_size}; "
                "foul-play choice-rank probe currently requires window_size=1"
            )
        metadata.append(
            {
                "label": label,
                "path": path,
                "policy_id": model_config.policy_id,
                "observation_schema_version": model_config.observation_schema_version,
                "window_size": model_config.window_size,
                "belief_set_source_hash": result.belief_set_source_hash,
            }
        )
        for record in states:
            state = record["state"]
            observation = observation_from_player_state(
                state,
                category_vocab=agent.vocab,
                spec=agent.spec,
                dex=agent.dex,
                **({"feature_masks": agent.feature_masks} if agent.feature_masks is not None else {}),
            )
            probs = evaluate_transformer_action_priors(
                model=agent.policy.model,
                result=agent.policy.result,
                observations=[observation],
                temperature=temperature,
                device=device,
            )
            ranked = _ranked_actions(state, probs)
            teacher_action = record.get("teacher_action_index")
            teacher_rank = None
            teacher_probability = None
            if teacher_action is not None:
                for row in ranked:
                    if int(row["action_index"]) == int(teacher_action):
                        teacher_rank = int(row["rank"])
                        teacher_probability = float(row["probability"])
                        break
            record.setdefault("checkpoints", {})[label] = {
                "top_action": _top_label(ranked),
                "teacher_rank": teacher_rank,
                "teacher_probability": (
                    round(teacher_probability, 8) if teacher_probability is not None else None
                ),
                "ranked_actions": ranked,
            }
    return metadata


def _provenance_warnings(
    checkpoints: Sequence[Mapping[str, Any]],
    *,
    set_source: Any,
) -> list[str]:
    warnings: list[str] = []
    schemas = sorted({str(item.get("observation_schema_version")) for item in checkpoints})
    if len(schemas) > 1:
        warnings.append(f"checkpoint observation schemas differ: {', '.join(schemas)}")
    belief_hashes = sorted({item.get("belief_set_source_hash") for item in checkpoints}, key=lambda x: str(x))
    if len(belief_hashes) > 1:
        warnings.append("checkpoint belief_set_source_hash values differ")
    active_hash = _set_source_hash(set_source)
    for item in checkpoints:
        label = str(item.get("label"))
        checkpoint_hash = item.get("belief_set_source_hash")
        if active_hash and checkpoint_hash != active_hash:
            warnings.append(
                f"{label} was trained with belief_set_source_hash={checkpoint_hash!r}, "
                f"but active source hash is {active_hash!r}"
            )
        if active_hash is None and checkpoint_hash:
            warnings.append(
                f"{label} was trained with belief_set_source_hash={checkpoint_hash!r}, "
                "but POKEZERO_BELIEF_SET_SOURCE is disabled for this probe"
            )
    return warnings


def _set_source_hash(set_source: Any) -> str | None:
    if set_source is None:
        return None
    direct = getattr(set_source, "source_hash", None)
    if direct:
        return str(direct)
    metadata = getattr(set_source, "metadata", None)
    nested = getattr(metadata, "source_hash", None)
    return str(nested) if nested else None


def _pairwise_summary(states: Sequence[Mapping[str, Any]], pairs: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for left_label, right_label in pairs:
        top_changes = 0
        rank_abs_deltas: list[float] = []
        top_prob_deltas: list[float] = []
        teacher_rank_deltas: list[float] = []
        js_values: list[float] = []
        spearman_values: list[float] = []
        changed_examples: list[dict[str, Any]] = []
        for record in states:
            checkpoints = record["checkpoints"]
            left = checkpoints[left_label]
            right = checkpoints[right_label]
            left_top = left["top_action"]
            right_top = right["top_action"]
            if left_top != right_top:
                top_changes += 1
                if len(changed_examples) < 12:
                    changed_examples.append(
                        {
                            "state_id": record["state_id"],
                            "turn": record["context"]["turn"],
                            "active": record["context"]["active"],
                            "opponent_active": record["context"]["opponent_active"],
                            "baseline_top_action": left_top,
                            "candidate_top_action": right_top,
                            "teacher_action": record.get("teacher_action_label"),
                        }
                    )
            left_dist = _distribution_by_action(left["ranked_actions"])
            right_dist = _distribution_by_action(right["ranked_actions"])
            left_ranks = _rank_by_action(left["ranked_actions"])
            right_ranks = _rank_by_action(right["ranked_actions"])
            for action in sorted(set(left_ranks) & set(right_ranks)):
                rank_abs_deltas.append(abs(left_ranks[action] - right_ranks[action]))
            if left["ranked_actions"] and right["ranked_actions"]:
                top_idx = int(right["ranked_actions"][0]["action_index"])
                top_prob_deltas.append(right_dist.get(top_idx, 0.0) - left_dist.get(top_idx, 0.0))
            if left.get("teacher_rank") is not None and right.get("teacher_rank") is not None:
                teacher_rank_deltas.append(float(left["teacher_rank"]) - float(right["teacher_rank"]))
            js_values.append(_js_divergence(left_dist, right_dist))
            spearman = _spearman_rank_correlation(left_ranks, right_ranks)
            if spearman is not None:
                spearman_values.append(spearman)

        state_count = len(states)
        summaries.append(
            {
                "baseline": left_label,
                "candidate": right_label,
                "state_count": state_count,
                "top_action_disagreement_rate": round(top_changes / state_count, 6) if state_count else None,
                "top_action_disagreements": top_changes,
                "mean_abs_legal_rank_delta": _round_or_none(_mean(rank_abs_deltas)),
                "mean_candidate_top_probability_delta": _round_or_none(_mean(top_prob_deltas)),
                "mean_teacher_rank_improvement": _round_or_none(_mean(teacher_rank_deltas)),
                "mean_js_divergence_bits": _round_or_none(_mean(js_values)),
                "mean_spearman_rank_correlation": _round_or_none(_mean(spearman_values)),
                "changed_top_action_examples": changed_examples,
            }
        )
    return summaries


def _strip_state_objects(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for record in records:
        output.append(
            {
                key: value
                for key, value in record.items()
                if key != "state"
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="append", required=True, metavar="PATH=USERNAME")
    parser.add_argument("--checkpoint", action="append", required=True, metavar="PATH[=LABEL]")
    parser.add_argument("--pair", action="append", metavar="BASELINE:CANDIDATE")
    parser.add_argument("--showdown-root", required=True)
    parser.add_argument("--max-states", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--allow-legacy-checkpoints",
        action="store_true",
        help=(
            "allow no-belief/pre-v2 checkpoints for explicit historical reproduction; "
            "do not use for current diversity evals"
        ),
    )
    args = parser.parse_args()

    if args.temperature <= 0.0:
        raise ValueError("--temperature must be positive")
    if args.max_states is not None and args.max_states <= 0:
        raise ValueError("--max-states must be positive")

    checkpoint_specs = _checkpoint_specs(args.checkpoint)
    labels = {label for label, _ in checkpoint_specs}
    pairs = _pair_specs(args.pair, labels)
    capture_specs = _capture_specs(args.capture)

    if not args.allow_legacy_checkpoints:
        require_current_family_checkpoint_paths(
            (path for _, path in checkpoint_specs),
            context="foul-play choice-rank probe",
        )

    print("[rank-probe] loading source metadata…", file=sys.stderr)
    vocab = gen3_category_vocabulary(args.showdown_root)
    # Prime the same caches build_agent / observation scoring will use; this also
    # fails early when the Showdown checkout is not built.
    load_showdown_dex_cached(args.showdown_root)
    set_source = load_gen3_randbat_source_cached(args.showdown_root) if _belief_set_source_enabled() else None
    if set_source is not None:
        print("[rank-probe] belief set source ENABLED", file=sys.stderr)
    # Keep vocab referenced so linters do not "simplify" away the early cache/gate.
    _ = vocab

    states, stats = _build_states_from_captures(capture_specs, set_source=set_source, max_states=args.max_states)
    if not states:
        raise SystemExit("no scoreable decision states found in capture(s)")
    print(f"[rank-probe] scoreable states: {len(states)}", file=sys.stderr)
    checkpoint_metadata = _score_checkpoints(
        states,
        checkpoint_specs,
        showdown_root=args.showdown_root,
        device=args.device,
        temperature=args.temperature,
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "captures": [
            {"path": path, "username": username}
            for path, username in capture_specs
        ],
        "checkpoints": checkpoint_metadata,
        "pairs": [
            {"baseline": left, "candidate": right}
            for left, right in pairs
        ],
        "temperature": args.temperature,
        "belief_set_source": set_source is not None,
        "belief_set_source_hash": _set_source_hash(set_source),
        "warnings": _provenance_warnings(checkpoint_metadata, set_source=set_source),
        "stats": stats,
        "pairwise": _pairwise_summary(states, pairs),
        "states": _strip_state_objects(states),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[rank-probe] wrote {args.out}", file=sys.stderr)

    for item in payload["pairwise"]:
        print(
            f"{item['baseline']} -> {item['candidate']}: "
            f"top-change={item['top_action_disagreement_rate']} "
            f"mean-js={item['mean_js_divergence_bits']} "
            f"teacher-rank-improvement={item['mean_teacher_rank_improvement']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
