"""Saved actual action siblings for A1; never collect, replay, or fit here.

Capture callers must branch the SAME restored root with a fixed opponent reply,
belief world and chance address. A hash binds evidence, not its scientific truth:
the upstream collector and its admission still need independent review.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = "pokezero.actual_value_siblings.v1"
KINDS = ("linear", "attention_mlp", "attention_hl_gauss")
OBSERVATION_FIELDS = ("categorical_ids", "numeric_features", "token_type_ids",
                      "attention_mask", "legal_action_mask", "schema_version")


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("lowercase SHA256 required")


def _text(value):
    if not isinstance(value, str) or not value:
        raise ValueError("nonempty identity required")


def _action(value):
    if type(value) is not int or not 0 <= value < 9:
        raise ValueError("integer action in [0,9) required")


def observation_document(observation):
    """Copy the actual dense v4 observation, preserving the root seat's view."""
    result = {}
    for name in OBSERVATION_FIELDS:
        value = observation[name] if isinstance(observation, dict) else getattr(observation, name)
        if hasattr(value, "tolist"):
            value = value.tolist()
        result[name] = json.loads(json.dumps(value, allow_nan=False))
    _validate_observation(result)
    return result


def _validate_observation(observation):
    if set(observation) != set(OBSERVATION_FIELDS) or observation["schema_version"] != "pokezero.observation.v4":
        raise ValueError("actual dense v4 observation required")
    for name, shape, kind in (("categorical_ids", (23, 41), "iu"),
                              ("numeric_features", (23, 132), "iuf"),
                              ("token_type_ids", (23,), "iu"),
                              ("attention_mask", (23,), "b"),
                              ("legal_action_mask", (9,), "b")):
        array = np.asarray(observation[name])
        if array.shape != shape or array.dtype.kind not in kind or not np.isfinite(array).all():
            raise ValueError("observation shape/dtype/finite mismatch: " + name)
        if name in ("categorical_ids", "token_type_ids") and (array < 0).any():
            raise ValueError("negative observation identity")
    if not any(observation["attention_mask"]):
        raise ValueError("observation requires an attended token")
    with np.errstate(over="ignore"):
        if not np.isfinite(np.asarray(observation["numeric_features"], dtype=np.float32)).all():
            raise ValueError("finite FP32 observation conversion required")


def capture_group(*, root, branches):
    """Create one group from actual post-joint-step captures, with no simulator.

    Each branch repeats parent/context identities so accidental grouping of
    adjacent turns, different replies or worlds refuses. Nonterminal observations
    must be encoded for root['seat'], even when the other seat acts next.
    Immediate terminal branches have no model observation and are kept/excluded.
    This API does not authorize the branching work that supplies these records.
    """
    group = {"root": json.loads(canonical_bytes(root)), "branches": []}
    for branch in branches:
        copied = dict(branch)
        if copied.get("observation") is not None:
            copied["observation"] = observation_document(copied["observation"])
            copied["observation_sha256"] = sha256(canonical_bytes(copied["observation"]))
        group["branches"].append(json.loads(canonical_bytes(copied)))
    validate_group(group)
    return group


def validate_group(group):
    root = group["root"]
    for name in ("battle_id", "world_id", "chance_id", "source_commit", "collector_id"):
        _text(root[name])
    for name in ("root_sha256", "source_receipt_sha256"):
        _digest(root[name])
    if root["seat"] not in ("p1", "p2") or type(root["turn"]) is not int or root["turn"] < 0:
        raise ValueError("root seat/turn required")
    if root.get("split") != "test":
        raise ValueError("predeclared test roots required; no train/validation reuse")
    if root.get("selection") != "all_legal_actions_before_values":
        raise ValueError("pre-value complete legal-action selection required")
    if (root.get("observation_view") != "public_plus_root_private"
            or root.get("belief_update") != "root_belief_plus_branch_reveals"):
        raise ValueError("root public/private view and branch-reveal belief contract required")
    legal = root["legal_actions"]
    if not isinstance(legal, list) or len(legal) < 2 or len(set(legal)) != len(legal):
        raise ValueError("at least two unique legal root actions required")
    for action in legal:
        _action(action)
    _action(root["opponent_action"])
    seen = set()
    for branch in group["branches"]:
        action = branch["action"]
        _action(action)
        if action in seen or action not in legal:
            raise ValueError("duplicate or illegal sibling action")
        seen.add(action)
        for name in ("root_sha256", "world_id", "chance_id", "opponent_action"):
            if branch[name] != root[name]:
                raise ValueError("sibling fixed context mismatch: " + name)
        if branch.get("observation_seat") != root["seat"]:
            raise ValueError("sibling observation must use root seat")
        _digest(branch["child_state_sha256"])
        if type(branch.get("terminal")) is not bool:
            raise ValueError("explicit immediate terminal status required")
        if branch["terminal"]:
            if (branch.get("observation") is not None or branch.get("terminal_capped") is not False
                    or type(branch.get("terminal_outcome")) not in (int, float)
                    or branch["terminal_outcome"] not in (-1, 0, 1)):
                raise ValueError("exact uncapped terminal outcome, no successor observation required")
            _digest(branch["terminal_receipt_sha256"])
        else:
            if branch.get("terminal_outcome") is not None:
                raise ValueError("nonterminal successor cannot carry invented terminal value")
            _validate_observation(branch["observation"])
            if sha256(canonical_bytes(branch["observation"])) != branch["observation_sha256"]:
                raise ValueError("sibling observation digest mismatch")
    if seen != set(legal):
        raise ValueError("complete root legal-action coverage required; missing siblings")
    return (root["battle_id"], root["turn"], root["seat"], root["root_sha256"],
            root["world_id"], root["chance_id"], root["opponent_action"])


def write_capture(path, *, groups, checkpoint_contract, admission_receipt_sha256):
    """Create-only saved capture; full observations belong on remote storage."""
    _digest(admission_receipt_sha256)
    seen = set()
    for group in groups:
        key = validate_group(group)
        if key in seen:
            raise ValueError("duplicate sibling root/context")
        seen.add(key)
    if not seen:
        raise ValueError("nonempty actual sibling capture required")
    document = dict(schema=SCHEMA, groups=groups, checkpoint_contract=checkpoint_contract,
                    admission_receipt_sha256=admission_receipt_sha256,
                    value_scale="root-seat return [-1,1]", model_fits=0)
    blob = canonical_bytes(document)
    with Path(path).open("xb") as stream:
        stream.write(blob)
    return sha256(blob)


def summarize_groups(groups, predict, thresholds):
    """Both heads choose their own top two on the same complete legal roster.

    Error thresholds are external held-out outcome RMSE, not sibling error or
    uncertainty. Report the comparison descriptively; never call it resolved.
    Groups containing immediate terminal actions are retained in coverage and
    excluded from the head-only statistic, rather than forwarding a root again.
    """
    if set(thresholds) != {"champion", "refit"} or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0
            for x in thresholds.values()):
        raise ValueError("finite nonnegative external RMSE for both heads required")
    rows = []; seen = set(); terminal_groups = terminal_actions = 0
    for group in groups:
        key = validate_group(group)
        if key in seen:
            raise ValueError("duplicate sibling root/context")
        seen.add(key)
    for group in groups:
        key = validate_group(group)
        branches = sorted(group["branches"], key=lambda b: b["action"])
        terminals = [b for b in branches if b["terminal"]]
        if terminals:
            terminal_groups += 1; terminal_actions += len(terminals)
            continue
        prediction = predict([b["observation"] for b in branches])
        row = {"root_identity": list(key), "actions": [b["action"] for b in branches], "heads": {}}
        if set(prediction) != set(thresholds):
            raise ValueError("both head predictions required")
        for name in thresholds:
            values = np.asarray(prediction[name], dtype=np.float64)
            if values.shape != (len(branches),) or not np.isfinite(values).all():
                raise ValueError("finite per-sibling predictions required")
            order = sorted(range(len(values)), key=lambda i: (-values[i], branches[i]["action"]))
            a, b = order[:2]; gap = float(values[a] - values[b])
            row["heads"][name] = dict(values_return_scale=values.tolist(),
                top_two_actions=[branches[a]["action"], branches[b]["action"]], gap_return_scale=gap,
                gap_exceeds_external_outcome_rmse=gap > thresholds[name])
        rows.append(row)
    summaries = {}
    for name, threshold in thresholds.items():
        gaps = np.array([r["heads"][name]["gap_return_scale"] for r in rows])
        count = sum(r["heads"][name]["gap_exceeds_external_outcome_rmse"] for r in rows)
        summaries[name] = dict(external_outcome_rmse=threshold, scored_root_contexts=len(rows),
            gap_exceeds_external_rmse_count=count,
            share_gap_exceeds_external_outcome_rmse=count / len(rows) if rows else None,
            median_gap_return_scale=float(np.median(gaps)) if len(rows) else None)
    return dict(status="SAVED_SIBLING_GAPS_DESCRIPTIVE_ONLY" if rows else "NO_NONTERMINAL_SIBLING_GROUPS",
        captured_root_contexts=len(seen), terminal_groups_excluded=terminal_groups,
        immediate_terminal_actions=terminal_actions, heads=summaries, rows=rows,
        interpretation="External outcome RMSE includes outcome randomness; gap > RMSE does not identify action ranking correctness or uncertainty.",
        actual_sibling_error="UNMEASURED_REQUIRES_MATCHED_TERMINAL_CONTINUATIONS",
        aggregation="Equal weight per fixed root/world/chance/reply context; not world-averaged action Q.")


def analyze(capture_path, checkpoint_path, summary_path, head_path, *, capture_sha256,
            checkpoint_sha256, summary_sha256, head_sha256):
    """CPU-only actual champion trunk plus validation-selected refit, no fit."""
    from .neural_policy import (EntityTokenTransformerPolicy, parse_transformer_model_config,
                               observation_windows_to_torch, require_torch)
    from .observation import PokeZeroObservationV0
    from .value_precision_probe import checkpoint_contract, frozen_token_features, make_value_head
    torch = require_torch()
    blobs = []
    for path, expected in ((capture_path, capture_sha256), (checkpoint_path, checkpoint_sha256),
                           (summary_path, summary_sha256), (head_path, head_sha256)):
        _digest(expected); blob = Path(path).read_bytes()
        if sha256(blob) != expected:
            raise ValueError("saved artifact SHA binding mismatch")
        blobs.append(blob)
    capture, summary = json.loads(blobs[0]), json.loads(blobs[2])
    if (capture.get("schema") != SCHEMA or capture.get("value_scale") != "root-seat return [-1,1]"
            or not capture.get("groups") or capture.get("model_fits") != 0):
        raise ValueError("nonempty actual sibling capture required")
    _digest(capture["admission_receipt_sha256"])
    contract = checkpoint_contract(checkpoint_path)
    if contract["checkpoint_sha256"] != checkpoint_sha256 or capture["checkpoint_contract"] != contract:
        raise ValueError("capture checkpoint/observation contract mismatch")
    if (summary.get("schema") != "pokezero.value_probe_head_refits.v1"
            or summary.get("status") != "THREE_OUTCOME_HEAD_REFITS_COMPLETE_ZEROS_UNRESOLVED"
            or summary.get("checkpoint_contract") != contract or contract["value_calibration_transform"] is not None):
        raise ValueError("completed exact uncalibrated outcome head refit required")
    for kind in KINDS:
        metric = summary["heads"][kind]["validation_mse"]
        if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not math.isfinite(metric) or metric < 0:
            raise ValueError("finite validation selection required")
    selected = min(KINDS, key=lambda k: (summary["heads"][k]["validation_mse"], KINDS.index(k)))
    chosen = summary["heads"][selected]
    if chosen["artifact_sha256"] != head_sha256:
        raise ValueError("refit artifact is not validation-selected head")
    payload = torch.load(io.BytesIO(blobs[1]), map_location="cpu", weights_only=True)
    model = EntityTokenTransformerPolicy(parse_transformer_model_config(payload))
    model.load_state_dict(payload["state_dict"], strict=True); model.float().eval()
    fitted = torch.load(io.BytesIO(blobs[3]), map_location="cpu", weights_only=True)
    if (fitted.get("schema") != "pokezero.value_probe_head.v1" or fitted.get("kind") != selected
            or fitted.get("width") != contract["width"] or fitted.get("selected_epoch") != chosen["selected_epoch"]
            or fitted.get("features_manifest_sha256") != summary["features_manifest_sha256"]):
        raise ValueError("selected refit parameter contract mismatch")
    head = make_value_head(selected, contract["width"]).float().eval()
    head.load_state_dict(fitted["state_dict"], strict=True)
    for parameter in head.parameters():
        parameter.requires_grad_(False)

    def predict(observations):
        histories = [(PokeZeroObservationV0(**o),) for o in observations]
        tensors = observation_windows_to_torch(histories, window_size=1, device="cpu")
        for name, size in (("categorical_ids", model.config.categorical_vocab_size),
                           ("token_type_ids", model.config.token_type_vocab_size)):
            if (tensors[name] >= size).any():
                raise ValueError("observation exceeds checkpoint vocabulary")
        with torch.no_grad():
            tokens, output = frozen_token_features(model, tensors)
            values = head(tokens, tensors["attention_mask"][:, 0])[0]
        return dict(champion=output.value.cpu().numpy(), refit=values.cpu().numpy())

    thresholds = dict(champion=math.sqrt(summary["baseline_test"]["mse"]),
                      refit=math.sqrt(chosen["test"]["mse"]))
    result = summarize_groups(capture["groups"], predict, thresholds)
    result.update(schema="pokezero.actual_value_sibling_analysis.v1", selected_head_validation_only=selected,
        selected_epoch=chosen["selected_epoch"], capture_sha256=capture_sha256, checkpoint_sha256=checkpoint_sha256,
        head_summary_sha256=summary_sha256, head_sha256=head_sha256,
        threshold_source="Previously frozen A1 known-outcome TEST MSE; no sibling data selected head or threshold.",
        model_fits=0, new_games=0, replay_calls=0, device="cpu")
    return result
