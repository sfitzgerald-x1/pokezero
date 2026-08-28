#!/usr/bin/env python3
"""Write a receipt-bound, non-executing plan for the registered Stage-1 ladder.

This is intentionally a launcher *planner*, not an executor.  Stage 1 remains
blocked until its P0 receipts (including the patience controller smoke) exist.
The script makes a future launch auditable now: it accepts only the exact
registered recipe, rechecks the replay-bound corpus receipt and its cache-tree
identities, enumerates every registered arm/LR/seed, and writes one immutable
plan.  It never invokes a training subprocess.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


RECIPE_SCHEMA = "pokezero.expert-iteration.stage1-value-head-recipe.v1"
RECIPE_ID = "expert-iteration-stage1-value-head-ladder-20260827"
RECIPE_SHA256 = "63c506d72a9d348904152072d82f02e6852358d99d443848dc075d0b3a106033"
CORPUS_SCHEMA = "pokezero.expert-iteration.oracle-label-corpus.v1"
CORPUS_RECEIPT_SCHEMA = "pokezero.expert-iteration.oracle-label-corpus-receipt.v1"
MODEL_INPUT_HASH_SCHEMA = "pokezero.training-cache-model-input.v1"
PLAN_SCHEMA = "pokezero.expert-iteration.stage1-launch-plan.v1"


class Stage1LaunchRefusal(ValueError):
    """Raised when an input cannot license even a non-executing Stage-1 plan."""


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage1LaunchRefusal(f"{label} must be an object")
    return value


def _sequence(value: object, *, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise Stage1LaunchRefusal(f"{label} must be an array")
    return value


def _regular_bytes(path: Path, *, label: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise Stage1LaunchRefusal(f"{label} must be a regular file: {path}")
    return path.read_bytes()


def _json(path: Path, *, label: str) -> tuple[Mapping[str, Any], bytes]:
    raw = _regular_bytes(path, label=label)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise Stage1LaunchRefusal(f"{label} is not JSON") from exc
    return _mapping(value, label=label), raw


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def training_cache_tree_sha256(path: Path) -> str:
    if not path.is_dir() or path.is_symlink():
        raise Stage1LaunchRefusal(f"training cache must be a non-symlink directory: {path}")
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*"), key=lambda candidate: candidate.relative_to(path).as_posix()):
        relative = item.relative_to(path).as_posix()
        if item.is_symlink():
            raise Stage1LaunchRefusal(f"training cache contains symlink: {relative}")
        if item.is_dir():
            continue
        if not item.is_file():
            raise Stage1LaunchRefusal(f"training cache contains non-regular entry: {relative}")
        raw = item.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise Stage1LaunchRefusal(f"{label} must be a lowercase SHA-256")
    return value


def _validate_recipe(recipe: Mapping[str, Any]) -> None:
    if recipe.get("schema") != RECIPE_SCHEMA or recipe.get("recipe_id") != RECIPE_ID:
        raise Stage1LaunchRefusal("recipe is not the registered Stage-1 ladder")
    if recipe.get("status") != "REGISTERED-NOT-RUN":
        raise Stage1LaunchRefusal("recipe is not registered-not-run")
    source = _mapping(recipe.get("source"), label="recipe.source")
    _mapping(source.get("checkpoint"), label="recipe.source.checkpoint")
    corpus = _mapping(source.get("label_corpus"), label="recipe.source.label_corpus")
    if (
        corpus.get("required_schema") != CORPUS_SCHEMA
        or corpus.get("required_receipt_schema") != CORPUS_RECEIPT_SCHEMA
        or corpus.get("model_input_hash_schema") != MODEL_INPUT_HASH_SCHEMA
        or corpus.get("target") != "policy_continuation_value"
        or corpus.get("reader_verification_required") is not True
    ):
        raise Stage1LaunchRefusal("recipe does not require the registered replay-bound cache receipt")
    training = _mapping(recipe.get("training"), label="recipe.training")
    if training.get("learning_rate_schedule") != "constant":
        raise Stage1LaunchRefusal("recipe changes the registered constant learning-rate schedule")
    early_stopping = _mapping(training.get("early_stopping"), label="recipe.training.early_stopping")
    if early_stopping.get("split") != "heldout" or early_stopping.get("metric") != "mae":
        raise Stage1LaunchRefusal("recipe does not select by held-out MAE")
    arms = _sequence(recipe.get("arms"), label="recipe.arms")
    if len(arms) != 6:
        raise Stage1LaunchRefusal("recipe does not have exactly six Stage-1 arms")


def _validated_corpus_inputs(
    *, recipe: Mapping[str, Any], receipt_path: Path
) -> tuple[Mapping[str, Any], str, Path, Path]:
    receipt, receipt_raw = _json(receipt_path, label="corpus receipt")
    source = _mapping(recipe["source"], label="recipe.source")
    requirement = _mapping(source["label_corpus"], label="recipe.source.label_corpus")
    if (
        receipt.get("schema") != requirement["required_receipt_schema"]
        or receipt.get("status") != "VALIDATED_REPLAY_BOUND_ORACLE_CORPUS"
    ):
        raise Stage1LaunchRefusal("corpus receipt is not a validated replay-bound corpus")
    receipt_bank = _mapping(receipt.get("bank"), label="corpus receipt.bank")
    if receipt_bank.get("sha256") != requirement.get("source_bank_sha256"):
        raise Stage1LaunchRefusal("corpus receipt binds a different source bank")
    corpus_identity = _mapping(receipt.get("corpus"), label="corpus receipt.corpus")
    corpus_path = Path(str(corpus_identity.get("path", "")))
    corpus, corpus_raw = _json(corpus_path, label="materialized corpus")
    if _sha256(corpus_raw) != _digest(corpus_identity.get("sha256"), label="corpus receipt.corpus.sha256"):
        raise Stage1LaunchRefusal("materialized corpus differs from its validated receipt")
    if (
        corpus.get("schema") != requirement["required_schema"]
        or corpus.get("successor_observation_hash_schema")
        != requirement["model_input_hash_schema"]
    ):
        raise Stage1LaunchRefusal("materialized corpus lacks the exact cache-input hash contract")
    corpus_bank = _mapping(corpus.get("bank"), label="materialized corpus.bank")
    if corpus_bank.get("sha256") != requirement["source_bank_sha256"]:
        raise Stage1LaunchRefusal("materialized corpus binds a different source bank")
    caches = _mapping(receipt.get("training_caches"), label="corpus receipt.training_caches")
    paths: dict[str, Path] = {}
    for split in ("train", "heldout"):
        cache = _mapping(caches.get(split), label=f"corpus receipt.training_caches.{split}")
        path = Path(str(cache.get("path", "")))
        if training_cache_tree_sha256(path) != _digest(cache.get("tree_sha256"), label=f"{split} cache tree_sha256"):
            raise Stage1LaunchRefusal(f"{split} cache differs from its validated receipt")
        paths[split] = path
    return receipt, _sha256(receipt_raw), paths["train"], paths["heldout"]


def _conversion_command(*, source_checkpoint: str, run_root: Path, kind: str, seed: int) -> tuple[str, list[str]]:
    destination = run_root / "converted-heads" / f"{kind}-seed{seed}.pt"
    argv = ["python3", "scripts/convert_value_head.py", "--checkpoint", source_checkpoint, "--output", str(destination), "--head-init-seed", str(seed)]
    if kind == "mlp2":
        argv.extend(["--value-head-hidden", "256"])
    elif kind == "mlp3":
        argv.extend(["--value-head-hidden-layers", "256,256"])
    else:  # pragma: no cover - called only from the fixed recipe's arm names
        raise Stage1LaunchRefusal(f"unsupported converted head kind: {kind}")
    return str(destination), argv


def build_plan(
    *, recipe: Mapping[str, Any], corpus_receipt_path: Path, run_root: Path
) -> dict[str, object]:
    """Build all registered commands without executing any of them."""

    _validate_recipe(recipe)
    if run_root.exists() or run_root.is_symlink():
        raise Stage1LaunchRefusal(f"refusing a plan whose Stage-1 run root already exists: {run_root}")
    receipt, receipt_sha256, train_cache, heldout_cache = _validated_corpus_inputs(
        recipe=recipe, receipt_path=corpus_receipt_path
    )
    source = _mapping(recipe["source"], label="recipe.source")
    checkpoint = _mapping(source["checkpoint"], label="recipe.source.checkpoint")
    training = _mapping(recipe["training"], label="recipe.training")
    optimizer = _mapping(_mapping(recipe["engine_contract"], label="recipe.engine_contract")["optimizer"], label="recipe.engine_contract.optimizer")
    learning_rates = _sequence(training["learning_rate_grid"], label="recipe.training.learning_rate_grid")
    conversions: list[dict[str, object]] = []
    converted_paths: dict[str, str] = {}
    for kind, seed in (("mlp2", 43010001), ("mlp3", 43010002)):
        path, argv = _conversion_command(
            source_checkpoint=str(checkpoint["path"]), run_root=run_root, kind=kind, seed=seed
        )
        converted_paths[kind] = path
        conversions.append({"kind": kind, "initialization_seed": seed, "argv": argv})
    runs: list[dict[str, object]] = []
    for raw_arm in _sequence(recipe["arms"], label="recipe.arms"):
        arm = _mapping(raw_arm, label="recipe arm")
        arm_id = str(arm["id"])
        head = _mapping(arm["head"], label=f"recipe arm {arm_id}.head")
        kind = str(head["kind"])
        initial_checkpoint = str(checkpoint["path"]) if kind == "linear" else converted_paths[kind]
        seeds = _sequence(arm["training_seeds"], label=f"recipe arm {arm_id}.training_seeds")
        if len(seeds) != len(learning_rates):
            raise Stage1LaunchRefusal(f"recipe arm {arm_id} does not have a seed for every learning rate")
        for learning_rate, seed in zip(learning_rates, seeds, strict=True):
            run_id = f"{arm_id}-lr{learning_rate:g}-seed{seed}"
            output_dir = run_root / "runs" / run_id
            argv = [
                "python3", "-m", "pokezero.neural_cli", "train",
                "--data", str(train_cache),
                "--out", str(output_dir / "transformer-policy.pt"),
                "--summary-out", str(output_dir / "train-summary.json"),
                "--initial-checkpoint", initial_checkpoint,
                "--keep-cache-after-read",
                "--objective", "value-only",
                "--batch-size", str(training["batch_size"]),
                "--epochs", str(training["max_epochs"]),
                "--learning-rate", str(learning_rate),
                "--learning-rate-schedule", str(training["learning_rate_schedule"]),
                "--weight-decay", str(optimizer["weight_decay"]),
                "--value-loss-weight", str(training["value_loss_weight"]),
                "--value-ranking-loss-weight", str(training["value_ranking_loss_weight"]),
                "--max-grad-norm", str(training["max_grad_norm"]),
                "--amp", str(training["amp"]),
                "--training-seed", str(seed),
                "--train-batch-replay",
                "--value-selection-data", str(heldout_cache),
                "--value-selection-metric", "mae",
                "--value-selection-out", str(output_dir / "heldout-mae-by-epoch.json"),
            ]
            if arm["freeze_non_value_parameters"] is True:
                argv.append("--freeze-non-value-parameters")
            elif arm["freeze_policy_heads"] is True:
                argv.append("--freeze-policy-heads")
            else:
                raise Stage1LaunchRefusal(f"recipe arm {arm_id} has no approved value-only freeze mode")
            runs.append({"id": run_id, "arm": arm_id, "learning_rate": learning_rate, "training_seed": seed, "argv": argv})
    return {
        "schema": PLAN_SCHEMA,
        "status": "BLOCKED_PENDING_REGISTERED_P0_GATES",
        "execution": "NOT_EXECUTED_BY_THIS_TOOL",
        "recipe": {"recipe_id": recipe["recipe_id"], "checkpoint": checkpoint, "launch_gate": recipe["launch_gate"]},
        "corpus_receipt": {"path": str(corpus_receipt_path), "sha256": receipt_sha256, "status": receipt["status"]},
        "run_root": str(run_root),
        "head_conversions": conversions,
        "runs": runs,
        "registered_run_count": len(runs),
        "blocked_on": list(_sequence(_mapping(recipe["launch_gate"], label="recipe.launch_gate")["requires"], label="recipe.launch_gate.requires")),
        "early_stopping_contract": _mapping(training["early_stopping"], label="recipe.training.early_stopping"),
    }


def _write_new(path: Path, payload: object) -> None:
    if path.exists() or path.is_symlink():
        raise Stage1LaunchRefusal(f"refusing to overwrite plan receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise Stage1LaunchRefusal(f"refusing to overwrite plan receipt: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--corpus-receipt", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        recipe, recipe_raw = _json(args.recipe, label="Stage-1 recipe")
        if _sha256(recipe_raw) != RECIPE_SHA256:
            raise Stage1LaunchRefusal("recipe SHA-256 differs from the registered Stage-1 recipe")
        plan = build_plan(recipe=recipe, corpus_receipt_path=args.corpus_receipt, run_root=args.run_root)
        _write_new(args.out, plan)
    except Stage1LaunchRefusal as exc:
        print(f"EXPERT-ITERATION STAGE1 LAUNCH PLAN REFUSED: {exc}")
        return 2
    print(f"WROTE EXPERT-ITERATION STAGE1 LAUNCH PLAN: {args.out} runs={plan['registered_run_count']} status={plan['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
