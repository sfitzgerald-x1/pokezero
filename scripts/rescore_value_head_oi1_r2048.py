#!/usr/bin/env python3
"""Rescore Phase-3 heads on the exact OI-1 R=2048 confirmation states.

The confirmation corpus fixes source games, sibling arms, and policy-continuation
truth under the iteration-2533 checkpoint.  Phase-3 continuations deliberately
update more than their value heads, so the older ``rescore_value_head.py`` is
correct to refuse them: it assumes a bit-identical trunk so candidate policies
could have produced the original source games.  This tool does *not* make that
claim.  It always replays source games and selects sibling arms with the frozen
corpus checkpoint, then evaluates every candidate's value head on those exact
successor observation histories.  The candidates cannot select a different
pair, and their changed trunks are the object being measured.

R=64 screen rows, incomplete continuations, a changed shard inventory, source
arm drift, changed source-head reproduction, incompatible observation schemas,
or a pre-existing output directory all refuse.  The output is a set of
create-only cells accepted by ``value_head_ordering_auc_r2048.py``.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
for directory in (REPO / "src", REPO / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import value_head_ordering_auc_r2048 as oi  # noqa: E402
import value_head_sibling_probe as probe  # noqa: E402


CORPUS_SCHEMA = "pokezero.oi1-targeted-gap-bank.v1"
SOURCE_NAME = "__source_reproduce__"
# The immutable targeted-gap contract identifies this exact iteration-2533
# source model.  A path alone is never sufficient provenance for replay.
SOURCE_CHECKPOINT_SHA256 = "0897676c295a79bac0b24c347b8f6a72e1359b98c37327bc5639fb6229005937"
# The source checkpoint's belief provenance was sealed with the original
# VHProbe bank.  Replay may not silently substitute a different Showdown set
# source, even if the source model itself still loads.
SOURCE_BELIEF_SET_HASH = "f5a5265143d423af"
# These are instrument geometry, not operator-tunable performance switches.
# Source replay chooses the registered game/prefix/action states, so it must
# stay CPU-identical to the confirmation corpus.  Candidate heads are evaluated
# only after those states are fixed and therefore use the registered CUDA path.
SOURCE_DEVICE = "cpu"
CANDIDATE_DEVICE = "cuda"
SOURCE_MAX_DECISION_ROUNDS = 250
REPRODUCTION_TOL = 1e-4
_CANDIDATE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


class Refusal(RuntimeError):
    """The proposed rescore cannot be tied to the registered corpus."""


def _refuse(message: str) -> None:
    raise Refusal(f"REFUSING: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _refuse(f"cannot read strict JSON {path}: {exc}")
    if not isinstance(value, Mapping):
        _refuse(f"{path.name} must contain a JSON object")
    return value


def _exact_zero_caps(pair: Mapping[str, Any], *, label: str) -> None:
    """Validate the registered producer's sparse cap-count encoding.

    ``value_head_sibling_probe.py`` only emits ``capped_<arm>`` when one or
    more trials cap.  Its omission is therefore the pinned producer's canonical
    zero encoding; a present value must still be an exact integer zero.
    """
    for field in ("capped_a", "capped_b"):
        if field in pair and (type(pair[field]) is not int or pair[field] != 0):
            _refuse(f"{label} has a nonzero or malformed {field}")


def _complete_pair(pair: Mapping[str, Any], *, label: str) -> tuple[int, int, str]:
    if pair.get("rollouts_a") != oi.ROLLOUTS or pair.get("rollouts_b") != oi.ROLLOUTS:
        _refuse(f"{label} is not complete R={oi.ROLLOUTS} evidence")
    _exact_zero_caps(pair, label=label)
    for field in ("failed_a", "failed_b"):
        if not isinstance(pair.get(field), list) or pair[field] != []:
            _refuse(f"{label} lacks a verified empty {field} list")
    if pair.get("pairing_intact") is not True:
        _refuse(f"{label} does not retain paired rollout trials")
    for field in ("seed", "prefix"):
        if type(pair.get(field)) is not int:
            _refuse(f"{label} lacks integer {field}")
    if pair.get("seat") != "p1":
        _refuse(f"{label} has nonregistered seat {pair.get('seat')!r}")
    for field in ("arm_a", "arm_b"):
        if type(pair.get(field)) is not int:
            _refuse(f"{label} lacks integer {field}")
    for field in ("true_a", "true_b", "true_gap", "head_a", "head_b", "head_gap"):
        try:
            value = float(pair[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise Refusal(f"REFUSING: {label} lacks finite {field}") from exc
        if not math.isfinite(value):
            _refuse(f"{label} has non-finite {field}")
    if abs((float(pair["true_a"]) - float(pair["true_b"])) - float(pair["true_gap"])) > 1e-12:
        _refuse(f"{label} true_gap disagrees with true_a/true_b")
    if abs((float(pair["head_a"]) - float(pair["head_b"])) / 2.0 - float(pair["head_gap"])) > 1e-12:
        _refuse(f"{label} head_gap disagrees with head_a/head_b")
    return int(pair["seed"]), int(pair["prefix"]), str(pair["seat"])


def load_confirmation(corpus_path: Path, confirmation_dir: Path) -> tuple[
        list[dict[str, Any]], Mapping[str, Any], str, dict[str, str]]:
    """Load only the complete confirmation rows named by the sealed corpus bank."""
    corpus = _json(corpus_path)
    if corpus.get("schema_version") != CORPUS_SCHEMA:
        _refuse(f"{corpus_path.name} is not an OI-1 confirmation corpus bank")
    if corpus.get("contract_sha256") != oi.CONTRACT_SHA256:
        _refuse("corpus contract SHA differs from the registered R=2048 scorer")
    want = corpus.get("confirmation_shard_sha256")
    if not isinstance(want, Mapping) or not want:
        _refuse("corpus lacks a confirmation-shard digest map")
    expected = {str(name): digest for name, digest in want.items()}
    if any(not isinstance(digest, str) or len(digest) != 64 for digest in expected.values()):
        _refuse("corpus has malformed confirmation-shard digest metadata")
    found = {path.name: path for path in confirmation_dir.glob("confirm-*.json")}
    if set(found) != set(expected):
        _refuse("confirmation directory does not exactly match the corpus shard inventory")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for name in sorted(expected):
        path = found[name]
        if sha256_file(path) != expected[name]:
            _refuse(f"confirmation shard {name} differs from the corpus digest")
        raw = _json(path)
        config = raw.get("config")
        if not isinstance(config, Mapping):
            _refuse(f"confirmation shard {name} lacks config")
        if (config.get("rollouts"), config.get("rollout_seed_salt"), config.get("device")) != (
                oi.ROLLOUTS, "oi1-targeted-gap-confirm-v1", "cpu"):
            _refuse(f"confirmation shard {name} has foreign R=2048 rollout provenance")
        pairs = raw.get("pairs")
        if not isinstance(pairs, list):
            _refuse(f"confirmation shard {name} lacks pairs")
        for index, value in enumerate(pairs):
            if not isinstance(value, Mapping):
                _refuse(f"confirmation shard {name} pair {index} is not an object")
            row = dict(value)
            key = _complete_pair(row, label=f"confirmation shard {name} pair {index}")
            if key in seen:
                _refuse(f"duplicate confirmation key {key}")
            seen.add(key)
            rows.append(row)
    if len(rows) != corpus.get("complete_confirmation_pairs"):
        _refuse("confirmation pair count differs from sealed corpus bank")
    if len(rows) < oi.MIN_COMPLETE_PAIRS:
        _refuse("confirmation corpus falls below the registered complete-pair minimum")
    actual_eligible = sum(abs(float(row["true_gap"])) >= oi.TAU_PRIMARY for row in rows)
    if actual_eligible != corpus.get("primary_tau_eligible_pairs"):
        _refuse("confirmation tau-eligible count differs from sealed corpus bank")
    if actual_eligible < oi.MIN_PRIMARY_ELIGIBLE_PAIRS:
        _refuse("confirmation corpus falls below the registered tau-primary minimum")
    return rows, corpus, sha256_file(corpus_path), expected


def _check_candidate_observation_schema(source_result: Any, candidate_result: Any,
                                        showdown_root: Path) -> None:
    """Refuse a candidate whose value head would read a different observation schema."""
    from pokezero.neural_policy import (
        category_vocab_from_model_config,
        feature_masks_from_model_config,
        observation_spec_from_model_config,
    )
    source_config = source_result.model_config
    candidate_config = candidate_result.model_config
    if observation_spec_from_model_config(candidate_config) != observation_spec_from_model_config(source_config):
        _refuse("candidate observation specification differs from the frozen confirmation source")
    if feature_masks_from_model_config(candidate_config) != feature_masks_from_model_config(source_config):
        _refuse("candidate feature masks differ from the frozen confirmation source")
    if category_vocab_from_model_config(candidate_config, showdown_root) != category_vocab_from_model_config(source_config, showdown_root):
        _refuse("candidate category vocabulary differs from the frozen confirmation source")


def _metadata(*, corpus_sha256: str, shards: Mapping[str, str], pair_sha256: str,
              source_checkpoint_sha256: str, candidate_checkpoint_sha256: str,
              source_reproduction: Mapping[str, Any], source_belief_set_hash: str) -> dict[str, Any]:
    return {
        "schema": oi.SCHEMA,
        "contract_sha256": oi.CONTRACT_SHA256,
        "stage": "confirmation",
        "rollouts_per_arm": oi.ROLLOUTS,
        "screen_rows_included": False,
        "corpus_sha256": corpus_sha256,
        "confirmation_shard_sha256": dict(shards),
        "pair_set_sha256": pair_sha256,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "phase3_rescore": {
            "schema": "pokezero.phase3.oi1-r2048-rescore.v1",
            "candidate_checkpoint_sha256": candidate_checkpoint_sha256,
            "source_belief_set_hash": source_belief_set_hash,
            "source_reproduction": dict(source_reproduction),
            "semantics": (
                "Source games, p1 sibling arms, opponent replies, and R=2048 truth are rebuilt "
                "only from the frozen corpus checkpoint. Candidate checkpoints evaluate those "
                "same successor observation histories; candidate trunk identity is intentionally "
                "not required because Phase-3 continuation updates are the measured treatment."
            ),
        },
    }


def _safe_candidate_name(name: str) -> None:
    """Keep a candidate label a single output filename component."""
    if name == SOURCE_NAME or not _CANDIDATE_NAME.fullmatch(name):
        _refuse(f"candidate name {name!r} is not a safe single filename component")


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _validate_source_reproduction(value: Mapping[str, Any], *, expected_n: int) -> None:
    """Reject mutable source-replay claims before they can be published."""
    if type(value.get("n")) is not int or value["n"] != expected_n:
        _refuse("source reproduction count does not exactly cover the confirmation cell")
    if value.get("tol") != REPRODUCTION_TOL:
        _refuse(f"source reproduction tolerance must remain {REPRODUCTION_TOL:g}")
    for field in ("max_abs_delta", "mean_abs_delta"):
        number = value.get(field)
        if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
            _refuse(f"source reproduction has malformed {field}")
        if number > REPRODUCTION_TOL:
            _refuse(f"source reproduction {field} exceeds {REPRODUCTION_TOL:g}")
    if value.get("source_device") != SOURCE_DEVICE:
        _refuse("source reproduction must be replayed on CPU")
    if value.get("candidate_device") != CANDIDATE_DEVICE:
        _refuse("candidate values must use the registered CUDA evaluation path")
    if value.get("source_max_decision_rounds") != SOURCE_MAX_DECISION_ROUNDS:
        _refuse(f"source decision cap must remain {SOURCE_MAX_DECISION_ROUNDS}")


def write_cells_new(out_dir: Path, cells: Mapping[str, list[dict[str, Any]]], *,
                    corpus_sha256: str, shard_sha256: Mapping[str, str],
                    source_checkpoint_sha256: str,
                    checkpoint_sha256: Mapping[str, str],
                    source_reproduction: Mapping[str, Any],
                    source_belief_set_hash: str) -> None:
    """Publish every candidate only after the entire replay succeeded, once."""
    if out_dir.exists():
        _refuse(f"output directory {out_dir} already exists; rescore evidence is create-only")
    prepared: dict[str, bytes] = {}
    for name, rows in cells.items():
        _safe_candidate_name(name)
        if source_checkpoint_sha256 != SOURCE_CHECKPOINT_SHA256:
            _refuse("cell source checkpoint is not the registered iteration-2533 checkpoint")
        if source_belief_set_hash != SOURCE_BELIEF_SET_HASH:
            _refuse("cell source belief provenance is not the registered iteration-2533 value")
        if not _valid_digest(checkpoint_sha256.get(name)):
            _refuse(f"candidate {name} lacks a lowercase SHA-256 checkpoint digest")
        _validate_source_reproduction(source_reproduction, expected_n=len(rows))
        keyed = {oi.base.pair_key(row): row for row in rows}
        if len(keyed) != len(rows):
            _refuse(f"candidate {name} has duplicate confirmation keys")
        payload = {
            "pairs": rows,
            "oi1_targeted_gap_r2048": _metadata(
                corpus_sha256=corpus_sha256, shards=shard_sha256,
                pair_sha256=oi.pair_set_sha256(keyed),
                source_checkpoint_sha256=source_checkpoint_sha256,
                candidate_checkpoint_sha256=checkpoint_sha256[name],
                source_reproduction=source_reproduction,
                source_belief_set_hash=source_belief_set_hash),
        }
        prepared[name] = (json.dumps(payload, allow_nan=False, indent=1, sort_keys=True) + "\n").encode()
    try:
        out_dir.mkdir(parents=True, exist_ok=False)
        for name, payload in prepared.items():
            path = out_dir / f"{name}.json"
            with path.open("xb") as handle:
                handle.write(payload)
    except FileExistsError as exc:  # pragma: no cover - race after the directory check
        _refuse(f"output {exc.filename} already exists; rescore evidence is create-only")


def _parse(spec: str) -> tuple[str, Path]:
    name, marker, value = spec.partition("=")
    if not name or marker != "=" or not value:
        _refuse(f"--head requires unique NAME=CHECKPOINT (and reserves {SOURCE_NAME!r})")
    if name == SOURCE_NAME:
        _refuse(f"--head reserves {SOURCE_NAME!r} for the frozen source checkpoint")
    _safe_candidate_name(name)
    return name, Path(value)


def run(args: argparse.Namespace) -> None:
    rows, _corpus, corpus_sha256, shard_sha256 = load_confirmation(
        args.corpus, args.confirmation_dir)
    heads: dict[str, Path] = {}
    for raw in args.head:
        name, path = _parse(raw)
        if name in heads:
            _refuse(f"duplicate --head name {name!r}")
        heads[name] = path
    if not heads:
        _refuse("at least one Phase-3 --head is required")
    if not args.source_checkpoint.is_file():
        _refuse(f"source checkpoint does not exist: {args.source_checkpoint}")

    from pokezero.local_showdown import (
        LocalShowdownConfig, LocalShowdownEnv, env_config_from_checkpoint_provenance,
    )
    from pokezero.neural_policy import (
        TransformerSoftmaxPolicy, category_vocab_from_model_config,
        evaluate_transformer_action_priors, evaluate_transformer_observation_value,
        feature_masks_from_model_config, load_transformer_checkpoint,
        observation_spec_from_model_config,
    )
    from pokezero.replay_branching import replay_trajectory_branch
    from pokezero.rollout import RolloutConfig, continue_rollout_from_current_state
    from pokezero.search import player_observation_history

    source_sha = sha256_file(args.source_checkpoint)
    if source_sha != SOURCE_CHECKPOINT_SHA256:
        _refuse("source checkpoint digest is not the registered targeted-gap iteration-2533 checkpoint")
    # Source action sampling and source-head reproduction must preserve the CPU
    # geometry in the confirmation contract. Candidate values are independent
    # forward passes on already-fixed histories and may use the GPU.
    source_model, source_result = load_transformer_checkpoint(
        args.source_checkpoint, map_location=SOURCE_DEVICE)
    if sha256_file(args.source_checkpoint) != source_sha:
        _refuse("source checkpoint changed while it was loaded")
    loaded: dict[str, tuple[Any, Any]] = {SOURCE_NAME: (source_model, source_result)}
    head_sha: dict[str, str] = {SOURCE_NAME: source_sha}
    for name, path in heads.items():
        before = sha256_file(path)
        model, result = load_transformer_checkpoint(path, map_location=CANDIDATE_DEVICE)
        if sha256_file(path) != before:
            _refuse(f"candidate checkpoint {name} changed while it was loaded")
        _check_candidate_observation_schema(source_result, result, args.showdown_root)
        loaded[name] = (model, result)
        head_sha[name] = before

    source_spec = observation_spec_from_model_config(source_result.model_config)
    source_masks = feature_masks_from_model_config(source_result.model_config)
    source_vocab = category_vocab_from_model_config(source_result.model_config, args.showdown_root)
    source_belief = getattr(source_result, "belief_set_source_hash", None)
    if not isinstance(source_belief, str) or not source_belief:
        _refuse("source checkpoint lacks required belief-set provenance")
    if source_belief != SOURCE_BELIEF_SET_HASH:
        _refuse("source checkpoint belief-set provenance is not the registered iteration-2533 value")

    def make_env() -> Any:
        base = LocalShowdownConfig(showdown_root=args.showdown_root, set_belief_source=True)
        cfg = env_config_from_checkpoint_provenance(
            base, source_masks, context="rescore_value_head_oi1_r2048",
            required_specs=source_spec, required_vocabs=source_vocab)
        env = LocalShowdownEnv(cfg)
        if getattr(env, "belief_set_source_hash", None) != source_belief:
            _refuse("source checkpoint belief-set provenance differs from replay environment")
        return env

    def source_policy() -> Any:
        return TransformerSoftmaxPolicy(
            model=source_model, result=source_result, device=SOURCE_DEVICE,
            deterministic=False, sampling_temperature=1.0)

    by_seed: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)
    output: dict[str, list[dict[str, Any]]] = {name: [] for name in heads}
    reproduction: list[float] = []
    start = time.monotonic()
    source_config = RolloutConfig(max_decision_rounds=SOURCE_MAX_DECISION_ROUNDS)
    for ordinal, seed in enumerate(sorted(by_seed), start=1):
        wanted = sorted(by_seed[seed], key=lambda item: int(item["prefix"]))
        try:
            env = make_env()
            env.reset(seed=seed)
            source = continue_rollout_from_current_state(
                env=env, policies={"p1": source_policy(), "p2": source_policy()},
                config=source_config, seed=seed, battle_id=f"probe-{seed}",
                starting_decision_round_index=0)
        except Exception as exc:  # noqa: BLE001
            _refuse(f"source replay failed at seed {seed}: {type(exc).__name__}: {exc}")
        trajectory = source.trajectory
        print(f"[{ordinal}/{len(by_seed)}] seed={seed}: {len(wanted)} confirmation pairs, "
              f"{time.monotonic() - start:.0f}s elapsed", flush=True)
        for want in wanted:
            prefix, seat = int(want["prefix"]), str(want["seat"])
            opponent = "p2" if seat == "p1" else "p1"
            try:
                history = player_observation_history(
                    trajectory, player_id=seat, through_decision_round=prefix)
                arm_a, arm_b, opponent_action = probe._top_two_and_opponent(
                    trajectory, seat, prefix, source_model, source_result, SOURCE_DEVICE,
                    evaluate_transformer_action_priors, history)
            except Exception as exc:  # noqa: BLE001
                _refuse(f"cannot rederive source sibling arms at {(seed, prefix, seat)}: {exc}")
            if (arm_a, arm_b) != (want["arm_a"], want["arm_b"]):
                _refuse(f"source arm mismatch at {(seed, prefix, seat)}: "
                        f"replayed {(arm_a, arm_b)} vs corpus {(want['arm_a'], want['arm_b'])}")
            values: dict[str, dict[str, float]] = {name: {} for name in loaded}
            for label, arm in (("a", arm_a), ("b", arm_b)):
                try:
                    branch_env = make_env()
                    branch_env.reset(seed=seed)
                    branch = replay_trajectory_branch(
                        branch_env, trajectory, prefix_decision_round_count=prefix,
                        branch_actions={seat: arm, opponent: opponent_action},
                        check_prefix_observations=True)
                    successor_history, terminal = probe._post_branch_history(branch, seat, history)
                except Exception as exc:  # noqa: BLE001
                    _refuse(f"cannot replay source successor at {(seed, prefix, seat, label)}: {exc}")
                if terminal is not None:
                    _refuse(f"confirmation corpus pair {(seed, prefix, seat)} replayed to terminal successor")
                if successor_history is None:
                    fallback = branch_env.observe(seat)
                    if fallback is None or not want.get(f"observe_fallback_{label}"):
                        _refuse(f"source successor observation drift at {(seed, prefix, seat, label)}")
                    successor_history = (*history, fallback)
                elif want.get(f"observe_fallback_{label}"):
                    _refuse(f"source successor fallback flag drift at {(seed, prefix, seat, label)}")
                for name, (model, result) in loaded.items():
                    values[name][label] = evaluate_transformer_observation_value(
                        model=model, result=result, observations=successor_history,
                        device=(SOURCE_DEVICE if name == SOURCE_NAME else CANDIDATE_DEVICE))
            source_delta = max(abs(values[SOURCE_NAME]["a"] - float(want["head_a"])),
                               abs(values[SOURCE_NAME]["b"] - float(want["head_b"])))
            reproduction.append(source_delta)
            if source_delta > REPRODUCTION_TOL:
                _refuse(f"source head reproduction exceeds {REPRODUCTION_TOL:g} at "
                        f"{(seed, prefix, seat)} ({source_delta:.3e})")
            for name in heads:
                scored = dict(want)
                scored["head_a"] = values[name]["a"]
                scored["head_b"] = values[name]["b"]
                probe.finalize_pair_gaps(scored)
                if abs(float(scored["true_gap"]) - float(want["true_gap"])) > 1e-12:
                    _refuse(f"ground-truth mutation while rescoring {(seed, prefix, seat)}")
                output[name].append(scored)
    if len(reproduction) != len(rows):
        _refuse("not every confirmation state was reproduced")
    source_reproduction = {
        "n": len(reproduction), "tol": REPRODUCTION_TOL,
        "max_abs_delta": max(reproduction),
        "mean_abs_delta": sum(reproduction) / len(reproduction),
        "source_device": SOURCE_DEVICE,
        "candidate_device": CANDIDATE_DEVICE,
        "source_max_decision_rounds": SOURCE_MAX_DECISION_ROUNDS,
    }
    write_cells_new(
        args.out_dir, output, corpus_sha256=corpus_sha256, shard_sha256=shard_sha256,
        source_checkpoint_sha256=source_sha,
        checkpoint_sha256={name: head_sha[name] for name in heads},
        source_reproduction=source_reproduction,
        source_belief_set_hash=source_belief)
    print(f"WROTE OI1 R2048 PHASE3 RESCORES: {args.out_dir}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--confirmation-dir", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--head", action="append", default=[], metavar="NAME=CHECKPOINT")
    parser.add_argument("--showdown-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Refusal as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
