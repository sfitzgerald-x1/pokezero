#!/usr/bin/env python3
"""Run one source-bound, durable guided-MCTS-versus-raw-policy study.

This is deliberately a separate contract from ``mcts_mcts_h2h.py``.  The
candidate is context-aware engine MCTS; the baseline is the same checkpoint's
context-free deterministic masked-argmax policy.  Both seats of each seed are
played, and every completed game is atomically persisted before the next game
can begin.  The companion durable launcher owns the process terminal receipt.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pokezero.mcts_eval.head_to_head import (  # noqa: E402
    HeadToHeadError,
    MctsPolicySpec,
    PublicOnlyMctsPolicy,
    complete_pair,
    load_pair,
    play_mirrored_pair,
    summarize_complete_pairs,
    write_game_immutable,
)
from pokezero.public_decision_corpus import PublicDecisionRecord  # noqa: E402

# The mature MCTS-versus-MCTS runner owns the source-hash, immutable-write,
# Showdown-binding and durable-launcher primitives.  This runner intentionally
# reuses those primitives rather than carrying a second, subtly weaker version.
from mcts_mcts_h2h import (  # noqa: E402
    _durable_output_root,
    _require_durable_launcher_handoff,
    _runtime_spec,
    _sha256_file,
    _showdown_source_provenance,
    _source_provenance,
    _write_immutable_json,
    _write_progress_json,
)


MANIFEST_SCHEMA_VERSION = "pokezero.mcts-guided-vs-raw-manifest.v1"
PROGRESS_SCHEMA_VERSION = "pokezero.mcts-guided-vs-raw-progress.v1"
COMPLETE_SCHEMA_VERSION = "pokezero.mcts-guided-vs-raw-complete.v1"
PUBLIC_DECISION_EVIDENCE_SCHEMA_VERSION = "pokezero.mcts-guided-vs-raw-public-decision.v1"
RAW_SELECTOR = {
    "kind": "deterministic_masked_argmax",
    "deterministic": True,
    "exploration_epsilon": 0.0,
    "sampling_temperature": 1.0,
    "family_gated_selection": False,
    "search": False,
}
REGISTERED_ENGINE_CONFIG = {
    "approximate_hidden_duration_volatiles": True,
    "approximate_partial_trap_turns": True,
    "approximate_sleep_turns": True,
    "approximate_substitute_health": True,
    "c_puct": 1.4,
    "deep_ko_split": True,
    "depth_min": None,
    "early_stop": False,
    "early_stop_min_sims": 64,
    "fold_cross_check": False,
    "fpu_reduction": None,
    "ladder_saturation": 0.9,
    "leaf_batch": 1,
    "leaf_batch_fidelity_loss_ack": False,
    "leaf_eval": "model",
    "model_decision_time_ms": 1_000,
    "model_device": "cpu",
    "model_native_batch_guard_ms": 64,
    "model_priors": True,
    "model_world_workers": 1,
    "override_telemetry": True,
    "rollout_branch_on_damage": False,
    "rollout_count": 32,
    "rollout_leaf_eval": False,
    "rollout_max_plies": 200,
    "rollout_policy": "uniform",
    "rollout_seed": 0,
    "rollout_threads": 1,
    "rollout_threads_cpu_budget_ack": False,
    "root_selector_shadow": False,
    "sample_retry_factor": 4,
    "search_batch": 16,
    "search_depth": 2,
    "search_sims": 256,
    "search_time_ms": 100,
    "strict_fallbacks": True,
    "threads": 1,
    "use_opponent_priors": False,
    "worlds": 4,
    "worlds_min": None,
}
SOURCE_BOUND_ENGINE_PATHS = {"checkpoint_path", "model_path", "tables_path"}


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HeadToHeadError(f"{label} must be a JSON object.")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_manifest(path: str | Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HeadToHeadError(f"cannot read guided-vs-raw manifest: {error}") from error
    manifest = _mapping(payload, label="manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise HeadToHeadError(
            f"manifest schema {manifest.get('schema_version')!r} is not "
            f"{MANIFEST_SCHEMA_VERSION!r}."
        )
    return manifest


def _hex(value: object, *, label: str, length: int) -> str:
    text = str(value)
    if len(text) != length or any(character not in "0123456789abcdef" for character in text):
        raise HeadToHeadError(f"{label} must be a {length}-character lowercase SHA-256 value.")
    return text


def _seeds(manifest: Mapping[str, Any]) -> tuple[int, ...]:
    raw = manifest.get("seeds")
    if not isinstance(raw, list) or not raw:
        raise HeadToHeadError("manifest.seeds must be a non-empty JSON list.")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in raw):
        raise HeadToHeadError("manifest.seeds must contain non-negative integer seeds only.")
    seeds = tuple(raw)
    if len(set(seeds)) != len(seeds):
        raise HeadToHeadError("manifest.seeds contains a duplicate mirrored-pair key.")
    return seeds


def _bootstrap(manifest: Mapping[str, Any]) -> tuple[int, int, float]:
    raw = _mapping(manifest.get("bootstrap"), label="manifest.bootstrap")
    resamples = raw.get("resamples")
    seed = raw.get("seed")
    confidence = raw.get("confidence_level")
    if (
        isinstance(resamples, bool)
        or not isinstance(resamples, int)
        or resamples <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 < float(confidence) < 1.0
    ):
        raise HeadToHeadError("manifest.bootstrap is malformed.")
    return resamples, seed, float(confidence)


def _raw_spec(
    raw: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    source_commit: str,
    source_tree_sha256: str,
    engine_fingerprint: str,
    showdown_source_sha256: str,
) -> MctsPolicySpec:
    """Freeze the no-search baseline without pretending it is engine MCTS."""

    if dict(_mapping(raw.get("selector"), label="raw.selector")) != RAW_SELECTOR:
        raise HeadToHeadError(
            "raw.selector must exactly declare deterministic masked argmax with no search."
        )
    for field, actual in (
        ("source_commit", source_commit),
        ("source_tree_sha256", source_tree_sha256),
        ("engine_fingerprint", engine_fingerprint),
        ("checkpoint_sha256", checkpoint_sha256),
        ("showdown_source_sha256", showdown_source_sha256),
    ):
        if str(raw.get(field, "")) != actual:
            raise HeadToHeadError(f"raw.{field} does not match the active verified identity.")
    config_id = str(raw.get("config_id", ""))
    policy_id = str(raw.get("policy_id", ""))
    if not config_id or not policy_id:
        raise HeadToHeadError("raw config_id and policy_id must be non-empty.")
    return MctsPolicySpec(
        config_id=config_id,
        policy_id=policy_id,
        source_commit=source_commit,
        source_tree_sha256=source_tree_sha256,
        engine_fingerprint=engine_fingerprint,
        checkpoint_sha256=checkpoint_sha256,
        showdown_source_sha256=showdown_source_sha256,
        config={"policy_kind": "raw_transformer_policy", "selector": dict(RAW_SELECTOR)},
    )


@dataclass
class RawPolicyStats:
    """Explicit zero-search telemetry for the raw-policy baseline."""

    decisions: int = 0
    searched_decisions: int = 0
    fallback_decisions: int = 0
    model_evals: int = 0
    total_iterations: int = 0
    worlds_constructed: int = 0
    worlds_searched: int = 0
    prior_fallbacks: int = 0
    root_prior_fallbacks: int = 0
    branch_prior_fallbacks: int = 0
    opponent_prior_arm_decisions: int = 0
    override_measured_decisions: int = 0
    model_override_decisions: int = 0
    raw_forward_decisions: int = 0
    opponent_request_order_statuses: dict[str, int] | None = None
    opponent_request_order_root_fallback_statuses: dict[str, int] | None = None
    decision_wall_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.opponent_request_order_statuses is None:
            self.opponent_request_order_statuses = {}
        if self.opponent_request_order_root_fallback_statuses is None:
            self.opponent_request_order_root_fallback_statuses = {}

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


class DeterministicRawPolicyAdapter:
    """Give the raw policy a safe context entry point and honest telemetry."""

    requires_public_materialization_state = False

    def __init__(self, policy: Any, *, policy_id: str) -> None:
        self._policy = policy
        self.policy_id = policy_id
        self.stats = RawPolicyStats()

    def select_action(self, observation: Any, *, rng: Any) -> Any:
        started = time.perf_counter()
        try:
            decision = self._policy.select_action(observation, rng=rng)
            if not hasattr(decision, "policy_id"):
                raise HeadToHeadError("raw policy did not return a policy-identified decision.")
            self.stats.raw_forward_decisions += 1
            return replace(decision, policy_id=self.policy_id)
        finally:
            self.stats.decisions += 1
            self.stats.decision_wall_seconds += time.perf_counter() - started

    def select_action_with_context(self, context: Any, *, rng: Any) -> Any:
        # The raw policy only consumes its own observation.  It never receives
        # an opponent request, legal mask, or private history from this wrapper.
        return self.select_action(context.observation, rng=rng)

    def reset(self) -> None:
        self._policy.reset()

    def close(self) -> None:
        return None


def _validated_study(manifest: Mapping[str, Any], *, seeds: tuple[int, ...]) -> Mapping[str, Any]:
    study = _mapping(manifest.get("study"), label="manifest.study")
    if study.get("kind") != "guided_mcts_vs_raw_policy":
        raise HeadToHeadError("study.kind must be guided_mcts_vs_raw_policy.")
    if study.get("pairs") != len(seeds):
        raise HeadToHeadError("study.pairs must equal the exact manifest seed count.")
    if study.get("mirrored_games") != len(seeds) * 2:
        raise HeadToHeadError("study.mirrored_games must be exactly twice study.pairs.")
    if study.get("failure_retry_policy") != {
        "interrupted_before_runner_terminal": "resume_same_root_with_fresh_launcher_attempt",
        "nonzero_runner_exit": "terminal_failed_no_retry",
        "malformed_runner_terminal": "nonbankable_no_retry",
        "completed_game_units": "immutable_reuse_only",
    }:
        raise HeadToHeadError("study.failure_retry_policy is not the registered durable policy.")
    return study


def _require_registered_candidate_config(config: Mapping[str, Any]) -> None:
    """Reject a look-alike MCTS configuration before any game is played."""

    if config.get("model_priors") is not True or config.get("use_opponent_priors") is not False:
        raise HeadToHeadError("candidate must enable own model priors and disable opponent priors.")
    observed = {key: value for key, value in config.items() if key not in SOURCE_BOUND_ENGINE_PATHS}
    if observed != REGISTERED_ENGINE_CONFIG:
        raise HeadToHeadError("candidate differs from the registered one-second guided MCTS configuration.")


def _raw_witness_path(out_root: Path, *, seed: int, candidate_seat: str) -> Path:
    if candidate_seat not in {"p1", "p2"}:
        raise HeadToHeadError("raw selector witness has an invalid candidate seat.")
    return out_root / "raw-selector-witnesses" / f"seed-{seed}-{candidate_seat}.json"


def _public_decision_path(
    out_root: Path,
    *,
    seed: int,
    candidate_seat: str,
    record: PublicDecisionRecord,
) -> Path:
    if candidate_seat not in {"p1", "p2"}:
        raise HeadToHeadError("public decision evidence has an invalid candidate seat.")
    if record.seed != seed or record.acting_player != candidate_seat:
        raise HeadToHeadError("public decision evidence does not match its game identity.")
    return (
        out_root
        / "public-decision-records"
        / f"seed-{seed}-{candidate_seat}"
        / f"turn-{record.turn_index:03d}-{record.decision_id}.json"
    )


def _public_decision_payload(
    *,
    candidate: MctsPolicySpec,
    incumbent: MctsPolicySpec,
    candidate_seat: str,
    record: PublicDecisionRecord,
) -> dict[str, Any]:
    return {
        "schema_version": PUBLIC_DECISION_EVIDENCE_SCHEMA_VERSION,
        "seed": record.seed,
        "candidate_seat": candidate_seat,
        "candidate_provenance_sha256": candidate.provenance_sha256,
        "raw_provenance_sha256": incumbent.provenance_sha256,
        "record": record.to_dict(),
    }


def _public_decision_writer(
    out_root: Path,
    *,
    candidate: MctsPolicySpec,
    incumbent: MctsPolicySpec,
    seed: int,
    candidate_seat: str,
):
    """Persist guided decisions as individually immutable public replay units."""

    def write(record: PublicDecisionRecord) -> None:
        # The rollout hook reports both actors. Only the guided actor is in
        # scope for the override audit; retaining raw's private decision view
        # would add storage without adding a search hypothesis.
        if record.acting_player != candidate_seat:
            return
        path = _public_decision_path(
            out_root, seed=seed, candidate_seat=candidate_seat, record=record
        )
        _write_immutable_json(
            path,
            _public_decision_payload(
                candidate=candidate,
                incumbent=incumbent,
                candidate_seat=candidate_seat,
                record=record,
            ),
        )

    return write


def _raw_witness_payload(game: Any, *, raw_forward_decisions: int) -> dict[str, Any]:
    raw_decisions = game.incumbent_telemetry.decisions
    if raw_forward_decisions != raw_decisions:
        raise HeadToHeadError(
            "raw selector forward count must exactly equal the completed game's raw decision count."
        )
    return {
        "schema_version": "pokezero.mcts-guided-vs-raw-selector-witness.v1",
        "seed": game.seed,
        "candidate_seat": game.candidate_seat,
        "raw_provenance_sha256": game.incumbent.provenance_sha256,
        "raw_policy_id": game.incumbent.policy_id,
        "raw_selector": RAW_SELECTOR,
        "raw_forward_decisions": raw_forward_decisions,
        "raw_telemetry_decisions": raw_decisions,
    }


def _validate_raw_witness(out_root: Path, game: Any) -> Mapping[str, Any]:
    path = _raw_witness_path(out_root, seed=game.seed, candidate_seat=game.candidate_seat)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HeadToHeadError(f"completed game is missing its readable raw selector witness: {error}") from error
    if not isinstance(payload, Mapping):
        raise HeadToHeadError("raw selector witness is not a JSON object.")
    expected = _raw_witness_payload(
        game, raw_forward_decisions=game.incumbent_telemetry.decisions
    )
    if dict(payload) != expected:
        raise HeadToHeadError("raw selector witness differs from its immutable completed game.")
    return payload


def _validate_public_decision_evidence(out_root: Path, game: Any) -> tuple[PublicDecisionRecord, ...]:
    """Require a complete, source-bound public replay unit for every guided action."""

    root = out_root / "public-decision-records" / f"seed-{game.seed}-{game.candidate_seat}"
    if not root.is_dir():
        raise HeadToHeadError("completed game is missing its public decision evidence directory.")
    records: list[PublicDecisionRecord] = []
    expected_battle_id = f"mcts-h2h-{game.seed}-{game.candidate_seat}"
    for path in sorted(root.glob("turn-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise HeadToHeadError(f"cannot read public decision evidence {path}: {error}") from error
        if not isinstance(payload, Mapping):
            raise HeadToHeadError("public decision evidence is not a JSON object.")
        if (
            payload.get("schema_version") != PUBLIC_DECISION_EVIDENCE_SCHEMA_VERSION
            or payload.get("seed") != game.seed
            or payload.get("candidate_seat") != game.candidate_seat
            or payload.get("candidate_provenance_sha256") != game.candidate.provenance_sha256
            or payload.get("raw_provenance_sha256") != game.incumbent.provenance_sha256
        ):
            raise HeadToHeadError("public decision evidence does not match its completed game.")
        try:
            record = PublicDecisionRecord.from_dict(_mapping(payload.get("record"), label="public record"))
        except (TypeError, ValueError) as error:
            raise HeadToHeadError(f"public decision evidence has an invalid record: {error}") from error
        expected_path = _public_decision_path(
            out_root,
            seed=game.seed,
            candidate_seat=game.candidate_seat,
            record=record,
        )
        if path != expected_path:
            raise HeadToHeadError("public decision evidence path does not match its canonical record identity.")
        if record.battle_id != expected_battle_id or record.format_id != "gen3randombattle":
            raise HeadToHeadError("public decision evidence does not bind the completed game identity.")
        records.append(record)
    expected_count = game.candidate_telemetry.decisions
    if expected_count <= 0 or len(records) != expected_count:
        raise HeadToHeadError(
            "public decision evidence count must be positive and equal guided decision telemetry."
        )
    decision_ids = {record.decision_id for record in records}
    turn_indices = {record.turn_index for record in records}
    if len(decision_ids) != len(records) or len(turn_indices) != len(records):
        raise HeadToHeadError("public decision evidence contains duplicate guided decision identities.")
    return tuple(records)


def _validate_completed_game(game: Any) -> None:
    """Reject a persisted game whose live decision evidence is not admissible."""

    candidate = game.candidate_telemetry
    raw = game.incumbent_telemetry
    if candidate.fallback_decisions or raw.fallback_decisions:
        raise HeadToHeadError("a guided-vs-raw game recorded an action fallback.")
    if candidate.root_prior_fallbacks or raw.root_prior_fallbacks:
        raise HeadToHeadError("a guided-vs-raw game recorded a root policy-prior fallback.")
    if (
        raw.searched_decisions
        or raw.model_evals
        or raw.total_iterations
        or raw.worlds_constructed
        or raw.worlds_searched
    ):
        raise HeadToHeadError("raw-policy baseline recorded search work.")


def _validate_summary_evidence(summary: Mapping[str, Any]) -> None:
    """Require aggregate live-root and raw-no-search witnesses before completion."""

    if summary.get("candidate_root_prior_fallbacks") != 0:
        raise HeadToHeadError("guided candidate accumulated a root policy-prior fallback.")
    if summary.get("incumbent_root_prior_fallbacks") != 0:
        raise HeadToHeadError("raw baseline accumulated a root policy-prior fallback.")
    if not isinstance(summary.get("candidate_override_measured_decisions"), int) or summary[
        "candidate_override_measured_decisions"
    ] <= 0:
        raise HeadToHeadError(
            "guided candidate never exposed a live root model-action witness; result is nonbankable."
        )
    if summary.get("incumbent_model_evals") != 0 or summary.get("incumbent_iterations") != 0:
        raise HeadToHeadError("raw baseline summary contains search work.")


def _progress_writer(out_root: Path, *, candidate: MctsPolicySpec, incumbent: MctsPolicySpec):
    def write(event: str, *, seed: int, candidate_seat: str, decision: Any | None = None) -> None:
        payload: dict[str, Any] = {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "event": event,
            "seed": seed,
            "candidate_seat": candidate_seat,
            "candidate_provenance_sha256": candidate.provenance_sha256,
            "incumbent_provenance_sha256": incumbent.provenance_sha256,
        }
        if decision is not None:
            payload.update(
                {
                    "battle_id": decision.battle_id,
                    "decision_round_index": decision.decision_round_index,
                    "decision_round_count": decision.decision_round_count,
                    "requested_players": list(decision.requested_players),
                    "terminal": decision.terminal,
                    "terminal_capped": decision.terminal_capped,
                    "terminal_winner": decision.terminal_winner,
                }
            )
        _write_progress_json(
            out_root / "progress" / "current.json",
            payload,
            schema_version=PROGRESS_SCHEMA_VERSION,
        )

    return write


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--showdown-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_root = _durable_output_root(args.out_dir)
    _require_durable_launcher_handoff(out_root, runner_script=Path(__file__))
    manifest = _load_manifest(args.manifest)
    seeds = _seeds(manifest)
    study = _validated_study(manifest, seeds=seeds)
    resamples, bootstrap_seed, confidence_level = _bootstrap(manifest)
    max_decision_rounds = manifest.get("max_decision_rounds")
    if isinstance(max_decision_rounds, bool) or not isinstance(max_decision_rounds, int) or max_decision_rounds <= 0:
        raise HeadToHeadError("manifest.max_decision_rounds must be a positive integer.")

    from engine_build_fingerprint import assert_fresh, compute_fingerprint  # noqa: PLC0415
    from pokezero.dex import load_showdown_dex_cached  # noqa: PLC0415
    from pokezero.engine_search import EngineMctsPolicy, EnvTier2AnnotationSource  # noqa: PLC0415
    from pokezero.local_showdown import (  # noqa: PLC0415
        LocalShowdownConfig,
        LocalShowdownEnv,
        env_config_from_checkpoint_provenance,
    )
    from pokezero.mcts_eval.lattice import materialize_search_artifacts  # noqa: PLC0415
    from pokezero.mcts_eval.resolver import resolve_checkpoint_contract  # noqa: PLC0415
    from pokezero.neural_policy import (  # noqa: PLC0415
        category_vocab_from_model_config,
        feature_masks_from_model_config,
        load_transformer_model_config,
        load_transformer_policy,
        observation_spec_from_model_config,
    )
    from pokezero.randbat import load_gen3_randbat_source_cached  # noqa: PLC0415
    from pokezero.rollout import RolloutConfig, RolloutDriver  # noqa: PLC0415

    source = _source_provenance()
    source_commit = source["commit"]
    source_tree_sha256 = source["tree_sha256"]
    if _hex(manifest.get("source_tree_sha256"), label="manifest.source_tree_sha256", length=64) != source_tree_sha256:
        raise HeadToHeadError("manifest source-tree identity does not match the executing image.")
    checkpoint_sha256 = _sha256_file(args.checkpoint)
    if _hex(manifest.get("checkpoint_sha256"), label="manifest.checkpoint_sha256", length=64) != checkpoint_sha256:
        raise HeadToHeadError("manifest checkpoint identity does not match --checkpoint.")
    showdown = _showdown_source_provenance(args.showdown_root)
    showdown_source_sha256 = str(showdown["content_sha256"])
    if _hex(manifest.get("showdown_source_sha256"), label="manifest.showdown_source_sha256", length=64) != showdown_source_sha256:
        raise HeadToHeadError("manifest Showdown identity does not match --showdown-root.")
    assert_fresh()
    engine_fingerprint = str(compute_fingerprint()["fingerprint"])

    contract = resolve_checkpoint_contract(
        args.checkpoint,
        model_device=args.device,
        showdown_root=args.showdown_root,
        showdown_source_sha256=showdown_source_sha256,
        expected_showdown_source_sha256=showdown_source_sha256,
    )
    artifacts = materialize_search_artifacts(contract, showdown_root=args.showdown_root)
    candidate_raw = _mapping(manifest.get("candidate"), label="manifest.candidate")
    candidate, candidate_config = _runtime_spec(
        candidate_raw,
        role="guided candidate",
        checkpoint=args.checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        source_commit=source_commit,
        source_tree_sha256=source_tree_sha256,
        engine_fingerprint=engine_fingerprint,
        showdown_source_sha256=showdown_source_sha256,
        model_path=artifacts["model_path"],
        tables_path=artifacts["tables_path"],
        device=args.device,
    )
    incumbent = _raw_spec(
        _mapping(manifest.get("raw"), label="manifest.raw"),
        checkpoint_sha256=checkpoint_sha256,
        source_commit=source_commit,
        source_tree_sha256=source_tree_sha256,
        engine_fingerprint=engine_fingerprint,
        showdown_source_sha256=showdown_source_sha256,
    )
    _require_registered_candidate_config(candidate.config)

    model_config = load_transformer_model_config(args.checkpoint)
    vocabulary = category_vocab_from_model_config(model_config, args.showdown_root)
    env_config = env_config_from_checkpoint_provenance(
        LocalShowdownConfig(
            showdown_root=args.showdown_root,
            set_belief_source=True,
            category_vocab=vocabulary,
        ),
        feature_masks_from_model_config(model_config),
        required_specs=observation_spec_from_model_config(model_config),
        required_vocabs=vocabulary,
        context="guided-MCTS-versus-raw-policy paired runner",
    )
    dex = load_showdown_dex_cached(args.showdown_root)
    set_source = load_gen3_randbat_source_cached(args.showdown_root)
    write_progress = _progress_writer(out_root, candidate=candidate, incumbent=incumbent)

    _write_immutable_json(
        out_root / "manifest.json",
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "declared_manifest": manifest,
            "active_source": source,
            "active_engine_fingerprint": engine_fingerprint,
            "active_checkpoint_sha256": checkpoint_sha256,
            "active_showdown_source": showdown,
            "candidate": candidate.to_payload(),
            "raw": incumbent.to_payload(),
            "study": study,
            "seeds": list(seeds),
        },
    )

    raw_adapters: dict[tuple[int, str], DeterministicRawPolicyAdapter] = {}

    def session_factory(seed: int, candidate_seat: str):
        write_progress("game_started", seed=seed, candidate_seat=candidate_seat)
        env = LocalShowdownEnv(env_config)
        annotations = EnvTier2AnnotationSource(env)
        guided = PublicOnlyMctsPolicy(
            EngineMctsPolicy(
                dex=dex,
                set_source=set_source,
                config=candidate_config,
                policy_id=candidate.policy_id,
                annotation_source=annotations,
            )
        )
        raw_adapter = DeterministicRawPolicyAdapter(
            load_transformer_policy(
                args.checkpoint,
                device=args.device,
                deterministic=True,
                exploration_epsilon=0.0,
                sampling_temperature=1.0,
                family_gated_selection=False,
            ),
            policy_id=incumbent.policy_id,
        )
        raw_adapters[(seed, candidate_seat)] = raw_adapter
        raw = PublicOnlyMctsPolicy(raw_adapter)
        other_seat = "p2" if candidate_seat == "p1" else "p1"
        driver = RolloutDriver(
            env=env,
            policies={candidate_seat: guided, other_seat: raw},
            config=RolloutConfig(
                max_decision_rounds=max_decision_rounds,
                format_id="gen3randombattle",
                record_policy_timing=True,
                hide_opponent_legal_action_masks=True,
                public_decision_sink=_public_decision_writer(
                    out_root,
                    candidate=candidate,
                    incumbent=incumbent,
                    seed=seed,
                    candidate_seat=candidate_seat,
                ),
                decision_sink=lambda decision: write_progress(
                    "decision_committed", seed=seed, candidate_seat=candidate_seat, decision=decision
                ),
            ),
        )
        return driver, guided, raw

    all_games = []
    for seed in seeds:
        completed = load_pair(out_root, seed=seed, candidate=candidate, incumbent=incumbent)
        for game in completed.values():
            _validate_completed_game(game)
            _validate_raw_witness(out_root, game)
            _validate_public_decision_evidence(out_root, game)

        def on_game(game: Any) -> None:
            _validate_completed_game(game)
            _validate_public_decision_evidence(out_root, game)
            raw_adapter = raw_adapters.pop((game.seed, game.candidate_seat), None)
            if raw_adapter is None:
                raise HeadToHeadError("completed game has no retained raw-policy selector witness.")
            _write_immutable_json(
                _raw_witness_path(
                    out_root, seed=game.seed, candidate_seat=game.candidate_seat
                ),
                _raw_witness_payload(
                    game, raw_forward_decisions=raw_adapter.stats.raw_forward_decisions
                ),
            )
            write_game_immutable(out_root, game)
            write_progress("game_persisted", seed=game.seed, candidate_seat=game.candidate_seat)

        games = play_mirrored_pair(
            seed=seed,
            candidate=candidate,
            incumbent=incumbent,
            candidate_factory=lambda: None,
            incumbent_factory=lambda: None,
            driver_factory=lambda *_: None,
            completed=completed,
            session_factory=session_factory,
            on_game=on_game,
        )
        complete_pair(games, seed=seed, candidate=candidate, incumbent=incumbent)
        for game in games:
            _validate_completed_game(game)
            _validate_raw_witness(out_root, game)
            _validate_public_decision_evidence(out_root, game)
        all_games.extend(games)
        write_progress("pair_completed", seed=seed, candidate_seat="both")
        print(f"completed guided-vs-raw mirrored pair seed={seed}", flush=True)

    summary = summarize_complete_pairs(
        all_games,
        seeds=seeds,
        candidate=candidate,
        incumbent=incumbent,
        bootstrap_resamples=resamples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_confidence_level=confidence_level,
    )
    _validate_summary_evidence(summary)
    raw_forward_decisions = sum(
        int(_validate_raw_witness(out_root, game)["raw_forward_decisions"])
        for game in all_games
    )
    raw_telemetry_decisions = sum(game.incumbent_telemetry.decisions for game in all_games)
    if raw_forward_decisions <= 0 or raw_forward_decisions != raw_telemetry_decisions:
        raise HeadToHeadError(
            "aggregate raw forward evidence must be positive and equal raw decision telemetry."
        )
    _write_immutable_json(out_root / "summary.json", summary)
    complete = {
        "schema_version": COMPLETE_SCHEMA_VERSION,
        "status": "COMPLETE",
        "pairs": len(seeds),
        "games": len(all_games),
        "candidate_provenance_sha256": candidate.provenance_sha256,
        "raw_provenance_sha256": incumbent.provenance_sha256,
        "summary_sha256": _sha256_bytes((json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")),
        "raw_selector": RAW_SELECTOR,
    }
    _write_immutable_json(out_root / "COMPLETE.json", complete)
    print(json.dumps(complete, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
