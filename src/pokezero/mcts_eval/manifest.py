"""Matrix manifest + deterministic identities (plan deliverable 5).

The plan's identity split (section 2.1) exists so that changing *how much
hardware* a study uses never invalidates *what the study is*:

    experiment_id = sha256(frozen_contract_without_resource_profile + matrix_manifest)
    execution_id  = sha256(experiment_id + resource_profile)

Stages 1-4 (materialize, validate, smoke, corpus) are keyed by ``experiment_id``
and survive a concurrency change; stages 5-9 (timing, selection, matrix, merge,
report) are keyed by ``execution_id``. A marker whose identity does not match at
its own scope is terminal and never reused.

``config_id`` is the immutable identity of one lattice cell: depth, sims, batch,
worlds, and inference mode — the fields that change search semantics or wall
time. It is the join key across timing rows, strength rows, and the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

MANIFEST_SCHEMA_VERSION = "pokezero.mcts-depth-eval.manifest.v1"

# Plan section 4 A3: the initial broad lattice.
DEFAULT_DEPTHS: tuple[int, ...] = (2, 4, 6, 8, 10)
DEFAULT_SIMS: tuple[int, ...] = (512, 1024, 2048, 4096, 8192)
DEFAULT_BATCH = 16
DEFAULT_WORLDS = 4
# Section 2: FP-1000 is the frozen primary strength rung.
DEFAULT_FOULPLAY_RUNG_MS = 1000
DEFAULT_MAX_DECISION_ROUNDS = 250
# Section 3: 50 mirrored pairs scored, plus a pre-registered spare band.
DEFAULT_SEED_PAIRS = 50
DEFAULT_SPARE_PAIRS = 10
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class SearchConfig:
    """One lattice cell. ``inference_mode`` is part of identity because a served
    leaf evaluator can change both numerics and wall time."""

    depth: int
    sims: int
    batch: int = DEFAULT_BATCH
    worlds: int = DEFAULT_WORLDS
    inference_mode: str = "local"

    def __post_init__(self) -> None:
        if min(self.depth, self.sims, self.batch, self.worlds) <= 0:
            raise ValueError("depth, sims, batch, and worlds must be positive.")
        if self.batch > self.sims:
            raise ValueError(
                f"batch {self.batch} must be <= sims {self.sims} (virtual-loss fidelity)."
            )
        if self.inference_mode not in ("local", "served"):
            raise ValueError("inference_mode must be 'local' or 'served'.")

    @property
    def config_id(self) -> str:
        return (
            f"d{self.depth}-s{self.sims}-b{self.batch}-w{self.worlds}-{self.inference_mode}"
        )

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResourceProfile:
    """Execution resources. Deliberately excluded from ``experiment_id``."""

    concurrency: int = 1
    torch_threads: int = 1
    cpu_per_game: str = "2"
    accelerator: str = "cpu"
    inference_endpoint: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MatrixManifest:
    """The frozen experiment contract (plan section 2) plus the lattice."""

    checkpoint_manifest: Mapping[str, Any]
    configs: Sequence[SearchConfig]
    resource_profile: ResourceProfile = field(default_factory=ResourceProfile)
    worlds: int = DEFAULT_WORLDS
    foulplay_rung_ms: int = DEFAULT_FOULPLAY_RUNG_MS
    max_decision_rounds: int = DEFAULT_MAX_DECISION_ROUNDS
    seed_pairs: int = DEFAULT_SEED_PAIRS
    spare_pairs: int = DEFAULT_SPARE_PAIRS
    seed_band: str = "default"
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
    bootstrap_seed: int = 20260723
    decision_wall_gate_s: float = 15.0
    corpus_decisions: int = 256
    engine_revision: str | None = None
    showdown_revision: str | None = None
    foulplay_revision: str | None = None
    image_revision: str | None = None
    schema_version: str = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        ids = [config.config_id for config in self.configs]
        duplicates = {value for value in ids if ids.count(value) > 1}
        if duplicates:
            raise ValueError(f"duplicate config_id in matrix: {sorted(duplicates)}")

    @property
    def frozen_contract(self) -> dict[str, Any]:
        """Everything that defines WHAT is being measured — no resources."""
        return {
            "schema_version": self.schema_version,
            "checkpoint": dict(self.checkpoint_manifest),
            "worlds": self.worlds,
            "foulplay_rung_ms": self.foulplay_rung_ms,
            "max_decision_rounds": self.max_decision_rounds,
            "seed_pairs": self.seed_pairs,
            "spare_pairs": self.spare_pairs,
            "seed_band": self.seed_band,
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_seed": self.bootstrap_seed,
            "decision_wall_gate_s": self.decision_wall_gate_s,
            "corpus_decisions": self.corpus_decisions,
            "engine_revision": self.engine_revision,
            "showdown_revision": self.showdown_revision,
            "foulplay_revision": self.foulplay_revision,
            "image_revision": self.image_revision,
        }

    @property
    def matrix_payload(self) -> list[dict[str, Any]]:
        return [config.to_payload() for config in self.configs]

    @property
    def experiment_id(self) -> str:
        return canonical_hash(
            {"contract": self.frozen_contract, "matrix": self.matrix_payload}
        )

    @property
    def execution_id(self) -> str:
        return canonical_hash(
            {
                "experiment_id": self.experiment_id,
                "resource_profile": self.resource_profile.to_payload(),
            }
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "execution_id": self.execution_id,
            "frozen_contract": self.frozen_contract,
            "matrix": self.matrix_payload,
            "resource_profile": self.resource_profile.to_payload(),
        }

    def with_resource_profile(self, profile: ResourceProfile) -> "MatrixManifest":
        """Reducing concurrency creates a new EXECUTION under the same experiment."""
        return MatrixManifest(
            checkpoint_manifest=self.checkpoint_manifest,
            configs=self.configs,
            resource_profile=profile,
            worlds=self.worlds,
            foulplay_rung_ms=self.foulplay_rung_ms,
            max_decision_rounds=self.max_decision_rounds,
            seed_pairs=self.seed_pairs,
            spare_pairs=self.spare_pairs,
            seed_band=self.seed_band,
            bootstrap_resamples=self.bootstrap_resamples,
            bootstrap_seed=self.bootstrap_seed,
            decision_wall_gate_s=self.decision_wall_gate_s,
            corpus_decisions=self.corpus_decisions,
            engine_revision=self.engine_revision,
            showdown_revision=self.showdown_revision,
            foulplay_revision=self.foulplay_revision,
            image_revision=self.image_revision,
            schema_version=self.schema_version,
        )


def default_lattice(
    depths: Iterable[int] = DEFAULT_DEPTHS,
    sims: Iterable[int] = DEFAULT_SIMS,
    *,
    batch: int = DEFAULT_BATCH,
    worlds: int = DEFAULT_WORLDS,
    inference_mode: str = "local",
) -> tuple[SearchConfig, ...]:
    """The A3 lattice in a deterministic order (depth-major, then sims)."""
    return tuple(
        SearchConfig(depth=depth, sims=sim, batch=batch, worlds=worlds, inference_mode=inference_mode)
        for depth in sorted(set(depths))
        for sim in sorted(set(sims))
    )
