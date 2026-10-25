"""Immutable teacher exports for optional, label-only self-play collection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import json
import os
from pathlib import Path

from .search_labels import with_search_labels


@dataclass(frozen=True)
class SearchLabelCollectionConfig:
    showdown_root: Path
    fraction: float = 0.1
    refresh_iterations: int = 25
    device: str = "cpu"

    def __post_init__(self):
        object.__setattr__(self, "showdown_root", Path(self.showdown_root).expanduser().resolve())
        if not math.isfinite(self.fraction) or not 0 < self.fraction <= 1:
            raise ValueError("active search label fraction must be in (0, 1]")
        if type(self.refresh_iterations) is not int or self.refresh_iterations <= 0:
            raise ValueError("search teacher refresh interval must be positive")
        if self.device not in ("cpu", "cuda", "mps"):
            raise ValueError("search teacher export device must be cpu, cuda or mps")

    def to_dict(self):
        return {**asdict(self), "showdown_root": str(self.showdown_root)}


@dataclass(frozen=True)
class SearchLabelTeacher:
    settings: SearchLabelCollectionConfig
    search_config: object
    identity: dict

    def make_policy(self, *, env):
        from .dex import load_showdown_dex_cached
        from .randbat import load_gen3_randbat_source_cached
        from .engine_search import EngineMctsPolicy, EnvTier2AnnotationSource

        self.validate_artifacts()
        # Mandatory on every real teacher. Recording targets alone does not
        # establish a capture scope or admit previously seen refusal classes.
        return EngineMctsPolicy(
            dex=load_showdown_dex_cached(self.settings.showdown_root),
            set_source=load_gen3_randbat_source_cached(self.settings.showdown_root),
            config=self.search_config, scientific_capture=True,
            annotation_source=EnvTier2AnnotationSource(env),
        )

    def wrap(self, actor, env):
        return with_search_labels(actor, fraction=self.settings.fraction,
                                  teacher_factory=lambda: self.make_policy(env=env),
                                  teacher_identity=self.identity)

    def validate_artifacts(self):
        from .failure_capture import configured_quarantine
        from .mcts_eval.resolver import sha256_file
        from .randbat import load_gen3_randbat_source_cached
        config = self.search_config
        if self.identity["capture_build"] != configured_quarantine(required=True).build:
            raise ValueError("search label teacher build differs from admitted worker build")
        if self.settings.to_dict() != self.identity["collection"] or asdict(config) != self.identity["search_config"]:
            raise ValueError("search label teacher configuration identity drift")
        expected = _teacher_search_config(self.settings, config.checkpoint_path, config.model_path, config.tables_path)
        if asdict(config) != asdict(expected) or config.checkpoint_path != self.identity["checkpoint_path"]:
            raise ValueError("search label configuration differs from the frozen teacher")
        if load_gen3_randbat_source_cached(self.settings.showdown_root).metadata.source_hash != self.identity["belief_source_sha256"]:
            raise ValueError("search label belief source provenance drift")
        for name, target in (("checkpoint", config.checkpoint_path), ("model", config.model_path),
                             ("tables", config.tables_path)):
            if sha256_file(target) != self.identity[name + "_sha256"]:
                raise ValueError(f"search label {name} provenance drift")

    @classmethod
    def from_manifest(cls, path):
        """Load the same immutable teacher in an external collection worker."""
        from .engine_search import EngineMctsConfig
        from .mcts_eval.resolver import resolve_checkpoint_contract, validate_encoder_tables

        identity = json.loads(Path(path).read_text())
        if identity.get("schema") != "pokezero.search-label-teacher.v1":
            raise ValueError("unsupported search label teacher manifest")
        settings = SearchLabelCollectionConfig(**identity["collection"])
        quarantine = _require_capture()
        if identity["capture_build"] != quarantine.build:
            raise ValueError("search label teacher build differs from admitted worker build")
        config = EngineMctsConfig(**identity["search_config"])
        teacher = cls(settings, config, identity)
        teacher.validate_artifacts()
        contract = resolve_checkpoint_contract(config.checkpoint_path,
            expected_sha256=identity["checkpoint_sha256"], model_device=settings.device,
            showdown_root=settings.showdown_root)
        validate_encoder_tables(contract, config.tables_path)
        return teacher


def _require_capture():
    from .failure_capture import configured_quarantine, require_capture_admission
    quarantine = configured_quarantine(required=True)
    require_capture_admission(os.environ.get("POKEZERO_CAPTURE_ADMISSION"), build=quarantine.build,
                              scientific_path="engine-mcts", capture_directory=quarantine.directory)
    return quarantine


def prepare_search_label_teacher(settings, checkpoint, *, iteration):
    """Export/validate outside collection and freeze identity until the next refresh.

    Reuses the existing checkpoint-aware exporter and contract validator. No
    model is trained here; exports remain beside their immutable checkpoint.
    """
    from .randbat import load_gen3_randbat_source_cached
    from .mcts_eval.resolver import resolve_checkpoint_contract, sha256_file
    from .mcts_eval.lattice import materialize_search_artifacts

    checkpoint = Path(checkpoint).expanduser().resolve()
    quarantine = _require_capture()
    source = load_gen3_randbat_source_cached(settings.showdown_root)
    contract = resolve_checkpoint_contract(checkpoint, model_device=settings.device,
        showdown_root=settings.showdown_root, showdown_source_sha256=source.metadata.source_hash)
    artifacts = materialize_search_artifacts(contract, showdown_root=str(settings.showdown_root))
    config = _teacher_search_config(settings, checkpoint, artifacts["model_path"], artifacts["tables_path"])
    identity = {"schema": "pokezero.search-label-teacher.v1", "collection": settings.to_dict(),
                "checkpoint_path": str(checkpoint), "checkpoint_sha256": contract.checkpoint_sha256,
                "model_sha256": sha256_file(artifacts["model_path"]),
                "tables_sha256": sha256_file(artifacts["tables_path"]),
                "refresh_iteration": iteration, "capture_build": quarantine.build,
                "belief_source_sha256": source.metadata.source_hash,
                "search_config": asdict(config)}
    return SearchLabelTeacher(settings, config, identity)


def _teacher_search_config(settings, checkpoint, model, tables):
    from .engine_search import EngineMctsConfig
    return EngineMctsConfig(
        leaf_eval="model", checkpoint_path=str(checkpoint), model_path=str(model),
        tables_path=str(tables), model_device=settings.device,
        worlds=1, model_world_workers=1, search_depth=8, search_sims=2048, search_batch=64,
        c_puct=1.4, fpu_reduction=None, model_priors=True, use_opponent_priors=True,
        early_stop=False, strict_fallbacks=False, record_training_targets=True,
    )
