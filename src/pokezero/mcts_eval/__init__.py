"""Checkpoint-parameterized harness for the MCTS depth/throughput/strength study.

Implements the public-repository deliverables of
``docs/mcts_depth_strength_eval_plan.md``. Nothing here hard-codes a lineage,
architecture, observation width, history budget, policy id, or storage path:
the checkpoint reference is data, and every contract is derived from the
checkpoint's own stamped model configuration (plan section 2.1).
"""

from .manifest import (
    MatrixManifest,
    ResourceProfile,
    SearchConfig,
    default_lattice,
)
from .resolver import (
    CheckpointContract,
    ContractError,
    export_reuse_key,
    resolve_checkpoint_contract,
    sha256_file,
)
from .timing_corpus import (
    CorpusError,
    TimingCorpusManifest,
    TimingDecisionRecord,
    build_corpus,
    label_strata,
    read_corpus,
    write_corpus,
)

__all__ = [
    "CheckpointContract",
    "ContractError",
    "CorpusError",
    "MatrixManifest",
    "ResourceProfile",
    "SearchConfig",
    "TimingCorpusManifest",
    "TimingDecisionRecord",
    "build_corpus",
    "default_lattice",
    "export_reuse_key",
    "label_strata",
    "read_corpus",
    "resolve_checkpoint_contract",
    "sha256_file",
    "write_corpus",
]
