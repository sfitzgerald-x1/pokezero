"""Checkpoint-parameterized harness for the MCTS depth/throughput/strength study.

Implements the public-repository deliverables of
``docs/mcts_depth_strength_eval_plan.md``. Nothing here hard-codes a lineage,
architecture, observation width, history budget, policy id, or storage path:
the checkpoint reference is data, and every contract is derived from the
checkpoint's own stamped model configuration (plan section 2.1).
"""

from .controller import (
    Stage,
    RetryableFailure,
    TerminalFailure,
    next_incomplete_stage,
    run_pipeline,
)
from .manifest import (
    MatrixManifest,
    ResourceProfile,
    SearchConfig,
    default_lattice,
)
from .frontier import (
    FrontierRow,
    build_frontier,
    render_markdown,
)
from .resolver import (
    CheckpointContract,
    ContractError,
    export_reuse_key,
    resolve_checkpoint_contract,
    sha256_file,
)
from .report import (
    StrengthRow,
    TimingRow,
    pareto_frontier,
    render_markdown_table,
    render_report,
    select_candidates,
)
from .scoring import (
    GameResult,
    Interval,
    MergeError,
    bootstrap_indices,
    bootstrap_mean,
    bootstrap_paired_delta,
    pair_scores,
    parity_label,
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
    "GameResult",
    "Interval",
    "MergeError",
    "RetryableFailure",
    "Stage",
    "StrengthRow",
    "TerminalFailure",
    "TimingRow",
    "bootstrap_indices",
    "bootstrap_mean",
    "bootstrap_paired_delta",
    "next_incomplete_stage",
    "pair_scores",
    "pareto_frontier",
    "parity_label",
    "render_markdown_table",
    "render_report",
    "run_pipeline",
    "select_candidates",
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
