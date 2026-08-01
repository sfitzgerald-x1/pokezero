"""A3 timing-lattice execution (plan section 4).

End-to-end decision wall is defined by the plan as

    acting-player request available -> validated Showdown choice string ready

so the timer spans observation/legal-action construction, belief-world
construction, native search, root-action mapping, and choice serialization. It
EXCLUDES prefix replay (the corpus record's public prefix is replayed first to
warm the incremental fold) and network delivery, which the offline harness does
not have.

Early stop: after at least ``MIN_DECISIONS_BEFORE_EARLY_STOP`` decisions, a cell
whose running mean reaches the gate is persisted as ``gate_failed`` with its
partial sample and full telemetry. That is a RESULT, not a stage failure — the
plan is explicit that upper cells are allowed to fail the gate.
"""

from __future__ import annotations

from statistics import mean, median
import time
from typing import Any, Callable, Sequence

from .manifest import SearchConfig
from .report import (
    DECISION_WALL_GATE_S,
    MIN_CORPUS_DECISIONS_BEFORE_EARLY_STOP,
    TimingRow,
)
from ..observation import (
    OBSERVATION_SCHEMA_VERSION_V2_2,
    OBSERVATION_SCHEMA_VERSION_V3,
    OBSERVATION_SCHEMA_VERSION_V4,
)
from .resolver import CheckpointContract, ContractError
from .timing_corpus import TimingDecisionRecord


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def time_lattice_cell(
    config: SearchConfig,
    *,
    records: Sequence[TimingDecisionRecord],
    contract: CheckpointContract,
    showdown_root: str | None = None,
    decide: Callable[[TimingDecisionRecord, SearchConfig], dict[str, Any]] | None = None,
    gate_s: float = DECISION_WALL_GATE_S,
) -> TimingRow:
    """Time one lattice cell over the corpus.

    ``decide`` performs one decision and returns its telemetry; it is injected so
    the timing loop itself is testable without the native crate. The default
    builds the controlled engine-MCTS policy for this cell.
    """
    if decide is None:
        decide = _default_decider(contract, showdown_root)

    walls: list[float] = []
    depths: list[int] = []
    cap_hits = 0
    fallbacks = 0
    invalid_actions = 0
    encode_s = model_s = tree_s = 0.0
    root_actions: list[str] = []
    gate_failed = False

    for index, record in enumerate(records, start=1):
        started = time.perf_counter()
        telemetry = decide(record, config)
        walls.append(time.perf_counter() - started)

        realized = int(telemetry.get("max_depth_reached", 0))
        depths.append(realized)
        if realized >= config.depth - 1:
            cap_hits += 1
        fallbacks += int(telemetry.get("fallbacks", 0))
        invalid_actions += int(telemetry.get("invalid_actions", 0))
        encode_s += float(telemetry.get("encode_s", 0.0))
        model_s += float(telemetry.get("model_s", 0.0))
        tree_s += float(telemetry.get("tree_s", 0.0))
        root_actions.append(str(telemetry.get("root_action", "")))

        if index >= MIN_CORPUS_DECISIONS_BEFORE_EARLY_STOP and mean(walls) >= gate_s:
            # A result, not a harness failure: persist the partial sample.
            gate_failed = True
            break

    return TimingRow(
        config_id=config.config_id,
        depth=config.depth,
        sims=config.sims,
        decisions_timed=len(walls),
        mean_wall_s=mean(walls) if walls else 0.0,
        median_wall_s=median(walls) if walls else 0.0,
        p95_wall_s=_percentile(walls, 0.95),
        max_wall_s=max(walls) if walls else 0.0,
        realized_depth_mean=mean(depths) if depths else 0.0,
        realized_depth_max=max(depths) if depths else 0,
        cap_hit_rate=(cap_hits / len(depths)) if depths else 0.0,
        encode_s=encode_s,
        model_s=model_s,
        tree_s=tree_s,
        fallbacks=fallbacks,
        invalid_actions=invalid_actions,
        gate_failed=gate_failed,
        provenance_exact=True,
        root_argmax_by_decision=tuple(root_actions),
    )


def _default_decider(
    contract: CheckpointContract, showdown_root: str | None
) -> Callable[[TimingDecisionRecord, SearchConfig], dict[str, Any]]:
    """Live decider: controlled engine-MCTS over the checkpoint's own leaves.

    Artifacts (TorchScript trace + encoder tables) are materialized once, keyed
    by the contract, and the policy is rebuilt per cell because depth/sims are
    part of a cell's identity. The per-decision work is the plan's timed span:
    observation/legal-action construction is already in the corpus record, so
    this runs belief-world construction + native search + root-action mapping
    and returns the crate's own telemetry (never a derived phase).
    """
    from ..engine_search import EngineMctsConfig, EngineMctsPolicy

    artifacts = materialize_search_artifacts(contract, showdown_root=showdown_root)
    policies: dict[str, Any] = {}

    def decide(record: TimingDecisionRecord, config: SearchConfig) -> dict[str, Any]:
        raise NotImplementedError(
            "engine-MCTS timing needs the corpus record's event_prefix replayed into live "
            "env state before EngineMctsPolicy.select_action_with_context can be called: a "
            "TimingDecisionRecord is a public prefix + request-derived candidates, not an "
            "observation. Implement replay via LocalShowdownEnv (reset(seed=record.battle_seed) "
            "then advance through record.event_prefix, warming the incremental fold OUTSIDE "
            "the timed span per plan A2), then time select_action_with_context and read the "
            "crate telemetry from policy.stats. Artifacts are already materialized by "
            "materialize_search_artifacts(); pass decide=... to time a cell meanwhile."
        )

    return decide


def materialize_search_artifacts(
    contract: CheckpointContract, *, showdown_root: str | None
) -> dict[str, str]:
    """Export (once) the TorchScript trace + encoder tables this contract needs.

    Reuse is keyed by the contract's export key, so a different checkpoint,
    device, observation contract, Showdown source, or exporter revision cannot
    silently adopt someone else's artifact.
    """
    import subprocess
    import sys
    from pathlib import Path as _Path

    from .resolver import export_reuse_key, validate_encoder_tables

    key = export_reuse_key(contract)[:16]
    root = _Path(contract.checkpoint_path).parent / f".mcts-eval-artifacts-{key}"
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "model_ts.pt"
    tables_path = root / "encoder_tables.json"
    repo = _Path(__file__).resolve().parents[3]

    # Map the checkpoint's schema to the exporter's CLI choice, BEFORE the TorchScript export
    # below: an unmapped schema is a terminal contract error either way, and resolving it first
    # means it fails in milliseconds instead of after burning a full model trace.
    # A bare "everything else is v2.2" default silently exported the wrong tables for any newer
    # schema — v4 got past the resolver's schema gate and then died in validate_encoder_tables
    # on a v2.2-vs-v4 mismatch, making the gate a dead end. Raising closes the CLASS, not the
    # instance: otherwise the only thing standing between the next schema and that same dead
    # end is SUPPORTED_OBSERVATION_SCHEMAS upstream, an implicit coupling one edit away from
    # being wrong again.
    _EXPORTER_SCHEMA_CHOICES = {
        OBSERVATION_SCHEMA_VERSION_V4: "v4",
        OBSERVATION_SCHEMA_VERSION_V3: "v3",
        OBSERVATION_SCHEMA_VERSION_V2_2: "v2.2",
    }
    try:
        schema = _EXPORTER_SCHEMA_CHOICES[contract.schema_version]
    except KeyError:
        raise ContractError(
            f"no encoder-table exporter choice for observation schema "
            f"{contract.schema_version!r} (known: "
            f"{', '.join(sorted(_EXPORTER_SCHEMA_CHOICES))}). Add the mapping when "
            f"adding the schema; exporting v2.2 tables by default silently produces "
            f"the wrong string->row map."
        ) from None

    if not model_path.is_file():
        subprocess.run(
            # TorchScript traces bake device constants (see model.rs: "Artifacts
            # are PER-DEVICE"), so a CPU trace loaded on CUDA dies inside the
            # interpreter. Trace on the device the search will run on; the export
            # reuse key already includes model_device, so the cache stays correct.
            [sys.executable, str(repo / "scripts" / "export_model.py"),
             "--checkpoint", contract.checkpoint_path, "--out-dir", str(root),
             "--formats", "ts", "--device", contract.model_device],
            check=True,
        )
        produced = next(root.glob("*_ts.pt"), None) or next(root.glob("*.pt"), None)
        if produced and produced != model_path:
            produced.rename(model_path)
    if not tables_path.is_file():
        subprocess.run(
            # Always derive the layout from the checkpoint, never from the schema
            # default: a region-trimmed model has a narrower transition region, so
            # schema-default tables describe an observation it cannot consume.
            [sys.executable, str(repo / "scripts" / "export_encoder_tables.py"),
             "--showdown-root", showdown_root or "", "--observation-schema", schema,
             "--checkpoint", contract.checkpoint_path,
             "--out", str(tables_path)],
            check=True,
        )
    # Fail closed on root/leaf drift, against the checkpoint's own contract. A
    # REUSED artifact is covered because the contract's vocabulary is in the reuse
    # key, so tables exported against a different enumeration resolve to a
    # different path and can never be adopted here.
    validate_encoder_tables(contract, tables_path)
    return {"model_path": str(model_path), "tables_path": str(tables_path)}
