"""Normalize full-game search artifacts into strict capstone evidence.

The benchmark runners intentionally retain per-seed outcomes and search timing,
but their wire formats differ between local Showdown and controlled FoulPlay.
This module is the deployment-neutral boundary that turns either format into
one candidate arm and its same-seed raw control.  It rejects partial, fallback,
or privileged-mode artifacts rather than silently producing a strength row.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .capstone_analysis import (
    CapstoneArmEvidence,
    CapstoneGameOutcome,
    CapstonePairedArmEvidence,
    analyze_paired_capstone_arms,
)


CAPSTONE_PAIR_ARTIFACT_SCHEMA_VERSION = "pokezero.capstone-pair-artifact.v1"


@dataclass(frozen=True)
class NormalizedCapstonePair:
    """A candidate arm and raw control recovered from one benchmark artifact."""

    opponent_id: str
    arm_id: str
    band: str
    seat: str
    source_kind: str
    raw: CapstoneArmEvidence
    candidate: CapstoneArmEvidence
    value_leaf_provenance: Mapping[str, Any] | None
    candidate_wall_seconds: tuple[float, ...]
    source_path: str | None = None
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.opponent_id.strip():
            raise ValueError("opponent_id must be non-empty.")
        if not self.arm_id.strip():
            raise ValueError("arm_id must be non-empty.")
        if not self.band.strip():
            raise ValueError("band must be non-empty.")
        if self.seat not in {"p1", "p2"}:
            raise ValueError("seat must be p1 or p2.")
        if self.raw.arm_id != "raw":
            raise ValueError("normalized raw control must use arm_id 'raw'.")
        if self.candidate.arm_id != self.arm_id:
            raise ValueError("normalized candidate arm_id must match arm_id.")
        if self.raw.outcomes != tuple(sorted(self.raw.outcomes, key=lambda item: item.key)):
            raise ValueError("normalized raw outcomes must be sorted by capstone key.")
        if self.candidate.outcomes != tuple(sorted(self.candidate.outcomes, key=lambda item: item.key)):
            raise ValueError("normalized candidate outcomes must be sorted by capstone key.")
        if tuple(item.key for item in self.raw.outcomes) != tuple(item.key for item in self.candidate.outcomes):
            raise ValueError("normalized raw and candidate outcomes must share identical keys.")
        for outcome in (*self.raw.outcomes, *self.candidate.outcomes):
            if outcome.band != self.band or outcome.seat != self.seat:
                raise ValueError("normalized outcome band/seat must match its pair metadata.")
        for value in self.candidate_wall_seconds:
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("candidate_wall_seconds must contain finite non-negative values.")
        if (self.source_path is None) != (self.source_sha256 is None):
            raise ValueError("source_path and source_sha256 must be supplied together.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CAPSTONE_PAIR_ARTIFACT_SCHEMA_VERSION,
            "opponent_id": self.opponent_id,
            "arm_id": self.arm_id,
            "band": self.band,
            "seat": self.seat,
            "source_kind": self.source_kind,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "value_leaf_provenance": dict(self.value_leaf_provenance or {}),
            "raw": _arm_to_dict(self.raw),
            "candidate": _arm_to_dict(self.candidate),
            "candidate_wall_seconds": list(self.candidate_wall_seconds),
        }


def normalize_root_puct_play_artifact(
    payload: Mapping[str, Any],
    *,
    opponent_id: str,
    arm_id: str,
    band: str,
    seat: str,
    uses_value_leaves: bool = True,
    source_path: Path | None = None,
) -> NormalizedCapstonePair:
    """Recover one seat's raw/search pair from ``root-puct-play-benchmark`` JSON."""

    _require_seat(seat)
    if payload.get("strength_evidence_eligible") is False:
        raise ValueError("mechanics-only root-PUCT artifacts cannot be used for a strength capstone.")
    if payload.get("root_dirichlet") is not None:
        raise ValueError("primary capstone artifacts must use deterministic root priors.")
    matchups = _mappings(payload.get("matchups"), field="matchups")
    requires_time_budget_diagnostics = payload.get("root_time_budget_ms") is not None
    candidate_matchup = _select_root_puct_matchup(matchups, opponent_id=opponent_id, seat=seat)
    candidate_policy_id = _seat_policy_id(candidate_matchup, seat)
    raw_policy_id = candidate_policy_id.split("+root-puct", 1)[0]
    raw_matchup = _select_raw_matchup(
        matchups,
        opponent_id=opponent_id,
        seat=seat,
        raw_policy_id=raw_policy_id,
    )
    value_leaf = _value_leaf_provenance(payload, required=uses_value_leaves)
    _require_root_policy_lineage(
        raw_matchup=raw_matchup,
        seat=seat,
        value_leaf=value_leaf,
        required=uses_value_leaves,
    )
    candidate = _benchmark_matchup_arm(
        candidate_matchup,
        arm_id=arm_id,
        band=band,
        seat=seat,
        uses_value_leaves=uses_value_leaves,
        calibrated_value_copy=_value_leaf_label(value_leaf) if uses_value_leaves else None,
        require_search=True,
        require_time_budget_diagnostics=requires_time_budget_diagnostics,
    )
    raw = _benchmark_matchup_arm(
        raw_matchup,
        arm_id="raw",
        band=band,
        seat=seat,
        uses_value_leaves=False,
        calibrated_value_copy=None,
        require_search=False,
        require_time_budget_diagnostics=False,
    )
    return NormalizedCapstonePair(
        opponent_id=opponent_id,
        arm_id=arm_id,
        band=band,
        seat=seat,
        source_kind="root-puct-play-benchmark",
        raw=raw,
        candidate=candidate,
        value_leaf_provenance=value_leaf,
        candidate_wall_seconds=_benchmark_wall_seconds(candidate_matchup, seat=seat),
        source_path=str(source_path.resolve(strict=False)) if source_path is not None else None,
        source_sha256=_sha256_file(source_path) if source_path is not None else None,
    )


def normalize_controlled_foulplay_artifact(
    payload: Mapping[str, Any],
    *,
    arm_id: str,
    band: str,
    seat: str,
    uses_value_leaves: bool = True,
    source_path: Path | None = None,
) -> NormalizedCapstonePair:
    """Recover one seat's raw/search pair from controlled FoulPlay comparison JSON."""

    _require_seat(seat)
    if payload.get("comparison_mode") != "per-seed":
        raise ValueError("primary controlled FoulPlay capstone artifacts require comparison_mode='per-seed'.")
    if payload.get("complete") is not True:
        raise ValueError("primary controlled FoulPlay capstone artifact is incomplete.")
    if payload.get("opponent_crashes"):
        raise ValueError("primary controlled FoulPlay capstone artifacts may not exclude crashed seeds.")
    runs = _mapping(payload.get("runs"), field="runs")
    raw_payload = _mapping(runs.get("raw"), field="runs.raw")
    candidate_payload = _mapping(runs.get("root_puct"), field="runs.root_puct")
    for label, result in (("raw", raw_payload), ("candidate", candidate_payload)):
        if result.get("complete") is not True:
            raise ValueError(f"controlled FoulPlay {label} result is incomplete.")
        if result.get("pokezero_player") != seat:
            raise ValueError(f"controlled FoulPlay {label} result does not match seat {seat!r}.")
    _require_matching_foulplay_schedules(payload, raw_payload=raw_payload, candidate_payload=candidate_payload)
    root_config = _mapping(candidate_payload.get("root_puct"), field="runs.root_puct.root_puct")
    if root_config.get("opponent_legal_mask_mode") != "hidden":
        raise ValueError("primary controlled FoulPlay artifacts must keep the opponent legal mask hidden.")
    if root_config.get("allow_search_fallback") is not False:
        raise ValueError("primary controlled FoulPlay artifacts must disable search fallback.")
    root_planners = _mapping(
        root_config.get("opponent_action_policies"),
        field="runs.root_puct.root_puct.opponent_action_policies",
    )
    _require_checkpoint_only_planners(root_planners, field="controlled FoulPlay aggregate opponent-action planner")
    if root_config.get("root_dirichlet_alpha") is not None:
        raise ValueError("primary controlled FoulPlay artifacts must use deterministic root priors.")
    value_leaf = _value_leaf_provenance(candidate_payload, required=uses_value_leaves)
    _require_foulplay_policy_lineage(raw_payload, candidate_payload=candidate_payload, value_leaf=value_leaf, required=uses_value_leaves)
    raw = _controlled_foulplay_arm(
        raw_payload,
        arm_id="raw",
        band=band,
        seat=seat,
        uses_value_leaves=False,
        calibrated_value_copy=None,
        require_search=False,
    )
    candidate = _controlled_foulplay_arm(
        candidate_payload,
        arm_id=arm_id,
        band=band,
        seat=seat,
        uses_value_leaves=uses_value_leaves,
        calibrated_value_copy=_value_leaf_label(value_leaf) if uses_value_leaves else None,
        require_search=True,
    )
    return NormalizedCapstonePair(
        opponent_id="foul-play",
        arm_id=arm_id,
        band=band,
        seat=seat,
        source_kind="controlled-foulplay-comparison",
        raw=raw,
        candidate=candidate,
        value_leaf_provenance=value_leaf,
        candidate_wall_seconds=_controlled_foulplay_wall_seconds(candidate_payload),
        source_path=str(source_path.resolve(strict=False)) if source_path is not None else None,
        source_sha256=_sha256_file(source_path) if source_path is not None else None,
    )


def analyze_normalized_capstone_pairs(
    pairs: Iterable[NormalizedCapstonePair],
    *,
    expected_keys: Iterable[tuple[str, int, str]],
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20260710,
) -> dict[str, Any]:
    """Analyze normalized artifacts for one opponent under the strict primary contract."""

    selected = tuple(pairs)
    if not selected:
        raise ValueError("at least one normalized capstone pair is required.")
    opponents = {pair.opponent_id for pair in selected}
    if len(opponents) != 1:
        raise ValueError("a capstone analysis may contain exactly one opponent_id.")
    arm_ids = {pair.arm_id for pair in selected}
    per_arm: list[CapstonePairedArmEvidence] = []
    timing_by_arm: dict[str, list[float]] = {arm_id: [] for arm_id in arm_ids}
    provenance_by_arm: dict[str, Mapping[str, Any] | None] = {}
    for arm_id in sorted(arm_ids):
        arm_pairs = tuple(pair for pair in selected if pair.arm_id == arm_id)
        raw_outcomes = _merge_outcomes((pair.raw for pair in arm_pairs), arm_id=f"{arm_id}:raw")
        candidate_outcomes = _merge_outcomes((pair.candidate for pair in arm_pairs), arm_id=arm_id)
        first_candidate = arm_pairs[0].candidate
        for pair in arm_pairs:
            if pair.candidate.calibrated_value_copy != first_candidate.calibrated_value_copy:
                raise ValueError(f"{arm_id}: calibrated value-copy labels disagree across artifacts.")
            if _canonical_json(pair.value_leaf_provenance) != _canonical_json(arm_pairs[0].value_leaf_provenance):
                raise ValueError(f"{arm_id}: value-leaf provenance disagrees across artifacts.")
            timing_by_arm[arm_id].extend(pair.candidate_wall_seconds)
        provenance_by_arm[arm_id] = arm_pairs[0].value_leaf_provenance
        per_arm.append(
            CapstonePairedArmEvidence(
                arm_id=arm_id,
                baseline=CapstoneArmEvidence(arm_id="raw", outcomes=raw_outcomes, uses_value_leaves=False),
                candidate=CapstoneArmEvidence(
                    arm_id=arm_id,
                    outcomes=candidate_outcomes,
                    uses_value_leaves=first_candidate.uses_value_leaves,
                    calibrated_value_copy=first_candidate.calibrated_value_copy,
                ),
            )
        )
    report = analyze_paired_capstone_arms(
        per_arm,
        expected_keys=expected_keys,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    report["opponent_id"] = next(iter(opponents))
    for row in _mappings(report["primary_arms"], field="primary_arms"):
        arm_id = _string(row.get("arm_id"), field="primary_arms.arm_id")
        row["candidate_wall_seconds"] = _wall_readout(timing_by_arm[arm_id])
        provenance = provenance_by_arm[arm_id]
        if provenance is not None:
            row["value_leaf_provenance"] = dict(provenance)
    return report


def normalized_pair_from_dict(payload: Mapping[str, Any]) -> NormalizedCapstonePair:
    """Load a JSON-ready normalized pair artifact without trusting its counters."""

    if payload.get("schema_version") != CAPSTONE_PAIR_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported normalized capstone pair artifact schema.")
    raw = _arm_from_dict(_mapping(payload.get("raw"), field="raw"))
    candidate = _arm_from_dict(_mapping(payload.get("candidate"), field="candidate"))
    wall = tuple(_finite_nonnegative(item, field="candidate_wall_seconds") for item in _sequence(payload.get("candidate_wall_seconds")))
    provenance = payload.get("value_leaf_provenance")
    if provenance is not None and not isinstance(provenance, Mapping):
        raise ValueError("value_leaf_provenance must be an object when present.")
    source_sha256 = payload.get("source_sha256")
    source_path = payload.get("source_path")
    if source_sha256 is not None and not isinstance(source_sha256, str):
        raise ValueError("source_sha256 must be a string when present.")
    if source_path is not None and not isinstance(source_path, str):
        raise ValueError("source_path must be a string when present.")
    return NormalizedCapstonePair(
        opponent_id=_string(payload.get("opponent_id"), field="opponent_id"),
        arm_id=_string(payload.get("arm_id"), field="arm_id"),
        band=_string(payload.get("band"), field="band"),
        seat=_string(payload.get("seat"), field="seat"),
        source_kind=_string(payload.get("source_kind"), field="source_kind"),
        raw=raw,
        candidate=candidate,
        value_leaf_provenance=dict(provenance) if isinstance(provenance, Mapping) else None,
        candidate_wall_seconds=wall,
        source_path=source_path,
        source_sha256=source_sha256,
    )


def load_normalized_pair(path: Path) -> NormalizedCapstonePair:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    pair = normalized_pair_from_dict(payload)
    if pair.source_path is None or pair.source_sha256 is None:
        raise ValueError("persisted normalized capstone artifacts require source path and sha256 provenance.")
    source_path = Path(pair.source_path)
    if not source_path.is_file() or _sha256_file(source_path) != pair.source_sha256:
        raise ValueError("normalized capstone artifact source hash does not match its persisted source file.")
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source_payload, Mapping):
        raise ValueError("normalized capstone artifact source must contain a JSON object.")
    if pair.source_kind == "root-puct-play-benchmark":
        reconstructed = normalize_root_puct_play_artifact(
            source_payload,
            opponent_id=pair.opponent_id,
            arm_id=pair.arm_id,
            band=pair.band,
            seat=pair.seat,
            uses_value_leaves=pair.candidate.uses_value_leaves,
            source_path=source_path,
        )
    elif pair.source_kind == "controlled-foulplay-comparison":
        reconstructed = normalize_controlled_foulplay_artifact(
            source_payload,
            arm_id=pair.arm_id,
            band=pair.band,
            seat=pair.seat,
            uses_value_leaves=pair.candidate.uses_value_leaves,
            source_path=source_path,
        )
    else:
        raise ValueError(f"unsupported normalized capstone artifact source_kind: {pair.source_kind!r}")
    if reconstructed.to_dict() != pair.to_dict():
        raise ValueError("persisted normalized capstone artifact does not match source-derived evidence.")
    return reconstructed


def _select_root_puct_matchup(
    matchups: Sequence[Mapping[str, Any]],
    *,
    opponent_id: str,
    seat: str,
) -> Mapping[str, Any]:
    candidates = [
        matchup
        for matchup in matchups
        if _other_policy_id(matchup, seat) == opponent_id and "+root-puct" in _seat_policy_id(matchup, seat)
    ]
    if len(candidates) != 1:
        raise ValueError("expected exactly one root-PUCT matchup for the requested opponent and seat.")
    return candidates[0]


def _select_raw_matchup(
    matchups: Sequence[Mapping[str, Any]],
    *,
    opponent_id: str,
    seat: str,
    raw_policy_id: str,
) -> Mapping[str, Any]:
    candidates = [
        matchup
        for matchup in matchups
        if _other_policy_id(matchup, seat) == opponent_id and _seat_policy_id(matchup, seat) == raw_policy_id
    ]
    if len(candidates) != 1:
        raise ValueError("expected exactly one raw matchup for the requested opponent and seat.")
    return candidates[0]


def _benchmark_matchup_arm(
    matchup: Mapping[str, Any],
    *,
    arm_id: str,
    band: str,
    seat: str,
    uses_value_leaves: bool,
    calibrated_value_copy: str | None,
    require_search: bool,
    require_time_budget_diagnostics: bool,
) -> CapstoneArmEvidence:
    outcomes: list[CapstoneGameOutcome] = []
    for game in _mappings(matchup.get("game_results"), field="matchup.game_results"):
        if require_search and game.get("opponent_legal_mask_mode") != "hidden":
            raise ValueError("candidate root-PUCT artifact exposes a privileged opponent legal mask.")
        diagnostics = _mapping_or_empty(_mapping_or_empty(game.get("root_puct_by_player")).get(seat))
        searches = _nonnegative_int(diagnostics.get("root_puct_searches", 0), field="root_puct_searches")
        fallbacks = _nonnegative_int(diagnostics.get("root_puct_fallbacks", 0), field="root_puct_fallbacks")
        if require_search and searches <= 0:
            raise ValueError("candidate root-PUCT artifact contains a game with no executed search.")
        if require_search:
            _require_root_primary_diagnostics(
                diagnostics,
                require_time_budget_diagnostics=require_time_budget_diagnostics,
            )
        outcomes.append(
            CapstoneGameOutcome(
                band=band,
                seat=seat,
                seed=_nonnegative_int(game.get("seed"), field="game.seed"),
                score=_score(game.get(f"{seat}_score"), field=f"game.{seat}_score"),
                tied=bool(game.get("tied", False)),
                capped=bool(game.get("capped", False)),
                root_puct_fallbacks=fallbacks,
                privileged_fallbacks=0,
            )
        )
    return CapstoneArmEvidence(
        arm_id=arm_id,
        outcomes=tuple(sorted(outcomes, key=lambda item: item.key)),
        uses_value_leaves=uses_value_leaves,
        calibrated_value_copy=calibrated_value_copy,
    )


def _controlled_foulplay_arm(
    payload: Mapping[str, Any],
    *,
    arm_id: str,
    band: str,
    seat: str,
    uses_value_leaves: bool,
    calibrated_value_copy: str | None,
    require_search: bool,
) -> CapstoneArmEvidence:
    outcomes: list[CapstoneGameOutcome] = []
    for game in _mappings(payload.get("game_results"), field="controlled.game_results"):
        searches = _nonnegative_int(game.get("root_puct_searches", 0), field="root_puct_searches")
        fallbacks = _nonnegative_int(game.get("root_puct_fallbacks", 0), field="root_puct_fallbacks")
        if require_search and searches <= 0:
            raise ValueError("candidate controlled FoulPlay artifact contains a game with no executed search.")
        if require_search and "root_puct_fallbacks" not in game:
            raise ValueError("candidate controlled FoulPlay artifact is missing fallback diagnostics.")
        if require_search:
            if "root_puct_opponent_action_policies" not in game:
                raise ValueError(
                    "controlled FoulPlay per-game opponent-action planner evidence is missing."
                )
            planners = _mapping(
                game.get("root_puct_opponent_action_policies"),
                field="root_puct_opponent_action_policies",
            )
            _require_checkpoint_only_planners(
                planners,
                field="controlled FoulPlay per-game opponent-action planner",
            )
        decision_players = tuple(_strings(game.get("pokezero_decision_players"), field="pokezero_decision_players"))
        submitted_players = tuple(_strings(game.get("pokezero_submitted_choice_players"), field="pokezero_submitted_choice_players"))
        if set(decision_players) != {seat} or set(submitted_players) != {seat}:
            raise ValueError("controlled FoulPlay artifact does not prove side-relative dispatch/submission.")
        outcomes.append(
            CapstoneGameOutcome(
                band=band,
                seat=seat,
                seed=_nonnegative_int(game.get("seed"), field="game.seed"),
                score=_score(game.get("pokezero_score"), field="game.pokezero_score"),
                tied=bool(game.get("tied", False)),
                capped=bool(game.get("capped", False)),
                root_puct_fallbacks=fallbacks,
                privileged_fallbacks=0,
            )
        )
    return CapstoneArmEvidence(
        arm_id=arm_id,
        outcomes=tuple(sorted(outcomes, key=lambda item: item.key)),
        uses_value_leaves=uses_value_leaves,
        calibrated_value_copy=calibrated_value_copy,
    )


def _benchmark_wall_seconds(matchup: Mapping[str, Any], *, seat: str) -> tuple[float, ...]:
    values: list[float] = []
    for game in _mappings(matchup.get("game_results"), field="matchup.game_results"):
        timing_by_player = _mapping(game.get("policy_elapsed_seconds_by_player"), field="game.policy_elapsed_seconds_by_player")
        values.extend(_finite_nonnegative(item, field="policy_elapsed_seconds") for item in _sequence(timing_by_player.get(seat)))
    if not values:
        raise ValueError("candidate root-PUCT artifact is missing full per-decision wall-time samples.")
    return tuple(values)


def _controlled_foulplay_wall_seconds(payload: Mapping[str, Any]) -> tuple[float, ...]:
    values: list[float] = []
    for game in _mappings(payload.get("game_results"), field="controlled.game_results"):
        values.extend(_finite_nonnegative(item, field="policy_elapsed_seconds") for item in _sequence(game.get("policy_elapsed_seconds")))
    if not values:
        raise ValueError("candidate controlled FoulPlay artifact is missing per-decision wall-time samples.")
    return tuple(values)


def _require_root_policy_lineage(
    *,
    raw_matchup: Mapping[str, Any],
    seat: str,
    value_leaf: Mapping[str, Any] | None,
    required: bool,
) -> None:
    raw_provenance = _mapping(
        raw_matchup.get(f"{seat}_policy_provenance"),
        field=f"raw_matchup.{seat}_policy_provenance",
    )
    raw_sha = _string(raw_provenance.get("weights_sha256"), field="raw_matchup policy weights_sha256")
    if required:
        assert value_leaf is not None
        if value_leaf["policy_checkpoint_sha256"] != raw_sha:
            raise ValueError("value-leaf policy hash does not match the root-PUCT raw policy hash.")


def _require_matching_foulplay_schedules(
    payload: Mapping[str, Any],
    *,
    raw_payload: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
) -> None:
    schedules = (
        _mapping(payload.get("foulplay_random_seed_schedule"), field="foulplay_random_seed_schedule"),
        _mapping(raw_payload.get("foulplay_random_seed_schedule"), field="runs.raw.foulplay_random_seed_schedule"),
        _mapping(
            candidate_payload.get("foulplay_random_seed_schedule"),
            field="runs.root_puct.foulplay_random_seed_schedule",
        ),
    )
    canonical = tuple(_canonical_json(schedule) for schedule in schedules)
    if len(set(canonical)) != 1:
        raise ValueError("controlled FoulPlay raw/candidate startup seed schedules do not match.")
    schedule = schedules[0]
    seeds = tuple(_sequence(schedule.get("seeds")))
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("controlled FoulPlay startup seed schedule is missing concrete per-game seeds.")
    if schedule.get("count") != len(seeds):
        raise ValueError("controlled FoulPlay startup seed schedule count does not match its seeds.")
    for label, run in (("raw", raw_payload), ("candidate", candidate_payload)):
        games = _mappings(run.get("game_results"), field=f"runs.{label}.game_results")
        if len(games) != len(seeds):
            raise ValueError(f"controlled FoulPlay {label} game count does not match its startup seed schedule.")
        battle_start = _nonnegative_int(run.get("seed_start"), field=f"runs.{label}.seed_start")
        foulplay_start = _nonnegative_int(run.get("foulplay_random_seed"), field=f"runs.{label}.foulplay_random_seed")
        expected_battle_seeds = tuple(range(battle_start, battle_start + len(games)))
        if tuple(_nonnegative_int(game.get("seed"), field=f"runs.{label}.game.seed") for game in games) != expected_battle_seeds:
            raise ValueError(f"controlled FoulPlay {label} game seeds are not the declared contiguous seed band.")
        if tuple(seeds) != tuple(range(foulplay_start, foulplay_start + len(games))):
            raise ValueError(f"controlled FoulPlay {label} startup seeds do not match its declared schedule.")


def _require_foulplay_policy_lineage(
    raw_payload: Mapping[str, Any],
    *,
    candidate_payload: Mapping[str, Any],
    value_leaf: Mapping[str, Any] | None,
    required: bool,
) -> None:
    raw_sha = _string(raw_payload.get("checkpoint_sha256"), field="runs.raw.checkpoint_sha256")
    candidate_sha = _string(candidate_payload.get("checkpoint_sha256"), field="runs.root_puct.checkpoint_sha256")
    if raw_sha != candidate_sha:
        raise ValueError("controlled FoulPlay raw and candidate checkpoint hashes do not match.")
    if required:
        assert value_leaf is not None
        if value_leaf["policy_checkpoint_sha256"] != raw_sha:
            raise ValueError("value-leaf policy hash does not match the controlled FoulPlay raw checkpoint hash.")


def _require_root_primary_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    require_time_budget_diagnostics: bool = False,
) -> None:
    # A missing key is not zero evidence. The local benchmark may only be used
    # for a hidden-information primary row when it records these facts.
    for field in ("root_puct_searches", "root_puct_fallbacks", "root_puct_opponent_action_policies"):
        if field not in diagnostics:
            raise ValueError(f"candidate root-PUCT artifact is missing {field} diagnostics.")
    if _nonnegative_int(diagnostics["root_puct_fallbacks"], field="root_puct_fallbacks") != 0:
        raise ValueError("candidate root-PUCT artifact used a search fallback.")
    planners = _mapping(diagnostics["root_puct_opponent_action_policies"], field="root_puct_opponent_action_policies")
    _require_checkpoint_only_planners(planners, field="candidate root-PUCT opponent-action planner")
    if require_time_budget_diagnostics:
        for field in ("root_puct_time_budget_checks", "root_puct_time_budget_exhaustions"):
            if field not in diagnostics:
                raise ValueError(f"candidate root-PUCT artifact is missing {field} diagnostics.")
        checks = _nonnegative_int(diagnostics["root_puct_time_budget_checks"], field="root_puct_time_budget_checks")
        exhaustions = _nonnegative_int(
            diagnostics["root_puct_time_budget_exhaustions"],
            field="root_puct_time_budget_exhaustions",
        )
        searches = _nonnegative_int(diagnostics["root_puct_searches"], field="root_puct_searches")
        if checks != searches or exhaustions > checks:
            raise ValueError("candidate root-PUCT artifact has invalid time-budget diagnostics.")
    leaf_policies = diagnostics.get("root_puct_leaf_rollout_opponent_policies")
    if leaf_policies is not None:
        for policy_id, count in _mapping(leaf_policies, field="root_puct_leaf_rollout_opponent_policies").items():
            if policy_id != "checkpoint":
                raise ValueError("candidate root-PUCT artifact used a privileged leaf rollout opponent policy.")
            if _nonnegative_int(count, field="root_puct_leaf_rollout_opponent_policies count") <= 0:
                raise ValueError("candidate root-PUCT leaf rollout opponent evidence has no visits.")


def _require_checkpoint_only_planners(planners: Mapping[str, Any], *, field: str) -> None:
    if not planners:
        raise ValueError(f"{field} evidence is missing.")
    for planner_id, count in planners.items():
        if not isinstance(planner_id, str) or not planner_id.startswith("checkpoint"):
            raise ValueError(f"privileged opponent-action planner in {field}.")
        if _nonnegative_int(count, field=f"{field} count") <= 0:
            raise ValueError(f"{field} evidence has no searches.")


def _value_leaf_provenance(payload: Mapping[str, Any], *, required: bool) -> Mapping[str, Any] | None:
    value = payload.get("value_leaf")
    if value is None:
        if required:
            raise ValueError("value-leaf capstone artifact is missing frozen calibration provenance.")
        return None
    provenance = _mapping(value, field="value_leaf")
    for field in (
        "policy_checkpoint_sha256",
        "value_checkpoint_sha256",
        "value_calibration_source_checkpoint_sha256",
    ):
        _string(provenance.get(field), field=f"value_leaf.{field}")
    transform = _mapping(provenance.get("value_calibration_transform"), field="value_leaf.value_calibration_transform")
    _string(transform.get("method"), field="value_leaf.value_calibration_transform.method")
    if provenance.get("model_config_match") is not True or provenance.get("belief_set_source_hash_match") is not True:
        raise ValueError("value-leaf provenance does not prove model/belief compatibility.")
    if provenance["policy_checkpoint_sha256"] != provenance["value_calibration_source_checkpoint_sha256"]:
        raise ValueError("value-leaf provenance parent hash does not match the raw policy checkpoint.")
    return dict(provenance)


def _value_leaf_label(provenance: Mapping[str, Any] | None) -> str:
    if provenance is None:
        raise ValueError("value-leaf provenance is required for a value-leaf label.")
    transform = _mapping(provenance.get("value_calibration_transform"), field="value_calibration_transform")
    method = _string(transform.get("method"), field="value_calibration_transform.method")
    checkpoint_sha = _string(provenance.get("value_checkpoint_sha256"), field="value_checkpoint_sha256")
    return f"{method}:{checkpoint_sha}"


def _merge_outcomes(arms: Iterable[CapstoneArmEvidence], *, arm_id: str) -> tuple[CapstoneGameOutcome, ...]:
    outcomes: list[CapstoneGameOutcome] = []
    for arm in arms:
        outcomes.extend(arm.outcomes)
    keys = [outcome.key for outcome in outcomes]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{arm_id}: duplicate normalized capstone outcome keys.")
    return tuple(sorted(outcomes, key=lambda item: item.key))


def _arm_to_dict(arm: CapstoneArmEvidence) -> dict[str, Any]:
    return {
        "arm_id": arm.arm_id,
        "uses_value_leaves": arm.uses_value_leaves,
        "calibrated_value_copy": arm.calibrated_value_copy,
        "outcomes": [
            {
                "band": outcome.band,
                "seat": outcome.seat,
                "seed": outcome.seed,
                "score": outcome.score,
                "tied": outcome.tied,
                "capped": outcome.capped,
                "root_puct_fallbacks": outcome.root_puct_fallbacks,
                "privileged_fallbacks": outcome.privileged_fallbacks,
            }
            for outcome in arm.outcomes
        ],
    }


def _arm_from_dict(payload: Mapping[str, Any]) -> CapstoneArmEvidence:
    outcomes = []
    for item in _mappings(payload.get("outcomes"), field="outcomes"):
        outcomes.append(
            CapstoneGameOutcome(
                band=_string(item.get("band"), field="outcome.band"),
                seat=_string(item.get("seat"), field="outcome.seat"),
                seed=_nonnegative_int(item.get("seed"), field="outcome.seed"),
                score=_score(item.get("score"), field="outcome.score"),
                tied=bool(item.get("tied", False)),
                capped=bool(item.get("capped", False)),
                root_puct_fallbacks=_nonnegative_int(item.get("root_puct_fallbacks", 0), field="outcome.root_puct_fallbacks"),
                privileged_fallbacks=_nonnegative_int(item.get("privileged_fallbacks", 0), field="outcome.privileged_fallbacks"),
            )
        )
    return CapstoneArmEvidence(
        arm_id=_string(payload.get("arm_id"), field="arm.arm_id"),
        outcomes=tuple(sorted(outcomes, key=lambda item: item.key)),
        uses_value_leaves=bool(payload.get("uses_value_leaves", False)),
        calibrated_value_copy=(
            _string(payload.get("calibrated_value_copy"), field="arm.calibrated_value_copy")
            if payload.get("calibrated_value_copy") is not None
            else None
        ),
    )


def _wall_readout(samples: Sequence[float]) -> dict[str, float | int]:
    if not samples:
        raise ValueError("candidate wall-time samples are required.")
    ordered = sorted(samples)
    return {
        "decision_samples": len(ordered),
        "mean_seconds": sum(ordered) / len(ordered),
        "p95_seconds": ordered[math.ceil(0.95 * len(ordered)) - 1],
    }


def _seat_policy_id(matchup: Mapping[str, Any], seat: str) -> str:
    return _string(matchup.get(f"{seat}_policy_id"), field=f"matchup.{seat}_policy_id")


def _other_policy_id(matchup: Mapping[str, Any], seat: str) -> str:
    return _seat_policy_id(matchup, "p2" if seat == "p1" else "p1")


def _require_seat(seat: str) -> None:
    if seat not in {"p1", "p2"}:
        raise ValueError("seat must be p1 or p2.")


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object.")
    return value


def _mapping_or_empty(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mappings(value: object, *, field: str) -> list[Mapping[str, Any]]:
    return [_mapping(item, field=field) for item in _sequence(value)]


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _strings(value: object, *, field: str) -> list[str]:
    return [_string(item, field=field) for item in _sequence(value)]


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer.")
    return value


def _finite_nonnegative(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{field} must be a finite non-negative number.")
    return float(value)


def _score(value: object, *, field: str) -> float:
    score = _finite_nonnegative(value, field=field)
    if score not in {0.0, 0.5, 1.0}:
        raise ValueError(f"{field} must be 0.0, 0.5, or 1.0.")
    return score


def _canonical_json(value: Mapping[str, Any] | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
