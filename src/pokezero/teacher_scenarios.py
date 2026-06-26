"""Deterministic scripted-teacher scenario preflight fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from .collection import RolloutRecord, write_rollout_record
from .env import TerminalState
from .observation import ObservationSpec, PokeZeroObservationV0
from .policy import Policy, PolicyDecision, ScriptedTeacherPolicy
from .trajectory import BattleTrajectory, TrajectoryStep


TEACHER_SCENARIO_PREFLIGHT_SCHEMA_VERSION = "pokezero.teacher_scenario_preflight.v1"
TEACHER_SCENARIO_ROLLOUT_SCHEMA_VERSION = "pokezero.teacher_scenario_rollouts.v1"


@dataclass(frozen=True)
class TeacherScenario:
    scenario_id: str
    description: str
    observation: PokeZeroObservationV0
    expected_action_index: int
    expected_action_family: str
    expected_teacher_branch: str
    expected_reason_contains: str | None = None


def default_teacher_scenarios() -> tuple[TeacherScenario, ...]:
    """Return curated fixture states for important scripted-teacher branches."""

    return (
        TeacherScenario(
            scenario_id="damaging-super-effective",
            description="prefers a super-effective Gen 3 physical Shadow Ball over neutral Fire damage",
            observation=_observation(
                (True, True, False, False, False, False, False, False, False),
                metadata={
                    "self_active": {"species": "Charizard", "hp_fraction": 1.0, "status": "none"},
                    "opponent_active": {"species": "Xatu", "hp_fraction": 1.0, "status": "none"},
                    "action_candidates": [
                        _move(0, "flamethrower", "Flamethrower"),
                        _move(1, "shadowball", "Shadow Ball"),
                    ],
                },
            ),
            expected_action_index=1,
            expected_action_family="move",
            expected_teacher_branch="damaging_move",
            expected_reason_contains="eff=2",
        ),
        TeacherScenario(
            scenario_id="status-no-effect-electric-immunity",
            description="recognizes Thunder Wave has no effect into Ground typing",
            observation=_observation(
                (True, False, False, False, False, False, False, False, False),
                metadata={
                    "self_active": {"species": "Xatu", "hp_fraction": 1.0, "status": "none"},
                    "opponent_active": {"species": "Golem", "hp_fraction": 1.0, "status": "none"},
                    "action_candidates": [_move(0, "thunderwave", "Thunder Wave")],
                },
            ),
            expected_action_index=0,
            expected_action_family="move",
            expected_teacher_branch="status_no_effect",
            expected_reason_contains="no effect",
        ),
        TeacherScenario(
            scenario_id="status-pressure-glare-ghost",
            description="keeps Glare as status pressure into Ghost typing in Gen 3",
            observation=_observation(
                (True, True, False, False, False, False, False, False, False),
                metadata={
                    "self_active": {"species": "Snorlax", "hp_fraction": 1.0, "status": "none"},
                    "opponent_active": {"species": "Dusclops", "hp_fraction": 1.0, "status": "none"},
                    "action_candidates": [
                        _move(0, "glare", "Glare"),
                        _move(1, "tackle", "Tackle"),
                    ],
                },
            ),
            expected_action_index=0,
            expected_action_family="move",
            expected_teacher_branch="status_pressure",
            expected_reason_contains="status pressure",
        ),
        TeacherScenario(
            scenario_id="damaging-no-effect-ghost-immunity",
            description="marks Normal damage as ineffective into Ghost typing",
            observation=_observation(
                (True, False, False, False, False, False, False, False, False),
                metadata={
                    "self_active": {"species": "Snorlax", "hp_fraction": 1.0, "status": "none"},
                    "opponent_active": {"species": "Dusclops", "hp_fraction": 1.0, "status": "none"},
                    "action_candidates": [_move(0, "tackle", "Tackle")],
                },
            ),
            expected_action_index=0,
            expected_action_family="move",
            expected_teacher_branch="damaging_no_effect",
            expected_reason_contains="no effect",
        ),
        TeacherScenario(
            scenario_id="team-status-cure",
            description="uses Heal Bell when a teammate has meaningful status",
            observation=_observation(
                (True, True, False, False, False, False, False, False, False),
                metadata={
                    "self_active": {"species": "Snorlax", "hp_fraction": 1.0, "status": "none"},
                    "self_team": [
                        {"species": "Snorlax", "hp_fraction": 1.0, "status": "none"},
                        {"species": "Starmie", "hp_fraction": 1.0, "status": "par"},
                    ],
                    "opponent_active": {"species": "Xatu", "hp_fraction": 1.0, "status": "none"},
                    "action_candidates": [
                        _move(0, "growl", "Growl"),
                        _move(1, "healbell", "Heal Bell"),
                    ],
                },
            ),
            expected_action_index=1,
            expected_action_family="move",
            expected_teacher_branch="team_status_cure",
            expected_reason_contains="team status cure",
        ),
        TeacherScenario(
            scenario_id="recovery-low-hp",
            description="uses recovery when the active Pokemon is below the recovery threshold",
            observation=_observation(
                (True, True, False, False, False, False, False, False, False),
                metadata={
                    "self_active": {"species": "Starmie", "hp_fraction": 0.3, "status": "none"},
                    "opponent_active": {"species": "Xatu", "hp_fraction": 1.0, "status": "none"},
                    "action_candidates": [
                        _move(0, "tackle", "Tackle"),
                        _move(1, "recover", "Recover"),
                    ],
                },
            ),
            expected_action_index=1,
            expected_action_family="move",
            expected_teacher_branch="recovery",
            expected_reason_contains="recovery",
        ),
        TeacherScenario(
            scenario_id="setup-healthy-active",
            description="uses setup when healthy and competing only with low-impact status",
            observation=_observation(
                (True, True, False, False, False, False, False, False, False),
                metadata={
                    "self_active": {"species": "Snorlax", "hp_fraction": 1.0, "status": "none"},
                    "opponent_active": {"species": "Xatu", "hp_fraction": 1.0, "status": "none"},
                    "action_candidates": [
                        _move(0, "growl", "Growl"),
                        _move(1, "swordsdance", "Swords Dance"),
                    ],
                },
            ),
            expected_action_index=1,
            expected_action_family="move",
            expected_teacher_branch="setup",
            expected_reason_contains="setup",
        ),
        TeacherScenario(
            scenario_id="rapid-spin-clear-hazards",
            description="uses Rapid Spin when own side has hazards and the opponent does not block it",
            observation=_observation(
                (True, True, False, False, False, False, False, False, False),
                metadata={
                    "self_active": {"species": "Starmie", "hp_fraction": 1.0, "status": "none"},
                    "self_side_conditions": ["spikes"],
                    "opponent_active": {"species": "Xatu", "hp_fraction": 1.0, "status": "none"},
                    "action_candidates": [
                        _move(0, "tackle", "Tackle"),
                        _move(1, "rapidspin", "Rapid Spin"),
                    ],
                },
            ),
            expected_action_index=1,
            expected_action_family="move",
            expected_teacher_branch="rapid_spin_clear_hazards",
            expected_reason_contains="clears hazards",
        ),
        TeacherScenario(
            scenario_id="rapid-spin-no-hazards-chip",
            description="treats Rapid Spin without hazards as ordinary chip damage",
            observation=_observation(
                (True, True, False, False, False, False, False, False, False),
                metadata={
                    "self_active": {"species": "Starmie", "hp_fraction": 1.0, "status": "none"},
                    "self_side_conditions": [],
                    "opponent_active": {"species": "Xatu", "hp_fraction": 1.0, "status": "none"},
                    # No hazards, so Rapid Spin (Gen 3: 20 BP) is scored as ordinary chip damage and
                    # chosen over the weaker Constrict (10 BP). The alternative must stay below Gen 3
                    # Rapid Spin's BP for the rapid_spin_no_hazards branch to be the chosen action.
                    "action_candidates": [
                        _move(0, "constrict", "Constrict"),
                        _move(1, "rapidspin", "Rapid Spin"),
                    ],
                },
            ),
            expected_action_index=1,
            expected_action_family="move",
            expected_teacher_branch="rapid_spin_no_hazards",
            expected_reason_contains="no side hazards",
        ),
        TeacherScenario(
            scenario_id="rapid-spin-blocked-by-ghost",
            description="marks Rapid Spin as blocked by a Ghost target",
            observation=_observation(
                (True, True, False, False, False, False, False, False, False),
                metadata={
                    "self_active": {"species": "Starmie", "hp_fraction": 1.0, "status": "none"},
                    "self_side_conditions": ["spikes"],
                    "opponent_active": {"species": "Dusclops", "hp_fraction": 1.0, "status": "none"},
                    "action_candidates": [
                        _move(0, "rapidspin", "Rapid Spin"),
                        _move(1, "tackle", "Tackle"),
                    ],
                },
            ),
            expected_action_index=0,
            expected_action_family="move",
            expected_teacher_branch="rapid_spin_blocked_by_ghost",
            expected_reason_contains="blocked by Ghost",
        ),
        TeacherScenario(
            scenario_id="spikes-available",
            description="sets Spikes when layers remain available",
            observation=_observation(
                (True, True, False, False, False, False, False, False, False),
                metadata={
                    "self_active": {"species": "Snorlax", "hp_fraction": 1.0, "status": "none"},
                    "opponent_active": {"species": "Xatu", "hp_fraction": 1.0, "status": "none"},
                    "opponent_side_conditions": [],
                    "opponent_side_condition_counts": {},
                    "action_candidates": [
                        _move(0, "tackle", "Tackle"),
                        _move(1, "spikes", "Spikes"),
                    ],
                },
            ),
            expected_action_index=1,
            expected_action_family="move",
            expected_teacher_branch="spikes_available",
            expected_reason_contains="layers=0/3",
        ),
        TeacherScenario(
            scenario_id="spikes-maxed",
            description="recognizes Spikes are already maxed",
            observation=_observation(
                (True, True, False, False, False, False, False, False, False),
                metadata={
                    "self_active": {"species": "Snorlax", "hp_fraction": 1.0, "status": "none"},
                    "opponent_active": {"species": "Dusclops", "hp_fraction": 1.0, "status": "none"},
                    "opponent_side_conditions": ["spikes"],
                    "opponent_side_condition_counts": {"spikes": 3},
                    "action_candidates": [
                        _move(0, "spikes", "Spikes"),
                        _move(1, "tackle", "Tackle"),
                    ],
                },
            ),
            expected_action_index=0,
            expected_action_family="move",
            expected_teacher_branch="spikes_maxed",
            expected_reason_contains="already maxed",
        ),
        TeacherScenario(
            scenario_id="low-hp-preservation-switch",
            description="switches to a healthier safer target when the active Pokemon is critically low",
            observation=_observation(
                (True, False, False, False, True, False, False, False, False),
                metadata={
                    "self_active": {"species": "Charizard", "hp_fraction": 0.1, "status": "none"},
                    "opponent_active": {"species": "Golem", "hp_fraction": 1.0, "status": "none"},
                    "action_candidates": [
                        _move(0, "tackle", "Tackle"),
                        _switch(4, "Starmie", hp_fraction=1.0),
                    ],
                },
            ),
            expected_action_index=4,
            expected_action_family="switch",
            expected_teacher_branch="switch",
            expected_reason_contains="preserve=",
        ),
    )


def teacher_scenario_ids() -> tuple[str, ...]:
    return tuple(scenario.scenario_id for scenario in default_teacher_scenarios())


def run_teacher_scenario_preflight(
    *,
    policy: Policy | None = None,
    scenario_ids: Sequence[str] | None = None,
    rng_seed: int = 1,
) -> dict[str, Any]:
    selected_scenarios = _select_scenarios(scenario_ids)
    teacher = policy if policy is not None else ScriptedTeacherPolicy()
    scenario_results = [
        _run_scenario(teacher, scenario, rng_seed=rng_seed + index)
        for index, scenario in enumerate(selected_scenarios)
    ]
    failed = [result for result in scenario_results if not result["passed"]]
    branch_counts: dict[str, int] = {}
    for result in scenario_results:
        observed = result.get("observed")
        branch = observed.get("teacher_branch") if isinstance(observed, Mapping) else None
        if isinstance(branch, str) and branch:
            branch_counts[branch] = branch_counts.get(branch, 0) + 1
    return {
        "schema_version": TEACHER_SCENARIO_PREFLIGHT_SCHEMA_VERSION,
        "passed": not failed,
        "scenario_count": len(scenario_results),
        "passed_count": len(scenario_results) - len(failed),
        "failed_count": len(failed),
        "teacher_branch_counts": dict(sorted(branch_counts.items())),
        "scenarios": scenario_results,
    }


def build_teacher_scenario_rollout_records(
    *,
    policy: Policy | None = None,
    scenario_ids: Sequence[str] | None = None,
    rng_seed: int = 1,
    seed_start: int = 4_000_000,
    repeat: int = 1,
    format_id: str = "gen3randombattle",
) -> tuple[RolloutRecord, ...]:
    """Build one-step rollout records from deterministic teacher scenarios."""

    if repeat <= 0:
        raise ValueError("repeat must be positive.")
    if not format_id.strip():
        raise ValueError("format_id must be non-empty.")
    selected_scenarios = _select_scenarios(scenario_ids)
    teacher = policy if policy is not None else ScriptedTeacherPolicy()
    records: list[RolloutRecord] = []
    for repeat_index in range(repeat):
        for scenario_index, scenario in enumerate(selected_scenarios):
            seed_offset = repeat_index * len(selected_scenarios) + scenario_index
            seed = seed_start + seed_offset
            records.append(
                _scenario_rollout_record(
                    teacher,
                    scenario,
                    seed=seed,
                    rng_seed=rng_seed + seed_offset,
                    format_id=format_id,
                    repeat_index=repeat_index,
                )
            )
    return tuple(records)


def write_teacher_scenario_rollouts(
    output_path: Path,
    *,
    policy: Policy | None = None,
    scenario_ids: Sequence[str] | None = None,
    rng_seed: int = 1,
    seed_start: int = 4_000_000,
    repeat: int = 1,
    format_id: str = "gen3randombattle",
    append: bool = False,
) -> dict[str, Any]:
    """Write deterministic teacher scenario demonstrations as rollout JSONL."""

    selected_scenarios = _select_scenarios(scenario_ids)
    records = build_teacher_scenario_rollout_records(
        policy=policy,
        scenario_ids=tuple(scenario.scenario_id for scenario in selected_scenarios),
        rng_seed=rng_seed,
        seed_start=seed_start,
        repeat=repeat,
        format_id=format_id,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a" if append else "w", encoding="utf-8") as handle:
        for record in records:
            write_rollout_record(handle, record)
    return {
        "schema_version": TEACHER_SCENARIO_ROLLOUT_SCHEMA_VERSION,
        "path": str(output_path),
        "record_count": len(records),
        "repeat": repeat,
        "seed_start": seed_start,
        "rng_seed": rng_seed,
        "format_id": format_id,
        "scenario_ids": [scenario.scenario_id for scenario in selected_scenarios],
        "teacher_branch_counts": _teacher_branch_counts(records),
    }


def _select_scenarios(scenario_ids: Sequence[str] | None) -> tuple[TeacherScenario, ...]:
    scenarios = {scenario.scenario_id: scenario for scenario in default_teacher_scenarios()}
    if not scenario_ids:
        return tuple(scenarios.values())
    selected: list[TeacherScenario] = []
    unknown: list[str] = []
    for scenario_id in scenario_ids:
        key = str(scenario_id).strip()
        if not key:
            unknown.append(str(scenario_id))
            continue
        scenario = scenarios.get(key)
        if scenario is None:
            unknown.append(key)
            continue
        selected.append(scenario)
    if unknown:
        known = ", ".join(sorted(scenarios))
        raise ValueError(f"unknown teacher scenario(s): {', '.join(unknown)}. Known scenarios: {known}")
    return tuple(selected)


def _scenario_rollout_record(
    policy: Policy,
    scenario: TeacherScenario,
    *,
    seed: int,
    rng_seed: int,
    format_id: str,
    repeat_index: int,
) -> RolloutRecord:
    decision = policy.select_action(scenario.observation, rng=random.Random(rng_seed))
    _require_expected_scenario_decision(scenario, decision)
    metadata = {
        "source": "teacher_scenario_demo",
        "scenario_id": scenario.scenario_id,
        "scenario_description": scenario.description,
        "scenario_repeat_index": repeat_index,
        "policy_id": decision.policy_id,
        **dict(decision.metadata),
    }
    trajectory = BattleTrajectory(
        battle_id=f"teacher-scenario-{scenario.scenario_id}-{seed}",
        format_id=format_id,
        seed=seed,
        metadata={
            "source": "teacher_scenario_demo",
            "scenario_id": scenario.scenario_id,
            "scenario_description": scenario.description,
            "scenario_repeat_index": repeat_index,
        },
    )
    trajectory.append(
        TrajectoryStep(
            player_id="p1",
            turn_index=0,
            observation=scenario.observation,
            legal_action_mask=tuple(scenario.observation.legal_action_mask),
            action_index=decision.action_index,
            reward=0.0,
            metadata=metadata,
        )
    )
    trajectory.record_terminal(TerminalState(winner="p1", turn_count=1, capped=False))
    return RolloutRecord(
        battle_id=trajectory.battle_id,
        seed=seed,
        format_id=format_id,
        policy_ids={"p1": decision.policy_id},
        decision_round_count=1,
        elapsed_seconds=0.0,
        terminal=trajectory.terminal,
        trajectory=trajectory,
    )


def _require_expected_scenario_decision(scenario: TeacherScenario, decision: PolicyDecision) -> None:
    mismatches: list[str] = []
    if decision.action_index != scenario.expected_action_index:
        mismatches.append(
            f"action_index expected {scenario.expected_action_index}, observed {decision.action_index}"
        )
    observed_branch = decision.metadata.get("teacher_branch")
    if observed_branch != scenario.expected_teacher_branch:
        mismatches.append(
            f"teacher_branch expected {scenario.expected_teacher_branch!r}, observed {observed_branch!r}"
        )
    if mismatches:
        raise ValueError(
            f"teacher scenario demo {scenario.scenario_id!r} did not match curated expectation: "
            + "; ".join(mismatches)
        )


def _teacher_branch_counts(records: Sequence[RolloutRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for step in record.trajectory.steps:
            branch = step.metadata.get("teacher_branch")
            if isinstance(branch, str) and branch:
                counts[branch] = counts.get(branch, 0) + 1
    return dict(sorted(counts.items()))


def _run_scenario(policy: Policy, scenario: TeacherScenario, *, rng_seed: int) -> dict[str, Any]:
    expected = {
        "action_index": scenario.expected_action_index,
        "action_family": scenario.expected_action_family,
        "teacher_branch": scenario.expected_teacher_branch,
        "reason_contains": scenario.expected_reason_contains,
    }
    try:
        decision = policy.select_action(scenario.observation, rng=random.Random(rng_seed))
    except Exception as exc:  # noqa: BLE001 - scenario preflight should report every failure.
        return {
            "id": scenario.scenario_id,
            "description": scenario.description,
            "passed": False,
            "expected": expected,
            "observed": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    metadata = dict(decision.metadata)
    observed = {
        "action_index": decision.action_index,
        "action_family": metadata.get("action_family"),
        "teacher_branch": metadata.get("teacher_branch"),
        "teacher_reason": metadata.get("teacher_reason"),
        "teacher_score": metadata.get("teacher_score"),
    }
    failures: list[str] = []
    if decision.action_index != scenario.expected_action_index:
        failures.append("action_index")
    if observed["action_family"] != scenario.expected_action_family:
        failures.append("action_family")
    if observed["teacher_branch"] != scenario.expected_teacher_branch:
        failures.append("teacher_branch")
    if scenario.expected_reason_contains and scenario.expected_reason_contains not in str(observed["teacher_reason"] or ""):
        failures.append("teacher_reason")
    return {
        "id": scenario.scenario_id,
        "description": scenario.description,
        "passed": not failures,
        "expected": expected,
        "observed": observed,
        "failed_fields": failures,
        "error": None,
    }


def _observation(mask: tuple[bool, ...], *, metadata: Mapping[str, Any]) -> PokeZeroObservationV0:
    spec = ObservationSpec(categorical_feature_count=1, numeric_feature_count=1)
    return PokeZeroObservationV0(
        categorical_ids=tuple((0,) for _ in range(spec.token_count)),
        numeric_features=tuple((0.0,) for _ in range(spec.token_count)),
        token_type_ids=tuple(0 for _ in range(spec.token_count)),
        attention_mask=tuple(True for _ in range(spec.token_count)),
        legal_action_mask=mask,
        metadata=dict(metadata),
    )


def _move(action_index: int, move_id: str, move_name: str) -> dict[str, Any]:
    return {"action_index": action_index, "kind": "move", "legal": True, "move_id": move_id, "move_name": move_name}


def _switch(action_index: int, species: str, *, hp_fraction: float, status: str = "none") -> dict[str, Any]:
    return {
        "action_index": action_index,
        "kind": "switch",
        "legal": True,
        "pokemon": {"species": species, "hp_fraction": hp_fraction, "status": status},
    }
