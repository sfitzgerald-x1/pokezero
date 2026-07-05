"""Dense potential-based reward shaping (WS-E arm 1, docs/wse_shaping_and_coverage_design.md).

The value head is hazard/status-blind (dV self-response ~0.1% of spread in the 2026-07-03
probe read); dense shaping shortens the credit-assignment horizon so status/faint/hp
consequences reach the value target within a few decisions instead of an entire game.

Formulation
-----------
A player-relative potential over the GROUND-TRUTH battle state (both sides):

    Phi_p(s) =   hp_weight     * (own_hp_total  - foe_hp_total)  / 6
               + faint_weight  * (own_alive     - foe_alive)     / 6
               + sum_st status_weights[st] * (foe_status[st] - own_status[st]) / 6
               + hazard_weight * (foe_spikes_layers - own_spikes_layers) / 3

with the per-step shaping reward for the acting player's decision at state s_k:

    f_k = gamma * Phi_p(s_{k+1}) - Phi_p(s_k)        (potential-based; Ng et al. 1999)

where s_{k+1} is the same player's NEXT decision state (the per-player decision process is
what the dataset's discounting walks) and the terminal potential is 0 by default
(``terminal_mode='zero'``, the policy-invariant absorbing-state convention).
``terminal_mode='carry'`` instead freezes the final potential (Phi_T := Phi_{K-1}) which
reproduces the accumulate-and-keep behavior of the legacy hp/faint delta shaping.

Sign conventions: own KO -> Phi drops -> negative shaping; newly statused foe -> positive;
exactly symmetric states -> Phi = 0. Weights follow the WSE design doc's arm-1 structure
(hp 0.5 / faint 0.5 / status 0.25, hazards deliberately absent from the primary arm — the
``hazard_weight`` component exists for the oracle-fit/ranker tools and defaults to 0).

Ground truth at collection time: in self-play both sides are observed, and each player's
``self_team`` observation metadata is exact (request-derived, no belief involvement). The
combined state for a decision at turn t is (actor's own self view at t, opponent's most
recent self view at turn <= t). Belief-merged ``opponent_team`` views are never used.

Everything here is a pure function of a rollout record; the same code path serves cache
collection, train-time JSONL ingestion, and the stage-0 rescoring / oracle-fit / ranker
tools. Torch- and numpy-free.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from .collection import RolloutRecord

# Gen 3 non-volatile status conditions ("tox" is the distinct toxic-poison condition).
NON_VOLATILE_STATUSES = ("brn", "frz", "par", "psn", "slp", "tox")
TEAM_SIZE = 6
MAX_SPIKES_LAYERS = 3

HP_COMPONENT = "hp"
FAINT_COMPONENT = "faint"
HAZARD_COMPONENT = "hazard"
STATUS_COMPONENT_PREFIX = "status:"

TERMINAL_MODES = ("zero", "carry")


def component_names() -> tuple[str, ...]:
    """Canonical ordering of the potential's component vector (shared with oracle-fit)."""
    return (
        HP_COMPONENT,
        FAINT_COMPONENT,
        *(f"{STATUS_COMPONENT_PREFIX}{status}" for status in NON_VOLATILE_STATUSES),
        HAZARD_COMPONENT,
    )


@dataclass(frozen=True)
class ShapingConfig:
    """Weights for the potential function plus the terminal-potential convention.

    ``status_weights`` maps each non-volatile status to its own weight (sorted tuple of
    pairs so the frozen config stays hashable and serialization is canonical). Negative
    weights are allowed: the oracle-fit tool derives weights from data and the ranker
    must be able to evaluate deliberately-bad configs.
    """

    hp_weight: float = 0.0
    faint_weight: float = 0.0
    status_weights: tuple[tuple[str, float], ...] = ()
    hazard_weight: float = 0.0
    terminal_mode: str = "zero"

    def __post_init__(self) -> None:
        normalized: dict[str, float] = {}
        for status, weight in self.status_weights:
            key = str(status).strip().lower()
            if key not in NON_VOLATILE_STATUSES:
                raise ValueError(
                    f"unknown status condition {status!r}; expected one of {', '.join(NON_VOLATILE_STATUSES)}."
                )
            if key in normalized:
                raise ValueError(f"duplicate status weight: {key}.")
            normalized[key] = float(weight)
        object.__setattr__(
            self,
            "status_weights",
            tuple(sorted((status, weight) for status, weight in normalized.items())),
        )
        for name in ("hp_weight", "faint_weight", "hazard_weight"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite.")
        if any(not math.isfinite(weight) for _, weight in self.status_weights):
            raise ValueError("status weights must be finite.")
        if self.terminal_mode not in TERMINAL_MODES:
            raise ValueError(f"terminal_mode must be one of {', '.join(TERMINAL_MODES)}.")

    def status_weight(self, status: str) -> float:
        for key, weight in self.status_weights:
            if key == status:
                return weight
        return 0.0

    def is_zero(self) -> bool:
        return (
            self.hp_weight == 0.0
            and self.faint_weight == 0.0
            and self.hazard_weight == 0.0
            and all(weight == 0.0 for _, weight in self.status_weights)
        )

    def component_weights(self) -> dict[str, float]:
        weights = {
            HP_COMPONENT: self.hp_weight,
            FAINT_COMPONENT: self.faint_weight,
            HAZARD_COMPONENT: self.hazard_weight,
        }
        for status in NON_VOLATILE_STATUSES:
            weights[f"{STATUS_COMPONENT_PREFIX}{status}"] = self.status_weight(status)
        return weights

    def to_dict(self) -> dict[str, Any]:
        return {
            "hp_weight": self.hp_weight,
            "faint_weight": self.faint_weight,
            "status_weights": {status: weight for status, weight in self.status_weights},
            "hazard_weight": self.hazard_weight,
            "terminal_mode": self.terminal_mode,
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShapingConfig":
        if not isinstance(payload, Mapping):
            raise ValueError("shaping config must be a JSON object.")
        known = {"hp_weight", "faint_weight", "status_weights", "status_weight", "hazard_weight", "terminal_mode"}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown shaping config key(s): {', '.join(unknown)}.")
        if "status_weights" in payload and "status_weight" in payload:
            raise ValueError("shaping config cannot set both status_weights and status_weight.")
        status_weights: tuple[tuple[str, float], ...]
        if "status_weight" in payload:
            uniform = float(payload["status_weight"])
            status_weights = tuple((status, uniform) for status in NON_VOLATILE_STATUSES)
        else:
            raw = payload.get("status_weights") or {}
            if not isinstance(raw, Mapping):
                raise ValueError("status_weights must be a mapping of status -> weight.")
            status_weights = tuple((str(status), float(weight)) for status, weight in raw.items())
        return cls(
            hp_weight=float(payload.get("hp_weight", 0.0)),
            faint_weight=float(payload.get("faint_weight", 0.0)),
            status_weights=status_weights,
            hazard_weight=float(payload.get("hazard_weight", 0.0)),
            terminal_mode=str(payload.get("terminal_mode", "zero")),
        )

    @classmethod
    def from_json(cls, text: str) -> "ShapingConfig":
        return cls.from_dict(json.loads(text))


def _uniform_status_weights(weight: float) -> tuple[tuple[str, float], ...]:
    return tuple((status, weight) for status in NON_VOLATILE_STATUSES)


# WSE design-doc arm-1 weight structure: hp 0.5, faint 0.5, status 0.25 (uniform across
# the non-volatile statuses), NO hazard term in the primary arm (rewarding Spikes
# placement directly would hand-craft the answer the dV probe is supposed to detect).
SHAPING_PRESETS: Mapping[str, ShapingConfig] = {
    "wse-arm1": ShapingConfig(
        hp_weight=0.5,
        faint_weight=0.5,
        status_weights=_uniform_status_weights(0.25),
        hazard_weight=0.0,
    ),
}

# Spellings of an EXPLICIT unshaped request (distinct from "flag absent" for provenance
# cross-checks that follow the #507 adopt-from-checkpoint pattern).
EXPLICIT_UNSHAPED_SPECS = frozenset({"none", "off"})


def parse_shaping_spec(spec: str) -> ShapingConfig | None:
    """Parse a --shaping-weights value: preset name, inline JSON object, or @/path/to.json.

    Returns None for the explicit unshaped spellings ("none"/"off").
    """
    text = str(spec).strip()
    if not text:
        raise ValueError("shaping weights spec must be non-empty.")
    if text.lower() in EXPLICIT_UNSHAPED_SPECS:
        return None
    if text.lower() in SHAPING_PRESETS:
        return SHAPING_PRESETS[text.lower()]
    if text.startswith("@"):
        return ShapingConfig.from_json(Path(text[1:]).expanduser().read_text(encoding="utf-8"))
    if text.startswith("{"):
        return ShapingConfig.from_json(text)
    raise ValueError(
        f"unsupported shaping weights spec {spec!r}: expected a preset "
        f"({', '.join(sorted(SHAPING_PRESETS))}), inline JSON object, @/path/to.json, or 'none'."
    )


# ---------------------------------------------------------------------------
# Ground-truth side snapshots and component extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SideSnapshot:
    """Ground-truth summary of one side, from that player's own observation metadata."""

    hp_total: float = 0.0
    alive: int = 0
    status_counts: tuple[tuple[str, int], ...] = ()
    spikes_layers: int = 0

    def status_count(self, status: str) -> int:
        for key, count in self.status_counts:
            if key == status:
                return count
        return 0


EMPTY_SIDE = SideSnapshot()


def side_snapshot_from_observation_metadata(metadata: Mapping[str, Any] | None) -> SideSnapshot:
    """The acting player's own-side ground truth from ``self_team`` observation metadata.

    A player's view of its own team is exact (request-derived); fainted mons contribute
    zero hp, drop out of the alive count, and drop out of the status counts (metamon
    convention: a KO must not also be scored as "status cleared").
    """
    payload = metadata if isinstance(metadata, Mapping) else {}
    team = payload.get("self_team")
    hp_total = 0.0
    alive = 0
    status_counts: dict[str, int] = {}
    if isinstance(team, Sequence) and not isinstance(team, (str, bytes, bytearray)):
        for entry in team:
            if not isinstance(entry, Mapping):
                continue
            fainted = bool(entry.get("fainted", False))
            if fainted:
                continue
            alive += 1
            hp_total += _hp_fraction(entry)
            status = str(entry.get("status") or "").strip().lower()
            if status in NON_VOLATILE_STATUSES:
                status_counts[status] = status_counts.get(status, 0) + 1
    side_counts = payload.get("self_side_condition_counts")
    spikes = 0
    if isinstance(side_counts, Mapping):
        try:
            spikes = int(side_counts.get("spikes", 0) or 0)
        except (TypeError, ValueError):
            spikes = 0
    return SideSnapshot(
        hp_total=hp_total,
        alive=alive,
        status_counts=tuple(sorted(status_counts.items())),
        spikes_layers=max(0, min(MAX_SPIKES_LAYERS, spikes)),
    )


def _hp_fraction(entry: Mapping[str, Any]) -> float:
    try:
        value = float(entry.get("hp_fraction"))
    except (TypeError, ValueError):
        value = 0.0 if bool(entry.get("fainted", False)) else 1.0
    return min(1.0, max(0.0, value))


def components_from_sides(own: SideSnapshot, foe: SideSnapshot) -> dict[str, float]:
    """Player-relative normalized component vector; Phi = weights . components.

    Positive components mean the acting player's position is better: own hp/alive count
    positively, foe statuses and foe-side hazards count positively.
    """
    components: dict[str, float] = {
        HP_COMPONENT: (own.hp_total - foe.hp_total) / TEAM_SIZE,
        FAINT_COMPONENT: (own.alive - foe.alive) / TEAM_SIZE,
        HAZARD_COMPONENT: (foe.spikes_layers - own.spikes_layers) / MAX_SPIKES_LAYERS,
    }
    for status in NON_VOLATILE_STATUSES:
        components[f"{STATUS_COMPONENT_PREFIX}{status}"] = (
            foe.status_count(status) - own.status_count(status)
        ) / TEAM_SIZE
    return components


def potential_from_components(components: Mapping[str, float], config: ShapingConfig) -> float:
    weights = config.component_weights()
    return sum(weights[name] * components.get(name, 0.0) for name in component_names())


def potential_from_sides(own: SideSnapshot, foe: SideSnapshot, config: ShapingConfig) -> float:
    return potential_from_components(components_from_sides(own, foe), config)


# ---------------------------------------------------------------------------
# Record-level extraction (the single source of truth for every consumer)
# ---------------------------------------------------------------------------


def ground_truth_components_by_step_index(record: "RolloutRecord") -> dict[int, dict[str, float]]:
    """Per-step player-relative component vectors from a record's ground-truth views.

    For a step by player p at turn t: own side = p's ``self_team`` view at that step
    (exact); foe side = the opponent's most recent ``self_team`` view at turn <= t (both
    players observe at the same request boundary on shared turns; on asymmetric
    sub-requests, e.g. a lone forced switch, the opponent view is typically one request
    stale, occasionally more across consecutive asymmetric sub-requests). Staleness only
    delays when a change enters Phi, never correctness: PBRS policy invariance holds for
    ANY potential. Records with missing metadata degrade to empty sides (components 0).
    """
    steps = record.trajectory.steps
    views_by_player: dict[str, list[tuple[int, SideSnapshot]]] = {}
    snapshots: list[SideSnapshot] = []
    for step in steps:
        snapshot = side_snapshot_from_observation_metadata(step.observation.metadata)
        snapshots.append(snapshot)
        views_by_player.setdefault(step.player_id, []).append((step.turn_index, snapshot))

    components_by_step: dict[int, dict[str, float]] = {}
    for step_index, step in enumerate(steps):
        own = snapshots[step_index]
        foe = EMPTY_SIDE
        for player_id, views in views_by_player.items():
            if player_id == step.player_id:
                continue
            # Latest opponent self view at turn <= this step's turn (views are turn-ordered).
            for turn_index, snapshot in views:
                if turn_index > step.turn_index:
                    break
                foe = snapshot
            break
        components_by_step[step_index] = components_from_sides(own, foe)
    return components_by_step


def potentials_by_step_index(record: "RolloutRecord", *, config: ShapingConfig) -> dict[int, float]:
    return {
        step_index: potential_from_components(components, config)
        for step_index, components in ground_truth_components_by_step_index(record).items()
    }


def shaping_terms(
    potentials: Sequence[float],
    *,
    gamma: float,
    terminal_potential: float = 0.0,
) -> tuple[float, ...]:
    """Per-decision shaping rewards for one player's potential sequence.

    ``f_k = gamma * Phi_{k+1} - Phi_k`` with ``Phi_K = terminal_potential``. The
    discounted sum telescopes: sum_k gamma^k f_k = gamma^K * terminal_potential - Phi_0.
    """
    if not potentials:
        return ()
    values = [float(value) for value in potentials]
    terms = [gamma * values[k + 1] - values[k] for k in range(len(values) - 1)]
    terms.append(gamma * float(terminal_potential) - values[-1])
    return tuple(terms)


def potential_shaping_rewards_by_step_index(
    record: "RolloutRecord",
    *,
    config: ShapingConfig,
    gamma: float,
) -> dict[int, float]:
    """Per-step potential-based shaping rewards over a full record (both players).

    Each player's decisions form the state sequence; the term for their k-th decision is
    attached to that step index. Terminal potential is 0 (``terminal_mode='zero'``) or
    the final observed potential (``terminal_mode='carry'``).
    """
    if config.is_zero():
        return {index: 0.0 for index, _ in enumerate(record.trajectory.steps)}
    potentials = potentials_by_step_index(record, config=config)
    step_indices_by_player: dict[str, list[int]] = {}
    for step_index, step in enumerate(record.trajectory.steps):
        step_indices_by_player.setdefault(step.player_id, []).append(step_index)

    rewards: dict[int, float] = {}
    for step_indices in step_indices_by_player.values():
        player_potentials = [potentials[index] for index in step_indices]
        terminal_potential = 0.0 if config.terminal_mode == "zero" else player_potentials[-1]
        terms = shaping_terms(player_potentials, gamma=gamma, terminal_potential=terminal_potential)
        for step_index, term in zip(step_indices, terms, strict=True):
            rewards[step_index] = term
    return rewards


def annotate_record_with_shaping(
    record: "RolloutRecord",
    *,
    config: ShapingConfig,
    gamma: float,
) -> "RolloutRecord":
    """Copy of ``record`` whose steps carry their shaping component (raw reward untouched)."""
    from dataclasses import replace

    from .trajectory import BattleTrajectory

    rewards = potential_shaping_rewards_by_step_index(record, config=config, gamma=gamma)
    trajectory = record.trajectory
    annotated = BattleTrajectory(
        battle_id=trajectory.battle_id,
        format_id=trajectory.format_id,
        seed=trajectory.seed,
        metadata=dict(trajectory.metadata),
    )
    for step_index, step in enumerate(trajectory.steps):
        annotated.append(replace(step, shaping_reward=rewards.get(step_index, 0.0)))
    if trajectory.terminal is not None:
        annotated.record_terminal(trajectory.terminal)
    return replace(record, trajectory=annotated)


def resolve_shaping_config(value: "ShapingConfig | Mapping[str, Any] | str | None") -> ShapingConfig | None:
    """Coerce flag/JSON/config inputs to ShapingConfig (None stays None)."""
    if value is None:
        return None
    if isinstance(value, ShapingConfig):
        return value
    if isinstance(value, Mapping):
        return ShapingConfig.from_dict(value)
    return parse_shaping_spec(str(value))
