"""Versioned fixed-shape observation contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .actions import ACTION_COUNT

# v2 (the WS-1 C one-way break, docs/observation_compression_design.md + corrections layer):
# window_size=1 snapshots, the 24 recent-event tokens are dropped, and the token sequence gains
# a stats token plus a 128-slot transition-token block (K in tokens, corrections item 11).
# Checkpoints trained under v1 must load-and-refuse; replay them from their pinned tag
# (docs/model_versioning.md).
OBSERVATION_SCHEMA_VERSION_V2 = "pokezero.observation.v2"
# v2.1 (checkpoint-driven, NOT a one-way door): defender identity on move transition tokens,
# per-bucket revealed-move PP-validity bits, active-mon substitute HP fraction; the investment
# reserve carries forward. Unlike the v1->v2 break, v2 stays a fully supported encode mode for
# as long as live v2 training runs produce checkpoints: the schema version + numeric width
# resolve from each loaded checkpoint's model_config (feature_masks_from_model_config /
# env_config_with_checkpoint_masks latch family), so v2 checkpoints keep scoring through every
# harness while fresh trains stamp v2.1.
OBSERVATION_SCHEMA_VERSION_V2_1 = "pokezero.observation.v2.1"
# v2.2 (checkpoint-driven, third entry in the same dual-schema table): TURN-MERGED transition
# tokens — the transition block carries one token per turn/lead/replacement phase with two
# ordered sub-blocks (speed order explicit, negated/absent declarations representable) instead
# of one token per declared action. All v2.1 blocks (defender identity semantics, PP-validity
# bits, sub HP, per-mon pinned Tier-2 bits) carry forward; the appended second-sub-block
# columns extend the v2.1 census. Same resolution mechanism: the schema an env encodes comes
# from the loaded checkpoint's stamped model_config, so v2/v2.1 artifacts stay first-class.
# K BUDGET UNIT CHANGE (loud): the transition budget flag counts TOKENS in every schema, but
# a v2.2 token covers a WHOLE TURN — the v2/v2.1 K=64 horizon (~32 turns) is budget=32 under
# v2.2; an unchanged K roughly doubles the temporal horizon.
OBSERVATION_SCHEMA_VERSION_V2_2 = "pokezero.observation.v2.2"
# The CURRENT schema: what fresh artifacts (new trains, checkpoint-free encodes) are stamped
# with. Loading a checkpoint always overrides this default with the checkpoint's own schema.
# v2.2 earned the default slot (2026-07-08): under the schedule-uncompressed A/B reads the
# turn-merged arm matched or beat v2.1/v2 on every yardstick and holds the current bests;
# v2.1/v2 artifacts remain first-class via the checkpoint-driven latch.
OBSERVATION_SCHEMA_VERSION = OBSERVATION_SCHEMA_VERSION_V2_2
SUPPORTED_OBSERVATION_SCHEMA_VERSIONS = (
    OBSERVATION_SCHEMA_VERSION_V2,
    OBSERVATION_SCHEMA_VERSION_V2_1,
    OBSERVATION_SCHEMA_VERSION_V2_2,
)
LEGACY_OBSERVATION_SCHEMA_VERSIONS = ("pokezero.observation.v1",)
# Sentinel for artifacts whose payload carries NO observation schema version. For a one-way
# door, absent means unknown/legacy and must refuse — never "assume current spec".
UNVERSIONED_OBSERVATION_SCHEMA = "pokezero.observation.unversioned"
SHOWDOWN_PLAYER_SLOTS = ("p1", "p2")
FIELD_TOKEN_COUNT = 1
SELF_POKEMON_TOKEN_COUNT = 6
OPPONENT_POKEMON_TOKEN_COUNT = 6
ACTION_CANDIDATE_TOKEN_COUNT = ACTION_COUNT
# One stats token carries the global tendency (count, opportunity) pairs (design doc "Encoding").
STATS_TOKEN_COUNT = 1
# Transition-token slot budget: 128 tokens ≈ 64 turns of ordered history, truncated oldest-first
# (the truncated prefix is what the unbounded aggregates have already absorbed). The K ∈ {16-turn}
# ablation arm masks the budget down via config (ObservationFeatureMasks) — not a spec change.
TRANSITION_TOKEN_COUNT = 128


@dataclass(frozen=True)
class ObservationSpec:
    """Uniform token feature widths for efficient batching.

    Different token sections use different feature subsets; unused categorical
    or numeric columns should be padded by the encoder.

    ``schema_version`` keys the encoder's schema-conditional blocks (the v2.1 columns are
    written only under a v2.1 spec) and is stamped onto every encoded observation, so the
    numeric census and the version travel together — never a global constant the shapes
    silently drift away from.
    """

    categorical_feature_count: int
    numeric_feature_count: int
    stats_token_count: int = STATS_TOKEN_COUNT
    transition_token_count: int = TRANSITION_TOKEN_COUNT
    schema_version: str = OBSERVATION_SCHEMA_VERSION

    @property
    def token_count(self) -> int:
        return (
            FIELD_TOKEN_COUNT
            + SELF_POKEMON_TOKEN_COUNT
            + OPPONENT_POKEMON_TOKEN_COUNT
            + ACTION_CANDIDATE_TOKEN_COUNT
            + self.stats_token_count
            + self.transition_token_count
        )


@dataclass(frozen=True)
class ObservationFeatureMasks:
    """Ablation-arm feature masks (config, NOT spec — shapes and version are unchanged).

    Masked-off content is zeroed and attention-masked at encode time, so an arm trains and
    evaluates on the same spec version with the block simply dark:

    - ``stats_block``: the stats token + the per-opponent-mon tendency triple.
    - ``exact_state``: the exact-state layer (PP-ledger fractions, sleep/duration counters,
      sleep-clause / trapper / pending-Wish bits, computed expected stats).
    - ``transition_token_budget``: how many of the most recent transition tokens are filled
      (32 tokens = the K=16-turn ablation arm); the remaining slots stay zero + masked.
      UNIT NOTE: under schema v2.2 (turn-merged) each transition token is a WHOLE TURN,
      so the same number roughly doubles the temporal horizon — the v2/v2.1 K=64
      horizon is budget=32 under v2.2.
    - ``tier2_residuals``: whether transition tokens that CARRY Tier-2 residuals (populated
      by ``pokezero.tier2`` behind PR D's precision gate) write the reserved
      residual/validity slots. Tokens from the plain extraction path carry none, so the
      slots stay 0.0 either way for pipelines that never run the Tier-2 inference.
    - ``tier2_investment``: whether tokens carrying defender-side investment conclusions
      (populated by ``pokezero.investment`` behind ITS precision gate) write the reserved
      investment slot. A SEPARATE switch from ``tier2_residuals`` because the provenance
      differs: checkpoints trained after #505 but before the investment channel latched
      residuals live while the investment column was constant zero — one switch could not
      mask investment off for them without also darkening residuals. Default False until
      v2.1 training adopts the column; pre-v2.1 pipelines encode byte-identically.
    """

    stats_block: bool = True
    exact_state: bool = True
    transition_token_budget: int = TRANSITION_TOKEN_COUNT
    tier2_residuals: bool = True
    tier2_investment: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.transition_token_budget <= TRANSITION_TOKEN_COUNT:
            raise ValueError(
                f"transition_token_budget must be in 1..{TRANSITION_TOKEN_COUNT}, "
                f"got {self.transition_token_budget}."
            )


DEFAULT_OBSERVATION_FEATURE_MASKS = ObservationFeatureMasks()


@dataclass(frozen=True)
class ObservationPerspective:
    """Debug/provenance metadata for a player-relative observation.

    Model tensors are normalized to self/opponent sections. Raw Showdown seats
    are retained only so harnesses can audit normalization and submit selected
    actions back to the correct protocol side.
    """

    player_id: str
    showdown_slot: str
    opponent_showdown_slot: str

    def __post_init__(self) -> None:
        _require_showdown_slot("showdown_slot", self.showdown_slot)
        _require_showdown_slot("opponent_showdown_slot", self.opponent_showdown_slot)
        if self.showdown_slot == self.opponent_showdown_slot:
            raise ValueError("showdown_slot and opponent_showdown_slot must differ.")

    @classmethod
    def from_showdown_slot(cls, player_id: str, showdown_slot: str) -> "ObservationPerspective":
        return cls(
            player_id=player_id,
            showdown_slot=showdown_slot,
            opponent_showdown_slot=opponent_showdown_slot(showdown_slot),
        )


@dataclass(frozen=True)
class PokeZeroObservationV0:
    categorical_ids: Any
    numeric_features: Any
    token_type_ids: Any
    attention_mask: Any
    legal_action_mask: Any
    perspective: ObservationPerspective | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = OBSERVATION_SCHEMA_VERSION

    def validate(self, spec: ObservationSpec) -> None:
        require_current_observation_schema(self.schema_version, context="observation")
        if self.schema_version != spec.schema_version:
            raise ValueError(
                f"observation: schema {self.schema_version!r} does not match the validating "
                f"spec's {spec.schema_version!r} — {OBSERVATION_SCHEMA_VERSION_V2!r} and "
                f"{OBSERVATION_SCHEMA_VERSION_V2_1!r} are both supported but must never be "
                "mixed within one pipeline (checkpoint-driven resolution, no silent coercion)."
            )
        _require_outer_length("categorical_ids", self.categorical_ids, spec.token_count)
        _require_outer_length("numeric_features", self.numeric_features, spec.token_count)
        _require_outer_length("token_type_ids", self.token_type_ids, spec.token_count)
        _require_outer_length("attention_mask", self.attention_mask, spec.token_count)
        _require_outer_length("legal_action_mask", self.legal_action_mask, ACTION_COUNT)
        _require_inner_length("categorical_ids", self.categorical_ids, spec.categorical_feature_count)
        _require_inner_length("numeric_features", self.numeric_features, spec.numeric_feature_count)


def require_current_observation_schema(schema_version: str | None, *, context: str) -> None:
    """Refuse any observation schema outside the supported set, with a clean message.

    This is the data-side latch of the one-way door for LEGACY artifacts: production ingest
    paths call it so a stale v1 (or unversioned) artifact dies here — with the
    replay-from-pinned-tag guidance — instead of surfacing later as a bare tensor-shape error
    mid-training. During the v2/v2.1 dual-schema window BOTH current versions pass this gate;
    pairing an artifact with the RIGHT model is enforced downstream by the checkpoint-driven
    spec resolution plus the numeric-census guard, which names both schemas on a mismatch.
    """
    if schema_version in SUPPORTED_OBSERVATION_SCHEMA_VERSIONS:
        return
    if (
        schema_version in LEGACY_OBSERVATION_SCHEMA_VERSIONS
        or schema_version == UNVERSIONED_OBSERVATION_SCHEMA
        or not schema_version
    ):
        described = schema_version or UNVERSIONED_OBSERVATION_SCHEMA
        raise ValueError(
            f"{context}: observation schema {described!r} predates the supported specs "
            f"({OBSERVATION_SCHEMA_VERSION_V2!r}, {OBSERVATION_SCHEMA_VERSION_V2_1!r}) "
            "(window=1 + transition tokens + exact-state layer). Legacy data and checkpoints "
            "must be replayed from their pinned tag (docs/model_versioning.md)."
        )
    raise ValueError(f"{context}: unsupported observation schema version: {schema_version!r}.")


def opponent_showdown_slot(showdown_slot: str) -> str:
    _require_showdown_slot("showdown_slot", showdown_slot)
    return "p2" if showdown_slot == "p1" else "p1"


def _require_showdown_slot(name: str, value: str) -> None:
    if value not in SHOWDOWN_PLAYER_SLOTS:
        allowed = ", ".join(SHOWDOWN_PLAYER_SLOTS)
        raise ValueError(f"{name} must be one of {allowed}; got {value!r}.")


def _require_outer_length(name: str, values: Any, expected: int) -> None:
    actual = _dimension(values, 0)
    if actual != expected:
        raise ValueError(f"{name} must contain {expected} values, got {actual}.")


def _require_inner_length(name: str, rows: Any, expected: int) -> None:
    width = _dimension(rows, 1)
    if width is not None:
        if width != expected:
            raise ValueError(f"{name} rows must contain {expected} values, got {width}.")
        return
    for index, row in enumerate(rows):
        if len(row) != expected:
            raise ValueError(f"{name}[{index}] must contain {expected} values, got {len(row)}.")


def _dimension(values: Any, axis: int) -> int | None:
    shape = getattr(values, "shape", None)
    if shape is not None and len(shape) > axis:
        return int(shape[axis])
    if axis == 0:
        return len(values)
    return None
