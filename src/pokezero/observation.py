"""Versioned fixed-shape observation contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .actions import ACTION_COUNT

# v2 (the WS-1 C one-way break, docs/observation_compression_design.md + corrections layer):
# window_size=1 snapshots, the 24 recent-event tokens are dropped, and the token sequence gains
# an opponent-tendency-stats token plus a 128-slot transition-token block (K in tokens, corrections item 11).
# Checkpoints trained under v1 must load-and-refuse; replay them from their pinned tag
# (docs/model_versioning.md).
OBSERVATION_SCHEMA_VERSION_V2 = "pokezero.observation.v2"
# v2.1 (checkpoint-driven, NOT a one-way door): defender identity on move transition tokens,
# per-bucket revealed-move PP-validity bits, active-mon substitute HP fraction; the investment
# reserve carries forward. Unlike the v1->v2 break, v2 stays a fully supported encode mode for
# as long as live v2 training runs produce checkpoints: the schema version + numeric width
# resolve from each loaded checkpoint's model_config (feature_masks_from_model_config /
# env_config_from_checkpoint_provenance latch family), so v2 checkpoints keep scoring through every
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
# v3 (checkpoint-driven, fourth entry in the same dual-schema table; docs/observation_v3_spec.md):
# the v2.2 turn-merged semantic surface plus the documented V3 public signals, reorganized into a
# grouped 155-column numeric layout after removing 14 evidence-backed unreachable fields, with a
# 64-row turn-history tail. The private writer surface remains an encoder implementation detail;
# all consumers use the public V3 layout. Every v2.2 artifact retains its frozen output and
# checkpoint-driven resolution. V3 is NOT the fresh default until the Rust fold encoder mirrors it
# and the golden corpus regenerates at v3.
OBSERVATION_SCHEMA_VERSION_V3 = "pokezero.observation.v3"
# v4 (checkpoint-driven, fifth entry in the same table; docs/observation_v4_spec.md): the k0
# FEATURE PACK. V4 keeps the whole v3 semantic surface and adds enumerated CURRENT-STATE columns
# for public facts that previously reached only the search world, or only the history region:
# opponent forced recharge, per-side last executed move, Truant loaf phase, the currently TRACED
# ability, last-round damage dealt/taken, and the entry-hazard credit / expected-value pair
# (docs/k0-feature-pack-plan.md Parts A and B). The motivating question is whether a pure-Markov
# k0 policy (transition_token_budget=0) can match a k1 one once the facts k1's single history row
# was carrying are named as current state.
# V4 also RETIRES one live current-state column: NUMERIC_TIER2_INVESTMENT_PINNED (writer 139).
# The defender-side investment conclusion now narrows the belief candidate set instead
# (ObservationFeatureMasks.investment_belief_narrowing), which moves the candidate-set-count and
# uncertainty columns present in every schema plus the possible_items/moves/abilities surfaces —
# a strictly richer surface than 139's lossy +/-1 / +/-0.5 class projection of the same evidence.
# v2.1/v2.2/v3 keep the column: their checkpoints have it in their input layout. Retiring it at
# v4 is a clean census edit while v4 is unlaunched, and a loud schema break afterwards.
# NEW CONTRACT, NEW ARMS: v4 SHRINKS BOTH public censuses — numeric 133 vs v3's 155, categorical
# 41 vs 51 — because dropping the transition-history region and column 139 removes more slots
# (34 + 1 numeric, 12 categorical) than the feature pack adds (13 numeric, 2 categorical). The
# private WRITER surface grows; the public census does not, and the census is what every check
# below compares. So a v4
# checkpoint can never share a cache, an env, or a run with a v3 one. That is enforced at every
# layer (schema validation, an encode-time census EXACT match at v4 — a floor cannot catch a
# census that shrinks, and BOTH v4 censuses shrink, so both exact matches are load-bearing,
# which is why v4 is exact where v3/v2.2 still floor — the checkpoint-provenance latch, the
# search-lane contract pin) exactly as the v2->v2.1->v2.2->v3 breaks were.
OBSERVATION_SCHEMA_VERSION_V4 = "pokezero.observation.v4"
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
    OBSERVATION_SCHEMA_VERSION_V3,
    OBSERVATION_SCHEMA_VERSION_V4,
)
# Turn-merged transition-surface family: schemas whose transition block carries
# pokezero.turn_merged.TurnMergedToken rows — their encode requires
# normalize_for_player(include_turn_merged=True) and a vocabulary built with
# include_turn_merged=True. v3 extends v2.2 without changing that surface, so every
# ``schema == V2_2`` include_turn_merged/vocab latch is a membership test on this tuple.
# V4 is deliberately ABSENT: it carries no transition region at all, so it needs neither the
# turn-merged token stream nor the tt_phase/tt2_* vocabulary families.
TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS = (
    OBSERVATION_SCHEMA_VERSION_V2_2,
    OBSERVATION_SCHEMA_VERSION_V3,
)
# V3-lineage family: schemas whose numeric surface is the GROUPED projection of the private
# writer rows (rather than the frozen legacy positions) and which write the v3 public signals.
# V4 extends the projection with its own columns; every ``schema == V3`` writer gate is a
# membership test on this tuple.
GROUPED_LAYOUT_OBSERVATION_SCHEMA_VERSIONS = (
    OBSERVATION_SCHEMA_VERSION_V3,
    OBSERVATION_SCHEMA_VERSION_V4,
)
# k0-feature-pack family: schemas whose encode emits the pack's two categorical families (the
# last-executed-move switch sentinel and ``ability:<id>`` for the current Trace copy). Those rows
# change the vocabulary SIZE, hence every embedding table's shape, so the vocabulary builder takes
# them as an opt-in latch (``gen3_category_vocabulary(include_feature_pack_v4=...)``) exactly as it
# does the turn-merged families — and this tuple is what every such latch tests.
FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS = (OBSERVATION_SCHEMA_VERSION_V4,)
LEGACY_OBSERVATION_SCHEMA_VERSIONS = ("pokezero.observation.v1",)
# Sentinel for artifacts whose payload carries NO observation schema version. For a one-way
# door, absent means unknown/legacy and must refuse — never "assume current spec".
UNVERSIONED_OBSERVATION_SCHEMA = "pokezero.observation.unversioned"
SHOWDOWN_PLAYER_SLOTS = ("p1", "p2")
FIELD_TOKEN_COUNT = 1
SELF_POKEMON_TOKEN_COUNT = 6
OPPONENT_POKEMON_TOKEN_COUNT = 6
ACTION_CANDIDATE_TOKEN_COUNT = ACTION_COUNT
# One opponent-tendency-stats token carries the global tendency (count, opportunity) pairs (design doc "Encoding").
OPPONENT_TENDENCY_STATS_TOKEN_COUNT = 1
# Historical name consumed by the committed V2.2 token-format generator.
STATS_TOKEN_COUNT = OPPONENT_TENDENCY_STATS_TOKEN_COUNT
# Transition-token slot budget: 128 tokens ≈ 64 turns of ordered history, truncated oldest-first
# (the truncated prefix is what the unbounded aggregates have already absorbed). The K ∈ {16-turn}
# ablation arm masks the budget down via config (ObservationFeatureMasks) — not a spec change.
TRANSITION_TOKEN_COUNT = 128
# V3 shortens only the physical turn-merged history tail. The legacy constant above remains the
# frozen V2/V2.1/V2.2 capacity and the maximum accepted feature-mask value for old checkpoints.
V3_TRANSITION_TOKEN_COUNT = 64
# V4 REMOVES THE HISTORY REGION ENTIRELY — the plan's stated end goal ("full region trim, no
# synthesized history in search worlds, the simplest observation contract"). Not a budget of
# zero over a region that still exists: the tokens, the history numeric group, and the
# turn-merged categorical families are all gone from the contract. What the history rows were
# carrying is either named as current state by the feature pack or deliberately dropped.
#
# Consequences that make this more than a mask: the sequence is 23 tokens instead of 87, so
# every forward is ~3.8x shorter; there is no transition_token_budget knob to mis-set; and a
# v4 checkpoint cannot silently be fed synthesized history, because there is nowhere to put it.
V4_TRANSITION_TOKEN_COUNT = 0


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
    opponent_tendency_stats_token_count: int = OPPONENT_TENDENCY_STATS_TOKEN_COUNT
    transition_token_count: int = TRANSITION_TOKEN_COUNT
    schema_version: str = OBSERVATION_SCHEMA_VERSION

    @property
    def token_count(self) -> int:
        return (
            FIELD_TOKEN_COUNT
            + SELF_POKEMON_TOKEN_COUNT
            + OPPONENT_POKEMON_TOKEN_COUNT
            + ACTION_CANDIDATE_TOKEN_COUNT
            + self.opponent_tendency_stats_token_count
            + self.transition_token_count
        )


@dataclass(frozen=True)
class ObservationFeatureMasks:
    """Ablation-arm feature masks (config, NOT spec — shapes and version are unchanged).

    Masked-off content is zeroed and attention-masked at encode time, so an arm trains and
    evaluates on the same spec version with the block simply dark:

    - ``opponent_tendency_stats_block``: the opponent-tendency-stats token + the per-opponent-mon tendency triple.
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
      This exists because A2 is the pack's LARGEST single surface, and the k0-feature-pack
      plan's arm design is exactly ``k0+pack`` against ``k0+pack+lastmove``: two arms whose
      only difference is this column, so the read attributes whatever moves to A2 rather than
      to "the pack" as a bundle. Inert under every schema below v4, where the column does not
      exist — so toggling it can never perturb a v2.x/v3 encode.
    - ``investment_belief_narrowing``: whether a defender-side investment CONCLUSION also
      narrows that mon's belief candidate variants (``pokezero.investment`` calling
      ``PublicBattleBeliefEngine.narrow_candidate_variants``). Default False.

      NOT the same axis as ``tier2_investment``, which governs a COLUMN. This switch changes
      BELIEF STATE, and the belief state feeds columns that exist in EVERY schema —
      ``NUMERIC_CANDIDATE_SET_COUNT`` (5) and ``NUMERIC_UNCERTAINTY`` (6) on every
      opponent-mon token, plus the possible-items/moves/abilities counts and every sampled
      search world. So unlike every other mask here, turning this on is not an ablation of
      something already written: it perturbs encodes that v2/v2.1/v2.2/v3/v4 checkpoints were
      all trained against. Default OFF keeps them byte-identical; an arm that wants the
      richer belief opts in explicitly and trains fresh.

      Narrowing needs the investment inference to be RUNNING, so it is gated on
      ``tier2_residuals`` and the candidate-set source exactly as the tracker is; it does
      NOT require ``tier2_investment``, because the column and the belief write are
      independent consumers of the same conclusion.
    """

    opponent_tendency_stats_block: bool = True
    exact_state: bool = True
    transition_token_budget: int = TRANSITION_TOKEN_COUNT
    tier2_residuals: bool = True
    tier2_investment: bool = False
    feature_pack_last_move: bool = True
    investment_belief_narrowing: bool = False

    def __post_init__(self) -> None:
        # 0 is a valid budget: the transition region exists but is fully masked
        # (Markov-state-only ablations). The encoder fill/mask paths handle it. Under v4 there
        # is no region at all, so the budget is inert rather than meaningful — the encoder
        # never reads it, and observation_spec.transition_token_count is 0.
        if not 0 <= self.transition_token_budget <= TRANSITION_TOKEN_COUNT:
            raise ValueError(
                f"transition_token_budget must be in 0..{TRANSITION_TOKEN_COUNT}, "
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
