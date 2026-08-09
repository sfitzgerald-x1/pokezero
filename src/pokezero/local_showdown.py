"""Local Pokemon Showdown BattleStream-backed PokeZero environment."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, TextIO

if TYPE_CHECKING:
    from .category_vocab import CategoryVocabulary

from .belief import PublicBattleBeliefEngine, RevealedPokemonBelief
from .actions import ACTION_COUNT, MOVE_ACTION_COUNT
from .dex import load_showdown_dex_cached, normalize_id
from .tier2 import canonical_move_id
from .env import BattleFormat, BattleStartOverride, PlayerId, StepResult, TerminalState
from .observation import (
    DEFAULT_OBSERVATION_FEATURE_MASKS,
    FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS,
    TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS,
    ObservationFeatureMasks,
    ObservationSpec,
    PokeZeroObservationV0,
    opponent_showdown_slot,
)
from .randbat import load_gen3_randbat_source_cached
from .randbat_vocab import gen3_category_vocabulary
from .showdown import (
    DEFAULT_REPLAY_OBSERVATION_SPEC,
    PlayerRelativeBattleState,
    ShowdownPokemon,
    ShowdownReplayState,
    _is_active_protocol_ident,
    _is_current_public_active,
    _normalize_identifier,
    _ReplayParser,
    normalize_for_player,
    observation_from_player_state,
    showdown_choice_for_action,
)
from .investment import InvestmentLiveTracker
from .paths import portable_path
from .tier2 import Tier2LiveTracker, cb_whitelist_for_source, own_team_from_request

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Conventional locations for a pokemon-showdown checkout, tried in order. NONE of these may
# name a user: this is a public repo, and a maintainer's home directory in a tracked default
# leaks a username and a local filesystem layout, and makes the value useless to everyone else.
# `Path.home()` expresses the same location portably.
_SHOWDOWN_ROOT_CANDIDATES = (
    Path.home() / "workspace" / "pokerena" / "vendor" / "pokemon-showdown",
    _REPO_ROOT / "vendor" / "pokemon-showdown",
    _REPO_ROOT.parent / "pokemon-showdown",
)


def default_showdown_root() -> Path:
    """The Showdown checkout to use when a caller does not name one.

    ``POKEZERO_SHOWDOWN_ROOT`` wins outright. Otherwise the first candidate that looks like a
    real checkout (it has a ``data`` directory) is chosen, and failing that the first candidate
    is returned unchanged so callers report a stable, comprehensible path rather than None.

    Resolved on CALL, not at import: tests and tools set the environment variable around a
    subprocess or a fixture, and an import-time constant would silently ignore them.
    """
    override = os.environ.get("POKEZERO_SHOWDOWN_ROOT")
    if override:
        return Path(override)
    for candidate in _SHOWDOWN_ROOT_CANDIDATES:
        if (candidate / "data").is_dir():
            return candidate
    return _SHOWDOWN_ROOT_CANDIDATES[0]


# Back-compat alias for the ~40 call sites that read the module constant. Kept as a constant
# rather than removed, but note it freezes the value at import time; prefer the function.
DEFAULT_SHOWDOWN_ROOT = default_showdown_root()
BRIDGE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "battle_bridge.mjs"
PLAYER_IDS: tuple[PlayerId, PlayerId] = ("p1", "p2")
_DEFAULT_PLAYER_NAMES: Mapping[PlayerId, str] = {"p1": "PokeZero p1", "p2": "PokeZero p2"}


class LocalShowdownError(RuntimeError):
    """Raised when the local BattleStream bridge or simulator rejects a step."""


def env_config_from_checkpoint_provenance(
    env_config: LocalShowdownConfig,
    required_masks: "ObservationFeatureMasks | Sequence[ObservationFeatureMasks]",
    *,
    context: str,
    required_specs: "ObservationSpec | Sequence[ObservationSpec]" = (),
    required_vocabs: "CategoryVocabulary | Sequence[CategoryVocabulary]" = (),
) -> "LocalShowdownConfig":
    """Derive the env's encode-time masks, observation spec AND category vocabulary from provenance.

    Renamed from ``env_config_with_checkpoint_masks`` on 2026-07-29. The old name asserted one
    axis while the function latched three, and a label that stops the next reader from looking
    is the failure mode this whole change is about — the vocabulary axis went unlatched for
    months behind a comment that said MUST. A name is a claim about coverage; this one now
    matches what it does.

    The train/eval consistency latch for the mask axis (same failure shape as the #492
    belief-source mismatch): a checkpoint stamped with ablation masks (K=32 budget, stats-off,
    exact-state-off) must be evaluated on observations encoded the same way. Semantics:

    - no transformer checkpoints in play -> env config unchanged;
    - checkpoints agree on one mask set -> env adopts it (overriding the untouched default);
    - checkpoints DISAGREE -> hard fail (one env cannot encode two ways);
    - env carries an EXPLICIT non-default mask config that differs from the checkpoints'
      -> hard fail loudly (never silently prefer either side).

    ``required_specs`` extends the latch to the observation SCHEMA axis with identical
    semantics (the dual-schema window's core mechanism): pass each loaded checkpoint's
    ``observation_spec_from_model_config`` so a v2 checkpoint gets the v2 encode (121
    columns, no v2.1 blocks) and a v2.1 checkpoint the v2.1 encode — resolved from stamped
    provenance, never from the build's default. A v2 and a v2.1 checkpoint in one env, or an
    explicit non-default env spec that disagrees with the checkpoints', hard-fails loudly.

    ``required_vocabs`` extends it to the token ENUMERATION axis, with the same semantics
    again: pass each loaded checkpoint's ``category_vocab_from_model_config``. This axis
    existed unlatched until 2026-07-29 and is the one that fails *silently*. Masks and spec
    disagreements change the observation's SHAPE, so a mismatch tends to blow up in the
    forward pass. The vocabulary is a positional list of the same width whichever build wrote
    it: a token inserted mid-list renumbers everything after it and the encoder happily
    produces a well-formed tensor of embedding rows that mean something else. Nothing
    crashes; the model just scores a state it never saw.

    **Fail-closed (mirrors #948's required-not-defaulted move).** When any checkpoint
    provenance is supplied at all, ``required_vocabs`` is REQUIRED. Passing masks or a spec
    while omitting the vocabulary raises rather than quietly leaving the env to enumerate
    from the build — which is precisely the shape of the bug this closes, where
    ``LocalShowdownConfig.category_vocab``'s "callers MUST pass the model's vocabulary"
    contract was a comment that nothing enforced. A caller with no checkpoints in play
    (``required_masks`` and ``required_specs`` both empty) is still a no-op.
    """
    from .category_vocab import CategoryVocabulary
    from .observation import ObservationFeatureMasks, ObservationSpec

    if isinstance(required_masks, ObservationFeatureMasks):
        required_masks = (required_masks,)
    distinct: list[ObservationFeatureMasks] = []
    for masks in required_masks:
        if masks not in distinct:
            distinct.append(masks)
    if isinstance(required_specs, ObservationSpec):
        required_specs = (required_specs,)
    distinct_specs: list[ObservationSpec] = []
    for spec in required_specs:
        if spec not in distinct_specs:
            distinct_specs.append(spec)
    if isinstance(required_vocabs, CategoryVocabulary):
        required_vocabs = (required_vocabs,)
    distinct_vocabs: list[CategoryVocabulary] = []
    for vocab in required_vocabs:
        if vocab not in distinct_vocabs:
            distinct_vocabs.append(vocab)
    if not distinct and not distinct_specs and not distinct_vocabs:
        return env_config
    if not distinct_vocabs:
        # Fail closed. Provenance is in play, so the vocabulary is knowable and its absence
        # is an un-updated call site, not a legitimate case: every valid
        # TransformerPolicyConfig carries category_vocab (neural_policy.py __post_init__).
        raise ValueError(
            f"{context}: checkpoint provenance was supplied without required_vocabs. The "
            "categorical vocabulary is part of the observation contract — the model's "
            "embedding rows were learned against the enumeration stamped in the checkpoint, "
            "and re-deriving it from the build silently shifts every token after any that "
            "the build has since inserted. Pass category_vocab_from_model_config(config, "
            "showdown_root) for each loaded checkpoint."
        )
    if len(distinct) > 1:
        raise ValueError(
            f"{context}: checkpoints require conflicting observation feature masks "
            f"({', '.join(repr(masks) for masks in distinct)}); one env cannot encode both — "
            "evaluate them in separate runs."
        )
    if len(distinct_specs) > 1:
        schemas = sorted({spec.schema_version for spec in distinct_specs})
        raise ValueError(
            f"{context}: checkpoints require conflicting observation specs "
            f"({', '.join(repr(spec) for spec in distinct_specs)}); one env cannot encode "
            "two observation schemas. For eval, score them in separate runs; for "
            "iterate/resume, a training line keeps its own stamped schema "
            f"({' vs '.join(repr(schema) for schema in schemas)}) — continue it on the "
            "build it is pinned to instead of mixing it with fresh-stamped configs."
        )
    resolved = env_config
    if distinct:
        required = distinct[0]
        if resolved.feature_masks != required:
            if resolved.feature_masks != DEFAULT_OBSERVATION_FEATURE_MASKS:
                raise ValueError(
                    f"{context}: env feature masks {resolved.feature_masks!r} conflict with the "
                    f"loaded checkpoint's trained masks {required!r}. Refusing to encode observations "
                    "the model never trained on (the #492 train/eval-mismatch class); drop the "
                    "explicit env masks or evaluate a matching checkpoint."
                )
            resolved = replace(resolved, feature_masks=required)
    if distinct_specs:
        required_spec = distinct_specs[0]
        if resolved.observation_spec != required_spec:
            # Within-schema transition-region refinement: an explicit env spec that differs
            # from the checkpoint's trained spec ONLY in transition_token_count (same schema
            # version, all other fields equal) adopts the checkpoint's width. The region is a
            # capacity parameter of the schema and the checkpoint is authoritative for it
            # (region-trimmed models); encoding a different width than the model was trained
            # on has no valid use — the forward rejects the shape anyway.
            env_spec = resolved.observation_spec
            region_refinement = (
                env_spec.schema_version == required_spec.schema_version
                and replace(env_spec, transition_token_count=required_spec.transition_token_count)
                == required_spec
            )
            if resolved.observation_spec != DEFAULT_REPLAY_OBSERVATION_SPEC and not region_refinement:
                raise ValueError(
                    f"{context}: env observation spec {resolved.observation_spec!r} conflicts "
                    f"with the loaded checkpoint's trained spec {required_spec!r} "
                    f"(schema {required_spec.schema_version!r}). Refusing to encode a schema "
                    "the model never trained on (the census-mismatch class); drop the explicit "
                    "env spec or evaluate a matching checkpoint."
                )
            resolved = replace(resolved, observation_spec=required_spec)
    if len(distinct_vocabs) > 1:
        sizes = ", ".join(str(len(vocab.tokens)) for vocab in distinct_vocabs)
        raise ValueError(
            f"{context}: checkpoints were trained on different categorical vocabularies "
            f"({sizes} tokens). One env encodes one enumeration, and the mismatched model "
            "would index embedding rows it learned as other tokens — score them in "
            "separate runs."
        )
    required_vocab = distinct_vocabs[0]
    if resolved.category_vocab != required_vocab:
        if resolved.category_vocab is not None:
            raise ValueError(
                f"{context}: env category vocabulary ({len(resolved.category_vocab.tokens)} "
                f"tokens) conflicts with the loaded checkpoint's trained vocabulary "
                f"({len(required_vocab.tokens)} tokens). Refusing to encode token rows the "
                "model never trained on; drop the explicit env vocabulary or evaluate a "
                "matching checkpoint."
            )
        resolved = replace(resolved, category_vocab=required_vocab)
    return resolved


def belief_set_source_env_enabled() -> bool:
    """The single env flip point for candidate-set belief features (training AND eval sides).

    Every consumer must call this rather than re-parsing the variable: two independent parsers
    drifting apart is exactly the silent train/eval observation mismatch class.
    """
    return os.environ.get("POKEZERO_BELIEF_SET_SOURCE", "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LocalShowdownConfig:
    showdown_root: Path | str | None = None
    bridge_path: Path | str = BRIDGE_PATH
    node_binary: str = "node"
    observation_spec: ObservationSpec = DEFAULT_REPLAY_OBSERVATION_SPEC
    # Ablation-arm feature masks (config, not spec): masked-off blocks are zeroed +
    # attention-masked at encode time. Callers pairing the env with a model must keep these
    # consistent with the model config's stats_block_enabled / exact_state_enabled /
    # transition_token_budget fields.
    feature_masks: ObservationFeatureMasks = DEFAULT_OBSERVATION_FEATURE_MASKS
    # Category vocabulary used to convert token strings to embedding rows. When None it is built
    # from showdown_root, which is correct ONLY when no trained model is in play: the build's
    # enumeration is a positional list, so a token added since a checkpoint was trained
    # renumbers every token after it and the encoder resolves rows the model learned as other
    # values — silently, since the tensor stays well-formed.
    #
    # This used to say callers "MUST" pass the model's vocabulary, and nothing enforced it;
    # the production consumption sites did not. Enforcement now lives in
    # `env_config_from_checkpoint_provenance`, which requires `required_vocabs` whenever any
    # checkpoint provenance is supplied. Route env construction for a loaded checkpoint
    # through that helper rather than setting this field by hand.
    category_vocab: "CategoryVocabulary | None" = None
    read_timeout_seconds: float = 10.0
    # Whether the belief engine narrows opponent candidate sets via the Gen 3 randbats set source
    # (populates possible_moves / candidate_variants / possible ability+item). None defers to the
    # POKEZERO_BELIEF_SET_SOURCE env var so training and eval images flip together from one place;
    # set explicitly (True/False) to pin it (e.g. in tests). Revealed moves/ability/item do NOT
    # depend on this — they come straight from the protocol.
    set_belief_source: bool | None = None

    def resolved_showdown_root(self) -> Path:
        configured = self.showdown_root or os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT
        return Path(configured).expanduser().resolve()

    def resolved_bridge_path(self) -> Path:
        return Path(self.bridge_path).expanduser().resolve()

    def belief_set_source_enabled(self) -> bool:
        if self.set_belief_source is not None:
            return self.set_belief_source
        return belief_set_source_env_enabled()


@dataclass(frozen=True)
class SnapshotAnnotationCache:
    """Incremental annotation state paired with a restorable local snapshot.

    Restoring a snapshot must reproduce its public observation exactly. The Tier-2
    and investment trackers record when each action was first assessed, so rebuilding
    them from the final replay boundary can legitimately use more evidence and change
    their output. Every snapshot therefore retains immutable tracker clones; search
    restores receive independent clones before adding a branch suffix.
    """

    tier2_trackers: Mapping[PlayerId, Tier2LiveTracker] = field(default_factory=dict)
    investment_trackers: Mapping[PlayerId, InvestmentLiveTracker] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalShowdownSnapshot:
    """Restorable simulator plus local public-state snapshot for a live bridge battle."""

    battle_token: str
    battle_id: str
    format_id: BattleFormat
    observation_format_id: BattleFormat
    bridge_snapshot: Mapping[str, Any]
    protocol_lines: tuple[str, ...]
    latest_requests: Mapping[PlayerId, Mapping[str, Any]]
    first_requests: Mapping[PlayerId, Mapping[str, Any]]
    request_history: Mapping[PlayerId, tuple[Mapping[str, Any], ...]]
    replay: ShowdownReplayState
    belief_engine: PublicBattleBeliefEngine
    latest_turn: int
    terminal: TerminalState | None
    # Snapshot-local action translations are computed from the same determinized
    # world used for the branch. They avoid rebuilding public player state for
    # every repeated Root-PUCT visit without exposing data outside that world.
    search_choice_cache: Mapping[PlayerId, Mapping[int, str]] = field(default_factory=dict)
    # Incremental public-evidence trackers needed to reproduce the observation at
    # the snapshot boundary. These are independent of the optional search choices.
    annotation_cache: SnapshotAnnotationCache | None = None


@dataclass(frozen=True)
class PublicBattleMaterializationState:
    """Public/player-known source state for direct sampled-world construction.

    This intentionally excludes a simulator snapshot and the other player's request. The
    captured replay fold and belief engine contain only public protocol facts. The ``self_*``
    fields contain only the acting player's request-known state. In particular, cached move
    states retain PP for a Pokemon after it switches out, while the first request preserves exact
    team stats after a Pokemon faints.
    """

    player_id: PlayerId
    format_id: BattleFormat
    observation_format_id: BattleFormat
    replay: ShowdownReplayState
    belief_engine: PublicBattleBeliefEngine
    self_request: Mapping[str, Any]
    self_move_states: Mapping[str, tuple[Mapping[str, Any], ...]] = field(default_factory=dict)
    self_initial_request: Mapping[str, Any] = field(default_factory=dict)

    @property
    def deferred_opponent_action_player(self) -> PlayerId | None:
        """Return the opponent whose committed move must resolve after this switch.

        A Baton Pass forced switch interrupts a simultaneous turn after the opponent has already
        committed an action. Its identity is hidden, but the pending action itself is public
        timing information and must be sampled into a direct search world.
        """

        if _request_materialization_kind(self.self_request) != "force-switch":
            return None
        if self.player_id not in self.replay.pending_baton_pass:
            return None
        return "p2" if self.player_id == "p1" else "p1"


class LocalShowdownEnv:
    """Synchronous `PokeZeroEnv` backed by a one-battle Node BattleStream bridge."""

    def __init__(self, config: LocalShowdownConfig | None = None) -> None:
        self.config = config or LocalShowdownConfig()
        self._process: subprocess.Popen[str] | None = None
        self._stdout_queue: queue.Queue[str | None] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_lines: list[str] = []
        self._battle_id = "local-showdown"
        self._format_id: BattleFormat = "gen3randombattle"
        self._observation_format_id: BattleFormat = self._format_id
        self._lines: list[str] = []
        self._latest_requests: dict[PlayerId, Mapping[str, Any]] = {}
        self._latest_turn = 0
        self._terminal: TerminalState | None = None
        self._last_step_had_error = False
        # Cumulative bridge counters are sampled by Root-PUCT before and after one
        # decision. They intentionally live outside battle reset state so a warm
        # bridge shell can report a precise per-decision delta.
        self._bridge_round_trip_seconds = 0.0
        self._bridge_round_trip_count = 0
        self._bridge_node_processing_seconds = 0.0
        self._bridge_node_processing_count = 0
        # Nested slices of bridge-handle root branches. These stay cumulative
        # across warm-pool resets so Root-PUCT can take a per-decision delta.
        self._root_puct_branch_local_state_restore_seconds = 0.0
        self._root_puct_branch_local_state_restore_count = 0
        self._root_puct_branch_choice_encoding_seconds = 0.0
        self._root_puct_branch_choice_encoding_count = 0
        self._root_puct_branch_bridge_round_trip_seconds = 0.0
        self._root_puct_branch_bridge_round_trip_count = 0
        self._root_puct_branch_bridge_node_processing_seconds = 0.0
        self._root_puct_branch_bridge_node_processing_count = 0
        self._root_puct_branch_result_projection_seconds = 0.0
        self._root_puct_branch_result_projection_count = 0
        self._root_puct_branch_observation_projection_seconds = 0.0
        self._root_puct_branch_observation_projection_count = 0
        # Observation construction dominates the remaining W5 branch cost. Keep
        # its nested timings separate so the next optimization targets measured
        # state normalization, feature encoding, or belief-overlay work.
        self._root_puct_branch_observation_state_normalization_seconds = 0.0
        self._root_puct_branch_observation_state_normalization_count = 0
        self._root_puct_branch_observation_incremental_sync_seconds = 0.0
        self._root_puct_branch_observation_incremental_sync_count = 0
        self._root_puct_branch_observation_replay_snapshot_seconds = 0.0
        self._root_puct_branch_observation_replay_snapshot_count = 0
        self._root_puct_branch_observation_player_state_normalization_seconds = 0.0
        self._root_puct_branch_observation_player_state_normalization_count = 0
        self._root_puct_branch_observation_state_annotation_seconds = 0.0
        self._root_puct_branch_observation_state_annotation_count = 0
        self._root_puct_branch_observation_encoding_seconds = 0.0
        self._root_puct_branch_observation_encoding_count = 0
        self._root_puct_branch_belief_overlay_projection_seconds = 0.0
        self._root_puct_branch_belief_overlay_projection_count = 0
        # Persistent incremental state: the parser + belief engine are fed each new protocol line
        # / event exactly once (see _sync_incremental_state), so observations cost O(state) instead
        # of re-parsing and re-ingesting the whole accumulated log every call (O(n^2) per battle).
        self._parser = _ReplayParser(
            self._battle_id,
            complete_prefix=True,
            hp_visibility={"p1": "exact", "p2": "exact"},
        )
        # Shared, immutable candidate-set source (built once per process, cached). None when the
        # belief set source is disabled, in which case only protocol-revealed facts populate.
        self._belief_set_source = (
            load_gen3_randbat_source_cached(self.config.resolved_showdown_root())
            if self.config.belief_set_source_enabled()
            else None
        )
        self._belief_engine = PublicBattleBeliefEngine(
            format_id=self._observation_format_id,
            set_source=self._belief_set_source,
            # Unlike the tier2 producers, this narrowing lives INSIDE the engine (the facts are
            # protocol lines, not inferences), so the mask has to reach the constructor.
            item_belief_narrowing=self.item_belief_narrowing_active(),
        )
        self._parsed_line_count = 0
        self._belief_fed_count = 0
        # Tier-2 live residual trackers (#505 follow-up): one per perspective, created
        # lazily once that player's first request arrives (it carries the exact own-team
        # stats the residual math needs). Active only when the encode-time masks keep the
        # channel on AND the candidate-set source is enabled — mask-off arms and
        # pre-#505 checkpoints (whose provenance latches tier2_residuals=False) pay
        # nothing and encode byte-identically.
        self._first_requests: dict[PlayerId, Mapping[str, Any]] = {}
        self._request_history: dict[PlayerId, list[Mapping[str, Any]]] = {player: [] for player in PLAYER_IDS}
        self._tier2_trackers: dict[PlayerId, Tier2LiveTracker] = {}
        # Defender-side investment trackers (v2.1 batch 2): same lazy per-perspective
        # pattern, active only under the tier2 channel AND the tier2_investment mask
        # (default off — see ObservationFeatureMasks).
        self._investment_trackers: dict[PlayerId, InvestmentLiveTracker] = {}
        # Warm pool: the bridge process is reused across battles. Each battle gets a unique routing
        # token; events from a prior battle carry a stale token and are ignored (see _apply_event).
        self._battle_counter = 0
        self._battle_token: str | None = None
        # Search-only snapshots are safe only after this environment is initialized from an
        # explicit belief-sampled world. Keep the generic snapshot API available for diagnostics,
        # but reject the fast bridge-resident path for a live rollout.
        self._search_snapshot_permitted = False

    @property
    def belief_set_source_hash(self) -> str | None:
        """Provenance hash of the candidate-set source encoding observations (None when disabled)."""
        if self._belief_set_source is None:
            return None
        return self._belief_set_source.metadata.source_hash

    @property
    def protocol_lines(self) -> tuple[str, ...]:
        return tuple(self._lines)

    def reset(self, *, seed: int, format_id: BattleFormat = "gen3randombattle") -> None:
        self._reset(seed=seed, format_id=format_id, start_override=None)

    def reset_with_start_override(
        self,
        *,
        seed: int,
        format_id: BattleFormat | None = None,
        start_override: BattleStartOverride,
    ) -> None:
        effective_format_id = start_override.format_id if format_id is None else str(format_id)
        if effective_format_id != start_override.format_id:
            raise ValueError(
                "reset_with_start_override format_id must match "
                f"start_override.format_id {start_override.format_id!r}."
            )
        self._reset(seed=seed, format_id=effective_format_id, start_override=start_override)

    def generate_scenario_team(self, *, seed: int) -> tuple[Mapping[str, Any], ...]:
        """Generate one complete Gen 3 randbats party through the warmed Showdown bridge."""

        if not isinstance(seed, int):
            raise ValueError("scenario team generation seed must be an integer.")
        if self._process is None or self._process.poll() is not None:
            # Starting any ordinary battle initializes the bridge process; generation itself is
            # stateless and does not read the battle, so the temporary random-game shell is safe.
            self.reset(seed=seed)
        event = self._bridge_request_event(
            {"type": "scenario_generate_team", "seed": seed},
            "scenario_team_generated",
        )
        rows = event.get("team")
        if not isinstance(rows, list) or len(rows) != 6 or not all(isinstance(row, Mapping) for row in rows):
            raise LocalShowdownError(f"Bridge emitted malformed scenario team: {event!r}")
        return tuple(_json_clone_mapping(row) for row in rows)

    def _reset(
        self,
        *,
        seed: int,
        format_id: BattleFormat = "gen3randombattle",
        start_override: BattleStartOverride | None,
    ) -> None:
        previous_token = self._battle_token
        self._battle_id = f"local-{format_id}-{seed}"
        self._format_id = format_id
        self._observation_format_id = (
            str(start_override.observation_format_id)
            if start_override is not None and start_override.observation_format_id is not None
            else format_id
        )
        self._battle_counter += 1
        self._battle_token = f"b{self._battle_counter}"
        self._search_snapshot_permitted = start_override is not None
        self._lines = []
        self._latest_requests = {}
        self._latest_turn = 0
        self._terminal = None
        self._last_step_had_error = False
        self._parser = _ReplayParser(
            self._battle_id,
            complete_prefix=True,
            hp_visibility={"p1": "exact", "p2": "exact"},
        )
        self._belief_engine = PublicBattleBeliefEngine(
            format_id=self._observation_format_id,
            set_source=self._belief_set_source,
            item_belief_narrowing=self.item_belief_narrowing_active(),
        )
        self._parsed_line_count = 0
        self._belief_fed_count = 0
        self._first_requests = {}
        self._request_history = {player: [] for player in PLAYER_IDS}
        self._tier2_trackers = {}
        self._investment_trackers = {}
        # Reuse a live bridge process across battles (warm pool); only spawn when there is none or
        # the previous one died. Stale events from the prior battle carry previous_token and are
        # ignored by _apply_event, so a clean queue drain is not required.
        reuse = self._process is not None and self._process.poll() is None
        if not reuse:
            self.close()  # clean up a dead process / drain threads, then spawn fresh
            self._validate_runtime()
            self._start_bridge()
        elif previous_token is not None:
            self._send_command({"type": "end", "battleId": previous_token})
        try:
            self._bridge_request_boundary(
                {
                    "type": "start",
                    "battleId": self._battle_token,
                    "formatid": format_id,
                    "seed": showdown_seed_from_int(seed),
                    "players": _start_players_payload(start_override),
                }
            )
        except Exception:
            self.close()
            raise

    def requested_players(self) -> tuple[PlayerId, ...]:
        return requested_players_from_requests(self._latest_requests)

    def observe(self, player: PlayerId) -> PokeZeroObservationV0:
        return self._observe(player)

    def _observe(
        self,
        player: PlayerId,
        *,
        root_puct_branch_observation: bool = False,
    ) -> PokeZeroObservationV0:
        state_started_at = time.perf_counter() if root_puct_branch_observation else None
        try:
            state = self._state_for_player(
                player, root_puct_branch_observation=root_puct_branch_observation
            )
        finally:
            if state_started_at is not None:
                self._root_puct_branch_observation_state_normalization_seconds += max(
                    0.0, time.perf_counter() - state_started_at
                )
                self._root_puct_branch_observation_state_normalization_count += 1

        encoding_started_at = time.perf_counter() if root_puct_branch_observation else None
        root = self.config.resolved_showdown_root()
        # Prefer the explicitly-paired model vocabulary; otherwise build it from the root.
        # A turn-merged (v2.2/v3) spec needs the tt_phase/tt2_* families or every merged
        # label would land in the OOV band.
        vocab = self.config.category_vocab or gen3_category_vocabulary(
            root,
            include_turn_merged=(
                self.config.observation_spec.schema_version
                in TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS
            ),
            include_feature_pack_v4=(
                self.config.observation_spec.schema_version
                in FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS
            ),
        )
        try:
            observation = observation_from_player_state(
                state,
                category_vocab=vocab,
                spec=self.config.observation_spec,
                dex=load_showdown_dex_cached(root),
                feature_masks=self.config.feature_masks,
            )
        finally:
            if encoding_started_at is not None:
                self._root_puct_branch_observation_encoding_seconds += max(
                    0.0, time.perf_counter() - encoding_started_at
                )
                self._root_puct_branch_observation_encoding_count += 1

        # The belief view is derived from the same public protocol transcript as
        # the observation. Keeping it in metadata makes public-corpus capture
        # consistent across fixed-driver and controlled FoulPlay games without
        # exposing either player's request payload.
        overlay_started_at = time.perf_counter() if root_puct_branch_observation else None
        try:
            return replace(
                observation,
                metadata={**dict(observation.metadata), "belief_view": state.belief_view.to_overlay_payload()},
            )
        finally:
            if overlay_started_at is not None:
                self._root_puct_branch_belief_overlay_projection_seconds += max(
                    0.0, time.perf_counter() - overlay_started_at
                )
                self._root_puct_branch_belief_overlay_projection_count += 1

    def legal_actions(self, player: PlayerId) -> tuple[bool, ...]:
        return self.observe(player).legal_action_mask

    def public_materialization_state(self, player: PlayerId) -> PublicBattleMaterializationState:
        """Capture public/player-known state for a separate search environment.

        This is intentionally not ``snapshot()``: no Node simulator serialization and no opponent
        request crosses from the live rollout into search. The receiving environment starts a fresh
        belief-sampled world and uses this state only to construct its public branch point.
        """

        if player not in PLAYER_IDS:
            raise ValueError(f"player must be one of {', '.join(PLAYER_IDS)}; got {player!r}.")
        if self._terminal is not None:
            raise LocalShowdownError("Cannot materialize a terminal battle state.")
        request = self._latest_requests.get(player)
        if request is None:
            raise LocalShowdownError(f"Cannot materialize without a request for {player}.")
        self._sync_incremental_state()
        replay = self._parser.snapshot()
        return PublicBattleMaterializationState(
            player_id=player,
            format_id=self._format_id,
            observation_format_id=self._observation_format_id,
            # A replay snapshot contains request payloads, so explicitly strip them before the
            # state leaves the live environment. The acting player's request is carried separately.
            replay=replace(replay, requests={}),
            belief_engine=self._belief_engine.clone(),
            self_request=_json_clone_mapping(request),
            self_move_states=actor_move_states_from_request_history(
                self._request_history[player],
                initial_request=self._first_requests.get(player) or request,
            ),
            self_initial_request=_json_clone_mapping(self._first_requests.get(player) or request),
        )

    def materialize_public_world(
        self,
        *,
        state: PublicBattleMaterializationState,
        start_override: BattleStartOverride,
        seed: int,
        deferred_opponent_actions: Mapping[PlayerId, int] | None = None,
        deferred_opponent_action_priors: Mapping[PlayerId, Sequence[float]] | None = None,
    ) -> None:
        """Construct a belief-sampled branch point without replaying prior choices."""

        if state.format_id != state.observation_format_id:
            raise LocalShowdownError("Direct materialization requires matching source observation format.")
        if state.replay.winner is not None:
            raise LocalShowdownError("Cannot materialize a terminal public replay state.")
        self.reset_with_start_override(seed=seed, start_override=start_override)
        if self._battle_token is None:
            raise LocalShowdownError("Cannot materialize before the sampled world starts.")
        event = self._bridge_request_event(
            {
                "type": "materialize",
                "battleId": self._battle_token,
                "publicState": _public_materialization_payload(
                    state,
                    deferred_opponent_actions=deferred_opponent_actions,
                    deferred_opponent_action_priors=deferred_opponent_action_priors,
                ),
            },
            "materialized",
        )
        requests = event.get("boundaryRequests")
        if not isinstance(requests, Mapping):
            raise LocalShowdownError(f"Bridge emitted malformed materialization event: {event!r}")
        direct_requests = _json_clone_requests(requests)
        if not direct_requests:
            raise LocalShowdownError("Direct materialization produced no actionable request boundary.")
        # The bridge rebuilds its team in active-first order to construct the sampled world.  Its
        # generated actor request can therefore reorder the player's own party tokens even though
        # the player-visible request at this decision boundary is already known.  Keep that exact
        # actor request for encoding and choice validation; requests for every other seat remain
        # bridge-generated from the determinized simulator.
        direct_requests[state.player_id] = _json_clone_mapping(state.self_request)
        replay = replace(
            state.replay,
            battle_id=self._battle_id,
            requests=direct_requests,
        )
        self._lines = []
        self._latest_requests = direct_requests
        initial_request = (
            state.self_initial_request if state.self_initial_request else direct_requests.get(state.player_id)
        )
        self._first_requests = (
            {state.player_id: _json_clone_mapping(initial_request)}
            if isinstance(initial_request, Mapping)
            else dict(direct_requests)
        )
        self._latest_turn = replay.turn_number
        self._terminal = None
        self._last_step_had_error = False
        self._parser = _ReplayParser.from_snapshot(replay)
        self._belief_engine = state.belief_engine.clone()
        self._parsed_line_count = 0
        self._belief_fed_count = len(replay.public_events)
        self._tier2_trackers = {}
        self._investment_trackers = {}

    def materialize_scenario_state(self, *, scenario_state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Apply a validated scenario patch and rebuild a synthetic public boundary.

        This is intentionally a separate authoring seam from public-belief search materialization.
        The bridge starts from the packed Custom Game teams already loaded in this environment and
        accepts only typed active-slot, HP, PP, status, volatile, weather, and side-condition
        values. It never receives a caller-supplied Showdown snapshot. The returned boundary is
        represented as fully revealed synthetic protocol state so normal observation encoding and
        legal-action handling stay paired with the materialized simulator world.
        """

        if self._battle_token is None:
            raise LocalShowdownError("Cannot materialize a scenario before reset.")
        event = self._bridge_request_event(
            {
                "type": "scenario_materialize",
                "battleId": self._battle_token,
                "scenarioState": dict(scenario_state),
            },
            "scenario_materialized",
        )
        requests = event.get("boundaryRequests")
        state = event.get("state")
        if not isinstance(requests, Mapping) or not isinstance(state, Mapping):
            raise LocalShowdownError(f"Bridge emitted malformed scenario materialization: {event!r}")
        direct_requests = _json_clone_requests(requests)
        if set(direct_requests) != set(PLAYER_IDS):
            raise LocalShowdownError("Scenario materialization did not produce both player requests.")
        synthetic_lines = scenario_public_protocol_lines(state, direct_requests)
        synthetic_parser = _ReplayParser(
            self._battle_id,
            complete_prefix=True,
            hp_visibility={"p1": "exact", "p2": "exact"},
        )
        synthetic_parser.feed(synthetic_lines)
        _seed_scenario_parser_state(synthetic_parser, state)
        synthetic_replay = synthetic_parser.snapshot()
        synthetic_belief = PublicBattleBeliefEngine.from_events(
            synthetic_replay.public_events,
            format_id=self._observation_format_id,
            set_source=self._belief_set_source,
        )
        # The reveal ledger is intentionally retained in the belief engine, while the fabricated
        # disclosure lines never become transition-history training features. Future real protocol
        # lines are appended normally and incrementally folded into both objects.
        self._parser = _ReplayParser.from_snapshot(
            replace(synthetic_replay, public_events=(), public_lines=())
        )
        self._belief_engine = synthetic_belief
        self._lines = list(synthetic_lines)
        self._parsed_line_count = len(self._lines)
        self._belief_fed_count = 0
        self._latest_requests = direct_requests
        self._first_requests = _json_clone_requests(direct_requests)
        self._request_history = {
            player: [_json_clone_mapping(request)] for player, request in direct_requests.items()
        }
        turn = state.get("turn")
        self._latest_turn = int(turn) if isinstance(turn, int) else 1
        self._terminal = None
        self._last_step_had_error = False
        self._tier2_trackers = {}
        self._investment_trackers = {}
        return event

    def snapshot(self) -> LocalShowdownSnapshot:
        """Capture a restorable snapshot of the current live battle.

        The snapshot includes the Node simulator state plus the Python-side protocol parser inputs.
        It is an oracle simulator snapshot; hidden-info callers must not use it as a replacement for
        explicit belief sampling.
        """

        if self._battle_token is None:
            raise LocalShowdownError("Cannot snapshot before reset.")
        self._sync_incremental_state()
        event = self._bridge_request_event(
            {"type": "snapshot", "battleId": self._battle_token},
            "snapshot",
        )
        snapshot = event.get("snapshot")
        if not isinstance(snapshot, Mapping):
            raise LocalShowdownError(f"Bridge emitted malformed snapshot event: {event!r}")
        return self._local_snapshot(bridge_snapshot=_json_clone_mapping(snapshot))

    def snapshot_for_search(self) -> LocalShowdownSnapshot:
        """Store a sampled search-world snapshot inside the bridge and return only its handle.

        Search calls this only after a belief-sampled world has been materialized or replayed.
        Keeping the serialized simulator state in Node avoids copying it through the Python bridge
        for every root visit. This must never be used to snapshot a live hidden-information game.
        """

        if self._battle_token is None:
            raise LocalShowdownError("Cannot snapshot before reset.")
        if not self._search_snapshot_permitted:
            raise LocalShowdownError(
                "Bridge-resident search snapshots require a belief-sampled start override."
            )
        self._sync_incremental_state()
        event = self._bridge_request_event(
            {"type": "snapshot_search", "battleId": self._battle_token},
            "search_snapshot",
        )
        snapshot_id = event.get("snapshotId")
        if not isinstance(snapshot_id, str) or not snapshot_id:
            raise LocalShowdownError(f"Bridge emitted malformed search snapshot event: {event!r}")
        return self._local_snapshot(
            bridge_snapshot={"snapshot_id": snapshot_id},
            include_search_choice_cache=True,
        )

    def _local_snapshot(
        self,
        *,
        bridge_snapshot: Mapping[str, Any],
        include_search_choice_cache: bool = False,
    ) -> LocalShowdownSnapshot:
        """Capture the Python state paired with either a generic or bridge snapshot."""

        if self._battle_token is None:
            raise LocalShowdownError("Cannot snapshot before reset.")
        snapshot = LocalShowdownSnapshot(
            battle_token=self._battle_token,
            battle_id=self._battle_id,
            format_id=self._format_id,
            observation_format_id=self._observation_format_id,
            bridge_snapshot=bridge_snapshot,
            protocol_lines=tuple(self._lines),
            latest_requests=_json_clone_requests(self._latest_requests),
            first_requests=_json_clone_requests(self._first_requests),
            request_history=_json_clone_request_history(self._request_history),
            replay=self._parser.snapshot(),
            belief_engine=self._belief_engine.clone(),
            latest_turn=self._latest_turn,
            terminal=self._terminal,
            annotation_cache=self._annotation_cache(),
        )
        if not include_search_choice_cache:
            return snapshot

        # State normalization may initialize stateful annotation trackers. Return
        # the shell to the exact paired snapshot so creating a search handle is
        # observationally side-effect-free for any caller that keeps using it.
        search_choice_cache = self._search_choice_cache()
        annotation_cache = self._annotation_cache()
        self._restore_local_snapshot_state(snapshot)
        return replace(
            snapshot,
            search_choice_cache=search_choice_cache,
            annotation_cache=annotation_cache,
        )

    def restore(self, snapshot: LocalShowdownSnapshot) -> None:
        """Restore a snapshot into the current live bridge battle shell.

        Search uses this only for snapshots it created after replaying a
        sampled public-information world. The snapshot payload may come from
        an earlier shell in the same warm bridge process, which lets multiple
        determinized worlds coexist without ever serializing the live battle.
        """

        if self._battle_token is None:
            raise LocalShowdownError("Cannot restore before reset.")
        if (
            self._format_id != snapshot.format_id
            or self._observation_format_id != snapshot.observation_format_id
        ):
            raise ValueError("LocalShowdownSnapshot format does not match the current live battle shell.")
        self._bridge_request_event(
            {
                "type": "restore",
                "battleId": self._battle_token,
                "snapshot": snapshot.bridge_snapshot,
            },
            "restored",
        )
        self._restore_local_snapshot_state(snapshot)

    def restore_search_snapshot(self, snapshot: LocalShowdownSnapshot) -> None:
        """Clone a bridge-resident sampled-world snapshot into the current search shell."""

        if self._battle_token is None:
            raise LocalShowdownError("Cannot restore before reset.")
        if not self._search_snapshot_permitted:
            raise LocalShowdownError(
                "Bridge-resident search snapshots require a belief-sampled start override."
            )
        if (
            self._format_id != snapshot.format_id
            or self._observation_format_id != snapshot.observation_format_id
        ):
            raise ValueError("LocalShowdownSnapshot format does not match the current live battle shell.")
        snapshot_id = snapshot.bridge_snapshot.get("snapshot_id")
        if not isinstance(snapshot_id, str) or not snapshot_id:
            raise ValueError("LocalShowdownSnapshot does not contain a bridge-resident search handle.")
        self._bridge_request_event(
            {
                "type": "restore_search",
                "battleId": self._battle_token,
                "snapshotId": snapshot_id,
            },
            "search_restored",
        )
        self._restore_local_snapshot_state(snapshot)

    def step_from_search_snapshot(
        self,
        snapshot: LocalShowdownSnapshot,
        actions: Mapping[PlayerId, int],
    ) -> StepResult:
        """Restore one belief-sampled search handle and advance it in one bridge exchange.

        The retained Node snapshot belongs to a determinized search world, never a live battle.
        Python restores its paired public parser and belief state before deriving legal choices;
        the bridge then clones the retained world and submits those choices atomically.
        """

        return self._step_from_search_snapshot(snapshot, actions)

    def step_from_search_snapshot_for_player(
        self,
        snapshot: LocalShowdownSnapshot,
        actions: Mapping[PlayerId, int],
        *,
        observation_player: PlayerId,
    ) -> StepResult:
        """Advance a zero-rollout leaf while retaining only its evaluated view.

        Rollout tails still use ``step_from_search_snapshot`` and retain every
        requested observation. This narrower form only removes redundant work
        from immediate value-leaf evaluation.
        """

        if observation_player not in PLAYER_IDS:
            raise ValueError(f"observation_player must be one of {', '.join(PLAYER_IDS)}.")
        return self._step_from_search_snapshot(
            snapshot,
            actions,
            observation_players=(observation_player,),
        )

    def _step_from_search_snapshot(
        self,
        snapshot: LocalShowdownSnapshot,
        actions: Mapping[PlayerId, int],
        *,
        observation_players: tuple[PlayerId, ...] | None = None,
    ) -> StepResult:
        if self._battle_token is None:
            raise LocalShowdownError("Cannot restore before reset.")
        if not self._search_snapshot_permitted:
            raise LocalShowdownError(
                "Bridge-resident search snapshots require a belief-sampled start override."
            )
        if (
            self._format_id != snapshot.format_id
            or self._observation_format_id != snapshot.observation_format_id
        ):
            raise ValueError("LocalShowdownSnapshot format does not match the current live battle shell.")
        snapshot_id = snapshot.bridge_snapshot.get("snapshot_id")
        if not isinstance(snapshot_id, str) or not snapshot_id:
            raise ValueError("LocalShowdownSnapshot does not contain a bridge-resident search handle.")

        # Choice conversion uses only the public snapshot paired with this sampled world. Do not
        # read the current search shell, which may hold a branch from a prior root visit.
        local_restore_started_at = time.perf_counter()
        try:
            self._restore_local_snapshot_state(snapshot)
        finally:
            self._root_puct_branch_local_state_restore_seconds += max(
                0.0, time.perf_counter() - local_restore_started_at
            )
            self._root_puct_branch_local_state_restore_count += 1
        choice_encoding_started_at = time.perf_counter()
        try:
            choices = self._cached_search_choices(snapshot, actions)
        finally:
            self._root_puct_branch_choice_encoding_seconds += max(
                0.0, time.perf_counter() - choice_encoding_started_at
            )
            self._root_puct_branch_choice_encoding_count += 1
        return self._submit_step_choices(
            choices=choices,
            payload={
                "type": "restore_search_choices",
                "battleId": self._battle_token,
                "snapshotId": snapshot_id,
                "choices": choices,
            },
            root_puct_branch_step=True,
            observation_players=observation_players,
        )

    def release_search_snapshot(self, snapshot: LocalShowdownSnapshot) -> bool:
        """Release a bridge-resident search snapshot once its prepared world is no longer needed."""

        if self._battle_token is None:
            raise LocalShowdownError("Cannot release a search snapshot before reset.")
        snapshot_id = snapshot.bridge_snapshot.get("snapshot_id")
        if not isinstance(snapshot_id, str) or not snapshot_id:
            raise ValueError("LocalShowdownSnapshot does not contain a bridge-resident search handle.")
        event = self._bridge_request_event(
            {
                "type": "release_search_snapshot",
                "battleId": self._battle_token,
                "snapshotId": snapshot_id,
            },
            "search_snapshot_released",
        )
        released = event.get("released")
        if not isinstance(released, bool):
            raise LocalShowdownError(f"Bridge emitted malformed search snapshot release event: {event!r}")
        return released

    def _restore_local_snapshot_state(self, snapshot: LocalShowdownSnapshot) -> None:
        self._battle_id = snapshot.battle_id
        self._format_id = snapshot.format_id
        self._observation_format_id = snapshot.observation_format_id
        self._lines = list(snapshot.protocol_lines)
        self._latest_requests = _json_clone_requests(snapshot.latest_requests)
        self._first_requests = _json_clone_requests(snapshot.first_requests)
        self._request_history = {
            player: [_json_clone_mapping(request) for request in snapshot.request_history.get(player, ())]
            for player in PLAYER_IDS
        }
        self._latest_turn = snapshot.latest_turn
        self._terminal = snapshot.terminal
        self._last_step_had_error = False
        self._parser = _ReplayParser.from_snapshot(snapshot.replay)
        self._belief_engine = snapshot.belief_engine.clone()
        self._parsed_line_count = len(self._lines)
        self._belief_fed_count = len(snapshot.replay.public_events)
        annotation_cache = snapshot.annotation_cache
        if annotation_cache is None:
            self._tier2_trackers = {}
            self._investment_trackers = {}
        else:
            # The cache is a branch point: every restore gets fresh mutable
            # trackers so sibling Root-PUCT visits can only add their own suffix.
            self._tier2_trackers = {
                player: tracker.clone() for player, tracker in annotation_cache.tier2_trackers.items()
            }
            self._investment_trackers = {
                player: tracker.clone()
                for player, tracker in annotation_cache.investment_trackers.items()
            }

    def reseed_simulator_rng(self, seed: int) -> None:
        """Reset Showdown's battle PRNG at the current simulator state."""

        if self._battle_token is None:
            raise LocalShowdownError("Cannot reseed before reset.")
        showdown_seed = showdown_seed_from_int(seed)
        self._bridge_request_event(
            {
                "type": "reseed",
                "battleId": self._battle_token,
                "seed": showdown_seed,
            },
            "reseeded",
        )

    def step(self, actions: Mapping[PlayerId, int]) -> StepResult:
        choices = self._choices_for_actions(actions)
        return self._submit_step_choices(
            choices=choices,
            payload={"type": "choices", "battleId": self._battle_token, "choices": choices},
        )

    def _choices_for_actions(self, actions: Mapping[PlayerId, int]) -> dict[PlayerId, str]:
        requested = self.requested_players()
        if not requested:
            raise LocalShowdownError("Cannot step without requested players.")
        missing = [player for player in requested if player not in actions]
        if missing:
            raise LocalShowdownError(f"Missing actions for requested players: {', '.join(missing)}.")

        states: dict[PlayerId, PlayerRelativeBattleState] = {
            player: self._state_for_player(player) for player in requested
        }
        choices: dict[PlayerId, str] = {}
        for player in requested:
            try:
                choices[player] = showdown_choice_for_action(states[player], actions[player])
            except ValueError as exc:
                raise ValueError(f"{player}: {exc}") from exc
        return choices

    def _search_choice_cache(self) -> dict[PlayerId, dict[int, str]]:
        """Precompute legal choices once for a retained sampled-world snapshot."""

        cache: dict[PlayerId, dict[int, str]] = {}
        for player in self.requested_players():
            state = self._state_for_player(player)
            cache[player] = {
                action_index: showdown_choice_for_action(state, action_index)
                for action_index in range(ACTION_COUNT)
                if state.legal_action_mask[action_index]
            }
        return cache

    def _annotation_cache(self) -> SnapshotAnnotationCache:
        """Freeze tracker state so restoring a snapshot preserves its observation."""

        return SnapshotAnnotationCache(
            tier2_trackers={
                player: tracker.clone() for player, tracker in self._tier2_trackers.items()
            },
            investment_trackers={
                player: tracker.clone() for player, tracker in self._investment_trackers.items()
            },
        )

    def _cached_search_choices(
        self,
        snapshot: LocalShowdownSnapshot,
        actions: Mapping[PlayerId, int],
    ) -> dict[PlayerId, str]:
        """Use a snapshot's action translations, preserving legacy error paths as fallback."""

        requested = self.requested_players()
        cache = snapshot.search_choice_cache
        if not cache or any(player not in cache for player in requested):
            return self._choices_for_actions(actions)
        # Python considers ``1`` and ``1.0`` equal dictionary keys, whereas the
        # legacy translator rejects a float when it indexes the legal-action
        # mask. Defer non-integers so cache hits never weaken that validation.
        if any(not isinstance(actions.get(player), int) for player in requested):
            return self._choices_for_actions(actions)
        try:
            return {player: cache[player][actions[player]] for player in requested}
        except (KeyError, TypeError):
            return self._choices_for_actions(actions)

    def _submit_step_choices(
        self,
        *,
        choices: Mapping[PlayerId, str],
        payload: Mapping[str, Any],
        root_puct_branch_step: bool = False,
        observation_players: tuple[PlayerId, ...] | None = None,
    ) -> StepResult:
        self._last_step_had_error = False
        self._latest_requests = {}
        bridge_before = self.root_puct_bridge_timing_snapshot() if root_puct_branch_step else None
        try:
            self._bridge_request_boundary(payload)
        finally:
            if bridge_before is not None:
                bridge_after = self.root_puct_bridge_timing_snapshot()
                self._root_puct_branch_bridge_round_trip_seconds += max(
                    0.0,
                    float(bridge_after["bridge_round_trip_seconds"])
                    - float(bridge_before["bridge_round_trip_seconds"]),
                )
                self._root_puct_branch_bridge_round_trip_count += max(
                    0,
                    int(bridge_after["bridge_round_trip_count"])
                    - int(bridge_before["bridge_round_trip_count"]),
                )
                self._root_puct_branch_bridge_node_processing_seconds += max(
                    0.0,
                    float(bridge_after["bridge_node_processing_seconds"])
                    - float(bridge_before["bridge_node_processing_seconds"]),
                )
                self._root_puct_branch_bridge_node_processing_count += max(
                    0,
                    int(bridge_after["bridge_node_processing_count"])
                    - int(bridge_before["bridge_node_processing_count"]),
                )
        if self._last_step_had_error:
            raise LocalShowdownError("Showdown rejected a submitted choice.")

        projection_started_at = time.perf_counter() if root_puct_branch_step else None
        try:
            next_requested = self.requested_players()
            terminal = self.terminal()
            # A terminal branch has no leaf observation. This matches the generic
            # path and avoids rebuilding an already-finalized player view.
            players_to_observe = (
                ()
                if terminal is not None
                else (next_requested if observation_players is None else observation_players)
            )
            observation_started_at = time.perf_counter() if root_puct_branch_step else None
            observation_count = 0
            try:
                observations = {
                    player: self._observe(player, root_puct_branch_observation=root_puct_branch_step)
                    for player in players_to_observe
                }
                observation_count = len(observations)
            finally:
                if observation_started_at is not None:
                    self._root_puct_branch_observation_projection_seconds += max(
                        0.0, time.perf_counter() - observation_started_at
                    )
                    self._root_puct_branch_observation_projection_count += observation_count
            rewards = self._rewards()
            # On terminal we leave the bridge process alive (warm pool): the finished battle is freed
            # by the next reset()'s "end" command, or by close() on shutdown. This avoids a node
            # respawn per game.
            return StepResult(
                observations=observations,
                rewards=rewards,
                terminal=terminal,
                requested_players=next_requested,
            )
        finally:
            if projection_started_at is not None:
                self._root_puct_branch_result_projection_seconds += max(
                    0.0, time.perf_counter() - projection_started_at
                )
                self._root_puct_branch_result_projection_count += 1

    def terminal(self) -> Optional[TerminalState]:
        return self._terminal

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin:
                try:
                    self._send_command({"type": "close"})
                except Exception:
                    pass
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=1.0)
        finally:
            _close_process_pipes(process)
            if self._stdout_thread is not None:
                self._stdout_thread.join(timeout=1.0)
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=1.0)
            self._process = None
            self._stdout_queue = None
            self._stdout_thread = None
            self._stderr_thread = None

    def __enter__(self) -> "LocalShowdownEnv":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _validate_runtime(self) -> None:
        showdown_root = self.config.resolved_showdown_root()
        bridge_path = self.config.resolved_bridge_path()
        if not bridge_path.exists():
            raise FileNotFoundError(f"Missing BattleStream bridge: {bridge_path}")
        if not (showdown_root / "dist" / "sim" / "index.js").exists():
            raise FileNotFoundError(
                f"Missing built Pokemon Showdown simulator at {showdown_root / 'dist' / 'sim' / 'index.js'}. "
                "Set POKEZERO_SHOWDOWN_ROOT to a built Pokemon Showdown checkout."
            )
        if shutil.which(self.config.node_binary) is None:
            raise FileNotFoundError(f"Node binary not found: {self.config.node_binary}")

    def _start_bridge(self) -> None:
        showdown_root = self.config.resolved_showdown_root()
        env = {
            "PATH": os.environ.get("PATH", ""),
            "POKEZERO_SHOWDOWN_ROOT": str(showdown_root),
        }
        self._process = subprocess.Popen(
            [
                self.config.node_binary,
                str(self.config.resolved_bridge_path()),
                "--showdown-root",
                str(showdown_root),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        if self._process.stdout is None:
            raise LocalShowdownError("Bridge stdout was not created.")
        self._stdout_queue = queue.Queue()
        self._stdout_thread = threading.Thread(
            target=_drain_stdout,
            args=(self._process.stdout, self._stdout_queue),
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread = threading.Thread(
            target=_drain_stderr,
            args=(self._process.stderr, self._stderr_lines),
            daemon=True,
        )
        self._stderr_thread.start()

    def root_puct_bridge_timing_snapshot(self) -> dict[str, float | int]:
        """Return cumulative public-safe bridge timings for Root-PUCT diagnostics.

        ``bridge_python_orchestration_seconds`` is derived by the search layer
        as round-trip wall time less the bridge's own measured Node work.  It
        therefore covers IPC, JSON handling, event routing, and Python-side
        bridge orchestration, not another additive simulator stage.
        """

        return {
            "bridge_round_trip_seconds": self._bridge_round_trip_seconds,
            "bridge_round_trip_count": self._bridge_round_trip_count,
            "bridge_node_processing_seconds": self._bridge_node_processing_seconds,
            "bridge_node_processing_count": self._bridge_node_processing_count,
        }

    def root_puct_branch_step_timing_snapshot(self) -> dict[str, float | int]:
        """Return cumulative nested timing for fused sampled-world branch steps.

        These counters are populated only by ``step_from_search_snapshot``.
        They identify local setup and post-step observation work inside the
        additive branch-step wall time without exposing a simulator snapshot.
        """

        return {
            "branch_local_state_restore_seconds": self._root_puct_branch_local_state_restore_seconds,
            "branch_local_state_restore_count": self._root_puct_branch_local_state_restore_count,
            "branch_choice_encoding_seconds": self._root_puct_branch_choice_encoding_seconds,
            "branch_choice_encoding_count": self._root_puct_branch_choice_encoding_count,
            "branch_bridge_round_trip_seconds": self._root_puct_branch_bridge_round_trip_seconds,
            "branch_bridge_round_trip_count": self._root_puct_branch_bridge_round_trip_count,
            "branch_bridge_node_processing_seconds": self._root_puct_branch_bridge_node_processing_seconds,
            "branch_bridge_node_processing_count": self._root_puct_branch_bridge_node_processing_count,
            "branch_result_projection_seconds": self._root_puct_branch_result_projection_seconds,
            "branch_result_projection_count": self._root_puct_branch_result_projection_count,
            "branch_observation_projection_seconds": (
                self._root_puct_branch_observation_projection_seconds
            ),
            "branch_observation_projection_count": self._root_puct_branch_observation_projection_count,
            "branch_observation_state_normalization_seconds": (
                self._root_puct_branch_observation_state_normalization_seconds
            ),
            "branch_observation_state_normalization_count": (
                self._root_puct_branch_observation_state_normalization_count
            ),
            "branch_observation_incremental_sync_seconds": (
                self._root_puct_branch_observation_incremental_sync_seconds
            ),
            "branch_observation_incremental_sync_count": (
                self._root_puct_branch_observation_incremental_sync_count
            ),
            "branch_observation_replay_snapshot_seconds": (
                self._root_puct_branch_observation_replay_snapshot_seconds
            ),
            "branch_observation_replay_snapshot_count": (
                self._root_puct_branch_observation_replay_snapshot_count
            ),
            "branch_observation_player_state_normalization_seconds": (
                self._root_puct_branch_observation_player_state_normalization_seconds
            ),
            "branch_observation_player_state_normalization_count": (
                self._root_puct_branch_observation_player_state_normalization_count
            ),
            "branch_observation_state_annotation_seconds": (
                self._root_puct_branch_observation_state_annotation_seconds
            ),
            "branch_observation_state_annotation_count": (
                self._root_puct_branch_observation_state_annotation_count
            ),
            "branch_observation_encoding_seconds": self._root_puct_branch_observation_encoding_seconds,
            "branch_observation_encoding_count": self._root_puct_branch_observation_encoding_count,
            "branch_belief_overlay_projection_seconds": (
                self._root_puct_branch_belief_overlay_projection_seconds
            ),
            "branch_belief_overlay_projection_count": (
                self._root_puct_branch_belief_overlay_projection_count
            ),
        }

    def _bridge_request_event(
        self,
        payload: Mapping[str, Any],
        event_type: str,
    ) -> Mapping[str, Any]:
        started_at = time.perf_counter()
        self._send_command(payload)
        event = self._read_until_event_type(event_type)
        self._record_bridge_round_trip(started_at, event)
        return event

    def _bridge_request_boundary(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        started_at = time.perf_counter()
        self._send_command(payload)
        event = self._read_until_boundary()
        self._record_bridge_round_trip(started_at, event)
        return event

    def _record_bridge_round_trip(self, started_at: float, event: Mapping[str, Any]) -> None:
        """Accumulate one completed command/response exchange without changing behavior."""

        elapsed_seconds = max(0.0, time.perf_counter() - started_at)
        self._bridge_round_trip_seconds += elapsed_seconds
        self._bridge_round_trip_count += 1
        node_proc_ms = event.get("nodeProcMs")
        if (
            not isinstance(node_proc_ms, bool)
            and isinstance(node_proc_ms, (float, int))
            and math.isfinite(float(node_proc_ms))
            and node_proc_ms >= 0.0
        ):
            # The receipt timestamp originates on the bridge process. Clamp
            # tiny clock/scheduling discrepancies to preserve a non-negative
            # Python/IPC remainder in the exported diagnostic.
            self._bridge_node_processing_seconds += min(
                elapsed_seconds,
                float(node_proc_ms) / 1000.0,
            )
            self._bridge_node_processing_count += 1

    def _send_command(self, payload: Mapping[str, Any]) -> None:
        if self._process is None or self._process.stdin is None or self._process.poll() is not None:
            raise LocalShowdownError(self._bridge_exit_message())
        self._process.stdin.write(f"{json.dumps(payload, separators=(',', ':'))}\n")
        self._process.stdin.flush()

    def _read_until_boundary(self) -> Mapping[str, Any]:
        deadline = time.monotonic() + self.config.read_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LocalShowdownError(self._timeout_message())
            event = self._read_event(timeout=remaining)
            if event is None:
                continue
            if self._apply_event(event):
                return event

    def _read_event(self, *, timeout: float) -> Mapping[str, Any] | None:
        if self._stdout_queue is None:
            raise LocalShowdownError("Bridge is not running.")
        try:
            line = self._stdout_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if line is None:
            raise LocalShowdownError(self._bridge_exit_message())
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LocalShowdownError(f"Bridge emitted invalid JSON: {line.rstrip()}") from exc
        if not isinstance(event, Mapping):
            raise LocalShowdownError(f"Bridge emitted non-object event: {event!r}")
        return event

    def _read_until_event_type(self, event_type: str) -> Mapping[str, Any]:
        deadline = time.monotonic() + self.config.read_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LocalShowdownError(self._timeout_message())
            event = self._read_event(timeout=remaining)
            if event is None:
                continue
            if event.get("type") == "error":
                raise LocalShowdownError(str(event.get("message") or "Bridge error."))
            battle_id = event.get("battleId")
            if battle_id is not None and self._battle_token is not None and battle_id != self._battle_token:
                continue
            if event.get("type") == event_type:
                return event
            self._apply_event(event)

    def _apply_event(self, event: Mapping[str, Any]) -> bool:
        event_type = event.get("type")
        # On a reused (warm) process, events from a finished battle still drain through the queue;
        # they carry that battle's routing token, so ignore anything not for the current battle.
        # Global events (process-level errors, "closed") carry no battleId and are not filtered.
        battle_id = event.get("battleId")
        if battle_id is not None and self._battle_token is not None and battle_id != self._battle_token:
            return False
        if event_type == "error":
            raise LocalShowdownError(str(event.get("message") or "Bridge error."))
        if event_type == "ready":
            return True
        if event_type == "terminal":
            if self._terminal is None:
                self._terminal = TerminalState(winner=None, turn_count=self._latest_turn)
            return True
        if event_type != "stream":
            return False
        stream = event.get("stream")
        lines = event.get("lines")
        if not isinstance(stream, str) or not isinstance(lines, list):
            raise LocalShowdownError(f"Malformed bridge stream event: {event!r}")
        clean_lines = [str(line) for line in lines if str(line)]
        for line in clean_lines:
            if line.startswith("|error|"):
                self._last_step_had_error = True
                raise LocalShowdownError(f"Showdown emitted error: {line}")
        if stream == "omniscient":
            for line in clean_lines:
                self._lines.append(line)
                self._update_public_state(line)
            return False
        if stream in PLAYER_IDS:
            for line in clean_lines:
                if line.startswith("|request|"):
                    request = _decode_request_line(line)
                    side = request.get("side") if isinstance(request.get("side"), Mapping) else {}
                    side_id = side.get("id") if isinstance(side, Mapping) else None
                    if side_id == stream:
                        self._latest_requests[stream] = request
                        self._first_requests.setdefault(stream, request)
                        self._request_history[stream].append(_json_clone_mapping(request))
                        self._lines.append(line)
        return False

    def _update_public_state(self, line: str) -> None:
        if line.startswith("|turn|"):
            try:
                self._latest_turn = int(line.split("|", 2)[2])
            except (IndexError, ValueError):
                pass
            return
        if line.startswith("|win|"):
            winner_name = line.split("|", 2)[2] if len(line.split("|", 2)) >= 3 else ""
            self._terminal = TerminalState(winner=self._winner_slot(winner_name), turn_count=self._latest_turn)
            return
        if line == "|tie" or line.startswith("|tie|"):
            self._terminal = TerminalState(winner=None, turn_count=self._latest_turn)

    def _sync_incremental_state(self) -> None:
        """Feed newly-appended protocol lines to the persistent parser and belief engine once."""
        if len(self._lines) > self._parsed_line_count:
            self._parser.feed(self._lines[self._parsed_line_count :])
            self._parsed_line_count = len(self._lines)
        events = self._parser.public_events
        if len(events) > self._belief_fed_count:
            for event in events[self._belief_fed_count :]:
                self._belief_engine.ingest_event(event)
            self._belief_fed_count = len(events)

    def _state_for_player(
        self,
        player: PlayerId,
        *,
        root_puct_branch_observation: bool = False,
    ) -> PlayerRelativeBattleState:
        if player not in PLAYER_IDS:
            raise ValueError(f"player must be one of {', '.join(PLAYER_IDS)}; got {player!r}.")
        sync_started_at = time.perf_counter() if root_puct_branch_observation else None
        try:
            self._sync_incremental_state()
        finally:
            if sync_started_at is not None:
                self._root_puct_branch_observation_incremental_sync_seconds += max(
                    0.0, time.perf_counter() - sync_started_at
                )
                self._root_puct_branch_observation_incremental_sync_count += 1

        snapshot_started_at = time.perf_counter() if root_puct_branch_observation else None
        try:
            replay = self._parser.snapshot()
        finally:
            if snapshot_started_at is not None:
                self._root_puct_branch_observation_replay_snapshot_seconds += max(
                    0.0, time.perf_counter() - snapshot_started_at
                )
                self._root_puct_branch_observation_replay_snapshot_count += 1
        # Turn-merged (v2.2/v3) specs need the merged stream populated alongside the
        # per-action one (which stays the Tier-2 annotation substrate + pinned-bit source).
        turn_merged = (
            self.config.observation_spec.schema_version
            in TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS
        )
        def _normalize() -> PlayerRelativeBattleState:
            normalize_started_at = time.perf_counter() if root_puct_branch_observation else None
            try:
                return normalize_for_player(
                    replay,
                    player_id=player,
                    configured_showdown_slot=player,
                    format_id=self._observation_format_id,
                    belief_engine=self._belief_engine,
                    include_turn_merged=turn_merged,
                )
            finally:
                if normalize_started_at is not None:
                    self._root_puct_branch_observation_player_state_normalization_seconds += max(
                        0.0, time.perf_counter() - normalize_started_at
                    )
                    self._root_puct_branch_observation_player_state_normalization_count += 1

        state = _normalize()
        annotation_started_at = time.perf_counter() if root_puct_branch_observation else None
        try:
            tracker = self._tier2_tracker_for(player)
            investment_tracker = self._investment_tracker_for(player)

            if tracker is not None:
                state = replace(
                    state,
                    transition_tokens=tracker.annotate(
                        replay, state.transition_tokens, self._belief_engine
                    ),
                )
            if investment_tracker is not None:
                codes = investment_tracker.observe(
                    replay, state.transition_tokens, self._belief_engine
                )
                if codes:
                    state = replace(
                        state,
                        transition_tokens=tuple(
                            replace(token, investment=codes[index]) if index in codes else token
                            for index, token in enumerate(state.transition_tokens)
                        ),
                    )
            # NO REFRESH HERE, deliberately. An earlier version re-derived the player view
            # whenever a producer narrowed, on the reasoning that the view snapshotted a few
            # lines above was now stale and a root and its leaves would otherwise disagree.
            # That reasoning was wrong and the block was dead: deleting it produced BIT-IDENTICAL
            # encodes across 136k numeric rows with each switch on and off. `resolved_player_view`
            # does not re-summarize, and `_apply_variant_pin` runs only from the belief engine's
            # own `_upsert`/`_replace_belief` paths -- so pins are never re-applied at snapshot
            # time, which is what actually makes the observation independent of call count.
            #
            # The real consequence, which belongs in the limitations rather than being papered
            # over by a no-op: a narrowing does not reach the ENCODE until the next event that
            # re-summarizes that mon. The pin is recorded immediately and is monotone, so nothing
            # is lost -- it lands one reveal later than the conclusion.
            # v2.2: map the FINAL annotated per-action stream (tier2 residual/CB +
            # investment codes) onto the merged sub-blocks; the per-action stream stays
            # the annotation substrate and the per-mon pinned-surface derivation source.
            if turn_merged and (tracker is not None or investment_tracker is not None):
                from .turn_merged import annotate_turn_merged_tokens

                state = replace(
                    state,
                    turn_merged_tokens=annotate_turn_merged_tokens(
                        state.turn_merged_tokens, state.transition_tokens
                    ),
                )
        finally:
            if annotation_started_at is not None:
                self._root_puct_branch_observation_state_annotation_seconds += max(
                    0.0, time.perf_counter() - annotation_started_at
                )
                self._root_puct_branch_observation_state_annotation_count += 1
        return state

    def tier2_residuals_active(self) -> bool:
        """Whether this env populates Tier-2 residuals into transition tokens.

        Requires both the encode-time mask (checkpoint-latched via
        ``env_config_from_checkpoint_provenance``) AND the candidate-set source — without
        candidate variants every strike is unassessable, so the tracker is skipped
        outright and encodes stay byte-identical to a pre-#505 pipeline.
        """
        return bool(self.config.feature_masks.tier2_residuals) and self._belief_set_source is not None

    def _tier2_tracker_for(self, player: PlayerId) -> Tier2LiveTracker | None:
        if not self.tier2_residuals_active():
            return None
        tracker = self._tier2_trackers.get(player)
        if tracker is not None:
            return tracker
        request = self._first_requests.get(player) or self._latest_requests.get(player)
        if request is None:
            return None
        own_team = own_team_from_request(request)
        if not own_team:
            return None
        root = self.config.resolved_showdown_root()
        dex = load_showdown_dex_cached(root)
        tracker = Tier2LiveTracker(
            perspective_slot=player,
            own_team=own_team,
            dex=dex,
            whitelist=cb_whitelist_for_source(self._belief_set_source, dex),
            # The CB conclusion rides the SAME belief-narrowing switch as the investment
            # one rather than getting a third mask. Both are "a tier2 conclusion mutates
            # the Tier-1 candidate set", both perturb the same frozen legacy belief columns
            # in every schema, and a cache whose metadata says narrowing was off must mean
            # neither producer ran — one provenance bit for one class of distribution shift.
            narrow_belief_candidates=self.investment_belief_narrowing_active(),
        )
        self._tier2_trackers[player] = tracker
        return tracker

    def investment_active(self) -> bool:
        """Whether this env populates defender-side investment codes into tokens.

        Requires the tier2 channel (mask + candidate-set source) AND the separate
        tier2_investment provenance mask — default off, so existing pipelines encode
        byte-identically until v2.1 training adopts the column.
        """
        return self.tier2_residuals_active() and bool(self.config.feature_masks.tier2_investment)

    def investment_belief_narrowing_active(self) -> bool:
        """Whether TIER-2 CONCLUSIONS narrow the shared belief candidate sets.

        A THIRD switch, deliberately independent of ``tier2_investment``: that one governs
        the reserved investment COLUMN, this one governs BELIEF STATE. Narrowing moves the
        candidate-set count and uncertainty columns, which exist in every schema, so it must
        never ride on the column's switch. It still needs the tier2 channel + the
        candidate-set source, because without candidate variants there is nothing to narrow
        and no strike is assessable at all.

        Named for the investment inference that introduced it, but it now gates BOTH
        producers — the defender-side investment pin (``InvestmentLiveTracker``) and the
        attacker-side Choice Band conclusion (``Tier2LiveTracker``). They share one bit on
        purpose: the provenance question a cache's metadata has to answer is "did any tier2
        conclusion touch the Tier-1 candidate sets", and that is a single class of input
        distribution shift, not two.
        """
        return self.tier2_residuals_active() and bool(
            self.config.feature_masks.investment_belief_narrowing
        )

    def item_belief_narrowing_active(self) -> bool:
        """Whether PROTOCOL-CERTAIN item facts narrow the shared belief candidate sets.

        A FOURTH switch, and deliberately not gated on the tier2 channel the way
        ``investment_belief_narrowing`` is. That gate is right for the investment pins, which
        cannot exist without the damage inference running; these narrowings are read straight
        off ``|-enditem|``/``|-item|``/``|move|`` lines by the belief engine itself, so riding
        tier2 would make them silently inert on a k0 arm for no mechanical reason. The
        candidate-set source IS required: with no variants there is nothing to narrow.
        """
        return self.config.belief_set_source_enabled() and bool(
            self.config.feature_masks.item_belief_narrowing
        )

    def _investment_tracker_for(self, player: PlayerId) -> InvestmentLiveTracker | None:
        # The tracker exists when EITHER consumer of its conclusions is on: the reserved
        # column (tier2_investment) or the belief narrowing. With both off no tracker is
        # built and the env is byte-identical to a pre-investment pipeline.
        if not self.investment_active() and not self.investment_belief_narrowing_active():
            return None
        tracker = self._investment_trackers.get(player)
        if tracker is not None:
            return tracker
        request = self._first_requests.get(player) or self._latest_requests.get(player)
        if request is None:
            return None
        own_team = own_team_from_request(request)
        if not own_team:
            return None
        dex = load_showdown_dex_cached(self.config.resolved_showdown_root())
        tracker = InvestmentLiveTracker(
            perspective_slot=player,
            own_team=own_team,
            dex=dex,
            narrow_belief_candidates=self.investment_belief_narrowing_active(),
        )
        self._investment_trackers[player] = tracker
        return tracker

    def _winner_slot(self, winner_name: str) -> PlayerId | None:
        self._sync_incremental_state()
        players = self._parser.players
        # Defensive hardening (audit: reward-FINDINGS.md "Latent risk"). A real
        # ``|win|<name>`` must resolve to exactly one seat. Two seats sharing a
        # username would make the first-match below attribute every win to the
        # first slot (silent, catastrophic mis-attribution of the value target);
        # an unmapped winner name would silently downgrade a real win to a 0/0
        # draw. Both are impossible in default self-play (distinct
        # "PokeZero p1"/"PokeZero p2", players map populated at battle start), so
        # assert them — a future same-username or unmapped-name config then fails
        # loudly instead of corrupting the reward label. Assert-only: no value
        # change on the reachable path.
        p1_name = players.get("p1")
        p2_name = players.get("p2")
        assert p1_name is None or p1_name != p2_name, (
            "player usernames must be distinct to attribute a win to a seat; both "
            f"p1 and p2 are {p1_name!r}"
        )
        slot = next((s for s, name in players.items() if name == winner_name), None)
        assert slot is not None, (
            f"|win|{winner_name!r} did not resolve to a player slot; "
            f"players={dict(players)!r}"
        )
        return slot

    def _rewards(self) -> dict[PlayerId, float]:
        if self._terminal is None:
            return {"p1": 0.0, "p2": 0.0}
        if self._terminal.winner is None:
            return {"p1": 0.0, "p2": 0.0}
        return {
            "p1": 1.0 if self._terminal.winner == "p1" else -1.0,
            "p2": 1.0 if self._terminal.winner == "p2" else -1.0,
        }

    def _timeout_message(self) -> str:
        return f"Timed out waiting for BattleStream bridge output. {self._bridge_exit_message()}"

    def _bridge_exit_message(self) -> str:
        if self._process is not None and self._process.poll() is not None:
            stderr = "\n".join(self._stderr_lines[-20:])
            suffix = f" Stderr:\n{stderr}" if stderr else ""
            return f"BattleStream bridge exited with status {self._process.returncode}.{suffix}"
        return "BattleStream bridge is still running."


def showdown_seed_from_int(seed: int) -> str:
    digest = hashlib.sha256(str(int(seed)).encode("utf-8")).digest()
    parts = [int.from_bytes(digest[index : index + 2], "big") for index in range(0, 8, 2)]
    return ",".join(str(part) for part in parts)


def requested_players_from_requests(requests: Mapping[PlayerId, Mapping[str, Any]]) -> tuple[PlayerId, ...]:
    return tuple(player for player in PLAYER_IDS if _is_actionable_request(requests.get(player)))


def scenario_public_protocol_lines(
    state: Mapping[str, Any],
    requests: Mapping[PlayerId, Mapping[str, Any]],
) -> tuple[str, ...]:
    """Build a synthetic, fully revealed public root for a scenario-materialized battle.

    The battle bridge owns the true simulator state. This compact disclosure prefix gives the
    normal parser and belief engine the same active identities, HP, moves, abilities, items, and
    PP ledger without pretending those synthetic reveal actions were real transition history.
    Callers retain the belief result but clear the fabricated events before encoding history.
    """

    sides = state.get("sides")
    if not isinstance(sides, Mapping):
        raise LocalShowdownError("Scenario materialization state is missing sides.")
    lines: list[str] = [
        f"|player|p1|{_DEFAULT_PLAYER_NAMES['p1']}|",
        f"|player|p2|{_DEFAULT_PLAYER_NAMES['p2']}|",
    ]
    active_rows: dict[PlayerId, Mapping[str, Any]] = {}
    for player in PLAYER_IDS:
        side = sides.get(player)
        if not isinstance(side, Mapping):
            raise LocalShowdownError(f"Scenario materialization state is missing {player}.")
        pokemon_rows = side.get("pokemon")
        active_slot = side.get("activeSlot")
        if not isinstance(pokemon_rows, list) or not isinstance(active_slot, int):
            raise LocalShowdownError(f"Scenario materialization state has malformed {player} Pokemon.")
        if active_slot < 0 or active_slot >= len(pokemon_rows):
            raise LocalShowdownError(f"Scenario materialization state has an invalid {player} active slot.")
        opponent = "p2" if player == "p1" else "p1"
        target = f"{opponent}a: scenario-target"
        for row in pokemon_rows:
            if not isinstance(row, Mapping):
                raise LocalShowdownError(f"Scenario materialization state has an invalid {player} Pokemon row.")
            species = _scenario_protocol_field(row.get("species"), "species")
            details = _scenario_protocol_field(row.get("details"), "details")
            hp = row.get("hp")
            max_hp = row.get("maxHp")
            fainted = row.get("fainted")
            if not isinstance(hp, int) or not isinstance(max_hp, int) or not isinstance(fainted, bool):
                raise LocalShowdownError(f"Scenario materialization state has invalid HP for {player} {species}.")
            condition = _scenario_condition(row, player, species)
            ident = f"{player}a: {species}"
            lines.append(f"|switch|{ident}|{details}|{condition}")
            ability = _scenario_protocol_optional_field(row.get("ability"), "ability")
            if ability:
                lines.append(f"|-ability|{ident}|{ability}")
            item = _scenario_protocol_optional_field(row.get("item"), "item")
            if item:
                lines.append(f"|-item|{ident}|{item}")
            moves = row.get("moves")
            if not isinstance(moves, list):
                raise LocalShowdownError(f"Scenario materialization state has invalid moves for {player} {species}.")
            for move in moves:
                if not isinstance(move, Mapping):
                    raise LocalShowdownError(f"Scenario materialization state has an invalid move for {player} {species}.")
                move_id = _scenario_protocol_field(move.get("id"), "move id")
                pp = move.get("pp")
                max_pp = move.get("maxPp")
                if not isinstance(pp, int) or not isinstance(max_pp, int) or pp < 0 or max_pp < pp:
                    raise LocalShowdownError(f"Scenario materialization state has invalid PP for {player} {move_id}.")
                # Plain ``|move|`` lines charge one PP in the normal belief ledger. Emit the
                # exact number of charged uses first, then a Sleep-Talk-called synthetic reveal:
                # that protocol form proves the move belongs to the set while charging none of
                # the callee's PP. This preserves exact opponent PP even when it is still full.
                for _ in range(max_pp - pp):
                    lines.append(f"|move|{ident}|{move_id}|{target}")
                lines.append(f"|move|{ident}|{move_id}|{target}|[from] move: Sleep Talk")
        active_row = pokemon_rows[active_slot]
        if not isinstance(active_row, Mapping):
            raise LocalShowdownError(f"Scenario materialization state has no active {player} row.")
        active_rows[player] = active_row
    for player in PLAYER_IDS:
        active_row = active_rows[player]
        species = _scenario_protocol_field(active_row.get("species"), "species")
        details = _scenario_protocol_field(active_row.get("details"), "details")
        hp = active_row.get("hp")
        max_hp = active_row.get("maxHp")
        fainted = active_row.get("fainted")
        if not isinstance(hp, int) or not isinstance(max_hp, int) or not isinstance(fainted, bool):
            raise LocalShowdownError(f"Scenario materialization state has invalid active HP for {player}.")
        condition = _scenario_condition(active_row, player, species)
        lines.append(f"|switch|{player}a: {species}|{details}|{condition}")
    field = state.get("field")
    if not isinstance(field, Mapping):
        raise LocalShowdownError("Scenario materialization state is missing field conditions.")
    weather = _scenario_protocol_optional_field(field.get("weather"), "weather")
    if weather:
        suffix = "|[from] ability: scenario" if field.get("permanent") is True else ""
        lines.append(f"|-weather|{weather}{suffix}")
    for player in PLAYER_IDS:
        side = sides[player]
        assert isinstance(side, Mapping)
        side_conditions = side.get("sideConditions")
        if not isinstance(side_conditions, Mapping):
            raise LocalShowdownError(f"Scenario materialization state has invalid {player} side conditions.")
        for raw_name, raw_value in sorted(side_conditions.items()):
            name = _scenario_protocol_field(raw_name, "side condition")
            if not isinstance(raw_value, int) or raw_value < 1:
                raise LocalShowdownError(
                    f"Scenario materialization state has invalid {player} {name}."
                )
            repeats = raw_value if normalize_id(name) == "spikes" else 1
            for _ in range(repeats):
                lines.append(f"|-sidestart|{player}: scenario|{name}")
        active_volatiles = side.get("activeVolatiles")
        if not isinstance(active_volatiles, list):
            raise LocalShowdownError(f"Scenario materialization state has invalid {player} volatiles.")
        species = _scenario_protocol_field(active_rows[player].get("species"), "species")
        for volatile in active_volatiles:
            if not isinstance(volatile, Mapping):
                raise LocalShowdownError(
                    f"Scenario materialization state has an invalid {player} volatile."
                )
            name = _scenario_protocol_field(volatile.get("id"), "volatile")
            lines.append(f"|-start|{player}a: {species}|{name}")
    lines.append(f"|turn|{int(state.get('turn') or 1)}")
    for player in PLAYER_IDS:
        request = requests.get(player)
        if not isinstance(request, Mapping):
            raise LocalShowdownError(f"Scenario materialization did not return a {player} request.")
        lines.append(f"|request|{json.dumps(request, separators=(',', ':'))}")
    return tuple(lines)


def _scenario_condition(row: Mapping[str, Any], player: str, species: str) -> str:
    hp = row.get("hp")
    max_hp = row.get("maxHp")
    fainted = row.get("fainted")
    if not isinstance(hp, int) or not isinstance(max_hp, int) or not isinstance(fainted, bool):
        raise LocalShowdownError(f"Scenario materialization state has invalid HP for {player} {species}.")
    if fainted:
        return "0 fnt"
    status = row.get("status")
    if not isinstance(status, Mapping):
        raise LocalShowdownError(
            f"Scenario materialization state has invalid status for {player} {species}."
        )
    status_id = _scenario_protocol_optional_field(status.get("id"), "status")
    return f"{hp}/{max_hp}{f' {status_id}' if status_id else ''}"


def _seed_scenario_parser_state(parser: _ReplayParser, state: Mapping[str, Any]) -> None:
    """Latch exact public counters that compact synthetic protocol cannot reconstruct alone."""

    turn = state.get("turn")
    sides = state.get("sides")
    field = state.get("field")
    if not isinstance(turn, int) or not isinstance(sides, Mapping) or not isinstance(field, Mapping):
        raise LocalShowdownError("Scenario materialization returned malformed condition state.")
    weather = normalize_id(str(field.get("weather") or ""))
    turns_remaining = field.get("turnsRemaining")
    permanent = field.get("permanent")
    if not isinstance(turns_remaining, int) or not isinstance(permanent, bool):
        raise LocalShowdownError("Scenario materialization returned malformed weather state.")
    parser.weather = weather
    parser.weather_from_ability = bool(weather and permanent)
    parser.weather_set_turn = turn if weather else None
    parser.weather_upkeeps = 0 if permanent else max(0, 5 - turns_remaining)

    for player in PLAYER_IDS:
        side = sides.get(player)
        if not isinstance(side, Mapping):
            raise LocalShowdownError(f"Scenario materialization returned malformed {player} state.")
        side_conditions = side.get("sideConditions")
        if not isinstance(side_conditions, Mapping):
            raise LocalShowdownError(
                f"Scenario materialization returned malformed {player} side conditions."
            )
        parser.side_condition_counts[player] = {}
        parser.side_condition_set_turns[player] = {}
        for raw_name, raw_value in side_conditions.items():
            name = normalize_id(str(raw_name))
            if not name or not isinstance(raw_value, int) or raw_value < 1:
                raise LocalShowdownError(
                    f"Scenario materialization returned malformed {player} {raw_name}."
                )
            parser.side_condition_counts[player][name] = raw_value if name == "spikes" else 1
            if name != "spikes":
                parser.side_condition_set_turns[player][name] = turn - (5 - raw_value)

        active_volatiles = side.get("activeVolatiles")
        pokemon = side.get("pokemon")
        active_slot = side.get("activeSlot")
        if (
            not isinstance(active_volatiles, list)
            or not isinstance(pokemon, list)
            or not isinstance(active_slot, int)
            or not 0 <= active_slot < len(pokemon)
        ):
            raise LocalShowdownError(
                f"Scenario materialization returned malformed {player} active state."
            )
        by_id: dict[str, Mapping[str, Any]] = {}
        for item in active_volatiles:
            if not isinstance(item, Mapping):
                raise LocalShowdownError(
                    f"Scenario materialization returned malformed {player} volatile."
                )
            volatile_id = normalize_id(str(item.get("id") or ""))
            if not volatile_id or volatile_id in by_id:
                raise LocalShowdownError(
                    f"Scenario materialization returned duplicate or invalid {player} volatile."
                )
            by_id[volatile_id] = item
        parser.volatiles[player] = set(by_id)
        parser.confusion_elapsed[player] = int(
            by_id.get("confusion", {}).get("turnsElapsed") or 0
        )
        parser.encore_elapsed[player] = int(by_id.get("encore", {}).get("turnsElapsed") or 0)
        parser.wrap_trap_elapsed[player] = 0
        if "leechseed" in by_id:
            parser.leech_seed_source_sides[player] = "p2" if player == "p1" else "p1"
        else:
            parser.leech_seed_source_sides.pop(player, None)

        active = pokemon[active_slot]
        if not isinstance(active, Mapping):
            raise LocalShowdownError(
                f"Scenario materialization returned malformed {player} active Pokemon."
            )
        status = active.get("status")
        if not isinstance(status, Mapping):
            raise LocalShowdownError(
                f"Scenario materialization returned malformed {player} active status."
            )
        toxic = normalize_id(str(status.get("id") or "")) == "tox"
        engine_stage = int(status.get("toxicStage") or 0) if toxic else 0
        # Scenario materialization returns an ordinary action-request boundary. The parser's
        # observation convention at that boundary is one ahead of Showdown's current
        # statusState.stage. Internal value 16 preserves "current stage is already capped at
        # 15"; the model-facing feature remains clamped to 15.
        parser.toxic_stage[player] = min(16, engine_stage + 1) if toxic else 0
        parser.toxic_stage_known[player] = True
        # Scenario materialization always returns an ordinary action-request boundary. It
        # cannot carry the replay-only proof for a replacement that arrived after a prior
        # upkeep, so ensure a reused parser never retains one.
        parser.toxic_stage_zero_after_upkeep[player] = False
        parser.toxic_stage_zero_after_upkeep_expires_after_turn[player] = None
        parser.toxic_stage_zero_after_upkeep_ident[player] = None
        parser.toxic_faint_replacement_pending[player] = False
        parser.toxic_faint_replacement_expected_ident[player] = None
        parser.toxic_faint_replacement_invalid[player] = False


def _scenario_protocol_field(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LocalShowdownError(f"Scenario materialization state has an invalid {label}.")
    normalized = value.strip()
    if "|" in normalized or "\n" in normalized or "\r" in normalized:
        raise LocalShowdownError(f"Scenario materialization state has an unsafe {label}.")
    return normalized


def _scenario_protocol_optional_field(value: Any, label: str) -> str | None:
    if value is None or value == "":
        return None
    return _scenario_protocol_field(value, label)


def _start_players_payload(start_override: BattleStartOverride | None) -> dict[PlayerId, str | dict[str, str]]:
    player_teams = start_override.player_teams if start_override is not None else {}
    players: dict[PlayerId, str | dict[str, str]] = {}
    for player in PLAYER_IDS:
        name = _DEFAULT_PLAYER_NAMES[player]
        team = player_teams.get(player)
        players[player] = {"name": name, "team": team} if team else name
    return players


def _is_actionable_request(request: Mapping[str, Any] | None) -> bool:
    if not isinstance(request, Mapping):
        return False
    if request.get("wait"):
        return False
    if request.get("teamPreview"):
        return False
    force_switch = request.get("forceSwitch")
    if isinstance(force_switch, list) and any(bool(slot) for slot in force_switch):
        return True
    active = request.get("active")
    return isinstance(active, list) and bool(active)


def _decode_request_line(line: str) -> Mapping[str, Any]:
    prefix = "|request|"
    if not line.startswith(prefix):
        raise ValueError("request line must start with |request|")
    payload = json.loads(line[len(prefix) :])
    if not isinstance(payload, Mapping):
        raise ValueError("request payload must be a JSON object.")
    return payload


def _json_clone_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    cloned = json.loads(json.dumps(value, separators=(",", ":")))
    if not isinstance(cloned, Mapping):
        raise ValueError("expected JSON object clone.")
    return cloned


def _json_clone_requests(
    value: Mapping[PlayerId, Mapping[str, Any]],
) -> dict[PlayerId, Mapping[str, Any]]:
    cloned = _json_clone_mapping(value)
    return {player: request for player in PLAYER_IDS if isinstance((request := cloned.get(player)), Mapping)}


def _json_clone_request_history(
    value: Mapping[PlayerId, Sequence[Mapping[str, Any]]],
) -> dict[PlayerId, tuple[Mapping[str, Any], ...]]:
    return {
        player: tuple(_json_clone_mapping(request) for request in value.get(player, ()))
        for player in PLAYER_IDS
    }


def _public_materialization_payload(
    state: PublicBattleMaterializationState,
    *,
    deferred_opponent_actions: Mapping[PlayerId, int] | None = None,
    deferred_opponent_action_priors: Mapping[PlayerId, Sequence[float]] | None = None,
) -> dict[str, Any]:
    replay = state.replay
    sides: dict[PlayerId, dict[str, Any]] = {}
    belief_snapshot = state.belief_engine.snapshot()
    for player in PLAYER_IDS:
        rows = (
            _request_materialization_rows(state.self_request, self_move_states=state.self_move_states)
            if player == state.player_id
            else [_pokemon_materialization_row(pokemon) for pokemon in replay.public_revealed.get(player, ())]
        )
        blockers = set(replay.direct_materialization_blockers.get(player, ()))
        _apply_public_item_materialization_state(
            rows,
            belief_snapshot.side(player),
            blockers,
        )
        _apply_traced_ability_materialization_state(rows, replay.traced_ability.get(player))
        _apply_rest_sleep_provenance(rows, replay, player)
        toxic_stage = _materialization_toxic_stage(replay, player)
        if toxic_stage is None:
            # The policy can still observe status:tox with its legacy zero
            # feature, but a sampled engine world may not turn an incomplete
            # public prefix into a claimed ToxicCount=0.
            blockers.add("toxic-stage-unknown")
        sides[player] = {
            "pokemon": rows,
            "boosts": dict(replay.boosts.get(player, {})),
            "volatiles": list(replay.volatiles.get(player, ())),
            # Mean Look / Spider Web / Block: this slot's active mon is switch-locked by the
            # OPPOSING active. A separate key rather than a `volatiles` entry, because the two
            # lists have different owners and different consumers:
            #
            #   `volatiles` is `_update_volatiles`'s output, gated on TRACKED_VOLATILES -- the
            #   closed set that also enumerates the observation encoder's `volatile:<id>` vocab
            #   (randbat_vocab) and that the Node bridge's STATIC_PUBLIC_VOLATILES mirrors.
            #   Adding `trapped` there would (a) mint a v3 vocab row and unfreeze byte-frozen
            #   v3 tensors, which already carry this fact as NUMERIC_MEANLOOK_TRAP, and (b) make
            #   `materialize` throw, since the bridge cannot rebuild the linked `trapper`
            #   volatile on the source mon.
            #
            #   `meanlook_trap` is the parser's own per-slot tracker for `|-activate|SLOT|trapped`
            #   (spec v3 change 8). It reaches the OBSERVATION already; this is the world lane's
            #   twin, exactly as `truantPhase` below is the world lane's twin of the truant
            #   tracker. engine_world turns it into the engine's TRAPPED volatile; the bridge
            #   ignores the key, so the direct path is unchanged.
            #
            # Without it the world builder never saw a move trap: `sides[self].volatiles` was
            # `[]` on every Mean Look turn, so `_require_world_reproduces_trap` found no cause
            # for the request's disclosed `trapped` flag and refused the decision.
            "meanlookTrap": bool(replay.meanlook_trap.get(player, False)),
            # A fresh Substitute is exact. Deterministic fixed-damage chronology
            # carries world-portable depletion; other surviving hits remain
            # explicitly unknown so engine_world declines them.
            "substituteHealthState": replay.substitute_health_state.get(player, "absent"),
            "substituteDepletion": replay.substitute_depletion.get(player),
            "substituteMinDepletion": replay.substitute_min_depletion.get(player, 0),
            "materializationBlockers": sorted(blockers),
            # At an ordinary request the observation feature names the next residual; at the
            # post-upkeep forced-switch boundary it names the residual just paid. The helper
            # converts both public boundaries into Showdown's current statusState.stage.
            "toxicStage": toxic_stage,
            # Consecutive SUCCESSFUL stall-move uses (Protect/Detect/Endure — gen3
            # shares one `stall` volatile). The parser already derives this from
            # public protocol alone; the engine prices the NEXT attempt at
            # 0.5 ** count, so the count passes through with NO boundary offset
            # (unlike toxicStage above, whose feature runs one residual ahead).
            "stallCounter": int(replay.stall_counter.get(player, 0)),
            # Public last EXECUTED move (or the "switch" sentinel), transcribed from the
            # same truth table the engine already obeys. The engine needs this for far
            # more than the encore lock it was previously derived for: Encore's own
            # onStart READS it, so a world that omits it makes Encore fail outright
            # (Showdown: `if (!move) return false`) and every downstream Encore effect --
            # duration, the move-slot lock, and the same-turn redirect the engine already
            # implements -- silently never happens.
            "lastUsedMove": replay.last_used_move.get(player) or "",
            # gen3 Truant loaf parity for the active mon: True = loafs on its next move
            # attempt, False = acts, None = no holder OR a genuinely unknown phase. Unknown
            # includes a truncated prefix and a full-prefix Trace acquisition whose residual
            # event-queue membership cannot be recovered from public line order.
            #
            # This replaces a "moved last round -> loafs now" proxy computed downstream. The
            # sim's bit is a free-running toggle flipped at EVERY residual regardless of what
            # the mon did, so the proxy inverts permanently the first time a holder is kept
            # from moving by sleep, paralysis, flinch, freeze, recharge or a switch.
            "truantPhase": replay.truant_phase.get(player),
            # Live in-battle retype of the ACTIVE mon, which the species token cannot
            # express. The parser has produced this since the v3 obs work but only the
            # OBSERVATION path consumed it (`_apply_live_type_override`); the world was
            # still built from base Pokedex types, so a Kecleon whose Color Change had
            # retyped it reached the engine as plain Normal. That is wrong in BOTH
            # directions at once: it grants STAB the mon no longer has when it attacks,
            # and it prices incoming type effectiveness against the wrong defensive type.
            #
            # Format is the parser's: "type:<Type>" (Color Change -- the payload IS the
            # type) or "forme:<forme>" (Castform Forecast -- unresolved, the forme's type
            # is dex-resolved by the consumer). engine_world consumes only the "type:"
            # form; see the precedence note there for why Forecast stays derived.
            "liveTypeOverride": replay.live_type_override.get(player) or "",
            "sideConditions": dict(replay.side_condition_counts.get(player, {})),
            "sideConditionSetTurns": dict(replay.side_condition_set_turns.get(player, {})),
        }
    deferred_actions = dict(deferred_opponent_actions or {})
    deferred_priors = {
        player: tuple(values)
        for player, values in (deferred_opponent_action_priors or {}).items()
    }
    deferred_player = state.deferred_opponent_action_player
    if deferred_actions and deferred_priors:
        raise ValueError("Direct materialization received both a deferred action and deferred move priors.")
    if deferred_actions and set(deferred_actions) != {deferred_player}:
        raise ValueError("Direct materialization received an unexpected deferred opponent action.")
    if deferred_priors and set(deferred_priors) != {deferred_player}:
        raise ValueError("Direct materialization received unexpected deferred opponent move priors.")
    if any(
        isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < MOVE_ACTION_COUNT
        for action in deferred_actions.values()
    ):
        raise ValueError("Direct materialization received an invalid deferred opponent action.")
    if any(
        len(priors) != MOVE_ACTION_COUNT
        or any(
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value < 0.0
            for value in priors
        )
        or sum(priors) <= 0.0
        for priors in deferred_priors.values()
    ):
        raise ValueError("Direct materialization received invalid deferred opponent move priors.")
    return {
        "turn": replay.turn_number,
        "weather": replay.weather,
        "weatherSetTurn": replay.weather_set_turn,
        "weatherFromAbility": replay.weather_from_ability,
        "futureSight": dict(replay.future_sight),
        # A Wish is a public, one-turn slot condition.  The replay parser retains its
        # set turn for observation features, including harmless expired entries when
        # the landing Pokemon was already at full HP, so only expose a still-pending
        # Wish to the direct constructor.
        "wishSetTurns": _pending_wish_set_turns(replay),
        "leechSeedSourceSides": _active_leech_seed_source_sides(replay),
        # A Baton Pass declaration is public and its forced switch has not yet resolved. The
        # bridge needs the source-effect id so Showdown preserves the carried battle state.
        "pendingBatonPassSides": _pending_baton_pass_sides(replay, state),
        # The action has already been committed in the interrupted simultaneous turn but is not
        # yet protocol-visible. It is supplied by the opponent-action predictor, never the live
        # battle, and the bridge restores it before the actor's forced switch resolves.
        "deferredOpponentActions": deferred_actions,
        "deferredOpponentActionPriors": {
            player: list(priors) for player, priors in deferred_priors.items()
        },
        "selfPlayer": state.player_id,
        # The actor's request exposes the active-first team permutation used for both future
        # observations and `switch N` choices. This is player-known state, unlike the opponent's
        # party order, and lets the constructed simulator preserve it beyond the first boundary.
        "selfTeamOrder": [row["species"] for row in sides[state.player_id]["pokemon"]],
        "selfRequestKind": _request_materialization_kind(state.self_request),
        "selfActiveMoves": _request_active_moves(state.self_request),
        "selfActiveRequestState": _request_active_materialization_state(state.self_request),
        # The actor's request history retains exact PP state for Pokemon that were previously
        # active. If a used benched Pokemon has no such request-known snapshot, fail closed.
        "selfBenchedMoveHistory": _has_self_benched_move_history(state),
        "sides": sides,
    }


def _materialization_toxic_stage(replay: ShowdownReplayState, player: PlayerId) -> int | None:
    """Return the engine's pre-tick Toxic counter for the next residual.

    ``None`` is intentional: a snapshot that lacks the public provenance for
    an active Toxic counter is not allowed to silently materialize as stage 0.
    Missing provenance on a clean active side is a harmless zero, which keeps
    legacy snapshots from blocking worlds that have no Toxic counter.
    """

    def provenance_value(name: str, default: Any = None) -> tuple[bool, Any]:
        values = getattr(replay, name, None)
        if not isinstance(values, Mapping) or player not in values:
            return False, default
        return True, values[player]

    active = replay.public_active.get(player)
    proof_present, zero_after_upkeep = provenance_value("toxic_stage_zero_after_upkeep")
    if proof_present and type(zero_after_upkeep) is not bool:
        return None
    if not _is_current_public_active(active):
        return None
    condition = getattr(active, "condition", None)
    if not isinstance(condition, str):
        return None
    active_is_toxic = "tox" in condition.split()
    if not active_is_toxic:
        return None if zero_after_upkeep is True else 0
    post_upkeep_window = getattr(replay, "post_upkeep_window", None)
    if type(post_upkeep_window) is not bool:
        return None
    known_present, known = provenance_value("toxic_stage_known")
    if not known_present or known is not True:
        return None
    stage_present, tracked_stage = provenance_value("toxic_stage")
    if not stage_present or type(tracked_stage) is not int:
        return None
    if tracked_stage == 0:
        # A poisoned replacement that entered after upkeep missed the residual
        # that just ran. Its next Toxic tick is stage 1, so the engine's
        # pre-tick counter is correctly zero. No other active-Toxic zero has
        # enough public chronology to distinguish that fact from an incomplete
        # prefix, and therefore remains fail-closed.
        if not proof_present or zero_after_upkeep is not True:
            return None
        ident_present, proof_ident = provenance_value("toxic_stage_zero_after_upkeep_ident")
        deadline_present, deadline = provenance_value(
            "toxic_stage_zero_after_upkeep_expires_after_turn"
        )
        invalid_present, invalid = provenance_value("toxic_faint_replacement_invalid")
        pending_present, pending = provenance_value("toxic_faint_replacement_pending")
        expected_present, expected_ident = provenance_value(
            "toxic_faint_replacement_expected_ident"
        )
        active_ident = getattr(active, "ident", None)
        turn_number = getattr(replay, "turn_number", None)
        if (
            not ident_present
            or not isinstance(proof_ident, str)
            or not _is_active_protocol_ident(proof_ident)
            or not proof_ident.startswith(f"{player}a: ")
            or active_ident != proof_ident
            or not deadline_present
            or type(deadline) is not int
            or deadline < 1
            or type(turn_number) is not int
            or turn_number < 0
            or deadline != turn_number + (1 if post_upkeep_window else 0)
            or not invalid_present
            or invalid is not False
            or not pending_present
            or pending is not False
            or not expected_present
            or expected_ident is not None
        ):
            return None
        return 0
    if zero_after_upkeep is True:
        return None
    if not 1 <= tracked_stage <= 16:
        return None
    if post_upkeep_window is True:
        # Residuals have run but the next |turn| line has not. The raw public stage is the
        # multiplier just paid, which is the counter needed for the NEXT tick. The engine's
        # counter is pre-tick and must stay at 14 once Showdown's stage has saturated at 15.
        return min(14, max(0, tracked_stage))
    # At an ordinary action request, |turn| has advanced the public feature to the multiplier
    # that will be charged at the next residual; the simulator still holds the prior count.
    # Sentinel 16 distinguishes an already-saturated current stage from raw 15's current 14.
    # Both produce pre-tick counter 14: the vendored engine computes stage = counter + 1.
    return min(14, max(0, tracked_stage - 1))


def _pending_wish_set_turns(replay: ShowdownReplayState) -> dict[str, int]:
    """Return only Wish declarations that must still resolve at this boundary."""

    return {
        player: int(set_turn)
        for player, set_turn in replay.wish_set_turns.items()
        if player in PLAYER_IDS
        and isinstance(set_turn, int)
        # Forced switches can interrupt the declaration turn before its residual
        # phase; ordinary requests arrive on the next turn. Older entries can
        # remain in the public fold if the full-HP landing emitted no heal line,
        # but they are no longer a live simulator condition.
        and replay.turn_number - set_turn in {0, 1}
    }


def _pending_baton_pass_sides(
    replay: ShowdownReplayState,
    state: PublicBattleMaterializationState,
) -> list[PlayerId]:
    """Return the actor's Baton Pass only at its corresponding forced-switch boundary."""

    if _request_materialization_kind(state.self_request) != "force-switch":
        return []
    return [state.player_id] if state.player_id in replay.pending_baton_pass else []


def _active_leech_seed_source_sides(replay: ShowdownReplayState) -> dict[str, str]:
    """Return public Leech Seed provenance only for targets still carrying the effect."""

    source_sides: dict[str, str] = {}
    for target_side, source_side in replay.leech_seed_source_sides.items():
        if (
            target_side in PLAYER_IDS
            and source_side in PLAYER_IDS
            and target_side != source_side
            and "leechseed" in replay.volatiles.get(target_side, ())
        ):
            source_sides[target_side] = source_side
    return source_sides


def _pokemon_materialization_row(pokemon: ShowdownPokemon) -> dict[str, Any]:
    return {
        "species": pokemon.species,
        "details": pokemon.details,
        "condition": pokemon.condition,
        "active": pokemon.active,
    }


def _apply_traced_ability_materialization_state(
    rows: list[dict[str, Any]],
    traced: str | None,
) -> None:
    """Attach the ability the ACTIVE mon is currently borrowing via Trace.

    A sampled set's ability is the battle-start assignment, and for almost every mon that
    stays true. Trace does not: the holder publicly copies the opponent's ability, and a world
    rebuilt from the sampled set hands the engine `TRACE`, playing the mon without the copied
    ability at all -- damaging straight through a traced Flash Fire immunity, for instance.

    **Only the active mon, and only the CURRENT trace.** The first version of this read
    `belief.revealed_ability`, which is the right field for an ability a mon merely revealed
    (persistent) and the wrong one for a traced ability (transient: re-fired on every
    switch-in, dropped on switch-out). That version stamped the LAST ability the mon had ever
    traced, which handed a Gardevoir `levitate` from an earlier switch-in and silently granted
    it Spikes immunity -- turning a fix into a two-row regression. The parser now tracks the
    live copy and clears it on switch-out; this consumes that.

    Only the ability FIELD is seeded. gen3 does not fire the copied ability's Start event on
    acquisition (#962 patch 32), so no activation is simulated here.
    """

    if not traced:
        return
    for row in rows:
        if row.get("active"):
            row["revealedAbility"] = normalize_id(str(traced))


def _apply_public_item_materialization_state(
    rows: list[dict[str, Any]],
    beliefs: Sequence[RevealedPokemonBelief],
    blockers: set[str],
) -> None:
    """Attach only protocol-confirmed live item state to direct-world rows.

    A sampled set's item describes the battle-start assignment. Trick can publicly replace
    that item later, so starting the sampled world alone silently recreates the old holder.
    The belief engine records an audited ``current_public_item`` only for the corresponding
    protocol surface. Removals and unaudited mutations intentionally remain blockers: this
    constructor has no complete item-history representation, and guessing would create a
    mechanically false world.
    """

    rows_by_species: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        species = row.get("species")
        if isinstance(species, str):
            rows_by_species.setdefault(_materialization_identifier(species), []).append(row)

    for belief in beliefs:
        if not (belief.item_mutated or belief.item_removed):
            continue
        species = belief.species
        matching_rows = rows_by_species.get(_materialization_identifier(species), ())
        if len(matching_rows) != 1:
            blockers.add(f"item-state-ambiguous:{species or 'unknown'}")
            continue
        if belief.item_removed:
            blockers.add(f"item-state-removed:{species}")
            continue
        current_item = belief.current_public_item
        if not isinstance(current_item, str) or not current_item.strip():
            blockers.add(f"item-state-unconfirmed:{species}")
            continue
        matching_rows[0]["currentItem"] = current_item


def _mark_legacy_rest_refund_pending(row: dict[str, Any]) -> None:
    """Also set the PRE-SPLIT flag, for rows read back by a pre-split checkout.

    The split renamed this flag so that an old row could not be miscounted as
    producer B. Renaming alone creates the mirror hazard, which is worse: a row
    written HERE and replayed by a checkout older than the split matches none of
    that checkout's branches, falls through to ``approximate_sleep_turns``, and
    silently builds ``rest_turns=0`` instead of refusing. Stored corpora keep
    ``public_materialization`` verbatim and all four replay scripts
    (``leaf_root_parity``, ``bench_leaf_search``, ``leaf_vs_reality``,
    ``prior_mapping_assert``) pass ``approximate_sleep_turns=True``, so that is a
    silently wrong world, not a mislabelled refusal.

    Writing BOTH keys keeps every direction honest, because the three cases stay
    distinguishable by which keys are present:

        new row + new code  -> a producer flag is checked first; this one is never reached
        new row + OLD code  -> sees this flag and refuses, exactly as before the split
        OLD row + new code  -> carries ONLY this flag, so it gets the third legacy code

    That last line is why the legacy check must stay last, and why no live path may
    ever emit this flag on its own. Both are pinned by tests, not just by this comment:
    reordering the checks leaves the rest of the suite green.

    THIS IS A WORKAROUND, and the exit has been available the whole time. It exists only
    because a stored row cannot say which annotator wrote it — yet `golden_corpus` has
    carried a `record_type: "header"` record with a writer-owned-field guard since
    `aad856c5`, long before this PR. So the workaround was never blocked on missing
    infrastructure. #1091 (`39d26f4a`) is the *precedent* worth copying, not the
    enabler: it versions a corpus contract in that existing header (selectable
    observation schema) rather than inferring it. Stamp the annotator era the same way
    and BOTH this dual-write and the third reason code become unnecessary.
    """

    row["restSleepRefundPending"] = True


def _apply_rest_sleep_provenance(
    rows: list[dict[str, Any]],
    replay: ShowdownReplayState,
    player: PlayerId,
) -> None:
    """Attach the public Rest attempt count to each Rest-asleep mon's direct-world row.

    ``ShowdownReplayState.rest_sleep_counts`` records, per sleeping mon, that the sleep
    came from its OWN Rest and how many public move attempts have occurred. A trailing
    Sleep Talk/Snore run is held separately as ``rest_sleep_skipped_turns`` because Gen 3
    refunds it on switch-in. ``rest_sleep_refunded_turns`` retains refunds already applied
    by an earlier switch-in. All three values cross to the row: construction combines them
    with the resolved world's ability, because Early Bird burns two timer units per attempt
    while each skippedTime refund restores one.

    An ACTIVE sleeper with a nonzero skipped run, an attempt whose direct outcome is absent in a
    truncated protocol prefix, or malformed/inconsistent public Rest state cannot be represented
    by the Rust world (which has no ``skippedTime`` field). Its row carries an explicit marker so
    the downstream constructor fails closed before approximate sleep handling can invent a
    generic timer.

    Until this the only sleep provenance crossing into world construction was the pair of
    aggregate booleans (``self_sleep_clause_blocks`` / ``opponent_sleep_clause_blocks``),
    which say a side HAS a clause-engaging sleeper but never which mon — so a Rest-asleep
    mon arrived at the constructor indistinguishable from an opponent-induced one, and
    gen3's Sleep Clause Mod exempts exactly the first kind.

    KEY RECONSTRUCTION — the same trap family as the Heal Bell benched-cure caution.
    The tracker keys on the ident NAME (``_induced_sleep_victim_key``), because the lines
    that CLEAR an entry include Heal Bell's benched ``|-curestatus|p2: Name|slp|[silent]``
    form, whose position-less ident cannot be resolved to a species through
    ``public_active``. A materialization row, by contrast, carries the SPECIES. The two
    coincide here only because gen3 randbats runs under Nickname Clause and never
    nicknames anything: reconstructing the key from the row's species is therefore exact
    for this format and WRONG the moment nicknames are allowed. A cosmetic base-form
    fallback (``Unown`` -> ``Unown-Z``) is permitted only when it is one tracker key to
    one sleeping row. A missing or ambiguous correspondence marks the affected known
    sleeper rows unrepresentable; it must never fall through to generic sleep
    approximation. Do not extend this to a nicknamed format without carrying the ident
    through the row.

    Only a mon that is in the Rest map AND absent from the opposing side's induced-victim
    set is annotated, so the emitted field means exactly "this sleep is its own Rest's"
    and the consumer needs no second lookup to rule out an induced sleeper.
    """

    counts = replay.rest_sleep_counts
    refunded_turns = replay.rest_sleep_refunded_turns
    skipped_turns = replay.rest_sleep_skipped_turns
    pending_attempts = replay.rest_sleep_pending_attempt
    if not (counts or refunded_turns or skipped_turns or pending_attempts):
        return
    tracker_keys = {
        key
        for tracker in (counts, refunded_turns, skipped_turns, pending_attempts)
        for key in tracker
        if isinstance(key, str) and key.startswith(f"{player}:")
    }
    known_rows = [
        (index, species)
        for index, row in enumerate(rows)
        if isinstance(species := row.get("species"), str)
    ]
    sleeping_rows = [
        (index, species)
        for index, species in known_rows
        if "slp" in str(rows[index].get("condition") or "").split()
    ]
    exact_known_keys = {f"{player}:{_normalize_identifier(species)}" for _, species in known_rows}
    exact_candidates: dict[str, list[int]] = {}
    for index, species in sleeping_rows:
        key = f"{player}:{_normalize_identifier(species)}"
        if key in tracker_keys:
            exact_candidates.setdefault(key, []).append(index)

    row_tracker_keys: dict[int, str] = {}
    handled_tracker_keys: set[str] = set()
    for key, candidates in exact_candidates.items():
        if len(candidates) == 1:
            row_tracker_keys[candidates[0]] = key
        else:
            for index in candidates:
                rows[index]["restSleepProvenanceUnrepresentable"] = True
        handled_tracker_keys.add(key)

    # Cosmetic base forms are safe only as a one-to-one reconciliation after exact
    # matches claim their keys. Multiple sleeping formes for one base are ambiguous.
    base_candidates: dict[str, list[int]] = {}
    for index, species in sleeping_rows:
        if index in row_tracker_keys:
            continue
        normalized = _normalize_identifier(species)
        base = _normalize_identifier(species.split("-", 1)[0])
        if base != normalized:
            base_candidates.setdefault(base, []).append(index)
    for base, candidates in base_candidates.items():
        matching_keys = [
            key
            for key in tracker_keys - handled_tracker_keys
            if key not in exact_known_keys and key.partition(":")[2] == base
        ]
        if not matching_keys:
            continue
        if len(matching_keys) == 1 and len(candidates) == 1:
            key = matching_keys[0]
            row_tracker_keys[candidates[0]] = key
        else:
            for index in candidates:
                rows[index]["restSleepProvenanceUnrepresentable"] = True
        handled_tracker_keys.update(matching_keys)

    # ``induced_sleep_victims`` is keyed by the INDUCING side; this player's victims are
    # therefore recorded under its opponent.
    induced = set(replay.induced_sleep_victims.get(opponent_showdown_slot(player), ()))
    for index, key in row_tracker_keys.items():
        row = rows[index]
        if key not in counts:
            if key in refunded_turns or key in skipped_turns or key in pending_attempts:
                row["restSleepProvenanceUnrepresentable"] = True
            continue
        count = counts[key]
        if key in induced:
            row["restSleepProvenanceUnrepresentable"] = True
            continue
        pending = pending_attempts.get(key, False)
        if not isinstance(pending, bool):
            row["restSleepProvenanceUnrepresentable"] = True
            continue
        if pending:
            # DISTINCT from the active-refund case below, which used to share this
            # flag. Here the attempt is simply still unclassified: the snapshot was
            # taken between the `|cant|...|slp` and the `|upkeep|`/`|turn|` that
            # `_settle_pending_rest_sleep_attempts` uses to resolve it. skippedTime
            # may well be 0 and the world exactly buildable -- appending `|upkeep|`
            # to the same stream builds it. Nothing about the engine's
            # representation is at fault, so do not report an engine gap.
            row["restSleepAttemptUnsettled"] = True
            _mark_legacy_rest_refund_pending(row)
            continue
        skipped = skipped_turns.get(key, 0)
        refunded = refunded_turns.get(key, 0)
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or isinstance(refunded, bool)
            or not isinstance(refunded, int)
            or isinstance(skipped, bool)
            or not isinstance(skipped, int)
            or refunded < 0
            or skipped < 0
        ):
            row["restSleepProvenanceUnrepresentable"] = True
            continue
        if skipped:
            if bool(row.get("active")):
                # The simulator will apply skippedTime only on a future switch-in. The
                # direct world has no place to preserve that pending refund for an active mon.
                # A GENUINE engine-representation gap: the value is known and there is
                # nowhere to put it. Kept separate from the unsettled-attempt flag above
                # so the two are countable apart -- only this one is closed by adding a
                # pending-skipped-time field to `Pokemon`.
                #
                # RENAMED from `restSleepRefundPending`, which both producers used to
                # set. The old key survives in corpora captured before the split and
                # cannot be attributed to a producer, so `engine_world` gives it a
                # third code rather than folding it into this one.
                row["restSleepActiveRefundPending"] = True
                _mark_legacy_rest_refund_pending(row)
                # NO LONGER a bail-out. The flag now means "the refund is PENDING,
                # not folded into rest_turns", and engine_world builds the world
                # from it instead of refusing. That needs the attempt counts, and
                # `continue` was withholding exactly them: restSleepAttempts was
                # never written for these rows, so deleting the refusal alone would
                # have changed nothing -- the row would still fall through to
                # status_unsupported.
                #
                # Both older flags keep being set deliberately. A checkout predating
                # the engine field replaying one of these rows must still REFUSE
                # rather than silently read the attempts and drop the refund, which
                # is the same protection `_mark_legacy_rest_refund_pending` gives
                # across the producer split.
                if count < 0 or refunded + skipped > count:
                    row["restSleepProvenanceUnrepresentable"] = True
                    continue
                row["restSleepAttempts"] = count
                if refunded:
                    row["restSleepRefundedTime"] = refunded
                row["restSleepSkippedTime"] = skipped
                continue
        if count < 0 or refunded + skipped > count:
            # A malformed or incomplete public stream must not be coerced into a plausible
            # Rest counter. Mark it so construction cannot approximate it as induced sleep.
            row["restSleepProvenanceUnrepresentable"] = True
            continue
        row["restSleepAttempts"] = count
        if refunded:
            row["restSleepRefundedTime"] = refunded
        if skipped:
            row["restSleepSkippedTime"] = skipped

    if tracker_keys - handled_tracker_keys:
        # A public Rest tracker that cannot be tied to a revealed row must not let any
        # same-side sleeper pass through the generic approximation path.
        for index, _ in sleeping_rows:
            rows[index]["restSleepProvenanceUnrepresentable"] = True


def _materialization_identifier(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _request_materialization_rows(
    request: Mapping[str, Any],
    *,
    self_move_states: Mapping[str, tuple[Mapping[str, Any], ...]],
) -> list[dict[str, Any]]:
    side = request.get("side") if isinstance(request.get("side"), Mapping) else {}
    pokemon_rows = side.get("pokemon") if isinstance(side, Mapping) else None
    if not isinstance(pokemon_rows, list):
        raise LocalShowdownError("Direct materialization requires the acting player's team request.")
    rows: list[dict[str, Any]] = []
    for raw_row in pokemon_rows:
        if not isinstance(raw_row, Mapping):
            continue
        details = str(raw_row.get("details") or "")
        species = details.split(",", 1)[0].strip()
        if not species:
            ident = str(raw_row.get("ident") or "")
            species = ident.split(":", 1)[-1].strip()
        condition = raw_row.get("condition")
        if not species or not isinstance(condition, str):
            raise LocalShowdownError("Direct materialization found an invalid acting-player team row.")
        rows.append(
            {
                "species": species,
                "details": details,
                "condition": condition,
                "active": bool(raw_row.get("active")),
                "moves": [dict(move) for move in self_move_states.get(_request_pokemon_identity(raw_row), ())],
            }
        )
    if not rows:
        raise LocalShowdownError("Direct materialization requires a non-empty acting-player team.")
    _apply_struggle_only_move_state(rows, request)
    return rows


def _apply_struggle_only_move_state(
    rows: list[dict[str, Any]], request: Mapping[str, Any]
) -> None:
    """At a Struggle-only request, say so on the ACTIVE row: nothing is usable.

    THE DEFECT THIS CLOSES. These rows are the only source of
    ``sides[self].pokemon[].moves``, and their move state comes from
    ``actor_move_states_from_request_history``, which retains the most recent request
    per own Pokemon. That fold skips a request whose ``_request_active_moves`` is empty,
    and Showdown's Struggle branch is exactly that -- so the row stayed pinned to the last
    pp-BEARING request and advertised a usable move at a boundary where Showdown offers
    only Struggle, while ``selfActiveMoves`` (built from the CURRENT request one call
    later) correctly reported ``[]``. One payload, two views of the same request, built
    from two different requests. Measured live: ``sunnyday pp1 disabled:false`` against
    ``selfActiveMoves: []``.

    WHY HERE AND NOT IN THE FOLD, which is where this fix was first written and which was
    WRONG. Showdown clears ``moveSlot.disabled`` on switch-out and recomputes it every
    turn, so unusability is a property of ONE BOUNDARY, not of a Pokemon. The fold is a
    per-identity historical accumulator whose entries outlive the request that produced
    them: a marking written there rides the mon onto the bench and is never refreshed
    until it is active again with a pp-bearing request. Measured on the fold version --
    Bulbasaur Taunted into Struggle, then switched out -- the benched row read
    ``sunnyday 8/8 disabled, growth 64/64 disabled``: full PP and no legal move in any
    searched line, where ``origin/main`` correctly read both enabled. Applying the verdict
    at the payload boundary instead keeps it exactly as durable as the request it came
    from, and confines it to the one row the request describes.

    That placement also removes two defects of the fold version for free: duplicate idents
    (``attract_snorlax``'s two p2 Blisseys share a retained entry, so one Blissey's
    Struggle marked the other's moveset) and the ``no retained snapshot`` case, both of
    which are keyed by identity in the fold and by ``active`` here.

    WHY MARKING IS A RESTORATION AND NOT A GUESS. ``Pokemon.getMoves``
    (``sim/pokemon.ts:1017-1042``) folds ``moveSlot.pp <= 0`` into ``disabled`` for every
    slot and returns ``hasValidMove ? moves : []``. An empty return therefore MEANS
    Showdown computed ``disabled`` for every slot and every one came back true;
    ``getMoveRequestData`` (``:1104``) then discards that list and substitutes the Struggle
    row. This writes back the verdict Showdown had already reached. PP is left pinned --
    the Struggle request carries none -- but no consumer can now read it as selectable.

    WHAT THE ENGINE DOES WITH IT, and the case this does NOT fix.
    ``Pokemon::add_available_moves`` (poke-engine 0.0.47 ``genx/state.rs``) requires
    ``!disabled && pp > 0``, so it contributes nothing and ``get_all_options`` falls
    through to ``add_switches``. With a live bench that is exactly the option set the
    Struggle request also offers. With NO legal switch -- a trapped mon, or the archetypal
    last-mon PP stall -- ``add_switches`` adds nothing either and the engine pushes
    ``MoveChoice::None``, which ``engine_search._map_choices`` translates only to
    ``recharge`` and so cannot map onto a request offering ``struggle``. That decision
    still misses. It is not a regression (the pre-fix stale move failed to map on the same
    decision) but it is not fixed here, and the completing half is a ``none -> struggle``
    translation alongside the existing ``none -> recharge`` one.
    """

    if not _request_reports_only_struggle(request):
        return
    for row in rows:
        if not row["active"]:
            continue
        # The rows already hold `dict(move)` COPIES, so this cannot write through to the
        # retained `self_move_states` that later boundaries fold onto.
        for move in row["moves"]:
            move["disabled"] = True


def _request_active_moves(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    active = request.get("active")
    active_row = active[0] if isinstance(active, list) and active else None
    moves = active_row.get("moves") if isinstance(active_row, Mapping) else None
    if not isinstance(moves, list):
        return []
    copied: list[dict[str, Any]] = []
    for move in moves:
        if not isinstance(move, Mapping) or not isinstance(move.get("id"), str):
            continue
        pp = move.get("pp")
        maxpp = move.get("maxpp")
        if not isinstance(pp, int) or not isinstance(maxpp, int):
            continue
        copied.append(
            {
                "id": move["id"],
                "pp": pp,
                "maxpp": maxpp,
                "disabled": bool(move.get("disabled")),
            }
        )
    return copied


def _request_reports_only_struggle(request: Mapping[str, Any]) -> bool:
    """Whether this request is Showdown's Struggle branch, i.e. NOTHING is usable.

    MEASURED, not read. The provenance list in
    ``actor_move_states_from_request_history`` classified Struggle as "SOURCE ONLY --
    unverified by measurement; an attempt to produce a live Struggle request failed on
    packed-team format". Driving a gen3 Custom Game with a single-move Bulbasaur
    (Sunny Day, 8 PP) for eight turns produced the branch on turn 8, verbatim::

        [{"move": "Struggle", "id": "struggle", "target": "randomNormal", "disabled": false}]

    No ``pp``, no ``maxpp``, and in that capture the active row carried no other key -- in
    particular no ``trapped``, so switching was legal there. That is NOT general: see the
    no-legal-switch case in ``_apply_struggle_only_move_state``.

    WHY THE BRANCH IS AN ASSERTION ABOUT EVERY SLOT, which is the whole basis of the fix.
    ``Pokemon.getMoves`` (``sim/pokemon.ts:1017-1042``) builds the full moveSlots list,
    folds ``moveSlot.pp <= 0`` INTO ``disabled``, and returns ``hasValidMove ? moves : []``.
    An empty return therefore means Showdown computed ``disabled`` for every slot and every
    one came back true. ``getMoveRequestData`` (``:1104``) then throws that list away and
    substitutes the Struggle row.

    NARROW ON PURPOSE. The sibling pp-less shapes -- ``mustrecharge`` (measured, 6 times)
    and the two-turn charge lock -- come off ``getMoves``'s EARLY ``if (lockedMove)``
    return at ``:966``, which never evaluates the per-slot ``disabled`` at all. For those
    the real slots are usually perfectly usable and merely pre-empted for one turn, so the
    same marking would be a fabrication rather than a restoration. They are also inert
    downstream: poke-engine short-circuits both on the MUSTRECHARGE volatile and on
    ``active_is_charging_move`` before ``Pokemon::add_available_moves`` is consulted, so
    their ``disabled`` flags could not change an option set even if they were written. They
    keep the pre-existing skip.
    """

    active = request.get("active")
    active_row = active[0] if isinstance(active, list) and active else None
    moves = active_row.get("moves") if isinstance(active_row, Mapping) else None
    if not isinstance(moves, list) or len(moves) != 1:
        return False
    only = moves[0]
    if not isinstance(only, Mapping) or only.get("id") != "struggle":
        return False
    # The PP fields are what separate the substituted pseudo-move from a real move slot.
    # Struggle is not in any gen 3 randbats set and our own team is the battle-start
    # request team verbatim, so a pp-BEARING `struggle` row cannot occur here -- but the
    # check is what makes that a checked fact rather than an assumed one.
    return not isinstance(only.get("pp"), int) and not isinstance(only.get("maxpp"), int)


def _request_materialization_kind(request: Mapping[str, Any]) -> str:
    force_switch = request.get("forceSwitch")
    if isinstance(force_switch, list) and any(bool(entry) for entry in force_switch):
        return "force-switch"
    return "move"


def _request_active_materialization_state(request: Mapping[str, Any]) -> dict[str, bool]:
    """Return request-visible active constraints that affect the action boundary.

    These flags are supplied to the acting player by Showdown. Restoring them keeps the direct
    branch's legal action mask aligned even if the sampled simulator world cannot re-derive a
    public constraint from its freshly constructed internal state.
    """

    active = request.get("active")
    active_row = active[0] if isinstance(active, list) and active else None
    if not isinstance(active_row, Mapping):
        return {}
    return {
        name: True
        for name in ("trapped", "maybeTrapped", "maybeDisabled", "maybeLocked")
        if bool(active_row.get(name))
    }


def actor_move_states_from_request_history(
    requests: Sequence[Mapping[str, Any]],
    *,
    initial_request: Mapping[str, Any],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    """Return each actor-known active move state from its most recent request.

    A normal Showdown request carries exact PP only for the current active Pokemon. Keeping the
    most recent such state per own Pokemon is player-known information and lets direct search
    restore a previously active Pokemon after it has switched out.

    A request taken while that Pokemon was TRANSFORMED is skipped, and its earlier clean
    snapshot kept instead. This is the fix for the dominant half of `self_moveset_mismatch`
    -- 365 killed decisions in era 59, 44.8% of the construction channel, all seat p1.

    THE MECHANISM. `request["active"][0]["moves"]` is the USABLE moveset, so while Ditto is
    transformed it lists the COPIED moves. Retaining that was permanent: gen 3 reverts
    Transform on switch-out, so a benched Ditto never appears active again with its own set
    and no later request refreshes the entry. The stale row reached
    `engine_world._move_specs` as Ditto's request-known moveset, was compared against the
    root snapshot's `[transform]`, and refused every world with
    `request-known move 'bodyslam' is absent from the sampled moveset`. Failing closed there
    is CORRECT -- the input really is wrong -- so loosening that guard would hide the defect
    rather than fix it.

    ONE KNOWN STALENESS, named because this module's rule is to refuse inexact self PP
    rather than guess it. The retained snapshot predates the Transform USE that produced the
    transform, so a benched reverted Ditto/Mew is materialized with one PP too many on its
    transform slot -- measured at 16 retained where the truth is 15. Bounded to 1 PP on that
    one slot, and refreshed by any later clean request for that Pokemon. Deriving PP from the
    public move record removes it; that work is additive and this fix does not foreclose it.

    KEEP the earlier clean snapshot; do not erase the identity. Erasing also removes
    `self_moveset_mismatch`, but strands the Pokemon with no PP at all and `engine_world`
    then refuses it as `self_pp_unknown`. Measured on the golden-corpus `ditto_transform`
    sweep: erasing turned 7 refusals into 8, which is not a fix. Keeping the pre-transform
    snapshot is right on the merits too, because the copied moves spend the COPIED PP -- a
    transformed Ditto using Body Slam does not decrement Ditto's own Transform PP.

    HOW A TRANSFORMED REQUEST IS RECOGNISED, and why the obvious way fails. Comparing the
    usable moveset against the same request's `side.pokemon[].moves` does NOT work: that
    assumption (that Showdown reports base moves there, untouched by Transform) is false on
    this path, and it was measured rather than argued. Instrumenting the sweep printed

        usable=['bodyslam','curse','rest','shadowball'] own=['bodyslam','curse','rest','shadowball']

    -- the request reports the copied set in BOTH places, so a subset check inside one
    request can never fire.

    The FIRST request is the reference instead. It is battle-start, so it necessarily
    predates any Transform, and its `side.pokemon[].moves` give each own Pokemon's real
    moveset. A later active moveset that is not a subset of that is not this Pokemon's.

    Not Mimic: `mimic` appears in no gen 3 randbats set, and our own team is the
    battle-start request team verbatim, so it cannot carry one.
    """

    # The battle-start request is PASSED IN, not taken as `requests[0]`. Review showed a
    # truncated history does not degrade to a no-op -- it INVERTS this fix:
    #
    #   [clean, transformed] -> {'ditto': ['transform']}                 (correct)
    #   [transformed, clean] -> {'ditto': ['bodyslam', ...]}             (copied kept,
    #                                                                     REAL set dropped)
    #
    # and pre-PR that second order returned the clean set, so a truncated history would be
    # strictly worse than before. Two real surfaces already produce non-battle-start
    # histories: `materialize_scenario_state` seeds `_request_history` with a single
    # mid-battle request, and `restore_public_materialization` leaves it empty after
    # `_reset`, so any stepped search env starts mid-battle. Both production call sites
    # already hold the initial request one line away.
    #
    # REQUIRED, not defaulted. An earlier version took `initial_request=None` and fell back
    # to `requests[0]`, with a comment claiming that "removes the assumption". It did not --
    # it relocated the assumption to any caller that forgets, and review found such callers
    # already existed. Prose asserting an invariant the signature does not enforce is the
    # same shape as the two defects this PR already records.
    own_by_identity = _own_move_ids_by_identity(initial_request)
    states: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for request in requests:
        identity = _request_active_pokemon_identity(request)
        moves = _request_active_moves(request)
        if identity is None or not moves:
            # A pp-LESS request is still skipped here, and deliberately. What Showdown's
            # Struggle branch asserts -- every slot disabled -- is true of ONE BOUNDARY,
            # because `moveSlot.disabled` is cleared on switch-out and recomputed each
            # turn, whereas an entry in this fold outlives the request that wrote it and
            # rides the Pokemon onto the bench. Recording it here was measured wrong: a
            # Taunted-into-Struggle Bulbasaur that then switched out kept
            # `sunnyday 8/8 disabled, growth 64/64 disabled` on the bench. The verdict is
            # applied at the payload boundary instead; see
            # `_apply_struggle_only_move_state`.
            continue
        own = own_by_identity.get(identity)
        if own:
            usable = {
                _base_move_id(str(move.get("id")))
                for move in moves
                if isinstance(move.get("id"), str)
            }
            # SUBSET, and under the union above it is REQUIRED rather than merely safer.
            # This comment has been wrong twice, so here is the whole reasoning:
            #
            # Wrong v1: "Disable/Encore/Choice-locked requests list fewer moves." They do
            # not. `active[0].moves` is always the full current moveSlots list with
            # `disabled` flags.
            #
            # Wrong v2: "subset is the weaker, safer claim that nothing here needs, and
            # equality is indistinguishable on real input." That cited a live measurement of
            # ZERO strict-subset pp-bearing requests across 2,219 real actor requests -- but
            # that measurement was taken against a SINGLE ROW's moveSlots, before
            # `_own_move_ids_by_identity` began unioning duplicate idents. Union makes `own`
            # a superset by construction whenever idents collide, so a strict subset is the
            # NORMAL case there: measured, 4 usable moves against a 6-move union.
            #
            # So equality would reject every duplicate-ident team WHOSE DUPLICATES DIFFER in
            # moveset -- identical duplicates union back to 4 and equality still accepts. The
            # corpus's only case does differ, so subset is required here; the quantifier just
            # is not universal. The design is on firmer ground than its old justification
            # claimed. Equality and reversed-subset mutants now die at the unit level; they
            # survive only the LIVE gate, and only because p2 is not the search seat there.
            #
            # PROVENANCE. Three tiers, and note that "Wrong v1" above is MEASURED, not read:
            #   * MEASURED live -- (a) the 2,219-request scan, scope per-row movesets and
            #     pre-union; (b) `mustrecharge` from Hyper Beam, 6 times, carrying no `pp`, so
            #     `_request_active_moves` drops it; (c) ENCORE, which carries `pp` and does
            #     NOT narrow the list. An earlier version of this comment put Encore in the
            #     `if (lockedMove)` block alongside recharge, "sibling branches both returning
            #     pp-less dicts". That is FALSE and it contradicted "Wrong v1" four lines
            #     above. `onSemiLockMove` is registered only in the gen 1 and gen 2 mods
            #     (Bide), so `getSemiLockedMove` cannot return Encore; gen 3 Encore is
            #     `onDisableMove` + `onOverrideAction`, which DISABLES the other three slots.
            #     Driving `encore_wobbuffet` with the encored seat as p2 captured the real
            #     shape: four slots, all with `pp`, three flagged `disabled: True`. So Encore
            #     contributed zero strict subsets to the scan because it never narrows -- not
            #     because it was dropped upstream. That capture also shows a 0-PP slot staying
            #     listed with `disabled: True`, which independently confirms the "always the
            #     full moveSlots list" premise this whole argument rests on.
            #   * SAME BLOCK as a measured case, NOT measured -- SolarBeam's `twoturnmove`
            #     charge turn (7 gen 3 randbats sets) reaches the same `if (lockedMove)` block
            #     as the measured `mustrecharge`. The corpus has no SolarBeam scenario, which
            #     is consistent with all 6 pp-less requests in the scan logging as
            #     `recharge`. Every other producer of that block -- outrage, thrash,
            #     petaldance, rollout, iceball, uproar, bide, fly, dig, skullbash, skyattack,
            #     razorwind -- appears in ZERO gen 3 randbats sets.
            #   * MEASURED live, and it was this module's one SOURCE-ONLY entry until then
            #     -- Struggle, its own `!moves.length` branch. The earlier attempt to
            #     produce one failed on packed-team format; driving a gen3 Custom Game with
            #     a single-move Bulbasaur (Sunny Day, 8 PP) for eight turns produces it.
            #     The captured row is `{"move": "Struggle", "id": "struggle", "target":
            #     "randomNormal", "disabled": false}` -- pp-less, so `_request_active_moves`
            #     drops it, exactly as the source predicted. The drop is still correct HERE
            #     (see the skip above); the boundary-scoped verdict it used to lose is
            #     restored in `_apply_struggle_only_move_state`.
            #
            # Subset is kept because it is the weaker, safer claim: it says "every move the
            # player may pick is one this Pokemon knows", which is what makes the moveset
            # its own. Equality would additionally assert the request never narrows, which
            # nothing here needs and no measurement supports.
            #
            # It must be a subset test rather than an INTERSECTION test. Ditto's own set is
            # the single move `transform`, which no opponent carries, so for Ditto
            # "not a subset" and "no overlap" coincide -- but Mew has 7 randbats sets whose
            # four own moves overlap what it copies, and an intersection test would retain
            # the copied set for the whole Mew population. Pinned by a Mew test.
            if not usable <= own:
                continue
        states[identity] = tuple(_json_clone_mapping(move) for move in moves)
    return states


def _own_move_ids_by_identity(request: Mapping[str, Any]) -> dict[str, frozenset[str]]:
    """Each own Pokemon's real move ids, from the BATTLE-START request.

    Battle-start is what makes this reference trustworthy: it precedes any Transform, so
    these are base movesets even though the same field is unreliable in later requests.
    """

    side = request.get("side")
    rows = side.get("pokemon") if isinstance(side, Mapping) else None
    if not isinstance(rows, list):
        return {}
    own: dict[str, frozenset[str]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        raw = row.get("moves")
        if not isinstance(raw, list):
            continue
        ids = frozenset(
            _base_move_id(move) for move in raw if isinstance(move, str) and move
        )
        if not ids:
            continue
        identity = _request_pokemon_identity(row)
        if identity in own:
            # TWO ROWS, ONE IDENT. Species Clause makes this unreachable in randbats, but a
            # Custom Game team can carry duplicates, and review observed it live: the
            # `attract_snorlax` scenario puts two Blisseys on p2, both `p2: Blissey`, and a
            # dict overwrite kept the LAST row -- so the active Blissey's own moveset read as
            # another Pokemon's and its PP snapshot was dropped on the very first request.
            #
            # UNION rather than overwrite -- and NOT because it is lossless. An earlier
            # version of this comment claimed a copied moveset "is not a subset of the union
            # either", which review DISPROVED using the shape the corpus already contains:
            # `attract_snorlax`'s two p2 Blisseys union to six moves, and a mirror-match
            # Blissey's copied 4-set is drawn entirely from that union, so the copy slips
            # through and the transform is MASKED.
            #
            # Union is still right, for the honest reason: it converts a FALSE POSITIVE
            # (judging a real moveset transformed, dropping known-good PP, risking
            # `self_pp_unknown` -- the mis-fire review observed live) into a FALSE NEGATIVE
            # in an exotic case. Only ditto and mew carry Transform and Species Clause bars
            # duplicate idents in randbats, so the false negative needs a Custom Game
            # mirror-match Ditto; and when it lands, the result is merely the pre-PR state,
            # which `engine_world` still fails closed on. A false positive discards good
            # information on a path where nothing was wrong.
            #
            # REFUSING is wrong: `attract_snorlax` has duplicate idents today, so raising
            # would hard-error a valid corpus scenario over a non-defective input.
            #
            # And there is NO POSITIONAL ALTERNATIVE, checked rather than assumed: Showdown
            # reorders `side.pokemon[]` so the active mon is index 0 (measured -- the ditto
            # sweep goes [Ditto(active), Swampert] -> [Swampert(active), Ditto] across the
            # switch), so position is not stable across requests and identity is the only
            # cross-request key this interface offers. Duplicate idents are therefore
            # fundamentally ambiguous here; `_request_materialization_rows` keys its lookup
            # by the same identity and already gives both Blisseys the same rows, which is a
            # limit of the retained-state interface rather than of this guard.
            own[identity] = own[identity] | ids
        else:
            own[identity] = ids
    return own


def _base_move_id(move_id: str) -> str:
    """Collapse Showdown's resolved-power spellings so the two move lists are comparable.

    The same move is spelled differently in the two places this compares:
    `side.pokemon[].moves` carries resolved power (`return102`) while
    `active[0].moves[].id` carries the base id (`return`). Comparing raw would call every
    Return and Hidden Power carrier transformed and silently drop its PP snapshot -- a
    regression in the opposite direction, and one era counts could not separate from the
    fix, because both show up as fewer retained snapshots. `return102` (100 occurrences) is
    the ONLY digit-bearing spelling that appears in the Custom Game scenario sweep, and the
    era-59 randbats reproduction independently found the same single spelling -- two
    populations, not one, since the sweep alone would not license a randbats claim.

    Built on `tier2.canonical_move_id` rather than a local `rstrip("0123456789")`, for two
    reasons review supplied. It restricts the strip to the three BP-suffixed prefixes, so
    `conversion2` no longer collapses onto `conversion` -- both are real gen 3 moves, the
    only such collision in the 954-id gen 3 dex, unreachable in randbats but reachable in
    Custom Game. And it is the SAME helper `determinization._self_team_from_metadata_result`
    uses to build the root self_team, which is the exact reference `_move_specs` compares
    the retained moveset against; two normalizers for one comparison is how they drift.

    Hidden Power still needs its own collapse: gen 3 emits `hiddenpower<type>` on the team
    side against a bare `hiddenpower` id, so canonicalisation alone leaves them unequal. The
    types are what vary, NOT a BP suffix -- measured over the gen 3 randbats sets, all 1,682
    carry ZERO BP-suffixed Hidden Power across 13 distinct spellings (`hiddenpowergrass` 130
    sets, `fire` 108, `ground` 97, `ghost` 92, `flying` 84, `ice` 72, `bug` 37, `rock` 28,
    `fighting` 25, `electric` 24, `steel` 18, `dark` 11, `psychic` 1). An earlier version of
    this said "all spelled `hiddenpowerice`", which is one spelling of thirteen and the sixth
    most common.
    """

    canonical = canonical_move_id(move_id)
    if canonical.startswith("hiddenpower"):
        return "hiddenpower"
    return canonical


def _request_active_pokemon_identity(request: Mapping[str, Any]) -> str | None:
    side = request.get("side") if isinstance(request.get("side"), Mapping) else {}
    pokemon_rows = side.get("pokemon") if isinstance(side, Mapping) else None
    if not isinstance(pokemon_rows, list):
        return None
    for row in pokemon_rows:
        if isinstance(row, Mapping) and bool(row.get("active")):
            return _request_pokemon_identity(row)
    return None


def _request_pokemon_identity(row: Mapping[str, Any]) -> str:
    ident = str(row.get("ident") or "")
    if not ident:
        ident = str(row.get("details") or "").split(",", 1)[0]
    return _materialization_identity(ident)


def _materialization_identity(value: str) -> str:
    """Normalize request and protocol identifiers without retaining the player-side prefix."""

    return value.split(":", 1)[-1].strip().casefold()


def _has_self_benched_move_history(state: PublicBattleMaterializationState) -> bool:
    """Whether a previously active self Pokemon lacks a request-known move-state snapshot."""

    active = state.replay.public_active.get(state.player_id)
    active_ident = active.ident if active is not None else None
    if active_ident is None:
        raise LocalShowdownError("Direct materialization requires an acting-player active Pokemon.")
    active_identity = _materialization_identity(active_ident)
    known_identities = set(state.self_move_states)
    return any(
        event.event_type == "move"
        and event.actor_slot == state.player_id
        and event.actor_ident is not None
        and _materialization_identity(event.actor_ident) != active_identity
        and _materialization_identity(event.actor_ident) not in known_identities
        for event in state.replay.public_events
    )


def _drain_stdout(stream: TextIO, target: queue.Queue[str | None]) -> None:
    try:
        for line in stream:
            target.put(line)
    except (OSError, ValueError):
        pass
    finally:
        target.put(None)


def _close_process_pipes(process: subprocess.Popen[str]) -> None:
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None and not stream.closed:
            try:
                stream.close()
            except OSError:
                pass


def _drain_stderr(stream: TextIO | None, target: list[str]) -> None:
    if stream is None:
        return
    try:
        for line in stream:
            target.append(line.rstrip())
            if len(target) > 100:
                del target[: len(target) - 100]
    except (OSError, ValueError):
        pass
