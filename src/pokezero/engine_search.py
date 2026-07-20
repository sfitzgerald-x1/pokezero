"""Native-engine MCTS over belief-sampled worlds (engine swap plan v3).

FoulPlay's architecture on pokezero's belief engine: per decision, sample K
determinized worlds from the public belief (the existing
``gen3_randbat_belief_start_override`` planner), construct each as a
poke-engine state via the track-A world constructor, search each world
natively, and aggregate the acting side's root visit distributions across
worlds. Two leaf-eval modes, selected by ``EngineMctsConfig.leaf_eval``:

- ``"hp_fraction"`` (default, the POC path): poke-engine's built-in MCTS
  with its handcrafted evaluation for a fixed time budget — no learned
  model, no policy priors. Kept as the default until the paired read.
- ``"model"`` (the full in-crate pipeline): per world, the crate's
  ``search_batched_multi_encoded`` — the LIVE root fold state (maintained
  incrementally here, see ``_advance_live_fold``) plus per-branch
  synthesized events, per-outcome fold advance, native v2.2 leaf encode,
  batched TorchScript leaf evaluation, and the acting seat's decision arms
  weighted by the model's masked policy priors (opponent arms stay uniform
  — see docs/crate_search_design.md "Model priors"). NO strength claim is
  attached to this mode until the 200-seed paired FoulPlay read.

Shared boundaries (both modes):

- **Fail-closed construction.** Decisions whose worlds cannot be expressed
  exactly (see ``engine_world``'s reason taxonomy) fall back to uniform
  legal; the bench reports the rate and taxonomy rather than hiding it.
- **Uniform world weights.** FoulPlay weights worlds by sample likelihood;
  the belief planner does not expose one yet.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from .dex import ShowdownDex, normalize_id
from .determinization import (
    _gen3_randbat_belief_start_override_result,
    _move_from_public_event_line,
)
from .public_action_capture import public_action_rounds_from_trajectory_metadata
from .engine_world import EngineWorld, EngineWorldUnsupported, world_battle_spec
from .randbat import canonical_gen3_randbat_species_id
from .poke_engine_adapter import build_poke_engine_state
from .policy import PolicyContext, PolicyDecision, legal_action_indices

_fallback_logger = logging.getLogger("pokezero.engine_search.fallback")


class EngineSearchFallbackWarning(UserWarning):
    """A search decision fell back to uniform-legal instead of searching.

    Loud by design: any process running engine search (benches, sweeps,
    collection, integration tests) sees these in test output and default
    logging, can escalate them to hard errors with
    ``warnings.simplefilter("error", EngineSearchFallbackWarning)`` or
    ``EngineMctsConfig(strict_fallbacks=True)``, and can grep the stable
    logger name ``pokezero.engine_search.fallback``. Every occurrence must
    be attributable through the fallback/world-failure reason taxonomy —
    benches report the rate and the reasons rather than hiding either.
    (The one-time 0.0% bench rate does not hold on all seed trajectories:
    battles where the opponent publicly Transforms, or holds an item Trick
    swapped onto it, fail worlds closed for the rest of the battle by
    design — both leaf-eval modes hit the same wall on the same battles.
    Knock-Off REMOVALS no longer wall: the belief_view removal/swap
    distinction lets sampled worlds express "publicly holds no item".)
    """


class EngineSearchFallbackError(RuntimeError):
    """Raised instead of falling back when ``strict_fallbacks`` is set."""


class EngineSearchFoldMismatchWarning(UserWarning):
    """The live incremental root fold diverged from the whole-log batch refold
    (or an advance failed outright).

    Same loudness contract as the fallback warning: visible in test output
    and default logging, escalatable to a hard error via
    ``warnings.simplefilter("error", ...)`` or ``strict_fallbacks``, and
    greppable on the stable logger name ``pokezero.engine_search.fold``. The
    incremental fold is closure-proven (PR #718) and byte-exact over both
    corpora, so any occurrence is a real regression signal.
    """


_fold_logger = logging.getLogger("pokezero.engine_search.fold")


class EnvTier2AnnotationSource:
    """Env→policy surface for the live fold's Tier-2 annotation overlay.

    Tracker conclusions are ENV-side state (as-of-first-assessment —
    ``local_showdown._tier2_tracker_for``; they cannot be re-derived at
    decision time because a fresh tracker would assess as-of-now). This
    adapter reads the env's own per-player state derivation — the exact
    pattern corpus capture uses (``golden_corpus_fold.FoldSurfaceRecorder``;
    deterministic and tracker-idempotent) — and reduces the ANNOTATED
    per-action stream to a ``FoldState.apply_annotations`` overlay with
    ``build_fold_rows``' exact rule. It also exposes the boundary state for
    the strengthened fold cross-check (live fold products vs the production
    encoder state's surfaces — corpus generation's production-binding
    assertion, run live).
    """

    def __init__(self, env: Any) -> None:
        self._env = env

    def active(self) -> bool:
        probe = getattr(self._env, "tier2_residuals_active", None)
        return bool(probe()) if callable(probe) else False

    def boundary_state(self, player_id: str) -> Any:
        return self._env._state_for_player(player_id)  # noqa: SLF001 — FoldSurfaceRecorder pattern

    def overlay_for(self, player_id: str) -> dict[int, tuple]:
        """The env trackers' per-index conclusions, cumulative from battle
        start (``build_fold_rows``' derivation rule, verbatim)."""

        state = self.boundary_state(player_id)
        return {
            index: (token.residual, token.residual_valid, token.cb_bit, token.investment)
            for index, token in enumerate(state.transition_tokens)
            if token.residual is not None
            or token.residual_valid
            or token.cb_bit
            or token.investment
        }


@dataclass(frozen=True)
class EngineMctsConfig:
    worlds: int = 4
    search_time_ms: int = 100
    threads: int = 1
    # Documented approximation (see engine_world): model publicly-asleep mons
    # as freshly asleep instead of failing the whole world closed. Without it
    # the fallback rate is dominated by sleep (~60% of decisions in smokes).
    approximate_sleep_turns: bool = True
    # Belief sampling is stochastic; failed draws are retried up to
    # worlds * sample_retry_factor total attempts (mirrors the W1 retry fix).
    sample_retry_factor: int = 4
    # Documented approximation: a public Substitute is modeled at fresh
    # (maxhp/4) health, since remaining sub HP is not tracked publicly.
    approximate_substitute_health: bool = True
    # Escalate any decision-level fallback to EngineSearchFallbackError.
    # For sweeps/CI that require zero fallbacks; production keeps the safe
    # uniform-legal fallback (a crash mid-collection is worse than a miss).
    strict_fallbacks: bool = False
    # --- full in-crate pipeline (plan v3 "Integration endgame") ---
    # "hp_fraction": poke-engine's native MCTS + handcrafted eval (the POC
    # path; stays the default until the paired read). "model": per belief
    # world, the crate's search_batched_multi_encoded — live root fold +
    # per-branch observations + in-crate TorchScript leaf eval + self-side
    # model priors in PUCT selection.
    leaf_eval: str = "hp_fraction"
    # TorchScript artifact (scripts/export_model.py; per-device trace — a CPU
    # artifact must run on cpu) and the encoder tables JSON
    # (scripts/export_encoder_tables.py). Both required in model mode.
    model_path: str | None = None
    model_device: str = "cpu"
    tables_path: str | None = None
    # Per-world search budget (model mode). Keep search_batch << search_sims
    # (virtual-loss fidelity; docs/crate_search_design.md review caveats).
    search_sims: int = 256
    search_batch: int = 16
    search_depth: int = 2
    c_puct: float = 1.4
    deep_ko_split: bool = True
    # Self-side model priors in selection (the opponent side stays uniform in
    # this integration; docs/crate_search_design.md "Model priors").
    model_priors: bool = True
    # Debug cross-check: per decision, batch-refold the whole public log
    # (production's per-observe path, turn_merged.extract_transition_products)
    # and compare its surfaces against the live incremental fold's products.
    fold_cross_check: bool = False

    def __post_init__(self) -> None:
        if self.worlds <= 0 or self.search_time_ms <= 0 or self.threads <= 0:
            raise ValueError("worlds, search_time_ms, and threads must be positive.")
        if self.leaf_eval not in ("hp_fraction", "model"):
            raise ValueError(
                f"leaf_eval must be 'hp_fraction' or 'model', got {self.leaf_eval!r}."
            )
        if self.leaf_eval == "model":
            if not self.model_path or not self.tables_path:
                raise ValueError("leaf_eval='model' requires model_path and tables_path.")
            if self.search_sims <= 0 or self.search_batch <= 0 or self.search_depth <= 0:
                raise ValueError(
                    "search_sims, search_batch, and search_depth must be positive."
                )
            if self.search_batch > self.search_sims:
                raise ValueError(
                    "search_batch must be <= search_sims (keep batch << sims; "
                    "docs/crate_search_design.md review caveats)."
                )


@dataclass
class EngineMctsStats:
    """Cumulative per-policy telemetry; every fallback is counted, never hidden."""

    decisions: int = 0
    searched_decisions: int = 0
    fallback_decisions: int = 0
    # Decisions where the removal signal fired (an opposing mon's item is
    # publicly stripped): worlds constructed with that item cleared instead of
    # failing closed. Localizes which battles exercise the removal path.
    removed_item_decisions: int = 0
    worlds_attempted: int = 0
    worlds_searched: int = 0
    total_iterations: int = 0
    search_wall_seconds: float = 0.0
    decision_wall_seconds: float = 0.0
    world_failure_reasons: Counter = field(default_factory=Counter)
    fallback_reasons: Counter = field(default_factory=Counter)
    unmapped_choices: Counter = field(default_factory=Counter)
    # Model-mode telemetry (zero on the hp_fraction path).
    model_evals: int = 0
    lossy_renders: int = 0
    prior_fallbacks: int = 0
    fold_advanced_lines: int = 0
    fold_cross_checks: int = 0
    fold_cross_check_failures: int = 0
    # Tier-2 overlay telemetry (zero without an annotation source).
    fold_annotations_applied: int = 0
    fold_annotation_boundaries: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "decisions": self.decisions,
            "searched_decisions": self.searched_decisions,
            "fallback_decisions": self.fallback_decisions,
            "fallback_rate": self.fallback_decisions / self.decisions if self.decisions else 0.0,
            "removed_item_decisions": self.removed_item_decisions,
            "worlds_attempted": self.worlds_attempted,
            "worlds_searched": self.worlds_searched,
            "total_iterations": self.total_iterations,
            "search_wall_seconds": self.search_wall_seconds,
            "decision_wall_seconds": self.decision_wall_seconds,
            "world_failure_reasons": dict(self.world_failure_reasons),
            "fallback_reasons": dict(self.fallback_reasons),
            "unmapped_choices": dict(self.unmapped_choices),
            "model_evals": self.model_evals,
            "lossy_renders": self.lossy_renders,
            "prior_fallbacks": self.prior_fallbacks,
            "fold_advanced_lines": self.fold_advanced_lines,
            "fold_cross_checks": self.fold_cross_checks,
            "fold_cross_check_failures": self.fold_cross_check_failures,
            "fold_annotations_applied": self.fold_annotations_applied,
            "fold_annotation_boundaries": self.fold_annotation_boundaries,
        }
        if self.searched_decisions:
            payload["iterations_per_searched_decision"] = (
                self.total_iterations / self.searched_decisions
            )
            payload["search_wall_per_searched_decision"] = (
                self.search_wall_seconds / self.searched_decisions
            )
        if self.decisions:
            payload["wall_per_decision"] = self.decision_wall_seconds / self.decisions
        return payload


class EngineMctsPolicy:
    """ContextAwarePolicy running poke-engine MCTS over belief-sampled worlds."""

    def __init__(
        self,
        *,
        dex: ShowdownDex,
        set_source: Any,
        config: EngineMctsConfig | None = None,
        module: Any | None = None,
        policy_id: str = "engine-mcts",
        fixed_override: Any | None = None,
        annotation_source: Any | None = None,
    ) -> None:
        if module is None:
            import poke_engine as module  # noqa: PLC0415 — optional native dependency

        self.policy_id = policy_id
        # Test/scenario hook: bypass belief sampling and use this override as
        # every world (custom-game sweeps where the catalog cannot sample).
        self._fixed_override = fixed_override
        # Tier-2 overlay source (EnvTier2AnnotationSource protocol): when the
        # env runs with active Tier-2 trackers, the live fold must carry the
        # trackers' conclusions or every fold-derived annotated surface at
        # search leaves diverges from what the env encodes. None ⇒ the fold
        # stays unannotated (correct for tracker-inactive envs only).
        self._annotation_source = annotation_source
        self._dex = dex
        self._set_source = set_source
        self._config = config or EngineMctsConfig()
        self._module = module
        self.stats = EngineMctsStats()
        self._world_failures_before: dict[str, int] = {}
        # Live incremental root fold, per (battle, seat): the ledger's "live
        # root fold export". transitions_fold.FoldState from initial() at
        # battle start, advanced over exactly the new public lines at every
        # decision (_advance_live_fold) — never a whole-log refold.
        self._live_folds: dict[tuple[str, str], Any] = {}
        self._fold_consumed: dict[tuple[str, str], int] = {}
        self._fold_broken: set[tuple[str, str]] = set()
        self._tables_json: str | None = None
        self._native_model: Any | None = None
        if self._config.leaf_eval == "model":
            from pathlib import Path  # noqa: PLC0415 — model-mode-only dependency

            model_path = Path(str(self._config.model_path))
            if not model_path.exists():
                raise ValueError(f"model artifact not found: {model_path}")
            self._tables_json = Path(str(self._config.tables_path)).read_text(
                encoding="utf-8"
            )

    # Policy protocol (context-free path): uniform legal. Only reached if the
    # rollout driver cannot supply a context, which the bench never does.
    def select_action(self, observation, *, rng: random.Random) -> PolicyDecision:
        legal = legal_action_indices(observation.legal_action_mask)
        return PolicyDecision(action_index=rng.choice(legal), policy_id=self.policy_id)

    def select_action_with_context(
        self, context: PolicyContext, *, rng: random.Random
    ) -> PolicyDecision:
        started = time.perf_counter()
        decision = self._search(context, rng=rng)
        self.stats.decisions += 1
        self.stats.decision_wall_seconds += time.perf_counter() - started
        return decision

    # -----------------------------------------------------------------------------------------

    def _search(self, context: PolicyContext, *, rng: random.Random) -> PolicyDecision:
        self._world_failures_before = dict(self.stats.world_failure_reasons)
        if context.public_materialization_state is None:
            return self._fallback(context, rng, "no_public_state")
        # Live root fold: advanced at EVERY decision boundary (model mode and
        # cross-check debugging) so the fold state is current whichever
        # decisions end up searched.
        live_fold = None
        if self._config.leaf_eval == "model" or self._config.fold_cross_check:
            live_fold = self._advance_live_fold(context)
            if live_fold is None and self._config.leaf_eval == "model":
                return self._fallback(context, rng, "live_fold_broken")
        blocked_slots, encored_moves, removed_item_species = self._public_effect_signals(context)
        if removed_item_species:
            self.stats.removed_item_decisions += 1
        recharging_slots = self._recharging_slots(context)
        truant_slots = self._truant_loaf_slots(context)

        worlds: list[tuple[EngineWorld, Any]] = []
        attempts_budget = self._config.worlds * self._config.sample_retry_factor
        attempts = 0
        while len(worlds) < self._config.worlds and attempts < attempts_budget:
            attempts += 1
            self.stats.worlds_attempted += 1
            if self._fixed_override is not None:
                override, sample_failure = self._fixed_override, None
            else:
                override, sample_failure = _gen3_randbat_belief_start_override_result(
                    context=context,
                    set_source=self._set_source,
                    rng=rng,
                    witnessed_fallback=True,
                )
            if override is None:
                self.stats.world_failure_reasons[
                    f"belief_sample: {sample_failure or 'unknown'}"
                ] += 1
                continue
            try:
                world = world_battle_spec(
                    context.public_materialization_state,
                    override,
                    dex=self._dex,
                    approximate_sleep_turns=self._config.approximate_sleep_turns,
                    approximate_substitute_health=self._config.approximate_substitute_health,
                    blocked_slots=blocked_slots,
                    encored_moves=encored_moves,
                    removed_item_species=removed_item_species,
                    recharging_slots=recharging_slots,
                    truant_slots=truant_slots,
                    rng=rng,
                )
                state = build_poke_engine_state(world.spec, module=self._module)
            except EngineWorldUnsupported as error:
                key = error.reason
                if key in (
                    "volatile_unsupported",
                    "hidden_power_iv_mismatch",
                    "wish_carrier_ambiguous",
                    "self_world_mismatch",
                ):
                    key = f"{error.reason}: {error.detail}"
                self.stats.world_failure_reasons[key] += 1
                continue
            worlds.append((world, state))

        if not worlds:
            return self._fallback(context, rng, "no_worlds_constructed")

        if self._config.leaf_eval == "model":
            return self._search_model(context, worlds, live_fold, rng)

        aggregated: Counter = Counter()
        search_started = time.perf_counter()
        for world, state in worlds:
            result = self._module.monte_carlo_tree_search(
                state, self._config.search_time_ms, threads=self._config.threads
            )
            own_side = (
                result.side_one
                if world.slot_sides[context.player_id] == "side_one"
                else result.side_two
            )
            total = max(result.total_visits, 1)
            for entry in own_side:
                aggregated[entry.move_choice] += entry.visits / total
            self.stats.total_iterations += result.total_visits
            self.stats.worlds_searched += 1
        self.stats.search_wall_seconds += time.perf_counter() - search_started

        action_index = self._map_choices(context, aggregated)
        if action_index is None:
            return self._fallback(context, rng, "choices_unmapped")

        self.stats.searched_decisions += 1
        return PolicyDecision(
            action_index=action_index,
            policy_id=self.policy_id,
            metadata={
                "engine_mcts": {
                    "worlds_searched": len(worlds),
                    "aggregated_choices": {
                        choice: round(weight, 4) for choice, weight in aggregated.most_common()
                    },
                }
            },
        )



    # ------------------------------------------------------------------------------
    # Live incremental root fold (ledger item: "live root-fold export")
    # ------------------------------------------------------------------------------

    def _advance_live_fold(self, context: PolicyContext) -> Any | None:
        """Advance this battle's fold state over the NEW public lines only.

        ``transitions_fold.FoldState`` from ``initial()`` at battle start;
        each decision folds exactly the lines appended to
        ``replay.public_events`` since the previous decision (``|t:|``
        wall-clock lines are filtered inside ``advance_in_place`` — the
        schema-v2 rule). This is the #718-proven production-cheapening path:
        production refolds the WHOLE log per observe; the incremental
        advance is closure-proven and byte-exact over both corpora.

        Tier-2 annotation overlay: when an ``annotation_source`` is attached
        and its trackers are active, the env trackers' per-index conclusions
        are applied at EVERY boundary (``apply_annotations_in_place`` — the
        same per-boundary transition corpus validation replays), so the live
        fold's annotated surfaces (transition/tier2-pinned cells at search
        leaves) match what the env encodes. Without a source the fold stays
        unannotated — correct only for tracker-inactive envs.

        Returns None when this battle's fold is broken (an advance failed,
        the event stream rewound, or an overlay could not be applied) —
        model-mode callers fall back loudly, and the battle stays broken
        rather than searching on silent garbage.
        """
        from .transitions_fold import FoldState  # noqa: PLC0415 — keep import-light

        key = (str(getattr(context, "battle_id", "?")), context.player_id)
        if key in self._fold_broken:
            return None
        replay = context.public_materialization_state.replay
        events = replay.public_events
        fold = self._live_folds.get(key)
        consumed = self._fold_consumed.get(key, 0)
        if fold is None:
            self._drop_stale_folds(key[0])
            fold = FoldState.initial(perspective_slot=context.player_id)
            consumed = 0
        if len(events) < consumed:
            self._mark_fold_broken(context, key, "public event stream rewound")
            return None
        new_lines = [event.raw_line for event in events[consumed:]]
        try:
            fold.advance_in_place(new_lines)
        except Exception as error:  # noqa: BLE001 — loud, then fail closed
            self._mark_fold_broken(
                context, key, f"advance failed: {type(error).__name__}: {error}"
            )
            return None
        if not self._apply_tier2_overlay(context, key, fold):
            return None
        self._live_folds[key] = fold
        self._fold_consumed[key] = len(events)
        self.stats.fold_advanced_lines += len(new_lines)
        if self._config.fold_cross_check:
            self._fold_cross_check(context, fold, replay)
        return fold

    def _apply_tier2_overlay(
        self, context: PolicyContext, key: tuple[str, str], fold: Any
    ) -> bool:
        """Apply the env trackers' conclusions to the live fold (True = ok).

        The source's overlay is CUMULATIVE from battle start; per-boundary
        application keeps every index identifiable (within the action tail or
        the open window — ``FoldState._token_identity``'s contract). Already-
        applied indices are equality-checked by ``apply_annotations_in_place``
        (per-index immutability: a changed tracker conclusion is a real
        regression and breaks the fold loudly). A NEW annotation whose index
        already left the identifiable range would silently desynchronize the
        encoder-visible surface, so it breaks the fold loudly too (cannot
        happen in per-boundary operation — conclusions land at the first
        boundary after their strike).
        """
        source = self._annotation_source
        if source is None or not source.active():
            return True
        try:
            overlay = source.overlay_for(context.player_id)
            if overlay:
                tail_start = fold.action_total - len(fold.action_tail)
                stale = [
                    index
                    for index in overlay
                    if index not in fold.annotations
                    and not tail_start <= index <= fold.action_total
                ]
                if stale:
                    raise ValueError(
                        f"tracker annotations for indices {sorted(stale)[:8]} arrived "
                        f"outside the identifiable range [{tail_start}, "
                        f"{fold.action_total}] — encoder-visible surface would desync."
                    )
                before = len(fold.annotations)
                # The FULL cumulative overlay goes through: already-applied
                # indices are equality-checked inside (per-index immutability
                # — a changed tracker conclusion raises and breaks the fold).
                fold.apply_annotations_in_place(overlay)
                applied = max(0, len(fold.annotations) - before)
                if applied:
                    self.stats.fold_annotations_applied += applied
                    self.stats.fold_annotation_boundaries += 1
        except Exception as error:  # noqa: BLE001 — loud, then fail closed
            self._mark_fold_broken(
                context, key, f"tier2 overlay failed: {type(error).__name__}: {error}"
            )
            return False
        return True

    def _drop_stale_folds(self, battle_id: str) -> None:
        """Free fold state from earlier battles (drivers run one at a time)."""

        for key in [k for k in self._live_folds if k[0] != battle_id]:
            self._live_folds.pop(key, None)
            self._fold_consumed.pop(key, None)
        self._fold_broken = {k for k in self._fold_broken if k[0] == battle_id}

    def _mark_fold_broken(
        self, context: PolicyContext, key: tuple[str, str], reason: str
    ) -> None:
        self._fold_broken.add(key)
        message = (
            f"live-fold BROKEN: battle={key[0]} seat={key[1]} "
            f"round={getattr(context, 'decision_round_index', '?')} reason={reason}"
        )
        warnings.warn(message, EngineSearchFoldMismatchWarning, stacklevel=4)
        _fold_logger.warning(message)

    def _fold_cross_check(self, context: PolicyContext, fold: Any, replay: Any) -> None:
        """Debug gate: live fold products vs the production surfaces.

        With an active annotation source, the reference arm is the ENV's own
        per-player encoder state — the ANNOTATED per-action/merged streams,
        tendency stats, and the pinned Tier-2 reductions (corpus generation's
        production-binding assertion, ``golden_corpus_fold.build_fold_rows``,
        run live). Otherwise the reference is a from-scratch whole-log batch
        refold (``turn_merged.extract_transition_products``); both arms are
        then UNANNOTATED. Mismatches warn loudly and are counted;
        ``strict_fallbacks`` escalates to a hard error.
        """
        self.stats.fold_cross_checks += 1
        products = fold.products()
        source = self._annotation_source
        if source is not None and source.active():
            from .showdown import _normalize_identifier  # noqa: PLC0415

            state = source.boundary_state(context.player_id)
            tokens = tuple(state.transition_tokens)
            merged = tuple(state.turn_merged_tokens)
            tendencies = state.tendency_stats
            # Production's pinned reductions over the FULL annotated stream
            # (showdown.py tier2_cb_pinned_species / tier2_investment_pinned).
            opponent_slot = state.perspective.opponent_showdown_slot
            self_slot = state.perspective.showdown_slot
            want_cb = frozenset(
                _normalize_identifier(token.actor_species)
                for token in tokens
                if token.cb_bit and token.kind == "move" and token.actor_slot == opponent_slot
            )
            want_investment: dict[str, float] = {}
            for token in tokens:
                if (
                    token.investment
                    and token.kind == "move"
                    and token.actor_slot == self_slot
                    and token.defender_species
                ):
                    want_investment[_normalize_identifier(token.defender_species)] = max(
                        -1.0, min(1.0, token.investment)
                    )
            pinned_checks = (
                ("cb_pinned_species", products.cb_pinned_species == want_cb),
                ("investment_pinned", dict(products.investment_pinned) == want_investment),
            )
        else:
            from .turn_merged import extract_transition_products  # noqa: PLC0415

            tokens, merged, tendencies = extract_transition_products(
                replay, perspective_slot=context.player_id
            )
            pinned_checks = ()
        # A non-v2.2 env never builds merged tokens (include_turn_merged off)
        # while the fold always carries them — compare the merged surfaces
        # only when the reference arm has them (or the fold agrees empty).
        merged_checks = (
            (
                ("turn_merged_total", products.turn_merged_total == len(merged)),
                (
                    "turn_merged_tokens",
                    products.turn_merged_tokens == tuple(merged[-fold.merged_tail_limit :]),
                ),
            )
            if merged or products.turn_merged_total == 0
            else ()
        )
        mismatched = [
            name
            for name, ok in (
                ("transition_token_total", products.transition_token_total == len(tokens)),
                (
                    "transition_tokens",
                    products.transition_tokens == tuple(tokens[-fold.action_tail_limit :]),
                ),
                *merged_checks,
                ("tendency_stats", products.tendency_stats == tendencies),
                *pinned_checks,
            )
            if not ok
        ]
        if mismatched:
            self.stats.fold_cross_check_failures += 1
            message = (
                f"live-fold cross-check MISMATCH: battle={getattr(context, 'battle_id', '?')} "
                f"round={getattr(context, 'decision_round_index', '?')} "
                f"seat={context.player_id} surfaces={mismatched}"
            )
            if self._config.strict_fallbacks:
                raise EngineSearchFallbackError(message)
            warnings.warn(message, EngineSearchFoldMismatchWarning, stacklevel=5)
            _fold_logger.warning(message)

    # ------------------------------------------------------------------------------
    # Full in-crate pipeline (leaf_eval="model")
    # ------------------------------------------------------------------------------

    def _root_inputs_json(self, context: PolicyContext) -> str:
        """The crate encoder's sanctioned input surface for the LIVE decision.

        Field-for-field the golden corpus's row-inputs contract
        (``scripts/golden_encoder_backends.row_inputs_from_decision_row``):
        identifiers + the seat's ``observation_metadata`` verbatim + the
        public-materialization payload, built with the same helpers corpus
        generation uses — the crate consumes exactly the surface the
        root-parity gate proved byte-exact.
        """
        from .golden_corpus import _json_safe  # noqa: PLC0415
        from .local_showdown import _public_materialization_payload  # noqa: PLC0415

        state = context.public_materialization_state
        row = {
            "battle_id": str(getattr(context, "battle_id", "")),
            "battle_seed": int(getattr(context, "seed", 0) or 0),
            "format_id": str(getattr(context, "format_id", "")),
            "player_id": context.player_id,
            "observation_schema_version": context.observation.schema_version,
            "observation_metadata": _json_safe(
                dict(context.observation.metadata), context="observation_metadata"
            ),
            "public_materialization": _json_safe(
                _public_materialization_payload(state), context="public_materialization"
            ),
        }
        return json.dumps(row, sort_keys=True)

    def _native(self) -> Any:
        """The in-crate TorchScript search handle, loaded once per policy."""

        if self._native_model is None:
            import pokezero_search  # noqa: PLC0415 — optional native dependency

            if not getattr(pokezero_search, "MODEL_FEATURE_ENABLED", False):
                raise RuntimeError(
                    "pokezero_search was built without the model feature; rebuild via "
                    "scripts/build_search_crate_model.sh before leaf_eval='model'."
                )
            layout = json.loads(self._tables_json or "{}")["layout"]
            self._native_model = pokezero_search.NativeLeafModel(
                str(self._config.model_path),
                device=self._config.model_device,
                window=1,
                tokens=int(layout["token_count"]),
                categorical_features=int(layout["categorical_feature_count"]),
                numeric_features=int(layout["numeric_feature_count"]),
            )
        return self._native_model

    def _search_model(
        self,
        context: PolicyContext,
        worlds: list[tuple[EngineWorld, Any]],
        live_fold: Any,
        rng: random.Random,
    ) -> PolicyDecision:
        """Full in-crate pipeline per belief world.

        Per sampled world: engine state → ``search_batched_multi_encoded``
        (live root fold + per-branch synthesized-event observations +
        TorchScript leaf eval + self-side model priors) → the acting side's
        root visit distribution; distributions aggregate uniformly across
        worlds and map to an action through the same request-candidate
        correspondence as the hp_fraction path. Every failure shape stays
        inside the loud fallback taxonomy (world failures are counted per
        reason; a decision with zero searched worlds falls back).
        """
        import pokezero_search  # noqa: PLC0415 — optional native dependency

        from .observation import OBSERVATION_SCHEMA_VERSION_V2_2  # noqa: PLC0415

        schema = context.observation.schema_version
        if schema != OBSERVATION_SCHEMA_VERSION_V2_2:
            # Misconfiguration, not a per-decision condition: the model was
            # trained on v2.2 observations — never quietly search on another
            # schema's surface.
            raise EngineSearchFallbackError(
                f"leaf_eval='model' requires {OBSERVATION_SCHEMA_VERSION_V2_2} observations; "
                f"this env produced {schema!r}."
            )
        try:
            root_inputs = self._root_inputs_json(context)
            rust_fold = pokezero_search.FoldState.from_payload(live_fold.to_payload())
        except Exception as error:  # noqa: BLE001 — taxonomy, never a crash
            self.stats.world_failure_reasons[
                f"root_inputs: {type(error).__name__}: {str(error)[:120]}"
            ] += 1
            return self._fallback(context, rng, "root_inputs_failed")
        native = self._native()
        replay = context.public_materialization_state.replay
        turn = int(getattr(replay, "turn_number", 0) or 0)
        config = self._config

        aggregated: Counter = Counter()
        worlds_searched_here = 0
        search_started = time.perf_counter()
        for world, state in worlds:
            ctx_json = json.dumps(
                {
                    "p1": list(world.party_species["p1"]),
                    "p2": list(world.party_species["p2"]),
                    "turn": turn,
                }
            )
            world_seed = rng.getrandbits(63)
            try:
                report = json.loads(
                    native.search_batched_multi_encoded(
                        state.to_string(),
                        config.search_sims,
                        config.search_batch,
                        self._tables_json,
                        root_inputs,
                        ctx_json,
                        rust_fold,
                        config.search_depth,
                        config.c_puct,
                        world_seed,
                        config.deep_ko_split,
                        config.model_priors,
                    )
                )
            except Exception as error:  # noqa: BLE001 — count, keep the other worlds
                detail = str(error).splitlines()[0][:160] if str(error) else type(error).__name__
                self.stats.world_failure_reasons[f"crate_search: {detail}"] += 1
                continue
            side_key = (
                "side_one"
                if world.slot_sides[context.player_id] == "side_one"
                else "side_two"
            )
            entries = report[side_key]
            total = max(sum(entry["visits"] for entry in entries), 1)
            for entry in entries:
                aggregated[entry["move"]] += entry["visits"] / total
            self.stats.total_iterations += int(report["iterations"])
            self.stats.model_evals += int(report["model_evals"])
            self.stats.lossy_renders += int(report.get("lossy_renders") or 0)
            self.stats.prior_fallbacks += int(report.get("prior_fallbacks") or 0)
            self.stats.worlds_searched += 1
            worlds_searched_here += 1
        self.stats.search_wall_seconds += time.perf_counter() - search_started

        if not worlds_searched_here:
            return self._fallback(context, rng, "crate_search_failed")
        action_index = self._map_choices(context, aggregated)
        if action_index is None:
            return self._fallback(context, rng, "choices_unmapped")
        self.stats.searched_decisions += 1
        return PolicyDecision(
            action_index=action_index,
            policy_id=self.policy_id,
            metadata={
                "engine_mcts": {
                    "leaf_eval": "model",
                    "worlds_searched": worlds_searched_here,
                    "aggregated_choices": {
                        choice: round(weight, 4) for choice, weight in aggregated.most_common()
                    },
                }
            },
        )

    # Gen 3 pool's only recharge move; the recharge turn itself is public.
    _RECHARGE_MOVES = frozenset({"hyperbeam"})

    def _recharging_slots(self, context: PolicyContext) -> tuple[str, ...]:
        """Slots publicly forced to recharge THIS turn (Hyper Beam landed last round).

        Turn-exact signal: the round-indexed public action record (not the
        rolling event window) must show the opponent's action in the
        immediately-preceding round was a recharge move, and the rolling
        window must not carry a miss marker for it (a missed Hyper Beam does
        not recharge in gen3). If the record is unavailable the signal stays
        off — fail-open to the pre-fix behavior rather than inventing a lock.
        """

        opponent_slot = "p2" if context.player_id == "p1" else "p1"
        trajectory = getattr(context, "trajectory", None)
        round_index = getattr(context, "decision_round_index", None)
        if trajectory is None or not isinstance(round_index, int):
            return ()
        rounds = public_action_rounds_from_trajectory_metadata(trajectory)
        previous = rounds.get(round_index - 1)
        if previous is None:
            return ()
        action = previous.actions.get(opponent_slot)
        if action is None or action.kind != "move":
            return ()
        if normalize_id(str(action.move_id or "")) not in self._RECHARGE_MOVES:
            return ()
        # The round record proves the move happened but stores no hit/miss and
        # no actor identity. Require the ANCHOR: the |move| line must still be
        # visible in the rolling event window, its actor must match the
        # CURRENT active opponent (species continuity — double-faint guard),
        # and no adjacent |-miss| may follow. If the anchor scrolled out we
        # cannot verify the hit, so the lock stays OFF (fail-open to the
        # pre-fix behavior — never a wrong lock on a missed Hyper Beam).
        metadata = context.observation.metadata
        if not isinstance(metadata, Mapping):
            return ()
        belief_view = metadata.get("belief_view")
        opponents = belief_view.get("opponent_pokemon") if isinstance(belief_view, Mapping) else None
        active_species = next(
            (
                str(mon.get("species") or "")
                for mon in opponents or ()
                if isinstance(mon, Mapping) and mon.get("active")
            ),
            "",
        )
        if not active_species:
            return ()
        events = metadata.get("recent_public_events")
        if not isinstance(events, Sequence):
            return ()
        lines = [str(line) for line in events]
        for index in range(len(lines) - 1, -1, -1):
            parts = lines[index].split("|")
            if len(parts) < 4 or parts[1] != "move":
                continue
            if normalize_id(parts[3]) not in self._RECHARGE_MOVES:
                continue
            actor = parts[2]
            actor_species = actor.split(":", 1)[-1].strip() if ":" in actor else actor
            if normalize_id(actor_species) != normalize_id(active_species):
                return ()
            if not actor.strip().lower().startswith(opponent_slot):
                return ()
            if any(rest.startswith(f"|-miss|{actor}") for rest in lines[index + 1 : index + 3]):
                return ()
            return (opponent_slot,)
        return ()


    def _truant_loaf_slots(self, context: PolicyContext) -> tuple[str, ...]:
        """Slots whose active is a Truant mon that ACTED last round (loafs now).

        The alternation is public: a Truant mon that publicly moved in the
        immediately-preceding round loafs this turn. Evidence of acting comes
        from the round-indexed public action record (turn-exact). Without
        clear acted-last-round evidence the volatile stays off (fail-open:
        the mon is modeled as free to act — the pre-fix behavior).
        """

        trajectory = getattr(context, "trajectory", None)
        round_index = getattr(context, "decision_round_index", None)
        if trajectory is None or not isinstance(round_index, int):
            return ()
        rounds = public_action_rounds_from_trajectory_metadata(trajectory)
        previous = rounds.get(round_index - 1)
        if previous is None:
            return ()
        metadata = context.observation.metadata
        if not isinstance(metadata, Mapping):
            return ()
        slots: list[str] = []
        opponent_slot = "p2" if context.player_id == "p1" else "p1"
        belief_view = metadata.get("belief_view")
        opponents = belief_view.get("opponent_pokemon") if isinstance(belief_view, Mapping) else None
        for mon in opponents or ():
            if not isinstance(mon, Mapping) or not mon.get("active"):
                continue
            ability = normalize_id(str(mon.get("revealed_ability") or ""))
            possible = [normalize_id(str(a)) for a in mon.get("possible_abilities") or ()]
            if ability == "truant" or (not ability and possible == ["truant"]):
                action = previous.actions.get(opponent_slot)
                if action is not None and action.kind == "move":
                    slots.append(opponent_slot)
        # Self seat: our own Truant mon's phase from our own action record.
        self_team = metadata.get("self_team")
        if isinstance(self_team, Sequence):
            for row in self_team:
                if not isinstance(row, Mapping) or not row.get("active"):
                    continue
                if normalize_id(str(row.get("ability") or "")) == "truant":
                    action = previous.actions.get(context.player_id)
                    if action is not None and action.kind == "move":
                        slots.append(context.player_id)
        return tuple(slots)

    def _public_effect_signals(
        self, context: PolicyContext
    ) -> tuple[dict[str, str], dict[str, str], dict[str, tuple[str, ...]]]:
        """Public-information signals engine_world cannot see in the payload.

        - blocked_slots: the opponent's active is publicly Transformed (the
          belief engine tracks it; the payload does not) — the sampled world
          cannot express the copied moveset/stats, so construction must fail
          closed rather than search a silently wrong world.
        - encored_moves: the opponent's publicly-observed last move, consumed
          by engine_world only when that side carries the encore volatile.
        - removed_item_species: per slot, species whose held item was publicly
          STRIPPED (Knock Off, or a Trick that took the item and returned
          none) and not replaced since — the current public item state is
          exactly "no item", which the sampled world expresses by clearing
          the sampled set's item. Only true swaps (``item_mutated`` without
          ``item_removed``: the holder carries an item that is not the
          sampled assignment) stay fail-closed.
        """

        blocked: dict[str, str] = {}
        encored: dict[str, str] = {}
        removed: dict[str, tuple[str, ...]] = {}
        metadata = context.observation.metadata
        if not isinstance(metadata, Mapping):
            return blocked, encored, removed
        opponent_slot = "p2" if context.player_id == "p1" else "p1"
        belief_view = metadata.get("belief_view")
        opponents = belief_view.get("opponent_pokemon") if isinstance(belief_view, Mapping) else None
        active_species: str | None = None
        for mon in opponents or ():
            if not isinstance(mon, Mapping):
                continue
            if mon.get("item_mutated"):
                if mon.get("item_removed"):
                    species_id = normalize_id(str(mon.get("species") or ""))
                    if species_id:
                        removed[opponent_slot] = removed.get(opponent_slot, ()) + (species_id,)
                else:
                    # A live Trick swap: the current holder's item is not the
                    # sampled set's item and rule-outs stay frozen to the
                    # ORIGINAL assignment upstream — no sampled world can
                    # express it, so construction fails closed.
                    blocked[opponent_slot] = f"item mutated on {mon.get('species')}"
            if not mon.get("active"):
                continue
            active_species = str(mon.get("species") or "") or None
            if mon.get("transformed"):
                target = mon.get("transform_species") or "?"
                blocked[opponent_slot] = f"active transformed into {target}"
        if active_species:
            events = metadata.get("recent_public_events")
            for line in reversed(list(events) if isinstance(events, Sequence) else []):
                move = _move_from_public_event_line(
                    str(line),
                    opponent_slot=opponent_slot,
                    self_slot=context.player_id,
                    species=active_species,
                )
                if move is not None:
                    encored[opponent_slot] = move
                    break
        return blocked, encored, removed

    def _map_choices(
        self, context: PolicyContext, aggregated: Mapping[str, float]
    ) -> Optional[int]:
        candidates = context.observation.metadata.get("action_candidates")
        if not isinstance(candidates, Sequence):
            return None
        mask = context.observation.legal_action_mask

        move_index_by_id: dict[str, int] = {}
        hidden_power_index: Optional[int] = None
        switch_index_by_species: dict[str, int] = {}
        switch_index_by_canonical: dict[str, int] = {}
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or not candidate.get("legal"):
                continue
            index = candidate.get("action_index")
            if not isinstance(index, int) or not (0 <= index < len(mask)) or not mask[index]:
                continue
            if candidate.get("kind") == "move":
                move_id = normalize_id(str(candidate.get("move_id") or ""))
                if move_id:
                    move_index_by_id[move_id] = index
                    if move_id.startswith("hiddenpower"):
                        hidden_power_index = index
            elif candidate.get("kind") == "switch":
                pokemon = candidate.get("pokemon")
                species = (
                    normalize_id(str(pokemon.get("species") or ""))
                    if isinstance(pokemon, Mapping)
                    else ""
                )
                if species:
                    switch_index_by_species[species] = index
                    # Cosmetic-forme tolerance: the engine displays the
                    # collapsed base id ("switch unown") while the request
                    # candidate carries the lettered forme ("Unown-C");
                    # species clause keeps the canonical key unique per team.
                    switch_index_by_canonical[
                        canonical_gen3_randbat_species_id(species)
                    ] = index

        best_index: Optional[int] = None
        best_weight = 0.0
        for choice, weight in aggregated.items():
            index: Optional[int] = None
            if choice.startswith("switch "):
                species = normalize_id(choice[len("switch "):])
                index = switch_index_by_species.get(species)
                if index is None:
                    index = switch_index_by_canonical.get(
                        canonical_gen3_randbat_species_id(species)
                    )
            else:
                move_id = normalize_id(choice)
                index = move_index_by_id.get(move_id)
                if index is None and move_id.startswith("hiddenpower"):
                    # Engine ids are typed+BP; the request reports plain "hiddenpower".
                    index = hidden_power_index
            if index is None:
                self.stats.unmapped_choices[choice] += 1
                continue
            if weight > best_weight:
                best_weight = weight
                best_index = index
        return best_index

    def _fallback(
        self, context: PolicyContext, rng: random.Random, reason: str
    ) -> PolicyDecision:
        self.stats.fallback_decisions += 1
        self.stats.fallback_reasons[reason] += 1
        battle_id = getattr(context, "battle_id", "?")
        round_index = getattr(context, "decision_round_index", "?")
        player = getattr(context, "player_id", "?")
        # Per-decision world-failure context: the cumulative counters minus
        # the snapshot taken at the top of _search.
        delta = {
            key: count - self._world_failures_before.get(key, 0)
            for key, count in self.stats.world_failure_reasons.items()
            if count - self._world_failures_before.get(key, 0) > 0
        }
        message = (
            f"engine-search FALLBACK: battle={battle_id} round={round_index} seat={player} "
            f"reason={reason} world_failures={delta or '{}'}"
        )
        if self._config.strict_fallbacks:
            raise EngineSearchFallbackError(message)
        warnings.warn(message, EngineSearchFallbackWarning, stacklevel=3)
        _fallback_logger.warning(message)
        legal = legal_action_indices(context.observation.legal_action_mask)
        return PolicyDecision(
            action_index=rng.choice(legal),
            policy_id=self.policy_id,
            metadata={"engine_mcts": {"fallback": reason}},
        )


# ---------------------------------------------------------------------------------------------
# Bench CLI.
# ---------------------------------------------------------------------------------------------


class _ArgmaxComparePolicy:
    """Bench-only wrapper: the primary (model-mode) policy drives the game;
    the reference (hp_fraction engine MCTS) is ALSO asked on the first
    ``limit`` decisions and both argmax choices are recorded.

    Sanity contract: both decisions must be LEGAL under the request mask;
    AGREEMENT IS NOT EXPECTED — the two modes price leaves with different
    evaluations by design. The record shows both, per the gate's honesty rule.
    """

    def __init__(self, primary: Any, reference: Any, *, limit: int, records: list) -> None:
        self.primary = primary
        self.reference = reference
        self.limit = limit
        self.records = records
        self.policy_id = primary.policy_id

    def select_action(self, observation: Any, *, rng: random.Random) -> PolicyDecision:
        return self.primary.select_action(observation, rng=rng)

    def select_action_with_context(
        self, context: PolicyContext, *, rng: random.Random
    ) -> PolicyDecision:
        decision = self.primary.select_action_with_context(context, rng=rng)
        if len(self.records) < self.limit:
            mask = context.observation.legal_action_mask
            reference_rng = random.Random(
                (int(getattr(context, "seed", 0) or 0) * 1000003)
                + int(getattr(context, "decision_round_index", 0) or 0)
            )
            reference = self.reference.select_action_with_context(context, rng=reference_rng)
            primary_meta = (decision.metadata or {}).get("engine_mcts", {})
            reference_meta = (reference.metadata or {}).get("engine_mcts", {})
            self.records.append(
                {
                    "battle_id": str(getattr(context, "battle_id", "?")),
                    "round": getattr(context, "decision_round_index", None),
                    "model_action": decision.action_index,
                    "model_legal": bool(mask[decision.action_index]),
                    "hp_fraction_action": reference.action_index,
                    "hp_fraction_legal": bool(mask[reference.action_index]),
                    "agree": decision.action_index == reference.action_index,
                    "model_fallback": primary_meta.get("fallback"),
                    "hp_fraction_fallback": reference_meta.get("fallback"),
                }
            )
        return decision


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bench: engine MCTS over belief worlds (hp_fraction or full model pipeline)"
    )
    parser.add_argument("--showdown-root", required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=7000)
    parser.add_argument("--opponent", choices=("random-legal", "simple-legal"), default="simple-legal")
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--search-time-ms", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--fail-on-fallback", action="store_true",
                        help="exit nonzero if any decision fell back (CI gate)")
    parser.add_argument("--strict-sleep", action="store_true",
                        help="fail worlds closed on publicly-asleep mons instead of approximating")
    parser.add_argument("--out", default=None)
    # --- full in-crate pipeline (leaf_eval="model") ---
    parser.add_argument("--leaf-eval", choices=("hp-fraction", "model"), default="hp-fraction")
    parser.add_argument("--model-path", default=None,
                        help="TorchScript artifact (scripts/export_model.py)")
    parser.add_argument("--tables", default=None,
                        help="encoder tables JSON (scripts/export_encoder_tables.py)")
    parser.add_argument("--model-device", default="cpu")
    parser.add_argument("--sims", type=int, default=256,
                        help="per-world simulation budget (model mode)")
    parser.add_argument("--batch", type=int, default=16,
                        help="leaf-eval batch size (model mode; keep << sims)")
    parser.add_argument("--depth", type=int, default=2,
                        help="max decision plies (model mode)")
    parser.add_argument("--no-model-priors", action="store_true",
                        help="model mode with uniform priors (A/B kill switch)")
    parser.add_argument("--fold-cross-check", action="store_true",
                        help="debug: batch-refold vs the live incremental fold per decision")
    parser.add_argument("--argmax-compare", type=int, default=0,
                        help="model mode: also run the hp_fraction policy on the first N "
                             "decisions and record both argmaxes (legality sanity, not agreement)")
    args = parser.parse_args(argv)

    from .dex import load_showdown_dex
    from .local_showdown import LocalShowdownConfig, LocalShowdownEnv
    from .policy import RandomLegalPolicy, SimpleLegalPolicy
    from .randbat import Gen3RandbatSource
    from .rollout import RolloutConfig, RolloutDriver

    model_mode = args.leaf_eval == "model"
    dex = load_showdown_dex(args.showdown_root)
    set_source = Gen3RandbatSource.from_showdown_root(args.showdown_root)
    base_config = dict(
        worlds=args.worlds,
        search_time_ms=args.search_time_ms,
        threads=args.threads,
        approximate_sleep_turns=not args.strict_sleep,
    )
    config = EngineMctsConfig(
        **base_config,
        leaf_eval="model" if model_mode else "hp_fraction",
        model_path=args.model_path,
        tables_path=args.tables,
        model_device=args.model_device,
        search_sims=args.sims,
        search_batch=args.batch,
        search_depth=args.depth,
        model_priors=not args.no_model_priors,
        fold_cross_check=args.fold_cross_check,
    )
    # Model mode needs the belief candidate-set source: the v2.2 observation's
    # belief columns (candidate variants, possible sets) are part of the
    # surface the model was trained on. With the source attached and the
    # default tier2_residuals mask, the env's Tier-2 trackers are ACTIVE —
    # the policy therefore needs the annotation source (below) or its live
    # fold would present unannotated Tier-2 surfaces at search leaves.
    env_config = LocalShowdownConfig(
        showdown_root=args.showdown_root,
        set_belief_source=True if model_mode else None,
    )
    env = LocalShowdownEnv(env_config)
    annotation_source = EnvTier2AnnotationSource(env)
    policy = EngineMctsPolicy(
        dex=dex, set_source=set_source, config=config, annotation_source=annotation_source
    )
    compare_records: list[dict[str, Any]] = []
    p1_policy: Any = policy
    if model_mode and args.argmax_compare > 0:
        reference = EngineMctsPolicy(
            dex=dex, set_source=set_source, config=EngineMctsConfig(**base_config)
        )
        p1_policy = _ArgmaxComparePolicy(
            policy, reference, limit=args.argmax_compare, records=compare_records
        )
    opponent = RandomLegalPolicy() if args.opponent == "random-legal" else SimpleLegalPolicy()
    driver = RolloutDriver(
        env=env,
        policies={"p1": p1_policy, "p2": opponent},
        config=RolloutConfig(format_id="gen3randombattle"),
    )
    wins = 0
    games = []
    try:
        for offset in range(args.games):
            seed = args.seed_start + offset
            decisions_before = policy.stats.decisions
            fallbacks_before = policy.stats.fallback_decisions
            removed_before = policy.stats.removed_item_decisions
            fallback_reasons_before = Counter(policy.stats.fallback_reasons)
            world_failures_before = Counter(policy.stats.world_failure_reasons)
            result = driver.run(seed=seed, battle_id=f"engine-mcts-bench-{seed}")
            won = result.terminal.winner == "p1"
            wins += int(won)
            game_fallbacks = policy.stats.fallback_decisions - fallbacks_before
            games.append({
                "seed": seed,
                "winner": result.terminal.winner,
                "decision_rounds": result.decision_round_count,
                # Per-battle attribution: fallback walls cluster per battle
                # (an item mutation or Transform fails worlds closed for the
                # REST of that battle), so per-seed deltas are the surface
                # that localizes them.
                "decisions": policy.stats.decisions - decisions_before,
                "fallback_decisions": game_fallbacks,
                "removed_item_decisions": policy.stats.removed_item_decisions - removed_before,
                "fallback_reasons": dict(
                    Counter(policy.stats.fallback_reasons) - fallback_reasons_before
                ),
                "world_failure_reasons": dict(
                    Counter(policy.stats.world_failure_reasons) - world_failures_before
                ),
            })
            print(
                f"seed {seed}: winner={result.terminal.winner} rounds={result.decision_round_count}"
                + (f" fallbacks={game_fallbacks}" if game_fallbacks else "")
            )
    finally:
        env.close()

    report = {
        "config": {
            "worlds": args.worlds,
            "search_time_ms": args.search_time_ms,
            "threads": args.threads,
            "approximate_sleep_turns": not args.strict_sleep,
            "opponent": args.opponent,
            "games": args.games,
            "leaf_eval": config.leaf_eval,
            "model_path": args.model_path,
            "sims": args.sims,
            "batch": args.batch,
            "depth": args.depth,
            "model_priors": config.model_priors,
        },
        "wins": wins,
        "win_rate": wins / args.games if args.games else 0.0,
        "games": games,
        "engine_mcts": policy.stats.to_dict(),
    }
    if compare_records:
        illegal = [r for r in compare_records if not (r["model_legal"] and r["hp_fraction_legal"])]
        agreements = sum(1 for r in compare_records if r["agree"])
        report["argmax_compare"] = {
            "decisions": len(compare_records),
            "agreements": agreements,
            "illegal_decisions": len(illegal),
            "records": compare_records,
        }
        print(
            f"argmax compare: {agreements}/{len(compare_records)} agree, "
            f"{len(illegal)} illegal decisions (must be 0; agreement not required)"
        )
    printable = {k: v for k, v in report.items() if k != "games"}
    if "argmax_compare" in printable:
        printable["argmax_compare"] = {
            k: v for k, v in printable["argmax_compare"].items() if k != "records"
        }
    print(json.dumps(printable, indent=2))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2)
    fallback_count = policy.stats.fallback_decisions
    if fallback_count:
        import sys as _sys

        print(
            f"\n{'!' * 72}\n!! {fallback_count} FALLBACK DECISION(S) — reasons: "
            f"{dict(policy.stats.fallback_reasons)}\n"
            f"!! attribute every fallback via world_failure_reasons before accepting a run\n{'!' * 72}",
            file=_sys.stderr,
        )
        if args.fail_on_fallback:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
