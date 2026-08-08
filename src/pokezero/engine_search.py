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
  synthesized events, per-outcome fold advance, checkpoint-latched native
  leaf encode,
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
import hashlib
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
from .poke_engine_adapter import PokeEngineAttractUnsupportedError, build_poke_engine_state
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
    battles where the opponent publicly Transforms fail worlds closed for
    the rest of the battle by design — both leaf-eval modes hit the same
    wall on the same battles. Item mutations no longer wall: Knock-Off
    REMOVALS and public consumptions clear the sampled item
    (``removed_item_species``), and Trick SWAPS substitute the
    protocol-confirmed current item (``current_item_overrides``); only a
    mutation with no confirmed current item still fails closed.)
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


def _root_toxic_zero_after_upkeep_attestation(replay: object) -> dict[str, dict[str, bool | None]]:
    """Serialize only exact proof booleans for the Rust root handoff.

    The leaf has no replay snapshot to inspect.  Preserve malformed values as
    JSON ``null`` rather than coercing them with ``bool(...)`` so its decoder
    can fail closed before creating a Toxic re-entry latch.
    """

    def exact_bool_field(name: str, slot: str) -> bool | None:
        values = getattr(replay, name, None)
        value = values.get(slot) if isinstance(values, Mapping) else None
        return value if type(value) is bool else None

    post_upkeep_window = getattr(replay, "post_upkeep_window", None)
    exact_post_upkeep_window = (
        post_upkeep_window if type(post_upkeep_window) is bool else None
    )
    return {
        slot: {
            "proof": exact_bool_field("toxic_stage_zero_after_upkeep", slot),
            "pending": exact_bool_field("toxic_faint_replacement_pending", slot),
            "invalid": exact_bool_field("toxic_faint_replacement_invalid", slot),
            "post_upkeep_window": exact_post_upkeep_window,
        }
        for slot in ("p1", "p2")
    }


def _checkpoint_feature_masks_payload(model_config: Any) -> dict[str, Any]:
    """Encoder-table mask payload derived from checkpoint provenance."""

    return {
        "stats_block": bool(model_config.stats_block_enabled),
        "exact_state": bool(model_config.exact_state_enabled),
        "transition_token_budget": int(model_config.transition_token_budget),
        "tier2_residuals": bool(model_config.tier2_residuals),
        "tier2_investment": bool(model_config.tier2_investment),
    }


def _latch_encoder_tables_to_model_config(tables_json: str, model_config: Any) -> str:
    """Bind native leaf encoding to the checkpoint's exact observation contract.

    Encoder-table exports describe a schema's full capacity and carry library-default masks.
    A checkpoint may have trained on a narrower history or another mask combination. Search
    leaves must use that trained contract rather than the exporter defaults, or the root and
    leaves can disagree inside one tree.
    """

    try:
        payload = json.loads(tables_json)
        layout = payload["layout"]
        masks = layout["default_feature_masks"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid encoder tables contract: {error}") from error
    if not isinstance(layout, dict) or not isinstance(masks, dict):
        raise ValueError("encoder tables layout/default_feature_masks must be objects.")

    expected = {
        "schema_version": str(model_config.observation_schema_version),
        "token_count": int(model_config.token_count),
        "categorical_feature_count": int(model_config.categorical_feature_count),
        "numeric_feature_count": int(model_config.numeric_feature_count),
    }
    mismatches = {
        key: {"checkpoint": value, "tables": layout.get(key)}
        for key, value in expected.items()
        if layout.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "encoder tables do not match the checkpoint observation contract: "
            + json.dumps(mismatches, sort_keys=True)
        )

    latched_masks = _checkpoint_feature_masks_payload(model_config)
    layout["default_feature_masks"] = latched_masks
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _locked_aggregate_choice(
    world_reports: Sequence[tuple[str, Mapping[str, Any]]],
) -> Optional[str]:
    """Return the final aggregate argmax only when skipped visits cannot change it."""

    lower: Counter[str] = Counter()
    upper: Counter[str] = Counter()
    choices: set[str] = set()
    for side_key, report in world_reports:
        requested = int(report.get("requested_iterations", report.get("iterations", 0)))
        completed = int(report.get("iterations", 0))
        if requested <= 0 or completed < 0 or completed > requested:
            return None
        remaining = requested - completed
        entries = report.get(side_key)
        if not isinstance(entries, Sequence):
            return None
        visit_total = 0
        for entry in entries:
            if not isinstance(entry, Mapping):
                return None
            choice = str(entry.get("move") or "")
            visits = int(entry.get("visits", -1))
            if not choice or visits < 0:
                return None
            visit_total += visits
            choices.add(choice)
            # Each world's final aggregate weight is normalized by its full
            # requested budget. Every skipped visit may conservatively land
            # on any arm when constructing that arm's upper bound.
            lower[choice] += visits / requested
            upper[choice] += (visits + remaining) / requested
        if visit_total != completed:
            return None
    if not choices:
        return None
    leader = max(choices, key=lambda choice: lower[choice])
    if all(lower[leader] > upper[choice] for choice in choices if choice != leader):
        return leader
    return None


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
    # Permit a public Substitute only when the replay proves it is freshly
    # created at floor(maxhp/4). A surviving hit makes its remaining HP
    # unknowable, so engine_world fails closed instead of approximating it.
    approximate_substitute_health: bool = True
    # Documented approximation: a public partial trap (Wrap and kin) is modeled
    # with the engine's own no-duration shape, which holds the trap until the
    # trapper switches instead of the real 2-5 turn roll. Pessimistic about
    # escaping, but a searched world beats the uniform-legal fallback it
    # replaces (96 world failures in the 2026-07-26 depth study).
    approximate_partial_trap_turns: bool = True
    # Documented approximation: public confusion and Yawn are searched with the
    # engine's own clock, because the payload carries no remaining duration.
    # Confusion never expires inside a search (pessimistic); Yawn's counter
    # starts at 0, so an already-aged Yawn sleeps a turn late.
    approximate_hidden_duration_volatiles: bool = True
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
    # "hp_fraction_crate": the CRATE's own multi-ply PUCT tree
    # (pokezero_search.puct_search_multi) with the handcrafted HP-fraction leaf
    # evaluator instead of the model. Measurement instrument for the depth-decay
    # study (docs/mcts_handcrafted_leaf_depth_findings.md): it holds the search
    # STRUCTURE fixed — identical traverse/expand/finalize, identical
    # decision/chance node shape, identical depth and c_puct semantics — and
    # swaps only the leaf value function, which is what separates "the defect is
    # in the tree" from "the defect is in the learned value".
    leaf_eval: str = "hp_fraction"
    # TorchScript artifact (scripts/export_model.py; per-device trace — a CPU
    # artifact must run on cpu) and the encoder tables JSON
    # (scripts/export_encoder_tables.py), plus the source checkpoint whose
    # observation contract must govern both root and leaf encodes.
    model_path: str | None = None
    checkpoint_path: str | None = None
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
    # Seed the OPPONENT seat's PUCT priors from the checkpoint's opponent
    # action head instead of leaving them uniform. Default OFF: flag-off is
    # the uniform-opponent search every recorded result was produced under.
    #
    # The opponent head has always been exported and batched (export_model.py
    # OUTPUT_NAMES) and was discarded in the crate; the uniform-opponent design
    # is a known modelling gap that findings 13.4 cleared of causing the SEAT
    # RESIDUAL but did not clear as harmless against an EXTERNAL opponent,
    # whose non-uniform policy is exactly what uniform play mismodels.
    use_opponent_priors: bool = False
    # Opt-in safe STOP rule. A tree may stop at a completed batch only after
    # this floor and only when the unspent simulations cannot change its root
    # visit argmax. Multi-world aggregation applies a second safety bound.
    early_stop: bool = False
    early_stop_min_sims: int = 64
    # Debug cross-check: per decision, batch-refold the whole public log
    # (production's per-observe path, turn_merged.extract_transition_products)
    # and compare its surfaces against the live incremental fold's products.
    fold_cross_check: bool = False

    def __post_init__(self) -> None:
        if self.worlds <= 0 or self.search_time_ms <= 0 or self.threads <= 0:
            raise ValueError("worlds, search_time_ms, and threads must be positive.")
        if self.leaf_eval not in ("hp_fraction", "hp_fraction_crate", "model"):
            raise ValueError(
                "leaf_eval must be 'hp_fraction', 'hp_fraction_crate' or 'model', "
                f"got {self.leaf_eval!r}."
            )
        if self.leaf_eval == "hp_fraction_crate":
            if self.search_sims <= 0 or self.search_depth <= 0:
                raise ValueError(
                    "search_sims and search_depth must be positive for "
                    "leaf_eval='hp_fraction_crate'."
                )
        if self.leaf_eval == "model":
            if not self.model_path or not self.checkpoint_path or not self.tables_path:
                raise ValueError(
                    "leaf_eval='model' requires model_path, checkpoint_path, and tables_path."
                )
            if self.search_sims <= 0 or self.search_batch <= 0 or self.search_depth <= 0:
                raise ValueError(
                    "search_sims, search_batch, and search_depth must be positive."
                )
            if self.search_batch > self.search_sims:
                raise ValueError(
                    "search_batch must be <= search_sims (keep batch << sims; "
                    "docs/crate_search_design.md review caveats)."
                )
            if self.early_stop and not 0 < self.early_stop_min_sims <= self.search_sims:
                raise ValueError(
                    "early_stop_min_sims must be in 1..=search_sims when early_stop is enabled."
                )
        elif self.early_stop:
            raise ValueError("early_stop is supported only with leaf_eval='model'.")


def _world_failure_key(error: EngineWorldUnsupported) -> str:
    """Return a telemetry key that keeps the CAUSE while bounding cardinality.

    Fallback attribution is only useful at the granularity you can act on. An
    allowlist of reasons-that-carry-detail used to leave the rest as bare slugs,
    which is how ``materialization_blocker`` reached 2811 failures in the
    2026-07-26 study with no indication of WHICH blocker — the taxonomy that
    would have named the fix on day one. So detail is now the default.

    Cardinality is the reason the allowlist existed, and it is real: details
    naming a species or a slot would mint a key per species. Rather than drop
    those causes, strip the per-instance operand and keep the token KIND, so
    ``item-state-removed:Zapdos`` and ``item-state-removed:Snorlax`` share one
    actionable bucket.
    """

    detail = error.detail
    if error.reason == "materialization_blocker":
        _, _, tokens = detail.partition(":")
        kinds = sorted(
            {_blocker_bucket(str(token)) for token in tokens.split(",") if token.strip()}
        )
        return f"{error.reason}: {', '.join(kinds)}" if kinds else error.reason
    return f"{error.reason}: {detail}"


def _blocker_bucket(token: str) -> str:
    """Bucket one blocker token at the granularity that names its fix.

    Stripping the operand is right for ``item-state-*``, whose operand is a
    species -- one bucket per species is unbounded noise and the species is not
    what you would change. It is wrong for ``baton-pass``, where the operand is
    the VOLATILE and is the entire actionable content: the 2026-07-28 power run
    reported a bare ``materialization_blocker: baton-pass`` and gave no way to
    tell whether it was a Substitute worth supporting or a Bide worth refusing.
    The volatile ids are a small closed set, so keeping them stays bounded.
    """

    kind, _, operand = token.strip().partition(":")
    if kind == "baton-pass" and operand:
        return f"{kind}:{operand}"
    return kind


# Budget for one crate-error `world_failure_reasons` key. The refusal message is
# `attribution-unsafe renderer branch rejected before <lane>: <slugs>`, whose prefix
# alone eats ~68 chars, and two sides refusing with DIFFERENT slug sets is routine.
#
# THE ORIGINAL RATIONALE NO LONGER HOLDS, and is recorded rather than deleted because
# the NUMBER it justified is still the right one. It read: "Was 160, which the attract
# sub-case split (#1030) outgrew ... one fully-live attract slug is 68 more
# (`attract_empty_tail_ambiguous:paralyzed+miss+noop+volatile+cannot_act`)". That slug
# is no longer emittable at all — the immobilizer-marker change deleted
# `attract_empty_tail_ambiguous` and its five sub-case literals, because the engine now
# marks both move-time immobilizers and there is nothing left to refuse.
#
# So the 512 is now bounded by the SLEEP TALK family instead, whose slug composes one
# token per blocked effect family. That composition, not the retired attract one, is
# what a future widening has to be measured against — and `events.rs`'s
# `the_attribution_unsafe_label_is_deduplicated_and_sorted` pins the budget from the
# crate side, against the worst case the production order list can compose, so a slug
# that outgrows this constant fails there rather than silently truncating here.
_REASON_DETAIL_LIMIT = 512

# Addresses retained per fallback CLASS. Three is enough to replay one, confirm it
# reproduces, and see whether a second instance is the same shape.
#
# Bounds the report at 3 x distinct keys. That multiplicand is DATA-DEPENDENT, not a
# fixed taxonomy: crate reason keys interpolate operands (species in `'Pelipper' has
# public Rest skippedTime`, and engine_world.py mints keys carrying turn numbers and
# max-HP values), so the key space is bounded in LENGTH by _bounded_reason_detail but
# never in COUNT. Era 57 MEASURED 79 distinct raw keys (72 after the analyzer collapses
# the `crate_search: ` prefix), so ~237 small entries. Both counts come from
# analyze_probe.py over the era-57 shard reports, which survive on the shared PVC --
# it is the fallback ADDRESSES that were lost with the pods, since those were only ever
# written to pod stdout. An earlier version of this comment said the reports died too,
# and called the raw count unmeasured on that basis; it took one query to disprove.
#
# 79 is comfortably under the ceiling below, but it is a MEASUREMENT of one campaign,
# not a property of the key space, which is why the ceiling exists. Independently
# corroborated during review from 124 world_failure_reasons dumps recovered from pod
# logs on a different (51-patch) build: 33 raw -> 30 collapsed, a 1.10x ratio against
# this 79/72 = 1.097. Corroboration, not reproduction -- that recovery had no PVC
# access. In those 33 keys, 19 carried a species, 1 a move, and NONE carried a number,
# so the turn-number and max-HP key generators have zero observed instances.
_FALLBACK_SAMPLES_PER_CLASS = 3

# Ceiling on distinct keys, so the report size is bounded UNCONDITIONALLY rather than
# by however many classes the data happens to mint. 256 is ~3x the 79 raw keys era 57
# actually produced. Reason keys are EXEMPT (see _fallback), so the key total is the
# ceiling plus at most 7 and the worst case is ~790 entries -- measured 262 keys / 786
# entries / 124 KB. Overflow is COUNTED and emitted, never silently dropped: a truncated
# sample that looks complete is how a coverage claim goes wrong.
_FALLBACK_SAMPLE_KEY_CEILING = 256


def _bounded_reason_detail(text: str, limit: int = _REASON_DETAIL_LIMIT) -> str:
    """Bound a telemetry reason so overflow is VISIBLE, never silently aliasing.

    A plain ``text[:limit]`` is the wrong instrument for a measurement key. It
    drops the tail without saying so, and — worse — two different reasons that
    share a prefix collapse onto the SAME key: the attract sub-case sets
    ``{paralyzed+cannot_act, paralyzed+miss}`` and
    ``{paralyzed+cannot_act, paralyzed+miss+volatile}`` both truncated to the
    identical ``...paralyzed+mis`` bucket at 160. The `+volatile` arm is exactly
    the non-downgradeable mass the split exists to find, so the old seam could
    hide the answer inside a bucket that looked like a different question.

    Overflow now carries a digest of the FULL text, so distinct reasons keep
    distinct keys and the truncation announces itself in the label rather than
    having to be inferred from a suspiciously round length.
    """

    if len(text) <= limit:
        return text
    digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:12]
    suffix = f"~trunc:{digest}"
    # A limit too small to hold the suffix keeps the digest and drops the head:
    # the identity of the reason is worth more than a few leading characters,
    # and returning something LONGER than the requested bound would defeat the
    # only job this function has. `max(..., 0)` alone did not guard this.
    if limit <= len(suffix):
        return suffix[:limit]
    return f"{text[: limit - len(suffix)]}{suffix}"


# The closed vocabulary for `EngineSearchStats.choices_unmapped_causes`.
#
# Closed and greppable on purpose, the same discipline `UNRENDERABLE_FAMILY_ORDER` applies in
# the Rust renderer: a diagnosis counter whose keys are open-ended cannot be aggregated across
# an era, and an unregistered token is how a class silently stops being rankable.
# Named constants, so a typo is a NameError rather than a silently unregistered key. Review
# mutated the bare `"no_action_candidates"` literal at the call site to
# `"no_action_candidate"` and all 293 tests passed -- the vocabulary was enforced only by
# tests that never touched the production path.
_CAUSE_NO_ACTION_CANDIDATES = "no_action_candidates"
_CAUSE_AGGREGATED_EMPTY = "aggregated_empty"
_CAUSE_SWITCH_ONLY = "all_unmapped_switch_only"
_CAUSE_LEGALITY_MISMATCH = "all_unmapped_legality_mismatch"

#: Normalized ids for the crate's display of `MoveChoice::None` -- the engine's forced no-move.
#: Showdown names the same forced action `recharge` in the request it sends for that turn.
_ENGINE_FORCED_NO_MOVE_IDS = frozenset({"nomove", "none"})
_CAUSE_NO_LEGAL_ACTION = "no_legal_action_offered"
_CAUSE_NO_POSITIVE_WEIGHT = "mapped_but_no_positive_weight"
_CAUSE_UNCLASSIFIED = "unclassified_cause"

_CHOICES_UNMAPPED_CAUSES = (
    # The request carried no `action_candidates` at all. Not a legality mismatch -- the
    # observation itself is missing the field, so this is a plumbing failure, not a
    # belief/engine disagreement.
    _CAUSE_NO_ACTION_CANDIDATES,
    # The search produced no choices to map. Distinct from every choice failing: there was
    # nothing to fail. Reachable when every world aborted but the caller did not take the
    # `crate_search_failed` exit first.
    _CAUSE_AGGREGATED_EMPTY,
    # Every proposed choice failed to map AND no move was legal in the request. This is a
    # SWITCH-ONLY decision -- a force switch, or an active mon with no usable move -- and the
    # engine proposed only moves. The fix belongs on the policy side: propose a switch.
    _CAUSE_SWITCH_ONLY,
    # Every proposed choice failed to map WHILE some move WAS legal. The engine's world and
    # the request disagree about WHICH move is legal: PP exhaustion, Taunt, Disable, or a
    # choice/Encore lock. A belief or PP-derivation bug, not a policy one.
    _CAUSE_LEGALITY_MISMATCH,
    # Choices mapped, but every weight was non-positive, so `weight > best_weight` never
    # fired against the 0.0 seed. A zero-visit search rather than a mapping failure.
    #
    # PRECEDENCE, NOT EXCLUSIVITY. An earlier comment claimed this is "invisible in
    # `unmapped_choices`, since nothing was unmapped". False: with
    # `{"surf": 1.0, "earthquake": 0.0}` where only `earthquake` maps, this token fires AND
    # `unmapped_choices` records `surf`. The token names the PROXIMATE blocker -- the weight
    # comparison -- so a tail where most choices failed to map and one mapped at 0.0 is filed
    # here rather than as a legality mismatch. Defensible, but it means the two counters
    # overlap, and any inference of the form "unmapped_choices is non-empty, therefore this
    # cause is ruled out" is invalid.
    _CAUSE_NO_POSITIVE_WEIGHT,
    # NO legal action of any kind -- neither a move nor a switch. Split out of
    # `all_unmapped_switch_only`, which was reporting "policy: propose a switch" for an empty
    # candidate list, all-illegal candidates, an out-of-range mask, and an all-False mask.
    # None of those is a game state; all are plumbing or mask bugs.
    _CAUSE_NO_LEGAL_ACTION,
    # LAST, and emitted by no branch of `_classify_unmapped` -- reachable only through
    # `_registered_cause_or_unclassified`. Registered anyway, because the first version left
    # it OUT and that made the function whose job is keeping this vocabulary closed the one
    # thing able to emit a key outside it: an era aggregator iterating this tuple would drop
    # the bucket, silently losing the very class the degradation exists to keep measurable.
    # The Rust precedent this mirrors does the same and reasons about it explicitly --
    # `UNRENDERABLE_FAMILY_ORDER` has 14 entries where 13 arms can emit, and its comment
    # spells out that the 14th is reachable only through the degradation.
    _CAUSE_UNCLASSIFIED,
)


def _registered_cause_or_unclassified(cause: str) -> str:
    """Degrade an unregistered cause instead of trusting the caller.

    The Rust renderer's `registered_family_or_unclassified` does exactly this on the
    PRODUCTION path, and its comment explains why: a token nobody registered cannot be
    aggregated across an era, and an `assert` here would be worse than a bad key, because
    pyo3 maps a panic to `PanicException` and kills the campaign worker.

    The first version of this counter had no runtime reader of `_CHOICES_UNMAPPED_CAUSES` at
    all -- it was enforced only by tests that never executed `_map_choices`, so review
    mutated a bare string literal at the call site and 293 tests passed. Named constants
    make that a `NameError`; this makes a dynamically-built token measurable rather than
    silent.
    """

    return cause if cause in _CHOICES_UNMAPPED_CAUSES else _CAUSE_UNCLASSIFIED


def _classify_unmapped(
    *,
    aggregated: Mapping[str, float],
    mapped_any: bool,
    any_legal_move: bool,
    any_legal_switch: bool,
) -> str:
    """Name WHY `_map_choices` is about to return None.

    Keyword-only, so a future argument cannot be silently absorbed into the wrong slot --
    every parameter here is a bool or a mapping and three of them are bools, which is
    precisely the shape where positional calls go wrong unnoticed.
    """

    if not aggregated:
        return _CAUSE_AGGREGATED_EMPTY
    if mapped_any:
        # Something mapped, so the failure is the weight comparison, not the mapping.
        return _CAUSE_NO_POSITIVE_WEIGHT
    if any_legal_move:
        return _CAUSE_LEGALITY_MISMATCH
    if not any_legal_switch:
        # NO legal action of any kind. The request offered nothing -- not a switch-only
        # decision, and emphatically not a policy bug. An earlier version of this function
        # folded this into `switch_only` while its own comment admitted it was "a different
        # bug entirely", which is precisely the collapse this counter exists to prevent: an
        # operator reading `switch_only` goes and reads the policy, when the truth may be a
        # mask off-by-one or an empty candidate list. The separating bool was already
        # computed, passed in, and then deleted.
        return _CAUSE_NO_LEGAL_ACTION
    return _CAUSE_SWITCH_ONLY


@dataclass
class EngineMctsStats:
    """Cumulative per-policy telemetry; every fallback is counted, never hidden."""

    decisions: int = 0
    searched_decisions: int = 0
    fallback_decisions: int = 0
    # Decisions where the removal signal fired (a mon's item is publicly
    # stripped or consumed): worlds constructed with that item cleared instead
    # of failing closed. Localizes which battles exercise the removal path.
    # TELEMETRY ONLY (PR #741 review note): a decision where several mons carry
    # the signal — or where both this and item_override_decisions fire — bumps
    # each counter once per decision, so the counters can co-occur/over-count
    # relative to distinct battles; nothing gates on them.
    removed_item_decisions: int = 0
    # Decisions where the Trick-swap current-item override fired (a mon
    # publicly holds a protocol-confirmed item that is not the sampled
    # assignment): worlds constructed with the revealed CURRENT item
    # substituted instead of failing closed. Telemetry only, as above.
    item_override_decisions: int = 0
    worlds_attempted: int = 0
    # Worlds that survived CONSTRUCTION and were handed to the search. The
    # difference between this and `worlds_attempted` is belief-sampling and
    # world-building failure; the difference between this and `worlds_searched`
    # is the search ABORTING on an attribution-unsafe branch. Without this
    # counter those two very different defects are only separable by parsing the
    # `world_failure_reasons` taxonomy, which is why the second one has been
    # invisible. See `world_search_abort_rate` in `to_dict`.
    worlds_constructed: int = 0
    worlds_searched: int = 0
    total_iterations: int = 0
    search_wall_seconds: float = 0.0
    decision_wall_seconds: float = 0.0
    world_failure_reasons: Counter = field(default_factory=Counter)
    fallback_reasons: Counter = field(default_factory=Counter)
    # ADDRESSES, not just counts. A fallback is fully identified by
    # (battle_id, round, seat) -- the battle id carries the seed, so any entry
    # here replays as a single turn. Until this existed the addresses lived only
    # in pod logs, and deleting a Job deleted them: the era-57 probe left aggregate
    # counts and zero way to reproduce any of the 7,498 fallbacks it recorded, so no
    # refusal class could be debugged from a campaign run.
    #
    # Keyed BY CLASS and capped per class rather than globally. A global cap fills
    # with the dominant class -- era 57 was 49.5% one reason -- and the classes you
    # most need an address for are the rare ones. Keyed by fallback REASON too, for
    # the same argument on the other axis (see _fallback). Bounded by
    # _FALLBACK_SAMPLES_PER_CLASS x distinct keys.
    fallback_samples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    # ADDRESSES discarded because the class ceiling was hit -- one per dropped
    # occurrence, NOT one per lost class. Named for the unit it measures: it was
    # `..._keys_dropped`, and it read 1,001 where 2 classes had been lost, which is the
    # cite-a-number-for-a-different-quantity mistake this campaign keeps making.
    # Non-zero means the sample is INCOMPLETE across classes.
    fallback_sample_addresses_dropped: int = 0
    unmapped_choices: Counter = field(default_factory=Counter)
    # WHY `_map_choices` returned None, which `unmapped_choices` cannot say.
    #
    # `choices_unmapped` was 29 fallback decisions on era 60 and is a stop-condition term
    # in its own right -- GOAL.md requires it at zero independently of the fallback rate --
    # but the reason is a single opaque literal. So era 60 could say the class was 29 and not
    # one word about which cause produced it.
    #
    # FOUR call sites, not three, and the fourth is NOT a decision: the early-stop path calls
    # `_map_choices` as a PROBE to validate a locked choice, and on None it clears the lock
    # and proceeds -- the decision may then succeed. So this counter is NOT a partition of
    # `fallback_reasons["choices_unmapped"]`; a probe miss increments it with no fallback, and
    # a decision whose probe AND real call both miss contributes two. Latent today, since
    # early stop defaults off and no shipped config enables it, but the arithmetic matters to
    # anyone who tries to reconcile the two counters.
    #
    # This is the same shape of gap that `ambiguous_unrenderable` had before its family
    # split: one key at meaningful volume, no way to scope the fix. That split is what let
    # era 60 rank the abort channel at all, and it is the reason this counter exists.
    #
    # Deliberately NOT a sub-cased reason. `fallback_reasons` is a closed 7-literal set and
    # the per-decision address store bounds itself on that ("The reason set is closed and
    # small (7 literals), so this adds at most 7 keys"). Sub-casing the reason would
    # multiply the address keys and silently weaken that bound. A separate counter carries
    # the diagnosis with no effect on the reason vocabulary.
    #
    # The token set is closed and greppable; see `_CHOICES_UNMAPPED_CAUSES`.
    choices_unmapped_causes: Counter = field(default_factory=Counter)
    # Model-mode telemetry (zero on the hp_fraction path).
    model_evals: int = 0
    # Native per-phase search wall (crate-measured, never derived by
    # subtraction): leaf encoding, model forwards, and tree work. These are the
    # inputs to the depth/throughput study's phase attribution
    # (docs/mcts_depth_strength_eval_plan.md section 4).
    encode_wall_seconds: float = 0.0
    model_wall_seconds: float = 0.0
    tree_wall_seconds: float = 0.0
    # Sub-slice of encode_wall_seconds: per-leaf FoldStateInner deep clones.
    fold_clone_wall_seconds: float = 0.0
    # Encode decomposition: instruction->protocol-text rendering, the fold's
    # re-parse of that text, the observation tensor build, and action mapping.
    render_wall_seconds: float = 0.0
    fold_advance_wall_seconds: float = 0.0
    tensor_wall_seconds: float = 0.0
    action_map_wall_seconds: float = 0.0
    # tensor_s split: engine-state row inputs, the fold's derived products
    # (rebuilt per leaf), and the array write.
    row_input_wall_seconds: float = 0.0
    products_wall_seconds: float = 0.0
    row_write_wall_seconds: float = 0.0
    lossy_renders: int = 0
    # Sub-cases that were COUNTED rather than refused, keyed like world_failure_reasons.
    # A class that stops refusing must not stop being visible: `sleeptalk_...:ambiguous`
    # used to appear in `world_failure_reasons` precisely BECAUSE it aborted the world, so
    # making it usable would have deleted the only number that tracked it. Two eras were
    # spent unable to say what had changed in a class; that is the cost this avoids.
    lossy_subcase_renders: Counter = field(default_factory=Counter)
    attribution_unsafe_renders: int = 0
    prior_fallbacks: int = 0
    early_stop_triggered_worlds: int = 0
    early_stop_accepted_decisions: int = 0
    early_stop_full_budget_replays: int = 0
    early_stop_sims_saved: int = 0
    fold_advanced_lines: int = 0
    fold_cross_checks: int = 0
    fold_cross_check_failures: int = 0
    # Tier-2 overlay telemetry (zero without an annotation source).
    fold_annotations_applied: int = 0
    fold_annotation_boundaries: int = 0
    # Depth-reached instrumentation (hp_fraction_crate mode). The crate counts
    # the deepest decision node any traversal actually opened; the depth CAP is
    # only a real knob where these numbers sit at it. Recorded per SEARCHED
    # WORLD, so depth_reached_samples counts worlds, not decisions.
    depth_reached_samples: int = 0
    depth_reached_sum: int = 0
    depth_reached_max: int = 0
    depth_reached_histogram: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "decisions": self.decisions,
            "searched_decisions": self.searched_decisions,
            "fallback_decisions": self.fallback_decisions,
            "fallback_rate": self.fallback_decisions / self.decisions if self.decisions else 0.0,
            "removed_item_decisions": self.removed_item_decisions,
            "item_override_decisions": self.item_override_decisions,
            "worlds_attempted": self.worlds_attempted,
            "worlds_constructed": self.worlds_constructed,
            "worlds_searched": self.worlds_searched,
            # THE PER-WORLD ABORT RATE. `fallback_rate` is a LOWER BOUND on it.
            #
            # A decision falls back only when EVERY world fails (`_search_model`
            # returns `crate_search_failed` on `worlds_searched_here == 0`), so a
            # decision that searched 1 of 4 constructed worlds is not a fallback
            # and is invisible in every other metric -- while its belief
            # aggregate rests on a quarter of the sampled hypotheses. That is a
            # search-quality loss, not just a reporting gap.
            #
            # How much bigger the per-world rate is than `fallback_rate` depends
            # on how CORRELATED aborts are across the worlds of one decision, and
            # the honest answer is that we do not know yet -- measuring it is why
            # this counter exists. The bound:
            #
            #   * Linear (`fallback ~= abort`) when the refusal trigger is shared
            #     PUBLIC state. Attract and confusion are public volatiles, and
            #     our own sleeping mon's Sleep Talk moveset is fully known, so
            #     every world of that decision refuses together. Worlds also
            #     share blocked_slots/encored_moves/removed_item_species and the
            #     same self-side legal moves.
            #   * `fallback ~= abort^W` only if aborts were iid across worlds --
            #     which needs the trigger to be a belief-sampled OPPONENT
            #     property, since that is the only thing that differs per world.
            #
            # Even in the belief-sampled arm the exponent is AT MOST W and
            # collapses toward 1 as the posterior concentrates: worlds are
            # exchangeable draws from ONE decision-level posterior, gen3 randbat
            # pools are small, and a species whose pool reliably contains two
            # identical-tail moves refuses in every world. Duplicate draws (cf.
            # `scott/collapse-identical-worlds`) cut the effective W further.
            #
            # Do NOT reason from the per-world seed: `tree.rs` uses rng only in
            # `sample_branch_index`, for traversal. `expand_edge` prices EVERY
            # enumerated branch of a joint action, and `select()` is pure PUCT
            # with no rng -- so a refusal fires when the search expands an unsafe
            # joint action, not when it draws an unlucky outcome. Distinct seeds
            # do not buy independence.
            #
            # Caveat on the numerator: `worlds_constructed` is charged before
            # `_search_model`'s own pre-flight, so a `root_inputs_failed` or
            # `early_stop_replay_failed` fallback charges W phantom aborts here.
            # Both are visible in `fallback_reasons`; check there before reading
            # a near-1.0 abort rate as a renderer-refusal problem.
            "world_search_abort_rate": (
                1.0 - (self.worlds_searched / self.worlds_constructed)
                if self.worlds_constructed
                else None
            ),
            # NOT a per-world loss rate -- named for what it is. The denominator
            # counts construction ATTEMPTS including retries (`worlds` x
            # `sample_retry_factor`, 16 by default), so a decision that retried
            # its way to a full world set and one that gave up half of them can
            # read the same. Use `worlds_constructed` against `decisions x
            # config.worlds` if you want the per-world version.
            "belief_sample_rejection_rate": (
                1.0 - (self.worlds_constructed / self.worlds_attempted)
                if self.worlds_attempted
                else None
            ),
            "total_iterations": self.total_iterations,
            "search_wall_seconds": self.search_wall_seconds,
            "decision_wall_seconds": self.decision_wall_seconds,
            "world_failure_reasons": dict(self.world_failure_reasons),
            "fallback_reasons": dict(self.fallback_reasons),
            "fallback_samples": {k: list(v) for k, v in self.fallback_samples.items()},
            "fallback_sample_addresses_dropped": (
                self.fallback_sample_addresses_dropped
            ),
            "unmapped_choices": dict(self.unmapped_choices),
            "choices_unmapped_causes": dict(self.choices_unmapped_causes),
            "model_evals": self.model_evals,
            "encode_wall_seconds": self.encode_wall_seconds,
            "model_wall_seconds": self.model_wall_seconds,
            "tree_wall_seconds": self.tree_wall_seconds,
            "fold_clone_wall_seconds": self.fold_clone_wall_seconds,
            "render_wall_seconds": self.render_wall_seconds,
            "fold_advance_wall_seconds": self.fold_advance_wall_seconds,
            "tensor_wall_seconds": self.tensor_wall_seconds,
            "action_map_wall_seconds": self.action_map_wall_seconds,
            "row_input_wall_seconds": self.row_input_wall_seconds,
            "products_wall_seconds": self.products_wall_seconds,
            "row_write_wall_seconds": self.row_write_wall_seconds,
            "lossy_renders": self.lossy_renders,
            "lossy_subcase_renders": dict(self.lossy_subcase_renders),
            "attribution_unsafe_renders": self.attribution_unsafe_renders,
            "prior_fallbacks": self.prior_fallbacks,
            "early_stop_triggered_worlds": self.early_stop_triggered_worlds,
            "early_stop_accepted_decisions": self.early_stop_accepted_decisions,
            "early_stop_full_budget_replays": self.early_stop_full_budget_replays,
            "early_stop_sims_saved": self.early_stop_sims_saved,
            "fold_advanced_lines": self.fold_advanced_lines,
            "fold_cross_checks": self.fold_cross_checks,
            "fold_cross_check_failures": self.fold_cross_check_failures,
            "fold_annotations_applied": self.fold_annotations_applied,
            "fold_annotation_boundaries": self.fold_annotation_boundaries,
            "depth_reached_samples": self.depth_reached_samples,
            "depth_reached_max": self.depth_reached_max,
            "depth_reached_histogram": {
                str(depth): count
                for depth, count in sorted(self.depth_reached_histogram.items())
            },
        }
        if self.depth_reached_samples:
            payload["depth_reached_mean"] = (
                self.depth_reached_sum / self.depth_reached_samples
            )
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


def opponent_request_order(context, party_species) -> list[str] | None:
    """The opponent's Showdown request order at this decision, or None.

    Showdown keeps a player's active at request slot 0 and swaps the incoming
    mon into slot 0 on every switch-in (`sim/battle-actions.ts` `switchIn`, an
    unconditional slot swap that also fires for forced replacements and for
    `dragIn`). The resulting order is the label space of the model's opponent
    action head, so the crate needs it to gather opponent priors onto the right
    arms -- and the crate cannot derive it, having no pre-root protocol lines.

    This REUSES `determinization._public_opponent_team_index_walk`, which
    already maintains exactly this permutation as `current_order` while
    decoding recorded opponent switch actions. Reuse is the point: five
    hand-rolled reconstructions were each wrong, and the fifth -- diffing the
    opponent's public active across OUR decision rounds -- was wrong on 170 of
    811 decisions across 12 live games because the opponent also acts at rounds
    we are not requested at (a forced replacement after a faint), which that
    diff cannot see. The walk consumes the opponent's trajectory steps directly
    and reconciles them against the next observed active, so it sees those
    rounds; it is also the code that already handles Roar/Whirlwind drags and
    same-chunk faint replacements.

    Fails closed rather than guessing: the walk returns None when the public
    data is inconsistent, and sets `active_position` to None when it loses
    track of the permutation. Either means no order. A wrong order silently
    permutes opponent switch priors, which is strictly worse than the crate's
    documented one-swap fallback.
    """
    from .determinization import _public_opponent_team_index_walk

    party = [normalize_id(str(name)) for name in party_species]
    if not party:
        return None
    if len(set(party)) != len(party):
        # Slot swaps are resolved by species name downstream, so a duplicated
        # species makes the mapping ambiguous.
        return None
    opponent_slot = "p2" if getattr(context, "player_id", "p1") == "p1" else "p1"
    try:
        walk = _public_opponent_team_index_walk(
            context, opponent_slot=opponent_slot, team_size=len(party)
        )
    except Exception:  # noqa: BLE001 - never break search over telemetry
        return None
    if walk is None:
        return None
    _constraints, current_order, active_position = walk
    if active_position is None:
        # The walk stopped trusting its own permutation.
        return None
    if sorted(current_order) != list(range(len(party))):
        return None
    return [party[index] for index in current_order]


class EngineMctsPolicy:
    """ContextAwarePolicy running poke-engine MCTS over belief-sampled worlds."""

    # Declares the requirement the FoulPlay bridge gates on. Engine search
    # cannot run without a materialized public state; without this the bridge
    # passes None and every decision degrades to uniform-legal fallback.
    requires_public_materialization_state = True

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
        self._model_config: Any | None = None
        self._native_model: Any | None = None
        if self._config.leaf_eval == "model":
            from pathlib import Path  # noqa: PLC0415 — model-mode-only dependency

            from .neural_policy import load_transformer_model_config  # noqa: PLC0415

            model_path = Path(str(self._config.model_path))
            if not model_path.exists():
                raise ValueError(f"model artifact not found: {model_path}")
            checkpoint_path = Path(str(self._config.checkpoint_path))
            if not checkpoint_path.exists():
                raise ValueError(f"source checkpoint not found: {checkpoint_path}")
            tables_path = Path(str(self._config.tables_path))
            if not tables_path.exists():
                raise ValueError(f"encoder tables not found: {tables_path}")
            self._model_config = load_transformer_model_config(checkpoint_path)
            self._tables_json = _latch_encoder_tables_to_model_config(
                tables_path.read_text(encoding="utf-8"), self._model_config
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
        blocked_slots, encored_moves, removed_item_species, current_item_overrides, transformed_slots = (
            self._public_effect_signals(context)
        )
        if removed_item_species:
            self.stats.removed_item_decisions += 1
        if current_item_overrides:
            self.stats.item_override_decisions += 1
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
                    approximate_partial_trap_turns=self._config.approximate_partial_trap_turns,
                    approximate_hidden_duration_volatiles=self._config.approximate_hidden_duration_volatiles,
                    blocked_slots=blocked_slots,
                    encored_moves=encored_moves,
                    removed_item_species=removed_item_species,
                    current_item_overrides=current_item_overrides,
                    recharging_slots=recharging_slots,
                    truant_slots=truant_slots,
                    transformed_slots=transformed_slots,
                    rng=rng,
                )
                state = build_poke_engine_state(world.spec, module=self._module)
            except PokeEngineAttractUnsupportedError:
                # Upstream accepts ATTRACT but ignores its 50% Gen 3
                # immobilization. The adapter proves the local patch before it
                # permits a world; classify a missing patch as an attributed
                # fallback instead of silently searching an optimistic state.
                self.stats.world_failure_reasons["attract_patch_unavailable"] += 1
                continue
            except EngineWorldUnsupported as error:
                self.stats.world_failure_reasons[_world_failure_key(error)] += 1
                continue
            worlds.append((world, state))

        if not worlds:
            return self._fallback(context, rng, "no_worlds_constructed")

        # Counted HERE, at the single dispatch point, rather than at the append
        # above: this is exactly the list every search path is about to receive,
        # so the denominator cannot drift from what `worlds_searched` counts no
        # matter which leaf_eval runs.
        self.stats.worlds_constructed += len(worlds)

        if self._config.leaf_eval == "model":
            return self._search_model(context, worlds, live_fold, rng)
        if self._config.leaf_eval == "hp_fraction_crate":
            return self._search_hp_fraction_crate(context, worlds, rng)

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
                    # Always equal to `worlds_searched` here -- this legacy path
                    # has no per-world abort, so its abort rate is structurally
                    # zero. Emitted anyway so every leaf_eval carries the same
                    # denominator and a consumer never has to special-case which
                    # path a shard came from.
                    "worlds_constructed": len(worlds),
                    "aggregated_choices": {
                        choice: round(weight, 4) for choice, weight in aggregated.most_common()
                    },
                }
            },
        )

    def _search_hp_fraction_crate(
        self,
        context: PolicyContext,
        worlds: list[tuple[EngineWorld, Any]],
        rng: random.Random,
    ) -> PolicyDecision:
        """Crate PUCT tree, handcrafted HP-fraction leaves, no learned model.

        The controlled twin of ``_search_model``: same belief worlds, same
        per-world visit-share aggregation, same choice mapping, same fallback
        taxonomy. The ONLY differences are inside the tree —

        - leaves are priced by ``HpFractionEval`` instead of a TorchScript
          forward, so no observation encode, no fold chaining, no priors;
        - the acting side's root priors stay uniform (``model_priors`` has no
          meaning without a model);
        - leaf pricing is inline (``LeafPrice::Ready``), so the driver is the
          sequential one. ``search_batch`` is therefore inert here — it is the
          batched driver's virtual-loss width, and the sequential driver is the
          b=1 limit of it (crate test ``b=1 ≡ sequential``).

        ``max_depth``/``search_sims``/``c_puct``/``deep_ko_split`` carry exactly
        the meanings they carry in model mode: the identical ``MultiPlyConfig``
        reaches the identical ``traverse``/``finalize``.
        """

        import pokezero_search  # noqa: PLC0415 — optional native dependency

        config = self._config
        aggregated: Counter[str] = Counter()
        depths_reached: list[int] = []
        worlds_searched_here = 0
        search_started = time.perf_counter()
        for world, state in worlds:
            side_key = (
                "side_one"
                if world.slot_sides[context.player_id] == "side_one"
                else "side_two"
            )
            try:
                report = json.loads(
                    pokezero_search.puct_search_multi(
                        state.to_string(),
                        config.search_sims,
                        max_depth=config.search_depth,
                        c_puct=config.c_puct,
                        seed=rng.getrandbits(63),
                        deep_ko_split=config.deep_ko_split,
                    )
                )
            except Exception as error:  # noqa: BLE001 — count, keep the other worlds
                detail = (
                    _bounded_reason_detail(str(error).splitlines()[0])
                    if str(error)
                    else type(error).__name__
                )
                self.stats.world_failure_reasons[f"crate_search_hp: {detail}"] += 1
                continue
            entries = report[side_key]
            total = max(sum(entry["visits"] for entry in entries), 1)
            for entry in entries:
                aggregated[entry["move"]] += entry["visits"] / total
            reached = int(report["max_depth_reached"])
            depths_reached.append(reached)
            self.stats.depth_reached_samples += 1
            self.stats.depth_reached_sum += reached
            self.stats.depth_reached_max = max(self.stats.depth_reached_max, reached)
            self.stats.depth_reached_histogram[reached] += 1
            self.stats.total_iterations += int(report["iterations"])
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
                    "leaf_eval": "hp_fraction_crate",
                    "worlds_searched": worlds_searched_here,
                    # This path aborts worlds too (`crate_search_hp:`), so it
                    # needs the same per-decision denominator as the model path.
                    "worlds_constructed": len(worlds),
                    "max_depth_reached": max(depths_reached),
                    "depths_reached": tuple(depths_reached),
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

    def _validate_model_root_observation(self, observation: Any) -> None:
        """Fail closed if the live root is outside the checkpoint's trained contract."""

        from .showdown import TRANSITION_TOKEN_OFFSET  # noqa: PLC0415

        config = self._model_config
        if config is None:
            raise EngineSearchFallbackError("model observation contract is not loaded.")
        schema = observation.schema_version
        if schema != config.observation_schema_version:
            raise EngineSearchFallbackError(
                f"leaf_eval='model' checkpoint requires {config.observation_schema_version!r} "
                f"observations; this env produced {schema!r}."
            )
        token_count = len(observation.attention_mask)
        categorical_width = len(observation.categorical_ids[0]) if token_count else 0
        numeric_width = len(observation.numeric_features[0]) if token_count else 0
        actual_shape = (token_count, categorical_width, numeric_width)
        expected_shape = (
            int(config.token_count),
            int(config.categorical_feature_count),
            int(config.numeric_feature_count),
        )
        if actual_shape != expected_shape:
            raise EngineSearchFallbackError(
                f"model root observation shape {actual_shape!r} does not match checkpoint "
                f"shape {expected_shape!r}."
            )
        attended_history = sum(
            bool(value) for value in observation.attention_mask[TRANSITION_TOKEN_OFFSET:]
        )
        if attended_history > config.transition_token_budget:
            raise EngineSearchFallbackError(
                f"model root observation attends {attended_history} history tokens, exceeding "
                f"checkpoint budget {config.transition_token_budget}."
            )

    def _absorb_lossy_subcases(self, report: Mapping[str, Any]) -> None:
        """Count renders that were kept-but-lossy, per sub-case.

        Extracted so the seam is testable. `_search_model` needs a live native module and
        a real state to reach, so a counter absorbed inline there is pinned by nothing --
        and this particular counter exists BECAUSE an unpinned class went invisible for
        two eras. The key name must match what the crate emits (`model.rs`, `lossy_subcases`
        in the search report); a rename on either side silently zeroes the class, which is
        the failure mode, so both sides are asserted together by the test.
        """

        for subcase, count in (report.get("lossy_subcases") or {}).items():
            self.stats.lossy_subcase_renders[str(subcase)] += int(count)

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

        self._validate_model_root_observation(context.observation)
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

        world_runs: list[dict[str, Any]] = []
        search_started = time.perf_counter()

        def run_world(
            record: Mapping[str, Any], early_stop_min_sims: int
        ) -> Optional[dict]:
            try:
                search_args = [
                    record["state_str"],
                    config.search_sims,
                    config.search_batch,
                    self._tables_json,
                    root_inputs,
                    record["ctx_json"],
                    rust_fold,
                    config.search_depth,
                    config.c_puct,
                    record["seed"],
                    config.deep_ko_split,
                    config.model_priors,
                ]
                if early_stop_min_sims or config.use_opponent_priors:
                    # Preserve the old native call contract while the feature
                    # is disabled, so a stale image cannot break default
                    # full-budget search merely because Python was updated.
                    search_args.extend(
                        [early_stop_min_sims, record["side_key"] == "side_one"]
                    )
                if config.use_opponent_priors:
                    # Positional, and it follows the early-stop pair in the
                    # native signature -- hence the combined guard above: the
                    # pair must be present for this to land in the right slot.
                    # Appended ONLY when set, so a flag-off run makes exactly
                    # the call it always did.
                    search_args.append(True)
                report = json.loads(
                    native.search_batched_multi_encoded(*search_args)
                )
            except Exception as error:  # noqa: BLE001 — count, keep the other worlds
                detail = (
                    _bounded_reason_detail(str(error).splitlines()[0])
                    if str(error)
                    else type(error).__name__
                )
                reason = (
                    f"native_early_stop_unsupported: {detail}"
                    if early_stop_min_sims and isinstance(error, TypeError)
                    else detail
                )
                # Unsafe renderer branches abort the native world before a
                # chance outcome can be silently omitted from its expectation.
                # The native report is unavailable on that error path, so
                # retain the same observability counter at the fallback seam.
                if "attribution-unsafe renderer branch rejected before" in reason:
                    self.stats.attribution_unsafe_renders += 1
                self.stats.world_failure_reasons[f"crate_search: {reason}"] += 1
                return None
            # Invocation-level counters reflect actual compute. A stopped
            # world that is conservatively replayed at full budget counts both
            # invocations; worlds_searched is updated only for final records.
            self.stats.total_iterations += int(report["iterations"])
            self.stats.model_evals += int(report["model_evals"])
            # Reached depth, same accumulation the hp_fraction path already does.
            # Without this the model path -- the one every strength campaign runs
            # -- reports nothing about whether the depth CAP was ever binding, so
            # a flat depth ladder cannot be distinguished from a ladder whose
            # rungs all built the same undersized tree. Per WORLD, like the
            # hp_fraction path: depth_reached_samples counts worlds, not decisions.
            reached = report.get("max_depth_reached")
            if reached is not None:
                reached = int(reached)
                self.stats.depth_reached_samples += 1
                self.stats.depth_reached_sum += reached
                self.stats.depth_reached_max = max(self.stats.depth_reached_max, reached)
                self.stats.depth_reached_histogram[reached] += 1
            # Crate-measured phase walls are per-INVOCATION compute, exactly like
            # iterations/model_evals above: a conservatively replayed world spent
            # that encode/model/tree time twice and must report it.
            self.stats.encode_wall_seconds += float(report.get("encode_s") or 0.0)
            self.stats.model_wall_seconds += float(report.get("model_s") or 0.0)
            self.stats.tree_wall_seconds += float(report.get("tree_s") or 0.0)
            self.stats.fold_clone_wall_seconds += float(report.get("fold_clone_s") or 0.0)
            self.stats.render_wall_seconds += float(report.get("render_s") or 0.0)
            self.stats.fold_advance_wall_seconds += float(report.get("fold_advance_s") or 0.0)
            self.stats.tensor_wall_seconds += float(report.get("tensor_s") or 0.0)
            self.stats.action_map_wall_seconds += float(report.get("action_map_s") or 0.0)
            self.stats.row_input_wall_seconds += float(report.get("row_input_s") or 0.0)
            self.stats.products_wall_seconds += float(report.get("products_s") or 0.0)
            self.stats.row_write_wall_seconds += float(report.get("row_write_s") or 0.0)
            self.stats.lossy_renders += int(report.get("lossy_renders") or 0)
            self._absorb_lossy_subcases(report)
            self.stats.attribution_unsafe_renders += int(
                report.get("attribution_unsafe_renders") or 0
            )
            self.stats.prior_fallbacks += int(report.get("prior_fallbacks") or 0)
            return report

        for world, state in worlds:
            ctx_json = json.dumps(
                {
                    "p1": list(world.party_species["p1"]),
                    "p2": list(world.party_species["p2"]),
                    "turn": turn,
                    # Construction-only provenance for a root Toxic zero.
                    # The leaf context consumes this outside model metadata.
                    "toxic_stage_zero_after_upkeep": _root_toxic_zero_after_upkeep_attestation(
                        replay
                    ),
                    **(
                        {"opponent_request_order": opponent_order}
                        if (opponent_order := opponent_request_order(
                            context,
                            world.party_species[
                                "p2" if context.player_id == "p1" else "p1"
                            ],
                        ))
                        else {}
                    ),
                }
            )
            world_seed = rng.getrandbits(63)
            side_key = (
                "side_one"
                if world.slot_sides[context.player_id] == "side_one"
                else "side_two"
            )
            record: dict[str, Any] = {
                "state_str": state.to_string(),
                "ctx_json": ctx_json,
                "seed": world_seed,
                "side_key": side_key,
            }
            stop_floor = config.early_stop_min_sims if config.early_stop else 0
            report = run_world(record, stop_floor)
            if report is not None:
                record["report"] = report
                world_runs.append(record)

        stopped_runs = [
            record for record in world_runs if bool(record["report"].get("early_stopped"))
        ]
        self.stats.early_stop_triggered_worlds += len(stopped_runs)
        locked_choice: Optional[str] = None
        full_budget_replays = 0
        simulations_saved = 0
        replay_failed = False
        if stopped_runs:
            locked_choice = _locked_aggregate_choice(
                [(record["side_key"], record["report"]) for record in world_runs]
            )
            if locked_choice is not None and self._map_choices(
                context, Counter({locked_choice: 1.0})
            ) is None:
                locked_choice = None
            if locked_choice is not None:
                simulations_saved = sum(
                    int(record["report"].get("remaining_iterations") or 0)
                    for record in stopped_runs
                )
                self.stats.early_stop_accepted_decisions += 1
                self.stats.early_stop_sims_saved += simulations_saved
            else:
                # Per-world STOP does not by itself preserve the normalized
                # aggregate across belief worlds. Replay only stopped worlds
                # at full budget; this is intentionally fail-open to the
                # pre-feature behavior in ambiguous cases.
                final_runs: list[dict[str, Any]] = []
                for record in world_runs:
                    if not record["report"].get("early_stopped"):
                        final_runs.append(record)
                        continue
                    full_budget_replays += 1
                    report = run_world(record, 0)
                    if report is None:
                        replay_failed = True
                        break
                    record["report"] = report
                    final_runs.append(record)
                world_runs = final_runs
                self.stats.early_stop_full_budget_replays += full_budget_replays
        self.stats.search_wall_seconds += time.perf_counter() - search_started

        if replay_failed:
            return self._fallback(context, rng, "early_stop_replay_failed")
        worlds_searched_here = len(world_runs)
        if not worlds_searched_here:
            return self._fallback(context, rng, "crate_search_failed")
        self.stats.worlds_searched += worlds_searched_here
        aggregated: Counter[str] = Counter()
        for record in world_runs:
            entries = record["report"][record["side_key"]]
            total = max(sum(entry["visits"] for entry in entries), 1)
            for entry in entries:
                aggregated[entry["move"]] += entry["visits"] / total
        # Both counters are accumulated ONCE, above the loop. This block used to also
        # bump ``stats.worlds_searched`` and ``worlds_searched_here`` per record and
        # re-add the wall interval after the loop, so model-mode telemetry reported
        # every world twice and roughly twice the search time (the same interval, both
        # times measured from ``search_started``). The hp_fraction paths never had it.
        # Scores were never affected — nothing here feeds ``aggregated``.

        if not worlds_searched_here:
            return self._fallback(context, rng, "crate_search_failed")
        choice_weights = (
            Counter({locked_choice: 1.0}) if locked_choice is not None else aggregated
        )
        action_index = self._map_choices(context, choice_weights)
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
                    # Per-decision denominator. A decision that searched 1 of 4
                    # constructed worlds is NOT a fallback and so is invisible in
                    # `fallback_rate`, but its belief aggregate rests on a
                    # quarter of the sampled hypotheses. Carrying the
                    # denominator per decision is what lets a shard tell a
                    # healthy decision from a barely-survived one.
                    "worlds_constructed": len(worlds),
                    "aggregated_choices": {
                        choice: round(weight, 4) for choice, weight in aggregated.most_common()
                    },
                    "aggregated_choices_basis": (
                        "stopped_prefix" if locked_choice is not None else "full_budget"
                    ),
                    "early_stop": {
                        "enabled": config.early_stop,
                        "worlds_stopped": len(stopped_runs),
                        "aggregate_locked": locked_choice is not None,
                        "locked_choice": locked_choice,
                        "full_budget_replays": full_budget_replays,
                        "simulations_saved": simulations_saved,
                    },
                }
            },
        )

    # Gen 3 pool's only recharge move; the recharge turn itself is public.
    _RECHARGE_MOVES = frozenset({"hyperbeam"})

    def _recharging_slots(self, context: PolicyContext) -> tuple[str, ...]:
        """Slots publicly forced to recharge THIS turn (Hyper Beam landed last round).

        BOTH SIDES, since the self side went live. Our own slot comes from
        ``self_must_recharge`` and the opponent's from ``opponent_must_recharge`` -- two keys of
        the ONE parser ``must_recharge`` tracker, published per seat. The notes below describe
        the opponent side, whose reconstruction fallback predates the tracker; the self side has
        no fallback and is tracker-only, so an observation without the key simply carries no self
        lock.

        PREFERRED SOURCE: the parser's own ``must_recharge`` tracker, surfaced on the
        observation metadata as ``opponent_must_recharge`` (spec v4 pack A1). The parser reads
        the ``|-mustrecharge|SLOT`` line the sim emits when a recharge move LANDS, which is
        strictly better evidence than the reconstruction below: a missed Hyper Beam never emits
        it, the line names its own actor, and it cannot scroll out of a rolling window. Taking it
        here is what makes the world and the observation ONE PARSER TRUTH with TWO CONSUMERS
        rather than two derivations that can disagree.

        The reconstruction below remains the fallback for callers whose observation metadata
        predates the pack (older cached rollouts, hand-built contexts). It is strictly weaker,
        never stronger, so preferring the tracker can only remove wrong locks, never add one.

        FALLBACK — turn-exact signal: the round-indexed public action record (not the
        rolling event window) must show the opponent's action in the
        immediately-preceding round was a recharge move, and the rolling
        window must not carry a miss marker for it (a missed Hyper Beam does
        not recharge in gen3). If the record is unavailable the signal stays
        off — fail-open to the pre-fix behavior rather than inventing a lock.
        """

        opponent_slot = "p2" if context.player_id == "p1" else "p1"
        observation_metadata = getattr(context.observation, "metadata", None)

        # OUR OWN slot, from the same parser tracker. Until this existed only the opponent could
        # be locked.
        #
        # CORRECTED after review, because the obvious rationale is wrong and was asserted in
        # three places: the old world did NOT let our recharging mon pick any move. Showdown sets
        # `trapped: true` on a recharge request, so `engine_world` rejected these worlds outright
        # as `self_request_state_unsupported` and search FAILED CLOSED to the fallback. The real
        # effect of the asymmetry was that our recharge turns were UNSEARCHABLE, not mis-searched
        # -- except in the rare sub-case where the foe independently traps us, where the request
        # is accepted and the free-choice harm is real.
        #
        # It also forced leaf.rs to root-freeze the self-side MUSTRECHARGE volatile: deriving it
        # live would have contradicted a world that never carried it.
        #
        # `self_must_recharge` is the same `must_recharge` tracker that feeds
        # `opponent_must_recharge`, published per seat, so this is one parser truth applied to
        # both sides rather than a second derivation that can disagree with the first. Measured
        # on corpus/golden-v4: 1208 decision-row PAIRS (of 1295 rows; the rest have no partner
        # row in the corpus) where seat X's `self_must_recharge` equals seat Y's
        # `opponent_must_recharge`, zero disagreements.
        self_slot: tuple[str, ...] = ()
        if isinstance(observation_metadata, Mapping):
            if observation_metadata.get("self_must_recharge") is True:
                self_slot = (context.player_id,)

            tracked = observation_metadata.get("opponent_must_recharge")
            if tracked is True:
                return self_slot + (opponent_slot,)
            if tracked is False:
                # An explicit False from the tracker is a public PROOF of no lock, not an absent
                # signal — do not let the weaker fallback manufacture one behind it.
                return self_slot
        # The self side is settled by the tracker; the opponent side is not, so fall through to
        # the reconstruction and let it decide only the opponent. `self_slot` is () when we are
        # not locked, so one expression covers both cases -- review found the guarded and
        # unguarded forms were identical and one was dead.
        #
        # Prefixing HERE rather than inside the fallback is the point: that function has eleven
        # early `return ()` statements, each meaning "no OPPONENT lock", and any one of them
        # would otherwise silently drop our own. Pinned in
        # tests/test_recharging_slots_symmetry.py, which review demonstrated was necessary --
        # dropping the prefix left the entire suite green.
        return self_slot + self._opponent_recharging_fallback(context, opponent_slot)

    def _opponent_recharging_fallback(
        self, context: PolicyContext, opponent_slot: str
    ) -> tuple[str, ...]:
        """The pre-tracker reconstruction, for observations whose metadata predates the pack.

        Extracted verbatim from `_recharging_slots` when the self side became live: its eleven
        early `return ()` statements each meant "no OPPONENT lock", and once our own slot can
        also be locked those returns must no longer be able to discard it. Keeping them here and
        prefixing the self lock at the single call site is what removes the need to remember it
        eleven times -- but structure is not a test, and review showed dropping the prefix left
        the whole suite green, so it is pinned in tests/test_recharging_slots_symmetry.py.

        Strictly weaker than the tracker, never stronger, so it can only fail to add a lock.
        """
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
    ) -> tuple[
        dict[str, str],
        dict[str, str],
        dict[str, tuple[str, ...]],
        dict[str, dict[str, str]],
        dict[str, str],
    ]:
        """Public-information signals engine_world cannot see in the payload.

        - blocked_slots: the opponent's active is publicly Transformed (the
          belief engine tracks it; the payload does not) — the sampled world
          cannot express the copied moveset/stats, so construction must fail
          closed rather than search a silently wrong world.
        - encored_moves: the opponent's publicly-observed last move, consumed
          by engine_world only when that side carries the encore volatile.
        - removed_item_species: per slot, species whose held item is publicly
          GONE (Knock Off, an item-taking Trick, or a public consumption —
          berry eaten / White Herb) and not replaced since — the current
          public item state is exactly "no item", which the sampled world
          expresses by clearing the sampled set's item.
        - current_item_overrides: per slot, species → CURRENT item id for mons
          whose held item was Trick-swapped AND whose resulting item the
          protocol positively named (belief ``current_public_item``, set only
          by the audited ``|-item|...|[from] move: Trick`` line, holder
          identified from the line itself) — the sampled world substitutes the
          revealed current item for the sampled assignment's. A mutated mon
          with NO protocol-confirmed current item stays fail-closed: wrong
          items in searched worlds are strictly worse than falling back.

        Item signals cover BOTH seats: the self side's sampled world is the
        battle-START packed team, so after a Trick/eat our own mon's world
        item is just as stale as an opponent's (the self seat never walled —
        it was silently wrong; belief tracks both sides from the same public
        lines).
        """

        blocked: dict[str, str] = {}
        encored: dict[str, str] = {}
        transformed: dict[str, str] = {}
        removed: dict[str, tuple[str, ...]] = {}
        overridden: dict[str, dict[str, str]] = {}
        metadata = context.observation.metadata
        if not isinstance(metadata, Mapping):
            return blocked, encored, removed, overridden, transformed
        self_slot = context.player_id
        opponent_slot = "p2" if context.player_id == "p1" else "p1"
        belief_view = metadata.get("belief_view")
        opponents = belief_view.get("opponent_pokemon") if isinstance(belief_view, Mapping) else None
        self_rows = belief_view.get("self_pokemon") if isinstance(belief_view, Mapping) else None

        for slot, rows in ((opponent_slot, opponents), (self_slot, self_rows)):
            for mon in rows or ():
                if not isinstance(mon, Mapping):
                    continue
                species_id = normalize_id(str(mon.get("species") or ""))
                if mon.get("item_removed"):
                    # Publicly holds nothing NOW (stripped or consumed) —
                    # checked before item_mutated: a consumed berry does not
                    # mutate, and a mutated-then-consumed mon is still just
                    # itemless. Any stale current_public_item loses to this.
                    if species_id:
                        removed[slot] = removed.get(slot, ()) + (species_id,)
                elif mon.get("item_mutated"):
                    current_item = normalize_id(str(mon.get("current_public_item") or ""))
                    if species_id and current_item:
                        overridden.setdefault(slot, {})[species_id] = current_item
                    else:
                        # Mutated with no protocol-confirmed current item
                        # (unaudited mutation source, or a pre-override
                        # serialized payload): no sampled world can express
                        # it, so construction fails closed.
                        blocked[slot] = f"item mutated on {mon.get('species')} with unconfirmed current item"

        # Transform is symmetric: our OWN Ditto desyncs just as hard, and worse
        # -- the request advertises the copied moveset while the sampled world
        # still holds Ditto's real one, which surfaced as self_moveset_mismatch
        # rather than a transform block and was the larger half of the two.
        for slot, rows in ((opponent_slot, opponents), (self_slot, self_rows)):
            for mon in rows or ():
                if not isinstance(mon, Mapping) or not mon.get("active"):
                    continue
                if not mon.get("transformed"):
                    continue
                target = str(mon.get("transform_species") or "")
                if target:
                    transformed[slot] = target
                else:
                    # Publicly transformed but the copied species was never
                    # named: nothing to bake in, so keep failing closed.
                    blocked[slot] = "active transformed into an unnamed species"

        active_species: str | None = None
        for mon in opponents or ():
            if not isinstance(mon, Mapping):
                continue
            if not mon.get("active"):
                continue
            active_species = str(mon.get("species") or "") or None
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
        return blocked, encored, removed, overridden, transformed

    def _map_choices(
        self, context: PolicyContext, aggregated: Mapping[str, float]
    ) -> Optional[int]:
        candidates = context.observation.metadata.get("action_candidates")
        # `str` and `bytes` ARE Sequences, so `isinstance(candidates, Sequence)` alone lets a
        # stringified metadata field walk straight past this guard and land in the POLICY
        # bucket below -- defeating the purpose of the one token that exists to say "this is
        # plumbing, not a game state". Review found exactly that.
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            self.stats.choices_unmapped_causes[_CAUSE_NO_ACTION_CANDIDATES] += 1
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

        # Recorded BEFORE the mapping loop, because the loop's own bookkeeping cannot
        # distinguish "the engine proposed a move and no move was legal" from "the engine
        # proposed a move and a DIFFERENT move was legal". The first is a switch-only
        # decision -- a force switch, or an active mon with no usable move -- and the fix is
        # for the policy to propose a switch. The second is a legality mismatch between the
        # engine's world and the request: PP exhaustion, Taunt, Disable, or a choice/Encore
        # lock. Those are different bugs with different owners, and era 60 could not tell
        # them apart.
        any_legal_move = bool(move_index_by_id) or hidden_power_index is not None
        any_legal_switch = bool(switch_index_by_species)

        best_index: Optional[int] = None
        best_weight = 0.0
        mapped_any = False
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
                if index is None and move_id in _ENGINE_FORCED_NO_MOVE_IDS:
                    # The crate displays MoveChoice::None as "No Move" for a slot the engine has
                    # locked -- a recharge turn is the case that reaches here. Showdown's request
                    # for that turn offers exactly one candidate, and it is named `recharge`, so
                    # the two vocabularies describe the same forced action under different names
                    # and the lookup above misses.
                    #
                    # `depth_tactics_probe.py` already carried this translation ("No Move" ->
                    # "none"); `_map_choices` never got it, so before this the decision fell to
                    # `_fallback(..., "choices_unmapped")` -- a counter this file states at
                    # :673-675 must be zero independently of the fallback rate, and with the cause
                    # mislabelled `all_unmapped_legality_mismatch`. It only became reachable once
                    # `_recharging_slots` went symmetric and these worlds started building at all.
                    index = move_index_by_id.get("recharge")
            if index is None:
                self.stats.unmapped_choices[choice] += 1
                continue
            mapped_any = True
            if weight > best_weight:
                best_weight = weight
                best_index = index
        if best_index is None:
            self.stats.choices_unmapped_causes[
                _registered_cause_or_unclassified(_classify_unmapped(
                    aggregated=aggregated,
                    mapped_any=mapped_any,
                    any_legal_move=any_legal_move,
                    any_legal_switch=any_legal_switch,
                ))
            ] += 1
        return best_index

    def _fallback(
        self, context: PolicyContext, rng: random.Random, reason: str
    ) -> PolicyDecision:
        self.stats.fallback_decisions += 1
        self.stats.fallback_reasons[reason] += 1
        battle_id = getattr(context, "battle_id", "?")
        round_index = getattr(context, "decision_round_index", None)
        player = getattr(context, "player_id", "?")
        # Per-decision world-failure context: the cumulative counters minus
        # the snapshot taken at the top of _search.
        delta = {
            key: count - self._world_failures_before.get(key, 0)
            for key, count in self.stats.world_failure_reasons.items()
            if count - self._world_failures_before.get(key, 0) > 0
        }
        # Retain the address under EVERY class this decision failed on AND under the
        # fallback reason -- always, not only when there are no world failures.
        #
        # Keying on the reason unconditionally is what makes a rare REASON
        # addressable. Reasons and classes are separate axes: `choices_unmapped`
        # co-occurs with world failures, so a delta-only key files its one address
        # under a class whose three slots the dominant reason has already taken, and
        # the rare reason ends up with no address at all. That is the exact era-57
        # failure mode this store exists to prevent, on the other axis. The reason
        # set is closed and small (7 literals), so this adds at most 7 keys.
        # Reason key FIRST, and exempt from the ceiling. The EXEMPTION is what carries
        # the property -- reverting the ordering alone leaves the suite green, so the
        # ordering is defence-in-depth, not load-bearing, and is recorded as such rather
        # than claimed as tested. The ceiling
        # below exists to bound an unbounded CLASS space; applying it to reason keys
        # reintroduced the very bug this loop was fixed for -- past 256 classes a rare
        # reason lost its key and became unaddressable again, and because classes were
        # served first they took the last slot and the reason key was what got dropped.
        # Exactly backwards. The reason set is 7 closed literals and can never be the
        # thing that blows up a report, so it never competes.
        for key in [f"fallback:{reason}", *delta]:
            bucket = self.stats.fallback_samples.get(key)
            if bucket is None:
                if (not key.startswith("fallback:")
                        and len(self.stats.fallback_samples)
                        >= _FALLBACK_SAMPLE_KEY_CEILING):
                    self.stats.fallback_sample_addresses_dropped += 1
                    continue
                bucket = self.stats.fallback_samples[key] = []
            if len(bucket) >= _FALLBACK_SAMPLES_PER_CLASS:
                continue
            # One address per BATTLE. A refusal cause typically closes worlds for the
            # rest of the battle it appears in, so first-3 retention would hand back
            # rounds N, N+1 and N+2 of a single battle -- three views of one incident,
            # which cannot tell you whether the class generalises. Three DIFFERENT
            # battles can. A class confined to one battle keeps one address, which is
            # all replay needs, and the count still lives in world_failure_reasons.
            if any(entry["battle_id"] == str(battle_id) for entry in bucket):
                continue
            bucket.append({
                "battle_id": str(battle_id),
                "round": round_index,
                "seat": str(player),
                "reason": reason,
            })
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
    parser.add_argument("--checkpoint", default=None,
                        help="source transformer checkpoint whose observation contract is latched")
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
    parser.add_argument("--early-stop", action="store_true",
                        help="model mode: stop only when remaining sims cannot change the action")
    parser.add_argument("--early-stop-min-sims", type=int, default=64,
                        help="minimum per-world sims before safe STOP checks (default: 64)")
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
        checkpoint_path=args.checkpoint,
        tables_path=args.tables,
        model_device=args.model_device,
        search_sims=args.sims,
        search_batch=args.batch,
        search_depth=args.depth,
        model_priors=not args.no_model_priors,
        early_stop=args.early_stop,
        early_stop_min_sims=args.early_stop_min_sims,
        fold_cross_check=args.fold_cross_check,
    )
    # Model mode needs the belief candidate-set source: the checkpoint observation's
    # belief columns (candidate variants, possible sets) are part of the
    # surface the model was trained on. With the source attached and the
    # checkpoint's feature masks, the env's Tier-2 trackers may be active —
    # the policy therefore needs the annotation source (below) or a Tier-2-on
    # checkpoint's live fold would present unannotated surfaces at search leaves.
    env_config = LocalShowdownConfig(
        showdown_root=args.showdown_root,
        set_belief_source=True if model_mode else None,
    )
    if model_mode:
        from .local_showdown import env_config_from_checkpoint_provenance  # noqa: PLC0415
        from .neural_policy import (  # noqa: PLC0415
            category_vocab_from_model_config,
            feature_masks_from_model_config,
            load_transformer_model_config,
            observation_spec_from_model_config,
        )

        model_config = load_transformer_model_config(str(args.checkpoint))
        env_config = env_config_from_checkpoint_provenance(
            env_config,
            feature_masks_from_model_config(model_config),
            required_specs=observation_spec_from_model_config(model_config),
            required_vocabs=category_vocab_from_model_config(
                model_config, env_config.resolved_showdown_root()
            ),
            context="engine MCTS model benchmark",
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
            overrides_before = policy.stats.item_override_decisions
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
                "item_override_decisions": policy.stats.item_override_decisions - overrides_before,
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
            "use_opponent_priors": config.use_opponent_priors,
            "early_stop": config.early_stop,
            "early_stop_min_sims": config.early_stop_min_sims,
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
