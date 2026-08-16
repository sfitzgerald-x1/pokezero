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
import math
import random
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Optional, Sequence

from .dex import ShowdownDex, normalize_id
from .determinization import (
    _gen3_randbat_belief_start_override_result,
    _move_from_public_event_line,
)
from .public_action_capture import public_action_rounds_from_trajectory_metadata
from .engine_world import EngineWorld, EngineWorldUnsupported, world_battle_spec
from .randbat import canonical_gen3_randbat_species_id
from .poke_engine_adapter import (
    PokeEngineAttractUnsupportedError,
    PokeEngineMoveTrapUnsupportedError,
    PokeEngineUnavailableError,
    build_poke_engine_state,
)
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


def _root_toxic_action_phase_reset(replay: object, slot: str) -> bool:
    """Whether ``slot``'s active holds the action-phase Toxic reset proof.

    Same public fact as the second zero in
    ``local_showdown._materialization_toxic_stage``: this active entered badly
    poisoned during the current turn's ACTION phase, so Showdown's
    ``tox.onSwitchIn`` has zeroed ``statusState.stage`` and no residual has been
    charged since.

    The Rust leaf spends that fact on a different job — ``toxic_reentry_pending``
    is what lets a rounded ``/100`` residual be priced as stage one — so the two
    lanes have to agree. If only the materialization lane learned it, a world this
    change newly constructs would render a Toxic stint whose counter never leaves
    zero, which is a world that stopped refusing without starting to count.
    """

    def slot_value(name: str) -> Any:
        values = getattr(replay, name, None)
        return values.get(slot) if isinstance(values, Mapping) else None

    ident = slot_value("toxic_stage_reset_ident")
    active = slot_value("public_active")
    stage = slot_value("toxic_stage")
    return (
        isinstance(ident, str)
        and ident.startswith(f"{slot}a: ")
        and getattr(active, "ident", None) == ident
        and slot_value("toxic_stage_known") is True
        and type(stage) is int
        and stage == 0
        and getattr(replay, "post_upkeep_window", None) is False
    )


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

    def zero_counter_proof(slot: str) -> bool | None:
        # A malformed post-upkeep field still serializes as ``None`` so the leaf
        # decoder keeps failing closed; only an exact ``False`` — "no post-upkeep
        # replacement proof here" — is allowed to be answered by the action-phase
        # reset proof instead.
        after_upkeep = exact_bool_field("toxic_stage_zero_after_upkeep", slot)
        if after_upkeep is not False:
            return after_upkeep
        return _root_toxic_action_phase_reset(replay, slot)

    post_upkeep_window = getattr(replay, "post_upkeep_window", None)
    exact_post_upkeep_window = (
        post_upkeep_window if type(post_upkeep_window) is bool else None
    )
    return {
        slot: {
            "proof": zero_counter_proof(slot),
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


def _world_visit_shares(
    side_key: str, report: Mapping[str, Any]
) -> Optional[dict[str, float]]:
    """One world's per-arm visit SHARES, or None if the report cannot be read.

    Shares, not raw visits, so worlds searched at different budgets (a collapsed
    group runs at multiplicity x sims) weigh equally -- the same normalisation
    `_locked_aggregate_choice` applies, minus its unspent-simulation bound, which
    is meaningless once a rung has run to completion.
    """
    entries = report.get(side_key)
    if not isinstance(entries, Sequence):
        return None
    visits: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            return None
        choice = str(entry.get("move") or "")
        count = int(entry.get("visits", -1))
        if not choice or count < 0:
            return None
        visits[choice] = visits.get(choice, 0) + count
    total = sum(visits.values())
    if total <= 0:
        return None
    return {choice: count / total for choice, count in visits.items()}


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


def _is_identity_calibration(transform: Any) -> bool:
    """True only if ``transform`` is a no-op on EVERY field it carries.

    Two review findings are encoded here, both worth stating because both produced a guard
    that looked correct and covered nothing:

    * ``save_transformer_checkpoint`` persists ``to_dict()`` -- a plain **dict**. A first
      version read the fields with ``getattr`` and so saw ``None`` for every key on every
      real artifact, which made this identity branch unreachable: it would have refused every
      calibrated checkpoint, including an explicitly identity one, and no test caught it
      because the tests pickled a live dataclass, a shape production never writes.
    * That version also compared 3 of the 6 fields, so a clip-narrowed transform
      (``clip_min=-0.2, clip_max=0.2``) passed the guard while mapping 0.9 -> 0.2.

    So: normalise the dict to the dataclass, then compare every field against the default.

    A third finding, from the review of the fixed version: ``from_dict`` reads only the six
    keys it knows and **silently drops the rest**, so an identity-looking dict carrying an
    unknown ``gamma: 3.0`` parsed to the identity and passed. The day the on-disk shape gains
    a field and ``from_dict`` is not updated in lockstep, that is the ``getattr`` failure over
    again. Unknown keys, and dicts missing keys ``to_dict`` always writes, are refused: the
    rule is *not provably identity => refuse*, and a shape we do not fully understand is not
    provably anything.
    """
    from .neural_policy import ValueCalibrationTransform  # noqa: PLC0415

    known = {spec.name for spec in fields(ValueCalibrationTransform)}
    if isinstance(transform, Mapping):
        unknown = set(transform) - known
        if unknown:
            return False
        # A dict missing any key `to_dict` always writes is not a shape this code produced,
        # so identity cannot be established from it. Derived from an actual identity object
        # rather than a hardcoded literal, so it tracks the writer automatically (`points` is
        # excluded by construction: `to_dict` emits it only for isotonic).
        if not set(ValueCalibrationTransform().to_dict()) <= set(transform):
            return False
        try:
            transform = ValueCalibrationTransform.from_dict(transform)
        except (TypeError, ValueError, KeyError):
            # Unparseable => not PROVABLY identity => refuse. Never treat a shape we
            # failed to understand as benign.
            return False
    identity = ValueCalibrationTransform()
    for spec in fields(ValueCalibrationTransform):
        sentinel = object()
        if getattr(transform, spec.name, sentinel) != getattr(identity, spec.name):
            return False
    return True


def _fence_calibration_seam(payload: Mapping[str, Any], where: str) -> None:
    """Refuse a model-leaf search whose checkpoint carries a calibration the crate ignores.

    The crate applies NO calibration: ``model.rs`` maps the raw tanh through ``0.5*(v+1.0)``
    and nothing else touches a leaf value. ``scripts/export_model.py`` does not bake the
    transform into the trace either. So a checkpoint carrying one has a Python value axis and
    a crate value axis that differ, with nothing reporting the difference.

    SCOPE, corrected after review. Under ``leaf_eval="model"`` every value-axis quantity is
    crate-produced (``arm_q``, the root Q gap, early-stop on visit counts), so a transform
    does not by itself desync crate values from Python-side thresholds -- an earlier version
    of this docstring claimed that, and it was wrong. What it does break is any comparison
    that puts a crate-valued seat beside a Python-valued one, which the bridge permits
    directly (``--policy-mode engine-mcts`` vs ``--opponent-policy-mode root-puct``), and any
    figure read off the head in Python and attributed to search. On the current checkpoint
    the field is ``None``, so this fence is inert TODAY; it exists so the seam cannot open
    silently later.

    LIMITATION, stated because the fence's reach is narrower than its name suggests. It reads
    ``checkpoint_path``; the crate runs ``model_path`` via ``CModule::load_on_device``. So it
    catches "the checkpoint this trace came from declares a calibration" and does NOT catch a
    trace exported from some *other* checkpoint, nor a calibration baked into the trace
    itself. Pairing ``model_path`` to ``checkpoint_path`` is a separate provenance question
    and is not settled here. Nor does it cover the two benches that build
    ``pokezero_search.NativeLeafModel`` directly and never construct an ``EngineMctsPolicy``
    at all -- ``scripts/bench_leaf_search.py`` and ``scripts/bench_crate_search.py`` -- which
    bypass this fence entirely. An earlier revision of this list also named
    ``scripts/depth_tactics_probe.py``; that was WRONG. It builds an ``EngineMctsPolicy``
    (``depth_tactics_probe.py:891``) and so is fenced, as is
    ``truth_differential_census.py``; both merely mention ``NativeLeafModel`` in a docstring.
    A disclosure that over-states which paths are unguarded is its own kind of wrong.
    """
    from .neural_policy import NEURAL_POLICY_SCHEMA_VERSION  # noqa: PLC0415

    if "value_calibration_transform" not in payload:
        # Absence is benign ONLY on a schema that predates the field. On the CURRENT schema
        # every checkpoint carries it (verified: all 13 in checkpoints/), so a missing key
        # means the field was renamed or dropped -- which would give a real calibration a
        # silent path through the fence. Review found this branch failing open with no
        # artifact justifying it.
        if payload.get("schema_version") == NEURAL_POLICY_SCHEMA_VERSION:
            raise ValueError(
                f"REFUSING model-leaf search: {where} declares the current schema "
                f"({NEURAL_POLICY_SCHEMA_VERSION!r}) but carries no "
                "'value_calibration_transform' key. Every checkpoint on this schema writes "
                "one, so the field has been renamed or dropped and this fence can no longer "
                "see it. Update the fence alongside the schema."
            )
        return  # pre-provenance checkpoint: the field did not exist when it was written
    transform = payload["value_calibration_transform"]
    if transform is None or _is_identity_calibration(transform):
        return
    raise ValueError(
        f"REFUSING model-leaf search: {where} carries a value calibration transform "
        f"({transform!r}) and the search crate applies NONE -- it maps the raw tanh through "
        "0.5*(v+1.0). Crate leaf values would then sit on a different axis from any "
        "Python-evaluated seat compared against them, silently. Either add calibration to "
        "the crate, or run leaf_eval='hp_fraction_crate', or strip the transform "
        "deliberately and record that choice in the campaign config."
    )


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
    # First-play urgency for UNVISITED arms in the native PUCT selection.
    # None = the flat 0.5 the crate has always used (`MoveStats::mean` at zero
    # visits); a float r prices an unvisited arm at
    # clamp(parent mean in that seat's frame - r, 0, 1), which is KataGo/Leela's
    # `fpuReductionMax` minus the sqrt-of-visited-policy-mass scaling the
    # selection-tuning plan defers on purpose (one new mechanism per stage).
    #
    # Default None for the same reason `use_opponent_priors` defaults False:
    # flag-off must be the search every recorded result was produced under, and
    # that equivalence is asserted bit-for-bit, not argued.
    fpu_reduction: float | None = None
    # Per-decision override telemetry: does search play something other than the
    # raw policy head's argmax? Pure measurement -- it asks the crate for the
    # per-arm `prior` column and reads it; nothing on the search's path changes,
    # and flag-off is byte-identical for that reason rather than by recomputation.
    #
    # It also turns on the absorption the plan's offline legs were blocked on:
    # the root Q/visit gaps between the top two arms (H2) and the in-tree
    # opponent's top arm (H4), both of which the crate already emitted and this
    # module discarded.
    #
    # Default OFF like every other new axis here, and for one more reason
    # besides: turning it on appends a positional to the native call, so a stale
    # image would refuse every world (`native_override_telemetry_unsupported`).
    # An on-by-default field would make a Python-only update break a running
    # image.
    override_telemetry: bool = False
    # Opt-in safe STOP rule. A tree may stop at a completed batch only after
    # this floor and only when the unspent simulations cannot change its root
    # visit argmax. Multi-world aggregation applies a second safety bound.
    early_stop: bool = False
    early_stop_min_sims: int = 64
    # DYNAMIC BUDGET LADDER (docs/dynamic-search-budget-plan-20260812.md).
    #
    # `search_depth` and `worlds` remain the MAXIMA, unchanged in meaning. These
    # are the floors, and setting either turns that axis dynamic: the decision is
    # searched at the floor and escalated toward the cap only while its aggregate
    # choice is still ambiguous. Both unset is today's fixed budget exactly -- one
    # rung, one search per world, byte-identical positionals.
    #
    # "depth 3 min 6 max" is depth_min=3 with search_depth=6; "worlds 2 min 16
    # max" is worlds_min=2 with worlds=16.
    depth_min: int | None = None
    worlds_min: int | None = None
    #: Share of a rung's worlds that must reach the depth ceiling (D-1, since
    #: `depth_reached == cap` is unreachable by construction) before DEPTH is
    #: allowed to advance. NEAR-FULL by default, deliberately: a deeper search that
    #: did not fully fill the shallower depth can be WORSE than the shallower one,
    #: because the extra plies are explored too thinly to be backed up reliably.
    #: Depth marches forward on saturation pressure alone, so this threshold is the
    #: only thing standing between the ladder and a thin deep tree.
    #:
    #: This is the rule that makes the ladder coherent. Depth is meaningless
    #: relative to an unfilled budget: measured on this campaign's own canary,
    #: s1024 at depth 4 saturates (96.7% of samples at D-1) while the SAME budget
    #: at depth 6 reaches its ceiling on 4.8% -- so deepening without the sims to
    #: fill the new plies buys a thinly-populated deep tree and pays ~2x per
    #: decision for it. Depth therefore waits until the current depth is actually
    #: saturated, and saturation is bought by scaling WORLDS DOWN, which funds more
    #: sims per world at constant total compute.
    ladder_saturation: float = 0.9
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
                # Against `search_sims`, which on a DYNAMIC cell is the TOTAL for
                # the decision rather than a per-world budget -- so passing here
                # does not mean the floor fits every rung. `_search_model` clamps it
                # to the rung; see the note there for why clamping is the
                # conservative direction and not a relaxed bar.
                raise ValueError(
                    "early_stop_min_sims must be in 1..=search_sims when early_stop "
                    "is enabled. On a dynamic cell search_sims is the TOTAL budget "
                    "for a decision, so a floor that passes here can still exceed an "
                    "individual rung's per-world share; it is clamped to the rung."
                )
            if (self.depth_min is not None or self.worlds_min is not None) and (
                self.search_sims < self.worlds
            ):
                # A LADDER cell divides `search_sims` across its worlds, so a
                # budget below the world cap cannot give every world a simulation.
                # An earlier revision clamped instead, and the clamp was worse than
                # the config it was rescuing: at search_sims=2, worlds=4,
                # worlds_min=3 it ran TWO worlds while the floor said three -- it
                # broke the very floor that turned the axis dynamic -- and it left
                # rung 0 asking for fewer worlds than `decide()` had already
                # charged, which put a FALSE 0.25 into `world_search_abort_rate`
                # (the mirror of the -1.75 that review found). Refused instead:
                # this is not a config anyone wants, and degrading it silently
                # produces a cell whose name does not describe what ran.
                raise ValueError(
                    "search_sims must be >= worlds on a dynamic cell "
                    f"(got search_sims={self.search_sims}, worlds={self.worlds}); "
                    "search_sims is the TOTAL budget divided across belief worlds, "
                    "so a smaller budget cannot give every world one simulation."
                )
            if self.depth_min is not None and not (
                LADDER_MIN_DEPTH_FLOOR <= self.depth_min <= self.search_depth
            ):
                # The floor is 2, not 1, and that is a STRENGTH constraint rather
                # than a taste: a one-ply search is no better than the raw policy,
                # so a ladder allowed to start at depth 1 would spend its cheapest
                # and most common rung forfeiting the entire search advantage --
                # measured in this programme at raw ~44% against search ~54%.
                raise ValueError(
                    f"depth_min must be in {LADDER_MIN_DEPTH_FLOOR}..=search_depth "
                    f"when set (got {self.depth_min} with "
                    f"search_depth={self.search_depth}); depth 1 is one-ply and no "
                    "better than the raw policy."
                )
            if not 0.0 <= self.ladder_saturation <= 1.0:
                raise ValueError(
                    f"ladder_saturation must be in 0.0..=1.0 (got {self.ladder_saturation})."
                )
            if self.worlds_min is not None and not 0 < self.worlds_min <= self.worlds:
                raise ValueError(
                    "worlds_min must be in 1..=worlds when set "
                    f"(got {self.worlds_min} with worlds={self.worlds})."
                )
        elif self.early_stop:
            raise ValueError("early_stop is supported only with leaf_eval='model'.")
        elif self.depth_min is not None or self.worlds_min is not None:
            # Same standing as early_stop: the ladder re-invokes the native model
            # search, so outside leaf_eval='model' it would do nothing and the
            # cell would claim a dynamic budget it never had.
            raise ValueError(
                "depth_min/worlds_min are supported only with leaf_eval='model'."
            )
        if self.override_telemetry and self.leaf_eval != "model":
            # Refused rather than silently zero. The measurement needs the
            # model's root priors, which only the model path computes, so on any
            # other leaf_eval every counter would read 0 -- indistinguishable
            # from "search never overrides the model", which is the exact
            # always-reads-zero failure this counter exists to avoid.
            raise ValueError(
                "override_telemetry is supported only with leaf_eval='model'."
            )
        if self.fpu_reduction is not None and not 0.0 <= self.fpu_reduction <= 1.0:
            # Refused here as well as in the crate: Q is a win probability, so a
            # negative reduction is a first-play BONUS -- the opposite of the
            # mechanism -- and a shard that typed one would otherwise discover
            # it only from the search behaving backwards.
            raise ValueError(
                "fpu_reduction must be in 0.0..=1.0 when set, "
                f"got {self.fpu_reduction!r}."
            )


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


# Budget for one crate-error `world_failure_reasons` key. Was 160, which the
# attract sub-case split (#1030) outgrew: the refusal message is
# `attribution-unsafe renderer branch rejected before <lane>: <slugs>`, whose
# prefix alone eats ~68 chars, and one fully-live attract slug is 68 more
# (`attract_empty_tail_ambiguous:paralyzed+miss+noop+volatile+cannot_act`). Two
# sides refusing with DIFFERENT slug sets is routine and blew straight past 160.
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

# The attribute an ABORTING native search hangs its accumulated sub-case counts on, so a
# world that dies still reports what it observed. Must equal `ABORT_PAYLOAD_ATTR` in
# `rust/pokezero-search/src/abort_telemetry.rs`; a rename on either side silently zeroes
# the abort arm, which is the failure mode this whole change exists to fix, so the two
# spellings are asserted against each other in tests/test_engine_search.py.
#
# NAMESPACED rather than named after the report key (`lossy_subcases`). This is read off
# an arbitrary caught exception, and a generic name would let an unrelated exception that
# happens to carry that attribute inject counts into a measurement channel.
_ABORT_LOSSY_SUBCASES_ATTR = "pokezero_lossy_subcases"


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
#: Showdown names the same forced action `recharge` on a recharge turn and `struggle` when the
#: request would otherwise offer no move at all. ONE engine token, TWO request spellings; both
#: translations, and the admission test the Struggle one needs, live in `_map_choices`.
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


#: Why a SEARCHED decision could not be scored for a model override. Closed and
#: greppable, for the same reason `_CHOICES_UNMAPPED_CAUSES` is: an override rate
#: whose denominator excluded some decisions is only readable if the excluded set
#: can be named, and these five have different owners entirely.
_OVERRIDE_UNMEASURED_NO_PRIORS = "no_root_priors"
_OVERRIDE_UNMEASURED_ARMS_ABSENT = "prior_arms_absent"
_OVERRIDE_UNMEASURED_ARMS_MISALIGNED = "prior_arms_misaligned"
_OVERRIDE_UNMEASURED_PARTIAL_WORLDS = "priors_missing_in_some_worlds"
_OVERRIDE_UNMEASURED_UNMAPPED = "model_choice_unmapped"
_OVERRIDE_UNMEASURED_CAUSES = (
    # The crate priced no root priors for this decision: `model_priors` off, or a
    # root the prior path refused (an option list the action map could not
    # resolve, a prior row that underflowed, a root with a single forced
    # `MoveChoice::None`). The model has no argmax here to disagree with, and
    # calling that agreement would understate the override rate.
    _OVERRIDE_UNMEASURED_NO_PRIORS,
    # Priors are present but unpaired: `root_priors` is there and the arms carry
    # no `prior` column. That is a STALE IMAGE running new Python -- the crate
    # emits the column whenever the flag reaches it -- and it is the one cause
    # that means "the measurement is not installed", not "this decision resisted
    # measurement".
    _OVERRIDE_UNMEASURED_ARMS_ABSENT,
    # Names and values disagree in LENGTH. Defensive: the crate writes both off
    # one stat vector, so this cannot happen without a crate bug -- but pairing
    # them anyway would silently attribute one arm's prior to another arm, which
    # is a wrong answer rather than a missing one.
    _OVERRIDE_UNMEASURED_ARMS_MISALIGNED,
    # SOME searched worlds priced root priors and some did not. The model's
    # decision-level argmax is the prior mass aggregated the same way visits are,
    # so a subset aggregate is a different quantity from the one the rate claims
    # to measure. Refused rather than approximated.
    _OVERRIDE_UNMEASURED_PARTIAL_WORLDS,
    # The model's argmax display does not name any legal request action. The
    # search's own choice mapped (or the decision would have fallen back), so
    # this is a real engine/request vocabulary gap confined to the arm the model
    # liked most -- and it is NOT counted in `unmapped_choices`, by design: this
    # probe must not move a counter a stop condition reads.
    _OVERRIDE_UNMEASURED_UNMAPPED,
)


#: Forkable disagreement addresses retained per policy. The fork probe
#: (section 4b) samples ~50; 64 covers it with headroom while keeping the block
#: small enough to ride in every shard report. Overflow is COUNTED
#: (`override_disagreement_addresses_dropped`), never silently dropped -- a
#: truncated sample that looks complete is how a coverage claim goes wrong.
_OVERRIDE_DISAGREEMENT_ADDRESSES = 64

#: Per-decision root rows retained per policy. A 100-game shard makes a few
#: thousand decisions, so this holds a whole ordinary cell; past it the block
#: truncates and `root_decision_rows_dropped` says by how much. Sized against the
#: shard, not the campaign: at ~200 bytes a row this is a ~800 KB ceiling, next to
#: the ~124 KB the fallback address store is bounded at.
_ROOT_DECISION_ROWS = 4096


def _registered_override_cause(cause: str) -> str:
    """Degrade an unregistered override-unmeasured cause, same rule as below.

    Separate vocabulary, separate degradation: folding these into
    `_CHOICES_UNMAPPED_CAUSES` would put a telemetry token into the closed set a
    campaign stop condition is defined on.
    """
    return cause if cause in _OVERRIDE_UNMEASURED_CAUSES else _CAUSE_UNCLASSIFIED


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


def world_cache_key(record: Mapping[str, Any], side_key: str) -> tuple[str, str, str]:
    """Identity of a SEARCH PROBLEM, for collapsing duplicate belief draws.

    Two sampled worlds with the same serialized state, the same context and the
    same seat are not two hypotheses; they are one hypothesis drawn twice.

    The per-world SEED is deliberately excluded. Differing seeds are exactly
    what makes two identical completions look like distinct searches, which is
    the redundancy being removed -- include the seed and the collapse never
    fires.
    """
    return (str(record["state_str"]), str(record["ctx_json"]), str(side_key))


@dataclass
class EngineMctsStats:
    """Cumulative per-policy telemetry; every fallback is counted, never hidden."""

    decisions: int = 0
    searched_decisions: int = 0
    fallback_decisions: int = 0
    # Decisions an ORACLE-BELIEF arm injected the TRUE world for (value-gap plan
    # §4a). Incremented by whatever harness installs `fixed_override` per
    # decision, NOT by the search: a config flag can only witness that the arm was
    # REQUESTED, and this campaign already learned the difference the expensive way
    # on opponent priors (accepted, then refused inside the crate, reported as a
    # clean null). Zero on every other arm; equal to `decisions` on a healthy
    # oracle run, which is the check.
    oracle_belief_decisions: int = 0
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
    #: World-SEARCH attempts, charged once per world per rung. Distinct from
    #: `worlds_constructed`, which is a CONSTRUCTION count and is rightly charged
    #: once per decision: a ladder re-searches the same constructed worlds at each
    #: rung, so `worlds_searched` can exceed `worlds_constructed` and the abort rate
    #: computed against construction went NEGATIVE (measured -1.75). Found in review.
    world_search_attempts: int = 0
    worlds_searched: int = 0
    # Duplicate draws folded into another world's search (drawn N, searched 1
    # at N x budget), so `worlds_searched - worlds_collapsed ==
    # unique_worlds_searched`. Both sides count SUCCEEDING searches only: a
    # search that aborts contributes no records to `worlds_searched`, so
    # counting its collapse here would make the identity negative.
    worlds_collapsed: int = 0
    # Distinct belief completions that were searched AND returned a report.
    # Aborted searches are not counted here -- they are counted, per world draw,
    # in `world_failure_reasons`.
    unique_worlds_searched: int = 0
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
    # Within-batch selection collisions, PER SEAT (model mode only; the crate
    # reports these from `multiply_batched_encoded_core` and nowhere else,
    # because a collision is a property of a batch and the sequential core
    # finalizes every selection before taking the next).
    #
    # Summed across searched worlds, numerator AND denominator, so a consumer
    # divides pooled totals instead of averaging per-world ratios -- worlds do
    # not all contribute the same number of selections (early stop, aborts,
    # collapsed worlds searched at multiplicity x budget), and a mean of ratios
    # would weight a 64-sim world like a 2048-sim one.
    #
    # The seat mapping is already applied crate-side: `..._self_...` is the
    # searching seat whichever side it sat on. That matters because the
    # mechanism under audit is SIDE-absolute -- the round's unreconciled
    # provisionals land on `s2_stats` regardless of who is searching -- so a p1
    # decision and a p2 decision put the asymmetry in opposite raw counters and
    # only the seat-mapped view pools correctly.
    collision_rounds: int = 0
    collision_pending_rounds: int = 0
    collision_selections: int = 0
    collision_joint_repeats: int = 0
    collision_self_repeats: int = 0
    collision_opponent_repeats: int = 0
    collision_traversals: int = 0
    collision_leaf_repeats: int = 0
    early_stop_triggered_worlds: int = 0
    early_stop_accepted_decisions: int = 0
    early_stop_full_budget_replays: int = 0
    early_stop_sims_saved: int = 0
    # Dynamic budget ladder. `rungs_run` counts SEARCH PASSES, so it exceeds
    # `decisions` whenever the ladder escalates -- that is the cost, made visible
    # rather than hidden.
    ladder_rungs_run: int = 0
    ladder_escalations: int = 0
    #: Decisions that stopped because a WORLD was due to be dropped and the
    #: per-world leaders still DISAGREED. Named for the condition, not for a
    #: verdict: the previous name (`settled_early`) said the opposite of what the
    #: counter measures -- the ladder stops there precisely because the belief is
    #: NOT settled, so keeping the breadth is worth more than concentrating it.
    ladder_worlds_disagree_stops: int = 0
    ladder_decisions: int = 0
    #: Whether this run's cells were DYNAMIC. Emitted because `ladder_decisions > 0`
    #: is not the same question and using it as one broke a consumer: `_search_ladder`
    #: is the dispatch for EVERY model decision, so a FIXED cell also has
    #: `ladder_decisions == decisions` -- while carrying no row stamps, since round 3
    #: correctly made stamping conditional. The deploy analyzer's "is this fixed?"
    #: test was `ladder_decisions == 0`, so on this build it refused every fixed cell,
    #: including all five of the value-gap campaign it is named for. Found in review.
    ladder_dynamic: bool = False
    #: Decisions that stopped because depth WANTED to advance but the current depth
    #: was not saturated. The gate firing is the feature working, not a failure --
    #: but a cell where this dominates never deepened, and its range is decorative.
    ladder_unsaturated_stops: int = 0
    #: Rungs that ended in a fallback. The ladder stops there and keeps the last
    #: SUCCESSFUL rung's decision, so this is also the count of decisions whose
    #: escalation was abandoned.
    ladder_fallback_rungs: int = 0
    #: Addresses dropped because the rung that produced them was superseded. An
    #: override address is a forkable replay handle, so one from a discarded rung
    #: makes the shard claim an override the returned decision did not make.
    #: Non-zero is EXPECTED on a dynamic cell; it is emitted so the drop is never
    #: silent, since `override_disagreements` is also capped and a reader must be
    #: able to tell the two losses apart.
    override_addresses_superseded: int = 0
    #: Ladder decisions where NO rung produced a searched answer, i.e. rung 0 fell
    #: back. These are in `ladder_decisions` but contribute nothing to the override
    #: ledger, which is why the ladder identity needs them as a third term.
    ladder_unsearched_decisions: int = 0
    #: Depth BUMPS taken (a turn saturated, so the next turn tries one ply deeper).
    ladder_depth_rungs: int = 0
    #: Worlds actually dropped (per-world leaders agreed), which is what raises
    #: sims-per-world and clears the depth latch.
    ladder_world_drops: int = 0
    #: SHADOW searches run: a discarded second search at depth+1 whose only product is
    #: the answer to "would this allocation fill one more ply?". Bounded to one per world
    #: count per battle, so this is the honest overhead figure -- each one is an extra
    #: full budget on that turn, and nothing else in the run costs extra.
    ladder_shadow_probes: int = 0
    #: Shadow probes that FELL BACK, i.e. answered nothing. No ceiling is latched on
    #: these, because a latch must record a measurement rather than a failure.
    ladder_shadow_probe_failures: int = 0
    #: Ceilings LATCHED: the shadow said depth+1 would not fill, so this world count
    #: stops probing until worlds drops. This is the count of "the gate discriminated".
    ladder_depth_latches: int = 0
    fold_advanced_lines: int = 0
    fold_cross_checks: int = 0
    fold_cross_check_failures: int = 0
    # Tier-2 overlay telemetry (zero without an annotation source).
    fold_annotations_applied: int = 0
    fold_annotation_boundaries: int = 0
    # Conclusions the fold had already applied and has since PRUNED (they left the
    # identifiable window), re-seated so the cumulative source can keep offering
    # them. Counted, not silent: this is the path that used to break the fold, and
    # a class that stops refusing must not stop being visible.
    fold_annotations_resettled: int = 0
    fold_annotation_resettle_boundaries: int = 0
    # Depth-reached instrumentation (hp_fraction_crate mode). The crate counts
    # the deepest decision node any traversal actually opened; the depth CAP is
    # only a real knob where these numbers sit at it. Recorded per SEARCHED
    # WORLD, so depth_reached_samples counts worlds, not decisions.
    depth_reached_samples: int = 0
    depth_reached_sum: int = 0
    depth_reached_max: int = 0
    depth_reached_histogram: Counter = field(default_factory=Counter)
    #: depth -> traversals that OPENED a decision node there, summed over searched
    #: worlds. The occupancy PROFILE, as distinct from `depth_reached_histogram`, which
    #: is a histogram of per-world MAXIMA and therefore cannot tell a tree filled to
    #: depth 3 from one that sent a single traversal to depth 7. Every saturation
    #: decision in this module was made on the latter; this is the instrument that can
    #: answer it properly, and it makes the saturating depth a READ off one search.
    depth_occupancy: Counter = field(default_factory=Counter)
    # Override telemetry (config.override_telemetry; zero when it is off, and
    # zero on every non-model leaf_eval, which is why the config refuses that
    # combination instead of reporting it as "search never overrides").
    #
    # THE DENOMINATOR IS EXPORTED, not left to be derived. The question these
    # answer is "how often does search play something other than the model's own
    # argmax", and its denominator is NOT `searched_decisions`: a decision whose
    # model argmax could not be determined must not be counted as agreement.
    #   override_measured_decisions + search_override_unmeasured == searched_decisions
    # holds whenever the flag is on, and a consumer that subtracts instead of
    # reading `override_measured_decisions` gets the same number -- the identity
    # is pinned by test rather than assumed.
    #
    # ON A LADDER CELL the identity is
    #   override_measured_decisions + search_override_unmeasured
    #     + ladder_unsearched_decisions == ladder_decisions
    # -- THREE terms, not two. Both counters are charged inside `_search_model`,
    # which a ladder calls once per rung, so an escalating decision would otherwise
    # vote several times and the rate would come out weighted by how far each
    # decision climbed -- and incomparable to a fixed cell's, which is the
    # comparison the override study is entirely about. `_search_ladder` therefore
    # REWINDS the ledger to the winning rung's contribution, so each decision
    # contributes exactly one vote.
    #
    # The third term is the correction review forced: `ladder_decisions` counts a
    # decision that fell back at rung 0, and the override ledger cannot, so the
    # two-term form is simply false and a consumer deriving
    # `unmeasured := ladder_decisions - measured` overcounts by exactly the rung-0
    # fallbacks. Both forms are pinned by test.
    # THE RULE FOR ALL FOUR OVERRIDE SURFACES, stated once because review found the
    # first two rewound and the other two not, with nothing saying which:
    #
    #   * a COUNT or an ADDRESS is a claim about the decision the engine RETURNED,
    #     so it is rewound to the winning rung -- `override_measured_decisions`,
    #     `model_override_decisions`, `search_override_unmeasured` and
    #     `override_disagreements`. An address from a discarded rung is worse than a
    #     miscount: it is forkable, so a probe replays it and finds a decision the
    #     engine did not override.
    #   * a CAUSE TAXONOMY is a claim about what the run encountered, so it is not
    #     rewound -- `search_override_unmeasured_causes`. A cause a discarded rung
    #     hit is still a cause that happened, and it is nobody's denominator.
    #
    # Anything added here must say which of the two it is.
    override_measured_decisions: int = 0
    model_override_decisions: int = 0
    # The honesty half, and the one that matters most: searched decisions where
    # the model's argmax is UNKNOWN (priors off, a root the crate refused to
    # price, a display the request does not name, a stale image). Silently
    # booking these as "no override" is how an override rate reads low for a
    # reason that has nothing to do with search.
    search_override_unmeasured: int = 0
    # WHY it could not be measured. Same argument as `choices_unmapped_causes`:
    # one opaque count cannot be acted on, and the causes above have entirely
    # different owners. Closed token set -- see `_OVERRIDE_UNMEASURED_CAUSES`.
    search_override_unmeasured_causes: Counter = field(default_factory=Counter)
    # ADDRESSES for the disagreements, not just their count. The fork probe
    # (docs/mcts_value_gap_investigation_20260811.md section 4b) has to REPLAY
    # ~50 specific disagreement decisions and play both continuations, which
    # needs (battle_id, round, seat) plus the two action indices; a rate cannot
    # be forked. Same lesson as `fallback_samples`: aggregate counts with no
    # addresses left era 57 unable to reproduce any of the 7,498 events it
    # counted.
    #
    # FIRST-N, deliberately, and biased toward the shard's early battles: the
    # unbiased alternative is reservoir sampling, and the only rng in reach is
    # the decision rng whose draws seed the belief worlds -- consuming from it
    # would CHANGE THE SEARCH. A biased sample of forkable addresses beats an
    # unbiased perturbation of the thing being measured. Overflow is counted,
    # never silent.
    override_disagreements: list[dict[str, Any]] = field(default_factory=list)
    override_disagreement_addresses_dropped: int = 0
    # H2's measurement, absorbed from arms the crate already emitted: how far
    # apart are the ROOT VALUES of the two arms search is choosing between? If
    # those gaps sit inside leaf-eval noise, search has nothing to act on and a
    # deeper tree only sharpens an estimate of "these are the same".
    #
    # ONE denominator for both gaps, because both are computed on exactly the
    # decisions that had two arms with visits -- a decision with a single arm has
    # no gap of either kind, and giving them separate sample counts would invite
    # dividing one by the other's.
    root_arm_gap_samples: int = 0
    root_q_gap_sum: float = 0.0
    root_q_gap_histogram: Counter = field(default_factory=Counter)
    # The visit-share gap over the same two arms. The crate already computes a
    # leader/runner-up visit pair (`early_stop_leader_visits`) unconditionally and
    # emits it -- and it is NOT what is absorbed here, deliberately: that pair is
    # computed for `early_stop_side_one`, a positional whose default is True, so
    # on a p2 decision made through the pre-cascade call it describes the
    # OPPONENT's arms. Deriving the gap from the acting seat's own entries has no
    # seat parameter to get wrong.
    root_visit_gap_sum: float = 0.0
    root_visit_gap_histogram: Counter = field(default_factory=Counter)
    # H4's predictor side, also pure absorption: the in-tree opponent's top arm
    # was in every report and never read. These are the DENOMINATORS for the
    # offline join against the #1188 opponent journal (which holds FoulPlay's
    # actually-submitted moves per round); the arms themselves ride in
    # `root_decision_rows`, since accuracy cannot be computed here -- the policy
    # never sees what FoulPlay played.
    opponent_top_arm_decisions: int = 0
    # Decisions whose opponent seat was priced from the model rather than left
    # uniform. Zero whenever `use_opponent_priors` is off, which is the flag-off
    # twin -- so H4's model-prior leg is measurable only in the flag-on arm, while
    # its tree-visit leg is measurable in both.
    opponent_prior_arm_decisions: int = 0
    # PER-DECISION rows, the only channel by which per-decision search state
    # reaches a shard: the bridge aggregates decision metadata into counts and
    # keeps no per-round copy of it. Capped, because a shard summary's size must
    # not be set by however many decisions a run happens to make; overflow is
    # counted, never silent.
    #
    # ON A LADDER CELL THERE IS ONE ROW PER RUNG, NOT PER DECISION. `_search_ladder`
    # stamps each with `ladder_rung` and `ladder_superseded`; a per-decision
    # statistic (the top-1/top-2 Q gap distribution, the H4 opponent-arm join) MUST
    # filter `not row["ladder_superseded"]` or it is rung-weighted -- the same
    # defect as the override rate, on a different surface. A cost analysis wants the
    # unfiltered list. Fixed cells carry neither field.
    root_decision_rows: list[dict[str, Any]] = field(default_factory=list)
    root_decision_rows_dropped: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "decisions": self.decisions,
            "searched_decisions": self.searched_decisions,
            "fallback_decisions": self.fallback_decisions,
            "oracle_belief_decisions": self.oracle_belief_decisions,
            # GROSS, and correctly so under the per-turn design: with ONE PLAYED
            # search per decision a decision cannot fall back on a later rung after an
            # earlier one succeeded, so there is nothing to net out.
            # `ladder_recovered_fallbacks` existed only for the rung model and went
            # with it. A SHADOW probe that falls back does not touch this either -- it
            # is charged to `ladder_shadow_probe_failures`, because no played move
            # depended on it.
            "fallback_rate": self.fallback_decisions / self.decisions if self.decisions else 0.0,
            "override_addresses_superseded": self.override_addresses_superseded,
            "ladder_unsearched_decisions": self.ladder_unsearched_decisions,
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
            # Caveat on the numerator: `world_search_attempts` is charged before
            # `_search_model`'s own pre-flight, so a `root_inputs_failed` or
            # `early_stop_replay_failed` fallback charges W phantom aborts here.
            # Both are visible in `fallback_reasons`; check there before reading
            # a near-1.0 abort rate as a renderer-refusal problem.
            #
            # RUNGS PER DECISION is the ladder's cost multiplier, and the number to
            # read before any other ladder rate: `searched_decisions` counts RUNGS,
            # not decisions, so every rate built on it is per-rung on a ladder cell.
            # Reading one of those as per-decision once turned a measured 2x cost
            # REGRESSION into a reported 23% saving.
            "ladder_rungs_per_decision": (
                self.ladder_rungs_run / self.ladder_decisions
                if self.ladder_decisions
                else None
            ),
            "world_search_attempts": self.world_search_attempts,
            # Against SEARCH ATTEMPTS, not constructions: on a ladder cell the same
            # constructed world is searched once per rung, so the construction
            # denominator undercounts and drove this negative. Found in review.
            "world_search_abort_rate": (
                1.0 - (self.worlds_searched / self.world_search_attempts)
                if self.world_search_attempts
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
            "worlds_collapsed": self.worlds_collapsed,
            "unique_worlds_searched": self.unique_worlds_searched,
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
            "collision_rounds": self.collision_rounds,
            "collision_pending_rounds": self.collision_pending_rounds,
            "collision_selections": self.collision_selections,
            "collision_joint_repeats": self.collision_joint_repeats,
            "collision_self_repeats": self.collision_self_repeats,
            "collision_opponent_repeats": self.collision_opponent_repeats,
            "collision_traversals": self.collision_traversals,
            "collision_leaf_repeats": self.collision_leaf_repeats,
            "early_stop_triggered_worlds": self.early_stop_triggered_worlds,
            "early_stop_accepted_decisions": self.early_stop_accepted_decisions,
            "early_stop_full_budget_replays": self.early_stop_full_budget_replays,
            "early_stop_sims_saved": self.early_stop_sims_saved,
            "ladder_rungs_run": self.ladder_rungs_run,
            "ladder_escalations": self.ladder_escalations,
            "ladder_worlds_disagree_stops": self.ladder_worlds_disagree_stops,
            "ladder_decisions": self.ladder_decisions,
            # The predicate a consumer must branch on. NOT `ladder_decisions > 0`.
            "ladder_dynamic": self.ladder_dynamic,
            "ladder_unsaturated_stops": self.ladder_unsaturated_stops,
            "ladder_fallback_rungs": self.ladder_fallback_rungs,
            "ladder_depth_rungs": self.ladder_depth_rungs,
            "ladder_world_drops": self.ladder_world_drops,
            "ladder_shadow_probes": self.ladder_shadow_probes,
            "ladder_shadow_probe_failures": self.ladder_shadow_probe_failures,
            "ladder_depth_latches": self.ladder_depth_latches,
            "fold_advanced_lines": self.fold_advanced_lines,
            "fold_cross_checks": self.fold_cross_checks,
            "fold_cross_check_failures": self.fold_cross_check_failures,
            "fold_annotations_applied": self.fold_annotations_applied,
            "fold_annotation_boundaries": self.fold_annotation_boundaries,
            "fold_annotations_resettled": self.fold_annotations_resettled,
            "fold_annotation_resettle_boundaries": self.fold_annotation_resettle_boundaries,
            "depth_reached_samples": self.depth_reached_samples,
            "depth_reached_max": self.depth_reached_max,
            "depth_occupancy": {
                str(depth): count for depth, count in sorted(self.depth_occupancy.items())
            },
            "depth_reached_histogram": {
                str(depth): count
                for depth, count in sorted(self.depth_reached_histogram.items())
            },
            "override_measured_decisions": self.override_measured_decisions,
            "model_override_decisions": self.model_override_decisions,
            "search_override_unmeasured": self.search_override_unmeasured,
            "search_override_unmeasured_causes": dict(
                self.search_override_unmeasured_causes
            ),
            "override_disagreements": [dict(row) for row in self.override_disagreements],
            "override_disagreement_addresses_dropped": (
                self.override_disagreement_addresses_dropped
            ),
            "root_arm_gap_samples": self.root_arm_gap_samples,
            "root_q_gap_sum": self.root_q_gap_sum,
            "root_q_gap_histogram": dict(sorted(self.root_q_gap_histogram.items())),
            "root_visit_gap_sum": self.root_visit_gap_sum,
            "root_visit_gap_histogram": dict(
                sorted(self.root_visit_gap_histogram.items())
            ),
            "opponent_top_arm_decisions": self.opponent_top_arm_decisions,
            "opponent_prior_arm_decisions": self.opponent_prior_arm_decisions,
            "root_decision_rows": [dict(row) for row in self.root_decision_rows],
            "root_decision_rows_dropped": self.root_decision_rows_dropped,
        }
        if self.depth_reached_samples:
            payload["depth_reached_mean"] = (
                self.depth_reached_sum / self.depth_reached_samples
            )
        if self.collision_selections:
            # Per-selection repeat rates. The self/opponent pair is the whole
            # point: the deferred-leaf theory says the placeholder is
            # side-asymmetric, so it predicts these two SEPARATE, and only their
            # difference is evidence -- the joint rate pools them and cannot see
            # it. Read them against `collision_pending_rounds /
            # collision_rounds`, which says what share of rounds had a
            # placeholder in them at all.
            payload["collision_joint_rate"] = (
                self.collision_joint_repeats / self.collision_selections
            )
            payload["collision_self_rate"] = (
                self.collision_self_repeats / self.collision_selections
            )
            payload["collision_opponent_rate"] = (
                self.collision_opponent_repeats / self.collision_selections
            )
        if self.collision_traversals:
            payload["collision_leaf_rate"] = (
                self.collision_leaf_repeats / self.collision_traversals
            )
        if self.root_arm_gap_samples:
            payload["root_q_gap_mean"] = (
                self.root_q_gap_sum / self.root_arm_gap_samples
            )
            payload["root_visit_gap_mean"] = (
                self.root_visit_gap_sum / self.root_arm_gap_samples
            )
        if self.override_measured_decisions:
            # Only on its OWN denominator, and only when that denominator exists.
            # `model_override_decisions / searched_decisions` is the wrong number
            # -- it dilutes by however many decisions were unmeasurable -- and a
            # 0.0 emitted with the flag off is a claim ("search never overrides")
            # rather than an absence.
            payload["model_override_rate"] = (
                self.model_override_decisions / self.override_measured_decisions
            )
        if self.searched_decisions:
            # PER RUNG on a ladder cell: `searched_decisions` is charged once per
            # `_search_model` call and the ladder calls it once per rung. The
            # per-DECISION pair below is the one to compare across cells.
            payload["iterations_per_searched_decision"] = (
                self.total_iterations / self.searched_decisions
            )
            payload["search_wall_per_searched_decision"] = (
                self.search_wall_seconds / self.searched_decisions
            )
        # COVERAGE, not "at least one rung searched". `ladder_decisions` is charged
        # BEFORE the first rung runs, so a cell whose decisions nearly all fell back
        # at rung 0 still had a non-zero denominator and emitted a wall -- the
        # FALLBACK wall -- which the power report's cap read as a PASS. Gating on
        # `searched_decisions > 0` fixed the 100% case and left the 99.99% one:
        # review measured one searched rung against 10,000 decisions reporting 0.3 ms
        # and passing a 20 s cap. The requirement is that MOST decisions the engine
        # was asked to make actually reached a search.
        _searchable = self.ladder_decisions - self.ladder_unsearched_decisions
        if (
            self.ladder_decisions
            and self.searched_decisions
            and _searchable >= 0.9 * self.ladder_decisions
        ):
            #
            # The COST DENOMINATOR for any cross-cell comparison. A ladder decision
            # may run several rungs, so these are the only ladder rates that mean
            # "per decision the engine was asked to make".
            payload["iterations_per_ladder_decision"] = (
                self.total_iterations / self.ladder_decisions
            )
            payload["search_wall_per_ladder_decision"] = (
                self.search_wall_seconds / self.ladder_decisions
            )
        if self.decisions:
            payload["wall_per_decision"] = self.decision_wall_seconds / self.decisions
        return payload


def free_decision_features(context, sims_per_world, prior_share) -> dict[str, Any]:
    """The features a production depth rule may key on: FREE at decision time.

    "Free" is the whole discipline. Every value here is knowable BEFORE the search runs
    -- from the request's legal-action mask, from the root prior forward pass we already
    do, or from the replay's turn counter -- so a rule fitted on them can actually be
    evaluated in production. Anything that requires searching first (occupancy, the
    top-1 visit share, the Q gap, whether search overrode the model) is a LABEL and is
    recorded elsewhere on the row; mixing the two produces a rule that cannot be run.

    Returned flat and prefixed `f_` so a consumer can select the input columns without
    knowing the schema.
    """
    # The candidates live on `observation.metadata["action_candidates"]`, NOT on an
    # `observation.candidates` attribute -- `PokeZeroObservationV0` has no such field, so
    # reading it returned None and made every decision look FORCED with zero legal
    # actions. That shipped: the first collection shard carried f_legal_actions == 0 on
    # all 29 rows while `top_arms` listed real moves. The unit test did not catch it
    # because its fixture fabricated the attribute the code expected instead of the shape
    # `showdown.py` publishes.
    #
    # `action_candidates` always has 9 entries -- 4 move slots then 5 switch slots --
    # including illegal ones, so BOTH the `legal` flag and the mask bit are required.
    # Admission mirrors `_choice_vocabulary`, deliberately: a feature that counted a
    # different action set than the search chooses from would key the table on a
    # branching factor the search never faced.
    obs = getattr(context, "observation", None)
    mask = getattr(obs, "legal_action_mask", None)
    candidates = (getattr(obs, "metadata", None) or {}).get("action_candidates")
    readable = (isinstance(candidates, Sequence)
                and not isinstance(candidates, (str, bytes))
                and bool(candidates))
    moves = switches = 0
    if readable:
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or not candidate.get("legal"):
                continue
            index = candidate.get("action_index")
            if mask is not None:
                if not isinstance(index, int) or not (0 <= index < len(mask)):
                    continue
                if not mask[index]:
                    continue
            kind = candidate.get("kind")
            if kind == "move":
                moves += 1
            elif kind == "switch":
                switches += 1
    replay = getattr(
        getattr(context, "public_materialization_state", None), "replay", None
    )
    ordered = sorted((float(v) for v in (prior_share or {}).values()), reverse=True)
    total = sum(ordered)
    entropy = None
    if total > 0:
        entropy = -sum(
            (v / total) * math.log(v / total) for v in ordered if v > 0
        )
    return {
        # Branching, split because a switch changes the active mon and a move does not,
        # so they do not cost the same in tree width.
        # None, never 0, when the candidate list could not be read. A count of 0 is a
        # CLAIM -- and via `f_forced` below it is the strongest one available, "there was
        # nothing to decide" -- so manufacturing it from a failed read is how the
        # metadata-path defect stayed invisible for a whole collection run. An absence
        # propagates as an absence, and the offline fitter reports it as missing data.
        "f_legal_moves": moves if readable else None,
        "f_legal_switches": switches if readable else None,
        "f_legal_actions": (moves + switches) if readable else None,
        # FORCED: nothing to decide, so no depth is worth buying. Measured at 1.6% of
        # decisions overall and 3.2% past turn 30 -- a late-game phenomenon.
        "f_forced": ((moves + switches) <= 1) if readable else None,
        # The model's own confidence, from the forward pass already done. This is the
        # cheapest predictor of "was this decision ever going to be close".
        "f_root_prior_top1": ordered[0] / total if total > 0 else None,
        "f_root_prior_entropy": entropy,
        "f_turn": int(getattr(replay, "turn_number", 0) or 0),
        # The allocation itself: the table is indexed on this.
        "f_sims_per_world": sims_per_world,
    }


def native_search_args(
    config,
    record: Mapping[str, Any],
    *,
    tables_json,
    root_inputs,
    rust_fold,
    early_stop_min_sims: int,
    sims: int | None = None,
    depth: int | None = None,
) -> list:
    """The positional argument list for `search_batched_multi_encoded`.

    A module-level function, not an inline block, for ONE reason: the call
    contract is gated by tests/test_opponent_priors_flag.py, and until this was
    extracted those tests rebuilt this list themselves and asserted against
    their own copy. Measured in review: deleting `config.model_priors` from the
    real assembly here -- which disables model priors AND shifts
    `use_opponent_priors` two slots left into `early_stop_min_sims`, silently
    truncating the search budget, exactly the catastrophe those tests claim to
    guard -- left them at "6 passed, 1 skipped". They now call this.

    The ordering rules are load-bearing:

    * the first twelve positionals are the pre-flag contract, made byte for
      byte whenever the flag is off, so a stale image cannot be broken merely
      by updating Python;
    * `use_opponent_priors` FOLLOWS the early-stop pair in the native
      signature, so turning it on must also materialize that pair -- otherwise
      `True` lands in `early_stop_min_sims`.
    * `fpu_reduction` follows `use_opponent_priors` for the same reason one slot
      further out: setting it must materialize BOTH the early-stop pair and the
      opponent flag, or the float lands in `use_opponent_priors` and turns the
      opponent head on by accident while the FPU stays off. The cascade below is
      written as three widening conditions, not three independent ones, so that
      cannot happen.
    * `override_telemetry` follows `fpu_reduction`, one slot further out again.
      Asking for the arm names must materialize all three earlier slots, or a
      `True` lands in `fpu_reduction` -- which the crate's validator ACCEPTS
      (`1.0` is in range) and which would change selection, turning a pure
      telemetry flag into a search change. The widest condition, therefore, and
      first.

    `depth` overrides `config.search_depth` for ONE call: the dynamic budget
    ladder searches a decision at `depth_min` first and escalates toward the cap,
    so one decision issues several calls differing only in this positional.
    `None` means "use the configured cap", which keeps the list byte-identical
    for every fixed-budget caller.

    `sims` overrides `config.search_sims` for ONE call: #1009 concentrates
    duplicate belief worlds into a single deeper search, so a collapsed record
    is searched at multiplicity x the per-world budget. `None` means "use the
    configured budget", which is every uncollapsed caller.
    """
    search_args = [
        record["state_str"],
        config.search_sims if sims is None else sims,
        config.search_batch,
        tables_json,
        root_inputs,
        record["ctx_json"],
        rust_fold,
        config.search_depth if depth is None else depth,
        config.c_puct,
        record["seed"],
        config.deep_ko_split,
        config.model_priors,
    ]
    fpu_reduction = getattr(config, "fpu_reduction", None)
    override_telemetry = bool(getattr(config, "override_telemetry", False))
    if (
        early_stop_min_sims
        or config.use_opponent_priors
        or fpu_reduction is not None
        or override_telemetry
    ):
        search_args.extend([early_stop_min_sims, record["side_key"] == "side_one"])
    if config.use_opponent_priors or fpu_reduction is not None or override_telemetry:
        search_args.append(bool(config.use_opponent_priors))
    if fpu_reduction is not None or override_telemetry:
        # `None` is the crate's own default for this slot, so materializing it to
        # reach the slot behind it changes nothing -- unlike the two booleans
        # above, whose default is False and whose materialized value is the
        # config's.
        search_args.append(None if fpu_reduction is None else float(fpu_reduction))
    if override_telemetry:
        search_args.append(True)
    return search_args


#: EVERY counter that is a CLAIM ABOUT ONE DECISION rather than a measure of work
#: done. `_search_ladder` rewinds all of them to the winning rung, because
#: `_search_model` charges them once per RUNG and an escalating decision would
#: otherwise vote once per rung it happened to climb.
#:
#: This list exists because review found the same defect FOUR times on four
#: surfaces -- the override counters, the per-decision rows, the override addresses,
#: and then these -- each round fixing one more. Enumerating the class is the only
#: fix that generalises: a new per-decision counter is added HERE or the sibling
#: test refuses it.
#:
#: What is deliberately NOT here, and why:
#:   * `model_evals`, `total_iterations`, every `*_wall_seconds`, `worlds_searched`,
#:     `world_search_attempts`, the `collision_*` family -- measures of WORK. A rung
#:     really did that work; summing it across rungs is what a cost analysis wants
#:     and `ladder_rungs_per_decision` is how a reader divides it.
#:   * `depth_reached_*` -- work too, and per WORLD rather than per decision, so
#:     rewinding them would break the histogram's relationship to
#:     `total_iterations`. BUT A STATED LIMITATION COMES WITH THEM: on a dynamic cell
#:     `depth_reached_mean` is a mixture over rungs run at DIFFERENT depth caps, so
#:     it is not "how deep did search get" for a decision -- the early, shallow,
#:     most-numerous rungs dominate it. Read it against
#:     `ladder_rungs_per_decision`, and use `ladder_depth_rungs` against
#:     `ladder_unsaturated_stops` to ask whether depth actually advanced. Raised in
#:     review; recorded rather than silently inherited.
#:   * `search_override_unmeasured_causes`, `world_failure_reasons`,
#:     `fallback_reasons` -- CAUSE TAXONOMIES. A cause a discarded rung hit is still
#:     a cause the run encountered, and none is anyone's denominator.
#:   * `fallback_decisions` -- gross, and nothing to net: one PLAYED search per
#:     decision. The SHADOW is the only second search a decision can contain, and it
#:     the SHADOW's contribution out (see `_search_ladder`), which is the only
#:     second search a decision can now contain.
LADDER_PER_DECISION_CLAIMS = (
    "override_measured_decisions",
    "model_override_decisions",
    "search_override_unmeasured",
    # H2's headline. `root_q_gap_mean` is quoted as "the gap between the two arms
    # search is choosing between", i.e. per decision -- so a decision that climbed
    # three rungs contributed three gaps to it, from three different budgets.
    "root_arm_gap_samples",
    "root_q_gap_sum",
    "root_visit_gap_sum",
    # H4's predictor side, and both are named `_decisions`.
    "opponent_top_arm_decisions",
    "opponent_prior_arm_decisions",
    # "decisions where a stop was accepted" -- one per decision, not one per rung.
    "early_stop_accepted_decisions",
)

#: The same class, for claims held in a Counter rather than a number. They need a
#: SEPARATE mechanism and that is the whole reason this tuple exists: the generic
#: rewind snapshots with `getattr`, which for a mutable returns a REFERENCE, so
#: `now - before` compares the object with itself and the rewind is a silent no-op.
#: Review demonstrated it -- appending a histogram to the scalar tuple above raised
#: no error, changed nothing, and passed every test.
#:
#: These are the SIXTH surface of the same defect. `root_q_gap_histogram` is
#: incremented on the same line block as `root_q_gap_sum`, from the same datum, once
#: per rung -- so rewinding the sum and not the histogram left
#: `sum(histogram.values()) == root_arm_gap_samples` false, and the published H2 mean
#: per-decision while the quartiles read off the histogram stayed per-rung. Two
#: surfaces disagreeing is the failure this module names in its own comments.
LADDER_PER_DECISION_CLAIM_HISTOGRAMS = (
    "root_q_gap_histogram",
    "root_visit_gap_histogram",
)

#: Hard floor for a dynamic ladder's depth. Depth 1 is a one-ply search, which is
#: no better than the raw policy -- so the cheapest rung, the one most decisions
#: never leave, must still be a real search. Owner-set.
LADDER_MIN_DEPTH_FLOOR = 2

#: The id Showdown gives the recharge pseudo-move it substitutes for a locked mon's moveset.
_RECHARGE_REQUEST_MOVE_ID = "recharge"

#: The id Showdown gives the Struggle pseudo-move it substitutes when a mon's request would
#: otherwise offer NO move at all (``sim/pokemon.ts`` ``getMoveRequestData``, the
#: ``else if (!moves.length)`` arm). The engine has no Struggle arm to enumerate, so the same
#: state reaches ``_map_choices`` as the ``MoveChoice::None`` display -- see the translation there.
_STRUGGLE_REQUEST_MOVE_ID = "struggle"


def self_recharge_from_action_candidates(observation_metadata: Any) -> bool:
    """Whether the request's LEGAL CHOICE SET is exactly the recharge pseudo-move.

    Showdown's ``Pokemon.getMoveRequestData`` sets ``this.trapped = true`` whenever
    ``getLockedMove()`` returns anything, and ``getMoves(lockedMove)`` short-circuits to the
    single synthetic entry ``[{move: 'Recharge', id: 'recharge'}]`` when and only when that
    locked move is ``recharge`` (``sim/pokemon.ts:968``, ``:1084-1088``). ``mustrecharge`` is
    gen3's only ``onLockMove: 'recharge'``, so a request offering nothing but ``recharge`` is
    not evidence about the lock, it IS the lock, disclosed to the seat that has to act on it.

    READ FROM ``action_candidates``, NOT THE RAW REQUEST. Three reasons, in order of weight:

    1. ``action_candidates`` is published by ``_observation_metadata`` UNCONDITIONALLY, on every
       schema, unlike the v4-gated ``self_must_recharge`` this backstops. The raw request would
       also work, but only for callers holding a ``public_materialization_state``.
    2. It is what a RECORDED ROW carries. ``scripts/fidelity_gate_events.py``'s
       ``production_recharging_slots`` exists to be "``recharging_slots`` as production builds
       it", and it is handed observation metadata, not a ``PolicyContext``. A request-based rule
       would be UNMIRRORABLE there -- the payload's ``selfActiveMoves`` drops the synthetic entry
       (it carries no ``pp``/``maxpp``, so ``_request_active_moves`` filters it out), so the
       corpus row has no request-side trace of the lock at all. The gate would then seed worlds
       without MUSTRECHARGE while production seeds them with it, and stop measuring this change.
    3. It covers contexts that carry metadata but no materialization state -- ``fallback_replay``
       records and cached rollouts.

    This is the same fold ``fallback_replay._request_legal_choices`` applies to produce the
    ``request offered: recharge`` line in the refusal records that diagnosed this bug; the
    candidate's ``legal`` flag is ``bool(state.legal_action_mask[action_index])`` at the source
    (``showdown.py:7530``), so no separate mask read is needed.

    Deliberately narrow: the legal set must be EXACTLY one entry, a move, spelled ``recharge``.
    An Encore lock, a Choice lock and a mid-charge Solar Beam all present one legal move too,
    but under their real move id, and a partly-disabled moveset presents several. Seeding
    MUSTRECHARGE for any of those would model a mon that cannot act when it can -- silently
    wrong, which is worse than the refusal this removes.
    """

    if not isinstance(observation_metadata, Mapping):
        return False
    candidates = observation_metadata.get("action_candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return False
    legal = [
        candidate
        for candidate in candidates
        if isinstance(candidate, Mapping) and candidate.get("legal")
    ]
    if len(legal) != 1:
        return False
    only = legal[0]
    if only.get("kind") != "move":
        return False
    return normalize_id(str(only.get("move_id") or "")) == _RECHARGE_REQUEST_MOVE_ID


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
    permutes opponent switch priors, and nothing downstream can detect it.

    Returning None is now a REFUSAL the crate honours, not a hand-off to a
    fallback. It used to be the latter: `leaf.rs::root_opponent_order`
    substituted a one-swap approximation whenever this returned None, so the
    refusal became a confident wrong answer one layer down -- fail-closed here,
    fail-OPEN there. The crate refuses too as of #1194: the opponent action map
    is all-`None`, the node keeps uniform priors, and the refusal is counted in
    `prior_fallbacks`. Do not reintroduce a caller-side fallback on the
    strength of this docstring; there is no longer anything to degrade to, and
    that is deliberate.
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


def _leading_choice(weights: Mapping[str, float]) -> Optional[str]:
    """First strict maximum in insertion order, or None on an empty mapping.

    The SAME rule `_map_choices` applies to the visit aggregate. Every weighting
    compared anywhere in this file goes through it, so a tie resolves on the same
    side for all of them: a tie-break that differed between the model's priors
    and the search's visits would manufacture an override out of a tie, which is
    exactly how an override rate acquires a floor it did not earn.
    """
    best: Optional[str] = None
    best_weight = 0.0
    for choice, weight in weights.items():
        if weight > best_weight:
            best_weight = weight
            best = choice
    return best


def _leading_pair(weights: Mapping[str, float]) -> list[str]:
    """The top two choices by weight, insertion order breaking ties, 0-2 long."""
    ordered = sorted(
        enumerate(weights.items()), key=lambda pair: (-pair[1][1], pair[0])
    )
    return [choice for _position, (choice, weight) in ordered[:2] if weight > 0.0]


def _gap_bucket(gap: float) -> str:
    """A 0.01-wide bucket label for a gap in [0, 1].

    Bucketed rather than raw, for the reason `depth_reached_histogram` is: a
    per-decision value keyed raw would mint a key per decision and the block
    would grow with the run. Two decimals is finer than any leaf-eval noise floor
    H2 could plausibly compare against.
    """
    return f"{max(0.0, min(1.0, gap)):.2f}"


@dataclass(frozen=True)
class _RootArmAggregate:
    """The root's arms, pooled across a decision's searched worlds.

    Pooled the SAME way the visit aggregate that picks the action is: each world
    normalized, then summed over RECORDS, so a duplicated belief completion
    weighs its multiplicity on every quantity here. Anything else would compare
    differently-pooled numbers -- a per-world vote, or one world's arms, answers
    a question about a world rather than about the decision that was played.
    """

    #: Acting seat, summed per-world visit shares (totals to the record count).
    visit_share: Counter
    #: Acting seat, visit-share-weighted mean Q per arm, in the ACTING seat's
    #: frame. See `_aggregate_root_arms` on the frame flip.
    arm_q: dict[str, float]
    #: Acting seat, summed per-world prior shares. Empty when unmeasurable.
    prior_share: Counter
    #: Opponent seat, the same two shares. The opponent's arms are in the report
    #: and were never absorbed; they are H4's whole predictor side.
    opponent_visit_share: Counter
    opponent_prior_share: Counter
    #: Why the acting seat's prior aggregate cannot be used, or None.
    prior_cause: Optional[str]


def _aggregate_root_arms(world_runs: Sequence[Mapping[str, Any]]) -> _RootArmAggregate:
    """Pool both seats' root arms over a decision's world reports.

    PURE. It moves no counter, which is what lets the override probe run on every
    searched decision without disturbing `unmapped_choices` /
    `choices_unmapped_causes` -- campaign stop-condition terms that the early-stop
    path's probe call already inflates.

    MEASURABILITY IS NOT INFERRED FROM THE PRIOR VALUES on the acting seat. When
    the crate's root prior path falls back, `MoveStats::prior` keeps the uniform
    `1/n` it was constructed with, which in a report is indistinguishable from a
    model prior that happens to be flat -- so reading the arms alone would return
    a confident argmax (the first arm, on the tie-break) for a decision where the
    model expressed nothing, and the honesty counter would read zero while the
    whole measurement was fiction. `root_priors` is `null` exactly when the path
    did not resolve, so it is the authority; the arms only supply the pairing the
    visit-sorted entries destroy, and the two are CROSS-CHECKED as multisets so a
    pairing bug cannot pass as a value.

    The OPPONENT seat has no such authority -- the crate exports no
    `root_opponent_priors` -- so uniformity is the only available signal there and
    it is used as one: all-equal arms are refused rather than argmaxed.
    """
    visit_share: Counter = Counter()
    prior_share: Counter = Counter()
    opponent_visit_share: Counter = Counter()
    opponent_prior_share: Counter = Counter()
    q_weight: Counter = Counter()
    q_weighted: Counter = Counter()
    worlds_without_priors = 0
    arms_absent = 0
    arms_misaligned = 0
    for record in world_runs:
        report = record["report"]
        side_key = record["side_key"]
        acting_side_one = side_key == "side_one"
        entries = report.get(side_key) or []
        total = max(sum(entry["visits"] for entry in entries), 1)
        for entry in entries:
            choice = entry["move"]
            share = entry["visits"] / total
            visit_share[choice] += share
            # THE FRAME FLIP. `stats_to_json` prints `MoveStats::mean()` raw for
            # both seats, and `finalize` accumulates the side-ONE-absolute
            # expectation into both stat vectors (side two's virtual loss is
            # replaced with `expectation - 1.0`, netting the same sum) -- the
            # reflection lives in `puct`, at selection time, not in storage. So a
            # p2 decision's arms come out in the opponent's frame, and a Q gap
            # pooled across seats without this flip would average a win
            # probability against a loss probability.
            q_acting = entry["q"] if acting_side_one else 1.0 - entry["q"]
            q_weight[choice] += share
            q_weighted[choice] += share * q_acting
        opponent_entries = report.get(
            "side_two" if acting_side_one else "side_one"
        ) or []
        opponent_total = max(sum(entry["visits"] for entry in opponent_entries), 1)
        for entry in opponent_entries:
            opponent_visit_share[entry["move"]] += entry["visits"] / opponent_total
        root_priors = report.get("root_priors")
        if root_priors is None:
            worlds_without_priors += 1
            continue
        arm_priors = [entry.get("prior") for entry in entries]
        if any(prior is None for prior in arm_priors):
            arms_absent += 1
            continue
        if sorted(f"{prior:.6f}" for prior in arm_priors) != sorted(
            f"{prior:.6f}" for prior in root_priors
        ):
            # Values that do not answer to the authority. Defensive -- the crate
            # writes both off one vector -- but pairing them anyway would file one
            # arm's prior under another arm's name, which is a WRONG argmax rather
            # than a missing one.
            arms_misaligned += 1
            continue
        prior_total = sum(arm_priors) or 1.0
        for entry, prior in zip(entries, arm_priors):
            prior_share[entry["move"]] += prior / prior_total
        if len(opponent_entries) >= 2:
            opponent_arm_priors = [entry.get("prior") for entry in opponent_entries]
            if not any(prior is None for prior in opponent_arm_priors) and (
                len(set(f"{prior:.6f}" for prior in opponent_arm_priors)) > 1
            ):
                opponent_prior_total = sum(opponent_arm_priors) or 1.0
                for entry, prior in zip(opponent_entries, opponent_arm_priors):
                    opponent_prior_share[entry["move"]] += prior / opponent_prior_total
    # PRECEDENCE, from "the instrument is not installed" outwards to "this
    # decision resisted measurement". A stale image reports `prior_arms_absent`
    # on every decision, and that diagnosis must not hide behind a partial-worlds
    # count that is only its symptom.
    prior_cause: Optional[str] = None
    if arms_absent:
        prior_cause = _OVERRIDE_UNMEASURED_ARMS_ABSENT
    elif arms_misaligned:
        prior_cause = _OVERRIDE_UNMEASURED_ARMS_MISALIGNED
    elif worlds_without_priors and worlds_without_priors == len(world_runs):
        prior_cause = _OVERRIDE_UNMEASURED_NO_PRIORS
    elif worlds_without_priors:
        prior_cause = _OVERRIDE_UNMEASURED_PARTIAL_WORLDS
    elif not prior_share:
        # No worlds at all, or arms with no prior mass. Named rather than left to
        # produce an empty argmax.
        prior_cause = _OVERRIDE_UNMEASURED_NO_PRIORS
    return _RootArmAggregate(
        visit_share=visit_share,
        arm_q={
            choice: q_weighted[choice] / weight
            for choice, weight in q_weight.items()
            if weight > 0.0
        },
        prior_share=Counter() if prior_cause is not None else prior_share,
        opponent_visit_share=opponent_visit_share,
        opponent_prior_share=opponent_prior_share,
        prior_cause=prior_cause,
    )


@dataclass(frozen=True)
class _ChoiceVocabulary:
    """One decision's request action space, keyed by the engine's display names.

    Built once per decision by `EngineMctsPolicy._choice_vocabulary`; `action_index`
    is PURE, so a caller that is only measuring (the override telemetry) can
    translate a display without moving a counter that a stop condition reads.
    """

    move_index_by_id: dict[str, int]
    hidden_power_index: Optional[int]
    switch_index_by_species: dict[str, int]
    switch_index_by_canonical: dict[str, int]
    forced_struggle_index: Optional[int]
    # Recorded for `_classify_unmapped`, which has to tell "the engine proposed a
    # move and NO move was legal" from "a DIFFERENT move was legal".
    any_legal_move: bool
    any_legal_switch: bool

    def action_index(self, choice: str) -> Optional[int]:
        """The request action index an engine display names, or None."""
        index: Optional[int] = None
        if choice.startswith("switch "):
            species = normalize_id(choice[len("switch "):])
            index = self.switch_index_by_species.get(species)
            if index is None:
                index = self.switch_index_by_canonical.get(
                    canonical_gen3_randbat_species_id(species)
                )
        else:
            move_id = normalize_id(choice)
            index = self.move_index_by_id.get(move_id)
            if index is None and move_id.startswith("hiddenpower"):
                # Engine ids are typed+BP; the request reports plain "hiddenpower".
                index = self.hidden_power_index
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
                index = self.move_index_by_id.get(_RECHARGE_REQUEST_MOVE_ID)
                if index is None:
                    # SECOND vocabulary gap behind the SAME engine token: Struggle. The engine
                    # has no Struggle arm to enumerate -- `MoveChoice` is Move/Switch/None and
                    # gen3 `get_all_options` never synthesizes one -- so when
                    # `add_available_moves` adds nothing (every slot at 0 PP or disabled) and
                    # `add_switches` adds nothing (no live bench, or trapped), the terminal
                    # `if options.len() == 0 { push(MoveChoice::None) }` guard fires and the
                    # crate renders "No Move". Showdown, at the same state, substitutes the
                    # Struggle pseudo-move (`sim/pokemon.ts` `getMoveRequestData`: `else if
                    # (!moves.length) moves = [{ move: 'Struggle', id: 'struggle' }]`). One
                    # forced action, two names -- the recharge case one paragraph up, again.
                    #
                    # `forced_struggle_index`, NOT a bare `move_index_by_id.get("struggle")`:
                    # the admission test above is what distinguishes the SUBSTITUTED
                    # pseudo-move from a real Struggle move slot, and what keeps the
                    # engine-vs-request switch disagreement countable. Read it there.
                    #
                    # Recharge FIRST and Struggle only as the fallthrough, but the order is
                    # documentation, not disambiguation: the two can never both be offered.
                    # `getMoveRequestData` reaches the Struggle substitution ONLY when
                    # `getMoves` returned an EMPTY list, and a `recharge` lock makes `getMoves`
                    # return the one-element `[{Recharge}]` -- non-empty, so the substitution
                    # is unreachable on a recharge turn. Offering NEITHER is the ordinary turn,
                    # and there both lookups miss and the choice stays unmapped, which is
                    # correct: "No Move" against a request with real moves is a genuine
                    # engine/request disagreement, not a naming one.
                    index = self.forced_struggle_index
        return index


class EngineMctsPolicy:
    """ContextAwarePolicy running poke-engine MCTS over belief-sampled worlds."""

    # Declares the requirement the FoulPlay bridge gates on. Engine search
    # cannot run without a materialized public state; without this the bridge
    # passes None and every decision degrades to uniform-legal fallback.
    requires_public_materialization_state = True

    #: CLASS-LEVEL DEFAULT, load-bearing. `tests/test_engine_search.py::_policy()`
    #: builds this class through `object.__new__` to enter `_search_model`
    #: directly, so `__init__` never runs and an instance attribute alone leaves
    #: the construction loop raising `AttributeError` on a purely observational
    #: hook. Measured: the full tree caught exactly that in
    #: `WorldAbortRateTests.test_the_increment_is_reached_on_the_model_path_the_
    #: campaign_runs`. An instrument that crashes the search it was only supposed
    #: to watch is worse than no instrument.
    _world_observer: Any | None = None

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
        world_observer: Any | None = None,
    ) -> None:
        if module is None:
            import poke_engine as module  # noqa: PLC0415 — optional native dependency

        self.policy_id = policy_id
        # Test/scenario hook: bypass belief sampling and use this override as
        # every world (custom-game sweeps where the catalog cannot sample).
        self._fixed_override = fixed_override
        # Ladder scratch state, owned by `_search_ladder` and read by
        # `_search_model`. Instance attributes rather than parameters so
        # `_search_model`'s signature -- which several tests and the fallback
        # taxonomy key on -- does not move.
        # Ladder scratch state, owned by `_search_ladder` and read by
        # `_search_model`. ALL of it declared here rather than relying on getattr
        # defaults -- review found the comment claiming that while three of the
        # five were missing.
        self._ladder_depth_override: int | None = None
        self._ladder_sims_override: int | None = None
        self._ladder_saturated = False
        self._ladder_worlds_agree = True
        #: Per-rung staging for override ADDRESSES. `None` means "no ladder running",
        #: which is the state every non-model path and every direct `_search_model`
        #: caller sees, so those commit straight through the cap as they always did.
        self._ladder_pending_addresses: Optional[list[dict[str, Any]]] = None
        #: PER-BATTLE adaptive state. The budget ladder walks across TURNS, not within
        #: one decision: every decision spends exactly `search_sims` sim-equivalents at
        #: the current allocation, and the allocation for the NEXT turn is chosen from
        #: what this turn actually did. Keyed on `battle_id` and reset when it changes,
        #: so the reset is correct whether the bridge builds one policy per battle or
        #: reuses one across a whole shard.
        self._ladder_battle: Optional[str] = None
        self._ladder_worlds: Optional[int] = None
        self._ladder_depth: Optional[int] = None
        #: worlds count -> the deepest depth observed to SATURATE at that count. Latched
        #: on a failed probe and cleared when worlds drops, because dropping a world is
        #: the only thing that changes sims-per-world and therefore the only thing that
        #: can change the answer. Without the latch the state machine either parks one
        #: depth PAST the ceiling (up-only) or oscillates D <-> D+1 forever (both-ways),
        #: and in the oscillating case half of all turns run the thin tree the whole
        #: saturation rule exists to avoid.
        self._ladder_depth_ceiling: dict[int, int] = {}
        #: True when the CURRENT turn is a speculative probe of depth+1, so a failure
        #: can be attributed and charged rather than silently reverted.
        self._ladder_probing: bool = False
        # Measurement hook (fallback burndown plan 4 §3, direction 2): called
        # once per SUCCESSFULLY CONSTRUCTED world with the exact
        # `(context, EngineWorld, poke_engine.State)` triple search is about to
        # receive, so an oracle can project that world back into public protocol
        # facts and compare it with the observed log.
        #
        # A HOOK RATHER THAN A RE-SAMPLE, deliberately. A probe that re-sampled
        # its own worlds would be measuring a different draw than the one
        # searched, which is report 4 §4.2's failure shape ("the harness reported
        # success while measuring one thing twice") wearing a new badge. `None`
        # by default: production constructs this policy without it and the call
        # site is a single `is not None` test.
        self._world_observer = world_observer
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
        # Every Tier-2 conclusion already applied to this (battle, seat) fold, kept
        # after FoldState._prune_annotations drops it. The annotation SOURCE is
        # cumulative from battle start while the fold's annotation map is a WINDOW;
        # without this record the adapter cannot tell an already-applied conclusion
        # the fold has since pruned from one that is arriving late.
        self._fold_annotations_seen: dict[tuple[str, str], dict[int, tuple]] = {}
        self._tables_json: str | None = None
        self._model_config: Any | None = None
        self._native_model: Any | None = None
        if self._config.leaf_eval == "model":
            from pathlib import Path  # noqa: PLC0415 — model-mode-only dependency

            from .neural_policy import (  # noqa: PLC0415
                load_transformer_checkpoint_payload,
                parse_transformer_model_config,
            )

            model_path = Path(str(self._config.model_path))
            if not model_path.exists():
                raise ValueError(f"model artifact not found: {model_path}")
            checkpoint_path = Path(str(self._config.checkpoint_path))
            if not checkpoint_path.exists():
                raise ValueError(f"source checkpoint not found: {checkpoint_path}")
            tables_path = Path(str(self._config.tables_path))
            if not tables_path.exists():
                raise ValueError(f"encoder tables not found: {tables_path}")
            # ONE load serves both the model config and the calibration fence. This is the
            # model-leaf branch (`leaf_eval == "model"`) of `EngineMctsPolicy.__init__`, which
            # is exactly the path the fence governs; putting it here rather than in
            # `EngineMctsConfig.__post_init__` means no checkpoint is read while merely
            # constructing a config, and it fires before any search runs. The previous revision
            # of this fence defined the helper and never called it: dead code that reported a
            # guarded seam. `weights_only=True` is sufficient because the transform is
            # persisted as a plain dict of primitives.
            payload = load_transformer_checkpoint_payload(checkpoint_path)
            _fence_calibration_seam(payload, f"checkpoint {checkpoint_path}")
            self._model_config = parse_transformer_model_config(payload)
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

    def _notify_world_observer(
        self, context: PolicyContext, world: EngineWorld, state: Any
    ) -> None:
        """Hand one constructed world to the measurement hook, if any.

        A NAMED METHOD, not an inline `if`, so that deleting it is a mutation a
        battery can kill. Report 4 §4.4's third failure mode is a mutant that
        dies of the wrong cause; its mirror is a mutant that cannot be applied at
        all, and an inline guard buried in a 140-line method is the shape that
        survives. Direction 1's independent review found exactly this: mutant M20
        deleted an inline gate and survived the whole suite until the gate was
        extracted.

        Never allowed to break a search. An observer is telemetry; a raising one
        must not turn a measured run into a crashed one, and it must not silently
        change which worlds get searched either -- so the exception is swallowed
        HERE, after the world is already appended.
        """

        observer = self._world_observer
        if observer is None:
            return
        try:
            observer(context, world, state)
        except Exception as error:  # noqa: BLE001 — telemetry never breaks a run
            warnings.warn(
                f"world_observer raised: {type(error).__name__}: {error}",
                EngineSearchFallbackWarning,
                stacklevel=3,
            )

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
            except PokeEngineMoveTrapUnsupportedError:
                # Same shape, newly reachable. `require_move_trap_support` has always guarded
                # the TRAPPED volatile, but until the move trap was routed into the payload no
                # production world ever carried one, so the raise had no live caller and would
                # have escaped `_search` as a hard crash on an unpatched wheel. Attribute it
                # like the Attract probe: an unpatched wheel drops TRAPPED silently and would
                # hand the trapped seat its switch options back, so declining is correct --
                # but declining is a fallback, not a crashed run.
                self.stats.world_failure_reasons["move_trap_patch_unavailable"] += 1
                continue
            except PokeEngineUnavailableError as error:
                # BACKSTOP for the whole capability-probe family, because the specific handler
                # above does NOT cover every raise from the branch it guards. `_build_side_spec`
                # calls `require_move_trap_support()` with no module, so it resolves the engine
                # through `require_poke_engine()`, which raises the BASE
                # `PokeEngineUnavailableError` when `probe_poke_engine()` is not ready -- an
                # importable but mis-built wheel missing a required `State` method, exactly the
                # case these buckets exist for. That is neither the subclass above nor an
                # `EngineWorldUnsupported`, so it escaped `_search` entirely.
                #
                # Not fixed by plumbing `self._module` into the gate instead: production
                # constructs `EngineMctsPolicy` with `module=None`, so `build_poke_engine_state`
                # resolves the same global module two lines below. The two paths already agree;
                # what was missing was a handler.
                #
                # Pre-existing, not introduced here -- `require_charge_state_support()` sits two
                # lines away with the identical shape and has been reachable since `solarbeam`
                # became a tracked volatile. Attributed by exception class so the ledger
                # distinguishes "no engine at all" from a specific missing patch rather than
                # folding both into one bucket.
                self.stats.world_failure_reasons[
                    f"engine_capability_unavailable: {type(error).__name__}"
                ] += 1
                continue
            except EngineWorldUnsupported as error:
                self.stats.world_failure_reasons[_world_failure_key(error)] += 1
                continue
            worlds.append((world, state))
            self._notify_world_observer(context, world, state)

        if not worlds:
            return self._fallback(context, rng, "no_worlds_constructed")

        # Counted HERE, at the single dispatch point, rather than at the append
        # above: this is exactly the list every search path is about to receive,
        # so neither denominator can drift from what `worlds_searched` counts no
        # matter which leaf_eval runs.
        self.stats.worlds_constructed += len(worlds)
        # And the SEARCH-attempt denominator, which starts equal to it. On every
        # non-ladder path it stays equal, so `world_search_abort_rate` reads exactly
        # as it always did; `_search_ladder` adds its extra rungs on top.
        self.stats.world_search_attempts += len(worlds)

        if self._config.leaf_eval == "model":
            return self._search_ladder(context, worlds, live_fold, rng)
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
                        # Keyword, and omitted entirely when unset, so this call
                        # stays runnable against a pre-FPU wheel: the depth study
                        # measures this path and a TypeError here would look like
                        # a world failure rather than a stale image.
                        **(
                            {"fpu_reduction": config.fpu_reduction}
                            if config.fpu_reduction is not None
                            else {}
                        ),
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
            # A fresh fold has applied nothing, so the applied-conclusion record
            # must start empty: re-seating another battle's index into a new fold
            # is the fail-open this reset forecloses.
            self._fold_annotations_seen.pop(key, None)
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

        The source's overlay is CUMULATIVE from battle start; the fold's
        ``annotations`` map is a WINDOW — ``FoldState._prune_annotations``
        drops every index that has left both the action tail and the merged
        tail's representative set. This adapter reconciles the two, so
        ``index not in fold.annotations`` means "not currently held", NEVER
        "never applied": ``self._fold_annotations_seen`` is the record of what
        this fold has actually been given.

        Three cases, per overlay index:

        * currently held, or still identifiable (inside the action tail or the
          open window — ``FoldState._token_identity``'s contract): goes through
          unchanged. Already-applied indices are equality-checked by
          ``apply_annotations_in_place`` (per-index immutability: a changed
          tracker conclusion is a real regression and breaks the fold loudly).
        * applied earlier and since pruned: re-seated from the record before
          the apply, so the fold's OWN immutability check adjudicates it and
          its own pruning drops it again. A no-op on every product — the index
          has no token in the per-action tail and no representative in the
          merged tail, and its contribution to the unpruned full-stream
          reductions (``cb_pinned``, ``investment_pinned_state``) was banked at
          first application. ``tail_start`` never decreases and
          ``rep_index_map`` never regains a dropped index, so a re-seated entry
          is always pruned again and the post-apply state is byte-identical.
        * never applied and no longer identifiable: a genuinely late conclusion
          that would silently desynchronize the encoder-visible surface. Still
          breaks the fold loudly — this is the case the guard was written for.

        What protects the record, and what does NOT
        -------------------------------------------
        The record is refreshed at the bottom of this method from
        ``fold.annotations`` — the values the fold ACTUALLY applied, canonical-
        ized by ``apply_annotations_in_place`` — and no index is ever recorded
        twice with a different value.

        ⚠ **Per-index immutability does NOT establish that, and #1216's merge
        message wrongly said it did.** The equality check in
        ``transitions_fold.apply_annotations_in_place``
        (``transitions_fold.py:316``) runs only inside ``if existing is not
        None`` (``transitions_fold.py:329``) — i.e. only for indices the fold
        CURRENTLY HOLDS. An index the fold does not hold is applied FRESH, with
        nothing to compare against. Every index this record has to defend is
        precisely one the fold no longer holds, so immutability offers this path
        no protection at all. Do not re-derive that argument.

        What does establish it is MONOTONICITY of ``tail_start``. For
        ``record[index]`` to be overwritten with a DIFFERENT value, a pruned
        index would have to re-enter ``[tail_start, action_total]`` and be
        applied fresh from a changed overlay. It cannot:

        * ``tail_start = action_total - len(action_tail)`` is monotonic
          non-decreasing. ``FoldState._close_window``
          (``transitions_fold.py:814-817``) appends the token, trims the tail to
          ``action_tail_limit``, and only THEN increments ``action_total``:
          while the tail is still filling, both sides grow by one and
          ``tail_start`` is unchanged; once the tail sits at its limit,
          ``action_total`` grows alone and ``tail_start`` rises by one. It never
          falls.
        * an index enters the record only while the fold holds it, and is pruned
          only when ``index < tail_start`` and it has no merged-tail
          representative (``FoldState._prune_annotations``,
          ``transitions_fold.py:984-989``). Monotonicity then keeps
          ``index < tail_start`` at every later boundary, so the
          ``tail_start <= index`` branch above is unreachable for it forever,
          and the only way back into ``fold.annotations`` is the re-seat — which
          supplies the RECORDED value and is therefore adjudicated by the
          equality check rather than applied fresh. ``rep_index_map`` cannot
          readmit a dropped index either: its keys are
          ``expansion_cursor + _representative_offset(sub)``
          (``transitions_fold.py:951-956``) with a non-negative offset off a cursor
          that only advances and that the flatten-bijection assertion holds equal
          to ``action_total``, so every key it gains is at or above every key it
          has ever held (dropped by ``FoldState._prune_rep_index_map``,
          ``transitions_fold.py:979-982``).
        * the two sites where a fold's ``action_total`` can REGRESS both drop
          the record first: a rebuilt fold (``fold is None`` →
          ``_fold_annotations_seen.pop``) and a rewound event stream (→
          ``_mark_fold_broken``, which is sticky per ``(battle, seat)``, so that
          key never advances again). ``_drop_stale_folds`` drops other battles'
          records, and the per-seat key keeps seats apart.

        Stronger still, and worth writing down because it bounds the assertion's
        blast radius: in correct operation the record is never even RE-VISITED
        with a second value for one index. ``apply_annotations_in_place`` ends
        with ``self._prune_annotations()`` at ``transitions_fold.py:362``, at
        METHOD level rather than inside its loop, so it runs on every call — and
        a re-seated index is by construction ``< tail_start`` with no merged-tail
        representative, so it is dropped again before control returns here. A
        resettled index is therefore never re-recorded at all. And a still-HELD
        index cannot have changed value, because ``transitions_fold.py:338``
        writes ``self.annotations[index]`` only on the ``existing is None`` path
        (``:329`` returns early otherwise). Both halves of the loop below are
        thus benign in correct operation; the raise is unreachable.

        That is a real dependency rather than a restatement of immutability, and
        the thing that would actually break it is REPOPULATING ``action_tail``
        from a longer history than the live fold's — ``_clone``
        (``transitions_fold.py:1102``) and ``from_payload``
        (``transitions_fold.py:1280-1281``) are the only writers of the whole
        deque, and a fold rebuilt through either with a record carried across
        would lower ``tail_start`` while per-index immutability still held
        exactly as written. (⚠ An earlier revision of this paragraph offered a
        dynamic or growable ``action_tail_limit`` as the example. That is WRONG
        and review caught it: ``_close_window`` appends exactly one token per
        ``action_total`` increment, so raising the limit only stops a
        ``popleft`` — ``tail_start`` stays put — and lowering it pops more, which
        RAISES ``tail_start``. Neither direction can regress it. The wrong
        example is recorded rather than deleted because it was the one
        load-bearing reason offered for why this section is not a wording
        change.) So the record refresh below ASSERTS the invariant instead of
        assuming it.
        """
        source = self._annotation_source
        if source is None or not source.active():
            return True
        try:
            overlay = source.overlay_for(context.player_id)
            if overlay:
                before = len(fold.annotations)
                tail_start = fold.action_total - len(fold.action_tail)
                seen = self._fold_annotations_seen.get(key) or {}
                resettle: dict[int, tuple] = {}
                stale: list[int] = []
                for index in overlay:
                    if index in fold.annotations:
                        continue
                    if tail_start <= index <= fold.action_total:
                        continue
                    if index in seen:
                        resettle[index] = seen[index]
                    else:
                        stale.append(index)
                if stale:
                    raise ValueError(
                        f"tracker annotations for indices {sorted(stale)[:8]} arrived "
                        f"outside the identifiable range [{tail_start}, "
                        f"{fold.action_total}] — encoder-visible surface would desync."
                    )
                if resettle:
                    fold.annotations.update(resettle)
                    self.stats.fold_annotations_resettled += len(resettle)
                    self.stats.fold_annotation_resettle_boundaries += 1
                # The FULL cumulative overlay goes through: already-applied
                # indices are equality-checked inside (per-index immutability
                # — a changed tracker conclusion raises and breaks the fold).
                fold.apply_annotations_in_place(overlay)
                applied = max(0, len(fold.annotations) - before)
                if applied:
                    self.stats.fold_annotations_applied += applied
                    self.stats.fold_annotation_boundaries += 1
                # Refresh the record from what the fold HOLDS (canonicalized by
                # apply_annotations_in_place), never from what the source
                # OFFERED: the record's meaning is "applied once", and a re-seat
                # must hand the fold back a value the fold itself produced.
                #
                # This was `record.setdefault(index, values)` — silently keeping
                # the first value, i.e. defence-in-depth nothing could falsify.
                # It is now the live guard for the monotonicity invariant in the
                # docstring above. An explicit raise, not an `assert` statement,
                # because `python -O` strips those; it is caught below and turns
                # into the loud fold-broken path, which is this method's whole
                # failure convention.
                record = self._fold_annotations_seen.setdefault(key, {})
                for index, values in fold.annotations.items():
                    previous = record.get(index)
                    if previous is None:
                        record[index] = values
                    elif tuple(previous) != tuple(values):
                        raise AssertionError(
                            f"applied-conclusion record for token index {index} "
                            f"changed ({previous!r} -> {values!r}): a pruned index "
                            "was applied FRESH again, so tail_start regressed and "
                            "the re-seat path is no longer sound."
                        )
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
        for key in [k for k in self._fold_annotations_seen if k[0] != battle_id]:
            self._fold_annotations_seen.pop(key, None)

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

    def _absorb_aborted_lossy_subcases(self, error: BaseException) -> None:
        """Count the sub-cases an ABORTED world observed before it died.

        THE DISCARDED CASE. A native world that hits an attribution-unsafe branch (or any
        other mid-search error, or a contained poke-engine panic) returns before its
        report string is built, so the seam above -- which reads that report -- was never
        reached and every non-refusing diagnostic the world had accumulated was thrown
        away. So `lossy_subcase_renders` described only the clean-completion subset:
        precisely the subset that does not need diagnosing.

        WHAT SHARE OF WORLDS ABORT IS NOT MEASURED, and is deliberately not stated.
        Earlier revisions of this docstring, of `abort_telemetry.rs`, of `model.rs`, of
        `tests/test_engine_search.py` and of the gate workflow all said aborts are "~92%
        of the fallback residue" / "THE MAJORITY CASE". WITHDRAWN: the figure has no
        source. It appears nowhere on `main`, in no committed artifact and in no prior
        report -- all six occurrences arrived with this change -- and it is not derivable
        here either, because the native entry point this seam wraps
        (`search_batched_multi_encoded`) is behind the crate's `model` cargo feature,
        which no CI job and no sweep builds, so nothing available to this repository
        reaches the abort path at all. The committed evidence points the other way: over
        `docs/audit_artifacts/**/*.json` and `reports/**/*.json` (359 files, 22 with a
        non-empty `world_failure_reasons`), all 6,144 recorded failures are
        world-CONSTRUCTION failures -- 5,328 `self_moveset_mismatch`, 416
        `transform_unexpressible`, 160 `materialization_blocker`, 160
        `self_request_state_unsupported`, 64 `volatile_unsupported`, 16
        `encore_move_unknown` -- and exactly ZERO carry the `crate_search:` prefix this
        seam writes. The defect is categorical and needs no magnitude: an abort discarded
        EVERY count it had accumulated. Do not reintroduce a percentage without an
        artifact.

        #1158 paid for this directly: its Protect-marker counter read zero both when the
        fix never fired and when the fix fired but the world died at its NEXT unsafe
        branch, so it answered a narrower question than it was built for. That confound is
        what this method removes on a `model`-feature build; see `events.rs`,
        `protect_marker_rendered`.

        The counts ride on the exception as an ATTRIBUTE, never as text in its message:
        that message becomes the `world_failure_reasons` key, whose bytes are a
        measurement contract compared across eras and which `_bounded_reason_detail`
        truncates at 512 chars.

        Routed through `_absorb_lossy_subcases` rather than incrementing directly, so the
        abort path and the clean path share ONE accumulation point and land in
        `lossy_subcase_renders` identically. They can never both fire for one invocation
        --- a report and an exception are exclusive outcomes --- so nothing is
        double-counted. An early-stop world replayed at full budget contributes twice, on
        purpose: that is two invocations of real render work, the same convention
        `lossy_renders` and `total_iterations` already follow.

        SANITIZED HERE, not in the shared seam, because THIS is the untrusted boundary.
        The report path's counts come from a `format!` in the crate and a malformed one
        there is a bug worth surfacing; these come off an arbitrary caught exception, and
        an `isinstance(payload, Mapping)` check alone is NOT enough -- measured,
        `{"a": "many"}` raises ValueError and `{"a": None}` / `{"a": {"b": 1}}` raise
        TypeError out of `_absorb_lossy_subcases`'s unguarded `int(count)`, out of the
        `except Exception` in `run_world`, and (there is no outer try around
        `_search_model`) straight out of `decide()`. Unreachable from the wheel that
        ships, but it is exactly the "arbitrary exception" case the namespaced attribute
        name exists to bound, and a crash in the handler whose entire job is to keep the
        other worlds alive is a strictly worse defect than the one being fixed. Older
        wheels, and every non-native failure reaching this handler, carry no such
        attribute at all and no-op the same way.

        `bool` is excluded deliberately: it is an `int` subclass, so `True` would
        otherwise count as 1 and put a non-count into a measurement channel.
        """

        payload = getattr(error, _ABORT_LOSSY_SUBCASES_ATTR, None)
        if not isinstance(payload, Mapping):
            return
        counts = {
            str(subcase): count
            for subcase, count in payload.items()
            if isinstance(count, int) and not isinstance(count, bool)
        }
        self._absorb_lossy_subcases({"lossy_subcases": counts})

    def _turn_allocation(self, available_worlds: int) -> tuple[int, int | None, int]:
        """This TURN's allocation: (worlds, sims_per_world, depth).

        `search_sims` is the TOTAL sim-equivalents for ONE DECISION and it is spent
        exactly once, at whatever allocation the per-battle state currently names.
        4 worlds x 4,096 and 1 world x 16,384 are the same 16,384 spent two ways --
        that is the compute-neutrality the whole design rests on, and it means a
        dynamic cell costs the same per decision as a fixed one.

        THE LADDER WALKS ACROSS TURNS, NOT WITHIN A DECISION. An earlier revision read
        the owner's "once you have enough information to drop to 3w, then you can scale
        up sims" as a within-decision escalation and ran a FRESH full-budget search per
        rung, so a decision that climbed 6.7 rungs spent 6.7 x 16,384 ~ 110,000 sims.
        Measured on a canary against the banked fixed comparator: 36-63 s per decision
        against 10.06 s, a 3.6x-6.3x cost REGRESSION rather than the saving the feature
        is for. It also destroyed the saturation gate, because a rung that gets a fresh
        full budget saturates as easily at depth 8 as at depth 2 -- measured at 0-3
        unsaturated stops per ~25 decisions, i.e. depth marching unconditionally.

        With one budget per turn the gate becomes binding on its own: sims-per-world is
        set by the world count, so as depth climbs the same allocation fills less of the
        tree and the climb stops where the budget genuinely runs out.

        Returns `sims_per_world=None` on a FIXED cell so the native positional list is
        byte-identical to what it was before this feature existed.
        """
        cfg = self._config
        depth_cap = int(cfg.search_depth)
        worlds_cap = max(1, min(int(cfg.worlds), available_worlds))
        if cfg.depth_min is None and cfg.worlds_min is None:
            # NO LADDER. `sims=None` means "use the configured budget untouched", which
            # is what keeps a fixed cell poolable with every banked shard. A number here
            # would silently run it at budget/worlds -- the F1 defect.
            return (worlds_cap, None, depth_cap)
        worlds = worlds_cap if self._ladder_worlds is None else min(
            self._ladder_worlds, worlds_cap
        )
        depth = depth_cap if self._ladder_depth is None else min(
            self._ladder_depth, depth_cap
        )
        # Integer division, so the turn never costs MORE than the budget: 3 worlds of
        # 16,384 is 5,461 x 3 = 16,383, one short rather than one over.
        # `__post_init__` refuses `search_sims < worlds` on a dynamic cell, so this
        # cannot round to zero.
        per_world = int(cfg.search_sims) // worlds
        return (worlds, per_world, depth)

    def _reset_ladder_for_battle(self, battle_id: str) -> None:
        """Per-battle reset. A new team and a new opponent is a new problem."""
        cfg = self._config
        self._ladder_battle = battle_id
        self._ladder_worlds = max(1, int(cfg.worlds))
        self._ladder_depth = (
            int(cfg.search_depth) if cfg.depth_min is None
            else max(LADDER_MIN_DEPTH_FLOOR, int(cfg.depth_min))
        )
        self._ladder_depth_ceiling = {}
        self._ladder_probing = False

    def _advance_ladder_state(
        self, worlds: int, depth: int, saturated: bool, agree: bool
    ) -> None:
        """Choose the NEXT turn's allocation from what THIS turn measured.

        A WORLD is dropped once the per-world visit leaders AGREE -- more draws of a
        belief every draw already agrees on cannot buy information. Dropping raises
        sims-per-world, the only thing that can change the saturating depth, so it also
        clears the depth latch for the level it is moving to.

        DEPTH is NOT raised here. It moves only on evidence from a SHADOW search at
        depth+1 (see `_search_ladder`), never on a bare "the current depth saturated",
        because both naive rules are broken: up-only parks one depth PAST the ceiling
        (it moves to D+1 on saturating at D and has no way back), and up-and-down
        oscillates D <-> D+1 forever -- and in that case half of all turns play a move
        chosen from exactly the thin tree the saturation rule exists to avoid.
        """
        cfg = self._config
        worlds_floor = worlds if cfg.worlds_min is None else max(1, int(cfg.worlds_min))
        if agree and worlds > worlds_floor:
            self._ladder_worlds = worlds - 1
            # NOTHING to clear. The latch is KEYED BY WORLD COUNT, and worlds only ever
            # decrease within a battle, so the new count has no entry and probing is
            # allowed again automatically. An earlier revision popped `worlds - 1` here
            # to "clear the latch"; the latch is stored under `worlds`, so the pop
            # removed a key that was never set -- dead code that read like a load-bearing
            # invariant. Caught by a mutation screen: deleting it changed nothing.
            self.stats.ladder_world_drops += 1
        elif not agree:
            self.stats.ladder_worlds_disagree_stops += 1
        if not saturated:
            self.stats.ladder_unsaturated_stops += 1

    def _should_shadow_probe(self, worlds: int, depth: int, saturated: bool) -> bool:
        """Is a shadow search at depth+1 worth its budget on this turn?

        Only when the current depth SATURATED (so there is headroom to spend), the cap
        allows, and this world count has no latched ceiling yet. That bounds probes to
        at most one per world count per battle -- four for a 4->1 ladder -- which is what
        makes the doubled budget on those turns affordable: about four extra budgets
        across a ~30-turn battle, and `ladder_shadow_probes` reports the real count
        rather than leaving it assumed.
        """
        cfg = self._config
        if not saturated or depth >= int(cfg.search_depth):
            return False
        return depth < self._ladder_depth_ceiling.get(worlds, int(cfg.search_depth))

    def _search_ladder(
        self,
        context: PolicyContext,
        worlds: list[tuple[EngineWorld, Any]],
        live_fold: Any,
        rng: random.Random,
    ) -> PolicyDecision:
        """ONE PLAYED search per decision, plus an occasional discarded shadow.

        The move is always chosen from a depth already known to saturate. When there is
        headroom, a SHADOW search runs at depth+1 purely to answer "would this
        allocation fill one more ply?", and its decision is thrown away. That is what
        makes this better than spending a real turn at depth+1 and reverting: no move is
        ever played from a tree we then judge too thin.

        The shadow's CLAIMS are rewound and its WORK is not -- exactly the distinction
        `LADDER_PER_DECISION_CLAIMS` was built to express. It made no decision, so it
        gets no vote in any per-decision rate; it really did burn the simulations and
        the wall, so a cost analysis must see them.
        """
        dynamic = (
            self._config.depth_min is not None or self._config.worlds_min is not None
        )
        battle = str(getattr(context, "battle_id", "?"))
        if dynamic and battle != self._ladder_battle:
            self._reset_ladder_for_battle(battle)
        stage_worlds, stage_sims, stage_depth = self._turn_allocation(len(worlds))
        self.stats.ladder_decisions += 1
        if dynamic:
            self.stats.ladder_dynamic = True

        def _run(depth: int):
            """One search. Returns (decision, fell_back, saturated, agree)."""
            before = self.stats.fallback_decisions
            self.stats.ladder_rungs_run += 1
            self._ladder_saturated = False
            self._ladder_worlds_agree = True
            self._ladder_depth_override = depth
            self._ladder_sims_override = stage_sims
            self._ladder_pending_addresses = None
            try:
                out = self._search_model(context, worlds[:stage_worlds], live_fold, rng)
            finally:
                self._ladder_depth_override = None
                self._ladder_sims_override = None
            return (
                out,
                self.stats.fallback_decisions > before,
                bool(getattr(self, "_ladder_saturated", False)),
                bool(getattr(self, "_ladder_worlds_agree", True)),
            )

        rows_before = len(self.stats.root_decision_rows)
        decision, fell_back, saturated, agree = _run(stage_depth)
        if dynamic:
            for row in self.stats.root_decision_rows[rows_before:]:
                row["ladder_rung"] = 0
                row["ladder_superseded"] = False
                row["ladder_worlds"] = stage_worlds
                row["ladder_depth"] = stage_depth

        if fell_back:
            # Measured nothing, so it moves nothing: letting a broken turn choose the
            # next allocation would adapt to an absence.
            self.stats.ladder_unsearched_decisions += 1
            return decision

        if dynamic and self._should_shadow_probe(stage_worlds, stage_depth, saturated):
            claims_before = tuple(
                getattr(self.stats, name) for name in LADDER_PER_DECISION_CLAIMS
            )
            hists_before = tuple(
                Counter(getattr(self.stats, name))
                for name in LADDER_PER_DECISION_CLAIM_HISTOGRAMS
            )
            addresses_before = len(self.stats.override_disagreements)
            rows_before_shadow = len(self.stats.root_decision_rows)
            self.stats.ladder_shadow_probes += 1
            _, shadow_failed, shadow_saturated, _ = _run(stage_depth + 1)
            for name, value in zip(LADDER_PER_DECISION_CLAIMS, claims_before):
                setattr(self.stats, name, value)
            for name, hist in zip(LADDER_PER_DECISION_CLAIM_HISTOGRAMS, hists_before):
                live = getattr(self.stats, name)
                live.clear()
                live.update(hist)
            superseded = len(self.stats.override_disagreements) - addresses_before
            if superseded > 0:
                del self.stats.override_disagreements[addresses_before:]
                self.stats.override_addresses_superseded += superseded
            del self.stats.root_decision_rows[rows_before_shadow:]
            if shadow_failed:
                # No information either way, so latch NOTHING -- a latch on a failure
                # would record a ceiling the search never measured.
                self.stats.ladder_shadow_probe_failures += 1
            elif shadow_saturated:
                self._ladder_depth = stage_depth + 1
                self.stats.ladder_depth_rungs += 1
            else:
                self._ladder_depth_ceiling[stage_worlds] = stage_depth
                self.stats.ladder_depth_latches += 1

        if dynamic:
            self._advance_ladder_state(stage_worlds, stage_depth, saturated, agree)
        return decision


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

        # Relative to THIS RUNG's per-world budget. `early_stop_min_sims` is
        # validated against `search_sims`, which on a ladder cell is the TOTAL for
        # the decision -- so a floor of 64 against a total of 128 across 4 worlds is
        # a floor ABOVE the 32-sim rung, and the stop rule could never fire on the
        # rungs that need it most. Found in review.
        #
        # AND THE CLAMP IS THE CONSERVATIVE DIRECTION, which is worth stating because
        # it reads like a relaxed bar. Clamping the floor to the rung sets it to the
        # rung's ENTIRE budget, so the earliest a stop can fire is the last batch of
        # that rung -- it cannot fire early. The absolute evidence is smaller (32 sims
        # rather than 64) only because 32 is all that rung has; the alternative is not
        # "more evidence", it is early-stop being silently dead on every rung below
        # the floor while the cell's name still claims the feature. A fixed cell is
        # untouched: `_ladder_sims_override` is None there, so the configured floor
        # reaches the crate byte-identically and no banked measurement moves.
        stop_floor = config.early_stop_min_sims if config.early_stop else 0
        _rung_sims = getattr(self, "_ladder_sims_override", None)
        if stop_floor and _rung_sims:
            stop_floor = max(1, min(stop_floor, int(_rung_sims)))
        world_runs: list[dict[str, Any]] = []
        # Duplicate belief completions, grouped per DECISION by search-problem
        # identity. Never shared across turns.
        duplicates: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        search_started = time.perf_counter()

        def run_world(
            record: Mapping[str, Any],
            early_stop_min_sims: int,
            sims: int | None = None,
            weight: int = 1,
            depth: int | None = None,
        ) -> Optional[dict]:
            try:
                search_args = native_search_args(
                    config,
                    record,
                    tables_json=self._tables_json,
                    root_inputs=root_inputs,
                    rust_fold=rust_fold,
                    early_stop_min_sims=early_stop_min_sims,
                    sims=sims,
                    depth=depth,
                )
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
                    # The telemetry flag appends a positional, so a stale image
                    # rejects the call outright -- as a TypeError, from the same
                    # arity mismatch the early-stop flag hit. Named, because
                    # "every world failed" against a rebuilt-Python/old-image pod
                    # is otherwise indistinguishable from a search defect.
                    else f"native_override_telemetry_unsupported: {detail}"
                    if config.override_telemetry and isinstance(error, TypeError)
                    else detail
                )
                # Unsafe renderer branches abort the native world before a
                # chance outcome can be silently omitted from its expectation.
                # The native report is unavailable on that error path, so
                # retain the same observability counter at the fallback seam.
                if "attribution-unsafe renderer branch rejected before" in reason:
                    self.stats.attribution_unsafe_renders += weight
                # RECORD THE FAILURE REASON FIRST. `_absorb_aborted_lossy_subcases` is
                # written to swallow everything a malformed payload can throw, and its
                # own docstring says an escape would propagate out of `decide()` -- but
                # "written to" is not "proven to", and the ordering costs nothing. With
                # the absorb first, any escape loses this world's `world_failure_reasons`
                # entry, and those keys are a measurement contract compared across eras:
                # the fallback would be undercounted rather than merely undiagnosed, so
                # the failure mode would be a wrong number instead of a missing one.
                # Reason first, diagnostics second, is strictly fail-safer.
                #
                # WEIGHTED by the collapse multiplicity, for the same reason the depth
                # samples below are, and to keep the measurement contract this comment
                # invokes: this is a per-WORLD event whose denominators --
                # `worlds_attempted` and `worlds_constructed` -- are still counted per
                # DRAW. A refusal is deterministic in the state, so EVERY draw of an
                # aborting completion aborts; counting the abort once per SEARCH would
                # deflate these keys by the duplicate multiplicity while their
                # denominators kept the old unit, silently shifting
                # `world_search_abort_rate` and every cross-era ranking the
                # fallback-burndown campaign builds on them.
                self.stats.world_failure_reasons[f"crate_search: {reason}"] += weight
                # ... and everything ELSE the world observed before it aborted, which
                # this seam used to discard wholesale.
                self._absorb_aborted_lossy_subcases(error)
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
                self.stats.depth_reached_samples += weight
                self.stats.depth_reached_sum += reached * weight
                self.stats.depth_reached_histogram[reached] += weight
                self.stats.depth_reached_max = max(self.stats.depth_reached_max, reached)
            # OCCUPANCY, which is a different question from the max above and the one the
            # saturation gate actually needs. `max_depth_reached` is satisfied by a single
            # deep line, so a ceiling share computed from it means "the ceiling was
            # REACHABLE", not "the depth is filled" -- measured on this campaign at a gate
            # that fired on only 14-16% of turns. `depth_occupancy[d]` counts traversals
            # that opened a decision node at depth d, so ONE search at the cap yields the
            # whole profile and the saturating depth is a read rather than a scan.
            occ = report.get("depth_occupancy")
            if isinstance(occ, list) and occ:
                for depth, count in enumerate(occ):
                    if count:
                        self.stats.depth_occupancy[depth] += int(count) * weight
                record["_depth_occupancy"] = [int(c) for c in occ]
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
            # Per-INVOCATION like the phase walls above: a conservatively
            # replayed world collided that many times twice and must report it.
            # `.get(...) or 0` keeps a pre-collision-counter wheel readable.
            for field_name in (
                "collision_rounds",
                "collision_pending_rounds",
                "collision_selections",
                "collision_joint_repeats",
                "collision_self_repeats",
                "collision_opponent_repeats",
                "collision_traversals",
                "collision_leaf_repeats",
            ):
                setattr(
                    self.stats,
                    field_name,
                    getattr(self.stats, field_name) + int(report.get(field_name) or 0),
                )
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
            # CONCENTRATE duplicate belief completions instead of skipping them.
            #
            # Two worlds with the same serialized state, context and seat are one
            # hypothesis drawn twice. It is tempting to search it once and reuse
            # the answer, but that is WRONG about what the duplicate searches
            # were doing: the per-world seed drives chance-node sampling
            # (model.rs -> tree.rs sample_branch_index), so repeated draws of one
            # completion are INDEPENDENT Monte-Carlo estimates whose average
            # reduces the variance of that completion's contribution. Skipping
            # them keeps the estimator unbiased but drops its effective sample
            # size from N to 1 and reinvests nothing -- a strength regression
            # bought with wall-clock, worst in exactly the fully-revealed case
            # where every draw is the same.
            #
            # So: search each unique completion ONCE at N x the sim budget, and
            # append N records carrying that report. Same total compute as
            # searching N times, same belief weighting (aggregation gives every
            # record weight 1, so N records weigh N), and one tree with N x S
            # sims dominates the average of N trees with S sims -- deeper, and
            # UCB gets to exploit the budget instead of restarting cold N times.
            cache_key = world_cache_key(record, side_key)
            duplicates.setdefault(cache_key, []).append(record)

        for cache_key, records in duplicates.items():
            multiplicity = len(records)
            lead = records[0]
            sims = None
            if multiplicity > 1:
                # The whole point: N draws of one completion buy N x the sims on
                # ONE tree, not N cold restarts. Total compute is unchanged.
                sims = config.search_sims * multiplicity
            ladder_sims = getattr(self, "_ladder_sims_override", None)
            if ladder_sims is not None:
                # The rung's PER-WORLD budget, still scaled by multiplicity so a
                # collapsed group keeps searching one tree at N x its share -- the
                # #1009 property, preserved under the ladder.
                sims = ladder_sims * multiplicity
            report = run_world(
                lead, stop_floor, sims, multiplicity,
                # getattr, not attribute access: the policy is constructed by
                # several paths in this codebase and its tests, and a ladder that
                # is not running must not require its scratch state to exist.
                depth=getattr(self, "_ladder_depth_override", None),
            )
            if report is None:
                continue
            # Both counters move ONLY on a search that returned a report, and
            # only together. Incrementing `worlds_collapsed` before the call --
            # where the sim scaling is decided -- broke the invariant below the
            # moment a collapsed group aborted: the failing group's records never
            # reach `worlds_searched`, so the difference went NEGATIVE (measured:
            # -2 for one aborting 3-group). Aborts are the common case for a
            # duplicated completion, not a corner one.
            self.stats.worlds_collapsed += multiplicity - 1
            self.stats.unique_worlds_searched += 1
            for record in records:
                # SHALLOW copy: the top-level dicts are distinct but nested
                # values -- notably report["side_one"], the visit list anyone
                # would realistically mutate -- are STILL SHARED between twins.
                # Safe today only because no consumer mutates a report in place
                # (the replay path rebinds record["report"]). Do not read this as
                # isolation; deepen it if that ever changes.
                record["report"] = dict(report)
                record["_collapse_key"] = cache_key
                record["_collapse_multiplicity"] = multiplicity
                world_runs.append(record)

        # Count each SEARCH once, not each record. Duplicate draws share one
        # search, so counting records attributed a stopped search's savings to
        # twins that were never issued -- measured at 120 simulations_saved
        # where the true figure was 40.
        stopped_runs = []
        _stopped_seen: set[Any] = set()
        for record in world_runs:
            if not bool(record["report"].get("early_stopped")):
                continue
            marker = record.get("_collapse_key")
            if marker in _stopped_seen:
                continue
            _stopped_seen.add(marker)
            stopped_runs.append(record)
        # WORLDS, matching meta["worlds_stopped"] and full_budget_replays.
        # Counting searches here under-reported by the collapse multiplicity --
        # the mirror of the 120-vs-40 over-count this dedupe fixed.
        self.stats.early_stop_triggered_worlds += sum(
            int(r.get("_collapse_multiplicity", 1)) for r in stopped_runs
        )
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
                #
                # NOT COLLAPSED, deliberately: this walks RECORDS, so an N-group
                # replays N times at the base budget rather than once at N x it.
                # That is the pre-feature behaviour exactly -- which is the point
                # of a fail-open path -- and it keeps `full_budget_replays`
                # counting records like `worlds_stopped` does. The cost is N
                # identical searches on an ambiguous stop; measured and accepted
                # rather than silently inherited. Collapsing here would also have
                # to re-derive the weight for the depth samples, which is the bug
                # the replay-weight test above pins.
                final_runs: list[dict[str, Any]] = []
                for record in world_runs:
                    if not record["report"].get("early_stopped"):
                        final_runs.append(record)
                        continue
                    full_budget_replays += 1
                    # The RUNG's allocation, not the configured cap. Passing
                    # neither ran the replay at the total budget per world AND at
                    # the depth cap -- a depth the ladder had not licensed -- and
                    # those reports then fed the saturation test, handing the
                    # ladder a forged licence to deepen. Found in review.
                    # `rung_sims` ALONE, with no multiplicity factor. This loop
                    # walks RECORDS, so an N-group is replayed N times -- the
                    # multiplicity is already spent by the iteration. Scaling each
                    # replay by it as well cost N^2 x rung_sims for one group: at
                    # a 4,096 rung and a 3-group, 36,864 sims against an intended
                    # 12,288, i.e. 2.25x the whole decision's budget, and the
                    # oversized trees then fed the saturation test and re-forged
                    # the licence F3 removed. Found by review of the F3 fix.
                    rung_sims = getattr(self, "_ladder_sims_override", None)
                    report = run_world(
                        record,
                        0,
                        sims=rung_sims,
                        depth=getattr(self, "_ladder_depth_override", None),
                    )
                    if report is None:
                        replay_failed = True
                        break
                    record["report"] = report
                    final_runs.append(record)
                world_runs = final_runs
                self.stats.early_stop_full_budget_replays += full_budget_replays
        # THE LADDER'S SIGNALS. Two, each answering the question its own axis can
        # act on.
        #
        # NOT `_locked_aggregate_choice`: that predicate compares the visit
        # leader's edge against UNSPENT simulations, which is right where the
        # early-stop path uses it -- mid-search, at a batch boundary. Here the rung
        # has run to completion, `remaining` is 0, and it degenerates to "is there a
        # unique leader": measured at 447 of 447 canary decisions, which pinned the
        # ladder to its floor.
        #
        # * SATURATION licenses DEPTH, and nothing else does. A deeper search that
        #   did not fill the shallower depth explores its new plies too thinly to
        #   back them up and can be worse than the depth beneath it.
        # * AGREEMENT licenses dropping a WORLD. Once every world's leader agrees,
        #   more draws of that belief cannot buy information, so the budget is
        #   better spent concentrating what remains.
        self._ladder_worlds_agree = True
        # SATURATION: the share of this rung's worlds whose tree reached the depth
        # ceiling. D-1, not the cap: `depth_reached == cap` is unreachable by
        # construction (tree.rs:487/553), so a saturated depth-6 search reports 5.
        ceiling = max(
            0,
            int(getattr(self, "_ladder_depth_override", None) or config.search_depth) - 1,
        )
        reached = [
            int(r["report"]["max_depth_reached"])
            for r in world_runs
            if r.get("report", {}).get("max_depth_reached") is not None
        ]
        self._ladder_saturated = bool(reached) and (
            sum(1 for x in reached if x >= ceiling) / len(reached)
        ) >= float(config.ladder_saturation)
        if world_runs:
            # PER-WORLD LEADERS, not an aggregate. Aggregating first would let one
            # world's landslide hide two others' disagreement, and disagreement is
            # the whole signal: it says another draw of this belief can still change
            # the answer. `_world_visit_shares` normalises within each world so a
            # collapsed group, searched at multiplicity x sims, does not outvote the
            # rest on budget alone. A world whose report is unreadable is SKIPPED,
            # never counted as agreeing.
            per_world_leaders: list[str] = []
            for record in world_runs:
                side = _world_visit_shares(record["side_key"], record["report"])
                if side is None:
                    continue
                per_world_leaders.append(max(side, key=lambda m: side[m]))
            if per_world_leaders:
                self._ladder_worlds_agree = len(set(per_world_leaders)) <= 1
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
        # AFTER `searched_decisions`, so the two counters this telemetry
        # partitions can never be incremented on different sets of decisions.
        override = (
            self._record_root_telemetry(
                context, world_runs, aggregated, action_index
            )
            if config.override_telemetry
            else None
        )
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
                        # WORLDS, not searches: full_budget_replays counts
                        # records, so counting searches here put the two on
                        # different denominators for the same event.
                        "worlds_stopped": sum(
                            int(r.get("_collapse_multiplicity", 1)) for r in stopped_runs
                        ),
                        "aggregate_locked": locked_choice is not None,
                        "locked_choice": locked_choice,
                        "full_budget_replays": full_budget_replays,
                        "simulations_saved": simulations_saved,
                    },
                    # Present only with the flag on, so a flag-off decision's
                    # metadata is exactly what it has always been. The shard's
                    # aggregate lives in `policy_stats`; this is the per-decision
                    # row, which is what the fork probe forks on.
                    **({"override": override} if override is not None else {}),
                }
            },
        )

    def _record_root_telemetry(
        self,
        context: PolicyContext,
        world_runs: list[dict[str, Any]],
        aggregated: Mapping[str, float],
        search_action_index: int,
    ) -> dict[str, Any]:
        """Absorb this searched decision's ROOT state: override, gaps, opponent arm.

        One pass, one row, three hypotheses. The investigation plan's offline
        legs were blocked on the same thing and not on three different things:
        the crate emits the root's arms with their visits, values and (with
        `arm_priors`) their priors, and the Python boundary absorbed only the
        visit aggregate and threw the rest away. So this is mostly ABSORPTION,
        not new measurement --

          * override (H1/section 2): the model's argmax vs the search's, as
            ACTION INDICES;
          * the top-1/top-2 root Q and visit-share gaps (H2): whether the value
            head separates the two arms search is actually choosing between;
          * the in-tree opponent's top arm (H4): the predictor side of "does the
            opponent we search against play what FoulPlay plays", which joins to
            the #1188 opponent journal on (battle_id, round).

        Every exit either scores the override or names a cause: no searched
        decision is left out of both counters, which is what the plan's
        denominator (`overrides / (searched - unmeasured)`) rests on.
        """
        arms = _aggregate_root_arms(world_runs)
        worlds = max(len(world_runs), 1)
        # --- the override, on action indices -------------------------------
        cause: Optional[str] = arms.prior_cause
        model_choice: Optional[str] = None
        model_action: Optional[int] = None
        if cause is None:
            model_choice = _leading_choice(arms.prior_share)
            vocabulary = self._choice_vocabulary(context)
            model_action = (
                None if vocabulary is None or model_choice is None
                else vocabulary.action_index(model_choice)
            )
            if model_action is None:
                # `_map_choices` already returned an index for this decision, so
                # the candidate list exists and the search's own choice mapped:
                # what failed is this arm specifically.
                cause = _OVERRIDE_UNMEASURED_UNMAPPED
        model_override: Optional[bool] = None
        if cause is not None:
            self.stats.search_override_unmeasured += 1
            self.stats.search_override_unmeasured_causes[
                _registered_override_cause(cause)
            ] += 1
        else:
            self.stats.override_measured_decisions += 1
            # ACTION INDICES, not displays. The engine and the request name the
            # same forced action differently ("No Move" / `recharge` /
            # `struggle`, typed Hidden Power / plain `hiddenpower`), so comparing
            # displays reports a pure naming difference as an override -- the
            # translation `_ChoiceVocabulary` exists for. Both sides go through
            # it, so only a real change of action can move this counter.
            model_override = model_action != search_action_index
            if model_override:
                self.stats.model_override_decisions += 1
                self._record_override_address(
                    context, model_action, search_action_index, model_choice
                )
        # --- the two top-arm gaps (H2) ------------------------------------
        leaders = _leading_pair(aggregated)
        q_gap: Optional[float] = None
        visit_gap: Optional[float] = None
        if len(leaders) == 2:
            first, second = leaders
            first_q, second_q = arms.arm_q.get(first), arms.arm_q.get(second)
            # `aggregated` sums a per-world share, so it totals to the record
            # count; dividing puts the gap back on [0, 1] and makes a w1 cell
            # comparable to a w4 one.
            visit_gap = (aggregated[first] - aggregated[second]) / worlds
            if first_q is not None and second_q is not None:
                # ABSOLUTE. The visit leader can hold the LOWER value -- that is
                # PUCT tracking the prior, which is H1's finding and not H2's --
                # and folding both cases into one signed histogram would let them
                # cancel. The sign stays recoverable from the row, which carries
                # both values.
                q_gap = abs(first_q - second_q)
                self.stats.root_arm_gap_samples += 1
                self.stats.root_q_gap_sum += q_gap
                self.stats.root_q_gap_histogram[_gap_bucket(q_gap)] += 1
                self.stats.root_visit_gap_sum += visit_gap
                self.stats.root_visit_gap_histogram[_gap_bucket(visit_gap)] += 1
        # --- the in-tree opponent's arm (H4) ------------------------------
        opponent_choice = _leading_choice(arms.opponent_visit_share)
        if opponent_choice is not None:
            self.stats.opponent_top_arm_decisions += 1
        opponent_prior_choice = (
            _leading_choice(arms.opponent_prior_share)
            if self._config.use_opponent_priors and arms.opponent_prior_share
            else None
        )
        if opponent_prior_choice is not None:
            self.stats.opponent_prior_arm_decisions += 1
        row = {
            "battle_id": str(getattr(context, "battle_id", "?")),
            "round": getattr(context, "decision_round_index", None),
            "seat": str(getattr(context, "player_id", "?")),
            "model_argmax": model_action,
            "search_argmax": search_action_index,
            "model_override": model_override,
            "unmeasured_cause": cause,
            "model_choice": model_choice,
            # The two arms the search ranked first and second, with the values it
            # ranked them on, in the ACTING seat's frame. This is H2's raw datum;
            # the histograms above are its summary.
            "top_arms": [
                {
                    "move": choice,
                    "visit_share": round(aggregated[choice] / worlds, 6),
                    "q": (
                        None if arms.arm_q.get(choice) is None
                        else round(arms.arm_q[choice], 6)
                    ),
                }
                for choice in leaders
            ],
            # H4's predictor side. `opponent_prior_arm` is present only when the
            # opponent seat was actually priced from the model -- see
            # `_aggregate_root_arms` on why uniform priors are refused here.
            "opponent_top_arm": opponent_choice,
            "opponent_prior_arm": opponent_prior_choice,
        }
        # FREE features, for the static depth rule. Kept separate from the labels above
        # by the `f_` prefix: a rule may only be fitted on these, because only these are
        # knowable before the search that produced everything else.
        row.update(free_decision_features(
            context,
            getattr(self, "_ladder_sims_override", None) or int(self._config.search_sims),
            arms.prior_share,
        ))
        # OCCUPANCY, the label the rule is fitted AGAINST, pooled over this decision's
        # searched worlds. Distinct from `max_depth_reached`, which is a MAX and so
        # cannot tell a filled depth from a single deep line.
        occupancy: Counter = Counter()
        for record in world_runs:
            for depth, count in enumerate(record.get("_depth_occupancy") or []):
                if count:
                    occupancy[depth] += int(count)
        if occupancy:
            row["depth_occupancy"] = {
                str(depth): occupancy[depth] for depth in sorted(occupancy)
            }
        if len(self.stats.root_decision_rows) < _ROOT_DECISION_ROWS:
            self.stats.root_decision_rows.append(row)
        else:
            # Non-zero means the per-decision block is TRUNCATED -- the
            # aggregates above are not, so an H4 join over a truncated shard must
            # use its own row count as the denominator, never
            # `searched_decisions`.
            self.stats.root_decision_rows_dropped += 1
        return {
            "model_argmax": model_action,
            "search_argmax": search_action_index,
            "model_override": model_override,
            "unmeasured_cause": cause,
            "model_choice": model_choice,
            "root_q_gap": None if q_gap is None else round(q_gap, 6),
            "root_visit_gap": None if visit_gap is None else round(visit_gap, 6),
            "opponent_top_arm": opponent_choice,
        }

    def _record_override_address(
        self,
        context: PolicyContext,
        model_action: int,
        search_action_index: int,
        model_choice: Optional[str],
    ) -> None:
        """Retain a forkable address for one disagreement, or count the overflow.

        SEPARATE from `root_decision_rows` even though those carry the same
        fields, and for the reason `fallback_samples` is keyed per class: both
        stores truncate first-N, so a run long enough to fill the row block would
        keep only the disagreements that happened early. A store that fills ONLY
        on the rare event cannot be crowded out by the common one.
        """
        # STAGED when a ladder is running, so a rung the ladder is about to discard
        # cannot consume a cap slot. Appending eagerly meant a 4-rung decision that
        # overrode on every rung burned four slots, charged three of them to
        # `..._addresses_dropped`, and could lose the WINNING rung's address while
        # free slots remained -- measured at 62 pre-existing addresses. Found in
        # review. The cap is applied at commit time in `_search_ladder`.
        address = {
            # (battle_id, round, seat) is what replay needs -- the battle id
            # carries the seed, so the fork probe can reach this exact decision.
            "battle_id": str(getattr(context, "battle_id", "?")),
            "round": getattr(context, "decision_round_index", None),
            "seat": str(getattr(context, "player_id", "?")),
            "model_argmax": model_action,
            "search_argmax": search_action_index,
            "model_choice": model_choice,
        }
        staging = getattr(self, "_ladder_pending_addresses", None)
        if staging is not None:
            staging.append(address)
            return
        self._commit_override_address(address)

    def _commit_override_address(self, address: dict[str, Any]) -> None:
        """Apply the first-N cap to one address. The single point that can drop."""
        if len(self.stats.override_disagreements) >= _OVERRIDE_DISAGREEMENT_ADDRESSES:
            # Non-zero means the sample is INCOMPLETE, and truncated toward the
            # shard's early battles; `model_override_decisions` remains the count,
            # so the invariant a reader can rely on is
            #   len(override_disagreements) + override_disagreement_addresses_dropped
            #     == model_override_decisions
            # and NOT `len == count`, which the cap makes unsatisfiable.
            self.stats.override_disagreement_addresses_dropped += 1
            return
        self.stats.override_disagreements.append(address)

    # Gen 3 pool's only recharge move; the recharge turn itself is public.
    _RECHARGE_MOVES = frozenset({"hyperbeam"})

    def _recharging_slots(self, context: PolicyContext) -> tuple[str, ...]:
        """Slots publicly forced to recharge THIS turn (Hyper Beam landed last round).

        BOTH SIDES, since the self side went live. Our own slot comes from
        ``self_must_recharge`` and the opponent's from ``opponent_must_recharge`` -- two keys of
        the ONE parser ``must_recharge`` tracker, published per seat. The notes below describe
        the opponent side, whose reconstruction fallback predates the tracker.

        CORRECTED: the self side is no longer tracker-only. ``_feature_pack_metadata`` publishes
        ``self_must_recharge`` under the v4 schemas ALONE (deliberately -- an always-present key
        changed world seeding for the v2.2/v3 arms in flight), so on every earlier schema our own
        recharge turn carried no lock, the world got no ``mustrecharge`` volatile, and
        ``_require_world_reproduces_trap`` refused the request's disclosed ``trapped`` flag with
        ``self_request_state_unsupported``. The recharge turn was unsearchable on those schemas.

        The self side does have a second proof, and it is not a reconstruction:
        ``self_recharge_from_action_candidates`` reads the request's own legal choice set off the
        UNGATED ``action_candidates`` metadata. It is schema-independent, so it closes the gap
        without republishing the pack, and it is mirrorable from a recorded corpus row -- see
        ``scripts/fidelity_gate_events.py::production_recharging_slots``, which must stay a
        faithful mirror of this function or the four fidelity gates stop measuring it.

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
        #
        # SECOND PROOF, unioned rather than preferred: the request's own legal choice set, read
        # off the UNGATED `action_candidates` metadata. `getMoveRequestData` sets `trapped: true`
        # and `getMoves` collapses the moveset to the lone `recharge` pseudo-move exactly when
        # `mustrecharge` is held, so "the request offers nothing but recharge" is the SAME fact
        # the tracker reports, disclosed directly to us. A union rather than a fallback because
        # it cannot lose a lock the tracker found, and there is no reading of that request under
        # which the mon is free. See `self_recharge_from_action_candidates` for why the metadata
        # lane and not the raw request -- the gate harness has to be able to mirror this.
        #
        # It is not the `opponent_must_recharge is False` case in reverse: that False is a
        # negative proof about a seat whose request we cannot see, and the weaker reconstruction
        # must not overrule it. Here both inputs are positive proofs about OUR seat.
        self_slot: tuple[str, ...] = ()
        if self_recharge_from_action_candidates(observation_metadata):
            self_slot = (context.player_id,)
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

    def _choice_vocabulary(self, context: PolicyContext) -> Optional[_ChoiceVocabulary]:
        """The request's action space, indexed the way an engine display names it.

        Extracted from `_map_choices` for ONE reason: the override telemetry has
        to translate a SECOND engine display -- the model's prior argmax -- into
        an action index, and `_map_choices` cannot be called twice per decision
        without its counters double-counting. `unmapped_choices` and
        `choices_unmapped_causes` are campaign stop-condition terms
        (`choices_unmapped` is required at zero independently of the fallback
        rate), and the early-stop path's probe call already inflates the second
        one -- a documented wart. A telemetry probe that fired on EVERY searched
        decision would make both counters unreadable, so the translation is pure
        and the counting stays in `_map_choices`.
        """
        candidates = context.observation.metadata.get("action_candidates")
        # `str` and `bytes` ARE Sequences, so `isinstance(candidates, Sequence)` alone lets a
        # stringified metadata field walk straight past this guard and land in the POLICY
        # bucket below -- defeating the purpose of the one token that exists to say "this is
        # plumbing, not a game state". Review found exactly that.
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
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

        # The request's SUBSTITUTED Struggle, and only that. See the `_ENGINE_FORCED_NO_MOVE_IDS`
        # translation below for what it is translated to and why; this is the admission test,
        # kept beside the two counters it is built from.
        #
        # Showdown substitutes Struggle at exactly one site -- `sim/pokemon.ts`
        # `getMoveRequestData`, the `else if (!moves.length)` arm -- so a substituted Struggle
        # is ALWAYS the request's only move, and a `struggle` candidate sitting beside another
        # legal move is a real Struggle MOVE SLOT, not the pseudo-move. `local_showdown.py`'s
        # `_request_reports_only_struggle` separates the two by the absent `pp`/`maxpp` fields
        # and says of that check: "the check is what makes that a checked fact rather than an
        # assumed one". `_action_candidate_metadata` (`showdown.py:7563`) publishes no `pp`, so
        # the mirror available here is the count: exactly one legal move, spelled `struggle`.
        # Review MEASURED the unguarded version absorbing a genuine mismatch -- a
        # `gen3customgame` Blissey with moves `("Struggle", "Soft-Boiled")` mapped "No Move"
        # onto the Struggle slot while `softboiled` was legal, which is precisely the
        # `all_unmapped_legality_mismatch` this class is named for.
        #
        # NO LEGAL SWITCH is required too, and that clause is about OBSERVABILITY, not
        # legality. With a bench the request offers `['struggle', 'switch:X']`; the engine
        # proposing `MoveChoice::None` there means its world sees no switch where the request
        # has one, and pre-translation that disagreement was counted in `unmapped_choices`
        # (measured: `{"No Move": 3.0, "switch shedinja": 1.0}` mapped to the SWITCH and logged
        # the miss). Translating it would keep the decision searched and erase the only trace.
        # A campaign whose stop condition is `choices_unmapped == 0` cannot afford a path that
        # reaches zero by becoming unobservable, so that shape deliberately still misses.
        #
        # What survives both clauses is the request that offers ONE action, and it is Struggle.
        # There the translation cannot change which action is taken -- there is nothing else to
        # take -- it only stops a pure naming difference from being booked as a refusal.
        #
        # No separate `hidden_power_index is None` clause: the loop above writes EVERY legal
        # move id into `move_index_by_id`, including the Hidden Power one it additionally
        # remembers under its own name, so a legal Hidden Power beside Struggle already fails
        # the one-key test. A clause for it was written, and the null-world runner scored
        # dropping it EQUIVALENT against a 14-case differential battery -- an unfalsifiable
        # guard, removed rather than shipped. `test_a_legal_hidden_power_beside_struggle_
        # blocks_the_translation` pins the behaviour instead of the redundant clause.
        forced_struggle_index: Optional[int] = None
        if not any_legal_switch and list(move_index_by_id) == [_STRUGGLE_REQUEST_MOVE_ID]:
            forced_struggle_index = move_index_by_id[_STRUGGLE_REQUEST_MOVE_ID]

        return _ChoiceVocabulary(
            move_index_by_id=move_index_by_id,
            hidden_power_index=hidden_power_index,
            switch_index_by_species=switch_index_by_species,
            switch_index_by_canonical=switch_index_by_canonical,
            forced_struggle_index=forced_struggle_index,
            any_legal_move=any_legal_move,
            any_legal_switch=any_legal_switch,
        )

    def _map_choices(
        self, context: PolicyContext, aggregated: Mapping[str, float]
    ) -> Optional[int]:
        vocabulary = self._choice_vocabulary(context)
        if vocabulary is None:
            self.stats.choices_unmapped_causes[_CAUSE_NO_ACTION_CANDIDATES] += 1
            return None
        best_index: Optional[int] = None
        best_weight = 0.0
        mapped_any = False
        for choice, weight in aggregated.items():
            index = vocabulary.action_index(choice)
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
                    any_legal_move=vocabulary.any_legal_move,
                    any_legal_switch=vocabulary.any_legal_switch,
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
            "fpu_reduction": config.fpu_reduction,
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
