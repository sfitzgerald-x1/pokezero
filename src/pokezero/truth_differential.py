"""Truth-injection differential: run the full consumer chain on the TRUE world.

GOAL.md premise 1 -- *the true world always exists and is always constructible*
-- has been an argument. This module makes it a measurement.

In seeded self-play on ``LocalShowdownEnv`` the harness generated BOTH teams, so
the true hidden state is known at every decision. This module injects it through
the ``fixed_override`` constructor hook that ``EngineMctsPolicy`` already exposes
(``engine_search.EngineMctsPolicy.__init__``; consumed in ``_search`` where it
bypasses belief sampling), and then runs the ordinary consumer chain on that one
world: construction -> render/fold -> crate search -> choice mapping.

**Every refusal of the truth is a defect, definitionally.** There is no capture
to interpret afterwards and no mechanism to theorise: the failing predicate and
the address that tripped it are in hand at the moment of refusal.

What is and is not de-censored
------------------------------
Production stops a DECISION at its first refusing stage and files ONE
``fallback_reasons`` literal, and ``model.rs`` aborts a world at its FIRST
attribution-unsafe branch. That is why the historical queue's depth is unknowable
from inside (report 4 section 3.4). Offline injection does not have to inherit
all of it, and this module deliberately does not:

* **Per decision, every predicate is reported, not the first.** The probe reads
  the whole per-decision delta of ``world_failure_reasons``,
  ``fallback_reasons``, ``choices_unmapped_causes`` and ``unmapped_choices``, so
  a decision that trips three predicates reports three.
* **ONE stage downstream of a refusal is still attempted:** choice mapping, and
  only as a *consistency cross-check*. :func:`probe_choice_mapping` calls
  ``_map_choices`` even when no world survived, under its own
  ``probe:choices_unmapped:*`` key -- but it is **silent by construction on a healthy
  chain** and **blind to engine-proposed-choice mismatch** (Q4/Transform, PP
  exhaustion, Encore/Disable). Read its docstring before quoting a zero from it, and
  do NOT use it to discharge PLAN section 5's ``choices_unmapped = 0`` trigger.
  Nothing else is attempted: with no constructed state there is no search to run.
* **The pre-abort render inventory is kept.** ``_absorb_aborted_lossy_subcases``
  already recovers every lossy subcase a world observed *before* it aborted, and
  those land in ``lossy_subcase_renders``.
* **Multi-seed repeats widen the abort channel.** ``repeats`` re-runs the same
  truth world with different search seeds, which explores different chance
  branches; the union of abort labels is reported. UNTESTED and unused: every
  published run is ``repeats=1``.

**FOUR censoring seams remain.** An earlier revision of this docstring listed two
and asserted a ``probe_choice_mapping`` that did not exist -- one grep hit, the
docstring itself. The function is real as of this revision; the seam list below is
the corrected one. Independent review measured the gap with compound forcings on the
real runner, 39 decisions each: ``--force construct,unmapped`` and
``--force abort,unmapped`` each reported exactly ONE substantive predicate and no
``choices_unmapped``, and ``--force construct,abort`` reported only the construction
one. Production's first-refuser structure was intact ACROSS STAGES.

1. **Construction hides the search stages.** With no constructed state there is no
   world to search. Not removable by instrumentation; it is the shape of the chain.
2. **Within ONE ``world_battle_spec`` call** the first ``EngineWorldUnsupported``
   still hides any later construction blocker for that world. Removing it requires
   editing the 68 raise sites; external instrumentation cannot turn a ``raise``
   into a continue.
3. **Within ONE native search** the crate aborts the world at its first unsafe
   branch. ``repeats`` mitigates, it does not eliminate.
4. **``_fold_broken`` is STICKY per ``(battle, seat)``.** One live-fold break blinds
   the truth arm for the rest of that battle: later decisions refuse at
   ``live_fold_broken`` before construction is attempted. Measured on the control
   block, where 4 refused decisions are 2 root events.

What survives of the de-censoring claim, at the strength the measurements support:
the probe reports the whole per-decision delta of every counter it reads AND adds an
independent choice-mapping reading, so it is strictly less censored than production
-- but it is **not total per decision**. The damage is bounded and measurable: on the
published census, 12 refusals in 52,140 decisions, so at most 12 decisions can hide a
second predicate. The inventory is total **up to at most 12 masked predicates**.

Units, kept apart on purpose (report 4 section 9.2 / plan 4 reporting rules)
---------------------------------------------------------------------------
``world_failure_reasons`` counts WORLDS. ``fallback_reasons`` counts DECISIONS.
``lossy_subcase_renders`` counts BRANCH RENDERS. :class:`TruthDecisionRecord`
keeps them in three separate fields and :func:`aggregate_records` never co-ranks
them in one table.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from .determinization import _self_team_from_metadata_result
from .engine_search import EngineMctsConfig, EngineMctsPolicy
from .env import BattleStartOverride
from .policy import PolicyContext, PolicyDecision
from .showdown import _pokemon_metadata, _self_team_from_request
from .showdown_fixture import pack_team

__all__ = [
    "STAGES",
    "TruthRefusal",
    "TruthDecisionRecord",
    "TruthWorldBuilder",
    "TruthDifferentialProbe",
    "aggregate_records",
    "identity_witness",
    "mechanism_family",
    "probe_policy_config",
    "stage_for_predicate",
]


# --- taxonomy ----------------------------------------------------------------

#: Consumer-chain stages, in the order the truth passes through them.
STAGES = (
    # The harness could not build the true world at all. An INSTRUMENT gap, not
    # a defect in the tree under test -- counted and reported separately so a
    # broken truth source can never read as "the truth was accepted".
    "truth_source",
    # engine_search._search: _advance_live_fold returned None (model mode only).
    "live_fold",
    # engine_search._search: world_battle_spec / build_poke_engine_state.
    "construction",
    # engine_search._search_model: _root_inputs_json / FoldState.from_payload.
    "root_inputs",
    # engine_search._search_model.run_world: the native search raised. Includes
    # the tree.rs attribution-unsafe abort, which is only compiled live under
    # `--features model` (report 4 section 4.1).
    "crate_search",
    # engine_search._map_choices returned None.
    "choice_mapping",
    # The decision-level fallback literal (a closed 7-token set).
    "decision",
)

_CRATE_ABORT_MARKER = "attribution-unsafe renderer branch rejected before"


def stage_for_predicate(predicate: str) -> str:
    """Map a raw ``world_failure_reasons`` key to its consumer-chain stage."""

    if predicate.startswith("crate_search: "):
        return "crate_search"
    if predicate.startswith("root_inputs: "):
        return "root_inputs"
    if predicate.startswith("belief_sample: "):
        # Only reachable on the PRODUCTION arm; the truth arm never samples.
        return "construction"
    return "construction"


def mechanism_family(stage: str, predicate: str) -> str:
    """Bucket a predicate by the CHANNEL that produced it.

    Deliberately channel-derived rather than a guess at plan 4 section 4's
    mechanism families (*guard wider than its producer*, *consistency constraint
    the sampler ignores*, *plumbing/precedence*, *renderer expressivity*). Which
    of those a predicate belongs to is a judgement read off the consumer's actual
    inputs -- report 4 section 2.1 found all four captured mechanisms were wrong
    on contact, every one of them corrected only after someone dumped those
    inputs. A string classifier here would manufacture exactly that kind of
    wrong-on-contact label. The channel is objective; the family is assigned by
    review, seeded from the channel.
    """

    if stage == "truth_source":
        return "instrument:truth_source"
    if stage == "live_fold":
        return "plumbing:live_fold"
    if stage == "root_inputs":
        return "plumbing:root_inputs"
    if stage == "choice_mapping":
        return "choice_mapping"
    if stage == "decision":
        return "decision_literal"
    if stage == "crate_search":
        if _CRATE_ABORT_MARKER in predicate:
            return "renderer:attribution_unsafe"
        return "crate_search:other"
    # construction: the reason slug before the first ':' is already the campaign's
    # actionable bucket (`_world_failure_key` bounds its cardinality).
    slug = predicate.split(":", 1)[0].strip()
    if slug.startswith("belief_sample"):
        return "sampler:belief_sample"
    if slug in {
        "attract_patch_unavailable",
        "move_trap_patch_unavailable",
    } or slug.startswith("engine_capability_unavailable"):
        return "engine_capability"
    return f"construction:{slug}"


# --- records -----------------------------------------------------------------


@dataclass(frozen=True)
class TruthRefusal:
    """One predicate that rejected the TRUE world at one decision.

    ``count`` is the raw per-decision delta of the counter the predicate came
    from, so its UNIT is the unit of that counter -- worlds for
    ``world_failure_reasons``, decisions for ``fallback_reasons``. Never summed
    across stages.
    """

    stage: str
    predicate: str
    family: str
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "predicate": self.predicate,
            "family": self.family,
            "count": self.count,
        }


@dataclass
class TruthDecisionRecord:
    """The truth arm's verdict at one decision, plus the production arm's."""

    battle_id: str
    seed: int
    seat: str
    round: int
    turn: int

    # --- truth arm ---
    truth_available: bool = True
    truth_unavailable_reason: str | None = None
    #: True when the truth arm fell back, i.e. the true world was REFUSED.
    truth_rejected: bool = False
    truth_fallback_reason: str | None = None
    refusals: list[TruthRefusal] = field(default_factory=list)
    #: BRANCH RENDERS observed by the truth world (informational, never a refusal).
    lossy_subcase_renders: dict[str, int] = field(default_factory=dict)
    truth_worlds_constructed: int = 0
    truth_worlds_searched: int = 0
    truth_repeats: int = 1

    # --- production arm (belief sampling), same decision ---
    #: DECISIONS: production fell back here.
    production_fallback_reason: str | None = None
    #: Belief-sampling attempts that produced no completion, per reason. WORLDS.
    production_sample_failures: dict[str, int] = field(default_factory=dict)
    production_worlds_attempted: int = 0
    production_worlds_constructed: int = 0
    #: The isolated residual of plan 4 section 4: the truth constructs, but belief
    #: sampling found NO consistent completion in `worlds * sample_retry_factor`
    #: tries. A conditioning problem, not a guard problem -- never folded into
    #: the truth-rejection rate.
    sampler_search_failure: bool = False

    #: Bounded state for the exemplar store; only attached to the first record
    #: that carries a given predicate.
    exemplar: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "battle_id": self.battle_id,
            "seed": self.seed,
            "seat": self.seat,
            "round": self.round,
            "turn": self.turn,
            "truth_available": self.truth_available,
            "truth_unavailable_reason": self.truth_unavailable_reason,
            "truth_rejected": self.truth_rejected,
            "truth_fallback_reason": self.truth_fallback_reason,
            "refusals": [refusal.to_dict() for refusal in self.refusals],
            "lossy_subcase_renders": dict(self.lossy_subcase_renders),
            "truth_worlds_constructed": self.truth_worlds_constructed,
            "truth_worlds_searched": self.truth_worlds_searched,
            "truth_repeats": self.truth_repeats,
            "production_fallback_reason": self.production_fallback_reason,
            "production_sample_failures": dict(self.production_sample_failures),
            "production_worlds_attempted": self.production_worlds_attempted,
            "production_worlds_constructed": self.production_worlds_constructed,
            "sampler_search_failure": self.sampler_search_failure,
            "exemplar": self.exemplar,
        }


# --- the truth source --------------------------------------------------------


class TruthWorldBuilder:
    """Build the TRUE ``BattleStartOverride`` for a battle, once per battle.

    BOTH halves are rebuilt from the battle-opening requests, but the self half is
    by construction the content production already uses: production's own
    ``_root_self_team_payload`` reads the same opening snapshot through the same
    ``_self_team_from_metadata_result``. So the only CONTENT difference from a
    belief sample is the hidden half, which is what makes every refusal a refusal
    of the hidden half's truth.

    That equality is MEASURED, not asserted -- 233/233 decisions across 3 battles
    byte-identical; see ``selfhalf_check.py`` in the census run directory. An
    earlier revision of this docstring claimed the builder substituted "only the
    opponent half", which described the intent and not the code.

    The rows come from ``LocalShowdownEnv._first_requests`` -- the BATTLE-OPENING
    request per seat -- not from a live observation. Showdown reorders
    ``side.pokemon[]`` active-first as the battle runs (measured; see
    ``local_showdown`` "position is not stable across requests"), and
    ``determinization._root_self_team_payload`` takes the earliest snapshot for
    exactly that reason. Taking the opening request also avoids baking a
    Trace/Skill-Swap-mutated *current* ability into the packed set, because the
    metadata path prefers ``row["ability"]`` over ``row["baseAbility"]``.
    """

    def __init__(self, env: Any, *, set_source: Any, team_size: int = 6) -> None:
        self._env = env
        self._set_source = set_source
        self._team_size = team_size
        self._cache: dict[str, tuple[dict[str, str] | None, str | None]] = {}

    def reset(self) -> None:
        self._cache.clear()

    def _opening_rows(self, slot: str) -> list[Mapping[str, Any]] | None:
        requests = getattr(self._env, "_first_requests", None)
        if not isinstance(requests, Mapping):
            return None
        request = requests.get(slot)
        if request is None:
            return None
        rows = [_pokemon_metadata(mon) for mon in _self_team_from_request(request, slot)]
        return [row for row in rows if row is not None]

    def packed_teams(self, battle_id: str) -> tuple[dict[str, str] | None, str | None]:
        """Return ``({slot: packed}, None)`` or ``(None, failure_reason)``."""

        cached = self._cache.get(battle_id)
        if cached is not None:
            return cached
        packed: dict[str, str] = {}
        failure: str | None = None
        for slot in ("p1", "p2"):
            rows = self._opening_rows(slot)
            if rows is None or len(rows) != self._team_size:
                failure = f"opening request missing or short for {slot}"
                break
            team, reason = _self_team_from_metadata_result(
                rows, team_size=self._team_size, set_source=self._set_source
            )
            if team is None:
                failure = f"{slot}: {reason or 'unknown'}"
                break
            packed[slot] = pack_team(team)
        result = (None, failure) if failure else (packed, None)
        self._cache[battle_id] = result
        return result

    def override_for(
        self, context: PolicyContext
    ) -> tuple[BattleStartOverride | None, str | None]:
        battle_id = str(getattr(context, "battle_id", "?"))
        packed, failure = self.packed_teams(battle_id)
        if packed is None:
            return None, failure
        try:
            return (
                BattleStartOverride(
                    player_teams=dict(packed),
                    observation_format_id=context.format_id,
                ),
                None,
            )
        except Exception as error:  # noqa: BLE001 - instrument gap, never a crash
            return None, f"{type(error).__name__}: {error}"


# --- the probe ---------------------------------------------------------------

_COUNTER_FIELDS = (
    "world_failure_reasons",
    "fallback_reasons",
    "choices_unmapped_causes",
    "unmapped_choices",
    "lossy_subcase_renders",
)
_SCALAR_FIELDS = (
    "worlds_attempted",
    "worlds_constructed",
    "worlds_searched",
    "attribution_unsafe_renders",
    "fallback_decisions",
    "searched_decisions",
)


def _snapshot(stats: Any) -> dict[str, Any]:
    snap: dict[str, Any] = {name: dict(getattr(stats, name)) for name in _COUNTER_FIELDS}
    snap.update({name: int(getattr(stats, name)) for name in _SCALAR_FIELDS})
    return snap


def _counter_delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    return {
        key: count - before.get(key, 0)
        for key, count in after.items()
        if count - before.get(key, 0) > 0
    }


def probe_policy_config(production: EngineMctsConfig, *, sims: int | None = None) -> EngineMctsConfig:
    """The truth arm's config: ONE world, ONE attempt, everything else shared.

    ``worlds=1`` because there is exactly one truth. ``sample_retry_factor=1``
    because retrying a *fixed* override just re-runs the identical construction
    and would multiply one refusal into four identical ``world_failure_reasons``
    entries -- an inflated frequency for the queue, from the instrument.
    ``early_stop`` is forced off: a stopped world takes a different code path
    through ``_search_model`` and the census must measure one fixed chain.
    """

    import dataclasses

    resolved_sims = int(sims) if sims else production.search_sims
    return dataclasses.replace(
        production,
        worlds=1,
        sample_retry_factor=1,
        early_stop=False,
        search_sims=resolved_sims,
        # `search_batch <= search_sims` is a config invariant. Clamping here rather
        # than making the caller do it: a budget sweep that lowers sims below the
        # default batch otherwise dies in `__post_init__`, which reads as "that
        # budget is unsupported" when it is only unspelled.
        search_batch=min(production.search_batch, resolved_sims),
    )


class TruthDifferentialProbe:
    """Policy wrapper: the production policy plays, the truth arm is measured.

    Modelled on ``engine_search._ArgmaxComparePolicy``, which already establishes
    the "primary drives the game, reference is also asked on the same context"
    shape in this codebase. The production arm returns the decision, so the
    trajectory is exactly the trajectory production would have produced; the
    truth arm never influences play.
    """

    def __init__(
        self,
        *,
        primary: Any,
        truth_policy: EngineMctsPolicy,
        truth_builder: TruthWorldBuilder,
        records: list[TruthDecisionRecord],
        seed: int,
        repeats: int = 1,
        exemplar_store: MutableMapping[str, dict[str, Any]] | None = None,
        probe_enabled: Callable[[PolicyContext], bool] | None = None,
    ) -> None:
        self.primary = primary
        self.truth_policy = truth_policy
        self.truth_builder = truth_builder
        self.records = records
        self.seed = int(seed)
        self.repeats = max(1, int(repeats))
        self.exemplar_store: MutableMapping[str, dict[str, Any]] = (
            {} if exemplar_store is None else exemplar_store
        )
        self._probe_enabled = probe_enabled
        #: Instrument faults. Non-empty means the reading below is NOT clean.
        self.errors: list[str] = []
        self.probed_decisions = 0

    # -- Policy protocol -----------------------------------------------------

    @property
    def policy_id(self) -> str:
        return getattr(self.primary, "policy_id", "truth-differential")

    @property
    def stats(self) -> Any:
        return self.primary.stats

    def select_action(self, observation: Any, *, rng: random.Random) -> PolicyDecision:
        return self.primary.select_action(observation, rng=rng)

    def select_action_with_context(
        self, context: PolicyContext, *, rng: random.Random
    ) -> PolicyDecision:
        before = _snapshot(self.primary.stats)
        decision = self.primary.select_action_with_context(context, rng=rng)
        after = _snapshot(self.primary.stats)
        try:
            self._probe(context, before, after)
        except Exception as error:  # noqa: BLE001 - never break the run over telemetry
            self.errors.append(
                f"{getattr(context, 'battle_id', '?')}"
                f"/{getattr(context, 'decision_round_index', '?')}"
                f"/{getattr(context, 'player_id', '?')}: {type(error).__name__}: {error}"
            )
        return decision

    # -- the measurement -----------------------------------------------------

    def _probe(
        self,
        context: PolicyContext,
        prod_before: Mapping[str, Any],
        prod_after: Mapping[str, Any],
    ) -> None:
        replay = getattr(
            getattr(context, "public_materialization_state", None), "replay", None
        )
        record = TruthDecisionRecord(
            battle_id=str(getattr(context, "battle_id", "?")),
            seed=self.seed,
            seat=str(getattr(context, "player_id", "?")),
            # `or -1` is WRONG here and shipped: `0 or -1` is -1, so every round-0
            # decision was filed at round -1. Measured on the published census:
            # 1,462 of 52,140 records (2.8%) carried -1 and NONE carried 0. Exemplars
            # are advertised as replayable from `(seed, seat, round)`, and -1 is not an
            # address. Explicit None test, pinned by
            # `RoundIndexTests.test_round_zero_is_recorded_as_zero`.
            round=_round_index(context),
            turn=int(getattr(replay, "turn_number", 0) or 0),
            truth_repeats=self.repeats,
        )
        self._fill_production_arm(record, prod_before, prod_after)

        if self._probe_enabled is not None and not self._probe_enabled(context):
            record.truth_available = False
            record.truth_unavailable_reason = "probe_disabled_by_sampling_policy"
            self.records.append(record)
            return

        override, failure = self.truth_builder.override_for(context)
        if override is None:
            record.truth_available = False
            record.truth_unavailable_reason = failure or "unknown"
            self.records.append(record)
            return

        self.probed_decisions += 1
        self.truth_policy._fixed_override = override  # noqa: SLF001 - the documented hook
        refusals: dict[tuple[str, str], int] = {}
        lossy: Counter[str] = Counter()
        rejected = False
        fallback_reason: str | None = None
        constructed = 0
        searched = 0
        for repeat in range(self.repeats):
            before = _snapshot(self.truth_policy.stats)
            probe_rng = random.Random(
                f"{record.battle_id}|{record.seat}|{record.round}|{repeat}"
            )
            self.truth_policy.select_action_with_context(context, rng=probe_rng)
            after = _snapshot(self.truth_policy.stats)
            constructed = max(
                constructed, after["worlds_constructed"] - before["worlds_constructed"]
            )
            searched = max(
                searched, after["worlds_searched"] - before["worlds_searched"]
            )
            for key, count in _counter_delta(
                before["world_failure_reasons"], after["world_failure_reasons"]
            ).items():
                stage = stage_for_predicate(key)
                slot = (stage, key)
                refusals[slot] = max(refusals.get(slot, 0), count)
            for key, count in _counter_delta(
                before["fallback_reasons"], after["fallback_reasons"]
            ).items():
                rejected = True
                fallback_reason = fallback_reason or key
                slot = ("decision", f"fallback:{key}")
                refusals[slot] = max(refusals.get(slot, 0), count)
            for key, count in _counter_delta(
                before["choices_unmapped_causes"], after["choices_unmapped_causes"]
            ).items():
                slot = ("choice_mapping", f"choices_unmapped_cause:{key}")
                refusals[slot] = max(refusals.get(slot, 0), count)
            for key, count in _counter_delta(
                before["lossy_subcase_renders"], after["lossy_subcase_renders"]
            ).items():
                lossy[key] = max(lossy[key], count)

        # CROSS-STAGE: ask the mapping stage even when nothing survived to map.
        # Without this a decision that refused upstream reports nothing about choice
        # mapping, and "0 choices_unmapped" then means "not observed" for exactly the
        # decisions most likely to have a second defect.
        mapping_cause = probe_choice_mapping(self.truth_policy, context, None)
        if mapping_cause is not None:
            refusals[("choice_mapping", f"probe:choices_unmapped:{mapping_cause}")] = 1

        record.truth_rejected = rejected
        record.truth_fallback_reason = fallback_reason
        record.truth_worlds_constructed = constructed
        record.truth_worlds_searched = searched
        record.lossy_subcase_renders = dict(lossy)
        record.refusals = [
            TruthRefusal(
                stage=stage,
                predicate=predicate,
                family=mechanism_family(stage, predicate),
                count=count,
            )
            for (stage, predicate), count in sorted(refusals.items())
        ]
        if record.refusals:
            record.exemplar = self._maybe_exemplar(context, record, override)
        self.records.append(record)

    def _fill_production_arm(
        self,
        record: TruthDecisionRecord,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
    ) -> None:
        world_delta = _counter_delta(
            before["world_failure_reasons"], after["world_failure_reasons"]
        )
        fallback_delta = _counter_delta(
            before["fallback_reasons"], after["fallback_reasons"]
        )
        attempted = after["worlds_attempted"] - before["worlds_attempted"]
        constructed = after["worlds_constructed"] - before["worlds_constructed"]
        sample_failures = {
            key[len("belief_sample: ") :]: count
            for key, count in world_delta.items()
            if key.startswith("belief_sample: ")
        }
        record.production_fallback_reason = next(iter(fallback_delta), None)
        record.production_sample_failures = sample_failures
        record.production_worlds_attempted = attempted
        record.production_worlds_constructed = constructed
        # The residual family, isolated: EVERY attempt was a sampling failure and
        # nothing constructed. A decision that also tripped a construction guard
        # is a guard problem with a sampling symptom and does NOT count here.
        record.sampler_search_failure = bool(
            attempted > 0
            and constructed == 0
            and sample_failures
            and sum(sample_failures.values()) == sum(world_delta.values())
        )

    def _maybe_exemplar(
        self,
        context: PolicyContext,
        record: TruthDecisionRecord,
        override: BattleStartOverride,
    ) -> dict[str, Any] | None:
        """One exemplar per predicate, first occurrence wins.

        The packed teams are included in full: with ``(seed, seat, round)`` they
        are the whole replay input, so an inventory row reproduces from the row
        rather than from a description of it.
        """

        novel = [
            refusal.predicate
            for refusal in record.refusals
            if refusal.predicate not in self.exemplar_store
        ]
        if not novel:
            return None
        metadata = getattr(context.observation, "metadata", None)
        metadata = metadata if isinstance(metadata, Mapping) else {}
        payload = {
            "battle_id": record.battle_id,
            "seed": record.seed,
            "seat": record.seat,
            "round": record.round,
            "turn": record.turn,
            "predicates": [refusal.predicate for refusal in record.refusals],
            "truth_packed_teams": dict(override.player_teams),
            "self_active": _active_summary(metadata.get("self_team")),
            "opponent_active": _active_summary(metadata.get("opponent_team")),
            "action_candidates": _bounded_list(metadata.get("action_candidates"), 24),
        }
        for predicate in novel:
            self.exemplar_store[predicate] = payload
        return payload


def _round_index(context: Any) -> int:
    """The decision round, with 0 preserved. See the call site for what `or` cost."""

    value = getattr(context, "decision_round_index", None)
    return -1 if value is None else int(value)


def _request_legal_choices(context: PolicyContext) -> tuple[str, ...]:
    """The request's legal set, spelled the way the engine spells its choices.

    Delegated to `fallback_replay`, which already derives this for the refusal
    recorder, so the probe and the recorder cannot drift on what "legal" means.
    """

    try:
        from .fallback_replay import _request_legal_choices as _impl

        return tuple(_impl(context))
    except Exception:  # noqa: BLE001 - a probe must never break the run
        return ()


def probe_choice_mapping(policy: Any, context: PolicyContext, aggregated: Any) -> str | None:
    """Cross-check that the REQUEST's admitted legal set is mappable.

    **What this measures, stated at the strength it supports.** It offers
    ``_map_choices`` the request's own legal choices and reports whether the mapper
    accepts them. On a healthy chain it is **silent by construction**: it feeds
    ``fallback_replay._request_legal_choices``, whose docstring says it *"mirrors
    `_map_choices`'s admission rule exactly"*, into ``_map_choices``, which builds its
    index map from the same ``action_candidates``. Independent review instrumented it
    per choice over 4 games -- ``decisions=306, offered_total=1876,
    offered_that_map_individually=1876, decisions_where_ALL_offered_choices_map=306,
    min ratio 1.000``: **no observed degree of freedom.** A zero from this probe
    across a census therefore restates that two functions implement one admission
    rule; it is a consistency cross-check, not a measurement of the class.

    **BLIND to the production shape of the class.** Production's ``choices_unmapped``
    fires when the choice the ENGINE searched cannot be mapped -- Q4/Transform, PP
    exhaustion, an Encore/Disable legality mismatch. This probe only ever offers the
    request's own choices, so it cannot see any of them. Demonstrated: a search-side
    mapping failure produces **39 ``fallback:choices_unmapped`` and 0 probe keys**,
    the probe silent throughout.

    **Therefore it does NOT discharge PLAN section 5's ``choices_unmapped = 0``
    era-launch trigger.** That trigger is measured by the PRODUCTION path's
    ``fallback_reasons["choices_unmapped"]`` and ``choices_unmapped_causes`` only.
    Reading it off this probe would be wrong in exactly the Q4 shape.

    What it does add: production reaches ``_map_choices`` only when a world survived,
    so a decision refusing upstream previously reported nothing here at all. The probe
    asks anyway, under a distinct ``probe:choices_unmapped:*`` key so it can never be
    confused with a production ``choices_unmapped``.

    Returns the ``_CHOICES_UNMAPPED_CAUSES`` token when the request could not be
    mapped, else None. Side-effect-free with respect to the decision: nothing here
    feeds an action.

    NOT a full de-censoring, and the module docstring says so. Construction failure
    still hides the crate stage (there is no state to search), and the crate still
    aborts a world at its first unsafe branch.
    """

    from collections import Counter as _Counter

    probe_weights = aggregated
    if not probe_weights:
        # NOT an empty Counter. An empty aggregate makes `_map_choices` answer
        # "there was nothing to map", which is a fact about the CALL and not about
        # the request -- shipped that way for one revision and it fired on
        # 3,231 of 3,231 decisions, i.e. it measured nothing. Ask the question that
        # has an answer instead: offer the REQUEST's own legal choices, in the
        # engine's vocabulary, and see whether the mapper accepts them. A refusal
        # then means the request and the engine genuinely disagree.
        legal = _request_legal_choices(context)
        if not legal:
            return None  # nothing to ask; silence beats a false positive
        probe_weights = _Counter({choice: 1.0 for choice in legal})

    before = dict(policy.stats.choices_unmapped_causes)
    try:
        mapped = policy._map_choices(context, probe_weights)  # noqa: SLF001
    except Exception as error:  # noqa: BLE001 - a probe must never break the run
        return f"probe_raised:{type(error).__name__}"
    after = dict(policy.stats.choices_unmapped_causes)
    if mapped is not None:
        return None
    new = _counter_delta(before, after)
    return next(iter(new), _CAUSE_UNCLASSIFIED_PROBE)


#: Emitted when `_map_choices` returned None but registered no cause token.
_CAUSE_UNCLASSIFIED_PROBE = "probe_unclassified_cause"


def _active_summary(team: Any) -> dict[str, Any] | None:
    if not isinstance(team, Sequence):
        return None
    for row in team:
        if isinstance(row, Mapping) and row.get("active"):
            return {
                key: row.get(key)
                for key in ("species", "condition", "status", "item", "ability")
            }
    return None


def _bounded_list(value: Any, limit: int) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item)[:80] for item in list(value)[:limit]]


# --- aggregation -------------------------------------------------------------


def aggregate_records(records: Sequence[TruthDecisionRecord | Mapping[str, Any]]) -> dict[str, Any]:
    """Roll shard records into the census inventory.

    Three separately-keyed tables, never one: ``predicates`` (mixed units, each
    row carries its own unit), ``lossy_subcase_renders`` (BRANCH RENDERS) and the
    headline rates (DECISIONS).
    """

    rows = [r if isinstance(r, Mapping) else r.to_dict() for r in records]
    probed = 0
    unavailable: Counter[str] = Counter()
    rejected = 0
    sampler_failures = 0
    production_fallbacks = 0
    decisions = len(rows)
    predicate_decisions: Counter[str] = Counter()
    predicate_units: Counter[str] = Counter()
    predicate_stage: dict[str, str] = {}
    predicate_family: dict[str, str] = {}
    predicate_exemplar: dict[str, Any] = {}
    lossy: Counter[str] = Counter()
    fallback_literals: Counter[str] = Counter()
    production_fallback_literals: Counter[str] = Counter()
    sampler_reasons: Counter[str] = Counter()
    battles: set[str] = set()

    for row in rows:
        battles.add(str(row.get("battle_id")))
        if row.get("production_fallback_reason"):
            production_fallbacks += 1
            production_fallback_literals[str(row["production_fallback_reason"])] += 1
        if row.get("sampler_search_failure"):
            sampler_failures += 1
            for reason, count in (row.get("production_sample_failures") or {}).items():
                sampler_reasons[str(reason)] += int(count)
        if not row.get("truth_available", True):
            unavailable[str(row.get("truth_unavailable_reason") or "unknown")] += 1
            continue
        probed += 1
        if row.get("truth_rejected"):
            rejected += 1
            fallback_literals[str(row.get("truth_fallback_reason") or "unknown")] += 1
        for name, count in (row.get("lossy_subcase_renders") or {}).items():
            lossy[str(name)] += int(count)
        for refusal in row.get("refusals") or []:
            predicate = str(refusal["predicate"])
            predicate_decisions[predicate] += 1
            predicate_units[predicate] += int(refusal.get("count", 1))
            predicate_stage[predicate] = str(refusal["stage"])
            predicate_family[predicate] = str(refusal["family"])
        exemplar = row.get("exemplar")
        if exemplar:
            for predicate in exemplar.get("predicates", []):
                predicate_exemplar.setdefault(str(predicate), exemplar)

    return {
        "decisions_seen": decisions,
        "battles": len(battles),
        "truth_probed_decisions": probed,
        "truth_unavailable_decisions": sum(unavailable.values()),
        "truth_unavailable_reasons": dict(unavailable),
        "truth_rejected_decisions": rejected,
        "truth_rejection_rate": (rejected / probed) if probed else None,
        "truth_fallback_literals": dict(fallback_literals),
        "distinct_open_predicates": len(predicate_decisions),
        "sampler_search_failure_decisions": sampler_failures,
        "sampler_search_failure_rate": (sampler_failures / decisions) if decisions else None,
        "sampler_search_failure_reasons": dict(sampler_reasons),
        "production_fallback_decisions": production_fallbacks,
        "production_fallback_rate": (production_fallbacks / decisions) if decisions else None,
        "production_fallback_literals": dict(production_fallback_literals),
        "predicates": [
            {
                "predicate": predicate,
                "stage": predicate_stage[predicate],
                "family": predicate_family[predicate],
                # DECISIONS on which this predicate rejected the truth.
                "decisions": count,
                # The raw counter delta summed. Its unit is the source counter's
                # unit (worlds for construction/crate keys, decisions for
                # fallback literals) -- never compare across stages.
                "counter_units": predicate_units[predicate],
                "exemplar": predicate_exemplar.get(predicate),
            }
            for predicate, count in predicate_decisions.most_common()
        ],
        # BRANCH RENDERS. Reported in its own table for exactly the reason plan 4
        # keeps the three units apart.
        "lossy_subcase_renders": dict(lossy.most_common()),
    }


# --- identity witness --------------------------------------------------------

#: A symbol that exists ONLY in the tree that carries this module. Printing
#: `__file__` alone does not prove which SOURCE is loaded -- a stale `.pyc` has
#: the right `__file__` and the wrong bytes (report 4 section 4.2 case 2, where a
#: size-preserving line reorder was silently reused). Probing for a name is a
#: content check that a stale byte-compile cannot pass.
CONTENT_FINGERPRINT_SYMBOL = "TruthDifferentialProbe"


def identity_witness() -> dict[str, Any]:
    """Which tree is actually loaded, read from the LOADED modules.

    Never infer the arm from the command line (report 4 section 4.2: four
    distinct ways an A/B compared a tree to itself, all four reporting success
    while measuring one thing twice). Callers should print this in-process AND
    from a child spawned in a neutral cwd, and compare.
    """

    import sys

    import pokezero
    from pokezero import engine_search, truth_differential

    witness: dict[str, Any] = {
        "sys_executable": sys.executable,
        "sys_path_head": list(sys.path[:4]),
        "pokezero_file": pokezero.__file__,
        "engine_search_file": engine_search.__file__,
        "truth_differential_file": truth_differential.__file__,
        "truth_differential_present": hasattr(
            truth_differential, CONTENT_FINGERPRINT_SYMBOL
        ),
        "engine_search_fixed_override_hook": "fixed_override"
        in getattr(engine_search.EngineMctsPolicy.__init__, "__code__").co_varnames,
        "source_sha256": {},
    }
    for name, module in (
        ("truth_differential", truth_differential),
        ("engine_search", engine_search),
    ):
        path = getattr(module, "__file__", None)
        if path:
            try:
                with open(path, "rb") as handle:
                    witness["source_sha256"][name] = hashlib.sha256(
                        handle.read()
                    ).hexdigest()[:16]
            except OSError as error:  # pragma: no cover - diagnostics only
                witness["source_sha256"][name] = f"unreadable: {error}"
    try:
        import pokezero_search

        witness["pokezero_search_file"] = pokezero_search.__file__
        witness["pokezero_search_model_feature"] = bool(
            getattr(pokezero_search, "MODEL_FEATURE_ENABLED", False)
        )
        witness["pokezero_search_so_sha256"] = _extension_hash(pokezero_search)
    except Exception as error:  # noqa: BLE001 - the witness must never crash the run
        witness["pokezero_search_file"] = f"unavailable: {type(error).__name__}: {error}"
        witness["pokezero_search_model_feature"] = None
    try:
        import torch

        witness["torch_version"] = torch.__version__
    except Exception as error:  # noqa: BLE001
        witness["torch_version"] = f"unavailable: {type(error).__name__}: {error}"
    return witness


def _extension_hash(module: Any) -> str:
    """SHA-256 of the compiled crate, not of its Python shim."""

    import pathlib

    root = pathlib.Path(getattr(module, "__file__", "")).parent
    candidates = sorted(root.glob("*.so")) + sorted(root.glob("*.pyd"))
    if not candidates:
        return "no extension module found"
    digest = hashlib.sha256()
    for path in candidates:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def witness_json() -> str:
    return json.dumps(identity_witness(), indent=2, sort_keys=True)


if __name__ == "__main__":  # pragma: no cover - the neutral-cwd child entrypoint
    print(witness_json())
