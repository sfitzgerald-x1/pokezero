"""Replay a recorded fallback decision and read the state that refused it.

``fallback_addresses`` finds the address; ``fallback_replay_spec`` turns it into
a runnable spec; this module runs it and hands back the refusal. The point of
the chain is stated in the burndown plan: *diagnosis becomes reading, not
theorizing* -- at the refusal the cause is manifest (which predicate fired, on
which field, against which request flag), so no detector metric and no
acceptance era is needed to find it.

Two halves, deliberately separable.

**The recorder** (:func:`attach_refusal_recorder`) is an instrument, not a
replay. It wraps a live ``EngineMctsPolicy`` and captures, for every fallback
decision the policy takes, the per-decision state that produced it: the
world-failure classes that fired *on that decision* (uncanonicalised, with
counts), the world budget it burned, the engine's proposed choices, the
request's legal set, and the two sets' disagreement. It needs no address and no
shard -- attach it to any harness and the refusals become readable. It is
attached by wrapping bound methods on one instance, so nothing in the producer
changes and a run without it is bit-identical to today.

**The driver** (:func:`replay_fallback`) points the recorder at one recorded
address: run the battle the spec names, then say what happened at
``(round, seat)``. The verdict is a :class:`ReplayOutcome` and it is deliberately
five-valued, because "did the replay work" has more than two honest answers and
the ones in the middle are the informative ones.

On what a replay can and cannot claim
-------------------------------------
``fallback_replay_spec`` establishes that fidelity turns on which MCTS ran, not
on which script wrote the shard. The pokezero crate searches under an explicit
seed (``engine_search.py:1202``) and is reproducible; poke-engine's own
``monte_carlo_tree_search`` is not, **even at a fixed iteration count** --
measured, five runs at ``iterations=4000`` on one captured state gave five
different visit distributions, because its chance-node sampler builds a fresh
unseeded ``rand::rng()`` per sample. That is what the foul-play opponent runs,
and it produced the entire era 61-64 corpus.

This module does not paper over that. A spec whose fidelity is not ``exact``
still replays, and the run still produces real refusals worth reading, but a
diverged trajectory reports :data:`ReplayOutcome.SAME_BATTLE_DIFFERENT_ROUND`
rather than a match, and :attr:`ReplayResult.fidelity_caveat` carries the
spec's evidence verbatim -- so a report cannot quietly upgrade "I found a
refusal of the same class" into "I reproduced the recorded decision".

Measured on a real self-play harness (see the PR): two full 40-seed runs are
byte-identical, and replaying one seed alone in a fresh process reproduces the
recorded address at the recorded round. Measured over eras 61-64: 0 of 1,140
resolvable addresses are exactly reconstructible, because all of them are
foul-play.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence

from .fallback_replay_spec import FIDELITY_EXACT, ReplaySpec

__all__ = [
    "BattleRun",
    "DecisionSnapshot",
    "RefusalRecord",
    "RefusalRecorder",
    "ReplayOutcome",
    "ReplayResult",
    "attach_refusal_recorder",
    "engine_config_for",
    "engine_config_overrides",
    "replay_fallback",
    "rollout_settings",
    "unreachable_round_problem",
]


# --- the captured state -----------------------------------------------------


@dataclass(frozen=True)
class DecisionSnapshot:
    """Cumulative counters at the top of one decision.

    The refusal's *delta* is the only readable quantity: a class's cumulative
    total says it fired somewhere in the run, which is what the shard already
    told us and is exactly the resolution that made the campaign theorise.
    """

    world_failure_reasons: Mapping[str, int]
    unmapped_choices: Mapping[str, int]
    choices_unmapped_causes: Mapping[str, int]
    worlds_attempted: int
    worlds_constructed: int
    worlds_searched: int

    @classmethod
    def of(cls, stats: Any) -> "DecisionSnapshot":
        return cls(
            world_failure_reasons=dict(stats.world_failure_reasons),
            unmapped_choices=dict(stats.unmapped_choices),
            choices_unmapped_causes=dict(stats.choices_unmapped_causes),
            worlds_attempted=stats.worlds_attempted,
            worlds_constructed=stats.worlds_constructed,
            worlds_searched=stats.worlds_searched,
        )


#: Sentinel: the wrapped name was a CLASS method, not an instance attribute.
_ABSENT = object()

#: The only reason `_map_choices` can produce (`engine_search.py:1134`, `:1234`,
#: `:1917`). Used to decide whether a captured aggregate is the engine's actual
#: proposal for the refused decision -- see `RefusalRecord.engine_choices`.
_MAPPING_REASON = "choices_unmapped"


def _delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    return {
        key: count - before.get(key, 0)
        for key, count in after.items()
        if count - before.get(key, 0) > 0
    }


@dataclass(frozen=True)
class RefusalRecord:
    """One fallback decision, with the state that produced it.

    Everything here is per-decision. That is the whole difference between this
    and the shard: the shard records *that* a class fired in a run, and this
    records *which classes fired on the decision that refused*, which is the
    thing a fix is written against.
    """

    # --- the address this decision would be filed under ---
    battle_id: str
    round: int
    seat: str
    reason: str

    #: World-failure classes that fired ON THIS DECISION, raw keys, with counts.
    #: This is the same delta `_fallback` files addresses under
    #: (`engine_search.py:2380-2385`), kept whole instead of reduced to keys.
    world_failures: Mapping[str, int] = field(default_factory=dict)
    #: `(attempted, constructed, searched)` spent on this decision. A decision
    #: that constructed worlds and still refused is a different bug from one
    #: that constructed none, and the reason string alone does not separate them.
    worlds_attempted: int = 0
    worlds_constructed: int = 0
    worlds_searched: int = 0

    #: The engine's proposed choices for the refused decision: choice string ->
    #: aggregated visit share.
    #:
    #: Populated ONLY when the refusal came out of the mapping step
    #: (`reason == "choices_unmapped"`). Not a conservatism -- a correctness
    #: rule. The early-stop lock probe calls
    #: `_map_choices(context, Counter({locked_choice: 1.0}))`
    #: (`engine_search.py:1858-1861`) purely to test a lock, and
    #: `crate_search_failed` / `early_stop_replay_failed` then refuse without a
    #: second call. Reporting the last aggregate unconditionally therefore
    #: printed a SYNTHETIC single-choice probe as "the engine proposed", for a
    #: decision that may have searched zero worlds. Fabricated evidence, in the
    #: artifact this module exists to produce. Every observed call is kept on
    #: :attr:`map_choices_calls` instead, labelled for what it is.
    engine_choices: Mapping[str, float] = field(default_factory=dict)
    #: Every `_map_choices` invocation seen during this decision, in order,
    #: including lock probes. Diagnostic; not "the engine's proposal".
    map_choices_calls: tuple[Mapping[str, float], ...] = ()
    #: The request's legal choices, in the engine's own vocabulary, so the two
    #: sets are directly comparable. Built from
    #: `observation.metadata["action_candidates"]` filtered by
    #: `legal_action_mask` -- the same filter `_map_choices` applies
    #: (`engine_search.py:2279-2284`).
    request_legal_choices: tuple[str, ...] = ()
    #: Choices the engine proposed that the request did not offer. For a
    #: `choices_unmapped` refusal this is the finding: report 3 spent three
    #: corrections theorising about which move it was.
    unmapped_choices: Mapping[str, int] = field(default_factory=dict)
    choices_unmapped_causes: Mapping[str, int] = field(default_factory=dict)

    #: `f"{seed}:{seat}:{round}"` when the harness reseeds per decision
    #: (`foulplay_bridge.py:3541`); `None` under the per-battle stream regime.
    decision_rng_seed: str | None = None
    #: The pre-decision baseline was lost, so every delta on this record may span
    #: more than one decision. Carried on the record rather than only on the
    #: recorder, because the record is what gets serialized and read.
    degraded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "battle_id": self.battle_id,
            "round": self.round,
            "seat": self.seat,
            "reason": self.reason,
            "world_failures": dict(self.world_failures),
            "worlds_attempted": self.worlds_attempted,
            "worlds_constructed": self.worlds_constructed,
            "worlds_searched": self.worlds_searched,
            "engine_choices": dict(self.engine_choices),
            "map_choices_calls": [dict(call) for call in self.map_choices_calls],
            "request_legal_choices": list(self.request_legal_choices),
            "unmapped_choices": dict(self.unmapped_choices),
            "choices_unmapped_causes": dict(self.choices_unmapped_causes),
            "decision_rng_seed": self.decision_rng_seed,
            "degraded": self.degraded,
        }

    @property
    def locates(self) -> tuple[str, int, str]:
        return (self.battle_id, self.round, self.seat)


# --- the recorder -----------------------------------------------------------


def _request_legal_choices(context: Any) -> tuple[str, ...]:
    """The request's legal set, spelled the way the engine spells its choices.

    Mirrors `_map_choices`'s admission rule exactly -- `legal` AND permitted by
    `legal_action_mask` (`engine_search.py:2279-2284`) -- because a set built by
    a *different* rule would make every comparison against `engine_choices` a
    comparison with this function rather than with the request.

    ``normalize_id`` is applied for the same reason and is NOT optional. The
    mapping keys on ``normalize_id(move_id)`` (``engine_search.py:2286``) and
    ``normalize_id(species)`` (``:2293-2295``), and matches the engine's own
    choice string through ``normalize_id`` too (``:2332``, ``:2325``). Without
    it, ``Nidoran-F``, ``Mr. Mime`` and ``Unown-C`` never intersect
    ``engine_choices`` -- so a decision whose mapping SUCCEEDED renders as a
    legality mismatch, on precisely the ``choices_unmapped`` case this record
    exists to read.
    """
    from .dex import normalize_id  # noqa: PLC0415 - keeps the module import-light

    observation = getattr(context, "observation", None)
    metadata = getattr(observation, "metadata", None)
    candidates = metadata.get("action_candidates") if isinstance(metadata, Mapping) else None
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return ()
    # The bare attribute, as `_map_choices` uses it (`engine_search.py:2274`).
    # `getattr(...) or ()` raises "truth value of an array is ambiguous" on an
    # array-like mask -- an instrument that crashes the run it measures.
    mask = getattr(observation, "legal_action_mask", None)
    if mask is None:
        return ()
    choices: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or not candidate.get("legal"):
            continue
        index = candidate.get("action_index")
        if not isinstance(index, int) or not (0 <= index < len(mask)) or not mask[index]:
            continue
        if candidate.get("kind") == "move":
            move_id = normalize_id(str(candidate.get("move_id") or ""))
            if move_id:
                choices.append(move_id)
        elif candidate.get("kind") == "switch":
            pokemon = candidate.get("pokemon")
            species = normalize_id(
                str(pokemon.get("species") or "") if isinstance(pokemon, Mapping) else ""
            )
            if species:
                choices.append(f"switch {species}")
    return tuple(choices)


#: Where the single wrapper layer parks its bookkeeping on the policy.
_HOOK_ATTR = "_pz_refusal_hook"


class _Hook:
    """Exactly ONE wrapper layer per policy, fanning out to N recorders.

    Wrapping per recorder cannot be unwound out of order: with two recorders,
    the second captures the first's wrapper as "the original", so
    ``r1.detach(); r2.detach()`` reinstalls r1's wrapper permanently and every
    later decision feeds a detached recorder. Nesting is not exotic -- a corpus
    sweep that attaches around a run which itself attaches per battle produces
    it -- and it fails silently, which is worse than failing.

    One layer, a list of subscribers, and detach as list removal is
    order-independent by construction.
    """

    _WRAPPED = ("select_action_with_context", "_map_choices", "_fallback")

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.recorders: list["RefusalRecorder"] = []
        # `_ABSENT` distinguishes "there was an instance attribute" from "it was
        # a class method". Restoring the latter with setattr freezes a bound
        # copy onto the instance -- a self-referential cycle that makes the
        # policy unpicklable and a permanent divergence from its class.
        self._saved = {
            name: policy.__dict__.get(name, _ABSENT) for name in self._WRAPPED
        }
        base_select = getattr(policy, "select_action_with_context")
        base_map = getattr(policy, "_map_choices")
        base_fallback = getattr(policy, "_fallback")

        def select_action_with_context(context: Any, *, rng: Any) -> Any:
            for recorder in tuple(self.recorders):
                recorder._guard(recorder._begin_decision)
            return base_select(context, rng=rng)

        def _map_choices(context: Any, aggregated: Mapping[str, float]) -> Any:
            # Copied, not referenced: `aggregated` is a live Counter the search
            # keeps mutating between here and the refusal, so a reference would
            # record the state at report time rather than at call time.
            for recorder in tuple(self.recorders):
                recorder._guard(
                    lambda r=recorder: r._map_choices_calls.append(dict(aggregated))
                )
            return base_map(context, aggregated)

        def _fallback(context: Any, rng: Any, reason: str) -> Any:
            # Recorded BEFORE delegating. Precisely: the real `_fallback`
            # increments `fallback_decisions`, `fallback_reasons` and
            # `fallback_samples` -- NONE of which `DecisionSnapshot` reads, so
            # the ordering is not load-bearing today and a post-call capture
            # would currently produce identical records. It is held against a
            # `_fallback` that starts touching `world_failure_reasons`,
            # `unmapped_choices`, `choices_unmapped_causes` or the `worlds_*`
            # counters, at which point a post-call capture would fold the
            # refusal's own bookkeeping into its evidence. Pinned by
            # `test_capture_happens_before_the_policys_own_fallback_runs`.
            for recorder in tuple(self.recorders):
                recorder._guard(lambda r=recorder: r._record(context, reason))
            return base_fallback(context, rng, reason)

        policy.select_action_with_context = select_action_with_context
        policy._map_choices = _map_choices
        policy._fallback = _fallback
        policy.__dict__[_HOOK_ATTR] = self

    @classmethod
    def for_policy(cls, policy: Any) -> "_Hook":
        hook = policy.__dict__.get(_HOOK_ATTR)
        return hook if isinstance(hook, cls) else cls(policy)

    def add(self, recorder: "RefusalRecorder") -> None:
        self.recorders.append(recorder)

    def remove(self, recorder: "RefusalRecorder") -> None:
        if recorder in self.recorders:
            self.recorders.remove(recorder)
        if self.recorders:
            return
        for name, original in self._saved.items():
            if original is _ABSENT:
                self.policy.__dict__.pop(name, None)
            else:
                setattr(self.policy, name, original)
        self.policy.__dict__.pop(_HOOK_ATTR, None)


class RefusalRecorder:
    """Captures the per-decision state behind every fallback a policy takes.

    Installed by wrapping three bound methods on one ``EngineMctsPolicy``
    instance:

    * ``select_action_with_context`` -- to snapshot the cumulative counters
      before the decision, which is the only way to recover a per-decision
      delta from counters that are otherwise cumulative;
    * ``_map_choices`` -- to keep the ``aggregated`` counter the engine
      proposed, which exists nowhere else once the call returns and is the
      entire content of a ``choices_unmapped`` refusal;
    * ``_fallback`` -- to emit the record.

    Wrapping rather than subclassing is deliberate: the harnesses build their
    own policies, several of them deep inside a script, and a subclass would
    mean a fork of every harness. Detachable, and a detached run is
    bit-identical to an unrecorded one -- no counter is touched, no RNG draw is
    consumed.
    """

    #: Read by `DecisionSnapshot.of`. Validated at attach time, because attach
    #: is the only moment at which failing is free -- the alternative was an
    #: AttributeError at the first decision, which replaces the engine's safe
    #: uniform-legal fallback with a crash in the middle of a run.
    _REQUIRED_STATS = (
        "world_failure_reasons",
        "unmapped_choices",
        "choices_unmapped_causes",
        "worlds_attempted",
        "worlds_constructed",
        "worlds_searched",
    )

    def __init__(self, policy: Any) -> None:
        self._policy = policy
        self._records: list[RefusalRecord] = []
        self._snapshot: DecisionSnapshot | None = None
        self._map_choices_calls: list[Mapping[str, float]] = []
        self._detached = False
        #: Instrument failures, never silently swallowed and never raised into
        #: the run. Non-empty means some record is incomplete.
        self.errors: list[str] = []
        self._attach()

    # -- lifecycle --

    def _validate(self) -> None:
        stats = getattr(self._policy, "stats", None)
        if stats is None:
            raise AttributeError("policy has no `stats`; cannot record refusals")
        absent = [name for name in self._REQUIRED_STATS if not hasattr(stats, name)]
        if absent:
            raise AttributeError(
                f"policy.stats is missing {', '.join(absent)}; the recorder would "
                "attach and then fail inside the decision path"
            )
        for name in _Hook._WRAPPED:
            if not callable(getattr(self._policy, name, None)):
                raise AttributeError(f"policy has no callable {name}")

    def _attach(self) -> None:
        self._validate()
        _Hook.for_policy(self._policy).add(self)

    def _guard(self, action: Any) -> None:
        """Run instrument code so it can never break the run it is measuring.

        A diagnostic that turns a handled refusal into an unhandled exception
        has changed the outcome it exists to explain -- and does it in
        production, since the recorder is meant to be attachable to any harness.
        Failures are collected on `errors` instead, which is checkable.
        """
        try:
            action()
        except Exception as error:  # noqa: BLE001 -- an instrument must not raise
            self.errors.append(f"{type(error).__name__}: {error}")

    def _begin_decision(self) -> None:
        # Cleared FIRST, and unconditionally. If the snapshot below raises under
        # `_guard`, an earlier revision left `self._snapshot` holding the
        # PREVIOUS decision's counters, so the next record's `world_failures` and
        # `worlds_*` spanned two decisions -- the cumulative-vs-delta error this
        # whole module exists to prevent, arriving silently. The same revision
        # never cleared `_map_choices_calls` at all, so a probe from decision N
        # could be serialized into decision N+1's record.
        self._snapshot = None
        self._map_choices_calls = []
        self._snapshot = DecisionSnapshot.of(self._policy.stats)

    def detach(self) -> None:
        """Stop recording. Idempotent, and correct in ANY unwind order."""
        if self._detached:
            return
        self._detached = True
        hook = self._policy.__dict__.get(_HOOK_ATTR)
        if hook is not None:
            hook.remove(self)

    def __enter__(self) -> "RefusalRecorder":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.detach()

    # -- capture --

    def _record(self, context: Any, reason: str) -> None:
        stats = self._policy.stats
        before = self._snapshot
        degraded = before is None
        if before is None:
            # No usable baseline. Emitting a zero-delta record would read as
            # "nothing failed on this decision", which is a stronger and more
            # misleading claim than admitting the baseline was lost.
            before = DecisionSnapshot.of(stats)
            self.errors.append(
                f"no pre-decision snapshot for {getattr(context, 'battle_id', '?')} "
                f"round {getattr(context, 'decision_round_index', '?')}; deltas "
                "on that record are not trustworthy"
            )
        battle_id = str(getattr(context, "battle_id", "?"))
        round_index = getattr(context, "decision_round_index", None)
        seat = str(getattr(context, "player_id", "?"))
        seed = getattr(context, "seed", None)
        rng_seed = (
            f"{seed}:{seat}:{round_index}"
            if isinstance(seed, int) and isinstance(round_index, int)
            else None
        )
        self._records.append(
            RefusalRecord(
                battle_id=battle_id,
                round=round_index if isinstance(round_index, int) else -1,
                seat=seat,
                reason=reason,
                world_failures=_delta(
                    before.world_failure_reasons, stats.world_failure_reasons
                ),
                worlds_attempted=stats.worlds_attempted - before.worlds_attempted,
                worlds_constructed=stats.worlds_constructed - before.worlds_constructed,
                worlds_searched=stats.worlds_searched - before.worlds_searched,
                engine_choices=(
                    dict(self._map_choices_calls[-1])
                    if reason == _MAPPING_REASON and self._map_choices_calls
                    else {}
                ),
                map_choices_calls=tuple(
                    dict(call) for call in self._map_choices_calls
                ),
                request_legal_choices=_request_legal_choices(context),
                unmapped_choices=_delta(before.unmapped_choices, stats.unmapped_choices),
                choices_unmapped_causes=_delta(
                    before.choices_unmapped_causes, stats.choices_unmapped_causes
                ),
                decision_rng_seed=rng_seed,
                degraded=degraded,
            )
        )

    # -- results --

    @property
    def records(self) -> tuple[RefusalRecord, ...]:
        return tuple(self._records)

    def at(self, battle_id: str, round_index: int, seat: str) -> RefusalRecord | None:
        for record in self._records:
            if record.locates == (battle_id, round_index, seat):
                return record
        return None

    def reasons(self) -> Counter:
        return Counter(record.reason for record in self._records)


def attach_refusal_recorder(policy: Any) -> RefusalRecorder:
    """Start recording every fallback ``policy`` takes. See :class:`RefusalRecorder`."""
    return RefusalRecorder(policy)


# --- the driver -------------------------------------------------------------


class ReplayOutcome(str, Enum):
    """What the replay actually established. Five-valued on purpose.

    A two-valued verdict would have to fold ``SAME_BATTLE_DIFFERENT_ROUND`` into
    either "worked" or "failed", and it is neither: it is the signature of a
    trajectory that diverged, which is the expected result for every foul-play
    address and is itself the finding.
    """

    #: A fallback occurred at the recorded (battle, round, seat) with the
    #: recorded reason. The record is the recorded decision.
    #:
    #: READ THE CLASS BEFORE READING TOO MUCH INTO THIS. For a construction-side
    #: refusal -- `no_worlds_constructed`, which is 43 of era 64's 67 resolvable
    #: addresses, i.e. the MAJORITY -- the refusal happens at
    #: `engine_search.py:1101`, before the search reads `search_sims`,
    #: `search_depth`, `c_puct`, `deep_ko_split` or `leaf_eval`. So a match here
    #: establishes that the seed, the id grammar, the resolver, the seat-parity
    #: rule, the opponent policy, the belief-sampling stream and
    #: `worlds`/`sample_retry_factor` were all rebuilt correctly -- and says
    #: NOTHING about the search parameters, which were never consulted.
    REPRODUCED = "reproduced"
    #: A fallback occurred at exactly that decision, but for a different reason.
    #: The decision still refuses; something ahead of it in the order changed.
    REASON_CHANGED = "reason-changed"
    #: The recorded decision did not refuse, but the same battle refused
    #: elsewhere. Under a non-exact fidelity this is the ordinary divergence
    #: signature; under an exact one it is a regression or a build difference.
    SAME_BATTLE_DIFFERENT_ROUND = "same-battle-different-round"
    #: The battle ran and took no fallback at all. Under exact fidelity, this is
    #: what a fix looks like.
    #:
    #: NOTE the limit of this verdict, which the runner cannot see past: a
    #: refusal-free run and a run that ended before reaching `round` are
    #: indistinguishable from the refusal list alone. Distinguishing them needs
    #: the decision count, which no `BattleRunner` currently returns.
    NO_REFUSAL = "no-refusal"
    #: The recorder could not capture reliably, so no verdict is claimed. Never
    #: silently collapsed into `NO_REFUSAL`: a swallowed capture failure produces
    #: an empty record list, which is indistinguishable from a clean run.
    INSTRUMENT_FAILED = "instrument-failed"
    #: Refusals came back, but none of them from the recorded battle. Only
    #: reachable from a runner that replays MORE than the one battle -- a corpus
    #: sweep, or a harness that batches. `rollout_runner` always passes
    #: `spec.battle_id`, so it cannot produce this; that is a property of that
    #: runner, not of the verdict.
    ADDRESS_ABSENT = "address-absent"


@dataclass(frozen=True)
class ReplayResult:
    """The outcome of replaying one address, and everything read while doing it."""

    spec: ReplaySpec
    outcome: ReplayOutcome
    #: The refusal at the recorded address, when there was one.
    record: RefusalRecord | None
    #: Every refusal the replayed battle took, in order. Kept whole: under a
    #: diverged trajectory these are the only real refusals available, and they
    #: are still worth reading.
    all_records: tuple[RefusalRecord, ...]
    #: Set whenever the spec's fidelity is not `exact`. Its presence is what
    #: stops a diverged run being reported as a reproduction.
    fidelity_caveat: str | None = None
    #: Whether the runner was able to report instrument health at all. False
    #: means unknown, not healthy -- see :attr:`BattleRun.health_reported`.
    health_reported: bool = False
    #: Instrument failures during the replay. NON-EMPTY MEANS THE VERDICT IS NOT
    #: TRUSTWORTHY. The recorder deliberately does not raise into the search --
    #: an instrument must not change the run -- but swallowing without surfacing
    #: turned a capture failure into `NO_REFUSAL`, which this enum documents as
    #: "what a fix looks like". `_classify` refuses that combination now.
    instrument_errors: tuple[str, ...] = ()
    #: Things the RUNNER had to assume. Distinct from `instrument_errors`: these
    #: do not invalidate the capture, they qualify what was replayed. Serialized
    #: and printed, because an assumption nobody sees is a default.
    runner_notes: tuple[str, ...] = ()

    @property
    def trustworthy(self) -> bool:
        """True only when health was REPORTED and was good.

        The `health_reported` conjunct is the whole point: an empty error list
        from a runner with no error channel says nothing, and reading it as
        "healthy" is the same defect as reading an empty record list as
        NO_REFUSAL.
        """
        return (
            self.health_reported
            and not self.instrument_errors
            and not any(record.degraded for record in self.all_records)
        )

    @property
    def reproduced(self) -> bool:
        return self.outcome is ReplayOutcome.REPRODUCED

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "outcome": self.outcome.value,
            "record": self.record.to_dict() if self.record else None,
            "all_records": [record.to_dict() for record in self.all_records],
            "fidelity_caveat": self.fidelity_caveat,
            "health_reported": self.health_reported,
            "instrument_errors": list(self.instrument_errors),
            "runner_notes": list(self.runner_notes),
            "trustworthy": self.trustworthy,
        }


@dataclass(frozen=True)
class BattleRun:
    """What a :data:`BattleRunner` hands back: the refusals AND its own health.

    A plain list of records was not enough, and the way it failed is worth
    keeping written down. The errors used to travel as an attribute set on the
    runner callable, which `replay_fallback` read with `getattr`. That survives
    exactly one call shape. Measured:

    ==========================  ==================  ============
    runner                      outcome             trustworthy
    ==========================  ==================  ============
    `runner`                    instrument-failed   False
    `lambda s: runner(s)`       no-refusal          **True**
    `functools.partial(runner)` no-refusal          **True**
    ==========================  ==================  ============

    Any retry wrapper, timing decorator, corpus sweeper or test double reopened
    the hole -- and did not merely lose the errors, it affirmatively reported the
    result as trustworthy and the run as a clean `NO_REFUSAL`, which
    :class:`ReplayOutcome` documents as what a fix looks like. Returning the
    health alongside the records makes composition carry it by default: a
    wrapper that forwards the return value forwards everything.
    """

    records: tuple[RefusalRecord, ...] = ()
    #: Whether this runner is CAPABLE of reporting instrument health, i.e. is it
    #: wired to a recorder at all. Default False, and the coercion of a bare
    #: iterable leaves it False.
    #:
    #: Without it, `trustworthy` said True whenever `instrument_errors` was
    #: empty -- which is exactly the silence-reads-as-OK shape that produced the
    #: composition bug above, one level up. A runner that never had an errors
    #: channel is not evidence that nothing went wrong; it is the absence of
    #: evidence, and :data:`BattleRunner`'s docstring already promised to treat
    #: it that way.
    health_reported: bool = False
    #: Instrument failures. Non-empty blocks the absence verdicts.
    instrument_errors: tuple[str, ...] = ()
    #: Things the runner had to assume. These qualify, they do not invalidate.
    runner_notes: tuple[str, ...] = ()
    #: The search configuration the runner actually built, when it built one.
    #:
    #: Published because nothing else could bind the runner to
    #: :func:`engine_config_for`: replacing that call with an inline
    #: `EngineMctsConfig(...)` that drops the recorded overrides stayed green,
    #: including end-to-end -- construction-side refusals never read a search
    #: parameter, so the recorded address reproduces under a wrong search. The
    #: `RolloutConfig` half was already closed by `runner_notes` being
    #: observable; this gives the engine half the same behavioural hook.
    engine_config: Any = None


#: A callable that runs the battle a spec names and reports what happened.
#: Injected rather than hard-wired: the four writers stand their battles up four
#: different ways (two in-process drivers, one subprocess bridge, one grid
#: script), and binding the verdict logic to any one of them would make it
#: testable only where a Showdown build, a checkpoint and a patched engine are
#: all present.
#:
#: Returning a bare iterable of records is still accepted, for runners that have
#: no health to report -- but such a runner cannot report one, and
#: :func:`replay_fallback` treats its silence as "nothing to say" rather than as
#: "nothing went wrong". Prefer :class:`BattleRun`.
BattleRunner = Callable[[ReplaySpec], "BattleRun | Iterable[RefusalRecord]"]


def _as_battle_run(result: Any) -> BattleRun:
    if isinstance(result, BattleRun):
        return result
    return BattleRun(records=tuple(result))


def _classify(
    spec: ReplaySpec,
    records: Sequence[RefusalRecord],
    instrument_errors: Sequence[str] = (),
) -> tuple[ReplayOutcome, RefusalRecord | None]:
    target = (spec.battle_id, spec.round, spec.seat)
    for record in records:
        if record.locates == target:
            if record.reason == spec.reason:
                return ReplayOutcome.REPRODUCED, record
            return ReplayOutcome.REASON_CHANGED, record
    if instrument_errors:
        # Checked before NO_REFUSAL and only for the ABSENCE cases: a positive
        # match above stands on its own evidence, but "I saw nothing" from a
        # broken instrument is not a measurement, and NO_REFUSAL is documented
        # as what a fix looks like.
        return ReplayOutcome.INSTRUMENT_FAILED, None
    if not records:
        return ReplayOutcome.NO_REFUSAL, None
    if any(record.battle_id == spec.battle_id for record in records):
        return ReplayOutcome.SAME_BATTLE_DIFFERENT_ROUND, None
    return ReplayOutcome.ADDRESS_ABSENT, None


def replay_fallback(spec: ReplaySpec, run_battle: BattleRunner) -> ReplayResult:
    """Replay one recorded address and report what the replay established.

    ``run_battle`` stands the battle up however its harness requires and returns
    the refusals it took; see :data:`BattleRunner`.
    """
    run = _as_battle_run(run_battle(spec))
    records, errors, notes = run.records, run.instrument_errors, run.runner_notes
    health_reported = run.health_reported
    outcome, record = _classify(spec, records, errors)
    caveat = None
    if spec.fidelity != FIDELITY_EXACT:
        # Stated on every non-exact result, including a REPRODUCED one: a match
        # under an unpinned opponent is evidence, not proof, and the difference
        # is exactly what a corpus gate would otherwise launder.
        caveat = (
            f"spec fidelity is {spec.fidelity!r}, not {FIDELITY_EXACT!r}: "
            + "; ".join(spec.fidelity_notes)
        )
    return ReplayResult(
        spec=spec,
        outcome=outcome,
        record=record,
        all_records=records,
        fidelity_caveat=caveat,
        instrument_errors=errors,
        runner_notes=notes,
        health_reported=health_reported,
    )


def format_refusal(record: RefusalRecord) -> str:
    """A human read of one refusal -- the thing the burndown loop is for."""
    lines = [
        f"REFUSAL  {record.battle_id}  round={record.round}  seat={record.seat}",
        f"  reason: {record.reason}",
    ]
    if record.degraded:
        # Without this the zeros below print as a confident measurement -- the
        # same fabricated-evidence shape the lock-probe fix removed. `trustworthy`
        # catches it only through `ReplayResult`, and the recorder is documented
        # as usable with no address, no shard and no driver, where none exists.
        lines.append(
            "  *** DEGRADED: the pre-decision baseline was lost, so every delta "
            "below may span more than one decision ***"
        )
    lines += [
        f"  worlds: attempted={record.worlds_attempted} "
        f"constructed={record.worlds_constructed} searched={record.worlds_searched}",
    ]
    if record.world_failures:
        lines.append("  world-failure classes on THIS decision:")
        for key, count in sorted(
            record.world_failures.items(), key=lambda item: (-item[1], item[0])
        ):
            lines.append(f"    {count:5d}  {key}")
    if record.engine_choices:
        lines.append("  engine proposed:")
        for choice, weight in sorted(
            record.engine_choices.items(), key=lambda item: (-item[1], item[0])
        ):
            lines.append(f"    {weight:8.4f}  {choice}")
    if record.request_legal_choices:
        lines.append(f"  request offered: {', '.join(record.request_legal_choices)}")
    if record.unmapped_choices:
        # The disagreement, printed as a disagreement. Report 3 spent three
        # corrections on a Substitute legality divergence whose two sets were
        # both available at the refusal.
        lines.append("  proposed but NOT offered:")
        for choice, count in sorted(record.unmapped_choices.items()):
            lines.append(f"    {count:5d}  {choice}")
    if record.choices_unmapped_causes:
        lines.append(f"  mapping causes: {dict(record.choices_unmapped_causes)}")
    if record.decision_rng_seed:
        lines.append(f"  decision rng: random.Random({record.decision_rng_seed!r})")
    return "\n".join(lines)


def format_result(result: ReplayResult) -> str:
    spec = result.spec
    lines = [
        f"REPLAY {spec.battle_id} round={spec.round} seat={spec.seat} "
        f"reason={spec.reason}",
        f"  harness={spec.harness} seed={spec.seed} source={spec.source}",
        f"  OUTCOME: {result.outcome.value}",
    ]
    if not result.health_reported:
        lines.append(
            "  HEALTH NOT REPORTED: this runner has no instrument-error channel, "
            "so the absence of errors is unknown, not clean"
        )
    if result.instrument_errors:
        lines.append(
            f"  INSTRUMENT_ERROR ({len(result.instrument_errors)}): this verdict "
            "is not trustworthy"
        )
        for error in result.instrument_errors:
            lines.append(f"    {error}")
    for note in result.runner_notes:
        lines.append(f"  ASSUMED: {note}")
    if result.fidelity_caveat:
        lines.append(f"  CAVEAT: {result.fidelity_caveat}")
    if result.record is not None:
        lines.append("")
        lines.append(format_refusal(result.record))
    elif result.all_records:
        lines.append(
            f"  the recorded decision did not refuse; "
            f"{len(result.all_records)} other refusal(s) in this run:"
        )
        for record in result.all_records:
            lines.append(
                f"    {record.battle_id} r{record.round} {record.seat}: {record.reason}"
            )
    return "\n".join(lines)


def results_to_json(results: Iterable[ReplayResult]) -> str:
    return json.dumps([result.to_dict() for result in results], indent=2, sort_keys=True)


# --- a real runner, for the one harness this PR can stand up ---------------


class UnsupportedHarness(RuntimeError):
    """The spec names a harness :func:`rollout_runner` cannot stand up."""


def engine_config_overrides(spec: ReplaySpec) -> dict[str, Any]:
    """Search settings the spec pins, as `EngineMctsConfig` keyword overrides.

    Split out of :func:`rollout_runner` purely so it is testable without a
    Showdown build -- the falsy-vs-None distinction below is the kind of defect
    that hides behind an integration-only test.

    ``is not None``, never truthiness. A recorded ``c_puct: 0.0`` is a real
    setting (pure-visit selection) and ``deep_ko_split: false`` is a real
    producer flag (`hc_depth_grid.py:107`); a falsy test silently substitutes
    the dataclass defaults 1.4 and True, i.e. replays a different search and
    reports it as the recorded one.
    """
    overrides: dict[str, Any] = {}
    if spec.engine_c_puct is not None:
        overrides["c_puct"] = spec.engine_c_puct
    if spec.deep_ko_split is not None:
        overrides["deep_ko_split"] = spec.deep_ko_split
    return overrides


def engine_config_for(spec: ReplaySpec) -> Any:
    """Build the `EngineMctsConfig` a replay of ``spec`` must search under.

    A named function, not an inline call inside :func:`rollout_runner`, because
    the WIRING needs its own test.

    Under the shipped harness the refusals that actually occur are
    construction-side (`no_worlds_constructed`), and the construction loop
    (`engine_search.py:1049-1101`) reads only `worlds`, `sample_retry_factor`
    and the four `approximate_*` flags before refusing at `:1101`. It does NOT
    read `search_sims`, `search_depth`, `c_puct`, `deep_ko_split`, `leaf_eval`,
    `search_batch` or `model_priors`. So an end-to-end replay reproduces the
    recorded address even when every one of those is silently wrong -- deleting
    the override splat survived the whole integration suite, twice, including
    with non-default recorded values.

    `worlds` is the exception and is consulted: `attempts_budget = worlds *
    sample_retry_factor` at `:1050`, which is why a refusal in that class shows
    exactly `worlds * sample_retry_factor` attempts. An earlier revision of this
    docstring said "before any search parameter is consulted", which is wrong.

    The config is therefore asserted directly, and published on
    :attr:`BattleRun.engine_config` so the runner's use of it is observable too.
    """
    from .engine_search import EngineMctsConfig  # noqa: PLC0415

    return EngineMctsConfig(
        leaf_eval="hp_fraction_crate",
        worlds=spec.engine_worlds,
        search_sims=spec.engine_sims,
        search_depth=spec.engine_depth,
        **engine_config_overrides(spec),
    )


def rollout_settings(
    spec: ReplaySpec, *, default_rounds: int
) -> tuple[dict[str, Any], list[str]]:
    """Battle-level settings from the spec, plus what had to be assumed.

    ``max_decision_rounds`` BOUNDS THE BATTLE, so a defaulted value can end the
    replay before the recorded round is ever reached -- which renders as
    ``NO_REFUSAL``, documented as what a fix looks like. `hc_depth_grid` never
    records it, so for the only shipped harness the default is always taken:
    saying so in a docstring and then defaulting silently was the whole defect.
    `spec.round` is known, so the risk is checkable rather than hypothetical.

    `format_id` uses `is None`, not truthiness -- two functions after
    :func:`engine_config_overrides` condemns that exact pattern.
    """
    caveats: list[str] = []
    rounds = spec.max_decision_rounds
    if rounds is None:
        rounds = default_rounds
        caveats.append(
            f"max_decision_rounds not recorded by this shard; replaying at "
            f"{rounds} against a recorded round of {spec.round}"
        )
    format_id = spec.format_id
    if format_id is None:
        format_id = "gen3randombattle"
        caveats.append("format_id not recorded by this shard; assuming gen3randombattle")
    return {"max_decision_rounds": rounds, "format_id": format_id}, caveats


def unreachable_round_problem(settings: Mapping[str, Any], spec: ReplaySpec) -> str | None:
    """Refuse a replay whose bound cannot reach the recorded round.

    Not a caveat: the recorded decision provably cannot occur, so the run would
    report `NO_REFUSAL` about a decision it never attempted.
    """
    rounds = settings["max_decision_rounds"]
    if isinstance(rounds, int) and spec.round >= rounds:
        return (
            f"recorded round {spec.round} is not reachable under "
            f"max_decision_rounds={rounds}; the replay would report NO_REFUSAL "
            "about a decision it never ran"
        )
    return None


def rollout_runner(
    *,
    showdown_root: str,
    node_binary: str = "node",
    max_decision_rounds: int = 250,  # fallback only; the spec wins when it pins one
) -> BattleRunner:
    """A :data:`BattleRunner` for the in-process ``hc_depth_grid`` harness.

    Deliberately one harness, not four. This is the family whose battle is
    reconstructible (both policies pokezero, both seeded off the battle seed)
    *and* whose leaf evaluator is handcrafted, so it needs no TorchScript
    artifact and no GPU -- which is what makes an end-to-end replay something
    that can actually be run and shown rather than described.

    The other three are refused by name rather than approximated:
    ``mcts_acceptance_h2h`` and ``k0_grid_h2h`` run ``leaf_eval="model"`` and
    need materialised search artifacts, and the foul-play bridge needs a foul-play
    checkout and a subprocess -- and its trajectory does not rebuild anyway.
    Substituting a different leaf evaluator or a different opponent would
    produce a run that looks like a replay and is not one.

    Imports are deferred to call time: the module's recorder half is pure and
    must stay importable in an environment with no Showdown build, no
    ``poke_engine`` and no torch.
    """
    from pathlib import Path  # noqa: PLC0415

    from .collection import run_rollout_record_on_env  # noqa: PLC0415
    from .dex import load_showdown_dex_cached  # noqa: PLC0415
    from .engine_search import EngineMctsPolicy  # noqa: PLC0415
    from .local_showdown import LocalShowdownConfig, LocalShowdownEnv  # noqa: PLC0415
    from .randbat import load_gen3_randbat_source_cached  # noqa: PLC0415
    from .rollout import RolloutConfig  # noqa: PLC0415

    from .fallback_replay_spec import HARNESS_ROLLOUT_HC_GRID  # noqa: PLC0415

    dex = load_showdown_dex_cached(showdown_root)
    set_source = load_gen3_randbat_source_cached(showdown_root)
    env_config = LocalShowdownConfig(
        showdown_root=Path(showdown_root), node_binary=node_binary
    )

    def run(spec: ReplaySpec) -> BattleRun:
        if spec.harness != HARNESS_ROLLOUT_HC_GRID:
            raise UnsupportedHarness(
                f"rollout_runner stands up {HARNESS_ROLLOUT_HC_GRID!r} battles; "
                f"this spec is {spec.harness!r}"
            )
        for name in ("engine_sims", "engine_worlds", "engine_depth"):
            if getattr(spec, name) is None:
                # Filling in a default here would replay a different search and
                # report the result as the recorded one.
                raise UnsupportedHarness(
                    f"spec does not pin {name}; the shard did not record it "
                    f"(missing: {', '.join(spec.missing)})"
                )
        if spec.opponent_policy_id is None:
            raise UnsupportedHarness("spec does not pin the opponent policy")

        from .collection import policy_from_spec  # noqa: PLC0415

        config = engine_config_for(spec)
        candidate = EngineMctsPolicy(
            dex=dex,
            set_source=set_source,
            config=config,
            policy_id=f"engine-mcts-replay-d{spec.engine_depth}-s{spec.engine_sims}",
        )
        settings, notes = rollout_settings(spec, default_rounds=max_decision_rounds)
        unreachable = unreachable_round_problem(settings, spec)
        if unreachable is not None:
            raise UnsupportedHarness(unreachable)
        rollout_config = RolloutConfig(**settings)
        opponent = policy_from_spec(spec.opponent_policy_id)
        # scripts/hc_depth_grid.py:235 -- seat is seed parity, and the spec
        # resolver already refused any address that disagrees with it.
        candidate_seat = "p1" if spec.seed % 2 == 0 else "p2"
        opponent_seat = "p2" if candidate_seat == "p1" else "p1"
        env = LocalShowdownEnv(env_config)
        try:
            with attach_refusal_recorder(candidate) as recorder:
                run_rollout_record_on_env(
                    env=env,
                    policies={candidate_seat: candidate, opponent_seat: opponent},
                    rollout_config=rollout_config,
                    seed=spec.seed,
                    battle_id=spec.battle_id,
                )
                # Returned, not smuggled onto the callable: a recorder whose
                # failures go nowhere turns a broken instrument into a finding,
                # and an attribute on `run` is lost the moment anyone wraps it.
                return BattleRun(
                    records=recorder.records,
                    health_reported=True,
                    instrument_errors=tuple(recorder.errors),
                    runner_notes=tuple(notes),
                    engine_config=config,
                )
        finally:
            # A replay sweep opens one env per address; leaking the node
            # subprocess exhausts file descriptors long before the corpus ends.
            close = getattr(env, "close", None)
            if callable(close):
                close()

    return run
