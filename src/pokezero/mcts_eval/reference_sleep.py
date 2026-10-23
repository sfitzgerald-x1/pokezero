"""Exact finite sleep-history contract; not integrated into scientific collection.

Pinned Gen 3 conditions draw induced time uniformly from 2..5; Rest fixes it
at 3. Before each move opportunity, Early Bird subtracts two, otherwise one.
An observed sleeping attempt must leave positive time. Sleep Talk/Snore add
one skipped unit, an ordinary sleeping attempt clears that run, and reentry
refunds it. Filter EVERY observation, not just the final net decrement: a
later refund cannot resurrect a timer that would already have woken.

The caller supplies complete public-history evidence and ability hypotheses
with explicit prior masses. Do not assign equal posterior ability weights,
or sample a timer conditional on each ability while ignoring its evidence
likelihood. The joint normalization below includes that likelihood. Unknown
ability-history or attempt classification must be resolved or refused by the
caller, never coerced to ordinary/non-Early-Bird evidence.

This module does not extract protocol evidence, select a hidden-team proposal
law, restore a simulator, or bind Sleep Clause sources. Those integration
requirements remain outstanding. In particular, current sampled-team draws
must not be given invented finite prior masses merely to call this function.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import lcm
import random

from .paper_reference import ReferenceRefusal


@dataclass(frozen=True)
class SleepAttempt:
    early_bird: bool
    sleep_usable: bool
    stayed_asleep: bool

    def __post_init__(self):
        if any(type(value) is not bool for value in
               (self.early_bird, self.sleep_usable, self.stayed_asleep)):
            raise ReferenceRefusal("sleep attempt needs resolved boolean evidence")


@dataclass(frozen=True)
class SleepReentry:
    """A publicly observed switch-in; elapsed benched turns do not tick."""


@dataclass(frozen=True)
class SleepCure:
    """An explicit cure, distinct from a natural wake opportunity."""


SleepEvent = SleepAttempt | SleepReentry | SleepCure


@dataclass(frozen=True)
class SleepHypothesis:
    label: str
    prior_mass: Fraction
    events: tuple[SleepEvent, ...]

    def __post_init__(self):
        if (type(self.label) is not str or not self.label
                or type(self.prior_mass) is not Fraction or self.prior_mass <= 0
                or type(self.events) is not tuple
                or any(type(event) not in (SleepAttempt, SleepReentry, SleepCure)
                       for event in self.events)):
            raise ReferenceRefusal("sleep hypothesis needs explicit prior and complete typed history")


@dataclass(frozen=True)
class SleepOutcome:
    hypothesis: str
    initial_time: int
    time: int
    skipped_time: int
    probability: Fraction

    @property
    def asleep(self) -> bool:
        return self.time > 0


@dataclass(frozen=True)
class SleepPosterior:
    outcomes: tuple[SleepOutcome, ...]
    evidence_probability: Fraction

    def draw(self, rng: random.Random) -> SleepOutcome:
        """Exact rational sampling; no floating-point probability quantization."""
        denominator = lcm(*(row.probability.denominator for row in self.outcomes))
        weights = [row.probability.numerator * (denominator // row.probability.denominator)
                   for row in self.outcomes]
        draw = rng.randrange(sum(weights))
        for row, weight in zip(self.outcomes, weights):
            if draw < weight:
                return row
            draw -= weight
        raise AssertionError("normalized sleep distribution has no selected outcome")


def _advance(initial: int, events: tuple[SleepEvent, ...]) -> tuple[int, int] | None:
    time, skipped = initial, 0
    for event in events:
        if time == 0:
            raise ReferenceRefusal("sleep episode continues after an observed wake or cure")
        if type(event) is SleepReentry:
            time += skipped
            skipped = 0
        elif type(event) is SleepCure:
            time, skipped = 0, 0
        else:
            time -= 2 if event.early_bird else 1
            if event.stayed_asleep != (time > 0):
                return None
            if time <= 0:
                time, skipped = 0, 0
            else:
                skipped = skipped + 1 if event.sleep_usable else 0
    return time, skipped


def condition_sleep(hypotheses: tuple[SleepHypothesis, ...], *, kind: str) -> SleepPosterior:
    """Condition joint hypothesis/duration mass on all supplied observations.

    Rejection is explicit if no latent state can generate the public evidence.
    A truncated or unknown observation is not an event and must not be omitted
    to make a contradictory history acceptable.
    """
    if (kind not in ("induced", "rest") or type(kind) is not str
            or type(hypotheses) is not tuple or not hypotheses
            or any(type(row) is not SleepHypothesis for row in hypotheses)
            or len({row.label for row in hypotheses}) != len(hypotheses)):
        raise ReferenceRefusal("invalid sleep origin or hypothesis roster")
    initial_times = (2, 3, 4, 5) if kind == "induced" else (3,)
    prior_total = sum((row.prior_mass for row in hypotheses), Fraction())
    accepted = []
    for row in hypotheses:
        for initial in initial_times:
            state = _advance(initial, row.events)
            if state is not None:
                accepted.append((row.label, initial, *state, row.prior_mass / len(initial_times)))
    mass = sum((row[-1] for row in accepted), Fraction())
    if not mass:
        raise ReferenceRefusal("public sleep history has no compatible latent state")
    return SleepPosterior(tuple(SleepOutcome(label, initial, time, skipped, weight / mass)
                                for label, initial, time, skipped, weight in accepted),
                          mass / prior_total)
