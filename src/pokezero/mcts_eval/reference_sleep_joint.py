"""Opt-in bounded rejection from Q_H(T) times open-episode sleep evidence.

Q_H is the existing full-team proposal, not a catalog posterior. Public actions
are given. Closed episodes have no likelihood weight. Conditional on a full
team and the supported public trait timeline, distinct open episodes have
independent duration/passive-cure draws. Unknown evidence refuses the decision;
it is never treated as zero mass or silently omitted.
"""
from __future__ import annotations

from fractions import Fraction
import random
import re
from typing import Callable, Mapping

from .paper_reference import ReferenceRefusal
from .reference_sleep_evidence import SleepTraits, qualify_sleep_episode, qualify_sleep_traits
from .reference_sleep_history import PublicSleepLedger


def opening_sleep_traits(parties) -> dict[str, SleepTraits]:
    result = {}
    for side, team in parties.items():
        if side not in ("p1", "p2"):
            raise ReferenceRefusal("joint sleep needs resolved party sides")
        for mon in team:
            key = side + ":" + re.sub(r"[^a-z0-9]", "", mon.species.lower())
            if key in result or not mon.ability:
                raise ReferenceRefusal("joint sleep needs unique original species and abilities")
            result[key] = SleepTraits(mon.ability, mon.item or "")
    return result


def joint_sleep_proposal(ledger: PublicSleepLedger, traits: Mapping[str, SleepTraits],
                         rng: random.Random, *, check_deadline: Callable[[], None]):
    """Return assignments or rejection, with exact likelihood accounting.

    Accepting with product evidence probability BEFORE drawing conditional
    timers gives Q_H(T)P(z|T)1{E(T,z)} without inventing team prior masses.
    """
    qualified = []
    likelihood = Fraction(1)
    evidence = []
    check_deadline()
    compatibility = qualify_sleep_traits(ledger, traits)
    if compatibility.status == "unresolved":
        raise ReferenceRefusal("joint sleep trait history unresolved: " + compatibility.reason)
    if compatibility.status == "incompatible":
        return None, dict(evidence=[], likelihood=[0, 1], accepted=False, reason=compatibility.reason)
    # The direct Showdown materializer retains original abilities. Unlike the
    # native engine path it does NOT consume the replay's revealedAbility row.
    # Refuse unresolved runtime representation, never silently restore an
    # original ability after the qualifier reasoned with a copied one.
    for key, current in compatibility.current_traits:
        original = re.sub(r"[^a-z0-9]", "", traits[key].ability.lower())
        if current.ability != original:
            raise ReferenceRefusal("joint sleep current ability is not represented by the materializer")
    for index, episode in enumerate(ledger.episodes):
        if episode.closed:
            continue
        check_deadline()
        row = qualify_sleep_episode(ledger, traits, episode_index=index, at_action_boundary=True)
        probability = row.evidence_probability
        evidence.append(dict(episode=index, sleeper=episode.key, status=row.status, reason=row.reason,
            probability=None if probability is None else [probability.numerator, probability.denominator]))
        if row.status == "unresolved" or probability is None:
            error = ReferenceRefusal("joint sleep evidence unresolved: " + row.reason)
            error.sampling_diagnostic = {"sleep_evidence": evidence}
            raise error
        likelihood *= probability
        if row.status == "compatible":
            qualified.append((episode, row.posterior))
    check_deadline()
    receipt = dict(evidence=evidence, likelihood=[likelihood.numerator, likelihood.denominator])
    # Integer rejection is exact, including tiny passive-survival likelihoods.
    if rng.randrange(likelihood.denominator) >= likelihood.numerator:
        return None, dict(receipt, accepted=False)
    assignments = []
    for episode, posterior in qualified:
        # Independent child stream per episode, not reused for team rejection.
        outcome = posterior.draw(random.Random(rng.getrandbits(128)))
        if not outcome.asleep:
            raise ReferenceRefusal("open sleep episode produced an awake latent state")
        def identity(key):
            side, species = key.split(":", 1)
            return dict(side=side, species=species)
        assignments.append(dict(kind=episode.kind, sleeper=identity(episode.key),
            source=identity(episode.source_key), time=outcome.time, skippedTime=outcome.skipped_time))
    check_deadline()
    return assignments, dict(receipt, accepted=True)
