"""Fail-closed public sleep evidence for ONE explicitly supplied team hypothesis.

No team is drawn or reweighted here. A caller selects the episode explicitly;
the entire transcript still supplies trait history. A unit-mass hypothesis in
the finite timer calculation means conditional on this one supplied team, NOT
a probability assigned to it under a hidden-team proposal.

This is a bounded qualifier, not a public simulator. Unsupported transfers,
unclassified cures and unresolved residual membership return UNRESOLVED, not
zero likelihood. In particular it never derives a natural wake from a bare
cure, or a Shed Skin opportunity merely from a copied ability plus upkeep.
No production caller or collection is enabled by this module.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import re
from typing import Mapping

from ..randbat_vocab import GEN3_RANDBAT_ITEMS
from .paper_reference import ReferenceRefusal
from .reference_sleep import (
    SleepAttempt, SleepCureChance, SleepEvent, SleepHypothesis, SleepPosterior,
    SleepReentry, condition_sleep,
)
from .reference_sleep_history import PublicSleepEpisode, PublicSleepLedger, _key


# Canonical Gen 3 ability ids (Dex ability numbers 1..76). Do not classify an
# unknown ability as mechanically neutral. The oracle test verifies this set.
_ABILITIES = frozenset("""
airlock arenatrap battlearmor blaze chlorophyll clearbody cloudnine colorchange
compoundeyes cutecharm damp drizzle drought earlybird effectspore flamebody
flashfire forecast guts hugepower hustle hypercutter illuminate immunity
innerfocus insomnia intimidate keeneye levitate lightningrod limber liquidooze
magmaarmor magnetpull marvelscale minus naturalcure oblivious overgrow owntempo
pickup plus poisonpoint pressure purepower raindish rockhead roughskin runaway
sandstream sandveil serenegrace shadowtag shedskin shellarmor shielddust
soundproof speedboost static stench stickyhold sturdy suctioncups swarm swiftswim
synchronize thickfat torrent trace truant vitalspirit voltabsorb waterabsorb
waterveil whitesmoke wonderguard
""".split())


def _id(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


_ITEMS = frozenset(_id(item) for item in GEN3_RANDBAT_ITEMS) | {"", "chestoberry"}
_UPDATE_CURES = {"insomnia", "vitalspirit"}
_BERRIES = {"lumberry", "chestoberry"}


@dataclass(frozen=True)
class SleepTraits:
    """Original hypothetical traits, not recovered private simulator truth."""
    ability: str
    item: str = ""


@dataclass(frozen=True)
class SleepEvidenceWitness:
    line: int
    kind: str


@dataclass(frozen=True)
class QualifiedSleepEvidence:
    status: str  # compatible, incompatible, unresolved
    reason: str
    posterior: SleepPosterior | None = None
    witnesses: tuple[SleepEvidenceWitness, ...] = ()
    current_traits: tuple[tuple[str, SleepTraits], ...] = ()

    @property
    def evidence_probability(self) -> Fraction | None:
        if self.status == "incompatible":
            return Fraction(0)
        return self.posterior.evidence_probability if self.posterior is not None else None


class _EvidenceStop(Exception):
    def __init__(self, status: str, reason: str):
        self.status, self.reason = status, reason


def qualify_sleep_episode(
    ledger: PublicSleepLedger, opening_traits: Mapping[str, SleepTraits], *,
    episode_index: int, at_action_boundary: bool,
) -> QualifiedSleepEvidence:
    """Qualify one explicit episode, retaining distinct unknown/zero outcomes.

    Opening keys are nickname-free public ``p1:species``/``p2:species`` ids.
    Supported item inputs are the randbat pool plus Chesto for authored tests.
    Trace/Role Play/Skill Swap/Transform and switch resets are reconstructed;
    an in-episode trait mutation or unsupported transfer remains unresolved.
    Compatibility here concerns the supported sleep evidence, not complete
    team legality or all other mechanics. It does not choose a sampling law.
    """
    if type(episode_index) is not int:
        return QualifiedSleepEvidence("unresolved", "invalid-episode-or-boundary")
    return _qualify(ledger, opening_traits, episode_index=episode_index,
                    at_action_boundary=at_action_boundary)


def qualify_sleep_traits(ledger: PublicSleepLedger, opening_traits: Mapping[str, SleepTraits]):
    """Supported deterministic full-history trait constraints, NO episode weight."""
    return _qualify(ledger, opening_traits, episode_index=None, at_action_boundary=True)


def _qualify(ledger, opening_traits, *, episode_index, at_action_boundary):
    witnesses: list[SleepEvidenceWitness] = []
    events: list[SleepEvent] = []

    def halt(status: str, reason: str):
        raise _EvidenceStop(status, reason)

    def witness(line: int, kind: str):
        witnesses.append(SleepEvidenceWitness(line, kind))

    try:
        if (type(ledger) is not PublicSleepLedger
                or (episode_index is not None and (type(episode_index) is not int
                    or not 0 <= episode_index < len(ledger.episodes)))
                or at_action_boundary is not True or not isinstance(opening_traits, Mapping)):
            halt("unresolved", "invalid-episode-or-boundary")
        episode = (PublicSleepEpisode("", "", "induced", len(ledger.public_lines) + 1, (), True)
                   if episode_index is None else ledger.episodes[episode_index])
        selected = {event.line: event for event in episode.events}
        last = (-1 if episode_index is None else
                episode.events[-1].line if episode.closed else len(ledger.public_lines))
        original = {}
        for key, trait in opening_traits.items():
            if (type(key) is not str or not re.fullmatch(r"p[12]:[a-z0-9]+", key)
                    or type(trait) is not SleepTraits or type(trait.ability) is not str
                    or type(trait.item) is not str):
                halt("unresolved", "invalid-opening-traits")
            value = SleepTraits(_id(trait.ability), _id(trait.item))
            if value.ability not in _ABILITIES or value.item not in _ITEMS:
                halt("unresolved", "unsupported-opening-trait")
            original[key] = value
        current = dict(original)
        active: dict[str, str] = {}
        copied_since_turn: set[str] = set()
        pending_cure: tuple[Fraction, str, int] | None = None
        first_residual = True
        yawn_onset = (episode_index is not None and episode.onset_line > 0 and
            ledger.public_lines[episode.onset_line - 1].split("|")[1:4] ==
            ["-end", ledger.public_lines[episode.onset_line].split("|")[2], "move: Yawn"])
        trick_expected: dict[str, tuple[str, str]] = {}

        def traits(key: str) -> SleepTraits:
            if key not in current:
                halt("unresolved", "missing-opening-identity")
            return current[key]

        def source(fields: list[str]) -> str:
            sources = [field[5:] for field in fields if field.startswith("[of] ")]
            if len(sources) != 1:
                halt("unresolved", "missing-trait-source")
            return _key(sources[0])

        def active_pair(key: str, other: str):
            if (key[:2] == other[:2] or active.get(key[:2]) != key
                    or active.get(other[:2]) != other):
                halt("unresolved", "unproved-active-mutation-participants")

        def chance(line: int, probability: Fraction, survived: bool, kind: str):
            events.append(SleepCureChance(probability, survived))
            witness(line, kind)

        def update_survival(line: int):
            value = traits(episode.key)
            if value.ability in _UPDATE_CURES or value.item in _BERRIES:
                chance(line, Fraction(1), True, "deterministic-update-no-cure")

        for number, raw in enumerate(ledger.public_lines):
            fields = raw.split("|")
            command = fields[1]
            event = selected.get(number)
            inside = episode.onset_line <= number <= last
            # Announced causes are deterministic trait constraints even when
            # the episode is closed and its stochastic likelihood is omitted.
            if command in {"-activate", "-curestatus"} and len(fields) >= 4:
                announced = ([fields[3]] if command == "-activate" else fields[4:])
                for tag in announced:
                    cause = tag.removeprefix("[from] ")
                    if cause in {"ability: Shed Skin", "ability: Natural Cure",
                                 "ability: Insomnia", "ability: Vital Spirit"}:
                        if traits(_key(fields[2])).ability != _id(cause.partition(":")[2]):
                            halt("incompatible", "announced-cure-ability-contradiction")
            if command == "turn":
                copied_since_turn.clear()
            if ((command == "-start" and len(fields) >= 4 and fields[3] == "move: Yawn")
                    or (command == "-status" and len(fields) >= 4 and fields[3] == "slp")):
                # The original ability can block the public precursor even if
                # Skill Swap replaces it before the selected sleep onset.
                if traits(_key(fields[2])).ability in _UPDATE_CURES:
                    halt("incompatible", "public-sleep-immunity-contradiction")
            if command in {"switch", "drag"}:
                key = _key(fields[2])
                species = _id(fields[3].partition(",")[0])
                if key != key[:3] + species:
                    halt("unresolved", "nickname-species-ambiguity")
                traits(key)
                old = active.get(key[:2])
                if event is not None and event.kind == "switch_out":
                    value = traits(episode.key)
                    chance(number, Fraction(value.ability == "naturalcure"), True,
                           "switch-out-natural-cure-check")
                    update_survival(number)
                if old is not None:
                    current[old] = replace(traits(old), ability=original[old].ability)
                current[key] = replace(traits(key), ability=original[key].ability)
                copied_since_turn.discard(key)
                active[key[:2]] = key
            elif command == "-ability" and len(fields) >= 4:
                key, new = _key(fields[2]), _id(fields[3])
                old = traits(key)
                if new not in _ABILITIES:
                    halt("unresolved", "unsupported-ability-reveal")
                tags = fields[4:]
                if "[from] ability: Trace" in tags or "[from] move: Role Play" in tags:
                    other = source(tags)
                    active_pair(key, other)
                    announced_old = _id(fields[4])
                    if announced_old not in _ABILITIES:
                        halt("unresolved", "missing-original-ability-reveal")
                    if announced_old != old.ability:
                        halt("incompatible", "original-ability-reveal-contradiction")
                    if traits(other).ability != new:
                        halt("incompatible", "copied-ability-source-contradiction")
                    if "[from] ability: Trace" in tags:
                        if old.ability != "trace":
                            halt("incompatible", "trace-holder-contradiction")
                        copied_since_turn.add(key)
                    elif new == "wonderguard" or old.ability == new:
                        halt("incompatible", "role-play-contradiction")
                    current[key] = replace(old, ability=new)
                elif any(tag.startswith("[from]") for tag in tags):
                    halt("unresolved", "unsupported-ability-change")
                elif old.ability != new:
                    halt("incompatible", "public-ability-contradiction")
            elif command == "-activate" and len(fields) >= 4 and fields[3] == "Skill Swap":
                key, other = _key(fields[2]), source(fields[4:])
                a, b = traits(key), traits(other)
                active_pair(key, other)
                if inside and episode.key in (key, other):
                    halt("unresolved", "trait-change-inside-selected-episode")
                if a.ability == b.ability or "wonderguard" in (a.ability, b.ability):
                    halt("incompatible", "skill-swap-contradiction")
                current[key], current[other] = replace(a, ability=b.ability), replace(b, ability=a.ability)
            elif command == "-transform" and len(fields) >= 4:
                key, other = _key(fields[2]), _key(fields[3])
                active_pair(key, other)
                current[key] = replace(traits(key), ability=traits(other).ability)
            elif command == "-endability":
                halt("unresolved", "unresolved-ability-ending")
            elif command == "-activate" and len(fields) >= 4 and fields[3] == "move: Trick":
                key, other = _key(fields[2]), source(fields[4:])
                a, b = traits(key), traits(other)
                active_pair(key, other)
                if inside and episode.key in (key, other):
                    halt("unresolved", "trait-change-inside-selected-episode")
                if b.ability == "stickyhold" or not (a.item or b.item):
                    halt("incompatible", "trick-source-contradiction")
                current[key], current[other] = replace(a, item=b.item), replace(b, item=a.item)
                trick_expected = {key: ("-item", b.item) if b.item else ("-enditem", a.item),
                                  other: ("-item", a.item) if a.item else ("-enditem", b.item)}
            elif command in {"-item", "-enditem"} and len(fields) >= 4:
                key, item = _key(fields[2]), _id(fields[3])
                old = traits(key)
                if item not in _ITEMS:
                    halt("unresolved", "unsupported-item")
                if "[from] move: Trick" in fields[4:]:
                    if trick_expected.pop(key, None) != (command, item):
                        halt("incompatible", "trick-item-contradiction")
                elif command == "-item" and not any(tag.startswith("[from]") for tag in fields[4:]):
                    if old.item != item:
                        halt("incompatible", "public-item-contradiction")
                elif command == "-enditem" and not any(tag.startswith("[from] move:") and
                        tag != "[from] move: Knock Off" for tag in fields[4:]):
                    if "[from] move: Knock Off" in fields[4:] and old.ability == "stickyhold":
                        halt("incompatible", "sticky-hold-removal-contradiction")
                    if old.item != item:
                        halt("incompatible", "consumed-item-contradiction")
                    current[key] = replace(old, item="")
                    if inside and key == episode.key and item in _BERRIES and "[eat]" in fields[4:]:
                        pending_cure = (Fraction(1), "berry-cure", number)
                    elif inside and key == episode.key and item in _BERRIES:
                        halt("unresolved", "curing-item-removed-without-cure")
                else:
                    halt("unresolved", "unsupported-item-transfer")

            if event is None:
                continue
            kind = event.kind
            if pending_cure is not None and pending_cure[2] != number and kind != "cure":
                if pending_cure[1] != "heal-bell-check":
                    halt("unresolved", "unfinished-public-cure")
                chance(number, pending_cure[0], True, pending_cure[1])
                pending_cure = None
            if kind == "onset":
                value = traits(episode.key)
                chance(number, Fraction(value.ability in _UPDATE_CURES), True, "onset-immunity-check")
            elif kind == "attempt":
                update_survival(number)
                events.append(SleepAttempt(traits(episode.key).ability == "earlybird", event.sleep_usable, True))
                witness(number, "public-sleep-attempt")
            elif kind == "reentry":
                events.append(SleepReentry())
                witness(number, "public-reentry")
            elif kind == "residual_boundary":
                if pending_cure is not None:
                    if pending_cure[1] != "heal-bell-check":
                        halt("unresolved", "unfinished-public-cure")
                    chance(number, pending_cure[0], True, pending_cure[1])
                    pending_cure = None
                if event.fields == ("active",):
                    update_survival(number)
                    if traits(episode.key).ability == "shedskin" and not (first_residual and yawn_onset):
                        if episode.key in copied_since_turn:
                            halt("unresolved", "copied-residual-membership")
                        chance(number, Fraction(33, 100), True, "stable-shed-skin-no-cure")
                elif event.fields != ("bench",):
                    halt("unresolved", "unknown-residual-location")
                first_residual = False
            elif kind == "activation" and event.fields and event.fields[0] in {
                    "ability: Shed Skin", "ability: Insomnia", "ability: Vital Spirit"}:
                expected = _id(event.fields[0].partition(":")[2])
                if expected == "shedskin" and first_residual and yawn_onset:
                    halt("unresolved", "late-onset-residual-order")
                if traits(episode.key).ability != expected:
                    halt("incompatible", "cure-ability-contradiction")
                pending_cure = (Fraction(33, 100) if expected == "shedskin" else Fraction(1),
                                "observed-ability-cure", number)
            elif kind == "heal_bell":
                pending_cure = (Fraction(traits(episode.key).ability != "soundproof"), "heal-bell-check", number)
            elif kind == "team_cure":
                if "[from] move: Aromatherapy" not in event.fields:
                    halt("unresolved", "unsupported-team-cure")
                chance(number, Fraction(1), False, "public-aromatherapy")
            elif kind == "cure":
                if "[from] ability: Natural Cure" in event.fields:
                    chance(number, Fraction(traits(episode.key).ability == "naturalcure"), False,
                           "public-natural-cure")
                elif pending_cure is not None:
                    chance(number, pending_cure[0], False, pending_cure[1])
                    pending_cure = None
                else:
                    halt("unresolved", "unclassified-cure-not-a-proved-wake")
            elif kind == "faint":
                halt("unresolved", "selected-fainted-episode")
            elif kind in {"ability", "endability", "transform", "item"}:
                halt("unresolved", "trait-change-inside-selected-episode")
        if trick_expected:
            halt("unresolved", "unfinished-item-transfer")
        if pending_cure is not None:
            if pending_cure[1] != "heal-bell-check":
                halt("unresolved", "unfinished-public-cure")
            chance(len(ledger.public_lines), pending_cure[0], True, pending_cure[1])
        if not episode.closed and active.get(episode.key[:2]) == episode.key:
            update_survival(len(ledger.public_lines))
        if episode_index is None:
            return QualifiedSleepEvidence("compatible", "qualified-public-traits",
                                          SleepPosterior((), Fraction(1)), tuple(witnesses),
                                          tuple(sorted(current.items())))
        try:
            posterior = condition_sleep((SleepHypothesis("supplied-team", Fraction(1), tuple(events)),),
                                        kind=episode.kind)
        except ReferenceRefusal:
            halt("incompatible", "no-compatible-sleep-state")
        return QualifiedSleepEvidence("compatible", "qualified-supplied-hypothesis", posterior,
                                      tuple(witnesses), tuple(sorted(current.items())))
    except (_EvidenceStop, ReferenceRefusal, IndexError) as error:
        status = error.status if isinstance(error, _EvidenceStop) else "unresolved"
        reason = error.reason if isinstance(error, _EvidenceStop) else "unresolved-public-identity"
        return QualifiedSleepEvidence(status, reason, witnesses=tuple(witnesses))
