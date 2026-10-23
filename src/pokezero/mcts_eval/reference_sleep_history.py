"""Public sleep episode ledger, not a hidden-state likelihood or sampler.

Keep chronological episodes, including closed ones, and source identity. Do
not infer a natural wake from an untagged cure: berries and ability effects
can emit the same cure line. Ability/item changes, cures and residual
boundaries remain public events for a later likelihood evaluator. This
extractor neither assigns abilities nor certifies proposal support.

Only a caller with a proved action boundary may settle a final unclassified
sleep attempt. Arbitrary transcript prefixes must fail instead. No private
requests, simulator state or historical stopped-world reconstruction enters.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import re

from .paper_reference import ReferenceRefusal


def _key(ident: str) -> str:
    match = re.fullmatch(r"(p[12])(?:a)?: (.+)", ident)
    if match is None:
        raise ReferenceRefusal("sleep ledger needs a resolved singles identity")
    name = re.sub(r"[^a-z0-9]", "", match[2].lower())
    if not name:
        raise ReferenceRefusal("sleep ledger has an empty identity")
    return match[1] + ":" + name


@dataclass(frozen=True)
class PublicSleepEvent:
    kind: str
    line: int
    # Exact PUBLIC fields, not an error message or a private payload.
    fields: tuple[str, ...] = ()
    sleep_usable: bool | None = None


@dataclass(frozen=True)
class PublicSleepEpisode:
    key: str
    source_key: str
    kind: str
    onset_line: int
    events: tuple[PublicSleepEvent, ...]
    closed: bool = False


@dataclass(frozen=True)
class PublicSleepLedger:
    episodes: tuple[PublicSleepEpisode, ...]
    # REQUIRED for trait reconstruction: episode-local events alone do not
    # establish an ability/item baseline. Retain the complete immutable public
    # chronology, including changes to potential Trace/Transform source mons
    # before a sleep episode began. Never treat a missing local event as proof
    # of the original ability/item.
    public_lines: tuple[str, ...]


def public_sleep_history(lines: tuple[str, ...], *, at_action_boundary: bool) -> PublicSleepLedger:
    """Extract all public sleep episodes from a complete singles transcript.

    ``events`` deliberately preserves unresolved likelihood evidence. In
    particular, ``residual_boundary`` does NOT assert that a newly Yawn-slept
    Pokemon was exposed to Shed Skin earlier in that same residual phase.
    Consumers must not turn this ledger alone into a timer distribution.
    """
    if (type(lines) is not tuple or any(type(line) is not str for line in lines)
            or type(at_action_boundary) is not bool or lines.count("|start") != 1):
        raise ReferenceRefusal("sleep ledger requires a complete public transcript")
    episodes: list[PublicSleepEpisode] = []
    current: dict[str, int] = {}
    active: dict[str, str] = {}
    pending: dict[str, tuple[int, int]] = {}
    yawn: dict[str, str] = {}
    ended_yawn: dict[str, str] = {}

    def add(index: int, event: PublicSleepEvent) -> None:
        episodes[index] = replace(episodes[index], events=episodes[index].events + (event,))

    def settle(key: str, usable: bool) -> None:
        location = pending.pop(key, None)
        if location is None:
            return
        episode, position = location
        events = list(episodes[episode].events)
        events[position] = replace(events[position], sleep_usable=usable)
        episodes[episode] = replace(episodes[episode], events=tuple(events))

    def close(key: str, event: PublicSleepEvent) -> None:
        index = current.pop(key, None)
        if index is not None:
            add(index, event)
            episodes[index] = replace(episodes[index], closed=True)

    for number, line in enumerate(lines):
        fields = line.split("|")
        if len(fields) < 2:
            raise ReferenceRefusal("malformed public sleep transcript")
        command = fields[1]
        if command in {"request", "split"}:
            raise ReferenceRefusal("sleep ledger refuses private or split transport input")
        if ((command == "gen" and fields[2:] != ["3"])
                or (command == "gametype" and fields[2:] != ["singles"])):
            raise ReferenceRefusal("sleep ledger supports only Gen 3 singles")
        # Yawn's public end immediately precedes its attempted status. A
        # failed attempt must not lend its old source to a later Spore.
        if ended_yawn and not (command == "-status" and len(fields) >= 4
                               and fields[3] == "slp" and _key(fields[2]) in ended_yawn):
            ended_yawn.clear()
        if command in {"turn", "upkeep"}:
            for key in tuple(pending):
                settle(key, False)
            if command == "upkeep":
                for key, index in current.items():
                    add(index, PublicSleepEvent("residual_boundary", number,
                                               ("active" if key in active.values() else "bench",)))
            ended_yawn.clear()
        if command in {"switch", "drag", "replace"}:
            if len(fields) < 5:
                raise ReferenceRefusal("sleep ledger received an incomplete switch")
            key = _key(fields[2])
            old = active.get(key[:2])
            if command == "replace":
                # Illusion/identity rewriting is outside Gen 3 randbats; never
                # reinterpret it as a second switch-in/refund.
                raise ReferenceRefusal("sleep ledger refuses replacement identity rewriting")
            if old == key:
                raise ReferenceRefusal("switch-in repeats the currently active identity")
            if old is not None:
                # Roar/Whirlwind may drag an ordinary sleeping actor out before
                # upkeep. Its action has ended, so this is a valid settlement
                # boundary, not an impossible prefix. A Sleep Talk/Snore proof
                # would already have settled the attempt as refundable.
                settle(old, False)
                if old in current:
                    add(current[old], PublicSleepEvent("switch_out", number))
                yawn.pop(old, None)
            active[key[:2]] = key
            if key in current:
                if "slp" not in fields[4].split():
                    raise ReferenceRefusal("switch silently contradicts a sleep episode")
                add(current[key], PublicSleepEvent("reentry", number))
            elif "slp" in fields[4].split():
                raise ReferenceRefusal("sleeping switch lacks its public onset")
        elif command == "-start" and len(fields) >= 4 and fields[3] == "move: Yawn":
            key = _key(fields[2])
            sources = [field[5:] for field in fields[4:] if field.startswith("[of] ")]
            if len(sources) != 1:
                raise ReferenceRefusal("Yawn source identity is not public")
            yawn[key] = _key(sources[0])
        elif command == "-end" and len(fields) >= 4 and fields[3] == "move: Yawn":
            key = _key(fields[2])
            source = yawn.pop(key, None)
            if source is not None:
                ended_yawn[key] = source
        elif command == "-status" and len(fields) >= 4 and fields[3] == "slp":
            key = _key(fields[2])
            if active.get(key[:2]) != key:
                raise ReferenceRefusal("sleep onset is not tied to the public active identity")
            if key in current:
                raise ReferenceRefusal("sleep onset overlaps an existing episode")
            rest = "[from] move: Rest" in fields[4:]
            source = key if rest else ended_yawn.pop(key, None)
            if source is None:
                source = active.get("p2" if key[:2] == "p1" else "p1")
            if source is None:
                raise ReferenceRefusal("sleep onset lacks its public source identity")
            current[key] = len(episodes)
            episodes.append(PublicSleepEpisode(key, source, "rest" if rest else "induced",
                                                number, (PublicSleepEvent("onset", number, tuple(fields[3:])),)))
        elif command == "cant" and len(fields) >= 4:
            key = _key(fields[2])
            if active.get(key[:2]) != key:
                raise ReferenceRefusal("sleep attempt is not tied to the public active identity")
            if fields[3] == "slp":
                if key not in current or key in pending:
                    raise ReferenceRefusal("sleep attempt lacks a unique live episode")
                index = current[key]
                pending[key] = (index, len(episodes[index].events))
                add(index, PublicSleepEvent("attempt", number))
            else:
                settle(key, True)  # lower-priority cant proves sleepUsable
        elif command == "move" and len(fields) >= 4:
            key = _key(fields[2])
            if key in pending:
                if fields[3] not in {"Sleep Talk", "Snore"} or any(
                        field.startswith("[from]") for field in fields[4:]):
                    raise ReferenceRefusal("sleep attempt has no direct sleep-usable move proof")
                settle(key, True)
        elif command == "-activate" and len(fields) >= 4:
            key = _key(fields[2])
            if fields[3] in {"confusion", "move: Attract"}:
                settle(key, True)
            if fields[3] == "move: Heal Bell":
                for sleeper, index in current.items():
                    if sleeper[:2] == key[:2]:
                        add(index, PublicSleepEvent("heal_bell", number, tuple(fields[2:])))
            elif key in current:
                add(current[key], PublicSleepEvent("activation", number, tuple(fields[3:])))
        elif command == "-cureteam" and len(fields) >= 3:
            side = _key(fields[2])[:2]
            for key in tuple(current):
                if key[:2] == side:
                    close(key, PublicSleepEvent("team_cure", number, tuple(fields[3:])))
        elif ((command == "-curestatus" and len(fields) >= 4 and fields[3] == "slp")
              or (command == "faint" and len(fields) >= 3)):
            key = _key(fields[2])
            close(key, PublicSleepEvent("faint" if command == "faint" else "cure",
                                       number, tuple(fields[3:])))
            yawn.pop(key, None)
        elif command in {"-ability", "-endability", "-transform", "-item", "-enditem"} and len(fields) >= 3:
            key = _key(fields[2])
            if key in current:
                add(current[key], PublicSleepEvent(command[1:], number, tuple(fields[3:])))
        elif command == "-status" and len(fields) >= 4 and _key(fields[2]) in current:
            raise ReferenceRefusal("new status silently contradicts an open sleep episode")
    if pending and not at_action_boundary:
        raise ReferenceRefusal("sleep attempt remains unsettled in a truncated prefix")
    for key in tuple(pending):
        settle(key, False)
    return PublicSleepLedger(tuple(episodes), lines)
