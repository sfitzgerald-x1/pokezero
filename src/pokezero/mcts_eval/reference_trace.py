"""Bounded public Trace restoration, not a new team proposal or likelihood.

The full trait qualifier proves the acquisition donor's ability at that line.
This second scan binds that proof to the holder's current entry and restricts
its ENTIRE history to stateless copies. Unsupported histories refuse the
decision; they never reject a proposed team and draw a replacement.
"""
from __future__ import annotations

import re

from .paper_reference import ReferenceRefusal
from .reference_sleep_history import _key


STATELESS_TRACE = frozenset({"waterabsorb", "voltabsorb", "levitate", "pressure"})


def trace_restoration_assignments(ledger, opening_traits, compatibility):
    """Return the COMPLETE original-Trace roster after a compatible qualifier.

    Active holders without an explicit supported acquisition still refuse,
    including legitimate no-copy entries; absence is not a certificate.
    Switch-out clears a copy; donor switching/fainting does not. Never-seen bench members
    have their proposed original ability. All admitted holders have never had
    Truant (or another stateful copy), so their persistent truantTurn is false.
    """
    def refuse():
        raise ReferenceRefusal("joint Trace materialization needs resolved stateless public history")

    if compatibility.status != "compatible" or ledger.public_lines.count("|start") != 1:
        refuse()
    original = {key: re.sub(r"[^a-z0-9]", "", trait.ability.lower())
                for key, trait in opening_traits.items()}
    holders = {key for key, ability in original.items() if ability == "trace"}
    current = {key: "trace" for key in holders}
    entered = {key: None for key in holders}
    acquired = {key: None for key in holders}
    active = {}
    fainted = set()
    for number, raw in enumerate(ledger.public_lines):
        fields = raw.split("|")
        command = fields[1]
        if command in {"switch", "drag"}:
            key = _key(fields[2])
            old = active.get(key[:2])
            if old in holders:
                if acquired[old] is None:
                    refuse()
                current[old], entered[old], acquired[old] = "trace", None, None
            active[key[:2]] = key
            if key in holders:
                current[key], entered[key], acquired[key] = "trace", number, None
        elif command == "faint":
            key = _key(fields[2])
            if key in holders and acquired[key] is None:
                refuse()
            fainted.add(key)
        elif command == "-ability":
            key = _key(fields[2])
            if key not in holders:
                continue
            new = re.sub(r"[^a-z0-9]", "", fields[3].lower())
            if "[from] ability: Trace" in fields[4:]:
                donors = [_key(tag[5:]) for tag in fields[4:] if tag.startswith("[of] ")]
                if (new not in STATELESS_TRACE or active.get(key[:2]) != key
                        or key in fainted or entered[key] is None or current[key] != "trace"
                        or len(donors) != 1 or donors[0] in fainted
                        or active.get(donors[0][:2]) != donors[0]):
                    refuse()
                # Old ability, active donor, and donor ability at acquisition
                # were checked by qualify_sleep_traits, not today's donor.
                current[key], acquired[key] = new, number
            elif any(tag.startswith("[from]") for tag in fields[4:]) or new != current[key]:
                refuse()
        elif command in {"-transform", "-endability"}:
            if _key(fields[2]) in holders:
                refuse()
        elif command == "-activate" and len(fields) >= 4 and fields[3] == "Skill Swap":
            participants = [_key(fields[2])] + [_key(tag[5:]) for tag in fields[4:]
                                               if tag.startswith("[of] ")]
            if holders.intersection(participants):
                refuse()
    qualified = dict(compatibility.current_traits)
    rows = []
    for key in sorted(holders):
        live = active.get(key[:2]) == key and key not in fainted
        if qualified[key].ability != current[key]:
            refuse()
        if (live and entered[key] is None) or (entered[key] is not None and acquired[key] is None):
            refuse()
        side, species = key.split(":", 1)
        rows.append(dict(side=side, species=species, active=live,
                         ability=current[key] if live else "trace"))
    represented = {key for key in holders if (active.get(key[:2]) != key or key in fainted
                    or (acquired[key] is not None and acquired[key] > entered[key]))}
    for key, value in compatibility.current_traits:
        if value.ability != original[key] and key not in represented:
            raise ReferenceRefusal("joint sleep current ability is not represented by the materializer")
    return rows
