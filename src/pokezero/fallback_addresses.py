"""Reader for the fallback address store recorded by ``engine_search`` (#1063).

``EngineMctsStats.fallback_samples`` files a ``{battle_id, round, seat, reason}``
tuple for every fallback decision, keyed by class and by reason. The battle id
carries the seed, so -- as ``engine_search.py:663`` puts it -- *any entry here
replays as a single turn*. GOAL.md calls these addresses "the crown" and records
that "no reader or replay driver exists". This module is the reader half.

It exists because the campaign has repeatedly theorised about causes it could have
read. Four eras of addresses accumulated on disk while burndown report 3 spent
three successive corrections reasoning about a Substitute legality divergence for
which four exact addresses were already recorded.

Two hazards this module is built around, both previously paid for:

* ``fallback_samples`` is duplicated byte-identically under ``per_seat[X]`` and
  ``per_seat[X].policy_stats`` in some shard layouts. A naive walk double-counts
  every address. Addresses are therefore de-duplicated per shard.
* Reason keys embed variable payload -- notably the *foe's ability*, a bystander to
  the failure -- which shatters one class across many rows. In era 64 a single
  ``trapped`` class of 624 decisions was split across six ability-named keys and
  consequently mis-ranked in every prior era. :func:`canonical_key` collapses that
  payload; the raw key is always retained alongside it.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

__all__ = [
    "FallbackAddress",
    "canonical_key",
    "iter_shard_addresses",
    "load_addresses",
    "group_by_canonical_key",
]

# Quoted literals in a reason key are PAYLOAD -- the foe ability, a species, a move
# name. The failure is the same failure whichever value appears, so the canonical
# key holds the position and drops the value.
#
# Deliberately narrow: only quoted runs are rewritten. The reason-key grammar uses
# `,` to join independent slugs and `+` to join predicates within one slug (where
# only the FIRST predicate carries the family prefix). Splitting or rewriting on
# those separators strips prefixes and invents phantom bare rows -- an error this
# campaign has shipped twice. This substitution touches neither separator.
_QUOTED_LITERAL = re.compile(r"'[^']*'")

_PLACEHOLDER = "'?'"


def canonical_key(key: str) -> str:
    """Collapse payload-bearing reason keys to their ``(site, predicate)`` identity.

    >>> canonical_key("x: flags ['trapped'] (foe ability 'swarm')")
    "x: flags ['?'] (foe ability '?')"

    Keys with no quoted payload are returned unchanged.
    """
    return _QUOTED_LITERAL.sub(_PLACEHOLDER, key)


@dataclass(frozen=True, order=True)
class FallbackAddress:
    """One replayable fallback decision.

    ``battle_id`` carries the seed and ``round``/``seat`` locate the decision
    within it, so the triple is sufficient to reconstruct the refusal.
    """

    battle_id: str
    round: int
    seat: str
    reason: str
    key: str
    source: str = ""

    @property
    def canonical(self) -> str:
        return canonical_key(self.key)

    @property
    def locator(self) -> tuple[str, int, str]:
        """The replay coordinates, independent of which class recorded them."""
        return (self.battle_id, self.round, self.seat)


def _walk_sample_blocks(node: Any) -> Iterator[Mapping[str, Any]]:
    """Yield every ``fallback_samples`` mapping anywhere in a shard document."""
    if isinstance(node, Mapping):
        for name, value in node.items():
            if name == "fallback_samples" and isinstance(value, Mapping):
                yield value
            else:
                yield from _walk_sample_blocks(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_sample_blocks(item)


def iter_shard_addresses(document: Any, *, source: str = "") -> Iterator[FallbackAddress]:
    """Yield de-duplicated addresses from one parsed shard document.

    De-duplication is per shard and covers the ``per_seat`` mirroring described in
    the module docstring: the same (key, battle, round, seat) recorded twice by two
    views of the same statistics is one address, not two.
    """
    seen: set[tuple[str, str, int, str]] = set()
    for block in _walk_sample_blocks(document):
        for key, entries in block.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                battle_id = entry.get("battle_id")
                round_index = entry.get("round")
                seat = entry.get("seat")
                if not isinstance(battle_id, str) or not isinstance(seat, str):
                    continue
                if not isinstance(round_index, int) or isinstance(round_index, bool):
                    continue
                dedup = (key, battle_id, round_index, seat)
                if dedup in seen:
                    continue
                seen.add(dedup)
                reason = entry.get("reason")
                yield FallbackAddress(
                    battle_id=battle_id,
                    round=round_index,
                    seat=seat,
                    reason=reason if isinstance(reason, str) else "",
                    key=key,
                    source=source,
                )


def _iter_shard_paths(paths: Iterable[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(path.rglob("*.json"))
        else:
            yield path


def load_addresses(paths: Sequence[Path]) -> list[FallbackAddress]:
    """Load every address from the given shard files and/or directories.

    Unreadable or non-shard JSON is skipped rather than fatal: an era directory
    routinely holds summary files alongside seat shards.
    """
    out: list[FallbackAddress] = []
    for shard in _iter_shard_paths(paths):
        try:
            document = json.loads(shard.read_text())
        except (OSError, ValueError):
            continue
        out.extend(iter_shard_addresses(document, source=shard.name))
    return out


def group_by_canonical_key(
    addresses: Iterable[FallbackAddress],
) -> dict[str, list[FallbackAddress]]:
    """Group addresses by canonical key, preserving input order within a group."""
    grouped: dict[str, list[FallbackAddress]] = defaultdict(list)
    for address in addresses:
        grouped[address.canonical].append(address)
    return dict(grouped)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pokezero.fallback_addresses",
        description="Read the fallback address store out of probe shards.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Shard JSON files, or directories searched recursively for *.json.",
    )
    parser.add_argument(
        "--raw-keys",
        action="store_true",
        help="Group by the raw reason key instead of the canonical (site, predicate) key.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Write the full address corpus to this path as JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    addresses = load_addresses(args.paths)
    if not addresses:
        print("no fallback addresses found")
        return 1

    distinct_locators = {address.locator for address in addresses}
    print(f"addresses: {len(addresses)}")
    print(f"distinct replay locators: {len(distinct_locators)}")

    counts: Counter[str] = Counter(
        address.key if args.raw_keys else address.canonical for address in addresses
    )
    label = "raw" if args.raw_keys else "canonical"
    print(f"distinct {label} keys: {len(counts)}")
    for key, count in counts.most_common():
        print(f"{count:6d}  {key}")

    if args.json_out:
        payload = [
            {
                "battle_id": address.battle_id,
                "round": address.round,
                "seat": address.seat,
                "reason": address.reason,
                "key": address.key,
                "canonical_key": address.canonical,
                "source": address.source,
            }
            for address in addresses
        ]
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"wrote {len(payload)} addresses to {args.json_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
