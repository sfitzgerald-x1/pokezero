"""Reader for the fallback address store recorded by ``engine_search`` (#1063).

``EngineMctsStats.fallback_samples`` files a ``{battle_id, round, seat, reason}``
tuple for every fallback decision, keyed by class and by reason. The battle id
carries the seed, so -- as ``engine_search.py:663`` puts it -- *any entry here
replays as a single turn*. This module is the reader half of the reader/replay-driver
pair that the burndown goal records as not existing.

It exists because the campaign has repeatedly theorised about causes it could have
read. Four eras of addresses accumulated on disk while a burndown report spent three
successive corrections reasoning about a Substitute legality divergence for which
four exact addresses were already recorded.

Design notes, each paid for by a prior defect:

* **Counts here are not frequencies.** The producer caps addresses per class at
  ``_FALLBACK_SAMPLES_PER_CLASS`` (``engine_search.py:457``), so counting addresses
  ranks a class by how many raw variants it shattered into, not by how often it
  fired -- an inversion of 18x is reachable. :class:`CorpusScan` therefore also
  reads the uncapped ``world_failure_reasons`` / ``fallback_reasons`` totals, and
  the CLI orders by those.
* **Completeness is reported, never assumed.** ``fallback_sample_addresses_dropped``
  is surfaced; non-zero means occurrences exist with no replayable address.
* **Canonicalisation is allowlist-based** -- see the block above
  :func:`canonical_key`. Blanket literal-stripping merges distinct classes.
* **The locator includes the shard.** ``battle_id`` carries only the seed, and a
  depth/arm grid reuses one ``seed_start`` across shards.
* Addresses are de-duplicated per document (``fallback_samples`` is reachable by
  more than one path in shards that mirror stats under ``per_seat[X]``) and shard
  paths are de-duplicated across arguments.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

__all__ = [
    "CorpusScan",
    "FallbackAddress",
    "canonical_key",
    "iter_shard_addresses",
    "load_addresses",
    "scan_corpus",
    "group_by_canonical_key",
]

# Canonicalisation is ALLOWLIST-BASED, and that choice is the whole design.
#
# The tempting rule -- "a quoted literal is payload, strip them all" -- is wrong, and
# `_blocker_bucket` (`engine_search.py:410-425`) already records why: for
# `baton-pass` the quoted operand is the VOLATILE and is "the entire actionable
# content", so a bare `materialization_blocker: baton-pass` gave "no way to tell
# whether it was a Substitute worth supporting or a Bide worth refusing". The same
# holds for `volatile_unsupported` (['perish0'] vs ['flashfire']),
# `side_condition_unsupported` ('spikes' vs 'lightscreen'), `boost_unsupported`
# ('evasion' vs 'accuracy'), `weather_unsupported`, and `self_moveset_mismatch`
# (where commit 29ca5697 attributed the root cause ON the quoted move: "'shadowball'
# is decisive"). Blanket stripping merges fixed classes with unfixed ones.
#
# So: literals are KEPT unless their position is registered below as carrying a
# bystander. Adding a position is a deliberate act with a stated reason. An
# unrecognised key is returned unchanged -- over-splitting is recoverable by reading
# the raw keys, over-merging silently destroys the distinction that names the fix.
#
# The `,` (independent slugs) and `+` (predicates within one slug, only the first
# carrying the family prefix) separators are never rewritten: doing so strips
# prefixes and invents phantom bare rows, an error this campaign has shipped twice.

# A quoted run that is NOT glued to surrounding word characters. Guards against
# apostrophes inside interpolated native error strings ("can't materialize
# 'Zapdos'"), where a naive `'[^']*'` matches "'t materialize '" -- destroying the
# predicate and keeping the payload, both hazards at once.
_LITERAL = r"(?<![A-Za-z])'[^']*'(?![A-Za-z])"

# (compiled pattern, replacement) for positions whose operand is a BYSTANDER: it
# names who was present, not what failed.
_BYSTANDER_POSITIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    # The foe's ability in a `self_request_state_unsupported` trapped refusal. The
    # failure is "request says trapped, sampled world does not trap"; which ability
    # happened to be on the field is incidental. This shattered one class of 624
    # across six rows in era 64 and mis-ranked it in every prior era.
    (re.compile(rf"(foe ability ){_LITERAL}"), r"\1'?'"),
)

# Seat/slot appears in two spellings across producers -- `{slot}:` unquoted
# (`engine_world.py:2200`) and `side {slot!r}` quoted (`encore_move_unknown`). Left
# alone, one family splits by seat and the other does not, so the canonical key
# space is partitioned by an f-string accident and cannot be compared across
# families. Seat is already a first-class field on FallbackAddress, so it is
# normalised out of the key in BOTH spellings.
_SEAT_POSITIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(side )'p[12]'"), r"\1'?'"),
    (re.compile(r"(^|[:,] )p[12]:"), r"\1p?:"),
)


def canonical_key(key: str) -> str:
    """Collapse *bystander* payload so one class reads as one key.

    Only registered positions are rewritten; every other quoted literal is
    preserved because in this key space the operand is frequently the actionable
    content that names the fix.

    >>> canonical_key("x: flags ['trapped'] (foe ability 'swarm')")
    "x: flags ['trapped'] (foe ability '?')"
    >>> canonical_key("volatile_unsupported: side 'p1': ['perish0']")
    "volatile_unsupported: side '?': ['perish0']"
    """
    for pattern, replacement in _BYSTANDER_POSITIONS + _SEAT_POSITIONS:
        key = pattern.sub(replacement, key)
    return key


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
    def locator(self) -> tuple[str, str, int, str]:
        """The replay coordinates, independent of which class recorded them.

        ``source`` is part of the locator and must not be dropped. ``battle_id`` is
        ``f"{prefix}-{seed}"`` (`foulplay_bridge.py:2699`) and carries the seed and
        nothing else -- no arm, depth, sims or checkpoint. A depth/arm grid reuses
        one ``seed_start`` across shards, so ``(battle_id, round, seat)`` alone
        collides across genuinely different search configurations and undercounts
        distinct decisions. The shard path is the only carrier of that identity.
        """
        return (self.source, self.battle_id, self.round, self.seat)


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


def _iter_shard_paths(paths: Iterable[Path]) -> Iterator[tuple[Path, str]]:
    """Yield ``(path, label)`` pairs, each real file exactly once.

    De-duplicated by resolved path: overlapping arguments (``runs/ runs/a.json``)
    would otherwise read the same shard twice, and the per-document dedup inside
    :func:`iter_shard_addresses` cannot see across calls -- the naive walk one level
    up from where the guard sits.

    The label is the path relative to the argument root, not the basename: era
    directories contain duplicate basenames that differ only by parent directory.
    """
    seen: set[Path] = set()
    for path in paths:
        if path.is_dir():
            for shard in sorted(path.rglob("*.json")):
                resolved = shard.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield shard, str(shard.relative_to(path))
        else:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path, path.name


@dataclass
class CorpusScan:
    """Everything a corpus consumer needs, including what it does NOT have."""

    addresses: list[FallbackAddress] = field(default_factory=list)
    #: Sum of ``fallback_sample_addresses_dropped``. Non-zero means the address
    #: corpus is INCOMPLETE -- occurrences existed that no address was kept for.
    addresses_dropped: int = 0
    #: Uncapped occurrence counts from ``world_failure_reasons`` /
    #: ``fallback_reasons``, canonicalised. THIS is class frequency; address counts
    #: are not, because the producer caps samples per class.
    true_counts: Counter = field(default_factory=Counter)
    shards_read: int = 0
    shards_unreadable: int = 0

    @property
    def complete(self) -> bool:
        return self.addresses_dropped == 0


def _scan_document(document: Any, scan: CorpusScan, *, source: str) -> None:
    scan.addresses.extend(iter_shard_addresses(document, source=source))
    stats = (document.get("engine_mcts") or {}).get("policy_stats") or {}
    if not isinstance(stats, Mapping):
        return
    dropped = stats.get("fallback_sample_addresses_dropped")
    if isinstance(dropped, int) and not isinstance(dropped, bool):
        scan.addresses_dropped += dropped
    for field_name in ("world_failure_reasons", "fallback_reasons"):
        counts = stats.get(field_name)
        if not isinstance(counts, Mapping):
            continue
        prefix = "" if field_name == "world_failure_reasons" else "fallback:"
        for key, value in counts.items():
            if isinstance(value, int) and not isinstance(value, bool):
                scan.true_counts[canonical_key(f"{prefix}{key}")] += value


def scan_corpus(paths: Sequence[Path]) -> CorpusScan:
    """Read addresses AND the completeness/frequency context they need."""
    scan = CorpusScan()
    for shard, label in _iter_shard_paths(paths):
        try:
            document = json.loads(shard.read_text())
        except (OSError, ValueError):
            scan.shards_unreadable += 1
            continue
        if not isinstance(document, Mapping):
            scan.shards_unreadable += 1
            continue
        scan.shards_read += 1
        _scan_document(document, scan, source=label)
    return scan


def load_addresses(paths: Sequence[Path]) -> list[FallbackAddress]:
    """Load every address from the given shard files and/or directories.

    Unreadable or non-shard JSON is skipped rather than fatal: an era directory
    routinely holds summary files alongside seat shards. Use :func:`scan_corpus`
    when completeness matters -- this helper cannot report what it dropped.
    """
    return scan_corpus(paths).addresses


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
    missing = [str(path) for path in args.paths if not path.exists()]
    if missing:
        # A typo'd path must not read as an empty corpus.
        print(f"path does not exist: {', '.join(missing)}")
        return 2

    scan = scan_corpus(args.paths)
    addresses = scan.addresses
    if not addresses:
        print(f"no fallback addresses found ({scan.shards_read} shards read)")
        return 1

    distinct_locators = {address.locator for address in addresses}
    print(f"shards read: {scan.shards_read} ({scan.shards_unreadable} skipped)")
    print(f"addresses: {len(addresses)}")
    print(f"distinct replay locators: {len(distinct_locators)}")
    if not scan.complete:
        # Silent truncation reads as "covered everything". Say it loudly.
        print(
            f"INCOMPLETE CORPUS: {scan.addresses_dropped} addresses were dropped by the "
            f"per-class sample ceiling; some occurrences have no replayable address."
        )

    counts: Counter[str] = Counter(
        address.key if args.raw_keys else address.canonical for address in addresses
    )
    label = "raw" if args.raw_keys else "canonical"
    print(f"distinct {label} keys: {len(counts)}")

    # ORDER BY TRUE FREQUENCY, never by address count. The producer caps each raw key
    # at _FALLBACK_SAMPLES_PER_CLASS, so address counts rank a class by how many raw
    # variants it shattered into -- measured 18x inversions. Address count is reported
    # only as replay COVERAGE of the class.
    print(f"{'occurrences':>12}  {'addrs':>5}  class (ordered by true occurrences)")
    ordered = sorted(
        counts,
        key=lambda key: (-scan.true_counts.get(key, 0), -counts[key], key),
    )
    for key in ordered:
        true = scan.true_counts.get(key)
        shown = f"{true:12d}" if true is not None else f"{'unknown':>12}"
        print(f"{shown}  {counts[key]:5d}  {key}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
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
