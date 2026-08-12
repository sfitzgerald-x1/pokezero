#!/usr/bin/env python3
"""C157 reach census: which gen3 randbat moves can be rendered as the MINORITY outcome
when the engine merges a successful-but-no-op hit with a miss.

WHY THIS SCRIPT EXISTS RATHER THAN A TABLE IN A REPORT. A pool figure published from a
hand-copied accuracy table was wrong and had to be retracted; the replacement rule is that
accuracies come out of the ENGINE'S OWN move data with a loud failure mode. So every number
here is derived from `poke_engine::choices::MOVES` via
`cargo run -p pokezero-search --example dump_move_table`, crossed with the same 1682-variant
gen3 randbat universe the encoder is handed (`pokezero.randbat.load_gen3_randbat_source_cached`).
A move name in the pool that does not resolve in the engine table is a hard error, never a
silent skip — the one exception is the `hiddenpower<type>` family, which the engine models as
a single `HIDDENPOWER` entry and which is mapped explicitly, by name, so the exception is
visible rather than swallowed by a fallback.

THE MECHANISM. `combine_duplicate_instructions` merges branches with identical deltas. When a
move's whole effect cannot apply to the current defender, its successful no-op hit and its
miss are both empty and become ONE branch. Conditioned on the move having been attempted:

    P(hit, no effect) = accuracy
    P(miss)           = 1 - accuracy

crossing at 50%. Any immobilizer factor (paralysis 0.25, Attract 1/2) multiplies both and
cancels — and both immobilizers now carry `Instruction::MoveImmobilized`, so they are
separate marked branches anyway.

THREE FAMILIES ARE REPORTED, because they have three different dispositions in
`rust/pokezero-search/src/events.rs`:

  A. `status_fail` — already-statused defender. HANDLED before this change and after it.
  B. `volatile_fail` — defender already carries the move's volatile, accuracy < 100.
     THE DEFECT THIS CHANGE FIXES: the miss inference labelled the whole merged branch
     `|[miss]|`, which is the minority render above 50% accuracy.
  C. blocked OPPONENT-side boost, accuracy < 100. STILL LABELLED `|[miss]|`, disclosed with
     the reach below rather than left unstated. `events.rs` computes `boost_has_no_effect`
     for it and consumes it only for the self-target case.
  D. the accuracy-100 siblings of B. Deterministic, so not a mislabel — a MISSING
     `|-fail|` line and blank target. Bounded here, left to its own change.

Usage:
  scripts/c157_no_effect_hit_reach.py --showdown-root <built PS checkout> [--move-table x.json]

`--showdown-root` must be a BUILT Pokemon Showdown checkout (`node build`); the variant
universe is generated from it, which is what makes the denominator 1682 rather than a
sets.json row count.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Explicit, named exception to the loud-failure rule. Showdown carries one entry per Hidden
#: Power type; the engine carries a single `HIDDENPOWER`. A generic "unknown move -> skip"
#: fallback would also swallow a genuinely renamed move, which is the failure this script is
#: written against, so the mapping is by name and nothing else is forgiven.
_HIDDEN_POWER_PREFIX = "hiddenpower"


def _normalize(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def load_move_table(path: Path | None) -> dict[str, dict]:
    """The engine's own move table, keyed by normalized name.

    Runs the crate example when no cached JSON is given, so the default path cannot be a
    stale file. A non-zero exit is raised, not warned about.
    """

    if path is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        completed = subprocess.run(
            ["cargo", "run", "-q", "-p", "pokezero-search", "--example", "dump_move_table"],
            cwd=REPO_ROOT / "rust" / "pokezero-search",
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(completed.stdout)
    if not payload:
        raise SystemExit("the engine move table came back EMPTY; refusing to report zeros")
    return {_normalize(key): value for key, value in payload.items()}


def resolve(table: dict[str, dict], move: str) -> dict:
    key = _normalize(move)
    if key in table:
        return table[key]
    if key.startswith(_HIDDEN_POWER_PREFIX) and _HIDDEN_POWER_PREFIX in table:
        return table[_HIDDEN_POWER_PREFIX]
    raise SystemExit(
        f"move {move!r} is in the gen3 randbat pool and NOT in the engine move table. "
        "That is a real divergence between the pool and the engine the renderer runs on, "
        "not a reporting inconvenience -- fix it rather than skipping the move."
    )


def family_a(move: dict) -> bool:
    """Already-statused defender. Handled by `status_fail`."""
    return (
        move["target"] == "Opponent"
        and move["category"] == "Status"
        and move["accuracy"] < 100
        and move["has_status"]
    )


def family_b(move: dict) -> bool:
    """Already-carried volatile, sub-100% accuracy. The defect. Fixed by `volatile_fail`."""
    return (
        move["target"] == "Opponent"
        and move["category"] == "Status"
        and move["accuracy"] < 100
        and not move["has_status"]
        and move["volatile"].startswith("Opponent:")
    )


def family_c(move: dict) -> bool:
    """Blocked opponent boost, sub-100% accuracy, no volatile. Disclosed, NOT fixed."""
    return (
        move["target"] == "Opponent"
        and move["category"] == "Status"
        and move["accuracy"] < 100
        and not move["has_status"]
        and not move["has_volatile"]
        and move["has_boost"]
    )


def family_d(move: dict) -> bool:
    """The accuracy-100 volatile siblings: a missing `|-fail|`, not a mislabel."""
    return (
        move["target"] == "Opponent"
        and move["category"] == "Status"
        and move["accuracy"] >= 100
        and not move["has_status"]
        and move["volatile"].startswith("Opponent:")
    )


def family_e_grass_immune_reach(table, variants):
    """The Leech Seed / Grass-receiver pairing, both halves of the `|-immune|` case.

    Reported as a PAIRING rather than a move family: the render is wrong whenever a Leech Seed
    user faces a Grass-typed target, and it does not matter whether the target already carries
    the seed (that only decides WHICH wrong line was emitted before the fix). So the reach is
    the product of two independent pool populations, and both are counted here.
    """
    seeders = sum(1 for _s, _v, moves in variants if any(m["move_id"] == "LEECHSEED" for m in moves))
    return seeders


FAMILIES = (
    ("A  status_fail (already handled)", family_a),
    ("B  volatile_fail (THE DEFECT, fixed)", family_b),
    ("C  blocked opponent boost (disclosed, not fixed)", family_c),
    ("D  accuracy-100 volatile sibling (missing |-fail|)", family_d),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--showdown-root", required=True, type=Path)
    parser.add_argument("--move-table", type=Path, default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from pokezero.randbat import load_gen3_randbat_source_cached

    table = load_move_table(args.move_table)
    source = load_gen3_randbat_source_cached(args.showdown_root)
    variants = [
        (universe.species, entry.variant_id, [resolve(table, move) for move in entry.moves])
        for universe in source.universes.values()
        for entry in universe.variants
    ]
    total = len(variants)
    if total == 0:
        raise SystemExit("the variant universe is EMPTY; refusing to report zeros")
    print(f"gen3 randbat variant universe: {total} variants, {len(source.universes)} species")
    print(f"engine move table: {len(table)} moves (source: poke_engine::choices::MOVES)")

    for label, predicate in FAMILIES:
        members = sorted(
            {
                (name, move["accuracy"])
                for name, move in table.items()
                if predicate(move)
            },
            key=lambda row: row[1],
        )
        counts: Counter[str] = Counter()
        carriers = set()
        for _species, variant_id, moves in variants:
            hits: set[str] = set()
            for move in moves:
                if predicate(move):
                    hits.add(_normalize(move["move_id"]))
            counts.update(hits)
            if hits:
                carriers.add(variant_id)
        print(f"\n=== FAMILY {label} ===")
        print(f"  engine-table members: {len(members)}")
        print(f"  variants carrying >=1: {len(carriers)}/{total} ({len(carriers) / total:.2%})")
        for name, accuracy in members:
            hit_mass = accuracy / 100.0
            miss_mass = 1.0 - hit_mass
            verdict = "hit-no-op dominates" if hit_mass > miss_mass else "MISS dominates"
            print(
                f"    {name:14s} acc={accuracy:5.1f}  P(hit,no-op)={hit_mass:.3f} "
                f"P(miss)={miss_mass:.3f}  {verdict:20s} carriers={counts.get(name, 0):5d}"
            )

    # E: the Leech-Seed-into-Grass immunity, which is a PAIRING and not a move family.
    # ⚠ THE FIRST VERSION OF THIS BLOCK PRINTED A SILENT ZERO. It read `universe.types`,
    # which does not exist on that object, so `getattr(..., None)` returned None for every
    # species and the Grass count came out 0/1682 -- against a pool that obviously contains
    # Cacturne and Venusaur. An absent measurement rendering as a clean zero is the exact trap
    # this campaign keeps paying for, so the type lookup now goes through the dex and RAISES
    # when a species does not resolve.
    from pokezero.dex import load_showdown_dex_cached

    dex = load_showdown_dex_cached(args.showdown_root)
    seeders = family_e_grass_immune_reach(table, variants)
    grass = 0
    grass_species = set()
    for species, _variant_id, _moves in variants:
        info = dex.species_info(species)
        if info is None or not info.types:
            raise SystemExit(
                f"species {species!r} does not resolve to types in the dex; refusing to "
                "report a Grass count that silently treats it as non-Grass"
            )
        if any(str(t).lower() == "grass" for t in info.types):
            grass += 1
            grass_species.add(species)
    if grass == 0:
        raise SystemExit(
            "the Grass count came out ZERO over a pool that contains Grass species; that is "
            "an instrument failure, not a measurement"
        )
    print("\n=== FAMILY E  Leech Seed into a GRASS target -> |-immune| (fixed) ===")
    print(f"  Leech Seed carriers:  {seeders}/{total} ({seeders / total:.2%})")
    print(f"  Grass-typed variants: {grass}/{total} ({grass / total:.2%}) across "
          f"{len(grass_species)} species")
    print("  Both halves of the state are covered: already-seeded (was `|-fail|`) and")
    print("  not-seeded (was `|[miss]|`). Cross-side pairing, so the two populations are")
    print("  independent and either can be the receiver of a Baton-Passed seed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
