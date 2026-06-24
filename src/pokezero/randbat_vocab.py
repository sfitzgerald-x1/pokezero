"""Build the neural category embedding vocabulary from the closed Gen 3 random battle universe.

The observation encoder (``showdown.py``) hashes categorical strings into ids via
``stable_category_id``. Gen 3 random battles are a *closed* format: the set of possible
game-entity strings (species, moves, abilities, items, statuses) is finite and enumerable
from the Pokemon Showdown random-battle data (``data/random-battles/gen3/sets.json``) and
the generator's item logic (``teams.ts``). Enumerating them up front gives every legal
entity a dedicated, trainable embedding row instead of relying on whatever a sample of
games happened to contain, and drives play-time out-of-vocabulary down to only the
inherently-dynamic fields (HP ``condition:`` text, opponent usernames) that belong in the
reserved OOV block rather than in the vocabulary.

This module enumerates the *categorical strings* the encoder emits for those entities,
plus the bounded structural schema tokens, and maps them through the same
``stable_category_id`` used at encode time. Inherently dynamic fields are intentionally
excluded (left to the OOV block).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .dex import load_showdown_dex_cached
from .showdown import _normalize_identifier, stable_category_id

# Items the Gen 3 random-battle generator can assign, from
# data/random-battles/gen3/teams.ts getItem().
GEN3_RANDBAT_ITEMS = (
    "Stick",
    "Soul Dew",
    "Silk Scarf",
    "Thick Club",
    "Light Ball",
    "Lum Berry",
    "White Herb",
    "Choice Band",
    "Petaya Berry",
    "Salac Berry",
    "Liechi Berry",
    "Leftovers",
    "Twisted Spoon",
)

# Gen 3 volatile/major statuses plus the "no status" sentinel.
GEN3_STATUSES = ("brn", "par", "slp", "frz", "psn", "tox", "fnt", "none")

# Moves that can be selected/forced regardless of a Pokemon's movepool: Struggle (no PP),
# Recharge (after Hyper Beam etc.), and the bare in-battle "Hidden Power" label.
UNIVERSAL_MOVES = ("struggle", "recharge", "hiddenpower")

# Unown's cosmetic formes share one randbat set but appear with forme-specific display
# names in battle (e.g. "Unown-L"). The A forme is the base species "Unown" (no suffix),
# so cosmetic formes start at B. Enumerate them so each is collision-free in vocab.
UNOWN_FORMES = tuple(f"Unown-{letter}" for letter in "BCDEFGHIJKLMNOPQRSTUVWXYZ") + ("Unown-Exclamation", "Unown-Question")

# Bounded structural request kinds the env can surface (see showdown.py _request_kind).
REQUEST_KINDS = ("move", "force_switch", "team_preview", "wait", "none", "unknown")

# Token actor/target roles used by the encoder.
EVENT_ROLES = ("self", "opponent", "none")

# Showdown protocol message types that can appear as recent-event tokens. This is a
# superset for coverage; unobserved types are harmless extra rows.
EVENT_TYPES = (
    "move", "switch", "drag", "replace", "faint", "cant", "turn", "upkeep",
    "-damage", "-heal", "-sethp", "-status", "-curestatus", "-cureteam",
    "-boost", "-unboost", "-setboost", "-swapboost", "-clearboost", "-clearallboost",
    "-weather", "-fieldstart", "-fieldend", "-sidestart", "-sideend",
    "-ability", "-endability", "-item", "-enditem", "-start", "-end", "-activate",
    "-immune", "-miss", "-fail", "-crit", "-supereffective", "-resisted",
    "-transform", "-mega", "-formechange", "-prepare", "-mustrecharge",
    "-singleturn", "-singlemove", "-hint", "-message", "-block",
    "player", "win", "tie", "unknown",
    # Battle-log preamble / framing events.
    "t:", "gametype", "gen", "tier", "rule", "teamsize", "clearpoke", "poke",
    "teampreview", "start", "inactive", "inactiveoff", "raw", "j", "l", "c:",
)


def _gen3_sets_path(showdown_root: str | Path) -> Path:
    return Path(showdown_root) / "data" / "random-battles" / "gen3" / "sets.json"


def gen3_randbat_entities(showdown_root: str | Path) -> dict[str, tuple[str, ...]]:
    """Return the closed universe of Gen 3 randbat game entities.

    Keys: ``species``, ``moves``, ``abilities``, ``items``, ``statuses``. Species and move
    values are Showdown ids; abilities/items are display names (normalized when emitted).
    """
    data = json.loads(_gen3_sets_path(showdown_root).read_text(encoding="utf-8"))
    species: set[str] = set()
    moves: set[str] = set()
    abilities: set[str] = set()
    for species_id, info in data.items():
        species.add(str(species_id))
        for entry in info.get("sets", []):
            for move_id in entry.get("movepool", []):
                moves.add(str(move_id))
            for ability in entry.get("abilities", []):
                abilities.add(str(ability))
    return {
        "species": tuple(sorted(species)),
        "moves": tuple(sorted(moves)),
        "abilities": tuple(sorted(abilities)),
        "items": GEN3_RANDBAT_ITEMS,
        "statuses": GEN3_STATUSES,
    }


def gen3_randbat_category_strings(showdown_root: str | Path) -> dict[str, list[str]]:
    """Enumerate the categorical strings the encoder emits for the closed Gen 3 universe.

    Grouped by source so the vocabulary breakdown is auditable. Mirrors the templates in
    ``showdown.py``: ``species:<id>``, ``move:<id>``, ``belief:possible_move:<id>``,
    ``belief:possible_ability:<id>``, ``belief:possible_item:<id>``, ``status:<id>``, plus
    the bounded structural tokens. Inherently-dynamic strings (``condition:``, ``player:``,
    ``winner:<name>``) are excluded by design.
    """
    entities = gen3_randbat_entities(showdown_root)
    dex = load_showdown_dex_cached(showdown_root)
    groups: dict[str, list[str]] = {}

    # The encoder emits species/move tokens using Showdown *display names* (e.g.
    # "species:Mr. Mime", "move:Aerial Ace"), lowercased by stable_category_id. Include
    # both the display-name form (what the encoder emits at play time) and the id form
    # (belt-and-suspenders) so the vocabulary is collision-free and complete.
    def _species_strings(species_id: str) -> list[str]:
        info = dex.species_info(species_id)
        names = {species_id}
        if info is not None and info.name:
            names.add(info.name)
        return [f"species:{name}" for name in names]

    def _move_action_strings(move_id: str) -> list[str]:
        info = dex.move_info(move_id)
        names = {move_id}
        if info is not None and info.name:
            names.add(info.name)
        return [f"move:{name}" for name in names]

    species_strings = [s for species in entities["species"] for s in _species_strings(species)]
    # Unown cosmetic formes appear only as display names in battle.
    has_unown = any(_normalize_identifier(species) == "unown" for species in entities["species"])
    if has_unown:
        species_strings += [f"species:{forme}" for forme in UNOWN_FORMES]
    groups["species"] = species_strings

    move_strings = [s for move in entities["moves"] for s in _move_action_strings(move)]
    move_strings += [f"move:{move}" for move in UNIVERSAL_MOVES]
    move_strings += [s for move in UNIVERSAL_MOVES for s in _move_action_strings(move)]
    groups["move_action"] = move_strings
    groups["belief_move"] = [
        f"belief:possible_move:{_normalize_identifier(move)}" for move in entities["moves"]
    ]
    groups["belief_ability"] = [
        f"belief:possible_ability:{_normalize_identifier(ability)}" for ability in entities["abilities"]
    ]
    groups["belief_item"] = [
        f"belief:possible_item:{_normalize_identifier(item)}" for item in entities["items"]
    ]
    groups["status"] = [f"status:{status}" for status in entities["statuses"]]

    structural: list[str] = ["field", "action", "pokemon:self", "pokemon:opponent", "action:move", "action:switch", "winner:none"]
    structural += [f"request_kind:{kind}" for kind in REQUEST_KINDS]
    structural += [f"self_slot:{i}" for i in range(6)] + [f"opponent_slot:{i}" for i in range(6)]
    structural += [f"move_slot:{i}" for i in range(1, 5)] + [f"switch_slot:{i}" for i in range(1, 6)]
    structural += [f"event:{event_type}" for event_type in EVENT_TYPES]
    structural += [f"event_actor:{role}" for role in EVENT_ROLES] + [f"event_target:{role}" for role in EVENT_ROLES]
    # Encoder fallbacks for empty action slots.
    structural += [f"move:slot:{i}" for i in range(1, 5)] + [f"species:slot:{i}" for i in range(1, 6)]
    groups["structural"] = structural

    return groups


def build_gen3_randbat_category_vocabulary(showdown_root: str | Path) -> tuple[int, ...]:
    """Return the sorted, deduplicated category id vocabulary for the closed Gen 3 universe."""
    ids: set[int] = set()
    for strings in gen3_randbat_category_strings(showdown_root).values():
        for value in strings:
            ids.add(stable_category_id(value))
    ids.discard(0)
    return tuple(sorted(ids))


def gen3_randbat_vocabulary_breakdown(showdown_root: str | Path) -> dict[str, int]:
    """Distinct category ids contributed by each source group, plus the deduped total."""
    groups = gen3_randbat_category_strings(showdown_root)
    breakdown: dict[str, int] = {}
    all_ids: set[int] = set()
    for name, strings in groups.items():
        group_ids = {stable_category_id(value) for value in strings}
        group_ids.discard(0)
        breakdown[name] = len(group_ids)
        all_ids |= group_ids
    breakdown["total_distinct"] = len(all_ids)
    return breakdown


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Report/write the Gen 3 random battle embedding category vocabulary.",
    )
    parser.add_argument("--showdown-root", required=True, help="Built Pokemon Showdown checkout root.")
    parser.add_argument("--out", type=Path, default=None, help="Optional path to write the sorted vocab id list as JSON.")
    parser.add_argument("--json", action="store_true", help="Print the breakdown as JSON.")
    args = parser.parse_args(argv)

    breakdown = gen3_randbat_vocabulary_breakdown(args.showdown_root)
    vocab = build_gen3_randbat_category_vocabulary(args.showdown_root)
    if args.json:
        print(json.dumps({"breakdown": breakdown, "vocab_size": len(vocab)}, indent=2))
    else:
        print("Gen 3 randbat full-universe category vocabulary")
        for name, count in breakdown.items():
            if name != "total_distinct":
                print(f"  {name}: {count}")
        print(f"  full-universe vocab size: {breakdown['total_distinct']}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(list(vocab)), encoding="utf-8")
        print(f"wrote vocab to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
