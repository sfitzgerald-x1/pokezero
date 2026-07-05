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

from functools import lru_cache
import json
from pathlib import Path
from typing import Iterable, Mapping

from .category_vocab import CategoryVocabulary, build_category_vocabulary, normalize_category_value
from .dex import DYNAMIC_MOVE_EFFECT_LABELS, PHYSICAL_TYPES, SPECIAL_TYPES, load_showdown_dex_cached
from .showdown import TRACKED_VOLATILES, _normalize_identifier, stable_category_id

# The 17 Gen 3 types (no Fairy) and the three damage classes the encoder emits as raw type
# facts (type:<t> on pokemon/move tokens, move_category:<c> on move tokens).
GEN3_TYPES = tuple(sorted(PHYSICAL_TYPES | SPECIAL_TYPES))
GEN3_MOVE_CATEGORIES = ("Physical", "Special", "Status")

# Gen 3 weathers the encoder surfaces on the field token (weather:<id>, normalized ids).
GEN3_WEATHERS = ("raindance", "sunnyday", "sandstorm", "hail")

# move_effect:<id> labels are derived exactly from the dex over the closed randbat move universe
# (see gen3_randbat_category_strings) — the encoder and the vocab call the identical
# dex.move_info(...).effect_label, so coverage is exact and there is no OOV path; no static list.

# Active-mon volatile statuses the encoder surfaces (volatile:<id>, from |-start|/|-end|). Sourced
# from the encoder's closed TRACKED_VOLATILES set so every emitted token has an enumerated row
# (the encoder ignores any -start payload outside this set, so there is no volatile OOV path).
GEN3_VOLATILES = tuple(sorted(TRACKED_VOLATILES))

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

# Showdown can surface happiness-dependent move IDs such as ``return102`` in live requests/events.
# They are the same move for category-embedding purposes; the numeric suffix is dynamic battle
# mechanics, not a distinct Gen 3 randbat vocabulary row.
DYNAMIC_POWER_MOVE_ALIASES = ("return", "frustration")

# Unown's cosmetic formes share one randbat set but appear with forme-specific display
# names in battle (e.g. "Unown-L"). The A forme is the base species "Unown" (no suffix),
# so cosmetic formes start at B. Enumerate them so each is collision-free in vocab.
UNOWN_FORMES = tuple(f"Unown-{letter}" for letter in "BCDEFGHIJKLMNOPQRSTUVWXYZ") + ("Unown-Exclamation", "Unown-Question")

# Bounded structural request kinds the env can surface (see showdown.py _request_kind).
REQUEST_KINDS = ("move", "force_switch", "team_preview", "wait", "none", "unknown")

# Transition-token actor roles (player-relative; the v1 recent-event actor/target roles are gone
# with the recent-event tokens themselves).
TRANSITION_ROLES = ("self", "opponent")

# Transition-token structural families (observation spec v2, corrections item 9). Kinds/outcome/
# effectiveness/side-effect values mirror transitions.py's closed constants; a unit test keeps the
# two in lockstep so encoder emissions can never fall into OOV.
TRANSITION_KINDS = ("move", "switch", "cant")
TRANSITION_OUTCOMES = (
    "normal", "blocked", "immune", "absorbed", "hit-sub", "broke-sub", "endured",
)
TRANSITION_EFFECTIVENESS = ("neutral", "super", "resisted", "immune")
TRANSITION_SIDE_EFFECTS = (
    "none", "status-inflicted", "hazard-set", "hazard-clear", "weather-set",
    "boost", "drain", "heal", "charging",
)

# |cant| reasons the transition tokens can surface as action ids, audited against the pool's
# reachable emitters (normalized by _side_condition_identifier, so "ability: Truant" ->
# "truant" and the move-sourced "Focus Punch" -> "focuspunch"). Gen 3 |cant| sources:
# status (slp/frz/par), flinch, attract, recharge, Disable/Imprison/Taunt suppression,
# broken focus (|cant|POKEMON|Focus Punch| — vendored data/moves.ts onMoveAborted, no gen3
# override; Focus Punch IS in the gen3 randbats movepools), ability: Truant (Slaking),
# ability: Damp (Explosion/Self-Destruct block), and nopp. A comprehensive superset is fine —
# unobserved reasons are harmless extra rows; a MISSING reason OOVs on a live game.
GEN3_CANT_REASONS = (
    "slp", "frz", "par", "flinch", "attract", "recharge", "disable", "imprison",
    "taunt", "nopp", "partiallytrapped", "truant", "damp", "focuspunch",
)

# Turn-merged transition families (v2.1 batch 3; OPT-IN via include_turn_merged so the
# base vocabulary — and therefore every existing checkpoint's embedding size — is
# unchanged until the dual-schema resolution wires the mode in post-#512). tt_phase
# rides the SLOT column; the SECOND sub-block's categoricals need tt2_-prefixed
# families because categorical columns embed as an unordered bag per row — the prefix
# is what binds a label to the second mover. First-mover labels reuse the existing
# families/columns unchanged (including the per-action precedent that actor species and
# switch-target species share one family).
TURN_MERGED_PHASES = ("turn", "lead", "replacement", "extra")
TURN_MERGED_SECOND_STATUSES = ("negated", "pending", "absent")


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


def gen3_randbat_category_strings(
    showdown_root: str | Path, *, include_turn_merged: bool = False
) -> dict[str, list[str]]:
    """Enumerate the categorical strings the encoder emits for the closed Gen 3 universe.

    Grouped by source so the vocabulary breakdown is auditable. Mirrors the templates in
    ``showdown.py``: ``species:<id>``, ``move:<id>``, ``belief:possible_move:<id>``,
    ``belief:possible_ability:<id>``, ``belief:possible_item:<id>``, ``status:<id>``, plus
    the bounded structural tokens. Inherently-dynamic strings (``condition:``, ``player:``,
    ``winner:<name>``) are excluded by design. ``include_turn_merged`` adds the
    turn-merged transition families (tt_phase / tt2_*) on top — opt-in, so the base
    vocabulary size (and existing checkpoints' embedding tables) stay unchanged.
    """
    entities = gen3_randbat_entities(showdown_root)
    dex = load_showdown_dex_cached(showdown_root)
    groups: dict[str, list[str]] = {}

    # The species token uses the Showdown *display name* (e.g. "species:Mr. Mime"); the id
    # form is never emitted, so enumerate display-only (no dead id-form rows). Moves DO use
    # both forms: the action token emits the move id while event-detail emits the display
    # name, so moves keep both.
    def _species_display(species_id: str) -> str:
        info = dex.species_info(species_id)
        return info.name if info is not None and info.name else species_id

    def _move_action_strings(move_id: str) -> list[str]:
        info = dex.move_info(move_id)
        names = {move_id}
        if info is not None and info.name:
            names.add(info.name)
        return [f"move:{name}" for name in names]

    # Unown cosmetic formes are collapsed to the base species via aliases (see
    # gen3_randbat_cosmetic_aliases), so they are NOT enumerated as separate rows here.
    # Functional formes (Deoxys-Attack/-Defense/-Speed) are distinct sets.json keys and
    # therefore remain distinct species rows.
    groups["species"] = [f"species:{_species_display(species)}" for species in entities["species"]]

    move_strings = [s for move in entities["moves"] for s in _move_action_strings(move)]
    move_strings += [f"move:{move}" for move in UNIVERSAL_MOVES]
    move_strings += [s for move in UNIVERSAL_MOVES for s in _move_action_strings(move)]
    groups["move_action"] = move_strings
    # Revealed opponent moves feed the same belief-move buckets as inferred possible_moves (see
    # showdown.py). The protocol can reveal moves that never appear in a randbats *set* entry — the
    # generic "Hidden Power" (before its type is known) and "Struggle" (out of PP) — so the belief
    # universe must include the UNIVERSAL_MOVES too, or those tokens fall into OOV.
    groups["belief_move"] = [
        f"belief:possible_move:{_normalize_identifier(move)}"
        for move in (*entities["moves"], *UNIVERSAL_MOVES)
    ]
    groups["belief_ability"] = [
        f"belief:possible_ability:{_normalize_identifier(ability)}" for ability in entities["abilities"]
    ]
    groups["belief_item"] = [
        f"belief:possible_item:{_normalize_identifier(item)}" for item in entities["items"]
    ]
    groups["status"] = [f"status:{status}" for status in entities["statuses"]]

    structural: list[str] = ["field", "action", "stats", "pokemon:self", "pokemon:opponent", "action:move", "action:switch", "winner:none"]
    structural += [f"request_kind:{kind}" for kind in REQUEST_KINDS]
    # NOTE: party-slot tokens (self_slot/opponent_slot) are intentionally NOT enumerated — the
    # encoder no longer emits them (team order is arbitrary in randbats). The action SLOT column
    # still uses move_slot/switch_slot below.
    structural += [f"move_slot:{i}" for i in range(1, 5)] + [f"switch_slot:{i}" for i in range(1, 6)]
    # Transition-token families (spec v2): role, kind, outcome enum, effectiveness class,
    # side-effect category, and cant-reason action ids. The action column otherwise reuses the
    # move:/species: families enumerated above.
    structural += [f"transition:{role}" for role in TRANSITION_ROLES]
    structural += [f"tt_kind:{kind}" for kind in TRANSITION_KINDS]
    structural += [f"tt_outcome:{outcome}" for outcome in TRANSITION_OUTCOMES]
    structural += [f"tt_effectiveness:{eff}" for eff in TRANSITION_EFFECTIVENESS]
    structural += [f"tt_side_effect:{effect}" for effect in TRANSITION_SIDE_EFFECTS]
    structural += [f"cant:{reason}" for reason in GEN3_CANT_REASONS]
    # Encoder fallbacks for empty action slots.
    structural += [f"move:slot:{i}" for i in range(1, 5)] + [f"species:slot:{i}" for i in range(1, 6)]
    groups["structural"] = structural

    if include_turn_merged:
        # Turn-merged transition families (v2.1 batch 3). The first sub-block reuses the
        # existing families above; the second sub-block gets tt2_-prefixed rows (bag
        # binding — see the constants' comment), and tt_phase replaces tt_kind on the
        # SLOT column for merged rows.
        turn_merged: list[str] = [f"tt_phase:{phase}" for phase in TURN_MERGED_PHASES]
        turn_merged += [f"tt2_status:{status}" for status in TURN_MERGED_SECOND_STATUSES]
        turn_merged += [f"tt2_kind:{kind}" for kind in TRANSITION_KINDS]
        turn_merged += [f"tt2_outcome:{outcome}" for outcome in TRANSITION_OUTCOMES]
        turn_merged += [f"tt2_effectiveness:{eff}" for eff in TRANSITION_EFFECTIVENESS]
        turn_merged += [f"tt2_side_effect:{effect}" for effect in TRANSITION_SIDE_EFFECTS]
        turn_merged += [f"tt2_cant:{reason}" for reason in GEN3_CANT_REASONS]
        # Second-mover species (actor, switch-target, and Baton Pass follow-up share the
        # family — the same collapse the per-action PRIMARY/SECONDARY columns accept) and
        # second-mover move actions (extraction-normalized ids only; display forms are
        # never emitted in the action field).
        turn_merged += [f"tt2_species:{_species_display(species)}" for species in entities["species"]]
        turn_merged += [
            f"tt2_move:{_normalize_identifier(move)}"
            for move in (*entities["moves"], *UNIVERSAL_MOVES)
        ]
        groups["turn_merged"] = turn_merged

    # Raw mechanical type facts the encoder emits for pokemon/move tokens (closed + tiny).
    groups["types"] = (
        [f"type:{type_name}" for type_name in GEN3_TYPES]
        # The typeless "???" type: in Gen 3 Curse is typeless (it only became Ghost in Gen 6).
        # Enumerated so it gets a dedicated embedding row instead of hashing into the OOV band.
        + ["type:???"]
        + [f"move_category:{category}" for category in GEN3_MOVE_CATEGORIES]
    )

    # Weather surfaced on the field token (closed set; encoder normalizes the protocol id).
    groups["weather"] = [f"weather:{weather}" for weather in GEN3_WEATHERS]

    # Move-effect labels the encoder emits on move tokens (move_effect:<id>), derived from the dex
    # over the closed randbat move universe. The encoder calls the identical dex.move_info(...)
    # .effect_label at play time, so this is exact coverage with no OOV path.
    move_effects: set[str] = set(DYNAMIC_MOVE_EFFECT_LABELS)  # type-dependent labels (Curse)
    priorities: set[int] = set()
    for move in (*entities["moves"], *UNIVERSAL_MOVES):
        info = dex.move_info(move)
        if info is None:
            continue
        if info.effect_label:
            move_effects.add(info.effect_label)
        priorities.add(info.priority)
    groups["move_effects"] = [f"move_effect:{effect}" for effect in sorted(move_effects)]

    # Move priority brackets (move_priority:<n>) the encoder emits, derived over the move universe.
    groups["move_priorities"] = [f"move_priority:{priority}" for priority in sorted(priorities)]

    # Active-mon volatile statuses the encoder surfaces (volatile:<id>).
    groups["volatiles"] = [f"volatile:{name}" for name in GEN3_VOLATILES]

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


def gen3_randbat_cosmetic_aliases(showdown_root: str | Path) -> tuple[tuple[int, int], ...]:
    """Map cosmetic-forme species category ids onto their base-species id.

    Unown's lettered formes are purely cosmetic (identical stats/moves/ability, one shared
    randbat set), so the encoder's per-forme tokens (`species:Unown-L`, ...) should collapse
    onto a single trained `species:Unown` row rather than 27 sparse rows. Functional formes
    such as Deoxys-Attack are NOT cosmetic (separate sets, distinct stats) and are not
    aliased. Returns (alias_id, base_id) pairs for use as compact-embedding aliases.
    """
    entities = gen3_randbat_entities(showdown_root)
    has_unown = any(_normalize_identifier(species) == "unown" for species in entities["species"])
    if not has_unown:
        return ()
    base_id = stable_category_id("species:Unown")
    aliases: list[tuple[int, int]] = []
    for forme in UNOWN_FORMES:
        alias_id = stable_category_id(f"species:{forme}")
        if alias_id != base_id:
            aliases.append((alias_id, base_id))
    return tuple(sorted(set(aliases)))


def gen3_category_string_aliases(
    showdown_root: str | Path, *, include_turn_merged: bool = False
) -> dict[str, str]:
    """Category string aliases onto existing base rows."""
    entities = gen3_randbat_entities(showdown_root)
    aliases: dict[str, str] = {}
    has_unown = any(_normalize_identifier(species) == "unown" for species in entities["species"])
    if has_unown:
        aliases.update({f"species:{forme}": "species:Unown" for forme in UNOWN_FORMES})
    move_universe = {_normalize_identifier(move) for move in entities["moves"]} | {
        _normalize_identifier(move) for move in UNIVERSAL_MOVES
    }
    for move in DYNAMIC_POWER_MOVE_ALIASES:
        if move not in move_universe:
            continue
        for power in range(1, 103):
            aliases[f"move:{move}{power}"] = f"move:{move}"
    if include_turn_merged:
        # Mirror the base-family collapses for the second-mover families.
        if has_unown:
            aliases.update({f"tt2_species:{forme}": "tt2_species:Unown" for forme in UNOWN_FORMES})
        for move in DYNAMIC_POWER_MOVE_ALIASES:
            if move not in move_universe:
                continue
            for power in range(1, 103):
                aliases[f"tt2_move:{move}{power}"] = f"tt2_move:{move}"
    return aliases


@lru_cache(maxsize=8)
def _cached_category_vocabulary(
    showdown_root_key: str, oov_buckets: int, include_turn_merged: bool
) -> CategoryVocabulary:
    strings = [
        s
        for group in gen3_randbat_category_strings(
            showdown_root_key, include_turn_merged=include_turn_merged
        ).values()
        for s in group
    ]
    aliases = gen3_category_string_aliases(showdown_root_key, include_turn_merged=include_turn_merged)
    return build_category_vocabulary(strings, oov_buckets=oov_buckets, aliases=aliases)


def gen3_category_vocabulary(
    showdown_root: str | Path, *, oov_buckets: int = 16, include_turn_merged: bool = False
) -> CategoryVocabulary:
    """Build (cached) the string->row CategoryVocabulary for the closed Gen 3 randbat universe.

    This is the single source of truth used at BOTH observation-encode time (the env) and for
    the model's embedding size/config, so rows align deterministically from the closed universe.
    ``include_turn_merged`` appends the turn-merged transition families (v2.1 batch 3) —
    opt-in because it changes the vocabulary size and therefore the embedding-table shape;
    only turn-merged-mode configs may set it.
    """
    return _cached_category_vocabulary(
        str(Path(showdown_root).expanduser().resolve()), oov_buckets, include_turn_merged
    )


def canonicalize_with_cosmetic_aliases(
    vocab_ids: Iterable[int],
    showdown_root: str | Path,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    """Collapse cosmetic-forme ids in an arbitrary vocab onto their base species.

    Used to apply the Unown cosmetic-forme collapse to any compact vocabulary (e.g. one
    collected from observed training rollouts), not just the statically-enumerated
    randbat-dex universe. Drops cosmetic-forme ids from the vocab, ensures each alias base
    id is present, and returns the (vocab, aliases) pair. Functional formes (Deoxys) are not
    cosmetic aliases, so they pass through untouched.
    """
    aliases = gen3_randbat_cosmetic_aliases(showdown_root)
    alias_keys = {alias for alias, _ in aliases}
    base_ids = {base for _, base in aliases}
    vocab = {int(value) for value in vocab_ids if int(value) > 0}
    vocab -= alias_keys
    vocab |= base_ids
    return tuple(sorted(vocab)), aliases


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
