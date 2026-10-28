"""Private own-team facts and their partial v4 device projection.

The host adapter resolves wire strings through the checkpoint vocabulary and
static dex. Numeric column construction happens on device. Public active-state
overlays (Transform, live retyping, boosts, clocks, volatiles and last move) are
still required; this patch alone is not a production observation.
"""
import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokezero import showdown as reference
from pokezero.determinization import _gender_from_details

from .private_requests import parse_action_request


SCHEMA = "pokezero.private_team_facts.v1"
STATS = ("hp", "atk", "def", "spa", "spd", "spe")
CORE = ("HP_FRACTION", "ACTIVE", "LEGAL", "PRESENT", "LEVEL",
        *("BASE_" + name.upper() for name in STATS),
        *("ACTUAL_" + name.upper() for name in STATS), "GENDER_MALE", "GENDER_FEMALE",
        "REVEALED_MOVE_COUNT", "CANDIDATE_SET_COUNT", "UNCERTAINTY",
        "POSSIBLE_ABILITY_COUNT", "POSSIBLE_ITEM_COUNT", "POSSIBLE_MOVE_COUNT",
        "REVEALED_ABILITY", "REVEALED_ITEM")
NUMERIC_COLUMNS = tuple(reference.v4_numeric_index(getattr(reference, "NUMERIC_" + name)) for name in CORE)
CATEGORY_COLUMNS = (reference.CATEGORY_PRIMARY, reference.CATEGORY_SECONDARY, reference.CATEGORY_ROLE,
                    reference.CATEGORY_TYPE_1, reference.CATEGORY_TYPE_2,
                    reference.CATEGORY_BELIEF_ABILITY_OFFSET, reference.CATEGORY_BELIEF_ITEM_OFFSET)


class TeamFacts(NamedTuple):
    owner_side: jax.Array
    vocabulary_binding: jax.Array   # [8] uint32, exact ordered vocabulary/aliases
    present: jax.Array              # [...,6]
    active: jax.Array
    fainted: jax.Array
    level: jax.Array
    gender: jax.Array               # 0 unknown/genderless, 1 male, 2 female
    hp_numerator: jax.Array
    hp_denominator: jax.Array
    base_stats: jax.Array           # [...,6,6] static raw facts, no normalization
    actual_stats: jax.Array         # [...,6,6] private request facts
    categories: jax.Array           # [...,6,7] explicitly vocabulary rows
    ability_known: jax.Array
    item_known: jax.Array


class TeamFeaturePatch(NamedTuple):
    numeric: jax.Array              # [...,6,132]
    categorical: jax.Array          # [...,6,41]
    numeric_columns: jax.Array      # [132] bool; unported fields must not be used
    categorical_columns: jax.Array  # [41] bool
    admitted: jax.Array             # [...], owner and vocabulary matched


def vocabulary_binding(vocabulary):
    payload = json.dumps({"tokens": vocabulary.tokens, "oov_buckets": vocabulary.oov_buckets,
                          "aliases": dict(vocabulary.aliases)}, sort_keys=True, separators=(",", ":")).encode()
    return jnp.asarray(np.frombuffer(hashlib.sha256(payload).digest(), dtype=">u4").astype(np.uint32))


def parse_team_facts(request, *, owner_side, vocabulary, dex=None) -> TeamFacts:
    """Resolve one owning seat's raw request, never an opponent's private team.

    Category payloads are tagged vocabulary rows, not engine species/item IDs.
    The protocol adapter's string lookup is not the numeric observation encoder.
    The engine adapter will gather these same static facts without host strings.
    """
    action_request = parse_action_request(request, owner_side=owner_side)
    team = reference._self_team_from_request(request, f"p{owner_side + 1}")
    present, active, fainted = (np.zeros(6, bool) for _ in range(3))
    level, gender = (np.zeros(6, np.int32) for _ in range(2))
    numerator, denominator = np.zeros(6, np.float32), np.ones(6, np.float32)
    base, actual = np.zeros((6, 6), np.float32), np.zeros((6, 6), np.float32)
    categories = np.zeros((6, len(CATEGORY_COLUMNS)), np.int32)
    ability, item = np.zeros(6, bool), np.zeros(6, bool)
    for index, mon in enumerate(team):
        present[index], active[index] = True, mon.active
        condition = reference._condition_features(mon.condition)
        fainted[index] = condition.fainted
        parsed_level = reference._level_from_details(mon.details)
        if parsed_level is not None:
            if not 0 <= parsed_level < 2**31:
                raise ValueError("private level exceeds int32 transport")
            level[index] = parsed_level
        gender[index] = {"M": 1, "F": 2}.get(_gender_from_details(mon.details), 0)
        head = str(mon.condition or "").split()
        if head and "/" in head[0]:
            try:
                a, b = (float(x) for x in head[0].split("/", 1))
            except ValueError as exc:
                raise ValueError("invalid private HP condition") from exc
            limits = np.finfo(np.float32)
            if any(not np.isfinite(x) or abs(x) > limits.max or (x != 0 and abs(x) < limits.tiny) for x in (a, b)):
                raise ValueError("private HP exceeds normal float32 transport")
            if b != 0:
                numerator[index], denominator[index] = a, b
        info = reference._species_info_base_fallback(dex, mon.species) if dex is not None else None
        if info is not None:
            base[index] = [info.base_stats.get(name, 0) for name in STATS]
        if mon.stats:
            try:
                values = [float(mon.stats.get(name, 0)) for name in STATS]
            except (OverflowError, ValueError, TypeError) as exc:
                raise ValueError("private stat exceeds finite float32 transport") from exc
            if any(not np.isfinite(x) or abs(x) > np.finfo(np.float32).max for x in values):
                raise ValueError("private stat exceeds finite float32 transport")
            actual[index] = values
        ability[index], item[index] = bool(mon.ability), bool(mon.item)
        types = info.types if info is not None else ()
        labels = (f"species:{mon.species}", f"status:{condition.status}", "pokemon:self",
                  f"type:{types[0]}" if types else "", f"type:{types[1]}" if len(types) > 1 else "",
                  f"belief:possible_ability:{reference._normalize_identifier(mon.ability)}" if mon.ability else "",
                  f"belief:possible_item:{reference._normalize_identifier(mon.item)}" if mon.item else "")
        categories[index] = [vocabulary.encode(label) for label in labels]
    return TeamFacts(action_request.owner_side, vocabulary_binding(vocabulary),
        *(jnp.asarray(x) for x in (present, active, fainted, level, gender, numerator, denominator,
                                  base, actual, categories, ability, item)))


def project_team(facts: TeamFacts, *, observer_side, expected_vocabulary_binding) -> TeamFeaturePatch:
    """Write only the owned base columns; leave required public overlays explicit."""
    if type(observer_side) is int:
        if observer_side not in (0, 1):
            raise ValueError("observer side must be 0 or 1")
        observer = jnp.asarray(observer_side, jnp.int32)
    else:
        if getattr(observer_side, "dtype", None) != jnp.dtype(jnp.int32):
            raise ValueError("device observer sides must be int32, without coercion")
        observer = jnp.asarray(observer_side)
    binding = jnp.asarray(expected_vocabulary_binding)
    if binding.shape != (8,) or binding.dtype != jnp.uint32:
        raise ValueError("expected eight uint32 vocabulary binding words")
    admitted = (observer >= 0) & (observer <= 1) & (observer == facts.owner_side) & jnp.all(
        facts.vocabulary_binding == binding, axis=-1)
    visible = facts.present & admitted[..., None]
    hp = jnp.clip(facts.hp_numerator / jnp.where(facts.hp_denominator != 0, facts.hp_denominator, 1), 0, 1)
    zeros = jnp.zeros_like(hp)
    values = [hp, facts.active, ~facts.fainted, facts.present, jnp.minimum(1, facts.level / 100)]
    values.extend(jnp.minimum(1, facts.base_stats[..., i] / 200) for i in range(6))
    values.extend(jnp.minimum(1, facts.actual_stats[..., i] / 714) for i in range(6))
    values.extend([facts.gender == 1, facts.gender == 2, zeros, zeros, zeros,
                   facts.ability_known, facts.item_known, zeros, facts.ability_known, facts.item_known])
    compact = jnp.stack([jnp.asarray(value, jnp.float32) for value in values], axis=-1)
    numeric = jnp.zeros((*facts.present.shape, 132), jnp.float32).at[..., jnp.asarray(NUMERIC_COLUMNS)].set(
        jnp.where(visible[..., None], compact, 0))
    categorical = jnp.zeros((*facts.present.shape, 41), jnp.int32).at[..., jnp.asarray(CATEGORY_COLUMNS)].set(
        jnp.where(visible[..., None], facts.categories, 0))
    return TeamFeaturePatch(numeric, categorical,
        jnp.zeros(132, bool).at[jnp.asarray(NUMERIC_COLUMNS)].set(True),
        jnp.zeros(41, bool).at[jnp.asarray(CATEGORY_COLUMNS)].set(True), admitted)
