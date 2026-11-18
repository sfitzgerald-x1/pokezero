"""Selected Gen 3 critical chances, using the existing fixed [0,16) draw.

Stage zero, the catalog's +1 high-crit moves and Farfetch'd's Stick (+2) are represented. Focus Energy and
later-generation abilities are outside this subset (neither is in the Gen 3 randbats pool). No new chance stream
or accepted-team metadata is introduced.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


HIGH_CRIT = frozenset({"aeroblast", "aircutter", "blazekick", "crabhammer",
    "crosschop", "karatechop", "leafblade", "poisontail", "razorleaf", "razorwind",
    "skyattack", "slash"})
# sim/battle-actions.ts getDamage (f76228a1): gen <= 5 clamps the ratio to 0..5 and crits with randomChance(1,
# critMult[ratio]), critMult = [0, 16, 8, 4, 3, 2]. On the [0,16) draw that is roll < 16/critMult: ratio 1 -> 1,
# 2 -> 2, 3 -> 4, 5 -> 8. Ratio 4 (1/3) has no [0,16) threshold: only Stick on a high-crit move reaches it, which
# no Gen 3 randbats Farfetch'd carries; the exact-chance planners refuse it (CRIT_RATIO_UNREPRESENTABLE) and the
# probe harness reports it unknown, and the engine reads it as 5/16 here.
CRIT_THRESHOLD = (0, 1, 2, 4, 5, 8)


class CriticalTable(NamedTuple):
    high: jax.Array
    armor: jax.Array
    # Stick (appended): data/mods/gen4/items.ts stick (Gen 3 inherits it): onModifyCritRatio +2 when
    # user.species.id === 'farfetchd'.
    stick: jax.Array       # [I]
    farfetchd: jax.Array   # [S]


def from_metadata(metadata):
    vocab = metadata["vocabulary"]
    return CriticalTable(jnp.asarray([name in HIGH_CRIT for name in vocab["moves"]]),
        jnp.asarray([name in {"battlearmor", "shellarmor"} for name in vocab["abilities"]]),
        jnp.asarray([name == "stick" for name in vocab.get("items", ["_none"])]),
        jnp.asarray([name == "farfetchd" for name in vocab.get("species", ["_none"])]))


def ratio(table, move, *, item=None, species=None):
    """The Gen 3 ModifyCritRatio result: 1, +1 for a high-crit move, +2 for a Farfetch'd holding Stick (the attacker's
    current item and species; None: no Stick, as in Pursuit's intercept, which no Farfetch'd reaches)."""
    value = 1 + table.high[move].astype(jnp.int32)
    if item is not None:
        value = value + 2 * (table.stick[item] & table.farfetchd[species]).astype(jnp.int32)
    return jnp.minimum(value, 5)


def is_critical(table, move, target_ability, roll, *, item=None, species=None):
    """Ratio 1 is 1/16, 2 is 1/8, 3 (Stick) is 1/4 (CRIT_THRESHOLD). Armor veto follows the chance event."""
    threshold = jnp.asarray(CRIT_THRESHOLD, jnp.int32)[ratio(table, move, item=item, species=species)]
    return (jnp.asarray(roll) < threshold) & ~table.armor[target_ability]
