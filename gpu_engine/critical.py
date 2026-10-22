"""Selected Gen 3 critical chances, using the existing fixed [0,16) draw.

Only ordinary stage zero and the catalog's +1 high-crit moves are represented.
Focus Energy, critical-boosting items and later-generation abilities are outside
this subset. No new chance stream or accepted-team metadata is introduced.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


HIGH_CRIT = frozenset({"aeroblast", "aircutter", "blazekick", "crabhammer",
    "crosschop", "karatechop", "leafblade", "poisontail", "razorleaf", "razorwind",
    "skyattack", "slash"})


class CriticalTable(NamedTuple):
    high: jax.Array
    armor: jax.Array


def from_metadata(metadata):
    vocab = metadata["vocabulary"]
    return CriticalTable(jnp.asarray([name in HIGH_CRIT for name in vocab["moves"]]),
        jnp.asarray([name in {"battlearmor", "shellarmor"} for name in vocab["abilities"]]))


def is_critical(table, move, target_ability, roll):
    """Stage zero is 1/16; +1 is 1/8. Armor veto follows the chance event."""
    return (jnp.asarray(roll) < jnp.where(table.high[move], 2, 1)) & ~table.armor[target_ability]
