"""Gen 3 singles Pressure PP costs for the selected transition.

The accepted offline pool predates target metadata. The pinned catalog's
non-foe targets are retained here and checked against that source by the oracle;
base power and status/damage category do not determine Pressure eligibility.
Sleep Talk charges only its caller; other callers, two-turn and redirected
actions remain placeholder mechanics.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


# Pinned Gen 3 self/ally targets. Spikes is foeSide but has mustpressure, so
# unlike ordinary foeSide targeting it does consume an extra PP against Pressure.
NON_FOE_MOVES = frozenset("""
assist bide charge conversion followme mirrormove naturepower recover sleeptalk
stockpile camouflage detect growth imprison ingrain magiccoat metronome minimize
moonlight morningsun protect recycle snatch substitute swallow synthesis tailglow
wish acidarmor barrier swordsdance destinybond howl meditate refresh sharpen
teleport grudge milkdrink rest slackoff softboiled agility amnesia batonpass
bellydrum bulkup calmmind cosmicpower defensecurl doubleteam dragondance endure
focusenergy harden irondefense splash withdraw aromatherapy healbell lightscreen
mist reflect safeguard helpinghand
""".split())


class PressureTable(NamedTuple):
    pressure: jax.Array
    targets_foe: jax.Array
    curse: jax.Array
    ghost_type: jax.Array


def from_metadata(metadata):
    vocab = metadata["vocabulary"]
    names = vocab["moves"]
    return PressureTable(jnp.asarray([name == "pressure" for name in vocab["abilities"]]),
        jnp.asarray([name != "_none" and name not in NON_FOE_MOVES for name in names]),
        jnp.asarray([name == "curse" for name in names]),
        jnp.asarray(vocab["types"].index("ghost")))


def cost(table,move,own_types,foe_ability,foe_hp):
    """One base PP plus the current living foe's eligible Pressure event.

The caller gates actual use and excludes synthetic Struggle. This event precedes
accuracy, Protect, Substitute and immunity checks, but follows inability to act.
"""
    targets_foe = table.targets_foe[move] & (~table.curse[move]
        | jnp.any(own_types == table.ghost_type,axis=-1))
    extra = targets_foe & table.pressure[foe_ability] & (foe_hp > 0)
    return 1 + extra.astype(jnp.int32)
