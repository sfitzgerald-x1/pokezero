"""Selected Gen 3 Explosion/Selfdestruct events, not full move qualification."""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class SelfDestructTable(NamedTuple):
    move: jax.Array
    damp: jax.Array


def from_metadata(metadata):
    vocab=metadata["vocabulary"]
    return SelfDestructTable(
        jnp.asarray([name in {"explosion","selfdestruct"} for name in vocab["moves"]]),
        jnp.asarray([name=="damp" for name in vocab["abilities"]]))


def blocked(table,move,abilities,hp):
    """Damp's AnyTryMove veto precedes self-KO and target accuracy/Protect.

    Only the two current, live active holders participate, not their benches.
    PP has already been deducted when this event runs.
    """
    return table.move[move] & jnp.any(table.damp[abilities] & (hp>0),axis=1)
