"""Selected per-Pokemon Gen 3 move type/power, not general callbacks."""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class VariableMoveTable(NamedTuple):
    hidden_power: jax.Array
    returns: jax.Array
    frustration: jax.Array
    special_type: jax.Array
    dark_type: jax.Array


def from_metadata(metadata):
    vocab = metadata["vocabulary"]
    names = vocab["moves"]
    special = {"fire", "water", "grass", "ice", "electric", "dark", "psychic", "dragon"}
    return VariableMoveTable(
        jnp.asarray([name.startswith("hiddenpower") for name in names]),
        jnp.asarray([name == "return" for name in names]),
        jnp.asarray([name == "frustration" for name in names]),
        jnp.asarray([name in special for name in vocab["types"]]),
        jnp.asarray(vocab["types"].index("dark"), jnp.int32))


def resolve(table, move, power, move_type, physical, hp_type, hp_power, happiness):
    """Resolve before stat selection and every type/power-dependent modifier.

    Hidden Power aliases are metadata names only: the Pokemon supplies the
    type/power, as in the canonical battle move. Defaults match pinned source.
    """
    hidden = table.hidden_power[move]
    move_type = jnp.where(hidden, jnp.where(hp_type > 0, hp_type, table.dark_type), move_type)
    physical = jnp.where(hidden, ~table.special_type[move_type], physical)
    power = jnp.where(hidden, jnp.where(hp_power > 0, hp_power, 70), power)
    happiness = jnp.clip(happiness, 0, 255)
    power = jnp.where(table.returns[move], jnp.maximum(1, happiness * 10 // 25), power)
    power = jnp.where(table.frustration[move], jnp.maximum(1, (255-happiness) * 10 // 25), power)
    return power, move_type, physical
