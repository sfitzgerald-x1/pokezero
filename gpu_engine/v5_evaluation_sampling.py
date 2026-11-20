"""Pinned categorical selection for the owning Showdown evaluation only."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class EvaluationDraw:
    seed: int
    cluster: int
    frame: int
    seat: int

    def choose(self, logits, legal):
        from .v2_collector import sample_keys

        if (jax.config.jax_default_prng_impl != "threefry2x32"
                or jax.config.jax_threefry_partitionable is not True
                or jax.config.jax_high_dynamic_range_gumbel is not False):
            raise ValueError("S0 evaluation requires Threefry and the low-mode categorical")
        logits = np.asarray(logits, dtype=np.float32)
        legal = np.asarray(legal, dtype=bool)
        if logits.shape != (9,) or legal.shape != (9,) or not legal.any() or not np.isfinite(logits).all():
            raise ValueError("finite FP32 logits and nonempty nine-action legal mask required")
        if (any(type(x) is not int or not 0 <= x < 2**32 for x in (self.seed, self.cluster, self.frame))
                or type(self.seat) is not int or self.seat not in (0, 1)):
            raise ValueError("explicit uint32 evaluation address and physical seat required")
        key = sample_keys(jnp.asarray([self.cluster], jnp.uint32),
                          jnp.asarray([self.frame], jnp.uint32),
                          jnp.asarray([self.seed], jnp.uint32))[0, self.seat]
        action, probabilities = draw(key, logits, legal)
        action = int(action)
        probabilities = np.asarray(probabilities)
        if not legal[action] or not np.isfinite(probabilities).all() or probabilities[action] <= 0:
            raise ValueError("masked categorical selected an invalid action")
        return action, probabilities.tolist()


@jax.jit
def draw(key, logits, legal):
    masked = jnp.where(legal, jnp.asarray(logits, jnp.float32), -jnp.inf)
    return jax.random.categorical(key, masked, mode="low"), jax.nn.softmax(masked)
