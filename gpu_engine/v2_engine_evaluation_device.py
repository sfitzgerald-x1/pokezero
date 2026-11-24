"""Batched PROGRESS games using V2's transition and policy encoders."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import benchmark_device_selfplay as sampling
from . import engine_device_observation as encoder
from . import engine_device_trackers as trackers
from . import transition, v5_model
from .v2_engine_evaluation import disposition
from .v5_device_rollout import CAPPED, COMPLETE, REFUSED, RUNNING


class Batch(NamedTuple):
    state: object
    public: object
    requests: jax.Array
    status: jax.Array


def champion_slot(evaluator, params, observation):
    return evaluator(params, encoder.actor_inputs(observation))


def _observe(state, public, owners, legal, tables):
    owner_legal = legal[jnp.arange(len(owners)), owners]
    return encoder.encode(state, trackers.readout(public, owners), tables,
                          observer_sides=owners, legal_action_mask=owner_legal)


def _choose(output, observation):
    legal = observation.legal_action_mask
    logits = output.policy_logits
    action = jnp.argmax(jnp.where(legal, logits, -jnp.inf), axis=-1).astype(jnp.int32)
    valid = (output.valid & jnp.isfinite(output.value) & jnp.all(jnp.isfinite(logits), axis=-1)
             & (observation.error == 0) & jnp.any(legal, axis=-1))
    return action, valid


def make_batch(engine_tables, candidate_tables, model_config, opponent, *, max_requests):
    """Compile an evaluation-only batch with one candidate and one opponent per lane.

    `opponent` selects a legal engine dispatch action and reports whether its
    requested input was usable. Its parameters are passed dynamically so the
    same executable can serve multiple milestone checkpoints.
    """
    def evaluate(raw, seats, seeds, enabled, candidate_params, opponent_params):
        state, public = transition.initialize_with_public(
            raw, engine_tables, belief_tables=candidate_tables.belief)
        rows = jnp.arange(len(seats))
        start = Batch(state, public, jnp.zeros_like(seats), jnp.where(enabled, RUNNING, REFUSED))

        def tick(batch):
            live = batch.status == RUNNING
            state = batch.state._replace(done=batch.state.done | ~live)
            requested = sampling.request_seats(state)
            legal = transition.action_mask(state, engine_tables)
            own = _observe(state, batch.public, seats, legal, candidate_tables)
            output = v5_model.forward(candidate_params, encoder.actor_inputs(own), model_config,
                                      compute_dtype='float32')
            own_action, own_valid = _choose(output, own)
            own_dispatch = encoder.map_actions(own, state, seats, own_action)
            foe = 1 - seats
            foe_dispatch, foe_valid = opponent(opponent_params, state, batch.public, foe, legal,
                                               seeds, batch.requests)
            actions = jnp.zeros_like(legal[..., 0], dtype=jnp.int32)
            actions = actions.at[rows, seats].set(own_dispatch).at[rows, foe].set(foe_dispatch)
            invalid = ((requested[rows, seats] & ~own_valid)
                       | (requested[rows, foe] & ~foe_valid))
            stepped = transition.step_with_public(
                state._replace(done=state.done | invalid), actions, engine_tables, batch.public,
                belief_tables=candidate_tables.belief)
            after = stepped.step.state
            requests = batch.requests + live.astype(jnp.int32)
            refused, complete, capped = disposition(live=live,
                refused=invalid | stepped.step.invalid_actions, done=after.done,
                requests=requests, cap=max_requests)
            status = jnp.where(refused, REFUSED, jnp.where(complete, COMPLETE,
                               jnp.where(capped, CAPPED, batch.status)))
            return Batch(after, stepped.trackers, requests, status)

        end = jax.lax.while_loop(lambda batch: jnp.any(batch.status == RUNNING), tick, start)
        return end.status, end.state.winner, end.requests

    return jax.jit(evaluate)


def champion_opponent(evaluator, tables):
    def choose(params, state, public, sides, legal, seeds, requests):
        observation = _observe(state, public, sides, legal, tables)
        action, valid = _choose(champion_slot(evaluator, params, observation), observation)
        return encoder.map_actions(observation, state, sides, action), valid

    return choose
