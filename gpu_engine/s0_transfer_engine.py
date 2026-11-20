"""Evaluate finite transfer pairs on the existing engine refill collector."""

import time

import jax
import jax.numpy as jnp
import numpy as np

from . import transition, v2_collector, v5_device_rollout as rollout
from pokezero.s0_transfer_starts import BIRTH_FIELDS


def make_roster(initialize, pool, bindings, cells, starts, archives):
    """Use separate batches per orientation so stream and chance remain invariant."""
    streams = np.asarray([cell["cluster"] for cell in cells], np.int32)
    seeds = np.asarray([cell["action"] for cell in cells], np.uint32)
    starting = [starts[str(stream)] for stream in streams]
    team_indices = np.asarray([start["team_indices"] for start in starting], np.int32)
    genders = jnp.asarray([start["gender"] for start in starting], jnp.int32)
    keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray([cell["chance"] for cell in cells], jnp.uint32))
    raw = transition.state_from_pool(pool.arrays, team_indices)._replace(gender=genders, keys=keys)
    # The controller records the actual pool data used by the engine before
    # entry mechanics can alter items or abilities.
    host = jax.device_get({field: getattr(raw, field) for field in BIRTH_FIELDS})
    birth = [{field: host[field][i].tolist() for field in BIRTH_FIELDS} for i in range(len(cells))]
    if any(record != start["engine_birth"] for record, start in zip(birth, starting, strict=True)):
        raise ValueError("actual engine start differs from the declared physical roster")
    carry = rollout.initialize(initialize(raw), streams)
    count = len(cells)
    learner = np.asarray([[c["seat"] == 0, c["seat"] == 1] for c in cells])
    devices = v2_collector.RosterArrays(
        carry.state, carry.public, carry.truth, jnp.asarray(streams), jnp.asarray(seeds),
        jnp.asarray(learner), jnp.arange(count, dtype=jnp.int32),
        jnp.zeros((1,), jnp.int32), jnp.asarray([count], jnp.int32),
    )
    # Slot identity comes from the two distinct role loads, including when the
    # immutable archive digest is equal. Neither role is replaced with the other.
    roster = v2_collector.Roster(
        devices, bindings, archives, ((0, 1),), np.zeros(count, dtype=np.int32), streams,
        seeds.astype(np.int64), np.tile(np.array([[0, 1]], np.int32), (count, 1)), learner,
    )
    return roster, birth


def evaluate(prepared, pool, model_config, cell_groups, starts, actors, *, run_sha,
             lanes, max_requests, wall_seconds, publish):
    labels = rollout.compile_labels(pool.metadata(), model_config.label_vocabulary)
    initialize = jax.jit(lambda state: transition.initialize_with_public(
        state, prepared.engine_tables, belief_tables=prepared.observation_tables.belief,
    ))
    collector = None
    for cells in cell_groups:
        if not cells:
            continue
        role, seat = cells[0]["role"], cells[0]["seat"]
        if any(c["role"] != role or c["seat"] != seat for c in cells):
            raise ValueError("engine batch requires one role and orientation")
        sides = (role, "B") if seat == 0 else ("B", role)
        archives = tuple(actors[side][0] for side in sides)
        params = tuple(actors[side][1] for side in sides)
        if params[0] is params[1]:
            raise ValueError("both roles must be loaded independently")
        bindings = rollout.Bindings(model_config.category_vocabulary_sha256, model_config.label_vocabulary,
                                    prepared.source_sha256, archives, (True, True), run_sha)
        roster, birth = make_roster(initialize, pool, bindings, cells, starts, archives)
        if collector is None:
            collector = v2_collector.RefillCollector(
                prepared.engine_tables, prepared.observation_tables, model_config, labels, roster,
                lanes=lanes, max_requests=max_requests,
            )
            collector.compile(params, threads=2)
        else:
            collector.rebind(roster)
        start_clock = time.monotonic()

        play_batch(collector, params, roster, cells, birth, publish=publish,
                   deadline=start_clock + wall_seconds)


def play_batch(collector, params, roster, cells, birth, *, publish, deadline):
    """Publish terminal lanes at chunk boundaries, then close only unfinished cells."""
    published = set()

    def observe(book):
        records = v2_collector.terminal_records(roster, book)
        for i in np.flatnonzero(records["status"] != v2_collector.NOT_ENDED):
            if i in published:
                continue
            cell = cells[i]
            if int(records["streams"][i]) != cell["cluster"]:
                raise ValueError("engine receipt stream differs from its cell")
            status = rollout.STATUS[int(records["status"][i])]
            winner = int(records["winner"][i])
            witness = dict(status=status, winner=winner, frames=int(records["terminal_frame"][i]),
                           hp=records["terminal_hp"][i].tolist(), max_hp=records["terminal_max_hp"][i].tolist(),
                           birth=birth[i])
            score = None
            if status == "COMPLETE":
                if winner not in (0, 1, 2):
                    raise ValueError("engine returned malformed terminal winner")
                score = .5 if winner == 2 else float(winner == cell["seat"])
            else:
                witness["reason"] = f"engine-{status.lower()}"
            publish((cell, status, score, witness))
            published.add(i)
        return time.monotonic() >= deadline

    collector.run(params, observe_book=observe)
    for i, cell in enumerate(cells):
        if i not in published:
            if time.monotonic() < deadline:
                raise ValueError("engine batch ended without a cell terminal")
            publish((cell, "FAILED", None, dict(reason="engine batch wall deadline", birth=birth[i])))
