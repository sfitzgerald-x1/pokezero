"""V2-a refill collector: fixed device lanes over a declared game roster.

A fixed number of lanes plays a roster of N already-constructed games. When a
lane's game ends (COMPLETE, CAPPED or REFUSED) the lane takes the next roster
game of its static actor-pairing group, as in benchmark_device_selfplay.retire.
Per-step semantics mirror v5_device_rollout.make_collector.advance with the
actor-bank path: the same encoder, actor forward, sampling keys
sample_keys(stream, frame, seed), refusal rules, Frames record and per-game
terminal fields. Each game keeps the sampling seed of its original block.

Trainable learner rows are compacted on device into a row buffer each tick,
keyed by (stream, frame, owner). Nothing per lane is transferred to the host
per chunk except a few scalars. When the roster is exhausted, lanes shrink
through fixed smaller layouts so the long-game tail does not keep every lane
busy. At the end, finalize() builds the v5 row store (targets, masks and GAE
with the original equations), vectorized over rows and trajectories.

The production collector and its backend are not modified or replaced.
"""
from dataclasses import dataclass, field
from typing import NamedTuple
import math
import time

import jax
import jax.numpy as j
import numpy as np

from pokezero import v5_contract as contract
from pokezero.v5_model import V5ModelConfig
from . import transition, engine_device_trackers as trackers
from . import engine_device_observation as encoder, v5_model
from . import benchmark_device_selfplay as sampling
from . import v5_device_rollout as rollout
from .v5_device_rollout import RUNNING, COMPLETE, CAPPED, REFUSED, STATUS

EMPTY = 4  # lane-only status: no roster game assigned; never a game disposition
NOT_ENDED = -1  # per-game book status before the game ends
SCHEMA = 'pokezero.v2_refill_rows.v1'

# Row-store field groups, matching v5_batch_to_torch(training_batch(...), training=True).
DIRECT_FIELDS = ('categorical_ids', 'numeric_features', 'token_type_ids', 'attention_mask',
    'legal_action_mask', 'action_indices', 'action_probabilities', 'value_estimates',
    'opponent_semantic_action_targets', 'opponent_semantic_action_target_mask',
    'hidden_set_targets', 'hidden_set_species', 'hidden_set_target_mask')
TERMINAL_FIELDS = ('ppo_advantages', 'ppo_advantage_mask', 'ppo_value_targets', 'ppo_value_target_mask',
    'survival_targets', 'final_hp_targets', 'pokemon_end_target_mask', 'outcome_targets', 'outcome_target_mask')
CONSTANT_FIELDS = ('history_mask', 'action_probability_mask', 'training_weights', 'value_estimate_mask',
    'search_policy_targets', 'search_policy_target_mask', 'search_value_targets', 'search_value_target_mask')
KEY_FIELDS = ('streams', 'frames', 'owners', 'games', 'turns')


# --------------------------------------------------------------------------
# Roster: every declared game pre-constructed with its block's original calls.

class RosterArrays(NamedTuple):
    """Device-resident roster, read-only inside the kernels (leading axis N)."""
    state: object
    public: object
    truth: object
    streams: object       # [N] int32
    seeds: object         # [N] uint32 original block sampling seed
    learner: object       # [N,2] bool original learner seat mask
    queue: object         # [N] int32 roster indices ordered by group, then roster order
    queue_offset: object  # [G] int32
    queue_count: object   # [G] int32


@dataclass(frozen=True, eq=False)
class Roster:
    device: RosterArrays
    bindings: rollout.Bindings
    actor_archives: tuple      # global actor archive order; params follow this order
    groups: tuple              # static actor pair per group, ((seat0 actor, seat1 actor), ...)
    group_of_game: np.ndarray  # [N] int32
    streams: np.ndarray        # [N] int32
    seeds: np.ndarray          # [N] int64 (uint32 values)
    actor_indices: np.ndarray  # [N,2] int32 (global archive indices)
    learner: np.ndarray        # [N,2] bool

    @property
    def games(self):return len(self.streams)

    @property
    def counts(self):return np.bincount(self.group_of_game, minlength=len(self.groups)).astype(np.int64)


def roster_from_blocks(blocks):
    """Concatenate original construction blocks into one refill roster.

    Each block is (carry, actor_bank, sampling_seed): carry is the
    rollout.initialize(...) result of v5_device_backend._construct(batch)
    (state_from_pool with batch.constructor_seed, _gender, constructor_truth,
    initialize_with_public); actor_bank is the block's original ActorBank from
    actor_bank_from_assignments; sampling_seed is batch.sampling_seed.
    """
    blocks = tuple(blocks)
    if not blocks:raise ValueError('nonempty declared roster required')
    bindings = blocks[0][1].bindings
    archives = []
    for carry, bank, seed in blocks:
        if type(bank) is not rollout.ActorBank or bank.bindings != bindings:
            raise ValueError('same original bindings and ActorBank per block required')
        if type(carry) is not rollout.Carry or tuple(map(int, np.asarray(carry.streams))) != bank.streams:
            raise ValueError('block carry must hold exactly its bank streams in order')
        if type(seed) is not int or not 0 <= seed < 2**32:raise ValueError('original uint32 block sampling seed required')
        if np.asarray(carry.status).any() or np.asarray(carry.frames).any():raise ValueError('fresh unplayed block carry required')
        for digest in bank.actor_archives:
            if digest not in archives:archives.append(digest)
    lookup = {digest: i for i, digest in enumerate(archives)}
    pairs, learner, streams, seeds = [], [], [], []
    for carry, bank, seed in blocks:
        for row in range(len(bank.streams)):
            pairs.append(tuple(lookup[bank.actor_archives[a]] for a in bank.actor_indices[row]))
            learner.append(bank.learner_mask[row]);streams.append(bank.streams[row]);seeds.append(seed)
    if len(set(streams)) != len(streams):raise ValueError('unique declared streams across the roster required')
    groups = tuple(dict.fromkeys(pairs))
    group_of_game = np.asarray([groups.index(p) for p in pairs], np.int32)
    queue = np.concatenate([np.flatnonzero(group_of_game == g) for g in range(len(groups))]).astype(np.int32)
    counts = np.bincount(group_of_game, minlength=len(groups)).astype(np.int32)
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int32)
    cat = lambda *xs: j.concatenate(xs, axis=0)
    device = RosterArrays(
        jax.tree.map(cat, *(b[0].state for b in blocks)),
        jax.tree.map(cat, *(b[0].public for b in blocks)),
        jax.tree.map(cat, *(b[0].truth for b in blocks)),
        j.asarray(np.asarray(streams, np.int32)), j.asarray(np.asarray(seeds, np.uint32)),
        j.asarray(np.asarray(learner, bool)), j.asarray(queue), j.asarray(offsets), j.asarray(counts))
    return Roster(device, bindings, tuple(archives), groups, group_of_game, np.asarray(streams, np.int32),
        np.asarray(seeds, np.int64), np.asarray(pairs, np.int32), np.asarray(learner, bool))


# --------------------------------------------------------------------------
# Static lane layouts: groups own contiguous lane ranges at every level.

def lane_layout(counts, lanes):
    """Partition lanes over groups in proportion to their games (>=1 each)."""
    counts = [int(c) for c in counts]
    if type(lanes) is not int or lanes <= 0 or not counts or min(counts) <= 0:
        raise ValueError('positive lanes and nonempty groups required')
    if lanes < len(counts):raise ValueError('at least one lane per actor-pairing group required')
    total = sum(counts)
    if lanes >= total:
        layout = list(counts);layout[int(np.argmax(counts))] += lanes - total
        return tuple(layout)
    ideal = [lanes * c / total for c in counts]
    layout = [min(c, max(1, int(math.floor(x)))) for c, x in zip(counts, ideal)]
    while sum(layout) > lanes:
        g = max((i for i in range(len(layout)) if layout[i] > 1), key=lambda i: layout[i] - ideal[i])
        layout[g] -= 1
    while sum(layout) < lanes:
        g = max((i for i in range(len(layout)) if layout[i] < counts[i]), key=lambda i: ideal[i] - layout[i])
        layout[g] += 1
    return tuple(layout)


def shrink_levels(layout, *, factor=4, min_lanes=128):
    """Fixed smaller layouts for the tail, each group scaled by 1/factor."""
    if type(factor) is not int or factor < 2 or type(min_lanes) is not int or min_lanes <= 0:
        raise ValueError('integer shrink factor >=2 and positive minimum lanes required')
    levels = [tuple(layout)]
    while sum(levels[-1]) > min_lanes:
        nxt = tuple(max(1, -(-n // factor)) for n in levels[-1])
        if sum(nxt) < min_lanes:
            nxt = tuple(max(1, -(-n * min_lanes // sum(levels[-1]))) for n in levels[-1])
        if nxt == levels[-1]:break
        levels.append(nxt)
    return tuple(levels)


@dataclass(frozen=True)
class Level:
    layout: tuple
    group_of_lane: np.ndarray  # [L] int32
    group_first: np.ndarray    # [L] int32 first lane of the lane's group
    first_lane: np.ndarray     # [G] int32
    lane_actor: np.ndarray     # [L,2] int32
    forward: tuple             # ((actor, flat lane*2+seat indices or None for all rows), ...)
    refill: bool = True        # False for tail layouts (queue already exhausted)

    @property
    def lanes(self):return int(sum(self.layout))


def make_level(layout, groups, *, refill=True):
    first = np.concatenate(([0], np.cumsum(layout)[:-1])).astype(np.int32)
    group_of_lane = np.repeat(np.arange(len(layout), dtype=np.int32), layout)
    lane_actor = np.asarray(groups, np.int32)[group_of_lane]
    flat = lane_actor.reshape(-1)
    actors = sorted(set(map(int, flat)))
    forward = ((actors[0], None),) if len(actors) == 1 else tuple(
        (a, np.flatnonzero(flat == a).astype(np.int32)) for a in actors)
    return Level(tuple(layout), group_of_lane, first[group_of_lane], first, lane_actor, forward, refill)


# --------------------------------------------------------------------------
# Device carry.

class Lanes(NamedTuple):
    carry: object     # rollout.Carry per lane
    game: object      # [L] int32 roster index, -1 empty
    seed: object      # [L] uint32 sampling seed of the lane's game
    learner: object   # [L,2] bool learner seats of the lane's game
    position: object  # [L,2] int32 learner rows already emitted for the game
    bad: object       # [L,2] bool requested learner seat that was not trainable


class Book(NamedTuple):
    cursor: object           # [G] int32 next queue position per group
    status: object           # [N] int32 NOT_ENDED or final disposition
    winner: object           # [N] int32
    terminal_frame: object   # [N] int32
    terminal_hp: object      # [N,2,6] int32
    terminal_max_hp: object  # [N,2,6] int32
    bad: object              # [N,2] bool
    rows: object             # () int32 rows in the current row segment
    ticks: object            # () int32
    live: object             # () int32 live lane-steps
    physical: object         # () int32 lane-steps executed
    requested: object        # () int32 requested seat views


class Rows(NamedTuple):
    categorical_ids: object
    numeric_features: object
    token_type_ids: object
    attention_mask: object
    legal_action_mask: object
    action_indices: object
    action_probabilities: object
    value_estimates: object
    opponent_semantic_action_targets: object
    opponent_semantic_action_target_mask: object
    hidden_set_targets: object
    hidden_set_species: object
    hidden_set_target_mask: object
    physical: object
    games: object
    owners: object
    streams: object
    frames: object
    positions: object
    turns: object


def row_shapes(model_config):
    """Per-row shape/dtype of the compacted buffer."""
    del model_config
    return Rows((1, 29, 41, j.int32), (1, 29, 132, j.float32), (1, 29, j.int32), (1, 29, j.bool_),
        (9, j.bool_), (j.int32,), (j.float32,), (j.float32,), (j.int32,), (j.bool_,),
        (6, 64, j.float32), (6, j.int32), (6, j.bool_), (12, j.int32),
        (j.int32,), (j.int32,), (j.int32,), (j.int32,), (j.int32,), (j.int32,))


def allocate_rows(capacity, model_config=None):
    return Rows(*(j.zeros((capacity,) + tuple(spec[:-1]), spec[-1]) for spec in row_shapes(model_config)))


ROW_BYTES = sum(int(np.prod(spec[:-1])) * np.dtype(spec[-1]).itemsize for spec in row_shapes(None))


def sample_keys(streams, frames, seeds):
    """sample_keys(streams, frames, sampling_seed) with one seed per lane."""
    def one(seed, root, request):
        key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(seed), root), request)
        return jax.vmap(lambda side: jax.random.fold_in(key, side))(j.arange(2))
    return jax.vmap(one)(seeds, streams, frames)


def _select(mask, new, old):
    return sampling.select_rows(mask, new, old)


def _segment_sum(values, segments, count):
    return jax.ops.segment_sum(values, j.asarray(segments), num_segments=count)


# --------------------------------------------------------------------------
# One tick: original advance semantics, then book, row compaction, refill.

def _make_tick(engine_tables, observation_tables, model_config, labels, bindings, level, *,
        max_requests, emit_frames):
    move_classes = len(bindings.label_vocabulary.moves)
    lane_actor = level.lane_actor
    G = len(level.layout)

    def advance(lanes, params):
        # Mirrors v5_device_rollout.make_collector.advance (actor-bank path):
        # per-lane seed replaces the batch sampling seed and per-lane static
        # actor/learner assignments replace the per-row bank declaration.
        c = lanes.carry
        state = c.state;b = len(c.streams);r = j.arange(b)[:, None];sides = j.arange(2)[None, :]
        live = c.status == RUNNING;state = state._replace(done=state.done | ~live)
        requested = sampling.request_seats(state) & live[:, None]
        physical = rollout._physical(c.public)
        masks = transition.action_mask(state, engine_tables)
        observations = [encoder.encode(state, trackers.readout(c.public, j.full(b, s, j.int32)), observation_tables,
            observer_sides=j.full(b, s, j.int32), legal_action_mask=masks[:, s]) for s in range(2)]
        def output_finite(o):
            return o.valid & j.isfinite(o.value) & (j.abs(o.value) <= 1) & j.all(j.isfinite(o.policy_logits), axis=-1)\
                & (j.all(j.isfinite(o.value_logits), axis=-1) if o.value_logits is not None else True)
        inputs = jax.tree.map(lambda *xs: j.stack(xs, axis=1).reshape((2*b,) + xs[0].shape[1:]),
            *(encoder.actor_inputs(o) for o in observations))
        raw_logits = j.zeros((2*b, 9), j.float32);values = j.zeros(2*b, j.float32);finite = j.zeros(2*b, bool)
        for actor, indices in level.forward:
            if indices is None:
                out = v5_model.forward(params[actor], inputs, model_config, compute_dtype='bfloat16')
                raw_logits = raw_logits.at[:].set(out.policy_logits);values = values.at[:].set(out.value)
                finite = finite.at[:].set(output_finite(out))
            else:
                at = j.asarray(indices, j.int32)
                out = v5_model.forward(params[actor], jax.tree.map(lambda x: x[at], inputs), model_config, compute_dtype='bfloat16')
                raw_logits = raw_logits.at[at].set(out.policy_logits);values = values.at[at].set(out.value)
                finite = finite.at[at].set(output_finite(out))
        raw_logits = raw_logits.reshape(b, 2, 9);values = values.reshape(b, 2);finite = finite.reshape(b, 2)
        legal = j.stack([o.legal_action_mask for o in observations], axis=1)
        safe_logits = j.where(legal, j.nan_to_num(raw_logits, nan=0., posinf=0., neginf=0.), -j.inf)
        safe_logits = j.where(j.any(legal, axis=-1)[..., None], safe_logits, 0.)
        keys = sample_keys(c.streams, c.frames, lanes.seed)
        actions = jax.vmap(lambda key, logits: jax.random.categorical(key, logits))(
            keys.reshape(-1, 2), safe_logits.reshape(-1, 9)).reshape(b, 2).astype(j.int32)
        probabilities = j.take_along_axis(jax.nn.softmax(safe_logits, axis=-1), actions[..., None], axis=-1)[..., 0]
        engine_actions = j.stack([encoder.map_actions(observations[s], state, j.full(b, s, j.int32), actions[:, s]) for s in range(2)], axis=1)
        errors = j.stack([o.error for o in observations], axis=1)
        input_valid = finite & (errors == 0) & j.any(legal, axis=-1) & j.isfinite(probabilities) & (probabilities > 0)
        input_refused = live & j.any(requested & ~input_valid, axis=1)
        step_state = state._replace(done=state.done | input_refused)
        out = transition.step_with_public(step_state, engine_actions, engine_tables, c.public, belief_tables=observation_tables.belief)
        after = out.step.state;refused = input_refused | (live & out.step.invalid_actions)
        completed = live & ~refused & after.done
        frame = c.frames + live.astype(j.int32)
        capped = live & ~refused & ~completed & (frame >= max_requests)
        ended = completed | capped | refused
        status = j.where(refused, REFUSED, j.where(completed, COMPLETE, j.where(capped, CAPPED, c.status)))
        carry = c._replace(state=after, public=out.trackers, frames=frame, status=status,
            winner=j.where(completed, after.winner, c.winner), terminal_frame=j.where(ended, frame, c.terminal_frame),
            terminal_hp=j.where(completed[:, None, None], after.hp, c.terminal_hp),
            terminal_max_hp=j.where(completed[:, None, None], after.max_hp, c.terminal_max_hp))
        slot = j.clip(engine_actions, 0, 3);move = state.moves[r, sides, state.active, slot]
        from .state import usable_moves, bench_indices
        charge = state.charge_move[r, sides, state.active]
        move = j.where(charge > 0, charge, j.where(j.any(usable_moves(state), axis=-1), move, engine_tables.struggle))
        move_label = labels.move[j.clip(move, 0, len(labels.move)-1)]
        targets = j.take_along_axis(bench_indices(state.active), j.clip(engine_actions-4, 0, 4)[..., None], axis=-1)[..., 0]
        species_label = labels.species[j.clip(state.species[r, sides, targets], 0, len(labels.species)-1)]
        semantic = j.where(engine_actions < 4, move_label, j.where(species_label >= 0, move_classes+species_label, -1))
        opponent_targets = semantic[:, ::-1]
        opponent_mask = requested & requested[:, ::-1] & (opponent_targets >= 0) & ~refused[:, None]
        foe = 1 - sides;positions = j.clip(physical[:, :, 6:], 0, 5)
        hidden = c.truth.probabilities[r[:, :, None], foe[:, :, None], positions]
        hs = c.truth.species[r[:, :, None], foe[:, :, None], positions]
        hm = c.truth.known[r[:, :, None], foe[:, :, None], positions] & (physical[:, :, 6:] >= 0) & requested[:, :, None] & ~refused[:, None, None]
        valid = requested & input_valid & ~refused[:, None]
        trainable = valid & lanes.learner
        stack = lambda name: j.stack([getattr(o, name) for o in observations], axis=1)
        record = rollout.Frames(stack('numeric_features'), stack('categorical_ids'), stack('token_type_ids'), stack('attention_mask'),
            legal, c.streams, c.frames, state.turn, requested, trainable, valid, errors, actions, engine_actions, probabilities, values, finite, live & ~refused, physical,
            opponent_targets, opponent_mask, hidden, hs, hm, j.asarray(lane_actor, j.int32), lanes.learner, None)
        return carry, record, ended

    def compact(rows, count, record, lanes):
        # Trainable learner rows only, in (lane, owner) order, written as one
        # contiguous slab at the running count. Host guarantees slack >= 2L.
        b = len(lanes.game);flat = record.trainable.reshape(-1)
        idx = j.nonzero(flat, size=2*b, fill_value=0)[0]
        def take(x):return x.reshape((2*b,) + x.shape[2:])[idx]
        two = lambda x: j.broadcast_to(x[:, None], (b, 2))
        owner = j.broadcast_to(j.arange(2, dtype=j.int32)[None, :], (b, 2))
        slab = Rows(take(record.categorical_ids)[:, None], take(record.numeric_features)[:, None],
            take(record.token_type_ids)[:, None], take(record.attention_mask)[:, None], take(record.legal_action_mask),
            take(record.actions), take(record.probabilities), take(record.values),
            take(record.semantic_targets), take(record.semantic_mask), take(record.hidden_targets),
            take(record.hidden_species), take(record.hidden_mask), take(record.physical),
            take(two(lanes.game)), take(owner), take(two(record.streams)), take(two(record.frames)),
            take(lanes.position), take(two(record.turns)))
        rows = Rows(*(jax.lax.dynamic_update_slice(buffer, update.astype(buffer.dtype), (count,) + (0,)*(buffer.ndim-1))
            for buffer, update in zip(rows, slab)))
        return rows, count + j.sum(flat, dtype=j.int32)

    def tick(lanes, book, rows, roster, params):
        carry, record, ended = advance(lanes, params)
        live = lanes.carry.status == RUNNING
        n = book.status.shape[0]
        bad = lanes.bad | (record.requested & lanes.learner & ~record.trainable)
        # Per-game terminal records at the game's end, keyed by roster index.
        index = j.where(ended, lanes.game, n)
        book = book._replace(
            status=book.status.at[index].set(carry.status, mode='drop'),
            winner=book.winner.at[index].set(carry.winner, mode='drop'),
            terminal_frame=book.terminal_frame.at[index].set(carry.terminal_frame, mode='drop'),
            terminal_hp=book.terminal_hp.at[index].set(carry.terminal_hp, mode='drop'),
            terminal_max_hp=book.terminal_max_hp.at[index].set(carry.terminal_max_hp, mode='drop'),
            bad=book.bad.at[index].set(bad, mode='drop'),
            ticks=book.ticks + 1, live=book.live + j.sum(live, dtype=j.int32),
            physical=book.physical + len(lanes.game), requested=book.requested + j.sum(record.requested, dtype=j.int32))
        rows, count = compact(rows, book.rows, record, lanes)
        book = book._replace(rows=count)
        lanes = lanes._replace(carry=carry, position=lanes.position + record.trainable.astype(j.int32), bad=bad)
        if level.refill:lanes, book = refill(lanes, book, roster, ended | (lanes.game < 0))
        else:
            # Tail layouts exist only after every queue is exhausted: retire only.
            game = j.where(ended, -1, lanes.game);carry = lanes.carry
            lanes = lanes._replace(game=game, carry=carry._replace(status=j.where(game < 0, EMPTY, carry.status),
                state=carry.state._replace(done=carry.state.done | (game < 0))))
        return lanes, book, rows, (record if emit_frames else None)

    def refill(lanes, book, roster, available):
        # retire(): every ended lane takes its group's next declared game.
        L = len(lanes.game);N = roster.streams.shape[0]
        cs = j.cumsum(available.astype(j.int32))
        first = j.asarray(level.group_first)
        before = j.where(first > 0, cs[j.maximum(first - 1, 0)], 0)
        group = j.asarray(level.group_of_lane)
        position = book.cursor[group] + cs - before - 1
        fill = available & (position < roster.queue_count[group])
        index = roster.queue[j.clip(roster.queue_offset[group] + position, 0, N - 1)]
        take = lambda tree: jax.tree.map(lambda x: x[index], tree)
        fresh_state = take(roster.state)
        z = j.zeros(L, j.int32)
        fresh = rollout.Carry(fresh_state, take(roster.public), roster.streams[index], z, z, j.full(L, -1, j.int32), z,
            j.zeros_like(fresh_state.hp), j.zeros_like(fresh_state.max_hp), take(roster.truth))
        carry = _select(fill, fresh, lanes.carry)
        game = j.where(fill, index, j.where(available, -1, lanes.game))
        carry = carry._replace(status=j.where(game < 0, EMPTY, carry.status),
            state=carry.state._replace(done=carry.state.done | (game < 0)))
        lanes = Lanes(carry, game, j.where(fill, roster.seeds[index], lanes.seed),
            j.where(fill[:, None], roster.learner[index], lanes.learner),
            j.where(fill[:, None], 0, lanes.position), j.where(fill[:, None], False, lanes.bad))
        book = book._replace(cursor=book.cursor + _segment_sum(fill.astype(j.int32), level.group_of_lane, G))
        return lanes, book

    return tick, refill


# --------------------------------------------------------------------------
# Collector: compiled chunk per level, host loop, finalization.

@dataclass
class RunResult:
    roster: Roster
    book: Book
    segments: list            # [(Rows, count)]
    stats: dict
    frames: list = field(default_factory=list)  # emit_frames only: [(Frames, level)] per chunk


class RefillCollector:
    """Prepared once per roster layout; compiled kernels are reused per level."""

    def __init__(self, engine_tables, observation_tables, model_config, labels, roster, *, lanes,
            max_requests=4096, chunk_steps=8, shrink_factor=4, min_lanes=128, rows_per_game=128,
            emit_frames=False, donate=True):
        bindings = roster.bindings
        if type(model_config) is not V5ModelConfig or type(bindings) is not rollout.Bindings:
            raise ValueError('explicit model/config bindings required')
        if model_config.arm == 'D':raise ValueError('V2-a refill collector covers A/B; D teacher roots are not retained')
        if model_config.category_vocabulary_sha256 != bindings.category_sha256 or model_config.label_vocabulary.sha256 != bindings.label_vocabulary.sha256:
            raise ValueError('same actual actor category/full-label namespace required')
        if any(type(x) is not int or x <= 0 for x in (chunk_steps, max_requests, lanes, rows_per_game)):
            raise ValueError('finite positive lanes, chunk, request and row bounds required')
        if type(roster) is not Roster:raise ValueError('declared refill roster required')
        self.engine_tables = engine_tables;self.observation_tables = observation_tables
        self.model_config = model_config;self.labels = labels;self.roster = roster
        self.max_requests = max_requests;self.chunk_steps = chunk_steps;self.emit_frames = emit_frames
        layout = lane_layout(roster.counts, lanes)
        self.levels = tuple(make_level(x, roster.groups, refill=i == 0)
            for i, x in enumerate(shrink_levels(layout, factor=shrink_factor, min_lanes=min(min_lanes, lanes))))
        self.rows_per_game = rows_per_game;self.donate = donate
        self._jit = {};self._exe = {};self.compile_seconds = {}

    # -- jitted pieces (compiled executables are preferred when present) -----
    def _tick(self, level):
        return _make_tick(self.engine_tables, self.observation_tables, self.model_config, self.labels, self.roster.bindings,
            self.levels[level], max_requests=self.max_requests, emit_frames=self.emit_frames)

    def _cached(self, key, build):
        if key in self._exe:return self._exe[key]
        if key not in self._jit:self._jit[key] = build()
        return self._jit[key]

    def kernel(self, level):
        def build():
            tick, _ = self._tick(level);steps = self.chunk_steps;spec = self.levels[level]
            def chunk(lanes, book, rows, roster, params):
                def body(state, _):
                    lanes, book, rows, frames = tick(*state, roster, params)
                    return (lanes, book, rows), frames
                (lanes, book, rows), frames = jax.lax.scan(body, (lanes, book, rows), None, length=steps)
                live = _segment_sum((lanes.game >= 0).astype(j.int32), spec.group_of_lane, len(spec.layout))
                return lanes, book, rows, (live, book.cursor, book.rows), frames
            return jax.jit(chunk, donate_argnums=(0, 1, 2) if self.donate else ())
        return self._cached(('kernel', level), build)

    def start(self):
        def build():
            _, refill = self._tick(0);spec = self.levels[0];G = len(spec.layout)
            def start(roster):
                L = spec.lanes;N = roster.streams.shape[0]
                seed_index = j.zeros(L, j.int32)
                take = lambda tree: jax.tree.map(lambda x: x[seed_index], tree)
                state = take(roster.state);z = j.zeros(L, j.int32)
                carry = rollout.Carry(state._replace(done=j.ones(L, bool)), take(roster.public), j.zeros(L, j.int32), z,
                    j.full(L, EMPTY, j.int32), j.full(L, -1, j.int32), z, j.zeros_like(state.hp), j.zeros_like(state.max_hp), take(roster.truth))
                lanes = Lanes(carry, j.full(L, -1, j.int32), j.zeros(L, j.uint32), j.zeros((L, 2), bool),
                    j.zeros((L, 2), j.int32), j.zeros((L, 2), bool))
                zn = j.zeros(N, j.int32);zero = j.zeros((), j.int32)
                book = Book(j.zeros(G, j.int32), j.full(N, NOT_ENDED, j.int32), j.full(N, -1, j.int32), zn,
                    j.zeros((N, 2, 6), j.int32), j.zeros((N, 2, 6), j.int32), j.zeros((N, 2), bool), zero, zero, zero, zero, zero)
                return refill(lanes, book, roster, j.ones(L, bool))
            return jax.jit(start)
        return self._cached(('start',), build)

    def compactor(self, level):
        """Gather live lanes of layout `level` into layout `level+1`, per group."""
        def build():
            old, new = self.levels[level], self.levels[level + 1]
            def compact(lanes):
                live = lanes.game >= 0;L = new.lanes
                group = j.asarray(old.group_of_lane);first = j.asarray(old.group_first)
                cs = j.cumsum(live.astype(j.int32))
                rank = cs - j.where(first > 0, cs[j.maximum(first - 1, 0)], 0) - 1
                target = j.where(live, j.asarray(new.first_lane)[group] + rank, L)
                source = j.full(L, -1, j.int32).at[target].set(j.arange(len(live), dtype=j.int32), mode='drop')
                moved = jax.tree.map(lambda x: x[j.maximum(source, 0)], lanes)
                empty = source < 0
                carry = moved.carry._replace(status=j.where(empty, EMPTY, moved.carry.status),
                    state=moved.carry.state._replace(done=moved.carry.state.done | empty))
                return moved._replace(carry=carry, game=j.where(empty, -1, moved.game))
            return jax.jit(compact)
        return self._cached(('compact', level), build)

    def row_capacity(self):
        return self.rows_per_game * self.roster.games + self.chunk_steps * 2 * self.levels[0].lanes

    def compile(self, params, *, threads=None, progress=None):
        """Explicit lower+compile of every level kernel, the start and compactions."""
        if ('start',) in self._exe:return dict(self.compile_seconds)
        timings = {};device = self.roster.device
        lanes, book = jax.eval_shape(self.start(), device)
        rows = jax.eval_shape(lambda: allocate_rows(self.row_capacity()))
        begin = time.perf_counter();start = self.start().lower(device).compile()
        timings['start'] = time.perf_counter() - begin
        lowered = [];compacts = {}
        for level in range(len(self.levels)):
            begin = time.perf_counter()
            lowered.append(self.kernel(level).lower(lanes, book, rows, device, params))
            timings[f'lower_{self.levels[level].lanes}'] = time.perf_counter() - begin
            if progress:progress(dict(phase='LOWERED', lanes=self.levels[level].lanes, seconds=timings[f'lower_{self.levels[level].lanes}']))
            if level + 1 < len(self.levels):
                begin = time.perf_counter();compact = self.compactor(level)
                compacts[level] = compact.lower(lanes).compile();lanes = jax.eval_shape(compact, lanes)
                timings[f'compact_{self.levels[level].lanes}'] = time.perf_counter() - begin
        def build(level):
            begin = time.perf_counter();executable = lowered[level].compile();seconds = time.perf_counter() - begin
            if progress:progress(dict(phase='COMPILED', lanes=self.levels[level].lanes, seconds=seconds))
            return level, executable, seconds
        if threads and threads > 1 and len(lowered) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=threads) as pool:
                built = list(pool.map(build, range(len(lowered))))
        else:built = [build(level) for level in range(len(lowered))]
        for level, executable, seconds in built:
            self._exe[('kernel', level)] = executable;timings[f'compile_{self.levels[level].lanes}'] = seconds
        for level, executable in compacts.items():self._exe[('compact', level)] = executable
        self._exe[('start',)] = start
        self.compile_seconds.update(timings)
        return timings

    # -- host loop -----------------------------------------------------------
    def run(self, params, *, progress=None):
        roster = self.roster;counts = roster.counts
        if len(params) != len(roster.actor_archives):raise ValueError('one original parameter tree per roster actor archive required')
        capacity = self.row_capacity();segments = []
        rows = allocate_rows(capacity);level = 0
        lanes, book = self.start()(roster.device)
        jax.block_until_ready((rows, lanes, book))
        stats = dict(levels=[dict(lanes=x.lanes, layout=list(x.layout), chunks=0, ticks=0, seconds=0.) for x in self.levels],
            compactions=[], chunk_steps=self.chunk_steps, row_capacity=capacity, row_segments=1)
        frames = [];begin = time.perf_counter();queue_end = None;count = 0
        while True:
            need = self.chunk_steps * 2 * self.levels[level].lanes
            if capacity - count < need:
                segments.append((rows, count));rows = allocate_rows(capacity);count = 0
                book = book._replace(rows=j.zeros((), j.int32));stats['row_segments'] += 1
            clock = time.perf_counter()
            lanes, book, rows, out, chunk_frames = self.kernel(level)(lanes, book, rows, roster.device, params)
            live, cursor, count = jax.device_get(out)
            now = time.perf_counter();entry = stats['levels'][level]
            entry['chunks'] += 1;entry['ticks'] += self.chunk_steps;entry['seconds'] += now - clock
            live = np.asarray(live);cursor = np.asarray(cursor);count = int(count)
            if self.emit_frames:frames.append((jax.device_get(chunk_frames), level))
            exhausted = bool(np.all(cursor >= counts))
            if exhausted and queue_end is None:queue_end = dict(seconds=now - begin, ticks=sum(x['ticks'] for x in stats['levels']))
            if progress:progress(dict(level=self.levels[level].lanes, live=int(live.sum()), started=int(cursor.sum()),
                rows=count + sum(c for _, c in segments), elapsed_seconds=now - begin))
            if exhausted and not live.any():break
            while exhausted and level + 1 < len(self.levels) and np.all(live <= np.asarray(self.levels[level + 1].layout)):
                clock = time.perf_counter();lanes = jax.block_until_ready(self.compactor(level)(lanes))
                stats['compactions'].append(dict(lanes_from=self.levels[level].lanes, lanes_to=self.levels[level + 1].lanes,
                    live=int(live.sum()), seconds=time.perf_counter() - clock, at_seconds=time.perf_counter() - begin))
                level += 1
        total = time.perf_counter() - begin
        segments.append((rows, count))
        book = jax.block_until_ready(book)
        host = jax.device_get(book._replace(status=None, winner=None, terminal_frame=None, terminal_hp=None, terminal_max_hp=None, bad=None, cursor=None))
        stats.update(kernel_seconds=total, queue_exhausted=queue_end,
            tail_seconds=None if queue_end is None else total - queue_end['seconds'],
            tail_ticks=None if queue_end is None else int(host.ticks) - queue_end['ticks'],
            ticks=int(host.ticks), live_lane_steps=int(host.live), physical_lane_steps=int(host.physical),
            requested_views=int(host.requested), occupancy=int(host.live) / max(1, int(host.physical)),
            rows=sum(c for _, c in segments))
        for entry in stats['levels']:
            entry['seconds_per_tick'] = None if not entry['ticks'] else entry['seconds'] / entry['ticks']
        return RunResult(roster, book, segments, stats, frames)


# --------------------------------------------------------------------------
# Finalization: terminal targets and GAE, vectorized.

def trajectory_gae(values, trajectory, position, rewards_last, *, discount, policy_lambda, value_lambda):
    """Exact v5_loss.gae_targets per complete trajectory, batched over trajectories.

    Each trajectory t has rows at positions 0..len-1 (time order); reward is
    zero except the last row (rewards_last[t]); only the last row terminates.
    The float32 operation order equals the original Torch loop: right-aligned
    trajectories are swept backward one time index at a time.
    """
    values = np.asarray(values, np.float32);trajectory = np.asarray(trajectory, np.int64);position = np.asarray(position, np.int64)
    if not len(values):return np.zeros(0, np.float32), np.zeros(0, np.float32)
    count = int(trajectory.max()) + 1
    length = np.bincount(trajectory, minlength=count)
    width = int(length.max());col = width - length[trajectory] + position
    V = np.zeros((count, width), np.float32);V[trajectory, col] = values
    last = np.zeros((count, width), bool);last[trajectory, col] = position == length[trajectory] - 1
    NV = np.zeros_like(V);NV[:, :-1] = V[:, 1:]
    R = np.zeros_like(V);R[:, -1] = np.asarray(rewards_last, np.float32)
    terminated = last;continues = (~terminated).astype(np.float32)
    delta = (R + (np.float32(discount) * (~terminated).astype(np.float32)) * NV) - V
    def trace(lam):
        k = np.float32(discount * lam);running = np.zeros(count, np.float32);result = np.empty_like(delta)
        for index in range(width - 1, -1, -1):
            running = delta[:, index] + (k * continues[:, index]) * running
            result[:, index] = running
        return result
    advantage = trace(policy_lambda)[trajectory, col]
    target = (trace(value_lambda) + V)[trajectory, col]
    return advantage, target


@jax.jit
def _terminal_rows(games, owners, physical, status, winner, fraction):
    complete = status[games] == COMPLETE
    win = winner[games]
    outcome = j.where(complete, j.where(win == 2, 0., j.where(win == owners, 1., -1.)), 0.).astype(j.float32)
    token = j.arange(12)[None, :]
    side = j.where(token < 6, owners[:, None], 1 - owners[:, None])
    known = complete[:, None] & (physical >= 0)
    end = j.where(known, fraction[games[:, None], side, j.clip(physical, 0, 5)], 0.).astype(j.float32)
    return outcome, complete, end, known, (end > 0).astype(j.float32), status[games]


@dataclass(eq=False)
class RowStore:
    """One iteration's trainable learner rows; arrays may be longer than n."""
    arrays: dict
    n: int
    binding: dict
    stats: dict

    def __len__(self):return self.n

    def keys(self):
        """Host (stream, frame, owner) address of every row."""
        get = lambda name: np.asarray(jax.device_get(self.arrays[name][:self.n]))
        return np.stack([get('streams'), get('frames'), get('owners')], axis=1)

    def to_numpy(self):
        # Copy whole buffers to the host and slice there: no device-side copy of the row buffer.
        out = {name: np.asarray(jax.device_get(value))[:self.n] for name, value in self.arrays.items()}
        out.update(self._constants(np))
        return out

    def _constants(self, xp, device=None):
        n = self.n
        kw = {} if device is None else dict(device=device)
        if xp is np:
            return dict(history_mask=np.ones((n, 1), bool), action_probability_mask=np.ones(n, bool),
                training_weights=np.ones(n, np.float32), value_estimate_mask=np.ones(n, bool),
                search_policy_targets=np.zeros((n, 9), np.float32), search_policy_target_mask=np.zeros(n, bool),
                search_value_targets=np.zeros(n, np.float32), search_value_target_mask=np.zeros(n, bool))
        torch = xp
        return dict(history_mask=torch.ones((n, 1), dtype=torch.bool, **kw), action_probability_mask=torch.ones(n, dtype=torch.bool, **kw),
            training_weights=torch.ones(n, dtype=torch.float32, **kw), value_estimate_mask=torch.ones(n, dtype=torch.bool, **kw),
            search_policy_targets=torch.zeros((n, 9), dtype=torch.float32, **kw), search_policy_target_mask=torch.zeros(n, dtype=torch.bool, **kw),
            search_value_targets=torch.zeros(n, dtype=torch.float32, **kw), search_value_target_mask=torch.zeros(n, dtype=torch.bool, **kw))

    def to_torch(self, *, device=None, exact_dtypes=True, status_strings=True):
        """DLPack views (no copy) of the row store, v5_batch_to_torch(training=True) keys.

        exact_dtypes converts integer fields to int64 as v5_batch_to_torch does
        (a copy); otherwise integer tensors stay int32 zero-copy views.
        """
        import torch
        out = {}
        for name in DIRECT_FIELDS + TERMINAL_FIELDS + KEY_FIELDS:
            tensor = torch.from_dlpack(self.arrays[name])[:self.n]
            if device is not None:tensor = tensor.to(device)
            out[name] = tensor
        target = out['numeric_features'].device if device is None else device
        out.update(self._constants(torch, target))
        if exact_dtypes:
            from pokezero.v5_loss import INTEGER_FIELDS, FLOAT_FIELDS
            for name in out:
                if name in INTEGER_FIELDS:out[name] = out[name].long()
                elif name in FLOAT_FIELDS:out[name] = out[name].float()
        code = torch.from_dlpack(self.arrays['terminal_status'])[:self.n]
        out['terminal_status_code'] = code
        if status_strings:
            out['terminal_status'] = tuple(np.asarray(STATUS, dtype=object)[np.asarray(code.cpu())].tolist())
        out.update(self.binding)
        return out


def finalize(result, *, loss_config, discount=1.):
    """Build the iteration row store once all roster games have ended."""
    from pokezero.v5_loss import V5LossConfig
    if type(result) is not RunResult or type(loss_config) is not V5LossConfig:raise ValueError('run result and original loss recipe required')
    if isinstance(discount, bool) or not isinstance(discount, (int, float)) or not np.isfinite(discount) or not 0 <= discount <= 1:
        raise ValueError('finite declared discount required')
    begin = time.perf_counter();roster = result.roster;bindings = roster.bindings
    book = jax.device_get(result.book._replace(rows=None, ticks=None, live=None, physical=None, requested=None, cursor=None))
    status = np.asarray(book.status);winner = np.asarray(book.winner)
    hp = np.asarray(book.terminal_hp);maximum = np.asarray(book.terminal_max_hp);bad = np.asarray(book.bad)
    if np.any(status == NOT_ENDED):raise ValueError('every declared roster game must end before finalization')
    if np.any((status < 0) | (status >= len(STATUS))):raise ValueError('literal disposition outside enum')
    complete = status == COMPLETE
    if np.any(complete & ~np.isin(winner, (0, 1, 2))):raise ValueError('actual terminal winner required')
    if np.any(complete[:, None, None] & ((maximum <= 0) | (hp < 0) | (hp > maximum))):raise ValueError('literal terminal HP geometry required')
    fraction = np.zeros(hp.shape, np.float32)
    fraction[complete] = (hp[complete] / maximum[complete]).astype(np.float32)  # float64 quotient -> float32, as original
    if len(result.segments) == 1:
        rows, n = result.segments[0]
    else:
        n = sum(c for _, c in result.segments)
        rows = Rows(*(j.concatenate([getattr(seg, name)[:c] for seg, c in result.segments], axis=0) for name in Rows._fields))
    if n == 0:raise ValueError('no valid actual learner decisions')
    small = jax.device_get(dict(games=rows.games[:n], owners=rows.owners[:n], positions=rows.positions[:n],
        values=rows.value_estimates[:n], streams=rows.streams[:n], frames=rows.frames[:n]))
    games = np.asarray(small['games']).astype(np.int64);owners = np.asarray(small['owners']).astype(np.int64)
    position = np.asarray(small['positions']).astype(np.int64)
    trajectory = games * 2 + owners
    # Address uniqueness and contiguous per-trajectory positions, vectorized.
    order = np.lexsort((np.asarray(small['frames']), owners, games))
    t_sorted = trajectory[order];p_sorted = position[order]
    starts = np.r_[True, t_sorted[1:] != t_sorted[:-1]]
    expected = np.arange(n) - np.maximum.accumulate(np.where(starts, np.arange(n), 0))
    if not np.array_equal(p_sorted, expected):raise ValueError('duplicate or noncontiguous actual decision address')
    keys = np.stack([np.asarray(small['streams']), np.asarray(small['frames']), owners], axis=1)
    if len(np.unique(keys, axis=0)) != n:raise ValueError('duplicate actual decision address')
    eligible = complete[games] & ~bad[games, owners]
    outcome_host = np.where(winner[games] == 2, 0., np.where(winner[games] == owners, 1., -1.)).astype(np.float32)
    advantage = np.zeros(n, np.float32);value_target = np.zeros(n, np.float32)
    if eligible.any():
        selected = np.flatnonzero(eligible);_, compact_id = np.unique(trajectory[selected], return_inverse=True)
        last_reward = np.zeros(compact_id.max() + 1, np.float32);last_reward[compact_id] = outcome_host[selected]
        a, target = trajectory_gae(np.asarray(small['values'])[selected], compact_id, position[selected], last_reward,
            discount=discount, policy_lambda=loss_config.policy_lambda, value_lambda=loss_config.value_lambda)
        advantage[selected] = a;value_target[selected] = target
    outcome, om, end, end_mask, survival, codes = _terminal_rows(rows.games[:n], rows.owners[:n], rows.physical[:n],
        j.asarray(status), j.asarray(winner), j.asarray(fraction))
    arrays = {name: getattr(rows, name) for name in DIRECT_FIELDS}
    arrays.update(ppo_advantages=j.asarray(advantage), ppo_advantage_mask=j.asarray(eligible),
        ppo_value_targets=j.asarray(value_target), ppo_value_target_mask=j.asarray(eligible),
        survival_targets=survival, final_hp_targets=end, pokemon_end_target_mask=end_mask,
        outcome_targets=outcome, outcome_target_mask=om, terminal_status=codes,
        streams=rows.streams, frames=rows.frames, owners=rows.owners, games=rows.games, turns=rows.turns)
    jax.block_until_ready(arrays)
    binding = dict(schema=contract.TRAINING_CONTRACT_SCHEMA, observation_schema=contract.OBSERVATION_SCHEMA_VERSION_V5,
        category_vocabulary_sha256=bindings.category_sha256, label_vocabulary_sha256=bindings.label_vocabulary.sha256)
    stats = dict(rows=n, trajectories=int(len(np.unique(trajectory))), ppo_rows=int(eligible.sum()),
        finalize_seconds=time.perf_counter() - begin,
        dispositions={name: int((status == code).sum()) for code, name in enumerate(STATUS)})
    return RowStore(arrays, n, binding, stats)


def game_records(result):
    """Host per-game terminal record in roster order (stream, status, frame, winner)."""
    book = jax.device_get(result.book)
    return dict(streams=result.roster.streams, status=np.asarray(book.status), terminal_frame=np.asarray(book.terminal_frame),
        winner=np.asarray(book.winner), terminal_hp=np.asarray(book.terminal_hp), terminal_max_hp=np.asarray(book.terminal_max_hp))
