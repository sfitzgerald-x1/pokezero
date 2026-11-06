"""Paired head-to-head where seat A chooses by search at every decision and seat B plays its raw policy.

Same games as `v2_head_to_head.py`: a declared block of the diagnostic config, played twice on identical
initial states with A in seat 0 and then in seat 1 (the swapped copy has stream - SWAP_OFFSET), so the pairs,
teams and seeds equal that tool's. Both seat orders run as one lockstep batch on the GPU engine. Each step is
the collector's `advance` split at the action choice: one jitted call encodes both seats, forwards each
seat's actor and samples raw actions with the collector's keys; the host then replaces A's sampled action
with the search choice where it searches; a second jitted call maps actions and steps the engine with the
collector's disposition rules.

Search (A's head prices both seats; B's model is never used by search):
- `--pricer depth1`: D's FramePricer (vectorized, host-prepared worlds); `depth2`: the Depth2Pricer.
- W = 2 * `--world-blocks`: one pricer call per independent recipe seed block; any block refusal refuses.
- The detach boundary is D's own (`device_public_worlds.detach_roots`): A's own cells and the owner-redacted
  public tracker only. Every root is priced (the quarter selection is bypassed, as in U1).
- `--reduction`: none (raw play, no search), raw (full search path, then the raw sampled action: the plumbing
  gate), average (D's completed-Q argmax; at depth 2 over learner-max backups), hedge (Bayesian zero-sum LP,
  sampled; at depth 2 over per-world turn-2 zero-sum values), blend (the LP blended with the reply prior at
  --blend, sampled).
- A refused root plays the raw sampled action, exactly as raw play would.
- `--never-borrow`: at every searched decision, re-run detach, plans, worlds, prices and choice from a context
  whose true-state cells other than A's own are another game's; any byte difference is counted (gate c).

All paths are command-line arguments.
"""
import argparse
import collections
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from gpu_engine.v2_bench_collector import log
from gpu_engine.v2_head_to_head import SWAP_OFFSET, snapshot

BLOCK_DOMAIN = 'v2s.world-block.v1'
REDUCTIONS = ('none', 'raw', 'average', 'hedge', 'blend')
TURN_BANDS = ((0, 3), (4, 9), (10, 19), (20, 10**9))   # the engine's turn counter; the first band also holds turn 0


def world_block_seeds(seed, blocks):
    """Block 0 is the declared search seed; later blocks are fixed derivations of it (U1's construction)."""
    if type(seed) is not int or type(blocks) is not int or blocks < 1:raise ValueError('declared seed and block count required')
    return (seed,) + tuple(int.from_bytes(hashlib.sha256(json.dumps([BLOCK_DOMAIN, seed, k]).encode()).digest()[:8], 'big')
        for k in range(1, blocks))


def band(turn):
    for low, high in TURN_BANDS:
        if low <= turn <= high:return f'{max(low, 1)}-{high}' if high < 10**9 else f'{low}+'
    raise ValueError('nonnegative turn required')


def make_step(engine_tables, observation_tables, model_config, *, sampling_seed, max_requests):
    """The collector's advance (v5_device_rollout.make_collector, actor-bank path) split at the action choice.

    decide(carry, params, a_side) -> device dict; apply(carry, decided, actions) -> carry. With actions equal to
    decided['actions'] the pair is the collector's advance for a two-actor bank with A at `a_side` per row.
    """
    import jax
    import jax.numpy as j
    from gpu_engine import transition, engine_device_trackers as trackers, engine_device_observation as encoder, v5_model
    from gpu_engine import benchmark_device_selfplay as sampling
    from gpu_engine.v5_device_rollout import RUNNING, COMPLETE, CAPPED, REFUSED

    def decide(c, params, a_side):
        state = c.state;b = len(c.streams)
        live = c.status == RUNNING;state = state._replace(done=state.done | ~live)
        requested = sampling.request_seats(state) & live[:, None]
        masks = transition.action_mask(state, engine_tables)
        observations = [encoder.encode(state, trackers.readout(c.public, j.full(b, s, j.int32)), observation_tables,
            observer_sides=j.full(b, s, j.int32), legal_action_mask=masks[:, s]) for s in range(2)]
        def output_finite(o):
            return o.valid & j.isfinite(o.value) & (j.abs(o.value) <= 1) & j.all(j.isfinite(o.policy_logits), axis=-1) \
                & (j.all(j.isfinite(o.value_logits), axis=-1) if o.value_logits is not None else True)
        inputs = jax.tree.map(lambda *xs: j.stack(xs, axis=1).reshape((2 * b,) + xs[0].shape[1:]), *(encoder.actor_inputs(o) for o in observations))
        raw_logits = j.zeros((2 * b, 9), j.float32);values = j.zeros(2 * b, j.float32);finite = j.zeros(2 * b, bool)
        rows = j.arange(b)
        for actor, at in ((0, 2 * rows + a_side), (1, 2 * rows + 1 - a_side)):
            out = v5_model.forward(params[actor], jax.tree.map(lambda x: x[at], inputs), model_config, compute_dtype='bfloat16')
            raw_logits = raw_logits.at[at].set(out.policy_logits);values = values.at[at].set(out.value)
            finite = finite.at[at].set(output_finite(out))
        raw_logits = raw_logits.reshape(b, 2, 9);values = values.reshape(b, 2);finite = finite.reshape(b, 2)
        legal = j.stack([o.legal_action_mask for o in observations], axis=1)
        safe_logits = j.where(legal, j.nan_to_num(raw_logits, nan=0., posinf=0., neginf=0.), -j.inf)
        safe_logits = j.where(j.any(legal, axis=-1)[..., None], safe_logits, 0.)
        keys = sampling.sample_keys(c.streams, c.frames, sampling_seed=sampling_seed)
        actions = jax.vmap(lambda key, logits: jax.random.categorical(key, logits))(
            keys.reshape(-1, 2), safe_logits.reshape(-1, 9)).reshape(b, 2).astype(j.int32)
        probabilities = j.take_along_axis(jax.nn.softmax(safe_logits, axis=-1), actions[..., None], axis=-1)[..., 0]
        errors = j.stack([o.error for o in observations], axis=1)
        input_valid = finite & (errors == 0) & j.any(legal, axis=-1) & j.isfinite(probabilities) & (probabilities > 0)
        own = dict(numeric_features=j.stack([o.numeric_features for o in observations], axis=1)[rows, a_side],
            categorical_ids=j.stack([o.categorical_ids for o in observations], axis=1)[rows, a_side],
            token_type_ids=j.stack([o.token_type_ids for o in observations], axis=1)[rows, a_side],
            attention_mask=j.stack([o.attention_mask for o in observations], axis=1)[rows, a_side])
        return dict(state=state, live=live, requested=requested, legal=legal, raw_logits=raw_logits, actions=actions,
            input_valid=input_valid, observations=observations, turn=state.turn, own_inputs=own)

    def apply(c, decided, actions):
        state = decided['state'];b = len(c.streams);live = decided['live'];requested = decided['requested']
        observations = decided['observations']
        engine_actions = j.stack([encoder.map_actions(observations[s], state, j.full(b, s, j.int32), actions[:, s]) for s in range(2)], axis=1)
        input_refused = live & j.any(requested & ~decided['input_valid'], axis=1)
        step_state = state._replace(done=state.done | input_refused)
        out = transition.step_with_public(step_state, engine_actions, engine_tables, c.public, belief_tables=observation_tables.belief)
        after = out.step.state;refused = input_refused | (live & out.step.invalid_actions)
        completed = live & ~refused & after.done
        frame = c.frames + live.astype(j.int32)
        capped = live & ~refused & ~completed & (frame >= max_requests)
        ended = completed | capped | refused
        status = j.where(refused, REFUSED, j.where(completed, COMPLETE, j.where(capped, CAPPED, c.status)))
        return c._replace(state=after, public=out.trackers, frames=frame, status=status,
            winner=j.where(completed, after.winner, c.winner), terminal_frame=j.where(ended, frame, c.terminal_frame),
            terminal_hp=j.where(completed[:, None, None], after.hp, c.terminal_hp),
            terminal_max_hp=j.where(completed[:, None, None], after.max_hp, c.terminal_max_hp))

    return jax.jit(decide), jax.jit(apply)


def counterfactual(state, rows, owners, perm, own_fields):
    """Every true-state cell from game perm[row], except A's own cells (the detach boundary's inputs) at (row, owner)."""
    import jax
    import jax.numpy as j
    r = j.asarray(rows, j.int32);o = j.asarray(owners, j.int32);p = j.asarray(perm, j.int32)
    swapped = jax.tree.map(lambda x: x[p], state)
    return swapped._replace(**{name: getattr(swapped, name).at[r, o].set(getattr(state, name)[r, o]) for name in own_fields})


def _flatten(value, path, out):
    """(path, kind, payload) for every leaf, recursing through mappings (any Mapping, incl. MappingProxyType, in
    key order), named tuples, tuples and lists; arrays compare by dtype, shape and bytes. Unknown types raise."""
    from collections.abc import Mapping
    if isinstance(value, Mapping):
        out.append((path, 'mapping', tuple(value)))
        for key, item in value.items():_flatten(item, path + (key,), out)
    elif isinstance(value, tuple) and hasattr(value, '_fields'):
        out.append((path, 'namedtuple', (type(value).__name__,) + tuple(value._fields)))
        for key, item in zip(value._fields, value):_flatten(item, path + (key,), out)
    elif isinstance(value, (tuple, list)):
        out.append((path, type(value).__name__, len(value)))
        for i, item in enumerate(value):_flatten(item, path + (i,), out)
    elif value is None or isinstance(value, (bool, int, float, str, bytes)):
        out.append((path, type(value).__name__, value))
    elif hasattr(value, 'shape') and hasattr(value, 'dtype'):
        x = np.asarray(value)
        if x.dtype == object:raise TypeError('object array in a compared tree: ' + str(path))
        out.append((path, 'array', (x.dtype.str, x.shape, x.tobytes())))
    else:raise TypeError('uncompared type %s at %s' % (type(value).__name__, path))
    return out


def leaves_equal(a, b):
    """Exact structural and byte equality of two host/device trees (see _flatten)."""
    return _flatten(a, (), []) == _flatten(b, (), [])


def root_equal(a, b):
    return (a.address == b.address and a.source_binding_sha256 == b.source_binding_sha256
        and a.reply_private_history_available == b.reply_private_history_available
        and leaves_equal(a.own, b.own) and leaves_equal(a.view, b.view) and leaves_equal(a.public, b.public))


def world_equal(a, b):
    return (a.index == b.index and a.refusals == b.refusals and a.kernel_seed == b.kernel_seed
        and a.reply_private_history_available == b.reply_private_history_available
        and (a.state is None) == (b.state is None) and (a.state is None or (leaves_equal(a.state, b.state) and leaves_equal(a.public, b.public))))


class Search:
    """Host side of the search seat: detach, plans, priced worlds, reduction, choice."""

    def __init__(self, prepared, plan, params, *, pricer, reduction, blend, world_blocks, seed, capacities, depth2_options, workers=1):
        import jax
        from pokezero import v5_gumbel as original
        from gpu_engine import device_public_worlds as worlds, device_public_world_state as states, device_public_teacher as teacher
        from gpu_engine.device_public_teacher_batch import FramePricer
        self.original, self.worlds = original, worlds
        metadata = json.loads(prepared.metadata_json)
        self.source = worlds.compile_source(prepared.source, metadata, expected_source_hash=prepared.source.metadata.source_hash,
            expected_source_rows_sha256=plan.model.label_vocabulary.source_binding_sha256)
        self.materializer = states.compile_tables(self.source, metadata)
        executor = teacher.make_executor(prepared.engine_tables, prepared.observation_tables, enable_device_teacher=True, batch_joint=True)
        self.capacities = tuple(sorted(capacities));self.kind = pricer
        if pricer == 'depth1':
            self.pricer = FramePricer(executor, plan.model, capacity=self.capacities[-1], actor_capacity=1, vectorized=True, host_prep=True)
        elif pricer == 'depth2':
            from gpu_engine.device_depth2_pricer import Depth2Pricer
            self.pricer = Depth2Pricer(executor, plan.model, capacity=self.capacities[-1], actor_capacity=1, **depth2_options)
        else:raise ValueError('pricer depth1 or depth2 required')
        self.params = (params,);self.seed = seed;self.workers = workers;self.pool = None
        self._reduce = original.reduce_completed_q;self.events = collections.Counter()
        self.configure(reduction=reduction, blend=blend, world_blocks=world_blocks)

    def configure(self, *, reduction, blend, world_blocks):
        """Per-cell settings; the pricer and its compiled kernels are shared across cells of one process."""
        if reduction not in REDUCTIONS[1:]:raise ValueError('search reduction required')
        self.reduction = reduction;self.blend = blend;self.world_blocks = world_blocks
        self.recipes = tuple(self.original.Recipe(s) for s in world_block_seeds(self.seed, world_blocks))
        self.timing = collections.Counter();self.calls = [];self.events = collections.Counter()
        if self.kind == 'depth2' and reduction in ('hedge', 'blend') and self.workers > 1 and self.pool is None:
            import concurrent.futures, multiprocessing
            self.pool = concurrent.futures.ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context('spawn'))

    def _reduce_tolerating_rounding(self, *, root_logits, root_legal, worlds):
        """D's reduce_completed_q, except its known rounding edge does not abort the pricer call.

        When every reply-weighted value is +-1, the fsum of reply probabilities can exceed one by an ulp and
        RootImprovement.validate rejects completed-Q of 1 + 2**-52. The world prices are exact and in range;
        only D's target validation fails. The search seat uses the world prices, not D's target, so such a
        root counts as priced and the event is counted. Any other failure is re-raised.
        """
        from gpu_engine import search_hedge as hedge
        try:return self._reduce(root_logits=root_logits, root_legal=root_legal, worlds=worlds)
        except ValueError as error:
            average = hedge.averaging(worlds, root_legal)
            if (str(error) != 'finite in-range numerical value required' or average is None
                    or any(v is not None and abs(v) > 1 + 1e-12 for v in average.q)):raise
            self.events['d_reduction_rounding_edge'] += 1
            return self.original.Reduction('LABELED', None, 0, 0, 0., None)

    def _price(self, requests, recipe):
        """Price in calls of a fixed capacity (padding with the chunk's first request).

        Returns one (prices, reason) per request: the worlds' WorldPrices for the reduction, or None and the
        first refusing world's reason. Depth 2 backs each root up here (hedged turn-2 games in the pool).
        """
        from gpu_engine import search_hedge as hedge
        surfaces = []
        top = self.capacities[-1]
        for offset in range(0, len(requests), top):
            chunk = requests[offset:offset + top]
            size = next(c for c in self.capacities if c >= len(chunk))
            # The Depth2Pricer's fixed capacity pads with inactive rows itself; otherwise repeat real requests.
            padded = list(chunk) if getattr(self.pricer, 'fixed_capacity', False) else list(chunk) + [chunk[0]] * (size - len(chunk))
            with patch.object(self.original, 'reduce_completed_q', self._reduce_tolerating_rounding):
                output = self.pricer.price(padded, self.materializer, recipe, self.params)
            timing = self.pricer.last_timing
            self.calls.append({k: v for k, v in timing.items() if isinstance(v, (int, float, str))})
            for key in ('world_prepare_seconds', 'device_seconds', 'compile_seconds', 'lower_seconds', 'frame_wall_seconds'):
                self.timing[key] += timing.get(key, 0.) or 0.
            clock = time.perf_counter()
            if self.kind == 'depth1':
                for result in output[:len(chunk)]:
                    if result.status != 'LABELED':
                        surfaces.append((None, next((r['reason'] for r in result.world_reports if r.get('reason')), result.reason)));continue
                    surfaces.append((tuple(self.original.WorldPrices(r['opponent_requested'], r['reply_legal'], r['reply_logits'], r['joint_values'])
                        for r in result.world_reports), None))
            else:
                from gpu_engine import depth2_backups
                causes = depth2_backups.refusal_summary(output)['roots']
                reasons = [None if not causes[i]['refused'] else 'depth2:' + causes[i]['cause'] + ':' + str(next(
                    (r['reason'] for r in output.world_reports[i] if r['status'] != 'COMPLETE'), None)) for i in range(len(chunk))]
                if self.reduction in ('hedge', 'blend'):
                    roots = [i for i in range(len(chunk)) if reasons[i] is None]
                    fields = [{k: getattr(output, k)[i] for k in hedge.DEPTH2_FIELDS} for i in roots]
                    backed = dict(zip(roots, self.pool.map(hedge.depth2_hedged_root, fields, chunksize=4) if self.pool else map(hedge.depth2_hedged_root, fields)))
                else:backed = {i: depth2_backups.backup_learner_max(output, i) for i in range(len(chunk)) if reasons[i] is None}
                for i in range(len(chunk)):
                    prices = backed.get(i)
                    if reasons[i] is None and (prices is None or any(p is None for p in prices)):reasons[i] = 'depth2-world-unavailable'
                    surfaces.append((None, reasons[i]) if reasons[i] is not None else (prices, None))
            self.timing['backup_seconds'] += time.perf_counter() - clock
        return surfaces

    def choose(self, context, rows, owners, logits, legal, inputs, raw_actions, *, probe=None):
        """Search choices for the given rows: (action, record) per row; probe=(context, rows) re-runs gate c."""
        from gpu_engine import search_hedge as hedge
        from gpu_engine.device_public_teacher_batch import Request, _numpy_worlds
        clock = time.perf_counter()
        roots = self.worlds.detach_roots(context, self.source, rows=tuple(rows), owners=tuple(owners), enable_device_worlds=True)
        self.timing['detach_seconds'] += time.perf_counter() - clock
        per_block = [];plans_by_block = []
        for recipe in self.recipes:
            clock = time.perf_counter()
            plans = [self.worlds.plan_starts(root, self.source, recipe) for root in roots]
            self.timing['plan_seconds'] += time.perf_counter() - clock
            requests = [Request(p, tuple(map(float, logits[i])), tuple(map(bool, legal[i])),
                {**{k: np.asarray(v[i])[None, None] for k, v in inputs.items()}, 'history_mask': np.ones((1, 1), bool)}, 0, 0)
                for i, p in enumerate(plans)]
            plans_by_block.append(requests)
            per_block.append(self._price(requests, recipe))
        clock = time.perf_counter();out = []
        for i, row in enumerate(rows):
            surfaces = [block[i] for block in per_block]
            refusal = next((reason for prices, reason in surfaces if prices is None), None)
            record = dict(searched=refusal is None, refusal=refusal, strategy=None, value=None)
            action = int(raw_actions[i])
            if refusal is None:
                prices = tuple(w for p, _ in surfaces for w in p);root_legal = tuple(map(bool, legal[i]))
                if self.reduction == 'raw':choice = hedge.Choice(tuple(1. if a == action else 0. for a in range(9)), 0.)
                elif self.reduction == 'average':choice = hedge.averaging(prices, root_legal)
                elif self.reduction == 'hedge':choice = hedge.bayes_hedge(prices, root_legal, lam=0.)
                elif self.reduction == 'blend':choice = hedge.bayes_hedge(prices, root_legal, lam=self.blend)
                else:raise ValueError('search reduction required')
                if choice is None:record.update(searched=False, refusal='reduction-unknown-cell')
                else:
                    action = hedge.sample(choice.strategy, hedge.search_uniform(self.seed, roots[i].address))
                    if not legal[i][action]:raise ValueError('search chose an illegal action')
                    record.update(strategy=[round(p, 6) for p in choice.strategy], value=choice.value)
                    # Search minus raw in the evaluator's own terms: completed-Q of the played action minus the raw
                    # sampled action's, and minus the raw argmax's (0 when the choice equals them).
                    q = hedge.averaging(prices, root_legal).q
                    argmax = max((a for a in range(9) if legal[i][a]), key=lambda a: (float(logits[i][a]), -a))
                    record.update(q_gain=q[action] - q[int(raw_actions[i])], q_gain_vs_argmax=q[action] - q[argmax])
            out.append((action, record))
        self.timing['reduce_seconds'] += time.perf_counter() - clock
        if probe is not None:
            self._never_borrow(probe, rows, owners, roots, plans_by_block, out, logits, legal, inputs, raw_actions)
        return out

    def _never_borrow(self, probe, rows, owners, roots, plans_by_block, out, logits, legal, inputs, raw_actions):
        from gpu_engine.device_public_teacher_batch import _numpy_worlds
        stats = probe['stats'];context = probe['context']
        clock = time.perf_counter()
        other = self.worlds.detach_roots(context, self.source, rows=tuple(rows), owners=tuple(owners), enable_device_worlds=True)
        for i, (a, b) in enumerate(zip(roots, other)):
            if not root_equal(a, b):stats['root_differs'] += 1
        for recipe, requests in zip(self.recipes, plans_by_block):
            cf = [self.worlds.plan_starts(root, self.source, recipe) for root in other]
            for req, p in zip(requests, cf):
                if [(s.world, s.seed, s.opponent_fixtures, s.revealed_source_indices, s.refusals) for s in req.plans.starts] != \
                        [(s.world, s.seed, s.opponent_fixtures, s.revealed_source_indices, s.refusals) for s in p.starts]:
                    stats['plan_differs'] += 1
            cf_requests = [replace(req, plans=p) for req, p in zip(requests, cf)]
            ours = _numpy_worlds(requests, self.materializer, recipe);theirs = _numpy_worlds(cf_requests, self.materializer, recipe)
            for pa, pb in zip(ours, theirs):
                for wa, wb in zip(pa, pb):
                    stats['worlds_compared'] += 1
                    if not world_equal(wa, wb):stats['world_differs'] += 1
        # The full choice from the counterfactual context must be identical.
        saved = self.timing.copy();calls = len(self.calls)
        again = self.choose(context, rows, owners, logits, legal, inputs, raw_actions)
        self.timing = saved;del self.calls[calls:]
        for (x, rx), (y, ry) in zip(out, again):
            stats['decisions_compared'] += 1
            if x != y or rx != ry:stats['choice_differs'] += 1
        stats['probe_seconds'] += time.perf_counter() - clock


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', required=True);parser.add_argument('--iteration', type=int, default=1)
    parser.add_argument('--block', type=int, default=0, help='declared block index (2,048 games), played twice')
    parser.add_argument('--games', type=int, help='prefix of the block (gates and smoke only)')
    parser.add_argument('--a-run', required=True);parser.add_argument('--a-iteration', type=int, required=True)
    parser.add_argument('--b-run');parser.add_argument('--b-iteration', type=int)
    parser.add_argument('--pricer', choices=('depth1', 'depth2'), default='depth1')
    parser.add_argument('--reduction', choices=REDUCTIONS)
    parser.add_argument('--blend', type=float, default=.5, help='reply-prior weight lambda for --reduction blend')
    parser.add_argument('--world-blocks', type=int, default=4, help='W = 2 * blocks')
    parser.add_argument('--cells', help='JSON list of cells played in order by this process (label, b_run, b_iteration, '
        'reduction, and optionally blend, world_blocks, never_borrow); replaces the single-cell options')
    parser.add_argument('--search-seed', type=int, default=2026110500)
    parser.add_argument('--capacities', default='64,512', help='fixed pricer root capacities (padded)')
    parser.add_argument('--depth2-child-block', type=int, default=27)
    parser.add_argument('--depth2-fixed-capacity', action='store_true', help="the Depth2Pricer's fixed-capacity compile (inactive padding)")
    parser.add_argument('--workers', type=int, default=16, help='processes for depth-2 hedged turn-2 solves')
    parser.add_argument('--never-borrow', action='store_true', help='gate c: re-run every searched decision from a counterfactual true state')
    parser.add_argument('--max-steps', type=int, help='smoke only')
    parser.add_argument('--collector-check', type=int, default=0, help='compare K raw steps with the collector advance, bitwise')
    parser.add_argument('--out', required=True);parser.add_argument('--label')
    args = parser.parse_args(argv)
    if args.cells:cells = json.loads(Path(args.cells).read_text())
    else:
        if None in (args.b_run, args.b_iteration, args.reduction, args.label):raise SystemExit('--b-run, --b-iteration, --reduction and --label required')
        cells = [dict(label=args.label, b_run=args.b_run, b_iteration=args.b_iteration, reduction=args.reduction, blend=args.blend,
            world_blocks=args.world_blocks, never_borrow=args.never_borrow)]
    for cell in cells:
        if set(cell) - {'label', 'b_run', 'b_iteration', 'reduction', 'blend', 'world_blocks', 'never_borrow'} or cell['reduction'] not in REDUCTIONS:
            raise SystemExit('malformed cell: ' + json.dumps(cell))
    if len({c['label'] for c in cells}) != len(cells):raise SystemExit('cell labels must be unique')
    import jax
    import jax.numpy as j
    from pokezero import v5_gumbel as original
    from gpu_engine import v5_device_rollout as rollout, v5_model
    from gpu_engine.v5_device_backend import RuntimeBackend
    from gpu_engine.v2_fused import prepare
    begin = time.perf_counter();out = Path(args.out);out.mkdir(parents=True, exist_ok=True)
    context = prepare(argparse.Namespace(config=args.config, arm='B', iterations=2 if args.iteration else 1));plan = context['plan']
    loaded = {}
    def actor(run, iteration):
        if (run, iteration) not in loaded:
            snap = snapshot(run, iteration)
            params, config, _ = v5_model.load(Path(run, snap.archive), expected_archive_sha256=snap.archive_sha256,
                expected_checkpoint_sha256=snap.checkpoint_sha256, expected_category_vocabulary_sha256=plan.model.category_vocabulary_sha256,
                expected_label_vocabulary_sha256=plan.model.label_vocabulary.sha256, enable_v5=True)
            if replace(config, replicate=plan.model.replicate) != plan.model:raise ValueError('actor architecture differs from evaluation model')
            loaded[(run, iteration)] = (snap, params)
        return loaded[(run, iteration)]
    a, a_params = actor(args.a_run, args.a_iteration)
    backend = RuntimeBackend(context['prepared'], context['pool'], gender_policy=context['gender']);backend.check_plan(plan)
    batch = plan.iterations[args.iteration][args.block]
    if args.games:   # a prefix is a valid smaller batch; its initial states need not equal the full block's
        batch = replace(batch, games=batch.games[:args.games], gender_root_ids=batch.gender_root_ids[:args.games])
    swapped = replace(batch, games=tuple(replace(g, stream_id=g.stream_id - SWAP_OFFSET, battle_id=g.battle_id + '-swap') for g in batch.games))
    def fresh():
        carries = [jax.block_until_ready(backend._construct(declared)) for declared in (batch, swapped)]
        return jax.tree.map(lambda x, y: j.concatenate([x, y]), *carries)
    n = len(batch.games);a_side_host = np.repeat(np.asarray([0, 1], np.int32), n);a_side = j.asarray(a_side_host)
    pairs = [(g.stream_id, g.stream_id - SWAP_OFFSET) for g in batch.games]
    prepared = context['prepared']
    decide, apply = make_step(prepared.engine_tables, prepared.observation_tables, plan.model, sampling_seed=batch.sampling_seed,
        max_requests=plan.max_requests)
    collector_check = None
    if args.collector_check:
        # The raw path of this split step against the collector's own advance, from the same carry.
        b, b_params = actor(cells[0]['b_run'], cells[0]['b_iteration'])
        if a.archive_sha256 == b.archive_sha256:raise ValueError('the collector check needs two distinct archives')
        carry = fresh();params = (a_params, b_params)
        bindings = rollout.Bindings(plan.model.category_vocabulary_sha256, plan.model.label_vocabulary, prepared.source_sha256,
            (a.archive_sha256,) * 2, (True, True), backend.declaration_sha256)
        assignments = [dict(stream_id=int(s), learner_side=int(side), opponent=dict(archive_sha256=b.archive_sha256))
            for s, side in zip(np.asarray(carry.streams), a_side_host)]
        bank = rollout.actor_bank_from_assignments(bindings, (a.archive_sha256, b.archive_sha256), current_archive=a.archive_sha256,
            assignments=assignments)
        collect = rollout.make_collector(prepared.engine_tables, prepared.observation_tables, plan.model, backend.labels, bindings,
            sampling_seed=batch.sampling_seed, chunk_steps=args.collector_check, max_requests=plan.max_requests, actor_bank=bank)
        reference = jax.device_get(collect(carry, params).carry);ours = carry
        for _ in range(args.collector_check):ours = apply(ours, decide(ours, params, a_side), decide(ours, params, a_side)['actions'])
        ours = jax.device_get(ours)
        fields = {name: leaves_equal(getattr(reference, name), getattr(ours, name)) for name in reference._fields}
        collector_check = dict(steps=args.collector_check, all_equal=all(fields.values()), differing=[k for k, v in fields.items() if not v])
        log('COLLECTOR_CHECK', **collector_check)
    search = None
    setup = dict(args=args, plan=plan, batch=batch, n=n, pairs=pairs, a=a, a_side_host=a_side_host, a_side=a_side,
        decide=decide, apply=apply, collector_check=collector_check)
    for cell in cells:
        b, b_params = actor(cell['b_run'], cell['b_iteration'])
        cell = dict(dict(blend=args.blend, world_blocks=args.world_blocks, never_borrow=False), **cell)
        if cell['reduction'] != 'none':
            if search is None:
                search = Search(prepared, plan, a_params, pricer=args.pricer, reduction=cell['reduction'], blend=cell['blend'],
                    world_blocks=cell['world_blocks'], seed=args.search_seed, capacities=tuple(int(x) for x in args.capacities.split(',')),
                    depth2_options=dict(child_block=args.depth2_child_block, **(dict(fixed_capacity=True) if args.depth2_fixed_capacity else {})),
                    workers=args.workers)
            else:search.configure(reduction=cell['reduction'], blend=cell['blend'], world_blocks=cell['world_blocks'])
        play_cell(setup, cell, fresh(), (a_params, b_params), b, search if cell['reduction'] != 'none' else None, out,
            setup_seconds=time.perf_counter() - begin)
    return 0


def play_cell(setup, cell, carry, params, b, search, out, *, setup_seconds):
    import jax
    import jax.numpy as j
    from pokezero import v5_gumbel as original
    from gpu_engine import device_public_worlds as worlds
    from gpu_engine.v5_device_rollout import RootContext, STATUS
    args = setup['args'];n = setup['n'];a_side_host = setup['a_side_host'];a_side = setup['a_side'];pairs = setup['pairs']
    decide, apply = setup['decide'], setup['apply'];a = setup['a'];label = cell['label'];begin = time.perf_counter()
    log('SEARCH_HEAD_TO_HEAD_START', label=label, games=2 * n, reduction=cell['reduction'], pricer=args.pricer, b_run=cell['b_run'],
        worlds=2 * cell['world_blocks'] if search else 0, max_requests=setup['plan'].max_requests, setup_seconds=setup_seconds)
    decisions = collections.defaultdict(list);probe_stats = collections.Counter();step = 0;timing = collections.Counter()
    perm = np.roll(np.arange(2 * n), 1)
    with patch.object(original.Recipe, 'selected', lambda self, address: True):
        while True:
            clock = time.perf_counter()
            decided = decide(carry, params, a_side)
            host = jax.device_get({k: decided[k] for k in ('live', 'requested', 'legal', 'raw_logits', 'actions', 'input_valid', 'turn')})
            timing['decide_seconds'] += time.perf_counter() - clock
            if not host['live'].any() or (args.max_steps is not None and step >= args.max_steps):break
            frames = np.asarray(carry.frames)
            rows = np.flatnonzero(host['live'] & host['requested'][np.arange(2 * n), a_side_host] & host['input_valid'][np.arange(2 * n), a_side_host])
            actions = host['actions'].copy()
            if len(rows):
                raw = host['actions'][rows, a_side_host[rows]]
                logits = host['raw_logits'][rows, a_side_host[rows]];legal = host['legal'][rows, a_side_host[rows]]
                chosen = raw.copy();records = [dict(searched=False, refusal='no-search', strategy=None, value=None)] * len(rows)
                if search is not None:
                    clock = time.perf_counter()
                    inputs = jax.device_get({k: v[j.asarray(rows)] for k, v in decided['own_inputs'].items()})
                    root_context = RootContext(carry.state, carry.public, carry.streams, carry.frames)
                    probe = None
                    if cell['never_borrow']:
                        state = counterfactual(carry.state, rows, a_side_host[rows], perm, worlds.OWN_MON_FIELDS + worlds.OWN_SIDE_FIELDS)
                        probe = dict(context=RootContext(state, carry.public, carry.streams, carry.frames), stats=probe_stats)
                    result = search.choose(root_context, [int(r) for r in rows], [int(a_side_host[r]) for r in rows], logits, legal,
                        inputs, raw, probe=probe)
                    chosen = np.asarray([x for x, _ in result], np.int32);records = [r for _, r in result]
                    timing['search_seconds'] += time.perf_counter() - clock
                actions[rows, a_side_host[rows]] = chosen
                masked = np.where(legal, logits, -np.inf)
                for i, r in enumerate(rows):
                    decisions[int(r)].append(dict(frame=int(frames[r]), turn=int(host['turn'][r]), raw=int(raw[i]), argmax=int(np.argmax(masked[i])),
                        chosen=int(chosen[i]), **records[i]))
            clock = time.perf_counter()
            carry = apply(carry, decided, j.asarray(actions))
            timing['apply_seconds'] += time.perf_counter() - clock
            step += 1
            if step % 25 == 0:
                log('STEP', label=label, step=step, live=int(host['live'].sum()), searched_rows=len(rows), probe=dict(probe_stats),
                    search_timing={k: round(v, 1) for k, v in (search.timing if search else {}).items()},
                    elapsed=round(time.perf_counter() - begin, 1))
    final = jax.device_get(dict(streams=carry.streams, status=carry.status, winner=carry.winner, terminal_frame=carry.terminal_frame))
    index = {int(s): i for i, s in enumerate(final['streams'])}
    def score(stream, seat):
        i = index[stream];status = STATUS[int(final['status'][i])]
        if status != 'COMPLETE':return None
        winner = int(final['winner'][i]);return 0.5 if winner == 2 else 1.0 if winner == seat else 0.0
    pair_scores = [];excluded = 0;seat = {0: [], 1: []}
    for first, second in pairs:
        x, y = score(first, 0), score(second, 1)
        if x is not None:seat[0].append(x)
        if y is not None:seat[1].append(y)
        if x is None or y is None:excluded += 1;pair_scores.append(None);continue
        pair_scores.append((x + y) / 2)
    values = np.asarray([p for p in pair_scores if p is not None], np.float64);count = len(values)
    mean = float(values.mean()) if count else None;se = float(values.std(ddof=1) / math.sqrt(count)) if count > 1 else None
    bands = collections.defaultdict(collections.Counter)
    for row, items in decisions.items():
        for d in items:
            c = bands[band(d['turn'])];c['decisions'] += 1;c['searched'] += d['searched']
            c['changed_vs_raw_sample'] += d['chosen'] != d['raw'];c['changed_vs_raw_argmax'] += d['chosen'] != d['argmax']
            c['searched_changed_vs_raw_sample'] += d['searched'] and d['chosen'] != d['raw']
            c['searched_changed_vs_raw_argmax'] += d['searched'] and d['chosen'] != d['argmax']
            if d['refusal'] and d['refusal'] != 'no-search':c['refusal:' + str(d['refusal'])[:80]] += 1
            if d.get('q_gain') is not None:c['q_gain_sum'] += d['q_gain'];c['q_gain_vs_argmax_sum'] += d['q_gain_vs_argmax']
    total = collections.Counter()
    for c in bands.values():total.update(c)
    for c in list(bands.values()) + [total]:
        for key in ('q_gain', 'q_gain_vs_argmax'):
            c[key + '_mean'] = c[key + '_sum'] / c['searched'] if c['searched'] else None
    interval = lambda shift, scale: None if se is None else [scale * (mean - 1.645 * se - shift), scale * (mean + 1.645 * se - shift)]
    report = dict(schema='v2-search-head-to-head.v1', label=label,
        a=dict(run=args.a_run, iteration=a.iteration, archive_sha256=a.archive_sha256),
        b=dict(run=cell['b_run'], iteration=b.iteration, archive_sha256=b.archive_sha256),
        search=dict(reduction=cell['reduction'], pricer=args.pricer, blend=cell['blend'] if cell['reduction'] == 'blend' else None,
            worlds=2 * cell['world_blocks'] if search else 0, search_seed=args.search_seed,
            block_seeds=list(world_block_seeds(args.search_seed, cell['world_blocks'])) if search else [],
            capacities=args.capacities, never_borrow=cell['never_borrow']),
        games=dict(config=args.config, iteration=args.iteration, block=args.block, games_per_seat_order=n,
            sampling_seed=setup['batch'].sampling_seed),
        pairs=count, excluded_pairs=excluded, a_paired_score=mean, standard_error=se, interval_90=interval(0., 1.),
        a_score_minus_50_points=None if mean is None else 100 * (mean - .5), a_score_minus_50_interval_90_points=interval(.5, 100.),
        a_score_by_a_seat={str(k): float(np.mean(v)) if v else None for k, v in seat.items()},
        dispositions={name: int((np.asarray(final['status']) == code).sum()) for code, name in enumerate(STATUS)},
        decisions=dict(total=dict(total), by_turn_band={k: dict(v) for k, v in sorted(bands.items())},
            searched_share=total['searched'] / max(total['decisions'], 1),
            searched_change_rate_vs_raw_sample=total['searched_changed_vs_raw_sample'] / max(total['searched'], 1)),
        never_borrow=dict(probe_stats) if cell['never_borrow'] else None, collector_check=setup['collector_check'],
        timing=dict(steps=step, wall_seconds=time.perf_counter() - begin, process_setup_seconds=setup_seconds, **timing,
            search=dict(search.timing) if search else {}, pricer_calls=len(search.calls) if search else 0,
            search_events=dict(search.events) if search else {},
            search_ms_per_decision=1000 * timing['search_seconds'] / max(total['decisions'], 1) if search else None),
        scope='Paired engine head-to-head, same teams with seats swapped, search in seat A; not a Foul Play result.')
    (out / f'{label}.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    with (out / f'{label}-pairs.jsonl').open('w') as stream:
        for (first, second), value in zip(pairs, pair_scores):stream.write(json.dumps(dict(stream=first, swapped=second, score=value)) + '\n')
    with (out / f'{label}-decisions.jsonl').open('w') as stream:
        for row in sorted(decisions):stream.write(json.dumps(dict(stream=int(final['streams'][row]), a_side=int(a_side_host[row]), decisions=decisions[row])) + '\n')
    np.savez(out / f'{label}-games.npz', **final)
    log('SEARCH_HEAD_TO_HEAD', **{k: v for k, v in report.items() if k in ('label', 'pairs', 'excluded_pairs', 'a_paired_score',
        'a_score_minus_50_points', 'a_score_minus_50_interval_90_points', 'dispositions', 'never_borrow')},
        searched_share=report['decisions']['searched_share'], wall_seconds=report['timing']['wall_seconds'])


if __name__ == '__main__':
    raise SystemExit(main())
