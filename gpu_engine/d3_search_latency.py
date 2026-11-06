"""End-to-end wall time per search call at small capacity: which search configs can play at 1,000 ms per move (800 ms rule).

One CALL is one search decision for C roots (C games' moves batched together; --capacities, default 1,4,16), from the D3 bank
chunk's raw device roots and host frames to each root's move. Roots: the chunk's first C labeled requests, cycled when it has
fewer (d3_depthn_throughput.prepare: the gate's --d3-root --chunk builder, head override and evaluator header, imported), the
same C every call. Configs (--configs NAME@W/W; default depth2-fw and depth2-merged at W=2 and 8, depth3-prior-k3 and
depth3-searched-k3 at W=2), one pricer instance per (config, pricer capacity):
- d3_depthn_throughput's names are DepthNPricer(fixed_capacity=True, digest=False) built by its Caller; searched = the full-width
  depth-2 pass, searched_maps on the host, depth N at the maps' widths; depth2-fw is DepthNPricer at N=2, k=9.
- depth2-merged is depth-2 full width on the MERGED Depth2Pricer (device_depth2_pricer, d5150eac or later) built as
  d3_depth2_gate.make_pricer builds it for the gate and --pricer-only (digest=True, k_own=k_opp=9, the pricer's own child_block
  default as no --child-block leaves it) plus fixed_capacity=True: digest=True adds the leaf digests as kernel outputs (computed
  and transferred, unused here) and keeps the gate's compiled program (DIGEST_NOTE), and with one actor and a power-of-two C
  fixed_capacity yields the bucketed gate call's kernel key, so the compile cache is shared. Its one fused program (D3's
  turn-1 stage and the blocked turn 2) compiles per pricer capacity in its own kernel table, apart from DepthNPricer's.
A call's stages are CONTIGUOUS time.perf_counter() laps (every instant between its start and end is charged to exactly
one stage, so the stages sum to the call's wall time; seconds a pricer measures inside its own lap are moved, not added):
- host_prep: Provider._prepare_batched's per-root core for exactly these C roots (host_prep below): per scan step the device
  slice of the retained RootContext and the host copy of its stream/frame ids, ONE worlds.detach_roots pass over the step's
  (row, owner) pairs (part detach), the address check, the block-0 worlds.plan_starts under the declared recipe (part
  plan_starts), the root encoder inputs, prior and actors and the Request (part assemble). NOT timed: the record binding hashes
  (records, not requests) and the head override's root forward (gate.bind_roots at load; with --head-archive each Request takes
  that bound root prior and actors 0/0, as the gate's do). The first call is checked against the bank's own Requests.
- world_construction: worlds.plan_starts for blocks 1..K-1 (U1's world_block_seeds, quarter selection bypassed, as the gate)
  plus every price() call's own materialization, its timing['world_prepare_seconds'] (validate_requests +
  device_public_teacher_batch._numpy_worlds -> device_public_world_state.materialize_host, host numpy). Searched materializes
  twice (the pass and depth N), both counted. The header names the path resolved at run time and the source commit.
- pricer: each price() wall to a host-ready frame minus its materialization and compile: pack, P1, the ply loop, every ply's
  device_get and the host frame; searched adds as_depth2_frame + searched_maps on the host (parts depth2_pass,
  searched_maps_host, depthn). The pricer's own device / P1 / ply-loop / host-pack split is kept per call.
- solve: per root the backup and the root reduction, from per-(root, block) units of host slices (part slice: numpy views;
  part worlds: world_prices on every unit in process, or worlds_pool: pool.map over a persistent spawn pool, pickling and IPC
  included; part root_reduce). --solver takes a list (learner_max,hedge,MODULE:FUNC) and --solver-pool a list of pool sizes
  (0 = in process): every (solver, pool size) solves the SAME priced frames of each call, in an order rotated per call, and
  has its own end-to-end total = the shared host_prep + world_construction + pricer + its solve + its return.
  learner_max: depth2_backups.backup_learner_max on as_depth2_frame (depth 2, the reference; part as_depth2_frame) or on the
  merged Depth2Frame, depthn_backups.backup_learner_max (depth >= 3); then v5_gumbel.reduce_completed_q per block of two
  worlds; W = 2K composes the K blocks by the equal-weight mean of their completed-Q (U1's composition: every world weighted
  1/W) and re-derives the improved target and root value; any refused block refuses the root.
  hedge (gpu_engine.search_hedge of PR #1829, 583555d2, imported lazily; not on this branch): per node
  search_hedge.matrix_game(the full [own x opp] child grid)[0], priors ignored, through depthn_backups.backup on DepthNPricer
  frames and search_hedge.depth2_hedged_root on the merged Depth2Frame; at the root search_hedge.bayes_hedge(all W worlds,
  lam=0, weights 1/W, no iterations) INSTEAD of reduce_completed_q (root logits unused). The header records the module's
  file sha256 and --hedge-commit.
  MODULE:FUNC: depthn_backups.backup(frame, root, rule) at every depth, rule(grid, pi_own, pi_opp) -> value at every node
  (gpu_engine.depthn_backups:learner_max is its like-for-like baseline), then reduce_completed_q as learner_max; the merged
  Depth2Frame has no per-node hook and solves with depth2_backups learner-max (recorded per row).
- return: each root's move (learner_max / MODULE:FUNC: argmax of the improved target, lowest action on ties; hedge:
  search_hedge.sample(strategy, search_uniform(2026110500, address))) and a block_until_ready on any device array still held by
  the frames (DepthNPricer.price device_gets every field, so the transfer is inside pricer: expected 0, counted).
- compile: lower + compile seconds of programs the pricers compiled during the call (their `compiles`), moved out of pricer and
  EXCLUDED from the end-to-end total = host_prep + world_construction + pricer + solve + return.
No stage leaves device work queued: a device fence (a tiny op on the default device, block_until_ready) ends host_prep, every
price() and every return. A solve pool (--solver-pool N > 0) is N spawn workers with 1 BLAS/OpenMP thread each, created before
any timed call and warmed (solve modules imported, a 2x2 hedge LP solved); its startup seconds are reported in the header,
never inside a call. W = 2K worlds per root in K recipe blocks: `sequential` = K price passes of C roots inside one call (the
per-move latency of one game); `packed` (W > 2, unless --no-packed) = ONE pass at pricer capacity K*C holding K rows per
root (row b*C + i for root i). packed is a TIMING PROXY: price() takes one recipe and materialization refuses a start planned
under another recipe (device_public_world_state._materialize checks each start's seed against it), so it plans blocks
1..K-1 as sequential does (their cost counted) but prices K COPIES of block 0's requests under block 0's recipe; the device
work has the K-block shapes, the values are block 0's K times, not U1's composition.
Per (config, capacity, W, mode): 1 first (cold) call on move window 0, then --repeats warm calls, warm call i on its own
window of C roots (the chunk's labeled roots i*C .. i*C+C-1, cycled), so the statistics span N distinct moves, not one move
re-timed. The first call of a (config, capacity) is its cold call: its compile seconds are reported apart, and it also pays
untracked first-use compiles (detach_roots' program, eager readout ops, the fence) inside its stages; no first call enters
any statistic. Over the warm calls: median, p90 (numpy's linear interpolation) and max of every stage, part and the total;
a warm call that still compiled (e.g. a new pool size's gather/write) is flagged and logged (WARM_COMPILE). Per call the
roots' window, its distinct scan steps (one RootContext slice and detach pass each), expanded nodes, enumerated leaves and
each variant's refused roots are recorded, with their spread over the warm calls. Every call's Requests are checked against
the bank's own. Leaves per
world: nominal = 81 x the product of the slot widths at plies 2..N (the engine's own widths), measured = enumerated leaves /
(C x W). Memory: d3_depth2_throughput.memory_stats after each call, baseline before a (config, capacity)'s first call.
Compiles: every DepthNPricer in one process shares one program table (DepthNPricer._program keys: P1 by capacity, block
programs by (B, G, K), gather/write by pool rows, select by shape), valid because each has the same executor, params, forward
(the v5 head in bf16), digest=False and k1 widths: P1 compiles once per pricer capacity and each block program once per
process; a config's pricer at one capacity serves every W and mode that uses it (a packed K*C pricer is the sequential pricer
of capacity K*C), so a first call compiles only what nothing before it did (--isolate: one process and table per config). --compile-cache DIR enables JAX's persistent cache
(owning_worker_runtime.configure_compilation_cache) right after the bank's JAX import, before its first compile.
The rule: a (config, W, mode) fits at capacity C when the p90 of its warm end-to-end totals is <= 800 ms, leaving 200 ms of
slack under the 1,000 ms per-move budget. Reported: per config, W and mode the median and p90 per capacity and the largest
fitting C; per config the largest fitting C (with its W) and the largest fitting W (with its C), over sequential rows (packed
apart, as a proxy). Progress is JSON lines; latency.json (evaluator first, then the header) goes to a new --out directory and
the markdown summary is printed at the end of the log.
"""
import argparse
from contextlib import ExitStack, contextmanager, nullcontext
import dataclasses
from dataclasses import replace
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import time
from unittest.mock import patch

import numpy as np

from . import d3_depth2_gate as gate
from . import d3_depth2_throughput as t2
from . import d3_depthn_throughput as tn
from .d3_depthn_gate import split_searched

SCHEMA = 'pokezero.search-latency.v1'
BUDGET_MS = 1000.
LIMIT_MS = 800.
RULE = ('a (config, W, mode, solver, pool) fits at capacity C when the p90 over its N distinct move windows (warm call i prices the '
    'chunk\'s roots i*C .. i*C+C-1, cycled; N = --repeats) of the end-to-end totals is <= 800 ms (200 ms of slack under the 1,000 ms '
    'per-move budget); the median is shown beside it; the cold call (window 0) is excluded')
STAGES = ('host_prep', 'world_construction', 'pricer', 'solve', 'return')
COMPILE = 'compile'
LANES = 81
MODES = ('sequential', 'packed')
MERGED = 'depth2-merged'   # depth-2 full width on the merged Depth2Pricer (not a d3_depthn_throughput name)
MERGED_SOLVE = 'depth2_backups.backup_learner_max on the merged Depth2Frame (no per-node hook: a MODULE:FUNC --solver does not apply)'
# The hedge solver (gpu_engine/search_hedge.py of PR #1829, not on this branch: shipped next to the code, imported lazily).
HEDGE_MODULE = 'gpu_engine.search_hedge'
HEDGE_COMMIT = '583555d2'
HEDGE_REFERENCE_SHA256 = 'bf08d6312f20055dd62412f28054eb012c26a61643ed8a997d51c60ff64f33e6'   # sha256 of search_hedge.py at 583555d2
HEDGE_SEED = 2026110500   # search_uniform's seed for the move sample (pre-registered)
HEDGE_LAM = 0.            # bayes_hedge's blend weight: 0 = the zero-sum hedge (pre-registered; worlds weighted 1/W, no iterations)
HEDGE_SOLVE = ('per node search_hedge.matrix_game(full [own x opp] child grid)[0], priors ignored: depthn_backups.backup on DepthNPricer frames, '
    'search_hedge.depth2_hedged_root on the merged Depth2Frame; root search_hedge.bayes_hedge(all W worlds, lam=0) instead of reduce_completed_q '
    '(root logits unused); move search_hedge.sample(strategy, search_uniform(2026110500, address))')
# Per-root host slices the solve units carry (what depthn_backups / depth2_backups / search_hedge read).
DEPTHN_ROOT_FIELDS = ('pre', 'root_peer_requested', 'root_peer_logits', 't1_status', 't1_value', 't1_own_actions', 't1_opp_actions', 't1_own_logits',
    't1_peer_logits', 'leaf_status', 'leaf_value')
DEPTHN_NODE_FIELDS = ('status', 'value', 'own_actions', 'opp_actions', 'own_logits', 'opp_logits')
DEPTH2_ROOT_FIELDS = ('pre', 'root_peer_requested', 'root_peer_logits', 't1_status', 't1_value', 't1_own_logits', 't1_peer_logits', 't2_status', 't2_value')
DEFAULT_CONFIGS = 'depth2-fw@2/8,depth2-merged@2/8,depth3-prior-k3@2,depth3-searched-k3@2'
CODE_ROOT = Path(__file__).resolve().parent.parent
# Phase D constructor files (someone else's work): imported only if the timed path uses them, never edited here.
PHASE_D_MODULES = ('gpu_engine.phase_d_live_worlds', 'gpu_engine.phase_d_protocol_world', 'gpu_engine.v5_public_world', 'gpu_engine.public_sleep')
CONSTRUCTOR_NOTE = ("Codex's constructor batch 2 (own PP carry, removed items, sleep clocks) is not in the integration branch. If the "
    'head-to-head or live play uses a newer constructor, this column may understate or overstate construction cost.')
PACKED_NOTE = ('TIMING PROXY: one pass at K*C prices K copies of block 0\'s requests under block 0\'s recipe (price() takes one recipe and '
    'materialization refuses a start planned under another); blocks 1..K-1 are still planned and timed as in sequential; same device shapes as K '
    'blocks, values are block 0\'s K times, not U1\'s W composition')
STAGE_SCOPE = {
    'host_prep': 'Provider._prepare_batched\'s per-root core for exactly C roots: per-step RootContext device slice + stream/frame host copy, one '
        'worlds.detach_roots pass per step (part detach), address check, block-0 worlds.plan_starts (part plan_starts), encoder inputs, prior, actors '
        'and the Request (part assemble); not timed: record binding hashes and the head override\'s root forward (gate.bind_roots at load)',
    'world_construction': 'worlds.plan_starts for blocks 1..K-1 (part plan_starts) + every price() call\'s world_prepare_seconds (validate_requests + '
        '_numpy_worlds -> materialize_host; part materialize); searched materializes twice (pass and depth N)',
    'pricer': 'price() wall to a host frame minus its materialization and compile (pack, P1, ply loop, per-ply device_get, host frame); searched adds '
        'as_depth2_frame + searched_maps on the host; parts depth2_pass, searched_maps_host, depthn',
    'solve': 'per variant (solver, pool size) on the shared frames: per-(root, block) host slices (part slice), world_prices in process (part worlds) '
        'or pool.map with pickling and IPC (part worlds_pool), the root reduction (part root_reduce). learner_max: depth2_backups (depth 2: on '
        'as_depth2_frame, part as_depth2_frame) / depthn_backups + reduce_completed_q per block, W > 2 the equal-weight mean of the blocks\' completed-Q; '
        'hedge: matrix_game per node + bayes_hedge(all W worlds, lam=0); MODULE:FUNC: depthn_backups.backup with the rule + reduce_completed_q',
    'return': 'per variant: each root\'s move (argmax of the improved target; hedge: sample(strategy, search_uniform(2026110500, address))) + '
        'block_until_ready on any device array left in the frames (expected none) + fence',
    'compile': 'lower + compile seconds of programs compiled during the call (the pricers\' compiles); excluded from the total',
    'total': 'per variant: host_prep + world_construction + pricer (shared by the call\'s variants) + its solve + its return; compile excluded'}


def ints(text):
    return tn.ints(text)


def check_config(name):
    """A config name this tool times -> its parse: depth2-merged, or d3_depthn_throughput's depth2-fw, depth2-k{k},
    depth{N}-prior-k{k}, depth{N}-searched-k{k} (N >= 3)."""
    if name == MERGED:return dict(kind='merged', depth=2, mode='prior', k=9)
    try:config = tn.parse_config(name)
    except (ValueError, IndexError):raise ValueError(f'unknown config {name!r}') from None
    if config['kind'] == 'depth1':raise ValueError('depth1 has no search to solve (d3_depthn_throughput times it)')
    if config['mode'] not in tn.MODES or config['depth'] < 2 or not 1 <= config['k'] <= 9:raise ValueError(f'unknown config {name!r}')
    if config['kind'] == 'searched' and config['depth'] < 3:raise ValueError(f'{name}: searched needs depth >= 3 (its pass is depth 2)')
    return config


def parse_configs(text, default_worlds):
    """'NAME[@W/W...],...' -> {NAME: sorted W tuple}; an entry without @ takes default_worlds."""
    out = {}
    for item in str(text).split(','):
        name, at, worlds = item.strip().partition('@')
        if not name or name in out:raise ValueError(f'--configs: empty or repeated entry {item!r}')
        check_config(name)
        values = tuple(int(w) for w in worlds.split('/')) if at else tuple(default_worlds)
        if not values or any(w < 2 or w % 2 for w in values):raise ValueError(f'--configs {name}: even W >= 2 (two worlds per recipe block)')
        out[name] = tuple(sorted(set(values)))
    return out


def parse_solver(spec):
    """'learner_max' or 'hedge' -> None; 'MODULE:FUNC' -> (module, func)."""
    if spec in ('learner_max', 'hedge'):return None
    module, sep, func = str(spec).partition(':')
    if not sep or not module or not func or ':' in func:raise ValueError(f'--solver entries are learner_max, hedge or MODULE:FUNC, got {spec!r}')
    return module, func


def parse_solvers(text):
    """'learner_max,hedge,MODULE:FUNC' -> the solver specs in order (no repeats)."""
    specs = tuple(x.strip() for x in str(text).split(','))
    for spec in specs:parse_solver(spec)
    if len(set(specs)) != len(specs) or '' in specs:raise ValueError(f'--solver: empty or repeated entry in {text!r}')
    return specs


def importable(name):
    """find_spec without importing the module itself (parent packages only, so no JAX before the memory policy)."""
    try:return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):return False


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--d3-root', required=True, help='the D3 bank run root (prespec.json, runtime.json, run/)')
    parser.add_argument('--chunk', type=int, required=True, help='bank chunk index; chunks 0..N are collected')
    parser.add_argument('--memory-policy', action='store_true', help="apply the bank runtime's JAX memory policy")
    parser.add_argument('--out', required=True, help='new output directory (never overwritten)')
    parser.add_argument('--configs', default=DEFAULT_CONFIGS, help='NAME[@W/W...] list (d3_depthn_throughput names); default %(default)s')
    parser.add_argument('--worlds', default='2', help='W for --configs entries without @ (even, comma-separated)')
    parser.add_argument('--capacities', default='1,4,16', help='roots per call C (each its own fixed_capacity pricers; 2,8 addable)')
    parser.add_argument('--repeats', type=int, default=10, help='warm calls after the first one, per (config, capacity, W, mode)')
    parser.add_argument('--solver', default='learner_max', help='solvers timed on the same priced frames, comma-separated: learner_max, hedge '
        '(search_hedge, PR #1829) or MODULE:FUNC with rule(grid, pi_own, pi_opp) -> value')
    parser.add_argument('--solver-pool', default='0', help='solve pool sizes, comma-separated: 0 = in process, N = a persistent N-process spawn pool '
        '(started before any timed call); every series is solved once per (solver, pool size) on the same frames')
    parser.add_argument('--hedge-module', default=HEDGE_MODULE, help='the search_hedge module (default %(default)s)')
    parser.add_argument('--hedge-commit', default=HEDGE_COMMIT, help='the commit search_hedge.py was shipped from, recorded in the header (default %(default)s)')
    parser.add_argument('--no-packed', dest='packed', action='store_false', help='skip the packed W > 2 mode (one pass at capacity K*C)')
    parser.add_argument('--k1-own', type=int, default=9, help='turn-1 learner width');parser.add_argument('--k1-opp', type=int, default=9, help='turn-1 opponent width')
    parser.add_argument('--block-lanes', type=int, default=2187, help='DepthNPricer block_lanes')
    parser.add_argument('--max-roots', type=int, help='use only the first N requests of the chunk')
    parser.add_argument('--compile-cache', help='absolute directory: enable JAX\'s persistent compilation cache before the first compile')
    parser.add_argument('--source-commit', help='the source tree\'s commit, recorded when the code directory has no .git')
    parser.add_argument('--isolate', action='store_true', help='measure each config in its own subprocess (each reloads the chunk and compiles its own programs)')
    parser.add_argument('--only', help=argparse.SUPPRESS)   # one config; used by --isolate
    gate.add_head_arguments(parser)   # numpy only: no JAX before the memory policy
    args = parser.parse_args(argv)
    gate.check_head_arguments(parser, args)
    try:
        args.capacity_values = tuple(sorted(set(ints(args.capacities))));args.config_worlds = parse_configs(args.configs, ints(args.worlds))
        args.solver_specs = parse_solvers(args.solver);args.pool_sizes = tuple(sorted(set(ints(args.solver_pool))))
    except ValueError as error:parser.error(str(error))
    if min(args.pool_sizes) < 0:parser.error('--solver-pool sizes are >= 0 (0 = in process)')
    if args.chunk < 0 or args.repeats < 1 or args.block_lanes < 1 or min(args.capacity_values) < 1:parser.error('--chunk >= 0 and positive --capacities/--repeats/--block-lanes')
    if any(not 1 <= k <= 9 for k in (args.k1_own, args.k1_opp)):parser.error('turn-1 widths must be 1..9')
    if args.max_roots is not None and args.max_roots < 1:parser.error('positive --max-roots required')
    if args.only is not None and args.only not in args.config_worlds:parser.error(f'--only one of {list(args.config_worlds)}')
    if args.isolate and args.only:parser.error('--isolate runs --only itself')
    if args.compile_cache is not None and not Path(args.compile_cache).is_absolute():parser.error('--compile-cache must be an absolute directory')
    if Path(args.out).exists():parser.error(f'--out {args.out} exists: this tool never overwrites an output directory')
    for spec in args.solver_specs:
        target = parse_solver(spec)
        if target is not None and not importable(target[0]):parser.error(f'--solver module {target[0]} is not importable')
    if 'hedge' in args.solver_specs:
        if not importable(args.hedge_module):parser.error(f'--solver hedge needs {args.hedge_module} (gpu_engine/search_hedge.py of PR #1829 at {args.hedge_commit}, '
            'shipped next to the code; it is not on this branch)')
        if not importable('scipy'):parser.error('--solver hedge needs scipy (search_hedge solves its LPs with scipy.optimize.linprog / HiGHS)')
    return args


# Timing ---------------------------------------------------------------------------------------------------------------------------
class Clock:
    """Contiguous laps of one call: lap(stage, part) charges the time since the previous lap to `stage` (and `stage.part`), so
    every instant between the start and the last lap belongs to exactly one stage. move() re-charges seconds a callee measured
    INSIDE a lap (a pricer's materialization and compile) from one stage to another; the sum never changes."""
    def __init__(self, timer=time.perf_counter, start=None):
        self.timer = timer;self.stages = dict.fromkeys(STAGES + (COMPILE,), 0.);self.parts = {};self.start = self.last = timer() if start is None else start

    def _add(self, stage, part, seconds):
        self.stages[stage] += seconds
        if part is not None:key = f'{stage}.{part}';self.parts[key] = self.parts.get(key, 0.) + seconds

    def lap(self, stage, part=None):
        now = self.timer();seconds = now - self.last;self.last = now;self._add(stage, part, seconds);return seconds

    def move(self, seconds, source, target, *, source_part=None, target_part=None):
        self._add(source, source_part, -seconds);self._add(target, target_part, seconds)

    def note(self, stage, part, seconds):
        """A sub-timing measured inside a stage's lap: a part only, the stage sum is unchanged."""
        key = f'{stage}.{part}';self.parts[key] = self.parts.get(key, 0.) + seconds

    def record(self):
        return dict(stages=dict(self.stages), parts=dict(self.parts), wall_seconds=self.last - self.start, total_seconds=sum(self.stages[s] for s in STAGES))


def distribution(seconds):
    """Median, p90 (numpy's default linear interpolation) and max in ms over warm-call seconds; None fields when empty."""
    values = np.asarray([s for s in seconds if s is not None], np.float64) * 1e3
    if not values.size:return dict(n=0, median_ms=None, p90_ms=None, max_ms=None)
    return dict(n=int(values.size), median_ms=float(np.median(values)), p90_ms=float(np.percentile(values, 90)), max_ms=float(values.max()))


def nominal_leaves(k_own, k_opp):
    """Nominal leaves per world: 81 turn-1 cells x the product of the slot widths k_own[p] * k_opp[p] at plies 2..N."""
    return LANES * math.prod(int(a) * int(b) for a, b in zip(k_own, k_opp))


def device_fence():
    """Block until the default device has finished every queued op: a tiny op on its stream, synced."""
    import jax
    import jax.numpy as j
    jax.block_until_ready(j.zeros((), j.float32) + 1.)


def device_arrays(objects, depth=5, skip=('world_reports', 'timing')):
    """Every jax.Array reachable through dataclass fields (but `skip`: the frames' host-built reports and timing), dicts, lists
    and tuples (bounded depth); [] when JAX is not loaded."""
    jax = sys.modules.get('jax');found = []
    if jax is None:return found
    def walk(x, level):
        if isinstance(x, jax.Array):found.append(x)
        elif level >= depth or isinstance(x, (str, bytes, np.ndarray)) or x is None:return
        elif dataclasses.is_dataclass(x) and not isinstance(x, type):
            for f in dataclasses.fields(x):
                if f.name not in skip:walk(getattr(x, f.name), level + 1)
        elif isinstance(x, dict):
            for v in x.values():walk(v, level + 1)
        elif isinstance(x, (list, tuple)):
            for v in x:walk(v, level + 1)
    for x in objects:walk(x, 0)
    return found


# Solve ----------------------------------------------------------------------------------------------------------------------------
def reduce_block(prices, root_logits, root_legal, original):
    """reduce_completed_q over one block's two worlds -> its RootImprovement, or None when it refuses (a world without a price
    surface refuses, depthn_backups._root's rule)."""
    if any(p is None for p in prices):return None
    reduction = original.reduce_completed_q(root_logits=tuple(root_logits), root_legal=tuple(root_legal), worlds=tuple(prices))
    return reduction.improvement if reduction.status == 'LABELED' else None


def compose(improvements, root_logits, root_legal, original):
    """W = 2K from K blocks: K = 1 is reduce_completed_q's own result; K > 1 the equal-weight mean of the blocks' completed-Q
    (U1's composition, every world weighted 1/W), then the improved target softmax(logits + q) over the legal actions and the
    root value sum(target * q). Any refused block refuses the root."""
    refused = dict(status='REFUSED', completed_q=None, policy_target=None, root_value=None, blocks=len(improvements))
    if not improvements or any(x is None for x in improvements):return refused
    if len(improvements) == 1:
        x = improvements[0];return dict(status='LABELED', completed_q=tuple(x.completed_q), policy_target=tuple(x.policy_target), root_value=x.root_value, blocks=1)
    legal = tuple(bool(v) for v in root_legal);K = len(improvements)
    q = tuple(math.fsum(float(x.completed_q[a]) for x in improvements) / K if legal[a] else None for a in range(9))
    target = original._probabilities(tuple(float(logit) + (q[a] or 0.) for a, logit in enumerate(root_logits)), legal)
    return dict(status='LABELED', completed_q=q, policy_target=target, root_value=math.fsum(p * v for p, v in zip(target, q) if v is not None), blocks=K)


def load_hedge(name=HEDGE_MODULE, commit=HEDGE_COMMIT):
    """The search_hedge module, imported lazily; a clear SystemExit when it is not next to the code."""
    try:return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name and (name == error.name or name.startswith(error.name + '.')):
            raise SystemExit(f'--solver hedge needs {name} (gpu_engine/search_hedge.py of PR #1829 at {commit}, shipped next to the code; '
                f'it is not on this branch): {error}') from None
        raise


def root_fields(frame, row, kind):
    """One root's host slices (numpy views; pickling copies only them): a Depth2Frame's DEPTH2_ROOT_FIELDS, or a DepthNFrame's
    DEPTHN_ROOT_FIELDS, every intermediate ply's DEPTHN_NODE_FIELDS and its depth and slot widths."""
    if kind == 'depth2':return {k: getattr(frame, k)[row] for k in DEPTH2_ROOT_FIELDS}
    return dict(depth=int(frame.depth), k_own=tuple(int(x) for x in frame.k_own), k_opp=tuple(int(x) for x in frame.k_opp),
        nodes=[{k: getattr(node, k)[row] for k in DEPTHN_NODE_FIELDS} for node in frame.nodes], **{k: getattr(frame, k)[row] for k in DEPTHN_ROOT_FIELDS})


def unit_rule(unit):
    """The per-node rule of a depthn unit: learner_max, the hedge's matrix_game value on the full child grid (priors ignored) or
    MODULE:FUNC (resolved by name, so a pool worker needs no pickled callable); an in-process unit may carry its rule."""
    if unit.get('rule') is not None:return unit['rule']
    if unit['solver'] == 'hedge':
        game = importlib.import_module(unit['hedge']).matrix_game
        return lambda grid, pi_own, pi_opp: game(grid)[0]
    from gpu_engine import depthn_backups
    if unit['solver'] == 'learner_max':return depthn_backups.learner_max
    module, func = parse_solver(unit['solver']);return getattr(importlib.import_module(module), func)


def world_prices(unit):
    """One (root, block) solve unit -> its two WorldPrices (picklable both ways: the pool's unit of work, and the in-process path).
    depth2 (a Depth2Frame's slices): hedge -> search_hedge.depth2_hedged_root, else depth2_backups.backup_learner_max. depthn (a
    DepthNFrame's slices): depthn_backups.backup(view, 0, unit_rule(unit))."""
    one = lambda x: np.asarray(x)[None];f = unit['fields']
    if unit['kind'] == 'depth2':
        if unit['solver'] == 'hedge':
            hedge = importlib.import_module(unit['hedge']);return tuple(hedge.depth2_hedged_root({k: f[k] for k in hedge.DEPTH2_FIELDS}))
        from gpu_engine import depth2_backups
        return depth2_backups.backup_learner_max(SimpleNamespace(**{k: one(v) for k, v in f.items()}), 0)
    from gpu_engine import depthn_backups
    view = SimpleNamespace(depth=f['depth'], k_own=f['k_own'], k_opp=f['k_opp'], nodes=tuple(SimpleNamespace(**{k: one(v) for k, v in n.items()}) for n in f['nodes']),
        **{k: one(f[k]) for k in DEPTHN_ROOT_FIELDS})
    return depthn_backups.backup(view, 0, unit_rule(unit))


class Solver:
    """The per-root solve (module docstring). spec learner_max (rule None, the reference), hedge (search_hedge) or MODULE:FUNC
    (rule(grid, pi_own, pi_opp) -> value at every node through depthn_backups.backup). solve() slices every (root, block) into a
    unit, runs world_prices on each in process or through a pool (pickling and IPC inside the lap), then reduces each root."""
    def __init__(self, spec='learner_max', rule=None, hedge=None, hedge_name=None):
        self.spec, self.rule, self.hedge, self.hedge_name = spec, rule, hedge, hedge_name

    @classmethod
    def load(cls, spec, *, hedge_module=HEDGE_MODULE, hedge_commit=HEDGE_COMMIT):
        if spec == 'learner_max':return cls()
        if spec == 'hedge':return cls('hedge', hedge=load_hedge(hedge_module, hedge_commit), hedge_name=hedge_module)
        module, func = parse_solver(spec);rule = getattr(importlib.import_module(module), func, None)
        if not callable(rule):raise ValueError(f'--solver {spec}: {func} is not a callable in {module}')
        try:inspect.signature(rule).bind(None, None, None)
        except TypeError as error:raise ValueError(f'--solver {spec}: rule(grid, pi_own, pi_opp) -> value required ({error})') from None
        except ValueError:pass   # no introspectable signature (a builtin): trusted
        return cls(spec, rule)

    def describe(self):
        if self.spec == 'hedge':
            return dict(spec='hedge', stand_in=False, module=self.hedge_name, path=HEDGE_SOLVE, lam=HEDGE_LAM, seed=HEDGE_SEED, weights='1/W', iterations=0)
        if self.rule is None:
            return dict(spec='learner_max', stand_in=True, path='depth2_backups.backup_learner_max on as_depth2_frame (depth 2) or the merged Depth2Frame; '
                'depthn_backups.backup_learner_max (depth >= 3); then reduce_completed_q per block', note='the learner-max reference')
        return dict(spec=self.spec, stand_in=False, rule=dotted(self.rule), path='depthn_backups.backup(frame, root, rule) at every depth (the merged '
            'Depth2Frame: depth2_backups learner-max); then reduce_completed_q per block')

    def solve(self, views, requests, clock, pool=None):
        """views: one (frame, row offset) per block; requests: the call's block-0 Requests. Laps 'solve' (parts as_depth2_frame, slice,
        worlds or worlds_pool, root_reduce); returns one result per root."""
        depth = getattr(views[0][0], 'depth', None);kind = 'depth2' if depth is None else 'depthn'   # no depth: the merged Depth2Frame
        if kind == 'depthn' and self.spec == 'learner_max' and int(depth) == 2:   # the depth-2 reference: depth2_backups on as_depth2_frame
            from gpu_engine.device_depthn_pricer import as_depth2_frame
            converted = {}
            for frame, _ in views:
                if id(frame) not in converted:converted[id(frame)] = as_depth2_frame(frame)
            views, kind = [(converted[id(frame)], offset) for frame, offset in views], 'depth2'
            clock.lap('solve', 'as_depth2_frame')
        rule = self.rule if pool is None else None   # a pool worker resolves MODULE:FUNC by name
        units = [dict(solver=self.spec, hedge=self.hedge_name, kind=kind, rule=rule, fields=root_fields(frame, offset + i, kind))
            for i in range(len(requests)) for frame, offset in views]
        clock.lap('solve', 'slice')
        prices = list(pool.map(world_prices, units, chunksize=1)) if pool is not None else [world_prices(u) for u in units]
        clock.lap('solve', 'worlds' if pool is None else 'worlds_pool')
        K = len(views);results = [self.reduce(prices[i * K:(i + 1) * K], r) for i, r in enumerate(requests)]
        clock.lap('solve', 'root_reduce')
        return results

    def reduce(self, pairs, request):
        """One root from its blocks' WorldPrices pairs: hedge = search_hedge.bayes_hedge over all W worlds (lam=0; None refuses), else
        reduce_completed_q per block and compose."""
        from pokezero import v5_gumbel as original
        if self.spec == 'hedge':
            choice = self.hedge.bayes_hedge([w for pair in pairs for w in pair], tuple(request.root_legal), lam=HEDGE_LAM)
            if choice is None:return dict(status='REFUSED', strategy=None, root_value=None, worlds=2 * len(pairs))
            return dict(status='LABELED', strategy=tuple(float(x) for x in choice.strategy), root_value=float(choice.value), worlds=2 * len(pairs))
        return compose([reduce_block(pair, request.root_logits, request.root_legal, original) for pair in pairs], request.root_logits, request.root_legal, original)

    def moves(self, results, requests):
        """learner_max / MODULE:FUNC: argmax of the improved target; hedge: search_hedge.sample(strategy, search_uniform(2026110500, address))."""
        if self.spec != 'hedge':return [move_of(r) for r in results]
        return [None if r['strategy'] is None else self.hedge.sample(r['strategy'], self.hedge.search_uniform(HEDGE_SEED, q.plans.root.address))
            for r, q in zip(results, requests)]


def move_of(result):
    """The move: argmax of the improved target, lowest action on ties; None when the root refuses."""
    target = result.get('policy_target')
    return None if target is None else max(range(len(target)), key=lambda a: (target[a], -a))


# The solve pool -----------------------------------------------------------------------------------------------------------------------
_WORKER_ERROR = None
POOL_THREADS = {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}


def _init_worker(hedge_module):
    """A spawn worker's start: import the solve modules and, with the hedge, warm its LP path on a 2x2 game without a saddle."""
    global _WORKER_ERROR
    try:
        from pokezero import v5_gumbel   # noqa: F401
        from gpu_engine import depth2_backups, depthn_backups   # noqa: F401
        if hedge_module:importlib.import_module(hedge_module).matrix_game(np.asarray([[1., -1.], [-1., 1.]]))
    except Exception as error:_WORKER_ERROR = f'{type(error).__name__}: {error}'


def _ping(_):
    time.sleep(.02);return os.getpid(), _WORKER_ERROR


def start_pool(size, *, hedge_module=None, timer=time.perf_counter, context=None, timeout=600.):
    """A persistent spawn pool of `size` workers (1 BLAS/OpenMP thread each), returned once every worker has started, imported
    the solve modules and warmed the hedge LP: (pool, info). Its startup_seconds are reported apart and never enter a call.
    A worker that dies while starting makes multiprocessing.Pool respawn it forever, so startup is bounded by `timeout` seconds."""
    import multiprocessing
    clock = timer();deadline = time.monotonic() + timeout
    with patch.dict(os.environ, POOL_THREADS):pool = (context or multiprocessing.get_context('spawn')).Pool(size, initializer=_init_worker, initargs=(hedge_module,))
    seen, errors = set(), set()
    while len(seen) < size:
        try:answers = pool.map_async(_ping, range(2 * size), chunksize=1).get(timeout=max(1., deadline - time.monotonic()))
        except multiprocessing.TimeoutError:
            pool.terminate();raise SystemExit(f'the {size}-worker solve pool did not start within {timeout:.0f} s (a worker failing at import is respawned forever)') from None
        for pid, error in answers:
            seen.add(pid)
            if error:errors.add(error)
        if errors:pool.terminate();raise SystemExit(f'a solve pool worker failed to start: {sorted(errors)[0]}')
        if time.monotonic() > deadline:pool.terminate();raise SystemExit(f'only {len(seen)} of {size} solve pool workers answered within {timeout:.0f} s')
    return pool, dict(workers=size, answered=len(seen), startup_seconds=timer() - clock, start_method='spawn', threads_per_worker=POOL_THREADS,
        hedge_module=hedge_module, scope='created, imported and warmed before any timed call; per call the solve stage holds slicing, pickling, IPC and the worlds')


def variants_of(solvers):
    """A solver (in process) or a list of variants (name, solver, pool, pool_size) -> the variants."""
    if hasattr(solvers, 'solve'):return [SimpleNamespace(name=f'{getattr(solvers, "spec", "solver")}@pool0', solver=solvers, pool=None, pool_size=0)]
    return list(solvers)


def make_variants(solvers, pools):
    """Every (solver, pool size) pair, solver-major: pools maps size -> pool (0 -> None, in process)."""
    return [SimpleNamespace(name=f'{s.spec}@pool{n}', solver=s, pool=pool, pool_size=n) for s in solvers for n, pool in pools.items()]


def merged_pricer(module, executor, model, *, capacity, actor_capacity, k1_own, k1_opp):
    """The merged Depth2Pricer as d3_depth2_gate.make_pricer builds it for the gate and --pricer-only (digest=True, k_own=k_opp=9,
    child_block left to the pricer's default) plus capacity and fixed_capacity=True; returns (pricer, options). Refuses a pricer
    without fixed_capacity or digest (d5150eac or later required)."""
    cls = module.Depth2Pricer;parameters = inspect.signature(cls).parameters
    options = dict(digest=True, actor_capacity=actor_capacity, k_own=9, k_opp=9, k1_own=k1_own, k1_opp=k1_opp, capacity=capacity, fixed_capacity=True)
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        missing = sorted(k for k in options if k not in parameters)
        if missing:raise SystemExit(f'Depth2Pricer does not accept {missing}: pricer d5150eac or later required')
    return cls(executor, model, **options), options


# The search under test --------------------------------------------------------------------------------------------------------------
class Search:
    """One config's pricers at one pricer capacity: DepthNPricer configs built by d3_depthn_throughput.Caller (fixed_capacity=True,
    digest=False), sharing `programs` (the process's DepthNPricer program table) after binding the evaluator, so no bind resets
    it; depth2-merged one merged Depth2Pricer (merged_pricer) with its own kernel table."""
    def __init__(self, config, capacity, args, ctx, *, module=None, programs=None, depth2_module=None):
        self.name, self.capacity, self.ctx = config, capacity, ctx
        if config == MERGED:
            self.module = depth2_module or importlib.import_module('gpu_engine.device_depth2_pricer');self.config = check_config(config);self.pass_pricer = None
            self.pricer, self.options = merged_pricer(self.module, ctx.provider.executor, ctx.plan.model, capacity=capacity, actor_capacity=ctx.actor_capacity,
                k1_own=args.k1_own, k1_opp=args.k1_opp)
            self.options = dict(self.options, engine=dotted(type(self.pricer)), child_block=getattr(self.pricer, 'child_block', None))
            return
        caller = tn.Caller(config, SimpleNamespace(capacity=capacity, k1_own=args.k1_own, k1_opp=args.k1_opp, block_lanes=args.block_lanes), ctx, module=module)
        self.config, self.module = caller.config, caller.module
        self.pricer, self.pass_pricer = caller.pricer, getattr(caller, 'pass_pricer', None)
        self.options = dict(engine=dotted(type(self.pricer)), capacity=capacity, fixed_capacity=True, digest=False, block_lanes=args.block_lanes,
            k1_own=args.k1_own, k1_opp=args.k1_opp, shared_program_table=programs is not None)
        for pricer in self.pricers():
            if programs is not None and callable(getattr(pricer, 'bind', None)):pricer.bind(ctx.params);pricer.programs = programs

    def pricers(self):
        return [p for p in (self.pass_pricer, self.pricer) if p is not None]

    def widths(self):
        """The searched engine's slot widths at plies 2..N (searched: derived by the engine from its k's) and the pass's, if any."""
        plies = lambda k: (int(k),) if isinstance(k, (int, np.integer)) else tuple(int(x) for x in k)   # Depth2Pricer holds ints
        own, opp = plies(self.pricer.k_own), plies(self.pricer.k_opp)
        out = dict(k_own=own, k_opp=opp, nominal_leaves_per_world=nominal_leaves(own, opp))
        if self.pass_pricer is not None:
            p_own, p_opp = tuple(int(x) for x in self.pass_pricer.k_own), tuple(int(x) for x in self.pass_pricer.k_opp)
            out['depth2_pass'] = dict(k_own=p_own, k_opp=p_opp, nominal_leaves_per_world=nominal_leaves(p_own, p_opp))
        return out

    def _price(self, pricer, requests, recipe, clock, part, **kw):
        frame = pricer.price(requests, self.ctx.provider.materializer, recipe, self.ctx.params, **kw)
        self.ctx.fence();clock.lap('pricer', part)
        timing = frame.timing;materialize = float(timing.get('world_prepare_seconds') or 0.)
        # Compiles as the pricer reports them for this call (no dedupe): DepthNPricer's `compiles` (reset per price()), Depth2Pricer's
        # timing lower/compile seconds (0 when its kernel table already held the key).
        if hasattr(pricer, 'compiles'):compiles = tuple(pricer.compiles)
        else:
            lower, compile_ = float(timing.get('lower_seconds') or 0.), float(timing.get('compile_seconds') or 0.)
            compiles = ((('depth2_kernel', timing.get('root_capacity'), timing.get('child_block')), lower, compile_),) if lower or compile_ else ()
        compiled = float(sum(c[1] + c[2] for c in compiles))
        clock.move(materialize, 'pricer', 'world_construction', source_part=part, target_part='materialize')
        clock.move(compiled, 'pricer', COMPILE, source_part=part, target_part=part)
        number = lambda key: float(timing[key]) if isinstance(timing.get(key), (int, float)) else None
        loop, pack, device = number('ply_loop_seconds'), number('host_pack_seconds'), number('device_seconds')
        return frame, dict(part=part, roots=len(requests), device_seconds=device, frame_wall_seconds=number('frame_wall_seconds'), world_prepare_seconds=materialize,
            compile_seconds=compiled, p1_seconds=None if None in (loop, pack, device) else device - (loop - pack), ply_loop_seconds=loop, host_pack_seconds=pack,
            enumerated_leaves=timing.get('enumerated_leaves', timing.get('turn2_enumerated_leaves')),
            evaluated_leaf_lanes=timing.get('evaluated_leaf_lanes', timing.get('turn2_evaluated_leaf_lanes')),
            compiles=[dict(key=repr(c[0]), lower_seconds=float(c[1]), compile_seconds=float(c[2])) for c in compiles])

    def price(self, requests, recipe, clock):
        """One pricing of `requests` to a host frame: depth N, or (searched) the pass, the host maps and depth N. Returns (frame, infos)."""
        if self.config['kind'] != 'searched':   # depthn and merged: one pricing (infos['depthn'] is always the search's own pricing)
            part = 'depth2_merged' if self.config['kind'] == 'merged' else 'depthn'
            frame, info = self._price(self.pricer, requests, recipe, clock, part);return frame, dict(depthn=info)
        raw, first = self._price(self.pass_pricer, requests, recipe, clock, 'depth2_pass');k = self.config['k']
        own, opp, _ = split_searched(self.module.searched_maps(self.module.as_depth2_frame(raw), k, k));raw = None
        clock.lap('pricer', 'searched_maps_host')
        widths, slots = [int(own.shape[-1]), int(opp.shape[-1])], [int(self.pricer.k_own[0]), int(self.pricer.k_opp[0])]
        if widths[0] > slots[0] or widths[1] > slots[1]:raise ValueError(f'searched_maps widths {widths} exceed the pricer\'s ply-2 slot widths {slots}')
        frame, second = self._price(self.pricer, requests, recipe, clock, 'depthn', selection_maps={2: (own, opp)})
        return frame, dict(depth2_pass=first, depthn=second, maps_widths=widths, slot_widths=slots)


def expanded_nodes(frames):
    """Expanded nodes the call priced (turn-1 cells and every intermediate ply's nodes with status 2, once per distinct frame); None for a frame without them."""
    total, seen = 0, set()
    for frame in frames:
        if id(frame) in seen:continue
        seen.add(id(frame))
        if getattr(frame, 't1_status', None) is None:return None
        total += int(np.count_nonzero(np.asarray(frame.t1_status) == 2)) + sum(int(np.count_nonzero(np.asarray(n.status) == 2)) for n in getattr(frame, 'nodes', ()))
    return total


def count_spread(values):
    """min, median and max of per-call counts (None when any is unknown)."""
    if not values or any(v is None for v in values):return None
    return dict(min=float(min(values)), median=float(np.median(values)), max=float(max(values)))


def run_call(search, ctx, roots, worlds, mode, *, solver=None, variants=None, rotate=0, timer=time.perf_counter):
    """One call: host_prep, K - 1 plannings and the pricing (K sequential passes of C roots, or one packed pass at K*C) once, then
    for every variant (solver, pool size) its solve and return on the same frames, in an order rotated by `rotate`; all contiguous
    laps. record['stages'] covers the whole call (solve and return summed over variants: sum == wall_seconds); each variant's
    end-to-end total = host_prep + world_construction + pricer + its solve + its return. Returns (record, block-0 requests)."""
    variants = variants_of(variants if variants is not None else solver)
    blocks = worlds // 2;clock = Clock(timer)
    requests, parts, prep_info = ctx.prep(roots);ctx.fence();clock.lap('host_prep')
    for name, seconds in parts.items():clock.note('host_prep', name, seconds)
    count = len(requests);planned = [list(requests)]
    for block in range(1, blocks):
        recipe, context = ctx.blocks[block]
        with context():planned.append([ctx.replan(r, recipe) for r in requests])
        clock.lap('world_construction', 'plan_starts')
    views, infos = [], []
    if mode == 'sequential':
        for block in range(blocks):
            recipe, context = ctx.blocks[block]
            with context():frame, info = search.price(planned[block], recipe, clock)
            views.append((frame, 0));infos.append(dict(info, block=block))
    elif mode == 'packed':
        if blocks < 2:raise ValueError('packed needs W > 2')
        recipe, context = ctx.blocks[0]   # K copies of block 0: every start's seed is this recipe's (materialization checks it)
        with context():frame, info = search.price([r for _ in range(blocks) for r in planned[0]], recipe, clock)
        views = [(frame, block * count) for block in range(blocks)];infos.append(dict(info, block='packed'))
    else:raise ValueError(f'mode in {MODES}')
    common = sum(clock.stages[s] for s in ('host_prep', 'world_construction', 'pricer'));frames = [frame for frame, _ in views]
    order = variants[rotate % len(variants):] + variants[:rotate % len(variants)];last = clock.last;out = {}
    for v in order:
        sub = Clock(timer, start=last)
        results = v.solver.solve(views, requests, sub, pool=v.pool)
        moves = v.solver.moves(results, requests);leftover = device_arrays(frames)
        if leftover:
            import jax
            jax.block_until_ready(leftover)
        ctx.fence();sub.lap('return');last = sub.last
        for stage in ('solve', 'return'):clock.stages[stage] += sub.stages[stage]
        out[v.name] = dict(solver=getattr(v.solver, 'spec', None), pool=v.pool_size, solve_seconds=sub.stages['solve'], return_seconds=sub.stages['return'],
            parts=dict(sub.parts), total_seconds=common + sub.stages['solve'] + sub.stages['return'], labeled=sum(r['status'] == 'LABELED' for r in results),
            refused=sum(r['status'] != 'LABELED' for r in results), moves=moves, leftover_device_arrays=len(leftover))
    clock.last = last;record = clock.record()
    enumerated = [i['depthn'].get('enumerated_leaves') for i in infos]
    record.update(roots=count, worlds=worlds, mode=mode, blocks=blocks, common_seconds=common, variants={v.name: out[v.name] for v in variants},
        variant_order=[v.name for v in order], pricing=infos, enumerated_leaves=None if any(e is None for e in enumerated) else int(sum(enumerated)),
        window=getattr(roots, 'window', None), root_keys=[list(k) if isinstance(k, tuple) else k for k in getattr(roots, 'keys', [])],
        host_prep_steps=prep_info.get('steps'), detach_passes=prep_info.get('detach_passes'), expanded_nodes=expanded_nodes(frames))
    del record['total_seconds']   # every variant has its own end-to-end total
    return record, requests


def measure_series(search, ctx, roots_of, worlds, mode, repeats, *, solver=None, variants=None, progress, memory, baseline, capacity, timer=time.perf_counter):
    """1 first (cold) call on move window 0 + `repeats` warm calls, warm call i on window i (roots_of(i)), of one (config, capacity,
    W, mode), every variant solving the same frames (order rotated per call); one series per variant, statistics over the warm
    calls only. Every call's Requests are checked; a warm call that compiled is logged (WARM_COMPILE)."""
    variants = variants_of(variants if variants is not None else solver);calls = []
    for i in range(1 + repeats):
        roots = roots_of(i)
        with gate.stage(progress, 'latency_call', config=search.name, capacity=capacity, worlds=worlds, mode=mode, call=i, warm=i > 0, window=i) as info:
            record, requests = run_call(search, ctx, roots, worlds, mode, variants=variants, rotate=i, timer=timer)
            record['memory'] = memory();calls.append(record)
            info.update(compile_seconds=record['stages'][COMPILE], common_ms=1e3 * record['common_seconds'],
                total_ms={k: 1e3 * v['total_seconds'] for k, v in record['variants'].items()},
                solve_ms={k: 1e3 * v['solve_seconds'] for k, v in record['variants'].items()}, labeled={k: v['labeled'] for k, v in record['variants'].items()})
        ctx.check(requests, roots)
        if i > 0 and record['stages'][COMPILE] > 0 and progress is not None:
            progress('WARM_COMPILE', config=search.name, capacity=capacity, worlds=worlds, mode=mode, call=i, compile_seconds=record['stages'][COMPILE])
        for entry in record['pricing']:
            for key in ('depth2_pass', 'depthn'):
                for c in (entry.get(key) or {}).get('compiles', ()):
                    if progress is not None:progress('COMPILE', config=search.name, pricer_capacity=search.capacity, call=i, part=entry[key]['part'], **c)
    return [summarize(calls, search=search, capacity=capacity, worlds=worlds, mode=mode, baseline=baseline, variant=v) for v in variants]


def variant_view(call, name):
    """One call as one variant saw it: the shared stages with its own solve and return, its end-to-end total and parts."""
    v = call['variants'][name];stages = {k: call['stages'][k] for k in ('host_prep', 'world_construction', 'pricer', COMPILE)}
    stages.update(solve=v['solve_seconds'], **{'return': v['return_seconds']})
    return dict({k: x for k, x in call.items() if k not in ('variants', 'stages', 'parts')}, stages=stages, parts={**call['parts'], **v['parts']},
        total_seconds=v['total_seconds'], labeled=v['labeled'], refused=v['refused'], moves=v['moves'], leftover_device_arrays=v['leftover_device_arrays'])


def summarize(calls, *, search, capacity, worlds, mode, baseline, variant):
    """One series for one variant: the first call apart (its compile, wall and stages), warm-call statistics, leaves, memory and
    every call as this variant saw it."""
    calls = [variant_view(c, variant.name) for c in calls];first, warm = calls[0], calls[1:]
    parts = sorted({k for c in warm for k in c['parts']})
    peaks = [c['memory'].get('process_peak_bytes_so_far') for c in calls if c.get('memory') and c['memory'].get('process_peak_bytes_so_far') is not None]
    peak = max(peaks) if peaks else None;enumerated = first.get('enumerated_leaves')
    pass_leaves = [i['depth2_pass'].get('enumerated_leaves') for i in first['pricing'] if 'depth2_pass' in i]
    widths = search.widths();spec = getattr(variant.solver, 'spec', None)
    leaves = dict(nominal_per_world=widths['nominal_leaves_per_world'], measured_per_world=None if enumerated is None else enumerated / (capacity * worlds),
        enumerated=enumerated, scope='measured = the first call\'s enumerated leaves (depth N\'s, searched included) / (C x W); identical calls repeat it')
    if 'depth2_pass' in widths:
        leaves['depth2_pass'] = dict(nominal_per_world=widths['depth2_pass']['nominal_leaves_per_world'],
            measured_per_world=None if not pass_leaves or None in pass_leaves else sum(pass_leaves) / (capacity * worlds))
    solve = (HEDGE_SOLVE if spec == 'hedge' else MERGED_SOLVE) if search.config['kind'] == 'merged' else (HEDGE_SOLVE if spec == 'hedge' else spec)
    return dict(config=search.name, capacity=capacity, worlds=worlds, mode=mode, solver=spec, pool=variant.pool_size, variant=variant.name, blocks=worlds // 2,
        pricer_capacity=search.capacity, timing_proxy=mode == 'packed', proxy_note=PACKED_NOTE if mode == 'packed' else None,
        slot_widths=dict(k_own=widths['k_own'], k_opp=widths['k_opp']), engine=search.options, solve=solve,
        first_call=dict(compile_seconds=first['stages'][COMPILE], call_wall_seconds=first['wall_seconds'], total_seconds=first['total_seconds'],
            stages=dict(first['stages']), cold_of_config_capacity=None),
        warm={**{s: distribution([c['stages'][s] for c in warm]) for s in STAGES}, 'total': distribution([c['total_seconds'] for c in warm])},
        warm_parts={k: distribution([c['parts'].get(k, 0.) for c in warm]) for k in parts},
        warm_calls=len(warm), warm_calls_with_compiles=sum(c['stages'][COMPILE] > 0 for c in warm), warm_compile_seconds=sum(c['stages'][COMPILE] for c in warm),
        labeled=first['labeled'], leftover_device_arrays=max(c['leftover_device_arrays'] for c in calls), leaves=leaves,
        windows=dict(warm=[c.get('window') for c in warm], distinct_roots=len({json.dumps(k) for c in warm for k in c.get('root_keys') or ()}),
            root_slots=capacity * len(warm), scope='distinct roots over the warm windows; fewer than C x N when the chunk wraps'),
        spread=dict(leaves_per_world=count_spread([None if c.get('enumerated_leaves') is None else c['enumerated_leaves'] / (capacity * worlds) for c in warm]),
            expanded_nodes=count_spread([c.get('expanded_nodes') for c in warm]), refused_roots=count_spread([c.get('refused') for c in warm]),
            host_prep_steps=count_spread([c.get('host_prep_steps') for c in warm]), scope='per warm call (each its own move window)'),
        memory=dict(baseline=baseline, process_peak_bytes_so_far=peak,
            peak_over_baseline_in_use=None if peak is None or (baseline or {}).get('bytes_in_use') is None else peak - baseline['bytes_in_use']),
        calls=calls)


def measure_config(name, worlds, args, ctx, progress, *, solver=None, variants=None, module=None, programs=None, depth2_module=None, memory=t2.memory_stats,
        timer=time.perf_counter):
    """Every capacity, W and mode of one config, each solved by every variant; one Search per (config, pricer capacity), so a
    packed K*C pricer is reused as the sequential pricer of capacity K*C. The first series of each capacity is marked; its first
    call reports what it compiled."""
    out, searches = [], {};variants = variants_of(variants if variants is not None else solver)
    for capacity in args.capacity_values:
        roots_of = lambda window, capacity=capacity: ctx.roots(capacity, window);baseline = memory() or {};first = True
        for w in worlds:
            blocks = w // 2
            for mode in ('sequential',) + (('packed',) if blocks > 1 and args.packed else ()):
                size = capacity if mode == 'sequential' else blocks * capacity
                if size not in searches:searches[size] = Search(name, size, args, ctx, module=module, programs=programs, depth2_module=depth2_module)
                group = measure_series(searches[size], ctx, roots_of, w, mode, args.repeats, variants=variants, progress=progress, memory=memory,
                    baseline=baseline, capacity=capacity, timer=timer)
                for series in group:series['first_call']['cold_of_config_capacity'] = first
                first = False;out.extend(group)
    return out


# The bank chunk and the real host prep -------------------------------------------------------------------------------------------------
@contextmanager
def capture_chunk(captured, *, compile_cache=None):
    """While gate.bank_chunk_requests runs: right after its JAX import (gate._probe, which follows the memory policy and the
    Provider import), wrap Provider._prepare_batched to keep the (device roots, host frames, actor bank) it is given and, with
    compile_cache, enable JAX's persistent compilation cache before the collector's first compile."""
    probe = gate._probe
    with ExitStack() as stack:
        def hooked(worker, runtime, jax):
            probe(worker, runtime, jax)
            from gpu_engine.v5_device_teacher import Provider
            original = Provider._prepare_batched
            def keep(self, device_roots, frames, bank):
                captured.update(roots=device_roots, frames=frames, bank=bank);return original(self, device_roots, frames, bank)
            stack.enter_context(patch.object(Provider, '_prepare_batched', keep))
            if compile_cache:
                from pokezero.owning_worker_runtime import configure_compilation_cache
                captured['compile_cache'] = configure_compilation_cache(compile_cache)
        stack.enter_context(patch.object(gate, '_probe', hooked))
        yield captured
    if 'frames' not in captured:raise SystemExit('the bank chunk\'s _prepare_batched inputs were not captured (bank_chunk_requests changed shape)')


def window_roots(available, capacity, window):
    """Move window `window`: the labeled roots window*C .. window*C+C-1 of the chunk, cycled."""
    available = list(available)
    if not available:raise ValueError('the chunk has no priced request')
    return [available[(window * capacity + j) % len(available)] for j in range(capacity)]


def address_key(request):
    a = request.plans.root.address;return (int(a.stream), int(a.frame), int(a.owner))


def host_prep(ctx, roots):
    """host_prep for exactly these C roots (module docstring): Provider._prepare_batched's per-root core on the captured chunk.
    Returns (block-0 Requests in root order, {detach, plan_starts, assemble} seconds measured inside, {steps, detach_passes}): a
    lockstep live batch would pay one RootContext slice and detach pass, a chunk window pays one per distinct scan step."""
    import jax
    from gpu_engine import v5_device_teacher as teacher, device_public_worlds as worlds
    from gpu_engine.device_public_teacher_batch import Request
    frames, device_roots, source, recipe = ctx.chunk.frames, ctx.chunk.roots, ctx.provider.source, ctx.provider.recipe
    clock = time.perf_counter();contexts, hosts, batch, detached = {}, {}, {}, {}
    for step in sorted({s for s, _, _ in roots.slots}):
        contexts[step] = jax.tree.map(lambda x: x[step], device_roots);hosts[step] = jax.device_get((contexts[step].streams, contexts[step].frames))
    for step, row, owner in roots.slots:batch.setdefault(step, []).append((row, owner))
    for step, items in batch.items():
        for (row, owner), root in zip(items, worlds.detach_roots(contexts[step], source, rows=[r for r, _ in items], owners=[o for _, o in items],
                enable_device_worlds=True)):detached[(step, row, owner)] = root
    detach = time.perf_counter() - clock;plan = assemble = 0.;requests = []
    for (step, row, owner), bound in zip(roots.slots, roots.bound):
        clock = time.perf_counter();streams, frame_ids = hosts[step]
        if int(streams[row]) != int(frames.streams[step, row]) or int(frame_ids[row]) != int(frames.frames[step, row]):raise ValueError('PRE root context address differs')
        inputs = {name: getattr(frames, name)[step, row, owner][None, None] for name in teacher.INPUTS};inputs['history_mask'] = np.ones((1, 1), bool)
        logits, legal = tuple(map(float, frames.raw_policy_logits[step, row, owner])), tuple(map(bool, frames.legal_action_mask[step, row, owner]))
        learner, opponent = int(frames.actor_indices[step, row, owner]), int(frames.actor_indices[step, row, 1 - owner])
        if ctx.head_override:logits, learner, opponent = bound.root_logits, bound.learner_actor, bound.opponent_actor   # gate.bind_roots' head prior, actors 0/0
        middle = time.perf_counter();plans = worlds.plan_starts(detached[(step, row, owner)], source, recipe);after = time.perf_counter()
        requests.append(Request(plans, logits, legal, inputs, learner, opponent))
        plan += after - middle;assemble += (middle - clock) + (time.perf_counter() - after)
    return requests, dict(detach=detach, plan_starts=plan, assemble=assemble), dict(steps=len(contexts), detach_passes=len(batch))


def check_prepared(requests, bound):
    """The timed host_prep must rebuild the bank's own Requests: same address, legal mask, root prior, actors and world plans."""
    if len(requests) != len(bound):raise ValueError('host_prep returned a different root count')
    for mine, theirs in zip(requests, bound):
        a, b = mine.plans, theirs.plans
        same = (address_key(mine) == address_key(theirs) and tuple(mine.root_legal) == tuple(theirs.root_legal)
            and tuple(map(float, mine.root_logits)) == tuple(map(float, theirs.root_logits))
            and (mine.learner_actor, mine.opponent_actor) == (theirs.learner_actor, theirs.opponent_actor) and a.selected == b.selected
            and [(s.world, s.seed, s.refusals, s.revealed_source_indices) for s in a.starts] == [(s.world, s.seed, s.refusals, s.revealed_source_indices) for s in b.starts])
        if not same:raise ValueError(f'host_prep rebuilt a different request for root {address_key(theirs)}')


def prepare(args):
    """The chunk through d3_depthn_throughput.prepare (the gate's builder, head override, evaluator) with _prepare_batched's
    inputs captured; the address -> (step, row, owner) map; the recipe blocks; the real host prep, planner and fence."""
    captured = {}
    view = SimpleNamespace(**vars(args));view.capacity = 1;view.w_values = (2,)
    with capture_chunk(captured, compile_cache=args.compile_cache):base = tn.prepare(view)
    from pokezero import v5_gumbel as original
    from gpu_engine import v5_device_teacher as teacher, device_public_worlds as worlds
    frames, bank = captured['frames'], captured['bank'];slot_of = {}
    for step, row, owner in teacher._requests(frames, bank):
        slot_of[(int(frames.streams[step, row]), int(frames.frames[step, row]), int(owner))] = (int(step), int(row), int(owner))
    provider = base.provider;most = max(w for ws in args.config_worlds.values() for w in ws) // 2
    seeds = gate.world_block_seeds(provider.recipe.seed, most)
    bypass = lambda: patch.object(original.Recipe, 'selected', lambda self, address: True)   # U1's quarter-selection bypass, blocks > 0
    blocks = [(provider.recipe, nullcontext) if b == 0 else (original.Recipe(seed), bypass) for b, seed in enumerate(seeds)]
    ctx = SimpleNamespace(**vars(base));missing = [address_key(r) for r in base.available if address_key(r) not in slot_of]
    if missing:raise SystemExit(f'{len(missing)} labeled roots have no (step, row, owner) in the captured chunk, e.g. {missing[0]}')
    def roots(capacity, window=0):
        bound = window_roots(ctx.available, capacity, window)
        return SimpleNamespace(bound=bound, slots=[slot_of[address_key(r)] for r in bound], window=window, keys=[address_key(r) for r in bound])
    ctx.chunk = SimpleNamespace(roots=captured['roots'], frames=frames, bank=bank);ctx.head_override = bool(args.head_archive);ctx.seeds = seeds;ctx.blocks = blocks
    ctx.roots = roots;ctx.prep = lambda r: host_prep(ctx, r);ctx.fence = device_fence;ctx.check = lambda requests, r: check_prepared(requests, r.bound)
    ctx.replan = lambda request, recipe: replace(request, plans=worlds.plan_starts(request.plans.root, provider.source, recipe))
    ctx.compile_cache = captured.get('compile_cache')
    return ctx


# Header: the constructor path actually timed, the source commit, the note ------------------------------------------------------------------
def where(x):
    """(module, qualname) of a class or routine, else of the object's type; resolved at run time from the objects actually called."""
    target = x if inspect.isclass(x) or inspect.isroutine(x) else type(x)
    return getattr(target, '__module__', '?'), getattr(target, '__qualname__', getattr(target, '__name__', '?'))


def dotted(x):
    return '.'.join(where(x))


def constructor_path(provider, pricer_module, worlds_module, merged_module=None):
    """The world-construction path this run timed: the provider, its materializer and source objects' classes, the planner and
    detacher this tool calls, and the materialization DepthNPricer.price calls (its module's batch._numpy_worlds ->
    mapping.materialize_host), each module.qualname. uses_phase_d_constructor: whether any of them lives in a Phase D module."""
    batch = getattr(pricer_module, 'batch', None);builder = getattr(batch, '_numpy_worlds', None);materialize = getattr(getattr(batch, 'mapping', None), 'materialize_host', None)
    found = dict(provider=provider, materializer=provider.materializer, source=provider.source, root_detach=worlds_module.detach_roots,
        world_planner=worlds_module.plan_starts, world_builder=builder, world_materialize=materialize, pricer=getattr(pricer_module, 'DepthNPricer', None))
    if merged_module is not None:   # the merged Depth2Pricer materializes through its own module's batch._numpy_worlds
        found.update(merged_pricer=getattr(merged_module, 'Depth2Pricer', None), merged_world_builder=getattr(getattr(merged_module, 'batch', None), '_numpy_worlds', None))
    path = {k: 'UNRESOLVED' if v is None else dotted(v) for k, v in found.items()}
    modules = sorted({where(v)[0] for v in found.values() if v is not None})
    if merged_module is not None:path['merged_same_world_builder'] = path['merged_world_builder'] == path['world_builder']
    path.update(modules=modules, uses_phase_d_constructor=any(m in PHASE_D_MODULES for m in modules),
        phase_d_modules_loaded=[m for m in PHASE_D_MODULES if m in sys.modules],
        timed=f'{path["world_planner"]} (blocks >= 1) + {path["world_builder"]} -> {path["world_materialize"]} on {path["materializer"]}',
        host_prep=f'{__name__}.host_prep (the per-root core of {path["provider"]}._prepare_batched) with {path["root_detach"]} and {path["world_planner"]} (block 0)')
    return path


def source_commit(root=CODE_ROOT, given=None, *, run=subprocess.run):
    """`git rev-parse HEAD` of the code directory when it holds a .git (a worktree's .git file counts), with its dirty flag
    (tracked files); otherwise --source-commit; otherwise UNKNOWN."""
    root = Path(root);head = dirty = error = None
    if (root / '.git').exists():
        try:
            git = lambda *a: run(['git', '-C', str(root), *a], capture_output=True, text=True, check=True, timeout=60).stdout.strip()
            head = git('rev-parse', 'HEAD') or None;dirty = bool(git('status', '--porcelain', '--untracked-files=no'))
        except (OSError, subprocess.SubprocessError) as failure:head, error = None, str(failure)
    commit = head or given
    return dict(commit=commit or 'UNKNOWN', origin='git rev-parse HEAD' if head else '--source-commit' if given else 'UNKNOWN (no .git and no --source-commit)',
        code_dir=str(root), git_head=head, given=given, dirty=dirty, git_error=error,
        matches_given=None if not (head and given) else head.startswith(given) or given.startswith(head))


def hedge_header(module, commit=HEDGE_COMMIT, name=HEDGE_MODULE):
    """The hedge module actually loaded: its file and sha256, --hedge-commit, and whether the file is 583555d2's (when that is the commit)."""
    path = getattr(module, '__file__', None);digest = hashlib.sha256(Path(path).read_bytes()).hexdigest() if path and Path(path).is_file() else None
    return dict(module=name, file=path, sha256=digest, commit=commit, reference_commit=HEDGE_COMMIT, reference_sha256=HEDGE_REFERENCE_SHA256,
        matches_reference=None if digest is None or commit != HEDGE_COMMIT else digest == HEDGE_REFERENCE_SHA256, lam=HEDGE_LAM, seed=HEDGE_SEED,
        weights='1/W', iterations=0, solve=HEDGE_SOLVE)


def header(context, commit):
    return dict(world_construction_path=context['world_construction_path'], source_commit=commit, constructor_note=CONSTRUCTOR_NOTE,
        solvers=context['solvers'], solver_pools=context.get('solver_pools', []), hedge=context.get('hedge'), variants=context.get('variants'),
        rule=RULE, limit_ms=LIMIT_MS, budget_ms=BUDGET_MS, stages=STAGE_SCOPE, packed=PACKED_NOTE,
        variants_scope='every (solver, pool size) solves the SAME priced frames of each call, in an order rotated per call; its end-to-end total = '
            'host_prep + world_construction + pricer (shared) + its own solve and return')


# Selection and report -----------------------------------------------------------------------------------------------------------------
def largest_within(p90_by_capacity, limit_ms=LIMIT_MS):
    """The headline: the largest capacity C such that C and every smaller measured capacity have p90 <= limit_ms (None when the
    smallest fails); also every capacity's raw verdict, the largest capacity that fits on its own and whether the two agree."""
    verdicts = {c: p is not None and p <= limit_ms for c, p in sorted(p90_by_capacity.items())};largest = None
    for c, fits in verdicts.items():
        if not fits:break
        largest = c
    raw = max((c for c, fits in verdicts.items() if fits), default=None)
    return dict(largest_capacity=largest, largest_capacity_any=raw, monotone=raw == largest, verdicts={str(c): fits for c, fits in verdicts.items()})


def select(series, limit_ms=LIMIT_MS):
    """Rows (config, W, mode, solver variant) -> per-capacity warm total median/p90 and the largest fitting C; per config and
    variant the largest fitting C (with its W) and the largest fitting W (with its C), over sequential rows, packed apart as a proxy."""
    rows = {}
    for s in series:rows.setdefault((s['config'], s['worlds'], s['mode'], s.get('variant')), [s, {}])[1][s['capacity']] = s['warm']['total']
    out = []
    for (config, worlds, mode, variant), (one, by) in rows.items():
        out.append(dict(config=config, worlds=worlds, mode=mode, variant=variant, solver=one.get('solver'), pool=one.get('pool'),
            by_capacity={str(c): dict(median_ms=v['median_ms'], p90_ms=v['p90_ms'], max_ms=v['max_ms']) for c, v in sorted(by.items())},
            **largest_within({c: v['p90_ms'] for c, v in by.items()}, limit_ms)))
    def best(rows_, key):
        fitting = [r for r in rows_ if r['largest_capacity'] is not None]
        if not fitting:return None
        r = max(fitting, key=key);return dict(capacity=r['largest_capacity'], worlds=r['worlds'], mode=r['mode'])
    configs = {}
    for config, variant in dict.fromkeys((r['config'], r['variant']) for r in out):
        mine = [r for r in out if (r['config'], r['variant']) == (config, variant)];seq = [r for r in mine if r['mode'] == 'sequential']
        packed = [r for r in mine if r['mode'] == 'packed']
        configs.setdefault(config, {})[variant] = dict(measured_worlds=sorted({r['worlds'] for r in mine}),
            largest_capacity=best(seq, lambda r: (r['largest_capacity'], r['worlds'])), largest_worlds=best(seq, lambda r: (r['worlds'], r['largest_capacity'])),
            packed_proxy=None if not packed else dict(largest_capacity=best(packed, lambda r: (r['largest_capacity'], r['worlds'])),
                largest_worlds=best(packed, lambda r: (r['worlds'], r['largest_capacity']))))
    return dict(rule=RULE, limit_ms=limit_ms, rows=out, configs=configs)


def _ms(x):
    return '-' if x is None else f'{x:.0f}' if x >= 100 else f'{x:.1f}'


def _short(name):
    return name.replace('gpu_engine.', '')


def markdown(report):
    h = report['header'];path = h['world_construction_path'];commit = h['source_commit'];caps = report['capacities'];sel = report['selection']
    world = f'world_construction: {_short(path["world_planner"])} (blocks >= 1) + {_short(path["world_builder"])} -> {_short(path["world_materialize"])}'
    pools = '; '.join(f'pool {p["workers"]}: startup {p["startup_seconds"]:.1f} s (excluded)' for p in h['solver_pools']) or 'in process only'
    lines = ['## Search latency: end-to-end ms per call, warm calls (median / p90)', '',
        f'Rule: {RULE} (N = {report["repeats"]}). Solvers: {", ".join(x["spec"] for x in h["solvers"])} (every solver and pool size on the same priced frames); solve pools: {pools}.',
        f'World construction path timed: {path["timed"]}; materializer object {path["materializer"]}; Phase D constructor on the path: '
        f'{"yes" if path["uses_phase_d_constructor"] else "no"}.',
        f'Source commit: {commit["commit"]} ({commit["origin"]}' + ('' if commit.get('dirty') is None else f', dirty={commit["dirty"]}') + ').',
        f'Note: {CONSTRUCTOR_NOTE}', '']
    if h.get('hedge'):
        x = h['hedge'];lines[-1:-1] = [f'Hedge: {x["module"]} sha256 {x["sha256"]} (commit {x["commit"]}; 583555d2 file: {x["matches_reference"]}), lam={x["lam"]}, '
            f'seed {x["seed"]}, weights {x["weights"]}: {HEDGE_SOLVE}.']
    if any(s['config'] == MERGED for s in report['series']):
        merged = next(s['engine'] for s in report['series'] if s['config'] == MERGED)
        lines[-1:-1] = [f'{MERGED}: {merged.get("engine")} (digest={merged.get("digest")}, child_block={merged.get("child_block")}, fixed_capacity='
            f'{merged.get("fixed_capacity")}, the gate\'s construction); learner_max / MODULE:FUNC solve {MERGED_SOLVE}; hedge solve search_hedge.depth2_hedged_root. '
            'depth2-fw is DepthNPricer at N=2, k=9.']
    lines += [
        '| config | W | mode | solver | pool | ' + ' | '.join(f'C={c} median / p90' for c in caps) + ' | largest C, every smaller C fitting too (p90 <= 800 ms) |', '|---|---|---|---|---|' + '---|' * len(caps) + '---|']
    for r in sel['rows']:
        cells = [f'{_ms(r["by_capacity"][str(c)]["median_ms"])} / {_ms(r["by_capacity"][str(c)]["p90_ms"])} {"fit" if r["verdicts"][str(c)] else "over"}'
            if str(c) in r['by_capacity'] else '-' for c in caps]
        largest = ('-' if r['largest_capacity'] is None else str(r['largest_capacity'])) + ('' if r['monotone'] else f' (C={r["largest_capacity_any"]} fits alone)')
        lines.append(f'| {r["config"]} | {r["worlds"]} | {r["mode"] + (" (proxy)" if r["mode"] == "packed" else "")} | {r["solver"]} | {r["pool"]} | '
            + ' | '.join(cells) + f' | {largest} |')
    lines += ['', 'Largest fitting, sequential (playable today): ' + '; '.join(
        f'{c} [{n}]: C={v["largest_capacity"]["capacity"]} at W={v["largest_capacity"]["worlds"]}, W={v["largest_worlds"]["worlds"]} at C={v["largest_worlds"]["capacity"]}'
        if v['largest_capacity'] else f'{c} [{n}]: none of C={caps} at W={v["measured_worlds"]}' for c, by in sel['configs'].items() for n, v in by.items()) + '.',
        f'Packed rows are a timing proxy ({PACKED_NOTE}).', '',
        f'### Stage medians (ms), warm calls [{world}]', '',
        f'| config | W | mode | solver | pool | C | host_prep | world_construction ({_short(path["world_planner"])} + {_short(path["world_builder"])}) | pricer | solve | return | total p50 | total p90 | first-call compile s | leaves/world nominal / measured |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for s in report['series']:
        w = s['warm'];leaves = s['leaves']
        lines.append(f'| {s["config"]} | {s["worlds"]} | {s["mode"]} | {s["solver"]} | {s["pool"]} | {s["capacity"]} | ' + ' | '.join(_ms(w[k]['median_ms']) for k in STAGES)
            + f' | {_ms(w["total"]["median_ms"])} | {_ms(w["total"]["p90_ms"])} | {s["first_call"]["compile_seconds"]:.1f} | {leaves["nominal_per_world"]} / '
            + ('-' if leaves['measured_per_world'] is None else f'{leaves["measured_per_world"]:.0f}') + ' |')
    return '\n'.join(lines) + '\n'


def report_of(args, sections, context, commit, *, isolated_run):
    series = [s for name in args.config_worlds if name in sections for s in sections[name]]
    report = dict(evaluator=context['evaluator'], header=header(context, commit), schema=SCHEMA,
        value_scale=gate.value_scale(context['root_values'], dict(count=0), args.value_reference), weights_affect_speed=t2.WEIGHTS_NOTE,
        source=context['source'], devices=context['devices'], capacities=list(args.capacity_values), configs={k: list(v) for k, v in args.config_worlds.items()},
        repeats=args.repeats, packed=args.packed, block_lanes=args.block_lanes, k1_own=args.k1_own, k1_opp=args.k1_opp, roots_available=context['roots_available'],
        cycled=context['roots_available'] < max(args.capacity_values), block_seeds=context.get('block_seeds'), compile_cache=context.get('compile_cache'),
        isolated=isolated_run, memory_scope=t2.MEMORY_SCOPE + ' The captured chunk RootContext stays alive for host_prep and is in every baseline.'
            + (' (isolated: one process per config)' if isolated_run else ' Configs share one process and one compiled-program table.'),
        selection=select(series), series=series)
    report['summary_markdown'] = markdown(report)
    return report


def isolated(args, argv, out):
    """--isolate: each config in its own subprocess (`--only NAME --out OUT/NAME`); returns ({name: its series}, {name: context})."""
    argv = list(sys.argv[1:] if argv is None else argv);clean, skip = [], False
    for token in argv:
        if skip:skip = False;continue
        if token == '--isolate':continue
        if token == '--out':skip = True;continue
        if token.startswith('--out='):continue
        clean.append(token)
    sections, contexts = {}, {}
    for name in args.config_worlds:
        target = out / name
        code = subprocess.call([sys.executable, '-m', 'gpu_engine.d3_search_latency', *clean, '--only', name, '--out', str(target)])
        if code != 0:raise SystemExit(f'--isolate: the {name} subprocess exited {code}')
        child = json.loads((target / 'latency.json').read_text());sections[name] = child['sections'][name];contexts[name] = child.get('context')
    return sections, contexts


def main(argv=None):
    args = parse(argv)
    if importlib.util.find_spec('gpu_engine.device_depthn_pricer') is None:raise SystemExit('gpu_engine.device_depthn_pricer is not on this tree')
    commit = source_commit(CODE_ROOT, args.source_commit);gate.log('SOURCE_COMMIT', **commit)
    gate.log('EVALUATOR', override=bool(args.head_archive), archive=args.head_archive, archive_sha256=args.head_archive_sha256,
        checkpoint_sha256=args.head_checkpoint_sha256, value_reference=args.value_reference, weights_affect_speed=t2.WEIGHTS_NOTE,
        resolved='EVALUATOR_RESOLVED follows once the bank is loaded')
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    if args.isolate:
        gate.log('LATENCY_START', configs=args.config_worlds, capacities=args.capacity_values, isolate=True)
        sections, contexts = isolated(args, argv, out);context = next(c for c in contexts.values() if c)
        context['solver_pools'] = [dict(p, config=name) for name, c in contexts.items() if c for p in (c.get('solver_pools') or [])]   # pools per config process
    else:
        ctx = prepare(args)   # solver modules import after the memory policy
        solvers = [Solver.load(spec, hedge_module=args.hedge_module, hedge_commit=args.hedge_commit) for spec in args.solver_specs]
        hedge = next((s.hedge for s in solvers if s.spec == 'hedge'), None);pools, pool_infos = {}, []
        import jax
        from gpu_engine import device_depthn_pricer, device_public_worlds
        merged = importlib.import_module('gpu_engine.device_depth2_pricer') if MERGED in args.config_worlds else None
        try:
            for size in args.pool_sizes:   # persistent pools, started (and their startup timed) before any call
                if not size:pools[0] = None;continue
                pools[size], info = start_pool(size, hedge_module=args.hedge_module if hedge is not None else None);pool_infos.append(info);gate.log('SOLVER_POOL', **info)
            variants = make_variants(solvers, pools)
            context = dict(evaluator=ctx.evaluator, root_values=list(ctx.root_values), roots_available=len(ctx.available), source=ctx.source,
                devices=[d.device_kind for d in jax.devices()], block_seeds=list(ctx.seeds), compile_cache=ctx.compile_cache, solvers=[s.describe() for s in solvers],
                solver_pools=pool_infos, hedge=None if hedge is None else hedge_header(hedge, args.hedge_commit, args.hedge_module), variants=[v.name for v in variants],
                world_construction_path=constructor_path(ctx.provider, device_depthn_pricer, device_public_worlds, merged))
            gate.log('WORLD_CONSTRUCTION_PATH', **context['world_construction_path'], source_commit=commit['commit'], note=CONSTRUCTOR_NOTE)
            if context['hedge']:gate.log('HEDGE', **context['hedge'])
            names = [args.only] if args.only else list(args.config_worlds);programs = {}
            gate.log('LATENCY_START', configs={n: args.config_worlds[n] for n in names}, capacities=args.capacity_values, roots_available=context['roots_available'],
                variants=context['variants'], repeats=args.repeats, packed=args.packed, isolate=False)
            sections = {name: measure_config(name, args.config_worlds[name], args, ctx, gate.log, variants=variants, programs=programs) for name in names}
        finally:
            for pool in pools.values():
                if pool is not None:pool.terminate();pool.join()
        if args.only:
            (out / 'latency.json').write_text(json.dumps(dict(evaluator=ctx.evaluator, only=args.only, sections=sections, context=context),
                indent=1, allow_nan=False, default=str) + '\n')
            return 0
    report = report_of(args, sections, context, commit, isolated_run=args.isolate)
    (out / 'latency.json').write_text(json.dumps(report, indent=1, allow_nan=False, default=str) + '\n')   # evaluator first, then the header
    gate.log('LATENCY_COMPLETE', out=str(out / 'latency.json'), selection=report['selection']['configs'], hedge=report['header']['hedge'])
    print('=== LATENCY SUMMARY (markdown) ===\n' + report['summary_markdown'] + '=== END LATENCY SUMMARY ===', flush=True)
    return 0


if __name__ == '__main__':raise SystemExit(main())
