#!/usr/bin/env python3
"""Re-price a live DepthN run's search decisions exactly, and record the per-world value grids behind each one.

Design. One process, one GPU. It builds the live server's own DepthNLiveSearch (depthn_live_server.build on the server's
flags: bucket 1, max_requests 1, per-battle sessions, saved-root admission off), warms it up on the server's own
--warmup-requests file as the server does (depthn_live_server.read_requests, then search.warmup: the same carry reference),
and replays each selected cell's battle from its owner capture (owning-wire/<ordinal>.jsonl through
phase_d_owning_capture.read_capture) decision by decision, as depthn_live_bench's session mode does: a single-legal decision
is never sent (the live actor never sends one), every other one goes to process_many alone, its request built as the bench
builds it (the eight address keys, the wire schema, a fresh request_id, the rung and the bench's never-expiring budget; the
lines as the JSON list the wire carries). Nothing on the decision path is re-implemented: the replay is the live code (the
tree guard) on the live inputs.
- Tree guard (before anything loads): HEAD's gpu_engine/ and src/ trees, and the two scripts the replay executes
  (depthn_live_server.py, depthn_live_bench.py), must equal those of the live server tree LIVE_COMMIT (pinned in LIVE_TREES,
  and checked against the commit itself when the clone has it), with no tracked change and no untracked .py file there.
  --tree-commit (the server's label for the code served, which the bench passes through) must name HEAD when given.
  --accept-tree COMMIT replays against a modified tree instead: COMMIT must be HEAD (git rev-parse --verify COMMIT^{commit}),
  the guarded paths' trees are taken from it (git rev-parse COMMIT:<path>) and recorded with the live pins and the paths that
  differ from them, and the guarded paths and scripts/ must hold no tracked change and no untracked .py file. The header
  records it (tree_guard.modified_tree, replay_against: 'replay against a modified tree ...'); parity against the live run is
  still recorded, but expected to fail. Without the flag the guard is unchanged.
- Identity: the built search's run header (depthn_live_server.header) must match the live READY identity (LIVE_IDENTITY:
  seeds, session cap, precision pins, x64 off, sessions, champion and input sha256s) with the arm's rung, bucket 1 and
  max_requests 1; the replay refuses to warm up otherwise.
- Capture without changing a result: around each process_many call only, depthn_live_search.decide and
  constructor_worlds.prepare_roots are wrapped. DepthNLiveSearch._search looks both up through their modules at call time
  (`decide` a module global, `cw.prepare_roots` a module attribute); check_call_sites verifies that from its bytecode before
  anything loads. The wrappers keep references to the arguments and results, call the originals with the same objects and
  are restored when the call returns or raises.
  - decide: root_logits (9), root_legal (9) and, per world in block order (block k's world j is world 2k+j: averaging's
    order, and the constructor's world index), search_hedge's WorldPrices: opponent_requested, kept (opponent_legal: the
    replies turn 1 enumerated), root_peer_logits (opponent_logits) and grid (joint_values[a][b]: own lane a, opponent lane b;
    None -> null, floats as Python floats, never rounded).
  - prepare_roots: per world its index, world seed, kernel seed and refusals, and, read after the call from each constructed
    world's state (jax.device_get, as constructor_worlds.build does):
    - the opponent's (the non-owner side's) mapping facts through the constructor's own engine-id tables, metadata
      ['vocabulary'] (what phase_d_protocol_world and phase_d_live_worlds map with): the active party slot, its species
      (normalize_id), its four move ids (phase_d_live_worlds._source_move: Hidden Power collapsed; None for an empty slot),
      its charge move, and the party (opponent.party[]) in party (physical slot) order, each member's set: slot, order
      (its position in the peer's Showdown request order: the completed carry's own_order[peer], which
      constructor_worlds.order_peer sets; None when the carry has none), species, moves (its four move ids as the request
      spells them, Hidden Power typed: the engine move id's own typed name, e.g. hiddenpowerfire70, or, for a typeless
      engine entry, hiddenpower + the world's hidden_power_type (+ hidden_power_power); a power suffix stays, so a reader
      strips trailing digits; unlike active_moves and the lane labels, which keep _source_move's collapsed form),
      move_engine_ids, ability (the engine's current ability: Trace's copy included), base_ability, item (None: no item),
      fainted (hp <= 0, the engine's own rule), hp, max_hp; opponent.request_order lists the party slots in that request
      order (None without a carry order);
    - the opponent's encoder lanes: the second output of device_depth2_pricer._make_views(executor)(state, public, owner),
      the pricer's own peer view (build's root-view call; turn 1's peer view), on the live pricers' shared executor:
      legal (9), move_targets (4), switch_targets (5) and error bits; gpu_engine.state.usable_moves on the opponent's side
      (lane 0 is Struggle when none is usable, the charge move while one charges); and `lanes`, what each legal lane plays
      (move:<id> through move_targets, switch:<species> through switch_targets). The grid's column b is this lane space.
    Per world, when the opponent is requested, kept must equal the peer legal lanes (and opponent_requested any of them): a
    disagreement is recorded (kept_matches_peer_legal false), never raised.
  - Guard fields: every sent row's `root` (a dict on every row; `decided` false for a root that never reached decide, whose
    root_logits and root_legal are then null) holds own_moves: the owner's move id on each action lane 0-3, read from the
    request's own_request_line by the live guards' own depthn_effect_guard.own_moves (the live actor's) (None on a lane without a move);
    own_recharge_lanes: depthn_effect_guard.recharge_lane on each move lane 0-3 (true when its move is in
    depthn_effect_guard.RECHARGE_MOVES, the one set the live actor's guard uses, imported from it and recorded in the header
    as recharge_moves; a switch lane is never a recharge move); opponent_recharge: the peer must recharge at the root by
    constructor_worlds.line_facts on the request's lines (B2's rule), whatever the replayed tree does with it; and
    refusal_cause: null for a COMPLETE replay, else the constructor's RootRefusal kind when prepare_roots refused the root
    (OPPONENT_RECHARGE_UNREPRESENTED, PEER_REQUEST_ORDER_UNAVAILABLE, ...), else the replay's cause. A root refused
    OPPONENT_RECHARGE_UNREPRESENTED has no priced worlds: the live actor plays greedy there. A scorer applies the
    live actor's own-recharge guard (recharge_guarded: never an override INTO a recharge move greedy did not choose) to
    every operator's choice against the live join's greedy_action_index with them.
  - Effect guard: the row's effect_guard is the replayed response's own (the live search's UNREPRESENTED_EFFECT record,
    pokezero.depthn_effect_guard, read through the pricers' observer seam in the same process_many call, so one pass gives the
    row and its turn-2 priors): tripped, b (turn 1 / turn 2), c for the search's own choice, the reasons, the (a) flag,
    tier_b_legal, our guarded lanes, the conditions per side and the masses of every watched move (turn 1 per side and world,
    turn 2 per side, world and cell); None for a root the replay did not decide. The RR scorer applies it per operator.
  - Peer inputs (peer-inputs-<shard>.npz, every run): per decided root, each constructed world's depth-0 peer inputs, the
    fourth output of device_depth2_pricer._make_views(executor)(state, public, owner) on the completed world (the tensors
    the pricer's turn 1 evaluates root_peer_logits from), and that view's peer legal mask. Self-check: the worlds are
    stacked in block order as the pricer stacks them (a world without a state takes the first constructed world's, as
    pack_turn1 does) and the server's own champion (the pricer's forward, ChampionEvaluator's adapter, with the search's
    own parameters, under jax.jit) runs on them; on every world priced with the opponent requested its logits must equal
    the recorded root_peer_logits bit for bit. Per world: peer_inputs_index (its row in the sidecar), peer_self_check
    (true / false, null where not applicable; a priced, opponent-requested world whose capture raised is false, never
    skipped) and peer_self_check_max_abs_diff; per row peer_inputs_check (checked, equal, failed_worlds, written); the
    summary's peer_inputs_check counts them, lists the failures and passes only when every decided root had at least one
    checked world and every checked world was equal (a decided root with none counts as unchecked). Only decided roots are
    captured or written (a root refused after prepare_roots has no root_peer_logits). The sidecar holds,
    stacked over its K worlds, the champion adapter's own view slice (ChampionEvaluator.view: the first 23 tokens) of
    those inputs: numeric_features f32 [K,23,132], categorical_ids i32 [K,23,41], token_type_ids i32 [K,23],
    attention_mask bool [K,23], history_mask bool [K,1]; legal bool [K,9] (the peer view's legal mask); the key index
    ordinal i64 [K], decision_index i64 [K], world i32 [K] (block order = the constructor's world index); the recorded
    root_peer_logits f32 [K,9], opponent_requested bool [K] and self_check i8 [K] (1 equal, 0 not, -1 not checked); and
    header, a JSON string (schema PEER_INPUTS_SCHEMA, the tree guard, shard, the champion's identity, the slice rule, and
    each array's shape and dtype). Written with the summary (on interruption too), never over an existing file.
- Live join: the cell's telemetry/<ordinal>.jsonl row of the same decision_index (the actor's index): source,
  search.action_index / q_values / status / cause, greedy.action_index, request_kind, turn, played_action_index; its
  prefix_sha256 is held to the capture's. Parity over the decisions searched live (source 'search') and COMPLETE in the
  replay: action_equal, and q_equal (each q compared as a float after a JSON round trip of the replayed list; None only
  equals None). status_match: live searched vs replay COMPLETE (the client deadline and queue expiry have no counterpart here).
- Cells: --cells ORD..., or --sample N --sample-seed S: N of the arm's COMPLETE cells (results/<ordinal>.json: cell.arm,
  status) with an owner capture, stratified equally over (cell.seed % 2, cell.seat) (a remainder goes to the first strata in
  STRATA order), each stratum ranked by sha256 of (SAMPLE_DOMAIN, S, ordinal). --shard I --shards N keeps every N-th selected
  cell in ordinal order from the I-th. --max-decisions stops after that many sent decisions (smoke tests).
- Output (--out, a new directory): rows-<shard>.jsonl (one row per sent decision, whatever its status, written and fsynced as
  it goes), header-<shard>.json (tree guard, run header and identity check, warmup report, argv, cells, peer_inputs: the
  sidecar's path and schema, recharge_moves) and
  summary-<shard>.json (counts, parity, status mismatches, kept/legal disagreements, peer_inputs_check, timings; written on
  interruption too), peer-inputs-<shard>.npz (above);
  one progress JSON line per decision on stdout.
Bit-exactness: the live tree, inputs, seeds and warmup make the same worlds and programs; the device cannot be pinned here.
Run on the live servers' GPU model, image and environment (/out/env.sh: the same XLA flags), ideally on a copy of their
compilation cache so every program loads the live executables instead of re-autotuning; q_equal is the test.
Exit 0 when any decision was sent, 2 when none was.
"""
import collections
from contextlib import contextmanager
from dataclasses import dataclass
import dis
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0, str(ROOT))
from scripts import depthn_live_bench as bench   # noqa: E402
from gpu_engine import constructor_worlds as cw, depthn_live_search as live   # noqa: E402
from pokezero import depthn_effect_guard as actor, phase_d_owning_capture as owning   # noqa: E402   (the live guards' one module)

server = bench.server
ROW_SCHEMA = 'pokezero.depthn-rr-reprice.row.v1'
LIVE_COMMIT = '77235041b9f5623a62d7cb8ddfa884e72d76b4b2'   # the screen-r1 search servers' tree
LIVE_TREES = {'gpu_engine': '449cf36742de4998a490bad257332d997a20f306', 'src': 'f1f3d412e6130c88d9a7aa55e9dc0014cd02e59a',
    'scripts/depthn_live_server.py': 'd4dd5e8dc7c1c16f5c9b40e1a487c4d3d98e3d57', 'scripts/depthn_live_bench.py': '4ce26ff2dd1a870935505651906679c165a18042'}
# READY of the live servers (/tmp/v2e/eval-inputs/serve.sh): what depthn_live_server.header reports for them.
LIVE_IDENTITY = dict(constructor_seed=20261103, recipe_seed=20261103, session_cap=256, max_requests=1, saved_root_admission=False, refuse_classes=[],
    matmul_precision='highest', matmul_precision_pin='highest', jax_enable_x64=False, champion_precision='bf16',
    metadata_sha256='fad2db5adee11d6f9efb72a5967bde3e2b5c9970ac61b79d4af5e08bfbfb7052',
    vocabulary_sha256='9dad1fee6e294008d2157907acace39151577943c2d82b63ace4afc4dd85266f',
    champion_archive_sha256='e450515351bce1497ab50a712955fe17b761bbb5f424b23ec299e96cda5b165c',
    champion_checkpoint_sha256='0fd095923b4ac7e05d6e2b3ccab9c1e6869dff4893c2dae456caff10dce690be')
SAMPLE_DOMAIN = 'pokezero.depthn-rr-reprice.sample.v1'
STRATA = ((0, 0), (0, 1), (1, 0), (1, 1))   # (cell.seed % 2, cell.seat)
BENCH_ONLY = ('roster', 'producer_review', 'sessions', 'limit')   # the bench's roster flags have no meaning here
PEER_INPUTS_SCHEMA = 'pokezero.depthn-rr-reprice.peer-inputs.v1'
PEER_INPUT_KEYS = ('numeric_features', 'categorical_ids', 'token_type_ids', 'attention_mask', 'history_mask')   # ChampionEvaluator.view's keys
ENVIRONMENT = ('JAX_DEFAULT_MATMUL_PRECISION', 'JAX_ENABLE_X64', 'JAX_PLATFORMS', 'XLA_FLAGS', 'XLA_PYTHON_CLIENT_MEM_FRACTION', 'CUDA_VISIBLE_DEVICES',
    'TF_CPP_MIN_LOG_LEVEL')


# Command line ---------------------------------------------------------------------------------------------------------------------
def parser():
    p = bench.parser();p.prog = 'depthn_rr_reprice.py'
    for action in p._actions:
        if action.dest == 'roster':action.required = False
        if action.dest == 'warmup_requests':action.required = True   # the server's own warmup file
    p.add_argument('--run-directory', required=True, help='the live run directory (results/, telemetry/ and owning-wire/ per cell ordinal)')
    p.add_argument('--arm', choices=tuple(live.RUNGS), required=True, help='the search arm replayed (its cells; --rung must name it)')
    pick = p.add_mutually_exclusive_group(required=True)
    pick.add_argument('--cells', type=int, nargs='+', metavar='ORD', help='these cell ordinals')
    pick.add_argument('--sample', type=int, metavar='N', help='N of the arm\'s COMPLETE cells, stratified equally over (seed %% 2, seat)')
    p.add_argument('--sample-seed', type=int, help='the sample\'s seed (required with --sample)')
    p.add_argument('--shard', type=int, default=0);p.add_argument('--shards', type=int, default=1)
    p.add_argument('--max-decisions', type=int, help='stop after this many sent decisions (smoke tests)')
    p.add_argument('--accept-tree', metavar='COMMIT', help='replay against a modified tree: COMMIT must be HEAD and clean in the guarded paths '
        'and scripts/; its trees replace the live pins (module docstring). Parity against live is recorded but expected to fail')
    p.set_defaults(saved_root_admission=False, buckets=(1,))
    return p


def options(argv=None):
    p = parser();args = p.parse_args(argv);server.evaluators.check_arguments(p, args);server.check_precision_environment(p, args)
    if not args.champion_archive:p.error('--champion-archive with its two sha256 flags is required: the champion prices every root and leaf')
    for name in BENCH_ONLY:
        if getattr(args, name, None):p.error(f'--{name.replace("_", "-")} is the bench\'s; cells come from --cells or --sample')
    if args.buckets != (1,):p.error('the live servers priced at bucket 1 only')
    if args.max_requests not in (None, 1):p.error('the live servers took one request per batch (--max-requests 1)')
    if args.saved_root_admission:p.error('the live servers kept per-battle sessions (no --saved-root-admission)')
    if args.rung != args.arm:p.error(f'--rung must be the replayed arm ({args.arm})')
    if args.sample is not None and (args.sample < 1 or args.sample_seed is None):p.error('--sample N (>= 1) needs --sample-seed')
    if args.sample is None and args.sample_seed is not None:p.error('--sample-seed goes with --sample')
    if args.shards < 1 or not 0 <= args.shard < args.shards:p.error('--shard I --shards N with 0 <= I < N required')
    if args.max_decisions is not None and args.max_decisions < 1:p.error('--max-decisions must be >= 1')
    args.port = args.port or 0;args.max_requests = 1
    return args


# Guards -------------------------------------------------------------------------------------------------------------------------
def git(root, *arguments):
    return subprocess.check_output(['git', '-C', str(root), *arguments], text=True, stderr=subprocess.DEVNULL).strip()


MODIFIED_TREE = 'replay against a modified tree: parity against the live run is recorded but expected to fail'
ACCEPT_PATHS = (*LIVE_TREES, 'scripts')   # what --accept-tree holds clean: the guarded paths and every script the replay may run


def tree_guard(root=ROOT, *, tree_commit=None, accept_tree=None, run=git):
    """dict(head, live_commit, live_commit_present, trees) when HEAD's guarded paths are the live tree's (module docstring);
    with accept_tree (a commit that must be HEAD, clean in ACCEPT_PATHS), its trees instead, plus accepted_tree, modified_tree,
    live_trees, differs_from_live and the MODIFIED_TREE note. ValueError otherwise."""
    try:head = run(root, 'rev-parse', 'HEAD')
    except (OSError, subprocess.CalledProcessError) as error:raise ValueError(f'tree guard: {root} must be a git checkout ({error})') from None
    if tree_commit is not None and tree_commit != head:raise ValueError(f'tree guard: --tree-commit {tree_commit} is not HEAD {head}')
    if accept_tree is not None:return accepted_tree_guard(root, head, accept_tree, run=run)
    trees, present = {}, True
    for path, pinned in LIVE_TREES.items():
        try:theirs = run(root, 'rev-parse', f'{LIVE_COMMIT}:{path}')
        except subprocess.CalledProcessError:theirs, present = None, False   # a shallow clone without the live commit: the pin alone
        if theirs is not None and theirs != pinned:raise ValueError(f'tree guard: the pin of {path} ({pinned}) is not {LIVE_COMMIT[:8]}\'s ({theirs})')
        try:mine = run(root, 'rev-parse', f'HEAD:{path}')
        except subprocess.CalledProcessError:mine = None
        if mine != pinned:raise ValueError(f'tree guard: HEAD:{path} is {mine}, not the live tree {LIVE_COMMIT[:8]}\'s {pinned}')
        trees[path] = mine
    changed = run(root, 'status', '--porcelain', '--untracked-files=all', '--', *LIVE_TREES).splitlines()
    odd = [x for x in changed if not x.startswith('??') or x.endswith('.py')]
    if odd:raise ValueError('tree guard: the live paths differ from HEAD in the working tree: ' + '; '.join(odd[:5]))
    return dict(head=head, live_commit=LIVE_COMMIT, live_commit_present=present, trees=trees)


def accepted_tree_guard(root, head, commit, *, run=git):
    """tree_guard's --accept-tree branch (module docstring): ValueError unless `commit` is HEAD and the ACCEPT_PATHS are clean."""
    if type(commit) is not str or not commit.strip():raise ValueError('tree guard: --accept-tree needs a commit')
    try:resolved = run(root, 'rev-parse', '--verify', f'{commit}^{{commit}}')
    except subprocess.CalledProcessError:raise ValueError(f'tree guard: --accept-tree {commit} is not a commit of {root}') from None
    if resolved != head:raise ValueError(f'tree guard: --accept-tree {commit} ({resolved}) is not HEAD {head}')
    trees = {}
    for path in LIVE_TREES:
        try:trees[path] = run(root, 'rev-parse', f'{resolved}:{path}')
        except subprocess.CalledProcessError:raise ValueError(f'tree guard: --accept-tree {resolved[:8]} has no {path}') from None
    changed = run(root, 'status', '--porcelain', '--untracked-files=all', '--', *ACCEPT_PATHS).splitlines()
    odd = [x for x in changed if not x.startswith('??') or x.endswith('.py')]
    if odd:raise ValueError('tree guard: --accept-tree needs a clean HEAD; the working tree differs: ' + '; '.join(odd[:5]))
    return dict(head=head, live_commit=LIVE_COMMIT, live_commit_present=None, trees=trees, accepted_tree=resolved, modified_tree=True,
        live_trees=dict(LIVE_TREES), differs_from_live=[p for p in LIVE_TREES if trees[p] != LIVE_TREES[p]], note=MODIFIED_TREE)


def check_call_sites(function=None):
    """True when `function` (DepthNLiveSearch._search) reads `decide` as a global of depthn_live_search and `prepare_roots` as an
    attribute of the global `cw` at call time, so the wrappers take effect; RuntimeError otherwise."""
    function = live.DepthNLiveSearch._search if function is None else function
    found = {'decide': False, 'prepare_roots': False};previous = None
    for instruction in dis.get_instructions(function):
        if instruction.opname == 'LOAD_GLOBAL' and instruction.argval == 'decide':found['decide'] = True
        if (instruction.opname in ('LOAD_ATTR', 'LOAD_METHOD') and instruction.argval == 'prepare_roots' and previous is not None
                and previous.opname == 'LOAD_GLOBAL' and previous.argval == 'cw'):found['prepare_roots'] = True
        previous = instruction
    if not all(found.values()) or function.__globals__ is not vars(live) or live.cw is not cw:
        raise RuntimeError(f'the search does not look decide / cw.prepare_roots up through their modules at call time: {found}')
    return True


def identity_problems(header, arm):
    """The run header's differences from the live READY identity (LIVE_IDENTITY, the arm's rung, bucket 1), as strings."""
    want = dict(LIVE_IDENTITY, rung=arm)
    out = [f'{k}: {header.get(k)!r} != {v!r}' for k, v in want.items() if header.get(k) != v]
    pricer = header.get('pricer') or {}
    if pricer.get('buckets') != [1]:out.append(f"pricer buckets: {pricer.get('buckets')!r} != [1]")
    if pricer.get('W') != live.RUNGS[arm]:out.append(f"pricer W: {pricer.get('W')!r} != {live.RUNGS[arm]}")
    return out


# Cells ------------------------------------------------------------------------------------------------------------------------------
def read_json(path):
    return json.loads(Path(path).read_bytes())


def cell_of(run, ordinal):
    """dict(ordinal, arm, seed, seat, pair_ordinal, status) of one cell, from results/<ordinal>.json."""
    result = read_json(Path(run) / 'results' / f'{ordinal}.json');cell = result.get('cell') or {}
    if cell.get('ordinal') != ordinal:raise ValueError(f'results/{ordinal}.json is cell {cell.get("ordinal")!r}')
    return dict(ordinal=ordinal, arm=cell.get('arm'), seed=cell.get('seed'), seat=cell.get('seat'), pair_ordinal=cell.get('pair_ordinal'),
        status=result.get('status'))


def arm_cells(run, arm):
    """Every cell of `arm` with a result, in ordinal order."""
    ordinals = sorted(int(p.stem) for p in (Path(run) / 'results').glob('*.json') if p.stem.isdigit())
    return [c for c in (cell_of(run, o) for o in ordinals) if c['arm'] == arm]


def sample_rank(seed, ordinal):
    return hashlib.sha256(json.dumps([SAMPLE_DOMAIN, int(seed), int(ordinal)], separators=(',', ':')).encode()).hexdigest()


def sample_cells(cells, n, seed):
    """n cells stratified equally over STRATA (the remainder to the first strata), each stratum's cells by sample_rank; in
    ordinal order. ValueError when a stratum has too few."""
    strata = {s: [] for s in STRATA}
    for c in cells:
        key = (c['seed'] % 2, c['seat'])
        if key not in strata:raise ValueError(f'cell {c["ordinal"]}: seat {c["seat"]!r} outside (0, 1)')
        strata[key].append(c)
    out = []
    for i, key in enumerate(STRATA):
        want = n // len(STRATA) + (i < n % len(STRATA))
        if len(strata[key]) < want:raise ValueError(f'stratum (seed % 2, seat) = {key} has {len(strata[key])} cells, {want} required')
        out += sorted(strata[key], key=lambda c: sample_rank(seed, c['ordinal']))[:want]
    return sorted(out, key=lambda c: c['ordinal'])


def shard_cells(cells, shard, shards):
    return sorted(cells, key=lambda c: c['ordinal'])[shard::shards]


def select_cells(args):
    """The cells this shard replays (module docstring): --cells must be the arm's; --sample draws from the arm's COMPLETE
    cells with an owner capture."""
    run = Path(args.run_directory)
    if args.cells:
        cells = [cell_of(run, o) for o in dict.fromkeys(args.cells)]
        odd = [c['ordinal'] for c in cells if c['arm'] != args.arm]
        if odd:raise ValueError(f'cells {odd} are not {args.arm} cells')
    else:
        pool = [c for c in arm_cells(run, args.arm) if c['status'] == 'COMPLETE' and (run / 'owning-wire' / f'{c["ordinal"]}.jsonl').exists()]
        cells = sample_cells(pool, args.sample, args.sample_seed)
    missing = [c['ordinal'] for c in cells if not (run / 'owning-wire' / f'{c["ordinal"]}.jsonl').exists()]
    if missing:raise ValueError(f'cells {missing} have no owner capture')
    return shard_cells(cells, args.shard, args.shards)


def sent(request):
    """The live actor's (and the bench session mode's) rule: a single-legal decision is never sent."""
    return sum(bench.legal_mask(request) or ()) != 1


# Live join --------------------------------------------------------------------------------------------------------------------------
def live_index(path):
    """{decision_index: telemetry row} of one cell's actor telemetry ({} without the file), and the duplicated indices."""
    rows, duplicated = {}, []
    if not Path(path).exists():return rows, duplicated
    for raw in Path(path).read_bytes().splitlines():
        if not raw.strip():continue
        row = json.loads(raw);index = row.get('decision_index')
        if index in rows:duplicated.append(index)
        rows[index] = row
    return rows, duplicated


def live_fields(row):
    """What the replay compares and records of a live telemetry row (None without one)."""
    if row is None:return None
    search = row.get('search') if isinstance(row.get('search'), dict) else {};greedy = row.get('greedy') if isinstance(row.get('greedy'), dict) else {}
    return dict(source=row.get('source'), cause=row.get('cause'), request_kind=row.get('request_kind'), turn=row.get('turn'),
        played_action_index=row.get('played_action_index'), greedy_action_index=greedy.get('action_index'), prefix_sha256=row.get('prefix_sha256'),
        search_status=search.get('status'), search_cause=search.get('cause'), search_action_index=search.get('action_index'),
        search_q_values=search.get('q_values'), search_refusal=None if search.get('refusal') is None else str(search['refusal'])[:500],
        client_refused=search.get('client_refused'), server_id=search.get('server_id'), tree_commit=search.get('tree_commit'))


def _same_q(x, y):
    if x is None or y is None:return x is None and y is None
    if type(x) not in (int, float) or type(y) not in (int, float):return False   # never a bool
    return float(x) == float(y)


def q_equal(replayed, live_q):
    """Replayed q_values against the live ones, exactly as floats after a JSON round trip of the replayed list; None only
    equals None (per lane and for the whole list)."""
    a = None if replayed is None else json.loads(json.dumps(list(replayed)))
    if a is None or live_q is None:return a is None and live_q is None
    return isinstance(live_q, list) and len(a) == len(live_q) and all(_same_q(x, y) for x, y in zip(a, live_q))


def parity(response, found):
    """live_searched / replay_complete / status_match, and action_equal / q_equal where both searched (None elsewhere)."""
    complete = response.get('status') == 'COMPLETE'
    if found is None:return dict(live_found=False, live_searched=None, replay_complete=complete, status_match=None, action_equal=None, q_equal=None)
    searched = found['source'] == 'search';both = searched and complete;action = response.get('action_index')
    return dict(live_found=True, live_searched=searched, replay_complete=complete, status_match=searched == complete,
        action_equal=None if not both else type(action) is int and type(found['search_action_index']) is int and action == found['search_action_index'],
        q_equal=None if not both else q_equal(response.get('q_values'), found['search_q_values']))


# Capture --------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Decided:
    blocks:object
    root_logits:object
    root_legal:object


class Capture:
    """References to every decide call's arguments and every prepare_roots result while installed() is active."""
    def __init__(self):
        self.decided, self.prepared = [], []

    @contextmanager
    def installed(self):
        decide, prepare_roots = live.decide, cw.prepare_roots

        def decide_wrapper(blocks, root_logits, root_legal):
            self.decided.append(Decided(blocks, root_logits, root_legal))
            return decide(blocks, root_logits, root_legal)

        def prepare_wrapper(*arguments, **keywords):
            out = prepare_roots(*arguments, **keywords)
            self.prepared.append(out)
            return out

        live.decide, cw.prepare_roots = decide_wrapper, prepare_wrapper
        try:yield self
        finally:live.decide, cw.prepare_roots = decide, prepare_roots


def prices_fields(world):
    """A WorldPrices (or None: a world without a price surface) as JSON-ready fields."""
    if world is None:return dict(priced=False, opponent_requested=None, kept=None, root_peer_logits=None, grid=None)
    return dict(priced=True, opponent_requested=bool(world.opponent_requested), kept=[bool(x) for x in world.opponent_legal],
        root_peer_logits=[float(x) for x in world.opponent_logits],
        grid=[[None if v is None else float(v) for v in row] for row in world.joint_values])


def opponent_facts(state, owner, vocabulary, public=None):
    """The opponent's (1 - owner) mapping facts in one world's (host) engine state, through the constructor's engine-id tables
    (metadata['vocabulary']: normalize_id for species, abilities and items, phase_d_live_worlds._source_move for moves), its
    party in slot order with each member's set, and (from `public`, the completed carry, when it has own_order) the peer's
    request order (module docstring)."""
    import numpy as np
    from pokezero.dex import normalize_id
    from gpu_engine.phase_d_live_worlds import _source_move
    foe = 1 - owner;at = lambda name: np.asarray(getattr(state, name))[0, foe]
    species_ids, move_ids, hp, max_hp, charge = at('species'), at('moves'), at('hp'), at('max_hp'), at('charge_move');active = int(at('active'))
    abilities, base_abilities, items = at('ability'), at('base_ability'), at('item')
    name = lambda table, i: None if i <= 0 or i >= len(vocabulary.get(table, ())) else normalize_id(vocabulary[table][i]) or None
    species = lambda i: None if i <= 0 else normalize_id(vocabulary['species'][i])
    move = lambda i: None if i <= 0 else _source_move(vocabulary['moves'][i])
    hp_types, hp_powers = at('hidden_power_type'), at('hidden_power_power')

    def typed(i, slot):   # a party move id as the request spells it: Hidden Power keeps its type (module docstring)
        if i <= 0:return None
        value = normalize_id(vocabulary['moves'][i])
        if not value.startswith('hiddenpower'):return move(i)
        if value != 'hiddenpower':return value
        kind = name('types', int(hp_types[slot]));power = int(hp_powers[slot])
        return value if kind is None else f'hiddenpower{kind}' + (str(power) if power > 0 else '')
    order = getattr(public, 'own_order', None)
    order = None if order is None else [int(x) for x in np.asarray(order).reshape(-1, 2, 6)[0, foe]]
    party = [dict(slot=s, order=None if order is None or s not in order else order.index(s), species=species(int(species_ids[s])),
        species_engine_id=int(species_ids[s]), moves=[typed(int(m), s) for m in move_ids[s]], move_engine_ids=[int(m) for m in move_ids[s]],
        ability=name('abilities', int(abilities[s])), base_ability=name('abilities', int(base_abilities[s])), item=name('items', int(items[s])),
        fainted=bool(hp[s] <= 0), hp=int(hp[s]), max_hp=int(max_hp[s])) for s in range(len(species_ids))]
    inside = 0 <= active < len(party);moves = [int(m) for m in move_ids[active]] if inside else []
    return dict(side=f'p{foe + 1}', active_slot=active, active_species=party[active]['species'] if inside else None,
        active_moves=[move(m) for m in moves], active_move_engine_ids=moves, charge_move=move(int(charge[active])) if inside else None, party=party,
        request_order=order)


class PeerInputs:
    """Each decided root's depth-0 peer inputs (module docstring): captured per world from the views, checked against the
    server's champion on the stack the pricer evaluates, and kept (the champion adapter's view slice) for the shard's
    sidecar."""
    def __init__(self, executor, champion, params):
        from gpu_engine import device_depth2_pricer as depth2
        self.views = depth2._make_views(executor);self.champion, self.params, self.program = champion, params, None
        self.columns = {k: [] for k in PEER_INPUT_KEYS + ('legal', 'ordinal', 'decision_index', 'world', 'root_peer_logits', 'opponent_requested',
            'self_check')}

    def capture(self, state, public, owner):
        """(the V2 encoder inputs, host copies [1, 1, 29, F], the peer view's legal mask [9]) of one world."""
        import numpy as np
        _, peer, _, inputs = self.views(state, public, owner)
        legal = np.asarray(peer.legal_action_mask, bool).reshape(-1)
        if legal.shape != (9,):raise ValueError(f'one world\'s peer legal mask required, got {legal.shape}')
        return {k: np.asarray(v) for k, v in inputs.items()}, legal

    def forward(self, inputs):
        """The champion's policy logits [N, 9] f32 on stacked V2 inputs: its own adapter under jax.jit."""
        import jax
        import numpy as np
        if self.program is None:self.program = jax.jit(lambda p, x: self.champion(p, x))
        return np.asarray(jax.device_get(self.program(self.params, inputs)).policy_logits, np.float32)

    def annotate(self, key, worlds, captured):
        """Self-checks and keeps one decided root's worlds (block order). `key` (ordinal, decision_index); `captured` per world
        capture()'s pair or None (no state, or its capture failed). Sets peer_inputs_index / peer_self_check /
        peer_self_check_max_abs_diff on each world row."""
        import numpy as np
        first = next((x for x in captured if x is not None), None)
        if first is None:raise ValueError('no constructed world')
        batch = {k: np.concatenate([(x if x is not None else first)[0][k] for x in captured]) for k in first[0]}
        logits = self.forward(batch);view = {k: np.asarray(v) for k, v in self.champion.view(batch).items()}
        fields, kept = {}, []   # applied only once every world is done, so a failure leaves the sidecar's columns aligned
        for i, row in enumerate(worlds[:len(captured)]):
            checkable = row.get('priced') is True and row.get('opponent_requested') is True;checked = checkable and captured[i] is not None
            recorded = np.asarray(row['root_peer_logits'], np.float32) if checked else None
            # A checkable world whose inputs were not captured (its capture raised) fails: it is never skipped.
            same = (False if checkable else None) if not checked else bool(np.array_equal(logits[i], recorded))
            fields[i] = dict(peer_self_check=same, peer_inputs_index=None,
                peer_self_check_max_abs_diff=None if not checked else float(np.max(np.abs(logits[i].astype(np.float64) - recorded))))
            if captured[i] is None:continue
            kept.append((i, dict({k: (view[k][i, 0] if k != 'history_mask' else view[k][i]).copy() for k in PEER_INPUT_KEYS}, legal=captured[i][1],
                ordinal=key[0], decision_index=key[1], world=i, root_peer_logits=np.asarray(row.get('root_peer_logits') or [0.] * 9, np.float32),
                opponent_requested=bool(row.get('opponent_requested')), self_check=-1 if same is None else int(same))))
        for i, entry in kept:
            fields[i]['peer_inputs_index'] = len(self.columns['world'])
            for k, v in entry.items():self.columns[k].append(v)
        for i, value in fields.items():worlds[i].update(value)

    DTYPES = dict(numeric_features='float32', categorical_ids='int32', token_type_ids='int32', attention_mask='bool', history_mask='bool', legal='bool',
        ordinal='int64', decision_index='int64', world='int32', root_peer_logits='float32', opponent_requested='bool', self_check='int8')
    SHAPES = dict(numeric_features=(23, 132), categorical_ids=(23, 41), token_type_ids=(23,), attention_mask=(23,), history_mask=(1,), legal=(9,),
        root_peer_logits=(9,))

    def arrays(self):
        """The sidecar's arrays (module docstring), stacked over the K kept worlds."""
        import numpy as np
        out = {}
        for k, dtype in self.DTYPES.items():
            values = self.columns[k];shape = (len(values),) + self.SHAPES.get(k, ())
            out[k] = np.asarray(values, dtype).reshape(shape) if values else np.zeros(shape, dtype)
        return out

    def write(self, path, header):
        """peer-inputs-<shard>.npz: the arrays and `header` (with each array's shape and dtype) as a JSON string; never over a file."""
        import numpy as np
        arrays = self.arrays()
        document = dict(header, schema=PEER_INPUTS_SCHEMA, worlds=len(self.columns['world']),
            arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in arrays.items()})
        with Path(path).open('xb') as stream:
            np.savez_compressed(stream, header=np.asarray(json.dumps(document, sort_keys=True, default=repr)), **arrays)
            stream.flush();os.fsync(stream.fileno())
        return document


class PeerLanes:
    """The opponent's encoder lanes in one world: device_depth2_pricer._make_views(executor)'s second output (the peer
    observation), its executor programs jitted once by the executor itself, plus usable_moves (jitted once) on its side."""
    def __init__(self, executor):
        import jax
        from gpu_engine import device_depth2_pricer as depth2
        from gpu_engine.state import usable_moves
        self.views = depth2._make_views(executor);self.usable = jax.jit(usable_moves)

    def __call__(self, state, public, owner):
        import numpy as np
        _, peer, _, _ = self.views(state, public, owner)
        flat = lambda x: np.asarray(x).reshape(-1)
        out = dict(legal=[bool(x) for x in flat(peer.legal_action_mask)], move_targets=[int(x) for x in flat(peer.move_targets)],
            switch_targets=[int(x) for x in flat(peer.switch_targets)], error=int(flat(peer.error)[0]),
            usable=[bool(x) for x in np.asarray(self.usable(state))[0, 1 - owner]])
        if (len(out['legal']), len(out['move_targets']), len(out['switch_targets'])) != (9, 4, 5):raise ValueError('one world\'s peer view required')
        return out


def lane_labels(facts, peer):
    """What each legal opponent lane plays: move:<id> (lanes 0-3 through move_targets; lane 0 is Struggle when no move is usable
    and the charge move while one charges, as the encoder fills them) or switch:<species> (lanes 4-8 through switch_targets);
    None for an illegal lane."""
    legal, out = peer['legal'], []
    struggle = not any(peer['usable']);charged = facts['charge_move'] is not None
    for b in range(4):
        if not legal[b]:out.append(None)
        elif b == 0 and struggle:out.append('move:struggle')
        elif b == 0 and charged:out.append(f'move:{facts["charge_move"]}')
        else:
            slot = peer['move_targets'][b];moves = facts['active_moves']
            out.append(f'move:{moves[slot]}' if 0 <= slot < len(moves) else None)
    for i, slot in enumerate(peer['switch_targets']):
        party = facts['party']
        out.append(None if not legal[4 + i] else f'switch:{party[slot]["species"]}' if 0 <= slot < len(party) else None)
    return out


def describe(capture, peer_lanes, vocabulary, peer_inputs=None, key=None):
    """dict(root, prepared, worlds, notes, kept_disagreements) of one replayed decision's capture (module docstring); with
    `peer_inputs` (PeerInputs), each decided world's depth-0 peer inputs are self-checked and kept under `key`
    (ordinal, decision_index)."""
    import jax
    notes = [];items = [item for call in capture.prepared for item in call]
    if len(capture.prepared) > 1 or len(items) > 1:notes.append(f'{len(capture.prepared)} prepare_roots calls, {len(items)} roots')
    if len(capture.decided) > 1:notes.append(f'{len(capture.decided)} decide calls')
    item = items[0] if items else None;decided = capture.decided[0] if capture.decided else None
    prepared = worlds = None;captured = []
    if type(item) is cw.RootRefusal:prepared = dict(kind='RootRefusal', refusal_kind=item.kind, refusal_detail=str(item.detail)[:500], seconds=item.seconds)
    elif type(item) is cw.Prepared:
        made = item.made;owner = int(item.detached.address.owner)
        prepared = dict(kind='Prepared', owner=owner, legal=[bool(x) for x in item.legal], seconds=item.seconds, requested_worlds=made.requested_worlds,
            fully_constructed=bool(made.fully_constructed))
        worlds = []
        for i, w in enumerate(made.worlds):
            row = dict(world=i, index=w.index, seed=made.seeds[i] if i < len(made.seeds) else None, kernel_seed=w.kernel_seed, refusals=list(w.refusals),
                constructed=w.state is not None, opponent=None, peer=None)
            captured.append(None)
            if w.state is not None:
                try:
                    state, public = jax.device_get((w.state, w.public))   # as constructor_worlds.build reads a world
                    row['opponent'] = opponent_facts(state, owner, vocabulary, public);row['peer'] = peer_lanes(state, public, owner)
                    row['peer']['lanes'] = lane_labels(row['opponent'], row['peer'])
                    if peer_inputs is not None and decided is not None:captured[-1] = peer_inputs.capture(state, public, owner)   # decided roots only
                except Exception as error:   # a capture problem never stops the replay
                    notes.append(f'world {i}: {type(error).__name__}: {error}')
            worlds.append(row)
    elif item is not None:notes.append(f'prepare_roots returned {type(item).__name__}')
    root = disagreements = None
    if decided is not None:
        root = dict(root_logits=[float(x) for x in decided.root_logits], root_legal=[bool(x) for x in decided.root_legal])
        prices = [w for pair in decided.blocks for w in pair];worlds = worlds if worlds is not None else [dict(world=i) for i in range(len(prices))]
        if len(prices) != len(worlds):notes.append(f'{len(prices)} priced worlds for {len(worlds)} prepared')
        disagreements = 0
        for i, world in enumerate(prices):
            if i >= len(worlds):worlds.append(dict(world=i))
            row = worlds[i];row.update(prices_fields(world))
            if 'index' in row and row['index'] != i:notes.append(f'block-order world {i} is constructor world {row["index"]}')
            peer = row.get('peer')
            if world is None or peer is None:row['kept_matches_peer_legal'] = row['requested_matches_peer'] = None;continue
            row['requested_matches_peer'] = row['opponent_requested'] == any(peer['legal'])
            row['kept_matches_peer_legal'] = row['kept'] == peer['legal'] if row['opponent_requested'] else None
            disagreements += row['requested_matches_peer'] is False or row['kept_matches_peer_legal'] is False
        if peer_inputs is not None and captured:
            try:peer_inputs.annotate(key, worlds, captured)
            except Exception as error:   # recorded; the summary's peer_inputs_check fails on this root
                notes.append(f'peer inputs: {type(error).__name__}: {error}')
                for row in worlds:row['peer_self_check'] = False;row.setdefault('peer_inputs_index', None)
    return dict(root=root, prepared=prepared, worlds=worlds or [], notes=notes, kept_disagreements=disagreements)


REPLAY_FIELDS = ('q_values', 'completed_q', 'policy_target', 'root_value', 'action_mean_q', 'action_policy_target', 'reduction_note', 'block_causes',
    'world_refusals', 'bucket', 'batch_size', 'tree_commit', 'root_view_source', 'b2_view')


def decision_row(cell, row, request, response, wall, found, detail, *, shard, capture_seconds, live_problem=None):
    """One rows-<shard>.jsonl row: the address, the replay (bench.record plus the response's values), the live join, parity and
    the capture."""
    address = tuple(row[k] for k in ('battle_id', 'owner', 'stream_id', 'decision_index', 'generation', 'prefix_sha256'))
    replay = bench.record(None, cell['ordinal'], address, response, wall, request=request, turn=bench.turn_of(request['lines']))
    replay.update({k: response.get(k) for k in REPLAY_FIELDS}, request_id=response.get('request_id'))
    for key in ('roster', 'ordinal', 'address'):replay.pop(key, None)
    decided = detail['root'] is not None;moves = actor.own_moves(request.get('own_request_line'))   # guard fields (module docstring)
    prepared = detail['prepared'] or {}
    refusal = (None if response.get('status') == 'COMPLETE' else prepared.get('refusal_kind') if prepared.get('kind') == 'RootRefusal'
        else response.get('cause'))
    root = dict(decided=decided, **(detail['root'] if decided else dict(root_logits=None, root_legal=None)),
        own_moves=None if moves is None else list(moves),
        own_recharge_lanes=None if moves is None else [actor.recharge_lane(moves, lane) for lane in range(4)],
        opponent_recharge=opponent_recharge(request), refusal_cause=refusal)
    count = row.get('request_count')
    if count is None:count = sum(x.startswith('|request|') for x in request['lines'])
    out = dict(schema=ROW_SCHEMA, shard=shard, ordinal=cell['ordinal'], seat=cell.get('seat'), seed=cell.get('seed'), battle_id=row['battle_id'],
        owner=row['owner'], stream_id=row['stream_id'], decision_index=row['decision_index'], generation=row['generation'],
        prefix_sha256=row['prefix_sha256'], request_count=count, capture_turn=row.get('turn'), replay=replay, live=found,
        live_problem=live_problem, parity=parity(response, found), capture_seconds=capture_seconds, effect_guard=response.get('effect_guard'),
        **dict(detail, root=root))
    # A COMPLETE decision went through decide; a root that reached prepare (its prepare_seconds) through prepare_roots.
    checks = [w['peer_self_check'] for w in out['worlds'] if 'peer_self_check' in w]
    out['peer_inputs_check'] = None if not checks else dict(checked=sum(c is not None for c in checks), equal=sum(c is True for c in checks),
        failed_worlds=[w['world'] for w in out['worlds'] if w.get('peer_self_check') is False],
        written=sum(w.get('peer_inputs_index') is not None for w in out['worlds']))
    out['capture_ok'] = ((response.get('status') != 'COMPLETE' or decided)
        and ((response.get('timings') or {}).get('prepare_seconds') is None or detail['prepared'] is not None))
    return out


def opponent_recharge(request):
    """The peer must recharge at this root: constructor_worlds.line_facts' recharge (B2's rule) on the request's lines; None
    when the owner is not p1 or p2."""
    owner = {'p1': 0, 'p2': 1}.get(request.get('owner'))
    return None if owner is None else bool(cw.line_facts(request['lines'], owner).recharge[1 - owner])


BULK = ('worlds', 'root', 'prepared', 'effect_guard')   # per-world grids and facts: written, not kept for the summary


def encode_row(row):
    try:return json.dumps(row, sort_keys=True, allow_nan=False, default=repr)
    except ValueError:return json.dumps(dict(row, nonfinite=True), sort_keys=True, default=repr)


# Summary -------------------------------------------------------------------------------------------------------------------------
def peer_inputs_summary(rows):
    """The summary's peer_inputs_check (module docstring): written, checked and equal worlds over every decided root, the
    failures, and passed (at least one world checked, every checked world equal, and every decided root checked)."""
    decided = [r for r in rows if r['replay']['status'] == 'COMPLETE']
    checked = [r for r in decided if r.get('peer_inputs_check') and r['peer_inputs_check']['checked'] > 0]   # a root with no checked world is not
    address = lambda r: dict(ordinal=r['ordinal'], decision_index=r['decision_index'])
    worlds = sum(r['peer_inputs_check']['checked'] for r in checked);equal = sum(r['peer_inputs_check']['equal'] for r in checked)
    failures = [dict(address(r), worlds=r['peer_inputs_check']['failed_worlds']) for r in checked if r['peer_inputs_check']['failed_worlds']]
    seen = {id(r) for r in checked};unchecked = [address(r) for r in decided if id(r) not in seen]
    return dict(decided_roots=len(decided), roots_checked=len(checked),
        worlds_written=sum(r['peer_inputs_check']['written'] for r in decided if r.get('peer_inputs_check')),
        worlds_checked=worlds, worlds_equal=equal, failures=failures, unchecked_roots=unchecked,
        passed=worlds > 0 and equal == worlds and not failures and not unchecked)


def summarize(rows, *, shard, shards, cells, extra=None):
    """The shard's summary over its decision rows (module docstring)."""
    both = [r for r in rows if r['parity']['live_searched'] and r['parity']['replay_complete']]
    address = lambda r: dict(ordinal=r['ordinal'], decision_index=r['decision_index'])
    mismatched = [dict(address(r), live_source=r['live']['source'], live_cause=r['live']['cause'], live_status=r['live']['search_status'],
        replay_status=r['replay']['status'], replay_cause=r['replay']['cause'], replay_refusal=r['replay']['refusal']) for r in rows if r['parity']['status_match'] is False]
    return dict(shard=shard, shards=shards, cells=[c['ordinal'] for c in cells], decisions_sent=len(rows),
        complete=sum(r['replay']['status'] == 'COMPLETE' for r in rows), live_found=sum(r['parity']['live_found'] for r in rows),
        live_searched=sum(bool(r['parity']['live_searched']) for r in rows), statuses=dict(collections.Counter(r['replay']['status'] for r in rows)),
        causes=dict(collections.Counter(r['replay']['cause'] for r in rows if r['replay']['cause'])),
        parity=dict(both=len(both), action_equal=sum(bool(r['parity']['action_equal']) for r in both), q_equal=sum(bool(r['parity']['q_equal']) for r in both),
            action_mismatches=[address(r) for r in both if not r['parity']['action_equal']], q_mismatches=[address(r) for r in both if not r['parity']['q_equal']]),
        status_mismatches=mismatched, kept_legal_disagreements=sum(r['kept_disagreements'] or 0 for r in rows),
        live_problems=[dict(address(r), problem=r['live_problem']) for r in rows if r['live_problem']],
        capture_anomalies=[dict(address(r), notes=r['notes'], capture_ok=r['capture_ok']) for r in rows if r['notes'] or not r['capture_ok']],
        live_tree_commits=dict(collections.Counter(r['live']['tree_commit'] for r in rows if r['live'] and r['live']['tree_commit'])),
        wall=bench.percentiles(r['replay']['wall_seconds'] for r in rows), capture=bench.percentiles(r['capture_seconds'] for r in rows),
        peer_inputs_check=peer_inputs_summary(rows), **(extra or {}))


# Main ------------------------------------------------------------------------------------------------------------------------------
def write_json(path, value):
    with Path(path).open('x') as stream:stream.write(json.dumps(value, indent=1, sort_keys=True, default=repr) + '\n');stream.flush();os.fsync(stream.fileno())


def main(argv=None):
    args = options(argv);argv = list(sys.argv[1:] if argv is None else argv)
    try:guard = tree_guard(tree_commit=args.tree_commit, accept_tree=args.accept_tree);check_call_sites();cells = select_cells(args)
    except (ValueError, RuntimeError, OSError) as error:raise SystemExit(f'depthn_rr_reprice: {error}')
    if not cells:raise SystemExit('depthn_rr_reprice: no cell in this shard')
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False);shard = args.shard;run = Path(args.run_directory)
    saved = server.read_requests(args.warmup_requests)   # the server's own warmup file, read as the server reads it
    clock = time.perf_counter();search = server.build(args);build_seconds = time.perf_counter() - clock
    header = server.header(args, search);problems = identity_problems(header, args.arm)
    if problems:raise SystemExit('depthn_rr_reprice: the built search is not the live servers\': ' + '; '.join(problems))
    clock = time.perf_counter();warm = search.warmup(saved);warmup_seconds = time.perf_counter() - clock
    import jax
    pricer = search.pricers[search.buckets[0]];vocabulary = search.constructor.metadata['vocabulary'];peer_lanes = PeerLanes(pricer.executor)
    peer_inputs = PeerInputs(pricer.executor, pricer.forward, search.params[0])   # build's opponent actor 0: the champion
    sidecar = dict(path=f'peer-inputs-{shard}.npz', tree_guard=guard, shard=shard, champion_identity=list(pricer.forward.identity),
        slice=getattr(server.evaluators, 'SLICE_RULE', None), key=['ordinal', 'decision_index', 'world'])
    write_json(out / f'header-{shard}.json', dict(argv=argv, tree_guard=guard, run_header=header, live_identity=LIVE_IDENTITY, warmup=warm,
        peer_inputs=dict(path=sidecar['path'], schema=PEER_INPUTS_SCHEMA), recharge_moves=sorted(actor.RECHARGE_MOVES),
        warmup_requests=len(saved), build_seconds=build_seconds, warmup_seconds=warmup_seconds, cells=cells, run_directory=str(run),
        arm=args.arm, shard=shard, shards=args.shards, max_decisions=args.max_decisions, sample=args.sample, sample_seed=args.sample_seed,
        jax_version=jax.__version__, devices=[f'{d.platform}:{d.device_kind}' for d in jax.devices()], python=sys.version,
        environment={k: os.environ.get(k) for k in ENVIRONMENT}, pid=os.getpid(), **({'replay_against': MODIFIED_TREE} if guard.get('modified_tree') else {})))
    rows, skipped, skip_disagreements, cell_errors, done, finished = [], 0, [], [], False, False
    started = time.perf_counter()
    try:
        with (out / f'rows-{shard}.jsonl').open('x') as stream:
            for cell in cells:
                if done:break
                ordinal = cell['ordinal']
                try:
                    captured = list(owning.read_capture(run / 'owning-wire' / f'{ordinal}.jsonl'))
                    telemetry, duplicated = live_index(run / 'telemetry' / f'{ordinal}.jsonl')
                except (ValueError, KeyError, TypeError, OSError) as error:
                    cell_errors.append(dict(ordinal=ordinal, error=f'{type(error).__name__}: {error}'));continue
                for row in captured:
                    request = {k: row[k] for k in bench.KEYS};found = live_fields(telemetry.get(row['decision_index']))
                    if not sent(request):
                        skipped += 1
                        if found is not None and found['request_kind'] != 'forced':skip_disagreements.append(dict(ordinal=ordinal, decision_index=row['decision_index'],
                            live_request_kind=found['request_kind'], sent=False))
                        continue
                    if args.max_decisions is not None and len(rows) >= args.max_decisions:done = True;break
                    problem = ('no live telemetry row' if found is None else 'duplicated live telemetry rows' if row['decision_index'] in duplicated
                        else 'live prefix differs from the capture\'s' if found['prefix_sha256'] != row['prefix_sha256'] else None)
                    if found is not None and found['request_kind'] == 'forced':skip_disagreements.append(dict(ordinal=ordinal, decision_index=row['decision_index'],
                        live_request_kind='forced', sent=True))
                    body = dict(request, lines=list(request['lines']), schema=server.original.SCHEMA,
                        request_id=f'rr-{ordinal}-{row["decision_index"]}-{os.getpid()}', rung=search.rung, budget_seconds=bench.BUDGET)
                    capture = Capture()
                    with capture.installed():
                        clock = time.perf_counter();response = search.process_many([body])[0];wall = time.perf_counter() - clock
                    clock = time.perf_counter();detail = describe(capture, peer_lanes, vocabulary, peer_inputs,
                        (ordinal, row['decision_index']));seconds = time.perf_counter() - clock
                    record = decision_row(cell, row, body, response, wall, found, detail, shard=shard, capture_seconds=seconds, live_problem=problem)
                    stream.write(encode_row(record) + '\n');stream.flush();os.fsync(stream.fileno())
                    rows.append({k: v for k, v in record.items() if k not in BULK})   # the summary's fields; the grids stay on disk
                    p = record['parity']
                    print(json.dumps(dict(ordinal=ordinal, decision_index=row['decision_index'], status=response.get('status'), cause=response.get('cause'),
                        live_source=None if found is None else found['source'], action_equal=p['action_equal'], q_equal=p['q_equal'],
                        kept_disagreements=detail['kept_disagreements'], notes=len(detail['notes']), wall=wall, capture=seconds)), flush=True)
                    if not record['capture_ok']:
                        raise RuntimeError(f'cell {ordinal} decision {row["decision_index"]}: the capture missed a decide or prepare_roots call')
        finished = True
    finally:
        write_json(out / f'summary-{shard}.json', summarize(rows, shard=shard, shards=args.shards, cells=cells, extra=dict(
            skipped_single_legal=skipped, skip_disagreements=skip_disagreements, cell_errors=cell_errors, max_decisions_reached=done,
            interrupted=not finished, timings=dict(build_seconds=build_seconds, warmup_seconds=warmup_seconds,
            replay_seconds=time.perf_counter() - started))))
        peer_inputs.write(out / sidecar['path'], sidecar)
    s = read_json(out / f'summary-{shard}.json')
    print(json.dumps(dict(done=str(out), shard=shard, sent=s['decisions_sent'], complete=s['complete'], live_searched=s['live_searched'],
        parity=dict((k, s['parity'][k]) for k in ('both', 'action_equal', 'q_equal')), status_mismatches=len(s['status_mismatches']),
        kept_legal_disagreements=s['kept_legal_disagreements'], peer_inputs_check_passed=s['peer_inputs_check']['passed'])),
        flush=True)
    return 0 if rows else 2


if __name__ == '__main__':raise SystemExit(main())
