#!/usr/bin/env python3
"""Opponent-view parity: does the search's per-world OPPONENT view equal Foul Play's real observation of the same state?

usage: depthn_opponent_view_parity.py capture <the live server's flags> --collection RUN --fp-reply-data FPR --out DIR
           (--cells ORD... | --sample N --sample-seed S) [--shard I --shards N] [--max-positions N] [--sampled-worlds 2]
       depthn_opponent_view_parity.py report --capture DIR [DIR ...] --out DIR [--top 30] [--examples 3] [--tolerance 1e-5]
           [--vocabulary vocabulary.json]

Why. pi_FP (the Foul Play reply model) reads Foul Play's own env observation, LocalShowdownEnv.observe(Foul Play), v4: 23 tokens
x 132 numeric / 41 categorical. The RR diagnostic will feed it the search's per-world opponent view instead: per world,
device_depth2_pricer._make_views(executor)(state, public, owner) = (own.observation, peer.observation, own.inputs, peer.inputs),
peer = executor.observe(state, public, 1 - seat, masks[:, 1 - owner]); the view is the 23-token v4 prefix of peer.inputs (V2
layout, 29 tokens) and the opponent's legal lanes are peer.observation.legal_action_mask. On the same state both must agree.

Capture (one GPU; the live server's build, as scripts/depthn_rr_reprice.py replays a screen battle).
- Search: depthn_live_server.build on the server's flags (bucket 1, max_requests 1, per-battle sessions, saved-root admission
  off), its run header held to the live READY identity (depthn_rr_reprice.identity_problems), warmed up on the server's own
  --warmup-requests. The tree must be clean under gpu_engine/, src/ and scripts/; HEAD is recorded, and whether its trees are
  the screen servers' pinned ones (depthn_rr_reprice.LIVE_TREES) is recorded, not required: these cells are RAW-arm games, so
  no live search exists to match.
- Cells: RAW cells of the Foul Play collection (results/<ordinal>.json cell.arm RAW, status COMPLETE) with an owner capture,
  Foul Play reply rows and a battle seed outside the screen band 2103400000-2103400999 (fp_reply_data's exclusion); --cells,
  or --sample N --sample-seed S stratified over (seed % 2, seat) by depthn_rr_reprice.sample_cells; --shard I --shards N keeps
  every N-th cell in ordinal order from the I-th (depthn_rr_reprice.shard_cells).
- Env replay, per cell before anything is sent: fp_reply_data.replay's walk re-derived (pz-v2s-harness-pr
  gpu_engine/fp_reply_data.py:107-148, without its policy): LocalShowdownEnv.reset(seed) on the constructor's own owning config
  (the pinned Showdown runtime, the champion's v4 spec, vocabulary and masks), then per round our recorded
  played_action_index and Foul Play's journal action, every requested player observed in order as fp_reply_data does. Checked: Foul Play has a journal entry exactly in its requested rounds
  and its action is legal in its own view; each of our requests equals the captured own_request_line (rqid and the side's
  name ignored, fp_reply_data.py:58-62, 136). Kept per round: the turn, which of our decisions it holds, Foul Play's literal
  request and its env observation (fp_reply_data.py:195-199's arrays). Foul Play's TRUE team: its round-0 observation's
  metadata['self_team'] (Showdown's request order at battle start is the original party order).
- Round alignment: our decision's round is request_count - 1 (one |request| line per side per request boundary, a wait request
  included; Foul Play's journal round is the same per-boundary counter, foulplay_bridge.OpponentJournalEntry). A position is
  a sent decision (more than one legal action; the live actor never sends one) whose round the env replay assigns to this very
  decision, where Foul Play was requested and has a reply row (fp_reply_data's (ordinal, round)). Also checked and recorded,
  never dropped: the capture's, the env's and the reply row's turns agree, the re-computed env observation equals the reply
  row key by key, and Foul Play's journal action equals the row's label.
- Replay: every sent decision of the cell goes to process_many alone in decision order, its request built as the re-pricer
  builds it (the battle's session folds exactly as live). Around each call only, phase_d_protocol_world.sample is wrapped
  (ProtocolConstructor.prepare looks it up as `protocol.sample` at call time; check_call_site verifies that from its
  bytecode before anything loads): the wrapper keeps the sampler's arguments and its WorldSet, calls the original with the
  same objects and is restored when the call returns or raises. Nothing on the decision path changes.
- Worlds per position (after the call returns; every world built on the CPU device as prepare_roots builds them):
  - true: the sampler's own construction with Foul Play's team swapped in. The sampler draws
    start, reason = _gen3_randbat_belief_start_override_result(...) and calls v5_public_world.materialize(root, start,
    prepared=ProtocolInputs(_carried_root(root, detached, lines, metadata), lines, catalog, metadata, source), history=None,
    own_static=None, seed=kernel_seed) (phase_d_protocol_world.py:398-411). Here start = BattleStartOverride(our team exactly
    as the sampler builds it, determinization.py:384-390 (_root_self_team_payload or metadata['self_team'] ->
    _self_team_from_metadata_result -> pack_team); Foul Play's true team through the very same function on Foul Play's own
    round-0 self_team rows, so level, gender, item, ability, move order and an EV/IV spread verified against the request's
    stats are exact; observation_format_id the root's format, determinization.py:414-422), materialized the same way with
    world 0's kernel seed (phase_d_protocol_world.py:402-404). One fact the sampler's format never writes: happiness (packed
    field 11, '' = 255). Where the literal round-0 request's Return/Frustration power proves a unique value other than 255
    (phase_d_live_worlds._own_happiness, the constructor's own inversion), it is written; an ambiguous power leaves 255 and
    is noted. Nature is never written (neutral), as the sampler's: the stats check proves the spread under it.
  - true_private: the true world with Foul Play's request-exact private dynamic facts written over the materializer's draws:
    the HP of each of its party members (materialize draws a seen foe's HP uniformly inside its public bin) and its active
    member's PP (the literal request's active moves, when they list the member's moves in slot order; public PP charges
    against the source maximum otherwise). The public carry is untouched. This separates the materializer's hidden-state
    draws from encoder gaps.
  - sampled_0 .. sampled_{n-1} (--sampled-worlds, default 2): the constructor's own worlds of this decision.
  Each world's view: jax.device_get((state, public)), then _make_views on the live pricers' shared executor
  (search.pricers[bucket].executor, as the re-pricer obtains it).
- Output (--out, a new directory): arrays-<shard>-<ordinal>.npz per cell (env_*: the reply row's numeric [n,23,132],
  categorical [n,23,41], token types, attention [n,23] and legal [n,9]; view_*: the same per world kind [n,K,...], view_valid
  and view_error [n,K]; kinds [K]; ordinal, round, decision_index, turn [n]), positions-<shard>.jsonl (one row per position:
  alignment checks, the response, each world's construction facts, per-kind exact flags, its npz and row), header-<shard>.json
  and summary-<shard>.json (counts, non-aligned reasons, cell errors; written on interruption too); a progress line per
  position on stdout. Exit 0 when a position was captured, 2 when none was.

Report (CPU, numpy only; pokezero for column names when importable). Per world kind, over its constructed positions: the
exact-match share per key (numeric, categorical, token types, attention, legal) and overall; whether the self (1-6) and
opponent (7-12) tokens match up to order; per token, per numeric column and per categorical column the share of positions
that differ and the max |diff|; every differing (token, column) cell ranked (beyond --tolerance share, then exact share,
then max |diff|) with examples from both sides; legal-lane disagreement both ways; the sampled worlds' views against the true
world's. Names: tokens field / self0-5 / opp0-5 / move0-3 / switch0-4 / tendency_stats (engine_device_observation.TOKEN_TYPES),
numeric columns through showdown.V4_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX and the NUMERIC_* writer names (as
engine_device_observation.column_manifest names them), categorical columns through the CATEGORY_* layout. Written as
summary.json and summary.md in --out.
"""
import argparse
import collections
from contextlib import contextmanager
from dataclasses import dataclass, field
import dis
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0, str(ROOT))

POSITION_SCHEMA = 'pokezero.depthn-opponent-view-parity.position.v1'
REPORT_SCHEMA = 'pokezero.depthn-opponent-view-parity.report.v1'
TOKENS, NUMERIC, CATEGORICAL, ACTIONS, TEAM = 23, 132, 41, 9, 6
SCREEN_BAND = (2103400000, 2103401000)   # fp_reply_data.py:41: RR scores operators on those roots
MAX_ROUNDS = 400   # fp_reply_data.py:45
OBS_KEYS = ('numeric_features', 'categorical_ids', 'token_type_ids', 'attention_mask')
SIDES = dict(numeric='numeric_features', categorical='categorical_ids', token_type='token_type_ids', attention='attention_mask', legal='legal')
SELF_TOKENS, OPPONENT_TOKENS = list(range(1, 7)), list(range(7, 13))
SAMPLE_PARAMETERS = ('root', 'detached', 'lines', 'catalog', 'source', 'metadata', 'config', 'battle_id', 'seed', 'count')
ARRAY_KEYS = ('env_numeric', 'env_categorical', 'env_token_type', 'env_attention', 'env_legal', 'view_numeric', 'view_categorical',
    'view_token_type', 'view_attention', 'view_legal', 'view_valid', 'view_error', 'ordinal', 'round', 'decision_index', 'turn')
FPR_ARRAYS = ('numeric_features', 'categorical_ids', 'token_type_ids', 'attention_mask', 'legal')
FPR_SCALARS = ('round', 'turn', 'forced', 'label', 'seed', 'seat')


def _rr():
    from scripts import depthn_rr_reprice as rr
    return rr


class CellError(Exception):
    def __init__(self, kind, detail=''):
        self.kind, self.detail = kind, str(detail)[:500]
        super().__init__(kind + (f': {detail}' if detail else ''))


class Mismatch(Exception):
    pass


def read_json(path):
    return json.loads(Path(path).read_bytes())


def read_rows(path):
    path = Path(path)
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def plain(x):
    """JSON-ready: numpy scalars and arrays as Python values, non-finite floats as strings."""
    if isinstance(x, dict):return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):return [plain(v) for v in x]
    if isinstance(x, np.ndarray):return plain(x.tolist())
    if isinstance(x, np.bool_):return bool(x)
    if isinstance(x, np.integer):return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if math.isfinite(x) else repr(x)
    return x


def dumps(value, **keywords):
    return json.dumps(plain(value), sort_keys=True, allow_nan=False, default=repr, **keywords)


def write_json(path, value):
    with Path(path).open('x') as stream:stream.write(dumps(value, indent=1) + '\n');stream.flush();os.fsync(stream.fileno())


def in_screen_band(seed):
    return seed is not None and SCREEN_BAND[0] <= int(seed) < SCREEN_BAND[1]


# Layout names -------------------------------------------------------------------------------------------------------------------
def token_labels():
    """engine_device_observation.TOKEN_TYPES' order: field, six own members, six opponent members, nine action candidates
    (four moves, five switches), the opponent-tendency stats token."""
    return (['field'] + [f'self{i}' for i in range(6)] + [f'opp{i}' for i in range(6)] + [f'move{i}' for i in range(4)]
        + [f'switch{i}' for i in range(5)] + ['tendency_stats'])


RANGED = (('OPP_MOVE_PP_OFFSET', 16), ('OPP_MOVE_PP_VALID_OFFSET', 16), ('STAT_WEATHER_REVEAL_OFFSET', 8))


def numeric_names():
    """The 132 physical v4 numeric columns by their writer names (showdown.NUMERIC_*, resolved through
    V4_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX as engine_device_observation.column_manifest does; ranged writers as NAME[k])."""
    try:
        from pokezero import showdown as s
        order = tuple(s.V4_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX)
    except Exception:
        return [f'num{c}' for c in range(NUMERIC)]
    if len(order) != NUMERIC:return [f'num{c}' for c in range(NUMERIC)]
    legacy = collections.defaultdict(list)
    for name, value in vars(s).items():
        if name.startswith('NUMERIC_') and type(value) is int and not name.endswith('_OFFSET'):legacy[value].append(name[len('NUMERIC_'):])
    for name, width in RANGED:
        start = getattr(s, 'NUMERIC_' + name, None)
        if type(start) is int:
            for k in range(width):legacy[start + k].append(f'{name[:-len("_OFFSET")]}[{k}]')
    return ['/'.join(legacy.get(i) or [f'legacy{i}']) for i in order]


def categorical_names():
    """The 41 v4 categorical columns: the nine fixed ones, belief ability / item / move buckets, active volatiles, last used move,
    traced ability (showdown.CATEGORY_* layout)."""
    try:
        from pokezero import showdown as s
        out = [f'cat{c}' for c in range(CATEGORICAL)]
        for name, value in vars(s).items():
            if name.startswith('CATEGORY_') and type(value) is int and 0 <= value < s.CATEGORY_FIXED_COUNT and not name.endswith(('_COUNT', '_OFFSET')):
                out[value] = name[len('CATEGORY_'):]
        for start, width, label in ((s.CATEGORY_BELIEF_ABILITY_OFFSET, s.BELIEF_ABILITY_BUCKET_COUNT, 'BELIEF_ABILITY'),
                (s.CATEGORY_BELIEF_ITEM_OFFSET, s.BELIEF_ITEM_BUCKET_COUNT, 'BELIEF_ITEM'), (s.CATEGORY_BELIEF_MOVE_OFFSET, s.BELIEF_MOVE_BUCKET_COUNT, 'BELIEF_MOVE'),
                (s.CATEGORY_VOLATILE_OFFSET, s.VOLATILE_BUCKET_COUNT, 'VOLATILE')):
            for k in range(width):out[start + k] = f'{label}[{k}]'
        out[s.CATEGORY_LAST_USED_MOVE] = 'LAST_USED_MOVE';out[s.CATEGORY_TRACED_ABILITY] = 'TRACED_ABILITY'
        return out
    except Exception:
        return [f'cat{c}' for c in range(CATEGORICAL)]


def vocabulary_decoder(path):
    """A category id -> token decoder from the champion's vocabulary document (CategoryVocabulary: 0 is padding, row i is
    tokens[i - 1], then the OOV buckets), or None without a path."""
    if not path:return None
    tokens = list(read_json(path)['tokens'])
    return lambda i: '<pad>' if i == 0 else tokens[i - 1] if 0 < i <= len(tokens) else f'<oov:{i - 1 - len(tokens)}>' if i > 0 else f'<{i}>'


# Collection, cells and reply rows -------------------------------------------------------------------------------------------------
def load_game(run, cell):
    """What the env replay needs of one collected game (fp_reply_data.load_game, fp_reply_data.py:69-98, re-derived): Foul Play's
    and our seats, the battle seed, Foul Play's journal by round, our played actions by decision_index. CellError otherwise."""
    run = Path(run);ordinal = cell['ordinal']
    result = read_json(run / 'results' / f'{ordinal}.json')
    if result.get('status') != 'COMPLETE':raise CellError('not_complete')
    game = result.get('source_game') or {};journal = game.get('opponent_moves') or []
    if not journal:raise CellError('no_journal')
    fp = (result.get('benchmark') or {}).get('foulplay_player');ours = {'p1': 'p2', 'p2': 'p1'}.get(fp)
    if ours is None:raise CellError('unknown_seats', fp)
    seed = int(game['seed'])
    if in_screen_band(seed) or in_screen_band(cell.get('seed')):raise CellError('screen_band', seed)
    if cell.get('seed') is not None and int(cell['seed']) != seed:raise CellError('seed_differs', f'cell {cell["seed"]} game {seed}')
    by_round = {}
    for entry in journal:
        if entry.get('seat') != fp or entry.get('round') in by_round:raise CellError('journal_shape')
        by_round[int(entry['round'])] = entry
    telemetry = sorted((r for r in read_rows(run / 'telemetry' / f'{ordinal}.jsonl') if r.get('owner') == ours), key=lambda r: r['decision_index'])
    if [r['decision_index'] for r in telemetry] != list(range(len(telemetry))):raise CellError('telemetry_gap')
    return dict(ordinal=ordinal, seed=seed, seat=cell.get('seat'), fp=fp, ours=ours, journal=by_round,
        our_actions=[r.get('played_action_index') for r in telemetry], decisions=game.get('pokezero_decisions'))


class FpReplyRows:
    """fp_reply_data's shards (shard-*.npz: one row per Foul Play decision): an (ordinal, round) index, and the rows of chosen
    games loaded once per shard (copied out, so no whole shard array stays alive)."""
    def __init__(self, directory):
        self.directory = Path(directory);self.paths = sorted(self.directory.glob('shard-*.npz'))
        if not self.paths:raise ValueError(f'{directory}: no shard-*.npz')
        self.where, self.shapes = {}, {}
        for i, path in enumerate(self.paths):
            with np.load(path) as data:
                missing = [k for k in FPR_ARRAYS + FPR_SCALARS + ('ordinal',) if k not in data.files]
                if missing:raise ValueError(f'{path.name}: missing {missing}')
                ordinals, rounds = np.asarray(data['ordinal']).tolist(), np.asarray(data['round']).tolist()
            for row, key in enumerate(zip(map(int, ordinals), map(int, rounds))):
                if key in self.where:raise ValueError(f'{path.name}: (ordinal, round) {key} appears twice')
                self.where[key] = (i, row)
        self.ordinals = frozenset(o for o, _ in self.where)
        self.documents = {}
        for path in sorted(self.directory.glob('shard-*.json')):
            try:document = read_json(path)
            except (OSError, ValueError):continue
            self.documents[path.name] = {k: document.get(k) for k in ('schema', 'games', 'decisions', 'mismatched', 'checkpoint_sha256', 'shapes', 'encoder_parity')}

    def load(self, ordinals):
        """{ordinal: {round: row}} for these games; each row's arrays checked against the v4 shapes."""
        want = set(ordinals);by_path = collections.defaultdict(list);out = {o: {} for o in want}
        for (ordinal, round_), (i, row) in self.where.items():
            if ordinal in want:by_path[i].append((ordinal, round_, row))
        for i, items in sorted(by_path.items()):
            with np.load(self.paths[i]) as data:columns = {k: data[k] for k in FPR_ARRAYS + FPR_SCALARS}
            for ordinal, round_, row in items:
                value = {k: np.array(v[row]) for k, v in columns.items()}
                shapes = tuple(value[k].shape for k in FPR_ARRAYS)
                if shapes != ((TOKENS, NUMERIC), (TOKENS, CATEGORICAL), (TOKENS,), (TOKENS,), (ACTIONS,)):
                    raise ValueError(f'{self.paths[i].name}: row {row} has shapes {shapes}, not the v4 observation\'s')
                out[ordinal][round_] = value
        return out


def cell_problem(cell, run, fp_ordinals):
    """Why a collection cell cannot be replayed here (None when it can)."""
    if cell['arm'] != 'RAW':return 'not_raw'
    if cell['status'] != 'COMPLETE':return 'not_complete'
    if in_screen_band(cell['seed']):return 'screen_band'
    if not (Path(run) / 'owning-wire' / f'{cell["ordinal"]}.jsonl').exists():return 'no_owner_capture'
    if cell['ordinal'] not in fp_ordinals:return 'no_fp_reply_rows'
    return None


def select_cells(args, fp_ordinals):
    """--cells (each must be replayable) or --sample N --sample-seed S over the replayable cells (depthn_rr_reprice.sample_cells),
    then this shard's (depthn_rr_reprice.shard_cells)."""
    rr = _rr();run = Path(args.collection)
    if args.cells:
        cells = [rr.cell_of(run, o) for o in dict.fromkeys(args.cells)]
        bad = [(c['ordinal'], why) for c in cells if (why := cell_problem(c, run, fp_ordinals))]
        if bad:raise ValueError(f'cells not replayable: {bad}')
    else:
        ordinals = sorted(int(p.stem) for p in (run / 'results').glob('*.json') if p.stem.isdigit())
        pool = [c for c in (rr.cell_of(run, o) for o in ordinals) if cell_problem(c, run, fp_ordinals) is None]
        cells = rr.sample_cells(pool, args.sample, args.sample_seed)
    return rr.shard_cells(cells, args.shard, args.shards)


# Env replay and alignment ----------------------------------------------------------------------------------------------------------
def request_body(line):
    """A request without its rqid and the side's username (fp_reply_data._request_body, fp_reply_data.py:58-62)."""
    body = json.loads(line[len('|request|'):] if line.startswith('|request|') else line)
    body = dict(body);body.pop('rqid', None)
    if isinstance(body.get('side'), dict):body['side'] = {k: v for k, v in body['side'].items() if k != 'name'}
    return body


def observation_arrays(observation):
    """An env observation's arrays as fp_reply_data stores them (fp_reply_data.py:195-199)."""
    return dict(numeric_features=np.asarray(observation.numeric_features, np.float32), categorical_ids=np.asarray(observation.categorical_ids, np.int32),
        token_type_ids=np.asarray(observation.token_type_ids, np.int16), attention_mask=np.asarray(observation.attention_mask, bool),
        legal=np.asarray(observation.legal_action_mask, bool))


def replay_rounds(env, game, our_requests):
    """fp_reply_data.replay's walk (fp_reply_data.py:107-148) without its policy: ({round: dict(turn, requested, our_decision,
    fp_request, fp_observation, fp_action)}, Foul Play's round-0 self_team rows). Mismatch on any check."""
    env.reset(seed=game['seed'])
    journal = dict(game['journal']);table = {};ours = 0;round_ = 0;team = None
    while env.terminal() is None:
        requested = tuple(env.requested_players())
        if not requested:raise Mismatch('no requested players before terminal')
        entry = dict(turn=int(env._latest_turn), requested=list(requested), our_decision=None, fp_request=None, fp_observation=None, fp_action=None)
        actions = {}
        for player in requested:
            observation = env.observe(player)   # every requested player, in order, as fp_reply_data observes them (env state stays its)
            if player == game['fp']:
                choice = journal.pop(round_, None)
                if choice is None:raise Mismatch(f'Foul Play requested in round {round_} without a journal entry')
                action = int(choice['action'])
                if not 0 <= action < ACTIONS or not observation.legal_action_mask[action]:
                    raise Mismatch(f'Foul Play action {action} illegal in its own view, round {round_}')
                if team is None:
                    rows = list((observation.metadata or {}).get('self_team') or ())
                    if round_ != 0 or len(rows) != TEAM:raise Mismatch(f'Foul Play\'s first observation (round {round_}) has {len(rows)} self_team rows')
                    team = json.loads(json.dumps(rows, default=repr))
                entry.update(fp_request=json.loads(json.dumps(env._latest_requests.get(player))), fp_observation=observation_arrays(observation), fp_action=action)
                actions[player] = action
            else:
                if ours >= len(game['our_actions']):raise Mismatch(f'our seat requested beyond its {ours} recorded decisions')
                captured = our_requests.get(ours)
                if captured is None or request_body(captured) != request_body(json.dumps(env._latest_requests.get(player))):
                    raise Mismatch(f'our request differs from the capture at decision {ours}')
                entry['our_decision'] = ours;actions[player] = int(game['our_actions'][ours]);ours += 1
        table[round_] = entry
        env.step(actions);round_ += 1
        if round_ > MAX_ROUNDS:raise Mismatch('runaway replay')
    if journal:raise Mismatch(f'{len(journal)} journal entries never requested, first round {min(journal)}')
    if ours != len(game['our_actions']):raise Mismatch(f'our seat made {ours} of {len(game["our_actions"])} decisions')
    if team is None:raise Mismatch('Foul Play was never requested')
    return table, team


def align(row, table, fp_rows):
    """One captured decision's alignment: its round (request_count - 1), whether it is a position (the env replay puts this very
    decision in that round, Foul Play was requested there and has a reply row) and the recorded cross-checks."""
    round_ = int(row['request_count']) - 1;base = dict(round=round_, aligned=False, reason=None)
    entry = table.get(round_)
    if entry is None:return dict(base, reason='round_not_replayed')
    if entry['our_decision'] != row['decision_index']:return dict(base, reason='our_decision_elsewhere')
    if entry['fp_observation'] is None:return dict(base, reason='fp_not_requested')
    fpr = fp_rows.get(round_)
    if fpr is None:return dict(base, reason='no_fp_reply_row')
    turns = dict(capture=row.get('turn'), env=entry['turn'], fp_reply=int(fpr['turn']))
    equal = {k: bool(np.array_equal(np.asarray(entry['fp_observation'][k]), np.asarray(fpr[k]))) for k in OBS_KEYS + ('legal',)}
    return dict(base, aligned=True, turns=turns, turns_equal=len(set(turns.values())) == 1, env_equals_fp_reply=equal,
        fp_action=entry['fp_action'], fp_reply_label=int(fpr['label']), label_equal=entry['fp_action'] == int(fpr['label']),
        fp_reply_forced=bool(fpr['forced']))


# The true start and world ---------------------------------------------------------------------------------------------------------
def with_happiness(packed, rows, request):
    """`packed` with field 11 (happiness) set where the literal round-0 request's Return/Frustration power proves a unique value
    other than 255 (phase_d_live_worlds._own_happiness); (packed, values (None: default 255), notes)."""
    from pokezero.dex import normalize_id
    from gpu_engine.phase_d_live_worlds import Refusal, _own_happiness
    literal = ((request or {}).get('side') or {}).get('pokemon') or [];entries = packed.split(']');values, notes = [], []
    for i, entry in enumerate(entries):
        fields = entry.split('|');name = normalize_id((rows[i] or {}).get('species'));mon = literal[i] if i < len(literal) else None
        if mon is None or normalize_id(str(mon.get('details', '')).split(',')[0]) != name:
            values.append(None);notes.append(f'{name}: no literal round-0 request member');continue
        try:value = _own_happiness({}, tuple(mon.get('moves') or ()), name)
        except Refusal as error:value = None;notes.append(f'{name}: {error.kind}')
        values.append(value)
        if value is not None and value != 255:fields[11] = str(value)
        entries[i] = '|'.join(fields)
    return ']'.join(entries), values, notes


def true_start(root, team_rows, round0_request, set_source, owner):
    """(BattleStartOverride, info): our team exactly as the sampler builds it (determinization.py:384-390), Foul Play's true team
    through the same function on its own round-0 self_team rows, packed by the same pack_team (plus a proven happiness);
    (None, info with the reason) when either team cannot be reconstructed."""
    from pokezero.determinization import _root_self_team_payload, _self_team_from_metadata_result
    from pokezero.env import BattleStartOverride
    from pokezero.showdown_fixture import pack_team
    context = root.context;metadata = context.observation.metadata
    ours, why = _self_team_from_metadata_result(_root_self_team_payload(context, team_size=TEAM) or metadata.get('self_team'), team_size=TEAM,
        set_source=set_source)
    if ours is None:return None, dict(reason=f'own team: {why}')
    theirs, why = _self_team_from_metadata_result(team_rows, team_size=TEAM, set_source=set_source)
    if theirs is None:return None, dict(reason=f'foul play team: {why}')
    packed, happiness, notes = with_happiness(pack_team(theirs), team_rows, round0_request)
    start = BattleStartOverride(player_teams={f'p{owner + 1}': pack_team(ours), f'p{2 - owner}': packed}, observation_format_id=context.format_id)
    return start, dict(reason=None, species=[m.species for m in theirs], happiness=happiness, notes=notes,
        packed_sha256=hashlib.sha256(packed.encode()).hexdigest())


def kernel_seed(root, detached, seed, index):
    """World `index`'s kernel seed, as phase_d_protocol_world.sample derives it (phase_d_protocol_world.py:402-404)."""
    from gpu_engine.phase_d_protocol_world import SCHEMA
    identity = (SCHEMA, seed, root.context.battle_id, root.owner_lines_sha256, detached.address.stream, detached.address.frame,
        detached.address.owner, index)
    return int.from_bytes(hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).digest()[8:12], 'big')


@contextmanager
def cpu_device():
    import jax
    with jax.default_device(jax.devices('cpu')[0]):yield


def build_true_world(call, team_rows, round0_request, *, index=0):
    """(ProtocolWorld or None, info): the sampler's materialization (phase_d_protocol_world.py:398-411) of the true start, on the
    CPU device as prepare_roots builds worlds, with world `index`'s kernel seed."""
    import jax
    from gpu_engine import phase_d_protocol_world as pw, v5_public_world as vw
    named = call.named;root, detached = named['root'], named['detached'];owner = int(detached.address.owner)
    start, info = true_start(root, team_rows, round0_request, named['source'].original, owner)
    if start is None:return None, info
    info['kernel_seed'] = seed = kernel_seed(root, detached, named['seed'], index)
    with cpu_device():
        carried = pw._carried_root(root, detached, named['lines'], named['metadata'])
        inputs = pw.ProtocolInputs(carried, tuple(named['lines']), named['catalog'], named['metadata'], named['source'])
        world = vw.materialize(root, start, prepared=inputs, history=None, own_static=None, seed=seed)
        if world.state is not None:jax.block_until_ready((world.state, world.public))
    info.update(constructed=world.state is not None, refusals=list(world.refusals), origin_sha256=world.origin_sha256,
        triggered=sorted({str(d.get('kind')) for d in world.mapper_diagnostics if isinstance(d, dict) and d.get('status') == 'TRIGGERED'}))
    return world, info


def overlay_private(state, foe, request, vocabulary):
    """(state with Foul Play's request-exact HP of every party member and its active member's PP, notes). Members join the
    world's foe slots by species (the constructor's engine-id tables, metadata['vocabulary']); the active PP only when the
    literal active moves list the member's moves in slot order (a locked or reduced list keeps the world's PP)."""
    import jax.numpy as jnp
    from pokezero.dex import normalize_id
    from gpu_engine.phase_d_live_worlds import _source_move
    notes = [];hp = np.array(state.hp, copy=True);pp = np.array(state.pp, copy=True);max_hp = np.asarray(state.max_hp)
    species = [None if int(x) <= 0 else normalize_id(vocabulary['species'][int(x)]) for x in np.asarray(state.species)[0, foe]]
    members = ((request or {}).get('side') or {}).get('pokemon') or ();active_name = None
    for mon in members:
        name = normalize_id(str(mon.get('details', '')).split(',')[0])
        if mon.get('active'):active_name = name
        slots = [s for s, n in enumerate(species) if n == name]
        if len(slots) != 1:notes.append(f'{name}: {len(slots)} world slots');continue
        slot = slots[0];head = (str(mon.get('condition') or '').split() or [''])[0]
        try:
            if head == '0':value = 0
            else:
                value, maximum = map(int, head.split('/'))
                if maximum != int(max_hp[0, foe, slot]):notes.append(f'{name}: max_hp {maximum} != world {int(max_hp[0, foe, slot])}')
        except ValueError:notes.append(f'{name}: condition {head!r}');continue
        hp[0, foe, slot] = value
    active = (request or {}).get('active')
    if active and not any((request or {}).get('forceSwitch') or ()):
        slot = int(np.asarray(state.active)[0, foe]);literal = list(active[0].get('moves') or ())
        engine = [(i, int(m)) for i, m in enumerate(np.asarray(state.moves)[0, foe, slot]) if int(m) > 0]
        if species[slot] != active_name:notes.append(f'active: world {species[slot]} != request {active_name}')
        elif len(literal) != len(engine) or any(_source_move(str(m.get('id') or m.get('move') or '')) != _source_move(vocabulary['moves'][e])
                for m, (_, e) in zip(literal, engine)):notes.append('active moves: the literal list is not the member\'s slots (PP kept)')
        else:
            for m, (i, _) in zip(literal, engine):
                if type(m.get('pp')) is int:pp[0, foe, slot, i] = m['pp']
    return state._replace(hp=jnp.asarray(hp), pp=jnp.asarray(pp)), notes


# Views ----------------------------------------------------------------------------------------------------------------------------
def view_arrays(peer, inputs):
    """The opponent view of one world: peer.inputs' 23-token v4 prefix, peer.observation's legal lanes and error bits."""
    out = {}
    for key, tail in (('numeric_features', 2), ('categorical_ids', 2), ('token_type_ids', 1), ('attention_mask', 1)):
        a = np.asarray(inputs[key])
        if int(np.prod(a.shape[:-tail])) != 1:raise ValueError(f'{key}: one world\'s view required, got {a.shape}')
        out[key] = a.reshape(a.shape[-tail:])[:TOKENS]
    out['legal'] = np.asarray(peer.legal_action_mask, bool).reshape(-1);out['error'] = int(np.asarray(peer.error).reshape(-1)[0])
    shapes = (out['numeric_features'].shape, out['categorical_ids'].shape, out['token_type_ids'].shape, out['attention_mask'].shape, out['legal'].shape)
    if shapes != ((TOKENS, NUMERIC), (TOKENS, CATEGORICAL), (TOKENS,), (TOKENS,), (ACTIONS,)):raise ValueError(f'view shapes {shapes}')
    return out


class OpponentView:
    """device_depth2_pricer._make_views on the live executor: its second output (peer observation) and fourth (peer inputs)."""
    def __init__(self, executor):
        from gpu_engine import device_depth2_pricer as depth2
        self.views = depth2._make_views(executor)

    def __call__(self, state, public, owner):
        _, peer, _, inputs = self.views(state, public, owner)
        return view_arrays(peer, inputs)


# The sampler wrapper --------------------------------------------------------------------------------------------------------------
@dataclass
class SampleCall:
    named: dict
    result: object = None
    error: object = None


class SampleCapture:
    """References to every phase_d_protocol_world.sample call's arguments and WorldSet while installed() is active."""
    def __init__(self):
        self.calls = []

    @contextmanager
    def installed(self):
        from gpu_engine import phase_d_protocol_world as pw
        original = pw.sample

        def wrapper(*arguments, **keywords):
            call = SampleCall(dict(zip(SAMPLE_PARAMETERS, arguments), **keywords));self.calls.append(call)
            try:call.result = original(*arguments, **keywords)
            except BaseException as error:
                call.error = f'{type(error).__name__}: {error}'
                raise
            return call.result

        pw.sample = wrapper
        try:yield self
        finally:pw.sample = original


def check_call_site(function=None):
    """True when `function` (ProtocolConstructor.prepare) reads `sample` as an attribute of its global `protocol`
    (phase_d_protocol_world) at call time, so the wrapper takes effect; RuntimeError otherwise."""
    from gpu_engine import phase_d_protocol_constructor as pc, phase_d_protocol_world as pw
    function = pc.ProtocolConstructor.prepare if function is None else function
    found, previous = False, None
    for instruction in dis.get_instructions(function):
        if (instruction.opname in ('LOAD_ATTR', 'LOAD_METHOD') and instruction.argval == 'sample' and previous is not None
                and previous.opname == 'LOAD_GLOBAL' and previous.argval == 'protocol'):found = True
        previous = instruction
    if not found or function.__globals__.get('protocol') is not pw:
        raise RuntimeError('ProtocolConstructor.prepare does not look phase_d_protocol_world.sample up through its module at call time')
    return True


# Capture --------------------------------------------------------------------------------------------------------------------------
def kinds_of(sampled_worlds):
    return ('true', 'true_private') + tuple(f'sampled_{i}' for i in range(sampled_worlds))


def exact_flags(env, view):
    """Per key equality of one position's reply row and one view (values compared, dtypes aside)."""
    out = {k: bool(np.array_equal(np.asarray(env[k]), np.asarray(view[k]))) for k in OBS_KEYS + ('legal',)}
    out['observation'] = all(out[k] for k in OBS_KEYS);out['all'] = out['observation'] and out['legal']
    return out


def position_arrays(fpr, views, kinds):
    """One position's arrays: the reply row (env_*) and every kind's view (view_*; NaN / -1 / False where not constructed)."""
    K = len(kinds)
    out = dict(env_numeric=np.asarray(fpr['numeric_features'], np.float32), env_categorical=np.asarray(fpr['categorical_ids'], np.int32),
        env_token_type=np.asarray(fpr['token_type_ids'], np.int16), env_attention=np.asarray(fpr['attention_mask'], bool),
        env_legal=np.asarray(fpr['legal'], bool), view_numeric=np.full((K, TOKENS, NUMERIC), np.nan, np.float32),
        view_categorical=np.full((K, TOKENS, CATEGORICAL), -1, np.int32), view_token_type=np.full((K, TOKENS), -1, np.int16),
        view_attention=np.zeros((K, TOKENS), bool), view_legal=np.zeros((K, ACTIONS), bool), view_valid=np.zeros(K, bool),
        view_error=np.full(K, -1, np.int32))
    for k, kind in enumerate(kinds):
        v = views.get(kind)
        if v is None:continue
        out['view_numeric'][k] = v['numeric_features'];out['view_categorical'][k] = v['categorical_ids']
        out['view_token_type'][k] = v['token_type_ids'];out['view_attention'][k] = v['attention_mask']
        out['view_legal'][k] = v['legal'];out['view_valid'][k] = True;out['view_error'][k] = v['error']
    return out


def response_fields(response):
    return dict(status=response.get('status'), cause=response.get('cause'), refusal=None if response.get('refusal') is None else str(response['refusal'])[:300],
        constructed_worlds=response.get('constructed_worlds'), root_class=response.get('root_class'),
        world_refusal_kinds=sorted({k for w in response.get('world_refusals') or () for k in w.get('kinds') or ()}))


@dataclass
class Replay:
    """One shard's capture state: the live search, the opponent view, the env (rebuilt after an env error), the reply rows."""
    run: Path
    search: object
    views: object
    fp_rows: dict
    vocabulary: object
    environment_factory: object
    sampled_worlds: int = 2
    max_positions: object = None
    shard: int = 0
    out: object = None
    stream: object = None
    env: object = None
    positions: int = 0
    done: bool = False
    counts: collections.Counter = field(default_factory=collections.Counter)
    cell_errors: list = field(default_factory=list)

    @property
    def kinds(self):
        return kinds_of(self.sampled_worlds)

    def environment(self):
        if self.env is None:self.env = self.environment_factory()
        return self.env

    def close_environment(self):
        if self.env is not None:
            try:self.env.close()
            finally:self.env = None

    def view(self, state, public, owner):
        import jax
        state, public = jax.device_get((state, public))
        return self.views(state, public, owner)


def describe_position(ctx, cell, game, row, alignment, capture, response, table, team):
    """One position's JSONL record and arrays (module docstring: the worlds per position)."""
    fpr = ctx.fp_rows[cell['ordinal']][alignment['round']];kinds = ctx.kinds;views, worlds, notes = {}, {}, []
    calls = capture.calls;call = calls[0] if calls else None
    if len(calls) > 1:notes.append(f'{len(calls)} sample calls; the first is used')
    if call is None:notes.append('no constructor sample call (the root was refused before sampling)')
    elif call.error is not None:notes.append(f'sample raised: {call.error}'[:300])
    else:
        owner = int(call.named['detached'].address.owner);foe = 1 - owner
        if f'p{foe + 1}' != game['fp']:notes.append(f'the constructor\'s opponent p{foe + 1} is not Foul Play {game["fp"]}')
        made = call.result
        try:
            world, info = build_true_world(call, team, table[0]['fp_request'])
            if 'kernel_seed' in info and made is not None and made.worlds:info['kernel_seed_is_world0s'] = info['kernel_seed'] == made.worlds[0].kernel_seed
        except Exception as error:
            world, info = None, dict(reason=f'{type(error).__name__}: {error}'[:300])
        worlds['true'] = info
        if world is not None and world.state is not None:
            try:views['true'] = ctx.view(world.state, world.public, owner)
            except Exception as error:notes.append(f'true view: {type(error).__name__}: {error}'[:300])
            try:
                state, overlay_notes = overlay_private(world.state, foe, table[alignment['round']]['fp_request'], ctx.vocabulary)
                worlds['true_private'] = dict(notes=overlay_notes)
                views['true_private'] = ctx.view(state, world.public, owner)
            except Exception as error:notes.append(f'true_private: {type(error).__name__}: {error}'[:300])
        for i in range(ctx.sampled_worlds):
            w = made.worlds[i] if made is not None and i < len(made.worlds) else None
            worlds[f'sampled_{i}'] = None if w is None else dict(index=w.index, kernel_seed=w.kernel_seed, refusals=list(w.refusals),
                constructed=w.state is not None)
            if w is not None and w.state is not None:
                try:views[f'sampled_{i}'] = ctx.view(w.state, w.public, owner)
                except Exception as error:notes.append(f'sampled_{i} view: {type(error).__name__}: {error}'[:300])
    arrays = position_arrays(fpr, views, kinds)
    record = dict(schema=POSITION_SCHEMA, shard=ctx.shard, ordinal=cell['ordinal'], seed=game['seed'], seat=cell.get('seat'), fp=game['fp'], ours=game['ours'],
        battle_id=row['battle_id'], decision_index=row['decision_index'], request_count=row['request_count'], round=alignment['round'],
        turn=row.get('turn'), alignment={k: v for k, v in alignment.items() if k not in ('round', 'aligned', 'reason')},
        fp_reply_seat_seed=dict(seat=int(fpr['seat']), seed=int(fpr['seed'])), response=response_fields(response), worlds=worlds,
        views={k: dict(error=v['error']) for k, v in views.items()}, exact={k: exact_flags(fpr, v) for k, v in views.items()}, notes=notes)
    return record, arrays


def capture_cell(ctx, cell):
    """Replay one cell (module docstring): the env replay first, then every sent decision through process_many in decision order;
    the cell's positions go to positions-<shard>.jsonl and arrays-<shard>-<ordinal>.npz. CellError before anything is sent."""
    from pokezero import phase_d_owning_capture as owning
    rr = _rr();bench, server = rr.bench, rr.server;ordinal = cell['ordinal']
    game = load_game(ctx.run, cell)
    try:captured = list(owning.read_capture(ctx.run / 'owning-wire' / f'{ordinal}.jsonl'))
    except (ValueError, KeyError, TypeError, OSError) as error:raise CellError('capture_unreadable', error) from None
    if not captured:raise CellError('no_decisions')
    if any(r['owner'] != game['ours'] for r in captured):raise CellError('capture_owner', f'{captured[0]["owner"]} != {game["ours"]}')
    try:table, team = replay_rounds(ctx.environment(), game, {r['decision_index']: r['own_request_line'] for r in captured})
    except Mismatch as error:raise CellError('replay_mismatch', error) from None
    except Exception as error:   # one bad game, not the shard; the env may be wedged (fp_reply_data.py:187-190)
        ctx.close_environment();raise CellError('env_error', f'{type(error).__name__}: {error}') from None
    records, columns = [], collections.defaultdict(list)
    for row in captured:
        request = {k: row[k] for k in bench.KEYS}
        if not rr.sent(request):ctx.counts['skipped_single_legal'] += 1;continue
        if ctx.max_positions is not None and ctx.positions >= ctx.max_positions:ctx.done = True;break
        alignment = align(row, table, ctx.fp_rows.get(ordinal, {}))
        body = dict(request, lines=list(request['lines']), schema=server.original.SCHEMA, request_id=f'ov-{ordinal}-{row["decision_index"]}-{os.getpid()}',
            rung=ctx.search.rung, budget_seconds=bench.BUDGET)
        capture = SampleCapture()
        with capture.installed():
            clock = time.perf_counter();response = ctx.search.process_many([body])[0];wall = time.perf_counter() - clock
        ctx.counts['sent'] += 1
        if not alignment['aligned']:ctx.counts['not_aligned:' + alignment['reason']] += 1;continue
        clock = time.perf_counter()
        record, arrays = describe_position(ctx, cell, game, row, alignment, capture, response, table, team)
        record.update(npz=f'arrays-{ctx.shard}-{ordinal}.npz', row=len(records), wall_seconds=wall, describe_seconds=time.perf_counter() - clock)
        records.append(record);ctx.positions += 1;ctx.counts['positions'] += 1
        for k, v in arrays.items():columns[k].append(v)
        print(dumps(dict(ordinal=ordinal, decision_index=row['decision_index'], round=alignment['round'], status=response.get('status'),
            turns_equal=alignment['turns_equal'], env_equals_fp_reply=all(alignment['env_equals_fp_reply'].values()),
            exact={k: v['all'] for k, v in record['exact'].items()}, true=(record['worlds'].get('true') or {}).get('reason') or (record['worlds'].get('true') or {}).get('refusals'),
            notes=len(record['notes']), wall=wall)), flush=True)
    if records:
        arrays = {k: np.stack(v) for k, v in columns.items()}
        arrays.update(kinds=np.asarray(ctx.kinds), ordinal=np.asarray([r['ordinal'] for r in records], np.int64),
            round=np.asarray([r['round'] for r in records], np.int64), decision_index=np.asarray([r['decision_index'] for r in records], np.int64),
            turn=np.asarray([-1 if r['turn'] is None else r['turn'] for r in records], np.int64))
        with (Path(ctx.out) / f'arrays-{ctx.shard}-{ordinal}.npz').open('xb') as stream:np.savez(stream, **arrays)
        for record in records:ctx.stream.write(dumps(record) + '\n')
        ctx.stream.flush();os.fsync(ctx.stream.fileno())
    return len(records)


def tree_state(root=ROOT, run=None):
    """HEAD, refusing tracked changes under gpu_engine/, src/ and scripts/; whether the guarded trees are the screen servers'
    pinned ones is recorded only."""
    rr = _rr();run = rr.git if run is None else run
    try:
        head = run(root, 'rev-parse', 'HEAD')
        dirty = run(root, 'status', '--porcelain', '--untracked-files=no', '--', 'gpu_engine', 'src', 'scripts').splitlines()
    except Exception as error:raise ValueError(f'{root} must be a git checkout ({error})') from None
    if dirty:raise ValueError('the tree has tracked changes: ' + '; '.join(dirty[:5]))
    live = {}
    for path, pinned in rr.LIVE_TREES.items():
        try:live[path] = run(root, 'rev-parse', f'HEAD:{path}') == pinned
        except Exception:live[path] = None
    return dict(head=head, live_trees_equal=live)


def capture_parser():
    rr = _rr();p = rr.bench.parser();p.prog = 'depthn_opponent_view_parity.py capture'
    for action in p._actions:
        if action.dest == 'roster':action.required = False
        if action.dest == 'warmup_requests':action.required = True   # the server's own warmup file
    p.add_argument('--collection', required=True, help='the Foul Play collection run directory (results/, telemetry/, owning-wire/ per ordinal)')
    p.add_argument('--fp-reply-data', required=True, help='fp_reply_data\'s output directory (shard-*.npz: Foul Play\'s env observations)')
    pick = p.add_mutually_exclusive_group(required=True)
    pick.add_argument('--cells', type=int, nargs='+', metavar='ORD', help='these cell ordinals')
    pick.add_argument('--sample', type=int, metavar='N', help='N replayable RAW cells, stratified equally over (seed %% 2, seat)')
    p.add_argument('--sample-seed', type=int, help='the sample\'s seed (required with --sample)')
    p.add_argument('--shard', type=int, default=0);p.add_argument('--shards', type=int, default=1)
    p.add_argument('--max-positions', type=int, help='stop after this many positions (smoke tests)')
    p.add_argument('--sampled-worlds', type=int, default=2, help='the constructor\'s own worlds compared too (default 2: worlds 0 and 1)')
    p.set_defaults(saved_root_admission=False, buckets=(1,))
    return p


def capture_options(argv):
    rr = _rr();p = capture_parser();args = p.parse_args(argv)
    rr.server.evaluators.check_arguments(p, args);rr.server.check_precision_environment(p, args)
    if not args.champion_archive:p.error('--champion-archive with its two sha256 flags is required: the champion prices every root and leaf')
    for name in rr.BENCH_ONLY:
        if getattr(args, name, None):p.error(f'--{name.replace("_", "-")} is the bench\'s; cells come from --cells or --sample')
    if args.buckets != (1,):p.error('the live servers priced at bucket 1 only')
    if args.max_requests not in (None, 1):p.error('the live servers took one request per batch (--max-requests 1)')
    if args.saved_root_admission:p.error('the live servers kept per-battle sessions (no --saved-root-admission)')
    if not any(a == '--rung' or a.startswith('--rung=') for a in argv):p.error('--rung is required: the live servers\' rung (e.g. DN2W16)')
    if args.sample is not None and (args.sample < 1 or args.sample_seed is None):p.error('--sample N (>= 1) needs --sample-seed')
    if args.sample is None and args.sample_seed is not None:p.error('--sample-seed goes with --sample')
    if args.shards < 1 or not 0 <= args.shard < args.shards:p.error('--shard I --shards N with 0 <= I < N required')
    if args.max_positions is not None and args.max_positions < 1:p.error('--max-positions must be >= 1')
    if not 0 <= args.sampled_worlds <= rr.live.RUNGS[args.rung]:p.error(f'--sampled-worlds must be in 0..{rr.live.RUNGS[args.rung]}')
    args.port = args.port or 0;args.max_requests = 1
    return args


def make_environment(config):
    from pokezero.local_showdown import LocalShowdownEnv
    return LocalShowdownEnv(config)


def capture_main(argv):
    args = capture_options(argv);rr = _rr();server = rr.server
    try:tree = tree_state();check_call_site();fpr = FpReplyRows(args.fp_reply_data);cells = select_cells(args, fpr.ordinals)
    except (ValueError, RuntimeError, OSError) as error:raise SystemExit(f'depthn_opponent_view_parity: {error}')
    if not cells:raise SystemExit('depthn_opponent_view_parity: no cell in this shard')
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False);shard = args.shard
    rows = fpr.load([c['ordinal'] for c in cells])
    saved = server.read_requests(args.warmup_requests)
    clock = time.perf_counter();search = server.build(args);build_seconds = time.perf_counter() - clock
    header = server.header(args, search);problems = rr.identity_problems(header, args.rung)
    if problems:raise SystemExit('depthn_opponent_view_parity: the built search is not the live servers\': ' + '; '.join(problems))
    clock = time.perf_counter();warm = search.warmup(saved);warmup_seconds = time.perf_counter() - clock
    import jax
    config = search.constructor.owning_config
    ctx = Replay(run=Path(args.collection), search=search, views=OpponentView(search.pricers[search.buckets[0]].executor), fp_rows=rows,
        vocabulary=search.constructor.metadata['vocabulary'], environment_factory=lambda: make_environment(config), sampled_worlds=args.sampled_worlds,
        max_positions=args.max_positions, shard=shard, out=out)
    write_json(out / f'header-{shard}.json', dict(argv=list(argv), tree=tree, run_header=header, live_identity=rr.LIVE_IDENTITY, warmup=warm,
        warmup_requests=len(saved), build_seconds=build_seconds, warmup_seconds=warmup_seconds, cells=cells, collection=str(args.collection),
        fp_reply_data=str(args.fp_reply_data), fp_reply_shards=[p.name for p in fpr.paths], fp_reply_documents=fpr.documents,
        fp_reply_rows={str(o): len(r) for o, r in rows.items()}, kinds=list(ctx.kinds), rung=args.rung, shard=shard, shards=args.shards,
        max_positions=args.max_positions, sample=args.sample, sample_seed=args.sample_seed, jax_version=jax.__version__,
        devices=[f'{d.platform}:{d.device_kind}' for d in jax.devices()], python=sys.version, environment={k: os.environ.get(k) for k in rr.ENVIRONMENT},
        pid=os.getpid()))
    started, finished = time.perf_counter(), False
    try:
        with (out / f'positions-{shard}.jsonl').open('x') as stream:
            ctx.stream = stream
            for cell in cells:
                if ctx.done:break
                try:capture_cell(ctx, cell);ctx.counts['cells_replayed'] += 1
                except CellError as error:ctx.cell_errors.append(dict(ordinal=cell['ordinal'], kind=error.kind, detail=error.detail))
        finished = True
    finally:
        ctx.close_environment()
        counts = dict(ctx.counts)
        write_json(out / f'summary-{shard}.json', dict(shard=shard, shards=args.shards, cells=[c['ordinal'] for c in cells], positions=ctx.positions,
            cells_replayed=counts.pop('cells_replayed', 0), sent=counts.pop('sent', 0), skipped_single_legal=counts.pop('skipped_single_legal', 0),
            not_aligned={k.split(':', 1)[1]: v for k, v in counts.items() if k.startswith('not_aligned:')}, cell_errors=ctx.cell_errors,
            max_positions_reached=ctx.done, interrupted=not finished, seconds=time.perf_counter() - started))
    print(dumps(dict(done=str(out), shard=shard, positions=ctx.positions, cell_errors=len(ctx.cell_errors))), flush=True)
    return 0 if ctx.positions else 2


# Report ---------------------------------------------------------------------------------------------------------------------------
def load_captures(directories):
    """(position rows, shard summaries, kinds, arrays concatenated over every arrays-*.npz) of capture directories."""
    rows, summaries, kinds, arrays = [], [], None, collections.defaultdict(list)
    for directory in directories:
        directory = Path(directory)
        for path in sorted(directory.glob('positions-*.jsonl')):rows += read_rows(path)
        for path in sorted(directory.glob('summary-*.json')):summaries.append(read_json(path))
        for path in sorted(directory.glob('arrays-*.npz')):
            with np.load(path) as data:
                found = tuple(str(x) for x in data['kinds'])
                if kinds is None:kinds = found
                elif found != kinds:raise ValueError(f'{path}: world kinds {found} differ from {kinds}')
                for key in ARRAY_KEYS:arrays[key].append(np.asarray(data[key]))
    if kinds is None:raise ValueError('no arrays-*.npz in ' + ', '.join(map(str, directories)))
    return rows, summaries, kinds, {k: np.concatenate(v) for k, v in arrays.items()}


def up_to_order(env, view, tokens):
    """Per position: whether the rows of `tokens` (numeric and categorical together) are the same multiset on both sides."""
    def rows(side):
        numeric = np.nan_to_num(np.asarray(side['numeric'], np.float64)[:, tokens, :], nan=-1e300)
        return np.concatenate([numeric, np.asarray(side['categorical'], np.float64)[:, tokens, :]], axis=2)
    a, b = rows(env), rows(view);out = np.zeros(len(a), bool)
    for i in range(len(a)):
        out[i] = np.array_equal(a[i][np.lexsort(a[i].T[::-1])], b[i][np.lexsort(b[i].T[::-1])])
    return out


def diff_stats(env, view, *, tolerance):
    """The comparison of two sides over N positions (dicts numeric [N,23,132], categorical [N,23,41], token_type, attention [N,23],
    legal [N,9]): exact shares per key, per cell / token / column differ shares and max |diff|, legal lanes both ways, order."""
    a = np.asarray(env['numeric'], np.float64);b = np.asarray(view['numeric'], np.float64);n = len(a)
    differ = (a != b) & ~(np.isnan(a) & np.isnan(b))
    gap = np.where(differ, np.nan_to_num(np.abs(a - b), nan=np.inf, posinf=np.inf), 0.);beyond = gap > tolerance
    cat = np.asarray(env['categorical']) != np.asarray(view['categorical'])
    types = np.asarray(env['token_type']).astype(np.int64) != np.asarray(view['token_type']).astype(np.int64)
    attention = np.asarray(env['attention'], bool) != np.asarray(view['attention'], bool)
    legal_env, legal_view = np.asarray(env['legal'], bool), np.asarray(view['legal'], bool);legal = legal_env != legal_view
    mean = lambda x: x.mean(0) if n else np.zeros(x.shape[1:])
    peak = lambda x, axis: x.max(axis) if n else np.zeros(np.delete(x.shape, axis))
    share = lambda x: float(x.mean()) if n else None
    observation = ~(differ.any((1, 2)) | cat.any((1, 2)) | types.any(1) | attention.any(1))
    exact = dict(numeric=share(~differ.any((1, 2))), numeric_within_tolerance=share(~beyond.any((1, 2))), categorical=share(~cat.any((1, 2))),
        token_type=share(~types.any(1)), attention=share(~attention.any(1)), legal=share(~legal.any(1)), observation=share(observation),
        observation_and_legal=share(observation & ~legal.any(1)))
    return dict(positions=n, exact=exact, observation_exact=observation, numeric_cell_share=mean(differ), numeric_cell_beyond=mean(beyond),
        numeric_cell_max=peak(gap, 0), categorical_cell_share=mean(cat), token_numeric_share=mean(differ.any(2)),
        token_numeric_beyond=mean(beyond.any(2)), token_categorical_share=mean(cat.any(2)), token_type_share=mean(types), attention_share=mean(attention),
        numeric_column_share=mean(differ.any(1)), numeric_column_beyond=mean(beyond.any(1)), numeric_column_max=peak(gap, (0, 1)),
        categorical_column_share=mean(cat.any(1)), legal_lane_env_only=(legal_env & ~legal_view).sum(0), legal_lane_view_only=(~legal_env & legal_view).sum(0),
        order=dict(self=share(up_to_order(env, view, SELF_TOKENS)), opponent=share(up_to_order(env, view, OPPONENT_TOKENS))))


def ranked_cells(stats, kind, labels, names):
    """Every differing (token, column) cell of `kind` ('numeric' or 'categorical'), ranked: beyond-tolerance share, exact share,
    max |diff| (numeric), then token and column."""
    n = stats['positions']
    if kind == 'numeric':share, beyond, gap = stats['numeric_cell_share'], stats['numeric_cell_beyond'], stats['numeric_cell_max']
    else:share = beyond = stats['categorical_cell_share'];gap = None
    cells = sorted(zip(*np.nonzero(share > 0)), key=lambda tc: (-beyond[tc], -share[tc], -(gap[tc] if gap is not None else 0.), tc[0], tc[1]))
    return [dict(token=int(t), token_label=labels[t], column=int(c), column_name=names[c], share=float(share[t, c]),
        positions=int(round(float(share[t, c]) * n)), **(dict(beyond_tolerance_share=float(beyond[t, c]), max_abs_diff=float(gap[t, c])) if gap is not None else {}))
        for t, c in cells]


def cell_examples(env, view, keys, kind, token, column, limit, decode=None):
    """Up to `limit` positions where the cell differs (numeric: the largest |diff| first), with both sides' values."""
    side = 'numeric' if kind == 'numeric' else 'categorical'
    a = np.asarray(env[side])[:, token, column];b = np.asarray(view[side])[:, token, column]
    if kind == 'numeric':
        differ = (a != b) & ~(np.isnan(a) & np.isnan(b));gap = np.nan_to_num(np.abs(a.astype(np.float64) - b), nan=np.inf, posinf=np.inf)
        order = sorted(np.nonzero(differ)[0], key=lambda i: (-gap[i], i))
    else:order = list(np.nonzero(a != b)[0])
    out = []
    for i in order[:limit]:
        item = {k: int(keys[k][i]) for k in ('ordinal', 'round', 'decision_index', 'turn')}
        if kind == 'numeric':item.update(env=float(a[i]), view=float(b[i]))
        else:
            item.update(env=int(a[i]), view=int(b[i]))
            if decode is not None:item.update(env_token=decode(int(a[i])), view_token=decode(int(b[i])))
        out.append(item)
    return out


def kind_summary(env, view, keys, *, tolerance, top, examples, labels, numeric, categorical, decode=None, firsts=20):
    """One kind's comparison (diff_stats) as JSON-ready fields, with examples on the `top` cells of each ranking."""
    s = diff_stats(env, view, tolerance=tolerance)
    numeric_cells, categorical_cells = ranked_cells(s, 'numeric', labels, numeric), ranked_cells(s, 'categorical', labels, categorical)
    for item in numeric_cells[:top]:item['examples'] = cell_examples(env, view, keys, 'numeric', item['token'], item['column'], examples)
    for item in categorical_cells[:top]:item['examples'] = cell_examples(env, view, keys, 'categorical', item['token'], item['column'], examples, decode)
    tokens = [dict(token=t, label=labels[t], numeric_share=float(s['token_numeric_share'][t]), numeric_beyond_tolerance=float(s['token_numeric_beyond'][t]),
        categorical_share=float(s['token_categorical_share'][t]), token_type_share=float(s['token_type_share'][t]), attention_share=float(s['attention_share'][t]))
        for t in range(TOKENS)]
    columns = sorted((dict(column=c, name=numeric[c], share=float(s['numeric_column_share'][c]), beyond_tolerance_share=float(s['numeric_column_beyond'][c]),
        max_abs_diff=float(s['numeric_column_max'][c])) for c in range(NUMERIC) if s['numeric_column_share'][c] > 0),
        key=lambda x: (-x['beyond_tolerance_share'], -x['share'], x['column']))
    category_columns = sorted((dict(column=c, name=categorical[c], share=float(s['categorical_column_share'][c])) for c in range(CATEGORICAL)
        if s['categorical_column_share'][c] > 0), key=lambda x: (-x['share'], x['column']))
    mismatched = [{k: int(keys[k][i]) for k in ('ordinal', 'round', 'decision_index')} for i in np.nonzero(~s['observation_exact'])[0][:firsts]]
    return dict(positions=s['positions'], exact=s['exact'], order=s['order'], tokens=tokens, numeric_columns=columns, categorical_columns=category_columns,
        numeric_cells=numeric_cells, categorical_cells=categorical_cells,
        legal=dict(env_only=[int(x) for x in s['legal_lane_env_only']], view_only=[int(x) for x in s['legal_lane_view_only']]), first_mismatches=mismatched)


def side_of(data, prefix, k=None, mask=None):
    out = {name: data[f'{prefix}_{name}'] if k is None else data[f'{prefix}_{name}'][:, k] for name in SIDES}
    return out if mask is None else {name: v[mask] for name, v in out.items()}


def reason_key(text):
    return str(text).split(': ', 1)[-1][:120] if ': ' in str(text) else str(text)[:120]


def world_summary(rows, data, kinds):
    """Per kind: constructed positions, and why the others are not (the true world's reason or refusals, a sampled world's
    refusals, the position's notes); encoder error bits; the overlay's notes."""
    out = {}
    for k, kind in enumerate(kinds):
        valid = data['view_valid'][:, k];reasons = collections.Counter()
        for row in rows:
            if row.get('views', {}).get(kind) is not None:continue
            info = (row.get('worlds') or {}).get(kind)
            if info is None and kind == 'true_private' and 'true' not in (row.get('views') or {}):why = 'the true world has no view'
            elif info is None:why = 'no world (' + (row.get('notes') or ['not attempted'])[0][:100] + ')'
            elif info.get('reason'):why = info['reason']
            elif info.get('refusals'):why = ';'.join(sorted(map(str, info['refusals'])))[:160]
            else:why = 'view failed: ' + '; '.join(n for n in row.get('notes') or () if n.startswith(kind))[:120]
            reasons[why] += 1
        errors = collections.Counter(int(e) for e, v in zip(data['view_error'][:, k], valid) if v)
        out[kind] = dict(constructed=int(valid.sum()), not_constructed=dict(reasons.most_common()), error_bits=dict(sorted(errors.items())))
    if 'true_private' in kinds:
        notes = collections.Counter(reason_key(n) for r in rows for n in ((r.get('worlds') or {}).get('true_private') or {}).get('notes') or ())
        out['true_private']['overlay_notes'] = dict(notes.most_common(20))
    if 'true' in kinds:
        team = collections.Counter(reason_key(n) for r in rows for n in ((r.get('worlds') or {}).get('true') or {}).get('notes') or ())
        out['true']['team_notes'] = dict(team.most_common(20))
        out['true']['triggered'] = dict(collections.Counter(t for r in rows for t in ((r.get('worlds') or {}).get('true') or {}).get('triggered') or ()).most_common(20))
        seeds = [((r.get('worlds') or {}).get('true') or {}).get('kernel_seed_is_world0s') for r in rows]
        out['true']['kernel_seed_is_world0s'] = dict(collections.Counter(map(str, seeds)))
    return out


def alignment_summary(rows):
    n = len(rows);share = lambda x: x / n if n else None
    equal = collections.Counter(k for r in rows for k, v in (r.get('alignment') or {}).get('env_equals_fp_reply', {}).items() if v)
    return dict(positions=n, turns_equal=share(sum(bool((r.get('alignment') or {}).get('turns_equal')) for r in rows)),
        label_equal=share(sum(bool((r.get('alignment') or {}).get('label_equal')) for r in rows)),
        env_equals_fp_reply={k: share(equal[k]) for k in OBS_KEYS + ('legal',)},
        env_equals_fp_reply_all=share(sum(all((r.get('alignment') or {}).get('env_equals_fp_reply', {0: False}).values()) for r in rows)),
        seat_or_seed_differs=sum(r.get('fp_reply_seat_seed', {}).get('seed') != r.get('seed') or r.get('fp_reply_seat_seed', {}).get('seat') != r.get('seat') for r in rows),
        statuses=dict(collections.Counter(str((r.get('response') or {}).get('status')) for r in rows)))


def capture_counts(summaries):
    out = dict(shards=len(summaries), cells=sum(len(s.get('cells') or ()) for s in summaries), positions=sum(s.get('positions') or 0 for s in summaries),
        sent=sum(s.get('sent') or 0 for s in summaries), skipped_single_legal=sum(s.get('skipped_single_legal') or 0 for s in summaries),
        interrupted=sum(bool(s.get('interrupted')) for s in summaries))
    not_aligned = collections.Counter()
    for s in summaries:not_aligned.update(s.get('not_aligned') or {})
    errors = [e for s in summaries for e in s.get('cell_errors') or ()]
    out.update(not_aligned=dict(not_aligned.most_common()), cell_errors=dict(collections.Counter(e.get('kind') for e in errors).most_common()),
        cell_error_examples=errors[:10])
    return out


def build_report(directories, *, tolerance=1e-5, top=30, examples=3, vocabulary=None):
    rows, summaries, kinds, data = load_captures(directories)
    labels, numeric, categorical, decode = token_labels(), numeric_names(), categorical_names(), vocabulary_decoder(vocabulary)
    keys = {k: data[k] for k in ('ordinal', 'round', 'decision_index', 'turn')};env = side_of(data, 'env')
    out = dict(schema=REPORT_SCHEMA, captures=[str(d) for d in directories], positions=int(len(data['ordinal'])), kinds=list(kinds), tolerance=tolerance,
        capture=capture_counts(summaries), alignment=alignment_summary(rows), worlds=world_summary(rows, data, kinds), parity={}, sampled_vs_true={})
    common = dict(tolerance=tolerance, top=top, examples=examples, labels=labels, numeric=numeric, categorical=categorical, decode=decode)
    for k, kind in enumerate(kinds):
        valid = data['view_valid'][:, k]
        out['parity'][kind] = kind_summary(side_of(data, 'env', mask=valid), side_of(data, 'view', k, valid), {n: v[valid] for n, v in keys.items()}, **common)
    if 'true' in kinds:
        t = kinds.index('true')
        for k, kind in enumerate(kinds):
            if not kind.startswith('sampled_'):continue
            both = data['view_valid'][:, t] & data['view_valid'][:, k]
            s = kind_summary(side_of(data, 'view', t, both), side_of(data, 'view', k, both), {n: v[both] for n, v in keys.items()}, **dict(common, top=10))
            out['sampled_vs_true'][kind] = {f: s[f] for f in ('positions', 'exact', 'order', 'tokens', 'numeric_columns', 'categorical_columns', 'legal')}
            out['sampled_vs_true'][kind].update(numeric_cells=s['numeric_cells'][:10], categorical_cells=s['categorical_cells'][:10])
    return out


def _f(x, digits=3):
    return '-' if x is None else f'{x:.{digits}f}' if isinstance(x, float) else str(x)


def render_markdown(summary, *, top=30):
    """summary.md: alignment, worlds, exact shares, then per kind its ranked cells, tokens and legal lanes."""
    s = summary;out = ['# Opponent-view parity: the search\'s per-world opponent view against Foul Play\'s env observation', '',
        f'Captures: {", ".join(s["captures"])}. Positions (sent decisions where Foul Play was requested in the same round, with a reply row): '
        f'{s["positions"]}. Numeric tolerance for "beyond": {s["tolerance"]}.', '', '## Capture and alignment', '']
    c, a = s['capture'], s['alignment']
    out += [f'- cells {c["cells"]}, decisions sent {c["sent"]}, single-legal (never sent) {c["skipped_single_legal"]}, positions {c["positions"]}, interrupted shards {c["interrupted"]}',
        f'- not aligned (sent, not a position): {c["not_aligned"] or "none"}', f'- cell errors: {c["cell_errors"] or "none"}',
        f'- turns agree (capture, env replay, reply row): {_f(a["turns_equal"])}; Foul Play\'s journal action = the row\'s label: {_f(a["label_equal"])}',
        f'- re-computed env observation = reply row: all keys {_f(a["env_equals_fp_reply_all"])}; per key '
        + ', '.join(f'{k} {_f(v)}' for k, v in a['env_equals_fp_reply'].items()), f'- response statuses: {a["statuses"]}', '', '## Worlds', '',
        '| kind | constructed | not constructed (reasons) | encoder error bits |', '|---|---|---|---|']
    for kind, w in s['worlds'].items():
        out.append(f'| {kind} | {w["constructed"]} | {"; ".join(f"{k}: {v}" for k, v in list(w["not_constructed"].items())[:4]) or "-"} | {w["error_bits"] or "-"} |')
    if 'true' in s['worlds']:out += ['', f'True-team notes: {s["worlds"]["true"].get("team_notes") or "none"}; mapper checks TRIGGERED: {s["worlds"]["true"].get("triggered") or "none"}; '
        f'kernel seed is world 0\'s: {s["worlds"]["true"].get("kernel_seed_is_world0s")}']
    if 'true_private' in s['worlds']:out.append(f'Overlay notes (true_private): {s["worlds"]["true_private"].get("overlay_notes") or "none"}')
    out += ['', '## Exact match (share of constructed positions)', '',
        '| kind | positions | observation + legal | observation | numeric | numeric within tol | categorical | token types | attention | legal | self tokens up to order | opponent tokens up to order |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for kind, p in s['parity'].items():
        e = p['exact']
        out.append(f'| {kind} | {p["positions"]} | {_f(e["observation_and_legal"])} | {_f(e["observation"])} | {_f(e["numeric"])} | {_f(e["numeric_within_tolerance"])} | '
            f'{_f(e["categorical"])} | {_f(e["token_type"])} | {_f(e["attention"])} | {_f(e["legal"])} | {_f(p["order"]["self"])} | {_f(p["order"]["opponent"])} |')
    for kind, p in s['parity'].items():
        out += ['', f'## {kind}: differing features', '', f'Numeric cells, ranked (beyond-tolerance share, then share, then max |diff|); {len(p["numeric_cells"])} cells differ.', '',
            '| # | token | column | name | share | beyond tol | max abs diff | examples (ordinal/round: env -> view) |', '|---|---|---|---|---|---|---|---|']
        for i, x in enumerate(p['numeric_cells'][:top], 1):
            ex = '; '.join(f'{e["ordinal"]}/{e["round"]}: {e["env"]:.6g} -> {e["view"]:.6g}' for e in x.get('examples', ()))
            out.append(f'| {i} | {x["token_label"]} ({x["token"]}) | {x["column"]} | {x["column_name"]} | {_f(x["share"])} | {_f(x["beyond_tolerance_share"])} | {_f(x["max_abs_diff"], 6)} | {ex} |')
        out += ['', f'Categorical cells, ranked by share; {len(p["categorical_cells"])} cells differ.', '',
            '| # | token | column | name | share | examples (ordinal/round: env -> view) |', '|---|---|---|---|---|---|']
        for i, x in enumerate(p['categorical_cells'][:top], 1):
            ex = '; '.join(f'{e["ordinal"]}/{e["round"]}: {e.get("env_token", e["env"])} -> {e.get("view_token", e["view"])}' for e in x.get('examples', ()))
            out.append(f'| {i} | {x["token_label"]} ({x["token"]}) | {x["column"]} | {x["column_name"]} | {_f(x["share"])} | {ex} |')
        out += ['', 'Per token (share of positions where any column differs):', '', '| token | label | numeric | numeric beyond tol | categorical | token type | attention |',
            '|---|---|---|---|---|---|---|']
        for x in p['tokens']:
            if any(x[k] for k in ('numeric_share', 'categorical_share', 'token_type_share', 'attention_share')):
                out.append(f'| {x["token"]} | {x["label"]} | {_f(x["numeric_share"])} | {_f(x["numeric_beyond_tolerance"])} | {_f(x["categorical_share"])} | '
                    f'{_f(x["token_type_share"])} | {_f(x["attention_share"])} |')
        out += ['', 'Numeric columns (any token): ' + ('; '.join(f'{x["column"]} {x["name"]} {_f(x["share"])} (max {_f(x["max_abs_diff"], 6)})' for x in p['numeric_columns'][:top]) or 'none'),
            '', 'Categorical columns (any token): ' + ('; '.join(f'{x["column"]} {x["name"]} {_f(x["share"])}' for x in p['categorical_columns'][:top]) or 'none'),
            '', f'Legal lanes, positions where only the env marks a lane legal: {p["legal"]["env_only"]}; only the view: {p["legal"]["view_only"]}',
            '', f'First mismatching positions (ordinal, round, decision): {[(m["ordinal"], m["round"], m["decision_index"]) for m in p["first_mismatches"][:10]]}']
    if s['sampled_vs_true']:
        out += ['', '## Sampled worlds against the true world (how much sampled hidden information moves the opponent view)', '',
            '| kind | positions | observation | numeric | categorical | legal | top numeric cells | top categorical cells |', '|---|---|---|---|---|---|---|---|']
        for kind, p in s['sampled_vs_true'].items():
            e = p['exact']
            cells = lambda xs: '; '.join(f'{x["token_label"]}.{x["column_name"]} {_f(x["share"])}' for x in xs[:5]) or '-'
            out.append(f'| {kind} | {p["positions"]} | {_f(e["observation"])} | {_f(e["numeric"])} | {_f(e["categorical"])} | {_f(e["legal"])} | '
                f'{cells(p["numeric_cells"])} | {cells(p["categorical_cells"])} |')
    return '\n'.join(out) + '\n'


def report_main(argv):
    p = argparse.ArgumentParser(prog='depthn_opponent_view_parity.py report', description='CPU report over capture directories')
    p.add_argument('--capture', nargs='+', required=True, help='capture output directories (every shard\'s files in them)')
    p.add_argument('--out', required=True, help='where summary.json and summary.md go (neither may exist)')
    p.add_argument('--top', type=int, default=30);p.add_argument('--examples', type=int, default=3)
    p.add_argument('--tolerance', type=float, default=1e-5, help='numeric |diff| above this counts as "beyond tolerance" (default 1e-5)')
    p.add_argument('--vocabulary', help='the champion\'s vocabulary document, to decode categorical ids in examples')
    args = p.parse_args(argv)
    if args.top < 1 or args.examples < 0 or not args.tolerance >= 0:p.error('--top >= 1, --examples >= 0, --tolerance >= 0 required')
    summary = build_report(args.capture, tolerance=args.tolerance, top=args.top, examples=args.examples, vocabulary=args.vocabulary)
    out = Path(args.out);out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'summary.json', summary)
    with (out / 'summary.md').open('x') as stream:stream.write(render_markdown(summary, top=args.top))
    print(dumps(dict(done=str(out), positions=summary['positions'], exact={k: v['exact']['observation_and_legal'] for k, v in summary['parity'].items()})), flush=True)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ('capture', 'report'):
        print(__doc__.split('\n\n')[1], file=sys.stderr);return 2
    return capture_main(argv[1:]) if argv[0] == 'capture' else report_main(argv[1:])


if __name__ == '__main__':raise SystemExit(main())
