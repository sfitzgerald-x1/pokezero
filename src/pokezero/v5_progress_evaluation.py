"""CPU-only progress panels for immutable deferred training milestones."""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import signal
import statistics
import threading
import time

from . import v5_deferred_evaluation as deferred, v5_evaluation as evaluation, v5_run as run

SCHEMA = 'pokezero.v5.progress_showdown.v2'
OPPONENTS = ('foul_play', 'champion_9375', 'max_damage')
SEATS = {'foul_play': ('p1',), 'champion_9375': ('p1', 'p2'), 'max_damage': ('p1', 'p2')}
BANK_SEEDS = 1000
PROGRESS_FP_MS = 100
INTERVAL_GAMES = 1_000_000
Z95 = 1.959963984540054
TERMINAL = ('p1', 'p2', 'tie')
TRAP_SWITCH_SETTING = 'owning_allow_maybe_trapped_switches'


def _ref(value):
    return evaluation.Artifact(**value)


def load_profile(dispatch, protocol):
    """Verify fixed banks before any episode or model setup."""
    document = protocol.document()
    profile = document if document.get('schema') == SCHEMA else _ref(document['progress_showdown']).document()
    if profile['schema'] != SCHEMA or set(profile['banks']) != set(OPPONENTS):
        raise ValueError('all three original Showdown progress banks required')
    request = deferred.read_milestone(dispatch, staged_run_root=profile.get('staged_run_root'))
    if protocol.sha256 != request.protocol_sha256 or request.declared_games % INTERVAL_GAMES:
        raise ValueError('original 1M evaluation request and protocol required')
    declaration = json.loads((request.run_directory / 'declaration.json').read_bytes())
    if declaration['evaluation']['interval_games'] != INTERVAL_GAMES:
        raise ValueError('progress panels require the declared million-game interval')
    inventory = _ref(profile['inventory']).document()
    banks = {name: _ref(ref).document() for name, ref in profile['banks'].items()}
    validate_banks(banks, inventory)
    runtime = _ref(profile['runtime']).document()
    if profile['simulator_sha256'] != runtime['showdown']['compiled_tree_sha256']:
        raise ValueError('progress simulator identity differs from actual runtime')
    declared_games = sum(len(batch['games']) for batches in declaration['iterations'] for batch in batches)
    trap_switches = profile.get(TRAP_SWITCH_SETTING, declared_games == 100_000_000)
    if type(trap_switches) is not bool:
        raise ValueError('deferred evaluation trap switch setting must be boolean')
    actor_law = {
        'policy': 'raw_greedy', 'inference': 'float32', 'runtime': profile['runtime']['sha256'],
        'label_vocabulary': profile['label_vocabulary_sha256'],
    }
    effective_actor_law = dict(actor_law, **{TRAP_SWITCH_SETTING: trap_switches})
    declared_actor_law = effective_actor_law if TRAP_SWITCH_SETTING in profile else actor_law
    if profile['actor_law_sha256'] != run._sha(declared_actor_law):
        raise ValueError('actual greedy FP32 actor law must match the declared identity')
    for field in ('opponent_law_sha256s', 'opponent_sha256s', 'shards'):
        if set(profile[field]) != set(OPPONENTS):
            raise ValueError('explicit identity and concurrency for each opponent required')
    for name, law in profile['opponent_law_sha256s'].items():
        expected = profile['foul_play']['config'] if name == 'foul_play' else {
            'policy': 'untouched_raw_greedy' if name == 'champion_9375' else 'max_damage_seeded_ties',
            'runtime': profile['runtime']['sha256'],
        }
        if name not in OPPONENTS or law != run._sha(expected):
            raise ValueError('actual opponent law must match the declared identity')
    if type(profile['max_requests']) is not int or profile['max_requests'] <= 0:
        raise ValueError('finite positive request limit required')
    if any(type(n) is not int or not 1 <= n <= len(banks[key]) for key, n in profile['shards'].items()):
        raise ValueError('fixed opponent-specific shard counts required')
    # Derive an effective registry law even for an older declaration. Cells
    # played under conservative masking cannot satisfy the enabled evaluation.
    opponent_laws = dict(profile['opponent_law_sha256s'])
    if trap_switches:
        opponent_laws['champion_9375'] = run._sha({
            'policy': 'untouched_raw_greedy',
            'runtime': profile['runtime']['sha256'],
            TRAP_SWITCH_SETTING: True,
        })
    profile = dict(profile, actor_law_sha256=run._sha(effective_actor_law if trap_switches else actor_law),
                   opponent_law_sha256s=opponent_laws, **{TRAP_SWITCH_SETTING: trap_switches})
    return request, profile, banks


def validate_banks(banks, inventory, *, pairs_per_opponent=BANK_SEEDS):
    if inventory.get('complete') is not True:
        raise ValueError('complete training/protected seed and team inventory required')
    seen_seeds = set()
    seen_teams = set()
    protected = set(inventory['packed_team_sha256s'])
    shared = inventory.get('registered_s0_pairs', {})
    ranges = inventory['seed_ranges']
    if any(type(span['start']) is not int or type(span['stop']) is not int
           or not 0 <= span['start'] < span['stop'] <= 2**32 for span in ranges):
        raise ValueError('invalid protected seed inventory range')
    for opponent in OPPONENTS:
        rows = banks[opponent]
        if len(rows) != pairs_per_opponent:
            raise ValueError('fixed registered pair count required')
        start = rows[0]['seed']
        for offset, row in enumerate(rows):
            seed = row['seed']
            teams = row['teams']
            if (type(row['offset']) is not int or row['offset'] != offset
                    or type(seed) is not int or not 0 <= seed < 2**32 or seed != start + offset
                    or seed in seen_seeds or set(teams) != {'p1', 'p2'}
                    or any(type(team) is not str or not team for team in teams.values())):
                raise ValueError('ordered seed bank with both fixed physical teams required')
            team_hashes = {hashlib.sha256(team.encode()).hexdigest() for team in teams.values()}
            approval = shared.get(f'{opponent}:{seed}')
            collision = [span for span in ranges if span['start'] <= seed < span['stop']]
            if collision or team_hashes & protected:
                if (opponent != 'champion_9375' or approval != run._sha(teams)
                        or any(span.get('owner') != 's0_registered_overlap' for span in collision)):
                    raise ValueError('evaluation bank overlaps protected work')
            if len(team_hashes) != 2 or seen_teams & team_hashes:
                raise ValueError('opponent banks require disjoint physical teams')
            seen_teams.update(team_hashes)
            seen_seeds.add(seed)


def _packed(teams):
    from .env import BattleStartOverride
    return BattleStartOverride(teams, observation_format_id='gen3randombattle')


def cell_identity(request, profile, opponent, row, seat):
    """Exact shared identity. Stage, job, lane and milestone labels are excluded."""
    fp_seed = None
    if opponent == 'foul_play':
        fp_root = profile['foul_play']['config']['foulplay_random_seed']
        fp_seed = row['seed'] if fp_root is None else fp_root + row['offset']
    return dict(schema=SCHEMA, simulator=profile['simulator_sha256'], format='gen3randombattle',
                checkpoint_sha256=request.snapshot.checkpoint_sha256, actor_sha256=request.snapshot.archive_sha256,
                actor_law=profile['actor_law_sha256'], opponent=opponent,
                opponent_sha256=profile['opponent_sha256s'][opponent],
                opponent_law=profile['opponent_law_sha256s'][opponent], seed=row['seed'],
                teams_sha256=run._sha(row['teams']), seat=seat, max_requests=profile['max_requests'],
                foulplay_random_seed=fp_seed, foulplay_search_ms=PROGRESS_FP_MS if opponent == 'foul_play' else None)


def _result_from_row(value, identity):
    if value['identity'] != identity or value['status'] not in ('COMPLETE', 'UNKNOWN'):
        raise ValueError('unmatched original evaluation cell result')
    if value['status'] == 'COMPLETE':
        terminal = value['terminal']
        if terminal not in TERMINAL or value['win'] != int(terminal == identity['seat']):
            raise ValueError('scored result requires a matching terminal')
    elif value.get('win') is not None or not value.get('reason'):
        raise ValueError('unknown result cannot carry a score')
    return value


def read_shared_cell(registry, identity):
    """Read the immutable original cell, or None while it is pending."""
    digest = run._sha(identity)
    path = Path(registry) / digest[:2] / digest / 'result.json'
    try:
        return _result_from_row(json.loads(path.read_bytes()), identity)
    except FileNotFoundError:
        return None


def read_shared_start(registry, identity):
    digest = run._sha(identity)
    path = Path(registry) / digest[:2] / digest / 'started.json'
    try:
        value = json.loads(path.read_bytes())
    except FileNotFoundError:
        return None
    if value['identity'] != identity or not math.isfinite(value['started_unix']):
        raise ValueError('unmatched original evaluation start')
    return value


def run_shared_cell(registry, identity, evaluate, *, retry_unknown=False, require_live=None):
    """Coordinate exact-cell reuse with route consumers using the same registry."""
    digest = run._sha(identity)
    directory = Path(registry) / digest[:2] / digest
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'owner.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        if require_live is not None:
            require_live()
        result_path = directory / 'result.json'
        if result_path.exists():
            return _result_from_row(json.loads(result_path.read_bytes()), identity)
        first_start = None
        for attempt in range(2 if retry_unknown else 1):
            started = directory / ('started.json' if attempt == 0 else 'retry-started.json')
            outcome = directory / f'attempt-{attempt}.json'
            if outcome.is_file():
                value = _result_from_row(json.loads(outcome.read_bytes()), identity)
            elif started.exists():
                first = json.loads(started.read_bytes())
                if first['identity'] != identity:
                    raise ValueError('attempt belongs to a different registered cell')
                value = dict(first, status='UNKNOWN', win=None, reason='interrupted original attempt',
                             finished_unix=time.time())
                deferred._publish_equal(outcome, value)
            else:
                first = dict(identity=identity, started_unix=time.time())
                deferred._publish_equal(started, first)
                try:
                    terminal, witness = evaluate()
                    if terminal not in TERMINAL:
                        raise ValueError('no verified terminal')
                    value = dict(first, status='COMPLETE', terminal=terminal, witness=witness,
                                 win=int(terminal == identity['seat']))
                except Exception as error:
                    value = dict(first, status='UNKNOWN', win=None,
                                 reason=type(error).__name__ + ': ' + str(error)[:1024])
                value['finished_unix'] = time.time()
                if require_live is not None:
                    require_live()
                deferred._publish_equal(outcome, value)
            if first_start is None:
                first_start = value['started_unix']
            if value['status'] == 'COMPLETE':
                break
        value = dict(value, started_unix=first_start, prior_unknown_attempts=attempt)
        if require_live is not None:
            require_live()
        deferred._publish_equal(result_path, value)
        return value


def _fp_config(profile, request, bank, seat, fp_cpu=None):
    config = dict(profile['foul_play']['config'])
    config.update(checkpoint=str(request.checkpoint_path), seed_start=bank[0]['seed'], games=len(bank),
                  pokezero_player=seat, foulplay_cpu_affinity=fp_cpu,
                  owning_allow_maybe_trapped_switches=profile[TRAP_SWITCH_SETTING])
    cfg = evaluation._config(config)
    if (cfg.search_time_ms != PROGRESS_FP_MS or cfg.foulplay_mcts_iterations is not None
            or cfg.websocket_host != '127.0.0.1' or cfg.max_decision_rounds != profile['max_requests']):
        raise ValueError('fixed FP100 local Showdown progress controls required')
    return cfg


def _champion(profile, fp_config):
    from .live_protocol_runtime import champion_actor_factory
    from .local_showdown import LocalShowdownConfig
    from .neural_policy import (category_vocab_from_model_config, feature_masks_from_model_config,
                                load_transformer_checkpoint, observation_spec_from_model_config)
    checkpoint = _ref(profile['champion'])
    # The champion factory's legacy validation controls are an actor-construction
    # contract. No Foul Play process or search is used in this H2H opponent.
    cfg = replace(fp_config, checkpoint=checkpoint.verify(), foulplay_cpu_affinity=None, search_time_ms=1000)
    factory = champion_actor_factory(cfg)
    if factory.contract.checkpoint_sha256 != checkpoint.sha256:
        raise ValueError('untouched 9375 is required')
    model, result = load_transformer_checkpoint(checkpoint.path, map_location='cpu')
    model.eval()
    model_cfg = result.model_config
    config = LocalShowdownConfig(showdown_root=cfg.showdown_root, node_binary=cfg.node_binary,
        category_vocab=category_vocab_from_model_config(model_cfg, cfg.showdown_root),
        observation_spec=observation_spec_from_model_config(model_cfg),
        feature_masks=feature_masks_from_model_config(model_cfg), set_belief_source=True)
    return factory, (model, result, config)


def _mixed_game(env, current, opponent, row, seat, max_requests, battle_id):
    from .v5_python_collection import _terminal_winner

    other = 'p2' if seat == 'p1' else 'p1'
    actors = {seat: current.create_actor()}
    if isinstance(opponent, tuple):
        factory, runtime = opponent
        actors[other] = factory.actor(runtime)
    offsets = {owner: 0 for owner in actors}
    rng = {owner: random.Random(run._sha([row['seed'], owner, 'progress-action'])) for owner in ('p1', 'p2')}
    def feed():
        for owner, actor in actors.items():
            lines = tuple(env.owner_lines[owner][offsets[owner]:])
            if lines:
                actor.accept_owner_stream(battle_id=battle_id, owner=owner, generation=actor.generation,
                                          prefix_sha256=actor.prefix_sha256, lines=lines)
                offsets[owner] += len(lines)
    try:
        env.reset_with_start_override(seed=row['seed'], start_override=_packed(row['teams']))
        for i, (owner, actor) in enumerate(actors.items(), 1):
            actor.begin_game(battle_id=battle_id, owner=owner, seed=row['seed'], stream_id=i)
        for decision_index in range(max_requests):
            if env.terminal() is not None:
                break
            feed()
            actions = {}
            choices = {}
            boundaries = {}
            for owner in env.requested_players():
                if owner in actors:
                    actor = actors[owner]
                    boundary = actor.snapshot_boundary()
                    boundaries[owner] = boundary
                    actions[owner] = actor.decide(boundary, rng=rng[owner]).action_index
                    choices[owner] = actor.choice(boundary)
                else:
                    actions[owner] = opponent.select_action(env.observe(owner), rng=rng[owner]).action_index
            if not actions:
                raise ValueError('nonterminal Showdown game has no actionable requests')
            tentative = {owner: choices[owner] for owner, boundary in boundaries.items()
                         if actors[owner].tentative_switch(boundary)}
            routed = (env._choices_for_actions(actions, exclude=tentative) if tentative
                      else env._choices_for_actions(actions))
            if any(routed[owner] != choice for owner, choice in choices.items() if owner not in tentative):
                raise ValueError('owner-specific legal choice routing differs from the simulator')
            if tentative:
                def retry(owner):
                    actor, boundary = actors[owner], boundaries[owner]
                    actor.retry_trapped_switch(boundary, rng=rng[owner])
                    choices[owner] = actor.choice(boundary)
                    return choices[owner]

                env.step_with_guarded_switches(dict(routed, **tentative), tentative, retry,
                                               submission_id=decision_index)
            else:
                env.step(actions)
            for owner, choice in choices.items():
                if actors[owner].allow_maybe_trapped_switches is True:
                    actors[owner].confirm_choice(choice)
        terminal = env.terminal()
        if terminal is None or terminal.capped:
            raise ValueError('Showdown decision cap')
        feed()
        markers = [tuple(line for line in env.owner_lines[s] if line.startswith('|win|') or
                         line == '|tie' or line.startswith('|tie|')) for s in ('p1', 'p2')]
        if len(markers[0]) != 1 or markers[0] != markers[1]:
            raise ValueError('both owning terminal markers must agree')
        winner = _terminal_winner(env.owner_lines, markers[0][0], terminal)
        return ('p1', 'p2', 'tie')[winner], markers[0][0]
    finally:
        close_failed = False
        for actor in actors.values():
            close_failed |= actor.close_game()['status'] == 'FAILED_CLOSED'
        if close_failed:
            raise ValueError('owning Showdown actor failed to close')


def _evaluate_lane(dispatch, protocol, name, registry, selected, cores):
    from . import v5_foulplay
    from .live_protocol_actor import ActorContract
    from .local_showdown import LocalShowdownConfig
    from .observation import ObservationFeatureMasks
    from .policy import MaxDamagePolicy
    from .showdown import V4_REPLAY_OBSERVATION_SPEC
    from .v5_python_collection import OwningEnv

    os.sched_setaffinity(0, set(cores))
    request, profile, banks = load_profile(dispatch, protocol)
    published_root = dispatch.document()['request']['run_directory']

    def require_live():
        deferred.require_live_milestone(request, published_root=published_root)

    if profile['opponent_sha256s']['champion_9375'] != ActorContract.champion().checkpoint_sha256:
        raise ValueError('original 9375 checkpoint identity required')
    shared_pairs = _ref(profile['inventory']).document().get('registered_s0_pairs', {})
    fp_cpu = cores[0] if name == 'foul_play' else None
    if fp_cpu is not None:
        os.sched_setaffinity(0, set(cores[1:]))
    runtime = _ref(profile['runtime']).document()
    policy = dict(runtime=profile['runtime'], model_sha256=request.snapshot.model_sha256,
                  label_vocabulary_sha256=profile['label_vocabulary_sha256'], deterministic=True)
    current = evaluation._factory(policy, request, compute_dtype='float32', cpu_only=True,
                                  allow_maybe_trapped_switches=profile[TRAP_SWITCH_SETTING])
    fp_configs = {seat: _fp_config(profile, request, banks['foul_play'], seat, fp_cpu) for seat in SEATS['foul_play']}
    show = runtime['showdown']
    if any(cfg.showdown_root.resolve() != Path(show['root']).resolve() or cfg.node_binary != show['node']['path']
           for cfg in fp_configs.values()):
        raise ValueError('all progress panels require the same pinned Showdown')
    cfg = fp_configs['p1']
    env = None
    if name == 'foul_play':
        evaluation._require_foulplay_source(profile['foul_play'], {cfg.foulplay_root})
        if profile['opponent_sha256s']['foul_play'] != profile['foul_play']['foulplay_source']['sha256']:
            raise ValueError('actual FP component must match the cell identity')
    else:
        masks = current.contract.encoder_masks()
        masks['opponent_tendency_stats_block'] = masks.pop('stats_block')
        env = OwningEnv(LocalShowdownConfig(showdown_root=cfg.showdown_root, node_binary=cfg.node_binary,
            category_vocab=current.catalog.vocabulary, observation_spec=V4_REPLAY_OBSERVATION_SPEC,
            feature_masks=ObservationFeatureMasks(**masks), set_belief_source=True))
        if name == 'champion_9375':
            opponent = _champion(profile, cfg)
        else:
            if profile['opponent_sha256s'][name] != run._file_sha(Path(__file__).with_name('policy.py')):
                raise ValueError('original MaxDamagePolicy source must match the cell identity')
            opponent = MaxDamagePolicy(showdown_root=cfg.showdown_root, dex=current.catalog.dex)

    async def play_fp(offset, row, seat):
        config = fp_configs[seat]
        report = await v5_foulplay.run_panel(config, current, enable_v5=True, game_offsets=(offset,),
                                             packed_starts={offset: _packed(row['teams'])})
        result = report['rows'][0]
        if result['status'] != 'COMPLETE':
            raise ValueError(result['reason'] or result['status'])
        witness = result['source_game']['terminal_witness']
        return witness['winner_player'] or 'tie', dict(terminal_witness=witness,
                                                       source_result_sha256=result['source_result_sha256'])

    counts = dict(active_shared_cells=0, attempted=0, complete=0, unknown=0)
    try:
        for offset in selected:
            row = banks[name][offset]
            for seat in SEATS[name]:
                identity = cell_identity(request, profile, name, row, seat)
                if name == 'foul_play':
                    play = lambda: asyncio.run(play_fp(offset, row, seat))
                else:
                    play = lambda: _mixed_game(env, current, opponent, row, seat, profile['max_requests'],
                                               run._sha(identity))
                def attempt():
                    counts['attempted'] += 1
                    return play()
                # A registered S0 cell follows its original no-replay rule.
                result = run_shared_cell(registry, identity, attempt,
                    retry_unknown=f'{name}:{row["seed"]}' not in shared_pairs, require_live=require_live)
                field = 'active_shared_cells' if result is None else result['status'].lower()
                counts[field] += 1
    finally:
        if env is not None:
            env.close()
    return counts


def shard_offsets(count, shards, shard, lanes):
    if not 1 <= shards <= count or not 0 <= shard < shards or not 1 <= lanes <= 4:
        raise ValueError('valid original shard and 1–4 lanes required')
    offsets = list(range(shard, count, shards))
    return [offsets[i::lanes] for i in range(lanes)]


def execute_shard(dispatch, protocol, opponent, registry, *, shard, lanes):
    if os.environ.get('JAX_PLATFORMS') != 'cpu' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('launch separate CPU-only evaluator on jokla')
    _, profile, banks = load_profile(dispatch, protocol)
    _check_registry(profile, registry)
    groups = shard_offsets(len(banks[opponent]), profile['shards'][opponent], shard, lanes)
    cores = sorted(os.sched_getaffinity(0))
    if len(cores) < 4 * lanes:
        raise ValueError('four allocated CPUs per active evaluation lane required')
    # Spread lanes across the inherited cpuset in quota-only pods. Explicitly
    # assigned 4*lanes cpusets retain the same four-way per-process partition.
    salt = str(os.environ.get('V2_RUN_NAME', shard))
    offset = int(run._sha(salt)[:8], 16) % len(cores)
    cores = cores[offset:] + cores[:offset]
    with ProcessPoolExecutor(max_workers=lanes, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(_evaluate_lane, dispatch, protocol, opponent, registry, selected, cores[4*i:4*i+4])
                   for i, selected in enumerate(groups) if selected]
        combined = dict(active_shared_cells=0, attempted=0, complete=0, unknown=0)
        for future in futures:
            for field, count in future.result().items():
                combined[field] += count
        return combined


def measure_shard(dispatch, protocol, opponent, registry, *, shard, lanes):
    """Record whole-job work/memory for the first sizing run and later audits."""
    cgroup = Path('/sys/fs/cgroup')
    peak = cgroup / 'memory.peak'
    stop = threading.Event()
    sampled_peak = [0]
    def sample():
        while not stop.is_set():
            try:
                sampled_peak[0] = max(sampled_peak[0], int((cgroup/'memory.current').read_text()))
            except OSError:
                return
            stop.wait(1)
    sampler = None
    if not peak.is_file():
        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
    clock = time.monotonic()
    try:
        result = execute_shard(dispatch, protocol, opponent, registry, shard=shard, lanes=lanes)
    finally:
        stop.set()
        if sampler is not None:
            sampler.join()
    memory_peak = int(peak.read_text()) if peak.is_file() else sampled_peak[0] or None
    return dict(result, status='QUEUED' if result['active_shared_cells'] else 'COMPLETE',
                opponent=opponent, shard=shard, lanes=lanes, memory_peak_bytes=memory_peak,
                wall_seconds=time.monotonic()-clock)


def _interval(values):
    """Pointwise paired-cluster normal interval with a Jeffreys regularizer."""
    n = len(values)
    center = (sum(values) + .5) / (n + 1)
    # One half-observation at each endpoint keeps all-win/all-loss intervals finite in width.
    variance = (sum((x-center)**2 for x in values) + .5*center**2 + .5*(1-center)**2) / n
    half_width = Z95 * math.sqrt(variance / (n + 1))
    return max(0., center - half_width), min(1., center + half_width)


def summarize_panel(pairs):
    width = len(pairs[0]) if pairs else 0
    if width not in (1, 2) or any(len(pair) != width for pair in pairs):
        raise ValueError('statistics require the full fixed seat roster')
    seen = [value['win'] for pair in pairs for value in pair if value and value['status'] == 'COMPLETE']
    unknown = sum(value is not None and value['status'] == 'UNKNOWN' for pair in pairs for value in pair)
    pending = width * len(pairs) - len(seen) - unknown
    low, high = [], []
    for pair in pairs:
        wins = sum(value['win'] for value in pair if value and value['status'] == 'COMPLETE')
        missing = sum(not value or value['status'] != 'COMPLETE' for value in pair)
        low.append(wins / width)
        high.append((wins + missing) / width)
    bounds = [statistics.mean(low), statistics.mean(high)]
    if not seen:
        ci = None
    elif unknown or pending:
        worst = Z95 / (2 * math.sqrt(len(pairs)))
        ci = [max(0., bounds[0] - worst), min(1., bounds[1] + worst)]
    else:
        ci = list(_interval(low))
    return dict(games_completed=len(seen), unknown=unknown, pending=pending, games_declared=width*len(pairs),
                win_rate=statistics.mean(seen) if seen else None, ci95=ci,
                prior_unknown_attempts=sum(value.get('prior_unknown_attempts', 0)
                                           for pair in pairs for value in pair if value),
                identification_bounds=bounds)


def summarize(dispatch, protocol, registry, *, output=None):
    request, profile, banks = load_profile(dispatch, protocol)
    _check_registry(profile, registry)
    panel = {}
    all_rows = []
    starts = []
    for name in OPPONENTS:
        pairs = []
        for row in banks[name]:
            pair = []
            for seat in SEATS[name]:
                identity = cell_identity(request, profile, name, row, seat)
                result = read_shared_cell(registry, identity)
                pair.append(result)
                started = result or read_shared_start(registry, identity)
                if started is not None:
                    starts.append(started['started_unix'])
            pairs.append(pair)
        panel[name] = summarize_panel(pairs)
        all_rows.extend(row for pair in pairs for row in pair if row)
    document = dispatch.document()
    published = document['published_unix'] if 'published_unix' in document else Path(dispatch.path).stat().st_mtime
    terminal = all(value['pending'] == 0 for value in panel.values())
    finished = max((x['finished_unix'] for x in all_rows), default=published) if terminal else time.time()
    first = min(starts, default=finished)
    result = dict(schema=SCHEMA, declared_games=request.declared_games, protocol_sha256=protocol.sha256,
                  declaration_sha256=request.declaration_sha256, dispatch_sha256=dispatch.sha256,
                  panel=panel, published_unix=published, observed_unix=time.time(), complete=terminal,
                  work_started=bool(starts), queue_wait_seconds=max(0, first-published),
                  eval_lag_seconds=max(0, finished-published))
    if profile.get('staged_run_root') is not None:
        deferred.require_live_milestone(request, published_root=document['request']['run_directory'])
    else:
        deferred.require_live_milestone(request)
    if output is not None:
        deferred._publish_equal(output, result)
    return result


def _check_registry(profile, registry):
    declared = Path(profile['shared_cell_registry'])
    if not declared.is_absolute() or Path(registry).resolve() != declared.resolve():
        raise ValueError('route and progress workers must use the same registered cell store')


def pending_shards(dispatch, protocol, registry):
    request, profile, banks = load_profile(dispatch, protocol)
    _check_registry(profile, registry)
    result = []
    for name in OPPONENTS:
        count = profile['shards'][name]
        for shard in range(count):
            pending = sum(read_shared_cell(registry, cell_identity(request, profile, name, row, seat)) is None
                          for row in banks[name][shard::count] for seat in SEATS[name])
            if pending:
                result.append(dict(opponent=name, shard=shard, cells_pending=pending))
    return result


def _run_owned_shard(dispatch, protocol, registry, opponent, shard, lanes):
    os.setsid()
    print(json.dumps(measure_shard(dispatch, protocol, opponent, registry,
                                  shard=shard, lanes=lanes)), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('dispatch', 'dispatch-sha256', 'protocol', 'protocol-sha256', 'registry'):
        parser.add_argument('--'+arg, required=True)
    commands = parser.add_subparsers(dest='command', required=True)
    worker = commands.add_parser('evaluate')
    worker.add_argument('--opponent', choices=OPPONENTS, required=True)
    worker.add_argument('--shard', type=int, required=True)
    worker.add_argument('--lanes', type=int, default=4)
    worker.add_argument('--wall-seconds', type=int, default=3600)
    commands.add_parser('summarize').add_argument('--output', type=Path, required=True)
    commands.add_parser('pending')
    args = parser.parse_args(argv)
    dispatch = evaluation.Artifact(args.dispatch, args.dispatch_sha256)
    protocol = evaluation.Artifact(args.protocol, args.protocol_sha256)
    if args.command == 'evaluate':
        if not 1 <= args.wall_seconds <= 7200:
            parser.error('finite shard wall time must be at most 7200 seconds')
        for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[variable] = '1'
        process = multiprocessing.get_context('spawn').Process(target=_run_owned_shard, args=(
            dispatch, protocol, args.registry, args.opponent, args.shard, args.lanes))
        process.start()
        process.join(args.wall_seconds)
        if process.is_alive():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                process.kill()
            process.join()
            print(json.dumps(dict(status='UNKNOWN', reason='shard wall deadline', shard=args.shard)))
            return 124
        return 0 if process.exitcode == 0 else 1
    result = (summarize(dispatch, protocol, args.registry, output=args.output)
              if args.command == 'summarize' else pending_shards(dispatch, protocol, args.registry))
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
