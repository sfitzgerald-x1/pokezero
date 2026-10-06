"""Create-only continuations of the three already observed action disagreements.

Usage: register with a parent deep-screen registration/output and a fresh output
directory, then run/readout against that output. No deployment or training.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
ORDINALS = (3, 6, 8)
PLANNERS = ('deep_incumbent', 'paper_reference')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=lambda v: asdict(v))
        stream.write('\n')


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()


def reference_statistics_checkpoint(snapshot):
    """Auditable Q/N/M/F after an accepted decision, without latent world state."""
    return dict(battle_id=snapshot.battle_id, version=snapshot.version,
        faint_floor=snapshot.faint_floor, acknowledged=snapshot.acknowledged,
        rows=[dict(key_hex=row.state.key.hex(), actions=row.state.actions,
            faint_count=row.state.faint_count, visits=row.visits,
            totals=row.totals, count=row.count) for row in snapshot.rows])


def restore_reference_checkpoint(value):
    from pokezero.mcts_eval.paper_reference import DecisionState
    from pokezero.mcts_eval.paper_reference_exchange import MasterSnapshot, Statistics
    return MasterSnapshot(value['battle_id'], value['version'], value['faint_floor'],
        tuple(Statistics(DecisionState(bytes.fromhex(row['key_hex']), tuple(row['actions']), row['faint_count']),
            tuple(row['visits']), tuple(row['totals']), row['count']) for row in value['rows']),
        tuple((worker, sequence) for worker, sequence in value['acknowledged']))


def register_resume(previous, inputs):
    identity = 'root-03-paper_reference-search-r0'
    result_path = previous / (identity + '.json')
    result = json.loads(result_path.read_text())
    if result['status'] != 'REFUSED' or result['planner'] != 'paper_reference':
        raise RuntimeError('only the explicitly refused paper continuation can resume')
    paths = sorted((previous / identity).glob('boundary-*.json'))
    rows = [json.loads(p.read_text()) for p in paths]
    if not rows or [r['boundary'] for r in rows] != list(range(len(rows))):
        raise RuntimeError('resume requires a complete played prefix')
    if rows[0]['actions'][result['subject']] != result['first_action']:
        raise RuntimeError('resume first action drift')
    for path in paths:
        if sha(path) != result['step_hashes'].get(path.name):
            raise RuntimeError('resume prefix hash drift')
        inputs[str(path)] = sha(path)
    latest = next(p for p, r in reversed(list(zip(paths, rows))) if result['subject'] in r['actions'])
    checkpoint = json.loads(latest.read_text())['evidence'][result['subject']]['search_evidence']['statistics_checkpoint']
    restore_reference_checkpoint(checkpoint)  # Validate before registration.
    inputs[str(result_path)] = sha(result_path)
    return {identity: dict(result_path=str(result_path), checkpoint_step=str(latest),
        prefix_steps={p.name: str(p) for p in paths}, start_boundary=len(rows),
        prior_selections=sum(result['subject'] in r['actions'] for r in rows[1:]),
        recovery='accepted aggregate Q/N/M/F only; fresh processes/P caches/RNG sessions; failed partial work excluded')}


def retained_cells(previous, roots, inputs):
    """Link only complete, identity-matched cells; preserve refusals as history.

    A new registration can repair the SAME refused question, never drop it or
    overwrite the earlier files. No incomplete search tree is claimed restored.
    """
    path = previous / 'registration.json'
    old = json.loads(path.read_text())
    if old['roots'] != roots or old['replicates'] != [0, 1] or old['planners'] != list(PLANNERS):
        raise RuntimeError('retained continuation roster drift')
    inputs[str(path)] = sha(path)
    inputs.update(old['source_hashes'])
    inputs.update(old['retained_input_hashes'])
    retained, refused = dict(old.get('retained_cells', {})), list(old.get('preserved_noncomplete_history', []))
    for cell_path in sorted(previous.glob('root-*.json')):
        cell = json.loads(cell_path.read_text())
        expected = next((r for r in roots if r['ordinal'] == cell['ordinal']), None)
        identity = f"root-{cell['ordinal']:02d}-{cell['planner']}-{cell['mode']}-r{cell['replicate']}"
        if (expected is None or cell['identity'] != identity or cell_path.stem != identity
                or cell['decision_id'] != expected['decision_id']
                or cell['first_choice'] != expected['first_choices'][cell['planner']]
                or cell['mode'] not in ('raw', 'search') or cell['replicate'] not in (0, 1)
                or cell['registration_sha256'] != sha(path)):
            raise RuntimeError('retained cell provenance drift')
        inputs[str(cell_path)] = sha(cell_path)
        for name, digest in cell['step_hashes'].items():
            if Path(name).name != name or sha(previous / identity / name) != digest:
                raise RuntimeError('retained decision evidence drift')
            inputs[str(previous / identity / name)] = digest
        if cell['status'] == 'COMPLETE':
            retained[identity] = {'path': str(cell_path), 'sha256': sha(cell_path),
                'registration_sha256': sha(path), 'step_dir': str(previous / identity)}
        else:
            refused.append({'identity': identity, 'path': str(cell_path), 'sha256': sha(cell_path),
                'status': cell['status'], 'error': cell.get('error')})
    return retained, refused


def register(args):
    if git('status', '--porcelain'):
        raise RuntimeError('commit reviewed source before registration')
    parent = json.loads(args.parent_registration.read_text())
    original = parent['original']
    inputs = {str(args.parent_registration): sha(args.parent_registration),
        **parent['retained_input_hashes']}
    for path, expected in inputs.items():
        if sha(path) != expected:
            raise RuntimeError('parent retained input drift: ' + path)
    roots = []
    for ordinal in ORDINALS:
        cells = {}
        for arm in ('deep_incumbent', 'paper_reference_10s', 'paper_reference_30s'):
            path = args.parent_output / f'timing-{ordinal:02d}-{arm}.json'
            cell = json.loads(path.read_text())
            if cell['status'] != 'COMPLETE' or cell['decision_id'] != original['decision_ids'][ordinal]:
                raise RuntimeError('parent timing identity/terminal drift')
            inputs[str(path)] = sha(path)
            cells[arm] = cell
        if cells['paper_reference_10s']['choice'] != cells['paper_reference_30s']['choice']:
            raise RuntimeError('reference 10s/30s no longer agree')
        if cells['deep_incumbent']['choice'] == cells['paper_reference_10s']['choice']:
            raise RuntimeError('registered root is not a disagreement')
        roots.append({'ordinal': ordinal, 'decision_id': original['decision_ids'][ordinal],
            'first_choices': {'deep_incumbent': cells['deep_incumbent']['choice'],
                'paper_reference': cells['paper_reference_10s']['choice']}})
    retained, refused = retained_cells(args.reuse_complete_from, roots, inputs) if args.reuse_complete_from else ({}, [])
    resumes = register_resume(args.resume_refused_from, inputs) if args.resume_refused_from else {}
    args.output.mkdir(parents=True, exist_ok=False)
    modules = [*sorted((REPO / 'src/pokezero').rglob('*.py')), Path(__file__),
        *sorted((REPO / 'scripts').glob('battle_bridge*.mjs'))]
    manifest = dict(schema='pokezero.fixed-action-followthrough.v1', identity=args.output.name,
        registered_at=datetime.now(timezone.utc).isoformat(), source_commit=git('rev-parse', 'HEAD'),
        original=original, roots=roots, replicates=[0, 1], planners=list(PLANNERS),
        search_game_denominator=12, raw_control_game_denominator=12,
        retained_cells=retained, preserved_noncomplete_history=refused,
        resume_cells=resumes,
        source_hashes={str(path): sha(path) for path in modules}, retained_input_hashes=inputs,
        nominal_decision_seconds=10., max_boundaries=200, global_admission_seconds=7200,
        per_game_wall_seconds=2400, per_decision_safety_seconds=120,
        opponent='unchanged champion full masked policy sample at EVERY opponent request',
        chance='domain separated per root/replicate/boundary; same seed across arms',
        first_action='fixed to historical selected Showdown choice; never reselected/redrawn',
        reference=dict(workers=20, batch=10, alpha=.5, beta=1.,
            allow_earlier_compatible_template=True, max_known_set_draws=128,
            public_consumed_item_history=True,
            conditional_public_encore=True,
            checkpoint_aggregate_statistics_after_each_accepted_decision=True,
            persistent_tree_within_continuation=True, historical_tree_resurrected=False),
        incumbent=dict(depth=6, sims=4096, batch=16, worlds=4, model_priors=True,
            use_opponent_priors=False, leaf_eval='model', early_stop=False,
            model_world_workers=1, threads=1, model_decision_time_ms=10000,
            model_native_batch_guard_ms=64),
        limitations=['selected diagnostic positions, not representative strength',
            'two replicates per position do not establish superiority/equivalence',
            '10s nominal budgets, not matched completed work or equal CPU allocation',
            'incumbent follow-through is deadline-limited, not the historical full 16384 iterations',
            'no original root tree restored; reference retains statistics from later actual decisions',
            'extra hidden-set templates are a disclosed conditioning adaptation, not exact posterior',
            'corpus exact opponent HP remains a controlled information deviation',
            'opponent is not Foul Play; caps/refusals remain missing, never dropped'],
        hardware={'accelerator': 'cpu', 'logical_cpus': os.cpu_count()},
        whole_policy_strength_qualified=False, no_retraining_or_cluster_mutation=True)
    if retained or refused:
        manifest['limitations'].append('completed cells retained across disclosed materializer repair; only refused/unattempted cells rerun')
    if resumes:
        manifest['limitations'].append('paper continuation resumes after a disclosed Encore/last-move repair, preserving prior actions and accepted aggregate statistics; not a homogeneous-source rerun')
    save(args.output / 'registration.json', manifest)
    print(json.dumps({'status': 'REGISTERED', 'roots': ORDINALS, 'search_games': 12,
        'raw_controls': 12, 'output': str(args.output)}), flush=True)


def verify(m):
    if git('rev-parse', 'HEAD') != m['source_commit'] or git('status', '--porcelain'):
        raise RuntimeError('registered source identity/cleanliness drift')
    original = m['original']
    for path, expected in {**m['source_hashes'], **m['retained_input_hashes'],
            original['checkpoint_path']: original['checkpoint_sha256']}.items():
        if sha(path) != expected:
            raise RuntimeError('registered input drift: ' + path)
    for item in original['native_artifacts']:
        if sha(item['path']) != item['sha256']:
            raise RuntimeError('native artifact drift')
    from pokezero.randbat import load_gen3_randbat_source_cached
    if load_gen3_randbat_source_cached(original['showdown_root']).metadata.source_hash != original['showdown_set_source_hash']:
        raise RuntimeError('simulator source drift')
    if subprocess.check_output(['git', '-C', original['showdown_root'], 'rev-parse', 'HEAD'], text=True).strip() != original['showdown_commit']:
        raise RuntimeError('simulator commit drift')
    if subprocess.check_output(['git', '-C', original['showdown_root'], 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('dirty simulator source')


def prepare(live, record, records):
    from pokezero.mcts_eval.source_root_replay import source_bound_replay_prefix
    from pokezero.mcts_eval.policy_opponent_live_replay import validate_live_root
    from pokezero.public_replay_materializer import replay_public_action_rounds
    prefix = source_bound_replay_prefix(record, source_records=tuple(records.values()))
    replay = replay_public_action_rounds(live, seed=record.seed, format_id=record.format_id,
        public_action_rounds=prefix.public_action_rounds, start_override=None)
    return validate_live_root(live, record, replay)


def timeout(_sig, _frame):
    raise TimeoutError('registered per-decision safety cap; no raw fallback')


def run(args, m):
    import torch
    from pokezero.collection import env_config_with_policy_spec_masks
    from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv, showdown_choice_for_action
    from pokezero.mcts_eval.followthrough import masked_argmax, play_continuation
    from pokezero.mcts_eval.head_to_head import public_only_context
    from pokezero.mcts_eval.manifest import SearchConfig
    from pokezero.mcts_eval.paper_reference import ReferenceConfig
    from pokezero.mcts_eval.paper_reference_parallel import ParallelTrajectorySearch
    from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest, ShowdownWorkerFactory
    from pokezero.mcts_eval.paper_reference_showdown import ChampionEvaluator, decision_state
    from pokezero.mcts_eval.policy_opponent_profile import make_profile_decider, validate_selection
    from pokezero.mcts_eval.policy_opponent_roster import load_frozen_roster, verify_source_files
    from pokezero.mcts_eval.policy_opponent_source_prefixes import load_source_records
    from pokezero.mcts_eval.resolver import resolve_checkpoint_contract
    from pokezero.neural_policy import load_transformer_policy
    from pokezero.policy import PolicyContext
    from pokezero.trajectory import BattleTrajectory, TrajectoryStep

    verify(m)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    original = m['original']
    roster = load_frozen_roster(Path(original['roster_path']), expected_sha256=original['roster_sha256'])
    verify_source_files(Path(original['source_root']), roster)
    records = load_source_records(Path(original['source_root']), roster)
    cfg = env_config_with_policy_spec_masks(LocalShowdownConfig(
        showdown_root=Path(original['showdown_root']), set_belief_source=True),
        [f"neural:{original['checkpoint_path']}"], context='fixed-action follow-through diagnostic')
    evaluator = ChampionEvaluator(load_transformer_policy(Path(original['checkpoint_path']), device='cpu',
        deterministic=True, exploration_epsilon=0., sampling_temperature=1., family_gated_selection=False))
    contract = resolve_checkpoint_contract(original['checkpoint_path'], expected_sha256=original['checkpoint_sha256'],
        showdown_root=original['showdown_root'], showdown_source_sha256=original['showdown_set_source_hash'])
    search_cfg = SearchConfig(depth=6, sims=4096, batch=16, worlds=4, inference_mode='local')
    decider = make_profile_decider(contract, original['showdown_root'], arm='incumbent_mcts',
        mode='matched_deadline', opponent_seed=0, deadline_ms=10000, native_batch_guard_ms=64)
    native = decider._policy_for(search_cfg)
    live = LocalShowdownEnv(cfg)
    pool = None
    started = time.perf_counter()
    signal.signal(signal.SIGALRM, timeout)
    try:
        if not args.raw_only:
            pool = ParallelTrajectorySearch(ReferenceConfig(.5, 1.), ShowdownWorkerFactory(
                original['checkpoint_path'], original['checkpoint_sha256'], original['showdown_root'],
                original['showdown_set_source_hash'], allow_earlier_compatible_template=True,
                max_known_set_draws=128))
            print(json.dumps({'status': 'POOL_READY', 'startup_seconds': pool.startup_seconds}), flush=True)
        for replicate in m['replicates']:
            for root in m['roots']:
                ordinal = root['ordinal']
                record = records[roster['profile_roots'][ordinal]['source_relative_path']]
                if record.decision_id != root['decision_id']:
                    raise RuntimeError('fixed root identity drift')
                root_validation = prepare(live, record, records)
                snapshot = live.snapshot()  # Oracle-owned; never passed to either selector.
                lookup = {row['showdown_choice']: row['action_index'] for row in root_validation['legal_choices']}
                order = list(PLANNERS) if (replicate + ordinal) % 2 else list(reversed(PLANNERS))
                for planner in order:
                    modes = ('raw',) if args.raw_only else ('raw', 'search')
                    for mode in modes:
                        identity = f'root-{ordinal:02d}-{planner}-{mode}-r{replicate}'
                        if identity in m.get('retained_cells', {}):
                            continue
                        result_path = args.output / (identity + '.json')
                        if result_path.exists():
                            existing = json.loads(result_path.read_text())
                            if existing['status'] != 'COMPLETE' or existing['decision_id'] != record.decision_id:
                                raise RuntimeError('extant noncomplete cell may not be overwritten/retried')
                            continue
                        step_dir = args.output / identity
                        step_dir.mkdir(exist_ok=False)
                        if time.perf_counter() - started > m['global_admission_seconds']:
                            raise TimeoutError('registered global admission cap; unfinished roster retained')
                        live.restore(snapshot)
                        subject = record.acting_player
                        battle_id = f'followthrough:{identity}'
                        trajectory = BattleTrajectory(battle_id, record.format_id, record.seed)
                        first = lookup[root['first_choices'][planner]]
                        resume = m.get('resume_cells', {}).get(identity)
                        if resume:
                            if planner != 'paper_reference' or mode != 'search':
                                raise RuntimeError('invalid resume planner')
                            for name, path in resume['prefix_steps'].items():
                                row = json.loads(Path(path).read_text())
                                if set(row['actions']) != set(live.requested_players()):
                                    raise RuntimeError('resume request boundary drift')
                                for player, action in row['actions'].items():
                                    if not live.observe(player).legal_action_mask[action]:
                                        raise RuntimeError('resume played action is no longer legal')
                                live.reseed_simulator_rng(row['chance_seed'])
                                live.step(row['actions'])
                            row = json.loads(Path(resume['checkpoint_step']).read_text())
                            checkpoint = restore_reference_checkpoint(row['evidence'][subject]['search_evidence']['statistics_checkpoint'])
                            if checkpoint.battle_id != battle_id:
                                raise RuntimeError('resume checkpoint battle drift')
                            pool.restore_statistics(checkpoint)
                        if mode == 'search' and planner == 'deep_incumbent':
                            native.reset()
                            native.warm_public_prefix_for_replay(battle_id=battle_id, player_id=subject,
                                decision_round_index=record.turn_index,
                                public_materialization_state=live.public_materialization_state(subject))

                        def select(observation, boundary, seed):
                            if mode == 'raw':
                                return masked_argmax(evaluator, observation)
                            signal.alarm(m['per_decision_safety_seconds'])
                            begun = time.perf_counter()
                            try:
                                if planner == 'deep_incumbent':
                                    context = public_only_context(PolicyContext(player_id=subject,
                                        decision_round_index=record.turn_index + boundary, battle_id=battle_id,
                                        format_id=record.format_id, seed=record.seed, observation=observation,
                                        requested_players=tuple(live.requested_players()), trajectory=trajectory,
                                        requested_legal_action_masks={subject: tuple(observation.legal_action_mask)},
                                        requested_observations={subject: observation},
                                        public_materialization_state=live.public_materialization_state(subject)))
                                    before = decider._snapshot_stats(native)
                                    decision = native.select_action_with_context(context, rng=random.Random(seed))
                                    after = decider._snapshot_stats(native)
                                    index = int(decision.action_index)
                                    evidence = dict(root_action=showdown_choice_for_action(live._state_for_player(subject), index),
                                        max_depth_reached=decider._changed_depth(before['depth_reached_histogram'], after['depth_reached_histogram']),
                                        fallbacks=after['fallback_decisions']-before['fallback_decisions'],
                                        prior_fallbacks=after['prior_fallbacks']-before['prior_fallbacks'],
                                        invalid_actions=int(not observation.legal_action_mask[index]),
                                        total_iterations=after['total_iterations']-before['total_iterations'],
                                        model_evals=after['model_evals']-before['model_evals'],
                                        engine_mcts=dict(decision.metadata.get('engine_mcts', {})))
                                    # A singleton legal support is genuinely forced, not a raw fallback.
                                    if sum(observation.legal_action_mask) > 1:
                                        validate_selection(evidence, arm='incumbent_mcts', mode='matched_deadline',
                                            config=search_cfg, mask=observation.legal_action_mask, opponent_seed=seed,
                                            deadline_ms=10000, native_batch_guard_ms=64)
                                    elif evidence['fallbacks'] or evidence['prior_fallbacks']:
                                        raise RuntimeError('forced own request fell back')
                                else:
                                    request = PublicRootRequest.capture(live.public_materialization_state(subject), observation)
                                    root_state = decision_state(observation, player=subject)
                                    remaining = m['nominal_decision_seconds'] - (time.perf_counter() - begun)
                                    if remaining <= 0:
                                        raise TimeoutError('reference deadline expired during capture')
                                    measured = pool.search(request, root_state, battle_id=battle_id, seed=seed,
                                        deadline_seconds=remaining)
                                    index = int(measured.result.action.split(':')[1])
                                    draws = [d for receipt in measured.worker_receipts for d in receipt['evidence']['draws']]
                                    if not draws or not all(d['status'] == 'ROOT_VALIDATED' and d['released'] for d in draws):
                                        raise RuntimeError('reference world validation/release failed')
                                    if len(draws) != measured.result.world_draws or len(set(measured.worker_pids)) != 20:
                                        raise RuntimeError('reference world/worker denominator drift')
                                    evidence = asdict(measured)
                                    evidence['statistics_checkpoint'] = reference_statistics_checkpoint(pool._master.snapshot())
                                return index, dict(selector=planner, searched_every_later_own_request=True,
                                    elapsed_seconds=time.perf_counter()-begun, seed=seed, search_evidence=evidence)
                            finally:
                                signal.alarm(0)

                        def emit(row):
                            save(step_dir / f"boundary-{row['boundary']:03d}.json", row)
                            # Only own history may enter later subject context.
                            if subject in row['actions']:
                                obs = live.observe(subject)
                                trajectory.append(TrajectoryStep(player_id=subject,
                                    turn_index=record.turn_index + row['boundary'], observation=obs,
                                    legal_action_mask=tuple(obs.legal_action_mask), action_index=row['actions'][subject]))
                            own = row['evidence'].get(subject, {})
                            print(json.dumps({'cell': identity, 'boundary': row['boundary'],
                                'own_selector': own.get('selector'), 'actions': row['actions'],
                                'decision_seconds': own.get('elapsed_seconds')}), flush=True)

                        result = dict(identity=identity, decision_id=record.decision_id, ordinal=ordinal,
                            replicate=replicate, planner=planner, mode=mode, first_action=first,
                            first_choice=root['first_choices'][planner], subject=subject,
                            registration_sha256=sha(args.output / 'registration.json'),
                            root_validation=root_validation)
                        if resume:
                            result['retained_prefix_steps'] = resume['prefix_steps']
                            result['recovery'] = resume['recovery']
                        try:
                            result.update(play_continuation(live, subject=subject, first_action=first,
                                decision_id=record.decision_id, replicate=replicate, subject_selector=select,
                                opponent_evaluator=evaluator, emit=emit, max_boundaries=m['max_boundaries'],
                                wall_seconds=m['per_game_wall_seconds'],
                                start_boundary=resume['start_boundary'] if resume else 0,
                                prior_selections=resume['prior_selections'] if resume else 0))
                            if resume:
                                result['resumed_phase_elapsed_seconds'] = result['elapsed_seconds']
                                result['elapsed_seconds'] = None  # Do not invent total pre-failure wall time.
                        except Exception as error:
                            result.update(status='REFUSED', signed_outcome=None,
                                error=f'{type(error).__name__}: {error}',
                                failure_evidence=getattr(error, 'evidence', None))
                        result['step_hashes'] = {p.name: sha(p) for p in step_dir.glob('*.json')}
                        save(result_path, result)
                        print(json.dumps({k: result.get(k) for k in ('identity', 'status', 'winner',
                            'signed_outcome', 'followthrough_decisions', 'elapsed_seconds', 'error')}), flush=True)
                        if result['status'] != 'COMPLETE':
                            raise RuntimeError('noncomplete cell: no retry, redraw or missing-result scoring')
        verify(m)
        readout(args, m)
    finally:
        signal.alarm(0)
        if pool is not None:
            pool.close()
        live.close()
        decider.close()


def readout(args, m):
    cells = []
    sources = [(p, args.output / p.stem, sha(args.output / 'registration.json'))
        for p in sorted(args.output.glob('root-*.json'))]
    for retained in m.get('retained_cells', {}).values():
        path = Path(retained['path'])
        if sha(path) != retained['sha256']:
            raise RuntimeError('retained complete cell changed')
        sources.append((path, Path(retained['step_dir']), retained['registration_sha256']))
    for path, step_dir, registration_sha in sources:
        cell = json.loads(path.read_text())
        root = next((r for r in m['roots'] if r['ordinal'] == cell['ordinal']), None)
        if (root is None or cell['decision_id'] != root['decision_id']
                or cell['planner'] not in PLANNERS or cell['mode'] not in ('raw', 'search')
                or cell['replicate'] not in m['replicates']
                or cell['first_choice'] != root['first_choices'][cell['planner']]
                or cell['registration_sha256'] != registration_sha):
            raise RuntimeError('continuation roster/provenance drift')
        decision_steps = {name: (step_dir / name, digest) for name, digest in cell['step_hashes'].items()}
        for name, path in cell.get('retained_prefix_steps', {}).items():
            if name in decision_steps or path not in m['retained_input_hashes']:
                raise RuntimeError('resume prefix duplication/unregistered evidence')
            decision_steps[name] = (Path(path), m['retained_input_hashes'][path])
        for name, (step_path, digest) in decision_steps.items():
            if sha(step_path) != digest:
                raise RuntimeError('continuation decision evidence drift')
            row = json.loads(step_path.read_text())
            subject = cell['subject']
            if row['boundary'] == 0:
                if row['actions'][subject] != cell['first_action']:
                    raise RuntimeError('fixed first action drift')
            elif subject in row['actions']:
                own = row['evidence'][subject]
                expected_selector = cell['planner'] if cell['mode'] == 'search' else 'raw_masked_argmax'
                if own['selector'] != expected_selector:
                    raise RuntimeError('follow-through silently changed selector')
            for player in row['actions']:
                if player != subject and row['evidence'][player]['selector'] != 'champion_full_masked_policy_sample':
                    raise RuntimeError('shared explicit opponent drift')
        cells.append(cell)
        if cell['status'] == 'COMPLETE' and len(decision_steps) != cell['boundaries']:
            raise RuntimeError('complete game lacks every played boundary')
    expected = m['search_game_denominator'] + m['raw_control_game_denominator']
    complete = len(cells) == expected and all(c['status'] == 'COMPLETE' for c in cells)
    lookup = {(c['ordinal'], c['replicate'], c['planner'], c['mode']): c for c in cells}
    if len(lookup) != len(cells):
        raise RuntimeError('duplicate continuation cell')
    contrasts = []
    if complete:
        for root in m['roots']:
            for rep in m['replicates']:
                r, i = [lookup[(root['ordinal'], rep, p, 'search')]['signed_outcome'] for p in PLANNERS[::-1]]
                rr, ir = [lookup[(root['ordinal'], rep, p, 'raw')]['signed_outcome'] for p in PLANNERS[::-1]]
                contrasts.append(dict(ordinal=root['ordinal'], replicate=rep,
                    reference_minus_incumbent_search=(r-i)/2,
                    reference_minus_incumbent_raw=(rr-ir)/2,
                    reference_followthrough_gain=(r-rr)/2, incumbent_followthrough_gain=(i-ir)/2))
    report = dict(status='COMPLETE_SELECTED_POSITION_DIAGNOSTIC' if complete else 'INCOMPLETE_NO_STRENGTH_CLAIM',
        registered_games=expected, complete_games=sum(c['status']=='COMPLETE' for c in cells),
        refused_games=sum(c['status']=='REFUSED' for c in cells), capped_games=sum(c['status']=='CAPPED' for c in cells),
        unattempted_games=expected-len(cells), paired_contrasts=contrasts,
        aggregate={key: statistics.mean(c[key] for c in contrasts) for key in (
            'reference_minus_incumbent_search', 'reference_minus_incumbent_raw',
            'reference_followthrough_gain', 'incumbent_followthrough_gain')} if complete else None,
        source_hashes={str(p): sha(p) for p, _, _ in sources},
        preserved_noncomplete_history=m.get('preserved_noncomplete_history', []),
        whole_policy_strength_qualified=False, no_superiority_or_equivalence_claim=True,
        limitations=m['limitations'])
    path = args.output / ('READOUT.json' if complete else f'PARTIAL-READOUT-{len(cells):02d}.json')
    save(path, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'source_hashes'}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('register', 'run', 'readout'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parent-registration', type=Path)
    parser.add_argument('--parent-output', type=Path)
    parser.add_argument('--reuse-complete-from', type=Path)
    parser.add_argument('--resume-refused-from', type=Path)
    parser.add_argument('--raw-only', action='store_true')
    args = parser.parse_args()
    if args.phase == 'register':
        register(args)
    else:
        manifest = json.loads((args.output / 'registration.json').read_text())
        if args.phase == 'run':
            run(args, manifest)
        else:
            readout(args, manifest)


if __name__ == '__main__':
    main()
