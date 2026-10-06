"""Create-only, preregistered whole-game paper-reference/incumbent comparison.

Qualification uses disjoint seeds and never enters the confirmatory analysis.
The confirmatory roster is fixed at 64 seed clusters (256 games). Run sequentially
on the laptop; never deploy, retrain, merge PRs, redraw or overwrite a result.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import random
import signal
import subprocess
import time

from pokezero.mcts_eval.wider_search import ARMS, SEATS, analyze, game_identity, play_game, study_seed

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def git(*args, root=REPO):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def verify(m):
    if git('rev-parse', 'HEAD') != m['source_commit'] or git('status', '--porcelain'):
        raise RuntimeError('source must remain pinned and clean')
    for path, digest in m['input_hashes'].items():
        if sha(path) != digest:
            raise RuntimeError('source/input binding drift: ' + path)
    if (git('rev-parse', 'HEAD', root=m['showdown_root']) != m['showdown_commit']
            or git('status', '--porcelain', root=m['showdown_root'])):
        raise RuntimeError('pinned simulator drift')
    import pokezero_search
    from pokezero.randbat import load_gen3_randbat_source_cached
    native = Path(pokezero_search.__file__).parent
    if native != Path(m['native_package']):
        raise RuntimeError('wrong native package imported')
    if load_gen3_randbat_source_cached(m['showdown_root']).metadata.source_hash != m['set_source_hash']:
        raise RuntimeError('set-source binding drift')


def register(args):
    if git('status', '--porcelain'):
        raise RuntimeError('commit reviewed source before registration')
    import pokezero_search
    from pokezero.randbat import load_gen3_randbat_source_cached
    # Reuse an immutable compiled binary only when ALL its Rust source hashes
    # match the separately retained build evidence. Python conditioning uses the
    # new homogeneous source, not the old diagnostic Python files.
    receipt = json.loads(args.native_binding.read_text())
    native = Path(pokezero_search.__file__).parent
    hashes = {str(args.checkpoint): sha(args.checkpoint), str(args.native_binding): sha(args.native_binding)}
    for path in sorted(native.glob('*')):
        if path.is_file():
            hashes[str(path)] = sha(path)
    compiled_source_root = Path(next(path for path in receipt['source_hashes']
        if path.endswith('/rust/pokezero-search/Cargo.toml'))).parents[2]
    for relative in git('ls-files', 'src', 'scripts', 'rust/pokezero-search', 'third_party', 'pyproject.toml').splitlines():
        path = REPO / relative
        hashes[str(path)] = sha(path)
        if (relative.startswith('rust/pokezero-search/')
                and (path.suffix == '.rs' or path.name in ('Cargo.toml', 'Cargo.lock'))):
            expected = [value for original, value in receipt['source_hashes'].items()
                if original.endswith('/' + relative)]
            if len(expected) != 1 or expected[0] != sha(path):
                raise RuntimeError('compiled binary source binding differs: ' + relative)
        if relative.startswith('third_party/'):
            original = compiled_source_root / relative
            if not original.is_file() or sha(original) != sha(path):
                raise RuntimeError('compiled engine source/patch binding differs: ' + relative)
            hashes[str(original)] = sha(original)
    binary = list(native.glob('*.so'))
    if len(binary) != 1 or sha(binary[0]) not in receipt['source_hashes'].values():
        raise RuntimeError('native binary absent from retained build evidence')
    source = load_gen3_randbat_source_cached(args.showdown_root)
    seeds = [study_seed(i, qualification=args.qualification) for i in range(2 if args.qualification else 64)]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError('seed collision; do not silently replace seeds')
    m = dict(schema='pokezero.wider-search.v1', created_at=datetime.now(timezone.utc).isoformat(),
        phase='QUALIFICATION_NOT_STRENGTH' if args.qualification else 'FIXED_64_SEED_CONFIRMATION',
        source_commit=git('rev-parse', 'HEAD'), source_root=str(REPO), seeds=seeds, arms=ARMS, seats=SEATS,
        registered_games=4*len(seeds), checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint),
        showdown_root=str(args.showdown_root), showdown_commit=git('rev-parse', 'HEAD', root=args.showdown_root),
        set_source_hash=source.metadata.source_hash, native_package=str(native), input_hashes=hashes,
        nominal_decision_seconds=10., per_decision_safety_seconds=120,
        max_boundaries=200, per_game_wall_seconds=2400.,
        qualification_required_before_confirmation=True,
        opponent='same immutable champion; full masked policy sample at every requested opponent decision',
        initial_state='fresh Gen3 random battle; same generated teams/seed for each arm and candidate seat',
        first_action='selected by respective search, not a historical frozen intervention',
        rng='shared seed/seat/boundary domains across arms; opponent and chance independent of search',
        incumbent=dict(depth=6, sims=4096, batch=16, worlds=4, world_workers=1,
            model_priors=True, use_opponent_priors=False, leaf_eval='model', native_batch_guard_ms=64),
        reference=dict(workers=20, exchange_trajectories=10, alpha=.5, beta=1.,
            allow_earlier_compatible_template=True, max_known_set_draws=128,
            canonical_public_rules=True, retain_Q_N_M_F=True),
        primary_analysis=dict(unit='battle seed; two seats clustered', score='win=1, draw=.5, loss=0',
            contrast='mean reference score minus mean incumbent score within seed', alpha=.05,
            test='exact two-sided seed-cluster sign flip; exchangeability assumption disclosed',
            confidence='distribution-free Hoeffding bounded-mean 95% interval',
            positive_claim='full fixed roster; p<=.05 AND mean confidence lower bound>0',
            fixed_N=True, optional_stopping=False, historical_data_pooled=False,
            missing='no complete-case inference; full-roster worst-case bounds only'),
        limitations=['not equal CPU allocation: reference twenty workers vs incumbent one',
            'nominal 10s matched decision budgets, actual latency and work recorded separately',
            '128-draw compatible-template adaptation, not an exact hidden-state posterior',
            'local public HP surface may differ from online percentage-only observations',
            'same champion opponent; not Foul Play or unrestricted all-opponent superiority',
            '64 independent clusters may be insufficient for a modest advantage; no equivalence inference',
            'qualification outcomes never enter confirmation; refusal halts, no raw fallback or redraw'])
    verify(m)
    args.output.mkdir(exist_ok=False)
    save(args.output/'registration.json', m)


def timeout(*args):
    raise TimeoutError('registered decision safety cap; no raw fallback')


def run(args, m):
    import torch
    from pokezero.collection import env_config_with_policy_spec_masks
    from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv, showdown_choice_for_action
    from pokezero.mcts_eval.head_to_head import public_only_context
    from pokezero.mcts_eval.manifest import SearchConfig
    from pokezero.mcts_eval.paper_reference import ReferenceConfig
    from pokezero.mcts_eval.paper_reference_parallel import ParallelTrajectorySearch
    from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest, ShowdownWorkerFactory
    from pokezero.mcts_eval.paper_reference_showdown import ChampionEvaluator, decision_state
    from pokezero.mcts_eval.policy_opponent_profile import make_profile_decider, validate_selection
    from pokezero.mcts_eval.resolver import resolve_checkpoint_contract
    from pokezero.neural_policy import load_transformer_policy
    from pokezero.policy import PolicyContext
    from pokezero.trajectory import BattleTrajectory, TrajectoryStep
    from search_followthrough_diagnostic import reference_statistics_checkpoint
    verify(m)
    if m['phase'] != 'QUALIFICATION_NOT_STRENGTH':
        if args.qualification_readout is None:
            raise RuntimeError('confirmation requires completed disjoint qualification')
        qualification = json.loads(args.qualification_readout.read_text())
        if (qualification.get('status') != 'QUALIFICATION_COMPLETE_NOT_STRENGTH'
                or qualification.get('source_commit') != m['source_commit']
                or qualification.get('input_hashes') != m['input_hashes']):
            raise RuntimeError('qualification incomplete or binding differs')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    evaluator = ChampionEvaluator(load_transformer_policy(Path(m['checkpoint']), device='cpu',
        deterministic=True, exploration_epsilon=0., sampling_temperature=1., family_gated_selection=False))
    cfg = env_config_with_policy_spec_masks(LocalShowdownConfig(showdown_root=Path(m['showdown_root']),
        set_belief_source=True), [f"neural:{m['checkpoint']}"], context='wider full-game search comparison')
    live = LocalShowdownEnv(cfg)
    contract = resolve_checkpoint_contract(m['checkpoint'], expected_sha256=m['checkpoint_sha256'],
        showdown_root=m['showdown_root'], showdown_source_sha256=m['set_source_hash'])
    decider = make_profile_decider(contract, m['showdown_root'], arm='incumbent_mcts',
        mode='matched_deadline', opponent_seed=0, deadline_ms=10000, native_batch_guard_ms=64)
    search_cfg = SearchConfig(depth=6, sims=4096, batch=16, worlds=4, inference_mode='local')
    native = decider._policy_for(search_cfg)
    pool = None
    signal.signal(signal.SIGALRM, timeout)
    try:
        pool = ParallelTrajectorySearch(ReferenceConfig(.5, 1.), ShowdownWorkerFactory(
            m['checkpoint'], m['checkpoint_sha256'], m['showdown_root'], m['set_source_hash'],
            allow_earlier_compatible_template=True, max_known_set_draws=128))
        print(json.dumps(dict(status='POOL_READY', startup_seconds=pool.startup_seconds)), flush=True)
        for ordinal, seed in enumerate(m['seeds']):
            for subject in SEATS:
                order = ARMS if (ordinal + SEATS.index(subject)) % 2 else tuple(reversed(ARMS))
                for arm in order:
                    identity = game_identity(seed, subject, arm)
                    path = args.output/(identity+'.json')
                    if path.exists():
                        row = json.loads(path.read_text())
                        if row['status'] != 'COMPLETE' or row['registration_sha256'] != sha(args.output/'registration.json'):
                            raise RuntimeError('extant noncomplete game cannot be overwritten or retried')
                        continue
                    steps = args.output/identity
                    steps.mkdir(exist_ok=False)
                    live.reset(seed=seed)
                    battle_id = 'wider-search:' + identity
                    trajectory = BattleTrajectory(battle_id, 'gen3randombattle', seed)
                    native.reset()

                    def select(observation, boundary, rng_seed):
                        signal.alarm(m['per_decision_safety_seconds'])
                        begun = time.perf_counter()
                        try:
                            if arm == 'deep_incumbent':
                                context = public_only_context(PolicyContext(player_id=subject,
                                    decision_round_index=boundary, battle_id=battle_id, format_id='gen3randombattle',
                                    seed=seed, observation=observation, requested_players=tuple(live.requested_players()),
                                    trajectory=trajectory, requested_observations={subject: observation},
                                    requested_legal_action_masks={subject: tuple(observation.legal_action_mask)},
                                    public_materialization_state=live.public_materialization_state(subject)))
                                before = decider._snapshot_stats(native)
                                decision = native.select_action_with_context(context, rng=random.Random(rng_seed))
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
                                if sum(observation.legal_action_mask) > 1:
                                    validate_selection(evidence, arm='incumbent_mcts', mode='matched_deadline',
                                        config=search_cfg, mask=observation.legal_action_mask, opponent_seed=rng_seed,
                                        deadline_ms=10000, native_batch_guard_ms=64)
                                elif evidence['fallbacks'] or evidence['prior_fallbacks']:
                                    raise RuntimeError('forced request fell back')
                            else:
                                request = PublicRootRequest.capture(live.public_materialization_state(subject), observation)
                                remaining = 10 - (time.perf_counter() - begun)
                                if remaining <= 0:
                                    raise TimeoutError('reference deadline expired during public capture')
                                measured = pool.search(request, decision_state(observation, player=subject),
                                    battle_id=battle_id, seed=rng_seed,
                                    deadline_seconds=remaining)
                                index = int(measured.result.action.split(':')[1])
                                draws = [d for receipt in measured.worker_receipts for d in receipt['evidence']['draws']]
                                if (not draws or len(draws) != measured.result.world_draws
                                        or not all(d['status'] == 'ROOT_VALIDATED' and d['released'] for d in draws)
                                        or len(set(measured.worker_pids)) != 20):
                                    raise RuntimeError('reference world/worker/release witness failed')
                                evidence = asdict(measured)
                                evidence['statistics_checkpoint'] = reference_statistics_checkpoint(pool._master.snapshot())
                            trajectory.append(TrajectoryStep(player_id=subject, turn_index=boundary,
                                observation=observation, legal_action_mask=tuple(observation.legal_action_mask), action_index=index))
                            return index, dict(selector=arm, elapsed_seconds=time.perf_counter()-begun,
                                seed=rng_seed, search_evidence=evidence)
                        finally:
                            signal.alarm(0)

                    def emit(row):
                        save(steps/f"boundary-{row['boundary']:03d}.json", row)
                        print(json.dumps(dict(identity=identity, boundary=row['boundary'],
                            actions=row['actions'], decision_seconds=row['evidence'].get(subject, {}).get('elapsed_seconds'))), flush=True)

                    result = dict(identity=identity, seed=seed, subject=subject, arm=arm,
                        registration_sha256=sha(args.output/'registration.json'))
                    try:
                        result.update(play_game(live, subject=subject, decision_id=f'wider-search:{seed}:{subject}',
                            selector=select, opponent=evaluator, emit=emit,
                            max_boundaries=m['max_boundaries'], wall_seconds=m['per_game_wall_seconds']))
                    except Exception as error:
                        result.update(status='REFUSED', signed_outcome=None,
                            error=f'{type(error).__name__}: {error}', failure_evidence=getattr(error, 'evidence', None))
                    result['step_hashes'] = {p.name: sha(p) for p in steps.glob('*.json')}
                    save(path, result)
                    print(json.dumps({key: result.get(key) for key in ('identity', 'status', 'signed_outcome', 'error')}), flush=True)
                    if result['status'] != 'COMPLETE':
                        raise RuntimeError('noncomplete registered game; preserve evidence, no fallback/redraw')
        verify(m)
        readout(args, m)
    finally:
        signal.alarm(0)
        if pool is not None:
            pool.close()
        live.close()
        decider.close()


def readout(args, m):
    verify(m)
    rows = []
    for path in sorted(args.output.glob('seed-*.json')):
        row = json.loads(path.read_text())
        if row['registration_sha256'] != sha(args.output/'registration.json'):
            raise RuntimeError('result registration binding drift')
        actual = {p.name: sha(p) for p in (args.output/path.stem).glob('*.json')}
        if actual != row['step_hashes']:
            raise RuntimeError('durable decision evidence drift')
        rows.append(row)
    result = analyze(m['seeds'], rows)
    result.update(source_commit=m['source_commit'], input_hashes=m['input_hashes'], phase=m['phase'])
    if m['phase'] == 'QUALIFICATION_NOT_STRENGTH':
        result['statistically_supported_advantage'] = False
        result['inferential_test_allowed'] = False
        result['status'] = 'QUALIFICATION_COMPLETE_NOT_STRENGTH' if not result['missing_seed_clusters'] else 'QUALIFICATION_INCOMPLETE'
    save(args.output/'READOUT.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'run', 'readout'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--showdown-root', type=Path)
    parser.add_argument('--native-binding', type=Path)
    parser.add_argument('--qualification', action='store_true')
    parser.add_argument('--qualification-readout', type=Path)
    args = parser.parse_args()
    if args.mode == 'register':
        if not all((args.checkpoint, args.showdown_root, args.native_binding)):
            parser.error('registration requires checkpoint, showdown-root and native-binding')
        register(args)
    else:
        m = json.loads((args.output/'registration.json').read_text())
        (run if args.mode == 'run' else readout)(args, m)


if __name__ == '__main__':
    main()
