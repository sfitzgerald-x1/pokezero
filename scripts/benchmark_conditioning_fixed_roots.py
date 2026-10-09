"""Prepare/run a small source-bound conditioning benchmark, not a strength eval.

Three existing refusal positions and one initial-root control, no redraws or
game resumes. Both modes use the same instrumentation and registered champion;
the repair uses 32 particles, defensive action proposals and ONE weighted
native chance trial per particle/stage. Each trial selects from a fixed 256-original-seed
bank using fully supported PRNG hints and target/proposal correction. A native
observational pilot locates the hypothetical RNG call ordinals. Ten seconds includes
root inference, reconstruction, forward search and world cleanup. Startup,
accepted-prefix replay and root preparation are measured separately. This is
a prepared-root budget, not the original end-to-end decision budget. A single worker is deliberately
NOT clock-equivalent to the registered twenty-worker comparison.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time


SOURCE = Path(__file__).resolve().parents[1]
PIN = '89560aacee1e1634e678b5d01634433f6bbc8574'
REPAIR_MODE = 'early_encore_and_slow_baton_particles32'
SLOW_BATON_IDENTITY = 'seed-566883983-p2-paper_reference'
SLOW_BATON_TERMINAL_SHA = 'a7722ed2e333ff9bb73b02e07e24a4d73244b49cdded902c6e7484b24595f6b2'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def verify(bindings):
    for name, digest in bindings.items():
        path = Path(name)
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
            'benchmark input/source drift: ' + name)


def fixed_cases(manifest):
    deferred = manifest['execution_schedule']['deferred']
    require(set(deferred) == {'seed-1167174895-p2-paper_reference',
        'seed-3495319813-p1-paper_reference', 'seed-474486314-p1-paper_reference'},
        'exact three existing refusals required; no outcome-selected replacement')
    cases = []
    for identity, row in sorted(deferred.items()):
        require(row['accepted_boundaries'] == len(row['accepted_prefix_steps'])
            == len(row['accepted_prefix_hashes']), 'fixed refused prefix count drift')
        names = [f'boundary-{i:03d}.json.gz' for i in range(row['accepted_boundaries'])]
        require(set(names) == set(row['accepted_prefix_steps']) == set(row['accepted_prefix_hashes']),
            'fixed refused prefix gap/extra')
        parts = identity.split('-')
        cases.append(dict(name=identity, seed=int(parts[1]), player=parts[2],
            prefix=[dict(path=row['accepted_prefix_steps'][n], sha256=row['accepted_prefix_hashes'][n])
                for n in names], original_result=row['result_path'],
            original_result_sha256=row['result_sha256'], signed_outcome=None))
    control = cases[1]
    require(control['name'] == 'seed-3495319813-p1-paper_reference', 'fixed control identity drift')
    return [*cases, dict(control, name='initial-control-seed-3495319813-p1', prefix=[])]


def slow_baton_case(root, terminal):
    """Append the existing fourth refusal, never a replacement position."""
    require(terminal['status'] == 'REFUSED' and terminal['signed_outcome'] is None,
        'slow-Baton refusal must remain unscored')
    names = [f'boundary-{i:03d}.json.gz' for i in range(89)]
    require(set(terminal['step_hashes']) == set(names), 'slow-Baton retained prefix gap/extra')
    return dict(name=SLOW_BATON_IDENTITY, seed=566883983, player='p2',
        prefix=[dict(path=str(root/SLOW_BATON_IDENTITY/n), sha256=terminal['step_hashes'][n])
            for n in names], original_result=str(root/(SLOW_BATON_IDENTITY+'.json')),
        original_result_sha256=SLOW_BATON_TERMINAL_SHA, signed_outcome=None)


def prepare(registration, *, include_slow_baton=False):
    require(type(include_slow_baton) is bool, 'explicit fixed fourth-root option required')
    registration = registration.resolve()
    manifest = json.loads(registration.read_text())
    require(manifest['source_commit'] == PIN and manifest['registered_games'] == 256
        and len(manifest['seeds']) == 64, 'original study source/full roster drift')
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip() == PIN,
        'isolated benchmark must remain based on pinned science source')
    cases = fixed_cases(manifest)
    if include_slow_baton:
        terminal_path = registration.parent/(SLOW_BATON_IDENTITY+'.json')
        require(sha(terminal_path) == SLOW_BATON_TERMINAL_SHA, 'fixed slow-Baton refusal hash drift')
        cases.insert(3, slow_baton_case(registration.parent, json.loads(terminal_path.read_text())))
    paths = {registration, Path(__file__), Path(manifest['checkpoint']),
             SOURCE/'scripts/verify_combined_conditioning_benchmark.py',
             SOURCE/'tests/test_benchmark_conditioning_fixed_roots.py',
             SOURCE/'tests/test_conditioning_batch_transport.py',
             SOURCE/'tests/test_baton_pass_public_timing.py',
             SOURCE/'scripts/wider_search_comparison.py'}
    for case in cases:
        paths.add(Path(case['original_result']))
        for row in case['prefix']:
            require(sha(row['path']) == row['sha256'], 'accepted historical prefix hash drift')
            paths.add(Path(row['path']))
        original = json.loads(Path(case['original_result']).read_text())
        require(sha(case['original_result']) == case['original_result_sha256']
            and original['status'] == 'REFUSED' and original['signed_outcome'] is None,
            'original refusal must remain unscored and preserved')
    # Bind execution-relevant source files, including new untracked modules.
    paths.update((SOURCE/'src').rglob('*.py'))
    paths.update((SOURCE/'scripts').glob('*.mjs'))
    paths.update((SOURCE/'tests').glob('test_paper_reference*.py'))
    paths.update((Path(manifest['showdown_root'])/'dist').rglob('*.js'))
    paths.update((Path(manifest['showdown_root'])/'dist').rglob('*.json'))
    bindings = {str(path): sha(path) for path in sorted(paths)}
    require(bindings[manifest['checkpoint']] == manifest['checkpoint_sha256'], 'champion hash drift')
    verify(bindings)
    return dict(schema='pokezero.fixed-conditioning-benchmark.v1', status='PREPARED_NOT_RUN',
        source_root=str(SOURCE), base_source_commit=PIN, source_committed=False,
        original_registration=str(registration), original_registration_sha256=sha(registration),
        checkpoint=manifest['checkpoint'], checkpoint_sha256=manifest['checkpoint_sha256'],
        showdown_root=manifest['showdown_root'], set_source_hash=manifest['set_source_hash'],
        alpha=manifest['reference']['alpha'], beta=manifest['reference']['beta'],
        nominal_seconds=10.0, replicates=3, workers=1, max_transitions=256,
        budget_scope='prepared root: inference, reconstruction, search, cleanup; preparation excluded',
        max_known_set_draws=128, conditioning_batch_size=8,
        cases=cases, modes=['baseline_instrumented', REPAIR_MODE],
        include_slow_baton=include_slow_baton,
        slow_baton_public_timing_repair_applies_to_both_modes=True,
        pending_prior_kernel='original full-support joint rejection; particle options deliberately not propagated',
        history_particles=32, guide_history_actions=True, guided_fraction=.9,
        full_original_action_support_fraction=.1, action_importance_correction=True,
        history_chance_pool=1, batch_history_chance=False, guide_history_chance=True, history_seed_bank=256,
        membership_first=True, membership_first_scope='same species-conditioned joint kernel; seed coupling reordered',
        public_anchor_constraints=True,
        early_encore_potential=True,
        early_indicator_law='guarded own native Encore remaining implied by full observed history; apply before resampling, no native forcing',
        early_constraints='necessary public gender/level and PP-guarded own initial Encore ending',
        direct_initial_encore='uniform independent original timer kernel conditional on necessary observed own ending; constant factor cancels',
        encore_roll_guidance='native context-bound original 3..6 roll; full-support corrected heuristic proposal',
        auxiliary_pilots_per_stage_max=2,
        critical_alignment='at most one additional ORIGINAL-seed native pilot; no extrapolated damage or likelihood contribution',
        empty_guidance='explicitly elide uniform empirical bank and draw one original seed; no policy fallback',
        chance_likelihood_weight='selected original-bank/proposal weight times exact observation indicator; no native retry',
        finite_particle_posterior_approximation=True,
        empirical_belief_reused_within_decision=True,
        input_hashes=bindings, future_actions_must_match_history=False,
        exact_current_root_checks_preserved=True, historical_policy_likelihood_preserved=True,
        historical_chance_likelihood_preserved=True, source_adopted=False,
        strength_inference=False, original_refusals_remain_unscored=True,
        limitation='Small one-worker implementation benchmark; not a twenty-worker strength result.')


def trial_seed(case, replicate):
    return int.from_bytes(hashlib.sha256(
        f'conditioning-fixed-roots-v1:{case}:{replicate}'.encode()).digest()[:8], 'big')


def measured_trial(row):
    """Use disjoint top-level phases, not inclusive nested sampler timers."""
    evidence = row['worker_evidence']
    timings = evidence['search_phase_timing']
    require(len(timings) == 1 and timings[0]['schema'] == 'pokezero.reference-search-phases.v1',
        'one source-bound search measurement required per fixed trial')
    timing = timings[0]
    seconds = {}
    for name in ('root_inference', 'world_reconstruction_inclusive',
                 'forward_search_inclusive', 'world_cleanup'):
        value = timing['phase_timing'].get(name, {}).get('seconds', 0.0)
        require(type(value) in (float, int) and math.isfinite(value) and value >= 0,
            'invalid measured search phase: ' + name)
        seconds[name] = value
    wall = row['wall_search_seconds']
    require(type(wall) in (int, float) and math.isfinite(wall) and wall > 0
        and sum(seconds.values()) <= wall + 1e-6, 'search phase/wall accounting drift')
    counters = {name: timing[name] for name in (
        'completed_trajectories', 'forward_transitions', 'attempted_world_draws')}
    require(all(type(value) is int and value >= 0 for value in counters.values()),
        'invalid measured work counts')
    draws = evidence['draws']
    require(counters['attempted_world_draws'] == len(draws), 'outer world-call count drift')
    if 'completed_trajectories' in row:
        require(row['completed_trajectories'] == counters['completed_trajectories']
            and row['forward_transitions'] == counters['forward_transitions']
            and row['world_draws'] == counters['attempted_world_draws'], 'result/timing work drift')
    statuses = {status: sum(draw['status'] == status for draw in draws)
        for status in ('ROOT_VALIDATED', 'DEADLINE_CANCELLED', 'REFUSED', 'STARTED')}
    require(sum(statuses.values()) == len(draws), 'unknown measured world status')
    ordinary = dict(proposals_started=0, materialized_anchors=0, proposals_rejected=0,
                    accepted_worlds=0, history_steps_started=0, history_steps_completed=0)
    ordinary_receipts = staged_unavailable = membership_unclassified = 0
    membership = dict(complete_team_proposals=0, membership_matches=0, membership_rejections=0)
    for draw in draws:
        metrics = draw.get('conditioning_metrics', {})
        if metrics.get('counter_scope') == 'ordinary_joint_rejection':
            ordinary_receipts += 1
            for key in ordinary:
                value = metrics[key]
                require(type(value) is int and value >= 0, 'invalid ordinary conditioning count')
                ordinary[key] += value
        elif metrics.get('ordinary_counters_available') is False:
            staged_unavailable += 1
        party = draw.get('necessary_party_conditioning', {})
        if party.get('schema') == 'pokezero.history-necessary-party.v1':
            for key in membership:
                value = party[key]
                require(type(value) is int and value >= 0, 'invalid necessary-party count')
                membership[key] += value
            # A complete draw can be censored before its membership check.
            unclassified = (party['complete_team_proposals'] - party['membership_matches']
                            - party['membership_rejections'])
            require(unclassified >= 0, 'necessary-party accounting drift')
            membership_unclassified += unclassified
    particle_receipts = []
    for draw in draws:
        population = draw.get('substitute_particle_conditioning') or draw.get(
            'sampling_diagnostic', {}).get('bootstrap_population')
        if population is not None:
            require(population.get('schema') == 'pokezero.bootstrap-history-conditioning.v1',
                'unknown particle conditioning receipt')
            particle_receipts.append(population)
    return dict(case=row['case'], mode=row['mode'], replicate=row['replicate'], seed=row['seed'],
        particle_conditioning=particle_receipts,
        ordinary_joint_counts_available=bool(ordinary_receipts),
        status=row['status'], actual_child_exit_code=row['actual_child_exit_code'],
        actor_root_key=row['actor_root_key'], phase_seconds=seconds,
        measured_search_wall_seconds=wall,
        reconstruction_fraction_of_search_wall=seconds['world_reconstruction_inclusive']/wall,
        forward_search_fraction_of_search_wall=seconds['forward_search_inclusive']/wall,
        uninstrumented_search_wall_seconds=max(0.0, wall-sum(seconds.values())),
        outer_world_statuses=statuses, ordinary_joint_counts=ordinary,
        ordinary_joint_receipts=ordinary_receipts,
        observed_ordinary_world_yield_per_started_proposal=(
            ordinary['accepted_worlds']/ordinary['proposals_started']
            if ordinary_receipts and ordinary['proposals_started'] else None),
        staged_receipts_without_ordinary_counts=staged_unavailable,
        necessary_party_counts=membership,
        necessary_party_draws_censored_before_membership_check=membership_unclassified,
        neural_forwards=evidence['neural_forwards'], **counters)


def comparison_readout(plan, rows):
    expected = {(case['name'], mode, replicate) for case in plan['cases']
        for mode in plan['modes'] for replicate in range(plan['replicates'])}
    actual = [(row['case'], row['mode'], row['replicate']) for row in rows]
    require(len(actual) == len(set(actual)) and set(actual) == expected,
        'fixed benchmark requires every declared trial, including refusals and zero-work trials')
    measured = [measured_trial(row) for row in rows]
    by_key = {(r['case'], r['mode'], r['replicate']): r for r in measured}
    contrasts, groups = [], []
    for case in plan['cases']:
        for replicate in range(plan['replicates']):
            baseline = by_key[case['name'], 'baseline_instrumented', replicate]
            repair = by_key[case['name'], REPAIR_MODE, replicate]
            require(baseline['actor_root_key'] == repair['actor_root_key']
                and baseline['seed'] == repair['seed'] == trial_seed(case['name'], replicate),
                'paired fixed trial root/seed drift')
            contrasts.append(dict(case=case['name'], replicate=replicate,
                baseline_status=baseline['status'], repair_status=repair['status'],
                completed_trajectories_repair_minus_baseline=(
                    repair['completed_trajectories']-baseline['completed_trajectories']),
                reconstruction_seconds_repair_minus_baseline=(
                    repair['phase_seconds']['world_reconstruction_inclusive']
                    -baseline['phase_seconds']['world_reconstruction_inclusive']),
                forward_search_seconds_repair_minus_baseline=(
                    repair['phase_seconds']['forward_search_inclusive']
                    -baseline['phase_seconds']['forward_search_inclusive'])))
        for mode in plan['modes']:
            subset = [by_key[case['name'], mode, replicate] for replicate in range(plan['replicates'])]
            groups.append(dict(case=case['name'], mode=mode, trials=len(subset),
                successful_trials=sum(r['actual_child_exit_code'] == 0 for r in subset),
                median_completed_trajectories=statistics.median(r['completed_trajectories'] for r in subset),
                median_reconstruction_seconds=statistics.median(
                    r['phase_seconds']['world_reconstruction_inclusive'] for r in subset),
                median_forward_search_seconds=statistics.median(
                    r['phase_seconds']['forward_search_inclusive'] for r in subset)))
    return dict(measured_trials=measured, paired_contrasts=contrasts, fixed_case_groups=groups,
        inference='descriptive fixed-position implementation benchmark, not strength or adoption',
        limitations=[
            'Refusals and zero-work trials remain in every fixed-case denominator.',
            'Only disjoint search-level timers form the wall-time breakdown; nested inclusive times are not added.',
            'Ordinary joint-rejection counts are unavailable on specialized staged paths, not zero.',
            'Ordinary world yield uses started proposals, including deadline-censored work; it is not an uncensored posterior acceptance probability.',
            'Shared initial seeds do not imply identical RNG consumption after repair.',
            'One worker and three replicates per mode do not establish twenty-worker performance or statistical strength.',
            'The repair uses a finite bootstrap approximation and reuses an empirical belief within a decision; it is not exact-posterior or paper-fidelity evidence.'
        ])


def isolated_trial(value, *, command=None, timeout_seconds=120):
    """Always reap the explicitly owned child's session, including Node children."""
    process = subprocess.Popen(command or [sys.executable, str(Path(__file__)), '--child'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True)
    stdout = stderr = ''
    timeout = None
    try:
        stdout, stderr = process.communicate(json.dumps(value), timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        timeout = exc
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate(timeout=2)
        # communicate() can finish even if an orphan no longer owns a pipe.
        remaining = subprocess.check_output(['/bin/ps', '-axo', 'pgid='], text=True)
        if str(process.pid) in {line.strip() for line in remaining.splitlines()}:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        live = []
        for _ in range(20):
            table = subprocess.check_output(['/bin/ps', '-axo', 'pgid=,stat='], text=True)
            live = [line for line in table.splitlines() if len(line.split()) >= 2
                and line.split()[0] == str(process.pid) and not line.split()[1].startswith('Z')]
            if not live:
                break
            time.sleep(.025)
    if live:
        return dict(case=value['case']['name'], mode=value['mode'], replicate=value['replicate'],
            status='BENCHMARK_ERROR', cleanup_error='owned trial group still has live descendants',
            actual_child_exit_code=process.returncode, signed_outcome=None)
    if timeout is not None:
        return dict(case=value['case']['name'], mode=value['mode'], replicate=value['replicate'],
            status='BENCHMARK_ERROR', error=f'owned trial exceeded {timeout_seconds} seconds; no retry',
            actual_child_exit_code=process.returncode, stdout_tail=stdout[-2048:],
            stderr_tail=stderr[-2048:], signed_outcome=None, owned_group_no_live_processes=True)
    try:
        row = json.loads(stdout)
    except json.JSONDecodeError:
        row = dict(case=value['case']['name'], mode=value['mode'], replicate=value['replicate'],
            status='BENCHMARK_ERROR', error='trial did not produce complete JSON evidence',
            stdout_tail=stdout[-2048:], signed_outcome=None)
    row['actual_child_exit_code'] = process.returncode
    row['stderr_tail'] = stderr[-2048:]
    row['owned_group_no_live_processes'] = True
    return row


def child(value):
    plan, case, mode, replicate = (value[k] for k in ('plan', 'case', 'mode', 'replicate'))
    require(plan['source_root'] == str(SOURCE) and mode in plan['modes']
        and case in plan['cases'] and type(replicate) is int and 0 <= replicate < 3,
        'benchmark fixed scope drift')
    verify(plan['input_hashes'])
    # The shared venv has an editable installation in another checkout.
    sys.path.insert(0, str(SOURCE/'src'))
    from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory, PublicRootRequest
    from pokezero.mcts_eval.paper_reference_substitute import SubstituteHistoryTracker
    from pokezero.mcts_eval.paper_reference_pending import (
        PendingPolicyTransition, FaintReplacementTransition, requires_faint_encore_replay,
        requires_baton_interruption_replay)
    from pokezero.mcts_eval.paper_reference import ReferenceConfig, ReferenceRefusal, TrajectorySearch
    def check_imports():
        for name, module in tuple(sys.modules.items()):
            if name == 'pokezero' or name.startswith('pokezero.'):
                origin = Path(module.__file__).resolve()
                require(origin.is_relative_to(SOURCE/'src')
                    and plan['input_hashes'].get(str(origin)) == sha(origin),
                    'foreign or unbound scientific import: ' + name)
    check_imports()
    runtime = None
    record = dict(case=case['name'], mode=mode, replicate=replicate,
        seed=trial_seed(case['name'], replicate), signed_outcome=None, strength_inference=False)
    error = None
    began = time.perf_counter()
    try:
        runtime = ShowdownWorkerFactory(plan['checkpoint'], plan['checkpoint_sha256'],
            plan['showdown_root'], plan['set_source_hash'], allow_earlier_compatible_template=True,
            max_known_set_draws=128, staged_substitute_conditioning=True, conditioning_batch_size=8,
            history_particles=(32 if mode == REPAIR_MODE else 0),
            guide_history_actions=(mode == REPAIR_MODE),
            history_chance_pool=1,
            batch_history_chance=False,
            guide_history_chance=(mode == REPAIR_MODE),
            membership_first=(mode == REPAIR_MODE),
            public_anchor_constraints=(mode == REPAIR_MODE),
            early_encore_potential=(mode == REPAIR_MODE),
            collect_phase_timing=True)(0)
        record['startup_seconds'] = time.perf_counter() - began
        replay_began = time.perf_counter()
        runtime.env.reset(seed=case['seed'])
        tracker = SubstituteHistoryTracker()
        previous_public_transition = None
        def pending_for(public):
            substitute = tracker.certificate(public)
            if substitute is not None:
                return substitute
            pending = previous_public_transition if (
                public.deferred_opponent_action_player is not None
                or requires_baton_interruption_replay(public)) else None
            if requires_faint_encore_replay(public):
                require(previous_public_transition is not None, 'forced Encore replacement lacks public prior root')
                previous = previous_public_transition
                pending = FaintReplacementTransition(previous.before_state,
                    previous.before_observation, previous.own_action, previous.set_source_hash,
                    previous.prior_transition)
            return pending
        for boundary, item in enumerate(case['prefix']):
            require(sha(item['path']) == item['sha256'], 'historical accepted prefix drift')
            row = json.loads(gzip.decompress(Path(item['path']).read_bytes()))
            require(row['boundary'] == boundary and set(row['actions']) == set(runtime.env.requested_players()),
                'original prefix request/action drift')
            if case['player'] in row['actions']:
                state = runtime.env.public_materialization_state(case['player'])
                request = PublicRootRequest.capture(state, runtime.env.observe(case['player']),
                    pending_transition=pending_for(state))
                tracker.accept_own(request, row['actions'][case['player']])
                if not state.self_request.get('forceSwitch') and state.deferred_opponent_action_player is None:
                    previous_public_transition = PendingPolicyTransition.capture(request, row['actions'][case['player']])
            else:
                tracker.accept_opponent_only()
            runtime.env.reseed_simulator_rng(row['chance_seed'])
            runtime.env.step(row['actions'])
            require(runtime.env.terminal() is None, 'original accepted prefix unexpectedly terminal')
        state = runtime.env.public_materialization_state(case['player'])
        request = PublicRootRequest.capture(state, runtime.env.observe(case['player']),
            pending_transition=pending_for(state))
        record['accepted_prefix_replay_seconds'] = time.perf_counter() - replay_began
        prepare_began = time.perf_counter()
        prepared = runtime.prepare(request)
        record['root_preparation_seconds'] = time.perf_counter() - prepare_began
        record['actor_root_key'] = prepared.root.key.hex()
        sample_world = prepared.sample_world
        if case['name'] == SLOW_BATON_IDENTITY:
            def sample_world(rng):
                world = prepared.sample_world(rng)
                if 'slow_baton_native_first_world' not in record:
                    try:
                        require(runtime.env._search_snapshot_permitted,
                            'queue qualification may inspect only the owned hypothetical world')
                        battle = runtime.env.snapshot().bridge_snapshot['battle']
                        choices = [row['choice'] for row in battle['queue']]
                        require(battle['midTurn'] and 'residual' in choices and 'move' not in choices,
                            'slow-Baton world lost residual queue or invented another opponent move')
                        record['slow_baton_native_first_world'] = dict(
                            mid_turn=battle['midTurn'], queue_choices=choices,
                            live_hidden_state_used=False, native_outcome_modified=False)
                    except BaseException:
                        world.close()
                        raise
                return world
        search = TrajectorySearch(ReferenceConfig(plan['alpha'], plan['beta'], max_transitions=256))
        search.reset_battle('fixed-conditioning:' + case['name'])
        began_search = time.perf_counter()
        deadline = began_search + plan['nominal_seconds']
        prepared.set_sampling_deadline(deadline)
        try:
            result = search.search_batch(prepared.root, battle_id='fixed-conditioning:' + case['name'],
                evaluate_root=prepared.evaluate_root, sample_world=sample_world,
                seed=record['seed'], trajectories=100000, deadline_at=deadline)
            record.update(status='SEARCH_WORK_MEASURED' if result.trajectories else 'ZERO_COMPLETED_TRAJECTORIES',
                completed_trajectories=result.trajectories, world_draws=result.world_draws,
                forward_transitions=result.transitions, decision_elapsed_seconds=result.elapsed_seconds)
        except ReferenceRefusal as refused:
            error = refused
            record.update(status='REFUSED', error=type(refused).__name__ + ': ' + str(refused))
        finally:
            record['worker_evidence'] = prepared.evidence()
            record['wall_search_seconds'] = time.perf_counter() - began_search
            require(not prepared.sample_world.active, 'benchmark retained an owned world')
    except BaseException as exc:
        error = exc
        record.update(status='BENCHMARK_ERROR', error=type(exc).__name__ + ': ' + str(exc))
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except BaseException as exc:
                error = exc
                record['cleanup_error'] = type(exc).__name__ + ': ' + str(exc)
        try:
            verify(plan['input_hashes'])
            check_imports()
        except BaseException as exc:
            error = exc
            record['provenance_error'] = str(exc)
        print(json.dumps(record, allow_nan=False), flush=True)
    return 1 if error is not None or record.get('completed_trajectories', 0) == 0 else 0


def run_plan(path, output):
    plan = json.loads(path.read_text())
    require(not output.exists(), 'create-only benchmark output required')
    commands = subprocess.check_output(['/bin/ps', '-axo', 'command='], text=True)
    competing = ('execute_unstarted_collection_amendment.py run',
                 'stream_unstarted_collection_amendment.py --producer-pid')
    require(not any(marker in line for line in commands.splitlines() for marker in competing),
        'timing benchmark requires both pinned producer and replay observer to finish or an authorized stop')
    verify(plan['input_hashes'])
    output.mkdir(exist_ok=False)
    save(output/'registration.json', dict(plan, benchmark_plan_sha256=sha(path)))
    rows = []
    root_keys = {}
    for case in plan['cases']:
        for replicate in range(3):
            order = plan['modes'] if replicate % 2 == 0 else list(reversed(plan['modes']))
            for mode in order:
                value = dict(plan=plan, case=case, mode=mode, replicate=replicate)
                row = isolated_trial(value)
                save(output/f"{case['name']}-{mode}-r{replicate}.json", row)
                rows.append(row)
                print(json.dumps({k: row.get(k) for k in ('case','mode','replicate','status',
                    'completed_trajectories','actual_child_exit_code')}), flush=True)
                require(row['status'] != 'BENCHMARK_ERROR' and 'cleanup_error' not in row
                    and 'provenance_error' not in row, 'benchmark execution/provenance/cleanup failure; no retry')
                expected = root_keys.setdefault(case['name'], row['actor_root_key'])
                require(row['actor_root_key'] == expected, 'mode/replicate changed the fixed actor root')
    verify(plan['input_hashes'])
    save(output/'READOUT.json', dict(status='FIXED_ROOT_MEASUREMENTS_COMPLETE_NOT_STRENGTH',
        rows=rows, source_adopted=False, strength_inference=False,
        finite_particle_posterior_approximation=True, empirical_belief_reused_within_decision=True,
        successful_trials=sum(row['actual_child_exit_code'] == 0 for row in rows),
        failed_trials=sum(row['actual_child_exit_code'] != 0 for row in rows),
        **comparison_readout(plan, rows)))


def main():
    if sys.argv[1:] == ['--child']:
        raise SystemExit(child(json.load(sys.stdin)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--registration', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--include-slow-baton', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        require(args.registration is not None, 'pinned original registration required')
        plan = prepare(args.registration, include_slow_baton=args.include_slow_baton)
        save(args.output, plan)
        print(json.dumps(dict(status=plan['status'], fixed_cases=len(plan['cases']),
            trials=len(plan['cases'])*len(plan['modes'])*plan['replicates'],
            bindings=len(plan['input_hashes']), scientific_calls=0)))
    else:
        require(args.plan is not None, 'prepared fixed benchmark plan required')
        run_plan(args.plan, args.output)


if __name__ == '__main__':
    main()
