"""Create-only fixed-roster/source/seed-proposal evidence verification."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys


def require(ok, message):
    if not ok: raise RuntimeError(message)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_early_indicator(p, expected_hash, subject):
    require(p['schema']=='pokezero.necessary-own-encore-potential.v1'
        and p['live_hidden_state_used'] is False and p['native_outcome_modified'] is False
        and p['terminal_history_sha256']==expected_hash
        and len(p['public_history_sha256'])==64
        and type(p['accepted']) is bool and p['indicator']==int(p['accepted']),
        'early indicator public provenance drift')
    if not p['applicable']:
        require(p['accepted'] is True and 'hypothetical_native_remaining' not in p,
            'ambiguous constraint became a hard native rejection')
        return
    lock=p['certificate']
    require(lock['side']==subject and lock['actor_known_pp']>lock['maximum_pp_payments']
        and lock['required_remaining']==lock['observed_residuals_before_end']+1
        and type(p['hypothetical_native_remaining']) is int and p['hypothetical_native_remaining']>0
        and p['accepted']==(p['hypothetical_native_remaining']==lock['required_remaining']),
        'early timer rejection lacks the public PP/residual guard')


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();require(not args.output.exists(),'create-only verification required')
    plan=json.loads(args.plan.read_text());root=args.results;source=Path(plan['source_root'])
    spec=importlib.util.spec_from_file_location('fixed_benchmark',source/'scripts/benchmark_conditioning_fixed_roots.py')
    benchmark=importlib.util.module_from_spec(spec);spec.loader.exec_module(benchmark)
    benchmark.verify(plan['input_hashes'])
    require(plan['include_slow_baton'] is True and len(plan['cases']) == 5,
        'all four fixed failures and unchanged opening control required')
    require(sha(Path(plan['original_registration'])) == plan['original_registration_sha256'],
        'original study registration drift')
    require(benchmark.prepare(Path(plan['original_registration']), include_slow_baton=True) == plan,
        'prepared plan does not reproduce from pinned source and original evidence')
    durable_registration=json.loads((root/'registration.json').read_text())
    require(durable_registration == dict(plan, benchmark_plan_sha256=sha(args.plan)),
        'durable benchmark registration drift')
    sys.path.insert(0,str(source/'src'))
    from pokezero.mcts_eval import paper_reference_seed_bank as bank
    require(Path(bank.__file__).resolve()==source/'src/pokezero/mcts_eval/paper_reference_seed_bank.py','wrong seed proposal source')
    result_path=root/'READOUT.json';result=json.loads(result_path.read_text());rows=result['rows']
    expected=benchmark.comparison_readout(plan,rows)
    require(all(result[k]==v for k,v in expected.items()),'recomputed readout differs')
    require(len(rows)==30 and len({(r['case'],r['mode'],r['replicate']) for r in rows})==30,'fixed trial roster drift')
    banks=pilots=0;diversity=[];failures=[];pending_draws=0;slow_successes=0
    early_trial_checks=early_trial_rejects=0
    for row in rows:
        path=root/f"{row['case']}-{row['mode']}-r{row['replicate']}.json"
        require(json.loads(path.read_text())==row,'durable trial/readout drift')
        require(row['owned_group_no_live_processes'],'unreaped child group')
        require(row['signed_outcome'] is None,'diagnostic/refusal scored as game result')
        require(row['seed']==benchmark.trial_seed(row['case'],row['replicate']),'fixed RNG seed drift')
        if row['case'] == benchmark.SLOW_BATON_IDENTITY and row['actual_child_exit_code'] == 0:
            require(row['completed_trajectories'] > 0 and row['forward_transitions'] > 0,
                'slow-Baton world acceptance is not forward-search qualification')
            q=row['slow_baton_native_first_world']
            require(q['mid_turn'] and 'residual' in q['queue_choices'] and 'move' not in q['queue_choices']
                and not q['live_hidden_state_used'] and not q['native_outcome_modified'],
                'slow-Baton native residual queue drift')
            slow_successes+=1
        first_population=True
        for draw in row['worker_evidence']['draws']:
            pending=draw.get('pending_policy_conditioning')
            if pending:
                require(pending['prior_conditioning_kernel']==plan['pending_prior_kernel']
                    and not pending['live_opponent_action_used']
                    and pending['current_actor_root_key']==row['actor_root_key']
                    and len(pending['legal'])==len(pending['priors'])
                    and pending['sampled_action'] in pending['legal']
                    and all(math.isfinite(p) and p>=0 for p in pending['priors'])
                    and sum(pending['priors'])>0
                    and pending['attempts']==len(pending['rejected'])+1,
                    'pending prior support/source/actor provenance drift')
                if row['case']==benchmark.SLOW_BATON_IDENTITY:
                    require(pending['transition_kind']=='slow-baton-pass-native-residual-queue',
                        'wrong fourth-root transition')
                pending_draws+=1
            population=draw.get('substitute_particle_conditioning') or draw.get('sampling_diagnostic',{}).get('bootstrap_population')
            if not population:continue
            if plan['early_encore_potential'] and row['mode']==benchmark.REPAIR_MODE:
                constraint=population['necessary_public_anchor']
                expected_hash=constraint['public_history_sha256']
                subject='p2' if constraint['opponent']=='p1' else 'p1'
                if first_population:
                    for pool in population.get('chance_pools',[]):
                        rejected=0
                        for potential in pool.get('early_own_encore_potentials',[]):
                            verify_early_indicator(potential,expected_hash,subject)
                            early_trial_checks+=1
                            rejected+=int(not potential['accepted'])
                        require(pool['rejection_reasons'].get(
                            'necessary observed later own Encore expiry',0)==rejected,
                            'zero early likelihood missing from rejection accounting')
                        early_trial_rejects+=rejected
                    first_population=False
                for step in population.get('steps',[]):
                    potential=step['early_own_encore_potential']
                    verify_early_indicator(potential,expected_hash,subject)
                    require(potential['accepted'],'zero early likelihood reached accepted population')
            for pilot in population.get('chance_pilot_traces',[]):
                bank.validate_native_trace(pilot['trace'],pilot['pilot_seed']);pilots+=1
            for receipt in population.get('seed_banks',[]):
                if not receipt['bank_complete']:
                    require('selected_seed' not in receipt,'partial bank selected')
                    continue
                seeds=receipt['original_seeds'];compatible=receipt['compatible'];q=receipt['proposal_probabilities']
                length=receipt['bank_size'];mass=sum(compatible);fraction=receipt['guided_fraction'];index=receipt['selected_bank_index']
                require(len(seeds)==len(compatible)==len(q)==length,'seed bank size drift')
                require(compatible==[bank.matches_hints(seed,receipt['hints']) for seed in seeds],'seed guidance classification drift')
                computed=[1/length if not mass else (1-fraction)/length+(fraction/mass if c else 0) for c in compatible]
                require(q==computed and all(math.isfinite(x) and x>0 for x in q)
                        and math.isclose(sum(q),1.,abs_tol=1e-12),'seed proposal lost support or normalization')
                require(seeds[index]==receipt['selected_seed'] and math.isclose(
                    receipt['importance_weight'],(1/length)/q[index],abs_tol=1e-12),'seed target/proposal correction drift')
                banks+=1
            if population.get('complete'):
                weights=population['final_normalized_weights'];ancestry=population['final_ancestry_weights']
                require(all(w>0 and math.isfinite(w) for w in weights) and math.isclose(sum(weights),1.,abs_tol=1e-12),'bad final weights')
                require(len(weights)==population['final_particles'] and len(ancestry)==population['distinct_anchor_ancestors'],'diversity drift')
                require(math.isclose(1/math.fsum(w*w for w in weights),population['final_particle_ess'])
                        and math.isclose(1/math.fsum(w*w for w in ancestry.values()),population['final_ancestry_ess']),'ESS drift')
                if draw.get('ordinal')==0:
                    diversity.append(dict(case=row['case'],replicate=row['replicate'],
                        **{k:population[k] for k in ('build_seconds','final_particles','distinct_anchor_ancestors','final_particle_ess','final_ancestry_ess')}))
            elif row['mode']!='baseline_instrumented':
                failures.append(dict(case=row['case'],replicate=row['replicate'],draw_status=draw['status'],error=draw.get('error'),
                    initialized=population['initialized'],stages=[{k:s.get(k) for k in
                        ('stage','attempted','survivors','stage_complete','pre_resampling_ess')} for s in population['stages']]))
    summary=dict(schema='pokezero.seed-bank-benchmark-verification.v1',status='PROGRESS_NOT_GOAL_COMPLETE',
        source_adopted=False,strength_inference=False,original_roster_unchanged=True,source=str(source),
        bindings_verified=len(plan['input_hashes']),durable_trials_verified=30,readout_recomputed=True,
        complete_seed_banks_recomputed=banks,native_pilot_traces_recomputed=pilots,
        pending_worlds_verified=pending_draws,slow_baton_search_trials=slow_successes,
        unique_early_trial_indicators_verified=early_trial_checks,
        zero_early_trial_indicators_verified=early_trial_rejects,
        fixed_case_groups=result['fixed_case_groups'],first_draw_diversity=diversity,failed_populations=failures,
        successful_trials=result['successful_trials'],failed_trials=result['failed_trials'],
        readout_sha256=sha(result_path),plan_sha256=sha(args.plan),
        limitations=result['limitations']+['Concurrent unrelated laptop workloads were observed; cross-run timing comparisons are confounded.'],
        remaining_requirements=['qualify all original failures and meaningful diversity within the budget',
            'integrate qualified conditioning and slow-Baton repairs in fresh source',
            'complete registered wider sample and preregistered statistical comparison'])
    with args.output.open('x') as stream:json.dump(summary,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=summary['status'],bindings=summary['bindings_verified'],trials=30,
        verified_banks=banks,verified_native_pilots=pilots,successful_trials=result['successful_trials'],
        failed_trials=result['failed_trials'],groups=result['fixed_case_groups']),indent=2))


if __name__=='__main__':main()
