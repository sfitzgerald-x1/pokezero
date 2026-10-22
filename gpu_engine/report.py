"""Read-only historical cost-shape tables. Never measures or qualifies a run."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

BATCHES=(256,1024,2048,8192)
SCHEMAS=dict(games='pokezero.gpu_advancing_cost_shape.v1',refill='pokezero.gpu_refill_cost_shape.v1',
    expansion='pokezero.gpu_joint_expansion.v1',fidelity='pokezero.gpu_fidelity_trace.v1',
    ledger='pokezero.gpu_selected_effect_ledger.v1')


def require(condition,message):
    if not condition:raise ValueError(message)


def positive(value):
    require(type(value) in (int,float) and math.isfinite(value) and value>0,'positive finite measurement required')
    return value


def close(value,expected):
    require(type(value) in (int,float) and math.isfinite(value) and math.isclose(value,expected,rel_tol=1e-8,abs_tol=1e-8),
        'reported accounting/rate does not reconcile')


def assemble(paths):
    """Hash the exact bytes read; keep each measured source and workload apart."""
    require(set(paths)=={'games','refill','expansion','cpu','fidelity','ledger'},'all six existing receipts required')
    docs={};sources={}
    for name,path in paths.items():
        blob=Path(path).read_bytes();doc=json.loads(blob,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
        if name in SCHEMAS:require(doc.get('schema')==SCHEMAS[name],'receipt schema differs: '+name)
        if name in ('games','refill','fidelity'):require(doc.get('protocol_completed') is True,'incomplete receipt: '+name)
        docs[name]=doc;sources[name]=dict(path=str(Path(path).resolve()),sha256=hashlib.sha256(blob).hexdigest(),
            source_commit=doc.get('source_commit'),schema=doc.get('schema'))
    games=docs['games'];pool=games['team_pool_sha256']
    for name in ('refill','expansion','fidelity','ledger'):
        require(docs[name].get('team_pool_sha256',docs[name].get('pool_sha256'))==pool,'pool binding differs: '+name)
    require(games['weights']==docs['refill']['weights'],'champion binding differs')
    for name in ('refill','expansion'):
        require((docs[name]['device'],docs[name]['jax'])==(games['device'],games['jax']),'hardware/runtime differs: '+name)
    require([r['batch'] for r in games['rows']]==list(BATCHES),'four ordered advancing batches required')
    require([r['batch'] for r in docs['refill']['rows']]==list(BATCHES),'four ordered refill batches required')
    rows=[]
    for r in games['rows']:
        b=r['batch'];seconds=positive(r['elapsed_seconds']);requests=positive(r['lockstep_requests'])
        require(r['status'] in ('COMPLETE_COHORT','REQUEST_CAP'),'invalid advancing row')
        require(r['completed_games']+r['censored_games']==b,'fixed cohort count differs')
        require(len(r['completed_mask'])==b and sum(r['completed_mask'])==r['completed_games'],'completion mask differs')
        require(len(r['final_turns'])==b,'turn roster differs')
        close(r['games_per_second'],r['completed_games']/seconds)
        close(r['mean_end_to_end_request_ms'],seconds*1000/requests)
        close(r['live_game_requests_per_second'],r['live_game_requests']/seconds)
        close(r['ordinary_game_turns_per_second'],r['ordinary_game_turns']/seconds)
        turns=[t for t,done in zip(r['final_turns'],r['completed_mask']) if done]
        if turns:close(r['mean_completed_game_turns'],sum(turns)/len(turns))
        row={k:r[k] for k in ('batch','status','completed_games','censored_games','games_per_second',
            'mean_completed_game_turns','mean_end_to_end_request_ms','live_game_requests_per_second','ordinary_game_turns_per_second')}
        row.update(engine_step_ms=positive(r['engine']['mean_ms']),encode_ms=positive(r['encode']['mean_ms']),
            forward_ms=positive(r['forward']['mean_ms']),nominal_slot_requests_per_second=b*requests/seconds,
            average_gpu_utilization_percent=None,per_batch_gpu_memory_bytes=None)
        rows.append(row)
    refill=[]
    for r in docs['refill']['rows']:
        require(r['status']=='WINDOW_COMPLETE','invalid refill row');m=r['measured']
        seconds=positive(m['elapsed_seconds']);close(r['games_per_second'],m['completed_games']/seconds)
        refill.append(dict(batch=r['batch'],completed=m['completed_games'],capped=m['capped_games'],
            seconds=seconds,games_per_second=r['games_per_second']))
    expansion=[]
    require([r['nodes'] for r in docs['expansion']['rows']]==[256,1024,4096],'three expansion sizes required')
    for r in docs['expansion']['rows']:
        ms=positive(r['mean_ms']);require(r['child_slots']==r['nodes']*81,'joint action count differs')
        require(r['valid_child_transitions']+r['invalid_frozen_slots']==r['child_slots'],'child accounting differs')
        close(r['node_expansions_per_second'],r['nodes']*1000/ms)
        close(r['valid_child_transitions_per_second'],r['valid_child_transitions']*1000/ms)
        expansion.append({k:r[k] for k in ('nodes','mean_ms','node_expansions_per_second',
            'valid_child_transitions','invalid_frozen_slots','valid_child_transitions_per_second')})
    cpu=[]
    for name in ('engine','showdown'):
        r=docs['cpu'][name];seconds=positive(r['wall_s'])
        completed=sum(x['completed'] for x in r['outcomes'])
        require(completed==r['games'] and len(r['outcomes'])==r['games']+r['incomplete_games'],'CPU roster differs')
        close(r['games_per_second'],completed/seconds)
        cpu.append(dict(backend=name,completed=completed,censored=r['incomplete_games'],seconds=seconds,
            games_per_second=r['games_per_second'],mean_decision_seconds=seconds/positive(r['decisions'])))
    f=docs['fidelity'];counts=Counter(r['status'] for r in f['rows'])
    first=f['roster']['first_game']
    require(f['roster']['games']==200 and [r['game_id'] for r in f['rows']]==list(range(first,first+200)),
        'fixed ordered 200-pair fidelity receipt required')
    require(all(r['team_ids']==[2*r['game_id'],2*r['game_id']+1] for r in f['rows']),'fidelity team IDs differ')
    require(dict(counts)==f['status_counts'] and not counts['HARNESS_ERROR'],'fidelity counts differ')
    matched=counts['MATCHED_THROUGH_TURN20']+counts['MATCHED_TERMINAL']
    require(f['matched']==matched,'matched count differs');close(f['upper_nonmatch_fraction'],(200-matched)/200)
    shares=docs['ledger']['weighted_status_shares'];close(sum(shares.values()),1)
    require(set(shares)=={'SELECTED_EFFECT_IMPLEMENTED','PARTIAL','PLACEHOLDER'}
        and all(type(v) in (int,float) and 0<=v<=1 for v in shares.values()),'coverage fractions differ')
    require(docs['ledger']['game_qualified_move_share'] is None,'this historical report cannot assert qualified coverage')
    return dict(schema='pokezero.gpu_spike_historical_report.v1',status='HISTORICAL_REPORT_NO_NEW_MEASUREMENT',
        recommendation='DO_NOT_ADVANCE_TO_PRODUCTION_NOW',sources=sources,team_pool_sha256=pool,weights=games['weights'],
        device=games['device'],jax=games['jax'],rows=rows,refill_rows=refill,expansion_rows=expansion,cpu_rows=cpu,
        device_counters_after={k:docs[k].get('device_counters_after') for k in ('games','refill','expansion')},
        fidelity=dict(status_counts=dict(counts),matched=matched,fixed_denominator=200,nonmatched_or_unresolved=200-matched,
            upper_nonmatch_fraction=(200-matched)/200,covered_subset_divergence_rate=None),
        coverage=dict(weighted_status_shares=shares,game_qualified_move_share=None,
            qualification='Source-assessed selected effects do not prove >=85% correct effects or qualified games.'),
        limitations=['Historical sources differ; this is not one current/final-source benchmark.',
            'Component times are separate fixed-root samples, not summands of advancing elapsed time.',
            'Nominal batch slots/request time is not live game steps or ordinary turns; actual counters are retained.',
            'Refill is a separate warm-start window, not proven stationarity; batch prefixes are not independent cohorts.',
            'CPU random-legal/no-model baselines are not apples-to-apples champion comparisons.',
            'Device counters are after-only; average utilization and per-batch memory are unavailable.',
            'No new Mac/Metal measurement; the inspected local JAX0.12 runtime exposes CPU only.',
            'Fidelity unknowns remain in the fixed denominator, not an engine-bug rate or a complete-case estimate.',
            'Production encoding, belief, training and full Gen3 fidelity are separate decisions, not spike readout prerequisites.'])


def render_markdown(result):
    lines=['# GPU spike historical measurement tables','',
        f"{result['device']}; one GPU; JAX {result['jax']}; FP32/HIGHEST; unchanged champion both seats/all three heads.",'',
        '**Do not advance to production now.** Mechanics are unqualified and the observation is a non-production placeholder.',
        'This is a read-only report of existing measurements, not a new or unified-current-source run.','',
        '## Finite-cohort table','',f"Measured source: `{result['sources']['games']['source_commit']}`.",'',
        '| Batch | Engine ms | Encode ms | Forward ms | Advancing request ms | Nominal slots/s | Live requests/s | Actual games/s | Mean completed turns | Censored | Avg GPU / batch memory |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in result['rows']:
        lines.append(f"| {r['batch']} | {r['engine_step_ms']:.3f} | {r['encode_ms']:.3f} | {r['forward_ms']:.3f} | {r['mean_end_to_end_request_ms']:.3f} | {r['nominal_slot_requests_per_second']:.1f} | {r['live_game_requests_per_second']:.1f} | {r['games_per_second']:.3f} | {r['mean_completed_game_turns']:.2f} | {r['censored_games']} | unavailable |")
    lines+=['','Components are fixed-root samples, not additive to advancing latency. Actual games/s = completions/wall time, not nominal slots/s divided by the censored completed-game turn mean. Ordinary-turn rates are in JSON.','',
        '## Separate refill windows','',f"Measured source: `{result['sources']['refill']['source_commit']}`.",'',
        '| Batch | Completed | Capped | Window seconds | Games/s |','| ---: | ---: | ---: | ---: | ---: |']
    for r in result['refill_rows']:lines.append(f"| {r['batch']} | {r['completed']} | {r['capped']} | {r['seconds']:.3f} | {r['games_per_second']:.3f} |")
    lines+=['','## All-81 expansion','',f"Measured source: `{result['sources']['expansion']['source_commit']}`.",'',
        '| Parents | Mean ms | Parents/s | Valid children | Invalid frozen | Valid transitions/s |','| ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in result['expansion_rows']:lines.append(f"| {r['nodes']} | {r['mean_ms']:.3f} | {r['node_expansions_per_second']:.1f} | {r['valid_child_transitions']} | {r['invalid_frozen_slots']} | {r['valid_child_transitions_per_second']:.1f} |")
    lines+=['','## CPU environment baselines','',
        'CPU receipt does not embed a source commit; none is inferred from its filename. Input byte digest is retained. Random legal actions, no neural forward.','',
        '| Backend | Completed | Censored | Games/s | Wall seconds/decision |','| --- | ---: | ---: | ---: | ---: |']
    for r in result['cpu_rows']:lines.append(f"| {r['backend']} | {r['completed']} | {r['censored']} | {r['games_per_second']:.3f} | {r['mean_decision_seconds']:.6f} |")
    lines+=['','## Fidelity and coverage','',f"Fidelity source: `{result['sources']['fidelity']['source_commit']}`; fixed denominator 200. Counts: `{json.dumps(result['fidelity']['status_counts'],sort_keys=True)}`.",
        'Unknowns stay unresolved; covered-subset divergence is unavailable, not zero. The <10% gate is not demonstrated.',
        f"Coverage source: `{result['sources']['ledger']['source_commit']}`. Source-assessed shares: `{json.dumps(result['coverage']['weighted_status_shares'],sort_keys=True)}`.",
        result['coverage']['qualification'],'',
        '## Source bindings and limitations','']
    for name,s in result['sources'].items():lines.append(f"- {name}: source `{s['source_commit'] or 'not embedded'}`, input SHA256 `{s['sha256']}`.")
    lines+=['',f"After-only device counters (not averages/per-batch): `{json.dumps(result['device_counters_after'],sort_keys=True)}`.",'']
    lines+=['- '+note for note in result['limitations']]
    return '\n'.join(lines)+'\n'
