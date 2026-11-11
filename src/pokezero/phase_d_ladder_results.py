"""Fixed paired ladder readout with seed-clustered intervals and missing cells.

Pair units are declared seed×seat cells; the two seats share a seed, so the
confidence calculation uses their seed mean. Missing values stay identification
bounds and are never silently replaced by caps/ties or complete-case scores.
"""
from collections import Counter
import math
from statistics import fmean,variance
from . import phase_d_ladder_panel as panel


def paired(declaration,raw,search,*,rung,limit,expected_declaration_sha256):
    expected=panel.cells(declaration,rung,limit=limit);control=panel.cells(declaration,'RAW',limit=limit)
    def index(rows,arm):
        result={}
        for row in rows:
            cell=panel.checked_cell(declaration,row['cell']);key=cell['pair_ordinal']
            if cell['arm']!=arm or key>=limit or key in result:raise ValueError('foreign/duplicate fixed panel result')
            if row.get('declaration_sha256')!=expected_declaration_sha256:raise ValueError('original ladder declaration differs')
            score=row.get('score');known=row.get('status')=='COMPLETE'
            if known and score not in (0.,.5,1.)or not known and score is not None:raise ValueError('literal known terminal score or unknown required')
            result[key]=row
        return result
    left=index(raw,'RAW');right=index(search,rung);lower=[];upper=[];known_deltas={};changed=0
    for cell in expected:
        key=cell['pair_ordinal'];a=left.get(key,{}).get('score');b=right.get(key,{}).get('score')
        lower.append((0. if b is None else b)-(1. if a is None else a))
        upper.append((1. if b is None else b)-(0. if a is None else a))
        if a is not None and b is not None:
            known_deltas[key]=b-a;changed+=a!=b
    n=limit;known=len(known_deltas);cluster=[]
    for start in range(0,n,2):
        if start in known_deltas and start+1 in known_deltas:cluster.append((known_deltas[start]+known_deltas[start+1])/2)
    interval=None
    if len(cluster)>=2 and variance(cluster)>0:
        from scipy.stats import t
        radius=float(t.ppf(.95,len(cluster)-1))*math.sqrt(variance(cluster)/len(cluster))
        interval=[max(-1.,fmean(cluster)-radius),min(1.,fmean(cluster)+radius)]
    point=fmean(lower)if known==n else None
    identified=[fmean(lower),fmean(upper)]
    # A missing-cell screen can advance only when even the worst-case full
    # difference meets the bar. Never select survivors to trigger extension.
    promote=limit==500 and identified[0]>=declaration['screen_threshold_pp']/100
    return dict(declaration_sha256=expected_declaration_sha256,rung=rung,paired_cells=n,known_pairs=known,unknown_pairs=n-known,
        point_difference=point,full_roster_identification_bounds=identified,
        complete_seed_clusters=len(cluster),complete_seed_difference=fmean(cluster)if cluster else None,
        interval90=interval,interval_scope='PAIRED_T_SEED_CLUSTERS_TWO_SEATS_MEAN; missing clusters unavailable, not full-roster confidence',
        interval_status='AVAILABLE'if interval is not None else'UNAVAILABLE_TOO_FEW_OR_ZERO_SAMPLE_VARIANCE',
        discordant_known=changed,discordant_known_rate=None if known==0 else changed/known,
        discordant_full_bounds=[changed/n,(changed+n-known)/n],
        decision='PROMOTE_SAME_DECLARED_PREFIX_TO_2000'if promote else'NO_PROMOTION',
        screen_rule='Full500point>=.03; with missing cells require full-roster worst-case bound>=.03.',
        observed_positive_lower_bound=limit==2000 and interval is not None and interval[0]>0,
        strength_qualified=False,qualification_scope='Reviewed arm choice must retain the original missing-cell caveat; no automatic qualification.',
        no_refill=True)


def telemetry(rows,*,declared_decisions=None):
    counts=Counter();refusals=Counter();turns={};times=[];acting=[]
    seen=set()
    for row in rows:
        key=(row['battle_id'],row['owner'],row['decision_index'],row['prefix_sha256'])
        if key in seen:raise ValueError('duplicate original searched decision')
        seen.add(key);result=row['search'];status=result.get('status')
        if status not in ('COMPLETE','REFUSED'):raise ValueError('literal search decision disposition required')
        counts['decisions']+=1;counts['searched']+=status=='COMPLETE';counts['refused']+=status=='REFUSED'
        fallback=row['fell_back'];counts['raw_fallbacks']+=fallback
        if type(fallback)is not bool or fallback!=(status=='REFUSED'):raise ValueError('every refusal must retain counted raw fallback')
        if status=='REFUSED':refusals[result.get('refusal')or'UNNAMED_REFUSAL']+=1
        elapsed=result.get('client_wall_seconds')
        if elapsed is not None:
            if type(elapsed)not in (int,float)or not math.isfinite(elapsed)or elapsed<0:raise ValueError('finite inclusive client time required')
            times.append(elapsed)
        elapsed=row.get('acting_wall_seconds')
        if elapsed is not None:acting.append(elapsed)
    n=counts['decisions'];unknown=None if declared_decisions is None else declared_decisions-n
    if unknown is not None and unknown<0:raise ValueError('more telemetry rows than actual decisions')
    rank=lambda values:None if not values else sorted(values)[math.ceil(.95*len(values))-1]
    return dict(counts=dict(counts),refusals_by_kind=dict(refusals),refused_share=None if n==0 else counts['refused']/n,
        raw_fallback_share=None if n==0 else counts['raw_fallbacks']/n,client_p95_seconds=rank(times),acting_p95_seconds=rank(acting),
        declared_decisions=declared_decisions,missing_telemetry=unknown,
        scope='Actual client request including queue, constructor, search, transfer; fallback acting clock separately.')
