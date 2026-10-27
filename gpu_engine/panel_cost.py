"""Explicit saved-root latency accounting; no collection or launch authority.

Numeric percentile bounds, timed-root count, process integrity and exercised
comparison paths are separate facts. A refusal remains a refusal-path timing;
it is never replaced by a more convenient root or called comparison timing.
"""
from dataclasses import dataclass
import hashlib
import json
import math


SCHEMA = 'pokezero.saved-root-panel-cost.v1'
TIMING_SCOPE = 'binding-plan-native-jax-synchronized-result-and-progress-fsync.v1'
BUDGET_SCHEMA = 'pokezero.exact-comparator-operation-budget.v1'


def operation_budget(entry):
    """The identical operation budget must accompany cost and future entries."""
    value=entry.get('operation_budget')
    if (type(value) is not dict or set(value)!={'schema','operation','limit_seconds'}
            or value['schema']!=BUDGET_SCHEMA or value['operation']!=TIMING_SCOPE
            or type(value['limit_seconds']) not in (int,float)
            or not math.isfinite(value['limit_seconds']) or value['limit_seconds']<=0):
        raise ValueError('explicit full comparator operation budget required')
    digest=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        allow_nan=False).encode()).hexdigest()
    return value,digest


@dataclass(frozen=True)
class Contract:
    slots: int
    decision_limit_seconds: float
    timing_scope: str

    @property
    def threshold_seconds(self):
        return self.decision_limit_seconds / 2

    @property
    def rank(self):
        return (95 * self.slots + 99) // 100


def contract(entry):
    """Require a new explicit cost purpose; ordinary panel guards stay intact."""
    cfg=entry.get('cost_admission')
    if (type(cfg) is not dict or set(cfg)!={'schema','decision_limit_seconds','timing_scope','operation_budget_sha256'}
            or cfg['schema']!=SCHEMA or cfg['timing_scope']!=TIMING_SCOPE
            or type(entry.get('slots')) is not int or entry['slots']<20
            or type(cfg['decision_limit_seconds']) not in (int,float)
            or not math.isfinite(cfg['decision_limit_seconds']) or cfg['decision_limit_seconds']<=0):
        raise ValueError('explicit >=20 saved-root cost contract and decision limit required')
    budget,digest=operation_budget(entry)
    if cfg['operation_budget_sha256']!=digest or cfg['decision_limit_seconds']!=budget['limit_seconds']:
        raise ValueError('cost and future comparator must reference the same operation budget')
    return Contract(entry['slots'],float(cfg['decision_limit_seconds']),cfg['timing_scope'])


def summarize(entry, observations):
    """Bound nearest-rank p95 on the whole fixed roster, without dropping slots.

    COMPLETE has an exact full-slot duration. CENSORED has a known lower bound
    and unbounded upper duration. A missing DONE alone proves no positive lower
    bound: use MISSING, including a possible hang publishing the excluded witness.
    MISSING/UNATTEMPTED have [0, infinity]. A slow
    twentieth root does not change rank19/20 into a maximum-latency rule.
    Caller separately verifies attempt/input identity, witnesses and cleanup.
    """
    cfg=contract(entry)
    if type(observations) not in (list,tuple) or len(observations)!=cfg.slots:
        raise ValueError('one cost observation per fixed slot required')
    lower=[];upper=[];timed=0;paths={};states={}
    for slot,obs in enumerate(observations):
        if type(obs) is not dict or obs.get('slot')!=slot or type(obs.get('slot')) is not int:
            raise ValueError('exact ordered slot identity required; no duplicate or replacement')
        status=obs.get('status');states[status]=states.get(status,0)+1
        if status not in {'COMPLETE','CENSORED','MISSING','UNATTEMPTED'}:
            raise ValueError('explicit cost observation status required')
        if status in {'MISSING','UNATTEMPTED'}:
            lo=0.;hi=math.inf
            if set(obs)!={'slot','status'}:
                raise ValueError('untimed slots cannot carry timing or execution claims')
        else:
            allowed={'ordinary','conditional','refused','source_failure','unknown_inflight'}
            if 'ordinary_carry' in entry:allowed.add('ordinary_carry')
            if set(obs)!={'slot','status','seconds','path'} or obs['path'] not in allowed:
                raise ValueError('timed duration and actual exercised path required')
            duration=obs['seconds']
            if type(duration) not in (int,float) or not math.isfinite(duration) or duration<0:
                raise ValueError('finite nonnegative measured duration required')
            lo=float(duration);hi=lo if status=='COMPLETE' else math.inf
            timed+=1;paths[obs['path']]=paths.get(obs['path'],0)+1
        lower.append(lo);upper.append(hi)
    lo=sorted(lower)[cfg.rank-1];hi=sorted(upper)[cfg.rank-1]
    verdict=('MET' if hi<=cfg.threshold_seconds else 'FAILED'
        if lo>cfg.threshold_seconds else 'UNRESOLVED')
    return dict(fixed_slots=cfg.slots,timed_roots=timed,minimum_timed_roots_met=timed>=20,
        percentile_rank=cfg.rank,p95_lower_seconds=lo,
        p95_upper_seconds=hi if math.isfinite(hi) else 'unbounded',
        threshold_seconds=cfg.threshold_seconds,latency_threshold=verdict,
        statuses=states,exercised_paths=paths,
        unexercised_comparison_paths=[p for p in (('ordinary','ordinary_carry','conditional')
            if 'ordinary_carry' in entry else ('ordinary','conditional')) if paths.get(p,0)==0],
        attempt_integrity='NOT_ASSESSED_BY_LATENCY_ACCOUNTING',
        comparison_cost_qualified=False,scientific_fidelity_qualified=False)
