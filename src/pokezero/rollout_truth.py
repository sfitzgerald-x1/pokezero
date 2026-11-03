"""Fixed-bank value diagnostics. No engine/model work occurs in this module.

All means use the declared64 terminal continuations; missing continuations
produce bounds, never a survivor-renormalized point target. Values use[-1,1],
calibration uses win/draw/loss probability(1,.5,0). Rollout truth is conditional
on the frozen raw policy, fixed physical world and engine—not minimax truth.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import numpy as np

TRUTH = 'pokezero.fixed-bank-rollout-truth.v1'
CHAMPION = '0fd095923b4ac7e05d6e2b3ccab9c1e6869dff4893c2dae456caff10dce690be'


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Recipe:
    seed: int
    rollouts: int = 64
    max_requests: int = 4096
    def __post_init__(self):
        if type(self.seed)is not int or not 0<=self.seed<2**32 or self.rollouts!=64 or type(self.rollouts)is not int:
            raise ValueError('predeclared uint32 seed and exactly64 rollouts required')
        if type(self.max_requests)is not int or self.max_requests<=0:
            raise ValueError('positive fixed continuation cap required')
    def keys(self,selector,action,reply,chance_id):
        # Every declared branch/replicate owns predeclared streams shared by
        # every evaluated head. No key depends on outcomes or refusals.
        if set(selector)!={'stream','frame','owner'} or any(type(x)is not int or x<0 for x in selector.values()) or selector['owner']not in (0,1):
            raise ValueError('literal fixed selector required')
        if type(action)is not int or type(reply)is not int or not 0<=action<9 or not 0<=reply<9 or type(chance_id)is not str or len(chance_id)!=64:
            raise ValueError('same literal branch/chance coordinates required')
        return tuple(tuple(int(digest(dict(seed=self.seed,selector=selector,action=action,reply=reply,chance_id=chance_id,replicate=r,kind=k))[:8],16)
            for k in ('engine','policy'))for r in range(self.rollouts))


def summarize_rollouts(rows,*,declared=64):
    if declared!=64 or len(rows)!=64:raise ValueError('all64 declared dispositions required')
    counts={k:0 for k in ('COMPLETE','CAPPED','REFUSED','MISSING')};values=[]
    for i,row in enumerate(rows):
        if row['replicate']!=i or row['status']not in counts:raise ValueError('fixed rollout order/disposition required')
        counts[row['status']]+=1;v=row.get('outcome')
        if row['status']=='COMPLETE':
            if type(v)not in (int,float) or v not in (-1,0,1):raise ValueError('actual terminal result required')
            values.append(float(v))
        elif v is not None:raise ValueError('nonterminal rollout cannot invent outcome')
    unknown=64-len(values);total=sum(values)
    # No conditional point mean is offered: known returns only bound the full
    # intended estimator until every declared continuation has a terminal.
    return dict(dispositions=counts,known=len(values),unknown=unknown,
        value=None if unknown else total/64,lower=(total-unknown)/64,upper=(total+unknown)/64,
        standard_error=None if unknown or len(values)<2 else float(np.std(values,ddof=1)/8),
        outcomes=values if not unknown else None)


def _rank(x):
    order=np.argsort(x,kind='stable');out=np.empty(len(x),float);start=0
    while start<len(x):
        end=start+1
        while end<len(x)and x[order[end]]==x[order[start]]:end+=1
        out[order[start:end]]=(start+end-1)/2;start=end
    return out


def spearman(x,y):
    if len(x)<2:return None
    a,b=_rank(np.asarray(x)),_rank(np.asarray(y));a-=a.mean();b-=b.mean()
    d=float(np.linalg.norm(a)*np.linalg.norm(b))
    return None if d==0 else float(a@b/d)


def calibration(predictions,targets,*,bins=15):
    """Per-position Brier against actual terminal draws, not squared mean error.

    Each target is a nonempty sequence of actual outcomes. With64 full draws
    every child has equal weight; root calibration has one original outcome.
    """
    if type(bins)is not int or bins<=0 or len(predictions)!=len(targets):raise ValueError('fixed calibration geometry required')
    if not predictions:return None
    p=np.asarray(predictions,float)
    if not np.isfinite(p).all()or np.any(np.abs(p)>1):raise ValueError('finite bounded head values required')
    q=(p+1)/2;ys=[];brier=[]
    for v,t in zip(q,targets):
        if not t or any(type(x)not in (int,float)or x not in (-1,0,1) for x in t):raise ValueError('actual terminal targets only')
        y=(np.asarray(t,float)+1)/2;ys.append(float(y.mean()));brier.append(float(np.mean((v-y)**2)))
    idx=np.minimum((q*bins).astype(int),bins-1);table=[];ece=0.
    for k in range(bins):
        mask=idx==k;n=int(mask.sum())
        row=dict(bin=k,count=n,predicted=None,observed=None)
        if n:
            row.update(predicted=float(q[mask].mean()),observed=float(np.asarray(ys)[mask].mean()))
            ece+=n/len(q)*abs(row['predicted']-row['observed'])
        table.append(row)
    return dict(positions=len(q),brier=float(np.mean(brier)),ece=float(ece),bins=table,
        scope='Conditional on complete actual targets; missing positions remain separate in full denominator')


def metrics(root_rows,child_rows,*,bins=15):
    """One checkpoint on the SAME truth: no head can select its test positions.

    root rows: selector, prediction, outcome(original), backup(optional exact
    all-reply grid). child rows: selector, action, prediction, rollouts(all64).
    Absent/nonfinite predictions are unknown, never removed from denominators.
    """
    def usable(v):return type(v)in (int,float)and math.isfinite(v)and abs(v)<=1
    roots=[r for r in root_rows if usable(r.get('prediction'))and r.get('outcome')in (-1,0,1)]
    rc=calibration([r['prediction']for r in roots],[[r['outcome']]for r in roots],bins=bins)
    complete=[];groups={tuple(r['selector'][k]for k in ('stream','frame','owner')):[]for r in root_rows}
    for r in child_rows:
        t=summarize_rollouts(r['rollouts']);entry=dict(r,truth=t);groups.setdefault(tuple(r['selector'][k]for k in ('stream','frame','owner')),[]).append(entry)
        if t['unknown']==0 and usable(r.get('prediction')):complete.append(entry)
    cc=calibration([r['prediction']for r in complete],[r['truth']['outcomes']for r in complete],bins=bins)
    rho=spearman([r['prediction']for r in complete],[r['truth']['value']for r in complete])
    top=[];unknown_groups=0;small_gap=0
    for key,rows in groups.items():
        if len(rows)<2 or any(r['truth']['unknown']or not usable(r.get('prediction'))for r in rows):unknown_groups+=1;continue
        ordered=sorted(rows,key=lambda r:(-r['truth']['value'],r['action']));a,b=ordered[:2];gap=a['truth']['value']-b['truth']['value']
        if gap<.05:small_gap+=1;continue
        top.append(dict(selector=dict(zip(('stream','frame','owner'),key)),actions=[a['action'],b['action']],gap=gap,
            correct=a['prediction']>b['prediction']))
    sensitivity=[r for r in top if r['gap']>=.10]
    backups=[r for r in roots if usable(r.get('backup'))]
    bc=calibration([r['backup']for r in backups],[[r['outcome']]for r in backups],bins=bins)
    same_head=calibration([r['prediction']for r in backups],[[r['outcome']]for r in backups],bins=bins)
    missing_children=len(child_rows)-len(complete);rate=None if not top else sum(r['correct']for r in top)/len(top)
    complete_denominator=(bool(root_rows)and bool(child_rows)and not missing_children and len(roots)==len(root_rows)and len(backups)==len(root_rows)and unknown_groups==0)
    bars=dict(ece=None if cc is None else cc['ece']<=.03,spearman=None if rho is None else rho>=.6,
        top_two=None if rate is None else rate>=.6,backup_lower=None if bc is None else bc['brier']<same_head['brier'])
    return dict(root_slots=len(root_rows),child_slots=len(child_rows),root_known=len(roots),root_unknown=len(root_rows)-len(roots),
        child_known=len(complete),child_unknown=missing_children,root_calibration=rc,child_calibration=cc,
        child_spearman=rho,spearman_full_denominator_bounds=[-1,1]if missing_children else None,
        top_two=dict(gap_units='expected_outcome[-1,1]',primary_gap=.05,probability_equivalent=.025,eligible_known=len(top),correct=sum(r['correct']for r in top),resolution=rate,
            unknown_groups=unknown_groups,below_gap_groups=small_gap,rows=top,
            full_denominator_resolution_bounds=None if not(len(top)+unknown_groups)else[sum(r['correct']for r in top)/(len(top)+unknown_groups),(sum(r['correct']for r in top)+unknown_groups)/(len(top)+unknown_groups)]),
        top_two_probability_gap_05_sensitivity=dict(descriptive_only=True,not_a_bar=True,raw_expected_outcome_gap=.10,probability_gap=.05,
            eligible_known=len(sensitivity),correct=sum(r['correct']for r in sensitivity),resolution=None if not sensitivity else sum(r['correct']for r in sensitivity)/len(sensitivity),unknown_groups=unknown_groups),
        backup=dict(known=len(backups),unknown=len(root_rows)-len(backups),calibration=bc,matched_head_calibration=same_head,
            scope='Fixed chance/single recorded world; frozen9375 reply policy over ALL legal replies; max over legal own actions'),
        uncertainty=dict(child_truth='64 terminal draws, per-child Monte Carlo stderr and missing-return bounds in truth file',
            aggregate_metric_intervals='NOT_ESTIMATED; roots/children within a game correlated; conditional point metrics are not confidence/strength claims',
            bias='Frozen raw9375 policy and original engine/world; root outcomes use original collector policy'),
        bars=bars,decision=('PASS'if all(v is True for v in bars.values())else'FAIL')if complete_denominator else'UNKNOWN_INCOMPLETE_DENOMINATOR',
        interpretation='Head-specific predictions, shared immutable raw9375 truth. Conditional metrics never establish an unconditional pass with missing rows.')


def folds(selectors,*,seed):
    """Five predeclared game clusters; all frames/siblings/rollouts stay together."""
    if type(seed)is not int:raise ValueError('predeclared split seed required')
    return {s['stream']:int(digest(dict(seed=seed,stream=s['stream']))[:8],16)%5 for s in selectors}


def partition(selectors,*,seed,holdout_fold=0):
    """B-trunk critic crossfit roles; no fitting/model work here.

    Repeat k=0..4; every prediction is test/out-of-fold exactly once. Small-step
    head-only fitting uses the other3 folds, selection the next fold (including
    original public-head warmstart epoch0). Never select epochs on test; report
    crossfit separately from any final critic fit on the entire bank.
    """
    if type(holdout_fold)is not int or not 0<=holdout_fold<5:raise ValueError('one fixed test fold required')
    return {stream:('test'if f==holdout_fold else'validation'if f==(holdout_fold+1)%5 else'train')for stream,f in folds(selectors,seed=seed).items()}
