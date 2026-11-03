"""Fixed-root construction and latency summaries, including refused slots."""
from collections import Counter
import math
from .phase_d_saved_roots import address,band


def timing_roster(selected):
    if len(selected)<200:raise ValueError('original>=200 root construction roster required')
    groups=[[]for _ in range(4)]
    for row in selected:groups[band(row['turn'])].append(row)
    if any(len(g)<5 for g in groups):raise ValueError('five original metadata-selected roots per turn band required')
    return tuple(row for group in groups for row in group[:5])


def refusal_kind(reason):
    if reason.startswith('original-public-mapper:'):return ':'.join(reason.split(':')[:2])
    return reason.split(':',1)[0]


def summarize(roster,records,*,worlds,constructor_only):
    if worlds not in (2,8,16,32)or len(roster)<(200 if constructor_only else 20):raise ValueError('fixed root/budget denominator required')
    expected={address(row):row for row in roster}
    if len(expected)!=len(roster):raise ValueError('duplicate fixed root')
    found={}
    for row in records:
        key=tuple(row['address'])
        if key not in expected or key in found or row['requested_worlds']!=worlds:raise ValueError('foreign/duplicate result or world budget')
        full=row['constructed_worlds']
        if type(full)is not int or not 0<=full<=worlds:raise ValueError('literal constructed count required')
        found[key]=row
    by_turn={};kinds=Counter();times=[];constructed=timed=0
    for key,source in expected.items():
        row=found.get(key);turn=source['turn'];counts=by_turn.setdefault(turn,dict(requested=0,fully_constructed=0,timed=0))
        counts['requested']+=1
        if row is None:kinds['UNAVAILABLE_RESULT']+=1;continue
        full=row['constructed_worlds']==worlds;constructed+=full;counts['fully_constructed']+=full
        for kind in set(row['refusals']):kinds[refusal_kind(kind)]+=1
        seconds=row.get('decision_seconds')
        if seconds is not None:
            if not full or type(seconds)not in (int,float)or not math.isfinite(seconds)or seconds<0:raise ValueError('full finite actual search time required')
            times.append(seconds);timed+=1;counts['timed']+=1
    n=len(roster);rank=math.ceil(.95*n)-1;ordered=sorted(times+[math.inf]*(n-timed));p95=ordered[rank]
    return dict(requested_roots=n,requested_worlds=worlds,fully_constructed_roots=constructed,
        full_constructor_coverage=constructed/n,refusals_by_root_kind=dict(kinds),by_turn=by_turn,
        search_timed_roots=timed,search_untimed_roots=n-timed,decision_p95_seconds=p95 if math.isfinite(p95)else None,
        decision_p95_unavailable=not math.isfinite(p95),
        admission=constructed/n>=.9 if constructor_only else n>=20 and p95<=30.,
        scope='CONSTRUCTOR_ONLY_NO_SEARCH_TIME_CLAIM'if constructor_only else'FULL_FIXED_ROSTER_SEARCH_P95_UNTIMED_INFINITY',
        stop_rule='Constructor miss: one refusal-driven source batch and one remeasurement, then stop. No root redraw/refill.')


def _retired_claims(root,path,digest,*,declaration_sha256,count):
    """Read an explicitly closed attempt, never infer closure from a claim.

    The root owner supplies durable terminal receipts after stopping dispatch
    and verifying all workers terminated. Bind those original bytes, and the
    request/claim/START bytes, without manufacturing a scored result file.
    """
    import hashlib
    from pathlib import Path
    from .phase_d_ladder_panel import read_bound,validate,checked_cell
    if path is None:
        if digest is not None:raise ValueError('retired-claim manifest path required')
        return {}
    manifest=read_bound(path,digest)
    if (manifest.get('schema')!='pokezero.champion-search-ladder.retired-claims.v1'
            or manifest.get('declaration_sha256')!=declaration_sha256):
        raise ValueError('same original retired-claim declaration required')
    reference=manifest['closure'];closure=read_bound(reference['path'],reference['sha256'])
    if (closure.get('schema')!='pokezero.champion-search-ladder.attempt-closure.v1'
            or closure.get('declaration_sha256')!=declaration_sha256
            or type(closure.get('attempt_id'))is not str or not closure['attempt_id'].strip()
            or closure.get('state')!='CLOSED' or closure.get('dispatch_stopped')is not True
            or closure.get('all_workers_terminated')is not True):
        raise ValueError('retired claims require a closed attempt and terminated workers')
    evidence=closure.get('terminal_evidence')
    if type(evidence)is not list or not evidence:raise ValueError('original terminal attempt evidence required')
    for receipt in evidence:
        if hashlib.sha256(Path(receipt['path']).read_bytes()).hexdigest()!=receipt['sha256']:
            raise ValueError('original terminal attempt evidence differs')
    rows=manifest['cells'];ordinals=[row['cell']['ordinal']for row in rows]
    if (len(set(ordinals))!=len(ordinals)or sorted(ordinals)!=sorted(closure.get('retired_ordinals',()))
            or any(type(n)is not int or n%6 or not 0<=n<6*count for n in ordinals)):
        raise ValueError('exact fixed-prefix retired RAW cells required')
    retired={}
    for row in rows:
        cell=row['cell'];ordinal=cell['ordinal']
        if (root/'results'/f'{ordinal}.json').exists():raise ValueError('retired claim already has a result')
        request=read_bound(root/'requests'/f'{ordinal}.json',row['request_sha256'])
        claim=read_bound(root/'claims'/f'{ordinal}.json',row['claim_sha256'])
        start=read_bound(root/'starts'/f'{ordinal}.json',row['start_sha256'])
        if (request!=claim or request.get('schema')!='pokezero.champion-search-ladder.raw-request.v1'
                or request.get('cell')!=cell or request.get('declaration_sha256')!=declaration_sha256
                or Path(request['run_directory']).resolve()!=root
                or start!={'cell':cell,'request_sha256':row['request_sha256']}):
            raise ValueError('same original retired request/claim/start required')
        declaration=validate(read_bound(request['declaration'],declaration_sha256))
        checked_cell(declaration,cell)
        retired[ordinal]=dict(cell=cell,status='UNKNOWN',disposition='RETIRED_CLAIM_NO_RESULT',
            request_sha256=row['request_sha256'],claim_sha256=row['claim_sha256'],start_sha256=row['start_sha256'],
            closure_sha256=reference['sha256'],attempt_id=closure['attempt_id'])
    return retired


def freeze_prefix(run_directory,*,declaration_sha256,count=64,seed=20261103,
                  retired_claims=None,retired_claims_sha256=None):
    """Freeze first64 declared RAW games; closed status never selects survivors.

    Caller waits for this exact prefix to close. Every durable prefix in its
    captures is eligible, including caps/failures/partial tails. Other RAW
    games may continue appending; their files are never opened here.
    """
    from pathlib import Path
    import json
    from . import phase_d_owning_capture as capture,phase_d_saved_roots as roots
    root=Path(run_directory).resolve()
    if count!=64:raise ValueError('prospectively fixed first64 RAW cells required')
    retired=_retired_claims(root,retired_claims,retired_claims_sha256,
        declaration_sha256=declaration_sha256,count=count)
    rows=[];captures=0;closed=[]
    for pair in range(count):
        ordinal=6*pair;path=root/'results'/f'{ordinal}.json'
        if path.exists():result=json.loads(path.read_bytes())
        elif ordinal in retired:result=retired[ordinal]
        else:raise ValueError('fixed RAW prefix not closed: '+str(pair))
        cell=result['cell']
        if (cell.get('arm')!='RAW'or cell.get('pair_ordinal')!=pair or cell.get('ordinal')!=ordinal
                or (ordinal not in retired and result.get('declaration_sha256')!=declaration_sha256)
                or result.get('status')not in ('COMPLETE','UNKNOWN','FAILED','CAPPED')):
            raise ValueError('original closed RAW prefix cell binding differs')
        closed.append(dict(pair_ordinal=pair,status=result['status'],
            **({k:result[k]for k in ('disposition','request_sha256','claim_sha256','start_sha256','closure_sha256','attempt_id')}
               if ordinal in retired else {})))
        wire=root/'owning-wire'/f'{ordinal}.jsonl'
        if not wire.exists():continue
        captures+=1
        for row in capture.read_capture(wire):
            if row['cell']!=cell or row['declaration_sha256']!=declaration_sha256:raise ValueError('same original RAW owning capture required')
            rows.append(row)
    declaration=roots.declare(count=200,seed=seed)
    selected=roots.select(rows,declaration)
    return dict(declaration=declaration,selected=selected,capture_prefix=dict(pair_ordinals=[0,count-1],declared_games=count,
        games_with_capture=captures,games_without_capture=count-captures,eligible_root_records=len(rows),closed_cells=closed,
        retired_claim_manifest_sha256=retired_claims_sha256,
        rule='First64 declared RAW cells, all closed dispositions and retired partial captures retained; no score/constructor/outcome selection.'))
