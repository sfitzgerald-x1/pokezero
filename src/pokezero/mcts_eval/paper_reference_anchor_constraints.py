"""Necessary public predicates on ORIGINAL owned hypothetical anchor draws.

Every accepted full history entails M. Rejection conditioning on M therefore
preserves the normalized posterior: its constant P(M) cancels. No hidden actual
state is an input, no set is forced, and native errors/censoring still refuse.
"""
import hashlib
import re
from .paper_reference import ReferenceRefusal
from .paper_reference_pending import public_history


def identifier(value):
    return re.sub('[^a-z0-9]', '', value.lower())


def details(value):
    fields=[part.strip() for part in value.split(',')]
    return dict(species=identifier(fields[0]),
        gender=next((p for p in fields[1:] if p in ('M','F')),None),
        level=next((int(p[1:]) for p in fields[1:] if re.fullmatch(r'L\d+',p)),100))


def prepare_constraints(state, expected, subject):
    before=public_history(state)
    if expected[:len(before)]!=before:
        raise ReferenceRefusal('anchor constraints need an exact public prefix')
    opponent='p2' if subject=='p1' else 'p1'
    result=dict(schema='pokezero.necessary-public-anchor-constraints.v1',opponent=opponent,
        public_history_sha256=hashlib.sha256('\n'.join(expected).encode()).hexdigest(),
        static_details=[],own_encore=None,complete_proposals=0,matches=0,rejections={},
        exact_posterior=False,predicate_law='original anchor conditional on necessary predicate; constant normalization cancels')
    rows={}
    ambiguous=any((len(p:=line.split('|'))>1 and p[1].lstrip('-').lower() in
        ('transform','formechange','detailschange','replace')) or
        any(token in line.lower() for token in ('|transform|','illusion','imposter')) for line in expected)
    if not ambiguous:
        for line in expected:
            p=line.split('|')
            if len(p)>=5 and p[1] in ('switch','drag') and p[2].startswith(opponent+'a: '):
                row=details(p[3]); key=row['species']
                if key in rows and rows[key]!=row:
                    ambiguous=True; break
                rows[key]=row
        if not ambiguous: result['static_details']=list(rows.values())
    result['static_details_disabled_ambiguity']=ambiguous
    from ..local_showdown import _public_reference_encore
    certificate=_public_reference_encore(state,subject)
    if certificate is None: return result
    # Hard timer conditioning requires actor-known sufficient PP, even under
    # Pressure. Opponent PP and special PP drains are never assumed.
    paid,uses,active,ending=0,0,None,False
    for line in before:
        p=line.split('|')
        if len(p)>2 and p[1] in ('switch','drag') and p[2].startswith(subject+'a: '): active=p[2]
    if active is None: return result
    for line in expected[len(before):]:
        p=line.split('|'); kind=p[1] if len(p)>1 else ''; actor=p[2] if len(p)>2 else ''
        if (kind in ('faint','replace','transform','detailschange','formechange')
                or kind in ('switch','drag') and actor.startswith(subject+'a: ')
                or kind=='move' and len(p)>3 and identifier(p[3]) in ('spite','grudge','mimic','sketch','transform')):
            return result
        if kind=='move' and actor==active:
            if identifier(p[3])!=certificate['move']: return result
            uses+=1
        if kind=='upkeep': paid+=1
        if kind=='-start' and actor==active and p[3:4]==['Encore']: return result
        if kind=='-end' and actor==active and p[3:4]==['Encore']:
            ending=True; break
    if not ending: return result
    active_request=state.self_request.get('active',[])
    moves=active_request[0].get('moves',[]) if len(active_request)==1 else []
    slots=[m for m in moves if identifier(m.get('id',m.get('move','')))==certificate['move']]
    if len(slots)!=1 or type(slots[0].get('pp')) is not int or slots[0]['pp']<=2*uses: return result
    required=paid+1
    if required not in certificate['remaining_candidates']:
        raise ReferenceRefusal('necessary public Encore ending contradicts anchor support')
    result['own_encore']=dict(side=subject,ident=active,move=certificate['move'],
        required_remaining=required,observed_residuals_before_end=paid,
        maximum_pp_payments=2*uses,actor_known_pp=slots[0]['pp'])
    return result


def anchor_matches(env, anchor, constraints):
    if not getattr(env,'_search_snapshot_permitted',False):
        raise ReferenceRefusal('necessary constraints may inspect only owned hypothetical worlds')
    lock=constraints['own_encore']
    if lock is not None:
        evidence=anchor.get('encore_conditioning',{}).get(lock['side'])
        if not isinstance(evidence,dict): raise ReferenceRefusal('hypothetical anchor omitted Encore provenance')
        if evidence.get('sampled_remaining')!=lock['required_remaining']:
            return 'different necessary own Encore remaining'
    rows=constraints['static_details']
    if not rows: return None
    sides=env.snapshot().bridge_snapshot['battle']['sides']
    parties=[s for s in sides if s['id']==constraints['opponent']]
    if len(parties)!=1: raise ReferenceRefusal('hypothetical party missing native side')
    actual={}
    for mon in parties[0]['pokemon']:
        value=mon.get('details')
        if not isinstance(value,str): raise ReferenceRefusal('hypothetical native details missing')
        row=details(value)
        if row['species'] in actual: raise ReferenceRefusal('hypothetical species identity ambiguous')
        actual[row['species']]=row
    for row in rows:
        other=actual.get(row['species'])
        if other is None or other['level']!=row['level'] or row['gender']!=other['gender']:
            return 'different necessary public static details'
    return None


def conditional_encore_support(original, required):
    """Original uniform independent timer draw conditional on necessary M.

    The guard is proved before construction from public actor-known PP and the
    observed residual ledger. This removes a constant 1/len(original) factor;
    it does not set a timer outside the original support or change a native roll.
    Historical newly-applied Encore remains weighted native seed-bank guidance.
    """
    if (not isinstance(original,list) or not original or len(set(original))!=len(original)
            or any(type(v) is not int or v<1 for v in original)
            or type(required) is not int or required not in original):
        raise ReferenceRefusal('direct necessary Encore support outside original timer kernel')
    return [required],dict(original_support=list(original),conditional_support=[required],
        removed_constant_probability=1/len(original),native_timer_not_overridden_outside_support=True,
        deterministic_stream_coupling_changed=True,
        independent_kernel_law='original uniform timer conditional on necessary own public ending')
