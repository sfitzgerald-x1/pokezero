"""Move a NECESSARY already-observed-history indicator before resampling.

No native outcome is modified. On every path compatible with the complete
observations, each guarded indicator M is 1; on other paths final likelihood
is 0. Thus L_full * product(M) == L_full, even with repeated indicators. Original
full-support proposal corrections still apply. This is not an additional
likelihood probability or permission to turn a heuristic roll hint into truth.
"""
import hashlib

from .paper_reference import ReferenceRefusal
from .paper_reference_pending import public_history
from .paper_reference_anchor_constraints import prepare_constraints


def necessary_own_encore_potential(env, current, expected, subject):
    if not getattr(env,'_search_snapshot_permitted',False):
        raise ReferenceRefusal('early Encore potential requires owned hypothetical world')
    history=public_history(current)
    if expected[:len(history)]!=history:
        raise ReferenceRefusal('early Encore potential requires exact observed public prefix')
    receipt=dict(schema='pokezero.necessary-own-encore-potential.v1',applicable=False,
        accepted=True,indicator=1,live_hidden_state_used=False,native_outcome_modified=False,
        public_history_sha256=hashlib.sha256('\n'.join(history).encode()).hexdigest(),
        terminal_history_sha256=hashlib.sha256('\n'.join(expected).encode()).hexdigest(),
        law='full observed history implies every guarded early indicator; product is redundant with final likelihood')
    if 'encore' not in current.replay.volatiles.get(subject,()):
        receipt['reason']='no own active public Encore'; return True,receipt
    if current.self_request.get('forceSwitch') or current.deferred_opponent_action_player is not None:
        receipt['reason']='ambiguous nonordinary request boundary'; return True,receipt
    constraints=prepare_constraints(current,expected,subject)
    if constraints['static_details_disabled_ambiguity']:
        receipt['reason']='species-changing or conflicting public details'; return True,receipt
    lock=constraints['own_encore']
    if lock is None:
        receipt['reason']='no proven sufficient-PP undisrupted own Encore ending'; return True,receipt
    shell=env.snapshot().bridge_snapshot['battle']
    sides=[s for s in shell['sides'] if s['id']==subject]
    if len(sides)!=1: raise ReferenceRefusal('hypothetical own side absent in native Encore potential')
    active=[p for p in sides[0]['pokemon'] if p.get('isActive') is True]
    if len(active)!=1: raise ReferenceRefusal('hypothetical own active identity ambiguous')
    volatile=active[0].get('volatiles',{}).get('encore')
    if not isinstance(volatile,dict) or type(volatile.get('duration')) is not int or volatile['duration']<1:
        raise ReferenceRefusal('hypothetical native Encore timer missing or invalid')
    actual=volatile['duration']; accepted=actual==lock['required_remaining']
    receipt.update(applicable=True,accepted=accepted,indicator=int(accepted),certificate=lock,
        hypothetical_native_remaining=actual)
    return accepted,receipt
