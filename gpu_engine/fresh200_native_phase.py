"""Pure Native decision/suspension phase and sealed input retention contract.

Only closed passive Native boundaries and original submitted input bindings
are accepted. No expected State/Trackers, inferred opponent choice or PRNG is
used. Unrepresented queued actions and missing stall/commit history are fatal.
"""
from dataclasses import asdict, dataclass

from .fresh200_native_identity import PartyMap,reference_identifier


class UnmappedNativePhase(ValueError):
    pass


def need(value,message):
    if not value:raise UnmappedNativePhase(message)


@dataclass(frozen=True)
class OrdinaryInputs:
    actions: tuple
    # Per side: (original physical actor, actual committed Native move ID), or
    # None for an original non-move. Filled only after independent choice join.
    native_moves: tuple
    commit_call_ids: tuple


@dataclass(frozen=True)
class PhaseProof:
    kind: str
    turn: int
    active_turns: tuple
    requested: tuple
    pending_residual: bool
    queued_actor: int
    queued_actions: tuple
    queued_stall: tuple
    snapshot_sequence: int

    def record(self):return asdict(self)


def phase_from_native(snapshot,party:PartyMap,*,original_inputs=None,old_stall=None):
    battle=snapshot['battle'];proof=snapshot['witness'];receipt=proof['receipt']
    need(proof.get('requestOrigin')=='materialized_activeRequest' and receipt['poisoned'] is False
        and receipt['boundary'].get('requestOrigin')==proof['requestOrigin']
        and receipt['openFrameCount']==0 and receipt['control']['phase']=='snapshot', 'closed passive Native decision snapshot required')
    turn=battle.get('turn');need(type(turn) is int and turn>=1,'literal Native turn required')
    queue=battle.get('queue');need(type(queue) is list and all(type(x) is dict for x in queue),'complete actual Native queue required')
    requests=snapshot['boundaryRequests'];need(type(requests) is dict,'literal materialized Native requests required')
    requested=[];forced=[]
    for side in (0,1):
        request=requests.get(f'p{side+1}')
        need(request is None or type(request) is dict,'literal Native request or absence required')
        force=request.get('forceSwitch') if request else None
        need(force is None or type(force) is list and len(force)<=1 and all(type(x) is bool for x in force),'Native singles forced request required')
        forced.append(bool(force and any(force)))
        requested.append(request is not None and not request.get('wait',False) and (bool(request.get('active')) or forced[-1]))
    clocks=tuple(party.mon(s,party.active(s)).get('activeTurns') for s in (0,1))
    need(all(type(x) is int for x in clocks),'actual Native active clocks required')
    ended=battle.get('ended');need(type(ended) is bool,'literal Native terminal bit required')
    if ended:
        need(not any(requested),'terminal Native boundary retains an actionable private request')
        return PhaseProof('terminal',turn,clocks,tuple(requested),False,-1,(0,0),(0,0),receipt['lastSequence'])
    state=battle.get('requestState')
    if state=='move':
        need(queue==[] and battle.get('midTurn') is False and requested==[True,True] and not any(forced),
            'ordinary Native decision has pending queue or inconsistent requests')
        return PhaseProof('ordinary',turn-1,tuple(x-1 for x in clocks),tuple(requested),False,-1,(0,0),(0,0),receipt['lastSequence'])
    need(state=='switch' and any(forced),'unrepresented Native decision phase')
    need(all(x.get('choice') in ('move','residual') for x in queue),'unrepresented material Native queued action')
    residual=[x for x in queue if x['choice']=='residual'];moves=[x for x in queue if x['choice']=='move']
    need(len(residual)<=1 and len(moves)<=1,'ambiguous remaining Native residual/opponent action')
    actor=-1;actions=(0,0);stall=(0,0)
    if moves:
        need(bool(residual),'remaining Native move lacks its original residual queue continuation')
        actor,original=party.resolve_pokemon(moves[0]['pokemon'])
        need(not requested[actor] and original==party.active(actor),'pending Native actor is not the unchanged nonrequesting side')
        need(isinstance(original_inputs,OrdinaryInputs) and len(original_inputs.actions)==2
            and all(type(x) is int and 0<=x<9 for x in original_inputs.actions)
            and len(original_inputs.native_moves)==2 and len(original_inputs.commit_call_ids)==2 and len(set(original_inputs.commit_call_ids))==2
            and all(type(x) is int and x>=0 for x in original_inputs.commit_call_ids), 'sealed original inputs and actual Native commit join required')
        move=moves[0].get('moveid') or reference_identifier(moves[0].get('move'),'Move')
        need(original_inputs.native_moves[actor]==(original,move) and original_inputs.actions[actor]<4,
            'remaining Native queued move differs from original accepted input')
        need(type(old_stall) in (list,tuple) and len(old_stall)==2 and all(type(x) is int and x>=0 for x in old_stall),
            'actual Native old-stall freeze required for pending actor')
        actions=tuple(original_inputs.actions);stall=tuple(old_stall)
    pending=bool(residual)
    return PhaseProof('replacement-before-residual' if pending else 'replacement-after-residual',turn,clocks,
        tuple(requested),pending,actor,actions,stall,receipt['lastSequence'])


class ReplacementCounter:
    """Count actual completed resumed packets once, never snapshot reads."""
    def __init__(self):self.value=0;self.commands=set()
    def completed(self,nonce,*,before:PhaseProof,after:PhaseProof):
        need(type(nonce) is str and nonce and nonce not in self.commands,'unique closed Native replacement command required')
        self.commands.add(nonce)
        self.value=self.value+1 if before.pending_residual and after.pending_residual else 0
        return self.value


def retained_inputs(phase:PhaseProof,ordinary_packet):
    """External buffers come from the original INPUT packet, never GPU state."""
    shapes={'draws':(2,12),'phazing':(2,),'secondary':(2,),'sleep_talk':(2,),'effect':(2,5)}
    if phase.pending_residual:
        need(type(ordinary_packet) is dict and all(k in ordinary_packet for k in shapes),'sealed original chance packet required')
        from .fresh200_native import _operand,DRAW_BOUNDS,EFFECT_BOUNDS
        bounds=dict(draws=DRAW_BOUNDS,phazing=60,secondary=100,sleep_talk=60,effect=EFFECT_BOUNDS)
        return {name:_operand(ordinary_packet[name],shape,bounds[name],name) for name,shape in shapes.items()}
    def zero(shape):return [zero(shape[1:]) for _ in range(shape[0])] if shape else 0
    return {name:ordinary_packet[name] if phase.pending_residual else zero(shape) for name,shape in shapes.items()}
