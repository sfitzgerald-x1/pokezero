"""Fold closed passive Native frames into literal causal-history evidence.

No GPU state or request-slot guess is accepted. Original physical identities
and actual Native move-slot object tokens establish dispatch. This is one part
of the full physical projector: queue/residual/cause and owner-prefix admission
remain separate, explicitly unqualified gates.
"""
from dataclasses import dataclass
import re

from .fresh200_native_identity import to_id


class UnmappedNativeHistory(ValueError):
    pass


def need(value, message):
    if not value:raise UnmappedNativeHistory(message)


def fields(value):
    need(type(value) is dict and set(value)=={'object','fields'} and type(value['object']) is int,
        'literal Native data-object projection required')
    pairs=value['fields'];need(type(pairs) is list and all(type(x) is list and len(x)==2 and type(x[0]) is str for x in pairs),
        'literal Native field pairs required')
    result=dict(pairs);need(len(result)==len(pairs),'duplicate Native field identity')
    return result


def items(value):
    need(type(value) is dict and set(value)=={'object','items'} and type(value['object']) is int and type(value['items']) is list,
        'literal Native array projection required')
    return value['items']


def pokemon(value):
    need(type(value) is dict and value.get('kind')=='Pokemon' and type(value.get('object')) is int
        and type(value.get('side')) is int and value['side'] in (0,1)
        and type(value.get('original')) is int and 0<=value['original']<6, 'resolved original Native Pokemon required')
    return value['side'],value['original']


def move_id(value):
    if type(value) is str:return to_id(value)
    need(type(value) is dict and value.get('kind')=='Effect' and value.get('effectType')=='Move',
        'literal Native Move identity required')
    return to_id(value['id'])


def member(cache, identity):
    side,original=identity;need(type(cache) is dict and len(cache.get('sides',[]))==2,'complete Native member cache required')
    found=[m for m in cache['sides'][side]['pokemon'] if m['identity'].get('side')==side and m['identity'].get('original')==original]
    need(len(found)==1,'unique original Native member cache required')
    return found[0]['fields']


@dataclass
class Frame:
    call: int
    parent: object
    operation: str
    sequence: int
    enter: dict
    leave: object = None
    end: object = None


class NativeHistory:
    def __init__(self, *, battle_id, source_registry_hash):
        need(type(battle_id) is str and battle_id and type(source_registry_hash) is str and re.fullmatch('[a-f0-9]{64}',source_registry_hash),
            'explicit source-bound Native history identity required')
        self.battle_id=battle_id;self.source_registry_hash=source_registry_hash
        self.frames={};self.stack=[];self.events=[];self.pending_switches=[]
        self.last_kind=[0,0];self.last_index=[-1,-1];self.wish_amount=[0,0];self.transforms={}
        self.initial=False

    def bind_initial(self, certificate):
        need(not self.initial and not self.events and not self.stack and certificate.get('field_count')==73 and certificate.get('initial_history_absence_proved') is True
            and certificate.get('source_registry_hash')==self.source_registry_hash,
            'complete same-source initial Native history certificate required')
        self.initial=True

    def ancestor(self, frame, operation):
        current=frame.parent
        while current is not None:
            parent=self.frames[current]
            if parent.operation==operation:return parent
            current=parent.parent
        return None

    def within(self, frame, root):
        current=frame
        while current is not None:
            if current.call==root.call:return True
            current=self.frames.get(current.parent)
        return False

    def event(self, kind, frame, **value):
        self.events.append(dict(kind=kind,sequence=frame.end,call_id=frame.call,operation=frame.operation,**value))

    def _slot(self, frame, identity, move):
        root=self.ancestor(frame,'BattleActions.runMove')
        need(root is not None,'dispatch lacks actual Native runMove frame')
        candidates=[]
        for getter in self.frames.values():
            if (getter.operation!='Pokemon.getMoveData' or getter.end is None or getter.end>=frame.sequence
                    or not self.within(getter,root) or pokemon(getter.enter['subject'])!=identity):continue
            arguments=items(getter.enter['arguments'])
            if not arguments or move_id(arguments[0])!=move or getter.leave['result'] is None:continue
            result=getter.leave['result'];fields(result)  # validate literal slot object, retaining its Native token
            owner=getter
            while owner is not None and 'before' not in owner.enter:owner=self.frames.get(owner.parent)
            need(owner is not None,'actual Native getMoveData slot lacks material before-cache')
            slots=items(member(owner.enter['before'],identity)['moveSlots'])
            matches=[i for i,slot in enumerate(slots) if slot.get('object')==result['object']]
            need(len(matches)==1,'actual Native move-slot object does not resolve uniquely')
            need(fields(slots[matches[0]])['id']==move,'actual Native dispatch/slot ID disagrees')
            candidates.append((result['object'],matches[0],getter.call))
        need(candidates and len({(x[0],x[1]) for x in candidates})==1,
            'missing or ambiguous actual Native dispatch-slot witness; no ID/ordinal fallback')
        return candidates[-1][1],candidates[-1][2]

    def _commit(self, frame):
        side=frame.enter['subject'].get('n');need(type(side) is int and side in (0,1),'literal Native committing side required')
        choice=fields(frame.enter['before']['sides'][side]['fields']['choice'])
        for action in items(choice['actions']):
            action=fields(action);kind=action['choice'];need(kind in ('move','switch','pass'),'unrepresented Native committed choice')
            if kind=='switch':
                target=pokemon(action['target']);need(target[0]==side,'Native switch target crosses sides')
                self.pending_switches.append(dict(side=side,target=target[1],call=frame.call))
            self.event('COMMITTED_CHOICE',frame,side=side,choice=action)

    def _switch(self, frame):
        if not frame.leave['result']:return
        arguments=items(frame.enter['arguments']);target=pokemon(arguments[0])
        action=self.ancestor(frame,'Battle.runAction')
        if action is None:return
        resolved=fields(items(action.enter['arguments'])[0])
        if resolved.get('choice')!='switch':return  # actual initial entry or forced phazing is not a chosen switch
        need(pokemon(resolved['target'])==target,'Native queued and installed switch target differs')
        matches=[x for x in self.pending_switches if (x['side'],x['target'])==target]
        need(len(matches)==1,'installed Native chosen switch lacks unique committed target')
        installed=items(frame.leave['after']['sides'][target[0]]['fields']['active'])
        need(len(installed)==1 and pokemon(installed[0])==target,'Native successful switch did not install its literal target')
        self.pending_switches.remove(matches[0]);self.last_kind[target[0]]=1;self.last_index[target[0]]=target[1]
        self.event('INSTALLED_CHOSEN_SWITCH',frame,side=target[0],original_target=target[1],commit_call_id=matches[0]['call'])

    def _dispatch(self, frame):
        identity=pokemon(frame.enter['subject']);move=move_id(items(frame.enter['arguments'])[0]);side=identity[0]
        need(self.ancestor(frame,'BattleActions.runMove') is not None,'Native moveUsed lacks dispatch ancestry')
        if move=='struggle':slot=-1;getter=None;kind=3
        else:slot,getter=self._slot(frame,identity,move);kind=2
        self.last_kind[side]=kind;self.last_index[side]=-1 if move=='transform' else slot
        self.event('MOVE_USED',frame,side=side,original_actor=identity[1],move_id=move,physical_slot=slot,get_move_data_call_id=getter)

    def _transform(self, frame):
        identity=pokemon(frame.enter['subject']);donor=pokemon(items(frame.enter['arguments'])[0])
        if self.ancestor(frame,'BattleActions.runMove') is not None:self.last_index[identity[0]]=-1
        success=frame.leave['result'] is True
        self.event('TRANSFORM_ATTEMPT',frame,actor=list(identity),donor=list(donor),succeeded=success)
        if success:
            before=member(frame.enter['before'],donor);after=member(frame.leave['after'],identity)
            need(after['species']==before['species'],'Native Transform copied species lacks exact donor evidence')
            self.transforms[identity]=dict(donor=list(donor),donor_effective_species=before['species'],
                saved={name:after[name] for name in ('baseStoredStats','baseMoveSlots','baseHpType','baseHpPower')},
                call_id=frame.call,sequence=frame.end)

    def _wish(self, frame):
        arguments=items(frame.enter['arguments'])
        if frame.leave['result'] is not True or len(arguments)<2 or arguments[1]!='wish':return
        side=frame.enter['subject'].get('n');need(type(side) is int and side in (0,1),'literal Native Wish side required')
        slots=items(frame.leave['after']['sides'][side]['fields']['slotConditions'])
        need(len(slots)==1,'literal Native singles Wish slot required')
        wish=fields(fields(slots[0])['wish']);source=pokemon(wish['source'])
        need(source[0]==side,'unrepresented cross-side Native Wish source')
        hp=member(frame.enter['before'],source)['maxhp'];need(type(hp) is int and hp>0,'actual Native Wish source maxHP required')
        self.wish_amount[side]=hp//2
        self.event('CREATED_WISH',frame,side=side,source=list(source),source_maxhp=hp,amount=hp//2)

    def consume(self, rows):
        """Fold rows already verified by fresh200_witness_audit against the
        independent source/command roster; this never replaces that audit.
        """
        for row in rows:
            need(row['battleId']==self.battle_id,'Native history battle binding differs')
            operation=row['operation']
            if operation=='CALL_THROW':raise UnmappedNativeHistory('original Native exception makes history unsuccessful')
            if operation=='CALL_ENTER':
                value=row['frame'];call=value['callId'];parent=self.stack[-1] if self.stack else None
                need(call not in self.frames and value['parentCall']==parent,'Native causal frame gap/alias')
                frame=Frame(call,parent,value['operation'],row['sequence'],row['evidence']);self.frames[call]=frame;self.stack.append(call)
            elif operation=='CALL_RETURN':
                value=row['frame'];need(self.stack and self.stack[-1]==value['callId'],'Native causal return is unbalanced')
                frame=self.frames[self.stack.pop()];need(value['operation']==frame.operation,'Native causal operation differs')
                frame.leave=row['evidence'];frame.end=row['sequence']
                if frame.operation=='Side.commitChoices':self._commit(frame)
                elif frame.operation=='BattleActions.switchIn':self._switch(frame)
                elif frame.operation=='Pokemon.moveUsed':self._dispatch(frame)
                elif frame.operation=='Pokemon.transformInto':self._transform(frame)
                elif frame.operation=='Side.addSlotCondition':self._wish(frame)
            elif operation=='COMMAND_END':
                need(not self.stack,'Native command ended with open causal frames');self.frames.clear()
        return self

    def snapshot(self):
        need(self.initial and not self.stack,'bound Native initialization and closed causal frames required')
        return dict(last_action_kind=list(self.last_kind),last_action_index=list(self.last_index),
            wish_amount=list(self.wish_amount),transform_records=[dict(actor=list(k),**v) for k,v in sorted(self.transforms.items())],
            events=list(self.events),source_registry_hash=self.source_registry_hash,
            qualification=False,causal_coverage_complete=False,
            open_gates=['queue/residual chronology','complete disable/trap causes','all73 physical projector','literal owner-prefix binding'])
