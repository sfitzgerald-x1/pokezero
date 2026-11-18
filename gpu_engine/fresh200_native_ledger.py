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
        self.retained_status={};self.disable_causes={};self.trap_causes={};self.rest_states=set();self.encore_slots={}
        self.weather_creation=None;self.phase_epoch=0;self.old_stall=None;self.stall_refreshed={};self.lock_results={}
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

    @staticmethod
    def _effect(value):
        if type(value) is dict and value.get('kind')=='Effect' and type(value.get('id')) is str:
            return dict(id=to_id(value['id']),effect_type=value.get('effectType'),object=value.get('object'))
        return None

    def _cause(self, frame, explicit):
        value=self._effect(explicit)
        return value or self._effect(frame.enter.get('context',{}).get('effect'))

    def _enter_material(self, frame):
        if frame.operation=='Battle.endTurn':
            # Native refreshes live active caches here. Departed slot tokens
            # keep their old evidence until clearVolatile replaces the slots.
            before=frame.enter['before']
            for side in before['sides']:
                for ref in items(side['fields']['active']):
                    if not ref:continue
                    identity=pokemon(ref)
                    if member(before,identity)['hp']<=0:continue
                    self.disable_causes={k:v for k,v in self.disable_causes.items() if k[:2]!=identity}
                    self.trap_causes.pop(identity,None)
        elif frame.operation=='Battle.commitChoices':
            if frame.enter['before']['fields']['requestState']=='move':
                self.phase_epoch+=1;self.old_stall=None;self.stall_refreshed={}
        elif frame.operation=='Battle.runAction' and self.old_stall is None:
            action=fields(items(frame.enter['arguments'])[0])
            if action.get('choice')!='move':return
            self.old_stall=[]
            for side in frame.enter['before']['sides']:
                active=items(side['fields']['active']);need(len(active)==1,'Native singles stall-freeze actor required')
                volatile=fields(member(frame.enter['before'],pokemon(active[0]))['volatiles'])
                stall=fields(volatile['stall']) if 'stall' in volatile else None
                counter=stall['counter'] if stall is not None else 0
                need(type(counter) is int and counter>=0,'literal Native pre-dispatch stall counter required')
                self.old_stall.append(counter)

    def _faint(self, frame):
        identity=pokemon(frame.enter['subject']);before=member(frame.enter['before'],identity);after=member(frame.leave['after'],identity)
        if before['hp']>0 and after['hp']==0 and before['status']!='fnt':
            need(type(before['status']) is str,'literal Native pre-faint major status required')
            self.retained_status[identity]=dict(status=before['status'],state=before['statusState'],call_id=frame.call,sequence=frame.end)
            self.event('PRE_FAINT_MAJOR_STATUS',frame,actor=list(identity),status=before['status'])

    def _disable(self, frame):
        identity=pokemon(frame.enter['subject']);arguments=items(frame.enter['arguments'])
        need(arguments and type(arguments[0]) is str,'literal Native disableMove argument required')
        move=to_id(arguments[0]);cause=self._cause(frame,arguments[2] if len(arguments)>2 else None)
        before=items(member(frame.enter['before'],identity)['moveSlots']);after=items(member(frame.leave['after'],identity)['moveSlots'])
        need(len(before)==len(after),'Native disabling mutated slot roster')
        for slot,(old,new) in enumerate(zip(before,after)):
            old_fields=fields(old);new_fields=fields(new)
            if old_fields['id']!=move:continue
            need(old['object']==new['object'] and (new_fields['disabled'] is True or new_fields['disabled']=='hidden'),
                'Native disable attempt lacks its literal target cache')
            key=(*identity,old['object'])
            # Native loops over every matching slot, including an already-true
            # slot. Preserve ALL attempted causes, not only disabledSource's
            # last successful write; Choice/Encore overlap stays decomposable.
            self.disable_causes.setdefault(key,[]).append(dict(effect=cause,hidden=len(arguments)>1 and arguments[1] is True,
                physical_slot=slot,call_id=frame.call,sequence=frame.end))

    def _trap(self, frame):
        if frame.leave['result'] is not True:return
        identity=pokemon(frame.enter['subject']);after=member(frame.leave['after'],identity)
        need(after['trapped'] is True or after['trapped']=='hidden','Native successful tryTrap lacks combined trap cache')
        self.trap_causes.setdefault(identity,[]).append(dict(effect=self._cause(frame,None),call_id=frame.call,sequence=frame.end))

    def _condition_start(self, frame):
        if frame.leave['result'] is False:return
        arguments=items(frame.enter['arguments'])
        if len(arguments)<4 or arguments[0] not in ('Start','Restart'):return
        effect=self._effect(arguments[1]);target=arguments[3]
        if not effect or type(target) is not dict or target.get('kind')!='Pokemon':return
        if effect['id'] not in ('slp','stall','encore'):return
        identity=pokemon(target);state=arguments[2]
        need(type(state) is dict and type(state.get('object')) is int,'actual Native condition-state identity required')
        if effect['id']=='slp' and arguments[0]=='Start':
            source=self._effect(arguments[5]) if len(arguments)>5 else None
            if source and source['effect_type']=='Move' and source['id']=='rest':self.rest_states.add(state['object'])
        elif effect['id']=='stall':
            self.stall_refreshed[identity]=dict(epoch=self.phase_epoch,call_id=frame.call,sequence=frame.end)
        elif effect['id']=='encore' and arguments[0]=='Start':
            candidates=[]
            for getter in self.frames.values():
                if getter.operation!='Pokemon.getMoveData' or getter.end is None or not self.within(getter,frame):continue
                if pokemon(getter.enter['subject'])!=identity or getter.leave['result'] is None:continue
                result=getter.leave['result'];fields(result)
                owner=getter
                while owner is not None and 'before' not in owner.enter:owner=self.frames.get(owner.parent)
                need(owner is not None,'Encore slot object lacks literal Native cache')
                matches=[i for i,x in enumerate(items(member(owner.enter['before'],identity)['moveSlots'])) if x['object']==result['object']]
                need(len(matches)==1,'Encore actual slot object is ambiguous')
                candidates.append((result['object'],matches[0],getter.call))
            need(candidates and len({x[:2] for x in candidates})==1,'Encore lacks exact Native getMoveData slot witness')
            self.encore_slots[(*identity,state['object'])]=dict(physical_slot=candidates[-1][1],get_move_data_call_id=candidates[-1][2],sequence=frame.end)

    def _weather(self, frame):
        if frame.leave['result'] is not True:return
        arguments=items(frame.enter['arguments']);after=frame.leave['after']['fields']['field']
        if after['weather']=='':self.weather_creation=None;return
        state=fields(after['weatherState']);source=pokemon(state['source'])
        cause=self._cause(frame,arguments[2] if len(arguments)>2 else None)
        self.weather_creation=dict(weather=after['weather'],state_object=after['weatherState']['object'],source=list(source),
            effect=cause,call_id=frame.call,sequence=frame.end)
        self.event('WEATHER_SOURCE',frame,weather=self.weather_creation['weather'],state_object=self.weather_creation['state_object'],
            source=self.weather_creation['source'],effect=self.weather_creation['effect'])

    def _locks(self, frame):
        identity=pokemon(frame.enter['subject']);children=[f for f in self.frames.values() if f.parent==frame.call and f.end is not None]
        hard=[f for f in children if f.operation=='Pokemon.getLockedMove'];need(len(hard)==1,'actual Native hard-lock query required')
        hard_value=hard[0].leave['result'];need(hard_value is None or type(hard_value) is str,'literal Native hard-lock result required')
        semi=[f for f in children if f.operation=='Pokemon.getSemiLockedMove'];semi_value=None if hard_value else 'UNOBSERVED'
        semi_call=None
        if not hard_value and semi:
            need(len(semi)==1,'ambiguous Native semi-lock query')
            queried=any(f.parent==semi[0].call and f.operation=='Battle.priorityEvent' and f.end is not None for f in self.frames.values())
            if queried:
                semi_value=semi[0].leave['result'];need(semi_value is None or type(semi_value) is str,'literal Native semi-lock result required')
                semi_call=semi[0].call
        self.lock_results[identity]=dict(hard=hard_value,semi=semi_value,hard_call_id=hard[0].call,semi_call_id=semi_call,
            request_call_id=frame.call,sequence=frame.end)

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
                self._enter_material(frame)
            elif operation=='CALL_RETURN':
                value=row['frame'];need(self.stack and self.stack[-1]==value['callId'],'Native causal return is unbalanced')
                frame=self.frames[self.stack.pop()];need(value['operation']==frame.operation,'Native causal operation differs')
                frame.leave=row['evidence'];frame.end=row['sequence']
                if frame.operation=='Side.commitChoices':self._commit(frame)
                elif frame.operation=='BattleActions.switchIn':self._switch(frame)
                elif frame.operation=='Pokemon.moveUsed':self._dispatch(frame)
                elif frame.operation=='Pokemon.transformInto':self._transform(frame)
                elif frame.operation=='Side.addSlotCondition':self._wish(frame)
                elif frame.operation=='Pokemon.faint':self._faint(frame)
                elif frame.operation=='Pokemon.disableMove':self._disable(frame)
                elif frame.operation=='Pokemon.tryTrap':self._trap(frame)
                elif frame.operation=='Battle.singleEvent':self._condition_start(frame)
                elif frame.operation=='Field.setWeather':self._weather(frame)
                elif frame.operation=='Pokemon.getMoveRequestData':self._locks(frame)
                elif frame.operation=='Pokemon.clearVolatile':
                    identity=pokemon(frame.enter['subject']);self.trap_causes.pop(identity,None);self.lock_results.pop(identity,None)
                    self.disable_causes={k:v for k,v in self.disable_causes.items() if k[:2]!=identity}
            elif operation=='COMMAND_END':
                need(not self.stack,'Native command ended with open causal frames');self.frames.clear()
        return self

    def snapshot(self):
        need(self.initial and not self.stack,'bound Native initialization and closed causal frames required')
        return dict(last_action_kind=list(self.last_kind),last_action_index=list(self.last_index),
            wish_amount=list(self.wish_amount),transform_records=[dict(actor=list(k),**v) for k,v in sorted(self.transforms.items())],
            retained_status=[dict(actor=list(k),**v) for k,v in sorted(self.retained_status.items())],
            disable_causes=[dict(actor=list(k[:2]),slot_object=k[2],causes=v) for k,v in sorted(self.disable_causes.items())],
            trap_causes=[dict(actor=list(k),causes=v) for k,v in sorted(self.trap_causes.items())],
            rest_state_objects=sorted(self.rest_states),encore_slots=[dict(actor=list(k[:2]),state_object=k[2],**v) for k,v in sorted(self.encore_slots.items())],
            weather_creation=self.weather_creation,phase_epoch=self.phase_epoch,old_stall=self.old_stall,
            stall_refreshed=[dict(actor=list(k),**v) for k,v in sorted(self.stall_refreshed.items())],
            lock_results=[dict(actor=list(k),**v) for k,v in sorted(self.lock_results.items())],
            events=list(self.events),source_registry_hash=self.source_registry_hash,
            qualification=False,causal_coverage_complete=False,
            open_gates=['queue/residual chronology','complete disable/trap causes','all73 physical projector','literal owner-prefix binding'])
