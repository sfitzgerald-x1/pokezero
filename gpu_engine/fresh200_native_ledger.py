"""Fold closed passive Native frames into literal causal-history evidence.

No GPU state or request-slot guess is accepted. Original physical identities
and actual Native move-slot object tokens establish dispatch. This is one part
of the full physical projector: queue/residual/cause and owner-prefix admission
remain separate, explicitly unqualified gates.
"""
from dataclasses import dataclass
import re

from .fresh200_native_identity import to_id


SWITCH_CHOICES = frozenset(('switch', 'instaswitch'))
# Gen 3 Shadow Tag onFoeTrapPokemon in the pinned Native callback catalog.
SHADOW_TAG_TRAP_CALLBACK = 'Callback:404ef8be16446e7ea614f5f86395c8da0718dc0f3ea85965fd0bbf4fac41bd34'
# Gen 3 Magnet Pull onAnyTrapPokemon in the pinned Native callback catalog.
MAGNET_PULL_TRAP_CALLBACK = 'Callback:2d81d02b8604a58a7a605db2898fd18e1cfa6d5c00ef52e6658ab0afc491c8e7'


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


def retired_trap_actives(cache, victim, replacement):
    for ref in (victim, replacement):
        if items(cache['sides'][ref['side']]['fields']['active']) != [ref]:
            return False
    live = member(cache, pokemon(replacement))
    return (live['hp'] > 0 and live.get('fainted') is False
            and live.get('faintQueued') is False and live['ability'] == 'rockhead')


@dataclass
class Frame:
    call: int
    parent: object
    operation: str
    sequence: int
    enter: dict
    leave: object = None
    end: object = None
    trap_writers_before_faint_flush: object = None


class NativeHistory:
    def __init__(self, *, battle_id, source_registry_hash):
        need(type(battle_id) is str and battle_id and type(source_registry_hash) is str and re.fullmatch('[a-f0-9]{64}',source_registry_hash),
            'explicit source-bound Native history identity required')
        self.battle_id=battle_id;self.source_registry_hash=source_registry_hash
        self.frames={};self.stack=[];self.events=[];self.pending_switches=[]
        self.last_kind=[0,0];self.last_index=[-1,-1];self.wish_amount=[0,0];self.transforms={}
        self.retained_status={};self.disable_causes={};self.trap_causes={};self.rest_states=set();self.encore_slots={}
        self.last_active_install = [-1, -1]
        self.last_trap_refresh = -1
        self.double_faint_trap_proofs = {}
        self.retired_fainted_trap_proofs = {}
        self.weather_creation=None;self.phase_epoch=0;self.old_stall=None;self.stall_refreshed={};self.lock_results={}
        self.dispatch_slots={};self.charge_slots={}
        self.initial=False

    def bind_constructor(self, rows):
        """Bind a continuing Native start command BEFORE folding startup events.

        The caller first verifies these exact rows with the independent source
        and command roster. Unlike the offline startup certificate, this does
        not require ending/closing a live battle or erasing entry events.
        """
        need(not self.initial and not self.frames and not self.events,'fresh Native history constructor binding required')
        begins=[r for r in rows if r['operation']=='COMMAND_BEGIN'];ends=[r for r in rows if r['operation']=='COMMAND_END']
        need(len(begins)==len(ends)==1 and begins[0]['phase']==ends[0]['phase']=='start'
            and all(r['battleId']==self.battle_id for r in rows),'one independently verified Native start command required')
        players=[r for r in rows if r['operation']=='CALL_ENTER' and r['frame']['operation']=='Battle.setPlayer']
        need(len(players)==2 and [r['evidence'].get('side') for r in players]==['p1','p2'],
            'literal ordered Native p1/p2 constructors required')
        forbidden={'Side.chooseMove','Side.chooseSwitch','Side.commitChoices','Pokemon.moveUsed','Pokemon.transformInto'}
        need(not any(r['operation']=='CALL_THROW' or r['operation']=='CALL_ENTER' and r['frame']['operation'] in forbidden for r in rows),
            'Native constructor already contains action history')
        self.initial=True

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
        if candidates:
            need(len({(x[0],x[1]) for x in candidates})==1,'ambiguous actual Native dispatch-slot witness')
            return candidates[-1][1],candidates[-1][2],candidates[-1][0],False
        # A locked release legitimately skips deductPP/getMoveData. Its slot
        # must be the actual original charge token, still in this actor's
        # current roster, with an actual same-runMove hard-lock query.
        hard=[f for f in self.frames.values() if f.operation=='Pokemon.getLockedMove' and f.end is not None
            and f.end<frame.sequence and self.within(f,root) and pokemon(f.enter['subject'])==identity and f.leave['result']==move]
        need(hard,'missing actual Native dispatch-slot witness; no ID/ordinal fallback')
        volatile=fields(member(root.enter['before'],identity).get('volatiles'))
        charge=volatile.get('twoturnmove');record=self.charge_slots.get(charge.get('object')) if type(charge) is dict else None
        need(hard and record and record['actor']==identity and record['move_id']==move,
            'missing actual Native dispatch-slot witness; no ID/ordinal fallback')
        need(not any(f.operation=='Pokemon.deductPP' and self.within(f,root) for f in self.frames.values()),
            'locked release has contradictory PP-deduction evidence')
        slots=items(member(frame.enter['before'],identity)['moveSlots'])
        matches=[i for i,x in enumerate(slots) if x.get('object')==record['slot_object']]
        need(matches==[record['physical_slot']] and fields(slots[matches[0]])['id']==move,
            'stale or copied Native charge slot cannot establish execution history')
        return matches[0],record['get_move_data_call_id'],record['slot_object'],True

    def _commit(self, frame):
        side=frame.enter['subject'].get('n');need(type(side) is int and side in (0,1),'literal Native committing side required')
        before = frame.enter['before']
        choice = fields(before['sides'][side]['fields']['choice'])
        for action in items(choice['actions']):
            action = fields(action)
            kind = action['choice']
            need(kind in ('move', 'pass') or kind in SWITCH_CHOICES, 'unrepresented Native committed choice')
            if kind in SWITCH_CHOICES:
                forced = before.get('fields', {}).get('requestState') == 'switch'
                need((kind == 'instaswitch') == forced, 'Native switch commit differs from its request phase')
                actor, target = pokemon(action['pokemon']), pokemon(action['target'])
                need(actor[0] == target[0] == side and actor != target, 'Native switch actor/target differs')
                self.pending_switches.append(dict(side=side, actor=actor, target=target, kind=kind, call=frame.call))
            self.event('COMMITTED_CHOICE',frame,side=side,choice=action)

    def _installed_switch(self, frame, target):
        installed = items(frame.leave['after']['sides'][target[0]]['fields']['active'])
        need(len(installed) == 1 and pokemon(installed[0]) == target,
             'Native successful switch did not install its literal target')
        self.last_kind[target[0]] = 1
        self.last_index[target[0]] = target[1]

    def _switch(self, frame):
        if not frame.leave['result'] or frame.leave['result'] == 'pursuitfaint':return
        arguments=items(frame.enter['arguments']);target=pokemon(arguments[0])
        installs_before = tuple(self.last_active_install)
        previous_install = installs_before[target[0]]
        if frame.leave['result'] is True:
            self.last_active_install[target[0]] = frame.end
            self.retired_fainted_trap_proofs.clear()
        drag = self.ancestor(frame, 'BattleActions.dragIn')
        is_drag = len(arguments) == 4 and arguments[3] is True
        need(is_drag == (drag is not None), 'Native switch drag flag differs from its causal call')
        if is_drag:
            need(arguments[2] is None and frame.leave['result'] is True,
                 'forced Native drag did not report installation')
            # A drag invalidates the departing active's move slot. Keep the
            # incoming member's identity without inventing a committed choice.
            self._installed_switch(frame, target)
            self.event('INSTALLED_DRAG', frame, side=target[0], original_target=target[1], drag_call_id=drag.call)
            return
        action=self.ancestor(frame,'Battle.runAction')
        if action is None:return
        resolved = fields(items(action.enter['arguments'])[0])
        kind = resolved.get('choice')
        if kind not in SWITCH_CHOICES:
            return  # Initial entry has no committed switch choice.
        if kind == 'instaswitch':
            need(frame.leave['result'] is True, 'forced Native replacement did not report installation')
        actor = pokemon(resolved['pokemon'])
        need(pokemon(resolved['target']) == target, 'Native queued and installed switch target differs')
        matches = [choice for choice in self.pending_switches
                   if choice['actor'] == actor and choice['target'] == target and choice['kind'] == kind]
        need(len(matches)==1,'installed Native chosen switch lacks unique committed target')
        self._installed_switch(frame, target)
        if kind == 'instaswitch' and actor in self.retained_status:
            before, after = frame.enter['before'], frame.leave['after']
            departing = items(before['sides'][actor[0]]['fields']['active'])
            old, cleared = member(before, actor), member(after, actor)
            prior = self.retained_status[actor]
            if (len(departing) == 1 and pokemon(departing[0]) == actor and actor != target
                    and old['fainted'] is True and old['hp'] == 0 and old['status'] in ('fnt', prior['status'])
                    and cleared['fainted'] is True and cleared['hp'] == 0 and cleared['status'] == ''):
                prior['replacement_clear_sequence'] = frame.end
        self._retire_arena_trap(frame, actor, target, kind, installs_before)
        self._retire_shadow_tag(frame, actor, target, previous_install)
        self.pending_switches.remove(matches[0])
        self.event('INSTALLED_CHOSEN_SWITCH',frame,side=target[0],original_target=target[1],commit_call_id=matches[0]['call'])

    def _retire_arena_trap(self, frame, actor, target, kind, installs_before):
        if frame.leave['result'] is not True or actor == target or not self.trap_causes:
            return
        before, after = frame.enter['before'], frame.leave['after']
        side = actor[0]
        outgoing = items(before['sides'][side]['fields']['active'])
        installed = items(after['sides'][side]['fields']['active'])
        if len(outgoing) != 1 or not outgoing[0] or pokemon(outgoing[0]) != actor:
            return
        victims = items(before['sides'][1 - side]['fields']['active'])
        if len(victims) != 1 or not victims[0] or victims != items(after['sides'][1 - side]['fields']['active']):
            return
        victim = pokemon(victims[0])
        causes = [cause for cause in self.trap_causes.get(victim, ()) if cause.get('arena_trap_writer')]
        if not causes:
            return
        earlier, current = member(before, victim), member(after, victim)
        if (earlier['hp'] <= 0 or current['hp'] <= 0 or earlier['trapped'] != 'hidden'
                or current['trapped'] != 'hidden'):
            return
        for cause in causes:
            writer = cause['arena_trap_writer']
            if (writer['source'] == outgoing[0] and writer['victim'] == victims[0]
                    and writer['epoch'] + 1 == self.phase_epoch and cause['sequence'] < frame.sequence
                    and 'arena_trap_departure' not in cause):
                cause['arena_trap_departure'] = dict(replacement=installed[0], enter=frame.sequence,
                    end=frame.end, epoch=self.phase_epoch, call_id=frame.call, choice=kind,
                    installs_before=installs_before)

    def _retire_shadow_tag(self, frame, actor, target, previous_install):
        if frame.leave['result'] is not True or actor == target or not self.trap_causes:
            return
        before, after = frame.enter['before'], frame.leave['after']
        side = actor[0]
        outgoing = items(before['sides'][side]['fields']['active'])
        installed = items(after['sides'][side]['fields']['active'])
        opposing = items(before['sides'][1 - side]['fields']['active'])
        if (len(outgoing) != 1 or not outgoing[0] or pokemon(outgoing[0]) != actor
                or len(opposing) != 1 or not opposing[0]
                or opposing != items(after['sides'][1 - side]['fields']['active'])):
            return
        victim = pokemon(opposing[0])
        for cause in self.trap_causes.get(victim, ()):
            writer = cause.get('shadow_tag_writer')
            if (not writer or 'shadow_tag_departure' in cause or writer['source'] != outgoing[0]
                    or writer['victim'] != opposing[0] or writer['epoch'] + 1 != self.phase_epoch
                    or max(previous_install, self.last_active_install[victim[0]]) >= cause['sequence']
                    or cause['sequence'] >= frame.sequence):
                continue
            live = [member(cache, victim) for cache in (before, after)]
            source, replacement = member(before, actor), member(after, target)
            if (not all(mon['hp'] > 0 and mon['fainted'] is False and mon['trapped'] is True for mon in live)
                    or source['hp'] <= 0 or source['fainted'] is not False or source['ability'] != 'shadowtag'
                    or replacement['hp'] <= 0 or replacement['fainted'] is not False):
                continue
            cause['shadow_tag_departure'] = dict(replacement=installed[0], enter=frame.sequence,
                                                  end=frame.end, epoch=self.phase_epoch)

    def _dispatch(self, frame):
        identity=pokemon(frame.enter['subject']);move=move_id(items(frame.enter['arguments'])[0]);side=identity[0]
        need(self.ancestor(frame,'BattleActions.runMove') is not None,'Native moveUsed lacks dispatch ancestry')
        if move=='struggle':slot=-1;getter=None;kind=3;slot_object=None;retained=False
        else:slot,getter,slot_object,retained=self._slot(frame,identity,move);kind=2
        self.last_kind[side]=kind;self.last_index[side]=-1 if move=='transform' else slot
        self.dispatch_slots[identity]=dict(actor=identity,move_id=move,physical_slot=slot,slot_object=slot_object,
            get_move_data_call_id=getter,run_move_call_id=self.ancestor(frame,'BattleActions.runMove').call)
        self.event('MOVE_USED',frame,side=side,original_actor=identity[1],move_id=move,physical_slot=slot,
            get_move_data_call_id=getter,slot_object=slot_object,retained_charge_witness=retained)

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
        # Gen 3 inherits the literal slotCondition "Wish". Native resolves it
        # to the "wish" key before installing the condition.
        if frame.leave['result'] is not True or len(arguments)<2 or arguments[1] not in ('Wish','wish'):return
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
            self.last_trap_refresh = frame.sequence
            self.retired_fainted_trap_proofs.clear()
            for side in before['sides']:
                for ref in items(side['fields']['active']):
                    if not ref:continue
                    identity=pokemon(ref)
                    if member(before,identity)['hp']<=0:continue
                    self.disable_causes={k:v for k,v in self.disable_causes.items() if k[:2]!=identity}
                    self.trap_causes.pop(identity,None)
                    self.double_faint_trap_proofs.pop(identity, None)
        elif frame.operation=='Battle.faintMessages' and self.trap_causes:
            frame.trap_writers_before_faint_flush = {
                actor: tuple(dict(cause) for cause in causes) for actor, causes in self.trap_causes.items()
            }
        elif frame.operation=='Battle.commitChoices':
            self.retired_fainted_trap_proofs.clear()
            if frame.enter['before']['fields']['requestState']=='move':
                self.phase_epoch+=1;self.old_stall=None;self.stall_refreshed={}
        elif frame.operation=='Pokemon.tryTrap' and self.retired_fainted_trap_proofs:
            self.retired_fainted_trap_proofs.pop(pokemon(frame.enter['subject']), None)
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
        need(type(before.get('fainted')) is bool and type(before.get('faintQueued')) is bool
            and type(after.get('faintQueued')) is bool,'literal Native faint-queue transition required')
        if not before['fainted'] and not before['faintQueued'] and after['faintQueued'] and after['hp']==0 and before['status']!='fnt':
            need(type(before['status']) is str,'literal Native pre-faint major status required')
            self.retained_status[identity]=dict(status=before['status'],state=before['statusState'],call_id=frame.call,sequence=frame.end)
            self.event('PRE_FAINT_MAJOR_STATUS',frame,actor=list(identity),status=before['status'])

    def _bind_arena_trap_double_faint(self, frame):
        if not frame.trap_writers_before_faint_flush:
            return
        for victim, causes in frame.trap_writers_before_faint_flush.items():
            retained = [dict(cause) for cause in causes]
            for cause in retained:
                writer = cause.get('arena_trap_writer')
                if (not writer or 'arena_trap_departure' in cause or 'arena_trap_double_faint' in cause
                        or writer['epoch'] + 1 != self.phase_epoch
                        or max(self.last_active_install) >= cause['sequence']):
                    continue
                source = pokemon(writer['source'])
                if victim not in self.retained_status or source not in self.retained_status:
                    continue
                victim_faint = self.frames.get(self.retained_status[victim]['call_id'])
                source_faint = self.frames.get(self.retained_status[source]['call_id'])
                if (victim_faint is None or source_faint is None
                        or not cause['sequence'] < victim_faint.end < source_faint.end < frame.sequence):
                    continue
                if (victim_faint.enter['subject'] != writer['victim']
                        or source_faint.enter['subject'] != writer['source']):
                    continue
                action = self.ancestor(victim_faint, 'BattleActions.runMove')
                if action is None or action is not self.ancestor(source_faint, 'BattleActions.runMove'):
                    continue
                arguments = items(action.enter['arguments'])
                if len(arguments) < 2 or move_id(arguments[0]) != 'explosion' or arguments[1] != writer['victim']:
                    continue
                before, after = frame.enter['before'], frame.leave['after']
                snapshots = [cache for row in (victim_faint, source_faint, frame)
                             for cache in (row.enter['before'], row.leave['after'])]
                if not all(items(cache['sides'][ref['side']]['fields']['active']) == [ref]
                           for cache in snapshots for ref in (writer['victim'], writer['source'])):
                    continue
                if not all(member(cache, victim)['trapped'] == 'hidden' for cache in snapshots):
                    continue
                old, new = member(before, victim), member(after, victim)
                if any(cache['trapped'] != 'hidden' or cache['maybeTrapped'] is not True or cache['hp'] != 0
                       for cache in (old, new)):
                    continue
                source_before, source_after = member(before, source), member(after, source)
                if (old['fainted'] is False and new['fainted'] is True and source_before['hp'] == source_after['hp'] == 0
                        and source_before['fainted'] is False and source_after['fainted'] is True):
                    cleared = {pokemon(row.enter['subject']) for row in self.frames.values()
                               if row.operation == 'Pokemon.clearVolatile' and row.end is not None
                               and self.within(row, frame)}
                    if not {victim, source} <= cleared:
                        continue
                    cause['arena_trap_double_faint'] = dict(victim_faint=victim_faint.end, source_faint=source_faint.end,
                        flush_start=frame.sequence, flush_end=frame.end, epoch=self.phase_epoch)
            if any('arena_trap_double_faint' in cause for cause in retained):
                self.double_faint_trap_proofs[victim] = retained

    def _bind_shadow_tag_replacement_faint(self, frame):
        if frame.leave['result'] is not False:
            return
        for victim, causes in self.trap_causes.items():
            for cause in causes:
                writer = cause.get('shadow_tag_writer')
                departure = cause.get('shadow_tag_departure')
                if (not writer or not departure or 'shadow_tag_replacement_faint' in cause
                        or departure['epoch'] != self.phase_epoch
                        or self.last_active_install[1 - victim[0]] != departure['end']
                        or self.last_active_install[victim[0]] >= cause['sequence']):
                    continue
                replacement = pokemon(departure['replacement'])
                retained = self.retained_status.get(replacement)
                faint = self.frames.get(retained['call_id']) if retained else None
                if (faint is None or faint.enter['subject'] != departure['replacement']
                        or not departure['end'] < faint.sequence < faint.end < frame.sequence):
                    continue
                before, after = frame.enter['before'], frame.leave['after']
                snapshots = (faint.enter['before'], faint.leave['after'], before, after)
                if not all(items(cache['sides'][ref['side']]['fields']['active']) == [ref]
                           for cache in snapshots for ref in (writer['victim'], departure['replacement'])):
                    continue
                survivors = [member(cache, victim) for cache in snapshots]
                if not all(mon['hp'] > 0 and mon['fainted'] is False and mon['trapped'] is True
                           for mon in survivors):
                    continue
                old, new = member(before, replacement), member(after, replacement)
                if not (old['hp'] == new['hp'] == 0 and old['fainted'] is False and new['fainted'] is True):
                    continue
                if not any(child.operation == 'Pokemon.clearVolatile' and child.end is not None
                           and child.enter['subject'] == departure['replacement'] and self.within(child, frame)
                           and items(child.enter['arguments']) == [False] for child in self.frames.values()):
                    continue
                cause['shadow_tag_replacement_faint'] = dict(faint=faint.end, flush_start=frame.sequence,
                                                             flush_end=frame.end)


    def _bind_fainted_magnet_pull(self, frame):
        if frame.leave['result'] is not False:
            return
        for victim, causes in self.trap_causes.items():
            for cause in causes:
                writer = cause.get('magnet_pull_writer')
                if (not writer or 'magnet_pull_source_faint' in cause
                        or writer['epoch'] + 1 != self.phase_epoch
                        or max(self.last_active_install) >= cause['sequence']):
                    continue
                source = pokemon(writer['source'])
                retained = self.retained_status.get(source)
                faint = self.frames.get(retained['call_id']) if retained else None
                if (faint is None or faint.enter['subject'] != writer['source']
                        or not cause['sequence'] < faint.sequence < faint.end < frame.sequence):
                    continue
                before, after = frame.enter['before'], frame.leave['after']
                snapshots = (faint.enter['before'], faint.leave['after'], before, after)
                if not all(items(cache['sides'][ref['side']]['fields']['active']) == [ref]
                           for cache in snapshots for ref in (writer['victim'], writer['source'])):
                    continue
                survivors = [member(cache, victim) for cache in snapshots]
                if not all(mon['hp'] > 0 and mon['fainted'] is False and mon['trapped'] == 'hidden'
                           for mon in survivors):
                    continue
                old, new = member(before, source), member(after, source)
                if not (old['hp'] == new['hp'] == 0 and old['fainted'] is False and new['fainted'] is True):
                    continue
                if not any(child.operation == 'Pokemon.clearVolatile' and child.end is not None
                           and child.enter['subject'] == writer['source'] and self.within(child, frame)
                           and items(child.enter['arguments']) == [False] for child in self.frames.values()):
                    continue
                cause['magnet_pull_source_faint'] = dict(faint=faint.end, flush_start=frame.sequence,
                                                        flush_end=frame.end, epoch=self.phase_epoch)

    def _retired_faint_witness(self, frame, victim, cause):
        writer = cause.get('arena_trap_writer')
        departure = cause.get('arena_trap_departure')
        status = self.retained_status.get(victim)
        if (not writer or writer.get('hidden_argument') is not True
                or not departure or departure.get('choice') != 'switch' or not status):
            return None
        source, target = writer['source'], departure['replacement']
        if (writer['epoch'] + 1 != departure['epoch'] or departure['epoch'] != self.phase_epoch
                or max(departure['installs_before']) >= cause['sequence']
                or self.last_active_install[victim[0]] >= cause['sequence']
                or self.last_active_install[target['side']] != departure['end']
                or self.last_trap_refresh >= cause['sequence']
                or pokemon(source) == pokemon(target) or source['object'] == target['object']):
            return None
        faint = self.frames.get(status['call_id'])
        switch = self.frames.get(departure['call_id'])
        if faint is None or switch is None or not departure['end'] < faint.sequence < faint.end < frame.sequence:
            return None
        victim_ref = writer['victim']
        if faint.enter['subject'] != victim_ref:
            return None
        action = self.ancestor(faint, 'BattleActions.runMove')
        if action is None:
            return None
        move_args, faint_args = items(action.enter['arguments']), items(faint.enter['arguments'])
        if (len(move_args) < 2 or move_args[1] != victim_ref or move_id(move_args[0]) != 'explosion'
                or len(faint_args) != 2 or faint_args[0] != victim_ref):
            return None
        effect = self._effect(faint_args[1])
        if not effect or (effect['id'], effect['effect_type']) != ('explosion', 'Move'):
            return None
        before, queued = member(faint.enter['before'], victim), member(faint.leave['after'], victim)
        if (before['hp'] <= 0 or before.get('fainted') is not False or before.get('faintQueued') is not False
                or queued['hp'] != 0 or queued.get('fainted') is not False or queued.get('faintQueued') is not True):
            return None
        clears = []
        for call in self.frames.values():
            if not self.within(call, frame):
                continue
            if call.operation == 'Battle.commitChoices':
                return None
            if call.operation in ('Pokemon.tryTrap', 'Pokemon.clearVolatile') and pokemon(call.enter['subject']) == victim:
                if call.operation == 'Pokemon.tryTrap' or call.enter['subject'] != victim_ref:
                    return None
                clears.append(call)
        if len(clears) != 1:
            return None
        clear, = clears
        args = items(clear.enter['arguments'])
        if (self.ancestor(clear, 'Battle.faintMessages') is not frame
                or len(args) != 1 or args[0] is not False
                or not frame.sequence < clear.sequence < clear.end < frame.end):
            return None
        flush = (frame.enter['before'], clear.enter['before'], clear.leave['after'], frame.leave['after'])
        snapshots = (switch.leave['after'], faint.enter['before'], faint.leave['after'], *flush)
        if not all(retired_trap_actives(cache, victim_ref, target) for cache in snapshots):
            return None
        for cache in snapshots:
            state = member(cache, victim)
            if state['trapped'] != 'hidden' or state.get('maybeTrapped') is not True:
                return None
        if (any(member(cache, victim)['hp'] != 0 for cache in flush)
                or member(flush[0], victim).get('fainted') is not False
                or member(flush[-1], victim).get('fainted') is not True
                or member(flush[-1], victim).get('faintQueued') is not True):
            return None
        # Showdown drains the queue and sets fainted without clearing faintQueued.
        return dict(epoch=self.phase_epoch, faint_call_id=faint.call, faint_enter=faint.sequence,
            faint_end=faint.end, clear_call_id=clear.call, clear_enter=clear.sequence, clear_end=clear.end,
            flush_call_id=frame.call, flush_start=frame.sequence, flush_end=frame.end)

    def _bind_retired_fainted_trap(self, frame):
        if not frame.trap_writers_before_faint_flush:
            return
        for victim, causes in frame.trap_writers_before_faint_flush.items():
            retained = [dict(cause) for cause in causes]
            for cause in retained:
                witness = self._retired_faint_witness(frame, victim, cause)
                if witness is not None:
                    cause['arena_trap_retired_faint'] = witness
            if any('arena_trap_retired_faint' in cause for cause in retained):
                self.retired_fainted_trap_proofs[victim] = retained

    def _disable(self, frame):
        identity=pokemon(frame.enter['subject']);arguments=items(frame.enter['arguments'])
        need(arguments and type(arguments[0]) is str,'literal Native disableMove argument required')
        move=to_id(arguments[0]);cause=self._cause(frame,arguments[2] if len(arguments)>2 else None)
        old_member=member(frame.enter['before'],identity)
        before=items(old_member['moveSlots']);after=items(member(frame.leave['after'],identity)['moveSlots'])
        lock = None
        if cause and 'volatiles' in old_member:
            live = fields(old_member['volatiles'])
            if cause['id'] in ('choicelock', 'encore'):
                lock = live.get(cause['id'])
            elif cause['id'] == 'choiceband' and cause['effect_type'] == 'Item' and 'choicelock' in live:
                choice = live['choicelock']
                if self._effect(fields(choice).get('sourceEffect')) == cause:
                    lock = choice
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
                physical_slot=slot,call_id=frame.call,sequence=frame.end,lock_object=None if lock is None else lock['object']))

    def _clear_move_restrictions(self, frame, identity):
        if not any(key[:2] == identity for key in self.disable_causes):
            # Constructor clears can occur before the side is installed.
            return
        before = items(member(frame.enter['before'], identity)['moveSlots'])
        after = items(member(frame.leave['after'], identity)['moveSlots'])
        old = {slot['object']: fields(slot) for slot in before}
        # clearVolatile slices baseMoveSlots. A base slot can retain its cached
        # disable flag and source even though the lock itself was removed.
        retained = set()
        for slot in after:
            token = slot['object']
            value = fields(slot)
            prior = old.get(token)
            if prior and value['disabled'] in (True, 'hidden') and value['disabled'] == prior['disabled']:
                source = value.get('disabledSource')
                if type(source) is str and source and source == prior.get('disabledSource'):
                    retained.add(token)
        self.disable_causes = {key: causes for key, causes in self.disable_causes.items()
                               if key[:2] != identity or key[2] in retained}
        for key, causes in self.disable_causes.items():
            if key[:2] == identity:
                for cause in causes:
                    cause['clear_volatile_sequence'] = frame.end

    def _trap(self, frame):
        if frame.leave['result'] is not True:return
        identity=pokemon(frame.enter['subject']);after=member(frame.leave['after'],identity)
        need(after['trapped'] is True or after['trapped']=='hidden','Native successful tryTrap lacks combined trap cache')
        effect = self._cause(frame, None)
        cause = dict(effect=effect, call_id=frame.call, sequence=frame.end)
        if effect and (effect['id'], effect['effect_type']) == ('arenatrap', 'Ability') and after['trapped'] == 'hidden':
            context = frame.enter.get('context', {})
            state = context.get('effectState')
            if type(state) is dict and 'fields' in state:
                state = fields(state)
                source = state.get('target')
                if state.get('id') == 'arenatrap' and type(source) is dict and source.get('kind') == 'Pokemon':
                    sid, slot = pokemon(source)
                    before = frame.enter['before']
                    sources = items(before['sides'][sid]['fields']['active'])
                    victims = items(before['sides'][identity[0]]['fields']['active'])
                    owner = member(before, (sid, slot))
                    if (sid != identity[0] and sources == [source] and victims == [frame.enter['subject']]
                            and owner['hp'] > 0 and owner['ability'] == 'arenatrap'
                            and member(before, identity)['hp'] > 0 and member(before, identity)['trapped'] is False):
                        args = items(frame.enter['arguments'])
                        cause['arena_trap_writer'] = dict(source=source, victim=victims[0], epoch=self.phase_epoch,
                            hidden_argument=len(args) == 1 and args[0] is True)
        if effect and (effect['id'], effect['effect_type']) == ('magnetpull', 'Ability'):
            writer = self._magnet_pull_writer(frame, identity)
            if writer:
                cause['magnet_pull_writer'] = writer
        self.trap_causes.setdefault(identity, []).append(cause)

    def _shadow_tag_write(self, frame):
        event = self.frames.get(frame.parent)
        arguments = items(frame.enter['arguments'])
        if not event or event.operation != 'Battle.runEvent' or not arguments:
            return
        victim = arguments[0]
        if items(event.enter['arguments'])[:2] != ['TrapPokemon', victim]:
            return
        context = frame.enter.get('context', {})
        after_context = frame.leave.get('context', {})
        effect = self._cause(frame, None)
        state = context.get('effectState')
        if (not effect or (effect['id'], effect['effect_type']) != ('shadowtag', 'Ability')
                or not state or 'fields' not in state or after_context.get('effectState') != state
                or after_context.get('effect') != context.get('effect')
                or frame.leave['result'] != {'$native': 'undefined'}):
            return
        state = fields(state)
        source = state.get('target')
        if state.get('id') != 'shadowtag' or not source or source.get('kind') != 'Pokemon':
            return
        holder, identity = pokemon(source), pokemon(victim)
        if holder[0] == identity[0]:
            return
        before, after = frame.enter['before'], frame.leave['after']
        if not all(items(cache['sides'][ref['side']]['fields']['active']) == [ref]
                   for cache in (before, after) for ref in (source, victim)):
            return
        if not all(member(cache, holder)['ability'] == 'shadowtag'
                   and all(member(cache, actor)['hp'] > 0 and member(cache, actor)['fainted'] is False
                           for actor in (holder, identity)) for cache in (before, after)):
            return
        if member(before, identity)['trapped'] is not False or member(after, identity)['trapped'] is not True:
            return
        writer = dict(source=source, victim=victim, epoch=self.phase_epoch)
        self.trap_causes.setdefault(identity, []).append(dict(effect=effect, call_id=frame.call,
            sequence=frame.end, shadow_tag_writer=writer))


    def _magnet_pull_writer(self, frame, identity):
        callback = self.frames.get(frame.parent)
        if not callback or callback.operation != MAGNET_PULL_TRAP_CALLBACK:
            return None
        event = self.frames.get(callback.parent)
        victim = frame.enter['subject']
        context = frame.enter.get('context', {})
        callback_context = callback.enter.get('context', {})
        state = context.get('effectState')
        arguments = items(frame.enter['arguments'])
        if (not event or event.operation != 'Battle.runEvent'
                or items(event.enter['arguments'])[:2] != ['TrapPokemon', victim]
                or items(callback.enter['arguments'])[:1] != [victim]
                or arguments != [True] or arguments[0] is not True
                or not state or 'fields' not in state
                or callback_context.get('effectState') != state
                or callback_context.get('effect') != context.get('effect')):
            return None
        state = fields(state)
        source = state.get('target')
        if state.get('id') != 'magnetpull' or not source or source.get('kind') != 'Pokemon':
            return None
        holder = pokemon(source)
        if holder[0] == identity[0]:
            return None
        before, after = frame.enter['before'], frame.leave['after']
        if not all(items(cache['sides'][ref['side']]['fields']['active']) == [ref]
                   for cache in (before, after) for ref in (source, victim)):
            return None
        if not all(member(cache, holder)['ability'] == 'magnetpull'
                   and all(member(cache, actor)['hp'] > 0 and member(cache, actor)['fainted'] is False
                           for actor in (holder, identity)) for cache in (before, after)):
            return None
        if member(before, identity)['trapped'] is not False or member(after, identity)['trapped'] != 'hidden':
            return None
        return dict(source=source, victim=victim, epoch=self.phase_epoch)


    def _condition_start(self, frame):
        if frame.leave['result'] is False:return
        arguments=items(frame.enter['arguments'])
        if len(arguments)<4 or arguments[0] not in ('Start','Restart'):return
        effect=self._effect(arguments[1]);target=arguments[3]
        if not effect or type(target) is not dict or target.get('kind')!='Pokemon':return
        if effect['id'] not in ('slp','stall','encore','twoturnmove'):return
        identity=pokemon(target);state=arguments[2]
        need(type(state) is dict and type(state.get('object')) is int,'actual Native condition-state identity required')
        if effect['id']=='slp' and arguments[0]=='Start':
            source=self._effect(arguments[5]) if len(arguments)>5 else None
            if source and source['effect_type']=='Move' and source['id']=='rest':self.rest_states.add(state['object'])
        elif effect['id']=='stall':
            self.stall_refreshed[identity]=dict(epoch=self.phase_epoch,call_id=frame.call,sequence=frame.end)
        elif effect['id']=='twoturnmove' and arguments[0]=='Start':
            source=self._effect(arguments[5]) if len(arguments)>5 else None;dispatch=self.dispatch_slots.get(identity)
            root=self.ancestor(frame,'BattleActions.runMove')
            need(source and source['id']=='solarbeam' and dispatch and dispatch['move_id']==source['id']
                and root and dispatch['run_move_call_id']==root.call and dispatch['slot_object'] is not None,
                'charge start lacks actual same-move Native dispatch-slot provenance')
            self.charge_slots[state['object']]=dict(dispatch,condition_call_id=frame.call)
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
            slot_object,slot,call=candidates[-1]
            self.encore_slots[(*identity,state['object'])]=dict(slot_object=slot_object,physical_slot=slot,get_move_data_call_id=call,sequence=frame.end)

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
                elif frame.operation=='Battle.faintMessages':
                    self._bind_arena_trap_double_faint(frame)
                    self._bind_shadow_tag_replacement_faint(frame)
                    self._bind_fainted_magnet_pull(frame)
                    self._bind_retired_fainted_trap(frame)
                elif frame.operation=='Pokemon.disableMove':self._disable(frame)
                elif frame.operation=='Pokemon.tryTrap':self._trap(frame)
                elif frame.operation==SHADOW_TAG_TRAP_CALLBACK:self._shadow_tag_write(frame)
                elif frame.operation=='Battle.singleEvent':self._condition_start(frame)
                elif frame.operation=='Field.setWeather':self._weather(frame)
                elif frame.operation=='Pokemon.getMoveRequestData':self._locks(frame)
                elif frame.operation=='Pokemon.clearVolatile':
                    identity=pokemon(frame.enter['subject']);self.trap_causes.pop(identity,None);self.lock_results.pop(identity,None)
                    self.double_faint_trap_proofs.pop(identity, None)
                    self.retired_fainted_trap_proofs.pop(identity, None)
                    self._clear_move_restrictions(frame, identity)
                    self.charge_slots={k:v for k,v in self.charge_slots.items() if v['actor']!=identity}
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
            last_active_install=list(self.last_active_install),
            double_faint_trap_proofs=[dict(actor=list(k), causes=v) for k,v in sorted(self.double_faint_trap_proofs.items())],
            retired_fainted_trap_proofs=[dict(actor=list(k), causes=v) for k,v in sorted(self.retired_fainted_trap_proofs.items())],
            last_trap_refresh=self.last_trap_refresh,
            rest_state_objects=sorted(self.rest_states),encore_slots=[dict(actor=list(k[:2]),state_object=k[2],**v) for k,v in sorted(self.encore_slots.items())],
            weather_creation=self.weather_creation,phase_epoch=self.phase_epoch,old_stall=self.old_stall,
            stall_refreshed=[dict(actor=list(k),**v) for k,v in sorted(self.stall_refreshed.items())],
            lock_results=[dict(actor=list(k),**v) for k,v in sorted(self.lock_results.items())],
            charge_records=[dict(state_object=k,**dict(v,actor=list(v['actor']))) for k,v in sorted(self.charge_slots.items())],
            events=list(self.events),source_registry_hash=self.source_registry_hash,
            qualification=False,causal_coverage_complete=False,
            open_gates=['queue/residual chronology','complete disable/trap causes','all73 physical projector','literal owner-prefix binding'])
