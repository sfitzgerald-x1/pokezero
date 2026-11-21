import unittest

from gpu_engine.fresh200_native_ledger import NativeHistory, UnmappedNativeHistory


def obj(token,**values):return dict(object=token,fields=[[k,v] for k,v in values.items()])
def arr(token,*values):return dict(object=token,items=list(values))
def mon(side,original):return dict(kind='Pokemon',object=100+side*6+original,side=side,original=original)
def effect(name):return dict(kind='Effect',object=900,id=name,effectType='Move')


def cache(move='tackle',active=0):
    sides=[]
    for side in (0,1):
        members=[]
        for original in range(6):
            slots=[obj(1000+side*100+original*4+i,id=move,pp=10,disabled=False) for i in range(2)]
            members.append(dict(identity=mon(side,original),fields=dict(moveSlots=arr(2000+side*100+original,*slots),maxhp=317,
                species=dict(kind='Effect',object=3000+original,id='species'+str(original),effectType='Species'),
                baseStoredStats=obj(4000+original,atk=90),baseMoveSlots=arr(5000+original,*slots),baseHpType='Ground',baseHpPower=70)))
        sides.append(dict(pokemon=members,fields=dict(active=arr(6000+side,mon(side,active)),
            choice=obj(6100+side,actions=arr(6200+side)),slotConditions=arr(6300+side,obj(6400+side)))))
    return dict(sides=sides)


class Trace:
    def __init__(self):self.rows=[];self.stack=[];self.call=0
    def row(self,operation,evidence,frame):
        self.rows.append(dict(operation=operation,evidence=evidence,frame=frame,sequence=len(self.rows),battleId='b'))
    def enter(self,operation,subject=None,arguments=(),before=None):
        frame=dict(callId=self.call,parentCall=self.stack[-1]['callId'] if self.stack else None,operation=operation);self.call+=1
        evidence=dict(subject=subject,arguments=arr(7000+self.call,*arguments))
        if before is not None:evidence['before']=before
        self.row('CALL_ENTER',evidence,frame);self.stack.append(frame)
    def leave(self,result=None,after=None):
        evidence=dict(result=result)
        if after is not None:evidence['after']=after
        self.row('CALL_RETURN',evidence,self.stack.pop())
    def dispatch(self,*,move='tackle',slot=1,getter=True,transform=None,transform_success=False):
        before=cache(move);self.enter('BattleActions.runMove',arguments=(effect(move),mon(0,0)),before=before)
        if getter and move!='struggle':
            self.enter('Pokemon.deductPP',mon(0,0),(effect(move),),before)
            self.enter('Pokemon.getMoveData',mon(0,0),(effect(move),))
            self.leave(before['sides'][0]['pokemon'][0]['fields']['moveSlots']['items'][slot]);self.leave(1,after=before)
        self.enter('Pokemon.moveUsed',mon(0,0),(effect(move),),before);self.leave(after=before)
        if transform is not None:
            after=cache(move)
            if transform_success:after['sides'][0]['pokemon'][0]['fields']['species']=before['sides'][1]['pokemon'][transform]['fields']['species']
            self.enter('Pokemon.transformInto',mon(0,0),(mon(1,transform),),before)
            self.leave(transform_success,after=after)
        self.leave()


def ledger():
    value=NativeHistory(battle_id='b',source_registry_hash='a'*64)
    value.bind_initial(dict(field_count=73,initial_history_absence_proved=True,source_registry_hash='a'*64))
    return value


class NativeLedgerTests(unittest.TestCase):
    def test_dispatch_uses_actual_slot_object_even_with_duplicate_move_ids(self):
        trace=Trace();trace.dispatch(slot=1);value=ledger().consume(trace.rows).snapshot()
        self.assertEqual(value['last_action_kind'],[2,0]);self.assertEqual(value['last_action_index'],[1,-1])
        self.assertEqual(value['events'][0]['physical_slot'],1)
        self.assertIsInstance(value['events'][0]['get_move_data_call_id'],int)

    def test_missing_dispatch_slot_is_fatal_without_first_id_fallback(self):
        trace=Trace();trace.dispatch(getter=False)
        with self.assertRaisesRegex(UnmappedNativeHistory,'missing actual Native dispatch-slot witness; no ID/ordinal fallback'):
            ledger().consume(trace.rows)

    def test_failed_before_move_preserves_previous_side_history(self):
        trace=Trace();trace.dispatch(slot=1);value=ledger().consume(trace.rows)
        later=Trace();later.call=trace.call
        later.enter('BattleActions.runMove',arguments=(effect('tackle'),mon(0,0)),before=cache());later.leave()
        # Command boundaries release frame caches while retaining literal history.
        value.consume([dict(operation='COMMAND_END',battleId='b')]);value.consume(later.rows)
        self.assertEqual(value.snapshot()['last_action_index'],[1,-1])
        self.assertEqual(len(value.snapshot()['events']),1)

    def test_struggle_and_called_transform_do_not_invent_dispatch_slot(self):
        trace=Trace();trace.dispatch(move='struggle',getter=False);value=ledger().consume(trace.rows).snapshot()
        self.assertEqual(value['last_action_kind'],[3,0]);self.assertEqual(value['last_action_index'],[-1,-1])
        trace=Trace();trace.dispatch(move='sleeptalk',slot=1,transform=5,transform_success=False)
        value=ledger().consume(trace.rows).snapshot()
        self.assertEqual(value['last_action_kind'],[2,0]);self.assertEqual(value['last_action_index'],[-1,-1])
        self.assertEqual(value['events'][0]['move_id'],'sleeptalk');self.assertFalse(value['events'][1]['succeeded'])

    def test_installed_chosen_switch_is_distinct_from_initial_entry(self):
        trace=Trace();before=cache();target=mon(0,4)
        action=obj(8000,choice='switch',pokemon=mon(0,0),target=target)
        before['sides'][0]['fields']['choice']=obj(8001,actions=arr(8002,action))
        trace.enter('Side.commitChoices',dict(kind='Side',object=10,n=0),before=before);trace.leave(after=before)
        trace.enter('Battle.runAction',arguments=(action,),before=before)
        trace.enter('BattleActions.switchIn',arguments=(target,0),before=before);trace.leave(1,after=cache(active=4));trace.leave()
        value=ledger().consume(trace.rows).snapshot()
        self.assertEqual(value['last_action_kind'],[1,0]);self.assertEqual(value['last_action_index'],[4,-1])
        initial=Trace();initial.enter('Battle.runAction',arguments=(obj(8100,choice='start'),),before=cache())
        initial.enter('BattleActions.switchIn',arguments=(target,0),before=cache());initial.leave(1,after=cache(active=4));initial.leave()
        self.assertEqual(ledger().consume(initial.rows).snapshot()['last_action_kind'],[0,0])

    def root2_post5_trace(self, *, result=True, installed=5, is_drag=True):
        # Frozen selected[2], command 2120000002/0/12. Regirock (p2 slot 1)
        # used Earthquake (slot 0) before Whirlwind dragged in Clefable (slot 5).
        trace = Trace()
        before = cache(move='earthquake')
        active = before['sides'][1]['fields']['active']
        active['items'] = [mon(1, 1)]
        before['sides'][0]['pokemon'][0]['fields']['volatiles'] = obj(8100)
        before['sides'][1]['pokemon'][1]['fields']['volatiles'] = obj(8101)
        actor, target = mon(1, 1), mon(1, 5)
        move = effect('earthquake')
        trace.enter('BattleActions.runMove', arguments=(move, actor), before=before)
        trace.enter('Pokemon.getMoveData', actor, (move,), before)
        trace.leave(before['sides'][1]['pokemon'][1]['fields']['moveSlots']['items'][0])
        trace.enter('Pokemon.moveUsed', actor, (move,), before)
        trace.leave(after=before)
        trace.leave()
        trace.enter('Battle.runAction', arguments=(obj(8000, choice='move'),), before=before)
        trace.enter('BattleActions.dragIn', arguments=(dict(kind='Side', object=10, n=1), 0), before=before)
        trace.enter('BattleActions.switchIn', arguments=(target, 0, None, is_drag), before=before)
        after = cache()
        after['sides'][1]['fields']['active']['items'] = [mon(1, installed)]
        trace.leave(result, after=after)
        trace.leave(result, after=after)
        trace.leave()
        return trace

    def test_root2_post5_whirlwind_replaces_outgoing_move_slot_with_drag_target(self):
        value = ledger().consume(self.root2_post5_trace().rows).snapshot()
        self.assertEqual(value['last_action_kind'], [0, 1])
        self.assertEqual(value['last_action_index'], [-1, 5])
        self.assertEqual([event['kind'] for event in value['events']], ['MOVE_USED', 'INSTALLED_DRAG'])
        self.assertEqual(value['events'][0]['move_id'], 'earthquake')
        self.assertEqual(value['events'][1]['original_target'], 5)

    def test_unsuccessful_drag_preserves_move_slot(self):
        value = ledger().consume(self.root2_post5_trace(result=False, installed=1).rows).snapshot()
        self.assertEqual(value['last_action_kind'], [0, 2])
        self.assertEqual(value['last_action_index'], [-1, 0])
        self.assertEqual([event['kind'] for event in value['events']], ['MOVE_USED'])

    def test_drag_requires_literal_flag_and_matching_installed_slot(self):
        for opts, message in (({'is_drag': False}, 'drag flag'), ({'installed': 4}, 'literal target')):
            with self.subTest(opts=opts), self.assertRaisesRegex(UnmappedNativeHistory, message):
                ledger().consume(self.root2_post5_trace(**opts).rows)

    def test_wish_retains_actual_source_half_hp_without_current_actor_guess(self):
        trace=Trace();before=cache();after=cache()
        after['sides'][0]['fields']['slotConditions']=arr(8200,obj(8201,wish=obj(8202,source=mon(0,3),duration=2)))
        trace.enter('Side.addSlotCondition',dict(kind='Side',object=10,n=0),(0,'wish'),before);trace.leave(True,after=after)
        value=ledger().consume(trace.rows).snapshot()
        self.assertEqual(value['wish_amount'],[158,0]);self.assertEqual(value['events'][0]['source'],[0,3])

    def test_literal_wish_name_joins_only_to_successful_creation(self):
        side = dict(kind='Side', object=10, n=1)
        before, after = cache(), cache()
        after['sides'][1]['fields']['slotConditions'] = arr(
            8200, obj(8201, wish=obj(8202, source=mon(1, 2), duration=2)))
        for success in (False, True):
            with self.subTest(success=success):
                trace = Trace()
                trace.enter('Side.addSlotCondition', side, (0, 'Wish'), before)
                trace.leave(success, after=after if success else before)
                snapshot = ledger().consume(trace.rows).snapshot()
                self.assertEqual(snapshot['wish_amount'], [0, 158 if success else 0])
                wishes = [item for item in snapshot['events'] if item['kind'] == 'CREATED_WISH']
                self.assertEqual([item['source'] for item in wishes], [[1, 2]] if success else [])

    def test_transform_saves_actual_donor_and_native_restore_fields(self):
        trace=Trace();trace.dispatch(move='transform',transform=5,transform_success=True)
        value=ledger().consume(trace.rows).snapshot();saved=value['transform_records'][0]
        self.assertEqual(saved['actor'],[0,0]);self.assertEqual(saved['donor'],[1,5])
        self.assertEqual(set(saved['saved']),{'baseStoredStats','baseMoveSlots','baseHpType','baseHpPower'})
        self.assertEqual(value['last_action_index'],[-1,-1]);self.assertFalse(value['qualification'])

    def test_unbound_or_foreign_or_failed_native_evidence_is_rejected(self):
        value=NativeHistory(battle_id='b',source_registry_hash='a'*64)
        with self.assertRaises(UnmappedNativeHistory):value.snapshot()
        with self.assertRaisesRegex(UnmappedNativeHistory,'binding'):
            ledger().consume([dict(operation='COMMAND_END',battleId='other')])
        with self.assertRaisesRegex(UnmappedNativeHistory,'exception'):
            ledger().consume([dict(operation='CALL_THROW',battleId='b')])


if __name__=='__main__':unittest.main()
