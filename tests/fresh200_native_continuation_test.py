import copy
import unittest

from fresh200_native_ledger_test import Trace, arr, effect, ledger, mon, obj, cache
from fresh200_native_conditions_test import condition, living_cache, set_value
from gpu_engine.fresh200_native_ledger import NativeHistory, UnmappedNativeHistory


def charge():
    trace=Trace();before=living_cache()
    for slot in before['sides'][0]['pokemon'][0]['fields']['moveSlots']['items']:set_value(slot,'id','solarbeam')
    trace.enter('BattleActions.runMove',arguments=(effect('solarbeam'),mon(0,0)),before=before)
    trace.enter('Pokemon.deductPP',mon(0,0),(effect('solarbeam'),),before)
    trace.enter('Pokemon.getMoveData',mon(0,0),(effect('solarbeam'),))
    trace.leave(before['sides'][0]['pokemon'][0]['fields']['moveSlots']['items'][1]);trace.leave(1,after=before)
    trace.enter('Pokemon.moveUsed',mon(0,0),(effect('solarbeam'),),before);trace.leave(after=before)
    trace.enter('Battle.singleEvent',arguments=('Start',condition('twoturnmove'),obj(21000),mon(0,0),mon(1,0),effect('solarbeam')));trace.leave(True)
    trace.leave();return trace,before


def release(first,before,mutation=None):
    trace=Trace();trace.call=first.call;before=copy.deepcopy(before)
    before['sides'][0]['pokemon'][0]['fields']['volatiles']=obj(21001,twoturnmove=obj(21000,move='solarbeam',duration=1))
    trace.enter('BattleActions.runMove',arguments=(effect('solarbeam'),mon(0,0)),before=before)
    trace.enter('Pokemon.getLockedMove',mon(0,0));trace.leave('solarbeam')
    actual=copy.deepcopy(before)
    if mutation=='stale':actual['sides'][0]['pokemon'][0]['fields']['moveSlots']['items'][1]['object']+=100
    elif mutation=='ambiguous':actual['sides'][0]['pokemon'][0]['fields']['moveSlots']['items'][0]['object']=actual['sides'][0]['pokemon'][0]['fields']['moveSlots']['items'][1]['object']
    trace.enter('Pokemon.moveUsed',mon(0,0),(effect('solarbeam'),),actual);trace.leave(after=actual);trace.leave()
    return trace


class NativeContinuationTests(unittest.TestCase):
    def test_locked_release_reuses_actual_charge_slot_without_new_pp_getter(self):
        first,before=charge();value=ledger().consume(first.rows)
        value.consume([dict(operation='COMMAND_END',battleId='b')]);later=release(first,before);value.consume(later.rows)
        result=value.snapshot();self.assertEqual(result['last_action_index'],[1,-1])
        self.assertTrue(result['events'][-1]['retained_charge_witness'])
        self.assertEqual(result['events'][-1]['get_move_data_call_id'],result['events'][0]['get_move_data_call_id'])
        self.assertFalse(any(r.get('frame',{}).get('operation') in ('Pokemon.deductPP','Pokemon.getMoveData') for r in later.rows))

    def test_stale_or_aliased_charge_slot_never_uses_first_matching_id(self):
        for mutation in ('stale','ambiguous'):
            first,before=charge();value=ledger().consume(first.rows);value.consume([dict(operation='COMMAND_END',battleId='b')])
            with self.assertRaisesRegex(UnmappedNativeHistory,'stale or copied'):value.consume(release(first,before,mutation).rows)

    def test_lethal_damage_faint_can_begin_with_nonpositive_hp(self):
        for hp in (0,-7):
            trace=Trace();before=living_cache();before['sides'][0]['pokemon'][0]['fields'].update(hp=hp,fainted=False,faintQueued=False,status='tox')
            after=copy.deepcopy(before);after['sides'][0]['pokemon'][0]['fields'].update(hp=0,faintQueued=True)
            trace.enter('Pokemon.faint',mon(0,0),(),before);trace.leave(hp,after=after)
            result=ledger().consume(trace.rows).snapshot()['retained_status'];self.assertEqual(result[0]['status'],'tox')

    def test_repeated_faint_attempt_cannot_replace_original_major_status(self):
        trace=Trace();before=living_cache();before['sides'][0]['pokemon'][0]['fields'].update(hp=0,fainted=False,faintQueued=True,status='')
        trace.enter('Pokemon.faint',mon(0,0),(),before);trace.leave(0,after=before)
        self.assertEqual(ledger().consume(trace.rows).snapshot()['retained_status'],[])

    def test_continuing_constructor_binding_needs_no_end_or_close(self):
        rows=[dict(operation='COMMAND_BEGIN',battleId='b',phase='start')]
        rows.extend(dict(operation='CALL_ENTER',battleId='b',phase='start',frame=dict(operation='Battle.setPlayer'),evidence=dict(side=side)) for side in ('p1','p2'))
        rows.append(dict(operation='COMMAND_END',battleId='b',phase='start'))
        value=NativeHistory(battle_id='b',source_registry_hash='a'*64);value.bind_constructor(rows)
        self.assertEqual(value.snapshot()['last_action_kind'],[0,0])
        rows[1]['evidence']['side']='p2'
        with self.assertRaisesRegex(UnmappedNativeHistory,'ordered'):NativeHistory(battle_id='b',source_registry_hash='a'*64).bind_constructor(rows)


if __name__=='__main__':unittest.main()
