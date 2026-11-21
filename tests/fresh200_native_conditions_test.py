import copy
import unittest

from fresh200_native_ledger_test import Trace, arr, cache, effect, ledger, mon, obj


def condition(name):return dict(kind='Effect',object=19000,id=name,effectType='Status')
def ability(name):return dict(kind='Effect',object=19001,id=name,effectType='Ability')
def set_value(node,key,value):
    for pair in node['fields']:
        if pair[0]==key:pair[1]=value;return
    node['fields'].append([key,value])
def context(trace,value):trace.rows[-1]['evidence']['context']=dict(effect=value)
def living_cache():
    value=cache()
    for side in value['sides']:
        for member in side['pokemon']:
            member['fields'].update(hp=317,status='',statusState=obj(19100),volatiles=obj(19200),trapped=False)
    return value


class NativeConditionHistoryTests(unittest.TestCase):
    def test_overlapping_disables_keep_every_cause_and_refresh_epoch(self):
        trace=Trace();before=living_cache()
        for cause in ('choicelock','taunt'):
            after=copy.deepcopy(before)
            for slot in after['sides'][0]['pokemon'][0]['fields']['moveSlots']['items']:set_value(slot,'disabled',True)
            trace.enter('Pokemon.disableMove',mon(0,0),('tackle',),before);context(trace,condition(cause));trace.leave(after=after)
            before=after
        value=ledger().consume(trace.rows);result=value.snapshot()['disable_causes']
        self.assertEqual(len(result),2)  # two duplicate physical slots, never a first-ID guess
        self.assertEqual([x['effect']['id'] for x in result[1]['causes']],['choicelock','taunt'])
        refresh=Trace();refresh.call=trace.call;refresh.enter('Battle.endTurn',before=living_cache());refresh.leave()
        value.consume([dict(operation='COMMAND_END',battleId='b')]);value.consume(refresh.rows)
        self.assertEqual(value.snapshot()['disable_causes'],[])

    def test_only_actual_successful_trap_attempts_retain_source(self):
        trace=Trace();before=living_cache();after=copy.deepcopy(before);after['sides'][0]['pokemon'][0]['fields']['trapped']='hidden'
        trace.enter('Pokemon.tryTrap',mon(0,0),(True,),before);context(trace,ability('shadowtag'));trace.leave(False,after=before)
        trace.enter('Pokemon.tryTrap',mon(0,0),(True,),before);context(trace,ability('shadowtag'));trace.leave(True,after=after)
        causes=ledger().consume(trace.rows).snapshot()['trap_causes']
        self.assertEqual(len(causes[0]['causes']),1);self.assertEqual(causes[0]['causes'][0]['effect']['id'],'shadowtag')

    def test_pre_faint_major_status_is_literal_even_when_initially_healthy(self):
        for status in ('','tox'):
            trace=Trace();before=living_cache();before['sides'][0]['pokemon'][0]['fields'].update(status=status,statusState=obj(19300,id=status,stage=4))
            before['sides'][0]['pokemon'][0]['fields'].update(fainted=False,faintQueued=False)
            after=copy.deepcopy(before);after['sides'][0]['pokemon'][0]['fields']['hp']=0
            after['sides'][0]['pokemon'][0]['fields']['faintQueued']=True
            trace.enter('Pokemon.faint',mon(0,0),(),before);trace.leave(317,after=after)
            retained=ledger().consume(trace.rows).snapshot()['retained_status'][0]
            self.assertEqual(retained['status'],status);self.assertEqual(retained['state'],before['sides'][0]['pokemon'][0]['fields']['statusState'])

    def test_rest_stint_encore_slot_and_stall_use_observed_condition_identity(self):
        trace=Trace()
        trace.enter('Battle.singleEvent',arguments=('Start',condition('slp'),obj(19400),mon(0,0),mon(0,0),effect('rest')));trace.leave(True)
        trace.enter('Battle.singleEvent',arguments=('Start',condition('encore'),obj(19401),mon(0,0)))
        before=living_cache();trace.enter('Callback:encore',before=before)
        trace.enter('Pokemon.getMoveData',mon(0,0),('tackle',));trace.leave(before['sides'][0]['pokemon'][0]['fields']['moveSlots']['items'][1])
        trace.leave(True,after=before);trace.leave(True)
        trace.enter('Battle.singleEvent',arguments=('Restart',condition('stall'),obj(19402),mon(0,0)));trace.leave(True)
        result=ledger().consume(trace.rows).snapshot()
        self.assertEqual(result['rest_state_objects'],[19400]);self.assertEqual(result['encore_slots'][0]['physical_slot'],1)
        self.assertEqual(result['encore_slots'][0]['state_object'],19401);self.assertEqual(result['stall_refreshed'][0]['actor'],[0,0])

    def test_skipped_semilock_query_is_unknown_not_a_null_result(self):
        for queried in (True,False):
            trace=Trace();trace.enter('Pokemon.getMoveRequestData',mon(0,0),before=living_cache())
            trace.enter('Pokemon.getLockedMove',mon(0,0));trace.leave(None)
            trace.enter('Pokemon.getSemiLockedMove',mon(0,0),(True,))
            if queried:trace.enter('Battle.priorityEvent',arguments=('SemiLockMove',mon(0,0)));trace.leave(True)
            trace.leave(None);trace.leave(obj(19500),after=living_cache())
            result=ledger().consume(trace.rows).snapshot()['lock_results'][0]
            self.assertIsNone(result['hard'])
            self.assertEqual(result['semi'],None if queried else 'UNOBSERVED')

    def test_permanent_weather_provenance_uses_actual_native_source_and_effect(self):
        trace=Trace();before=living_cache();after=copy.deepcopy(before)
        after['fields']=dict(field=dict(weather='sunnyday',weatherState=obj(19600,source=mon(1,4))))
        trace.enter('Field.setWeather',dict(kind='NativeService',object=50),('sunnyday',mon(1,4)),before)
        context(trace,ability('drought'));trace.leave(True,after=after)
        result=ledger().consume(trace.rows).snapshot()['weather_creation']
        self.assertEqual(result['source'],[1,4]);self.assertEqual(result['effect']['id'],'drought');self.assertEqual(result['state_object'],19600)

    def test_old_stall_is_frozen_from_actual_actors_after_switches(self):
        trace=Trace();before=living_cache();before['fields']=dict(requestState='move')
        trace.enter('Battle.commitChoices',before=before);trace.leave()
        execution=living_cache();execution['sides'][0]['fields']['active']=arr(19700,mon(0,4))
        execution['sides'][0]['pokemon'][0]['fields']['volatiles']=obj(19701,stall=obj(19702,counter=3))
        execution['sides'][1]['pokemon'][0]['fields']['volatiles']=obj(19703,stall=obj(19704,counter=2))
        trace.enter('Battle.runAction',arguments=(obj(19705,choice='move',pokemon=mon(1,0)),),before=execution);trace.leave()
        result=ledger().consume(trace.rows).snapshot()
        self.assertEqual(result['old_stall'],[0,2]);self.assertEqual(result['phase_epoch'],1)

    def test_clear_volatile_invalidates_old_restriction_evidence(self):
        trace=Trace();before=living_cache();after=copy.deepcopy(before)
        for slot in after['sides'][0]['pokemon'][0]['fields']['moveSlots']['items']:set_value(slot,'disabled',True)
        trace.enter('Pokemon.disableMove',mon(0,0),('tackle',),before);context(trace,condition('taunt'));trace.leave(after=after)
        trace.enter('Pokemon.clearVolatile',mon(0,0),(),after);trace.leave(after=before)
        self.assertEqual(ledger().consume(trace.rows).snapshot()['disable_causes'],[])


if __name__=='__main__':unittest.main()
