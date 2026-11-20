import copy
import unittest

import numpy as np

from gpu_engine.fresh200_native_initial import (ConstructorCoordinates, MissingInitialProjection,
    project_initial, EXTERNAL)


def fixture():
    vocabulary=dict(species=['','duplicate'],types=['','Poison','Ground'],
        moves=['','Tackle','Hidden Power Ground'],items=['','Leftovers'],abilities=['','Stench'])
    sides=[];facts=[]
    for side in (0,1):
        pokemon=[]
        for original in range(6):
            pokemon.append(dict(set=dict(species='duplicate'),status='',volatiles={},transformed=False,lastMove=None,
                addedType='',maxhp=200,baseMaxhp=200,hp=200,fainted=False,
                moveSlots=[dict(id='tackle',pp=10,disabled=False),dict(id='hiddenpower',pp=15,disabled=False)],
                gender='M',hpType='Ground',hpPower=70,storedStats=dict(atk=90,def_=80,spa=70,spd=60,spe=50),
                item='leftovers',ability='stench',baseAbility='stench',
                boosts=dict(atk=-1,def_=0,spa=0,spd=0,spe=0,accuracy=0,evasion=0),
                isActive=original==0,isStarted=True,activeTurns=1,trapped=False,
                truantTurn=False,itemKnockedOff=False,types=['Poison'],baseTypes=['Poison']))
            mon=pokemon[-1];mon['storedStats']['def']=mon['storedStats'].pop('def_');mon['boosts']['def']=mon['boosts'].pop('def_')
            facts.append(dict(side=side,original=original,current=original,level=80,happiness=17,pokeball='pokeball'))
        sides.append(dict(id=f'p{side+1}',n=side,team='123456',pokemon=pokemon,active=[f'[Pokemon:p{side+1}a]'],
            sideConditions={},slotConditions=[{}]))
    battle=dict(sides=sides,started=True,ended=False,turn=1,midTurn=False,requestState='move',queue=[],winner='',field=dict(weather=''))
    receipt=dict(poisoned=False,openFrameCount=0,battleId='b',control=dict(phase='snapshot'),lastSequence=7,sourceRegistryHash='a'*64)
    snapshot=dict(battle=battle,boundaryRequests={'p1':{'active':[{}]},'p2':{'active':[{}]}},
        witness=dict(receipt=receipt,elidedProperties=facts))
    rows=[dict(operation='COMMAND_BEGIN',phase=phase,battleId='b') for phase in ('start','snapshot','end','close')]
    rows.extend(dict(operation='CALL_ENTER',phase='start',battleId='b',frame=dict(operation='Battle.setPlayer',callId=i)) for i in (0,1))
    coordinates=ConstructorCoordinates(species=(('duplicate',)*6,('duplicate',)*6),
        move_ids=np.tile(np.array([1,2,0,0],np.int32),(2,6,1)),keys=np.array([0,2**32-1],np.uint32))
    return snapshot,rows,vocabulary,coordinates


class NativeInitialProjectionTests(unittest.TestCase):
    def test_complete_typed_coverage_and_live_scalar_observation(self):
        snapshot,rows,vocab,coordinates=fixture()
        values,certificate=project_initial(snapshot,rows,vocab,coordinates)
        self.assertEqual(len(values),73)
        self.assertEqual(set(certificate['physical_fields'])|set(certificate['external_fields']),set(values))
        self.assertFalse(set(certificate['physical_fields'])&EXTERNAL)
        self.assertEqual(values['level'][0,0,0],80)
        self.assertEqual(values['happiness'][0,0,0],17)  # absent serialized set field is not defaulted to255
        self.assertEqual(values['boosts'][0,0,0,0],-1)
        self.assertEqual(values['types'][0,0,0].tolist(),[1,0])
        self.assertEqual(values['active_turns'].tolist(),[[0,0]])
        self.assertEqual(values['keys'].tolist(),[[0,2**32-1]])
        self.assertEqual(values['keys'].dtype,np.dtype('uint32'))
        self.assertEqual(values['disabled'].dtype,np.dtype('bool'))
        self.assertFalse(certificate['qualification']);self.assertFalse(certificate['general_prefix_applicable'])

    def test_constructor_coordinates_bind_hidden_power_by_slot_without_alias_guess(self):
        snapshot,rows,vocab,coordinates=fixture()
        values,_=project_initial(snapshot,rows,vocab,coordinates)
        self.assertEqual(values['moves'][0,0,0].tolist(),[1,2,0,0])
        wrong=ConstructorCoordinates(coordinates.species,coordinates.move_ids[...,::-1],coordinates.keys)
        with self.assertRaisesRegex(MissingInitialProjection,'cardinality|identity'):
            project_initial(snapshot,rows,vocab,wrong)
        snapshot['battle']['sides'][0]['pokemon'][0]['moveSlots'].reverse()
        with self.assertRaisesRegex(MissingInitialProjection,'positional'):
            project_initial(snapshot,rows,vocab,coordinates)

    def test_native_values_are_independent_of_device_history(self):
        snapshot,rows,vocab,coordinates=fixture()
        before,_=project_initial(snapshot,rows,vocab,coordinates)
        snapshot['battle']['sides'][0]['pokemon'][0]['hp']=123
        snapshot['battle']['sides'][0]['pokemon'][0]['storedStats']['spe']=51
        after,_=project_initial(snapshot,rows,vocab,coordinates)
        self.assertEqual(after['hp'][0,0,0],123);self.assertEqual(after['stats'][0,0,0,5],51)
        self.assertEqual(before['hp'][0,0,0],200)
        self.assertEqual(set(ConstructorCoordinates.__dataclass_fields__),{'species','move_ids','keys'})

    def test_action_history_and_suspension_are_refusals(self):
        original=fixture()
        for operation in ('Side.chooseMove','Pokemon.moveUsed','Pokemon.transformInto','Side.addSlotCondition'):
            snapshot,rows,vocab,coordinates=copy.deepcopy(original)
            rows.append(dict(operation='CALL_ENTER',phase='start',battleId='b',frame=dict(operation=operation)))
            with self.assertRaisesRegex(MissingInitialProjection,'historical'):
                project_initial(snapshot,rows,vocab,coordinates)
        snapshot,rows,vocab,coordinates=fixture();snapshot['battle']['midTurn']=True
        with self.assertRaisesRegex(MissingInitialProjection,'phase'):
            project_initial(snapshot,rows,vocab,coordinates)

    def test_active_conditions_and_ambiguous_traps_are_never_defaulted(self):
        for key,value in (('status','slp'),('volatiles',{'encore':{'move':'tackle'}}),('trapped','hidden'),('transformed',True)):
            snapshot,rows,vocab,coordinates=fixture();snapshot['battle']['sides'][0]['pokemon'][0][key]=value
            with self.assertRaises(MissingInitialProjection):project_initial(snapshot,rows,vocab,coordinates)
        snapshot,rows,vocab,coordinates=fixture();snapshot['battle']['field']=dict(weather='sunnyday',weatherState={})
        with self.assertRaisesRegex(MissingInitialProjection,'weather.*causal'):
            project_initial(snapshot,rows,vocab,coordinates)

    def test_missing_scalar_or_invalid_numeric_native_data_fails(self):
        snapshot,rows,vocab,coordinates=fixture();del snapshot['witness']['elidedProperties'][0]['happiness']
        with self.assertRaisesRegex(MissingInitialProjection,'happiness'):
            project_initial(snapshot,rows,vocab,coordinates)
        snapshot,rows,vocab,coordinates=fixture();snapshot['battle']['sides'][0]['pokemon'][0]['hp']=0
        with self.assertRaisesRegex(MissingInitialProjection,'faint/HP'):
            project_initial(snapshot,rows,vocab,coordinates)
        snapshot,rows,vocab,coordinates=fixture();snapshot['battle']['sides'][0]['pokemon'][0]['hpPower']=2**40
        with self.assertRaisesRegex(MissingInitialProjection,'overflow'):
            project_initial(snapshot,rows,vocab,coordinates)

    def test_other_battle_or_failed_native_call_cannot_supply_initial_history(self):
        snapshot,rows,vocab,coordinates=fixture();rows[0]['battleId']='other'
        with self.assertRaisesRegex(MissingInitialProjection,'identity'):
            project_initial(snapshot,rows,vocab,coordinates)
        snapshot,rows,vocab,coordinates=fixture();rows.append(dict(operation='CALL_THROW',phase='start',battleId='b'))
        with self.assertRaisesRegex(MissingInitialProjection,'exception'):
            project_initial(snapshot,rows,vocab,coordinates)

    def test_optional_winner_uses_actual_bound_live_absence_or_undefined(self):
        for tag in ('missing','undefined'):
            snapshot,rows,vocab,coordinates=fixture();del snapshot['battle']['winner']
            native=dict(state=dict(fields=dict(winner={'$native':tag},ended=False)))
            snapshot['witness']['nativeObservation']=native
            snapshot['witness']['receipt']['boundary']=dict(nativeObservation=copy.deepcopy(native))
            values,certificate=project_initial(snapshot,rows,vocab,coordinates)
            self.assertEqual(values['winner'].tolist(),[-1])
            self.assertEqual(certificate['winner_observation']['value'],{'$native':tag})

    def test_omitted_winner_without_live_observation_is_fatal(self):
        snapshot,rows,vocab,coordinates=fixture();del snapshot['battle']['winner']
        with self.assertRaisesRegex(MissingInitialProjection,'nativeObservation'):
            project_initial(snapshot,rows,vocab,coordinates)

    def test_live_winner_must_match_receipt_and_nonterminal_state(self):
        for changed in ('binding','value','ended'):
            snapshot,rows,vocab,coordinates=fixture();del snapshot['battle']['winner']
            native=dict(state=dict(fields=dict(winner={'$native':'undefined'},ended=False)))
            snapshot['witness']['nativeObservation']=native
            snapshot['witness']['receipt']['boundary']=dict(nativeObservation=copy.deepcopy(native))
            if changed=='binding':native['state']['fields']['winner']={'$native':'missing'}
            elif changed=='value':
                native['state']['fields']['winner']='someone'
                snapshot['witness']['receipt']['boundary']['nativeObservation']=copy.deepcopy(native)
            else:
                native['state']['fields']['ended']=True
                snapshot['witness']['receipt']['boundary']['nativeObservation']=copy.deepcopy(native)
            with self.assertRaises(MissingInitialProjection):project_initial(snapshot,rows,vocab,coordinates)


if __name__=='__main__':
    unittest.main()
