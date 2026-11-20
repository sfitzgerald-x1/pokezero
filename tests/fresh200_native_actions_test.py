import copy
import unittest

from gpu_engine.fresh200_native_identity import PartyMap
from gpu_engine.fresh200_native_actions import LockResults, native_action_relation, owner_request_mask


def fixture(ids=('tackle', 'protect', 'surf', 'toxic'), permutation='123456'):
    sides=[]
    for side in (0, 1):
        pokemon=[dict(set=dict(species='duplicate'), fainted=False, trapped=False,
            moveSlots=[dict(id=move,pp=10,disabled=False) for move in ids]) for _ in range(6)]
        sides.append(dict(id=f'p{side+1}',n=side,team=permutation,pokemon=pokemon,active=[f'[Pokemon:p{side+1}a]']))
    battle=dict(sides=sides,ended=False)
    party=PartyMap(battle,[['duplicate']*6,['duplicate']*6])
    request=dict(side=dict(id='p1',pokemon=[dict(active=i==0,condition='100/100') for i in range(6)]),
        active=[dict(moves=[dict(id=move,pp=10,disabled=False) for move in ids])])
    return party,request,party.mon(0,party.active(0))


class NativeActionContractTests(unittest.TestCase):
    def test_owner_mask_keeps_pp_zero_and_positional_invalid_row(self):
        _,request,_=fixture()
        request['active'][0]['moves']=[dict(id='tackle',pp=0),None,dict(id='surf',disabled=True)]
        self.assertEqual(owner_request_mask(request,owner=0),[True,False,False,False]+[True]*5)
        request['active'][0]['maybeTrapped']=True
        self.assertEqual(owner_request_mask(request,owner=0),[True,False,False,False]+[False]*5)

    def test_original_switch_targets_survive_duplicate_species_and_permutation(self):
        party,request,mon=fixture(permutation='413256')
        self.assertEqual(party.active(0),1)
        result=native_action_relation(request,party,owner=0,locks=LockResults(None,None))
        self.assertEqual(result['physical_commands'][4:],['switch 4','switch 3','switch 2','switch 5','switch 6'])
        self.assertEqual(result['physical_mask'],[True]*9)
        mon['trapped']='hidden'
        result=native_action_relation(request,party,owner=0,locks=LockResults(None,None))
        self.assertEqual(result['physical_mask'][4:],[False]*5)
        self.assertEqual(result['owner_request_mask'][4:],[True]*5)

    def test_native_duplicate_id_acceptance_does_not_use_selected_slot_pp(self):
        party,request,mon=fixture(ids=('tackle','tackle','protect','toxic'))
        mon['moveSlots'][0].update(pp=0,disabled=True)
        request['active'][0]['moves'][0]['disabled']=True
        result=native_action_relation(request,party,owner=0,locks=LockResults(None,None))
        self.assertTrue(result['physical_mask'][0])
        self.assertFalse(result['owner_request_mask'][0])
        self.assertEqual(result['physical_commands'][0],'move 1')
        self.assertTrue(result['numeric_relations'][-4]['execution_slot_requires_causal_witness'])
        self.assertFalse(result['dispatch_history_verified'])

    def test_forced_replacement_overrides_traps_without_requiring_move_lock_query(self):
        party,request,mon=fixture();mon['trapped']=True
        request['forceSwitch']=[True];request.pop('active')
        party.mon(0,3)['fainted']=True;request['side']['pokemon'][3]['condition']='0 fnt'
        result=native_action_relation(request,party,owner=0,locks=LockResults())
        self.assertEqual(result['physical_mask'],[False]*4+[True,True,False,True,True])
        self.assertEqual(result['owner_request_mask'],result['physical_mask'])

    def test_hard_solarbeam_and_recharge_precede_all_ordinary_filters(self):
        party,request,mon=fixture(ids=('tackle','solarbeam','solarbeam','protect'));mon['trapped']=True
        request['active'][0]['trapped']=True
        for move in mon['moveSlots']:move.update(pp=0,disabled=True)
        request['active'][0]['moves']=[dict(id='solarbeam')]
        result=native_action_relation(request,party,owner=0,locks=LockResults('solarbeam'))
        self.assertEqual(result['physical_mask'],[False,True,True,False]+[False]*5)
        self.assertEqual(result['physical_commands'][1:3],['move 1','move 1'])
        request['active'][0]['moves']=[dict(id='recharge')]
        result=native_action_relation(request,party,owner=0,locks=LockResults('recharge'))
        self.assertEqual(result['physical_mask'],[True]+[False]*8)
        self.assertEqual(result['owner_request_mask'][:4],[True,False,False,False])

    def test_semi_lock_requires_causal_slot_and_retains_all_numeric_aliases(self):
        party,request,mon=fixture(ids=('tackle','tackle','surf','toxic'))
        for move in mon['moveSlots']:move.update(pp=0,disabled=True)
        request['active'][0]['moves']=[dict(id='tackle')]
        with self.assertRaisesRegex(ValueError,'exact causal'):
            native_action_relation(request,party,owner=0,locks=LockResults(None,'tackle'))
        result=native_action_relation(request,party,owner=0,locks=LockResults(None,'tackle',1))
        self.assertEqual(result['physical_commands'][1],'move 1')
        request['active'][0]['moves']=[dict(id=move) for move in ('tackle','tackle','surf','toxic')]
        result=native_action_relation(request,party,owner=0,locks=LockResults(None,'tackle',1))
        self.assertEqual(result['physical_commands'][1],'move 2')
        aliases=[x for x in result['numeric_relations'] if x['kind']=='semi-lock']
        self.assertEqual([x['physical_lanes'] for x in aliases],[[1]]*4)

    def test_hidden_restriction_struggle_does_not_repair_advertised_mask(self):
        party,request,mon=fixture()
        for move in mon['moveSlots']:move['disabled']='hidden'
        request['active'][0]['maybeTrapped']=True
        result=native_action_relation(request,party,owner=0,locks=LockResults(None,None))
        self.assertEqual(result['owner_request_mask'],[True]*4+[False]*5)
        self.assertEqual(result['physical_mask'],[True,False,False,False]+[True]*5)
        aliases=[x for x in result['numeric_relations'] if x['kind']=='struggle']
        self.assertEqual([x['physical_lanes'] for x in aliases],[[0]]*4)
        self.assertFalse(result['qualification'])

    def test_unknown_locks_and_inconsistent_requests_are_fatal(self):
        party,request,mon=fixture()
        with self.assertRaisesRegex(ValueError,'missing actual Native lock'):
            native_action_relation(request,party,owner=0,locks=LockResults(None))
        bad=copy.deepcopy(request);bad['active'][0]['moves'].reverse()
        with self.assertRaisesRegex(ValueError,'physical slots differ'):
            native_action_relation(bad,party,owner=0,locks=LockResults(None,None))
        mon['moveSlots'][0]['disabled']=None
        with self.assertRaisesRegex(ValueError,'PP/Disable'):
            native_action_relation(request,party,owner=0,locks=LockResults(None,None))
        with self.assertRaisesRegex(ValueError,'another seat'):
            owner_request_mask(request,owner=1)

    def test_wait_absent_and_terminal_never_create_actions(self):
        party,request,_=fixture();request['wait']=True
        for value in (None,request):
            result=native_action_relation(value,party,owner=0,locks=LockResults())
            self.assertEqual(result['physical_mask'],[False]*9)
            self.assertEqual(result['numeric_relations'],[])
        request.pop('wait');party.battle['ended']=True
        self.assertEqual(native_action_relation(request,party,owner=0,locks=LockResults())['physical_mask'],[False]*9)


if __name__=='__main__':
    unittest.main()
