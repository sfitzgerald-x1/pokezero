import copy
import json
import unittest

from fresh200_native_actions_test import fixture as action_fixture
from gpu_engine.fresh200_native_phase import (OrdinaryInputs,ReplacementCounter,UnmappedNativePhase,
    phase_from_native,retained_inputs)
from gpu_engine.fresh200_native import step_packet_from_native
from gpu_engine.fresh200_owner_history import OwnerHistory
from gpu_engine.fresh200_boundary_gate import BoundaryAddress,BoundaryGate


def boundary():
    party,request,_=action_fixture();peer=copy.deepcopy(request);peer['side']['id']='p2'
    party.battle.update(turn=4,midTurn=False,requestState='move',queue=[])
    for side in (0,1):
        for mon in party.battle['sides'][side]['pokemon']:mon['activeTurns']=4
    proof=dict(requestOrigin='materialized_activeRequest',receipt=dict(poisoned=False,openFrameCount=0,
        control=dict(phase='snapshot'),boundary=dict(requestOrigin='materialized_activeRequest'),lastSequence=50))
    return dict(battle=party.battle,witness=proof,boundaryRequests={'p1':request,'p2':peer}),party


def suspended():
    snapshot,party=boundary();snapshot['battle'].update(midTurn=True,requestState='switch',
        queue=[dict(choice='move',pokemon='[Pokemon:p1a]',moveid='tackle'),dict(choice='residual')])
    snapshot['boundaryRequests']['p1']['wait']=True
    snapshot['boundaryRequests']['p2'].pop('active');snapshot['boundaryRequests']['p2']['forceSwitch']=[True]
    inputs=OrdinaryInputs((1,3),((0,'tackle'),(0,'batonpass')),(40,41))
    return snapshot,party,inputs


def chance():
    return dict(draws=[[0]*12 for _ in (0,1)],phazing=[0,0],secondary=[0,0],sleep_talk=[0,0],
        entry_ties=[0,0],initial_tie=0,effect=[[0]*5 for _ in (0,1)])


def request(owner='p1',**values):return '|request|'+json.dumps(dict(side=dict(id=owner),**values),separators=(',',':'))


class NativeTransportContractTests(unittest.TestCase):
    def test_ordinary_and_suspended_clocks_use_distinct_native_phase(self):
        snapshot,party=boundary();ordinary=phase_from_native(snapshot,party)
        self.assertEqual((ordinary.turn,ordinary.active_turns),(3,(3,3)))
        snapshot,party,inputs=suspended();pending=phase_from_native(snapshot,party,original_inputs=inputs,old_stall=[2,0])
        self.assertEqual((pending.turn,pending.active_turns),(4,(4,4)))
        self.assertTrue(pending.pending_residual);self.assertEqual(pending.queued_actor,0)
        self.assertEqual(pending.queued_actions,(1,3));self.assertEqual(pending.queued_stall,(2,0))

    def test_remaining_opponent_requires_actual_commit_and_old_stall(self):
        snapshot,party,inputs=suspended()
        with self.assertRaisesRegex(UnmappedNativePhase,'commit join'):
            phase_from_native(snapshot,party,old_stall=[0,0])
        with self.assertRaisesRegex(UnmappedNativePhase,'old-stall'):
            phase_from_native(snapshot,party,original_inputs=inputs)
        snapshot['battle']['queue'].insert(0,dict(choice='move',pokemon='[Pokemon:p2a]',moveid='surf'))
        with self.assertRaisesRegex(UnmappedNativePhase,'ambiguous'):
            phase_from_native(snapshot,party,original_inputs=inputs,old_stall=[0,0])

    def test_replacement_counter_counts_closed_resumes_and_clears_after_residual(self):
        initial,party=boundary();ordinary=phase_from_native(initial,party)
        snapshot,party,inputs=suspended();pending=phase_from_native(snapshot,party,original_inputs=inputs,old_stall=[2,0])
        counter=ReplacementCounter();self.assertEqual(counter.completed('first',before=ordinary,after=pending),0)
        self.assertEqual(counter.completed('resume',before=pending,after=pending),1)
        snapshot['battle']['queue']=[];after=phase_from_native(snapshot,party)
        self.assertFalse(after.pending_residual);self.assertEqual(after.queued_actor,-1)
        self.assertEqual(counter.completed('last',before=pending,after=after),0)
        with self.assertRaisesRegex(UnmappedNativePhase,'unique'):counter.completed('last',before=pending,after=after)

    def test_native_phase_selects_saved_inputs_but_fresh_entry_ties(self):
        snapshot,party,inputs=suspended();phase=phase_from_native(snapshot,party,original_inputs=inputs,old_stall=[2,0])
        old=chance();old['draws'][0][0]=1;fresh=chance();fresh['entry_ties']=[1,0]
        packet=step_packet_from_native(phase,fresh,old,stream=12,request=1)
        self.assertEqual(packet['draws'][0][0],1);self.assertEqual(packet['entry_ties'],[1,0]);self.assertTrue(packet['resuming'])
        packet['draws'][0][0]=0;self.assertEqual(old['draws'][0][0],1)
        fresh['effect'][0][0]=16
        with self.assertRaisesRegex(ValueError,'sealed operand'):step_packet_from_native(phase,fresh,old,stream=12,request=1)
        initial,party=boundary();ordinary=phase_from_native(initial,party)
        self.assertEqual(retained_inputs(ordinary,None)['draws'],[[0]*12,[0]*12])

    def test_phase_refuses_probed_requests_or_unsupported_queued_actions(self):
        snapshot,party=boundary();snapshot['witness']['requestOrigin']='getRequests'
        with self.assertRaisesRegex(UnmappedNativePhase,'passive'):phase_from_native(snapshot,party)
        snapshot,party,inputs=suspended();snapshot['battle']['queue'].append(dict(choice='unknown'))
        with self.assertRaisesRegex(UnmappedNativePhase,'unrepresented material'):
            phase_from_native(snapshot,party,original_inputs=inputs,old_stall=[0,0])

    def test_actionable_wait_actionable_owner_history_never_reuses_old_reference(self):
        history=OwnerHistory('p1');first=history.append(['|turn|1',request(active=[{}])])[0]
        history.bind_reference(first,'one');self.assertEqual(history.endpoint(),'one')
        history.append([request(wait=True)])
        with self.assertRaisesRegex(ValueError,'actionable'):history.endpoint()
        third=history.append([request(active=[{}])])[0]
        self.assertEqual(third.generation,3)
        with self.assertRaisesRegex(ValueError,'stale or missing'):history.endpoint()
        history.bind_reference(third,'three');self.assertEqual(history.endpoint(),'three')

    def test_public_tail_requires_explicit_current_prefix_reencoding(self):
        history=OwnerHistory('p1');cut=history.append([request(active=[{}])])[0];history.bind_reference(cut,'before')
        history.append(['|-damage|p2a: Literal|50/100'])
        with self.assertRaisesRegex(ValueError,'stale'):history.endpoint()
        with self.assertRaisesRegex(ValueError,'current literal'):history.bind_reference(cut,'bad')
        history.bind_reference(history.current_cut(),'after');self.assertEqual(history.endpoint(),'after')

    def test_peer_private_or_duplicate_json_is_rejected_before_ingestion(self):
        history=OwnerHistory('p1')
        for line in (request('p2',active=[{}]),'|request|{"side":{"id":"p1"},"wait":true,"wait":false}'):
            with self.assertRaises(ValueError):history.append([line])
        self.assertEqual(history.lines,[])

    def test_expected_values_are_inaccessible_until_complete_native_freeze(self):
        address=BoundaryAddress(12,8,'PRE');gate=BoundaryGate(address);calls=[]
        def load(actual,receipt):calls.append(actual);return receipt['reference']['sha256']
        with self.assertRaisesRegex(ValueError,'before Native'):gate.read_expected(load)
        self.assertEqual(calls,[])
        def seal(actual,projection):
            self.assertEqual(actual,address);self.assertEqual(projection,{'independent':'Native'})
            return dict(address=actual,field_count=73,source_registry_hash='b'*64,reference=dict(path='native.json',bytes=7,sha256='a'*64))
        receipt=gate.freeze({'independent':'Native'},seal);receipt['reference']['sha256']='c'*64
        self.assertEqual(gate.read_expected(load),'a'*64);self.assertEqual(calls,[address])
        with self.assertRaisesRegex(ValueError,'already frozen'):gate.freeze({},seal)


if __name__=='__main__':unittest.main()
