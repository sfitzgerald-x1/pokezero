"""Authored wire/controller controls; no model, simulator, native or FoulPlay."""
import asyncio
import json
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import patch

import test_live_protocol_bridge as fixtures
from pokezero import foulplay_bridge as bridge
from pokezero.live_protocol_bridge import OwningWireSession


class OwningChoiceAckTests(unittest.TestCase):
    def fixture(self,**kwargs):
        f=fixtures.OwningBridgeTests(methodName='test_both_seats_actual_owner_only_exact_observation_choice_and_terminal')
        f.setUp();self.addCleanup(f.doCleanups)
        config=f.config(**kwargs);return f,config
    def acks(self,battle,*,submission=0):
        return [dict(type='choice_ack',battleId=battle,player=p,choice='move 1',submissionId=submission) for p in ('p1','p2')]
    def run_wire(self,f,config,events):
        memory=fixtures.MemoryBridge(events,auto_ack=False)
        return asyncio.run(f.game(config,events,memory=memory))
    def test_terminal_before_late_acks_and_stream_interleaving_never_enter_actor(self):
        for owner in ('p1','p2'):
            f,config=self.fixture(owner=owner);events=f.events(config);acks=self.acks(events[0]['battleId'])
            for late in (False,True):
                wire=(events[:5]+events[5:]+acks if late else events[:5]+acks[:1]+events[5:]+acks[1:])
                result,memory,_=self.run_wire(f,config,wire)
                self.assertEqual(memory.sent[1]['submissionId'],0)
                self.assertEqual(result.encoder_provenance['choice_acknowledgements']['acknowledged'],2)
                self.assertEqual(result.encoder_provenance['choice_acknowledgements']['pending_players'],[])
                self.assertEqual(f.backends[-1].lines,[fixtures.request(owner),'|win|'+config.pokezero_username])
    def test_exact_ack_envelopes_fail_closed_with_first_event_evidence(self):
        for mutation in ('battle','player','choice','duplicate','stale','missing-id','bool-id','extra','unsolicited'):
            with self.subTest(mutation=mutation):
                f,config=self.fixture();events=f.events(config);acks=self.acks(events[0]['battleId']);bad=dict(acks[0])
                if mutation=='battle':bad['battleId']='foreign'
                if mutation=='player':bad['player']='p3'
                if mutation=='choice':bad['choice']='move 2'
                if mutation=='stale':bad['submissionId']=1
                if mutation=='missing-id':del bad['submissionId']
                if mutation=='bool-id':bad['submissionId']=False
                if mutation=='extra':bad['unexpected']=1
                prefix=events[:4]+[bad]+events[4:] if mutation=='unsolicited' else events[:5]+(
                    [acks[0],bad] if mutation=='duplicate' else [bad])+events[5:]
                with self.assertRaises(ValueError) as caught:self.run_wire(f,config,prefix)
                witness=caught.exception.owning_encoder_receipt['first_bridge_failure_event']
                self.assertEqual(witness['event_type'],'choice_ack')
                self.assertEqual(json.loads(witness['payload_json']),bad)
                self.assertEqual(f.trajectories,[])
    def test_missing_ack_is_bounded_even_after_all_terminal_witnesses(self):
        f,config=self.fixture(owning_terminal_timeout_seconds=.005);events=f.events(config)
        wire=events[:5]+self.acks(events[0]['battleId'])[:1]+events[5:]
        memory=fixtures.MemoryBridge(wire,auto_ack=False,stall=True)
        with self.assertRaises(TimeoutError) as caught:asyncio.run(f.game(config,wire,memory=memory))
        self.assertEqual(caught.exception.owning_encoder_receipt['choice_acknowledgements']['pending_players'],['p2'])
        self.assertEqual(f.trajectories,[])
    def two_round_wire(self,f,config):
        events=f.events(config);battle=events[0]['battleId'];acks=self.acks(battle)
        next_streams=[dict(type='stream',battleId=battle,stream=seat,lines=[fixtures.request(seat)]) for seat in ('p1','p2')]
        ready=dict(type='ready',battleId=battle,requested=['p1','p2'])
        return events[:5]+acks[:1]+next_streams+[ready]+acks[1:],events[5:],battle
    def test_ready_before_last_ack_defers_exact_context_then_accepts_next_round(self):
        f,config=self.fixture();prefix,terminal,battle=self.two_round_wire(f,config)
        result,memory,_=self.run_wire(f,config,prefix+self.acks(battle,submission=1)+terminal)
        self.assertEqual([row['submissionId'] for row in memory.sent if row['type']=='choices'],[0,1])
        self.assertEqual(result.pokezero_decisions,2)
    def test_repeated_choice_stale_ack_is_distinguished_by_submission_id(self):
        f,config=self.fixture();prefix,terminal,battle=self.two_round_wire(f,config)
        with self.assertRaisesRegex(ValueError,'stale'):
            self.run_wire(f,config,prefix+self.acks(battle,submission=0)+terminal)
    def test_deferred_ready_cannot_use_later_owner_or_request_cache(self):
        f,config=self.fixture();prefix,terminal,battle=self.two_round_wire(f,config)
        changed=dict(type='stream',battleId=battle,stream='p1',lines=['|turn|2'])
        wire=prefix[:-1]+[changed]+prefix[-1:]+terminal
        with self.assertRaisesRegex(ValueError,'ready context changed'):self.run_wire(f,config,wire)
    def test_registration_precedes_send_and_all_unknown_or_simulator_errors_are_retained(self):
        f,config=self.fixture();seen=[];original=OwningWireSession.prepare_submission
        def registered(session,*args,**kwargs):
            value=original(session,*args,**kwargs);seen.append(dict(session.pending_choices));return value
        with patch.object(OwningWireSession,'prepare_submission',registered):
            result,memory,_=asyncio.run(f.game(config,f.events(config)))
        self.assertEqual(seen,[memory.sent[1]['choices']])
        for kind in ('mystery','stream_end','error'):
            f,config=self.fixture();events=f.events(config)
            event=dict(type=kind,battleId=events[0]['battleId'],stream='p1',message='AUTHORED failure',source={'line':17})
            with self.assertRaises((ValueError,RuntimeError)) as caught:self.run_wire(f,config,events[:4]+[event]+events[4:])
            witness=caught.exception.owning_encoder_receipt['first_bridge_failure_event']
            self.assertEqual(witness['event_type'],kind);self.assertEqual(witness['message'],'AUTHORED failure')
            self.assertEqual(json.loads(witness['payload_json']),event)
    def test_real_bridge_error_conversion_preserves_bounded_source_envelope(self):
        f,config=self.fixture();events=f.events(config)
        event=dict(type='error',battleId=events[0]['battleId'],message='AUTHORED error',errorOrigin='stream',extra='x'*12000)
        class ErrorBridge(fixtures.MemoryBridge):
            async def next_event(self):
                value=await super().next_event()
                if value.get('type')=='error':raise bridge._BattleBridge._event_error(value)
                return value
        memory=ErrorBridge(events[:4]+[event],auto_ack=False)
        with self.assertRaisesRegex(RuntimeError,'AUTHORED error') as caught:asyncio.run(f.game(config,memory.events,memory=memory))
        witness=caught.exception.owning_encoder_receipt['first_bridge_failure_event']
        self.assertEqual(witness['event_type'],'error');self.assertTrue(witness['payload_truncated'])
        self.assertLessEqual(len(witness['payload_json']),8192)


class NodeAckSourceTests(unittest.TestCase):
    def test_actual_node_functions_echo_only_opt_in_identity_after_each_fake_write(self):
        source=(Path(__file__).resolve().parents[1]/'scripts/battle_bridge.mjs').read_text()
        functions=source[source.index('function choiceSubmissionId('):source.index('async function endBattle(')]
        script='''
const trace=[]; const events=[];
const emit=x=>{events.push(x);trace.push('ack:'+x.player)};
const invalidateBoundaryRequests=()=>{};
const battle={battleId:'AUTHORED',streams:{
 p1:{write:async choice=>{trace.push('write:p1')}},
 p2:{write:async choice=>{trace.push('write:p2');events.push({type:'stream'},{type:'terminal'})}}
}};
const requireBattle=()=>battle;
'''+functions+'''
await sendChoices({choices:{p1:'move 1',p2:'move 2'},submissionId:7});
if (JSON.stringify(trace)!==JSON.stringify(['write:p1','ack:p1','write:p2','ack:p2'])) throw Error('write/ack order');
if (events[0].submissionId!==7 || events[3].submissionId!==7) throw Error('identity echo');
events.length=0;
await sendChoices({choices:{p1:'move 1'}});
if (JSON.stringify(events)!==JSON.stringify([{type:'choice_ack',battleId:'AUTHORED',player:'p1',choice:'move 1'}])) throw Error('legacy bytes');
for (const value of [true,false,-1,1.5,2147483648,'0',null]) {
 let refused=false;try {choiceSubmissionId({submissionId:value})} catch {refused=true}
 if (!refused) throw Error('invalid control id');
}
console.log('AUTHORED_NODE_ACK_PASS');
'''
        node=shutil.which('node');self.assertIsNotNone(node)
        result=subprocess.run([node,'--input-type=module','-e',script],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.strip(),'AUTHORED_NODE_ACK_PASS')
