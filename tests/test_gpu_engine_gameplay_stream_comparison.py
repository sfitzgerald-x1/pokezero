"""Literal tiny/fake call graphs only: no engine/Native/model/game is run."""
import ast
from collections import namedtuple
from copy import deepcopy
from pathlib import Path
import json
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gpu_engine import engine_gameplay_trace as trace, finite_gameplay_comparison as comparison
from gpu_engine.state import BattleState, empty_state
from gpu_engine import benchmark_retained_games as bench
from gpu_engine.transition import ChanceInputs
import test_gpu_engine_benchmark_retained_games as benchmark_fixture


class Packet(NS):
    def _asdict(self):return vars(self)


def chance(batch=1):
    return ChanceInputs(np.zeros((batch,2,12),np.int32),*[np.zeros((batch,2),np.int32)for _ in range(4)],np.zeros(batch,np.int32))


def state(batch=1, *, hp=100, done=False):
    # Authored storage: engine storage defaults (empty_state), no initialize/step. A field this fixture does not author
    # takes empty_state's shape, dtype and fill, so a leaf a later tier adds arrives shaped. (Every unauthored field used
    # to be zeros of shape (batch,), the fidelity M1/M2 per-mon leaves too, and entry.depart indexes those per mon:
    # recharge_counter in its M1 block, the perish, yawn and partial-trap fields in its M2 block.)
    fields = {name:np.array(value) for name,value in empty_state(batch)._asdict().items()}
    for name in ('species','level','hidden_power_type','hidden_power_power','happiness','gender','hp','max_hp',
                 'status','status_counter','sleep_skipped','rest_origin','choice_move','last_move','encore_move','encore_counter',
                 'charge_move','charge_counter','item','ability','base_ability','volatile','confusion_counter','protect_counter',
                 'substitute_hp','trap_source','truant_loaf'):
        fields[name]=np.zeros((batch,2,6),np.int32)
    fields.update(species=np.broadcast_to(np.arange(1,13).reshape(1,2,6),(batch,2,6)).copy(),
        level=np.full((batch,2,6),80,np.int32),hp=np.full((batch,2,6),hp,np.int32),max_hp=np.full((batch,2,6),100,np.int32),
        stats=np.full((batch,2,6,6),100,np.int32),types=np.ones((batch,2,6,2),np.int32),base_types=np.ones((batch,2,6,2),np.int32),
        moves=np.ones((batch,2,6,4),np.int32),pp=np.full((batch,2,6,4),10,np.int32),disabled=np.zeros((batch,2,6,4),bool),
        boosts=np.zeros((batch,2,6,7),np.int32),active=np.zeros((batch,2),np.int32),
        ability=np.ones((batch,2,6),np.int32),entry_pending=np.zeros((batch,2),bool),
        forced_switch=np.zeros((batch,2),bool),baton_pass=np.zeros((batch,2),bool),trapped=np.zeros((batch,2),bool),
        last_action_kind=np.zeros((batch,2),np.int32),last_action_index=np.zeros((batch,2),np.int32),
        keys=np.zeros((batch,2),np.uint32),volatile=np.zeros((batch,2,6),np.uint32),
        done=np.full(batch,done,bool),winner=np.zeros(batch,np.int32),pending_residual=np.zeros(batch,bool))
    return BattleState(**fields)


def result(batch=1, *, frame=0, hp=100, done=False):
    value, config=benchmark_fixture.result(batch=batch,frames=np.full(batch,frame),done=np.full(batch,done))
    core=value.state.core;core.engine=state(batch,hp=hp,done=done)
    core.cursor=NS(stream_id=np.arange(101,101+batch,dtype=np.int32),turn=np.full(batch,frame+1,np.int32),
        phase=np.zeros(batch,np.int32),generation=np.ones((batch,2),np.int32))
    core.risks=Packet(risk=np.zeros(batch,np.int32));core.execution=NS(refusal=np.zeros(batch,np.int32))
    core.admitted=np.ones(batch,bool)
    words=np.zeros((batch,2,1,64,16),np.int32);words[:,:,0,0,0]=1;words[:,:,0,0,1]=frame+1
    valid=np.zeros(words.shape[:-1],bool);valid[:,:,0,0]=True
    core.frame_input_report=Packet(raw_words=np.zeros((batch,64,10),np.int32),raw_count=np.zeros(batch,np.int32),
        event_turn=np.zeros((batch,64),np.int32),event_phase=np.zeros((batch,64),np.int32),event_generation=np.zeros((batch,64,2),np.int32),
        switch_details=np.zeros((batch,64,6),np.int32),refusal=np.zeros(batch,np.int32),owner_full_words=words,
        owner_full_valid=valid,owner_full_more=np.zeros((batch,2,1),bool))
    for request in core.requests:request.actions.facts.mechanics=np.ones((batch,4),np.int32)
    return value,config


def write_trace(path, *, frames=2, max_bytes=10**6, batch=1):
    writer=trace.TraceWriter(path,max_bytes=max_bytes);initial,_=result(batch)
    roster=[dict(stream_id=101+i,team_indices=[2*i,2*i+1])for i in range(batch)]
    header=dict(runtime_binding='r'*64,source={'commit':'source'},pool_receipt={'pool_sha256':'a'*64},
        team_indices=np.arange(batch*2).reshape(batch,2),stream_ids=np.arange(101,101+batch),max_requests=frames,
        constructor_seed=401,sampling_seed=402,action_names=('', 'tackle','struggle'))
    writer.begin(initial=initial,**header);before=initial
    for frame in range(frames):
        after,_=result(batch,frame=frame+1,hp=100-(frame+1),done=frame==frames-1)
        writer.decision(request=frame,before=before,after=after,actions=np.zeros((batch,2),np.int32),
            active=np.ones(batch,bool),executed=np.ones(batch,bool),invalid=np.zeros(batch,bool),chance=chance(batch));before=after
    cohort=[dict(**cell,status='COMPLETED',executed_steps=frames)for cell in roster]
    descriptor=writer.finish(cohort)
    measured={k:header[k]for k in ('runtime_binding','source','pool_receipt','max_requests','constructor_seed','sampling_seed')}
    measured.update(cohort=cohort,original_denominator=batch,gameplay_trace=descriptor)
    return measured


class LiteralStateShapeTests(unittest.TestCase):
    def test_every_field_has_the_engine_shape(self):
        # The fixture authors values, not layout: every BattleState leaf has empty_state's shape, a leaf a later tier adds
        # included. A per-mon leaf left at (batch,) failed only where entry.depart indexed it per mon, behind a refusal.
        for batch in (1, 3):
            literal, engine = state(batch), empty_state(batch)
            for name in BattleState._fields:
                with self.subTest(batch=batch, field=name):
                    self.assertEqual(np.shape(getattr(literal, name)), np.shape(getattr(engine, name)))


class GameplayCaptureTests(unittest.TestCase):
    def test_actual_boundaries_semantics_hashchain_terminal_and_bounded_capture(self):
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'trace';measured=write_trace(path,batch=2)
            descriptor=measured['gameplay_trace'];self.assertTrue(descriptor['complete'])
            self.assertEqual(descriptor['sha256'],trace.digest_file(path));self.assertEqual(descriptor['attempts'],{101:2,102:2})
            for terminal in descriptor['terminals']:
                self.assertTrue(terminal['done']);self.assertEqual(terminal['winner'],0)
                self.assertEqual(terminal['hp'],[[98]*6]*2);self.assertEqual(terminal['max_hp'],[[100]*6]*2)
                self.assertEqual(terminal['frame'],2);self.assertTrue(terminal['executed']);self.assertFalse(terminal['invalid'])
            with tempfile.TemporaryDirectory()as spool:
                header,counts=comparison._spool(path,measured,descriptor['sha256'],Path(spool))
                self.assertEqual(counts,{101:2,102:2})
                rows=[json.loads(line)['payload']for line in(Path(spool)/'101').read_text().splitlines()]
                self.assertEqual(rows[1]['selected'][0]['move'],'tackle')
                self.assertEqual(rows[1]['before']['state']['hp'],[[[100]*6]*2])
                self.assertEqual(rows[2]['after']['state']['hp'],[[[98]*6]*2])
                self.assertEqual(comparison.decode_state(rows[2]['after']['state'],header['state_layout']).hp.tolist(),[[[98]*6]*2])
            with self.assertRaises(FileExistsError):trace.TraceWriter(path)
            bad=deepcopy(measured);bad['runtime_binding']='changed'
            with tempfile.TemporaryDirectory()as spool,self.assertRaisesRegex(ValueError,'measurement'):
                comparison._spool(path,bad,descriptor['sha256'],Path(spool))
            path.write_bytes(path.read_bytes()+b' ')
            with tempfile.TemporaryDirectory()as spool,self.assertRaisesRegex(ValueError,'bytes changed'):
                comparison._spool(path,measured,descriptor['sha256'],Path(spool))

    def test_capture_capacity_and_read_failure_do_not_change_terminal_or_denominator(self):
        with tempfile.TemporaryDirectory()as tmp:
            measured=write_trace(Path(tmp)/'tiny',max_bytes=4096)
            self.assertFalse(measured['gameplay_trace']['complete']);self.assertEqual(measured['original_denominator'],1)
            self.assertEqual(measured['gameplay_trace']['attempts'],{101:2});self.assertEqual(len(measured['gameplay_trace']['terminals']),1)
            self.assertEqual(measured['gameplay_trace']['written'],{101:0})
            writer=trace.TraceWriter(Path(tmp)/'io')
            writer.begin(initial=result()[0],runtime_binding='r',source={},pool_receipt={},team_indices=[[0,1]],
                stream_ids=[101],max_requests=1,constructor_seed=1,sampling_seed=2,action_names=('', 'tackle'))
            with patch.object(trace,'digest_file',side_effect=OSError('owned test IO failure')):
                receipt=writer.finish([dict(stream_id=101)])
            self.assertFalse(receipt['complete']);self.assertIsNone(receipt['sha256'])

    def test_noncontiguous_missing_full_capture_never_becomes_empty_match(self):
        actual,_=result();packet=trace.boundary(actual,0)['full'][0]
        self.assertEqual(len(comparison.captured_words(packet)),1)
        for bad in ({'available':False},dict(packet,packets=[dict(valid=[False,True]+[False]*62,more=False,words=[[0]*16])])):
            with self.assertRaises(ValueError):comparison.captured_words(bad)

    def test_terminal_and_json_dtype_joins_are_literal_not_status_or_cast_guesses(self):
        actual,_=result();packet=trace.boundary(actual,0)['observations'][0]['arrays']
        decoded=comparison.observation_arrays(packet)
        self.assertEqual(decoded['numeric_features'].dtype,np.float32)
        self.assertEqual(decoded['categorical_ids'].dtype,np.int32)
        for key,value in(('categorical_ids',[[.5]*41]*23),('attention_mask',[1]*23),('numeric_features',[[float('nan')]*132]*23)):
            bad=deepcopy(packet);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):comparison.observation_arrays(bad)
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'trace';measured=write_trace(path);bad=deepcopy(measured)
            bad['gameplay_trace']['terminals'][0]['winner']=1
            with tempfile.TemporaryDirectory()as spool,self.assertRaisesRegex(ValueError,'literal executed terminal'):
                comparison._spool(path,bad,bad['gameplay_trace']['sha256'],Path(spool))

    def test_selected_switch_is_original_physical_target_and_wait_is_ignored(self):
        actual,_=result();actual.state.core.engine=actual.state.core.engine._replace(active=np.array([[2,0]],np.int32))
        actual.state.core.requests[1].actions.facts.request=actual.state.core.requests[1].actions.facts.request._replace(wait=np.ones(1,bool))
        selected=trace.selected_actions(actual,np.array([[5,0]],np.int32),0,('', 'tackle'))
        self.assertEqual(selected[0]['physical'],1);self.assertFalse(selected[1]['requested'])

    def test_default_inference_admission_preparation_ast_is_unchanged(self):
        import subprocess
        before=ast.parse(subprocess.check_output(['git','show','c2e00592^:gpu_engine/benchmark_retained_games.py'],text=True))
        after=ast.parse(Path(bench.__file__).read_text())
        for name in ('prepare','_checked','_network','_admission','_decide','_keys'):
            get=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
            self.assertEqual(ast.dump(get(before)),ast.dump(get(after)))


class GameplayReferenceTests(unittest.TestCase):
    def test_initial_owner_append_refusal_is_local_and_native_keeps_advancing(self):
        calls=[]
        class Native:
            def __init__(self,config):self.index=0;self.owner_transcripts=[['|initial'],['|initial']]
            def reset_pool_sets(self,*args,**kw):calls.append('reset')
            def snapshot(self):return NS()
            def step(self,actions):
                calls.append('step');self.index+=1
                for lines in self.owner_transcripts:lines.append('|frame|'+str(self.index))
            def close(self):calls.append('close')
        class Owner:
            def __init__(self,*args,owner,**kw):self.owner=owner;self.frame=0
            def append(self,lines,*,constructor=False):
                if self.owner=='p2':
                    calls.append('p2-initial-refusal');raise ValueError('UNIT initial p2 reference refusal')
                if not constructor:self.frame+=1
                word=[0]*16;word[0]=1;word[1]=self.frame+1
                return dict(words=[word],raw_kinds={'move':len(lines)},refused_lines=[])
            def observation(self):
                value,_=result(frame=self.frame)
                return {k:np.asarray(getattr(value.owners[0].encoded.observation,k)[0])for k in comparison.original.SHAPES}
            def close(self):pass
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'trace';measured=write_trace(path)
            parties=[[dict(species=f'mon{s*6+i}')for i in range(1,7)]for s in(0,1)]
            with patch.object(comparison.original.source_provenance,'_pool',return_value={'pool_sha256':'a'*64}),\
                 patch.object(comparison.original,'pool_sets',return_value=parties),\
                 patch.object(comparison.original,'pack_sets',return_value=['same-p1','same-p2']),\
                 patch.object(comparison.original,'OwningShowdownEnv',Native),\
                 patch.object(comparison,'OwnerReference',Owner),\
                 patch.object(comparison,'state_differences',return_value=[]),\
                 patch.object(comparison,'native_evidence',side_effect=lambda snap,env:{'test_frame':env.index}),\
                 patch.object(comparison,'native_actions',return_value={'p1':0,'p2':0}):
                actual=comparison.compare_trace(measured,path,NS(metadata=lambda:{'vocabulary':{}}),object(),
                    expected_trace_sha256=measured['gameplay_trace']['sha256'],native_seed_start=700,
                    showdown_root='/not-opened',node='not-run',env_config=comparison.original.LocalShowdownConfig(),
                    original_ivs=NS(pool_sha256='a'*64),out_path=Path(tmp)/'out')
            self.assertEqual(calls,['reset','p2-initial-refusal','step','step','close'])
            summary=actual['summary'];conditional=summary['conditional_on_continuous_measured_state_agreement']
            self.assertEqual((conditional['matched'],conditional['unknown'],summary['actual_engine_decisions']),(2,2,2))
            self.assertEqual(actual['roots'][0]['initial_owner_reference_errors'],[None,'UNIT initial p2 reference refusal'])

    def test_existing_chance_control_uses_literal_saved_arrays_and_retains_unresolved(self):
        before=state()._replace(pending_residual=np.ones(1,bool),residual_draws=np.ones((1,2,12),np.int32),
            secondary_draws=np.full((1,2),2,np.int32),sleep_talk_draws=np.full((1,2),3,np.int32))
        layout={k:dict(dtype=str(v.dtype),shape=list(v.shape[1:]))for k,v in before._asdict().items()}
        captured=dict(stream_id=101,request=2,before={'state':trace.state_packet(before,0)},
            chance={k:v.tolist()for k,v in chance()._asdict().items() if v is not None})
        with tempfile.TemporaryDirectory()as tmp:
            control=comparison.ChanceControl(Path(tmp)/'native',comparison.original.LocalShowdownConfig())
            self.assertEqual(control.initialized(),[])
            literal=control.begin(captured,layout)
            self.assertEqual(literal['draws'],[[1]*12]*2);self.assertEqual(literal['secondary'],[2,2]);self.assertEqual(literal['sleep_talk'],[3,3])
            control.audit.write_text(json.dumps({'kind':'UNRESOLVED','role':'NO_NEW_HOOK'})+'\n')
            self.assertEqual(control.end(),[{'kind':'UNRESOLVED','role':'NO_NEW_HOOK'}]);self.assertFalse(control.control.exists())
            self.assertIn('fidelity_bridge.mjs',Path(control.config.bridge_path).read_text())

    def test_incremental_original_backend_ingests_each_owner_delta_once_and_rejects_peer(self):
        calls=[]
        class Backend:
            def __init__(self,*a,**k):calls.append(('open',k['owner']))
            def ingest(self,lines):calls.append(('ingest',tuple(lines)))
            def encode(self,snapshot):
                calls.append(('encode',snapshot.owner,snapshot.request_count))
                value,_=result();return value.owners[0].encoded.observation,None
            def close(self):calls.append(('close',))
        owner='p1';request=lambda n:'|request|'+json.dumps({'side':{'id':owner},'rqid':n})
        with patch.object(comparison.owning_python_reference,'PythonBackend',Backend),\
             patch.object(comparison.owning_python_reference.ActorContract,'champion',return_value=NS(namespace='fake')):
            reference=comparison.OwnerReference(object(),object(),owner=owner,root_id=101,seed=701)
            with patch.object(comparison.public_wire,'project_line',side_effect=lambda state,line,**kw:(state,NS(events=[]))),\
                 patch.object(comparison.v4_encoder,'wire_kwargs',return_value={}):
                initial=['|switch|p1a: A|A|100/100','|switch|p2a: B|B|100/100',request(1)]
                reference.append(initial,constructor=True);reference.append(['|',request(2)])
                reference.observation();before=tuple(reference.history)
                with self.assertRaises(ValueError):reference.append(['|request|'+json.dumps({'side':{'id':'p2'}})])
                self.assertEqual(tuple(reference.history),before)
            reference.close()
        self.assertEqual([c for c in calls if c[0]=='ingest'],[('ingest',tuple(initial)),('ingest',('|',request(2)))])
        self.assertIn(('encode','p1',2),calls);self.assertEqual(calls[-1],('close',))
    def test_native_dispatch_maps_actual_move_and_physical_switch_via_original_choices(self):
        snapshot=NS(latest_requests={'p1':{'active':[{'moves':[{'id':'hiddenpowerice'}]}]},'p2':{}},
            bridge_snapshot={'battle':{'sides':[{}, {'pokemon':[{'set':{'species':n}}for n in ('c','b','a','d','e','f')]}]}})
        env=NS(requested_players=lambda:('p1','p2'),_state_for_player=lambda player:player,
            legal_actions=lambda player: [True]+[False]*8)
        selected=[dict(requested=True,kind='move',move='hiddenpowerfire'),dict(requested=True,kind='switch',physical=2)]
        with patch.object(comparison,'showdown_choice_for_action',side_effect=lambda player,a:'move 1'if player=='p1'else'switch 1'):
            self.assertEqual(comparison.native_actions(env,snapshot,selected,[list('abcdef'),list('abcdef')]),{'p1':0,'p2':0})
        selected[1]['physical']=5
        with patch.object(comparison,'showdown_choice_for_action',side_effect=lambda player,a:'move 1'if player=='p1'else'switch 1'),self.assertRaisesRegex(ValueError,'NOT_UNIQUE_OR_LEGAL'):
            comparison.native_actions(env,snapshot,selected,[list('abcdef'),list('abcdef')])

    def test_reducer_full_unknown_denominators_and_state_difference_is_not_stream_bug(self):
        reducer=comparison.Reducer([101,102]);columns=dict(status='COMPARED',numeric_bad=[0]*132,
            numeric_max=[0.]*132,categorical_bad=[0]*41,other_bad={})
        reducer.add(dict(state_status='AGREE',owners=[dict(status='MATCH',columns=columns),dict(status='UNKNOWN',refusals=[dict(kind='-start')])]))
        reducer.add(dict(state_status='DIVERGE',owners=[dict(status='MISMATCH',columns=columns)]*2))
        summary=reducer.summary();conditional=summary['conditional_on_continuous_measured_state_agreement']
        self.assertEqual((conditional['matched'],conditional['mismatched'],conditional['unknown']),(1,0,1))
        self.assertEqual(conditional['match_fraction_bounds'],[.5,1.]);self.assertEqual(summary['original_owner_decision_views'],4)
        for column in summary['numeric_columns']+summary['categorical_columns']:
            self.assertEqual((column['expected_cells'],column['compared_cells'],column['unknown_cells']),(92,23,69))
        self.assertEqual(summary['refusals_by_event_kind'],{'-start':1})

    def test_original_full_represented_state_predicate_detects_boost_pp_and_terminal_drift(self):
        actual,_=result();boundary=trace.boundary(actual,0);layout={k:dict(dtype=str(v.dtype),shape=list(v.shape[1:]))for k,v in actual.state.core.engine._asdict().items()}
        originals=[[f'mon{i}'for i in range(s*6+1,s*6+7)]for s in(0,1)];sides=[]
        for names in originals:
            sides.append(dict(pokemon=[dict(set={'species':name},types=['Normal','Normal'],hp=100,maxhp=100,status='',fainted=False,ability='pressure',item='',
                storedStats={k:100 for k in('atk','def','spa','spd','spe')},boosts={k:0 for k in('atk','def','spa','spd','spe','accuracy','evasion')},
                moveSlots=[dict(id='tackle',pp=10)]*4)for name in names]))
        snapshot=NS(bridge_snapshot={'battle':{'sides':sides,'field':{'weather':''}}})
        env=NS(requested_players=lambda:('p1','p2'),terminal=lambda:None)
        vocabulary=dict(abilities=['','pressure'],items=[''],moves=['','tackle'],types=['','normal','water'])
        check=lambda:comparison.state_differences(boundary,snapshot,env,originals,vocabulary,layout)
        self.assertEqual(check(),[])
        sides[1]['pokemon'][0]['types']=['Water']
        self.assertEqual(check(),[dict(field='1:0:types',gpu=['normal','normal'],showdown=['water'])])
        evidence=comparison.native_evidence(snapshot,env)
        self.assertEqual(evidence['battle']['sides'][1]['pokemon'][0]['types'],['Water'])
        sides[1]['pokemon'][0]['types']=['Normal','Normal']
        sides[0]['pokemon'][4]['types']=['Water']
        self.assertEqual(check(),[dict(field='0:4:types',gpu=['normal','normal'],showdown=['water'])])
        del sides[0]['pokemon'][4]['types']
        with self.assertRaisesRegex(ValueError,'earlier captures retain their original scope'):check()
        sides[0]['pokemon'][4]['types']=['Normal','Normal']
        sides[1]['pokemon'][0]['boosts']['atk']=1;self.assertTrue(any(d['field'].endswith('boosts')for d in check()))
        sides[1]['pokemon'][0]['boosts']['atk']=0;sides[1]['pokemon'][0]['moveSlots'][0]['pp']=9
        self.assertTrue(any(d['field'].endswith('pp')for d in check()))
        env.terminal=lambda:NS(winner='p1',capped=False);self.assertTrue(any(d['field']=='terminal'for d in check()))

    def test_actual_original_advancement_seam_nonfatal_full_roster_and_no_rerun(self):
        calls=[];closed=[]
        class Native:
            def __init__(self,config):self.config=config;self.index=0;self.owner_transcripts=[['|initial'],['|initial']]
            def reset_pool_sets(self,packed,*,seed):
                self.seed=seed;calls.append(('reset',seed,packed))
                if seed==700:raise ValueError('authored first-root refusal')
            def snapshot(self):return NS(latest_requests={p:{'active':[{'moves':[{'id':'tackle'}]}]}for p in('p1','p2')},bridge_snapshot={'battle':{'sides':[{},{}]}})
            def requested_players(self):return ('p1','p2')
            def legal_actions(self,player):return [True]+[False]*8
            def _state_for_player(self,player):return player
            def step(self,actions):
                control=json.loads((Path(self.config.bridge_path).parent/'chance.json').read_text())
                self.assertion=control['draws']==[[0]*12]*2
                if not self.assertion:raise AssertionError('actual chance transport changed')
                calls.append(('step',self.seed,actions));self.index+=1
                for owner in self.owner_transcripts:owner.append('|frame|'+str(self.index))
            def close(self):closed.append(self.seed)
        class Owner:
            def __init__(self,*args,**kwargs):self.frame=0
            def append(self,lines,*,constructor=False):
                if not constructor:self.frame+=1
                word=[0]*16;word[0]=1;word[1]=self.frame+1
                return dict(words=[word],raw_kinds={'move':len(lines)},refused_lines=[])
            def observation(self):
                actual,_=result(frame=self.frame)
                return {k:np.asarray(getattr(actual.owners[0].encoded.observation,k)[0])for k in comparison.original.SHAPES}
            def close(self):pass
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'trace';measured=write_trace(path,batch=2)
            pool=NS(metadata=lambda:dict(vocabulary={}))
            parties=[[dict(species=f'mon{s*6+i}')for i in range(1,7)]for s in(0,1)]
            with patch.object(comparison.original.source_provenance,'_pool',return_value={'pool_sha256':'a'*64}),\
                 patch.object(comparison.original,'pool_sets',return_value=parties),\
                 patch.object(comparison.original,'pack_sets',return_value=['same-p1','same-p2']),\
                 patch.object(comparison.original,'OwningShowdownEnv',Native),\
                 patch.object(comparison,'OwnerReference',Owner),\
                 patch.object(comparison,'state_differences',return_value=[]),\
                 patch.object(comparison,'native_evidence',side_effect=lambda snap,env:{'literal_test_frame':env.index}),\
                 patch.object(comparison,'showdown_choice_for_action',return_value='move 1'):
                actual=comparison.compare_trace(measured,path,pool,object(),expected_trace_sha256=measured['gameplay_trace']['sha256'],
                    native_seed_start=700,showdown_root='/not-opened',node='not-run',env_config=comparison.original.LocalShowdownConfig(),
                    original_ivs=NS(pool_sha256='a'*64),out_path=Path(tmp)/'decisions')
            self.assertEqual(calls,[('reset',700,['same-p1','same-p2']),('reset',701,['same-p1','same-p2']),
                ('step',701,{'p1':0,'p2':0}),('step',701,{'p1':0,'p2':0})])
            self.assertEqual(closed,[700,701]);summary=actual['summary']
            self.assertEqual((summary['original_roots'],summary['actual_engine_decisions'],summary['original_owner_decision_views']),(2,4,8))
            conditional=summary['conditional_on_continuous_measured_state_agreement']
            self.assertEqual((conditional['matched'],conditional['mismatched'],conditional['unknown']),(4,0,0))
            self.assertEqual(summary['nonagreement_or_unknown_state_views'],4)
            self.assertEqual(summary['refusals_by_event_kind'],{'NATIVE_CONSTRUCTOR_REFUSAL':4})
            for column in summary['numeric_columns']+summary['categorical_columns']:
                self.assertEqual((column['expected_cells'],column['compared_cells'],column['unknown_cells']),(184,92,92))
            self.assertFalse(actual['full_B4_qualified'])


class BenchmarkGameplaySeamTests(unittest.TestCase):
    def test_opt_in_captures_once_before_completion_without_extra_fold_or_forward(self):
        from contextlib import ExitStack
        import jax.numpy as j
        first,config=result(batch=2);session=NS(state=first.state,result=first)
        runtime=bench.Runtime('r'*64)
        record=dict(prepared=object(),config=config,metadata={},params=None)
        costs=dict(model_rows=4,forward_ms=1.,compile_ms=0.)
        original_chance=chance(2)
        def advance(session,actions,*,active_rows,timing_callback,chance_inputs=None):
            if chance_inputs is not None:self.assertIs(chance_inputs,original_chance)
            after,_=result(batch=2,frame=1,hp=99,done=True);session.result=after;session.state=after.state
            session.diagnostics=NS(active_rows=active_rows,executed_rows=active_rows,invalid_actions=np.zeros(2,bool),placeholder_moves=np.zeros(2,np.int32))
            for name in('engine','requests','public_p1','public_p2','encode_p1','encode_p2','admission_p1','admission_p2'):timing_callback(name,.001)
            return after
        with tempfile.TemporaryDirectory()as tmp,ExitStack()as stack:
            writer=trace.TraceWriter(Path(tmp)/'trace')
            stack.enter_context(patch.object(bench,'_checked',return_value=record))
            stack.enter_context(patch.object(bench.driver,'_checked',return_value={'request':{'catalog':NS(names=('', 'tackle'))}}))
            construct=stack.enter_context(patch.object(bench.driver,'construct',return_value=session))
            advancing=stack.enter_context(patch.object(bench.driver,'advance',side_effect=advance))
            deciding=stack.enter_context(patch.object(bench,'_decide',return_value=(j.zeros((2,2),j.int32),np.zeros(2,bool),costs)))
            sampler=stack.enter_context(patch.object(bench.driver.transition,'sample_chance_inputs',return_value=original_chance))
            stack.enter_context(patch.object(bench.driver,'latest_diagnostics',side_effect=lambda _:session.diagnostics))
            stack.enter_context(patch.object(bench.source_provenance,'_pool',return_value={'pool_sha256':'a'*64}))
            stack.enter_context(patch.object(bench,'_source_snapshot',return_value={'commit':'fake'}))
            actual=bench.measure(runtime,NS(arrays={'species':np.zeros((4,6),np.int32)}),np.array([[0,1],[2,3]]),
                stream_ids=np.array([101,102],np.int32),max_requests=2,enable_unqualified=True,gameplay_trace=writer)
            self.assertEqual((construct.call_count,advancing.call_count,deciding.call_count),(1,1,1))
            self.assertEqual(sampler.call_count,1)
            self.assertEqual((actual['completed_games'],actual['refused_games'],actual['original_denominator']),(2,0,2))
            self.assertTrue(actual['gameplay_trace']['complete']);self.assertEqual(len(actual['gameplay_trace']['terminals']),2)
            self.assertGreaterEqual(actual['host_attribution']['samples'][0]['benchmark']['gameplay_capture_ms'],0)
            with tempfile.TemporaryDirectory()as spool:
                _,counts=comparison._spool(writer.path,actual,actual['gameplay_trace']['sha256'],Path(spool))
                self.assertEqual(counts,{101:1,102:1})
            # The original default route adds neither a sampler call nor a
            # trace/terminal artifact; its ordinary completion counts persist.
            first,_=result(batch=2);session=NS(state=first.state,result=first)
            construct.return_value=session;sampler.reset_mock();advancing.reset_mock();deciding.reset_mock()
            default=bench.measure(runtime,NS(arrays={'species':np.zeros((4,6),np.int32)}),np.array([[0,1],[2,3]]),
                stream_ids=np.array([101,102],np.int32),max_requests=2,enable_unqualified=True)
            self.assertEqual((advancing.call_count,deciding.call_count,sampler.call_count),(1,1,0))
            self.assertNotIn('gameplay_trace',default)
            for key in('completed_games','refused_games','original_denominator','executed_game_steps','model_owner_rows'):
                self.assertEqual(default[key],actual[key])


if __name__=='__main__':unittest.main()
