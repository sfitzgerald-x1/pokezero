"""Inert boundary/transport controls. No actual engine turn or Native battle."""
import ast
from collections import namedtuple
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from gpu_engine import device_observation_collector as c
from gpu_engine.device_observation_parity import AgreementParity, SHAPES


def view():
    return {key: np.zeros(shape, bool if kind == 'b' else np.float32 if kind == 'f' else np.int32)
            for key, (shape, kind) in SHAPES.items()}


class Ref:
    def __init__(self, poison=False): self.poison=poison;self.feeds=[];self.calls=0
    def feed(self, lines): self.feeds.append(list(lines))
    def view(self):
        self.calls+=1
        if self.poison:return None,dict(source='B2_PROTOCOL',kind='unsupported',column='ALL',reason='actual unsupported line')
        return view(),None
    def metadata(self):return dict(feed_count=len(self.feeds))


class Factory:
    def __init__(self, poison=False):self.refs=[];self.poison=poison
    def game(self, **kw):
        value=Ref(self.poison);self.refs.append(value);return value


class Env:
    def __init__(self, config):self.n=0;self.owner_transcripts=[[],[]];self.closed=False
    def reset_pool_sets(self, packed, *, seed):
        self.owner_transcripts=[['|turn|0','|request|literal-p1'],['|turn|0','|request|literal-p2']]
    def snapshot(self):return NS(turn=self.n)
    def step(self, actions):
        self.n+=1
        for lines in self.owner_transcripts:lines.extend(['|move|actual','|turn|'+str(self.n)])
    def close(self):self.closed=True


class Coupling:
    def __init__(self, folder, config):self.config=config;self.ends=0
    def initialized(self):return []
    def end(self):self.ends+=1;return []


class Writer:
    def __init__(self):self.events=[];self.boundaries=[];self.rows=[]
    def event(self,row):self.events.append(row)
    def boundary(self,row):
        payload=c._json(row)
        self.boundaries.append(json.loads(payload));return hashlib.sha256(payload.encode()).hexdigest()
    def decision(self,metadata,candidate,reference):self.rows.append((metadata,candidate,reference))


State=namedtuple('State','turn done winner')
Chance=namedtuple('Chance','draws phazing secondary sleep_talk entry_ties initial_tie')


class Runtime:
    def __init__(self, *, poison=False, post_diverge=False, fail=False, native_fail=False):
        self.plan=dict(showdown_root='/not-invoked',node='not-invoked',sampling_seed=2,max_requests=3,max_turns=100)
        self.factory=Factory(poison);self.config=object();self.post_diverge=post_diverge
        self.fail=fail;self.native_fail=native_fail;self.samples=[];self.received=[];self.coupled=[]
    def state(self,n):return State(np.array([n],np.int32),np.array([n>=3]),np.array([2],np.int32))
    def construct(self,cell):return self.state(0),object(),self.state(0)
    def parties(self,raw,cell):return [[dict(species=str(side*6+p))for p in range(6)]for side in (0,1)]
    def boundary(self,s):return dict(state={k:np.asarray(getattr(s,k)).tolist()for k in s._fields},requested=[True,True])
    def differences(self,s,snapshot,env,parties):
        return [dict(field='literal_hp',gpu=1,showdown=2)]if self.post_diverge and int(s.turn[0])==1 else []
    def observe(self,s,t):
        v={k:x[None]for k,x in view().items()};v['error']=np.zeros(1,np.int32)
        return np.ones((1,2,9),bool),np.ones((1,2),bool),np.ones((1,2,4),bool),(NS(**v),NS(**v))
    def selection(self,s,a,r,u):return [dict(requested=True,kind='move',move='surf')]*2
    def sample(self,s):
        chance=Chance(*(np.zeros((1,2,12)if k=='draws'else(1,)if k=='initial_tie'else(1,2),np.int32)for k in Chance._fields))
        self.samples.append(chance);return chance
    def advance(self,s,a,t,chance):
        self.received.append(chance)
        if self.fail:raise ValueError('actual failed step')
        return NS(step=NS(state=self.state(int(s.turn[0])+1),invalid_actions=np.array([False])),trackers=t)
    def chance_control(self,coupling,s,chance,cell,request):self.coupled.append(chance);return dict(nonce=request)


CELL=dict(stream_id=12,root_id=22,native_seed=32,team_indices=[0,1])


class DeviceObservationCollectorTests(unittest.TestCase):
    def run_game(self,runtime):
        writer=Writer();reducer=AgreementParity(source_bindings={'inert':'fixture'})
        native=[]
        def env(config):v=Env(config);native.append(v);return v
        def native_actions(*args):
            if runtime.native_fail:raise ValueError('native action mapping refused')
            return {'p1':0,'p2':0}
        with patch('gpu_engine.finite_stream_comparison.pack_sets',return_value=['set1','set2']), \
                patch('gpu_engine.finite_gameplay_comparison.native_evidence',side_effect=lambda snapshot,env:dict(turn=snapshot.turn)), \
                patch('gpu_engine.finite_gameplay_comparison.native_actions',side_effect=native_actions):
            result=c.collect_game(runtime,CELL,writer,reducer,env_factory=env,coupling_factory=Coupling,folder='/not-created')
        self.assertTrue(native[0].closed)
        return result,writer,reducer.report(),native[0]

    def test_future_divergence_never_erases_current_pre_view_and_one_same_chance(self):
        r=Runtime(post_diverge=True);result,w,report,env=self.run_game(r)
        self.assertEqual(result['status'],'TERMINAL');self.assertEqual(result['executed_steps'],3)
        self.assertEqual(report['counts']['attempted'],6);self.assertEqual(report['counts']['agreement'],2)
        self.assertEqual(report['counts']['compared'],2);self.assertEqual(report['counts']['state_disagreement'],4)
        self.assertEqual([x[0]['agreement']for x in w.rows],[True,True,False,False,False,False])
        self.assertEqual(len(r.samples),3);self.assertEqual(env.n,3)
        for sampled,received,coupled in zip(r.samples,r.received,r.coupled):
            self.assertIs(sampled,received);self.assertIs(sampled,coupled)
        self.assertEqual(w.boundaries[2]['owning_lines_pre'][0][0],'|turn|0')
        self.assertIn('|turn|1',w.boundaries[2]['owning_lines_post'][0])
        self.assertEqual(w.events[0]['kind'],'START');self.assertEqual(w.events[-1]['kind'],'END')

    def test_protocol_poison_does_not_stop_engine_or_shrink_agreeing_denominator(self):
        r=Runtime(poison=True);result,w,report,env=self.run_game(r)
        self.assertEqual(result['status'],'TERMINAL');self.assertEqual(env.n,3)
        self.assertEqual(report['counts']['agreement'],6);self.assertEqual(report['counts']['comparison_refused'],6)
        self.assertEqual(report['counts'].get('compared',0),0)
        self.assertEqual(report['joint_match_fraction_on_ALL_agreement_views'],0)
        self.assertTrue(all(x[1]is not None and x[2]is None for x in w.rows))
        self.assertFalse(report['full_B4_pass']);self.assertFalse(report['collection_provenance_independently_verified'])

    def test_native_failure_continues_engine_and_engine_failure_ends_cell(self):
        result,w,p,env=self.run_game(Runtime(native_fail=True))
        self.assertEqual(result['status'],'TERMINAL');self.assertEqual(result['executed_steps'],3)
        self.assertEqual(p['counts']['agreement'],2);self.assertEqual(p['counts']['state_agreement_unknown'],4)
        self.assertEqual(env.n,0)
        result,w,p,env=self.run_game(Runtime(fail=True))
        self.assertEqual(result['status'],'ENGINE_ADVANCEMENT_FAILED');self.assertEqual(result['executed_steps'],0)
        self.assertEqual(p['counts']['attempted'],2);self.assertEqual(p['counts']['agreement'],2)
        self.assertEqual(env.n,0)

    def test_lossless_evidence_and_physical_binding_not_tensor_derived(self):
        with tempfile.TemporaryDirectory()as folder:
            writer=c.Evidence(Path(folder)/'fresh')
            data=view();data['numeric_features'][0,0]=np.nan
            data['switch_targets']=np.array([2**31+1,0,1,2,3],np.uint32)
            physical=writer.boundary(dict(before=dict(hp=12),after=dict(hp=11)))
            writer.decision(dict(decision_id='x',physical_binding_sha256=physical),data,None);writer.close()
            row=json.loads((Path(folder)/'fresh/decisions.jsonl').read_text())
            with np.load(Path(folder)/'fresh'/row['tensor_path'],allow_pickle=False)as archive:
                self.assertTrue(np.isnan(archive['candidate__numeric_features'][0,0]))
                self.assertEqual(int(archive['candidate__switch_targets'][0]),2**31+1)
                self.assertFalse(any(k.startswith('reference__')for k in archive.files))
            self.assertEqual(physical,hashlib.sha256((Path(folder)/'fresh/physical.jsonl').read_bytes().rstrip(b'\n')).hexdigest())

    def test_declared_actions_source_and_shape_contract(self):
        mask=np.zeros((2,9),bool);mask[0,[1,7]]=True
        requested=np.array([True,False]);a=c.choose_legal(mask,requested,seed=1,stream=3,step=2)
        np.testing.assert_array_equal(a,c.choose_legal(mask,requested,seed=1,stream=3,step=2))
        self.assertIn(int(a[0,0]),(1,7));self.assertEqual(int(a[0,1]),0)
        with self.assertRaises(ValueError):c.choose_legal(mask.astype(int),requested,seed=1,stream=3,step=2)
        source=Path(c.__file__).read_text();tree=ast.parse(source)
        calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)]
        steps=[n for n in calls if isinstance(n.func,ast.Attribute)and n.func.attr=='step_with_public']
        self.assertEqual(len(steps),1)
        self.assertEqual(ast.unparse(next(k.value for k in steps[0].keywords if k.arg=='chance_inputs')),'chance')
        for forbidden in ('compare_trace','captured_words','build_v5_model','forward','generate_teams'):
            self.assertFalse(any(isinstance(n.func,ast.Attribute)and n.func.attr==forbidden for n in calls))

    def test_original_chance_control_uses_same_literal_pending_arrays(self):
        # Pure original packet/control transport, not a step or constructor.
        from gpu_engine.state import empty_state
        from gpu_engine.finite_gameplay_comparison import ChanceControl
        from pokezero.local_showdown import LocalShowdownConfig
        s=empty_state(1)._replace(pending_residual=np.array([True]),
            residual_draws=np.ones((1,2,12),np.int32),secondary_draws=np.full((1,2),77,np.int32),
            sleep_talk_draws=np.full((1,2),59,np.int32))
        chance=Chance(*(np.zeros((1,2,12)if k=='draws'else(1,)if k=='initial_tie'else(1,2),np.int32)for k in Chance._fields))
        runtime=object.__new__(c.Runtime)
        with tempfile.TemporaryDirectory()as folder:
            control=ChanceControl(Path(folder)/'control',LocalShowdownConfig())
            value=runtime.chance_control(control,s,chance,dict(stream_id=2**31+1),4)
            self.assertEqual(value['nonce'],str(2**31+1)+':4')
            self.assertEqual(value['draws'],np.ones((2,12),int).tolist())
            self.assertEqual(value['secondary'],[77,77]);self.assertEqual(value['sleep_talk'],[59,59])
            self.assertEqual(json.loads(control.control.read_text()),value)
            self.assertEqual(control.end(),[])
        # Fresh inputs remain zero; selectors never mutate the supplied tape.
        self.assertFalse(chance.draws.any())

    def test_plan_hash_order_refusal_and_disabled_before_work(self):
        ref=dict(path='/stored/file',sha256='a'*64)
        p=dict(schema=c.SCHEMA,source_commit='b'*40,pool=ref,vocabulary=ref,labels=ref,gender_policy=ref,
            showdown_root='/showdown',node='/node',node_sha256='c'*64,constructor_seed=1,sampling_seed=2,
            gender_seed=3,gender_domain=4,max_requests=2,max_turns=100,cells=[CELL],
            policy='PCG64_SEEDSEQUENCE_UNIFORM_PHYSICAL_LEGAL_V1')
        self.assertIs(c.validate_plan(p),p)
        bad=copy.deepcopy(p);bad['cells'].append(copy.deepcopy(CELL))
        with self.assertRaises(ValueError):c.validate_plan(bad)
        bad=copy.deepcopy(p);bad['cells'][0]['team_indices']=[1,1]
        with self.assertRaises(ValueError):c.validate_plan(bad)
        with patch.object(c,'prepare',side_effect=AssertionError('must not prepare')):
            with self.assertRaises(ValueError):c.collect(p,out='/shared/scott/no-work')
            with self.assertRaises(ValueError):c.collect(p,out='/tmp/local-data',enable_collection=True)
        with tempfile.TemporaryDirectory()as folder:
            path=Path(folder)/'config';path.write_text(json.dumps(p))
            loaded=c.checked_document(dict(path=str(path),sha256=c.digest(path)))
            self.assertEqual(loaded,p)
            path.write_text('{}')
            with self.assertRaises(ValueError):c.checked_document(dict(path=str(path),sha256='f'*64))

    def test_actual_config_call_uses_semantic_category_not_champion_document_hash(self):
        # Execute the actual config call AST only: no catalogs, model, source
        # loader, device compiler or prospective game is constructed.
        from pokezero.category_vocab import CategoryVocabulary
        from pokezero.v5_contract import LabelVocabulary
        from pokezero.v5_model import V5ModelConfig
        from pokezero.v5_phasec import category_binding
        vocabulary=CategoryVocabulary(('move:surf','pokemon:self'),16,{})
        labels=dict(source_binding_sha256='a'*64,moves=['surf'],species_variants=[['gardevoir',1]])
        tree=ast.parse(Path(c.__file__).read_text())
        prepare=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='prepare')
        assignment=next(n for n in prepare.body if isinstance(n,ast.Assign)
            and any(isinstance(x,ast.Name)and x.id=='model_config'for x in n.targets))
        context=dict(V5ModelConfig=V5ModelConfig,LabelVocabulary=LabelVocabulary,vocabulary=vocabulary,
            category_binding=category_binding,labels=labels,vocab_doc=dict(category_vocab_sha256='b'*64))
        exec(compile(ast.Module(body=[assignment],type_ignores=[]),'actual-config-call','exec'),context)
        self.assertEqual(context['model_config'].category_vocabulary_sha256,category_binding(vocabulary))
        self.assertNotEqual(context['model_config'].category_vocabulary_sha256,'b'*64)


if __name__=='__main__':unittest.main()
