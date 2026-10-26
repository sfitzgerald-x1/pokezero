"""Authored collector integration only; no fresh scientific roster or credit."""
from copy import deepcopy
from dataclasses import asdict,replace
import contextlib
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

AVAILABLE=all(importlib.util.find_spec(n) for n in ('jax','numpy','poke_engine','pokezero_search'))
if AVAILABLE:
    import pokezero_search
    AVAILABLE=bool(getattr(pokezero_search,'CONDITIONED_ORACLE_ENABLED',False))


@unittest.skipUnless(AVAILABLE,'isolated JAX and both native bindings required')
class ProspectivePanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        import numpy as np
        from gpu_engine import (prospective_panel as p,source_provenance as s,
            future_state as f,transition as t,collect_fidelity_panel as c)
        spec=importlib.util.spec_from_file_location('original_fixtures',
            Path(__file__).with_name('test_constructor_provenance.py'))
        fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
        fixtures.ConstructorProvenanceTests.setUpClass()
        cls.Fixture=fixtures.ConstructorProvenanceTests
        cls.jax,cls.j,cls.np,cls.p,cls.s,cls.f,cls.t,cls.c=jax,j,np,p,s,f,t,c

    def fixture(self,side=0,**options):
        helper=self.Fixture('test_capture_is_not_a_correspondence_and_preserves_pool_nature_evs_and_gender')
        self.addCleanup(helper.doCleanups)
        state,pool,context,catalog,assignment=helper.fixture(side,**options)
        sources=self.p.Sources(pool,context,helper.runtime,42)
        source=helper.capture(state,pool,context,assignment)
        entry=dict(slots=2,team_start=0,prefix_cycle=1,seed=471,rng_domain=19,
            source_commit='a'*40,pool_sha256=context.pool_sha256,max_cases=4096,
            case_batch=4,mass_tolerance=1e-5,probability_tolerance=1e-5)
        if options.get('generated'):
            entry['gender_initialization']=dict(seed=7,domain=13,root_start=42,
                policy=json.loads(context.gender_policy_json))
        return state,sources,source,entry,catalog

    def table(self,metadata):
        from gpu_engine import baton,entry,items,secondary
        j=self.j
        return self.t.MoveTable(j.array([0,0,0,95,50]),j.array([0,1,1,13,1]),
            j.zeros(5,bool),j.full(5,100),j.array([True,True,True,False,False]),
            j.zeros(5,j.int32),j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(4),
            baton=baton.from_metadata(metadata),entry=entry.from_metadata(metadata),
            items=items.from_metadata(metadata),secondary=secondary.from_metadata(metadata))

    def record(self,state,initial,source,sources,actions):
        return dict(slot=0,teams=[0,1],state=self.f._sample(state,0),actions=list(actions),
            constructor_provenance=asdict(source),post_entry_state=self.f._sample(initial,0),
            constructor_context_sha256=sources.context.sha256)

    def plan(self,state,initial,source,sources,entry,table,record):
        return self.p.plan_for_record(state,initial,record,source,sources,table,entry,
            record_sha256=self.s._sha(record))

    def test_ordinary_post_trace_prefix_keeps_originals_and_current_root_binding(self):
        state,sources,source,entry,_=self.fixture()
        table=self.table(sources.context.metadata())
        # Entry Trace is not implemented by the JAX kernel. This is the same
        # represented post-Trace fixture whose physical native wire is checked
        # independently in test_constructor_provenance, not Trace effect credit.
        initial=state._replace(entry_pending=self.j.zeros((1,2),bool),
            ability=state.ability.at[:,0,0].set(1))
        self.assertEqual(int(initial.ability[0,0,0]),1)
        root=initial._replace(turn=initial.turn+1)
        record=self.record(root,initial,source,sources,(0,0))
        plan=self.plan(root,initial,source,sources,entry,table,record)
        self.assertNotIsInstance(plan,self.f.Refused,plan)
        self.assertEqual(plan.native_input.certificate.input_fields,self.f._freeze(self.f._sample(root,0)))
        actor=self.f._native_wire(plan.native_input.raw_native)['side_one']['pokemon'][0]
        self.assertEqual((actor['ability'],actor['base_ability']),('static','trace'))
        self.assertEqual(dict(plan.native_input.certificate.static_native)['side_one.pokemon[0]'][0],'jolly')

    def test_both_seat_suspension_dispatch_retains_all_four_native_cases(self):
        from gpu_engine.conditional_law import ConditionalConsumerPlan
        from gpu_engine.fidelity_panel import JaxCaseExecutor,compare_slot
        j=self.j
        for side in (0,1):
            state,sources,source,entry,_=self.fixture(side,trace=False)
            table=self.table(sources.context.metadata());initial=self.t.initialize(state,table)
            chance=self.t.ChanceInputs(j.zeros((1,2,12),j.int32).at[:,:,4].set(15).at[:,:,5].set(7),
                j.full((1,2),59),j.full((1,2),99),j.full((1,2),59),j.array([[0,1]]),j.array([0]))
            root=self.jax.jit(self.t.step_with_chance)(initial,j.zeros((1,2),j.int32),table,chance).state
            self.assertTrue(bool(root.pending_residual[0]))
            actions=(4,0) if side==0 else (0,4)
            record=self.record(root,initial,source,sources,actions)
            plan=self.plan(root,initial,source,sources,entry,table,record)
            self.assertIsInstance(plan,ConditionalConsumerPlan,plan)
            self.assertEqual(len(plan.native_evidence()),4)
            result=compare_slot(plan,JaxCaseExecutor(root,actions,table,plan,batch_size=4),
                plan.native_results,batch_size=4,mass_tolerance=1e-5,probability_tolerance=1e-5)
            self.assertEqual(result['status'],'AGREE',result)

    def test_frozen_record_and_current_lineage_refuse_without_fallback(self):
        state,sources,source,entry,_=self.fixture(trace=False)
        table=self.table(sources.context.metadata());initial=self.t.initialize(state,table)
        record=self.record(initial,initial,source,sources,(0,0));digest=self.s._sha(record)
        changed=deepcopy(record);changed['actions']=[1,0]
        with mock.patch('gpu_engine.chance_consumers.plan_for_state',side_effect=AssertionError('fallback')), \
             mock.patch('gpu_engine.source_provenance.native_input_with_provenance',side_effect=AssertionError('native construction')):
            plan=self.p.plan_for_record(initial,initial,changed,source,sources,table,entry,record_sha256=digest)
            self.assertIsInstance(plan,self.f.Refused)
            self.assertIn('FROZEN_CONSTRUCTOR_SOURCE_CHANGED',repr(plan.gaps))
            wrong=replace(source,root_id=43)
            plan=self.plan(initial,initial,wrong,sources,entry,table,self.record(initial,initial,wrong,sources,(0,0)))
            self.assertIsInstance(plan,self.f.Refused)
            missing=self.f.Refused((self.f.Gap('source','MISSING_ORIGINAL','authored'),))
            plan=self.plan(initial,initial,missing,sources,entry,table,self.record(initial,initial,missing,sources,(0,0)))
            self.assertIs(plan,missing)
            with mock.patch('gpu_engine.conditional_law.plan_for_suspension',side_effect=AssertionError('conditional fallback')):
                for original in (None,source):
                    for field in ('constructor_provenance','post_entry_state'):
                        broken=deepcopy(record);broken.pop(field)
                        plan=self.p.plan_for_record(initial,initial,broken,original,sources,table,entry,
                            record_sha256=self.s._sha(broken))
                        self.assertIsInstance(plan,self.f.Refused)
            mutated=initial._replace(stats=initial.stats.at[:,0,0,1].add(1),pending_residual=self.j.array([True]))
            plan=self.plan(mutated,initial,source,sources,entry,table,self.record(mutated,initial,source,sources,(4,0)))
            self.assertIn('IMMUTABLE_CONSTRUCTOR_IDENTITY_CHANGED',repr(plan.gaps))

    def test_late_conditional_refusal_retains_whole_slot_before_jax(self):
        from gpu_engine import conditioned_future as conditioned
        from gpu_engine.conditional_law import plan_for_suspension
        state,sources,source,entry,_=self.fixture(trace=False)
        table=self.table(sources.context.metadata());initial=self.t.initialize(state,table)
        root=self.t.step(initial,self.j.zeros((1,2),self.j.int32),table).state
        record=self.record(root,initial,source,sources,(4,0))
        actual=conditioned.native_successors;calls=[]
        def evaluate(*args,**kwargs):
            calls.append(tuple(args[2]))
            return self.f.Refused((self.f.Gap('authored','LAST_CASE','kept'),)) if len(calls)==4 else actual(*args,**kwargs)
        with mock.patch.object(conditioned,'native_successors',side_effect=evaluate), \
             mock.patch('gpu_engine.fidelity_panel.JaxCaseExecutor',side_effect=AssertionError('JAX before native admission')):
            plan=self.plan(root,initial,source,sources,entry,table,record)
        self.assertIsInstance(plan,self.f.Refused,plan)
        self.assertEqual(len(calls),4)
        self.assertIn('"completed_native_cases": 3',repr(plan.gaps))

    def test_all_inputs_and_context_close_before_any_plan(self):
        from gpu_engine import prospective_panel as prospective,panel_owner as owner
        state,sources,source,entry,_=self.fixture(trace=False)
        rows=(dict(slot=0,teams=[0,1],prefix_requests=0),dict(slot=1,teams=[0,1],prefix_requests=0))
        calls=[]
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'panel'
            def dispatch(root,initial,record,original,bound,table,config,*,record_sha256):
                frozen=json.loads((out/'inputs-frozen.json').read_bytes())
                with gzip.open(out/'inputs.jsonl.gz','rt') as stream:records=[json.loads(line) for line in stream]
                self.assertEqual(len(records),2)
                self.assertEqual(hashlib.sha256((out/'constructor-context.json').read_bytes()).hexdigest(),
                    frozen['constructor_context_file_sha256'])
                self.assertTrue(all('constructor_provenance' in r and 'post_entry_state' in r for r in records))
                self.assertEqual(self.s._sha(record),record_sha256)
                calls.append(record['slot'])
                return self.f.Refused((self.f.Gap('authored','UNSUPPORTED','kept'),))
            with mock.patch.object(self.c,'runtime_admission',return_value={}), \
                 mock.patch('gpu_engine.benchmark.sole_gpu',return_value=type('Device',(),{'device_kind':'authored CPU fixture'})()), \
                 mock.patch.object(prospective,'admit',return_value=sources), \
                 mock.patch.object(self.c,'source_rows',return_value=rows), \
                 mock.patch.object(prospective,'plan_for_record',side_effect=dispatch), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.c.run(dict(entry,constructor_provenance={}),Path('authored-unused.npz'),out)
            self.assertEqual(calls,[0,1])
            entry=dict(entry,constructor_provenance={})
            tally=owner.reconstruct(out,entry)
            self.assertEqual(tally['counts'],{'UNKNOWN':2})
            (out/'constructor-context.json').unlink()
            self.assertFalse(owner.reconstruct(out,entry)['input_freeze_valid'])
            frozen=json.loads((out/'inputs-frozen.json').read_bytes());frozen.pop('constructor_context_file_sha256')
            (out/'inputs-frozen.json').write_text(json.dumps(frozen))
            self.assertFalse(owner.reconstruct(out,entry)['input_freeze_valid'])

    def test_unresolved_generated_gender_keeps_original_slots_without_redraw(self):
        state,sources,source,entry,_=self.fixture(generated=True,trace=False)
        policy=deepcopy(entry['gender_initialization']['policy']);policy['codes'][1]=-1;policy['unresolved']=['ditto']
        context=replace(sources.context,gender_policy_json=self.s._json(policy))
        values=asdict(context);values.pop('sha256');context=replace(context,sha256=self.s._sha(values))
        sources=replace(sources,context=context)
        entry=deepcopy(entry);entry['gender_initialization']['policy']=policy
        rows=(dict(slot=0,teams=[0,1],prefix_requests=1),dict(slot=1,teams=[0,1],prefix_requests=0))
        with mock.patch.object(self.c,'source_rows',return_value=rows):
            generated=self.c.generate_inputs(sources.pool.arrays,self.table(context.metadata()),
                dict(entry,prefix_cycle=2),species_vocabulary=context.metadata()['vocabulary']['species'],
                constructors=sources)
        roots,_,actual_rows,traces,errors,_,_,originals,_,source_errors=generated
        self.assertEqual(actual_rows,rows);self.assertEqual(traces,[[],[]])
        self.assertTrue(errors.all());self.assertTrue(source_errors.all())
        self.assertTrue(all(isinstance(v,self.f.Refused) for v in originals))
        self.assertTrue((self.np.asarray(roots.gender)==-1).all())

    def test_one_unresolved_root_does_not_readdress_the_other_gender_draws(self):
        from gpu_engine.gender import initialize_genders
        state,sources,source,entry,catalog=self.fixture(generated=True,trace=False)
        metadata=sources.pool.metadata();metadata['vocabulary']['species'].append('unresolvedform')
        policy=deepcopy(entry['gender_initialization']['policy'])
        policy.update(species=metadata['vocabulary']['species'],codes=[-1,3,-1],unresolved=['unresolvedform'])
        arrays={name:self.np.array(value) for name,value in sources.pool.arrays.items()}
        arrays['species'][1,:]=2
        arrays['metadata_utf8']=self.np.frombuffer(json.dumps(metadata).encode(),self.np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'pool.npz';self.np.savez_compressed(path,**arrays)
            pool=self.s.load_constructor_pool(path)
            digest=self.s._pool(pool)['pool_sha256']
            catalog=deepcopy(catalog);catalog.update(species=metadata['vocabulary']['species'],
                weights=[0,4.0,None],unresolved=['unresolvedform'])
            catalog['provenance']['pool_sha256']=digest;policy['provenance']['pool_sha256']=digest
            context=self.s.constructor_context(pool,catalog,runtime=sources.runtime,
                cohort_id='authored-mixed-unresolved',gender_policy=policy)
            sources=replace(sources,pool=pool,context=context)
            entry=deepcopy(entry);entry['pool_sha256']=digest;entry['gender_initialization']['policy']=policy
            rows=(dict(slot=0,teams=[1,1],prefix_requests=0),dict(slot=1,teams=[0,0],prefix_requests=1))
            table=self.table(context.metadata())
            with mock.patch.object(self.c,'source_rows',return_value=rows):
                result=self.c.generate_inputs(pool.arrays,table,dict(entry,prefix_cycle=2),
                    species_vocabulary=metadata['vocabulary']['species'],constructors=sources)
            baseline=self.t.state_from_pool(pool.arrays,[[0,0]],seed=entry['seed'])
            baseline=initialize_genders(baseline,policy,metadata['vocabulary']['species'],
                seed=7,domain=13,root_ids=[43])
            self.assertEqual(result[9].tolist(),[True,False])
            self.assertEqual(self.np.asarray(result[0].gender[1]).tolist(),self.np.asarray(baseline.gender[0]).tolist())
            self.assertIsInstance(result[7][0],self.f.Refused)
            self.assertIsInstance(result[7][1],self.s.ConstructorProvenance)
            self.assertEqual(result[7][1].root_id,43)
            self.assertEqual(len(result[3][0]),0);self.assertEqual(len(result[3][1]),1)

    def test_actual_catalog_regeneration_rejects_changed_source(self):
        state,sources,source,entry,_=self.fixture(trace=False)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);showdown=root/'showdown'
            (showdown/'dist/sim').mkdir(parents=True)
            (showdown/'dist/sim/dex.js').write_text("exports.Dex={mod:()=>({species:{get:id=>({exists:id.toLowerCase()==='ditto',gen:1,name:'Ditto',weightkg:4})}})};\n")
            for name in ('sim/pokemon.ts','data/random-battles/gen3/teams.ts'):
                path=showdown/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('// authored source\n')
            for args in (['git','init','-q',str(showdown)],['git','-C',str(showdown),'add','.'],
                ['git','-C',str(showdown),'-c','user.name=Scott','-c','user.email=fixture@example.invalid','commit','-qm','authored source']):
                subprocess.run(args,check=True,capture_output=True)
            commit=subprocess.check_output(['git','-C',str(showdown),'rev-parse','HEAD'],text=True).strip()
            module=Path(self.p.__file__).parent
            tree=json.loads(subprocess.check_output(['node','-e',
                'console.log(JSON.stringify(require(process.argv[1]).compiledTree(process.argv[2])))',
                str(module/'gender_policy.cjs'),str(showdown/'dist')],text=True))
            metadata=sources.pool.metadata();metadata.update(source_commit_label=commit,compiled_tree=tree)
            arrays={**sources.pool.arrays,'metadata_utf8':self.np.frombuffer(json.dumps(metadata).encode(),self.np.uint8)}
            path=root/'pool.npz';self.np.savez_compressed(path,**arrays)
            entry=deepcopy(entry);entry['pool_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            header=root/'header.json';header.write_text(json.dumps(dict(metadata,pool_sha256=entry['pool_sha256'])))
            catalog=json.loads(subprocess.check_output(['node',str(module/'constructor_catalog.cjs'),str(showdown),str(header),commit],text=True))
            entry['constructor_provenance']=dict(schema=self.p.SCHEMA,catalog=catalog,
                runtime=asdict(sources.runtime),root_start=42,showdown_root=str(showdown))
            admitted=self.p.admit(entry,path)
            self.assertEqual(admitted.context.pool_sha256,entry['pool_sha256'])
            forged=deepcopy(entry);forged['constructor_provenance']['catalog']['weights'][1]=5
            with self.assertRaisesRegex(ValueError,'ACTUAL_CONSTRUCTOR_SOURCE_MISMATCH'):
                self.p.admit(forged,path)
            policy=json.loads(subprocess.check_output(['node',str(module/'gender_policy.cjs'),str(showdown),str(header),commit],text=True))
            generated=deepcopy(entry);generated['gender_initialization']=dict(seed=7,domain=13,root_start=42,policy=policy)
            admitted=self.p.admit(generated,path)
            self.assertIsNotNone(admitted.context.gender_policy_json)
            generated['gender_initialization']['policy']['codes'][1]=0
            with self.assertRaisesRegex(ValueError,'ACTUAL_CONSTRUCTOR_SOURCE_MISMATCH'):
                self.p.admit(generated,path)
            (showdown/'dist/sim/dex.js').write_text('exports.Dex={};\n')
            with self.assertRaises(subprocess.CalledProcessError):self.p.admit(entry,path)
