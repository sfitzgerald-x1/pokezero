"""Authored protocol/integration controls; no scientific roster or replay."""
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
import contextlib
import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock


def protocol(entry):
    from gpu_engine import ordinary_panel as p
    return dict(entry,ordinary_carry=dict(schema=p.SCHEMA,criterion=p.CRITERION,max_output_nodes=100000))


class ProtocolTests(unittest.TestCase):
    def test_new_method_requires_strict_protocol_before_source_generation(self):
        from gpu_engine import ordinary_panel as p,collect_fidelity_panel as collector
        valid=protocol(dict(constructor_provenance={}))
        self.assertIsNone(p.contract({}))
        for cfg in (None,False,{},dict(valid['ordinary_carry'],max_output_nodes=True),
                    dict(valid['ordinary_carry'],max_output_nodes=0),dict(valid['ordinary_carry'],criterion='tolerance'),
                    dict(valid['ordinary_carry'],new_knob=1)):
            with self.subTest(cfg=cfg),mock.patch.object(collector,'runtime_admission',side_effect=AssertionError('admission')):
                with self.assertRaises(ValueError):collector.run(dict(valid,ordinary_carry=cfg),Path('unused'),Path('unused'))
        with self.assertRaises(ValueError):p.contract(protocol({}))

    def test_malformed_certificate_is_a_whole_unknown_slot_during_reconstruction(self):
        from gpu_engine import panel_owner as owner
        from tests.test_gpu_engine_panel_owner import OwnerTests,ENTRY
        helper=OwnerTests()
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);frozen=helper.inputs(out)
            invalid=dict(slot=0,input_jsonl_sha256=frozen['input_jsonl_sha256'],status='AGREE',
                identified_mass='1',unknown_mass='0',consumer_certificate=[],
                outcomes=[dict(key=[['hp',1]],jax_probability='1',native_probability=1.)])
            with gzip.open(out/'slot-0000.json.gz','wt') as stream:json.dump(invalid,stream)
            result=owner.reconstruct(out,ENTRY)
        self.assertEqual(result['counts'],dict(UNKNOWN=2))
        self.assertEqual(result['valid_completed_slots'],0)
        self.assertEqual(result['incomplete_slots']['0'],'INCOMPLETE_OR_INVALID_SLOT: ValueError')



AVAILABLE=all(importlib.util.find_spec(n) for n in ('jax','numpy','poke_engine','pokezero_search'))
if AVAILABLE:
    import pokezero_search
    AVAILABLE=hasattr(pokezero_search,'conditioned_step')


@unittest.skipUnless(AVAILABLE,'isolated Native98 full-PP/conditioned bindings required')
class OrdinaryPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_prospective_fidelity_panel import ProspectivePanelTests
        ProspectivePanelTests.setUpClass()
        cls.Fixture=ProspectivePanelTests

    def fixture(self):
        f=self.Fixture('test_all_inputs_and_context_close_before_any_plan');self.addCleanup(f.doCleanups)
        state,sources,original,entry,catalog=f.fixture(trace=False)
        table=f.t.tables_from_metadata(sources.context.metadata());initial=f.t.initialize(state,table)
        entry=protocol(dict(entry,constructor_provenance=dict(schema=f.p.SCHEMA,catalog=catalog,
            runtime=asdict(sources.runtime),root_start=sources.root_start,showdown_root='authored-unused')))
        record=f.record(initial,initial,original,sources,(0,0));record.update(source_error=False,terminal=False)
        return f,initial,sources,original,entry,table,record

    def planned(self):
        from gpu_engine import ordinary_panel as p
        f,state,sources,original,entry,table,record=self.fixture()
        planned=f.plan(state,state,original,sources,entry,table,record)
        self.assertIsInstance(planned,p.Dispatch)
        self.assertEqual(planned.admission,p.dispatch('ordinary_carry'))
        return f,state,sources,entry,table,record,planned

    def test_retained_constructor_dispatch_and_paired_full_laws_are_auditable(self):
        from gpu_engine import ordinary_law as law,ordinary_panel as p,panel_owner as owner
        from gpu_engine.collect_fidelity_panel import encoded
        f,state,sources,entry,table,record,planned=self.planned()
        result=law.compare_plan(planned.plan,state,(0,0),table,cap=entry['max_cases'],batch_size=4,
            max_nodes=100000,retain_evidence=True)
        self.assertEqual(result['status'],'AGREE',result)
        result.update(slot=0,input_jsonl_sha256='frozen',root_dispatch=planned.admission,
            consumer_certificate=p.consumer_certificate(planned.plan))
        saved=json.loads(encoded(result))
        self.assertEqual(owner.validate_slot(saved,entry,'frozen',0,root_record=record,
            constructor_context=sources.frozen_context()),'AGREE')
        self.assertEqual(len(saved['ordinary_output_laws']['emissions']),planned.plan.law.cardinality)
        self.assertTrue(any(e['jax']['capsule']['pending_residual'] for e in saved['ordinary_output_laws']['emissions']))
        return saved,planned.plan,entry

    def test_coverage_alias_mass_space_criterion_and_nonfinite_corruption_refuse(self):
        from gpu_engine import ordinary_law as law,ordinary_panel as p
        from gpu_engine.collect_fidelity_panel import encoded
        f,state,sources,entry,table,record,planned=self.planned()
        result=law.compare_plan(planned.plan,state,(0,0),table,cap=4096,batch_size=4,max_nodes=100000,retain_evidence=True)
        evidence=json.loads(encoded(result['ordinary_output_laws']))
        def omitted(v):v['emissions'].pop()
        def duplicated(v):v['emissions'][-1]=deepcopy(v['emissions'][0])
        def changed_mass(v):v['emissions'][0]['mass']='1'
        def changed_domain(v):v['axes'][0]['observations']=[]
        def changed_space(v):v['space'][0][1]=True
        def changed_criterion(v):v['criterion']='legacy-per-key-tolerance'
        def changed_copy(v):v['emissions'][0]['native']['expressions'][0]={'name':'residual_draws.0.0','copy':'unbound'}
        def changed_constant(v):v['emissions'][0]['native']['expressions'][0]={'name':'residual_draws.0.0','constant':True}
        def nonfinite(v):v['emissions'][0]['native']['physical'][0][1]=float('nan')
        def projection(v):
            for side in ('jax','native'):v['emissions'][0][side]['physical'].pop()
        def duplicate_physical(v):
            for side in ('jax','native'):v['emissions'][0][side]['physical'][-1]=v['emissions'][0][side]['physical'][0]
        def changed_keys(v):
            for side in ('jax','native'):v['emissions'][0][side]['capsule']['keys'][0]^=1
        def changed_turn(v):
            for side in ('jax','native'):v['emissions'][0][side]['capsule']['turn']+=1
        def changed_replacement(v):
            for side in ('jax','native'):v['emissions'][0][side]['capsule']['replacement_count']=1
        for change in (omitted,duplicated,changed_mass,changed_domain,changed_space,changed_criterion,changed_copy,changed_constant,nonfinite,projection,duplicate_physical,changed_keys,changed_turn,changed_replacement):
            corrupt=deepcopy(evidence);change(corrupt)
            with self.subTest(change=change.__name__),self.assertRaises(ValueError):p.audit_evidence(corrupt,planned.plan,entry)

    def test_root_preflight_can_choose_legacy_but_caps_and_drift_never_do(self):
        from gpu_engine import ordinary_panel as p,ordinary_law as law,chance_consumers
        f,state,sources,original,entry,table,record=self.fixture()
        current=f.s.native_input_with_provenance(state,0,sources.context.metadata(),provenance=original,
            context=sources.context,runtime=sources.runtime,expected_root_id=42,expected_pool_sha256=entry['pool_sha256'])
        with mock.patch.object(p.ordinary,'_native_requirements',side_effect=ValueError('STEP_UNSUPPORTED_SELECTED_MOVE')), \
             mock.patch.object(chance_consumers,'plan_for_state',return_value='legacy') as legacy, \
             mock.patch.object(law,'plan_for_state',side_effect=AssertionError('new execution')):
            result=p.route_current(state,(0,0),table,sources.context.metadata(),current,entry)
            self.assertEqual(result.admission['route'],'legacy');self.assertEqual(legacy.call_count,1)
        for message in ('native execution failure','STEP_UNSUPPORTED_SELECTED_MOVE: extra','ORDINARY_NATIVE_BUILD_NOT_CURRENT'):
            with mock.patch.object(p.ordinary,'_native_requirements',side_effect=ValueError(message)), \
                 mock.patch.object(chance_consumers,'plan_for_state',side_effect=AssertionError('fallback')):
                with self.assertRaises(ValueError):p.route_current(state,(0,0),table,sources.context.metadata(),current,entry)
        with mock.patch.object(chance_consumers,'plan_for_state',wraps=chance_consumers.plan_for_state) as calls:
            small=dict(entry,max_cases=1)
            # The supported native root stays on the admitted path even when
            # the JAX consumer cap refuses it; no legacy plan is attempted.
            actual=f.plan(state,state,original,sources,small,table,record)
            self.assertEqual(actual.admission,p.dispatch('ordinary_carry'))
            self.assertIsInstance(actual.plan,f.f.Refused)
            self.assertTrue(all(call.kwargs.get('exact_move_slots') is True for call in calls.call_args_list))
        actual_plan=law.plan_for_state(state,(0,0),table,sources.context.metadata(),cap=4096,native_input=current)
        used={axis.coordinate for axis in actual_plan.law.axes if axis.observations}
        inactive=next(name for name,_ in __import__('gpu_engine.finite_law',fromlist=['COORDINATES']).COORDINATES if name not in used)
        requirements=law._native_requirements(current,(0,0))
        extra=[*requirements,dict(coordinate=inactive,kind='identity',argument=0)]
        cap=dict(entry,max_cases=actual_plan.law.cardinality)
        with mock.patch.object(law,'_native_requirements',return_value=extra):
            refused=p.route_current(state,(0,0),table,sources.context.metadata(),current,cap)
        self.assertEqual(refused.admission,p.dispatch('ordinary_carry'))
        self.assertIsInstance(refused.plan,f.f.Refused)
        self.assertIn('CHANCE_CARDINALITY_CAP',repr(refused.plan))

    def test_late_positive_output_refusal_is_whole_slot_without_partial_evidence(self):
        from gpu_engine import ordinary_law as law
        f,state,sources,entry,table,record,planned=self.planned()
        actual=law._native_output;calls=[]
        def late(*args):
            calls.append(True)
            if len(calls)==2:raise law.WholeSlotUnknown('AUTHORED_LATE_OUTPUT',dict(offending='kept'))
            return actual(*args)
        with mock.patch.object(law,'_native_output',side_effect=late):
            result=law.compare_plan(planned.plan,state,(0,0),table,cap=4096,batch_size=4,
                max_nodes=100000,retain_evidence=True)
        self.assertEqual((result['status'],result['identified_mass'],result['unknown_mass']),('UNKNOWN','0','1'))
        self.assertEqual(result['completed_cases'],1)
        self.assertEqual(result['detail'],dict(offending='kept'))
        self.assertNotIn('ordinary_output_laws',result)

    def test_cells_are_precounted_but_not_materialized_before_executor_creation(self):
        from gpu_engine import ordinary_law as law
        f,state,sources,entry,table,record,planned=self.planned()
        cells=law.algebra.execution_cells;executor=law.JaxCaseExecutor;produced=[]
        def tracked(*args,**kwargs):
            admitted=cells(*args,**kwargs)  # The cap must reject before yielding.
            def iterate():
                for cell in admitted:
                    produced.append(cell)
                    yield cell
            return iterate()
        def create_executor(*args,**kwargs):
            self.assertEqual(produced,[])
            return executor(*args,**kwargs)
        with mock.patch.object(law.algebra,'execution_cells',side_effect=tracked), \
             mock.patch.object(law,'JaxCaseExecutor',side_effect=create_executor) as execute:
            refused=law.compare_plan(planned.plan,state,(0,0),table,
                cap=planned.plan.law.cardinality-1,batch_size=1,max_nodes=100000)
            self.assertEqual(refused['status'],'UNKNOWN')
            self.assertFalse(produced)
            execute.assert_not_called()
            result=law.compare_plan(planned.plan,state,(0,0),table,
                cap=planned.plan.law.cardinality,batch_size=1,max_nodes=100000,retain_evidence=True)
        self.assertEqual(result['status'],'AGREE',result)
        self.assertEqual(len(produced),planned.plan.law.cardinality)
        self.assertEqual(len(result['ordinary_output_laws']['emissions']),len(produced))

    def test_collector_freezes_and_persists_both_full_laws_before_terminal_audit(self):
        from gpu_engine import ordinary_panel as p,prospective_panel,panel_owner as owner
        f,state,sources,original,entry,table,record=self.fixture()
        rows=tuple(dict(slot=i,teams=[0,1],prefix_requests=0) for i in range(2))
        actual=prospective_panel.plan_for_record;called=[]
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'panel'
            def planned(*args,**kwargs):
                self.assertTrue((out/'inputs-frozen.json').is_file())
                self.assertTrue((out/'constructor-context.json').is_file())
                called.append(True);return actual(*args,**kwargs)
            actual_inputs=f.c.generate_inputs
            def inputs(*args,**kwargs):
                def choose(state,table,*,salt):
                    b=state.hp.shape[0]
                    return f.j.zeros((b,2),f.j.int32),f.j.zeros((b,),bool),f.j.ones((b,2),bool)
                return actual_inputs(*args,choose=choose,**kwargs)
            with mock.patch.object(f.c,'runtime_admission',return_value={}), \
                 mock.patch.object(f.c,'generate_inputs',side_effect=inputs), \
                 mock.patch('gpu_engine.benchmark.sole_gpu',return_value=type('Device',(),{'device_kind':'authored CPU'})()), \
                 mock.patch.object(prospective_panel,'admit',return_value=sources), \
                 mock.patch.object(f.c,'source_rows',return_value=rows), \
                 mock.patch.object(prospective_panel,'plan_for_record',side_effect=planned), \
                 contextlib.redirect_stdout(io.StringIO()):
                f.c.run(entry,Path('unused-authored.npz'),out)
                result=owner.reconstruct(out,entry)
            self.assertEqual(len(called),2)
            self.assertEqual(result['counts'],dict(AGREE=2),result)
            for i in range(2):
                with gzip.open(out/f'slot-{i:04d}.json.gz','rt') as stream:saved=json.load(stream)
                self.assertEqual(saved['root_dispatch'],p.dispatch('ordinary_carry'))
                self.assertIn('ordinary_output_laws',saved)

    def test_owner_binds_method_and_complete_certificate_to_original_root(self):
        from gpu_engine import ordinary_law as law,ordinary_panel as p,panel_owner as owner
        from gpu_engine.collect_fidelity_panel import encoded
        f,state,sources,entry,table,record,planned=self.planned()
        result=law.compare_plan(planned.plan,state,(0,0),table,cap=4096,batch_size=4,max_nodes=100000,retain_evidence=True)
        result.update(slot=0,input_jsonl_sha256='frozen',root_dispatch=planned.admission,
            consumer_certificate=p.consumer_certificate(planned.plan))
        saved=json.loads(encoded(result))
        for field in ('kind','source','table','axes','cardinality','native_input','native_actions','address'):
            corrupted=deepcopy(saved);corrupted['consumer_certificate'].pop(field)
            with self.subTest(field=field),self.assertRaises(ValueError):
                owner.validate_slot(corrupted,entry,'frozen',0,root_record=record,constructor_context=sources.frozen_context())
        for kind in ('missing_certificate','malformed_certificate','legacy_route','unregistered','partial_unknown','changed_context'):
            corrupted=deepcopy(saved);cfg=deepcopy(entry);context=sources.frozen_context()
            if kind=='missing_certificate':corrupted.pop('consumer_certificate')
            if kind=='malformed_certificate':corrupted['consumer_certificate']=[]
            if kind=='legacy_route':corrupted['root_dispatch']=p.dispatch('legacy')
            if kind=='unregistered':cfg.pop('ordinary_carry')
            if kind=='partial_unknown':corrupted.update(status='UNKNOWN',identified_mass='0',unknown_mass='1',category='authored')
            if kind=='changed_context':context['root_start']+=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                owner.validate_slot(corrupted,cfg,'frozen',0,root_record=record,constructor_context=context)

    def test_cost_clock_persists_complete_ordinary_route_for_twenty_authored_roots(self):
        from gpu_engine import prospective_panel,panel_owner as owner,panel_cost_runtime as runtime
        from tests.test_gpu_engine_panel_cost import entry as cost_entry
        f,state,sources,original,entry,table,record=self.fixture();entry.update(cost_entry());entry['prefix_cycle']=1
        rows=tuple(dict(slot=i,teams=[0,1],prefix_requests=0) for i in range(20));actual=f.c.generate_cost_inputs
        attempt='c'*64;start=time.monotonic()
        def inputs(*args,**kwargs):
            def choose(state,table,*,salt):
                b=state.hp.shape[0]
                return f.j.zeros((b,2),f.j.int32),f.j.zeros((b,),bool),f.j.ones((b,2),bool)
            return actual(*args,choose=choose,**kwargs)
        with tempfile.TemporaryDirectory() as directory,mock.patch.dict(os.environ,{runtime.ENV_ATTEMPT:attempt}):
            out=Path(directory)/'panel'
            with mock.patch.object(f.c,'runtime_admission',return_value={}), \
                 mock.patch('gpu_engine.benchmark.sole_gpu',return_value=type('Device',(),{'device_kind':'authored CPU'})()), \
                 mock.patch.object(prospective_panel,'admit',return_value=sources), \
                 mock.patch.object(f.c,'cost_source_rows',return_value=rows), \
                 mock.patch.object(f.c,'generate_cost_inputs',side_effect=inputs), \
                 mock.patch.object(runtime,'notify_start'),mock.patch.object(runtime,'notification_fd',return_value=-1), \
                 contextlib.redirect_stdout(io.StringIO()):
                f.c.run_cost(entry,Path('unused-authored.npz'),out)
                physical=owner.reconstruct(out,entry)
                timing=runtime.Watch(entry,out,attempt,os.getpid(),start).reconstruct(physical['input_freeze_valid'])
            self.assertEqual(physical['counts'],dict(AGREE=20),physical)
            self.assertTrue(timing['timing_evidence_valid'],timing['timing_issues'])
            self.assertEqual(timing['latency']['exercised_paths'],dict(ordinary_carry=20))
            self.assertTrue(all(v['completed_cases']==v['planned_cases']>0 for v in timing['execution_evidence'].values()))


if __name__=='__main__':unittest.main()
