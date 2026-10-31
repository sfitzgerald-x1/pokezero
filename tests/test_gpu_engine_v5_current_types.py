"""Source/device type-proof joins only; no worlds, kernels, models or games."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import jax.numpy as j
import numpy as np

from gpu_engine import v5_public_branch as branch
from gpu_engine import v5_hypothetical_batch as batch
from gpu_engine import engine_lifecycle_callbacks as life
from gpu_engine import engine_color_change_callbacks as color
import test_gpu_engine_color_change_owning as fixtures


class CurrentTypesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ColorChangeOwningTests.setUpClass()
        cls.fixture=fixtures.ColorChangeOwningTests()
        cls.before,cls.after,cls.tape,_,cls.consumers,cls.fold=cls.fixture.component()

    def join(self,*,tape=None,engine=None,before=None,publics=None,healthy=None,runtime=None):
        tape=self.tape if tape is None else tape
        engine=self.after if engine is None else engine
        before=self.before if before is None else before
        publics=tuple(x[1] for x in self.consumers) if publics is None else publics
        health=j.ones(8,bool) if healthy is None else healthy
        prior=NS(engine=before,admitted=health,originals=(object(),object()),
            constructors=(object(),object()),unsafe=(object(),object()))
        arrival=object();frame=NS(state=engine,callbacks=tape,after_registry=arrival)
        orders=(object(),object());admitted=j.array([True]*5+[False]*3)
        calls=[]
        def request(*args):
            calls.append(args)
            return args[-1]
        r=dict(self.fixture.runtime if runtime is None else runtime,request_fast=request)
        result=branch._requests_after_frame(r,frame,prior,publics,orders,admitted)
        for owner,args in enumerate(calls):
            for got,want in zip(args[:7],(engine,orders[owner],prior.originals[owner],
                    prior.constructors[owner],prior.unsafe[owner],arrival,admitted),strict=True):
                self.assertIs(got,want)
        self.assertEqual(len(calls),2)
        return result

    def test_real_receipt_both_owners_b8_keeps_exact_original_evidence(self):
        values=self.join()
        for owner,value in enumerate(values):
            expected=color.expected_owned_types(self.after,self.before,self.consumers[owner][1],
                owner=owner,tape=self.tape,data=self.fixture.helper.device.color_change,
                categories=self.fixture.runtime['public'].device.causal.color_categories,
                type_tables=self.fixture.catalog.types.device,healthy=j.ones(8,bool))
            np.testing.assert_array_equal(value,expected)
            np.testing.assert_array_equal(value[:5],self.after.types[:5,owner])
            self.assertEqual(value.shape,(8,6,2))
        # The original default source check would compare changed current
        # types with sealed base types and fail on each actually changed seat.
        rows=np.arange(5);targets=1-rows%2;physical=np.asarray(self.after.active)[rows,targets]
        self.assertTrue(np.all(np.asarray(self.after.types)[rows,targets,physical,0]!=
            np.asarray(self.after.base_types)[rows,targets,physical,0]))

    def test_foreign_stale_and_non_color_receipts_never_authorize_changed_private_types(self):
        foreign=self.tape._replace(move_binding=self.tape.move_binding.at[0].add(j.uint32(1)))
        stale=self.tape._replace(event_turn=self.tape.event_turn.at[:,0].add(1))
        words=self.tape.words.at[:,2,9].set(self.fixture.pressure)
        ledger=self.tape.execution.ledger
        wrong=self.tape._replace(words=words,execution=self.tape.execution._replace(
            ledger=ledger._replace(words=ledger.words.at[:,2,9].set(self.fixture.pressure))))
        for name,tape in (('foreign',foreign),('stale-prefix',stale),('non-color',wrong)):
            with self.subTest(name=name):
                values=self.join(tape=tape)
                for row in range(5):
                    owner=1-row%2;slot=int(self.after.active[row,owner])
                    self.assertEqual(int(values[owner][row,slot,0]),self.fixture.normal)
                    self.assertNotEqual(int(values[owner][row,slot,0]),int(self.after.types[row,owner,slot,0]))

    def test_existing_prior_public_override_needs_healthy_current_encounter(self):
        frame=life.LifecycleResult(self.after,self.tape.cursor,self.tape,
            life.snapshot(self.tape.cursor),life.snapshot(self.tape.cursor),j.zeros(8,bool),j.zeros(8,j.int32))
        publics=tuple(type(self).fold(stream,frame,public).public for stream,public in self.consumers)
        empty=self.tape._replace(count=j.zeros(8,j.int32))
        values=self.join(tape=empty,before=self.after,publics=publics)
        for owner,value in enumerate(values):
            np.testing.assert_array_equal(value[:5],self.after.types[:5,owner])
        unhealthy=self.join(tape=empty,before=self.after,publics=publics,healthy=j.zeros(8,bool))
        corrupt=tuple(p._replace(context=p.context._replace(types=p.context.types._replace(
            encounter=p.context.types.encounter+1))) for p in publics)
        stale=self.join(tape=empty,before=self.after,publics=corrupt)
        for result in (unhealthy,stale):
            for row in range(5):
                owner=1-row%2;slot=int(self.after.active[row,owner])
                self.assertEqual(result[owner][row,slot].tolist(),[-1,-1])

    def test_disabled_context_and_mode_retain_closed_none_default(self):
        for update in ({'direct_hit_context':None},{'direct_hit_mode':'closed'},
                {'direct_hit_context':NS(data=NS(color_change=None))}):
            with self.subTest(update=update):
                self.assertEqual(self.join(runtime=dict(self.fixture.runtime,**update)),(None,None))

    def test_scalar_and_batch_use_same_pre_fold_join_and_original_root_stays_closed(self):
        root=Path(branch.__file__).parent
        for filename,expected in (('v5_public_branch.py','_requests_after_frame(runtime, frame, current, publics, ledgers, good)'),
                ('v5_hypothetical_batch.py','branch._requests_after_frame(r, frame, core, publics, orders, good)')):
            tree=ast.parse((root/filename).read_text())
            calls=[ast.unparse(x) for x in ast.walk(tree) if isinstance(x,ast.Call)]
            self.assertIn(expected,calls)
        # No broadened historical restoration/attestation: unchanged core()
        # retains its original source-bound current-root refusal behavior.
        import subprocess
        old=ast.parse(subprocess.check_output(['git','show','5461516a5e80b4c97ff9ea4f7c7ac8a536457a82:gpu_engine/v5_public_branch.py'],text=True))
        tree=ast.parse((root/'v5_public_branch.py').read_text())
        get=lambda t:next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='core')
        self.assertEqual(ast.dump(get(old)),ast.dump(get(tree)))


class ModeBindingTests(unittest.TestCase):
    def test_cached_executor_checks_original_mode_edge_before_reuse(self):
        # Reuse the literal production edge snapshot, without constructing an
        # engine, public-world root, model, or executor compiler graph.
        tree=ast.parse(Path(batch.__file__).read_text())
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='_Executor')
        init=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='__init__')
        names=next(ast.literal_eval(x.value) for x in init.body if isinstance(x,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='names' for t in x.targets))
        self.assertIn('direct_hit_mode',names)
        runtime={name:object() for name in names};runtime['catalog']=object()
        runtime['direct_hit_mode']='kernel-event'
        value=batch._Executor.__new__(batch._Executor)
        value.prepared=object();value.adapter=object();value.runtime=runtime
        value.route=object();value.run=object()
        value.edges=tuple((name,runtime[name]) for name in names)
        batch._REGISTERED[value]=(value.prepared,value.adapter,runtime,value.route,value.run,value.edges)
        with patch.object(batch.driver,'_checked',return_value=runtime), \
                patch.object(batch.driver,'_checked_witness'), \
                patch.object(batch.v5_observation,'check_adapter',return_value=runtime['catalog']):
            value.check()
            original=runtime['direct_hit_mode'];runtime['direct_hit_mode']='closed'
            with self.assertRaisesRegex(ValueError,'static hypothetical executor edges changed'):
                value.check()
            runtime['direct_hit_mode']=original
            value.check()


if __name__=='__main__':unittest.main()
