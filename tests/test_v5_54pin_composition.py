"""Host/source/fake compiler joins only; no engine/model/Native invocation."""
import ast
from contextlib import ExitStack
import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

ASSEMBLY='40228424'  # original 59-pin composition, retained as a historical control
BASE='6caf071b'  # reviewed ordinary source inherited by this branch

CONTEXTS=('direct_hit','substitute','toxic','recoil','calm_mind','swords_dance','spikes','thunder_wave','sleep_powder')


class CompositionTests(unittest.TestCase):
    def runtime(self):
        result={name:object() for name in ('request_fast','installations','encode_fast','frame_witness_compiled','catalog')}
        result.update(public=NS(causal=NS(projector=NS(device=object()))),request={'engine_table':object()},
            execution_compiled=NS(device=object()),kernel_risk_compiled=NS(device=object()),
            speed_profile=NS(device=object()),no_output_compiled=None,leftovers_context=None,status_residual_context=None,
            direct_hit_mode='kernel-event')
        result.update({name+'_context':object() for name in CONTEXTS})
        return result

    def test_exact_ordinary_inventory_plus_existing_hypothesis_and_matching_schema(self):
        from gpu_engine import engine_source_frame_witness as witness
        witness._source();self.assertEqual(len(witness.SOURCE_PINS),92)
        old=subprocess.check_output(['git','show','2b0ec76a:gpu_engine/engine_source_frame_witness.py'],text=True)
        inventory=next(ast.literal_eval(n.value) for n in ast.parse(old).body if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='SOURCE_PINS' for t in n.targets))
        historical=subprocess.check_output(['git','show',ASSEMBLY+':gpu_engine/engine_source_frame_witness.py'],text=True)
        assembled=next(ast.literal_eval(n.value) for n in ast.parse(historical).body if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='SOURCE_PINS' for t in n.targets))
        self.assertEqual(set(assembled)-set(inventory),{'v5_hypothetical_profile.py'})
        baseline=subprocess.check_output(['git','show',BASE+':gpu_engine/engine_source_frame_witness.py'],text=True)
        inherited=next(ast.literal_eval(n.value) for n in ast.parse(baseline).body if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='SOURCE_PINS' for t in n.targets))
        self.assertEqual(set(witness.SOURCE_PINS)-set(inherited),{'infatuation.py','transform.py','forecast.py'})
        self.assertEqual(set(inherited)-set(witness.SOURCE_PINS),set())
        for file,expected,historical_count in [('engine_full_v4_event_device.py',62,34),('engine_lifecycle_event_device.py',53,41),('engine_kernel_frame_risk.py',107,28)]:
            original=subprocess.check_output(['git','show','2b0ec76a:gpu_engine/'+file],text=True)
            original_table=next(n for n in ast.parse(original).body if isinstance(n,ast.ClassDef) and n.name=='Tables')
            self.assertEqual(sum(isinstance(n,ast.AnnAssign) for n in original_table.body),historical_count)
            self.assertEqual(Path('gpu_engine',file).read_bytes(),subprocess.check_output(['git','show',BASE+':gpu_engine/'+file]))
            tree=ast.parse(Path('gpu_engine',file).read_text());table=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Tables')
            self.assertEqual(sum(isinstance(n,ast.AnnAssign) for n in table.body),expected)
            for call in ast.walk(tree):
                if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='Tables':self.assertEqual(len(call.args),expected)
        for file in ['v5_hypothetical_profile.py','engine_calm_mind_callbacks.py','engine_swords_dance_callbacks.py','engine_spikes_callbacks.py']:
            parent='f99a51c3' if file=='v5_hypothetical_profile.py' else 'e5d38e3c'
            self.assertEqual(Path('gpu_engine',file).read_bytes(),subprocess.check_output(['git','show',parent+':gpu_engine/'+file]))

    def test_batched_original_kernel_seam_forwards_all_contexts_and_seals_edges(self):
        from gpu_engine import v5_hypothetical_batch as batch
        r=self.runtime();prepared,adapter=object(),object();captured={};leech=object()
        class AtKernel(RuntimeError):pass
        def frame(*args,**kwargs):captured.update(kwargs);raise AtKernel
        with ExitStack() as stack:
            stack.enter_context(patch.object(batch.driver,'_checked',return_value=r))
            stack.enter_context(patch.object(batch.driver,'_checked_witness'))
            history=stack.enter_context(patch.object(batch.driver,'_leech_history',return_value=leech))
            stack.enter_context(patch.object(batch.v5_observation,'check_adapter',return_value=r['catalog']))
            stack.enter_context(patch.object(batch.jax,'jit',side_effect=lambda function:function))
            stack.enter_context(patch.object(batch.driver,'_frame',side_effect=frame))
            executor=batch._Executor(prepared,adapter)
            core=NS(admitted=np.ones(1,bool))
            publics=(object(),object())
            with self.assertRaises(AtKernel):executor.run(core,publics,(),object(),object(),object(),np.ones(1,bool))
            history.assert_called_once_with(r,core,publics,core.admitted)
            self.assertIs(captured['leech_history'],leech)
            for name in CONTEXTS:self.assertIs(captured[name+'_context'],r[name+'_context'])
            self.assertIsInstance(captured['hypothetical_profile'],batch.hypothesis.Context)
            self.assertIs(captured['hypothetical_profile'].data,r['speed_profile'].device)
            executor.check()
            for name in CONTEXTS:
                original=r[name+'_context']
                for replacement in (None,object()):
                    r[name+'_context']=replacement
                    with self.subTest(name=name,cleared=replacement is None),self.assertRaisesRegex(ValueError,'edges changed'):executor.check()
                r[name+'_context']=original
            original=r['direct_hit_mode']
            for replacement in (None,'changed-mode'):
                r['direct_hit_mode']=replacement
                with self.assertRaisesRegex(ValueError,'edges changed'):executor.check()
            r['direct_hit_mode']=original
            executor.check()

    def test_scalar_original_kernel_seam_forwards_all_contexts_without_enabling_profiles(self):
        from gpu_engine import v5_public_branch as branch
        r=self.runtime();captured={};leech=object();opponent=object();root=NS(prepared=object(),public=object(),history=NS(owner=0))
        current=NS(requests=(NS(actions=object()),NS(actions=object())),orders=(object(),object()),cursor=object(),admitted=np.ones(1,bool))
        class AtKernel(RuntimeError):pass
        def frame(*args,**kwargs):captured.update(kwargs);raise AtKernel
        with ExitStack() as stack:
            stack.enter_context(patch.object(branch.driver,'_checked',return_value=r))
            stack.enter_context(patch.object(branch,'core',return_value=current))
            stack.enter_context(patch.object(branch.actions,'route_choice',return_value=NS(admitted=np.ones(1,bool),engine_action=np.zeros(1,np.int32))))
            stack.enter_context(patch.object(branch,'_opponent_public',return_value=opponent))
            history=stack.enter_context(patch.object(branch.driver,'_leech_history',return_value=leech))
            stack.enter_context(patch.object(branch,'_stream',return_value=object()))
            stack.enter_context(patch.object(branch.driver,'_frame',side_effect=frame))
            with self.assertRaises(AtKernel):branch.step(root,(0,0),chance_inputs=object())
            history.assert_called_once_with(r,current,(root.public,opponent),current.admitted)
            self.assertIs(captured['leech_history'],leech)
        for name in CONTEXTS:self.assertIs(captured[name+'_context'],r[name+'_context'])
        self.assertIsNone(captured['hypothetical_profile'])


    def test_whole_ordinary_driver_and_exact_private_adapter_extension(self):
        class Driver(ast.NodeTransformer):
            def visit_FunctionDef(self,n):
                if n.name=='_frame':
                    pairs=list(zip(n.args.kwonlyargs,n.args.kw_defaults))
                    extra=[d for a,d in pairs if a.arg=='hypothetical_profile']
                    if extra:self_case.assertEqual(ast.dump(extra[0]),'Constant(value=None)')
                    pairs=[(a,d)for a,d in pairs if a.arg!='hypothetical_profile']
                    n.args.kwonlyargs=[a for a,d in pairs];n.args.kw_defaults=[d for a,d in pairs]
                return self.generic_visit(n)
            def visit_Call(self,n):
                n=self.generic_visit(n)
                for k in n.keywords:
                    if k.arg=='_hypothetical_profile':self_case.assertEqual(ast.unparse(k.value),'hypothetical_profile')
                n.keywords=[k for k in n.keywords if k.arg!='_hypothetical_profile'];return n
            def visit_BoolOp(self,n):
                if ast.unparse(n)=='weather_profile_data is None and hypothetical_profile is None':return n.values[0]
                return self.generic_visit(n)
        self_case=self
        old=subprocess.check_output(['git','show','2b0ec76a:gpu_engine/engine_retained_encoder_driver.py'],text=True)
        new=subprocess.check_output(['git','show',ASSEMBLY+':gpu_engine/engine_retained_encoder_driver.py'],text=True)
        normalize=lambda s:ast.dump(Driver().visit(ast.parse(s)),include_attributes=False)
        self.assertEqual(normalize(old),normalize(new))
        original='engine = _rows(source_ready, core.engine, result.step.state)'
        self.assertEqual(new.count(original),1)
        changed=new.replace(original,'engine = _rows(~source_ready, core.engine, result.step.state)')
        self.assertNotEqual(normalize(old),normalize(changed))
        class Adapter(ast.NodeTransformer):
            def visit_Call(self,n):
                n=self.generic_visit(n)
                for k in n.keywords:
                    if k.arg=='sleep_powder_context':self_case.assertIn(ast.unparse(k.value),("r['sleep_powder_context']","runtime['sleep_powder_context']"))
                n.keywords=[k for k in n.keywords if k.arg!='sleep_powder_context'];return n
            def visit_Assign(self,n):
                if any(isinstance(t,ast.Name)and t.id=='names'for t in n.targets)and isinstance(n.value,ast.Tuple):
                    n.value.elts=[x for x in n.value.elts if not(isinstance(x,ast.Constant)and x.value=='sleep_powder_context')]
                return self.generic_visit(n)
        for file in ('v5_public_branch.py','v5_hypothetical_batch.py'):
            old=subprocess.check_output(['git','show','dc981dd8:gpu_engine/'+file],text=True)
            new=subprocess.check_output(['git','show',ASSEMBLY+':gpu_engine/'+file],text=True)
            normalize=lambda s:ast.dump(Adapter().visit(ast.parse(s)),include_attributes=False)
            self.assertEqual(normalize(old),normalize(new))


    def test_current_assembly_preserves_the_reviewed_driver_and_only_extends_chance_width(self):
        from gpu_engine import transition
        self.assertEqual(transition.EFFECT_BOUNDS,(16,16,4,2,2))
        for file in ('engine_retained_encoder_driver.py','v5_public_branch.py'):
            self.assertEqual(Path('gpu_engine',file).read_bytes(),
                subprocess.check_output(['git','show',BASE+':gpu_engine/'+file]))
        old=subprocess.check_output(['git','show',BASE+':gpu_engine/v5_hypothetical_batch.py'],text=True)
        functions=[n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef) and n.name=='_chance']
        self.assertEqual(len(functions),1)
        function=functions[0];lines=old.splitlines(True)
        chance=''.join(lines[function.lineno-1:function.end_lineno])
        replacements=(('(1,2,4)','(1,2,len(transition.EFFECT_BOUNDS))',2),
            ('np.array([16,16,4,2])','np.array(transition.EFFECT_BOUNDS)',1))
        for before,after,count in replacements:
            self.assertEqual(chance.count(before),count)
            chance=chance.replace(before,after)
        lines[function.lineno-1:function.end_lineno]=[chance]
        expected=''.join(lines)
        current=Path('gpu_engine/v5_hypothetical_batch.py').read_text()
        shape=lambda text:ast.dump(ast.parse(text),include_attributes=False)
        self.assertEqual(shape(expected),shape(current))
        for before,after,count in (
                ('(1,2,len(transition.EFFECT_BOUNDS))','(1,2,4)',2),
                ('np.array(transition.EFFECT_BOUNDS)','np.array(transition.EFFECT_BOUNDS[:-1])',1),
                ('np.asarray(x)>=limit','np.asarray(x)>limit',1),
                ("direct_hit_context=r['direct_hit_context']",'direct_hit_context=None',1)):
            self.assertEqual(current.count(before),count)
            changed=current.replace(before,after)
            self.assertNotEqual(shape(expected),shape(changed))


    def test_forecast_keeps_complete_integrated_private_sources(self):
        normalize=lambda s:ast.dump(ast.parse(s),include_attributes=False)
        for file in ('engine_retained_encoder_driver.py','v5_public_branch.py','v5_hypothetical_batch.py'):
            old=subprocess.check_output(['git','show','766833fc8e6f40d2ffa80aa95a19fd940a519f33:gpu_engine/'+file],text=True)
            self.assertEqual(normalize(old),normalize(Path('gpu_engine',file).read_text()))

if __name__=='__main__':unittest.main()
