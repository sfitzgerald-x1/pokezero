"""Host/source/fake compiler joins only; no engine/model/Native invocation."""
import ast
from contextlib import ExitStack
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

CONTEXTS=('direct_hit','substitute','toxic','recoil','calm_mind','swords_dance','spikes','thunder_wave','sleep_powder')
ORDINARY_SOURCE='237eb88f'


class CompositionTests(unittest.TestCase):
    def runtime(self):
        result={name:object() for name in ('request_fast','installations','encode_fast','frame_witness_compiled','catalog')}
        result.update(public=NS(causal=NS(projector=NS(device=object()))),request={'engine_table':object()},
            execution_compiled=NS(device=object()),kernel_risk_compiled=NS(device=object()),
            speed_profile=NS(device=object()),no_output_compiled=None,leftovers_context=None,status_residual_context=None)
        result.update({name+'_context':object() for name in CONTEXTS})
        result.update(direct_hit_context=NS(data=NS(leech_seed=None)),direct_hit_mode='kernel-event')
        return result

    def test_exact_ordinary_inventory_plus_forecast_and_matching_schema(self):
        from gpu_engine import engine_source_frame_witness as witness
        witness._source();self.assertEqual(len(witness.SOURCE_PINS),90)
        old=subprocess.check_output(['git','show',ORDINARY_SOURCE+':gpu_engine/engine_source_frame_witness.py'],text=True)
        inventory=next(ast.literal_eval(n.value) for n in ast.parse(old).body if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='SOURCE_PINS' for t in n.targets))
        self.assertEqual(set(witness.SOURCE_PINS)-set(inventory),{'forecast.py'})
        self.assertFalse(set(inventory)-set(witness.SOURCE_PINS))
        self.assertIn('v5_hypothetical_profile.py',witness.SOURCE_PINS)
        for file,expected in [('engine_full_v4_event_device.py',62),('engine_lifecycle_event_device.py',53),('engine_kernel_frame_risk.py',107)]:
            tree=ast.parse(Path('gpu_engine',file).read_text());table=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Tables')
            self.assertEqual(sum(isinstance(n,ast.AnnAssign) for n in table.body),expected)
            for call in ast.walk(tree):
                if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='Tables':self.assertEqual(len(call.args),expected)
        for file in ['v5_hypothetical_profile.py','engine_calm_mind_callbacks.py','engine_swords_dance_callbacks.py','engine_spikes_callbacks.py']:
            self.assertEqual(Path('gpu_engine',file).read_bytes(),subprocess.check_output(['git','show',ORDINARY_SOURCE+':gpu_engine/'+file]))

    def test_batched_original_kernel_seam_forwards_all_contexts_and_seals_edges(self):
        from gpu_engine import v5_hypothetical_batch as batch
        r=self.runtime();prepared,adapter=object(),object();captured={}
        class AtKernel(RuntimeError):pass
        def frame(*args,**kwargs):captured.update(kwargs);raise AtKernel
        with ExitStack() as stack:
            stack.enter_context(patch.object(batch.driver,'_checked',return_value=r))
            stack.enter_context(patch.object(batch.driver,'_checked_witness'))
            stack.enter_context(patch.object(batch.v5_observation,'check_adapter',return_value=r['catalog']))
            stack.enter_context(patch.object(batch.jax,'jit',side_effect=lambda function:function))
            stack.enter_context(patch.object(batch.driver,'_frame',side_effect=frame))
            executor=batch._Executor(prepared,adapter)
            with self.assertRaises(AtKernel):executor.run(NS(admitted=np.ones(1,bool)),(),(),object(),object(),object(),np.ones(1,bool))
            for name in CONTEXTS:self.assertIs(captured[name+'_context'],r[name+'_context'])
            self.assertIsInstance(captured['hypothetical_profile'],batch.hypothesis.Context)
            self.assertIs(captured['hypothetical_profile'].data,r['speed_profile'].device)
            self.assertIsNone(captured['leech_history'])
            executor.check()
            for name in CONTEXTS:
                original=r[name+'_context']
                for replacement in (None,object()):
                    r[name+'_context']=replacement
                    with self.subTest(name=name,cleared=replacement is None),self.assertRaisesRegex(ValueError,'edges changed'):executor.check()
                r[name+'_context']=original
            executor.check()

    def test_scalar_original_kernel_seam_forwards_all_contexts_without_enabling_profiles(self):
        from gpu_engine import v5_public_branch as branch
        r=self.runtime();captured={};root=NS(prepared=object(),public=object(),history=NS(owner=0))
        current=NS(requests=(NS(actions=object()),NS(actions=object())),orders=(object(),object()),cursor=object(),admitted=np.ones(1,bool))
        class AtKernel(RuntimeError):pass
        def frame(*args,**kwargs):captured.update(kwargs);raise AtKernel
        with ExitStack() as stack:
            stack.enter_context(patch.object(branch.driver,'_checked',return_value=r))
            stack.enter_context(patch.object(branch,'core',return_value=current))
            stack.enter_context(patch.object(branch.actions,'route_choice',return_value=NS(admitted=np.ones(1,bool),engine_action=np.zeros(1,np.int32))))
            stack.enter_context(patch.object(branch,'_opponent_public',return_value=object()))
            stack.enter_context(patch.object(branch,'_stream',return_value=object()))
            stack.enter_context(patch.object(branch.driver,'_frame',side_effect=frame))
            with self.assertRaises(AtKernel):branch.step(root,(0,0),chance_inputs=object())
        for name in CONTEXTS:self.assertIs(captured[name+'_context'],r[name+'_context'])
        self.assertIsNone(captured['hypothetical_profile'])
        self.assertIsNone(captured['leech_history'])


    def test_whole_ordinary_driver_and_private_adapters_unchanged(self):
        # Forecast extends the source inventory without altering these already
        # integrated forwarding paths. Compare complete ASTs: no new call,
        # context, or admission gate may be normalized away.
        normalize=lambda s:ast.dump(ast.parse(s),include_attributes=False)
        for file in ('engine_retained_encoder_driver.py','v5_public_branch.py','v5_hypothetical_batch.py'):
            old=subprocess.check_output(['git','show',ORDINARY_SOURCE+':gpu_engine/'+file],text=True)
            new=Path('gpu_engine',file).read_text()
            self.assertEqual(normalize(old),normalize(new))
            if file=='engine_retained_encoder_driver.py':
                gate='engine = _rows(source_ready, core.engine, result.step.state)'
                self.assertEqual(new.count(gate),1)
                changed=new.replace(gate,'engine = _rows(~source_ready, core.engine, result.step.state)')
                self.assertNotEqual(normalize(old),normalize(changed))


if __name__=='__main__':unittest.main()
