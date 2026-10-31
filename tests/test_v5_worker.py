"""Bound artifact/constructor wiring with mocks; no runtime or model allocation."""
from contextlib import ExitStack,redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch,Mock

from pokezero import v5_worker as w,v5_run as run,v5_phasec as phase
from pokezero.category_vocab import CategoryVocabulary
from pokezero.live_protocol_actor import ActorContract,_vocab_sha
from test_v5_contract import vocabulary
from test_v5_phasec import original_metadata
from test_v5_run import plan as tiny_plan


def record(path,value):
    blob=json.dumps(value,sort_keys=True,allow_nan=False).encode() if not isinstance(value,bytes) else value
    path.write_bytes(blob);return dict(path=str(path),sha256=hashlib.sha256(blob).hexdigest())


class WorkerTests(unittest.TestCase):
    def fixture(self,root,*,evaluation=True):
        vocab=CategoryVocabulary(tuple('unit'+str(i) for i in range(111)),16,{})
        contract=replace(ActorContract.champion(),category_vocab_sha256=_vocab_sha(vocab),categorical_vocab_size=vocab.size)
        meta=original_metadata();meta['metadata']['model_config']['categorical_vocab_size']=vocab.size
        inventory=record(root/'inventory.json',meta)
        profile=phase.ChampionProfile.read(inventory['path'],expected_sha256=inventory['sha256'])
        pool=record(root/'pool.npz',b'synthetic-placeholder-not-loaded-as-NPZ')
        with patch.object(ActorContract,'champion',return_value=contract),patch.object(phase,'GAMES_PER_ARM',4):
            recipe=phase.bind_plan(profile,vocabulary(),category_vocabulary=vocab,run_id='unit',arm='B',replicate=0,
                iterations=tiny_plan().iterations,pool_sha256=pool['sha256'],selection_seed=9)
        if evaluation:recipe=replace(recipe,evaluation=run.EvaluationProtocol('c'*64))
        document=record(root/'plan.json',recipe.manifest())
        category=record(root/'vocabulary.json',dict(tokens=vocab.tokens,oov_buckets=16,aliases={},
            category_vocab_sha256=contract.category_vocab_sha256,checkpoint_sha256=contract.checkpoint_sha256,
            source_hash=contract.belief_set_source_hash))
        runtime=dict(schema=w.SCHEMA,source_commit='a'*40,control_inventory=inventory,category_vocabulary=category,
            constructor_pool=pool,gender_policy=None,showdown=dict(root='/unit/showdown',commit=contract.showdown_revision,
                compiled_tree_sha256='d'*64,node=record(root/'node',b'fake-node-never-executed')),
            versions=dict(python='unit',jax='unit',numpy='unit',torch='unit',torch_cuda='unit'),
            memory_policy=dict(environment=dict(zip(w.ENV_KEYS,('false',None,None))),torch_fraction=None),
            precision_policy=dict(training='bfloat16',inference='bfloat16',jax_x64=False,jax_matmul='highest',torch_matmul='highest'))
        runtime_record=record(root/'runtime.json',runtime)
        return recipe,contract,document,runtime_record,runtime

    def load(self,document,runtime,contract):
        with patch.object(ActorContract,'champion',return_value=contract),patch.object(phase,'GAMES_PER_ARM',4):
            return w.load_inputs(declaration=document['path'],expected_declaration_sha256=document['sha256'],
                runtime=runtime['path'],expected_runtime_sha256=runtime['sha256'],arm='B0',enable_v5=True)

    def test_opt_in_precedes_all_artifact_access_and_inspection_does_not_compile(self):
        with patch.object(w.Path,'read_bytes') as read:
            with self.assertRaisesRegex(ValueError,'enablement'):
                w.load_inputs(declaration='/no',expected_declaration_sha256='a'*64,runtime='/no',expected_runtime_sha256='b'*64,arm='B0')
            read.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            recipe,contract,doc,rt,_=self.fixture(Path(directory))
            with patch.object(ActorContract,'champion',return_value=contract),patch.object(phase,'GAMES_PER_ARM',4),patch.object(w,'execute') as execute,redirect_stdout(io.StringIO()):
                result=w.main(['--enable-v5','--inspect-only','--arm','B0','--declaration',doc['path'],
                    '--expected-declaration-sha256',doc['sha256'],'--runtime',rt['path'],
                    '--expected-runtime-sha256',rt['sha256'],'--run-root',str(Path(directory)/'out')])
            self.assertEqual(result,0);execute.assert_not_called()

    def test_exact_original_cells_controls_hashes_and_immutable_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            recipe,contract,doc,rt,raw=self.fixture(Path(directory));inputs=self.load(doc,rt,contract)
            self.assertEqual(inputs.plan,recipe);self.assertEqual(inputs.plan.games,recipe.games)
            self.assertEqual(inputs.declaration_sha256,doc['sha256'])
            changed=inputs.runtime;changed['source_commit']='f'*40
            self.assertEqual(inputs.runtime['source_commit'],raw['source_commit'])
            before=Path(doc['path']).read_bytes();value=json.loads(before)
            value['iterations'][0][0]['games'].reverse();Path(doc['path']).write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'digest'):self.load(doc,rt,contract)
            Path(doc['path']).write_bytes(before)
            Path(raw['constructor_pool']['path']).write_bytes(b'foreign-pool')
            with self.assertRaisesRegex(ValueError,'digest'):self.load(doc,rt,contract)

    def test_missing_or_changed_controls_and_reordered_cells_never_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);recipe,contract,doc,rt,_=self.fixture(root)
            for mutation in ('missing_discount','constant_schedule','wrong_value_lambda','reorder','wrong_role'):
                value=recipe.manifest()
                if mutation=='missing_discount':value.pop('discount')
                if mutation=='constant_schedule':value['train'].pop('learning_rate_schedule')
                if mutation=='wrong_value_lambda':value['loss']['value_lambda']=.99
                if mutation=='reorder':value['iterations'][0][0]['games']=tuple(reversed(value['iterations'][0][0]['games']))
                if mutation=='wrong_role':value['model']['replicate']=1
                altered=record(root/(mutation+'.json'),value)
                if mutation=='reorder':
                    # A differently declared order is preserved EXACTLY, not
                    # silently sorted back into somebody else's original bank.
                    loaded=self.load(altered,rt,contract)
                    self.assertEqual(loaded.plan.games[:2],tuple(reversed(recipe.games[:2])))
                else:
                    with self.subTest(mutation=mutation),self.assertRaises((ValueError,KeyError)):
                        self.load(altered,rt,contract)

    def test_runtime_metadata_and_memory_policy_are_explicit_not_guessed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,contract,doc,rt,raw=self.fixture(root)
            for mutation in ('no_torch','bad_memory','missing_memory','foreign_node','foreign_literal_vocab'):
                value=json.loads(json.dumps(raw))
                if mutation=='no_torch':value['versions'].pop('torch')
                if mutation=='bad_memory':value['memory_policy']['torch_fraction']=True
                if mutation=='missing_memory':value.pop('memory_policy')
                if mutation=='foreign_node':value['showdown']['node']['sha256']='b'*64
                if mutation=='foreign_literal_vocab':
                    v=json.loads(Path(value['category_vocabulary']['path']).read_bytes());v['tokens'][0]='foreign'
                    value['category_vocabulary']=record(root/'foreign-vocab.json',v)
                altered=record(root/(mutation+'.json'),value)
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.load(doc,altered,contract)

    def test_main_emits_compact_dispositions_and_never_reports_stopped_as_success(self):
        argv=['--enable-v5','--arm','B0','--declaration','/unit/plan','--expected-declaration-sha256','a'*64,
            '--runtime','/unit/runtime','--expected-runtime-sha256','b'*64,'--run-root','/unit/out']
        for status,expected_exit in [('STOPPED',1),('COMPLETE_DECLARATION',0)]:
            result=dict(status=status,reason='literal-source-reason',statuses={1:'REFUSED',2:'NOT_ATTEMPTED'})
            output=io.StringIO();hook=object()
            with patch.object(w,'load_inputs',return_value=object()),patch.object(w,'execute',return_value=result) as execute,redirect_stdout(output):
                self.assertEqual(w.main(argv,evaluation_hook=hook),expected_exit)
            report=json.loads(output.getvalue());self.assertEqual(report['status'],status)
            self.assertEqual(report['disposition_counts'],{'REFUSED':1,'NOT_ATTEMPTED':1})
            self.assertFalse(report['scientific_qualification']);self.assertNotIn('statuses',report)
            self.assertIs(execute.call_args.kwargs['evaluation_hook'],hook)

    def test_execution_requires_declared_panel_before_runtime_and_forwards_exact_api(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);recipe,contract,doc,rt,_=self.fixture(root);inputs=self.load(doc,rt,contract)
            with patch.object(w,'_source') as source:
                with self.assertRaisesRegex(run.MissingDependency,'evaluation'):w.execute(inputs,root/'run')
                source.assert_not_called()
            events=[];prepared=object();pool=object();public=object();backend=object();hook=lambda request:None
            with patch.object(w,'_source',side_effect=lambda _:events.append('source')),patch.object(w,'_memory',side_effect=lambda *a,**k:events.append('memory')),patch.object(w,'_probe',side_effect=lambda _:events.append('probe')),patch.object(w,'prepare_runtime',side_effect=lambda _:(events.append('prepare') or (prepared,pool,public,contract))),patch.object(phase,'build_backend',return_value=backend) as build,patch.object(run,'run',return_value={'unit':True}) as launch:
                self.assertEqual(w.execute(inputs,root/'run',resume=True,evaluation_hook=hook),{'unit':True})
            self.assertEqual(events,['source','memory','probe','prepare'])
            self.assertIs(build.call_args.args[0],inputs.plan);self.assertEqual(build.call_args.args[1:],(prepared,pool))
            self.assertEqual(launch.call_args.kwargs,dict(backend=backend,resume=True,enable_v5=True,evaluation_hook=hook))

    def test_memory_policy_accepts_only_cold_package_torch_and_no_jax(self):
        policy=dict(environment=dict(zip(w.ENV_KEYS,('false',None,None))),torch_fraction=None)
        cold=NS(cuda=NS(is_initialized=lambda:False))
        with patch.dict(sys.modules,{'torch':cold,'jax':None}),patch.dict(w.os.environ,{},clear=True):
            w._memory(policy,apply=True)
            self.assertEqual(w.os.environ[w.ENV_KEYS[0]],'false')
        for value in (NS(cuda=NS(is_initialized=lambda:True)),NS(cuda=NS(is_initialized=lambda:None)),NS(cuda=NS())):
            with patch.dict(sys.modules,{'torch':value,'jax':None}),patch.dict(w.os.environ,{},clear=True):
                with self.assertRaisesRegex(ValueError,'CUDA initialization'):w._memory(policy,apply=True)
                self.assertNotIn(w.ENV_KEYS[0],w.os.environ)
        with patch.dict(sys.modules,{'torch':cold,'jax':object()}),patch.dict(w.os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,'JAX imports'):w._memory(policy,apply=True)
            self.assertNotIn(w.ENV_KEYS[0],w.os.environ)

    def test_probe_rejects_absent_or_non_bf16_torch_before_any_compiler(self):
        with tempfile.TemporaryDirectory() as directory:
            *_,runtime=self.fixture(Path(directory))
            runtime['versions']['python']='.'.join(map(str,sys.version_info[:3]))
            cuda=NS(is_available=lambda:True,device_count=lambda:1,get_device_name=lambda _: 'NVIDIA GB300',
                is_bf16_supported=lambda:False,set_per_process_memory_fraction=Mock())
            torch=NS(__version__='unit',version=NS(cuda='unit'),cuda=cuda,set_float32_matmul_precision=Mock())
            jax=NS(__version__='unit',devices=lambda:[NS(platform='gpu',device_kind='NVIDIA GB300')],config=NS(update=Mock()))
            with patch.dict(sys.modules,{'torch':None,'jax':jax,'numpy':NS(__version__='unit')}):
                with self.assertRaises(ModuleNotFoundError):w._probe(runtime)
            with patch.dict(sys.modules,{'torch':torch,'jax':jax,'numpy':NS(__version__='unit')}):
                with self.assertRaisesRegex(ValueError,'GB300'):w._probe(runtime)
                cuda.is_bf16_supported=lambda:True;w._probe(runtime)
                torch.set_float32_matmul_precision.assert_called_once_with('highest')
                self.assertEqual(jax.config.update.call_args_list[0].args,('jax_enable_x64',False))

    def test_actual_compile_wiring_uses_original_driver_and_all_optional_modes_only(self):
        # Real source functions are IMPORTED but every compiler/loader is mocked.
        from pokezero import dex,randbat
        from gpu_engine import source_provenance,public_belief,v4_encoder,transition
        from gpu_engine import engine_public_events as events,engine_request_source as requests,engine_own_facts as own
        from gpu_engine import engine_retained_encoder_driver as driver
        from gpu_engine import engine_leftovers_callbacks as leftovers,engine_status_residual_callbacks as status
        from gpu_engine import engine_direct_hit_controls as hit,engine_substitute_callbacks as sub,engine_toxic_callbacks as toxic
        from gpu_engine import engine_recoil_callbacks as recoil,engine_calm_mind_callbacks as calm,engine_swords_dance_callbacks as sword
        from gpu_engine import engine_spikes_callbacks as spikes,engine_thunder_wave_callbacks as thunder
        with tempfile.TemporaryDirectory() as directory:
            _,contract,doc,rt,raw=self.fixture(Path(directory));inputs=self.load(doc,rt,contract)
            metadata=dict(compiled_tree=dict(sha256=raw['showdown']['compiled_tree_sha256']),vocabulary=dict(moves=['doubleedge','calmmind','swordsdance','spikes','thunderwave']))
            pool=NS(metadata=lambda:metadata);v=inputs.vocabulary
            catalog=NS(moves=object(),vocabulary=CategoryVocabulary(tuple(v['tokens']),v['oov_buckets'],v['aliases']),masks=NS(**dict(contract.feature_masks)))
            modules=[(leftovers,'compile_leftovers'),(status,'compile_status_residual'),(hit,'compile_direct_hit_controls'),(sub,'compile_substitute'),(toxic,'compile_toxic'),(recoil,'compile_recoil'),(calm,'compile_calm_mind'),(sword,'compile_swords_dance'),(spikes,'compile_spikes'),(thunder,'compile_thunder_wave')]
            with ExitStack() as stack:
                stack.enter_context(patch.object(ActorContract,'champion',return_value=contract))
                for module,name,result in [(dex,'load_showdown_dex_cached',object()),(randbat,'load_gen3_randbat_source_cached',NS(metadata=NS(source_hash=contract.belief_set_source_hash))),(source_provenance,'load_constructor_pool',pool),(source_provenance,'_pool',{'pool_sha256':inputs.plan.pool_sha256}),(public_belief,'compile_source',object()),(v4_encoder,'compile_encoder',catalog),(transition,'tables_from_metadata',object()),(events,'compile_catalog',object()),(requests,'load_source_rules',object()),(requests,'compile_request_source',object()),(own,'compile_own_facts',object())]:
                    stack.enter_context(patch.object(module,name,return_value=result))
                compilers=[stack.enter_context(patch.object(module,name,return_value=object())) for module,name in modules]
                prepare=stack.enter_context(patch.object(driver,'prepare',return_value=object()))
                prepared,got_pool,public,got_contract=w.prepare_runtime(inputs)
                options=prepare.call_args.kwargs
                metadata['vocabulary']['moves']=[]
                w.prepare_runtime(inputs)
                absent=prepare.call_args.kwargs
            self.assertIs(prepared,prepare.return_value);self.assertIs(got_pool,pool);self.assertIs(got_contract,contract)
            self.assertIs(public.category_vocab,catalog.vocabulary);self.assertIs(public.set_belief_source,True)
            self.assertEqual((options['public_mode'],options['frame_input_mode']),('retained-full','source-witness'))
            self.assertIs(options['account_source_sites'],True);self.assertIs(options['track_kernel_risks'],True)
            for name in ('status_residual','direct_hit','substitute','toxic','recoil','calm_mind','swords_dance','spikes','thunder_wave'):
                self.assertEqual(options[name+'_mode'],'kernel-event');self.assertIsNotNone(options[name+'_compiled'])
            self.assertEqual(options['leftovers_mode'],'retained')
            for compiler in compilers[:5]:self.assertEqual(compiler.call_count,2)
            for compiler in compilers[5:]:compiler.assert_called_once()
            for name in ('recoil','calm_mind','swords_dance','spikes','thunder_wave'):
                self.assertEqual(absent[name+'_mode'],'closed');self.assertIsNone(absent[name+'_compiled'])


if __name__=='__main__':unittest.main()
