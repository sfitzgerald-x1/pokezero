"""Real file-backed capture/runtime schemas; compiler/device boundaries are inert."""
import ast
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict,replace
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from pokezero import v5_worker as worker,v5_evaluation_capture as capture,v5_run as run
from pokezero.category_vocab import CategoryVocabulary
from pokezero.live_protocol_actor import ActorContract,_vocab_sha
from test_v5_phasec import original_metadata
import test_v5_worker as training_fixture
from test_v5_worker import record

BASE='47c381c1fd8f40fc8358955b4ebdbfbb6f20f079'


def fixture(root,*,gender=False):
    vocab=CategoryVocabulary(tuple('unit'+str(i) for i in range(111)),16,{})
    contract=replace(ActorContract.champion(),category_vocab_sha256=_vocab_sha(vocab),categorical_vocab_size=vocab.size)
    meta=original_metadata();meta['metadata']['model_config']['categorical_vocab_size']=vocab.size
    inventory=record(root/'inventory.json',meta)
    pool=record(root/'pool.npz',b'UNIT pool bytes; never loaded as arrays')
    category=record(root/'category.json',dict(tokens=vocab.tokens,oov_buckets=16,aliases={},
        category_vocab_sha256=contract.category_vocab_sha256,checkpoint_sha256=contract.checkpoint_sha256,
        source_hash=contract.belief_set_source_hash))
    gender_value={'UNIT':'literal gender document; original constructor not invoked'} if gender else None
    runtime=dict(schema=worker.SCHEMA,source_commit='a'*40,control_inventory=inventory,category_vocabulary=category,
        constructor_pool=pool,gender_policy=record(root/'gender.json',gender_value) if gender else None,
        showdown=dict(root='/unit/showdown',commit=contract.showdown_revision,compiled_tree_sha256='d'*64,
            node=record(root/'node',b'UNIT executable bytes; never invoked')),
        versions=dict(python='unit',jax='unit',numpy='unit',torch='unit',torch_cuda='unit'),
        memory_policy=dict(environment=dict(zip(worker.ENV_KEYS,('false',None,None))),torch_fraction=None),
        precision_policy=dict(training='bfloat16',inference='bfloat16',jax_x64=False,jax_matmul='highest',torch_matmul='highest'))
    runtime_ref=record(root/'runtime.json',runtime)
    def roster(name,arm,rep,stream,role,pool_sha,*,uses_gender=False):
        batch=run.Batch((run.Game(stream,'UNIT-game-'+str(stream),stream+1,(0,1)),),stream+10,stream+11,
            **(dict(gender_seed=91,gender_domain=92,gender_root_ids=(stream+12,)) if uses_gender else {}))
        return record(root/name,dict(schema=capture.ROSTER,role=role,arm=arm,replicate=rep,
            pool_sha256=pool_sha,batches=[asdict(batch)]))
    training=[roster(arm+str(rep)+'.json',arm,rep,10+i,capture.TRAINING,str(i+1)*64)
        for i,(arm,rep) in enumerate((('A',0),('B',0),('D',0),('B',1)))]
    heldout=roster('heldout.json','EVALUATION',0,100,capture.NONTRAINING,pool['sha256'],uses_gender=gender)
    bank=dict(schema=capture.BANK,role=capture.NONTRAINING,heldout_roster=heldout,training_rosters=training,
        selectors=[dict(stream=100,frame=0,owner=0)],world_recipe=dict(seed=93),opponent_replies=[[0,1]],
        collector_archive_sha256='b'*64,opponent_archive_sha256='b'*64,collector_seats=[0],max_requests=30)
    return contract,record(root/'bank.json',bank),runtime_ref,runtime,bank,gender_value


class CaptureRuntimeTests(unittest.TestCase):
    def prepare(self,bank,runtime):
        return worker.prepare_capture_runtime(bank,runtime=runtime['path'],expected_runtime_sha256=runtime['sha256'],enable_v5=True)

    def boundaries(self,stack,contract):
        calls=[];outputs=tuple(object() for _ in range(4))
        stack.enter_context(patch.object(ActorContract,'champion',return_value=contract))
        stack.enter_context(patch.object(worker,'_plan',side_effect=AssertionError('capture must not load a training Plan')))
        source=stack.enter_context(patch.object(worker,'_source',side_effect=lambda rt:calls.append(('source',rt))))
        memory=worker._memory
        def check_memory(policy,*,apply):
            calls.append(('memory',apply))
            if not apply:memory(policy,apply=False)
        stack.enter_context(patch.object(worker,'_memory',side_effect=check_memory))
        probe=stack.enter_context(patch.object(worker,'_probe',side_effect=lambda rt:calls.append(('probe',rt))))
        compile_=stack.enter_context(patch.object(worker,'_prepare_runtime',side_effect=lambda rt,v,**kw:
            (calls.append(('compile',rt,v,kw)) or outputs)))
        return calls,outputs,source,probe,compile_

    def test_real_bank_and_runtime_no_plan_and_original_boundary_order(self):
        for gender in (False,True):
            with self.subTest(gender=gender),tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
                contract,bank,rt,raw,_,policy=fixture(Path(tmp),gender=gender)
                calls,outputs,_,_,compile_=self.boundaries(stack,contract)
                result=self.prepare(bank,rt)
                self.assertEqual(result,(*outputs,policy))
                self.assertEqual([x[0] for x in calls],['memory','source','memory','probe','compile'])
                self.assertEqual([x[1] for x in calls if x[0]=='memory'],[False,True])
                self.assertEqual(compile_.call_args.args[0],raw)
                self.assertEqual(compile_.call_args.kwargs,{'expected_pool_sha256':raw['constructor_pool']['sha256']})
                self.assertEqual(compile_.call_args.args[1],json.loads(Path(raw['category_vocabulary']['path']).read_bytes()))
                self.assertNotIn('plan',compile_.call_args.kwargs)

    def test_opt_in_hash_pool_source_profile_and_gender_refuse_before_compilation(self):
        with patch.object(worker,'_record') as read:
            with self.assertRaisesRegex(ValueError,'enablement'):
                worker.prepare_capture_runtime({'path':'/no','sha256':'a'*64},runtime='/no',expected_runtime_sha256='b'*64)
            read.assert_not_called()
        for mutation in ('runtime-bytes','bank-bytes','pool-bytes','different-pool','source','category','profile',
                'versions','memory','gender-omitted','gender-extra','missing-training-roster'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
                root=Path(tmp);contract,bank,rt,raw,doc,_=fixture(root,gender=mutation=='gender-omitted')
                if mutation=='runtime-bytes':Path(rt['path']).write_text('{}')
                elif mutation=='bank-bytes':Path(bank['path']).write_text('{}')
                elif mutation=='pool-bytes':Path(raw['constructor_pool']['path']).write_bytes(b'foreign bytes')
                elif mutation=='different-pool':raw['constructor_pool']=record(root/'other-pool',b'different valid pool bytes')
                elif mutation=='source':raw['showdown']['commit']='f'*40
                elif mutation=='category':
                    v=json.loads(Path(raw['category_vocabulary']['path']).read_bytes());v['tokens'][0]='foreign'
                    raw['category_vocabulary']=record(root/'other-category.json',v)
                elif mutation=='profile':
                    v=original_metadata();v['checkpoint_sha256']='f'*64;raw['control_inventory']=record(root/'other-profile.json',v)
                elif mutation=='versions':raw['versions'].pop('torch')
                elif mutation=='memory':raw['memory_policy']['torch_fraction']=True
                elif mutation=='gender-omitted':raw['gender_policy']=None
                elif mutation=='gender-extra':raw['gender_policy']=record(root/'unexpected-gender.json',{'UNIT':True})
                elif mutation=='missing-training-roster':
                    doc['training_rosters'].pop();bank=record(root/'bank-mutant.json',doc)
                if mutation not in ('runtime-bytes','bank-bytes','pool-bytes'):
                    rt=record(root/'runtime-mutant.json',raw)
                _,_,source,probe,compile_=self.boundaries(stack,contract)
                with self.assertRaises((ValueError,KeyError)):self.prepare(bank,rt)
                source.assert_not_called();probe.assert_not_called();compile_.assert_not_called()

    def test_original_source_and_probe_refusal_stop_before_shared_compiler(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract,bank,rt,_,_,_=fixture(Path(tmp))
            with patch.object(ActorContract,'champion',return_value=contract),\
                    patch.object(worker.subprocess,'check_output',return_value='f'*40),\
                    patch.object(worker,'_memory',wraps=worker._memory) as memory,\
                    patch.object(worker,'_probe') as probe,patch.object(worker,'_prepare_runtime') as compile_:
                with self.assertRaisesRegex(ValueError,'source differs'):self.prepare(bank,rt)
                self.assertEqual(memory.call_count,1);self.assertFalse(memory.call_args.kwargs['apply'])
                probe.assert_not_called();compile_.assert_not_called()
            with ExitStack() as stack:
                calls,_,_,probe,compile_=self.boundaries(stack,contract)
                probe.side_effect=ValueError('actual runtime version refusal')
                with self.assertRaisesRegex(ValueError,'version refusal'):self.prepare(bank,rt)
                self.assertEqual([x[0] for x in calls],['memory','source','memory']);compile_.assert_not_called()

    def test_training_wrapper_forwards_real_validated_plan_without_changed_contract(self):
        authored=training_fixture.WorkerTests()
        with tempfile.TemporaryDirectory() as tmp:
            _,contract,doc,rt,_=authored.fixture(Path(tmp));inputs=authored.load(doc,rt,contract)
            outputs=tuple(object() for _ in range(4))
            with patch.object(worker,'_prepare_runtime',return_value=outputs) as compile_:
                self.assertIs(worker.prepare_runtime(inputs),outputs)
            self.assertEqual(compile_.call_args.args,(inputs.runtime,inputs.vocabulary))
            self.assertEqual(compile_.call_args.kwargs,{'expected_pool_sha256':inputs.plan.pool_sha256})

    def test_original_compiler_validator_and_all_other_function_asts_preserved(self):
        old=ast.parse(subprocess.check_output(['git','show',BASE+':src/pokezero/v5_worker.py'],text=True))
        new=ast.parse(Path(worker.__file__).read_text())
        funcs=lambda t:{n.name:n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        a,b=funcs(old),funcs(new)
        # Exact old validator statements, only its pool source becomes a parameter.
        # Select by lexical calls rather than relying on the wrapper's statement count.
        start=next(i for i,n in enumerate(a['load_inputs'].body) if ast.unparse(n).startswith('_keys(runtime_document,'))
        end=next(i for i,n in enumerate(a['load_inputs'].body) if ast.unparse(n).startswith('plan = _plan('))
        statements=deepcopy(a['load_inputs'].body[start:end])
        class Validator(ast.NodeTransformer):
            def visit_Subscript(self,node):
                if ast.unparse(node)=="declaration_document['pool_sha256']":return ast.copy_location(ast.Name('expected_pool_sha256',ast.Load()),node)
                return self.generic_visit(node)
        statements=[Validator().visit(n) for n in statements]
        self.assertEqual(ast.dump(ast.Module(statements,[])),ast.dump(ast.Module(b['_runtime_metadata'].body[1:-1],[])))
        expected=deepcopy(a['prepare_runtime']);expected.name='_prepare_runtime';expected.args=deepcopy(b['_prepare_runtime'].args)
        class Compiler(ast.NodeTransformer):
            def visit_Attribute(self,node):
                text=ast.unparse(node)
                if text=='inputs.plan.pool_sha256':return ast.copy_location(ast.Name('expected_pool_sha256',ast.Load()),node)
                if text=='inputs.vocabulary':return ast.copy_location(ast.Name('vocabulary',ast.Load()),node)
                return self.generic_visit(node)
            def visit_Assign(self,node):
                if ast.unparse(node)=='runtime = inputs.runtime':return None
                return self.generic_visit(node)
        expected=Compiler().visit(expected)
        self.assertEqual(ast.dump(expected),ast.dump(b['_prepare_runtime']))
        for name in a.keys()-{'load_inputs','prepare_runtime'}:
            self.assertEqual(ast.dump(a[name]),ast.dump(b[name]),name)


if __name__=='__main__':unittest.main()
