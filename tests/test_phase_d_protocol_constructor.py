"""Inert extraction, shared-holder and fixed CPU-coverage ledger controls."""
import ast
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from gpu_engine.phase_d_protocol_constructor import ProtocolConstructor
from gpu_engine import phase_d_protocol_constructor as constructor_module
from gpu_engine.phase_d_ladder_search import Search
from pokezero.phase_d_ladder_panel import encoded,write_new
from pokezero.category_vocab import CategoryVocabulary
from pokezero.live_protocol_actor import ActorContract,_vocab_sha


def script():
    path=Path(__file__).resolve().parents[1]/'scripts'/'phase_d_constructor_coverage.py'
    spec=importlib.util.spec_from_file_location('constructor_coverage',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class ConstructorExtraction(unittest.TestCase):
    def constructor(self,**kwargs):
        catalog=object();functions=NS(catalog=catalog)
        return ProtocolConstructor(catalog=catalog,source=object(),metadata={},owning_config=object(),
            count=16,functions=functions,**kwargs)

    def test_reviewed_prepare_body_bytes_are_unchanged(self):
        import inspect
        from gpu_engine import phase_d_protocol_constructor as module
        source=inspect.getsource(module);tree=ast.parse(source)
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef)and x.name=='ProtocolConstructor')
        method=next(x for x in cls.body if isinstance(x,ast.FunctionDef)and x.name=='prepare')
        body='\n'.join(source.splitlines()[method.lineno:method.end_lineno])
        # Exact reviewed 3975b1fa Search._prepare_protocol body, before extraction.
        self.assertEqual(hashlib.sha256(body.encode()).hexdigest(),
            '93776b7997069c281389bdec10b2c35b0651d3ff291f373caabd0bcb2777cc49')

    def test_search_delegates_and_preserves_real_shared_holders(self):
        seen=set();sessions={};constructor=self.constructor(seen=seen,sessions=sessions)
        self.assertIs(constructor.seen,seen);self.assertIs(constructor.sessions,sessions)
        search=Search.__new__(Search);search.constructor=constructor
        request={'unchanged':'request'};expected=object()
        with patch.object(ProtocolConstructor,'prepare',autospec=True,return_value=expected)as prepare:
            self.assertIs(search._prepare_protocol(request,19),expected)
        prepare.assert_called_once_with(constructor,request,19)

    def test_constructor_counts_all_refused_slots_without_search_time(self):
        constructor=self.constructor();worlds=[NS(index=i,state=None if i==3 else object(),
            refusals=('named-refusal',)if i==3 else())for i in range(16)]
        made=NS(fully_constructed=False,worlds=worlds)
        with patch.object(ProtocolConstructor,'prepare',return_value=(object(),made,(True,)*9)):
            result=constructor.construct({'request_id':'inert'})
        self.assertEqual(result['constructed_worlds'],15);self.assertEqual(result['requested_worlds'],16)
        self.assertEqual(result['world_refusals'],[dict(world=3,kinds=['named-refusal'])])
        self.assertEqual(result['status'],'REFUSED');self.assertIsNone(result['decision_seconds'])
        self.assertIsNone(result['action_index']);self.assertNotIn('archive_sha256',result)

    def test_cli_selects_cpu_before_importing_jax_and_accepts_no_archive(self):
        path=Path(__file__).resolve().parents[1]/'scripts'/'phase_d_constructor_coverage.py'
        code="""import os,runpy,sys
from types import SimpleNamespace as NS
module=runpy.run_path(sys.argv[1])
assert 'jax' not in sys.modules
os.environ['JAX_PLATFORMS']='gpu'
module['select_platform'](NS(platform='cpu',compilation_cache='/inert/cache'))
assert os.environ['JAX_PLATFORMS']=='cpu'
assert os.environ['JAX_COMPILATION_CACHE_DIR']=='/inert/cache'
assert 'jax' not in sys.modules
assert '--archive' not in module['parser']()._option_string_actions
print('CPU selected before JAX; no archive input')
"""
        result=subprocess.run([sys.executable,'-c',code,str(path)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_fixed_ledger_keeps_interrupted_cell_unknown_and_never_replays(self):
        module=script();constructor=self.constructor()
        with TemporaryDirectory()as directory:
            root=Path(directory);rows=[]
            for i in range(200):
                rows.append(dict(cell=dict(ordinal=0),owner='p1',decision_index=i,prefix_sha256=f'{i:064x}',
                    turn=(1,4,10,20)[i//50],capture_path=str(root/'inert.jsonl'),capture_row=i,
                    battle_id='inert',stream_id=1,generation=i+1,lines=['|request|inert'],own_request_line='|request|inert'))
            roster=root/'roster.json';write_new(roster,dict(selected=rows))
            review=root/'review.json';write_new(review,dict(scope='original producer review'))
            output=root/'out';write_new(output/'claims'/'0.json',dict(original_interrupted_claim=True))
            args=NS(roster=str(roster),expected_roster_sha256=hashlib.sha256(roster.read_bytes()).hexdigest(),
                producer_review=str(review),expected_producer_review_sha256=hashlib.sha256(review.read_bytes()).hexdigest(),
                output=str(output),expected_metadata_sha256='a'*64,expected_vocabulary_sha256='b'*64,
                worlds=16,seed=20261103,platform='cpu')
            def constructed(request,index):return dict(requested_worlds=16,constructed_worlds=16,status='COMPLETE',constructor_seconds=.1)
            with patch('pokezero.phase_d_owning_capture.read_capture',side_effect=lambda path:iter(rows)),\
                    patch.object(ProtocolConstructor,'construct',side_effect=constructed)as construct,\
                    patch('builtins.print'):
                result=module.measure(args,constructor);self.assertEqual(construct.call_count,199)
                self.assertEqual(module.measure(args,constructor),result);self.assertEqual(construct.call_count,199)
            self.assertEqual(result['measured_roots'],200)
            self.assertEqual(result['summary']['fully_constructed_roots'],199)
            self.assertEqual(result['summary']['refusals_by_root_kind'],{'INTERRUPTED_ORIGINAL_ATTEMPT_UNKNOWN_NO_REPLAY':1})
            self.assertIsNone(result['summary']['decision_p95_seconds'])
            record=json.loads((output/'records'/'0.json').read_bytes())
            self.assertEqual(record['constructed_worlds'],0);self.assertIsNone(record['decision_seconds'])


class WrappedStartupVocabulary(unittest.TestCase):
    def fixture(self):
        vocab=CategoryVocabulary(('field','pokemon:self','species:pikachu'),oov_buckets=7,
            aliases={'species:pikachucosplay':'species:pikachu'})
        contract=replace(ActorContract.champion(),category_vocab_sha256=_vocab_sha(vocab),categorical_vocab_size=vocab.size)
        # The actual published wrapper's nine fields, not flattened dataclass kwargs.
        document=dict(aliases=dict(vocab.aliases),category_vocab_sha256=contract.category_vocab_sha256,
            checkpoint_sha256=contract.checkpoint_sha256,oov_buckets=vocab.oov_buckets,
            scope='ORIGINAL_STATIC_RUNTIME_CATEGORY_VOCABULARY',size=vocab.size,
            source_hash=contract.belief_set_source_hash,tokens=list(vocab.tokens))
        return vocab,contract,document

    def test_actual_dataclass_preserves_wrapper_tokens_aliases_and_oov(self):
        vocab,contract,document=self.fixture()
        resolved=constructor_module.vocabulary_from_document(document,contract)
        self.assertIs(type(resolved),CategoryVocabulary)
        self.assertEqual(resolved.tokens,vocab.tokens);self.assertEqual(resolved.aliases,vocab.aliases)
        self.assertEqual(resolved.oov_buckets,7);self.assertEqual(resolved.size,vocab.size)
        self.assertEqual(resolved.encode('species:pikachucosplay'),resolved.encode('species:pikachu'))
        for name,value in (('checkpoint_sha256','a'*64),('category_vocab_sha256','b'*64),
                ('source_hash','foreign-source'),('size',vocab.size+1),('oov_buckets',8),
                ('tokens',list(reversed(document['tokens'])))):
            with self.subTest(field=name),self.assertRaises(ValueError):
                constructor_module.vocabulary_from_document(dict(document,**{name:value}),contract)

    def test_load_inputs_uses_wrapped_document_and_actual_startup_config_apis(self):
        vocab,contract,document=self.fixture()
        from pokezero import dex,randbat,v5_contract
        from pokezero.local_showdown import LocalShowdownConfig
        from gpu_engine import public_belief,v4_encoder,device_public_worlds
        metadata=dict(source_commit_label=contract.showdown_revision,vocabulary=dict(moves=['','tackle']),moves=[None,dict(pp=35)])
        source=NS(metadata=NS(source_hash=contract.belief_set_source_hash));bound=object();catalog=object()
        with TemporaryDirectory()as directory:
            root=Path(directory);vp=root/'vocabulary.json';mp=root/'metadata.json'
            write_new(vp,document);write_new(mp,metadata)
            args=NS(showdown_root='/inert/showdown',node_binary='/inert/node',vocabulary=str(vp),metadata=str(mp),
                expected_vocabulary_sha256=hashlib.sha256(vp.read_bytes()).hexdigest(),
                expected_metadata_sha256=hashlib.sha256(mp.read_bytes()).hexdigest())
            with patch.object(ActorContract,'champion',return_value=contract),\
                    patch('subprocess.check_output',side_effect=[contract.showdown_revision+'\n','']),\
                    patch.object(randbat,'load_gen3_randbat_source_cached',autospec=True,return_value=source),\
                    patch.object(dex,'load_showdown_dex_cached',autospec=True,return_value=object()),\
                    patch.object(public_belief,'compile_source',autospec=True,return_value=object()),\
                    patch.object(v4_encoder,'compile_encoder',autospec=True,return_value=catalog)as compile_encoder,\
                    patch.object(v5_contract,'source_variant_census',autospec=True,return_value=dict(source_rows_sha256='c'*64)),\
                    patch.object(device_public_worlds,'compile_source',autospec=True,return_value=bound)as compile_source:
                got_catalog,got_bound,got_metadata,config=constructor_module.load_inputs(args)
            self.assertIs(got_catalog,catalog);self.assertIs(got_bound,bound);self.assertEqual(got_metadata,metadata)
            self.assertIs(type(config),LocalShowdownConfig);self.assertEqual(config.category_vocab.tokens,vocab.tokens)
            self.assertEqual(config.category_vocab.aliases,vocab.aliases);self.assertEqual(config.category_vocab.oov_buckets,7)
            self.assertEqual(config.node_binary,'/inert/node');self.assertIs(config.observation_spec,v5_contract.V4_SPEC)
            self.assertEqual(config.feature_masks.opponent_tendency_stats_block,dict(contract.feature_masks)['stats_block'])
            compiled_vocab=compile_encoder.call_args.args[2]
            self.assertIs(compiled_vocab,config.category_vocab)
            self.assertEqual(compile_encoder.call_args.kwargs['feature_masks'],dict(contract.feature_masks))
            self.assertEqual(compile_source.call_args.args,(source,metadata))
            self.assertEqual(compile_source.call_args.kwargs,dict(expected_source_hash=contract.belief_set_source_hash,
                expected_source_rows_sha256='c'*64))


if __name__=='__main__':unittest.main()
