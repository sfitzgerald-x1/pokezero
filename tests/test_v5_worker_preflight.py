"""Independent artifact/entry guards; fake paths/providers only."""
import hashlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from pokezero import v5_worker as w,v5_run as run
import test_v5_worker as authored


class WorkerBoundaryControls(unittest.TestCase):
    def test_noncallable_panel_refuses_before_source_memory_probe_or_prepare(self):
        fixture=authored.WorkerTests()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,contract,doc,rt,_=fixture.fixture(root);inputs=fixture.load(doc,rt,contract)
            with patch.object(w,'_source') as source,patch.object(w,'_memory') as memory,\
                    patch.object(w,'_probe') as probe,patch.object(w,'prepare_runtime') as prepare,\
                    patch('pokezero.v5_phasec.build_backend') as backend,patch.object(run,'run') as launch:
                prepare.return_value=(object(),object(),object(),object())
                with self.assertRaises((ValueError,run.MissingDependency)):
                    w.execute(inputs,root/'out',evaluation_hook=object())
                for boundary in (source,memory,probe,prepare,backend,launch):boundary.assert_not_called()

    def test_loader_path_node_must_equal_digest_bound_declared_node(self):
        fixture=authored.WorkerTests()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);*_,runtime=fixture.fixture(root)
            alternate=root/'different-path';alternate.mkdir();other=alternate/'node'
            other.write_bytes(b'different-node-never-executed');other.chmod(0o755)
            def git(args,**kwargs):
                if 'status' in args:return ''
                return (runtime['showdown']['commit'] if args[2]==runtime['showdown']['root'] else runtime['source_commit'])+'\n'
            with patch.object(w.subprocess,'check_output',side_effect=git),patch.dict(os.environ,{'PATH':str(alternate)}):
                with self.assertRaisesRegex(ValueError,'[Nn]ode|runtime|source'):
                    w._source(runtime)


class WorkerCategoryIntegrationControl(unittest.TestCase):
    def test_direct_recipe_category_join_uses_full_original_binding(self):
        fixture=authored.WorkerTests()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,contract,doc,rt,_=fixture.fixture(root);inputs=fixture.load(doc,rt,contract)
            value=inputs.vocabulary
            # Exact original private_team.vocabulary_binding JSON payload; no
            # JAX compilation/array/model is needed to expose this mismatch.
            full=hashlib.sha256(json.dumps(dict(tokens=value['tokens'],oov_buckets=value['oov_buckets'],
                aliases=value['aliases']),sort_keys=True,separators=(',',':')).encode()).hexdigest()
            self.assertNotEqual(full,contract.category_vocab_sha256)
            self.assertEqual(inputs.plan.model.category_vocabulary_sha256,full,
                'original v5_observation.compile_adapter requires this full binding, not tokens-only ActorContract hash')


if __name__=='__main__':unittest.main()
