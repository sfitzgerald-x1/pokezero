"""Portable tracked Git/source binding controls with inert bytes; no FP execution."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from pokezero import v5_evaluation as evaluation,v5_endpoint as endpoint

REPO=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root,*arguments):
    return subprocess.check_output(['git','-C',str(root),*arguments],stderr=subprocess.PIPE)


class FoulPlaySourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='pokezero-fp-source-unit-')
        self.addCleanup(self.temporary.cleanup);self.root=Path(self.temporary.name)
        self.fp=self.root/'source/third_party/foul-play';self.fp.parent.mkdir(parents=True)
        self.fp.mkdir()
        git(self.fp,'init','--quiet')
        for name in ('login.py','rest.py','capture.py'):
            (self.fp/name).write_text('# inert original source fixture '+name+'\n')
        git(self.fp,'add','.')
        git(self.fp,'-c','user.name=Unit','-c','user.email=unit@example.invalid','commit','--quiet','-m','UNIT-OWNED-BASE')
        self.pin=git(self.fp,'rev-parse','HEAD').decode().strip()
        # Three bound tracked edits exercise exact-diff integrity, not FP
        # mechanics. Separate authored local control applies the actual three
        # ORIGINAL patches to995525 and checks the reald4b94a1e diff.
        for name in ('login.py','rest.py','capture.py'):
            with (self.fp/name).open('a')as stream:stream.write('# inert bound patch '+name+'\n')
        difference=git(self.fp,'diff','--binary')
        self.python=self.root/'venv-foulplay/bin/python';self.python.parent.mkdir(parents=True)
        self.python.write_bytes(b'UNIT-INERT-PYTHON-NOT-EXECUTABLE')
        self.pyvenv=self.root/'venv-foulplay/pyvenv.cfg';self.pyvenv.write_bytes(b'UNIT-INERT-CONFIG')
        self.native=self.root/'venv-foulplay/lib/unit-native';self.native.parent.mkdir(parents=True);self.native.write_bytes(b'UNIT-INERT-NATIVE-NOT-EXECUTABLE')
        def ref(path):return dict(path=str(path),sha256=sha(path))
        manifest=REPO/'third_party/poke-engine-gen3-patches.txt'
        patches=[n.strip()for n in manifest.read_text().splitlines()if n.strip()and not n.lstrip().startswith('#')]
        names=['.gitmodules','scripts/setup_foulplay_eval.sh','scripts/setup_poke_engine.sh','scripts/verify_foulplay_poke_engine.py','scripts/verify_poke_engine_source.py','scripts/apply_poke_engine_patches.py','third_party/poke-engine-base-source.json','third_party/poke-engine-gen3-patches.txt']+['third_party/'+n for n in patches]+['third_party/foulplay-'+n+'.patch'for n in ('local-nosec','rest-cant','capture')]
        self.component=dict(status='ORIGINAL_FOULPLAY_COMPONENT_BUILD_COMPLETE_NOT_SELECTED',source_commit='1'*40,
            foulplay_gitlink=self.pin,native_patch_count=100,native_patch_manifest_sha256=sha(manifest),
            input_sha256={n:sha(REPO/n)for n in names},foulplay_applied_diff_sha256=hashlib.sha256(difference).hexdigest(),
            runtime=dict(python_executable=ref(self.python),pyvenv_config=ref(self.pyvenv),native=ref(self.native)))
        self.path=self.root/'component.json';self.protocol=dict(foulplay_commit=self.pin,foulplay_python=ref(self.python));self.publish()

    def publish(self):
        self.path.write_text(json.dumps(self.component,sort_keys=True)+'\n')
        self.protocol['foulplay_source']=dict(path=str(self.path),sha256=sha(self.path))

    def test_three_bound_changes_and_independent_dependency_commit_pass(self):
        self.assertTrue(git(self.fp,'status','--porcelain','--untracked-files=no'))
        evaluation._require_foulplay_source(self.protocol,(self.fp,self.fp))
        self.assertNotEqual(self.component['source_commit'],git(REPO,'rev-parse','HEAD').decode().strip())

    def test_unexpected_tracked_mutation_and_staged_mutation_refuse(self):
        file=self.fp/'login.py';file.write_bytes(file.read_bytes()+b'\n# UNIT-FOREIGN-TRACKED-MUTATION\n')
        with self.assertRaisesRegex(ValueError,'tracked worktree'):
            evaluation._require_foulplay_source(self.protocol,(self.fp,))
        git(self.fp,'add','login.py')
        with self.assertRaisesRegex(ValueError,'tracked worktree'):
            evaluation._require_foulplay_source(self.protocol,(self.fp,))

    def test_wrong_actual_gitlink_refuses(self):
        git(self.fp,'-c','user.name=Unit','-c','user.email=unit@example.invalid','commit','--quiet','--allow-empty','-m','UNIT-OWNED-FOREIGN-COMMIT')
        with self.assertRaisesRegex(ValueError,'declared FoulPlay source'):
            evaluation._require_foulplay_source(self.protocol,(self.fp,))

    def test_bad_artifact_digest_and_rebound_wrong_diff_refuse(self):
        self.protocol['foulplay_source']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'document digest'):
            evaluation._require_foulplay_source(self.protocol,(self.fp,))
        self.component['foulplay_applied_diff_sha256']='0'*64;self.publish()
        with self.assertRaisesRegex(ValueError,'tracked worktree'):
            evaluation._require_foulplay_source(self.protocol,(self.fp,))

    def test_foreign_current_build_input_or_changed_native_refuse(self):
        self.component['input_sha256']['third_party/foulplay-capture.patch']='0'*64;self.publish()
        with self.assertRaisesRegex(ValueError,'build input'):
            evaluation._require_foulplay_source(self.protocol,(self.fp,))
        self.component['input_sha256']['third_party/foulplay-capture.patch']=sha(REPO/'third_party/foulplay-capture.patch');self.publish()
        self.native.write_bytes(b'UNIT-CHANGED-NATIVE')
        with self.assertRaisesRegex(ValueError,'artifact digest'):
            evaluation._require_foulplay_source(self.protocol,(self.fp,))

    def test_both_protocols_require_explicit_source_artifact(self):
        import test_v5_evaluation as milestone,test_v5_endpoint as final
        # Metadata-only existing fixtures: no panel or factory invocation.
        with tempfile.TemporaryDirectory()as directory:
            fixture=milestone.EvaluationTests();fixture.setUp()
            try:
                reference,document,_,_=fixture.fixture(Path(directory))
                del document['foulplay_source']
                path=Path(reference.path);path.write_text(json.dumps(document))
                with self.assertRaisesRegex(ValueError,'explicit evaluation fields'):
                    evaluation._protocol(evaluation.Artifact(str(path),sha(path)))
            finally:fixture.doCleanups()
        with tempfile.TemporaryDirectory()as directory,patch.object(endpoint,'TRAINING_GAMES',1):
            reference,document,_=final.fixture(Path(directory));del document['foulplay_source']
            path=Path(reference.path);path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError,'explicit evaluation fields'):
                endpoint._protocol(evaluation.Artifact(str(path),sha(path)))

if __name__=='__main__':unittest.main()
