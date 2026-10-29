import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class RecordedCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[1]/'scripts/compare_recorded_v4_encoder.py'
        spec=importlib.util.spec_from_file_location('recorded_cli_fixture',path)
        cls.cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.cli)

    def args(self,path):
        return ['--checkpoint','unused','--scratch-checkpoint','unused-new','--showdown','unused','--node','unused',
            '--collection','unused','--collection-sha256','a'*64,'--cohort-sha256','b'*64,'--output',str(path),'--mode','paired-forward']

    def test_snapshot_is_hash_bound_no_replace_and_readonly(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'authored-bytes';source.write_bytes(b'authored-not-a-model');dest=root/'copy'
            sha=hashlib.sha256(source.read_bytes()).hexdigest()
            result=self.cli.snapshot_checkpoint(source,dest,sha)
            self.assertEqual(dest.read_bytes(),source.read_bytes());self.assertEqual(result['sha256'],sha)
            self.assertEqual(dest.stat().st_mode&0o222,0)
            with self.assertRaises(FileExistsError):self.cli.snapshot_checkpoint(source,dest,sha)
            with self.assertRaises(ValueError):self.cli.snapshot_checkpoint(source,root/'bad','0'*64)
            self.assertFalse((root/'bad').exists());self.assertEqual(source.read_bytes(),b'authored-not-a-model')

    def test_late_failure_keeps_failed_attempt_and_no_comparison_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'new-output'
            with patch.object(self.cli,'build_runtime',return_value=(object(),{'authored':True})),patch(
                    'pokezero.recorded_encoder_collection.compare_collection',side_effect=ValueError('late fixed source SHA mismatch')):
                self.assertEqual(self.cli.main(self.args(path)),1)
            receipt=json.loads((path/'attempt.json').read_bytes())
            self.assertEqual(receipt['status'],'FAILED');self.assertEqual(receipt['actual_exit'],1)
            self.assertFalse((path/'comparison.json').exists());self.assertFalse(receipt['full_encoder_admitted'])
            with self.assertRaises(FileExistsError):self.cli.main(self.args(path))

    def test_partial_write_failure_never_exposes_final_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'new-output';real_open=Path.open
            class BrokenWrite:
                def __init__(self,stream):self.stream=stream
                def __enter__(self):self.stream.__enter__();return self
                def __exit__(self,*args):return self.stream.__exit__(*args)
                def write(self,blob):self.stream.write(blob[:17]);raise OSError('authored partial write')
            def failing(path,*args,**kw):
                stream=real_open(path,*args,**kw)
                return BrokenWrite(stream) if path.name=='.comparison.json.pending' and args==('xb',) else stream
            result={'metrics':{'views':4},'public_wire_qualified':False,'full_encoder_admitted':False}
            with patch.object(self.cli,'build_runtime',return_value=(object(),{'authored':True})),patch(
                    'pokezero.recorded_encoder_collection.compare_collection',return_value=result),patch.object(Path,'open',failing):
                self.assertEqual(self.cli.main(self.args(output)),1)
            self.assertFalse((output/'comparison.json').exists());self.assertFalse((output/'.comparison.json.pending').exists())
            self.assertEqual(json.loads((output/'attempt.json').read_bytes())['actual_exit'],1)

    def test_atomic_publication_preserves_existing_final_and_foreign_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);final=root/'comparison.json';pending=root/'.comparison.json.pending'
            self.cli.publish_comparison(root,b'{"complete":true}')
            self.assertEqual(json.loads(final.read_bytes()),{'complete':True});self.assertFalse(pending.exists())
            with self.assertRaises(FileExistsError):self.cli.publish_comparison(root,b'other')
            self.assertEqual(json.loads(final.read_bytes()),{'complete':True});self.assertFalse(pending.exists())
            pending.write_bytes(b'foreign temporary evidence')
            with self.assertRaises(FileExistsError):self.cli.publish_comparison(root,b'other')
            self.assertEqual(pending.read_bytes(),b'foreign temporary evidence')
