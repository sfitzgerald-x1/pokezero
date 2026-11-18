"""Source/input/scope guard checks; no engine, actor, or runtime preparation."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gpu_engine import fresh200_capture as capture, fresh200_driver as driver


class EntryGuardTests(unittest.TestCase):
    def test_replay_attempt_names_are_bounded_and_cannot_name_a_collection(self):
        self.assertEqual(driver.stage_name('collect', None), 'collect')
        self.assertEqual(driver.stage_name('replay', 'r62'), 'replay-r62')
        for stage, attempt in (('collect', 'r62'), ('replay', '../overwrite'), ('replay', 'x'*33), ('replay', '')):
            with self.subTest(stage=stage, attempt=attempt), self.assertRaises(ValueError):driver.stage_name(stage, attempt)

    def test_new_adapter_keeps_the_exact_original_capture_producer(self):
        original = dict(engine_commit='1'*40, producer_commit='2'*40, entry_sha256='3'*64, runtime_sha256='4'*64)
        self.assertIs(driver.capture_identity(original, dict(original), integration_commit='1'*40, runtime_digest='4'*64), original)
        for key in original:
            changed = dict(original);changed[key] = '5'*len(original[key])
            with self.subTest(key=key), self.assertRaises(ValueError):
                driver.capture_identity(changed, original, integration_commit='1'*40, runtime_digest='4'*64)

    def test_runtime_derivation_changes_only_source_commit(self):
        parent = dict(source_commit='1'*40, precision=dict(jax_x64=False), pool='literal')
        child = dict(parent, source_commit='2'*40)
        self.assertIs(driver.runtime_derivation(parent, child, '2'*40), child)
        for changed in (dict(child, pool='replacement'), dict(child, precision=dict(jax_x64=True)), dict(parent)):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'only the runtime source_commit'):
                driver.runtime_derivation(parent, changed, '2'*40)

    def test_launch_requires_exact_source_selection_resources_and_scope(self):
        launch = dict(schema='pokezero.fresh200.launch.v1', integration_commit='1'*40, producer_commit='2'*40,
            selection_policy=capture.selection_policy(), resources=dict(gpus=1,cpu=8,memory_bytes=48*1024**3,
                max_allocation_seconds=7200,utilisation_target_percent=30),
            scope='one fixed fresh200 device corpus and same-prefix candidate replay; no scoring or refill')
        check = lambda value: driver.validate_launch(value, integration_commit='1'*40, producer_commit='2'*40, policy=capture.selection_policy())
        self.assertIs(check(launch), launch)
        for key, value in (('gpus', 2), ('cpu', 32), ('memory_bytes', 128*1024**3), ('max_allocation_seconds', 7201)):
            changed = copy.deepcopy(launch);changed['resources'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):check(changed)
        changed = copy.deepcopy(launch);changed['selection_policy']['refill'] = True
        with self.assertRaises(ValueError):check(changed)
        changed = dict(launch, scope='scored run')
        with self.assertRaises(ValueError):check(changed)
        changed = dict(launch, integration_commit='3'*40)
        with self.assertRaises(ValueError):check(changed)

    def test_document_requires_absolute_path_digest_and_bounded_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'input.json';data = b'{"literal":17}\n';path.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            self.assertEqual(driver.read_document(path, digest), dict(literal=17))
            for target, expected, bound in ((path.name, digest, 64), (path, '0'*64, 64), (path, digest, 3)):
                with self.subTest(target=target, bound=bound), self.assertRaises(ValueError):
                    driver.read_document(target, expected, max_bytes=bound)

    def test_journal_requires_exact_order_hashes_and_single_initial_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = []
            for index, event in enumerate((dict(kind='RUN_STARTED', payload={}),
                    dict(kind='SNAPSHOT', payload=dict(iteration=-1)), dict(kind='BATCH_STARTED', payload={}))):
                path = Path(directory)/('%09d.json' % index);data = driver.canonical(event);path.write_bytes(data)
                manifest.append(dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), kind=event['kind']))
            self.assertEqual(len(driver.read_journal(manifest)), 3)
            for changed in (list(reversed(manifest)), manifest[:-1], manifest+[manifest[-1]], []):
                with self.subTest(changed=changed), self.assertRaises(ValueError):driver.read_journal(changed)
            changed = copy.deepcopy(manifest);changed[-1]['kind'] = 'OTHER'
            with self.assertRaisesRegex(ValueError, 'kind changed'):driver.read_journal(changed)

    def test_loaded_project_namespace_cannot_be_reused_across_sources(self):
        with self.assertRaisesRegex(ValueError, 'separate stage process'):driver.import_engine(Path('/unused'))
        root = Path(driver.__file__).resolve().parents[1]
        driver.assert_engine_modules(root)
        shadow = SimpleNamespace(__file__='/outside/pokezero/shadow.py')
        with patch.dict(sys.modules, {'pokezero._fresh200_shadow': shadow}), self.assertRaisesRegex(ValueError, 'shadowed'):
            driver.assert_engine_modules(root)


if __name__ == '__main__':unittest.main()
