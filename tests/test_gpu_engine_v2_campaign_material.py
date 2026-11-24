"""CPU fixtures for launching a campaign before its full roster exists."""

import hashlib
import json
from pathlib import Path
import shlex
from tempfile import TemporaryDirectory
from threading import Event, Thread
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from gpu_engine import gender, source_provenance
from gpu_engine import v2_material as producer
from gpu_engine import v2_material_contract as material
from gpu_engine.v2_pool_assembly import assemble, file_sha
from pokezero.v2_fused_state import verify_material_resume
from pokezero.v5_run import _json
from scripts import bind_v2_campaign_packet as binder, supervise_v2_campaign as supervisor
from tests.test_gpu_engine_v2_campaign_pool import make_shard as legacy_test_shard


SPECIES = ['_none', *(f'fixture{number}' for number in range(128))]


def make_shard(path, start, stop, row_start, *, value=1):
    ref = legacy_test_shard(path, start, stop, row_start, value=value)
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    metadata = json.loads(arrays['metadata_utf8'].tobytes())
    metadata['vocabulary']['species'] = SPECIES
    arrays['metadata_utf8'] = np.frombuffer(material.encode(metadata), dtype=np.uint8)
    np.savez_compressed(path, **arrays)
    return dict(ref, sha256=file_sha(path))


class CampaignMaterialTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.parent = make_shard(self.root / 'parent.npz', 90, 94, 0)
        self.policy = self.root / 'policy.json'
        self.make_policy(self.parent, None, self.policy)
        inventory = self.root / 'inventory.json'
        inventory.write_bytes(material.encode(dict(complete=True, seed_ranges=[dict(start=118, stop=120)])))
        self.plan = dict(version=1, root=str(self.root / 'a'), parent_pool=self.artifact(self.parent),
                         gender_policy=self.artifact(self.policy),
                         showdown=dict(root='/x', commit='a' * 40, node=self.artifact(self.policy)),
                         preserved_team_rows=4, protected_seeds=[[118, 120]],
                         protected_seed_inventories=[self.artifact(inventory)], shards=[
                             dict(row_start=0, seeds=[90, 94], existing=self.artifact(self.parent)),
                             dict(row_start=4, seeds=[100, 103]),
                             dict(row_start=7, seeds=[103, 106]),
                             dict(row_start=10, seeds=[108, 112]),
                             dict(row_start=14, seeds=[115, 117]),
                         ],
                         windows=[dict(rounds=[100, 102], rows=[4, 8]),
                                  dict(rounds=[102, 103], rows=[10, 14])])
        self.plan_ref = self.write_plan(self.plan)
        self.generated = []

    def artifact(self, path):
        if isinstance(path, dict):
            return {key: path[key] for key in ('path', 'sha256')}
        return dict(path=str(path), sha256=file_sha(path))

    def write_plan(self, plan):
        path = self.root / 'p.json'
        path.write_bytes(material.encode(plan))
        return self.artifact(path)

    def make_policy(self, pool, showdown, path):
        policy = dict(schema=gender.SCHEMA, species=SPECIES, codes=[-1, *([3] * (len(SPECIES) - 1))],
                      provenance={'pool_sha256': pool['sha256']})
        path.write_bytes(material.encode(policy))
        return self.artifact(path)

    def generate(self, path, *, teams, seed_start, **kwargs):
        self.generated.append(seed_start)
        return make_shard(path, seed_start, seed_start + teams, 0, value=seed_start % 50 + 1)

    def publish(self, start, stop, **kwargs):
        kwargs.setdefault('generate', self.generate)
        return producer.publish(self.plan_ref, start, stop, policy_writer=self.make_policy, **kwargs)

    def test_a_incremental_matches_separate_upfront_shards_and_full_digest(self):
        up_front = self.root / 'up'
        up_front.mkdir()
        shard_refs = [self.parent]
        for shard in self.plan['shards'][1:]:
            start, end = shard['seeds']
            generated = self.generate(up_front / f'{start}.npz', teams=end - start, seed_start=start)
            shard_refs.append(dict(generated, row_start=shard['row_start']))
        legacy = assemble(dict(shards=shard_refs, required_team_rows=16, preserved_team_rows=4,
                               parent_pool=self.artifact(self.parent)), up_front / 'complete.npz')
        first = self.publish(0, 1, workers=2)
        self.assertFalse(material.manifest_path(self.plan, 1).exists())
        self.assertFalse(material.shard_path(self.plan, self.plan['shards'][-1]).exists())
        last = self.publish(1, 2, workers=2)
        for expected, shard in zip(shard_refs, self.plan['shards'], strict=True):
            actual = material.shard_path(self.plan, shard)
            self.assertEqual(actual.read_bytes(), Path(expected['path']).read_bytes())
        final = json.loads((Path(self.plan['root']) / 'z' / 'd.json').read_bytes())
        self.assertEqual(final['pool_sha256'], legacy['sha256'])
        self.assertEqual(final['ledger_sha256'], last)
        self.assertEqual(last, self.publish(1, 2))
        self.assertEqual(json.loads(material.manifest_path(self.plan, 1).read_bytes())['previous_sha256'], first)
        print(f'fixture a digest {legacy["sha256"]}', flush=True)

    def controller(self):
        declaration = {'path': '/x/d', 'sha256': 'd' * 64}
        main = dict(output='v2a-012abc-gpu', declaration=declaration, stops=[102, 103],
                    restore={'milestone_sha256s': {'1000000': '1' * 64, '2000000': '2' * 64}},
                    max_exec_seconds=3600, min_exec_seconds=1200)
        packet = dict(main=main, children=[], launcher={'path': 'launch.py'}, checkout='/tree',
                      reference_run='/reference', previous_runtime='/old', runtime={'path': '/r', 'sha256': 'b' * 64},
                      resilience={'path': '/x/s', 'phase_observer_sha256': 'p', 'signal_driver_sha256': 's'},
                      cpu=32, memory='896Gi', stop_key_reference='env=existing:key',
                      material=dict(plan=self.plan_ref, ready=[], producer=dict(
                          lookahead=1, workers=1, cpu=4, memory='4Gi', max_exec_seconds=1800, nodes=[])))
        controller = supervisor.Supervisor(packet, self.root / 'ledger.json', 'fixture', 100000)
        pointer = {'round_ledger': {'completed_rounds': 100, 'latest_snapshot': {'iteration': 99}}}
        return controller, pointer

    def test_b_missing_corrupt_and_digest_mismatched_material_waits(self):
        controller, _ = self.controller()
        segment = controller.segments[0]
        with patch.object(supervisor, 'material_bytes', side_effect=producer.read_bytes), patch.object(
            supervisor, 'verify_material_files', side_effect=producer.verify_files
        ), patch.object(supervisor, 'shared_text') as state, patch.object(supervisor.subprocess, 'run') as launch:
            controller.submit(segment)
            self.assertIn('missing', controller.ledger['material_wait']['detail'])
            self.publish(0, 1)
            shard = material.shard_path(self.plan, self.plan['shards'][1])
            original = shard.read_bytes()
            shard.unlink()
            controller.submit(segment)
            self.assertIn('missing', controller.ledger['material_wait']['detail'])
            shard.write_bytes(b'corrupt')
            controller.submit(segment)
            self.assertIn('checksum differs', controller.ledger['material_wait']['detail'])
            shard.write_bytes(original)
            path = material.manifest_path(self.plan, 0)
            document = json.loads(path.read_bytes())
            document['shards'][0]['artifact']['sha256'] = '0' * 64
            path.write_bytes(material.encode(document))
            controller.submit(segment)
            self.assertIn('verification differs', controller.ledger['material_wait']['detail'])
            state.assert_not_called()
            launch.assert_not_called()
            self.assertEqual(controller.ledger['attempts'], [])
            self.assertEqual(controller.ledger['reserved_gpu_seconds'], 0)

    def test_c_early_launch_and_d_exact_retry(self):
        from gpu_engine.v2_fused import parse_args
        from gpu_engine.v5_device_backend import RuntimeBackend
        import jax
        self.assertEqual(jax.default_backend(), 'cpu')

        controller, pointer = self.controller()
        self.publish(0, 1)
        manifest = material.manifest_path(self.plan, 0)
        controller.packet['material']['ready'] = [self.artifact(manifest)]

        def payload():
            return json.dumps(pointer)

        def record_attempt(controller, outcome):
            attempt = controller.ledger['attempts'][-1]
            attempt.update(outcome=outcome, status='RELEASING',
                           committed_round=supervisor.completed_rounds(pointer),
                           state_sha256=hashlib.sha256(payload().encode()).hexdigest())
            controller.release(attempt)

        with patch.object(supervisor, 'material_bytes', side_effect=producer.read_bytes), patch.object(
            supervisor, 'verify_material_files', side_effect=producer.verify_files
        ), patch.object(supervisor, 'verify_initial_root'), patch.object(
            supervisor, 'verify_source'
        ), patch.object(supervisor, 'shared_text', side_effect=lambda *a: payload()), patch.object(
            supervisor, 'read_resource', return_value=None
        ), patch.object(supervisor.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as launch:
            controller.submit(controller.segments[0])
            self.assertFalse(material.manifest_path(self.plan, 1).exists())
            self.assertEqual(launch.call_count, 1)

            def bound_rows(round_number, attempt_number):
                launcher = launch.call_args.args[0]
                self.assertEqual(launcher[launcher.index('--resume-run') + 1], controller.packet['main']['output'])
                cli = shlex.split(launcher[-1])
                args = parse_args([*cli[cli.index('--out'):], '--require-resume'])
                self.assertEqual(args.segment_start[0], round_number)
                self.assertEqual(args.segment_start[2], attempt_number)
                self.assertEqual(args.adopted_milestone_sha256s, ['1' * 64, '2' * 64])
                start = cli.index('--material') + 1
                _, _, doc = material.load_segment(cli[start:start + 5], 102, round_number)
                pool = source_provenance.load_constructor_pool(doc['pool']['path'])
                self.assertEqual(json.loads(pool.receipt_json)['pool_sha256'], doc['pool']['sha256'])
                backend = RuntimeBackend.__new__(RuntimeBackend)
                backend.pool = pool
                backend.material_address = (self.plan_ref['sha256'], doc['row_offset'])
                backend.prepared = SimpleNamespace(source=None, catalog=SimpleNamespace(source=None))
                backend._plan = SimpleNamespace(model=SimpleNamespace(label_vocabulary=None))
                backend.measure_runtime = False
                backend.initialize = lambda state: state
                batch = SimpleNamespace(
                    games=[SimpleNamespace(team_indices=(4, 5), stream_id=0),
                           SimpleNamespace(team_indices=(6, 7), stream_id=1)],
                    constructor_seed=12, gender_seed=None)
                with patch('gpu_engine.engine_request_source._gender', side_effect=lambda state, *a: state._replace(
                    gender=state.gender * 0
                )), patch('gpu_engine.v5_device_rollout.constructor_truth', return_value=None), patch(
                    'gpu_engine.v5_device_rollout.initialize', side_effect=lambda state, *a, **kw: state
                ):
                    actual = backend._construct(batch)
                    return cli[start + 4], [np.asarray(value).tobytes() for value in jax.tree.leaves(actual)]

            first_manifest, original_rows = bound_rows(100, 1)
            pointer['material_sha256'] = first_manifest
            pointer['round_ledger'].update(completed_rounds=101, latest_snapshot={'iteration': 100})
            record_attempt(controller, 'RETRY')
            controller = supervisor.Supervisor(controller.packet, controller.path, 'fixture', 100000)
            controller.submit(controller.segments[0])
            replay_manifest, replay_rows = bound_rows(101, 2)
            self.assertEqual((first_manifest, original_rows), (replay_manifest, replay_rows))
            verify_material_resume(dict(pointer, material_sha256=replay_manifest), [100, 102], first_manifest)
            with self.assertRaisesRegex(ValueError, 'original material manifest'):
                verify_material_resume(dict(pointer, material_sha256='0' * 64), [100, 102], first_manifest)
            pointer['round_ledger'].update(completed_rounds=102, latest_snapshot={'iteration': 101})
            record_attempt(controller, 'ADVANCE')
            controller.submit(controller.segments[1])
            self.assertEqual(launch.call_count, 2)

            def missing(paths):
                return {str(path) for path in paths if not path.exists()}

            with patch.object(supervisor, 'missing_material', side_effect=missing):
                controller.ledger['segment'] = 0
                controller.advance_producer()
                generator = launch.call_args.args[0]
                self.assertIn('--gpus', generator)
                self.assertEqual(generator[generator.index('--gpus') + 1], '0')
                self.assertIn('--start 1 --stop 2', generator[-1])
                controller.ledger['segment'] = 1
            self.publish(1, 2)
            pointer['material_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'prior owned attempt'):
                controller.submit(controller.segments[1])
            prior = controller.ledger['attempts'][-1]
            valid_state_sha256 = prior['state_sha256']
            prior['state_sha256'] = hashlib.sha256(payload().encode()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'admitted on this lineage'):
                controller.submit(controller.segments[1])
            prior['state_sha256'] = valid_state_sha256
            pointer['material_sha256'] = first_manifest
            controller.submit(controller.segments[1])
            self.assertEqual(launch.call_count, 4)

    def test_completion_waits_for_verified_material_and_reconciles_outputs(self):
        controller, _ = self.controller()
        controller.ledger['segment'] = len(controller.segments)
        with patch.object(controller, 'advance_producer') as producer_step, patch.object(
            controller, 'material_complete', side_effect=[False, True]
        ), patch.object(controller, 'reconciled_rounds', return_value=103) as reconciled, patch.object(
            supervisor.time, 'sleep'
        ) as wait:
            self.assertEqual(controller.work(), 0)
        self.assertEqual(producer_step.call_count, 2)
        wait.assert_called_once_with(supervisor.TICK_SECONDS)
        reconciled.assert_called_once_with(controller.packet['main']['output'])
        self.assertEqual(controller.ledger['status'], 'COMPLETE')

    def test_e_full_packet_matches_release_serialization(self):
        config = {key: key for key in ('checkout', 'source_commit', 'launcher', 'resilience', 'cpu', 'memory',
                                       'reference_run', 'stop_key_reference')}
        config.update(main={'output': 'v2a-012abc-gpu'}, children=[], previous_runtime={'path': '/x/r'})
        declarations = dict(main=dict(path='d.json', sha256='a' * 64, effective_sha256='b' * 64, stops=[1250]),
                            children=[])
        runtime = dict(path='/x/r2', sha256='c' * 64)
        packet = binder.build_packet(config, declarations, Path('/x'), runtime)
        self.assertEqual(hashlib.sha256((_json(packet) + '\n').encode()).hexdigest(), '5a3a923b76c79bdb7fde050cb5c0e6345474bfec01e1ef368b0920849878f764')

    def test_f_overlapping_writer_is_refused_and_rerun_does_not_write(self):
        entered, proceed = Event(), Event()
        exceptions = []

        def held_generator(*args, **kwargs):
            entered.set()
            if not proceed.wait(15):
                raise RuntimeError('timed out waiting for fixture collision probe')
            return self.generate(*args, **kwargs)

        def writer():
            try:
                self.publish(0, 1, generate=held_generator)
            except Exception as error:
                exceptions.append(error)

        thread = Thread(target=writer)
        thread.start()
        try:
            self.assertTrue(entered.wait(15))
            with self.assertRaises(BlockingIOError):
                self.publish(0, 1)
        finally:
            proceed.set()
            thread.join(15)
        self.assertFalse(thread.is_alive())
        self.assertFalse(exceptions)
        self.assertCountEqual(self.generated, [100, 103])
        self.publish(0, 1)
        self.assertCountEqual(self.generated, [100, 103])


    def test_protected_seeds_and_previously_used_parent_rows_are_refused_early(self):
        plan = dict(self.plan, protected_seeds=[[101, 102]])
        with self.assertRaisesRegex(ValueError, 'protected evaluation'):
            material.validate_plan(plan)
        # Move an old seed into a new round and supply a different claimed prefix.
        old_ref = self.plan['shards'][0]['existing']
        self.plan['shards'][0]['seeds'] = [130, 134]
        parent = make_shard(self.root / 'misplaced.npz', 130, 134, 0)
        self.plan['shards'][0]['existing'] = self.artifact(parent)
        self.plan['shards'][1]['seeds'] = [90, 93]
        self.plan_ref = self.write_plan(self.plan)
        self.assertEqual(self.plan['parent_pool'], old_ref)
        with self.assertRaisesRegex(ValueError, 'reuses the parent roster'):
            self.publish(0, 1)
        self.assertFalse(material.manifest_path(self.plan, 0).exists())
        self.assertEqual(self.generated, [])

    def test_wrong_seed_file_with_matching_sha_cannot_receive_verification(self):
        self.publish(0, 1)
        ref, doc = material.bound_manifest(self.plan, self.plan_ref['sha256'], 0, None, producer.read_bytes)
        path = material.verification_path(self.plan, 0)
        path.unlink()
        path.parent.rmdir()
        shard = doc['shards'][0]['artifact']
        source = Path(shard['path'])
        source.unlink()
        wrong = make_shard(source, 200, 203, 4)
        doc['shards'][0]['artifact'] = self.artifact(wrong)
        bad_binding = material.encode(doc['shards'][0]['artifact'])
        (source.parent / 'd.json').write_bytes(bad_binding)
        producer.shard_receipt(self.plan, 4).write_bytes(bad_binding)
        manifest = Path(ref['path'])
        manifest.write_bytes(material.encode(doc))
        with self.assertRaisesRegex(ValueError, 'seed ledger differs'):
            self.publish(0, 1)
        self.assertFalse(path.exists())

    def test_missing_admitted_shard_prevents_finalization_and_cannot_regenerate(self):
        self.publish(0, 1)
        path = material.shard_path(self.plan, self.plan['shards'][1])
        path.unlink()
        (path.parent / 'd.json').unlink()
        path.parent.rmdir()
        with self.assertRaises(FileNotFoundError):
            self.publish(1, 2)
        self.assertEqual(self.generated.count(100), 1)
        self.assertFalse((Path(self.plan['root']) / 'z' / 'd.json').exists())

    def test_corrupt_remote_shard_is_a_material_wait(self):
        with patch.object(supervisor.subprocess, 'run', return_value=SimpleNamespace(
            returncode=1, stdout='/x: FAILED\n', stderr='sha256sum: WARNING: 1 computed checksum did NOT match\n'
        )):
            with self.assertRaisesRegex(ValueError, 'material artifact could not be verified'):
                supervisor.verify_material_refs([dict(path='/x', sha256='a' * 64)])

    def test_failed_producer_admission_can_reconcile_after_manual_publication(self):
        controller, _ = self.controller()
        controller.ledger['producer'] = dict(job='v2m-c01d1e-cpu', submitted_at='2020-01-01T00:00:00Z',
                                             start=0, stop=2, launcher_exit=1)
        with patch.object(supervisor, 'material_bytes', side_effect=producer.read_bytes), patch.object(
            supervisor, 'missing_material', side_effect=lambda paths: {str(path) for path in paths if not path.exists()}
        ), patch.object(supervisor, 'read_resource', return_value=None):
            controller.advance_producer()
            self.assertNotIn('producer', controller.ledger)
            self.assertIn('reconciled', controller.ledger['producer_failure']['reason'])
            self.publish(0, 2)
            controller.advance_producer()
            self.assertNotIn('producer_failure', controller.ledger)


    def test_incorrect_compact_pool_cannot_receive_verification(self):
        self.publish(0, 1)
        ref, doc = material.bound_manifest(self.plan, self.plan_ref['sha256'], 0, None, producer.read_bytes)
        path = material.verification_path(self.plan, 0)
        path.unlink()
        path.parent.rmdir()
        pool = Path(doc['pool']['path'])
        pool.unlink()
        replacement = make_shard(pool, 100, 106, 4, value=37)
        doc['pool'] = self.artifact(replacement)
        doc['gender_policy'] = self.make_policy(doc['pool'], None, Path(doc['gender_policy']['path']))
        Path(ref['path']).write_bytes(material.encode(doc))
        with self.assertRaisesRegex(ValueError, 'segment pool differs from its verified shards'):
            self.publish(0, 1)
        self.assertFalse(path.exists())

    def test_final_receipt_rejects_changed_digest(self):
        self.publish(0, 2)
        path = Path(self.plan['root']) / 'z' / 'd.json'
        document = json.loads(path.read_bytes())
        document['pool_sha256'] = '0' * 64
        path.write_bytes(material.encode(document))
        with self.assertRaisesRegex(ValueError, 'final material receipt differs'):
            self.publish(1, 2)


    def test_declared_reservation_and_inventory_ranges_cannot_be_shortened(self):
        with self.assertRaisesRegex(ValueError, 'first new training row'):
            material.validate_plan(dict(self.plan, preserved_team_rows=2))
        ref = self.write_plan(dict(self.plan, protected_seeds=[[119, 120]]))
        with self.assertRaisesRegex(ValueError, 'differ from their pinned inventories'):
            material.load_plan(ref, producer.read_bytes)
        ref = self.write_plan(dict(self.plan, protected_seed_inventories=[]))
        with self.assertRaisesRegex(ValueError, 'inventories are required'):
            material.load_plan(ref, producer.read_bytes)

    def test_compact_pool_with_optional_constructor_field_is_refused(self):
        self.publish(0, 1)
        ref, doc = material.bound_manifest(self.plan, self.plan_ref['sha256'], 0, None, producer.read_bytes)
        verified = material.verification_path(self.plan, 0)
        verified.unlink()
        verified.parent.rmdir()
        pool = Path(doc['pool']['path'])
        with np.load(pool, allow_pickle=False) as archive:
            arrays = dict(archive)
        np.savez_compressed(pool, **arrays, happiness=np.zeros((6, 6), dtype=np.uint8))
        doc['pool'] = self.artifact(pool)
        doc['gender_policy'] = self.make_policy(doc['pool'], None, Path(doc['gender_policy']['path']))
        Path(ref['path']).write_bytes(material.encode(doc))
        with self.assertRaisesRegex(ValueError, 'unexpected field'):
            self.publish(0, 1)
        self.assertFalse(verified.exists())

    def test_recovery_will_not_verify_another_segments_gender_policy(self):
        with patch.object(producer, 'finalize'):
            self.publish(0, 2)
        first, previous = material.bound_manifest(self.plan, self.plan_ref['sha256'], 0, None, producer.read_bytes)
        ref, doc = material.bound_manifest(self.plan, self.plan_ref['sha256'], 1, first['sha256'], producer.read_bytes)
        verified = material.verification_path(self.plan, 1)
        verified.unlink()
        verified.parent.rmdir()
        path = Path(doc['gender_policy']['path'])
        path.write_bytes(Path(previous['gender_policy']['path']).read_bytes())
        doc['gender_policy'] = self.artifact(path)
        Path(ref['path']).write_bytes(material.encode(doc))
        with self.assertRaisesRegex(ValueError, 'different constructor material'):
            self.publish(1, 2)
        self.assertFalse(verified.exists())

    def test_conflicting_shared_shard_never_gets_a_verification_receipt(self):
        self.plan['windows'][1]['rows'] = [8, 14]
        self.plan_ref = self.write_plan(self.plan)
        with patch.object(producer, 'finalize'):
            self.publish(0, 2)
        first, _ = material.bound_manifest(self.plan, self.plan_ref['sha256'], 0, None, producer.read_bytes)
        ref, doc = material.bound_manifest(self.plan, self.plan_ref['sha256'], 1, first['sha256'], producer.read_bytes)
        verified = material.verification_path(self.plan, 1)
        verified.unlink()
        verified.parent.rmdir()
        doc['shards'][0]['artifact']['sha256'] = '0' * 64
        Path(ref['path']).write_bytes(material.encode(doc))
        with self.assertRaisesRegex(ValueError, 'disagree about a shared shard'):
            self.publish(1, 2)
        self.assertFalse(verified.exists())

    def test_lost_verified_manifest_refuses_to_regenerate_shards(self):
        self.publish(0, 1)
        path = material.shard_path(self.plan, self.plan['shards'][1])
        path.unlink()
        (path.parent / 'd.json').unlink()
        path.parent.rmdir()
        material.manifest_path(self.plan, 0).unlink()
        with self.assertRaisesRegex(ValueError, 'published manifest is missing'):
            self.publish(0, 1)
        self.assertEqual(self.generated.count(100), 1)
        self.assertFalse(path.exists())


    def test_lost_segment_receipts_cannot_recreate_previously_admitted_shards(self):
        self.plan['windows'][1]['rows'] = [8, 14]
        self.plan_ref = self.write_plan(self.plan)
        self.publish(0, 2)
        previous = material.segment_directory(self.plan, 0)
        for member in ('b.npz', 'c.json', 'm.json'):
            (previous / member).unlink()
        previous.rmdir()
        verified = material.verification_path(self.plan, 0)
        verified.unlink()
        verified.parent.rmdir()
        shared = material.shard_path(self.plan, self.plan['shards'][2])
        shared.unlink()
        (shared.parent / 'd.json').unlink()
        shared.parent.rmdir()
        with self.assertRaisesRegex(ValueError, 'published manifest is missing'):
            self.publish(0, 1)
        self.assertFalse(shared.exists())
        self.assertEqual(self.generated.count(103), 1)

    def test_verification_refuses_recompressed_shard_bound_to_other_bytes(self):
        self.publish(0, 1)
        ref, doc = material.bound_manifest(self.plan, self.plan_ref['sha256'], 0, None, producer.read_bytes)
        verified = material.verification_path(self.plan, 0)
        verified.unlink()
        verified.parent.rmdir()
        path = Path(doc['shards'][0]['artifact']['path'])
        with np.load(path, allow_pickle=False) as archive:
            arrays = dict(archive)
        np.savez(path, **arrays)
        doc['shards'][0]['artifact'] = self.artifact(path)
        (path.parent / 'd.json').write_bytes(material.encode(doc['shards'][0]['artifact']))
        Path(ref['path']).write_bytes(material.encode(doc))
        with self.assertRaisesRegex(ValueError, 'publication receipt'):
            self.publish(0, 1)
        self.assertFalse(verified.exists())

    def test_recovery_checks_the_constructor_gender_code_types(self):
        self.publish(0, 1)
        ref, doc = material.bound_manifest(self.plan, self.plan_ref['sha256'], 0, None, producer.read_bytes)
        verified = material.verification_path(self.plan, 0)
        verified.unlink()
        verified.parent.rmdir()
        path = Path(doc['gender_policy']['path'])
        policy = json.loads(path.read_bytes())
        policy['codes'][0] = -1.0
        path.write_bytes(material.encode(policy))
        doc['gender_policy'] = self.artifact(path)
        Path(ref['path']).write_bytes(material.encode(doc))
        with self.assertRaisesRegex(ValueError, 'invalid species codes'):
            self.publish(0, 1)
        self.assertFalse(verified.exists())

    def test_parent_array_inventory_is_checked_before_first_publication(self):
        path = Path(self.parent['path'])
        with np.load(path, allow_pickle=False) as archive:
            arrays = dict(archive)
        np.savez_compressed(path, **arrays, happiness=np.zeros((4, 6), dtype=np.uint8))
        self.plan['parent_pool'] = self.plan['shards'][0]['existing'] = self.artifact(path)
        self.plan_ref = self.write_plan(self.plan)
        with self.assertRaisesRegex(ValueError, 'parent constructor pool has an incompatible array inventory'):
            self.publish(0, 1)
        self.assertEqual(self.generated, [])


    def test_closing_receipt_prevents_rebuilding_a_lost_final_segment(self):
        self.publish(0, 2)
        segment = material.segment_directory(self.plan, 1)
        for member in ('b.npz', 'c.json', 'm.json'):
            (segment / member).unlink()
        segment.rmdir()
        verified = material.verification_path(self.plan, 1)
        verified.unlink()
        verified.parent.rmdir()
        shard = material.shard_path(self.plan, self.plan['shards'][3])
        shard.unlink()
        (shard.parent / 'd.json').unlink()
        shard.parent.rmdir()
        receipt = producer.shard_receipt(self.plan, 10)
        receipt.unlink()
        receipt.parent.rmdir()
        with self.assertRaisesRegex(ValueError, 'published manifest is missing'):
            self.publish(1, 2)
        self.assertFalse(segment.exists())
        self.assertFalse(shard.exists())
        self.assertEqual(self.generated.count(108), 1)


    def test_closing_receipt_refuses_to_verify_a_changed_final_manifest(self):
        self.publish(0, 2)
        path = material.manifest_path(self.plan, 1)
        doc = json.loads(path.read_bytes())
        doc['unknown'] = 'changed after finalization'
        path.write_bytes(material.encode(doc))
        verified = material.verification_path(self.plan, 1)
        verified.unlink()
        verified.parent.rmdir()
        with self.assertRaisesRegex(ValueError, 'final material receipt differs'):
            self.publish(1, 2)
        self.assertFalse(verified.exists())


    def test_verified_rerun_checks_both_shard_receipts(self):
        self.publish(0, 1)
        local = material.shard_path(self.plan, self.plan['shards'][1]).parent / 'd.json'
        original = json.loads(local.read_bytes())
        local.write_bytes(material.encode(dict(original, sha256='0' * 64)))
        with self.assertRaisesRegex(ValueError, 'publication receipt'):
            self.publish(0, 1)
        self.assertEqual(self.generated.count(100), 1)

    def test_missing_final_pool_prevents_reissuing_segment_verification(self):
        self.publish(0, 2)
        verified = material.verification_path(self.plan, 1)
        verified.unlink()
        verified.parent.rmdir()
        final_pool = Path(self.plan['root']) / 'z' / 't.npz'
        final_pool.unlink()
        with self.assertRaisesRegex(FileNotFoundError, 't.npz'):
            self.publish(1, 2)
        self.assertFalse(verified.exists())
        self.assertEqual(self.generated.count(108), 1)
