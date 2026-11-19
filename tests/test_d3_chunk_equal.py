"""D3 chunk gate admission with synthetic declarations, no collection or model forward."""

from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gpu_engine import d3_chunk_equal
from pokezero import checkpoint_self_moves, v5_contract, v5_run
from test_v5_run import plan


class ChunkGateAdmissionTests(unittest.TestCase):
    def legacy_origin(self, directory, recipe, *, tamper):
        directory.mkdir()
        journal = directory / 'journal'
        journal.mkdir()
        declaration = replace(recipe, self_moves=False).manifest()
        (directory / 'declaration.json').write_text(json.dumps(declaration))
        contract = v5_contract.contract_descriptor()
        metadata = dict(schema=checkpoint_self_moves.CHECKPOINT_SCHEMA,
            training_schema=v5_contract.TRAINING_CONTRACT_SCHEMA,
            observation_schema=v5_contract.OBSERVATION_SCHEMA_VERSION_V5,
            source_contract=contract, source_contract_sha256=checkpoint_self_moves._sha(contract),
            model=declaration['model'], loss=declaration['loss'], training=declaration['train'],
            steps=0, rows=0, progress=None)
        checkpoint = (directory / 'origin.pt')
        checkpoint.write_text(json.dumps(dict(metadata=metadata, metadata_sha256=checkpoint_self_moves._sha(metadata))))
        archive = directory / 'origin.npz'
        archive.write_bytes(b'legacy archive')
        snapshot = v5_run.Snapshot(-1, checkpoint.name, hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            archive.name, hashlib.sha256(archive.read_bytes()).hexdigest(), checkpoint_self_moves._sha(metadata['model']))
        previous = None
        for sequence, (kind, payload) in enumerate((
                ('RUN_STARTED', {'declaration_sha256': checkpoint_self_moves._sha(declaration)}),
                ('CHECKPOINT', dict(iteration=-1, path=snapshot.checkpoint, sha256=snapshot.checkpoint_sha256, steps=0, rows=0)),
                ('SNAPSHOT', asdict(snapshot)))):
            event = dict(sequence=sequence, previous=previous, kind=kind, payload=payload)
            previous = checkpoint_self_moves._sha(event)
            (journal / f'{sequence:09d}.json').write_text(json.dumps(dict(event, sha256=previous)))
        if tamper:
            archive.write_bytes(b'substituted archive')
        return replace(snapshot, checkpoint='initial.pt', archive='initial.npz')

    def run_gate(self, recipe, metadata, *, document=None, origin=False, tamper=False,
                 error=RuntimeError, message='collector reached'):
        snapshot = v5_run.Snapshot(-1, 'initial.pt', 'b' * 64, 'initial.npz', 'c' * 64, 'd' * 64)
        params = object()
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            run = root / 'run'
            args = ['--d3-root', str(root), '--out', str(root / 'out'), '--variant', 'batched_prep']
            if origin:
                original_run = root / 'original'
                snapshot = self.legacy_origin(original_run, recipe, tamper=tamper)
                metadata = dict(metadata, checkpoint_sha256=snapshot.checkpoint_sha256,
                                weights_archive_sha256=snapshot.archive_sha256)
                args.extend(['--initializer-run', str(original_run)])
                # Deserialize a metadata-only fixture without Torch. The production
                # verifier still checks both file digests and the original journal.
                def load_checkpoint(stream, *, map_location, weights_only):
                    self.assertEqual(map_location, 'cpu')
                    self.assertIs(weights_only, True)
                    return json.load(stream)
                stack.enter_context(patch.dict(sys.modules, torch=SimpleNamespace(load=load_checkpoint)))
            journal = run / 'journal'
            journal.mkdir(parents=True)
            (run / 'declaration.json').write_text(json.dumps(recipe.manifest() if document is None else document))
            (journal / '000000000.json').write_text(json.dumps(dict(
                kind='DECLARATION', payload=dict(declaration_sha256=v5_run._sha(recipe.manifest())))))
            (journal / '000000001.json').write_text(json.dumps(dict(kind='SNAPSHOT', payload=asdict(snapshot))))
            (root / 'runtime.json').write_text('{}')
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(patch('pokezero.v5_worker._record', return_value={'constructor_pool': {'sha256': recipe.pool_sha256}}))
            stack.enter_context(patch('pokezero.v5_worker._runtime_metadata', return_value=(None, {}, None, None)))
            stack.enter_context(patch('pokezero.v5_worker._probe'))
            prepared = SimpleNamespace(source_sha256='f' * 64, engine_tables=object(), observation_tables=object())
            prepare = stack.enter_context(patch('pokezero.v5_worker.prepare_device_runtime', return_value=(prepared, None, None, None)))
            load = stack.enter_context(patch('gpu_engine.v5_model.load', return_value=(params, recipe.model, metadata)))
            provider = stack.enter_context(patch('gpu_engine.v5_device_teacher.Provider'))
            backend = stack.enter_context(patch('gpu_engine.v5_device_backend.RuntimeBackend')).return_value
            backend.declaration_sha256 = 'e' * 64
            stack.enter_context(patch('gpu_engine.v5_device_rollout.Bindings'))
            stack.enter_context(patch('gpu_engine.v5_device_rollout.actor_bank_from_assignments'))
            collector = stack.enter_context(patch('gpu_engine.v5_device_rollout.make_collector',
                                                  side_effect=RuntimeError('collector reached')))
            # Use real plan admission and control flow, stopping as collection would begin.
            with self.assertRaisesRegex(error, message):
                d3_chunk_equal.main(args)
            if error is RuntimeError:
                collector.assert_called_once()
                self.assertEqual(prepare.call_args.args[0].plan, recipe)
                self.assertEqual(provider.call_count, 2)
                self.assertTrue(all(call.args[1] == recipe for call in provider.call_args_list))
                backend.check_plan.assert_called_once_with(recipe)
                load.assert_called_once()
                self.assertEqual(load.call_args.args[0], run / snapshot.archive)
                self.assertEqual(load.call_args.kwargs['expected_archive_sha256'], snapshot.archive_sha256)
                self.assertEqual(load.call_args.kwargs['expected_checkpoint_sha256'], snapshot.checkpoint_sha256)
            else:
                collector.assert_not_called()
                provider.assert_not_called()

    def test_missing_and_true_plan_flags_reach_the_collector_with_matching_actors(self):
        off = plan(arm='D')
        self.run_gate(off, {})
        self.run_gate(replace(off, self_moves=True), {'self_moves': True})

    def test_malformed_or_unbound_plan_flags_refuse(self):
        off = plan(arm='D')
        for value in (False, 1, 'true', None):
            with self.subTest(value=value):
                self.run_gate(off, {}, document=dict(off.manifest(), self_moves=value),
                              error=ValueError, message='omitted or declared as true')
        self.run_gate(off, {}, document=dict(off.manifest(), self_moves=True),
                      error=SystemExit, message='rebuilt plan differs')

    def test_actor_and_plan_mismatch_refuses_before_collection(self):
        off = plan(arm='D')
        self.run_gate(off, {'self_moves': True}, error=ValueError, message='differs from the plan')
        self.run_gate(replace(off, self_moves=True), {}, error=ValueError, message='requires --initializer-run')

    def test_off_initializer_requires_original_archive_and_checkpoint_evidence(self):
        on = replace(plan(arm='D'), self_moves=True)
        self.run_gate(on, {}, origin=True)
        self.run_gate(on, {}, origin=True, tamper=True,
                      error=ValueError, message='original initializer archive binding differs')


if __name__ == '__main__':
    unittest.main()
