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

from gpu_engine import d3_chunk_equal, d3_depth2_gate, d3_depthn_gate
from pokezero import checkpoint_self_moves, v5_contract, v5_run
from test_v5_run import plan


class ChunkGateAdmissionTests(unittest.TestCase):
    def write_events(self, journal, events):
        previous = None
        for sequence, (kind, payload) in enumerate(events):
            event = dict(sequence=sequence, previous=previous, kind=kind, payload=payload)
            previous = checkpoint_self_moves._sha(event)
            (journal / f'{sequence:09d}.json').write_text(json.dumps(dict(event, sha256=previous)))

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
        self.write_events(journal, (
                ('RUN_STARTED', {'declaration_sha256': checkpoint_self_moves._sha(declaration)}),
                ('CHECKPOINT', dict(iteration=-1, path=snapshot.checkpoint, sha256=snapshot.checkpoint_sha256, steps=0, rows=0)),
                ('SNAPSHOT', asdict(snapshot))))
        if tamper:
            archive.write_bytes(b'substituted archive')
        return replace(snapshot, checkpoint='initial.pt', archive='initial.npz')

    def run_gate(self, recipe, metadata, *, gate='chunk', document=None, origin=False, tamper=False, model_sha256=None,
                 corrupt=None, error=RuntimeError, message='collector reached'):
        model_sha256 = v5_run._sha(recipe.model.metadata()) if model_sha256 is None else model_sha256
        snapshot = v5_run.Snapshot(-1, 'initial.pt', 'b' * 64, 'initial.npz', 'c' * 64, model_sha256)
        params = object()
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            run = root / 'run'
            args = ['--d3-root', str(root), '--out', str(root / 'out')]
            args.extend(['--variant', 'batched_prep'] if gate == 'chunk' else ['--chunk', '0', '--pricer-only'])
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
            declaration = recipe.manifest() if document is None else document
            (run / 'declaration.json').write_text(json.dumps(declaration))
            events = [('RUN_STARTED', dict(declaration_sha256=v5_run._sha(declaration))), ('SNAPSHOT', asdict(snapshot))]
            if corrupt == 'duplicate':
                events.append(('SNAPSHOT', asdict(snapshot)))
            self.write_events(journal, events)
            if corrupt == 'declaration':
                (run / 'declaration.json').write_text(json.dumps(dict(declaration, contract_sha256='a' * 64)))
            if corrupt == 'snapshot':
                path = journal / '000000001.json'
                event = json.loads(path.read_text())
                event['payload']['archive'] = 'substituted.npz'
                path.write_text(json.dumps(event))
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
                if gate == 'chunk':
                    d3_chunk_equal.main(args)
                elif gate == 'depth':
                    d3_depth2_gate.main(args)
                elif gate in ('depthn', 'champion'):
                    d3_depthn_gate.load_roots(SimpleNamespace(saved_roots=None, d3_root=root, chunk=0,
                        memory_policy=False, initializer_run=None,
                        champion_archive='champion.npz' if gate == 'champion' else None))
                else:
                    d3_depth2_gate.bank_chunk_requests(SimpleNamespace(d3_root=root, memory_policy=False))
            if error is RuntimeError:
                collector.assert_called_once()
                self.assertEqual(prepare.call_args.args[0].plan, recipe)
                self.assertEqual(provider.call_count, 2 if gate == 'chunk' else 1)
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
                              error=ValueError, message='explicit bool run self_moves|omitted or declared as true')
        self.run_gate(off, {}, document=dict(off.manifest(), contract_sha256='a' * 64),
                      error=SystemExit, message='rebuilt plan differs')

    def test_actor_and_plan_mismatch_refuses_before_collection(self):
        off = plan(arm='D')
        self.run_gate(off, {'self_moves': True}, error=ValueError, message='differs from the plan')
        self.run_gate(replace(off, self_moves=True), {}, error=ValueError, message='requires --initializer-run')

    def test_a_different_snapshot_model_digest_refuses_before_collection(self):
        for enabled in (False, True):
            with self.subTest(self_moves=enabled):
                self.run_gate(replace(plan(arm='D'), self_moves=enabled), {'self_moves': enabled},
                    model_sha256='d' * 64, error=ValueError, message='snapshot model digest differs')

    def test_off_initializer_requires_original_archive_and_checkpoint_evidence(self):
        on = replace(plan(arm='D'), self_moves=True)
        self.run_gate(on, {}, origin=True)
        self.run_gate(on, {}, origin=True, tamper=True,
                      error=ValueError, message='original initializer archive binding differs')

    def test_depth_gate_passes_the_bound_plan_to_collection(self):
        off = plan(arm='D')
        on = replace(off, self_moves=True)
        self.run_gate(off, {}, gate='depth')
        self.run_gate(on, {'self_moves': True}, gate='depth')
        self.run_gate(on, {}, gate='depth', origin=True)
        self.run_gate(on, {'self_moves': True}, gate='depthn')

    def test_depth_gate_refuses_incompatible_actors_and_legacy_callers_refuse_on_banks(self):
        off = plan(arm='D')
        on = replace(off, self_moves=True)
        self.run_gate(off, {'self_moves': True}, gate='depth', error=ValueError, message='differs from the plan')
        self.run_gate(on, {}, gate='depth', error=ValueError, message='requires --initializer-run')
        self.run_gate(on, {'self_moves': True}, gate='legacy', error=ValueError, message='not threaded')
        self.run_gate(on, {'self_moves': True}, gate='champion', error=ValueError, message='not threaded')

    def test_depth_gate_verifies_destination_declaration_and_initial_snapshot(self):
        on = replace(plan(arm='D'), self_moves=True)
        for corrupt, message in (('declaration', 'RUN_STARTED original declaration binding differs'),
                                 ('snapshot', 'original ordered run journal changed'),
                                 ('duplicate', 'exactly one initial snapshot')):
            with self.subTest(corrupt=corrupt):
                self.run_gate(on, {}, gate='depth', origin=True, corrupt=corrupt,
                              error=ValueError, message=message)


if __name__ == '__main__':
    unittest.main()
