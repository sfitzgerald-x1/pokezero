"""Explicit D preparation controls at CLI, worker and backend boundaries."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero import v5_phasec as phase, v5_run as runs, v5_worker as worker
from test_v5_run import plan


class BatchedPreparationPlumbingTests(unittest.TestCase):
    def test_backend_preserves_queue_and_only_enables_preparation_explicitly(self):
        from gpu_engine import v5_device_backend, v5_device_teacher
        prepared, pool, expected = object(), object(), object()
        for enabled in (False, True):
            with self.subTest(enabled=enabled), \
                    patch.object(v5_device_teacher, 'Provider') as provider, \
                    patch.object(v5_device_backend, 'RuntimeBackend', return_value=expected) as backend:
                actual = phase.build_backend(plan('D'), prepared, pool,
                    device_rollout=True, device_teacher=True, teacher_frame_batch=True,
                    teacher_vectorized=True, teacher_queue_chunks=2, teacher_batched_prep=enabled)
                self.assertIs(actual, expected)
                self.assertEqual(provider.call_args.kwargs, dict(frame_batch=True,
                    chunk_batch=True, vectorized=True, **({'batched_prep': True} if enabled else {})))
                self.assertIs(backend.call_args.kwargs['teacher'], provider.return_value)
                self.assertEqual(backend.call_args.kwargs['teacher_queue_chunks'], 2)

    def test_backend_refuses_invalid_preparation_before_constructing_runtime(self):
        from gpu_engine import v5_device_backend, v5_device_teacher
        for bad in ({'teacher_batched_prep': 1}, {'teacher_batched_prep': None},
                    {'teacher_batched_prep': True, 'teacher_vectorized': False}):
            with self.subTest(bad=bad), patch.object(v5_device_teacher, 'Provider') as provider, \
                    patch.object(v5_device_backend, 'RuntimeBackend') as backend:
                options = dict(device_rollout=True, device_teacher=True,
                    teacher_frame_batch=True, teacher_vectorized=True)
                options.update(bad)
                with self.assertRaisesRegex(ValueError, 'batched preparation'):
                    phase.build_backend(plan('D'), object(), object(), **options)
                provider.assert_not_called()
                backend.assert_not_called()

    def test_worker_refuses_invalid_preparation_before_source_or_model_access(self):
        inputs = SimpleNamespace(plan=plan('D'))
        for bad in ({'teacher_batched_prep': 1}, {'teacher_batched_prep': None},
                    {'teacher_batched_prep': True, 'teacher_vectorized': False}):
            with self.subTest(bad=bad), patch.object(worker, '_source') as source, \
                    patch.object(worker, 'prepare_device_runtime') as prepare:
                options = dict(device_rollout=True, device_teacher=True,
                    teacher_frame_batch=True, teacher_vectorized=True)
                options.update(bad)
                with self.assertRaisesRegex(ValueError, 'batched preparation'):
                    worker.execute(inputs, Path('/unit/run'), **options)
                source.assert_not_called()
                prepare.assert_not_called()

    def test_worker_forwards_preparation_to_original_backend_and_preserves_run(self):
        declared = replace(plan('D'), evaluation=runs.EvaluationProtocol('c' * 64))
        inputs = SimpleNamespace(plan=declared, runtime={'memory_policy': {}}, gender_policy=None)
        prepared, pool, public, contract, backend = (object() for _ in range(5))
        hook = lambda request: None
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(worker, '_source'), patch.object(worker, '_memory'), \
                patch.object(worker, '_probe'), \
                patch.object(worker, 'prepare_device_runtime', return_value=(prepared, pool, public, contract)), \
                patch.object(phase, 'build_backend', return_value=backend) as build, \
                patch.object(runs, 'run', return_value={'status': 'UNIT'}) as run:
            output = worker.execute(inputs, Path(directory), evaluation_hook=hook,
                device_rollout=True, device_teacher=True, teacher_frame_batch=True,
                teacher_vectorized=True, teacher_queue_chunks=2, teacher_batched_prep=True)
            self.assertEqual(output, {'status': 'UNIT'})
            self.assertTrue(build.call_args.kwargs['teacher_batched_prep'])
            self.assertEqual(build.call_args.kwargs['teacher_queue_chunks'], 2)
            self.assertIs(build.call_args.args[0], declared)
            self.assertEqual(run.call_args.kwargs,
                dict(backend=backend, resume=False, enable_v5=True, evaluation_hook=hook))

    def test_cli_transfers_explicit_option_and_leaves_default_absent(self):
        base = ['--enable-v5', '--arm', 'D0', '--declaration', '/unit/plan',
            '--expected-declaration-sha256', 'a' * 64, '--runtime', '/unit/runtime',
            '--expected-runtime-sha256', 'b' * 64, '--run-root', '/unit/run',
            '--device-rollout', '--device-teacher', '--teacher-frame-batch', '--teacher-vectorized']
        for enabled in (False, True):
            with self.subTest(enabled=enabled), patch.object(worker, 'load_inputs', return_value=object()), \
                    patch.object(worker, 'execute', return_value={'status': 'COMPLETE_DECLARATION'}) as execute, \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(worker.main(base + (['--teacher-batched-prep'] if enabled else [])), 0)
                self.assertEqual(execute.call_args.kwargs.get('teacher_batched_prep', False), enabled)
                if not enabled:
                    self.assertNotIn('teacher_batched_prep', execute.call_args.kwargs)


if __name__ == '__main__':
    unittest.main()
