from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import queue
import threading
import time
import unittest
from unittest.mock import Mock, patch
from uuid import UUID

from gpu_engine import v2_distributed


class DistributedLaunchTests(unittest.TestCase):
    def test_supervisor_drains_coordinated_checkpoint_error_exits(self):
        completed = queue.Queue()
        code = v2_distributed.CHECKPOINT_WRITE_EXIT
        for rank in (0, 3, 1, 2):
            completed.put((rank, code))
        self.assertEqual(v2_distributed._wait_workers(completed, 4, time.monotonic()+2),
                         dict.fromkeys(range(4), code))

    def test_workers_receive_valid_cuda_selectors_in_visible_device_order(self):
        import torch

        first = UUID('11111111-1111-4111-8111-111111111111')
        second = 'GPU-22222222-2222-4222-8222-222222222222'
        devices = [SimpleNamespace(uuid=first), SimpleNamespace(uuid=second)]
        process = Mock(returncode=0)
        process.wait.return_value = 0
        process.poll.return_value = 0
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'run'
            with (patch.object(torch.cuda, 'device_count', return_value=len(devices)),
                  patch.object(torch.cuda, 'get_device_properties', side_effect=devices.__getitem__),
                  patch.object(v2_distributed.subprocess, 'Popen', return_value=process) as spawn,
                  patch.object(v2_distributed, '_observe_idle') as observe):
                rc = v2_distributed.main(['--ranks', '2', '--timeout-seconds', '10', '--out', str(out),
                                          '--config', 'unused', '--reference-run', 'unused'])
        self.assertEqual(rc, 0)
        observe.assert_not_called()
        actual = [(call.args[0][call.args[0].index('--rank')+1], call.kwargs['env']['CUDA_VISIBLE_DEVICES'])
                  for call in spawn.call_args_list]
        self.assertEqual(actual, [('0', f'GPU-{first}'), ('1', second)])


class IdleGuardTests(unittest.TestCase):
    def test_full_joint_idle_window_is_required(self):
        evidence = v2_distributed._IdleEvidence(100)
        ranks = [(1, 20, (10, 0)), (2, 20, (10, 0))]
        cache = (3, 20)
        for now in range(0, 1201, 15):
            evidence.observe_gpu(now, True)
            if now % 300 == 0:
                evidence.observe_progress(now, cache, ranks)
                self.assertEqual(evidence.stalled(now, started=0), now > 900)
        self.assertFalse(evidence.stalled(1200, started=400))

    def test_activity_and_unknown_samples_reset_the_idle_window(self):
        cases = ('busy-gpu', 'missing-gpu', 'cpu', 'cache', 'log', 'pid', 'missing-progress', 'timing-gap')
        for interruption in cases:
            with self.subTest(interruption=interruption):
                evidence = v2_distributed._IdleEvidence(100)
                ranks = [(1, 0, (0, 0)), (2, 0, (0, 0))]
                cache = (3, 20)
                for now in range(0, 1201, 15):
                    if interruption != 'timing-gap' or not 570 <= now < 600:
                        evidence.observe_gpu(now, now != 600 or interruption not in ('busy-gpu', 'missing-gpu'))
                    if now == 600 and interruption == 'cpu':
                        ranks[0] = (1, 300, (0, 0))
                    if now == 600 and interruption == 'log':
                        ranks[0] = (1, 0, (1, 1))
                    if now == 600 and interruption == 'pid':
                        ranks[0] = (3, 0, (0, 0))
                    if now == 600 and interruption == 'cache':
                        cache = (4, 30)
                    if now % 300 == 0:
                        sample = None if now == 600 and interruption == 'missing-progress' else cache
                        evidence.observe_progress(now, sample, list(ranks))
                        self.assertFalse(evidence.stalled(now, started=0))

    def test_allocated_gpu_query_checks_presence_and_uniqueness(self):
        devices = ['GPU-a', 'GPU-b']
        with patch.object(v2_distributed.subprocess, 'check_output', return_value='GPU-a, 0\nGPU-b, 0\nGPU-c, 80'):
            self.assertTrue(v2_distributed._gpu_idle(devices))
        for output in ('GPU-a, 0', 'GPU-a, 0\nGPU-b, 0\nGPU-b, 0'):
            with (self.subTest(output=output),
                  patch.object(v2_distributed.subprocess, 'check_output', return_value=output)):
                with self.assertRaises(ValueError):
                    v2_distributed._gpu_idle(devices)

    def test_observer_timeout_leaves_the_completion_queue_empty(self):
        completed = queue.Queue()
        process = Mock(**{'poll.return_value': None})
        retire = threading.Event()
        now = time.monotonic()
        with (patch.object(v2_distributed, '_gpu_idle', side_effect=subprocess.TimeoutExpired('nvidia-smi', 8)),
              patch.object(v2_distributed, '_cache_progress', return_value=(1, 0)),
              patch.object(v2_distributed, '_rank_progress', return_value=(1, 0, (0, 0))),
              patch.object(retire, 'wait', side_effect=lambda _: retire.set())):
            v2_distributed._observe_idle(completed, [process], [Mock()], ['GPU-a'], '/cache', now, retire)
        self.assertTrue(completed.empty())

    def test_idle_candidate_retires_only_running_workers(self):
        completed = queue.Queue()
        completed.put(('idle', time.monotonic()))
        processes = [Mock(**{'poll.return_value': None}) for _ in range(2)]
        self.assertEqual(v2_distributed._wait_workers(completed, 2, time.monotonic() + 2, processes=processes),
                         {'idle': 124})

    def test_idle_candidate_cannot_interrupt_checkpoint_exit_drain(self):
        completed = queue.Queue()
        code = v2_distributed.CHECKPOINT_WRITE_EXIT
        completed.put((0, code))
        completed.put(('idle', time.monotonic()))
        completed.put((1, code))
        processes = [Mock(**{'poll.return_value': None}) for _ in range(2)]
        retire = threading.Event()
        self.assertEqual(v2_distributed._wait_workers(
            completed, 2, time.monotonic() + 2, processes=processes, retire=retire,
        ), {0: code, 1: code})
        self.assertTrue(retire.is_set())
