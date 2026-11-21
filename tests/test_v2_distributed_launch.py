from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import queue
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
        first = UUID('11111111-1111-4111-8111-111111111111')
        second = 'GPU-22222222-2222-4222-8222-222222222222'
        devices = [SimpleNamespace(uuid=first), SimpleNamespace(uuid=second)]
        torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: len(devices),
                                                      get_device_properties=devices.__getitem__))
        process = Mock(returncode=0)
        process.wait.return_value = 0
        process.poll.return_value = 0
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'run'
            with (patch.dict(sys.modules, torch=torch),
                  patch.object(v2_distributed.subprocess, 'Popen', return_value=process) as spawn):
                rc = v2_distributed.main(['--ranks', '2', '--timeout-seconds', '10', '--out', str(out)])
        self.assertEqual(rc, 0)
        actual = [(call.args[0][call.args[0].index('--rank')+1], call.kwargs['env']['CUDA_VISIBLE_DEVICES'])
                  for call in spawn.call_args_list]
        self.assertEqual(actual, [('0', f'GPU-{first}'), ('1', second)])
