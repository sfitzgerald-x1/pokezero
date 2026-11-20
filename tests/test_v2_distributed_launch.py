"""Per-node supervisor: rendezvous across nodes, all-nodes-or-nothing, topology guard.

The two-node tests run the real supervisor twice as subprocesses (2 "nodes" x 4
ranks) over a loopback TCP rendezvous with the gloo backend and a collective
probe instead of the fused worker; they need torch and are skipped without it.
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import queue
import unittest
from unittest.mock import Mock, patch
from uuid import UUID

from gpu_engine import v2_distributed

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def supervisor(out, node, nodes, port, *, ranks=4, rendezvous_timeout=60, timeout=120):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / 'src'), str(ROOT)]), JAX_PLATFORMS='cpu')
    command = [sys.executable, '-m', 'gpu_engine.v2_distributed', '--ranks', str(ranks), '--nodes', str(nodes),
               '--node-index', str(node), '--master', f'127.0.0.1:{port}', '--backend', 'gloo', '--probe',
               '--timeout-seconds', str(timeout), '--rendezvous-timeout', str(rendezvous_timeout),
               '--collective-timeout', '60', '--out', str(out)]
    return subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


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
        torch_stub = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: len(devices),
                                                           get_device_properties=devices.__getitem__))
        process = Mock(returncode=0)
        process.wait.return_value = 0
        process.poll.return_value = 0
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'run'
            with (patch.dict(sys.modules, torch=torch_stub),
                  patch.object(v2_distributed.subprocess, 'Popen', return_value=process) as spawn):
                rc = v2_distributed.main(['--ranks', '2', '--timeout-seconds', '10', '--out', str(out)])
            exit_record = json.loads((out / 'rank-exit.json').read_text())
        self.assertEqual(rc, 0)
        actual = [(call.args[0][call.args[0].index('--rank')+1], call.kwargs['env']['CUDA_VISIBLE_DEVICES'])
                  for call in spawn.call_args_list]
        self.assertEqual(actual, [('0', f'GPU-{first}'), ('1', second)])
        self.assertEqual((exit_record['world_size'], exit_record['nodes'], exit_record['ranks_per_node']), (2, 1, 2))
        self.assertTrue(exit_record['master'].startswith('127.0.0.1:'))

    def test_second_node_numbers_global_ranks_and_passes_the_shared_master(self):
        devices = [SimpleNamespace(uuid=f'GPU-{i}{i}{i}{i}{i}{i}{i}{i}-1111-4111-8111-111111111111') for i in range(1, 5)]
        torch_stub = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: len(devices),
                                                           get_device_properties=devices.__getitem__))
        process = Mock(returncode=0); process.wait.return_value = 0; process.poll.return_value = 0
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'run'
            with (patch.dict(sys.modules, torch=torch_stub),
                  patch.object(v2_distributed.subprocess, 'Popen', return_value=process) as spawn):
                rc = v2_distributed.main(['--ranks', '4', '--nodes', '2', '--node-index', '1', '--master', 'node0.example:29500',
                                          '--timeout-seconds', '10', '--out', str(out)])
            record = json.loads((out / 'rank-exit-node-1.json').read_text())
            logs = [(out / f'rank-{r}.log').exists() for r in range(4, 8)]
        self.assertEqual(rc, 0)
        self.assertEqual(logs, [True] * 4)
        commands = [call.args[0] for call in spawn.call_args_list]
        self.assertEqual([c[c.index('--rank') + 1] for c in commands], ['0', '1', '2', '3'])
        self.assertTrue(all(c[c.index('--node-index') + 1] == '1' and c[c.index('--master') + 1] == 'node0.example:29500'
                            and c[c.index('--nodes') + 1] == '2' for c in commands))
        self.assertEqual(sorted(int(k) for k in record['codes']), [4, 5, 6, 7])
        self.assertEqual((record['world_size'], record['nodes'], record['node_index']), (8, 2, 1))

    def test_multi_node_requires_master_and_valid_index(self):
        for argv in (['--ranks', '4', '--nodes', '2', '--timeout-seconds', '10', '--out', 'x'],
                     ['--ranks', '4', '--nodes', '2', '--node-index', '2', '--master', 'h:1', '--timeout-seconds', '10', '--out', 'x'],
                     ['--ranks', '4', '--nodes', '2', '--master', 'nohost', '--timeout-seconds', '10', '--out', 'x'],
                     ['--ranks', '4', '--rendezvous-timeout', '0', '--timeout-seconds', '10', '--out', 'x']):
            with self.assertRaises(ValueError):
                v2_distributed.main(argv)

    def test_resume_refuses_a_different_world_size_with_both_sizes_named(self):
        distribution = v2_distributed.Distribution(0, 8, None, nodes=2, ranks_per_node=4)
        distribution.contract = dict(schema='v2-fused-distributed.v1', world_size=8, other='x')
        saved = dict(distribution=dict(schema='v2-fused-distributed.v1', world_size=4, other='x'))
        with self.assertRaisesRegex(ValueError, r'topology mismatch.*world size 4 \(ddp\).*world size 8 \(2 node\(s\) x 4, ddp\)'):
            distribution.check_resume(saved)
        with self.assertRaisesRegex(ValueError, 'identical'):
            distribution.check_resume(dict(distribution=dict(distribution.contract, other='y')))
        distribution.check_resume(dict(distribution=dict(distribution.contract)))
        self.assertEqual(distribution.placement(), dict(world_size=8, nodes=2, ranks_per_node=4, topology='ddp', learner_world=8))
        learner_node = v2_distributed.Distribution(5, 8, None, nodes=2, topology='learner-node', learner_group=object())
        self.assertEqual((learner_node.learner_world, learner_node.is_learner, learner_node.exchange().learners), (4, False, 4))
        self.assertTrue(v2_distributed.Distribution(3, 8, None, nodes=2, topology='learner-node', learner_group=object()).is_learner)
        learner_node.contract = dict(schema='v2-fused-distributed.v1', world_size=8, topology='learner-node', learner_world=4)
        with self.assertRaisesRegex(ValueError, r'topology mismatch.*\(ddp\).*learner-node'):
            learner_node.check_resume(dict(distribution=dict(schema='v2-fused-distributed.v1', world_size=8)))
        with self.assertRaises(ValueError):
            v2_distributed.Distribution(0, 8, object(), nodes=2, topology='learner-node')   # learner group required
        self.assertEqual((v2_distributed.Distribution(6, 8, None, nodes=2).node_index,
                          v2_distributed.Distribution(6, 8, None, nodes=2).local_rank), (1, 2))


@unittest.skipIf(torch is None, 'torch required')
class TwoNodeRendezvousTests(unittest.TestCase):
    def test_two_nodes_of_four_ranks_rendezvous_over_tcp(self):
        port = free_port()
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'run'
            nodes = [supervisor(out, node, 2, port) for node in (1, 0)]   # node 1 may start before the master
            outputs = [(node.wait(timeout=180), node.stdout.read()) for node in nodes]
            logs = {p.name: p.read_text()[-3000:] for p in sorted(out.glob('rank-*.log'))}
            self.assertEqual([code for code, _ in outputs], [0, 0], (outputs, logs))
            probes = [json.loads((out / f'probe-rank-{r}.json').read_text()) for r in range(8)]
            exits = [json.loads((out / f'rank-exit-node-{n}.json').read_text()) for n in (0, 1)]
        for probe in probes:
            self.assertEqual((probe['world_size'], probe['nodes'], probe['ranks_per_node'], probe['topology']), (8, 2, 4, 'ddp'))
            self.assertEqual([(r['rank'], r['node'], r['local']) for r in probe['ranks']],
                             [(r, r // 4, r % 4) for r in range(8)])
        self.assertEqual([sorted(int(k) for k in e['codes']) for e in exits], [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertTrue(all(code == 0 for e in exits for code in e['codes'].values()))
        bench = probes[0]['bench']
        self.assertGreater(bench['cross_node_bandwidth_gbps'], 0)
        self.assertGreater(bench['all_reduce_ms'], 0)
        self.assertGreater(bench['all_to_all_gbps_per_rank'], 0)
        self.assertNotIn('cross_node_bandwidth_gbps', probes[1]['bench'])   # only the node-0/node-1 pair measures the link

    def test_missing_partner_node_fails_every_rank_within_the_rendezvous_deadline(self):
        port = free_port()
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'run'
            started = time.monotonic()
            node = supervisor(out, 0, 2, port, rendezvous_timeout=5, timeout=120)
            code, output = node.wait(timeout=90), node.stdout.read()
            elapsed = time.monotonic() - started
            record = json.loads((out / 'rank-exit-node-0.json').read_text())
            logs = ''.join((out / f'rank-{r}.log').read_text() for r in range(4))
            self.assertFalse(any((out / f'probe-rank-{r}.json').exists() for r in range(8)))
        self.assertEqual(code, 1, output)
        self.assertLess(elapsed, 60)
        self.assertTrue(any(c not in (0, None) for c in record['codes'].values()), record)
        self.assertIn('could not rendezvous with all 8 ranks', logs)


if __name__ == '__main__':
    unittest.main()
