"""Authored device-counter/clock fixtures, not GPU performance evidence."""
import ctypes
import subprocess
from types import SimpleNamespace
import unittest
from unittest import mock

from gpu_engine import device_telemetry as t


UUID = 'GPU-00112233-4455-6677-8899-aabbccddeeff'


class TelemetryTests(unittest.TestCase):
    def test_counter_identity_finite_values_and_ranges(self):
        self.assertEqual(t.parse_counters(UUID+', 50, 100, 200\n', UUID),
            dict(utilization_percent=50, memory_used_mib=100, memory_total_mib=200))
        for output in ('', UUID+', N/A, 1, 2', UUID+', nan, 1, 2',
                       UUID+', 101, 1, 2', UUID+', 1, -1, 2', UUID+', 1, 3, 2',
                       UUID+', 1, 0, 0', 'GPU-other, 1, 1, 2',
                       (UUID+', 1, 1, 2\n')*2):
            with self.subTest(output=output), self.assertRaises(ValueError):
                t.parse_counters(output, UUID)

    def test_binding_uses_cuda_uuid_not_nvidia_physical_ordinal(self):
        cuda = mock.Mock()
        cuda.cuInit.return_value = cuda.cuDeviceGet.return_value = 0
        def count(out):
            ctypes.cast(out, ctypes.POINTER(ctypes.c_int))[0] = 1
            return 0
        def identifier(out, handle):
            out[:] = bytes.fromhex('00112233445566778899aabbccddeeff')
            return 0
        cuda.cuDeviceGetCount.side_effect = count
        cuda.cuDeviceGetUuid_v2.side_effect = identifier
        device = SimpleNamespace(platform='gpu', local_hardware_id=0)
        with mock.patch.object(t.ctypes, 'CDLL', return_value=cuda):
            self.assertEqual(t.cuda_device_uuid(device), UUID)
            cuda.cuInit.return_value = 3
            with self.assertRaisesRegex(RuntimeError, 'cuInit'):
                t.cuda_device_uuid(device)
        for d in (SimpleNamespace(platform='cpu', local_hardware_id=0),
                  SimpleNamespace(platform='gpu', local_hardware_id=1)):
            with self.assertRaises(RuntimeError):t.cuda_device_uuid(d)
        cuda.cuInit.return_value = 0
        cuda.cuDeviceGetCount.side_effect = lambda out: 0  # count remains zero
        with mock.patch.object(t.ctypes, 'CDLL', return_value=cuda), self.assertRaisesRegex(RuntimeError, 'exactly one'):
            t.cuda_device_uuid(device)

    def test_boundaries_failures_gaps_and_unavailable_are_not_zero(self):
        def sample(a, b, util=40):
            return dict(start_monotonic=a, end_monotonic=b, status='OK',
                        utilization_percent=util, memory_used_mib=100+util)
        samples = [sample(0, .2, 100), sample(1.1, 1.2, 20),
                   sample(2.1, 2.2, 60), sample(2.9, 3.1, 100)]
        result = t.summarize(samples, 1, 3, cadence=1, timeout=.5, worker_stopped=True)
        self.assertEqual(result['status'], 'SAMPLED')
        self.assertEqual(result['mean_reported_utilization_percent'], 40)
        self.assertEqual(result['maximum_sampled_device_used_memory_mib'], 160)
        self.assertEqual(result['boundary_excluded_samples'], 2)
        self.assertEqual(result['samples'], samples)
        failed = dict(start_monotonic=2.3, end_monotonic=2.8, status='TimeoutExpired')
        result = t.summarize(samples+[failed], 1, 3, cadence=1, timeout=.5, worker_stopped=True)
        self.assertEqual((result['status'], result['failed_queries']), ('PARTIAL', 1))
        result = t.summarize([failed], 1, 3, cadence=1, timeout=.5, worker_stopped=True)
        self.assertEqual(result['status'], 'UNAVAILABLE')
        self.assertIsNone(result['mean_reported_utilization_percent'])
        self.assertIsNone(result['maximum_sampled_device_used_memory_mib'])
        result = t.summarize([sample(1.1, 1.2)], 1, 10, cadence=1, timeout=.5, worker_stopped=True)
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertEqual(result['maximum_sample_gap_seconds'], 8.8)

    def test_background_query_is_uuid_bound_timed_and_stops(self):
        with mock.patch.object(t.subprocess, 'check_output', return_value=UUID+', 50, 100, 200') as command:
            window = t.DeviceWindow(UUID)
            start = window.start()
            window._thread.join(timeout=.01)
            result = window.finish(t.time.perf_counter())
        self.assertTrue(result['worker_stopped'])
        self.assertGreaterEqual(result['query_count'], 1)
        self.assertEqual(result['device_uuid'], UUID)
        self.assertIn('--id='+UUID, command.call_args.args[0])
        self.assertEqual(command.call_args.kwargs['timeout'], .5)
        self.assertGreaterEqual(result['window_end_monotonic'], start)
        with self.assertRaises(RuntimeError):window.start()

    def test_missing_executable_timeout_invalid_values_remain_failures(self):
        for error in (FileNotFoundError(), subprocess.TimeoutExpired('nvidia-smi', .5),
                      subprocess.CalledProcessError(1, 'nvidia-smi'), ValueError()):
            with self.subTest(error=error), mock.patch.object(t.subprocess, 'check_output', side_effect=error):
                window = t.DeviceWindow(UUID)
                window.start()
                window._thread.join(timeout=.01)
                result = window.finish(t.time.perf_counter())
                self.assertTrue(result['worker_stopped'])
                self.assertEqual(result['status'], 'UNAVAILABLE')
                self.assertGreater(result['failed_queries'], 0)
                self.assertIsNone(result['maximum_sampled_device_used_memory_mib'])
        for options in (dict(cadence=0), dict(timeout=2), dict(timeout=float('nan'))):
            with self.assertRaises(ValueError):t.DeviceWindow(UUID, **options)
        for identifier in ('0', 'GPU-not-a-uuid', 'MIG-'+UUID[4:]):
            with self.assertRaises(ValueError):t.DeviceWindow(identifier)

    def test_startup_failure_can_be_cleaned_up_without_masking_original(self):
        window = t.DeviceWindow(UUID)
        with mock.patch.object(t.threading.Thread, 'start', side_effect=RuntimeError('start failed')):
            with self.assertRaisesRegex(RuntimeError, 'start failed'):window.start()
        result = window.finish(t.time.perf_counter())
        self.assertTrue(result['worker_stopped'])
        self.assertEqual(result['status'], 'UNAVAILABLE')
