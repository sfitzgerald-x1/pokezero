"""Bounded device-wide samples, not exact peaks or process attribution.

CUDA ordinals can be remapped; bind the sole JAX GPU through the CUDA driver
UUID before querying nvidia-smi. No CUDA/NVML dependency is loaded on import.
"""
import csv
import ctypes
import math
import statistics
import subprocess
import threading
import time
import uuid


def cuda_device_uuid(device):
    """Require one driver-visible device as well as the caller's sole JAX GPU."""
    if device.platform != 'gpu' or device.local_hardware_id != 0:
        raise RuntimeError('telemetry requires sole CUDA ordinal zero')
    cuda = ctypes.CDLL('libcuda.so.1')
    signatures = (
        ('cuInit', [ctypes.c_uint]),
        ('cuDeviceGetCount', [ctypes.POINTER(ctypes.c_int)]),
        ('cuDeviceGet', [ctypes.POINTER(ctypes.c_int), ctypes.c_int]),
        ('cuDeviceGetUuid_v2', [ctypes.c_void_p, ctypes.c_int]),
    )
    for name, args in signatures:
        fn = getattr(cuda, name)
        fn.argtypes, fn.restype = args, ctypes.c_int
    def call(name, *args):
        if getattr(cuda, name)(*args) != 0:
            raise RuntimeError('CUDA telemetry identity failed: ' + name)
    call('cuInit', 0)
    count, handle = ctypes.c_int(), ctypes.c_int()
    call('cuDeviceGetCount', ctypes.byref(count))
    if count.value != 1:
        raise RuntimeError('telemetry requires exactly one CUDA-visible GPU')
    call('cuDeviceGet', ctypes.byref(handle), 0)
    identifier = (ctypes.c_ubyte * 16)()
    call('cuDeviceGetUuid_v2', identifier, handle.value)
    return 'GPU-' + str(uuid.UUID(bytes=bytes(identifier)))


def parse_counters(output, identifier):
    rows = list(csv.reader(output.strip().splitlines()))
    if len(rows) != 1 or len(rows[0]) != 4 or rows[0][0].strip() != identifier:
        raise ValueError('device identity or counter shape mismatch')
    utilization, used, total = (float(x.strip()) for x in rows[0][1:])
    if (not all(math.isfinite(x) for x in (utilization, used, total))
            or not 0 <= utilization <= 100 or not 0 <= used <= total or total <= 0):
        raise ValueError('invalid device counters')
    return dict(utilization_percent=utilization, memory_used_mib=used, memory_total_mib=total)


def summarize(samples, start, end, *, cadence, timeout, worker_stopped):
    """Exclude calls crossing either boundary, retaining their raw records."""
    valid = [s for s in samples if s['status'] == 'OK'
             and start <= s['start_monotonic'] <= s['end_monotonic'] <= end]
    points = [start] + [s['end_monotonic'] for s in valid] + [end]
    failures = sum(s['status'] != 'OK' for s in samples)
    gaps = [b-a for a, b in zip(points, points[1:])]
    # This is sampled coverage, not the device's unknown internal averaging window.
    partial = failures or not worker_stopped or max(gaps) > 2 * cadence + timeout
    return dict(status=('UNAVAILABLE' if not valid else 'PARTIAL' if partial else 'SAMPLED'),
        window_start_monotonic=start, window_end_monotonic=end,
        cadence_seconds=cadence, query_timeout_seconds=timeout,
        query_count=len(samples), valid_in_window_samples=len(valid), failed_queries=failures,
        boundary_excluded_samples=sum(s['status'] == 'OK' and s not in valid for s in samples),
        maximum_sample_gap_seconds=max(gaps), worker_stopped=worker_stopped,
        mean_reported_utilization_percent=statistics.fmean(s['utilization_percent'] for s in valid) if valid else None,
        maximum_sampled_device_used_memory_mib=max(s['memory_used_mib'] for s in valid) if valid else None,
        attribution='Device-wide; exclusivity requires separate allocation evidence. No benchmark-only attribution.',
        interpretation='Mean of reported utilization samples and maximum sampled device-used memory, not continuous means, exact peaks or live-array memory.',
        samples=samples)


class DeviceWindow:
    """Poll off-thread; stop/join is bounded and outside the measured clock."""
    def __init__(self, identifier, *, cadence=1.0, timeout=0.5):
        if (not isinstance(identifier, str) or not identifier.startswith('GPU-')
                or str(uuid.UUID(identifier[4:])) != identifier[4:]):
            raise ValueError('full canonical GPU UUID required')
        if not all(math.isfinite(x) and x > 0 for x in (cadence, timeout)) or timeout > cadence:
            raise ValueError('finite positive timeout <= cadence required')
        self.identifier, self.cadence, self.timeout = identifier, cadence, timeout
        self.samples = []
        self._stop, self._lock = threading.Event(), threading.Lock()
        self._thread = None

    def start(self):
        if self._thread is not None:
            raise RuntimeError('telemetry window cannot restart')
        self.start_monotonic = time.perf_counter()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self.start_monotonic

    def _poll(self):
        while not self._stop.is_set():
            start = time.perf_counter()
            record = dict(start_monotonic=start, status='OK')
            try:
                output = subprocess.check_output([
                    'nvidia-smi', '--id=' + self.identifier,
                    '--query-gpu=uuid,utilization.gpu,memory.used,memory.total',
                    '--format=csv,noheader,nounits'], text=True, timeout=self.timeout,
                    stderr=subprocess.DEVNULL)
                record.update(parse_counters(output, self.identifier))
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                record['status'] = type(error).__name__
                if isinstance(error, subprocess.CalledProcessError):
                    record['exit_code'] = error.returncode
            record['end_monotonic'] = time.perf_counter()
            with self._lock:
                self.samples.append(record)
            if record['status'] == 'FileNotFoundError':
                break
            self._stop.wait(max(0, self.cadence - (time.perf_counter() - start)))

    def finish(self, end):
        if not hasattr(self, 'start_monotonic') or end < self.start_monotonic:
            raise RuntimeError('started positive telemetry window required')
        self._stop.set()
        # Thread creation/start itself can fail or be interrupted.
        if self._thread is not None and self._thread.ident is not None:
            self._thread.join(timeout=self.timeout + 0.25)
        with self._lock:
            samples = [dict(s) for s in self.samples]
        return dict(device_uuid=self.identifier, **summarize(samples, self.start_monotonic, end,
            cadence=self.cadence, timeout=self.timeout,
            worker_stopped=self._thread is None or not self._thread.is_alive()))
