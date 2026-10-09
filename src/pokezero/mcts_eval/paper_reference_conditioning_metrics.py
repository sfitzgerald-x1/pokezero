"""Opt-in wall-clock telemetry; never supplies a seed, world or search value.

Phase times are inclusive and may nest. In particular, anchor sampling is part
of history conditioning, not extra elapsed time to add to the factory total.
"""
from contextlib import contextmanager
import time


@contextmanager
def phase(evidence, name):
    timing = evidence.get('phase_timing')
    if timing is None:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        row = timing.setdefault(name, {'calls': 0, 'seconds': 0.0})
        row['calls'] += 1
        row['seconds'] += max(0.0, time.perf_counter() - started)
