"""Opt-in aggregate callback timings, never policy inputs or qualification.

The callback total contains the component spans. Model evaluation includes
lock wait and canonical setup/encoding/forward/output extraction, not pure
network latency. Native request creation and the Rust bridge are outside these
Python spans. Instrumentation can change deadline/cancellation behavior.
"""
from __future__ import annotations

from threading import Lock
from time import perf_counter_ns


PHASES = (
    "callback_total", "payload_binding", "view_reconstruction",
    "distribution_binding", "observation_and_surface", "model_evaluation",
    "output_certification",
)


class PolicyOpponentDiagnostics:
    """Bounded thread-safe numerical sink; no per-call samples or private data.

Clock failures invalidate the diagnostic instead of replacing a policy result
or masking its exception. There is no user-supplied clock, hook or output sink.
"""

    def __init__(self) -> None:
        self._lock = Lock()
        # calls, failed calls, successfully timed calls, elapsed nanoseconds.
        self._rows = {phase: [0, 0, 0, 0] for phase in PHASES}
        self._faults = 0

    def phase(self, phase: str) -> _Phase:
        if phase not in self._rows:
            raise ValueError("unknown policy-opponent diagnostic phase")
        return _Phase(self, phase)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "schema": "pokezero.policy-opponent.callback-diagnostics.v1",
                "valid": self._faults == 0,
                "timing_faults": self._faults,
                "phases": {phase: dict(calls=row[0], failed_calls=row[1],
                    timed_calls=row[2], elapsed_seconds=row[3] / 1_000_000_000)
                    for phase, row in self._rows.items()},
                "callback_total_contains_components_do_not_sum": True,
                "model_evaluation_includes_lock_wait_and_canonical_setup": True,
                "native_request_and_bridge_outside_python_spans": True,
                "instrumentation_can_change_deadlines": True,
                "qualifies_uninstrumented_runtime": False,
            }


class _Phase:
    __slots__ = ("_sink", "_phase", "_start")

    def __init__(self, sink: PolicyOpponentDiagnostics, phase: str) -> None:
        self._sink, self._phase, self._start = sink, phase, None

    def __enter__(self) -> None:
        try:
            stamp = perf_counter_ns()
            if type(stamp) is not int or stamp < 0:
                raise ValueError("invalid diagnostic clock")
            self._start = stamp
        except Exception:
            # Do not retain the exception or any invocation-owned data.
            with self._sink._lock:
                self._sink._faults += 1

    def __exit__(self, exc_type, exc, traceback) -> bool:
        elapsed = None
        if self._start is not None:
            try:
                end = perf_counter_ns()
                if type(end) is not int or end < self._start:
                    raise ValueError("invalid diagnostic clock")
                elapsed = end - self._start
            except Exception:
                with self._sink._lock:
                    self._sink._faults += 1
        with self._sink._lock:
            row = self._sink._rows[self._phase]
            row[0] += 1
            row[1] += int(exc_type is not None)
            if elapsed is not None:
                row[2] += 1
                row[3] += elapsed
        return False
