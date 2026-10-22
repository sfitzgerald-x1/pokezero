"""Opt-in, non-score-bearing phase visibility for one controlled benchmark.

This bounded mutable file is not a game result or a watchdog heartbeat. It
contains no actions, observations, requests, simulator states or exceptions.
Use a new path per invocation; an existing artifact is never adopted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
import uuid


SCHEMA_VERSION = "pokezero.fullgame-phase-progress.v1"
_PHASES = frozenset({"run", "game", "bridge_wait", "decision", "candidate_policy",
                     "opponent_policy", "foulplay_choice", "submission"})


class FullgamePhaseProgress:
    """Single-writer atomic snapshot; I/O/ownership faults fail closed."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.run_id = uuid.uuid4().hex
        self.sequence = 0
        self.last_returned: dict[str, object] | None = None
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also rejects symlinks, malformed files and files
        # from earlier invocations. Never reinterpret a stopped run as resumed.
        initial = self._payload("run", "started", seed=None, seat=None, decision_round=None)
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(initial, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            # Publish complete bytes exclusively: even the first observer sees
            # either no file or valid JSON. link never replaces an existing path.
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _payload(self, phase: str, status: str, *, seed: int | None,
                 seat: str | None, decision_round: int | None) -> dict[str, object]:
        if phase not in _PHASES or status not in {"started", "returned"}:
            raise ValueError("unknown full-game progress phase/status")
        if seat not in {None, "p1", "p2"}:
            raise ValueError("invalid full-game progress seat")
        for value in (seed, decision_round):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("progress addresses require non-negative integers")
        current = {"phase": phase, "status": status, "seed": seed, "seat": seat,
                   "decision_round": decision_round, "monotonic_ns": time.monotonic_ns(),
                   "unix_ns": time.time_ns()}
        return {"schema_version": SCHEMA_VERSION, "run_id": self.run_id,
                "pid": os.getpid(), "sequence": self.sequence,
                "score_bearing": False, "current": current,
                "last_returned": current if status == "returned" else self.last_returned}

    def emit(self, phase: str, status: str, *, seed: int | None = None,
             seat: str | None = None, decision_round: int | None = None) -> None:
        payload = self._payload(phase, status, seed=seed, seat=seat, decision_round=decision_round)
        if self.path.is_symlink():
            raise ValueError("refusing a replaced full-game progress symlink")
        existing = json.loads(self.path.read_text(encoding="utf-8"))
        if (not isinstance(existing, dict) or existing.get("schema_version") != SCHEMA_VERSION
                or existing.get("run_id") != self.run_id
                or existing.get("sequence") != self.sequence):
            raise ValueError("full-game progress ownership changed")
        payload["sequence"] = self.sequence + 1
        fd, name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
        self.sequence += 1
        self.last_returned = payload["last_returned"]
