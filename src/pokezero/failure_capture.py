"""Shared, fail-closed quarantine capture; never a scientific replay driver.

Scientific owners supply an isolated quarantine directory and a build manifest.
The same decision scope is used by world construction, rendering and crate
boundaries. Capture is durable BEFORE a failure is raised or a world dropped.
No serializer substitutes repr(), guesses an instruction, or consults RNG.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from functools import wraps
from enum import Enum
from hashlib import sha256
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import struct
import time
import traceback
from uuid import uuid4


class CaptureFailed(BaseException):
    """Do not let an existing `except Exception: continue` hide capture loss."""


def _count_capture_loss(quarantine, error):
    if not getattr(error, "capture_failure_counted", False):
        quarantine.failures += 1
        error.capture_failure_counted = True
    return error


def runtime_build(native=None):
    """Identify actual loaded binaries and Python sources, not a planned image."""
    binaries = {}
    for name in ("pokezero_search", "poke_engine"):
        path = getattr(native, "__file__", None) if name == "pokezero_search" else None
        if path is None:
            try:
                spec = importlib.util.find_spec(name)
                path = spec.origin if spec else None
            except (ImportError, ValueError):
                path = None
        if path and Path(path).is_file():
            origin = Path(path)
            candidates = [origin]
            if origin.name == "__init__.py":
                candidates.extend(sorted(origin.parent.rglob("*.so")))
                candidates.extend(sorted(origin.parent.rglob("*.pyd")))
            binaries[name] = {p.name: digest_file(p) for p in candidates}
        else:
            binaries[name] = unavailable("not loaded/found")
    sources = {str(path.relative_to(Path(__file__).parent)): digest_file(path)
               for path in sorted(Path(__file__).parent.rglob("*.py"))}
    build = dict(native_binaries=binaries, python_sources_sha256=sha256(canonical(sources)).hexdigest(),
                 source_commit=os.environ.get("POKEZERO_SOURCE_COMMIT", unavailable("not supplied")))
    build["fingerprint"] = sha256(canonical(build)).hexdigest()
    return build


def unavailable(reason):
    return {"availability": "unavailable", "reason": reason}


def snapshot(value):
    """Detached snapshot; repeated objects are copied once within this call.

    The memo never crosses calls, so subsequent input mutation is observed.
    It retains originals as well as copies to prevent temporary id reuse.
    """
    return _snapshot(value, {}, set())


def _snapshot(value, memo, visiting):
    if value is None or type(value) in (str, int, float, bool):
        if type(value) is float and not math.isfinite(value):
            return dict(encoding="ieee754-binary64-big-endian-hex",
                        data=struct.pack(">d", value).hex())
        return value
    identity = id(value)
    if identity in visiting:
        raise TypeError("capture cannot serialize a cyclic input")
    if identity in memo:
        original, copied = memo[identity]
        assert original is value
        return copied
    visiting.add(identity)
    try:
        copied = _snapshot_container(value, memo, visiting)
        memo[identity] = (value, copied)
        return copied
    finally:
        visiting.remove(identity)


def _snapshot_container(value, memo, visiting):
    if isinstance(value, Enum):
        return _snapshot(value.value, memo, visiting)
    if is_dataclass(value) and not isinstance(value, type):
        return _snapshot(asdict(value), memo, visiting)
    if isinstance(value, Mapping):
        return {str(k): _snapshot(v, memo, visiting) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_snapshot(v, memo, visiting) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_snapshot(v, memo, visiting) for v in value),
                      key=lambda v: json.dumps(v, sort_keys=True))
    if type(value).__module__.startswith("numpy") and hasattr(value, "tobytes"):
        return dict(encoding="numpy-c-order-hex", dtype=value.dtype.str,
                    shape=list(value.shape), data=value.tobytes(order="C").hex())
    raise TypeError("capture cannot serialize " + type(value).__name__)


def canonical(value):
    return json.dumps(snapshot(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def digest_file(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Quarantine:
    def __init__(self, directory, *, build):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.build = snapshot(build)
        self.reason_counts = Counter()
        self.events = self.successes = self.failures = 0

    def write(self, kind, payload):
        """Exclusive immutable files, file AND directory fsync, private mode."""
        try:
            return self._write(kind, payload)
        except BaseException as error:
            raise _count_capture_loss(self, CaptureFailed(
                f"capture.failed writing {kind}: {error}")) from error

    def _write(self, kind, payload):
        name = f"{kind}-{uuid4().hex}.json"
        path = self.directory / name
        data = canonical(payload) + b"\n"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        fd = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return {"file": name, "sha256": sha256(data).hexdigest()}

    def telemetry(self):
        return dict(events=self.events, capture_successes=self.successes,
                    capture_failures=self.failures, reason_counts=dict(self.reason_counts))


_active = ContextVar("pokezero_failure_capture", default=None)


def current():
    return _active.get()


class DecisionCapture:
    def __init__(self, quarantine, context, rng):
        self.quarantine = quarantine
        public = getattr(context, "public_materialization_state", None)
        replay = getattr(public, "replay", None)
        self.fields = dict(
            root={k: getattr(context, k, None) for k in
                  ("battle_id", "seed", "player_id", "decision_round_index")},
            request=snapshot(getattr(public, "self_request", unavailable("not supplied"))),
            initial_request=snapshot(getattr(public, "self_initial_request", unavailable("not supplied"))),
            self_move_states=snapshot(getattr(public, "self_move_states", unavailable("not supplied"))),
            observation={k:snapshot(getattr(getattr(context, "observation", None), k,
                                           unavailable("not supplied"))) for k in
                         ("categorical_ids", "numeric_features", "token_type_ids", "attention_mask",
                          "legal_action_mask", "perspective", "metadata", "schema_version")},
            public_log_prefix=snapshot(getattr(replay, "public_lines", unavailable("not supplied"))),
            rng_state=rng.getstate(),
            world_spec=unavailable("world not yet constructed"),
            engine_state=unavailable("engine state not yet constructed"),
            instruction_list=unavailable("transition not yet generated"),
            offending_instruction=unavailable("no instruction identified"),
        )
        self._errors = []  # Hold objects: id reuse must not suppress another failure.
        self.receipts = []
        self._lazy = {}

    def defer(self, name, provider):
        """Retain a failure-only evidence provider; never evaluate on success.

        The owner freezes roots once and keeps native sessions alive. Providers
        must reconstruct the precise active branch/batch without RNG or writes
        to scientific rosters. Failure to produce evidence remains fail-closed.
        """
        if name in self._lazy or not isinstance(name, str) or not callable(provider):
            raise CaptureFailed('capture.failed: duplicate/invalid lazy evidence provider')
        self._lazy[name] = provider

    def update(self, **fields):
        try:
            self.fields.update(snapshot(fields))
        except BaseException as error:
            if isinstance(error, CaptureFailed):
                raise _count_capture_loss(self.quarantine, error)
            raise _count_capture_loss(self.quarantine, CaptureFailed(
                f"capture.failed snapshotting decision fields: {error}")) from error

    def capture(self, stage, error=None, *, codes=None, witness=None):
        if isinstance(error, CaptureFailed):
            raise _count_capture_loss(self.quarantine, error)
        if error is not None:
            for observed, link in self._errors:
                if observed is error:
                    return link
        q = self.quarantine
        if getattr(error, "pokezero_capture_failed", False) is True:
            _count_capture_loss(q, error)
            codes = ["capture.failed"]
        codes = list(getattr(error, "failure_codes", None) or codes or [stage + ".contract_failure"])
        native = getattr(error, "pokezero_failure_witness", None)
        if getattr(error, "reason", None):
            codes = ["world." + str(error.reason)]
        codes = sorted(set(codes))
        q.events += 1
        try:
            prior_events = getattr(error, "pokezero_refusal_events", None)
            if prior_events is not None:
                prior_events = json.loads(prior_events) if isinstance(prior_events, str) else prior_events
                _capture_native_events(self, prior_events)
            if native is not None:
                native = json.loads(native) if isinstance(native, str) else snapshot(native)
                witness = native
                codes = sorted(set(native.get("reason_codes") or codes))
            if getattr(error, "pokezero_capture_failed", False) is True:
                codes = sorted(set(codes) | {"capture.failed"})
            q.reason_counts.update(codes)
            payload = dict(schema="pokezero.quarantined-failure.v1", stage=stage,
                reason_codes=codes, build=q.build, captured_unix=time.time(),
                **snapshot(self.fields))
            if self._lazy:
                payload['lazy_evidence'] = {name:snapshot(provider())
                    for name,provider in self._lazy.items()}
            if witness is not None:
                payload["origin_witness"] = snapshot(witness)
                for key in ("engine_state", "instruction_list", "offending_instruction"):
                    if key in witness:
                        payload[key] = snapshot(witness[key])
            if error is not None:
                payload["error"] = dict(type=type(error).__name__, message=str(error),
                    traceback="".join(traceback.format_exception(error)))
            link = q.write("failure", payload)
            link["reason_codes"] = codes
        except BaseException as capture_error:
            if isinstance(capture_error, CaptureFailed):
                raise _count_capture_loss(q, capture_error)
            raise _count_capture_loss(q, CaptureFailed(
                f"capture.failed at {stage}; original={type(error).__name__}: "
                f"{error}; capture={capture_error}")) from capture_error
        q.successes += 1
        self.receipts.append(link)
        if error is not None:
            self._errors.append((error, link))
        return link


@contextmanager
def decision_scope(quarantine, context, rng):
    if current() is not None or quarantine is None:
        yield current()
        return
    try:
        capture = DecisionCapture(quarantine, context, rng)
    except BaseException as error:
        raise _count_capture_loss(quarantine, CaptureFailed(
            f"capture.failed serializing decision inputs: {error}")) from error
    token = _active.set(capture)
    try:
        # Abrupt death leaves this unmatched input receipt, not a clean exit.
        entry = quarantine.write("decision-input", dict(schema="pokezero.capture-input.v1",
            build=quarantine.build, **snapshot(capture.fields)))
        capture.update(input_receipt=entry)
        yield capture
    except BaseException as error:
        capture.capture("decision", error)
        quarantine.write("decision-terminal", dict(schema="pokezero.capture-terminal.v1",
            input=entry, status="REFUSED", failures=capture.receipts,
            telemetry=quarantine.telemetry()))
        raise
    finally:
        _active.reset(token)
    quarantine.write("decision-terminal", dict(schema="pokezero.capture-terminal.v1",
        input=entry, status="RETURNED", failures=capture.receipts,
        telemetry=quarantine.telemetry()))


_configured = {}


def configured_quarantine(*, required=False):
    directory = os.environ.get("POKEZERO_QUARANTINE_DIR")
    if not directory:
        if required:
            raise ValueError("scientific engine policy requires POKEZERO_QUARANTINE_DIR")
        return None
    key = str(Path(directory).resolve())
    if key not in _configured:
        _configured[key] = Quarantine(key, build=runtime_build())
    return _configured[key]


def captured_decision(function):
    @wraps(function)
    def wrapped(self, context, *, rng):
        quarantine = getattr(self, "_quarantine", None) or configured_quarantine()
        try:
            with decision_scope(quarantine, context, rng):
                return function(self, context, rng=rng)
        finally:
            stats = getattr(self, "stats", None)
            if quarantine is not None and stats is not None:
                stats.capture_telemetry = quarantine.telemetry()
    return wrapped


def record_refusal(stage, error=None, *, codes=None, witness=None):
    capture = current()
    return capture.capture(stage, error, codes=codes, witness=witness) if capture else None


def _capture_native_events(capture, events):
    if not isinstance(events, list):
        raise CaptureFailed("capture.failed: native refusal events are not an array")
    for event in events:
        if (not isinstance(event, dict) or not event.get("reason_codes")
                or not isinstance(event["reason_codes"], list)
                or any(not isinstance(c, str) or not c for c in event["reason_codes"])):
            raise CaptureFailed("capture.failed: malformed native refusal event")
        capture.capture("crate.prior", codes=event["reason_codes"], witness=event)


def record_native_refusals(report):
    capture = current()
    if capture is None:
        return
    events = report.get("capture_refusals", [])
    if (not isinstance(events, list) or type(report.get("prior_fallbacks", 0)) is not int
            or len(events) != report.get("prior_fallbacks", 0)):
        raise CaptureFailed("capture.failed: native prior refusal counts lack complete origin evidence")
    _capture_native_events(capture, events)


def boundary(stage, *, arguments=(), result_field=None):
    """Capture at the producing boundary, before outer fallback handlers run."""
    def decorate(function):
        signature = inspect.signature(function)
        @wraps(function)
        def wrapped(*args, **kwargs):
            capture = current()
            if capture is None:
                return function(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            fields = {k: bound.arguments[k] for k in arguments if k in bound.arguments}
            if "rng" in bound.arguments and bound.arguments["rng"] is not None:
                fields["rng_state_before_boundary"] = bound.arguments["rng"].getstate()
            capture.update(**{stage: fields})
            if stage == "world.construct":
                capture.update(world_spec=unavailable("this world has not been constructed"),
                    engine_state=unavailable("this world's native state has not been constructed"),
                    instruction_list=unavailable("this world's transition has not been generated"))
            try:
                result = function(*args, **kwargs)
                if result_field == "engine_state":
                    capture.update(engine_state=result.to_string())
                elif result_field == "world_spec":
                    capture.update(world_spec=result.spec)
                return result
            except BaseException as error:
                capture.capture(stage, error)
                raise
        return wrapped
    return decorate


def _read_evidence(link, root):
    if not isinstance(link, dict) or not isinstance(link.get("file"), str):
        raise ValueError("missing file-backed evidence link")
    path = (Path(root) / link["file"]).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError("evidence link escapes admission directory or is missing")
    if digest_file(path) != link.get("sha256"):
        raise ValueError("evidence digest mismatch")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("evidence must be a JSON object")
    return value


def require_fixture_coverage(observed_codes, manifest, *, build_fingerprint, evidence_root):
    """Admission, not inference: every observed code needs a verified fixture."""
    entries = manifest.get("fixtures", [])
    for code in sorted(set(observed_codes)):
        matches = [row for row in entries if code in row.get("reason_codes", [])]
        qualified = False
        for row in matches:
            try:
                artifacts = {}
                for kind in ("bundle", "diagnostic_receipt", "fixture", "verification"):
                    artifacts[kind] = _read_evidence(row[kind], evidence_root)
                fixture, verification = artifacts["fixture"], artifacts["verification"]
                diagnostic = artifacts["diagnostic_receipt"]
                bundle = artifacts["bundle"]
                source_root = bundle.get("root", bundle.get("origin", {}).get("root"))
                original_codes = set(bundle.get("reason_codes", []) + bundle.get("refusal_codes", [])
                    + diagnostic.get("refusal_codes", []) + diagnostic.get("subcases", []))
                original_codes |= {"renderer." + c for c in list(original_codes)}
                qualified |= (verification.get("build_fingerprint") == build_fingerprint
                    and type(verification.get("exit_code")) is int and verification["exit_code"] == 0
                    and code in verification.get("reason_codes", []) and bool(verification.get("test"))
                    and verification.get("fixture_sha256") == row["fixture"]["sha256"]
                    and fixture.get("bundle_sha256") == row["bundle"]["sha256"]
                    and fixture.get("diagnostic_receipt_sha256") == row["diagnostic_receipt"]["sha256"]
                    and code in original_codes and source_root is not None
                    and source_root == diagnostic.get("root") == fixture.get("root")
                    and diagnostic.get("status") == "REFUSAL_RECOVERED"
                    and diagnostic.get("scientific_writes") == 0
                    and diagnostic.get("actions_selected") == 0
                    and diagnostic.get("games_scored") == 0)
            except (OSError, ValueError, KeyError, TypeError):
                continue
        if not qualified:
            raise ValueError("fixture gate: missing bundle-derived verified class " + code)


def _first_engine_capture_inventory(manifest, inventory, *, evidence_root, capture_directory):
    """Audit first use without inventing a historical captured failure.

    The prospective roster's own directory is excluded: new failures need
    fixtures before the *next roster*, not between games of this roster.
    Previous runs and qualification failures remain in the persistent scope.
    """
    audit = _read_evidence(inventory.get("capture_history_audit"), evidence_root)
    if (not isinstance(manifest.get("fixtures"), list)
            or audit.get("schema") != "pokezero.engine-mcts-capture-history.v1"
            or audit.get("path") != "engine-mcts"
            or not isinstance(audit.get("adoption_boundary"), str)
            or not audit["adoption_boundary"].strip()
            or not isinstance(audit.get("audited_locations"), list)
            or not audit["audited_locations"]
            or any(not isinstance(p, str) or not p.strip() for p in audit["audited_locations"])):
        raise ValueError("first engine capture admission requires an explicit history audit")
    root = Path(evidence_root).resolve()
    history_name = audit.get("capture_history_root")
    prospective_name = audit.get("prospective_capture_directory")
    if (not isinstance(history_name, str) or not history_name
            or not isinstance(prospective_name, str) or not prospective_name
            or capture_directory is None):
        raise ValueError("first engine capture admission must bind its history and prospective directory")
    history = (root / history_name).resolve()
    prospective = (root / prospective_name).resolve()
    if (not history.is_relative_to(root) or not history.is_dir()
            or prospective == history or not prospective.is_relative_to(history)
            or Path(capture_directory).resolve() != prospective):
        raise ValueError("first engine capture history/prospective directory is invalid")
    legacy = _read_evidence(audit.get("legacy_history"), root)
    if (legacy.get("schema") != "pokezero.legacy-refusal-history.v1"
            or legacy.get("path") != "engine-mcts"
            or legacy.get("capture_completeness") not in ("incomplete", "unavailable")
            or not isinstance(legacy.get("sources"), list) or not legacy["sources"]):
        raise ValueError("legacy fallback history must be disclosed separately from captured history")
    for source in legacy["sources"]:
        _read_evidence(source, root)
    observed = set(manifest["observed_reason_codes"])
    for path in history.rglob("failure-*.json"):
        actual = path.resolve()
        if not actual.is_relative_to(history):
            raise ValueError("captured history link escapes audited history")
        if actual.is_relative_to(prospective):
            continue
        value = json.loads(actual.read_text())
        codes = value.get("reason_codes") if isinstance(value, dict) else None
        if (not isinstance(codes, list) or not codes
                or any(not isinstance(c, str) or not c for c in codes)
                or not set(codes) <= observed):
            raise ValueError("capture history contains an unrepresented refusal class")


def require_capture_admission(path, *, build, scientific_path="p2-informed-one-ply",
                              capture_directory=None):
    if not path:
        raise ValueError("scientific policy requires a reviewed capture/fixture admission manifest before a roster")
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "pokezero.capture-admission.v1":
        raise ValueError("unsupported capture admission schema")
    observed = manifest.get("observed_reason_codes")
    if not isinstance(observed, list) or any(not isinstance(c, str) or not c for c in observed):
        raise ValueError("capture admission must name the observed refusal classes")
    inventory = _read_evidence(manifest.get("observed_inventory"), path.parent)
    first_engine = manifest.get("admission_mode") == "first-capture-enabled-engine-mcts"
    if first_engine:
        if scientific_path != "engine-mcts":
            raise ValueError("first capture admission is restricted to engine-mcts")
        _first_engine_capture_inventory(manifest, inventory, evidence_root=path.parent,
                                        capture_directory=capture_directory)
    sources = inventory.get("source_runs")
    if (inventory.get("schema") != "pokezero.observed-refusal-inventory.v1"
            or inventory.get("path") != scientific_path
            or not isinstance(sources, list)
            or not isinstance(inventory.get("reason_codes"), list)
            or set(observed) != set(inventory["reason_codes"])
            or bool(sources) != bool(observed)
            or (not first_engine and not observed)):
        raise ValueError("capture admission inventory must cover its recorded refusal history")
    recovered_codes = set()
    run_ids = set()
    for source in inventory["source_runs"]:
        terminal = _read_evidence(source["terminal"], path.parent)
        diagnostic = _read_evidence(source["diagnostic_receipt"], path.parent)
        bundle = _read_evidence(source["bundle"], path.parent)
        run_id = source.get("run_id")
        codes = source.get("reason_codes")
        root = bundle.get("root", bundle.get("origin", {}).get("root"))
        if (not isinstance(run_id, str) or not run_id or run_id in run_ids
                or terminal.get("run_id") != run_id
                or type(terminal.get("exit_code")) is not int
                or terminal.get("status") != "CLOSED"
                or diagnostic.get("source_terminal_sha256") != source["terminal"]["sha256"]
                or diagnostic.get("source_bundle_sha256") != source["bundle"]["sha256"]
                or root is None or root != diagnostic.get("root")
                or diagnostic.get("status") != "REFUSAL_RECOVERED"
                or not diagnostic.get("build_fingerprint")
                or any(type(diagnostic.get(k)) is not int or diagnostic[k] != 0 for k in
                       ("scientific_writes", "actions_selected", "games_scored"))
                or not isinstance(codes, list) or not codes
                or set(codes) != set(diagnostic.get("reason_codes", []))):
            raise ValueError("observed inventory does not bind closed-run diagnostic evidence")
        run_ids.add(run_id)
        recovered_codes.update(codes)
    if recovered_codes != set(observed):
        raise ValueError("observed inventory omits or invents a recovered refusal class")
    if manifest.get("build_fingerprint") != build["fingerprint"]:
        raise ValueError("capture admission does not bind the actual runtime build")
    require_fixture_coverage(observed, manifest, build_fingerprint=build["fingerprint"],
                             evidence_root=path.parent)
    return dict(file=path.name, sha256=digest_file(path), reason_codes=observed)
