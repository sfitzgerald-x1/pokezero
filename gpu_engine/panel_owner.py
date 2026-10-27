"""One owned, calendar-bounded exact panel attempt; never retry or infer exit 0."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import secrets
import subprocess
import sys
import time
import zlib

from .collect_fidelity_panel import atomic_json
from . import future_state as mapping
from .finite_law import compare_laws


def validate_slot(record, entry, input_sha, slot, *, root_record=None, constructor_context=None):
    if record["slot"] != slot or record["input_jsonl_sha256"] != input_sha:
        raise ValueError("slot/input identity mismatch")
    status = record["status"]
    cfg=None;admitted=None
    if 'ordinary_carry' in entry:
        from .ordinary_panel import contract,audit_dispatch
        cfg=contract(entry)
        if root_record is None or constructor_context is None:raise ValueError('FROZEN_ROOT_AUDIT_REQUIRED')
        expected,admitted=audit_dispatch(root_record,constructor_context,entry)
        if record.get('root_dispatch')!=expected:raise ValueError('ROOT_DISPATCH_CHANGED')
    certificate=record.get('consumer_certificate')
    if certificate is not None and type(certificate) is not dict:
        raise ValueError('INVALID_CONSUMER_CERTIFICATE_TYPE')
    if cfg is not None and record['root_dispatch']['route']=='ordinary_carry':
        from .ordinary_law import OrdinaryPlan
        from .ordinary_panel import consumer_certificate
        from .collect_fidelity_panel import encoded
        if isinstance(admitted,OrdinaryPlan):
            if certificate is None or encoded(certificate)!=encoded(consumer_certificate(admitted)):
                raise ValueError('COMPLETE_BOUND_ORDINARY_CERTIFICATE_REQUIRED')
        elif certificate is not None or status!='UNKNOWN':
            raise ValueError('REFUSED_ORDINARY_PLAN_CANNOT_PROVIDE_LAW')
    if cfg is None and (any(name in record for name in ('root_dispatch','comparison_schema','ordinary_output_laws'))
            or certificate is not None and certificate.get('kind')=='ordinary_carry'):
        raise ValueError('UNREGISTERED_ORDINARY_OUTPUT_METHOD')
    if cfg is not None and certificate is not None:
        kind=('ordinary_carry' if certificate.get('kind')=='ordinary_carry' else
            'conditional' if 'native_cases' in certificate else 'legacy')
        if kind!=record['root_dispatch']['route']:raise ValueError('CONSUMER_CERTIFICATE_ROUTE_CHANGED')
    if status == "UNKNOWN":
        if cfg is not None and 'ordinary_output_laws' in record:raise ValueError('UNKNOWN_CANNOT_ACCEPT_PARTIAL_LAW')
        if (record["identified_mass"], record["unknown_mass"]) != ("0", "1") or not record["category"]:
            raise ValueError("invalid whole-slot unknown")
        return status
    if status not in {"AGREE", "DIVERGE"} or (record["identified_mass"], record["unknown_mass"]) != ("1", "0"):
        raise ValueError("invalid complete law status")
    if cfg is not None and record['root_dispatch']['route']=='ordinary_carry':
        from dataclasses import asdict
        from .ordinary_law import OrdinaryPlan
        from .ordinary_panel import audit_evidence,EVIDENCE
        from .collect_fidelity_panel import encoded
        if (not isinstance(admitted,OrdinaryPlan) or record.get('comparison_schema')!=EVIDENCE
                or 'outcomes' in record):raise ValueError('REGISTERED_COMPLETE_ORDINARY_METHOD_REQUIRED')
        compared=audit_evidence(record['ordinary_output_laws'],admitted,entry)
        if (record.get('comparison')!=json.loads(encoded(asdict(compared)))
                or record.get('exact_cases')!=admitted.law.cardinality
                or record.get('execution_cells')!=admitted.law.cardinality
                or (compared.half_l1==0)!=(status=='AGREE')):
            raise ValueError('STORED_ORDINARY_CONCLUSION_DISAGREES_WITH_FULL_LAWS')
        return status
    if 'comparison_schema' in record or 'ordinary_output_laws' in record:
        raise ValueError('OUTPUT_METHOD_DOES_NOT_MATCH_ROOT_ROUTE')
    exact, native, seen = {}, {}, set()
    for outcome in record["outcomes"]:
        key = mapping._freeze(outcome["key"])
        if key in seen:
            raise ValueError("duplicate stored outcome key")
        seen.add(key)
        rational = Fraction(outcome["jax_probability"])
        probability = outcome["native_probability"]
        if rational < 0 or not math.isfinite(probability) or probability < 0:
            raise ValueError("invalid stored probability")
        if rational:
            exact[key] = rational
        if probability:
            native[key] = probability
    compared = compare_laws(exact, native, mass_tolerance=entry["mass_tolerance"])
    agrees = (not compared["exact_only"] and not compared["native_only"]
              and compared["max_probability_difference"] <= entry["probability_tolerance"])
    if agrees != (status == "AGREE"):
        raise ValueError("stored conclusion disagrees with full law")
    return status


def reconstruct(panel, entry):
    """Inspect valid CLOSED files once at terminal; progress is not authority."""
    fixed = entry["slots"]
    counts, issues = Counter(), {}
    frozen_records=[] if 'ordinary_carry' in entry else None
    try:
        if frozen_records is not None:
            from .ordinary_panel import contract
            from .collect_fidelity_panel import runtime_admission
            contract(entry);runtime_admission(entry)
        frozen = json.loads((panel / "inputs-frozen.json").read_bytes())
        prospective=entry.get("constructor_provenance") is not None
        if prospective and "constructor_context_file_sha256" not in frozen:
            raise ValueError("opt-in source freeze requires constructor context evidence")
        if "constructor_context_file_sha256" in frozen:
            context_payload=(panel/"constructor-context.json").read_bytes()
            if hashlib.sha256(context_payload).hexdigest()!=frozen["constructor_context_file_sha256"]:
                raise ValueError("prospective constructor context changed after source freeze")
            context=json.loads(context_payload)
        digest, number = hashlib.sha256(), 0
        with gzip.open(panel / "inputs.jsonl.gz", "rb") as stream:
            for line in stream:
                item = json.loads(line)
                if item["slot"] != number:
                    raise ValueError("input roster order/identity mismatch")
                if prospective and (not {"constructor_provenance","post_entry_state","constructor_context_sha256"}<=item.keys()
                        or not isinstance(item["constructor_provenance"],dict)
                        or set(item["post_entry_state"])!=set(mapping.FIELD_POLICY)
                        or item["constructor_context_sha256"]!=context["context"]["sha256"]):
                    raise ValueError("opt-in source record has missing constructor evidence")
                if frozen_records is not None:frozen_records.append(item)
                digest.update(line)
                number += 1
        if (number != fixed or frozen["slots"] != fixed
                or digest.hexdigest() != frozen["input_jsonl_sha256"]
                or hashlib.sha256((panel/"inputs.jsonl.gz").read_bytes()).hexdigest() != frozen["compressed_sha256"]):
            raise ValueError("input freeze did not complete or hash mismatch")
    except (OSError, ValueError, KeyError, TypeError, EOFError, zlib.error) as error:
        return dict(fixed_slots=fixed, valid_completed_slots=0, attempted_slots=0,
                    unattempted_slots=fixed, counts={"UNKNOWN":fixed},
                    input_freeze_valid=False, stop_reason=f"INPUT_FREEZE_INVALID: {type(error).__name__}")
    current = None
    try:
        receipt = json.loads((panel / "current-slot.json").read_bytes())
        if (receipt["status"] == "STARTED" and receipt["inputs"] == frozen
                and type(receipt["slot"]) is int and 0 <= receipt["slot"] < fixed):
            current = receipt["slot"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    valid = attempted = 0
    for slot in range(fixed):
        path = panel / f"slot-{slot:04d}.json.gz"
        if not path.exists():
            counts["UNKNOWN"] += 1
            if slot == current:
                attempted += 1
                issues[str(slot)] = "INTERRUPTED_WITHOUT_COMPLETE_LAW"
            else:
                issues[str(slot)] = "UNATTEMPTED"
            continue
        attempted += 1
        try:
            with gzip.open(path, "rb") as stream:
                record = json.load(stream)
            counts[validate_slot(record, entry, frozen["input_jsonl_sha256"], slot,
                root_record=None if frozen_records is None else frozen_records[slot],
                constructor_context=None if frozen_records is None else context)] += 1
            valid += 1
        except (OSError, ValueError, KeyError, TypeError, EOFError, ZeroDivisionError, zlib.error) as error:
            counts["UNKNOWN"] += 1
            issues[str(slot)] = f"INCOMPLETE_OR_INVALID_SLOT: {type(error).__name__}"
    return dict(fixed_slots=fixed, valid_completed_slots=valid, attempted_slots=attempted,
        unattempted_slots=fixed-attempted, counts=dict(counts), incomplete_slots=issues,
        input_freeze_valid=True, input_jsonl_sha256=frozen["input_jsonl_sha256"])


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def drain_owned(worker, *, grace_seconds=10, kill_seconds=5):
    """Always reap the actual child; require process-group absence for success."""
    errors = []
    def send(sig):
        try:
            os.killpg(worker.pid, sig)
        except ProcessLookupError:
            pass
        except OSError as error:
            errors.append(type(error).__name__)
    actual, absent = None, False
    try:
        actual = worker.poll()
        if actual is None:
            send(signal.SIGTERM)
            actual = worker.wait(timeout=grace_seconds)
    except BaseException as error:
        errors.append(f"INITIAL_REAP: {type(error).__name__}")
    finally:
        # Even an unexpected wait/poll failure must reach the final kill/reap.
        try:
            alive = group_alive(worker.pid)
        except OSError as error:
            alive = True
            errors.append(f"GROUP_STATE_UNKNOWN: {type(error).__name__}")
        if alive:
            send(signal.SIGKILL)
        if actual is None:
            try:
                actual = worker.wait(timeout=kill_seconds)
            except BaseException as error:
                errors.append(f"FINAL_REAP_UNKNOWN: {type(error).__name__}")
        try:
            deadline = time.monotonic() + kill_seconds
            while group_alive(worker.pid) and time.monotonic() < deadline:
                time.sleep(.05)
            absent = not group_alive(worker.pid)
        except OSError as error:
            errors.append(f"FINAL_GROUP_UNKNOWN: {type(error).__name__}")
    return dict(actual_exit=actual, reaped=actual is not None,
                owned_group_absent=absent, cleanup_errors=errors)


def own(command, *, out, panel, entry, calendar_seconds):
    if 'cost_admission' in entry:
        raise ValueError('cost entry requires the explicit cost owner')
    return _own(command,out=out,panel=panel,entry=entry,calendar_seconds=calendar_seconds)


def own_cost(command, *, out, panel, entry, calendar_seconds):
    from .panel_cost import contract
    contract(entry)
    return _own(command,out=out,panel=panel,entry=entry,calendar_seconds=calendar_seconds)


def _own(command, *, out, panel, entry, calendar_seconds):
    if type(calendar_seconds) is not int or calendar_seconds <= 0:
        raise ValueError("positive prospective calendar bound required")
    out.mkdir()  # exclusive; no reuse/resume
    start, timed_out, received = time.monotonic(), False, []
    watch=None;slot_watchdog=False;watchdog_receipt=None;attempt=None;environment=None;notify_read=None;notify_write=None
    if 'operation_budget' in entry:
        from .panel_cost import operation_budget
        from .panel_cost_runtime import ENV_ATTEMPT,ENV_NOTIFY
        operation_budget(entry)
        attempt=secrets.token_hex(32);environment=dict(os.environ)
        environment[ENV_ATTEMPT]=attempt
        notify_read,notify_write=os.pipe()
        os.set_blocking(notify_read,False);os.set_blocking(notify_write,False)
        environment[ENV_NOTIFY]=str(notify_write)
    worker, owner_error = None, None
    cleanup = dict(actual_exit=None,reaped=False,owned_group_absent=False,cleanup_errors=[])
    # Defer ordinary termination BEFORE spawn. A signal between Popen and its
    # return cannot throw away the only owned child handle.
    def defer(signum, frame):
        received.append(signum)
    previous = {sig:signal.signal(sig, defer) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with (out / "worker.stdout").open("xb") as stdout, (out / "worker.stderr").open("xb") as stderr:
            try:
                worker = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True,
                    **({'env':environment,'pass_fds':(notify_write,)} if environment is not None else {}))
                if notify_write is not None:
                    os.close(notify_write);notify_write=None
                if attempt is not None:
                    from .panel_cost_runtime import Watch
                    watch=Watch(entry,panel,attempt,worker.pid,start,notify_fd=notify_read)
                atomic_json(out / "launch.json", dict(owner_pid=os.getpid(), worker_pid=worker.pid,
                    worker_pgid=worker.pid, start_unix=time.time(), calendar_seconds=calendar_seconds,
                    command=command,launch_monotonic=start,
                    timing_binding=None if watch is None else watch.bound))
                while worker.poll() is None and not received:
                    slot_deadline=None
                    if watch is not None:
                        observed=watch.observe();slot_deadline=observed['deadline']
                        if observed['expired']:
                            slot_watchdog=True;watchdog_receipt=dict(observed,observed_monotonic=time.monotonic());break
                    remaining = calendar_seconds - (time.monotonic()-start)
                    if remaining <= 0:
                        timed_out = True
                        break
                    try:
                        wait_seconds=min(1.,remaining)
                        if slot_deadline is not None:
                            slot_remaining=slot_deadline-time.monotonic()
                            if slot_remaining<=0:
                                slot_watchdog=True;watchdog_receipt=dict(observed,expired=True,
                                    observed_monotonic=time.monotonic());break
                            wait_seconds=min(wait_seconds,slot_remaining)
                        if watch is None:worker.wait(timeout=wait_seconds)
                        else:watch.wait(wait_seconds)
                    except subprocess.TimeoutExpired:
                        pass
                # Observe completion against the collection deadline BEFORE
                # cleanup/reconstruction. Late actual0 is still a capped run.
                if time.monotonic() > start + calendar_seconds:
                    timed_out = True
            except BaseException as error:
                owner_error = f"{type(error).__name__}: {error}"
            finally:
                if worker is not None:
                    try:
                        cleanup = drain_owned(worker)
                    except BaseException as error:
                        cleanup["cleanup_errors"].append(f"UNKNOWN_DRAIN: {type(error).__name__}")
                        cleanup["actual_exit"] = worker.poll()
                        cleanup["reaped"] = cleanup["actual_exit"] is not None
                        # Unknown group state remains false, never presumed gone.
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    tally = reconstruct(panel, entry)
    actual = cleanup["actual_exit"]
    complete = (actual == 0 and cleanup["reaped"] and cleanup["owned_group_absent"]
        and not cleanup["cleanup_errors"] and not owner_error and not received
        and not timed_out and not slot_watchdog and tally["valid_completed_slots"] == entry["slots"])
    timing_result=None
    if watch is not None:
        try:timing_result=watch.reconstruct(tally['input_freeze_valid'])
        except Exception as error:
            timing_result=dict(timing_evidence_valid=False,operation_overruns=[],
                reconstruction_error=f'{type(error).__name__}: {error}')
    for fd in (notify_read,notify_write):
        if fd is not None:os.close(fd)
    if timing_result is not None and tally['input_freeze_valid']:
        # The in-scope START fsync may be the interrupted operation itself. A
        # received authenticated live START still proves that slot was attempted.
        for frame in timing_result.get('live_starts',[]):
            key=str(frame['slot'])
            if tally.get('incomplete_slots',{}).get(key)=='UNATTEMPTED':
                tally['incomplete_slots'][key]='INTERRUPTED_AFTER_LIVE_TIMING_START'
                tally['attempted_slots']+=1;tally['unattempted_slots']-=1
    overruns=bool(timing_result and timing_result['operation_overruns'])
    if timing_result is not None and (not timing_result['timing_evidence_valid'] or overruns):
        complete=False
    terminal = dict(status="ATTEMPT_COMPLETE" if complete else "FAILED_OR_INTERRUPTED",
        actual_worker_exit=actual, worker_reaped=cleanup["reaped"], timed_out=timed_out,
        owned_group_absent=cleanup["owned_group_absent"], cleanup_errors=cleanup["cleanup_errors"],
        owner_error=owner_error, owner_signals=received, elapsed_seconds=time.monotonic()-start,
        **tally, fidelity_qualified=False, production_go=False)
    terminal['slot_watchdog_stopped_worker']=slot_watchdog
    terminal['slot_watchdog_receipt']=watchdog_receipt
    if watch is not None:
        terminal['slot_timing']=timing_result
    # Preserve actual/unknown exit in the owner log even if disk publication
    # itself fails. That failure is never an experiment success.
    print(json.dumps(terminal, sort_keys=True), flush=True)
    atomic_json(out / "owner-terminal.json", terminal)
    return 0 if complete else 124 if timed_out or slot_watchdog or overruns else actual if actual is not None and actual > 0 else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entry", type=Path, required=True)
    p.add_argument("--expected-entry-sha", required=True)
    p.add_argument("--team-pool", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument('--cost-admission',action='store_true')
    args = p.parse_args()
    blob = args.entry.read_bytes()
    if hashlib.sha256(blob).hexdigest() != args.expected_entry_sha:
        raise ValueError("reviewed entry hash mismatch")
    entry = json.loads(blob)
    panel = args.out / "panel"
    command = [sys.executable, "-m", "gpu_engine.collect_fidelity_panel", "--entry", str(args.entry),
        "--expected-entry-sha", args.expected_entry_sha, "--team-pool", str(args.team_pool), "--out", str(panel)]
    if args.cost_admission:command.append('--cost-admission')
    raise SystemExit((own_cost if args.cost_admission else own)(command, out=args.out, panel=panel, entry=entry,
                         calendar_seconds=entry["calendar_seconds"]))


if __name__ == "__main__":
    main()
