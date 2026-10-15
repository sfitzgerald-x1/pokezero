"""Prospective, root-local continuation reservations; never launch authority.

Reservations consume action VALUES, not successful bank digests. All eight
cells remain consumed after a crash, refusal, missing capture, or failed run.
Previously collected banks are reusable only through their owning validators.
An independently reviewed history declaration is necessary: a caller-supplied
successful donor list cannot establish absence of other attempts. Coverage is
limited to that declaration and this journal, not a whole-machine census.

Like the ledger, hashes detect drift, not deliberate operator circumvention.
Legacy preparation remains explicitly non-exhaustive and is not upgraded.
"""
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat

from .search_over_raw import digest, require
from .search_over_raw_ledger import _write_new
from .search_over_raw_outcomes import snapshot_identity
from .search_over_raw_outcome_run import ContinuationRunContract


HISTORY_SCHEMA = "pokezero.search-over-raw.continuation-history-review.v1"
FRESH_HISTORY_SCHEMA = "pokezero.search-over-raw.continuation-history-review.v2"
NAMESPACE_SCHEMA = "pokezero.search-over-raw.fresh-continuation-namespace.v1"
JOURNAL_SCHEMA = "pokezero.search-over-raw.continuation-admission.v1"
ATTEMPT_SCHEMA = "pokezero.search-over-raw.continuation-reservation.v1"
_FLAGS = dict(collection_authorized=False, holdout_authorized=False,
    execution_ready=False, runtime_qualified=False, retry_authorized=False)


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _path(path):
    """Reject symlinks in any component, including a missing owner's ancestors."""
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink in admission path")
    require(path == path.resolve(), "canonical admission path required")
    return path


def _load(path):
    path = _path(path)
    require(path.is_file() and path.stat().st_size <= 8_000_000, "bounded admission receipt required")
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate admission JSON field")
            result[key] = value
        return result
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite admission JSON")))
    require(type(value) is dict, "admission receipt must be an object")
    return value


def _inventory(directory):
    directory = _path(directory)
    if not directory.exists():
        return None  # Missing is uncertainty, never evidence of no attempt.
    require(directory.is_dir(), "prior owner directory required")
    rows = {}
    for path in sorted(directory.rglob("*")):
        _path(path)
        require(path.is_dir() or path.is_file(), "ordinary prior owner files required")
        if path.is_file():
            rows[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest(rows)


def _root_identity(plan):
    from .search_over_raw_root_plan import ArchivedRootEvaluationPlan
    require(type(plan) is ArchivedRootEvaluationPlan, "actual archived root plan required")
    root = next(r for r in plan.source["roots"] if r["root_id"] == plan.root_id)
    _, _, snapshot = plan.archive.selected(root)
    from .search_over_raw_world_inputs import _state_payload
    snapshot_sha = digest(snapshot_identity(snapshot))
    private_sha = digest(_state_payload(snapshot))
    # Extraction can invoke callbacks. Finish it before revalidate's final
    # source/owner byte checks, never after them. Bind the extracted snapshot
    # to the already frozen private archive rather than a fresh callback value.
    receipt = plan.revalidate()
    require(private_sha == digest(_state_payload(snapshot)) == plan.bound["private_auditor_snapshot_sha256"],
        "extracted private snapshot differs from frozen archive")
    raw = plan.readers["raw"]
    return dict(root_id=plan.root_id, public_record_sha256=plan.bound["public_record_sha256"],
        snapshot_sha256=snapshot_sha,
        continuation_contract=receipt["continuation_contract"],
        raw_owner=dict(root_id=plan.root_id, selected_sha256=digest(plan.selections["raw"]),
            runtime_sha256=raw.runtime_sha, full_public_request_sha256=raw.public_request_sha,
            directory=str(_path(raw.directory)), job_sha256=raw.job_sha,
            attempt_binding_sha256=raw.binding))


def history_review_template(plan, *, ledger_directory, prior_attempts=()):
    """Uncleared draft only; bind the reviewed file BEFORE ledger creation.

    Prior attempts are explicit redacted descriptors (directory, target_actions,
    job_sha256, attempt_binding_sha256, runtime_sha256, inventory_sha256).
    They may describe missing or failed owners and never count as scored banks.
    The reviewer must resolve coverage; this function cannot assert it.
    """
    return dict(schema=HISTORY_SCHEMA, status="UNKNOWN_PRIOR_COVERAGE",
        ledger_directory=str(_path(ledger_directory)), root_identity=_root_identity(plan),
        prior_attempts=deepcopy(list(prior_attempts)),
        all_prior_attempts_in_declared_root_scope_accounted_for=False,
        whole_machine_census=False, **_FLAGS)


def fresh_history_review_template(contract, *, ledger_directory, raw_runtime_sha256,
                                  continuation_runtime_sha256):
    """Prospective recipe, NOT a declaration about already observed roots.

    This explicitly versioned alternative breaks the v1 review/ledger/root
    cycle. The review can be pinned before ledger creation because neither
    future root bytes nor the contract's own input-hash envelope occurs here.
    Reviewer clearance remains external; the template cannot assert freshness.
    """
    from .search_over_raw_reconciled import validate_contract, controller_input_hashes
    validate_contract(contract)
    require(_sha(raw_runtime_sha256) and _sha(continuation_runtime_sha256),
        "prospective raw and continuation runtime recipes required")
    recipe = {k: deepcopy(v) for k, v in contract.items() if k not in
        {"input_hashes", "execution_ready", "collection_authorized", "remaining_gates"}}
    directory = _path(ledger_directory)
    return dict(schema=FRESH_HISTORY_SCHEMA, status="UNKNOWN_FRESH_NAMESPACE",
        ledger_directory=str(directory), contract_recipe_sha256=digest(recipe),
        exploration_slots=deepcopy(contract["panels"]["exploration"]["root_slots"]),
        raw_owner_namespace=str(directory / "shared-raw-runs"),
        raw_runtime_sha256=raw_runtime_sha256,
        continuation_runtime_sha256=continuation_runtime_sha256,
        controller_input_hashes=controller_input_hashes(), prior_attempts=[],
        reviewed_before_first_source_work=False, whole_machine_census=False, **_FLAGS)


def _fresh_review(ledger, review_path):
    ledger.verify()
    review_path = _path(review_path)
    review = _load(review_path)
    review_sha = hashlib.sha256(review_path.read_bytes()).hexdigest()
    require(ledger.contract["input_hashes"].get(str(review_path)) == review_sha,
        "fresh namespace review must be prospectively bound before ledger creation")
    expected = fresh_history_review_template(ledger.contract, ledger_directory=ledger.directory,
        raw_runtime_sha256=review.get("raw_runtime_sha256"),
        continuation_runtime_sha256=review.get("continuation_runtime_sha256"))
    expected.update(status="CLEAR_FRESH_DECLARED_NAMESPACE", reviewed_before_first_source_work=True)
    require(digest(review) == digest(expected), "unknown or mismatched fresh namespace review; no imports")
    require(all(ledger.contract["input_hashes"].get(p) == h
        for p, h in review["controller_input_hashes"].items()), "fresh controller code must be pinned")
    return review, review_sha


def initialize_fresh_namespace(ledger, *, history_review_path):
    """Create-only ownership namespace BEFORE source/raw work, no worker launch.

    Interrupted initialization is stranded. It is never repaired, reused as a
    fresh declaration, or upgraded from v1. The actual raw supervisor must use
    the canonical root directory and its own exclusive creation semantics.
    """
    from .search_over_raw_reconciled import ReconciledPhaseALedger
    require(type(ledger) is ReconciledPhaseALedger, "actual reconciled ledger required")
    review, review_sha = _fresh_review(ledger, history_review_path)
    parent = _path(ledger.directory / "continuation-admission")
    raw_parent = _path(review["raw_owner_namespace"])
    require(not parent.exists() and not raw_parent.exists(), "fresh namespace already occupied")
    parent.mkdir(mode=0o700, exist_ok=False)
    raw_parent.mkdir(mode=0o700, exist_ok=False)
    claim = dict(schema=NAMESPACE_SCHEMA, contract_sha256=ledger.contract_sha256,
        review_path=str(_path(history_review_path)), review_sha256=review_sha, review=review, **_FLAGS)
    _write_new(parent / "namespace.json", claim)
    return dict(namespace_initialized=True, scientific_result_validated=False, **_FLAGS)


class ContinuationAdmissionJournal:
    """Create-only reservation namespace, serialized independently per root.

    Initialization and reservation are local writes, not scientific admission.
    A stranded or malformed journal blocks THIS root, not another configuration
    or root. No rollback, repair, retry, scoring, or production launch is here.
    """
    def __init__(self, ledger, *, root_id):
        from .search_over_raw_reconciled import ReconciledPhaseALedger
        require(type(ledger) is ReconciledPhaseALedger, "actual reconciled ledger required")
        self.ledger, self.root_id = ledger, root_id
        self.directory = _path(ledger.directory / "continuation-admission" / digest(dict(root_id=root_id)))

    @classmethod
    def create(cls, ledger, plan, *, history_review_path):
        journal = cls(ledger, root_id=plan.root_id)
        ledger.verify()
        require(digest(plan.contract) == ledger.contract_sha256, "plan/ledger contract mismatch")
        review_path = _path(history_review_path)
        review_sha = hashlib.sha256(review_path.read_bytes()).hexdigest()
        require(ledger.contract["input_hashes"].get(str(review_path)) == review_sha,
            "history review must be prospectively bound before ledger creation")
        review = _load(review_path)
        identity = _root_identity(plan)
        journal._review_owners(review, identity, review_path, review_sha)
        require(digest(identity) == digest(_root_identity(plan)), "root changed during history review")
        ledger.verify()
        # An interrupted initializer is intentionally stranded, never reopened
        # as a fresh root. Do not create the supervisor's per-attempt directory.
        _path(journal.directory.parent).mkdir(exist_ok=True)
        journal.directory.mkdir(exist_ok=False, mode=0o700)
        (journal.directory / "attempts").mkdir(mode=0o700)
        (journal.directory / "runs").mkdir(mode=0o700)
        _write_new(journal.directory / "baseline.json", dict(schema=JOURNAL_SCHEMA,
            contract_sha256=ledger.contract_sha256, review_path=str(review_path), review_sha256=review_sha,
            review=review, root_identity=identity, **_FLAGS))
        with journal._locked():
            journal._check(plan)
        return journal

    def _review_owners(self, review, identity, review_path, review_sha):
        if review.get("schema") == FRESH_HISTORY_SCHEMA:
            current, sha = _fresh_review(self.ledger, review_path)
            require(digest(current) == digest(review) and sha == review_sha, "fresh review drift")
            parent = _path(self.directory.parent)
            claim = _load(parent / "namespace.json")
            require(digest(claim) == digest(dict(schema=NAMESPACE_SCHEMA, contract_sha256=self.ledger.contract_sha256,
                review_path=str(_path(review_path)), review_sha256=review_sha, review=review, **_FLAGS)),
                "fresh namespace was not initialized under this contract")
            names = {digest(dict(root_id=r["root_id"])) for r in review["exploration_slots"]}
            require(all(p.name == "namespace.json" or p.name in names and _path(p).is_dir()
                for p in parent.iterdir()), "undeclared root in fresh namespace")
            raw_parent = _path(review["raw_owner_namespace"])
            require(raw_parent.is_dir() and all(p.name in names and _path(p).is_dir()
                for p in raw_parent.iterdir()), "undeclared raw owner in fresh namespace")
            slot = next((r for r in review["exploration_slots"] if r["root_id"] == self.root_id), None)
            require(slot is not None and identity["continuation_contract"]["selection_seed"] == slot["source_seed"],
                "fresh root is not in the fixed exploration slots")
            require(identity["raw_owner"]["directory"] == str(raw_parent / digest(dict(root_id=self.root_id)))
                and identity["raw_owner"]["runtime_sha256"] == review["raw_runtime_sha256"],
                "fresh first raw owner path or runtime substituted")
            return {}  # v2 has no imports; subsequent reservations consume work.
        require(not (self.directory.parent / "namespace.json").exists(), "v1 import into fresh namespace forbidden")
        require(set(review) == {"schema", "status", "ledger_directory", "root_identity", "prior_attempts",
            "all_prior_attempts_in_declared_root_scope_accounted_for", "whole_machine_census", *_FLAGS}
            and review["schema"] == HISTORY_SCHEMA and review["status"] == "CLEAR_DECLARED_ROOT_HISTORY"
            and digest(review["root_identity"]) == digest(identity)
            and review["ledger_directory"] == str(self.ledger.directory)
            and review["all_prior_attempts_in_declared_root_scope_accounted_for"] is True
            and review["whole_machine_census"] is False
            and all(review[k] is v for k, v in _FLAGS.items()), "unknown or mismatched prior coverage")
        return self._imports(review, identity)

    @contextmanager
    def _locked(self):
        _path(self.directory)
        require(self.directory.is_dir(), "initialized root admission journal required")
        lock = _path(self.directory / "lock")
        fd = os.open(lock, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            info = os.fstat(fd)
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                and (info.st_dev, info.st_ino) == (lock.stat().st_dev, lock.stat().st_ino),
                "admission lock inode changed")
            yield
            _path(lock)
            require((info.st_dev, info.st_ino) == (lock.stat().st_dev, lock.stat().st_ino),
                "admission lock replaced during transaction")
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _imports(self, review, identity):
        rows = review["prior_attempts"]
        require(type(rows) is list, "explicit reviewed prior attempt roster required")
        owners, directories = {}, set()
        mask = identity["continuation_contract"]["root_binding"]["legal_action_mask"]
        for row in rows:
            require(type(row) is dict and set(row) == {"directory", "target_actions", "job_sha256",
                "attempt_binding_sha256", "runtime_sha256", "inventory_sha256"}, "redacted prior attempt descriptor required")
            directory = _path(row["directory"])
            require(str(directory) == row["directory"] and str(directory) not in directories
                and not directory.is_relative_to(self.directory), "distinct external prior owners required")
            directories.add(str(directory))
            actions = row["target_actions"]
            require(type(actions) is list and actions and all(type(a) is int and 0 <= a < len(mask)
                and mask[a] for a in actions) and actions == sorted(set(actions)), "exact legal prior action values required")
            require(all(_sha(row[k]) for k in ("job_sha256", "attempt_binding_sha256", "runtime_sha256"))
                and (row["inventory_sha256"] is None or _sha(row["inventory_sha256"])), "prior owner hashes required")
            require(_inventory(directory) == row["inventory_sha256"], "reviewed prior inventory drift")
            # No unpickling and no inference of exit0. A missing/malformed owner
            # still consumes the review's entire declared target set.
            if (directory / "job.pickle").exists():
                require(hashlib.sha256((directory / "job.pickle").read_bytes()).hexdigest() == row["job_sha256"],
                    "reviewed prior job drift")
            for action in actions:
                require(action not in owners, "prior action redraw or overlapping owners")
                owners[action] = row
        require(len({r["runtime_sha256"] for r in rows}) <= 1, "prior continuation runtime mismatch")
        return owners

    def _durable(self, baseline):
        """Byte guards with no owner callbacks; run progress is not a score."""
        paths = [self.directory / "baseline.json", self.directory / "lock", _path(baseline["review_path"])]
        if baseline["review"].get("schema") == FRESH_HISTORY_SCHEMA:
            paths.append(self.directory.parent / "namespace.json")
        attempts = _path(self.directory / "attempts")
        runs = _path(self.directory / "runs")
        require(attempts.is_dir() and runs.is_dir(), "ordinary admission directories required")
        paths += sorted(attempts.iterdir())
        run_names = []
        for run in sorted(runs.iterdir()):
            require(_path(run).is_dir(), "ordinary reserved run directory required")
            run_names.append(run.name)
            # Do not freeze changing terminal/log output during a healthy run.
            # The create-only owner claim and frozen job may not change.
            paths += [run / name for name in ("claim.json", "job.pickle")]
        identity = baseline["root_identity"]
        paths.append(self.ledger.directory / "owned-shared-raw" / (digest(dict(root_id=self.root_id))+".json"))
        mask = identity["continuation_contract"]["root_binding"]["legal_action_mask"]
        paths += [self.ledger.directory / "owned-shared-cells" / (
            digest(dict(root_id=self.root_id, action=a, replicate=r))+".json")
            for a, legal in enumerate(mask) if legal for r in range(8)]
        hashes = {}
        for path in paths:
            _path(path)
            require(not path.exists() or path.is_file(), "ordinary admission file required")
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        return dict(files=hashes, runs=run_names,
            imports={r["directory"]:_inventory(r["directory"]) for r in baseline["review"]["prior_attempts"]})

    def _check_run(self, directory, record):
        directory = _path(directory)
        if (directory / "claim.json").exists():
            claim = _load(directory / "claim.json")
            require(claim.get("job_sha256") == record["job_sha256"]
                and claim.get("binding_sha256") == record["attempt_binding_sha256"]
                and digest(claim.get("contract")) == digest(record["run_contract"]),
                "reservation/supervisor claim mismatch")
        if (directory / "job.pickle").exists():
            require(hashlib.sha256((directory / "job.pickle").read_bytes()).hexdigest() == record["job_sha256"],
                "reserved owner job bytes changed")

    def _check(self, plan):
        self.ledger.verify()
        require(plan.root_id == self.root_id and digest(plan.contract) == self.ledger.contract_sha256,
            "journal root/contract mismatch")
        require({p.name for p in self.directory.iterdir()} == {"baseline.json", "lock", "attempts", "runs"},
            "incomplete or unexpected root journal files")
        baseline = _load(self.directory / "baseline.json")
        require(set(baseline) == {"schema", "contract_sha256", "review_path", "review_sha256", "review", "root_identity", *_FLAGS}
            and baseline["schema"] == JOURNAL_SCHEMA
            and baseline["contract_sha256"] == self.ledger.contract_sha256
            and all(baseline[k] is v for k, v in _FLAGS.items()), "journal baseline mismatch")
        durable = self._durable(baseline)
        review_path = _path(baseline["review_path"])
        require(self.ledger.contract["input_hashes"].get(str(review_path)) == baseline["review_sha256"]
            == hashlib.sha256(review_path.read_bytes()).hexdigest()
            and digest(_load(review_path)) == digest(baseline["review"]), "bound history review drift")
        review = baseline["review"]
        identity = _root_identity(plan)
        require(digest(identity) == digest(baseline["root_identity"]),
            "journal private root, continuation law or actual raw owner drift")
        owners = self._review_owners(review, identity, review_path, baseline["review_sha256"])
        attempts = _path(self.directory / "attempts")
        runs = _path(self.directory / "runs")
        require(attempts.is_dir() and runs.is_dir(), "ordinary admission directories required")
        records = {}
        mask = identity["continuation_contract"]["root_binding"]["legal_action_mask"]
        for path in sorted(attempts.iterdir()):
            record = _load(path)
            require(set(record) == {"schema", "root_identity_sha256", "job_sha256", "job_inputs_sha256",
                "attempt_binding_sha256", "runtime_sha256", "run_contract", "target_actions", "replicates", *_FLAGS}
                and record["schema"] == ATTEMPT_SCHEMA and path.name == digest(record)+".json"
                and record["root_identity_sha256"] == digest(identity)
                and digest(record["replicates"]) == digest(list(range(8)))
                and all(record[k] is v for k, v in _FLAGS.items()), "invalid or partial action reservation")
            require(all(_sha(record[k]) for k in ("job_sha256", "job_inputs_sha256", "attempt_binding_sha256", "runtime_sha256"))
                and digest(record["run_contract"]) == digest(ContinuationRunContract().receipt()), "reservation job/runtime/law hashes required")
            actions = record["target_actions"]
            require(type(actions) is list and actions
                and all(type(a) is int and 0 <= a < len(mask) and mask[a] for a in actions)
                and actions == sorted(set(actions)), "legal reserved action values required")
            owner = dict(record, directory=str(runs / path.stem))
            for action in actions:
                require(action not in owners, "reserved action redraw or overlapping owners")
                owners[action] = owner
            records[path.stem] = record
        require(len({r["runtime_sha256"] for r in owners.values()}) <= 1, "reserved continuation runtime mismatch")
        if review["schema"] == FRESH_HISTORY_SCHEMA:
            require(all(r["runtime_sha256"] == review["continuation_runtime_sha256"] for r in owners.values()),
                "reservation runtime differs from prospective fresh recipe")
        require(all(_path(p).is_dir() and p.name in records for p in runs.iterdir()), "undeclared owned run in journal")
        for run in runs.iterdir():
            self._check_run(run, records[run.name])
        raw_path = self.ledger.directory / "owned-shared-raw" / (digest(dict(root_id=self.root_id))+".json")
        if raw_path.exists():
            require(digest(_load(raw_path)) == digest(identity["raw_owner"]), "actual shared raw owner mismatch")
        # Only this fixed root's legal cells: no expensive whole-history census.
        for action, legal in enumerate(mask):
            if not legal:
                continue
            for replicate in range(8):
                path = self.ledger.directory / "owned-shared-cells" / (
                    digest(dict(root_id=self.root_id, action=action, replicate=replicate))+".json")
                if path.exists():
                    cell = _load(path)
                    require(action in owners and cell.get("root_id") == self.root_id
                        and type(cell.get("action")) is int and cell["action"] == action
                        and type(cell.get("replicate")) is int and cell["replicate"] == replicate
                        and all(cell.get(k) == owners[action][k] for k in
                            ("directory", "job_sha256", "attempt_binding_sha256")),
                        "undeclared or redrawn prior cell owner; coverage incomplete")
        require(digest(identity) == digest(_root_identity(plan)), "root changed during journal reconciliation")
        self.ledger.verify()
        require(durable == self._durable(baseline), "durable admission bytes changed during owner callbacks")
        return baseline, owners, records

    def consumed_actions(self, plan):
        with self._locked():
            _, owners, _ = self._check(plan)
            return tuple(sorted(owners))

    def reserve(self, plan, job):
        """Irreversibly reserve ALL cells before an independently authorized run."""
        with self._locked():
            baseline, owners, before = self._check(plan)
            durable = self._durable(baseline)
            plan.verify_prepared_job(job)
            prepared = deepcopy(plan._prepared[id(job)])
            record = dict(schema=ATTEMPT_SCHEMA, root_identity_sha256=digest(baseline["root_identity"]),
                job_sha256=prepared["pickle_sha256"], job_inputs_sha256=prepared["inputs"],
                attempt_binding_sha256=plan.continuation_attempt_binding(job),
                runtime_sha256=job.input_binding()["runtime_sha256"], run_contract=ContinuationRunContract().receipt(),
                target_actions=list(prepared["target_actions"]), replicates=list(range(8)), **_FLAGS)
            require(not set(record["target_actions"]) & set(owners), "action already consumed; retry/redraw forbidden")
            if baseline["review"]["schema"] == FRESH_HISTORY_SCHEMA:
                require(record["runtime_sha256"] == baseline["review"]["continuation_runtime_sha256"],
                    "continuation runtime differs from prospective fresh recipe")
            require(not owners or record["runtime_sha256"] == next(iter(owners.values()))["runtime_sha256"],
                "continuation runtime differs from consumed work")
            # Finish all callback-bearing reads, then recheck original durable
            # state. One create-only record commits the entire action set.
            _, _, after = self._check(plan)
            import pickle
            require(digest(before) == digest(after) and prepared == plan._prepared[id(job)]
                and prepared["inputs"] == digest(job.input_binding())
                and prepared["pickle_sha256"] == hashlib.sha256(pickle.dumps(job)).hexdigest(),
                "reservation inputs changed during callbacks")
            require(durable == self._durable(baseline), "durable admission bytes changed during reservation callbacks")
            reservation = digest(record)
            _write_new(self.directory / "attempts" / (reservation+".json"), record)
            return dict(reservation_sha256=reservation, record=record,
                run_directory=str(self.directory / "runs" / reservation),
                reserved_cells=8*len(record["target_actions"]), scientific_result_validated=False,
                declared_history_reconciled=True, whole_machine_census=False, **_FLAGS)

    def verify_reservation(self, plan, job, reservation):
        """Reservation identity check, never sufficient scientific admission."""
        with self._locked():
            baseline, _, records = self._check(plan)
            durable = self._durable(baseline)
            plan.verify_prepared_job(job)
            key = reservation["reservation_sha256"]
            require(_sha(key) and digest(records.get(key)) == digest(reservation["record"]), "unowned or changed reservation")
            record = records[key]
            prepared = plan._prepared[id(job)]
            directory = _path(self.directory / "runs" / key)
            require(reservation["run_directory"] == str(directory)
                and record["job_sha256"] == prepared["pickle_sha256"]
                and record["job_inputs_sha256"] == prepared["inputs"]
                and record["target_actions"] == list(prepared["target_actions"])
                and record["attempt_binding_sha256"] == plan.continuation_attempt_binding(job),
                "reservation job, aliases or actual owner changed")
            self._check_run(directory, record)
            _, _, final_records = self._check(plan)
            require(digest(records) == digest(final_records), "reservation changed during verification callbacks")
            import pickle
            require(record["job_inputs_sha256"] == digest(job.input_binding())
                and record["job_sha256"] == hashlib.sha256(pickle.dumps(job)).hexdigest()
                and all(reservation.get(k) is v for k, v in _FLAGS.items()),
                "job or reservation authority changed during final verification")
            require(durable == self._durable(baseline), "durable admission bytes changed during verification callbacks")
            return dict(reservation_identity_validated=True, **_FLAGS)

    def prepare_new_job(self, plan, *, config, runtime, _synthetic_job_class=None):
        """No-launch preparation from consumed attempts, not supplied successes.

        Reservation is a separate serialized transaction. A competing prepare
        can observe the same new action, but only ONE reservation can commit.
        """
        with self._locked():
            baseline, owners, before = self._check(plan)
            durable = self._durable(baseline)
            selected = plan.revalidate()["root_evidence"]["actions"]
            targets = tuple(sorted(set(selected.values())-set(owners)))
            job = None
            if targets:
                job, _ = plan.prepare_continuation_job(config=config, runtime=runtime,
                    actions_to_collect=targets, _synthetic_job_class=_synthetic_job_class)
            _, after_owners, after = self._check(plan)
            require(before == after and owners == after_owners, "consumed history changed during preparation")
            require(durable == self._durable(baseline), "durable admission bytes changed during preparation callbacks")
            return job, dict(consumed_action_values=sorted(owners), new_action_values=list(targets),
                declared_history_reconciled=True, whole_machine_census=False,
                missing_consumed_cells_remain_uncertain=True, reservation_required_before_launch=True, **_FLAGS)
