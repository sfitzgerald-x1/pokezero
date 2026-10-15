"""Reserved owned continuations for one staged exploration configuration/root.

This joins existing archived selectors to the actual supervisor and independent
owning reader. It does NOT generate source roots, run selectors, qualify a
runtime, summarize an entire stage, or grant collection/holdout authority.
The scientific entry requires separately admitted exploration. Disposable
synthetic tests have a distinct, explicitly non-scientific entry condition.
"""
from copy import deepcopy
import hashlib
import pickle

from .search_over_raw import SearchConfiguration, digest, require
from .search_over_raw_ledger import _write_new
from .search_over_raw_continuation_admission import (
    ContinuationAdmissionJournal, FRESH_HISTORY_SCHEMA, _load, _path,
)
from .search_over_raw_outcome_run import OwnedContinuationJob, OwningRunValidator, supervise_outcome_run
from .search_over_raw_reconciled import ReconciledPhaseALedger, controller_input_hashes
from .search_over_raw_root_plan import ArchivedRootEvaluationPlan, OwnedContinuationBank, _owner_inputs


def produce_reserved_continuation(ledger, stage, configuration, *, plan, journal, config, runtime,
                                 _synthetic_job_class=None):
    """One create-only root attempt; consumed missing/capped work is never redone.

    Returns an actual OwnedContinuationBank only after normal supervisor exit,
    independent process/transport validation and the prepared-plan join. A
    normal zero-work selector launches no continuation. Operational failures
    after claiming the root stop this ledger; there is no repair/retry path.
    """
    require(type(ledger) is ReconciledPhaseALedger and type(plan) is ArchivedRootEvaluationPlan
        and type(journal) is ContinuationAdmissionJournal and journal.ledger is ledger
        and journal.root_id == plan.root_id, "exact archived plan/ledger/journal required")
    ledger.verify()
    cfg = next((c for c in ledger._stage_configs(stage) if c.identity == configuration), None)
    require(cfg is not None, "stage-local registered producer configuration required")
    require(not (stage == "A3" and cfg.seconds == 1.),
        "A3 one-second producer must reuse frozen A2, never launch")
    synthetic = _synthetic_job_class is not None
    if synthetic:
        require(isinstance(_synthetic_job_class, type) and issubclass(_synthetic_job_class, OwnedContinuationJob)
            and getattr(_synthetic_job_class, "synthetic_process_fixture", False) is True
            and plan.allow_synthetic is True and ledger.contract["execution_ready"] is False
            and ledger.contract["collection_authorized"] is False,
            "explicit unadmitted disposable synthetic scope required")
        require(all(ledger.contract["input_hashes"].get(p) == h for p, h in controller_input_hashes().items()),
            "synthetic producer code must be pinned")
    else:
        require(plan.allow_synthetic is False, "synthetic selectors cannot enter scientific production")
        ledger.verify_exploration(stage, configuration)
        require(plan.source_owner is not None and plan.source_owner.ledger is ledger,
            "production continuation requires original owned fresh source")
    require(digest(plan.contract) == ledger.contract_sha256
        and tuple(c.identity for c in plan.configurations) ==
            (SearchConfiguration("raw", seconds=1.).identity, configuration),
        "single configuration plus shared raw plan required; no factorial loop")
    require(not (ledger.directory / "selection.json").exists()
        and not (stage == "A2" and (ledger.directory / "a2-freeze.json").exists()), "exploration is frozen")
    claims = _path(ledger.directory / "producer-root-claims")
    claims.mkdir(exist_ok=True, mode=0o700)
    claim = claims / (digest(dict(stage=stage, configuration=configuration, root_id=plan.root_id))+".json")
    frozen = dict(contract_sha256=ledger.contract_sha256, stage=stage, configuration=configuration,
        root_id=plan.root_id, plan_inputs_sha256=plan._frozen, runtime_sha256=digest(runtime),
        synthetic_process_fixture=synthetic, retry_authorized=False, holdout_authorized=False)
    # A failed/interrupted owner cannot be resumed by invoking this function.
    _write_new(claim, frozen)
    claim_bytes = claim.read_bytes()
    def claim_guard():
        ledger.verify()
        require(_path(claim).read_bytes() == claim_bytes and digest(_load(claim)) == digest(frozen),
            "producer claim changed")
    def finish_without_run():
        # Reconciliation is callback-bearing even when no job is prepared.
        journal.consumed_actions(plan)
        plan.revalidate()
        require(plan._inputs() == plan._frozen, "producer plan changed before no-run return")
        claim_guard()
    try:
        baseline = _load(journal.directory / "baseline.json")
        require(baseline["review"]["schema"] == FRESH_HISTORY_SCHEMA,
            "producer requires prospectively declared fresh namespace, not v1 import")
        journal.consumed_actions(plan)
        receipt = plan.revalidate()
        status = receipt["root_evidence"]["selector_statuses"][configuration]
        if status == "UNQUALIFIED_AT_CEILING":
            finish_without_run()
            return None, dict(status=status, continuation_launched=False, new_action_values=[],
                scored_search_action=False, synthetic_process_fixture=synthetic,
                collection_authorized=False, holdout_authorized=False, retry_authorized=False)
        require(status == "SELECTED", "productive owned selector required")
        job, prepared = journal.prepare_new_job(plan, config=config, runtime=runtime,
            _synthetic_job_class=_synthetic_job_class)
        if job is None:
            finish_without_run()
            return None, dict(status="NO_NEW_ACTION_VALUES", continuation_launched=False, **prepared)
        reservation = journal.reserve(plan, job)
        frozen_job = deepcopy(job.input_binding())
        def verify():
            ledger.verify()
            claim_guard()
            if not synthetic:
                ledger.verify_exploration(stage, configuration)
            journal.verify_reservation(plan, job, reservation)
            require(digest(job.input_binding()) == digest(frozen_job), "producer job changed")
            claim_guard()  # Journal/job callbacks must not alter the original claim.
        def validate_shape(result):
            # Terminal ownership is deliberately validated AFTER terminal.json
            # exists, not fabricated inside the supervisor's pre-terminal hook.
            require(type(result) is dict and digest(result.get("bank")) == result.get("bank_sha256")
                and digest({k: result.get(k) for k in frozen_job}) == digest(frozen_job)
                and digest(result.get("outcomes", {}).get("actions")) == digest(job.actions),
                "producer pre-terminal result shape drift")
            return True
        terminal = supervise_outcome_run(job, directory=reservation["run_directory"],
            binding_sha256=reservation["record"]["attempt_binding_sha256"],
            verify_admission=verify, validate_result=validate_shape)
        require(terminal.get("supervisor_exit_code") == 0 and terminal.get("worker_exit_code") == 0
            and terminal.get("status") == "OUTCOME_RUN_COMPLETE", "continuation producer operational failure")
        reader = OwningRunValidator(directory=reservation["run_directory"],
            job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),
            binding_sha256=reservation["record"]["attempt_binding_sha256"],
            root_binding=job.root_binding, expected_snapshot=job.snapshot, runtime=job.runtime,
            allow_synthetic=synthetic)
        result = terminal["result"]
        accepted = plan.accept_continuation_result(job, result, owning_reader=reader)
        bank = OwnedContinuationBank(plan, job, result, reader)
        result_sha = digest(result)
        accepted_sha = digest(accepted)
        readout_sha = digest(reader.last_validation)
        owner_inputs = _owner_inputs((plan,), (bank,))
        verify()  # Last callback-bearing reservation/runtime acceptance.
        require(_owner_inputs((plan,), (bank,)) == owner_inputs
            and digest(reader.last_validation) == readout_sha
            and digest(result) == result_sha and digest(accepted) == accepted_sha,
            "accepted producer ownership/result changed during final callbacks")
        claim_guard()
        return bank, dict(status="OWNED_CONTINUATIONS_VALIDATED", continuation_launched=True,
            reservation_sha256=reservation["reservation_sha256"],
            run_directory=reservation["run_directory"], new_action_values=prepared["new_action_values"],
            reserved_cells=reservation["reserved_cells"], synthetic_process_fixture=synthetic,
            result_sha256=result_sha, accepted_result_sha256=accepted_sha,
            runtime_qualified_by_this_code=False, collection_authorized=False,
            holdout_authorized=False, retry_authorized=False)
    except BaseException as error:
        if not (ledger.directory / "operational-stop.json").exists():
            try:
                ledger.stop(type(error).__name__)
            except BaseException as stop_error:
                error.add_note("Unable to persist operational stop: "+type(stop_error).__name__)
        raise
