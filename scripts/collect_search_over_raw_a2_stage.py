"""Concrete retained-root A2 stage, called only by a future reviewed producer.

No CLI, source replay, registration or runtime authority is provided. The
producer owns the single auditor shell and one unchanged external deadline;
the successful bank stays live for A3. Any stage failure clears the bank.
"""
import json
from pathlib import Path
import time

from qualify_search_over_raw_a1 import collect_root
from benchmark_search_over_raw_feasibility import save_new
from pokezero.mcts_eval.search_over_raw import require
from pokezero.mcts_eval.search_over_raw_archive import SealedComparisonBank
from pokezero.mcts_eval.search_over_raw_stages import ExplorationStages, a2_configurations, key


def collect_a2_from_bank(*, plan, bank, output, env, evaluator, factory, checkpoint_contract,
                         source, showdown, verify, deadline_at, ownership=None, progress_sink=None,
                         clock=time.perf_counter):
    """Run thirteen selectors/root, all paired outcomes, then seal A3 choices.

    Deadline is the producer's ORIGINAL absolute monotonic deadline, not a new
    stage allocation. This guard supplements—not replaces—the external process
    supervisor. A2's instrumented world observations establish original-team
    agreement only, not full current-world/tree-path equality or value accuracy.
    """
    require(isinstance(bank, SealedComparisonBank), "actual sealed comparison bank required")
    progress = None
    try:
        from math import isfinite
        require(type(deadline_at) in (int, float) and isfinite(deadline_at), "original finite deadline required")
        require(callable(verify), "producer provenance/deadline verifier required")
        output = Path(output).resolve()
        require(not output.is_relative_to(Path(__file__).resolve().parents[1])
            and not output.is_relative_to(Path(plan["historical_directory"]).resolve()),
            "stage output must be outside executing checkout and historical evidence")
        def guard():
            require(clock() < deadline_at, "original attempt deadline reached; no new stage clock")
            verify()
        guard()
        bank.validate_roster(plan["comparable_root_contract"]["root_slots"])
        manifest = bank.manifest()
        ledger = ExplorationStages(plan, manifest)
        output.mkdir(parents=True, exist_ok=False)
        roster = a2_configurations()
        slots = plan["phase_a_cohort"]["panels"]["exploration"]["root_slots"]
        progress = dict(stage="A2:prepared", status="COLLECTING_EXPLORATORY_A2",
            fixed_roster=[dict(root_id=slot["root_id"], configuration=key(cfg),
                status="UNSTARTED_UNCERTAIN", contrast_interval=[-1., 1.], action=None)
                for slot in slots for cfg in roster],
            continuation_roster=[dict(root_id=slot["root_id"], configuration=key(cfg), replicate=rep,
                status="UNSTARTED_UNCERTAIN", action=None, signed_outcome=None)
                for slot in slots for cfg in roster for rep in range(8)],
            roots_completed=0, selections_completed=0, continuations_completed=0,
            continuations_capped=0, scientific_strength_evidence=False, holdout_authorized=False,
            old_a1_not_promoted_to_a2=True, runtime_authorized=False)
        save_new(output / "stage-manifest.json", dict(bank=manifest,
            fixed_root_denominator=200, fixed_seed_denominator=32,
            full_selector_denominator=len(slots)*len(roster),
            full_continuation_alias_denominator=len(slots)*len(roster)*8,
            original_deadline_at=deadline_at, same_world_qualification_pending=True))
        for slot in slots:
            root_id = slot["root_id"]
            if root_id not in manifest["root_public_bindings"]:
                continue  # Original fourteen + source-validated missing, no redraw.
            guard()
            root, context, pending, snapshot = bank.selected_for_auditor(root_id)
            directory = output / f"source-{slot['source_seed']}" / f"root-{slot['root_slot']}"
            directory.mkdir(parents=True, exist_ok=False)
            collect_root(root=root, context=context, pending=pending, snapshot=snapshot,
                output=directory, progress=progress, env=env, evaluator=evaluator, factory=factory,
                contract=checkpoint_contract, source=source, showdown=showdown, verify=guard,
                ownership=ownership, namespace=plan["execution_source_contract"]["namespace"],
                completion_status="COLLECTED_A2_CENSORING_AWARE", boundary_censoring=True,
                configuration_roster=roster, matched_statistics_root=True)
            audit = json.loads((directory / "audit.json").read_text())
            selections = {key(cfg): json.loads((directory / f"{key(cfg)}-selected.json").read_text())
                for cfg in roster}
            ledger.record_root(audit, selections, public_record_sha256=root["public_record_sha256"],
                selection_seed=slot["source_seed"], legal_choices=sum(context.observation.legal_action_mask))
            if progress_sink is not None:
                progress_sink(progress)
        guard()
        frozen = ledger.freeze_for_a3()
        save_new(output / "a3-exploration-freeze.json", frozen)
        progress.update(stage="A2:accounted:A3-frozen", status="A2_ACCOUNTED_NOT_MECHANISM_QUALIFIED")
        save_new(output / "stage-terminal.json", progress)
        if progress_sink is not None:
            progress_sink(progress)
        return ledger, progress
    except BaseException as error:
        bank.close()
        if progress is not None:
            progress.update(status="FAILED_NO_RETRY", error_type=type(error).__name__, retry_authorized=False)
            try:
                save_new(output / "stage-failure.json", progress)
                if progress_sink is not None:
                    progress_sink(progress)
            except BaseException as evidence_error:
                error.add_note("Stage failure evidence also failed: " + type(evidence_error).__name__)
        raise
