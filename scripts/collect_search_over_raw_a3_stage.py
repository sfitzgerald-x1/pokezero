"""One prospective retained-root A3 sweep after a sealed A2 choice.

Callable only by a future reviewed producer, not a CLI or runtime admission.
Reuse raw and four frozen 1s selections and every overlapping continuation,
including capped null outcomes; run only eight new 3s/10s search selectors.
The original producer owns the bank/shell/deadline through subsequent direct
checks. This stage cannot establish mechanism, fidelity or held-out gain.
"""
import json
from math import isfinite
from pathlib import Path
import time

from benchmark_search_over_raw_feasibility import save_new
from qualify_search_over_raw_a1 import collect_root
from pokezero.mcts_eval.search_over_raw import digest, require
from pokezero.mcts_eval.search_over_raw_archive import SealedComparisonBank
from pokezero.mcts_eval.search_over_raw_stages import ExplorationStages, a2_configurations, key


def collect_a3_from_bank(*, plan, bank, ledger, a2_output, output, env, evaluator, factory,
                         checkpoint_contract, source, showdown, verify, deadline_at,
                         ownership=None, progress_sink=None, clock=time.perf_counter):
    require(isinstance(bank, SealedComparisonBank), "actual sealed comparison bank required")
    progress = None
    try:
        require(isinstance(ledger, ExplorationStages), "actual in-process A2 ledger required")
        require(type(deadline_at) in (int, float) and isfinite(deadline_at), "original finite deadline required")
        require(callable(verify), "producer provenance/deadline verifier required")
        output, a2_output = Path(output).resolve(), Path(a2_output).resolve()
        require(not output.is_relative_to(Path(__file__).resolve().parents[1])
            and not output.is_relative_to(Path(plan["historical_directory"]).resolve())
            and not output.is_relative_to(a2_output) and not a2_output.is_relative_to(output),
            "A3 output must be separate from source, historical and A2 evidence")
        def guard():
            require(clock() < deadline_at, "original attempt deadline reached; no new stage clock")
            verify()
        guard()
        bank.validate_roster(plan["comparable_root_contract"]["root_slots"])
        manifest = bank.manifest()
        a2_manifest = json.loads((a2_output / "stage-manifest.json").read_text())
        require(a2_manifest["original_deadline_at"] == deadline_at, "A3 cannot extend A2/producer deadline")
        frozen = json.loads((a2_output / "a3-exploration-freeze.json").read_text())
        ledger.begin_a3(plan, manifest, frozen, deadline_at=deadline_at)
        output.mkdir(parents=True, exist_ok=False)
        roster = ledger.a3_configurations()
        slots = plan["phase_a_cohort"]["panels"]["exploration"]["root_slots"]
        progress = dict(stage="A3:prepared", status="COLLECTING_EXPLORATORY_A3",
            fixed_roster=[dict(root_id=slot["root_id"], configuration=key(cfg),
                status="UNSTARTED_UNCERTAIN", contrast_interval=[-1., 1.], action=None)
                for slot in slots for cfg in roster],
            continuation_roster=[dict(root_id=slot["root_id"], configuration=key(cfg), replicate=rep,
                status="UNSTARTED_UNCERTAIN", action=None, signed_outcome=None)
                for slot in slots for cfg in roster for rep in range(8)],
            roots_completed=0, selections_completed=0, selections_reused=0,
            continuations_completed=0, continuations_capped=0,
            continuations_reused=0, continuations_reused_capped=0,
            scientific_strength_evidence=False, holdout_authorized=False, runtime_authorized=False)
        save_new(output / "stage-manifest.json", dict(bank=manifest, frozen_choice_sha256=digest(frozen),
            original_deadline_at=deadline_at, full_root_denominator=200, full_seed_denominator=32,
            full_selector_denominator=len(slots)*len(roster),
            full_continuation_alias_denominator=len(slots)*len(roster)*8,
            per_measured_root_new_selectors=8, per_measured_root_reused_selectors=5))
        for slot in slots:
            root_id = slot["root_id"]
            if root_id not in manifest["root_public_bindings"]:
                continue
            guard()
            relative = Path(f"source-{slot['source_seed']}") / f"root-{slot['root_slot']}"
            prior_directory = a2_output / relative
            prior = json.loads((prior_directory / "audit.json").read_text())
            prior_selections = {key(cfg): json.loads((prior_directory / f"{key(cfg)}-selected.json").read_text())
                for cfg in a2_configurations()}
            reuse = ledger.a3_reuse(root_id, prior, prior_selections)
            root, context, pending, snapshot = bank.selected_for_auditor(root_id)
            directory = output / relative
            directory.mkdir(parents=True, exist_ok=False)
            collect_root(root=root, context=context, pending=pending, snapshot=snapshot,
                output=directory, progress=progress, env=env, evaluator=evaluator, factory=factory,
                contract=checkpoint_contract, source=source, showdown=showdown, verify=guard,
                ownership=ownership, namespace=plan["execution_source_contract"]["namespace"],
                completion_status="COLLECTED_A3_CENSORING_AWARE", boundary_censoring=True,
                configuration_roster=roster, matched_statistics_root=True, reused_root_evidence=reuse)
            audit = json.loads((directory / "audit.json").read_text())
            selections = {key(cfg): json.loads((directory / f"{key(cfg)}-selected.json").read_text()) for cfg in roster}
            ledger.record_a3_root(audit, selections)
            if progress_sink is not None:
                progress_sink(progress)
        guard()
        readout = ledger.a3_readout()
        save_new(output / "exploratory-readout.json", readout)
        progress.update(stage="A3:accounted", status="A3_ACCOUNTED_DIRECT_CHECKS_PENDING")
        save_new(output / "stage-terminal.json", progress)
        if progress_sink is not None:
            progress_sink(progress)
        return readout, progress
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
