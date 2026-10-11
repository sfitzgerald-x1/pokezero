"""Pure planning for the closed A1 panel's collection-only amendment.

No experiment, registration, admission, replay or retry entry point exists here.
The original failed attempt stays immutable. The outcome-informed exploratory
amendment may collect only original never-started sources after separate review
and authority; it cannot supply useful holdout inference or a new replication.
"""
from copy import deepcopy
from dataclasses import asdict
import argparse
import json
import os
from pathlib import Path

from .resolver import sha256_file
from .search_over_raw import SearchConfiguration, digest, require


SCHEMA = "pokezero.search-over-raw.a1-collection-amendment-plan.v1"
CENSORING = "CAPPED_UNKNOWN_CONTINUE_FIXED_ROSTER"
PARTIAL_REVIEW = "FAILED_NO_RETRY_WITH_NINE_ROOT_DESCRIPTIVE_PARTIAL_EVIDENCE"
COLLECTED = "COLLECTED_A1_CENSORING_AWARE"


def _identities(registration):
    expected = [SearchConfiguration("raw", seconds=1.)] + [
        SearchConfiguration(arm, belief, "model", 1., 20 if arm == "reference" else 1)
        for arm in ("incumbent", "reference") for belief in ("public", "oracle")]
    require(registration["configurations"] == [asdict(cfg) for cfg in expected],
        "original A1 scientific settings required")
    return ["raw" if c.arm == "raw" else c.identity
        for c in (SearchConfiguration(**row) for row in registration["configurations"])]


def plan_collection_amendment(registration, terminal, review):
    """Derive—not choose—the only remaining roster from reviewed evidence.

    This is intentionally specific to r1's independently audited nine-root
    failure. Another terminal shape requires its own review and implementation;
    this planner must not silently generalize a review to a different failure.
    """
    require(registration["schema"] == "pokezero.search-over-raw.a1-exploration.v1"
        and review["disposition"] == PARTIAL_REVIEW
        and review["registration_sha256"] == digest(registration)
        and review["source_commit"] == registration["source_commit"]
        and review.get("independent_reviewer")
        and all(review[field] is False for field in
            ("scientific_admission", "holdout_authorized", "retry_authorized", "continuation_authorized")),
        "exact independent closed-attempt review required")
    require(terminal["status"] == "FAILED_NO_RETRY" and terminal["exit_code"] == 1
        and terminal["registration_sha256"] == digest(registration)
        and all(terminal[field] is False for field in
            ("retry_authorized", "holdout_opened", "scientific_strength_evidence", "phase_a_admission", "phase_b_authorized")),
        "failed attempt remains closed and unadmitted")
    c = registration["phase_a_cohort"]
    require(digest(c) == registration["phase_a_cohort_sha256"]
        and c["execution_ready"] is False and c["validation_inference_not_yet_admitted"] is True,
        "original unopened cohort binding required")
    panel = c["panels"]["exploration"]
    slots, seeds = panel["root_slots"], panel["seeds"]
    validation = c["panels"]["validation"]
    require(len(slots) == len(validation["root_slots"]) == 200
        and len(seeds) == len(validation["seeds"]) == 32
        and len(set(seeds + validation["seeds"])) == 64
        and registration["seeds"] == seeds,
        "full 400-slot disjoint two-panel design required")
    identities = _identities(registration)
    require(len(identities) == len(set(identities)) == 5 and identities[0] == "raw",
        "exact five-selector A1 roster required")
    expected = {(row["root_id"], cfg) for row in slots for cfg in identities}
    cells = {(row["root_id"], row["configuration"]): row for row in terminal["fixed_roster"]}
    aliases = {(row["root_id"], row["configuration"], row["replicate"]): row
        for row in terminal["continuation_roster"]}
    require(len(terminal["fixed_roster"]) == len(cells) == 1000 and set(cells) == expected
        and len(terminal["continuation_roster"]) == len(aliases) == 8000
        and set(aliases) == {(root, cfg, rep) for root, cfg in expected for rep in range(8)},
        "full fixed cells/aliases required; no duplicate or replacement")
    sources = terminal["source_roster"]
    require(set(sources) == {str(seed) for seed in seeds}
        and all(row["status"] in {"COMPLETE", "UNSTARTED_UNCERTAIN"} for row in sources.values()),
        "exact reviewed source state required")
    remaining_seeds = [seed for seed in seeds if sources[str(seed)]["status"] == "UNSTARTED_UNCERTAIN"]
    retained, unavailable, remaining = [], [], []
    for slot in slots:
        root, seed = slot["root_id"], slot["source_seed"]
        rows = [cells[root, cfg] for cfg in identities]
        continuation_rows = [aliases[root, cfg, rep] for cfg in identities for rep in range(8)]
        if seed in remaining_seeds:
            require(all(row["status"] == "UNSTARTED_UNCERTAIN" and row["action"] is None
                and row["contrast_interval"] == [-1., 1.] for row in rows)
                and all(row["status"] == "UNSTARTED_UNCERTAIN" and row["action"] is None
                    and row["signed_outcome"] is None for row in continuation_rows),
                "remaining source was already exposed/attempted")
            remaining.append(deepcopy(slot))
        elif all(row["status"] == "COMPLETE_EXPLORATION" for row in rows):
            require(all(row["status"] == "COMPLETE" for row in continuation_rows),
                "retained root lacks complete reviewed aliases")
            retained.append(root)
        else:
            require(all(row["status"] in {"UNSTARTED_UNCERTAIN", "SELECTED_UNMEASURED"}
                and row["contrast_interval"] == [-1., 1.] for row in rows),
                "unavailable root was scored or partly admitted")
            unavailable.append(root)
    fixed = review["fixed_roster"]
    require((len(retained), len(unavailable), len(remaining_seeds), len(remaining)) == (9, 5, 30, 186)
        and terminal["roots_completed"] == fixed["completed_roots"] == 9
        and terminal["source_games_completed"] == 2
        and fixed["cells"] == {"COMPLETE_EXPLORATION": 45,
            "SELECTED_UNMEASURED": 5, "UNSTARTED_UNCERTAIN": 950},
        "independently reviewed nine/five/186 partition required")
    execution = deepcopy(registration["source_contract"])
    require(execution["panels"] == {"exploration": panel}
        and execution["max_source_boundaries"] == execution["max_continuation_boundaries"] == 250
        and execution["retry_authorized"] is False,
        "original source roster/caps required")
    execution["panels"] = {"exploration": dict(seeds=remaining_seeds, root_slots=remaining)}
    execution.update(boundary_censoring_policy=CENSORING,
        stop_policy="CAPPED continuations remain null and collection continues over the fixed roster; source, worker, selection, ownership, provenance, materialization or cleanup failures still stop the attempt")
    return dict(schema=SCHEMA, original_registration_sha256=digest(registration),
        original_terminal_sha256=digest(terminal), independent_partial_review_sha256=digest(review),
        phase_a_cohort_sha256=digest(c), phase_a_cohort=deepcopy(c),
        retained_root_ids=retained, unavailable_root_ids=unavailable,
        execution_source_contract=execution, full_accounting_source_contract=deepcopy(registration["source_contract"]),
        full_root_denominator=200, full_seed_denominator=32,
        full_selector_denominator=1000, full_continuation_alias_denominator=8000,
        retained_cells=[deepcopy(row) for row in terminal["fixed_roster"] if row["root_id"] in retained],
        retained_aliases=[deepcopy(row) for row in terminal["continuation_roster"] if row["root_id"] in retained],
        external_runtime_cap_seconds=execution["attempt_wall_seconds"],
        cleanup_grace_seconds=execution["cleanup_grace_seconds"],
        requires_new_registration=True, requires_new_independent_review=True,
        requires_separate_runtime_authority=True, runtime_authorized=False,
        retry_authorized=False, source_replay_authorized=False, replacements_permitted=False,
        holdout_authorized=False, phase_a_admission=False, phase_b_authorized=False,
        scientific_strength_evidence=False,
        analytical_extension="Outcome-informed exploration-only collection amendment, not an independent replication. Nine reviewed roots retained; five unavailable roots remain [-1,1]. No failed-root identical-action algebraic extension is admitted. Censored outcomes are null, with paired sharp bounds and shared-action aliases, never loss/draw imputation.",
        comparable_root_contract=dict(
            storage="trusted same-process SealedComparisonBank; full auditor payload only, never serialized or passed to public adapters",
            root_slots=deepcopy(remaining), capacity=186, parent_memory_limit_bytes=8 * 1024**3,
            memory_rationale="8GiB parent RSS high-water ceiling leaves28GiB of this36GiB host for existing workers/OS; representative retained-copy footprint remains a prelaunch qualification, not a measured claim",
            auditor_shell="one owned same-format shell after original source environments close; reject bridge-local snapshot_id handles",
            lifecycle="revalidate each completed source's fixed-priority selected/missing partition; capture selected roots before each source archive closes; seal exact186 accounted slots (captured plus source-validated missing); retain through A2/A3; clear on completion, failure or unchanged14400s deadline",
            source_replay_authorized=False, private_persistence_authorized=False,
            full_a2_a3_root_denominator=200, full_a2_a3_seed_denominator=32,
            old_source_a2_a3_uncertain_root_ids=retained + unavailable,
            a2="public/oracle for each arm; model/hp_fraction/raw_rollout at1s; same sampler streams, algorithm and allocations per arm/belief; actual-world receipts and visited-state value labels required",
            a3="freeze one leaf per arm/belief from full-denominator A2 before1/3/10s sweep on the same retained roots; public candidates deployable, oracle diagnostic only; searched-line fidelity required",
            selection_rule="exploration-only equal-source-seed mean of pointwise lower contrast bounds; maximize separately per arm/belief, ties model then hp_fraction then raw_rollout; freeze before downstream stage",
            skipped_configurations="remain unmeasured; no outcomes or claims invented for unused full-cohort configurations",
            launch_barriers=["cross-closed-environment restore and repeated-restore immutability",
                "representative bank memory qualification and source/input ownership checks",
                "staged A2/A3 producer and direct-check collectors implemented and tested",
                "executable registration binds this full staged contract and own clean source",
                "independent prospective review and separate runtime authority"]),
        a2_a3_comparable_roots_pending="Comparable-root storage/contract specified; staged producer, representative memory qualification, direct checks and runtime admission remain unfinished. No A1-only launch or old source replay.")


def load_reviewed_collection(directory, *, expected_review_sha256):
    """Read and rehash only this attempt's finite independent evidence manifest.

    Current source identity is deliberately NOT inferred from this historical
    receipt: a new executable registration must bind its own clean source and
    scientific inputs. A changed historical artifact fails this read-only gate.
    """
    directory = Path(directory).resolve()
    read = lambda path: json.loads(path.read_text())
    review_path = directory / "terminal-independent-review.json"
    require(type(expected_review_sha256) is str and len(expected_review_sha256) == 64
        and sha256_file(review_path) == expected_review_sha256, "independent review file binding drift")
    review = read(review_path)
    require({"registration.json", "terminal.json"} <= set(review["evidence_file_sha256"]),
        "review lacks core evidence bindings")
    for name, expected in review["evidence_file_sha256"].items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory), "review evidence escapes attempt")
        require(sha256_file(path) == expected, "historical evidence drift")
    registration, terminal = read(directory / "registration.json"), read(directory / "terminal.json")
    result = plan_collection_amendment(registration, terminal, review)
    selected_roots = {row["root_id"] for row in terminal["fixed_roster"]
        if row["status"] != "UNSTARTED_UNCERTAIN"}
    expected_paths = {str(directory / f"source-{slot['source_seed']}" / f"root-{slot['root_slot']}"):
        slot["root_id"] in result["retained_root_ids"]
        for slot in registration["source_contract"]["panels"]["exploration"]["root_slots"]
        if slot["root_id"] in selected_roots}
    manifests = review["root_evidence_manifests"]
    require(len(manifests) == len(expected_paths) == 10
        and {str(Path(m["path"]).resolve()) for m in manifests} == set(expected_paths),
        "review lacks exact completed/failed root coverage")
    for manifest in manifests:
        root = Path(manifest["path"]).resolve()
        require(root.is_relative_to(directory)
            and manifest["file_count"] == len(manifest["file_sha256"])
            and manifest["manifest_sha256"] == digest(manifest["file_sha256"])
            and manifest["fully_completed"] is expected_paths[str(root)], "root manifest scope/count/digest drift")
        require({path.name for path in root.iterdir() if path.is_file()} == set(manifest["file_sha256"]),
            "root evidence inventory drift")
        for name, expected in manifest["file_sha256"].items():
            path = (root / name).resolve()
            require(path.is_relative_to(root), "root evidence escapes manifest")
            require(sha256_file(path) == expected, "historical root evidence drift")
    result["historical_directory"] = str(directory)
    result["independent_partial_review_file_sha256"] = sha256_file(directory / "terminal-independent-review.json")
    return result


def main(argv=None):
    """Write a new proposal only. There is deliberately no run/register mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    original, output = args.original.resolve(), args.output.resolve()
    source_root = Path(__file__).resolve().parents[3]
    require(not output.is_relative_to(original) and not output.is_relative_to(source_root),
        "proposal must not change old evidence or executing source")
    result = load_reviewed_collection(original, expected_review_sha256=args.review_sha256)
    result["proposal_implementation_sha256"] = sha256_file(Path(__file__).resolve())
    result["artifact_kind"] = "PROPOSAL_ONLY_NOT_EXECUTABLE_REGISTRATION"
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(proposal=str(output), proposal_sha256=sha256_file(output),
        retained_roots=9, unavailable_roots=5, remaining_roots=186, runtime_authorized=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
