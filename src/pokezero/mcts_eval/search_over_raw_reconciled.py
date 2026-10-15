"""Versioned staged Phase A controller for already reviewed fresh manifests.

No source allocation, source games, adapters or scientific launch entrypoint.
The caller still owns receipt validation, worker supervision and runtime
qualification. Hashes detect drift, not a malicious operator forging clearance.
Legacy v2 contracts and collection-amendment banks are never upgraded here.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from .search_over_raw import SearchConfiguration, digest, require
from .search_over_raw_screening import (
    ARMS, DEPLOYABLE_LEAVES, ScreeningDesign, StageSummary, cluster_intervals,
    choose_evaluators, choose_ceilings, screen,
)

SCHEMA = "pokezero.search-over-raw.phase-a.v3"
HOLDOUT_ONLY_GATE = "frozen staged selections and separate holdout clearance"
CONFIGURATION_REVIEW_SCHEMA = "pokezero.search-over-raw.configuration-recipe-review.v2"


def deployment_configurations():
    return tuple(SearchConfiguration(arm, "public", leaf, seconds,
        20 if arm == "reference" else 1)
        for arm in ARMS for leaf in DEPLOYABLE_LEAVES for seconds in (1., 3., 10.))


def a1_configurations():
    return tuple(SearchConfiguration(arm, belief, "model", 1.,
        20 if arm == "reference" else 1) for arm in ARMS for belief in ("public", "oracle"))


def a2_configurations():
    return tuple(cfg for cfg in deployment_configurations() if cfg.seconds == 1.)


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def panel_roster(panel):
    require(type(panel) is dict and set(panel) == {"seeds", "root_slots"}, "manifest contains only source identities and quotas")
    seeds, slots = panel["seeds"], panel["root_slots"]
    require(type(seeds) is list and 32 <= len(seeds) <= 200
        and all(type(s) is int and 0 <= s < 2**32 for s in seeds)
        and len(set(seeds)) == len(seeds), "fixed >=32 unique uint32 source seeds required")
    require(type(slots) is list and len(slots) == 200
        and all(type(r) is dict and set(r) == {"root_id", "source_seed", "root_slot"}
            and type(r["root_id"]) is str and r["root_id"]
            and type(r["source_seed"]) is int and r["source_seed"] in seeds
            and type(r["root_slot"]) is int and r["root_slot"] >= 0 for r in slots)
        and len({r["root_id"] for r in slots}) == 200,
        "fixed 200 unique root slots required; no continuation amendment bank")
    roster = tuple((s, tuple(r["root_id"] for r in slots if r["source_seed"] == s)) for s in seeds)
    require(all(quota for _, quota in roster), "every seed must retain a root quota")
    for s in seeds:
        require(sorted(r["root_slot"] for r in slots if r["source_seed"] == s)
            == list(range(len(next(q for seed, q in roster if seed == s)))),
            "unique contiguous root ordinals including diagnostic slot zero required")
    return roster


def reconciled_contract(namespace, *, panels, excluded_seeds, source_review, input_hashes=None):
    """Bind supplied manifests only. Does NOT draw or register scientific seeds."""
    from .search_over_raw_fixed_work import FixedWorkDiagnosticContract
    require(set(panels) == {"exploration", "validation"}, "both fixed source-disjoint panels required")
    rosters = {name: panel_roster(panel) for name, panel in panels.items()}
    for roster in rosters.values():
        ScreeningDesign(namespace, tuple(s for s, _ in roster))
    require(len(rosters["exploration"]) == len(rosters["validation"]), "equal panel seed counts required")
    exp = {s for s, _ in rosters["exploration"]}
    val = {s for s, _ in rosters["validation"]}
    require(type(excluded_seeds) is list and excluded_seeds
        and all(type(s) is int and 0 <= s < 2**32 for s in excluded_seeds)
        and excluded_seeds == sorted(set(excluded_seeds)), "bound canonical exposure inventory required")
    require(not exp & val and not (exp | val) & set(excluded_seeds), "exposed or overlapping source seeds")
    roots = [root for roster in rosters.values() for _, quota in roster for root in quota]
    require(len(set(roots)) == 400, "cross-panel root identity overlap")
    require(source_review.get("status") == "CLEAR_FRESH_SOURCE_DESIGN"
        and source_review.get("panels_sha256") == digest(panels)
        and source_review.get("excluded_seeds_sha256") == digest(excluded_seeds)
        and source_review.get("all_known_exposures_included") is True
        and source_review.get("nine_roots_and_two_sources_excluded") is True,
        "prospective independent freshness review required")
    bindings = {} if input_hashes is None else dict(input_hashes)
    require(all(type(p) is str and Path(p).is_absolute() and _sha(h) for p, h in bindings.items()),
        "absolute-path input hashes required")
    return dict(schema=SCHEMA, namespace=namespace, panels=deepcopy(panels),
        excluded_seeds=deepcopy(excluded_seeds), source_review=deepcopy(source_review),
        input_hashes=bindings, continuation_replicates=8,
        source_policy="raw_argmax_both_seats", candidate_seat="p1",
        continuation_policy="raw_argmax_after_initial_sampled_opponent_reply",
        paired_randomness="root/replicate/step/domain; independent of selected action",
        source_root_rule="seed-derived priorities over completed public request catalog; no replacements",
        configurations=[dict(asdict(c), identity=c.identity) for c in deployment_configurations()],
        a1_configurations=[dict(asdict(c), identity=c.identity) for c in a1_configurations()],
        protocol="independent A1/A2; A2 evaluator freeze then A3 ceiling freeze",
        fixed_work_diagnostic=dict(FixedWorkDiagnosticContract().receipt(),
            root_subset="root_slot zero of every exploration seed"),
        validation_inference=dict(method="paired_seed_cluster_percentile_bootstrap",
            nominal_alpha_per_arm=.05, bonferroni=False, bootstrap_draws=10000,
            lower_order_index=500, shared_arm_resample_indices=True,
            looks="one sealed full panel; no interim admission",
            family_error_control_claim=False, confirmatory_strength=False),
        execution_ready=False, remaining_gates=["source and per-configuration runtime qualification",
            "separate collection authority", "frozen staged selections and separate holdout clearance"],
        historical_outcomes_pooled=False, collection_authorized=False, phase_b_authorized=False)


def validate_contract(contract):
    require(contract.get("schema") == SCHEMA, "reconciled v3 contract required; no legacy upgrade")
    expected = reconciled_contract(contract["namespace"], panels=contract["panels"],
        excluded_seeds=contract["excluded_seeds"], source_review=contract["source_review"],
        input_hashes=contract["input_hashes"])
    mutable_admission = {"execution_ready", "remaining_gates", "collection_authorized"}
    require(set(contract) == set(expected), "unexpected reconciled core fields")
    # Equality alone aliases bool/int/float values. These fields describe the
    # prospectively frozen procedure, including flags, counts and configuration
    # identities; validate their canonical JSON representations, not coercions.
    require(all(digest(contract[k]) == digest(v) for k, v in expected.items() if k not in mutable_admission),
        "reconciled core drift; no changed inference, source law or roster")
    require(type(contract["execution_ready"]) is bool and type(contract["collection_authorized"]) is bool
        and type(contract["remaining_gates"]) is list
        and all(type(x) is str and x for x in contract["remaining_gates"]), "explicit admission state required")


def controller_input_paths():
    """Fresh required controller paths; not proof of current file bytes."""
    directory = Path(__file__).resolve().parent
    names = ("search_over_raw.py", "search_over_raw_ledger.py",
                     "search_over_raw_archive.py", "search_over_raw_source.py", "paper_reference_runtime.py",
                     "search_over_raw_reconciled.py", "search_over_raw_screening.py",
                     "search_over_raw_adapters.py", "search_over_raw_fixed_work.py",
                     "search_over_raw_fixed_work_run.py",
                     "search_over_raw_diagnostic_producer.py",
                     "search_over_raw_reference_values.py", "search_over_raw_value_diagnostics.py",
                     "search_over_raw_native_chain.py",
                     "search_over_raw_exploration_workflow.py",
                     "search_over_raw_public_fold.py",
                     "search_over_raw_source_public_history.py",
                     "search_over_raw_protocol_channels.py",
                     "search_over_raw_actor_replay_inputs.py",
                     "search_over_raw_actor_projection.py",
                     "search_over_raw_actor_replay_projection.py",
                     "search_over_raw_native_leaf_cells.py",
                     "search_over_raw_after_raw.py",
                     "search_over_raw_after_owner.py",
                     "search_over_raw_after_bank.py",
                     "search_over_raw_after_admission.py",
                     "search_over_raw_after_population.py",
                     "search_over_raw_source_actor_history.py",
                     "search_over_raw_timed_run.py",
                     "search_over_raw_oracle.py", "search_over_raw_native_oracle.py",
                     "search_over_raw_native_oracle_run.py", "paper_reference_factory.py",
                     "search_over_raw_timed_reference.py",
                     "search_over_raw_reference_timed_run.py",
                     "search_over_raw_reference_oracle_run.py",
                     "search_over_raw_raw_run.py",
                     "search_over_raw_root_plan.py",
                     "search_over_raw_continuation_admission.py",
                     "search_over_raw_continuation_producer.py",
                     "search_over_raw_selector_producer.py",
                     "search_over_raw_consumed_materialization.py",
                     "search_over_raw_ancestry.py",
                     "search_over_raw_native_fixed_values.py",
                     "search_over_raw_native_fixed_value_run.py",
                     "search_over_raw_annotation_boundary.py",
                     "search_over_raw_actor_timed_run.py",
                     "search_over_raw_supervision.py", "search_over_raw_evidence.py",
                     "search_over_raw_nested_ownership.py",
                     "search_over_raw_source_run.py",
                     "search_over_raw_stage_run.py",
                     "search_over_raw_exploration_entry.py",
                     "search_over_raw_resources.py",
                     "search_over_raw_attempt_cost.py",
                     "search_over_raw_attempt_journal.py",
                     "search_over_raw_observer_cost.py",
                     "search_over_raw_world_inputs.py", "search_over_raw_correspondence.py",
                     "search_over_raw_atomic_work.py", "search_over_raw_native_atomic_work.py",
                     "search_over_raw_outcomes.py",
                     "search_over_raw_outcome_run.py",
                     "search_over_raw_native_boundary.py",
                     "search_over_raw_native_boundary_records.py",
                     "search_over_raw_native_edge.py",
                     "search_over_raw_showdown_boundary.py",
                     "search_over_raw_boundary_transport.py",
                     "search_over_raw_boundary_run.py",
                     "search_over_raw_native_prefix.py",
                     "search_over_raw_construction.py",
                     "search_over_raw_materialization.py", "search_over_raw_materialization_run.py",
                     "search_over_raw_paired_materialization.py",
                     "search_over_raw_paired_replay.py",
                     "search_over_raw_paired_labels.py",
                     "search_over_raw_paired_chain.py",
                     "paper_reference.py", "paper_reference_parallel.py")
    native = directory.parents[2] / "rust" / "pokezero-search" / "src"
    paths = [directory / name for name in names] + [native / name for name in
        ("atomic_work.rs", "model.rs", "tree.rs", "lib.rs", "scoped_boundary.rs", "native_edge.rs", "visited_value.rs", "searched_ancestry.rs")]
    paths += [directory.parent / name for name in ("engine_world.py","engine_search.py","poke_engine_adapter.py","showdown_fixture.py")]
    paths += [directory.parent / "local_showdown.py", directory.parents[2] / "scripts" / "battle_bridge.mjs",
        directory.parent / "public_decision_corpus.py",
        directory.parents[2] / "scripts" / "battle_bridge_owned_continuation.mjs",
        directory.parents[2] / "scripts" / "battle_bridge_owned_snapshot.mjs"]
    paths += [directory.parents[2] / "scripts" / "collect_search_over_raw_reconciled_exploration.py"]
    return tuple(paths)


def controller_input_hashes():
    """Required code provenance; caller additionally binds models/runtime files."""
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in controller_input_paths()}


def configuration_review_template(contract, *, stage, configuration,
                                  raw_runtime_sha256, selector_recipe_sha256,
                                  qualification_evidence_hashes, cost_scope):
    """Reviewable before source work: no future public or original-team hashes.

    Clearance is external. Code provenance and qualification evidence are
    distinct; even a cleared recipe grants no collection/holdout authority.
    A3 candidate recipes may be reviewed before A2 chooses its evaluator.
    """
    validate_contract(contract)
    configs = a1_configurations() if stage == "A1" else (
        a2_configurations() if stage == "A2" else deployment_configurations() if stage == "A3" else ())
    require(configuration in {c.identity for c in configs}, "exact stage/configuration recipe required")
    require(_sha(raw_runtime_sha256) and _sha(selector_recipe_sha256), "canonical runtime recipe hashes required")
    require(type(qualification_evidence_hashes) is dict and qualification_evidence_hashes
        and all(type(p) is str and str(Path(p).resolve()) == p and _sha(h)
            for p,h in qualification_evidence_hashes.items()), "absolute qualification evidence file hashes required")
    require(type(cost_scope) is str and cost_scope.strip(), "measured cost scope and limitations required")
    code_hashes = controller_input_hashes()
    require(not set(qualification_evidence_hashes) & set(code_hashes),
        "qualification evidence must be separate from controller source provenance")
    recipe = {k:deepcopy(v) for k,v in contract.items() if k not in
        {"input_hashes","execution_ready","collection_authorized","remaining_gates"}}
    return dict(schema=CONFIGURATION_REVIEW_SCHEMA, status="UNKNOWN_CONFIGURATION_RECIPE",
        stage=stage, configuration_sha256=configuration, contract_recipe_sha256=digest(recipe),
        raw_runtime_sha256=raw_runtime_sha256, selector_recipe_sha256=selector_recipe_sha256,
        qualification_evidence_hashes=deepcopy(qualification_evidence_hashes), cost_scope=cost_scope,
        controller_input_hashes=code_hashes, reviewed_before_first_source_work=False,
        atomic_work_and_cleanup_validated=False, resource_cost_qualification_reviewed=False,
        zero_work_is_local_unqualified=True, collection_authorized=False, holdout_authorized=False,
        phase_b_authorized=False, runtime_qualified_by_this_code=False)


def selector_runtime_recipe(configuration, runtime):
    """Only root-specific original-team truth may vary in A1 oracle recipes."""
    require(type(runtime) is dict and digest(runtime.get("configuration")) == digest(asdict(configuration)),
        "runtime configuration differs from reviewed recipe")
    recipe = deepcopy(runtime)
    if configuration.belief == "oracle":
        recipe.pop("team_oracle",None)
        if configuration.arm == "reference": recipe.pop("oracle_payload_sha256",None)
    return recipe


class ReconciledPhaseALedger:
    """Create-only stage evidence and irreversible freezes on the v3 path."""
    def __init__(self, directory):
        # Reuse durable create-only IO, not legacy inference or selection.
        from .search_over_raw_ledger import _load
        self.directory = Path(directory).resolve()
        envelope = _load(self.directory / "contract.json")
        self.contract = envelope["contract"]
        self.contract_sha256 = envelope["contract_sha256"]
        require(digest(self.contract) == self.contract_sha256, "ledger contract digest mismatch")
        validate_contract(self.contract)

    @classmethod
    def create(cls, directory, contract):
        from .search_over_raw_ledger import _snapshot, _write_new
        contract = _snapshot(contract)
        validate_contract(contract)
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        for stage in ("A1", "A2", "A3"):
            (directory / stage).mkdir()
        _write_new(directory / "contract.json", dict(contract=contract, contract_sha256=digest(contract)))
        return cls(directory)

    def verify(self, *, execution=False):
        from .search_over_raw_ledger import _load
        require(digest(self.contract) == self.contract_sha256, "in-memory contract drift")
        require(_load(self.directory / "contract.json") == dict(contract=self.contract,
            contract_sha256=self.contract_sha256), "durable contract drift")
        validate_contract(self.contract)
        for name, expected in self.contract["input_hashes"].items():
            require(hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, "bound input drift: " + name)
        require(not (self.directory / "operational-stop.json").exists(), "operational integrity failure stops this ledger")
        if execution:
            require(self.contract["execution_ready"] and self.contract["collection_authorized"]
                and self.contract["remaining_gates"] == [], "scientific execution admission is incomplete")
            require(all(self.contract["input_hashes"].get(p) == h for p, h in controller_input_hashes().items()),
                "current controller code must be bound before scientific execution")

    def stop(self, error_type):
        from .search_over_raw_ledger import _write_new
        require(type(error_type) is str and error_type.isidentifier(), "sanitized failure class required")
        _write_new(self.directory / "operational-stop.json", dict(contract_sha256=self.contract_sha256,
            error_type=error_type, retry_authorized=False, phase_b_authorized=False))

    def verify_exploration(self, stage, configuration):
        """Stage-local admission; downstream freeze/holdout cannot gate A1/A2.

        This does not qualify a source/runtime, grant collection, or waive an
        unknown gate. The prospective contract must separately clear those.
        The strict execution check remains unchanged for holdout admission.
        """
        self.verify()
        require(stage in {"A1", "A2", "A3"}
            and any(c.identity == configuration for c in self._stage_configs(stage)),
            "stage-local registered configuration required")
        require(self.contract["execution_ready"] is True and self.contract["collection_authorized"] is True
            and set(self.contract["remaining_gates"]) <= {HOLDOUT_ONLY_GATE},
            "exploration source/runtime/collection admission is incomplete")
        require(all(self.contract["input_hashes"].get(p) == h for p, h in controller_input_hashes().items()),
            "current controller code must be bound before exploration execution")
        require(not (self.directory / "selection.json").exists(), "exploration is frozen")
        require(stage != "A2" or not (self.directory / "a2-freeze.json").exists(), "A2 is frozen")
        return dict(stage=stage, configuration=configuration, exploration_admission_validated=True,
            holdout_authorized=False, phase_b_authorized=False, retry_authorized=False)

    def _stage_configs(self, stage):
        require(stage in {"A1", "A2", "A3"}, "unknown stage")
        if stage == "A1":
            return a1_configurations()
        if stage == "A2":
            return a2_configurations()
        frozen = self._a2()
        return tuple(SearchConfiguration(arm, "public", leaf, seconds,
            20 if arm == "reference" else 1) for arm, leaf in frozen["evaluators"] if leaf is not None
            for seconds in (1., 3., 10.))

    def record_stage(self, stage, configuration, root_intervals, *, accepted_units, status, evidence_sha256,
                     _from_reconciled_evidence=False):
        from .search_over_raw_ledger import _snapshot, _write_new
        self.verify()
        require(not (self.directory / "selection.json").exists(), "exploration is frozen")
        require(stage != "A2" or not (self.directory / "a2-freeze.json").exists(), "A2 is frozen")
        cfg = next((c for c in self._stage_configs(stage) if c.identity == configuration), None)
        require(cfg is not None and _sha(evidence_sha256), "bound stage configuration/evidence required")
        if self.contract["execution_ready"] and status == "QUALIFIED":
            self.verify_exploration(stage, configuration)
            require(_from_reconciled_evidence or (stage == "A3" and cfg.seconds == 1.),
                "execution-ready qualification requires reconciled receipts, not supplied intervals")
        claim = self.directory / stage / f"{configuration}-evidence-claim.json"
        require(not claim.exists() or _from_reconciled_evidence,
            "claimed receipt reconciliation cannot be bypassed with manual intervals")
        if _from_reconciled_evidence:
            from .search_over_raw_ledger import _load
            require(claim.exists() and digest(_load(self.directory / stage /
                f"{configuration}-root-evidence.json")) == evidence_sha256,
                "reconciled evidence must exist before recording its summary")
        roots = _snapshot(root_intervals)
        roster = panel_roster(self.contract["panels"]["exploration"])
        intervals = cluster_intervals(roster, roots)
        summary = StageSummary("A2" if stage == "A1" else stage, cfg.arm, cfg.belief, cfg.leaf,
            cfg.seconds, math.fsum(x[0] for x in intervals)/len(intervals), accepted_units, status,
            "wall_clock_diagnostic" if stage == "A1" else "deployment")
        require(status != "UNSTARTED" or not roots, "unstarted stage cannot adopt outcomes")
        require(status != "UNQUALIFIED_AT_CEILING" or (accepted_units == 0 and not roots),
            "zero-work configuration cannot adopt a scored action")
        if stage == "A3" and cfg.seconds == 1.:
            from .search_over_raw_ledger import _load
            prior = _load(self.directory / "A2" / f"{configuration}.json")
            require(roots == prior["root_intervals"] and evidence_sha256 == prior["evidence_sha256"]
                and accepted_units == prior["summary"]["accepted_units"]
                and status == prior["summary"]["status"], "A3 one-second evidence must reuse frozen A2 exactly")
        if status == "OPERATIONAL_STOP":
            self.stop("StageOperationalStop")
            raise ValueError("operational integrity failure stops selection")
        artifact = dict(contract_sha256=self.contract_sha256, stage=stage, configuration=configuration,
            root_intervals=roots, summary=asdict(summary), evidence_sha256=evidence_sha256)
        _write_new(self.directory / stage / f"{configuration}.json", artifact)
        return artifact

    def _shared_action_evidence(self, row, *, create, attempted_actions=None):
        """One immutable receipt per root/action/replicate across all consumers."""
        from .search_over_raw_ledger import _load, _write_new
        directory = self.directory / "shared-actions" / digest(row["root_id"])
        binding = dict(root_id=row["root_id"], source_seed=row["source_seed"],
            public_record_sha256=row["public_record_sha256"],
            continuation_contract_sha256=row["continuation_contract_sha256"],
            raw_selected_sha256=row["selected_sha256"]["raw"], raw_runtime_sha256=row["runtime_sha256"]["raw"])
        records = {"binding": binding}
        actions = set(row["actions"].values())
        if attempted_actions is not None:
            require(type(attempted_actions) is list
                and all(type(action) is int for action in attempted_actions)
                and attempted_actions == sorted(set(attempted_actions))
                and set(attempted_actions) <= actions,
                "exact frozen owned attempted-action scope required")
            # Any actual supplied receipt is bound even if scope is malformed;
            # missing captures in an attempted action still bind all eight.
            supplied = {f"{action}:{rep}" for action in attempted_actions for rep in range(8)}
            require(set(row["outcome_sha256"]) <= supplied
                and set(row["validated_outcome_sha256"]) <= supplied,
                "supplied continuation outside owned attempted scope")
            actions = set(attempted_actions)
        for action in actions:
            for rep in range(8):
                name = f"{action}:{rep}"
                records[f"cell-{action}-{rep}"] = dict(receipt_sha256=row["outcome_sha256"].get(name),
                    validated_sha256=row["validated_outcome_sha256"].get(name))
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        for name, receipt in records.items():
            path = directory / (name+".json")
            if path.exists():
                require(_load(path) == receipt, "shared action/control evidence changed between consumers")
            else:
                require(create, "shared action/control evidence missing")
                _write_new(path, receipt)

    def configuration_recipe_review(self, stage, configuration, path):
        """Check prospective qualification before production, not authority.

        Actual archive/owner joins and a separate execution admission are still
        required. The returned bytes let callers guard the original review
        across callback-bearing preparation, launch and final validation.
        """
        self.verify()
        require(configuration in {c.identity for c in self._stage_configs(stage)},
            "stage-local configuration recipe required")
        path = Path(path).resolve(); before = path.read_bytes()
        require(self.contract["input_hashes"].get(str(path)) == hashlib.sha256(before).hexdigest(),
            "configuration recipe must be prospectively pinned")
        review = json.loads(before)
        expected = configuration_review_template(self.contract,stage=stage,configuration=configuration,
            raw_runtime_sha256=review.get("raw_runtime_sha256"),
            selector_recipe_sha256=review.get("selector_recipe_sha256"),
            qualification_evidence_hashes=review.get("qualification_evidence_hashes"),
            cost_scope=review.get("cost_scope"))
        expected.update(status="CLEAR_CONFIGURATION_RECIPE",reviewed_before_first_source_work=True,
            atomic_work_and_cleanup_validated=True,resource_cost_qualification_reviewed=True)
        require(digest(review) == digest(expected), "exact prospective configuration recipe review required")
        require(all(self.contract["input_hashes"].get(p) == h for p,h in
            {**review["controller_input_hashes"],**review["qualification_evidence_hashes"]}.items()),
            "code and separate qualification evidence must be prospectively pinned")
        return deepcopy(review), before

    def record_stage_evidence(self, stage, configuration, bundles, *, public_bindings,
                              runtime_hashes, qualification_review_path,
                              validate_selector, validate_outcome, _owned_stage_inputs=None):
        """Reconcile supplied receipts only; never execute a scientific collector.

        Per-configuration qualification is externally reviewed and code/input
        bound. A durable claim precedes validators: malformed evidence poisons
        this ledger; it cannot be silently repaired and retried. Shared raw
        evidence must be identical across every configuration and stage.
        """
        from .search_over_raw_ledger import _load, _snapshot, _write_new
        from .search_over_raw_evidence import reconcile_root_evidence
        self.verify()
        require(not (self.directory / "selection.json").exists(), "exploration is frozen")
        require(stage != "A2" or not (self.directory / "a2-freeze.json").exists(), "A2 is frozen")
        cfg = next((c for c in self._stage_configs(stage) if c.identity == configuration), None)
        require(cfg is not None, "stage-local registered configuration required")
        require(not (stage == "A3" and cfg.seconds == 1.),
            "A3 one-second evidence must reuse the frozen A2 record, not revalidate or rerun")
        slots = {r["root_id"]: r for r in self.contract["panels"]["exploration"]["root_slots"]}
        require(type(public_bindings) is dict and set(public_bindings) <= set(slots)
            and all(_sha(h) for h in public_bindings.values()), "supplied public source manifest required")
        require(type(bundles) is dict and set(bundles) <= set(public_bindings),
            "unregistered or unbound root evidence")
        frozen_inputs = digest(dict(bundles=bundles, public_bindings=public_bindings,
            runtime_hashes=runtime_hashes))
        path = Path(qualification_review_path).resolve()
        before = path.read_bytes()
        review_sha = hashlib.sha256(before).hexdigest()
        require(self.contract["input_hashes"].get(str(path)) == review_sha,
            "per-configuration review must be prospectively input-bound")
        review = json.loads(before)
        require(review.get("schema") in (None,CONFIGURATION_REVIEW_SCHEMA),
            "unsupported configuration review schema; no silent upgrade")
        per_root_runtime=stage == "A1" and cfg.belief == "oracle"
        prospective = review.get("schema") == CONFIGURATION_REVIEW_SCHEMA
        if prospective:
            from .search_over_raw_root_plan import _OwnedStageInputs
            inputs = _owned_stage_inputs
            require(type(inputs) is _OwnedStageInputs and inputs.ledger is self
                and (inputs.stage,inputs.configuration) == (stage,configuration)
                and bundles is inputs.bundles and public_bindings is inputs.public_bindings
                and runtime_hashes is inputs.runtime_hashes
                and getattr(validate_selector,"__self__",None) is inputs
                and getattr(validate_selector,"__func__",None) is _OwnedStageInputs.validate_selector
                and getattr(validate_outcome,"__self__",None) is inputs
                and getattr(validate_outcome,"__func__",None) is _OwnedStageInputs.validate_outcome,
                "v2 configuration recipe requires exact owned ingestion, not generic callbacks")
            preparation = getattr(self,"_owned_stage_handoffs",{}).get((stage,configuration))
            require(type(preparation) is tuple and len(preparation) == 3 and preparation[0] is inputs,
                "current owned preparation handoff required; stranded claims cannot reopen")
            preparation_path,preparation_bytes = preparation[1:]
            def check_preparation():
                require(getattr(self,"_owned_stage_handoffs",{}).get((stage,configuration)) is preparation
                    and preparation_path.read_bytes() == preparation_bytes,
                    "owned preparation claim or current handoff changed during ingestion")
            check_preparation()
            require(digest(_load(preparation_path)) == digest(dict(stage=stage,configuration=configuration,
                contract_sha256=self.contract_sha256,qualification_review_sha256=review_sha,retry_authorized=False)),
                "owned preparation claim binding mismatch")
            checked,checked_bytes = self.configuration_recipe_review(stage,configuration,path)
            require(checked_bytes == before and digest(checked) == digest(review), "configuration review changed")
            zero_work_policy = True
        elif per_root_runtime:
            require(type(runtime_hashes) is dict and runtime_hashes
                and all(type(row) is dict and "raw" in row for row in runtime_hashes.values())
                and len({row["raw"] for row in runtime_hashes.values()}) == 1,
                "A1 original-team roots require one common raw runtime")
            require(type(runtime_hashes) is dict and set(runtime_hashes) == set(bundles) == set(public_bindings)
                and all(type(value) is dict and set(value) == {"raw",cfg.identity}
                    and all(_sha(h) for h in value.values()) for value in runtime_hashes.values())
                and review.get("runtime_binding_schema") == "pokezero.search-over-raw.a1-original-team-root-runtimes.v1"
                and review.get("runtime_by_root_sha256") == digest(runtime_hashes)
                and "runtime_sha256" not in review,
                "A1 original-team exact prospective root/runtime map required")
            runtime_review_matches=True
        else:
            require("runtime_by_root_sha256" not in review and "runtime_binding_schema" not in review,
                "public configuration cannot adopt original-team root-runtime scope")
            runtime_review_matches=review.get("runtime_sha256") == runtime_hashes.get(cfg.identity)
        if not prospective:
            zero_work_policy = review.get("zero_work_is_local_unqualified",False)
            require(type(zero_work_policy) is bool, "explicit reviewed zero-work policy required")
            require(review.get("status") in ({"CLEAR_CONFIGURATION_AT_CEILING","CLEAR_UNQUALIFIED_AT_CEILING"}
                    if zero_work_policy else {"CLEAR_CONFIGURATION_AT_CEILING"})
                and review.get("configuration_sha256") == cfg.identity
                and runtime_review_matches
                and review.get("public_bindings_sha256") == digest(public_bindings)
                and review.get("atomic_work_and_cleanup_validated") is True,
                "qualified configuration, source bindings and work/cleanup review required")
        claim = self.directory / stage / f"{configuration}-evidence-claim.json"
        _write_new(claim, dict(configuration=configuration, stage=stage,
            contract_sha256=self.contract_sha256, bundles_sha256=digest(bundles),
            qualification_review_sha256=review_sha, retry_authorized=False))
        try:
            if prospective:
                # Observed owners must match the recipe AFTER the durable claim.
                # Future root truth is independently checked by their archive
                # readers, not supplied by the prospective review.
                for plan in inputs.plans.values():
                    raw_reader,selector = plan.readers["raw"],plan.readers[configuration]
                    require(raw_reader.runtime_sha == review["raw_runtime_sha256"]
                        and digest(raw_reader.runtime) == raw_reader.runtime_sha
                        and digest(selector.runtime) == selector.runtime_sha
                        and digest(selector_runtime_recipe(cfg,selector.runtime)) == review["selector_recipe_sha256"],
                        "observed owner runtime differs from prospective configuration recipe")
            rows = {}
            raw = SearchConfiguration("raw", seconds=1.)
            for root_id, bundle in bundles.items():
                row = reconcile_root_evidence(bundle, namespace=self.contract["namespace"],
                    slot=slots[root_id], configurations=(raw, cfg),
                    public_record_sha256=public_bindings[root_id], runtime_hashes=
                        runtime_hashes[root_id] if per_root_runtime else runtime_hashes,
                    validate_selector=validate_selector, validate_outcome=validate_outcome,
                    allow_unqualified_at_ceiling=zero_work_policy)
                # Bind actual shared raw receipt and its validation, not merely
                # an action or a coincident seed. No per-arm control redraw.
                if _owned_stage_inputs is None:
                    self._shared_action_evidence(row, create=True)
                rows[root_id] = row
            self.verify()
            require(digest(dict(bundles=bundles, public_bindings=public_bindings,
                runtime_hashes=runtime_hashes)) == frozen_inputs,
                "collector input evidence drift during reconciliation")
            require(path.read_bytes() == before, "configuration review drift during reconciliation")
            if prospective: check_preparation()
            handoff = None
            if _owned_stage_inputs is not None:
                from .search_over_raw_root_plan import _OwnedStageInputs
                require(type(_owned_stage_inputs) is _OwnedStageInputs
                    and _owned_stage_inputs.ledger is self
                    and (_owned_stage_inputs.stage,_owned_stage_inputs.configuration) == (stage,configuration),
                    "exact private owned-stage inputs required")
                handoff = _owned_stage_inputs.finalize(rows)
                # The final owned hook also performs callback-bearing runtime
                # checks. Recheck prospective ledger/review/bundle invariants
                # after it, before any successful stage evidence write.
                self.verify()
                require(path.read_bytes() == before and digest(dict(bundles=bundles,
                    public_bindings=public_bindings,runtime_hashes=runtime_hashes)) == frozen_inputs,
                    "owned final hook changed ledger admission inputs")
                if prospective: check_preparation()
                require(handoff["schema"] == "pokezero.search-over-raw.owned-stage-inputs.v2"
                    and set(handoff["attempted_actions_by_root"]) == set(rows),
                    "owned attempted-action roster differs from validated roots")
                for root,row in rows.items():
                    self._shared_action_evidence(row, create=True,
                        attempted_actions=handoff["attempted_actions_by_root"][root])
            intervals = {root: row["root_intervals"][configuration] for root, row in rows.items()}
            accepted = sum(row["accepted_units"][configuration] for row in rows.values())
            status = "QUALIFIED" if accepted > 0 else "UNQUALIFIED_AT_CEILING"
            if accepted > 0:
                require(prospective or review["status"] == "CLEAR_CONFIGURATION_AT_CEILING",
                    "productive stage requires configuration qualification review")
            else:
                require(zero_work_policy and (prospective or review["status"] == "CLEAR_UNQUALIFIED_AT_CEILING")
                    and rows and all(row["selector_statuses"][configuration] == "UNQUALIFIED_AT_CEILING"
                        for row in rows.values()), "zero-work stage requires observed owned attempts and local unqualified review")
                # Keep every observed zero-work selector in root_evidence,
                # without manufacturing a scored action in the stage summary.
                intervals = {}
            evidence = dict(contract_sha256=self.contract_sha256, stage=stage, configuration=configuration,
                qualification_review_sha256=review_sha, root_evidence=_snapshot(rows),
                root_intervals=intervals, accepted_units=accepted, runtime_qualified_by_this_code=False)
            if per_root_runtime:
                evidence["selector_runtime_by_root"]=deepcopy(runtime_hashes)
            if handoff is not None:
                evidence["owned_input_handoff"] = handoff
            if prospective:
                evidence["owned_preparation_claim_sha256"] = hashlib.sha256(preparation_bytes).hexdigest()
            _write_new(self.directory / stage / f"{configuration}-root-evidence.json", evidence)
            return self.record_stage(stage, configuration, intervals, accepted_units=accepted,
                status=status, evidence_sha256=digest(evidence), _from_reconciled_evidence=True)
        except BaseException as error:
            if not (self.directory / "operational-stop.json").exists():
                try: self.stop(type(error).__name__)
                except BaseException as stop_error:
                    error.add_note("Could not persist operational stop: "+type(stop_error).__name__)
            raise

    def record_owned_stage_evidence(self, stage, configuration, *, plans, continuations,
                                    qualification_review_path,source_panel=None,input_unavailable=None):
        """Ingest actual archived owners, without collection or caller callbacks.

        Missing continuation results remain uncertain; supplied malformed or
        failed attempts cannot clear. This does not upgrade the generic evidence
        API, qualify a runtime, admit source generation or open the holdout.
        """
        from .search_over_raw_root_plan import _OwnedStageInputs
        from .search_over_raw_ledger import _load, _write_new
        self.verify()
        path = Path(qualification_review_path).resolve()
        review_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        require(self.contract["input_hashes"].get(str(path)) == review_sha,
            "per-configuration review must be prospectively input-bound")
        prospective = _load(path).get("schema") == CONFIGURATION_REVIEW_SCHEMA
        if prospective:
            require(not (self.directory / "selection.json").exists(), "exploration is frozen")
            require(stage != "A2" or not (self.directory / "a2-freeze.json").exists(), "A2 is frozen")
            cfg = next((c for c in self._stage_configs(stage) if c.identity == configuration),None)
            require(cfg is not None, "stage-local registered configuration required")
            require(not (stage == "A3" and cfg.seconds == 1.), "A3 one-second evidence must reuse frozen A2")
            # Construction also inspects owner/archive input scope. Claim it
            # first, so a malformed scope or interrupted preparation cannot be
            # silently replaced by a second owned-ingestion attempt.
            preparation_path = self.directory / stage / f"{configuration}-owned-preparation-claim.json"
            _write_new(preparation_path,dict(stage=stage,configuration=configuration,
                contract_sha256=self.contract_sha256,qualification_review_sha256=review_sha,retry_authorized=False))
            preparation_bytes = preparation_path.read_bytes()
        try:
            inputs = _OwnedStageInputs(self,stage,configuration,plans,continuations,source_panel=source_panel,
                input_unavailable=input_unavailable)
            if prospective:
                require(preparation_path.read_bytes() == preparation_bytes,
                    "owned preparation claim changed during construction")
                if not hasattr(self,"_owned_stage_handoffs"): self._owned_stage_handoffs = {}
                self._owned_stage_handoffs[(stage,configuration)] = (inputs,preparation_path,preparation_bytes)
            return self.record_stage_evidence(stage,configuration,inputs.bundles,
                public_bindings=inputs.public_bindings,runtime_hashes=inputs.runtime_hashes,
                qualification_review_path=qualification_review_path,
                validate_selector=inputs.validate_selector,validate_outcome=inputs.validate_outcome,
                _owned_stage_inputs=inputs)
        except BaseException as error:
            if prospective and not (self.directory / "operational-stop.json").exists():
                try: self.stop(type(error).__name__)
                except BaseException as stop_error:
                    error.add_note("Could not persist operational stop: "+type(stop_error).__name__)
            raise
        finally:
            if prospective:
                getattr(self,"_owned_stage_handoffs",{}).pop((stage,configuration),None)

    def _rows(self, stage):
        from .search_over_raw_ledger import _load
        artifacts, summaries = {}, []
        roster = panel_roster(self.contract["panels"]["exploration"])
        for cfg in self._stage_configs(stage):
            artifact = _load(self.directory / stage / f"{cfg.identity}.json")
            require(artifact["contract_sha256"] == self.contract_sha256 and artifact["stage"] == stage
                and artifact["configuration"] == cfg.identity and _sha(artifact["evidence_sha256"]), "stage receipt drift")
            rows = cluster_intervals(roster, artifact["root_intervals"])
            row = StageSummary(**artifact["summary"])
            evidence_path = self.directory / stage / f"{cfg.identity}-root-evidence.json"
            require(not (self.directory / stage / f"{cfg.identity}-evidence-claim.json").exists()
                or evidence_path.exists(), "claimed reconciliation has missing evidence")
            if self.contract["execution_ready"] and row.status == "QUALIFIED":
                require(evidence_path.exists() or (stage == "A3" and cfg.seconds == 1.),
                    "execution-ready stage requires reconciled root evidence")
            if evidence_path.exists():
                evidence = _load(evidence_path)
                if "owned_preparation_claim_sha256" in evidence:
                    preparation_path = self.directory / stage / f"{cfg.identity}-owned-preparation-claim.json"
                    require(hashlib.sha256(preparation_path.read_bytes()).hexdigest()
                        == evidence["owned_preparation_claim_sha256"], "accepted owned preparation claim drift")
                require(digest(evidence) == artifact["evidence_sha256"]
                    and evidence["root_intervals"] == artifact["root_intervals"]
                    and evidence["accepted_units"] == row.accepted_units,
                    "reconciled root evidence differs from stage summary")
                if "prospective_configuration_unavailable" in evidence:
                    from .search_over_raw_stage_run import read_unavailable_stage_evidence
                    read_unavailable_stage_evidence(self,stage,cfg.identity,evidence)
                    require(row.status=="UNSTARTED" and row.accepted_units==0,
                        "prospective unavailability must remain unstarted in selection")
                handoff = evidence.get("owned_input_handoff",{})
                scope = None
                if "owned_input_handoff" in evidence:
                    require(handoff.get("schema") in {
                        "pokezero.search-over-raw.owned-stage-inputs.v1",
                        "pokezero.search-over-raw.owned-stage-inputs.v2"},
                        "unknown owned-stage input schema")
                    require(handoff["schema"].endswith(".v2")
                        or "attempted_actions_by_root" not in handoff,
                        "legacy owned-stage receipt cannot supply attempted scope")
                if handoff.get("schema") == "pokezero.search-over-raw.owned-stage-inputs.v2":
                    scope = handoff["attempted_actions_by_root"]
                    require(type(scope) is dict and set(scope) == set(evidence["root_evidence"]),
                        "frozen owned attempted-action roster drift")
                if "input_unavailable_by_root" in handoff or "input_unavailable_by_root" in evidence:
                    gaps=handoff.get("input_unavailable_by_root",evidence.get("input_unavailable_by_root"))
                    panel=handoff.get("source_panel",evidence.get("source_panel"))
                    fixed_slots={r["root_id"] for r in self.contract["panels"]["exploration"]["root_slots"]}
                    require(type(gaps) is dict and gaps and type(panel) is dict
                        and panel["contract_sha256"]==self.contract_sha256
                        and type(panel["observed_root_ids"]) is list and type(panel["missing_root_ids"]) is list
                        and len(set(panel["observed_root_ids"]))==len(panel["observed_root_ids"])
                        and len(set(panel["missing_root_ids"]))==len(panel["missing_root_ids"])
                        and not set(panel["observed_root_ids"])&set(panel["missing_root_ids"])
                        and set(panel["observed_root_ids"])|set(panel["missing_root_ids"])==fixed_slots
                        and not set(gaps)&set(evidence["root_evidence"])
                        and set(gaps)|set(evidence["root_evidence"])==set(panel["observed_root_ids"])
                        and not set(gaps)&set(artifact["root_intervals"]),
                        "durable stage lost available/unavailable/source-missing partition")
                    if "input_unavailable_by_root" in evidence:
                        require(evidence["schema"]=="pokezero.search-over-raw.empty-panel-stage.v1"
                            and evidence["status"]==row.status=="UNSTARTED"
                            and evidence["selector_attempted"] is False and not evidence["root_evidence"],
                            "all-unavailable inputs are unstarted, never observed zero-work")
                    from .search_over_raw_selector_producer import read_recorded_actor_input_unavailable
                    for root,receipt in gaps.items():
                        require(receipt["root_id"]==root,"durable unavailable root coordinate drift")
                        read_recorded_actor_input_unavailable(self,receipt,configuration=cfg)
                for root_id,root in evidence["root_evidence"].items():
                    self._shared_action_evidence(root, create=False,
                        attempted_actions=None if scope is None else scope[root_id])
                if "owned_input_handoff" in evidence:
                    for relative,sha in evidence["owned_input_handoff"].get("shared_raw_provenance",{}).items():
                        require(relative.startswith("owned-shared-raw/")
                            and (self.directory/relative).resolve().is_relative_to(self.directory.resolve())
                            and digest(_load(self.directory/relative)) == sha,
                            "actual shared raw owner provenance changed")
                    for relative,sha in evidence["owned_input_handoff"]["shared_cell_provenance"].items():
                        require(relative.startswith("owned-shared-cells/")
                            and (self.directory/relative).resolve().is_relative_to(self.directory.resolve())
                            and digest(_load(self.directory/relative)) == sha,
                            "actual shared continuation provenance changed")
            expected_stage, expected_lane = (("A2", "wall_clock_diagnostic") if stage == "A1"
                else (stage, "deployment"))
            require((row.stage, row.arm, row.belief, row.leaf, row.seconds, row.lane) ==
                (expected_stage, cfg.arm, cfg.belief, cfg.leaf, cfg.seconds, expected_lane)
                and row.lower_mean == math.fsum(x[0] for x in rows)/len(rows), "stage summary/evidence mismatch")
            require(row.status not in {"UNSTARTED", "UNQUALIFIED_AT_CEILING"}
                or (row.accepted_units == 0 and not artifact["root_intervals"]), "unmeasured receipt cannot score outcomes")
            if stage == "A3" and cfg.seconds == 1.:
                prior = _load(self.directory / "A2" / f"{cfg.identity}.json")
                require(artifact["root_intervals"] == prior["root_intervals"]
                    and artifact["evidence_sha256"] == prior["evidence_sha256"]
                    and row.accepted_units == prior["summary"]["accepted_units"]
                    and row.status == prior["summary"]["status"], "A3 one-second reuse drift")
            artifacts[cfg.identity] = digest(artifact)
            summaries.append(row)
        return artifacts, summaries

    def report_stage(self, stage):
        """Independent descriptive A1/A2/A3 report; no cross-stage pass gate."""
        from .search_over_raw_ledger import _load
        self.verify()
        artifacts, summaries = self._rows(stage)
        roster = panel_roster(self.contract["panels"]["exploration"])
        reports = []
        for cfg, summary in zip(self._stage_configs(stage), summaries):
            artifact = _load(self.directory / stage / f"{cfg.identity}.json")
            intervals = cluster_intervals(roster, artifact["root_intervals"])
            row = dict(asdict(summary), stage=stage, configuration_sha256=cfg.identity,
                root_slots=sum(len(quota) for _, quota in roster), source_seeds=len(roster),
                identification_interval=[math.fsum(x[i] for x in intervals)/len(intervals) for i in (0, 1)],
                uncertain_roots=sum(artifact["root_intervals"].get(root, [-1., 1.])[0]
                    != artifact["root_intervals"].get(root, [-1., 1.])[1] for _, quota in roster for root in quota),
                seed_intervals=[dict(source_seed=seed, interval=list(interval))
                    for (seed, _), interval in zip(roster, intervals)])
            reports.append(row)
        return dict(stage=stage, contract_sha256=self.contract_sha256, artifacts=artifacts,
            configurations=reports, other_stages_required=stage == "A3", confirmatory_strength=False,
            collection_authorized=False, holdout_authorized=False, phase_b_authorized=False)

    def freeze_evaluators(self):
        from .search_over_raw_ledger import _write_new
        self.verify()
        artifacts, rows = self._rows("A2")
        value = dict(contract_sha256=self.contract_sha256, artifacts=artifacts,
            evaluators=choose_evaluators(rows), a1_prerequisite=False, runtime_authorized=False)
        _write_new(self.directory / "a2-freeze.json", value)
        return deepcopy(value)

    def _a2(self):
        from .search_over_raw_ledger import _load
        value = _load(self.directory / "a2-freeze.json")
        artifacts, rows = self._rows("A2")
        require(value == dict(contract_sha256=self.contract_sha256, artifacts=artifacts,
            evaluators=[list(r) for r in choose_evaluators(rows)], a1_prerequisite=False,
            runtime_authorized=False), "frozen A2 receipt drift")
        return value

    def freeze(self):
        from .search_over_raw_ledger import _write_new
        self.verify()
        a2 = self._a2()
        artifacts, rows = self._rows("A3")
        evaluators = tuple(tuple(r) for r in a2["evaluators"])
        choices = choose_ceilings(evaluators, rows)
        value = dict(schema="pokezero.search-over-raw.selection.v3",
            contract_sha256=self.contract_sha256, a2_sha256=digest(a2), artifacts=artifacts,
            choices=choices, validation_opened=False, phase_b_authorized=False)
        _write_new(self.directory / "selection.json", value)
        return deepcopy(value)

    def _selection(self):
        from .search_over_raw_ledger import _load
        value = _load(self.directory / "selection.json")
        a2 = self._a2()
        artifacts, rows = self._rows("A3")
        choices = choose_ceilings(tuple(tuple(r) for r in a2["evaluators"]), rows)
        require(value == dict(schema="pokezero.search-over-raw.selection.v3",
            contract_sha256=self.contract_sha256, a2_sha256=digest(a2), artifacts=artifacts,
            choices=[list(r) for r in choices], validation_opened=False, phase_b_authorized=False),
            "frozen selection drift")
        return value

    def validation_once(self, collect, *, review_path):
        """Requires separate bound clearance; claim before callback, never retry."""
        from .search_over_raw_ledger import _load, _snapshot, _write_new
        self.verify(execution=True)
        selection = self._selection()
        review_path = Path(review_path).resolve()
        review_bytes = review_path.read_bytes()
        review = json.loads(review_bytes)
        analysis_sha = hashlib.sha256(Path(__file__).with_name("search_over_raw_screening.py").read_bytes()).hexdigest()
        require(review.get("status") == "CLEAR_ONE_HOLDOUT_SCREENING_ATTEMPT"
            and review.get("contract_sha256") == self.contract_sha256
            and review.get("selection_sha256") == digest(selection)
            and review.get("analysis_code_sha256") == analysis_sha
            and review.get("user_authorized_holdout") is True
            and review.get("screening_risk_reviewed") is True
            and _sha(review.get("calibration_result_sha256"))
            and _sha(review.get("runtime_qualification_sha256")), "separate bound holdout clearance required")
        configurations = {arm: None if leaf is None else SearchConfiguration(arm, "public", leaf,
            seconds, 20 if arm == "reference" else 1).identity for arm, leaf, seconds in selection["choices"]}
        require(any(c is not None for c in configurations.values()), "neither arm available; holdout must stay unopened")
        require(type(review.get("qualified_configurations")) is list
            and sorted(review["qualified_configurations"]) == sorted(c for c in configurations.values() if c is not None),
            "qualify actual available choices only; no missing arm or factorial prerequisite")
        evidence_paths = review.get("evidence_files", {})
        require(set(evidence_paths) == {"calibration_result", "runtime_qualification"}
            and all(type(p) is str and Path(p).is_absolute() for p in evidence_paths.values()),
            "bound calibration and runtime evidence paths required")
        def verify_evidence():
            require(hashlib.sha256(Path(__file__).with_name("search_over_raw_screening.py").read_bytes()).hexdigest()
                == analysis_sha, "screening code drift during validation")
            for name, path in evidence_paths.items():
                require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == review[name + "_sha256"],
                    "holdout clearance evidence drift: " + name)
            calibration = _load(Path(evidence_paths["calibration_result"]))
            require(calibration.get("schema") == "pokezero.staged-bootstrap-synthetic-calibration.v1.results"
                and calibration.get("calibration_complete") is True
                and calibration.get("input_sha256", {}).get(str(Path(__file__).with_name("search_over_raw_screening.py"))) == analysis_sha,
                "completed calibration must bind current screening code")
        verify_evidence()
        for name in ("validation-claim.json", "validation-result.json", "validation-failure.json"):
            if (self.directory / name).exists():
                raise FileExistsError("held-out attempt already has evidence: " + name)
        review_sha = hashlib.sha256(review_bytes).hexdigest()
        claim = dict(contract_sha256=self.contract_sha256, selection_sha256=digest(selection),
            review_sha256=review_sha, configurations=configurations, status="VALIDATION_ATTEMPT_CLAIMED",
            retries_permitted=False)
        _write_new(self.directory / "validation-claim.json", claim)
        def verify_attempt():
            self.verify(execution=True)
            require(digest(self._selection()) == claim["selection_sha256"]
                and _load(self.directory / "validation-claim.json") == claim
                and hashlib.sha256(review_path.read_bytes()).hexdigest() == review_sha,
                "selection, claim or holdout review drift during validation")
            verify_evidence()
        try:
            roots = _snapshot(collect(deepcopy(self.contract), deepcopy(configurations)))
            require(set(roots) == set(ARMS), "both declared arms required; unavailable is explicit null")
            verify_attempt()
            roster = panel_roster(self.contract["panels"]["validation"])
            intervals = {}
            for arm in ARMS:
                if configurations[arm] is None:
                    require(roots[arm] is None, "unavailable arm cannot invent held-out evidence")
                    intervals[arm] = None
                else:
                    require(type(roots[arm]) is dict, "full-roster interval mapping required")
                    intervals[arm] = cluster_intervals(roster, roots[arm])
            gate = screen(ScreeningDesign(self.contract["namespace"], tuple(s for s, _ in roster)), intervals)
            verify_attempt()
            receipt = dict(contract_sha256=self.contract_sha256, selection_sha256=digest(selection),
                review_sha256=review_sha, root_intervals=roots, screening=gate,
                status="VALIDATION_ATTEMPT_FINISHED", phase_b_authorized=False,
                full_game_strength_established=False)
        except BaseException as error:
            _write_new(self.directory / "validation-failure.json", dict(contract_sha256=self.contract_sha256,
                selection_sha256=digest(selection), status="VALIDATION_ATTEMPT_FAILED",
                error_type=type(error).__name__, all_validation_roots_uncertain=True,
                retries_permitted=False, phase_b_authorized=False))
            raise
        _write_new(self.directory / "validation-result.json", receipt)
        return receipt
