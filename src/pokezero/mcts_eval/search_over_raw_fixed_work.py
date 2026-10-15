"""Fixed accepted-work backend for the reconciled A2 diagnostic only.

This is not a collector or execution admission. A future producer must supply
an independently reviewed external 1800-second emergency supervisor and report
whole-process CPU including children, startup and cleanup. Neither an adapter
selection nor its elapsed time is a completed resource-cost qualification.
"""
from dataclasses import asdict, dataclass

from .search_over_raw import digest, require
from .search_over_raw_adapters import PublicModelSearchAdapter, _incumbent_runtime


@dataclass(frozen=True)
class FixedWorkDiagnosticContract:
    accepted_target: int = 64
    incumbent_iterations_per_world: int = 16
    incumbent_worlds: int = 4
    incumbent_batch: int = 16
    reference_workers: int = 20
    reference_trajectories_per_worker: int = 4
    reference_exchange_batch: int = 10
    emergency_watchdog_seconds: int = 1800

    def __post_init__(self):
        expected = (64, 16, 4, 16, 20, 4, 10, 1800)
        require(all(type(v) is int for v in asdict(self).values())
            and tuple(asdict(self).values()) == expected,
            "fixed-work target and backend boundary require a prospective amendment")

    def accepted_boundary(self, arm):
        require(arm in {"incumbent", "reference"}, "fixed-work diagnostic requires a search arm")
        return (self.incumbent_iterations_per_world * self.incumbent_worlds if arm == "incumbent"
            else self.reference_workers * self.reference_trajectories_per_worker)

    def receipt(self):
        return dict(schema="pokezero.search-over-raw.fixed-work.v1", **asdict(self),
            search_decision_ceiling=None, deployment_eligible=False,
            incumbent_accepted_unit="complete_native_iteration_with_committed_backup",
            reference_accepted_unit="complete_reference_trajectory_with_committed_backup",
            reference_boundary_rule="first_complete_symmetric_20_worker_allocation_at_or_above_target",
            equal_cpu_claim=False, watchdog_is_external=True,
            watchdog_result="DIAGNOSTIC_INCOMPLETE", runtime_qualified=False)


class FixedWorkDiagnosticAdapter(PublicModelSearchAdapter):
    """Public model/HP/terminal comparisons; unchanged tree/worlds/priors.

The inherited SearchConfiguration's one-second value is an identity placeholder
only: no decision deadline is passed to either backend. This distinct runtime
binding, not that legacy configuration identity alone, identifies the diagnostic.
Refusal still poisons this owned adapter; a producer must not retry it or promote
its diagnostic action into the deployment lane.
"""
    def __init__(self, configuration, *, fixed_work, **kwargs):
        require(type(fixed_work) is FixedWorkDiagnosticContract,
            "explicit fixed-work contract required")
        require(configuration.arm in {"incumbent", "reference"}
            and configuration.belief == "public" and configuration.seconds == 1.,
            "fixed-work diagnostic requires public search and its canonical identity placeholder")
        self.fixed_work = fixed_work
        super().__init__(configuration, **kwargs)
        self.runtime_configuration["fixed_work_diagnostic"] = fixed_work.receipt()
        if configuration.arm == "incumbent":
            self.runtime_configuration["incumbent"].update(sims=16, native_batch_guard_ms=0)
            if configuration.leaf == "raw_rollout":
                self.runtime_configuration["incumbent_leaf"]["deadline"] = "none; external emergency watchdog only"
        self.runtime_sha256 = digest(self.runtime_configuration)

    def _make_incumbent_runtime(self, contract, showdown_root, policy_opponent_diagnostics):
        return _incumbent_runtime(contract, showdown_root, self.configuration.seconds,
            leaf=self.configuration.leaf, fixed_work_iterations=self.fixed_work.incumbent_iterations_per_world,
            **({"policy_opponent_diagnostics": policy_opponent_diagnostics}
               if policy_opponent_diagnostics is not None else {}))

    def _validate_incumbent_selection(self, evidence, mask, selection_seed):
        from .policy_opponent_profile import validate_selection
        validate_selection(evidence, arm="incumbent_mcts", mode="fixed_work",
            config=self._search_config, mask=mask, opponent_seed=selection_seed,
            deadline_ms=1000, native_batch_guard_ms=0,
            **({"model_leaf_override": "raw_policy_terminal"
                if self.configuration.leaf == "raw_rollout" else self.configuration.leaf}
                if self.configuration.leaf != "model" else {}))

    def _search_reference(self, request, root, statistics_root, selection_seed, started):
        return self._pool.search(self._diagnostic_request(request), root,
            battle_id="search-over-raw-root:"+statistics_root, seed=selection_seed,
            trajectories_per_worker=self.fixed_work.reference_trajectories_per_worker)

    def _validate_diagnostic_work(self, measured):
        require(measured.result.trajectories == self.fixed_work.accepted_boundary("reference")
            and measured.result.deadline_exhausted is False,
            "fixed-work reference allocation incomplete or decision deadline applied")
        counts = {i: 0 for i in range(self.fixed_work.reference_workers)}
        for row in measured.worker_receipts:
            worker, batch = row["worker"], row["batch"]
            require(type(worker) is int and worker in counts and not batch.deadline_exhausted,
                "fixed-work reference worker or deadline witness drift")
            counts[worker] += batch.trajectories
        require(all(v == self.fixed_work.reference_trajectories_per_worker for v in counts.values()),
            "fixed-work reference must complete the frozen symmetric allocation")

    def _execution_receipt(self, elapsed, evidence):
        accepted = (evidence["total_iterations"] if self.configuration.arm == "incumbent"
            else evidence["result"]["trajectories"])
        require(type(accepted) is int and accepted == self.fixed_work.accepted_boundary(self.configuration.arm),
            "fixed-work diagnostic lacks the complete accepted allocation")
        return dict(nominal_search_seconds=None, exceeded_nominal_seconds=None,
            fixed_work_diagnostic=dict(contract=self.fixed_work.receipt(),
                accepted_target=self.fixed_work.accepted_target, accepted_actual=accepted,
                selection_wall_seconds=elapsed, deployment_eligible=False,
                cpu_seconds_including_children=None, resource_cost_complete=False,
                startup_cleanup_and_watchdog_evidence_required=True))
