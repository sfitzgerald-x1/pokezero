"""Public/model decision adapters for prospective Phase A root experiments.

Not an admission controller: source/native/runtime qualification and durable
attempt receipts remain the collector's responsibility. Implicit oracle and
unimplemented incumbent raw-terminal leaves are rejected. Reference
leaf ablations change valuation only, retaining champion priors and the tree.
The reference retains its genuine twenty-worker kernel; the incumbent retains
the historical depth6/batch16/worlds4 configuration at one worker.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import math
import random
import time

from .search_over_raw import SearchConfiguration, _raw_action, digest, require


def validate_reference_work(measured) -> None:
    """Reconcile actual new work, including zero-backup deadline cancellations."""
    require(len(measured.worker_pids) == 20 and len(set(measured.worker_pids)) == 20
        and measured.result.trajectories > 0, "reference requires twenty workers and new trajectories")
    draws_total = completed = transitions = 0
    for receipt in measured.worker_receipts:
        batch, draws = receipt["batch"], receipt["evidence"]["draws"]
        require(len(draws) == batch.world_draws, "reference draw count drift")
        valid = 0
        for draw in draws:
            if draw["status"] == "ROOT_VALIDATED" and draw.get("released") is True:
                valid += 1
            elif draw["status"] == "DEADLINE_CANCELLED":
                diagnostic = draw.get("sampling_diagnostic", {})
                checked, deadline = diagnostic.get("checked_at"), diagnostic.get("deadline_at")
                require(diagnostic.get("schema") == "pokezero.world-sampling-deadline.v1"
                    and type(checked) in (int, float) and type(deadline) in (int, float)
                    and math.isfinite(checked) and math.isfinite(deadline) and checked >= deadline
                    and diagnostic.get("accepted_world") is False and diagnostic.get("backed_up") is False
                    and batch.deadline_exhausted, "cancelled world lacks clock/zero-backup witness")
            else:
                raise ValueError("refused or unreleased reference world")
        require(0 <= batch.trajectories <= valid and (valid > 0 or batch.transitions == 0),
            "cancelled reference attempts counted as completed work")
        draws_total += len(draws)
        completed += batch.trajectories
        transitions += batch.transitions
    require((draws_total, completed, transitions) == (measured.result.world_draws,
        measured.result.trajectories, measured.result.transitions), "reference aggregate work drift")


def _incumbent_runtime(contract, showdown_root, seconds, *, leaf="model"):
    from .manifest import SearchConfig
    from .policy_opponent_profile import make_profile_decider

    require(leaf in ("model", "hp_fraction", "raw_rollout"), "incumbent leaf requires an implemented model-tree valuation")
    decider = make_profile_decider(contract, showdown_root, arm="incumbent_mcts",
        mode="matched_deadline", opponent_seed=0, deadline_ms=round(seconds * 1000),
        native_batch_guard_ms=64,
        **({"model_leaf_override": "raw_policy_terminal" if leaf == "raw_rollout" else leaf}
           if leaf != "model" else {}))
    config = SearchConfig(depth=6, sims=4096, batch=16, worlds=4, inference_mode="local")
    try:
        return decider, decider._policy_for(config), config
    except BaseException:
        decider.close()
        raise


class PublicModelSearchAdapter:
    """Accept a public policy context, never a source env or true snapshot.

    Every root receives independent search statistics and one attempt. Any
    refusal poisons this adapter; no fallback, restart or retry is performed.
    A controller may persist the sanitized failure type and uncertain interval.
    It must not treat this object as scientific execution authorization.
    """

    def _check_configuration(self, configuration):
        require(configuration.belief == "public" and (configuration.leaf == "model"
            or configuration.arm == "reference"
            or (configuration.arm == "incumbent" and configuration.leaf in {"hp_fraction", "raw_rollout"})),
            "oracle and incumbent alternative-leaf adapters remain required")

    def _reference_worker_factory(self, factory):
        if self.configuration.leaf != "model":
            from .search_over_raw_leaves import ReferenceLeafWorkerFactory
            return ReferenceLeafWorkerFactory(factory, self.configuration.leaf)
        return factory

    def _diagnostic_request(self, request):
        return request

    def _prepare_incumbent(self, request):
        require(getattr(self._native, "_fixed_override", None) is None,
            "public incumbent contains an oracle override")

    def _validate_diagnostic_work(self, measured):
        pass

    def _selection_evidence(self, evidence, request):
        return evidence

    def __init__(self, configuration: SearchConfiguration, *, checkpoint_contract,
                 showdown_root: str, evaluator=None, reference_factory=None,
                 initial_dispatch_workers: int = 6):
        from .paper_reference_runtime import ShowdownWorkerFactory

        self._check_configuration(configuration)
        require(configuration.workers == (20 if configuration.arm == "reference" else 1),
            "resource allocation differs from the registered arm")
        require(type(initial_dispatch_workers) is int and 1 <= initial_dispatch_workers <= 20,
            "invalid reference dispatch width")
        require(configuration.arm != "incumbent" or configuration.seconds * 1000 > 64,
            "incumbent deadline must exceed its native batch guard")
        if configuration.arm == "reference":
            require(isinstance(reference_factory, ShowdownWorkerFactory), "explicit reference factory required")
            require((reference_factory.checkpoint, reference_factory.checkpoint_sha256,
                reference_factory.showdown_root, reference_factory.set_source_hash) == (
                checkpoint_contract.checkpoint_path, checkpoint_contract.checkpoint_sha256,
                showdown_root, checkpoint_contract.showdown_source_sha256),
                "reference checkpoint/simulator/catalog binding differs")
        if configuration.arm == "raw":
            from .paper_reference_showdown import ChampionEvaluator
            require(isinstance(evaluator, ChampionEvaluator)
                and evaluator.policy.weights_sha256 == checkpoint_contract.checkpoint_sha256,
                "raw evaluator must bind the unchanged champion")
        self.configuration = configuration
        self.contract = checkpoint_contract
        self._closed = self._poisoned = False
        self._attempted = set()
        self._decider = self._native = self._search_config = self._pool = None
        self._evaluator = evaluator
        self.last_failure = None
        self.runtime_configuration = dict(configuration=asdict(configuration),
            checkpoint_sha256=checkpoint_contract.checkpoint_sha256,
            set_source_hash=checkpoint_contract.showdown_source_sha256,
            root_statistics="independent_per_root",
            incumbent=dict(depth=6, sims=4096, batch=16, worlds=4, native_batch_guard_ms=64)
                if configuration.arm == "incumbent" else None,
            reference_factory=asdict(reference_factory) if configuration.arm == "reference" else None,
            initial_dispatch_workers=initial_dispatch_workers if configuration.arm == "reference" else None)
        if configuration.belief == "oracle":
            self.runtime_configuration["team_oracle"] = self.oracle.receipt()
        self.runtime_sha256 = digest(self.runtime_configuration)
        if configuration.arm == "incumbent":
            self._decider, self._native, self._search_config = _incumbent_runtime(
                checkpoint_contract, showdown_root, configuration.seconds,
                **({"leaf": configuration.leaf} if configuration.leaf != "model" else {}))
            if configuration.leaf != "model":
                self.runtime_configuration["incumbent_leaf"] = dict(leaf=configuration.leaf,
                    tree="unchanged_encoded_model_tree", priors="unchanged_champion",
                    model_forwards="retained", value_frame="side_one_absolute")
                if configuration.leaf == "raw_rollout":
                    self.runtime_configuration["incumbent_leaf"].update(
                        policy="both_seats_own_raw_masked_argmax", rollout_cap=250,
                        cap="refusal_without_value_fallback", deadline="whole_round_cancel_without_backup",
                        rollout_count=1, rollout_threads=1, branch_on_damage=True,
                        seed="selection_seed_then_sha256_world_domain_v1")
                self.runtime_sha256 = digest(self.runtime_configuration)
        elif configuration.arm == "reference":
            from .paper_reference import ReferenceConfig
            from .paper_reference_parallel import ParallelTrajectorySearch
            if configuration.leaf != "model":
                from .search_over_raw_leaves import ROLLOUT_CAP
                self.runtime_configuration["reference_leaf"] = dict(leaf=configuration.leaf,
                    rollout_cap=ROLLOUT_CAP, rollout_policy="raw_argmax_both_seats",
                    tree="unchanged_trajectory_reference", priors="unchanged_champion")
                self.runtime_sha256 = digest(self.runtime_configuration)
            self._pool = ParallelTrajectorySearch(ReferenceConfig(.5, 1.), self._reference_worker_factory(reference_factory),
                workers=20, batch_size=10, initial_dispatch_workers=initial_dispatch_workers)

    def select(self, context, *, root_id: str, selection_seed: int, pending_transition=None) -> dict:
        from .head_to_head import public_only_context
        from .paper_reference_runtime import PublicRootRequest
        from .paper_reference_showdown import decision_state

        require(not self._closed and not self._poisoned, "adapter closed or poisoned; no retry")
        require(type(root_id) is str and bool(root_id) and root_id not in self._attempted,
            "root requires one fresh attempt")
        require(type(selection_seed) is int and 0 <= selection_seed < 2**64, "invalid selection seed")
        self._attempted.add(root_id)
        started = time.perf_counter()
        try:
            require(context.player_id in context.requested_players
                and context.trajectory.terminal is None, "nonterminal candidate request required")
            public = public_only_context(context)
            request = PublicRootRequest.capture(public.public_materialization_state,
                public.observation, pending_transition=pending_transition)
            require(request.set_source_hash == self.contract.showdown_source_sha256,
                "root catalog differs from runtime binding")
            # Strip extraneous metadata on the actor observation as well as the
            # simultaneous opponent requests/history removed by the sanitizer.
            public = replace(public, observation=request.observation,
                requested_observations={public.player_id: request.observation})
            mask = tuple(request.observation.legal_action_mask)
            arm = self.configuration.arm
            evidence = {}
            if arm == "raw":
                before = self._evaluator.forwards
                legal, evaluation = self._evaluator(request.observation)
                action = _raw_action(legal, evaluation.priors)
                require(self._evaluator.forwards - before == 1, "raw requires exactly one real forward")
                evidence = dict(selector="deterministic_masked_argmax", model_forwards=1,
                    legal_actions=list(legal), priors=list(evaluation.priors), signed_value=evaluation.value)
            elif arm == "incumbent":
                from .policy_opponent_profile import validate_selection
                self._native.reset()
                if self.configuration.leaf == "raw_rollout":
                    self._native._config = replace(self._native._config, rollout_seed=selection_seed)
                self._prepare_incumbent(request)
                # Distinct root ID prevents statistics/fold carry-over from
                # other sampled roots from masquerading as fresh decisions.
                public = replace(public, battle_id="search-over-raw-root:" + root_id)
                before = self._decider._snapshot_stats(self._native)
                decision = self._native.select_action_with_context(public, rng=random.Random(selection_seed))
                after = self._decider._snapshot_stats(self._native)
                action = decision.action_index
                evidence = dict(root_action=f"action:{action}",
                    max_depth_reached=self._decider._changed_depth(before["depth_reached_histogram"], after["depth_reached_histogram"]),
                    fallbacks=after["fallback_decisions"]-before["fallback_decisions"],
                    prior_fallbacks=after["prior_fallbacks"]-before["prior_fallbacks"],
                    invalid_actions=int(type(action) is not int or not 0 <= action < len(mask) or not mask[action]),
                    total_iterations=after["total_iterations"]-before["total_iterations"],
                    model_evals=after["model_evals"]-before["model_evals"],
                    engine_mcts=dict(decision.metadata.get("engine_mcts", {})))
                if sum(mask) > 1:
                    validate_selection(evidence, arm="incumbent_mcts", mode="matched_deadline",
                        config=self._search_config, mask=mask, opponent_seed=selection_seed,
                        deadline_ms=round(self.configuration.seconds*1000), native_batch_guard_ms=64,
                        **({"model_leaf_override": "raw_policy_terminal"
                            if self.configuration.leaf == "raw_rollout" else self.configuration.leaf}
                            if self.configuration.leaf != "model" else {}))
                else:
                    require(not evidence["fallbacks"] and not evidence["prior_fallbacks"], "forced root fell back")
                if self.configuration.leaf != "model" and sum(mask) > 1:
                    from ..engine_search import require_model_leaf_witness
                    require_model_leaf_witness({"engine_mcts": evidence["engine_mcts"]},
                        model_leaf_override="raw_policy_terminal" if self.configuration.leaf == "raw_rollout"
                            else self.configuration.leaf)
            else:
                root = decision_state(request.observation, player=public.player_id)
                remaining = self.configuration.seconds - (time.perf_counter()-started)
                require(remaining > 0, "reference ceiling expired during public capture")
                measured = self._pool.search(self._diagnostic_request(request), root, battle_id="search-over-raw-root:"+root_id,
                    seed=selection_seed, deadline_seconds=remaining)
                validate_reference_work(measured)
                self._validate_diagnostic_work(measured)
                if self.configuration.leaf != "model":
                    from .search_over_raw_leaves import validate_leaf_work
                    validate_leaf_work(measured, self.configuration.leaf)
                encoded = measured.result.action
                require(type(encoded) is str and encoded.startswith("action:") and encoded[7:].isdigit(),
                    "invalid reference action encoding")
                action = int(encoded[7:])
                evidence = asdict(measured)
            require(type(action) is int and 0 <= action < len(mask) and mask[action], "illegal selected action")
            evidence = self._selection_evidence(evidence, request)
            elapsed = time.perf_counter()-started
            return dict(root_id=root_id, configuration_sha256=self.configuration.identity,
                runtime_sha256=self.runtime_sha256,
                action=action, status="SELECTED", elapsed_seconds=elapsed,
                nominal_search_seconds=self.configuration.seconds,
                exceeded_nominal_seconds=arm != "raw" and elapsed > self.configuration.seconds,
                evidence=evidence, independent_root_statistics=True, scientific_strength_evidence=False)
        except BaseException as error:
            self._poisoned = True
            self.last_failure = dict(root_id=root_id, error_type=type(error).__name__,
                status="UNCERTAIN_REFUSED", retry_authorized=False)
            from .policy_opponent_profile import refusal_diagnostic
            diagnostic = refusal_diagnostic(error)
            if diagnostic is not None:
                self.last_failure["native_diagnostic"] = diagnostic
            raise

    def close(self):
        if not self._closed:
            self._closed = True
            if self._pool is not None:
                self._pool.close()
            if self._decider is not None:
                self._decider.close()
