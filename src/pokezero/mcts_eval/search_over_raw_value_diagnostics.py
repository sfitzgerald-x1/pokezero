"""Opt-in reference visited-leaf predictions and post-selection raw labels.

The sampling unit is a model-evaluation event, not a distinct information set.
Keep the K smallest deterministic random priorities per worker/decision. Neither
predictions, terminal outcomes nor evaluation latency enters the priority. Full
hypothetical snapshots stay worker-local and are cleared after labeling. Capture
overhead IS inside the search clock; this does not qualify an uninstrumented arm.
Labels are eight new raw-argmax continuations, never tree backups or reused leaf
prices. A boundary cap or the original producer deadline leaves a null label.
This exploratory diagnostic cannot establish a value-head cause or arm strength.
"""
from copy import deepcopy
from dataclasses import dataclass, asdict
import hashlib
import math
import pickle
import time

from .paper_reference_parallel import PreparedDecision
from .paper_reference_showdown import ShowdownTrajectoryWorld, decision_state
from .search_over_raw import _raw_action, require, rng_seed


@dataclass(frozen=True)
class ValueDiagnosticContract:
    sample_seed: int
    original_deadline_at: float
    leaves_per_worker: int = 2
    replicates: int = 8
    maximum_boundaries: int = 250
    maximum_snapshot_bytes: int = 8 * 1024 * 1024

    def __post_init__(self):
        require(type(self.sample_seed) is int and 0 <= self.sample_seed < 2**64,
            "diagnostic sample seed must be explicit")
        require(type(self.original_deadline_at) in (int, float)
            and math.isfinite(self.original_deadline_at), "original producer deadline required")
        require(type(self.leaves_per_worker) is int and 1 <= self.leaves_per_worker <= 8
            and type(self.replicates) is int and self.replicates == 8
            and type(self.maximum_boundaries) is int and self.maximum_boundaries == 250
            and type(self.maximum_snapshot_bytes) is int and 0 < self.maximum_snapshot_bytes <= 8*1024*1024,
            "unregistered diagnostic sampling/label/memory contract")


class VisitedValueBank:
    """One worker, one decision, outcome-independent bounded private reservoir."""
    def __init__(self, contract, *, worker, root_key, clock=time.perf_counter):
        require(isinstance(contract, ValueDiagnosticContract) and type(worker) is int and worker >= 0
            and type(root_key) is bytes and bool(root_key), "invalid visited-value bank binding")
        self.contract, self.worker, self.root_key, self.clock = contract, worker, root_key, clock
        self.seen, self.rows, self.bytes, self.closed = 0, {}, 0, False

    def observe(self, world, state, model):
        require(not self.closed and isinstance(world, ShowdownTrajectoryWorld)
            and world.env._search_snapshot_permitted is True, "owned hypothetical visited state required")
        require(type(model.value) in (int, float) and math.isfinite(model.value) and -1 <= model.value <= 1,
            "invalid actual model prediction")
        ordinal = self.seen
        self.seen += 1
        priority = rng_seed("visited-model-sample.v1", self.contract.sample_seed,
            self.worker, self.root_key.hex(), ordinal)
        if len(self.rows) == self.contract.leaves_per_worker and (priority, ordinal) >= max(self.rows):
            return
        # This is the ACTUAL sampled search world, not the source-game snapshot.
        require(world.env.terminal() is None and world.subject in world.env.requested_players()
            and decision_state(world.env.observe(world.subject), player=world.subject) == state,
            "prediction and visited-state binding differ")
        snapshot = deepcopy(world.env.snapshot())
        payload = pickle.dumps(snapshot, protocol=5)
        size = len(payload)
        victim = max(self.rows) if len(self.rows) == self.contract.leaves_per_worker else None
        total = self.bytes + size - (self.rows[victim]["size"] if victim is not None else 0)
        require(total <= self.contract.maximum_snapshot_bytes, "visited-state private memory cap reached")
        if victim is not None:
            del self.rows[victim]
        self.bytes = total
        self.rows[(priority, ordinal)] = dict(snapshot=snapshot, size=size, env=world.env,
            evaluator=world.evaluator, state=state, subject=world.subject,
            snapshot_sha256=hashlib.sha256(payload).hexdigest(), model_signed_value=model.value)

    def label(self, progress_sink=None):
        require(not self.closed, "visited-value labels cannot be retried")
        self.closed = True  # Consume before any actual continuation.
        began = self.clock()
        result = dict(schema="pokezero.visited-value.v1", worker=self.worker,
            root_information_key=self.root_key.hex(), contract=asdict(self.contract),
            sampling_unit="successful_model_evaluation_event", model_evaluations_seen=self.seen,
            sampling="smallest_seeded_priorities_per_worker_decision", leaves=[],
            instrumentation_can_change_selection=True, labels_used_for_backup=False,
            source_truth_transported=False, scientific_strength_evidence=False)
        boundaries, accounted = 0, 0
        def report():
            if progress_sink is not None:
                progress_sink(dict(schema="pokezero.visited-value-progress.v1",
                    boundaries=boundaries, labels_accounted=accounted))
        try:
            for (priority, ordinal), item in sorted(self.rows.items()):
                env, subject = item["env"], item["subject"]
                require(env._search_snapshot_permitted is True, "labeling lost owned hypothetical shell")
                require(hashlib.sha256(pickle.dumps(item["snapshot"], protocol=5)).hexdigest()
                    == item["snapshot_sha256"], "captured visited snapshot mutated")
                row = dict(evaluation_ordinal=ordinal, priority=priority,
                    information_key=item["state"].key.hex(), snapshot_sha256=item["snapshot_sha256"],
                    model_signed_value=item["model_signed_value"], subject=subject, labels=[])
                result["leaves"].append(row)
                current = None
                try:
                    for replicate in range(self.contract.replicates):
                        label = dict(replicate=replicate, status="DEADLINE_UNCERTAIN", boundaries=0,
                            signed_outcome=None, continuation_policy="raw_argmax_both_seats")
                        row["labels"].append(label)
                        if self.clock() >= self.contract.original_deadline_at:
                            accounted += 1
                            report()
                            continue
                        if current is None:
                            current = env.snapshot()
                        env.restore(deepcopy(item["snapshot"]))
                        require(env.terminal() is None and subject in env.requested_players()
                            and decision_state(env.observe(subject), player=subject) == item["state"],
                            "label restore differs from actual visited prediction")
                        for step in range(self.contract.maximum_boundaries + 1):
                            if self.clock() >= self.contract.original_deadline_at:
                                break
                            terminal = env.terminal()
                            if terminal is not None:
                                require(terminal.winner in {None, "p1", "p2"}, "invalid diagnostic terminal")
                                if terminal.capped:
                                    label["status"] = "CAPPED_UNCERTAIN"
                                else:
                                    label.update(status="COMPLETE", signed_outcome=0. if terminal.winner is None
                                        else 1. if terminal.winner == subject else -1.)
                                break
                            if step == self.contract.maximum_boundaries:
                                label["status"] = "CAPPED_UNCERTAIN"
                                break
                            requested = env.requested_players()
                            require(requested and set(requested) <= {"p1", "p2"}, "invalid raw label boundary")
                            actions = {}
                            for seat in requested:
                                legal, evaluation = item["evaluator"](env.observe(seat))
                                actions[seat] = _raw_action(legal, evaluation.priors)
                            env.reseed_simulator_rng(rng_seed("visited-model-label.v1",
                                self.contract.sample_seed, self.worker, self.root_key.hex(), ordinal, replicate, step))
                            env.step(actions)
                            label["boundaries"] = step + 1
                            boundaries += 1
                            if label["boundaries"] % 32 == 0:
                                report()
                        accounted += 1
                        report()
                finally:
                    if current is not None:
                        env.restore(current)
        finally:
            self.rows.clear()
            self.bytes = 0
        result["label_seconds"] = self.clock() - began
        result["calibration"] = calibration_bounds(result["leaves"])
        return result

    def close(self):
        self.closed = True
        self.rows.clear()
        self.bytes = 0


def calibration_bounds(leaves):
    """Ten fixed bins, including all eight labels and all unknown outcomes.

    Squared-error bounds are for win-score targets {0, .5, 1}, not a binary
    Brier claim. These descriptive leaf-event bins are not independent trials.
    """
    bins = [dict(bin=index, leaves=0, labels=0, unknown_labels=0, prediction_sum=0.,
        score_lower_sum=0., score_upper_sum=0., error_lower_sum=0., error_upper_sum=0.) for index in range(10)]
    for leaf in leaves:
        p = (leaf["model_signed_value"] + 1.)/2.
        require(math.isfinite(p) and 0 <= p <= 1 and len(leaf["labels"]) == 8,
            "invalid diagnostic calibration roster")
        row = bins[min(int(p*10), 9)]
        row["leaves"] += 1
        row["prediction_sum"] += p
        for rep, label in enumerate(leaf["labels"]):
            require(label["replicate"] == rep, "diagnostic replicate drift")
            value = label["signed_outcome"]
            if label["status"] == "COMPLETE":
                require(type(value) in (int, float) and value in (-1, 0, 1), "invalid diagnostic score")
                scores = [(value + 1.)/2.]
            else:
                require(label["status"] in {"CAPPED_UNCERTAIN", "DEADLINE_UNCERTAIN"} and value is None,
                    "uncertain diagnostic label must remain null")
                scores = [0., .5, 1.]
                row["unknown_labels"] += 1
            row["labels"] += 1
            row["score_lower_sum"] += min(scores)
            row["score_upper_sum"] += max(scores)
            row["error_lower_sum"] += min((p-score)**2 for score in scores)
            row["error_upper_sum"] += max((p-score)**2 for score in scores)
    for row in bins:
        n = row["labels"]
        row["mean_prediction"] = row["prediction_sum"]/row["leaves"] if n else None
        row["win_score_interval"] = [row["score_lower_sum"]/n, row["score_upper_sum"]/n] if n else None
        row["squared_error_interval"] = [row["error_lower_sum"]/n, row["error_upper_sum"]/n] if n else None
    return bins


class _ObservedWorld:
    def __init__(self, world, bank):
        self.world, self.bank = world, bank

    def frame(self):
        return self.world.frame()

    def advance(self, *args):
        return self.world.advance(*args)

    def close(self):
        self.world.close()

    def evaluate(self, state):
        model = self.world.evaluate(state)
        self.bank.observe(self.world, state, model)
        return model


@dataclass(frozen=True)
class ReferenceValueDiagnosticFactory:
    base: object
    contract: ValueDiagnosticContract

    def __post_init__(self):
        from .paper_reference_runtime import ShowdownWorkerFactory
        from .search_over_raw_oracle import OracleWorkerFactory
        from .search_over_raw_leaves import ReferenceLeafWorkerFactory
        from .search_over_raw_belief_diagnostics import BeliefDiagnosticWorkerFactory
        require(isinstance(self.base, (ShowdownWorkerFactory, OracleWorkerFactory, ReferenceLeafWorkerFactory,
            BeliefDiagnosticWorkerFactory))
            and isinstance(self.contract, ValueDiagnosticContract), "explicit reference diagnostic factory required")

    def __call__(self, index):
        return _ValueRuntime(self.base(index), self.contract, index)


class _ValueRuntime:
    def __init__(self, base, contract, index):
        self.base, self.contract, self.index, self.bank = base, contract, index, None

    def prepare(self, request):
        from .search_over_raw_leaves import ReferenceLeafWorld
        if self.bank is not None:
            self.bank.close()
        prepared = self.base.prepare(request)
        self.bank = bank = VisitedValueBank(self.contract, worker=self.index, root_key=prepared.root.key)

        def sample_world(rng):
            world = prepared.sample_world(rng)
            if isinstance(world, ReferenceLeafWorld):
                # Capture the genuine model prediction before alternative pricing.
                world.observe_model = lambda state, model: bank.observe(world.world, state, model)
                return world
            require(isinstance(world, ShowdownTrajectoryWorld), "genuine reference world required")
            return _ObservedWorld(world, bank)

        return PreparedDecision(prepared.root, prepared.evaluate_root, sample_world,
            prepared.evidence, prepared.set_sampling_deadline, bank.label)

    def close(self):
        try:
            if self.bank is not None:
                self.bank.close()
        finally:
            self.base.close()
