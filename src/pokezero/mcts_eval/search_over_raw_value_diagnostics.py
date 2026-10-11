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
import heapq
import math
import pickle
import time

from .paper_reference_parallel import PreparedDecision
from .paper_reference_showdown import ShowdownTrajectoryWorld, decision_state
from .search_over_raw import _raw_action, digest, require, rng_seed


@dataclass(frozen=True)
class ValueDiagnosticContract:
    sample_seed: int
    original_deadline_at: float
    leaves_per_worker: int = 2
    replicates: int = 8
    maximum_boundaries: int = 250
    maximum_snapshot_bytes: int = 8 * 1024 * 1024
    # A separate, explicitly requested engineering extension. Native invocations
    # are not reference workers; no equal-sampling or Showdown-fidelity claim.
    incumbent_sampling: str | None = None

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
        require(self.incumbent_sampling in (None, "per_native_invocation.v1"),
            "native sampling extension must be explicit")


def diagnostic_configurations(configs, contract):
    return [cfg for cfg in configs if cfg.arm == "reference" or
        (cfg.arm == "incumbent" and contract.incumbent_sampling is not None)]


def maximum_diagnostic_labels(cfg, contract):
    return (20 if cfg.arm == "reference" else 4) * contract.leaves_per_worker * 8


class NativeValueDiagnosticSession:
    """One owned serial incumbent decision; opaque frontiers never cross Python.

    Keep one K-event reservoir per actual native invocation (up to four), NOT
    per belief draw. Collapsed draws get no duplicate labels. All retained
    invocations, including cancelled rounds, remain in the diagnostic roster.
    This is engineering-only until searched-line Showdown fidelity is proved.
    """
    def __init__(self, contract, *, root_key, native_module=None):
        require(isinstance(contract, ValueDiagnosticContract)
            and contract.incumbent_sampling == "per_native_invocation.v1"
            and type(root_key) is str and bool(root_key) and len(root_key) % 2 == 0
            and all(c in "0123456789abcdef" for c in root_key), "explicit native diagnostic binding required")
        if native_module is None:
            import pokezero_search as native_module
        require(hasattr(native_module, "NativeVisitedValueBank"), "prospective native diagnostic build required")
        self.contract, self.root_key, self.module = contract, root_key, native_module
        self.invocations, self.closed = [], False

    def begin(self, record):
        require(not self.closed and len(self.invocations) < 4, "native diagnostic invocation cap; no retry")
        handle = self.module.NativeVisitedValueBank(sample_seed=self.contract.sample_seed,
            worker=len(self.invocations), root_information_key=self.root_key,
            original_deadline_at=self.contract.original_deadline_at,
            leaves_per_native_invocation=self.contract.leaves_per_worker,
            maximum_payload_bytes=self.contract.maximum_snapshot_bytes)
        row = dict(handle=handle, binding=dict(invocation=len(self.invocations), world_seed=record["seed"],
            root_native_state_sha256=hashlib.sha256(record["state_str"].encode()).hexdigest()))
        self.invocations.append(row)  # Consume slot before FFI; never retry failed work.
        return handle

    def complete(self, handle, raw_report):
        import json
        require(not self.closed and self.invocations and self.invocations[-1]["handle"] is handle
            and "selection_report_sha256" not in self.invocations[-1]["binding"], "native diagnostic completion drift")
        report = json.loads(raw_report)
        n = report["model_evals"]
        require(type(n) is int and n >= 1, "native diagnostic actual model rows required")
        self.invocations[-1]["binding"].update(selection_report_sha256=hashlib.sha256(raw_report.encode()).hexdigest(),
            model_evaluations_seen=n-1)

    def bindings(self):
        require(not self.closed and all("selection_report_sha256" in row["binding"] for row in self.invocations),
            "failed native diagnostic invocation cannot become a selection")
        return deepcopy([row["binding"] for row in self.invocations])

    def label(self):
        import json
        bindings = self.bindings()
        self.closed = True
        try:
            return [dict(invocation=index, binding=binding, evidence=json.loads(row["handle"].label()))
                for index, (row, binding) in enumerate(zip(self.invocations, bindings))]
        finally:
            for row in self.invocations:
                row["handle"].close()
            self.invocations.clear()

    def close(self):
        self.closed = True
        for row in self.invocations:
            row["handle"].close()
        self.invocations.clear()


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


def reconcile_collected_values(result, *, selected, contract, workers, information_key, arm="reference"):
    """Validate the complete selection-bound sanitized roster before publication.

    No missing workers/leaves/replicates may become complete-case calibration.
    A failed RPC has an unknown sampled denominator, not zero sampled leaves.
    This is descriptive event-weighted calibration, not independent trials.
    """
    require(isinstance(contract, ValueDiagnosticContract) and type(workers) is int and workers > 0,
        "explicit collector diagnostic contract required")
    if type(result) is dict and result.get("schema") == "pokezero.selected-native-visited-value.engineering.v1":
        require(arm == "incumbent", "native labels cannot be reference Showdown evidence")
        return _reconcile_native_values(result, selected=selected, contract=contract,
            workers=workers, information_key=information_key)
    require(arm == "reference", "incumbent requires native engineering evidence, not reference labels")
    require(type(result) is dict and set(result) == {"schema", "root_id", "selected_receipt_sha256",
        "configuration_sha256", "runtime_sha256", "workers", "labels_used_for_backup",
        "scientific_strength_evidence"} and result["schema"] == "pokezero.selected-visited-value.v1"
        and result["root_id"] == selected["root_id"]
        and result["selected_receipt_sha256"] == digest(selected)
        and result["configuration_sha256"] == selected["configuration_sha256"]
        and result["runtime_sha256"] == selected["runtime_sha256"]
        and result["labels_used_for_backup"] is False and result["scientific_strength_evidence"] is False,
        "diagnostic selected/runtime binding drift")
    rows = result["workers"]
    require(type(rows) in (list, tuple) and len(rows) == workers
        and all(type(row) is dict and set(row) == {"worker", "evidence"}
            and type(row["worker"]) is int and row["worker"] == index for index, row in enumerate(rows)),
        "diagnostic full worker roster required")
    leaves, seen = [], 0
    evidence_fields = {"schema", "worker", "root_information_key", "contract", "sampling_unit",
        "model_evaluations_seen", "sampling", "leaves", "instrumentation_can_change_selection",
        "labels_used_for_backup", "source_truth_transported", "scientific_strength_evidence",
        "label_seconds", "calibration"}
    for worker in rows:
        index, evidence = worker["worker"], worker["evidence"]
        require(type(evidence) is dict and set(evidence) == evidence_fields
            and evidence["schema"] == "pokezero.visited-value.v1"
            and type(evidence["worker"]) is int and evidence["worker"] == index
            and evidence["root_information_key"] == information_key
            and digest(evidence["contract"]) == digest(asdict(contract))
            and evidence["sampling_unit"] == "successful_model_evaluation_event"
            and evidence["sampling"] == "smallest_seeded_priorities_per_worker_decision"
            and evidence["instrumentation_can_change_selection"] is True
            and all(evidence[field] is False for field in ("labels_used_for_backup",
                "source_truth_transported", "scientific_strength_evidence"))
            and type(evidence["label_seconds"]) in (int, float)
            and math.isfinite(evidence["label_seconds"]) and evidence["label_seconds"] >= 0,
            "diagnostic worker binding/schema drift")
        n, sampled = evidence["model_evaluations_seen"], evidence["leaves"]
        require(type(n) is int and n >= 0 and type(sampled) is list
            and len(sampled) == min(n, contract.leaves_per_worker), "diagnostic sampled denominator drift")
        ordinals, priorities = set(), []
        for leaf in sampled:
            require(type(leaf) is dict and set(leaf) == {"evaluation_ordinal", "priority", "information_key",
                "snapshot_sha256", "model_signed_value", "subject", "labels"}, "private/malformed diagnostic leaf")
            ordinal = leaf["evaluation_ordinal"]
            require(type(ordinal) is int and 0 <= ordinal < n and ordinal not in ordinals
                and type(leaf["priority"]) is int and leaf["priority"] == rng_seed("visited-model-sample.v1",
                    contract.sample_seed, index, information_key, ordinal)
                and leaf["subject"] == "p1"
                and type(leaf["information_key"]) is str and bool(leaf["information_key"])
                and len(leaf["information_key"]) % 2 == 0
                and all(c in "0123456789abcdef" for c in leaf["information_key"])
                and type(leaf["snapshot_sha256"]) is str and len(leaf["snapshot_sha256"]) == 64
                and all(c in "0123456789abcdef" for c in leaf["snapshot_sha256"])
                and type(leaf["model_signed_value"]) in (int, float)
                and math.isfinite(leaf["model_signed_value"]) and -1 <= leaf["model_signed_value"] <= 1,
                "diagnostic actual prediction/state binding drift")
            ordinals.add(ordinal)
            priorities.append((leaf["priority"], ordinal))
            require(type(leaf["labels"]) is list and len(leaf["labels"]) == 8,
                "diagnostic full label roster required")
            for rep, label in enumerate(leaf["labels"]):
                require(type(label) is dict and set(label) == {"replicate", "status", "boundaries",
                    "signed_outcome", "continuation_policy"} and type(label["replicate"]) is int
                    and label["replicate"] == rep and label["continuation_policy"] == "raw_argmax_both_seats"
                    and type(label["boundaries"]) is int and 0 <= label["boundaries"] <= 250,
                    "diagnostic label roster/boundary drift")
        expected = heapq.nsmallest(contract.leaves_per_worker,
            ((rng_seed("visited-model-sample.v1", contract.sample_seed, index, information_key, ordinal), ordinal)
                for ordinal in range(n)))
        require(priorities == expected
            and digest(evidence["calibration"]) == digest(calibration_bounds(sampled)),
            "diagnostic calibration/priority drift")
        leaves.extend(sampled)
        seen += n
    calibration = calibration_bounds(leaves)
    return dict(schema="pokezero.collected-visited-value.v1", selection_sha256=digest(selected),
        workers=workers, model_evaluations_seen=seen, sampled_leaves=len(leaves),
        labels=8*len(leaves), unknown_labels=sum(row["unknown_labels"] for row in calibration),
        calibration=calibration, weighting="sampled_model_evaluation_events_not_independent_trials",
        mechanism_qualified=False, scientific_strength_evidence=False)


def _reconcile_native_values(result, *, selected, contract, workers, information_key):
    require(contract.incumbent_sampling == "per_native_invocation.v1" and workers == 1
        and set(result) == {"schema", "root_id", "selected_receipt_sha256", "configuration_sha256",
            "runtime_sha256", "native_invocations", "labels_used_for_backup", "scientific_strength_evidence"}
        and result["root_id"] == selected["root_id"] and result["selected_receipt_sha256"] == digest(selected)
        and result["configuration_sha256"] == selected["configuration_sha256"]
        and result["runtime_sha256"] == selected["runtime_sha256"]
        and result["labels_used_for_backup"] is False and result["scientific_strength_evidence"] is False,
        "native selected/runtime/extension binding drift")
    rows = result["native_invocations"]
    bindings = selected["evidence"]["visited_value_native_invocations"]
    require(type(rows) is list and len(rows) <= 4 and len(rows) == len(bindings),
        "native complete invocation roster required")
    work = selected["evidence"]
    require(type(work.get("model_evals")) is int and work["model_evals"] >= 0
        and type(work.get("visited_value_root_legal_actions")) is int
        and work["visited_value_root_legal_actions"] >= 1,
        "native independently recorded model work/root witness required")
    calls = work.get("engine_mcts", {}).get("time_budget", {}).get("native_invocations")
    require(calls is not None or not rows, "native actual invocation ledger missing")
    if calls is not None:
        require(type(calls) is list and len(calls) == len(rows)
            and all(type(call) is dict and call.get("status") == "completed"
                and call.get("world_seed") == binding.get("world_seed")
                for call, binding in zip(calls, bindings)), "native actual invocation ledger omission/order drift")
    if not rows:
        require(work["model_evals"] == 0 and work["visited_value_root_legal_actions"] == 1,
            "empty native roster requires an actual forced/no-native decision")
    fields = {"schema", "worker", "root_information_key", "sample_seed", "original_deadline_at",
        "root_native_state_sha256", "selection_report_sha256", "leaves_per_native_invocation", "replicates",
        "maximum_boundaries", "model_evaluations_seen", "sampling_unit", "sampling", "value_frame",
        "retained_payload_bytes", "maximum_payload_bytes", "payload_cap_is_not_rss_qualification", "leaves",
        "label_lossy_subcases", "label_engine", "searched_line_showdown_fidelity", "scientific_calibration_evidence",
        "scientific_strength_evidence", "labels_used_for_backup", "private_frontiers_exported",
        "instrumentation_can_change_selection"}
    leaves, seen = [], 0
    def sha(value):
        return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    for index, row in enumerate(rows):
        require(type(row) is dict and set(row) == {"invocation", "binding", "evidence"}
            and type(row["invocation"]) is int and row["invocation"] == index
            and row["binding"] == bindings[index], "native invocation selection binding drift")
        binding, evidence = row["binding"], row["evidence"]
        require(set(binding) == {"invocation", "world_seed", "root_native_state_sha256",
            "selection_report_sha256", "model_evaluations_seen"} and binding["invocation"] == index
            and type(binding["world_seed"]) is int and 0 <= binding["world_seed"] < 2**64
            and sha(binding["root_native_state_sha256"]) and sha(binding["selection_report_sha256"])
            and type(evidence) is dict and set(evidence) == fields
            and evidence["schema"] == "pokezero.native-visited-value.engineering.v1"
            and type(evidence["worker"]) is int and evidence["worker"] == index
            and evidence["root_information_key"] == information_key
            and evidence["sample_seed"] == contract.sample_seed
            and evidence["original_deadline_at"] == contract.original_deadline_at
            and all(evidence[key] == binding[key] for key in ("root_native_state_sha256",
                "selection_report_sha256", "model_evaluations_seen"))
            and evidence["leaves_per_native_invocation"] == contract.leaves_per_worker
            and evidence["replicates"] == 8 and evidence["maximum_boundaries"] == 250
            and evidence["maximum_payload_bytes"] == contract.maximum_snapshot_bytes
            and type(evidence["retained_payload_bytes"]) is int
            and 0 <= evidence["retained_payload_bytes"] <= contract.maximum_snapshot_bytes
            and evidence["sampling_unit"] == "successful_model_evaluation_event"
            and evidence["sampling"] == "smallest_seeded_priorities_per_native_invocation"
            and evidence["value_frame"] == "self_relative"
            and evidence["label_engine"] == "poke_engine_unqualified_against_showdown"
            and evidence["payload_cap_is_not_rss_qualification"] is True
            and evidence["instrumentation_can_change_selection"] is True
            and all(evidence[key] is False for key in ("searched_line_showdown_fidelity",
                "scientific_calibration_evidence", "scientific_strength_evidence", "labels_used_for_backup",
                "private_frontiers_exported")), "native actual prediction/engineering schema drift")
        lossy = evidence["label_lossy_subcases"]
        require(type(lossy) is dict and all(type(key) is str and type(count) is int and count >= 0
            for key, count in lossy.items()), "native label abort ledger drift")
        n, sampled = evidence["model_evaluations_seen"], evidence["leaves"]
        require(type(n) is int and n >= 0 and type(sampled) is list
            and len(sampled) == min(n, contract.leaves_per_worker), "native sampled denominator drift")
        priorities = []
        for leaf in sampled:
            require(type(leaf) is dict and set(leaf) == {"evaluation_ordinal", "priority", "native_state_sha256",
                "snapshot_sha256", "model_signed_value", "labels"}
                and type(leaf["evaluation_ordinal"]) is int and 0 <= leaf["evaluation_ordinal"] < n
                and type(leaf["priority"]) is int and sha(leaf["native_state_sha256"])
                and sha(leaf["snapshot_sha256"])
                and type(leaf["model_signed_value"]) in (int, float)
                and math.isfinite(leaf["model_signed_value"]) and -1 <= leaf["model_signed_value"] <= 1,
                "private/malformed native visited leaf")
            ordinal = leaf["evaluation_ordinal"]
            priorities.append((leaf["priority"], ordinal))
            require(type(leaf["labels"]) is list and len(leaf["labels"]) == 8,
                "native full label roster required")
            for rep, label in enumerate(leaf["labels"]):
                require(type(label) is dict and set(label) == {"replicate", "status", "signed_outcome",
                    "boundaries", "continuation_policy", "seed"} and type(label["replicate"]) is int
                    and label["replicate"] == rep and type(label["boundaries"]) is int
                    and 0 <= label["boundaries"] <= 250
                    and label["continuation_policy"] == "raw_argmax_both_seats"
                    and type(label["seed"]) is int and label["seed"] == rng_seed("native-visited-label.v1",
                        contract.sample_seed, index, information_key, ordinal, rep), "native full label stream drift")
        expected = heapq.nsmallest(contract.leaves_per_worker,
            ((rng_seed("visited-model-sample.v1", contract.sample_seed, index, information_key, ordinal), ordinal)
                for ordinal in range(n)))
        require(priorities == expected, "native outcome-independent sample drift")
        calibration_bounds(sampled)  # Validates all predictions/statuses/eight-label denominators.
        leaves.extend(sampled)
        seen += n
    require(work["model_evals"] == seen + len(rows), "native model-evaluation ledger omission drift")
    calibration = calibration_bounds(leaves)
    return dict(schema="pokezero.collected-native-visited-value.engineering.v1", selection_sha256=digest(selected),
        workers=1, native_invocations=len(rows), model_evaluations_seen=seen, sampled_leaves=len(leaves),
        labels=8*len(leaves), unknown_labels=sum(row["unknown_labels"] for row in calibration),
        calibration=calibration, weighting="sampled_events_per_native_invocation_not_reference_workers",
        label_engine="poke_engine_unqualified_against_showdown", searched_line_showdown_fidelity=False,
        scientific_calibration_evidence=False, mechanism_qualified=False, scientific_strength_evidence=False)


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
