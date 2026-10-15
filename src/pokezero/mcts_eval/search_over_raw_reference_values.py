"""Prospective worker-local, independently read reference visited-state labels.

Private sampled snapshots never leave their original reference worker. Each
leaf owns a fresh bridge; each of its eight raw/raw cells owns a fresh clone.
The parent can bank sanitized labels only after normal worker-generation close.
This is not source collection, runtime qualification, or full-tree cost proof.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import time

from .search_over_raw import _run_continuation_cell, digest, require, rng_seed
from .search_over_raw_outcomes import (ContinuationCapture, OwnedContinuationLifecycle,
    RAW_LABEL_SCHEMA, RAW_LABEL_POLICY, snapshot_identity, validate_observed_outcome, _sha)
from .search_over_raw_outcome_run import ContinuationProcessOwner, verify_outcome_runtime
from .search_over_raw_value_diagnostics import VisitedValueBank, _ValueRuntime, calibration_bounds
from .search_over_raw_world_inputs import _state_payload
from .paper_reference_parallel import owned_process_identity


@dataclass(frozen=True)
class ReferenceValueLabelRecipe:
    sample_seed: int
    runtime: dict
    leaves_per_worker: int = 2
    replicates: int = 8
    maximum_boundaries: int = 250
    maximum_snapshot_bytes: int = 8 * 1024 * 1024

    def __post_init__(self):
        require(type(self.sample_seed) is int and 0 <= self.sample_seed < 2**64
            and type(self.leaves_per_worker) is int and 1 <= self.leaves_per_worker <= 8
            and type(self.replicates) is int and self.replicates == 8
            and type(self.maximum_boundaries) is int and self.maximum_boundaries == 250
            and type(self.maximum_snapshot_bytes) is int and 0 < self.maximum_snapshot_bytes <= 8*1024*1024,
            "explicit fixed-work reference label recipe required")
        object.__setattr__(self, "runtime", deepcopy(self.runtime))
        verify_outcome_runtime(self.runtime)

    def receipt(self):
        return dict(schema="pokezero.reference-owned-value-recipe.v1",
            **{k:v for k,v in asdict(self).items() if k != "runtime"}, runtime_sha256=digest(self.runtime),
            policy=RAW_LABEL_POLICY, search_decision_ceiling=None, retries_permitted=False,
            deployment_eligible=False, collection_authorized=False)


def generation_inputs(descriptor, *, worker, live=True):
    require(type(descriptor) is dict and set(descriptor) == {"path", "sha256"},
        "actual parent-published generation descriptor required")
    path = Path(descriptor["path"])
    require(path.is_absolute() and path == path.resolve()
        and not any(p.is_symlink() for p in (path,*path.parents)), "original canonical generation path required")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == descriptor["sha256"], "original reference generation bytes changed")
    row = json.loads(raw)
    require(type(row["pid"]) is int and type(row["controller_pid"]) is int
        and type(row["worker_index"]) is int and row["worker_index"] == worker
        and row["protocol"] == "active-generations-v1" and row["group"] == row["pid"]
        and path.name == f"group-{row['pid']}-{row['generation']}.json", "reference generation identity drift")
    if live:
        require(row["pid"] == os.getpid() == os.getpgrp() and row["controller_pid"] == os.getppid()
            and row["birth_identity"] == owned_process_identity(os.getpid())
            and os.path.samefile(path, path.parent / "active" / path.name),
            "original live isolated reference generation required")
    return path, row


class RawLabelCapture(ContinuationCapture):
    def export(self):
        bank = super().export()
        bank["schema"] = RAW_LABEL_SCHEMA
        return bank


def run_owned_leaf(item, *, recipe, descriptor, worker, decision, event_id, env_factory=None, progress_sink=None):
    """Actual fresh clone/process owner followed by separate transport reader."""
    from ..local_showdown import LocalShowdownEnv
    from .search_over_raw_supervision import _save_new
    path, generation = generation_inputs(descriptor, worker=worker)
    snapshot = item["snapshot"]
    before = hashlib.sha256(pickle.dumps(snapshot, protocol=5)).hexdigest()
    require(before == item["snapshot_sha256"], "actual visited snapshot changed")
    config = item["env"].config
    runtime_sha = verify_outcome_runtime(recipe.runtime, config)
    evaluator = item["evaluator"]
    require(env_factory is None or getattr(env_factory, "synthetic_process_fixture", False) is True,
        "production label env cannot be substituted")
    if env_factory is None:
        from .paper_reference_showdown import ChampionEvaluator
        require(type(evaluator) is ChampionEvaluator
            and evaluator.policy.weights_sha256 == recipe.runtime["files"][recipe.runtime["checkpoint_path"]],
            "actual label evaluator differs from frozen champion")
    env = LocalShowdownEnv(config) if env_factory is None else env_factory(config)
    directory = path.parent / ("labels-"+generation["generation"]) / (f"decision-{decision}") / event_id
    directory.mkdir(parents=True, mode=0o700, exist_ok=False)
    root_binding = dict(root_id=event_id, source_seed=recipe.sample_seed,
        generation_sha256=descriptor["sha256"], worker=worker, decision=decision,
        **snapshot_identity(snapshot), observation_sha256=digest(_state_payload(item["observation"])),
        captured_snapshot_sha256=before,
        information_key=item["state"].key.hex(), model_signed_value=item["model_signed_value"])
    _save_new(directory / "claim.json", dict(root_binding=root_binding, recipe=recipe.receipt()))
    observer = RawLabelCapture(root_binding, runtime_sha)
    lifecycle = OwnedContinuationLifecycle(env, expected_snapshot=snapshot, root_binding=root_binding)
    ownership = dict(controller_pid=os.getpid(), registration_sha256=generation["registration_sha256"])
    owner = ContinuationProcessOwner(directory=directory, ownership=ownership, runtime=recipe.runtime, config=config)
    primary = None
    rows = []
    boundaries = 0
    with owner.installed(env):
        try:
            with observer.installed(env):
                for rep in range(8):
                    law = dict(root_id=event_id, action=0, replicate=rep, subject=item["subject"],
                        namespace="reference-owned-visited-raw.v1", max_boundaries=250, policy=RAW_LABEL_POLICY)
                    lifecycle.prepare(snapshot, law)
                    try:
                        observer.begin_cell(snapshot, **law)
                        def tick(count):
                            if progress_sink is not None and count % 32 == 0:
                                progress_sink(boundaries+count, rep)
                        result = _run_continuation_cell(env, snapshot, item["subject"], 0, rep,
                            lambda obs:(lambda p:(p[0],p[1].priors))(evaluator(obs)),
                            law["namespace"], event_id, 250, observer, lifecycle, raw_from_start=True,
                            boundary_sink=tick)
                        lifecycle.release()
                    except BaseException as failure:
                        lifecycle.abort(failure)
                        raise
                    row = dict(root_id=event_id, action=0, replicate=rep, **result)
                    observer.finish_cell(row)
                    rows.append(row)
                    boundaries += result["boundaries"]
                    if progress_sink is not None:
                        progress_sink(boundaries, rep+1)
            bank = observer.export()
            owner.seal(bank)
        except BaseException as error:
            primary = error
            raise
        finally:
            if env._process is not None:
                try:
                    env.close()
                except BaseException as error:
                    if primary is None:
                        raise
                    primary.add_note("Owned visited label cleanup failed: "+type(error).__name__)
    # Separate reader joins actual wait4, independent process transport and
    # exact immutable private prediction inputs, never a supplied success bit.
    require(owner.phase == "closed" and owner.reaped is not None and owner.reaped[0] == 0
        and owner.process.returncode == 0 and not owner.wait_ownership_lost
        and env._process is None and env._battle_token is None
        and before == hashlib.sha256(pickle.dumps(snapshot, protocol=5)).hexdigest(),
        "normally reaped label bridge and unchanged sampled snapshot required")
    generation_inputs(descriptor, worker=worker)
    closed = json.loads((directory / "bridge-close.json").read_bytes())
    require(closed["normal_close"] is True and closed["forced_cleanup"] is False
        and closed["pumps_drained"] is True and closed["bank_sha256"] == digest(bank)
        and closed["transport_sha256"] == digest(owner.records)
        and owner.records[-2:] == [dict(kind="command",payload=dict(type="close")),
            dict(kind="event",payload=dict(type="closed"))], "actual label bridge close evidence drift")
    contract = dict(root_binding=root_binding, root_id=event_id, subject=item["subject"],
        namespace="reference-owned-visited-raw.v1", max_boundaries=250, replicates=8,
        selection_seed=recipe.sample_seed, policy=RAW_LABEL_POLICY)
    labels = []
    for row, cell in zip(rows, bank["cells"]):
        witness = validate_observed_outcome(row, contract, cell, expected_snapshot=snapshot, runtime_sha256=runtime_sha)
        require(witness == digest(row), "independent visited terminal/cap readback missing")
        first = next(r["payload"] for r in cell["records"] if r["kind"] == "boundary")
        require(first["policies"][item["subject"]]["observation_sha256"] == root_binding["observation_sha256"],
            "cloned raw label observation differs from actual prediction input")
        labels.append(dict(replicate=row["replicate"], status="COMPLETE" if row["status"] == "COMPLETE" else "CAPPED_UNCERTAIN",
            signed_outcome=row["signed_outcome"], boundaries=row["boundaries"], continuation_policy="raw_argmax_both_seats"))
    _save_new(directory / "labels-readout.json", dict(schema="pokezero.reference-owned-label-readout.v1",
        root_binding=root_binding, bank_sha256=digest(bank), transport_sha256=digest(owner.records),
        runtime_sha256=runtime_sha, subject=item["subject"], labels=labels))
    proof = dict(directory=str(directory), event_id=event_id, bank_sha256=digest(bank),
        generation_sha256=descriptor["sha256"], bridge_pid=owner.process.pid,
        labels_sha256=digest(labels), files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.iterdir())})
    require(set(proof["files"]) == {"claim.json","bridge-launch.json","bridge-close.json","bridge-transport.json","labels-readout.json"},
        "original label receipt roster drift")
    return labels, proof


class OwnedVisitedValueBank(VisitedValueBank):
    def __init__(self, recipe, *, worker, root_key, descriptor, decision, env_factory=None):
        require(type(recipe) is ReferenceValueLabelRecipe and type(root_key) is bytes and root_key,
            "owned reference value recipe required")
        generation_inputs(descriptor, worker=worker)
        self.contract, self.worker, self.root_key, self.clock = recipe, worker, root_key, time.perf_counter
        self.descriptor, self.decision, self.env_factory = deepcopy(descriptor), decision, env_factory
        self.seen, self.rows, self.bytes, self.closed = 0, {}, 0, False

    def _capture_observation(self, world, state):
        observation = world._observations.get(state.key)
        require(observation is not None, "original actual model observation required")
        return observation  # No second observe() can redefine the prediction.

    def label(self, progress_sink=None):
        require(not self.closed, "owned visited labels cannot be retried")
        self.closed = True
        began, leaves, proofs, boundaries = self.clock(), [], [], 0
        try:
            for (priority, ordinal), item in sorted(self.rows.items()):
                event_id = digest(dict(generation=self.descriptor["sha256"], decision=self.decision,
                    worker=self.worker, root_key=self.root_key.hex(), ordinal=ordinal,
                    snapshot=item["snapshot_sha256"], observation=digest(_state_payload(item["observation"]))))
                def report(count, accounted):
                    if progress_sink is not None:
                        progress_sink(dict(schema="pokezero.visited-value-progress.v1", boundaries=boundaries+count,
                            labels_accounted=8*len(leaves)+accounted))
                labels, proof = run_owned_leaf(item, recipe=self.contract, descriptor=self.descriptor,
                    worker=self.worker, decision=self.decision, event_id=event_id, env_factory=self.env_factory,
                    progress_sink=report)
                leaves.append(dict(evaluation_ordinal=ordinal, priority=priority,
                    information_key=item["state"].key.hex(), snapshot_sha256=item["snapshot_sha256"],
                    model_signed_value=item["model_signed_value"], subject=item["subject"], labels=labels))
                proofs.append(proof)
                boundaries += sum(row["boundaries"] for row in labels)
        finally:
            self.rows.clear()
            self.bytes = 0
        result = dict(schema="pokezero.reference-owned-visited-values.v1", worker=self.worker,
            decision=self.decision, generation=deepcopy(self.descriptor), recipe=self.contract.receipt(),
            root_information_key=self.root_key.hex(), model_evaluations_seen=self.seen,
            leaves=leaves, proofs=proofs, calibration=calibration_bounds(leaves),
            label_seconds=self.clock()-began, private_snapshots_exported=False, labels_used_for_backup=False)
        path,generation=generation_inputs(self.descriptor,worker=self.worker)
        directory=path.parent/("labels-"+generation["generation"])/f"decision-{self.decision}"
        directory.mkdir(parents=True,mode=0o700,exist_ok=True)
        from .search_over_raw_supervision import _save_new
        _save_new(directory/"worker-readout.json",result)
        return result


@dataclass(frozen=True)
class ReferenceOwnedValueFactory:
    base: object
    recipe: ReferenceValueLabelRecipe
    requires_owned_value_generation = True

    def __post_init__(self):
        from .paper_reference_runtime import ShowdownWorkerFactory
        base=self.base
        seen=set()
        while not isinstance(base,ShowdownWorkerFactory):
            require(id(base) not in seen and len(seen)<8 and hasattr(base,"base"),
                "actual reference label factory chain required")
            seen.add(id(base));base=base.base
        require(type(self.recipe) is ReferenceValueLabelRecipe
            and self.recipe.runtime["checkpoint_path"] == base.checkpoint
            and self.recipe.runtime["files"][base.checkpoint] == base.checkpoint_sha256,
            "owned label factory differs from actual reference champion")

    def __call__(self, index):
        return OwnedValueRuntime(self.base(index), self.recipe, index)


class OwnedValueRuntime(_ValueRuntime):
    def bind_owned_value_generation(self, descriptor):
        generation_inputs(descriptor, worker=self.index)
        require(not hasattr(self,"generation"), "reference label generation cannot be rebound")
        self.generation, self.decision = deepcopy(descriptor), 0

    def _new_bank(self, root_key):
        require(hasattr(self,"generation"), "actual parent generation handshake required")
        self.decision += 1
        return OwnedVisitedValueBank(self.contract, worker=self.index, root_key=root_key,
            descriptor=self.generation, decision=self.decision)


def read_completed_reference_values(pool, selected, values, recipe):
    """Parent readback joins exact normal worker exits and immutable label files."""
    require(pool._closed and not pool._poisoned and pool.cleanup_evidence["status"] == "CLEAN_OBSERVED_SCOPE"
        and selected is pool._last_result and type(recipe) is ReferenceValueLabelRecipe
        and type(values) is tuple and len(values) == pool.workers,
        "original normally closed diagnostic pool/selection/full roster required")
    return read_reference_value_receipts(values,recipe,worker_pids=pool.worker_pids,
        generation_paths=pool._ownership_receipts,decision_id=selected.decision_id)


def read_reference_value_receipts(values, recipe, *, worker_pids, generation_paths, decision_id):
    """Durable metadata reader; caller must separately prove enclosing exit."""
    require(type(recipe) is ReferenceValueLabelRecipe and type(values) is tuple
        and len(values) == len(worker_pids) and set(generation_paths) == set(range(len(worker_pids)))
        and type(decision_id) is int and decision_id > 0, "exact externally owned label worker roster required")
    leaves = []
    verify_outcome_runtime(recipe.runtime)
    for index, worker in enumerate(values):
        require(worker["worker"] == index, "owned value worker order drift")
        evidence = worker["evidence"]
        require(type(evidence) is dict and set(evidence)=={"schema","worker","decision","generation","recipe",
            "root_information_key","model_evaluations_seen","leaves","proofs","calibration","label_seconds",
            "private_snapshots_exported","labels_used_for_backup"}
            and evidence["schema"] == "pokezero.reference-owned-visited-values.v1"
            and evidence["worker"] == index and evidence["decision"] == decision_id
            and evidence["recipe"] == recipe.receipt() and evidence["private_snapshots_exported"] is False
            and evidence["labels_used_for_backup"] is False, "owned value recipe/selection drift")
        path, launch = generation_inputs(evidence["generation"], worker=index, live=False)
        require(path == generation_paths[index] and launch["pid"] == worker_pids[index],
            "label belongs to another worker generation")
        readout_dir=path.parent/("labels-"+launch["generation"])/f"decision-{decision_id}"
        require(readout_dir == readout_dir.resolve() and not any(p.is_symlink() for p in (readout_dir,*readout_dir.parents))
            and not (readout_dir/"worker-readout.json").is_symlink()
            and json.loads((readout_dir/"worker-readout.json").read_bytes())==evidence
            and type(evidence["label_seconds"]) in (int,float) and math.isfinite(evidence["label_seconds"])
            and evidence["label_seconds"]>=0, "immutable original worker diagnostic readout changed")
        closed = json.loads(path.with_name("closed-"+path.name).read_bytes())
        require(closed["launch_sha256"] == evidence["generation"]["sha256"]
            and type(closed["pid"]) is int and closed["pid"] == launch["pid"]
            and closed["controller_pid"] == launch["controller_pid"] and closed["generation"] == launch["generation"]
            and type(closed["worker_exit_code"]) is int and closed["worker_exit_code"] == 0
            and closed["cleanup_protocol"] == "live_leader_ack_before_reap.v2"
            and closed["clean_acknowledgment"] is True and closed["forced_cleanup"] is False
            and closed["runtime_close_failed"] is False
            and type(closed["cleanup_exception_count"]) is int and closed["cleanup_exception_count"] == 0
            and closed["group_live"] is False, "normal original reference label worker closure required")
        n = evidence["model_evaluations_seen"]
        require(type(n) is int and n >= 0 and len(evidence["leaves"]) == min(n,recipe.leaves_per_worker)
            and len(evidence["proofs"]) == len(evidence["leaves"]), "full sampled label denominator required")
        expected = sorted((rng_seed("visited-model-sample.v1",recipe.sample_seed,index,evidence["root_information_key"],o),o)
            for o in range(n))[:recipe.leaves_per_worker]
        require([(l["priority"],l["evaluation_ordinal"]) for l in evidence["leaves"]] == expected,
            "outcome-independent original reservoir required")
        for leaf, proof in zip(evidence["leaves"], evidence["proofs"]):
            require(type(leaf) is dict and set(leaf)=={"evaluation_ordinal","priority","information_key",
                "snapshot_sha256","model_signed_value","subject","labels"}
                and type(proof) is dict and _sha(proof["event_id"]) and _sha(leaf["snapshot_sha256"])
                and type(leaf["labels"]) is list and len(leaf["labels"])==8,
                "exact sanitized leaf/full label roster required")
            directory = path.parent / ("labels-"+launch["generation"]) / f"decision-{decision_id}" / proof["event_id"]
            require(str(directory) == proof["directory"] and directory == directory.resolve()
                and not any(p.is_symlink() for p in (directory,*directory.parents))
                and set(proof["files"]) == {"claim.json","bridge-launch.json","bridge-close.json","bridge-transport.json","labels-readout.json"}
                and {p.name for p in directory.iterdir()} == set(proof["files"])
                and all((directory/name).is_file() and not (directory/name).is_symlink() for name in proof["files"])
                and all(hashlib.sha256((directory/name).read_bytes()).hexdigest() == sha for name,sha in proof["files"].items())
                and proof["generation_sha256"] == evidence["generation"]["sha256"]
                and proof["labels_sha256"] == digest(leaf["labels"]), "original independent label receipt bytes changed")
            bridge = json.loads((directory/"bridge-close.json").read_bytes())
            bridge_launch = json.loads((directory/"bridge-launch.json").read_bytes())
            transport = json.loads((directory/"bridge-transport.json").read_bytes())["records"]
            require(bridge["launch_sha256"] == digest(bridge_launch)
                and bridge_launch["parent_pid"] == launch["pid"] and bridge_launch["group"] == launch["pid"]
                and bridge_launch["pid"] == proof["bridge_pid"] and bridge_launch["argv"] == recipe.runtime["argv"]
                and bridge_launch["runtime_sha256"] == bridge["runtime_sha256"] == digest(recipe.runtime)
                and bridge_launch["binding_sha256"] == bridge["binding_sha256"] == launch["registration_sha256"]
                and type(bridge_launch["birth_identity"]) is str and bridge_launch["birth_identity"]
                and bridge["transport_sha256"] == digest(transport), "actual bridge launch/runtime/transport join drift")
            claim = json.loads((directory/"claim.json").read_bytes())
            readout = json.loads((directory/"labels-readout.json").read_bytes())
            require(readout["schema"] == "pokezero.reference-owned-label-readout.v1"
                and readout["root_binding"] == claim["root_binding"]
                and readout["labels"] == leaf["labels"] and readout["subject"] == leaf["subject"]
                and readout["bank_sha256"] == bridge["bank_sha256"] == proof["bank_sha256"]
                and readout["transport_sha256"] == bridge["transport_sha256"]
                and readout["runtime_sha256"] == bridge["runtime_sha256"] == digest(recipe.runtime),
                "immutable independently checked label rows differ from returned labels")
            root = claim["root_binding"]
            require(claim["recipe"] == recipe.receipt() and root["root_id"] == proof["event_id"]
                and root["generation_sha256"] == evidence["generation"]["sha256"]
                and root["worker"] == index and root["decision"] == decision_id
                and root["captured_snapshot_sha256"] == leaf["snapshot_sha256"]
                and root["information_key"] == leaf["information_key"]
                and root["model_signed_value"] == leaf["model_signed_value"]
                and proof["event_id"] == digest(dict(generation=evidence["generation"]["sha256"],
                    decision=decision_id,worker=index,root_key=evidence["root_information_key"],
                    ordinal=leaf["evaluation_ordinal"],snapshot=leaf["snapshot_sha256"],
                    observation=root["observation_sha256"])), "original prediction/label event binding drift")
            require(bridge["pid"] == proof["bridge_pid"] and bridge["parent_pid"] == launch["pid"]
                and type(bridge["exit_code"]) is int and bridge["exit_code"] == 0
                and bridge["normal_close"] is True and bridge["forced_cleanup"] is False
                and bridge["reaped"] is True and bridge["pumps_drained"] is True
                and bridge["bank_sha256"] == proof["bank_sha256"], "original label bridge close/reap missing")
        require(evidence["calibration"] == calibration_bounds(evidence["leaves"]), "worker calibration drift")
        leaves.extend(evidence["leaves"])
    complete=sum(l["status"]=="COMPLETE" for leaf in leaves for l in leaf["labels"])
    return dict(schema="pokezero.completed-reference-owned-values.v1", recipe=recipe.receipt(), workers=len(worker_pids),
        sampled_leaves=len(leaves), labels=8*len(leaves), calibration=calibration_bounds(leaves),
        complete_labels=complete,uncertain_labels=8*len(leaves)-complete,
        evidence_sha256=digest(values), independently_validated_label_evidence=True,
        independent_terminal_labels=complete>0, runtime_qualified=False,
        resource_cost_complete=False, scientific_collection_authorized=False)


from .search_over_raw_fixed_work_run import FixedWorkRunValidator


class ReferenceValueRunValidator(FixedWorkRunValidator):
    """Independent completed enclosing run + original label receipt readback.

    No Process handles, historical PID queries, pickle loading or launches.
    External job/config/model bindings remain the caller's prospective inputs.
    """
    def __init__(self, *, recipe, **kwargs):
        require(type(recipe) is ReferenceValueLabelRecipe and kwargs["configuration"].arm == "reference"
            and kwargs["configuration"].leaf in {"model","hp_fraction"}
            and recipe.runtime["checkpoint_path"] == kwargs["checkpoint_path"]
            and recipe.runtime["files"][kwargs["checkpoint_path"]] == kwargs["checkpoint_sha256"],
            "completed reference labels require the same prospectively frozen champion")
        self.recipe=recipe
        super().__init__(**kwargs)

    def _inputs(self):
        return digest(dict(base=super()._inputs(),recipe=self.recipe.receipt()))

    def _selected_result(self,result):
        return result

    def __call__(self, selected):
        witness=super().__call__(selected)  # Actual enclosing normal terminal, not an in-worker bit.
        before=digest(selected)
        bundle=selected.get("owned_reference_value_diagnostic")
        original=deepcopy(selected)
        original.pop("owned_reference_value_diagnostic",None)
        require(type(bundle) is dict and set(bundle)=={"schema","root_id","selected_receipt_sha256","values","summary"}
            and bundle["schema"] == "pokezero.selected-owned-reference-values.v1"
            and bundle["root_id"] == self.root_id
            and bundle["selected_receipt_sha256"] == digest(original), "original pre-label selected receipt drift")
        paths={}
        for path in (self.directory/"owned-groups").glob("group-*.json"):
            row=json.loads(path.read_bytes())
            index=row.get("worker_index")
            require(type(index) is int and 0<=index<20 and index not in paths, "exact label-generation worker indices required")
            paths[index]=path
        values=tuple(bundle["values"])
        require(len(values)==20 and all(v["evidence"]["root_information_key"]==self.consumed["information_key"]
            and all(l["subject"]==self.consumed["subject"] for l in v["evidence"]["leaves"]) for v in values),
            "original fixed-work root/subject diagnostic binding drift")
        summary=read_reference_value_receipts(values,self.recipe,worker_pids=selected["evidence"]["worker_pids"],
            generation_paths=paths,decision_id=selected["evidence"]["decision_id"])
        require(summary==bundle["summary"] and digest(selected)==before and self._inputs()==self.frozen,
            "completed owned label summary/input drift")
        self._files()
        self.last_validation["reference_value_labels"]=summary
        return witness
