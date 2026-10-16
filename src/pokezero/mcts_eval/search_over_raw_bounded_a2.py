"""Scott's bounded exploration, separate from staged/holdout admission.

Uses the current source, actor-timed selector and continuation owners. No
qualification flags, historical ledger, failed attempt or holdout are upgraded.
The only launch authority is the prospectively bound steer and entry review.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import resource
import time

from .search_over_raw import SearchConfiguration, digest, require
from .search_over_raw_ledger import _write_new

SCHEMA = "pokezero.search-over-raw.bounded-a2-exploration.v1"
SPEC_SCHEMA = SCHEMA + ".entry"
A1_SCHEMA = "pokezero.search-over-raw.bounded-a1-exploration.v1"
WALL_SECONDS = 28800


def experiment(value):
    require(value.get("schema") in (SCHEMA,A1_SCHEMA), "explicit bounded A1 or A2 schema required")
    return "A1" if value["schema"] == A1_SCHEMA else "A2"


def configurations(stage="A2"):
    require(stage in ("A1","A2"), "only the separately authorized A1 or A2 pair is allowed")
    if stage == "A1":
        return (SearchConfiguration("raw",seconds=1.),) + tuple(
            SearchConfiguration("incumbent",belief,"model",1.,1) for belief in ("public","oracle"))
    return (SearchConfiguration("raw", seconds=1.),) + tuple(
        SearchConfiguration("incumbent", "public", leaf, 1., 1)
        for leaf in ("model", "hp_fraction"))


def contract(namespace, seeds, excluded_seeds, authority, *, stage="A2", authority_addendum=None):
    """Data only: caller supplies a prospectively frozen fresh seed roster."""
    import uuid
    require(type(namespace) is str and str(uuid.UUID(namespace)) == namespace,
        "canonical fresh exploration namespace required")
    require(type(seeds) is list and len(seeds) == 32 and len(set(seeds)) == 32
        and all(type(s) is int and 0 <= s < 2**32 for s in seeds), "exact 32 source seeds required")
    require(type(excluded_seeds) is list and excluded_seeds == sorted(set(excluded_seeds))
        and excluded_seeds and all(type(s) is int and 0 <= s < 2**32 for s in excluded_seeds)
        and not set(seeds) & set(excluded_seeds), "fresh, exposure-disjoint source roster required")
    require(type(authority) is dict and set(authority) == {"path", "sha256"}
        and Path(authority["path"]).is_absolute() and type(authority["sha256"]) is str
        and len(authority["sha256"]) == 64, "explicit bound steer required")
    slots = [dict(root_id=f"exploration:{seed}:{slot}", source_seed=seed, root_slot=slot)
        for i, seed in enumerate(seeds) for slot in range(7 if i < 8 else 6)]
    result = dict(schema=A1_SCHEMA if stage == "A1" else SCHEMA, namespace=namespace,
        panels={"exploration":dict(seeds=deepcopy(seeds), root_slots=slots)},
        excluded_seeds=deepcopy(excluded_seeds), authority=deepcopy(authority),
        configurations=[asdict(c) for c in configurations(stage)],
        candidate_seat="p1", exclude_opening_requests=True,
        source_policy="raw_argmax_both_seats", continuation_replicates=8,
        continuation_policy="raw_argmax_after_initial_sampled_opponent_reply",
        source_root_rule="seed-derived priorities over completed public request catalog; no replacements",
        max_source_boundaries=250, max_continuation_boundaries=250,
        attempt_wall_seconds=WALL_SECONDS, retries_permitted=False,
        collection_authorized=True, execution_ready=False, runtime_qualified=False,
        holdout_authorized=False, phase_b_authorized=False)
    if authority_addendum is not None:
        require(type(authority_addendum) is dict and set(authority_addendum) == {"path","sha256"}
            and Path(authority_addendum["path"]).is_absolute()
            and type(authority_addendum["sha256"]) is str and len(authority_addendum["sha256"]) == 64,
            "explicit unchanged steer addendum required")
        result["authority_addendum"] = deepcopy(authority_addendum)
    return result


def validate_contract(value):
    expected = contract(value["namespace"], value["panels"]["exploration"]["seeds"],
        value["excluded_seeds"], value["authority"],stage=experiment(value),
        authority_addendum=value.get("authority_addendum"))
    require(digest(value) == digest(expected), "bounded A2 contract drift; no extra stages or changed denominator")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_recipe(cfg, checkpoint, stage):
    from .search_over_raw_actor_timed_run import actor_timed_runtime_recipe, RUNTIME_KEY
    if stage == "A2":return actor_timed_runtime_recipe(cfg,checkpoint,compact_native_atomic_work=True)
    # A1 uses the existing original public/oracle native timed routes on BOTH
    # sides, with the same legacy atomic encoding and no actor-only handoff.
    # Do not silently compare the new annotated public observer to an older
    # unannotated oracle observer, or invent sampled-opponent annotations.
    public = SearchConfiguration("incumbent","public","model",1.,1)
    value = actor_timed_runtime_recipe(public,checkpoint)
    value.pop(RUNTIME_KEY)
    value["configuration"] = asdict(cfg)
    if cfg.belief == "oracle":
        from .search_over_raw_native_oracle import NATIVE_ORACLE_CAPTURE
        from .search_over_raw_construction import SCHEMA as CONSTRUCTION_SCHEMA, construction_source_hashes
        value["native_team_oracle_observation"] = deepcopy(NATIVE_ORACLE_CAPTURE)
        value["native_oracle_construction"] = dict(schema=CONSTRUCTION_SCHEMA,source_hashes=construction_source_hashes(),
            private_controller_only=True,instrumentation_can_change_deadlines=True,runtime_qualified=False)
        from .search_over_raw_native_oracle_run import _runtime
    else:
        from .search_over_raw_timed_run import _runtime
    _runtime(cfg,value,digest(value))
    return value


def prepare_entry(directory, *, checkpoint_path, checkpoint_sha256, showdown_root,
        model_path, tables_path, authority_path, exposure_files, stage="A2", authority_addendum_path=None):
    """Freeze one entry without launching a source or claiming qualification.

    This consumes no scientific attempt. The resulting exact spec still needs
    Astra's one prospective entry review before the create-only run can start.
    """
    import secrets
    import sys
    import uuid
    import poke_engine.poke_engine as engine
    import pokezero_search.pokezero_search as native
    from ..collection import env_config_with_policy_spec_masks
    from ..local_showdown import LocalShowdownConfig
    from ..randbat import load_gen3_randbat_source_cached
    from .resolver import resolve_checkpoint_contract
    from .search_over_raw import ENGINEERING_EXCLUDED_SEEDS
    from .search_over_raw_world_inputs import _state_payload
    from .search_over_raw_outcome_run import outcome_runtime_binding
    from .search_over_raw_fixed_work_run import fixed_work_reader_source_hashes
    from .search_over_raw_actor_timed_run import actor_timed_runtime_recipe
    from .search_over_raw_raw_run import raw_runtime
    require(exposure_files, "explicit prior exposure registrations required")
    excluded = set(ENGINEERING_EXCLUDED_SEEDS) | set(range(4711,4736)) | {2026101024,2026101025}
    def visit(value, key=""):
        if type(value) is dict:
            for k,v in value.items():visit(v,k)
        elif type(value) is list:
            if key in {"seeds","source_seeds","excluded_seeds","engineering_excluded_seeds"}:
                excluded.update(v for v in value if type(v) is int and 0 <= v < 2**32)
            for v in value:visit(v,key)
        elif key in {"seed","source_seed","fixture_seed"} and type(value) is int and 0 <= value < 2**32:
            excluded.add(value)
    inventory = {}
    for supplied in sorted(set(str(Path(p).resolve()) for p in exposure_files)):
        visit(json.loads(Path(supplied).read_bytes()))
        inventory[supplied] = _sha(supplied)
    seeds = []
    while len(seeds) < 32:
        seed = secrets.randbits(32)
        if seed not in excluded and seed not in seeds:seeds.append(seed)
    checkpoint_path,showdown_root,model_path,tables_path,authority_path = (
        str(Path(p).resolve()) for p in (checkpoint_path,showdown_root,model_path,tables_path,authority_path))
    addendum = None if authority_addendum_path is None else dict(
        path=str(Path(authority_addendum_path).resolve()),sha256=_sha(authority_addendum_path))
    c = contract(str(uuid.uuid4()),seeds,sorted(excluded),dict(path=authority_path,sha256=_sha(authority_path)),
        stage=stage,authority_addendum=addendum)
    source = load_gen3_randbat_source_cached(showdown_root)
    checkpoint = resolve_checkpoint_contract(checkpoint_path,expected_sha256=checkpoint_sha256,
        model_device="cpu",showdown_root=showdown_root,showdown_source_sha256=source.metadata.source_hash,
        model_path=model_path,tables_path=tables_path)
    config = env_config_with_policy_spec_masks(LocalShowdownConfig(showdown_root=Path(showdown_root),
        set_belief_source=True),["neural:"+checkpoint_path],context="Scott bounded "+stage+" exploration")
    extras = [model_path,tables_path,str(Path(engine.__file__).resolve()),str(Path(native.__file__).resolve())]
    runtime = outcome_runtime_binding(config,checkpoint_path=checkpoint_path,extra_files=extras)
    files = {**runtime["files"],**fixed_work_reader_source_hashes(),
        str(Path(sys.executable).resolve()):_sha(Path(sys.executable).resolve())}
    directory = Path(directory).resolve()
    authority_files = {authority_path:_sha(authority_path)}
    if addendum is not None:authority_files[addendum["path"]] = addendum["sha256"]
    spec = dict(schema=c["schema"]+".entry",contract=c,output_directory=str(directory/"run"),
        source_config=_state_payload(config),continuation_config=_state_payload(config),
        checkpoint_contract=_state_payload(checkpoint),source_runtime=runtime,continuation_runtime=deepcopy(runtime),
        selector_files=files,raw_runtime=raw_runtime(configurations(stage)[0],checkpoint_sha256,source.metadata.source_hash),
        recipes={cfg.identity:runtime_recipe(cfg,checkpoint,stage)
            for cfg in configurations(stage)[1:]}, exposure_inventory=inventory,
        input_hashes={**files,**inventory,**authority_files})
    directory.mkdir(parents=True,exist_ok=False,mode=0o700)
    path = directory/"entry-spec.json"
    _write_new(path,spec)
    read_spec(path)
    return dict(entry_spec=str(path),entry_sha256=_sha(path),source_seeds=seeds,
        source_slots=200,exposure_registration_count=len(inventory),excluded_seed_count=len(excluded),
        worker_launched=False,review_required=True)


def read_spec(path, *, expected_sha=None):
    from .search_over_raw_exploration_entry import _decode, _json
    from .search_over_raw_actor_timed_run import actor_timed_runtime_recipe
    from .search_over_raw_raw_run import raw_runtime
    from .search_over_raw_outcome_run import verify_outcome_runtime
    from .search_over_raw_source_run import source_recipe
    raw = Path(path).read_bytes()
    require(expected_sha is None or hashlib.sha256(raw).hexdigest() == expected_sha, "entry bytes changed")
    spec = _json(raw)
    require(set(spec) == {"schema", "contract", "output_directory", "source_config", "checkpoint_contract",
        "source_runtime", "continuation_config", "continuation_runtime", "selector_files", "raw_runtime",
        "recipes", "input_hashes", "exposure_inventory"} and spec["schema"] == spec["contract"]["schema"]+".entry",
        "exact bounded exploration entry required")
    validate_contract(spec["contract"])
    stage = experiment(spec["contract"])
    require(str(Path(spec["output_directory"]).resolve()) == spec["output_directory"], "canonical output path required")
    checkpoint = _decode(spec["checkpoint_contract"])
    config, continuation = (_decode(spec[k]) for k in ("source_config", "continuation_config"))
    source_recipe(config, checkpoint, spec["source_runtime"], capture_actor_history=True)
    verify_outcome_runtime(spec["continuation_runtime"], continuation)
    require(digest(spec["source_config"]) == digest(spec["continuation_config"])
        and spec["continuation_runtime"]["checkpoint_path"] == checkpoint.checkpoint_path
        and spec["continuation_runtime"]["files"][checkpoint.checkpoint_path] == checkpoint.checkpoint_sha256,
        "source and continuations must retain the original champion and encoding")
    require(digest(spec["raw_runtime"]) == digest(raw_runtime(configurations(stage)[0],
        checkpoint.checkpoint_sha256, checkpoint.showdown_source_sha256)), "shared raw recipe changed")
    expected = {c.identity:runtime_recipe(c,checkpoint,stage) for c in configurations(stage)[1:]}
    require(digest(spec["recipes"]) == digest(expected), "exact prospective two-selector recipe required")
    from .search_over_raw_fixed_work_run import fixed_work_reader_source_hashes
    files = {**spec["source_runtime"]["files"], **spec["continuation_runtime"]["files"],
        **fixed_work_reader_source_hashes(), **spec["selector_files"],
        spec["contract"]["authority"]["path"]:spec["contract"]["authority"]["sha256"]}
    require(type(spec["exposure_inventory"]) is dict and spec["exposure_inventory"], "bound exposure inventory required")
    files.update(spec["exposure_inventory"])
    if "authority_addendum" in spec["contract"]:
        amendment = spec["contract"]["authority_addendum"]
        files[amendment["path"]] = amendment["sha256"]
    require(digest(files) == digest(spec["input_hashes"])
        and all(_sha(p) == sha for p, sha in files.items()), "entry code, exposure or runtime bytes changed")
    return spec, config, continuation, checkpoint


def _cpu():
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + children.ru_utime + children.ru_stime


def _pins(directory):
    return {str(p.resolve()):_sha(p) for p in Path(directory).rglob("*") if p.is_file()}


def continuation_counts(evidence, outcomes, stage="A2"):
    """Count fixed selector aliases, retaining the shared action identity."""
    cells = {(r["action"],r["replicate"]):r for r in outcomes}
    result = {}
    for cfg in configurations(stage):
        name = "raw" if cfg.arm == "raw" else cfg.identity
        counts = {k:0 for k in ("COMPLETE","CAPPED","REFUSED","UNSTARTED","UNVALIDATED")}
        action = evidence["actions"].get(name)
        for rep in range(8):
            cell = cells.get((action,rep)) if action is not None else None
            status = "UNSTARTED" if cell is None else cell["status"]
            if status == "COMPLETE" and evidence["validated_outcome_sha256"].get(f"{action}:{rep}") is None:
                status = "UNVALIDATED"
            counts[status] += 1
        result[name] = counts
    return result


def collect_source(spec, config, checkpoint, seed, directory, guard, *, root_capture=None,
                   capture_actor_history=True):
    """Same source policy and independent bridge reader as the staged route."""
    from .search_over_raw_source_run import (SourceProcessOwner, _production_components,
        read_source_owner, source_recipe, _policies_identity)
    from .search_over_raw_source_actor_history import SourceActorHistoryCapture
    from .search_over_raw_source import AuditedRawPolicy, collect_raw_source
    from .search_over_raw_archive import SealedSourceArchive, retained_archive_root_identity
    from .search_over_raw_nested_ownership import current_scope
    from ..trajectory import trajectory_to_dict
    scope = current_scope()
    require(scope is not None and scope.pid == os.getpid(), "original externally owned auditor required")
    directory.mkdir(parents=True, exist_ok=False)
    c = spec["contract"]
    require(type(capture_actor_history) is bool,"explicit source actor-history scope required")
    _write_new(directory/"claim.json", dict(schema="pokezero.search-over-raw.source-attempt.v1",
        contract_sha256=digest(c), source_seed=seed, source_sha256=None,
        recipe_sha256=digest(source_recipe(config,checkpoint,spec["source_runtime"],capture_actor_history=capture_actor_history)),
        runtime_sha256=digest(spec["source_runtime"]), source_review_sha256=digest(c["authority"]),
        auditor_binding_sha256=scope.binding, auditor_pid=scope.pid, synthetic_process_fixture=False,
        retry_authorized=False, holdout_authorized=False))
    env = None
    primary_error = None
    archive = SealedSourceArchive()
    try:
        env, raw = _production_components(config,checkpoint)
        history = SourceActorHistoryCapture(config) if capture_actor_history else None
        def capture(context, action):
            archive.capture_public(context, action)
            if root_capture is not None:
                root_capture(env, archive, context, action, history)
        policies = {seat:AuditedRawPolicy(raw[seat], checkpoint_sha256=checkpoint.checkpoint_sha256,
            public_context_sink=capture if seat == "p1" else None,
            actor_history=history if seat == "p1" else None) for seat in ("p1","p2")}
        identity = _policies_identity(policies)
        owner = SourceProcessOwner(source_seed=seed, actor_history=history, directory=directory,
            ownership=dict(controller_pid=scope.pid,registration_sha256=scope.binding,
                directory=str(scope.directory/"owned-groups")), runtime=spec["source_runtime"], config=config)
        driver = {}
        def accept(result):
            counts = {seat:sum(s.player_id == seat for s in result.trajectory.steps) for seat in policies}
            require(all(counts[s] == policies[s].verified_decisions for s in policies), "source policy counter drift")
            driver.update(schema="pokezero.search-over-raw.source-driver.v1",
                decision_boundaries=result.decision_round_count, decisions_by_seat=counts,
                trajectory_sha256=digest(trajectory_to_dict(result.trajectory)), terminal_capped=result.terminal.capped)
        with owner.installed(env):
            guard()
            source = collect_raw_source(c,panel="exploration",source_seed=seed,env=env,policies=policies,
                max_decision_rounds=250,sealed_pre_step_sink=archive.capture_private,_trusted_result_sink=accept)
            require(identity == _policies_identity(policies) and driver, "source policy drift")
            owner.seal(source)
            env.close()
        driver["source_sha256"] = digest(source)
        _write_new(directory/"source.json",source)
        _write_new(directory/"driver.json",driver)
        readout = read_source_owner(directory,source,spec["source_runtime"],c)
        require(readout["source_pipeline_validated"], "source pipeline not validated")
        frozen = {r["root_id"]:retained_archive_root_identity(archive,r["source_request_index"])
            for r in source["roots"]}
        guard()
        return archive, source, frozen
    except BaseException as error:
        primary_error = error
        try:
            archive.close()
        except BaseException as cleanup_error:
            raise error from cleanup_error
        raise
    finally:
        if env is not None and env._process is not None:
            try:
                env.close()
            except BaseException as cleanup_error:
                if primary_error is not None:
                    raise primary_error from cleanup_error
                raise


def select(spec, config, checkpoint, archive, source, root, cfg, directory, guard):
    """Actual current actor or raw job, then its original independent reader."""
    from .search_over_raw_archive import prepare_archived_raw_job
    from .search_over_raw_raw_run import RawSelectionRunValidator, supervise_raw_selection
    from .search_over_raw_actor_timed_run import (prepare_archived_actor_timed_job,
        ActorAnnotatedTimedRunValidator, supervise_actor_timed)
    from .search_over_raw_world_inputs import _state_payload
    stage = experiment(spec["contract"])
    runtime = deepcopy(spec["raw_runtime"] if cfg.arm == "raw" else spec["recipes"][cfg.identity])
    catalog = None
    if cfg.belief == "oracle":
        from ..randbat import load_gen3_randbat_source_cached
        from .search_over_raw_archive import capture_archived_team_oracle
        catalog = load_gen3_randbat_source_cached(checkpoint.showdown_root)
        oracle,_ = capture_archived_team_oracle(archive,source,contract=spec["contract"],
            root_id=root["root_id"],set_source=catalog)
        runtime["team_oracle"] = oracle.receipt()
    kwargs = dict(contract=spec["contract"],root_id=root["root_id"],configuration=cfg,
        checkpoint_contract=checkpoint,showdown_root=checkpoint.showdown_root,runtime_configuration=runtime)
    if cfg.arm == "raw":
        job, binding = prepare_archived_raw_job(archive,source,**kwargs)
        supervise, reader_cls = supervise_raw_selection, RawSelectionRunValidator
        extra = {}
    elif cfg.belief == "oracle":
        from .search_over_raw_archive import prepare_archived_native_oracle_timed_job
        from .search_over_raw_native_oracle_run import NativeOracleTimedRunValidator,supervise_native_oracle_timed
        job, projection = prepare_archived_native_oracle_timed_job(archive,source,**kwargs,set_source=catalog)
        binding = projection["archived_root_binding"]
        supervise,reader_cls = supervise_native_oracle_timed,NativeOracleTimedRunValidator
        extra = dict(oracle_receipt=job.expected_oracle_receipt,public_request=binding["public_request_witness"],
            expected_override_sha256=job.expected_override_sha256,
            expected_reconstruction_catalog_sha256=job.expected_reconstruction_catalog_sha256)
    elif stage == "A1":
        from .search_over_raw_archive import prepare_archived_native_timed_job
        from .search_over_raw_timed_run import NativeTimedRunValidator,supervise_native_timed
        job,binding = prepare_archived_native_timed_job(archive,source,**kwargs)
        supervise,reader_cls = supervise_native_timed,NativeTimedRunValidator
        extra = {}
    else:
        job, binding = prepare_archived_actor_timed_job(archive,source,**kwargs,
            source_config=config,source_config_sha256=digest(_state_payload(config)))
        supervise, reader_cls = supervise_actor_timed, ActorAnnotatedTimedRunValidator
        extra = dict(annotation_sha256=job.annotation_boundary._original_sha)
    payload_sha = hashlib.sha256(pickle.dumps(job)).hexdigest()
    attempt = digest(dict(entry=digest(spec),root=root["root_id"],configuration=cfg.identity,job_sha256=payload_sha))
    terminal = supervise(job,directory=directory,binding_sha256=attempt,verify_admission=guard)
    require(terminal["supervisor_exit_code"] == 0 and terminal["worker_exit_code"] == 0,
        "selector failed; no retry")
    witness = binding["public_request_witness"]
    reader = reader_cls(directory=directory,job_sha256=payload_sha,binding_sha256=attempt,
        configuration=cfg,root_id=job.root_id,statistics_root_id=job.statistics_root_id,
        selection_seed=job.selection_seed,runtime_configuration=runtime,runtime_sha256=digest(runtime),
        consumed_binding=witness["consumed_binding"], legal_action_mask=binding["public_evidence_binding"]["legal_action_mask"],
        decision_round=None if cfg.arm == "raw" else job.context.decision_round_index,
        file_hashes=spec["selector_files"], checkpoint_path=checkpoint.checkpoint_path,
        checkpoint_sha256=checkpoint.checkpoint_sha256,showdown_root=checkpoint.showdown_root,
        public_request_sha256=witness["public_request_sha256"],
        **extra)
    selected = terminal["result"]
    require(reader(selected) == digest(selected), "selector independent readback failed")
    guard()
    return selected, reader


def measure_root(spec, config, continuation, checkpoint, archive, source, root, directory, guard):
    from .search_over_raw_actor_timed_run import ActorInputUnavailable
    from .search_over_raw_root_plan import ArchivedRootEvaluationPlan
    from .search_over_raw_outcome_run import supervise_outcome_run, OwningRunValidator
    from .search_over_raw_archive import retained_archive_root_identity
    started, cpu = time.monotonic(), _cpu()
    directory.mkdir(parents=True, exist_ok=False)
    stage = experiment(spec["contract"])
    cfgs, selected, readers, unavailable = [], {}, {}, {}
    for cfg in configurations(stage):
        name = "raw" if cfg.arm == "raw" else cfg.identity
        try:
            selected[name], readers[name] = select(spec,config,checkpoint,archive,source,root,cfg,
                directory/name,guard)
            cfgs.append(cfg)
        except ActorInputUnavailable as error:
            unavailable[name] = error.reason
    extra = {}
    if any(cfg.belief == "oracle" for cfg in cfgs):
        from ..randbat import load_gen3_randbat_source_cached
        extra["oracle_set_source"] = load_gen3_randbat_source_cached(checkpoint.showdown_root)
    plan = ArchivedRootEvaluationPlan(archive,source,contract=spec["contract"],root_id=root["root_id"],
        configurations=tuple(cfgs),selections=selected,selector_readers=readers,**extra)
    accepted = plan.revalidate()
    outcomes = []
    if any(name != "raw" for name in accepted["root_evidence"]["actions"]):
        job,_ = plan.prepare_continuation_job(config=continuation,runtime=spec["continuation_runtime"])
        attempt = plan.continuation_attempt_binding(job)
        # Create-only entire root reserves every unique action once. No donors,
        # later retries, missing-cell repair, or cross-run bank import exists.
        _write_new(directory/"continuation-reservation.json",dict(binding_sha256=attempt,
            cells=accepted["continuation_cells"],retry_authorized=False))
        def admit():
            guard();plan.verify_prepared_job(job);return True
        terminal = supervise_outcome_run(job,directory=directory/"outcomes",binding_sha256=attempt,
            verify_admission=admit,validate_result=lambda value:True)
        require(terminal["supervisor_exit_code"] == 0 and terminal["worker_exit_code"] == 0,
            "continuation failed; no retry")
        reader = OwningRunValidator(directory=directory/"outcomes",job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),
            binding_sha256=attempt,root_binding=job.root_binding,expected_snapshot=job.snapshot,runtime=job.runtime)
        accepted = plan.accept_continuation_result(job,terminal["result"],owning_reader=reader)
        outcomes = terminal["result"]["outcomes"]["outcomes"]
    evidence = accepted["root_evidence"]
    guard()
    _write_new(directory/"accepted.json",accepted)
    row = dict(root_id=root["root_id"],source_seed=source["source_seed"],
        entry_sha256=digest(spec),source_sha256=digest(source),
        archive_sha256=retained_archive_root_identity(archive,root["source_request_index"]),
        root_intervals={cfg.identity:evidence["root_intervals"].get(cfg.identity,[-1.,1.])
            for cfg in configurations(stage)[1:]}, input_unavailable=unavailable,
        accepted_evidence=accepted,source_pipeline_validated=True,
        source_receipt_files=_pins(Path(spec["output_directory"])/"sources"/str(source["source_seed"])),
        continuation_cell_counts=continuation_counts(evidence,outcomes,stage),
        receipt_files=_pins(directory),elapsed_wall_seconds=time.monotonic()-started,
        cpu_seconds_including_reaped_children=_cpu()-cpu,scientific_strength_claim=False)
    if stage == "A1":row["oracle_minus_public_interval"] = direct_belief_interval(evidence,outcomes)
    return row


def direct_belief_interval(evidence, outcomes):
    """Shared raw cancels: compare oracle to public using the same action bank."""
    from .search_over_raw_screening import paired_root_interval
    public,oracle = configurations("A1")[1:]
    actions = evidence["actions"]
    if public.identity not in actions or oracle.identity not in actions:return [-1.,1.]
    cells = {(row["action"],row["replicate"]):dict(status=row["status"],
        validated=evidence["validated_outcome_sha256"].get(f'{row["action"]}:{row["replicate"]}') is not None,
        score=(row["signed_outcome"]+1)/2 if row["signed_outcome"] is not None else None) for row in outcomes}
    return list(paired_root_interval(actions[oracle.identity],actions[public.identity],cells))


@dataclass(frozen=True)
class BoundedA2Job:
    spec_path: str
    spec_sha256: str
    def __call__(self, ownership, phase):
        phase("startup")
        spec,config,continuation,checkpoint = read_spec(self.spec_path,expected_sha=self.spec_sha256)
        status = "BOUNDED_"+experiment(spec["contract"])+"_RECORDED"
        output = Path(spec["output_directory"])
        began = time.monotonic()
        def guard():
            read_spec(self.spec_path,expected_sha=self.spec_sha256)
            return True
        phase("selection")
        consecutive_unavailable = 0
        try:
            for seed in spec["contract"]["panels"]["exploration"]["seeds"]:
                if time.monotonic()-began >= WALL_SECONDS: break
                archive = None
                try:
                    archive,source,frozen = collect_source(spec,config,checkpoint,seed,output/"sources"/str(seed),guard)
                    from .search_over_raw_archive import retained_archive_root_identity
                    for root in source["roots"]:
                        if time.monotonic()-began >= WALL_SECONDS: break
                        require(retained_archive_root_identity(archive,root["source_request_index"])
                            == frozen[root["root_id"]], "original source archive changed")
                        row = measure_root(spec,config,continuation,checkpoint,archive,source,root,
                            output/"roots"/digest(root["root_id"]),guard)
                        _write_new(output/"completed-roots"/(digest(root["root_id"])+".json"),row)
                        print(json.dumps(dict(root_id=root["root_id"],status="ROOT_RECORDED")),flush=True)
                        consecutive_unavailable = consecutive_unavailable+1 if row["input_unavailable"] else 0
                        if consecutive_unavailable >= 3:
                            _write_new(output/"input-stop.json",dict(root_id=root["root_id"],
                                reasons=row["input_unavailable"],consecutive_unavailable=3,retry_authorized=False))
                            return dict(status=status,holdout_authorized=False)
                finally:
                    if archive is not None: archive.close()
            return dict(status=status,holdout_authorized=False)
        except BaseException as error:
            import traceback
            _write_new(output/"failure.json",dict(error_type=type(error).__name__,message=str(error),
                frames=[dict(file=f.filename,line=f.lineno,function=f.name) for f in traceback.extract_tb(error.__traceback__)],
                retry_authorized=False))
            raise
        finally:
            phase("cleanup")


def summarize(spec, rows, terminal):
    """Fixed 200-root bounds; same resampled seed indices for both leaves."""
    from .search_over_raw_screening import cluster_intervals, ScreeningDesign
    slots = spec["contract"]["panels"]["exploration"]["root_slots"]
    seeds = spec["contract"]["panels"]["exploration"]["seeds"]
    by_id = {s["root_id"]:s for s in slots}
    require(len(rows) == len({r["root_id"] for r in rows}) and all(r["root_id"] in by_id
        and r["source_seed"] == by_id[r["root_id"]]["source_seed"] for r in rows), "root roster drift")
    roster = tuple((s,tuple(r["root_id"] for r in slots if r["source_seed"] == s)) for s in seeds)
    stage = experiment(spec["contract"])
    cfgs = configurations(stage)
    clustered, result = {}, {}
    for cfg in cfgs[1:]:
        roots = {r["root_id"]:r["root_intervals"][cfg.identity] for r in rows}
        intervals = cluster_intervals(roster,roots)
        clustered[cfg.identity] = intervals
        bounds = [math.fsum(v[j] for v in intervals)/32 for j in (0,1)]
        known = sum(r[0] == r[1] for r in roots.values())
        name = cfg.belief if stage == "A1" else cfg.leaf
        result[name] = dict(fixed_seed_weighted_mean_bounds=bounds,
            mean_delta=bounds[0] if bounds[0] == bounds[1] else None,
            roots_with_identified_contrast=known,uncertain_root_contrasts=200-known,
            input_unavailable_roots=sum(cfg.identity in r.get("input_unavailable",{}) for r in rows),
            bootstrap_envelope=None)
    if stage == "A1":
        roots = {r["root_id"]:r["oracle_minus_public_interval"] for r in rows}
        intervals = cluster_intervals(roster,roots)
        clustered["oracle_minus_public"] = intervals
        bounds = [math.fsum(v[j] for v in intervals)/32 for j in (0,1)]
        known = sum(lo == hi for lo,hi in roots.values())
        result["oracle_minus_public"] = dict(fixed_seed_weighted_mean_bounds=bounds,
            mean_delta=bounds[0] if bounds[0] == bounds[1] else None,
            roots_with_identified_contrast=known,uncertain_root_contrasts=200-known,bootstrap_envelope=None)
    alias_counts = {}
    for cfg in cfgs:
        name = "raw" if cfg.arm == "raw" else cfg.identity
        counts = {k:sum(r["continuation_cell_counts"][name][k] for r in rows)
            for k in ("COMPLETE","CAPPED","REFUSED","UNSTARTED","UNVALIDATED")}
        counts["UNSTARTED"] += (200-len(rows))*8
        alias_counts["raw" if cfg.arm == "raw" else cfg.belief if stage == "A1" else cfg.leaf] = counts
    rng = random.Random(ScreeningDesign(spec["contract"]["namespace"],tuple(seeds)).random_seed)
    boot = {key:[[],[]] for key in clustered}
    for _ in range(10000):
        indices = rng.choices(range(32),k=32)
        for key, intervals in clustered.items():
            for j in (0,1):boot[key][j].append(math.fsum(intervals[i][j] for i in indices)/32)
    for cfg in cfgs[1:]:
        name = cfg.belief if stage == "A1" else cfg.leaf
        result[name]["bootstrap_envelope"] = [sorted(boot[cfg.identity][0])[250],
            sorted(boot[cfg.identity][1])[9749]]
    if stage == "A1":result["oracle_minus_public"]["bootstrap_envelope"] = [
        sorted(boot["oracle_minus_public"][0])[250],sorted(boot["oracle_minus_public"][1])[9749]]
    n = len(rows)
    readout = dict(schema=spec["contract"]["schema"]+".readout",planned_roots=200,
        planned_source_seeds=32,recorded_roots=n,missing_root_slots=200-n,bootstrap_draws=10000,
        continuation_alias_cell_counts=alias_counts,
        continuation_count_scope="1600 fixed aliases per selector; shared action cells reused, not independent replicates",
        analysis="descriptive two-sided 95 percent seed-cluster percentile envelope of missing-data endpoints; not holdout admission",
        estimand="equal weight each of32 frozen source seeds, fixed root quotas within seed",
        elapsed_wall_seconds=terminal["elapsed_wall_seconds"],
        cpu_seconds=terminal["cpu_seconds_including_reaped_children"],
        cpu_scope="observed enclosing auditor wait4 CPU, including descendants it actually reaped; abnormal cleanup is not complete campaign cost",
        wall_scope="enclosing auditor launch through actual reap; external preparation and final readout excluded",
        enclosing_wall_seconds_per_recorded_root=terminal["elapsed_wall_seconds"]/n if n else None,
        enclosing_cpu_seconds_per_recorded_root=terminal["cpu_seconds_including_reaped_children"]/n
            if n and terminal["cpu_seconds_including_reaped_children"] is not None else None,
        supervisor_exit_code=terminal["supervisor_exit_code"],worker_exit_code=terminal["worker_exit_code"],
        retry_authorized=False,holdout_authorized=False,phase_b_authorized=False,significance_claim=False)
    if stage == "A1":readout["oracle_minus_public"] = result.pop("oracle_minus_public")
    readout["beliefs" if stage == "A1" else "leaves"] = result
    return readout


def run(spec_path, review_path, *, check=False, expected_experiment="A2"):
    from .search_over_raw_exploration_entry import ExplorationAuditorContract, _json
    from .search_over_raw_supervision import _supervise_owned_job
    spec,_,_,_ = read_spec(spec_path)
    stage = experiment(spec["contract"])
    require(stage == expected_experiment, "CLI mode differs from original reviewed entry")
    sha = _sha(spec_path)
    review_bytes = Path(review_path).read_bytes()
    review = _json(review_bytes)
    require(review == dict(status="CLEAR_BOUNDED_"+stage+"_EXPLORATION_ENTRY",entry_spec_sha256=sha,
        reviewer="Astra",reviewed_before_first_source_work=True,production_qualification_required=False,
        holdout_authorized=False), "actual one-time independent entry review required")
    if check:return dict(status="BOUNDED_"+stage+"_ENTRY_CHECKED",worker_launched=False)
    output = Path(spec["output_directory"])
    output.mkdir(parents=True,exist_ok=False,mode=0o700)
    (output/"completed-roots").mkdir()
    _write_new(output/"entry.json",spec)
    _write_new(output/"entry-review.json",review)
    def admit():
        read_spec(spec_path,expected_sha=sha)
        require(Path(review_path).read_bytes() == review_bytes,"original entry review changed")
        return True
    terminal = _supervise_owned_job(BoundedA2Job(str(Path(spec_path).resolve()),sha),directory=output/"auditor",
        binding_sha256=digest(dict(spec=sha,review=hashlib.sha256(review_bytes).hexdigest())),
        verify_admission=admit,contract=ExplorationAuditorContract(WALL_SECONDS),kind="staged-exploration-run",
        work_phase="selection",validate_result=lambda value:require(value["status"] == "BOUNDED_"+stage+"_RECORDED",
            "bounded result shape drift"))
    rows, rejected = [], []
    for path in sorted((output/"completed-roots").glob("*.json")):
        try:
            row = _json(path.read_bytes())
            require(row["entry_sha256"] == digest(spec) and row["source_pipeline_validated"] is True
                and all(_sha(p) == expected for p,expected in row["receipt_files"].items()),
                "accepted root owning files changed")
            original = Path(spec["output_directory"])/"roots"/digest(row["root_id"])/"accepted.json"
            require(str(original) in row["receipt_files"] and digest(_json(original.read_bytes()))
                == digest(row["accepted_evidence"]), "original accepted evidence changed")
            evidence = row["accepted_evidence"]["root_evidence"]
            require(digest(row["root_intervals"]) == digest({c.identity:evidence["root_intervals"].get(
                c.identity,[-1.,1.]) for c in configurations(stage)[1:]}), "root intervals differ from accepted evidence")
            outcomes_path = original.parent/"outcomes"/"terminal.json"
            outcomes = _json(outcomes_path.read_bytes())["result"]["outcomes"]["outcomes"] if outcomes_path.exists() else []
            require(digest(row["continuation_cell_counts"]) == digest(continuation_counts(evidence,outcomes,stage)),
                "continuation counts differ from original owned action cells")
            if stage == "A1":require(row["oracle_minus_public_interval"] == direct_belief_interval(evidence,outcomes),
                "direct belief contrast differs from original shared action cells")
            source_directory = Path(spec["output_directory"])/"sources"/str(row["source_seed"])
            require(row["source_receipt_files"] == _pins(source_directory), "original source receipt files changed")
            from .search_over_raw_source_run import read_source_owner
            source = _json((source_directory/"source.json").read_bytes())
            require(row["source_sha256"] == digest(source), "completed root source digest changed")
            read_source_owner(source_directory,source,spec["source_runtime"],spec["contract"])
            rows.append(row)
        except (ValueError,KeyError,OSError) as error:
            rejected.append(dict(path=str(path),error_type=type(error).__name__))
    readout = summarize(spec,rows,terminal)
    readout["rejected_root_receipts"] = rejected
    readout["stop_reason"] = "wall_cap" if terminal["supervisor_exit_code"] == 124 else (
        "operational_failure" if terminal["supervisor_exit_code"] else "fixed_roster_attempt_finished")
    if (output/"input-stop.json").exists():readout["stop_reason"] = "three_consecutive_input_unavailable"
    _write_new(output/"readout.json",readout)
    return readout
