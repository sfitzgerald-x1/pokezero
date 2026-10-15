"""Staged selectors from an actual private archive; no source generation.

Recipe review, source archive, prepared job, real supervisor and independent
owning reader are joined here. The archive stays in the trusted process and is
never transported here; the private owner handle cannot be persisted.
Scientific entry still requires separate exploration authority.
The explicit authored-process route is unadmitted and cannot qualify a runtime.
"""
from copy import deepcopy
from dataclasses import asdict,fields,replace,MISSING
import hashlib
from inspect import getattr_static
import pickle
from pathlib import Path
from types import MethodType

from .search_over_raw import SearchConfiguration,digest,require
from .search_over_raw_ledger import _load,_write_new
from .search_over_raw_reconciled import ReconciledPhaseALedger,selector_runtime_recipe
from .search_over_raw_archive import (
    SealedSourceArchive,archived_exploration_root_binding,
    prepare_archived_raw_job,prepare_archived_native_timed_job,prepare_archived_reference_timed_job,
    prepare_archived_native_oracle_timed_job,prepare_archived_reference_oracle_timed_job,
    capture_archived_team_oracle,
)
from .search_over_raw_continuation_admission import _fresh_review,_path,_inventory,NAMESPACE_SCHEMA,_FLAGS
from .search_over_raw_raw_run import RawSelectionJob,RawSelectionRunValidator,supervise_raw_selection
from .search_over_raw_timed_run import NativeTimedAdapterJob,NativeTimedRunValidator,supervise_native_timed
from .search_over_raw_actor_timed_run import (RUNTIME_KEY, ActorInputUnavailable, UNAVAILABLE_REASONS,
    ActorAnnotatedTimedJob, ActorAnnotatedTimedRunValidator, prepare_archived_actor_timed_job,
    supervise_actor_timed, _actor_runtime)
from .search_over_raw_reference_timed_run import (
    ReferenceTimedAdapterJob,ReferenceTimedRunValidator,supervise_reference_timed,
)
from .search_over_raw_native_oracle_run import (
    NativeOracleTimedAdapterJob,NativeOracleTimedRunValidator,supervise_native_oracle_timed,
)
from .search_over_raw_reference_oracle_run import (
    ReferenceOracleTimedAdapterJob,ReferenceOracleTimedRunValidator,supervise_reference_oracle_timed,
)
from .search_over_raw_supervision import _supervise_owned_job
from .search_over_raw_world_inputs import _state_payload,public_request_witness


_ORIGINAL_LEDGER_CLASS = ReconciledPhaseALedger
_ORIGINAL_LEDGER_VERIFY = ReconciledPhaseALedger.verify
_ORIGINAL_EXPLORATION_VERIFY = ReconciledPhaseALedger.verify_exploration


def _verify_selector_admission(ledger, stage, configuration, *, synthetic):
    """One fresh ledger scan at the existing admission boundary; no cache.

    Only the unchanged canonical methods are known to be callback-free here.
    Wrappers/overrides retain the original two-call order and all their checks.
    Static lookup must not invoke an overridden descriptor before integrity.
    """
    require(type(synthetic) is bool, "explicit selector admission scope required")
    if synthetic:
        ledger.verify()
        return
    if (type(ledger) is _ORIGINAL_LEDGER_CLASS
            and type(ledger).__getattribute__ is object.__getattribute__
            and getattr_static(ledger, "verify", None) is _ORIGINAL_LEDGER_VERIFY
            and getattr_static(ledger, "verify_exploration", None) is _ORIGINAL_EXPLORATION_VERIFY):
        verify, exploration = ledger.verify, ledger.verify_exploration
        if (type(verify) is MethodType and verify.__self__ is ledger
                and verify.__func__ is _ORIGINAL_LEDGER_VERIFY
                and type(exploration) is MethodType and exploration.__self__ is ledger
                and exploration.__func__ is _ORIGINAL_EXPLORATION_VERIFY):
            _ORIGINAL_EXPLORATION_VERIFY(ledger, stage, configuration)
            return
    ledger.verify()
    ledger.verify_exploration(stage, configuration)


def _route(cfg, runtime=None):
    if type(runtime) is dict and RUNTIME_KEY in runtime:
        _actor_runtime(cfg, runtime, digest(runtime))
        return ActorAnnotatedTimedJob,prepare_archived_actor_timed_job,ActorAnnotatedTimedRunValidator,supervise_actor_timed
    if cfg.arm == "raw": return RawSelectionJob,prepare_archived_raw_job,RawSelectionRunValidator,supervise_raw_selection
    if cfg.arm == "incumbent" and cfg.belief == "public":
        return NativeTimedAdapterJob,prepare_archived_native_timed_job,NativeTimedRunValidator,supervise_native_timed
    if cfg.arm == "reference" and cfg.belief == "public":
        return ReferenceTimedAdapterJob,prepare_archived_reference_timed_job,ReferenceTimedRunValidator,supervise_reference_timed
    if cfg.arm == "incumbent" and cfg.belief == "oracle":
        return NativeOracleTimedAdapterJob,prepare_archived_native_oracle_timed_job,NativeOracleTimedRunValidator,supervise_native_oracle_timed
    require(cfg.arm == "reference" and cfg.belief == "oracle", "supported staged selector required")
    return ReferenceOracleTimedAdapterJob,prepare_archived_reference_oracle_timed_job,ReferenceOracleTimedRunValidator,supervise_reference_oracle_timed


def _job_envelope(job,cfg):
    request=job._request if cfg.arm == "raw" else job._public_request
    request=replace(request,state=job.context.public_materialization_state,
        observation=job.context.observation,pending_transition=job.pending_transition)
    value=dict(configuration=asdict(job.configuration),runtime=job.expected_runtime_configuration,
        checkpoint=_state_payload(job.checkpoint_contract),showdown=job.showdown_root,
        root=job.root_id,statistics_root=job.statistics_root_id,seed=job.selection_seed,
        public=public_request_witness(request),expected_public=job.expected_public_request_sha256,
        consumed=job._consumed_binding,mask=list(job._mask if cfg.arm == "raw" else job._legal_action_mask),
        player=job.context.player_id,decision_round=job.context.decision_round_index,
        run_contract=_state_payload(job.contract),reference_factory=None,oracle=None)
    if cfg.arm == "reference":value["reference_factory"]=_state_payload(job.reference_factory)
    if type(job) is ActorAnnotatedTimedJob:
        value["actor_annotations"] = dict(boundary_sha256=job.annotation_boundary._original_sha,
            payload=_state_payload(job.annotation_boundary))
    if cfg.belief == "oracle":
        value["oracle"]=dict(payload=_state_payload(job.oracle),receipt=job.expected_oracle_receipt,
            catalog=job.expected_reconstruction_catalog_sha256,
            override=job.expected_override_sha256 if cfg.arm == "incumbent" else None)
    return value


class OwnedSelectorRun:
    """Private reusable handle; existence or matching labels is not ownership."""
    def __init__(self, *, ledger,job,selection,reader,archived_binding,claim_path,claim_bytes,source_owner=None):
        self.ledger,self.job,self.selection,self.reader = ledger,job,selection,reader
        self.source_owner=source_owner
        self._check_types()
        self.archived_binding=deepcopy(archived_binding)
        self.claim_path,self.claim_bytes=claim_path,claim_bytes
        self.frozen=self._inputs()

    def __getstate__(self): raise TypeError("private selector owner handle cannot be persisted")

    def _check_types(self):
        cls,_,reader_cls,_=_route(self.job.configuration,self.job.expected_runtime_configuration)
        require(type(self.ledger) is ReconciledPhaseALedger and type(self.reader) is reader_cls
            and (type(self.job) is cls or self.reader.allow_synthetic is True and isinstance(self.job,cls)
                and getattr(self.job,"synthetic_process_fixture",False) is True),
            "exact selector job and independent reader required")

    def _inputs(self):
        identities=[(id(obj),type(obj).__module__,type(obj).__qualname__)
            for obj in (self.ledger,self.job,self.selection,self.reader,self.source_owner)]
        return digest(dict(identities=identities,contract=self.ledger.contract_sha256,job_inputs=self.job._inputs(),
            source_owner=None if self.source_owner is None else self.source_owner._identity(),
            job_sha256=hashlib.sha256(pickle.dumps(self.job)).hexdigest(),selection=self.selection,
            reader_inputs=self.reader._inputs(),readout=self.reader.last_validation,
            directory_inventory=_inventory(self.reader.directory),archived_binding=self.archived_binding,
            claim_path=str(self.claim_path),claim_bytes_sha256=hashlib.sha256(self.claim_path.read_bytes()).hexdigest()))

    def validate(self, archive, source):
        self._check_types()
        if self.source_owner is not None:self.source_owner.require_inputs(self.ledger,archive,source)
        self.ledger.verify()
        require(self._inputs() == self.frozen and self.claim_path.read_bytes() == self.claim_bytes,
            "original selector owner inputs or claim changed")
        require(self.reader(self.selection) == digest(self.selection), "actual selector ownership required")
        bound=archived_exploration_root_binding(archive,source,contract=self.ledger.contract,
            root_id=self.job.statistics_root_id)
        require(digest(bound) == digest(self.archived_binding), "reused selector differs from actual archive")
        if self.source_owner is not None:self.source_owner.require_inputs(self.ledger,archive,source)
        self.ledger.verify()
        self._check_types()
        require(self._inputs() == self.frozen and self.claim_path.read_bytes() == self.claim_bytes,
            "selector owner changed during final archive/reader callbacks")
        return self


def _actor_source_review(ledger, source_owner, path):
    from .search_over_raw_source_run import TrustedOwnedSource
    require(type(source_owner) is TrustedOwnedSource, "exact actor source owner required")
    source_owner.require_inputs(ledger,source_owner.archive,source_owner.source)
    path=_path(path); raw=path.read_bytes(); sha=hashlib.sha256(raw).hexdigest()
    claim=_load(source_owner.directory/"claim.json"); review=_load(path)
    require(ledger.contract["input_hashes"].get(str(path)) == sha == claim["source_review_sha256"]
        and review["status"] == "CLEAR_SOURCE_RECIPE" and digest(review["recipe"]) == claim["recipe_sha256"],
        "actor configuration must match the original claim-bound source review")
    source_owner._original_owner()
    require(path.read_bytes() == raw, "original actor source review changed")
    return review["recipe"],dict(review_path=str(path),review_sha256=sha,recipe_sha256=claim["recipe_sha256"],
        config_sha256=review["recipe"]["config_sha256"])


def _actor_source_configuration(ledger, source_owner, config, path, checkpoint):
    from .search_over_raw_source_run import source_recipe
    recipe,binding=_actor_source_review(ledger,source_owner,path)
    before=digest(_state_payload(config))
    require(before == binding["config_sha256"] and digest(source_recipe(config,checkpoint,source_owner.runtime,
        capture_actor_history=source_owner._actor_history is not None)) == digest(recipe),
        "actor source configuration/checkpoint differs from original reviewed source")
    source_owner._original_owner()
    require(digest(_state_payload(config)) == before, "original actor source configuration changed")
    return binding


def _actor_unavailable_path(ledger, root_id, configuration):
    return _path(ledger.directory/"selector-producer-claims"/(digest(dict(root_id=root_id,
        configuration=configuration))+"-input-unavailable.json"))


def read_recorded_actor_input_unavailable(ledger, receipt, *, configuration=None):
    """Durable prelaunch gap consistency after private archives are closed.

    The original in-auditor reader proves the archive join. This readback
    rechecks that record, source/review binding and absence of candidate work;
    it neither reconstructs private annotations nor independently proves why
    they were unavailable. Every such root remains uncertain.
    """
    keys={"schema","status","reason","contract_sha256","root_id","configuration","source_seed",
        "source_sha256","archived_binding_sha256","source_configuration","runtime","selector_launched",
        "selector_attempted","continuation_authorized","observed_zero_work","retry_authorized","holdout_authorized"}
    require(type(receipt) is dict and set(receipt)==keys
        and receipt["schema"] == "pokezero.search-over-raw.actor-input-unavailable.v1"
        and receipt["status"] == "INPUT_UNAVAILABLE" and type(receipt["reason"]) is str
        and receipt["reason"] in UNAVAILABLE_REASONS
        and receipt["contract_sha256"] == ledger.contract_sha256
        and type(receipt["root_id"]) is str and type(receipt["source_seed"]) is int
        and all(receipt[k] is False for k in ("selector_launched","selector_attempted",
            "continuation_authorized","observed_zero_work","retry_authorized","holdout_authorized")),
        "exact typed prelaunch actor input-unavailable receipt required")
    before=digest(receipt);cfg=SearchConfiguration(**receipt["configuration"])
    require(configuration is None or digest(asdict(cfg)) == digest(asdict(configuration)),
        "unavailable root belongs to another configuration")
    _actor_runtime(cfg,receipt["runtime"],digest(receipt["runtime"]))
    slots=[s for s in ledger.contract["panels"]["exploration"]["root_slots"]
        if s["root_id"]==receipt["root_id"]]
    require(len(slots)==1 and slots[0]["source_seed"]==receipt["source_seed"],
        "unavailable root differs from fixed exploration source slot")
    binding=receipt["source_configuration"]
    require(type(binding) is dict and set(binding)=={"review_path","review_sha256","recipe_sha256","config_sha256"},
        "exact original source review binding required")
    path=_actor_unavailable_path(ledger,receipt["root_id"],cfg.identity)
    source_directory=_path(ledger.directory/"source-attempts"/str(receipt["source_seed"]))
    source_path=source_directory/"source.json";source_claim_path=source_directory/"claim.json"
    review_path=_path(binding["review_path"])
    paths=(path,source_path,source_claim_path,review_path)
    raw={p:p.read_bytes() for p in paths}
    source,claim,review=_load(source_path),_load(source_claim_path),_load(review_path)
    from .search_over_raw_source_run import source_review_template
    expected=source_review_template(ledger.contract,review["recipe"],
        qualification_evidence_hashes=review["qualification_evidence_hashes"])
    expected.update(status="CLEAR_SOURCE_RECIPE",reviewed_before_first_source_work=True,
        source_runtime_and_cleanup_reviewed=True,source_policy_and_terminal_reviewed=True)
    sha=hashlib.sha256(raw[review_path]).hexdigest()
    require(digest(_load(path))==before and digest(review)==digest(expected)
        and ledger.contract["input_hashes"].get(str(review_path))==sha==binding["review_sha256"]
        ==claim["source_review_sha256"]
        and digest(review["recipe"])==binding["recipe_sha256"]==claim["recipe_sha256"]
        and review["recipe"]["config_sha256"]==binding["config_sha256"]
        and source["source_seed"]==claim["source_seed"]==receipt["source_seed"]
        and type(source["source_seed"]) is int and type(claim["source_seed"]) is int
        and source["contract_sha256"]==claim["contract_sha256"]==ledger.contract_sha256
        and digest(source)==receipt["source_sha256"]
        and len([r for r in source["roots"] if r["root_id"]==receipt["root_id"]])==1
        and receipt["runtime"]["checkpoint_sha256"]==source["checkpoint_sha256"]
        ==review["recipe"]["checkpoint"]["checkpoint_sha256"]
        and type(receipt["archived_binding_sha256"]) is str and len(receipt["archived_binding_sha256"])==64
        and all(c in "0123456789abcdef" for c in receipt["archived_binding_sha256"]),
        "durable unavailable source/review/checkpoint binding drift")
    claim_path=path.with_name(path.name.replace("-input-unavailable.json",".json"))
    directory=ledger.directory/"selector-runs"/digest(dict(root_id=receipt["root_id"],configuration=cfg.identity))
    ledger.verify()
    require(digest(receipt)==before and all(p.read_bytes()==value for p,value in raw.items())
        and not claim_path.exists() and not directory.exists(),
        "unavailable actor evidence or candidate absence changed during readback")
    return deepcopy(receipt)


def read_actor_input_unavailable(ledger, source_owner, receipt, *, configuration=None):
    """Existing producer evidence, not an owner, attempt, or zero-work result."""
    read_recorded_actor_input_unavailable(ledger,receipt,configuration=configuration)
    require(type(receipt) is dict and receipt.get("schema") == "pokezero.search-over-raw.actor-input-unavailable.v1"
        and receipt.get("status") == "INPUT_UNAVAILABLE" and receipt.get("reason") in UNAVAILABLE_REASONS,
        "typed prelaunch actor input-unavailable receipt required")
    before=digest(receipt)
    cfg=SearchConfiguration(**receipt["configuration"])
    require(configuration is None or digest(asdict(cfg)) == digest(asdict(configuration)),
        "unavailable root belongs to another configuration")
    _actor_runtime(cfg,receipt["runtime"],digest(receipt["runtime"]))
    path=_actor_unavailable_path(ledger,receipt["root_id"],cfg.identity)
    raw=path.read_bytes()
    require(digest(_load(path)) == before and receipt["contract_sha256"] == ledger.contract_sha256
        and all(receipt[k] is False for k in ("selector_launched","selector_attempted",
            "continuation_authorized","observed_zero_work","retry_authorized","holdout_authorized")),
        "original unavailable receipt/type drift")
    recipe,source_binding=_actor_source_review(ledger,source_owner,receipt["source_configuration"]["review_path"])
    require(digest(source_binding) == digest(receipt["source_configuration"])
        and receipt["source_seed"] == source_owner.source["source_seed"]
        and receipt["source_sha256"] == digest(source_owner.source)
        and receipt["runtime"]["checkpoint_sha256"] == recipe["checkpoint"]["checkpoint_sha256"]
        and receipt["archived_binding_sha256"] == digest(archived_exploration_root_binding(source_owner.archive,
            source_owner.source,contract=ledger.contract,root_id=receipt["root_id"])),
        "unavailable actor source/root/configuration drift")
    claim_path=path.with_name(path.name.replace("-input-unavailable.json",".json"))
    directory=ledger.directory/"selector-runs"/digest(dict(root_id=receipt["root_id"],configuration=cfg.identity))
    require(not claim_path.exists() and not directory.exists(), "started actor work cannot become input-unavailable")
    source_owner._original_owner(); ledger.verify()
    require(digest(receipt) == before and path.read_bytes() == raw
        and not claim_path.exists() and not directory.exists(), "unavailable actor evidence changed during callbacks")
    return deepcopy(receipt)


def produce_archived_selector(ledger,stage,configuration,*,archive,source,root_id,
        checkpoint_contract,showdown_root,runtime,file_hashes,qualification_review_path,
        history_review_path,selector="search",reference_factory=None,set_source=None,
        prior_owner=None,source_owner=None,source_config=None,source_review_path=None,
        _synthetic_job_class=None,_synthetic_job_kwargs=None,_synthetic_actual_actor=False):
    """Prepare/own one selector, or validate a supplied exact prior owner.

    No factorial prerequisite, source game, continuation, selection freeze or
    holdout access. The canonical raw owner is reused across stages/arms. Same
    public model work can also be reused between A1 and A2. Directory existence
    never infers reuse, process success or permission to retry.
    """
    require(type(ledger) is ReconciledPhaseALedger and type(archive) is SealedSourceArchive,
        "exact ledger and trusted source archive required")
    review,review_bytes=ledger.configuration_recipe_review(stage,configuration,qualification_review_path)
    cfg=next(c for c in ledger._stage_configs(stage) if c.identity == configuration)
    require(not (stage == "A3" and cfg.seconds == 1.), "A3 one-second selector must reuse frozen A2 record")
    require(not (ledger.directory/"selection.json").exists()
        and not (stage == "A2" and (ledger.directory/"a2-freeze.json").exists()), "exploration is frozen")
    require(selector in {"raw","search"}, "explicit raw/search selector role required")
    run_cfg=SearchConfiguration("raw",seconds=1.) if selector == "raw" else cfg
    cls,builder,reader_cls,supervisor=_route(run_cfg,runtime)
    actor = cls is ActorAnnotatedTimedJob
    require(type(_synthetic_actual_actor) is bool, "exact authored actor scope required")
    synthetic=_synthetic_job_class is not None or _synthetic_actual_actor
    if synthetic:
        require((_synthetic_actual_actor and actor and _synthetic_job_class is None and _synthetic_job_kwargs is None
                or not actor and isinstance(_synthetic_job_class,type) and issubclass(_synthetic_job_class,cls)
                    and getattr(_synthetic_job_class,"synthetic_process_fixture",False) is True)
            and ledger.contract["execution_ready"] is False and ledger.contract["collection_authorized"] is False,
            "explicit unadmitted authored selector scope required")
    else:
        require(_synthetic_job_kwargs is None, "authored inputs cannot enter production")
        ledger.verify_exploration(stage,configuration)
        require(source_owner is not None,"production selector requires its original owned fresh source")
    if source_owner is not None:
        from .search_over_raw_source_run import TrustedOwnedSource
        require(type(source_owner) is TrustedOwnedSource,"exact trusted fresh source owner required")
        source_owner.require_inputs(ledger,archive,source)
    if actor:
        require(source_owner is not None, "actor route requires exact original owned source")
        actor_source = _actor_source_configuration(ledger,source_owner,source_config,source_review_path,checkpoint_contract)
        if synthetic: require(_load(source_owner.directory/"claim.json")["synthetic_process_fixture"] is True,
            "authored actor selector requires an explicitly authored source")
    require(type(runtime) is dict and type(file_hashes) is dict
        and digest(runtime.get("configuration")) == digest(asdict(run_cfg)),
        "exact selector runtime and file inventory required")
    require((digest(runtime) == review["raw_runtime_sha256"] if selector == "raw" else
        digest(selector_runtime_recipe(cfg,runtime)) == review["selector_recipe_sha256"]),
        "selector runtime differs from prospective recipe")
    history,history_sha=_fresh_review(ledger,history_review_path)
    namespace_path=_path(ledger.directory/"continuation-admission"/"namespace.json")
    namespace_bytes=namespace_path.read_bytes()
    require(digest(_load(namespace_path)) == digest(dict(schema=NAMESPACE_SCHEMA,
        contract_sha256=ledger.contract_sha256,review_path=str(_path(history_review_path)),
        review_sha256=history_sha,review=history,**_FLAGS)), "fresh namespace not initialized under this contract")
    require(history["raw_runtime_sha256"] == review["raw_runtime_sha256"]
        and root_id in {r["root_id"] for r in history["exploration_slots"]}, "fixed fresh root/raw recipe required")
    directory=_path(Path(history["raw_owner_namespace"])/digest(dict(root_id=root_id)) if selector == "raw" else
        ledger.directory/"selector-runs"/digest(dict(root_id=root_id,configuration=configuration)))
    review_path=_path(qualification_review_path)
    inputs_sha=digest(dict(source=source,runtime=runtime,files=file_hashes,
        checkpoint=_state_payload(checkpoint_contract),showdown=showdown_root,
        reference_factory=_state_payload(reference_factory),set_source=_state_payload(set_source),
        actor_source=actor_source if actor else None))
    def admission_guard():
        if source_owner is not None:source_owner.require_inputs(ledger,archive,source)
        _verify_selector_admission(ledger,stage,configuration,synthetic=synthetic)
        if actor:
            require(digest(_actor_source_configuration(ledger,source_owner,source_config,source_review_path,
                checkpoint_contract)) == digest(actor_source), "original actor source recipe changed")
        require(review_path.read_bytes() == review_bytes and namespace_path.read_bytes() == namespace_bytes
            and digest(dict(source=source,runtime=runtime,files=file_hashes,
                checkpoint=_state_payload(checkpoint_contract),showdown=showdown_root,
                reference_factory=_state_payload(reference_factory),set_source=_state_payload(set_source),
                actor_source=actor_source if actor else None)) == inputs_sha,
            "selector admission/source/runtime inputs changed")
    if prior_owner is not None:
        require(type(prior_owner) is OwnedSelectorRun and prior_owner.ledger is ledger
            and prior_owner.job.configuration == run_cfg and prior_owner.job.statistics_root_id == root_id
            and prior_owner.reader.directory == directory and prior_owner.reader.runtime_sha == digest(runtime)
            and digest(prior_owner.reader.files) == digest(file_hashes)
            and prior_owner.reader.checkpoint_path == checkpoint_contract.checkpoint_path
            and prior_owner.reader.showdown_root == showdown_root
            and prior_owner.reader.allow_synthetic is synthetic,
            "exact original canonical selector owner required; labels cannot infer reuse")
        require(prior_owner.source_owner is source_owner,"selector reuse cannot replace original source owner")
        try:
            prior_owner.validate(archive,source);admission_guard()
            require(prior_owner._inputs()==prior_owner.frozen,"reused selector changed during final source callbacks")
            return prior_owner,dict(status="OWNED_SELECTOR_REUSED",selector_launched=False,
                runtime_qualified_by_this_code=False,collection_authorized=False,holdout_authorized=False)
        except BaseException as error:
            if not (ledger.directory/"operational-stop.json").exists():
                try: ledger.stop(type(error).__name__)
                except BaseException as stop_error:error.add_note("Unable to persist operational stop: "+type(stop_error).__name__)
            raise
    claims=_path(ledger.directory/"selector-producer-claims");claims.mkdir(exist_ok=True,mode=0o700)
    claim_path=claims/(digest(dict(root_id=root_id,configuration=run_cfg.identity))+".json")
    actor_job = None
    if actor:
        admission_guard()
        unavailable_path = _actor_unavailable_path(ledger,root_id,run_cfg.identity)
        if unavailable_path.exists():
            receipt = read_actor_input_unavailable(ledger,source_owner,_load(unavailable_path),configuration=run_cfg)
            admission_guard()
            return None,receipt
        require(not claim_path.exists() and not directory.exists(), "consumed actor selector cannot be prepared again")
        try:
            actor_job, actor_binding = builder(archive,source,contract=ledger.contract,root_id=root_id,
                configuration=run_cfg,checkpoint_contract=checkpoint_contract,showdown_root=showdown_root,
                runtime_configuration=runtime,source_config=source_config,
                source_config_sha256=actor_source["config_sha256"],synthetic_process_fixture=synthetic)
        except ActorInputUnavailable as gap:
            admission_guard()  # Mutation/provenance failure wins over ordinary unavailability.
            bound=archived_exploration_root_binding(archive,source,contract=ledger.contract,root_id=root_id)
            admission_guard()
            receipt=dict(schema="pokezero.search-over-raw.actor-input-unavailable.v1",status="INPUT_UNAVAILABLE",
                contract_sha256=ledger.contract_sha256,root_id=root_id,configuration=asdict(run_cfg),
                source_seed=source["source_seed"],source_sha256=digest(source),archived_binding_sha256=digest(bound),
                source_configuration=deepcopy(actor_source),runtime=deepcopy(runtime),reason=gap.reason,
                selector_launched=False,selector_attempted=False,continuation_authorized=False,
                observed_zero_work=False,retry_authorized=False,holdout_authorized=False)
            admission_guard()
            _write_new(unavailable_path,receipt)
            return None,read_actor_input_unavailable(ledger,source_owner,receipt,configuration=run_cfg)
        admission_guard()
    claim=dict(contract_sha256=ledger.contract_sha256,root_id=root_id,configuration=run_cfg.identity,
        first_stage=stage,qualification_review_sha256=hashlib.sha256(review_bytes).hexdigest(),
        history_review_sha256=history_sha,run_directory=str(directory),synthetic_process_fixture=synthetic,
        retry_authorized=False,holdout_authorized=False)
    _write_new(claim_path,claim);claim_bytes=claim_path.read_bytes()
    def claim_guard():
        admission_guard()
        require(_path(claim_path).read_bytes() == claim_bytes, "selector producer claim changed")
    try:
        claim_guard()
        bound=archived_exploration_root_binding(archive,source,contract=ledger.contract,root_id=root_id)
        bound_sha=digest(bound)
        witness=bound["public_request_witness"]
        run_contract=next(f.default for f in fields(cls) if f.name == "contract")
        expected=dict(configuration=asdict(run_cfg),runtime=deepcopy(runtime),
            checkpoint=_state_payload(checkpoint_contract),showdown=showdown_root,
            root=root_id+(":raw" if selector == "raw" else ":"+configuration),statistics_root=root_id,
            seed=bound["slot"]["source_seed"],public=deepcopy(witness),
            expected_public=witness["public_request_sha256"],consumed=deepcopy(witness["consumed_binding"]),
            mask=deepcopy(bound["public_evidence_binding"]["legal_action_mask"]),
            player=witness["consumed_binding"]["subject"],decision_round=bound["source_request_index"],
            run_contract=_state_payload(run_contract),reference_factory=_state_payload(reference_factory)
                if run_cfg.arm == "reference" else None,oracle=None)
        if run_cfg.belief == "oracle":
            oracle,projection=capture_archived_team_oracle(archive,source,contract=ledger.contract,
                root_id=root_id,set_source=set_source)
            require(projection["archived_root_sha256"] == bound_sha,"original selector team/archive drift")
            override=None
            if run_cfg.arm == "incumbent":
                from .search_over_raw_oracle import opening_team
                from ..env import BattleStartOverride
                from ..showdown_fixture import pack_team
                root=next(r for r in source["roots"] if r["root_id"] == root_id)
                context,_,_=archive.selected(root)
                own=opening_team(context.public_materialization_state.self_initial_request,oracle.subject,set_source)
                opponent="p2" if oracle.subject == "p1" else "p1"
                override=digest(asdict(BattleStartOverride(player_teams={oracle.subject:pack_team(own),
                    opponent:pack_team(oracle.opponent_team)},observation_format_id="gen3randombattle")))
            expected["oracle"]=dict(payload=_state_payload(oracle),receipt=deepcopy(projection["team_oracle"]),
                catalog=projection["reconstruction_catalog_sha256"],override=override)
        expected_sha=digest(expected)
        if actor:
            expected["actor_annotations"] = dict(boundary_sha256=actor_job.annotation_boundary._original_sha,
                payload=_state_payload(actor_job.annotation_boundary))
            expected_sha=digest(expected)
        claim_guard()
        args=dict(contract=ledger.contract,root_id=root_id,configuration=run_cfg,
            checkpoint_contract=checkpoint_contract,showdown_root=showdown_root,runtime_configuration=runtime)
        if run_cfg.arm == "reference": args["reference_factory"]=reference_factory
        if run_cfg.belief == "oracle": args["set_source"]=set_source
        job=actor_job if actor else builder(archive,source,**args)[0]
        require(type(job) is cls, "actual prepared archived selector job required")
        require(digest(_job_envelope(job,run_cfg)) == expected_sha,
            "returned selector differs from original prospective job envelope")
        claim_guard()
        if synthetic and not actor:
            kwargs={} if _synthetic_job_kwargs is None else deepcopy(_synthetic_job_kwargs)
            original_fields={f.name for f in fields(job)}
            extra_fields={f.name:f for f in fields(_synthetic_job_class) if f.name not in original_fields}
            require(type(kwargs) is dict and set(kwargs)<=set(extra_fields),
                "authored fields cannot replace production job inputs")
            # The production constructor has already captured and sanitized the
            # public request, deliberately stripping its catalog. Recapturing
            # that transport is invalid. Preserve the entire prepared instance,
            # including its original frozen guards; only add authored fields.
            original_inputs=job._inputs()
            authored=object.__new__(_synthetic_job_class)
            authored.__dict__.update(deepcopy(job.__dict__))
            for name,field in extra_fields.items():
                if name in kwargs:value=kwargs[name]
                elif field.default is not MISSING:value=deepcopy(field.default)
                elif field.default_factory is not MISSING:value=field.default_factory()
                else:raise ValueError("required authored field missing: "+name)
                object.__setattr__(authored,name,value)
            require(authored._inputs() == original_inputs,"authored transport changed original prepared inputs")
            job=authored
        job_inputs,job_sha=job._inputs(),hashlib.sha256(pickle.dumps(job)).hexdigest()
        binding=digest(dict(claim=claim,job_inputs=job_inputs,job_sha256=job_sha,archive_sha256=bound_sha))
        def verify():
            claim_guard()
            require(digest(archived_exploration_root_binding(archive,source,contract=ledger.contract,root_id=root_id)) == bound_sha,
                "actual archive changed after selector preparation")
            require(job._inputs() == job_inputs and hashlib.sha256(pickle.dumps(job)).hexdigest() == job_sha,
                "original prepared selector job changed")
            require(digest(_job_envelope(job,run_cfg)) == expected_sha and digest(expected) == expected_sha,
                "original prospective selector job envelope changed")
            claim_guard()
        reader_args=dict(directory=directory,job_sha256=job_sha,binding_sha256=binding,configuration=run_cfg,
            root_id=expected["root"],statistics_root_id=root_id,selection_seed=bound["slot"]["source_seed"],
            runtime_configuration=runtime,runtime_sha256=digest(runtime),consumed_binding=witness["consumed_binding"],
            legal_action_mask=bound["public_evidence_binding"]["legal_action_mask"],
            decision_round=None if selector == "raw" else bound["source_request_index"],file_hashes=file_hashes,
            checkpoint_path=checkpoint_contract.checkpoint_path,checkpoint_sha256=checkpoint_contract.checkpoint_sha256,
            showdown_root=showdown_root,allow_synthetic=synthetic,contract=run_contract,
            public_request_sha256=witness["public_request_sha256"])
        if run_cfg.belief == "oracle":
            reader_args.update(oracle_receipt=job.expected_oracle_receipt,public_request=witness,
                expected_reconstruction_catalog_sha256=job.expected_reconstruction_catalog_sha256)
            if run_cfg.arm == "incumbent": reader_args["expected_override_sha256"]=job.expected_override_sha256
            else: reader_args["expected_oracle_payload_sha256"]=digest(_state_payload(job.oracle))
        if actor: reader_args["annotation_sha256"] = job.annotation_boundary._original_sha
        reader=reader_cls(**reader_args)
        reader_inputs=reader._inputs()
        verify()
        if synthetic and not actor:
            terminal=_supervise_owned_job(job,directory=directory,binding_sha256=binding,
                verify_admission=verify,contract=job.contract,kind=reader._kind,
                work_phase="selection",validate_result=job.read_selection)
        else:
            terminal=supervisor(job,directory=directory,binding_sha256=binding,verify_admission=verify)
        require(terminal.get("supervisor_exit_code") == 0 and terminal.get("worker_exit_code") == 0
            and terminal.get("status") == reader._complete_status,"selector producer operational failure")
        selection=terminal["result"]
        require(reader(selection) == digest(selection) and reader.last_validation["synthetic_process_fixture"] is synthetic,
            "independent actual selector owner required")
        owner=OwnedSelectorRun(ledger=ledger,job=job,selection=selection,reader=reader,
            archived_binding=bound,claim_path=claim_path,claim_bytes=claim_bytes,source_owner=source_owner)
        owner.validate(archive,source);verify()
        require(reader._inputs() == reader_inputs and owner._inputs() == owner.frozen,
            "accepted selector owner changed during final callbacks")
        claim_guard()
        require(owner._inputs()==owner.frozen,"selector changed during final source/admission callbacks")
        return owner,dict(status=selection["status"],selector_launched=True,
            run_directory=str(directory),selection_sha256=digest(selection),synthetic_process_fixture=synthetic,
            runtime_qualified_by_this_code=False,collection_authorized=False,holdout_authorized=False,retry_authorized=False)
    except BaseException as error:
        if not (ledger.directory/"operational-stop.json").exists():
            try: ledger.stop(type(error).__name__)
            except BaseException as stop_error:error.add_note("Unable to persist operational stop: "+type(stop_error).__name__)
        raise
