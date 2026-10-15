"""Private no-launch archive → owned selectors → unique continuation handoff.

This preparation is not stage admission or source/runtime qualification. It
retains no source environment, exports no private snapshot in its public receipt,
and never starts a model, search, simulator, registration or continuation.
"""
from copy import copy, deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import pickle
from pathlib import Path

from .search_over_raw import digest, require
from .search_over_raw_archive import archived_exploration_root_binding
from .search_over_raw_evidence import reconcile_root_evidence, selector_key
from .search_over_raw_outcome_run import OwnedContinuationJob, OwningRunValidator, ContinuationRunContract, verify_outcome_runtime
from .search_over_raw_outcomes import OutcomeReceiptValidator, snapshot_identity
from .search_over_raw_raw_run import RawSelectionRunValidator
from .search_over_raw_timed_run import NativeTimedRunValidator
from .search_over_raw_reference_timed_run import ReferenceTimedRunValidator
from .search_over_raw_native_oracle_run import NativeOracleTimedRunValidator
from .search_over_raw_reference_oracle_run import ReferenceOracleTimedRunValidator
from .search_over_raw_world_inputs import _state_payload


def _selector_reader_class(cfg, runtime=None):
    from .search_over_raw_actor_timed_run import RUNTIME_KEY, _actor_runtime, ActorAnnotatedTimedRunValidator
    if type(runtime) is dict and RUNTIME_KEY in runtime:
        _actor_runtime(cfg,runtime,digest(runtime))
        return ActorAnnotatedTimedRunValidator
    if cfg.belief == "oracle":
        from .search_over_raw_reconciled import a1_configurations
        require(cfg in a1_configurations() and cfg.arm != "raw",
            "original-team owner is A1 model/one-second only")
        return {"incumbent":NativeOracleTimedRunValidator,"reference":ReferenceOracleTimedRunValidator}[cfg.arm]
    require(cfg.belief == "public" and cfg.leaf in {"model","hp_fraction"},
        "fixed-work selectors remain outside this handoff")
    return {"raw":RawSelectionRunValidator,"incumbent":NativeTimedRunValidator,
        "reference":ReferenceTimedRunValidator}[cfg.arm]


class ArchivedRootEvaluationPlan:
    """A private controller object, with revalidation required at each handoff.

Public model/HP owners are the default. A1 original-team owners require an
explicit private reconstruction catalog and actual archive-truth join.
Production qualification and stage execution admission remain separate.
Opted-in timed owners additionally bind the full public request at actual
capture and native attempts or reference worker preparation/exports. Compact
owners remain explicitly partial; transport is not evaluator/transition truth.
"""
    def __init__(self, archive, source, *, contract, root_id, configurations, selections,
                 selector_readers, allow_synthetic=False, oracle_set_source=None,source_owner=None):
        require(type(allow_synthetic) is bool, "explicit synthetic handoff scope required")
        self.source_owner=source_owner
        self._original_source_owner=source_owner
        if source_owner is not None:
            from .search_over_raw_source_run import TrustedOwnedSource
            require(type(source_owner) is TrustedOwnedSource and digest(contract)==source_owner.ledger.contract_sha256,
                "exact source owner/contract required")
            source_owner.require_inputs(source_owner.ledger,archive,source)
        self.archive = archive
        self.source, self.contract = deepcopy(source), deepcopy(contract)
        self.root_id, self.configurations = root_id, tuple(configurations)
        self.selections, self.readers = deepcopy(selections), dict(selector_readers)
        self.allow_synthetic = allow_synthetic
        self.bound = archived_exploration_root_binding(archive,self.source,contract=self.contract,root_id=root_id)
        names = {selector_key(c) for c in self.configurations}
        require(len(names) == len(self.configurations) and set(self.selections) == names == set(self.readers)
            and sum(c.arm == "raw" for c in self.configurations) == 1, "exact unique stage-local owned selector roster required")
        for cfg in self.configurations:
            reader = self.readers[selector_key(cfg)]
            expected = _selector_reader_class(cfg,getattr(reader,"runtime",None))
            require(type(reader) is expected,"actual exact stage-local owning reader required")
        has_oracle=any(c.belief == "oracle" for c in self.configurations)
        from ..randbat import Gen3RandbatSource
        require((has_oracle and type(oracle_set_source) is Gen3RandbatSource)
            or (not has_oracle and oracle_set_source is None),
            "A1 oracle requires an explicit private catalog; public defaults carry none")
        self._oracle_source=deepcopy(oracle_set_source)
        self._oracle,self._oracle_projection,self._oracle_override_sha=None,None,None
        if has_oracle:
            # Establish trusted origin before the first callback-bearing owner read.
            self._oracle,self._oracle_projection=self._extract_original_team()
            from ..env import BattleStartOverride
            from ..showdown_fixture import pack_team
            from .search_over_raw_oracle import opening_team
            root=next(r for r in self.source["roots"] if r["root_id"] == root_id)
            context,_,_=archive.selected(root)
            own=opening_team(context.public_materialization_state.self_initial_request,self._oracle.subject,self._oracle_source)
            opponent="p2" if self._oracle.subject == "p1" else "p1"
            self._oracle_override_sha=digest(asdict(BattleStartOverride(
                player_teams={self._oracle.subject:pack_team(own),opponent:pack_team(self._oracle.opponent_team)},
                observation_format_id="gen3randombattle")))
        self._frozen = self._inputs()
        self._prepared = {}
        self.revalidate()

    def __getstate__(self):
        raise TypeError("private root evaluation plan cannot be persisted or sent to a selector")

    def _inputs(self):
        return digest(dict(source=self.source, contract=self.contract, root=self.root_id, bound=self.bound,
            source_owner=None if self.source_owner is None else dict(identity=id(self.source_owner),
                class_name=(type(self.source_owner).__module__,type(self.source_owner).__qualname__),
                inputs=self.source_owner._identity()),
            configurations=[asdict(c) for c in self.configurations], selections=self.selections,
            reader_inputs={name:reader._inputs() for name,reader in self.readers.items()},
            original_team=_state_payload(self._oracle),original_projection=self._oracle_projection,
            original_catalog=_state_payload(self._oracle_source),native_original_override=self._oracle_override_sha,
            allow_synthetic=self.allow_synthetic))

    def _extract_original_team(self):
        from .search_over_raw_archive import capture_archived_team_oracle
        return capture_archived_team_oracle(self.archive,self.source,contract=self.contract,
            root_id=self.root_id,set_source=self._oracle_source)

    def _validate_source_owner(self):
        require(self.source_owner is self._original_source_owner,"original root-plan source owner replaced")
        if self.source_owner is None:return
        from .search_over_raw_source_run import TrustedOwnedSource
        owner=self.source_owner
        require(type(owner) is TrustedOwnedSource and digest(self.source)==digest(owner.source)
            and digest(self.contract)==owner.ledger.contract_sha256,
            "root plan differs from original owned source/contract")
        owner.require_inputs(owner.ledger,self.archive,owner.source)

    def _check_original_team(self):
        if self._oracle is None:return
        oracle,projection=self._extract_original_team()
        require(digest(_state_payload(oracle)) == digest(_state_payload(self._oracle))
            and digest(projection) == digest(self._oracle_projection)
            and digest(projection["public_request_witness"]) == digest(self.bound["public_request_witness"]),
            "original archived A1 team/public/catalog changed")

    def _read_selector(self, cfg, selected, binding):
        name = selector_key(cfg); reader = self.readers[name]
        require(type(reader) is _selector_reader_class(cfg,getattr(reader,"runtime",None)), "owning reader replaced after preparation")
        witness = self.bound["public_request_witness"]
        require(digest(binding) == digest(self.bound["public_evidence_binding"])
            and reader.configuration == cfg and reader.root_id == self.root_id+":"+name
            and reader.statistics_root_id == self.root_id and type(reader.seed) is int
            and reader.seed == self.bound["slot"]["source_seed"]
            and digest(reader.consumed) == digest(witness["consumed_binding"])
            and tuple(reader.mask) == tuple(binding["legal_action_mask"])
            and reader.runtime_sha == selected["runtime_sha256"], "archive/actual owner public-root binding drift")
        if cfg.arm == "raw":
            require(reader.public_request_sha == witness["public_request_sha256"],
                "owned raw control differs from archive's full public request")
        elif cfg.belief == "oracle":
            self._check_original_team()
            require(digest(reader.oracle_receipt) == digest(self._oracle.receipt())
                and digest(reader.public_request) == digest(witness)
                and reader.catalog_sha == self._oracle_projection["reconstruction_catalog_sha256"],
                "A1 owning reader differs from original archive truth/public/catalog")
            if cfg.arm == "incumbent":
                require(reader.override_sha == self._oracle_override_sha
                    and reader.timed_public_request_sha == witness["public_request_sha256"],
                    "A1 native override/full request differs from original archive")
            else:
                require(reader.oracle_sha == digest(_state_payload(self._oracle))
                    and reader.reference_public_request_sha == witness["public_request_sha256"],
                    "A1 reference full team/request differs from original archive")
        elif cfg.arm == "incumbent" and getattr(reader,"timed_public_request_sha",None) is not None:
            require(reader.timed_public_request_sha == witness["public_request_sha256"],
                "owned timed native differs from archive's full public request")
        elif type(reader) is ReferenceTimedRunValidator and reader.reference_public_request_sha is not None:
            require(reader.reference_public_request_sha == witness["public_request_sha256"],
                "owned timed reference differs from archive's full public request")
        validated = reader(selected)
        require(validated == digest(selected) and reader.last_validation is not None
            and reader.last_validation["selected_sha256"] == validated
            and reader.last_validation["process_receipts_validated"] is True
            and (reader.last_validation["synthetic_process_fixture"] is False or self.allow_synthetic),
            "selector handoff requires exact normal owned evidence; synthetic evidence is not production evidence")
        return validated

    def revalidate(self):
        require(self._inputs() == self._frozen, "frozen root evaluation inputs changed")
        self._validate_source_owner()
        self._check_original_team()
        actual = archived_exploration_root_binding(self.archive,self.source,contract=self.contract,root_id=self.root_id)
        require(digest(actual) == digest(self.bound), "actual archived public root/private snapshot changed")
        binding = self.bound["public_evidence_binding"]
        contract = dict(root_binding=deepcopy(binding), root_id=self.root_id, namespace=self.contract["namespace"],
            subject="p1", max_boundaries=250, replicates=8, selection_seed=self.bound["slot"]["source_seed"],
            policy="raw_argmax_after_initial_sampled_opponent_reply")
        bundle = dict(root_binding=deepcopy(binding), continuation_contract=contract,
            selections=deepcopy(self.selections), outcomes=[])
        owners = {}
        def read_selector(cfg, selected, binding):
            validated = self._read_selector(cfg,selected,binding)
            name = selector_key(cfg); reader = self.readers[name]
            # Preserve what THIS reader actually validated before invoking a
            # later owner. A later callback cannot establish a new baseline.
            owners[name] = dict(readout=deepcopy(reader.last_validation),
                inventory=sorted(str(p.relative_to(reader.directory/"owned-groups"))
                    for p in (reader.directory/"owned-groups").rglob("*") if p.is_file()))
            return validated
        evidence = reconcile_root_evidence(bundle, namespace=self.contract["namespace"], slot=self.bound["slot"],
            configurations=self.configurations, public_record_sha256=self.bound["public_record_sha256"],
            runtime_hashes={name:reader.runtime_sha for name,reader in self.readers.items()},
            validate_selector=read_selector, validate_outcome=lambda row,law:None,
            allow_unqualified_at_ceiling=True)
        # Extraction can invoke callbacks. Finish it BEFORE checking every
        # owner's original readout and durable bytes, never after those checks.
        self._check_original_team()
        final_bound=archived_exploration_root_binding(self.archive,self.source,
            contract=self.contract,root_id=self.root_id)
        self._validate_source_owner()  # Before original selector readout/file guards.
        for name, frozen in owners.items():
            reader = self.readers[name]
            reader._files()
            owned = reader.directory/"owned-groups"
            require(digest(reader.last_validation) == digest(frozen["readout"])
                and hashlib.sha256((reader.directory/"job.pickle").read_bytes()).hexdigest() == reader.job_sha
                and owned.is_dir() and sorted(str(p.relative_to(owned)) for p in owned.rglob("*")
                    if p.is_file()) == frozen["inventory"]
                and all((reader.directory/path).resolve().is_relative_to(reader.directory)
                    and hashlib.sha256((reader.directory/path).read_bytes()).hexdigest() == sha
                    for path,sha in frozen["readout"]["receipt_file_sha256"].items()),
                "earlier owner receipts changed during later selector validation")
        require(self._inputs() == self._frozen
            and digest(actual) == digest(final_bound), "archive or owners changed during handoff")
        require(self._inputs() == self._frozen,"original A1 inputs changed during final origin check")
        correspondence = {selector_key(cfg):cfg.arm == "raw" or
            ((cfg.arm == "incumbent"
                and getattr(self.readers[selector_key(cfg)],"timed_public_request_sha",None) is not None
                or type(self.readers[selector_key(cfg)]) in (ReferenceTimedRunValidator,ReferenceOracleTimedRunValidator)
                and self.readers[selector_key(cfg)].reference_public_request_sha is not None)
                and owners[selector_key(cfg)]["readout"].get("actual_work",{}).get(
                    "full_public_request_consumption_validated") is True) for cfg in self.configurations}
        return dict(schema="pokezero.search-over-raw.archived-root-evaluation-plan.v1",
            plan_sha256=self._frozen, root_evidence=evidence, continuation_contract=contract,
            selector_owning_readout_sha256={name:digest(frozen["readout"]) for name,frozen in owners.items()},
            action_aliases={str(action):sorted(name for name,value in evidence["actions"].items() if value == action)
                for action in sorted(set(evidence["actions"].values()))},
            continuation_cells=[dict(action=action,replicate=rep) for action in sorted(set(evidence["actions"].values()))
                for rep in range(8)], unqualified_has_no_action_or_continuation=True,
            synthetic_fixture_permitted=self.allow_synthetic,
            full_public_correspondence_by_selector=correspondence,
            original_team_archive_correspondence_by_selector={selector_key(c):
                correspondence[selector_key(c)] and owners[selector_key(c)]["readout"].get("actual_work",{}).get(
                    "original_team_consumption",{}).get("prepared",False)
                if c.arm == "incumbent" else correspondence[selector_key(c)] and
                    owners[selector_key(c)]["readout"].get("actual_work",{}).get("original_team_consumption",{}).get(
                        "original_team_preparation_validated",False)
                for c in self.configurations if c.belief == "oracle"},
            full_timed_public_correspondence_validated=any(cfg.arm != "raw" for cfg in self.configurations)
                and all(correspondence.values()),
            source_freshness_validated=False, runtime_qualified=False, execution_ready=False,
            owned_source_pipeline_validated=self.source_owner is not None,
            collection_authorized=False, holdout_authorized=False)

    def prepare_continuation_job(self, *, config, runtime, actions_to_collect=None, _synthetic_job_class=None):
        """Prepare a frozen selected-action subset; never launch or fill old gaps."""
        receipt = self.revalidate()
        actions = receipt["root_evidence"]["actions"]
        targets = tuple(sorted(set(actions.values()))) if actions_to_collect is None else actions_to_collect
        require(type(targets) is tuple and targets and all(type(a) is int for a in targets)
            and tuple(sorted(set(targets))) == targets and set(targets) <= set(actions.values()),
            "nonempty exact sorted selected-action subset required")
        aliases = {name:action for name,action in actions.items() if action in targets}
        root = next(r for r in self.source["roots"] if r["root_id"] == self.root_id)
        _, _, snapshot = self.archive.selected(root)
        # paired_continuations deduplicates action values and emits action cells,
        # while these aliases preserve each selector's separate contrast.
        job_class = OwnedContinuationJob
        if _synthetic_job_class is not None:
            require(self.allow_synthetic and isinstance(_synthetic_job_class,type)
                and issubclass(_synthetic_job_class,OwnedContinuationJob)
                and getattr(_synthetic_job_class,"synthetic_process_fixture",False) is True,
                "synthetic continuation substitution requires explicit fixture scope")
            job_class = _synthetic_job_class
        job = job_class(deepcopy(config),deepcopy(snapshot),
            deepcopy(self.bound["public_evidence_binding"]),deepcopy(runtime),"p1",
            deepcopy(aliases),self.contract["namespace"])
        frozen = digest(job.input_binding())
        job_sha = hashlib.sha256(pickle.dumps(job)).hexdigest()
        require(receipt == self.revalidate(), "root evaluation changed during continuation preparation")
        require(digest(job.input_binding()) == frozen and hashlib.sha256(pickle.dumps(job)).hexdigest() == job_sha,
            "continuation job changed during final preparation validation")
        self._prepared[id(job)] = dict(inputs=frozen,pickle_sha256=job_sha,job_class=type(job),target_actions=targets)
        return job, receipt

    def verify_prepared_job(self, job):
        """Use immediately before launch AND acceptance under separate authority."""
        prepared = self._prepared.get(id(job))
        require(prepared is not None and type(job) is prepared["job_class"]
            and (type(job) is OwnedContinuationJob or self.allow_synthetic
                and getattr(job,"synthetic_process_fixture",False) is True)
            and prepared["inputs"] == digest(job.input_binding())
            and prepared["pickle_sha256"] == hashlib.sha256(pickle.dumps(job)).hexdigest(),
            "unprepared or changed continuation job")
        receipt = self.revalidate()
        targets = prepared["target_actions"]
        actions = receipt["root_evidence"]["actions"]
        require(prepared["inputs"] == digest(job.input_binding())
            and prepared["pickle_sha256"] == hashlib.sha256(pickle.dumps(job)).hexdigest()
            and type(targets) is tuple and targets and all(type(a) is int for a in targets)
            and tuple(sorted(set(targets))) == targets and set(targets) <= set(actions.values())
            and digest(job.root_binding) == digest(self.bound["public_evidence_binding"])
            and job.actions == {name:a for name,a in actions.items() if a in targets} and job.subject == "p1"
            and job.namespace == self.contract["namespace"], "prepared root/actions/continuation law drift")
        return receipt  # Never an admission grant.

    def continuation_attempt_binding(self, job, *, run_contract=ContinuationRunContract()):
        """Prospective local attempt identity, not a registration/admission grant."""
        self.verify_prepared_job(job)
        require(type(run_contract) is ContinuationRunContract, "distinct continuation run contract required")
        return digest(dict(schema="pokezero.search-over-raw.planned-continuation-attempt.v1",
            plan_sha256=self._frozen, job_sha256=self._prepared[id(job)]["pickle_sha256"],
            job_inputs=job.input_binding(), run_contract=run_contract.receipt(), collection_authorized=False))

    def prepare_new_continuation_job(self, *, donor_banks, config, runtime, _synthetic_job_class=None):
        """Read donors and prepare only new action values, without launch authority.

        A targeted action remains consumed even with capped/missing cells. An
        empty new-action set returns no job, not an empty scientific attempt.
        """
        require(type(donor_banks) is tuple and all(type(b) is OwnedContinuationBank for b in donor_banks),
            "exact immutable donor-bank roster required")
        plans = _unique_objects((self,)+tuple(b.plan for b in donor_banks))
        before = _owner_inputs(plans,donor_banks)
        receipt = self.revalidate()
        readouts = {reader:digest(reader.last_validation) for reader in self.readers.values()}
        consumed = set()
        for bank in donor_banks:
            _compatible_bank(self,bank)
            require(digest(bank.job.runtime) == digest(runtime), "continuation runtime differs from donor")
            targets = set(bank.plan._prepared[id(bank.job)]["target_actions"])
            require(not consumed & targets, "donor target actions overlap; no redraw or gap filling")
            consumed.update(targets)
            accepted = bank.plan.accept_continuation_result(bank.job,bank.result,owning_reader=bank.reader)
            require(not accepted["synthetic_process_fixture"] or self.allow_synthetic,
                "synthetic donor cannot enter production preparation")
            require(all(digest(reader.last_validation) == sha for reader,sha in readouts.items()),
                "earlier owned readout changed during donor preparation")
            for reader in _owner_readers((bank.plan,),(bank,)):
                readouts.setdefault(reader,digest(reader.last_validation))
        require(_owner_inputs(plans,donor_banks) == before, "donor/current ownership changed during preparation")
        targets = tuple(sorted(set(receipt["root_evidence"]["actions"].values()) - consumed))
        if targets:
            job,_ = self.prepare_continuation_job(config=config,runtime=runtime,actions_to_collect=targets,
                _synthetic_job_class=_synthetic_job_class)
        else: job = None
        require(_owner_inputs(plans,donor_banks) == before,
            "donor/current ownership changed during new-action preparation")
        require(all(digest(reader.last_validation) == sha for reader,sha in readouts.items()),
            "owned readout changed during final new-action preparation")
        return job,dict(schema="pokezero.search-over-raw.new-action-preparation.v1",
            root_id=self.root_id,new_action_values=list(targets),reused_target_action_values=sorted(consumed),
            no_job_required=job is None,missing_cells_not_retried=True,
            donor_roster_exhaustive=False,execution_ready=False,
            runtime_qualified=False,collection_authorized=False,holdout_authorized=False)

    def accept_continuation_result(self, job, result, *, owning_reader):
        """Read a supplied normally closed owned run; never launch or retry.

        Join the full producer result to its exact prepared job and durable
        terminal, then independently validate observed protocol per cell. Missing
        or unvalidated cells remain uncertain on the full planned action roster.
        This neither admits stage execution nor qualifies evaluator/dynamics.
        """
        prepared = self.verify_prepared_job(job)
        require(type(owning_reader) is OwningRunValidator, "actual continuation owning reader required")
        reader = owning_reader
        def reader_identity():
            return digest(dict(directory=str(reader.directory.resolve()),job_sha256=reader.job_sha,
                binding=reader.binding,contract=reader.contract,allow_synthetic=reader.allow_synthetic,
                root=reader.root,snapshot=snapshot_identity(reader.snapshot),runtime=reader.runtime))
        reader_identity_sha = reader_identity()
        require(reader.job_sha == self._prepared[id(job)]["pickle_sha256"]
            and reader.binding == self.continuation_attempt_binding(job,
                run_contract=ContinuationRunContract(reader.contract["emergency_watchdog_seconds"]))
            and digest(reader.root) == digest(job.root_binding)
            and digest(snapshot_identity(reader.snapshot)) == digest(snapshot_identity(job.snapshot))
            and digest(reader.runtime) == digest(job.runtime), "prepared job/owning continuation reader drift")
        original_sha = digest(result); saved = deepcopy(result); expected = job.input_binding()
        require(type(saved) is dict and set(saved) == set(expected) | {"bank", "bank_sha256", "outcomes",
            "synthetic_process_fixture", "runtime_qualified", "resource_cost_complete"}
            and digest({key:saved[key] for key in expected}) == digest(expected)
            and saved["bank_sha256"] == digest(saved["bank"])
            and saved["runtime_qualified"] is False and saved["resource_cost_complete"] is False
            and type(saved["synthetic_process_fixture"]) is bool
            and (saved["synthetic_process_fixture"] is False or self.allow_synthetic),
            "exact prepared continuation result required")
        validator = OutcomeReceiptValidator(saved["bank"],expected_snapshot=job.snapshot,
            root_binding=job.root_binding,runtime_sha256=digest(job.runtime),
            validate_provenance_and_cleanup=reader)
        require(reader.last_validation is not None and reader.last_validation["process_receipts_validated"] is True
            and (reader.last_validation["synthetic_process_fixture"] is False or self.allow_synthetic),
            "normally owned continuation evidence required")
        owning_readout = deepcopy(reader.last_validation)
        require(saved["synthetic_process_fixture"] is owning_readout["synthetic_process_fixture"]
            and saved["synthetic_process_fixture"] is getattr(job,"synthetic_process_fixture",False),
            "prepared job/producer/owning synthetic scope mismatch")
        terminal_path = reader.directory/"terminal.json"
        terminal_bytes = terminal_path.read_bytes()
        require(hashlib.sha256(terminal_bytes).hexdigest() == owning_readout["receipt_file_sha256"]["terminal.json"]
            and digest(json.loads(terminal_bytes)["result"]) == original_sha,
            "supplied outcomes differ from actual owned producer result")
        outcomes = saved["outcomes"]
        require(type(outcomes) is dict and set(outcomes) == {"root_id","actions","outcomes","status"}
            and outcomes["root_id"] == self.root_id and digest(outcomes["actions"]) == digest(job.actions)
            and type(outcomes["outcomes"]) is list and outcomes["status"] in {"COMPLETE","UNCERTAIN"},
            "actual continuation action aliases/root/row roster drift")
        keys = {(cell["law"]["action"],cell["law"]["replicate"]) for cell in saved["bank"]["cells"]}
        planned_keys = {(action,rep) for action in self._prepared[id(job)]["target_actions"] for rep in range(8)}
        require(keys <= planned_keys, "captured continuation outside frozen selected actions")
        rows = [dict(root_id=self.root_id,**deepcopy(row)) for row in outcomes["outcomes"]]
        row_keys = {(row["action"],row["replicate"]) for row in rows}
        require(row_keys <= planned_keys and len(row_keys) == len(rows),
            "returned continuation outside frozen target action cells")
        require(outcomes["status"] == ("COMPLETE" if row_keys == planned_keys
            and all(row["status"] == "COMPLETE" for row in rows) else "UNCERTAIN"),
            "continuation aggregate status contradicts the full fixed roster")
        binding = self.bound["public_evidence_binding"]
        bundle = dict(root_binding=deepcopy(binding),continuation_contract=deepcopy(prepared["continuation_contract"]),
            selections=deepcopy(self.selections),outcomes=rows)
        # Freeze per-cell bank and result before re-reading selectors. Their
        # callbacks cannot replace the previously validated outcome evidence.
        reconciled = reconcile_root_evidence(bundle,namespace=self.contract["namespace"],slot=self.bound["slot"],
            configurations=self.configurations,public_record_sha256=self.bound["public_record_sha256"],
            runtime_hashes={name:r.runtime_sha for name,r in self.readers.items()},
            validate_selector=self._read_selector,validate_outcome=validator,allow_unqualified_at_ceiling=True)
        require(prepared == self.verify_prepared_job(job), "plan changed during continuation acceptance")
        reader._check_inputs();verify_outcome_runtime(reader.runtime,job.config)
        require(reader_identity() == reader_identity_sha
            and digest(result) == original_sha and digest(reader.last_validation) == digest(owning_readout)
            and hashlib.sha256((reader.directory/"job.pickle").read_bytes()).hexdigest() == reader.job_sha
            and all(hashlib.sha256((reader.directory/name).read_bytes()).hexdigest() == sha
                for name,sha in owning_readout["receipt_file_sha256"].items()),
            "continuation result/owning evidence changed during final acceptance")
        return dict(schema="pokezero.search-over-raw.accepted-root-continuations.v1",
            plan_sha256=self._frozen,job_sha256=reader.job_sha,attempt_binding_sha256=reader.binding,
            producer_result_sha256=original_sha,owning_readout_sha256=digest(owning_readout),
            root_evidence=reconciled,continuation_contract=deepcopy(prepared["continuation_contract"]),
            synthetic_process_fixture=saved["synthetic_process_fixture"],
            evaluator_correctness_established=False,dynamics_correctness_established=False,
            runtime_qualified=False,execution_ready=False,collection_authorized=False,holdout_authorized=False)


@dataclass(frozen=True)
class OwnedContinuationBank:
    """Private supplied bank and its original preparing plan, never a digest grant."""
    plan: ArchivedRootEvaluationPlan
    job: OwnedContinuationJob
    result: dict
    reader: OwningRunValidator

    def __post_init__(self):
        require(type(self.plan) is ArchivedRootEvaluationPlan and type(self.result) is dict
            and type(self.reader) is OwningRunValidator, "actual preparing plan/result/owner required")

    def __getstate__(self):
        raise TypeError("private owned continuation bank cannot be persisted or sent to a selector")


def _unique_objects(objects):
    return tuple({id(obj):obj for obj in objects}.values())


def _archive_identity(plan):
    from .search_over_raw_archive import retained_archive_root_identity
    return retained_archive_root_identity(plan.archive,plan.bound["source_request_index"])


def _bank_identity(bank):
    job,result,reader = bank.job,bank.result,bank.reader
    saved = bank.plan._prepared.get(id(job))
    return dict(bank_identity=id(bank),plan_identity=id(bank.plan),job_identity=id(job),
        job_class=(type(job).__module__,type(job).__qualname__),inputs=job.input_binding(),
        job_sha=hashlib.sha256(pickle.dumps(job)).hexdigest(),
        prepared=None if saved is None else dict(inputs=saved["inputs"],pickle_sha256=saved["pickle_sha256"],
            job_class=(saved["job_class"].__module__,saved["job_class"].__qualname__),
            target_actions=saved["target_actions"]),
        result_identity=id(result),result_sha=digest(result),reader_identity=id(reader),
        directory=str(reader.directory.resolve()),job=reader.job_sha,binding=reader.binding,
        contract=reader.contract,allow_synthetic=reader.allow_synthetic,root=reader.root,
        snapshot=snapshot_identity(reader.snapshot),runtime=reader.runtime)


def _owner_readers(plans,banks):
    return _unique_objects(tuple(r for p in plans for r in p.readers.values())+tuple(b.reader for b in banks))


def _owner_inputs(plans,banks):
    identities = dict(plans=[dict(identity=id(p),archive_identity=id(p.archive),inputs=p._inputs(),
        archive=_archive_identity(p),readers={n:(id(r),type(r).__module__,type(r).__qualname__,r._inputs())
            for n,r in p.readers.items()}) for p in plans],banks=[_bank_identity(b) for b in banks])
    inventories = {}
    directories=_unique_objects(tuple(reader.directory for reader in _owner_readers(plans,banks))+
        tuple(p.source_owner.directory for p in plans if p.source_owner is not None))
    for owner_directory in directories:
        directory = owner_directory.resolve()
        paths = sorted(p for p in directory.rglob("*") if p.is_file())
        require(all(not p.is_symlink() and p.resolve().is_relative_to(directory) for p in paths),
            "owned stage file inventory must remain inside actual owner")
        inventories[str(directory)] = {str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}
    return digest(identities),inventories


def _compatible_bank(plan,bank):
    donor = bank.plan
    require(type(bank) is OwnedContinuationBank and digest(donor.contract) == digest(plan.contract)
        and donor.root_id == plan.root_id and digest(donor.bound) == digest(plan.bound)
        and digest(donor.selections["raw"]) == digest(plan.selections["raw"])
        and donor.readers["raw"]._inputs() == plan.readers["raw"]._inputs(),
        "donor/current root, private snapshot, public request or actual raw owner differs")


class _OwnedStageInputs:
    """Private owner-to-ledger join; no supplied bundles or success digests.

    Construction only freezes supplied identities/bytes. All independent owner
    validation happens after the ledger's durable create-only claim. None means
    no continuation result supplied, not proof that no attempt was launched.
    A supplied failed or malformed attempt must be retained and rejected.
    """
    def __init__(self, ledger, stage, configuration, plans, continuations,source_panel=None,input_unavailable=None):
        from .search_over_raw import SearchConfiguration
        self.ledger,self.stage,self.configuration = ledger,stage,configuration
        self.source_panel=source_panel
        self._original_source_panel=source_panel
        self.original_unavailable=input_unavailable
        self.input_unavailable={} if input_unavailable is None else dict(input_unavailable)
        self._unavailable_sha=digest(self.input_unavailable)
        from .search_over_raw_selector_producer import _actor_unavailable_path
        self._unavailable_files={str(_actor_unavailable_path(ledger,root,configuration)):
            hashlib.sha256(_actor_unavailable_path(ledger,root,configuration).read_bytes()).hexdigest()
            for root in self.input_unavailable}
        cfg = next((c for c in ledger._stage_configs(stage) if c.identity == configuration),None)
        if ledger.contract["execution_ready"]:
            require(source_panel is not None,"production stage requires the complete fixed owned source roster")
        if source_panel is not None:
            from .search_over_raw_source_run import TrustedExplorationPanel
            require(type(source_panel) is TrustedExplorationPanel,"exact retained exploration panel required")
            source_panel.require_stage_inputs(ledger,plans,input_unavailable=self.input_unavailable,configuration=cfg)
            require(digest(self.input_unavailable) == self._unavailable_sha, "original unavailable partition changed")
        require(not self.input_unavailable or source_panel is not None,
            "unavailable roots require the original complete owned source panel")
        self._source_panel_sha=None if source_panel is None else source_panel.frozen
        cfg = next((c for c in ledger._stage_configs(stage) if c.identity == configuration),None)
        require(cfg is not None and ((cfg.belief == "public" and cfg.leaf in {"model","hp_fraction"})
            or stage == "A1" and cfg.belief == "oracle" and cfg.leaf == "model" and cfg.seconds == 1.),
            "owned ingestion requires stage-local public model/HP or explicit A1 original-team configuration")
        require(type(plans) is dict and plans and type(continuations) is dict
            and set(plans) == set(continuations), "exact supplied plan/continuation root roster required")
        self.original_plans,self.original_continuations = plans,continuations
        self.plans,self.continuations = dict(plans),dict(continuations)
        self.banks,self.owner_plans = {},{}
        self.cfg = cfg; self.raw = SearchConfiguration("raw",seconds=1.)
        stage_configs = set(ledger._stage_configs(stage)) | {self.raw}
        self.bundles,self.public_bindings,self.runtime_hashes = {},{},None
        self.validated,self.outcome_readers,self.readouts = {},{},{}
        oracle_recipe = None
        raw_runtime_sha = None
        slots = {s["root_id"] for s in ledger.contract["panels"]["exploration"]["root_slots"]}
        for root,plan in self.plans.items():
            require(type(plan) is ArchivedRootEvaluationPlan and root == plan.root_id and root in slots
                and digest(plan.contract) == digest(ledger.contract)
                and {self.raw,cfg} <= set(plan.configurations) <= stage_configs,
                "exact archived stage-local root/configuration/contract required")
            if ledger.contract["execution_ready"]:
                require(plan.source_owner is not None and plan.source_owner.ledger is ledger,
                    "production stage requires original owned source provenance")
            hashes = {selector_key(c):plan.readers[selector_key(c)].runtime_sha for c in (self.raw,cfg)}
            if cfg.belief == "oracle":
                require(raw_runtime_sha is None or raw_runtime_sha == hashes["raw"],
                    "A1 original-team roots require one common raw runtime")
                raw_runtime_sha = hashes["raw"]
                from .search_over_raw_reconciled import selector_runtime_recipe
                recipe = selector_runtime_recipe(cfg,plan.readers[cfg.identity].runtime)
                # Only original-team truth varies by root. Checkpoint, search,
                # allocation, source/catalog and observation recipe stay fixed.
                require(oracle_recipe is None or digest(oracle_recipe) == digest(recipe),
                    "A1 per-root runtime may vary only in original-team truth")
                oracle_recipe = recipe
                if self.runtime_hashes is None:self.runtime_hashes={}
                self.runtime_hashes[root]=hashes
            else:
                require(self.runtime_hashes is None or self.runtime_hashes == hashes,
                    "public stage roots require the same reviewed selector runtime")
                self.runtime_hashes = hashes
            data = self.continuations[root]
            if data is None: banks = ()
            elif type(data) is tuple and len(data) == 3 and type(data[2]) is OwningRunValidator:
                banks = (OwnedContinuationBank(plan,*data),)
            else:
                require(type(data) is tuple and data and all(type(b) is OwnedContinuationBank for b in data),
                    "supplied actual continuation bank roster required")
                banks = data
            self.banks[root] = banks
            self.owner_plans[root] = _unique_objects((plan,)+tuple(b.plan for b in banks))
            # Raw rows are reconstructed, never taken from an earlier accepted
            # readout. The owner/producer join is checked after the claim.
            producer_rows = []
            for bank in banks:
                payload = bank.result.get("outcomes")
                supplied = payload.get("outcomes",[]) if type(payload) is dict else []
                if type(supplied) is list: producer_rows.extend(supplied)
            # Preserve malformed attempted results for protected owner rejection,
            # rather than failing before the durable stage claim exists.
            projected_actions = {value for c in (self.raw,cfg)
                if type(value := plan.selections[selector_key(c)].get("action")) is int}
            rows = [dict(deepcopy(row),root_id=root) for row in producer_rows
                if type(row) is dict and type(row.get("action")) is int
                and row["action"] in projected_actions] if type(producer_rows) is list else []
            binding = deepcopy(plan.bound["public_evidence_binding"])
            law = dict(root_binding=deepcopy(binding),root_id=root,namespace=ledger.contract["namespace"],
                subject="p1",max_boundaries=250,replicates=8,selection_seed=plan.bound["slot"]["source_seed"],
                policy="raw_argmax_after_initial_sampled_opponent_reply")
            self.bundles[root] = dict(root_binding=binding,continuation_contract=law,
                selections={selector_key(c):deepcopy(plan.selections[selector_key(c)])
                    for c in (self.raw,cfg)},outcomes=rows)
            self.public_bindings[root] = plan.bound["public_record_sha256"]
        self.frozen = {root:self._identity(root) for root in self.plans}

    def _identity(self, root):
        return _owner_inputs(self.owner_plans[root],self.banks[root])

    def _validate_root(self, root):
        if root in self.validated: return
        plan = self.plans[root]
        receipt = plan.revalidate()
        require(receipt["full_timed_public_correspondence_validated"] is True,
            "owned stage requires actual full archived public request correspondence")
        if self.cfg.belief == "oracle":
            require(receipt["original_team_archive_correspondence_by_selector"].get(self.cfg.identity) is True,
                "A1 stage requires original archive team correspondence for this selector")
        full = receipt["root_evidence"]
        validated_readers = dict((id(r),r) for r in plan.readers.values())
        baseline = {key:digest(r.last_validation) for key,r in validated_readers.items()}
        validators = {}; runtime_sha = None
        for bank in self.banks[root]:
            _compatible_bank(plan,bank)
            job,result,reader = bank.job,bank.result,bank.reader
            require(type(job.actions) is dict and job.actions and all(type(a) is int for a in job.actions.values()),
                "exact target action values required")
            targets = set(job.actions.values())
            require(not set(validators) & targets, "continuation target actions overlap; no redraw or gap filling")
            require(runtime_sha is None or runtime_sha == digest(job.runtime), "composed continuation runtime differs")
            runtime_sha = digest(job.runtime)
            accepted_bank = bank.plan.accept_continuation_result(job,result,owning_reader=reader)
            require(not accepted_bank["synthetic_process_fixture"] or plan.allow_synthetic,
                "synthetic donor cannot enter production stage")
            require({key:digest(r.last_validation) for key,r in validated_readers.items()} == baseline,
                "earlier owned readout changed during later bank validation")
            for r in tuple(bank.plan.readers.values())+(reader,):
                if id(r) not in validated_readers:
                    validated_readers[id(r)] = r; baseline[id(r)] = digest(r.last_validation)
            validator = OutcomeReceiptValidator(result["bank"],expected_snapshot=job.snapshot,
                root_binding=job.root_binding,runtime_sha256=runtime_sha,validate_provenance_and_cleanup=reader)
            for action in targets: validators[action] = validator
        def route_outcome(row,law):
            validator = validators.get(row["action"])
            return None if validator is None else validator(row,law)
        # Full owners are validated before projecting current aliases. Donor
        # aliases/configurations may legitimately come from an earlier stage.
        accepted = reconcile_root_evidence(self.bundles[root],namespace=plan.contract["namespace"],
            slot=plan.bound["slot"],configurations=(self.raw,self.cfg),
            public_record_sha256=plan.bound["public_record_sha256"],runtime_hashes=
                self.runtime_hashes[root] if self.cfg.belief == "oracle" else self.runtime_hashes,
            validate_selector=lambda cfg,selected,binding:full["selected_sha256"][selector_key(cfg)],
            validate_outcome=route_outcome,
            allow_unqualified_at_ceiling=True)
        require({key:digest(r.last_validation) for key,r in validated_readers.items()} == baseline,
            "full owned readout changed during contrast projection")
        self.readouts[root] = baseline
        self.validated[root] = deepcopy(accepted)
        self.outcome_readers[root] = route_outcome

    def validate_selector(self, cfg, selected, binding):
        root = binding["root_id"]
        require(root in self.plans and cfg in (self.raw,self.cfg)
            and digest(selected) == digest(self.bundles[root]["selections"][selector_key(cfg)])
            and digest(binding) == digest(self.bundles[root]["root_binding"]),
            "stage callback differs from actual frozen selector/root")
        self._validate_root(root)
        plan = self.plans[root]
        result = plan._read_selector(cfg,selected,binding)
        # Freeze THIS owner's return immediately. A later selector cannot make
        # a mutated earlier readout become the accepted baseline.
        reader = plan.readers[selector_key(cfg)]
        self.readouts[root][id(reader)] = digest(reader.last_validation)
        return result

    def validate_outcome(self, row, law):
        root = row["root_id"]
        require(root in self.validated and digest(law) == digest(self.bundles[root]["continuation_contract"])
            and digest(row) in {digest(r) for r in self.bundles[root]["outcomes"]},
            "stage callback differs from actual owned outcome/law")
        validator = self.outcome_readers[root]
        return validator(row,law)

    def finalize(self, rows):
        require(set(rows) == set(self.plans) == set(self.validated), "owned stage validation roster changed")
        # A generic caller cannot attach this scope using unvalidated digests.
        for root,row in rows.items():
            expected = deepcopy(self.validated[root])
            if row["schema"].endswith(".v1"):
                expected["schema"] = row["schema"]
                expected.pop("selector_statuses",None); expected.pop("unqualified_choice_is_uncertain",None)
            require(digest(row) == digest(expected), "stage reconciliation differs from actual owned root")
        require(self.source_panel is self._original_source_panel,"original stage source panel replaced")
        if self.source_panel is not None:self.source_panel.require_stage_inputs(self.ledger,self.plans,
            input_unavailable=self.input_unavailable,configuration=self.cfg)
        for plan in _unique_objects(tuple(p for group in self.owner_plans.values() for p in group)):
            plan._validate_source_owner()  # Finish source callbacks before global selector/bank guards.
        for root,plan in self.plans.items():
            for owner_plan in self.owner_plans[root]:
                for reader in owner_plan.readers.values(): reader._files()
            for bank in self.banks[root]:
                bank.reader._check_inputs(); verify_outcome_runtime(bank.reader.runtime,bank.job.config)
            readers = _owner_readers(self.owner_plans[root],self.banks[root])
            require({id(r):digest(r.last_validation) for r in readers} == self.readouts[root],
                "earlier owned readout changed during later stage callbacks")
        # All callback-bearing validation is over. Compare original references
        # and bytes globally, including the first root after the last one.
        if self.source_panel is not None:
            require(self.source_panel.frozen==self._source_panel_sha,"original stage panel baseline changed")
            self.source_panel._original_inputs()
        require(digest(self.input_unavailable) == self._unavailable_sha
            and (self.original_unavailable is None or digest(self.original_unavailable) == self._unavailable_sha),
            "original unavailable stage partition changed during final callbacks")
        require(set(self.original_plans) == set(self.plans)
            and set(self.original_continuations) == set(self.continuations)
            and all(self.original_plans[r] is p and self.original_continuations[r] is self.continuations[r]
                and self._identity(r) == self.frozen[r]
                for r,p in self.plans.items()), "owned stage inputs/evidence changed during ingestion")
        from .search_over_raw_ledger import _load,_write_new
        raw_provenance = {}
        for root,plan in self.plans.items():
            reader = plan.readers["raw"]
            receipt = dict(root_id=root,selected_sha256=rows[root]["selected_sha256"]["raw"],
                runtime_sha256=reader.runtime_sha,full_public_request_sha256=reader.public_request_sha,
                directory=str(reader.directory.resolve()),job_sha256=reader.job_sha,
                attempt_binding_sha256=reader.binding)
            relative = "owned-shared-raw/"+digest(dict(root_id=root))+".json"
            path = self.ledger.directory/relative
            if path.exists():
                require(_load(path) == receipt,"actual shared raw owner changed between consumers")
            else:
                path.parent.mkdir(parents=True,exist_ok=True);_write_new(path,receipt)
            raw_provenance[relative] = digest(receipt)
        # Equal labels are not proof of reused cells. Bind each consumed cell's
        # actual producer/capture, without a whole-bank all-stage prerequisite.
        provenance = {}
        for root,banks in self.banks.items():
            if not banks: continue
            owners = {action:bank for bank in banks for action in bank.job.actions.values()}
            for action in owners:
                bank = owners.get(action)
                if bank is None: continue
                result,reader = bank.result,bank.reader
                cells = {(cell["law"]["action"],cell["law"]["replicate"]):cell for cell in result["bank"]["cells"]}
                for rep in range(8):
                    cell = cells.get((action,rep))
                    receipt = dict(root_id=root,action=action,replicate=rep,
                        capture_sha256=None if cell is None else digest(cell),supplied=cell is not None,
                        job_sha256=reader.job_sha,attempt_binding_sha256=reader.binding,
                        directory=str(reader.directory.resolve()))
                    relative = "owned-shared-cells/"+digest(dict(root_id=root,action=action,replicate=rep))+".json"
                    path = self.ledger.directory/relative
                    if path.exists():
                        require(_load(path) == receipt,"actual shared continuation cell changed between consumers")
                    else:
                        path.parent.mkdir(parents=True,exist_ok=True);_write_new(path,receipt)
                    provenance[relative] = digest(receipt)
        # Epistemic absence is not an attempted continuation. Freeze this
        # consumer's scope after validating the original bank jobs; later first
        # attempts must neither upgrade nor invalidate an earlier uncertain row.
        attempted_actions = {root:sorted(set(rows[root]["actions"].values()) &
            {action for bank in banks for action in bank.job.actions.values()})
            for root,banks in self.banks.items()}
        result = dict(schema="pokezero.search-over-raw.owned-stage-inputs.v2",
            source_panel=None if self.source_panel is None else self.source_panel.receipt(),
            attempted_actions_by_root=attempted_actions,
            observed_roots=len(rows),full_public_transport_validated=True,
            actual_selector_owners_validated=True, supplied_continuation_owners_validated=True,
            original_team_archive_correspondence_validated=self.cfg.belief == "oracle",
            original_team_diagnostic_only=self.cfg.belief == "oracle",
            synthetic_process_fixture=any(r.last_validation["synthetic_process_fixture"]
                for root in self.plans for r in _owner_readers(self.owner_plans[root],self.banks[root])),
            unsupplied_continuation_roots=sorted(r for r,banks in self.banks.items() if not banks),
            unsupplied_attempts_proven_absent=False,runtime_qualified=False,
            shared_raw_provenance=raw_provenance,
            shared_cell_provenance=provenance,one_supplied_bank_per_root=False,
            new_actions_with_reused_prior_banks_supported=True,
            collection_authorized=False,holdout_authorized=False)
        if self.input_unavailable:
            result["input_unavailable_by_root"]=deepcopy(self.input_unavailable)
        require(digest(self.input_unavailable) == self._unavailable_sha
            and (self.original_unavailable is None or digest(self.original_unavailable) == self._unavailable_sha)
            and all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha
                for path,sha in self._unavailable_files.items()), "unavailable inputs changed after final stage callback")
        return result
