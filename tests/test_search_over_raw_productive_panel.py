"""Actual full-panel ownership with authored outcomes; no scientific work.

Only immutable fixture inputs and typed synthetic job bodies are authored.
Source capture, job preparation, supervision, readers, plans, journals, stage
ingestion and irreversible freezes are the actual implementation.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.local_showdown import LocalShowdownConfig, showdown_seed_from_int
from pokezero.mcts_eval.search_over_raw import SearchConfiguration, digest
from pokezero.mcts_eval.search_over_raw_ledger import _write_new
from pokezero.mcts_eval.search_over_raw_reconciled import (
    a1_configurations, a2_configurations, deployment_configurations,
    configuration_review_template, controller_input_hashes, selector_runtime_recipe,
)
from pokezero.mcts_eval.search_over_raw_source_run import (
    collect_owned_exploration_panel, source_recipe, source_review_template,
)
from pokezero.mcts_eval.search_over_raw_stage_run import (
    StageSelectorRecipe, exploration_recipe_keys, run_owned_exploration_stages,
)
from pokezero.mcts_eval.search_over_raw_archive import archived_exploration_root_binding
from pokezero.mcts_eval.search_over_raw_outcome_run import outcome_runtime_binding
from pokezero.mcts_eval.search_over_raw_raw_run import RawSelectionRunContract, raw_runtime
from pokezero.mcts_eval.search_over_raw_supervision import supervise_fixed_work
from pokezero.mcts_eval.search_over_raw_world_inputs import (
    TIMED_NATIVE_PUBLIC_CAPTURE, TIMED_REFERENCE_PUBLIC_CAPTURE, _state_payload,
)
import test_search_over_raw_selector_producer as selectors
import test_search_over_raw_continuation_admission as admission
import test_search_over_raw_timed_run as native
import test_search_over_raw_timed_public as public
import test_search_over_raw_native_oracle_run as native_oracle
import test_search_over_raw_reference_timed_run as reference
import test_search_over_raw_reference_public as reference_public
import test_search_over_raw_source_run as sources
import test_search_over_raw_outcome_run as outcomes
from test_search_over_raw_oracle_preparation import opening


@dataclass(frozen=True,kw_only=True)
class AuthoredTimedZeroOracleJob(selectors.AuthoredNativeOracleJob):
    def __call__(self,ownership,phase):
        phase("startup");phase("selection")
        time.sleep(self.authored_selection["elapsed_seconds"]+.01)
        phase("cleanup");return deepcopy(self.authored_selection)


class AuthoredEmptyHpRuntime(reference_public.EmptyPublicRuntime):
    def prepare(self,request):
        prepared=super().prepare(request)
        return replace(prepared,evidence=lambda:dict(draws=[],leaf_ablation=dict(
            schema="pokezero.search-over-raw.leaf-batch.v1",leaf="hp_fraction",rollout_cap=250,
            root_priors="unchanged_champion",root_value_used_for_backup=False,evaluations=[])))


@dataclass(frozen=True,kw_only=True)
class AuthoredHpReferenceJob(selectors.AuthoredReferenceJob):
    def __call__(self,ownership,phase):
        # Only the authored empty worker runtime changes. Its actual twenty
        # worker exports still pass the independent HP leaf/process readers.
        with patch.object(reference_public,"EmptyPublicRuntime",AuthoredEmptyHpRuntime):
            return super().__call__(ownership,phase)


@dataclass(frozen=True)
class ProductivePanelAuditor:
    synthetic_process_fixture = True
    profile_staging: bool = False
    productive_reference: bool = False
    available_sources: int = 1
    fixed_diagnostics: bool = False
    integrated_diagnostics: bool = False
    after_population: bool = False
    reference_values: bool = False
    nonempty_reference_values: bool = False
    preserve_replay_prefix: bool = False
    unavailable_native_replay_root: bool = False
    paired_native_values: bool = False

    def __call__(self, ownership, phase):
        if self.paired_native_values:
            from test_engine_world import _payload,_dex
            with patch("pokezero.local_showdown._public_materialization_payload",return_value=_payload(_dex())):
                return self._run(ownership,phase)
        return self._run(ownership,phase)

    def _run(self,ownership,phase):
        phase("startup")
        directory = Path(ownership["directory"]).parent
        fixtures = directory / "authored-fixtures"; fixtures.mkdir()
        class RetainedFixtureDirectory:
            def __init__(self, *args, **kwargs): self.name = tempfile.mkdtemp(dir=fixtures)
            def cleanup(self): pass  # Ancestor independently rereads closure.
        panel = None
        phase("selection")
        with patch.object(tempfile, "TemporaryDirectory", RetainedFixtureDirectory):
            helper = selectors.SelectorProducerTests(); helper.setUp()
            h, o, c = helper.h, helper.o, helper.c
            try:
                from pokezero.randbat import Gen3RandbatSource, RandbatSourceMetadata
                catalog = Gen3RandbatSource(RandbatSourceMetadata("gen3randombattle",3,None,None,None,"a"*64),{})
                cfgs = {cfg.identity:cfg for cfg in a1_configurations()+a2_configurations()+deployment_configurations()}
                raw = raw_runtime(SearchConfiguration("raw",seconds=1.),o.files[str(o.checkpoint)],"a"*64)
                h.raw_sha = digest(raw); h.review["raw_runtime_sha256"] = h.raw_sha
                recipes, hashes = {}, {**controller_input_hashes(),
                    str(c.evidence):hashlib.sha256(c.evidence.read_bytes()).hexdigest()}
                for stage, identity in exploration_recipe_keys():
                    cfg = cfgs[identity]; factory = None
                    if cfg.arm == "reference":
                        runtime, factory = reference.owner_runtime(cfg,str(o.checkpoint),str(o.showdown),o.files[str(o.checkpoint)])
                        factory = replace(factory,set_source_hash="a"*64)
                        runtime.update(reference_factory=asdict(factory),set_source_hash="a"*64,
                            full_public_request_observation=deepcopy(TIMED_REFERENCE_PUBLIC_CAPTURE))
                        if cfg.belief == "oracle":
                            from pokezero.mcts_eval.search_over_raw_reference_oracle_run import REFERENCE_ORACLE_CAPTURE
                            runtime.update(reference_team_oracle_observation=deepcopy(REFERENCE_ORACLE_CAPTURE),
                                oracle_reconstruction_catalog_sha256=digest(_state_payload(catalog)))
                    else:
                        runtime = (native_oracle.observed_runtime(cfg,o.files[str(o.checkpoint)])
                            if cfg.belief == "oracle" else native.runtime(cfg,o.files[str(o.checkpoint)]))
                        runtime.update(set_source_hash="a"*64,full_public_request_observation=deepcopy(TIMED_NATIVE_PUBLIC_CAPTURE))
                    review_path = fixtures / (stage+"-"+identity+"-review.json")
                    review = configuration_review_template(h.contract,stage=stage,configuration=identity,
                        raw_runtime_sha256=h.raw_sha,selector_recipe_sha256=digest(selector_runtime_recipe(cfg,runtime)),
                        qualification_evidence_hashes={str(c.evidence):hashlib.sha256(c.evidence.read_bytes()).hexdigest()},
                        cost_scope="Authored full-panel integration only; no production or complete cost qualification.")
                    review.update(status="CLEAR_CONFIGURATION_RECIPE",reviewed_before_first_source_work=True,
                        atomic_work_and_cleanup_validated=True,resource_cost_qualification_reviewed=True)
                    _write_new(review_path,review); hashes[str(review_path)] = hashlib.sha256(review_path.read_bytes()).hexdigest()
                    recipes[(stage,identity)] = StageSelectorRecipe(runtime=runtime,
                        qualification_review_path=str(review_path),reference_factory=factory)
                showdown = fixtures/"source-showdown"; sim = showdown/"dist/sim"; sim.mkdir(parents=True)
                assert type(self.available_sources) is int and 1 <= self.available_sources <= 32
                short_seeds=h.contract["panels"]["exploration"]["seeds"][:self.available_sources]
                settings = dict(rounds=300,mode="complete",roundsBySeed={
                    json.dumps(showdown_seed_from_int(seed),separators=(",",":")):1 for seed in short_seeds})
                assert not self.nonempty_reference_values or (self.reference_values and not self.after_population)
                if self.nonempty_reference_values:
                    import re
                    (sim/"source-settings.json").write_text(json.dumps(settings))
                    toy=Path(__file__).with_name("test_owned_continuation.mjs").read_text()
                    for variable,name in (("fakeRuntime","index.js"),("fakeState","state.js")):
                        (sim/name).write_text(re.search(r"const "+variable+r" = `([\s\S]*?)`;",toy).group(1))
                else:
                    (sim/"index.js").write_text(json.dumps(settings)); (sim/"state.js").write_text("authored-not-engine")
                if self.after_population:
                    # Freeze the actual pure filter/serializer seam BEFORE
                    # the source runtime and every prospective review.
                    official=Path("/Users/scott/workspace/pokerena/vendor/pokemon-showdown/dist/sim/battle.js").read_text()
                    start=official.index("const splitRegex = ");stop=official.index("\nclass Battle {",start)
                    (sim/"battle.js").write_text(official[start:stop]+"\nclass Battle {\n}")
                    (sim/"state.js").write_text("state.log = battle.log;")
                bridge = fixtures/"source-bridge"; bridge.mkdir()
                scripts = Path(__file__).resolve().parents[1]/"scripts"
                (bridge/"battle_bridge.mjs").write_bytes(Path(__file__).with_name("source_bridge_fixture.mjs").read_bytes())
                for name in ("battle_bridge_owned_snapshot.mjs","battle_bridge_owned_continuation.mjs"):
                    (bridge/name).write_bytes((scripts/name).read_bytes())
                config = LocalShowdownConfig(showdown_root=str(showdown),bridge_path=str(bridge/"battle_bridge.mjs"),
                    set_belief_source=False,read_timeout_seconds=.5)
                checkpoint = sources.CheckpointContract(checkpoint_path=str(o.checkpoint),checkpoint_sha256=o.files[str(o.checkpoint)],
                    policy_id="authored",schema_version="authored",token_count=1,categorical_feature_count=1,numeric_feature_count=1,
                    transition_token_count=0,category_vocab=(),architecture={"window_size":1},feature_masks={},model_device="cpu",
                    showdown_root=str(showdown),showdown_source_sha256="a"*64)
                source_runtime = outcome_runtime_binding(config,checkpoint_path=o.checkpoint)
                label_config=replace(config,bridge_path=str(scripts/"battle_bridge.mjs"),read_timeout_seconds=2.) if self.nonempty_reference_values else None
                label_runtime=source_runtime if label_config is None else outcome_runtime_binding(label_config,checkpoint_path=o.checkpoint)
                source_review = fixtures/"source-review.json"
                review = source_review_template(h.contract,source_recipe(config,checkpoint,source_runtime),
                    qualification_evidence_hashes={str(c.evidence):hashlib.sha256(c.evidence.read_bytes()).hexdigest()})
                review.update(status="CLEAR_SOURCE_RECIPE",reviewed_before_first_source_work=True,
                    source_runtime_and_cleanup_reviewed=True,source_policy_and_terminal_reviewed=True)
                _write_new(source_review,review);hashes[str(source_review)] = hashlib.sha256(source_review.read_bytes()).hexdigest()
                if self.fixed_diagnostics or self.integrated_diagnostics:
                    from pokezero.mcts_eval.search_over_raw_diagnostic_producer import (
                        FixedWorkRecipe,diagnostic_configurations,diagnostic_manifest,diagnostic_review_template)
                    from pokezero.mcts_eval.search_over_raw_fixed_work import FixedWorkDiagnosticContract
                    from pokezero.mcts_eval.search_over_raw_fixed_work_run import fixed_work_reader_source_hashes
                    diagnostic_files=fixed_work_reader_source_hashes()
                    diagnostic_files.update(o.files)
                    diagnostic_files.update({str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sim.glob("*.js")})
                    fixed_recipes={c.identity:FixedWorkRecipe(unqualified_reason="authored missing runtime; unrelated recipe may proceed")
                        for c in diagnostic_configurations()}
                    diagnostic_cfg=next(c for c in diagnostic_configurations() if c.arm=="reference" and c.leaf=="model")
                    _,factory=reference.owner_runtime(diagnostic_cfg,str(o.checkpoint),str(showdown),o.files[str(o.checkpoint)])
                    factory=replace(factory,set_source_hash="a"*64)
                    fixed_recipes[diagnostic_cfg.identity]=FixedWorkRecipe(runtime=dict(
                        fixed_work_diagnostic=FixedWorkDiagnosticContract().receipt(),authored_fixture_only=True),reference_factory=factory)
                    if self.reference_values:
                        assert self.fixed_diagnostics or self.integrated_diagnostics
                        from pokezero.mcts_eval.search_over_raw_reference_values import ReferenceValueLabelRecipe
                        from pokezero.mcts_eval.search_over_raw_diagnostic_producer import ReferenceValueFixedWorkRecipe,fixed_work_runtime_recipe
                        labels=ReferenceValueLabelRecipe(71,label_runtime,leaves_per_worker=1)
                        diagnostic_files.update(label_runtime["files"])
                        fixed_recipes[diagnostic_cfg.identity]=ReferenceValueFixedWorkRecipe(
                            runtime=fixed_work_runtime_recipe(diagnostic_cfg,checkpoint,reference_factory=factory,
                                owned_reference_value_recipe=labels),reference_factory=factory,owned_reference_value_recipe=labels)
                    if self.integrated_diagnostics:
                        from pokezero.mcts_eval.search_over_raw_ancestry import NativeAncestryContract
                        from pokezero.mcts_eval.search_over_raw_exploration_workflow import NativeChainRecipe
                        from pokezero.mcts_eval.search_over_raw_native_chain import native_chain_review_template
                        from pokezero.mcts_eval.search_over_raw_materialization import BOOL_OPTIONS,MAP_OPTIONS
                        import test_search_over_raw_native_chain as native_chain_fixture
                        import test_search_over_raw_materialization_run as materializer_fixture
                        import test_search_over_raw_construction as construction_fixture
                        import test_search_over_raw_showdown_boundary as boundary_fixture
                        # The production shared source champion/simulator stays
                        # unchanged; only this explicit authored constructor bridge differs.
                        constructor=fixtures/"constructor-bridge";constructor.mkdir()
                        if self.paired_native_values:
                            from test_search_over_raw_paired_chain import paired_bridge
                            (constructor/"fixture.mjs").write_text(paired_bridge(materializer_fixture.BRIDGE))
                        else:(constructor/"fixture.mjs").write_text(materializer_fixture.BRIDGE)
                        for name in ("battle_bridge_owned_snapshot.mjs","battle_bridge_owned_continuation.mjs"):
                            (constructor/name).write_bytes((scripts/name).read_bytes())
                        snapshot=construction_fixture.map_fixture()[0]
                        if self.after_population or self.paired_native_values:snapshot["battle"]["log"]=[]
                        for side in snapshot["battle"]["sides"]:
                            if self.paired_native_values:side["name"]=h.outcome.snapshot.replay.players[side["id"]]
                            for mon in side["pokemon"]:
                                mon.update(moveSlots=[dict(id="move",pp=15,disabled=False)],hpType="Ice",hpPower=70)
                        snapshot["boundaryRequests"]=boundary_fixture.requests(snapshot["battle"]["sides"])
                        if self.paired_native_values:
                            for request in snapshot["boundaryRequests"].values():request["active"][0]["moves"][0]["pp"]=15
                        if self.unavailable_native_replay_root:
                            snapshot["boundaryRequests"]["p1"]["active"][0]["moves"][0]["disabled"]=True
                        fixture_path=showdown/"fixture.json"
                        fixture_path.write_text(json.dumps(dict(mode="complete",snapshot=snapshot)))
                        chain_config=replace(config,bridge_path=str(constructor/"fixture.mjs"),read_timeout_seconds=2.)
                        chain_runtime=outcome_runtime_binding(chain_config,checkpoint_path=o.checkpoint,extra_files=[fixture_path])
                        options={k:False for k in BOOL_OPTIONS};options.update({k:None for k in MAP_OPTIONS})
                        helper_path=str(Path(native_chain_fixture.__file__).resolve())
                        helper_files={helper_path:hashlib.sha256(Path(helper_path).read_bytes()).hexdigest()}
                        chain_review=fixtures/"native-chain-review.json"
                        chain_reviewed=native_chain_review_template(h.contract,config=chain_config,runtime=chain_runtime,
                            options=options,helper_files=helper_files,preserve_prefix=self.preserve_replay_prefix,
                            paired_labels=self.paired_native_values)
                        chain_reviewed.update(status="CLEAR_NATIVE_DIAGNOSTIC_CHAIN_RECIPE",independent_review_completed=True,
                            reviewed_before_first_source_work=True)
                        _write_new(chain_review,chain_reviewed)
                        hashes.update(helper_files);hashes[str(chain_review)]=hashlib.sha256(chain_review.read_bytes()).hexdigest()
                        chain_recipe=NativeChainRecipe(config=chain_config,runtime=chain_runtime,options=options,
                            helper_files=helper_files,review_path=str(chain_review),preserve_replay_prefix=self.preserve_replay_prefix,
                            paired_continuation_labels=self.paired_native_values)
                        if self.after_population:
                            from pokezero.mcts_eval.search_over_raw_after_population import (
                                NativeAfterRecipe,after_population_review_template)
                            after_review=fixtures/"native-after-review.json"
                            after_reviewed=after_population_review_template(h.contract,config=config,runtime=source_runtime)
                            after_reviewed.update(status="CLEAR_NULL_ONLY_AFTER_POPULATION_RECIPE",
                                independent_review_completed=True,reviewed_before_first_source_work=True)
                            _write_new(after_review,after_reviewed)
                            hashes[str(after_review)]=hashlib.sha256(after_review.read_bytes()).hexdigest()
                            after_recipe=NativeAfterRecipe(config=config,runtime=source_runtime,review_path=str(after_review))
                        # Only incumbent/model is ready; other configurations
                        # remain explicitly unqualified without blocking it.
                        reference_value_recipe=fixed_recipes[diagnostic_cfg.identity]
                        fixed_recipes={c.identity:FixedWorkRecipe(unqualified_reason="authored unavailable diagnostic recipe")
                            for c in diagnostic_configurations()}
                        if self.reference_values:fixed_recipes[diagnostic_cfg.identity]=reference_value_recipe
                        native_cfg=next(c for c in diagnostic_configurations() if c.arm=="incumbent" and c.leaf=="model")
                        fixed_recipes[native_cfg.identity]=FixedWorkRecipe(runtime=dict(
                            fixed_work_diagnostic=FixedWorkDiagnosticContract().receipt(),authored_fixture_only=True),
                            native_ancestry_contract=NativeAncestryContract(17,capture_native_edges=True))
                        diagnostic_files.update(chain_runtime["files"])
                    fixed_manifest=diagnostic_manifest(h.contract,fixed_recipes,checkpoint_contract=checkpoint,
                        showdown_root=str(showdown),file_hashes=diagnostic_files)
                    fixed_review=fixtures/"fixed-work-review.json"
                    reviewed=diagnostic_review_template(fixed_manifest)
                    reviewed.update(status="CLEAR_FIXED_WORK_DIAGNOSTIC_ROSTER",independent_review_completed=True,
                        reviewed_before_first_source_work=True)
                    _write_new(fixed_review,reviewed);hashes[str(fixed_review)]=hashlib.sha256(fixed_review.read_bytes()).hexdigest()
                with patch.object(admission,"controller_input_hashes",return_value=hashes):h.initialize()
                # Everything above is prospective. No original owned source,
                # selected public request or private team yet exists.
                class Env(sources.AuthoredSourceEnv):
                    def __init__(self,config):
                        if label_config is None:super().__init__(config)
                        else:
                            from pokezero.local_showdown import LocalShowdownEnv
                            LocalShowdownEnv.__init__(self,config)
                            self.rounds,self.mode,self.n=settings["rounds"],settings["mode"],0
                            self.calls=[];self.snapshot_transport=False
                    def reset(self,*,seed,format_id):
                        self.rounds = settings["roundsBySeed"].get(json.dumps(showdown_seed_from_int(seed),separators=(",",":")),settings["rounds"])
                        return super().reset(seed=seed,format_id=format_id)
                    def public_materialization_state(self,seat):
                        state = super().public_materialization_state(seat)
                        state.belief_engine.set_source = catalog
                        return replace(state,self_initial_request=opening(seat),replay=deepcopy(h.outcome.snapshot.replay))
                    def snapshot_actionable_boundary(self):
                        snap = deepcopy(h.outcome.snapshot)
                        state = self.public_materialization_state("p1")
                        snap.latest_requests["p1"] = deepcopy(state.self_request)
                        snap.bridge_snapshot["boundaryRequests"]["p1"] = deepcopy(state.self_request)
                        snap.first_requests.update(p1=opening("p1"),p2=opening("p2"))
                        return replace(snap,format_id="gen3randombattle",observation_format_id="gen3randombattle")
                class SourceFactory:
                    synthetic_process_fixture = True
                    def __new__(cls,config,checkpoint):
                        env=Env(config);return env,{seat:sources.AuthoredPolicy(checkpoint,env.mode) for seat in ("p1","p2")}
                panel = collect_owned_exploration_panel(h.ledger,stage="A1",configuration=a1_configurations()[0].identity,
                    config=config,checkpoint_contract=checkpoint,bridge_runtime=source_runtime,
                    source_review_path=source_review,history_review_path=h.review_path,_synthetic_factory=SourceFactory)
                assert len(panel.owners) == 32 and len(panel.by_root) == self.available_sources
                if self.fixed_diagnostics and not self.integrated_diagnostics:
                    from pokezero.mcts_eval.search_over_raw_diagnostic_producer import produce_owned_fixed_work_diagnostics
                    from test_search_over_raw_diagnostic_producer import AuthoredPanelFixedReferenceJob
                    class FixedTransport:
                        synthetic_process_fixture=True
                        def __call__(self,**kwargs):
                            return dict(job_class=AuthoredPanelFixedReferenceJob,
                                job_kwargs={} if label_config is None else dict(authored_label_config=label_config))
                    owned,report=produce_owned_fixed_work_diagnostics(h.ledger,panel,recipes=fixed_recipes,
                        checkpoint_contract=checkpoint,showdown_root=str(showdown),file_hashes=diagnostic_files,
                        review_path=fixed_review,_synthetic_job_transport=FixedTransport())
                    for owner in owned.values():owner.validate(h.ledger)
                    # Adversarial changes only inside this disposable authored
                    # fixture. Restore exact originals; never edit real evidence.
                    first=next(iter(owned.values()));original_selected=first.selected
                    object.__setattr__(first,"selected",deepcopy(original_selected))
                    with unittest.TestCase().assertRaises(ValueError):first.validate(h.ledger)
                    object.__setattr__(first,"selected",original_selected)
                    original_seed=first.reader.expected["selection_seed"]
                    first.reader.expected["selection_seed"]=original_seed+1
                    with unittest.TestCase().assertRaises(ValueError):first.validate(h.ledger)
                    first.reader.expected["selection_seed"]=original_seed
                    terminal_path=first.reader.directory/"terminal.json";original_terminal=terminal_path.read_bytes()
                    terminal_path.write_bytes(original_terminal+b"\n")
                    with unittest.TestCase().assertRaises(ValueError):first.validate(h.ledger)
                    terminal_path.write_bytes(original_terminal)
                    from pokezero.mcts_eval.search_over_raw_fixed_work_run import FixedWorkRunValidator
                    real_read=FixedWorkRunValidator.__call__;original_runtime=first.selected["runtime_sha256"]
                    def mutate_after_read(reader,selected):
                        answer=real_read(reader,selected)
                        if reader is first.reader:selected["runtime_sha256"]="b"*64
                        return answer
                    with patch.object(FixedWorkRunValidator,"__call__",mutate_after_read):
                        with unittest.TestCase().assertRaises(ValueError):first.validate(h.ledger)
                    first.selected["runtime_sha256"]=original_runtime
                    first.validate(h.ledger)
                    from collections import Counter
                    return dict(required_slots=report["required_slots"],status_counts=dict(Counter(s["status"] for s in report["slots"])),
                        actual_accepted_work=sum(s["accepted_work"] or 0 for s in report["slots"]),
                        owners_validated_while_archives_live=len(owned),
                        owner_mutation_checks=4,
                        reference_value_summaries=sum("reference_value_labels" in s.get("owning_readout",{}) for s in report["slots"]),
                        reference_value_labels=sum(s.get("owning_readout",{}).get("reference_value_labels",{}).get("labels",0) for s in report["slots"]),
                        reference_terminal_label_populations=sum(s.get("owning_readout",{}).get("reference_value_labels",{}).get("independent_terminal_labels",False)
                            for s in report["slots"]),
                        deployment_selection_untouched=not (h.ledger.directory/"a2-freeze.json").exists(),
                        terminal_truth_validated=report["terminal_truth_validated"],runtime_qualified=report["runtime_qualified"])
                productive_reference=self.productive_reference
                class Transport:
                    synthetic_process_fixture = True
                    def __call__(self,*,stage,configuration,role,root_id,source_owner,runtime,prior_owner):
                        cfg=configuration
                        bound=archived_exploration_root_binding(source_owner.archive,source_owner.source,
                            contract=h.ledger.contract,root_id=root_id)
                        witness=bound["public_request_witness"]
                        context,pending,_=source_owner.archive.selected(source_owner.source["roots"][0])
                        if role == "raw":
                            legal=[i for i,v in enumerate(context.observation.legal_action_mask) if v]
                            selected=dict(root_id=root_id+":raw",statistics_root_id=root_id,selection_seed=bound["slot"]["source_seed"],
                                configuration_sha256=SearchConfiguration("raw",seconds=1.).identity,runtime_sha256=digest(runtime),
                                action=0,status="SELECTED",elapsed_seconds=.005,nominal_search_seconds=1.,exceeded_nominal_seconds=False,
                                independent_root_statistics=True,scientific_strength_evidence=False,
                                raw_selection_run_contract=RawSelectionRunContract().receipt(),observed_runtime_configuration=deepcopy(runtime),
                                evidence=dict(selector="deterministic_masked_argmax",model_forwards=1,legal_actions=legal,
                                    priors=[1/len(legal)]*len(legal),signed_value=.1,
                                    consumed_binding=deepcopy(witness["consumed_binding"]),public_request_witness=deepcopy(witness)))
                            return dict(_synthetic_job_class=selectors.AuthoredRawJob,_synthetic_job_kwargs=dict(authored_selection=selected))
                        if cfg.arm == "reference":
                            return dict(_synthetic_job_class=selectors.AuthoredReferenceOracleJob if cfg.belief == "oracle" else
                                AuthoredHpReferenceJob if cfg.leaf == "hp_fraction" else
                                reference_public.AuthoredProductiveReferenceJob if productive_reference else selectors.AuthoredReferenceJob,
                                _synthetic_job_kwargs=dict(authored_source=catalog) if cfg.belief == "oracle" else {})
                        if cfg.belief == "oracle":
                            from pokezero.mcts_eval.search_over_raw_archive import capture_archived_team_oracle
                            from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest
                            from pokezero.mcts_eval.search_over_raw_oracle import opening_team
                            from pokezero.env import BattleStartOverride
                            from pokezero.showdown_fixture import pack_team
                            from test_search_over_raw_native_oracle import NativeOracleTests
                            oracle,_=capture_archived_team_oracle(source_owner.archive,source_owner.source,
                                contract=h.ledger.contract,root_id=root_id,set_source=catalog)
                            request=PublicRootRequest.capture(context.public_materialization_state,context.observation,pending_transition=pending)
                            truth=NativeOracleTests()
                            truth.f=SimpleNamespace(oracle=oracle,request=request,source=catalog,context=context)
                            truth.witness=witness
                            truth.override=BattleStartOverride(player_teams={"p1":pack_team(opening_team(request.state.self_initial_request,"p1",catalog)),
                                "p2":pack_team(oracle.opponent_team)},observation_format_id="gen3randombattle")
                            truth.policy=SimpleNamespace(_fixed_override=truth.override,_set_source=catalog)
                            helper=native_oracle.NativeOracleRunTests();helper.truth=truth;helper.cfg=cfg;helper.runtime=runtime
                            helper.inputs=dict(decision_round=context.decision_round_index,consumed_binding=witness["consumed_binding"])
                            selected=helper.selected(zero=True,attempts=0)
                            cls=AuthoredTimedZeroOracleJob
                        elif cfg.leaf == "model":
                            helper=public.TimedPublicTests();helper.f=SimpleNamespace(context=context,root_id=root_id,pending=pending)
                            helper.bound=bound;helper.witness=witness;helper.cfg=cfg
                            try:selected=helper.no_attempt_selection()
                            finally:helper.doCleanups()
                            cls=selectors.AuthoredNativeJob
                        else:
                            with patch.object(native,"BINDING",witness["consumed_binding"]):
                                selected=native.selected_fixture(cfg,runtime,decision_round=context.decision_round_index)
                            manifest=selected["evidence"]["consumed_world_inputs"]
                            manifest["adapter_public_request_witness"]=deepcopy(witness)
                            manifest["public_request_observations"]=[dict(ordinal=i,witness=deepcopy(witness)) for i in range(len(manifest["rows"]))]
                            action=1 if cfg.seconds == 10. else 0
                            selected.update(action=action);selected["evidence"]["root_action"]=f"action:{action}"
                            cls=selectors.AuthoredNativeJob
                        selected.update(root_id=root_id+":"+cfg.identity,statistics_root_id=root_id,selection_seed=bound["slot"]["source_seed"],
                            runtime_sha256=digest(runtime),observed_runtime_configuration=deepcopy(runtime))
                        selected["native_timed_run_contract"]=(native_oracle.NativeOracleTimedRunContract().receipt()
                            if cfg.belief == "oracle" else native.NativeTimedRunContract().receipt())
                        if selected["action"] is None:
                            selected["strict_engine_diagnostic"]["battle_id"]="search-over-raw-root:"+root_id
                            selected["original_failure_receipt"].update(root_id=selected["root_id"],
                                engine_fallback_diagnostic=deepcopy(selected["strict_engine_diagnostic"]))
                        return dict(_synthetic_job_class=cls,_synthetic_job_kwargs=dict(authored_selection=selected))
                # Optional diagnostics are confined to this disposable authored
                # auditor. They do not alter production guards or report private
                # locals, packet contents, file paths or a runtime qualification.
                import cProfile
                import pstats
                profiler=cProfile.Profile() if self.profile_staging else None
                if profiler is not None:profiler.enable()
                try:
                    runner=run_owned_exploration_stages;extra={}
                    if self.integrated_diagnostics:
                        from pokezero.mcts_eval.search_over_raw_exploration_workflow import run_owned_exploration_workflow
                        paired_values=self.paired_native_values
                        class DiagnosticTransport:
                            synthetic_process_fixture=True
                            def __call__(self,*,configuration,job,source_owner):
                                if configuration.arm=="reference":
                                    from test_search_over_raw_diagnostic_producer import AuthoredPanelFixedReferenceJob
                                    return dict(job_class=AuthoredPanelFixedReferenceJob,
                                        job_kwargs={} if label_config is None else dict(authored_label_config=label_config))
                                bound=archived_exploration_root_binding(source_owner.archive,source_owner.source,
                                    contract=h.ledger.contract,root_id=job.statistics_root_id)
                                if paired_values:
                                    from test_search_over_raw_paired_chain import AuthoredPairedNativeJob
                                    native_job_cls=AuthoredPairedNativeJob
                                else:native_job_cls=native_chain_fixture.AuthoredRetainedNativeJob
                                return dict(job_class=native_job_cls,
                                    job_kwargs=dict(public_witness=deepcopy(bound["public_request_witness"])))
                        runner=run_owned_exploration_workflow
                        extra=dict(diagnostic_recipes=fixed_recipes,native_chain_recipe=chain_recipe,
                            diagnostic_review_path=str(fixed_review),diagnostic_file_hashes=diagnostic_files,
                            _synthetic_diagnostic_transport=DiagnosticTransport(),
                            _synthetic_chain_helpers=native_chain_fixture.AuthoredChainHelpers(),
                            _synthetic_chain_transport=native_chain_fixture.AuthoredChainTransport())
                        if self.paired_native_values:
                            from test_search_over_raw_paired_chain import AuthoredPairedChainTransport
                            extra["_synthetic_chain_transport"]=AuthoredPairedChainTransport()
                        if self.after_population:extra["native_after_recipe"]=after_recipe
                    result=runner(h.ledger,panel,recipes=recipes,checkpoint_contract=checkpoint,
                        showdown_root=str(o.showdown),file_hashes=o.files,raw_runtime=raw,history_review_path=h.review_path,
                        continuation_config=h.outcome.config,continuation_runtime=h.outcome.runtime,set_source=catalog,
                        _synthetic_selector_transport=Transport(),_synthetic_continuation_job_class=outcomes.SyntheticContinuationJob,
                        **extra)
                    if self.integrated_diagnostics:
                        workflow=result;result=deepcopy(workflow["staged"])
                        # Only data recipes cross to the external parent. No
                        # private archive or live owner is serialized or rebound.
                        result["authored_external_readback_inputs"]=dict(ledger_directory=str(h.ledger.directory),report=workflow,
                            diagnostic_recipes=[dict(configuration=k,recipe=_state_payload(v)) for k,v in fixed_recipes.items()],
                            native_chain_recipe=_state_payload(chain_recipe),diagnostic_review_path=str(fixed_review),
                            diagnostic_file_hashes=diagnostic_files,checkpoint_contract=_state_payload(checkpoint))
                        if self.after_population:
                            result["authored_external_readback_inputs"]["native_after_recipe"]=_state_payload(after_recipe)
                        recorded=[r for r in workflow["native_chains"] if r["status"]=="CHAIN_RECORDED"]
                        result["authored_workflow_summary"]=dict(required_diagnostic_slots=workflow["diagnostic_population"]["required_slots"],
                            native_chain_rows=len(workflow["native_chains"]),recorded_chains=len(recorded),
                            constructor_runs=sum(r["diagnostic"]["actual_constructor_runs"] for r in recorded),
                            replay_runs=sum(r["diagnostic"]["actual_replay_runs"] for r in recorded),
                            diagnostic_results_used_for_selection=workflow["diagnostic_results_used_for_selection"])
                finally:
                    if profiler is not None:profiler.disable()
                if profiler is not None:
                    statistics=pstats.Stats(profiler).stats
                    def profile_row(key,row):
                        primitive,total,self_seconds,cumulative,_=row
                        return dict(module=Path(key[0]).name,function=key[2],line=key[1],
                            primitive_calls=primitive,total_calls=total,self_seconds=self_seconds,
                            cumulative_seconds=cumulative)
                    result["authored_staging_profile"]=dict(
                        scope="Current authored auditor thread wall-time attribution, including waits; nested worker execution excluded; instrumentation affects timing.",
                        cumulative_times_overlap=True,production_cost_qualified=False,
                        authored_packet_construction=[profile_row(k,v) for k,v in statistics.items()
                            if Path(k[0]).name == Path(__file__).name and k[2] == "__call__"],
                        top_self=[profile_row(k,v) for k,v in sorted(statistics.items(),key=lambda item:item[1][2],reverse=True)[:25]],
                        top_cumulative=[profile_row(k,v) for k,v in sorted(statistics.items(),key=lambda item:item[1][3],reverse=True)[:25]])
                # Inspect actual durable launches/reservations, not just scores.
                attempts=list((h.directory/"continuation-admission").glob("*/attempts/*.json"))
                result["authored_reservations"]=[{key:json.loads(p.read_bytes())[key] for key in
                    ("target_actions","replicates","job_sha256","attempt_binding_sha256")} for p in attempts]
                hp=SearchConfiguration("incumbent",leaf="hp_fraction",seconds=1.)
                evidence=[json.loads((h.directory/stage/(cfg.identity+"-root-evidence.json")).read_bytes())
                    for stage,cfg in (("A2",hp),("A3",replace(hp,seconds=3.)),("A3",replace(hp,seconds=10.)))]
                handoffs=[row["owned_input_handoff"] for row in evidence]
                raw_provenance=[row["shared_raw_provenance"] for row in handoffs]
                assert raw_provenance[0] == raw_provenance[1] == raw_provenance[2]
                cells=[row["shared_cell_provenance"] for row in handoffs]
                assert len(cells[0]) == 8*self.available_sources and cells[0] == cells[1]
                assert len(cells[2]) == 16*self.available_sources and all(cells[2][k] == v for k,v in cells[0].items())
                if productive_reference:
                    model=SearchConfiguration("reference",seconds=1.,workers=20)
                    rows=[json.loads((h.directory/stage/(cfg.identity+"-root-evidence.json")).read_bytes())
                        for stage,cfg in (("A1",model),("A2",model),("A3",replace(model,seconds=3.)),("A3",replace(model,seconds=10.)))]
                    assert all(set(row["root_intervals"]) == set(panel.by_root) for row in rows)
                    handoffs=[row["owned_input_handoff"] for row in rows]
                    assert all(row["shared_raw_provenance"] == raw_provenance[0] for row in handoffs)
                    assert all(row["shared_cell_provenance"] == cells[0] for row in handoffs[:2])
                    assert all(row["shared_cell_provenance"] == cells[2] for row in handoffs[2:])
                    result["authored_reference_capture_reuse_validated"]=True
                result["authored_prior_owner_capture_reuse_validated"]=True
                result["archives_closed"]=all(not owner.archive._public and not owner.archive._private for owner in panel.owners)
                result["scientific_measurements"]=0
                return result
            except BaseException as error:
                import traceback
                terminals=[dict(path=str(p.relative_to(h.directory)),receipt=json.loads(p.read_bytes()))
                    for p in h.directory.rglob("terminal.json")]
                failures=[row for row in terminals if row["receipt"].get("supervisor_exit_code") != 0]
                for row in failures:
                    detail=h.directory/Path(row["path"]).parent/"authored-reference-error.json"
                    if detail.exists():row["authored_reference_error"]=json.loads(detail.read_bytes())
                _write_new(directory/"authored-fixture-error.json",dict(error=type(error).__name__,message=str(error),
                    traceback=traceback.format_exc(),failed_authored_terminals=failures))
                raise
            finally:
                phase("cleanup")
                if panel is not None:
                    for owner in panel.owners:owner.close()
                helper.doCleanups()


class ProductivePanelTests(unittest.TestCase):
    def test_actual_productive_reference_selector_has_twenty_owned_public_workers(self):
        helper=selectors.SelectorProducerTests();helper.setUp();self.addCleanup(helper.doCleanups)
        helper.initialize(cfg=SearchConfiguration("reference",seconds=1.,workers=20))
        from pokezero.mcts_eval.search_over_raw_selector_producer import produce_archived_selector
        owner,receipt=produce_archived_selector(helper.h.ledger,helper.stage,helper.cfg.identity,
            archive=helper.a.archive,source=helper.a.source,root_id=helper.a.root_id,checkpoint_contract=helper.checkpoint,
            showdown_root=str(helper.o.showdown),runtime=helper.search_runtime,file_hashes=helper.o.files,
            qualification_review_path=helper.c.review_path,history_review_path=helper.h.review_path,
            reference_factory=helper.factory,_synthetic_job_class=reference_public.AuthoredProductiveReferenceJob,
            _synthetic_job_kwargs={})
        self.assertTrue(receipt["selector_launched"])
        self.assertEqual(owner.selection["status"],"SELECTED")
        readout=owner.reader.last_validation
        self.assertGreater(readout["actual_work"]["accepted_units"],0)
        self.assertEqual(len(readout["owned_generation_pids"]),20)
        rows=owner.selection["evidence"]["worker_receipts"]
        self.assertEqual({row["worker"] for row in rows},set(range(20)))
        self.assertTrue(all(row["evidence"]["worker_public_request_preparation"]["worker"] == row["worker"] for row in rows))
        self.assertTrue(all(draw["released"] and draw["consumed_world_input"]["consumed"]
            for row in rows for draw in row["evidence"]["draws"]))
        self.assertFalse(readout["runtime_qualified"])

    def test_actual_empty_hp_reference_worker_exports(self):
        helper=selectors.SelectorProducerTests();helper.setUp();self.addCleanup(helper.doCleanups)
        helper.initialize(cfg=SearchConfiguration("reference",leaf="hp_fraction",seconds=1.,workers=20))
        from pokezero.mcts_eval.search_over_raw_selector_producer import produce_archived_selector
        owner,_=produce_archived_selector(helper.h.ledger,helper.stage,helper.cfg.identity,
            archive=helper.a.archive,source=helper.a.source,root_id=helper.a.root_id,checkpoint_contract=helper.checkpoint,
            showdown_root=str(helper.o.showdown),runtime=helper.search_runtime,file_hashes=helper.o.files,
            qualification_review_path=helper.c.review_path,history_review_path=helper.h.review_path,
            reference_factory=helper.factory,_synthetic_job_class=AuthoredHpReferenceJob,_synthetic_job_kwargs={})
        self.assertEqual(owner.selection["status"],"UNQUALIFIED_AT_CEILING")
        self.assertEqual(len(owner.selection["evidence"]["worker_receipts"]),20)
        self.assertTrue(all(row["evidence"]["leaf_ablation"]["evaluations"] == []
            for row in owner.selection["evidence"]["worker_receipts"]))

    def test_actual_productive_full_panel_freeze_and_cross_stage_no_redraw(self):
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root).resolve()/"auditor"
            result=supervise_fixed_work(ProductivePanelAuditor(),directory=directory,binding_sha256="f"*64,verify_admission=lambda:True)
            error=directory/"authored-fixture-error.json"
            self.assertEqual(result["supervisor_exit_code"],0,json.loads(error.read_bytes()) if error.exists() else result)
            summary=result["result"]
            self.assertEqual(summary["source_panel"]["source_count"],32)
            self.assertEqual(len(summary["source_panel"]["observed_root_ids"]),1)
            self.assertEqual(len(summary["source_panel"]["missing_root_ids"]),199)
            self.assertEqual(summary["launches"],dict(raw=1,search=8,continuation=2,reused_selectors=11,reused_a3_records=1))
            self.assertEqual(dict(summary["evaluator_freeze"]["evaluators"]),dict(incumbent="hp_fraction",reference=None))
            self.assertEqual([r["status"] for r in summary["reports"]["A1"]["configurations"]],["UNQUALIFIED_AT_CEILING"]*4)
            self.assertEqual(len(summary["reports"]["A2"]["configurations"]),4)
            self.assertEqual(len(summary["reports"]["A3"]["configurations"]),3)
            self.assertEqual(len(summary["authored_reservations"]),2)
            self.assertEqual(sorted(row["target_actions"] for row in summary["authored_reservations"]),[[0],[1]])
            self.assertTrue(all(row["replicates"] == list(range(8)) for row in summary["authored_reservations"]))
            for key in ("job_sha256","attempt_binding_sha256"):
                self.assertEqual(len({row[key] for row in summary["authored_reservations"]}),2)
            self.assertTrue(summary["authored_prior_owner_capture_reuse_validated"])
            self.assertTrue(summary["archives_closed"])
            self.assertFalse(summary["private_archive_returned"])
            self.assertFalse(summary["holdout_authorized"])
            self.assertEqual(summary["scientific_measurements"],0)
            print(json.dumps(dict(authored_only=True,stage_wall_seconds=summary["elapsed_seconds"],
                stage_auditor_cpu_seconds=summary["auditor_cpu_seconds"],outer_wall_seconds=result["elapsed_wall_seconds"],
                outer_wait4_cpu_seconds=result["cpu_seconds_including_reaped_children"],
                wait4_job_high_water_bytes=result["maxrss"],resource_cost_complete=False)))

    def test_actual_two_root_both_arm_panel_preserves_cross_arm_banks_and_a3(self):
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root).resolve()/"auditor"
            result=supervise_fixed_work(ProductivePanelAuditor(productive_reference=True,available_sources=2),
                directory=directory,binding_sha256="f"*64,verify_admission=lambda:True)
            error=directory/"authored-fixture-error.json"
            self.assertEqual(result["supervisor_exit_code"],0,json.loads(error.read_bytes()) if error.exists() else result)
            summary=result["result"]
            self.assertEqual(summary["source_panel"]["source_count"],32)
            self.assertEqual(len(summary["source_panel"]["observed_root_ids"]),2)
            self.assertEqual(len(summary["source_panel"]["missing_root_ids"]),198)
            self.assertEqual(summary["launches"],dict(raw=2,search=20,continuation=4,reused_selectors=26,reused_a3_records=2))
            self.assertEqual(dict(summary["evaluator_freeze"]["evaluators"]),dict(incumbent="hp_fraction",reference="model"))
            self.assertEqual(len(summary["reports"]["A3"]["configurations"]),6)
            self.assertEqual(len(summary["authored_reservations"]),4)
            self.assertEqual(sorted(row["target_actions"] for row in summary["authored_reservations"]),[[0],[0],[1],[1]])
            self.assertTrue(all(row["replicates"] == list(range(8)) for row in summary["authored_reservations"]))
            for key in ("job_sha256","attempt_binding_sha256"):
                self.assertEqual(len({row[key] for row in summary["authored_reservations"]}),4)
            self.assertTrue(summary["authored_prior_owner_capture_reuse_validated"])
            self.assertTrue(summary["authored_reference_capture_reuse_validated"])
            self.assertTrue(summary["archives_closed"])
            self.assertFalse(summary["private_archive_returned"])
            self.assertFalse(summary["holdout_authorized"])
            self.assertEqual(summary["scientific_measurements"],0)
            print(json.dumps(dict(authored_only=True,available_roots=2,both_arms_productive=True,
                stage_wall_seconds=summary["elapsed_seconds"],stage_auditor_cpu_seconds=summary["auditor_cpu_seconds"],
                outer_wall_seconds=result["elapsed_wall_seconds"],outer_wait4_cpu_seconds=result["cpu_seconds_including_reaped_children"],
                wait4_job_high_water_bytes=result["maxrss"],resource_cost_complete=False)))


if __name__ == "__main__": unittest.main()
