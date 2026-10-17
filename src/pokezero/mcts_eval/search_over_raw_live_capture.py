"""Live source shadow selection; no archive reconstruction or launch authority.

Observe all eligible p1 boundaries, then apply the unchanged completed-catalog
root selection. The source still commits the original audited raw decision.
Work counters below are production telemetry, NOT atomic-backup validation.
"""
from copy import deepcopy
from dataclasses import replace
import math
import json
import random
import time

from ..root_prior_correspondence import SCHEMA, boundary_fingerprint
from .search_over_raw import require

SOURCE_SCHEMA="pokezero.search-over-raw.source-live-shadow.v2"


def shadow_recipe(checkpoint,*,total,leaves,scan_policy=None):
    from .search_over_raw_world_inputs import _state_payload
    require(type(leaves) is list and (leaves==(["model","hp_fraction"] if total==320 else ["model"])
        if scan_policy is None else total==4096 and leaves==["model","hp_fraction"]
        and scan_policy=="chronological_deviation_scan_v1"),
        "original fixed-work leaves required")
    recipe=dict(schema="pokezero.search-over-raw.live-shadow-recipe.v1",total=total,
        leaves=list(leaves),selectors={leaf:_state_payload(selector_config(checkpoint,total=total,leaf=leaf))
            for leaf in leaves},world_rng="per-decision",selection_seed="source_seed",
        invocation_schedule="every p1 boundary including opening",
        source_play_uses_search=False,root_rule="unchanged completed-catalog seed priority")
    if scan_policy is not None:
        recipe.update(schema="pokezero.search-over-raw.live-shadow-recipe.v2",scan_policy=scan_policy,
            invocation_schedule="opening warmup and every eligible p1 boundary until declared scan cutoff",
            root_rule="all scanned eligible boundaries in source/request order; no quota sampling")
    return recipe


def selector_config(checkpoint, *, total, leaf):
    from ..engine_search import EngineMctsConfig
    require(type(total) is int and total in (320,1024,4096)
        and leaf in ("model","hp_fraction"),"declared fixed total and supported leaf required")
    return EngineMctsConfig(worlds=4,search_time_ms=5000,threads=1,leaf_eval="model",
        model_path=checkpoint.model_path,checkpoint_path=checkpoint.checkpoint_path,
        tables_path=checkpoint.tables_path,model_device="cpu",search_sims=total//4,
        search_batch=16,search_depth=6,c_puct=1.4,deep_ko_split=True,model_priors=True,
        use_opponent_priors=False,override_telemetry=True,root_prior_exact_aggregate=True,
        early_stop=False,model_world_workers=1,strict_fallbacks=True,
        **({"model_leaf_override":"hp_fraction"} if leaf=="hp_fraction" else {}))


def build_selectors(env,checkpoint,*,total,leaves):
    """Explicit production construction; called only by an admitted owner."""
    from ..dex import load_showdown_dex_cached
    from ..randbat import load_gen3_randbat_source_cached
    from ..engine_search import EngineMctsPolicy,EnvTier2AnnotationSource
    result={}
    for leaf in leaves:
        policy=EngineMctsPolicy(dex=load_showdown_dex_cached(checkpoint.showdown_root),
            set_source=load_gen3_randbat_source_cached(checkpoint.showdown_root),
            annotation_source=EnvTier2AnnotationSource(env),policy_id="live-shadow-"+leaf,
            config=selector_config(checkpoint,total=total,leaf=leaf))
        policy.warm_model_runtime()
        result[leaf]=policy
    return result


def native_prior(metadata,legal):
    allocation=metadata.get("engine_mcts",{}).get("override",{}).get("root_allocation",{})
    require(allocation.get("prior_authority") is True and allocation.get("prior_cause") is None,
        "native_prior_unavailable")
    arms=allocation.get("arms")
    require(type(arms) is list and arms,"native_prior_unavailable")
    indices=[a.get("action_index") for a in arms]
    require(all(type(i) is int for i in indices) and len(indices)==len(set(indices))
        and set(indices)==set(legal),"native_prior_mapping_incomplete")
    values={str(a["action_index"]):a.get("model_prior_pre_display") for a in arms}
    require(all(type(p) in (int,float) and math.isfinite(p) and p>=0 for p in values.values())
        and math.isclose(math.fsum(values.values()),1.,rel_tol=0.,abs_tol=2e-5),
        "native_pre_display_prior_unavailable")
    return values  # Preserve the production aggregate without renormalization.


class LiveBoundaryCapture:
    def __init__(self,env,selectors,*,source_seed,total,guard,publish):
        require(type(source_seed) is int and 0<=source_seed<2**32 and total in (320,1024,4096)
            and selectors and set(selectors)<={"model","hp_fraction"},"fixed live source recipe required")
        from ..engine_search import EnvTier2AnnotationSource
        for leaf,policy in selectors.items():
            config=policy._config
            require(config.worlds==4 and config.search_sims==total//4 and config.leaf_eval=="model"
                and config.model_leaf_override==("hp_fraction" if leaf=="hp_fraction" else None)
                and config.strict_fallbacks is True
                and config.model_decision_time_ms is None and config.early_stop is False
                and config.root_prior_exact_aggregate is True and config.override_telemetry is True
                and type(policy._annotation_source) is EnvTier2AnnotationSource
                and policy._annotation_source._env is env and policy._fixed_override is None,
                "original live environment annotation and fixed public selector required")
        self.env,self.selectors,self.seed,self.total=env,dict(selectors),source_seed,total
        self.guard,self.publish=guard,publish
        self.rows={leaf:{} for leaf in selectors}
        self.poisoned=False
        self.configurations=self._configurations()

    def _configurations(self):
        from .search_over_raw_world_inputs import _state_payload
        from ..engine_search import EnvTier2AnnotationSource
        require(all(type(policy._annotation_source) is EnvTier2AnnotationSource
            and policy._annotation_source._env is self.env and policy._fixed_override is None
            for policy in self.selectors.values()),"original public live annotation required")
        return {leaf:_state_payload(policy._config) for leaf,policy in self.selectors.items()}

    def _guard(self):
        require(self._configurations()==self.configurations,"original live selector configuration changed")
        self.guard()

    def manifest(self,recipe):
        from .search_over_raw import digest
        require(not self.poisoned and recipe["total"]==self.total
            and recipe["leaves"]==list(self.selectors)
            and recipe["selectors"]==self.configurations,"original live capture recipe required")
        self._guard()
        roster=sorted(self.rows[next(iter(self.rows))])
        require(all(sorted(rows)==roster for rows in self.rows.values()),"partial live selector invocation roster")
        return dict(recipe_sha256=digest(recipe),source_seed=self.seed,
            invocations=[dict(round=index,rows={leaf:digest(rows[index]) for leaf,rows in self.rows.items()})
                for index in roster],source_play_uses_search=False)

    def _actual_boundary(self,context):
        return boundary_fingerprint(replace(context,
            public_materialization_state=self.env.public_materialization_state(context.player_id)))

    def __call__(self,context,decision):
        require(not self.poisoned,"live capture is closed after an operational failure")
        require(context.seed==self.seed and context.player_id=="p1","original live p1 source context required")
        # Advance production policy state at the opening too, exactly as the
        # independent checker does. Only the completed catalog excludes it.
        index=context.decision_round_index
        require(all(index not in rows for rows in self.rows.values()),"live boundary retry forbidden")
        self._guard()
        original=boundary_fingerprint(context)
        require(self._actual_boundary(context)==original,"selector context is not the live boundary")
        legal=[i for i,ok in enumerate(context.observation.legal_action_mask) if ok]
        raw=list(decision.metadata["policy_distribution"])
        require(len(raw)==len(context.observation.legal_action_mask) and legal
            and all(type(p) in (int,float) and math.isfinite(p) and p>=0 for p in raw)
            and all(raw[i]==0 for i in range(len(raw)) if i not in legal)
            and math.isclose(math.fsum(raw),1.,rel_tol=0.,abs_tol=2e-5)
            and decision.action_index==max(legal,key=lambda i:raw[i]),"audited live raw forward required")
        for leaf,policy in self.selectors.items():
            self._guard()
            # Reserve before invocation: even an exception cannot be retried.
            row=dict(seed=self.seed,round=index,boundary_sha256=original,legal=legal,
                raw_action=decision.action_index,raw_prior=list(raw),variants={})
            self.rows[leaf][index]=row
            start=time.perf_counter();cpu=time.process_time()
            before=(policy.stats.total_iterations,policy.stats.worlds_searched)
            try:
                from ..engine_search import EngineSearchFallbackError
                try:selected=policy.select_action_with_context(context,rng=random.Random(self.seed))
                except EngineSearchFallbackError as refused:
                    diagnostic=json.loads(refused.engine_search_fallback_diagnostic or "{}")
                    if diagnostic.get("reason") not in {"no_worlds_constructed","choices_unmapped"}:raise
                    require(boundary_fingerprint(context)==original and self._actual_boundary(context)==original,
                        "refused shadow selection mutated original source boundary")
                    row["variants"]["sampled4"]=dict(status="UNAVAILABLE",error="production_constructor_refusal",
                        fallback_diagnostic=diagnostic)
                    continue
                require(boundary_fingerprint(context)==original and self._actual_boundary(context)==original,
                    "shadow selection mutated original source boundary")
                self._guard()
                iterations=policy.stats.total_iterations-before[0]
                worlds=policy.stats.worlds_searched-before[1]
                item=dict(action=selected.action_index,iterations=iterations,worlds_searched=worlds,
                    work_evidence="production telemetry; not atomic backup validation")
                block=selected.metadata.get("engine_mcts",{})
                item["fallback_reason"]=block.get("fallback")
                item["world_failures"]=deepcopy(block.get("world_failures",{}))
                require(type(selected.action_index) is int and selected.action_index in legal,
                    "live selector returned an illegal action")
                try:
                    item["native_prior_pre_display"]=native_prior(selected.metadata,legal)
                    require(iterations==self.total and worlds==4,"fixed_native_work_unavailable")
                    item["status"]="MEASURED"
                except ValueError as unavailable:
                    item.update(status="UNAVAILABLE",error=str(unavailable))
                row["variants"]["sampled4"]=item
            except BaseException as error:
                self.poisoned=True
                row["variants"]["sampled4"]=dict(status="OPERATIONAL_STOP",error=type(error).__name__)
                raise
            finally:
                row["elapsed_seconds"]=time.perf_counter()-start
                row["cpu_seconds"]=time.process_time()-cpu
                try:self.publish(leaf,deepcopy(row))
                except BaseException:
                    self.poisoned=True
                    raise
        self._guard()

    def selected_report(self,leaf,source,bindings):
        """A report only; does not certify source lifecycle or authorize reads."""
        require(not self.poisoned and source["source_seed"]==self.seed
            and source["source_terminal_complete"] is True,"closed original source required")
        expected=[];rows=[]
        for root in source["roots"]:
            index=root["source_request_index"]
            require(index in self.rows[leaf],"selected root lacks its original live observation")
            row=deepcopy(self.rows[leaf][index])
            require(row["raw_action"]==root["public_record"]["recorded_action_index"],
                "live source raw action differs from catalog")
            rows.append(row);expected.append(dict(seed=self.seed,round=index,variant="sampled4"))
        return dict(schema=SCHEMA,complete=True,bindings=deepcopy(bindings),expected=expected,rows=rows,
            missing_root_ids=list(source["missing_root_ids"]),scientific_strength_claim=False,
            scope="live selection witness only; owning source/continuation readers and independent exact gate remain required")
