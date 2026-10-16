from copy import deepcopy
from dataclasses import replace
import random
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from pokezero.engine_search import EngineMctsConfig,EnvTier2AnnotationSource
from pokezero.policy import PolicyContext,PolicyDecision
from pokezero.trajectory import BattleTrajectory
from pokezero.root_prior_correspondence import fingerprint
from pokezero.mcts_eval.search_over_raw_source import AuditedRawPolicy
from pokezero.mcts_eval.search_over_raw_live_capture import LiveBoundaryCapture,selector_config
from test_search_over_raw_source import FakeChampion,SourceEnv,SHA


class FakeLivePolicy:
    def __init__(self,env,leaf,*,iterations=320,metadata=True,mutate=False):
        checkpoint=SimpleNamespace(model_path="model",checkpoint_path="checkpoint",tables_path="tables")
        self._config=selector_config(checkpoint,total=320,leaf=leaf)
        self._annotation_source=EnvTier2AnnotationSource(env);self._fixed_override=None
        self.stats=SimpleNamespace(total_iterations=0,worlds_searched=0)
        self.iterations,self.metadata,self.mutate=iterations,metadata,mutate
        self.draws=[]

    def select_action_with_context(self,context,*,rng):
        self.draws.append(rng.random())
        self.stats.total_iterations+=self.iterations;self.stats.worlds_searched+=4
        if self.mutate:self._annotation_source._env.state["turn"]+=1
        md={"engine_mcts":{"override":{"root_allocation":{
            "prior_authority":True,"prior_cause":None,"arms":[
                {"action_index":0,"model_prior_pre_display":.3},
                {"action_index":1,"model_prior_pre_display":.7}]}}}} if self.metadata else {}
        return PolicyDecision(0,"authored-shadow",metadata=md)


class LiveCaptureTests(unittest.TestCase):
    def setUp(self):
        self.env=SourceEnv();self.env.state={"turn":2}
        self.env.public_materialization_state=lambda seat:deepcopy(self.env.state)
        self.context=PolicyContext("p1",1,"authored","gen3randombattle",7,
            self.env.observe("p1"),("p1","p2"),BattleTrajectory("authored","gen3randombattle",7),
            public_materialization_state=deepcopy(self.env.state))
        self.raw=FakeChampion().select_action(self.context.observation,rng=random.Random(0))
        def identity(context):
            return fingerprint(dict(index=context.decision_round_index,observation=context.observation.numeric_features,
                state=context.public_materialization_state))
        self.addCleanup(patch.stopall)
        patch("pokezero.mcts_eval.search_over_raw_live_capture.boundary_fingerprint",side_effect=identity).start()

    def capture(self,**kwargs):
        leaves={leaf:FakeLivePolicy(self.env,leaf,**kwargs) for leaf in ("model","hp_fraction")}
        published=[]
        capture=LiveBoundaryCapture(self.env,leaves,source_seed=7,total=320,guard=Mock(),
            publish=lambda leaf,row:published.append((leaf,row)))
        return capture,leaves,published

    def test_shadow_choices_never_replace_raw_or_consume_its_rng_and_all_boundaries_are_retained(self):
        capture,policies,published=self.capture()
        global_rng=random.getstate()
        for index in (0,1,2,3):capture(replace(self.context,decision_round_index=index),self.raw)
        self.assertEqual(random.getstate(),global_rng)
        self.assertEqual(self.env.calls,[])
        self.assertEqual(self.raw.action_index,1)
        self.assertEqual(len(published),8)  # Opening advances the persistent selector too.
        self.assertEqual(policies["model"].draws,policies["hp_fraction"].draws)
        self.assertEqual(len(set(policies["model"].draws)),1)  # Declared per-decision reset law.
        source=dict(source_seed=7,source_terminal_complete=True,missing_root_ids=[],roots=[
            dict(source_request_index=3,public_record={"recorded_action_index":1}),
            dict(source_request_index=1,public_record={"recorded_action_index":1})])
        report=capture.selected_report("model",source,{"authored":True})
        self.assertEqual([r["round"] for r in report["rows"]],[3,1])
        self.assertTrue(all(r["variants"]["sampled4"]["action"]==0 for r in report["rows"]))
        with self.assertRaisesRegex(ValueError,"retry"):capture(self.context,self.raw)

    def test_unavailable_mapping_or_work_stays_unavailable(self):
        for kwargs in ({"metadata":False},{"iterations":160}):
            capture,_,published=self.capture(**kwargs);capture(self.context,self.raw)
            self.assertFalse(capture.poisoned)
            self.assertTrue(all(row["variants"]["sampled4"]["status"]=="UNAVAILABLE" for _,row in published))

    def test_constructor_refusal_is_uncertain_but_operational_fallback_stops(self):
        from pokezero.engine_search import EngineSearchFallbackError
        for reason,stop in (("no_worlds_constructed",False),("crate_search_failed",True)):
            capture,policies,published=self.capture()
            refused=EngineSearchFallbackError("authored",diagnostic={"reason":reason})
            policies["model"].select_action_with_context=Mock(side_effect=refused)
            if stop:
                with self.assertRaises(EngineSearchFallbackError):capture(self.context,self.raw)
            else:capture(self.context,self.raw)
            self.assertEqual(capture.poisoned,stop)
            self.assertEqual(published[0][1]["variants"]["sampled4"]["status"],
                "OPERATIONAL_STOP" if stop else "UNAVAILABLE")

    def test_mutation_or_publication_failure_poison_capture_and_preserve_failure(self):
        capture,_,published=self.capture(mutate=True)
        with self.assertRaisesRegex(ValueError,"mutated"):capture(self.context,self.raw)
        self.assertTrue(capture.poisoned)
        self.assertEqual(published[0][1]["variants"]["sampled4"]["status"],"OPERATIONAL_STOP")
        with self.assertRaisesRegex(ValueError,"closed"):capture(replace(self.context,decision_round_index=2),self.raw)
        self.setUp()
        capture,_,_=self.capture();capture.publish=Mock(side_effect=RuntimeError("authored sink failure"))
        with self.assertRaises(RuntimeError):capture(self.context,self.raw)
        self.assertTrue(capture.poisoned)

    def test_raw_policy_hook_retains_original_observation_and_returns_raw_decision(self):
        observed=[]
        audited=AuditedRawPolicy(FakeChampion(),checkpoint_sha256=SHA,
            live_context_sink=lambda context,decision:observed.append((context,decision)))
        decision=audited.select_action_with_context(self.context,rng=random.Random(0))
        self.assertEqual(decision.action_index,1)
        self.assertIs(observed[0][0].observation,self.context.observation)
        self.assertIs(observed[0][1],decision)

    def test_fixed_recipe_matches_live_checker_settings(self):
        checkpoint=SimpleNamespace(model_path="model",checkpoint_path="checkpoint",tables_path="tables")
        cfg=selector_config(checkpoint,total=320,leaf="model")
        self.assertEqual((cfg.worlds,cfg.search_sims,cfg.search_batch,cfg.search_depth),(4,80,16,6))
        self.assertTrue(cfg.root_prior_exact_aggregate)
        self.assertIsNone(cfg.model_decision_time_ms)

    def test_manifest_binds_actual_full_config_and_later_drift_refuses(self):
        from pokezero.mcts_eval.search_over_raw_live_capture import shadow_recipe
        checkpoint=SimpleNamespace(model_path="model",checkpoint_path="checkpoint",tables_path="tables")
        capture,policies,_=self.capture();capture(self.context,self.raw)
        recipe=shadow_recipe(checkpoint,total=320,leaves=["model","hp_fraction"])
        self.assertEqual(len(capture.manifest(recipe)["invocations"]),1)
        changed=deepcopy(recipe);changed["selectors"]["model"]={"unexecuted":"configuration"}
        with self.assertRaisesRegex(ValueError,"recipe"):capture.manifest(changed)
        policies["model"]._config=replace(policies["model"]._config,c_puct=1.5)
        with self.assertRaisesRegex(ValueError,"configuration changed"):
            capture(replace(self.context,decision_round_index=2),self.raw)


if __name__=="__main__":unittest.main()
