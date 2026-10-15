"""Bounded exploration denominator/authority tests; no fresh scientific work."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from pokezero.mcts_eval import search_over_raw_bounded_a2 as a2
from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_reconciled import validate_contract, ReconciledPhaseALedger
from pokezero.mcts_eval.search_over_raw_ledger import _write_new


def specimen(stage="A2"):
    return dict(contract=a2.contract("5aa69871-fcb8-41f6-ad9e-9868f8ec58c1",list(range(100,132)),
        [1,2,3],dict(path="/bound/steer.md",sha256="a"*64),stage=stage))


class BoundedA2Tests(unittest.TestCase):
    def test_exact_two_leaf_contract_exploration_only(self):
        c=specimen()["contract"]
        validate_contract(c)
        self.assertEqual(len(c["panels"]["exploration"]["root_slots"]),200)
        self.assertEqual(set(c["panels"]),{"exploration"})
        self.assertEqual(len(c["configurations"]),3)
        self.assertFalse(c["runtime_qualified"])
        self.assertFalse(c["execution_ready"])
        self.assertTrue(c["collection_authorized"])
        for field in ("attempt_wall_seconds","continuation_replicates","runtime_qualified","holdout_authorized"):
            broken=deepcopy(c);broken[field]=1
            with self.assertRaises(ValueError):validate_contract(broken)

    def test_no_legacy_staged_or_holdout_ledger_upgrade(self):
        c=specimen()["contract"]
        with tempfile.TemporaryDirectory() as directory:
            _write_new(Path(directory)/"contract.json",dict(contract=c,contract_sha256=digest(c)))
            with self.assertRaisesRegex(ValueError,"cannot enter"):
                ReconciledPhaseALedger(directory)

    def test_seed_replacement_or_exposure_overlap_rejected(self):
        c=specimen()["contract"]
        for seeds in (list(range(100,131)),[100]*32,[1]+list(range(101,132))):
            with self.assertRaises(ValueError):a2.contract(c["namespace"],seeds,c["excluded_seeds"],c["authority"])

    def summary(self,rows):
        for row in rows:
            row.setdefault("continuation_cell_counts",{("raw" if c.arm == "raw" else c.identity):
                dict(COMPLETE=0,CAPPED=0,REFUSED=0,UNSTARTED=8,UNVALIDATED=0) for c in a2.configurations()})
        return a2.summarize(specimen(),rows,dict(elapsed_wall_seconds=10.,
            cpu_seconds_including_reaped_children=8.,supervisor_exit_code=124,worker_exit_code=-9))

    def test_empty_stopped_run_keeps_all_200_uncertain(self):
        r=self.summary([])
        self.assertEqual(r["missing_root_slots"],200)
        self.assertIsNone(r["enclosing_wall_seconds_per_recorded_root"])
        for leaf in r["leaves"].values():
            self.assertEqual(leaf["fixed_seed_weighted_mean_bounds"],[-1.,1.])
            self.assertEqual(leaf["bootstrap_envelope"],[-1.,1.])
            self.assertIsNone(leaf["mean_delta"])
            self.assertEqual(leaf["uncertain_root_contrasts"],200)
        self.assertFalse(r["significance_claim"])
        self.assertEqual(r["continuation_alias_cell_counts"]["model"]["UNSTARTED"],1600)

    def test_one_known_root_does_not_become_complete_case_mean(self):
        c=specimen()["contract"];root=c["panels"]["exploration"]["root_slots"][0]
        row=dict(root_id=root["root_id"],source_seed=root["source_seed"],
            root_intervals={cfg.identity:[.5,.5] for cfg in a2.configurations()[1:]})
        r=self.summary([row])
        self.assertEqual(r["recorded_roots"],1)
        self.assertEqual(r["missing_root_slots"],199)
        for leaf in r["leaves"].values():
            self.assertIsNone(leaf["mean_delta"])
            self.assertEqual(leaf["roots_with_identified_contrast"],1)
            self.assertLess(leaf["fixed_seed_weighted_mean_bounds"][0],0.)
            self.assertGreater(leaf["fixed_seed_weighted_mean_bounds"][1],0.)

    def test_full_identified_panel_recovers_mean_and_shared_bootstrap(self):
        c=specimen()["contract"]
        rows=[dict(root_id=r["root_id"],source_seed=r["source_seed"],
            root_intervals={cfg.identity:[.125,.125] for cfg in a2.configurations()[1:]})
            for r in c["panels"]["exploration"]["root_slots"]]
        r=self.summary(rows)
        self.assertEqual(r["missing_root_slots"],0)
        self.assertEqual(r["leaves"]["model"],r["leaves"]["hp_fraction"])
        self.assertEqual(r["leaves"]["model"]["mean_delta"],.125)
        self.assertEqual(r["leaves"]["model"]["bootstrap_envelope"],[.125,.125])

    def test_duplicate_or_foreign_root_rejected(self):
        r=dict(root_id="foreign",source_seed=100,root_intervals={})
        with self.assertRaises(ValueError):self.summary([r])

    def test_continuation_aliases_keep_refused_and_unvalidated_uncertain(self):
        model,hp=a2.configurations()[1:]
        evidence=dict(actions={"raw":1,model.identity:2,hp.identity:1},validated_outcome_sha256={"1:0":"a"*64})
        rows=[dict(action=1,replicate=0,status="COMPLETE"),dict(action=2,replicate=0,status="REFUSED"),
            dict(action=2,replicate=1,status="COMPLETE")]
        counts=a2.continuation_counts(evidence,rows)
        self.assertEqual(counts["raw"],counts[hp.identity])
        self.assertEqual(counts[model.identity]["REFUSED"],1)
        self.assertEqual(counts[model.identity]["UNVALIDATED"],1)
        self.assertEqual(sum(counts[model.identity].values()),8)

    def test_a1_is_exact_public_oracle_model_pair_without_calendar_gate(self):
        c=specimen("A1")["contract"]
        validate_contract(c)
        self.assertEqual([(cfg.arm,cfg.belief,cfg.leaf,cfg.seconds,cfg.workers)
            for cfg in a2.configurations("A1")],[
                ("raw","public","model",1.,1),("incumbent","public","model",1.,1),
                ("incumbent","oracle","model",1.,1)])
        self.assertNotIn("not_before_utc",c)
        self.assertFalse(hasattr(a2,"calendar_admission"))
        changed=deepcopy(c);changed["not_before_utc"]="2026-10-15T07:00:00Z"
        with self.assertRaises(ValueError):validate_contract(changed)

    def test_addendum_is_bound_separately_without_changing_original_authority(self):
        old=specimen("A1")["contract"]
        amendment=dict(path="/bound/addendum.md",sha256="b"*64)
        c=a2.contract(old["namespace"],old["panels"]["exploration"]["seeds"],old["excluded_seeds"],
            old["authority"],stage="A1",authority_addendum=amendment)
        validate_contract(c)
        self.assertEqual(c["authority"],old["authority"])
        self.assertEqual(c["authority_addendum"],amendment)
        c["authority_addendum"]["extra"]=True
        with self.assertRaises(ValueError):validate_contract(c)

    def test_a1_empty_readout_retains_direct_and_raw_comparisons(self):
        r=a2.summarize(specimen("A1"),[],dict(elapsed_wall_seconds=1.,
            cpu_seconds_including_reaped_children=1.,supervisor_exit_code=1,worker_exit_code=1))
        self.assertEqual(set(r["beliefs"]),{"public","oracle"})
        self.assertNotIn("leaves",r)
        for comparison in [*r["beliefs"].values(),r["oracle_minus_public"]]:
            self.assertEqual(comparison["fixed_seed_weighted_mean_bounds"],[-1.,1.])
            self.assertIsNone(comparison["mean_delta"])
        self.assertEqual(r["continuation_alias_cell_counts"]["oracle"]["UNSTARTED"],1600)

    def test_a1_direct_contrast_cancels_unknown_shared_raw(self):
        public,oracle=a2.configurations("A1")[1:]
        evidence=dict(actions={"raw":0,public.identity:1,oracle.identity:2},
            validated_outcome_sha256={f"{action}:{rep}":"a"*64 for action in (1,2) for rep in range(8)})
        outcomes=[dict(action=action,replicate=rep,status="COMPLETE",signed_outcome=score)
            for action,score in ((1,-1),(2,1)) for rep in range(8)]
        self.assertEqual(a2.direct_belief_interval(evidence,outcomes),[1.,1.])
        evidence["actions"][oracle.identity]=1
        self.assertEqual(a2.direct_belief_interval(evidence,[]),[0.,0.])
        evidence["actions"].pop(public.identity)
        self.assertEqual(a2.direct_belief_interval(evidence,[]),[-1.,1.])

    def test_a1_runtime_keeps_matching_original_observers_without_future_truth(self):
        checkpoint=SimpleNamespace(checkpoint_sha256="a"*64,showdown_source_sha256="b"*16)
        public,oracle=a2.configurations("A1")[1:]
        p,o=(a2.runtime_recipe(cfg,checkpoint,"A1") for cfg in (public,oracle))
        for runtime in (p,o):
            self.assertNotIn("original_actor_annotation_handoff",runtime)
            self.assertNotIn("compact_native_atomic_work",runtime)
            self.assertNotIn("team_oracle",runtime)
        for field in ("incumbent","native_atomic_work","timed_native_atomic_work","full_public_request_observation"):
            self.assertEqual(p[field],o[field])
