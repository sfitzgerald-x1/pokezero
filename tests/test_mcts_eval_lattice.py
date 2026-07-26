import unittest
from pokezero.mcts_eval.lattice import time_lattice_cell
from pokezero.mcts_eval.manifest import SearchConfig
from pokezero.mcts_eval.resolver import CheckpointContract
from tests.test_mcts_eval_timing_corpus import _record

C = CheckpointContract(checkpoint_path="/c.pt", checkpoint_sha256="a"*64, policy_id="p",
    schema_version="pokezero.observation.v3", token_count=87, categorical_feature_count=51,
    numeric_feature_count=155, transition_token_count=64, architecture={}, feature_masks={},
    model_device="cpu")

class T(unittest.TestCase):
    def test_gate_failure_is_a_result_with_partial_sample(self):
        recs = [_record(i) for i in range(200)]
        slow = lambda rec, cfg: {"max_depth_reached": 3, "root_action": "move 1"}
        import time as _t
        def slow_decide(rec, cfg):
            _t.sleep(0.001)
            return {"max_depth_reached": 3, "root_action": "move 1"}
        row = time_lattice_cell(SearchConfig(depth=10, sims=8192), records=recs,
                                contract=C, decide=slow_decide, gate_s=0.0005)
        self.assertTrue(row.gate_failed)
        self.assertFalse(row.eligible)
        self.assertEqual(row.decisions_timed, 64)   # stops at the plan's minimum
        self.assertGreater(row.mean_wall_s, 0)

    def test_fast_cell_is_eligible_and_reports_phases(self):
        recs = [_record(i) for i in range(64)]
        def fast(rec, cfg):
            return {"max_depth_reached": cfg.depth-1, "root_action": "move 2",
                    "encode_s": 0.01, "model_s": 0.02, "tree_s": 0.005}
        row = time_lattice_cell(SearchConfig(depth=4, sims=512), records=recs, contract=C, decide=fast)
        self.assertTrue(row.eligible)
        self.assertEqual(row.decisions_timed, 64)
        self.assertEqual(row.cap_hit_rate, 1.0)
        self.assertAlmostEqual(row.model_s, 0.02*64, places=5)
        self.assertEqual(len(row.root_argmax_by_decision), 64)

    def test_fallbacks_make_cell_ineligible(self):
        recs = [_record(i) for i in range(64)]
        def fb(rec, cfg): return {"max_depth_reached": 2, "fallbacks": 1, "root_action": "x"}
        row = time_lattice_cell(SearchConfig(depth=4, sims=512), records=recs, contract=C, decide=fb)
        self.assertFalse(row.eligible)
        self.assertEqual(row.fallbacks, 64)

if __name__ == "__main__":
    unittest.main()
