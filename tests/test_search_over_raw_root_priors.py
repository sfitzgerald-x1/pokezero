"""Pure diagnostic arithmetic/roster tests; no source, model or search."""
from copy import deepcopy
import math
import unittest
from pokezero.mcts_eval.search_over_raw_root_priors import make_contract,validate_contract,distribution_metrics,seed_cluster_interval,buffer_arrays


class RootPriorTests(unittest.TestCase):
    def test_native_buffer_types_are_canonical_not_model_tensor_types(self):
        import numpy as np
        from types import SimpleNamespace
        from pokezero.golden_corpus import GOLDEN_ARRAY_FIELDS
        arrays={name:np.ones((2,3) if rank == 2 else (2,),dtype=dtype)
            for name,dtype,rank in GOLDEN_ARRAY_FIELDS}
        result=buffer_arrays({k:v.tobytes() for k,v in arrays.items()},SimpleNamespace(**arrays))
        for name,want in arrays.items():
            self.assertEqual(result[name].dtype,want.dtype)
            np.testing.assert_array_equal(result[name],want)

    def contract(self):
        return make_contract("authored",list(range(100,132)),[1,2,3],dict(path="/authored/steer",sha256="a"*64))

    def test_exact_100_slots_and_32_fresh_seeds_without_continuations(self):
        c=self.contract();validate_contract(c)
        self.assertEqual(len(c["panels"]["exploration"]["root_slots"]),100)
        self.assertEqual(c["continuations"],0)
        self.assertIsNone(c["wall_clock_search_budget"])
        self.assertEqual(c["consecutive_unavailable_cap"],3)
        self.assertFalse(c["source_actor_history_capture"])
        for key in ("native_iterations_per_world","native_worlds","holdout_authorized"):
            broken=deepcopy(c);broken[key]=True
            with self.assertRaises(ValueError):validate_contract(broken)
        with self.assertRaises(ValueError):make_contract("x",[1]+list(range(101,132)),[1],{})

    def test_identical_and_opposed_distributions_no_hidden_smoothing(self):
        m=distribution_metrics([.25,.75],[.25,.75],1)
        self.assertTrue(m["argmax_agreement"])
        self.assertEqual(m["kl_left_to_right"],0)
        self.assertEqual(m["right_probability_of_raw_argmax"],.75)
        m=distribution_metrics([1.,0.],[0.,1.],0)
        self.assertFalse(m["argmax_agreement"])
        self.assertEqual(m["kl_left_to_right"],"INFINITE_POSITIVE_MASS_AT_ZERO")
        self.assertEqual(m["right_probability_of_raw_argmax"],0)

    def test_directed_kl_and_invalid_probabilities(self):
        m=distribution_metrics([.5,.5],[.25,.75],0)
        self.assertAlmostEqual(m["kl_left_to_right"],.5*math.log(2)+.5*math.log(2/3))
        self.assertNotAlmostEqual(m["kl_left_to_right"],m["kl_right_to_left"])
        for vector in ([.3,.3],[-.1,1.1],[float("nan"),0.]):
            with self.assertRaises(ValueError):distribution_metrics([.5,.5],vector,0)

    def test_descriptive_interval_resamples_whole_seeds_without_finite_infinite_kl(self):
        result=seed_cluster_interval([0.,0.,1.],[1,1,2])
        self.assertEqual(result["interval"],[0.,1.])
        self.assertEqual(result["observed_seed_clusters"],2)
        self.assertFalse(result["full_fixed_panel_inference"])
        self.assertIsNone(seed_cluster_interval([1.],[1])["interval"])
        result=seed_cluster_interval([0.,"INFINITE_POSITIVE_MASS_AT_ZERO"],[1,2])
        self.assertIsNone(result["interval"])
        self.assertEqual(result["reason"],"infinite_KL_no_finite_interval")


if __name__ == "__main__":unittest.main()
