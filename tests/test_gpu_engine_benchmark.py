"""Reporting/scope guards; these tests do not run a performance measurement."""
import importlib.util
import unittest
from unittest import mock


@unittest.skipUnless(importlib.util.find_spec("jax"),"requires isolated spike JAX")
class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        from gpu_engine import benchmark
        self.b = benchmark

    def test_samples_require_positive_finite_measurements(self):
        for samples in ([],[0],[float("nan")],[float("inf")],[-1]):
            with self.assertRaises(ValueError): self.b.summarize_samples(samples)
        stats = self.b.summarize_samples([1.0,2.0,3.0])
        self.assertEqual(stats["engine_step_ms"],2.0)
        self.assertEqual(stats["median_step_ms"],2.0)

    def test_exactly_one_gpu_is_required_not_a_cpu_fallback(self):
        for devices in ([],[mock.sentinel.a,mock.sentinel.b]):
            with mock.patch.object(self.b.jax,"devices",return_value=devices),self.assertRaises(RuntimeError):
                self.b.sole_gpu()
        with mock.patch.object(self.b.jax,"devices",return_value=[mock.sentinel.a]):
            self.assertIs(self.b.sole_gpu(),mock.sentinel.a)

    def test_partial_step_table_never_manufactures_game_rate(self):
        result = dict(device="test GPU",jax="test",rows=[dict(batch=256,engine_step_ms=1.25)])
        text = self.b.render_markdown(result)
        self.assertIn("NOT the final throughput benchmark",self.b.__doc__)
        self.assertIn("| 256 | 1.250 | — | — | — | — | — | — |",text)
        self.assertIn("No completed games",text)
        self.assertIn("includes host dispatch",text)

    def test_invalid_workload_refuses_before_device_or_pool_access(self):
        for options in (dict(batches=()),dict(batches=(0,)),dict(repeats=1),dict(warmup=0)):
            with mock.patch.object(self.b,"sole_gpu") as gpu,self.assertRaises(ValueError):
                self.b.measure("not-a-pool",**options)
            gpu.assert_not_called()


if __name__ == "__main__":
    unittest.main()
