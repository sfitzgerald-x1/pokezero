"""Reporting/scope guards; these tests do not run a performance measurement."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
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

    def test_pool_path_replacement_cannot_change_loaded_snapshot_digest(self):
        np = self.b.np
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"pool.npz"
            replacement = Path(directory)/"replacement.npz"
            fields = ("species","level","types","stats","moves","pp","item","ability",
                      "hidden_power_type","hidden_power_power")
            for destination,value in ((path,1),(replacement,2)):
                np.savez(destination,**{name:np.array([value]) for name in fields},
                    metadata_utf8=np.frombuffer(json.dumps({"test":value}).encode(),dtype=np.uint8))
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            real_load = np.load
            def replace_then_load(snapshot,**kwargs):
                replacement.replace(path)
                return real_load(snapshot,**kwargs)
            with mock.patch.object(np,"load",side_effect=replace_then_load):
                pool,metadata,digest = self.b.load_bound_pool(path)
            self.assertEqual(pool["species"].tolist(),[1])
            self.assertEqual(metadata,{"test":1})
            self.assertEqual(digest,expected)
            self.assertNotEqual(digest,hashlib.sha256(path.read_bytes()).hexdigest())

    def test_in_place_pool_change_during_read_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"pool.npz"
            path.write_bytes(b"fixture")
            first = SimpleNamespace(st_dev=1,st_ino=2,st_size=7,st_mtime_ns=3)
            changed = SimpleNamespace(st_dev=1,st_ino=2,st_size=7,st_mtime_ns=4)
            with mock.patch.object(self.b.os,"fstat",side_effect=[first,changed]), \
                    self.assertRaisesRegex(RuntimeError,"changed while reading"):
                self.b.load_bound_pool(path)


if __name__ == "__main__":
    unittest.main()
