"""Benchmark preparation validates the real helper before any model access."""
from dataclasses import replace
import unittest
from unittest.mock import patch
from gpu_engine import benchmark_retained_games as benchmark
import test_gpu_engine_toxic_runtime as fixture


class ValidatedGPUReached(RuntimeError):
    pass


class BenchmarkToxicAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.ToxicRuntimeTests.setUpClass()
        cls.source=fixture.ToxicRuntimeTests

    def prepare(self,helper):
        r=self.source.runtime
        return benchmark.prepare(r['request_compiled'],r['own_compiled'],'/no/model',
            expected_weights_sha256='c'*64,expected_checkpoint_sha256='a'*64,
            enable_unqualified=True,leftovers_compiled=r['leftovers_compiled'],leftovers_mode='retained',
            toxic_compiled=helper,toxic_mode='kernel-event',frame_input_mode='source-witness')

    def test_original_helper_reaches_gpu_boundary_after_real_mode_validation(self):
        with patch.object(benchmark.precision_benchmark,'digest',return_value='c'*64), \
                patch.object(benchmark,'sole_gpu',side_effect=ValidatedGPUReached), \
                patch.object(benchmark.champion,'load') as load:
            with self.assertRaises(ValidatedGPUReached):self.prepare(self.source.toxic)
            load.assert_not_called()

    def test_all_default_modes_keep_validation_order_before_gpu(self):
        r=self.source.runtime
        with patch.object(benchmark.precision_benchmark,'digest',return_value='c'*64), \
                patch.object(benchmark,'sole_gpu',side_effect=ValidatedGPUReached), \
                patch.object(benchmark.driver,'prepare') as prepare:
            with self.assertRaises(ValidatedGPUReached):
                benchmark.prepare(r['request_compiled'],r['own_compiled'],'/no/model',
                    expected_weights_sha256='c'*64,expected_checkpoint_sha256='a'*64,
                    enable_unqualified=True)
            prepare.assert_not_called()

    def test_cloned_helper_refuses_before_checkpoint_filesystem_or_gpu(self):
        with patch.object(benchmark.precision_benchmark,'digest') as digest, \
                patch.object(benchmark,'sole_gpu') as gpu, \
                patch.object(benchmark.champion,'load') as load:
            with self.assertRaises(ValueError):self.prepare(replace(self.source.toxic))
            digest.assert_not_called();gpu.assert_not_called();load.assert_not_called()


if __name__=='__main__':unittest.main()
