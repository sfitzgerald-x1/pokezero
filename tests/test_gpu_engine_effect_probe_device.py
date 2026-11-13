"""The effect probe's device mode end to end: a representative subset of the committed scenarios with --device's comparison.

Each case also runs through step_with_public, its engine state must equal step_with_chance's, and the device public
carry must equal B2's fold of the case's Showdown lines (gpu_engine/effect_probe_device.py). step_with_public's CPU
compile peaks far above the light tier at any optimization level, so the subset runs it eagerly
(Engine.enable_device(eager=True): the same program op by op, no step compile) on one lane; step_with_chance and B2's
fold stay jitted. Eager steps take minutes, so this module is in the extended tier (tests/gpu_component_extended.json),
not per pull request; the device mode's pure parts stay in tests/test_gpu_engine_effect_probe.py, and the full set runs
as a pod script before every engine merge.
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import unittest

from _showdown_root import requires_showdown, showdown_root
from gpu_engine import effect_probe as probe

ROOT = Path(__file__).resolve().parents[1]
NODE = os.environ.get('NODE') or shutil.which('node')
HAS_JAX = bool(importlib.util.find_spec('jax'))
POOL = os.environ.get('POKEZERO_GPU_TEAM_POOL')
# Device mode's end-to-end subset (eager steps are slow): a White Herb received by Trick with a negative stage, the WH10
# catcher's site (Trick's -item lines after the herb's -enditem / -clearnegativeboost, B2's broken pairing).
DEVICE_SUBSET = ('wh_trick_receive_negative',)


@requires_showdown()
@unittest.skipUnless(NODE and HAS_JAX and POOL, 'requires node, JAX and POKEZERO_GPU_TEAM_POOL')
class DeviceModeTests(unittest.TestCase):
    """A representative subset in device mode: state results unchanged and every case EXACT on the public path."""
    @classmethod
    def setUpClass(cls):
        cls.metadata, _ = probe.load_pool(POOL, None)
        cls.scenarios = [probe.load_scenario(ROOT / f'gpu_engine/effect_probes/{name}.json') for name in DEVICE_SUBSET]
        lanes = max(len(probe.enumerate_cases(s)) for s in cls.scenarios)
        cls.sd = probe.Showdown(NODE, showdown_root())
        cls.device = probe.Device.for_scenarios(cls.scenarios, cls.sd, showdown_root(), cls.metadata, lanes)
        cls.engine = probe.Engine(cls.metadata, lanes);cls.engine.enable_device(cls.device.belief_tables, eager=True)

    @classmethod
    def tearDownClass(cls):
        cls.sd.close()

    def test_subset_is_exact_on_the_device_public_path(self):
        for scenario in self.scenarios:
            with self.subTest(scenario=scenario['move']):
                result = probe.run_scenario(scenario, self.engine, self.sd, self.metadata, device=self.device)
                self.assertTrue(result['expectation_met'], result['expectation_detail'])
                self.assertTrue(result['device']['state_equal'])
                self.assertTrue(result['device']['met'], result['device']['detail'] + ' ' + json.dumps(
                    [s for r in result['results'] for s in r['device']['steps'] if s['diffs']][:2], default=str)[:2000])
                self.assertGreater(result['device']['steps_compared'], result['cases'])


if __name__ == '__main__':
    unittest.main()
