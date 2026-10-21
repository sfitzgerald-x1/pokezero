"""Small JIT-capable component checks; no model, collector or throughput claim."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

JAX_AVAILABLE = importlib.util.find_spec("jax") is not None
NODE = os.environ.get("NODE") or shutil.which("node")
SHOWDOWN = os.environ.get("POKEZERO_SHOWDOWN_ROOT")


@unittest.skipUnless(JAX_AVAILABLE, "requires isolated spike JAX environment")
class MechanicsTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as jnp
        from gpu_engine import mechanics as m
        self.jax, self.np, self.m = jax, jnp, m

    def damage(self, **kwargs):
        args = dict(level=100, power=100, attack=200, defense=200)
        args.update(kwargs)
        return int(self.m.gen3_damage(**args))

    def test_q12_rounding_preserves_pinned_unsigned_wrap(self):
        self.assertEqual(self.m.modify(self.np.array([1,2,3,4]),3,2).tolist(),[1,3,4,6])
        self.assertEqual(int(self.m.modify(2000000,3,2)),902848)
        self.assertEqual(int(self.m.modify(1048576,1,1)),0)
        self.assertEqual(self.m.modify(self.np.array([2000000,2000000]),
            self.np.array([3,3]),self.np.array([2,2])).tolist(),[902848,902848])

    def test_boosts_floor_and_clip(self):
        self.assertEqual(self.m.boosted_stat(101,self.np.array([-9,-1,0,1,9])).tolist(),[25,67,101,151,404])

    def test_base_stab_type_random_and_immunity(self):
        self.assertEqual(self.damage(),86)
        self.assertEqual(self.damage(stab=True),129)
        self.assertEqual(self.damage(stab=True,effectiveness_num=2,roll=85),219)
        self.assertEqual(self.damage(effectiveness_den=4,roll=85),17)
        self.assertEqual(self.damage(immune=True),0)
        self.assertEqual(self.damage(power=0),0)

    def test_burn_and_weather_happen_before_plus_two(self):
        self.assertEqual(self.damage(burned=True),44)
        self.assertEqual(self.damage(burned=True,guts=True),86)
        self.assertEqual(self.damage(burned=True,physical=False),86)
        self.assertEqual(self.damage(weather_num=3,weather_den=2),128)

    def test_critical_boost_asymmetry_and_screen(self):
        self.assertEqual(self.damage(critical=True,attack_stage=-6,defense_stage=6),172)
        self.assertGreater(self.damage(critical=True,attack_stage=2),172)
        self.assertGreater(self.damage(critical=True,defense_stage=-2),172)
        self.assertEqual(self.damage(screen=True),44)
        self.assertEqual(self.damage(screen=True,critical=True),172)
        self.assertEqual(self.damage(power=960,attack=1000,defense=1,critical=True),564228)

    def test_identity_stab_event_is_skipped_only_for_typeless_moves(self):
        self.assertEqual(self.damage(power=1300,attack=1000,defense=1),43426)
        self.assertEqual(self.damage(power=1300,attack=1000,defense=1,typeless=True),1092002)

    def test_resolved_item_stat_multipliers_and_low_physical_floor(self):
        self.assertEqual(self.damage(attack_num=3,attack_den=2),128)
        self.assertEqual(self.damage(attack_num=2),170)
        self.assertEqual(self.damage(level=1,power=1,attack=1,defense=1000),3)
        self.assertEqual(self.damage(level=1,power=1,attack=1,defense=1000,physical=False),2)

    def test_accuracy_separate_stages_and_fractional_threshold(self):
        m = self.m
        self.assertTrue(bool(m.accuracy_hit(90,-1,0,67)))
        self.assertFalse(bool(m.accuracy_hit(90,-1,0,68)))
        self.assertTrue(bool(m.accuracy_hit(10,6,-6,89)))
        self.assertFalse(bool(m.accuracy_hit(10,6,-6,90)))
        self.assertTrue(bool(m.accuracy_hit(1,-6,6,99,always_hit=True)))

    def test_gen3_paralysis_speed_quarter(self):
        self.assertEqual(self.m.paralyzed_speed(self.np.array([101,103,1]),True).tolist(),[25,26,0])

    def test_sleep_wakes_before_action_and_rest_counter(self):
        m = self.m
        r = m.before_move(m.SLP,1)
        self.assertEqual((int(r.status),int(r.counter),bool(r.can_act)),(m.NONE,0,True))
        r = m.before_move(m.SLP,3)
        self.assertEqual((int(r.counter),bool(r.can_act)),(2,False))
        self.assertTrue(bool(m.before_move(m.SLP,3,sleep_usable=True).can_act))
        self.assertTrue(bool(m.before_move(m.SLP,2,early_bird=True).can_act))

    def test_freeze_draw_and_defrost_event_boundary(self):
        m = self.m
        self.assertEqual(int(m.before_move(m.FRZ,0,thaw_roll=19).status),m.NONE)
        self.assertFalse(bool(m.before_move(m.FRZ,0,thaw_roll=20).can_act))
        r = m.before_move(m.FRZ,0,defrost_move=True)
        self.assertTrue(bool(r.can_act))
        self.assertEqual(int(r.status),m.FRZ)

    def test_confusion_paralysis_and_flinch_event_order(self):
        m = self.m
        r = m.before_move(m.NONE,0,2,confusion_roll=50)
        self.assertTrue(bool(r.confusion_self_hit)); self.assertFalse(bool(r.can_act))
        self.assertTrue(bool(m.before_move(m.NONE,0,1,confusion_roll=99).can_act))
        self.assertFalse(bool(m.before_move(m.PAR,0,paralysis_roll=24).can_act))
        self.assertTrue(bool(m.before_move(m.PAR,0,paralysis_roll=25).can_act))
        self.assertEqual(int(m.before_move(m.NONE,0,3,flinched=True).confusion_counter),3)
        self.assertEqual(int(m.before_move(m.SLP,3,3).confusion_counter),3)

    def test_toxic_floors_before_multiplication_caps_and_skips_bench(self):
        m = self.m
        r = m.major_status_residual(500,101,m.TOX,2)
        self.assertEqual((int(r.damage),int(r.counter)),(18,3))
        self.assertEqual(int(m.major_status_residual(500,101,m.TOX,15).damage),90)
        r = m.major_status_residual(500,101,m.TOX,2,active=False)
        self.assertEqual((int(r.hp),int(r.counter),int(r.damage)),(500,2,0))

    def test_residual_minimum_lethality_and_fainted_do_not_tick(self):
        m = self.m
        self.assertEqual(int(m.major_status_residual(1,7,m.BRN,0).hp),0)
        r = m.major_status_residual(0,100,m.TOX,2)
        self.assertEqual((int(r.counter),int(r.damage)),(2,0))
        self.assertEqual(int(m.major_status_residual(100,101,m.PSN,0).damage),12)

    def test_jit_batched_components_equal_eager(self):
        m = self.m
        fn = lambda atk: m.gen3_damage(level=80,power=95,attack=atk,defense=150,stab=True,roll=85)
        x = self.np.arange(100,116)
        result = self.jax.jit(fn)(x); result.block_until_ready()
        self.assertTrue(bool(self.np.all(result == fn(x))))
        fn = self.jax.jit(m.major_status_residual)
        r = fn(self.np.array([100,100]),self.np.array([101,101]),self.np.array([m.BRN,m.TOX]),self.np.array([0,2]))
        r.hp.block_until_ready()
        self.assertEqual(r.damage.tolist(),[12,18])

    @unittest.skipUnless(NODE and SHOWDOWN, "requires pinned Showdown and Node arithmetic oracle")
    def test_against_pinned_showdown_arithmetic(self):
        defaults = dict(level=100,power=100,attack=200,defense=200,
            attack_stage=0,defense_stage=0,physical=True,critical=False,stab=False,typeless=False,
            burned=False,guts=False,attack_num=1,attack_den=1,defense_num=1,
            defense_den=1,weather_num=1,weather_den=1,phase1_num=1,phase1_den=1,screen=False,
            effectiveness_num=1,effectiveness_den=1,immune=False,roll=100)
        changes = [{},dict(burned=True),dict(phase1_num=3,phase1_den=2),
            dict(phase1_num=3,phase1_den=2,weather_num=3,weather_den=2,burned=True),
            dict(critical=True,attack_stage=-3,defense_stage=4),
            dict(stab=True,roll=85),dict(weather_num=3,weather_den=2,screen=True),
            dict(attack=337,defense=153,attack_stage=3,defense_stage=-2,
                 attack_num=3,attack_den=2,critical=True,stab=True,roll=91),
            dict(attack=1,defense=1000,power=1,level=1),
            dict(attack=1,defense=1000,power=1,level=1,physical=False),
            dict(effectiveness_num=4),dict(effectiveness_den=4,roll=85),
            dict(immune=True),dict(power=0),dict(burned=True,guts=True),
            dict(weather_den=2,burned=True,critical=True,stab=True),
            dict(physical=False,attack_stage=-2,defense_stage=3,roll=87),
            dict(power=60000,attack=2000,defense=1,stab=True,critical=True),
            dict(power=960,attack=1000,defense=1,critical=True,stab=False),
            dict(power=1300,attack=1000,defense=1),
            dict(power=1300,attack=1000,defense=1,typeless=True)]
        rows = [{**defaults,**c} for c in changes]
        helper = Path(__file__).with_name("gpu_damage_oracle.cjs")
        result = subprocess.run([NODE,str(helper),str(Path(SHOWDOWN).resolve())],
            input=json.dumps(rows),text=True,capture_output=True,check=True)
        expected = json.loads(result.stdout)
        self.assertEqual([int(self.m.gen3_damage(**r)) for r in rows],expected["damage"])
        self.assertEqual(int(self.m.modify(2000000,3,2)),expected["large_modifier"])
        self.assertEqual(self.m.paralyzed_speed(self.np.array([101,103,1]),True).tolist(),
                         expected["paralysis_speed"])


if __name__ == "__main__":
    unittest.main()
