"""Authored source-domain guards; no transitions, panel, or coverage credit."""
from fractions import Fraction
from functools import partial
import unittest
from unittest.mock import patch

from gpu_engine import chance_consumers as c
from gpu_engine import future_state as mapping
from gpu_engine.finite_law import COORDINATES, partition
from test_gpu_engine_future_state import action_fixture, FAKE


def fixture():
    data, meta = action_fixture()
    extra = ["overheat", "psychoboost", "struggle", "icebeam", "thunderbolt", "thunder", "doubleteam", "haze",
             "raindance", "sunnyday", "sleeptalk", "encore", "roar", "protect", "spore", "leechseed", "painsplit", "reversal", "flail", "tailglow", "bellydrum"]
    meta["vocabulary"]["moves"] += extra
    return data, meta


def table(meta):
    names = meta["vocabulary"]["moves"]
    field = lambda values, default=0: [values.get(name, default) for name in names]
    return dict(power=field({"overheat":140,"psychoboost":140,"tackle": 40, "struggle": 50, "icebeam": 95, "thunderbolt": 95, "thunder": 120}),
        accuracy=field({"overheat":90,"psychoboost":90,"thunder": 70}, 100), always_hit=field({"splash": True, "haze": True, "doubleteam": True}, False),
        priority=field({"protect": 3, "roar": -6}), struggle=names.index("struggle"),
        effects=dict(status=field({"spore": 5}),
            self_boosts=field({"doubleteam": [0, 0, 0, 0, 0, 0, 1], "tailglow": [0,0,2,0,0,0,0]}, [0]*7),
            haze=field({"haze": True}, False), protect=field({"protect": True}, False), endure=field({}),
        ), secondary=dict(chance=field({"icebeam": 10, "thunderbolt": 10, "thunder": 30}),
            status=field({"icebeam": 6, "thunderbolt": 1, "thunder": 1}),
            contact=field({}, False), static=[False]*len(meta['vocabulary']['abilities']),
            synchronize=[False]*len(meta['vocabulary']['abilities'])),
        weather=dict(move_weather=field({"raindance": 1, "sunnyday": 2}), thunder=field({"thunder": True}, False)),
        sleep_talk=dict(move=field({"sleeptalk": True}, False),
            callable=[name not in {"_none", "sleeptalk", "struggle"} for name in names]),
        encore=dict(move=field({"encore": True}, False),
            allowed_last=[name not in {"_none", "struggle", "encore"} for name in names]),
        phazing=dict(move=field({"roar": True}, False)),
        sustain=dict(seed=field({"leechseed": True}, False)),
        variable_moves=dict(hidden_power=field({},False),returns=field({},False),
            frustration=field({},False),hp_dependent=field({"reversal":True,"flail":True},False)),
    )


class ChanceConsumerTests(unittest.TestCase):
    def test_plain_binary64_threshold_refines_actual_integer_rolls(self):
        data,meta=fixture();self.choose(data,meta,0,'return');move=meta['vocabulary']['moves'].index('return')
        t=table(meta);t['accuracy'][move]=80
        data['boosts'][0][0][5]=2;data['boosts'][1][0][6]=5
        plan=self.plan(data,meta,kernel_table=t)
        self.assertIsInstance(plan,c.ConsumerPlan,plan)
        self.assertEqual([len(cell.members) for cell in self.cells(plan,'draws.0.6')],[51,49])

    def test_plain_actual_table_outside_accuracy_domain_also_refuses(self):
        data,meta=fixture();self.choose(data,meta,0,'return');t=table(meta)
        t['accuracy'][meta['vocabulary']['moves'].index('return')]=101
        plan=self.plan(data,meta,kernel_table=t)
        self.assertIsInstance(plan,mapping.Refused)
        self.assertEqual(plan.gaps[0].code,'ACCURACY_DOMAIN')

    def test_trace_compound_eyes_and_weather_sand_veil_cuts_are_retained(self):
        data,meta=fixture();t=self.hustle_table(data,meta)
        meta['vocabulary']['abilities']+=['compoundeyes','sandveil']
        t['combat']['hustle']+=[False,False]
        t['combat']['compound_eyes']=[False]*4+[True,False]
        t['entry']['trace']+=[False,False];t['entry']['weather_ability']+=[0,0]
        t['secondary']['static']+=[False,False];t['secondary']['synchronize']+=[False,False]
        t['weather']['sand_veil']=[False]*5+[True]
        self.choose(data,meta,0,'return');t['accuracy'][meta['vocabulary']['moves'].index('return')]=85
        data['ability'][0][0]=3;data['base_ability'][0][0]=3
        data['ability'][1][0]=5;data['base_ability'][1][0]=5
        data['ability'][1][1]=4;data['base_ability'][1][1]=4
        data['weather']=3;data['weather_counter']=5
        plan=self.plan(data,meta,actions=(0,4),kernel_table=t)
        self.assertIsInstance(plan,c.ConsumerPlan,plan)
        axis=next(a for a in plan.axes if a.coordinate=='draws.0.6')
        self.assertIn(Fraction(88),{o.argument for o in axis.observations})
        self.assertIn(Fraction(68),{o.argument for o in axis.observations})

    def hustle_table(self,data,meta,*,hidden=False):
        meta['vocabulary']['abilities']+=['hustle','trace']
        data['ability'][0][0]=2;data['base_ability'][0][0]=2
        t=table(meta);t['type']=[1]*len(meta['vocabulary']['moves'])
        t['combat']={'hustle':[False,False,True,False],
            'physical_type':[n in {'normal','ground'} for n in meta['vocabulary']['types']]}
        t['entry']={'trace':[False,False,False,True],'weather_ability':[0]*4}
        if hidden:
            self.choose(data,meta,0,'hiddenpowerice')
            move=meta['vocabulary']['moves'].index('hiddenpowerice')
            t['variable_moves']['hidden_power'][move]=True
            t['variable_moves']['dark_type']=meta['vocabulary']['types'].index('dark')
        return t

    def test_hustle_accuracy_partition_uses_actor_hidden_power_type(self):
        data,meta=fixture()
        meta['vocabulary']['types']+=['ground','dark']
        if 'hiddenpowerice' not in meta['vocabulary']['moves']:meta['vocabulary']['moves'].append('hiddenpowerice')
        t=self.hustle_table(data,meta,hidden=True)
        data['hidden_power_type'][0][0]=meta['vocabulary']['types'].index('ground')
        p=self.plan(data,meta,kernel_table=t)
        self.assertIsInstance(p,c.ConsumerPlan,p)
        self.assertEqual([len(cell.members) for cell in self.cells(p,'draws.0.6')],[80,20])
        data['hidden_power_type'][0][0]=meta['vocabulary']['types'].index('ice')
        p=self.plan(data,meta,kernel_table=t)
        self.assertEqual([len(cell.members) for cell in self.cells(p,'draws.0.6')],[100])

    def test_hustle_type_status_fractional_and_always_hit_cuts(self):
        data,meta=fixture();t=self.hustle_table(data,meta)
        self.choose(data,meta,0,'return')
        data['boosts'][0][0][5]=-4
        p=self.plan(data,meta,kernel_table=t)
        self.assertIsInstance(p,c.ConsumerPlan,p)
        self.assertEqual([len(cell.members) for cell in self.cells(p,'draws.0.6')],[43,57])
        data['boosts'][0][0][5]=0
        move=meta['vocabulary']['moves'].index('return');t['power'][move]=0
        p=self.plan(data,meta,kernel_table=t)
        self.assertEqual([len(cell.members) for cell in self.cells(p,'draws.0.6')],[80,20])
        t['always_hit'][move]=True
        p=self.plan(data,meta,kernel_table=t)
        self.assertEqual([len(cell.members) for cell in self.cells(p,'draws.0.6')],[100])

    def test_trace_hustle_closure_preserves_other_seats_accuracy_cut(self):
        data,meta=fixture();t=self.hustle_table(data,meta)
        self.choose(data,meta,0,'return')
        # Opponent's Trace entrant copies Hustle. Both possible ability states
        # remain in the conservative closure; no sampled execution is needed.
        data['ability'][0][0]=3;data['base_ability'][0][0]=3
        data['ability'][1][1]=2;data['base_ability'][1][1]=2
        p=self.plan(data,meta,actions=(0,4),kernel_table=t)
        self.assertIsInstance(p,c.ConsumerPlan,p)
        self.assertEqual([len(cell.members) for cell in self.cells(p,'draws.0.6')],[80,20])

    def test_hustle_actual_table_accuracy_outside_finite_domain_refuses_before_engines(self):
        for value in (-1,101,85.5,100.0,False):
            data,meta=fixture();t=self.hustle_table(data,meta);self.choose(data,meta,0,'return')
            t['accuracy'][meta['vocabulary']['moves'].index('return')]=value
            plan=self.plan(data,meta,kernel_table=t)
            with self.subTest(value=value):
                self.assertIsInstance(plan,mapping.Refused)
                self.assertEqual(plan.gaps[0].code,'ACCURACY_DOMAIN')

    def plan(self, data=None, meta=None, actions=(0, 0), *, cap=100_000_000, kernel_table=None):
        if data is None:
            data, meta = fixture()
        construct = partial(mapping.native_input_from_jax, module=FAKE)
        with patch.object(c.mapping, "native_input_from_jax", side_effect=construct):
            return c.plan_for_state(data, None, actions,
                table(meta) if kernel_table is None else kernel_table, meta, cap=cap)

    def choose(self, data, meta, side, name, slot=0):
        data["moves"][side][0][slot] = meta["vocabulary"]["moves"].index(name)

    def cells(self, plan, coordinate):
        axis = next(a for a in plan.axes if a.coordinate == coordinate)
        return partition(dict(COORDINATES)[coordinate], axis.observations)

    def test_synchronize_after_static_keeps_other_seats_paralysis_axis(self):
        for actor in (0,1):
            data,meta=fixture();meta['vocabulary']['abilities']+=['synchronize','static']
            data['ability'][actor][0]=2;data['base_ability'][actor][0]=2
            data['ability'][1-actor][0]=3;data['base_ability'][1-actor][0]=3
            # Return is a non-paralysis contact fixture: reflection is the
            # only path by which the waiting Static holder can become paralyzed.
            for seat in (0,1):self.choose(data,meta,seat,'return')
            move=meta['vocabulary']['moves'].index('return')
            t=table(meta);t['power'][move]=102;t['secondary']['contact'][move]=True
            t['secondary']['static'][3]=True;t['secondary']['synchronize'][2]=True
            plan=self.plan(data,meta,kernel_table=t)
            self.assertIsInstance(plan,c.ConsumerPlan,plan)
            self.assertEqual(len(self.cells(plan,f'draws.{1-actor}.2')),2)
            t['secondary']['synchronize'][2]=False
            plain=self.plan(data,meta,kernel_table=t)
            self.assertEqual(len(self.cells(plain,f'draws.{1-actor}.2')),1)

    def test_trace_copied_synchronize_is_in_reflected_status_closure(self):
        for actor in (0,1):
            data,meta=fixture();meta['vocabulary']['abilities']+=['synchronize','trace']
            meta['vocabulary']['moves'].append('bodyslam');move=len(meta['vocabulary']['moves'])-1
            data['ability'][actor][0]=2;data['base_ability'][actor][0]=2
            data['ability'][1-actor][1]=3;data['base_ability'][1-actor][1]=3
            data['moves'][actor][0][0]=move
            t=table(meta);t['power'][move]=85;t['secondary']['contact'][move]=True
            t['secondary']['chance'][move]=30;t['secondary']['status'][move]=1
            t['secondary']['synchronize'][2]=True
            t['entry']={'trace':[False,False,False,True],'weather_ability':[0,0,0,0]}
            actions=(0,4) if actor==0 else (4,0)
            plan=self.plan(data,meta,actions=actions,kernel_table=t)
            self.assertIsInstance(plan,c.ConsumerPlan,plan)
            self.assertEqual(len(self.cells(plan,f'draws.{actor}.2')),2)
            t['entry']['trace'][3]=False
            plain=self.plan(data,meta,actions=actions,kernel_table=t)
            self.assertEqual(len(self.cells(plain,f'draws.{actor}.2')),1)

    def test_synchronize_toxic_activates_both_sides_residual_status_coin(self):
        for actor in (0,1):
            data,meta=fixture();meta['vocabulary']['abilities'].append('synchronize')
            meta['vocabulary']['moves'].append('toxic');move=len(meta['vocabulary']['moves'])-1
            data['ability'][1-actor][0]=2;data['base_ability'][1-actor][0]=2
            data['moves'][actor][0][0]=move
            t=table(meta);t['effects']['status'][move]=4;t['secondary']['synchronize'][2]=True
            actions=(0,1) if actor==0 else (1,0)
            plan=self.plan(data,meta,actions=actions,kernel_table=t)
            self.assertIsInstance(plan,c.ConsumerPlan,plan)
            self.assertEqual(len(self.cells(plan,'draws.1.9')),2)
            self.assertEqual(len(self.cells(plan,f'draws.{actor}.2')),1)
            t['secondary']['synchronize'][2]=False
            plain=self.plan(data,meta,actions=actions,kernel_table=t)
            self.assertEqual(len(self.cells(plain,'draws.1.9')),1)

    def test_complete_physical_inventory_and_exact_unit_mass(self):
        plan = self.plan()
        self.assertIsInstance(plan, c.ConsumerPlan)
        self.assertEqual(tuple(a.coordinate for a in plan.axes), tuple(name for name, _ in COORDINATES))
        self.assertEqual(plan.cardinality, 4608)
        self.assertEqual(sum(case.mass for case in plan.bounded_cases()), 1)

    def test_pain_split_changes_hp_without_adding_a_chance_coordinate(self):
        data,meta=fixture()
        self.choose(data,meta,0,"painsplit")
        for hp in (1,101,999):
            data["hp"][0][0]=hp
            plan=self.plan(data,meta,actions=(0,1),cap=2)  # foe's existing unique Splash slot
            self.assertIsInstance(plan,c.ConsumerPlan,plan)
            self.assertEqual(plan.cardinality,2)  # conservative binary action-order coin
            self.assertEqual(sum(case.mass for case in plan.bounded_cases()),1)

    def test_zero_catalog_hp_moves_retain_existing_critical_and_damage_axes(self):
        for name in ('reversal','flail'):
            data,meta=fixture();self.choose(data,meta,0,name)
            plan=self.plan(data,meta,actions=(0,1))  # foe's unique Splash slot
            self.assertIsInstance(plan,c.ConsumerPlan,plan)
            self.assertEqual(len(self.cells(plan,'draws.0.4')),3)
            self.assertEqual(len(self.cells(plan,'draws.0.5')),16)
            self.assertEqual(len(self.cells(plan,'draws.1.5')),1)
            self.assertEqual(sum(case.mass for case in plan.bounded_cases()),1)

    def test_tail_glow_changes_no_accuracy_or_random_coordinate(self):
        data,meta=fixture();self.choose(data,meta,0,'tailglow')
        for stage in (-6,0,5,6):
            data['boosts'][0][0][2]=stage
            plan=self.plan(data,meta,actions=(0,1),cap=2)
            self.assertIsInstance(plan,c.ConsumerPlan,plan)
            self.assertEqual(plan.cardinality,2)
            self.assertEqual(sum(case.mass for case in plan.bounded_cases()),1)

    def test_successful_hit_spa_drop_adds_no_random_coordinate(self):
        for name in ('overheat','psychoboost'):
            baseline=None
            for stage in (-6,-5,0,6):
                data,meta=fixture();self.choose(data,meta,0,name)
                data['boosts'][0][0][2]=stage
                plan=self.plan(data,meta,actions=(0,1))
                self.assertIsInstance(plan,c.ConsumerPlan,plan)
                self.assertEqual(sum(case.mass for case in plan.bounded_cases()),1)
                self.assertGreater(plan.cardinality,2)  # full hit/crit/damage law remains
                signature=tuple((a.coordinate,a.observations) for a in plan.axes)
                if baseline is None:baseline=signature
                self.assertEqual(signature,baseline)

    def test_earlier_freeze_and_paralysis_activate_later_status_gates(self):
        data, meta = fixture()
        self.choose(data, meta, 0, "icebeam")
        self.choose(data, meta, 1, "thunderbolt")
        plan = self.plan(data, meta)
        self.assertEqual(len(self.cells(plan, "draws.1.1")), 2)
        self.assertEqual(len(self.cells(plan, "draws.0.2")), 2)
        self.assertEqual(len(self.cells(plan, "draws.0.1")), 1)

    def test_belly_drum_hp_and_attack_change_add_no_random_coordinate(self):
        data,meta=fixture();self.choose(data,meta,0,'bellydrum')
        for hp,atk in ((300,0),(150,-6),(151,6)):
            data['hp'][0][0]=hp;data['boosts'][0][0][0]=atk
            plan=self.plan(data,meta,actions=(0,1),cap=2)
            self.assertIsInstance(plan,c.ConsumerPlan,plan)
            self.assertEqual(plan.cardinality,2)
            self.assertEqual(sum(case.mass for case in plan.bounded_cases()),1)

    def test_haze_boosts_and_weather_refine_the_same_accuracy_coordinate(self):
        data, meta = fixture()
        data["boosts"][0][0][5] = -1
        self.choose(data, meta, 0, "thunder")
        self.choose(data, meta, 1, "haze")
        plan = self.plan(data, meta)
        axis = next(a for a in plan.axes if a.coordinate == "draws.0.6")
        self.assertEqual({o.argument for o in axis.observations}, {Fraction(53), Fraction(70)})
        self.choose(data, meta, 1, "sunnyday")
        plan = self.plan(data, meta)
        axis = next(a for a in plan.axes if a.coordinate == "draws.0.6")
        self.assertIn(Fraction(38), {o.argument for o in axis.observations})
        self.choose(data, meta, 1, "doubleteam")
        plan = self.plan(data, meta)
        self.assertGreater(len(self.cells(plan, "draws.0.6")), 1)

    def test_encore_reachable_override_dispatches_sleeptalk_and_zero_pp_callable(self):
        data, meta = fixture()
        self.choose(data, meta, 0, "encore")
        self.choose(data, meta, 1, "sleeptalk", 1)
        self.choose(data, meta, 1, "icebeam", 2)
        # The diagnostic identity may be stale; the retained slot owns Encore.
        data["last_move"][1][0] = 0
        data['last_action_kind'][1] = 2
        data['last_action_index'][1] = 1
        data["pp"][1][0][2] = 0
        plan = self.plan(data, meta)
        self.assertIsInstance(plan, c.ConsumerPlan, plan)
        self.assertIn(meta["vocabulary"]["moves"].index("icebeam"), plan.reachable_moves[1])
        self.assertEqual(len(self.cells(plan, "sleep_talk.1")), 3)
        self.assertEqual(len(self.cells(plan, "draws.0.7")), 4)
        self.assertEqual(len(self.cells(plan, "draws.0.1")), 2)

    def test_phazing_retains_all_bench_participants_and_joint_moduli(self):
        data, meta = fixture()
        self.choose(data, meta, 0, "roar")
        data["hp"][1][4] = 0
        plan = self.plan(data, meta)
        self.assertEqual(plan.reachable_participants[1], (0, 1, 2, 3, 5))
        self.assertEqual(len(self.cells(plan, "phazing.0")), 60)

    def test_secondary_union_preserves_grace_correlation_not_independent_events(self):
        data, meta = fixture()
        self.choose(data, meta, 0, "icebeam")
        plan = self.plan(data, meta)
        cells = self.cells(plan, "secondary.0")
        self.assertEqual([cell.mass for cell in cells], [Fraction(1, 10), Fraction(1, 10), Fraction(4, 5)])

    def test_optional_legacy_trace_table_none_or_absent_stays_inactive(self):
        data,meta=fixture()
        for entry in ({'weather_ability':[0,0],'trace':None},{'weather_ability':[0,0]}):
            t=table(meta);t['entry']=entry
            plan=self.plan(data,meta,actions=(0,1),kernel_table=t)
            self.assertIsInstance(plan,c.ConsumerPlan,plan)
            self.assertEqual(len(self.cells(plan,'phazing.0')),1)

    def test_trace_recipient_copying_static_keeps_contact_axis_both_seats(self):
        for actor in (0,1):
            data,meta=fixture();meta['vocabulary']['abilities']+=['static','trace']
            meta['vocabulary']['moves'].append('bodyslam')
            body=meta['vocabulary']['moves'].index('bodyslam')
            data['moves'][actor][0][0]=body
            data['ability'][actor][0]=2;data['base_ability'][actor][0]=2
            data['ability'][1-actor][1]=3;data['base_ability'][1-actor][1]=3
            t=table(meta);t['entry']={'trace':[False,False,False,True],'weather_ability':[0,0,0,0]}
            t['secondary']['static'][2]=True
            t['power'][body]=85
            t['secondary']['contact'][body]=True
            t['secondary']['chance'][body]=30
            t['secondary']['status'][body]=1
            actions=(0,4) if actor==0 else (4,0)
            plan=self.plan(data,meta,actions=actions,kernel_table=t)
            self.assertIsInstance(plan,c.ConsumerPlan,plan)
            self.assertEqual([x.mass for x in self.cells(plan,f'phazing.{actor}')],
                [Fraction(1,3),Fraction(2,3)])
            t['entry']['trace'][3]=False
            without_copy=self.plan(data,meta,actions=actions,kernel_table=t)
            self.assertEqual(len(self.cells(without_copy,f'phazing.{actor}')),1)

    def test_static_has_exact_thirds_on_one_existing_auxiliary_axis(self):
        data, meta = fixture(); meta['vocabulary']['abilities'].append('static')
        data['ability'][1][0] = 2
        t = table(meta); t['secondary']['static'][2] = True
        t['secondary']['contact'][meta['vocabulary']['moves'].index('tackle')] = True
        plan = self.plan(data, meta, actions=(0, 1), kernel_table=t)
        self.assertIsInstance(plan, c.ConsumerPlan, plan)
        self.assertEqual([x.mass for x in self.cells(plan, 'phazing.0')], [Fraction(1,3), Fraction(2,3)])
        self.assertEqual(len(self.cells(plan, 'phazing.1')), 1)
        self.assertEqual(len(self.cells(plan, 'draws.0.2')), 2)
        data['ability'][1][0] = 0
        plan = self.plan(data, meta, actions=(0, 1), kernel_table=t)
        self.assertEqual(len(self.cells(plan, 'phazing.0')), 1)
        data['ability'][1][1] = 2
        plan = self.plan(data, meta, actions=(0, 4), kernel_table=t)
        self.assertEqual(len(self.cells(plan, 'phazing.0')), 2)

    def test_caller_union_refines_shared_phazing_static_axis_without_multiplying_it(self):
        data, meta = fixture(); meta['vocabulary']['abilities'].append('static')
        data['ability'][1][0] = 2
        self.choose(data, meta, 0, 'sleeptalk')
        self.choose(data, meta, 0, 'tackle', 1)
        self.choose(data, meta, 0, 'roar', 2)
        t = table(meta); t['secondary']['static'][2] = True
        t['secondary']['contact'][meta['vocabulary']['moves'].index('tackle')] = True
        plan = self.plan(data, meta, actions=(0, 1), kernel_table=t)
        self.assertIsInstance(plan, c.ConsumerPlan, plan)
        self.assertEqual(len(self.cells(plan, 'phazing.0')), 60)
        self.assertEqual(sum(x.mass for x in self.cells(plan, 'phazing.0')), 1)
        t['secondary']['contact'][meta['vocabulary']['moves'].index('roar')] = True
        out = self.plan(data, meta, actions=(0, 1), kernel_table=t)
        self.assertIsInstance(out, mapping.Refused)
        self.assertEqual(out.gaps[0].code, 'CONTACT_PHAZING_OVERLAP')

    def test_stall_only_uses_admitted_denominators_and_one_shared_axis(self):
        data, meta = fixture()
        self.choose(data, meta, 0, "protect")
        data["protect_counter"][0][0] = 4
        plan = self.plan(data, meta)
        self.assertEqual([cell.mass for cell in self.cells(plan, "draws.0.8")], [Fraction(1, 4), Fraction(3, 4)])
        data["protect_counter"][0][0] = 3
        self.assertIsInstance(self.plan(data, meta), mapping.Refused)

    def test_literal_order_coordinate_mapping_and_commutativity_proofs(self):
        data, meta = fixture()
        data["status"][0][0] = 2
        data["status"][1][0] = 3
        data["volatile"][0][0] = data["volatile"][1][0] = 4
        plan = self.plan(data, meta)
        self.assertEqual(len(self.cells(plan, "draws.1.9")), 2)
        self.assertEqual(len(self.cells(plan, "draws.0.10")), 2)
        self.assertEqual(len(self.cells(plan, "draws.1.10")), 1)
        self.assertEqual(len(self.cells(plan, "draws.0.9")), 1)
        self.assertIn("own live holder", next(a for a in plan.axes if a.coordinate == "draws.0.9").inactive_reason)
        plan = self.plan(actions=(4, 4))
        self.assertEqual(len(self.cells(plan, "draws.0.11")), 2)
        self.assertEqual(len(self.cells(plan, "draws.1.10")), 1)

    def test_cap_precount_refuses_without_case_iteration_or_truncation(self):
        out = self.plan(cap=4607)
        self.assertIsInstance(out, mapping.Refused)
        self.assertEqual(out.gaps[0].code, "CHANCE_CARDINALITY_CAP")
        self.assertIn("required=4608", out.gaps[0].detail)
        self.assertEqual(self.plan(cap=0).gaps[0].code, "INVALID_EXPLICIT_CASE_CAP")

    def test_new_unsupported_successor_refuses_whole_slot_and_raw_is_retained(self):
        data, meta = fixture()
        plan = self.plan(data, meta)
        data["pending_residual"] = True
        data["queued_actor"] = 1
        out = plan.canonical_successor(data, None)
        self.assertIsInstance(out, mapping.Refused)
        self.assertIn("LIVE_SUSPENSION_INTERFACE", {g.code for g in out.gaps})
        self.assertTrue(out.raw_jax)

    def test_source_changes_are_not_grandfathered_by_prior_review(self):
        with patch.dict(c.SOURCE_PINS, {"transition.py": "not-the-reviewed-source"}):
            out = self.plan()
        self.assertEqual(out.gaps[0].code, "CONSUMER_SOURCE_DRIFT")


if __name__ == "__main__":
    unittest.main()
