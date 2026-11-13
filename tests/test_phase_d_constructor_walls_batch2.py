"""Focused exactness controls for the second bounded constructor batch."""
from dataclasses import replace
from types import SimpleNamespace as NS
import random
import unittest
import numpy as np
from gpu_engine import phase_d_protocol_world as protocol, state, v5_hypothetical_state as mapping
from pokezero.dex import load_showdown_dex_cached
from _showdown_root import showdown_root
from test_gpu_engine_v5_public_world import TypedMappingTests


class LiveTypesTests(unittest.TestCase):
    def test_opt_in_preserves_original_base_and_default_refusal(self):
        spec, metadata = TypedMappingTests().fixture()
        dex = load_showdown_dex_cached(showdown_root())
        for base in (None, ('normal',)):
            changed = replace(spec, side_one=replace(spec.side_one,pokemon=(replace(spec.side_one.pokemon[0],types=('ice',),base_types=base),)))
            args = dict(metadata=metadata,dex=dex,seed=1,turn=1,active_turns=(0,0),happiness=((7,),(255,)))
            refused = mapping.map_spec(changed,**args)
            self.assertIsNone(refused.state)
            self.assertIn('changed-types-base-unavailable' if base is None else 'transformed-types:0:0',refused.refusals)
            admitted = mapping.map_spec(changed,**args,live_types=True)
            self.assertFalse(admitted.refusals)
            self.assertEqual(admitted.state.types[0,0,0].tolist(),[2,0])
            self.assertEqual(admitted.state.base_types[0,0,0].tolist(),[1,0])
            forged = replace(changed,side_one=replace(changed.side_one,pokemon=(replace(changed.side_one.pokemon[0],base_types=('ice',)),)))
            self.assertIn('base-types-differ-original-species',mapping.map_spec(forged,**args,live_types=True).refusals)


class PressureTests(unittest.TestCase):
    def carry(self,charge,pressure,lines=()):
        physical=state.empty_state(1)
        physical=physical._replace(moves=physical.moves.at[0,1,0,0].set(1),pp=physical.pp.at[0,1,0,0].set(24),max_hp=physical.max_hp.at[0,1,0].set(100))
        view={'present':np.array([[1,0,0,0,0,0]]),'species':np.array([[1,0,0,0,0,0]]),
              'hp_percent':np.array([[100,0,0,0,0,0]]),'revealed_moves':np.zeros((1,6,4),int),
              'move_pp_used':np.zeros((1,6,4),int),'mon_features':{'stall_counter':np.zeros((1,6))}}
        view['revealed_moves'][0,0,0]=1;view['move_pp_used'][0,0,0]=charge
        checks=[]
        result=protocol._carry_public_foe(physical,NS(view=view),{'vocabulary':{'species':['_none','luvdisc'],'moves':['_none','surf']}},
            NS(party_species={'p2':('luvdisc',)}),1,random.Random(1),checks,pressure=pressure,leppa=protocol._leppa_side(lines,1))
        return result,checks

    def test_exactly_one_extra_charge_requires_pressure_and_no_consumed_leppa(self):
        result,checks=self.carry(25,True)
        self.assertFalse(result[3]);self.assertEqual(int(result[0].pp[0,1,0,0]),0)
        self.assertTrue(any(x.get('pressure_floor') for x in checks))
        for charge,pressure,lines in ((26,True,()),(25,False,()),
            (25,True,('|-activate|p2a: Luvdisc|item: Leppa Berry|Surf|[consumed]',)),
            (25,True,('|-enditem|p2a: Luvdisc|Leppa Berry|[eat]',))):
            with self.subTest(charge=charge,pressure=pressure,lines=lines):
                result,_=self.carry(charge,pressure,lines)
                self.assertIn('public-opponent-PP-charge-exceeds-source-max',result[3])
        result,_=self.carry(25,True,('|-item|p2a: Luvdisc|Leppa Berry',))
        self.assertFalse(result[3])

    def test_mean_look_remains_refused(self):
        payload={'sides':{'p1':{'meanlookTrap':True,'volatiles':[]},'p2':{'volatiles':[]}}}
        checks=protocol.preflight_checks(payload)
        self.assertTrue(any(x['kind']=='native-probed-or-hidden-volatile-clock' and x['status']=='TRIGGERED' for x in checks))


if __name__ == '__main__': unittest.main()
