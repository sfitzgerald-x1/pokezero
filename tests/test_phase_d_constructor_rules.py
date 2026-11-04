import unittest
from gpu_engine.phase_d_constructor_rules import public_hp_bin,request_ordered_packed,literal_hidden_power_agrees


class ConstructorRulesTests(unittest.TestCase):
    def test_bins_invert_actual_public_percent(self):
        for maximum in range(1,601):
            observed={}
            for hp in range(maximum+1):
                shown=100 if hp==maximum else min(99,(100*hp+maximum-1)//maximum)
                observed.setdefault(shown,[]).append(hp)
            for shown,values in observed.items():
                lo,hi=public_hp_bin(maximum,shown)
                self.assertEqual(list(range(lo,hi+1)),values)

    def test_99_contains_below_full_tail(self):
        self.assertEqual(public_hp_bin(350,99),(344,349))
        self.assertEqual(public_hp_bin(350,100),(350,350))

    def test_impossible_public_bin_refuses(self):
        with self.assertRaises(ValueError):public_hp_bin(1,50)

    def test_repack_keeps_literal_fields(self):
        names=tuple('abcdef');entries=[n+'||item|ability|move|nature|ev|F|iv||81|42'for n in names]
        self.assertEqual(request_ordered_packed(']'.join(entries),names,tuple('dfabce')),']'.join(entries[names.index(n)]for n in'dfabce'))
        with self.assertRaises(ValueError):request_ordered_packed(']'.join(entries),names,tuple('aaaaaa'))

    def test_typed_hidden_power_is_own_fact(self):
        self.assertTrue(literal_hidden_power_agrees(('hiddenpowerbug',),'hiddenpowerbug70'))
        self.assertFalse(literal_hidden_power_agrees(('hiddenpowerbug',),'hiddenpowersteel70'))
        self.assertTrue(literal_hidden_power_agrees((),None))
