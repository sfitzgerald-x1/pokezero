"""Lexical tendency transport round trips and canonical payload refusal."""
import importlib.util
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PublicTendencyProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_gpu_engine_public_tendency import PublicTendencyTests
        PublicTendencyTests.setUpClass.__func__(cls)
        from gpu_engine import public_tendency_protocol as codec
        cls.codec=codec

    def test_actual_raw_lexical_facts_round_trip_bit_exact(self):
        lines=[*self.prefix,'|move|p1a: Snorlax|Body Slam|p2a: Absol',
            '|move|p1a: Snorlax|Body Slam|p2a: Absol|[from] move: Sleep Talk',
            '|move|p1a: Snorlax|Body Slam|p2a: Absol|[from]lockedmove',
            '|cant|p1a: Snorlax|recharge','|-prepare|p1a: Snorlax|Solar Beam',
            '|switch|p2a: Gardevoir|Gardevoir, L80|100/100|[from] move: Baton Pass',
            '|drag|p2a: Absol|Absol, L80|100/100','|replace|p2a: Gardevoir|Gardevoir, L80|100/100',
            '|faint|p1a: Snorlax','|-weather|RainDance|[from] ability: Drizzle|[of] p2a: Gardevoir',
            '|-activate|p2a: Gardevoir|move: Pursuit','|-activate|p1a: Snorlax|Protect',
            '|-immune|p2a: Gardevoir|[from] ability: Water Absorb','|-start|p2a: Gardevoir|ability: Flash Fire',
            '|','|upkeep','|win|p2','|t:|123']
        for line in lines:
            with self.subTest(line=line):
                p=self.codec.project_line(line,catalog=self.catalog)
                self.assertTrue(self.codec.valid_words(p.payload,p.side))
                got,ok=self.jax.jit(self.codec.unpack)(self.j.array([p.payload],self.j.int32),self.j.array([p.side],self.j.int32))
                self.assertTrue(bool(ok.all()));self.assertFalse(p.full_line_consumers_complete)
                expected=self.t.decode_line(line,catalog=self.catalog)
                for a,b in zip(got,expected):self.assertEqual(a.tolist(),[b.item()])

    def test_reserved_bits_and_cross_kind_words_refuse_without_changing_good_row(self):
        p=self.codec.project_line('|turn|1',catalog=self.catalog)
        for col,value in ((0,99),(1,1),(2,1),(3,0),(4,0),(5,32),(5,1),(6,1),(7,2)):
            payload=self.j.array([p.payload,p.payload],self.j.int32).at[0,col].set(value)
            facts,ok=self.jax.jit(self.codec.unpack)(payload,self.j.array([p.side,p.side],self.j.int32))
            self.assertEqual(ok.tolist(),[False,True],(col,value))
            import numpy as np
            self.assertFalse(self.codec.valid_words(np.asarray(payload[0]),p.side))
            state=self.t.initialize(2,self.tables,species_binding=self.tables.species_binding,complete_prefix=True)
            result,accepted=self.t.apply(state,self.tables,facts,producer_binding=self.tables.binding,valid=ok)
            self.assertEqual(accepted.tolist(),[True,True]);self.assertEqual(result.turn.tolist(),[0,1])
        for line,col,value,side in (
            ('|-immune|p2a: Absol',None,None,0),
            ('|-immune|p2a: Absol',7,9,1),
            ('|move|p1a: Snorlax|Surf|p2a: Absol',3,-1,0),
            ('|move|p1a: Snorlax|Surf|p2a: Absol',5,3,0),
            ('|cant|p1a: Snorlax|par',None,None,-1),
            ('|faint|p1a: Snorlax',None,None,-1)):
            p=self.codec.project_line(line,catalog=self.catalog);payload=self.j.array([p.payload],self.j.int32)
            if col is not None:payload=payload.at[0,col].set(value)
            _,ok=self.jax.jit(self.codec.unpack)(payload,self.j.array([side],self.j.int32))
            self.assertEqual(ok.tolist(),[False],line)
            self.assertFalse(self.codec.valid_words(tuple(payload[0].tolist()),side))

    def test_toxic_and_tendency_host_words_require_no_jax_scalar_conversion(self):
        from gpu_engine import public_toxic as toxic, public_toxic_protocol as toxic_codec

        with patch.object(self.j, 'asarray', side_effect=AssertionError('host projection allocated a JAX array')):
            tendency = self.codec.project_line('|turn|1', catalog=self.catalog)
            poison = toxic_codec.project_line('|turn|1', identity_ids={})
            self.assertEqual((tendency.side, tendency.payload), (-1, (self.t.TURN, 0, 0, -1, 1, 0, 0, 9)))
            self.assertEqual((poison.side, poison.payload), (-1, (toxic.TURN, 0, 1, -1, 0, 33, 0, 0)))
            self.assertTrue(poison.shared_event)
            self.assertFalse(poison.timestamp)
            with self.assertRaisesRegex(ValueError, 'unsigned integer fraction'):
                toxic_codec.project_line('|-damage|p1a: Snorlax|bad', identity_ids={})
            with self.assertRaisesRegex(ValueError, 'cannot share one public identity word'):
                toxic_codec.project_line('|', identity_ids={'p1a: Snorlax': 1, 'p1a: Alias': 1})
            with self.assertRaisesRegex(ValueError, 'canonical catalog species spelling'):
                self.codec.project_line('|switch|p2a: absol|absol, L80|100/100', catalog=self.catalog)
            with self.assertRaisesRegex(ValueError, 'int32 turn'):
                self.codec.project_line('|turn|2147483648', catalog=self.catalog)


if __name__=='__main__':unittest.main()
