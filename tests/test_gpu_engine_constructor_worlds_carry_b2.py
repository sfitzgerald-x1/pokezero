"""Real phase_d_live_worlds.from_b2 roots and their sampled worlds through constructor_worlds.complete_world, both owners (CPU only).

Split from test_gpu_engine_constructor_worlds_carry (its fixtures and helpers stay there) so each module fits the light tier.
"""
import unittest
import numpy as np
from gpu_engine import constructor_worlds as cw, engine_device_negative_items as ni

from tests.test_gpu_engine_constructor_worlds_carry import (
    leaves)


class FromB2WorldTests(unittest.TestCase):
    """Real phase_d_live_worlds.from_b2 roots (tests/test_phase_d_device_worlds' authored B2 fold: each side's lead revealed) and their
    16 sampled worlds through complete_world, for both owners."""
    @classmethod
    def setUpClass(cls):
        import test_phase_d_device_worlds as D
        D.WorldsTests.setUpClass();cls.fx=D.WorldsTests('test_real_b2_fold_to_16_and32_source_worlds_both_owners')
        cls.tables=cw.carry_tables(cls.fx.metadata,cls.fx.catalog)

    def test_real_from_b2_worlds_complete(self):
        from gpu_engine import phase_d_live_worlds as live
        from pokezero.v5_contract import DecisionAddress
        f=self.fx;st=self.tables.species_types;normal=f.metadata['vocabulary']['types'].index('Normal')
        for owner in(0,1):
            root=live.from_b2(f.adapter(owner),f.request(owner),f.source,f.metadata,address=DecisionAddress(1,0,owner))
            made=live.sample_worlds(root,f.source,f.tables,count=16,seed=9);self.assertTrue(made.fully_constructed)
            done=[cw.complete_world(w,self.tables,owner)for w in made.worlds]   # no CarryLayoutError on the real carry
            for w,d in zip(made.worlds,done):
                with self.subTest(owner=owner,world=w.index):
                    t,c=w.public,d.public
                    self.assertFalse(np.asarray(t.shown_types).any())   # from_b2's zeros
                    species,active,shown=np.asarray(c.species)[0],np.asarray(c.active_slot)[0],np.asarray(c.shown_types)[0]
                    for side in(0,1):   # each side's revealed lead is filled with its species' types; every other slot stays 0
                        at=int(active[side]);self.assertGreater(int(species[side,at]),0)
                        np.testing.assert_array_equal(shown[side,at],st[species[side,at]]);np.testing.assert_array_equal(shown[side,at],[normal,0])
                        rest=np.arange(6)!=at;self.assertFalse(species[side,rest].any());self.assertFalse(shown[side,rest].any())
                    for key in cw.CREDIT_KEYS:
                        self.assertEqual((np.shape(t.field_features[key]),np.shape(c.field_features[key])),((1,),(1,2)))
                    removed=np.asarray(c.field_features['self_items_removed_credit'])
                    self.assertEqual(removed[0,owner],int(np.asarray(t.field_features['self_items_removed_credit'])[0]))
                    self.assertIsNone(t.negative_items);self.assertIs(type(c.negative_items),ni.State)
                    changed=('shown_types','negative_items','self_items_removed_credit','opp_items_removed_credit')
                    self.assertTrue(all(x is y for(_,x),(_,y)in zip(leaves(t,changed),leaves(c,changed))))
                    self.assertIs(d.state,w.state);self.assertIs(cw.complete_world(d,self.tables,owner),d)   # a second pass is a no-op
            # with the fold's own lines: the peer's lead is its request position 0 and the only switch-in, so the order stands; no Spikes
            lines=(f'|switch|p{owner+1}a: Alpha|Alpha, L80|291/291',f'|switch|p{2-owner}a: Beta|Beta, L80|100/100','|turn|1')
            facts=cw.line_facts(lines,owner);self.assertEqual(facts,cw.LineFacts(('beta',),()))
            for d in done:
                e=cw.complete_world(d,self.tables,owner,facts=facts);self.assertIs(e,d)
                order=np.asarray(d.public.own_order)[0,1-owner];self.assertEqual(int(order[0]),int(np.asarray(d.state.active)[0,1-owner]))
            # every world completes alike (the completion reads only the public species and the engine types)
            for name in('shown_types',):
                for d in done[1:]:np.testing.assert_array_equal(getattr(d.public,name),getattr(done[0].public,name))
            for d in done[1:]:
                for key in cw.CREDIT_KEYS:np.testing.assert_array_equal(d.public.field_features[key],done[0].public.field_features[key])
                for x,y in zip(d.public.negative_items,done[0].public.negative_items):np.testing.assert_array_equal(x,y)


    def test_b2s_fold_never_records_a_tracers_own_ability(self):
        """The real fold on Showdown's Gen 3 setAbility line form (the copied ability, then the tracer's own Trace as a positional
        argument): the tracer's public original stays 0 and its borrowed ability is the copy; the donor's original is the copy.
        fix_peer_base_ability's precondition (a public original) therefore never holds for a tracer."""
        from gpu_engine import phase_d_live_worlds as live
        from pokezero.v5_contract import DecisionAddress
        from pokezero.recorded_encoder_runner import JaxRecordedAdapter
        f=self.fx;pressure=f.metadata['vocabulary']['abilities'].index('pressure')
        for owner,tracer in((0,1),(1,1),(0,0)):
            with self.subTest(owner=owner,tracer=tracer):
                donor=1-tracer;seat=lambda side:f'p{side+1}a: '+('Alpha'if side==owner else'Beta')
                lines=(f'|switch|p{owner+1}a: Alpha|Alpha, L80|291/291',f'|switch|p{2-owner}a: Beta|Beta, L80|100/100',
                    f'|-ability|{seat(tracer)}|Pressure|Trace|[from] ability: Trace|[of] {seat(donor)}','|turn|1')
                a=JaxRecordedAdapter(f.catalog,owner_side=owner);a.reset(stream_id=1,battle_id='authored');a.feed(lines)
                root=live.from_b2(a,f.request(owner),f.source,f.metadata,address=DecisionAddress(1,0,owner))
                original,borrowed=np.asarray(root.public.original_ability)[0],np.asarray(root.public.borrowed_ability)[0]
                self.assertEqual((int(original[tracer,0]),int(borrowed[tracer,0]),int(original[donor,0])),(0,pressure,pressure))
                self.assertEqual(cw.line_facts(lines,owner).tracers[tracer],('alpha'if tracer==owner else'beta',))


if __name__ == '__main__':unittest.main()
