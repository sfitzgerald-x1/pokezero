"""Explicit source-visible summary refreshes and deferred standing pins."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PublicBeliefSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_public_hazard_exposure as fixtures
        fixtures.PublicHazardExposureTests.setUpClass.__func__(cls)
        from gpu_engine import public_belief_snapshot as snapshot
        cls.snapshot=snapshot

    def initial(self,batch=1):
        empty=self.b.initialize(self.j.zeros((batch,2,6),self.j.int32),self.host.device)
        return empty,self.snapshot.initialize(empty,self.host.device)

    def populated(self,batch=1):
        empty,cache=self.initial(batch)
        ids=self.j.zeros_like(empty.species).at[:,0,0].set(self.host.species_id('Snorlax'))
        state=self.b.initialize(ids,self.host.device)
        mask=ids>0;cache,state,ok=self.snapshot.refresh(cache,state,self.host.device,source_refresh=mask)
        self.assertTrue(bool(ok.all()));return state,cache,mask

    def test_pin_registration_does_not_change_visible_count_until_explicit_refresh(self):
        state,cache,mask=self.populated();before,ok=self.snapshot.read(cache,state,self.host.device)
        self.assertTrue(bool(ok.all()));self.assertEqual(int(before.count[0,0,0]),12)
        variants=self.host.variants[self.host.species_id('Snorlax')]
        pin=self.host.pin_mask('Snorlax',[v.to_summary() for v in variants[:3]])
        state,changed=self.b.narrow_pin(state,pin,valid=mask);self.assertTrue(bool(changed.any()))
        pending,ok=self.snapshot.read(cache,state,self.host.device)
        self.assertTrue(bool(ok.all()));self.assertEqual(int(pending.count[0,0,0]),12)
        fn=self.jax.jit(lambda c,s:self.snapshot.refresh(c,s,self.host.device,source_refresh=mask))
        refreshed,state,ok=fn(cache,state);self.assertTrue(bool(ok.all()))
        visible,ok=self.snapshot.read(refreshed,state,self.host.device)
        self.assertTrue(bool(ok.all()));self.assertEqual(int(visible.count[0,0,0]),3)
        self.assertEqual(int(refreshed.generation[0,0,0]),2)
        self.assertTrue(bool(self.j.array_equal(cache.hard,refreshed.hard)))

    def test_basis_evidence_changes_require_refresh_even_if_hard_bits_match(self):
        state,cache,mask=self.populated()
        changed=state._replace(has_evidence=state.has_evidence|mask)
        _,ok=self.snapshot.read(cache,changed,self.host.device);self.assertFalse(bool(ok[0,0,0]))
        cache,changed,accepted=self.snapshot.refresh(cache,changed,self.host.device,source_refresh=mask)
        self.assertTrue(bool(accepted.all()));_,ok=self.snapshot.read(cache,changed,self.host.device);self.assertTrue(bool(ok.all()))
        with self.assertRaises(ValueError):self.snapshot.initialize(changed,self.host.device)
        with self.assertRaises(ValueError):self.snapshot.refresh(cache,changed,self.host.device,source_refresh=True)

    def test_conflict_counts_only_advance_on_refreshed_rows_and_namespaces_refuse(self):
        state,cache,mask=self.populated(2)
        # The actual source predicate singles out the first variant's move31.
        state,_=self.b.add_evidence(state,self.host.device,self.host.predicate_id('move','move31'),valid=mask)
        variants=self.host.variants[self.host.species_id('Snorlax')]
        pin=self.host.pin_mask('Snorlax',[variants[1].to_summary()])
        state,_=self.b.narrow_pin(state,pin,valid=mask)
        refresh=self.j.zeros_like(mask).at[1,0,0].set(True)
        fn=self.jax.jit(lambda c,s:self.snapshot.refresh(c,s,self.host.device,source_refresh=refresh))
        out,updated,ok=fn(cache,state);self.assertTrue(bool(ok.all()))
        self.assertEqual(updated.conflicts[:,0,0].tolist(),[0,1])
        self.assertEqual(out.generation[:,0,0].tolist(),[1,2])
        _,visible_ok=self.snapshot.read(out,updated,self.host.device);self.assertEqual(visible_ok[:,0,0].tolist(),[False,True])
        bad=cache._replace(source_binding=cache.source_binding.at[0].add(1))
        refused,unchanged,ok=fn(bad,state);self.assertFalse(bool(ok[1,0,0]))
        for a,b in zip(self.jax.tree.leaves((bad,state)),self.jax.tree.leaves((refused,unchanged))):self.assertTrue(bool(self.j.array_equal(a,b)))
        for counter in (-1,2**31-1):
            bad_state=state._replace(conflicts=state.conflicts.at[1,0,0].set(counter))
            refused,unchanged,ok=fn(cache,bad_state);self.assertFalse(bool(ok[1,0,0]))
            for a,b in zip(self.jax.tree.leaves((cache,bad_state)),self.jax.tree.leaves((refused,unchanged))):self.assertTrue(bool(self.j.array_equal(a,b)))


if __name__=='__main__':unittest.main()
