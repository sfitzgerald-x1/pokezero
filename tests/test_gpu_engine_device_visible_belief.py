"""Public source/snapshot algebra only; no engine turn, model or replay."""
from types import SimpleNamespace as NS
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_device_visible_belief as v, engine_device_trackers as tr
from gpu_engine import public_belief as b
from gpu_engine.state import empty_state
from pokezero.belief import PublicBattleBeliefEngine, belief_key
from pokezero.randbat import (Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata)


def fixture(batch=1):
    rows=tuple(Gen3RandbatVariant(str(i),'fixture','Swalot','Fixture',80,
        ('Tackle','Rest','Protect','move'+str(i)),
        'Early Bird' if i%2==0 else 'Sticky Hold','Leftovers' if i<2 else 'Choice Band') for i in range(4))
    source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'fixture'),
        {'swalot':Gen3RandbatSpeciesUniverse('Swalot',80,rows)})
    host=b.compile_source(source)
    def table(kind,names):
        masks=[host.device.pool]+[host.device.predicates[host.predicate_id(kind,n)] for n in names]
        return j.stack(masks,axis=1)
    tables=NS(species_initial_mask=host.device.pool,
        item_mask=table('item',('Leftovers','Choice Band')),
        ability_mask=table('ability',('Early Bird','Sticky Hold')))
    shape=(batch,2,6)
    s=empty_state(batch)._replace(species=j.ones(shape,j.int32),level=j.full(shape,80,j.int32),
        hp=j.full(shape,100,j.int32),max_hp=j.full(shape,100,j.int32),done=j.zeros(batch,bool))
    t=tr.initialize(s)._replace(belief_mask=j.broadcast_to(host.device.pool[1],shape+(2,)))
    p=s.active[:,1]
    t=v.refresh(t,tables,1,p,j.ones(batch,bool))
    return s,t,tables,source,host,rows


def summary(t,tables,owner=0):
    view=tr.readout(t,j.full(len(t.error),owner,j.int32))
    state=b.BeliefState(view.species,j.ones_like(view.present),view.belief_mask,
        view.belief_pin,view.belief_pin_active,j.ones_like(view.present),j.zeros_like(view.species))
    _,result=b.materialize(state,NS(pool=tables.species_initial_mask))
    return view,result


class VisibleBeliefTests(unittest.TestCase):
    def test_exact_python_standing_refusal_deferred_refresh_and_inconsistent_fallback(self):
        s,t,tables,source,host,rows=fixture()
        py=PublicBattleBeliefEngine(format_id='gen3randombattle',set_source=source)
        py._upsert(showdown_slot='p2',species='Swalot');key=belief_key('p2','Swalot')
        pin=lambda indices:j.asarray(host.pin_mask('Swalot',[rows[i].to_summary() for i in indices]))[None]
        for indices in ((),(0,1),(1,),(3,),(1,)):
            changed=py.narrow_candidate_variants(key,[rows[i].to_summary() for i in indices])
            t,actual=v.register_variant_survivors(t,tables,1,s.active[:,1],pin(indices),True)
            self.assertEqual(bool(actual[0]),changed)
            self.assertEqual(int(t.visible_belief.conflicts[0,1,0]),py.variant_pin_conflicts.get(key,0))
        # snapshot alone does not call _with_set_summary or apply a new pin.
        self.assertEqual(py.snapshot().side('p2')[0].candidate_set_count,4)
        self.assertEqual(int(summary(t,tables)[1].count[0,0]),4)
        py._upsert(showdown_slot='p2',species='Swalot')
        t=v.refresh(t,tables,1,s.active[:,1],True)
        self.assertEqual(int(summary(t,tables)[1].count[0,0]),1)
        self.assertEqual(py.snapshot().side('p2')[0].candidate_set_count,1)
        # Fresh hard evidence disagrees with the pin: retain it, flag conflict,
        # and keep the hard-evidence summary rather than zeroing either input.
        old=py.snapshot().side('p2')[0]
        py._replace_belief(old,revealed_moves=('move2',))
        hard=host.device.predicates[host.predicate_id('move','move2'),1]
        t=t._replace(belief_mask=t.belief_mask.at[0,1,0].set(hard))
        t=v.refresh(t,tables,1,s.active[:,1],True)
        view,got=summary(t,tables)
        self.assertEqual(int(got.reason[0,0]),0);self.assertTrue(bool(got.pin_conflict[0,0]))
        self.assertEqual(int(got.count[0,0]),py.snapshot().side('p2')[0].candidate_set_count)
        self.assertEqual(int(t.visible_belief.conflicts[0,1,0]),py.variant_pin_conflicts[key])
        before=t.visible_belief.conflicts
        summary(t,tables);summary(t,tables)
        np.testing.assert_array_equal(before,t.visible_belief.conflicts)
        # Off-source reveal gives the original fallback with reason2; the
        # retained standing pin must not manufacture fallback confidence.
        py._replace_belief(py.snapshot().side('p2')[0],revealed_moves=('offsource',))
        t=t._replace(belief_mask=t.belief_mask.at[0,1,0].set(0))
        t=v.refresh(t,tables,1,s.active[:,1],True)
        _,got=summary(t,tables)
        self.assertEqual(int(got.reason[0,0]),2);self.assertFalse(bool(got.pin_conflict[0,0]))
        self.assertEqual(int(got.count[0,0]),4)
        self.assertEqual(py.snapshot().side('p2')[0].candidate_set_count,4)

    def test_excluded_names_are_hard_facts_and_repeat_does_not_refresh_pending_pin(self):
        s,t,tables,source,host,rows=fixture()
        py=PublicBattleBeliefEngine(format_id='gen3randombattle',set_source=source)
        old=py._upsert(showdown_slot='p2',species='Swalot')
        py._rule_out_items(old,('leftovers',),'public non-proc fixture')
        t=v.exclude(t,tables,1,s.active[:,1],j.asarray([1]),True,kind='item')
        self.assertEqual(t.belief_mask[0,1,0].tolist(),[12,0])
        self.assertFalse(bool(t.visible_belief.pin_active.any()))
        pin=j.asarray(host.pin_mask('Swalot',[rows[2].to_summary()]))[None]
        py.narrow_candidate_variants(belief_key('p2','Swalot'),[rows[2].to_summary()])
        t,_=v.register_variant_survivors(t,tables,1,s.active[:,1],pin,True)
        py._rule_out_items(py.snapshot().side('p2')[0],('leftovers',),'same known non-proc')
        t=jax.jit(lambda t:v.exclude(t,tables,1,s.active[:,1],j.asarray([1]),True,kind='item'))(t)
        self.assertEqual(int(summary(t,tables)[1].count[0,0]),2)
        self.assertEqual(py.snapshot().side('p2')[0].candidate_set_count,2)
        self.assertFalse(bool(t.visible_belief.visible_pin_active[0,1,0]))
        py._replace_belief(py.snapshot().side('p2')[0],ruled_out_abilities=('Early Bird',))
        t=v.exclude(t,tables,1,s.active[:,1],j.asarray([1]),True,kind='ability')
        _,got=summary(t,tables)
        self.assertEqual(int(got.count[0,0]),1);self.assertTrue(bool(got.pin_conflict[0,0]))
        self.assertEqual(py.snapshot().side('p2')[0].candidate_set_count,1)

    def test_jit_mixed_current_subjects_empty_disjoint_unknown_and_clone_preserve(self):
        s,t,tables,_,_,_=fixture(4)
        # Actual per-row seat operands must not create a BxB gather.
        sides=j.asarray([0,1,0,1],j.int32);p=s.active[j.arange(4),sides]
        incoming=j.asarray([[1,0],[2,0],[0,0],[16,0]],j.uint32)
        @jax.jit
        def run(t):
            t,changed=v.register_variant_survivors(t,tables,sides,p,incoming,j.asarray([True,True,True,True]))
            t=v.refresh(t,tables,sides,p,j.asarray([True,False,True,True]))
            return t,changed
        out,changed=run(t)
        self.assertEqual(changed.tolist(),[True,True,False,False])
        self.assertEqual(out.visible_belief.visible_pin_active[j.arange(4),sides,0].tolist(),[True,False,False,False])
        self.assertEqual(out.visible_belief.pin.shape,(4,2,6,2))
        cloned=tr.clone_for_worlds(out,j.asarray([1,0,1],j.int32))
        np.testing.assert_array_equal(cloned.visible_belief.pin,out.visible_belief.pin[j.asarray([1,0,1])])
        # Malformed public subject is unavailable; no clipped other mon update.
        bad,_=v.register_variant_survivors(out,tables,1,j.full(4,9,j.int32),j.ones((4,2),j.uint32),True)
        for a,b_ in zip(jax.tree.leaves(out),jax.tree.leaves(bad),strict=True):np.testing.assert_array_equal(a,b_)

    def test_current_negative_rule_refreshes_visible_basis_without_item_or_pin_reveal(self):
        import test_gpu_engine_device_negative_items as n
        from gpu_engine import engine_device_public_hooks as h
        s,tables,leftovers,_=n.fixture(batch=2)
        c=h.initialize(s,tables);after=s._replace(hp=s.hp.at[:,1,0].set(160))
        h.negative_hp_line(c,after,1,True,source_class=n.n.HP_ACTION)
        h.turn_end(c,s,after,j.zeros(2,bool),upkeep=j.ones(2,bool),leftovers=leftovers)
        view,got=summary(c.trackers,tables)
        self.assertEqual(got.count[:,0].tolist(),[2,2]);self.assertEqual(got.reason[:,0].tolist(),[0,0])
        self.assertFalse(bool(view.item.any()));self.assertFalse(bool(view.belief_pin_active.any()))
        self.assertFalse(bool(c.trackers.negative_items.known.any()))

    def test_frozen_actual_mask_disables_optional_pin_transport_and_marks_contract_constant(self):
        from gpu_engine import engine_device_belief as device,engine_device_observation as encoder,v4_encoder
        from pokezero.recorded_mirror_roots import CHAMPION_CONTRACT
        import test_gpu_engine_device_observation as f
        f.DeviceObservationTests.setUpClass()
        fixture_=f.DeviceObservationTests();_,view,_,_=fixture_.facts()
        table=fixture_.tables.belief
        self.assertFalse(CHAMPION_CONTRACT['feature_masks']['investment_belief_narrowing'])
        self.assertFalse(bool(fixture_.catalog.masks.investment_belief_narrowing))
        self.assertFalse(bool(table.standing_pins_enabled))
        masks=dict(CHAMPION_CONTRACT['feature_masks'],investment_belief_narrowing=True)
        with self.assertRaisesRegex(ValueError,'producer is not installed'):v4_encoder.Masks.from_checkpoint(masks)
        hard=table.species_initial_mask[view.species]
        view=view._replace(belief_mask=hard)
        baseline=device.project(view,table)
        transported=view._replace(belief_pin=hard,belief_pin_active=j.ones_like(view.present))
        # A coherent optional carrier cannot override the actual false mask.
        result=jax.jit(lambda p:device.project(p,table))(transported)
        for name in ('effective','weights','count','reason','pin_conflict'):
            np.testing.assert_array_equal(getattr(result,name),getattr(baseline,name))
        self.assertFalse(bool(result.pin_conflict.any()))
        manifest=encoder.column_manifest(transported)
        cell=next(x for x in manifest if x['section']=='belief' and x['kind']=='numeric' and x['column']==131)
        self.assertEqual(cell['status'],'TRACKED');self.assertIn('constant0',cell['reason'])


if __name__=='__main__':unittest.main()
