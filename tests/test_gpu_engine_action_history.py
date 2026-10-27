"""Rust lastmove/PP/Sleep Talk regression cases adapted to JAX's state layout.

Authored correctness cases only; no scientific roots or coverage credit.
"""
import importlib.util
import unittest


def fixture_components():
    import jax
    import jax.numpy as j
    from gpu_engine import (baton, effects, encore, entry, phazing, pursuit,
                            sleep_talk, solar_beam, transition)
    names=['_none','splash','bodyslam','protect','struggle','sleeptalk',
           'encore','batonpass','roar','pursuit','solarbeam']
    types=['_none','normal','fighting','flying','poison','ground','rock','bug','ghost',
           'steel','fire','water','grass','electric','psychic','ice','dragon','dark']
    meta={'vocabulary':{'moves':names,'types':types,'abilities':['_none'],
        'items':['_none'],'species':['_none','ditto','pikachu','blissey','snorlax','skarmory','slaking']},
        'moves':[None]+[{'id':name,'flags':{'protect':1,'bypasssub':name=='encore',
            'failencore':name in {'encore','struggle'},'charge':name=='solarbeam',
            'nosleeptalk':name=='sleeptalk'}} for name in names[1:]]}
    n=len(names)
    table=transition.MoveTable(j.array([0,0,85,0,50,0,0,0,0,40,120]),
        j.ones(n,j.int32).at[9].set(17).at[10].set(12),
        j.ones(n,bool).at[9].set(False).at[10].set(False),j.full(n,100),
        j.ones(n,bool).at[2].set(False),j.zeros(n,j.int32).at[3].set(3).at[8].set(-6),
        j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(4),
        effects=effects.from_metadata(meta),encore=encore.from_metadata(meta),
        entry=entry.from_metadata(meta),baton=baton.from_metadata(meta),
        phazing=phazing.from_metadata(meta),pursuit=pursuit.from_metadata(meta),
        sleep_talk=sleep_talk.from_metadata(meta),solar_beam=solar_beam.from_metadata(meta))
    return jax,j,meta,table


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class ActionHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gpu_engine import action_history as h, mechanics as m, transition as t
        from gpu_engine.state import empty_state
        cls.jax,cls.j,cls.meta,cls.table=fixture_components()
        cls.h,cls.m,cls.t=h,m,t
        cls.empty=staticmethod(empty_state)
        cls.step=staticmethod(cls.jax.jit(t.step_with_chance))

    def live(self):
        j=self.j;s=self.empty(1,seed=112)
        return s._replace(hp=j.full_like(s.hp,1000),max_hp=j.full_like(s.hp,1000),
            stats=j.full_like(s.stats,200).at[...,0].set(1000).at[:,0,:,5].set(301),
            species=j.broadcast_to(j.arange(1,7),(1,2,6)),level=j.full_like(s.level,100),
            types=s.types.at[...,0].set(1),moves=j.ones_like(s.moves),
            pp=j.full_like(s.pp,10),done=j.array([False]))

    def chance(self):
        j=self.j
        draws=j.full((1,2,12),99,j.int32).at[:,:,0].set(0).at[:,:,4].set(15).at[:,:,5].set(15)
        draws=draws.at[:,:,7:].set(0)
        return self.t.ChanceInputs(draws,j.full((1,2),59),j.full((1,2),99),
            j.full((1,2),59),j.zeros((1,2),j.int32),j.zeros(1,j.int32))

    def play(self,state,actions=None,chance=None,table=None):
        return self.step(state,self.j.zeros((1,2),self.j.int32) if actions is None else actions,
            self.table if table is None else table,self.chance() if chance is None else chance).state

    def assert_history(self,state,side,kind,index):
        self.assertEqual((int(state.last_action_kind[0,side]),int(state.last_action_index[0,side])),(kind,index))

    def test_initial_none_is_not_an_entry_callback_or_active_age(self):
        s=self.live()._replace(entry_pending=self.j.ones((1,2),bool))
        initialized=self.t.initialize(s,self.table)
        for side in range(2):self.assert_history(initialized,side,self.h.NONE,-1)

    def test_duplicate_identity_records_actual_dispatch_slot_in_both_seats(self):
        s=self.live();r=self.play(s,self.j.array([[2,3]]))
        for side,slot in enumerate((2,3)):
            self.assert_history(r,side,self.h.MOVE,slot)
            self.assertEqual(r.pp[0,side,0].tolist(),[9 if k==slot else 10 for k in range(4)])

    def test_before_move_blockers_preserve_history_and_pp(self):
        for side in range(2):
            for status,counter,flinch,confusion,column,roll in (
                (self.m.PAR,0,0,0,2,0),(self.m.SLP,3,0,0,1,99),
                (self.m.FRZ,0,0,0,1,99),(0,0,1,0,1,99),(0,0,0,3,3,99)):
                s=self.live();s=self.h.record_switch(s,side,self.j.array([0]),eligible=self.j.array([True]))
                s=s._replace(status=s.status.at[0,side,0].set(status),
                    status_counter=s.status_counter.at[0,side,0].set(counter),
                    volatile=s.volatile.at[0,side,0].set(flinch),
                    confusion_counter=s.confusion_counter.at[0,side,0].set(confusion))
                c=self.chance();c=c._replace(draws=c.draws.at[0,side,column].set(roll))
                r=self.play(s,chance=c)
                self.assert_history(r,side,self.h.SWITCH,0)
                self.assertEqual(int(r.pp[0,side,0,0]),10)

    def test_miss_and_failed_encore_still_record_after_pp(self):
        for side in range(2):
            s=self.live();s=s._replace(moves=s.moves.at[0,side,0,2].set(2))
            table=self.table._replace(accuracy=self.table.accuracy.at[2].set(0))
            actions=self.j.zeros((1,2),self.j.int32).at[0,side].set(2)
            r=self.play(s,actions,table=table)
            self.assert_history(r,side,self.h.MOVE,2)
            self.assertEqual(int(r.pp[0,side,0,2]),9)
            self.assertEqual(int(r.hp[0,1-side,0]),1000)
            # Faster Encore targets an initial NONE and fails, but counts as used.
            s=s._replace(moves=s.moves.at[0,side,0,2].set(6),
                stats=s.stats.at[0,side,0,5].set(500))
            r=self.play(s,actions)
            self.assert_history(r,side,self.h.MOVE,2)
            self.assertEqual(int(r.encore_counter[0,1-side,0]),0)

    def test_wake_thaw_and_confusion_snapout_record_but_truant_does_not(self):
        for side in range(2):
            for status,counter,confusion in ((self.m.SLP,1,0),(self.m.FRZ,0,0),(0,0,1)):
                s=self.live();s=s._replace(status=s.status.at[0,side,0].set(status),
                    status_counter=s.status_counter.at[0,side,0].set(counter),
                    confusion_counter=s.confusion_counter.at[0,side,0].set(confusion))
                c=self.chance();c=c._replace(draws=c.draws.at[0,side,1].set(0))
                r=self.play(s,chance=c)
                self.assert_history(r,side,self.h.MOVE,0)
                self.assertEqual(int(r.pp[0,side,0,0]),9)
            s=self.live();s=s._replace(truant_loaf=s.truant_loaf.at[0,side,0].set(True))
            table=self.table._replace(entry=self.table.entry._replace(truant=self.j.array([True])))
            r=self.play(s,table=table)
            self.assert_history(r,side,self.h.NONE,-1)
            self.assertEqual(int(r.pp[0,side,0,0]),10)

    def test_sleep_talk_records_caller_once_and_struggle_survives_self_faint(self):
        for side in range(2):
            s=self.live();s=s._replace(moves=s.moves.at[0,side,0,2].set(5),
                status=s.status.at[0,side,0].set(self.m.SLP),
                status_counter=s.status_counter.at[0,side,0].set(3))
            r=self.play(s,self.j.zeros((1,2),self.j.int32).at[0,side].set(2))
            self.assert_history(r,side,self.h.MOVE,2)
            self.assertEqual(int(r.last_move[0,side,0]),5)
            self.assertEqual(r.pp[0,side,0].tolist(),[10,10,9,10])
            s=self.live();s=s._replace(pp=s.pp.at[0,side,0].set(0),hp=s.hp.at[0,side,0].set(1))
            r=self.play(s)
            self.assert_history(r,side,self.h.STRUGGLE,-1)
            self.assertEqual(int(r.hp[0,side,0]),0)
            self.assertEqual(int(r.last_move[0,side,0]),0)  # Per-mon cleanup is distinct.

    def test_committed_switch_and_hazard_ko_retain_target_index(self):
        for side in range(2):
            for hazard in (False,True):
                s=self.live();s=s._replace(spikes=s.spikes.at[0,side].set(3 if hazard else 0),
                    hp=s.hp.at[0,side,2].set(250))
                r=self.play(s,self.j.zeros((1,2),self.j.int32).at[0,side].set(5))
                self.assert_history(r,side,self.h.SWITCH,2)
                self.assertEqual(int(r.active[0,side]),2)
                self.assertEqual(int(r.hp[0,side,2]),0 if hazard else 250)

    def test_baton_pause_then_replacement_and_phaze_record_real_switch(self):
        for side in range(2):
            s=self.live();s=s._replace(moves=s.moves.at[0,side,0,1].set(7),
                stats=s.stats.at[0,side,0,5].set(500))
            paused=self.play(s,self.j.zeros((1,2),self.j.int32).at[0,side].set(1))
            self.assertTrue(bool(paused.pending_residual[0]))
            self.assert_history(paused,side,self.h.MOVE,1)
            resumed=self.play(paused,self.j.zeros((1,2),self.j.int32).at[0,side].set(4))
            self.assert_history(resumed,side,self.h.SWITCH,1)
            s=self.live();s=s._replace(moves=s.moves.at[0,1-side,0,2].set(8))
            r=self.play(s,self.j.zeros((1,2),self.j.int32).at[0,1-side].set(2))
            self.assert_history(r,side,self.h.SWITCH,int(r.active[0,side]))

    def test_pursuit_interception_and_locked_release_record_without_double_pp(self):
        for side in range(2):
            s=self.live();s=s._replace(moves=s.moves.at[0,side,0,3].set(9),
                volatile=s.volatile.at[0,side,0].set(1))
            actions=self.j.full((1,2),4,self.j.int32).at[0,side].set(3)
            r=self.play(s,actions)
            self.assert_history(r,side,self.h.MOVE,3)
            self.assert_history(r,1-side,self.h.SWITCH,1)
            self.assertEqual(int(r.pp[0,side,0,3]),9)
            s=self.live();s=s._replace(moves=s.moves.at[0,side,0,3].set(10))
            actions=self.j.zeros((1,2),self.j.int32).at[0,side].set(3)
            charged=self.play(s,actions);released=self.play(charged,actions)
            self.assert_history(released,side,self.h.MOVE,3)
            self.assertEqual(int(released.pp[0,side,0,3]),9)

    def test_duplicate_encore_uses_exact_pp_slot_for_override_mask_and_expiry(self):
        from gpu_engine import encore
        for side in range(2):
            s=self.live();s=self.h.record_move(s,side,self.j.array([2]),eligible=self.j.array([True]))
            s=s._replace(last_move=s.last_move.at[0,side,0].set(1),
                pp=s.pp.at[0,side,0,0].set(0).at[0,side,0,2].set(1))
            encored=encore.apply(s,self.j.array([side]),self.j.array([6]),self.j.array([0]),
                self.table.encore,eligible=self.j.array([True]),queued=self.j.array([True]))
            self.assertEqual(int(encored.encore_move[0,side,0]),1)
            self.assertEqual(self.t.action_mask(encored,self.table)[0,side,:4].tolist(),[False,False,True,False])
            move,slot=encore.override(encored,self.j.array([side]),self.j.array([2]),self.j.array([0]),struggle=False)
            self.assertEqual((int(move[0]),int(slot[0])),(1,2))
            r=self.play(encored,self.j.zeros((1,2),self.j.int32).at[0,side].set(2))
            self.assertEqual(int(r.pp[0,side,0,2]),0)
            self.assertEqual(int(r.encore_counter[0,side,0]),0)
            self.assert_history(r,side,self.h.MOVE,2)

    def test_invalid_and_terminal_requests_preserve_all_history_with_batching(self):
        s=self.live();s=self.h.record_switch(s,0,self.j.array([0]),eligible=self.j.array([True]))
        batch=self.jax.tree.map(lambda x:self.j.concatenate((x,x),axis=0),s)
        batch=batch._replace(done=batch.done.at[1].set(True))
        chance=self.jax.tree.map(lambda x:self.j.concatenate((x,x),axis=0),self.chance())
        r=self.step(batch,self.j.array([[99,0],[0,0]]),self.table,chance).state
        for before,after in zip(self.jax.tree.leaves(batch),self.jax.tree.leaves(r)):
            self.assertTrue(bool(self.j.array_equal(before,after)))
        s=s._replace(pp=s.pp.at[0,0,0,0].set(0))  # Other slots usable: no Struggle.
        r=self.play(s)
        for before,after in zip(self.jax.tree.leaves(s),self.jax.tree.leaves(r)):
            self.assertTrue(bool(self.j.array_equal(before,after)))

    def test_stale_diagnostic_identity_cannot_change_encore_future(self):
        for side in range(2):
            s=self.live();s=self.h.record_move(s,side,self.j.array([2]),eligible=self.j.array([True]))
            s=s._replace(moves=s.moves.at[0,1-side,0,0].set(6),
                stats=s.stats.at[0,1-side,0,5].set(500))
            results=[]
            for old_identity in (0,1,2,5):
                changed=s._replace(last_move=s.last_move.at[0,side,0].set(old_identity))
                r=self.play(changed);results.append(r)
                self.assertEqual(int(r.encore_move[0,side,0]),1)
                self.assertEqual(int(r.pp[0,side,0,2]),9)
            for r in results[1:]:
                for a,b in zip(self.jax.tree.leaves(results[0]),self.jax.tree.leaves(r)):
                    self.assertTrue(bool(self.j.array_equal(a,b)))


if __name__=='__main__':unittest.main()
