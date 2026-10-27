"""Rust Rest/refund cases adapted to JAX; authored controls, no panel credit."""
import importlib.util
import unittest


def fixture_components():
    import jax
    import jax.numpy as j
    from gpu_engine import effects,entry,items,sleep_talk,transition
    names=['_none','rest','sleeptalk','splash','solarbeam','struggle','healbell','spore']
    types=['_none','normal','fighting','flying','poison','ground','rock','bug','ghost',
           'steel','fire','water','grass','electric','psychic','ice','dragon','dark']
    abilities=['_none','naturalcure','earlybird','insomnia','vitalspirit','soundproof','trace']
    meta={'vocabulary':{'moves':names,'types':types,'abilities':abilities,
        'items':['_none','lumberry'],'species':['_none','ditto','pikachu','blissey','snorlax','skarmory','slaking']},
        'moves':[None]+[{'id':name,'source_type':'Normal','category':'Status',
            'flags':{'charge':int(name=='solarbeam')}} for name in names[1:]]}
    n=len(names)
    table=transition.MoveTable(j.zeros(n,j.int32),j.ones(n,j.int32),j.zeros(n,bool),j.full(n,100),
        j.ones(n,bool),j.zeros(n,j.int32),j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(5),
        effects=effects.from_metadata(meta),entry=entry.from_metadata(meta),items=items.from_metadata(meta),
        sleep_talk=sleep_talk.from_metadata(meta))
    return jax,j,meta,table


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class RestOriginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gpu_engine import transition as t
        from gpu_engine.state import empty_state
        cls.jax,cls.j,cls.meta,cls.table=fixture_components()
        cls.t=t;cls.empty=staticmethod(empty_state);cls.step=staticmethod(cls.jax.jit(t.step_with_chance))

    def live(self,batch=1):
        j=self.j;s=self.empty(batch,seed=819)
        return s._replace(hp=j.full_like(s.hp,50),max_hp=j.full_like(s.max_hp,101),
            stats=j.full_like(s.stats,200).at[...,0].set(101).at[:,0,:,5].set(301),
            level=j.full_like(s.level,100),types=s.types.at[...,0].set(1),
            species=j.broadcast_to(j.arange(1,7),(batch,2,6)),
            moves=j.broadcast_to(j.array([1,2,3,0]),s.moves.shape),pp=j.full_like(s.pp,10).at[...,3].set(0),
            done=j.zeros(batch,bool))

    def actions(self,side,action,batch=1):
        return self.j.full((batch,2),2,self.j.int32).at[:,side].set(action)

    def chance(self,batch=1):
        j=self.j;draws=j.zeros((batch,2,12),j.int32).at[:,:,1:4].set(99).at[:,:,4:7].set(15)
        return self.t.ChanceInputs(draws,j.zeros((batch,2),j.int32),j.full((batch,2),99),
            j.zeros((batch,2),j.int32),j.zeros((batch,2),j.int32),j.zeros(batch,j.int32))

    def play(self,s,side,action):
        return self.step(s,self.actions(side,action,s.hp.shape[0]),self.table,self.chance(s.hp.shape[0])).state

    def assert_sleep(self,s,side,status,counter,bank,origin):
        self.assertEqual((int(s.status[0,side,0]),int(s.status_counter[0,side,0]),
            int(s.sleep_skipped[0,side,0]),bool(s.rest_origin[0,side,0])),(status,counter,bank,origin))

    def sleeper(self,side,*,counter=3,bank=0,origin=True):
        s=self.live();return s._replace(status=s.status.at[0,side,0].set(5),
            status_counter=s.status_counter.at[0,side,0].set(counter),
            sleep_skipped=s.sleep_skipped.at[0,side,0].set(bank),
            rest_origin=s.rest_origin.at[0,side,0].set(origin))

    def test_only_successful_rest_creates_origin_and_wake_recast_resets_bank(self):
        for side in range(2):
            s=self.live();self.assertFalse(bool(s.rest_origin.any()))
            out=self.play(s,side,0);self.assert_sleep(out,side,5,3,0,True)
            self.assertEqual((int(out.hp[0,side,0]),int(out.pp[0,side,0,0])),(101,9))
            full=s._replace(hp=s.max_hp)
            out=self.play(full,side,0);self.assert_sleep(out,side,0,0,0,False)
            self.assertEqual(int(out.pp[0,side,0,0]),9)
            # A non-sleep-usable attempt while still asleep discards the bank.
            out=self.play(self.sleeper(side,counter=3,bank=1),side,0)
            self.assert_sleep(out,side,5,2,0,True);self.assertEqual(int(out.pp[0,side,0,0]),10)
            # Wake then Rest in the same dispatch starts an independent stint.
            out=self.play(self.sleeper(side,counter=1,bank=2),side,0)
            self.assert_sleep(out,side,5,3,0,True);self.assertEqual(int(out.pp[0,side,0,0]),9)

    def test_sleep_talk_bank_discard_and_switch_back_both_seats(self):
        for side in range(2):
            for ordinary in (False,True):
                s=self.play(self.live(),side,0)
                s=self.play(s,side,1);self.assert_sleep(s,side,5,2,1,True)
                s=self.play(s,side,2 if ordinary else 1)
                self.assert_sleep(s,side,5,1,0 if ordinary else 2,True)
                s=self.play(s,side,4);self.assert_sleep(s,side,5,1,0 if ordinary else 2,True)
                s=self.play(s,side,4);self.assert_sleep(s,side,5,1 if ordinary else 3,0,True)

    def test_early_bird_and_later_flinch_still_record_the_sleep_phase(self):
        for side in range(2):
            s=self.sleeper(side);s=s._replace(ability=s.ability.at[0,side,0].set(2),
                base_ability=s.base_ability.at[0,side,0].set(2),volatile=s.volatile.at[0,side,0].set(1))
            out=self.play(s,side,1);self.assert_sleep(out,side,5,1,1,True)
            self.assertEqual(int(out.pp[0,side,0,1]),10)
            out=self.play(out,side,2);self.assert_sleep(out,side,0,0,0,False)

    def test_cures_clear_origin_and_primary_sleep_never_inherits_it(self):
        from gpu_engine import entry,effects
        for side in range(2):
            # Natural Cure observes the current, possibly copied ability.
            for current,base,cured in ((1,6,True),(0,1,False)):
                s=self.sleeper(side,counter=2,bank=1)
                s=s._replace(ability=s.ability.at[0,side,0].set(current),
                    base_ability=s.base_ability.at[0,side,0].set(base))
                out=self.play(s,side,4);self.assert_sleep(out,side,0 if cured else 5,0 if cured else 2,0 if cured else 1,not cured)
                self.assertEqual(int(out.ability[0,side,0]),base)
            s=self.sleeper(side,counter=2,bank=1);s=s._replace(item=s.item.at[0,side,0].set(1))
            out=self.play(s,side,2);self.assert_sleep(out,side,0,0,0,False)
            for immune in (3,4):
                s=self.sleeper(side,counter=2,bank=1);s=s._replace(ability=s.ability.at[0,side,0].set(immune))
                out,_=entry.enter(s,side,self.table.entry,eligible=self.j.array([True]))
                self.assert_sleep(out,side,0,0,0,False)
            for soundproof in (False,True):
                s=self.sleeper(side,counter=2,bank=1)
                s=s._replace(active=s.active.at[0,side].set(1),ability=s.ability.at[0,side,0].set(5 if soundproof else 0))
                out=effects.cure_team(s,side,self.table.effects,eligible=self.j.array([True]))
                self.assert_sleep(out,side,5 if soundproof else 0,2 if soundproof else 0,1 if soundproof else 0,soundproof)
            s=self.live();s=s._replace(moves=s.moves.at[0,1-side,0,2].set(7))
            out=self.play(s,side,2);self.assert_sleep(out,side,5,2 if side==0 else 1,0,False)

    def test_refund_caps_only_rest_and_ineligible_rows_remain_unchanged(self):
        from gpu_engine import entry,sleep_talk
        for side in range(2):
            for origin,expected in ((True,3),(False,5)):
                s=self.sleeper(side,counter=2,bank=3,origin=origin)
                out=entry.restore_sleep(s,side,eligible=self.j.array([True]))
                self.assert_sleep(out,side,5,expected,0,origin)
                out=sleep_talk.before_move(s,side,self.j.array([5]),eligible=self.j.array([True]),sleep_usable=self.j.array([True]))
                self.assertEqual(int(out.sleep_skipped[0,side,0]),3 if origin else 4)
            s=self.sleeper(side,counter=1,bank=2)
            out=sleep_talk.before_move(s,side,self.j.array([0]),eligible=self.j.array([False]),sleep_usable=self.j.array([True]))
            for a,b in zip(s,out):self.assertTrue(bool(self.j.array_equal(a,b)))

    def test_rest_fields_are_preserved_on_hazard_ko_and_batch_prefix(self):
        for side in range(2):
            s=self.sleeper(side,counter=2,bank=1)
            s=self.play(s,side,4);s=s._replace(hp=s.hp.at[0,side,0].set(1),spikes=s.spikes.at[0,side].set(1))
            out=self.play(s,side,4);self.assert_sleep(out,side,5,3,0,True)
            self.assertEqual(int(out.hp[0,side,0]),0);self.assertTrue(bool(out.forced_switch[0,side]))
        s=self.live(3);s=s._replace(done=self.j.array([False,False,True]))
        full=self.play(s,0,0);prefix=self.play(self.jax.tree.map(lambda x:x[:2],s),0,0)
        for a,b in zip(full,prefix):self.assertTrue(bool(self.j.array_equal(a[:2],b)))
        for a,b in zip(s,full):self.assertTrue(bool(self.j.array_equal(a[2:],b[2:])))

    def test_pinned_source_rest_refund_cycles_both_seats(self):
        import json,os,shutil,subprocess
        from pathlib import Path
        root=os.environ.get('POKEZERO_SHOWDOWN_ROOT');node=os.environ.get('NODE') or shutil.which('node')
        if not root or not node:self.skipTest('requires pinned source simulator')
        rows=json.loads(subprocess.check_output([node,str(Path(__file__).with_name('gpu_rest_origin_oracle.cjs')),root],text=True))
        self.assertEqual(len(rows),8)
        for row in rows:
            self.assertTrue(row['calledPools'])
            for pool in row['calledPools']:self.assertEqual(sorted(pool),['rest','splash'])
            side,variant=row['side'],row['variant'];s=self.live()
            ability={'early':2,'naturalcure':1}.get(variant,0)
            s=s._replace(ability=s.ability.at[0,side,0].set(ability),base_ability=s.base_ability.at[0,side,0].set(ability))
            actions=[0,1]+([2 if variant=='discard' else 1] if variant in ('consecutive','discard') else [])+[4,4,2]
            self.assertEqual(len(actions),len(row['snapshots']))
            for action,want in zip(actions,row['snapshots']):
                s=self.play(s,side,action)
                self.assertEqual((int(s.status[0,side,0]),int(s.status_counter[0,side,0]),int(s.sleep_skipped[0,side,0]),int(s.hp[0,side,0])),
                    (5 if want['status']=='slp' else 0,want['time'],want['skipped'],want['hp']),(side,variant,action))
                self.assertEqual(bool(s.rest_origin[0,side,0]),want['status']=='slp')
