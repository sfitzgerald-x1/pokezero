"""Authored Gen3 Trace entry fixtures, not collection or coverage evidence."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

NODE=os.environ.get('NODE') or shutil.which('node')
SHOWDOWN=os.environ.get('POKEZERO_SHOWDOWN_ROOT')


@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
class TraceEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        from gpu_engine import entry,transition,weather,secondary
        from gpu_engine.state import empty_state
        cls.jax,cls.j,cls.e,cls.t=jax,j,entry,transition
        cls.empty=staticmethod(empty_state)
        cls.abilities=['_none','trace','static','intimidate','drizzle','drought','sandstream',
            'truant','levitate','waterveil','magmaarmor','limber','immunity','insomnia',
            'vitalspirit','owntempo','naturalcure','hustle','sandveil']
        types=['_none','normal','fighting','flying','poison','ground','rock','bug','ghost',
            'steel','fire','water','grass','electric','psychic','ice','dragon','dark']
        cls.meta={'vocabulary':{'abilities':cls.abilities,'types':types,
            'moves':['_none','splash','struggle','bodyslam'],'items':['_none'],
            'species':['_none','ditto','porygon2']},
            'moves':[None,{'id':'splash','flags':{}},{'id':'struggle','flags':{'contact':1}},
                {'id':'bodyslam','flags':{'contact':1},'secondary':{'chance':30,'status':'par'}}]}
        cls.et=entry.from_metadata(cls.meta)
        cls.table=transition.MoveTable(j.array([0,0,50,85]),j.array([0,1,0,1]),
            j.array([False,False,True,True]),j.array([100,100,100,100]),
            j.array([True,True,True,False]),j.zeros(4,j.int32),
            j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(2),
            entry=cls.et,weather=weather.from_metadata(cls.meta),secondary=secondary.from_metadata(cls.meta))

    def fixture(self,side=0,foe='static',status=0,batch=1):
        j=self.j;s=self.empty(batch,seed=79)
        ability=s.ability.at[:,side,1].set(1).at[:,1-side,0].set(self.abilities.index(foe))
        return s._replace(species=j.ones_like(s.species).at[:,side,1].set(2),
            level=j.full_like(s.level,100),stats=j.full_like(s.stats,200).at[...,0].set(1000)
                .at[:,side,:,5].set(300).at[:,1-side,:,5].set(101),
            types=s.types.at[...,0].set(1),hp=j.full_like(s.hp,1000),max_hp=j.full_like(s.hp,1000),
            moves=j.ones_like(s.moves),pp=j.full_like(s.pp,10),ability=ability,base_ability=ability,
            gender=j.ones_like(s.gender).at[:,1,:].set(2),status=s.status.at[:,side,1].set(status),
            done=j.zeros_like(s.done))

    def incoming(self,state,side):
        return state._replace(active=state.active.at[:,side].set(1))

    def test_trace_copies_without_copied_start_and_preserves_originals(self):
        for side in (0,1):
            for foe in ('static','intimidate','drizzle','drought','sandstream','truant','trace'):
                s=self.incoming(self.fixture(side,foe),side)
                for run in (self.e.enter,self.jax.jit(self.e.enter)):
                    r,ko=run(s,side,self.et,eligible=self.j.array([True]))
                    self.assertEqual(int(r.ability[0,side,1]),self.abilities.index(foe))
                    self.assertEqual(int(r.base_ability[0,side,1]),1)
                    self.assertEqual(int(r.weather[0]),0)
                    self.assertEqual(int(r.boosts[0,1-side,0,0]),0)
                    self.assertFalse(bool(r.truant_loaf[0,side,1]))
                    self.assertFalse(bool(ko[0]))

    def test_passive_cures_clear_only_matching_status_and_confusion_storage(self):
        for side in (0,1):
            for foe,status in (('waterveil',2),('magmaarmor',6),('limber',1),('immunity',3),
                    ('immunity',4),('insomnia',5),('vitalspirit',5)):
                s=self.incoming(self.fixture(side,foe,status),side)
                s=s._replace(status_counter=s.status_counter.at[:,side,1].set(3),
                    sleep_skipped=s.sleep_skipped.at[:,side,1].set(2))
                r,_=self.e.enter(s,side,self.et,eligible=self.j.array([True]))
                self.assertEqual((int(r.status[0,side,1]),int(r.status_counter[0,side,1]),
                    int(r.sleep_skipped[0,side,1])),(0,0,0))
            s=self.incoming(self.fixture(side,'owntempo',3),side)
            s=s._replace(confusion_counter=s.confusion_counter.at[:,side,1].set(3))
            r,_=self.e.enter(s,side,self.et,eligible=self.j.array([True]))
            self.assertEqual((int(r.confusion_counter[0,side,1]),int(r.status[0,side,1])),(0,3))

    def test_trace_happens_after_hazards_and_never_on_ko_ineligible_or_absent_foe(self):
        for side in (0,1):
            s=self.incoming(self.fixture(side,'levitate',batch=4),side)
            s=s._replace(spikes=s.spikes.at[:,side].set(1),hp=s.hp.at[1,side,1].set(125)
                .at[3,1-side,0].set(0))
            r,ko=self.jax.jit(self.e.enter)(s,side,self.et,eligible=self.j.array([True,True,False,True]))
            self.assertEqual(r.hp[:,side,1].tolist(),[875,0,1000,875])
            self.assertEqual(r.ability[:,side,1].tolist(),[self.abilities.index('levitate'),1,1,1])
            self.assertEqual(ko.tolist(),[False,True,False,False])
            for before,after in zip(s,r):
                self.assertTrue(bool(self.j.array_equal(before[2],after[2])))

    def test_failed_trace_does_not_retry_when_only_the_opponent_enters(self):
        for side in (0,1):
            s=self.incoming(self.fixture(side,'static'),side)
            s=s._replace(hp=s.hp.at[:,1-side,0].set(0),
                entry_pending=s.entry_pending.at[:,side].set(True))
            first=self.t.initialize(s,self.table)
            self.assertEqual(int(first.ability[0,side,1]),1)
            following=first._replace(active=first.active.at[:,1-side].set(1),
                ability=first.ability.at[:,1-side,1].set(2),
                base_ability=first.base_ability.at[:,1-side,1].set(2),
                entry_pending=first.entry_pending.at[:,1-side].set(True))
            after=self.t.initialize(following,self.table)
            self.assertEqual(int(after.ability[0,side,1]),1)
            self.assertFalse(bool(self.j.any(after.entry_pending)))

    def test_truant_uses_pre_trace_current_ability_and_startup_context(self):
        for side in (0,1):
            for startup in (False,True):
                s=self.incoming(self.fixture(side,'truant',batch=3),side)
                current=self.j.array([1,7,1]);original=self.j.array([7,1,1])
                s=s._replace(ability=s.ability.at[:,side,1].set(current),
                    base_ability=s.base_ability.at[:,side,1].set(original))
                r,_=self.e.enter(s,side,self.et,eligible=self.j.ones(3,bool),startup=startup)
                self.assertEqual(r.truant_loaf[:,side,1].tolist(),[False,not startup,False])
                self.assertTrue(bool(self.j.array_equal(r.base_ability,s.base_ability)))

    def test_natural_cure_copy_waits_for_departure_and_nonmatching_status_stays(self):
        for side in (0,1):
            for foe in ('naturalcure','waterveil','magmaarmor','limber','insomnia','vitalspirit'):
                s=self.incoming(self.fixture(side,foe,3),side)
                s=s._replace(status_counter=s.status_counter.at[:,side,1].set(3))
                r,_=self.e.enter(s,side,self.et,eligible=self.j.array([True]))
                self.assertEqual((int(r.status[0,side,1]),int(r.status_counter[0,side,1])),(3,3))
                if foe=='naturalcure':
                    retired=self.e.depart(r,side,self.et,eligible=self.j.array([True]))
                    self.assertEqual((int(retired.status[0,side,1]),int(retired.ability[0,side,1])),(0,1))

    @unittest.skipUnless(NODE and SHOWDOWN,'requires pinned simulator source')
    def test_pinned_simulator_trace_handlers_and_passive_updates(self):
        result=subprocess.run([NODE,str(Path(__file__).with_name('gpu_trace_oracle.cjs')),SHOWDOWN],
            text=True,capture_output=True,check=True)
        data=json.loads(result.stdout)
        for row in data['copies']:
            self.assertEqual(row['current'],row['foe'])
            self.assertEqual(row['original'],'trace')
            self.assertEqual(row['copied_start_calls'],0)
        self.assertTrue(all(row['status']=='' for row in data['cures']))
        self.assertFalse(data['own_tempo_confusion'])
        self.assertEqual(data['no_retry'],['trace','trace'])
        self.assertEqual(data['natural_cure'],['psn','','trace'])


if __name__=='__main__':unittest.main()
