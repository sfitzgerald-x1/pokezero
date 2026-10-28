"""Authored nine-slot ordinary switch laws, with complete pending successors."""
import importlib.util
import json
from pathlib import Path
import unittest

AVAILABLE=all(importlib.util.find_spec(n) for n in ('jax','poke_engine','pokezero_search'))
if AVAILABLE:
    import pokezero_search
    AVAILABLE=bool(getattr(pokezero_search,'CONDITIONED_ORACLE_ENABLED',False))


@unittest.skipUnless(AVAILABLE,'requires isolated current native bindings and JAX')
class OrdinarySwitchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        import poke_engine
        from gpu_engine import future_state,ordinary_law,transition
        from gpu_engine.state import empty_state
        cls.jax,cls.j,cls.engine,cls.f,cls.law,cls.tr= jax,j,poke_engine,future_state,ordinary_law,transition
        cls.empty=staticmethod(empty_state)
        cls.catalog=json.loads((Path(__file__).parent/'data/gpu_ordinary_switch_catalog.json').read_text())
        cls.meta=cls.catalog['metadata'];cls.table=transition.tables_from_metadata(cls.meta)

    def id(self,kind,name):return self.meta['vocabulary'][kind].index(name)

    def fixture(self,switcher=0,*,move='earthquake',incoming_hp=30,immunity=False):
        j=self.j;foe=1-switcher;s=self.empty(1,seed=1681)
        # Authored values differ from every preserved scientific root. Party2
        # switches to party3: raw action6 addresses the third inactive slot.
        s=s._replace(hp=j.full_like(s.hp,300),max_hp=j.full_like(s.max_hp,300),
            stats=j.full_like(s.stats,173).at[...,0].set(300).at[:,switcher,:,5].set(311)
                .at[:,foe,:,5].set(151),level=j.full_like(s.level,77),
            species=j.full_like(s.species,self.id('species','ditto')),
            types=s.types.at[...,0].set(self.id('types','normal')),
            moves=j.full_like(s.moves,self.id('moves','splash')),pp=j.full_like(s.pp,10),
            active=s.active.at[:,switcher].set(2).at[:,foe].set(5),done=j.array([False]),turn=j.array([17]))
        s=s._replace(species=s.species.at[:,switcher,2].set(self.id('species','golduck'))
                .at[:,switcher,3].set(self.id('species','weezing' if immunity else 'flareon'))
                .at[:,foe,5].set(self.id('species','camerupt')),
            ability=s.ability.at[:,switcher,2].set(self.id('abilities','cloudnine'))
                .at[:,switcher,3].set(self.id('abilities','levitate' if immunity else 'flashfire'))
                .at[:,foe,5].set(self.id('abilities','magmaarmor')),
            item=s.item.at[:,switcher,2].set(self.id('items','leftovers'))
                .at[:,switcher,3].set(self.id('items','choiceband')).at[:,foe,5].set(self.id('items','choiceband')),
            types=s.types.at[:,switcher,2,0].set(self.id('types','water'))
                .at[:,switcher,3,0].set(self.id('types','poison' if immunity else 'fire'))
                .at[:,foe,5,0].set(self.id('types','fire')).at[:,foe,5,1].set(self.id('types','ground')),
            moves=s.moves.at[:,foe,5].set(j.array([self.id('moves',name) for name in
                ([name for name in ('explosion','fireblast','rockslide','earthquake') if name!=move][:3]+[move])])),
            hp=s.hp.at[:,switcher,3].set(incoming_hp))
        s=s._replace(base_ability=s.ability)
        actions=[3,3];actions[switcher]=6
        return s,tuple(actions)

    def compare(self,s,a,*,expected='AGREE',table=None):
        native=self.f.native_input_from_jax(s,0,self.meta,module=self.engine)
        self.assertIsInstance(native,self.f.NativeInput,native)
        table=self.table if table is None else table
        plan=self.law.plan_for_state(s,a,table,self.meta,cap=262144,native_input=native)
        self.assertIsInstance(plan,self.law.OrdinaryPlan,plan)
        result=self.law.compare_plan(plan,s,a,table,cap=262144,batch_size=64,
            max_nodes=100000,retain_evidence=True)
        self.assertEqual(result['status'],expected,
            {k:(dict(v, witness='retained in full emissions') if k=='comparison' else v)
                for k,v in result.items() if k!='ordinary_output_laws'})
        return result,native

    def emissions(self,result):return result['ordinary_output_laws']['emissions']

    def assert_carry(self,result,*,pending,forced=None):
        for emission in self.emissions(result):
            for engine in ('jax','native'):
                out=emission[engine];c=out['capsule']
                self.assertEqual(c['pending_residual'],pending)
                self.assertEqual(c['queued_actor'],-1)
                self.assertEqual(tuple(c['queued_actions']),(0,0))
                self.assertEqual(tuple(c['queued_stall']),(0,0))
                self.assertEqual(c['replacement_count'],0)
                if forced is not None:self.assertEqual(tuple(c['forced_switch']),forced)
                self.assertEqual(c['turn'],18)
                self.assertEqual(len(out['expressions']),30)
                for expression in out['expressions']:
                    self.assertIn('copy' if pending else 'constant',expression)
                    if not pending:self.assertEqual(expression['constant'],0)

    def test_switch_earthquake_ko_survival_and_immunity_complete_laws_both_seats(self):
        for side in (0,1):
            for hp,immune in ((30,False),(300,False),(300,True)):
                with self.subTest(side=side,hp=hp,immune=immune):
                    result,_=self.compare(*self.fixture(side,incoming_hp=hp,immunity=immune))
                    if hp==30:self.assert_carry(result,pending=True)
                    elif immune:self.assert_carry(result,pending=False)

    def test_hazard_ko_cancels_waiting_move_or_switch_and_preserves_bank(self):
        for side in (0,1):
            for other_switch in (False,True):
                with self.subTest(side=side,other_switch=other_switch):
                    s,a=self.fixture(side,incoming_hp=37)
                    s=s._replace(spikes=s.spikes.at[:,side].set(1))
                    if other_switch:
                        a=list(a);a[1-side]=4;a=tuple(a)
                    r,_=self.compare(s,a);self.assert_carry(r,pending=True)
                    foe=('side_one','side_two')[1-side]
                    for e in self.emissions(r):
                        physical=dict(e['native']['physical'])
                        self.assertEqual(physical[foe+'.active_index'],'5')
                        self.assertEqual(physical[foe+'.pokemon[5].moves[3].pp'],10)

    def test_switch_versus_baton_and_terminal_explosion_has_no_queued_switch(self):
        for side in (0,1):
            for move in ('batonpass','explosion'):
                with self.subTest(side=side,move=move):
                    s,a=self.fixture(side,move=move,incoming_hp=30)
                    if move=='explosion':s=s._replace(hp=s.hp.at[:,1-side,:5].set(0))
                    r,_=self.compare(s,a)
                    self.assert_carry(r,pending=move=='batonpass')

    def test_residual_poison_faint_is_closed_and_entrant_tie_uses_residual_coin(self):
        for side in (0,1):
            s,a=self.fixture(side,move='splash',incoming_hp=30)
            s=s._replace(status=s.status.at[:,side,3].set(3))
            r,_=self.compare(s,a);self.assert_carry(r,pending=False)
            # Only the incoming speed ties; outgoing speeds remain311/151.
            # If the terminal foe faints first, later poison must not execute.
            s,a=self.fixture(side,move='splash',incoming_hp=37)
            s=s._replace(stats=s.stats.at[:,side,3,5].set(151),
                status=s.status.at[:,side,3].set(3).at[:,1-side,5].set(3),
                hp=s.hp.at[:,1-side,:5].set(0).at[:,1-side,5].set(37))
            r,n=self.compare(s,a);self.assert_carry(r,pending=False)
            import pokezero_search
            cuts=json.loads(pokezero_search.conditioned_step_requirements(n.raw_native,json.dumps(a)))['requirements']
            self.assertIn(dict(coordinate='draws.1.9',kind='identity',argument=0),cuts)
            mon=('side_one','side_two')[side]+'.pokemon[3].hp'
            self.assertEqual({dict(e['native']['physical'])[mon] for e in self.emissions(r)},{0,37})

    def test_tied_voluntary_switch_hazard_cancellation_uses_its_own_coin(self):
        s,a=self.fixture(0,incoming_hp=37)
        s=s._replace(stats=s.stats.at[:,1,5,5].set(311),spikes=s.spikes.at[:,:].set(1),
            hp=s.hp.at[:,1,0].set(37))
        r,n=self.compare(s,(6,4));self.assert_carry(r,pending=True)
        import pokezero_search
        cuts=json.loads(pokezero_search.conditioned_step_requirements(n.raw_native,'[6,4]'))['requirements']
        self.assertIn(dict(coordinate='draws.0.11',kind='identity',argument=0),cuts)
        self.assertEqual({tuple(dict(e['native']['physical'])[side+'.active_index']
            for side in ('side_one','side_two')) for e in self.emissions(r)},{('3','5'),('2','0')})

    def test_incoming_static_contact_consumers_and_wrong_damage_table(self):
        for side in (0,1):
            s,a=self.fixture(side,move='bodyslam',incoming_hp=300)
            s=s._replace(ability=s.ability.at[:,side,3].set(self.id('abilities','static')))
            s=s._replace(base_ability=s.ability)
            r,n=self.compare(s,a)
            import pokezero_search
            cuts=json.loads(pokezero_search.conditioned_step_requirements(n.raw_native,json.dumps(a)))['requirements']
            self.assertIn(dict(coordinate=f'phazing.{1-side}',kind='less_than',argument=20),cuts)
        s,a=self.fixture(incoming_hp=300)
        table=self.table._replace(power=self.table.power.at[self.id('moves','earthquake')].set(1))
        r,_=self.compare(s,a,table=table,expected='DIVERGE')
        self.assertGreater(r['comparison']['half_l1'],0)

    def test_fainted_predecessor_and_duplicate_species_keep_exact_party_slot(self):
        for side in (0,1):
            s,a=self.fixture(side,incoming_hp=300)
            s=s._replace(hp=s.hp.at[:,side,0].set(0),
                species=s.species.at[:,side,1].set(self.id('species','flareon')))
            r,n=self.compare(s,a)
            typed=self.f.native_actions_from_jax(s,0,a,n,exact_move_slots=True)
            self.assertIsInstance(typed,self.f.NativeMoveSlotActions)
            legacy=self.f.native_actions_from_jax(s,0,a,n)
            self.assertIsInstance(legacy,self.f.Refused)
            self.assertIn('DUPLICATE_OR_RESERVED_SWITCH_ID',{g.code for g in legacy.gaps})
            for e in self.emissions(r):
                self.assertEqual(dict(e['native']['physical'])[('side_one','side_two')[side]+'.active_index'],'3')

    def test_flash_fire_wisp_accuracy_and_protect_complete_laws_both_seats(self):
        for side in (0,1):
            for protect in (False,True):
                s,a=self.fixture(side,move='willowisp',incoming_hp=300)
                # A copied Flash Fire on a non-Fire type exercises absorption
                # rather than Fire-type status immunity. The base is retained.
                s=s._replace(ability=s.ability.at[:,side,2].set(self.id('abilities','flashfire')),
                    types=s.types.at[:,side,2,0].set(self.id('types','normal')),
                    moves=s.moves.at[:,side,2,2].set(self.id('moves','protect' if protect else 'splash')))
                a=list(a);a[side]=2;a=tuple(a)
                self.compare(s,a)



if __name__=='__main__':unittest.main()
