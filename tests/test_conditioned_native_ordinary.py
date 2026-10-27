"""Authored complete ordinary carry laws; no scientific roster replay."""
import importlib.util
import json
import unittest
from unittest import mock

AVAILABLE=all(importlib.util.find_spec(n) for n in ('jax','poke_engine','pokezero_search'))
if AVAILABLE:
    import pokezero_search
    AVAILABLE=hasattr(pokezero_search,'conditioned_step')


class CommonRefinementTests(unittest.TestCase):
    def test_native_cut_inside_jax_cell_is_retained_before_execution(self):
        from gpu_engine import ordinary_law as law
        from gpu_engine.finite_law import Axis,COORDINATES,Observation,Plan,CardinalityRefusal
        plan=Plan(tuple(Axis(name,(Observation('less_than',20),),('authored JAX consumer',))
            if name=='secondary.0' else Axis(name,(),inactive_reason='authored inactive consumer')
            for name,_ in COORDINATES))
        rows=[dict(coordinate='secondary.0',kind='less_than',argument=10)]
        refined=law.refine(plan,rows,cap=3)
        axis=refined.cells()[[n for n,_ in COORDINATES].index('secondary.0')]
        self.assertEqual([c.members for c in axis],[tuple(range(10)),tuple(range(10,20)),tuple(range(20,100))])
        with self.assertRaises(CardinalityRefusal):law.refine(plan,rows,cap=2)
        with self.assertRaises(ValueError):law.refine(plan,[dict(rows[0],argument=True)],cap=3)


@unittest.skipUnless(AVAILABLE,'requires isolated fixed-chance native bindings and JAX')
class OrdinaryCarryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_conditioned_native_resume import ConditionedResumeTests
        ConditionedResumeTests.setUpClass.__func__(cls)
        from gpu_engine import baton,critical,effects,entry,items,pressure,secondary,self_destruct
        cls.names=cls.meta['vocabulary']['moves']+['explosion','protect','endure','willowisp']
        cls.meta={**cls.meta,'vocabulary':{**cls.meta['vocabulary'],'moves':cls.names,
                'species':['_none','ditto','pikachu'],'items':['_none','leftovers','lightball']},
            'moves':[None]+[dict(id=name,source_type='Electric' if name=='thunderbolt' else 'Fire' if name=='willowisp' else 'Normal',
                category='Special' if name=='thunderbolt' else 'Physical' if name in ('bodyslam','explosion','struggle') else 'Status',
                flags={'protect':1,'contact':int(name=='bodyslam')}) for name in cls.names[1:]]}
        j=cls.j
        cls.table=cls.t.MoveTable(j.array([0,0,95,0,50,85,250,0,0,0]),
            j.array([0,1,13,1,0,1,1,1,1,10]),j.array([False,False,False,False,True,True,True,False,False,False]),
            j.full(10,100).at[9].set(75),j.array([True,True,False,True,False,False,False,True,True,False]),
            j.array([0,0,0,0,0,0,0,3,3,0]),cls.table.type_num,cls.table.type_den,j.array(4),
            baton=baton.from_metadata(cls.meta),critical=critical.from_metadata(cls.meta),
            effects=effects.from_metadata(cls.meta),entry=entry.from_metadata(cls.meta),items=items.from_metadata(cls.meta),
            pressure=pressure.from_metadata(cls.meta),secondary=secondary.from_metadata(cls.meta),
            self_destruct=self_destruct.from_metadata(cls.meta))

    def fixture(self,names=('splash','splash'),*,fast=0):
        j=self.j
        state=self.empty(1,seed=449)._replace(hp=j.full((1,2,6),1000),max_hp=j.full((1,2,6),1000),
            stats=j.full((1,2,6,6),200).at[...,0].set(1000),level=j.full((1,2,6),100),
            species=j.ones((1,2,6),j.int32),types=j.zeros((1,2,6,2),j.int32).at[...,0].set(1),
            moves=j.full((1,2,6,4),3),pp=j.full((1,2,6,4),10),done=j.array([False]),turn=j.array([7]))
        state=state._replace(stats=state.stats.at[:,fast,:,5].set(301).at[:,1-fast,:,5].set(101),
            moves=state.moves.at[:,0,0,2].set(self.names.index(names[0])).at[:,1,0,1].set(self.names.index(names[1])),
            base_ability=state.ability)
        return state,(2,1)

    def compare(self,state,actions,table=None,*,cap=16384,expected='AGREE',native_input=None):
        from gpu_engine import ordinary_law as law
        native=self.f.native_input_from_jax(state,0,self.meta,module=self.engine) if native_input is None else native_input
        self.assertIsInstance(native,self.f.NativeInput,native)
        result=law.compare_slot(state,actions,self.table if table is None else table,self.meta,
            cap=cap,batch_size=64,max_nodes=100000,native_input=native)
        self.assertEqual(result['status'],expected,result)
        return result,native

    def test_first_and_second_baton_both_seats_keep_exact_slots_and_full_bank(self):
        for pivot in (0,1):
            for first in (False,True):
                with self.subTest(pivot=pivot,first=first):
                    names=['splash','splash'];names[pivot]='batonpass'
                    s,a=self.fixture(names,fast=pivot if first else 1-pivot)
                    self.compare(s,a)

    def test_existing_pending_stall_storage_gap_diverges_without_erasing_physical_fields(self):
        # Both capsules retain the same next-attempt denominators. Native
        # retains physical stall storage until residual; JAX clears it at the
        # beginning of an ordinary request. This is an exposed engine gap, not
        # permission for a physical-key quotient or false AGREE.
        for pivot in (0,1):
            names=['splash','splash'];names[pivot]='batonpass'
            s,a=self.fixture(names,fast=pivot)
            s=s._replace(protect_counter=s.protect_counter.at[:,0,0].set(4).at[:,1,0].set(2))
            result,native_input=self.compare(s,a,expected='DIVERGE')
            self.assertEqual(result['comparison']['half_l1'],1)
            from gpu_engine import ordinary_law as law,carry_law as algebra
            from gpu_engine.fidelity_panel import JaxCaseExecutor
            plan=law.plan_for_state(s,a,self.table,self.meta,cap=16384,native_input=native_input)
            cell=next(algebra.execution_cells(plan.law,cap=16384))
            reference=JaxCaseExecutor(s,a,self.table,plan,batch_size=1)((cell.case,))[0]
            v=dict(cell.case.values)
            chance={'draws':[[v[f'draws.{side}.{i}'] for i in range(12)] for side in range(2)],
                **{field:[v[f'{field}.{side}'] for side in range(2)]
                    for field in ('phazing','secondary','sleep_talk','entry_ties')},'initial_tie':v['initial_tie']}
            request=dict(version=1,schema='pokezero.ordinary-step',actions=a,chance=chance,address=dict(plan.address))
            raw=json.loads(pokezero_search.conditioned_step(native_input.raw_native,json.dumps(request)))['branches'][0]
            native=law._native_output(raw,native_input.certificate)
            self.assertEqual(reference.capsule,native.capsule)
            self.assertEqual(self.f.field_differences(reference.physical,native.physical),
                (('side_one.side_conditions.protect',0,2),('side_two.side_conditions.protect',0,1)))

    def test_closed_and_action_phase_single_joint_faints_and_terminal_exhaustion(self):
        self.compare(*self.fixture())
        for side in (0,1):
            for kind in ('single','joint','terminal'):
                with self.subTest(side=side,kind=kind):
                    names=['splash','splash'];names[side]='explosion'
                    s,a=self.fixture(names,fast=side)
                    if kind in ('joint','terminal'):s=s._replace(hp=s.hp.at[:,1-side,0].set(1))
                    if kind=='terminal':s=s._replace(hp=s.hp.at[:,:,1:].set(0))
                    self.compare(s,a)

    def test_target_and_contact_paralysis_damage_rolls_and_critical_extremes(self):
        for side in (0,1):
            for move in ('thunderbolt','bodyslam'):
                with self.subTest(side=side,move=move):
                    names=['splash','splash'];names[side]=move
                    s,a=self.fixture(names,fast=side)
                    if move=='bodyslam':s=s._replace(ability=s.ability.at[:,1-side,0].set(1),
                        base_ability=s.base_ability.at[:,1-side,0].set(1))
                    self.compare(s,a)

    def test_initial_and_new_paralysis_and_stall_success_failures(self):
        for move in ('protect','endure'):
            for side in (0,1):
                with self.subTest(move=move,side=side):
                    names=['splash','splash'];names[side]=move
                    s,a=self.fixture(names)
                    s=s._replace(status=s.status.at[:,side,0].set(1),
                        protect_counter=s.protect_counter.at[:,side,0].set(4))
                    self.compare(s,a)
        # A fast hit can apply paralysis before the slower actor's gate.
        s,a=self.fixture(('thunderbolt','splash'))
        self.compare(s,a)

    def test_wrong_actual_jax_threshold_diverges_on_native_common_refinement(self):
        s,a=self.fixture(('thunderbolt','splash'))
        secondary=self.table.secondary._replace(chance=self.table.secondary.chance.at[2].set(20))
        result,_=self.compare(s,a,self.table._replace(secondary=secondary),expected='DIVERGE')
        self.assertGreater(result['comparison']['half_l1'],0)

    def test_caps_and_missing_or_changed_retained_input_refuse_whole_slot(self):
        from gpu_engine import ordinary_law as law
        s,a=self.fixture(('thunderbolt','splash'))
        self.compare(s,a,cap=1,expected='UNKNOWN')
        result=law.compare_slot(s,a,self.table,self.meta,cap=512,batch_size=64,max_nodes=1000,native_input=None)
        self.assertEqual(result['status'],'UNKNOWN')
        original=self.f.native_input_from_jax(s,0,self.meta,module=self.engine)
        from gpu_engine.fidelity_panel import native_executor
        plan=law.plan_for_state(s,a,self.table,self.meta,cap=16384,native_input=original)
        self.assertIsInstance(plan,law.OrdinaryPlan,plan)
        with self.assertRaisesRegex(ValueError,'PUBLIC_STRING_EXECUTOR_CANNOT_EXECUTE_SLOT_API_ACTIONS'):
            native_executor(plan)
        changed=s._replace(hp=s.hp.at[:,0,0].set(900))
        self.compare(changed,a,native_input=original,expected='UNKNOWN')
        # Existing retained provenance is used; no reconstruction from a
        # current JAX snapshot may silently replace native-only static fields.
        with mock.patch.object(self.f,'native_input_from_jax',side_effect=AssertionError('unexpected reconstruction')):
            self.compare(s,a,native_input=original)

    def test_low_nonimmune_minimum_and_true_immunity_complete_laws_both_seats(self):
        table=self.table._replace(type_den=self.table.type_den.at[13,16].set(2)
            .at[1,6].set(2).at[1,9].set(2))
        for side in (0,1):
            for move in ('thunderbolt','bodyslam','explosion'):
                for kind in ('single','double','immune'):
                    with self.subTest(side=side,move=move,kind=kind):
                        names=['splash','splash'];names[side]=move
                        s,a=self.fixture(names,fast=side)
                        electric=move=='thunderbolt'
                        s=s._replace(stats=s.stats.at[:,side,0,1].set(1).at[:,side,0,3].set(1)
                                .at[:,1-side,0,2].set(30000).at[:,1-side,0,4].set(30000),
                            types=s.types.at[:,1-side,0,0].set((5 if electric else 8) if kind=='immune' else (12 if electric else 6))
                                .at[:,1-side,0,1].set((16 if electric else 9) if kind=='double' else 0))
                        self.compare(s,a,table)

    def test_primary_burn_and_light_ball_complete_laws_both_seats_and_orders(self):
        for burner in (0,1):
            for fast_burner in (False,True):
                for move,species in (('thunderbolt',1),('thunderbolt',2),('bodyslam',2)):
                    with self.subTest(burner=burner,fast=fast_burner,move=move,species=species):
                        names=[move,move];names[burner]='willowisp'
                        s,a=self.fixture(names,fast=burner if fast_burner else 1-burner)
                        s=s._replace(item=s.item.at[:,1-burner,0].set(2),
                            species=s.species.at[:,1-burner,0].set(species))
                        self.compare(s,a)

    def test_primary_burn_status_and_fire_immunity_and_baton_outputs(self):
        for burner in (0,1):
            for kind in ('fire','paralyzed','burned','protect','first_baton','second_baton'):
                with self.subTest(burner=burner,kind=kind):
                    names=['splash','splash'];names[burner]='willowisp'
                    if 'baton' in kind:names[1-burner]='batonpass'
                    if kind=='protect':names[1-burner]='protect'
                    s,a=self.fixture(names,fast=1-burner if kind=='first_baton' else burner)
                    if kind=='fire':s=s._replace(types=s.types.at[:,1-burner,0,0].set(10))
                    if kind in ('paralyzed','burned'):
                        s=s._replace(status=s.status.at[:,1-burner,0].set(1 if kind=='paralyzed' else 2))
                    self.compare(s,a)

    def test_wrong_actual_jax_primary_accuracy_keeps_native_cut_and_diverges(self):
        s,a=self.fixture(('willowisp','splash'))
        changed=self.table._replace(accuracy=self.table.accuracy.at[9].set(65))
        result,_=self.compare(s,a,changed,expected='DIVERGE')
        self.assertGreater(result['comparison']['half_l1'],0)

    def tie(self,state):
        return state._replace(stats=state.stats.at[:,:,0,5].set(200))

    def test_tied_action_order_retains_both_seat_baton_and_ko_carry(self):
        for pivot in (0,1):
            for lethal in (False,True):
                with self.subTest(pivot=pivot,lethal=lethal):
                    names=['thunderbolt','thunderbolt'];names[pivot]='batonpass'
                    s,a=self.fixture(names);s=self.tie(s)
                    if lethal:s=s._replace(hp=s.hp.at[:,pivot,0].set(1))
                    self.compare(s,a)

    def test_tied_residuals_use_independent_status_coin_and_items_before_status(self):
        for loser in (0,1):
            for kind in ('poison','items','nonterminal'):
                with self.subTest(loser=loser,kind=kind):
                    s,a=self.fixture();s=self.tie(s)
                    s=s._replace(status=s.status.at[:,:,0].set(2 if kind=='items' else 3),
                        hp=s.hp.at[:,:,0].set(1))
                    if kind!='nonterminal':s=s._replace(hp=s.hp.at[:,:,1:].set(0))
                    if kind=='items':s=s._replace(item=s.item.at[:,:,0].set(1),
                        hp=s.hp.at[:,1-loser,0].set(100))
                    self.compare(s,a)

    def test_initial_paralysis_and_new_paralysis_created_and_broken_ties(self):
        for actor in (0,1):
            for kind in ('initial','created','broken','priority'):
                with self.subTest(actor=actor,kind=kind):
                    names=['splash','splash'];names[actor]='protect' if kind=='priority' else 'thunderbolt'
                    s,a=self.fixture(names);s=self.tie(s)
                    if kind=='initial':s=s._replace(stats=s.stats.at[:,actor,0,5].set(800),
                        status=s.status.at[:,actor,0].set(1))
                    if kind=='created':s=s._replace(stats=s.stats.at[:,1-actor,0,5].set(800),
                        status=s.status.at[:,actor,0].set(3))
                    if kind=='broken':s=s._replace(status=s.status.at[:,actor,0].set(3))
                    self.compare(s,a)
                    if kind in ('created','broken'):
                        # A pre-move speed snapshot would wrongly heal the
                        # slower paralyzed holder after terminal poison. The
                        # created-tie side is the source-proven limiting case:
                        # its item still precedes status at equal speed.
                        terminal=s._replace(hp=s.hp.at[:,:,1:].set(0)
                            .at[:,actor,0].set(1).at[:,1-actor,0].set(700),
                            item=s.item.at[:,1-actor,0].set(1))
                        _,native=self.compare(terminal,a)
                        from gpu_engine import ordinary_law as law
                        plan=law.plan_for_state(terminal,a,self.table,self.meta,cap=16384,native_input=native)
                        hp=[]
                        for secondary_roll in (0,99):
                            chance=dict(draws=[[0,99,99,99,15,7,0,0,0,0,0,0] for _ in range(2)],
                                phazing=[59,59],secondary=[99,99],sleep_talk=[59,59],entry_ties=[0,1],initial_tie=0)
                            chance['secondary'][actor]=secondary_roll
                            raw=json.loads(pokezero_search.conditioned_step(native.raw_native,json.dumps(dict(
                                version=1,schema='pokezero.ordinary-step',actions=a,chance=chance,address=dict(plan.address)))))
                            self.assertEqual(len(raw['branches']),1)
                            wire=self.f._native_wire(raw['branches'][0]['state'])
                            holder=wire['side_two' if actor==0 else 'side_one']['pokemon'][0]
                            self.assertEqual(holder['status'],'paralyze' if secondary_roll==0 else 'none')
                            hp.append(holder['hp'])
                        self.assertEqual(hp[1]-hp[0],62 if kind=='broken' else 0)

    def test_unequal_residual_speed_still_precedes_suborder(self):
        for loser in (0,1):
            for loser_first in (False,True):
                with self.subTest(loser=loser,loser_first=loser_first):
                    s,a=self.fixture(fast=loser if loser_first else 1-loser)
                    s=s._replace(status=s.status.at[:,:,0].set(2),
                        item=s.item.at[:,:,0].set(1),hp=s.hp.at[:,:,1:].set(0)
                            .at[:,loser,0].set(1).at[:,1-loser,0].set(100))
                    self.compare(s,a)


if __name__=='__main__':unittest.main()
