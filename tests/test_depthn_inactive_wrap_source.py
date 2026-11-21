"""Public party joins and the existing engine lifecycle for a suspended partial trap."""
from types import SimpleNamespace as NS
import unittest

import numpy as np

from gpu_engine import phase_d_protocol_world as protocol
from gpu_engine.phase_d_wrap_source import inactive_sources
from tests.test_depthn_trap_walls import host_world


class PublicSourceTests(unittest.TestCase):
    def case(self,owner=0,source=0):
        seat=f'p{source+1}';target=f'p{2-source}'
        lines=(f'|switch|{seat}a: Binder|Shuckle, L98, M|100/100',
            f'|switch|{target}a: Target|Octillery, L87, F|100/100','|turn|46',
            f'|-activate|{target}a: Target|move: Wrap|[of] {seat}a: Binder',
            f'|-damage|{target}a: Target|24/100|[from] move: Wrap|[partiallytrapped]','|upkeep','|turn|47',
            f'|-damage|{target}a: Target|12/100|[from] move: Wrap|[partiallytrapped]','|upkeep','|turn|48',
            f'|switch|{seat}a: Replacement|Ho-Oh, L70|100/100',f'|faint|{seat}a: Replacement')
        world=host_world(active=(0,0));world.species[0,source]=[2,3,1,4,5,6];world.species[0,1-source]=[7,8,9,10,11,12]
        world.hp[0,source,0]=0
        meta=dict(vocabulary=dict(species=('','shuckle','hooh','a','b','c','d','octillery','e','f','g','h','i')))
        request=dict(side=dict(id=f'p{owner+1}',pokemon=[dict(ident=seat+': '+('Binder' if k==2 else 'Other'+str(k)),
            details=meta['vocabulary']['species'][int(world.species[0,source,k])]+', L100') for k in range(6)]))
        view=dict(present=np.ones((1,6),bool),species=np.asarray([[3,1,2,4,5,6]],np.int32))
        args=dict(lines=lines,request=request,detached=NS(view=view),metadata=meta,assignment={0:1,1:2,2:0,3:3,4:4,5:5},owner=owner,
            replacement=dict(pending_residual=True))
        return world,(1-source,),protocol._wrap_facts(lines),args

    def test_both_owners_and_permuted_foe_slots_keep_the_real_bench_source(self):
        for owner in (0,1):
            for source in (0,1):
                with self.subTest(owner=owner,source=source):
                    world,wrapped,facts,args=self.case(owner,source)
                    slots=inactive_sources(world,wrapped,facts,**args);self.assertEqual(slots,{1-source:2})
                    out=protocol._write_wraps(world,wrapped,facts,17,sources=slots)
                    self.assertEqual(int(out.partial_trap_source[0,1-source,0]),2)
                    self.assertIn(int(out.partial_trap_counter[0,1-source,0]),(1,2,3,4))
                    self.assertEqual(int(world.partial_trap_counter[0,1-source,0]),0)
                    # Joining a departed source changes neither the support nor that side's draw.
                    active_facts=tuple(None if f is None else dict(f,trapper_active=True) for f in facts)
                    old=protocol._write_wraps(world,wrapped,active_facts,17)
                    np.testing.assert_array_equal(out.partial_trap_counter,old.partial_trap_counter)

    def test_missing_wrong_side_ambiguous_and_conflicting_identities_refuse(self):
        cases=[]
        for mutate in ('missing','duplicate','wrongside','notactive','conflict','wrongrequest','dead','duplicateown','duplicatephysical'):
            world,wrapped,facts,args=self.case();lines=list(args['lines'])
            if mutate=='missing':lines[3]=lines[3].split('|[of]')[0]
            elif mutate=='duplicate':lines[3]+='|[of] p1a: Binder'
            elif mutate=='wrongside':lines[3]=lines[3].replace('[of] p1a','[of] p2a')
            elif mutate=='notactive':lines[0]=lines[0].replace('Binder','Another')
            elif mutate=='conflict':lines.insert(1,'|replace|p1a: Binder|Unown, L100|100/100')
            elif mutate=='wrongrequest':args['request']['side']['pokemon'][2]['details']='Unown, L100'
            elif mutate=='dead':world.hp[0,0,2]=0
            elif mutate=='duplicateown':args['request']['side']['pokemon'][1]['ident']='p1: Binder'
            elif mutate=='duplicatephysical':world.species[0,0,1]=1
            args['lines']=lines;cases.append((mutate,world,wrapped,facts,args))
        world,wrapped,facts,args=self.case(owner=1,source=0);args['assignment'][0]=2;cases.append(('noninjective',world,wrapped,facts,args))
        world,wrapped,facts,args=self.case(owner=1,source=0);args['detached'].view['species'][0,0]=1;cases.append(('duplicatepublic',world,wrapped,facts,args))
        for name,world,wrapped,facts,args in cases:
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'PUBLIC_WRAP_SOURCE_IDENTITY_UNAVAILABLE'):
                inactive_sources(world,wrapped,facts,**args)

    def test_only_a_pre_residual_replacement_can_use_the_new_join(self):
        for replacement,later in ((None,()),(dict(pending_residual=False),()),(dict(pending_residual=True),('|upkeep',))):
            world,wrapped,facts,args=self.case();args.update(replacement=replacement,lines=args['lines']+later)
            with self.subTest(replacement=replacement,later=later),self.assertRaisesRegex(ValueError,'PUBLIC_WRAP_TRAPPER_NOT_ACTIVE'):
                inactive_sources(world,wrapped,facts,**args)
        # Existing active sources need none of the additional public joins.
        self.assertEqual(inactive_sources(None,(1,),(None,dict(trapper_active=True)),lines=(),request=None,
            detached=None,metadata=None,assignment=None,owner=0,replacement=None),{})


class ReplacementLifecycleTests(unittest.TestCase):
    def test_bench_or_newly_returned_source_releases_at_the_existing_residual(self):
        from tests.test_gpu_engine_linked_traps import LinkedTrapTests
        from gpu_engine.state import partially_trapped
        helper=LinkedTrapTests();helper.setUp();j=helper.j
        for source in (0,1):
            holder=1-source
            initial=helper.live(own='splash',foe='splash')
            paused=initial._replace(hp=initial.hp.at[0,source,0].set(0),
                forced_switch=initial.forced_switch.at[0,source].set(True),pending_residual=j.asarray([True]),
                active_turns=j.asarray([[0 if s==source else 5 for s in (0,1)]],j.int32),
                partial_trap_counter=initial.partial_trap_counter.at[0,holder,0].set(4),
                partial_trap_source=initial.partial_trap_source.at[0,holder,0].set(2))
            self.assertEqual(int(paused.partial_trap_source[0,holder,0]),2)
            self.assertFalse(bool(partially_trapped(paused)[0,holder]))
            returned=paused._replace(active=paused.active.at[0,source].set(2))
            self.assertTrue(bool(partially_trapped(returned)[0,holder])) # age0 affects residual release, not request trapping
            for replacement in (1,2):
                with self.subTest(source=source,replacement=replacement):
                    actions=[[0,0]];actions[0][source]=replacement+3 # active0: action4 -> slot1, action5 -> slot2
                    after=helper.play(paused,actions).state
                    self.assertEqual(int(after.active[0,source]),replacement)
                    self.assertEqual(int(after.hp[0,holder,0]),1000)
                    self.assertEqual(int(after.partial_trap_counter[0,holder,0]),0)
                    self.assertEqual(int(after.partial_trap_source[0,holder,0]),-1)
                    self.assertFalse(bool(partially_trapped(after)[0,holder]))


if __name__=='__main__':unittest.main()
