"""Both-seat installed-native regression, derived from the pinned source law."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('poke_engine'),'requires isolated native binding')
class NativeNaturalCureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import poke_engine
        from gpu_engine import future_state
        cls.engine,cls.f=poke_engine,future_state

    def fixture(self,side,*,status=0):
        from tests.test_gpu_engine_future_state import action_fixture
        data,meta=action_fixture();v=meta['vocabulary']
        for name in ('rest','spore'):
            if name not in v['moves']:v['moves'].append(name)
        for name in ('naturalcure','trace'):
            if name not in v['abilities']:v['abilities'].append(name)
        for name in ('altaria','mew'):
            if name not in v['species']:v['species'].append(name)
        rest,spore,splash=[v['moves'].index(n) for n in ('rest','spore','splash')]
        for s in (0,1):
            for p in range(6):data['moves'][s][p]=[splash]*4
            data['moves'][s][0]=[rest if s==side else spore,splash,10,11]
            data['stats'][s][0][5]=300 if s==side else 100
            data['stats'][s][1][5]=200
            data['species'][s][0]=v['species'].index('altaria' if s==side else 'mew')
        data['ability'][side][0]=data['base_ability'][side][0]=v['abilities'].index('naturalcure')
        data['hp'][side][0]=50;data['status'][side][0]=status
        if status==4:data['status_counter'][side][0]=3
        out=self.f.native_input_from_jax(data,None,meta,module=self.engine)
        self.assertIsInstance(out,self.f.NativeInput,type(out).__name__)
        return out.state,v['species'][data['species'][side][1]]

    def successors(self,state,side,own,other):
        before=state.to_string()
        choices=(own,other) if side==0 else (other,own)
        branches=self.engine.generate_instructions(state,*choices)
        self.assertEqual(state.to_string(),before)
        total=sum(b.percentage/100 for b in branches)
        self.assertAlmostEqual(total,1,places=6)
        results=[]
        for branch in branches:
            next_state=state.apply_instructions(branch)
            self.assertEqual(next_state.reverse_instructions(branch).to_string(),before)
            results.append((branch.percentage/100,next_state))
        return results

    def only(self,state,side,own,other):
        out=self.successors(state,side,own,other)
        self.assertEqual(len(out),1)
        self.assertEqual(out[0][0],1)
        return out[0][1]

    def mon(self,state,side):
        return (state.side_one if side==0 else state.side_two).pokemon[0]

    def test_sleep_cleanup_uses_current_ability_before_base_restoration(self):
        for side in (0,1):
            for rest,natural,refund in ((3,0,0),(2,0,1),(0,2,0)):
                for current,base,cured in (('naturalcure','trace',True),('none','naturalcure',False)):
                    state,target=self.fixture(side);pieces=state.to_string().split('/')
                    fields=pieces[side].split('=');mon=fields[0].split(',')
                    mon[8:10]=[current.upper(),base.upper()];mon[18:21]=['SLEEP',str(rest),str(natural)]
                    if refund:mon.append('refund:'+str(refund))
                    fields[0]=','.join(mon);pieces[side]='='.join(fields)
                    state=self.engine.State.from_string('/'.join(pieces))
                    out=self.only(state,side,target,'none');mon=self.mon(out,side)
                    self.assertEqual(mon.ability.lower(),base)
                    self.assertEqual(mon.status.lower(),'none' if cured else 'sleep')
                    self.assertEqual((mon.rest_turns,mon.sleep_turns,mon.rest_sleep_pending_refund),
                        (0,0,0) if cured else (rest,natural,refund))

    def test_new_primary_sleep_after_cure_matches_all_weighted_source_cases(self):
        for side in (0,1):
            state,target=self.fixture(side)
            state=self.only(state,side,'rest','splash')
            state=self.only(state,side,target,'splash')
            self.assertEqual(self.mon(state,side).rest_turns,0)
            state=self.only(state,side,'altaria','spore')
            self.assertEqual(self.mon(state,side).rest_turns,0)
            law=[(1,state)]
            for expected in (0,.25,.5,.75,1):
                law=[(mass*p,nxt) for mass,old in law for p,nxt in self.successors(old,side,'splash','splash')]
                self.assertAlmostEqual(sum(mass for mass,_ in law),1,places=6)
                awake=sum(mass for mass,nxt in law if self.mon(nxt,side).status.lower()=='none')
                self.assertAlmostEqual(awake,expected,places=6)

    def test_poison_toxic_and_other_major_cures_preserve_reversal(self):
        for side in (0,1):
            for status in (1,2,3,4):
                state,target=self.fixture(side,status=status)
                out=self.only(state,side,target,'none')
                self.assertEqual(self.mon(out,side).status.lower(),'none')
                self.assertEqual((out.side_one if side==0 else out.side_two).side_conditions.toxic_count,0)
