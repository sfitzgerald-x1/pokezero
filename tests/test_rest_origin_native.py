"""Actual native91 full ordinary Rest successor/carry controls, not panel evidence."""
import copy
import importlib.util
import unittest

AVAILABLE=all(importlib.util.find_spec(n) for n in ('jax','poke_engine'))


@unittest.skipUnless(AVAILABLE,'requires isolated JAX and native91 binding')
class NativeRestOriginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import poke_engine
        from gpu_engine import future_state
        from tests import test_gpu_engine_rest_origin as cases
        cases.RestOriginTests.setUpClass();cls.h=cases.RestOriginTests()
        cls.engine,cls.f=poke_engine,future_state

    def construct(self,s):
        out=self.f.native_input_from_jax(s,0,self.h.meta,module=self.engine)
        self.assertIsInstance(out,self.f.NativeInput,getattr(out,'gaps',()))
        return out

    def advance(self,s,native,certificate,side,action):
        actions=self.h.actions(side,action)
        out=self.h.play(s,side,action)
        names=[]
        for seat in range(2):
            a=int(actions[0,seat]);p=int(s.active[0,seat])
            if a<4:name=self.h.meta['vocabulary']['moves'][int(s.moves[0,seat,p,a])]
            else:
                bench=[i for i in range(6) if i!=p];target=bench[a-4]
                name=self.h.meta['vocabulary']['species'][int(s.species[0,seat,target])]
            names.append(name)
        before=native.to_string();branches=self.engine.generate_instructions(native,*names)
        self.assertEqual(native.to_string(),before)
        self.assertAlmostEqual(sum(b.percentage/100 for b in branches),1,places=6)
        left=self.f.canonical_jax_future(out,0,self.h.meta,certificate=certificate)
        self.assertIsInstance(left,self.f.CanonicalFuture,getattr(left,'gaps',()))
        states=[]
        for b in branches:
            nxt=native.apply_instructions(b)
            self.assertEqual(nxt.reverse_instructions(b).to_string(),before)
            right=self.f.canonical_native_future(nxt,certificate)
            self.assertIsInstance(right,self.f.CanonicalFuture,getattr(right,'gaps',()))
            self.assertEqual(left.key,right.key,self.f.field_differences(left.key,right.key))
            states.append(nxt)
        self.assertTrue(states)
        # Every positive branch has the same complete future key; representative
        # carry is valid only after the equality assertion above for ALL branches.
        return out,states[0]

    def test_both_seat_rest_refund_discard_reentry_wake_and_recast_cycles(self):
        for side in range(2):
            for ordinary in (False,True):
                s=self.h.live();n=self.construct(s);native=n.state
                for action in (0,1,2 if ordinary else 1,4,4):
                    s,native=self.advance(s,native,n.certificate,side,action)
                while int(s.status[0,side,0])==5:
                    s,native=self.advance(s,native,n.certificate,side,2)
                # Exact original history is carried; a new Rest stint resets bank.
                s=s._replace(hp=s.hp.at[0,side,0].set(50))
                # Intentional authored HP intervention is applied to both roots.
                parts=native.to_string().split('/');fields=parts[side].split('=');mon=fields[0].split(',');mon[6]='50'
                fields[0]=','.join(mon);parts[side]='='.join(fields);native=self.engine.State.from_string('/'.join(parts))
                s,native=self.advance(s,native,n.certificate,side,0)
                self.h.assert_sleep(s,side,5,3,0,True)

    def test_early_bird_and_current_natural_cure_match_complete_successors(self):
        for side in range(2):
            for ability,base in ((2,2),(1,6),(0,1)):
                s=self.h.sleeper(side,counter=3,bank=0)
                s=s._replace(ability=s.ability.at[0,side,0].set(ability),base_ability=s.base_ability.at[0,side,0].set(base))
                n=self.construct(s);native=n.state
                actions=(1,2) if ability==2 else (1,4,4,2)
                for a in actions:s,native=self.advance(s,native,n.certificate,side,a)

    def test_typed_rest_fields_exact_keys_and_legacy_natural_refusals(self):
        keys=[]
        for counter,refund in ((3,0),(2,0),(2,1),(1,2)):
            s=self.h.sleeper(0,counter=counter,bank=refund);n=self.construct(s)
            key=self.f.canonical_native_future(n.state,n.certificate)
            self.assertIsInstance(key,self.f.CanonicalFuture);keys.append(key.key)
        self.assertEqual(len(set(keys)),len(keys))
        s=self.h.live();data=self.f._sample(s,0);del data['rest_origin']
        out=self.f.native_input_from_jax(data,None,self.h.meta,module=self.engine)
        self.assertIsInstance(out,self.f.Refused);self.assertIn('REST_ORIGIN_REQUIRED',{g.code for g in out.gaps})
        for origin,status,counter,refund in ((False,0,0,1),(1,5,3,0),(True,0,3,0),(True,5,0,0),(True,5,4,0),(True,5,2,4),(True,5,2,-1)):
            data=copy.deepcopy(self.f._sample(s,0));data={k:[list(vv) if isinstance(vv,tuple) else vv for vv in v] if k in ('rest_origin','status','status_counter','sleep_skipped') else v for k,v in data.items()}
            data['rest_origin'][0][0]=origin;data['status'][0][0]=status;data['status_counter'][0][0]=counter;data['sleep_skipped'][0][0]=refund
            out=self.f.native_input_from_jax(data,None,self.h.meta,module=self.engine)
            self.assertIsInstance(out,self.f.Refused)
        natural=self.h.sleeper(0,counter=3,origin=False)
        out=self.f.native_input_from_jax(natural,0,self.h.meta,module=self.engine)
        self.assertIsInstance(out,self.f.Refused);self.assertIn('SLEEP_ORIGIN_AND_LATENT_DURATION',{g.code for g in out.gaps})

    def test_native_stale_cured_rest_and_mixed_elapsed_fields_are_not_erased(self):
        s=self.h.sleeper(0,counter=2,bank=1);n=self.construct(s)
        for status,remaining,elapsed,refund in (('NONE',2,0,1),('SLEEP',2,1,0),('SLEEP',4,0,0),('SLEEP',2,0,4),('SLEEP',0,0,1)):
            parts=n.state.to_string().split('/');fields=parts[0].split('=');mon=fields[0].split(',')
            mon[18:21]=[status,str(remaining),str(elapsed)]
            mon=[x for x in mon if not x.startswith('refund:')]
            if refund:mon.append('refund:'+str(refund))
            fields[0]=','.join(mon);parts[0]='='.join(fields);raw='/'.join(parts)
            out=self.f.canonical_native_future(raw,n.certificate)
            self.assertIsInstance(out,self.f.Refused);self.assertEqual(out.raw_native,raw)
        # Native91 Lum clears status directly and leaves the Rest clocks.
        # Preserve that known reference limitation as refusal, never agreement.
        s=self.h.sleeper(0,counter=2,bank=1);s=s._replace(item=s.item.at[0,0,0].set(1))
        n=self.construct(s)
        branches=self.engine.generate_instructions(n.state,'splash','splash')
        for branch in branches:
            out=self.f.canonical_native_future(n.state.apply_instructions(branch),n.certificate)
            self.assertIsInstance(out,self.f.Refused)
            self.assertIn('NATIVE_LIVE_FUTURE_FIELD',{g.code for g in out.gaps})

    @unittest.skipUnless(importlib.util.find_spec('pokezero_search'),'requires conditioned native binding')
    def test_conditional_execution_guard_still_refuses_rest_sleep(self):
        import json
        import pokezero_search
        if not getattr(pokezero_search,'CONDITIONED_ORACLE_ENABLED',False):
            self.skipTest('requires conditioned native feature')
        from tests import test_conditioned_native_resume as cases
        cases.ConditionedResumeTests.setUpClass();h=cases.ConditionedResumeTests()
        for side in range(2):
            _,native,_,capsule,_=h.fixture(side,99)
            parts=native.to_string().split('/');fields=parts[side].split('=');mon=fields[0].split(',')
            mon[18:21]=['SLEEP','2','0'];fields[0]=','.join(mon);parts[side]='='.join(fields)
            with self.assertRaisesRegex(ValueError,'UNSUPPORTED_PARTY_CONTEXT'):
                pokezero_search.conditioned_resume('/'.join(parts),json.dumps(capsule),1,[0,1])

    def test_native_charge_callee_pool_discrepancy_remains_visible(self):
        # Native91 get_sleep_talk_choices filters only NONE/SLEEPTALK. The
        # pinned Gen3 simulator and JAX also exclude charge moves. Preserve
        # this separate mismatch rather than weaken JAX or call it agreement.
        s=self.h.sleeper(0);s=s._replace(moves=s.moves.at[0,0,0,3].set(4),pp=s.pp.at[0,0,0,3].set(10))
        n=self.construct(s);branches=self.engine.generate_instructions(n.state,'sleeptalk','splash')
        self.assertFalse(bool(self.h.table.sleep_talk.callable[4]))
        native_charge_mass=0
        jax=self.h.play(s,0,1)
        key=self.f.canonical_jax_future(jax,0,self.h.meta,certificate=n.certificate)
        self.assertIsInstance(key,self.f.CanonicalFuture)
        for branch in branches:
            nxt=n.state.apply_instructions(branch)
            raw=self.f._native_wire(nxt.to_string())
            if 'solarbeam' in raw['side_one']['volatile_statuses']:
                native_charge_mass+=branch.percentage/100
                other=self.f.canonical_native_future(nxt,n.certificate)
                self.assertIsInstance(other,self.f.CanonicalFuture)
                self.assertNotEqual(key.key,other.key)
        self.assertAlmostEqual(native_charge_mass,1/3,places=6)

    def test_hazard_ko_preserves_the_exact_rest_fields_and_refuses_unsupported_carry(self):
        for side in range(2):
            s=self.h.sleeper(side,counter=2,bank=1)
            s=self.h.play(s,side,4)
            s=s._replace(hp=s.hp.at[0,side,0].set(1),spikes=s.spikes.at[0,side].set(1))
            n=self.construct(s)
            names=['splash','splash'];names[side]='ditto'
            branches=self.engine.generate_instructions(n.state,*names)
            out=self.h.play(s,side,4)
            self.h.assert_sleep(out,side,5,3,0,True)
            for branch in branches:
                nxt=n.state.apply_instructions(branch)
                mon=(nxt.side_one if side==0 else nxt.side_two).pokemon[0]
                self.assertEqual((mon.hp,mon.status.lower(),mon.rest_turns,mon.sleep_turns,mon.rest_sleep_pending_refund),(0,'sleep',3,0,0))
                self.assertEqual(nxt.reverse_instructions(branch).to_string(),n.raw_native)
            # This is field-level carry evidence, not expansion of the existing
            # conditional/native replacement domain to sleeping entrants.
            mapped=self.f.native_input_from_jax(out,0,self.h.meta,module=self.engine)
            self.assertIsInstance(mapped,self.f.Refused)
            self.assertIn('LIVE_SUSPENSION_INTERFACE',{g.code for g in mapped.gaps})

    def test_whole_authored_ordinary_rest_laws_pass_both_bound_consumers(self):
        from collections import defaultdict
        from fractions import Fraction
        from gpu_engine import chance_consumers as c,finite_law
        from gpu_engine.fidelity_panel import JaxCaseExecutor,native_executor,require_future
        for side in range(2):
            for counter,action in ((3,1),(1,2)):
                state=self.h.sleeper(side,counter=counter)
                actions=tuple(int(x) for x in self.h.actions(side,action)[0])
                plan=c.plan_for_state(state,0,actions,self.h.table,self.h.meta,cap=8)
                self.assertIsInstance(plan,c.ConsumerPlan,getattr(plan,'gaps',plan))
                cases=tuple(plan.bounded_cases());self.assertEqual(sum(x.mass for x in cases),1)
                values=JaxCaseExecutor(state,actions,self.h.table,plan,batch_size=4)(cases)
                exact=defaultdict(Fraction)
                for case,value in zip(cases,values):exact[require_future(value).key]+=case.mass
                native=defaultdict(float)
                for probability,value in native_executor(plan):native[require_future(value).key]+=probability
                diff=finite_law.compare_laws(dict(exact),dict(native),mass_tolerance=1e-6)
                self.assertFalse(diff['exact_only']);self.assertFalse(diff['native_only'])
                self.assertLessEqual(diff['max_probability_difference'],1e-6)
