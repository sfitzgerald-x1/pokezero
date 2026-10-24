"""Fresh constructor/retention guards; no old root recovery or coverage credit."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest
from copy import deepcopy

from gpu_engine import future_state as f
from test_gpu_engine_future_state import action_fixture, FAKE

NODE=os.environ.get('NODE') or shutil.which('node')


class GenderCorrespondenceTests(unittest.TestCase):
    def live(self):
        data,meta=action_fixture()
        data['gender']=[[1,2,0,1,2,0],[2,1,0,2,1,0]]
        return data,meta

    def test_all12_constructor_genders_and_complete_future_keys(self):
        data,meta=self.live()
        native=f.native_input_from_jax(data,None,meta,module=FAKE)
        self.assertIsInstance(native,f.NativeInput,native)
        left=f.canonical_jax_future(data,None,meta,certificate=native.certificate)
        right=f.canonical_native_future(native.state,native.certificate)
        self.assertEqual(left.key,right.key)
        for side,name in enumerate(f.SIDE_NAMES):
            for p in range(6):
                expected={0:'none',1:'male',2:'female'}[data['gender'][side][p]]
                self.assertEqual(dict(left.key)[f'{name}.pokemon[{p}].gender'],expected)

    def test_unknown_is_not_genderless_and_sensitive_bench_inventory_stays_refused(self):
        data,meta=self.live();meta['vocabulary']['moves'].append('attract')
        data['moves'][1][5][0]=len(meta['vocabulary']['moves'])-1
        data['gender'][0][5]=-1
        result=f.native_input_from_jax(data,None,meta,module=FAKE)
        self.assertIn('GENDER_NOT_RETAINED',{g.code for g in result.gaps})
        data['gender'][0][5]=0
        self.assertIsInstance(f.native_input_from_jax(data,None,meta,module=FAKE),f.NativeInput)
        del data['gender']
        result=f.native_input_from_jax(data,None,meta,module=FAKE)
        self.assertIn('GENDER_NOT_RETAINED',{g.code for g in result.gaps})
        data,meta=self.live();meta['vocabulary']['abilities'].append('cutecharm')
        data['ability'][1][5]=len(meta['vocabulary']['abilities'])-1;data['gender'][0][0]=-1
        result=f.native_input_from_jax(data,None,meta,module=FAKE)
        self.assertIn('GENDER_NOT_RETAINED',{g.code for g in result.gaps})

    def test_invalid_shape_codes_and_future_gender_changes_refuse(self):
        data,meta=self.live()
        for bad in (3,-2,True):
            changed=deepcopy(data);changed['gender'][1][5]=bad
            result=f.native_input_from_jax(changed,None,meta,module=FAKE)
            self.assertIsInstance(result,f.Refused)
        changed=deepcopy(data);changed['gender']=[[1]]
        self.assertIsInstance(f.native_input_from_jax(changed,None,meta,module=FAKE),f.Refused)
        native=f.native_input_from_jax(data,None,meta,module=FAKE)
        changed=deepcopy(data);changed['gender'][1][5]=1
        result=f.canonical_jax_future(changed,None,meta,certificate=native.certificate)
        self.assertIn('RETAINED_GENDER_CHANGED',{g.code for g in result.gaps})
        native.state.side_two.pokemon[5].gender='male'
        result=f.canonical_native_future(native.state,native.certificate)
        self.assertIn('NATIVE_GENDER_CHANGED',{g.code for g in result.gaps})

    def test_unknown_to_genderless_change_not_hidden_by_static_cache(self):
        data,meta=self.live();data['gender'][0][0]=-1
        native=f.native_input_from_jax(data,None,meta,module=FAKE)
        data['gender'][0][0]=0
        result=f.canonical_jax_future(data,None,meta,certificate=native.certificate)
        self.assertIn('RETAINED_GENDER_CHANGED',{g.code for g in result.gaps})

    def test_known_genders_do_not_admit_unsupported_positive_attraction_future(self):
        data,meta=self.live();native=f.native_input_from_jax(data,None,meta,module=FAKE)
        native.state.side_one.volatile_statuses={'attract'}
        result=f.canonical_native_future(native.state,native.certificate)
        self.assertIn('NATIVE_UNREPRESENTED_VOLATILE',{g.code for g in result.gaps})
        from gpu_engine.fidelity_panel import require_future, WholeSlotUnknown
        with self.assertRaises(WholeSlotUnknown):require_future(result)


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class GenderInitializationTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import gender
        from gpu_engine.state import empty_state
        self.jax,self.j,self.g,self.empty=jax,j,gender,empty_state
        self.vocab=['_none','mixed','female','male','genderless','unresolved']
        self.policy=dict(schema=gender.SCHEMA,species=self.vocab,codes=[-1,3,2,1,0,-1])

    def live(self,batch=4):
        s=self.empty(batch,seed=121)
        return s._replace(species=self.j.ones_like(s.species))

    def initialize(self,s,roots=None,**options):
        defaults=dict(seed=17,domain=424242,root_ids=list(range(len(s.species))) if roots is None else roots)
        defaults.update(options)
        return self.g.initialize_genders(s,self.policy,self.vocab,**defaults)

    def test_legacy_unknown_and_fresh_fixed_or_mixed_preserve_every_prior_tensor(self):
        s=self.live();self.assertTrue(bool(self.j.all(s.gender==-1)))
        s=s._replace(species=s.species.at[:,:,1].set(2).at[:,:,2].set(3).at[:,:,3].set(4))
        out=self.initialize(s)
        self.assertEqual(out.gender[:,:,1].tolist(),[[2,2]]*4)
        self.assertEqual(out.gender[:,:,2].tolist(),[[1,1]]*4)
        self.assertEqual(out.gender[:,:,3].tolist(),[[0,0]]*4)
        self.assertTrue(bool(self.j.all((out.gender[:,:,0]==1)|(out.gender[:,:,0]==2))))
        for name in s._fields:
            if name!='gender':self.assertTrue(bool(self.j.array_equal(getattr(s,name),getattr(out,name))),name)
        with self.assertRaisesRegex(ValueError,'uninitialized'):self.initialize(out)

    def test_addressing_is_repeatable_batch_prefix_shard_and_order_stable(self):
        s=self.live(16);roots=list(range(100,116));all_=self.initialize(s,roots)
        self.assertTrue(bool(self.j.array_equal(all_.gender,self.initialize(s,roots).gender)))
        for a,b in ((0,3),(3,8),(8,16)):
            part=self.jax.tree.map(lambda v:v[a:b],s)
            self.assertTrue(bool(self.j.array_equal(all_.gender[a:b],self.initialize(part,roots[a:b]).gender)))
        reverse=self.jax.tree.map(lambda v:v[::-1],s)
        self.assertTrue(bool(self.j.array_equal(all_.gender[::-1],self.initialize(reverse,roots[::-1]).gender)))
        self.assertGreater(len({tuple(row.flatten().tolist()) for row in all_.gender}),1)
        self.assertFalse(bool(self.j.array_equal(all_.gender,self.initialize(s,roots,domain=424243).gender)))

    def test_explicit_constructor_overrides_win_and_unresolved_needs_override(self):
        s=self.live(1)._replace(species=self.live(1).species.at[0,0,0].set(3).at[0,1,5].set(5))
        with self.assertRaisesRegex(ValueError,'unresolved'):self.initialize(s)
        explicit=self.j.full_like(s.species,-1).at[0,0,0].set(2).at[0,1,5].set(0)
        result=self.initialize(s,explicit_genders=explicit)
        self.assertEqual(int(result.gender[0,0,0]),2)
        self.assertEqual(int(result.gender[0,1,5]),0)

    def test_bad_domains_ids_lookup_and_explicit_fields_fail_closed(self):
        s=self.live(2)
        for options in (dict(seed=True),dict(domain=-1),dict(root_ids=[1,1]),
                        dict(root_ids=[-1,2]),dict(root_ids=[2**32,2]),
                        dict(root_ids=[1]),dict(explicit_genders=[[[1]]]),
                        dict(explicit_genders=self.j.full_like(s.gender,3))):
            with self.subTest(options=options),self.assertRaises(ValueError):self.initialize(s,**options)
        for policy in ({},dict(self.policy,codes=[-1,True,2,1,0,-1]),
                       dict(self.policy,species=self.vocab[::-1])):
            with self.assertRaises(ValueError):self.g.initialize_genders(s,policy,self.vocab,
                seed=17,domain=424242,root_ids=[0,1])

    def test_collector_startup_and_prefix_preserve_legacy_fields_actions_and_keys(self):
        import numpy as np
        from gpu_engine import collect_fidelity_panel as collector
        from test_gpu_engine_entry import EntryTests
        fixture=EntryTests();fixture.setUp();template=fixture.live(1)
        pool={name:np.repeat(np.asarray(getattr(template,name)[0]),1000,axis=0)
              for name in ('species','level','types','stats','moves','pp','item','ability',
                           'hidden_power_type','hidden_power_power')}
        pool['species'][:]=1
        pool['ability'][:,0]=1  # Drizzle: an observable startup callback.
        # Authored source constructor only, not a real-team measurement.
        vocab=['_none','mixed']
        policy=dict(schema=self.g.SCHEMA,species=vocab,codes=[-1,3])
        entry=dict(slots=1000,team_start=0,prefix_cycle=2,seed=29,rng_domain=3123)
        legacy=collector.generate_inputs(pool,fixture.table,entry)
        fresh=collector.generate_inputs(pool,fixture.table,dict(entry,gender_initialization=dict(
            seed=17,domain=424242,root_start=8000,policy=policy)),species_vocabulary=vocab)
        for name in legacy[0]._fields:
            if name!='gender':np.testing.assert_array_equal(getattr(legacy[0],name),getattr(fresh[0],name),err_msg=name)
        for old,new in zip(legacy[1:],fresh[1:]):
            if isinstance(old,np.ndarray):np.testing.assert_array_equal(old,new)
            else:self.assertEqual(old,new)
        self.assertTrue(np.all(legacy[0].gender==-1))
        self.assertTrue(np.all(np.isin(fresh[0].gender,[1,2])))
        self.assertFalse(np.any(fresh[0].entry_pending))
        self.assertTrue(np.all(fresh[0].weather[::2]==fixture.w.RAIN))
        self.assertTrue(np.all(fresh[0].weather_counter[::2]==0))
        initial=fixture.t.state_from_pool(pool,np.arange(2000).reshape(1000,2),seed=29)
        expected=self.g.initialize_genders(initial,policy,vocab,seed=17,domain=424242,
            root_ids=list(range(8000,9000)))
        np.testing.assert_array_equal(expected.gender,fresh[0].gender)


@unittest.skipUnless(NODE,'requires Node for offline source table')
class GenderPolicyTests(unittest.TestCase):
    def test_fixed_form_proof_and_fair_mixed_law_ignore_gender_ratios(self):
        helper=Path(__file__).parents[1]/'gpu_engine/gender_policy.cjs'
        script="""const {buildPolicy:f}=require(process.argv[1]);
const rows={mixed:{exists:true,gen:1,genderRatio:{M:.875,F:.125}},
 female:{exists:true,gen:1,gender:'F'},male:{exists:true,gen:1,gender:'M'},
 none:{exists:true,gen:1,gender:'N'},form:{exists:true,gen:2,baseSpecies:'none',gender:'N'},
 bad:{exists:true,gen:2,baseSpecies:'male',gender:'F'}};
const dex={species:{get:id=>rows[id]}};
console.log(JSON.stringify(f(dex,['_none','mixed','female','male','none','form','bad'])));"""
        x=json.loads(subprocess.check_output([NODE,'-e',script,str(helper)],text=True))
        self.assertEqual(x['codes'],[-1,3,2,1,0,0,-1])
        self.assertEqual(x['unresolved'],['bad'])


if __name__=='__main__':unittest.main()
