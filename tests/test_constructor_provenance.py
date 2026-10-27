"""Authored constructor/provenance controls, not scientific roster coverage."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

AVAILABLE=all(importlib.util.find_spec(name) for name in ('jax','numpy','poke_engine','pokezero_search'))
if AVAILABLE:
    import pokezero_search
    AVAILABLE=bool(getattr(pokezero_search,'CONDITIONED_ORACLE_ENABLED',False))


@unittest.skipUnless(AVAILABLE,'requires isolated JAX and both native bindings')
class ConstructorProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        import numpy as np
        import poke_engine
        from gpu_engine import source_provenance as p, future_state as f, transition as t
        from gpu_engine.state import empty_state
        cls.jax,cls.j,cls.np,cls.engine,cls.p,cls.f,cls.t=jax,j,np,poke_engine,p,f,t
        cls.empty=staticmethod(empty_state)
        cls.runtime=p.attest_native_runtime()

    def fixture(self,replacing=0,*,generated=False,happiness=None,trace=True):
        j,np=self.j,self.np
        types=['_none','normal','fighting','flying','poison','ground','rock','bug','ghost',
            'steel','fire','water','grass','electric','psychic','ice','dragon','dark']
        names=['_none','splash','batonpass','thunderbolt','struggle']
        meta={'vocabulary':{'species':['_none','ditto'],'types':types,'moves':names,
            'abilities':['_none','static','trace'],'items':['_none','leftovers'],
            'natures':['_none','jolly','serious']},
            'moves':[None]+[{'id':name,'source_type':'Electric' if name=='thunderbolt' else 'Normal',
                'category':'Special' if name=='thunderbolt' else 'Status','flags':{'protect':1}}
                for name in names[1:]],
            'compiled_tree':{'sha256':'1'*64,'files':1},'source_commit_label':'f'*40}
        meta['type_chart']=[[1]*17 for _ in range(17)]
        for move in meta['moves'][1:]:
            move.update(base_power=95 if move['id']=='thunderbolt' else 50 if move['id']=='struggle' else 0,
                type=13 if move['id']=='thunderbolt' else 1,accuracy=True,priority=0)
        state=self.empty(1,seed=471)._replace(species=j.ones((1,2,6),j.int32),level=j.full((1,2,6),100),
            types=j.zeros((1,2,6,2),j.int32).at[...,0].set(1),
            hp=j.full((1,2,6),1000),max_hp=j.full((1,2,6),1000),
            stats=j.full((1,2,6,6),200).at[...,0].set(1000)
                .at[:,replacing,:,5].set(300).at[:,replacing,1,5].set(200).at[:,1-replacing,:,5].set(101),
            moves=j.ones((1,2,6,4),j.int32).at[:,replacing,0,0].set(2).at[:,1-replacing,0,0].set(3),
            pp=j.full((1,2,6,4),10),ability=j.zeros((1,2,6),j.int32)
                .at[:,replacing,0].set(2 if trace else 0).at[:,1-replacing,0].set(1),
            gender=j.full((1,2,6),1,j.int32).at[:,1,:].set(2),
            entry_pending=j.ones((1,2),bool),done=j.array([False]))
        state=state._replace(base_ability=state.ability)
        if happiness is not None:state=state._replace(happiness=j.full((1,2,6),happiness,j.int32))
        data=self.f._sample(state,0)
        arrays={name:np.asarray(data[name]) for name in self.p._POOL_STATE_FIELDS}
        arrays['nature']=np.asarray([[1]*6,[2]*6],np.uint8)
        arrays['evs']=np.tile(np.asarray([20,40,60,80,100,120],np.uint16),(2,6,1))
        if not generated:arrays['gender']=np.asarray(data['gender'],np.int8)
        if happiness is not None:arrays['happiness']=np.asarray(data['happiness'],np.uint8)
        arrays['metadata_utf8']=np.frombuffer(json.dumps(meta).encode(),np.uint8)
        tmp=tempfile.TemporaryDirectory(prefix='pokezero-authored-constructor-');self.addCleanup(tmp.cleanup)
        path=Path(tmp.name)/'pool.npz';np.savez_compressed(path,**arrays)
        pool=self.p.load_constructor_pool(path)
        root=Path(self.p.__file__).parent
        catalog={'schema':'pokezero.gen3-constructor-catalog.v1','species':['_none','ditto'],
            'weights':[0,4.0],'unresolved':[],'provenance':{'showdown_commit':'f'*40,'compiled_tree':meta['compiled_tree'],
                'pool_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'generator_sources':{name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                    for name in ('constructor_catalog.cjs','gender_policy.cjs')},'scope':'authored test catalog'}}
        policy=None
        if generated:
            from gpu_engine.gender import initialize_genders
            policy={'schema':'pokezero.gen3-gender-policy.v1','species':['_none','ditto'],
                'codes':[-1,3],'unresolved':[],'provenance':{
                    **{k:catalog['provenance'][k] for k in ('showdown_commit','compiled_tree','pool_sha256')},
                    'generator_sources':{name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                        for name in ('gender_policy.cjs','generate_teams.cjs')}}}
            state=initialize_genders(state._replace(gender=j.full((1,2,6),-1,j.int32)),
                policy,meta['vocabulary']['species'],seed=7,domain=13,root_ids=[42])
        context=self.p.constructor_context(pool,catalog,runtime=self.runtime,
            cohort_id='authored-constructor',gender_policy=policy)
        assignment={'kind':'explicit_pool'} if not generated else {
            'kind':'generated','policy_sha256':self.p._sha(policy),'seed':7,'domain':13,'root_id':42}
        return state,pool,context,catalog,assignment

    def capture(self,state,pool,context,assignment):
        return self.p.capture_before_entry(state,0,pool,(0,1),context=context,runtime=self.runtime,
            root_id=42,gender_assignment=assignment)

    def construct(self,state,source,context,**changes):
        args=dict(provenance=source,context=context,runtime=self.runtime,
            expected_root_id=42,expected_pool_sha256=context.pool_sha256,module=self.engine)
        args.update(changes)
        return self.p.native_input_with_provenance(state,0,context.metadata(),**args)

    def test_capture_is_not_a_correspondence_and_preserves_pool_nature_evs_and_gender(self):
        state,pool,context,_,assignment=self.fixture()
        before=self.f._sample(state,0)
        source=self.capture(state,pool,context,assignment)
        self.assertIsInstance(source,self.p.ConstructorProvenance,source)
        self.assertNotIsInstance(source,self.f.Certificate)
        self.assertEqual(self.f._sample(state,0),before)
        self.assertIsInstance(self.construct(state,source,context),self.f.Refused)
        self.assertTrue(all(before['entry_pending']))
        self.assertNotEqual(context.pool_metadata_sha256,context.mapping_metadata_sha256)
        static=dict(source.static_native)
        self.assertEqual(static['side_one.pokemon[0]'][:4],('jolly',(20,40,60,80,100,120),'male',4.0))
        self.assertEqual(static['side_two.pokemon[0]'][0],'serious')
        self.assertFalse(pool.arrays['nature'].flags.writeable)
        with self.assertRaises(ValueError):pool.arrays['nature'].setflags(write=True)
        with self.assertRaises(TypeError):pool.arrays['nature']=None

    def test_pre_entry_history_cannot_be_fabricated_from_a_later_action(self):
        state,pool,context,_,assignment=self.fixture()
        for kind,index in ((1,0),(2,0),(3,-1),(0,0)):
            changed=state._replace(last_action_kind=state.last_action_kind.at[0,0].set(kind),
                last_action_index=state.last_action_index.at[0,0].set(index))
            refused=self.capture(changed,pool,context,assignment)
            self.assertIsInstance(refused,self.f.Refused)
            self.assertIn('PRE_ENTRY_ACTION_HISTORY_MUST_BE_NONE',repr(refused.gaps))

    def native_reference(self,state,pool,replacing):
        data=self.f._sample(state,0);meta=pool.metadata();sides=[]
        for side in range(2):
            party=[]
            for p in range(6):
                ability=meta['vocabulary']['abilities'][data['ability'][side][p]]
                if ability=='_none':ability='none'
                party.append(self.engine.Pokemon(id='ditto',level=100,types=('normal','typeless'),
                    base_types=('normal','typeless'),hp=1000,maxhp=1000,
                    ability=ability,base_ability=ability,item='none',
                    nature=meta['vocabulary']['natures'][int(pool.arrays['nature'][side,p])],
                    evs=tuple(map(int,pool.arrays['evs'][side,p])),gender='male' if side==0 else 'female',
                    terastallized=False,tera_type='normal',
                    weight_kg=4.0,attack=200,defense=200,special_attack=200,special_defense=200,
                    speed=data['stats'][side][p][5],
                    moves=[self.engine.Move(id=meta['vocabulary']['moves'][m],pp=10) for m in data['moves'][side][p]]))
            sides.append(self.engine.Side(pokemon=party,active_index='1' if side==replacing else '0'))
        native=self.engine.State(side_one=sides[0],side_two=sides[1],weather='none',weather_turns_remaining=0,
            terrain='none',terrain_turns_remaining=0,trick_room=False,trick_room_turns_remaining=0,team_preview=False)
        branches=self.engine.generate_instructions(native,'ditto' if replacing==0 else 'none',
            'none' if replacing==0 else 'ditto')
        self.assertEqual(len(branches),1)
        return native.apply_instructions(branches[0])

    def test_native_trace_copy_preserves_original_base_through_departure(self):
        from gpu_engine import baton,entry,items,secondary,conditioned_future as c
        for replacing in (0,1):
            state,pool,context,_,assignment=self.fixture(replacing)
            source=self.capture(state,pool,context,assignment)
            self.assertIsInstance(source,self.p.ConstructorProvenance,source)
            reference=self.native_reference(state,pool,replacing)
            wire=self.f._native_wire(reference.to_string())
            actor=wire[self.f.SIDE_NAMES[replacing]]['pokemon'][0]
            self.assertEqual((actor['ability'],actor['base_ability']),('static','trace'))
            post=state._replace(entry_pending=self.j.zeros((1,2),bool),
                ability=state.ability.at[:,replacing,0].set(1))
            mapped=self.construct(post,source,context)
            self.assertIsInstance(mapped,self.f.NativeInput,mapped)
            left=self.f.canonical_native_future(mapped.state,mapped.certificate)
            right=self.f.canonical_native_future(reference,mapped.certificate)
            self.assertEqual(self.f.field_differences(left,right),())
            meta=context.metadata();j=self.j
            table=self.t.MoveTable(j.array([0,0,0,95,50]),j.array([0,1,1,13,1]),
                j.zeros(5,bool),j.full(5,100),j.array([True,True,True,False,False]),
                j.zeros(5,j.int32),j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(4),
                baton=baton.from_metadata(meta),entry=entry.from_metadata(meta),
                items=items.from_metadata(meta),secondary=secondary.from_metadata(meta))
            chance=self.t.ChanceInputs(j.zeros((1,2,12),j.int32),j.full((1,2),59),
                j.full((1,2),99),j.full((1,2),59),j.array([[0,1]]),j.array([0]))
            paused=self.jax.jit(self.t.step_with_chance)(post,j.zeros((1,2),j.int32),table,chance).state
            bound=c.suspended_input_from_jax(paused,0,meta,certificate=mapped.certificate,module=self.engine)
            self.assertIsInstance(bound,c.SuspendedNativeInput,bound)
            branches=c.native_successors(bound,1,[0,1])
            self.assertNotIsInstance(branches,self.f.Refused,branches)
            self.assertEqual(len(branches),1)
            actions=j.array([[4,0] if replacing==0 else [0,4]])
            resumed=self.jax.jit(self.t.step_with_chance)(paused,actions,table,chance).state
            actual=c.canonical_resumed_jax(resumed,0,meta,native_input=bound)
            self.assertNotIsInstance(actual,self.f.Refused,actual)
            self.assertEqual(self.f.field_differences(actual,branches[0][1]),())

    def test_original_ability_cannot_be_rewritten_before_or_after_entry(self):
        state,pool,context,_,assignment=self.fixture()
        changed=state._replace(base_ability=state.base_ability.at[:,0,0].set(0))
        refused=self.capture(changed,pool,context,assignment)
        self.assertIsInstance(refused,self.f.Refused)
        self.assertIn('CONSTRUCTOR_BASE_ABILITY_CHANGED',repr(refused.gaps))
        source=self.capture(state,pool,context,assignment)
        post=state._replace(entry_pending=self.j.zeros((1,2),bool))
        changed=post._replace(base_ability=post.base_ability.at[:,0,0].set(0))
        refused=self.construct(changed,source,context)
        self.assertIsInstance(refused,self.f.Refused)
        self.assertIn('IMMUTABLE_CONSTRUCTOR_IDENTITY_CHANGED',repr(refused.gaps))

    def test_missing_tampered_and_wrong_root_provenance_never_make_a_certificate(self):
        state,pool,context,_,assignment=self.fixture();source=self.capture(state,pool,context,assignment)
        post=state._replace(entry_pending=self.j.zeros((1,2),bool))
        for bad in (None,replace(source,static_native=()),replace(source,root_id=43)):
            self.assertIsInstance(self.construct(post,bad,context),self.f.Refused)
        self.assertIsInstance(self.construct(post,source,context,expected_root_id=43),self.f.Refused)
        self.assertIsInstance(self.construct(post,source,context,expected_pool_sha256='0'*64),self.f.Refused)
        changed=post._replace(species=post.species.at[:,0,1].set(0))
        self.assertIsInstance(self.construct(changed,source,context),self.f.Refused)
        changed=post._replace(gender=post.gender.at[:,0,1].set(2))
        self.assertIsInstance(self.construct(changed,source,context),self.f.Refused)
        copied=state._replace(ability=state.ability.at[:,0,0].set(1))
        self.assertIsInstance(self.capture(copied,pool,context,assignment),self.f.Refused)

    def test_generated_gender_record_retains_policy_seed_domain_and_exact_root(self):
        state,pool,context,catalog,assignment=self.fixture(generated=True)
        source=self.capture(state,pool,context,assignment)
        self.assertIsInstance(source,self.p.ConstructorProvenance,source)
        self.assertEqual(dict(source.gender_assignment),assignment)
        for key in ('policy_sha256','seed','domain','root_id'):
            bad=dict(assignment);bad.pop(key)
            self.assertIsInstance(self.capture(state,pool,context,bad),self.f.Refused)
        bad=dict(assignment);bad['policy_sha256']='2'*64
        self.assertIsInstance(self.capture(state,pool,context,bad),self.f.Refused)
        changed=state._replace(gender=state.gender.at[:,0,0].set(3-state.gender[0,0,0]))
        self.assertIsInstance(self.capture(changed,pool,context,assignment),self.f.Refused)
        missing=self.p.constructor_context(pool,catalog,runtime=self.runtime,cohort_id='authored-constructor')
        self.assertIsInstance(self.capture(state,pool,missing,assignment),self.f.Refused)
        policy=json.loads(context.gender_policy_json);policy['provenance']['pool_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'GENDER_POLICY_SOURCE_MISMATCH'):
            self.p.constructor_context(pool,catalog,runtime=self.runtime,
                cohort_id='authored-constructor',gender_policy=policy)

    def test_explicit_happiness_is_retained_and_default_is_bound(self):
        for happiness in (None,0,123):
            state,pool,context,_,assignment=self.fixture(happiness=happiness)
            source=self.capture(state,pool,context,assignment)
            self.assertIsInstance(source,self.p.ConstructorProvenance,source)
            expected=255 if happiness is None else happiness
            self.assertEqual(dict(source.pool_constructor_fields)['happiness'],((expected,)*6,)*2)
            rebuilt=self.t.state_from_pool(pool.arrays,[[0,1]])
            self.assertEqual(self.f._sample(rebuilt,0)['happiness'],((expected,)*6,)*2)
            changed=state._replace(happiness=state.happiness.at[:,0,0].set((expected+1)%256))
            self.assertIsInstance(self.capture(changed,pool,context,assignment),self.f.Refused)

    def test_catalog_and_both_native_consumers_are_bound_before_context_creation(self):
        state,pool,context,catalog,assignment=self.fixture()
        payload=json.loads(self.runtime.payload_json)
        self.assertEqual(set(payload['consumers']),{'poke_engine','pokezero_search'})
        self.assertTrue(all(len(v['binary_sha256'])==64 for v in payload['consumers'].values()))
        for key in ('poke_engine','pokezero_search'):
            bad=deepcopy(payload);bad['consumers'][key]['binary_sha256']='0'*64
            forged=self.p.RuntimeAttestation(self.p._json(bad),self.p._sha(bad))
            with self.assertRaisesRegex(ValueError,'NATIVE_RUNTIME_IDENTITY_CHANGED'):
                self.p.constructor_context(pool,catalog,runtime=forged,cohort_id='authored-constructor')
        for field in ('pool_sha256','showdown_commit'):
            bad=deepcopy(catalog);bad['provenance'][field]='0'*len(bad['provenance'][field])
            with self.assertRaises(ValueError):
                self.p.constructor_context(pool,bad,runtime=self.runtime,cohort_id='authored-constructor')

    def test_catalog_generator_preserves_unresolved_weights_for_whole_slot_refusal(self):
        node=shutil.which('node')
        self.assertIsNotNone(node,'Node is required for constructor-catalog source checks')
        catalog=Path(self.p.__file__).with_name('constructor_catalog.cjs')
        script="""const {buildCatalog}=require(process.argv[1]);
const dex={species:{get:id=>({exists:id==='ditto',gen:1,weightkg:4})}};
const good=buildCatalog(dex,['_none','ditto']);if(good.weights[1]!==4)process.exit(2);
for(const weight of [0,-1,NaN]){const bad=buildCatalog({species:{get:()=>({exists:true,gen:1,weightkg:weight})}},['_none','ditto']);if(bad.weights[1]!==null||bad.unresolved[0]!=='ditto')process.exit(3);}
const missing=buildCatalog(dex,['_none','missing']);if(missing.weights[1]!==null||missing.unresolved[0]!=='missing')process.exit(4);
"""
        result=subprocess.run([node,'-e',script,str(catalog)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        state,pool,context,catalog,assignment=self.fixture()
        missing=deepcopy(catalog);missing['weights'][1]=None;missing['unresolved']=['ditto']
        unknown=self.p.constructor_context(pool,missing,runtime=self.runtime,cohort_id='authored-constructor')
        value=self.capture(state,pool,unknown,assignment)
        self.assertIsInstance(value,self.f.Refused,value)
        self.assertIn('ORIGINAL_WEIGHT_UNRESOLVED',repr(value.gaps))


if __name__=='__main__':unittest.main()
