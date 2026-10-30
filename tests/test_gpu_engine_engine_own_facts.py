"""Literal source/constructor packets into retained own writers, no games/models."""
from dataclasses import replace, fields, is_dataclass
from collections.abc import Mapping
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_own_facts as own, engine_request_source as source
from gpu_engine import v4_encoder, public_belief, public_inputs, public_wire
from gpu_engine import engine_public_events as events, source_provenance, transition
from gpu_engine import private_actions, private_team
import test_gpu_engine_request_source as request_fixture
import test_gpu_engine_v4_encoder as v4_fixture
from pokezero.category_vocab import CategoryVocabulary
from pokezero.dex import ShowdownDex, normalize_id
from pokezero.randbat import Gen3RandbatSource, Gen3RandbatSpeciesUniverse, Gen3RandbatVariant, RandbatSourceMetadata


class DeviceOwnFactsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        request_fixture.RequestSourceTests.setUpClass()
        v4_fixture.V4EncoderTests.setUpClass()
        cls.f=request_fixture.RequestSourceTests()
        base=v4_fixture.V4EncoderTests
        cls.metadata=cls.f.metadata
        cls.dex=ShowdownDex(dict(base.inputdex.moves,**cls.f.dex.moves),
            dict(base.inputdex.species,**cls.f.dex.species),base.inputdex.type_chart)
        labels=set(base.vocab.tokens)|set(cls.f.vocab.tokens)
        for ability in cls.metadata['vocabulary']['abilities'][1:]:
            labels|={'ability:'+ability,'belief:possible_ability:'+ability}
        cls.vocab=CategoryVocabulary(tuple(sorted(labels)))
        names=cls.metadata['vocabulary']['species'][1:]
        abilities=cls.metadata['vocabulary']['abilities'][1:]
        variants={name:Gen3RandbatSpeciesUniverse(name,80,(Gen3RandbatVariant(
            'literal-'+name,'fixture',name,'fixture',80,('Surf','Hidden Power Ice'),ability,'Leftovers'),))
            for name,ability in zip(names,abilities)}
        literal=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-own-source'),variants)
        cls.host=public_belief.compile_source(literal)
        cls.catalog=v4_encoder.compile_encoder(cls.host,cls.dex,cls.vocab,
            checkpoint_sha256='a'*64,feature_masks=base.masks)
        cls.table=transition.tables_from_metadata(cls.metadata)
        cls.lifecycle=events.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,
            dex=cls.dex,exposure=events.HealthExposure(source.SHOWDOWN,'gen3randombattle',True,False))
        cls.rules=source.load_source_rules(cls.f.root,cls.metadata,node=cls.f.node)
        cls.request=source.compile_request_source(cls.metadata,vocabulary=cls.vocab,dex=cls.dex,
            checkpoint_sha256='a'*64,rules=cls.rules,lifecycle_catalog=cls.lifecycle)
        cls.compiled=own.compile_own_facts(cls.request,cls.catalog)
        # Exactly six distinct literal members per side, not a generated roster.
        species=np.array([[1,2,3,4,5,6],[3,4,5,6,1,2]],np.int32)
        cls.arrays={name:np.array(value[:2],copy=True) for name,value in cls.f.pool.arrays.items()}
        cls.arrays['species']=species
        cls.arrays['level']=np.full((2,6),80,np.int32)
        cls.arrays['ability']=species.copy()
        cls.arrays['gender']=np.array([[1,2,0,1,2,0],[2,0,1,2,0,1]],np.int32)
        cls.arrays['item']=np.ones((2,6),np.int32)
        cls.arrays['stats']=np.broadcast_to(np.array([201,151,163,175,187,199],np.int32),(2,6,6)).copy()
        cls.arrays['stats']+=np.arange(6,dtype=np.int32)[None,:,None]
        cls.arrays['moves']=np.broadcast_to(np.array([1,2,0,0],np.int32),(2,6,4)).copy()
        cls.arrays['pp']=np.broadcast_to(np.array([24,16,0,0],np.int32),(2,6,4)).copy()
        cls.arrays['hidden_power_type']=np.full((2,6),15,np.int32)
        cls.arrays['hidden_power_power']=np.full((2,6),70,np.int32)
        cls.arrays['types']=np.zeros((2,6,2),np.int32)
        for team in range(2):
            for slot,sid in enumerate(species[team]):
                for col,t in enumerate(cls.dex.species_info(names[sid-1]).types):
                    cls.arrays['types'][team,slot,col]=cls.metadata['vocabulary']['types'].index(t.lower())
        cls.pool=cls.make_pool(cls.arrays)
        cls.start=cls.construct()

    @classmethod
    def make_pool(cls,arrays):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'literal.npz'
            np.savez_compressed(path,**dict(arrays,metadata_utf8=np.frombuffer(json.dumps(cls.metadata).encode(),np.uint8)))
            return source_provenance.load_constructor_pool(path)

    @classmethod
    def construct(cls,pool=None):
        return source.construct_from_pool(pool or cls.pool,np.array([[0,1],[0,1]],np.int64),
            compiled=cls.request,stream_ids=j.array([71,72],j.int32),seed=31)

    def component(self,snapshot,state):
        # A tiny state component still runs the actual source packet producer;
        # no caller-attested SourceRequest is admitted by this adapter.
        requests=tuple(source.produce_request(self.request,state,ledger,reg,snapshot.registry,
            lifecycle_admitted=snapshot.cursor.admitted) for ledger,reg in zip(snapshot.orders,snapshot.registrations))
        return source._bind_constructed(state,snapshot.cursor,snapshot.orders,snapshot.registrations,
            requests,snapshot.callbacks,snapshot.registry)

    def raw(self,snapshot,owners):
        v=self.metadata['vocabulary'];inputs=[]
        status_names=('','par','brn','psn','tox','slp','frz')
        state=snapshot.state
        for row in range(2):
            sides=[]
            forced=np.asarray(state.forced_switch[row])|(np.asarray(state.hp[row])[np.arange(2),np.asarray(state.active[row])]<=0)
            for side in range(2):
                party=[];reg=snapshot.registrations[side]
                for slot in snapshot.orders[side].request_to_physical[row].tolist():
                    if slot<0:continue
                    sid=int(state.species[row,side,slot])
                    ppmax=reg.constructor.device.max_pp[row,slot]
                    moves=[dict(id=v['moves'][mid],pp=int(state.pp[row,side,slot,k]),maxpp=int(ppmax[k]))
                        for k,mid in enumerate(state.moves[row,side,slot].tolist()) if mid]
                    stat=dict(zip(('hp','atk','def','spa','spd','spe'),state.stats[row,side,slot].tolist()))
                    party.append(dict(species=v['species'][sid],level=int(state.level[row,side,slot]),
                        gender=int(state.gender[row,side,slot]),hp=int(state.hp[row,side,slot]),
                        maximum=int(state.max_hp[row,side,slot]),status=status_names[int(state.status[row,side,slot])]
                            if int(state.hp[row,side,slot]) else '',
                        types=[v['types'][t].title() for t in state.types[row,side,slot].tolist() if t],
                        ability=v['abilities'][int(state.ability[row,side,slot])],
                        base_ability=v['abilities'][int(state.base_ability[row,side,slot])],
                        item=v['items'][int(state.item[row,side,slot])],stats=stat,moves=moves,
                        hp_type=v['types'][int(state.hidden_power_type[row,side,slot])].title(),
                        rooted=False,linked=False,visible_trap=False,choice=''))
                sides.append(dict(party=party,forced=bool(forced[side])))
            inputs.append(dict(sides=sides,replacement=bool(forced.any()),
                public_species=[v['species'][sid] for sid in snapshot.registry.revealed_species_id[row].tolist()]))
        raw=json.loads(subprocess.check_output([self.f.node,str(Path(__file__).with_name('gpu_engine_own_facts_oracle.cjs')),str(self.f.root)],
            input=json.dumps(inputs),text=True))
        return [raw[row][side] for row,side in enumerate(owners.tolist())]

    def same(self,left,right):
        for actual,expected in zip(jax.tree.leaves(left),jax.tree.leaves(right)):
            self.assertEqual(actual.dtype,expected.dtype)
            np.testing.assert_array_equal(actual,expected)

    def compare(self,snapshot,owners=j.array([0,1],j.int32)):
        result=own.produce_own_facts(snapshot,self.compiled,owner_sides=owners)
        self.assertEqual(result.admitted.tolist(),[True,True],result.reason.tolist())
        raw=self.raw(snapshot,owners)
        expected=v4_encoder.parse_own_requests(self.catalog,raw,owner_sides=owners.tolist())
        self.same(result.facts,expected)
        # Existing meaningful retained own and action features, not a new writer.
        for projector,arguments in ((private_team.project_team,dict(expected_vocabulary_binding=self.catalog.species.device.vocabulary_binding)),
                                    (private_actions.project_actions,{})):
            if projector is private_team.project_team:
                left=projector(result.facts.actions.team,observer_side=owners,**arguments)
                right=projector(expected.actions.team,observer_side=owners,**arguments)
            else:
                left=projector(result.facts.actions,self.catalog.actions.device,observer_side=owners,**arguments)
                right=projector(expected.actions,self.catalog.actions.device,observer_side=owners,**arguments)
            self.same(left,right);self.assertTrue(bool(left.admitted.all()));self.assertTrue(bool(left.numeric.any()))
        return result,expected

    def test_actual_constructor_every_own_leaf_and_retained_writer_both_owners_full_bench(self):
        out,_=self.compare(self.start)
        self.assertEqual(out.facts.status.present.tolist(),[[True]*6]*2)
        self.assertEqual(out.facts.targets.known[...,0].tolist(),[[True]*6]*2)
        self.assertEqual(self.compiled.device.source_stat_keys.tolist(),[False,True,True,True,True,True])
        self.assertNotEqual(self.compiled.device.species.tolist(),list(range(7)))
        self.assertNotEqual(self.compiled.device.ability.tolist(),list(range(7)))
        self.assertEqual(out.facts.hazard.levitate.sum(axis=-1).tolist(),[1,1])

    def test_actual_installed_switch_and_reentry_request_order_both_owners(self):
        installed=source.advance_source(self.start,j.full((2,2),4,j.int32),compiled=self.request)
        self.assertEqual(installed.registry.generation.tolist(),[[1,1]]*2)
        self.compare(installed)
        returned=source.advance_source(installed,j.full((2,2),4,j.int32),compiled=self.request)
        self.assertEqual(returned.registry.generation.tolist(),[[2,2]]*2)
        self.compare(returned)

    def test_actual_short_party_padding_and_source_force_wait_both_owners(self):
        arrays={name:np.array(value,copy=True) for name,value in self.arrays.items()}
        for value in arrays.values():
            if value.shape[:2]==(2,6):
                value[0,3:]=0;value[1,2:]=0
        short=self.construct(self.make_pool(arrays))
        out,_=self.compare(short)
        self.assertEqual(out.facts.status.present.sum(axis=-1).tolist(),[3,2])
        self.assertFalse(bool(out.facts.targets.known[:,3:].any()))
        state=self.start.state._replace(forced_switch=self.start.state.forced_switch.at[:,0].set(True))
        out,_=self.compare(self.component(self.start,state))
        self.assertEqual(out.facts.actions.request.force_switch.tolist(),[True,False])
        self.assertEqual(out.facts.actions.request.wait.tolist(),[False,True])

    def test_status_faint_hp_derivation_and_gen3_base_ability_not_current(self):
        state=self.start.state
        state=state._replace(status=state.status.at[:,0,1].set(5).at[:,1,1].set(4)
            .at[:,0,2].set(999).at[:,1,2].set(999),
            hp=state.hp.at[:,0,2].set(0).at[:,1,2].set(0),
            ability=state.ability.at[:,0,0].set(2).at[:,1,0].set(2))
        out,_=self.compare(self.component(self.start,state))
        self.assertEqual(out.facts.status.status[:,2].tolist(),[0,0])
        self.assertEqual(out.facts.targets.known[:,2,0].tolist(),[False,False])
        self.assertEqual(out.facts.targets.stats[:,2,0].tolist(),[0,0])
        self.assertEqual(out.facts.exact_abilities.ability[:,0].tolist(),[
            self.catalog.abilities.ability_id('Static'),self.catalog.abilities.ability_id('Immunity')])
        self.assertEqual(out.facts.hazard.levitate[:,0].tolist(),[False,False])

    def test_existing_v4_writer_composition_uses_actual_own_packet(self):
        helper=v4_fixture.V4EncoderTests();helper.catalog=self.catalog
        lines=['|gametype|singles','|gen|3','|tier|[Gen 3] Random Battle',
            '|switch|p1a: pikachu|pikachu, L80, M|201/201',
            '|switch|p2a: snorlax|snorlax, L80, F|201/201',
            '|-sidestart|p1: literal|Spikes','|-sidestart|p2: literal|Spikes','|turn|1']
        streams=j.array([71,72],j.int32)
        state=v4_encoder.initialize(self.catalog,stream_ids=streams)
        for chunk in helper.chunks(helper.projections(lines),batch=2):
            state,error=v4_encoder.fold_chunk(state,chunk._replace(stream_id=streams),self.catalog)
            self.assertEqual(error.tolist(),[0,0])
        owners=j.array([0,1],j.int32)
        actual,expected=self.compare(self.start,owners)
        encoded=own.encode_source(state,self.start,self.compiled,owner_sides=owners)
        reference=v4_encoder.encode(state,expected,self.catalog,observer_sides=owners)
        self.same(encoded,reference)
        self.assertEqual(encoded.observation.admitted.tolist(),[True,True])
        self.assertTrue(bool(encoded.observation.numeric_features.any()))
        self.assertTrue(bool(encoded.observation.categorical_ids.any()))
        self.assertEqual(encoded.full_encoder_admitted.tolist(),[False,False])
        mismatched=state._replace(cursor=state.cursor._replace(stream_id=streams.at[0].set(171)))
        refused=own.encode_source(mismatched,self.start,self.compiled,owner_sides=owners)
        self.assertEqual(refused.observation.admitted.tolist(),[False,True])
        for field in ('numeric_features','categorical_ids','legal_action_mask','attention_mask'):
            self.assertFalse(bool(j.any(getattr(refused.observation,field)[0])))
            np.testing.assert_array_equal(getattr(refused.observation,field)[1],getattr(encoded.observation,field)[1])

    def atomic(self,result,row):
        self.assertFalse(bool(result.admitted[row]))
        for value in jax.tree.leaves(result.facts):
            self.assertFalse(bool(j.any(value[row])))

    def test_invalid_owner_and_mixed_source_refusal_zero_every_leaf(self):
        out=own.produce_own_facts(self.start,self.compiled,owner_sides=j.array([-1,1],j.int32))
        self.atomic(out,0);self.assertTrue(bool(out.admitted[1]))
        self.same(jax.tree.map(lambda x:x[1],out.facts),jax.tree.map(lambda x:x[1],
            own.produce_own_facts(self.start,self.compiled,owner_sides=j.array([0,1],j.int32)).facts))
        bad=self.start.state._replace(charge_move=self.start.state.charge_move.at[0,0,0].set(1))
        out=own.produce_own_facts(self.component(self.start,bad),self.compiled,owner_sides=j.array([0,1],j.int32))
        self.atomic(out,0);self.assertTrue(bool(out.admitted[1]))
        for owners in (j.array([0.,1.]),j.zeros((2,1),j.int32)):
            with self.assertRaises(ValueError):own.produce_own_facts(self.start,self.compiled,owner_sides=owners)

    def test_missing_gender_and_changed_constructor_origins_refuse(self):
        for name,value in (('gender',-1),('stats',999),('species',999)):
            state=self.start.state
            state=state._replace(**{name:getattr(state,name).at[0,0,1].set(value)})
            out=own.produce_own_facts(self.component(self.start,state),self.compiled,owner_sides=j.array([0,1],j.int32))
            self.atomic(out,0);self.assertTrue(bool(out.admitted[1]))
        arrays=dict(self.arrays,gender=self.arrays['gender'].copy())
        arrays['gender'][0,1]=-1
        absent_bench=self.construct(self.make_pool(arrays))
        out=own.produce_own_facts(absent_bench,self.compiled,owner_sides=j.array([0,1],j.int32))
        self.atomic(out,0);self.assertTrue(bool(out.admitted[1]))
        arrays=dict(self.arrays);arrays.pop('gender')
        absent=self.construct(self.make_pool(arrays))
        out=own.produce_own_facts(absent,self.compiled,owner_sides=j.array([0,1],j.int32))
        self.atomic(out,0);self.atomic(out,1)
        transformed=self.component(self.start,self.start.state._replace(
            last_move=self.start.state.last_move.at[0,0,0].set(7)))
        out=own.produce_own_facts(transformed,self.compiled,owner_sides=j.array([0,1],j.int32))
        self.atomic(out,0);self.assertTrue(bool(out.admitted[1]))

    def test_owner_selection_precedes_off_owner_private_validation(self):
        original=own.produce_own_facts(self.start,self.compiled,owner_sides=j.array([0,1],j.int32))
        state=self.start.state
        changes={}
        for name in ('stats','gender','status','base_ability','item','hp','max_hp'):
            array=getattr(state,name)
            changes[name]=array.at[0,1,3:].set(-999).at[1,0,3:].set(999999)
        # Component packet producer refuses the OTHER owner rows, while the
        # selected owner's actual source packet remains admitted and unchanged.
        changed=self.component(self.start,state._replace(**changes))
        out=own.produce_own_facts(changed,self.compiled,owner_sides=j.array([0,1],j.int32))
        self.same(out,original)

    def test_static_holder_and_all_catalog_array_edges_checked_after_hot_jit(self):
        self.compare(self.start)

    def test_every_own_static_leaf_and_nested_v4_holder_refuses_replacement(self):
        owners=j.array([0,1],j.int32)
        own.produce_own_facts(self.start,self.compiled,owner_sides=owners)
        original=self.compiled.device
        try:
            for key,value in zip(original._fields,original):
                for readonly in (False,True):
                    foreign=np.asarray(value).copy();foreign.setflags(write=not readonly)
                    object.__setattr__(self.compiled,'device',original._replace(**{key:foreign}))
                    with self.subTest(leaf=key,readonly=readonly):
                        with self.assertRaises(ValueError):own.produce_own_facts(self.start,self.compiled,owner_sides=owners)
        finally:object.__setattr__(self.compiled,'device',original)
        # Replacing any nested dataclass table bypasses a root identity-only
        # guard. Exercise every such holder in the original v4 catalog graph.
        visited=set();holders=[]
        def visit(value):
            if id(value) in visited:return
            visited.add(id(value))
            if is_dataclass(value):
                for f in fields(value):
                    child=getattr(value,f.name)
                    if isinstance(child,jax.Array) or isinstance(child,tuple) and any(
                            isinstance(x,jax.Array) for x in jax.tree.leaves(child)):
                        holders.append((value,f.name,child))
                    visit(child)
            elif isinstance(value,Mapping):
                for child in value.values():visit(child)
            elif isinstance(value,(tuple,list)):
                for child in value:visit(child)
        visit(self.catalog)
        self.assertGreater(len(holders),10)
        for holder,key,value in holders:
            replacement=j.array(value) if isinstance(value,jax.Array) else jax.tree.map(lambda x:
                j.array(x) if isinstance(x,jax.Array) else x,value)
            try:
                object.__setattr__(holder,key,replacement)
                with self.subTest(holder=type(holder).__name__,key=key):
                    with self.assertRaises(ValueError):own.produce_own_facts(self.start,self.compiled,owner_sides=owners)
            finally:object.__setattr__(holder,key,value)
        self.assertTrue(bool(own.produce_own_facts(self.start,self.compiled,owner_sides=owners).admitted.all()))
        for holder in (replace(self.compiled),):
            with self.assertRaises(ValueError):own.produce_own_facts(self.start,holder,owner_sides=j.array([0,1],j.int32))
        with self.assertRaises(ValueError):own.compile_own_facts(self.request,replace(self.catalog))
        original=self.catalog.actions.device
        try:
            for foreign in (np.asarray(original.numeric).copy(),j.array(original.numeric)):
                object.__setattr__(self.catalog.actions,'device',original._replace(numeric=foreign))
                with self.assertRaises(ValueError):own.produce_own_facts(self.start,self.compiled,owner_sides=j.array([0,1],j.int32))
        finally:object.__setattr__(self.catalog.actions,'device',original)
        with self.assertRaises(ValueError):
            jax.jit(lambda:own.produce_own_facts(self.start,self.compiled,owner_sides=j.array([0,1],j.int32)))()
        self.compare(self.start)

    def test_source_snapshot_copy_static_namespace_and_cached_boundary_no_host_reads(self):
        with self.assertRaises(ValueError):own.produce_own_facts(replace(self.start),self.compiled,owner_sides=j.array([0,1],j.int32))
        other=v4_encoder.compile_encoder(self.host,self.dex,self.vocab,checkpoint_sha256='b'*64,
            feature_masks=v4_fixture.V4EncoderTests.masks)
        with self.assertRaises(ValueError):own.compile_own_facts(self.request,other)
        # Warm first. Runtime checks use retained identities; no np conversion,
        # string request parser or device_get may be called on the checked path.
        owners=j.array([0,1],j.int32)
        expected=own.produce_own_facts(self.start,self.compiled,owner_sides=owners)
        original_asarray=np.asarray
        def no_device_array(value,*args,**kwargs):
            if isinstance(value,jax.Array):raise AssertionError('host device read')
            return original_asarray(value,*args,**kwargs)
        with (patch.object(np,'asarray',side_effect=no_device_array),
              patch.object(jax,'device_get',side_effect=AssertionError('host read')),
              patch.object(v4_encoder,'parse_own_requests',side_effect=AssertionError('host parse'))):
            actual=own.produce_own_facts(self.start,self.compiled,owner_sides=owners)
        self.same(actual,expected)


if __name__=='__main__':unittest.main()
