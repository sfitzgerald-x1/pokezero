"""Authored components and pinned source methods; no generated rosters/games."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import NamedTuple
import unittest

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_request_source as s, engine_private_actions as a
from gpu_engine import engine_request_order as order, engine_public_events as events
from gpu_engine import private_actions, public_moves, source_provenance, transition
from pokezero.category_vocab import CategoryVocabulary
from pokezero.dex import ShowdownDex
from _showdown_root import showdown_root
import test_gpu_engine_private_actions as action_fixture
import test_gpu_engine_public_belief as public_fixture


class Arrival(NamedTuple):
    stream_id: object
    engine_binding: object
    generation: object
    physical_active: object
    public_slot: object
    encounter: object
    established: object
    revealed_species_id: object


@unittest.skipUnless(shutil.which('node') and (showdown_root()/'dist/sim/pokemon.js').is_file(),
                     'requires pinned static Showdown source')
class RequestSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        action_fixture.PrivateActionTests.setUpClass()
        f=action_fixture.PrivateActionTests(); cls.reference=f
        cls.root=showdown_root(); cls.node=shutil.which('node')
        base=Path(s.__file__).parent
        tree=json.loads(subprocess.check_output([cls.node,'-e',
            'console.log(JSON.stringify(require(process.argv[1]).compiledTree(process.argv[2])))',
            str(base/'gender_policy.cjs'),str(cls.root/'dist')],text=True))
        species=dict(f.dex.species)
        for name,types in [('dugtrio',('Ground',)),('magneton',('Electric','Steel')),('wobbuffet',('Psychic',))]:
            species[name]=replace(species['pikachu'],id=name,name=name,types=types)
        cls.dex=ShowdownDex(dict(f.dex.moves),species,{})
        labels=set(f.vocab.tokens)
        for name,info in species.items():
            labels.add('species:'+name)
            labels.update('type:'+t for t in info.types)
        for ability in ('arenatrap','magnetpull','shadowtag'):
            labels.add('belief:possible_ability:'+ability)
        labels.add('belief:possible_item:choiceband')
        cls.vocab=CategoryVocabulary(tuple(sorted(labels)))
        cls.reference.vocab=cls.vocab
        cls.metadata=dict(schema='authored-request-component',source_commit_label=s.SHOWDOWN,
            compiled_tree=tree,vocabulary=dict(
                species=['_none','pikachu','gengar','snorlax','dugtrio','magneton','wobbuffet'],
                moves=['_none','surf','hiddenpowerice','curse','reversal','struggle','toxic','transform','hyperbeam',
                    'taunt','imprison','encore','sleeptalk'],
                types=['_none','normal','fighting','flying','poison','ground','rock','bug','ghost','steel',
                    'fire','water','grass','electric','psychic','ice','dragon','dark'],
                abilities=['_none','static','levitate','immunity','arenatrap','magnetpull','shadowtag'],
                items=['_none','leftovers','choiceband']))
        cls.metadata['moves']=[None]+[dict(id=name,base_power=(f.dex.move_info(name).base_power
            if f.dex.move_info(name) else 0),type=1,source_type='Normal',category='Physical',pp=10,
            accuracy=100,priority=0,flags={}) for name in cls.metadata['vocabulary']['moves'][1:]]
        cls.metadata['type_chart']=[[1]*17 for _ in range(17)]
        cls.table=transition.tables_from_metadata(cls.metadata)
        public_fixture.PublicBeliefTests.setUpClass()
        public_catalog=public_moves.compile_moves(public_fixture.PublicBeliefTests.host,cls.dex,cls.vocab)
        cls.lifecycle_catalog=events.compile_catalog(cls.metadata,cls.table,public_moves=public_catalog,
            dex=cls.dex,exposure=events.HealthExposure(s.SHOWDOWN,'gen3randombattle',True,False))
        cls.rules=s.load_source_rules(cls.root,cls.metadata,node=cls.node)
        cls.compiled=s.compile_request_source(cls.metadata,vocabulary=cls.vocab,dex=cls.dex,
            checkpoint_sha256='a'*64,rules=cls.rules,lifecycle_catalog=cls.lifecycle_catalog)
        species=np.array([[1,2,1,0,0,0],[3,2,0,0,0,0],[5,2,0,0,0,0]],np.int32)
        count=len(species)
        present=species>0
        moves=np.zeros((count,6,4),np.int32);moves[0,:3]=[1,2,3,0];moves[1:,:2]=[4,1,0,0]
        pp=np.zeros_like(moves);pp[0,:3]=[24,16,16,0];pp[1:,:2]=[1,24,0,0]
        types=np.zeros((count,6,2),np.int32)
        for row in range(count):
            for slot in range(6):
                if species[row,slot]:
                    for i,t in enumerate(cls.dex.species_info(cls.metadata['vocabulary']['species'][species[row,slot]]).types):
                        types[row,slot,i]=cls.metadata['vocabulary']['types'].index(t.lower())
        stats=np.where(present[...,None],np.full((count,6,6),150,np.int32),0)
        stats[...,0]=np.where(present,201,0)
        arrays=dict(species=species,level=np.where(present,80,0).astype(np.int32),types=types,stats=stats,
            moves=moves,pp=pp,item=present.astype(np.int32),ability=present.astype(np.int32),
            hidden_power_type=np.where(present,15,0).astype(np.int32),hidden_power_power=np.where(present,70,0).astype(np.int32),
            gender=np.where(present,1,0).astype(np.int32),nature=np.zeros((count,6),np.int32),
            evs=np.zeros((count,6,6),np.int32),metadata_utf8=np.frombuffer(json.dumps(cls.metadata).encode(),np.uint8))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'literal.npz';np.savez_compressed(path,**arrays)
            cls.pool=source_provenance.load_constructor_pool(path)

    def initial(self,indices=None):
        indices=np.array([[0,1],[0,1]],np.int64) if indices is None else indices
        state=transition.state_from_pool(self.pool.arrays,indices,seed=31,table=None)
        state=s._gender(state,self.pool,indices,None)._replace(entry_pending=j.zeros((2,2),j.bool_))
        owners=j.array([0,1],j.int32);streams=j.array([71,72],j.int32)
        reg=s._registration(state,self.pool,indices,streams,owners)
        engine=j.broadcast_to(self.rules.device.engine_binding,(2,8))
        sizes=j.sum(state.species[j.arange(2),owners]>0,axis=-1).astype(j.int32)
        ledger=order.initialize_order(streams,owners,sizes,engine_binding=engine)
        arrival=Arrival(streams,engine,j.zeros((2,2),j.int32),state.active,
            j.zeros((2,2),j.int32),j.ones((2,2),j.int32),j.ones((2,2),j.bool_),state.species[:,:,0])
        return state,ledger,reg,arrival

    def produce(self,state,ledger,reg,arrival):
        return s.produce_request(self.compiled,state,ledger,reg,arrival,lifecycle_admitted=j.ones(2,j.bool_))

    def raw(self,state,ledger,reg,arrival):
        v=self.metadata['vocabulary']; result=[]
        for row in range(2):
            sides=[]
            hp=np.asarray(state.hp[row]);active=np.asarray(state.active[row])
            forced=np.asarray(state.forced_switch[row])|(hp[np.arange(2),active]<=0)
            for side in range(2):
                physical=(ledger.request_to_physical[row].tolist() if side==int(ledger.owner_side[row]) else
                    [int(active[side])]+[i for i in range(6) if i!=active[side] and int(state.species[row,side,i])>0])
                party=[]
                for slot in physical:
                    if slot<0:continue
                    species=int(state.species[row,side,slot])
                    if not species:continue
                    original_max=(np.asarray(reg.constructor.device.max_pp[row,slot]) if side==int(ledger.owner_side[row])
                        else np.asarray(self.pool.arrays['pp'][side,slot]))
                    move_rows=[dict(id=v['moves'][int(mid)],pp=int(state.pp[row,side,slot,k]),maxpp=int(original_max[k]))
                        for k,mid in enumerate(state.moves[row,side,slot].tolist()) if mid]
                    types=[v['types'][int(t)].title() for t in state.types[row,side,slot].tolist() if t]
                    # Showdown spells these two differently from title().
                    types=[{'Psychic':'Psychic'}.get(t,t) for t in types]
                    ability=v['abilities'][int(state.ability[row,side,slot])]
                    party.append(dict(species=v['species'][species],level=int(state.level[row,side,slot]),
                        gender=int(state.gender[row,side,slot]),hp=int(state.hp[row,side,slot]),
                        maximum=int(state.max_hp[row,side,slot]),types=types,ability=ability,
                        base_ability=v['abilities'][int(state.base_ability[row,side,slot])],
                        item=v['items'][int(state.item[row,side,slot])] if state.item[row,side,slot] else '',
                        stats=dict(zip(('hp','atk','def','spa','spd','spe'),state.stats[row,side,slot].tolist())),
                        moves=move_rows,hp_type=v['types'][int(state.hidden_power_type[row,side,slot])].title(),
                        rooted=bool(int(state.volatile[row,side,slot])&8),
                        linked=bool(int(state.trap_source[row,side,slot])==int(active[1-side]) and int(state.trap_source[row,side,slot])>=0),
                        visible_trap=bool(state.trapped[row,side]),choice=v['moves'][int(state.choice_move[row,side,slot])] if state.choice_move[row,side,slot] else ''))
                sides.append(dict(party=party,forced=bool(forced[side])))
            result.append(dict(sides=sides,replacement=bool(forced.any()),
                public_species=[v['species'][int(x)] for x in arrival.revealed_species_id[row].tolist()]))
        output=subprocess.check_output([self.node,str(Path(__file__).with_name('gpu_engine_request_source_oracle.cjs')),str(self.root)],
            input=json.dumps(result),text=True)
        return [raw[int(ledger.owner_side[row])] for row,raw in enumerate(json.loads(output))]

    def compare(self,state,ledger,reg,arrival):
        produced=self.produce(state,ledger,reg,arrival)
        self.assertEqual(produced.actions.admitted.tolist(),[True,True],produced.reason.tolist())
        for row,raw in enumerate(self.raw(state,ledger,reg,arrival)):
            owner=int(ledger.owner_side[row])
            facts=private_actions.parse_action_facts(raw,owner_side=owner,vocabulary=self.vocab,
                catalog=s._checked(self.compiled)['catalog'])
            for actual,expected in zip(jax.tree.leaves(produced.actions.facts),jax.tree.leaves(facts)):
                np.testing.assert_array_equal(actual[row],expected)
            numeric,categories,legal=self.reference.expected(raw,owner,dex=self.dex)
            np.testing.assert_allclose(produced.actions.features.numeric[row],numeric,rtol=0,atol=1e-6)
            np.testing.assert_array_equal(produced.actions.features.categorical[row],categories)
            self.assertEqual(produced.actions.features.legal[row].tolist(),list(legal))
        return produced

    def test_actual_get_requests_and_get_moves_every_action_cell_both_owners(self):
        state,ledger,reg,arrival=self.initial()
        out=self.compare(state,ledger,reg,arrival)
        self.assertEqual(out.actions.facts.request.request_id.tolist(),[-1,-1])
        self.assertTrue(bool(out.actions.features.legal[:,4:].any()))
        self.assertEqual(s.route_choice(out,ledger,j.array([4,4],j.int32)).engine_action.tolist(),[4,4])

    def test_source_pp_zero_disabled_and_struggle_has_missing_pp(self):
        state,ledger,reg,arrival=self.initial()
        state=state._replace(pp=state.pp.at[0,0,0,0].set(0))
        out=self.compare(state,ledger,reg,arrival)
        self.assertFalse(bool(out.actions.features.legal[0,0]))
        state=state._replace(pp=state.pp.at[0,0,0].set(0).at[1,1,0].set(0))
        out=self.compare(state,ledger,reg,arrival)
        self.assertEqual(out.actions.facts.request.pp_known.tolist(),[[False]*4,[False]*4])
        self.assertEqual(out.actions.facts.request.move_present.tolist(),[[True,False,False,False]]*2)
        self.assertEqual(s.route_choice(out,ledger,j.zeros(2,j.int32)).engine_action.tolist(),[0,0])

    def test_actual_choice_band_source_disable_and_synthetic_pp_absence(self):
        state,ledger,reg,arrival=self.initial()
        state=state._replace(item=state.item.at[0,0,0].set(2),
            choice_move=state.choice_move.at[0,0,0].set(1))
        out=self.compare(state,ledger,reg,arrival)
        self.assertEqual(out.actions.features.legal[0,:4].tolist(),[True,False,False,False])
        state=state._replace(pp=state.pp.at[0,0,0,0].set(0))
        out=self.compare(state,ledger,reg,arrival)
        self.assertEqual(out.actions.facts.request.pp_known[0].tolist(),[False]*4)
        self.assertEqual(s.route_choice(out,ledger,j.zeros(2,j.int32)).engine_action.tolist(),[0,0])

    def test_source_forced_and_other_seat_wait_from_actual_kernel_phase(self):
        state,ledger,reg,arrival=self.initial()
        state=state._replace(forced_switch=state.forced_switch.at[0,0].set(True).at[1,0].set(True))
        out=self.compare(state,ledger,reg,arrival)
        self.assertEqual(out.actions.facts.request.force_switch.tolist(),[True,False])
        self.assertEqual(out.actions.facts.request.wait.tolist(),[False,True])

    def test_actual_and_possible_trappers_differ_from_physical_server_mask(self):
        state,ledger,reg,arrival=self.initial()
        # Each authored case changes only the server foe and the already-public
        # arrival species; no opponent roster species is used as public evidence.
        for species,ability in ((4,1),(4,4),(5,1),(5,5),(6,6)):
            with self.subTest(species=species,ability=ability):
                changed=state._replace(species=state.species.at[0,1,0].set(species),
                    ability=state.ability.at[0,1,0].set(ability))
                known=arrival._replace(revealed_species_id=arrival.revealed_species_id.at[0,1].set(species))
                out=self.compare(changed,ledger,reg,known)
                self.assertEqual(bool(out.actions.facts.request.trapped[0]),species in (4,6))
                if species==4 and ability==1:
                    self.assertTrue(bool(out.actions.engine_legal[0,4]))
                    self.assertFalse(bool(out.actions.features.legal[0,4]))

    def test_gen3_steel_levitate_shadowtag_and_installed_owner_order(self):
        state,ledger,reg,arrival=self.initial(np.array([[2,0],[0,1]],np.int64))
        state=state._replace(species=state.species.at[0,1,0].set(5))
        arrival=arrival._replace(revealed_species_id=arrival.revealed_species_id.at[0,1].set(5))
        out=self.compare(state,ledger,reg,arrival)
        self.assertFalse(bool(out.actions.facts.request.trapped[0]))  # No Gen3 alternate Foe callback
        self.assertTrue(bool(out.actions.engine_legal[0,4]))  # actual foe is Static
        state,ledger,reg,arrival=self.initial()
        receipt=order.SwitchInstalled(ledger.stream_id,ledger.owner_side,ledger.engine_binding,ledger.generation,
            j.array([True,False]),j.array([0,0],j.int32),j.array([1,-1],j.int32))
        ledger=order.install_switch(ledger,receipt)
        state=state._replace(active=state.active.at[0,0].set(1),ability=state.ability.at[0,0,1].set(2),
            species=state.species.at[0,1,0].set(4))
        arrival=arrival._replace(physical_active=state.active,
            generation=arrival.generation.at[0,0].set(1),
            revealed_species_id=arrival.revealed_species_id.at[0].set(j.array([2,4],j.int32)))
        out=self.compare(state,ledger,reg,arrival)
        self.assertFalse(bool(out.actions.facts.request.trapped[0]))  # actual Levitate
        state,ledger,reg,arrival=self.initial()
        state=state._replace(ability=state.ability.at[0,0,0].set(6),species=state.species.at[0,1,0].set(6))
        arrival=arrival._replace(revealed_species_id=arrival.revealed_species_id.at[0,1].set(6))
        self.assertFalse(bool(self.compare(state,ledger,reg,arrival).actions.facts.request.trapped[0]))
        state=state._replace(ability=state.ability.at[0,1,0].set(6))
        self.assertTrue(bool(self.compare(state,ledger,reg,arrival).actions.facts.request.trapped[0]))

    def test_gen3_public_alternate_magnetpull_vs_actual_callback_both_owners(self):
        state,ledger,reg,arrival=self.initial(np.array([[2,0],[0,2]],np.int64))
        # Both owners retain their unchanged Steel constructor active and live
        # bench. Public Magneton alone supplies no FoeMaybeTrapPokemon handler.
        state=state._replace(species=state.species.at[0,1,0].set(5).at[1,0,0].set(5))
        arrival=arrival._replace(revealed_species_id=j.full((2,2),5,j.int32))
        payload=json.loads(s._rules(self.rules)[2])
        magneton=payload['species'][5]
        self.assertIn('magnetpull',magneton['abilities'])
        self.assertNotIn('magnetpull',magneton['alternate_maybe_trap_abilities'])
        self.assertFalse(bool(self.rules.device.possible_magnet[5]))
        self.assertIn('arenatrap',payload['species'][4]['alternate_maybe_trap_abilities'])
        self.assertIn('shadowtag',payload['species'][6]['alternate_maybe_trap_abilities'])
        for actual,expected in ((1,False),(5,True)):
            with self.subTest(actual_foe_ability=actual):
                changed=state._replace(ability=state.ability.at[0,1,0].set(actual).at[1,0,0].set(actual))
                raw=self.raw(changed,ledger,reg,arrival)
                for request in raw:
                    active=request['active'][0]
                    self.assertEqual(bool(active.get('trapped',False) or active.get('maybeTrapped',False)),expected)
                    if not expected:
                        self.assertNotIn('trapped',active)
                        self.assertNotIn('maybeTrapped',active)
                out=self.compare(changed,ledger,reg,arrival)
                self.assertEqual(out.actions.facts.request.trapped.tolist(),[expected]*2)
                self.assertEqual(out.actions.features.legal[:,4].tolist(),[not expected]*2)
                self.assertEqual(out.actions.engine_legal[:,4].tolist(),[not expected]*2)

    def test_missing_arrival_gender_origin_locks_or_disable_visibility_refuse(self):
        state,ledger,reg,arrival=self.initial()
        cases=[(state,arrival._replace(established=arrival.established.at[0,1].set(False)),s.ARRIVAL),
            (state._replace(gender=state.gender.at[0,0,0].set(-1)),arrival,s.GENDER),
            (state._replace(stats=state.stats.at[0,0,0,1].add(1)),arrival,s.CHANGED),
            (state._replace(charge_move=state.charge_move.at[0,0,0].set(1)),arrival,s.LOCK),
            (state._replace(last_move=state.last_move.at[0,0,0].set(7)),arrival,s.LOCK),
            (state._replace(last_move=state.last_move.at[0,0,0].set(8)),arrival,s.LOCK),
            (state._replace(disabled=state.disabled.at[0,0,0,0].set(True)),arrival,s.DISABLE),
            (state._replace(trapped=state.trapped.at[0,0].set(True)),arrival,s.TRAP_ORIGIN),
            (state._replace(entry_pending=state.entry_pending.at[0,0].set(True)),arrival,s.PHASE)]
        for changed,known,reason in cases:
            out=self.produce(changed,ledger,reg,known)
            self.assertEqual(out.actions.admitted.tolist(),[False,True])
            self.assertTrue(int(out.reason[0])&reason)
            self.assertFalse(bool(out.actions.features.numeric[0].any()))
            self.assertFalse(bool(out.actions.features.categorical[0].any()))

    def test_missing_request_origins_are_monotone_across_later_safe_moves(self):
        state,ledger,reg,arrival=self.initial()
        for move in (7,8,9,10,12):
            self.assertTrue(bool(self.rules.device.unsupported_history[move]))
            # A foe's unsupported request-changing condition matters too; a
            # later move cannot certify that a missing volatile disappeared.
            changed=state._replace(last_move=state.last_move.at[0,1,0].set(move))
            carried=s._advance_registration(reg,changed,self.rules.device)
            out=self.produce(state,ledger,carried,arrival)
            self.assertEqual(out.actions.admitted.tolist(),[False,True])
            self.assertTrue(int(out.reason[0])&s.LOCK)
        self.assertFalse(bool(self.rules.device.unsupported_history[11]))  # actual Encore counters retained

    def test_static_source_descriptor_and_compiled_tree_are_bound(self):
        payload=json.loads(s._rules(self.rules)[2])
        self.assertEqual(payload['source_tree']['descriptor']['compiled_tree'],self.metadata['compiled_tree'])
        self.assertEqual(payload['source_files']['sim/battle.ts'],
            'b5958545c4539ca60aead0280f5eadad9e098777d38e0d56a50f143e22a5cc6d')
        bad=deepcopy(self.metadata);bad['compiled_tree']['sha256']='0'*64
        with self.assertRaises(subprocess.CalledProcessError):
            s.load_source_rules(self.root,bad,node=self.node)

    def test_mutable_rule_and_registration_edges_never_enter_checked_source(self):
        state,ledger,reg,arrival=self.initial()
        self.produce(state,ledger,reg,arrival)
        with self.assertRaises(ValueError):self.produce(state,ledger,replace(reg),arrival)
        with self.assertRaises(ValueError):s._rules(replace(self.rules))
        with self.assertRaises(ValueError):
            jax.jit(lambda:self.produce(state,ledger,reg,arrival))()
        for value in reg.original:
            view=np.asarray(value)
            with self.assertRaises(ValueError):view.setflags(write=True)
        with self.assertRaises(ValueError):
            self.produce(state._replace(choice_move=state.choice_move.astype(j.float32)),ledger,reg,arrival)


if __name__=='__main__':unittest.main()
