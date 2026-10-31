"""Literal pure specs/current owner histories; no scientific banks or models."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from pokezero.poke_engine_adapter import PokemonSpec,MoveSpec,SideSpec,BattleSpec
from pokezero.gen3_damage import gen3_stat,gen3_hp_stat
from pokezero.showdown_fixture import FixturePokemon,pack_team
from pokezero.env import BattleStartOverride
from pokezero.live_protocol_actor import ActorContract,_vocab_sha
from pokezero.local_showdown import LocalShowdownConfig
from pokezero.observation import ObservationFeatureMasks
from pokezero.showdown import V4_REPLAY_OBSERVATION_SPEC
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,RandbatSourceMetadata
from pokezero.randbat import Gen3RandbatVariant
from pokezero.category_vocab import CategoryVocabulary
from pokezero.dex import ShowdownDex
from gpu_engine import v5_hypothetical_state as mapping,v5_public_world as world,v5_public_branch as branch
from gpu_engine import v5_owning_history as capture,public_wire,v4_encoder,transition


class TypedMappingTests(unittest.TestCase):
    def fixture(self):
        mon=PokemonSpec('snorlax',80,('normal',),100,200,100,101,102,103,104,
            (MoveSpec('hiddenpowerice70',16),MoveSpec('tackle',56)),gender='M')
        side=SideSpec((mon,),boosts={'attack':2},side_conditions={'spikes':2,'toxic_count':3,'protect':1},
            wish=(1,100),last_used_move='move:1')
        metadata={'vocabulary':{'species':['_none','snorlax'],'types':['_none','normal','ice'],
            'moves':['_none','hiddenpower','tackle','struggle'],'abilities':['_none'],'items':['_none']}}
        return BattleSpec(side,side),metadata

    def test_exact_fields_hp_alias_raw_stats_clocks_and_happiness(self):
        spec,metadata=self.fixture()
        result=mapping.map_spec(spec,metadata=metadata,seed=1,turn=7,active_turns=(2,4),happiness=((7,),(255,)))
        self.assertFalse(result.refusals)
        state=result.state
        self.assertEqual(state.stats[0,0,0].tolist(),[200,100,101,102,103,104])
        self.assertEqual(state.happiness[0,:,0].tolist(),[7,255])
        self.assertEqual(state.active_turns.tolist(),[[2,4]])
        self.assertEqual(int(state.hidden_power_type[0,0,0]),2)
        self.assertEqual(int(state.hidden_power_power[0,0,0]),70)
        self.assertEqual(state.moves[0,0,0,:2].tolist(),[1,2])
        self.assertEqual(int(state.status_counter[0,0,0]),3)
        self.assertEqual(int(state.protect_counter[0,0,0]),1)
        self.assertEqual(state.wish_counter.tolist(),[[1,1]])
        self.assertEqual(state.turn.tolist(),[6]) # public turn7, six completed turns

    def test_unrepresentable_fields_refuse_without_approximating(self):
        spec,metadata=self.fixture()
        for changed in (replace(spec,side_one=replace(spec.side_one,force_switch=True)),
                replace(spec,side_one=replace(spec.side_one,volatile_statuses=('encore',))),
                replace(spec,side_one=replace(spec.side_one,pokemon=(replace(spec.side_one.pokemon[0],rest_turns=2),)))):
            result=mapping.map_spec(changed,metadata=metadata,seed=1,turn=7,active_turns=(2,4),happiness=((7,),(255,)))
            self.assertIsNone(result.state);self.assertTrue(result.refusals)


class PublicWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_engine_weather_public as fixtures
        from gpu_engine import public_belief,engine_public_events,engine_request_source,engine_own_facts
        from gpu_engine import engine_retained_encoder_driver,engine_leftovers_callbacks,engine_direct_hit_controls
        fixtures.WeatherPublicTests.setUpClass();base=fixtures.WeatherPublicTests
        specs=(('tackle','Normal',35,35,'Physical'),('quickattack','Normal',40,30,'Physical'),
            ('watergun','Water',40,25,'Special'),('ember','Fire',40,25,'Special'))
        metadata=deepcopy(base.metadata);moves=dict(base.catalog.dex.moves);ids=[]
        for name,typ,power,pp,category in specs:
            ids.append(len(metadata['moves']));metadata['vocabulary']['moves'].append(name)
            metadata['moves'].append(dict(id=name,base_power=power,type=metadata['vocabulary']['types'].index(typ.lower()),
                source_type=typ,category=category,pp=pp,accuracy=100,priority=0,flags={'protect':1}))
            moves[name]=replace(base.catalog.dex.moves['surf'],id=name,name=name,type=typ,category=category,
                gen3_category=category,base_power=power,pp=pp,target='normal',status=None,accuracy=100.)
        dex=ShowdownDex(moves,base.catalog.dex.species,base.catalog.dex.type_chart);names=tuple(spec[0] for spec in specs)
        variants={name:Gen3RandbatSpeciesUniverse(name,80,tuple(Gen3RandbatVariant(name+'-'+ability,'literal',name,
            'literal',80,names,ability,'Leftovers') for ability in ('Pressure','Synchronize')))
            for name in metadata['vocabulary']['species'][1:]}
        original_source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'v5-literal-world'),variants)
        labels=set(base.catalog.vocabulary.tokens)
        for name in names:labels|={'move:'+name,'belief:possible_move:'+name}
        labels|={'ability:synchronize','belief:possible_ability:synchronize'}
        catalog=v4_encoder.compile_encoder(public_belief.compile_source(original_source),dex,
            CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=base.catalog.masks.__dict__)
        table=transition.tables_from_metadata(metadata)
        producer=engine_public_events.compile_catalog(metadata,table,public_moves=catalog.moves,dex=dex,
            exposure=engine_public_events.HealthExposure(engine_public_events.SHOWDOWN,'gen3randombattle',True,False))
        original=base.f.f
        rules=engine_request_source.load_source_rules(original.root,metadata,node=original.node)
        request=engine_request_source.compile_request_source(metadata,vocabulary=catalog.vocabulary,dex=dex,
            checkpoint_sha256='a'*64,rules=rules,lifecycle_catalog=producer)
        own=engine_own_facts.compile_own_facts(request,catalog)
        prepared=engine_retained_encoder_driver.prepare(request,own,public_mode='retained-full',
            leftovers_compiled=engine_leftovers_callbacks.compile_leftovers(metadata,engine_catalog=producer),
            leftovers_mode='retained',account_source_sites=True,track_kernel_risks=True,
            direct_hit_compiled=engine_direct_hit_controls.compile_direct_hit_controls(engine_catalog=producer),
            direct_hit_mode='kernel-event',frame_input_mode='source-witness',
            fake_forward=lambda _:engine_retained_encoder_driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
        cls.f=NS(metadata=metadata,names=names,move_ids=ids,max_pp=tuple(spec[3]*8//5 for spec in specs),prepared=prepared)
        cls.runtime=engine_retained_encoder_driver._checked(prepared);cls.catalog=catalog
        rows=cls.catalog.source
        cls.source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,rows.source_hash),
            {name:Gen3RandbatSpeciesUniverse(name,80,rows.variants[index]) for index,name in enumerate(rows.species_names) if index})
        masks=dict(cls.catalog.masks.__dict__);masks['opponent_tendency_stats_block']=masks.pop('stats_block')
        cls.config=LocalShowdownConfig(showdown_root='/literal-source-only',set_belief_source=True,
            category_vocab=cls.catalog.vocabulary,observation_spec=V4_REPLAY_OBSERVATION_SPEC,
            feature_masks=ObservationFeatureMasks(**masks))
        cls.contract=replace(ActorContract.champion(),category_vocab_sha256=_vocab_sha(cls.config.category_vocab),
            categorical_vocab_size=cls.config.category_vocab.size,belief_set_source_hash=rows.source_hash)

    def fixture(self,*,spent=False,foe_damage100=False,slow_own=False,foe_lead=0,own_level=80,all_own_levels=False):
        metadata=self.f.metadata;names=metadata['vocabulary']['species'][1:]
        pp=[self.catalog.dex.move_info(name).max_pp for name in self.f.names]
        teams=[];own=[]
        for name in names:
            from pokezero.determinization import _gen3_randbat_fixture_spread
            spread=_gen3_randbat_fixture_spread({},species=name,moves=self.f.names,item='Leftovers',level=80,set_source=self.source)
            info=self.catalog.dex.species_info(name)
            stats={key:gen3_stat(info.base_stats[key],spread['ivs'][key],spread['evs'][key],80) for key in ('atk','def','spa','spd','spe')}
            maximum=gen3_hp_stat(info.base_stats['hp'],spread['ivs']['hp'],spread['evs']['hp'],80)
            mon=FixturePokemon(name,self.f.names,'Pressure','Leftovers',80,gender='M',
                evs=spread['evs'],ivs=spread['ivs'])
            teams.append(mon)
            own.append(dict(ident='p1: '+name,details=name+', L80, M',condition=f'{maximum}/{maximum}',
                active=len(own)==0,moves=list(self.f.names),baseAbility='Pressure',item='Leftovers',stats=stats))
        own_team=list(teams)
        if own_level!=80:
            self.assertFalse(slow_own)
            for index in range(len(teams)) if all_own_levels else (0,):
                mon=teams[index];spread=_gen3_randbat_fixture_spread({},species=mon.species,moves=self.f.names,
                    item='Leftovers',level=own_level,set_source=self.source)
                own_team[index]=replace(mon,level=own_level,ivs=spread['ivs'],evs=spread['evs'])
                info=self.catalog.dex.species_info(mon.species)
                own[index]['details']=f'{mon.species}, L{own_level}, M'
                maximum=gen3_hp_stat(info.base_stats['hp'],spread['ivs']['hp'],spread['evs']['hp'],own_level)
                own[index]['condition']=f'{maximum}/{maximum}'
                own[index]['stats']={key:gen3_stat(info.base_stats[key],spread['ivs'][key],spread['evs'][key],own_level)
                    for key in ('atk','def','spa','spd','spe')}
        first=own[0];name=first['details'].split(',')[0]
        foe_name=names[foe_lead]
        maximum=int(first['condition'].split('/')[1])
        lines=[f'|switch|p1a: {name}|{name}, L{own_level}, M|{maximum}/{maximum}',
            f'|switch|p2a: {foe_name}|{foe_name}, L80, M|100/100',f'|-ability|p1a: {name}|Pressure|[silent]','|turn|1']
        request=dict(side=dict(id='p1',pokemon=own),active=[dict(moves=[dict(id=move,pp=maximum,maxpp=maximum,disabled=False)
            for move,maximum in zip(self.f.names,pp)])])
        initial_json=json.dumps(request,sort_keys=True,separators=(',',':'))
        wire,packets=public_wire.project_lines(public_wire.WireState(),lines,**v4_encoder.wire_kwargs(self.catalog))
        words=tuple(tuple(map(int,event.words())) for packet in packets for event in packet.events)
        # Explicit literal source-component record, not a scientific capture.
        frame=capture.Frame(0,(words,),((True,)*len(words),),(False,),initial_json,())
        frames=[frame]
        if spent or foe_damage100:
            extra=[]
            if spent:
                extra+=['|move|p2a: '+foe_name+'|Tackle|p1a: '+name,f'|-damage|p1a: {name}|{maximum-1}/{maximum}']
                request['side']['pokemon'][0]['condition']=f'{maximum-1}/{maximum}'
            if foe_damage100:extra+=[f'|-damage|p2a: {foe_name}|100/100']
            extra+=['|','|upkeep','|turn|2']
            _,packets=public_wire.project_lines(wire,extra,**v4_encoder.wire_kwargs(self.catalog))
            words=tuple(tuple(map(int,event.words())) for packet in packets for event in packet.events)
            frames.append(capture.Frame(1,(words,),((True,)*len(words),),(False,),
                json.dumps(request,sort_keys=True,separators=(',',':')),()))
        constructor=(('binding',(0,)*8),('species',tuple(range(1,7))),
            ('moves',tuple(tuple(self.f.move_ids) for _ in range(6))),('max_pp',tuple(tuple(pp) for _ in range(6))))
        history=capture.History((1,id(self.f.prepared),3),0,0,3401,self.catalog,constructor,tuple(frames))
        with ExitStack() as stack:
            stack.enter_context(patch('subprocess.Popen',side_effect=AssertionError('no simulator')))
            stack.enter_context(patch('pokezero.local_showdown.load_gen3_randbat_source_cached',return_value=self.source))
            stack.enter_context(patch('pokezero.local_showdown.load_showdown_dex_cached',return_value=self.catalog.dex))
            result=capture.public_root(history,config=self.config,contract=self.contract,battle_id='literal-component',seed=31)
        self.assertIsNotNone(result.root,result.decoded.refusal_counts)
        foe_team=teams[foe_lead:]+teams[:foe_lead]
        start=BattleStartOverride(player_teams={'p1':pack_team(own_team),'p2':pack_team(foe_team)},observation_format_id='gen3randombattle')
        if slow_own:
            # A coherent literal own source set with a different raw Speed.
            # The opponent's full source-ordered sampled set stays unchanged.
            ivs={**teams[0].ivs,'spe':0};own_team=[replace(teams[0],ivs=ivs),*teams[1:]]
            request['side']['pokemon'][0]['stats']['spe']=gen3_stat(
                self.catalog.dex.species_info(name).base_stats['spe'],0,teams[0].evs['spe'],80)
            frames=[replace(frame,request_json=json.dumps({**json.loads(frame.request_json),
                'side':{**json.loads(frame.request_json)['side'],'pokemon':[
                    {**row,'stats':request['side']['pokemon'][i]['stats']} for i,row in enumerate(json.loads(frame.request_json)['side']['pokemon'])]}},
                sort_keys=True,separators=(',',':'))) for frame in frames]
            history=replace(history,frames=tuple(frames))
            with ExitStack() as stack:
                stack.enter_context(patch('subprocess.Popen',side_effect=AssertionError('no simulator')))
                stack.enter_context(patch('pokezero.local_showdown.load_gen3_randbat_source_cached',return_value=self.source))
                stack.enter_context(patch('pokezero.local_showdown.load_showdown_dex_cached',return_value=self.catalog.dex))
                result=capture.public_root(history,config=self.config,contract=self.contract,battle_id='literal-component',seed=31)
            self.assertIsNotNone(result.root,result.decoded.refusal_counts)
            start=replace(start,player_teams={'p1':pack_team(own_team),'p2':pack_team(foe_team)})
        static=world.OwnStatic(history.stream_id,0,(0,)*8,tuple(range(7,13)))
        return result.root,start,history,static

    def test_original_pure_mapper_no_native_and_exact_owned_happiness_shared_public(self):
        root,start,history,static=self.fixture(spent=True)
        retained=world.public_carry(history,self.runtime)
        with patch('pokezero.engine_world.build_poke_engine_state',side_effect=AssertionError('no Native')):
            result=world.materialize(root,start,prepared=self.f.prepared,history=history,own_static=static,seed=5,retained=retained)
        self.assertIsNotNone(result.branch,result.refusals)
        self.assertIs(result.branch.public,retained)
        self.assertEqual(result.branch.state.happiness[0,0].tolist(),list(range(7,13)))
        self.assertEqual(int(result.branch.state.pp[0,1,0,0]),self.f.max_pp[0]-2)
        self.assertEqual(result.branch.state.active_turns.tolist(),[[1,1]])

    def test_foreign_own_static_or_history_and_unknown_sleep_refuse(self):
        root,start,history,static=self.fixture()
        result=world.materialize(root,start,prepared=self.f.prepared,history=history,
            own_static=replace(static,constructor_binding=(1,)*8),seed=5)
        self.assertIsNone(result.branch);self.assertIn('same-actual-own-static-constructor-required',result.refusals)
        altered=replace(root,owner_lines_sha256='a'*64)
        result=world.materialize(altered,start,prepared=self.f.prepared,history=history,own_static=static,seed=5)
        self.assertIsNone(result.branch);self.assertIn('public-root-owning-history-bytes',result.refusals)
        payload=deepcopy(root.materialization_payload);payload['sides']['p1']['pokemon'][0]['condition']+=' slp'
        with patch('pokezero.engine_world.require_rest_turns_support',side_effect=AssertionError('no Native probe')):
            result=world.materialize(replace(root,materialization_payload=payload),start,
                prepared=self.f.prepared,history=history,own_static=static,seed=5)
        self.assertIsNone(result.branch);self.assertIn('native-probed-sleep-clock',result.refusals)
        payload=deepcopy(root.materialization_payload);payload['selfActiveRequestState']={'trapped':True}
        with patch('gpu_engine.v5_public_world.world_battle_spec',side_effect=AssertionError('no Native trap probe')):
            result=world.materialize(replace(root,materialization_payload=payload),start,
                prepared=self.f.prepared,history=history,own_static=static,seed=5)
        self.assertIsNone(result.branch);self.assertIn('native-probed-hard-trapped-own-request',result.refusals)

    def test_opponent_own_inputs_and_one_original_kernel_keep_unknown_profiles(self):
        root,start,history,static=self.fixture()
        result=world.materialize(root,start,prepared=self.f.prepared,history=history,own_static=static,seed=5)
        self.assertIsNotNone(result.branch,result.refusals)
        current=branch.core(result.branch,self.runtime)
        self.assertEqual(current.requests[1].actions.facts.request.owner_side.tolist(),[1])
        other=branch._opponent_public(result.branch,current,self.runtime)
        self.assertEqual(other.context.toxic.visibility.tolist(),[[2,1]])
        self.assertEqual(int(other.mons.hp_denominator[0,1,0]),int(current.engine.max_hp[0,1,0]))
        # Each viewer knows its own Pressure. The other player's private Start
        # does not become public ability evidence in that viewer's policy view.
        pressure=self.catalog.abilities.ability_id('pressure')
        self.assertEqual(int(other.context.abilities.original[0,0,0]),0)
        self.assertEqual(int(other.context.abilities.original[0,1,0]),pressure)
        draws=j.full((1,2,12),99,j.int32).at[:,:,4].set(1)
        chance=transition.ChanceInputs(draws,*[j.zeros((1,2),j.int32) for _ in range(4)],j.zeros(1,j.int32))
        before=result.branch.public
        stepped=branch.step(result.branch,(0,0),chance_inputs=chance)
        self.assertIsNotNone(stepped.raw_frame,stepped.refusals)
        self.assertIsNotNone(stepped.state)
        # The missing ordinary-birth residual profile stays UNKNOWN; it is
        # never fabricated to turn this component into a complete D provider.
        self.assertTrue(stepped.refusals)
        self.assertIs(result.branch.public,before)
        plain=transition.step_with_chance(current.engine,j.array([[0,0]],j.int32),
            self.runtime['request']['engine_table'],chance)
        for actual,expected in zip(jax.tree.leaves(stepped.state),jax.tree.leaves(plain.state)):
            np.testing.assert_array_equal(actual,expected)

    def test_public_actor_inputs_same_across_worlds_opponent_uses_own_sampled_set(self):
        from gpu_engine import v5_observation,private_team
        from pokezero import v5_contract as contract
        from pokezero.v5_model import V5ModelConfig
        census=contract.source_variant_census(self.source)
        labels=contract.LabelVocabulary(census['source_rows_sha256'],tuple(self.catalog.moves.names[1:]),census['species_variants'])
        category=np.asarray(private_team.vocabulary_binding(self.catalog.vocabulary),dtype='>u4').tobytes().hex()
        config=V5ModelConfig(self.catalog.vocabulary.size,category,labels,width=8,layers=1,attention_heads=2,
            expert_width=8,value_hidden=(8,8))
        adapter=v5_observation.compile_adapter(self.catalog,model_config=config)
        root,start,history,static=self.fixture();retained=world.public_carry(history,self.runtime)
        variants=[]
        for ability in ('Pressure','Synchronize'):
            altered=replace(start,player_teams={**dict(start.player_teams),
                'p2':start.player_teams['p2'].replace('|Pressure|','|'+ability+'|')})
            result=world.materialize(root,altered,prepared=self.f.prepared,history=history,
                own_static=static,seed=5,retained=retained)
            self.assertIsNotNone(result.branch,result.refusals)
            variants.append((branch.policy_observation(result.branch,adapter,owner=0),
                branch.policy_observation(result.branch,adapter,owner=1)))
        for name in ('numeric_features','categorical_ids','token_type_ids','attention_mask','history_mask'):
            np.testing.assert_array_equal(variants[0][0].inputs[name],variants[1][0].inputs[name],err_msg=name)
        np.testing.assert_array_equal(variants[0][0].legal_action_mask,variants[1][0].legal_action_mask)
        self.assertFalse(np.array_equal(variants[0][1].inputs['categorical_ids'],variants[1][1].inputs['categorical_ids']))
        self.assertNotIn('happiness',variants[0][0].inputs)
        self.assertTrue(bool(np.asarray(variants[0][1].inputs['attention_mask'])[...,23:].all()))

    def test_percentage100_after_damage_is_not_a_full_health_history_proof(self):
        root,start,history,static=self.fixture(foe_damage100=True)
        result=world.materialize(root,start,prepared=self.f.prepared,history=history,own_static=static,seed=5)
        self.assertIsNone(result.branch)
        self.assertIn('opponent-own-private-hp-history-unrepresented',result.refusals)


if __name__=='__main__':unittest.main()
