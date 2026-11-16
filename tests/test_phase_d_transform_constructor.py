"""Ditto/Mew constructor fixtures through the original B2 fold, on CPU pods."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as NS
import json
import unittest

import jax
import jax.numpy as j
import numpy as np

from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from pokezero.dex import MoveInfo,SpeciesInfo,ShowdownDex,normalize_id
from pokezero.category_vocab import CategoryVocabulary
from pokezero.determinization import _fixture_from_variant_payload
from pokezero.gen3_damage import gen3_stat,gen3_hp_stat
from pokezero.v5_contract import DecisionAddress,source_variant_census
from pokezero.recorded_encoder_runner import JaxRecordedAdapter
from gpu_engine import phase_d_live_worlds as live,phase_d_transform as transform_facts
from gpu_engine import device_public_worlds as plans,device_public_world_state as states,public_belief,v4_encoder,transform


def source_fixture():
    names=('ditto','mew','snorlax','kecleon','alakazam','articuno')
    abilities=('limber','synchronize','immunity','colorchange','synchronize','pressure')
    types=(('Normal',),('Psychic',),('Normal',),('Normal',),('Psychic',),('Ice','Flying'))
    moves={'transform':10,'softboiled':10,'psychic':10,'tackle':35,'protect':10,'hiddenpowerice':15,'hiddenpowerwater':15,'struggle':1}
    universes={};info={}
    for index,name in enumerate(names):
        selected=('transform',) if name=='ditto' else ('transform','softboiled','psychic','hiddenpowerice') if name=='mew' else ('tackle','protect','hiddenpowerwater')
        variant=Gen3RandbatVariant(name+'-fixture','authored',name,'authored',80,selected,abilities[index],'leftovers')
        universes[name]=Gen3RandbatSpeciesUniverse(name,80,(variant,))
        info[name]=dict(gender='N',types=list(types[index]),baseStats={k:48 if name=='ditto' else 100 for k in ('hp','atk','def','spa','spd','spe')})
    raw=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'transform-authored'),universes,
        move_metadata={name:dict(type='Ice' if name.endswith('ice') else 'Water' if name.endswith('water') else 'Normal',category='Physical') for name in moves},species_metadata=info)
    vocabulary=dict(species=['',*names],abilities=['',*dict.fromkeys(abilities)],types=['','Normal','Psychic','Flying','Ice','Water'],
        items=['','leftovers'],moves=['',*moves])
    metadata=dict(vocabulary=vocabulary,moves=[None,*[dict(pp=pp) for pp in moves.values()]])
    source=plans.compile_source(raw,metadata,expected_source_hash='transform-authored',expected_source_rows_sha256=source_variant_census(raw)['source_rows_sha256'])
    return source,metadata


def request(source,name,owner=0):
    names=[name,*[n for n in source.original.universes if n!=name]];party=[]
    for slot,species in enumerate(names):
        variant=source.original.universe_for(species).variants[0]
        fixture=_fixture_from_variant_payload(variant.to_summary(),fallback_species=species,set_source=source.original,gender='N')
        base=source.original.species_metadata[species]['baseStats'];level=variant.level
        maximum=gen3_hp_stat(base['hp'],fixture.ivs['hp'],fixture.evs['hp'],level)
        stats={k:gen3_stat(base[k],fixture.ivs[k],fixture.evs[k],level) for k in ('atk','def','spa','spd','spe')}
        party.append(dict(ident=f'p{owner+1}: '+species,details=species+', L80',condition=f'{maximum}/{maximum}',stats=stats,
            moves=list(variant.moves),baseAbility=variant.ability,item=variant.item,active=slot==0))
    pp={'tackle':56,'hiddenpowerice':24,'hiddenpowerwater':24}
    return dict(rqid=1,side=dict(id=f'p{owner+1}',pokemon=party),active=[dict(moves=[dict(id=m,move=m,pp=pp.get(m,16),maxpp=pp.get(m,16),disabled=False) for m in party[0]['moves']])])


class TransformConstructorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source,cls.metadata=source_fixture()
        raw=cls.source.original
        moves={n:MoveInfo(n,n,'Normal','Physical','Physical',50,100,0,False,False,False,None,{},'normal',False,pp=m['pp'])
            for n,m in zip(cls.metadata['vocabulary']['moves'][1:],cls.metadata['moves'][1:])}
        dex=ShowdownDex(moves,{n:SpeciesInfo(n,n,tuple(v['types']),v['baseStats']) for n,v in raw.species_metadata.items()},{})
        labels={'field','stats','pokemon:self','pokemon:opponent','status:none'}
        labels|={'belief:possible_ability:'+n for n in cls.metadata['vocabulary']['abilities'][1:]}
        labels|={'belief:possible_item:leftovers','move:hiddenpower'}
        for kind in ('species','abilities','items','moves','types'):
            token={'abilities':'ability','items':'item','moves':'move','types':'type'}.get(kind,kind)
            labels|={token+':'+name for name in cls.metadata['vocabulary'][kind][1:]}
        masks=dict(exact_state=True,feature_pack_last_move=True,investment_belief_narrowing=False,item_belief_narrowing=False,stats_block=True,tier2_investment=True,tier2_residuals=True,transition_token_budget=0)
        cls.catalog=v4_encoder.compile_encoder(public_belief.compile_source(raw),dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=masks)
        from pokezero.live_protocol_runtime import CompiledEncoderFunctions
        cls.precision=jax.config.jax_default_matmul_precision
        jax.config.update('jax_default_matmul_precision','highest')
        cls.functions=CompiledEncoderFunctions(cls.catalog)
        cls.tables=states.compile_tables(cls.source,cls.metadata)

    @classmethod
    def tearDownClass(cls):jax.config.update('jax_default_matmul_precision',cls.precision)

    def adapter(self,lines,owner):
        adapter=JaxRecordedAdapter(self.catalog,owner_side=owner,compiled_functions=self.functions)
        adapter.reset(stream_id=1,battle_id='transform-authored')
        adapter.feed(tuple(x for x in lines if not x.startswith('|request|')))
        return adapter

    def transformed(self,name,owner):
        before=request(self.source,name,owner);after=deepcopy(before)
        after['side']['pokemon'][0]['moves']=['tackle','protect','hiddenpowerwater']
        after['active'][0]['moves']=[dict(id=m,move=title,pp=pp,maxpp=5,disabled=False) for m,title,pp in (
            ('tackle','Tackle',4),('protect','Protect',5),('hiddenpower','Hidden Power Water 70',3))]
        own,peer=f'p{owner+1}',f'p{2-owner}';maximum=before['side']['pokemon'][0]['condition']
        lines=('|start',f'|switch|{own}a: {name}|{name}, L80|{maximum}',f'|switch|{peer}a: snorlax|snorlax, L80|100/100',
            '|turn|1','|request|'+json.dumps(before),f'|move|{own}a: {name}|Transform|{peer}a: snorlax',
            f'|-transform|{own}a: {name}|{peer}a: snorlax','|upkeep','|turn|2','|request|'+json.dumps(after))
        adapter=self.adapter(lines,owner)
        return before,after,lines,adapter

    def test_exact_own_base_saves_copy_and_failencore_history(self):
        for name in ('ditto','mew'):
            for owner in (0,1):
                with self.subTest(name=name,owner=owner):
                    before,after,lines,adapter=self.transformed(name,owner)
                    root=live.from_b2(adapter,after,self.source,self.metadata,address=DecisionAddress(1,1,owner),lines=lines)
                    own=root.own;v=self.source.vocabulary
                    self.assertTrue(own['transformed']);self.assertEqual(int(own['species'][0]),v['species'].index(name))
                    self.assertEqual(own['pp'][0].tolist(),[4,5,3,0]);self.assertEqual(own['transform_pp'][0],15)
                    self.assertEqual(int(own['hidden_power_power'][0]),70)
                    self.assertEqual((int(own['last_action_kind']),int(own['last_action_index'])),(2,-1))
                    self.assertEqual(own['transform_stats'].tolist(),list(before['side']['pokemon'][0]['stats'].values()))
                    arrays=states._empty_arrays()
                    for field in plans.OWN_MON_FIELDS+plans.OWN_SIDE_FIELDS:arrays[field][0,owner]=own[field]
                    state=states.empty_state(1)._replace(**{k:j.asarray(v) for k,v in arrays.items()})
                    reverted,_=transform.revert(state,owner,True)
                    np.testing.assert_array_equal(reverted.moves[0,owner,0],own['transform_moves'])
                    np.testing.assert_array_equal(reverted.pp[0,owner,0],own['transform_pp'])
                    np.testing.assert_array_equal(reverted.stats[0,owner,0,1:],own['transform_stats'])

    def test_copied_request_without_transform_evidence_still_refuses(self):
        before,after,lines,adapter=self.transformed('ditto',0)
        abilities=adapter.state.context.abilities._replace(transformed=j.zeros((1,2,6),bool))
        adapter.state=adapter.state._replace(context=adapter.state.context._replace(abilities=abilities))
        with self.assertRaisesRegex(ValueError,'OWN_STATIC_IV_FACTS_UNAVAILABLE_OR_AMBIGUOUS'):
            live.from_b2(adapter,after,self.source,self.metadata,address=DecisionAddress(1,1,0))

    def test_opposing_ditto_and_mew_copy_our_donor_and_charge_copied_pp(self):
        from gpu_engine import engine_device_public_hooks as hooks,future_state
        for name in ('ditto','mew'):
            for owner in (0,1):
                with self.subTest(name=name,owner=owner):
                    first=request(self.source,'snorlax',owner);packet=request(self.source,'articuno',owner)
                    own,peer=f'p{owner+1}',f'p{2-owner}'
                    lines=('|start',f'|switch|{own}a: snorlax|snorlax, L80|'+first['side']['pokemon'][0]['condition'],
                        f'|switch|{peer}a: {name}|{name}, L80|100/100','|turn|1','|request|'+json.dumps(first),
                        f'|move|{peer}a: {name}|Transform|{own}a: snorlax',f'|-transform|{peer}a: {name}|{own}a: snorlax',
                        '|upkeep','|turn|2',f'|move|{peer}a: {name}|Tackle|{own}a: snorlax','|upkeep','|turn|3',
                        f'|switch|{own}a: articuno|articuno, L80|'+packet['side']['pokemon'][0]['condition'],
                        f'|-ability|{own}a: articuno|Pressure|[silent]',f'|move|{peer}a: {name}|Hidden Power|{own}a: articuno',
                        '|upkeep','|turn|4','|request|'+json.dumps(packet))
                    adapter=self.adapter(lines,owner)
                    root=live.from_b2(adapter,packet,self.source,self.metadata,address=DecisionAddress(1,1,owner),lines=lines)
                    self.assertEqual(root.public.copied_pp_used[0,1-owner].tolist(),[1,0,2,0])
                    made=live.sample_worlds(root,self.source,self.tables,count=16,seed=7)
                    self.assertTrue(made.fully_constructed,[w.refusals for w in made.worlds])
                    world=made.worlds[0];s=world.state;foe=1-owner;at=int(s.active[0,foe])
                    donor=transform_facts.donor_slot(s.species[0,owner],self.source.vocabulary['species'].index('snorlax'))
                    np.testing.assert_array_equal(s.stats[0,foe,at,1:],s.stats[0,owner,donor,1:])
                    np.testing.assert_array_equal(s.moves[0,foe,at],s.moves[0,owner,donor])
                    np.testing.assert_array_equal(s.types[0,foe,at],s.base_types[0,owner,donor])
                    self.assertEqual(s.pp[0,foe,at].tolist(),[4,5,3,0]);self.assertEqual(s.transform_pp[0,foe,0],15)
                    c=hooks.Context(world.public)
                    prior_moves=np.array(c.trackers.observed_moves);prior_pp=np.array(c.trackers.pp_used);prior_belief=np.array(c.trackers.belief_mask)
                    moved=hooks.move(c,s,j.array([foe]),s.moves[:,foe,at,0],j.array([True]),pp_use=True,count_turn=False)
                    np.testing.assert_array_equal(moved.trackers.observed_moves,prior_moves)
                    np.testing.assert_array_equal(moved.trackers.pp_used,prior_pp)
                    np.testing.assert_array_equal(moved.trackers.belief_mask,prior_belief)
                    self.assertEqual(moved.trackers.copied_pp_used[0,foe,0],2)
                    refused=future_state.native_input_from_jax(s,0,self.metadata,module=NS())
                    self.assertIn('TRANSFORM_UNREPRESENTED',[gap.code for gap in refused.gaps])


if __name__=='__main__':unittest.main()
