"""Final-tail authored public facts/storage fixtures; no game/native oracle."""
import copy
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
import jax.numpy as j
import numpy as np
from gpu_engine import phase_d_tail as tail
from gpu_engine.phase_d_guard_diagnostic import payload_checks
from gpu_engine.phase_d_constructor_rules import own_pp_carry
from gpu_engine.v5_hypothetical_state import map_spec
from gpu_engine.state import legal_actions
from gpu_engine.transition import sample_chance_inputs
from gpu_engine.sustain import wish_residual
from pokezero.poke_engine_adapter import PokemonSpec,MoveSpec,SideSpec,BattleSpec


def fixture(owner='p1',paid=False):
    party=[dict(ident=owner+': '+n,details=n+', L100',condition='0 fnt' if i==0 else '100/100',active=i==0,moves=['tackle']) for i,n in enumerate('abcdef')]
    request=dict(forceSwitch=[True],side=dict(id=owner,pokemon=party))
    foe='p2' if owner=='p1' else 'p1'
    payload=dict(turn=4,selfPlayer=owner,selfRequestKind='force-switch',pendingBatonPassSides=[],wishSetTurns={},
        sides={owner:dict(pokemon=[dict(species='a',active=True,condition='0 fnt')],volatiles=[]),
               foe:dict(pokemon=[dict(species='g',active=True,condition='55/100')],volatiles=[])})
    lines=('|start',f'|switch|{owner}a: a|a, L100|100/100',f'|switch|{foe}a: g|g, L100|100/100',
        '|turn|3','|upkeep','|turn|4',f'|move|{foe}a: g|Tackle|{owner}a: a',f'|-damage|{owner}a: a|0 fnt',f'|faint|{owner}a: a')
    if paid:lines+=('|upkeep',)
    return request,payload,lines


def storage():
    def side(names,dead=False):
        return SideSpec(tuple(PokemonSpec(id=n,level=100,types=('normal',),base_types=('normal',),hp=0 if dead and i==0 else 100,
            maxhp=101 if i==0 else 303,attack=100,defense=100,special_attack=100,special_defense=100,speed=100,moves=(MoveSpec('tackle',16),)) for i,n in enumerate(names)),force_switch=dead)
    spec=BattleSpec(side('abcdef',True),side('ghijkl'))
    meta=dict(vocabulary=dict(species=('',)+tuple('abcdefghijkl'),types=('','normal'),abilities=('',),items=('',),moves=('','tackle')))
    return spec,meta


class ReplacementFacts(unittest.TestCase):
    def test_public_consumed_queue_both_seats_and_paid_residual(self):
        for owner in ('p1','p2'):
            for paid in (False,True):
                request,payload,lines=fixture(owner,paid);original=copy.deepcopy(payload)
                phase=tail.replacement_phase(request,payload,lines)
                self.assertEqual(phase['queued_actor'],-1);self.assertEqual(phase['pending_residual'],not paid)
                self.assertEqual(phase['ages'],(2,2));self.assertEqual(payload,original)
                checks=payload_checks(request,payload,lines,owner=int(owner[1])-1).rows
                self.assertFalse(any(row['status']=='TRIGGERED' for row in checks),checks)

    def test_missing_queue_living_pivot_partial_residual_and_turn_refuse(self):
        request,payload,lines=fixture()
        for modified,expected in ((tuple(x for x in lines if not x.startswith('|move|')),'QUEUE_UNAVAILABLE'),
                (lines[:-1]+('|-damage|p1a: a|0 fnt|[from] psn',),'PARTIAL_RESIDUAL'),
                (tuple(x for x in lines if not x.startswith('|turn|')),'TURN_UNAVAILABLE'),
                (lines+('|drag|p2a: h|h, L100|55/100',),'QUEUE_UNAVAILABLE')):
            with self.assertRaisesRegex(ValueError,expected):tail.replacement_phase(request,payload,modified)
        living=copy.deepcopy(request);living['side']['pokemon'][0]['condition']='100/100'
        with self.assertRaisesRegex(ValueError,'PHASE_UNAVAILABLE'):tail.replacement_phase(living,payload,lines)
        payload['pendingBatonPassSides']=['p1']
        with self.assertRaisesRegex(ValueError,'PIVOT_UNSUPPORTED'):tail.replacement_phase(request,payload,lines)

    def test_actual_storage_cursor_hp_status_flags_and_frozen_capsule_restored(self):
        request,payload,lines=fixture();phase=tail.replacement_phase(request,payload,lines);spec,meta=storage()
        spec=replace(spec,side_one=replace(spec.side_one,last_used_move='move:0'))
        projected,cursors=tail.mapping_spec(spec,phase)
        self.assertEqual(projected.side_one.active_index,1);self.assertEqual(spec.side_one.active_index,0)
        self.assertEqual([mon.hp for mon in projected.side_one.pokemon],[mon.hp for mon in spec.side_one.pokemon])
        mapped=map_spec(projected,metadata=meta,seed=71,turn=4,active_turns=(1,1),happiness=((255,)*6,)*2)
        self.assertIsNotNone(mapped.state,mapped.refusals);keys=np.asarray(mapped.state.keys).copy();checks=[]
        facts=dict(replacement=phase,fresh_substitutes={},wish_set_turns={},turn=4)
        state=tail.restore(mapped.state,spec,cursors,facts,metadata=meta,checks=checks)
        self.assertEqual(np.asarray(state.active).tolist(),[[0,0]]);self.assertEqual(int(state.hp[0,0,0]),0)
        self.assertEqual(np.asarray(state.forced_switch).tolist(),[[True,False]])
        self.assertEqual(int(state.last_move[0,0,0]),1);self.assertEqual(int(state.last_move[0,0,1]),0)
        self.assertTrue(bool(state.pending_residual[0]));self.assertEqual(int(state.queued_actor[0]),-1)
        self.assertEqual(int(state.turn[0]),4);self.assertEqual(np.asarray(state.active_turns).tolist(),[[2,2]])
        np.testing.assert_array_equal(state.keys,keys);np.testing.assert_array_equal(state.residual_draws,sample_chance_inputs(state).draws)
        self.assertFalse(np.asarray(legal_actions(state))[0,0,:4].any());self.assertTrue(np.asarray(legal_actions(state))[0,0,4:].all())
        self.assertIn('SAMPLED_HYPOTHETICAL_FUTURE',checks[0]['randomness'])
        self.assertFalse(np.asarray(state.queued_actions).any());self.assertFalse(np.asarray(state.queued_stall).any())
        # No unrelated active effects are erased by storage projection.
        bad=replace(spec,side_one=replace(spec.side_one,boosts={'attack':1}))
        with self.assertRaisesRegex(ValueError,'FAINTED_EFFECT'):tail.mapping_spec(bad,phase)
        bad=replace(spec,side_two=replace(spec.side_two,slow_uturn_move=True))
        with self.assertRaisesRegex(ValueError,'QUEUE_UNAVAILABLE'):tail.mapping_spec(bad,phase)

    def test_current_damage_preserves_shown_exact_own_and_unknown_halves(self):
        from gpu_engine.engine_device_damage_history import History
        for owner in (0,1):
            zero=j.zeros((1,2,2,2),j.float32);history=History(j.zeros((1,2),j.int32),j.ones((1,2),j.int32),zero,zero)
            dealt=j.array([[[.1,.001],[.2,.002]]],j.float32);taken=j.array([[[.3,.003],[.4,.004]]],j.float32)
            result=tail.carry_current_damage(history,dealt,taken,owner);foe=1-owner
            np.testing.assert_array_equal(result.dealt[0,owner,0],dealt[0,owner])
            np.testing.assert_array_equal(result.dealt[0,foe,1],dealt[0,foe])
            np.testing.assert_array_equal(result.taken[0,owner,1],taken[0,owner])
            np.testing.assert_array_equal(result.taken[0,foe,0],taken[0,foe])
            self.assertFalse(np.asarray(result.dealt[0,owner,1]).any());self.assertFalse(np.asarray(result.taken[0,foe,1]).any())



class FreshSubstituteAndWish(unittest.TestCase):
    def fixture(self):
        _,payload,lines=fixture();payload['selfRequestKind']='move'
        payload['sides']['p1']['pokemon'][0]['condition']='100/100'
        data=payload['sides']['p1'];data.update(volatiles=['substitute'],substituteHealthState='full',substituteOriginSpecies='a',substituteDepletion=0,substituteMinDepletion=0)
        lines+=('|move|p1a: a|Substitute|p1a: a','|-start|p1a: a|Substitute')
        return payload,lines

    def test_fresh_positive_provenance_only_value_projection_and_exact_health(self):
        payload,lines=self.fixture();before=copy.deepcopy(payload);fresh=tail.fresh_substitutes(payload,lines)
        self.assertEqual(fresh,{'p1':'a'});projected=tail.defer_substitutes(payload,fresh)
        self.assertEqual(projected['sides']['p1']['volatiles'],[]);self.assertEqual(payload,before)
        spec,meta=storage();living=replace(spec.side_one,pokemon=(replace(spec.side_one.pokemon[0],hp=100),)+spec.side_one.pokemon[1:],force_switch=False)
        mapped=map_spec(replace(spec,side_one=living),metadata=meta,seed=71,turn=4,active_turns=(1,1),happiness=((255,)*6,)*2)
        state=tail.restore(mapped.state,spec,(),dict(replacement=None,fresh_substitutes=fresh,wish_set_turns={},turn=4),metadata=meta,checks=[])
        self.assertEqual(int(state.substitute_hp[0,0,0]),101//4)
        for event in ('|-activate|p1a: a|Substitute|[damage]','|move|p1a: a|Baton Pass|p1a: a','|switch|p1a: b|b, L100|100/100'):
            with self.assertRaisesRegex(ValueError,'NOT_FRESH_OR_PASSED'):tail.fresh_substitutes(payload,lines+(event,))
        for field,value in (('substituteHealthState','unknown'),('substituteHealthState','exact'),('substituteDepletion',1),('substituteOriginSpecies','b'),('substituteMinDepletion',True)):
            bad=copy.deepcopy(payload);bad['sides']['p1'][field]=value
            with self.assertRaisesRegex(ValueError,'NOT_FRESH_OR_PASSED'):tail.fresh_substitutes(bad,lines)

    def test_wish_recipient_half_and_public_timing_no_caster_size_input(self):
        spec,meta=storage();living=replace(spec.side_one,pokemon=(replace(spec.side_one.pokemon[0],hp=10),)+spec.side_one.pokemon[1:],force_switch=False)
        mapped=map_spec(replace(spec,side_one=living),metadata=meta,seed=71,turn=4,active_turns=(1,1),happiness=((255,)*6,)*2)
        state=tail.restore(mapped.state,spec,(),dict(replacement=None,fresh_substitutes={},wish_set_turns={'p1':3},turn=4),metadata=meta,checks=[])
        self.assertEqual(int(state.wish_counter[0,0]),1);self.assertEqual(int(state.wish_amount[0,0]),50)
        # The frozen residual reads the eventual recipient, not stored caster/amount.
        healed=wish_residual(j.array([10]),j.array([303]),state.wish_counter[:,0],eligible=True)
        self.assertEqual(int(healed.hp[0]),161);self.assertEqual(int(healed.counter[0]),0)
        self.assertEqual(tail.wish_facts({'turn':4,'wishSetTurns':{'p1':3}}),{'p1':3})
        for start in (True,1,5):
            with self.assertRaisesRegex(ValueError,'TIMING_UNAVAILABLE'):tail.wish_facts({'turn':4,'wishSetTurns':{'p1':start}})

class FullProtocolTail(unittest.TestCase):
    def test_final_phase_substitute_and_wish_exist_before_any_public_carry_escape(self):
        import hashlib,json
        from unittest.mock import patch
        from test_phase_d_batch2_facts import ProductionTransport,packet,request
        from gpu_engine import phase_d_protocol_world as protocol
        public,start,inputs,physical,world=ProductionTransport().fixture()
        payload=copy.deepcopy(public.materialization_payload);payload.update(turn=3,selfPlayer='p1',selfRequestKind='force-switch',wishSetTurns={'p1':2})
        for seat,data in payload['sides'].items():
            for k,row in enumerate(data['pokemon']):row.update(active=k==(1 if seat=='p1' else 0),condition='0 fnt' if seat=='p1' and k==1 else '100/100')
        payload['sides']['p2'].update(volatiles=['substitute'],substituteHealthState='full',substituteOriginSpecies='g',substituteDepletion=0,substituteMinDepletion=0)
        forced=packet(active=1);forced.pop('active');forced['forceSwitch']=[True];forced['side']['pokemon'][1]['condition']='0 fnt'
        lines=tuple(line for line in inputs.lines if not line.startswith('|-status|'))+(
            '|move|p2a: g|Substitute|p2a: g','|-start|p2a: g|Substitute','|turn|3','|move|p2a: g|Tackle|p1a: b','|-damage|p1a: b|0 fnt','|faint|p1a: b',request(forced))
        own=copy.deepcopy(inputs.root.own);own['status'][:]=0;own['hp'][1]=0;own['forced_switch'][()]=True
        own['active_turns'][()]=1;own['wish_counter'][()]=1;own['wish_amount'][()]=50
        carry=inputs.root.public;carry.status[:]=0
        detached=replace(inputs.root,own=own);state=NS(self_request=forced,belief_engine=public.context.public_materialization_state.belief_engine)
        public=replace(public,context=NS(public_materialization_state=state),materialization_payload=payload,
            owner_lines_sha256=hashlib.sha256(''.join(x+'\n' for x in lines).encode()).hexdigest())
        inputs=replace(inputs,root=detached,lines=lines)
        physical=physical._replace(hp=physical.hp.at[0,0,1].set(0),active=physical.active.at[0,0].set(0))
        def member(name,dead=False):
            return PokemonSpec(id=name,level=100,types=('normal',),base_types=('normal',),hp=0 if dead else 100,maxhp=100,
                attack=1,defense=1,special_attack=1,special_defense=1,speed=1,moves=(MoveSpec('tackle',16),))
        spec=BattleSpec(SideSpec(tuple(member(n,n=='b') for n in 'abcdef'),active_index=1,force_switch=True),SideSpec(tuple(member(n) for n in 'ghijkl')))
        world=NS(spec=spec,party_species=world.party_species)
        def mapper(projected,**kwargs):
            self.assertEqual(projected.side_one.active_index,0);self.assertEqual(projected.side_one.pokemon[1].hp,0)
            return NS(state=physical)
        def rebase(*args,state,**kwargs):
            self.assertEqual(int(state.active[0,0]),1);self.assertEqual(int(state.hp[0,0,1]),0)
            self.assertTrue(bool(state.forced_switch[0,0]));self.assertTrue(bool(state.pending_residual[0]))
            self.assertEqual(int(state.turn[0]),3);self.assertEqual(np.asarray(state.active_turns).tolist(),[[1,1]])
            self.assertEqual(int(state.substitute_hp[0,1,0]),25);self.assertEqual(int(state.wish_counter[0,0]),1)
            self.assertEqual(np.asarray(state.pp)[0,0,:,0].tolist(),[4,5,16,16,16,16])
            self.assertFalse(np.asarray(legal_actions(state))[0,0,:4].any())
            return 'literal-restored-carry'
        unpack=lambda team:tuple(NS(species=part.split('|')[0],ability='',item='') for part in team.split(']'))
        with patch.object(protocol,'battle_spec_from_payload',return_value=world),patch.object(protocol,'map_spec',side_effect=mapper), \
                patch.object(protocol,'unpack_team',side_effect=unpack),patch.object(protocol.public_worlds,'_revealed',return_value=()), \
                patch.object(protocol.states,'_rebase',side_effect=rebase):
            made=protocol.materialize_protocol(public,start,inputs=inputs,seed=17)
        self.assertIsNotNone(made.state,made.refusals);self.assertEqual(made.public,'literal-restored-carry')

class B2ReplacementTransport(unittest.TestCase):
    def test_original_b2_faint_request_exact_own_and_current_damage_both_seats(self):
        import json
        from test_phase_d_device_worlds import WorldsTests
        from pokezero.v5_contract import DecisionAddress
        from gpu_engine import phase_d_live_worlds as live
        WorldsTests.setUpClass();helper=WorldsTests()
        for owner in (0,1):
            adapter=helper.adapter(owner);packet=helper.request(owner);own='p'+str(owner+1);foe='p'+str(2-owner)
            wire=(f'|move|{foe}a: Beta|Tackle|{own}a: Alpha',f'|-damage|{own}a: Alpha|0 fnt',f'|faint|{own}a: Alpha')
            adapter.feed(wire);packet.pop('active');packet['forceSwitch']=[True];packet['side']['pokemon'][0]['condition']='0 fnt'
            payload=dict(turn=1,selfPlayer=own,selfRequestKind='force-switch',pendingBatonPassSides=[],wishSetTurns={},
                sides={own:dict(pokemon=[dict(species='Alpha',condition='0 fnt',active=True)],volatiles=[]),
                       foe:dict(pokemon=[dict(species='Beta',condition='100/100',active=True)],volatiles=[])})
            lines=('|start',f'|switch|{own}a: Alpha|Alpha, L80|291/291',f'|switch|{foe}a: Beta|Beta, L80|100/100','|turn|1')+wire+('|request|'+json.dumps(packet),)
            facts=tail.facts(NS(materialization_payload=payload),packet,lines)
            with self.assertRaisesRegex(ValueError,'OWN_COMPLETE_ORDINARY'):
                live.from_b2(adapter,packet,helper.source,helper.metadata,address=DecisionAddress(1,0,owner))
            root=live.from_b2(adapter,packet,helper.source,helper.metadata,address=DecisionAddress(1,0,owner),defer_sleep=True,protocol_facts=facts)
            self.assertEqual(int(root.own['hp'][0]),0);self.assertTrue(bool(root.own['forced_switch']))
            self.assertEqual(int(root.own['active']),0);self.assertEqual(int(root.own['active_turns']),1)
            np.testing.assert_array_equal(root.public.damage_history.taken[0,owner,1],adapter.state.context.damage.current_taken[0,owner])
            self.assertFalse(np.asarray(root.public.damage_history.taken[0,1-owner,1]).any())
            for wrong,kind in (({**facts,'fresh_substitutes':{own:'alpha'}},'PUBLIC_SUBSTITUTE_B2_PRESENCE_DISAGREEMENT'),
                    ({**facts,'wish_set_turns':{own:0}},'PUBLIC_WISH_PUBLIC_HISTORY_DISAGREEMENT'),
                    ({**facts,'turn':2},'PUBLIC_TAIL_TURN_DISAGREEMENT')):
                with self.assertRaisesRegex(ValueError,kind):
                    live.from_b2(adapter,packet,helper.source,helper.metadata,address=DecisionAddress(1,0,owner),defer_sleep=True,protocol_facts=wrong)

if __name__=='__main__':unittest.main()
