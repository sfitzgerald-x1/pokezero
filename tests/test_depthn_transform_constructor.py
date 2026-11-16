"""Transform storage and exact-request constraints; authored fixtures, no saved roots."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as NS
import json
import unittest

import jax.numpy as j
import numpy as np

from pokezero.poke_engine_adapter import PokemonSpec,MoveSpec,SideSpec,BattleSpec
from gpu_engine import phase_d_transform as facts,transform
from gpu_engine.v5_hypothetical_state import map_spec
from gpu_engine.state import empty_state


def fixture(name='ditto',owner=0):
    base_moves=(MoveSpec('transform',15),) if name=='ditto' else (
        MoveSpec('softboiled',16),MoveSpec('transform',15),MoveSpec('psychic',16),MoveSpec('hiddenpowerice70',24))
    base=PokemonSpec(name,80,('normal',) if name=='ditto' else ('psychic',),100,200,101,102,103,104,105,
        base_moves,ability='limber' if name=='ditto' else 'synchronize',gender='N',base_types=('normal',) if name=='ditto' else ('psychic',))
    copied=replace(base,id='snorlax',types=('normal',),attack=201,defense=202,special_attack=203,special_defense=204,speed=205,
        ability='immunity',base_ability=base.ability,pre_transform=base,
        moves=(MoveSpec('tackle',4),MoveSpec('protect',5),MoveSpec('hiddenpowerwater63',3)))
    donor=replace(copied,pre_transform=None,base_ability='immunity',base_types=('normal',),maxhp=300)
    sides=[SideSpec((donor,)),SideSpec((donor,))]
    sides[owner]=SideSpec((copied,),volatile_statuses=('transformed','typechange'))
    vocabulary=dict(species=['','ditto','mew','snorlax'],types=['','normal','psychic','ice','water'],
        abilities=['','limber','synchronize','immunity'],items=[''],
        moves=['','transform','softboiled','psychic','hiddenpowerice','tackle','protect','hiddenpowerwater'])
    return BattleSpec(*sides),dict(vocabulary=vocabulary),base


class TransformMappingTests(unittest.TestCase):
    def test_copy_fallback_keeps_absent_slots_neutral(self):
        from pokezero.engine_world import _apply_transform
        spec,metadata,base=fixture()
        donor=replace(spec.side_two.pokemon[0],moves=(*spec.side_two.pokemon[0].moves,MoveSpec('none',0,True)))
        sides=dict(p1=SideSpec((base,)),p2=SideSpec((donor,)))
        copied=_apply_transform(sides,{'p1':'snorlax'})
        mapped=map_spec(BattleSpec(copied['p1'],copied['p2']),metadata=metadata,seed=7,
            turn=3,active_turns=(1,1),happiness=((255,),(255,)))
        self.assertFalse(mapped.refusals)
        self.assertEqual(mapped.state.pp[0,0,0].tolist(),[5,5,5,0])

    def test_retained_request_order_paid_base_pp_and_choice_lock(self):
        from test_phase_d_transform_constructor import source_fixture,request
        source,metadata=source_fixture();metadata['vocabulary']['items'].append('choiceband')
        for copied_transform in (False,True):
            spec,_,base=fixture();copy=spec.side_one.pokemon[0];donor=spec.side_two.pokemon[0]
            first='transform' if copied_transform else 'tackle'
            copied=(MoveSpec(first,4),*copy.moves[1:])
            copy=replace(copy,moves=copied,item='choiceband',pre_transform=replace(base,moves=(MoveSpec('transform',16),)))
            donor=replace(donor,moves=copied)
            spec=BattleSpec(SideSpec((donor,copy),active_index=1,volatile_statuses=('transformed','typechange'),last_used_move='move:0'),SideSpec((donor,)))
            mapped=map_spec(spec,metadata=metadata,seed=1,turn=2,active_turns=(1,1),happiness=((255,255),(255,)))
            self.assertFalse(mapped.refusals)
            before=request(source,'ditto');after=deepcopy(before)
            after['side']['pokemon'][0]['moves']=[first,'protect','hiddenpowerwater']
            after['side']['pokemon'][0]['item']='choiceband'
            after['active'][0]['moves']=[dict(id=mid,move=title,pp=pp,maxpp=5) for mid,title,pp in
                ((first,first,4),('protect','Protect',5),('hiddenpower','Hidden Power Water 63',3))]
            lines=('|start','|switch|p1a: ditto|ditto, L80|100/200','|switch|p2a: snorlax|snorlax, L80|100/100',
                '|request|'+json.dumps(before),'|move|p1a: ditto|Transform|p2a: snorlax','|-transform|p1a: ditto|p2a: snorlax',
                '|request|'+json.dumps(after))
            transform_id=metadata['vocabulary']['moves'].index('transform')
            public=NS(context=NS(moves=NS(last_kind=np.array([[2,1]]),last_id=np.array([[transform_id,0]])),
                active=NS(truant_known=np.zeros((1,2),bool),truant_loaf=np.zeros((1,2),bool))))
            catalog=NS(moves=NS(names=metadata['vocabulary']['moves']))
            result=facts.retained_copy(mapped.state,public,after,lines,metadata,catalog,0)
            self.assertEqual(result.transform_pp[0,0].tolist(),[15,0,0,0])
            self.assertEqual(int(result.active[0,0]),1)
            self.assertEqual((int(result.last_action_kind[0,0]),int(result.last_action_index[0,0])),(2,-1))
            self.assertEqual(int(result.last_move[0,0,1]),transform_id)
            self.assertEqual(int(result.choice_move[0,0,1]),transform_id if copied_transform else 0)

    def test_ditto_and_mew_both_seats_revert_to_base(self):
        for name in ('ditto','mew'):
            for owner in (0,1):
                spec,metadata,base=fixture(name,owner)
                kw=dict(metadata=metadata,seed=7,turn=3,active_turns=(1,1),happiness=((255,),(255,)))
                mapped=map_spec(spec,**kw);self.assertEqual(mapped.refusals,())
                state=mapped.state
                self.assertEqual(int(state.species[0,owner,0]),metadata['vocabulary']['species'].index(name))
                self.assertEqual(state.transform_pp[0,owner].tolist(),[m.pp for m in base.moves]+[0]*(4-len(base.moves)))
                self.assertEqual(state.hidden_power_power[0,owner,0],63)
                restored,_=transform.revert(state,j.array([owner]),j.array([True]))
                sides=[spec.side_one,spec.side_two];sides[owner]=replace(sides[owner],pokemon=(base,),volatile_statuses=())
                expected=map_spec(BattleSpec(*sides),**kw).state
                for field in state._fields:
                    np.testing.assert_array_equal(getattr(restored,field),getattr(expected,field),err_msg=field)
                host=map_spec(spec,**kw,arrays=lambda:{k:np.array(v) for k,v in empty_state(1,seed=7)._asdict().items()})
                for field in state._fields:
                    np.testing.assert_array_equal(getattr(host.state,field),getattr(state,field),err_msg=field)

    def test_transformed_bench_refuses(self):
        spec,metadata,base=fixture()
        side=replace(spec.side_one,pokemon=(base,spec.side_one.pokemon[0]),volatile_statuses=())
        result=map_spec(replace(spec,side_one=side),metadata=metadata,seed=7,turn=3,active_turns=(1,1),happiness=((255,255),(255,)))
        self.assertIn('transformed-bench:0:1',result.refusals)

    def test_request_keeps_copied_hidden_power_and_rejects_disagreeing_donors(self):
        _,metadata,_=fixture();v=metadata['vocabulary']
        rows=[dict(id='tackle',move='Tackle',pp=4),dict(id='protect',move='Protect',pp=5),
            dict(id='hiddenpower',move='Hidden Power Water 63',pp=3,disabled=True)]
        moves,pp,disabled,typ,power=facts.copied_request(dict(moves=rows),['tackle','protect','hiddenpowerwater'],v)
        self.assertEqual((typ,power),(4,63));self.assertEqual(pp.tolist(),[4,5,3,0]);self.assertTrue(disabled[2])
        facts.check_donor(moves,typ,power,moves,typ,power,v['moves'])
        for donor_moves,donor_type,donor_power in ((moves,3,63),(moves,4,70),([1,2,3,0],4,63)):
            with self.assertRaisesRegex(ValueError,'PUBLIC_TRANSFORM_DONOR_DISAGREES'):
                facts.check_donor(moves,typ,power,donor_moves,donor_type,donor_power,v['moves'])
        broken=deepcopy(rows);broken[2]['move']='Hidden Power Water'
        with self.assertRaisesRegex(ValueError,'OWN_TRANSFORM_HIDDEN_POWER_UNAVAILABLE'):
            facts.copied_request(dict(moves=broken),['tackle','protect','hiddenpowerwater'],v)

    def test_donor_must_be_unique(self):
        for species in ([1,2,3],[4,4,0]):
            with self.assertRaisesRegex(ValueError,'PUBLIC_TRANSFORM_DONOR_UNAVAILABLE'):facts.donor_slot(species,4)


class TransformHistoryTests(unittest.TestCase):
    start=('|switch|p1a: donor|Kecleon, L80|100/100','|switch|p2a: copy|Ditto, L80|100/100')
    def test_later_trace_confirmation_does_not_reveal_unknown_copied_owner_view(self):
        lines=('|switch|p1a: bird|Articuno, L80|100/100','|-ability|p1a: bird|Pressure|[silent]',
            '|switch|p2a: copy|Ditto, L80|100/100','|-transform|p2a: copy|p1a: bird',
            '|switch|p1a: tracer|Gardevoir, L80|100/100',
            '|-ability|p1a: tracer|Pressure|[from] ability: Trace|[of] p2a: copy')
        read=facts.history(lines)
        self.assertEqual(read['public']['p2','ditto'],'')
        self.assertEqual(read['public']['p1','gardevoir'],'pressure')

    def test_only_overrides_before_the_copy_refuse(self):
        copy='|-transform|p2a: copy|p1a: donor'
        change='|-start|p1a: donor|typechange|Water|[from] ability: Color Change'
        self.assertFalse(facts.history((*self.start,copy,change))['copies']['p2']['overridden'])
        self.assertTrue(facts.history((*self.start,change,copy))['copies']['p2']['overridden'])
        trace='|-ability|p1a: donor|Pressure|[from] ability: Trace|[of] p2a: copy'
        self.assertTrue(facts.history((*self.start,trace,copy))['copies']['p2']['overridden'])

    def test_departed_borrow_survives_until_return_and_private_pressure_stays_private(self):
        lines=(*self.start,'|-ability|p1a: donor|Pressure|[silent]','|-transform|p2a: copy|p1a: donor',
            '|switch|p2a: bench|Mew, L80|100/100')
        read=facts.history(lines)
        self.assertEqual(read['borrowed']['p2','ditto'],'transform')
        self.assertEqual(read['public']['p2','ditto'],'');self.assertNotIn('p2',read['copies'])
        returned=facts.history((*lines,'|switch|p2a: copy|Ditto, L80|100/100'))
        self.assertNotIn(('p2','ditto'),returned['borrowed'])
        traced=facts.history((*lines,'|-ability|p2a: copy|Pressure|[from] ability: Trace|[of] p1a: donor'))
        self.assertEqual(traced['borrowed']['p2','ditto'],'trace')

    def test_trace_reveals_the_donors_copy_and_arrival_restores_original_ability(self):
        lines=('|switch|p1a: donor|Articuno, L80|100/100','|switch|p2a: tracer|Porygon2, L80|100/100',
            '|-ability|p2a: tracer|Pressure|[from] ability: Trace|[of] p1a: donor')
        read=facts.history(lines)
        self.assertEqual(read['public']['p1','articuno'],'pressure')
        self.assertEqual(read['public']['p2','porygon2'],'pressure')
        read=facts.history((*lines,'|switch|p2a: bench|Mew, L80|100/100','|switch|p2a: tracer|Porygon2, L80|100/100'))
        self.assertEqual(read['public']['p2','porygon2'],'trace')
        copied=facts.history((*lines,'|switch|p2a: copy|Ditto, L80|100/100','|-transform|p2a: copy|p1a: donor'))
        self.assertEqual(copied['public']['p2','ditto'],'pressure')

    def test_field_and_explicit_activation_ability_reveals(self):
        for species,line,ability in (
            ('Tyranitar','|-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: donor','sandstream'),
            ('Arbok','|-activate|p1a: donor|ability: Shed Skin','shedskin'),
            ('Arcanine','|-start|p1a: donor|ability: Flash Fire','flashfire')):
            lines=(f'|switch|p1a: donor|{species}, L80|100/100','|switch|p2a: copy|Ditto, L80|100/100',line,
                '|-transform|p2a: copy|p1a: donor')
            self.assertEqual(facts.history(lines)['public']['p2','ditto'],ability)

    def test_truant_uses_the_transformers_clock_or_refuses(self):
        arrays={name:np.array(value) for name,value in empty_state(1)._asdict().items()}
        arrays['transformed'][0,1]=True;arrays['ability'][0,1,0]=1
        port=NS(known=np.array([[False,False]]),loaf=np.array([[False,True]]))
        public=NS(public_port=port);v=dict(abilities=['','truant'])
        with self.assertRaisesRegex(ValueError,'PUBLIC_TRANSFORM_TRUANT_PHASE_UNAVAILABLE'):facts.restore_truant(arrays,public,v)
        port.known[0,1]=True;facts.restore_truant(arrays,public,v)
        self.assertTrue(arrays['truant_loaf'][0,1,0])


if __name__=='__main__':unittest.main()
