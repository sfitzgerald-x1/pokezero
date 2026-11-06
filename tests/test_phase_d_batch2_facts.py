"""Owning/public fact transport controls, no game or native oracle execution."""
import copy
from dataclasses import replace
import json
import hashlib
import random
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import jax.numpy as j
import numpy as np
from gpu_engine.phase_d_constructor_rules import own_pp_carry,sleep_support
from gpu_engine import phase_d_protocol_world as protocol
from gpu_engine import phase_d_live_worlds as live
from gpu_engine.state import empty_state,legal_actions
from gpu_engine.mechanics import before_move
from gpu_engine import entry

NAMES=tuple('abcdef')
def packet(active=0,pp=5,move='tackle',order=NAMES):
    return dict(side=dict(id='p1',pokemon=[dict(ident='p1: '+name,details=name+', L100',condition='100/100',active=NAMES.index(name)==active,moves=[move]) for name in order]),
        active=[dict(moves=[dict(id=move,pp=pp,maxpp=16,disabled=False)])])
def request(p):return '|request|'+json.dumps(p)
def carry_fixture():
    species=np.arange(1,13,dtype=np.int32).reshape(1,2,6)
    features={k:np.zeros((1,2,6),np.int32) for k in ('sleep_turns','rest_sleep','sleep_skipped_turns','sleep_cant_pending')}
    return NS(species=species,present=np.ones((1,2,6),bool),status=np.zeros((1,2,6),np.int32),mon_features=features,
        item_removed=np.zeros((1,2,6),bool),shown_item=np.ones((1,2,6),np.int32))
def metadata():return dict(vocabulary=dict(species=('',)+tuple('abcdefghijkl'),abilities=('','pressure','earlybird')))

class OwnPPCarry(unittest.TestCase):
    def test_bench_latest_exact_snapshot_public_pressure_and_never_active(self):
        first=packet();last=packet(active=1,pp=12,order=tuple('bacdef'))
        lines=('|start','|switch|p1a: a|a, L100|100/100','|switch|p2a: x|x, L100|100/100','|-ability|p2a: x|Pressure',request(first),
            '|move|p1a: a|Tackle|p2a: x','|drag|p1a: b|b, L100|100/100',request(last))
        actual=own_pp_carry(lines,last,{'tackle':16})
        self.assertEqual([v[0] for v in actual],[12,3,16,16,16,16])
        # Later active packet replaces ALL earlier cost arithmetic, never max PP.
        newer=packet(pp=2)
        self.assertEqual(own_pp_carry(lines+(request(newer),),newer,{'tackle':16})[0][0],2)

    def test_called_move_and_locked_charge_release_do_not_spend_twice(self):
        first=packet(move='solarbeam');locked=packet(move='solarbeam');locked['active'][0]['moves']=[dict(id='solarbeam',disabled=False)]
        last=packet(active=1,move='solarbeam')
        lines=('|start','|switch|p1a: a|a, L100|100/100',request(first),'|move|p1a: a|Solar Beam||[still]',
            '|-prepare|p1a: a|Solar Beam',request(locked),'|move|p1a: a|Solar Beam|p2a: x|[from] lockedmove',
            '|drag|p1a: b|b, L100|100/100',request(last))
        self.assertEqual(own_pp_carry(lines,last,{'solarbeam':16})[0][0],4)
        forced=copy.deepcopy(first);forced.pop('active');forced['forceSwitch']=[True]
        self.assertEqual(own_pp_carry(lines[:6]+(request(forced),)+lines[6:],last,{'solarbeam':16})[0][0],4)
        first=packet(move='sleeptalk');last=packet(active=1,move='sleeptalk')
        lines=('|start','|switch|p1a: a|a, L100|100/100',request(first),'|move|p1a: a|Sleep Talk|p1a: a',
            '|move|p1a: a|Tackle|p2a: x|[from] move: Sleep Talk','|switch|p1a: b|b, L100|100/100',request(last))
        self.assertEqual(own_pp_carry(lines,last,{'sleeptalk':16})[0][0],4)

    def test_missing_history_or_pp_mutation_refuses_until_exact_refresh(self):
        first=packet();last=packet(active=1)
        lines=('|start','|switch|p1a: a|a, L100|100/100',request(first),'|move|p2a: x|Spite|p1a: a',
            '|switch|p1a: b|b, L100|100/100',request(last))
        with self.assertRaisesRegex(ValueError,'OWN_PP_NONMOVE_CHANGE_UNAVAILABLE'):own_pp_carry(lines,last,{'tackle':16})
        refreshed=packet(pp=1)
        self.assertEqual(own_pp_carry(lines+(request(refreshed),),refreshed,{'tackle':16})[0][0],1)
        with self.assertRaisesRegex(ValueError,'COMPLETE_UNIQUE_START'):own_pp_carry(lines[1:],last,{'tackle':16})
        foreign=copy.deepcopy(last);foreign['side']['id']='p2'
        with self.assertRaises(ValueError):own_pp_carry(lines+(request(foreign),),last,{'tackle':16})

class SleepFacts(unittest.TestCase):
    def test_declared_support_and_one_unit_early_bird_refunds(self):
        for count in range(3):self.assertEqual(sleep_support(count,True),(3-count,))
        self.assertEqual(sleep_support(0,False),(2,3,4,5));self.assertEqual(sleep_support(1,False),(1,2,3,4))
        self.assertEqual(sleep_support(1,False,early_bird=True),(1,2,3))
        self.assertEqual(sleep_support(0,True,early_bird=True,refunded=1,attempt_refunds=(0,)),(2,))
        self.assertEqual(sleep_support(0,False,early_bird=True,refunded=1,attempt_refunds=(0,)),(2,3,4))
        self.assertEqual(sleep_support(1,True,early_bird=True,skipped=1),(1,))
        with self.assertRaisesRegex(ValueError,'EMPTY'):sleep_support(3,True)
        with self.assertRaisesRegex(ValueError,'UNSETTLED'):sleep_support(0,True,pending=True)
        with self.assertRaisesRegex(ValueError,'CHRONOLOGY_UNAVAILABLE'):sleep_support(0,False,refunded=1)
        # Two survived Early Bird attempts separated by one refund rule out initial2..3.
        self.assertEqual(sleep_support(1,False,refunded=1,attempt_rates=(2,2),attempt_refunds=(0,1)),(1,2))

    def test_full_history_preserves_attempts_refunds_and_public_ability(self):
        p=packet()
        lines=('|start','|switch|p1a: a|a, L100|100/100',request(p),'|-status|p1a: a|slp|[from] move: Rest',
            '|cant|p1a: a|slp','|move|p1a: a|Sleep Talk|p1a: a','|upkeep','|switch|p1a: b|b, L100|100/100',
            '|switch|p1a: a|a, L100|100/100 slp',request(p))
        fact=protocol._sleep_origins(lines)[('p1','a')]
        self.assertEqual((len(fact['attempt_abilities']),fact['refunded'],fact['skipped'],fact['pending']),(1,1,0,False))
        self.assertEqual(sleep_support(0,True,early_bird=True,refunded=fact['refunded'],attempt_refunds=fact['attempt_refunds']),(2,))
        # Only actual public ability declarations alter attempt factors.
        fact=protocol._sleep_origins(lines[:4]+('|-ability|p1a: a|Early Bird',)+lines[4:])[('p1','a')]
        self.assertEqual(fact['attempt_abilities'],['earlybird'])

    def test_deferred_bridge_preserves_source_and_unknown_status(self):
        payload=dict(sides=dict(p1=dict(pokemon=[dict(species='a',condition='10/10 slp',restSleepAttempts=1)]),p2=dict(pokemon=[dict(species='g',condition='33/100 mystery')])))
        original=copy.deepcopy(payload);projected,sleepers=protocol._defer_sleep(payload)
        self.assertEqual(payload,original);self.assertEqual(sleepers,((0,'a'),))
        self.assertEqual(projected['sides']['p1']['pokemon'][0],dict(species='a',condition='10/10'))
        self.assertEqual(projected['sides']['p2'],payload['sides']['p2'])

    def test_both_seats_restored_before_legal_or_sleep_usable_checks(self):
        physical=empty_state(1)._replace(species=j.arange(1,13,dtype=j.int32).reshape(1,2,6),hp=j.ones((1,2,6),j.int32),
            moves=j.ones((1,2,6,4),j.int32),pp=j.full((1,2,6,4),10,j.int32),ability=j.ones((1,2,6),j.int32),done=j.array([False]))
        carry=carry_fixture();carry.status[0,:,0]=5;carry.mon_features['rest_sleep'][0,:,0]=1
        roots=NS(public=carry);origins={(side,name):dict(rest=True,attempt_abilities=[],attempt_refunds=[],refunded=0,skipped=0,pending=False) for side,name in (('p1','a'),('p2','g'))}
        restored,checks=protocol._restore_sleep(physical,roots,metadata(),((0,'a'),(1,'g')),origins,random.Random(3),{(0,'a'):'pressure',(1,'g'):'pressure'})
        self.assertEqual(np.asarray(restored.status)[0,:,0].tolist(),[5,5]);self.assertEqual(np.asarray(restored.status_counter)[0,:,0].tolist(),[3,3])
        self.assertEqual(np.asarray(restored.rest_origin)[0,:,0].tolist(),[True,True]);self.assertTrue(np.asarray(legal_actions(restored))[0,:,:4].all())
        for seat in (0,1):
            status=restored.status[0,seat,0];clock=restored.status_counter[0,seat,0]
            self.assertFalse(bool(before_move(status,clock).can_act));self.assertTrue(bool(before_move(status,clock,sleep_usable=True).can_act))
            self.assertTrue(bool(before_move(status,1).can_act))
        self.assertTrue(all(c['validation']=='HIDDEN_CLOCK_SUPPORT_NOT_EXACT_OWN_EQUALITY' for c in checks))
        # Missing infliction/public identity is a refusal, never healthy zero.
        with self.assertRaisesRegex(ValueError,'STINT_UNAVAILABLE'):protocol._restore_sleep(physical,roots,metadata(),((0,'a'),),{},random.Random(1),{(0,'a'):'pressure'})
        with self.assertRaisesRegex(ValueError,'UNRESTORED_STATUS'):protocol._restore_sleep(restored,roots,metadata(),(),{},random.Random(1),{})

    def test_induced_hidden_support_uses_local_world_rng_without_mutating_carry(self):
        physical=empty_state(1)._replace(species=j.arange(1,13,dtype=j.int32).reshape(1,2,6))
        carry=carry_fixture();carry.status[0,1,0]=5;carry.mon_features['sleep_turns'][0,1,0]=1
        before=copy.deepcopy(carry.mon_features)
        origins={('p2','g'):dict(rest=False,attempt_abilities=[None],attempt_refunds=[0],refunded=0,skipped=0,pending=False)}
        values=[]
        for seed in (7,7,8):
            state,checks=protocol._restore_sleep(physical,NS(public=carry),metadata(),((1,'g'),),origins,random.Random(seed),{(1,'g'):'earlybird'})
            self.assertEqual(checks[0]['support'],[1,2,3]);values.append(int(state.status_counter[0,1,0]))
        self.assertEqual(values[0],values[1]);self.assertTrue(set(values)<={1,2,3})
        for key,value in before.items():np.testing.assert_array_equal(carry.mon_features[key],value)


class RemovedItems(unittest.TestCase):
    def test_matching_positive_removal_and_unrelated_blockers(self):
        carry=carry_fixture();carry.item_removed[0,0,0]=True;carry.item_removed[0,1,1]=True
        payload=dict(sides={s:dict(pokemon=[dict(species=n,condition='100/100')],volatiles=[],materializationBlockers=['item-state-removed:'+n]) for s,n in (('p1','a'),('p2','h'))})
        old=copy.deepcopy(payload);old_reveals=carry.shown_item.copy()
        removed=protocol._removed_items(payload,NS(public=carry),metadata(),())
        self.assertEqual(removed,{'p1':('a',),'p2':('h',)});protocol._preflight(payload,removed)
        self.assertEqual(protocol._public_removals(NS(public=carry),metadata()),removed)
        self.assertEqual(payload,old);np.testing.assert_array_equal(carry.shown_item,old_reveals)
        payload['sides']['p2']['materializationBlockers'].append('toxic-stage-unknown')
        with self.assertRaisesRegex(ValueError,'public-materialization-blocker'):protocol._preflight(payload,removed)
        with self.assertRaisesRegex(ValueError,'PUBLIC_ITEM_SWAP_NOT_SUPPORTED'):
            protocol._removed_items(old,NS(public=carry),metadata(),('|-item|p1a: a|Leftovers|[from] move: Trick',))
        payload['sides']['p2']['materializationBlockers']=['item-state-removed:g']
        with self.assertRaisesRegex(ValueError,'public-materialization-blocker'):protocol._preflight(payload,removed)

class ProductionTransport(unittest.TestCase):
    def fixture(self):
        from pokezero.v5_contract import DecisionAddress
        from pokezero.v5_teacher_public_root import PublicRoot
        from pokezero.env import BattleStartOverride
        from gpu_engine.device_public_worlds import Root,OWN_MON_FIELDS,OWN_SIDE_FIELDS
        physical=empty_state(1)._replace(species=j.arange(1,13,dtype=j.int32).reshape(1,2,6),
            hp=j.full((1,2,6),100,j.int32),max_hp=j.full((1,2,6),100,j.int32),
            moves=j.zeros((1,2,6,4),j.int32).at[:,:,:,0].set(1),pp=j.full((1,2,6,4),16,j.int32),
            active=j.array([[1,0]],j.int32))
        own={name:np.array(getattr(physical,name)[0,0]) for name in OWN_MON_FIELDS+OWN_SIDE_FIELDS}
        own['status'][0]=5
        carry=carry_fixture();carry.status[0,:,0]=5;carry.mon_features['rest_sleep'][0,:,0]=1;carry.item_removed[0,1,1]=True
        first=packet(pp=5);last=packet(active=1,pp=5)
        lines=('|start','|switch|p1a: a|a, L100|100/100','|switch|p2a: g|g, L100|100/100',request(first),
            '|-status|p1a: a|slp|[from] move: Rest','|-status|p2a: g|slp|[from] move: Rest',
            '|move|p1a: a|Tackle|p2a: g','|switch|p1a: b|b, L100|100/100',request(last))
        payload=dict(turn=2,sides={seat:dict(pokemon=[dict(species=n,condition='100/100'+(' slp' if i==0 else '')) for i,n in enumerate(names)],
            volatiles=[],materializationBlockers=['item-state-removed:h'] if seat=='p2' else []) for seat,names in (('p1','abcdef'),('p2','ghijkl'))})
        address=DecisionAddress(1,0,0);source=NS(source_hash='same')
        state=NS(self_request=last,belief_engine=NS(set_source=NS(metadata=source)))
        public=PublicRoot(address,hashlib.sha256(''.join(x+'\n' for x in lines).encode()).hexdigest(),NS(public_materialization_state=state),payload,())
        view=dict(present=np.ones((1,6),bool),species=np.arange(7,13).reshape(1,6),hp_percent=np.full((1,6),100),
            revealed_moves=np.zeros((1,6,4),np.int32),move_pp_used=np.zeros((1,6,4),np.int32),
            mon_features=dict(toxic_stage=np.zeros((1,6),np.int32),stall_counter=np.zeros((1,6),np.int32)))
        detached=Root(address,own,view,carry,'same')
        meta=metadata();meta['vocabulary']['moves']=('','tackle');meta['moves']=[dict(pp=0),dict(pp=10)]
        catalog=NS(source=source,dex=object())
        teams={seat:']'.join('|'.join([n]+['']*11) for n in names) for seat,names in (('p1','abcdef'),('p2','ghijkl'))}
        start=BattleStartOverride(teams)
        world=NS(spec=NS(side_one=NS(pokemon=(None,)*6),side_two=NS(pokemon=(None,)*6)),party_species={'p1':tuple('abcdef'),'p2':tuple('ghijkl')})
        inputs=protocol.ProtocolInputs(detached,lines,catalog,meta,object())
        return public,start,inputs,physical,world

    def test_production_restores_before_rebase_and_reports_nonfatal_pp(self):
        public,start,inputs,physical,world=self.fixture();before=copy.deepcopy(public.materialization_payload)
        def bridge(payload,*args,**kwargs):
            self.assertTrue(all('slp' not in row['condition'] for side in payload['sides'].values() for row in side['pokemon']))
            self.assertEqual(kwargs['removed_item_species'],{'p1':(),'p2':('h',)})
            return world
        def rebase(*args,state,**kwargs):
            self.assertEqual(np.asarray(state.status)[0,:,0].tolist(),[5,5])
            self.assertEqual(np.asarray(state.status_counter)[0,:,0].tolist(),[3,3])
            self.assertEqual(np.asarray(state.rest_origin)[0,:,0].tolist(),[True,True])
            self.assertEqual(np.asarray(state.pp)[0,0,:,0].tolist(),[4,5,16,16,16,16])
            return 'inert-carried-public'
        unpack=lambda team:tuple(NS(species=part.split('|')[0],ability='',item='') for part in team.split(']'))
        with patch.object(protocol,'battle_spec_from_payload',side_effect=bridge),patch.object(protocol,'map_spec',return_value=NS(state=physical)), \
                patch.object(protocol,'unpack_team',side_effect=unpack),patch.object(protocol.public_worlds,'_revealed',return_value=()), \
                patch.object(protocol.states,'_rebase',side_effect=rebase):
            made=protocol.materialize_protocol(public,start,inputs=inputs,seed=17)
        self.assertIsNotNone(made.state,made.refusals);self.assertEqual(public.materialization_payload,before)
        receipt=next(row for row in made.mapper_diagnostics if row['kind']=='own-request-mapper-disagreement:pp')
        self.assertEqual((receipt['status'],receipt['fatal']),('TRIGGERED',False))
        self.assertTrue(any(row['kind']=='FINAL_PUBLIC_CARRY' and row['status']=='CLEAR' for row in made.mapper_diagnostics))

    def test_removed_item_still_requires_original_reveal_posterior(self):
        public,start,inputs,physical,world=self.fixture()
        variant=NS(level=100,moves=('tackle',),ability='',item='leftovers')
        revealed=((0,None,((0,variant),),'N','g'),)
        def unpack(item):
            return lambda team:tuple(NS(species=part.split('|')[0],ability='',item=item if part.split('|')[0]=='g' else '',level=100,moves=('tackle',)) for part in team.split(']'))
        for item,expected in (('oranberry',False),('leftovers',True)):
            with patch.object(protocol,'battle_spec_from_payload',return_value=world) as bridge,patch.object(protocol,'map_spec',return_value=NS(state=physical)), \
                    patch.object(protocol,'unpack_team',side_effect=unpack(item)),patch.object(protocol.public_worlds,'_revealed',return_value=revealed), \
                    patch.object(protocol.states,'_rebase',return_value='inert'):
                made=protocol.materialize_protocol(public,start,inputs=inputs,seed=17)
            self.assertEqual(made.state is not None,expected,made.refusals)
            if expected:self.assertEqual(int(made.state.item[0,1,1]),0)
            else:
                self.assertEqual(made.refusals,('sampled-foe-outside-b2-public-posterior',));bridge.assert_not_called()
                self.assertTrue(any(c['kind']=='PUBLIC_SLEEP_WORLD_POSTERIOR' and c['status']=='NOT_EVALUABLE' for c in made.mapper_diagnostics))

    def test_unrevealed_pressure_cannot_change_exact_own_pp(self):
        public,start,inputs,physical,world=self.fixture();values=[]
        for ability in ('','pressure'):
            unpack=lambda team:tuple(NS(species=part.split('|')[0],ability=ability if part.split('|')[0] in 'ghijkl' else '',item='') for part in team.split(']'))
            with patch.object(protocol,'battle_spec_from_payload',return_value=world),patch.object(protocol,'map_spec',return_value=NS(state=physical)), \
                    patch.object(protocol,'unpack_team',side_effect=unpack),patch.object(protocol.public_worlds,'_revealed',return_value=()), \
                    patch.object(protocol.states,'_rebase',return_value='inert'):
                made=protocol.materialize_protocol(public,start,inputs=inputs,seed=17)
            self.assertIsNotNone(made.state,made.refusals);values.append(np.asarray(made.state.pp)[0,0].tolist())
        self.assertEqual(values[0],values[1]);self.assertEqual(values[0][0][0],4)

    def test_independent_foe_failures_and_sibling_sleep_evidence_survive(self):
        public,start,inputs,physical,world=self.fixture();view=copy.deepcopy(inputs.root.view)
        view['hp_percent'][0,0]=-1;view['revealed_moves'][0,0,0]=1;view['move_pp_used'][0,0,0]=99
        view['mon_features']['toxic_stage'][0,0]=99;view['mon_features']['stall_counter'][0,0]=99
        root=replace(inputs.root,view=view);physical=physical._replace(status=physical.status.at[0,1,0].set(4));checks=[]
        _,_,_,failures=protocol._carry_public_foe(physical,root,inputs.metadata,world,1,random.Random(1),checks)
        self.assertTrue({'public-opponent-PP-charge-exceeds-source-max','PUBLIC_TOXIC_STAGE_INVALID','PUBLIC_PROTECT_STREAK_INVALID'}<=set(failures))
        self.assertTrue(any(row['kind']=='literal positive maxHP and public0..100percent required' and row['status']=='TRIGGERED' for row in checks))
        checks=[];origins={('p2','g'):dict(rest=True,attempt_abilities=[],attempt_refunds=[],refunded=0,skipped=0,pending=False)}
        with self.assertRaisesRegex(ValueError,'STINT_UNAVAILABLE'):
            protocol._restore_sleep(physical,inputs.root,inputs.metadata,((0,'a'),(1,'g')),origins,random.Random(1),{(1,'g'):''},diagnostics=checks)
        self.assertTrue(any(row['kind']=='public-sleep-posterior' and row['side']==1 for row in checks))
        self.assertTrue(any(row['kind']=='PUBLIC_SLEEP_STINT_UNAVAILABLE' and row['side']==0 for row in checks))

if __name__=='__main__':unittest.main()
