"""Agreement and coordinate boundaries for the remaining public trap projections."""
import copy
from types import SimpleNamespace as NS
import unittest

import numpy as np

from gpu_engine import phase_d_protocol_world as protocol
from gpu_engine.state import ROOTED
from tests.test_depthn_trap_walls import host_world


def switches(world,side):
    import jax.numpy as jnp
    from gpu_engine.state import BattleState,legal_actions
    device=BattleState(**{k:jnp.asarray(v) for k,v in world._asdict().items()})
    return np.asarray(legal_actions(device))[0,side,4:].tolist()


def payload(owner, *, rooted=False, foe_meanlook=False):
    return dict(selfPlayer=f'p{owner+1}', selfActiveRequestState=dict(trapped=True),
        sides={f'p{s+1}':dict(volatiles=['ingrain'] if s==owner and rooted else [],
            meanlookTrap=bool(s!=owner and foe_meanlook)) for s in (0,1)})


class IngrainTests(unittest.TestCase):
    def test_retained_rooted_bit_closes_only_the_holders_switches(self):
        for owner in (0,1):
            world=host_world(active=(2,5));bits=world.volatile.copy();bits[0,owner,world.active[0,owner]]|=ROOTED
            rooted=world._replace(volatile=bits)
            self.assertEqual(switches(rooted,owner),[False]*5)
            self.assertEqual(switches(rooted,1-owner),[True]*5)

    def test_only_our_ingrain_is_deferred_and_the_original_payload_is_retained(self):
        for owner in (0,1):
            with self.subTest(owner=owner):
                original=payload(owner,rooted=True);before=copy.deepcopy(original)
                projected,meanlook,wrapped=protocol._defer_traps(original,owner)
                self.assertEqual(original,before)
                self.assertFalse(projected['selfActiveRequestState']['trapped'])
                self.assertEqual(projected['sides'][f'p{owner+1}']['volatiles'],[])
                self.assertEqual((meanlook,wrapped),(False,()))
                foreign=payload(owner);foreign['sides'][f'p{2-owner}']['volatiles']=['ingrain']
                self.assertIs(protocol._defer_traps(foreign,owner)[0],foreign)
                self.assertTrue(foreign['selfActiveRequestState']['trapped'])

    def test_rooted_agreement_reads_the_active_slot_and_rejects_both_disagreements(self):
        for owner in (0,1):
            for shown in (False,True):
                for held in (False,True):
                    with self.subTest(owner=owner,shown=shown,held=held):
                        volatile=np.zeros(6,np.uint32);volatile[3]=ROOTED if held else 0
                        detached=NS(own=dict(active=np.asarray(3),volatile=volatile))
                        if shown==held:self.assertEqual(protocol._rooted_agreement(payload(owner,rooted=shown),detached,owner),shown)
                        else:
                            with self.assertRaisesRegex(ValueError,'OWN_INGRAIN_B2_PAYLOAD_DISAGREEMENT'):
                                protocol._rooted_agreement(payload(owner,rooted=shown),detached,owner)


class OpponentMeanLookTests(unittest.TestCase):
    def detached(self,slot=4):
        active=np.zeros((1,6),bool);active[0,slot]=True
        flags=np.zeros((1,6),np.int32);flags[0,slot]=1
        return NS(view=dict(active=active,mon_features=dict(meanlook_trap=flags)))

    def test_projection_does_not_remove_our_request_trap(self):
        for owner in (0,1):
            original=payload(owner,foe_meanlook=True);before=copy.deepcopy(original)
            projected,held=protocol._defer_foe_meanlook(original,owner)
            self.assertTrue(held);self.assertEqual(original,before)
            self.assertFalse(projected['sides'][f'p{2-owner}']['meanlookTrap'])
            self.assertTrue(projected['selfActiveRequestState']['trapped'])
            free=payload(owner);self.assertIs(protocol._defer_foe_meanlook(free,owner)[0],free)

    def test_public_slot_is_rebased_to_each_worlds_physical_actives(self):
        for owner in (0,1):
            world=host_world(active=(2,5));detached=self.detached(slot=4)
            out=protocol._write_foe_meanlook(world,detached,True,owner)
            foe=1-owner;at=int(world.active[0,foe])
            self.assertEqual(int(out.trap_source[0,foe,at]),int(world.active[0,owner]))
            self.assertEqual(np.count_nonzero(out.trap_source>=0),1)
            self.assertTrue(np.all(world.trap_source==-1))
            self.assertEqual(switches(out,foe),[False]*5)
            self.assertEqual(switches(out,owner),[True]*5)
            hp=out.hp.copy();hp[0,owner,world.active[0,owner]]=0
            self.assertEqual(switches(out._replace(hp=hp),foe),[True]*5)

    def test_missing_disagreeing_or_benched_public_flags_refuse(self):
        bad=[]
        bad.append((NS(view={}),True))
        bad.append((self.detached(),False))
        free=self.detached();free.view['mon_features']['meanlook_trap'][:]=0;bad.append((free,True))
        benched=self.detached();benched.view['active'][0,4]=False;benched.view['active'][0,1]=True;bad.append((benched,True))
        for detached,held in bad:
            with self.subTest(held=held),self.assertRaisesRegex(ValueError,'OPPONENT_MEANLOOK_B2_PAYLOAD_DISAGREEMENT'):
                protocol._write_foe_meanlook(host_world(),detached,held,0)
        world=host_world();self.assertIs(protocol._write_foe_meanlook(world,NS(view={}),False,0),world)




class IngrainConstructionTests(unittest.TestCase):
    def test_real_cpu_and_typed_bridges_defer_ingrain_before_exact_own_restoration(self):
        """Traverse materialize_protocol with both real bridges; only the final carry rebasing is observed."""
        from dataclasses import replace
        import hashlib
        import json
        from unittest.mock import patch
        import jax.numpy as jnp
        from pokezero.dex import MoveInfo,SpeciesInfo,ShowdownDex
        from pokezero.env import BattleStartOverride
        from pokezero.v5_contract import DecisionAddress
        from pokezero.v5_teacher_public_root import PublicRoot
        from pokezero.engine_world import battle_spec_from_payload,EngineWorldUnsupported
        from gpu_engine.device_public_worlds import Root,OWN_MON_FIELDS,OWN_SIDE_FIELDS
        from gpu_engine.v5_hypothetical_state import map_spec
        from tests.test_phase_d_batch2_facts import metadata
        from gpu_engine.state import legal_actions

        names=('abcdef','ghijkl')
        stats=dict(hp=100,atk=100,**{'def':100},spa=100,spd=100,spe=100)
        dex=ShowdownDex({'tackle':MoveInfo('tackle','Tackle','Normal','Physical','Physical',50,100,0,False,False,False,None,{},'normal',False,pp=10)},
            {name:SpeciesInfo(name,name,('Normal',),stats) for side in names for name in side},{})
        teams={f'p{side+1}':']'.join('|'.join([name,'','','','tackle','Serious','','N','','','100','255']) for name in party) for side,party in enumerate(names)}
        start=BattleStartOverride(teams)
        meta=metadata();meta['vocabulary'].update(moves=('','tackle'),types=('','normal','flying','steel','fire','ice'),items=('',));meta['moves']=[dict(pp=0),dict(pp=10)]
        source=NS(source_hash='same')
        for owner in (0,1):
            with self.subTest(owner=owner):
                own,foe=f'p{owner+1}',f'p{2-owner}'
                active=[0,0];active[owner]=1
                payload=dict(turn=2,selfPlayer=own,selfRequestKind='move',pendingBatonPassSides=[],wishSetTurns={},
                    selfActiveRequestState=dict(trapped=False),sides={f'p{s+1}':dict(pokemon=[dict(species=name,condition='341/341' if s==owner else '100/100',active=i==active[s]) for i,name in enumerate(party)],volatiles=[]) for s,party in enumerate(names)})
                baseline=battle_spec_from_payload(payload,start,dex=dex,approximate_sleep_turns=False,
                    approximate_substitute_health=False,approximate_partial_trap_turns=False,approximate_hidden_duration_volatiles=False)
                mapped=map_spec(baseline.spec,metadata=meta,dex=dex,seed=17,turn=2,active_turns=(1,1),happiness=((255,)*6,)*2,held_volatiles=True)
                self.assertIsNotNone(mapped.state,mapped.refusals)
                physical=mapped.state
                own_fields={name:np.array(getattr(physical,name)[0,owner]) for name in OWN_MON_FIELDS+OWN_SIDE_FIELDS}
                at=int(own_fields['active']);own_fields['volatile'][at]|=ROOTED
                party=[dict(ident=own+': '+name,details=name+', L100',condition=f'{int(own_fields["hp"][i])}/{int(own_fields["max_hp"][i])}',
                    active=i==at,moves=['tackle'],baseAbility='',item='') for i,name in enumerate(names[owner])]
                packet=dict(rqid=1,side=dict(id=own,pokemon=party),active=[dict(trapped=True,moves=[dict(id='tackle',pp=16,maxpp=16,disabled=False)])])
                lines=('|start',f'|switch|{own}a: {names[owner][at]}|{names[owner][at]}, L100|{int(own_fields["hp"][at])}/{int(own_fields["max_hp"][at])}',
                    f'|switch|{foe}a: {names[1-owner][0]}|{names[1-owner][0]}, L100|100/100','|turn|1',f'|-start|{own}a: {names[owner][at]}|Ingrain','|request|'+json.dumps(packet))
                payload['sides'][own]['volatiles']=['ingrain'];payload['selfActiveRequestState']['trapped']=True
                original=copy.deepcopy(payload)
                with self.assertRaises(EngineWorldUnsupported) as unsupported:
                    battle_spec_from_payload(payload,start,dex=dex,approximate_sleep_turns=False,
                        approximate_substitute_health=False,approximate_partial_trap_turns=False,approximate_hidden_duration_volatiles=False)
                self.assertEqual(unsupported.exception.reason,'volatile_unsupported')
                carry=protocol.public_worlds.public.initialize(physical)._replace(item_removed=jnp.zeros((1,2,6),bool))
                address=DecisionAddress(1,0,owner)
                view=dict(present=np.ones((1,6),bool),active=np.eye(6,dtype=bool)[[0]],species=np.asarray(physical.species)[0,1-owner][None,:],
                    hp_percent=np.full((1,6),100),revealed_moves=np.zeros((1,6,4),np.int32),move_pp_used=np.zeros((1,6,4),np.int32),
                    mon_features=dict(toxic_stage=np.zeros((1,6),np.int32),stall_counter=np.zeros((1,6),np.int32),meanlook_trap=np.zeros((1,6),np.int32)))
                detached=Root(address,own_fields,view,carry,'same')
                context=NS(public_materialization_state=NS(self_request=packet,belief_engine=NS(set_source=NS(metadata=source))))
                public=PublicRoot(address,hashlib.sha256(''.join(x+'\n' for x in lines).encode()).hexdigest(),context,payload,())
                inputs=protocol.ProtocolInputs(detached,lines,NS(source=source,dex=dex),meta,object())
                seen=[]
                def rebase(*args,state,**kwargs):
                    seen.append(True)
                    self.assertTrue(int(np.asarray(state.volatile)[0,owner,at])&ROOTED)
                    mask=np.asarray(legal_actions(protocol._device(state)))
                    self.assertFalse(mask[0,owner,4:].any());self.assertTrue(mask[0,1-owner,4:].all())
                    return detached.public
                with patch.object(protocol,'battle_spec_from_payload',wraps=battle_spec_from_payload) as cpu, \
                        patch.object(protocol,'map_spec',wraps=map_spec) as typed, \
                        patch.object(protocol.public_worlds,'_revealed',return_value=()),patch.object(protocol,'_set_posteriors',return_value={}), \
                        patch.object(protocol.states,'_rebase',side_effect=rebase),patch.object(protocol.states,'_rebase_host',side_effect=rebase):
                    made=protocol.materialize_protocol(public,start,inputs=inputs,seed=17)
                self.assertIsNotNone(made.state,made.refusals);self.assertTrue(seen)
                self.assertEqual(cpu.call_count,1);self.assertEqual(typed.call_count,1)
                self.assertNotIn('ingrain',cpu.call_args.args[0]['sides'][own]['volatiles'])
                self.assertEqual(payload,original)


if __name__=='__main__':unittest.main()
