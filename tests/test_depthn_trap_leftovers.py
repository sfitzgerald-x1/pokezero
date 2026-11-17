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

    def test_only_our_rooted_request_is_deferred_and_the_volatile_is_retained(self):
        for owner in (0,1):
            with self.subTest(owner=owner):
                original=payload(owner,rooted=True);before=copy.deepcopy(original)
                projected,meanlook,wrapped=protocol._defer_traps(original,owner)
                self.assertEqual(original,before)
                self.assertFalse(projected['selfActiveRequestState']['trapped'])
                self.assertEqual(projected['sides'][f'p{owner+1}']['volatiles'],['ingrain'])
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


if __name__=='__main__':unittest.main()
