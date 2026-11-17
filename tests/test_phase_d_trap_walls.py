"""The trap walls through the original B2 adapter (authored folds; pod or CI runtime): from_b2's own trap flags on both routes,
our partial trap admitted only on the protocol route (the one that draws its count), B2's partial-trap bag and clock as the
protocol world's agreement check reads them, and our Mean Look link. Pure checks: test_depthn_trap_walls; real roots:
phase_d_trap_walls_replay."""
import json
from types import SimpleNamespace as NS
import unittest

import numpy as np

from gpu_engine import phase_d_live_worlds as live, phase_d_protocol_world as protocol, phase_d_tail as tail
from pokezero.v5_contract import DecisionAddress
import test_depthn_trap_walls as pure


class FromB2TrapRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_phase_d_device_worlds import WorldsTests
        from pokezero.live_protocol_runtime import CompiledEncoderFunctions
        import jax
        WorldsTests.setUpClass();cls.helper=WorldsTests()
        # One B2 fold/encode compile for every case, under the live server's precision (live_protocol_runtime.validate:
        # jax_default_matmul_precision 'highest'): set for this class, restored after it (as the knock-off fixture does).
        precision=jax.config.jax_default_matmul_precision;jax.config.update('jax_default_matmul_precision','highest')
        cls.addClassCleanup(jax.config.update,'jax_default_matmul_precision',precision)
        cls.functions=CompiledEncoderFunctions(cls.helper.catalog)

    def adapter(self, owner):
        """WorldsTests.adapter's authored fold, sharing one set of compiled encoder functions across the cases."""
        from pokezero.recorded_encoder_runner import JaxRecordedAdapter
        a=JaxRecordedAdapter(self.helper.catalog,owner_side=owner,compiled_functions=self.functions);a.reset(stream_id=1,battle_id='authored')
        a.feed((f'|switch|p{owner+1}a: Alpha|Alpha, L80|291/291',f'|switch|p{2-owner}a: Beta|Beta, L80|100/100','|turn|1'))
        return a

    def roots(self, owner, wire, flags=None):
        """(original-route root or its Refusal, protocol-route root or its Refusal, lines) after the authored fold plus wire."""
        own,foe=f'p{owner+1}',f'p{2-owner}'
        adapter=self.adapter(owner);packet=self.helper.request(owner);packet['active'][0].update(flags or {})
        adapter.feed(wire)
        lines=('|start',f'|switch|{own}a: Alpha|Alpha, L80|291/291',f'|switch|{foe}a: Beta|Beta, L80|100/100','|turn|1')+tuple(wire)
        turn=sum(1 for x in lines if x.startswith('|turn|'))
        payload=dict(turn=turn,selfPlayer=own,selfRequestKind='move',pendingBatonPassSides=[],wishSetTurns={},
            sides={own:dict(pokemon=[dict(species='Alpha',condition='291/291',active=True)],volatiles=[]),
                   foe:dict(pokemon=[dict(species='Beta',condition='100/100',active=True)],volatiles=[])})
        facts=tail.facts(NS(materialization_payload=payload),packet,lines+('|request|'+json.dumps(packet),))
        address=DecisionAddress(1,0,owner);out=[]
        for kw in (dict(),dict(defer_sleep=True,protocol_facts=facts)):
            try:out.append(live.from_b2(adapter,packet,self.helper.source,self.helper.metadata,address=address,**kw))
            except live.Refusal as error:out.append(error)
        return out[0],out[1],lines

    def test_request_trap_flags_leave_the_side_flag_false_on_both_routes(self):
        """`trapped` and `maybeTrapped` are both the world's to reproduce: neither refuses (maybeTrapped used to,
        OWN_TRAP_STATE_UNAVAILABLE) and neither seals state.trapped, on either route."""
        for owner in (0,1):
            for flags in ({'trapped':True},{'maybeTrapped':True}):
                with self.subTest(owner=owner,flags=flags):
                    for root in self.roots(owner,(),flags)[:2]:
                        self.assertNotIsInstance(root,Exception,root);self.assertFalse(bool(root.own['trapped']))

    def test_our_partial_trap_only_on_the_protocol_route_with_b2s_clock(self):
        """Our wrapped active: the original route still refuses (it cannot draw the hidden count); the protocol route admits
        it, and B2's bag and public clock (|turn| lines since the start) agree with the lines as the protocol world checks."""
        for owner in (0,1):
            own,foe=f'p{owner+1}',f'p{2-owner}'
            start=(f'|-activate|{own}a: Alpha|move: Wrap|[of] {foe}a: Beta',)   # the authored namespace has no Wrap |move| line
            for later,turns in (((),0),(('|','|upkeep','|turn|2'),1)):
                with self.subTest(owner=owner,turns=turns):
                    older,root,lines=self.roots(owner,start+later,{'trapped':True})
                    self.assertIsInstance(older,live.Refusal);self.assertRegex(str(older),'OWN_VOLATILE_OR_LATENT_CLOCK_UNAVAILABLE.*partiallytrapped')
                    self.assertNotIsInstance(root,Exception,root)
                    at=int(root.own['active']);bit=live.hooks.VOLATILE_NAMES.index('partiallytrapped')
                    self.assertTrue(bool(root.view['own_volatiles'][0,at,bit]))
                    self.assertEqual(int(root.view['own_mon_features']['wrap_trap_turns'][0,at]),turns)
                    facts=protocol._wrap_facts(lines)
                    self.assertEqual((facts[owner]['turns'],facts[1-owner]),(turns,None))
                    protocol._wrap_agreement(facts,(owner,),root,owner)
                    with self.assertRaisesRegex(ValueError,'PUBLIC_WRAP_B2_DISAGREEMENT'):
                        protocol._wrap_agreement(tuple(None if f is None else dict(f,turns=f['turns']+1) for f in facts),(owner,),root,owner)
                    with self.assertRaisesRegex(ValueError,'PUBLIC_WRAP_PAYLOAD_DISAGREEMENT'):
                        protocol._wrap_agreement(facts,(),root,owner)

    def test_the_opponents_partial_trap_in_b2s_view(self):
        """The opponent wrapped by our active: from_b2 admits it on both routes (the protocol world draws it; the original route's
        materializer refuses the opponent's clock itself), and B2's view holds it with its public clock."""
        for owner in (0,1):
            own,foe=f'p{owner+1}',f'p{2-owner}'
            wire=(f'|-activate|{foe}a: Beta|move: Wrap|[of] {own}a: Alpha','|','|upkeep','|turn|2')
            with self.subTest(owner=owner):
                older,root,lines=self.roots(owner,wire)
                for r in (older,root):self.assertNotIsInstance(r,Exception,r)
                shown=np.asarray(root.view['active'][0]).astype(bool);bit=live.hooks.VOLATILE_NAMES.index('partiallytrapped')
                self.assertEqual(np.asarray(root.view['volatiles'])[0,shown,bit].tolist(),[True])
                self.assertEqual(np.asarray(root.view['mon_features']['wrap_trap_turns'])[0,shown].tolist(),[1])
                protocol._wrap_agreement(protocol._wrap_facts(lines),(1-owner,),root,owner)

    def test_our_mean_look_link(self):
        """B2 links our Mean Look trap on both routes (coordinate 0); the protocol world rebases it to the world's trapper, and an
        opposing switch clears it (the trap ends when the trapper leaves)."""
        for owner in (0,1):
            own,foe=f'p{owner+1}',f'p{2-owner}'
            trapped=(f'|-activate|{own}a: Alpha|trapped',)   # the authored namespace has no Mean Look |move| line
            with self.subTest(owner=owner):
                for root in self.roots(owner,trapped,{'trapped':True})[:2]:
                    self.assertNotIsInstance(root,Exception,root)
                    at=int(root.own['active']);self.assertEqual(int(root.own['trap_source'][at]),0)
                    world=pure.host_world(active=(at,3) if owner==0 else (3,at))
                    self.assertEqual(int(protocol._write_own_meanlook(world,root,True,owner).trap_source[0,owner,at]),3)
                gone=trapped+('|','|upkeep','|turn|2',f'|switch|{foe}a: Gamma|Gamma, L80|100/100')
                for root in self.roots(owner,gone)[:2]:
                    self.assertNotIsInstance(root,Exception,root);self.assertEqual(int(root.own['trap_source'][int(root.own['active'])]),-1)


if __name__ == '__main__':
    unittest.main()
