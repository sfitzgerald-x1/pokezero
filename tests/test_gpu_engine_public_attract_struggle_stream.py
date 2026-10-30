"""Captured activation spellings retain their real sleep and raw consumers."""
from dataclasses import replace
import importlib.util
import json
import unittest


class ActivationGrammarTests(unittest.TestCase):
    def test_closed_attract_and_struggle_shapes(self):
        from gpu_engine.public_annotations import classify
        self.assertEqual(classify('|-activate|p1a: Nick|move: Attract|[of] p2a: Absol'), 'attract')
        self.assertEqual(classify('|-activate|p2a: Absol|move: Struggle'), 'target')
        for line in ('|-activate|p1a: Nick|move: Attract',
            '|-activate|p1a: Nick|Attract|[of] p2a: Absol',
            '|-activate|p1: Nick|move: Attract|[of] p2a: Absol',
            '|-activate|p1a: Nick|move: Attract|[of] p2: Absol',
            '|-activate|p1a: Nick|move: Attract|[of] p2a: Absol|extra',
            '|-activate|p1a: Nick|move: Attract|[from] ability: Cute Charm|[of] p2a: Absol',
            '|-activate|p2a: Absol|move: Struggle|extra',
            '|-activate|p2a: Absol|move: Struggle|[of] p1a: Nick'):
            self.assertIsNone(classify(line), line)


@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
class ActivationStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_gpu_engine_public_switch_context import SwitchContextTests
        SwitchContextTests.setUpClass()
        cls.f = SwitchContextTests.f
        cls.step = cls.f.jax.jit(lambda s,c:cls.f.e.fold_chunk(s,c,cls.f.catalog))
        cls.encode = cls.f.jax.jit(lambda s,o,p:cls.f.e.encode(s,o,cls.f.catalog,observer_sides=p))

    def consume(self, lines, owner=0):
        from gpu_engine import public_wire
        import numpy as np
        state = self.f.e.initialize(self.f.catalog,stream_ids=self.f.j.array([1],self.f.j.int32),
            owner_sides=np.array([owner],np.int32))
        host = public_wire.WireState()
        for line in lines:
            host,projections = public_wire.project_lines(host,[line],**self.f.e.wire_kwargs(self.f.catalog))
            for chunk in self.f.chunks(projections):
                state,error = type(self).step(state,chunk)
                self.assertEqual(error.tolist(),[0],line)
        return state,host

    def source(self, state, lines, owner, sleeping_side=None):
        import numpy as np
        from pokezero import showdown
        packet = self.f.request(owner)
        if sleeping_side == owner:
            packet['side']['pokemon'][1]['condition'] = '120/247 slp'
        own = self.f.e.parse_own_requests(self.f.catalog,[packet],owner_sides=[owner])
        out = type(self).encode(state,own,self.f.j.array([owner],self.f.j.int32)).observation
        self.assertEqual(out.admitted.tolist(),[True])
        parser = showdown._ReplayParser('activation-fixture',complete_prefix=True,
            hp_visibility={f'p{s+1}':'exact' if s==owner else 'percentage' for s in (0,1)})
        for line in [*lines,'|request|'+json.dumps(packet)]:parser._feed_line(line)
        replay = parser.snapshot()
        normalized = showdown.normalize_for_player(replay,player_id=f'p{owner+1}',
            configured_showdown_slot=f'p{owner+1}',format_id='gen3randombattle',set_source=self.f.public.source)
        masks = replace(showdown.DEFAULT_OBSERVATION_FEATURE_MASKS,exact_state=True,
            opponent_tendency_stats_block=True,feature_pack_last_move=True,transition_token_budget=0)
        expected = showdown.observation_from_player_state(normalized,category_vocab=self.f.vocab,
            spec=showdown.V4_REPLAY_OBSERVATION_SPEC,dex=self.f.catalog.dex,feature_masks=masks)
        np.testing.assert_allclose(np.asarray(out.numeric_features[0]),expected.numeric_features,atol=1e-6,rtol=0)
        for field in ('categorical_ids','attention_mask','legal_action_mask','token_type_ids'):
            np.testing.assert_array_equal(np.asarray(getattr(out,field)[0]),getattr(expected,field))

    def test_attract_preserves_consumed_belief_sleep_before_optional_cant_and_reentry(self):
        from gpu_engine import public_wire, public_inputs
        for side,ident,source,bench,species in ((0,'p1a: Nick','p2a: Absol','p1a: Gardevoir','Snorlax'),
                (1,'p2a: Absol','p1a: Nick','p2a: Gardevoir','Absol')):
            activation=f'|-activate|{ident}|move: Attract|[of] {source}'
            base=[*self.f.public.prefix,'|turn|1',f'|-start|{ident}|Attract',
                f'|-status|{ident}|slp|[from] move: Rest','|upkeep','|turn|2',f'|cant|{ident}|slp']
            for owner in (0,1):
                before,host=self.consume(base,owner)
                self.assertTrue(bool(before.context.sleep.pending[0,side,0]))
                next_host,frame=public_wire.project_line(host,activation,**self.f.e.wire_kwargs(self.f.catalog))
                facts=[e for e in frame.events if e.kind==public_inputs.Kind.SLEEP]
                self.assertEqual(facts,[])
                self.assertFalse(any(e.kind==public_inputs.Kind.MOVE for e in frame.events))
                for cant in (False,True):
                    lines=base+[activation,activation]+([f'|cant|{ident}|Attract'] if cant else [])
                    state,_=self.consume(lines,owner)
                    self.assertEqual(int(state.context.sleep.skipped[0,side,0]),0)
                    self.assertTrue(bool(state.context.sleep.pending[0,side,0]))
                    self.source(state,lines,owner,side)
                    final=lines+['|upkeep','|turn|3',f'|switch|{bench}|Gardevoir, L80|100/100',
                        f'|switch|{ident}|{species}, L80|100/100 slp']
                    state,_=self.consume(final,owner)
                    self.assertEqual(int(state.context.sleep.count[0,side,0]),1)
                    self.source(state,final,owner,side)
                for bad in (activation.replace(ident,'p1a: Unknown',1),
                        activation.replace(source,source[:2]+': '+source.split(': ',1)[1]),
                        activation+'|extra'):
                    with self.assertRaises(ValueError):
                        public_wire.project_line(host,bad,**self.f.e.wire_kwargs(self.f.catalog))
                self.assertEqual(host.live,next_host.live)

    def test_struggle_announcement_does_not_synthesize_move_or_pp(self):
        from gpu_engine import public_wire,public_inputs
        for owner in (0,1):
            base=[*self.f.public.prefix,'|turn|1']
            before,host=self.consume(base,owner)
            _,frame=public_wire.project_line(host,'|-activate|p2a: Absol|move: Struggle',
                **self.f.e.wire_kwargs(self.f.catalog))
            self.assertFalse(any(e.kind==public_inputs.Kind.MOVE for e in frame.events))
            self.assertFalse(any(e.kind==public_inputs.Kind.SLEEP for e in frame.events))
            lines=base+['|-activate|p2a: Absol|move: Struggle']
            state,_=self.consume(lines,owner)
            self.source(state,lines,owner)
            import numpy as np
            for field in ('move_ids','uses','last_kind','last_id'):
                np.testing.assert_array_equal(getattr(state.context.moves,field),getattr(before.context.moves,field))
            np.testing.assert_array_equal(state.context.refresh.controller.first_selected,before.context.refresh.controller.first_selected)
            lines+=['|move|p2a: Absol|Struggle|p1a: Nick']
            state,_=self.consume(lines,owner)
            self.source(state,lines,owner)
