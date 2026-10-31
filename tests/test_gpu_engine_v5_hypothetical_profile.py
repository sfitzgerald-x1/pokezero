"""Hypothetical observer components only; no scientific root/model/Native."""
import ast
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np
from gpu_engine import v5_public_world as world,v5_public_branch as branch,v5_hypothetical_profile as h
from gpu_engine import engine_weather_speed_profile as speed,engine_lifecycle_callbacks as life,transition
from gpu_engine import v5_owning_history as capture,public_wire,v4_encoder
import test_gpu_engine_v5_public_world as f


class ProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        f.PublicWorldTests.setUpClass();cls.f=f.PublicWorldTests();cls.runtime=cls.f.runtime

    def root(self):
        root,start,history,static=self.f.fixture(spent=True,slow_own=True)
        result=world.materialize(root,start,prepared=self.f.f.prepared,history=history,own_static=static,seed=5)
        self.assertIsNotNone(result.branch,result.refusals)
        return result.branch

    def test_separate_origin_without_birth_or_hp_turn_reset_and_sticky_math(self):
        root=self.root();before=root.state
        with (patch.object(speed,'qualify_birth',side_effect=AssertionError('no ordinary birth')),
                patch.object(speed,'qualify_birth_device',side_effect=AssertionError('no ordinary birth'))):
            context=h.qualify_origin(root)
        self.assertIs(type(context.record),h.HypotheticalRecord)
        self.assertFalse(hasattr(context.record,'qualified_birth'))
        self.assertTrue(bool(context.record.qualified_origin.all()))
        self.assertEqual(root.root.materialization_payload['turn'],2)
        self.assertEqual(int(before.turn[0]),1) # ORIGINAL completed-turn storage
        self.assertEqual(int(branch.core(root,self.runtime).cursor.turn[0]),2)
        self.assertLess(int(before.hp[0,0,0]),int(before.max_hp[0,0,0]))
        self.assertIs(root.state,before)
        streams=context.record.stream_id
        with self.assertRaises(ValueError):speed.retain_device(context.record,before,streams,context.data)
        changed=before._replace(stats=before.stats.at[0,0,0,5].add(1))
        rejected=h.retain_device(context.record,changed,streams,context.data)
        self.assertFalse(bool(rejected.qualified_origin.any()))
        self.assertFalse(bool(h.retain_device(rejected,before,streams,context.data).qualified_origin.any()))
        foreign=context.record._replace(origin_binding=j.zeros((1,8),j.uint32))
        self.assertFalse(bool(h.retain_device(foreign,before,streams,context.data).qualified_origin.any()))
        self.assertFalse(bool(h.retain_device(context.record,before,streams+1,context.data).qualified_origin.any()))

    def test_actual_public_history_turn1_and_turn7_map_completed_turns_and_cursor(self):
        for displayed in (1,7):
            with self.subTest(displayed=displayed):
                root,start,history,static=self.f.fixture()
                if displayed>1:
                    wire,_=public_wire.project_lines(public_wire.WireState(),
                        [line for line in capture.decode(history).lines if not line.startswith('|request|')],
                        **v4_encoder.wire_kwargs(self.f.catalog))
                    frames=list(history.frames)
                    for number in range(1,displayed):
                        wire,packets=public_wire.project_lines(wire,['|','|upkeep',f'|turn|{number+1}'],
                            **v4_encoder.wire_kwargs(self.f.catalog))
                        words=tuple(tuple(map(int,event.words())) for p in packets for event in p.events)
                        frames.append(capture.Frame(number,(words,),((True,)*len(words),),(False,),frames[0].request_json,()))
                    history=replace(history,frames=tuple(frames))
                    with ExitStack() as stack:
                        stack.enter_context(patch('subprocess.Popen',side_effect=AssertionError('no Native')))
                        stack.enter_context(patch('pokezero.local_showdown.load_gen3_randbat_source_cached',return_value=self.f.source))
                        stack.enter_context(patch('pokezero.local_showdown.load_showdown_dex_cached',return_value=self.f.catalog.dex))
                        result=capture.public_root(history,config=self.f.config,contract=self.f.contract,
                            battle_id='literal-clock-component',seed=31)
                    self.assertIsNotNone(result.root,result.decoded.refusal_counts);root=result.root
                result=world.materialize(root,start,prepared=self.f.f.prepared,history=history,own_static=static,seed=5)
                self.assertIsNotNone(result.branch,result.refusals)
                self.assertEqual(root.materialization_payload['turn'],displayed)
                self.assertEqual(int(result.branch.state.turn[0]),displayed-1)
                self.assertEqual(int(branch.core(result.branch,self.runtime).cursor.turn[0]),displayed)

    def test_actual_positive_leftovers_and_both_full_folds_match_default_physical_rng(self):
        root=self.root();current=branch.core(root,self.runtime)
        draws=j.full((1,2,12),99,j.int32).at[:,:,4].set(1)
        chance=transition.ChanceInputs(draws,*[j.zeros((1,2),j.int32) for _ in range(4)],j.zeros(1,j.int32))
        result=branch.step(root,(0,0),chance_inputs=chance,hypothetical_profiles=True)
        self.assertFalse(result.refusals,(result.refusals,result.diagnostics))
        self.assertEqual(len(result.encoded),2)
        self.assertTrue(all(bool(value.observation.admitted.all()) for value in result.encoded))
        self.assertIs(type(result.raw_frame.weather_profile),h.HypotheticalRecord)
        self.assertTrue(bool(result.raw_frame.weather_profile.qualified_origin.all()))
        self.assertEqual(dict(result.diagnostics)['cutpoints_complete'],1)
        plain=transition.step_with_chance(current.engine,j.array([[0,0]],j.int32),self.runtime['request']['engine_table'],chance)
        for actual,expected in zip(jax.tree.leaves(result.state),jax.tree.leaves(plain.state)):
            np.testing.assert_array_equal(actual,expected)
        # Fault injection reuses this tiny component's actual same-call frame;
        # no second numerical execution or invented scalar result is needed.
        frame=result.raw_frame;execution=frame.callbacks.execution;risks=execution.risks
        bad=frame._replace(callbacks=frame.callbacks._replace(execution=execution._replace(
            risks=risks._replace(carry=risks.carry._replace(counts=risks.carry.counts.at[:,0].add(1))))))
        with patch.object(branch.driver,'_frame',return_value=bad):
            refused=branch.step(root,(0,0),chance_inputs=chance,hypothetical_profiles=True)
        self.assertTrue(refused.refusals);self.assertFalse(refused.encoded)
        self.assertEqual(dict(refused.diagnostics)['cutpoints_complete'],0)
        # The ordinary keyword still refuses a hypothesis; only the explicitly
        # named new context may use this conditional hypothetical math scope.
        context=h.qualify_origin(root);tape=life._tape(current.cursor,self.runtime['request']['lifecycle'],64)
        with self.assertRaises(ValueError):transition.step_with_chance(current.engine,j.array([[0,0]],j.int32),
            self.runtime['request']['engine_table'],chance,_public_trace=tape,_lifecycle_trace=True,
            _weather_hp_context=(context.record,context.data))


def normalized_hypothetical_default(text,*,as_tree=False):
    """Normalize only the explicit hypothetical observer extension, preserving numerics."""
    class Default(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            if node.name=='step_with_chance':
                keep=[(a,d) for a,d in zip(node.args.kwonlyargs,node.args.kw_defaults) if a.arg!='_hypothetical_profile']
                node.args.kwonlyargs=[a for a,d in keep];node.args.kw_defaults=[d for a,d in keep]
            return self.generic_visit(node)
        def visit_Assign(self,node):
            if any(isinstance(a,ast.Name) and a.id in ('_hypothetical','_weather_observer_context','_profile_qualified') for a in node.targets):return None
            return self.generic_visit(node)
        def visit_Name(self,node):
            if node.id=='_hypothetical':return ast.Constant(False)
            if node.id=='_weather_observer_context':node.id='_weather_hp_context'
            return node
        def visit_UnaryOp(self,node):
            node=self.generic_visit(node)
            if isinstance(node.op,ast.Not) and isinstance(node.operand,ast.Constant):return ast.Constant(not node.operand.value)
            return node
        def visit_BoolOp(self,node):
            node=self.generic_visit(node);isand=isinstance(node.op,ast.And)
            for a in node.values:
                if isinstance(a,ast.Constant) and bool(a.value)!=isand:return ast.Constant(not isand)
            node.values=[a for a in node.values if not isinstance(a,ast.Constant)]
            return node.values[0] if len(node.values)==1 else node
        def visit_IfExp(self,node):
            node=self.generic_visit(node)
            return (node.body if node.test.value else node.orelse) if isinstance(node.test,ast.Constant) else node
        def visit_If(self,node):
            node=self.generic_visit(node)
            return (node.body if node.test.value else node.orelse) if isinstance(node.test,ast.Constant) else node
        def visit_BinOp(self,node):
            node=self.generic_visit(node)
            if isinstance(node.op,ast.Add) and isinstance(node.right,ast.Tuple) and not node.right.elts:return node.left
            return node
        def visit_Call(self,node):
            node=self.generic_visit(node)
            node.keywords=[k for k in node.keywords if k.arg!='hypothetical']
            if isinstance(node.func,ast.Name) and node.func.id=='_profile_qualified':
                return ast.Attribute(node.args[0],'qualified_birth',ast.Load())
            return node
    tree=Default().visit(ast.parse(text))
    return tree if as_tree else ast.dump(tree,include_attributes=False)


class DefaultAST(unittest.TestCase):
    def test_whole_default_transition_ast_and_mutations(self):
        original=subprocess.check_output(['git','show','552bd326:gpu_engine/transition.py'],text=True)
        current=Path('gpu_engine/transition.py').read_text()
        normalize=normalized_hypothetical_default
        self.assertEqual(normalize(original),normalize(current))
        self.assertNotEqual(normalize(original),normalize(current.replace('2 + chance[:,7]','3 + chance[:,7]')))
        self.assertNotEqual(normalize(original),normalize(current.replace('compound_eyes=compound_eyes','compound_eyes=sand_veil')))


class HypotheticalWeatherMath(unittest.TestCase):
    def test_original_current_origin_weather_order_matches_ordinary_conditional_math(self):
        import test_gpu_engine_engine_weather_hp_segment as fixture
        from gpu_engine import engine_weather_hp_segment as segment
        fixture.WeatherHPSegmentTests.setUpClass();f=fixture.WeatherHPSegmentTests()
        values=f.case(prior=128);tape,before,pre,after,speeds,_,writes=values;data=f.data
        # A separately tagged current partial-HP/turn2 origin. No ordinary
        # birth record fields are reused to create this hypothetical record.
        broad=lambda value:j.broadcast_to(value,(1,8))
        record=h.HypotheticalRecord(speed._members(pre,data,birth=False),tape.stream_id,
            broad(data.engine_binding),broad(data.source_binding),broad(data.binding),
            pre.stats[...,5],pre.base_ability,j.ones((1,8),j.uint32))
        actual=segment.order_segment_device(tape,before,pre,after,speeds,record,data,
            pre_writes_state=writes,eligible=True,hypothetical=True)
        expected=f.run_case(values)
        f.same(actual.tape,expected.tape)
        self.assertEqual(actual.supported.tolist(),[True]);self.assertEqual(actual.reordered.tolist(),[True])
        self.assertIs(type(actual.record),h.HypotheticalRecord)
        self.assertEqual(actual.tape.refusal.tolist(),[128])
