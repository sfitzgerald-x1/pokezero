"""Compile expected instructions once; always traverse current catalog edges."""
import ast
import copy
from dataclasses import dataclass
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import jax.numpy as j
from gpu_engine import v4_encoder as e

BASE = 'de4180168925e163cb87d3220d58975086f8e96f'


@dataclass
class Facts:
    tokens: object
    metadata: object
    sequence: object
    array: object


class StaticInstructionReuseTests(unittest.TestCase):
    def setUp(self):
        self.value = 10 ** 50
        self.facts = Facts(tuple('literal-'+str(i) for i in range(4)),
                           {'source': self.value}, [self.value], j.array([3], j.int32))
        self.catalog = object.__new__(e.Catalog)
        object.__setattr__(self.catalog, 'vocabulary', self.facts)
        self.record = (('vocabulary', e._static_snapshot(self.facts, set())),)
        self.plan = e._compile_static_plan(self.record)
        e._CATALOGS[self.catalog] = self.record
        e._STATIC_PLANS[self.catalog] = self.plan
        e._register_static_instructions(self.catalog, self.plan)

    def tearDown(self):
        e._STATIC_INSTRUCTION_CODES.pop(self.catalog, None)
        e._CATALOGS.pop(self.catalog, None)
        e._STATIC_PLANS.pop(self.catalog, None)
        self.assertIsNone(e._GUARD_PASS.get())

    def test_current_mutations_and_equal_replacements_still_refuse(self):
        code = e._STATIC_INSTRUCTION_CODES[self.catalog]
        previous = None
        changes = [
            (lambda: self.facts.metadata.__setitem__('source', self.value+1),
             lambda: self.facts.metadata.__setitem__('source', self.value)),
            (lambda: self.facts.sequence.append(self.value), lambda: self.facts.sequence.pop()),
            (lambda: setattr(self.facts, 'tokens', tuple(list(tokens))),
             lambda: setattr(self.facts, 'tokens', tokens)),
            (lambda: setattr(self.facts, 'metadata', dict(metadata)),
             lambda: setattr(self.facts, 'metadata', metadata)),
            (lambda: setattr(self.facts, 'array', j.array([3], j.int32)),
             lambda: setattr(self.facts, 'array', array)),
        ]
        tokens, metadata, array = self.facts.tokens, self.facts.metadata, self.facts.array
        for mutate, restore in changes:
            with e._driver_guard_pass():
                scope = e._GUARD_PASS.get()
                self.assertIsNot(scope, previous)
                previous = scope
                self.assertEqual(scope.completed, set())
                self.assertIs(scope.immutable_nodes(self.plan, self.catalog), code.flags)
                self.assertEqual(scope.plans, {})
                e.check_catalog(self.catalog)
                mutate()
                try:
                    with self.assertRaises(ValueError):
                        e.check_catalog(self.catalog)
                finally:
                    restore()
                e.check_catalog(self.catalog)
            self.assertFalse(scope.active)
            self.assertEqual(scope.completed, set())
        # A mutation before a new scope is also read, not a prior acceptance.
        self.facts.metadata['source'] = self.value+1
        with e._driver_guard_pass(), self.assertRaises(ValueError):
            e.check_catalog(self.catalog)
        self.facts.metadata['source'] = self.value

    def test_untrusted_or_replaced_instruction_lineage_uses_original_classifier(self):
        original = e._STATIC_INSTRUCTION_CODES[self.catalog]
        expected = e._GuardPass().immutable_nodes(self.plan)

        def fallback(plan):
            with e._driver_guard_pass():
                scope = e._GUARD_PASS.get()
                self.assertEqual(scope.immutable_nodes(plan, self.catalog), expected)
                self.assertIs(scope.plans[id(plan)][0], plan)
                e.check_catalog(self.catalog)
                self.facts.sequence.append(self.value)
                try:
                    with self.assertRaises(ValueError):
                        e.check_catalog(self.catalog)
                finally:
                    self.facts.sequence.pop()

        e._STATIC_INSTRUCTION_CODES.pop(self.catalog)
        fallback(self.plan)
        forged = e._StaticInstructionCode(self.plan, self.plan.nodes,
            (True,)*len(self.plan.nodes), original.classifier, original.array_type)
        e._STATIC_INSTRUCTION_CODES[self.catalog] = forged
        fallback(self.plan)
        e._STATIC_INSTRUCTION_CODES[self.catalog] = original
        replacement = e._compile_static_plan(self.record)
        e._STATIC_PLANS[self.catalog] = replacement
        fallback(replacement)
        e._STATIC_PLANS[self.catalog] = self.plan
        # Replaced policy/code is not executed through an old lineage.
        with patch.object(e, '_compile_immutable_instructions', side_effect=AssertionError('untrusted code')) as changed:
            fallback(self.plan)
            changed.assert_not_called()
        origin = e._STATIC_INSTRUCTION_ORIGINS.pop(original)
        try:
            fallback(self.plan)
        finally:
            e._STATIC_INSTRUCTION_ORIGINS[original] = origin

    def test_genuine_encoder_factory_registers_only_expected_instruction_code(self):
        import test_gpu_engine_v4_encoder as fixture
        fixture.V4EncoderTests.setUpClass()
        catalog = fixture.V4EncoderTests.catalog
        plan = e._STATIC_PLANS[catalog]
        code = e._STATIC_INSTRUCTION_CODES[catalog]
        self.assertEqual(code.flags, e._GuardPass().immutable_nodes(plan))
        self.assertTrue(any(code.flags))
        prior = None
        for _ in range(2):
            with e._driver_guard_pass():
                current = e._GUARD_PASS.get()
                self.assertIsNot(current, prior)
                prior = current
                self.assertEqual(current.completed, set())
                self.assertIs(current.immutable_nodes(plan, catalog), code.flags)
                self.assertEqual(current.plans, {})
                e.check_catalog(catalog)
            self.assertFalse(current.active)
            self.assertEqual(current.completed, set())

    def test_source_reverse_is_exact_original_program(self):
        root = Path(__file__).resolve().parents[1]
        old = subprocess.check_output(['git', 'show', BASE+':gpu_engine/v4_encoder.py'], cwd=root, text=True)
        current = ast.parse(Path(e.__file__).read_text())
        additions = {'_StaticInstructionCode', '_compile_immutable_instructions', '_register_static_instructions'}
        registries = {'_STATIC_INSTRUCTION_CODES', '_STATIC_INSTRUCTION_ORIGINS'}
        current.body = [n for n in current.body if not (
            isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in additions
            or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in registries for t in n.targets))]
        guard = next(n for n in current.body if isinstance(n, ast.ClassDef) and n.name == '_GuardPass')
        method = next(n for n in guard.body if isinstance(n, ast.FunctionDef) and n.name == 'immutable_nodes')
        method.args.args = method.args.args[:2]
        method.args.defaults = []
        start = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'key' for t in n.targets))
        method.body = method.body[start:]
        class Reverse(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                if isinstance(node.func, ast.Attribute) and ast.unparse(node.func) == 'scope.immutable_nodes':
                    node.args = node.args[:1]
                return node
            def visit_Expr(self, node):
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == '_register_static_instructions':
                    return None
                return self.generic_visit(node)
        current = Reverse().visit(current)
        self.assertEqual(ast.dump(current, include_attributes=False), ast.dump(ast.parse(old), include_attributes=False))
        for path in ('gpu_engine/engine_retained_encoder_driver.py', 'gpu_engine/engine_source_frame_witness.py'):
            self.assertEqual((root/path).read_bytes(), subprocess.check_output(['git', 'show', BASE+':'+path], cwd=root))


if __name__ == '__main__':
    unittest.main()
