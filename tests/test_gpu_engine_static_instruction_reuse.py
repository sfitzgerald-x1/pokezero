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

# V2 self_moves (gpu_engine/private_self_moves.py) changed v4_encoder.py after the static-instruction reuse. Its documented
# delta, reversed below so the source-reverse pin still proves everything else is the original program: one import and two
# module constants added; three dataclass fields added; Masks.from_checkpoint, compile_encoder, parse_own_requests and
# encode changed exactly by these (current, original) replacements over ast.unparse; parse_own_requests' docstring grew.
SELF_MOVES_ADDED = frozenset({'OPTIONAL_KEYS', 'ENCODE_ONLY'})
SELF_MOVES_FIELDS = {'Masks': 'self_moves: bool = False', 'Catalog': 'self_moves: object = None', 'OwnFacts': 'self_moves: object = None'}
SELF_MOVES_DELTAS = {
    'Masks': (
        ("        from pokezero.observation import resolve_self_moves\n"
         "        required = set(cls.__dataclass_fields__) - OPTIONAL_KEYS\n"
         "        if not isinstance(values, dict) or not required <= set(values) <= set(cls.__dataclass_fields__):",
         "        if not isinstance(values, dict) or set(values) != set(cls.__dataclass_fields__):"),
        ("return cls(**{**values, 'self_moves': resolve_self_moves(values.get('self_moves'))})", "return cls(**values)")),
    'compile_encoder': (
        ("    self_moves = private_self_moves.compile_self_moves(moves, features, vocabulary) if masks.self_moves else None\n", ""),
        ("masks={k: v for k, v in feature_masks.items() if k not in ENCODE_ONLY}", "masks=feature_masks"),
        (", checkpoint_sha256, self_moves=self_moves)", ", checkpoint_sha256)")),
    'parse_own_requests': (
        ("owner_sides, own_lines=None):", "owner_sides):"),
        ("    if own_lines is not None and len(own_lines) != len(requests):\n"
         "        raise ValueError('one own line history per request required')\n", ""),
        (", catalog.binding, None if catalog.self_moves is None or own_lines is None else private_self_moves.parse_own_moves("
         "request, owner_side=owner, catalog=catalog.self_moves, lines=own_lines[len(rows)])))", ", catalog.binding))")),
    'encode': (
        ("    if catalog.masks.self_moves:\n"
         "        if catalog.self_moves is None or own.self_moves is None:\n"
         "            raise ValueError('self_moves requires the self-move catalog and own move facts (parse_own_requests own_lines)')\n"
         "        result = v4_assembly.overlay(result, private_self_moves.project(own.self_moves, mons, own.status, "
         "catalog.self_moves.device, observer_side=observer, exact_state=catalog.masks.exact_state), section='own')\n", ""),),
}
SELF_MOVES_DOCSTRINGS = {'parse_own_requests': 'Read only the supplied owning packets, never the opponent request.'}


def reverse_self_moves(module):
    """`module` without the V2 self_moves delta above; each replacement must match exactly once."""
    body = []
    for node in module.body:
        if (isinstance(node, ast.ImportFrom) and [a.name for a in node.names] == ['private_self_moves']
                or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in SELF_MOVES_ADDED for t in node.targets)):
            continue
        name = getattr(node, 'name', None)
        if name in SELF_MOVES_FIELDS:
            kept = [n for n in node.body if ast.unparse(n) != SELF_MOVES_FIELDS[name]]
            assert len(kept) == len(node.body) - 1, name
            node.body = kept
        if name in SELF_MOVES_DOCSTRINGS:
            node.body[0].value.value = SELF_MOVES_DOCSTRINGS[name]
        if name in SELF_MOVES_DELTAS:
            text = ast.unparse(node)
            for new, original in SELF_MOVES_DELTAS[name]:
                assert text.count(new) == 1, (name, new)
                text = text.replace(new, original)
            node = ast.parse(text).body[0]
        body.append(node)
    module.body = body
    return module


def reverse_posterior(module):
    """Remove the reviewed Stage 2 delta before checking the earlier encoder pin."""
    for index, node in enumerate(module.body):
        text = ast.unparse(node)
        if isinstance(node, ast.Assign) and text.startswith('OPTIONAL_KEYS = '):
            assert ast.literal_eval(node.value.args[0]) == {'self_moves', 'set_posterior_narrowing'}
            node.value.args[0] = ast.Set(elts=[ast.Constant('self_moves')])
        elif getattr(node, 'name', None) == 'Masks':
            assert text.count('    set_posterior_narrowing: bool = False\n') == 1
            module.body[index] = ast.parse(text.replace('    set_posterior_narrowing: bool = False\n', '')).body[0]
        elif getattr(node, 'name', None) == 'compile_encoder':
            optional = ("    feature_masks = dict(feature_masks)\n"
                        "    if not masks.set_posterior_narrowing:\n"
                        "        feature_masks.pop('set_posterior_narrowing', None)\n")
            assert text.count(optional) == 1
            module.body[index] = ast.parse(text.replace(optional, '')).body[0]
        elif getattr(node, 'name', None) == 'encode':
            posterior = '    posterior = summary.effective[rows, jnp.clip(observer, 0, 1)]\n'
            delta = (
                ("observer_sides, set_posterior=None):", "observer_sides):"),
                ("    if catalog.masks.set_posterior_narrowing:\n"
                 "        if set_posterior is None:\n"
                 "            raise ValueError('source-bound set posterior inputs required')\n"
                 "        from .set_posterior_features import apply\n"
                 "        summary, posterior_ok = apply(summary, set_posterior, state.beliefs, catalog.source.device)\n"
                 "        valid &= posterior_ok\n", ''),
                (posterior + "    weights = beliefs.posterior_weights\n"
                 "    if catalog.masks.set_posterior_narrowing:\n"
                 "        weights = set_posterior.weights\n"
                 "        from .public_belief_features import _bits\n"
                 "        selected = _bits(posterior, weights.shape[-1])\n"
                 "        weight_ok = jnp.all(jnp.isfinite(weights) & (weights >= 0) & ((weights == 0) | selected), axis=(1, 2))\n"
                 "        weight_ok &= jnp.all(jnp.where(result.attention_mask[:, 7:13], jnp.abs(weights.sum(-1) - 1) < 1e-06, weights.sum(-1) == 0), axis=-1)\n"
                 "        valid &= weight_ok\n", ''),
                ("    return Encoded(result, jnp.where(admitted[:, None, None], posterior, 0), "
                 "jnp.where(admitted[:, None, None], weights, 0),", posterior +
                 "    return Encoded(result, jnp.where(admitted[:, None, None], posterior, 0), "
                 "jnp.where(admitted[:, None, None], beliefs.posterior_weights, 0),"),
            )
            for added, original in delta:
                assert text.count(added) == 1, added
                text = text.replace(added, original, 1)
            module.body[index] = ast.parse(text).body[0]
    return module


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
        current = reverse_self_moves(reverse_posterior(current))
        self.assertEqual(ast.dump(current, include_attributes=False), ast.dump(ast.parse(old), include_attributes=False))
        path='gpu_engine/engine_retained_encoder_driver.py'
        self.assertEqual((root/path).read_bytes(),subprocess.check_output(['git','show','b8092a85:'+path],cwd=root))
        # Re-certified literal pins may change; every guard statement and the
        # complete source inventory remain bound to the integrated baseline.
        from gpu_engine import engine_source_frame_witness as witness
        witness._source()
        path='gpu_engine/engine_source_frame_witness.py'
        def without_pins(text):
            tree=ast.parse(text)
            tree.body=[n for n in tree.body if not(isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='SOURCE_PINS'for t in n.targets))]
            return ast.dump(tree,include_attributes=False)
        # The reviewed witness body is integration 11ce1be7's (M2's 7476bce7 moved ENCORE's cutpoints from 12 to 16).
        self.assertEqual(without_pins((root/path).read_text()),without_pins(subprocess.check_output(['git','show','11ce1be7:'+path],cwd=root,text=True)))


if __name__ == '__main__':
    unittest.main()
