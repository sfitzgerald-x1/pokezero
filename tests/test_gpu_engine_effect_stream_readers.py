"""Every use of transition's effect stream is a consumer the exact-chance plans observe; pure Python, no JAX import.

finite_law's effect.{s}.{c} axes are enumerated only where chance_consumers, phase_d_chance and ordinary_law gate them, and
an inactive axis carries a fixed 0. So the proof is lossless only while every place transition.py reads an effect value is
one of the consumers those plans observe, reading the column and with the arithmetic the plans assume. A new, moved or
rewritten reader would misprice an exact root without any error. transition.py's SOURCE_PINS refuse any edit until the pins
are refreshed, but a refresh is mechanical (scripts/refresh_source_pins.py). This test makes it insufficient: every use of
the stream, and of the whole ChanceInputs, is keyed by its enclosing function, the expression around it and a digest of its
whole enclosing statement. So a new reader, a column swap, or a changed threshold or offset changes the snapshot, and its
author has to re-review the plans' effect gating (chance_consumers.plan_for_state, phase_d_chance.plan_for_state,
ordinary_law.plan_for_state) before updating REVIEWED. `python3 tests/test_gpu_engine_effect_stream_readers.py --print`
prints the current snapshot with each statement.
"""
import ast
import collections
import hashlib
from pathlib import Path
import sys
import unittest

TRANSITION = Path(__file__).resolve().parents[1] / 'gpu_engine' / 'transition.py'
STREAM_NAMES = frozenset({'effect_draws', '_effect'})   # loads and stores (an AugAssign target is a store)
STREAM_ATTRS = frozenset({'effect_draws', 'effect'})    # attribute loads, e.g. chance_inputs.effect, state.effect_draws
WHOLE = frozenset({'chance_inputs'})                    # any use of the whole object: indexing, unpacking, _asdict(), zip
FIELD_STRINGS = frozenset({'effect', 'effect_draws'})   # a string naming the field, e.g. getattr(x, 'effect')

# (enclosing function, the expression around the use, digest of the enclosing statement) -> count
REVIEWED = {
    # step_with_public: a missing ChanceInputs is sampled (the sampled path, never an enumerated one)
    ('step_with_public', 'chance_inputs is None', '1455bf5b131674e9'): 1,
    ('step_with_public', 'sample_chance_inputs(state) if chance_inputs is None else chance_inputs', '1455bf5b131674e9'): 1,
    # step_with_chance: the shape check over every field, and the six non-effect fields read by attribute
    ('step_with_chance', 'zip(chance_inputs, shapes, strict=True)', 'fa1719d7f1411a5c'): 1,
    ('step_with_chance', 'chance_inputs.draws', '653cd7a43c8cba1a'): 1,
    ('step_with_chance', 'chance_inputs.entry_ties', '57f87950e8336da9'): 1,
    ('step_with_chance', 'chance_inputs.initial_tie', '80e55ffb01720a47'): 1,
    ('step_with_chance', 'chance_inputs.initial_tie', '9b7a3d3766e1c57c'): 1,
    ('step_with_chance', 'chance_inputs.phazing', '01581d9737c0f395'): 1,
    ('step_with_chance', 'chance_inputs.secondary', 'f5b5a9ab28d576ff'): 1,
    ('step_with_chance', 'chance_inputs.sleep_talk', '6be5687bad2958b1'): 1,
    # the step's effect block: the caller's ChanceInputs.effect, sampled when None, the saved block when resuming
    ('step_with_chance', 'chance_inputs.effect', '89911b9f6b3ff76e'): 1,
    ('step_with_chance', 'effect_draws = chance_inputs.effect', '89911b9f6b3ff76e'): 2,
    ('step_with_chance', 'effect_draws is None', 'fedd903f55b03880'): 1,
    ('step_with_chance', 'effect_draws = effect_chance(splits[:, 1])', '0f130d6ceb9f662e'): 1,
    ('step_with_chance', 'effect_draws = jnp.where(resuming[:, None, None], state.effect_draws, effect_draws)',
     'b46e3faddf766b3e'): 1,
    ('step_with_chance', 'jnp.where(resuming[:, None, None], state.effect_draws, effect_draws)', 'b46e3faddf766b3e'): 2,
    # effect.s.0 / effect.s.1: Bonemerang's second hit, read only under _second. Column 0 is the crit check (== 0, or
    # crits.is_critical's high-crit threshold); column 1 is the damage roll 85+d in both damage calls.
    ('step_with_chance', '_effect = effect_draws[rows, side]', '7f5df1dc7c8e6bf2'): 1,
    ('step_with_chance', 'effect_draws[rows, side]', '7f5df1dc7c8e6bf2'): 1,
    ('step_with_chance', '_effect[:, 0]', '5a8ffe9b41bf21f8'): 1,
    ('step_with_chance', '_effect[:, 0]', '03f7a7675074d496'): 1,
    ('step_with_chance', '_effect[:, 1]', 'd8920e3f51f499bc'): 1,
    ('step_with_chance', '_effect[:, 1]', 'f743e714554dfea8'): 1,
    # effect.0.3: the Wrap pair tie bit
    ('step_with_chance', 'effect_draws[:, 0, 3, None]', '3d5baa07a30ba9d5'): 1,
    # effect.s.2: Yawn's sleep duration 2+d on the drowsy mon's side
    ('step_with_chance', 'effect_draws[rows, side, 2]', 'ea96e049c275d980'): 1,
    # effect.1.3: the Perish pair tie, used only when the end speeds are equal
    ('step_with_chance', 'effect_draws[:, 1, 3]', '6ce6f3f90b0dd01e'): 1,
    # the save of the block into state.effect_draws when the residual is deferred (a suspended turn)
    ('step_with_chance', 'jnp.where(defer_residual[:, None, None], effect_draws, 0)', 'fe528e3dd29798d3'): 1,
}


def _statement(node, parents):
    while not isinstance(node, ast.stmt):
        node = parents[node]
    return node


def _render(stmt):
    """A compound statement contributes only its header; its body's statements are keyed on their own."""
    if isinstance(stmt, (ast.If, ast.While)):
        return f'{type(stmt).__name__.lower()} {ast.unparse(stmt.test)}'
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        return f'for {ast.unparse(stmt.target)} in {ast.unparse(stmt.iter)}'
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return 'with ' + ', '.join(ast.unparse(item) for item in stmt.items)
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return f'def {stmt.name}({ast.unparse(stmt.args)})'
    return ast.unparse(stmt)


def readers(source):
    """(snapshot Counter, {key: rendered statement}, [line of each field-name string constant]) for `source`."""
    tree = ast.parse(source)
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def enclosing(node):
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node.name
        return '<module>'

    def around(node):
        parent = parents.get(node)
        if isinstance(parent, (ast.keyword, ast.Starred)):
            parent = parents[parent]
        return ast.unparse(parent)

    found, statements, strings = collections.Counter(), {}, []
    for node in ast.walk(tree):
        stream = isinstance(node, ast.Name) and node.id in STREAM_NAMES
        whole = isinstance(node, ast.Name) and node.id in WHOLE and isinstance(node.ctx, ast.Load)
        attr = isinstance(node, ast.Attribute) and node.attr in STREAM_ATTRS and isinstance(node.ctx, ast.Load)
        if isinstance(node, ast.Constant) and node.value in FIELD_STRINGS:
            strings.append(node.lineno)
        if stream or whole or attr:
            text = _render(_statement(node, parents))
            key = (enclosing(node), around(node), hashlib.sha256(text.encode()).hexdigest()[:16])
            found[key] += 1
            statements[key] = text
    return found, statements, strings


class EffectStreamReaderTests(unittest.TestCase):
    def test_every_effect_use_in_transition_is_reviewed(self):
        found, statements, strings = readers(TRANSITION.read_text())
        self.assertEqual(strings, [], 'transition.py names the effect field in a string (getattr or similar); the plans '
                         'cannot see such a reader. Read the field by attribute instead.')
        new = {k: v for k, v in found.items() if REVIEWED.get(k) != v}
        gone = {k: v for k, v in REVIEWED.items() if found.get(k) != v}
        detail = '\n'.join(f'  {k} x{v}\n    {statements[k]}' for k, v in new.items())
        self.assertEqual((new, gone), ({}, {}),
                         'transition.py uses the effect stream somewhere, or in a way, the exact-chance plans have not '
                         "reviewed. Re-check the effect gating in chance_consumers, phase_d_chance and ordinary_law's "
                         'plan_for_state, then update REVIEWED (--print shows the current snapshot).\nNew or changed:\n'
                         + detail)

    def test_the_scan_sees_each_kind_of_change(self):
        base = TRANSITION.read_text()
        snapshot = readers(base)

        def changed(text):
            return readers(text) != snapshot

        appended = {
            'new reader': '\ndef _new(effect_draws):\n    return effect_draws[:, 0, 2]\n',
            'alias': '\ndef _alias(effect_draws):\n    e = effect_draws\n    return e\n',
            'pass-through': '\ndef _through(effect_draws):\n    return helper(x=effect_draws)\n',
            'augmented assignment': '\ndef _aug(effect_draws):\n    effect_draws += 1\n',
            'getattr by name': '\ndef _named(c):\n    return getattr(c, "effect")\n',
            'positional field': '\ndef _positional(chance_inputs):\n    return chance_inputs[6]\n',
            'unpacking': '\ndef _unpack(chance_inputs):\n    *_, last = chance_inputs\n    return last\n',
        }
        for label, extra in appended.items():
            with self.subTest(label):
                self.assertTrue(changed(base + extra))
        # In-place rewrites of the reviewed consumers keep every read count the same but change the statement.
        crit, roll = 'critical2=_effect[:,0]==0', 'roll=85 + _effect[:, 1])'
        self.assertEqual((base.count(crit), base.count(roll)), (1, 2))
        rewrites = {
            'column swap': base.replace(crit, 'critical2=_effect[:,1]==0').replace(roll, 'roll=85 + _effect[:, 0])'),
            'threshold': base.replace(crit, 'critical2=_effect[:,0]<2'),
            'offset': base.replace(roll, 'roll=84 + _effect[:, 1])', 1),
        }
        for label, text in rewrites.items():
            with self.subTest(label):
                self.assertNotEqual(text, base)
                self.assertTrue(changed(text))


if __name__ == '__main__':
    if sys.argv[1:] == ['--print']:
        found, statements, strings = readers(TRANSITION.read_text())
        for key, count in sorted(found.items()):
            print(f'    {key!r}: {count},\n        # {statements[key][:160]}')
        print('field-name strings:', strings)
    else:
        unittest.main()
