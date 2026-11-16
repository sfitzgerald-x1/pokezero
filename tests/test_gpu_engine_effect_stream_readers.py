"""Every use of transition's effect stream is a consumer the exact-chance plans observe; pure Python, no JAX import.

finite_law's effect.{s}.{c} axes are enumerated only where chance_consumers, phase_d_chance and ordinary_law gate them, and
an inactive axis carries a fixed 0. So the proof is lossless only while every place transition.py reads an effect value is
one of the consumers those plans observe, reading the column and with the arithmetic the plans assume. A new, moved or
rewritten reader would misprice an exact root without any error. transition.py's SOURCE_PINS refuse any edit until the pins
are refreshed, but a refresh is mechanical (scripts/refresh_source_pins.py). This test makes it insufficient: every use of
the stream, and of the whole ChanceInputs, is keyed by its enclosing function, the expression around it and a digest of its
whole enclosing statement, and so is every assignment to the gates that switch a consumer on (GATES). So a new reader, a column swap, or a changed threshold or offset changes the snapshot, and its
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
# The conditions that switch an effect consumer on, which the plans' gating copies: every assignment to these names is
# keyed too, so a changed gate (say hits>1 becoming hits>0) needs the same re-review as a changed read.
GATES = frozenset({'_second', '_yawn_counter', '_yawn_holder', '_yawn_sleep', '_wrap_holder', '_wrap_held', '_wrap_tick',
                   '_perish_continues', '_perish_holder', '_perish_first', 'tie', 'event_keys'})

# (enclosing function, the expression around the use, digest of the enclosing statement) -> count
# Infatuation review: old columns/saved replay are unchanged, the independent
# fifth column is observed for an existing relation or same-turn opposing Attract.
REVIEWED = {
    ('step_with_chance', '(effect_draws, effect_chance(splits[:, 1])[..., 4:])', 'a2d13e46f33d9f41'): 1,
    ('step_with_chance', '_effect = effect_draws[rows, side]', '7f5df1dc7c8e6bf2'): 1,
    ('step_with_chance', '_effect[:, 0]', '03f7a7675074d496'): 1,
    ('step_with_chance', '_effect[:, 0]', '5a8ffe9b41bf21f8'): 1,
    ('step_with_chance', '_effect[:, 1]', 'd8920e3f51f499bc'): 1,
    ('step_with_chance', '_effect[:, 1]', 'f743e714554dfea8'): 1,
    ('step_with_chance', 'chance_inputs.draws', '653cd7a43c8cba1a'): 1,
    ('step_with_chance', 'chance_inputs.effect', '89911b9f6b3ff76e'): 1,
    ('step_with_chance', 'chance_inputs.entry_ties', '57f87950e8336da9'): 1,
    ('step_with_chance', 'chance_inputs.initial_tie', '6461257890dce00a'): 1,
    ('step_with_chance', 'chance_inputs.initial_tie', '9b7a3d3766e1c57c'): 1,
    ('step_with_chance', 'chance_inputs.phazing', '01581d9737c0f395'): 1,
    ('step_with_chance', 'chance_inputs.secondary', 'f5b5a9ab28d576ff'): 1,
    ('step_with_chance', 'chance_inputs.sleep_talk', '6be5687bad2958b1'): 1,
    ('step_with_chance', 'effect_draws = chance_inputs.effect', '89911b9f6b3ff76e'): 2,
    ('step_with_chance', 'effect_draws = effect_chance(splits[:, 1])', '0f130d6ceb9f662e'): 1,
    ('step_with_chance', 'effect_draws = jnp.concatenate((effect_draws, effect_chance(splits[:, 1])[..., 4:]), axis=-1)', 'a2d13e46f33d9f41'): 1,
    ('step_with_chance', 'effect_draws = jnp.where(resuming[:, None, None], state.effect_draws, effect_draws)', 'b46e3faddf766b3e'): 1,
    ('step_with_chance', 'effect_draws is None', 'fedd903f55b03880'): 1,
    ('step_with_chance', 'effect_draws.shape', 'e62018b93d213ded'): 1,
    ('step_with_chance', 'effect_draws[:, 0, 3, None]', '3d5baa07a30ba9d5'): 1,
    ('step_with_chance', 'effect_draws[:, 1, 3]', '6ce6f3f90b0dd01e'): 1,
    ('step_with_chance', 'effect_draws[:, side, 2]', 'cb5783f78d5814b5'): 1,
    ('step_with_chance', 'effect_draws[rows, side, 2]', '846f2df9dab07bc4'): 1,
    ('step_with_chance', 'effect_draws[rows, side, 2]', 'e16344807b2214e1'): 1,
    ('step_with_chance', 'effect_draws[rows, side, 2]', 'ea96e049c275d980'): 1,
    ('step_with_chance', 'effect_draws[rows, side, 4]', '57342f1bbef6bd5b'): 1,
    ('step_with_chance', 'effect_draws[rows, side]', '7f5df1dc7c8e6bf2'): 1,
    ('step_with_chance', 'gate _perish_continues', 'c56f7684931b73f4'): 1,
    ('step_with_chance', 'gate _perish_first', '6ce6f3f90b0dd01e'): 1,
    ('step_with_chance', 'gate _perish_holder', 'c3217ad029f1bd93'): 1,
    ('step_with_chance', 'gate _second', 'c9e42262b234db09'): 1,
    ('step_with_chance', 'gate _wrap_held', '22569d4bbf65f598'): 1,
    ('step_with_chance', 'gate _wrap_holder', '7a99fb53264f7804'): 1,
    ('step_with_chance', 'gate _wrap_tick', 'd01dc288c92376e7'): 1,
    ('step_with_chance', 'gate _yawn_counter', 'e8e7928c1fd8ab6e'): 1,
    ('step_with_chance', 'gate _yawn_holder', 'f39efe625b1ee9be'): 1,
    ('step_with_chance', 'gate _yawn_sleep', '0088b5859e70205d'): 1,
    ('step_with_chance', 'gate _yawn_sleep', 'c57fe97b2ab6eeec'): 1,
    ('step_with_chance', 'gate event_keys', '06962d48c87c7b2f'): 1,
    ('step_with_chance', 'gate tie', '3d5baa07a30ba9d5'): 1,
    ('step_with_chance', 'jnp.where(defer_residual[:, None, None], effect_draws, 0)', 'fe528e3dd29798d3'): 1,
    ('step_with_chance', 'jnp.where(resuming[:, None, None], state.effect_draws, effect_draws)', 'b46e3faddf766b3e'): 2,
    ('step_with_chance', 'zip(chance_inputs, shapes, strict=True)', 'fa1719d7f1411a5c'): 1,
    ('step_with_public', 'chance_inputs is None', '1455bf5b131674e9'): 1,
    ('step_with_public', 'sample_chance_inputs(state) if chance_inputs is None else chance_inputs', '1455bf5b131674e9'): 1,
}


# Fidelity tier D: the effect[s][2] value transition passes on as `sleep_roll` is consumed in secondary.apply_contact
# (2 + sleep_roll for an Effect Spore sleep) and passed through by pursuit.intercept. Every use of that name in those
# modules, and every assignment to the gates that decide an Effect Spore sleep, is keyed the same way.
SLEEP_ROLL_MODULES = ('secondary.py', 'pursuit.py')
SLEEP_ROLL_NAMES = frozenset({'sleep_roll'})
SLEEP_ROLL_GATES = frozenset({'spore', 'status', 'rolled', 'proc', 'slept'})
SLEEP_ROLL_FUNCTIONS = frozenset({'apply_contact', 'intercept'})   # the gate names recur elsewhere in secondary.py


def sleep_roll_readers():
    found, statements = collections.Counter(), {}
    for name in SLEEP_ROLL_MODULES:
        counts, texts, _ = readers((TRANSITION.with_name(name)).read_text(), names=SLEEP_ROLL_NAMES, attrs=frozenset(),
                                   whole_names=frozenset(), gates=SLEEP_ROLL_GATES)
        for key, count in counts.items():
            if key[0] in SLEEP_ROLL_FUNCTIONS:
                found[(name, *key)] += count;statements[(name, *key)] = texts[key]
    return found, statements
SLEEP_ROLL_REVIEWED = {
    ('pursuit.py', 'intercept', 'jnp.zeros_like(contact_roll) if sleep_roll is None else sleep_roll', '78dc27448c71386a'): 1,
    ('pursuit.py', 'intercept', 'sleep_roll is None', '78dc27448c71386a'): 1,
    ('secondary.py', 'apply_contact', 'gate proc', '4223e413f02161c2'): 1,
    ('secondary.py', 'apply_contact', 'gate rolled', '3becf7407f15669f'): 1,
    ('secondary.py', 'apply_contact', 'gate slept', '4ea0471ca753fbc2'): 1,
    ('secondary.py', 'apply_contact', 'gate spore', '78091197fa96a1e4'): 1,
    ('secondary.py', 'apply_contact', 'gate status', 'dfc5804fe1a7d584'): 1,
    ('secondary.py', 'apply_contact', 'gate status', 'e486262ac6f66245'): 1,
    ('secondary.py', 'apply_contact', 'jnp.asarray(sleep_roll, jnp.int32)', '18069047a31d8b59'): 1,
    ('secondary.py', 'apply_contact', 'sleep_roll is None', 'c61efa852f8640ca'): 1,
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


def readers(source, names=STREAM_NAMES, attrs=STREAM_ATTRS, whole_names=WHOLE, gates=GATES):
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
        stream = isinstance(node, ast.Name) and node.id in names
        whole = isinstance(node, ast.Name) and node.id in whole_names and isinstance(node.ctx, ast.Load)
        attr = isinstance(node, ast.Attribute) and node.attr in attrs and isinstance(node.ctx, ast.Load)
        if isinstance(node, ast.Constant) and node.value in FIELD_STRINGS:
            strings.append(node.lineno)
        if stream or whole or attr:
            text = _render(_statement(node, parents))
            key = (enclosing(node), around(node), hashlib.sha256(text.encode()).hexdigest()[:16])
            found[key] += 1
            statements[key] = text
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            found_gates = sorted({n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name) and n.id in gates})
            for gate in found_gates:
                text = ast.unparse(node)
                key = (enclosing(node), 'gate ' + gate, hashlib.sha256(text.encode()).hexdigest()[:16])
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

    def test_every_sleep_roll_use_is_reviewed(self):
        found, statements = sleep_roll_readers()
        new = {k: v for k, v in found.items() if SLEEP_ROLL_REVIEWED.get(k) != v}
        gone = {k: v for k, v in SLEEP_ROLL_REVIEWED.items() if found.get(k) != v}
        detail = '\n'.join(f'  {k} x{v}\n    {statements[k]}' for k, v in new.items())
        self.assertEqual((new, gone), ({}, {}), 'The Effect Spore sleep duration (effect[s][2], sleep_roll) is used or '
                         'gated somewhere, or in a way, the exact-chance plans have not reviewed. Re-check the plans\' '
                         'effect.s.2 gating, then update SLEEP_ROLL_REVIEWED (--print-sleep shows the snapshot).\n' + detail)

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
            'second-hit gate': base.replace('(table.damage_moves.hits[move]>1)', '(table.damage_moves.hits[move]>0)'),
            'yawn gate': base.replace('(_yawn_counter==1)', '(_yawn_counter==2)'),
        }
        for label, text in rewrites.items():
            with self.subTest(label):
                self.assertNotEqual(text, base)
                self.assertTrue(changed(text))


if __name__ == '__main__':
    if sys.argv[1:] == ['--print-sleep']:
        found, statements = sleep_roll_readers()
        for key, count in sorted(found.items()):
            print(f'    {key!r}: {count},\n        # {statements[key][:160]}')
    elif sys.argv[1:] == ['--print']:
        found, statements, strings = readers(TRANSITION.read_text())
        for key, count in sorted(found.items()):
            print(f'    {key!r}: {count},\n        # {statements[key][:160]}')
        print('field-name strings:', strings)
    else:
        unittest.main()
