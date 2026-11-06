"""Inert source/hash fixtures only; no production module is imported or run."""
import ast
import hashlib
from pathlib import Path
import unittest


GPU = Path(__file__).resolve().parents[1] / 'gpu_engine'
MODULES = tuple('action_history absorption baton combat critical damage_moves effects encore entry focus_punch items linked_traps mechanics phazing pressure pursuit secondary self_destruct sleep_talk solar_beam state sustain transition variable_moves weather'.split())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def bindings():
    tree = ast.parse((GPU / 'phase_d_chance.py').read_text())
    return {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in ('SOURCE_PINS', 'DOMAIN_SHA256')}


class ChanceBindingSourceTests(unittest.TestCase):
    def test_all_thirteen_literal_source_bindings_match_frozen_bytes(self):
        pins = bindings()['SOURCE_PINS']
        self.assertEqual(len(pins), 13)
        self.assertEqual(pins['transition.py'], '643de79452d7416ee52f841742f5d6a7c627a656f405e3998e67e0c530a05c5a')
        self.assertEqual({name: digest((GPU / name).read_bytes()) for name in pins}, pins)

    def test_combined_domain_binds_all_twenty_five_original_modules(self):
        tree = ast.parse((GPU / 'phase_d_chance.py').read_text())
        node = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'DOMAIN_MODULES')
        # Read the literal tuple("...".split()) source, without evaluating it.
        self.assertEqual(ast.literal_eval(node.args[0].func.value).split(), list(MODULES))
        self.assertEqual(len(MODULES), 25)
        actual = digest(b''.join(name.encode() + b'\0' + (GPU / (name + '.py')).read_bytes() + b'\0'
                                  for name in MODULES))
        self.assertEqual(actual, '9d2d21a53bd50ab2beb7b335841223b906328736584a130bfbecc8371d4094a2')
        self.assertEqual(actual, bindings()['DOMAIN_SHA256'])

    def test_consumer_program_is_byte_unchanged_outside_two_bindings(self):
        text = (GPU / 'phase_d_chance.py').read_text()
        program = ''.join(line for line in text.splitlines(keepends=True)
                          if not line.startswith(('SOURCE_PINS =', 'DOMAIN_SHA256 =')))
        self.assertEqual(digest(program.encode()), '3aa564c9e929dec710e1643bb5be25da3ee4393ca40835d34a8f21c57cfe9577')

    def test_mapper_constructor_trackers_search_and_finite_law_stay_frozen(self):
        # Literal file hashes from the explicit 3d811479 baseline. Transition
        # itself is already fixed by the source and transitive-domain fixtures.
        frozen = {
            'finite_law.py': '34317244b8430387ab46c148c578828f4d94da6ebb80d3556b197613f92703f0',
            'future_state.py': 'c22e4ee4978c0a5c59b9be33c1bae217f86a902f4a29a16e779195347c35c996',
            'phase_d_live_worlds.py': '013fb24aad256d0f531b3eb21683dd704b17e9e22d624e8709ab48632d77765d',
            'phase_d_protocol_world.py': '4e5e99683ff7598f64d64d9244f0ce14d864b5f6ed8489644569ae6fd48f9bfb',
            'device_public_world_state.py': 'f60ed7690b93de3c22c7499b2bfd3f9201618cbbf7e2a90798e2731ed66fbc87',
            'engine_device_public_hooks.py': '586d8e808191957fbba266f994e41afc123baea6e7aa3d4a321c82168686b7ec',
            'phase_d_search.py': '8db0eabc8dc0b143b9a287581254232a6b891b0f4028f7a9b7986f06b9735288',
        }
        self.assertEqual({name: digest((GPU / name).read_bytes()) for name in frozen}, frozen)


if __name__ == '__main__':
    unittest.main()
