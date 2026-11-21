"""The per-module domain gates (chance_consumers, phase_d_chance, ordinary_law, conditional_law) refuse exactly what the
single DOMAIN_SHA256 / KERNEL_SHA256 literals they replaced refused, and now name the module.

- Equivalence at the change. At its base (71fe82d8) each aggregate literal is the digest of the base's domain bytes;
  at the change (790a9715) every per-module pin is the sha256 of those same bytes, in the same module order. So the
  per-module gates admit exactly the tree the aggregate gates admitted.
- A one-byte change to ANY of the 29 domain modules, as the real gate code reads it, refuses in all four gates and names
  that module.
- A pin that does not match its source (the table edited, not the module) refuses. So does a module list that the
  table no longer names in order, since the aggregate bound names and order as well as bytes.
- The plan gates refuse through plan_for_state, end to end, before any planning."""
from functools import partial
import hashlib
from pathlib import Path
import re
import subprocess
import unittest
from unittest.mock import patch

from gpu_engine import chance_consumers as cc, conditional_law as cl, ordinary_law as ol, phase_d_chance as pd
from gpu_engine import future_state as mapping
from tests import _source_pins as pins
from test_gpu_engine_chance_consumers import fixture, table
from test_gpu_engine_future_state import FAKE

ENGINE = Path(cc.__file__).parent
BASE = '71fe82d8953a48d53910c3b29eb0b2037fa849ca'     # integrated tree with aggregate literals
CHANGE = '790a9715be8bc49181a43a3f763422e89936c5b2'   # per-module pins at the same integrated domain bytes
READ = Path.read_bytes


def one_byte(target):
    """Path.read_bytes as the gates see it, with the first byte of gpu_engine/`target` flipped."""
    def read(self):
        data = READ(self)
        return bytes([data[0] ^ 1]) + data[1:] if self.name == target and self.parent == ENGINE else data
    return patch.object(Path, 'read_bytes', read)


def plan(module, data, meta):
    construct = partial(mapping.native_input_from_jax, module=FAKE)
    with patch.object(module.mapping, 'native_input_from_jax', side_effect=construct):
        return module.plan_for_state(data, None, (0, 0), table(meta), meta, cap=100_000_000)


def show(ref, path):
    return subprocess.check_output(['git', 'show', f'{ref}:{path}'], cwd=ENGINE.parent)


class DomainGateTests(unittest.TestCase):
    def test_the_per_module_pins_admit_exactly_what_the_aggregates_admitted(self):
        base = {n: show(BASE, f'gpu_engine/{n}.py') for n in pins.modules('chance_consumers.py', 'DOMAIN_MODULES', show(BASE, 'gpu_engine/chance_consumers.py').decode())}
        read = lambda name: base[name[:-3]]
        for file, literal, holder, listing in (('chance_consumers.py', 'DOMAIN_SHA256', 'chance_consumers.py', 'DOMAIN_MODULES'),
                                               ('phase_d_chance.py', 'DOMAIN_SHA256', 'phase_d_chance.py', 'DOMAIN_MODULES'),
                                               ('ordinary_law.py', 'DOMAIN_SHA256', 'chance_consumers.py', 'DOMAIN_MODULES'),
                                               ('conditional_law.py', 'KERNEL_SHA256', 'conditional_law.py', 'KERNEL_MODULES')):
            with self.subTest(aggregate=f'{file}:{literal}'):
                text = show(BASE, f'gpu_engine/{file}').decode()
                old = re.search(rf"^{literal} = '([0-9a-f]{{64}})'$", text, re.M).group(1)
                names = pins.modules(holder, listing, show(BASE, f'gpu_engine/{holder}').decode())
                self.assertEqual(pins.domain_digest(names, read), old)
        for (file, var), (holder, listing) in pins.DOMAIN_TABLES.items():
            with self.subTest(table=f'{file}:{var}'):
                names = pins.modules(holder, listing, show(BASE, f'gpu_engine/{holder}').decode())
                self.assertEqual(pins.modules(holder, listing, show(CHANGE, f'gpu_engine/{holder}').decode()), names)
                at_change = pins.table(file, var, show(CHANGE, f'gpu_engine/{file}').decode())
                self.assertEqual(list(at_change.items()), [(n + '.py', hashlib.sha256(base[n]).hexdigest()) for n in names])

    def test_the_gates_admit_the_tree_as_it_is(self):
        self.assertIsNone(cc._domain_drift(ENGINE))
        self.assertIsNone(pd._domain_drift(ENGINE))
        cl._source_certificate()   # no native build in it; ordinary_law's needs one, so its domain gate is shown refusing below

    def test_a_one_byte_change_to_any_domain_module_refuses_in_every_gate_naming_it(self):
        self.assertEqual(len(cc.DOMAIN_MODULES), pins.PIN_COUNTS[('chance_consumers_pins.py', 'DOMAIN_PINS')])
        self.assertEqual(cc.DOMAIN_MODULES, pd.DOMAIN_MODULES)
        self.assertEqual(cc.DOMAIN_MODULES, cl.KERNEL_MODULES)
        for name in cc.DOMAIN_MODULES:
            with self.subTest(module=name), one_byte(name + '.py'):
                self.assertEqual(cc._domain_drift(ENGINE), name + '.py')
                self.assertEqual(pd._domain_drift(ENGINE), name + '.py')
                with self.assertRaisesRegex(ValueError, rf'^ORDINARY_CONSUMER_DOMAIN_DRIFT: {name}\.py$'):
                    ol._source_certificate()
                with self.assertRaisesRegex(ValueError, rf'^CONSUMER_DOMAIN_DRIFT: {name}\.py$'):
                    cl._source_certificate()

    def test_the_plan_gates_refuse_a_domain_change_end_to_end(self):
        # items.py is a domain module that neither SOURCE_PINS names, so the refusal is the domain gate's own
        self.assertNotIn('items.py', cc.SOURCE_PINS)
        self.assertNotIn('items.py', pd.SOURCE_PINS)
        for module in (cc, pd):
            with self.subTest(gate=module.__name__):
                data, meta = fixture()
                with one_byte('items.py'):
                    out = plan(module, data, meta)
                self.assertIsInstance(out, mapping.Refused)
                self.assertEqual((out.gaps[0].code, out.gaps[0].detail), ('CONSUMER_DOMAIN_DRIFT', 'items.py'))
                with patch.dict(module.DOMAIN_PINS, {'items.py': '0' * 64}):
                    out = plan(module, data, meta)
                self.assertEqual((out.gaps[0].code, out.gaps[0].detail), ('CONSUMER_DOMAIN_DRIFT', 'items.py'))

    def test_plan_identity_uses_the_same_domain_bytes_the_gate_checked(self):
        expected = pins.domain_digest(cc.DOMAIN_MODULES, lambda name: READ(ENGINE/name))
        for module in (cc, pd):
            with self.subTest(gate=module.__name__):
                data, meta = fixture()
                reads = []

                def changing_read(path):
                    content = READ(path)
                    if path == ENGINE/'items.py':
                        reads.append(content)
                        if len(reads) > 1:
                            return bytes([content[0] ^ 1]) + content[1:]
                    return content

                with patch.object(Path, 'read_bytes', changing_read):
                    result = plan(module, data, meta)
                self.assertIsInstance(result, module.ConsumerPlan)
                self.assertEqual(dict(result.source_certificate)['transitive_kernel_domain'], expected)
                self.assertEqual(len(reads), 1)

    def test_a_pin_that_does_not_match_its_source_refuses(self):
        self.assertIs(cc.DOMAIN_PINS, __import__('gpu_engine.chance_consumers_pins', fromlist=['DOMAIN_PINS']).DOMAIN_PINS)
        wrong = {'transition.py': '0' * 64}
        with patch.dict(cc.DOMAIN_PINS, wrong):
            self.assertEqual(cc._domain_drift(ENGINE), 'transition.py')
        with patch.dict(pd.DOMAIN_PINS, wrong):
            self.assertEqual(pd._domain_drift(ENGINE), 'transition.py')
        with patch.dict(ol.DOMAIN_PINS, wrong), self.assertRaisesRegex(ValueError, r'^ORDINARY_CONSUMER_DOMAIN_DRIFT: transition\.py$'):
            ol._source_certificate()
        with patch.dict(cl.KERNEL_PINS, wrong), self.assertRaisesRegex(ValueError, r'^CONSUMER_DOMAIN_DRIFT: transition\.py$'):
            cl._source_certificate()

    def test_a_module_list_the_table_no_longer_names_in_order_refuses(self):
        swapped = (cc.DOMAIN_MODULES[1], cc.DOMAIN_MODULES[0]) + cc.DOMAIN_MODULES[2:]
        with patch.object(cc, 'DOMAIN_MODULES', swapped):
            self.assertEqual(cc._domain_drift(ENGINE), 'DOMAIN_MODULES')
            with self.assertRaisesRegex(ValueError, r'^ORDINARY_CONSUMER_DOMAIN_DRIFT: DOMAIN_MODULES$'):
                ol._source_certificate()   # ordinary_law hashes chance_consumers.DOMAIN_MODULES
        with patch.object(pd, 'DOMAIN_MODULES', pd.DOMAIN_MODULES[:-1]):
            self.assertEqual(pd._domain_drift(ENGINE), 'DOMAIN_MODULES')
        with patch.object(cl, 'KERNEL_MODULES', swapped), self.assertRaisesRegex(ValueError, r'^CONSUMER_DOMAIN_DRIFT: KERNEL_MODULES$'):
            cl._source_certificate()


if __name__ == '__main__':
    unittest.main()
