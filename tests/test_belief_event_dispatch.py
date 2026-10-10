"""Keep public event extraction identical to the original typing dispatch."""
from collections import UserDict
from collections.abc import Mapping as RuntimeMapping
from copy import deepcopy
from types import MappingProxyType, SimpleNamespace
from typing import Mapping
import unittest
from unittest.mock import patch

import pokezero.belief as belief
import test_policy_opponent_prefix as prefix_tests


def legacy_event_value(event, name):
    if isinstance(event, Mapping):
        value = event.get(name)
    else:
        value = getattr(event, name, None)
    return str(value) if value is not None else None


class BeliefEventDispatchTests(unittest.TestCase):
    def test_mapping_and_attribute_values_match_original_dispatch(self):
        class CustomMapping(RuntimeMapping):
            value = 'attribute must not be used'
            def __getitem__(self, name):
                if name != 'value':
                    raise KeyError(name)
                return 'mapping'
            def __iter__(self):
                return iter(('value',))
            def __len__(self):
                return 1
        class VirtualMapping:
            value = 'attribute must not be used'
            def get(self, name):
                return 'virtual' if name == 'value' else None
        RuntimeMapping.register(VirtualMapping)
        events = [{}, SimpleNamespace(), CustomMapping(), VirtualMapping()]
        for value in (None, False, 0, '', 'text'):
            events.extend(({'value': value}, UserDict(value=value),
                MappingProxyType({'value': value}), SimpleNamespace(value=value)))
        for event in events:
            for name in ('value', 'missing'):
                with self.subTest(event_type=type(event).__name__, name=name):
                    self.assertEqual(belief._event_value(event, name), legacy_event_value(event, name))

    def test_hostile_class_lookup_conversion_order_and_errors_are_preserved(self):
        class Spoofed:
            value = 'attribute'
            @property
            def __class__(self):
                return dict
            def get(self, name):
                raise AssertionError('must not call get')
        class Hostile(Spoofed):
            @property
            def __class__(self):
                raise AssertionError('must not inspect __class__')
        class Value:
            def __init__(self, trace, raises):
                self.trace, self.raises = trace, raises
            def __str__(self):
                self.trace.append('str')
                if self.raises:
                    raise ValueError('conversion failure')
                return 'converted'
        class TracedDict(dict):
            def __init__(self, trace, lookup_raises, conversion_raises):
                self.trace, self.lookup_raises = trace, lookup_raises
                super().__init__(value=Value(trace, conversion_raises))
            def get(self, name):
                self.trace.append(('get', name))
                if self.lookup_raises:
                    raise KeyError('lookup failure')
                return super().get(name)
        class TracedAttribute:
            def __init__(self, trace, lookup_raises, conversion_raises):
                self.trace, self.lookup_raises, self.conversion_raises = trace, lookup_raises, conversion_raises
            @property
            def value(self):
                self.trace.append('getattr')
                if self.lookup_raises:
                    raise ValueError('attribute failure')
                return Value(self.trace, self.conversion_raises)
        factories = [lambda trace: Spoofed(), lambda trace: Hostile()]
        for kind in (TracedDict, TracedAttribute):
            for lookup_raises, conversion_raises in ((False, False), (True, False), (False, True)):
                factories.append(lambda trace, k=kind, l=lookup_raises, c=conversion_raises: k(trace, l, c))
        for index, factory in enumerate(factories):
            outcomes = []
            for method in (legacy_event_value, belief._event_value):
                trace = []
                try:
                    outcome = ('return', method(factory(trace), 'value'))
                except Exception as error:
                    outcome = ('raise', type(error), str(error))
                outcomes.append((outcome, trace))
            with self.subTest(case=index):
                self.assertEqual(outcomes[0], outcomes[1])

    def test_full_public_views_and_source_calls_match_original_in_both_seats(self):
        case = prefix_tests.PublicPrefixParityTests()
        for mirrored in (False, True):
            lines = prefix_tests.projected(prefix_tests.mirror(prefix_tests.history())
                if mirrored else prefix_tests.history())
            supplied = prefix_tests.bundle()
            if mirrored:
                import json
                supplied = json.loads(json.dumps(supplied).replace('p2:', 'p1:').replace('"p2"', '"p1"'))
            before = deepcopy(supplied)
            for cut in (len(prefix_tests.LINES), 10, len(lines)):
                options = dict(opponent_slot='p1' if mirrored else 'p2')
                source = prefix_tests.CountingSource()
                with patch.object(belief, '_event_value', legacy_event_value):
                    reference = case.make(lines[:cut], supplied, set_source=source, **options)
                legacy_calls = deepcopy(source.calls)
                # The complete engine comparison includes the source's identity.
                # Reset only this test double's trace so both builds begin alike.
                source.calls.clear()
                actual = case.make(lines[:cut], supplied, set_source=source, **options)
                with self.subTest(mirrored=mirrored, cut=cut):
                    case.assert_views(reference, actual)
                    self.assertEqual(legacy_calls, source.calls)
                    self.assertEqual(supplied, before)
