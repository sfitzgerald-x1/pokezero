"""Pure-text optimization parity; no policy, game or search measurement."""
from concurrent.futures import ThreadPoolExecutor
import re
import sys
import unittest

from pokezero.randbat import _normalize_id, _normalize_plain_text
from _showdown_root import requires_showdown, showdown_root


def original(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


class NormalizationParityTests(unittest.TestCase):
    def setUp(self):
        _normalize_plain_text.cache_clear()
        self.addCleanup(_normalize_plain_text.cache_clear)

    def test_identifier_and_mixed_unicode_sequences_match_original(self):
        values = ["", "Mr. Mime", "Ho-Oh", "Hidden Power [Fire]", "Unown-?",
            "Return102", "p1: A B", "\u0130", "\u212a", "\u00df", "\u017f",
            "A\u0301 Z\U0001f4099", "\ud800X\udfff", "a" * 128, "B!" * 1000]
        for left in values:
            for right in values:
                text = left + right
                with self.subTest(text=repr(text)):
                    self.assertEqual(_normalize_id(text), original(text))

    def test_every_unicode_codepoint_matches_original(self):
        # Include surrogates: the original operation is Python text, not UTF8 IO.
        for codepoint in range(sys.maxunicode + 1):
            text = chr(codepoint)
            actual, expected = _normalize_id(text), original(text)
            if actual != expected:
                self.fail(f"Unicode codepoint{codepoint:x}: {actual!r} != {expected!r}")

    def test_dynamic_unhashable_objects_are_converted_on_every_call(self):
        class Dynamic:
            __hash__ = None
            def __init__(self):
                self.calls = 0
            def __str__(self):
                self.calls += 1
                return ("Mr. Mime", "Ho-Oh", "Hidden Power Fire")[self.calls - 1]
            def __eq__(self, other):
                raise AssertionError("original objects must not become cache keys")
        value = Dynamic()
        self.assertEqual([_normalize_id(value) for _ in range(3)],
            ["mrmime", "hooh", "hiddenpowerfire"])
        self.assertEqual(value.calls, 3)

    def test_conversion_failures_are_not_hidden_by_a_cached_prior_result(self):
        class Dynamic:
            def __init__(self):
                self.calls = 0
            def __str__(self):
                self.calls += 1
                if self.calls > 1:
                    raise ValueError("conversion failure")
                return "Ice Beam"
        value = Dynamic()
        self.assertEqual(_normalize_id(value), "icebeam")
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, "conversion failure"):
                _normalize_id(value)
        self.assertEqual(value.calls, 3)

    def test_invalid_str_return_preserves_type_error(self):
        class Invalid:
            def __str__(self):
                return 7
        for normalize in (original, _normalize_id):
            with self.assertRaises(TypeError):
                normalize(Invalid())

    def test_str_subclass_keeps_lower_and_exception_behavior(self):
        class Observable(str):
            calls = 0
            def __str__(self):
                return self
            def lower(self):
                self.calls += 1
                if self.calls > 2:
                    raise RuntimeError("lower failure")
                return ("Fire Blast", "Ice Beam")[self.calls - 1]
        value = Observable("Original")
        self.assertEqual(type(str(value)), Observable)
        # This custom lower() deliberately returns uppercase text. The original
        # regex strips those capitals; do not silently normalize a second time.
        self.assertEqual(_normalize_id(value), "irelast")
        self.assertEqual(_normalize_id(value), "ceeam")
        with self.assertRaisesRegex(RuntimeError, "lower failure"):
            _normalize_id(value)
        self.assertEqual(_normalize_plain_text.cache_info().currsize, 0)

    def test_custom_conversion_returning_str_subclass_is_not_cached(self):
        class Text(str):
            def lower(self):
                return "Thunder Wave"
            def __hash__(self):
                raise AssertionError("str subclasses must not be hashed")
        class Source:
            def __str__(self):
                return Text("different contents")
        self.assertEqual(_normalize_id(Source()), original(Source()))
        self.assertEqual(_normalize_plain_text.cache_info().currsize, 0)

    def test_cache_has_fixed_entry_bound_and_evicts_without_changing_results(self):
        self.assertEqual(_normalize_plain_text.cache_parameters()["maxsize"], 4096)
        for index in range(4200):
            text = f"Ability {index}"
            self.assertEqual(_normalize_id(text), original(text))
        info = _normalize_plain_text.cache_info()
        self.assertEqual(info.currsize, 4096)
        self.assertEqual(info.misses, 4200)
        self.assertEqual(_normalize_id("Ability 0"), "ability0")
        self.assertEqual(_normalize_plain_text.cache_info().misses, 4201)

    def test_long_strings_bypass_cache_and_short_exact_strings_hit(self):
        text = "A!" * 65
        before = _normalize_plain_text.cache_info()
        for _ in range(3):
            self.assertEqual(_normalize_id(text), original(text))
        self.assertEqual(_normalize_plain_text.cache_info(), before)
        _normalize_id("A" * 128)
        _normalize_id("A" * 128)
        info = _normalize_plain_text.cache_info()
        self.assertEqual((info.misses, info.hits, info.currsize), (1, 1, 1))
        for value in (["Ice Beam"], {"move": "Ice Beam"}, None, 7, b"Fire Blast"):
            self.assertEqual(_normalize_id(value), original(value))

    def test_concurrent_calls_match_original_and_keep_cache_bounded(self):
        inputs = [f"Move {i % 137}!" for i in range(4096)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(_normalize_id, inputs))
        self.assertEqual(results, [original(value) for value in inputs])
        self.assertLessEqual(_normalize_plain_text.cache_info().currsize, 4096)


@requires_showdown()
class CatalogNormalizationParityTests(unittest.TestCase):
    def test_pinned_full_catalog_entities_and_categories_match_original(self):
        from pokezero.randbat_vocab import gen3_randbat_category_strings, gen3_randbat_entities
        strings = gen3_randbat_category_strings(showdown_root())
        entities = gen3_randbat_entities(showdown_root())
        values = [value for group in (*strings.values(), *entities.values()) for value in group]
        self.assertGreater(len(values), 1000)
        for value in values:
            with self.subTest(value=value):
                self.assertEqual(_normalize_id(value), original(value))


if __name__ == "__main__":
    unittest.main()
