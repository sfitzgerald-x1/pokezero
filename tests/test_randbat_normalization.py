"""Pure-text optimization parity; no policy, game or search measurement."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from itertools import combinations
import re
import sys
import unittest

from pokezero.randbat import (
    Gen3RandbatSource, Gen3RandbatVariant, _normalize_id, _normalize_move,
    _normalize_plain_text, _normalized_variant_move_set, _revealed_move_matches_variant,
)
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


def legacy_variant_matches(variant, *, revealed_moves=(), revealed_ability=None,
        revealed_item=None, ruled_out_abilities=(), ruled_out_items=()):
    # Independent copy of the complete pre-cache match operation and order.
    normalized_moves = {_normalize_move(move) for move in variant.moves}
    if any(not _revealed_move_matches_variant(move, normalized_moves) for move in revealed_moves):
        return False
    if _normalize_id(variant.ability) in {_normalize_id(a) for a in ruled_out_abilities}:
        return False
    if revealed_ability and _normalize_id(variant.ability) != _normalize_id(revealed_ability):
        return False
    if revealed_item and _normalize_id(variant.item) != _normalize_id(revealed_item):
        return False
    if _normalize_id(variant.item) in {_normalize_id(item) for item in ruled_out_items}:
        return False
    return True


def catalog_variant(moves=("Surf", "Ice Beam", "Hidden Power [Fire]", "Recover")):
    return Gen3RandbatVariant("a", "a", "Starmie", "fixture", 100,
                             moves, "Natural Cure", "Leftovers")


class VariantMoveCacheParityTests(unittest.TestCase):
    def setUp(self):
        _normalized_variant_move_set.cache_clear()
        self.addCleanup(_normalized_variant_move_set.cache_clear)

    def test_exact_tuple_hits_and_retains_immutable_normalized_moves(self):
        variant = catalog_variant()
        self.assertTrue(variant.matches(revealed_moves=("Hidden Power",)))
        self.assertTrue(variant.matches(revealed_moves=("Surf",)))
        info = _normalized_variant_move_set.cache_info()
        self.assertEqual((info.misses, info.hits, info.currsize), (1, 1, 1))
        stored = _normalized_variant_move_set(variant.moves)
        self.assertIs(type(stored), frozenset)
        with self.assertRaises(AttributeError):
            stored.add("thunderbolt")
        fresh = variant.to_summary()
        fresh["moves"].append("Thunderbolt")
        self.assertEqual(variant.to_summary()["moves"], list(variant.moves))
        self.assertFalse(variant.matches(revealed_moves=("Thunderbolt",)))

    def test_unicode_hiddenpower_duplicate_and_query_matrix_matches_original(self):
        move_sets = ((), ("Surf", "Surf"), ("Hidden Power Ice",),
                     ("\u0130", "\u212a", "\ud800X", "A\u0301"), catalog_variant().moves)
        for moves in move_sets:
            variant = catalog_variant(moves)
            for revealed in ((), ("Surf",), ("Hidden Power",), ("Hidden Power Ice",),
                             ("Hidden Power Fire",), ("Surf", "Recover"), ("Secret",)):
                for ability, item, ban_ability, ban_item in (
                        (None, None, (), ()), ("Natural Cure", "Leftovers", (), ()),
                        ("Unknown", None, (), ()), (None, "Unknown", (), ()),
                        (None, None, ("Natural Cure",), ()),
                        (None, None, (), ("Leftovers",))):
                    query = dict(revealed_moves=revealed, revealed_ability=ability,
                        revealed_item=item, ruled_out_abilities=ban_ability,
                        ruled_out_items=ban_item)
                    self.assertEqual(variant.matches(**query), legacy_variant_matches(variant, **query))

    def test_dynamic_custom_move_text_preserves_conversion_and_exception_order(self):
        class Dynamic(str):
            calls = 0
            def __str__(self):
                return self
            def lower(self):
                self.calls += 1
                if self.calls > 2:
                    raise ValueError("dynamic conversion")
                return ("surf", "icebeam")[self.calls - 1]
            def __hash__(self):
                raise AssertionError("custom string must not enter tuple cache")
        left, right = Dynamic("x"), Dynamic("x")
        actual, expected = catalog_variant((left,)), catalog_variant((right,))
        for _ in range(2):
            self.assertEqual(actual.matches(revealed_moves=("Surf",)),
                legacy_variant_matches(expected, revealed_moves=("Surf",)))
        for variant, match in ((actual, actual.matches),
                               (expected, lambda **q: legacy_variant_matches(expected, **q))):
            with self.assertRaisesRegex(ValueError, "dynamic conversion"):
                match(revealed_moves=("Surf",))
        self.assertEqual((left.calls, right.calls), (3, 3))
        self.assertEqual(_normalized_variant_move_set.cache_info().currsize, 0)
        class Exploding:
            def __str__(self):
                raise ValueError("query order")
        variant = catalog_variant()
        # Original lazy rejection never converts the later ability/item query.
        self.assertFalse(variant.matches(revealed_moves=("Missing",), revealed_ability=Exploding()))
        with self.assertRaisesRegex(ValueError, "query order"):
            variant.matches(revealed_moves=("Surf",), revealed_ability=Exploding())

    def test_current_fields_not_identity_and_mutable_move_fallback(self):
        variant = catalog_variant(("Surf",))
        self.assertTrue(variant.matches(revealed_moves=("Surf",)))
        object.__setattr__(variant, "moves", ("Ice Beam",))
        self.assertFalse(variant.matches(revealed_moves=("Surf",)))
        self.assertTrue(variant.matches(revealed_moves=("Ice Beam",)))
        self.assertEqual(_normalized_variant_move_set.cache_info().currsize, 2)
        mutable = ["Surf"]
        object.__setattr__(variant, "moves", mutable)
        self.assertTrue(variant.matches(revealed_moves=("Surf",)))
        mutable[:] = ["Recover"]
        self.assertFalse(variant.matches(revealed_moves=("Surf",)))
        self.assertTrue(variant.matches(revealed_moves=("Recover",)))
        self.assertEqual(_normalized_variant_move_set.cache_info().currsize, 2)

    def test_variant_and_tuple_subclasses_retain_original_iteration(self):
        class Extended(Gen3RandbatVariant):
            pass
        self.assertTrue(Extended(**vars(catalog_variant())).matches(revealed_moves=("Surf",)))
        class Moves(tuple):
            calls = 0
            def __iter__(self):
                self.calls += 1
                return super().__iter__()
            def __hash__(self):
                raise AssertionError("custom tuple must not be hashed")
        moves = Moves(("Surf",))
        variant = catalog_variant(moves)
        for _ in range(3):
            self.assertTrue(variant.matches(revealed_moves=("Surf",)))
        self.assertEqual(moves.calls, 3)
        self.assertEqual(_normalized_variant_move_set.cache_info().currsize, 0)

    def test_long_or_large_noncanonical_inputs_keep_original_fallback(self):
        for moves in (("x" * 129,), tuple(f"Move{i}" for i in range(5))):
            variant = catalog_variant(moves)
            for revealed in ((), moves[:1]):
                self.assertEqual(variant.matches(revealed_moves=revealed),
                    legacy_variant_matches(variant, revealed_moves=revealed))
        self.assertEqual(_normalized_variant_move_set.cache_info().currsize, 0)
        variant = catalog_variant((None, 7, ["Surf"]))
        self.assertEqual(variant.matches(), legacy_variant_matches(variant))
        self.assertEqual(_normalized_variant_move_set.cache_info().currsize, 0)

    def test_cache_bound_and_eviction_keep_output_parity(self):
        self.assertEqual(_normalized_variant_move_set.cache_parameters()["maxsize"], 2048)
        for index in range(2060):
            variant = catalog_variant((f"Move{index}",))
            self.assertEqual(variant.matches(), legacy_variant_matches(variant))
        self.assertEqual(_normalized_variant_move_set.cache_info().currsize, 2048)
        before = _normalized_variant_move_set.cache_info().misses
        self.assertTrue(catalog_variant(("Move0",)).matches())
        self.assertEqual(_normalized_variant_move_set.cache_info().misses, before + 1)

    def test_concurrent_matches_equal_original_without_mutable_cached_outputs(self):
        variants = [catalog_variant((f"Move{i % 31}", "Hidden Power Fire")) for i in range(512)]
        query = dict(revealed_moves=("Hidden Power",), ruled_out_items=("Unknown",))
        with ThreadPoolExecutor(max_workers=4) as executor:
            actual = list(executor.map(lambda v: v.matches(**query), variants))
        self.assertEqual(actual, [legacy_variant_matches(v, **query) for v in variants])
        self.assertLessEqual(_normalized_variant_move_set.cache_info().currsize, 31)


@requires_showdown()
class CatalogVariantMoveCacheParityTests(unittest.TestCase):
    def test_every_pinned_variant_all_move_subsets_and_filter_channels_match_original(self):
        source = Gen3RandbatSource.from_showdown_root(showdown_root(), use_cache=False)
        # CI and qualification use this exact Showdown/Dex catalog, not a
        # conveniently sized substitute with weaker differential coverage.
        self.assertEqual(source.metadata.source_hash, "f5a5265143d423af")
        self.assertEqual(len(source.universes), 220)
        self.assertEqual(sum(len(u.variants) for u in source.universes.values()), 1682)
        self.addCleanup(_normalized_variant_move_set.cache_clear)
        checked = 0
        for universe in source.universes.values():
            for variant in universe.variants:
                move_subsets = [subset for count in range(len(variant.moves) + 1)
                    for subset in combinations(variant.moves, count)]
                queries = [dict(revealed_moves=moves) for moves in move_subsets]
                queries.extend((dict(revealed_moves=("Hidden Power",)),
                    dict(revealed_ability=variant.ability), dict(revealed_ability="Unknown"),
                    dict(revealed_item=variant.item), dict(revealed_item="Unknown"),
                    dict(ruled_out_abilities=(variant.ability,)),
                    dict(ruled_out_items=(variant.item,))))
                for query in queries:
                    self.assertEqual(variant.matches(**query), legacy_variant_matches(variant, **query))
                    checked += 1
            for revealed in ((), ("Hidden Power",), ("Impossible Move",)):
                expected = tuple(v for v in universe.variants
                    if legacy_variant_matches(v, revealed_moves=revealed))
                actual = universe.filter_variants(revealed_moves=revealed)
                self.assertEqual(actual, expected)
                self.assertTrue(all(a is b for a, b in zip(actual, expected)))
        self.assertGreater(checked, 20000)


if __name__ == "__main__":
    unittest.main()
