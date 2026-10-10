"""Exact source-metadata parity only; no game, callback or search measurement."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from pokezero.randbat import Gen3RandbatSource, RandbatSourceMetadata, _portable_metadata_items


def metadata(**changes):
    fields = dict(format_id="gen3randombattle", generation=3,
                  showdown_root="/opt/showdown",
                  sets_path="/opt/showdown/data/sets.json",
                  generator_path="/opt/showdown/dist/teams.js", source_hash="source-a")
    fields.update(changes)
    return RandbatSourceMetadata(**fields)


def legacy_payload(source):
    # Independent copy of the pre-optimization serializer, not the new fallback.
    payload = asdict(source)
    root = source.showdown_root
    for key in ("sets_path", "generator_path"):
        value = payload.get(key)
        if not value:
            continue
        if not isinstance(value, (str, os.PathLike)):
            payload[key] = None
            continue
        candidate = Path(value)
        relative = None
        if root:
            try:
                if candidate.is_relative_to(root):
                    relative = str(candidate.relative_to(root))
            except (ValueError, TypeError):
                relative = None
        if relative is not None:
            payload[key] = relative
            continue
        if (not candidate.is_absolute() and not candidate.drive and not candidate.root
                and ".." not in candidate.parts and not value.startswith("~")):
            continue
        payload[key] = None
    payload["showdown_root"] = None
    return payload


class MetadataCacheParityTests(unittest.TestCase):
    def setUp(self):
        _portable_metadata_items.cache_clear()
        self.addCleanup(_portable_metadata_items.cache_clear)

    def test_path_matrix_matches_original_and_keeps_key_order(self):
        roots = (None, "", "/opt/showdown", "/opt/showdown/", "~", "~/showdown",
                 "relative", ".", "/", "C:\\showdown")
        paths = (None, "", "/opt/showdown/data/sets.json", "/opt/showdown-old/sets.json",
                 "/elsewhere/sets.json", "~/elsewhere/sets.json", "../sets.json",
                 "data/../sets.json", "data/sets.json", "./data/sets.json", ".",
                 "C:\\Users\\name\\sets.json", "//host/share/sets.json", "a\u0130\ud800.json")
        for root in roots:
            for left in paths:
                for right in paths:
                    source = metadata(showdown_root=root, sets_path=left, generator_path=right)
                    with self.subTest(root=root, left=left, right=right):
                        expected = legacy_payload(source)
                        self.assertEqual(source.to_payload(), expected)
                        self.assertEqual(list(source.to_payload()), list(expected))

    def test_cache_hit_returns_fresh_mapping_without_path_work(self):
        source = metadata()
        first = source.to_payload()
        first.update(source_hash="corrupted", sets_path="/private/path", extra=[1])
        with patch("pokezero.randbat.Path", side_effect=AssertionError("repeated path work")):
            second = source.to_payload()
        self.assertEqual(second, legacy_payload(source))
        self.assertIsNot(first, second)
        self.assertEqual(_portable_metadata_items.cache_info().hits, 1)

    def test_equal_instances_share_cache_not_payloads(self):
        first, second = metadata().to_payload(), metadata().to_payload()
        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        self.assertEqual((_portable_metadata_items.cache_info().misses,
                          _portable_metadata_items.cache_info().hits), (1, 1))

    def test_each_identity_field_participates_in_key(self):
        base = metadata()
        base.to_payload()
        for field, value in (("format_id", "other"), ("generation", 4),
                ("showdown_root", "/different"), ("sets_path", "different.json"),
                ("generator_path", None), ("source_hash", "source-b")):
            changed = replace(base, **{field: value})
            self.assertEqual(changed.to_payload(), legacy_payload(changed))
        self.assertEqual(_portable_metadata_items.cache_info().misses, 7)

    def test_forced_dataclass_mutation_cannot_reuse_stale_identity(self):
        source = metadata()
        source.to_payload()
        object.__setattr__(source, "source_hash", "new-identity")
        self.assertEqual(source.to_payload(), legacy_payload(source))
        self.assertEqual(source.to_payload()["source_hash"], "new-identity")

    def test_mutable_wrong_typed_fields_keep_deepcopy_semantics(self):
        source = metadata(format_id={"nested": [1]}, generation=[3], source_hash=["x"])
        first = source.to_payload()
        first["format_id"]["nested"].append(2)
        first["generation"].append(4)
        first["source_hash"].append("y")
        self.assertEqual(source.to_payload(), legacy_payload(source))
        self.assertEqual(_portable_metadata_items.cache_info().currsize, 0)

    def test_str_subclass_preserves_dynamic_conversion_and_bypasses_cache(self):
        class Dynamic(str):
            conversions = 0
            def __fspath__(self):
                self.conversions += 1
                return "/outside/path"
            def __hash__(self):
                raise AssertionError("custom text must not be hashed")
        source = metadata(sets_path=Dynamic("/outside/path"))
        for _ in range(3):
            self.assertEqual(source.to_payload(), legacy_payload(source))
        self.assertEqual(_portable_metadata_items.cache_info().currsize, 0)

    def test_custom_pathlike_is_converted_on_every_call_and_errors_survive(self):
        class Dynamic:
            calls = 0
            def __deepcopy__(self, memo):
                return self
            def __fspath__(self):
                self.calls += 1
                if self.calls > 2:
                    raise ValueError("path conversion failed")
                return "/outside/path"
            def __hash__(self):
                raise AssertionError("custom paths must not be hashed")
        value = Dynamic()
        source = metadata(sets_path=value)
        self.assertIsNone(source.to_payload()["sets_path"])
        self.assertIsNone(source.to_payload()["sets_path"])
        with self.assertRaisesRegex(ValueError, "path conversion failed"):
            source.to_payload()
        self.assertEqual(value.calls, 3)
        self.assertEqual(_portable_metadata_items.cache_info().currsize, 0)

    def test_dataclass_subclass_retains_extra_fields_and_deepcopy(self):
        @dataclass(frozen=True)
        class Extended(RandbatSourceMetadata):
            extra: object = None
        source = Extended(**asdict(metadata()), extra={"items": [1]})
        first = source.to_payload()
        first["extra"]["items"].append(2)
        self.assertEqual(source.to_payload(), legacy_payload(source))
        self.assertEqual(_portable_metadata_items.cache_info().currsize, 0)

    def test_large_keys_and_int_subclasses_bypass_cache(self):
        class Generation(int):
            def __hash__(self):
                raise AssertionError("custom numbers must not be hashed")
        for source in (metadata(source_hash="x" * 4097), metadata(generation=2 ** 65),
                       metadata(generation=Generation(3)), metadata(generation=True)):
            self.assertEqual(source.to_payload(), legacy_payload(source))
        self.assertEqual(_portable_metadata_items.cache_info().currsize, 0)
        metadata(source_hash="x" * 4096, generation=2 ** 63).to_payload()
        self.assertEqual(_portable_metadata_items.cache_info().currsize, 1)

    def test_subclass_helper_name_collision_does_not_change_serialization(self):
        @dataclass(frozen=True)
        class Extended(RandbatSourceMetadata):
            extra: object = None
            def _portable_payload_uncached(self):
                return {"unrelated_subclass_helper": True}
        source = Extended(**asdict(metadata()), extra={"items": [1]})
        first = source.to_payload()
        self.assertEqual(first, legacy_payload(source))
        first["extra"]["items"].append(2)
        self.assertEqual(source.to_payload(), legacy_payload(source))
        self.assertEqual(_portable_metadata_items.cache_info().currsize, 0)

    def test_cache_is_bounded_and_eviction_preserves_output(self):
        self.assertEqual(_portable_metadata_items.cache_parameters()["maxsize"], 128)
        for index in range(140):
            source = metadata(source_hash=str(index))
            self.assertEqual(source.to_payload(), legacy_payload(source))
        self.assertEqual(_portable_metadata_items.cache_info().currsize, 128)
        source = metadata(source_hash="0")
        self.assertEqual(source.to_payload(), legacy_payload(source))
        self.assertEqual(_portable_metadata_items.cache_info().misses, 141)

    def test_concurrent_calls_match_original_and_do_not_share_mappings(self):
        sources = [metadata(source_hash=str(index % 17)) for index in range(300)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda source: source.to_payload(), sources))
        self.assertEqual(results, [legacy_payload(source) for source in sources])
        self.assertEqual(len({id(result) for result in results}), len(results))
        self.assertLessEqual(_portable_metadata_items.cache_info().currsize, 128)

    def test_serialized_cache_roundtrip_remains_idempotent(self):
        source = metadata()
        first = source.to_payload()
        rebuilt = Gen3RandbatSource.from_payload({"metadata": first, "universes": {}})
        self.assertEqual(rebuilt.metadata.to_payload(), first)
        self.assertEqual(rebuilt.to_payload()["metadata"], first)

    def test_source_serialization_cannot_poison_belief_metadata(self):
        source = Gen3RandbatSource.from_payload({"metadata": metadata().to_payload(),
            "universes": {"starmie": {"species": "Starmie", "level": 100,
                "variants": [{"variant_id": "a", "source_set_id": "a", "moves": ["Surf"],
                              "ability": "Natural Cure", "item": "Leftovers"}]}}})
        args = dict(format_id="gen3randombattle", species="Starmie", revealed_moves=())
        first = source.summarize(**args)
        first.source_metadata["source_hash"] = "poisoned"
        first.candidate_variants[0]["moves"].append("Recover")
        second = source.summarize(**args)
        self.assertEqual(second.source_metadata, legacy_payload(source.metadata))
        self.assertEqual(second.candidate_variants[0]["moves"], ["Surf"])

    def test_portable_json_bytes_match_original(self):
        for source in (metadata(), metadata(showdown_root=None, sets_path="data/sets.json"),
                       metadata(sets_path="~/outside/sets.json")):
            self.assertEqual(json.dumps(source.to_payload(), separators=(",", ":")),
                             json.dumps(legacy_payload(source), separators=(",", ":")))

    def test_wrong_typed_paths_and_deepcopy_failures_keep_original_behavior(self):
        for value in (12345, [], {}, False, ["a"]):
            source = metadata(sets_path=value)
            self.assertEqual(source.to_payload(), legacy_payload(source))
        class Broken:
            def __deepcopy__(self, memo):
                raise RuntimeError("deepcopy failed")
        source = metadata(source_hash=Broken())
        for serializer in (lambda: source.to_payload(), lambda: legacy_payload(source)):
            with self.assertRaisesRegex(RuntimeError, "deepcopy failed"):
                serializer()


if __name__ == "__main__":
    unittest.main()
