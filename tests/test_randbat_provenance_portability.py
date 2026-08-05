"""Serialized randbats provenance must never carry a machine-specific path.

`RandbatSourceMetadata` reaches much further than its name suggests: `source_metadata` is copied
onto every `RevealedPokemonBelief`, so a golden-corpus row carries it once per believed mon, and it
also lands in the on-disk source cache, the sidecar payload and the damage-attestation artifact.

Scope, stated precisely because the first version of this file got it wrong. `portable_path`
already collapses `$HOME` to `~` when the metadata is COMPUTED, so a freshly built source is
`~/workspace/...`, not an absolute home path. The absolute form that reached
`tests/data/golden_corpus_sample/rows.jsonl` — the one file the public-invariant guard still has to
allowlist — comes from a cache file written BEFORE that was added. Relativizing at serialization
makes the payload independent of both, which is why these tests assert on the SERIALIZED form and
construct their inputs directly.

Deliberately checkout-free. The earlier version gated every assertion on `@requires_showdown()`, and
CI has no Showdown checkout — so the whole class skipped and the change had zero CI protection. It
also asserted the in-memory object keeps ABSOLUTE paths, which is false twice over: `portable_path`
collapses them at construction, and a cache round-trip replaces them with the relativized form.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from pokezero.randbat import Gen3RandbatSource, RandbatSourceMetadata

ROOT = "/opt/showdown"


def _metadata(**overrides) -> RandbatSourceMetadata:
    fields = {
        "format_id": "gen3randombattle",
        "generation": 3,
        "showdown_root": ROOT,
        "sets_path": f"{ROOT}/data/random-battles/gen3/sets.json",
        "generator_path": f"{ROOT}/dist/data/random-battles/gen3/teams.js",
        "source_hash": "deadbeef",
    }
    fields.update(overrides)
    return RandbatSourceMetadata(**fields)


class RandbatProvenancePortabilityTest(unittest.TestCase):
    def test_every_serialized_path_is_relative(self) -> None:
        payload = _metadata().to_payload()
        self.assertIsNone(payload["showdown_root"])
        for key in ("sets_path", "generator_path"):
            with self.subTest(key=key):
                self.assertFalse(
                    Path(payload[key]).is_absolute(), f"{key}={payload[key]!r} is absolute"
                )
        self.assertEqual(payload["sets_path"], "data/random-battles/gen3/sets.json")

    def test_portable_does_not_mean_empty(self) -> None:
        """The artifact must still be identifiable: relative paths plus a live ``source_hash``."""
        payload = _metadata().to_payload()
        self.assertTrue(str(payload["generator_path"]).endswith("teams.js"))
        self.assertEqual(payload["source_hash"], "deadbeef")
        self.assertEqual(payload["format_id"], "gen3randombattle")

    def test_a_sibling_directory_is_dropped_rather_than_raising(self) -> None:
        """`/opt/showdown-old/...` passes a naive `startswith("/opt/showdown")` prefix test.

        `to_payload` runs once per believed mon per decision, plus from the sidecar and the
        attestation script, so a serializer that can throw is a live crash. An earlier version
        raised here; the value is now dropped instead.
        """
        payload = _metadata(sets_path="/opt/showdown-old/data/sets.json").to_payload()
        self.assertIsNone(payload["sets_path"])
        self.assertEqual(payload["generator_path"], "dist/data/random-battles/gen3/teams.js")

    def test_a_home_collapsed_path_outside_the_root_is_dropped(self) -> None:
        """The shape `portable_path` ACTUALLY emits, and the one an earlier guard missed.

        `portable_path` collapses `$HOME` to `~`, and `Path("~/workspace/...").is_absolute()` is
        **False** — so an `is_absolute()` guard let the real symlinked-build-layout leak straight
        through, durably, surviving cache write and re-read. The predicate is "did relativization
        succeed", not "does it start with a slash".
        """
        payload = _metadata(
            sets_path="~/workspace/pokerena/vendor/pokemon-showdown/data/sets.json"
        ).to_payload()
        self.assertIsNone(payload["sets_path"])

    def test_serialization_never_raises_on_any_path_shape(self) -> None:
        """Loss of function is worse than a cosmetic path. This is a serializer; it must not throw.

        Raising here made a real layout — a checkout under `$HOME` with `data/` symlinked outside
        it — impossible to LOAD AT ALL, because `from_showdown_root` writes the cache through this
        method. That converted a provenance leak into a dead environment.
        """
        for bad in (
            "/elsewhere/data/sets.json",
            "~/somewhere/else/sets.json",
            "../outside/sets.json",
            "",
        ):
            with self.subTest(path=bad):
                payload = _metadata(sets_path=bad).to_payload()
                value = payload["sets_path"]
                self.assertTrue(
                    value is None or not Path(value).is_absolute(),
                    f"leaked {value!r}",
                )

    def test_a_rootless_metadata_drops_absolute_paths(self) -> None:
        """`from_payload` on an already-serialized cache yields `showdown_root=None`.

        Re-serializing that must not become a hole: with no root there is nothing to relativize
        against, so an absolute path can only be refused.
        """
        already_relative = _metadata(
            showdown_root=None,
            sets_path="data/random-battles/gen3/sets.json",
            generator_path="dist/data/random-battles/gen3/teams.js",
        )
        self.assertIsNone(already_relative.to_payload()["showdown_root"])
        # No root => nothing to relativize against => the machine-specific value is dropped.
        self.assertIsNone(_metadata(showdown_root=None).to_payload()["sets_path"])

    def test_the_cache_round_trip_is_stable(self) -> None:
        """Serialize -> load -> serialize must be a fixed point.

        The on-disk source cache is written from `to_payload()` and read back through
        `from_payload`, so a non-idempotent transform would make a source's provenance depend on how
        many times it had been cached.
        """
        first = _metadata().to_payload()
        reloaded = Gen3RandbatSource.from_payload(
            {"metadata": first, "universes": {}, "move_metadata": {}, "species_metadata": {}}
        )
        self.assertEqual(reloaded.metadata.to_payload(), first)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
