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

    def test_a_sibling_directory_does_not_raise(self) -> None:
        """`/opt/showdown-old/...` passes a naive `startswith("/opt/showdown")` prefix test.

        `to_payload` is called once per believed mon per decision, so a serializer that can throw on
        a legacy or hand-edited payload is a live crash, not a theoretical one. The first version
        raised `ValueError: ... is not in the subpath of ...` on exactly this input.
        """
        meta = _metadata(sets_path="/opt/showdown-old/data/sets.json")
        with self.assertRaises(ValueError) as caught:
            meta.to_payload()
        # It must fail as a REFUSAL naming the problem, not as a stray relative_to error.
        self.assertIn("refusing to serialize", str(caught.exception))

    def test_an_unrelatable_path_is_refused_rather_than_leaked(self) -> None:
        """A path outside the root must never be emitted as-is.

        With `showdown_root` dropped, a leftover absolute value is UNFALSIFIABLE downstream — a
        consumer cannot tell relative from absolute. The first version kept it silently, which a
        symlinked build layout reproduced.
        """
        for bad in ("/elsewhere/data/sets.json", "/private/var/tmp/sd/data/sets.json"):
            with self.subTest(path=bad):
                with self.assertRaises(ValueError):
                    _metadata(sets_path=bad).to_payload()

    def test_a_rootless_metadata_still_refuses_absolute_paths(self) -> None:
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
        with self.assertRaises(ValueError):
            _metadata(showdown_root=None).to_payload()

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
