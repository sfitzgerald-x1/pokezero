"""The exact-spread fork rides CHECKPOINT provenance, in both directions.

``_encode_expected_stats(..., exact_spreads=schema_v4)`` has TWO independent branches, and both
matter: it corrects Def/SpA/SpD/Spe for the Hidden Power ``HPivs`` override on 716 of 1682 real
candidate variants (42.6%), AND it corrects the Atk/HP band, whose legacy trim under-estimated by
+14..+17. An earlier revision of this file described only the first and left the second uncovered,
so the band could be reverted with every test green. v4 takes the corrected
values; v2.1/v2.2/v3 keep the legacy iv=31 approximation on purpose, because three lineages are
training against those encodes and shifting their input distribution mid-run is worse than a
known one-point error.

That makes the gate a **provenance** question, not a config question: which spreads a checkpoint
receives must be decided by the schema stamped in that checkpoint, so a v3 checkpoint can never be
fed v4 spreads and a v4 checkpoint can never be fed v3 spreads.

Which leaks are actually SILENT, stated precisely because an earlier revision of this docstring got
it wrong. ``schema_v4`` has exactly one input (``showdown.py:4077``:
``spec.schema_version == OBSERVATION_SCHEMA_VERSION_V4``), and v4 differs from every other schema on
all three width axes -- 132/41/23 against v3's 155/51/87 and the v2.2 default's 155/51/151. So
substituting a SPEC (e.g. falling back to ``DEFAULT_REPLAY_OBSERVATION_SPEC``) changes the tensor
shape and the forward rejects it loudly; that is a crash, not a silent leak, and naming it as one
-- as this file first did -- implies a whole class of leaks is covered here when it is not. The
silent failure is a gate keyed off something ORTHOGONAL to the spec: an env var, a launcher flag, a
global default read at the call site. That is the mutant the acceptance runs use, and it is the only
shape this file can honestly claim to catch.

What this file adds over ``test_observation_spec_v4.py``'s coverage, which pins the same gate:

- that file asserts ``exact_spreads=True`` vs ``False`` on a DIRECT call, then pins the call site
  with ``assertIn("exact_spreads=schema_v4", inspect.getsource(...))`` -- a source-text assertion,
  and it says so: *"Reaching this behaviourally needs a set-source-backed belief inside a full v3
  encode, which this harness does not build."* This builds it. The chain exercised end to end is
  checkpoint file -> ``load_transformer_model_config`` -> ``observation_spec_from_model_config``
  -> ``spec.schema_version`` -> ``schema_v4`` -> ``exact_spreads``, plus
  ``resolve_checkpoint_contract`` and the encoder-tables latch on the artifact-export side. The
  first arrow is driven through a REAL consumer (``collection.env_config_with_policy_spec_masks``,
  which takes only a ``neural:<path>`` string), and the derived schema is checked against a raw
  payload read rather than against the derivation itself -- two earlier revisions compared a
  derivation with itself and were inert.
  This file is ADD-ONLY: that assertion is still present and is left in place deliberately, since
  it is nearly free and fails on a call-site edit even in an environment with no Showdown checkout,
  where everything here skips. What changes is that it is no longer the only thing pinning the gate.

Two traps this fixture is built to avoid, both of which produced meaningless green runs while it
was being written:

1. **The block is unreachable without candidate variants.** ``_encode_expected_stats`` early-outs
   to nothing when the belief carries none, and the stored golden-corpus surface does NOT carry
   them -- they are a runtime derivation from the randbats set source. Reconstructing a state from
   a recorded row therefore yields an all-zero expected-stat block and a comparison that cannot
   fail. So the state here is built through ``normalize_for_player(..., set_source=...)``, the
   production MECHANISM -- though note the feature is off by default, since every consumer gates
   ``set_source`` on ``belief_set_source_env_enabled()`` and that reads ``POKEZERO_BELIEF_SET_SOURCE``
   defaulting to ``"0"``. The fixture passes it explicitly and asserts the variants arrived.
2. **v4 REMAPS these columns.** The writer uses legacy indices (``NUMERIC_EXPECTED_HP`` = 66) and
   a projection maps them into the grouped layout, where HP is 53. Reading the module constant out
   of a v4 array reads a different feature entirely and reports "no difference" for every species.
   Columns are resolved per schema through ``numeric_index_if_present_for_schema``.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from _showdown_root import requires_showdown, showdown_root_str

from pokezero import showdown
from pokezero.observation import (
    OBSERVATION_SCHEMA_VERSION_V3,
    OBSERVATION_SCHEMA_VERSION_V4,
)

#: (level, {column: direction}) per species, MEASURED at each species' REAL generator level.
#:
#: The level matters, and an earlier revision got it wrong: it probed every species at L79, but
#: `sets.json` gives Salamence 73, Raikou 74, Magneton 85 -- only Charizard is 79. Which columns
#: move is level-dependent (at L79 Salamence moves DEF/SPD; at its real L73 it moves SPA instead),
#: so the old fixture covered the Def/SpA/SpD loop only at a level the generator never produces for
#: that species. The DIRECTIONS are level-robust -- independent review swept L50..L100 and found no
#: sign flips -- it is the discriminating POWER that varies with level.
#:
#: Two things must stay covered, and the reason for the first is NOT what an earlier revision of
#: this comment said. The `if exact_spreads:` blocks are not independent: the pre-pass under
#: `if exact_spreads and variants:` builds `exact_variant_spreads`, and the Atk/HP band CONSUMES
#: that same list ("Already computed above for the Def/SpA/SpD/Spe block"). So reverting the loop
#: reverts both, and only the BAND is independently revertible -- which is precisely why band
#: columns must be covered. Independent review measured the two mutants as indistinguishable
#: (25 failures each, identical breakdown), disproving the "either can be reverted alone" claim
#: this comment used to make. And both DIRECTIONS: the loop lowers stats (HPivs drops an IV to 30)
#: while the band corrects a legacy trim that UNDER-estimated by +14..+17, so it moves up.
#: Charizard's HP_LOW (+0.021, ~15x the Def/SpA deltas) is the gate's largest single effect and is
#: in the direction an earlier revision asserted was impossible.
DISCRIMINATING = {
    "Salamence": (
        73,
        {
            "NUMERIC_EXPECTED_ATK_HIGH": "lower",
            "NUMERIC_EXPECTED_ATK_LOW": "lower",
            "NUMERIC_EXPECTED_HP_HIGH": "lower",
            "NUMERIC_EXPECTED_HP_LOW": "lower",
            "NUMERIC_EXPECTED_SPA": "lower",
        },
    ),
    "Charizard": (
        79,
        {
            "NUMERIC_EXPECTED_ATK_LOW": "higher",
            "NUMERIC_EXPECTED_HP_HIGH": "lower",
            "NUMERIC_EXPECTED_HP_LOW": "higher",
            "NUMERIC_EXPECTED_SPA": "lower",
        },
    ),
    "Raikou": (
        74,
        {
            "NUMERIC_EXPECTED_ATK_HIGH": "higher",
            "NUMERIC_EXPECTED_ATK_LOW": "higher",
            "NUMERIC_EXPECTED_DEF": "lower",
        },
    ),
    "Magneton": (
        85,
        {
            "NUMERIC_EXPECTED_ATK_HIGH": "higher",
            "NUMERIC_EXPECTED_ATK_LOW": "higher",
            "NUMERIC_EXPECTED_DEF": "lower",
        },
    ),
}

#: Salamence for column breadth, Charizard because it is the only fixture carrying BOTH correction
#: directions, so a sign error anywhere in the gate cannot pass.
PRIMARY_SPECIES = ("Salamence", "Charizard")

#: Widths per schema, so the checkpoint stamps a self-consistent contract. Taken from the specs
#: rather than written down, since a mismatch here would fail the contract resolver for the wrong
#: reason.
SCHEMAS = (OBSERVATION_SCHEMA_VERSION_V3, OBSERVATION_SCHEMA_VERSION_V4)


def _torch():
    try:
        import torch
    except ImportError:  # pragma: no cover - optional dependency
        return None
    return torch


def _stamped_contract(path: Path) -> tuple[str, int, int]:
    """(schema, token_count, transition_token_count) read RAW from the payload dict.

    The widths come from the same independent read as the schema, which is what closes a real
    fragility independent review demonstrated: `DEFAULT_REPLAY_OBSERVATION_SPEC` tracks the CURRENT
    schema, so the day it is bumped to v3 a consumer that ignores the checkpoint would return a v3
    default, match the stamped v3 schema, and `test_a_v3_checkpoint_can_never_receive_v4_spreads`
    would pass against a consumer that never opened the file. Verified: with the default at v2.2
    both directions die on that mutant; with the default at v3 the v3 direction goes green. A
    region-trimmed width cannot be produced by any default, so asserting it keeps the check alive
    across that change.
    """
    payload = _torch().load(path, map_location="cpu", weights_only=True)
    config = payload["model_config"]
    return (
        str(config["observation_schema_version"]),
        int(config["token_count"]),
        int(config["transition_token_count"]),
    )


def _stamped_schema(path: Path) -> str:
    """The schema stamped in a checkpoint, read RAW from the payload dict.

    Deliberately not `observation_spec_from_model_config` / `load_transformer_model_config`: those
    are the derivation under test, and comparing a derivation against itself is the tautology two
    revisions of this file shipped. A raw `torch.load` + dict lookup is an independent source, so a
    consumer that ignores the file and returns a default spec fails the comparison instead of
    satisfying it.
    """
    payload = _torch().load(path, map_location="cpu", weights_only=True)
    return str(payload["model_config"]["observation_schema_version"])


def _registry_spec(schema_version: str):
    """A spec from the module registry, for REFERENCE encodes only.

    Named separately from the latched spec so the two can never be confused at a call site: the
    directional tests must encode with the spec the CONSUMER LATCH produced from a checkpoint, and
    an accidental registry lookup there is exactly what made the first revision of this file inert.
    """
    return showdown.observation_spec_for_schema(schema_version)


def _write_checkpoint(
    path: Path,
    schema_version: str,
    vocab: tuple[str, ...],
    *,
    transition_token_count: int | None = None,
) -> Path:
    """A checkpoint stamped with `schema_version` and that schema's own widths.

    Only `schema_version` + `model_config` are written: `load_transformer_model_config` reads
    exactly those, and `resolve_checkpoint_contract` goes through it. No state_dict is needed,
    which keeps the fixture from depending on the model's parameter shapes.
    """
    from pokezero.neural_policy import NEURAL_POLICY_SCHEMA_VERSION, TransformerPolicyConfig

    spec = showdown.observation_spec_for_schema(schema_version)
    # A REGION-TRIMMED stamp when asked. This is what makes the v3 direction impossible for a
    # schema-keyed lookup to reproduce: no registry spec and no ambient default will ever carry
    # `transition_token_count=32`, so a consumer that ignores the file cannot accidentally match.
    # Not a contrivance -- `resolver.py` documents the real case ("a region-trimmed 39-token server
    # drives a 39-token encode; without this, collectors encoded the default layout and every
    # forward 400'd"). Trimming applies only to schemas that HAVE a transition region: v4 has none,
    # so the v4 direction stays schema-keyed, which is stated where it is relied on.
    if transition_token_count is None:
        transition_token_count = spec.transition_token_count
    token_count = spec.token_count - (spec.transition_token_count - transition_token_count)
    config = TransformerPolicyConfig.compact_category(
        category_vocab=vocab,
        category_oov_buckets=16,
        observation_schema_version=schema_version,
        # Every width from the SPEC, not defaulted: the default token_count is v2.2's 151, and
        # the config cross-checks token_count == fixed prefix + transition region.
        token_count=token_count,
        categorical_feature_count=spec.categorical_feature_count,
        numeric_feature_count=spec.numeric_feature_count,
        transition_token_count=transition_token_count,
        # Must not exceed the physical region: the default budget is 128, v3's region is 64, and
        # v4 has none at all (a nonzero budget there is rejected outright).
        transition_token_budget=transition_token_count,
        policy_id=f"spread-gate-provenance-{schema_version}",
    )
    _torch().save(
        {
            "schema_version": NEURAL_POLICY_SCHEMA_VERSION,
            "model_config": config.to_dict(),
        },
        path,
    )
    return path


@requires_showdown("the fork is measured against the real randbats set source")
class SpreadGateRidesCheckpointProvenanceTest(unittest.TestCase):
    """Both directions, behaviourally, through the checkpoint."""

    @classmethod
    def setUpClass(cls) -> None:
        if _torch() is None:
            raise unittest.SkipTest("PyTorch is not installed in this environment")
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.randbat import Gen3RandbatSource
        from pokezero.randbat_vocab import gen3_category_vocabulary

        cls.root = showdown_root_str()
        cls.dex = load_showdown_dex_cached(cls.root)
        cls.set_source = Gen3RandbatSource.from_showdown_root(cls.root)
        cls.vocabs = {
            schema: gen3_category_vocabulary(
                cls.root,
                include_turn_merged=schema == OBSERVATION_SCHEMA_VERSION_V3,
                include_feature_pack_v4=schema == OBSERVATION_SCHEMA_VERSION_V4,
            )
            for schema in SCHEMAS
        }

    def _state(self, species: str, level: int):
        """A state whose opponent belief carries REAL candidate variants.

        Built through `normalize_for_player` with the set source, which is the production
        MECHANISM -- but note the feature is OFF by default: every consumer gates `set_source` on
        `local_showdown.belief_set_source_env_enabled()`, which reads `POKEZERO_BELIEF_SET_SOURCE`
        and defaults to `"0"`. So on a default run `_encode_expected_stats` reaches no variants and
        this whole fork is inert under either schema. This file therefore pins a path that is
        enabled by an env flag, not one that is always live; an earlier revision called it "how
        production gets them" without that qualification, which overstates it.

        Passed explicitly here rather than via the env flag so the coverage does not depend on the
        ambient environment. A state reconstructed from a recorded corpus row has no variants at
        all, and the expected-stat block is then all zeros -- the vacuity this fixture avoids.
        """
        lines = [
            "|start",
            f"|switch|p1a: Kangaskhan|Kangaskhan, L{level}, F|100/100",
            f"|switch|p2a: {species}|{species}, L{level}|100/100",
            "|turn|1",
        ]
        replay = showdown.parse_showdown_replay(
            lines, battle_id="battle-gen3randombattle-spreadgate"
        )
        state = showdown.normalize_for_player(
            replay,
            player_id="p1",
            configured_showdown_slot="p1",
            format_id="gen3randombattle",
            set_source=self.set_source,
            include_turn_merged=True,
        )
        beliefs = [
            mon
            for mon in state.belief_view.opponent_pokemon
            if mon.candidate_variants
        ]
        self.assertTrue(
            beliefs,
            f"{species}: no opponent belief carries candidate variants, so the expected-stat "
            "block is unreachable and any comparison below is vacuous",
        )
        return state

    def _expected_stats(self, state, spec, names) -> dict[str, float]:
        """Encode under `spec` and read the named columns at THAT schema's indices.

        Takes a spec OBJECT, not a version string: the string form let the caller re-derive the
        spec from the module registry, which is what made the checkpoint round-trip inert.
        """
        import numpy

        schema_version = spec.schema_version
        observation = showdown.observation_from_player_state(
            state,
            category_vocab=self.vocabs[schema_version],
            spec=spec,
            dex=self.dex,
        )
        grid = numpy.asarray(observation.numeric_features)
        opponent_token = showdown.OPPONENT_POKEMON_TOKEN_OFFSET
        out = {}
        for name in names:
            physical = showdown.numeric_index_if_present_for_schema(
                schema_version, getattr(showdown, name)
            )
            self.assertIsNotNone(
                physical, f"{schema_version} does not carry {name}"
            )
            out[name] = float(grid[opponent_token, physical])
        return out

    def _reference(self, species: str, level: int, names, *, exact: bool) -> dict[str, float]:
        """The absolute value each column MUST hold, from a direct call with the flag set.

        Computed independently of the call-site gate, which is the point: comparing the v3 encode
        against the v4 encode only detects that the two DIFFER, and a gate keyed off global config
        collapses both to one value, so both comparisons fail with messages naming the wrong
        culprit. Asserting the absolute value instead makes exactly the violated direction fail,
        and say what it means.

        Read at the LEGACY index because this is the writer's own row, before the grouped-layout
        projection; the projection relocates the column without changing its value.
        """
        state = self._state(species, level)
        # Exactly ONE revealed opponent, asserted rather than assumed: `_expected_stats` reads the
        # first opponent token (`OPPONENT_POKEMON_TOKEN_OFFSET`) while this picks the first belief
        # carrying variants, and those coincide only while the fixture reveals a single mon. With
        # two, the reference and the encoded value could describe different mons and the absolute
        # comparison would be meaningless.
        self.assertEqual(
            [mon.species for mon in state.belief_view.opponent_pokemon],
            [species],
            "fixture reveals more than one opponent; the reference and the read token may differ",
        )
        belief = next(
            mon for mon in state.belief_view.opponent_pokemon if mon.candidate_variants
        )
        details = f"{species}, L{level}"
        row = [0.0] * 512
        showdown._encode_expected_stats(
            row,
            self.dex,
            base_species=species,
            battle_species=species,
            details=details,
            belief=belief,
            exact_spreads=exact,
        )
        return {name: float(row[getattr(showdown, name)]) for name in names}

    def _latched_from_checkpoint(self, path: Path):
        """Hand a real CONSUMER nothing but a PATH; return the spec it derives.

        Takes no schema argument ON PURPOSE. Three revisions of this helper were circular, and the
        third was circular in a way that survived the reviewer's own stub:

        1. Returned a spec whose only used field was `schema_version`, which callers then fed back
           into `observation_spec_for_schema` -- the checkpoint contributed one string the test had
           already asserted.
        2. Passed `required_specs=...` into `env_config_from_checkpoint_provenance` and read the
           spec back out. That latch does `replace(resolved, observation_spec=required_spec)`
           (local_showdown.py:232), i.e. returns the same object, so it was "assert the object I
           supplied"; the widths check beside it was a tautology, since
           `contract.numeric_feature_count` is built by the identical expression the test evaluated
           (resolver.py:236-237).
        3. Took `schema_version` and asserted the derived spec equalled it. Still circular: any
           spec source keyed on the requested schema satisfies it, including a stub that never
           opens the file.

        Now the only input is a path, and the assertion is against `_stamped_schema`, a RAW payload
        read that shares no code with the derivation. A consumer that ignores the checkpoint and
        returns a default spec fails here rather than passing.

        What this still does NOT establish, stated so nobody over-reads it: that the *encoder*
        consults the checkpoint. It cannot -- the encoder legitimately consults only the spec. The
        chain proven is `checkpoint file -> consumer -> spec -> schema_v4 -> exact_spreads`, with
        the first arrow now load-bearing.
        """
        from pokezero.collection import env_config_with_policy_spec_masks
        from pokezero.local_showdown import LocalShowdownConfig

        env_config = env_config_with_policy_spec_masks(
            LocalShowdownConfig(showdown_root=self.root),
            [f"neural:{path}"],
            context="spread-gate-provenance",
        )
        spec = env_config.observation_spec
        schema, token_count, transition_token_count = _stamped_contract(path)
        self.assertEqual(
            (spec.schema_version, spec.token_count, spec.transition_token_count),
            (schema, token_count, transition_token_count),
            "the consumer's spec disagrees with what is stamped in the checkpoint -- it did not "
            "derive it from the file",
        )
        return env_config, spec

    def test_the_fork_is_observable_at_all_on_this_fixture(self) -> None:
        """Reachability FIRST. Without this the two directional tests can both pass on a fixture
        where the corrected and legacy values coincide, which is true for most of the pool."""
        for species in PRIMARY_SPECIES:
            level, directions = DISCRIMINATING[species]
            names = tuple(directions)
            state = self._state(species, level)
            legacy = self._expected_stats(
                state, _registry_spec(OBSERVATION_SCHEMA_VERSION_V3), names
            )
            corrected = self._expected_stats(
                state, _registry_spec(OBSERVATION_SCHEMA_VERSION_V4), names
            )
            for name in names:
                with self.subTest(species=species, column=name):
                    self.assertNotEqual(
                        legacy[name],
                        corrected[name],
                            f"exact_spreads has NO EFFECT on {species}/{name}: the gate collapsed "
                        "to a constant, or its branch for this column was reverted (the "
                        "Def/SpA/SpD/Spe loop or the Atk/HP band), or the fixture stopped "
                        "discriminating. Check the diff before the randbats data.",
                    )
                    # Direction PER COLUMN, from the measured map. A blanket "corrected is lower"
                    # is false: the Def/SpA/SpD/Spe loop lowers stats (HPivs drops an IV to 30)
                    # while the Atk/HP band corrects a legacy trim that under-estimated by
                    # +14..+17, so those move UP. One direction for everything both mis-describes
                    # the gate and lets a sign error through on the half it has backwards.
                    if directions[name] == "lower":
                        self.assertLess(
                            corrected[name],
                            legacy[name],
                            f"{species}/{name}: corrected should be LOWER",
                        )
                    else:
                        self.assertGreater(
                            corrected[name],
                            legacy[name],
                            f"{species}/{name}: corrected should be HIGHER (the legacy band "
                            "under-estimated it)",
                        )

    def test_a_v3_checkpoint_can_never_receive_v4_spreads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Region-TRIMMED on purpose (32 of v3's 64). No registry spec and no ambient
            # default carries this width, so a consumer that never opens the file cannot match it
            # even if the default is bumped to v3 -- which review showed would otherwise turn this
            # very test green against an ignoring consumer.
            path = _write_checkpoint(
                Path(tmp) / "v3.pt",
                OBSERVATION_SCHEMA_VERSION_V3,
                tuple(self.vocabs[OBSERVATION_SCHEMA_VERSION_V3].tokens),
                transition_token_count=32,
            )
            _, spec = self._latched_from_checkpoint(path)
        for species in PRIMARY_SPECIES:
            level, directions = DISCRIMINATING[species]
            names = tuple(directions)
            state = self._state(species, level)
            legacy = self._reference(species, level, names, exact=False)
            corrected = self._reference(species, level, names, exact=True)
            got = self._expected_stats(state, spec, names)
            for name in names:
                with self.subTest(species=species, column=name):
                    self.assertNotEqual(
                        legacy[name],
                        corrected[name],
                        f"exact_spreads has NO EFFECT on {species}/{name}: the gate collapsed "
                        "to a constant, or its branch for this column was reverted, or the "
                        "fixture stopped discriminating. Check the diff before the randbats "
                        "data.",
                    )
                    self.assertEqual(
                        got[name],
                        legacy[name],
                        f"{name}: a v3 checkpoint did NOT receive the legacy spread it trains "
                        f"against (got {got[name]!r}, legacy {legacy[name]!r}, v4-corrected "
                        f"{corrected[name]!r}). Three lineages train against the legacy value; "
                        "changing it shifts their input distribution mid-run, silently and at full "
                        "array width.",
                    )

    def test_a_v4_checkpoint_can_never_receive_v3_spreads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # NOT default-proof, unlike the v3 direction, and stated here because this is where
            # someone debugging a spuriously-green run would look. v4 has no transition region to
            # trim, so the stamped contract is schema-keyed; if `DEFAULT_REPLAY_OBSERVATION_SPEC`
            # ever becomes v4, a consumer that ignores the checkpoint would return a v4 default,
            # match the stamped v4 contract, and this test would pass without the file being read.
            # Measured: with the default forced to v4 only the v3 direction still fails. The v3
            # direction stays default-proof via its trimmed region, so the pair as a whole keeps
            # one live pin in that scenario.
            path = _write_checkpoint(
                Path(tmp) / "v4.pt",
                OBSERVATION_SCHEMA_VERSION_V4,
                tuple(self.vocabs[OBSERVATION_SCHEMA_VERSION_V4].tokens),
            )
            _, spec = self._latched_from_checkpoint(path)
        for species in PRIMARY_SPECIES:
            level, directions = DISCRIMINATING[species]
            names = tuple(directions)
            state = self._state(species, level)
            legacy = self._reference(species, level, names, exact=False)
            corrected = self._reference(species, level, names, exact=True)
            got = self._expected_stats(state, spec, names)
            for name in names:
                with self.subTest(species=species, column=name):
                    self.assertNotEqual(
                        legacy[name],
                        corrected[name],
                        f"exact_spreads has NO EFFECT on {species}/{name}: the gate collapsed "
                        "to a constant, or its branch for this column was reverted, or the "
                        "fixture stopped discriminating. Check the diff before the randbats "
                        "data.",
                    )
                    self.assertEqual(
                        got[name],
                        corrected[name],
                        f"{name}: a v4 checkpoint did NOT receive the corrected spread (got "
                        f"{got[name]!r}, v4-corrected {corrected[name]!r}, legacy "
                        f"{legacy[name]!r}). v4 is unlaunched precisely so it can start correct.",
                    )

    def test_more_than_one_species_discriminates(self) -> None:
        """Breadth across pools AND levels, at each species' own generator level.

        The original justification -- "if a randbats change takes Hidden Power off Salamence's pool
        the tests above go green while measuring nothing" -- turned out to be FALSE, and review
        demonstrated it: the directional tests carry their own reachability guards, so removing the
        fork's effect turns them red rather than green (a half-revert of the Atk/HP band produced 25
        failed subtests, every one from a guard).

        So this test's real value is narrower and worth stating honestly: it exercises two species
        the primaries do not (Raikou, Magneton) and, more importantly, exercises every fixture at
        its REAL generator level, which is where discriminating power actually varies. Salamence at
        L79 moves DEF/SPD; at its real L73 it moves SPA instead. A fixture pinned to the wrong level
        tests a state the generator never produces.
        """
        for species, (level, directions) in sorted(DISCRIMINATING.items()):
            names = tuple(directions)
            with self.subTest(species=species):
                state = self._state(species, level)
                legacy = self._expected_stats(
                    state, _registry_spec(OBSERVATION_SCHEMA_VERSION_V3), names
                )
                corrected = self._expected_stats(
                    state, _registry_spec(OBSERVATION_SCHEMA_VERSION_V4), names
                )
                moved = [name for name in names if legacy[name] != corrected[name]]
                self.assertEqual(
                    sorted(moved),
                    sorted(names),
                    f"{species}: expected {names} to move and only {moved} did",
                )

    def test_the_fixture_literal_still_covers_both_branches_and_directions(self) -> None:
        """A LINT on the fixture literal. Touches no encoder, deliberately separate.

        This used to be tacked onto the end of the breadth test, where it read as coverage; review
        pointed out it cannot fail on any code change, only on a fixture edit. It replaces a dead
        `assertGreaterEqual(len(observed), 3)`, which could only fire after a subTest already had.
        """
        covered = {name for _, directions in DISCRIMINATING.values() for name in directions}
        self.assertTrue(
            any(name.startswith("NUMERIC_EXPECTED_HP") for name in covered)
            or any(name.startswith("NUMERIC_EXPECTED_ATK") for name in covered),
            "no Atk/HP-band column is covered, so that branch could be reverted alone",
        )
        self.assertTrue(
            covered & {"NUMERIC_EXPECTED_DEF", "NUMERIC_EXPECTED_SPA", "NUMERIC_EXPECTED_SPD"},
            "no Def/SpA/SpD column is covered, so that loop could be reverted alone",
        )
        self.assertEqual(
            {"lower", "higher"},
            {
                value
                for _, directions in DISCRIMINATING.values()
                for value in directions.values()
            },
            "the fixture no longer covers both correction directions, so a sign error passes",
        )


@requires_showdown("the latch is compared against real exported tables")
class EncoderTablesLatchRidesTheSameProvenanceTest(unittest.TestCase):
    """The artifact-export half: a checkpoint may not be served tables of another schema.

    The Python root encode takes its spreads from the checkpoint's spec (above); the crate
    encodes leaves from exported tables. If those disagree the model scores states it never
    trained on, so `validate_encoder_tables` compares the tables' layout against the CONTRACT --
    never against the current build.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if _torch() is None:
            raise unittest.SkipTest("PyTorch is not installed in this environment")
        from pokezero.randbat_vocab import gen3_category_vocabulary

        cls.root = showdown_root_str()
        cls.vocabs = {
            schema: gen3_category_vocabulary(
                cls.root,
                include_turn_merged=schema == OBSERVATION_SCHEMA_VERSION_V3,
                include_feature_pack_v4=schema == OBSERVATION_SCHEMA_VERSION_V4,
            )
            for schema in SCHEMAS
        }

    def _tables(self, schema_version: str, tmp: Path) -> Path:
        import sys

        scripts = str(Path(__file__).resolve().parents[1] / "scripts")
        sys.path.insert(0, scripts)
        try:
            import export_encoder_tables
        finally:
            sys.path.remove(scripts)
        payload = export_encoder_tables.build_tables(
            self.root, observation_schema_version=schema_version
        )
        path = tmp / f"tables_{schema_version.replace('.', '_')}.json"
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return path

    def test_crossed_tables_are_refused_in_both_directions(self) -> None:
        from pokezero.mcts_eval.resolver import ContractError, resolve_checkpoint_contract

        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            tables = {schema: self._tables(schema, tmp) for schema in SCHEMAS}
            checkpoints = {
                schema: _write_checkpoint(
                    tmp / f"ckpt_{schema.replace('.', '_')}.pt",
                    schema,
                    tuple(self.vocabs[schema].tokens),
                )
                for schema in SCHEMAS
            }
            # MATCHED first. Without this the crossed assertions below could both pass because
            # the resolver rejects every table it is handed, for a reason unrelated to schema.
            for schema in SCHEMAS:
                with self.subTest(matched=schema):
                    contract = resolve_checkpoint_contract(
                        checkpoints[schema], tables_path=tables[schema]
                    )
                    self.assertEqual(contract.schema_version, schema)
            for checkpoint_schema, tables_schema in (
                (OBSERVATION_SCHEMA_VERSION_V3, OBSERVATION_SCHEMA_VERSION_V4),
                (OBSERVATION_SCHEMA_VERSION_V4, OBSERVATION_SCHEMA_VERSION_V3),
            ):
                with self.subTest(checkpoint=checkpoint_schema, tables=tables_schema):
                    with self.assertRaises(ContractError) as caught:
                        resolve_checkpoint_contract(
                            checkpoints[checkpoint_schema],
                            tables_path=tables[tables_schema],
                        )
                    self.assertIn("schema_version", str(caught.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
