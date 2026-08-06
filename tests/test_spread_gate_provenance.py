"""The exact-spread fork rides CHECKPOINT provenance, in both directions.

``_encode_expected_stats(..., exact_spreads=schema_v4)`` corrects Def/SpA/SpD/Spe for the Hidden
Power ``HPivs`` override on 716 of 1682 real candidate variants (42.6%). v4 takes the corrected
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
  ``resolve_checkpoint_contract`` and the encoder-tables latch on the artifact-export side.
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
   production path, and the fixture asserts the variants arrived.
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

#: Real gen3 randbats species whose POOL makes the fork observable in a full encode, with the
#: columns that move. Found by sweeping the pool, not chosen: most species' candidate sets agree
#: on all four stats, so an arbitrary fixture would compare equal values under both schemas and
#: pass no matter which spreads the checkpoint received.
#:
#: Salamence is the primary because TWO columns move, so a write that corrects one and not the
#: other cannot pass. The others are kept as a breadth check: if a randbats change ever takes
#: Hidden Power off Salamence's pool the primary would silently stop discriminating, and the
#: breadth assertion below is what notices.
DISCRIMINATING = {
    # MEASURED, per column, including the DIRECTION. An earlier revision listed four columns and
    # asserted a blanket "corrected is always lower, since HPivs drops an IV to 30". Independent
    # review showed both halves wrong: the gate has a SECOND, independent branch for the Atk/HP
    # band (showdown.py:6864 -- separate from the Def/SpA/SpD/Spe loop above it), so six columns
    # move on Salamence, not two; and the band moves UP on several species, because the legacy
    # HP-trim approximation was wrong by +14..+17 in the low direction. The largest single effect
    # of the gate -- Charizard's HP_LOW, +0.021, about 15x the Def/SpA deltas -- is in the
    # direction the old assertion declared impossible.
    #
    # Listing only the Def/SpA/SpD columns also left the band branch UNCOVERED: setting
    # `if exact_spreads:` to `if False:` at showdown.py:6864 half-reverted the gate with the whole
    # file green.
    "Salamence": {
        "NUMERIC_EXPECTED_ATK_HIGH": "lower",
        "NUMERIC_EXPECTED_ATK_LOW": "lower",
        "NUMERIC_EXPECTED_DEF": "lower",
        "NUMERIC_EXPECTED_HP_HIGH": "lower",
        "NUMERIC_EXPECTED_HP_LOW": "lower",
        "NUMERIC_EXPECTED_SPD": "lower",
    },
    # The primary for DIRECTION, because it moves BOTH ways: a sign flip anywhere in the gate
    # cannot satisfy it, and it is the only fixture here that covers the band's upward correction.
    "Charizard": {
        "NUMERIC_EXPECTED_ATK_LOW": "higher",
        "NUMERIC_EXPECTED_HP_HIGH": "lower",
        "NUMERIC_EXPECTED_HP_LOW": "higher",
        "NUMERIC_EXPECTED_SPA": "lower",
    },
    "Raikou": {
        "NUMERIC_EXPECTED_ATK_HIGH": "higher",
        "NUMERIC_EXPECTED_ATK_LOW": "higher",
        "NUMERIC_EXPECTED_DEF": "lower",
    },
    "Magneton": {
        "NUMERIC_EXPECTED_ATK_HIGH": "higher",
        "NUMERIC_EXPECTED_ATK_LOW": "higher",
        "NUMERIC_EXPECTED_DEF": "lower",
    },
}

#: The species carrying the widest column set and the one carrying both directions. Both are
#: exercised by the directional tests, so neither the Def/SpA/SpD loop nor the Atk/HP band can be
#: reverted alone.
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


def _registry_spec(schema_version: str):
    """A spec from the module registry, for REFERENCE encodes only.

    Named separately from the latched spec so the two can never be confused at a call site: the
    directional tests must encode with the spec the CONSUMER LATCH produced from a checkpoint, and
    an accidental registry lookup there is exactly what made the first revision of this file inert.
    """
    return showdown.observation_spec_for_schema(schema_version)


def _write_checkpoint(path: Path, schema_version: str, vocab: tuple[str, ...]) -> Path:
    """A checkpoint stamped with `schema_version` and that schema's own widths.

    Only `schema_version` + `model_config` are written: `load_transformer_model_config` reads
    exactly those, and `resolve_checkpoint_contract` goes through it. No state_dict is needed,
    which keeps the fixture from depending on the model's parameter shapes.
    """
    from pokezero.neural_policy import NEURAL_POLICY_SCHEMA_VERSION, TransformerPolicyConfig

    spec = showdown.observation_spec_for_schema(schema_version)
    config = TransformerPolicyConfig.compact_category(
        category_vocab=vocab,
        category_oov_buckets=16,
        observation_schema_version=schema_version,
        # Every width from the SPEC, not defaulted: the default token_count is v2.2's 151, and
        # the config cross-checks token_count == fixed prefix + transition region.
        token_count=spec.token_count,
        categorical_feature_count=spec.categorical_feature_count,
        numeric_feature_count=spec.numeric_feature_count,
        transition_token_count=spec.transition_token_count,
        # Must not exceed the physical region: the default budget is 128, v3's region is 64, and
        # v4 has none at all (a nonzero budget there is rejected outright).
        transition_token_budget=spec.transition_token_count,
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

    def _state(self, species: str, level: int = 79):
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

    def _reference(self, species: str, names, *, exact: bool) -> dict[str, float]:
        """The absolute value each column MUST hold, from a direct call with the flag set.

        Computed independently of the call-site gate, which is the point: comparing the v3 encode
        against the v4 encode only detects that the two DIFFER, and a gate keyed off global config
        collapses both to one value, so both comparisons fail with messages naming the wrong
        culprit. Asserting the absolute value instead makes exactly the violated direction fail,
        and say what it means.

        Read at the LEGACY index because this is the writer's own row, before the grouped-layout
        projection; the projection relocates the column without changing its value.
        """
        state = self._state(species)
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
        details = f"{species}, L79"
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

    def _spec_from_checkpoint(self, schema_version: str, tmp: Path):
        """checkpoint file -> CONSUMER latch -> the spec object an encode actually receives.

        Deliberately routed through `env_config_from_checkpoint_provenance`, the latch every
        real consumer uses (`local_showdown`, `collection`, the probes), and the returned spec
        OBJECT is handed to `observation_from_player_state` verbatim.

        The first version of this helper returned a spec and the callers used only
        `spec.schema_version`, re-looking the spec back up out of the module registry via
        `observation_spec_for_schema`. Independent review showed that made the whole chain
        decorative: `resolve_checkpoint_contract` derives `contract.schema_version` by calling the
        same two functions the helper then called itself, so the `assertEqual` fully determined the
        string and nothing downstream could observe the checkpoint. Proved by stubbing this method
        out entirely -- no file, no torch, no resolver -- with both directional tests still green.

        Now the checkpoint's own widths are load-bearing (the spec object carries
        `numeric_feature_count` / `token_count` / `transition_token_count` from the stamped
        config), and a consumer that drops the checkpoint spec in favour of the env default turns
        these tests red rather than leaving them green.
        """
        from pokezero.local_showdown import (
            LocalShowdownConfig,
            env_config_from_checkpoint_provenance,
        )
        from pokezero.mcts_eval.resolver import resolve_checkpoint_contract
        from pokezero.neural_policy import (
            feature_masks_from_model_config,
            load_transformer_model_config,
            observation_spec_from_model_config,
        )

        path = _write_checkpoint(
            tmp / f"{schema_version.replace('.', '_')}.pt",
            schema_version,
            tuple(self.vocabs[schema_version].tokens),
        )
        contract = resolve_checkpoint_contract(path)
        self.assertEqual(
            contract.schema_version,
            schema_version,
            "the contract did not adopt the checkpoint's stamped schema",
        )
        config = load_transformer_model_config(path)
        env_config = env_config_from_checkpoint_provenance(
            LocalShowdownConfig(),
            feature_masks_from_model_config(config),
            context="spread-gate-provenance",
            required_specs=observation_spec_from_model_config(config),
            required_vocabs=self.vocabs[schema_version],
        )
        spec = env_config.observation_spec
        # The latch must have ADOPTED the checkpoint's spec, not kept the env default. Without
        # this the helper could hand back `DEFAULT_REPLAY_OBSERVATION_SPEC` and the directional
        # tests would report the default schema's spreads as the checkpoint's.
        self.assertEqual(
            spec.schema_version,
            schema_version,
            "the env latch did not adopt the checkpoint's spec",
        )
        self.assertEqual(
            (spec.numeric_feature_count, spec.token_count),
            (contract.numeric_feature_count, contract.token_count),
            "the latched spec's widths disagree with the checkpoint contract",
        )
        return contract, spec

    def test_the_fork_is_observable_at_all_on_this_fixture(self) -> None:
        """Reachability FIRST. Without this the two directional tests can both pass on a fixture
        where the corrected and legacy values coincide, which is true for most of the pool."""
        for species in PRIMARY_SPECIES:
            directions = DISCRIMINATING[species]
            names = tuple(directions)
            state = self._state(species)
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
                        f"{species}/{name} is equal under both schemas, so this fixture cannot "
                        "distinguish which spreads a checkpoint received",
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
            _, spec = self._spec_from_checkpoint(OBSERVATION_SCHEMA_VERSION_V3, Path(tmp))
        for species in PRIMARY_SPECIES:
            names = tuple(DISCRIMINATING[species])
            state = self._state(species)
            legacy = self._reference(species, names, exact=False)
            corrected = self._reference(species, names, exact=True)
            got = self._expected_stats(state, spec, names)
            for name in names:
                with self.subTest(species=species, column=name):
                    self.assertNotEqual(
                        legacy[name], corrected[name], f"{name}: references coincide; vacuous"
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
            _, spec = self._spec_from_checkpoint(OBSERVATION_SCHEMA_VERSION_V4, Path(tmp))
        for species in PRIMARY_SPECIES:
            names = tuple(DISCRIMINATING[species])
            state = self._state(species)
            legacy = self._reference(species, names, exact=False)
            corrected = self._reference(species, names, exact=True)
            got = self._expected_stats(state, spec, names)
            for name in names:
                with self.subTest(species=species, column=name):
                    self.assertNotEqual(
                        legacy[name], corrected[name], f"{name}: references coincide; vacuous"
                    )
                    self.assertEqual(
                        got[name],
                        corrected[name],
                        f"{name}: a v4 checkpoint did NOT receive the corrected spread (got "
                        f"{got[name]!r}, v4-corrected {corrected[name]!r}, legacy "
                        f"{legacy[name]!r}). v4 is unlaunched precisely so it can start correct.",
                    )

    def test_more_than_one_species_discriminates(self) -> None:
        """Breadth, so the primary fixture cannot silently stop discriminating.

        If a randbats change takes Hidden Power off Salamence's pool, the three tests above go
        green while measuring nothing. This asserts the fork is visible across several pools.
        """
        observed = []
        for species, directions in sorted(DISCRIMINATING.items()):
            names = tuple(directions)
            with self.subTest(species=species):
                state = self._state(species)
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
                observed.append(species)
        # NOT a count of the literal's length -- that can only fail after a subTest already has,
        # which review flagged as dead. Assert the properties the breadth exists for: both gate
        # BRANCHES and both correction DIRECTIONS stay covered, so neither branch can be reverted
        # alone and a sign error cannot pass.
        covered = {name for d in DISCRIMINATING.values() for name in d}
        self.assertTrue(
            any(name.startswith("NUMERIC_EXPECTED_HP") for name in covered),
            "no HP-band column is covered, so the Atk/HP branch could be reverted alone",
        )
        self.assertTrue(
            covered & {"NUMERIC_EXPECTED_DEF", "NUMERIC_EXPECTED_SPA", "NUMERIC_EXPECTED_SPD"},
            "no Def/SpA/SpD column is covered, so that loop could be reverted alone",
        )
        self.assertEqual(
            {"lower", "higher"},
            {value for d in DISCRIMINATING.values() for value in d.values()},
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
