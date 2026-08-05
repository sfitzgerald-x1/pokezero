"""Bit-exact Python/Rust parity for the complete V4 (history-free k0 pack) surface.

The V3 twin (``test_rust_encoder_v3.py``) proves the turn-merged surface. This proves the
opposite shape: no transition region at all, plus the feature-pack columns the native encoder
reads off the observation metadata.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import unittest

try:
    import numpy
    import pokezero_search
except (ImportError, OSError):  # pragma: no cover - optional native gate
    numpy = None
    pokezero_search = None

from pokezero.golden_corpus import load_golden_corpus
from pokezero.observation import OBSERVATION_SCHEMA_VERSION_V3, OBSERVATION_SCHEMA_VERSION_V4
from pokezero.showdown import observation_from_player_state
from _showdown_root import showdown_root_str

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
SAMPLE_DIR = REPO_ROOT / "tests" / "data" / "golden_corpus_sample"
DEFAULT_SHOWDOWN_ROOT = Path(showdown_root_str())


def _showdown_root() -> Path:
    return Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT)


def _available() -> bool:
    root = _showdown_root()
    return (
        numpy is not None
        and pokezero_search is not None
        and hasattr(pokezero_search, "NativeEncoder")
        and (root / "data" / "random-battles" / "gen3" / "sets.json").exists()
    )


@unittest.skipUnless(_available(), "requires numpy, the native crate, and a Showdown checkout")
class RustEncoderV4ParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(SCRIPTS))
        import export_encoder_tables
        import golden_encoder_backends

        cls.exporter = export_encoder_tables
        cls.backends = golden_encoder_backends
        cls.corpus = load_golden_corpus(SAMPLE_DIR)

    @classmethod
    def tearDownClass(cls) -> None:
        sys.path.remove(str(SCRIPTS))

    def _fixture(self):
        header = copy.deepcopy(self.corpus.header)
        header["observation"].update(
            {
                "schema_version": OBSERVATION_SCHEMA_VERSION_V4,
                "token_count": 23,
                "categorical_feature_count": 41,
                "numeric_feature_count": 132,
            }
        )
        # DEEPCOPY, and not optional: row_inputs_from_decision_row returns
        # row.observation_metadata VERBATIM (golden_encoder_backends.py:82-86), so the
        # `metadata[...]` writes below — especially opponent_team.append — would write
        # THROUGH to the shared class-level corpus row. Without this, consecutive tests
        # inherit each other's mutations: team sizes 3, 5, 7, 9, 11 with duplicated
        # species, which is species-clause-illegal, collapses the per-mon matchup cells
        # (evidence is keyed by normalized species, so duplicates share one entry), and
        # saturates the divisor via the index rather than the formula.
        inputs = copy.deepcopy(
            self.backends.row_inputs_from_decision_row(self.corpus.decision_rows[0])
        )
        inputs["observation_schema_version"] = OBSERVATION_SCHEMA_VERSION_V4
        metadata = inputs["observation_metadata"]
        # Every pack surface, on both seats, so a column the Rust port forgot cannot pass.
        metadata.update(
            {
                "self_must_recharge": True,
                "opponent_must_recharge": True,
                "self_truant_loaf": True,
                "opponent_truant_loaf": True,
                "self_choice_locked": True,
                "opponent_choice_locked": True,
                "self_item_swapped": True,
                "opponent_item_swapped": True,
                "self_last_damage_dealt": 0.4,
                "self_last_damage_taken": 0.2,
                "opponent_last_damage_dealt": 0.25,
                "opponent_last_damage_taken": 0.5,
                "self_last_used_move": "bodyslam",
                "opponent_last_used_move": "switch",
                "opponent_arrived_by_baton_pass": True,
                "self_traced_ability": "levitate",
                "opponent_traced_ability": "intimidate",
                # RAW Part-B ledgers. The settled column values are derived from these
                # below by the very function the encoder uses, so the fixture cannot assert a
                # number the production path would not produce.
                "self_hazard_damage_suffered": 0.36,
                "opponent_hazard_damage_suffered": 0.75,
                "self_items_removed": 1,
                "opponent_items_removed": 2,
                # POPULATED, not empty. An empty dict made the matchup columns unreachable and
                # the parity assertion vacuous over the very feature this file is named for —
                # the shipped encoder read zero there while Python wrote real values, and the
                # test could not see it. Keys are normalized species; filled in per-row below.
                "opponent_matchup_switch_evidence": {},
            }
        )
        metadata["self_side_condition_counts"] = {"spikes": 2}
        metadata["opponent_side_condition_counts"] = {"spikes": 3}
        # The corpus row has only ONE revealed opponent mon, so reveal two more. Without a
        # bench the "pair reaches every opponent token" property is untestable, and a Rust
        # write nested under the active-mon guard would pass unnoticed.
        for species, level in (("Skarmory", 79), ("Blissey", 79)):
            metadata["opponent_team"].append(
                {
                    "ability": None,
                    "active": False,
                    "condition": "100/100",
                    "details": f"{species}, L{level}, M",
                    "fainted": False,
                    "hp_fraction": 1.0,
                    "ident": f"p2: {species}",
                    "item": None,
                    "moves": [],
                    "showdown_slot": "p2",
                    "species": species,
                    "stats": None,
                    "status": "none",
                }
            )
        # Give EVERY opponent mon a distinct (switched, stayed) cell, so a Rust write that
        # lands on the wrong token, the wrong column, or only the active mon cannot pass.
        # The two members must also stay DISTINCT PER MON: with a symmetric pair the switched
        # and stayed columns are element-wise equal and a transposed write in the Rust array
        # literal is invisible. (index + 1, index % 3 + 5) is asymmetric for every index in
        # 0..5 and peaks at 7/8, so no cell saturates the divisor and collapses back to equal.
        from pokezero.showdown import _normalize_identifier

        metadata["opponent_matchup_switch_evidence"] = {
            _normalize_identifier(mon["species"]): [index + 1, (index % 3) + 5]
            for index, mon in enumerate(metadata["opponent_team"])
        }
        return header, inputs, metadata

    def _publish_credit_values(self, inputs, state, dex):
        """Mirror production: the encoder derives Part B once, metadata carries the result."""
        from pokezero.showdown import field_credit_values

        inputs["observation_metadata"].update(field_credit_values(state, dex=dex))

    def test_complete_v4_surface_matches_byte_for_byte(self) -> None:
        header, inputs, metadata = self._fixture()
        spec, masks = self.backends.observation_contract_from_header(header)
        self.assertEqual(spec.transition_token_count, 0)

        reference = self.backends.PythonReferenceBackend(
            showdown_root=_showdown_root(), header=header
        )
        state = self.backends.state_from_row_inputs(inputs)
        self._publish_credit_values(inputs, state, reference._dex)
        state = self.backends.state_from_row_inputs(inputs)
        observation = observation_from_player_state(
            state,
            category_vocab=reference._vocab,
            spec=spec,
            dex=reference._dex,
            feature_masks=masks,
        )
        want = self.backends.arrays_dict_from_observation_arrays(
            self.backends.GoldenObservationArrays.from_observation(observation)
        )
        self.assertEqual(want["numeric_features"].shape, (23, 132))
        self.assertEqual(want["categorical_ids"].shape, (23, 41))

        tables = self.exporter.build_tables(
            str(_showdown_root()),
            observation_schema_version=OBSERVATION_SCHEMA_VERSION_V4,
        )
        # The fixture must actually light up every pack column, or parity is vacuous.
        numeric_columns = tables["layout"]["numeric_columns"]
        for column_name in (
            "NUMERIC_TRUANT_LOAF",
            "NUMERIC_LAST_DAMAGE_DEALT",
            "NUMERIC_LAST_DAMAGE_TAKEN",
            "NUMERIC_CHOICE_LOCKED",
            "NUMERIC_ITEM_SWAPPED",
            "NUMERIC_SELF_HAZARD_CREDIT",
            "NUMERIC_OPP_HAZARD_CREDIT",
            "NUMERIC_SELF_HAZARD_EXPECTED",
            "NUMERIC_SELF_ITEMS_REMOVED_CREDIT",
            "NUMERIC_OPP_ITEMS_REMOVED_CREDIT",
            "NUMERIC_MON_SWITCHED_VS_ACTIVE",
            "NUMERIC_MON_STAYED_VS_ACTIVE",
        ):
            self.assertTrue(
                numpy.any(want["numeric_features"][:, numeric_columns[column_name]]),
                f"V4 parity fixture did not exercise {column_name}",
            )

        tables_json = json.dumps(
            tables, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        rust = self.backends.RustBackend(tables_json=tables_json, header=header)
        got = rust.encode(inputs)
        for name in self.backends.ARRAY_NAMES:
            self.assertEqual(
                numpy.ascontiguousarray(got[name]).tobytes(),
                numpy.ascontiguousarray(want[name]).tobytes(),
                name,
            )

    def test_the_pack_categoricals_match_too(self) -> None:
        header, inputs, metadata = self._fixture()
        tables = self.exporter.build_tables(
            str(_showdown_root()),
            observation_schema_version=OBSERVATION_SCHEMA_VERSION_V4,
        )
        index = tables["vocab"]["index"]
        cat_columns = tables["layout"]["categorical_columns"]
        offsets = tables["layout"]["token_offsets"]
        tables_json = json.dumps(
            tables, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        rust = self.backends.RustBackend(tables_json=tables_json, header=header)
        cats = numpy.ascontiguousarray(rust.encode(inputs)["categorical_ids"])
        self_active = next(
            i for i, mon in enumerate(metadata["self_team"]) if mon["active"]
        )
        opp_active = next(
            i for i, mon in enumerate(metadata["opponent_team"]) if mon["active"]
        )
        self_token = offsets["self_pokemon"] + self_active
        opp_token = offsets["opponent_pokemon"] + opp_active
        # A2: a real move on our side, the BATON PASS sentinel on theirs.
        self.assertEqual(
            cats[self_token, cat_columns["CATEGORY_LAST_USED_MOVE"]], index["move:bodyslam"]
        )
        self.assertEqual(
            cats[opp_token, cat_columns["CATEGORY_LAST_USED_MOVE"]],
            index["lastmove:batonpass"],
        )
        # A4 and A1: the current Trace copy, and mustrecharge in the volatile bag.
        self.assertEqual(
            cats[self_token, cat_columns["CATEGORY_TRACED_ABILITY"]], index["ability:levitate"]
        )
        volatile_offset = cat_columns["CATEGORY_VOLATILE_OFFSET"]
        bag = set(cats[opp_token, volatile_offset : volatile_offset + 6].tolist())
        self.assertIn(index["volatile:mustrecharge"], bag)

    def test_the_matchup_pair_reaches_every_opponent_token(self) -> None:
        """Not just the active one: the pair also says which BENCH mon will face what we have
        out. A Rust write nested under the active-mon guard would pass a single-token check."""
        header, inputs, metadata = self._fixture()
        tables = self.exporter.build_tables(
            str(_showdown_root()),
            observation_schema_version=OBSERVATION_SCHEMA_VERSION_V4,
        )
        tables_json = json.dumps(
            tables, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        rust = self.backends.RustBackend(tables_json=tables_json, header=header)
        nums = numpy.ascontiguousarray(rust.encode(inputs)["numeric_features"])
        column = tables["layout"]["numeric_columns"]["NUMERIC_MON_SWITCHED_VS_ACTIVE"]
        stayed_column = tables["layout"]["numeric_columns"]["NUMERIC_MON_STAYED_VS_ACTIVE"]
        offset = tables["layout"]["token_offsets"]["opponent_pokemon"]
        # Bound by the BLOCK WIDTH as well as the team length. With the fixture leak fixed the
        # team is 3 and this never clamps, but a metadata team longer than the six-slot block
        # would otherwise walk into action-candidate rows and report them as missing matchup
        # writes — a test failure that blames the encoder for the fixture's mistake.
        block_width = (
            tables["layout"]["token_offsets"]["action_candidates"] - offset
        )
        team_size = min(len(metadata["opponent_team"]), block_width)
        lit = [
            index
            for index in range(offset, offset + team_size)
            if nums[index, column] != 0.0
        ]
        # EVERY revealed opponent token, not just "more than one" — a partial write that
        # reached two of three tokens would satisfy a >1 assertion.
        self.assertEqual(
            len(lit), team_size, "matchup pair did not reach every opponent token"
        )
        # And the guard above only means something if the two columns can be told apart:
        # if the fixture ever makes them element-wise equal, a transposed write passes.
        switched_values = [nums[index, column] for index in range(offset, offset + team_size)]
        stayed_values = [
            nums[index, stayed_column] for index in range(offset, offset + team_size)
        ]
        self.assertNotEqual(
            switched_values,
            stayed_values,
            "fixture made switched/stayed symmetric — a transposition would be invisible",
        )

    def test_the_a2_ablation_mask_is_honoured_by_the_native_encoder(self) -> None:
        """The mask exists solely for the k0+pack vs k0+pack+lastmove arm pair. If the crate
        ignores it the two arms encode identically and the comparison is meaningless."""
        from pokezero.observation import ObservationFeatureMasks

        header, inputs, _ = self._fixture()
        tables = self.exporter.build_tables(
            str(_showdown_root()),
            observation_schema_version=OBSERVATION_SCHEMA_VERSION_V4,
            masks=ObservationFeatureMasks(feature_pack_last_move=False),
        )
        self.assertIs(
            tables["layout"]["default_feature_masks"]["feature_pack_last_move"], False
        )
        tables_json = json.dumps(
            tables, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        rust = self.backends.RustBackend(tables_json=tables_json, header=header)
        cats = numpy.ascontiguousarray(rust.encode(inputs)["categorical_ids"])
        column = tables["layout"]["categorical_columns"]["CATEGORY_LAST_USED_MOVE"]
        self.assertEqual(
            int(cats[:, column].max()), 0, "ablated last-move column was still written"
        )

    def test_a_v3_row_is_refused_against_v4_tables(self) -> None:
        header, inputs, _ = self._fixture()
        tables = self.exporter.build_tables(
            str(_showdown_root()),
            observation_schema_version=OBSERVATION_SCHEMA_VERSION_V4,
        )
        tables_json = json.dumps(
            tables, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        wrong = copy.deepcopy(inputs)
        wrong["observation_schema_version"] = OBSERVATION_SCHEMA_VERSION_V3
        rust = self.backends.RustBackend(tables_json=tables_json, header=header)
        with self.assertRaisesRegex(ValueError, "does not match encoder-table layout"):
            rust.encode(wrong)

    def test_a_level_100_opponent_encodes_identically(self) -> None:
        """The whole corpus is levelled below 100, so nothing here ever exercised the case
        Showdown signals by OMITTING the level token -- and the native encoder read that
        omission as "no level information" and zeroed eleven numeric cells, where Python reads
        it as level 100 and writes real values.

        Nine gen3 randbats species are L100 (Beautifly, Ditto, Ledian, Luvdisc, Magcargo,
        Nosepass, Shedinja, Spinda, Unown), so this was reachable in every real battle against
        one. It stayed invisible because the fixture's details strings all carry ", L<n>".
        """
        header, inputs, metadata = self._fixture()
        for mon in metadata["opponent_team"]:
            # Exactly what sim/pokemon.ts::getUpdatedDetails emits at level 100.
            mon["details"] = f"{mon['species']}, M"
        spec, masks = self.backends.observation_contract_from_header(header)
        reference = self.backends.PythonReferenceBackend(
            showdown_root=_showdown_root(), header=header
        )
        state = self.backends.state_from_row_inputs(inputs)
        self._publish_credit_values(inputs, state, reference._dex)
        state = self.backends.state_from_row_inputs(inputs)
        observation = observation_from_player_state(
            state,
            category_vocab=reference._vocab,
            spec=spec,
            dex=reference._dex,
            feature_masks=masks,
        )
        want = self.backends.arrays_dict_from_observation_arrays(
            self.backends.GoldenObservationArrays.from_observation(observation)
        )
        tables = self.exporter.build_tables(
            str(_showdown_root()),
            observation_schema_version=OBSERVATION_SCHEMA_VERSION_V4,
        )
        numeric_columns = tables["layout"]["numeric_columns"]
        # Reachability, on the OPPONENT tokens specifically. Scanning the whole column is
        # useless here: the self team always carries ", L<n>", so `numpy.any` is True whether
        # or not the rewrite above landed -- a fixture drift that stopped reaching the L100
        # case would leave both this guard and the parity assertion silently green.
        level_column = want["numeric_features"][:, numeric_columns["NUMERIC_LEVEL"]]
        offsets = tables["layout"]["token_offsets"]
        opponent = slice(
            offsets["opponent_pokemon"],
            offsets["opponent_pokemon"] + len(metadata["opponent_team"]),
        )
        self.assertTrue(
            numpy.all(level_column[opponent] == 1.0),
            "fixture did not reach the L100 token-omitted case: opponent levels are "
            f"{level_column[opponent]!r}",
        )
        rust = self.backends.RustBackend(
            tables_json=json.dumps(
                tables, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ),
            header=header,
        )
        got = rust.encode(inputs)
        for name in self.backends.ARRAY_NAMES:
            self.assertEqual(
                numpy.ascontiguousarray(got[name]).tobytes(),
                numpy.ascontiguousarray(want[name]).tobytes(),
                name,
            )

    def test_a_cosmetic_unown_forme_encodes_identically(self) -> None:
        """A cosmetic forme resolves to nothing in the native dex, so the mon encoded blank.

        gen3 randbats emit Unown as lettered cosmetic formes -- `Unown-C`, `Unown-Z`,
        `Unown-Exclamation` -- which are NOT separate Pokedex entries. Python retries the lookup
        against the collapsed base id (`showdown._species_info_base_fallback` ->
        `randbat.canonical_gen3_randbat_species_id`); the native `Tables::species_info` was a bare
        normalized lookup, so every one of them missed and left CATEGORY_TYPE_1/2, all six
        NUMERIC_BASE_* and all ten NUMERIC_EXPECTED_* columns at zero while Python wrote real
        values.

        Reachable in every battle against an Unown, and invisible to the rest of this file because
        the fixture's species are all ordinary ones.
        """
        header, inputs, metadata = self._fixture()
        # Distinct cosmetic suffixes, including both word-suffix formes -- a fallback that only
        # handled single letters would pass a letters-only fixture.
        formes = ["Unown-C", "Unown-Z", "Unown-Exclamation", "Unown-Question"]
        for index, mon in enumerate(metadata["opponent_team"]):
            forme = formes[index % len(formes)]
            mon["species"] = forme
            mon["details"] = f"{forme}, L100"
        spec, masks = self.backends.observation_contract_from_header(header)
        reference = self.backends.PythonReferenceBackend(
            showdown_root=_showdown_root(), header=header
        )
        state = self.backends.state_from_row_inputs(inputs)
        self._publish_credit_values(inputs, state, reference._dex)
        state = self.backends.state_from_row_inputs(inputs)
        observation = observation_from_player_state(
            state,
            category_vocab=reference._vocab,
            spec=spec,
            dex=reference._dex,
            feature_masks=masks,
        )
        want = self.backends.arrays_dict_from_observation_arrays(
            self.backends.GoldenObservationArrays.from_observation(observation)
        )
        tables = self.exporter.build_tables(
            str(_showdown_root()),
            observation_schema_version=OBSERVATION_SCHEMA_VERSION_V4,
        )
        numeric_columns = tables["layout"]["numeric_columns"]
        offsets = tables["layout"]["token_offsets"]
        opponent = slice(
            offsets["opponent_pokemon"],
            offsets["opponent_pokemon"] + len(metadata["opponent_team"]),
        )
        # Reachability on the OPPONENT tokens: Python must actually have resolved the base forme
        # and written non-zero base stats. Without this the parity assertion below would pass
        # trivially if BOTH sides zeroed -- which is exactly the pre-fix native behaviour, and the
        # shape a fixture drift would silently restore.
        hp_column = want["numeric_features"][:, numeric_columns["NUMERIC_BASE_HP"]]
        self.assertTrue(
            numpy.all(hp_column[opponent] > 0.0),
            "fixture did not reach a resolvable cosmetic forme: opponent NUMERIC_BASE_HP is "
            f"{hp_column[opponent]!r}",
        )
        rust = self.backends.RustBackend(
            tables_json=json.dumps(
                tables, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ),
            header=header,
        )
        got = rust.encode(inputs)
        for name in self.backends.ARRAY_NAMES:
            self.assertEqual(
                numpy.ascontiguousarray(got[name]).tobytes(),
                numpy.ascontiguousarray(want[name]).tobytes(),
                name,
            )

    def test_a_details_string_that_carries_no_level_at_all_still_matches(self) -> None:
        """The SECOND half of the level-100 fix, which the L100 test above cannot reach.

        `_level_from_details` returns None for a missing or empty details string, and
        `_encode_expected_stats` then coerces that None to 100 rather than zeroing an otherwise
        deterministic block. Porting only the parser half left these two shapes diverging on all
        ten expected-stat columns -- the same sentinel collision, one input shape over.

        A mutation sweep is what surfaced the gap: reverting the caller's coercion to the old
        early-return left every other test in this file green, because they all supply a details
        string the parser resolves.
        """
        tables = self.exporter.build_tables(
            str(_showdown_root()),
            observation_schema_version=OBSERVATION_SCHEMA_VERSION_V4,
        )
        tables_json = json.dumps(
            tables, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        for label, details in (("empty", ""), ("absent", None)):
            with self.subTest(details=label):
                header, inputs, metadata = self._fixture()
                for mon in metadata["opponent_team"]:
                    mon["details"] = details
                spec, masks = self.backends.observation_contract_from_header(header)
                reference = self.backends.PythonReferenceBackend(
                    showdown_root=_showdown_root(), header=header
                )
                state = self.backends.state_from_row_inputs(inputs)
                self._publish_credit_values(inputs, state, reference._dex)
                state = self.backends.state_from_row_inputs(inputs)
                want = self.backends.arrays_dict_from_observation_arrays(
                    self.backends.GoldenObservationArrays.from_observation(
                        observation_from_player_state(
                            state,
                            category_vocab=reference._vocab,
                            spec=spec,
                            dex=reference._dex,
                            feature_masks=masks,
                        )
                    )
                )
                # Reachability: Python must WRITE the coerced block, or "they agree" would just
                # mean "both wrote zeros", which is the defect rather than the fix.
                columns = tables["layout"]["numeric_columns"]
                offsets = tables["layout"]["token_offsets"]
                opponent = slice(
                    offsets["opponent_pokemon"],
                    offsets["opponent_pokemon"] + len(metadata["opponent_team"]),
                )
                block = want["numeric_features"][opponent, columns["NUMERIC_EXPECTED_HP"]]
                self.assertTrue(
                    numpy.all(block > 0),
                    f"Python zeroed the expected-stat block for details={label}, so this "
                    f"assertion could not distinguish the fix from the bug: {block!r}",
                )
                # ...and the level column is what proves the fixture REACHED the level-free
                # case. `block > 0` alone does not: it is true for an ordinary details string
                # too, so a rewrite that silently stopped landing would leave this test green
                # with zero coverage -- the same hole the L100 test above was just fixed for.
                # Python writes 0.0 here precisely because the parser returned None, and 0.79
                # if the rewrite missed.
                level_column = want["numeric_features"][:, columns["NUMERIC_LEVEL"]]
                self.assertTrue(
                    numpy.all(level_column[opponent] == 0.0),
                    f"fixture did not reach the level-free case for details={label}: opponent "
                    f"levels are {level_column[opponent]!r}",
                )
                rust = self.backends.RustBackend(tables_json=tables_json, header=header)
                got = rust.encode(inputs)
                for name in self.backends.ARRAY_NAMES:
                    self.assertEqual(
                        numpy.ascontiguousarray(got[name]).tobytes(),
                        numpy.ascontiguousarray(want[name]).tobytes(),
                        name,
                    )

    def test_a_hidden_power_candidate_matches_on_the_four_non_hp_columns(self) -> None:
        """EXPECTED_DEF/SPA/SPD/SPE parity when the candidate carries a Hidden Power IV override.

        This is the case the rest of this file could not see. The corpus row's single opponent has
        18 candidate variants, and at least one of them has no override on each of the four stats,
        so ``max-over-candidates == flat iv=31`` and the shipped Rust encoder's flat value passed
        parity by FIXTURE LUCK. Pinning the set to one overriding variant separates them: the
        native side emitted 169 where Python emitted 168 (measured, before the port).

        The reachability assertion is the point -- it fails if the fixture ever drifts back to a
        shape where the two formulas agree, rather than letting the parity claim go quietly
        vacuous again. That is the same "Rust spread fork" defect the plan lists: the native
        encoder keeping an approximation after Python was fixed, invisible because the one parity
        test that could have caught it had no overriding candidate.
        """
        header, inputs, metadata = self._fixture()
        tables = self.exporter.build_tables(
            str(_showdown_root()),
            observation_schema_version=OBSERVATION_SCHEMA_VERSION_V4,
        )
        tables_json = json.dumps(
            tables, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        columns = tables["layout"]["numeric_columns"]
        four = [
            columns[name]
            for name in (
                "NUMERIC_EXPECTED_DEF",
                "NUMERIC_EXPECTED_SPA",
                "NUMERIC_EXPECTED_SPD",
                "NUMERIC_EXPECTED_SPE",
            )
        ]
        rust = self.backends.RustBackend(tables_json=tables_json, header=header)

        # Hidden Power FIGHTING overrides all four IVs to 30 (data/typechart.ts), so a pinned
        # variant carrying it is strictly below the flat iv=31 value on every column here.
        pinned = copy.deepcopy(inputs)
        entries = pinned["observation_metadata"]["belief_view"]["opponent_pokemon"]
        self.assertTrue(entries, "no opponent belief entries to pin")
        for entry in entries:
            entry["candidate_variants"] = [
                {
                    "variant_id": "pinned-hp-fighting",
                    "level": 79,
                    "moves": ["hiddenpowerfighting", "surf", "toxic", "protect"],
                    "ability": "levitate",
                    "item": "leftovers",
                }
            ]

        baseline_arr = numpy.asarray(rust.encode(inputs)["numeric_features"])
        pinned_arr = numpy.asarray(rust.encode(pinned)["numeric_features"])
        self.assertTrue(
            numpy.any(baseline_arr[:, four] != pinned_arr[:, four]),
            "the pinned override candidate did not move any of the four columns, so this parity "
            "assertion would pass on a flat iv=31 encoder just as the old fixture did",
        )

        spec, masks = self.backends.observation_contract_from_header(header)
        reference = self.backends.PythonReferenceBackend(
            showdown_root=_showdown_root(), header=header
        )
        state = self.backends.state_from_row_inputs(pinned)
        self._publish_credit_values(pinned, state, reference._dex)
        state = self.backends.state_from_row_inputs(pinned)
        observation = observation_from_player_state(
            state,
            category_vocab=reference._vocab,
            spec=spec,
            dex=reference._dex,
            feature_masks=masks,
        )
        want = self.backends.arrays_dict_from_observation_arrays(
            self.backends.GoldenObservationArrays.from_observation(observation)
        )
        got = rust.encode(pinned)
        for name in self.backends.ARRAY_NAMES:
            self.assertEqual(
                numpy.ascontiguousarray(got[name]).tobytes(),
                numpy.ascontiguousarray(want[name]).tobytes(),
                name,
            )

    def test_a_non_list_moves_payload_abandons_the_band_on_both_sides(self) -> None:
        """The call-site guard for a malformed candidate variant, which the Rust unit tests
        cannot reach -- they drive the spread core directly, below the guard.

        `as_array` flattens a missing/null/scalar `moves` to an EMPTY list, which would then be
        evaluated as a real moveless set; Python returns None and abandons the whole band. The
        two must agree, and a mutation sweep found this the one uncovered branch of the port.
        """
        header, inputs, metadata = self._fixture()
        tables = self.exporter.build_tables(
            str(_showdown_root()),
            observation_schema_version=OBSERVATION_SCHEMA_VERSION_V4,
        )
        rust = self.backends.RustBackend(
            tables_json=json.dumps(
                tables, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ),
            header=header,
        )
        columns = tables["layout"]["numeric_columns"]
        low = columns["NUMERIC_EXPECTED_HP_LOW"]
        high = columns["NUMERIC_EXPECTED_HP_HIGH"]

        def band(encoded):
            arr = numpy.asarray(encoded["numeric_features"])
            return arr[:, low], arr[:, high]

        # REACHABILITY FIRST. If the unmodified fixture has no spread at all, "the band
        # collapsed" is indistinguishable from "there was never a band", and the assertion
        # below would hold no matter what the guard did.
        base_low, base_high = band(rust.encode(inputs))
        self.assertTrue(
            numpy.any(base_low != base_high),
            "fixture carries no real band, so the collapse assertion would be vacuous",
        )

        # The honest "no set source" state the guard must fall back to.
        empty = copy.deepcopy(inputs)
        for entry in empty["observation_metadata"]["belief_view"]["opponent_pokemon"]:
            entry["candidate_variants"] = []
        none_low, none_high = band(rust.encode(empty))

        # ONE bad candidate among good ones. Corrupting them ALL would make the test pass
        # whether or not the guard exists -- every candidate would then yield the same moveless
        # spread, so min == max and the band collapses either way. A mutation sweep caught
        # exactly that: deleting the guard left the all-corrupt version green.
        broken = copy.deepcopy(inputs)
        entries = broken["observation_metadata"]["belief_view"]["opponent_pokemon"]
        self.assertTrue(entries, "no opponent belief entries to corrupt")
        corrupted = 0
        for entry in entries:
            variants = entry.get("candidate_variants") or []
            if len(variants) < 2:
                continue
            # `moves` present but NOT a list -- the shape as_array silently flattens to empty.
            variants[0] = dict(variants[0], moves=None)
            corrupted += 1
        self.assertTrue(corrupted, "no opponent had >=2 variants, so the guard is unreachable")

        got_low, got_high = band(rust.encode(broken))
        self.assertTrue(
            numpy.array_equal(got_low, none_low) and numpy.array_equal(got_high, none_high),
            "one unevaluable candidate did not abandon the whole band -- it was skipped, so the "
            "reported bound is a min/max over a strict subset of the real candidate set",
        )

        # ...and the same row through PYTHON, because the claim is parity, not self-consistency.
        # Comparing two Rust encodes would pass with both sides equally wrong.
        spec, masks = self.backends.observation_contract_from_header(header)
        reference = self.backends.PythonReferenceBackend(
            showdown_root=_showdown_root(), header=header
        )
        state = self.backends.state_from_row_inputs(broken)
        self._publish_credit_values(broken, state, reference._dex)
        state = self.backends.state_from_row_inputs(broken)
        want = self.backends.arrays_dict_from_observation_arrays(
            self.backends.GoldenObservationArrays.from_observation(
                observation_from_player_state(
                    state,
                    category_vocab=reference._vocab,
                    spec=spec,
                    dex=reference._dex,
                    feature_masks=masks,
                )
            )
        )
        got = rust.encode(broken)
        for name in self.backends.ARRAY_NAMES:
            self.assertEqual(
                numpy.ascontiguousarray(got[name]).tobytes(),
                numpy.ascontiguousarray(want[name]).tobytes(),
                name,
            )

if __name__ == "__main__":
    unittest.main()
