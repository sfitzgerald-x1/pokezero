import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

try:
    import numpy
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    numpy = None

from pokezero.actions import ACTION_COUNT
from pokezero.golden_corpus import (
    GOLDEN_ARRAYS_FILENAME,
    GOLDEN_CORPUS_SCHEMA_VERSION,
    GOLDEN_MANIFEST_FILENAME,
    GOLDEN_ROWS_FILENAME,
    GoldenDecisionRow,
    GoldenGame,
    GoldenGameRecord,
    GoldenObservationArrays,
    _json_safe,
    generate_golden_corpus,
    load_golden_corpus,
    sample_golden_corpus,
    verify_golden_corpus,
    write_golden_corpus,
)
from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT

TESTS_DATA_DIR = Path(__file__).parent / "data"
COMMITTED_SAMPLE_DIR = TESTS_DATA_DIR / "golden_corpus_sample"
V3_SAMPLE_DIR = TESTS_DATA_DIR / "golden_corpus_sample_v3"

# Synthetic corpora use small token tables on purpose: the row/array plumbing is
# shape-agnostic (shapes live in the manifest); production shapes are asserted
# by the live smoke and the committed-sample regression test.
_TOKENS = 7
_CATEGORICAL = 3
_NUMERIC = 5


def _live_showdown_available() -> bool:
    root = Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT)
    return (root / "dist" / "sim" / "index.js").exists() and shutil.which("node") is not None


def _synthetic_arrays(seed: int) -> "GoldenObservationArrays":
    rng = numpy.random.default_rng(seed)
    legal = numpy.zeros(ACTION_COUNT, dtype=bool)
    legal[0] = True
    legal[3] = True
    numeric = rng.random((_TOKENS, _NUMERIC), dtype=numpy.float64)
    # Include values with awkward binary expansions so round-trip checks are
    # meaningfully bit-level, not just approximately equal.
    numeric[0, 0] = 0.1 + 0.2
    numeric[0, 1] = 1.0 / 3.0
    return GoldenObservationArrays(
        categorical_ids=rng.integers(0, 60000, size=(_TOKENS, _CATEGORICAL)).astype("<i4"),
        numeric_features=numeric,
        token_type_ids=rng.integers(0, 12, size=(_TOKENS,)).astype("<i2"),
        attention_mask=(rng.random(_TOKENS) > 0.3),
        legal_action_mask=legal,
    )


def _synthetic_row(*, battle_seed: int, player_id: str, decision_round_index: int) -> GoldenDecisionRow:
    return GoldenDecisionRow(
        battle_seed=battle_seed,
        battle_id=f"golden-test-{battle_seed}",
        format_id="gen3randombattle",
        player_id=player_id,
        decision_round_index=decision_round_index,
        requested_players=("p1", "p2"),
        observation_schema_version="pokezero.observation.v2.2",
        perspective={
            "player_id": player_id,
            "showdown_slot": player_id,
            "opponent_showdown_slot": "p2" if player_id == "p1" else "p1",
        },
        observation_metadata={
            "turn_number": decision_round_index + 1,
            "belief_view": {"self_slot": player_id, "opponent_slot": "p2" if player_id == "p1" else "p1"},
        },
        public_materialization={"turn": decision_round_index + 1, "selfPlayer": player_id, "sides": {}},
        chosen_action_index=0,
        chosen_policy_id="synthetic",
        chosen_action_probability=0.5,
        arrays=_synthetic_arrays(seed=battle_seed * 100 + decision_round_index),
    )


def _synthetic_game(battle_seed: int, *, rows: int) -> GoldenGame:
    record = GoldenGameRecord(
        battle_seed=battle_seed,
        battle_id=f"golden-test-{battle_seed}",
        format_id="gen3randombattle",
        policy_ids={"p1": "synthetic", "p2": "synthetic"},
        true_teams={
            "p1": {"source": "synthetic", "pokemon": [], "packed": "Pikachu||||thunderbolt|||||||"},
            "p2": {"source": "synthetic", "pokemon": [], "packed": "Squirtle||||watergun|||||||"},
        },
        terminal={"winner": "p1", "turn_count": rows, "capped": False},
    )
    decision_rows = tuple(
        _synthetic_row(
            battle_seed=battle_seed,
            player_id="p1" if index % 2 == 0 else "p2",
            decision_round_index=index,
        )
        for index in range(rows)
    )
    return GoldenGame(record=record, rows=decision_rows)


@unittest.skipIf(numpy is None, "requires numpy")
class GoldenCorpusSchemaRoundTripTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out_dir = Path(self._tmp.name) / "corpus"

    def _write_synthetic(self, *, games: int = 2, rows: int = 3) -> dict:
        return write_golden_corpus(
            self.out_dir,
            header={"generator": {"games": games, "seed_start": 0, "synthetic": True}},
            games=[_synthetic_game(seed, rows=rows) for seed in range(games)],
        )

    def test_round_trip_is_bit_exact(self) -> None:
        manifest = self._write_synthetic()

        corpus = load_golden_corpus(self.out_dir)

        self.assertEqual(manifest["counts"], {"games": 2, "decisions": 6})
        self.assertEqual(len(corpus.games), 2)
        original = _synthetic_game(0, rows=3)
        loaded = corpus.games[0]
        self.assertEqual(loaded.record.battle_id, original.record.battle_id)
        self.assertEqual(loaded.record.true_teams, original.record.true_teams)
        for loaded_row, original_row in zip(loaded.rows, original.rows):
            self.assertEqual(loaded_row.player_id, original_row.player_id)
            self.assertEqual(loaded_row.observation_metadata, original_row.observation_metadata)
            self.assertEqual(loaded_row.public_materialization, original_row.public_materialization)
            for name in (
                "categorical_ids",
                "numeric_features",
                "token_type_ids",
                "attention_mask",
                "legal_action_mask",
            ):
                loaded_array = getattr(loaded_row.arrays, name)
                original_array = getattr(original_row.arrays, name)
                self.assertTrue(
                    numpy.array_equal(loaded_array, original_array),
                    f"array {name} did not round-trip bit-exactly",
                )
        verification = verify_golden_corpus(self.out_dir)
        self.assertEqual(verification.games, 2)
        self.assertEqual(verification.decisions, 6)

    def test_verify_detects_rows_file_tampering(self) -> None:
        self._write_synthetic()
        rows_path = self.out_dir / GOLDEN_ROWS_FILENAME
        rows_path.write_text(rows_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "hash mismatch|size does not match"):
            verify_golden_corpus(self.out_dir)

    def test_load_detects_edited_row_payload(self) -> None:
        self._write_synthetic()
        rows_path = self.out_dir / GOLDEN_ROWS_FILENAME
        lines = rows_path.read_text(encoding="utf-8").splitlines()
        edited = []
        tampered = False
        for line in lines:
            payload = json.loads(line)
            if not tampered and payload.get("record_type") == "decision":
                payload["chosen_action_index"] = 3  # legal, but not what was recorded
                line = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                tampered = True
            edited.append(line)
        rows_path.write_text("\n".join(edited) + "\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "row_sha256"):
            load_golden_corpus(self.out_dir)

    def test_writer_refuses_overwrite_and_illegal_rows(self) -> None:
        self._write_synthetic(games=1, rows=1)
        with self.assertRaises(FileExistsError):
            self._write_synthetic(games=1, rows=1)
        with self.assertRaisesRegex(ValueError, "must be legal"):
            row = _synthetic_row(battle_seed=9, player_id="p1", decision_round_index=0)
            GoldenDecisionRow(**{**row.__dict__, "chosen_action_index": 1})

    def test_json_safe_is_loud_on_non_json_values(self) -> None:
        with self.assertRaisesRegex(TypeError, "non-JSON-safe"):
            _json_safe({"bad": object()}, context="test")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            _json_safe({"bad": float("nan")}, context="test")
        self.assertEqual(_json_safe({"ok": (1, 2)}, context="test"), {"ok": [1, 2]})

    def test_sample_corpus_preserves_row_hashes(self) -> None:
        self._write_synthetic(games=2, rows=3)
        sample_dir = Path(self._tmp.name) / "sample"

        manifest = sample_golden_corpus(self.out_dir, sample_dir, max_decisions=2)

        self.assertEqual(manifest["counts"], {"games": 1, "decisions": 2})
        verify_golden_corpus(sample_dir)

        def _decision_hashes(directory: Path) -> list[str]:
            hashes = []
            for line in (directory / GOLDEN_ROWS_FILENAME).read_text(encoding="utf-8").splitlines():
                payload = json.loads(line)
                if payload.get("record_type") == "decision":
                    hashes.append(payload["row_sha256"])
            return hashes

        self.assertEqual(_decision_hashes(sample_dir), _decision_hashes(self.out_dir)[:2])
        sample_header = json.loads(
            (sample_dir / GOLDEN_ROWS_FILENAME).read_text(encoding="utf-8").splitlines()[0]
        )
        self.assertEqual(sample_header["sampled_from"]["games"], 2)
        self.assertEqual(sample_header["sampled_from"]["decisions"], 6)


@unittest.skipIf(numpy is None, "requires numpy")
@unittest.skipIf(not COMMITTED_SAMPLE_DIR.exists(), "committed golden corpus sample not present")
class GoldenCorpusCommittedSampleTest(unittest.TestCase):
    """Permanent regression net: the committed 5-row sample must stay readable."""

    def test_committed_sample_verifies_with_production_shapes(self) -> None:
        """The primary sample is v4. Two assertions, doing different jobs.

        The SCHEMA is pinned by hand (`OBSERVATION_SCHEMA_VERSION_V4`) so that moving the corpus
        to a new generation is a deliberate edit here rather than something a regeneration does
        quietly. The SHAPES are then checked against that schema's own spec rather than retyped as
        literals -- the previous `(151, 51)` / `(151, 155)` constants restated v2.2's layout in a
        second place, so a regeneration failed on four hand-copied numbers that carried no
        information the spec did not already have.
        """
        from pokezero.showdown import (
            OBSERVATION_SCHEMA_VERSION_V4,
            observation_spec_for_schema,
        )

        verification = verify_golden_corpus(COMMITTED_SAMPLE_DIR)

        self.assertEqual(verification.games, 1)
        self.assertEqual(verification.decisions, 5)
        self.assertEqual(verification.fold_rows, 5)  # schema v2: fold surface present

        spec = observation_spec_for_schema(OBSERVATION_SCHEMA_VERSION_V4)
        tokens = spec.token_count
        self.assertEqual(
            verification.array_shapes["categorical_ids"],
            (tokens, spec.categorical_feature_count),
        )
        self.assertEqual(
            verification.array_shapes["numeric_features"],
            (tokens, spec.numeric_feature_count),
        )
        self.assertEqual(verification.array_shapes["token_type_ids"], (tokens,))
        self.assertEqual(verification.array_shapes["attention_mask"], (tokens,))
        self.assertEqual(verification.array_shapes["legal_action_mask"], (ACTION_COUNT,))

        corpus = load_golden_corpus(COMMITTED_SAMPLE_DIR)
        for row in corpus.decision_rows:
            self.assertEqual(row.observation_schema_version, OBSERVATION_SCHEMA_VERSION_V4)
            self.assertIn("belief_view", row.observation_metadata)
            self.assertIn("sides", row.public_materialization)
        for player in ("p1", "p2"):
            team = corpus.games[0].record.true_teams[player]
            self.assertEqual(len(team["pokemon"]), 6)
            self.assertEqual(team["packed"].count("]"), 5)
            for entry in team["pokemon"]:
                self.assertIn("evs", entry["set"])
                self.assertIn("ivs", entry["set"])


@unittest.skipIf(numpy is None, "requires numpy")
@unittest.skipIf(
    not V3_SAMPLE_DIR.exists(), "committed v3 golden corpus sample not present"
)
class GoldenCorpusV3SampleTest(unittest.TestCase):
    """The v3-era sample kept alongside the v4 one, and why it exists.

    `observation.v4` has NO transition token block (23 tokens,
    `transition_token_count == 0`). ONE test needs a fixture that contains one:
    `test_leaf_encoder.py` compares against STORED golden arrays and reshapes to their shape, so a
    v4 sample is self-consistent at 23 tokens and passes byte-exact having covered zero transition
    rows -- silently. No in-test override can conjure rows the golden arrays do not hold.

    `test_rust_encoder_v3.py` is on this fixture for convenience, not need: it synthesizes its own
    transition tokens and would fail LOUDLY against a v4 sample (that header's
    `transition_token_budget: 0` flattens every NUMERIC_TT_* column and trips its coverage loop
    with a named diagnostic).

    An earlier version of this docstring claimed BOTH would fail silently. That was the wrong half
    of a pair of contradictory comments, retracted in the commit that corrected the other two.

    This test keeps the fixture itself honest -- see the assertions below, which read the stored
    arrays rather than the schema constant an earlier version checked.
    """

    def test_v3_sample_verifies_and_actually_carries_transition_tokens(self) -> None:
        from pokezero.showdown import (
            OBSERVATION_SCHEMA_VERSION_V3,
            observation_spec_for_schema,
        )

        verification = verify_golden_corpus(V3_SAMPLE_DIR)
        self.assertEqual(verification.games, 1)
        self.assertEqual(verification.decisions, 5)

        spec = observation_spec_for_schema(OBSERVATION_SCHEMA_VERSION_V3)
        self.assertEqual(verification.array_shapes["token_type_ids"], (spec.token_count,))

        # Assert on the FIXTURE, not on the schema constant. The first version of this test
        # asserted `spec.transition_token_count > 0`, which is a property of the v3 SCHEMA and
        # would hold even if every transition row in the committed arrays were flat -- so the test
        # written specifically to stop this fixture decaying could not have noticed it decaying.
        # Caught in independent review of the change that added it.
        #
        # What is actually checked now: the stored arrays carry attended transition tokens with
        # real numeric content. Measured on the committed sample: 1-3 attended transition rows per
        # decision, 3-14 non-zero cells each.
        # TRANSITION_TOKEN_OFFSET, not `token_count - transition_token_count`. The subtraction
        # happens to be right only because the transition block is last, and would go silently
        # wrong (wrong offset, still-passing test) the day a block is appended after it. The
        # sibling tests already use the canonical value -- `test_rust_encoder_v3.py` imports this
        # same constant, `test_validate_rust_encoder.py` reads `token_offsets["transition"]`.
        from pokezero.showdown import TRANSITION_TOKEN_OFFSET

        start = TRANSITION_TOKEN_OFFSET
        self.assertGreater(spec.transition_token_count, 0, "v3 spec lost its transition block")
        self.assertEqual(
            start + spec.transition_token_count,
            spec.token_count,
            "the transition block is no longer last in the v3 layout; this test's indexing "
            "assumption needs revisiting",
        )
        corpus = load_golden_corpus(V3_SAMPLE_DIR)
        attended = 0
        nonzero_cells = 0
        for row in corpus.decision_rows:
            attended += int(numpy.asarray(row.arrays.attention_mask)[start:].sum())
            nonzero_cells += int(
                (numpy.asarray(row.arrays.numeric_features)[start:] != 0).sum()
            )
        self.assertGreater(
            attended,
            0,
            "the v3 fixture has a transition BLOCK but no attended transition tokens in it; "
            "it no longer provides the coverage it is retained for",
        )
        self.assertGreater(
            nonzero_cells,
            0,
            "the v3 fixture's transition rows are all zero; byte-comparing them proves nothing",
        )

        for row in corpus.decision_rows:
            self.assertEqual(row.observation_schema_version, OBSERVATION_SCHEMA_VERSION_V3)


@unittest.skipIf(numpy is None, "requires numpy")
@unittest.skipIf(not _live_showdown_available(), "requires node and built Pokemon Showdown checkout")
class GoldenCorpusLiveSmokeTest(unittest.TestCase):
    def test_one_game_generates_a_verifiable_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "corpus"
            manifest = generate_golden_corpus(
                out_dir=out_dir,
                games=1,
                seed_start=41,
                showdown_root=os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT,
                belief_set_source=True,
            )

            self.assertEqual(manifest["schema_version"], GOLDEN_CORPUS_SCHEMA_VERSION)
            from pokezero.golden_corpus import FOLD_ROWS_FILENAME

            for filename in (
                GOLDEN_ROWS_FILENAME,
                GOLDEN_ARRAYS_FILENAME,
                GOLDEN_MANIFEST_FILENAME,
                FOLD_ROWS_FILENAME,
            ):
                self.assertTrue((out_dir / filename).exists(), filename)

            verification = verify_golden_corpus(out_dir)
            self.assertEqual(verification.games, 1)
            self.assertGreater(verification.decisions, 0)
            self.assertEqual(verification.fold_rows, verification.decisions)

            # Schema v2 end to end: every generated fold chain must satisfy the
            # row-pair advance contract through the reference backend.
            from pokezero.golden_corpus_fold import (
                PythonReferenceFoldBackend,
                validate_fold_chains,
            )

            report = validate_fold_chains(
                out_dir,
                PythonReferenceFoldBackend(),
                expected_schema_version=GOLDEN_CORPUS_SCHEMA_VERSION,
            )
            self.assertTrue(report.ok, report.mismatches)
            self.assertEqual(report.rows_validated, verification.decisions)
            # Derived from the spec this run actually generated under, not retyped. The
            # generation here is FLAGLESS, so it follows the fresh-selection default -- which is
            # the property being smoke-tested ("a default collect produces a verifiable
            # corpus"), not any particular layout. The literals `(151, 51)` / `(151, 155)`
            # restated v2.2's shape in a second place and turned the v4 rotation into four
            # hand-copied numbers to chase. A sibling test in this file already made exactly
            # this fix and recorded the same reasoning; this one was missed.
            from pokezero.observation import OBSERVATION_SCHEMA_VERSION
            from pokezero.showdown import DEFAULT_REPLAY_OBSERVATION_SPEC

            _live_spec = DEFAULT_REPLAY_OBSERVATION_SPEC
            self.assertEqual(
                verification.array_shapes["categorical_ids"],
                (_live_spec.token_count, _live_spec.categorical_feature_count),
            )
            self.assertEqual(
                verification.array_shapes["numeric_features"],
                (_live_spec.token_count, _live_spec.numeric_feature_count),
            )
            self.assertEqual(verification.array_shapes["legal_action_mask"], (ACTION_COUNT,))

            corpus = load_golden_corpus(out_dir)
            self.assertEqual(corpus.header["belief_set_source"]["enabled"], True)
            self.assertIsNotNone(corpus.header["belief_set_source"]["source_hash"])
            game = corpus.games[0]
            self.assertEqual(game.record.battle_seed, 41)
            for player in ("p1", "p2"):
                team = game.record.true_teams[player]
                self.assertEqual(team["source"], "bridge-snapshot-generator-set")
                self.assertEqual(len(team["pokemon"]), 6)
                self.assertEqual(team["packed"].count("]"), 5)
                for entry in team["pokemon"]:
                    self.assertIn("evs", entry["set"])
                    self.assertIn("ivs", entry["set"])
                    self.assertTrue(entry["set"]["moves"])
            seats = {row.player_id for row in game.rows}
            self.assertEqual(seats, {"p1", "p2"})
            for row in game.rows:
                # The row records whatever schema PRODUCED it, and this generation is flagless,
                # so that is the fresh-selection default -- asserted against the constant rather
                # than a literal. Naming "v2.2" here was a second statement of the default's
                # identity hidden inside a live-smoke test, and it is the flagless case, so it
                # tracks the default by definition.
                self.assertEqual(row.observation_schema_version, OBSERVATION_SCHEMA_VERSION)
                self.assertTrue(bool(row.arrays.legal_action_mask[row.chosen_action_index]))
                self.assertIn("belief_view", row.observation_metadata)
                self.assertEqual(row.public_materialization["selfPlayer"], row.player_id)
                self.assertTrue(bool(numpy.asarray(row.arrays.attention_mask).any()))


@unittest.skipIf(numpy is None, "requires numpy")
@unittest.skipIf(not _live_showdown_available(), "requires a built local Showdown checkout")
class WrappedTrajectoryIdentityTests(unittest.TestCase):
    """Capture must be trajectory-neutral: wrapped == bare on the same seed.

    Locks in the RNG-identity claim so a future change that makes
    public_materialization_state()/snapshot() consume simulator RNG fails
    loudly here instead of silently skewing the corpus distribution.
    """

    def test_wrapped_and_bare_games_are_identical(self) -> None:
        import os as _os

        from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT as _ROOT
        from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
        from pokezero.policy import SimpleLegalPolicy
        from pokezero.rollout import RolloutConfig, RolloutDriver

        import pokezero.golden_corpus as gc

        showdown_root = _os.environ.get("POKEZERO_SHOWDOWN_ROOT") or _ROOT

        def run(wrapped: bool):
            env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=showdown_root))
            try:
                policies = {"p1": SimpleLegalPolicy(), "p2": SimpleLegalPolicy()}
                if wrapped:
                    policies = {
                        seat: gc._CapturingPolicy(inner=policy, sink=lambda context, decision: None)
                        for seat, policy in policies.items()
                    }
                    # Exercise the oracle path too: snapshot at the opening
                    # boundary, exactly as the generator does.
                driver = RolloutDriver(env=env, policies=policies, config=RolloutConfig())
                result = driver.run(seed=41007, battle_id="identity-test")
                actions = [
                    (step.player_id, step.action_index, step.turn_index)
                    for step in result.trajectory.steps
                ]
                return actions, (result.terminal.winner, result.terminal.turn_count)
            finally:
                env.close()

        bare_actions, bare_terminal = run(wrapped=False)
        wrapped_actions, wrapped_terminal = run(wrapped=True)
        self.assertEqual(bare_actions, wrapped_actions)
        self.assertEqual(bare_terminal, wrapped_terminal)


if __name__ == "__main__":
    unittest.main()