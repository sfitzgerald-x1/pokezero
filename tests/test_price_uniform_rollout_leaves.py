"""Contract pins for the canonical uniform-rollout leaf writer.

The external replay/state corpus is intentionally unavailable in CI.  These
tests therefore exercise the writer's complete key, provenance, state-hash,
fixed-opponent, output, and native-report contract with a fake row pricer.
The crate-boundary behavior of that pricer is exercised in
``test_rollout_leaf_arbiter`` on the model wheel.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCRIPT = REPO / "scripts" / "price_uniform_rollout_leaves.py"
SPEC = importlib.util.spec_from_file_location("price_uniform_rollout_leaves", SCRIPT)
writer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = writer
SPEC.loader.exec_module(writer)


CHECKPOINT = "a" * 64
BANK_SHA = "b" * 64
BELIEF_HASH = "c" * 16
ENGINE = "d" * 64
SHOWDOWN = "e" * 40
SOURCE = "f" * 40
EXTENSION = "1" * 64
WRITER = "2" * 64
VALIDATOR = "3" * 64


def _bank_document() -> dict:
    return {
        "schema": "pokezero.phase0.vhprobe-pairs.v1",
        "provenance": [{"belief_set_source_hash": BELIEF_HASH}],
        "pairs": [
            {"seed": 1, "prefix": 2, "seat": "p1", "arm_a": 0, "arm_b": 3},
            {"seed": 7, "prefix": 11, "seat": "p2", "arm_a": 4, "arm_b": 8},
        ],
    }


def _state_row(seed: int, prefix: int, seat: str, arm: int, opponent_action: int) -> dict:
    state = f"private state {seed}/{prefix}/{seat}/{arm}"
    return {
        "seed": seed,
        "prefix": prefix,
        "seat": seat,
        "arm": arm,
        "subject_action": arm,
        "opponent_action": opponent_action,
        "state": state,
        "state_sha256": hashlib.sha256(state.encode()).hexdigest(),
    }


def _terminal_row(seed: int, prefix: int, seat: str, arm: int, opponent_action: int, winner: str | None) -> dict:
    terminal = {"winner": winner, "turn_count": prefix + 1, "capped": False}
    value = 1.0 if winner == "p1" else 0.0 if winner == "p2" else 0.5
    return {
        "seed": seed,
        "prefix": prefix,
        "seat": seat,
        "arm": arm,
        "subject_action": arm,
        "opponent_action": opponent_action,
        "terminal": terminal,
        "terminal_value": value,
        "terminal_sha256": hashlib.sha256(writer.canonical_json(terminal)).hexdigest(),
    }


def _state_document(bank_sha: str = BANK_SHA) -> dict:
    return {
        "schema": writer.STATE_SCHEMA,
        "provenance": {
            "bank_sha256": bank_sha,
            "checkpoint_sha256": CHECKPOINT,
            "belief_set_source_hash": BELIEF_HASH,
            "engine_build_fingerprint": ENGINE,
            "showdown_commit": SHOWDOWN,
            "state_builder_source_commit": SOURCE,
            "branch_rule": writer.BRANCH_RULE,
        },
        "leaves": [
            _state_row(1, 2, "p1", 0, 5),
            _state_row(1, 2, "p1", 3, 5),
            _state_row(7, 11, "p2", 4, 6),
            _state_row(7, 11, "p2", 8, 6),
        ],
    }


def _report(values: list[float], **config: object) -> dict:
    return {
        "schema": writer.ROW_PRICER_SCHEMA,
        "value_frame": "side_one_absolute",
        "rollout_policy": "uniform",
        "rollouts": config["rollouts"],
        "rollout_max_plies": config["max_plies"],
        "rollout_seed": config["seed"],
        "rollout_threads": config["threads"],
        "rollout_branch_on_damage": config["branch_on_damage"],
        "values": values,
        "leaves_priced": len(values),
        "rollouts_run": len(values) * int(config["rollouts"]),
        "rollout_plies": 12,
        "rollout_terminal_hits": len(values) * int(config["rollouts"]),
        "rollout_cap_hits": 0,
        "rollout_dead_ends": 0,
        "rollout_terminal_fraction": 1.0,
        "rollout_fallback_fraction": 0.0,
        "rollout_mean_plies": 1.0,
    }


def _fake_price(states, ordinals, **config):
    assert len(states) == len(ordinals)
    return _report([0.1 + 0.1 * index for index in range(len(states))], **config), EXTENSION


class UniformRolloutLeafWriterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.bank_path = root / "bank.json"
        self.states_path = root / "states.json"
        self.bank_path.write_text(json.dumps(_bank_document()), encoding="utf-8")
        self.states_path.write_text(json.dumps(_state_document()), encoding="utf-8")

    def _loaded(self):
        expected_keys, belief_hash = writer.load_banked_keys(self.bank_path)
        return writer.load_leaf_states(
            self.states_path,
            expected_keys=expected_keys,
            bank_sha256=BANK_SHA,
            belief_hash=belief_hash,
            checkpoint_sha256=CHECKPOINT,
        )

    def _document(self):
        rows, provenance = self._loaded()
        return writer.build_uniform_leaf_document(
            rows,
            provenance,
            bank_sha256=BANK_SHA,
            state_corpus_sha256="3" * 64,
            rollouts=3,
            max_plies=40,
            seed=99,
            threads=2,
            branch_on_damage=False,
            price=_fake_price,
            writer_sha256=WRITER,
            validator_sha256=VALIDATOR,
        )

    def test_emits_exact_keyed_values_without_publishing_private_states(self) -> None:
        document = self._document()

        self.assertEqual(document["schema"], writer.OUTPUT_SCHEMA)
        self.assertEqual(document["provenance"]["rollout_policy"], "uniform")
        self.assertEqual(document["provenance"]["value_frame"], "side_one_absolute")
        self.assertEqual(document["provenance"]["checkpoint_sha256"], CHECKPOINT)
        self.assertEqual(document["provenance"]["native_extension_sha256"], EXTENSION)
        self.assertEqual(document["provenance"]["deploy_validator_sha256"], VALIDATOR)
        self.assertEqual(document["rollout_ledger"]["rollouts_run"], 12)
        self.assertEqual(len(document["leaves"]), 4)
        self.assertTrue(all("state" not in row for row in document["leaves"]))
        self.assertTrue(all("state_sha256" in row for row in document["leaves"]))
        self.assertEqual(
            {(row["seed"], row["prefix"], row["seat"], row["arm"]) for row in document["leaves"]},
            {(1, 2, "p1", 0), (1, 2, "p1", 3), (7, 11, "p2", 4), (7, 11, "p2", 8)},
        )

    def test_key_derived_ordinals_are_input_order_independent(self) -> None:
        original = self._document()
        source = _state_document()
        source["leaves"].reverse()
        self.states_path.write_text(json.dumps(source), encoding="utf-8")
        reordered = self._document()

        ordinal_by_key = lambda doc: {
            (row["seed"], row["prefix"], row["seat"], row["arm"]): row["ordinal"]
            for row in doc["leaves"]
        }
        self.assertEqual(ordinal_by_key(original), ordinal_by_key(reordered))

    def test_refuses_a_missing_or_foreign_canonical_leaf(self) -> None:
        source = _state_document()
        source["leaves"].pop()
        self.states_path.write_text(json.dumps(source), encoding="utf-8")
        expected_keys, belief_hash = writer.load_banked_keys(self.bank_path)

        with self.assertRaisesRegex(writer.UniformLeafWriterError, "missing 1 canonical leaves"):
            writer.load_leaf_states(
                self.states_path,
                expected_keys=expected_keys,
                bank_sha256=BANK_SHA,
                belief_hash=belief_hash,
                checkpoint_sha256=CHECKPOINT,
            )

    def test_refuses_a_state_hash_lie(self) -> None:
        source = _state_document()
        source["leaves"][0]["state_sha256"] = "0" * 64
        self.states_path.write_text(json.dumps(source), encoding="utf-8")

        with self.assertRaisesRegex(writer.UniformLeafWriterError, "does not match the serialized state"):
            self._loaded()

    def test_refuses_different_opponent_actions_for_sibling_arms(self) -> None:
        source = _state_document()
        source["leaves"][1]["opponent_action"] = 9
        self.states_path.write_text(json.dumps(source), encoding="utf-8")

        with self.assertRaisesRegex(writer.UniformLeafWriterError, "differs between the two arms"):
            self._loaded()

    def test_refuses_state_corpus_with_another_checkpoint_or_bank(self) -> None:
        source = _state_document()
        source["provenance"]["checkpoint_sha256"] = "0" * 64
        self.states_path.write_text(json.dumps(source), encoding="utf-8")

        with self.assertRaisesRegex(writer.UniformLeafWriterError, "checkpoint_sha256 does not match"):
            self._loaded()

        source = _state_document(bank_sha="0" * 64)
        self.states_path.write_text(json.dumps(source), encoding="utf-8")
        with self.assertRaisesRegex(writer.UniformLeafWriterError, "bank_sha256 does not match"):
            self._loaded()

    def test_refuses_a_native_pricer_that_changes_the_declared_config(self) -> None:
        rows, provenance = self._loaded()

        def wrong_config(states, ordinals, **config):
            report = _report([0.5] * len(states), **config)
            report["rollout_threads"] = 1
            return report, EXTENSION

        with self.assertRaisesRegex(writer.UniformLeafWriterError, "rollout_threads=1, expected 2"):
            writer.build_uniform_leaf_document(
                rows, provenance, bank_sha256=BANK_SHA, state_corpus_sha256="3" * 64,
                rollouts=3, max_plies=40, seed=99, threads=2, branch_on_damage=False,
                price=wrong_config, writer_sha256=WRITER, validator_sha256=VALIDATOR,
            )

    def test_refuses_an_out_of_range_native_value(self) -> None:
        rows, provenance = self._loaded()

        def impossible_value(states, ordinals, **config):
            return _report([0.5, 0.5, 1.1, 0.5], **config), EXTENSION

        with self.assertRaisesRegex(writer.UniformLeafWriterError, "finite probability in \\[0, 1\\]"):
            writer.build_uniform_leaf_document(
                rows, provenance, bank_sha256=BANK_SHA, state_corpus_sha256="3" * 64,
                rollouts=3, max_plies=40, seed=99, threads=2, branch_on_damage=False,
                price=impossible_value, writer_sha256=WRITER, validator_sha256=VALIDATOR,
            )

    def test_refuses_a_cap_or_dead_end_fallback_instead_of_calling_the_blend_uniform(self) -> None:
        rows, provenance = self._loaded()

        def fallback_blend(states, ordinals, **config):
            report = _report([0.5] * len(states), **config)
            report["rollout_terminal_hits"] -= 1
            report["rollout_cap_hits"] = 1
            report["rollout_terminal_fraction"] = report["rollout_terminal_hits"] / report["rollouts_run"]
            report["rollout_fallback_fraction"] = 1 / report["rollouts_run"]
            return report, EXTENSION

        with self.assertRaisesRegex(writer.UniformLeafWriterError, "nonterminal rollouts"):
            writer.build_uniform_leaf_document(
                rows, provenance, bank_sha256=BANK_SHA, state_corpus_sha256="3" * 64,
                rollouts=3, max_plies=40, seed=99, threads=2, branch_on_damage=False,
                price=fallback_blend, writer_sha256=WRITER, validator_sha256=VALIDATOR,
            )

    def test_records_a_completed_terminal_successor_as_an_exact_uniform_value(self) -> None:
        source = _state_document()
        source["leaves"][0] = _terminal_row(1, 2, "p1", 0, 5, "p1")
        self.states_path.write_text(json.dumps(source), encoding="utf-8")

        document = self._document()

        terminal_leaf = next(row for row in document["leaves"] if row["arm"] == 0)
        self.assertEqual(terminal_leaf["uniform_value"], 1.0)
        self.assertEqual(terminal_leaf["value_source"], "exact_terminal")
        self.assertIn("terminal_sha256", terminal_leaf)
        self.assertNotIn("state_sha256", terminal_leaf)
        self.assertEqual(document["provenance"]["native_priced_leaves"], 3)
        self.assertEqual(document["provenance"]["terminal_successor_leaves"], 1)
        self.assertEqual(document["rollout_ledger"]["rollouts_run"], 9)
        self.assertEqual(document["rollout_ledger"]["total_leaves"], 4)

    def test_refuses_a_capped_or_forged_terminal_successor(self) -> None:
        source = _state_document()
        terminal = _terminal_row(1, 2, "p1", 0, 5, None)
        terminal["terminal"]["capped"] = True
        terminal["terminal_sha256"] = hashlib.sha256(writer.canonical_json(terminal["terminal"])).hexdigest()
        source["leaves"][0] = terminal
        self.states_path.write_text(json.dumps(source), encoding="utf-8")

        with self.assertRaisesRegex(writer.UniformLeafWriterError, "uncapped completed battle"):
            self._loaded()

        terminal = _terminal_row(1, 2, "p1", 0, 5, "p1")
        terminal["terminal_value"] = 0.0
        source["leaves"][0] = terminal
        self.states_path.write_text(json.dumps(source), encoding="utf-8")
        with self.assertRaisesRegex(writer.UniformLeafWriterError, "exact side-one terminal result"):
            self._loaded()

    def test_refuses_an_unreviewed_validator_before_pricing(self) -> None:
        noop = Path(self.tmp.name) / "noop_validator.py"
        noop.write_text("# A zero-exit process is not an estimand gate.\n", encoding="utf-8")

        with self.assertRaisesRegex(writer.UniformLeafWriterError, "pinned reader identity"):
            writer._snapshot_reviewed_validator(noop, snapshot_dir=Path(self.tmp.name))

    def test_validates_the_private_transaction_snapshots_after_public_paths_change(self) -> None:
        """Replaced public validator/bank/state paths cannot alter a PASS.

        The initial source is the trusted tiny validator: it writes a complete
        PASS document with the exact bank and uniform-artifact bytes it read.
        The injected pricing stage replaces every original public input.  The
        writer must validate its pre-price snapshots and only then publish the
        byte-identical uniform artifact, rather than reopening any mutable
        pathname.
        """

        trusted_source = """\
import hashlib
import json
import sys
from pathlib import Path
bank = Path(sys.argv[sys.argv.index("--banked-pairs") + 1]).read_bytes()
uniform = Path(sys.argv[sys.argv.index("--uniform-leaves") + 1]).read_bytes()
out = Path(sys.argv[sys.argv.index("--out") + 1])
out.write_text(json.dumps({
    "schema": "pokezero.rollout-leaf-estimand-verdict.v1",
    "verdict": "PASS",
    "failures": [],
    "n_positions": 465,
    "n_leaves": 930,
    "seen_bank_sha256": hashlib.sha256(bank).hexdigest(),
    "seen_uniform_sha256": hashlib.sha256(uniform).hexdigest(),
    "uniform_leaf_artifact_sha256": hashlib.sha256(uniform).hexdigest(),
}), encoding="utf-8")
"""
        validator = Path(self.tmp.name) / "reviewed_validator.py"
        validator.write_text(trusted_source, encoding="utf-8")
        output = Path(self.tmp.name) / "uniform.json"
        verdict = Path(self.tmp.name) / "verdict.json"
        original_bank_sha256 = hashlib.sha256(self.bank_path.read_bytes()).hexdigest()
        original_states_sha256 = hashlib.sha256(self.states_path.read_bytes()).hexdigest()
        args = types.SimpleNamespace(
            checkpoint_sha256=CHECKPOINT,
            rollouts=3,
            rollout_max_plies=40,
            rollout_seed=99,
            rollout_threads=2,
            rollout_branch_on_damage="false",
            out=output,
            validator=validator,
            verdict_out=verdict,
            banked_pairs=self.bank_path,
            leaf_states=self.states_path,
        )
        expected_keys = {
            writer.LeafKey(seed=index, prefix=0, seat="p1", arm=0)
            for index in range(writer.CANONICAL_LEAVES)
        }

        def price_then_replace_public_inputs(*_args, **kwargs):
            validator.write_text("raise SystemExit(71)\n", encoding="utf-8")
            self.bank_path.write_text('{"replacement": "bank"}\n', encoding="utf-8")
            self.states_path.write_text('{"replacement": "states"}\n', encoding="utf-8")
            return {
                "schema": writer.OUTPUT_SCHEMA,
                "provenance": {
                    "bank_sha256": kwargs["bank_sha256"],
                    "state_corpus_sha256": kwargs["state_corpus_sha256"],
                },
            }

        with (
            mock.patch.object(
                writer,
                "REVIEWED_DEPLOY_VALIDATOR_SHA256",
                hashlib.sha256(trusted_source.encode("utf-8")).hexdigest(),
            ),
            mock.patch.object(writer, "build_parser", return_value=types.SimpleNamespace(parse_args=lambda _argv: args)),
            mock.patch.object(writer, "load_banked_keys", return_value=(expected_keys, BELIEF_HASH)),
            mock.patch.object(writer, "load_leaf_states", return_value=({}, {})),
            mock.patch.object(writer, "build_uniform_leaf_document", side_effect=price_then_replace_public_inputs),
        ):
            self.assertEqual(writer.main([]), 0)

        self.assertEqual(validator.read_text(encoding="utf-8"), "raise SystemExit(71)\n")
        published = output.read_bytes()
        validated = json.loads(verdict.read_text(encoding="utf-8"))
        self.assertEqual(validated["verdict"], "PASS")
        self.assertEqual(validated["seen_bank_sha256"], original_bank_sha256)
        self.assertEqual(validated["seen_uniform_sha256"], hashlib.sha256(published).hexdigest())
        self.assertEqual(json.loads(published)["provenance"]["bank_sha256"], original_bank_sha256)
        self.assertEqual(
            json.loads(published)["provenance"]["state_corpus_sha256"], original_states_sha256
        )

    def test_zero_exit_validator_without_a_fresh_verdict_is_not_a_validation(self) -> None:
        noop = Path(self.tmp.name) / "noop_validator.py"
        noop.write_text("# exits zero without creating --out\n", encoding="utf-8")
        verdict = Path(self.tmp.name) / "verdict.json"

        with self.assertRaisesRegex(writer.UniformLeafWriterError, "does not exist"):
            writer._run_reviewed_validator(
                noop,
                banked_pairs=self.bank_path,
                uniform_leaves=self.states_path,
                verdict_out=verdict,
                expected_positions=writer.CANONICAL_POSITIONS,
                expected_leaves=writer.CANONICAL_LEAVES,
            )
        self.assertFalse(verdict.exists())

    def test_requires_a_complete_passing_deploy_verdict(self) -> None:
        verdict = Path(self.tmp.name) / "verdict.json"
        verdict.write_text(
            json.dumps({
                "schema": writer.VERDICT_SCHEMA,
                "verdict": "PASS",
                "failures": [],
                "n_positions": writer.CANONICAL_POSITIONS,
                "n_leaves": writer.CANONICAL_LEAVES,
                "uniform_leaf_artifact_sha256": "4" * 64,
            }),
            encoding="utf-8",
        )
        writer._require_passing_verdict(
            verdict,
            expected_positions=writer.CANONICAL_POSITIONS,
            expected_leaves=writer.CANONICAL_LEAVES,
            expected_uniform_leaf_artifact_sha256="4" * 64,
        )

        verdict.write_text(
            json.dumps({
                "schema": writer.VERDICT_SCHEMA,
                "verdict": "PASS",
                "failures": [],
                "n_positions": writer.CANONICAL_POSITIONS - 1,
                "n_leaves": writer.CANONICAL_LEAVES,
                "uniform_leaf_artifact_sha256": "4" * 64,
            }),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(writer.UniformLeafWriterError, "wrong position count"):
            writer._require_passing_verdict(
                verdict,
                expected_positions=writer.CANONICAL_POSITIONS,
                expected_leaves=writer.CANONICAL_LEAVES,
                expected_uniform_leaf_artifact_sha256="4" * 64,
            )

        verdict.write_text(
            json.dumps({
                "schema": writer.VERDICT_SCHEMA,
                "verdict": "PASS",
                "failures": [],
                "n_positions": writer.CANONICAL_POSITIONS,
                "n_leaves": writer.CANONICAL_LEAVES,
                "uniform_leaf_artifact_sha256": "0" * 64,
            }),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(writer.UniformLeafWriterError, "exact uniform artifact bytes"):
            writer._require_passing_verdict(
                verdict,
                expected_positions=writer.CANONICAL_POSITIONS,
                expected_leaves=writer.CANONICAL_LEAVES,
                expected_uniform_leaf_artifact_sha256="4" * 64,
            )

    def test_refuses_to_overwrite_an_existing_artifact(self) -> None:
        output = Path(self.tmp.name) / "existing.json"
        output.write_text("user evidence", encoding="utf-8")

        with self.assertRaisesRegex(writer.UniformLeafWriterError, "refusing to overwrite"):
            writer._write_new(output, "new evidence")
        self.assertEqual(output.read_text(encoding="utf-8"), "user evidence")

    def test_publish_does_not_overwrite_an_artifact_created_after_preflight(self) -> None:
        output = Path(self.tmp.name) / "racing-evidence.json"
        real_link = writer.os.link

        def another_writer_wins(source, destination):
            Path(destination).write_text("other evidence", encoding="utf-8")
            return real_link(source, destination)

        with mock.patch.object(writer.os, "link", side_effect=another_writer_wins):
            with self.assertRaisesRegex(writer.UniformLeafWriterError, "refusing to overwrite"):
                writer._write_new(output, "our evidence")
        self.assertEqual(output.read_text(encoding="utf-8"), "other evidence")

    def test_main_refuses_aliased_artifact_and_verdict_paths_before_any_write(self) -> None:
        shared = Path(self.tmp.name) / "shared-output.json"
        args = types.SimpleNamespace(
            checkpoint_sha256=CHECKPOINT,
            rollouts=3,
            rollout_max_plies=40,
            rollout_seed=99,
            rollout_threads=2,
            rollout_branch_on_damage="false",
            out=shared,
            validator=Path(self.tmp.name) / "unread-validator.py",
            verdict_out=shared,
            banked_pairs=self.bank_path,
            leaf_states=self.states_path,
        )

        with mock.patch.object(
            writer,
            "build_parser",
            return_value=types.SimpleNamespace(parse_args=lambda _argv: args),
        ):
            self.assertEqual(writer.main([]), 2)
        self.assertFalse(shared.exists())

    def test_hashes_the_loaded_native_extension_not_the_python_package_shim(self) -> None:
        extension = Path(self.tmp.name) / "pokezero_search.cpython-test.so"
        extension.write_bytes(b"native extension bytes")
        package_shim = Path(self.tmp.name) / "__init__.py"
        package_shim.write_bytes(b"from .pokezero_search import *\n")
        package = types.SimpleNamespace(
            __file__=str(package_shim),
            pokezero_search=types.SimpleNamespace(__file__=str(extension)),
        )

        self.assertEqual(
            writer._native_extension_sha256(package), hashlib.sha256(extension.read_bytes()).hexdigest()
        )

    def test_hashes_a_top_level_native_extension_layout_too(self) -> None:
        extension = Path(self.tmp.name) / "pokezero_search.cpython-test.so"
        extension.write_bytes(b"top-level native extension bytes")
        package = types.SimpleNamespace(__file__=str(extension))

        self.assertEqual(
            writer._native_extension_sha256(package), hashlib.sha256(extension.read_bytes()).hexdigest()
        )

    def test_refuses_a_package_shim_when_the_extension_path_is_not_available(self) -> None:
        package = types.SimpleNamespace(__file__=str(Path(self.tmp.name) / "__init__.py"))

        with self.assertRaisesRegex(writer.UniformLeafWriterError, "native extension"):
            writer._native_extension_sha256(package)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
