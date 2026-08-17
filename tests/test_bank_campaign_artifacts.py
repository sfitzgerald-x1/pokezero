"""Demonstrated failing inputs for every guard in `scripts/bank_campaign_artifacts.py`.

This suite is the deliverable, not the tool's paperwork. The banking tool exists because two
artifacts in the search-ceiling program silently lost whole sections when an input was **absent
rather than wrong**, and because the program adopted the rule that *every gate and guard ships
with a demonstrated failing input* -- three guards in this codebase were true by construction
and sat green for months. So each test below constructs the offending input and shows the
refusal, by code AND by the words the message must contain; a test that merely asserts "it
raised" would pass against a tool that refuses everything.

The suite grew a second layer after an independent review found **four inputs that produced a
written artifact which should have been refused**, two of which banked a directory that failed
its own `shasum -a 256 -c SHA256SUMS`. Every one of those is now a test in
`ReviewFoundTheseWroteAnArtifactTests`, constructed from the review's own reproduction, so the
regression is pinned by the input that caused it rather than by a description of it. The lesson
those four shared is worth stating: each came from **trusting a fact measured earlier about a
directory other processes can write to**, which is why the staged bytes are now re-hashed before
the rename instead of the source being assumed to hold still.

Four properties get their own tests because they are the ones that failed in the field:

* **absent != null != empty != zero.** `rollouts` missing, `rollouts: null`, `rollouts: 0` and
  `shards: []` must produce four different codes with four different messages. A tool that
  collapses them tells a caller to fix the wrong thing.
* **atomicity.** A failure injected *during* the copy loop must leave the destination
  non-existent and no staging directory behind. This one is injected rather than argued,
  because "validate everything first" is a claim about code order that a later edit can break
  silently.
* **the instrument-2 field set is enforced for every field it declares.** Rather than trusting
  a hand-written list of assertions, the meta-test drops each declared field in turn and
  requires a refusal -- so a field added to `ARTIFACT_KINDS` and forgotten in the validator
  cannot pass.
* **the estimand caveat is not forgeable.** A caller supplying its own `estimand` is refused,
  because the whole point is that the caveat travels with the number whether the writer wants
  it to or not.

Public-repo hygiene: every path, image reference, pod and node name below is a placeholder.
Nothing in this file names a real cluster, registry, namespace or shared filesystem.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bank_campaign_artifacts.py"
_SPEC = importlib.util.spec_from_file_location("bank_campaign_artifacts", _SCRIPT)
bank = importlib.util.module_from_spec(_SPEC)
# Registered BEFORE execution, unlike the other by-path script loaders in this suite:
# `@dataclass` resolves annotations through `sys.modules[cls.__module__]`, so a module executed
# while unregistered raises AttributeError inside dataclasses. A launcher importing this tool
# normally (`scripts/` on sys.path) gets the same registration for free.
sys.modules[_SPEC.name] = bank
_SPEC.loader.exec_module(bank)

ARBITER_KIND = "phase1.instrument2.rollout-arbiter.v1"
CAMPAIGN_ID = "bank-test-cell-20260817"

# Placeholder provenance values. Deliberately fictional and deliberately not shaped like any
# deployment: a fixture that copies real cluster strings puts them in the public repo forever.
PLACEHOLDER_CHECKPOINT = "/checkpoints/lineage-a/iteration-0001/transformer-policy.pt"
PLACEHOLDER_IMAGE = "images.example.invalid/pokezero@sha256:" + "1c" * 32
PLACEHOLDER_LAUNCHER = "launchers/rollout-arbiter-cell.sh"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class _Fixture:
    """A source directory of shards plus a complete, passing stamp to perturb.

    One field per perturbation. A fixture that changes two things at once pins neither, because
    the first check short-circuits and the second is never reached.
    """

    def __init__(self, root: Path, *, shard_count: int = 2) -> None:
        self.root = root
        self.source_dir = root / "harvest"
        self.store_dir = root / "campaign-store"
        self.out_dir = self.store_dir / CAMPAIGN_ID
        (self.source_dir / "results").mkdir(parents=True)
        self.store_dir.mkdir(parents=True)

        shards: list[dict[str, Any]] = []
        for index in range(shard_count):
            relative = f"results/shard{index}.json"
            payload = json.dumps({"shard": index, "pairs": [{"true_gap": 0.01 * index}]}).encode()
            (self.source_dir / relative).write_bytes(payload)
            shards.append({
                "file": relative,
                "sha256": _sha256_bytes(payload),
                "exit_status": 0,
                "pod": f"bank-test-pod-{index}",
                "node": f"node-{index}.example.invalid",
            })

        self.checkpoint_file = root / "transformer-policy.pt"
        self.checkpoint_file.write_bytes(b"placeholder-weights")
        self.checkpoint_sha256 = _sha256_bytes(b"placeholder-weights")

        self.stamp: dict[str, Any] = {
            "artifact_kind": ARBITER_KIND,
            "campaign_id": CAMPAIGN_ID,
            "cell": "Phase 1 instrument 2 -- oracle-leaf rollout arbiter (placeholder cell)",
            "checkpoint": PLACEHOLDER_CHECKPOINT,
            "checkpoint_sha256": self.checkpoint_sha256,
            "seed_band": "10000000-10000199 stride 100",
            "seeds": [10000000, 10000100],
            "rollouts": 64,
            "sims": 2048,
            "depth": 8,
            "arms": "rollout_crate vs raw, paired mirror",
            "image": PLACEHOLDER_IMAGE,
            "launcher": PLACEHOLDER_LAUNCHER,
            "launcher_flags": ["--mode", "fanout", "--shards", str(shard_count)],
            "rollout_policy": "uniform",
            "rollout_fallback_fraction": 0.0,
            "leaf_batch": 1,
            "rollout_threads": 1,
            "rollout_threads_cpu_budget_ack": True,
            "shards": shards,
        }

    def bank(self, **kwargs: Any) -> Any:
        return bank.bank_artifact(self.stamp, self.source_dir, self.out_dir, **kwargs)


class _FixtureCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fixture = _Fixture(Path(self._tmp.name))

    def refuse(self, **kwargs: Any) -> bank.BankRefusal:
        with self.assertRaises(bank.BankRefusal) as caught:
            self.fixture.bank(**kwargs)
        return caught.exception

    def assertRefusedWith(self, code: str, *, contains: str = "", **kwargs: Any) -> str:
        refusal = self.refuse(**kwargs)
        self.assertIn(code, refusal.codes, f"expected {code}, got {refusal.codes}")
        message = refusal.messages_for(code)[0]
        if contains:
            self.assertIn(contains, message)
        # A refusal must write nothing, whatever it refused for.
        self.assertFalse(self.fixture.out_dir.exists(), "a refusal created the output directory")
        return message


# ----------------------------------------------------------------------------------------------
# The happy path, so the refusals below are known to be refusing something real
# ----------------------------------------------------------------------------------------------
class BanksACompleteCellTests(_FixtureCase):
    def test_a_complete_stamp_banks_shards_provenance_and_sha256sums(self) -> None:
        result = self.fixture.bank()
        self.assertEqual(result.status, "banked")
        self.assertEqual(result.shard_count, 2)

        provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))
        self.assertEqual(provenance["schema"], bank.PROVENANCE_SCHEMA)
        self.assertEqual(provenance["artifact_kind"], ARBITER_KIND)
        self.assertEqual(provenance["shard_count"], 2)
        self.assertEqual(provenance["checkpoint"], PLACEHOLDER_CHECKPOINT)
        self.assertEqual(provenance["checkpoint_sha256"], self.fixture.checkpoint_sha256)
        # Per-shard exit status, pod and node -- the store contract's minimum.
        self.assertEqual([s["exit_status"] for s in provenance["shards"]], [0, 0])
        self.assertEqual([s["pod"] for s in provenance["shards"]],
                         ["bank-test-pod-0", "bank-test-pod-1"])
        self.assertEqual([s["node"] for s in provenance["shards"]],
                         ["node-0.example.invalid", "node-1.example.invalid"])
        # The {file, sha256, bytes} projection the store's existing readers consume.
        self.assertEqual([entry["file"] for entry in provenance["per_file"]],
                         ["results/shard0.json", "results/shard1.json"])
        self.assertTrue(all(entry["bytes"] > 0 for entry in provenance["per_file"]))

        for shard in self.fixture.stamp["shards"]:
            banked = self.fixture.out_dir / shard["file"]
            self.assertTrue(banked.is_file())
            self.assertEqual(bank.sha256_file(banked), shard["sha256"])

    def test_sha256sums_is_in_sha256sum_format_and_covers_every_shard(self) -> None:
        result = self.fixture.bank()
        lines = result.sha256sums_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        # Sorted by path, like the store's existing SHA256SUMS files.
        self.assertEqual([line.split("  ", 1)[1] for line in lines],
                         ["results/shard0.json", "results/shard1.json"])
        for line in lines:
            digest, _, relative = line.partition("  ")
            self.assertRegex(digest, r"\A[0-9a-f]{64}\Z")
            self.assertTrue((self.fixture.out_dir / relative).is_file())
            self.assertEqual(bank.sha256_file(self.fixture.out_dir / relative), digest)

    def test_the_standard_checksum_tool_verifies_the_banked_directory(self) -> None:
        """A later reader must be able to check the bank without this tool.

        `SHA256SUMS` is only useful if `sha256sum -c` / `shasum -a 256 -c` accepts it; a
        self-consistent format only this module can read is a private ledger, not evidence.

        Scope, measured rather than assumed: `shasum -c` tolerates a single-space separator, so
        this test does NOT pin the two-space convention -- the format test above does. What it
        pins is that the digests are the file's own, kill-confirmed by writing reversed digests.
        """
        tool = shutil.which("sha256sum") or shutil.which("shasum")
        if tool is None:  # pragma: no cover - environment dependent
            self.skipTest("no sha256sum/shasum on PATH")
        argv = [tool, "-c", bank.SHA256SUMS_FILENAME]
        if tool.endswith("shasum"):
            argv = [tool, "-a", "256", "-c", bank.SHA256SUMS_FILENAME]
        self.fixture.bank()
        done = subprocess.run(
            argv, cwd=self.fixture.out_dir, capture_output=True, text=True, check=False
        )
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(done.stdout.count(": OK"), 2)

    def test_the_estimand_caveats_travel_on_the_artifact(self) -> None:
        """Phase 1 instrument 2's numbers may not travel without what they are numbers OF."""
        provenance = json.loads(
            self.fixture.bank().provenance_path.read_text(encoding="utf-8")
        )
        estimand = provenance["estimand"]
        self.assertEqual(estimand["rollout_policy"], "uniform")
        self.assertIn("uniformly at random", estimand["prices"])
        self.assertIn("true_a/true_b", estimand["is_not"])
        self.assertIn("POLICY continuation", estimand["is_not"])
        self.assertIn("confounded", estimand["consequence"])
        self.assertEqual(estimand["rollout_fallback_fraction"], 0.0)
        self.assertEqual(estimand["leaf_batch"], 1)
        self.assertEqual(estimand["rollout_threads"], 1)
        self.assertIs(estimand["rollout_threads_cpu_budget_ack"], True)

    def test_a_nonzero_fallback_fraction_is_labelled_a_blend_not_an_oracle(self) -> None:
        self.fixture.stamp["rollout_fallback_fraction"] = 0.42
        estimand = json.loads(
            self.fixture.bank().provenance_path.read_text(encoding="utf-8")
        )["estimand"]
        self.assertEqual(estimand["rollout_fallback_fraction"], 0.42)
        self.assertIn("BLEND", estimand["fallback_note"])

    def test_a_widened_leaf_batch_is_labelled_a_fidelity_loss(self) -> None:
        self.fixture.stamp["leaf_batch"] = 8
        estimand = json.loads(
            self.fixture.bank().provenance_path.read_text(encoding="utf-8")
        )["estimand"]
        self.assertIn("fidelity LOSS", estimand["leaf_batch_note"])

    def test_re_banking_the_same_artifact_is_a_no_op_not_a_rewrite(self) -> None:
        first = self.fixture.bank(banked_at_utc="2026-08-17T00:00:00Z")
        before = first.provenance_path.read_text(encoding="utf-8")
        second = self.fixture.bank(banked_at_utc="2026-08-18T11:22:33Z")
        self.assertEqual(second.status, "unchanged")
        self.assertEqual(first.provenance_path.read_text(encoding="utf-8"), before)


# ----------------------------------------------------------------------------------------------
# absent != null != empty != zero
# ----------------------------------------------------------------------------------------------
class AbsentIsNotNullIsNotEmptyIsNotZeroTests(_FixtureCase):
    def test_four_distinct_codes_and_four_distinct_messages(self) -> None:
        """The failure this tool exists for: an ABSENT input read as a zero or a null.

        `rollouts` missing, null and 0 are three defects with three fixes, and `shards: []`
        is a fourth. Collapsing any pair of them sends the caller to the wrong field.
        """
        missing = dict(self.fixture.stamp)
        missing.pop("rollouts")
        nulled = dict(self.fixture.stamp, rollouts=None)
        zeroed = dict(self.fixture.stamp, rollouts=0)
        emptied = dict(self.fixture.stamp, shards=[])

        observed: dict[str, tuple[str, str]] = {}
        for label, stamp in (
            ("missing", missing), ("null", nulled), ("zero", zeroed), ("empty", emptied),
        ):
            with self.assertRaises(bank.BankRefusal) as caught:
                bank.bank_artifact(stamp, self.fixture.source_dir, self.fixture.out_dir)
            reasons = caught.exception.reasons
            self.assertEqual(len(reasons), 1, f"{label}: {caught.exception.codes}")
            observed[label] = (reasons[0].code, reasons[0].message)

        self.assertEqual(observed["missing"][0], bank.MISSING_REQUIRED_FIELD)
        self.assertEqual(observed["null"][0], bank.NULL_REQUIRED_FIELD)
        self.assertEqual(observed["zero"][0], bank.INVALID_FIELD_VALUE)
        self.assertEqual(observed["empty"][0], bank.EMPTY_SHARD_LIST)
        self.assertEqual(len({code for code, _ in observed.values()}), 4)
        self.assertEqual(len({message for _, message in observed.values()}), 4)

        self.assertIn("is ABSENT", observed["missing"][1])
        self.assertIn("present but NULL", observed["null"][1])
        self.assertIn("'rollouts' = 0", observed["zero"][1])
        self.assertIn("VALUE, not an absence", observed["zero"][1])
        self.assertIn("EMPTY list", observed["empty"][1])
        self.assertFalse(self.fixture.out_dir.exists())

    def test_zero_is_refused_for_every_positive_integer_field(self) -> None:
        for name in ("rollouts", "sims", "depth", "leaf_batch", "rollout_threads"):
            with self.subTest(field=name):
                stamp = dict(self.fixture.stamp)
                stamp[name] = 0
                with self.assertRaises(bank.BankRefusal) as caught:
                    bank.bank_artifact(stamp, self.fixture.source_dir, self.fixture.out_dir)
                self.assertIn(bank.INVALID_FIELD_VALUE, caught.exception.codes)

    def test_an_empty_string_is_refused_and_is_not_the_absent_message(self) -> None:
        self.fixture.stamp["cell"] = ""
        message = self.assertRefusedWith(bank.INVALID_FIELD_VALUE, contains="'cell' = \"\"")
        self.assertNotIn("is ABSENT", message)


# ----------------------------------------------------------------------------------------------
# Field-level guards
# ----------------------------------------------------------------------------------------------
class RequiredFieldGuardTests(_FixtureCase):
    def _drop(self, kind_name: str, *names: str) -> bank.BankRefusal:
        stamp = {
            key: value for key, value in self.fixture.stamp.items() if key not in names
        }
        stamp["artifact_kind"] = kind_name
        with self.assertRaises(bank.BankRefusal) as caught:
            bank.bank_artifact(stamp, self.fixture.source_dir, self.fixture.out_dir)
        self.assertFalse(self.fixture.out_dir.exists())
        return caught.exception

    def test_every_declared_field_of_every_kind_is_actually_enforced(self) -> None:
        """The guard against a field added to the schema and forgotten in the validator.

        Data-driven both ways: the loop reads `ARTIFACT_KINDS`, so a new required field arrives
        with its own failing-input demonstration for free -- and a field declared but never
        checked fails here instead of passing silently. One field is dropped per subtest,
        because a fixture that changes two things at once pins neither.
        """
        for kind in bank.ARTIFACT_KINDS.values():
            for spec in kind.required:
                if spec.name == "shards":
                    continue  # covered by EMPTY_SHARD_LIST and the shard-record tests below
                with self.subTest(kind=kind.name, field=spec.name):
                    refusal = self._drop(kind.name, spec.name)
                    self.assertIn(bank.MISSING_REQUIRED_FIELD, refusal.codes)
                    self.assertTrue(
                        any(f"'{spec.name}'" in message
                            for message in refusal.messages_for(bank.MISSING_REQUIRED_FIELD)),
                        f"the refusal for {spec.name} does not name it: {refusal.codes}",
                    )
            for group in kind.required_one_of:
                names = tuple(spec.name for spec in group)
                with self.subTest(kind=kind.name, one_of=names):
                    refusal = self._drop(kind.name, *names)
                    self.assertIn(bank.MISSING_REQUIRED_ALTERNATIVE, refusal.codes)

    def test_the_instrument2_required_set_is_pinned(self) -> None:
        """Adding or removing a required field for this cell must be a deliberate edit."""
        kind = bank.ARTIFACT_KINDS[ARBITER_KIND]
        self.assertEqual(kind.field_names(), (
            "campaign_id", "cell", "checkpoint", "checkpoint_sha256",
            "rollouts", "sims", "depth", "arms", "image", "launcher", "launcher_flags",
            "shards",
            "rollout_policy", "rollout_fallback_fraction", "leaf_batch", "rollout_threads",
            "rollout_threads_cpu_budget_ack",
            "seed_band", "seeds",
        ))
        self.assertEqual(
            tuple(spec.name for spec in kind.file_required),
            ("file", "sha256", "exit_status", "pod", "node"),
        )

    def test_a_malformed_checkpoint_sha256_is_refused(self) -> None:
        self.fixture.stamp["checkpoint_sha256"] = "definitely-not-a-sha256"
        self.assertRefusedWith(bank.INVALID_FIELD_VALUE, contains="'checkpoint_sha256'")

    def test_a_stringified_integer_is_refused(self) -> None:
        self.fixture.stamp["sims"] = "2048"
        self.assertRefusedWith(bank.INVALID_FIELD_VALUE, contains="'sims' = \"2048\"")

    def test_a_truthy_string_does_not_pass_for_the_cpu_budget_acknowledgement(self) -> None:
        # The string "yes" is truthy in Python, and this field is an assertion about a
        # deployment: a bool test that accepts any truthy value asserts nothing.
        self.fixture.stamp["rollout_threads_cpu_budget_ack"] = "yes"
        self.assertRefusedWith(
            bank.INVALID_FIELD_VALUE, contains="rollout_threads_cpu_budget_ack"
        )

    def test_a_seed_band_and_seed_list_may_not_both_be_absent(self) -> None:
        self.fixture.stamp.pop("seed_band")
        self.fixture.stamp.pop("seeds")
        self.assertRefusedWith(bank.MISSING_REQUIRED_ALTERNATIVE, contains="seed_band")

    def test_an_empty_seed_list_is_refused_even_though_seed_band_is_present(self) -> None:
        self.fixture.stamp["seeds"] = []
        self.assertRefusedWith(bank.INVALID_FIELD_VALUE, contains="'seeds' = []")

    def test_the_launcher_flag_list_may_be_empty_but_the_key_may_not_be_absent(self) -> None:
        self.fixture.stamp["launcher_flags"] = []
        self.assertEqual(self.fixture.bank().status, "banked")

    def test_all_unmet_requirements_are_reported_together(self) -> None:
        """One run per missing field would teach callers to guess. Report the whole list."""
        for name in ("checkpoint_sha256", "image", "launcher"):
            self.fixture.stamp.pop(name)
        refusal = self.refuse()
        self.assertEqual(len(refusal.reasons), 3)
        self.assertIn("3 unmet requirements", str(refusal))
        for name in ("checkpoint_sha256", "image", "launcher"):
            self.assertTrue(any(f"'{name}'" in message for message in
                                refusal.messages_for(bank.MISSING_REQUIRED_FIELD)))


# ----------------------------------------------------------------------------------------------
# Shard-level guards
# ----------------------------------------------------------------------------------------------
class ShardGuardTests(_FixtureCase):
    def test_a_declared_shard_file_that_does_not_exist_is_refused(self) -> None:
        self.fixture.stamp["shards"].append({
            "file": "results/shard9.json",
            "sha256": _sha256_bytes(b"never written"),
            "exit_status": 0,
            "pod": "bank-test-pod-9",
            "node": "node-9.example.invalid",
        })
        self.assertRefusedWith(bank.SHARD_FILE_ABSENT, contains="results/shard9.json")

    def test_a_shard_whose_recomputed_sha256_disagrees_is_refused(self) -> None:
        """The declared sha is a citation. An unverified one cites nothing."""
        target = self.fixture.source_dir / self.fixture.stamp["shards"][1]["file"]
        target.write_bytes(b'{"shard": 1, "pairs": [], "edited": true}')
        message = self.assertRefusedWith(bank.SHARD_SHA256_MISMATCH, contains="hashes to")
        self.assertIn(bank.sha256_file(target), message)

    def test_two_records_for_one_file_are_refused(self) -> None:
        self.fixture.stamp["shards"].append(dict(self.fixture.stamp["shards"][0]))
        self.assertRefusedWith(
            bank.DUPLICATE_SHARD_FILE, contains="the same declared name as shards[0]")

    def test_a_shard_path_escaping_the_source_directory_is_refused(self) -> None:
        for relative in ("../outside.json", "/absolute/shard.json", "results/../../up.json"):
            with self.subTest(path=relative):
                stamp = dict(self.fixture.stamp)
                stamp["shards"] = [dict(stamp["shards"][0], file=relative)]
                with self.assertRaises(bank.BankRefusal) as caught:
                    bank.bank_artifact(stamp, self.fixture.source_dir, self.fixture.out_dir)
                self.assertIn(bank.UNSAFE_SHARD_PATH, caught.exception.codes)

    def test_a_shard_missing_its_exit_status_is_refused(self) -> None:
        self.fixture.stamp["shards"][0].pop("exit_status")
        message = self.assertRefusedWith(bank.MISSING_REQUIRED_FIELD, contains="'exit_status'")
        self.assertIn("shards[0]", message)

    def test_a_null_exit_status_is_refused_separately_from_an_absent_one(self) -> None:
        absent = dict(self.fixture.stamp)
        absent["shards"] = [
            {k: v for k, v in absent["shards"][0].items() if k != "exit_status"}
        ]
        nulled = dict(self.fixture.stamp)
        nulled["shards"] = [dict(nulled["shards"][0], exit_status=None)]
        codes = []
        for stamp in (absent, nulled):
            with self.assertRaises(bank.BankRefusal) as caught:
                bank.bank_artifact(stamp, self.fixture.source_dir, self.fixture.out_dir)
            codes.append(caught.exception.codes[0])
        self.assertEqual(codes, [bank.MISSING_REQUIRED_FIELD, bank.NULL_REQUIRED_FIELD])

    def test_a_shard_missing_its_pod_or_node_is_refused(self) -> None:
        for name in ("pod", "node"):
            with self.subTest(field=name):
                stamp = dict(self.fixture.stamp)
                stamp["shards"] = [
                    {k: v for k, v in stamp["shards"][0].items() if k != name}
                ]
                with self.assertRaises(bank.BankRefusal) as caught:
                    bank.bank_artifact(stamp, self.fixture.source_dir, self.fixture.out_dir)
                self.assertIn(bank.MISSING_REQUIRED_FIELD, caught.exception.codes)

    def test_a_bare_filename_instead_of_a_record_is_refused(self) -> None:
        self.fixture.stamp["shards"] = ["results/shard0.json"]
        self.assertRefusedWith(bank.SHARD_NOT_A_RECORD, contains="not a record")

    def test_a_failed_shard_is_refused_unless_the_failure_is_acknowledged(self) -> None:
        self.fixture.stamp["shards"][1]["exit_status"] = 137
        self.assertRefusedWith(bank.UNACKNOWLEDGED_SHARD_FAILURE, contains="exited 137")

    def test_an_acknowledged_failed_shard_banks_and_says_so_on_the_artifact(self) -> None:
        self.fixture.stamp["shards"][1]["exit_status"] = 137
        result = self.fixture.bank(allow_nonzero_shard_exit=True)
        provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))
        self.assertEqual(provenance["nonzero_exit_shards_acknowledged"], ["results/shard1.json"])
        self.assertEqual(provenance["shards"][1]["exit_status"], 137)


# ----------------------------------------------------------------------------------------------
# Instrument-2 estimand guards
# ----------------------------------------------------------------------------------------------
class EstimandGuardTests(_FixtureCase):
    def test_a_rollout_policy_with_no_declared_estimand_is_refused(self) -> None:
        """A caveat the tool has never been told cannot be stated, and silence reads as 'none'."""
        self.fixture.stamp["rollout_policy"] = "greedy_policy_continuation"
        self.assertRefusedWith(
            bank.UNDECLARED_ESTIMAND, contains="has no declared estimand caveat"
        )

    def test_extra_rollout_threads_without_the_cpu_budget_acknowledgement_are_refused(self) -> None:
        self.fixture.stamp["rollout_threads"] = 8
        self.fixture.stamp["rollout_threads_cpu_budget_ack"] = False
        self.assertRefusedWith(
            bank.UNACKNOWLEDGED_CPU_BUDGET, contains="rollout_threads=8 > 1"
        )

    def test_a_single_thread_needs_no_acknowledgement(self) -> None:
        self.fixture.stamp["rollout_threads"] = 1
        self.fixture.stamp["rollout_threads_cpu_budget_ack"] = False
        self.assertEqual(self.fixture.bank().status, "banked")

    def test_a_caller_supplied_estimand_is_refused_rather_than_honoured(self) -> None:
        self.fixture.stamp["estimand"] = {"prices": "whatever the author preferred"}
        self.assertRefusedWith(bank.DERIVED_FIELD_SUPPLIED, contains="estimand")

    def test_every_derived_key_is_refused_if_supplied(self) -> None:
        for name in sorted(bank.DERIVED_KEYS):
            with self.subTest(key=name):
                stamp = dict(self.fixture.stamp)
                stamp[name] = "forged"
                with self.assertRaises(bank.BankRefusal) as caught:
                    bank.bank_artifact(stamp, self.fixture.source_dir, self.fixture.out_dir)
                self.assertIn(bank.DERIVED_FIELD_SUPPLIED, caught.exception.codes)

    def test_the_base_kind_does_not_require_the_arbiter_fields(self) -> None:
        """The kinds are genuinely different; this suite is not testing one kind twice."""
        stamp = {
            key: value for key, value in self.fixture.stamp.items()
            if key not in {spec.name for spec in bank.ROLLOUT_ARBITER_REQUIRED}
        }
        stamp["artifact_kind"] = "campaign.v1"
        result = bank.bank_artifact(stamp, self.fixture.source_dir, self.fixture.out_dir)
        provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))
        self.assertNotIn("estimand", provenance)


# ----------------------------------------------------------------------------------------------
# Destination, kind and checkpoint guards
# ----------------------------------------------------------------------------------------------
class DestinationGuardTests(_FixtureCase):
    def test_a_destination_holding_a_differing_artifact_is_refused(self) -> None:
        self.fixture.bank()
        self.fixture.stamp["sims"] = 4096  # same campaign id, different run
        with self.assertRaises(bank.BankRefusal) as caught:
            self.fixture.bank()
        self.assertIn(bank.DESTINATION_HOLDS_DIFFERENT_ARTIFACT, caught.exception.codes)
        # The banked artifact is untouched: still the first run's stamp.
        provenance = json.loads(
            (self.fixture.out_dir / bank.PROVENANCE_FILENAME).read_text(encoding="utf-8")
        )
        self.assertEqual(provenance["sims"], 2048)

    def test_a_dumped_directory_is_not_silently_upgraded(self) -> None:
        """Shards with no stamp is the 'dumped, not banked' state the store contract names."""
        self.fixture.out_dir.mkdir(parents=True)
        (self.fixture.out_dir / "stray.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(bank.BankRefusal) as caught:
            self.fixture.bank()
        self.assertIn(bank.DESTINATION_HOLDS_DIFFERENT_ARTIFACT, caught.exception.codes)

    def test_overwrite_moves_the_displaced_artifact_aside_instead_of_deleting_it(self) -> None:
        self.fixture.bank()
        self.fixture.stamp["sims"] = 4096
        result = self.fixture.bank(overwrite=True)
        self.assertEqual(result.status, "banked")
        self.assertIsNotNone(result.replaced_path)
        displaced = json.loads(
            (result.replaced_path / bank.PROVENANCE_FILENAME).read_text(encoding="utf-8")
        )
        self.assertEqual(displaced["sims"], 2048)
        current = json.loads(result.provenance_path.read_text(encoding="utf-8"))
        self.assertEqual(current["sims"], 4096)

    def test_banking_into_a_directory_not_named_after_the_campaign_is_refused(self) -> None:
        """The campaign nearly lost to a scratch directory, made unrepresentable."""
        scratch = self.fixture.store_dir / "scratch"
        with self.assertRaises(bank.BankRefusal) as caught:
            bank.bank_artifact(self.fixture.stamp, self.fixture.source_dir, scratch)
        self.assertIn(bank.CAMPAIGN_ID_DIRECTORY_MISMATCH, caught.exception.codes)
        self.assertFalse(scratch.exists())

    def test_an_absent_source_directory_is_refused(self) -> None:
        with self.assertRaises(bank.BankRefusal) as caught:
            bank.bank_artifact(
                self.fixture.stamp, self.fixture.root / "no-such-harvest", self.fixture.out_dir
            )
        self.assertIn(bank.SOURCE_DIR_ABSENT, caught.exception.codes)

    def test_an_unknown_artifact_kind_is_refused(self) -> None:
        self.fixture.stamp["artifact_kind"] = "campaign.v99"
        self.assertRefusedWith(bank.UNKNOWN_ARTIFACT_KIND, contains="campaign.v99")

    def test_a_kind_declared_two_different_ways_is_refused(self) -> None:
        self.assertRefusedWith(
            bank.ARTIFACT_KIND_MISMATCH, contains="campaign.v1", kind="campaign.v1"
        )

    def test_a_stamp_that_is_not_an_object_is_refused(self) -> None:
        with self.assertRaises(bank.BankRefusal) as caught:
            bank.bank_artifact(
                [self.fixture.stamp], self.fixture.source_dir, self.fixture.out_dir
            )
        self.assertIn(bank.STAMP_NOT_A_MAPPING, caught.exception.codes)

    def test_a_checkpoint_that_rehashes_differently_is_refused(self) -> None:
        other = self.fixture.root / "other-weights.pt"
        other.write_bytes(b"different-placeholder-weights")
        message = self.assertRefusedWith(
            bank.CHECKPOINT_SHA256_MISMATCH, contains="hashes to", verify_checkpoint=other
        )
        self.assertIn(bank.sha256_file(other), message)

    def test_the_matching_checkpoint_passes_verification(self) -> None:
        result = self.fixture.bank(verify_checkpoint=self.fixture.checkpoint_file)
        self.assertEqual(result.status, "banked")

    def test_an_unhashable_checkpoint_is_not_a_silent_pass(self) -> None:
        self.assertRefusedWith(
            bank.CHECKPOINT_SHA256_MISMATCH,
            contains="not a file",
            verify_checkpoint=self.fixture.root / "absent.pt",
        )


# ----------------------------------------------------------------------------------------------
# Atomicity
# ----------------------------------------------------------------------------------------------
class AtomicityTests(_FixtureCase):
    def _siblings(self) -> list[str]:
        return sorted(entry.name for entry in self.fixture.store_dir.iterdir())

    def test_a_validation_refusal_leaves_the_destination_and_its_parent_untouched(self) -> None:
        before = self._siblings()
        self.fixture.stamp.pop("checkpoint_sha256")
        self.refuse()
        self.assertFalse(self.fixture.out_dir.exists())
        self.assertEqual(self._siblings(), before)

    def test_a_failure_injected_DURING_the_write_leaves_nothing_behind(self) -> None:
        """The atomicity claim, injected rather than argued.

        "Validate everything, then rename" is a statement about code order, and a later edit
        can move a write above a check without any test noticing. So this fails the copy of the
        SECOND shard -- after the first has already been written into staging -- and requires
        that the destination still does not exist and that no staging directory survives.
        """
        before = self._siblings()
        calls: list[Path] = []

        def explode(source: Path, destination: Path) -> None:
            calls.append(destination)
            if len(calls) == 1:
                destination.write_bytes(source.read_bytes())
                return
            raise OSError(28, "No space left on device")

        with patch.object(bank, "_copy_file", explode):
            with self.assertRaises(bank.BankRefusal) as caught:
                self.fixture.bank()

        self.assertIn(bank.WRITE_FAILED, caught.exception.codes)
        self.assertIn("No space left on device", str(caught.exception))
        self.assertEqual(len(calls), 2, "the injection did not reach the second shard")
        self.assertFalse(self.fixture.out_dir.exists(), "a partial artifact was left behind")
        self.assertEqual(self._siblings(), before, "a staging directory survived the failure")

    def test_an_overwrite_that_fails_leaves_the_banked_artifact_in_place(self) -> None:
        """The displaced artifact is moved aside only AFTER staging is complete."""
        self.fixture.bank()
        self.fixture.stamp["sims"] = 4096
        with patch.object(bank, "_copy_file", side_effect=OSError(5, "I/O error")):
            with self.assertRaises(bank.BankRefusal):
                self.fixture.bank(overwrite=True)
        provenance = json.loads(
            (self.fixture.out_dir / bank.PROVENANCE_FILENAME).read_text(encoding="utf-8")
        )
        self.assertEqual(provenance["sims"], 2048)
        self.assertEqual(self._siblings(), [CAMPAIGN_ID])


# ----------------------------------------------------------------------------------------------
# CLI surface -- importable as a library AND runnable, with a non-zero exit on refusal
# ----------------------------------------------------------------------------------------------
class CommandLineTests(_FixtureCase):
    def _run(self, *argv: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(_SCRIPT), *argv],
            capture_output=True, text=True, check=False,
        )

    def _write_stamp(self, stamp: dict[str, Any] | None = None) -> Path:
        path = self.fixture.root / "stamp.json"
        path.write_text(
            json.dumps(self.fixture.stamp if stamp is None else stamp), encoding="utf-8"
        )
        return path

    def test_the_cli_banks_a_complete_cell_and_exits_zero(self) -> None:
        stamp = self._write_stamp()
        done = self._run(
            "--stamp", str(stamp),
            "--source-dir", str(self.fixture.source_dir),
            "--out-dir", str(self.fixture.out_dir),
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("banked: 2 files", done.stdout)
        self.assertTrue((self.fixture.out_dir / bank.PROVENANCE_FILENAME).is_file())

    def test_the_cli_exits_nonzero_and_names_the_missing_input(self) -> None:
        self.fixture.stamp.pop("checkpoint_sha256")
        stamp = self._write_stamp()
        done = self._run(
            "--stamp", str(stamp),
            "--source-dir", str(self.fixture.source_dir),
            "--out-dir", str(self.fixture.out_dir),
        )
        self.assertEqual(done.returncode, bank.REFUSAL_EXIT)
        self.assertIn(bank.MISSING_REQUIRED_FIELD, done.stderr)
        self.assertIn("'checkpoint_sha256' is ABSENT", done.stderr)
        self.assertEqual(done.stdout, "")
        self.assertFalse(self.fixture.out_dir.exists())

    def test_a_stamp_that_is_not_json_is_refused_not_traced(self) -> None:
        path = self.fixture.root / "stamp.json"
        path.write_text("{not json", encoding="utf-8")
        done = self._run(
            "--stamp", str(path),
            "--source-dir", str(self.fixture.source_dir),
            "--out-dir", str(self.fixture.out_dir),
        )
        self.assertEqual(done.returncode, bank.REFUSAL_EXIT)
        self.assertIn(bank.STAMP_NOT_JSON, done.stderr)
        self.assertNotIn("Traceback", done.stderr)

    def test_the_stamp_can_arrive_on_stdin(self) -> None:
        done = subprocess.run(
            [sys.executable, str(_SCRIPT),
             "--stamp", "-",
             "--source-dir", str(self.fixture.source_dir),
             "--out-dir", str(self.fixture.out_dir)],
            input=json.dumps(self.fixture.stamp),
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_the_required_field_set_is_printable_for_launchers_and_reviewers(self) -> None:
        done = self._run("--print-required-fields", ARBITER_KIND)
        self.assertEqual(done.returncode, 0, done.stderr)
        for name in bank.ARTIFACT_KINDS[ARBITER_KIND].field_names():
            self.assertIn(name, done.stdout)

    def test_printing_an_unknown_kind_is_a_refusal(self) -> None:
        done = self._run("--print-required-fields", "campaign.v99")
        self.assertEqual(done.returncode, bank.REFUSAL_EXIT)
        self.assertIn(bank.UNKNOWN_ARTIFACT_KIND, done.stderr)

    def test_a_launcher_can_import_it_with_nothing_but_the_standard_library(self) -> None:
        """The launchers import this module; they must not need the project installed.

        `-S` is the load-bearing flag, and it was added after this test PASSED a deliberately
        injected `import numpy`: with `-I` alone the interpreter still imports its own
        site-packages, so in a venv that has numpy the check was true by construction. With
        `-I -S` no site directory is on the path at all, so any non-stdlib import in the tool
        fails here instead of in a cluster launcher. Kill-confirmed by re-injecting the import.
        """
        done = subprocess.run(
            [sys.executable, "-I", "-S", "-c",
             "import sys; sys.path.insert(0, sys.argv[1]); "
             "import bank_campaign_artifacts as b; "
             "print(b.PROVENANCE_SCHEMA, len(b.ARTIFACT_KINDS))",
             str(_SCRIPT.parent)],
            capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}, check=False,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(
            done.stdout.strip(), f"{bank.PROVENANCE_SCHEMA} {len(bank.ARTIFACT_KINDS)}")


class PublicRepoHygieneTests(unittest.TestCase):
    """The tool ships in the public repo; nothing about any deployment may ride along.

    Scoped by VALUE, not by phrasing: the needles are the literal strings that must never
    appear. They are assembled from halves so that this file is not itself the only match --
    a guard whose sole hit is its own pattern list covers nothing.
    """

    # Structural PATTERNS, not a denylist of real values. The first version of this guard listed
    # the vendor, the region, the PVC name and a real node-pool id, split across `+` -- which
    # both narrowed the guard to the instances someone remembered AND put those instances in the
    # public repo in reconstructable form. A pattern catches the class without naming any
    # instance, so nothing about a deployment is committed even by the check.
    #
    # A few patterns are written with a one-character class (`/sh[a]red`) purely so the file does
    # not contain the literal it searches for. Same regex, no self-match.
    FORBIDDEN_PATTERNS = (
        (r"(?<![\w./])/sh[a]red(?:/|\b)", "the shared filesystem mount root"),
        (r"\.compute\.intern[a]l\b", "internal node DNS suffix"),
        (r"\bnp-[0-9a-f]{8}\b", "node-pool identifier"),
        (r"\b[a-z]{2}-[a-z]{3,}[0-9]-[a-z]\b", "cloud region/zone string"),
        (r"\bkubect[l]\b|\bkubeconfi[g]\b", "kube CLI or its config"),
        (r"--namespac[e]\b|--contex[t]\b", "cluster-selecting flags"),
        (r"\b(?:registry|ccr)\.[a-z0-9.-]+\.(?:com|io|net)\b", "registry hostname"),
        (r"\b[a-z][a-z0-9-]*-nf[s]\b", "NFS volume / PVC name"),
        (r"\b(?:10|127)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "RFC1918 / loopback address"),
        (r"\b192\.168\.\d{1,3}\.\d{1,3}\b", "RFC1918 address"),
        (r"\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b", "RFC1918 address"),
    )

    #: Synthetic strings that match the patterns above and name nothing real. Assembled from
    #: fragments for the same reason the patterns are: so the file never contains them literally.
    SYNTHETIC_HITS = (
        "a harvest under " + "/sh" + "ared/some-campaign",
        "worker-3" + ".compute." + "internal",
        "node pool " + "np-" + "0123abcd" + "-7",
        "zone " + "xx-" + "example1" + "-a",
        "run " + "kube" + "ctl get pods",
        "pass " + "--name" + "space to select",
        "image at " + "registry." + "example.com/x",
        "volume " + "scratch-" + "nfs",
        "endpoint " + "10." + "0.0.1",
        "endpoint " + "192." + "168.1.1",
        "endpoint " + "172." + "16.0.1",
    )

    def test_neither_the_tool_nor_this_suite_names_a_deployment(self) -> None:
        for path in (_SCRIPT, Path(__file__).resolve()):
            text = path.read_text(encoding="utf-8")
            for pattern, what in self.FORBIDDEN_PATTERNS:
                with self.subTest(path=path.name, what=what):
                    found = re.search(pattern, text)
                    self.assertIsNone(
                        found, f"{path.name} contains {what}: {found.group(0) if found else ''}"
                    )

    def test_every_pattern_matches_a_synthetic_example_of_what_it_claims_to_catch(self) -> None:
        """The demonstrated failing input for the hygiene guard itself.

        Without this, a typo in any pattern leaves it permanently green -- the shape this repo
        has already found three times. Each pattern must match at least one synthetic hit, and
        no pattern may match ordinary prose.
        """
        for pattern, what in self.FORBIDDEN_PATTERNS:
            with self.subTest(what=what):
                self.assertTrue(
                    any(re.search(pattern, sample) for sample in self.SYNTHETIC_HITS),
                    f"the pattern for {what} matches none of the synthetic examples",
                )
                self.assertIsNone(re.search(pattern, "a clean line of ordinary code"))

    def test_the_tool_hardcodes_no_absolute_paths(self) -> None:
        """Paths arrive as arguments. A default would become somebody's production path."""
        source = _SCRIPT.read_text(encoding="utf-8").splitlines()
        offenders = [
            line for line in source
            if ('"/' in line or "'/" in line) and not line.lstrip().startswith("#")
        ]
        self.assertEqual(offenders, [])


# ----------------------------------------------------------------------------------------------
# The six inputs an independent review turned into a WRITTEN artifact
# ----------------------------------------------------------------------------------------------
class ReviewFoundTheseWroteAnArtifactTests(_FixtureCase):
    """Regressions pinned by the exact input that produced the wrong artifact.

    An adversarial review of the first version found four inputs that banked something which
    should have been refused, and two more that raised a bare exception instead of a refusal.
    Each test below is that review's reproduction, kept as the demonstrated failing input.
    """

    def _shasum(self) -> subprocess.CompletedProcess[str]:
        tool = shutil.which("sha256sum") or shutil.which("shasum")
        if tool is None:  # pragma: no cover - environment dependent
            self.skipTest("no sha256sum/shasum on PATH")
        argv = ([tool, "-a", "256", "-c"] if tool.endswith("shasum") else [tool, "-c"])
        return subprocess.run(
            argv + [bank.SHA256SUMS_FILENAME],
            cwd=self.fixture.out_dir, capture_output=True, text=True, check=False,
        )

    def _add_shard(self, name: str, payload: bytes) -> None:
        path = self.fixture.source_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        self.fixture.stamp["shards"].append({
            "file": name,
            "sha256": _sha256_bytes(payload),
            "exit_status": 0,
            "pod": "bank-test-pod-x",
            "node": "node-x.example.invalid",
        })

    # BLOCKING 1
    def test_a_shard_named_like_the_stamp_is_refused_not_overwritten(self) -> None:
        """It used to be copied in, then overwritten by the stamp written on top of it.

        The banked directory then claimed a sha256 for `PROVENANCE.json` whose bytes it had
        destroyed, and failed its own `shasum -c` with `PROVENANCE.json: FAILED`. Highly
        reachable: Phase 0 item 1 means pointing --source-dir at directories that already
        contain both of these names.
        """
        for reserved in (bank.PROVENANCE_FILENAME, bank.SHA256SUMS_FILENAME):
            with self.subTest(reserved=reserved):
                fixture = _Fixture(Path(tempfile.mkdtemp()))
                payload = b'{"i am": "a shard, not a stamp"}'
                (fixture.source_dir / reserved).write_bytes(payload)
                fixture.stamp["shards"].append({
                    "file": reserved, "sha256": _sha256_bytes(payload), "exit_status": 0,
                    "pod": "bank-test-pod-x", "node": "node-x.example.invalid",
                })
                with self.assertRaises(bank.BankRefusal) as caught:
                    fixture.bank()
                self.assertIn(bank.RESERVED_SHARD_NAME, caught.exception.codes)
                self.assertFalse(fixture.out_dir.exists())

    def test_a_reserved_name_is_caught_case_insensitively(self) -> None:
        """A case-insensitive filesystem collides on `provenance.json` too."""
        payload = b"{}"
        (self.fixture.source_dir / "provenance.json").write_bytes(payload)
        self.fixture.stamp["shards"].append({
            "file": "provenance.json", "sha256": _sha256_bytes(payload), "exit_status": 0,
            "pod": "bank-test-pod-x", "node": "node-x.example.invalid",
        })
        self.assertRefusedWith(bank.RESERVED_SHARD_NAME, contains="collides with a name")

    # BLOCKING 2
    def test_one_file_declared_under_two_spellings_is_refused(self) -> None:
        """`results/./shard0.json` evaded a raw-string duplicate check.

        The artifact then claimed 2 shards with 1 file banked, and `shasum -c` PASSED both
        lines because both paths resolve to the same bytes -- so nothing downstream could catch
        a pooled analysis counting the shard twice and shrinking the apparent SE.
        """
        for spelling, expected in (
            ("results/./shard0.json", bank.NON_CANONICAL_SHARD_PATH),
            ("results//shard0.json", bank.NON_CANONICAL_SHARD_PATH),
            ("./results/shard0.json", bank.NON_CANONICAL_SHARD_PATH),
        ):
            with self.subTest(spelling=spelling):
                fixture = _Fixture(Path(tempfile.mkdtemp()))
                fixture.stamp["shards"].append(
                    dict(fixture.stamp["shards"][0], file=spelling)
                )
                with self.assertRaises(bank.BankRefusal) as caught:
                    fixture.bank()
                self.assertIn(expected, caught.exception.codes)
                self.assertFalse(fixture.out_dir.exists())

    def test_one_file_declared_under_two_cases_is_refused_on_this_filesystem(self) -> None:
        """On a case-insensitive filesystem `Results/shard0.json` is the same file."""
        probe = self.fixture.source_dir / "results" / "SHARD0.json"
        case_insensitive = probe.is_file()
        self.fixture.stamp["shards"].append(
            dict(self.fixture.stamp["shards"][0], file="results/SHARD0.json")
        )
        refusal = self.refuse()
        if case_insensitive:
            self.assertIn(bank.DUPLICATE_SHARD_FILE, refusal.codes)
        else:  # pragma: no cover - depends on the filesystem the suite runs on
            self.assertIn(bank.SHARD_FILE_ABSENT, refusal.codes)

    def test_one_file_hardlinked_under_two_names_is_refused(self) -> None:
        """The case no path comparison can catch, so it needs the inode.

        A hardlink gives one file two unrelated names: `Path.resolve()` cannot connect them,
        the copy duplicates the content into two banked files, and both SHA256SUMS lines
        verify -- so the declared shard count exceeds the distinct evidence with nothing
        downstream able to notice. This test is why the duplicate key is (device, inode) and
        not a resolved path; the resolved-path version passed a mutation that deleted it.
        """
        original = self.fixture.source_dir / self.fixture.stamp["shards"][0]["file"]
        link = self.fixture.source_dir / "results" / "shard0-copy.json"
        os.link(original, link)
        self.assertNotEqual(str(original.resolve()), str(link.resolve()))  # paths cannot tell
        self.fixture.stamp["shards"].append(
            dict(self.fixture.stamp["shards"][0], file="results/shard0-copy.json")
        )
        message = self.assertRefusedWith(bank.DUPLICATE_SHARD_FILE, contains="hardlink")
        self.assertIn("shards[0]", message)

    def test_the_duplicate_refusal_names_the_record_it_collides_with(self) -> None:
        self.fixture.stamp["shards"].append(dict(self.fixture.stamp["shards"][0]))
        message = self.assertRefusedWith(bank.DUPLICATE_SHARD_FILE)
        self.assertIn("shards[0]", message)
        # The old message claimed "SHA256SUMS would carry only one of them", which the review
        # falsified: it carried both. The message now says what actually goes wrong.
        self.assertNotIn("would carry only one of them", message)
        self.assertIn("shrinking the apparent standard error", message)

    # BLOCKING 3
    def test_an_interrupt_in_the_overwrite_window_keeps_the_campaign_directory(self) -> None:
        """The rollback used to live in `except OSError`, so Ctrl-C skipped it.

        The store was then left holding only `<campaign-id>.replaced-<ts>` and nothing at the
        campaign id -- the scratch-directory loss with a timestamp suffix. Ctrl-C is the
        likeliest interruption of a long copy, so it is the case that must roll back.
        """
        for injected in (KeyboardInterrupt, SystemExit):
            with self.subTest(injected=injected.__name__):
                fixture = _Fixture(Path(tempfile.mkdtemp()))
                fixture.bank()
                fixture.stamp["sims"] = 4096
                real_rename = bank.os.rename

                def interrupt_in_the_window(src, dst, _exc=injected):
                    real_rename(src, dst)
                    if str(src).endswith(CAMPAIGN_ID):  # the displacement just happened
                        raise _exc("interrupted between displace and install")

                with patch.object(bank.os, "rename", interrupt_in_the_window):
                    with self.assertRaises(injected):
                        fixture.bank(overwrite=True)

                self.assertTrue(
                    fixture.out_dir.is_dir(),
                    "the store lost the directory its campaign id addresses",
                )
                provenance = json.loads(
                    (fixture.out_dir / bank.PROVENANCE_FILENAME).read_text(encoding="utf-8")
                )
                self.assertEqual(provenance["sims"], 2048, "the original artifact was not restored")
                self.assertEqual(
                    sorted(p.name for p in fixture.store_dir.iterdir()), [CAMPAIGN_ID],
                    "a displaced copy or staging directory was left behind",
                )

    # BLOCKING 4
    def test_a_file_sitting_at_the_destination_is_a_refusal_not_a_traceback(self) -> None:
        """`iterdir()` on a file raised NotADirectoryError and the CLI exited 1."""
        self.fixture.out_dir.parent.mkdir(parents=True, exist_ok=True)
        self.fixture.out_dir.write_text("a leftover file at the campaign id's path")
        refusal = self.refuse()
        self.assertIn(bank.DESTINATION_NOT_A_DIRECTORY, refusal.codes)
        self.assertEqual(
            self.fixture.out_dir.read_text(), "a leftover file at the campaign id's path"
        )

    def test_the_cli_exits_three_when_the_destination_is_a_file(self) -> None:
        self.fixture.out_dir.parent.mkdir(parents=True, exist_ok=True)
        self.fixture.out_dir.write_text("leftover")
        stamp = self.fixture.root / "stamp.json"
        stamp.write_text(json.dumps(self.fixture.stamp), encoding="utf-8")
        done = subprocess.run(
            [sys.executable, str(_SCRIPT), "--stamp", str(stamp),
             "--source-dir", str(self.fixture.source_dir),
             "--out-dir", str(self.fixture.out_dir)],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(done.returncode, bank.REFUSAL_EXIT)
        self.assertIn(bank.DESTINATION_NOT_A_DIRECTORY, done.stderr)
        self.assertNotIn("Traceback", done.stderr)

    # BLOCKING 5
    def test_a_non_finite_number_anywhere_in_the_stamp_is_refused(self) -> None:
        """A NaN used to be written verbatim, making PROVENANCE.json invalid JSON.

        `JSON.parse` then rejects the whole file and `jq` yields `null` at exit 0 -- a number
        silently becoming the "this does not apply" absence this tool exists to distinguish.
        """
        for label, value in (
            ("nan", float("nan")), ("inf", float("inf")), ("-inf", float("-inf")),
        ):
            with self.subTest(value=label):
                fixture = _Fixture(Path(tempfile.mkdtemp()))
                fixture.stamp["harvest_mean_gap"] = value
                with self.assertRaises(bank.BankRefusal) as caught:
                    fixture.bank()
                self.assertIn(bank.UNSERIALIZABLE_PROVENANCE, caught.exception.codes)
                self.assertIn(
                    "harvest_mean_gap",
                    caught.exception.fields_for(bank.UNSERIALIZABLE_PROVENANCE),
                )
                self.assertFalse(fixture.out_dir.exists())

    def test_a_non_finite_number_nested_deep_in_the_stamp_is_named_by_path(self) -> None:
        self.fixture.stamp["analysis"] = {"buckets": [{"mean": 0.1}, {"mean": float("nan")}]}
        refusal = self.refuse()
        self.assertEqual(
            refusal.fields_for(bank.UNSERIALIZABLE_PROVENANCE),
            ("analysis.buckets[1].mean",),
        )

    def test_a_non_finite_rollout_fallback_fraction_is_refused_twice_over(self) -> None:
        """The declared-field check must not accept a NaN just because NaN comparisons are False."""
        self.fixture.stamp["rollout_fallback_fraction"] = float("nan")
        refusal = self.refuse()
        self.assertIn(bank.INVALID_FIELD_VALUE, refusal.codes)
        self.assertIn(bank.UNSERIALIZABLE_PROVENANCE, refusal.codes)

    def test_every_written_provenance_survives_a_strict_json_reader(self) -> None:
        """The structural half of the same guard: allow_nan=False on the writer."""
        text = self.fixture.bank().provenance_path.read_text(encoding="utf-8")

        def reject(constant: str) -> Any:
            raise ValueError(f"non-JSON constant {constant}")

        json.loads(text, parse_constant=reject)  # would raise on NaN/Infinity/-Infinity

    def test_the_writer_itself_refuses_a_non_finite_number(self) -> None:
        """`allow_nan=False` needs its own failing input, or it is untested belt-and-braces.

        The validation scan above refuses first, so no end-to-end input can reach this line --
        and a mutation reverting it to `allow_nan=True` left the whole suite green. Exercised
        directly, it reddens.
        """
        with self.assertRaises(ValueError):
            bank._provenance_json({"schema": "x", "mean": float("nan")})
        with self.assertRaises(ValueError):
            bank._provenance_json({"schema": "x", "rate": float("inf")})
        # And it still writes ordinary numbers.
        self.assertIn('"mean": 0.5', bank._provenance_json({"mean": 0.5}))

    # BLOCKING 6
    def test_a_source_mutated_between_hashing_and_copying_is_refused(self) -> None:
        """Hash-then-copy banked a directory that failed its own SHA256SUMS.

        The realistic trigger is banking while a straggler shard is still flushing. The fix
        re-hashes the STAGED bytes before the rename, so self-consistency is an invariant of
        the write rather than an assumption that the source held still.
        """
        real_copy = bank._copy_file

        def mutate_then_copy(source: Path, destination: Path) -> None:
            if source.name == "shard1.json":
                source.write_bytes(source.read_bytes() + b"    APPENDED BY A CONCURRENT WRITER")
            real_copy(source, destination)

        with patch.object(bank, "_copy_file", mutate_then_copy):
            refusal = self.refuse()
        self.assertIn(bank.STAGED_COPY_MISMATCH, refusal.codes)
        self.assertIn("would have failed its own SHA256SUMS",
                      refusal.messages_for(bank.STAGED_COPY_MISMATCH)[0])
        self.assertFalse(self.fixture.out_dir.exists())
        self.assertEqual(sorted(p.name for p in self.fixture.store_dir.iterdir()), [])

    def test_a_truncation_between_hashing_and_copying_is_refused_too(self) -> None:
        """Not only growth: a shard that shrinks must not bank either."""
        real_copy = bank._copy_file

        def truncate_then_copy(source: Path, destination: Path) -> None:
            if source.name == "shard0.json":
                source.write_bytes(b"")
            real_copy(source, destination)

        with patch.object(bank, "_copy_file", truncate_then_copy):
            refusal = self.refuse()
        self.assertIn(bank.STAGED_COPY_MISMATCH, refusal.codes)
        self.assertFalse(self.fixture.out_dir.exists())

    def test_the_banked_directory_verifies_against_its_own_sha256sums(self) -> None:
        """The invariant all of the above exist to protect, asserted with the external tool."""
        self.fixture.bank()
        done = self._shasum()
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertNotIn("FAILED", done.stdout)


# ----------------------------------------------------------------------------------------------
# The smaller findings from the same review, each with its own failing input
# ----------------------------------------------------------------------------------------------
class ReviewNitTests(_FixtureCase):
    def test_a_zero_width_only_value_does_not_pass_as_a_required_string(self) -> None:
        """U+200B, U+FEFF and U+00AD all survive str.strip(), so strip() is not an empty test."""
        for label, blank in (
            ("U+200B zero-width space", "​"),
            ("U+FEFF BOM", "﻿"),
            ("U+00AD soft hyphen", "­"),
            ("mixed blanks", " ​\t­ "),
        ):
            with self.subTest(blank=label):
                fixture = _Fixture(Path(tempfile.mkdtemp()))
                fixture.stamp["cell"] = blank
                with self.assertRaises(bank.BankRefusal) as caught:
                    fixture.bank()
                self.assertIn(bank.INVALID_FIELD_VALUE, caught.exception.codes)
                self.assertIn(
                    "zero-width or non-printing",
                    caught.exception.messages_for(bank.INVALID_FIELD_VALUE)[0],
                )

    def test_a_zero_width_only_value_is_refused_in_every_required_string_field(self) -> None:
        for name in ("checkpoint", "arms", "image", "launcher", "seed_band"):
            with self.subTest(field=name):
                fixture = _Fixture(Path(tempfile.mkdtemp()))
                fixture.stamp[name] = "​"
                with self.assertRaises(bank.BankRefusal) as caught:
                    fixture.bank()
                self.assertIn(bank.INVALID_FIELD_VALUE, caught.exception.codes)

    def test_a_derived_key_is_refused_however_it_is_spelled(self) -> None:
        """`Estimand` and `"estimand "` used to bank a forged caveat beside the real one."""
        for spelling in ("Estimand", "ESTIMAND", "estimand ", " estimand", "PER_FILE"):
            with self.subTest(spelling=spelling):
                fixture = _Fixture(Path(tempfile.mkdtemp()))
                fixture.stamp[spelling] = {"prices": "whatever the author preferred"}
                with self.assertRaises(bank.BankRefusal) as caught:
                    fixture.bank()
                self.assertIn(bank.DERIVED_FIELD_SUPPLIED, caught.exception.codes)

    def test_a_derived_key_forged_inside_a_shard_record_is_refused(self) -> None:
        """The guard used to be a top-level set intersection, so this banked."""
        self.fixture.stamp["shards"][0]["estimand"] = {"prices": "a forgery in the evidence"}
        message = self.assertRefusedWith(bank.DERIVED_FIELD_SUPPLIED, contains="shards[0]")
        self.assertIn("two answers to what the numbers mean", message)

    def test_a_symlinked_shard_is_refused(self) -> None:
        """`path.is_file()` and `shutil.copy2` both follow links, so this banked outside bytes."""
        outside = self.fixture.root / "outside.json"
        outside.write_bytes(b"CONTENT FROM OUTSIDE THE HARVEST")
        link = self.fixture.source_dir / "results" / "linked.json"
        link.symlink_to(outside)
        self.fixture.stamp["shards"].append({
            "file": "results/linked.json",
            "sha256": _sha256_bytes(b"CONTENT FROM OUTSIDE THE HARVEST"),
            "exit_status": 0, "pod": "bank-test-pod-x", "node": "node-x.example.invalid",
        })
        self.assertRefusedWith(bank.SYMLINKED_SHARD, contains="symlink")

    def test_a_shard_reached_through_a_symlinked_directory_is_refused(self) -> None:
        outside_dir = self.fixture.root / "elsewhere"
        outside_dir.mkdir()
        (outside_dir / "shard.json").write_bytes(b"OUTSIDE")
        (self.fixture.source_dir / "linkdir").symlink_to(outside_dir)
        self.fixture.stamp["shards"].append({
            "file": "linkdir/shard.json", "sha256": _sha256_bytes(b"OUTSIDE"),
            "exit_status": 0, "pod": "bank-test-pod-x", "node": "node-x.example.invalid",
        })
        self.assertRefusedWith(bank.SYMLINKED_SHARD)

    def test_a_zero_byte_shard_is_refused_unless_the_emptiness_is_the_finding(self) -> None:
        """Every other guard reads True: the hash of nothing is honestly recorded."""
        (self.fixture.source_dir / "results" / "empty.json").write_bytes(b"")
        self.fixture.stamp["shards"].append({
            "file": "results/empty.json", "sha256": _sha256_bytes(b""),
            "exit_status": 0, "pod": "bank-test-pod-x", "node": "node-x.example.invalid",
        })
        self.assertRefusedWith(bank.EMPTY_SHARD_FILE, contains="ZERO bytes")

    def test_an_acknowledged_zero_byte_shard_banks(self) -> None:
        (self.fixture.source_dir / "results" / "empty.json").write_bytes(b"")
        self.fixture.stamp["shards"].append({
            "file": "results/empty.json", "sha256": _sha256_bytes(b""),
            "exit_status": 0, "pod": "bank-test-pod-x", "node": "node-x.example.invalid",
        })
        result = self.fixture.bank(allow_empty_shard=True)
        provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))
        self.assertIn(0, [entry["bytes"] for entry in provenance["per_file"]])

    def test_a_signal_killed_shard_can_be_banked_with_acknowledgement(self) -> None:
        """Python reports SIGKILL as -9. The OOM case is where banking-with-ack matters most."""
        self.fixture.stamp["shards"][1]["exit_status"] = -9
        refusal = self.refuse()
        self.assertIn(bank.UNACKNOWLEDGED_SHARD_FAILURE, refusal.codes)
        self.assertIn("exited -9", refusal.messages_for(bank.UNACKNOWLEDGED_SHARD_FAILURE)[0])

        result = self.fixture.bank(allow_nonzero_shard_exit=True)
        provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))
        self.assertEqual(provenance["nonzero_exit_shards_acknowledged"], ["results/shard1.json"])
        self.assertEqual(provenance["shards"][1]["exit_status"], -9)

    def test_a_declared_byte_count_is_verified_rather_than_trusted(self) -> None:
        self.fixture.stamp["shards"][0]["bytes"] = 999999
        self.assertRefusedWith(bank.SHARD_BYTES_MISMATCH, contains="999999")

    def test_a_correct_declared_byte_count_is_accepted(self) -> None:
        path = self.fixture.source_dir / self.fixture.stamp["shards"][0]["file"]
        self.fixture.stamp["shards"][0]["bytes"] = path.stat().st_size
        self.assertEqual(self.fixture.bank().status, "banked")

    def test_a_duplicate_key_in_the_stamp_json_is_refused(self) -> None:
        """`{"rollouts": 0, "rollouts": 64}` parses to 64 and the discarded answer is invisible."""
        raw = json.dumps(self.fixture.stamp)
        doctored = raw.replace('"rollouts": 64', '"rollouts": 0, "rollouts": 64', 1)
        self.assertEqual(json.loads(doctored)["rollouts"], 64)  # what the default parser does
        with self.assertRaises(bank.BankRefusal) as caught:
            bank._load_stamp_json(doctored)
        self.assertIn(bank.DUPLICATE_STAMP_KEY, caught.exception.codes)
        self.assertEqual(caught.exception.fields_for(bank.DUPLICATE_STAMP_KEY), ("rollouts",))

    def test_the_cli_refuses_a_duplicate_key_without_a_traceback(self) -> None:
        raw = json.dumps(self.fixture.stamp).replace(
            '"sims": 2048', '"sims": 1, "sims": 2048', 1)
        path = self.fixture.root / "stamp.json"
        path.write_text(raw, encoding="utf-8")
        done = subprocess.run(
            [sys.executable, str(_SCRIPT), "--stamp", str(path),
             "--source-dir", str(self.fixture.source_dir),
             "--out-dir", str(self.fixture.out_dir)],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(done.returncode, bank.REFUSAL_EXIT)
        self.assertIn(bank.DUPLICATE_STAMP_KEY, done.stderr)
        self.assertNotIn("Traceback", done.stderr)

    def test_an_undeclared_file_beside_a_banked_artifact_is_not_blessed_as_unchanged(self) -> None:
        """`unchanged` used to mean "everything declared is here", not "nothing else is"."""
        self.fixture.bank()
        (self.fixture.out_dir / "NOTES.md").write_text("dumped beside the bank", encoding="utf-8")
        refusal = self.refuse()
        self.assertIn(bank.DESTINATION_HOLDS_DIFFERENT_ARTIFACT, refusal.codes)
        self.assertTrue((self.fixture.out_dir / "NOTES.md").is_file())

    def test_an_undeclared_shard_beside_a_banked_artifact_is_not_blessed_either(self) -> None:
        self.fixture.bank()
        (self.fixture.out_dir / "results" / "shard-NOT-IN-STAMP.json").write_text("{}")
        self.assertIn(
            bank.DESTINATION_HOLDS_DIFFERENT_ARTIFACT, self.refuse().codes
        )

    def test_a_refusal_names_the_field_machine_readably(self) -> None:
        """A launcher branching on which input was wrong must not have to parse prose."""
        self.fixture.stamp.pop("checkpoint_sha256")
        self.fixture.stamp["shards"][0].pop("pod")
        refusal = self.refuse()
        self.assertEqual(
            sorted(refusal.fields_for(bank.MISSING_REQUIRED_FIELD)),
            ["checkpoint_sha256", "shards[0].pod"],
        )

    def test_a_stale_staging_directory_is_reported_rather_than_silently_left(self) -> None:
        """SIGKILL cannot be caught, so debris is possible; invisible debris is not acceptable."""
        self.fixture.store_dir.mkdir(parents=True, exist_ok=True)
        stale = self.fixture.store_dir / f"{bank.STAGING_PREFIX}{CAMPAIGN_ID}-killed"
        stale.mkdir()
        result = self.fixture.bank()
        self.assertEqual(result.stale_staging, (stale,))
        self.assertTrue(stale.is_dir(), "the tool deleted evidence instead of reporting it")

    def test_the_help_text_states_the_exit_code_contract(self) -> None:
        done = subprocess.run(
            [sys.executable, str(_SCRIPT), "--help"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(done.returncode, 0)
        self.assertIn(f"{bank.REFUSAL_EXIT} = REFUSED", done.stdout)
        self.assertIn("2 = bad invocation", done.stdout)


# ----------------------------------------------------------------------------------------------
# The staging (INPUTS) kind, and the one-schema-string-one-shape property
# ----------------------------------------------------------------------------------------------
STAGING_KIND = "staging.v1"
STAGING_CAMPAIGN_ID = "bank-test-staging-20260817"


class StagingKindTests(unittest.TestCase):
    """An INPUTS cell: staged bytes, no games, no seeds, no results.

    The kind exists because the store already holds a hand-written staging stamp whose shape is
    disjoint from the results shape -- no shards, no seeds, no search budget -- and one schema
    string naming two disjoint layouts is precisely what stops a reader telling them apart.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.source_dir = self.root / "staged"
        (self.source_dir / "artifacts").mkdir(parents=True)
        self.out_dir = self.root / "campaign-store" / STAGING_CAMPAIGN_ID
        files = []
        for name, payload in (
            ("artifacts/encoder_tables.json", b'{"schema_version": "placeholder"}'),
            ("staged-models/model_ts_cpu.pt", b"placeholder torchscript bytes"),
        ):
            path = self.source_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            files.append({"file": name, "sha256": _sha256_bytes(payload)})
        self.stamp: dict[str, Any] = {
            "artifact_kind": STAGING_KIND,
            "campaign_id": STAGING_CAMPAIGN_ID,
            "cell": "instrument-2 artifact staging (placeholder)",
            "checkpoint": PLACEHOLDER_CHECKPOINT,
            "checkpoint_sha256": _sha256_bytes(b"placeholder-weights"),
            "staged_at_utc": "2026-08-17T10:30:00Z",
            "staged_from": "/lineages/lineage-a/iteration-0001",
            "staging_pod": "bank-test-staging-pod",
            "staging_node": "node-s.example.invalid",
            "image_id": PLACEHOLDER_IMAGE,
            "no_results_banked_here": (
                "INPUTS only: no games, no win rate, no X, no Y. A strength number from this "
                "arm belongs in its own results cell with its own provenance."
            ),
            "staged_files": files,
        }

    def bank(self, **kwargs: Any) -> Any:
        return bank.bank_artifact(self.stamp, self.source_dir, self.out_dir, **kwargs)

    def test_an_inputs_cell_banks_and_carries_the_staging_schema_string(self) -> None:
        result = self.bank()
        provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))
        self.assertEqual(provenance["schema"], bank.STAGING_SCHEMA)
        self.assertNotEqual(provenance["schema"], bank.PROVENANCE_SCHEMA)
        self.assertEqual(provenance["artifact_kind"], STAGING_KIND)
        self.assertEqual(provenance["staged_file_count"], 2)
        self.assertEqual(len(provenance["per_file"]), 2)
        # No results vocabulary leaks into an inputs cell.
        for absent in ("shards", "shard_count", "estimand", "nonzero_exit_shards_acknowledged"):
            self.assertNotIn(absent, provenance)

    def test_the_two_shapes_never_share_one_schema_string(self) -> None:
        """The property the review asked for: tell them apart from `schema` alone."""
        results_schemas = {
            kind.schema for kind in bank.ARTIFACT_KINDS.values() if kind.files_field == "shards"
        }
        staging_schemas = {
            kind.schema for kind in bank.ARTIFACT_KINDS.values()
            if kind.files_field == "staged_files"
        }
        self.assertEqual(results_schemas, {bank.PROVENANCE_SCHEMA})
        self.assertEqual(staging_schemas, {bank.STAGING_SCHEMA})
        self.assertEqual(results_schemas & staging_schemas, set())

    def test_a_stamp_declaring_the_matching_schema_is_accepted_and_rewritten_once(self) -> None:
        """Agreeing with the tool is not forgery; the key is not duplicated in the output."""
        self.stamp["schema"] = bank.STAGING_SCHEMA
        text = self.bank().provenance_path.read_text(encoding="utf-8")
        self.assertEqual(text.count('"schema"'), 1)
        self.assertEqual(json.loads(text)["schema"], bank.STAGING_SCHEMA)

    def test_a_stamp_declaring_the_wrong_schema_is_refused(self) -> None:
        """This is the exact defect the review found in the store: one string, two shapes."""
        self.stamp["schema"] = bank.PROVENANCE_SCHEMA
        with self.assertRaises(bank.BankRefusal) as caught:
            self.bank()
        self.assertIn(bank.SCHEMA_MISMATCH, caught.exception.codes)
        self.assertIn(
            "two shapes behind one schema string",
            caught.exception.messages_for(bank.SCHEMA_MISMATCH)[0],
        )
        self.assertFalse(self.out_dir.exists())

    def test_an_inputs_cell_cannot_be_banked_without_its_own_required_fields(self) -> None:
        for spec in bank.ARTIFACT_KINDS[STAGING_KIND].required:
            if spec.name == "staged_files":
                continue
            with self.subTest(field=spec.name):
                stamp = {k: v for k, v in self.stamp.items() if k != spec.name}
                with self.assertRaises(bank.BankRefusal) as caught:
                    bank.bank_artifact(stamp, self.source_dir, self.out_dir)
                self.assertIn(bank.MISSING_REQUIRED_FIELD, caught.exception.codes)
                self.assertIn(spec.name, caught.exception.fields_for(bank.MISSING_REQUIRED_FIELD))

    def test_an_empty_staged_file_list_is_refused_and_names_that_field(self) -> None:
        self.stamp["staged_files"] = []
        with self.assertRaises(bank.BankRefusal) as caught:
            self.bank()
        self.assertIn(bank.EMPTY_SHARD_LIST, caught.exception.codes)
        self.assertIn("'staged_files' is an EMPTY list",
                      caught.exception.messages_for(bank.EMPTY_SHARD_LIST)[0])

    def test_the_staging_kind_does_not_require_results_fields(self) -> None:
        """Otherwise it would be the results kind wearing a different name."""
        declared = set(bank.ARTIFACT_KINDS[STAGING_KIND].field_names())
        for results_only in ("rollouts", "sims", "depth", "arms", "seeds", "seed_band", "shards"):
            self.assertNotIn(results_only, declared)

    def test_the_staging_kind_still_verifies_every_staged_sha256(self) -> None:
        target = self.source_dir / "artifacts" / "encoder_tables.json"
        target.write_bytes(b'{"schema_version": "edited after hashing"}')
        with self.assertRaises(bank.BankRefusal) as caught:
            self.bank()
        self.assertIn(bank.SHARD_SHA256_MISMATCH, caught.exception.codes)

    def test_a_staged_file_needs_no_exit_status(self) -> None:
        """Staging is one operation; a per-file exit status would be invented provenance."""
        self.assertNotIn(
            "exit_status",
            [spec.name for spec in bank.ARTIFACT_KINDS[STAGING_KIND].file_required],
        )
        self.assertEqual(self.bank().status, "banked")


if __name__ == "__main__":
    unittest.main()
