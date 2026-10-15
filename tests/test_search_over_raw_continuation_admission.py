"""Disposable authored roots and toy owners; no scientific collection."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_continuation_admission import (
    ContinuationAdmissionJournal, history_review_template, _inventory,
    fresh_history_review_template, initialize_fresh_namespace,
)
from pokezero.mcts_eval.search_over_raw_root_plan import OwnedContinuationBank
import test_search_over_raw_owned_stage as stage_fixture
import test_search_over_raw_outcome_run as outcome_fixture
import test_search_over_raw_bank_reuse as reuse_fixture
import test_search_over_raw_root_plan as root_fixture
import test_search_over_raw_reconciled as ledger_fixture
from pokezero.mcts_eval.search_over_raw_reconciled import ReconciledPhaseALedger, controller_input_hashes


class FreshLifecycleTests(unittest.TestCase):
    def setUp(self):
        # Static authored recipe files only, not an observed source/raw owner.
        self.base = ledger_fixture.ReconciledLedgerTest(); self.base.setUp()
        self.addCleanup(self.base.tearDown)
        self.outcome = outcome_fixture.OutcomeRunTests(); self.outcome.setUp()
        self.addCleanup(self.outcome.doCleanups)
        self.directory = self.base.root.resolve() / "prospective-ledger"
        self.review_path = self.base.root.resolve() / "fresh-review.json"
        from pokezero.mcts_eval.search_over_raw import SearchConfiguration
        from pokezero.mcts_eval.search_over_raw_raw_run import raw_runtime
        from pokezero.mcts_eval.search_over_raw_outcome_run import verify_outcome_runtime
        self.raw_sha = digest(raw_runtime(SearchConfiguration("raw", seconds=1.),
            hashlib.sha256(b"authored unused model").hexdigest(), "a"*64))
        self.continuation_sha = verify_outcome_runtime(self.outcome.runtime, self.outcome.config)
        self.contract = deepcopy(self.base.contract)
        self.review = fresh_history_review_template(self.contract, ledger_directory=self.directory,
            raw_runtime_sha256=self.raw_sha, continuation_runtime_sha256=self.continuation_sha)
        self.review.update(status="CLEAR_FRESH_DECLARED_NAMESPACE", reviewed_before_first_source_work=True)

    def initialize(self, *, occupied=None, alter_review=None):
        if alter_review: alter_review(self.review)
        self.review_path.write_text(json.dumps(self.review))
        self.contract["input_hashes"] = {**controller_input_hashes(),
            str(self.review_path): hashlib.sha256(self.review_path.read_bytes()).hexdigest()}
        self.ledger = ReconciledPhaseALedger.create(self.directory, self.contract)
        if occupied: (self.directory / occupied).mkdir()
        with patch("subprocess.Popen") as launch:
            result = initialize_fresh_namespace(self.ledger, history_review_path=self.review_path)
        launch.assert_not_called()
        self.assertFalse(result["collection_authorized"])

    def observed_plan(self, *, canonical=True):
        # Real order: review -> ledger -> namespace -> NEW authored archive/raw
        # owning process -> baseline. No pre-existing roots are rebound.
        fixture = root_fixture.RootPlanTests()
        original_archive = root_fixture.archive_fixture_module.ExplorationRootCrosswalkTests.setUp
        original_recipe = ledger_fixture.ReconciledLedgerTest.setUp
        original_supervisor = root_fixture._supervise_owned_job
        original_reader = root_fixture.RawSelectionRunValidator
        root_id = self.contract["panels"]["exploration"]["root_slots"][0]["root_id"]
        raw_directory = self.directory / "shared-raw-runs" / digest(dict(root_id=root_id))
        self.assertFalse(raw_directory.exists())
        def static_recipe(base):
            original_recipe(base)
            base.contract = deepcopy(self.contract)
        def authored_archive(archive):
            original_archive(archive)
            index = archive.source["roots"][0]["source_request_index"]
            archive.archive._private[index] = deepcopy(self.outcome.snapshot)
        def first_owner(job, **kwargs):
            self.assertTrue((self.directory / "continuation-admission" / "namespace.json").is_file())
            if canonical: kwargs["directory"] = raw_directory
            return original_supervisor(job, **kwargs)
        def first_reader(**kwargs):
            if canonical: kwargs["directory"] = raw_directory
            return original_reader(**kwargs)
        with patch.object(root_fixture.archive_fixture_module.ExplorationRootCrosswalkTests,
                          "setUp", authored_archive), patch.object(root_fixture,
                          "_supervise_owned_job", side_effect=first_owner), patch.object(
                          ledger_fixture.ReconciledLedgerTest, "setUp", static_recipe), patch.object(
                          root_fixture, "RawSelectionRunValidator", side_effect=first_reader):
            fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        return fixture.plan()

    def test_genuine_prospective_lifecycle_reserves_before_continuation(self):
        self.assertNotIn("root_identity", self.review)
        self.assertNotIn("contract_sha256", self.review)
        self.assertFalse(self.directory.exists())
        self.initialize()
        plan = self.observed_plan()
        journal = ContinuationAdmissionJournal.create(self.ledger, plan, history_review_path=self.review_path)
        with patch("subprocess.Popen") as launch:
            job, _ = journal.prepare_new_job(plan, config=self.outcome.config, runtime=self.outcome.runtime,
                _synthetic_job_class=outcome_fixture.SyntheticContinuationJob)
            reservation = journal.reserve(plan, job)
            journal.verify_reservation(plan, job, reservation)
        launch.assert_not_called()
        self.assertEqual(journal.consumed_actions(plan), (0,))
        self.assertEqual(reservation["reserved_cells"], 8)
        self.assertFalse(Path(reservation["run_directory"]).exists())
        self.assertFalse(reservation["execution_ready"])
        with self.assertRaises(FileExistsError):
            ContinuationAdmissionJournal.create(self.ledger, plan, history_review_path=self.review_path)
        self.assertEqual(journal.consumed_actions(plan), (0,))

    def test_occupied_namespace_is_not_fresh(self):
        with self.assertRaisesRegex(ValueError, "already occupied"):
            self.initialize(occupied="shared-raw-runs")
        self.assertFalse((self.directory / "continuation-admission").exists())

    def test_fresh_review_cannot_import_or_change_slot_or_exclusion(self):
        with self.assertRaisesRegex(ValueError, "no imports"):
            self.initialize(alter_review=lambda r: r["prior_attempts"].append({"directory":"/missing"}))
        self.assertFalse((self.directory / "continuation-admission").exists())

    def test_review_wrong_slot_mapping_rejected(self):
        with self.assertRaisesRegex(ValueError, "no imports"):
            self.initialize(alter_review=lambda r: r["exploration_slots"][0].update(source_seed=1))

    def test_boolean_integer_alias_in_review_rejected(self):
        with self.assertRaisesRegex(ValueError, "no imports"):
            self.initialize(alter_review=lambda r: r["exploration_slots"][0].update(source_seed=False))

    def test_integer_authority_flag_in_review_rejected(self):
        with self.assertRaisesRegex(ValueError, "no imports"):
            self.initialize(alter_review=lambda r: r.update(collection_authorized=0))

    def test_namespace_flag_alias_and_callback_drift_rejected(self):
        self.initialize(); plan = self.observed_plan()
        journal = ContinuationAdmissionJournal.create(self.ledger, plan, history_review_path=self.review_path)
        namespace = self.directory / "continuation-admission" / "namespace.json"
        claim = json.loads(namespace.read_text()); claim["collection_authorized"] = 0
        namespace.write_text(json.dumps(claim))
        with self.assertRaisesRegex(ValueError, "not initialized"):
            journal.consumed_actions(plan)

    def test_namespace_drift_during_reservation_callback_rejects_before_write(self):
        self.initialize(); plan = self.observed_plan()
        journal = ContinuationAdmissionJournal.create(self.ledger, plan, history_review_path=self.review_path)
        job, _ = plan.prepare_continuation_job(config=self.outcome.config, runtime=self.outcome.runtime,
            _synthetic_job_class=outcome_fixture.SyntheticContinuationJob)
        namespace = self.directory / "continuation-admission" / "namespace.json"
        original = plan.verify_prepared_job
        def drift(value):
            result = original(value)
            namespace.write_text(namespace.read_text()+"\n")
            return result
        with patch.object(plan, "verify_prepared_job", side_effect=drift):
            with self.assertRaisesRegex(ValueError, "durable admission bytes"):
                journal.reserve(plan, job)
        self.assertEqual(list((journal.directory / "attempts").iterdir()), [])

    def test_review_recipe_exclusions_cannot_be_rebound(self):
        self.contract["excluded_seeds"] = [9999998]
        self.contract["source_review"]["excluded_seeds_sha256"] = digest([9999998])
        with self.assertRaisesRegex(ValueError, "no imports"):
            self.initialize()

    def test_first_raw_owner_cannot_be_substituted(self):
        self.initialize()
        plan = self.observed_plan(canonical=False)
        with self.assertRaisesRegex(ValueError, "first raw owner"):
            ContinuationAdmissionJournal.create(self.ledger, plan, history_review_path=self.review_path)
        self.assertFalse((self.directory / "continuation-admission" / digest(dict(root_id=plan.root_id))).exists())

    def test_reinitialization_and_v1_import_into_fresh_scope_rejected(self):
        self.initialize()
        with self.assertRaisesRegex(ValueError, "already occupied"):
            initialize_fresh_namespace(self.ledger, history_review_path=self.review_path)
        plan = self.observed_plan()
        journal = ContinuationAdmissionJournal(self.ledger, root_id=plan.root_id)
        legacy = history_review_template(plan, ledger_directory=self.directory)
        legacy.update(status="CLEAR_DECLARED_ROOT_HISTORY",
            all_prior_attempts_in_declared_root_scope_accounted_for=True)
        with self.assertRaisesRegex(ValueError, "v1 import"):
            journal._review_owners(legacy, __import__(
                "pokezero.mcts_eval.search_over_raw_continuation_admission", fromlist=["_root_identity"]
                )._root_identity(plan), self.review_path, "a"*64)

    def test_prospective_runtime_is_binding_even_for_first_reservation(self):
        self.review["continuation_runtime_sha256"] = "b"*64
        self.initialize(); plan = self.observed_plan()
        journal = ContinuationAdmissionJournal.create(self.ledger, plan, history_review_path=self.review_path)
        job, _ = plan.prepare_continuation_job(config=self.outcome.config, runtime=self.outcome.runtime,
            _synthetic_job_class=outcome_fixture.SyntheticContinuationJob)
        with self.assertRaisesRegex(ValueError, "prospective fresh recipe"):
            journal.reserve(plan, job)
        self.assertEqual(journal.consumed_actions(plan), ())


class FailingContinuationJob(outcome_fixture.SyntheticContinuationJob):
    def _components(self):
        env, _ = super()._components()
        def evaluator(_):
            raise RuntimeError("authored continuation failure")
        return env, evaluator


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.h = stage_fixture.OwnedStageTests(); self.h.setUp()
        self.addCleanup(self.h.doCleanups)
        self.f = self.h.fixture()
        self.selected, self.reader = self.h.selector(self.f, action=1)
        self.outcome = outcome_fixture.OutcomeRunTests(); self.outcome.setUp()
        self.addCleanup(self.outcome.doCleanups)
        index = self.f.archive_fixture.source["roots"][0]["source_request_index"]
        self.f.archive_fixture.archive._private[index] = deepcopy(self.outcome.snapshot)
        self.pre_plan, _ = self.h.plan(self.f, self.selected, self.reader)
        self.review_path = self.f.owner_fixture.root / "history-review.json"
        self.ledger_directory = self.f.owner_fixture.root / "owned-ledger"

    def initialize(self, *, prior=(), clear=True, pin=True):
        review = history_review_template(self.pre_plan, ledger_directory=self.ledger_directory,
            prior_attempts=prior)
        if clear:
            review.update(status="CLEAR_DECLARED_ROOT_HISTORY",
                all_prior_attempts_in_declared_root_scope_accounted_for=True)
        self.review_path.write_text(json.dumps(review))
        hashes = stage_fixture.controller_input_hashes()
        if pin:
            hashes[str(self.review_path)] = hashlib.sha256(self.review_path.read_bytes()).hexdigest()
        with patch.object(stage_fixture, "controller_input_hashes", return_value=hashes):
            ledger = self.h.ledger([(self.f, self.selected, self.reader)])
        plan, _ = self.h.plan(self.f, self.selected, self.reader)
        journal = ContinuationAdmissionJournal.create(ledger, plan, history_review_path=self.review_path)
        self.ledger, self.plan, self.journal = ledger, plan, journal
        return journal

    def job(self, *, actions=None, job_class=outcome_fixture.SyntheticContinuationJob):
        return self.plan.prepare_continuation_job(config=self.outcome.config,
            runtime=self.outcome.runtime, actions_to_collect=actions,
            _synthetic_job_class=job_class)[0]

    def test_no_launch_and_missing_attempt_consumes_all_cells(self):
        with patch("subprocess.Popen") as launch:
            journal = self.initialize()
            job = self.job()
            reservation = journal.reserve(self.plan, job)
            self.assertEqual(journal.consumed_actions(self.plan), (0, 1))
            prepared, receipt = journal.prepare_new_job(self.plan, config=self.outcome.config,
                runtime=self.outcome.runtime, _synthetic_job_class=outcome_fixture.SyntheticContinuationJob)
            self.assertIsNone(prepared)
            journal.verify_reservation(self.plan, job, reservation)
        launch.assert_not_called()
        self.assertEqual(reservation["reserved_cells"], 16)
        self.assertFalse(Path(reservation["run_directory"]).exists())
        self.assertEqual(receipt["new_action_values"], [])
        self.assertTrue(receipt["missing_consumed_cells_remain_uncertain"])
        for flag in ("collection_authorized", "holdout_authorized", "execution_ready", "retry_authorized"):
            self.assertFalse(reservation[flag])
        with self.assertRaisesRegex(ValueError, "already consumed"):
            journal.reserve(self.plan, self.job())
        reopened = ContinuationAdmissionJournal(self.ledger, root_id=self.plan.root_id)
        self.assertEqual(reopened.consumed_actions(self.plan), (0, 1))

    def test_unknown_history_and_unpinned_review_block_no_journal(self):
        with self.assertRaisesRegex(ValueError, "unknown or mismatched"):
            self.initialize(clear=False)
        self.assertFalse((self.ledger_directory / "continuation-admission").exists())

    def test_unpinned_review_block(self):
        with self.assertRaisesRegex(ValueError, "prospectively bound"):
            self.initialize(pin=False)

    def test_partial_journal_cannot_be_reinitialized(self):
        journal = self.initialize()
        (journal.directory / "baseline.json").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            journal.consumed_actions(self.plan)
        with self.assertRaises(FileExistsError):
            ContinuationAdmissionJournal.create(self.ledger, self.plan, history_review_path=self.review_path)

    def test_crash_during_record_write_strands_consumed_scope(self):
        journal = self.initialize()
        def interrupted(path, record):
            path.write_text('{"schema":')
            raise OSError("authored write crash")
        with patch("pokezero.mcts_eval.search_over_raw_continuation_admission._write_new", side_effect=interrupted):
            with self.assertRaises(OSError):
                journal.reserve(self.plan, self.job())
        with self.assertRaises(ValueError):
            journal.reserve(self.plan, self.job())
        self.assertEqual(len(list((journal.directory / "attempts").iterdir())), 1)

    def test_competing_reservations_only_one_commits(self):
        journal = self.initialize(); job = self.job(actions=(1,))
        other = ContinuationAdmissionJournal(self.ledger, root_id=self.plan.root_id)
        barrier = threading.Barrier(2); results = []
        def reserve(owner):
            barrier.wait()
            try:
                results.append(owner.reserve(self.plan, job))
            except ValueError as exc:
                results.append(str(exc))
        workers = [threading.Thread(target=reserve, args=(owner,)) for owner in (journal, other)]
        for worker in workers: worker.start()
        for worker in workers: worker.join(timeout=10); self.assertFalse(worker.is_alive())
        self.assertEqual(sum(type(row) is dict for row in results), 1, results)
        self.assertTrue(any("already consumed" in row for row in results if type(row) is str))
        self.assertEqual(journal.consumed_actions(self.plan), (1,))

    def test_callback_symlink_to_same_lock_inode_rejects_before_write(self):
        journal = self.initialize(); job = self.job()
        lock = journal.directory / "lock"; moved = self.f.owner_fixture.root / "moved-lock"
        original = self.plan.verify_prepared_job; changed = False
        def late(value):
            nonlocal changed
            result = original(value)
            if not changed:
                changed = True; lock.rename(moved); lock.symlink_to(moved)
            return result
        with patch.object(self.plan, "verify_prepared_job", side_effect=late):
            with self.assertRaisesRegex(ValueError, "symlink"):
                journal.reserve(self.plan, job)
        self.assertEqual(list((journal.directory / "attempts").iterdir()), [])

    def test_missing_reviewed_prior_owner_is_consumed_not_scored(self):
        directory = self.f.owner_fixture.root / "missing-prior-owner"
        prior = dict(directory=str(directory), target_actions=[0], job_sha256="a"*64,
            attempt_binding_sha256="b"*64, runtime_sha256=self._runtime_sha(), inventory_sha256=None)
        journal = self.initialize(prior=(prior,))
        self.assertEqual(journal.consumed_actions(self.plan), (0,))
        job, receipt = journal.prepare_new_job(self.plan, config=self.outcome.config,
            runtime=self.outcome.runtime, _synthetic_job_class=outcome_fixture.SyntheticContinuationJob)
        self.assertEqual(set(job.actions.values()), {1})
        self.assertEqual(receipt["consumed_action_values"], [0])
        self.assertFalse(receipt["execution_ready"])
        journal.reserve(self.plan, job)

    def _runtime_sha(self):
        from pokezero.mcts_eval.search_over_raw_outcome_run import verify_outcome_runtime
        return verify_outcome_runtime(self.outcome.runtime, self.outcome.config)

    def test_failed_malformed_prior_terminal_still_consumes_targets(self):
        directory = self.f.owner_fixture.root / "failed-prior-owner"; directory.mkdir()
        (directory / "terminal.json").write_text('{"unfinished":')
        prior = dict(directory=str(directory), target_actions=[0, 1], job_sha256="a"*64,
            attempt_binding_sha256="b"*64, runtime_sha256=self._runtime_sha(), inventory_sha256=_inventory(directory))
        journal = self.initialize(prior=(prior,))
        self.assertEqual(journal.consumed_actions(self.plan), (0, 1))
        (directory / "terminal.json").write_text('{}')
        with self.assertRaisesRegex(ValueError, "inventory drift"):
            journal.consumed_actions(self.plan)

    def test_undeclared_known_ledger_cell_blocks_omitted_donor(self):
        journal = self.initialize()
        directory = self.ledger.directory / "owned-shared-cells"; directory.mkdir()
        cell = dict(root_id=self.plan.root_id, action=0, replicate=7,
            directory=str(self.outcome.directory), job_sha256="a"*64, attempt_binding_sha256="b"*64,
            capture_sha256=None, supplied=False)
        path = directory / (digest(dict(root_id=self.plan.root_id, action=0, replicate=7))+".json")
        path.write_text(json.dumps(cell))
        with self.assertRaisesRegex(ValueError, "coverage incomplete"):
            journal.prepare_new_job(self.plan, config=self.outcome.config, runtime=self.outcome.runtime,
                _synthetic_job_class=outcome_fixture.SyntheticContinuationJob)
        self.assertEqual(list((journal.directory / "attempts").iterdir()), [])

    def test_same_labels_wrong_owner_rejected(self):
        journal = self.initialize(); reservation = journal.reserve(self.plan, self.job())
        cells = self.ledger.directory / "owned-shared-cells"; cells.mkdir()
        cell = dict(root_id=self.plan.root_id, action=0, replicate=0,
            directory=str(self.outcome.directory), job_sha256=reservation["record"]["job_sha256"],
            attempt_binding_sha256=reservation["record"]["attempt_binding_sha256"], supplied=False, capture_sha256=None)
        (cells / (digest(dict(root_id=self.plan.root_id, action=0, replicate=0))+".json")).write_text(json.dumps(cell))
        with self.assertRaisesRegex(ValueError, "redrawn prior cell"):
            journal.consumed_actions(self.plan)

    def test_symlink_journal_or_owner_rejected(self):
        journal = self.initialize()
        (journal.directory / "runs" / "alias").symlink_to(self.outcome.directory, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            journal.consumed_actions(self.plan)

    def test_conflicting_run_claim_blocks_reconciliation_not_just_verifier(self):
        journal = self.initialize(); reservation = journal.reserve(self.plan, self.job(actions=(0,)))
        directory = Path(reservation["run_directory"]); directory.mkdir()
        claim = dict(job_sha256="f"*64, binding_sha256=reservation["record"]["attempt_binding_sha256"],
            contract=reservation["record"]["run_contract"])
        (directory / "claim.json").write_text(json.dumps(claim))
        with self.assertRaisesRegex(ValueError, "claim mismatch"):
            journal.prepare_new_job(self.plan, config=self.outcome.config, runtime=self.outcome.runtime,
                _synthetic_job_class=outcome_fixture.SyntheticContinuationJob)
        self.assertEqual(len(list((journal.directory / "attempts").iterdir())), 1)

    def test_conflicting_run_job_bytes_block_reconciliation(self):
        journal = self.initialize(); reservation = journal.reserve(self.plan, self.job(actions=(0,)))
        directory = Path(reservation["run_directory"]); directory.mkdir()
        (directory / "job.pickle").write_bytes(b"not the reserved job")
        with self.assertRaisesRegex(ValueError, "job bytes changed"):
            journal.consumed_actions(self.plan)

    def test_boolean_coordinate_cannot_alias_integer_cell(self):
        journal = self.initialize(); reservation = journal.reserve(self.plan, self.job())
        cells = self.ledger.directory / "owned-shared-cells"; cells.mkdir()
        cell = dict(root_id=self.plan.root_id, action=False, replicate=False,
            directory=reservation["run_directory"], job_sha256=reservation["record"]["job_sha256"],
            attempt_binding_sha256=reservation["record"]["attempt_binding_sha256"], supplied=False, capture_sha256=None)
        (cells / (digest(dict(root_id=self.plan.root_id, action=0, replicate=0))+".json")).write_text(json.dumps(cell))
        with self.assertRaisesRegex(ValueError, "coverage incomplete"):
            journal.consumed_actions(self.plan)

    def test_boolean_root_identity_drift_cannot_alias_review(self):
        journal = self.initialize()
        path = journal.directory / "baseline.json"; baseline = json.loads(path.read_text())
        identity = baseline["root_identity"]; identity["continuation_contract"]["selection_seed"] = False
        path.write_text(json.dumps(baseline))
        with self.assertRaisesRegex(ValueError, "raw owner drift"):
            journal.consumed_actions(self.plan)

    def test_late_owner_callback_mutation_does_not_establish_new_baseline(self):
        journal = self.initialize(); reservation = journal.reserve(self.plan, self.job())
        path = journal.directory / "attempts" / (reservation["reservation_sha256"]+".json")
        original = self.plan.revalidate; count = 0
        def late():
            nonlocal count
            result = original(); count += 1
            if count == 2:
                path.write_text(path.read_text()+" ")
            return result
        with patch.object(self.plan, "revalidate", side_effect=late):
            with self.assertRaisesRegex(ValueError, "bytes changed during owner callbacks"):
                journal.consumed_actions(self.plan)

    def test_final_archive_callback_cannot_rewrite_validated_raw_owner(self):
        journal = self.initialize(); journal.reserve(self.plan, self.job())
        path = self.plan.readers["raw"].directory / "terminal.json"
        original = self.plan.archive.selected; count = 0
        def late(root):
            nonlocal count
            result = original(root); count += 1
            if count == 6:
                path.write_text(path.read_text()+" ")
            return result
        with patch.object(self.plan.archive, "selected", side_effect=late):
            with self.assertRaises(ValueError):
                journal.consumed_actions(self.plan)
        self.assertEqual(count, 6)

    def install_reserved_cell(self, reservation):
        cells = self.ledger.directory / "owned-shared-cells"; cells.mkdir(exist_ok=True)
        cell = dict(root_id=self.plan.root_id, action=0, replicate=0,
            directory=reservation["run_directory"], job_sha256=reservation["record"]["job_sha256"],
            attempt_binding_sha256=reservation["record"]["attempt_binding_sha256"], supplied=False, capture_sha256=None)
        path = cells / (digest(dict(root_id=self.plan.root_id, action=0, replicate=0))+".json")
        path.write_text(json.dumps(cell)); return path

    def test_middle_prepare_callback_cannot_rebaseline_shared_capture(self):
        journal = self.initialize(); reservation = journal.reserve(self.plan, self.job(actions=(0,)))
        path = self.install_reserved_cell(reservation); original = self.plan.prepare_continuation_job
        def late(**kwargs):
            result = original(**kwargs)
            cell = json.loads(path.read_text()); cell["capture_sha256"] = "a"*64; cell["supplied"] = True
            path.write_text(json.dumps(cell)); return result
        with patch.object(self.plan, "prepare_continuation_job", side_effect=late):
            with self.assertRaisesRegex(ValueError, "bytes changed during preparation callbacks"):
                journal.prepare_new_job(self.plan, config=self.outcome.config, runtime=self.outcome.runtime,
                    _synthetic_job_class=outcome_fixture.SyntheticContinuationJob)

    def test_middle_reserve_callback_cannot_rebaseline_shared_capture(self):
        journal = self.initialize(); reservation = journal.reserve(self.plan, self.job(actions=(0,)))
        path = self.install_reserved_cell(reservation); job = self.job(actions=(1,))
        original = self.plan.verify_prepared_job
        def late(value):
            result = original(value)
            cell = json.loads(path.read_text()); cell["capture_sha256"] = "a"*64; cell["supplied"] = True
            path.write_text(json.dumps(cell)); return result
        with patch.object(self.plan, "verify_prepared_job", side_effect=late):
            with self.assertRaisesRegex(ValueError, "bytes changed during reservation callbacks"):
                journal.reserve(self.plan, job)
        self.assertEqual(len(list((journal.directory / "attempts").iterdir())), 1)

    def test_middle_verification_callback_cannot_rebaseline_shared_capture(self):
        journal = self.initialize(); job = self.job()
        reservation = journal.reserve(self.plan, job); path = self.install_reserved_cell(reservation)
        original = self.plan.verify_prepared_job
        def late(value):
            result = original(value)
            cell = json.loads(path.read_text()); cell["capture_sha256"] = "a"*64; cell["supplied"] = True
            path.write_text(json.dumps(cell)); return result
        with patch.object(self.plan, "verify_prepared_job", side_effect=late):
            with self.assertRaisesRegex(ValueError, "bytes changed during verification callbacks"):
                journal.verify_reservation(self.plan, job, reservation)

    def test_actual_failed_worker_remains_consumed_not_successful(self):
        from pokezero.mcts_eval.search_over_raw_outcome_run import supervise_outcome_run
        # Define a picklable fixture class at module scope below, with original
        # owned supervision rather than interpreting process disappearance.
        journal = self.initialize(); job = self.job(job_class=FailingContinuationJob)
        reservation = journal.reserve(self.plan, job)
        terminal = supervise_outcome_run(job, directory=reservation["run_directory"],
            binding_sha256=reservation["record"]["attempt_binding_sha256"],
            verify_admission=lambda: journal.verify_reservation(self.plan, job, reservation),
            validate_result=lambda _: True)
        self.assertEqual(terminal["supervisor_exit_code"], 1)
        self.assertEqual(terminal["worker_exit_code"], 1)
        self.assertEqual(terminal["status"], "OPERATIONAL_STOP")
        self.assertIsNone(terminal["result"])
        self.assertEqual(journal.consumed_actions(self.plan), (0, 1))
        with self.assertRaisesRegex(ValueError, "already consumed"):
            journal.reserve(self.plan, self.job())

    def test_duplicate_json_field_blocks(self):
        journal = self.initialize(); reservation = journal.reserve(self.plan, self.job())
        path = journal.directory / "attempts" / (reservation["reservation_sha256"]+".json")
        path.write_text('{"schema":"first","schema":"second"}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            journal.consumed_actions(self.plan)

    def test_partial_replicates_cannot_release_unwritten_eighth(self):
        journal = self.initialize(); reservation = journal.reserve(self.plan, self.job())
        path = journal.directory / "attempts" / (reservation["reservation_sha256"]+".json")
        record = reservation["record"]; record["replicates"] = list(range(7))
        path.unlink()
        (path.parent / (digest(record)+".json")).write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "partial action reservation"):
            journal.consumed_actions(self.plan)

    def test_stranded_root_does_not_block_another_reviewed_root(self):
        f = self.h.fixture(seed=1); selected, reader = self.h.selector(f, action=1)
        index = f.archive_fixture.source["roots"][0]["source_request_index"]
        f.archive_fixture.archive._private[index] = deepcopy(self.outcome.snapshot)
        pre, _ = self.h.plan(f, selected, reader)
        other_review = f.owner_fixture.root / "history-review.json"
        hashes = stage_fixture.controller_input_hashes()
        for plan, path in ((self.pre_plan, self.review_path), (pre, other_review)):
            review = history_review_template(plan, ledger_directory=self.ledger_directory)
            review.update(status="CLEAR_DECLARED_ROOT_HISTORY",
                all_prior_attempts_in_declared_root_scope_accounted_for=True)
            path.write_text(json.dumps(review)); hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with patch.object(stage_fixture, "controller_input_hashes", return_value=hashes):
            ledger = self.h.ledger([(self.f, self.selected, self.reader), (f, selected, reader)])
        first, _ = self.h.plan(self.f, self.selected, self.reader)
        second, _ = self.h.plan(f, selected, reader)
        broken = ContinuationAdmissionJournal.create(ledger, first, history_review_path=self.review_path)
        healthy = ContinuationAdmissionJournal.create(ledger, second, history_review_path=other_review)
        (broken.directory / "attempts" / "stranded.json").write_text('{"unfinished":')
        with self.assertRaises(ValueError):
            broken.consumed_actions(first)
        job, receipt = healthy.prepare_new_job(second, config=self.outcome.config, runtime=self.outcome.runtime,
            _synthetic_job_class=outcome_fixture.SyntheticContinuationJob)
        self.assertEqual(receipt["new_action_values"], [0, 1])
        healthy.reserve(second, job)
        self.assertEqual(healthy.consumed_actions(second), (0, 1))
        self.assertFalse((ledger.directory / "operational-stop.json").exists())

    def test_orphan_run_without_reservation_blocks_root(self):
        journal = self.initialize(); (journal.directory / "runs" / ("a"*64)).mkdir()
        with self.assertRaisesRegex(ValueError, "undeclared owned run"):
            journal.consumed_actions(self.plan)

    def test_seven_captures_do_not_release_eighth_or_allow_retry(self):
        from pokezero.mcts_eval.search_over_raw_outcome_run import OwningRunValidator, supervise_outcome_run
        journal = self.initialize(); job = self.job(job_class=reuse_fixture.MissingCellsSyntheticJob)
        reservation = journal.reserve(self.plan, job); record = reservation["record"]
        terminal = supervise_outcome_run(job, directory=reservation["run_directory"],
            binding_sha256=record["attempt_binding_sha256"],
            verify_admission=lambda: journal.verify_reservation(self.plan, job, reservation),
            validate_result=lambda _: True)
        self.assertEqual(terminal["supervisor_exit_code"], 0, terminal)
        self.assertEqual(terminal["result"]["outcomes"]["status"], "UNCERTAIN")
        self.assertEqual(len(terminal["result"]["bank"]["cells"]), 14)
        reader = OwningRunValidator(directory=reservation["run_directory"], job_sha256=record["job_sha256"],
            binding_sha256=record["attempt_binding_sha256"], root_binding=job.root_binding,
            expected_snapshot=job.snapshot, runtime=job.runtime, allow_synthetic=True)
        bank = OwnedContinuationBank(self.plan, job, terminal["result"], reader)
        self.h.ingest(self.ledger, {self.plan.root_id:self.plan}, {self.plan.root_id:(bank,)})
        self.assertEqual(journal.consumed_actions(self.plan), (0, 1))
        self.assertTrue(any(not json.loads(p.read_text())["supplied"]
            for p in (self.ledger.directory / "owned-shared-cells").iterdir()))
        with self.assertRaisesRegex(ValueError, "already consumed"):
            journal.reserve(self.plan, self.job())


if __name__ == "__main__":
    unittest.main()
