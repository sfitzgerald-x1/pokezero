"""Real disposable supervisors, authored roots; no scientific collection."""
from pathlib import Path
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw import SearchConfiguration, digest
from pokezero.mcts_eval.search_over_raw_continuation_admission import ContinuationAdmissionJournal
from pokezero.mcts_eval.search_over_raw_continuation_producer import produce_reserved_continuation
import test_search_over_raw_continuation_admission as admission_fixture
import test_search_over_raw_owned_stage as stage_fixture
import test_search_over_raw_outcome_run as outcome_fixture


class ProducerTests(unittest.TestCase):
    def setUp(self):
        self.h = admission_fixture.FreshLifecycleTests(); self.h.setUp()
        self.addCleanup(self.h.doCleanups)
        self.h.initialize(); self.h.observed_plan()
        self.s = stage_fixture.OwnedStageTests(); self.s.setUp()
        self.addCleanup(self.s.doCleanups)
        self.cfg = SearchConfiguration("incumbent", seconds=1.)

    def plan(self, *, cfg=None, zero=False, action=1):
        cfg = cfg or self.cfg; f = self.h.fixture
        selected, reader = self.s.selector(f, cfg=cfg, zero=zero, action=action)
        plan = f.plan(configurations=(f.cfg, cfg), selections={"raw":f.selected, cfg.identity:selected},
            selector_readers={"raw":f.reader, cfg.identity:reader})
        return plan, cfg

    def journal(self, plan):
        return ContinuationAdmissionJournal.create(self.h.ledger, plan, history_review_path=self.h.review_path)

    def produce(self, plan, journal, *, cfg=None, job_class=outcome_fixture.SyntheticContinuationJob):
        return produce_reserved_continuation(self.h.ledger, "A2", (cfg or self.cfg).identity,
            plan=plan, journal=journal, config=self.h.outcome.config, runtime=self.h.outcome.runtime,
            _synthetic_job_class=job_class)

    def test_actual_reservation_supervisor_reader_plan_chain_and_no_retry(self):
        plan, _ = self.plan(); journal = self.journal(plan)
        bank, receipt = self.produce(plan, journal)
        self.assertEqual(receipt["status"], "OWNED_CONTINUATIONS_VALIDATED")
        self.assertEqual(receipt["reserved_cells"], 16)
        self.assertEqual(journal.consumed_actions(plan), (0, 1))
        self.assertTrue(bank.reader.last_validation["process_receipts_validated"])
        self.assertTrue(Path(receipt["run_directory"]).is_dir())
        self.assertFalse(receipt["collection_authorized"])
        self.assertFalse(receipt["runtime_qualified_by_this_code"])
        with patch("subprocess.Popen") as launch, self.assertRaises(FileExistsError):
            self.produce(plan, journal)
        launch.assert_not_called()

    def test_zero_work_produces_no_action_or_continuation_and_no_global_stop(self):
        plan, _ = self.plan(zero=True); journal = self.journal(plan)
        with patch("subprocess.Popen") as launch:
            bank, receipt = self.produce(plan, journal)
        launch.assert_not_called()
        self.assertIsNone(bank)
        self.assertEqual(receipt["status"], "UNQUALIFIED_AT_CEILING")
        self.assertFalse(receipt["scored_search_action"])
        self.assertEqual(journal.consumed_actions(plan), ())
        self.assertFalse((self.h.directory / "operational-stop.json").exists())

    def test_second_configuration_reuses_consumed_values_no_new_worker(self):
        plan, _ = self.plan(); journal = self.journal(plan)
        bank, _ = self.produce(plan, journal)
        cfg = SearchConfiguration("incumbent", "public", "hp_fraction", 1.)
        other, cfg = self.plan(cfg=cfg)
        with patch("subprocess.Popen") as launch:
            second, receipt = self.produce(other, journal, cfg=cfg)
        launch.assert_not_called()
        self.assertIsNone(second)
        self.assertEqual(receipt["status"], "NO_NEW_ACTION_VALUES")
        self.assertEqual(receipt["new_action_values"], [])
        self.assertTrue(receipt["missing_consumed_cells_remain_uncertain"])
        self.assertEqual(journal.consumed_actions(other), (0, 1))
        self.assertTrue(bank.reader.last_validation["process_receipts_validated"])

    def test_failed_owned_worker_stops_ledger_and_cannot_retry_or_score(self):
        plan, _ = self.plan(); journal = self.journal(plan)
        with self.assertRaisesRegex(ValueError, "operational failure"):
            self.produce(plan, journal, job_class=admission_fixture.FailingContinuationJob)
        self.assertTrue((self.h.directory / "operational-stop.json").exists())
        self.assertEqual(len(list((journal.directory / "attempts").iterdir())), 1)
        with patch("subprocess.Popen") as launch, self.assertRaisesRegex(ValueError, "integrity failure"):
            self.produce(plan, journal)
        launch.assert_not_called()
        self.assertFalse(any((self.h.directory / "A2").iterdir()))

    def test_no_fixture_marker_means_no_authority_and_no_launch(self):
        plan, _ = self.plan(); journal = self.journal(plan)
        with patch("subprocess.Popen") as launch, self.assertRaisesRegex(ValueError, "synthetic selectors"):
            produce_reserved_continuation(self.h.ledger, "A2", self.cfg.identity,
                plan=plan, journal=journal, config=self.h.outcome.config, runtime=self.h.outcome.runtime)
        launch.assert_not_called()
        self.assertFalse((self.h.directory / "producer-root-claims").exists())

    def test_a3_one_second_route_cannot_launch(self):
        plan, _ = self.plan(); journal = self.journal(plan)
        with patch.object(self.h.ledger, "_stage_configs", return_value=(self.cfg,)), \
                patch("subprocess.Popen") as launch, self.assertRaisesRegex(ValueError, "reuse frozen A2"):
            produce_reserved_continuation(self.h.ledger, "A3", self.cfg.identity,
                plan=plan, journal=journal, config=self.h.outcome.config, runtime=self.h.outcome.runtime,
                _synthetic_job_class=outcome_fixture.SyntheticContinuationJob)
        launch.assert_not_called()
        self.assertFalse((self.h.directory / "producer-root-claims").exists())

    def late_mutation(self, mutation):
        plan, _ = self.plan(); journal = self.journal(plan)
        original_accept = plan.accept_continuation_result
        original_verify = journal.verify_reservation
        accepted = {}
        def capture(job, result, **kwargs):
            value = original_accept(job, result, **kwargs)
            accepted.update(result=result, reader=kwargs["owning_reader"])
            return value
        def late(*args):
            value = original_verify(*args)
            if accepted and not accepted.get("mutated"):
                accepted["mutated"] = True
                mutation(accepted)
            return value
        with patch.object(plan, "accept_continuation_result", side_effect=capture), \
                patch.object(journal, "verify_reservation", side_effect=late), \
                self.assertRaisesRegex(ValueError, "changed"):
            self.produce(plan, journal)
        self.assertTrue(accepted.get("mutated"))
        self.assertTrue((self.h.directory / "operational-stop.json").exists())
        self.assertEqual(len(list((journal.directory / "attempts").iterdir())), 1)

    def test_late_terminal_bytes_cannot_become_new_accepted_baseline(self):
        def change(accepted):
            path = accepted["reader"].directory / "terminal.json"
            path.write_text(path.read_text()+"\n")
        self.late_mutation(change)

    def test_late_rehashed_result_cannot_become_new_accepted_baseline(self):
        def change(accepted):
            result = accepted["result"]
            result["bank"]["late_authored_mutation"] = True
            result["bank_sha256"] = digest(result["bank"])
        self.late_mutation(change)

    def test_late_reader_scope_cannot_become_new_accepted_baseline(self):
        self.late_mutation(lambda a: setattr(a["reader"], "allow_synthetic", False))

    def test_late_producer_claim_is_guarded_after_journal_callbacks(self):
        def change(_):
            path = next((self.h.directory / "producer-root-claims").iterdir())
            path.write_text(path.read_text()+"\n")
        self.late_mutation(change)

    def test_zero_work_callback_cannot_change_claim_before_no_run_return(self):
        plan, _ = self.plan(zero=True); journal = self.journal(plan)
        original = plan.revalidate; changed = False
        def late():
            nonlocal changed
            value = original()
            claims = self.h.directory / "producer-root-claims"
            if claims.exists() and not changed:
                changed = True; path = next(claims.iterdir())
                path.write_text(path.read_text()+"\n")
            return value
        with patch.object(plan, "revalidate", side_effect=late), patch("subprocess.Popen") as launch, \
                self.assertRaisesRegex(ValueError, "producer claim changed"):
            self.produce(plan, journal)
        launch.assert_not_called()
        self.assertTrue(changed)
        self.assertEqual(list((journal.directory / "attempts").iterdir()), [])

    def test_no_new_values_callback_cannot_change_claim_before_return(self):
        plan, _ = self.plan(); journal = self.journal(plan)
        self.produce(plan, journal)
        cfg = SearchConfiguration("incumbent", "public", "hp_fraction", 1.)
        other, _ = self.plan(cfg=cfg)
        original = other.revalidate; changed = False
        target = self.h.directory / "producer-root-claims" / (digest(dict(
            stage="A2", configuration=cfg.identity, root_id=other.root_id))+".json")
        def late():
            nonlocal changed
            value = original()
            if target.exists() and not changed:
                changed = True; target.write_text(target.read_text()+"\n")
            return value
        with patch.object(other, "revalidate", side_effect=late), patch("subprocess.Popen") as launch, \
                self.assertRaisesRegex(ValueError, "producer claim changed"):
            self.produce(other, journal, cfg=cfg)
        launch.assert_not_called()
        self.assertTrue(changed)
        self.assertEqual(len(list((journal.directory / "attempts").iterdir())), 1)

    def test_failed_stop_write_preserves_original_error(self):
        plan, _ = self.plan(); journal = self.journal(plan)
        with patch.object(journal, "consumed_actions", side_effect=RuntimeError("authored primary")), \
                patch.object(self.h.ledger, "stop", side_effect=OSError("authored disk failure")), \
                self.assertRaisesRegex(RuntimeError, "authored primary") as raised:
            self.produce(plan, journal)
        self.assertIn("OSError", raised.exception.__notes__[0])
        self.assertEqual(list((journal.directory / "attempts").iterdir()), [])
