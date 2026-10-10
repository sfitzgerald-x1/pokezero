import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from pokezero.mcts_eval.search_over_raw import SearchConfiguration, phase_a_contract
from pokezero.mcts_eval.search_over_raw_ledger import PhaseALedger


NAMESPACE = "a267ac75-c45d-476a-8071-d29648191825"


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = SearchConfiguration("reference", workers=20).identity
        self.contract = phase_a_contract(NAMESPACE, excluded_seeds=[], configurations=[
            SearchConfiguration("raw"), SearchConfiguration("incumbent"),
            SearchConfiguration("reference", workers=20),
            SearchConfiguration("reference", belief="oracle", workers=20)])
        self.contract.update(execution_ready=True, remaining_gates=[])
        self.ledger = PhaseALedger.create(self.root / "ledger", self.contract)

    def tearDown(self):
        self.temp.cleanup()

    def intervals(self, panel):
        return {r["root_id"]: [.125, .125] for r in self.contract["panels"][panel]["root_slots"]}

    def freeze(self):
        self.ledger.record_exploration(self.config, self.intervals("exploration"))
        return self.ledger.freeze(self.config)

    def test_existing_ledger_is_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            PhaseALedger.create(self.root / "ledger", self.contract)

    def test_one_shot_validation_recomputes_summary_and_never_authorizes_phase_b(self):
        self.freeze()
        result = self.ledger.validation_once(lambda contract, config: self.intervals("validation"))
        self.assertEqual(result["gate"]["status"], "PHASE_A_GAIN_VALIDATED")
        self.assertFalse(result["phase_b_authorized"])
        other = PhaseALedger(self.root / "ledger")
        with self.assertRaises(FileExistsError):
            other.validation_once(lambda *_: self.fail("must not rerun"))

    def test_unfrozen_holdout_cannot_be_opened(self):
        with self.assertRaises(FileNotFoundError):
            self.ledger.validation_once(lambda *_: self.fail("must not start"))

    def test_freeze_closes_exploration_and_reselection(self):
        self.freeze()
        with self.assertRaisesRegex(ValueError, "frozen"):
            self.ledger.record_exploration(self.config, self.intervals("exploration"))
        with self.assertRaises(FileExistsError):
            self.ledger.freeze(self.config)

    def test_missing_roots_remain_uncertain_and_gate_is_closed(self):
        self.freeze()
        result = self.ledger.validation_once(lambda *_: {})
        self.assertEqual(result["summary"]["uncertain_roots"], 200)
        self.assertEqual(result["summary"]["identification_interval"], [-1., 1.])
        self.assertEqual(result["gate"]["status"], "NO_VALIDATED_GAIN")

    def test_failure_or_orphan_claim_never_permits_retry(self):
        self.freeze()
        def fail(*_):
            raise RuntimeError("PRIVATE_FAILURE_DETAIL")
        with self.assertRaises(RuntimeError):
            self.ledger.validation_once(fail)
        failure = (self.root / "ledger" / "validation-failure.json").read_text()
        self.assertNotIn("PRIVATE_FAILURE_DETAIL", failure)
        (self.root / "ledger" / "validation-failure.json").unlink()
        with self.assertRaises(FileExistsError):
            self.ledger.validation_once(lambda *_: self.fail("orphan claim must not restart"))

    def test_incomplete_admission_never_calls_worker(self):
        contract = dict(self.contract, execution_ready=False, remaining_gates=["runtime qualification"])
        ledger = PhaseALedger.create(self.root / "unready", contract)
        with self.assertRaisesRegex(ValueError, "admission"):
            ledger.validation_once(lambda *_: self.fail("must not launch"))
        self.assertFalse((self.root / "unready" / "validation-claim.json").exists())

    def test_bound_input_drift_prevents_validation(self):
        path = self.root / "fixture.txt"
        path.write_text("before")
        contract = dict(self.contract, input_hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest()})
        ledger = PhaseALedger.create(self.root / "bound", contract)
        ledger.record_exploration(self.config, self.intervals("exploration"))
        ledger.freeze(self.config)
        path.write_text("after")
        with self.assertRaisesRegex(ValueError, "input drift"):
            ledger.validation_once(lambda *_: self.fail("must not launch"))

    def test_modified_exploration_and_contract_are_detected(self):
        self.freeze()
        path = self.root / "ledger" / "exploration" / f"{self.config}.json"
        value = json.loads(path.read_text())
        value["summary"]["identification_interval"] = [.5, .5]
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "artifact drift"):
            self.ledger.validation_once(lambda *_: self.fail("must not launch"))
        path = self.root / "ledger" / "contract.json"
        value = json.loads(path.read_text())
        value["contract"]["candidate_seat"] = "p2"
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "contract drift"):
            self.ledger.verify()


if __name__ == "__main__":
    unittest.main()
