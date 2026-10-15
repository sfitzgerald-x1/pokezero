"""Fresh-byte admission checks on authored ledgers; never runs a selector."""
from copy import deepcopy
import cProfile
import hashlib
import json
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw_reconciled import (
    ReconciledPhaseALedger, HOLDOUT_ONLY_GATE, a2_configurations, controller_input_hashes,
)
from pokezero.mcts_eval.search_over_raw_selector_producer import _verify_selector_admission
import test_search_over_raw_reconciled as ledger_fixture


class SelectorLedgerAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = ledger_fixture.ReconciledLedgerTest(); self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.bound = self.fixture.root/"authored-bound-input.bin"
        self.original = b"authored-bound-input\n" * 200_000
        self.bound.write_bytes(self.original)
        self.cfg = a2_configurations()[0].identity
        self.serial = 0

    def ledger(self, **changes):
        self.serial += 1
        contract = deepcopy(self.fixture.contract)
        contract.update(execution_ready=True, collection_authorized=True,
            remaining_gates=[HOLDOUT_ONLY_GATE], input_hashes={**controller_input_hashes(),
                str(self.bound):hashlib.sha256(self.bound.read_bytes()).hexdigest()})
        contract.update(changes)
        # Canonical production class, not the fixture module's aliased class.
        return ReconciledPhaseALedger.create(self.fixture.root/f"admission-{self.serial}", contract)

    def admit(self, ledger, *, synthetic=False, stage="A2", configuration=None):
        return _verify_selector_admission(ledger, stage, configuration or self.cfg, synthetic=synthetic)

    def measure(self, ledger, operation):
        original_read = Path.read_bytes
        reads = []
        def read(path):
            if path == self.bound: reads.append(path)
            return original_read(path)
        profile = cProfile.Profile()
        with patch.object(Path, "read_bytes", autospec=True, side_effect=read):
            began = time.process_time_ns()
            result = profile.runcall(operation)
            cpu_ns = time.process_time_ns()-began
        self.assertIsNone(result)
        scans = sum(row.callcount for row in profile.getstats()
            if row.code is ReconciledPhaseALedger.verify.__code__)
        return dict(scans=scans, bound_input_reads=len(reads), cpu_ns=cpu_ns)

    def test_real_ledger_before_after_scans_and_alternating_cpu_measurement(self):
        ledger = self.ledger()
        def before():
            ledger.verify()
            ledger.verify_exploration("A2", self.cfg)
        results = dict(before=[], after=[])
        for ordinal in range(6):
            order = ("before", "after") if ordinal % 2 == 0 else ("after", "before")
            for name in order:
                value = self.measure(ledger, before if name == "before" else lambda:self.admit(ledger))
                self.assertEqual(value["scans"], 2 if name == "before" else 1)
                self.assertEqual(value["bound_input_reads"], 2 if name == "before" else 1)
                results[name].append(value)
        synthetic = self.measure(ledger, lambda:self.admit(ledger, synthetic=True))
        self.assertEqual((synthetic["scans"], synthetic["bound_input_reads"]), (1, 1))
        print("SELECTOR_ADMISSION_MEASUREMENT:"+json.dumps(dict(bound_bytes=len(self.original),
            repetitions=6, **{name:dict(scans=sum(r["scans"] for r in rows),
                bound_input_reads=sum(r["bound_input_reads"] for r in rows),
                cpu_ns=sum(r["cpu_ns"] for r in rows)) for name,rows in results.items()}), sort_keys=True))

    def test_production_rejects_unadmitted_or_unknown_gates(self):
        for changes in (dict(execution_ready=False), dict(collection_authorized=False),
                dict(remaining_gates=["unknown upstream constraint"])):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.admit(self.ledger(**changes))

    def test_configuration_selection_and_a2_freezes_still_refuse(self):
        with self.assertRaisesRegex(ValueError, "stage-local registered configuration"):
            self.admit(self.ledger(), configuration="0"*64)
        for filename in ("selection.json", "a2-freeze.json"):
            ledger = self.ledger()
            (ledger.directory/filename).write_text("{}")
            with self.subTest(filename=filename), self.assertRaises(ValueError): self.admit(ledger)

    def test_synthetic_integrity_check_does_not_grant_execution_authority(self):
        ledger = self.ledger(execution_ready=False, collection_authorized=False,
            remaining_gates=["separate collection authority"])
        self.admit(ledger, synthetic=True)
        self.assertFalse(ledger.contract["execution_ready"])
        self.assertFalse(ledger.contract["collection_authorized"])
        with self.assertRaises(ValueError): self.admit(ledger)
        with self.assertRaises(ValueError): ledger.verify(execution=True)
        self.assertFalse((ledger.directory/"validation-claim.json").exists())
        self.assertFalse((ledger.directory/"source-attempts").exists())

    def test_same_size_same_timestamp_input_mutation_cannot_reuse_success(self):
        for synthetic in (False, True):
            ledger = self.ledger()
            self.admit(ledger, synthetic=synthetic)
            before = self.bound.stat()
            self.bound.write_bytes(b"x"*len(self.original))
            os.utime(self.bound, ns=(before.st_atime_ns, before.st_mtime_ns))
            with self.subTest(synthetic=synthetic), self.assertRaisesRegex(ValueError, "bound input drift"):
                self.admit(ledger, synthetic=synthetic)
            self.bound.write_bytes(self.original)

    def test_durable_contract_and_operational_stop_are_freshly_checked(self):
        for synthetic in (False, True):
            for mutation in ("contract", "stop"):
                ledger = self.ledger()
                self.admit(ledger, synthetic=synthetic)
                if mutation == "contract":
                    path = ledger.directory/"contract.json"
                    envelope = json.loads(path.read_bytes()); envelope["contract_sha256"] = "0"*64
                    path.write_text(json.dumps(envelope))
                else:
                    ledger.stop("AuthoredFailure")
                with self.subTest(synthetic=synthetic, mutation=mutation), self.assertRaises(ValueError):
                    self.admit(ledger, synthetic=synthetic)

    def test_scope_is_not_coerced(self):
        ledger = self.ledger()
        for synthetic in (0, 1, None, "false"):
            with self.subTest(synthetic=synthetic), self.assertRaisesRegex(ValueError, "explicit selector admission scope"):
                self.admit(ledger, synthetic=synthetic)

    def test_shadowed_exploration_method_cannot_bypass_independent_integrity(self):
        ledger = self.ledger()
        admission_calls = []
        ledger.verify_exploration = lambda *a:admission_calls.append(a)
        self.bound.write_bytes(b"x"*len(self.original))
        with self.assertRaisesRegex(ValueError, "bound input drift"): self.admit(ledger)
        self.assertEqual(admission_calls, [])

    def test_stateful_verify_wrapper_retains_both_calls_and_detects_intervening_drift(self):
        ledger = self.ledger()
        original_verify = ledger.verify
        calls = []
        def verify():
            calls.append("verify")
            original_verify()
            if len(calls) == 1: self.bound.write_bytes(b"x"*len(self.original))
        ledger.verify = verify
        with self.assertRaisesRegex(ValueError, "bound input drift"): self.admit(ledger)
        self.assertEqual(calls, ["verify", "verify"])

    def test_shadowed_descriptor_lookup_preserves_original_call_order(self):
        ledger = self.ledger()
        original_verify = ledger.verify
        calls = []
        def verify():
            calls.append("verify"); original_verify()
        ledger.verify = verify
        def descriptor(instance):
            calls.append("lookup")
            return lambda *args:calls.append("admission")
        with patch.object(ReconciledPhaseALedger, "verify_exploration", property(descriptor)):
            self.admit(ledger)
        self.assertEqual(calls, ["verify", "lookup", "admission"])

    def test_unconditional_shortcut_fails_both_override_regressions(self):
        # Demonstrate that the negatives distinguish the initial unsafe
        # shortcut, rather than merely passing on every implementation.
        def unsafe(ledger, stage, configuration, *, synthetic):
            if synthetic: ledger.verify()
            else: ledger.verify_exploration(stage, configuration)
        names = ("test_shadowed_exploration_method_cannot_bypass_independent_integrity",
            "test_stateful_verify_wrapper_retains_both_calls_and_detects_intervening_drift")
        result = unittest.TestResult()
        with patch(__name__+"._verify_selector_admission", unsafe):
            unittest.TestSuite(type(self)(name) for name in names).run(result)
        self.assertEqual(result.testsRun, 2)
        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.failures), 2)
