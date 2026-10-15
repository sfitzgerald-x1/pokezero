"""Synthetic captures only; never opens a native engine, source or Showdown."""
from copy import deepcopy
import hashlib
import unittest

from pokezero.mcts_eval.search_over_raw_correspondence import (
    REQUIRED_FIELDS, boundary_bindings, digest, reconcile_scoped_searched_line,
)

ROOT, REPORT = "a"*64, "b"*64


def capture():
    steps = []
    for index in range(2):
        steps.append(dict(key=[index, 0], parent=None if index == 0 else [index-1, 0],
            depth=index, before=f"state:{index}", after=f"state:{index+1}", branch_on_damage=False,
            side_one_choice={"kind":"move", "engine_id":"seismictoss"},
            side_two_choice={"kind":"move", "engine_id":"splash"}))
    return dict(root_native_state="state:0", steps=steps, native_frontier="state:2",
        model_signed_value=.2, evaluation_ordinal=7, private_controller_only=True)


def receipts(private):
    rows = []
    for binding in boundary_bindings(private, root_binding=ROOT, selection_report_sha256=REPORT):
        projection = dict(active_identities={"p1":"primeape", "p2":"blissey"},
            hp={"p1":[120, 200], "p2":[500, 600]}, fainted={"p1":False, "p2":False},
            status={"p1":"none", "p2":"none"},
            action_identity={"p1":"move:seismictoss", "p2":"move:splash"},
            effective_legal_choices={"p1":["move:seismictoss"], "p2":["move:splash"]},
            branch_transition={"terminal":False, "requested_players":["p1", "p2"]},
            native_pp={"p1":{"seismictoss":15}, "p2":{"splash":40}})
        rows.append(dict(ordinal=binding["ordinal"], binding=binding,
            native=deepcopy(projection), showdown=deepcopy(projection)))
    return rows


def edge_capture():
    private = capture()
    for step in private["steps"]:
        instructions = "[Damage(side_one, 12), Damage(side_two, 24)]"
        step["native_edge"] = dict(schema="pokezero.native-captured-edge.v1",
            root=step["depth"] == 0, branch_index=step["key"][1],
            branch_on_damage=step["branch_on_damage"], instruction_count=2,
            instructions_debug=instructions,
            instructions_debug_sha256=hashlib.sha256(instructions.encode()).hexdigest(),
            actions={seat: dict(step[field], slot=slot) for seat, field, slot in (
                ("p1", "side_one_choice", 0), ("p2", "side_two_choice", 1))})
    return private


class ScopedCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.capture = capture()
        self.rows = receipts(self.capture)

    def reconcile(self, rows=None, private=None, validator=lambda row, binding:digest(row)):
        return reconcile_scoped_searched_line(self.capture if private is None else private,
            self.rows if rows is None else rows, root_binding=ROOT,
            selection_report_sha256=REPORT, validate_boundary=validator)

    def test_all_actual_coordinates_bound_and_no_runtime_or_equivalence_claim(self):
        result = self.reconcile()
        self.assertEqual(result["status"], "REQUIRED_FIELDS_MATCHED")
        self.assertEqual((result["path_boundaries"], result["required_boundary_records"]), (2, 3))
        self.assertEqual([row["ordinal"] for row in result["boundaries"]], [0, 1, 2])
        self.assertFalse(result["searched_line_runtime_qualified"])
        self.assertFalse(result["full_engine_equivalence"])
        self.assertFalse(result["exact_rng_coupling_claim"])
        self.assertNotIn("state:0", str(result))
        self.assertNotIn("seismictoss", str(result))

    def test_only_native_pp_difference_is_known_nongating_class(self):
        self.rows[1]["native"]["native_pp"]["p1"]["seismictoss"] = 16
        result = self.reconcile()
        self.assertEqual(result["status"], "REQUIRED_FIELDS_MATCHED")
        differences = result["boundaries"][1]["documented_differences"]
        self.assertEqual(len(differences), 1)
        self.assertEqual(differences[0]["classification"], "engine_pp_model")
        self.assertFalse(differences[0]["required_field"])
        self.assertFalse(result["original_failure_reclassified"])

    def test_each_required_difference_limits_comparison_even_with_known_pp(self):
        for field in REQUIRED_FIELDS:
            rows = deepcopy(self.rows)
            rows[1]["native"][field] = {"synthetic_difference":True}
            rows[1]["native"]["native_pp"] = {"synthetic_difference":16}
            with self.subTest(field=field):
                result = self.reconcile(rows)
                self.assertEqual(result["status"], "LIMITED_COMPARISON")
                self.assertEqual(result["required_field_mismatches"], 1)
                self.assertEqual(result["boundaries"][1]["required_fields"][field], "MISMATCH")

    def test_missing_effective_legal_set_cannot_be_substituted_by_legal_chosen_action(self):
        del self.rows[1]["native"]["effective_legal_choices"]
        result = self.reconcile()
        self.assertEqual(result["uncheckable_required_fields"], 1)
        self.assertEqual(result["status"], "LIMITED_COMPARISON")
        self.assertEqual(result["boundaries"][1]["required_fields"]["action_identity"], "MATCH")
        self.assertEqual(result["boundaries"][1]["required_fields"]["effective_legal_choices"], "UNCHECKABLE")

    def test_missing_entire_boundary_remains_in_full_ancestry_denominator(self):
        result = self.reconcile(self.rows[:2])
        self.assertEqual((result["required_boundary_records"], result["recorded_boundary_records"]), (3, 2))
        self.assertEqual(result["uncheckable_required_fields"], len(REQUIRED_FIELDS))
        self.assertEqual(len(result["boundaries"]), 3)

    def test_unvalidated_projection_is_not_match_or_known_pp_evidence(self):
        self.rows[0]["native"]["native_pp"] = {"different":1}
        result = self.reconcile(validator=lambda row, binding:None)
        self.assertEqual(result["uncheckable_required_fields"], 3*len(REQUIRED_FIELDS))
        self.assertTrue(all(not row["documented_differences"] for row in result["boundaries"]))
        with self.assertRaises(ValueError): self.reconcile(validator=lambda row, binding:"c"*64)
        with self.assertRaises(ValueError): self.reconcile(validator=None)

    def test_wrong_root_report_state_action_and_ordinal_bindings_rejected(self):
        for field in ("root_binding", "selection_report_sha256", "before_sha256", "after_sha256",
                      "chosen_actions_sha256", "capture_sha256", "ancestry_sha256", "ordinal"):
            rows = deepcopy(self.rows)
            rows[1]["binding"][field] = 99 if field == "ordinal" else "c"*64
            with self.subTest(field=field):
                with self.assertRaises(ValueError): self.reconcile(rows)

    def test_duplicate_unexpected_or_extra_projection_rejected(self):
        bad = deepcopy(self.rows[0]); bad["ordinal"] = 9
        cases = [self.rows + [self.rows[0]], self.rows + [bad]]
        rows = deepcopy(self.rows); rows[0]["native"]["claimed_full_equivalence"] = True
        cases.append(rows)
        for rows in cases:
            with self.assertRaises(ValueError): self.reconcile(rows)

    def test_broken_native_ancestry_not_repaired_or_replaced(self):
        for mutation in ("prestate", "parent", "endpoint", "duplicate", "action", "marker"):
            private = deepcopy(self.capture)
            if mutation == "prestate": private["steps"][1]["before"] = "different"
            if mutation == "parent": private["steps"][1]["parent"] = None
            if mutation == "endpoint": private["native_frontier"] = "different"
            if mutation == "duplicate": private["steps"][1]["key"] = [0, 0]
            if mutation == "action": private["steps"][1]["side_one_choice"] = {"kind":"guessed"}
            if mutation == "marker": private["private_controller_only"] = False
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError): self.reconcile(private=private)

    def test_validator_arguments_are_copies_and_original_mutation_detected(self):
        def copied(row, binding):
            sha = digest(row)
            row.clear(); binding.clear()
            return sha
        self.assertEqual(self.reconcile(validator=copied)["status"], "REQUIRED_FIELDS_MATCHED")
        def changed(row, binding):
            self.capture["model_signed_value"] = .9
            return digest(row)
        with self.assertRaises(ValueError): self.reconcile(validator=changed)

    def test_nonfinite_private_prediction_or_boundary_never_certifies(self):
        private = deepcopy(self.capture); private["model_signed_value"] = float("nan")
        with self.assertRaises(ValueError): self.reconcile(private=private)
        self.rows[0]["native"]["hp"] = float("nan")
        with self.assertRaises(ValueError): self.reconcile()

    def test_structural_boolean_integer_alias_not_equal(self):
        self.rows[1]["native"]["fainted"]["p1"] = 0
        self.assertEqual(self.reconcile()["required_field_mismatches"], 1)
        rows = receipts(self.capture); rows[1]["binding"]["ordinal"] = True
        with self.assertRaises(ValueError): self.reconcile(rows)

    def test_no_scope_claim_when_no_boundary_receipts(self):
        result = self.reconcile([])
        self.assertEqual(result["status"], "LIMITED_COMPARISON")
        self.assertEqual(result["uncheckable_required_fields"], 3*len(REQUIRED_FIELDS))


class CapturedEdgeContractsTests(unittest.TestCase):
    def bindings(self, private):
        return boundary_bindings(private, root_binding=ROOT, selection_report_sha256=REPORT)

    def test_edge_records_are_bound_without_native_helper_or_scope_upgrade(self):
        private = edge_capture()
        original = deepcopy(private)
        result = reconcile_scoped_searched_line(private, receipts(private),
            root_binding=ROOT, selection_report_sha256=REPORT,
            validate_boundary=lambda row, binding: digest(row))
        self.assertEqual(private, original)
        self.assertEqual(result["status"], "REQUIRED_FIELDS_MATCHED")
        self.assertEqual(len(result["boundaries"]), 3)
        for flag in ("full_engine_equivalence", "exact_rng_coupling_claim",
                     "causal_mechanism_established", "searched_line_runtime_qualified",
                     "scientific_collection_authorized"):
            self.assertFalse(result[flag])
        changed = deepcopy(private)
        changed["steps"][0]["native_edge"]["instruction_count"] = 3
        self.assertNotEqual(self.bindings(private)[0]["capture_sha256"],
                            self.bindings(changed)[0]["capture_sha256"])

    def test_edge_schema_and_complete_seat_roster_are_required(self):
        for mutation in ("missing", "extra", "schema", "seat_missing", "seat_extra", "action_extra"):
            private = edge_capture()
            edge = private["steps"][0]["native_edge"]
            if mutation == "missing": edge.pop("instruction_count")
            if mutation == "extra": edge["claimed_physics"] = True
            if mutation == "schema": edge["schema"] = "unknown"
            if mutation == "seat_missing": edge["actions"].pop("p2")
            if mutation == "seat_extra": edge["actions"]["p3"] = deepcopy(edge["actions"]["p1"])
            if mutation == "action_extra": edge["actions"]["p1"]["inferred"] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.bindings(private)

    def test_edge_coordinates_and_branch_law_cannot_be_substituted(self):
        for field, value in (("root", False), ("root", 1), ("branch_index", 1),
                             ("branch_index", False), ("branch_on_damage", True),
                             ("branch_on_damage", 0)):
            private = edge_capture()
            private["steps"][0]["native_edge"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.bindings(private)
        private = edge_capture()
        private["steps"][1]["native_edge"]["root"] = True
        with self.assertRaises(ValueError): self.bindings(private)

    def test_instruction_count_and_exact_byte_hash_are_checked(self):
        for field, value in (("instruction_count", -1), ("instruction_count", 65537),
                             ("instruction_count", True), ("instruction_count", 2.0),
                             ("instructions_debug", ""), ("instructions_debug", []),
                             ("instructions_debug", "different"),
                             ("instructions_debug_sha256", "a" * 64)):
            private = edge_capture()
            private["steps"][0]["native_edge"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.bindings(private)
        for count in (0, 65536):
            private = edge_capture()
            private["steps"][0]["native_edge"]["instruction_count"] = count
            self.assertEqual(len(self.bindings(private)), 3)

    def test_instruction_cap_counts_utf8_bytes_and_accepts_exact_boundary(self):
        private = edge_capture()
        edge = private["steps"][0]["native_edge"]
        edge["instructions_debug"] = "é" * (8388608 // 2)
        edge["instructions_debug_sha256"] = hashlib.sha256(edge["instructions_debug"].encode()).hexdigest()
        self.assertEqual(len(self.bindings(private)), 3)
        edge["instructions_debug"] += "é"
        edge["instructions_debug_sha256"] = hashlib.sha256(edge["instructions_debug"].encode()).hexdigest()
        with self.assertRaises(ValueError): self.bindings(private)

    def test_move_and_switch_slots_preserve_exact_allowed_ranges(self):
        for kind, maximum in (("move", 4), ("switch", 6)):
            for slot in (0, maximum - 1):
                private = edge_capture()
                step = private["steps"][0]
                step["side_one_choice"] = dict(kind=kind, engine_id="same-id")
                step["native_edge"]["actions"]["p1"] = dict(kind=kind, engine_id="same-id", slot=slot)
                self.assertEqual(len(self.bindings(private)), 3)
            for slot in (-1, maximum, True, 0.0, None):
                private = edge_capture()
                step = private["steps"][0]
                step["side_one_choice"] = dict(kind=kind, engine_id="same-id")
                step["native_edge"]["actions"]["p1"] = dict(kind=kind, engine_id="same-id", slot=slot)
                with self.subTest(kind=kind, slot=slot), self.assertRaises(ValueError):
                    self.bindings(private)

    def test_synthetic_actions_keep_their_original_slot_and_name_law(self):
        for kind, engine_id in (("struggle", "struggle"), ("none", "No Move")):
            private = edge_capture()
            step = private["steps"][0]
            step["side_one_choice"] = dict(kind=kind, engine_id=engine_id)
            step["native_edge"]["actions"]["p1"] = dict(kind=kind, engine_id=engine_id, slot=None)
            self.assertEqual(len(self.bindings(private)), 3)
            for field, value in (("slot", 0), ("engine_id", "wrong")):
                changed = deepcopy(private)
                changed["steps"][0]["native_edge"]["actions"]["p1"][field] = value
                with self.subTest(kind=kind, field=field), self.assertRaises(ValueError):
                    self.bindings(changed)

    def test_both_seats_must_match_the_actual_captured_action(self):
        for seat in ("p1", "p2"):
            for field, value in (("kind", "switch"), ("engine_id", "different")):
                private = edge_capture()
                private["steps"][0]["native_edge"]["actions"][seat][field] = value
                with self.subTest(seat=seat, field=field), self.assertRaises(ValueError):
                    self.bindings(private)
