"""Real offline native toy support and authored ownership/protocol boundaries.

No learned evaluator, production runtime, source panel, or scientific games.
"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pokezero.mcts_eval.search_over_raw_correspondence import _state_sha, boundary_bindings
from pokezero.mcts_eval.search_over_raw_native_edge import (
    EDGE_SCHEMA, SUPPORT_SCHEMA, FALSE_FLAGS, validate_captured_edge,
    read_native_edge_support, extract_native_edge_support, observe_supported_prefix,
)
from pokezero.mcts_eval.search_over_raw_ancestry import (
    EDGE_SCHEMA as ANCESTRY_EDGE_SCHEMA, NativeAncestryContract, NativeAncestrySession,
)
from test_search_over_raw_correspondence import capture, ROOT, REPORT
from test_search_over_raw_native_boundary import report
from pokezero.mcts_eval.search_over_raw_native_boundary import extract_native_capture_boundaries
import test_search_over_raw_ancestry as ancestry_fixture
import test_search_over_raw_consumed_materialization as owner_fixture
from pokezero.mcts_eval.search_over_raw_consumed_materialization import join_owned_raw_replays
from pokezero.mcts_eval.search_over_raw_native_edge import (SUCCESSOR_SCHEMA,SUCCESSOR_SCOPE,ENDPOINT_FIELDS,
    read_native_successor_support,extract_native_successor_support,compare_projected_successors)
from pokezero.mcts_eval.search_over_raw_native_boundary import project_native_boundary
from pokezero.mcts_eval.search_over_raw import digest


def add_edge(step, slot=0):
    edge = dict(schema=EDGE_SCHEMA, root=step["depth"] == 0, branch_index=step["key"][1],
        branch_on_damage=step["branch_on_damage"], instruction_count=0, instructions_debug="[]",
        instructions_debug_sha256=_state_sha("[]"), actions={seat:dict(choice, slot=slot)
            for seat, choice in (("p1", step["side_one_choice"]), ("p2", step["side_two_choice"]))})
    step["native_edge"] = edge
    return edge


def prospective_capture():
    value = capture()
    for step in value["steps"]: add_edge(step)
    return value


def support_report(step, supported=True):
    # Authored transport fixture; not engine semantics evidence.
    index = step["key"][1]
    return dict(schema=SUPPORT_SCHEMA, before_utf8_sha256=_state_sha(step["before"]),
        after_utf8_sha256=_state_sha(step["after"]), captured_edge=deepcopy(step["native_edge"]),
        prestate_terminal=False, joint_action_admissible=True, regenerated_branches=index+1,
        matching_successor_branch_indices=[index] if supported else [], successor_in_native_support=supported,
        recorded_branch_instructions_match=True, recorded_edge_supported=supported,
        private_controller_only=True, **{flag:False for flag in FALSE_FLAGS})


def observations(n=4):
    fields = ("active_identities", "hp", "fainted", "status", "effective_legal_choices", "action_identity")
    return [dict(ordinal=i, observed_required_fields={k:"MATCH" for k in fields}) for i in range(n)]


class AuthoredEdgeAncestryJob(ancestry_fixture.AuthoredAncestryJob):
    def __call__(self, ownership, phase):
        selected = super().__call__(ownership, phase)
        calls = selected["evidence"]["consumed_world_inputs"]["calls"]
        exports = selected["evidence"]["native_searched_ancestry"]
        for worker, (row, call) in enumerate(zip(exports, calls)):
            payload = json.loads(row["export_utf8"])
            payload["schema"] = ANCESTRY_EDGE_SCHEMA
            for leaf in payload["evaluations"]:
                for step in leaf["capture"]["steps"]: add_edge(step)
            exports[worker] = ancestry_fixture.row_from_payload(payload, call, worker)
        return selected


class NativeEdgeTests(unittest.TestCase):
    def test_prospective_binding_and_mutation_rejected(self):
        step = prospective_capture()["steps"][0]
        self.assertEqual(validate_captured_edge(step), step["native_edge"])
        for mode in ("slot", "id", "root", "law", "index", "boolindex", "count", "hash", "extra"):
            bad = deepcopy(step); edge = bad["native_edge"]
            if mode == "slot": edge["actions"]["p1"]["slot"] = True
            if mode == "id": edge["actions"]["p1"]["engine_id"] = "different"
            if mode == "root": edge["root"] = False
            if mode == "law": edge["branch_on_damage"] = not edge["branch_on_damage"]
            if mode == "index": edge["branch_index"] += 1
            if mode == "boolindex": edge["branch_index"] = False
            if mode == "count": edge["instruction_count"] = True
            if mode == "hash": edge["instructions_debug"] = "altered"
            if mode == "extra": edge["physics_validated"] = True
            with self.subTest(mode=mode), self.assertRaises(ValueError): validate_captured_edge(bad)

    def test_exact_helper_report_and_all_scope_flags_remain_false(self):
        step = prospective_capture()["steps"][0]; value = support_report(step)
        self.assertEqual(read_native_edge_support(json.dumps(value), step), value)
        for mode in ("before", "after", "edge", "boolcount", "duplicates", "terminal", "logical", "count", "schema"):
            bad = deepcopy(value)
            if mode in ("before", "after"): bad[mode+"_utf8_sha256"] = "a"*64
            if mode == "edge": bad["captured_edge"]["branch_index"] += 1
            if mode == "boolcount": bad["regenerated_branches"] = True
            if mode == "duplicates": bad["matching_successor_branch_indices"] *= 2
            if mode == "terminal": bad["prestate_terminal"] = True
            if mode == "logical": bad["recorded_edge_supported"] = False
            if mode == "count": bad["regenerated_branches"] = 4097
            if mode == "schema": bad["schema"] = "old"
            with self.subTest(mode=mode), self.assertRaises(ValueError): read_native_edge_support(json.dumps(bad), step)
        for flag in FALSE_FLAGS:
            bad = dict(value, **{flag:True})
            with self.subTest(flag=flag), self.assertRaises(ValueError): read_native_edge_support(json.dumps(bad), step)
        with self.assertRaises(ValueError):
            read_native_edge_support(json.dumps(value)[:-1]+',"schema":"duplicate"}', step)

    def test_all_edges_keep_original_coordinates_and_legacy_unknown(self):
        value = prospective_capture(); calls = []
        def helper(before, after, edge_json):
            calls.append((before, after, json.loads(edge_json)))
            step = next(s for s in value["steps"] if s["after"] == after)
            return json.dumps(support_report(step, supported=len(calls) == 1))
        rows = extract_native_edge_support(value, root_binding=ROOT, selection_report_sha256=REPORT, native_support=helper)
        self.assertEqual([r["ordinal"] for r in rows], [1,2])
        self.assertEqual([r["status"] for r in rows], ["SUPPORTED", "UNSUPPORTED"])
        self.assertEqual([c[:2] for c in calls], [(s["before"], s["after"]) for s in value["steps"]])
        legacy = extract_native_edge_support(capture(), root_binding=ROOT, selection_report_sha256=REPORT,
            native_support=Mock(side_effect=AssertionError("do not regenerate historical slots")))
        self.assertTrue(all(r["status"] == "UNCHECKABLE" and r["support_report"] is None for r in legacy))
        absent = extract_native_edge_support(value, root_binding=ROOT, selection_report_sha256=REPORT)
        self.assertTrue(all(r["status"] == "UNCHECKABLE" for r in absent))

    def test_support_callback_failure_is_not_retried_or_converted_to_negative_evidence(self):
        helper = Mock(side_effect=RuntimeError("helper failed"))
        with self.assertRaisesRegex(RuntimeError, "helper failed"):
            extract_native_edge_support(prospective_capture(), root_binding=ROOT,
                selection_report_sha256=REPORT, native_support=helper)
        helper.assert_called_once()

    def test_callback_mutation_of_original_path_rejected(self):
        value = prospective_capture()
        def helper(before, after, edge):
            step = next(s for s in value["steps"] if s["after"] == after)
            raw = json.dumps(support_report(step))
            value["model_signed_value"] += .01
            return raw
        with self.assertRaisesRegex(ValueError, "mutated"):
            extract_native_edge_support(value, root_binding=ROOT, selection_report_sha256=REPORT, native_support=helper)

    def test_duplicate_ids_use_prospective_slot_not_first_matching_name(self):
        value = prospective_capture(); add_edge(value["steps"][0], slot=1)
        def exporter(state, root):
            result = report(state, root); side = result["p1"]
            side["active_moves"][1] = dict(side["active_moves"][0], slot=1)
            option = dict(kind="move", slot=1, engine_id="seismictoss")
            for key in ("engine_options", "search_arms", "effective_legal_choices"): side[key].append(option.copy())
            # P2 keeps slot 0; it has no slot-1 legal option.
            return json.dumps(result)
        value["steps"][0]["native_edge"]["actions"]["p2"]["slot"] = 0
        rows = extract_native_capture_boundaries(value, root_binding=ROOT,
            selection_report_sha256=REPORT, native_extract=exporter)
        self.assertEqual(rows[1]["native"]["action_identity"]["p1"]["slot"], 1)
        self.assertIsNone(rows[1]["native"]["branch_transition"])

    def test_monotone_prefix_does_not_recover_after_endpoint_action_or_support_gap(self):
        for gap in ("endpoint", "action", "support", "unknown", "root"):
            observed = observations(); support = [dict(ordinal=i, status="SUPPORTED") for i in range(1,4)]
            if gap == "endpoint": observed[1]["observed_required_fields"]["hp"] = "MISMATCH"
            if gap == "action": observed[1]["observed_required_fields"]["action_identity"] = "MISMATCH"
            if gap == "support": support[0]["status"] = "UNSUPPORTED"
            if gap == "unknown": support[0]["status"] = "UNCHECKABLE"
            if gap == "root": observed[0]["observed_required_fields"]["status"] = "UNCHECKABLE"
            rows = observe_supported_prefix(observed, support)
            with self.subTest(gap=gap):
                self.assertFalse(rows[-1]["observed_prefix_match"])
                self.assertTrue(rows[-1]["local_required_pre_action_after_match"])
                self.assertEqual(rows[-1]["first_uncertain_or_divergent_ordinal"], 0 if gap == "root" else 1)
                self.assertFalse(rows[-1]["branch_transition_validated"])

    def test_full_supported_prefix_is_only_observed_agreement(self):
        rows = observe_supported_prefix(observations(), [dict(ordinal=i, status="SUPPORTED") for i in range(1,4)])
        self.assertTrue(all(r["observed_prefix_match"] for r in rows))
        self.assertTrue(all(not r["matched_causal_trajectory"] and not r["independently_validated"] for r in rows))
        with self.assertRaises(ValueError): observe_supported_prefix(observations(), [])

    def test_v2_ancestry_is_prospective_opt_in_and_cannot_upgrade_partial_history(self):
        fixture = ancestry_fixture.AncestryTests(methodName="runTest")
        row, call = fixture.fixture(); payload = json.loads(row["export_utf8"])
        payload["schema"] = ANCESTRY_EDGE_SCHEMA
        for leaf in payload["evaluations"]:
            for step in leaf["capture"]["steps"]: add_edge(step)
        edge_row = ancestry_fixture.row_from_payload(payload, call)
        contract = NativeAncestryContract(17, capture_native_edges=True)
        result = fixture.read(edge_row, call, contract=contract)
        self.assertEqual(len(result["retained_evaluations"]),2)
        with self.assertRaises(ValueError): fixture.read(edge_row, call)
        with self.assertRaises(ValueError): fixture.read(row, call, contract=contract)
        del payload["evaluations"][0]["capture"]["steps"][0]["native_edge"]
        with self.assertRaises(ValueError): fixture.read(ancestry_fixture.row_from_payload(payload, call), call, contract=contract)

    def test_v2_session_passes_opt_in_without_changing_legacy_constructor(self):
        for enabled in (True,False):
            module = SimpleNamespace(NativeVisitedValueBank=Mock(return_value=Mock()))
            session = NativeAncestrySession(NativeAncestryContract(17,capture_native_edges=enabled),
                root_key="aa",native_module=module)
            session.begin(dict(state_str="fixture",seed=7))
            self.assertEqual(module.NativeVisitedValueBank.call_args.kwargs.get("capture_native_edges",False),enabled)
            if not enabled: self.assertNotIn("capture_native_edges",module.NativeVisitedValueBank.call_args.kwargs)
            session.close()

    def test_controller_binds_new_native_and_python_sources(self):
        from pokezero.mcts_eval.search_over_raw_reconciled import controller_input_hashes
        names = {p.rsplit("/",1)[-1] for p in controller_input_hashes()}
        self.assertTrue({"native_edge.rs", "search_over_raw_native_edge.py"} <= names)

    def owned_fixture(self, **kwargs):
        fixture = owner_fixture.OwnedRawReplayJoinTests(methodName="runTest")
        self.addCleanup(fixture.doCleanups)
        contract = NativeAncestryContract(17,capture_native_edges=True)
        with patch.object(owner_fixture, "AuthoredAncestryJob", AuthoredEdgeAncestryJob), \
             patch.object(owner_fixture, "CONTRACT", contract):
            owned = fixture.fixture(**kwargs)
        return fixture, owned, contract

    @staticmethod
    def owned_helper(before, after, edge_json):
        edge = json.loads(edge_json)
        step = dict(before=before, after=after, key=[0,edge["branch_index"]], depth=0 if edge["root"] else 1,
            branch_on_damage=edge["branch_on_damage"], native_edge=edge,
            side_one_choice={k:edge["actions"]["p1"][k] for k in ("kind","engine_id")},
            side_two_choice={k:edge["actions"]["p2"][k] for k in ("kind","engine_id")})
        return json.dumps(support_report(step))

    def join_owners(self, fixture, owned, contract, helper=None):
        native, refs, replays, reader = owned
        return join_owned_raw_replays(native, refs, replays, native_validator=reader,
            ancestry_contract=contract, native_extract=fixture.extractor, native_support=helper or self.owned_helper)

    def test_owned_join_retains_all_edges_and_keeps_divergent_showdown_uncertain(self):
        fixture, owned, contract = self.owned_fixture(native_driven=True)
        result = self.join_owners(fixture, owned, contract)
        self.assertEqual(result["native_edge_support_counts"], dict(SUPPORTED=2,UNSUPPORTED=0,UNCHECKABLE=6))
        self.assertEqual(result["required_arriving_action_records"],8)
        record = next(r for r in result["records"] if r["native_action_driven_origin_join"])
        self.assertEqual(record["native_edge_support"][0]["status"], "SUPPORTED")
        self.assertFalse(record["supported_prefix_observations"][1]["observed_prefix_match"])
        self.assertFalse(result["branch_transition_validated"])
        self.assertTrue(all(len(r["supported_prefix_observations"]) == r["required_boundaries"] for r in result["records"]))

    def test_late_support_helper_cannot_mutate_owner_or_replay_scope(self):
        for target in ("native_input", "materializer_scope", "replay_scope", "replay_wrapper"):
            fixture, owned, contract = self.owned_fixture(native_driven=True)
            native, refs, replays, reader = owned
            calls = 0
            def helper(before, after, edge_json):
                nonlocal calls
                calls += 1
                raw = self.owned_helper(before, after, edge_json)
                if calls == 3:  # First support callback in the final roots pass.
                    if target == "native_input": native["evidence"]["consumed_world_inputs"]["calls"][0]["report_sha256"] = "f"*64
                    if target == "materializer_scope": refs[0]["validator"].allow_synthetic = False
                    if target == "replay_scope": replays[0]["validator"].allow_synthetic = False
                    if target == "replay_wrapper": replays[0]["result"]["root_origin"]["native_capture_sha256"] = "f"*64
                return raw
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.join_owners(fixture, owned, contract, helper)
            self.assertGreaterEqual(calls,3)

    def test_real_native_engine_support_is_read_in_python_without_model(self):
        repo = Path(__file__).resolve().parents[1]
        completed = subprocess.run(["/opt/homebrew/opt/rustup/bin/cargo", "test", "--offline",
            "--no-default-features", "--lib", "native_edge::tests::reader_fixture", "--", "--nocapture"],
            cwd=repo/"rust"/"pokezero-search",capture_output=True,text=True,timeout=60)
        self.assertEqual(completed.returncode,0,completed.stderr[-3000:])
        fixtures = [json.loads(line.split("NATIVE_EDGE_READER_FIXTURE:",1)[1]) for line in completed.stdout.splitlines()
            if "NATIVE_EDGE_READER_FIXTURE:" in line]
        self.assertEqual(len(fixtures),1)
        fixture = fixtures[0]
        checked = read_native_edge_support(json.dumps(fixture["support"]),fixture["step"])
        self.assertTrue(checked["recorded_edge_supported"])
        self.assertFalse(checked["independent_physics_validated"])


class NativeSuccessorTests(unittest.TestCase):
    """Real offline native toy exports; reference AFTER views remain authored."""
    @classmethod
    def setUpClass(cls):
        repo=Path(__file__).resolve().parents[1]
        completed=subprocess.run(["/opt/homebrew/opt/rustup/bin/cargo","test","--offline","--no-default-features",
            "--lib","native_edge::tests::successor_reader_fixture","--","--nocapture"],cwd=repo/"rust/pokezero-search",
            capture_output=True,text=True,timeout=60)
        if completed.returncode:raise AssertionError(completed.stderr[-3000:])
        fixtures=[json.loads(line.split("NATIVE_SUCCESSOR_READER_FIXTURE:",1)[1]) for line in completed.stdout.splitlines()
            if "NATIVE_SUCCESSOR_READER_FIXTURE:" in line]
        if len(fixtures)!=1:raise AssertionError("actual native fixture absent")
        cls.fixture=fixtures[0]

    def case(self,branch_index=0):
        step=deepcopy(self.fixture["step"])
        support=read_native_successor_support(json.dumps(self.fixture["support"]),step)
        chosen=next(r for r in support["positive_successors"] if r["branch_index"]==branch_index)
        projection=project_native_boundary(chosen["boundary"])
        pre=project_native_boundary(support["prestate_boundary"])
        binding=dict(original="authored-scoped-comparison-coordinate")
        obs=[dict(ordinal=0,observed_required_fields={k:"MATCH" for k in ENDPOINT_FIELDS}),
            dict(ordinal=1,observed_required_fields={**{k:"MATCH" for k in ENDPOINT_FIELDS},"action_identity":"MATCH"},
                reference_projection=dict(showdown=projection),reference_snapshot_sha256="a"*64)]
        native=[dict(ordinal=0,native=pre,binding=binding),dict(ordinal=1,native={},binding=binding)]
        rows=[dict(ordinal=1,binding=binding,support_report=support)]
        return obs,native,rows

    def compare(self,case,**kwargs):
        return compare_projected_successors(*case,**{"native_action_driven_origin_join":True,**kwargs})

    def test_support_prefix_cannot_recover_after_earlier_gap(self):
        case=self.case();obs,native,rows=case
        obs.append(deepcopy(obs[1]));obs[2]["ordinal"]=2
        native[1]["native"]=deepcopy(native[0]["native"])
        native.append(deepcopy(native[1]));native[2]["ordinal"]=2
        rows.append(deepcopy(rows[0]));rows[1]["ordinal"]=2
        obs[0]["observed_required_fields"]["status"]="UNCHECKABLE"
        result=self.compare(case)
        self.assertEqual(result[0]["status"],"UNCHECKABLE")
        self.assertEqual(result[1]["status"],"WITHIN_PROJECTED_NATIVE_SUPPORT")
        self.assertFalse(result[1]["scoped_support_prefix_checked"])
        self.assertEqual(result[1]["first_uncertain_or_outside_support_ordinal"],1)

    def test_real_complete_roster_read_and_sampled_branch_agreement(self):
        obs,native,rows=self.case();result=self.compare((obs,native,rows))[0]
        self.assertEqual(result["status"],"WITHIN_PROJECTED_NATIVE_SUPPORT")
        self.assertIn(0,result["matching_branch_indices"])
        self.assertEqual(result["checked_positive_native_branches"],len(rows[0]["support_report"]["positive_successors"]))
        self.assertFalse(result["branch_transition_validated"])
        self.assertFalse(result["matching_mass_is_cross_engine_probability"])

    def test_different_supported_random_branch_does_not_become_divergence(self):
        support=self.fixture["support"]
        first=project_native_boundary(support["positive_successors"][0]["boundary"])
        other=next(r for r in support["positive_successors"]
            if any(digest(project_native_boundary(r["boundary"])[k])!=digest(first[k]) for k in ENDPOINT_FIELDS))
        case=self.case(other["branch_index"])
        # Endpoint equality with the sampled native branch would fail; support
        # membership checks the full roster instead, with no same-RNG premise.
        case[0][1]["observed_required_fields"]["hp"]="MISMATCH"
        value=self.compare(case)[0]
        self.assertEqual(value["status"],"WITHIN_PROJECTED_NATIVE_SUPPORT")
        self.assertIn(other["branch_index"],value["matching_branch_indices"])
        self.assertFalse(value["equal_rng_coupling"])

    def test_genuinely_outside_scoped_support_and_pp_is_not_required(self):
        case=self.case();case[0][1]["reference_projection"]["showdown"]["hp"]["p1"][1]+=1
        self.assertEqual(self.compare(case)[0]["status"],"OUTSIDE_PROJECTED_NATIVE_SUPPORT")
        case=self.case();case[0][1]["reference_projection"]["showdown"]["native_pp"]["p1"][0]["pp"]+=100
        value=self.compare(case)[0]
        self.assertEqual(value["status"],"WITHIN_PROJECTED_NATIVE_SUPPORT")
        self.assertEqual(value["native_pp_class"],"engine_pp_model")

    def test_missing_prestate_action_after_or_support_remains_uncertain(self):
        for mode in ("pre","action","after","support","origin","coordinate"):
            case=self.case()
            if mode=="pre":case[0][0]["observed_required_fields"]["status"]="UNCHECKABLE"
            if mode=="action":case[0][1]["observed_required_fields"]["action_identity"]="UNCHECKABLE"
            if mode=="after":case[0][1]["reference_projection"]["showdown"]["effective_legal_choices"]=None
            if mode=="support":case[2][0]["support_report"]=None
            if mode=="coordinate":case[2][0]["binding"]={}
            if mode=="coordinate":
                with self.assertRaises(ValueError):self.compare(case)
            else:
                value=self.compare(case,native_action_driven_origin_join=mode!="origin")[0]
                with self.subTest(mode=mode):
                    self.assertEqual(value["status"],"UNCHECKABLE");self.assertIsNone(value["matching_native_percentage"])

    def test_duplicate_projected_branches_preserve_mass_not_extra_transitions(self):
        case=self.case();report_value=case[2][0]["support_report"]
        report_value["positive_successors"][1]["boundary"]=deepcopy(report_value["positive_successors"][0]["boundary"])
        # This unit compares already-read projections. Completeness/raw binding
        # reader is separately tested; identical projection is legal in actual
        # exports even when native states differ outside scoped required fields.
        result=self.compare(case)
        self.assertEqual(len(result),1)
        self.assertTrue({0,1}<=set(result[0]["matching_branch_indices"]))
        self.assertEqual(result[0]["matching_native_percentage"],sum(r["percentage"] for r in report_value["positive_successors"]
            if r["branch_index"] in result[0]["matching_branch_indices"]))

    def test_full_roster_mass_state_and_projection_tampering_rejected(self):
        step=self.fixture["step"]
        for mode in ("drop","duplicate","mass","allmass","index","state","pre","scope","unknown","zero"):
            value=deepcopy(self.fixture["support"])
            if mode=="drop":value["positive_successors"].pop()
            if mode=="duplicate":value["positive_successors"].append(deepcopy(value["positive_successors"][0]))
            if mode=="mass":value["positive_successors"][0]["percentage"]+=1
            if mode=="allmass":value["branch_masses"][0]["percentage"]=float("nan")
            if mode=="index":value["branch_masses"][0]["branch_index"]=True
            if mode=="state":value["positive_successors"][0]["state_utf8"]+="changed"
            if mode=="pre":value["prestate_boundary"]["state_utf8_sha256"]="a"*64
            if mode=="scope":value["equal_rng_coupling"]=True
            if mode=="unknown":value["extra"]=0
            if mode=="zero":value["branch_masses"][0]["percentage"]=0
            with self.subTest(mode=mode),self.assertRaises(ValueError):read_native_successor_support(json.dumps(value),step)

    def test_missing_legacy_support_does_not_infer_slots_or_retry(self):
        value=capture();helper=Mock()
        rows=extract_native_successor_support(value,root_binding=ROOT,selection_report_sha256=REPORT,native_successor_support=helper)
        helper.assert_not_called();self.assertEqual(len(rows),2)
        self.assertTrue(all(r["status"]=="UNCHECKABLE" for r in rows))
        helper=Mock(side_effect=RuntimeError("authored failure"))
        with self.assertRaisesRegex(RuntimeError,"authored failure"):
            extract_native_successor_support(prospective_capture(),root_binding=ROOT,selection_report_sha256=REPORT,native_successor_support=helper)
        helper.assert_called_once()

    def test_owner_join_keeps_full_missing_roster_and_late_mutation_rejected(self):
        helpers=NativeEdgeTests(methodName="runTest");self.addCleanup(helpers.doCleanups)
        fixture,owned,contract=helpers.owned_fixture(native_driven=True)
        native,refs,replays,reader=owned
        def exporter(before,after,edge_json):
            edge=json.loads(edge_json);index=edge["branch_index"];count=index+1
            step=dict(before=before,after=after,key=[0,index],depth=0 if edge["root"] else 1,
                branch_on_damage=edge["branch_on_damage"],native_edge=edge,
                side_one_choice={k:edge["actions"]["p1"][k] for k in ("kind","engine_id")},
                side_two_choice={k:edge["actions"]["p2"][k] for k in ("kind","engine_id")})
            old=support_report(step);old["matching_successor_branch_indices"]=list(range(count))
            return json.dumps(dict(schema=SUCCESSOR_SCHEMA,captured_support=old,
                prestate_boundary=json.loads(fixture.extractor(before,edge["root"])),
                branch_masses=[dict(branch_index=i,percentage=100./count) for i in range(count)],
                positive_successors=[dict(branch_index=i,percentage=100./count,state_utf8=after,
                    boundary=json.loads(fixture.extractor(after,False))) for i in range(count)],
                complete_positive_mass_roster=True,scope=SUCCESSOR_SCOPE,private_controller_only=True,
                independent_physics_validated=False,showdown_transition_validated=False,equal_rng_coupling=False,
                runtime_qualified=False,scientific_collection_authorized=False))
        def join(helper):
            return join_owned_raw_replays(native,refs,replays,native_validator=reader,ancestry_contract=contract,
                native_extract=fixture.extractor,native_support=helpers.owned_helper,native_successor_support=helper)
        result=join(exporter)
        counts=result["projected_successor_support_counts"]
        self.assertEqual(sum(counts.values()),result["required_arriving_action_records"])
        self.assertEqual(result["required_arriving_action_records"],8)
        self.assertGreaterEqual(counts["UNCHECKABLE"],6)
        self.assertFalse(result["branch_transition_validated"])
        from pokezero.mcts_eval.search_over_raw_consumed_materialization import reconcile_owned_replay_diagnostic
        diagnostic=reconcile_owned_replay_diagnostic(native,refs,replays,native_validator=reader,
            ancestry_contract=contract,native_extract=fixture.extractor,native_support=helpers.owned_helper,
            native_successor_support=exporter)
        self.assertEqual(diagnostic["owned_replay_join"]["projected_successor_support_counts"],counts)
        self.assertEqual(diagnostic["costed_unique_jobs"],3)
        self.assertFalse(diagnostic["resource_cost_complete"])
        self.assertEqual(diagnostic["deployment_accepted_units_added"],0)
        calls=0
        def changed(before,after,edge_json):
            nonlocal calls
            calls+=1;raw=exporter(before,after,edge_json)
            if calls==3:replays[0]["validator"].allow_synthetic=False
            return raw
        with self.assertRaises(ValueError):join(changed)
        self.assertGreaterEqual(calls,3)

    def test_archived_diagnostic_forwards_helpers_through_final_join_without_filling_legacy(self):
        attachment=owner_fixture.ArchivedDiagnosticAttachmentTests(methodName="runTest")
        self.addCleanup(attachment.doCleanups)
        archive,selected,reader=attachment.fixture()
        first=Mock(side_effect=AssertionError("legacy capture cannot infer slots"))
        second=Mock(side_effect=AssertionError("legacy capture cannot infer successors"))
        from pokezero.mcts_eval.search_over_raw_consumed_materialization import reconcile_archived_exploration_diagnostic
        result=reconcile_archived_exploration_diagnostic(selected,[],[],archive=archive.archive,
            source=archive.source,contract=archive.contract,root_id=archive.root_id,native_validator=reader,
            ancestry_contract=owner_fixture.CONTRACT,native_extract=owner_fixture.OwnedAncestryRootJoinTests.extractor,
            native_support=first,native_successor_support=second)
        first.assert_not_called();second.assert_not_called()
        counts=result["owned_replay_join"]["projected_successor_support_counts"]
        self.assertEqual(counts["UNCHECKABLE"],result["owned_replay_join"]["required_arriving_action_records"])
        self.assertTrue(result["root_to_panel_attachment_validated"])
        self.assertFalse(result["qualification_changed"])
        self.assertFalse(result["scientific_collection_authorized"])


if __name__ == "__main__": unittest.main()
