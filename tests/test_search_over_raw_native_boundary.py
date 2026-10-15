"""Synthetic exports and an author-written Rust toy; no source games or model."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import unittest

from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_correspondence import _state_sha
from pokezero.mcts_eval.search_over_raw_native_boundary import (
    SCHEMA, FLAGS, extract_native_capture_boundaries, read_native_boundary,
)
from test_search_over_raw_correspondence import capture, ROOT, REPORT


def report(text="state:0", root=True):
    def side(species, move):
        options=[dict(kind="move",slot=0,engine_id=move)]
        return dict(active_identity=dict(slot=0,engine_id=species),hp=[100,100],fainted=False,
            status="none",active_moves=[dict(slot=slot,engine_id=move if slot==0 else "none",
                pp=15 if slot==0 else 0,disabled=slot!=0) for slot in range(4)],
            party_identities=[dict(slot=0,engine_id=species)],engine_options=deepcopy(options),
            search_arms=deepcopy(options),effective_legal_choices=deepcopy(options),
            interior_empty_fallback=False,force_switch=False,force_trapped=False,slow_uturn_move=False)
    return dict(schema=SCHEMA,state_utf8_sha256=_state_sha(text),root=root,team_preview=False,
        native_outcome=0,terminal=False,terminal_choice_law="no_new_choices_at_terminal",
        root_search_admissible=True,p1=side("primeape","seismictoss"),p2=side("blissey","splash"),
        native_pp_class="engine_pp_model",**{flag:False for flag in FLAGS})


def read(value, **kwargs):
    return read_native_boundary(value,**dict(dict(state_text="state:0",root=True),**kwargs))


class NativeBoundaryTests(unittest.TestCase):
    def test_consumers_share_prospectively_bound_record_kernel(self):
        from pokezero.mcts_eval import search_over_raw_native_boundary as wrapper
        from pokezero.mcts_eval import search_over_raw_native_boundary_records as kernel
        from pokezero.mcts_eval import search_over_raw_native_edge as edge
        from pokezero.mcts_eval.search_over_raw_reconciled import controller_input_hashes
        import hashlib
        self.assertIs(wrapper.read_native_boundary, kernel.read_native_boundary)
        self.assertIs(wrapper.project_native_boundary, kernel.project_native_boundary)
        self.assertIs(edge._choice, kernel._choice)
        path = Path(kernel.__file__).resolve()
        self.assertEqual(controller_input_hashes()[str(path)], hashlib.sha256(path.read_bytes()).hexdigest())

    def extract(self, private=None, exporter=None):
        return extract_native_capture_boundaries(capture() if private is None else private,
            root_binding=ROOT,selection_report_sha256=REPORT,
            native_extract=exporter or (lambda state,root:json.dumps(report(state,root))))

    def test_actual_native_export_bindings_and_scope(self):
        value=report(); before=deepcopy(value)
        self.assertEqual(read(value),value); self.assertEqual(value,before)
        for mutation in ("root","state","scope","schema","ppclass","terminal","outcome","extra"):
            bad=deepcopy(value)
            if mutation=="root":bad["root"]=False
            if mutation=="state":bad["state_utf8_sha256"]="c"*64
            if mutation=="scope":bad["party_mapping_validated"]=True
            if mutation=="schema":bad["schema"]="old"
            if mutation=="ppclass":bad["native_pp_class"]="bug"
            if mutation=="terminal":bad["terminal"]=True
            if mutation=="outcome":bad["native_outcome"]=True
            if mutation=="extra":bad["full_engine_equivalence"]=True
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):read(bad)

    def test_every_actual_boundary_requested_in_order_no_transition_inference(self):
        calls=[]
        def exporter(state,root): calls.append((state,root)); return json.dumps(report(state,root))
        rows=self.extract(exporter=exporter)
        self.assertEqual(calls,[("state:0",True),("state:1",False),("state:2",False)])
        self.assertEqual([v["ordinal"] for v in rows],[0,1,2])
        self.assertIsNone(rows[0]["native"]["action_identity"])
        self.assertEqual(rows[1]["native"]["action_identity"]["p1"],
            dict(kind="move",slot=0,engine_id="seismictoss"))
        self.assertTrue(all(v["native"]["branch_transition"] is None for v in rows))
        self.assertTrue(all(not v["independently_validated"] and not v["party_mapping_validated"] for v in rows))
        self.assertEqual(rows[1]["binding"]["after_sha256"],_state_sha("state:1"))

    def test_duplicate_move_and_switch_names_leave_action_identity_unknown(self):
        for kind in ("move","switch"):
            private=capture()
            if kind=="switch": private["steps"][0]["side_one_choice"]=dict(kind="switch",engine_id="primeape")
            def exporter(state,root):
                value=report(state,root); side=value["p1"]
                if kind=="move":
                    side["active_moves"][1]=dict(slot=1,engine_id="seismictoss",pp=0,disabled=True)
                    # Even one legal slot cannot recover identity lost by a full-catalog capture.
                else:
                    side["party_identities"] += [dict(slot=slot,engine_id="primeape") for slot in (1,2)]
                    option=dict(kind="switch",slot=1,engine_id="primeape")
                    for key in ("engine_options","search_arms","effective_legal_choices"):side[key].append(option.copy())
                return json.dumps(value)
            rows=self.extract(private=private,exporter=exporter)
            self.assertIsNone(rows[1]["native"]["action_identity"])

    def test_illegal_or_unresolvable_incoming_choice_not_certified(self):
        private=capture(); private["steps"][0]["side_one_choice"]["engine_id"]="madeup"
        self.assertIsNone(self.extract(private=private)[1]["native"]["action_identity"])

    def test_preview_active_switch_and_duplicate_catalog_resolution(self):
        private=capture()
        private["steps"][0]["side_one_choice"]=dict(kind="switch",engine_id="primeape")
        private["steps"][0]["side_two_choice"]=dict(kind="switch",engine_id="blissey")
        def exporter(state,root, duplicate=False):
            value=report(state,root)
            if root:
                value["team_preview"]=True
                for seat in ("p1","p2"):
                    side=value[seat]; option=dict(kind="switch",**side["active_identity"])
                    for key in ("engine_options","search_arms","effective_legal_choices"):side[key]=[option.copy()]
                if duplicate:
                    side=value["p1"]; side["party_identities"].append(dict(slot=1,engine_id="primeape"))
                    for key in ("engine_options","search_arms","effective_legal_choices"):
                        side[key].append(dict(kind="switch",slot=1,engine_id="primeape"))
            return json.dumps(value)
        rows=self.extract(private=private,exporter=exporter)
        self.assertEqual(rows[1]["native"]["action_identity"]["p1"]["slot"],0)
        rows=self.extract(private=private,exporter=lambda state,root:exporter(state,root,True))
        self.assertIsNone(rows[1]["native"]["action_identity"])

    def test_full_effective_set_not_replaced_by_selected_action(self):
        value=report()
        value["p1"]["active_moves"][1]=dict(slot=1,engine_id="tackle",pp=32,disabled=False)
        option=dict(kind="move",slot=1,engine_id="tackle")
        for key in ("engine_options","search_arms","effective_legal_choices"):value["p1"][key].append(option.copy())
        self.assertEqual(len(read(value)["p1"]["effective_legal_choices"]),2)
        value["p1"]["effective_legal_choices"].pop()
        with self.assertRaises(ValueError):read(value)

    def test_terminal_residual_options_and_empty_fallback_remain_separate(self):
        value=report(root=False); value["native_outcome"]=-1; value["terminal"]=True
        for seat in ("p1","p2"):value[seat]["effective_legal_choices"]=[]
        self.assertTrue(read(value,root=False)["p1"]["engine_options"])
        value["native_outcome"]=0; value["terminal"]=False
        for seat in ("p1","p2"):
            value[seat]["engine_options"]=[]; value[seat]["effective_legal_choices"]=[]
            value[seat]["search_arms"]=[dict(kind="none",slot=None,engine_id="No Move")]
            value[seat]["interior_empty_fallback"]=True
        self.assertEqual(read(value,root=False)["p1"]["effective_legal_choices"],[])
        value["p1"]["effective_legal_choices"]=deepcopy(value["p1"]["search_arms"])
        with self.assertRaises(ValueError):read(value,root=False)

    def test_slot_health_status_and_option_substitutions_rejected(self):
        for mutation in ("boolslot","duplicatemove","moveid","switch","struggle","extraoption",
                         "hp","fainted","status","nan","partyslot","active","fallback","rootreject"):
            value=report(); side=value["p1"]
            if mutation=="boolslot":side["active_moves"][0]["slot"]=False
            if mutation=="duplicatemove":side["active_moves"][1]["slot"]=0
            if mutation=="moveid":side["engine_options"][0]["engine_id"]="different"
            if mutation=="switch":side["engine_options"][0]=dict(kind="switch",slot=0,engine_id="primeape")
            if mutation=="struggle":side["engine_options"][0]=dict(kind="struggle",slot=0,engine_id="struggle")
            if mutation=="extraoption":side["engine_options"][0]["chosen"]=True
            if mutation=="hp":side["hp"][0]=101
            if mutation=="fainted":side["fainted"]=0
            if mutation=="status":side["status"]="cured"
            if mutation=="nan":side["hp"][0]=float("nan")
            if mutation=="partyslot":side["party_identities"].append(deepcopy(side["party_identities"][0]))
            if mutation=="active":side["active_identity"]["engine_id"]="different"
            if mutation=="fallback":side["interior_empty_fallback"]=True
            if mutation=="rootreject":value["root_search_admissible"]=False
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):read(value)

    def test_capture_mutation_or_parameter_substitution_invalidates(self):
        private=capture()
        def exporter(state,root):
            private["model_signed_value"]+=.1
            return json.dumps(report(state,root))
        with self.assertRaisesRegex(ValueError,"changed during extraction"):self.extract(private,exporter)
        with self.assertRaises(ValueError):self.extract(exporter=lambda state,root:json.dumps(report()))
        with self.assertRaises(ValueError):self.extract(exporter=lambda state,root:report(state,root))

    def test_real_rust_toy_export_to_python_reader(self):
        repo=Path(__file__).resolve().parents[1]
        completed=subprocess.run(["cargo","test","--offline","--lib",
            "scoped_boundary::tests::reader_fixture","--","--nocapture"],
            cwd=repo/"rust"/"pokezero-search",capture_output=True,text=True,timeout=60)
        self.assertEqual(completed.returncode,0,completed.stderr[-3000:])
        rows=[line.split("NATIVE_BOUNDARY_FIXTURE:",1)[1] for line in completed.stdout.splitlines()
            if "NATIVE_BOUNDARY_FIXTURE:" in line]
        self.assertEqual(len(rows),1)
        value=json.loads(rows[0])
        text=(repo/"rust"/"pokezero-search"/"src"/"test_fixtures"/"minimal.state").read_text().strip()
        checked=read_native_boundary(value,state_text=text,root=True)
        self.assertEqual(len(checked["p1"]["effective_legal_choices"]),2)
        self.assertEqual(checked["p1"]["active_identity"]["engine_id"],"charmander")
        self.assertEqual(digest(checked),digest(value))
        previews=[line.split("NATIVE_PREVIEW_FIXTURE:",1)[1] for line in completed.stdout.splitlines()
            if "NATIVE_PREVIEW_FIXTURE:" in line]
        states=[line.split("NATIVE_PREVIEW_STATE:",1)[1] for line in completed.stdout.splitlines()
            if "NATIVE_PREVIEW_STATE:" in line]
        self.assertEqual((len(previews),len(states)),(1,1))
        preview=read_native_boundary(json.loads(previews[0]),state_text=states[0],root=True)
        self.assertEqual(preview["p1"]["engine_options"][0],
            dict(kind="switch",slot=0,engine_id="charmander"))


if __name__ == "__main__": unittest.main()
