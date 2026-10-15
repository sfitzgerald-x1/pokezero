"""Authored boundary records only; no native runtime, source games or models."""
from copy import deepcopy
import hashlib
import unittest

from pokezero.mcts_eval.search_over_raw_native_boundary_records import (
    SCHEMA, FLAGS, _state_sha, digest, read_native_boundary, project_native_boundary,
)


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


class NativeBoundaryRecordTests(unittest.TestCase):
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

    def test_every_scope_promotion_and_side_field_drift_refuses(self):
        for flag in FLAGS:
            value = report()
            value[flag] = True
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                read(value)
        for field in report()["p1"]:
            value = report()
            del value["p1"][field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                read(value)
        value = report()
        value["p2"]["independently_validated"] = True
        with self.assertRaises(ValueError):
            read(value)

    def test_reader_returns_a_detached_record_without_changing_input(self):
        value = report()
        before = deepcopy(value)
        checked = read(value)
        checked["p1"]["hp"][0] = 0
        checked["p2"]["active_moves"][0]["pp"] = 0
        self.assertEqual(value, before)
        self.assertIsNot(checked, value)

    def test_projection_is_detached_and_never_infers_action_or_transition(self):
        value = report()
        side = value["p1"]
        side["active_moves"][1] = dict(slot=1,engine_id="tackle",pp=32,disabled=False)
        move = dict(kind="move",slot=1,engine_id="tackle")
        for field in ("engine_options", "search_arms", "effective_legal_choices"):
            side[field].insert(0, deepcopy(move))
        checked = read(value)
        before = deepcopy(checked)
        projection = project_native_boundary(checked)
        self.assertEqual(set(projection), {"active_identities", "hp", "fainted", "status",
            "effective_legal_choices", "native_pp", "action_identity", "branch_transition"})
        self.assertIsNone(projection["action_identity"])
        self.assertIsNone(projection["branch_transition"])
        self.assertEqual([choice["slot"] for choice in projection["effective_legal_choices"]["p1"]], [0, 1])
        self.assertEqual(projection["native_pp"]["p1"][1], dict(slot=1,engine_id="tackle",pp=32))
        projection["active_identities"]["p1"]["engine_id"] = "changed"
        projection["hp"]["p1"][0] = 0
        projection["effective_legal_choices"]["p1"][0]["slot"] = 3
        projection["native_pp"]["p1"][0]["pp"] = 0
        self.assertEqual(checked, before)

    def test_exact_state_text_root_and_canonical_digest_laws(self):
        text = "state:é"
        self.assertEqual(_state_sha(text), hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(read(report(text), state_text=text)["state_utf8_sha256"], _state_sha(text))
        for bad in ("", None, False, 1):
            with self.subTest(state=bad), self.assertRaises(ValueError):
                _state_sha(bad)
        for bad in (False, 1, "true", None):
            with self.subTest(root=bad), self.assertRaises(ValueError):
                read(report(), root=bad)
        self.assertEqual(digest({"b": 1, "a": "é"}),
            hashlib.sha256(b'{"a":"\\u00e9","b":1}').hexdigest())
        with self.assertRaises(ValueError):
            digest(float("nan"))

    def test_preview_and_empty_root_keep_existing_option_acceptance_laws(self):
        value = report()
        value["team_preview"] = True
        for seat in ("p1", "p2"):
            option = dict(kind="switch", **value[seat]["active_identity"])
            for field in ("engine_options", "search_arms", "effective_legal_choices"):
                value[seat][field] = [deepcopy(option)]
        self.assertEqual(read(value), value)
        value["team_preview"] = False
        with self.assertRaises(ValueError):
            read(value)
        value = report()
        for seat in ("p1", "p2"):
            for field in ("engine_options", "search_arms", "effective_legal_choices"):
                value[seat][field] = []
        value["root_search_admissible"] = False
        self.assertEqual(read(value), value)

    def test_signed_integer_pp_is_retained_not_reinterpreted_as_showdown_pp(self):
        value = report()
        value["p1"]["active_moves"][0]["pp"] = -1
        value["p1"]["active_moves"][1]["pp"] = 700
        checked = read(value)
        self.assertEqual(project_native_boundary(checked)["native_pp"]["p1"][0]["pp"], -1)
        self.assertEqual(checked["p1"]["active_moves"][1]["pp"], 700)
        for bad in (True, 1.0, "15"):
            value = report()
            value["p1"]["active_moves"][0]["pp"] = bad
            with self.subTest(pp=bad), self.assertRaises(ValueError):
                read(value)
