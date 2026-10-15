"""Author-written serialized snapshots only; no simulator, model or source."""
from copy import deepcopy
import unittest

from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_showdown_boundary import freeze_party_mapping, project_showdown_boundary, commands_for_native_actions
from test_search_over_raw_native_boundary import report


def mon(species, move, position=0):
    return dict(position=position,isActive=position==0,set=dict(species=species,moves=[move],name="same nickname"),
        species=f"[Species:{species}]",hp=100,maxhp=100,fainted=False,status="",
        moveSlots=[dict(id=move,pp=15,disabled=False)],hpType="Ice",hpPower=70)


def snapshot():
    sides = [dict(id=seat,team="12",active=[f"[Pokemon:{seat}a]"],
        pokemon=[mon(species,move),mon(bench,"tackle",1)])
        for seat,species,move,bench in (("p1","primeape","seismictoss","charmander"),
            ("p2","blissey","splash","squirtle"))]
    return dict(battle=dict(formatid="gen3customgame",gameType="singles",ended=False,sides=sides),
        boundaryRequests=requests(sides),terminalScheduled=False)


def requests(sides):
    result={}
    for side in sides:
        party=side["pokemon"]
        result[side["id"]]=dict(side=dict(id=side["id"],pokemon=[dict(active=index==0,
            condition="0 fnt" if p["hp"] == 0 else f"{p['hp']}/{p['maxhp']}" +
                (f" {p['status']}" if p["status"] else "")) for index,p in enumerate(party)]),
            active=[dict(moves=[dict(id=v["id"],disabled=v["disabled"]) for v in party[0]["moveSlots"]])])
    return result


def root_native():
    value=report()
    for seat,species in (("p1","charmander"),("p2","squirtle")):
        value[seat]["party_identities"].append(dict(slot=1,engine_id=species))
    return value


def mapping(root=None,native=None,order=None):
    return freeze_party_mapping(root or snapshot(),native or root_native(),
        native_to_original=order or {"p1":[0,1],"p2":[0,1]})


class ShowdownBoundaryTests(unittest.TestCase):
    def test_native_command_resolution_uses_full_slots_and_explicit_wait_or_recharge(self):
        root=snapshot(); frozen=mapping(root)
        root["battle"]["sides"][0]["pokemon"][0]["moveSlots"].append(dict(id="tackle",pp=20,disabled=False))
        root["boundaryRequests"]["p1"]["active"][0]=dict(moves=[dict(id="tackle")],trapped=True)
        root["boundaryRequests"]["p2"].pop("active");root["boundaryRequests"]["p2"]["wait"]=True
        actions=dict(p1=dict(kind="move",slot=1,engine_id="tackle"),p2=dict(kind="none",slot=None,engine_id="No Move"))
        self.assertEqual(commands_for_native_actions(root,frozen,actions),dict(p1="move 1"))
        root["boundaryRequests"]["p2"].pop("wait")
        root["boundaryRequests"]["p2"]["active"]=[dict(moves=[dict(id="recharge")],trapped=True)]
        self.assertEqual(commands_for_native_actions(root,frozen,actions),dict(p1="move 1",p2="move 1"))
        actions["p1"]=dict(kind="switch",slot=1,engine_id="charmander")
        with self.assertRaises(ValueError):commands_for_native_actions(root,frozen,actions)

    def test_native_command_resolution_rejects_absent_hidden_ambiguous_and_terminal_requests(self):
        actions=dict(p1=dict(kind="move",slot=0,engine_id="seismictoss"),p2=dict(kind="move",slot=0,engine_id="splash"))
        for mode in ("missing","withheld","duplicate","terminal","wrong-slot","wrong-id","unknown-action"):
            root=snapshot();frozen=mapping(root); target=deepcopy(actions)
            if mode=="missing":root["boundaryRequests"].pop("p2")
            if mode=="withheld":root["boundaryRequests"]["p2"]["active"][0]["maybeTrapped"]=True
            if mode=="duplicate":root["battle"]["sides"][0]["pokemon"][0]["moveSlots"].append(dict(id="seismictoss",pp=1,disabled=False))
            if mode=="terminal":root["battle"]["ended"]=True;root["boundaryRequests"]={}
            if mode=="wrong-slot":target["p1"]["slot"]=1
            if mode=="wrong-id":target["p2"]["engine_id"]="wrong"
            if mode=="unknown-action":target=None
            with self.subTest(mode=mode),self.assertRaises(ValueError):commands_for_native_actions(root,frozen,target)

    def test_native_switch_command_uses_current_index_after_party_reorder(self):
        root=snapshot();frozen=mapping(root);after=deepcopy(root)
        side=after["battle"]["sides"][0];side["pokemon"].reverse();side["team"]="21"
        for current,p in enumerate(side["pokemon"]):p.update(position=current,isActive=current==0)
        after["boundaryRequests"]=requests(after["battle"]["sides"])
        actions=dict(p1=dict(kind="switch",slot=0,engine_id="primeape"),p2=dict(kind="move",slot=0,engine_id="splash"))
        self.assertEqual(commands_for_native_actions(after,frozen,actions),dict(p1="switch 2",p2="move 1"))

    def test_actual_snapshot_projection_full_catalog_and_no_provenance_promotion(self):
        root=snapshot(); frozen=mapping(root); before=deepcopy((root,frozen))
        result=project_showdown_boundary(root,frozen)
        fields=result["showdown"]
        self.assertEqual(fields["active_identities"]["p1"],dict(slot=0,engine_id="primeape"))
        self.assertEqual(fields["hp"]["p1"],[100,100])
        self.assertEqual(fields["effective_legal_choices"]["p1"],[dict(kind="move",slot=0,engine_id="seismictoss"),
            dict(kind="switch",slot=1,engine_id="charmander")])
        self.assertEqual(fields["native_pp"]["p1"][0],dict(slot=0,engine_id="seismictoss",pp=15))
        for flag in ("independently_validated","snapshot_provenance_validated","construction_provenance_validated",
                     "branch_transition_validated","runtime_qualified","scientific_collection_authorized"):
            self.assertFalse(result[flag])
        self.assertIsNone(fields["action_identity"]); self.assertIsNone(fields["branch_transition"])
        self.assertEqual((root,frozen),before)

    def test_party_reorder_uses_original_permutation_not_species_or_nickname(self):
        root=snapshot(); frozen=mapping(root); after=deepcopy(root)
        side=after["battle"]["sides"][0]; side["pokemon"].reverse(); side["team"]="21"
        for current,p in enumerate(side["pokemon"]):p["position"],p["isActive"]=current,current==0
        after["boundaryRequests"]=requests(after["battle"]["sides"])
        fields=project_showdown_boundary(after,frozen)["showdown"]
        self.assertEqual(fields["active_identities"]["p1"],dict(slot=1,engine_id="charmander"))
        self.assertIn(dict(kind="switch",slot=0,engine_id="primeape"),fields["effective_legal_choices"]["p1"])

    def test_explicit_native_order_can_differ_from_original_order(self):
        root=snapshot(); native=root_native(); native["p1"]["party_identities"].reverse()
        for index,p in enumerate(native["p1"]["party_identities"]):p["slot"]=index
        frozen=mapping(root,native,{"p1":[1,0],"p2":[0,1]})
        fields=project_showdown_boundary(root,frozen)["showdown"]
        self.assertEqual(fields["active_identities"]["p1"]["slot"],1)
        self.assertIn(dict(kind="switch",slot=0,engine_id="charmander"),fields["effective_legal_choices"]["p1"])

    def test_duplicate_species_remain_distinct_with_explicit_construction_slots(self):
        root=snapshot(); native=root_native()
        root["battle"]["sides"][0]["pokemon"][1]["species"]="[Species:primeape]"
        root["battle"]["sides"][0]["pokemon"][1]["set"]["species"]="primeape"
        native["p1"]["party_identities"][1]["engine_id"]="primeape"
        frozen=mapping(root,native)
        fields=project_showdown_boundary(root,frozen)["showdown"]
        self.assertEqual(fields["active_identities"]["p1"]["slot"],0)
        self.assertIn(dict(kind="switch",slot=1,engine_id="primeape"),fields["effective_legal_choices"]["p1"])

    def test_form_or_transform_changes_state_not_original_identity(self):
        root=snapshot(); frozen=mapping(root); after=deepcopy(root)
        after["battle"]["sides"][0]["pokemon"][0]["species"]="[Species:blissey]"
        result=project_showdown_boundary(after,frozen)
        self.assertEqual(result["showdown"]["active_identities"]["p1"],dict(slot=0,engine_id="blissey"))
        after["battle"]["sides"][0]["pokemon"][0]["set"]["species"]="blissey"
        with self.assertRaisesRegex(ValueError,"original-team identity"):project_showdown_boundary(after,frozen)

    def test_disabled_trapped_and_full_set_not_chosen_only(self):
        root=snapshot(); frozen=mapping(root)
        root["battle"]["sides"][0]["pokemon"][0]["moveSlots"].append(dict(id="tackle",pp=20,disabled=True))
        root["boundaryRequests"]=requests(root["battle"]["sides"])
        root["boundaryRequests"]["p1"]["active"][0]["trapped"]=True
        choices=project_showdown_boundary(root,frozen)["showdown"]["effective_legal_choices"]["p1"]
        self.assertEqual(choices,[dict(kind="move",slot=0,engine_id="seismictoss")])

    def test_locked_request_uses_full_move_slot_not_request_index(self):
        root=snapshot(); frozen=mapping(root)
        mon=root["battle"]["sides"][0]["pokemon"][0]
        mon["moveSlots"].append(dict(id="tackle",pp=20,disabled=False))
        root["boundaryRequests"]["p1"]["active"][0]=dict(moves=[dict(id="tackle")],trapped=True)
        fields=project_showdown_boundary(root,frozen,before_snapshot=root,
            incoming_choices={"p1":"move 1","p2":"move 1"})["showdown"]
        self.assertEqual(fields["effective_legal_choices"]["p1"],[dict(kind="move",slot=1,engine_id="tackle")])
        self.assertEqual(fields["action_identity"]["p1"]["slot"],1)

    def test_struggle_recharge_wait_and_replacement_are_distinct(self):
        for move,kind,engine in (("struggle","struggle","struggle"),("recharge","none","No Move")):
            root=snapshot(); frozen=mapping(root)
            root["boundaryRequests"]["p1"]["active"][0]=dict(moves=[dict(id=move)],trapped=True)
            fields=project_showdown_boundary(root,frozen)["showdown"]
            self.assertEqual(fields["effective_legal_choices"]["p1"],[dict(kind=kind,slot=None,engine_id=engine)])
        root=snapshot(); frozen=mapping(root)
        one=root["boundaryRequests"]["p1"]; one.pop("active"); one["forceSwitch"]=[True]
        two=root["boundaryRequests"]["p2"]; two.pop("active"); two["wait"]=True
        fields=project_showdown_boundary(root,frozen,before_snapshot=root,incoming_choices={"p1":"switch 2"})["showdown"]
        self.assertEqual(fields["effective_legal_choices"]["p1"],[dict(kind="switch",slot=1,engine_id="charmander")])
        self.assertEqual(fields["effective_legal_choices"]["p2"],[dict(kind="none",slot=None,engine_id="No Move")])
        self.assertEqual(fields["action_identity"]["p2"]["kind"],"none")

    def test_terminal_not_a_win_draw_label_or_pending_terminal(self):
        root=snapshot(); frozen=mapping(root)
        root["terminalScheduled"]=True
        self.assertFalse(project_showdown_boundary(root,frozen)["reference_terminal"])
        root["battle"]["ended"]=True; root["boundaryRequests"]={}
        result=project_showdown_boundary(root,frozen)
        self.assertTrue(result["reference_terminal"])
        self.assertEqual(result["showdown"]["effective_legal_choices"],dict(p1=[],p2=[]))
        self.assertIsNone(result["showdown"]["branch_transition"])
        self.assertNotIn("score",result)

    def test_fainted_position_zero_and_pending_faint_use_actual_pinned_laws(self):
        root=snapshot(); frozen=mapping(root)
        mon=root["battle"]["sides"][0]["pokemon"][0]
        mon["hp"]=0; mon["fainted"]=True; mon["isActive"]=False
        root["boundaryRequests"]=requests(root["battle"]["sides"])
        one=root["boundaryRequests"]["p1"]; one.pop("active"); one["forceSwitch"]=[True]
        result=project_showdown_boundary(root,frozen)
        self.assertTrue(result["showdown"]["fainted"]["p1"])
        self.assertTrue(one["side"]["pokemon"][0]["active"])
        self.assertEqual(result["showdown"]["effective_legal_choices"]["p1"],
            [dict(kind="switch",slot=1,engine_id="charmander")])
        root["battle"]["ended"]=True
        self.assertTrue(project_showdown_boundary(root,frozen)["reference_terminal"])
        root["battle"]["ended"]=False; mon["fainted"]=False; mon["isActive"]=True
        self.assertFalse(project_showdown_boundary(root,frozen)["showdown"]["fainted"]["p1"])
        mon["isActive"]=False
        with self.assertRaises(ValueError):project_showdown_boundary(root,frozen)

    def test_hidden_or_missing_legality_and_ambiguous_slot_stay_unknown(self):
        for mode in ("missing","maybe","duplicate","preview"):
            root=snapshot(); frozen=mapping(root)
            if mode=="missing":del root["boundaryRequests"]["p1"]
            if mode=="maybe":root["boundaryRequests"]["p1"]["active"][0]["maybeTrapped"]=True
            if mode=="duplicate":root["battle"]["sides"][0]["pokemon"][0]["moveSlots"].append(
                dict(id="seismictoss",pp=0,disabled=True))
            if mode=="preview":root["boundaryRequests"]["p1"]=dict(teamPreview=True,
                side=root["boundaryRequests"]["p1"]["side"])
            result=project_showdown_boundary(root,frozen)
            self.assertIsNone(result["showdown"]["effective_legal_choices"])
            self.assertTrue(result["limitations"]["p1"])

    def test_hidden_power_uses_actual_private_type_and_power(self):
        root=snapshot(); frozen=mapping(root)
        mon=root["battle"]["sides"][0]["pokemon"][0]; mon["moveSlots"][0]["id"]="hiddenpower"
        root["boundaryRequests"]=requests(root["battle"]["sides"])
        fields=project_showdown_boundary(root,frozen)["showdown"]
        self.assertEqual(fields["effective_legal_choices"]["p1"][0]["engine_id"],"hiddenpowerice70")
        mon["hpPower"]=False
        with self.assertRaises(ValueError):project_showdown_boundary(root,frozen)

    def test_request_snapshot_health_or_party_drift_never_repaired(self):
        root=snapshot(); frozen=mapping(root)
        for mode in ("health","active","status","size","id","disabled","slot","format","position","set","perm"):
            bad=deepcopy(root)
            if mode=="health":bad["boundaryRequests"]["p1"]["side"]["pokemon"][0]["condition"]="99/100"
            if mode=="active":bad["boundaryRequests"]["p1"]["side"]["pokemon"][0]["active"]=False
            if mode=="status":bad["battle"]["sides"][0]["pokemon"][0]["status"]="unknown"
            if mode=="size":bad["boundaryRequests"]["p1"]["side"]["pokemon"].pop()
            if mode=="id":bad["boundaryRequests"]["p1"]["side"]["id"]="p2"
            if mode=="disabled":bad["boundaryRequests"]["p1"]["active"][0]["moves"][0]["disabled"]="hidden"
            if mode=="slot":bad["battle"]["sides"][0]["active"]=["[Pokemon:p1b]"]
            if mode=="format":bad["battle"]["gameType"]="doubles"
            if mode=="position":bad["battle"]["sides"][0]["pokemon"][0]["position"]=False
            if mode=="set":bad["battle"]["sides"][0]["pokemon"][1]["set"]["moves"].append("splash")
            if mode=="perm":bad["battle"]["sides"][0]["team"]="11"
            with self.subTest(mode=mode),self.assertRaises(ValueError):project_showdown_boundary(bad,frozen)

    def test_map_omission_alias_substitution_and_promotion_rejected(self):
        root=snapshot(); native=root_native()
        for order in ({"p1":[0,0],"p2":[0,1]}, {"p1":[False,1],"p2":[0,1]},
                      {"p1":[1,0],"p2":[0,1]}, {"p1":[0,1]}):
            with self.assertRaises(ValueError):freeze_party_mapping(root,native,native_to_original=order)
        frozen=mapping(root); frozen["construction_provenance_validated"]=True
        with self.assertRaises(ValueError):project_showdown_boundary(root,frozen)

    def test_invalid_or_extra_incoming_choice_cannot_be_certified(self):
        root=snapshot(); frozen=mapping(root)
        result=project_showdown_boundary(root,frozen,before_snapshot=root,incoming_choices={"p1":"move 4","p2":"move 1"})
        self.assertIsNone(result["showdown"]["action_identity"])
        with self.assertRaises(ValueError):project_showdown_boundary(root,frozen,before_snapshot=root,
            incoming_choices={"p1":"move 1","p2":"move 1","p3":"move 1"})


if __name__ == "__main__":unittest.main()
