"""Actual Python construction with fake native bindings; no model or battle."""
from copy import deepcopy
from dataclasses import asdict,replace
import json
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import patch

from pokezero.engine_world import battle_spec_from_payload,world_battle_spec
from pokezero.poke_engine_adapter import build_poke_engine_state
from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_construction import (
    NativeConstructionCapture,construction_source_hashes,read_native_construction,derive_party_mapping,
)
from pokezero.mcts_eval.search_over_raw_world_inputs import NativeWorldInputs,NativeInputHandle,validate_fixed_work_inputs
from test_engine_world import _dex,_payload,_override
from test_poke_engine_adapter import fake_construction_module,RecordingState
from test_search_over_raw_world_inputs import BINDING,CONTEXT,arguments


def construct_actual_python(recorder, *, text="complete-state", override=None):
    """Real assembly functions, author-written dex/packed teams, fake State."""
    override=_override() if override is None else override
    recorder.attempt(CONTEXT,len(recorder.rows),override)
    world=battle_spec_from_payload(_payload(_dex()),override,dex=_dex(),construction_observer=recorder.world_built)
    module=fake_construction_module()
    def state_factory(**kwargs):
        state=RecordingState(**kwargs);state.serialized=text
        return state
    module.State=state_factory
    state=build_poke_engine_state(world.spec,module=module,construction_observer=recorder.native_built)
    recorder.constructed(CONTEXT,world,state)
    return world,state


def witness():
    recorder=NativeWorldInputs(BINDING,16,record_atomic_work=True,record_construction=True)
    world,state=construct_actual_python(recorder)
    row=recorder.rows[0]
    return recorder,world,state,row["construction_receipt"]


def read(capture, row, binding=BINDING):
    return read_native_construction(capture,binding=binding,ordinal=row["ordinal"],draw_sha256=row["draw_sha256"],
        constructed_sha256=row["constructed_sha256"],state_utf8_sha256=row["native_state_utf8_sha256"],
        source_hashes=construction_source_hashes())


def map_fixture():
    recorder,world,state,capture=witness();construction=read(capture,recorder.rows[0])
    sides=[];native={"root":True,"state_utf8_sha256":construction["state_utf8_sha256"]}
    for seat in ("p1","p2"):
        rows=construction["slots"][seat]
        # Original Showdown order deliberately differs from native p1 order.
        original=list(reversed(rows)) if seat=="p1" else rows
        party=[dict(position=i,isActive=i==0,set=deepcopy(row["sampled_identity"]),
            species=f"[Species:{row['current_engine_id']}]",hp=100,maxhp=100,fainted=False,status="")
            for i,row in enumerate(original)]
        sides.append(dict(id=seat,active=[f"[Pokemon:{seat}a]"],
            team="".join(str(i+1) for i in range(len(party))),pokemon=party))
        native[seat]=dict(party_identities=[dict(slot=i,engine_id=v["current_engine_id"]) for i,v in enumerate(rows)])
    root=dict(battle=dict(formatid="gen3customgame",gameType="singles",ended=False,sides=sides),
        boundaryRequests={},terminalScheduled=False)
    return root,native,construction


class ConstructionTests(unittest.TestCase):
    def test_observed_world_and_native_assembly_bind_actual_draw_spec_and_state(self):
        recorder,world,state,capture=witness();before=deepcopy(capture)
        checked=read(capture,recorder.rows[0])
        self.assertEqual(capture,before)
        self.assertEqual([v["native_slot"] for v in checked["slots"]["p1"]],[0,1])
        self.assertEqual([v["sampled_identity"]["species"] for v in checked["slots"]["p1"]],["swampert","starmie"])
        self.assertEqual([p.id for p in state.kwargs["side_one"].pokemon],["swampert","starmie"])
        self.assertEqual(digest(capture["events"][0]["value"]["world"]),digest(asdict(world)))
        self.assertTrue(checked["construction_order_consistent"])
        for flag in ("process_ownership_validated","construction_provenance_validated","branch_transition_validated",
            "runtime_qualified","scientific_collection_authorized"):
            self.assertFalse(checked[flag])

    def test_world_and_native_observers_receive_copies_not_mutable_search_inputs(self):
        override=_override();payload=_payload(_dex());before=deepcopy((override,payload))
        def mutate(value):
            value["override"]["player_teams"].clear();value["payload"].clear();value["world"]["spec"].clear()
        world=battle_spec_from_payload(payload,override,dex=_dex(),construction_observer=mutate)
        self.assertEqual((override,payload),before)
        spec_before=asdict(world.spec)
        def native_mutate(value):value["spec"].clear();value["native_state"]="substitution"
        state=build_poke_engine_state(world.spec,module=fake_construction_module(),construction_observer=native_mutate)
        self.assertEqual(asdict(world.spec),spec_before);self.assertEqual(state.to_string(),"ORIG")

    def test_failed_missing_or_duplicate_hooks_cannot_be_rehabilitated(self):
        recorder,world,state,capture=witness()
        for mode in ("missing-world","missing-native","duplicate","wrong-spec","wrong-draw"):
            observer=NativeConstructionCapture(binding=BINDING,ordinal=0,draw_sha256=recorder.rows[0]["draw_sha256"])
            world_event=deepcopy(capture["events"][0]["value"]);native_event=deepcopy(capture["events"][1]["value"])
            with self.subTest(mode=mode):
                if mode=="wrong-draw":world_event["override"]["player_teams"]["p1"]="changed"
                if mode=="wrong-spec":native_event["spec"]["weather"]="rain"
                try:
                    if mode!="missing-world":observer.world_built(world_event)
                    if mode=="duplicate":observer.world_built(world_event)
                    if mode!="missing-native":observer.native_built(native_event)
                except ValueError:pass
                with self.assertRaises(ValueError):observer.export(expected_world=asdict(world),expected_state=state.to_string())

    def test_reader_rejects_rehashed_constructor_reordering_or_external_binding_drift(self):
        recorder,world,state,capture=witness();row=recorder.rows[0]
        for mode in ("binding","ordinal","bool-ordinal","draw","state","spec","order","source","missing-event"):
            changed=deepcopy(capture)
            if mode=="binding":changed["binding"]["subject"]="p2"
            if mode=="ordinal":changed["ordinal"]=1
            if mode=="bool-ordinal":changed["ordinal"]=False
            if mode=="draw":changed["events"][0]["value"]["override"]["player_teams"]["p1"]="changed"
            if mode=="state":changed["events"][1]["value"]["native_state"]="other"
            if mode=="spec":changed["events"][1]["value"]["spec"]["weather"]="rain"
            if mode=="order":changed["events"][0]["value"]["world"]["party_species"]["p1"]=("starmie","swampert")
            if mode=="source":changed["source_hashes"][next(iter(changed["source_hashes"]))]="b"*64
            if mode=="missing-event":changed["events"].pop()
            with self.subTest(mode=mode),self.assertRaises(ValueError):read(changed,row)
        # Even a self-consistently rehashed world/native spec must retain the
        # actual sampled construction order, not the caller's species ordering.
        changed=deepcopy(capture);changed["events"][0]["value"]["world"]["spec"]["side_one"]["pokemon"]=tuple(
            reversed(changed["events"][0]["value"]["world"]["spec"]["side_one"]["pokemon"]))
        changed["events"][1]["value"]["spec"]=deepcopy(changed["events"][0]["value"]["world"]["spec"])
        with self.assertRaisesRegex(ValueError,"species/slot"):
            read_native_construction(changed,binding=BINDING,ordinal=0,draw_sha256=row["draw_sha256"],
                constructed_sha256=digest(changed["events"][0]["value"]["world"]),
                state_utf8_sha256=row["native_state_utf8_sha256"],source_hashes=construction_source_hashes())

    def test_mapping_derived_from_unique_sets_not_names_species_or_caller_permutation(self):
        root,native,construction=map_fixture();before=deepcopy((root,native,construction))
        result=derive_party_mapping(root,native,construction)
        self.assertTrue(result["unique_set_mapping_derived"])
        self.assertEqual(result["mapping"]["native_to_original"],dict(p1=[1,0],p2=[0,1]))
        self.assertFalse(result["construction_provenance_validated"])
        self.assertEqual((root,native,construction),before)
        # A later switch changes CURRENT order, never original slot identity.
        side=root["battle"]["sides"][0];side["pokemon"].reverse();side["team"]="21"
        for i,p in enumerate(side["pokemon"]):p["position"],p["isActive"]=i,i==0
        self.assertEqual(derive_party_mapping(root,native,construction)["mapping"]["native_to_original"],dict(p1=[1,0],p2=[0,1]))

    def test_same_species_distinct_complete_sets_map_but_identical_sets_do_not(self):
        root,native,construction=map_fixture()
        # Author-written mapping fixture only, not an actual-constructor receipt.
        rows=construction["slots"]["p1"]
        rows[1]["sampled_identity"]["species"]=rows[0]["sampled_identity"]["species"]
        rows[1]["current_engine_id"]=rows[0]["current_engine_id"]
        native["p1"]["party_identities"][1]["engine_id"]=rows[0]["current_engine_id"]
        party=root["battle"]["sides"][0]["pokemon"]
        party[0]["set"]=deepcopy(rows[1]["sampled_identity"])
        party[0]["species"]=party[1]["species"]
        self.assertEqual(derive_party_mapping(root,native,construction)["mapping"]["native_to_original"]["p1"],[1,0])
        party[0]["set"]=deepcopy(party[1]["set"])
        self.assertIsNone(derive_party_mapping(root,native,construction)["mapping"])

    def test_malformed_identity_fields_remain_unknown(self):
        for key,value in (("level",True),("level",101),("species",None),("moves",[None]),
                          ("nature",""),("gender","unknown"),("happiness",0),("shiny",True),
                          ("evs",dict.fromkeys(("hp","atk","def","spa","spd","spe"),True))):
            root,native,construction=map_fixture()
            root["battle"]["sides"][0]["pokemon"][1]["set"][key]=value
            with self.subTest(key=key,value=value):
                self.assertIsNone(derive_party_mapping(root,native,construction)["mapping"])

    def test_unsupported_packed_tail_or_shiny_cannot_alias_default_identity(self):
        for index,value in ((9,"S"),(11,"0"),(11,"255,Fire")):
            recorder,world,state,capture=witness();row=recorder.rows[0]
            packed=capture["events"][0]["value"]["override"]["player_teams"]["p1"].split("]")
            first=packed[0].split("|");first[index]=value;packed[0]="|".join(first)
            changed=deepcopy(capture)
            changed["events"][0]["value"]["override"]["player_teams"]["p1"]="]".join(packed)
            sha=digest(changed["events"][0]["value"]["override"]);changed["draw_sha256"]=sha
            with self.subTest(index=index,value=value),self.assertRaisesRegex(ValueError,"unsupported packed"):
                read_native_construction(changed,binding=BINDING,ordinal=0,draw_sha256=sha,
                    constructed_sha256=row["constructed_sha256"],state_utf8_sha256=row["native_state_utf8_sha256"],
                    source_hashes=construction_source_hashes())

    def test_observer_failures_do_not_log_private_exception_messages(self):
        def fail(value):raise ValueError("private sampled secret sentinel")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            world=battle_spec_from_payload(_payload(_dex()),_override(),dex=_dex(),construction_observer=fail)
            build_poke_engine_state(world.spec,module=fake_construction_module(),construction_observer=fail)
        self.assertEqual(len(caught),2)
        self.assertTrue(all("ValueError" in str(w.message) and "sentinel" not in str(w.message) for w in caught))

    def test_missing_or_duplicate_set_identities_remain_unknown(self):
        for mode in ("missing","collision","count"):
            root,native,construction=map_fixture()
            if mode=="missing":root["battle"]["sides"][0]["pokemon"][1]["set"].pop("ivs")
            if mode=="collision":root["battle"]["sides"][0]["pokemon"][1]["set"]=deepcopy(
                root["battle"]["sides"][0]["pokemon"][0]["set"])
            if mode=="count":construction["slots"]["p1"].pop()
            with self.subTest(mode=mode):
                result=derive_party_mapping(root,native,construction)
                self.assertIsNone(result["mapping"]);self.assertTrue(result["limitations"])

    def test_world_wrapper_passes_opt_in_observer_without_changing_legacy_call(self):
        observer=lambda value:None;state=object();override=_override()
        with patch("pokezero.local_showdown._public_materialization_payload",return_value=_payload(_dex())), \
             patch("pokezero.engine_world.battle_spec_from_payload",return_value="world") as build:
            self.assertEqual(world_battle_spec(state,override,dex=_dex()),"world")
            self.assertNotIn("construction_observer",build.call_args.kwargs)
            world_battle_spec(state,override,dex=_dex(),construction_observer=observer)
            self.assertIs(build.call_args.kwargs["construction_observer"],observer)


if __name__ == "__main__":unittest.main()
