"""Actual parser/normalization/encoder on authored actor histories, no games."""
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
import json
from pathlib import PurePosixPath, PureWindowsPath
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
from pokezero.showdown import _ReplayParser, V2_1_REPLAY_OBSERVATION_SPEC, V2_2_REPLAY_OBSERVATION_SPEC
from pokezero.mcts_eval.search_over_raw_actor_projection import (
    ActorObservationFold, compile_actor_program, digest, SCHEMA, CONTEXT_SCHEMA, ENCODING_LAW,
    _state_payload)
from pokezero.observation import ObservationFeatureMasks
from test_local_showdown import request_payload
from test_showdown import _phase2_fake_dex
from pokezero.randbat_vocab import gen3_category_vocabulary
from pokezero.belief import PublicBattleBeliefEngine


class _ProbeEnum(Enum):
    VALUE = 7


@dataclass
class _ProbeState:
    value: object
    cache: object


class StatePayloadContractsTests(unittest.TestCase):
    def test_mapping_order_is_canonical_without_erasing_key_types(self):
        first = {"z": [1, True], 2: "integer-key", "a": "last"}
        second = MappingProxyType({"a": "last", 2: "integer-key", "z": [1, True]})
        self.assertEqual(_state_payload(first), _state_payload(second))
        self.assertNotEqual(digest(_state_payload({2: "value"})),
                            digest(_state_payload({"2": "value"})))
        self.assertNotEqual(digest(_state_payload(False)), digest(_state_payload(0)))

    def test_sequence_kinds_and_set_order_remain_distinct(self):
        self.assertEqual(_state_payload({3, 1, 2})["items"], [1, 2, 3])
        values = [[1, 2], (1, 2), {1, 2}, frozenset({1, 2})]
        payloads = [_state_payload(value) for value in values]
        self.assertEqual([value["container"] for value in payloads],
                         ["list", "tuple", "set", "frozenset"])
        self.assertEqual(len({digest(value) for value in payloads}), 4)

    def test_dataclasses_enums_and_portable_paths_keep_full_identity(self):
        value = _ProbeState(_ProbeEnum.VALUE, {"path": PurePosixPath("catalog/sets.json")})
        payload = _state_payload(value)
        self.assertEqual(payload["dataclass"], type(value).__module__ + "." + type(value).__qualname__)
        self.assertEqual(payload["fields"]["value"],
                         {"enum": _ProbeEnum.__module__ + "." + _ProbeEnum.__qualname__, "value": 7})
        self.assertEqual(_state_payload(PureWindowsPath("C:/catalog/sets.json")),
                         {"path": "C:\\catalog\\sets.json"})
        old = digest(payload)
        value.cache["path"] = PurePosixPath("catalog/changed.json")
        self.assertNotEqual(digest(_state_payload(value)), old)

    def test_known_mutable_state_keeps_all_cache_fields(self):
        engine = PublicBattleBeliefEngine()
        engine._serialization_probe_cache = {"seen": 1}
        original = _state_payload(engine)
        self.assertEqual(original["object"], "pokezero.belief.PublicBattleBeliefEngine")
        engine._serialization_probe_cache["seen"] = 2
        self.assertNotEqual(digest(original), digest(_state_payload(engine)))

    def test_cycles_and_unsupported_objects_refuse_but_shared_values_are_valid(self):
        loop = []; loop.append(loop)
        mapping = {}; mapping["self"] = mapping
        for value in (loop, mapping):
            with self.subTest(value_type=type(value).__name__), self.assertRaisesRegex(ValueError, "cyclic"):
                _state_payload(value)
        for value in (object(), SimpleNamespace(value=1)):
            with self.subTest(value_type=type(value).__name__), self.assertRaisesRegex(ValueError, "unsupported"):
                _state_payload(value)
        shared = [1, 2]
        self.assertEqual(_state_payload([shared, shared]), _state_payload([[1, 2], [1, 2]]))

    def test_nonfinite_binding_never_becomes_a_digest(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                digest(_state_payload({"config": value}))


def own_request(actor="p1", *, active="Charizard", hp="250/250", pp=55):
    row=request_payload(actor)
    names=("tackle","protect","transform","hyperbeam")
    row["active"][0]["moves"]=[dict(id=move,move=move,pp=pp if index==0 else 8,
        maxpp=56 if index==0 else 8,disabled=False) for index,move in enumerate(names)]
    for mon in row["side"]["pokemon"]:
        mon["moves"]=list(names)
        mon["stats"]={"atk":130,"def":140,"spa":150,"spd":160,"spe":170}
        mon["baseAbility"]="blaze";mon["item"]="leftovers"
        mon["active"]=mon["ident"]==actor+": "+active
        if mon["active"]:mon["condition"]=hp
    return row


def prefix(actor):
    other="p2" if actor=="p1" else "p1"
    return ["|player|p1|PokeZero p1|", "|player|p2|PokeZero p2|", "|gen|3", "|start",
        "|switch|"+actor+"a: Charizard|Charizard, L80, M|250/250",
        "|switch|"+other+"a: Snorlax|Snorlax, L80, M|100/100", "|turn|1"]


def program(actor="p1", *, events=None):
    opening=own_request(actor,pp=56)
    return dict(schema=SCHEMA,actor=actor,encoding_law=ENCODING_LAW,opening_request=opening,
        events=events or [dict(kind="protocol",lines=prefix(actor)),dict(kind="own_request",request=opening),
            dict(kind="observe",coordinate=0)])


class ActorProjectionTests(unittest.TestCase):
    def setUp(self):
        base=LocalShowdownConfig(set_belief_source=False,observation_spec=V2_2_REPLAY_OBSERVATION_SPEC)
        # Read-only closed vocabulary enumeration, no battle or model. Keep
        # the existing v2.2 vocabulary guard; do not replace it with a fallback.
        vocab=gen3_category_vocabulary(base.resolved_showdown_root(),include_turn_merged=True,extra_moves=("tackle",))
        self.config=replace(base,category_vocab=vocab)
        self.dex=patch("pokezero.local_showdown.load_showdown_dex_cached",return_value=_phase2_fake_dex())
        self.dex.start();self.addCleanup(self.dex.stop)

    def compile(self,data,config=None):
        config=self.config if config is None else config
        return compile_actor_program(data,config=config,expected_program_sha256=digest(data),
            expected_config_sha256=digest(_state_payload(config)))

    def ordinary(self,data,config=None):
        # Independent normal environment path, no snapshot/state restoration.
        env=LocalShowdownEnv(self.config if config is None else config)
        actor=data["actor"];other="p2" if actor=="p1" else "p1"
        env._parser=_ReplayParser(data.get("battle_id","actor-projection"),complete_prefix=True,
            hp_visibility={actor:"exact",other:"percentage"})
        observations,states=[],[]
        for row in data["events"]:
            if row["kind"]=="protocol":
                env._apply_event(dict(type="stream",stream="omniscient",lines=deepcopy(row["lines"])))
            elif row["kind"]=="own_request":
                env._apply_event(dict(type="stream",stream=actor,lines=["|request|"+json.dumps(row["request"])]))
            elif env.terminal() is not None or actor not in env.requested_players():
                observations.append(None);states.append(None)
            else:
                observations.append(env.observe(actor));states.append(env.public_materialization_state(actor))
        self.assertIsNone(env._process)
        return observations,states

    def assert_parity(self,data,config=None):
        before=deepcopy(data)
        actual=self.compile(data,config)
        observations,states=self.ordinary(data,config)
        self.assertEqual(data,before)
        self.assertEqual([digest(_state_payload(v)) for v in actual.observations],
            [digest(_state_payload(v)) for v in observations])
        def public(state,observation):
            if state is None:return None
            engine=state.belief_engine.clone();engine.set_source=None
            return _state_payload(replace(state,belief_engine=engine))
        self.assertEqual([digest(public(state,obs)) for state,obs in zip(actual.materialization_states,actual.observations)],
            [digest(public(state,obs)) for state,obs in zip(states,observations)])
        self.assertFalse(actual.receipt["bridge_process_started"])
        self.assertFalse(actual.receipt["model_loaded"])
        self.assertFalse(actual.receipt["fabricated_snapshot_used"])
        self.assertFalse(actual.receipt["complete_actor_history_validated"])
        self.assertFalse(actual.receipt["independent_terminal_labels"])
        return actual

    def test_both_actors_use_actual_existing_encoder_and_keep_schema_vocab(self):
        for actor in ("p1","p2"):
            for spec in (V2_1_REPLAY_OBSERVATION_SPEC,V2_2_REPLAY_OBSERVATION_SPEC):
                with self.subTest(actor=actor,spec=spec.schema_version):
                    config=replace(self.config,observation_spec=spec)
                    result=self.assert_parity(program(actor),config)
                    self.assertEqual(result.receipt["encoded_observations"],1)

    def test_v2_preserves_bound_original_observation_origin_without_masking_metadata(self):
        data=program();data.update(schema=CONTEXT_SCHEMA,battle_id="local-gen3randombattle-42")
        result=self.assert_parity(data)
        self.assertEqual(result.observations[0].metadata["battle_id"],data["battle_id"])
        self.assertEqual(result.materialization_states[0].replay.battle_id,data["battle_id"])
        self.assertEqual(result.receipt["schema"],CONTEXT_SCHEMA)
        legacy=self.compile(program())
        self.assertEqual(legacy.observations[0].metadata["battle_id"],"actor-projection")
        self.assertEqual(legacy.receipt["schema"],SCHEMA)

    def test_origin_is_explicit_versioned_and_guarded_through_callbacks(self):
        for value in (None,False,"", "source\nother", "x"*201):
            data=program();data.update(schema=CONTEXT_SCHEMA,battle_id=value)
            with self.subTest(value=value),self.assertRaises(ValueError):self.compile(data)
        data=program();data["battle_id"]="foreign"
        with self.assertRaisesRegex(ValueError,"externally bound"):self.compile(data)
        data=program();data["schema"]=CONTEXT_SCHEMA
        with self.assertRaisesRegex(ValueError,"externally bound"):self.compile(data)
        fold=ActorObservationFold(config=self.config,actor="p1",opening_request=own_request(),
            expected_opening_sha256=digest(own_request()),battle_id="original")
        fold.env._parser.battle_id="replacement"
        with self.assertRaisesRegex(ValueError,"inputs/state changed"):fold.check()

    def test_switch_bench_pp_history_and_repeated_observation_calls_match(self):
        data=program()
        data["events"] += [dict(kind="protocol",lines=["|move|p1a: Charizard|Tackle|p2a: Snorlax",
            "|-damage|p2a: Snorlax|91/100", "|turn|2"]),
            dict(kind="own_request",request=own_request(pp=55)),dict(kind="observe",coordinate=1),
            dict(kind="protocol",lines=["|switch|p1a: Snorlax|Snorlax, L80, M|350/350", "|turn|3"]),
            dict(kind="own_request",request=own_request(active="Snorlax",hp="350/350",pp=54)),
            dict(kind="observe",coordinate=2),dict(kind="observe",coordinate=3)]
        result=self.assert_parity(data)
        states=result.materialization_states[-1].self_move_states
        self.assertEqual(states["charizard"][0]["pp"],55)
        self.assertEqual(states["snorlax"][0]["pp"],54)
        self.assertEqual(result.receipt["observed_requests"],3)

    def test_transformed_request_does_not_overwrite_genuine_opening_moves(self):
        data=program()
        copied=own_request(pp=5)
        copied["active"][0]["moves"]=[dict(id="bodyslam",move="Body Slam",pp=5,maxpp=5,disabled=False)]
        copied["side"]["pokemon"][0]["moves"]=["bodyslam"]
        data["events"] += [dict(kind="protocol",lines=["|move|p1a: Charizard|Transform|p2a: Snorlax",
            "|-transform|p1a: Charizard|p2a: Snorlax", "|turn|2"]),
            dict(kind="own_request",request=copied),dict(kind="observe",coordinate=1),
            dict(kind="protocol",lines=["|switch|p1a: Snorlax|Snorlax, L80, M|350/350", "|turn|3"]),
            dict(kind="own_request",request=own_request(active="Snorlax",hp="350/350")),
            dict(kind="observe",coordinate=2)]
        result=self.assert_parity(data)
        states=result.materialization_states[-1].self_move_states
        self.assertEqual(states["charizard"][0]["id"],"tackle")
        self.assertNotIn("bodyslam",str(states["charizard"]))

    def test_recharge_and_struggle_requests_do_not_become_permanent_pp_banks(self):
        for restricted in ("recharge","struggle"):
            data=program();request=own_request()
            request["active"][0]["moves"]=[dict(id=restricted,move=restricted)]
            data["events"] += [dict(kind="protocol",lines=["|move|p1a: Charizard|Hyper Beam|p2a: Snorlax",
                "|-damage|p2a: Snorlax|52/100", "|turn|2"]),dict(kind="own_request",request=request),
                dict(kind="observe",coordinate=1)]
            with self.subTest(restricted=restricted):
                result=self.assert_parity(data)
                self.assertNotIn(restricted,str(result.materialization_states[-1].self_move_states))

    def test_forced_replacement_wait_and_terminal_coordinates_are_not_dropped(self):
        data=program()
        forced=own_request(hp="0 fnt");forced.pop("active");forced["forceSwitch"]=[True]
        wait=own_request();wait.pop("active");wait["wait"]=True
        data["events"] += [dict(kind="protocol",lines=["|-damage|p1a: Charizard|0 fnt", "|faint|p1a: Charizard"]),
            dict(kind="own_request",request=forced),dict(kind="observe",coordinate=1),
            dict(kind="own_request",request=wait),dict(kind="observe",coordinate=2),
            dict(kind="protocol",lines=["|win|PokeZero p1"]),dict(kind="observe",coordinate=3)]
        result=self.assert_parity(data)
        self.assertFalse(any(result.observations[1].legal_action_mask[:4]))
        self.assertTrue(any(result.observations[1].legal_action_mask[4:]))
        self.assertEqual(result.observations[-2:],(None,None))
        self.assertEqual(result.receipt["coordinates"],[0,1,2,3])

    def test_missing_or_foreign_opening_and_request_split_protocol_are_unavailable(self):
        for change in (lambda data:data.__setitem__("opening_request",{}),
                lambda data:data.__setitem__("opening_request",own_request("p2")),
                lambda data:data["opening_request"]["side"]["pokemon"][0].pop("moves"),
                lambda data:data["events"][0]["lines"].append("|request|OPPONENT_PRIVATE_TEAM"),
                lambda data:data["events"][0]["lines"].append("|split|p2"),
                lambda data:data["events"][1].__setitem__("request",own_request("p2"))):
            data=program();change(data)
            with self.assertRaises(ValueError):self.compile(data)

    def test_original_program_config_and_coordinate_mutation_cannot_publish(self):
        data=program();old=digest(data);data["events"][0]["lines"].append("|turn|2")
        with self.assertRaises(ValueError):
            compile_actor_program(data,config=self.config,expected_program_sha256=old,
                expected_config_sha256=digest(_state_payload(self.config)))
        data=program();data["events"].append(dict(kind="observe",coordinate=False))
        with self.assertRaises(ValueError):self.compile(data)
        original=LocalShowdownEnv._observe
        data=program()
        def mutate(env,actor,*args,**kwargs):
            value=original(env,actor,*args,**kwargs)
            data["events"][0]["lines"].append("|message|late-substitution")
            return value
        with patch.object(LocalShowdownEnv,"_observe",side_effect=mutate),self.assertRaisesRegex(ValueError,"changed during"):
            self.compile(data)

    def test_projection_shell_cannot_open_or_restore_a_simulator(self):
        opening=own_request(pp=56)
        fold=ActorObservationFold(config=self.config,actor="p1",opening_request=opening,
            expected_opening_sha256=digest(opening))
        for method in ("_start_bridge","_send_command","_read_event","reset","reset_with_start_override",
                "step","snapshot","snapshot_actionable_boundary","restore","observe_search_snapshot"):
            with self.subTest(method=method),self.assertRaisesRegex(ValueError,"cannot launch"):
                getattr(fold.env,method)()

    def test_real_public_catalog_and_annotation_trackers_match_repeated_normal_encoding(self):
        # Static catalog/dex loading only, not a simulator reset or model.
        self.dex.stop()
        config=replace(self.config,set_belief_source=True,
            feature_masks=ObservationFeatureMasks(tier2_investment=True,item_belief_narrowing=True))
        data=program()
        data["events"] += [dict(kind="protocol",lines=["|move|p1a: Charizard|Hyper Beam|p2a: Snorlax",
            "|-damage|p2a: Snorlax|51/100", "|upkeep", "|turn|2"]),
            dict(kind="own_request",request=own_request(pp=55)),dict(kind="observe",coordinate=1),
            dict(kind="observe",coordinate=2)]
        original=LocalShowdownEnv._state_for_player;calls=[]
        def annotate(env,actor,*args,**kwargs):
            result=original(env,actor,*args,**kwargs)
            calls.append((bool(env._tier2_trackers),bool(env._investment_trackers),len(env._parser.public_events)))
            return result
        with patch.object(LocalShowdownEnv,"_state_for_player",autospec=True,side_effect=annotate):
            result=self.assert_parity(data,config)
        self.assertEqual(result.receipt["encoded_observations"],3)
        self.assertTrue(calls)
        self.assertTrue(all(tier2 and investment for tier2,investment,_ in calls))
        self.assertGreater(max(count for _,_,count in calls),min(count for _,_,count in calls))


if __name__ == "__main__":unittest.main()
