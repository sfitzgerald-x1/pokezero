"""Authored transport chronology through real read/encoder/raw-policy seams.

No simulator, source game, model or claimed process lifecycle is exercised.
The transport queue holds authored bridge messages, not private snapshots.
"""
from copy import deepcopy
from dataclasses import replace
import json
import pickle
import queue
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
from pokezero.showdown import _ReplayParser, V2_2_REPLAY_OBSERVATION_SPEC
from pokezero.randbat_vocab import gen3_category_vocabulary
from pokezero.mcts_eval.search_over_raw_source import AuditedRawPolicy
from pokezero.mcts_eval.search_over_raw_source_run import SourceProcessOwner, TrustedOwnedSource
from pokezero.mcts_eval.search_over_raw_source_actor_history import (
    SourceActorHistoryCapture, read_source_actor_program)
import test_search_over_raw_actor_projection as actor_fixture
from test_search_over_raw_source import FakeChampion, SHA
from test_showdown import _phase2_fake_dex


class SourceActorHistoryTests(unittest.TestCase):
    def setUp(self):
        config = LocalShowdownConfig(set_belief_source=False, observation_spec=V2_2_REPLAY_OBSERVATION_SPEC)
        vocab = gen3_category_vocabulary(config.resolved_showdown_root(), include_turn_merged=True, extra_moves=("tackle",))
        self.config = replace(config, category_vocab=vocab)
        self.dex = patch("pokezero.local_showdown.load_showdown_dex_cached", return_value=_phase2_fake_dex())
        self.dex.start(); self.addCleanup(self.dex.stop)

    def setup_capture(self, *, exact_opponent=False):
        self.env = LocalShowdownEnv(self.config)
        self.env._parser = _ReplayParser("actor-projection", complete_prefix=True,
            hp_visibility={"p1":"exact", "p2":"exact" if exact_opponent else "percentage"})
        self.env._battle_token = "owned"
        self.env._stdout_queue = queue.Queue()
        # Actual SourceProcessOwner event/observation methods, with explicitly
        # authored ownership fields. This test does not certify a live child.
        self.owner = SourceProcessOwner.__new__(SourceProcessOwner)
        self.owner.env, self.owner.records, self.owner.phase, self.owner.token = self.env, [], "started", "owned"
        self.capture = SourceActorHistoryCapture(self.config)
        self.owner.actor_history = self.capture
        self.capture.bind(self.owner, self.env, self.owner.records)
        self.env._bridge_process_observer = self.owner
        self.policy = AuditedRawPolicy(FakeChampion(), checkpoint_sha256=SHA, actor_history=self.capture)
        self.receive(dict(type="started", battleId="owned", formatid="gen3randombattle", seed=[1,2,3,4]))

    def receive(self, event, *, apply=True):
        self.env._stdout_queue.put(json.dumps(event))
        received = self.env._read_event(timeout=.01)
        if apply: self.env._apply_event(received)
        return len(self.owner.records)-1

    def stream(self, seat, lines):
        return self.receive(dict(type="stream", battleId="owned", stream=seat, lines=lines))

    def opening(self, *, private_opponent=False):
        visible = actor_fixture.prefix("p1")
        omniscient = deepcopy(visible)
        if private_opponent: omniscient[5] = "|switch|p2a: Snorlax|Snorlax, L80, M|213/350"
        self.stream("omniscient", omniscient)
        request = actor_fixture.own_request(pp=56)
        self.stream("p1", visible + ["|request|" + json.dumps(request)])
        self.stream("p2", ["|request|" + json.dumps({"side":{"id":"p2"},"canary":"NEVER_IMPORT_OPPONENT_HISTORY"})])

    def observe_and_select(self):
        value = self.env.observe("p1")
        self.policy.select_action(value, rng=random.Random(0))
        return value

    def boundary(self):
        return self.receive(dict(type="snapshot_actionable_boundary", battleId="owned", available=True,
            requested=["p1"], snapshot={"private":"not-exported"}), apply=False)

    def read(self, cutoff):
        return read_source_actor_program(self.capture, records=deepcopy(self.owner.records),
            cutoff_record_index=cutoff, config=self.config)

    def test_actual_read_observe_and_raw_input_seams_match_with_repeated_calls(self):
        self.setup_capture(); self.opening()
        self.observe_and_select(); self.observe_and_select()
        cutoff = self.boundary(); self.capture.seal()
        result = self.read(cutoff)
        self.assertEqual(result.receipt["status"], "INPUT_CORRESPONDENCE_MATCHED")
        self.assertEqual(result.receipt["observation_coordinates"], [0,1])
        self.assertEqual(result.receipt["raw_input_count"], 2)
        self.assertTrue(result.receipt["complete_actor_history_validated"])
        self.assertFalse(result.receipt["process_ownership_validated"])
        self.assertFalse(result.receipt["construction_provenance_validated"])
        self.assertNotIn("NEVER_IMPORT_OPPONENT_HISTORY", json.dumps(result.program))
        self.assertNotIn("NEVER_IMPORT_OPPONENT_HISTORY", json.dumps(result.receipt))
        self.assertNotIn("NEVER_IMPORT_OPPONENT_HISTORY", json.dumps(self.capture._events))
        for obj in (self.capture, result):
            with self.assertRaises(TypeError): pickle.dumps(obj)

    def test_source_origin_is_captured_from_actual_parser_and_preserved_in_full_input(self):
        self.setup_capture()
        self.env._parser.battle_id="local-gen3randombattle-42"
        self.opening();observed=self.observe_and_select()
        cutoff=self.boundary();self.capture.seal();result=self.read(cutoff)
        self.assertEqual(result.receipt["status"],"INPUT_CORRESPONDENCE_MATCHED")
        self.assertEqual(result.program["battle_id"],observed.metadata["battle_id"])
        self.assertEqual(result.compiled.observations[0].metadata["battle_id"],"local-gen3randombattle-42")
        self.assertEqual(self.capture._observations[0][0]["battle_id"],"local-gen3randombattle-42")

    def test_existing_omniscient_private_hp_mismatch_is_not_masked_away(self):
        self.setup_capture(exact_opponent=True); self.opening(private_opponent=True)
        observed = self.observe_and_select(); original = deepcopy(observed)
        cutoff = self.boundary(); self.capture.seal()
        result = self.read(cutoff)
        self.assertEqual(observed, original)
        self.assertEqual(result.receipt["status"], "ENCODING_MISMATCH")
        self.assertEqual(result.receipt["mismatched_observation_coordinates"], [0])
        self.assertFalse(result.receipt["complete_actor_history_validated"])
        self.assertTrue(result.receipt["historical_encoding_unchanged"])

    def test_late_actor_packet_cannot_repair_an_earlier_observation(self):
        self.setup_capture(); self.opening(); self.observe_and_select()
        delta = ["|move|p1a: Charizard|Tackle|p2a: Snorlax", "|-damage|p2a: Snorlax|91/100", "|turn|2"]
        # Omniscient pump delivers the new turn before p1's pump. The next
        # encoded input therefore differs at its original received cutoff.
        self.stream("omniscient", delta); self.observe_and_select()
        early = self.boundary()
        self.stream("p1", delta)
        self.observe_and_select(); late = self.boundary(); self.capture.seal()
        early_result, late_result = self.read(early), self.read(late)
        self.assertEqual(early_result.receipt["observation_coordinates"], [0,1])
        self.assertEqual(early_result.receipt["mismatched_observation_coordinates"], [1])
        self.assertEqual(late_result.receipt["mismatched_observation_coordinates"], [1])
        self.assertEqual(late_result.receipt["observation_coordinates"], [0,1,2])
        self.assertNotIn("|turn|2", json.dumps(early_result.program))

    def test_foreign_equal_input_mutation_and_missing_actual_policy_inputs_refuse(self):
        self.setup_capture(); self.opening()
        value = self.env.observe("p1")
        with self.assertRaisesRegex(ValueError, "original source encoding"):
            self.capture.policy_input(deepcopy(value))
        cutoff = self.boundary(); self.capture.seal()
        with self.assertRaisesRegex(ValueError, "actual source raw inputs missing"): self.read(cutoff)
        value.metadata["late"] = "substituted"
        with self.assertRaisesRegex(ValueError, "observation changed"): self.read(cutoff)

    def test_policy_mutation_is_rejected_at_actual_consumption_seam(self):
        self.setup_capture(); self.opening()
        value = self.env.observe("p1")
        original = self.policy.policy.select_action
        def mutate(observation, **kwargs):
            decision = original(observation, **kwargs)
            observation.metadata["mutated_by_policy"] = True
            return decision
        self.policy.policy.select_action = mutate
        with self.assertRaisesRegex(ValueError, "mutated/substituted"):
            self.policy.select_action(value, rng=random.Random(0))

    def test_policy_callback_cannot_disable_its_own_post_input_guard(self):
        self.setup_capture(); self.opening()
        value=self.env.observe("p1"); original=self.policy.policy.select_action
        def disable(observation, **kwargs):
            decision=original(observation,**kwargs)
            self.policy.actor_history=None
            return decision
        self.policy.policy.select_action=disable
        with self.assertRaisesRegex(ValueError,"witness changed"):
            self.policy.select_action(value,rng=random.Random(0))

    def test_capture_owner_and_transport_references_cannot_be_replaced(self):
        for target in ("owner","env","records"):
            self.setup_capture(); self.opening(); self.observe_and_select()
            cutoff=self.boundary(); self.capture.seal()
            original=LocalShowdownEnv._observe
            def substitute(env,player,**kwargs):
                value=original(env,player,**kwargs)
                if target=="owner":
                    self.capture._owner=SimpleNamespace(env=self.env,records=self.owner.records,actor_history=self.capture)
                elif target=="env":
                    replacement=LocalShowdownEnv(self.config)
                    self.owner.env=self.capture._env=replacement
                else:
                    self.owner.records=self.capture._records=deepcopy(self.owner.records)
                return value
            with patch.object(LocalShowdownEnv,"_observe",side_effect=substitute),self.assertRaisesRegex(ValueError,"capture changed"):
                self.read(cutoff)

    def mock_handle(self, directory):
        """Final callback contract shape only, NOT a lifecycle clearance."""
        handle=TrustedOwnedSource.__new__(TrustedOwnedSource)
        handle.source={"roots":[{"root_id":"authored-selected"}]}
        handle.archive=object();handle._actor_history=self.capture;handle.directory=Path(directory)
        (handle.directory/"bridge-transport.json").write_text(json.dumps({"records":self.owner.records}))
        handle._original_owner=lambda:self.capture.verify(self.owner.records)
        handle.validate=lambda:handle._original_owner()
        return handle

    def test_final_validation_cannot_mutate_any_returned_program_component(self):
        import pokezero.mcts_eval.search_over_raw_source_actor_history as history
        import pokezero.mcts_eval.search_over_raw_source_run as source
        mutations=(lambda value:value.program["events"][0]["lines"].append("|turn|99"),
            lambda value:value.compiled.observations[0].metadata.__setitem__("late",True),
            lambda value:value.compiled.materialization_states[0].self_request.__setitem__("late",True),
            lambda value:setattr(value.compiled.materialization_states[0].belief_engine,"late",True),
            lambda value:value.compiled.receipt.__setitem__("bridge_process_started",True))
        for mutation in mutations:
            self.setup_capture();self.opening();self.observe_and_select()
            cutoff=self.boundary();self.capture.seal()
            with tempfile.TemporaryDirectory() as directory:
                handle=self.mock_handle(directory);saved=[];original=history.read_source_actor_program
                def retain(*args,**kwargs):
                    value=original(*args,**kwargs);saved.append(value);return value
                def late_validate():
                    handle._original_owner()
                    if saved:mutation(saved[0])
                handle.validate=late_validate
                joined={"authored-selected":{"source_record_index":cutoff}}
                with patch.object(source,"read_archived_source_snapshot_join",return_value=joined), \
                        patch.object(history,"read_source_actor_program",side_effect=retain), \
                        self.assertRaisesRegex(ValueError,"final source checks"):
                    handle.actor_program("authored-selected",config=self.config)

    def test_last_join_callback_cannot_mutate_original_capture_after_validation(self):
        import pokezero.mcts_eval.search_over_raw_source_run as source
        self.setup_capture();self.opening();self.observe_and_select()
        cutoff=self.boundary();self.capture.seal()
        with tempfile.TemporaryDirectory() as directory:
            handle=self.mock_handle(directory);calls=[]
            def join(*args):
                calls.append(1)
                if len(calls)==2:self.capture._events[0]["payload"]["lines"].append("|turn|99")
                return {"authored-selected":{"source_record_index":cutoff}}
            with patch.object(source,"read_archived_source_snapshot_join",side_effect=join), \
                    self.assertRaisesRegex(ValueError,"history/transport changed"):
                handle.actor_program("authored-selected",config=self.config)

    def test_full_materialization_guard_supports_real_static_catalog(self):
        import pokezero.mcts_eval.search_over_raw_source_actor_history as history
        self.dex.stop()
        self.config=replace(self.config,set_belief_source=True)
        self.setup_capture();self.opening();self.observe_and_select()
        cutoff=self.boundary();self.capture.seal()
        value=self.read(cutoff)
        old=history.retained_program_identity(value)
        value.compiled.materialization_states[0].self_request["late"]=True
        self.assertNotEqual(old,history.retained_program_identity(value))

    def test_transport_capture_and_config_cannot_change_during_compilation(self):
        self.setup_capture(); self.opening(); self.observe_and_select()
        cutoff = self.boundary(); self.capture.seal()
        changed = deepcopy(self.owner.records); changed[1]["payload_sha256"] = "f"*64
        with self.assertRaisesRegex(ValueError, "transport changed"):
            read_source_actor_program(self.capture, records=changed, cutoff_record_index=cutoff, config=self.config)
        original = LocalShowdownEnv._observe
        def mutate(env, player, **kwargs):
            result = original(env, player, **kwargs)
            self.capture._events[0]["payload"]["lines"].append("|turn|99")
            return result
        with patch.object(LocalShowdownEnv, "_observe", side_effect=mutate), self.assertRaisesRegex(ValueError, "changed"):
            self.read(cutoff)

    def test_missing_opening_and_midbattle_first_request_stay_unavailable(self):
        self.setup_capture()
        self.stream("omniscient", actor_fixture.prefix("p1")+["|turn|2"])
        self.stream("p1", actor_fixture.prefix("p1")+["|turn|2", "|request|"+json.dumps(actor_fixture.own_request())])
        self.observe_and_select(); cutoff=self.boundary(); self.capture.seal()
        with self.assertRaisesRegex(ValueError, "mid-battle"): self.read(cutoff)

    def test_uninstrumented_policy_and_environment_remain_default_off(self):
        self.setup_capture(); self.opening()
        self.env._bridge_process_observer = None
        count = len(self.owner.records)
        value = self.env.observe("p1")
        policy = AuditedRawPolicy(FakeChampion(), checkpoint_sha256=SHA)
        decision = policy.select_action(value, rng=random.Random(0))
        self.assertEqual(decision.action_index, 1)
        self.assertIsNone(policy.actor_history)
        self.assertEqual(len(self.owner.records), count)


if __name__ == "__main__": unittest.main()
