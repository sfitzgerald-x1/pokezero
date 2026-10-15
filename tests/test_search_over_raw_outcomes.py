"""Actual Python transport seams with synthetic streams; no engines or games."""
from copy import deepcopy
import hashlib
import io
import json
import queue
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest

from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv, LocalShowdownSnapshot
from pokezero.mcts_eval.search_over_raw import digest, paired_continuations
from pokezero.mcts_eval.search_over_raw_outcomes import (
    ContinuationCapture, OwnedContinuationLifecycle, OutcomeReceiptValidator, validate_observed_outcome,
)
from pokezero.mcts_eval.search_over_raw_world_inputs import SNAPSHOT_INPUT_FIELDS

SHA = "a" * 64
NAMESPACE = "86fbb43c-dd13-47f5-8c91-82c1758765eb"
BINDING = dict(root_id="synthetic-root", source_seed=123, public_record_sha256=SHA,
    information_key="public-fixture", legal_action_mask=[True, True])
CONTRACT = dict(root_binding=BINDING, root_id=BINDING["root_id"], namespace=NAMESPACE,
    subject="p1", max_boundaries=250, replicates=8, selection_seed=123,
    policy="raw_argmax_after_initial_sampled_opponent_reply")


def snapshot():
    fields = {k: {} for k in SNAPSHOT_INPUT_FIELDS}
    fields.update(format_id="fixture", observation_format_id="fixture", terminal=None,
        protocol_lines=("|player|p1|Fixture One|", "|player|p2|Fixture Two|"), latest_turn=3,
        bridge_snapshot=dict(battle={"hidden": "private-fixture"},
            boundaryRequests={"p1": {"active": [{}]}, "p2": {"active": [{}]}}, terminalScheduled=False))
    return SimpleNamespace(**fields, battle_token="source-token", battle_id="source")


class SyntheticTransport(LocalShowdownEnv):
    """Use real step/send/read hooks, but no process or simulator is started."""
    def __init__(self, terminal_line="|win|Fixture One", stale=False):
        self._bridge_evidence_observer = None
        self._battle_token = "owned-synthetic-token"
        self._process = SimpleNamespace(stdin=io.StringIO(), poll=lambda: None)
        self._stdout_queue = queue.Queue()
        self.terminal_line, self.stale, self.index = terminal_line, stale, 0
    def close(self):
        self._process = None
    def _exchange(self, payload, events):
        self._send_command(payload)
        for event in events:
            self._stdout_queue.put(json.dumps(dict(battleId=self._battle_token, **event)))
        return [self._read_event(timeout=.01) for _ in events]
    def restore(self, state):
        self.index = 0
        self._exchange(dict(type="restore", battleId=self._battle_token, snapshot=state.bridge_snapshot),
            [dict(type="restored", requested=["p1", "p2"])])
    def _bridge_request_event(self, payload, event_type):
        return self._exchange(payload, [dict(type=event_type, seed=payload["seed"])])[0]
    def terminal(self):
        if self.index == 2:
            winner = "p1" if self.terminal_line == "|win|Fixture One" else "p2" if self.terminal_line == "|win|Fixture Two" else None
            return SimpleNamespace(capped=False, winner=winner)
        return None
    def requested_players(self):
        return ("p1", "p2") if self.index == 0 else ("p1",)
    def observe(self, seat):
        return dict(seat=seat, boundary=self.index, request="synthetic-observation")
    def _choices_for_actions(self, actions):
        return {seat: "move " + str(action + 1) for seat, action in actions.items()}
    def _submit_step_choices(self, *, choices, payload):
        events = [dict(type="choice_ack", player=seat, choice=choice) for seat, choice in choices.items()]
        self.index += 1
        if self.index == 2:
            if self.stale:
                self._stdout_queue.put(json.dumps(dict(type="stream", battleId="old-token",
                    stream="omniscient", lines=["|win|Unknown"])))
                self._read_event(timeout=.01)
            if self.terminal_line is not None:
                events.append(dict(type="stream", stream="omniscient", lines=[self.terminal_line]))
            events.append(dict(type="terminal"))
        else:
            events += [dict(type="stream", stream="p1", lines=["|request|private-player-data"]),
                dict(type="ready", requested=["p1"])]
        self._exchange(payload, events)


class OwnedSyntheticTransport(SyntheticTransport):
    """New lifecycle uses real send/read hooks, synthetic ack/EOF only."""
    def __init__(self, terminal_line="|win|Fixture One", stale=False):
        super().__init__(terminal_line, stale)
        self._battle_token = None

    def _restore_local_snapshot_state(self, state):
        self.index = 0  # Synthetic policy fixture, not a parser/simulator claim.

    def _bridge_request_event(self, payload, event_type):
        if payload["type"] == "clone_snapshot":
            return self._exchange(payload, [dict(type="cloned", cloneId=payload["cloneId"],
                snapshotJsonSha256=hashlib.sha256(payload["snapshotJson"].encode()).hexdigest(),
                requested=["p1", "p2"])])[0]
        if payload["type"] == "release_clone":
            return self._exchange(payload, [dict(type="stream_end", stream=seat)
                for seat in ("omniscient", "p1", "p2")] + [dict(type="clone_released",
                cloneId=payload["cloneId"], snapshotJsonSha256=payload["snapshotJsonSha256"])])[-1]
        return super()._bridge_request_event(payload, event_type)


class EntrypointSyntheticEnv(LocalShowdownEnv):
    """Real Python state/transport/step, toy observation and action translation.

    The bridge loads ONLY the synthetic runtime; no actual engine or model.
    """
    def observe(self, seat):
        return dict(seat=seat, boundary=self._latest_turn, request="synthetic-observation")

    def _choices_for_actions(self, actions):
        return {seat: "move " + str(action + 1) for seat, action in actions.items()}


def parser_snapshot(env):
    """Empty real parser/belief objects, never reset or a source game."""
    state = snapshot()
    requests = {seat: dict(active=[{}], side=dict(id=seat)) for seat in ("p1", "p2")}
    env._parser.feed(list(state.protocol_lines))
    return LocalShowdownSnapshot(battle_token=state.battle_token, battle_id=env._battle_id,
        format_id=env._format_id, observation_format_id=env._observation_format_id,
        bridge_snapshot=dict(battle=dict(sides=[dict(pokemon=[dict(set=dict(moves=["fixture"]))])],
            log=[], ended=False), boundaryRequests=deepcopy(requests), terminalScheduled=False),
        protocol_lines=state.protocol_lines, latest_requests=deepcopy(requests), first_requests=deepcopy(requests),
        request_history={seat: (deepcopy(requests[seat]),) for seat in requests},
        replay=env._parser.snapshot(), belief_engine=env._belief_engine.clone(), latest_turn=3, terminal=None)


class OwnedLifecycleTests(unittest.TestCase):
    def setup_cell(self):
        state, env = snapshot(), OwnedSyntheticTransport()
        lifecycle = OwnedContinuationLifecycle(env, expected_snapshot=state, root_binding=deepcopy(BINDING))
        observer = ContinuationCapture(BINDING, SHA)
        return state, env, lifecycle, observer

    def run_cells(self, state, env, lifecycle, observer, **kwargs):
        with observer.installed(env):
            return paired_continuations(env=env, snapshot=state, subject="p1", actions={"raw": 0},
                evaluator=kwargs.pop("evaluator", lambda _: ([0, 1], [.25, .75])), namespace=NAMESPACE,
                root_id=BINDING["root_id"], max_boundaries=250, continuation_observer=observer,
                continuation_lifecycle=lifecycle, **kwargs)

    def test_owned_sink_waits_for_release_without_changing_rng_choices_or_labels(self):
        state, env, lifecycle, observer = self.setup_cell()
        seen = []
        def sink(row):
            self.assertIsNone(env._battle_token); self.assertIsNone(observer.active)
            self.assertEqual(observer.cells[row["action"], row["replicate"]]["records"][-1]["payload"]["type"], "clone_released")
            seen.append(row)
        result = self.run_cells(state, env, lifecycle, observer, outcome_sink=sink)
        original = paired_continuations(env=SyntheticTransport(), snapshot=state, subject="p1", actions={"raw": 0},
            evaluator=lambda _: ([0, 1], [.25, .75]), namespace=NAMESPACE, root_id=BINDING["root_id"], max_boundaries=250)
        self.assertEqual(result, original); self.assertEqual(len(seen), 8)
        tokens = [cell["battle_token"] for cell in observer.export()["cells"]]
        self.assertEqual(len(set(tokens)), 8); self.assertTrue(all(token.startswith("outcome-") for token in tokens))
        commands = [r["payload"]["type"] for cell in observer.export()["cells"] for r in cell["records"] if r["kind"] == "command"]
        self.assertNotIn("start", commands); self.assertNotIn("restore", commands); self.assertNotIn("end", commands)
        self.assertEqual(commands.count("clone_snapshot"), 8); self.assertEqual(commands.count("release_clone"), 8)

    def test_local_failure_releases_once_preserves_primary_and_poisoned_cell_never_retries(self):
        state, env, lifecycle, observer = self.setup_cell(); calls = []
        exchange = env._bridge_request_event
        def traced(payload, typ): calls.append(payload["type"]); return exchange(payload, typ)
        env._bridge_request_event = traced
        primary = RuntimeError("primary evaluator failure")
        with self.assertRaises(RuntimeError) as error:
            self.run_cells(state, env, lifecycle, observer, evaluator=lambda _: (_ for _ in ()).throw(primary))
        self.assertIs(error.exception, primary); self.assertEqual(calls, ["clone_snapshot", "release_clone"])
        self.assertIsNone(env._battle_token); self.assertTrue(lifecycle.poisoned)
        with self.assertRaisesRegex(ValueError, "failed continuation"): observer.export()
        with self.assertRaisesRegex(ValueError, "failed"):
            self.run_cells(state, env, lifecycle, observer)
        self.assertEqual(calls, ["clone_snapshot", "release_clone"])

    def test_ambiguous_clone_or_release_never_retries_and_cannot_publish(self):
        for mode in ("clone-hash", "release-hash", "release-missing", "missing-end", "duplicate-end"):
            state, env, lifecycle, observer = self.setup_cell(); calls, sinks = [], []
            exchange = env._bridge_request_event
            def faulty(payload, typ):
                calls.append(payload["type"])
                result = exchange(payload, typ)
                if (mode == "clone-hash" and typ == "cloned") or (mode == "release-hash" and typ == "clone_released"):
                    result["snapshotJsonSha256"] = "b"*64
                if typ == "clone_released":
                    if mode == "release-missing": raise RuntimeError("lost release ack")
                    if mode in ("missing-end", "duplicate-end"):
                        records = observer.active["records"]
                        end = next(r for r in records if r["payload"].get("type") == "stream_end")
                        if mode == "missing-end": records.remove(end)
                        else: records.append(deepcopy(end))
                return result
            env._bridge_request_event = faulty
            with self.subTest(mode=mode), self.assertRaises((ValueError, RuntimeError)):
                self.run_cells(state, env, lifecycle, observer, outcome_sink=sinks.append)
            self.assertEqual(sinks, []); self.assertEqual(calls.count("clone_snapshot"), 1)
            self.assertLessEqual(calls.count("release_clone"), 1); self.assertTrue(lifecycle.poisoned)
            with self.assertRaises(ValueError): observer.export()

    def test_cleanup_failure_is_secondary_to_primary(self):
        state, env, lifecycle, observer = self.setup_cell(); exchange = env._bridge_request_event
        def faulty(payload, typ):
            if typ == "clone_released": raise RuntimeError("secondary cleanup failure")
            return exchange(payload, typ)
        env._bridge_request_event = faulty
        primary = RuntimeError("primary evaluator failure")
        with self.assertRaises(RuntimeError) as error:
            self.run_cells(state, env, lifecycle, observer, evaluator=lambda _: (_ for _ in ()).throw(primary))
        self.assertIs(error.exception, primary)
        self.assertTrue(any("secondary cleanup failure" in note for note in primary.__notes__))

    def test_bound_bridge_auto_cleanup_never_scores_or_sends_second_release(self):
        for drift in (False, True):
            state, env, lifecycle, observer = self.setup_cell(); calls = []
            exchange = env._bridge_request_event
            def traced(payload, typ): calls.append(payload["type"]); return exchange(payload, typ)
            env._bridge_request_event = traced
            def failed_step(**kwargs):
                env._exchange(kwargs["payload"], [dict(type="clone_failed", cloneId=lifecycle.clone_id,
                    snapshotJsonSha256="b"*64 if drift else lifecycle.snapshot_sha, cleanupReleased=True)])
                raise RuntimeError("primary bridge failure")
            env._submit_step_choices = failed_step
            with self.subTest(drift=drift), self.assertRaisesRegex(RuntimeError, "primary bridge failure") as error:
                self.run_cells(state, env, lifecycle, observer)
            self.assertNotIn("release_clone", calls); self.assertTrue(lifecycle.poisoned)
            self.assertEqual(env._battle_token is None, not drift)
            if drift: self.assertTrue(any("owning supervisor" in note for note in error.exception.__notes__))
            with self.assertRaises(ValueError): observer.export()

    def test_initial_requests_must_match_root_and_input_drift_cannot_retry(self):
        state, env, lifecycle, observer = self.setup_cell(); exchange = env._bridge_request_event
        def bad_requests(payload, typ):
            result = exchange(payload, typ)
            if typ == "cloned": result["requested"] = ["p1"]
            return result
        env._bridge_request_event = bad_requests
        with self.assertRaisesRegex(ValueError, "request drift"):
            self.run_cells(state, env, lifecycle, observer)
        self.assertIsNone(env._battle_token)  # matching identity allowed strict failure cleanup
        state, env, lifecycle, observer = self.setup_cell(); state.latest_turn = 4
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            self.run_cells(state, env, lifecycle, observer)
        self.assertEqual(env._process.stdin.getvalue(), "")

    def test_real_python_snapshot_restoration_is_exact_and_independently_mutable(self):
        from pokezero.mcts_eval.search_over_raw_world_inputs import _snapshot_payload
        env = LocalShowdownEnv(LocalShowdownConfig(set_belief_source=False))
        state = parser_snapshot(env); frozen = digest(_snapshot_payload(state))
        env._restore_local_snapshot_state(state)
        restored = LocalShowdownSnapshot(battle_token="owned", battle_id=env._battle_id,
            format_id=env._format_id, observation_format_id=env._observation_format_id,
            bridge_snapshot=state.bridge_snapshot, protocol_lines=tuple(env._lines),
            latest_requests=env._latest_requests, first_requests=env._first_requests,
            request_history={seat: tuple(rows) for seat, rows in env._request_history.items()},
            replay=env._parser.snapshot(), belief_engine=env._belief_engine,
            latest_turn=env._latest_turn, terminal=env._terminal)
        self.assertEqual(digest(_snapshot_payload(restored)), frozen)
        env._latest_requests["p1"]["active"].append({"owned": True})
        env._request_history["p2"][0]["active"].pop()
        self.assertEqual(digest(_snapshot_payload(state)), frozen)

    def test_actual_python_to_bridge_pipeline_on_synthetic_runtime_and_no_source_start(self):
        # Share the declared synthetic Node fixture, not a real simulator build.
        text = Path(__file__).with_name("test_owned_continuation.mjs").read_text()
        with tempfile.TemporaryDirectory(prefix="pokezero-python-synthetic-bridge-") as folder:
            sim = Path(folder) / "dist" / "sim"; sim.mkdir(parents=True)
            for variable, filename in (("fakeRuntime", "index.js"), ("fakeState", "state.js")):
                content = re.search(r"const " + variable + r" = `([\s\S]*?)`;", text).group(1)
                (sim / filename).write_text(content)
            env = EntrypointSyntheticEnv(LocalShowdownConfig(showdown_root=folder, set_belief_source=False,
                read_timeout_seconds=2.0))
            state = parser_snapshot(env)
            lifecycle = OwnedContinuationLifecycle(env, expected_snapshot=state, root_binding=deepcopy(BINDING))
            observer = ContinuationCapture(BINDING, SHA)
            try:
                result = self.run_cells(state, env, lifecycle, observer)
                process = env._process
                self.assertIsNone(process.poll()); self.assertIsNone(env._battle_token)
                self.assertTrue(all(row["signed_outcome"] == 1 and row["boundaries"] == 1 for row in result["outcomes"]))
                bank = observer.export()
                for row, cell in zip(result["outcomes"], bank["cells"]):
                    complete = dict(root_id=BINDING["root_id"], **row)
                    self.assertEqual(validate_observed_outcome(complete, CONTRACT, cell,
                        expected_snapshot=state, runtime_sha256=SHA), digest(complete))
                env.close()  # Explicit synthetic-process cleanup, not a qualification receipt.
                self.assertEqual(process.returncode, 0)
            finally:
                env.close()


class ObservedOutcomeTests(unittest.TestCase):
    def test_prospective_clone_bytes_are_hashed_not_exported_as_private_snapshot(self):
        env, state = SyntheticTransport(), snapshot()
        observer = ContinuationCapture(BINDING, SHA)
        encoded = json.dumps(state.bridge_snapshot)
        with observer.installed(env):
            observer.begin_cell(state, root_id=BINDING["root_id"], action=0, replicate=0,
                subject="p1", namespace=NAMESPACE, max_boundaries=250)
            observer.command(dict(type="clone_snapshot", battleId=env._battle_token,
                cloneId="synthetic-cell", snapshotJson=encoded))
            observer.finish_cell(dict(root_id=BINDING["root_id"], action=0, replicate=0,
                status="UNSTARTED", signed_outcome=None, boundaries=0))
        bank = observer.export()
        command = bank["cells"][0]["records"][0]["payload"]
        self.assertNotIn("snapshotJson", command)
        self.assertEqual(command["snapshotJsonSha256"], hashlib.sha256(encoded.encode()).hexdigest())
        self.assertNotIn("private-fixture", json.dumps(bank))

    def capture(self, terminal_line="|win|Fixture One", stale=False, isolated=True):
        state = snapshot()
        env = (OwnedSyntheticTransport if isolated else SyntheticTransport)(terminal_line, stale)
        lifecycle = OwnedContinuationLifecycle(env, expected_snapshot=state, root_binding=BINDING) if isolated else None
        observer = ContinuationCapture(BINDING, SHA)
        with observer.installed(env):
            result = paired_continuations(env=env, snapshot=state, subject="p1", actions={"raw": 0, "search": 1},
                evaluator=lambda _: ([0, 1], [.25, .75]), namespace=NAMESPACE,
                root_id=BINDING["root_id"], max_boundaries=250, continuation_observer=observer,
                continuation_lifecycle=lifecycle)
        rows = [dict(root_id=BINDING["root_id"], **row) for row in result["outcomes"]]
        return state, rows, observer.export(), env

    def reader(self, row, capture, state):
        return validate_observed_outcome(row, CONTRACT, capture, expected_snapshot=state, runtime_sha256=SHA)

    def test_real_transport_seams_capture_and_validate_all_shared_cells(self):
        state, rows, bank, env = self.capture()
        self.assertEqual(len(bank["cells"]), 16)
        validator = OutcomeReceiptValidator(bank, expected_snapshot=state, root_binding=BINDING,
            runtime_sha256=SHA, validate_provenance_and_cleanup=digest)  # synthetic ownership only
        self.assertTrue(all(validator(row, CONTRACT) == digest(row) for row in rows))
        self.assertIsNone(env._bridge_evidence_observer)
        self.assertFalse(bank["runtime_qualified"])
        self.assertFalse(bank["cleanup_validated"])
        serialized = json.dumps(bank)
        self.assertNotIn("private-player-data", serialized)
        self.assertNotIn("private-fixture", serialized)
        self.assertIn('"snapshotJsonSha256"', serialized)
        self.assertIsNone(env._battle_token)

    def test_protocol_loss_win_and_tie_not_terminal_state_guesses(self):
        for terminal, value in (("|win|Fixture One", 1), ("|win|Fixture Two", -1), ("|tie", 0)):
            with self.subTest(terminal=terminal):
                state, rows, bank, _ = self.capture(terminal)
                self.assertEqual(rows[0]["signed_outcome"], value)
                self.assertEqual(self.reader(rows[0], bank["cells"][0], state), digest(rows[0]))
        state, rows, bank, _ = self.capture(None)
        self.assertEqual(rows[0]["signed_outcome"], 0)  # producer's bare-terminal fallback
        self.assertIsNone(self.reader(rows[0], bank["cells"][0], state))

    def test_unvalidated_ownership_missing_evidence_and_noncomplete_stay_uncertain(self):
        state, rows, bank, _ = self.capture()
        validator = OutcomeReceiptValidator(bank, expected_snapshot=state, root_binding=BINDING,
            runtime_sha256=SHA, validate_provenance_and_cleanup=lambda _: None)
        self.assertIsNone(validator(rows[0], CONTRACT))
        self.assertIsNone(self.reader(rows[0], None, state))
        for status in ("CAPPED", "REFUSED", "UNSTARTED"):
            self.assertIsNone(self.reader(dict(rows[0], status=status, signed_outcome=None), bank["cells"][0], state))

    def test_approved_capture_cannot_be_mutated_to_an_opposite_label(self):
        state, rows, bank, _ = self.capture()
        validator = OutcomeReceiptValidator(bank, expected_snapshot=state, root_binding=BINDING,
            runtime_sha256=SHA, validate_provenance_and_cleanup=digest)
        row = dict(rows[0], signed_outcome=-1)
        cell = validator.cells[0, 0]
        next(r["payload"] for r in cell["records"] if r["payload"].get("stream") == "omniscient")["lines"] = ["|win|Fixture Two"]
        cell["row_sha256"] = digest(row)
        with self.assertRaisesRegex(ValueError, "approved.*changed"): validator(row, CONTRACT)
        validator = OutcomeReceiptValidator(bank, expected_snapshot=state, root_binding=BINDING,
            runtime_sha256=SHA, validate_provenance_and_cleanup=digest)
        validator.identity["information_key"] = "changed"
        with self.assertRaisesRegex(ValueError, "approved.*changed"): validator(rows[0], CONTRACT)

    def test_reused_shell_cannot_score_even_when_generic_provenance_callback_clears(self):
        state, rows, bank, _ = self.capture(isolated=False)
        validator = OutcomeReceiptValidator(bank, expected_snapshot=state, root_binding=BINDING,
            runtime_sha256=SHA, validate_provenance_and_cleanup=digest)
        self.assertTrue(all(validator(row, CONTRACT) is None for row in rows))
        cell = deepcopy(bank["cells"][0]); cell["battle_token"] = state.battle_token
        for record in cell["records"]:
            if "battleId" in record["payload"]: record["payload"]["battleId"] = state.battle_token
        self.assertIsNone(self.reader(rows[0], cell, state))

    def test_provenance_callback_cannot_change_external_inputs_or_original_bank(self):
        for mode in ("binding", "snapshot", "bank"):
            state, rows, bank, _ = self.capture()
            binding = deepcopy(BINDING)
            def bad_callback(supplied):
                if mode == "binding": binding["source_seed"] = 456
                if mode == "snapshot": state.latest_turn = 99
                if mode == "bank": bank["cells"].pop()
                return digest(supplied)
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "callback changed"):
                OutcomeReceiptValidator(bank, expected_snapshot=state, root_binding=binding,
                    runtime_sha256=SHA, validate_provenance_and_cleanup=bad_callback)

    def test_supplied_label_cannot_override_actual_protocol_even_with_rehashed_receipt(self):
        state, rows, bank, _ = self.capture()
        row, cell = dict(rows[0], signed_outcome=-1), bank["cells"][0]
        cell["row_sha256"] = digest(row)
        with self.assertRaisesRegex(ValueError, "contradicts"):
            self.reader(row, cell, state)

    def test_stale_winner_ignored_unknown_duplicate_conflicting_winners_rejected(self):
        state, rows, bank, _ = self.capture(stale=True)
        self.reader(rows[0], bank["cells"][0], state)
        for lines in (["|win|Unknown"], ["|win|Fixture One", "|tie"], ["|win|Fixture One"]*2):
            cell = deepcopy(bank["cells"][0])
            next(r["payload"] for r in cell["records"] if r["kind"] == "event"
                and r["payload"].get("stream") == "omniscient" and r["payload"].get("battleId") == cell["battle_token"])["lines"] = lines
            with self.subTest(lines=lines), self.assertRaises(ValueError): self.reader(rows[0], cell, state)

    def test_transport_mutations_rejected_by_independent_reader(self):
        state, rows, bank, _ = self.capture()
        original = bank["cells"][0]
        for mode in ("snapshot", "seed", "ack-seed", "choices", "ack-choice", "missing-ack", "policy", "later-policy", "bool-action", "request", "missing-boundary", "terminal-early", "count", "identity", "bool-replicate"):
            cell = deepcopy(original)
            records = cell["records"]
            find = lambda kind, typ: next(r["payload"] for r in records if r["kind"] == kind and r["payload"].get("type") == typ)
            if mode == "snapshot": find("command", "clone_snapshot")["snapshotJsonSha256"] = "b"*64
            if mode == "seed": find("command", "reseed")["seed"] = "1,2,3,4"
            if mode == "ack-seed": find("event", "reseeded")["seed"] = "1,2,3,4"
            if mode == "choices": find("command", "choices")["choices"]["p1"] = "move 99"
            if mode == "ack-choice": find("event", "choice_ack")["choice"] = "move 99"
            if mode == "missing-ack": records.remove(next(r for r in records if r["payload"].get("type") == "choice_ack"))
            if mode == "policy": next(r["payload"] for r in records if r["kind"] == "boundary")["actions"]["p1"] = 1
            if mode == "later-policy": [r["payload"] for r in records if r["kind"] == "boundary"][1]["actions"]["p1"] = 0
            if mode == "bool-action": [r["payload"] for r in records if r["kind"] == "actions"][1]["actions"]["p1"] = True
            if mode == "request": find("event", "cloned")["requested"] = ["p1"]
            if mode == "missing-boundary": records.pop()
            if mode == "terminal-early": records.insert(0, dict(kind="event", payload=dict(type="stream", battleId=cell["battle_token"], stream="omniscient", lines=["|win|Fixture One"])))
            if mode == "count":
                row = dict(rows[0], boundaries=1); cell["row_sha256"] = digest(row)
            if mode == "identity": cell["law"]["root_id"] = "wrong"
            if mode == "bool-replicate": cell["law"]["replicate"] = False
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.reader(row if mode == "count" else rows[0], cell, state)

    def test_reader_requires_ordered_bound_clone_and_strict_release_not_distinct_token_only(self):
        state, rows, bank, _ = self.capture(); original = bank["cells"][0]
        for mode in ("clone-id", "clone-ack-id", "release-id", "release-ack-hash", "no-release", "legacy-ended",
                "duplicate-release", "missing-end", "duplicate-end", "early-end", "post-release", "source-token"):
            cell = deepcopy(original); records = cell["records"]
            find = lambda typ: next(r for r in records if r["payload"].get("type") == typ)
            if mode == "clone-id": find("clone_snapshot")["payload"]["cloneId"] = "wrong"
            if mode == "clone-ack-id": find("cloned")["payload"]["cloneId"] = "wrong"
            if mode == "release-id": find("release_clone")["payload"]["cloneId"] = "wrong"
            if mode == "release-ack-hash": find("clone_released")["payload"]["snapshotJsonSha256"] = "b"*64
            if mode == "no-release": records.remove(find("release_clone"))
            if mode == "legacy-ended": find("clone_released")["payload"]["type"] = "ended"
            if mode == "duplicate-release": records.append(deepcopy(find("clone_released")))
            if mode == "missing-end": records.remove(find("stream_end"))
            if mode == "duplicate-end": records.insert(-1, deepcopy(find("stream_end")))
            if mode == "early-end": records.insert(2, records.pop(records.index(find("stream_end"))))
            if mode == "post-release": records.append(dict(kind="event", payload=dict(type="stream",
                battleId=cell["battle_token"], stream="p1", lines_sha256=SHA)))
            if mode == "source-token":
                cell["battle_token"] = state.battle_token
                for record in records:
                    if "battleId" in record["payload"]: record["payload"]["battleId"] = state.battle_token
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.reader(rows[0], cell, state)
        # Rehashing a legacy restore capture or merely assigning fresh tokens
        # cannot synthesize the missing prospective lifecycle.
        cell = deepcopy(original)
        records = cell["records"]
        clone = next(r for r in records if r["payload"].get("type") == "clone_snapshot")
        clone["payload"] = dict(type="restore", battleId=cell["battle_token"], snapshot_sha256=cell["bridge_snapshot_sha256"])
        self.assertIsNone(self.reader(rows[0], cell, state))

    def test_frozen_policy_contract_and_nonterminal_root_required(self):
        state, rows, bank, _ = self.capture()
        for key, value in (("policy", "different"), ("replicates", True), ("max_boundaries", 200),
                ("selection_seed", True), ("namespace", "")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_observed_outcome(rows[0], dict(CONTRACT, **{key: value}), bank["cells"][0],
                    expected_snapshot=state, runtime_sha256=SHA)

    def test_actual_reconciliation_uses_reader_and_preserves_unknown_terminal_bounds(self):
        from test_search_over_raw_evidence import ReconciledEvidenceTest
        fixture = ReconciledEvidenceTest(); fixture.setUp()
        state, rows, bank, _ = self.capture()
        # Join the real reconciliation fixture to this exact captured root.
        fixture.slot["root_id"] = BINDING["root_id"]
        fixture.bundle["root_binding"] = deepcopy(BINDING)
        fixture.bundle["continuation_contract"] = deepcopy(CONTRACT)
        for key, selected in fixture.bundle["selections"].items():
            selected["root_id"] = BINDING["root_id"] + ":" + key
            selected["statistics_root_id"] = BINDING["root_id"]
        fixture.bundle["outcomes"] = rows
        validator = OutcomeReceiptValidator(bank, expected_snapshot=state, root_binding=BINDING,
            runtime_sha256=SHA, validate_provenance_and_cleanup=digest)
        accepted = fixture.reconcile(outcome_validator=validator)
        self.assertEqual(accepted["validated_action_cells"], 16)
        self.assertEqual(accepted["root_intervals"][fixture.inc.identity], [0., 0.])
        state, rows, bank, _ = self.capture(None)
        fixture.bundle["outcomes"] = rows
        validator = OutcomeReceiptValidator(bank, expected_snapshot=state, root_binding=BINDING,
            runtime_sha256=SHA, validate_provenance_and_cleanup=digest)
        uncertain = fixture.reconcile(outcome_validator=validator)
        self.assertEqual(uncertain["validated_action_cells"], 0)
        self.assertEqual(uncertain["root_intervals"][fixture.inc.identity], [-1., 1.])

    def test_expected_snapshot_and_runtime_are_not_taken_from_capture(self):
        state, rows, bank, _ = self.capture()
        other = deepcopy(state); other.request_history["p1"] = ["different-actual-root"]
        with self.assertRaisesRegex(ValueError, "snapshot"): self.reader(rows[0], bank["cells"][0], other)
        with self.assertRaises(ValueError): OutcomeReceiptValidator(bank, expected_snapshot=state,
            root_binding=BINDING, runtime_sha256="b"*64, validate_provenance_and_cleanup=digest)
        with self.assertRaises(ValueError): OutcomeReceiptValidator(bank, expected_snapshot=state,
            root_binding=BINDING, runtime_sha256=SHA, validate_provenance_and_cleanup=lambda _: "b"*64)

    def test_capture_failure_default_off_and_cached_labels_not_upgraded(self):
        state, env = snapshot(), SyntheticTransport()
        original = paired_continuations(env=env, snapshot=state, subject="p1", actions={"raw": 0},
            evaluator=lambda _: ([0, 1], [.25, .75]), namespace=NAMESPACE, root_id=BINDING["root_id"], max_boundaries=250)
        observer = ContinuationCapture(BINDING, SHA)
        cached = dict(original, namespace=NAMESPACE, subject="p1", max_boundaries=250, replicates=8)
        with observer.installed(env):
            paired_continuations(env=env, snapshot=state, subject="p1", actions={"raw": 0},
                evaluator=lambda _: self.fail("cached cell cannot run"), namespace=NAMESPACE,
                root_id=BINDING["root_id"], max_boundaries=250, cached_outcomes=cached, continuation_observer=observer)
        self.assertEqual(observer.export()["cells"], [])
        observer = ContinuationCapture(BINDING, SHA)
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"), observer.installed(env):
            paired_continuations(env=env, snapshot=state, subject="p1", actions={"raw": 0},
                evaluator=lambda _: (_ for _ in ()).throw(RuntimeError("synthetic failure")), namespace=NAMESPACE,
                root_id=BINDING["root_id"], max_boundaries=250, continuation_observer=observer)
        self.assertIsNone(env._bridge_evidence_observer)
        with self.assertRaisesRegex(ValueError, "failed continuation"): observer.export()


if __name__ == "__main__": unittest.main()
