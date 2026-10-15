"""Synthetic prospective codec tests only; no native builds or source games."""
from copy import deepcopy
import hashlib
import json
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pokezero.mcts_eval.search_over_raw_native_atomic_compact import (
    SCHEMA, snapshot_patch, validate_compact_native_report, validate_timed_compact_native_report,
)
from pokezero.mcts_eval.search_over_raw_native_atomic_work import (
    _bits, _float, _f32, _snapshot, digest, state_hash, validate_native_report,
)
from test_search_over_raw_native_atomic_work import (
    native_report, reserved_cancelled_report, terminal_report,
)
from test_search_over_raw_timed_native import timed_report
from test_search_over_raw_timed_native import timed_arguments
from test_search_over_raw_world_inputs import BINDING, arguments, construct, native_manifest
from pokezero.mcts_eval.search_over_raw_world_inputs import (
    NativeInputHandle, NativeWorldInputs, validate_fixed_work_inputs, validate_timed_native_inputs,
)


def compact_fixture(report):
    """Test-only conversion, never used to admit a native or historical run."""
    result = deepcopy({key: value for key, value in report.items() if key != "atomic_work"})
    source = report["atomic_work"]
    work = deepcopy({key: value for key, value in source.items() if key != "rounds"})
    work["schema"] = SCHEMA
    work["initial"] = deepcopy(source["rounds"][0]["before_collection"]
        if source["rounds"] else source["final"])
    work["rounds"] = []
    for round in source["rounds"]:
        compact = dict(ordinal=round["ordinal"], status=round["status"],
            before_sha256=digest(round["before_collection"]),
            collection_edits=snapshot_patch(round["before_collection"], round["after_collection"]),
            collection_sha256=digest(round["after_collection"]),
            after_sha256=digest(round["after"]), accepted=round["accepted"])
        if round["status"] == "CANCELLED":
            compact["attempted_paths"] = deepcopy(round["attempted_paths"])
        else:
            compact["row_value_bits"] = deepcopy(round["row_value_bits"])
            compact["commits"] = [dict(path=deepcopy(commit["path"]), end=deepcopy(commit["end"]),
                before_sha256=digest(commit["before"]), after_sha256=digest(commit["after"]),
                root_value_bits=commit["root_value_bits"]) for commit in round["commits"]]
        work["rounds"].append(compact)
    result["atomic_work"] = work
    return result


def growing_report(expansions):
    """Independently authored two-depth backups with a growing chance arena.

    Each new joint edge is priced at exactly .5; one root edge enters a child
    with `expansions` legal joint options. Expected totals are dyadic formulas,
    not outputs of the replay implementation under test.
    """
    report = native_report(1, batch_size=1)
    first = report["atomic_work"]["rounds"][0]
    def branch(pending=None, child=None):
        return dict(probability_bits=_bits(1.), value_sum_bits=_bits(.5 if pending is None else 0.),
            visits=1, pending_row=pending, child=child, terminal_bits=None, no_expand=False)
    for snapshot in (first["after_collection"], first["commits"][0]["before"]):
        snapshot["chances"] = [[branch(pending=0)]]
    for snapshot in (first["after"], first["commits"][0]["after"]):
        snapshot["chances"] = [[branch()]]
        snapshot["decisions"][0]["side_one"][0]["total_bits"] = _bits(.5)
        snapshot["decisions"][0]["side_two"][0]["total_bits"] = _bits(.5)
    first["row_value_bits"] = [_bits(.5)]
    first["commits"][0]["root_value_bits"] = _bits(.5)
    # Collected reservations retain total 0 for p1 and 1 for p2.
    first["commits"][0]["before"] = deepcopy(first["after_collection"])
    state = deepcopy(first["after"])
    for index in range(expansions):
        before = deepcopy(state)
        collected = deepcopy(state)
        if index == 0:
            collected["decisions"].append(dict(visits=0, depth=1, children=[],
                side_one=[dict(visits=0, total_bits=0) for _ in range(expansions)],
                side_two=[dict(visits=0, total_bits=0) for _ in range(expansions)]))
            collected["chances"][0][0]["child"] = 1
        chance_index = len(collected["chances"])
        collected["chances"].append([branch(pending=0)])
        collected["decisions"][1]["children"].append([index, index, chance_index])
        path = [dict(decision=0, side_one=0, side_two=0, chance=0, branch=0),
            dict(decision=1, side_one=index, side_two=index, chance=chance_index, branch=None)]
        for node_index, option in ((0, 0), (1, index)):
            node = collected["decisions"][node_index]
            node["visits"] += 1
            for seat in ("side_one", "side_two"):
                node[seat][option]["visits"] += 1
            node["side_two"][option]["total_bits"] = _bits((index+1)*.5+1 if node_index == 0 else 1.)
        after = deepcopy(collected)
        root = after["decisions"][0]
        for seat in ("side_one", "side_two"):
            root[seat][0]["total_bits"] = _bits((index+2)*.5)
            after["decisions"][1][seat][index]["total_bits"] = _bits(.5)
        after["chances"][chance_index][0].update(pending_row=None, value_sum_bits=_bits(.5))
        after["chances"][0][0].update(visits=index+2, value_sum_bits=_bits((index+2)*.5))
        commit = dict(path=path, end=dict(kind="expanded"), before=deepcopy(collected),
            after=deepcopy(after), root_value_bits=_bits(.5))
        report["atomic_work"]["rounds"].append(dict(ordinal=index+1, status="COMMITTED",
            before_collection=before, after_collection=collected, row_value_bits=[_bits(.5)],
            commits=[commit], after=after, accepted=1))
        state = deepcopy(after)
    count = expansions+1
    report.update(iterations=count, root_value=.5,
        side_one=[dict(move="move", visits=count, q=.5)], side_two=[dict(move="move", visits=count, q=.5)])
    joint = report["joint_action_witness"]
    joint.update(root_completed_traversals=count, tree_completed_selections=1+2*expansions,
        tree_distinct_node_pairs=count, root_side_one_visits=[count], root_side_two_visits=[count],
        root_pairs=[dict(side_one_option=0, side_two_option=0, completed_visits=count)])
    report["atomic_work"].update(final=state, requested_iterations=count, joint_action_witness=deepcopy(joint))
    return report


def bindings(report):
    work = report["atomic_work"]
    return dict(state_sha256=state_hash("complete-state"), requested_iterations=work["requested_iterations"],
        batch_size=work["batch_size"], chance_seed=7)


def intermediate_invalid_report():
    """Astra's adversarial two-depth prefix, valid again at the final tree.

    Probability mass lies within the existing tolerance. Four near-one root
    expectations overflow the p2 virtual-loss invariant; a lower child price
    then restores valid totals. Snapshots are authored independently of replay.
    """
    report = native_report(5, batch_size=4); work = report["atomic_work"]
    initial = deepcopy(work["rounds"][0]["before_collection"])
    def stats(visits, total): return dict(visits=visits, total_bits=_bits(total))
    def branch(probability, value, pending=None, child=None):
        return dict(probability_bits=_bits(probability), value_sum_bits=_bits(value), visits=1,
            pending_row=pending, child=child, terminal_bits=None, no_expand=False)
    def step(decision, chance, selected):
        return dict(decision=decision, side_one=0, side_two=0, chance=chance, branch=selected)
    def authored_commit(state, path, end, prices=None):
        before = deepcopy(state)
        if end["kind"] == "expanded":
            for item in state["chances"][path[-1]["chance"]]:
                if item["pending_row"] is not None:
                    item["value_sum_bits"] = _bits(prices[item["pending_row"]]); item["pending_row"] = None
            value = None
        else: value = 1.
        for selected in reversed(path):
            branches = state["chances"][selected["chance"]]
            if selected["branch"] is not None:
                item = branches[selected["branch"]]; item["visits"] += 1
                item["value_sum_bits"] = _bits(_float(item["value_sum_bits"])+value)
            value = 0.
            for item in branches:
                mean = _f32(_float(item["value_sum_bits"])/_f32(item["visits"]))
                value = _f32(value+_f32(_float(item["probability_bits"])*mean))
            node = state["decisions"][selected["decision"]]
            node["side_one"][0]["total_bits"] = _bits(_float(node["side_one"][0]["total_bits"])+value)
            node["side_two"][0]["total_bits"] = _bits(_float(node["side_two"][0]["total_bits"])+_f32(value-1.))
        return dict(path=path, end=end, before=before, after=deepcopy(state), root_value_bits=_bits(value))
    probability = .5000152587890625
    collected = deepcopy(initial)
    collected["decisions"][0].update(visits=1, side_one=[stats(1, 0)], side_two=[stats(1, 1)],
        children=[[0, 0, 0]])
    collected["chances"] = [[branch(probability, 0, 0), branch(probability, 0, 1)]]
    state = deepcopy(collected)
    first_commit = authored_commit(state, [step(0, 0, None)], dict(kind="expanded"), [1., 1.])
    first = dict(ordinal=0, status="COMMITTED", before_collection=initial, after_collection=collected,
        row_value_bits=[_bits(1.), _bits(1.)], commits=[first_commit], after=deepcopy(state), accepted=1)
    before = deepcopy(state); collected = deepcopy(state); root = collected["decisions"][0]
    root["visits"] = 5; root["side_one"][0]["visits"] = 5; root["side_two"][0]["visits"] = 5
    root["side_two"][0]["total_bits"] = _bits(_float(root["side_two"][0]["total_bits"])+4.)
    collected["decisions"].append(dict(visits=1, depth=1, side_one=[stats(1, 0)], side_two=[stats(1, 1)],
        children=[[0, 0, 1]]))
    collected["chances"][0][0]["child"] = 1; collected["chances"].append([branch(1., 0., 0)])
    state = deepcopy(collected)
    commits = [authored_commit(state, [step(0, 0, 0)], dict(kind="ready", value_bits=_bits(1.)))
        for _ in range(3)]
    commits.append(authored_commit(state, [step(0, 0, 0), step(1, 1, None)], dict(kind="expanded"), [0.]))
    second = dict(ordinal=1, status="COMMITTED", before_collection=before, after_collection=collected,
        row_value_bits=[_bits(0.)], commits=commits, after=deepcopy(state), accepted=4)
    work.update(rounds=[first, second], final=deepcopy(state))
    report["joint_action_witness"].update(tree_completed_selections=6, tree_distinct_node_pairs=2)
    work["joint_action_witness"] = deepcopy(report["joint_action_witness"])
    for seat in ("one", "two"):
        mean = _f32(_float(state["decisions"][0]["side_"+seat][0]["total_bits"])/5.)
        report["side_"+seat][0]["q"] = float(f"{mean:.6f}")
    report["root_value"] = report["side_one"][0]["q"]
    return report


class CompactNativeTests(unittest.TestCase):
    def test_valid_final_cannot_hide_invalid_correctly_hashed_intermediate_tree(self):
        source = intermediate_invalid_report()
        _snapshot(source["atomic_work"]["final"])
        prefix = source["atomic_work"]["rounds"][1]["commits"][2]["after"]
        with self.assertRaisesRegex(ValueError, "invalid native total"): _snapshot(prefix)
        with self.assertRaisesRegex(ValueError, "invalid native total"):
            validate_native_report(source, **bindings(source))
        compact = compact_fixture(source)
        with self.assertRaisesRegex(ValueError, "invalid native total"):
            validate_compact_native_report(compact, **bindings(compact))

    def test_recomputed_fingerprints_cannot_bless_changed_arithmetic_or_collection(self):
        for mode in ("backup", "oldchance", "unrelated", "cancel"):
            source = reserved_cancelled_report() if mode == "cancel" else native_report(64)
            rounds = source["atomic_work"]["rounds"]
            if mode == "backup":
                rounds[0]["commits"][0]["after"]["decisions"][0]["side_one"][0]["total_bits"] = _bits(.25)
            elif mode == "oldchance":
                rounds[1]["after_collection"]["chances"][0][0]["value_sum_bits"] = _bits(4.5)
            elif mode == "unrelated":
                rounds[0]["commits"][0]["after"]["chances"][0][1]["probability_bits"] = _bits(.5)
            else:
                rounds[0]["after"]["decisions"][0]["side_two"][0]["total_bits"] = _bits(.5)
            # All fingerprints are honestly recomputed from the altered trees;
            # rejection must arise from replay/collection/rollback, not stale SHA.
            compact = compact_fixture(source)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_compact_native_report(compact, **bindings(compact))

    def test_timed_prefix_keeps_complete_batches_and_deadline_laws(self):
        for source in (timed_report(), timed_report(0), timed_report(cancelled=True),
                timed_report(64, exhausted=False), reserved_cancelled_report()):
            compact = compact_fixture(source)
            result = validate_timed_compact_native_report(compact, **bindings(compact), time_budget_ms=8)
            self.assertEqual(result["completed_iterations"], source["iterations"])
            self.assertFalse(result["owning_process_validated"])
            self.assertFalse(result["on_time_deployability_established"])
        for source in (timed_report(3), timed_report(0, exhausted=False), timed_report(elapsed=7.9)):
            compact = compact_fixture(source)
            with self.assertRaises(ValueError):
                validate_timed_compact_native_report(compact, **bindings(compact), time_budget_ms=8)
        source = timed_report(1); source["atomic_work"]["batch_size"] = 2
        compact = compact_fixture(source)
        self.assertEqual(validate_timed_compact_native_report(compact, **bindings(compact),
            time_budget_ms=8)["completed_iterations"], 1)

    def test_exact_backups_terminal_prices_multibatch_and_zero_work(self):
        for source in (native_report(), native_report(64), native_report(0),
                terminal_report(0.), terminal_report(1.), growing_report(5)):
            original = deepcopy(source); compact = compact_fixture(source); saved = deepcopy(compact)
            old = validate_native_report(source, **bindings(source))
            new = validate_compact_native_report(compact, **bindings(source))
            for key in ("completed_iterations", "committed_rounds", "independent_terminal_labels", "runtime_qualified"):
                self.assertEqual(old[key], new[key])
            self.assertEqual(source, original); self.assertEqual(compact, saved)
            self.assertEqual(new["schema"], SCHEMA)

    def test_nonempty_cancellation_requires_actual_restored_fingerprint(self):
        compact = compact_fixture(reserved_cancelled_report())
        self.assertEqual(validate_compact_native_report(compact, **bindings(compact))["completed_iterations"], 0)
        for mode in ("accepted", "restore", "deadline", "excess"):
            altered = deepcopy(compact); round = altered["atomic_work"]["rounds"][0]
            if mode == "accepted": round["accepted"] = 1
            if mode == "restore": round["after_sha256"] = round["collection_sha256"]
            if mode == "deadline": altered["time_budget_exhausted"] = False
            if mode == "excess": round["attempted_paths"].append(deepcopy(round["attempted_paths"][0]))
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_compact_native_report(altered, **bindings(altered))

    def test_actual_call_scope_and_encoding_are_not_substitutable(self):
        compact = compact_fixture(native_report())
        with self.assertRaises(ValueError): validate_native_report(compact, **bindings(compact))
        with self.assertRaises(ValueError): validate_compact_native_report(native_report(), **bindings(compact))
        for key, value in dict(state_sha256="a"*64, requested_iterations=32, batch_size=1, chance_seed=9).items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_compact_native_report(compact, **dict(bindings(compact), **{key: value}))
        for key in ("runtime_qualified", "independent_terminal_labels"):
            changed = deepcopy(compact); changed["atomic_work"][key] = True
            with self.assertRaises(ValueError): validate_compact_native_report(changed, **bindings(changed))

    def test_path_arithmetic_prefix_tail_and_joint_tampering_refuse(self):
        for mode in ("value", "path", "before", "after", "tail", "joint", "price", "missing", "reversed"):
            report = compact_fixture(native_report()); work = report["atomic_work"]
            round = work["rounds"][0]; commit = round["commits"][0]
            if mode == "value": commit["root_value_bits"] = _bits(.25)
            if mode == "path": commit["path"][0]["chance"] = 9
            if mode == "before": commit["before_sha256"] = "a"*64
            if mode == "after": commit["after_sha256"] = "a"*64
            if mode == "tail": work["final"]["decisions"][0]["side_one"][0]["total_bits"] = _bits(9.)
            if mode == "joint": work["joint_action_witness"]["tree_completed_selections"] += 1
            if mode == "price": round["row_value_bits"][0] = _bits(.75)
            if mode == "missing": round["commits"].pop()
            if mode == "reversed": round["commits"].reverse()
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_compact_native_report(report, **bindings(report))

    def test_noncanonical_duplicate_boolean_or_unknown_collection_edits_refuse(self):
        compact = compact_fixture(native_report())
        for mode in ("duplicate", "reverse", "boolpath", "unknown", "baseline", "container"):
            report = deepcopy(compact); edits = report["atomic_work"]["rounds"][0]["collection_edits"]
            if mode == "duplicate": edits.append(deepcopy(edits[0]))
            if mode == "reverse": edits.reverse()
            if mode == "boolpath": edits[0]["path"] = ["decisions", False, "visits"]
            if mode == "unknown": edits[0]["path"] = ["unknown"]
            if mode == "baseline": edits[0]["before_length"] = True
            if mode == "container": edits[0] = dict(kind="set", path=["chances"], before=[], after=[])
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_compact_native_report(report, **bindings(report))

    def test_growing_two_depth_payload_is_bounded_without_legacy_expansion(self):
        measurements = []
        for size in (32, 64, 128):
            source = growing_report(size)
            validate_native_report(source, **bindings(source))
            compact = compact_fixture(source)
            started = time.perf_counter()
            result = validate_compact_native_report(compact, **bindings(compact))
            elapsed = time.perf_counter()-started
            encode = lambda value: len(json.dumps(value, separators=(",", ":"), allow_nan=False).encode())
            # Actual-native terminal reproduction is still pending. This
            # explicitly synthetic envelope counts BOTH parsed and text copies.
            envelope = dict(native_report=compact, native_report_text=json.dumps(compact,
                separators=(",", ":"), allow_nan=False))
            measurements.append((size, encode(source), encode(compact), elapsed, encode(envelope)))
            self.assertEqual(result["completed_iterations"], size+1)
            self.assertLess(encode(compact), 16*1024*1024)
            self.assertLess(encode(envelope), 16*1024*1024)
        self.assertGreater(measurements[-1][1]/measurements[-2][1], 3.)
        self.assertLess(measurements[-1][2]/measurements[-2][2], 2.2)
        self.assertLess(measurements[-1][2], measurements[-1][1]/10)
        print("SYNTHETIC_COMPACT_BYTES_AND_READER_SECONDS", measurements)


def observer_fixture(*, timed=True, completed=16, compact=True, cancelled=False):
    """Original observer/dispatch/finish; native return alone is mocked."""
    source = timed_report(completed, requested=64, cancelled=cancelled) if timed else native_report(64)
    report = compact_fixture(source) if compact else source
    raw = json.dumps(report, indent=2)
    recorder = NativeWorldInputs(BINDING, 16, deadline_limited=timed, record_atomic_work=True,
        record_timed_atomic_work=timed, record_compact_atomic_work=compact)
    for _ in range(4): construct(recorder)
    base = SimpleNamespace(search_batched_multi_encoded=Mock(return_value=raw))
    args = timed_arguments(requested=64) if timed else arguments(iterations=64)
    NativeInputHandle(base, recorder).search_batched_multi_encoded(*args)
    work = report["iterations"]
    manifest = native_manifest(recorder, work)
    evidence = dict(total_iterations=work, consumed_world_inputs=manifest)
    if timed:
        readout = manifest["calls"][0]["atomic_work_validation"]
        evidence["engine_mcts"] = dict(time_budget=dict(scope="whole_model_decision", requested_ms=1000,
            native_batch_guard_ms=64, deadline_elapsed_ms=1001., deadline_overshoot_ms=1., exhausted=True,
            worlds_budget_skipped=0 if work else 4,
            native_invocations=[dict(status="completed", world_seed=7, multiplicity=4,
                requested_iterations=64, completed_iterations=work, remaining_iterations=64-work,
                time_budget_ms=8, time_budget_elapsed_ms=readout["native_elapsed_ms"],
                time_budget_batch_overshoot_ms=readout["native_overshoot_ms"], time_budget_exhausted=True,
                root_visits=dict(side_one=work, side_two=work))]))
    return evidence, recorder, base, args, raw


def read_observer(evidence, *, timed=True, compact=True):
    if timed:
        return validate_timed_native_inputs(evidence, binding=BINDING, iterations_per_world=16,
            batch_size=16, expected_worlds=4, nominal_seconds=1., adapter_elapsed_seconds=1.2,
            compact_native_atomic_work=compact)
    return validate_fixed_work_inputs(evidence, arm="incumbent", binding=BINDING,
        statistics_root_id="authored-only-root", compact_native_atomic_work=compact)


class CompactNativeObserverTests(unittest.TestCase):
    def test_actual_fixed_dispatch_raw_bytes_allocation_and_independent_readback(self):
        evidence, recorder, base, args, raw = observer_fixture(timed=False)
        self.assertEqual(base.search_batched_multi_encoded.call_args.args, tuple(args))
        self.assertEqual(base.search_batched_multi_encoded.call_args.kwargs,
            dict(record_atomic_work=True, record_compact_atomic_work=True))
        manifest = evidence["consumed_world_inputs"]; call = manifest["calls"][0]
        self.assertEqual(manifest["native_atomic_encoding"], SCHEMA)
        self.assertIs(call["compact_atomic_observation"], True)
        self.assertEqual(call["native_report_utf8"], raw)
        self.assertEqual(call["report_utf8_sha256"], hashlib.sha256(raw.encode()).hexdigest())
        self.assertEqual(call["native_call_parameters"], dict(state_sha256=state_hash("complete-state"),
            requested_iterations=64, batch_size=16, chance_seed=7))
        self.assertEqual(call["atomic_work_validation"]["schema"], SCHEMA)
        self.assertFalse(call["atomic_work_validation"]["runtime_qualified"])
        self.assertEqual(read_observer(evidence, timed=False), digest(manifest))
        self.assertEqual(len(recorder.rows), 4); self.assertTrue(all(row["consumed"] for row in recorder.rows))
        with self.assertRaises(ValueError): read_observer(evidence, timed=False, compact=False)

    def test_timed_prefix_zero_and_cancelled_tail_preserve_full_input_roster(self):
        for count, cancelled in ((16, False), (16, True), (0, False), (0, True)):
            evidence, recorder, base, args, _ = observer_fixture(completed=count, cancelled=cancelled)
            result = read_observer(evidence)
            self.assertEqual(result["completed_iterations"], count)
            self.assertFalse(result["runtime_qualified"])
            self.assertFalse(result["scientific_collection_authorized"])
            self.assertEqual(len(evidence["consumed_world_inputs"]["rows"]), 4)
            self.assertEqual(args[24], 8)
            self.assertEqual(base.search_batched_multi_encoded.call_args.kwargs,
                dict(record_atomic_work=True, record_compact_atomic_work=True))
            with self.assertRaises(ValueError): read_observer(evidence, compact=False)

    def test_v1_defaults_keep_native_kwargs_manifest_and_readback(self):
        for timed in (False, True):
            evidence, _, base, _, _ = observer_fixture(timed=timed, compact=False)
            self.assertEqual(base.search_batched_multi_encoded.call_args.kwargs, dict(record_atomic_work=True))
            manifest = evidence["consumed_world_inputs"]
            self.assertNotIn("native_atomic_encoding", manifest)
            self.assertNotIn("compact_atomic_observation", manifest["calls"][0])
            read_observer(evidence, timed=timed, compact=False)
            with self.assertRaises(ValueError): read_observer(evidence, timed=timed, compact=True)

    def test_format_conflicts_and_unsupported_sessions_refuse_before_dispatch(self):
        for kwargs in (dict(record_compact_atomic_work=False), dict(record_compact_atomic_work=0),
                dict(record_compact_atomic_work=None), dict(record_atomic_work=False)):
            recorder = NativeWorldInputs(BINDING, 16, record_atomic_work=True, record_compact_atomic_work=True)
            construct(recorder); base = SimpleNamespace(search_batched_multi_encoded=Mock())
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                NativeInputHandle(base, recorder).search_batched_multi_encoded(*arguments(), **kwargs)
            base.search_batched_multi_encoded.assert_not_called()
            self.assertEqual(recorder.calls, []); self.assertFalse(recorder.rows[0]["consumed"])
        recorder = NativeWorldInputs(BINDING, 16, record_atomic_work=True)
        construct(recorder); base = SimpleNamespace(search_batched_multi_encoded=Mock())
        with self.assertRaises(ValueError):
            NativeInputHandle(base, recorder).search_batched_multi_encoded(*arguments(), record_compact_atomic_work=True)
        base.search_batched_multi_encoded.assert_not_called()
        compact = NativeWorldInputs(BINDING, 16, record_atomic_work=True, record_compact_atomic_work=True)
        construct(compact)
        with self.assertRaises(ValueError):
            NativeInputHandle(base, compact, value_session=SimpleNamespace()).search_batched_multi_encoded(*arguments())
        self.assertEqual(compact.calls, []); base.search_batched_multi_encoded.assert_not_called()

    def test_unreviewed_schema_return_is_not_autodetected_or_finished(self):
        for expected in (False, True):
            source = native_report()
            report = source if expected else compact_fixture(source)
            recorder = NativeWorldInputs(BINDING, 16, deadline_limited=False, record_atomic_work=True,
                record_compact_atomic_work=expected)
            construct(recorder)
            base = SimpleNamespace(search_batched_multi_encoded=Mock(return_value=json.dumps(report)))
            NativeInputHandle(base, recorder).search_batched_multi_encoded(*arguments())
            self.assertTrue(recorder.errors)
            self.assertNotIn("atomic_work_validation", recorder.calls[0])
            with self.assertRaises(ValueError): native_manifest(recorder)

    def test_separately_injected_keyword_or_positional_ancestry_bank_refuses_pre_dispatch(self):
        for mode in ("keyword", "positional"):
            recorder = NativeWorldInputs(BINDING, 16, deadline_limited=False, record_atomic_work=True,
                record_compact_atomic_work=True)
            construct(recorder); base = SimpleNamespace(search_batched_multi_encoded=Mock())
            args = arguments(); kwargs = {}
            bank = object()
            if mode == "keyword": kwargs["visited_value_bank"] = bank
            else:
                args += [None]*(31-len(args))
                args.append(bank)  # Existing native ABI visited_value_bank slot.
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "excludes visited banks"):
                NativeInputHandle(base, recorder).search_batched_multi_encoded(*args, **kwargs)
            base.search_batched_multi_encoded.assert_not_called()
            self.assertEqual(recorder.calls, []); self.assertFalse(recorder.rows[0]["consumed"])

    def test_call_scope_identity_changes_cannot_swap_readers_after_dispatch(self):
        for mode in ("encoding", "timed", "atomic", "recorder", "base"):
            recorder = NativeWorldInputs(BINDING, 16, deadline_limited=False, record_atomic_work=True,
                record_compact_atomic_work=True)
            construct(recorder); base = SimpleNamespace(search_batched_multi_encoded=Mock())
            handle = NativeInputHandle(base, recorder)
            def changed(*args, **kwargs):
                if mode == "encoding": recorder.record_compact_atomic_work = False
                if mode == "timed": recorder.record_timed_atomic_work = True
                if mode == "atomic": recorder.record_atomic_work = False
                if mode == "recorder": handle.recorder = deepcopy(recorder)
                if mode == "base": handle.base = SimpleNamespace()
                return json.dumps(compact_fixture(native_report()))
            base.search_batched_multi_encoded.side_effect = changed
            with self.subTest(mode=mode), self.assertRaises(ValueError): handle.search_batched_multi_encoded(*arguments())
            self.assertEqual(recorder.calls[0]["status"], "REFUSED")
            self.assertNotIn("atomic_work_validation", recorder.calls[0])

    def test_constructor_format_cannot_be_changed_before_dispatch_or_publication(self):
        for original in (False, True):
            recorder = NativeWorldInputs(BINDING, 16, deadline_limited=False, record_atomic_work=True,
                record_compact_atomic_work=original)
            construct(recorder); base = SimpleNamespace(search_batched_multi_encoded=Mock())
            recorder.record_compact_atomic_work = not original
            with self.assertRaisesRegex(ValueError, "encoding scope changed"):
                NativeInputHandle(base, recorder).search_batched_multi_encoded(*arguments())
            base.search_batched_multi_encoded.assert_not_called(); self.assertEqual(recorder.calls, [])
            with self.assertRaises(ValueError): native_manifest(recorder)

    def test_fresh_readback_rejects_marker_changes_and_honestly_rehashed_backups(self):
        for timed in (False, True):
            evidence, _, _, _, _ = observer_fixture(timed=timed)
            for mode in ("manifest", "call", "alias", "raw", "arithmetic"):
                changed = deepcopy(evidence); manifest = changed["consumed_world_inputs"]; call = manifest["calls"][0]
                if mode == "manifest": manifest["native_atomic_encoding"] = "native-atomic-backups-v1"
                if mode == "call": call.pop("compact_atomic_observation")
                if mode == "alias": call["compact_atomic_observation"] = 1
                if mode == "raw": call["native_report_utf8"] += " "
                if mode == "arithmetic":
                    report = call["native_report"]
                    report["atomic_work"]["rounds"][0]["commits"][0]["root_value_bits"] = _bits(.25)
                    call["native_report_utf8"] = json.dumps(report)
                    call["report_utf8_sha256"] = hashlib.sha256(call["native_report_utf8"].encode()).hexdigest()
                    call["report_sha256"] = digest(report)
                with self.subTest(timed=timed, mode=mode), self.assertRaises(ValueError):
                    read_observer(changed, timed=timed)
            read_observer(evidence, timed=timed)

    def test_scope_requires_real_booleans_and_explicit_atomic_observation(self):
        for value in (1, 0, None, "compact"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                NativeWorldInputs(BINDING, 16, record_atomic_work=True, record_compact_atomic_work=value)
        with self.assertRaises(ValueError): NativeWorldInputs(BINDING, 16, record_compact_atomic_work=True)
        evidence, _, _, _, _ = observer_fixture()
        for value in (1, 0, None):
            with self.assertRaises(ValueError): read_observer(evidence, compact=value)
        with self.assertRaises(ValueError):
            validate_fixed_work_inputs(evidence, arm="reference", binding=BINDING,
                statistics_root_id="authored-only-root", compact_native_atomic_work=True)


class CompactNativeRuntimeTests(unittest.TestCase):
    def test_distinct_actor_recipe_default_stability_and_adapter_encoding_digest(self):
        from dataclasses import asdict
        from pokezero.mcts_eval.search_over_raw import SearchConfiguration
        from pokezero.mcts_eval.search_over_raw_actor_timed_run import (
            actor_timed_runtime_recipe, RUNTIME_KEY, annotation_runtime_scope)
        from pokezero.mcts_eval.search_over_raw_adapters import PublicModelSearchAdapter
        from pokezero.mcts_eval.search_over_raw_world_inputs import WorldInputSearchAdapter, TIMED_NATIVE_PUBLIC_CAPTURE
        from test_search_over_raw_timed_run import runtime
        checkpoint = SimpleNamespace(checkpoint_sha256="b"*64, showdown_source_sha256=BINDING["set_source_hash"])
        def parent_init(adapter, configuration, **kwargs):
            adapter.configuration = configuration
            adapter.runtime_configuration = runtime(configuration, checkpoint.checkpoint_sha256)
            adapter.runtime_configuration.update(reference_factory=None, initial_dispatch_workers=None)
        for leaf in ("model", "hp_fraction"):
            cfg = SearchConfiguration("incumbent", leaf=leaf, seconds=1.)
            default = actor_timed_runtime_recipe(cfg, checkpoint)
            expected = runtime(cfg, checkpoint.checkpoint_sha256)
            expected.update(reference_factory=None, initial_dispatch_workers=None,
                full_public_request_observation=deepcopy(TIMED_NATIVE_PUBLIC_CAPTURE))
            expected[RUNTIME_KEY] = annotation_runtime_scope()
            self.assertEqual(default, expected)
            compact = actor_timed_runtime_recipe(cfg, checkpoint, compact_native_atomic_work=True)
            self.assertNotEqual(digest(default), digest(compact))
            self.assertIs(compact["compact_native_atomic_work"], True)
            self.assertEqual(compact["native_atomic_work"]["schema"], SCHEMA)
            with patch.object(PublicModelSearchAdapter, "__init__", parent_init):
                adapter = WorldInputSearchAdapter(cfg, timed_native_atomic_work=True,
                    compact_native_atomic_work=True, expected_public_request_sha256="a"*64)
            observed = deepcopy(adapter.runtime_configuration); observed[RUNTIME_KEY] = annotation_runtime_scope()
            self.assertEqual(observed, compact)
            self.assertEqual(adapter.runtime_sha256, digest(adapter.runtime_configuration))

    def test_non_timed_oracle_reference_fixed_label_and_wrong_type_flags_refuse_pre_runtime(self):
        from pokezero.mcts_eval.search_over_raw import SearchConfiguration
        from pokezero.mcts_eval.search_over_raw_adapters import PublicModelSearchAdapter
        from pokezero.mcts_eval.search_over_raw_world_inputs import WorldInputSearchAdapter, WorldInputFixedWorkAdapter
        cases = [(WorldInputSearchAdapter, SearchConfiguration("incumbent"), {}),
            (WorldInputSearchAdapter, SearchConfiguration("reference", workers=20), dict(timed_reference_atomic_work=True)),
            (WorldInputFixedWorkAdapter, SearchConfiguration("incumbent"), dict(timed_native_atomic_work=True)),
            (WorldInputSearchAdapter, SearchConfiguration("incumbent", belief="oracle"), dict(timed_native_atomic_work=True)),
            (WorldInputSearchAdapter, SearchConfiguration("incumbent", leaf="raw_rollout"), dict(timed_native_atomic_work=True)),
            (WorldInputSearchAdapter, SearchConfiguration("incumbent"), dict(timed_native_atomic_work=True, value_diagnostics=object()))]
        with patch.object(PublicModelSearchAdapter, "__init__") as create:
            for adapter, cfg, options in cases:
                with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                    adapter(cfg, compact_native_atomic_work=True, **options)
            for bad in (1, 0, None):
                with self.assertRaises(ValueError): WorldInputSearchAdapter(SearchConfiguration("incumbent"),
                    timed_native_atomic_work=True, compact_native_atomic_work=bad)
            create.assert_not_called()

    def test_owning_selector_reader_requires_matching_prospective_recipe_for_both_formats(self):
        from pokezero.mcts_eval.search_over_raw import SearchConfiguration
        from test_search_over_raw_timed_run import runtime, selected_fixture, read_selection
        for leaf in ("model", "hp_fraction"):
            for zero in (False, True):
                cfg = SearchConfiguration("incumbent", leaf=leaf, seconds=1.)
                compact_runtime = runtime(cfg, compact=True)
                compact_selected = selected_fixture(cfg, compact_runtime, compact=True, zero=zero)
                result = read_selection(compact_selected, cfg, compact_runtime)
                self.assertEqual(result["accepted_units"], 0 if zero else 16)
                self.assertFalse(result["runtime_qualified"])
                self.assertFalse(result["scientific_collection_authorized"])
                for schema_mismatch in (False, True):
                    changed = deepcopy(compact_runtime)
                    if schema_mismatch: changed["native_atomic_work"]["schema"] = "native-atomic-backups-v1"
                    else: changed.pop("compact_native_atomic_work")
                    with self.assertRaises(ValueError): read_selection(compact_selected, cfg, changed)
                for expected, actual in ((False, True), (True, False)):
                    frozen = runtime(cfg, compact=expected)
                    substituted = selected_fixture(cfg, frozen, compact=actual, zero=zero)
                    with self.assertRaises(ValueError): read_selection(substituted, cfg, frozen)

    def test_oracle_runtime_law_does_not_silently_adopt_public_compact_recipe(self):
        from pokezero.mcts_eval.search_over_raw import SearchConfiguration
        from pokezero.mcts_eval.search_over_raw_timed_run import _runtime_law
        from test_search_over_raw_timed_run import runtime
        cfg = SearchConfiguration("incumbent", seconds=1.)
        compact = runtime(cfg, compact=True)
        with self.assertRaisesRegex(ValueError, "compact recipe scope drift"):
            _runtime_law(cfg, compact, digest(compact), configuration_check=lambda c:None)

    def test_original_native_and_actor_job_factories_forward_only_frozen_compact_marker(self):
        from pokezero.mcts_eval.search_over_raw_timed_run import NativeTimedAdapterJob
        from pokezero.mcts_eval.search_over_raw_actor_timed_run import ActorAnnotatedTimedJob
        from test_search_over_raw_timed_run import NativeTimedJobWiringTests
        harness = NativeTimedJobWiringTests()
        for compact in (False, True):
            job = harness.job()
            if compact:
                from pokezero.mcts_eval.search_over_raw_world_inputs import native_atomic_runtime_scopes
                job.expected_runtime_configuration["compact_native_atomic_work"] = True
                job.expected_runtime_configuration["native_atomic_work"], job.expected_runtime_configuration[
                    "timed_native_atomic_work"] = native_atomic_runtime_scopes(compact=True)
            # Factory-only verification does not bypass or claim constructor or
            # public-request/annotation admission. The native job is genuine;
            # its copied actor shell supplies only factory-read fields.
            with patch("pokezero.mcts_eval.search_over_raw_world_inputs.WorldInputSearchAdapter") as create:
                NativeTimedAdapterJob._make_adapter(job)
                self.assertEqual(create.call_args.kwargs.get("compact_native_atomic_work", False), compact)
            actor = object.__new__(ActorAnnotatedTimedJob)
            for key in ("configuration", "expected_runtime_configuration", "checkpoint_contract", "showdown_root",
                    "expected_public_request_sha256"):
                object.__setattr__(actor, key, getattr(job, key))
            object.__setattr__(actor, "annotation_boundary", object())
            with patch("pokezero.mcts_eval.search_over_raw_actor_timed_run.ActorAnnotatedTimedAdapter") as create:
                ActorAnnotatedTimedJob._make_adapter(actor)
                self.assertEqual(create.call_args.kwargs.get("compact_native_atomic_work", False), compact)


if __name__ == "__main__": unittest.main()
