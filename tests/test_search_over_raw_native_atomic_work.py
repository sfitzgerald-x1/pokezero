"""Authored float32/deadline receipts; no native builds, models, owners, or games."""
from copy import deepcopy
import unittest

from pokezero.mcts_eval.search_over_raw_native_atomic_work import (
    SCHEMA, _bits, state_hash, validate_native_report, validate_timed_native_report,
)


def native_report(iterations=16, text="complete-state", batch_size=16, chance_seed=7):
    """Exact dyadic two-outcome toy; constructor independent of the reader."""
    def stats(visits, total): return dict(visits=visits,total_bits=_bits(total))
    def snapshot(visits, one, two, branches):
        return dict(decisions=[dict(visits=visits, depth=0,
            side_one=[stats(visits,one)],side_two=[stats(visits,two)],
            children=[[0,0,0]] if branches else [])],chances=[branches] if branches else [])
    def branches(samples, pending=False):
        return [dict(probability_bits=_bits(p), value_sum_bits=_bits(0 if pending else total),
            visits=visits,pending_row=row if pending else None,child=None,terminal_bits=None,no_expand=True)
            for row,p,total,visits in ((0,.25,.25*samples,samples),(1,.75,.75,1))]
    previous = snapshot(0,0,0,[]); rounds = []
    for start in range(0,iterations,batch_size):
        size = min(batch_size,iterations-start)
        collected = snapshot(start+size,start*.625,start*.625+size,
            branches(max(1,start),pending=start==0))
        state = deepcopy(collected); commits = []
        for offset in range(size):
            expanded = start+offset == 0
            step = dict(decision=0,side_one=0,side_two=0,chance=0,branch=None if expanded else 0)
            end = (dict(kind="expanded") if expanded else dict(kind="row",row=0) if start==0
                else dict(kind="ready",value_bits=_bits(.25)))
            after = snapshot(start+size,(start+offset+1)*.625,
                start*.625+size-(offset+1)*.375,branches(max(1,start+offset+1)))
            commits.append(dict(path=[step],end=end,before=state,after=after,root_value_bits=_bits(.625)))
            state = deepcopy(after)
        rounds.append(dict(ordinal=len(rounds),status="COMMITTED",before_collection=previous,
            after_collection=collected,row_value_bits=[_bits(.25),_bits(.75)] if start==0 else [],
            commits=commits,after=state,accepted=size))
        previous = deepcopy(state)
    joint = dict(schema="completed-joint-actions-v1",scope="single_native_tree",
        basis="finalized_backups_not_reservations",root_completed_traversals=iterations,
        root_pairs=[dict(side_one_option=0,side_two_option=0,completed_visits=iterations)] if iterations else [],
        root_distinct_pairs=1 if iterations else 0,root_side_one_options=1,root_side_two_options=1,
        root_side_one_visits=[iterations],root_side_two_visits=[iterations],
        root_side_one_moves=["move"],root_side_two_moves=["move"],
        root_side_one_report_order=[0],root_side_two_report_order=[0],
        tree_completed_selections=iterations,tree_distinct_node_pairs=1 if iterations else 0)
    work = dict(schema=SCHEMA,scope="observed_float32_backup_consistency",
        independent_terminal_labels=False,runtime_qualified=False,
        input_state_utf8_sha256=state_hash(text),requested_iterations=iterations,
        batch_size=batch_size,chance_seed=chance_seed,rounds=rounds,final=previous,joint_action_witness=joint)
    return dict(iterations=iterations,joint_action_witness=deepcopy(joint),atomic_work=work,
        root_value=.625 if iterations else .5,
        side_one=[dict(move="move",visits=iterations,q=.625 if iterations else .5)],
        side_two=[dict(move="move",visits=iterations,q=.625 if iterations else .5)])


def validate(report, **overrides):
    return validate_native_report(report, **dict(dict(state_sha256=state_hash("complete-state"),
        requested_iterations=16,batch_size=16,chance_seed=7),**overrides))


class NativeAtomicTests(unittest.TestCase):
    def test_exact_chance_expectation_and_no_pseudocount_as_extra_work(self):
        report = native_report()
        self.assertEqual(validate(report)["completed_iterations"],16)
        self.assertEqual(report["atomic_work"]["final"]["chances"][0][0]["visits"],16)
        self.assertEqual(report["atomic_work"]["rounds"][0]["commits"][0]["root_value_bits"],_bits(.625))
        self.assertFalse(validate(report)["independent_terminal_labels"])
        self.assertEqual(report,native_report())

    def test_full_multi_round_allocation(self):
        result = validate(native_report(64),requested_iterations=64)
        self.assertEqual(result["committed_rounds"],4)
        self.assertEqual(result["completed_iterations"],64)

    def test_actual_parameter_substitution_rejected(self):
        for key,value in dict(state_sha256="a"*64,requested_iterations=32,batch_size=4,chance_seed=8).items():
            with self.subTest(key=key),self.assertRaises(ValueError): validate(native_report(),**{key:value})

    def test_partial_or_duplicate_commit_not_accepted(self):
        for mode in ("missing","duplicate","reverse","count","tail"):
            report=native_report(); work=report["atomic_work"]; r=work["rounds"][0]
            if mode=="missing": r["commits"].pop()
            if mode=="duplicate": r["commits"].append(deepcopy(r["commits"][0]))
            if mode=="reverse": r["commits"].reverse()
            if mode=="count": r["accepted"]=True
            if mode=="tail": work["final"]["decisions"][0]["side_one"][0]["total_bits"]=_bits(9)
            with self.subTest(mode=mode),self.assertRaises(ValueError):validate(report)

    def test_backup_value_path_virtual_loss_and_unrelated_mutations_rejected(self):
        for mode in ("value","one","two","path","probability","visits","pending","nan"):
            report=native_report(); c=report["atomic_work"]["rounds"][0]["commits"][0]
            if mode=="value": c["root_value_bits"]=_bits(.25)
            if mode=="one": c["after"]["decisions"][0]["side_one"][0]["total_bits"]=_bits(.25)
            if mode=="two": c["after"]["decisions"][0]["side_two"][0]["total_bits"]=_bits(16)
            if mode=="path": c["path"][0]["chance"]=1
            if mode=="probability": c["after"]["chances"][0][0]["probability_bits"]=_bits(.5)
            if mode=="visits": c["after"]["chances"][0][0]["visits"]=2
            if mode=="pending": c["after"]["chances"][0][0]["pending_row"]=0
            if mode=="nan": c["root_value_bits"]=_bits(float("nan"))
            with self.subTest(mode=mode),self.assertRaises(ValueError):validate(report)

    def test_collection_reservations_cannot_be_backups(self):
        report=native_report()
        report["atomic_work"]["rounds"][0]["after_collection"]["decisions"][0]["side_one"][0]["total_bits"]=_bits(10)
        with self.assertRaises(ValueError):validate(report)

    def test_unvisited_new_edges_and_unresolved_leaves_are_rejected(self):
        report=native_report(); work=report["atomic_work"]
        r=work["rounds"][0]
        for seat in ("side_one","side_two"):
            r["before_collection"]["decisions"][0][seat].append(dict(visits=0,total_bits=0))
        snapshots=[r["after_collection"],r["after"],work["final"]]
        snapshots += [s for c in r["commits"] for s in (c["before"],c["after"])]
        for snapshot in snapshots:
            if len(snapshot["chances"]) > 1: continue
            node=snapshot["decisions"][0]
            for seat in ("side_one","side_two"):node[seat].append(dict(visits=0,total_bits=0))
            node["children"].append([1,1,1])
            snapshot["chances"].append([dict(probability_bits=_bits(1),value_sum_bits=0,
                visits=1,pending_row=999,child=None,terminal_bits=None,no_expand=True)])
        with self.assertRaisesRegex(ValueError,"new native edges"):validate(report)

    def test_unused_prices_and_selected_action_drift_rejected(self):
        for mode in ("price","q","move","visits","boolpair"):
            report=native_report()
            if mode=="price":report["atomic_work"]["rounds"][0]["row_value_bits"].append(_bits(.5))
            if mode=="q":report["side_two"][0]["q"] = .25
            if mode=="move":report["side_one"][0]["move"] = "different"
            if mode=="visits":report["side_one"][0]["visits"] = 15
            if mode=="boolpair":
                for joint in (report["joint_action_witness"],report["atomic_work"]["joint_action_witness"]):
                    joint["root_pairs"][0]["side_one_option"] = False
            with self.subTest(mode=mode),self.assertRaises(ValueError):validate(report)

    def test_round_chain_scope_and_joint_witness_rejected(self):
        for mode in ("chain","joint","scope","labels","noatomic"):
            report=native_report(64); w=report["atomic_work"]
            if mode=="chain":w["rounds"][1]["before_collection"]["decisions"][0]["visits"]+=1
            if mode=="joint":w["joint_action_witness"]["tree_completed_selections"]+=1
            if mode=="scope":w["runtime_qualified"]=True
            if mode=="labels":w["independent_terminal_labels"]=True
            if mode=="noatomic":del report["atomic_work"]
            with self.subTest(mode=mode),self.assertRaises(ValueError):validate(report,requested_iterations=64)

    def test_zero_work_cancellation_and_unrestored_cancellation(self):
        report=native_report(0); w=report["atomic_work"]
        w["requested_iterations"]=16
        w["rounds"]=[dict(ordinal=0,status="CANCELLED",before_collection=deepcopy(w["final"]),
            after_collection=deepcopy(w["final"]),after=deepcopy(w["final"]),accepted=0,attempted_paths=[])]
        report["time_budget_exhausted"]=True
        self.assertEqual(validate(report)["completed_iterations"],0)
        w["rounds"][0]["accepted"]=1
        with self.assertRaises(ValueError):validate(report)


def timed_report(completed=16, requested=64, budget=8, elapsed=8.5, exhausted=True, cancelled=False):
    report = native_report(completed)
    report["atomic_work"]["requested_iterations"] = requested
    report.update(requested_iterations=requested, remaining_iterations=requested-completed,
        time_budget_enabled=True, time_budget_ms=budget, time_budget_elapsed_ms=elapsed,
        time_budget_batch_overshoot_ms=max(0.,elapsed-budget), time_budget_exhausted=exhausted)
    if cancelled:
        work = report["atomic_work"]
        before = deepcopy(work["final"])
        work["rounds"].append(dict(ordinal=len(work["rounds"]), status="CANCELLED",
            before_collection=deepcopy(before), after_collection=deepcopy(before),
            after=deepcopy(before), accepted=0, attempted_paths=[]))
    return report


def terminal_report(value=1.0):
    report = native_report()
    round = report["atomic_work"]["rounds"][0]
    round["row_value_bits"] = [_bits(.25)]
    root_value = .25 * .25 + .75 * value

    def stamp(snapshot, completed):
        root = snapshot["decisions"][0]
        root["side_one"][0]["total_bits"] = _bits(completed * root_value)
        root["side_two"][0]["total_bits"] = _bits(16 - completed * (1 - root_value))
        branch = snapshot["chances"][0][1]
        branch.update(terminal_bits=_bits(value), value_sum_bits=_bits(value), pending_row=None)

    stamp(round["after_collection"], 0)
    for index, commit in enumerate(round["commits"]):
        stamp(commit["before"], index)
        stamp(commit["after"], index + 1)
        commit["root_value_bits"] = _bits(root_value)
    stamp(round["after"], 16)
    stamp(report["atomic_work"]["final"], 16)
    report["root_value"] = root_value
    report["side_one"][0]["q"] = root_value
    report["side_two"][0]["q"] = root_value
    return report


def reserved_cancelled_report(*, requested=64, batch_size=16):
    report = timed_report(0, requested=requested)
    work = report["atomic_work"]
    work["batch_size"] = batch_size
    source = native_report()["atomic_work"]["rounds"][0]
    before = deepcopy(work["final"])
    work["rounds"] = [dict(ordinal=0, status="CANCELLED", before_collection=deepcopy(before),
        after_collection=deepcopy(source["after_collection"]), after=deepcopy(before), accepted=0,
        attempted_paths=[deepcopy(commit["path"]) for commit in source["commits"]])]
    return report


class TerminalAndCancellationContractsTests(unittest.TestCase):
    def test_declared_terminal_price_has_consistent_initial_pseudocount(self):
        for value, expected in ((0.0, .0625), (1.0, .8125)):
            report = terminal_report(value)
            original = deepcopy(report)
            self.assertEqual(validate(report)["completed_iterations"], 16)
            self.assertEqual(report["root_value"], expected)
            self.assertEqual(report, original)
            self.assertFalse(validate(report)["independent_terminal_labels"])

    def test_inconsistent_declared_terminal_pseudocount_refuses(self):
        report = native_report()
        round = report["atomic_work"]["rounds"][0]
        round["row_value_bits"] = [_bits(.25)]
        snapshots = [round["after_collection"], round["after"], report["atomic_work"]["final"]]
        snapshots += [snapshot for commit in round["commits"] for snapshot in (commit["before"], commit["after"])]
        for snapshot in snapshots:
            snapshot["chances"][0][1].update(terminal_bits=_bits(1.0), value_sum_bits=_bits(.75), pending_row=None)
        with self.assertRaisesRegex(ValueError, "terminal price"):
            validate(report)

    def test_nonempty_cancellation_restores_reservations_without_accepting_work(self):
        report = reserved_cancelled_report()
        original = deepcopy(report)
        result = validate_timed_native_report(report, state_sha256=state_hash("complete-state"),
            requested_iterations=64, batch_size=16, chance_seed=7, time_budget_ms=8)
        self.assertEqual((result["completed_iterations"], result["cancelled_rounds"]), (0, 1))
        self.assertFalse(result["owning_process_validated"])
        self.assertEqual(report, original)
        report["atomic_work"]["rounds"][0]["after"]["decisions"][0]["side_two"][0]["total_bits"] = _bits(1.)
        with self.assertRaisesRegex(ValueError, "not restored"):
            validate_timed_native_report(report, state_sha256=state_hash("complete-state"),
                requested_iterations=64, batch_size=16, chance_seed=7, time_budget_ms=8)

    def test_cancelled_reservations_cannot_exceed_batch_or_remaining_allocation(self):
        for requested, batch_size in ((64, 1), (8, 16)):
            report = reserved_cancelled_report(requested=requested, batch_size=batch_size)
            with self.subTest(requested=requested, batch_size=batch_size), self.assertRaisesRegex(ValueError, "bound allocation"):
                validate_timed_native_report(report, state_sha256=state_hash("complete-state"),
                    requested_iterations=requested, batch_size=batch_size, chance_seed=7, time_budget_ms=8)


class TimedNativeReportTests(unittest.TestCase):
    def read(self, report, **overrides):
        return validate_timed_native_report(report, **dict(dict(state_sha256=state_hash("complete-state"),
            requested_iterations=64,batch_size=16,chance_seed=7,time_budget_ms=8),**overrides))

    def test_prefix_and_cancelled_tail_not_fixed_allocation(self):
        result = self.read(timed_report(cancelled=True))
        self.assertEqual((result["completed_iterations"], result["remaining_iterations"],
            result["committed_rounds"],result["cancelled_rounds"]), (16,48,1,1))
        self.assertEqual(result["native_overshoot_ms"], .5)
        self.assertFalse(result["runtime_qualified"])
        self.assertFalse(result["owning_process_validated"])

    def test_zero_deadline_without_batches_and_cancelled_tail(self):
        for cancelled in (False,True):
            result = self.read(timed_report(0,cancelled=cancelled))
            self.assertEqual(result["completed_iterations"],0)
            self.assertEqual(result["status"],"ZERO_COMMIT_DEADLINE_RETURN")

    def test_full_work_final_batch_can_overrun_without_exhaustion_flag(self):
        result = self.read(timed_report(64,exhausted=False))
        self.assertEqual(result["completed_iterations"],64)
        self.assertEqual(result["native_overshoot_ms"],.5)
        self.assertFalse(result["on_time_deployability_established"])

    def test_no_partial_commit_or_false_deadline_completion(self):
        cases = [timed_report(3), timed_report(0,exhausted=False), timed_report(elapsed=7.9)]
        for report in cases:
            with self.subTest(report=report["iterations"]),self.assertRaises(ValueError): self.read(report)

    def test_foreign_args_and_malformed_duration_count_or_scope_refuse(self):
        for key, value in dict(time_budget_ms=True, time_budget_enabled=False,
            remaining_iterations=True, requested_iterations=16, time_budget_elapsed_ms=float("nan"),
            time_budget_batch_overshoot_ms=0., time_budget_exhausted=1).items():
            report=timed_report(); report[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): self.read(report)
        with self.assertRaises(ValueError): self.read(timed_report(),time_budget_ms=9)
