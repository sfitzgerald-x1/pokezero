"""Real reference-kernel commits on toy worlds only; no Pokémon sources."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
import unittest

from pokezero.mcts_eval.paper_reference import (
    Evaluation, ReferenceRefusal, SamplingDeadlineExceeded, TrajectorySearch, ReferenceConfig,
)
from pokezero.mcts_eval.search_over_raw_atomic_work import (
    ReferenceAtomicWork, digest, update_record, validate_reference_batch, validate_reference_atomic_work,
)
from test_paper_reference import ROOT, ToyWorld

BINDING = dict(root_binding="a"*64, subject="p1", set_source_hash="b"*64, information_key=ROOT.key.hex())


def input_witness():
    return dict(input_sha256="c"*64, consumed=True, snapshot_fields_excluded=[],
        ownership_metadata_excluded=["battle_token","battle_id"], owned_hypothetical_only=True,
        runtime_qualified=False)


def fixture(*, depth=3, trajectories=4, search=None, observer=None, clock=None, deadline=None,
            factory=None):
    search = search or TrajectorySearch(ReferenceConfig(.5, 1))
    if search._battle_id is None:
        search.reset_battle("battle-1")
    observer = observer or ReferenceAtomicWork(BINDING)
    draws = []
    def sample(rng):
        row = dict(ordinal=search._ordinal-1, status="ROOT_VALIDATED", released=False,
            consumed_world_input=input_witness())
        draws.append(row)
        try:
            world = factory(rng) if factory else ToyWorld(depth=depth, value=1.)
        except SamplingDeadlineExceeded:
            row.update(status="DEADLINE_CANCELLED", consumed_world_input={})
            raise
        close = world.close
        def cleanup():
            close()
            row["released"] = True
        world.close = cleanup
        return world
    batch = search.search_batch(ROOT, battle_id="battle-1", evaluate_root=lambda s: Evaluation((.5,.5),0),
        sample_world=sample, seed=123, trajectories=trajectories, atomic_work=observer,
        **({"clock":clock} if clock else {}), deadline_at=deadline)
    evidence = dict(draws=draws, atomic_work=observer.evidence(draws))
    sequence = 1 if search._exported_statistics is None else search._exported_statistics.sequence+1
    update = search.export_statistics(worker_id="0", sequence=sequence)
    return search, observer, evidence, batch, update


class AtomicWorkTests(unittest.TestCase):
    def check(self, evidence, batch, update):
        return validate_reference_batch(evidence, batch=batch, update=update, root=ROOT,
            battle_id="battle-1", worker_id="0", sequence=update.sequence if hasattr(update,"sequence") else update["sequence"])

    def test_actual_kernel_path_backups_and_cleanup_reconcile(self):
        search, observer, evidence, batch, update = fixture()
        self.assertEqual(self.check(evidence, batch, update), digest(evidence["atomic_work"]))
        self.assertEqual(self.check(evidence, asdict(batch), update_record(update)), digest(evidence["atomic_work"]))
        self.assertEqual(batch.trajectories, 4)
        self.assertEqual([len(r["path"]) for r in evidence["atomic_work"]["rows"]], [1,2,3,3])
        self.assertTrue(all(r["cleanup_completed"] for r in evidence["atomic_work"]["rows"]))
        self.assertIsNone(observer.current)
        self.assertFalse(evidence["atomic_work"]["independent_terminal_labels"])

    def test_instrumentation_preserves_actions_values_rng_and_counts(self):
        instrumented, _, _, batch, _ = fixture(trajectories=12)
        plain = TrajectorySearch(ReferenceConfig(.5,1)); plain.reset_battle("battle-1")
        control = plain.search_batch(ROOT, battle_id="battle-1", evaluate_root=lambda s: Evaluation((.5,.5),0),
            sample_world=lambda rng:ToyWorld(depth=3,value=1.), seed=123, trajectories=12)
        self.assertEqual((batch.trajectories,batch.transitions,batch.world_draws),
            (control.trajectories,control.transitions,control.world_draws))
        self.assertEqual(instrumented.nodes, plain.nodes)
        self.assertEqual(instrumented._local_statistics, plain._local_statistics)

    def test_each_atomic_dimension_or_missing_cleanup_refuses(self):
        _, _, original, batch, update = fixture()
        for field, value in (("signed_value",-.5), ("cleanup_completed",False),
                ("input_sha256","d"*64), ("status","REFUSED"), ("ordinal",True),
                ("transitions",99), ("sampled",False)):
            evidence=deepcopy(original); evidence["atomic_work"]["rows"][0][field]=value
            with self.subTest(field=field), self.assertRaises(ValueError): self.check(evidence,batch,update)
        evidence=deepcopy(original); evidence["atomic_work"]["rows"][0]["path"][0]["action_index"]=True
        with self.assertRaises(ValueError): self.check(evidence,batch,update)
        evidence=deepcopy(original); evidence["atomic_work"]["rows"][0]["path"]=[]
        with self.assertRaises(ValueError): self.check(evidence,batch,update)

    def test_missing_extra_duplicate_or_out_of_order_attempts_refuse(self):
        _, _, original, batch, update=fixture()
        for rows in (original["atomic_work"]["rows"][:-1],
                original["atomic_work"]["rows"]+[original["atomic_work"]["rows"][-1]],
                list(reversed(original["atomic_work"]["rows"]))):
            evidence=deepcopy(original); evidence["atomic_work"]["rows"]=deepcopy(rows)
            with self.assertRaises(ValueError): self.check(evidence,batch,update)

    def test_exported_own_statistics_and_boolean_alias_drift_refuse(self):
        _, _, evidence,batch,update=fixture()
        bad=update_record(update); bad["rows"][-1]["totals"][0]=-.5
        with self.assertRaises(ValueError): self.check(evidence,batch,bad)
        bad=deepcopy(evidence); bad["atomic_work"]["rows"][0]["after"][0]["visits"][0]=True
        with self.assertRaises(ValueError): self.check(bad,batch,update)

    def test_partial_actual_backup_is_not_certified(self):
        search=TrajectorySearch(ReferenceConfig(.5,1)); search.reset_battle("battle-1")
        search._own_backup=lambda node,index,value: None
        _, _, evidence,batch,update=fixture(search=search)
        with self.assertRaisesRegex(ValueError,"full commit"): self.check(evidence,batch,update)

    def test_mid_trajectory_deadline_retains_attempt_without_backup(self):
        now=[0.]
        def factory(rng):
            world=ToyWorld(value=1.)
            advance=world.advance
            def slow(*args):
                now[0]+=.6
                return advance(*args)
            world.advance=slow
            return world
        _, _, evidence,batch,update=fixture(factory=factory,clock=lambda:now[0],deadline=1.)
        self.check(evidence,batch,update)
        self.assertEqual([r["status"] for r in evidence["atomic_work"]["rows"]],["COMMITTED","CANCELLED"])
        self.assertEqual((batch.world_draws,batch.trajectories,batch.transitions),(2,1,2))
        cancel=evidence["atomic_work"]["rows"][-1]
        self.assertEqual(cancel["before"],cancel["after"])
        self.assertTrue(cancel["cleanup_completed"])

    def test_sampling_deadline_no_owned_world_or_backup(self):
        now=[0.]
        def fail(rng):
            now[0]=2.
            raise SamplingDeadlineExceeded()
        _, _, evidence,batch,update=fixture(factory=fail,clock=lambda:now[0],deadline=1.)
        self.check(evidence,batch,update)
        self.assertEqual(batch.trajectories,0)
        self.assertFalse(evidence["atomic_work"]["rows"][0]["sampled"])

    def test_cleanup_failure_poisoned_not_successful_even_after_commit(self):
        search=TrajectorySearch(ReferenceConfig(.5,1)); search.reset_battle("battle-1")
        observer=ReferenceAtomicWork(BINDING)
        def factory(rng):
            world=ToyWorld()
            def fail(): raise RuntimeError("cleanup failure")
            world.close=fail
            return world
        with self.assertRaisesRegex(RuntimeError,"cleanup failure"):
            fixture(search=search,observer=observer,factory=factory)
        self.assertEqual(observer.rows[-1]["status"],"REFUSED")
        self.assertFalse(observer.rows[-1]["cleanup_completed"])
        with self.assertRaises(ReferenceRefusal): search.export_statistics(worker_id="0",sequence=1)

    def test_incremental_evidence_preserves_rng_ordinals_not_replayed_rows(self):
        search,observer,first,batch,update=fixture(trajectories=2)
        self.check(first,batch,update)
        _,_,second,batch,update=fixture(search=search,observer=observer,trajectories=2)
        self.check(second,batch,update)
        self.assertEqual([r["ordinal"] for r in first["atomic_work"]["rows"]],[0,1])
        self.assertEqual([r["ordinal"] for r in second["atomic_work"]["rows"]],[2,3])

    def test_scope_root_draw_and_state_alias_refuse(self):
        _,_,original,batch,update=fixture()
        for mutate in (
            lambda e:e["atomic_work"].update(runtime_qualified=True),
            lambda e:e["atomic_work"]["binding"].update(information_key="00"),
            lambda e:e["draws"][0].update(released=False),
            lambda e:e["atomic_work"]["rows"][0]["root"].update(faint_count=True),
            lambda e:e["atomic_work"]["rows"][0]["after"][0]["state"]["actions"].append("move:forged"),
        ):
            evidence=deepcopy(original); mutate(evidence)
            with self.assertRaises(ValueError): self.check(evidence,batch,update)

    def test_empty_batch_cannot_smuggle_exported_counts(self):
        _,_,evidence,batch,update=fixture(clock=lambda:2.,deadline=1.)
        self.check(evidence,batch,update)
        self.assertEqual(evidence["atomic_work"]["rows"],[])
        bad=update_record(update); bad["rows"]="malformed"
        with self.assertRaises(ValueError): self.check(evidence,batch,bad)
        _,_,_,_,populated=fixture()
        with self.assertRaises(ValueError): self.check(evidence,batch,populated)

    def test_update_identity_bound_to_expected_worker_sequence_battle(self):
        _,_,evidence,batch,update=fixture()
        for key,value in (("battle_id","other"),("worker_id","other"),("sequence",999),("sequence",True)):
            bad=update_record(update); bad[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_reference_batch(evidence,batch=batch,update=bad,root=ROOT,
                    battle_id="battle-1",worker_id="0",sequence=1)

    def test_original_trajectory_error_survives_cleanup_and_observer_errors(self):
        for broken_observer in (False,True):
            observer=ReferenceAtomicWork(BINDING)
            if broken_observer:
                def observer_fail(search): raise KeyError("observer failure")
                observer.refuse=observer_fail
            def factory(rng):
                world=ToyWorld()
                def advance(*args): raise RuntimeError("primary trajectory failure")
                def close(): raise ValueError("cleanup failure")
                world.advance=advance; world.close=close
                return world
            with self.assertRaisesRegex(RuntimeError,"primary trajectory failure") as error:
                fixture(observer=observer,factory=factory)
            self.assertTrue(any("cleanup failed" in n for n in error.exception.__notes__))
            if broken_observer:
                self.assertTrue(any("observer failed" in n for n in error.exception.__notes__))

    def test_imported_master_statistics_are_never_new_own_work(self):
        from pokezero.mcts_eval.paper_reference import DecisionState
        from pokezero.mcts_eval.paper_reference_exchange import StatisticsMaster,Statistics,WorkerUpdate
        shared_leaf=DecisionState(b"leaf:1",("move:a",),0)
        master=StatisticsMaster("battle-1")
        snapshot=master.accept(WorkerUpdate("battle-1","other",1,(
            Statistics(ROOT,(10,2),(5.,1.),12), Statistics(shared_leaf,(6,),(3.,),6))))
        search=TrajectorySearch(ReferenceConfig(.5,1)); search.reset_battle("battle-1")
        search.merge_statistics(snapshot)
        _,_,evidence,batch,update=fixture(search=search,trajectories=1)
        self.check(evidence,batch,update)
        root_own=next(r for r in update.rows if r.state==ROOT)
        self.assertEqual(root_own.count,1)
        self.assertEqual(search.nodes[ROOT.key].count,13)
        leaf_own=next(r for r in update.rows if r.state==shared_leaf)
        self.assertEqual(leaf_own.count,1)
        self.assertEqual(batch.trajectories,1)


def selection_fixture():
    """Twenty independent actual toy-kernel runs, no processes or games."""
    receipts=[]
    for worker in range(20):
        _,_,evidence,batch,update=fixture()
        exported=update_record(update); exported["worker_id"]=str(worker)
        receipts.append(dict(worker=worker,sequence=1,evidence=evidence,batch=asdict(batch),own_statistics=exported))
    root_rows=[next(r for r in rec["own_statistics"]["rows"] if r["state"]["key"]==ROOT.key.hex())
        for rec in receipts]
    visits=[sum(r["visits"][i] for r in root_rows) for i in range(2)]
    totals=[sum(r["totals"][i] for r in root_rows) for i in range(2)]
    result=dict(trajectories=80,transitions=sum(r["batch"]["transitions"] for r in receipts),world_draws=80,
        root_visits=visits,root_means=[q/n if n else None for q,n in zip(totals,visits)],
        action=ROOT.actions[max(range(2),key=lambda i:(visits[i],-i))])
    return dict(result=result,worker_receipts=receipts)


class AtomicToyRuntime:
    def __init__(self,index): self.index=index

    def prepare(self,request):
        from pokezero.mcts_eval.paper_reference_parallel import PreparedDecision
        audit=ReferenceAtomicWork(BINDING)
        draws=[]; consumed=0
        def sample(rng):
            row=dict(ordinal=len(draws),status="ROOT_VALIDATED",released=False,
                consumed_world_input=input_witness())
            draws.append(row)
            world=ToyWorld(depth=3,value=1.)
            close=world.close
            def cleanup():
                close(); row["released"]=True
            world.close=cleanup
            return world
        def evidence():
            nonlocal consumed
            result=dict(draws=deepcopy(draws[consumed:]))
            consumed=len(draws)
            result["atomic_work"]=audit.evidence(result["draws"])
            return result
        return PreparedDecision(ROOT,lambda state:Evaluation((.5,.5),0.),sample,evidence,atomic_work=audit)

    def close(self): pass


class AtomicSelectionTests(unittest.TestCase):
    def test_all_twenty_workers_and_persisted_values_reconciled(self):
        evidence=selection_fixture()
        result=validate_reference_atomic_work(evidence,binding=BINDING,battle_id="battle-1")
        self.assertEqual(result["accepted_trajectories"],80)
        self.assertFalse(result["runtime_qualified"])

    def test_missing_worker_historical_counts_wrong_action_and_aggregate_refuse(self):
        original=selection_fixture()
        for mutation in (
            lambda e:e["worker_receipts"].pop(),
            lambda e:e["result"].update(trajectories=81),
            lambda e:e["result"].update(action="invented"),
            lambda e:e["result"].update(root_means=[0.,0.]),
            lambda e:e["worker_receipts"][0].update(sequence=2),
            lambda e:e["worker_receipts"][0]["own_statistics"].update(battle_id="other"),
        ):
            evidence=deepcopy(original); mutation(evidence)
            with self.assertRaises(ValueError):
                validate_reference_atomic_work(evidence,binding=BINDING,battle_id="battle-1")

    def test_actual_spawned_worker_and_master_reconcile_incremental_exchanges(self):
        from pokezero.mcts_eval.paper_reference_parallel import ParallelTrajectorySearch
        from test_paper_reference_parallel import Request
        with ParallelTrajectorySearch(ReferenceConfig(.5,1),AtomicToyRuntime,workers=2) as pool:
            selected=pool.search(Request(),ROOT,battle_id="battle-1",seed=123,trajectories_per_worker=12)
            self.assertEqual(selected.result.trajectories,24)
            self.assertEqual(len(selected.worker_receipts),4)
            for receipt in selected.worker_receipts:
                self.assertIn("own_statistics",receipt)
                validate_reference_batch(receipt["evidence"],batch=receipt["batch"],
                    update=receipt["own_statistics"],root=ROOT,battle_id="battle-1",
                    worker_id=str(receipt["worker"]),sequence=receipt["sequence"])
        self.assertTrue(all(not process.is_alive() for process in pool._processes))


if __name__ == "__main__": unittest.main()
