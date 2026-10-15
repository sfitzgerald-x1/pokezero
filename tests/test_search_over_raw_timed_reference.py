"""Toy kernel/owned processes and authored receipts; no Pokémon sources or games."""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pokezero.mcts_eval.paper_reference import ReferenceConfig, ReferenceRefusal, SamplingDeadlineExceeded, SearchResult, BatchResult
from pokezero.mcts_eval.paper_reference_parallel import ParallelTrajectorySearch, ParallelRefusal, ParallelResult
from pokezero.mcts_eval.search_over_raw import SearchConfiguration, digest
from pokezero.mcts_eval.search_over_raw_atomic_work import update_record, validate_reference_atomic_work
from pokezero.mcts_eval.search_over_raw_timed_reference import (
    SCHEMA, validate_reference_timed_inputs, read_reference_timed_selection,
)
from pokezero.mcts_eval.search_over_raw_world_inputs import reference_manifest, WorldInputSearchAdapter
from test_search_over_raw_atomic_work import fixture, BINDING, AtomicToyRuntime
from test_paper_reference_parallel import Request
from test_paper_reference import ROOT, ToyWorld


def evidence_fixture(*, zero=False, seconds=1., encoded=False, hp=False):
    receipts = []
    for worker in range(20):
        _, _, evidence, batch, update = fixture(**(dict(clock=lambda:2.1, deadline=2.) if zero else {}))
        exported = update_record(update); exported["worker_id"] = str(worker)
        receipts.append(dict(worker=worker, sequence=1, evidence=evidence, batch=asdict(batch),
            own_statistics=exported, preparation_seconds=.001, master_version=worker+1))
        if hp:
            evidence["leaf_ablation"] = dict(schema="pokezero.search-over-raw.leaf-batch.v1",
                leaf="hp_fraction", rollout_cap=250, root_priors="unchanged_champion",
                root_value_used_for_backup=False, evaluations=[])
    if encoded:
        def convert(value):
            if type(value) is dict:
                for key, item in value.items():
                    if key == "actions" and item == list(ROOT.actions):
                        value[key] = ["action:0", "action:1"]
                    else: convert(item)
            elif type(value) in (list, tuple):
                for item in value: convert(item)
        convert(receipts)
    root_rows = [next((r for r in rec["own_statistics"]["rows"]
        if r["state"]["key"] == ROOT.key.hex()), dict(visits=[0,0], totals=[0.,0.])) for rec in receipts]
    visits = [sum(r["visits"][i] for r in root_rows) for i in range(2)]
    totals = [sum(r["totals"][i] for r in root_rows) for i in range(2)]
    start, deadline, finish = 1., 1.+seconds, 1.+seconds+.1
    result = dict(action=None if zero else ("action:0" if encoded else ROOT.actions[0]),
        trajectories=0 if zero else 80, transitions=sum(r["batch"]["transitions"] for r in receipts),
        world_draws=sum(r["batch"]["world_draws"] for r in receipts), root_visits=visits,
        root_means=[q/n if n else None for q,n in zip(totals,visits)], elapsed_seconds=finish-start,
        deadline_exhausted=True, deadline_overrun_seconds=max(0.,finish-deadline))
    # Positive authored prefix ends at a worker-observed deadline, not a quota.
    stop_at = deadline+.05
    if not zero:
        for rec in receipts: rec["batch"]["deadline_exhausted"] = True
    if not zero:
        winner = max(range(2), key=lambda i:(visits[i],-i))
        result["action"] = "action:"+str(winner) if encoded else ROOT.actions[winner]
    evidence = dict(result=result, worker_receipts=receipts, worker_pids=list(range(100,120)),
        exchange_count=20, startup_seconds=.01, decision_id=1,
        initial_dispatch_schedule=[dict(worker=w, seconds_from_decision_start=w*.001,
            released_by_first_update=None if w<6 else w-6, master_version=0 if w<6 else w-5,
            deadline_expired=False) for w in range(20)],
        reference_timed_prefix=dict(schema=SCHEMA, battle_id="battle-1", selection_seed=7,
            decision_id=1, nominal_search_seconds=seconds, decision_started_monotonic=start,
            adapter_started_monotonic=start,
            absolute_deadline_monotonic=deadline, decision_finished_monotonic=finish,
            initial_dispatch_workers=6, acknowledged_done_workers=list(range(20)),
            stop_acknowledgements=[dict(worker=w,sequence=1,master_version=w+1,limit=0,
                decided_monotonic=stop_at,acknowledged_monotonic=stop_at,
                deadline_expired=True,worker_deadline_exhausted=True)
                for w in range(20)],
            status="UNQUALIFIED_AT_CEILING" if zero else "SELECTED", retry_authorized=False,
            runtime_qualified=False, on_time_deployability_established=False))
    evidence["consumed_world_inputs"] = reference_manifest(evidence, BINDING)
    for receipt,ack in zip(receipts,evidence["reference_timed_prefix"]["stop_acknowledgements"]):
        receipt["timed_acknowledgement"]=deepcopy(ack)
    return evidence


def runtime(cfg):
    value = dict(configuration=asdict(cfg), root_statistics="independent_per_root", incumbent=None,
        checkpoint_sha256="a"*64, set_source_hash=BINDING["set_source_hash"], initial_dispatch_workers=6,
        consumed_world_inputs=dict(schema="pokezero.search-over-raw.consumed-world-inputs.v1",
            initial_inputs_only=True, snapshot_fields_excluded=[], ownership_metadata_excluded=["battle_token","battle_id"],
            instrumentation_can_change_deadlines=True, qualifies_uninstrumented_runtime=False),
        reference_atomic_work=dict(schema="pokezero.search-over-raw.reference-atomic-work.v1",
            scope="actual_own_backup_and_cleanup; not independent simulator labels",
            instrumentation_can_change_deadlines=True, runtime_qualified=False),
        timed_reference_atomic_work=dict(schema=SCHEMA, zero_work_not_scored=True,
            fresh_single_attempt=True, on_time_deployability_established=False, runtime_qualified=False))
    if cfg.leaf == "hp_fraction":
        value["reference_leaf"] = dict(leaf="hp_fraction", rollout_cap=250, rollout_policy="raw_argmax_both_seats",
            tree="unchanged_trajectory_reference", priors="unchanged_champion")
    return value


def selection_fixture(cfg, zero=False):
    evidence = evidence_fixture(zero=zero, seconds=cfg.seconds, encoded=True, hp=cfg.leaf=="hp_fraction")
    evidence["reference_timed_prefix"]["battle_id"] = "search-over-raw-root:root"
    for receipt in evidence["worker_receipts"]:
        receipt["own_statistics"]["battle_id"] = "search-over-raw-root:root"
    readout = validate_reference_timed_inputs(evidence, binding=BINDING,
        battle_id="search-over-raw-root:root", selection_seed=7, nominal_seconds=cfg.seconds)
    evidence["timed_reference_readout"] = readout
    elapsed = evidence["result"]["elapsed_seconds"]+.01
    selected = dict(root_id="root:"+cfg.identity, statistics_root_id="root", selection_seed=7,
        configuration_sha256=cfg.identity, runtime_sha256=digest(runtime(cfg)),
        independent_root_statistics=True, scientific_strength_evidence=False, evidence=evidence,
        status=readout["status"], action=None if zero else int(evidence["result"]["action"][7:]),
        elapsed_seconds=elapsed, nominal_search_seconds=cfg.seconds, exceeded_nominal_seconds=elapsed>cfg.seconds)
    if zero:
        selected.update(attempt_consumed=True, retry_authorized=False, continuation_authorized=False, runtime_qualified=False)
    return selected


@dataclass
class AuthoredPublicContext:
    player_id: str
    observation: object
    requested_observations: dict
    public_materialization_state: object
    trajectory: object
    requested_players: tuple = ("p1",)


class TimedAtomicToyRuntime(AtomicToyRuntime):
    def prepare(self, request):
        if request.failure and self.index == 0:
            raise ReferenceRefusal("deliberate timed operational refusal")
        time.sleep(request.preparation_delay)
        return super().prepare(request)


class TimedReferenceTests(unittest.TestCase):
    def check(self, evidence, seconds=1.):
        return validate_reference_timed_inputs(evidence, binding=BINDING,
            battle_id="battle-1", selection_seed=7, nominal_seconds=seconds)

    def read(self, selected, cfg):
        return read_reference_timed_selection(selected, configuration=cfg, runtime_configuration=runtime(cfg),
            runtime_sha256=digest(runtime(cfg)), root_id="root:"+cfg.identity, statistics_root_id="root",
            selection_seed=7, consumed_binding=BINDING, legal_action_mask=(True,True))

    def test_positive_prefix_not_fixed_quota_and_normal_empty_zero(self):
        for zero in (False, True):
            evidence = evidence_fixture(zero=zero)
            readout = self.check(evidence)
            self.assertEqual(readout["accepted_units"], 0 if zero else 80)
            self.assertEqual(readout["status"], "UNQUALIFIED_AT_CEILING" if zero else "SELECTED")
            self.assertFalse(readout["process_ownership_validated"])
            self.assertFalse(readout["runtime_qualified"])
        with self.assertRaises(ValueError):
            validate_reference_atomic_work(evidence_fixture(zero=True), binding=BINDING,battle_id="battle-1")

    def test_every_ceiling_both_model_hp_lanes_and_zero_selected(self):
        for seconds in (1.,3.,10.):
            for leaf in ("model","hp_fraction"):
                cfg = SearchConfiguration("reference", leaf=leaf, seconds=seconds, workers=20)
                for zero in (True,False):
                    self.read(selection_fixture(cfg,zero),cfg)

    def test_all_worker_exports_and_done_are_required_even_without_draws(self):
        for mutate in (
            lambda e:e["worker_receipts"].pop(),
            lambda e:e["worker_receipts"][0].update(sequence=2),
            lambda e:e["worker_receipts"][0]["own_statistics"].update(worker_id="1"),
            lambda e:e["reference_timed_prefix"]["acknowledged_done_workers"].pop(),
            lambda e:e["reference_timed_prefix"]["acknowledged_done_workers"].__setitem__(1,True),
            lambda e:e["worker_pids"].__setitem__(0,True),
            lambda e:e["worker_pids"].__setitem__(0,101),
            lambda e:e["worker_receipts"][0]["batch"].update(deadline_exhausted=False),
            lambda e:e["worker_receipts"][0]["evidence"]["atomic_work"].update(binding=dict(BINDING,subject="p2")),
        ):
            evidence = evidence_fixture(zero=True); mutate(evidence)
            with self.assertRaises(ValueError):self.check(evidence)

    def test_zero_has_no_stale_action_or_inferred_missing_work(self):
        for mutate in (
            lambda e:e["result"].update(action=ROOT.actions[0]),
            lambda e:e["result"].pop("action"),
            lambda e:e["result"].update(root_visits=[1,0]),
            lambda e:e["result"].update(root_means=[0.,None]),
            lambda e:e["result"].update(trajectories=True),
            lambda e:e["reference_timed_prefix"].update(status="SELECTED"),
            lambda e:e["reference_timed_prefix"].update(retry_authorized=True),
        ):
            evidence=evidence_fixture(zero=True);mutate(evidence)
            with self.assertRaises(ValueError):self.check(evidence)

    def test_protocol_clock_original_deadline_and_fifo_mutations_refuse(self):
        for mutate in (
            lambda e:e["reference_timed_prefix"].update(selection_seed=True),
            lambda e:e["reference_timed_prefix"].update(absolute_deadline_monotonic=3.),
            lambda e:e["reference_timed_prefix"].update(decision_finished_monotonic=float("nan")),
            lambda e:e["result"].update(deadline_overrun_seconds=1.),
            lambda e:e["result"].update(elapsed_seconds=0.),
            lambda e:e["initial_dispatch_schedule"][7].update(worker=6),
            lambda e:e["initial_dispatch_schedule"][6].update(deadline_expired=True),
            lambda e:e["initial_dispatch_schedule"][6].update(released_by_first_update=1,master_version=2),
            lambda e:e["worker_receipts"][0].update(master_version=0),
            lambda e:e["worker_receipts"][0].update(preparation_seconds=-1),
            lambda e:e["reference_timed_prefix"]["stop_acknowledgements"].pop(),
            lambda e:e["reference_timed_prefix"]["stop_acknowledgements"][0].update(limit=1),
            lambda e:e["reference_timed_prefix"]["stop_acknowledgements"][0].update(sequence=2),
            lambda e:e["reference_timed_prefix"]["stop_acknowledgements"][0].update(acknowledged_monotonic=1.5),
            lambda e:e["reference_timed_prefix"]["stop_acknowledgements"][0].update(worker_deadline_exhausted=False),
        ):
            evidence=evidence_fixture(zero=True);mutate(evidence)
            with self.assertRaises(ValueError):self.check(evidence)

    def test_selected_external_inputs_cached_accounting_action_and_runtime_refuse(self):
        cfg=SearchConfiguration("reference",workers=20)
        for zero in (False,True):
            for mutate in (
                lambda s:s.update(selection_seed=True),
                lambda s:s.update(runtime_sha256="c"*64),
                lambda s:s.update(statistics_root_id="other"),
                lambda s:s.update(action=2),
                lambda s:s["evidence"]["timed_reference_readout"].update(accepted_units=999),
            ):
                selected=selection_fixture(cfg,zero);mutate(selected)
                with self.assertRaises(ValueError):self.read(selected,cfg)
        selected=selection_fixture(cfg,True);selected["retry_authorized"]=True
        with self.assertRaises(ValueError):self.read(selected,cfg)

    def test_bad_configuration_opt_in_rejected_before_runtime_startup(self):
        for cfg in (SearchConfiguration("incumbent"), SearchConfiguration("reference",workers=20,leaf="raw_rollout"),
            SearchConfiguration("reference",workers=20,seconds=2.)):
            with self.assertRaises(ValueError):
                WorldInputSearchAdapter(cfg,timed_reference_atomic_work=True,checkpoint_contract=None,showdown_root="")
        with self.assertRaises(ValueError):
            WorldInputSearchAdapter(SearchConfiguration("reference",workers=20),timed_reference_atomic_work=True,
                initial_dispatch_workers=5,checkpoint_contract=None,showdown_root="")

    def test_actual_twenty_workers_expired_capture_deadline_retains_empty_exports(self):
        with ParallelTrajectorySearch(ReferenceConfig(.5,1),TimedAtomicToyRuntime,workers=20,
            initial_dispatch_workers=6) as pool:
            expired=time.perf_counter()-.01
            selected=pool.search(Request(),ROOT,battle_id="battle-1",seed=7,
                deadline_seconds=1.,deadline_at=expired,adapter_started_at=expired-1.,timed_zero_work=True)
            evidence=asdict(selected);evidence["consumed_world_inputs"]=reference_manifest(evidence,BINDING)
            self.assertEqual(self.check(evidence)["accepted_units"],0)
            self.assertEqual(len(selected.worker_receipts),20)
            self.assertEqual(selected.result.world_draws,0)
            self.assertIsNone(selected.result.action)
            self.assertTrue(all(row["deadline_expired"] for row in selected.initial_dispatch_schedule))
            with self.assertRaises(ReferenceRefusal):
                pool.search(Request(),ROOT,battle_id="other",seed=7,trajectories_per_worker=1)
        self.assertTrue(all(not p.is_alive() and p.exitcode==0 for p in pool._processes))

    def test_actual_twenty_worker_positive_prefix_and_operational_refusal(self):
        for failure in (False,True):
            with ParallelTrajectorySearch(ReferenceConfig(.5,1),TimedAtomicToyRuntime,workers=20,
                initial_dispatch_workers=6) as pool:
                began=time.perf_counter()
                args=dict(battle_id="battle-1",seed=7,deadline_seconds=1.,
                    deadline_at=began+1.,adapter_started_at=began,timed_zero_work=True)
                if failure:
                    with self.assertRaisesRegex(ParallelRefusal,"nonbankable"):
                        pool.search(Request(failure=True),ROOT,**args)
                else:
                    selected=pool.search(Request(),ROOT,**args)
                    evidence=asdict(selected);evidence["consumed_world_inputs"]=reference_manifest(evidence,BINDING)
                    self.assertGreater(self.check(evidence)["accepted_units"],0)
                    nonfinal=next(row for row in evidence["worker_receipts"]
                        if row["timed_acknowledgement"]["limit"] == 10)
                    bad=deepcopy(evidence)
                    index=evidence["worker_receipts"].index(nonfinal)
                    bad["worker_receipts"][index]["batch"]["deadline_exhausted"]=True
                    bad["worker_receipts"][index]["timed_acknowledgement"].update(
                        limit=0,worker_deadline_exhausted=True)
                    with self.assertRaisesRegex(ValueError,"continued after|acknowledgement"):
                        self.check(bad)
                    with self.assertRaises(ReferenceRefusal):pool.search(Request(),ROOT,**args)
            self.assertTrue(all(not p.is_alive() for p in pool._processes))

    def test_pool_opt_in_rejects_historical_fixed_or_changed_dispatch(self):
        with ParallelTrajectorySearch(ReferenceConfig(.5,1),TimedAtomicToyRuntime,workers=2) as pool:
            with self.assertRaises(ReferenceRefusal):
                pool.search(Request(),ROOT,battle_id="b",seed=7,deadline_seconds=1.,
                    deadline_at=time.perf_counter()+1.,adapter_started_at=time.perf_counter(),timed_zero_work=True)
        with ParallelTrajectorySearch(ReferenceConfig(.5,1),TimedAtomicToyRuntime,workers=20,
            initial_dispatch_workers=6) as pool:
            for changes in (dict(trajectories_per_worker=4),dict(timed_zero_work=1),dict(deadline_at=None),
                dict(deadline_at=time.perf_counter()+100.)):
                began=time.perf_counter()
                args=dict(deadline_seconds=1.,deadline_at=began+1.,adapter_started_at=began,timed_zero_work=True)
                args.update(changes)
                with self.assertRaises(ReferenceRefusal):pool.search(Request(),ROOT,battle_id="b",seed=7,**args)
            pool.search(Request(),ROOT,battle_id="b",seed=7,trajectories_per_worker=1)
            with self.assertRaises(ReferenceRefusal):
                pool.search(Request(),ROOT,battle_id="c",seed=7,deadline_seconds=1.,
                    deadline_at=time.perf_counter()+1.,adapter_started_at=time.perf_counter(),timed_zero_work=True)

    def test_done_after_positive_ack_is_an_operational_protocol_refusal(self):
        # Exercise the actual master protocol with an authored premature done;
        # there are no spawned workers or model/simulator calls in this test.
        pool=object.__new__(ParallelTrajectorySearch)
        pool.workers=20;pool.batch_size=10;pool.initial_dispatch_workers=6
        pool._closed=pool._poisoned=False;pool._master=None;pool._decision_id=0
        pool._connections=[Mock() for _ in range(20)];pool.worker_pids=tuple(range(100,120))
        pool.startup_seconds=.01
        _,_,evidence,batch,update=fixture()
        pool._receive=Mock(side_effect=[iter([(0,("update",1,0,update,batch,.01,evidence))]),
            iter([(0,("done",1,0))])])
        began=time.perf_counter()
        with self.assertRaisesRegex(ParallelRefusal,"stop acknowledgement"):
            pool.search(Request(),ROOT,battle_id="battle-1",seed=7,deadline_seconds=1.,
                deadline_at=began+1.,adapter_started_at=began,timed_zero_work=True)
        self.assertEqual(pool._connections[0].send.call_args.args[0][-1],10)
        self.assertTrue(pool._poisoned)

    def test_update_after_actual_stop_ack_is_an_operational_protocol_refusal(self):
        pool=object.__new__(ParallelTrajectorySearch)
        pool.workers=20;pool.batch_size=10;pool.initial_dispatch_workers=6
        pool._closed=pool._poisoned=False;pool._master=None;pool._decision_id=0
        pool._connections=[Mock() for _ in range(20)];pool.worker_pids=tuple(range(100,120))
        pool.startup_seconds=.01
        _,_,evidence,batch,update=fixture()
        batch=replace(batch,deadline_exhausted=True)
        message=("update",1,0,update,batch,.01,evidence)
        pool._receive=Mock(side_effect=[iter([(0,message)]),iter([(0,message)])])
        origin=time.perf_counter()-2.
        with self.assertRaisesRegex(ParallelRefusal,"update arrived after"):
            pool.search(Request(),ROOT,battle_id="battle-1",seed=7,deadline_seconds=1.,
                deadline_at=origin+1.,adapter_started_at=origin,timed_zero_work=True)
        self.assertEqual(pool._connections[0].send.call_args.args[0][-1],0)
        self.assertTrue(pool._poisoned)

    def test_partial_and_unsampled_cancellations_retained_but_never_backed_up(self):
        for sampled in (False,True):
            evidence=evidence_fixture(zero=not sampled)
            now=[1.]
            def factory(rng):
                if not sampled:
                    now[0]=2.05;raise SamplingDeadlineExceeded()
                world=ToyWorld(value=1.)
                advance=world.advance
                def slow(*args):
                    now[0]+=.6
                    return advance(*args)
                world.advance=slow
                return world
            _,_,partial,batch,update=fixture(factory=factory,clock=lambda:now[0],deadline=2.)
            if not sampled:
                draw=partial["draws"][0];draw.pop("consumed_world_input")
                draw["sampling_diagnostic"]=dict(schema="pokezero.world-sampling-deadline.v1",
                    checked_at=2.05,deadline_at=2.,accepted_world=False,backed_up=False)
            rec=evidence["worker_receipts"][0]
            rec.update(evidence=partial,batch=asdict(batch),own_statistics=update_record(update))
            root_rows=[next((r for r in rct["own_statistics"]["rows"] if r["state"]["key"]==ROOT.key.hex()),
                dict(visits=[0,0],totals=[0.,0.])) for rct in evidence["worker_receipts"]]
            visits=[sum(r["visits"][i] for r in root_rows) for i in range(2)]
            totals=[sum(r["totals"][i] for r in root_rows) for i in range(2)]
            result=evidence["result"]
            result.update(trajectories=sum(r["batch"]["trajectories"] for r in evidence["worker_receipts"]),
                world_draws=sum(r["batch"]["world_draws"] for r in evidence["worker_receipts"]),
                transitions=sum(r["batch"]["transitions"] for r in evidence["worker_receipts"]),
                root_visits=visits,root_means=[q/n if n else None for q,n in zip(totals,visits)])
            if sampled:result["action"]=ROOT.actions[max(range(2),key=lambda i:(visits[i],-i))]
            evidence["consumed_world_inputs"]=reference_manifest(evidence,BINDING)
            accepted=self.check(evidence)["accepted_units"]
            self.assertEqual(accepted,result["trajectories"])
            cancelled=partial["atomic_work"]["rows"][-1]
            self.assertEqual(cancelled["status"],"CANCELLED")
            self.assertEqual(cancelled["before"],cancelled["after"])
            bad=deepcopy(evidence);bad["worker_receipts"][0]["evidence"]["atomic_work"]["rows"][-1]["status"]="REFUSED"
            with self.assertRaises(ValueError):self.check(bad)
            if not sampled:
                bad=deepcopy(evidence)
                bad["worker_receipts"][0]["evidence"]["draws"][0]["sampling_diagnostic"]["deadline_at"]=1.9
                bad["consumed_world_inputs"]=reference_manifest(bad,BINDING)
                with self.assertRaises(ValueError):self.check(bad)

    def test_actual_adapter_handoff_keeps_absolute_capture_deadline_and_one_shot_status(self):
        from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory
        for leaf in ("model","hp_fraction"):
            for zero in (False,True):
                cfg=SearchConfiguration("reference",workers=20,leaf=leaf,seconds=1.)
                obs=SimpleNamespace(legal_action_mask=(True,True))
                request=SimpleNamespace(state=SimpleNamespace(player_id="p1"),observation=obs,
                    set_source_hash=BINDING["set_source_hash"])
                context=AuthoredPublicContext("p1",obs,{"p1":obs,"p2":"private-canary"},request.state,
                    SimpleNamespace(terminal=None))
                contract=SimpleNamespace(checkpoint_path="authored-weights",checkpoint_sha256="a"*64,
                    showdown_source_sha256=BINDING["set_source_hash"])
                factory=ShowdownWorkerFactory(contract.checkpoint_path,contract.checkpoint_sha256,
                    "authored-showdown",contract.showdown_source_sha256)
                pool=Mock()
                with patch("pokezero.mcts_eval.paper_reference_parallel.ParallelTrajectorySearch",return_value=pool), \
                    patch("pokezero.mcts_eval.paper_reference_runtime.PublicRootRequest.capture",return_value=request), \
                    patch("pokezero.mcts_eval.head_to_head.public_only_context",side_effect=lambda c:c), \
                    patch("pokezero.mcts_eval.paper_reference_showdown.decision_state",return_value=replace(ROOT,actions=("action:0","action:1"))), \
                    patch("pokezero.mcts_eval.search_over_raw_world_inputs._binding",return_value=BINDING), \
                    patch("pokezero.mcts_eval.search_over_raw_adapters.time.perf_counter",return_value=100.) as clock:
                    adapter=WorldInputSearchAdapter(cfg,timed_reference_atomic_work=True,
                        checkpoint_contract=contract,showdown_root="authored-showdown",reference_factory=factory)
                    def search(*args,**kw):
                        evidence=evidence_fixture(zero=zero,encoded=True,hp=leaf=="hp_fraction")
                        prefix=evidence["reference_timed_prefix"]
                        prefix.update(battle_id=kw["battle_id"],selection_seed=kw["seed"])
                        for key in ("adapter_started_monotonic","decision_started_monotonic",
                            "decision_finished_monotonic","absolute_deadline_monotonic"):
                            prefix[key]+=99.
                        for row in prefix["stop_acknowledgements"]:
                            row["acknowledged_monotonic"]+=99.;row["decided_monotonic"]+=99.
                        for row in evidence["worker_receipts"]:
                            row["own_statistics"]["battle_id"]=kw["battle_id"]
                            row["batch"]=BatchResult(**row["batch"])
                            row["timed_acknowledgement"]["acknowledged_monotonic"]+=99.
                            row["timed_acknowledgement"]["decided_monotonic"]+=99.
                        clock.return_value=101.2
                        return ParallelResult(SearchResult(**evidence["result"]),tuple(evidence["worker_pids"]),
                            tuple(evidence["worker_receipts"]),20,.01,1,tuple(evidence["initial_dispatch_schedule"]),prefix)
                    pool.search.side_effect=search
                    selected=adapter.select(context,root_id="root:"+cfg.identity,statistics_root_id="root",selection_seed=7)
                    self.assertEqual(pool.search.call_args.kwargs["deadline_at"],101.)
                    self.assertEqual(pool.search.call_args.kwargs["adapter_started_at"],100.)
                    self.assertTrue(pool.search.call_args.kwargs["timed_zero_work"])
                    self.assertEqual(selected["status"],"UNQUALIFIED_AT_CEILING" if zero else "SELECTED")
                    self.assertAlmostEqual(selected["elapsed_seconds"],1.2)
                    read_reference_timed_selection(selected,configuration=cfg,
                        runtime_configuration=adapter.runtime_configuration,runtime_sha256=adapter.runtime_sha256,
                        root_id="root:"+cfg.identity,statistics_root_id="root",selection_seed=7,
                        consumed_binding=BINDING,legal_action_mask=(True,True))
                    with self.assertRaises(ValueError):
                        adapter.select(context,root_id="other",statistics_root_id="other",selection_seed=7)
                    adapter.close();pool.close.assert_called_once()



if __name__ == "__main__": unittest.main()
