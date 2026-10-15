"""Disposable process and exact existing CLI checks; no scientific execution."""
from dataclasses import dataclass
import json
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.mcts_eval import search_over_raw_collector_cost as cost
from pokezero.mcts_eval.search_over_raw_ledger import _write_new
from pokezero.mcts_eval.search_over_raw_supervision import _supervise_owned_job, observe_original_supervisor_launch
from pokezero.mcts_eval.search_over_raw_fixed_work import FixedWorkDiagnosticContract
from pokezero.mcts_eval.paper_reference_parallel import owned_process_identity, live_owned_groups
from test_search_over_raw_supervision import SyntheticJob, BINDING
from test_search_over_raw_nested_ownership import NestedJob
from pokezero.mcts_eval.search_over_raw_outcome_run import OwnedContinuationJob


@dataclass(frozen=True)
class LocatedAuthoredContinuationJob(OwnedContinuationJob):
    """Diagnostic-only fixture; export code locations, never private errors."""
    synthetic_process_fixture = True
    def __call__(self, ownership, phase):
        try:
            return super().__call__(ownership, phase)
        except BaseException as failure:
            import traceback
            source = Path(__file__).resolve().parents[1]
            frames = [dict(file=str(Path(row.filename).relative_to(source)), line=row.lineno, function=row.name)
                for row in traceback.extract_tb(failure.__traceback__) if Path(row.filename).is_relative_to(source)]
            _write_new(Path(ownership["directory"]).parent / "authored-failure-location.json",
                dict(error_type=type(failure).__name__, frames=frames[-6:]))
            raise


@dataclass(frozen=True)
class StoppedOrdinaryJob(SyntheticJob):
    def __call__(self, ownership, phase):
        from pokezero.mcts_eval.paper_reference_parallel import record_owned_process
        phase("startup"); phase("selection")
        child = subprocess.Popen([sys.executable, "-c", "import signal; signal.pause()"], start_new_session=True)
        record_owned_process(child.pid, ownership)
        os.kill(child.pid, signal.SIGSTOP)
        _write_new(Path(ownership["directory"]).parent / "authored-stopped.json", dict(pid=child.pid))
        signal.pause()


def fixture(directory, mode):
    """Authored command, not an alternate public collector/admission route."""
    claim = cost._check_claim(directory)
    header = json.loads(sys.stdin.buffer.read())
    launch = cost._launch(claim)
    assert header == dict(pid=os.getpid(), claim_sha256=launch["claim_sha256"])
    directory = Path(directory)
    def register(descriptor):
        if mode == "before_registration":
            _write_new(directory / "authored-preheader.json", dict(pid=descriptor["launch"]["pid"]))
            signal.pause()
        if mode == "registration_failure":
            raise RuntimeError("authored external registration refusal")
        _write_new(directory / "auditor-registration.json", dict(schema=cost.SCHEMA + ".registration",
            claim_sha256=launch["claim_sha256"], cli_launch_sha256=cost._sha(directory / "launch.json"),
            descriptor=descriptor))
    with observe_original_supervisor_launch(register):
        _write_new(directory / "child-ready.json", dict(schema=cost.SCHEMA + ".ready", cli_pid=os.getpid(),
            claim_sha256=launch["claim_sha256"], cli_launch_sha256=cost._sha(directory / "launch.json"),
            hook_installed_before_entry=True))
        if mode == "before_entry":
            signal.pause()
        if mode == "nested":
            job = NestedJob(str(Path(claim["ledger_directory"]) / "nested"), "hang_grandchild")
        elif mode == "stopped":
            # -m runs this file as __main__; pickle must bind the importable
            # fixture class, not the worker interpreter's own __main__.
            from test_search_over_raw_collector_cost import StoppedOrdinaryJob as ImportedStoppedJob
            job = ImportedStoppedJob()
        else:
            job = SyntheticJob("cpu_child" if mode == "normal" else "hang" if mode == "hang" else "complete")
        result = _supervise_owned_job(job, directory=Path(claim["ledger_directory"]) / "exploration-entry-run",
            binding_sha256=BINDING, verify_admission=lambda: True, contract=FixedWorkDiagnosticContract(),
            kind="staged-exploration-run", work_phase="selection", validate_result=lambda result: None)
        if mode == "normal":
            # Non-calling-thread and final-output CPU belong to the CLI wait4.
            def burn():
                began = time.thread_time()
                while time.thread_time() - began < .1:
                    pass
            worker = threading.Thread(target=burn); worker.start(); worker.join()
        print(json.dumps(dict(status=result["status"], holdout_authorized=False,
            phase_b_authorized=False, resource_cost_complete=False)), flush=True)
        _write_new(directory / "child-close.json", dict(schema=cost.SCHEMA + ".close", cli_pid=os.getpid(),
            claim_sha256=launch["claim_sha256"], error_type=None, query_census=cost._query_census()))
        return result["supervisor_exit_code"]


class CollectorInvocationCostTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def claim(self, mode="normal", watchdog=10.):
        directory = self.root / "cost"; directory.mkdir()
        ledger = self.root / "ledger"; ledger.mkdir()
        spec = self.root / "authored.json"; _write_new(spec, dict(authored=True))
        argv = [sys.executable, "-B", "-m", "test_search_over_raw_collector_cost", "fixture", str(directory), mode]
        _write_new(directory / "claim.json", dict(schema=cost.SCHEMA + ".claim", directory=str(directory),
            ledger_directory=str(ledger), spec_path=str(spec), input_hashes={str(spec): cost._sha(spec)},
            framework_hashes={**cost._bindings(), str(Path(__file__).resolve()): cost._sha(__file__)}, entry_job=None,
            argv=argv, executable_sha256=cost._executable_sha(), owner_pid=os.getpid(), owner_birth=owned_process_identity(os.getpid()),
            watchdog_seconds=watchdog, synthetic_process_fixture=True,
            admission_override=False, retries_permitted=False, holdout_authorized=False))
        return directory, ledger

    def admitted_authored_collector(self, *, p2_only_terminal=False, owned_clone_transport=False,
            native_selector=False, compact_native_selector=False):
        """Real checkpoint loader/env/driver; scripted protocol, not science.

        No monkeypatch crosses into either collector interpreter. All recipes
        and evidence are prospective disposable fixtures, not qualification
        of a champion, native search, or a production simulator.
        """
        from dataclasses import asdict
        from pokezero.local_showdown import LocalShowdownConfig
        from pokezero.collection import env_config_with_policy_spec_masks
        from pokezero.neural_policy import (EntityTokenTransformerPolicy, TransformerPolicyConfig,
            TransformerTrainingConfig, TransformerTrainingResult, save_transformer_checkpoint, require_torch)
        from pokezero.randbat import load_gen3_randbat_source_cached
        from pokezero.randbat_vocab import gen3_category_vocabulary
        from pokezero.mcts_eval.resolver import resolve_checkpoint_contract
        from pokezero.mcts_eval.search_over_raw import SearchConfiguration, digest
        from pokezero.mcts_eval.search_over_raw_world_inputs import _state_payload
        from pokezero.mcts_eval.search_over_raw_outcome_run import outcome_runtime_binding
        from pokezero.mcts_eval.search_over_raw_raw_run import raw_runtime
        from pokezero.mcts_eval.search_over_raw_source_run import source_recipe, source_review_template
        from pokezero.mcts_eval.search_over_raw_stage_run import unavailable_configuration_review_template
        from pokezero.mcts_eval.search_over_raw_continuation_admission import fresh_history_review_template
        from pokezero.mcts_eval.search_over_raw_fixed_work_run import fixed_work_reader_source_hashes
        from test_search_over_raw_exploration_entry import ExplorationEntryTests
        from test_search_over_raw_actor_projection import own_request
        from pokezero.mcts_eval.search_over_raw_reconciled import configuration_review_template, selector_runtime_recipe
        from pokezero.mcts_eval.search_over_raw_actor_timed_run import actor_timed_runtime_recipe
        helper = ExplorationEntryTests(); helper.setUp(); self.addCleanup(helper.doCleanups)
        self.assertTrue(not compact_native_selector or native_selector)
        showdown = Path("/Users/scott/workspace/pokerena/vendor/pokemon-showdown").resolve()
        self.assertTrue((showdown / "dist/sim/index.js").is_file(), "existing read-only Showdown required; never build/install")
        catalog = load_gen3_randbat_source_cached(showdown)
        vocab = gen3_category_vocabulary(showdown, include_turn_merged=True, extra_moves=("tackle",))
        native_files = []
        native_fixture = None
        if native_selector:
            self.assertTrue(owned_clone_transport, "nonempty fixture requires original owned clone transport")
            from test_native_fixed_work_adapter import LoadedNativeValueAdapterTests
            import pokezero_search
            import poke_engine
            # Legacy fixture retains its original ABI pin. Compact is a distinct
            # prospective engineering fixture, not an upgrade of old evidence.
            expected_search = ("50f85a461639008af199136ccd3838acff84ab19e452e1ad71f59f57506c5200"
                if compact_native_selector else "bdd3228dce2a2762127a54b995447ff49406af4c9ccd0372a57ce8a2068cf9ed")
            if compact_native_selector:
                wheel = Path(os.environ["POKEZERO_COMPACT_TEST_WHEEL"]).resolve(strict=True)
                self.assertEqual(cost._sha(wheel),
                    "52a12aee9648bd7b002e67e5a2e4ed8a209031be79a61805466aa1c903ac7ff2")
                native_files.append(wheel)
            for package, expected in ((pokezero_search, expected_search),
                    (poke_engine, "1a709252987ed29330d48b25e12cf759a3470c5a8c6ada04a6767ddbed625c20")):
                directory = Path(package.__file__).resolve().parent
                binaries = list(directory.glob("*.so")); self.assertEqual(len(binaries), 1)
                self.assertEqual(cost._sha(binaries[0]), expected, "matching reviewed native ABI required")
                native_files.extend([Path(package.__file__).resolve(), binaries[0]])
            # Existing random-model fixture authors disposable TorchScript and
            # tables; no champion/checkpoint/scientific source is modified.
            LoadedNativeValueAdapterTests.setUpClass()
            self.addCleanup(LoadedNativeValueAdapterTests.tearDownClass)
            native_fixture = LoadedNativeValueAdapterTests(); self.addCleanup(native_fixture.doCleanups)
            checkpoint = native_fixture.checkpoint_contract
            checkpoint_path = Path(checkpoint.checkpoint_path)
            native_files.extend([Path(checkpoint.model_path), Path(checkpoint.tables_path)])
        else:
            checkpoint_path = self.root / "authored-untrained.pt"
            cfg = TransformerPolicyConfig.compact_category(category_vocab=vocab.tokens, category_oov_buckets=vocab.oov_buckets,
                observation_schema_version="pokezero.observation.v4", policy_id="authored-collector-cost-only",
                window_size=1, embedding_dim=8, attention_heads=2, transformer_layers=1, feedforward_dim=16, dropout=0.)
            torch = require_torch()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(2026101401)
                model = EntityTokenTransformerPolicy(cfg)
                save_transformer_checkpoint(checkpoint_path, model, result=TransformerTrainingResult(cfg,
                    TransformerTrainingConfig(device="cpu"), (), belief_set_source_hash=catalog.metadata.source_hash))
            checkpoint = resolve_checkpoint_contract(checkpoint_path, model_device="cpu", showdown_root=showdown,
                showdown_source_sha256=catalog.metadata.source_hash)
        bridge = self.root / "authored-protocol"; bridge.mkdir()
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        (bridge / "battle_bridge.mjs").write_bytes(Path(__file__).with_name("source_bridge_fixture.mjs").read_bytes())
        for name in ("battle_bridge_owned_snapshot.mjs", "battle_bridge_owned_continuation.mjs"):
            (bridge / name).write_bytes((scripts / name).read_bytes())
        clone_files = []
        if owned_clone_transport:
            import re
            # Reuse the existing clone-only transport; never import Showdown,
            # regenerate source teams or turn its scripted winner into science.
            clones = bridge / "authored-clones"; clones.mkdir()
            text = Path(__file__).with_name("test_owned_continuation.mjs").read_text()
            for variable, name in (("fakeRuntime", "index.js"), ("fakeState", "state.js")):
                path = clones / name
                payload = re.search(r"const " + variable + r" = `([\s\S]*?)`;", text).group(1)
                path.write_text(payload.replace("|win|Fixture One", "|win|PokeZero p1"))
                clone_files.append(path)
        settings_path = bridge / "source-settings.json"
        settings = dict(rounds=1 if p2_only_terminal else 300, mode="complete", snapshotTransport=True,
            ownedContinuationTransport=owned_clone_transport,
            actorRequests={"p1": {"wait": True} if p2_only_terminal else own_request("p1"), "p2": own_request("p2")},
            actorPrefix=["|player|p1|PokeZero p1|", "|player|p2|PokeZero p2|", "|gen|3", "|start",
                "|switch|p1a: Charizard|Charizard, L80, M|250/250",
                "|switch|p2a: Charizard|Charizard, L80, M|250/250", "|turn|1"])
        if native_selector:
            from pokezero.local_showdown import showdown_seed_from_int
            prefix, request = native_fixture.authored_protocol()
            prefix = [line.replace("|player|p1|authored p1|", "|player|p1|PokeZero p1|")
                .replace("|player|p2|authored p2|", "|player|p2|PokeZero p2|") for line in prefix]
            p2_request = json.loads(json.dumps(request).replace("p1", "p2"))
            selected_seed = helper.h.contract["panels"]["exploration"]["seeds"][0]
            # Freeze source identities before work. Exactly one non-forced p1
            # request; all other seeds are short p2-only terminal sources.
            settings.update(rounds=1, actorRequests={"p1":{"wait":True}, "p2":p2_request},
                actorPrefix=prefix, requestsBySeed={json.dumps(showdown_seed_from_int(selected_seed),
                    separators=(",", ":")): {"p1":request, "p2":{"wait":True}}})
        _write_new(settings_path, settings)
        config = env_config_with_policy_spec_masks(LocalShowdownConfig(showdown_root=str(showdown),
            bridge_path=str(bridge / "battle_bridge.mjs"), set_belief_source=True, read_timeout_seconds=2.),
            ["neural:" + str(checkpoint_path)], context="authored collector cost integration")
        # Includes source/settings files used by catalog/Dex as well as dist;
        # no existing source checkout or model is modified by this fixture.
        catalog_files = [p.resolve() for folder in (showdown / "data", showdown / "config")
            for p in folder.rglob("*") if p.is_file()]
        runtime = outcome_runtime_binding(config, checkpoint_path=checkpoint_path,
            extra_files=[settings_path, *clone_files, *native_files, *catalog_files])
        def alter(spec, contract):
            spec.update(source_config=_state_payload(config), checkpoint_contract=_state_payload(checkpoint),
                source_runtime=runtime, continuation_config=_state_payload(config), continuation_runtime=runtime,
                emergency_watchdog_seconds=1200., raw_runtime=raw_runtime(SearchConfiguration("raw", seconds=1.),
                    checkpoint.checkpoint_sha256, checkpoint.showdown_source_sha256),
                selector_files={**fixed_work_reader_source_hashes(), **runtime["files"],
                    str(Path(sys.executable).resolve()): cost._executable_sha()})
            for row in spec["recipes"]:
                stage, identity, path = row["stage"], row["configuration"], row["review_path"]
                available = SearchConfiguration("incumbent", seconds=1.)
                if native_selector and identity == available.identity:
                    selector_runtime = actor_timed_runtime_recipe(available, checkpoint,
                        **({"compact_native_atomic_work":True} if compact_native_selector else {}))
                    evidence = json.loads(Path(path).read_bytes())["qualification_evidence_hashes"]
                    review = configuration_review_template(contract, stage=stage, configuration=identity,
                        raw_runtime_sha256=digest(spec["raw_runtime"]),
                        selector_recipe_sha256=digest(selector_runtime_recipe(available, selector_runtime)),
                        qualification_evidence_hashes=evidence,
                        cost_scope="Authored nonempty accounting integration; not production qualification.")
                    review.update(status="CLEAR_CONFIGURATION_RECIPE", reviewed_before_first_source_work=True,
                        atomic_work_and_cleanup_validated=True, resource_cost_qualification_reviewed=True)
                    Path(path).write_text(json.dumps(review))
                    row.clear(); row.update(stage=stage, configuration=identity, runtime=selector_runtime,
                        review_path=path, reference_factory=None)
                    continue
                reason = "Authored accounting fixture has no admitted selector runtime."
                review = unavailable_configuration_review_template(contract, stage=stage, configuration=identity, reason=reason)
                review.update(status="CLEAR_PROSPECTIVE_CONFIGURATION_UNAVAILABLE", independent_review_completed=True,
                    reviewed_before_first_source_work=True)
                Path(path).write_text(json.dumps(review))
                row.clear(); row.update(stage=stage, configuration=identity, unavailable_reason=reason, review_path=path)
            prior = json.loads(Path(spec["source_review_path"]).read_bytes())
            source_review = source_review_template(contract, source_recipe(config, checkpoint, runtime),
                qualification_evidence_hashes=prior["qualification_evidence_hashes"])
            source_review.update(status="CLEAR_SOURCE_RECIPE", reviewed_before_first_source_work=True,
                source_runtime_and_cleanup_reviewed=True, source_policy_and_terminal_reviewed=True)
            Path(spec["source_review_path"]).write_text(json.dumps(source_review))
            history = fresh_history_review_template(contract, ledger_directory=helper.directory,
                raw_runtime_sha256=digest(spec["raw_runtime"]), continuation_runtime_sha256=digest(runtime))
            history.update(status="CLEAR_FRESH_DECLARED_NAMESPACE", reviewed_before_first_source_work=True)
            Path(spec["history_review_path"]).write_text(json.dumps(history))
        helper.make(alter=alter)
        helper.preflight()  # Real read-only prospective original admission.
        return helper

    def check_actual_admitted_cli_empty_panel_cost(self, *, p2_only_terminal):
        from pokezero.mcts_eval.search_over_raw_exploration_entry import (ExplorationAuditorJob,
            read_exploration_entry, read_entry_observer_cost)
        from pokezero.mcts_eval.search_over_raw_attempt_journal import AttemptJournalHandle, read_attempt_journal_cost
        helper = self.admitted_authored_collector(p2_only_terminal=p2_only_terminal)
        directory = self.root / "actual-admitted-cost"
        # CPU-only authored execution, no training or champion inference.
        with patch.dict(os.environ, {"POKEZERO_TORCH_NUM_THREADS": "1", "POKEZERO_TORCH_NUM_INTEROP_THREADS": "1"}):
            terminal = cost.run_collector_cost(helper.directory, helper.spec_path, directory=directory, watchdog_seconds=1200.)
        self.assertEqual(terminal["actual_exit_code"], 0, terminal)
        claim = cost._check_claim(directory)
        job = ExplorationAuditorJob(**claim["entry_job"])
        outer = json.loads((helper.directory / "exploration-entry-run/terminal.json").read_bytes())
        completed = read_exploration_entry(job, outer)
        observed = read_entry_observer_cost(job, outer)
        journal = helper.directory / "attempt-cost-journal"
        census = read_attempt_journal_cost(AttemptJournalHandle(str(journal), cost._sha(journal / "metadata.json")),
            expected_outer_terminal=outer)
        measured = cost.read_collector_cost(directory)
        # Failure evidence is deliberately limited to code-owned categories and
        # numeric/boolean completeness fields, never private snapshots/errors.
        diagnostics = dict(incomplete_components=measured["incomplete_components"],
            observer_unknown_queries=observed["observer_window"]["own_query_usage_unknown"],
            publication_usage_unknown=observed["publication_usage_unknown"],
            journal_census_complete=census["declared_supervised_attempt_census_complete"],
            journal_normal_closed=census["all_journal_writers_normal_closed"], attempts=[])
        from pokezero.mcts_eval.search_over_raw_resources import read_sampling_cpu_components
        for attempt in census["attempts"]:
            trace = json.loads((Path(attempt["directory"]) / "memory-trace.json").read_bytes())
            sampling = read_sampling_cpu_components(trace)
            diagnostics["attempts"].append(dict(cpu_complete=attempt.get("measurement_window_cpu_wall_complete"),
                rss_unknown_cpu_records=sampling["unknown_sample_cpu_records"],
                rss_pending_query_count=len(sampling["pending_sampler_cleanup_pids"]),
                rss_query_cost_complete=sampling["rss_query_cost_window_complete"]))
        self.assertTrue(measured["invocation_cpu_wall_complete"], diagnostics)
        self.assertEqual(measured["total_invocation_cpu_seconds"], terminal["actual_cpu_seconds_including_reaped_children"])
        self.assertTrue(census["declared_supervised_attempt_census_complete"])
        self.assertTrue(census["all_journal_writers_normal_closed"])
        self.assertEqual(len(census["attempts"]), 1)  # Original auditor, no extra cost ownership depth.
        result = outer["result"]
        self.assertEqual(result["launches"], dict(raw=0, search=0, continuation=0, reused_selectors=0, reused_a3_records=0))
        self.assertEqual(result["source_panel"]["source_count"], 32)
        self.assertEqual(result["source_panel"]["observed_root_ids"], [])
        self.assertEqual(len(result["source_panel"]["missing_root_ids"]), 200)
        self.assertFalse(measured["resource_cost_complete"])
        self.assertFalse(measured["runtime_qualified"])
        self.assertFalse(measured["holdout_authorized"])
        self.assertEqual(list((helper.directory / "shared-raw-runs").iterdir()), [])
        for path in (helper.directory / "source-attempts").iterdir():
            source = json.loads((path / "source.json").read_bytes())
            driver = json.loads((path / "driver.json").read_bytes())
            close = json.loads((path / "bridge-close.json").read_bytes())
            self.assertEqual(source["status"], "COMPLETE" if p2_only_terminal else "UNCERTAIN_SOURCE")
            self.assertEqual(source["decision_boundaries"], 1 if p2_only_terminal else 250)
            self.assertEqual(source["eligible_requests"], 0)
            self.assertEqual(source["roots"], [])
            self.assertEqual(source["source_terminal_complete"], p2_only_terminal)
            self.assertEqual(source["verified_raw_decisions"], 1 if p2_only_terminal else 500)
            self.assertEqual(driver["terminal_capped"], not p2_only_terminal)
            self.assertEqual(driver["decisions_by_seat"], {"p1": 0, "p2": 1} if p2_only_terminal else {"p1": 250, "p2": 250})
            self.assertTrue(close["normal_close"])
            self.assertEqual(close["exit_code"], 0)
        # No completed reader may bank missing census or rehash a CLI status.
        close_path = directory / "child-close.json"; original = close_path.read_bytes()
        changed = json.loads(original); changed["query_census"]["pending_queries"] = 1
        close_path.write_text(json.dumps(changed))
        self.assertFalse(cost.read_collector_cost(directory)["invocation_cpu_wall_complete"])
        close_path.write_bytes(original)
        self.assertTrue(cost.read_collector_cost(directory)["invocation_cpu_wall_complete"])

    def test_actual_admitted_cli_empty_panel_original_readers_and_complete_invocation_cost(self):
        # Both-seat 250-cap regression remains required, not replaced by the
        # short terminal diagnosis below. Original production cap is unchanged.
        self.check_actual_admitted_cli_empty_panel_cost(p2_only_terminal=False)

    def test_actual_admitted_cli_p2_only_terminal_cost_diagnosis(self):
        # Real original CLI/loader/env/driver/readers, but authored protocol and
        # untrained model. No p1 root exists; every fixed p1 slot stays missing.
        self.check_actual_admitted_cli_empty_panel_cost(p2_only_terminal=True)

    def test_authored_source_snapshot_original_continuation_job_and_reader(self):
        # Dependency coverage only, NOT nonempty full-CLI cost coverage. Source
        # and terminal streams are scripted; checkpoint/env/evaluator/job and
        # owning readers are original, without policy or receipt substitutions.
        from pokezero.local_showdown import LocalShowdownEnv
        from pokezero.mcts_eval.search_over_raw_exploration_entry import _decode
        from pokezero.mcts_eval.search_over_raw_outcome_run import (OwnedContinuationJob,
            OwningRunValidator, supervise_outcome_run)
        from pokezero.public_decision_corpus import PublicDecisionRecord, PublicObservation, _public_belief_view
        from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest
        from pokezero.mcts_eval.search_over_raw_world_inputs import public_request_witness
        import hashlib
        import pickle
        helper = self.admitted_authored_collector(owned_clone_transport=True)
        config = _decode(helper.spec["source_config"])
        env = LocalShowdownEnv(config); self.addCleanup(env.close)
        env.reset(seed=17)
        for player in env.requested_players(): env.observe(player)
        observation = env.observe("p1")
        snapshot = env.snapshot_actionable_boundary()
        legal = [i for i, allowed in enumerate(observation.legal_action_mask) if allowed]
        self.assertGreaterEqual(len(legal), 2)
        record = PublicDecisionRecord(decision_id="authored-continuation-dependency",
            battle_id=snapshot.battle_token, seed=17, format_id=snapshot.format_id,
            acting_player="p1", turn_index=0, recorded_action_index=legal[0],
            observation=PublicObservation.from_observation(observation), history=(),
            current_legal_action_mask=tuple(observation.legal_action_mask), public_resolved_action_rounds=(),
            public_belief_view=_public_belief_view(observation.metadata))
        request = PublicRootRequest.capture(env.public_materialization_state("p1"), observation)
        root_binding = dict(root_id="authored-source:17:0", source_seed=17,
            public_record_sha256=cost.digest(record.to_dict()),
            information_key=public_request_witness(request)["consumed_binding"]["information_key"],
            legal_action_mask=list(observation.legal_action_mask))
        env.close()
        diagnostic = os.environ.get("POKEZERO_AUTHORED_LOCATE_CONTINUATION") == "1"
        job_class = LocatedAuthoredContinuationJob if diagnostic else OwnedContinuationJob
        job = job_class(config, snapshot, root_binding,
            helper.spec["continuation_runtime"], "p1", {"raw":legal[0], "candidate":legal[1]},
            "authored-source-clone-dependency")
        directory = self.root / "original-continuations"
        with patch.dict(os.environ, {"POKEZERO_TORCH_NUM_THREADS":"1", "POKEZERO_TORCH_NUM_INTEROP_THREADS":"1"}):
            terminal = supervise_outcome_run(job, directory=directory, binding_sha256="a"*64,
                verify_admission=lambda:True, validate_result=lambda result:None)
        failure_location = directory / "authored-failure-location.json"
        diagnostics = json.loads(failure_location.read_bytes()) if failure_location.exists() else terminal["error_type"]
        self.assertEqual(terminal["status"], "OUTCOME_RUN_COMPLETE", diagnostics)
        self.assertEqual(terminal["worker_exit_code"], 0)
        reader = OwningRunValidator(directory=directory,
            job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(), binding_sha256="a"*64,
            root_binding=root_binding, expected_snapshot=snapshot, runtime=helper.spec["continuation_runtime"],
            allow_synthetic=diagnostic)
        bank = terminal["result"]["bank"]
        self.assertEqual(reader(bank), cost.digest(bank))
        self.assertEqual(len(bank["cells"]), 16)  # Two distinct actions, eight fixed replicates each.
        self.assertEqual({(cell["law"]["action"], cell["law"]["replicate"]) for cell in bank["cells"]},
            {(action, replicate) for action in legal[:2] for replicate in range(8)})
        self.assertEqual(len({cell["battle_token"] for cell in bank["cells"]}), 16)
        outcomes = terminal["result"]["outcomes"]["outcomes"]
        self.assertEqual(len(outcomes), 16)
        self.assertTrue(all(row["status"] == "COMPLETE" and row["signed_outcome"] == 1. for row in outcomes))
        self.assertTrue(all(cell["source_battle_token"] == snapshot.battle_token for cell in bank["cells"]))
        self.assertFalse(terminal["runtime_qualified"])
        self.assertFalse(terminal["resource_cost_complete"])
        closed = json.loads((directory / "bridge-close.json").read_bytes())
        self.assertTrue(closed["normal_close"])
        self.assertEqual(closed["exit_code"], 0)
        changed = dict(terminal, worker_exit_code=7)
        path = directory / "terminal.json"; original = path.read_bytes()
        path.write_text(json.dumps(changed))
        with self.assertRaises(ValueError): reader(bank)
        path.write_bytes(original)
        self.assertEqual(reader(bank), cost.digest(bank))

    def test_actual_admitted_cli_nonempty_native_work_continuations_and_invocation_cost(self):
        """Original scheduler/readers, real random-model native work; toy source/outcomes."""
        self.check_actual_admitted_cli_nonempty_native_work_continuations_and_invocation_cost()

    def test_actual_admitted_cli_compact_nonempty_native_work_continuations_and_invocation_cost(self):
        """Separate prospective compact runtime; original CLI and full roster unchanged."""
        self.check_actual_admitted_cli_nonempty_native_work_continuations_and_invocation_cost(
            compact_native_selector=True)

    def check_actual_admitted_cli_nonempty_native_work_continuations_and_invocation_cost(self,
            *, compact_native_selector=False):
        from pokezero.mcts_eval.search_over_raw import SearchConfiguration
        from pokezero.mcts_eval.search_over_raw_exploration_entry import ExplorationAuditorJob, read_exploration_entry
        from pokezero.mcts_eval.search_over_raw_attempt_journal import AttemptJournalHandle, read_attempt_journal_cost
        helper = self.admitted_authored_collector(owned_clone_transport=True, native_selector=True,
            compact_native_selector=compact_native_selector)
        directory = self.root / "actual-nonempty-cost"
        environment = {"POKEZERO_TORCH_NUM_THREADS":"1", "POKEZERO_TORCH_NUM_INTEROP_THREADS":"1"}
        located = os.environ.get("POKEZERO_AUTHORED_LOCATE_ENTRY") == "1"
        if located:
            # Diagnostic-only interception, explicitly forbidden as positive
            # original-CLI evidence. Never capture exception text/private state.
            diagnostic = self.root / "authored-entry-diagnostic"; diagnostic.mkdir()
            (diagnostic / "sitecustomize.py").write_text('''from pathlib import Path
import traceback
from pokezero.mcts_eval.search_over_raw_exploration_entry import ExplorationAuditorJob
from pokezero.mcts_eval.search_over_raw_ledger import _write_new
original = ExplorationAuditorJob.__call__
def locate(self, ownership, phase):
    try:
        return original(self, ownership, phase)
    except BaseException as failure:
        source = Path(__import__("pokezero").__file__).resolve().parents[2]
        frames = [dict(file=str(Path(row.filename).relative_to(source)), line=row.lineno, function=row.name)
            for row in traceback.extract_tb(failure.__traceback__) if Path(row.filename).is_relative_to(source)]
        _write_new(Path(self.ledger_directory) / "authored-entry-failure-location.json",
            dict(error_type=type(failure).__name__, frames=frames[-8:]))
        raise
ExplorationAuditorJob.__call__ = locate
''')
            environment["PYTHONPATH"] = str(diagnostic) + os.pathsep + os.environ["PYTHONPATH"]
        with patch.dict(os.environ, environment):
            terminal = cost.run_collector_cost(helper.directory, helper.spec_path, directory=directory, watchdog_seconds=1200.)
        # Only code-owned failure categories; never private snapshots/errors.
        outer_path = helper.directory / "exploration-entry-run/terminal.json"
        outer = json.loads(outer_path.read_bytes()) if outer_path.exists() else None
        measured = cost.read_collector_cost(directory)
        diagnostics = dict(cli_exit=terminal["actual_exit_code"], cli_error_type=terminal["error_type"],
            outer_status=None if outer is None else outer["status"],
            outer_error_type=None if outer is None else outer["error_type"], incomplete=measured["incomplete_components"])
        failure = helper.directory / "authored-entry-failure-location.json"
        if failure.exists(): diagnostics["authored_code_location"] = json.loads(failure.read_bytes())
        if compact_native_selector:
            print("COMPACT_INVOCATION_MEASUREMENT:"+json.dumps(dict(actual_exit_code=measured["actual_exit_code"],
                invocation_cpu_wall_complete=measured["invocation_cpu_wall_complete"],
                actual_cli_cpu_seconds_including_reaped_children=measured["actual_cli_cpu_seconds_including_reaped_children"],
                elapsed_invocation_wall_seconds=measured["elapsed_invocation_wall_seconds"]), sort_keys=True), flush=True)
        self.assertEqual(terminal["actual_exit_code"], 0, diagnostics)
        self.assertFalse(located, "diagnostic interception cannot establish original-CLI positive coverage")
        self.assertTrue(measured["invocation_cpu_wall_complete"], diagnostics)
        self.assertIsNotNone(outer, "normal completed invocation requires original auditor terminal")
        claim = cost._check_claim(directory)
        job = ExplorationAuditorJob(**claim["entry_job"])
        completed = read_exploration_entry(job, outer)
        self.assertEqual(completed["status"], "STAGED_EXPLORATION_OWNER_VALIDATED")
        self.assertEqual(completed["source_count"], 32)
        self.assertFalse(completed["full_resource_cost_complete"])
        result = outer["result"]; panel = result["source_panel"]
        self.assertEqual(panel["source_count"], 32)
        self.assertEqual(len(panel["observed_root_ids"]), 1)
        self.assertEqual(len(panel["missing_root_ids"]), 199)
        root = panel["observed_root_ids"][0]
        configuration = SearchConfiguration("incumbent", seconds=1.).identity
        for stage in ("A1", "A2"):
            row = next(row for row in result["reports"][stage]["configurations"]
                if row["configuration_sha256"] == configuration)
            self.assertGreater(row["accepted_units"], 0)
            self.assertEqual(row["root_slots"], 200)
            self.assertEqual(row["source_seeds"], 32)
        self.assertEqual(result["launches"]["search"], 1)  # Same original owner reused in A2.
        self.assertGreater(result["launches"]["reused_selectors"], 0)
        self.assertEqual(result["launches"]["continuation"], 1)
        roots = []
        for seed in helper.ledger.contract["panels"]["exploration"]["seeds"]:
            attempt = helper.directory / "source-attempts" / str(seed)
            source = json.loads((attempt / "source.json").read_bytes())
            closed = json.loads((attempt / "bridge-close.json").read_bytes())
            self.assertTrue(source["source_terminal_complete"])
            self.assertEqual(source["verified_raw_decisions"], 1)
            self.assertTrue(closed["normal_close"]); self.assertEqual(closed["exit_code"], 0)
            roots.extend(source["roots"])
        self.assertEqual([row["root_id"] for row in roots], [root])
        self.assertGreaterEqual(sum(roots[0]["public_record"]["current_legal_action_mask"]), 2)
        journal = helper.directory / "attempt-cost-journal"
        census = read_attempt_journal_cost(AttemptJournalHandle(str(journal), cost._sha(journal / "metadata.json")),
            expected_outer_terminal=outer)
        self.assertTrue(census["declared_supervised_attempt_census_complete"])
        self.assertTrue(census["all_journal_writers_normal_closed"])
        continuations = []; selectors = []
        for attempt in census["attempts"]:
            path = Path(attempt["directory"])
            if not (path / "terminal.json").exists(): continue
            recorded = json.loads((path / "terminal.json").read_bytes())
            if recorded["schema"] == "pokezero.search-over-raw.outcome-run-supervision.v1":
                continuations.append(recorded)
                self.assertEqual(recorded["status"], "OUTCOME_RUN_COMPLETE")
                self.assertEqual(recorded["result"]["root_binding"]["root_id"], root)
                bank = recorded["result"]["bank"]
                self.assertIn(len(bank["cells"]), (8, 16))  # Original alias law may share raw's action.
                self.assertEqual(len({cell["battle_token"] for cell in bank["cells"]}), len(bank["cells"]))
                self.assertTrue(all(row["status"] == "COMPLETE" for row in recorded["result"]["outcomes"]["outcomes"]))
                closed = json.loads((path / "bridge-close.json").read_bytes())
                self.assertTrue(closed["normal_close"]); self.assertEqual(closed["exit_code"], 0)
            elif recorded["schema"] == "pokezero.search-over-raw.actor-annotated-native-timed-run-supervision.v1":
                selectors.append(recorded)
                if compact_native_selector:
                    from pokezero.mcts_eval.search_over_raw_native_atomic_compact import SCHEMA
                    consumed = recorded["result"]["evidence"]["consumed_world_inputs"]
                    self.assertEqual(consumed["native_atomic_encoding"], SCHEMA)
                    self.assertTrue(consumed["calls"])
                    self.assertTrue(all(call["native_report"]["atomic_work"]["schema"] == SCHEMA
                        for call in consumed["calls"]))
                    terminal_bytes = (path/"terminal.json").stat().st_size
                    self.assertLessEqual(terminal_bytes, 16_777_216)
                    print("COMPACT_COLLECTOR_MEASUREMENT:"+json.dumps(dict(terminal_bytes=terminal_bytes,
                        accepted_units=sum(call["completed_iterations"] for call in consumed["calls"]),
                        source_count=32, observed_roots=1, missing_roots=199), sort_keys=True))
        self.assertEqual(len(continuations), 1)
        self.assertEqual(len(selectors), 1)
        self.assertFalse(measured["complete_workflow_cost"])
        self.assertFalse(measured["resource_cost_complete"])
        # Original cost reader must join the nonempty worker result, not accept
        # a caller-replaced failed-terminal exit as a normally completed run.
        path = Path(next(attempt["directory"] for attempt in census["attempts"]
            if (Path(attempt["directory"]) / "terminal.json").exists()
            and json.loads((Path(attempt["directory"]) / "terminal.json").read_bytes()).get("schema")
                == "pokezero.search-over-raw.actor-annotated-native-timed-run-supervision.v1")) / "terminal.json"
        original = path.read_bytes(); changed = json.loads(original); changed["worker_exit_code"] = 7
        path.write_text(json.dumps(changed))
        self.assertFalse(cost.read_collector_cost(directory)["invocation_cpu_wall_complete"])
        path.write_bytes(original)
        self.assertTrue(cost.read_collector_cost(directory)["invocation_cpu_wall_complete"])

    def test_actual_normal_cli_cpu_includes_reaped_children_other_thread_and_output_without_double_addition(self):
        directory, ledger = self.claim()
        terminal = cost._run_claimed_invocation(directory)
        self.assertEqual(terminal["actual_exit_code"], 0, terminal)
        outer = json.loads((ledger / "exploration-entry-run/terminal.json").read_bytes())
        self.assertGreater(outer["cpu_seconds_including_reaped_children"], .12)
        self.assertGreaterEqual(terminal["actual_cpu_seconds_including_reaped_children"],
            outer["cpu_seconds_including_reaped_children"] + .09)
        self.assertEqual(json.loads(terminal["stdout"])["status"], "STAGED_EXPLORATION_RUN_COMPLETE")
        before = {p.name: p.read_bytes() for p in directory.iterdir()}
        with patch("os.wait4") as wait, patch.object(cost, "owned_process_identity") as identity:
            readback = cost.read_collector_cost(directory)
        wait.assert_not_called(); identity.assert_not_called()
        self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir()})
        self.assertEqual(readback["actual_cli_cpu_seconds_including_reaped_children"],
            terminal["actual_cpu_seconds_including_reaped_children"])
        self.assertFalse(readback["nested_counters_added_again"])
        self.assertFalse(readback["invocation_cpu_wall_complete"])
        self.assertIsNone(readback["total_invocation_cpu_seconds"])

    def test_timeout_before_entry_reaps_only_original_cli_no_auditor_invented(self):
        directory, ledger = self.claim("before_entry", 1.)
        terminal = cost._run_claimed_invocation(directory)
        self.assertTrue((directory / "child-ready.json").exists())
        self.assertFalse((ledger / "exploration-entry-run").exists())
        self.assertIs(terminal["watchdog_reached"], True)
        self.assertEqual(terminal["actual_exit_code"], -signal.SIGKILL)
        self.assertEqual(terminal["cleanup_errors"], [])
        self.assertEqual(terminal["verified_cleanup_groups"], [])
        self.assertFalse(cost.read_collector_cost(directory)["invocation_cpu_wall_complete"])

    def test_timeout_before_registration_keeps_original_worker_inert_and_cleans_exact_launch(self):
        directory, ledger = self.claim("before_registration", 1.5)
        terminal = cost._run_claimed_invocation(directory)
        self.assertTrue((directory / "authored-preheader.json").exists(), terminal)
        self.assertFalse((directory / "auditor-registration.json").exists())
        original = json.loads((ledger / "exploration-entry-run/launch.json").read_bytes())
        self.assertIn(original["pid"], terminal["verified_cleanup_groups"])
        self.assertFalse((ledger / "exploration-entry-run/worker-result.json").exists())
        self.assertEqual(terminal["cleanup_errors"], [])
        self.assertFalse(live_owned_groups(set(terminal["verified_cleanup_groups"])))

    def test_timeout_registered_stopped_ordinary_descendant_is_killed_before_original_auditor(self):
        directory, ledger = self.claim("stopped", 2.)
        terminal = cost._run_claimed_invocation(directory)
        stopped = json.loads((ledger / "exploration-entry-run/authored-stopped.json").read_bytes())
        self.assertIn(stopped["pid"], terminal["verified_cleanup_groups"])
        self.assertEqual(terminal["actual_exit_code"], -signal.SIGKILL)
        self.assertEqual(terminal["cleanup_errors"], [])
        self.assertFalse(live_owned_groups(set(terminal["verified_cleanup_groups"])))
        self.assertFalse(cost.read_collector_cost(directory)["invocation_cpu_wall_complete"])

    def test_timeout_registered_nested_generations_keep_original_depth_and_all_groups(self):
        directory, ledger = self.claim("nested", 3.)
        terminal = cost._run_claimed_invocation(directory)
        original = json.loads((ledger / "exploration-entry-run/launch.json").read_bytes())
        child = json.loads((ledger / "nested/child/nested-detach.json").read_bytes())
        self.assertEqual(child["depth"], 1)
        self.assertIn(original["pid"], terminal["verified_cleanup_groups"])
        self.assertGreaterEqual(len(terminal["verified_cleanup_groups"]), 3)
        self.assertEqual(terminal["cleanup_errors"], [])
        self.assertFalse(live_owned_groups(set(terminal["verified_cleanup_groups"])))

    def test_registration_refusal_is_failure_without_execution_or_retry(self):
        directory, ledger = self.claim("registration_failure")
        terminal = cost._run_claimed_invocation(directory)
        self.assertEqual(terminal["actual_exit_code"], 1)
        root = json.loads((ledger / "exploration-entry-run/terminal.json").read_bytes())
        self.assertEqual(root["error_type"], "RuntimeError")
        self.assertIsNone(root["result"])
        self.assertFalse((ledger / "exploration-entry-run/worker-result.json").exists())
        with patch.object(cost.subprocess, "Popen") as spawn:
            with self.assertRaises(FileExistsError): cost._run_claimed_invocation(directory)
        spawn.assert_not_called()
        self.assertFalse(cost.read_collector_cost(directory)["invocation_cpu_wall_complete"])

    def test_terminal_cli_cleanup_never_inspects_or_signals_historical_auditor(self):
        directory, _ = self.claim()
        claim = cost._check_claim(directory)
        with patch("os.waitid", return_value=SimpleNamespace(si_code=os.CLD_EXITED)), \
                patch.object(cost, "_reap", return_value=(0, "actual-usage")), \
                patch.object(cost, "freeze_launcher") as freeze, patch.object(cost, "_original_root") as root, \
                patch.object(cost, "signal_owned_group") as kill:
            result = cost._stop_cli(claim, SimpleNamespace(pid=123456), "unused historical birth")
        self.assertEqual(result, ((0, "actual-usage"), [], []))
        freeze.assert_not_called(); root.assert_not_called(); kill.assert_not_called()

    def test_missing_ready_proof_never_authorizes_saved_auditor_signalling(self):
        directory, _ = self.claim()
        with patch("os.waitid", return_value=None), patch.object(cost, "freeze_launcher", return_value=123456), \
                patch.object(cost, "_reap", return_value=(-signal.SIGKILL, "actual-usage")), \
                patch.object(cost, "_original_root") as root, patch.object(cost, "signal_owned_group") as kill:
            result = cost._stop_cli(cost._check_claim(directory), SimpleNamespace(pid=123456), "birth")
        root.assert_not_called()
        self.assertEqual(result[1], [])
        self.assertTrue(result[2])
        kill.assert_called_once_with(123456, signal.SIGKILL)

    def test_lost_wait_ownership_is_retained_unknown_without_repoll_or_signalling(self):
        directory, _ = self.claim()
        owner_birth = cost._check_claim(directory)["owner_birth"]
        child = SimpleNamespace(pid=987654321, stdin=io.BytesIO(), returncode=None)
        def identity(pid):
            # Popen is mocked for the lost-wait boundary; do not let a real
            # parent ps query consume that same mocked process factory.
            return "authored birth" if pid == child.pid else owner_birth
        with patch.object(cost, "owned_process_identity", side_effect=identity), \
                patch.object(cost.subprocess, "Popen", return_value=child), \
                patch.object(cost, "_reap", side_effect=ChildProcessError("authored lost wait ownership")) as reap, \
                patch.object(cost, "_stop_cli") as stop:
            terminal = cost._run_claimed_invocation(directory)
        self.addCleanup(cost._UNRESOLVED_CHILD_HANDLES.pop, str(directory), None)
        self.assertEqual(reap.call_count, 1)
        stop.assert_not_called()
        self.assertIs(terminal["wait_ownership_lost"], True)
        self.assertIsNone(terminal["actual_exit_code"])
        self.assertIsNone(terminal["actual_cpu_seconds_including_reaped_children"])
        self.assertFalse(cost.read_collector_cost(directory)["whole_cli_lifetime_wait4_observed"])

    def test_cost_child_close_failure_cannot_replace_original_admission_error(self):
        directory, ledger = self.claim()
        claim = dict(cost._check_claim(directory), synthetic_process_fixture=False,
            owner_pid=os.getppid(), owner_birth="authored birth")
        launch = dict(pid=os.getpid(), group=os.getpid(), birth_identity="authored birth", claim_sha256="a"*64)
        original = cost._write_new
        def write(path, value):
            if path.name == "child-close.json":
                raise OSError("authored supplementary measurement failure")
            return original(path, value)
        header = json.dumps(dict(pid=os.getpid(), claim_sha256="a"*64)).encode()
        with patch.object(cost, "_check_claim", return_value=claim), patch.object(cost, "_launch", return_value=launch), \
                patch.object(cost, "owned_process_identity", return_value="authored birth"), \
                patch.object(cost, "_sha", return_value="a"*64), patch("os.getpgrp", return_value=os.getpid()), \
                patch.object(cost.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(header))), \
                patch.object(cost, "_write_new", side_effect=write):
            with self.assertRaisesRegex(RuntimeError, "original admission failure") as stopped:
                with cost.collector_cost_child(directory, ledger, claim["spec_path"]):
                    raise RuntimeError("original admission failure")
        self.assertIn("cost-child close evidence failed: OSError", stopped.exception.__notes__)

    def test_changed_inputs_and_terminal_arithmetic_refuse_readback_without_queries(self):
        directory, _ = self.claim()
        terminal = cost._run_claimed_invocation(directory)
        changed = dict(terminal); changed["actual_cpu_seconds_including_reaped_children"] += 1
        (directory / "terminal.json").write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "actual wait4"): cost.read_collector_cost(directory)
        (directory / "terminal.json").write_text(json.dumps(terminal))
        claim = cost._check_claim(directory); Path(claim["spec_path"]).write_text('{"authored":false}')
        with self.assertRaisesRegex(ValueError, "sources changed"): cost.read_collector_cost(directory)

    def test_terminal_exact_types_scope_and_usage_presence_cannot_be_reclassified(self):
        directory, _ = self.claim()
        terminal = cost._run_claimed_invocation(directory)
        for key, value in (("wait_ownership_lost", 0), ("watchdog_reached", 0), ("error_type", False),
                ("cleanup_errors", [False]), ("cleanup_errors", [""]), ("cpu_scope", "all machine CPU"),
                ("wall_scope", "sum of nested wall exposures"), ("maxrss_scope", "exact aggregate physical peak"),
                ("maxrss_units", "seconds"), ("maxrss", None), ("cli_pid", None), ("actual_exit_code", True),
                ("stdout", {}), ("verified_cleanup_groups", [True]), ("resource_cost_complete", 0)):
            with self.subTest(key=key, value=value):
                changed = dict(terminal); changed[key] = value
                (directory / "terminal.json").write_text(json.dumps(changed))
                with self.assertRaises(ValueError): cost.read_collector_cost(directory)
        changed = dict(terminal, extra_complete_claim=True)
        (directory / "terminal.json").write_text(json.dumps(changed))
        with self.assertRaises(ValueError): cost.read_collector_cost(directory)
        (directory / "terminal.json").write_text(json.dumps(terminal))
        self.assertFalse(cost.read_collector_cost(directory)["collector_startup_other_threads_final_output_included"])

    def test_final_readback_rechecks_external_input_bytes_after_evidence_inventory(self):
        directory, _ = self.claim()
        cost._run_claimed_invocation(directory)
        spec = Path(cost._check_claim(directory)["spec_path"])
        original = cost._sha; terminal_reads = 0
        def hash_and_drift(path):
            nonlocal terminal_reads
            value = original(path)
            if Path(path).name == "terminal.json":
                terminal_reads += 1
                if terminal_reads == 2:
                    spec.write_text('{"authored":false}')
            return value
        with patch.object(cost, "_sha", side_effect=hash_and_drift):
            with self.assertRaisesRegex(ValueError, "sources changed"):
                cost.read_collector_cost(directory)

    def test_old_cli_check_mode_remains_read_only_and_cost_check_override_refuses(self):
        import collect_search_over_raw_reconciled_exploration as cli
        with patch("pokezero.mcts_eval.search_over_raw_exploration_entry.prepare_exploration_entry") as prepare, \
                patch.object(cost.subprocess, "Popen") as spawn:
            self.assertEqual(cli.main(["--ledger", str(self.root), "--entry-spec", str(self.root / "unused.json"), "--check"]), 0)
            prepare.assert_called_once(); spawn.assert_not_called()
        with self.assertRaises(SystemExit) as stopped:
            cli.main(["--ledger", str(self.root), "--entry-spec", str(self.root / "unused.json"),
                "--check", "--cost-directory", str(self.root / "never-created")])
        self.assertEqual(stopped.exception.code, 2)
        self.assertFalse((self.root / "never-created").exists())

    def test_reader_closed_join_arithmetic_and_pending_query_refusal_with_explicit_mocked_census(self):
        # Covers reader control/arithmetic only. This is NOT a positive
        # admitted collector integration or production-cost qualification.
        from dataclasses import asdict
        from pokezero.mcts_eval.search_over_raw_exploration_entry import ExplorationAuditorJob
        directory, ledger = self.claim()
        terminal = cost._run_claimed_invocation(directory)
        original = cost._check_claim(directory)
        fake_job = ExplorationAuditorJob(str(ledger), original["spec_path"], "a"*64, "b"*64, "c"*64)
        claim = dict(original, synthetic_process_fixture=False, entry_job=asdict(fake_job))
        root = cost._original_root(original, terminal["cli_pid"])
        journal = ledger / "attempt-cost-journal"; journal.mkdir(); _write_new(journal / "metadata.json", dict(authored=True))
        original_terminal = cost._json(Path(root["directory"]) / "terminal.json")
        original_worker = cost._json(Path(root["directory"]) / "worker-result.json")
        from pokezero.mcts_eval.search_over_raw_resources import read_completed_process_cost
        original_cost = read_completed_process_cost(original_terminal, original_worker, work_phase="selection")
        census = dict(declared_supervised_attempt_census_complete=True, all_journal_writers_normal_closed=True,
            attempts=[dict(directory=root["directory"], parent_directory=None, measurement_window_cpu_wall_complete=True,
                phase_costs=original_cost["phase_costs"], elapsed_job_wall_seconds=original_cost["elapsed_wall_seconds"],
                job_cpu_seconds_including_reaped_children=original_cost["job_cpu_seconds_including_reaped_children"])])
        observer = dict(observer_window=dict(own_query_usage_unknown=0), publication_usage_unknown=False,
            known_disjoint_cpu_subtotal_seconds=999999.)
        close_path = directory / "child-close.json"; close_bytes = close_path.read_bytes()
        def read():
            with patch.object(cost, "_check_claim", return_value=claim), \
                    patch.object(cost, "_original_root", return_value=root), \
                    patch("pokezero.mcts_eval.search_over_raw_exploration_entry.read_exploration_entry"), \
                    patch("pokezero.mcts_eval.search_over_raw_exploration_entry.read_entry_observer_cost", return_value=observer), \
                    patch("pokezero.mcts_eval.search_over_raw_attempt_journal.read_attempt_journal_cost", return_value=census):
                return cost.read_collector_cost(directory)
        accepted = read()
        self.assertTrue(accepted["invocation_cpu_wall_complete"], accepted["incomplete_components"])
        self.assertEqual(accepted["total_invocation_cpu_seconds"], terminal["actual_cpu_seconds_including_reaped_children"])
        self.assertFalse(accepted["complete_workflow_cost"])
        self.assertFalse(accepted["resource_cost_complete"])
        breakdown = accepted["attempt_phase_breakdown"]
        self.assertEqual(len(breakdown["attempts"]), 1)
        self.assertEqual(breakdown["attempts"][0]["phase_wall_seconds"], original_cost["phase_costs"])
        self.assertIsNone(breakdown["attempts"][0]["phase_cpu_seconds"])
        self.assertIsNone(breakdown["phase_wall_totals"])
        self.assertIsNone(breakdown["phase_cpu_totals"])
        self.assertFalse(breakdown["nested_counters_added_to_invocation"])
        for key, value in (("pending_queries", 1), ("pending_queries", False), ("lost_wait_ownership_queries", 1),
                ("pending_legacy_samplers", 1)):
            with self.subTest(key=key, value=value):
                close = json.loads(close_bytes); close["query_census"][key] = value
                close_path.write_text(json.dumps(close))
                refused = read()
                self.assertFalse(refused["invocation_cpu_wall_complete"])
                self.assertIsNone(refused["attempt_phase_breakdown"])
        close_path.write_bytes(close_bytes)
        observer["observer_window"]["own_query_usage_unknown"] = 1
        self.assertFalse(read()["invocation_cpu_wall_complete"])
        observer["observer_window"]["own_query_usage_unknown"] = 0
        census["all_journal_writers_normal_closed"] = False
        self.assertFalse(read()["invocation_cpu_wall_complete"])
        census["all_journal_writers_normal_closed"] = True
        census["attempts"][0]["measurement_window_cpu_wall_complete"] = False
        self.assertFalse(read()["invocation_cpu_wall_complete"])
        census["attempts"][0]["measurement_window_cpu_wall_complete"] = True
        original_phases = census["attempts"][0]["phase_costs"]
        census["attempts"][0]["phase_costs"] = None
        refused = read()
        self.assertFalse(refused["invocation_cpu_wall_complete"])
        self.assertIsNone(refused["attempt_phase_breakdown"])
        self.assertEqual(refused["incomplete_components"],
            ["original collector join refused at original_attempt_phase_breakdown: ValueError"])
        census["attempts"][0]["phase_costs"] = original_phases
        self.assertTrue(read()["invocation_cpu_wall_complete"])
        with patch.object(cost, "_check_claim", return_value=claim), \
                patch.object(cost, "_original_root", side_effect=ValueError("PRIVATE_SENTINEL")):
            refused = cost.read_collector_cost(directory)
        self.assertEqual(refused["incomplete_components"],
            ["original collector join refused at original_entry_binding: ValueError"])
        self.assertFalse(refused["invocation_cpu_wall_complete"])
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(refused))

    def test_original_nested_journal_phase_breakdown_preserves_overlapping_windows_without_addition(self):
        import test_search_over_raw_attempt_journal as journal_fixture
        from pokezero.mcts_eval.search_over_raw import digest
        from pokezero.mcts_eval.search_over_raw_attempt_journal import read_attempt_journal_cost
        helper = journal_fixture.AttemptJournalTests(); helper.setUp(); self.addCleanup(helper.doCleanups)
        handle, terminal = helper.launch()
        self.assertEqual(terminal["supervisor_exit_code"], 0)
        census = read_attempt_journal_cost(handle, expected_outer_terminal=terminal)
        original = digest(census)
        before = {str(p): p.read_bytes() for p in helper.root.rglob("*") if p.is_file()}
        with patch("subprocess.Popen") as spawn, patch("os.wait4") as wait:
            breakdown = cost._attempt_phase_breakdown(census)
        spawn.assert_not_called(); wait.assert_not_called()
        self.assertEqual(digest(census), original)
        self.assertEqual(before, {str(p): p.read_bytes() for p in helper.root.rglob("*") if p.is_file()})
        self.assertEqual(len(breakdown["attempts"]), 2)
        by_directory = {row["directory"]: row for row in breakdown["attempts"]}
        outer = by_directory[str(helper.root / "outer")]
        child = by_directory[str(helper.root / "child")]
        self.assertIsNone(outer["parent_directory"])
        self.assertEqual(child["parent_directory"], outer["directory"])
        self.assertEqual(outer["phase_wall_seconds"], terminal["phase_costs"])
        self.assertEqual(outer["job_cpu_seconds_including_reaped_children"],
            census["known_nonoverlapping_job_cpu_seconds"])
        self.assertTrue(breakdown["nested_windows_may_overlap"])
        self.assertIsNone(breakdown["phase_wall_totals"])
        self.assertIsNone(breakdown["phase_cpu_totals"])
        self.assertFalse(breakdown["phase_cpu_measured"])
        self.assertFalse(breakdown["nested_counters_added_to_invocation"])
        for row in breakdown["attempts"]:
            self.assertIsNone(row["phase_cpu_seconds"])
            self.assertLessEqual(sum(row["phase_wall_seconds"].values()), row["elapsed_job_wall_seconds"])
        outer["phase_wall_seconds"]["startup_wall_seconds"] = -1
        self.assertEqual(digest(census), original)  # Returned rows do not alias original evidence.

    def test_incomplete_or_failed_attempts_cannot_supply_phase_breakdown(self):
        import test_search_over_raw_attempt_journal as journal_fixture
        from pokezero.mcts_eval.search_over_raw_attempt_journal import read_attempt_journal_cost
        helper = journal_fixture.AttemptJournalTests(); helper.setUp(); self.addCleanup(helper.doCleanups)
        handle, terminal = helper.launch("prelaunch_fail")
        census = read_attempt_journal_cost(handle, expected_outer_terminal=terminal)
        self.assertTrue(census["declared_supervised_attempt_census_complete"])
        self.assertTrue(any(row.get("phase_costs") is None for row in census["attempts"]))
        with self.assertRaisesRegex(ValueError, "per-attempt phase"):
            cost._attempt_phase_breakdown(census)
        for flag in ("declared_supervised_attempt_census_complete", "all_journal_writers_normal_closed"):
            changed = dict(census, **{flag: False})
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "closed original"):
                cost._attempt_phase_breakdown(changed)

    def test_phase_projection_rejects_unknown_nonfinite_boolean_or_unreconciled_costs(self):
        from copy import deepcopy
        phases = dict(process_bootstrap_wall_seconds=1., startup_wall_seconds=1., selection_wall_seconds=3.,
            cleanup_wall_seconds=1., receipt_reap_and_verification_wall_seconds=1.)
        row = dict(directory=str(self.root / "outer"), parent_directory=None,
            measurement_window_cpu_wall_complete=True, phase_costs=phases,
            elapsed_job_wall_seconds=8., job_cpu_seconds_including_reaped_children=5.)
        census = dict(declared_supervised_attempt_census_complete=True, all_journal_writers_normal_closed=True,
            attempts=[row])
        self.assertEqual(cost._attempt_phase_breakdown(census)["attempts"][0]["phase_wall_seconds"], phases)
        for value in (True, -1., float("nan"), float("inf"), None):
            changed = deepcopy(census); changed["attempts"][0]["phase_costs"]["startup_wall_seconds"] = value
            with self.subTest(value=value), self.assertRaises(ValueError): cost._attempt_phase_breakdown(changed)
        for mutate in (lambda r: r["phase_costs"].pop("cleanup_wall_seconds"),
                lambda r: r["phase_costs"].update(continuations_wall_seconds=1.),
                lambda r: r["phase_costs"].update(unobserved_wall_seconds=1.),
                lambda r: r.update(elapsed_job_wall_seconds=6.),
                lambda r: r.update(job_cpu_seconds_including_reaped_children=None),
                lambda r: r.update(measurement_window_cpu_wall_complete=False)):
            changed = deepcopy(census); mutate(changed["attempts"][0])
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): cost._attempt_phase_breakdown(changed)

    def test_actual_canonical_cli_refuses_unadmitted_ledger_and_retains_cpu_without_scientific_launch(self):
        from test_search_over_raw_exploration_entry import ExplorationEntryTests
        helper = ExplorationEntryTests(); helper.setUp(); self.addCleanup(helper.doCleanups)
        helper.make(admitted=False)
        from pokezero.mcts_eval.search_over_raw_exploration_entry import _evidence_inventory
        before = _evidence_inventory(helper.directory)
        directory = self.root / "actual-cost"
        terminal = cost.run_collector_cost(helper.directory, helper.spec_path, directory=directory, watchdog_seconds=10.)
        self.assertEqual(terminal["actual_exit_code"], 1, terminal)
        self.assertGreater(terminal["actual_cpu_seconds_including_reaped_children"], 0)
        self.assertEqual(before, _evidence_inventory(helper.directory))
        close = json.loads((directory / "child-close.json").read_bytes())
        self.assertEqual(close["error_type"], "ValueError")
        self.assertFalse((directory / "auditor-registration.json").exists())
        self.assertFalse(cost.read_collector_cost(directory)["invocation_cpu_wall_complete"])

    def test_actual_private_child_waits_for_delayed_launch_publication_before_original_admission(self):
        from test_search_over_raw_exploration_entry import ExplorationEntryTests
        helper = ExplorationEntryTests(); helper.setUp(); self.addCleanup(helper.doCleanups)
        helper.make(admitted=False)
        directory = self.root / "delayed-cost"
        actual = cost._write_new
        def delayed(path, value):
            if path.name == "launch.json":
                time.sleep(1.)  # Force parent publication behind child import.
            return actual(path, value)
        with patch.object(cost, "_write_new", side_effect=delayed):
            terminal = cost.run_collector_cost(helper.directory, helper.spec_path, directory=directory, watchdog_seconds=10.)
        self.assertEqual(terminal["actual_exit_code"], 1, terminal)
        self.assertTrue((directory / "child-ready.json").is_file())
        self.assertEqual(json.loads((directory / "child-close.json").read_bytes())["error_type"], "ValueError")
        self.assertFalse((helper.directory / "exploration-entry-run").exists())


if __name__ == "__main__":
    if sys.argv[1:2] == ["fixture"]:
        raise SystemExit(fixture(sys.argv[2], sys.argv[3]))
    unittest.main()
