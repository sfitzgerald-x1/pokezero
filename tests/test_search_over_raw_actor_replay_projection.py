"""Real encoder on authored actor/replay rows, not live ownership or science."""
from copy import deepcopy
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.local_showdown import LocalShowdownEnv
from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_actor_replay_inputs import SCHEMA as INPUT_SCHEMA
from pokezero.mcts_eval.search_over_raw_actor_replay_projection import (
    project_original_actor_replay, projection_identity)
from pokezero.mcts_eval.search_over_raw_source_actor_history import retained_program_identity
from pokezero.mcts_eval.search_over_raw_world_inputs import _state_payload
import test_search_over_raw_actor_projection as actor_fixture
import test_search_over_raw_source_actor_history as source_fixture
import test_search_over_raw_native_chain as chain_fixture


def coherent_actor_root():
    """Predeclared authored team/order/requests; never an actor snapshot."""
    root, _, _ = chain_fixture.construction_fixture.map_fixture()
    # Match the captured native own order rather than the old mapping test's
    # deliberately reversed party. Every actual constructor still validates.
    root["battle"]["sides"][0]["pokemon"].reverse()
    requests = {}
    for side in root["battle"]["sides"]:
        mons = []
        for index, mon in enumerate(side["pokemon"]):
            mon["position"], mon["isActive"] = index, index == 0
            own = mon["set"]
            mon["moveSlots"] = [dict(id=move, pp=15, disabled=False) for move in own["moves"]]
            mon.update(hpType="Ice", hpPower=70)
            mons.append(dict(ident=side["id"]+": "+own["species"].capitalize(),
                details=own["species"].capitalize()+", L"+str(own["level"]),
                active=index == 0, condition="100/100", moves=list(own["moves"]),
                baseAbility=own["ability"], ability=own["ability"], item=own["item"],
                stats=dict(atk=130,def_=140,spa=150,spd=160,spe=170)))
            mons[-1]["stats"]["def"] = mons[-1]["stats"].pop("def_")
        requests[side["id"]] = dict(side=dict(id=side["id"],pokemon=mons),
            active=[dict(moves=[dict(id=move,move=move,pp=15,maxpp=16,disabled=False)
                for move in side["pokemon"][0]["set"]["moves"]])])
    root["battle"]["log"] = []
    root["boundaryRequests"] = requests
    return root


class CoherentActorSourceEnv(chain_fixture.source_fixture.AuthoredSourceEnv):
    def observe(self, seat):
        # Both this actual source seam and projection use the same authored
        # static dex. Actual encoder state is not replaced by authored tensors.
        return LocalShowdownEnv.observe(self, seat)

    def public_materialization_state(self, seat):
        value = LocalShowdownEnv.public_materialization_state(self, seat)
        # Declared authored catalog witness only; original real replay/request,
        # PP chronology and annotations remain intact. No physical catalog or
        # production runtime is qualified by this disposable fixture.
        value.belief_engine.set_source = SimpleNamespace(metadata=SimpleNamespace(source_hash="a"*64))
        return value


class CoherentActorSourceFactory:
    synthetic_process_fixture = True
    def __new__(cls, config, checkpoint):
        env = CoherentActorSourceEnv(config)
        return env, {seat:chain_fixture.source_fixture.AuthoredPolicy(checkpoint,env.mode) for seat in ("p1","p2")}


def coherent_native_extract(state, root):
    value = json.loads(chain_fixture.consumed_fixture.OwnedAncestryRootJoinTests.extractor(state,root))
    for seat, move in (("p1","earthquake"),("p2","bodyslam")):
        side = value[seat]; side["active_moves"][0]["engine_id"] = move
        for key in ("engine_options","search_arms","effective_legal_choices"):
            side[key] = [dict(kind="move",slot=0,engine_id=move)]
    return json.dumps(value,sort_keys=True,separators=(",",":"))


@dataclass(frozen=True)
class AuthoredCoherentNativeJob(chain_fixture.AuthoredRetainedNativeJob):
    def _run(self, ownership, phase):
        selected = super()._run(ownership,phase)
        # Generate the authored actor actions inside the original worker,
        # before selected/capture receipts exist. Exact ancestry membership,
        # edge hashes and command resolution are still independently checked.
        for index, (row, call) in enumerate(zip(selected["evidence"]["native_searched_ancestry"],
                selected["evidence"]["consumed_world_inputs"]["calls"])):
            value = json.loads(row["export_utf8"])
            for leaf in value["evaluations"]:
                for step in leaf["capture"]["steps"]:
                    step["side_one_choice"]["engine_id"] = "earthquake"
                    step["side_two_choice"]["engine_id"] = "bodyslam"
                    chain_fixture.edge_fixture.add_edge(step)
            selected["evidence"]["native_searched_ancestry"][index] = \
                chain_fixture.ancestry_fixture.row_from_payload(value,call,index)
        return selected


class ActorReplayProjectionTests(unittest.TestCase):
    def setUp(self):
        self.h = source_fixture.SourceActorHistoryTests()
        self.h.setUp(); self.addCleanup(self.h.doCleanups)
        self.config = self.h.config

    def source(self, *, mismatch=False):
        self.h.setup_capture(exact_opponent=mismatch)
        self.h.opening(private_opponent=mismatch)
        self.h.observe_and_select(); self.h.observe_and_select()
        cutoff = self.h.boundary(); self.h.capture.seal()
        return self.h.read(cutoff)

    def inputs(self, *, actor="p1", requests=None, suffixes=None, requested=None, kinds=None):
        requests = [actor_fixture.own_request(pp=55), actor_fixture.own_request(hp="190/250", pp=54)] \
            if requests is None else requests
        suffixes = [[], ["|move|p2a: Snorlax|Tackle|p1a: Charizard", "|-damage|p1a: Charizard|190/250", "|turn|2"]] \
            if suffixes is None else suffixes
        requested = [True]*len(requests) if requested is None else requested
        kinds = ["actionable"]*len(requests) if kinds is None else kinds
        rows = [dict(snapshot_index=i, observed_generation=i, actor_requested=active, boundary_kind=kind,
            visible_suffix=deepcopy(lines), visible_suffix_sha256=digest(lines),
            own_request=deepcopy(request), own_request_sha256=digest(request))
            for i, (request, lines, active, kind) in enumerate(zip(requests, suffixes, requested, kinds))]
        return dict(schema=INPUT_SCHEMA, actor=actor, rows=rows,
            materialized_shell_prefix_used=False, snapshot_transport_consistent=True)

    def project(self, source, inputs):
        return project_original_actor_replay(source, inputs, config=self.config,
            expected_source_identity=retained_program_identity(source) if source is not None else None,
            expected_replay_sha256=digest(inputs))

    def test_original_calls_and_actual_requests_match_independent_normal_encoder(self):
        source, inputs = self.source(), self.inputs()
        source_before, inputs_before = retained_program_identity(source), deepcopy(inputs)
        with patch("subprocess.Popen") as launch:
            result = self.project(source, inputs)
        launch.assert_not_called()
        program = deepcopy(source.program)
        for i, row in enumerate(inputs["rows"]):
            program["events"].extend([dict(kind="protocol", lines=row["visible_suffix"]),
                dict(kind="own_request", request=row["own_request"]),
                dict(kind="observe", coordinate=2+i)])
        helper = actor_fixture.ActorProjectionTests()
        helper.config = self.config
        expected, _ = helper.ordinary(program)
        self.assertEqual([digest(_state_payload(v)) for v in result.observations],
            [digest(_state_payload(v)) for v in expected[2:]])
        self.assertEqual(retained_program_identity(source), source_before)
        self.assertEqual(inputs, inputs_before)
        self.assertEqual(result.actor_program(1),program)
        self.assertLess(result.program_cutoffs[0],result.program_cutoffs[1])
        self.assertEqual(result.receipt["encoded_boundaries"], 2)
        self.assertEqual(result.materialization_states[1].self_request, inputs["rows"][1]["own_request"])
        self.assertNotIn("NEVER_IMPORT_OPPONENT_HISTORY", str(_state_payload(result)))
        for key in ("source_process_join_validated", "constructor_process_join_validated",
                "replay_process_join_validated", "native_evaluation_input_parity",
                "complete_paired_continuation_qualified", "independent_terminal_labels",
                "runtime_qualified", "collection_authorized"):
            self.assertFalse(result.receipt[key])

    def test_root_pp_difference_is_retained_not_an_identity_gate(self):
        result = self.project(self.source(), self.inputs())
        self.assertFalse(result.receipt["pp_equality_used_as_identity_gate"])
        self.assertEqual(result.receipt["root_pp_differences"], [dict(slot=0, source_pp=56,
            actual_pp=55, source_maxpp=56, actual_maxpp=56)])
        self.assertEqual(result.materialization_states[0].self_request["active"][0]["moves"][0]["pp"], 55)

    def test_wait_and_terminal_roster_never_encode_stale_requests(self):
        source = self.source(); own = actor_fixture.own_request()
        inputs = self.inputs(requests=[own, own, own], suffixes=[[], [], ["|win|PokeZero p1"]],
            requested=[True, False, False], kinds=["actionable", "waiting", "terminal"])
        result = self.project(source, inputs)
        self.assertIsNotNone(result.observations[0])
        self.assertEqual(result.observations[1:], (None, None))
        self.assertEqual(result.receipt["coordinates"], [0, 1, 2])
        self.assertEqual(result.receipt["unavailable_reasons"], [None, "ACTOR_NOT_REQUESTED", "TERMINAL"])
        self.assertEqual(result.program_cutoffs[1:],(None,None))
        self.assertIsNone(result.actor_program(1))

    def test_missing_actionable_request_cannot_be_repaired_by_a_later_request(self):
        own = actor_fixture.own_request()
        inputs = self.inputs(requests=[own, None, own], suffixes=[[], [], []])
        result = self.project(self.source(), inputs)
        self.assertEqual(result.observations[1:], (None, None))
        self.assertEqual(result.receipt["unavailable_reasons"], [None, "ACTUAL_OWN_REQUEST_UNAVAILABLE",
            "EARLIER_ACTIONABLE_OWN_REQUEST_UNAVAILABLE"])

    def test_absent_source_or_sampled_opponent_preserves_every_uncertain_coordinate(self):
        cases = [(None, self.inputs(), "ORIGINAL_ACTOR_HISTORY_ABSENT"),
            (self.source(), self.inputs(actor="p2"), "SAMPLED_ACTOR_HISTORY_UNAVAILABLE")]
        for source, inputs, reason in cases:
            result = self.project(source, inputs)
            self.assertEqual(result.observations, (None, None))
            self.assertEqual(result.receipt["unavailable_reasons"], [reason, reason])
            self.assertEqual(result.receipt["unavailable_boundaries"], 2)
            self.assertIsNone(result.program)
            self.assertEqual(result.program_cutoffs,(None,None))

    def test_source_encoding_mismatch_is_not_an_omission_or_fallback(self):
        source = self.source(mismatch=True)
        self.assertEqual(source.receipt["status"], "ENCODING_MISMATCH")
        result = self.project(source, self.inputs())
        self.assertEqual(result.observations, (None, None))
        self.assertEqual(result.receipt["unavailable_reasons"], ["ORIGINAL_SOURCE_ENCODING_UNAVAILABLE"]*2)

    def test_party_order_moves_items_and_stats_cannot_silently_change(self):
        source = self.source()
        for mutate in (lambda mons: mons.reverse(), lambda mons: mons[0]["moves"].reverse(),
                lambda mons: mons[0].__setitem__("item", "choiceband"),
                lambda mons: mons[0]["stats"].__setitem__("spe", 171)):
            inputs = self.inputs(); mutate(inputs["rows"][0]["own_request"]["side"]["pokemon"])
            inputs["rows"][0]["own_request_sha256"] = digest(inputs["rows"][0]["own_request"])
            result = self.project(source, inputs)
            self.assertEqual(result.observations, (None, None))
            self.assertEqual(result.receipt["unavailable_reasons"], ["OWN_KNOWLEDGE_ANCHOR_MISMATCH"]*2)

    def test_missing_root_anchor_or_duplicate_party_identity_is_uncertain(self):
        source = self.source()
        for request in (None, actor_fixture.own_request()):
            if request is not None:
                mons = request["side"]["pokemon"]; mons[1]["ident"] = mons[0]["ident"]
            result = self.project(source, self.inputs(requests=[request], suffixes=[[]]))
            self.assertEqual(result.observations, (None,))
            self.assertEqual(result.receipt["unavailable_reasons"], ["OWN_REQUEST_ANCHOR_UNAVAILABLE"])

    def test_empty_active_list_at_force_switch_is_not_a_pp_crash(self):
        own = actor_fixture.own_request(); own["active"] = []; own["forceSwitch"] = [True]
        result = self.project(self.source(), self.inputs(requests=[own], suffixes=[[]]))
        self.assertIsNotNone(result.observations[0])
        self.assertEqual(result.receipt["root_pp_differences"], [])

    def test_hash_roster_and_shell_prefix_substitutions_refuse(self):
        source = self.source(); inputs = self.inputs()
        for mutate in (lambda data: data.__setitem__("materialized_shell_prefix_used", True),
                lambda data: data["rows"][0].__setitem__("snapshot_index", False),
                lambda data: data["rows"][1].__setitem__("observed_generation", 0),
                lambda data: data["rows"][0].__setitem__("own_request_sha256", "f"*64)):
            value = deepcopy(inputs); mutate(value)
            with self.assertRaises(ValueError): self.project(source, value)
        value = deepcopy(inputs); value["rows"][0]["visible_suffix"] = ["|turn|99"]
        value["rows"][0]["visible_suffix_sha256"] = digest(value["rows"][0]["visible_suffix"])
        with self.assertRaisesRegex(ValueError, "shell prefix"): self.project(source, value)
        with self.assertRaisesRegex(ValueError, "independently bound"):
            project_original_actor_replay(source, inputs, config=self.config,
                expected_source_identity="f"*64, expected_replay_sha256=digest(inputs))

    def test_final_encoder_callback_cannot_mutate_original_replay_input(self):
        source = self.source(); inputs = self.inputs(); original = LocalShowdownEnv._observe
        calls = []
        def mutate(env, player, **kwargs):
            value = original(env, player, **kwargs); calls.append(value)
            if len(calls) == 4: inputs["rows"][1]["own_request"]["late"] = True
            return value
        with patch.object(LocalShowdownEnv, "_observe", side_effect=mutate), \
                self.assertRaisesRegex(ValueError, "inputs or prior outputs changed"):
            self.project(source, inputs)

    def test_later_encoder_callback_cannot_mutate_prior_returned_output(self):
        source = self.source(); original = LocalShowdownEnv._observe; calls = []
        def mutate(env, player, **kwargs):
            value = original(env, player, **kwargs); calls.append(value)
            if len(calls) == 4: calls[2].metadata["late"] = True
            return value
        with patch.object(LocalShowdownEnv, "_observe", side_effect=mutate), \
                self.assertRaisesRegex(ValueError, "inputs or prior outputs changed"):
            self.project(source, self.inputs())

    def test_private_projection_cannot_be_serialized_and_full_state_changes_identity(self):
        value = self.project(self.source(), self.inputs())
        with self.assertRaises(TypeError): pickle.dumps(value)
        before = projection_identity(value)
        value.materialization_states[1].self_request["late"] = True
        self.assertNotEqual(projection_identity(value), before)
        before=projection_identity(value)
        value.program["events"][0]["lines"].append("|turn|999")
        self.assertNotEqual(projection_identity(value),before)


@dataclass(frozen=True)
class AuthoredActorReplayJoinJob(chain_fixture.AuthoredRetainedChainJob):
    """Real disposable source/native/constructor/replay owners; authored physics.

    Source capture is explicit: absent histories retain the uncertain roster;
    the coherent capture fixture exercises root and AFTER inputs. Neither mode
    fabricates an actor snapshot or qualifies model inputs or simulator physics.
    """
    capture_actor_history: bool = False

    def __call__(self, ownership, phase):
        import pokezero.mcts_eval.search_over_raw_actor_replay_projection as projection
        original = chain_fixture.produce_owned_native_diagnostic_chain
        receipts = []
        def join(*args, **kwargs):
            private, report = original(*args, **kwargs)
            owner = private["native_owner"]
            material, replay = private["materializations"][0], private["replays"][0]
            def read(material_row=material, replay_row=replay):
                return projection.read_owned_original_actor_replay(owner, material_row, replay_row,
                    ledger=args[0], source_config=self.config,
                    native_extract=chain_fixture.AuthoredChainHelpers.extract)
            actual = read()
            if self.capture_actor_history:
                if actual.receipt["encoded_boundaries"] != 2 or actual.receipt["unavailable_reasons"] != [None,None]:
                    raise ValueError("coherent original p1 history did not encode root and actual AFTER boundary: "+
                        str(actual.receipt["unavailable_reasons"]))
                if actual.materialization_states[1].self_request["active"][0]["moves"][0]["pp"] != 14:
                    raise ValueError("actual AFTER own request/PP was not retained")
                # Even identical authored teams do not create sampled-p2
                # history. Test this through the actual source program and
                # replay input reader, not a caller-supplied uncertainty flag.
                from pokezero.mcts_eval.search_over_raw_actor_replay_inputs import read_actor_replay_inputs
                from pokezero.mcts_eval.search_over_raw_boundary_run import raw_root_json
                source = owner.source_owner.actor_program(owner.job.statistics_root_id,config=self.config)
                path = replay["validator"]
                inputs = read_actor_replay_inputs(replay["result"]["capture"],
                    process_records=json.loads((path.directory/"bridge-transport.json").read_bytes())["records"][:-2],
                    initial_snapshot_json=raw_root_json(path.raw_root),clone_binding=path.expected["clone_binding"],
                    request_ids=path.expected["request_ids"],actor="p2",runtime=path.runtime)
                other = projection.project_original_actor_replay(source,inputs,config=self.config,
                    expected_source_identity=retained_program_identity(source),expected_replay_sha256=digest(inputs))
                if other.receipt["unavailable_reasons"] != ["SAMPLED_ACTOR_HISTORY_UNAVAILABLE"]*2:
                    raise ValueError("sampled opponent acquired the original private history")
            elif actual.receipt["unavailable_reasons"] != ["ORIGINAL_ACTOR_HISTORY_ABSENT"]*2:
                raise ValueError("absent original history was not kept uncertain")
            if not actual.receipt["synthetic_scope"] or not actual.receipt["actual_native_path_crosswalk_validated"]:
                raise ValueError("owned authored scope/crosswalk not retained")
            from pokezero.mcts_eval.search_over_raw_after_admission import admit_owned_native_after
            leaf_plan=private["leaf_cell_plan"]
            event=next(e for e in leaf_plan.receipt()["events"]
                if e["canonical_construction_ordinal"]==replay["construction_ordinal"]
                and e["evaluation_ordinal"]==replay["evaluation_ordinal"])
            admission=admit_owned_native_after(private,ledger=args[0],event_id=event["event_id"],
                expected_plan_sha256=leaf_plan.sha256,source_config=self.config,
                native_extract=chain_fixture.AuthoredChainHelpers.extract)
            admitted=admission.validate()
            if not admitted["original_endpoint_provenance_validated"] or admitted["after_snapshot_index"]!=1:
                raise ValueError("original evaluated AFTER endpoint was not admitted")
            if admission.raw_after==replay["validator"].raw_root:
                raise ValueError("PRE/root state substituted for actual AFTER endpoint")
            if admission.programs["p2"] is not None or admitted["complete_actor_histories_validated"]:
                raise ValueError("original source opponent history leaked into sampled p2 admission")
            if self.capture_actor_history:
                if admission.programs["p1"]!=actual.actor_program(1) or not admitted["original_p1_history_provenance_validated"]:
                    raise ValueError("original AFTER actor chronology was not retained")
            elif admission.programs["p1"] is not None:
                raise ValueError("absent original actor history acquired a program")
            try:pickle.dumps(admission)
            except TypeError:pass
            else:raise ValueError("private original AFTER admission serialized")
            from dataclasses import replace
            changed=deepcopy(admission.raw_after);changed["battle"]["log"].append("|turn|999")
            try:replace(admission,raw_after=changed).validate()
            except ValueError:pass
            else:raise ValueError("substituted AFTER endpoint retained admission")
            # A dropped canonical coordinate contradicts the original plan;
            # it cannot be converted into ordinary missing data.
            bad_chain=dict(private,replays=list(private["replays"])[1:])
            try:admit_owned_native_after(bad_chain,ledger=args[0],event_id=event["event_id"],
                expected_plan_sha256=leaf_plan.sha256,source_config=self.config,
                native_extract=chain_fixture.AuthoredChainHelpers.extract)
            except ValueError:pass
            else:raise ValueError("dropped canonical path became ordinary missing data")
            for roster in ("materializations","replays"):
                rows=list(private[roster]);bad=dict(rows[1],result=deepcopy(rows[1]["result"]))
                bad["result"]["scientific_collection_authorized"]=True;rows[1]=bad
                promoted=dict(private,**{roster:rows})
                try:admit_owned_native_after(promoted,ledger=args[0],event_id=event["event_id"],
                    expected_plan_sha256=leaf_plan.sha256,source_config=self.config,
                    native_extract=chain_fixture.AuthoredChainHelpers.extract)
                except ValueError:pass
                else:raise ValueError("promoted noncanonical "+roster+" wrapper became original admission evidence")
            if self.capture_actor_history:
                import pokezero.mcts_eval.search_over_raw_after_admission as admission_module
                original_compile=admission_module.compile_actor_program
                def mutate_after_compile(program,**kwargs):
                    value=original_compile(program,**kwargs)
                    program["events"][0]["lines"].append("|turn|999")
                    return value
                with patch.object(admission_module,"compile_actor_program",side_effect=mutate_after_compile):
                    try:admit_owned_native_after(private,ledger=args[0],event_id=event["event_id"],
                        expected_plan_sha256=leaf_plan.sha256,source_config=self.config,
                        native_extract=chain_fixture.AuthoredChainHelpers.extract)
                    except ValueError as error:
                        if "original AFTER admission" not in str(error):raise
                    else:raise ValueError("final encoder callback changed admitted AFTER actor history")
            admitted["scientific_measurements"]=0
            receipts.append(admitted)
            from pokezero.mcts_eval.search_over_raw_after_admission import execute_original_native_after_bank
            original_admission,bank=execute_original_native_after_bank(private,ledger=args[0],ownership=ownership,
                event_id=event["event_id"],expected_plan_sha256=leaf_plan.sha256,source_config=self.config,
                native_extract=chain_fixture.AuthoredChainHelpers.extract,runtime=self.runtime,
                policies={seat:chain_fixture.source_fixture.AuthoredPolicy(self.checkpoint,self.mode) for seat in ("p1","p2")})
            bank_receipt=bank.validate()
            if bank_receipt["required_cells"]!=8 or bank_receipt["bridge_owned_cells"]!=0 \
                    or any(r["signed_outcome"] is not None for r in bank_receipt["cells"]):
                raise ValueError("unsupported sampled actor was not kept in all eight original cells")
            if digest(original_admission.validate())!=digest(admission.validate()):
                raise ValueError("original owning admission changed on bank entry")
            if actual.receipt["native_evaluation_input_parity"] or actual.receipt["independent_terminal_labels"]:
                raise ValueError("owning projection overclaimed evaluator truth")
            receipts.append(actual.receipt)
            # The good capture cannot clear a promoted/substituted result body.
            bad_material = dict(material, result=deepcopy(material["result"]))
            bad_material["result"]["scientific_collection_authorized"] = True
            try: read(material_row=bad_material)
            except ValueError: pass
            else: raise ValueError("good constructor capture cleared a promoted wrapper")
            bad_replay = dict(replay, result=deepcopy(replay["result"]))
            bad_replay["result"]["runtime_sha256"] = "f"*64
            try: read(replay_row=bad_replay)
            except ValueError: pass
            else: raise ValueError("good replay capture cleared a substituted runtime")
            # This mutation happens only after all owning readers, at the
            # encoding callback. It must not become the readout baseline.
            original_project = projection.project_original_actor_replay
            old = deepcopy(material["validator"].last_validation)
            def mutate(*a, **kw):
                value = original_project(*a, **kw)
                material["validator"].last_validation["synthetic_process_fixture"] = False
                return value
            try:
                with patch.object(projection, "project_original_actor_replay", side_effect=mutate):
                    try: read()
                    except ValueError as error:
                        if "owning readouts changed" not in str(error): raise
                    else: raise ValueError("late owning readout mutation cleared")
            finally:
                material["validator"].last_validation = old
            owner.validate(args[0])
            return private, report
        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(patch.object(chain_fixture,"produce_owned_native_diagnostic_chain",side_effect=join))
            if self.capture_actor_history:
                collect = chain_fixture.collect_owned_exploration_source
                def capture(*a, **kw):
                    handle = collect(*a, **dict(kw,capture_actor_history=True,_synthetic_factory=CoherentActorSourceFactory))
                    source = handle.actor_program(handle.source["roots"][0]["root_id"],config=self.config)
                    if source.receipt["status"] != "INPUT_CORRESPONDENCE_MATCHED":
                        actual = {row["observation_ordinal"]:value for row,value in handle._actor_history._observations}
                        changes = []
                        def compare(left,right,path=""):
                            if type(left) is dict and type(right) is dict:
                                for key in sorted(set(left)|set(right)):compare(left.get(key),right.get(key),path+"/"+str(key))
                            elif type(left) is list and type(right) is list and len(left)==len(right):
                                for i,(x,y) in enumerate(zip(left,right)):compare(x,y,path+"/"+str(i))
                            elif left != right and len(changes)<20:changes.append((path,left,right))
                        for coordinate,value in zip(source.compiled.receipt["coordinates"],source.compiled.observations):
                            compare(_state_payload(actual[coordinate]),_state_payload(value),str(coordinate))
                        raise ValueError("authored source encoding mismatch before native work: "+str(changes))
                    return handle
                stack.enter_context(patch.object(chain_fixture,"collect_owned_exploration_source",side_effect=capture))
                stack.enter_context(patch.object(chain_fixture.AuthoredChainHelpers,"extract",staticmethod(coherent_native_extract)))
                stack.enter_context(patch.object(chain_fixture,"AuthoredRetainedNativeJob",AuthoredCoherentNativeJob))
                stack.enter_context(patch("pokezero.local_showdown.load_showdown_dex_cached",
                    return_value=chain_fixture.construction_fixture._dex()))
            try:
                result = super().__call__(ownership, phase)
            except BaseException:
                import traceback
                path = Path(ownership["directory"]).parent/"authored-error.json"
                body = json.loads(path.read_bytes()) if path.is_file() else {}
                body["traceback"] = traceback.format_exc()
                path.write_text(json.dumps(body))
                raise
        return dict(result, actor_projection_receipts=receipts,
            substituted_wrapper_checks=2, late_readout_mutation_rejected=True)


class OwnedActorReplayJoinIntegrationTests(unittest.TestCase):
    def test_capture_enabled_owned_chain_encodes_actual_after_and_keeps_p2_unavailable(self):
        helper = chain_fixture.RetainedNativeChainIntegrationTests()
        helper.setUp(); self.addCleanup(helper.doCleanups)
        source_fixture = chain_fixture.source_fixture
        source_runtime = source_fixture.outcome_runtime_binding
        constructor_runtime = chain_fixture.outcome_runtime_binding
        prepare_source = helper.s.prepare
        evidence_file = Path(__file__).resolve()
        def source_with_capture(*args, **kwargs):
            return prepare_source(*args, **dict(kwargs,job_mode="actor_history"))
        def prepare_source_runtime(config, *, checkpoint_path, extra_files=()):
            path = Path(config.resolved_showdown_root())/"dist/sim/index.js"
            settings = json.loads(path.read_bytes())
            root = coherent_actor_root()
            settings["actorRequests"] = deepcopy(root["boundaryRequests"])
            settings["actorPrefix"] = ["|player|p1|PokeZero p1|", "|player|p2|PokeZero p2|", "|gen|3", "|start",
                "|switch|p1a: Swampert|Swampert, L84|100/100",
                "|switch|p2a: Snorlax|Snorlax, L80|100/100", "|turn|1"]
            path.write_text(json.dumps(settings))
            return source_runtime(config,checkpoint_path=checkpoint_path,extra_files=(*extra_files,evidence_file))
        def prepare_constructor_runtime(config, *, checkpoint_path, extra_files=()):
            path = Path(config.resolved_showdown_root())/"fixture.json"
            fixture = json.loads(path.read_bytes())
            fixture.update(snapshot=coherent_actor_root(),nonterminalReplay=True)
            path.write_text(json.dumps(fixture))
            return constructor_runtime(config,checkpoint_path=checkpoint_path,extra_files=(*extra_files,evidence_file))
        # Both actual source packets and constructor output are independently
        # predeclared in the coherent fixture BEFORE their respective recipe,
        # runtime and whole-chain review hashes. No post-result grafting.
        with patch.object(helper.s,"prepare",side_effect=source_with_capture), \
                patch.object(source_fixture,"outcome_runtime_binding",side_effect=prepare_source_runtime), \
                patch.object(chain_fixture,"outcome_runtime_binding",side_effect=prepare_constructor_runtime):
            original = helper.prepare()
        job = AuthoredActorReplayJoinJob(**vars(original),capture_actor_history=True)
        terminal = helper.s.run_job(job)
        error = helper.s.directory/"authored-error.json"
        self.assertEqual(terminal["supervisor_exit_code"],0,error.read_text() if error.exists() else terminal)
        result = terminal["result"]
        self.assertEqual((result["constructor_runs"],result["replay_runs"]),(4,8))
        admission,projection=result["actor_projection_receipts"]
        self.assertTrue(admission["original_endpoint_provenance_validated"])
        self.assertTrue(admission["original_p1_history_provenance_validated"])
        self.assertFalse(admission["sampled_p2_history_provenance_validated"])
        self.assertEqual(projection["encoded_boundaries"],2)
        self.assertFalse(projection["native_evaluation_input_parity"])
        self.assertFalse(projection["runtime_qualified"])
        self.assertTrue(result["late_readout_mutation_rejected"])

    def test_disposable_owned_chain_preserves_absence_and_rejects_late_substitutions(self):
        helper = chain_fixture.RetainedNativeChainIntegrationTests()
        helper.setUp(); self.addCleanup(helper.doCleanups)
        # Build serializer/filter bytes and raw logs BEFORE the original
        # runtime, source review and chain recipe are frozen. The old chain
        # fixture has no log-consumer contract and must not be promoted by
        # skipping this reader's stronger requirements.
        prepare_source = helper.s.prepare
        map_fixture = chain_fixture.construction_fixture.map_fixture
        def source_with_logs(*args, **kwargs):
            return prepare_source(*args, **dict(kwargs, job_mode="public_history"))
        def root_with_log():
            root, native, constructor = map_fixture()
            root["battle"]["log"] = []
            return root, native, constructor
        with patch.object(helper.s, "prepare", side_effect=source_with_logs), \
                patch.object(chain_fixture.construction_fixture, "map_fixture", side_effect=root_with_log):
            original = helper.prepare()
        job = AuthoredActorReplayJoinJob(**vars(original))
        terminal = helper.s.run_job(job)
        error = helper.s.directory/"authored-error.json"
        self.assertEqual(terminal["supervisor_exit_code"], 0,
            error.read_text() if error.exists() else terminal)
        result = terminal["result"]
        self.assertEqual((result["constructor_runs"], result["replay_runs"]), (4, 8))
        self.assertEqual(result["substituted_wrapper_checks"], 2)
        self.assertTrue(result["late_readout_mutation_rejected"])
        self.assertEqual(len(result["actor_projection_receipts"]), 2)
        self.assertFalse(result["scientific_collection_authorized"])
