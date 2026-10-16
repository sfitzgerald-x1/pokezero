"""Prospective fresh-root decomposition, with no continuation labels or holdout.

Reuses the source owner, native constructor/dispatch recorder and enclosing
auditor. Sixty-four fixed native iterations expose actual priors on all four
worlds, not the historical timed run's incomplete searched-world mixture.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import time

from .search_over_raw import SearchConfiguration, digest, require
from .search_over_raw_world_inputs import _state_payload, public_request_witness
from .search_over_raw_ledger import _write_new

SCHEMA = "pokezero.search-over-raw.root-prior-decomposition.v1"


class RootInputUnavailable(ValueError):
    pass


def make_contract(namespace, seeds, excluded, authority):
    require(type(seeds) is list and len(seeds) == len(set(seeds)) == 32
        and all(type(s) is int and 0 <= s < 2**32 for s in seeds)
        and not set(seeds) & set(excluded), "32 fresh exposure-disjoint seeds required")
    return dict(schema=SCHEMA, namespace=namespace,
        panels={"exploration":dict(seeds=seeds, root_slots=[
            dict(root_id=f"exploration:{s}:{j}", source_seed=s, root_slot=j)
            for i,s in enumerate(seeds) for j in range(4 if i < 4 else 3)])},
        excluded_seeds=excluded, authority=authority, candidate_seat="p1",
        exclude_opening_requests=True, source_policy="raw_argmax_both_seats",
        native_iterations_per_world=16, native_worlds=4, continuations=0,
        source_root_rule="seed-derived priorities over completed public request catalog; no replacements",
        max_source_boundaries=250, consecutive_unavailable_cap=3,
        source_actor_history_capture=False,
        wall_clock_search_budget=None, emergency_calendar_seconds=3600,
        holdout_authorized=False, phase_b_authorized=False, retraining_authorized=False,
        retries_permitted=False, collection_authorized=True, runtime_qualified=False)


def validate_contract(c):
    require(digest(c) == digest(make_contract(c["namespace"],c["panels"]["exploration"]["seeds"],
        c["excluded_seeds"],c["authority"])), "root-prior diagnostic contract drift")


def distribution_metrics(left, right, raw_argmax):
    require(len(left) == len(right) and len(left) > 0
        and all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in [*left,*right])
        and abs(math.fsum(left)-1) < 2e-5 and abs(math.fsum(right)-1) < 2e-5,
        "normalized comparable distributions required")
    def kl(p,q):
        if any(a > 0 and b == 0 for a,b in zip(p,q)):
            return "INFINITE_POSITIVE_MASS_AT_ZERO"
        return math.fsum(a*math.log(a/b) for a,b in zip(p,q) if a > 0)
    return dict(argmax_agreement=max(range(len(left)),key=lambda i:left[i]) ==
        max(range(len(right)),key=lambda i:right[i]), kl_left_to_right=kl(left,right),
        kl_right_to_left=kl(right,left), right_probability_of_raw_argmax=right[raw_argmax])


def seed_cluster_interval(values, seeds, *, draws=10000):
    """Descriptive percentile bootstrap; resample seeds, retain all their roots."""
    import numpy as np
    require(len(values) == len(seeds) and len(values) > 0, "paired observed metric/seed rows required")
    groups = sorted(set(seeds))
    if any(type(v) is str for v in values):
        return dict(interval=None,reason="infinite_KL_no_finite_interval",observed_seed_clusters=len(groups))
    if len(groups) < 2:
        return dict(interval=None,reason="fewer_than_two_observed_seed_clusters",observed_seed_clusters=len(groups))
    totals = np.asarray([math.fsum(v for v,s in zip(values,seeds) if s == g) for g in groups])
    counts = np.asarray([seeds.count(g) for g in groups])
    weights = np.random.default_rng(20261016).multinomial(len(groups),[1/len(groups)]*len(groups),size=draws)
    means = (weights @ totals)/(weights @ counts)
    return dict(interval=np.quantile(means,[.025,.975]).tolist(),observed_seed_clusters=len(groups),
        method="descriptive_95_percentile_seed_cluster_bootstrap",draws=draws,bootstrap_seed=20261016,
        full_fixed_panel_inference=False)


def buffer_arrays(buffers, observation):
    import numpy as np
    from ..golden_corpus import GOLDEN_ARRAY_FIELDS
    types = {name:dtype for name,dtype,_ in GOLDEN_ARRAY_FIELDS}
    return {name:np.frombuffer(buffers[name],dtype=dtype).reshape(
        np.asarray(getattr(observation,name)).shape) for name,dtype in types.items()}


def encode_call(call, rows, observation, native, tables):
    """Exact retained input content first, then depth-zero encoding/forward."""
    import numpy as np
    import pokezero_search
    from .search_over_raw_public_fold import read_dispatch_inputs
    from .search_over_raw_exploration_entry import _decode
    p = read_dispatch_inputs(call, rows=rows, expected_tables=tables)
    actual = call["native_report"]["root_priors"]
    if actual is None:
        raise RootInputUnavailable("native root has no model prior")
    fold = _decode(p["fold"])
    require(_state_payload(fold) == p["fold"], "native fold decode did not round trip")
    encoder = pokezero_search.LeafEncoder(json.dumps(p["tables"]),json.dumps(p["public_inputs"]),
        json.dumps(p["context"]),p["state"])
    buffers = encoder.encode_leaf(p["state"],pokezero_search.FoldState.from_payload(fold),
        p["context"]["turn"],engine_authoritative=False)
    arrays = buffer_arrays(buffers,observation)
    _, logits, priors, width = native.eval_obs_flat(1,
        arrays["categorical_ids"].flatten().tolist(),arrays["numeric_features"].flatten().tolist(),
        arrays["token_type_ids"].flatten().tolist(),arrays["attention_mask"].flatten().tolist(),[True])
    mapping = encoder.self_action_map(p["state"],root=True,engine_authoritative=False)
    require(all(type(i) is int and 0 <= i < width for _,i in mapping), "native root map unavailable")
    gathered = [priors[i] for _,i in mapping]
    total = math.fsum(gathered)
    require(total > 0 or len(mapping) == 1, "native root prior has no gathered mass")
    gathered = [1.] if len(mapping) == 1 else [x/total for x in gathered]
    require(type(actual) is list and len(actual) == len(gathered)
        and max(abs(a-b) for a,b in zip(actual,gathered)) < 2e-5,
        "independent root encode/forward does not reproduce actual native priors")
    differences = {}
    for name,got in arrays.items():
        want = np.asarray(getattr(observation,name),dtype=got.dtype)
        indices = np.argwhere(got != want)
        differences[name] = dict(count=len(indices),positions=indices.tolist())
    return dict(input_sha256=call["input_sha256"], arrays={k:v.tolist() for k,v in arrays.items()},
        array_differences_from_raw=differences, policy_logits=list(logits),
        unmasked_model_priors=list(priors), native_option_map=mapping,
        native_option_priors=actual, native_recomputed_option_priors=gathered)


def native_measure(context, boundary, checkpoint, seed, *, pending=None, oracle=None, evidence_path=None):
    """Run the unchanged fixed-work constructor and native search; never labels."""
    from .search_over_raw_adapters import _incumbent_runtime
    from .search_over_raw_world_inputs import NativeWorldInputs,NativeInputHandle,_binding,validate_fixed_work_inputs
    from .paper_reference_runtime import PublicRootRequest
    from ..showdown_fixture import pack_team
    from ..env import BattleStartOverride
    request = PublicRootRequest.capture(context.public_materialization_state,context.observation,pending_transition=pending)
    boundary.validate(request,checkpoint)
    decider, policy, _ = _incumbent_runtime(checkpoint,checkpoint.showdown_root,1.,fixed_work_iterations=16)
    original = policy._native
    try:
        policy._annotation_source = boundary
        if oracle is not None:
            from .search_over_raw_oracle import opening_team
            oracle.validate(request,policy._set_source)
            own = opening_team(request.state.self_initial_request,"p1",policy._set_source)
            policy._fixed_override = BattleStartOverride(player_teams={"p1":pack_team(own),
                "p2":pack_team(oracle.opponent_team)},observation_format_id="gen3randombattle")
        recorder = NativeWorldInputs(_binding(request),16,deadline_limited=False,record_atomic_work=True,
            record_construction=True,record_public_fold=True,expected_public_request=public_request_witness(request),
            pending_transition=pending)
        policy._world_attempt_observer = recorder.attempt
        policy._world_observer = recorder.constructed
        policy._world_construction_observer = recorder.world_built
        policy._native_construction_observer = recorder.native_built
        policy._native = lambda:NativeInputHandle(original(),recorder)
        require(policy._config.model_decision_time_ms is None and policy._config.worlds == 4,
            "diagnostic cannot inherit a wall-clock decision ceiling")
        # Direct diagnostic dispatch deliberately includes forced roots rather
        # than production's single-action shortcut, so oracle calls are observed.
        decision = policy._search(context,rng=random.Random(seed))
        evidence = dict(total_iterations=policy.stats.total_iterations)
        manifest = recorder.finish(evidence,attempted=policy.stats.worlds_attempted,
            constructed=policy.stats.worlds_constructed,searched=policy.stats.worlds_searched)
        evidence["consumed_world_inputs"] = manifest
        if evidence_path is not None:
            _write_new(evidence_path,dict(status="UNVALIDATED_NATIVE_INPUTS",evidence=evidence,
                search_metadata=decision.metadata))
        validate_fixed_work_inputs(evidence,arm="incumbent",binding=_binding(request),
            statistics_root_id=context.battle_id,construction_required=True,
            expected_public_request_sha256=boundary.public_request_sha256,native_public_fold_required=True)
        from ..engine_search import _latch_encoder_tables_to_model_config
        from ..neural_policy import load_transformer_model_config
        # Production latches exporter defaults to the pinned checkpoint masks.
        # Independently reproduce that law; do not compare dispatch to unlatched
        # defaults or merely trust the policy's in-memory table object.
        tables = json.loads(_latch_encoder_tables_to_model_config(
            Path(checkpoint.tables_path).read_text(),load_transformer_model_config(checkpoint.checkpoint_path)))
        calls = [encode_call(call,manifest["rows"],context.observation,original(),tables)
            for call in manifest["calls"]]
        vocabulary = policy._choice_vocabulary(context)
        width = len(context.observation.legal_action_mask)
        worlds = []
        for call, encoded in zip(manifest["calls"],calls):
            vector = [0.]*width
            for (label,_),probability in zip(encoded["native_option_map"],encoded["native_option_priors"]):
                index = vocabulary.action_index(label)
                require(type(index) is int and context.observation.legal_action_mask[index],
                    "native option has no legal original request correspondence")
                vector[index] += probability
            vector = [v/math.fsum(vector) for v in vector]
            for ordinal in call["construction_ordinals"]:
                worlds.append(dict(construction_ordinal=ordinal,call_ordinal=call["call_ordinal"],
                    distribution=vector))
        require(sorted(w["construction_ordinal"] for w in worlds) == list(range(4)),
            "all four original world slots required")
        aggregate = [math.fsum(w["distribution"][i] for w in worlds)/4 for i in range(width)]
        # Join the independently encoded vectors to the production aggregate,
        # whose arms are visit-sorted and must NOT be zipped to root_priors.
        from ..engine_search import _aggregate_root_arms
        rows_by_ordinal = {r["ordinal"]:r for r in manifest["rows"]}
        actual_arms = _aggregate_root_arms([
            dict(report=c["native_report"],side_key=rows_by_ordinal[o]["slot_sides"][context.player_id])
            for c in manifest["calls"] for o in c["construction_ordinals"]])
        require(actual_arms.prior_cause is None,"actual search aggregate has no authoritative model prior")
        actual_vector = [0.]*width
        for label,weight in actual_arms.prior_share.items():
            index = vocabulary.action_index(label)
            require(type(index) is int and context.observation.legal_action_mask[index],
                "actual aggregate has no original legal action correspondence")
            actual_vector[index] += weight/4
        require(max(abs(a-b) for a,b in zip(aggregate,actual_vector)) < 2e-5,
            "independent per-world average differs from actual search aggregate")
        override = decision.metadata.get("engine_mcts",{}).get("override")
        if override is not None:
            require(override["root_allocation"]["prior_authority"] is True,
                "recorded search aggregate lacks prior authority")
            for arm in override["root_allocation"]["arms"]:
                require(arm["model_prior"] is not None and arm["action_index"] is not None
                    and abs(arm["model_prior"]-actual_vector[arm["action_index"]]) < 2e-5,
                    "recorded model-prior telemetry differs from actual aggregate")
        return dict(worlds=worlds,aggregated_prior=aggregate,calls=calls,
            actual_aggregated_prior=actual_vector,actual_model_argmax=None if override is None else override["model_argmax"],
            actual_native_evidence=manifest,search_metadata=decision.metadata,
            fixed_iterations=policy.stats.total_iterations,oracle_receipt=oracle.receipt() if oracle else None)
    finally:
        policy._native = original
        policy._world_attempt_observer = policy._world_observer = None
        policy._world_construction_observer = policy._native_construction_observer = None
        decider.close()


@dataclass(frozen=True)
class RootPriorJob:
    spec_path: str
    entry_sha256: str
    def __call__(self, ownership, phase):
        from .search_over_raw_bounded_a2 import collect_source
        from .search_over_raw_annotation_boundary import ActorAnnotationBoundary
        from .search_over_raw_actor_timed_run import ActorInputUnavailable
        from .search_over_raw_raw_run import RawSelectionJob,raw_runtime
        from .paper_reference_runtime import PublicRootRequest
        from .search_over_raw_oracle import TeamOracle
        from ..randbat import load_gen3_randbat_source_cached
        from ..public_decision_corpus import PublicObservation
        phase("startup")
        spec,config,checkpoint = read_spec(self.spec_path)
        out = Path(spec["output_directory"]); began = time.time(); consecutive = 0
        phase("selection")
        try:
            for seed in spec["contract"]["panels"]["exploration"]["seeds"]:
                captures = {}
                def capture(env, archive, context, action, history):
                    # Original audited raw-policy callback; no unused whole-
                    # game actor-history recorder or extra observe() call.
                    from .search_over_raw_annotation_boundary import _CAPTURE_ORIGIN
                    from ..local_showdown import LocalShowdownEnv
                    pending = archive._public[context.decision_round_index][1]
                    require(type(env) is LocalShowdownEnv and history is None,
                        "root probe requires the original live source seam without history replay")
                    request = PublicRootRequest.capture(context.public_materialization_state,
                        context.observation,pending_transition=pending)
                    actual = PublicRootRequest.capture(env.public_materialization_state("p1"),
                        context.observation,pending_transition=pending)
                    require(public_request_witness(actual) == public_request_witness(request),
                        "actual source root differs from archived request")
                    captures[context.decision_round_index] = ActorAnnotationBoundary(
                        deepcopy(env._state_for_player("p1")),
                        PublicObservation.from_observation(context.observation),
                        public_request_witness(request)["public_request_sha256"],
                        bool(env.tier2_residuals_active()),_capture_origin=_CAPTURE_ORIGIN)
                def guard():
                    require(time.time()-began < 3600,"root-prior emergency calendar cap")
                    require(hashlib.sha256(Path(self.spec_path).read_bytes()).hexdigest() == self.entry_sha256,
                        "original root-prior entry changed")
                    return True
                archive = None
                try:
                    archive,source,_ = collect_source(spec,config,checkpoint,seed,out/"sources"/str(seed),guard,
                        root_capture=capture,capture_actor_history=False)
                    for root in source["roots"]:
                        guard()
                        context,pending,snapshot = archive.selected(root)
                        row = dict(root_id=root["root_id"],source_seed=seed,status="INPUT_UNAVAILABLE",
                            source_sha256=digest(source),public_record_sha256=root["public_record_sha256"],
                            source_pipeline_validated=True)
                        try:
                            boundary = captures[root["source_request_index"]]
                            request = PublicRootRequest.capture(context.public_materialization_state,
                                context.observation,pending_transition=pending)
                            boundary.validate(request,checkpoint)
                            raw_cfg = SearchConfiguration("raw",seconds=1.)
                            # archive.selected above verifies the original canonical
                            # source record. The typed raw job additionally binds
                            # this exact request; it does not require admission of
                            # the diagnostic contract to a staged inference ledger.
                            raw_job = RawSelectionJob(context=context,pending_transition=pending,
                                root_id=root["root_id"]+":raw",statistics_root_id=root["root_id"],
                                selection_seed=seed,configuration=raw_cfg,checkpoint_contract=checkpoint,
                                showdown_root=checkpoint.showdown_root,expected_runtime_configuration=raw_runtime(
                                    raw_cfg,checkpoint.checkpoint_sha256,checkpoint.showdown_source_sha256),
                                expected_public_request_sha256=boundary.public_request_sha256)
                            raw = raw_job(ownership,lambda _name:None)
                            inputs = out/"root-inputs"/digest(root["root_id"])
                            inputs.mkdir(parents=True,exist_ok=False,mode=0o700)
                            _write_new(inputs/"raw.json",raw)
                            # Public context is not mutated or augmented with oracle state.
                            public = native_measure(context,boundary,checkpoint,seed,pending=pending,
                                evidence_path=inputs/"public-native.json")
                            oracle = TeamOracle.capture(snapshot,request,load_gen3_randbat_source_cached(checkpoint.showdown_root))
                            truth = native_measure(context,boundary,checkpoint,seed,pending=pending,oracle=oracle,
                                evidence_path=inputs/"oracle-native.json")
                            width = len(context.observation.legal_action_mask);raw_dist = [0.]*width
                            for i,p in zip(raw["evidence"]["legal_actions"],raw["evidence"]["priors"]):raw_dist[i]=p
                            vectors = dict(raw=raw_dist,oracle=truth["aggregated_prior"],aggregate=public["aggregated_prior"])
                            vectors.update({f"sample{w['construction_ordinal']}":w["distribution"] for w in public["worlds"]})
                            names = list(vectors)
                            row.update(status="COMPLETE",raw=raw,raw_observation=PublicObservation.from_observation(
                                context.observation).to_dict(),public=public,oracle=truth,distributions=vectors,
                                pairings={a+"__"+b:distribution_metrics(vectors[a],vectors[b],raw["action"])
                                    for i,a in enumerate(names) for b in names[i+1:]})
                            consecutive = 0
                        except (ActorInputUnavailable,RootInputUnavailable) as error:
                            row["unavailable_reason"] = getattr(error,"reason",str(error));consecutive += 1
                        _write_new(out/"completed-roots"/(digest(root["root_id"])+".json"),row)
                        print(json.dumps(dict(root_id=root["root_id"],status=row["status"])),flush=True)
                        if consecutive >= 3:return dict(status="THREE_CONSECUTIVE_UNAVAILABLE")
                    for missing in source["missing_root_ids"]:
                        _write_new(out/"completed-roots"/(digest(missing)+".json"),dict(root_id=missing,
                            source_seed=seed,status="INPUT_UNAVAILABLE",source_sha256=digest(source),
                            source_pipeline_validated=True,unavailable_reason="source_missing_fixed_slot"))
                        consecutive += 1
                        if consecutive >= 3:return dict(status="THREE_CONSECUTIVE_UNAVAILABLE")
                finally:
                    if archive is not None:archive.close()
            return dict(status="FIXED_ROOT_ROSTER_ATTEMPTED")
        finally:phase("cleanup")


def read_spec(path):
    from .search_over_raw_exploration_entry import _decode
    from .search_over_raw_outcome_run import verify_outcome_runtime
    s = json.loads(Path(path).read_bytes());validate_contract(s["contract"])
    require(all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in s["input_hashes"].items()),
        "root-prior entry source/exposure/runtime drift")
    config,checkpoint = _decode(s["source_config"]),_decode(s["checkpoint_contract"])
    verify_outcome_runtime(s["source_runtime"],config)
    return s,config,checkpoint


def prepare(directory, parent_entry, authority_path):
    """Reuse pinned champion/configuration, not the old roster or admission."""
    import secrets
    import sys
    import uuid
    from .search_over_raw_exploration_entry import _decode
    from .search_over_raw_outcome_run import outcome_runtime_binding
    from .search_over_raw_fixed_work_run import fixed_work_reader_source_hashes
    parent_path = Path(parent_entry).resolve();parent = json.loads(parent_path.read_bytes())
    config,checkpoint = _decode(parent["source_config"]),_decode(parent["checkpoint_contract"])
    excluded = sorted(set(parent["contract"]["excluded_seeds"]) |
        set(parent["contract"]["panels"]["exploration"]["seeds"]))
    seeds = []
    while len(seeds) < 32:
        seed = secrets.randbits(32)
        if seed not in excluded and seed not in seeds:seeds.append(seed)
    authority_path = Path(authority_path).resolve()
    sha = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    c = make_contract(str(uuid.uuid4()),seeds,excluded,dict(path=str(authority_path),sha256=sha(authority_path)))
    # The old exposure inventory is retained and checked as evidence of the
    # exclusions; no historical registration is edited or admitted again.
    exposure = {**parent["exposure_inventory"],str(parent_path):sha(parent_path)}
    require(all(sha(p) == h for p,h in exposure.items()),"prior exposure inventory drift")
    extras = [p for p in parent.get("selector_files",parent["input_hashes"]) if p.endswith((".so",".dylib"))]
    extras += [checkpoint.model_path,checkpoint.tables_path]
    runtime = outcome_runtime_binding(config,checkpoint_path=checkpoint.checkpoint_path,extra_files=extras)
    files = {**runtime["files"],**fixed_work_reader_source_hashes(),str(authority_path):sha(authority_path),
        str(Path(sys.executable).resolve()):sha(Path(sys.executable).resolve()),**exposure}
    cli = Path(__file__).resolve().parents[3]/"scripts/measure_search_over_raw_root_priors.py"
    files[str(cli)] = sha(cli)
    directory = Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False,mode=0o700)
    s = dict(schema=SCHEMA+".entry",contract=c,output_directory=str(directory/"run"),
        source_config=_state_payload(config),checkpoint_contract=_state_payload(checkpoint),
        source_runtime=runtime,input_hashes=files,exposure_inventory=exposure,
        interpretation="root-prior diagnostic; 16 fixed iterations in each of four worlds, public and team-only oracle; not a replay of historical timed mixtures; no continuations or strength claim",
        original_actor_capture="actual audited p1 raw-policy callback boundary before opponent policy; live request/state join and checkpoint-law reencoding; no whole-game actor-history replay",
        kl_definition="both directed KLs, no smoothing; positive mass at zero is explicitly infinite")
    path = directory/"entry-spec.json";_write_new(path,s);read_spec(path)
    return dict(entry_spec=str(path),entry_sha256=sha(path),source_seeds=seeds,roots=100,worker_launched=False)


def run(path, review_path):
    from .search_over_raw_supervision import _supervise_owned_job
    from .search_over_raw_exploration_entry import ExplorationAuditorContract
    s,_,_ = read_spec(path);entry_sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    review_bytes = Path(review_path).read_bytes();review = json.loads(review_bytes)
    require(review.get("status") == "CLEAR_ROOT_PRIOR_DECOMPOSITION_ENTRY"
        and review.get("entry_spec_sha256") == entry_sha and review.get("reviewer") == "Astra"
        and review.get("reviewed_before_first_source_work") is True
        and review.get("holdout_authorized") is False,"one prospective Astra entry review required")
    out = Path(s["output_directory"]);out.mkdir(parents=True,exist_ok=False,mode=0o700)
    (out/"completed-roots").mkdir();_write_new(out/"entry.json",s);_write_new(out/"entry-review.json",review)
    def admit():
        read_spec(path)
        require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == entry_sha
            and Path(review_path).read_bytes() == review_bytes,"reviewed entry changed")
        return True
    terminal = _supervise_owned_job(RootPriorJob(str(Path(path).resolve()),entry_sha),directory=out/"auditor",
        binding_sha256=digest(dict(entry=entry_sha,review=hashlib.sha256(review_bytes).hexdigest())),
        verify_admission=admit,contract=ExplorationAuditorContract(3600),kind="staged-exploration-run",
        work_phase="selection",validate_result=lambda value:require(value["status"] in {
            "FIXED_ROOT_ROSTER_ATTEMPTED","THREE_CONSECUTIVE_UNAVAILABLE"},"root prior result drift"))
    rows = [json.loads(p.read_bytes()) for p in sorted((out/"completed-roots").glob("*.json"))]
    roster = {r["root_id"] for r in s["contract"]["panels"]["exploration"]["root_slots"]}
    require(len({r["root_id"] for r in rows}) == len(rows)
        and all(r["root_id"] in roster for r in rows),"duplicate or foreign diagnostic root")
    admit()
    complete = [r for r in rows if r["status"] == "COMPLETE"]
    summaries = {}
    for key in sorted({k for r in complete for k in r["pairings"]}):
        metrics = [r["pairings"][key] for r in complete];n=len(metrics);missing=100-n
        matches=sum(r["argmax_agreement"] for r in metrics)
        summaries[key] = dict(observed_roots=n,uncertain_fixed_slots=missing,
            observed_argmax_agreement=matches/n, fixed_denominator_agreement_bounds=[matches/100,(matches+missing)/100])
        seeds = [r["source_seed"] for r in complete]
        summaries[key]["argmax_agreement_seed_clustered_95"] = seed_cluster_interval(
            [int(r["argmax_agreement"]) for r in metrics],seeds)
        for field in ("kl_left_to_right","kl_right_to_left","right_probability_of_raw_argmax"):
            vals = [r[field] for r in metrics];infinite=sum(type(v) is str for v in vals)
            summaries[key][field] = dict(infinite_count=infinite,
                observed_mean="INFINITE" if infinite else math.fsum(vals)/n,
                seed_clustered_95=seed_cluster_interval(vals,seeds))
    readout = dict(schema=SCHEMA+".readout",planned_roots=100,planned_seeds=32,
        recorded_roots=len(rows),complete_roots=len(complete),uncertain_fixed_slots=100-len(complete),
        pairings=summaries,supervisor_exit_code=terminal["supervisor_exit_code"],
        worker_exit_code=terminal["worker_exit_code"],terminal_status=terminal["status"],
        elapsed_seconds=terminal["elapsed_wall_seconds"],cpu_seconds=terminal["cpu_seconds_including_reaped_children"],
        continuation_games=0,scientific_strength_claim=False,holdout_authorized=False,phase_b_authorized=False,
        historical_mixture_reproduced=False,complete_case_scope="descriptive observed roots, not a registered full-panel effect")
    _write_new(out/"readout.json",readout)
    return readout
