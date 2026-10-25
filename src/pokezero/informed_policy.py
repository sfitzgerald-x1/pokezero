"""Opt-in P2 gameplay adapter for informed-adversary root minimax.

Reuses the incumbent's public-belief materializer and checkpoint-bound root
encoder, not its retry/drop/aggregate search loop. Four draws are required;
one unsupported draw or successor refuses the whole decision. No raw fallback.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
from typing import Any

from .determinization import _gen3_randbat_belief_start_override_result
from .engine_search import (EngineMctsConfig, EngineMctsPolicy,
                            _root_toxic_zero_after_upkeep_attestation)
from .engine_world import world_battle_spec
from .failure_capture import (Quarantine, captured_decision, current,
                              record_refusal, runtime_build, snapshot, unavailable,
                              require_capture_admission)
from .informed_root import exact_informed_one_ply, sample_informed_equilibrium
from .poke_engine_adapter import build_poke_engine_state
from .policy import PolicyContext, PolicyDecision
from .shallow_exact import ExactSearchRefused
from .shallow_exact_native import (
    NativePosition, NativeRenderRefused, validate_encoder_frame,
)
from .exact_session import NativeSessionBackend as NativeExactBackend, FixedShapeModel


@dataclass(frozen=True)
class InformedPolicyConfig:
    worlds: int = 4
    batch_size: int = 1024
    bulk_encode_workers: int = 8
    max_leaves: int = 100_000
    max_chance_branches: int = 200_000
    max_joint_actions: int = 324
    max_seconds: float = 60.0
    solver_tolerance: float = 1e-8

    def __post_init__(self):
        if (type(self.worlds) is not int or self.worlds != 4
                or any(type(v) is not int or v <= 0 for v in
                       (self.batch_size,self.bulk_encode_workers,self.max_leaves,
                        self.max_chance_branches,self.max_joint_actions))
                or self.batch_size > 4096 or self.bulk_encode_workers > 32
                or not math.isfinite(self.max_seconds) or self.max_seconds <= 0
                or not math.isfinite(self.solver_tolerance)
                or not 1e-9 <= self.solver_tolerance <= 1e-3):
            raise ValueError("P2 requires four worlds and positive bounded work/LP limits")


@dataclass
class InformedPolicyStats:
    decisions: int = 0
    completed_decisions: int = 0
    refusals: int = 0
    decision_wall_seconds: float = 0.0
    refusal_reasons: Counter = field(default_factory=Counter)
    last_decision: dict = field(default_factory=dict)
    capture_telemetry: dict | None = None

    def to_dict(self):
        return dict(decisions=self.decisions,completed_decisions=self.completed_decisions,
            refusals=self.refusals,decision_wall_seconds=self.decision_wall_seconds,
            refusal_reasons=dict(self.refusal_reasons),last_decision=dict(self.last_decision),
            failure_capture=self.capture_telemetry)


class InformedRootPolicy:
    requires_public_materialization_state = True

    def __init__(self, *, dex, set_source, checkpoint_path: str, model_path: str,
                 tables_path: str, device: str, trace_path: Path,
                 config: InformedPolicyConfig | None = None,
                 policy_id: str = "informed-one-ply", module=None, native=None, model=None,
                 capture_admission_path: Path | None = None):
        if os.environ.get("POKEZERO_ENUMERATE_ROLLS") != "1":
            raise ExactSearchRefused("P2 requires fresh-process exact damage-roll opt-in")
        self.policy_id = policy_id
        self.model_device = device
        self.config = config or InformedPolicyConfig()
        self._adapter = EngineMctsPolicy(dex=dex, set_source=set_source, module=module,
            policy_id=policy_id, config=EngineMctsConfig(leaf_eval="model", worlds=4,
                sample_retry_factor=1, strict_fallbacks=True, search_depth=1,
                checkpoint_path=checkpoint_path, model_path=model_path,
                tables_path=tables_path, model_device=device))
        if native is None:
            import pokezero_search as native
        self._native = native
        self._model = model
        # Exclusive per-shard trace creation is deliberately before any game.
        # A collection owner supplies the existing parent and a fresh filename.
        self._quarantine = Quarantine(
            os.environ.get("POKEZERO_QUARANTINE_DIR") or Path(trace_path).parent / "quarantine",
            build=runtime_build(native))
        self._capture_admission = require_capture_admission(
            capture_admission_path or os.environ.get("POKEZERO_CAPTURE_ADMISSION"),
            build=self._quarantine.build)
        self._trace = Path(trace_path).open("x",encoding="utf-8")
        self.stats = InformedPolicyStats()
        self._witness: dict[str, Any] = {}
        # Process-start initialization, outside every decision clock. Apply
        # thread settings before native model work, then execute a fixed three
        # dummy calls without using the game's RNG or any source root.
        try:
            from .neural_policy import require_torch
            require_torch()
            layout = json.loads(self._adapter._tables_json)['layout']
            self._observation_spec = dict(window=1,tokens=int(layout['token_count']),
                categorical_features=int(layout['categorical_feature_count']),
                numeric_features=int(layout['numeric_feature_count']))
            self._model = FixedShapeModel(model if model is not None else self._adapter._native(),
                self._observation_spec,self.config.batch_size)
        except BaseException:
            self._trace.close()
            raise

    def reset(self):
        self._adapter.reset()
        self._witness = {}

    def close(self):
        self._trace.close()

    def select_action(self, observation, *, rng: random.Random):
        raise ExactSearchRefused("P2 requires complete public decision context")

    def _positions(self, context: PolicyContext, rng: random.Random):
        adapter, native = self._adapter, self._native
        if context.public_materialization_state is None:
            raise ExactSearchRefused("P2 has no public materialization state")
        adapter._validate_model_root_observation(context.observation)
        fold = adapter._advance_live_fold(context)
        if fold is None:
            raise ExactSearchRefused("P2 public root fold refused")
        root_inputs = adapter._root_inputs_json(context)
        fold_payload = fold.to_payload()
        self._witness.update(root_inputs=root_inputs, fold_payload=fold_payload, draws=[])
        blocked, encored, removed, items, transformed = adapter._public_effect_signals(context)
        recharging, truant = adapter._recharging_slots(context), adapter._truant_loaf_slots(context)
        turn = int(context.public_materialization_state.replay.turn_number or 0)
        positions = []
        for ordinal in range(self.config.worlds):
            draw = {"ordinal":ordinal,"rng_state_before_draw":rng.getstate()}
            self._witness["draws"].append(draw)
            if current():
                current().update(world_ordinal=ordinal, rng_state_before_draw=draw["rng_state_before_draw"],
                    player_teams=unavailable("belief draw has not returned"),
                    world_spec=unavailable("this world has not been constructed"),
                    engine_state=unavailable("this world has not been constructed"),
                    instruction_list=unavailable("this world's transition has not been generated"))
            adapter.stats.worlds_attempted += 1
            override, reason = _gen3_randbat_belief_start_override_result(
                context=context,set_source=adapter._set_source,rng=rng,witnessed_fallback=True)
            if override is None:
                draw["refusal"] = reason
                error = ExactSearchRefused(f"P2 belief draw {ordinal} refused: {reason}")
                error.failure_codes = ["belief." + str(reason or "unknown")]
                raise error
            draw["player_teams"] = dict(override.player_teams)
            if current():
                current().update(world_ordinal=ordinal, rng_state_before_draw=draw["rng_state_before_draw"],
                                 player_teams=draw["player_teams"])
            world = world_battle_spec(context.public_materialization_state,override,
                dex=adapter._dex, approximate_sleep_turns=True,
                approximate_substitute_health=True, approximate_partial_trap_turns=True,
                approximate_hidden_duration_volatiles=True,
                blocked_slots=blocked, encored_moves=encored, removed_item_species=removed,
                current_item_overrides=items, transformed_slots=transformed,
                recharging_slots=recharging, truant_slots=truant, rng=rng)
            state = build_poke_engine_state(world.spec,module=adapter._module)
            side = world.slot_sides[context.player_id]
            validate_encoder_frame(root_inputs,side)
            ctx = {s:list(world.party_species[s]) for s in ("p1","p2")}
            ctx["turn"] = turn
            ctx["toxic_stage_zero_after_upkeep"] = _root_toxic_zero_after_upkeep_attestation(
                context.public_materialization_state.replay)
            state_str = state.to_string()
            draw.update(state_str=state_str,ctx=ctx,self_side=side)
            if current():
                current().update(world_spec=world.spec, engine_state=state_str,
                                 event_context=ctx, root_inputs=root_inputs, fold_payload=fold_payload)
            encoder = native.LeafEncoder(adapter._tables_json,root_inputs,json.dumps(ctx),state_str)
            positions.append(NativePosition(state_str,
                {s:tuple(world.party_species[s]) for s in ("p1","p2")},
                turn,native.FoldState.from_payload(fold_payload),encoder,side,
                origin=snapshot(dict(world_ordinal=ordinal, world_spec=world.spec,
                    player_teams=draw["player_teams"], rng_state_before_draw=draw["rng_state_before_draw"],
                    event_context=ctx, **({"world.construct":current().fields.get("world.construct")}
                                        if current() else {})))))
        return positions, turn

    def _record(self, row):
        self._trace.write(json.dumps(row,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
        self._trace.flush()
        os.fsync(self._trace.fileno())

    def record_failure(self, stage: str, error: BaseException):
        """A bridge/owner-side failure is not a clean partial or completed run."""
        self.stats.refusals += 1
        self.stats.refusal_reasons[stage+":"+type(error).__name__] += 1
        row = dict(schema="pokezero.informed-root-run-failure.v1",status="REFUSED",
                   stage=stage,error_type=type(error).__name__,error=str(error),
                   error_traceback="".join(traceback.format_exception(error)))
        self._record(row)
        self.stats.last_decision = row

    @captured_decision
    def select_action_with_context(self, context: PolicyContext, *, rng: random.Random):
        started = time.perf_counter()
        self.stats.decisions += 1
        row = dict(schema="pokezero.informed-root-decision.v1",battle_id=context.battle_id,
                   game_seed=context.seed,player_id=context.player_id,
                   decision_index=context.decision_round_index,
                   information_model="informed_adversary_root_minimax",
                   leaf_model_device=self.model_device,
                   startup_warmup=self._model.warmup,
                   config=asdict(self.config))
        self._witness = {}
        backend = None
        preparation_seconds = None
        try:
            positions, turn = self._positions(context,rng)
            side = {"p1":0,"p2":1}.get(context.player_id)
            if side is None:
                raise ExactSearchRefused("P2 requires a p1/p2 actor")
            vocabulary = self._adapter._choice_vocabulary(context)
            if vocabulary is None:
                raise ExactSearchRefused("P2 request lacks action candidates")
            backend = NativeExactBackend(self._native,self._model,packed_leaves=True,
                bulk_encode_workers=self.config.bulk_encode_workers,
                max_outcomes_per_call=self.config.max_chance_branches,
                observation_spec=self._observation_spec)
            backend.prepare_roots(positions)
            # Check EVERY own root action against the real request before
            # expansion; never condition on whether a sampled action maps.
            legal = {i for i,p in enumerate(context.observation.legal_action_mask) if p}
            for position in positions:
                own = backend.actions(position,root=True)[side]
                mapped = [vocabulary.action_index(a) for a in own]
                if None in mapped or len(set(mapped)) != len(mapped) or set(mapped) != legal:
                    raise ExactSearchRefused("P2 native/request own action surface disagrees")
            remaining = self.config.max_seconds-(time.perf_counter()-started)
            preparation_seconds = time.perf_counter()-started
            if remaining <= 0:
                raise ExactSearchRefused("P2 decision deadline reached during world construction")
            result = exact_informed_one_ply(positions,backend,acting_side=side,
                batch_size=self.config.batch_size,max_leaves=self.config.max_leaves,
                max_chance_branches=self.config.max_chance_branches,
                max_joint_actions=self.config.max_joint_actions,max_seconds=remaining,
                solver_tolerance=self.config.solver_tolerance,cross_cell_batches=True)
            sampling = sample_informed_equilibrium(result,game_seed=context.seed,turn=turn,
                                                   decision_index=context.decision_round_index)
            action = vocabulary.action_index(sampling["action"])
            if action not in legal:
                raise ExactSearchRefused("P2 sampled semantic action is not legal in request")
            probability = sampling["mix"][sampling["action_index"]]
            row.update(status="COMPLETE",turn=turn,sampling=sampling,request_action_index=action,
                result=asdict(result),certificate_gap=result.equilibrium.gap,
                stage_seconds=dict(backend.stage_seconds),
                model_call_timings=list(backend.model_call_timings),
                root_state_sha256=[sha256(p.state_str.encode()).hexdigest() for p in positions],
                elapsed_seconds=time.perf_counter()-started)
            self._record(row)
            self.stats.completed_decisions += 1
            self.stats.last_decision = dict(status="COMPLETE",game_seed=context.seed,turn=turn,
                decision_index=context.decision_round_index,sampling=sampling,
                elapsed_seconds=row["elapsed_seconds"],certificate_gap=result.equilibrium.gap)
            return PolicyDecision(action_index=action,policy_id=self.policy_id,
                action_probability=probability,metadata={"informed_root":self.stats.last_decision})
        except BaseException as error:
            # PyO3 panics (and owner interruptions) can bypass Exception. Save
            # the actual sampled decision before re-raising; NEVER recover it.
            if isinstance(error, NativeRenderRefused):
                self._witness["native_render_refusal"] = error.witness
            diagnostics = dict(
                before_failure_capture_seconds=time.perf_counter()-started,
                preparation_seconds=preparation_seconds,
                exact_work=getattr(error,'work',unavailable('core work counters not attached to this failure')),
                backend_stage_seconds=dict(backend.stage_seconds) if backend is not None else {},
                model_call_timings=list(backend.model_call_timings) if backend is not None else [],
                backend_counts={key:getattr(backend,key,None) for key in
                    ('encode_calls','forward_batches','unique_leaf_evaluations',
                     'capture_context_updates','captured_leaf_rows')},
                timing_scope='Backend categories can overlap; capture_context_snapshot overlaps fold_history_advance. '
                             'Failure serialization and filesystem synchronization occur after this snapshot.')
            if current():
                current().update(decision_diagnostics=diagnostics)
            witness = error.witness if isinstance(error, NativeRenderRefused) else None
            codes = (["renderer." + code for code in
                      (list(witness.get("lossy_subcases") or []) + list(witness.get("lossy") or [])
                       + list(witness.get("attribution_unsafe_reasons") or []))]
                     if witness else getattr(error,'failure_codes',None) or ["p2.decision.contract_failure"])
            receipt = record_refusal("p2.decision", error, codes=codes, witness=witness)
            self.stats.refusals += 1
            self.stats.refusal_reasons.update(receipt["reason_codes"] if receipt else codes)
            row.update(status="REFUSED",error_type=type(error).__name__,error=str(error),
                error_traceback=traceback.format_exc(),failure_witness=self._witness,
                quarantine_receipt=receipt,capture_telemetry=self._quarantine.telemetry(),
                failure_diagnostics=diagnostics,
                elapsed_seconds=time.perf_counter()-started)
            self._record(row)
            self.stats.last_decision = {k:row[k] for k in
                ("status","game_seed","decision_index","error_type","error","elapsed_seconds")}
            raise
        finally:
            self.stats.decision_wall_seconds += time.perf_counter()-started
            self._witness = {}
