"""Public-root-only fresh-world factory for the opt-in trajectory reference.

Never accepts a live environment, simulator snapshot, opponent request, or
historical numeric opponent action. The actor's opening request and public
belief ledger are the only team inputs. Every call samples and materializes a
new hypothetical world and checks exact actor/public observation identity.
Pending Baton Pass commitments require a public-only previous-root certificate
and joint champion-policy/chance conditioning, never the deferred-prior argmax.
"""

from __future__ import annotations

from dataclasses import asdict
import random
import re
import math
import time
from typing import Any

from .paper_reference import ReferenceRefusal
from .paper_reference_sampling import KnownSetTraits, PaperHiddenTeamSampler
from .paper_reference_sleep import induced_sleep_certificates, induced_sleep_support
from .paper_reference_showdown import ChampionEvaluator, ShowdownTrajectoryWorld, decision_state
from ..determinization import _self_team_from_metadata_result, player_belief_view_from_payload
from ..env import BattleStartOverride
from ..local_showdown import LocalShowdownEnv, PublicBattleMaterializationState, _public_reference_encore
from ..public_decision_corpus import _public_belief_view
from ..randbat import Gen3RandbatSource, canonical_gen3_randbat_species_id
from ..showdown import _pokemon_metadata, _self_team_from_request
from ..showdown_fixture import pack_team


def _level(details: Any) -> int | None:
    if not isinstance(details, str):
        return None
    for part in details.split(",")[1:]:
        if re.fullmatch(r"L\d+", part.strip()):
            return int(part.strip()[1:])
    # In Showdown public details, an omitted level is level 100.
    return 100


def _gender(details: Any) -> str | None:
    if isinstance(details, str):
        return next((part.strip() for part in details.split(",")[1:]
            if part.strip() in ("M", "F")), None)
    return None


def _maximum_hp(condition: str | None) -> int | None:
    if isinstance(condition, str):
        match = re.match(r"\d+/(\d+)(?:\s|$)", condition)
        if match and int(match[1]) != 100:
            # The retained corpus sometimes carries exact HP, a controlled
            # information deviation from PS percentage-only observations.
            # Condition the draw on it; never silently change that root.
            return int(match[1])
    return None


class PublicRootWorldFactory:
    def __init__(self, *, env: LocalShowdownEnv, state: PublicBattleMaterializationState,
                 observation: Any, evaluator: ChampionEvaluator, set_source: Gen3RandbatSource,
                 allow_earlier_compatible_template: bool = False,
                 max_known_set_draws: int = 10, pending_transition: Any = None,
                 staged_substitute_conditioning: bool = False,
                 history_particles: int = 0,
                 guide_history_actions: bool = False,
                 history_chance_pool: int = 1,
                 batch_history_chance: bool = False,
                 guide_history_chance: bool = False,
                 guide_history_accuracy: bool = False,
                 membership_first: bool = False,
                 native_membership_batch_size: int = 0,
                 public_anchor_constraints: bool = False,
                 guide_anchor_genders: bool = False,
                 early_encore_potential: bool = False,
                 collect_phase_timing: bool = False, diagnostic_oracle_team=None) -> None:
        if type(collect_phase_timing) is not bool:
            raise ReferenceRefusal('phase timing requires an explicit boolean opt-in')
        self.collect_phase_timing = collect_phase_timing
        from .paper_reference_particles import validate_particle_count
        self.history_particles = validate_particle_count(history_particles)
        from .paper_reference_chance_pool import validate_chance_pool_count
        self.history_chance_pool = validate_chance_pool_count(history_chance_pool,particles=history_particles)
        if type(batch_history_chance) is not bool or batch_history_chance and not history_particles:
            raise ReferenceRefusal('batched historical chance requires explicit particles and boolean opt-in')
        self.batch_history_chance = batch_history_chance
        if type(guide_history_chance) is not bool or guide_history_chance and (not history_particles or batch_history_chance):
            raise ReferenceRefusal('guided historical seeds require particles and no batched chance allocation')
        self.guide_history_chance = guide_history_chance
        if type(guide_history_accuracy) is not bool or guide_history_accuracy and not guide_history_chance:
            raise ReferenceRefusal('accuracy guidance requires explicit guided historical seeds')
        self.guide_history_accuracy = guide_history_accuracy
        if type(membership_first) is not bool or membership_first and not history_particles:
            raise ReferenceRefusal('membership-first conditioning requires explicit particles and boolean opt-in')
        self.membership_first = membership_first
        from .paper_reference_native_membership import validate_native_membership
        self.native_membership_batch_size = validate_native_membership(
            native_membership_batch_size, membership_first=membership_first)
        if type(public_anchor_constraints) is not bool or public_anchor_constraints and not history_particles:
            raise ReferenceRefusal('public anchor constraints require explicit particles and boolean opt-in')
        self.public_anchor_constraints = public_anchor_constraints
        if type(guide_anchor_genders) is not bool or guide_anchor_genders and not public_anchor_constraints:
            raise ReferenceRefusal('anchor gender guidance requires explicit public anchor constraints')
        self.guide_anchor_genders = guide_anchor_genders
        self.anchor_gender_guidance_rows = []
        if (type(early_encore_potential) is not bool
                or early_encore_potential and (not history_particles or not public_anchor_constraints)):
            raise ReferenceRefusal('early Encore potential requires explicit particles and public anchor constraints')
        self.early_encore_potential = early_encore_potential
        # Populated only by a necessary-predicate certificate on an original,
        # nonnested hypothetical anchor; normal/reference defaults are unchanged.
        self.necessary_public_encore_support = {}
        if (type(guide_history_actions) is not bool or guide_history_actions and not history_particles):
            raise ReferenceRefusal('historical action guidance requires explicit particles and boolean opt-in')
        self.guide_history_actions = guide_history_actions
        self.history_population = None
        if type(staged_substitute_conditioning) is not bool:
            raise ReferenceRefusal('constant-chance conditioning requires an explicit boolean opt-in')
        if state.replay.requests:
            raise ReferenceRefusal("public root must strip replay request payloads")
        from .paper_reference_pending import (
            requires_faint_encore_replay, requires_baton_interruption_replay, validate_transition)
        from .paper_reference_substitute import (SubstituteHistoryTransition,
            requires_substitute_replay, validate_substitute_transition)
        if isinstance(pending_transition, SubstituteHistoryTransition) or requires_substitute_replay(state):
            validate_substitute_transition(pending_transition, state, observation, set_source.metadata.source_hash)
        elif (state.deferred_opponent_action_player is not None or requires_faint_encore_replay(state)
                or requires_baton_interruption_replay(state)):
            validate_transition(pending_transition, state, observation, set_source.metadata.source_hash)
        elif pending_transition is not None:
            raise ReferenceRefusal('nonpending public root cannot carry a pending certificate')
        if state.observation_format_id != "gen3randombattle":
            raise ReferenceRefusal("reference factory supports only public Gen 3 random-battle roots")
        if env.belief_set_source_hash != set_source.metadata.source_hash:
            raise ReferenceRefusal("reference environment and public encoder source binding disagree")
        self.root = decision_state(observation, player=state.player_id)
        # Strict actor-only opening request reconstruction avoids mutated live
        # abilities/items and the opponent's real party. No private truth builder.
        rows = [_pokemon_metadata(mon) for mon in
            _self_team_from_request(state.self_initial_request, state.player_id)]
        own_team, failure = _self_team_from_metadata_result(rows, team_size=6, set_source=set_source)
        if own_team is None:
            raise ReferenceRefusal(f"actor-known opening party refuses: {failure}")
        payload = _public_belief_view(observation.metadata)
        view = player_belief_view_from_payload(payload)
        if (view is None or view.self_slot != state.player_id
                or not isinstance(payload.get("opponent_pokemon"), list)
                or len(view.opponent_pokemon) != len(payload["opponent_pokemon"])):
            raise ReferenceRefusal("public opponent belief has invalid or silently dropped members")
        if state.belief_engine.resolved_player_view(state.player_id).to_overlay_payload() != payload:
            raise ReferenceRefusal("public root observation and materialization belief ledger disagree")
        details = {str(row.get("species", "")).casefold(): row.get("details")
            for row in observation.metadata.get("opponent_team", [])}
        known = []
        for mon in view.opponent_pokemon:
            # A mutated held item is CURRENT state, not the generator's original
            # assignment. The public materializer applies current mutations later.
            item = mon.original_public_item
            if item is None and not mon.item_mutated:
                item = mon.revealed_item
            ability = mon.revealed_ability
            universe = set_source.universes.get(canonical_gen3_randbat_species_id(mon.species))
            # Validation only, NOT a draw from the catalog. The compact Dex
            # metadata used by the encoder intentionally omits ability tables.
            original_abilities = {re.sub(r"[^a-z0-9]", "", str(a).lower())
                for a in (v.ability for v in universe.variants)} if universe is not None else set()
            if ability and re.sub(r"[^a-z0-9]", "", ability.lower()) not in original_abilities:
                # A copied Transform/Trace ability is not an original-set trait.
                # Refuse unexpected ledger corruption rather than forcing it.
                if not mon.transformed:
                    raise ReferenceRefusal("public original-set ability is incompatible with species")
                ability = None
            known.append(KnownSetTraits(mon.species, mon.revealed_moves, ability, item,
                _level(details.get(mon.species.casefold())),
                mon.gender or _gender(details.get(mon.species.casefold())), _maximum_hp(mon.condition),
                mon.ruled_out_abilities, mon.ruled_out_items))
        self.known = tuple(known)
        self.env, self.state, self.evaluator = env, state, evaluator
        self.set_source = set_source
        self.pending_transition = pending_transition
        self.allow_earlier_compatible_template = allow_earlier_compatible_template
        self.max_known_set_draws = max_known_set_draws
        self.staged_substitute_conditioning = staged_substitute_conditioning
        self.own_team = own_team
        self.sampler = PaperHiddenTeamSampler(env, set_source=set_source,
            allow_earlier_compatible_template=allow_earlier_compatible_template,
            max_known_set_draws=max_known_set_draws)
        self.diagnostic_oracle_team = None
        if diagnostic_oracle_team is not None:
            self.install_diagnostic_oracle(diagnostic_oracle_team)
        self.active = False
        self.receipts: list[dict[str, Any]] = []
        self.sampling_deadline_at = None
        self.constant_chance_plan = None
        if staged_substitute_conditioning and isinstance(pending_transition, SubstituteHistoryTransition):
            from .paper_reference_staged_chance import build_constant_chance_plan, build_staged_prefix_joint_plan
            from .paper_reference_wake_wrap_chance import build_wake_wrap_plan
            self.constant_chance_plan = build_wake_wrap_plan(self)
            if self.constant_chance_plan is None:
                self.constant_chance_plan = build_constant_chance_plan(self)
            if self.constant_chance_plan is None:
                self.constant_chance_plan = build_staged_prefix_joint_plan(self)

    def install_diagnostic_oracle(self, team):
        from .search_over_raw_oracle import OracleTeamSampler
        if getattr(self, 'active', False) or getattr(self, 'receipts', ()) or getattr(self, 'history_population', None):
            raise ReferenceRefusal('cannot replace a sampled or retained belief with oracle truth')
        self.sampler = OracleTeamSampler(team, self.set_source)
        self.diagnostic_oracle_team = team

    def bind_sampling_deadline(self, deadline):
        if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
            raise ReferenceRefusal('invalid world-sampling deadline')
        if self.active:
            raise ReferenceRefusal('cannot change the deadline of an owned sampled world')
        self.sampling_deadline_at = deadline

    def enable_team_diagnostics(self, root_binding):
        """Explicit prospective instrumentation; never a source-truth input."""
        if (self.active or self.receipts or hasattr(self, 'team_diagnostic_root_binding')
                or type(root_binding) is not str or len(root_binding) != 64
                or any(c not in '0123456789abcdef' for c in root_binding)):
            raise ReferenceRefusal('team diagnostics require a fresh bound public root')
        self.team_diagnostic_root_binding = root_binding
        self.collect_sampled_team_traits = True

    def _observe_team(self, world, evidence):
        if hasattr(self, 'team_diagnostic_root_binding'):
            from .search_over_raw_belief_diagnostics import capture_sampled_team
            try:
                capture_sampled_team(self, world, evidence)
                # Observation is inside the same decision clock. Crossing it
                # cannot turn an unfinished world into accepted search work.
                self.check_sampling_deadline()
            except BaseException:
                world.close()
                evidence.pop('sampled_original_team', None)
                raise
        return world

    def check_sampling_deadline(self):
        from .paper_reference import SamplingDeadlineExceeded
        deadline = self.sampling_deadline_at
        if deadline is not None:
            checked = time.perf_counter()
            if checked >= deadline:
                error = SamplingDeadlineExceeded('unfinished sampled world reached the decision deadline')
                error.sampling_diagnostic = dict(
                    schema='pokezero.world-sampling-deadline.v1',
                    deadline_at=deadline, checked_at=checked,
                    accepted_world=False, backed_up=False,
                )
                raise error

    def __call__(self, hidden_rng: random.Random) -> ShowdownTrajectoryWorld:
        if self.active:
            raise ReferenceRefusal("warm reference world is still owned by another trajectory")
        self.active = True
        evidence: dict[str, Any] = {"ordinal": len(self.receipts), "status": "STARTED"}
        if getattr(self, 'diagnostic_oracle_team', None) is not None:
            from .search_over_raw_oracle import team_sha256
            evidence.update(diagnostic_oracle_team_sha256=team_sha256(self.diagnostic_oracle_team),
                information_scope="original_opponent_team_only")
        self.receipts.append(evidence)
        try:
            self.check_sampling_deadline()
            if self.pending_transition is not None:
                from .paper_reference_substitute import SubstituteHistoryTransition, condition_substitute_world
                if isinstance(self.pending_transition, SubstituteHistoryTransition):
                    if self.history_particles:
                        from .paper_reference_particles import HypotheticalHistoryPopulation
                        if self.history_population is None:
                            self.history_population = HypotheticalHistoryPopulation(self)
                        return self._observe_team(self.history_population.draw(hidden_rng, evidence), evidence)
                    return self._observe_team(condition_substitute_world(self, hidden_rng, evidence), evidence)
                from .paper_reference_pending import condition_pending_world
                return self._observe_team(condition_pending_world(self, hidden_rng, evidence), evidence)
            draw = self.sampler.draw(self.known, hidden_rng)
            evidence.update(packed_team_sha256=draw.packed_team_sha256,
                known_draws=[asdict(row) for row in draw.known], forced_sets=draw.forced_sets,
                unknown_party_seeds=list(draw.unknown_party_seeds))
            if getattr(self, 'collect_sampled_team_traits', False):
                from .search_over_raw_belief_diagnostics import sampled_team_origin
                evidence['sampled_team_origin'] = sampled_team_origin(draw)
            opponent = "p2" if self.state.player_id == "p1" else "p1"
            override = BattleStartOverride(player_teams={self.state.player_id: pack_team(self.own_team),
                opponent: pack_team(draw.team)}, observation_format_id="gen3randombattle")
            if self.anchor_gender_guidance_rows:
                from .paper_reference_anchor_seed import guided_anchor_seed
                seed, proposal = guided_anchor_seed(hidden_rng,
                    {self.state.player_id:self.own_team,opponent:draw.team},self.set_source,
                    self.anchor_gender_guidance_rows,opponent,self.check_sampling_deadline)
                evidence['materialization_seed_proposal']=proposal
            else:
                seed = hidden_rng.getrandbits(32)
            evidence["materialization_seed"] = seed
            durations, conditioning = {}, {}
            for side in ('p1', 'p2'):
                certificate = _public_reference_encore(self.state, side)
                if certificate is not None:
                    support=certificate['remaining_candidates']
                    direct=None
                    if side in self.necessary_public_encore_support:
                        from .paper_reference_anchor_constraints import conditional_encore_support
                        support,direct=conditional_encore_support(support,self.necessary_public_encore_support[side])
                    durations[side] = hidden_rng.choice(support)
                    conditioning[side] = {**certificate, 'sampled_remaining': durations[side]}
                    if direct is not None: conditioning[side]['necessary_public_direct_conditioning']=direct
            evidence['encore_conditioning'] = conditioning
            sleep_draws, sleep_conditioning = {}, {}
            teams = {self.state.player_id: self.own_team, opponent: draw.team}
            for key, certificate in induced_sleep_certificates(self.state).items():
                side, species = key.split(':', 1)
                candidates = [mon for mon in teams[side] if re.sub('[^a-z0-9]', '', mon.species.lower()) == species]
                if len(candidates) != 1:
                    raise ReferenceRefusal('induced sleep cannot match public victim to sampled party')
                support = induced_sleep_support(certificate, candidates[0].ability)
                if not support:
                    raise ReferenceRefusal('induced sleep sampled ability contradicts public survival')
                sleep_draws[key] = hidden_rng.choice(support)
                sleep_conditioning[key] = dict(certificate=certificate, support=support, sampled=sleep_draws[key])
            evidence['induced_sleep_conditioning'] = sleep_conditioning
            self.env.materialize_public_world(state=self.state, start_override=override, seed=seed,
                reference_rest_sleep=True, reference_consumed_items=True,
                reference_encore_durations=durations, reference_induced_sleep=sleep_draws,
                reference_turn_clocks=True, reference_attract=True, reference_yawn=True)
            if decision_state(self.env.observe(self.state.player_id), player=self.state.player_id) != self.root:
                raise ReferenceRefusal("fresh sampled world does not preserve exact player-known root")
            self.check_sampling_deadline()
            evidence["status"] = "ROOT_VALIDATED"
            return self._observe_team(ShowdownTrajectoryWorld(self.env, subject=self.state.player_id,
                evaluator=self.evaluator, release=lambda: self._release(evidence)), evidence)
        except Exception as exc:
            from .paper_reference import SamplingDeadlineExceeded
            self.active = False
            evidence.update(status='DEADLINE_CANCELLED' if isinstance(exc, SamplingDeadlineExceeded)
                            else 'REFUSED', error=f"{type(exc).__name__}: {exc}")
            if getattr(exc, "sampling_diagnostic", None) is not None:
                evidence["sampling_diagnostic"] = exc.sampling_diagnostic
            raise

    def _release(self, evidence: dict[str, Any]) -> None:
        self.active = False
        # Release is ownership accounting, not a completed-trajectory assertion.
        evidence["released"] = True

    def close(self):
        if self.active:
            raise ReferenceRefusal('cannot close a factory while a trajectory owns its world')
        if self.history_population is not None:
            self.history_population.close()
