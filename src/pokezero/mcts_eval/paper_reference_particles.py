"""Default-off bootstrap filtering of hypothetical public-history prefixes.

The default transition proposal is the ORIGINAL champion policy/chance law.
An explicit opt-in defensive action-guided mixture carries pi(action)/q(action)
weights and retains full policy support. Chance remains unchanged. Multinomial
resampling retains multiplicity; it does not give each feasible team equal weight.
The final potential also requires the exact actor/public root. This targets the original joint conditioning law,
but finite populations and reused empirical draws are APPROXIMATIONS, not exact
posterior samples or a paper-prescribed reconstruction algorithm.

Every stage must finish before its survivors can be used. Zero survivors or a
partial/deadline-censored population REFUSES, with no retry or first-match bank.
Snapshots belong only to explicitly hypothetical worlds, never the live game.
"""
from dataclasses import dataclass
import copy
import math
import time

from .paper_reference import ReferenceRefusal, SamplingDeadlineExceeded
from .paper_reference_pending import public_history
from .paper_reference_showdown import ShowdownTrajectoryWorld, decision_state
from .paper_reference_conditioning_metrics import phase


def validate_particle_count(value):
    if type(value) is not int or (value != 0 and not 16 <= value <= 256):
        raise ReferenceRefusal('history particles require explicit integer 0 or 16..256')
    return value


def release_unique(particles, release):
    """Resampling aliases immutable particles; release each owner exactly once."""
    errors, seen = [], set()
    for particle in particles:
        if id(particle) in seen:
            continue
        seen.add(id(particle))
        try:
            release(particle)
        except BaseException as error:
            errors.append(error)
    if errors:
        raise errors[0]


@dataclass(frozen=True)
class WeightedAdvance:
    particle: object
    weight: float


def history_failure_diagnostic(error, receipt):
    """Retain the ORIGINAL cancellation certificate, never invent clock facts.

    The whole-game validator needs the authoritative top-level deadline schema.
    Extra population diagnostics must not hide its zero-backup/clock evidence.
    Missing, inconsistent or non-deadline evidence remains a nonbankable failure.
    This changes receipts only, not proposals, weights, cancellation or cleanup.
    """
    cause = copy.deepcopy(getattr(error, 'sampling_diagnostic', None))
    population = copy.deepcopy(receipt)
    clock = cause
    # Follow only our declared diagnostic wrappers, with a fixed bound. Never
    # infer a timestamp from elapsed work or search arbitrary receipt branches.
    for _ in range(32):
        if (not isinstance(clock, dict) or clock.get('schema') not in (
                'pokezero.fixed-chance-pool-failure.v1',
                'pokezero.bootstrap-history-conditioning-failure.v1')):
            break
        clock = clock.get('cause_sampling_diagnostic')
    if isinstance(error, SamplingDeadlineExceeded) and isinstance(clock, dict):
        checked, deadline = clock.get('checked_at'), clock.get('deadline_at')
        if (clock.get('schema') == 'pokezero.world-sampling-deadline.v1'
                and type(checked) in (int, float) and type(deadline) in (int, float)
                and math.isfinite(checked) and math.isfinite(deadline) and checked >= deadline
                and clock.get('accepted_world') is False and clock.get('backed_up') is False):
            return dict(clock, bootstrap_population=population, cause_sampling_diagnostic=cause)
    return dict(schema='pokezero.bootstrap-history-conditioning-failure.v1',
        bootstrap_population=population, cause_sampling_diagnostic=cause)


def bootstrap_population(*, count, stages, initial, advance, release, rng, check, receipt,
                         frontier=None, frontier_weights=None, stage_offset=0):
    """Complete fixed-size propagate/filter/resample stages; return final owners.

    ``advance`` returns a new owned particle, WeightedAdvance, or None (zero
    observation weight). WeightedAdvance carries the target/proposal correction.
    It must never mutate an input particle. Callbacks must clean up resources if
    they raise before returning. Native/policy errors are NOT zero likelihood.
    """
    if (type(count) is not int or count < 1 or type(stages) is not int or stages < 1):
        raise ReferenceRefusal('invalid fixed bootstrap population/stage count')
    current, partial, weights = ([] if frontier is None else list(frontier)), [], []
    completed = False
    receipt.update(population=count, initialized=0, stages=[], complete=False,
                   empirical_law=True, exact_posterior=False, partial_stage_used=False)
    try:
        if frontier is None:
            if stage_offset != 0 or frontier_weights is not None:
                raise ReferenceRefusal('cold population cannot have retained prefix weights/offset')
            for ordinal in range(count):
                check()
                current.append(initial(ordinal))
                receipt['initialized'] += 1
                check()
        else:
            if (type(stage_offset) is not int or stage_offset < 1 or not current
                    or frontier_weights is None or len(frontier_weights) != len(current)
                    or any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0
                           for w in frontier_weights)
                    or not math.isclose(math.fsum(frontier_weights), 1., rel_tol=1e-12)):
                raise ReferenceRefusal('retained complete prefix has invalid final weights/offset')
            check()
            old = current
            current = rng.choices(old, weights=frontier_weights, k=count)
            retained = {id(p) for p in current}
            release_unique([p for p in old if id(p) not in retained], release)
            receipt.update(initialized=count, fresh_initialization_calls=0,
                retained_prefix_resampling='weighted multinomial with multiplicity')
            check()
        for stage in range(stage_offset, stage_offset + stages):
            row = dict(stage=stage, input_particles=count, attempted=0,
                       survivors=0, stage_complete=False, resampled=False)
            receipt['stages'].append(row)
            for particle in current:
                check()
                row['attempted'] += 1
                child = advance(particle, stage)
                if child is not None:
                    weight = child.weight if isinstance(child, WeightedAdvance) else 1.
                    child = child.particle if isinstance(child, WeightedAdvance) else child
                    partial.append(child)
                    if (type(weight) not in (float, int) or not math.isfinite(weight) or weight <= 0):
                        raise ReferenceRefusal('surviving particle has invalid importance weight')
                    weights.append(weight)
                    row['survivors'] += 1
                check()
            total = math.fsum(weights)
            row.update(stage_complete=True,
                empirical_incremental_likelihood=total/count,
                pre_resampling_ess=(total**2/math.fsum(w*w for w in weights) if weights else 0),
                survivor_importance_weights=list(weights))
            if not partial:
                error = ReferenceRefusal('history particle population has zero matching survivors')
                error.sampling_diagnostic = copy.deepcopy(receipt)
                raise error
            old, current = current, []
            release_unique(old, release)
            if stage + 1 == stage_offset + stages:
                receipt['final_normalized_weights'] = [w/total for w in weights]
                current, partial = partial, []
            else:
                # Retain multiplicity, with target/proposal weights. Equal
                # indicator weights preserve the prior prototype's RNG path.
                current = ([rng.choice(partial) for _ in range(count)]
                    if len(set(weights)) == 1 else rng.choices(partial, weights=weights, k=count))
                retained = {id(p) for p in current}
                discarded, partial = [p for p in partial if id(p) not in retained], []
                release_unique(discarded, release)
                row['resampled'] = True
            weights = []
        check()
        receipt.update(complete=True, final_particles=len(current))
        completed = True
        return current
    except BaseException as error:
        error.sampling_diagnostic = history_failure_diagnostic(error, receipt)
        raise
    finally:
        if not completed:
            release_unique([*current, *partial], release)


@dataclass(frozen=True)
class HistoryParticle:
    snapshot: object
    ancestor: int
    anchor_receipt: dict
    steps: tuple
    substitute_hp: dict | None = None
    initial_importance_weight: float = 1.


class HypotheticalHistoryPopulation:
    """One explicit finite belief population, owned by one prepared decision."""
    def __init__(self, factory):
        self.factory = factory
        self.particles = []
        self.receipt = dict(schema='pokezero.bootstrap-history-conditioning.v1',
            proposal=('defensive public-action-guided policy mixture; original chance law'
                if getattr(factory, 'guide_history_actions', False)
                else 'original hidden-anchor/full-champion-policy/chance law'),
            action_importance_correction=bool(getattr(factory, 'guide_history_actions', False)),
            fixed_chance_trials=getattr(factory, 'history_chance_pool', 1),
            batched_history_chance=bool(getattr(factory, 'batch_history_chance', False)),
            chance_allocation=('32 for observed critical-hit round, declared K otherwise; fixed before proposals'
                if getattr(factory, 'batch_history_chance', False) else 'same declared K for every stage'),
            chance_weight=('sum matching original-bank/proposal importance weights / fixed pool size'
                if getattr(factory,'guide_history_chance',False)
                else 'number of matching original chance draws / fixed pool size'),
            chance_pools=[],
            initial_anchor_seed_weights=[],
            initial_anchor_proposals=[],
            seed_banks=[],chance_pilot_traces=[],
            guided_history_chance=bool(getattr(factory, 'guide_history_chance', False)),
            guided_history_accuracy=bool(getattr(factory, 'guide_history_accuracy', False)),
            observation_potential='exact public prefix; final exact actor/public root',
            resampling='multinomial with multiplicity after each complete nonfinal stage',
            posterior_approximation='finite particle population; genealogy can collapse',
            fresh_anchor_per_trajectory=False, independent_posterior_worlds=False,
            live_hidden_state_used=False, live_opponent_action_used=False,
            cohort_adoption=False, strength_inference=False)
        self.draws = 0
        self._retained_frontier = []
        self._retained_weights = None
        self._stage_offset = 0

    def adopt_complete_prefix(self, previous):
        """Transfer only a strictly extended, same-anchor owned complete bank.

        This is sequential finite filtering, not a claim of bit-identical cold
        draws or exact posterior samples. Earlier accepted own-root constraints
        remain part of the information history; only NEW public transitions
        receive new observation/proposal weights. No live opponent choices or
        numeric live chance seeds enter this bank.
        """
        from .paper_reference_prefix_frontier import compatible_complete_prefix
        if not compatible_complete_prefix(previous, self.factory):
            return False
        import hashlib
        import json
        from dataclasses import asdict
        parent = copy.deepcopy(previous.receipt)
        parent_sha = hashlib.sha256(json.dumps(parent, sort_keys=True,
            separators=(',', ':'), allow_nan=False, default=asdict).encode()).hexdigest()
        offset = len(previous.factory.pending_transition.continuation_actions) + 1
        self.receipt = parent
        self.receipt['retained_previous_stages'] = copy.deepcopy(parent['stages'])
        self.receipt['retained_conditioning_prefix'] = dict(
            schema='pokezero.complete-history-prefix-frontier.v1',
            parent_population_receipt_sha256=parent_sha,
            parent_actor_root_key=previous.factory.root.key.hex(),
            parent_public_history=list(public_history(previous.factory.state)),
            completed_boundaries=offset, final_weights=list(parent['final_normalized_weights']),
            previous_own_root_information_retained=True, fresh_anchor_draws=0,
            cold_stream_equivalence=False, exact_posterior=False,
            likelihood_scope='old complete empirical posterior; weight only new public/own-root observations')
        self._retained_weights = list(parent['final_normalized_weights'])
        self._stage_offset = offset
        self._retained_frontier, previous.particles = previous.particles, []
        self.receipt['complete'] = False
        return True

    def _release(self, particle):
        if not self.factory.env.release_search_snapshot(particle.snapshot):
            raise ReferenceRefusal('owned hypothetical particle snapshot disappeared')

    def close(self):
        particles, self.particles, self._retained_frontier = (
            [*self.particles, *self._retained_frontier], [], [])
        release_unique(particles, self._release)

    def _build(self, rng):
        from .paper_reference_factory import PublicRootWorldFactory
        factory = self.factory
        transition = factory.pending_transition
        prior = PublicRootWorldFactory(env=factory.env, state=transition.before_state,
            observation=transition.before_observation, evaluator=factory.evaluator,
            set_source=factory.set_source,
            allow_earlier_compatible_template=factory.allow_earlier_compatible_template,
            max_known_set_draws=factory.max_known_set_draws,
            pending_transition=transition.prior_transition)
        prior.bind_sampling_deadline(factory.sampling_deadline_at)
        self.receipt['nested_anchor_kernel'] = ('retained complete empirical posterior; no new anchor draws'
            if self._stage_offset else 'unchanged original kernel')
        if getattr(factory, 'collect_phase_timing', False):
            self.receipt['phase_timing'] = {}
        subject = factory.state.player_id
        opponent = 'p2' if subject == 'p1' else 'p1'
        expected = public_history(factory.state)
        own_actions = (transition.own_action, *transition.continuation_actions)
        # Membership is only a necessary observation predicate. Preserve the
        # ORIGINAL complete proposal, without forcing later traits or sets.
        from .paper_reference_particle_party import condition_necessary_party
        if not self._stage_offset:
            self.receipt['necessary_party'] = condition_necessary_party(
                prior, expected, opponent, factory.check_sampling_deadline,
                membership_first=getattr(factory, 'membership_first', False),
                native_membership_batch_size=getattr(factory, 'native_membership_batch_size', 0),
                deadline_at=factory.sampling_deadline_at)

        constraints=None
        if (not self._stage_offset and getattr(factory,'public_anchor_constraints',False)
                and prior.pending_transition is None):
            from .paper_reference_anchor_constraints import prepare_constraints, anchor_matches
            constraints=prepare_constraints(transition.before_state,expected,subject)
            self.receipt['necessary_public_anchor']=constraints
            if getattr(factory,'guide_anchor_genders',False):
                prior.anchor_gender_guidance_rows=copy.deepcopy(constraints['static_details'])
                constraints['anchor_gender_guidance']='full-support original 32-bit seed-bank proposal; initial ratio applied once'
            self.receipt['nested_anchor_kernel']='original anchor conditional on necessary public static details/guarded own Encore end'
            lock=constraints['own_encore']
            if lock is not None:
                prior.necessary_public_encore_support={lock['side']:lock['required_remaining']}
                constraints['direct_independent_timer_conditioning']=True

        def original_initial(ordinal):
            world, snapshot = prior(rng), None
            try:
                anchor = copy.deepcopy(prior.receipts[-1])
                if constraints is not None:
                    constraints['complete_proposals']+=1
                    factory.check_sampling_deadline()
                    reason=anchor_matches(factory.env,anchor,constraints)
                    if reason is not None:
                        counts=constraints['rejections']; counts[reason]=counts.get(reason,0)+1
                        return None
                    constraints['matches']+=1
                snapshot = factory.env.snapshot_for_search()
                from .paper_reference_anchor_seed import initial_anchor_weight
                particle = HistoryParticle(snapshot, ordinal, anchor, (),
                    initial_importance_weight=initial_anchor_weight(anchor))
                self.receipt['initial_anchor_seed_weights'].append(particle.initial_importance_weight)
                from .paper_reference_initial_ledger import retain_initial_anchor
                retain_initial_anchor(self.receipt,ordinal=ordinal,anchor=anchor,
                    importance_weight=particle.initial_importance_weight)
                world.close()
                return particle
            except BaseException:
                if snapshot is not None:
                    factory.env.release_search_snapshot(snapshot)
                raise
            finally:
                world.close()

        def initial(ordinal):
            for _ in range(2048 if constraints is not None else 1):
                factory.check_sampling_deadline()
                with phase(self.receipt, 'initial_anchor_inclusive'):
                    particle=original_initial(ordinal)
                if particle is not None: return particle
            raise ReferenceRefusal('necessary public anchor exhausted fixed original-proposal cap')

        from .paper_reference_showdown import ChampionEvaluator
        from .paper_reference_stage_memo import StageIdentityMemo
        from .paper_reference_stage_inputs import build_stage_inputs
        memo_receipt=dict(enabled=type(factory.evaluator) is ChampionEvaluator,
            deterministic_one_snapshot_champion_only=True,
            fresh_action_and_chance_draws=True,
            cached_components=['paired snapshot projection', 'own legality',
                'opponent observation and canonical priors', 'pre-trial public history'],
            native_pilots_and_transitions_cached=False)
        self.receipt["historical_input_memo"]=memo_receipt
        policy_memo=StageIdentityMemo(memo_receipt)

        def advance(particle, stage):
            env = factory.env
            own = own_actions[stage]
            def build():
                return build_stage_inputs(env=env,particle=particle,
                    subject=subject,opponent=opponent,own=own,evaluator=factory.evaluator,
                    need_history=bool(getattr(factory,'guide_history_actions',False)
                        or getattr(factory,'guide_history_chance',False)
                        or getattr(factory,'batch_history_chance',False)))
            with phase(self.receipt, 'stage_inputs_inclusive'):
                inputs = (policy_memo.get(particle,stage,build) if memo_receipt['enabled'] else build())
            if inputs is None:
                return None
            requested = inputs.requested
            actions = {}
            if own is not None:
                actions[subject] = own
            legal, probabilities, opponent_action = (), (), None
            importance_weight, proposal_receipt = 1., None
            if opponent in requested:
                opponent_observation = inputs.opponent_observation
                legal, probabilities = inputs.legal, inputs.probabilities
                if (not legal or len(legal) != len(probabilities)
                        or any(not math.isfinite(p) or p < 0 for p in probabilities)
                        or sum(probabilities) <= 0):
                    raise ReferenceRefusal('particle full champion policy invalid')
                if getattr(factory, 'guide_history_actions', False):
                    from .paper_reference_action_proposal import (
                        defensive_action_proposal, public_guidance_actions)
                    history = inputs.history
                    compatible, guidance = public_guidance_actions(opponent_observation,
                        legal, history, expected, opponent)
                    prior, proposal, proposal_receipt = defensive_action_proposal(
                        legal, probabilities, compatible)
                    opponent_action = rng.choices(legal, weights=proposal, k=1)[0]
                    index = tuple(legal).index(opponent_action)
                    importance_weight = prior[index]/proposal[index]
                    proposal_receipt.update(guidance=guidance, original_probabilities=list(prior),
                        proposal_probabilities=list(proposal), selected_original_probability=prior[index],
                        selected_proposal_probability=proposal[index], importance_weight=importance_weight)
                else:
                    opponent_action = rng.choices(legal, weights=probabilities, k=1)[0]
                actions[opponent] = opponent_action
            if not requested or set(actions) != requested:
                return None
            from .paper_reference_chance_pool import fixed_chance_pool
            pool_receipt=dict(stage=stage,ancestor=particle.ancestor,
                initial_anchor_importance_weight=particle.initial_importance_weight,
                rejection_reasons={},first_public_mismatches=[])
            # Save BEFORE any pilot/trial, so zero-survivor refusals retain the
            # proposal context too. This consumes no RNG and changes no weight.
            history = inputs.history
            history_available = bool(getattr(factory, 'guide_history_actions', False)
                or getattr(factory, 'guide_history_chance', False)
                or getattr(factory, 'batch_history_chance', False))
            metadata = getattr(inputs.opponent_observation, 'metadata', {})
            candidates = metadata.get('action_candidates') if isinstance(metadata, dict) else None
            candidate_rows = ([dict(action_index=row.get('action_index'), kind=row.get('kind'),
                move_id=row.get('move_id'), switched_species=row.get('switched_species')
                    or (row.get('pokemon', {}).get('species') if isinstance(row.get('pokemon'), dict) else None))
                for row in candidates if isinstance(row, dict) and row.get('action_index') in legal]
                if isinstance(candidates, (list, tuple)) else None)
            pool_receipt['pre_trial_proposal'] = dict(
                history_available=history_available,
                history_length=len(history) if history_available else None,
                history_tail=list(history[-2:]) if history_available else [],
                expected_suffix=list(expected[len(history):len(history)+8]) if history_available else [],
                expected_suffix_length=max(0, len(expected)-len(history)) if history_available else None,
                own_action=own, opponent_action=opponent_action,
                opponent_legal=list(legal), opponent_priors=list(probabilities),
                opponent_action_candidates=candidate_rows,
                action_proposal=proposal_receipt)
            self.receipt['chance_pools'].append(pool_receipt)
            hints = None
            if getattr(factory, 'guide_history_chance', False):
                from .paper_reference_seed_bank import (chance_hints, validate_native_trace,
                    critical_alignment_hints, draw_guided_seed)
                history = inputs.history
                def pilot_trace(pilot_seed, purpose):
                    from ..local_showdown import LocalShowdownEnv
                    # Only the canonical auxiliary-only path can skip unused
                    # Python projections. Every real trial restores this owned
                    # snapshot atomically before reading/accepting its outcome.
                    auxiliary_options = ({'auxiliary_trace_only': True}
                        if type(env) is LocalShowdownEnv else {})
                    with phase(self.receipt, 'native_chance_pilot'):
                        pilot = env.conditioning_batch_from_search_snapshot(particle.snapshot, actions,
                            chance_seeds=[pilot_seed], expected_history=expected,
                            deadline_at=factory.sampling_deadline_at, fixed_pool=True, trace_chance=True,
                            **auxiliary_options)
                    factory.check_sampling_deadline()
                    if pilot['consumed'] != 1 or pilot['deadline_reached']:
                        raise ReferenceRefusal('historical chance pilot incomplete')
                    trace = pilot['chance_traces'][0]
                    validate_native_trace(trace, pilot_seed)
                    self.receipt['chance_pilot_traces'].append(dict(stage=stage,ancestor=particle.ancestor,
                        pilot_seed=pilot_seed,trace=trace,hints=chance_hints(trace,history,expected,
                            guide_accuracy=getattr(factory, 'guide_history_accuracy', False)),
                        purpose=purpose, contributes_observation_weight=False))
                    return trace
                trace = pilot_trace(rng.getrandbits(64), 'original observational pilot')
                alignment = critical_alignment_hints(trace,history,expected)
                if alignment:
                    auxiliary = dict(stage=stage,ancestor=particle.ancestor,
                        role='auxiliary_pilot_only',importance_weight_used_for_transition=False)
                    index=len(self.receipt['seed_banks'])
                    self.receipt['seed_banks'].append(auxiliary)
                    with phase(self.receipt, 'auxiliary_seed_bank'):
                        seed,_=draw_guided_seed(rng,alignment,factory.check_sampling_deadline,auxiliary)
                    trace=pilot_trace(seed,'one bounded original-seed critical-alignment pilot')
                    self.receipt['chance_pilot_traces'][-1]['auxiliary_seed_bank_receipt_index']=index
                hints = chance_hints(trace, history, expected,
                    guide_accuracy=getattr(factory, 'guide_history_accuracy', False)) or None
                pool_receipt['seed_guidance_elided_no_hints'] = hints is None
            def reject(reason,actual=None):
                reasons=pool_receipt['rejection_reasons']
                reasons[reason]=reasons.get(reason,0)+1
                if actual is not None and len(pool_receipt['first_public_mismatches'])<4:
                    index=next((i for i,(e,a) in enumerate(zip(expected,actual)) if e!=a),
                               min(len(expected),len(actual)))
                    pool_receipt['first_public_mismatches'].append(dict(index=index,
                        expected=expected[index] if index<len(expected) else None,
                        hypothetical=actual[index] if index<len(actual) else None))
                return None
            def trial(ordinal, explicit_seed=None):
                seed_receipt, seed_weight = None, 1.
                if hints is not None:
                    from .paper_reference_seed_bank import draw_guided_seed
                    seed_receipt = dict(stage=stage,ancestor=particle.ancestor,trial=ordinal)
                    index = len(self.receipt['seed_banks'])
                    self.receipt['seed_banks'].append(seed_receipt)
                    with phase(self.receipt, 'transition_seed_bank'):
                        chance_seed, seed_weight = draw_guided_seed(rng,hints,factory.check_sampling_deadline,seed_receipt)
                    seed_receipt = {k:v for k,v in seed_receipt.items() if k not in
                        ('original_seeds','compatible','proposal_probabilities')}
                    seed_receipt['seed_bank_receipt_index'] = index
                else:
                    chance_seed = rng.getrandbits(64) if explicit_seed is None else explicit_seed
                with phase(self.receipt, 'native_chance_transition'):
                    env.step_from_search_snapshot_for_conditioning(particle.snapshot,
                        actions, chance_seed=chance_seed)
                if env.terminal() is not None:
                    return reject('different terminal')
                with phase(self.receipt, 'post_transition_public_capture'):
                    current = env.public_materialization_state(subject)
                    actual = public_history(current)
                if expected[:len(actual)] != actual:
                    return reject('different exact public prefix',actual)
                early_potential=None
                if getattr(factory,'early_encore_potential',False):
                    from .paper_reference_encore_potential import necessary_own_encore_potential
                    with phase(self.receipt, 'early_encore_potential'):
                        accepted,early_potential=necessary_own_encore_potential(env,current,expected,subject)
                    pool_receipt.setdefault('early_own_encore_potentials',[]).append(early_potential)
                    if not accepted:
                        return reject('necessary observed later own Encore expiry')
                hp = None
                if stage + 1 == len(own_actions):
                    if (subject not in env.requested_players() or actual != expected
                            or decision_state(env.observe(subject), player=subject) != factory.root
                            or current.deferred_opponent_action_player != factory.state.deferred_opponent_action_player):
                        return reject('different final exact actor/public root',actual)
                    hp = {side['id']: side['pokemon'][0]['volatiles']['substitute']['hp']
                        for side in env.snapshot().bridge_snapshot['battle']['sides']
                        if 'substitute' in side['pokemon'][0]['volatiles']}
                    if not hp or any(type(v) is not int or v <= 0 for v in hp.values()):
                        raise ReferenceRefusal('particle final hypothetical Substitute HP invalid')
                step = dict(own_action=own, opponent_action=opponent_action,
                    initial_anchor_importance_weight=particle.initial_importance_weight,
                    opponent_legal=list(legal), opponent_priors=list(probabilities), chance_seed=chance_seed,
                    action_proposal=proposal_receipt,chance_pool=pool_receipt,chance_trial_ordinal=ordinal,
                    chance_seed_proposal=seed_receipt)
                if early_potential is not None: step['early_own_encore_potential']=early_potential
                with phase(self.receipt, 'survivor_snapshot'):
                    child = HistoryParticle(env.snapshot_for_search(), particle.ancestor,
                        particle.anchor_receipt, (*particle.steps, step), hp)
                return WeightedAdvance(child, seed_weight) if hints is not None else child
            if getattr(factory, 'batch_history_chance', False):
                from .paper_reference_batched_chance import batched_chance_pool, public_chance_pool_size
                history = inputs.history
                count = public_chance_pool_size(history, expected, getattr(factory, 'history_chance_pool', 1))
                selected = batched_chance_pool(factory=factory, snapshot=particle.snapshot,
                    actions=actions, expected=expected, count=count, trial=trial, rng=rng, receipt=pool_receipt)
            else:
                selected=fixed_chance_pool(count=getattr(factory,'history_chance_pool',1),
                    trial=trial,release=self._release,rng=rng,check=factory.check_sampling_deadline,receipt=pool_receipt)
            if selected is None:return None
            # New children use the default unit initial weight. Resampling
            # already accounts for this initial ratio after the first stage.
            return WeightedAdvance(selected.particle,
                particle.initial_importance_weight*importance_weight*selected.weight)

        began = time.perf_counter()
        frontier, self._retained_frontier = self._retained_frontier, []
        self.particles = bootstrap_population(count=factory.history_particles,
            stages=len(own_actions)-self._stage_offset, initial=initial, advance=advance,
            release=self._release, rng=rng, check=factory.check_sampling_deadline, receipt=self.receipt,
            frontier=frontier if self._stage_offset else None,
            frontier_weights=self._retained_weights, stage_offset=self._stage_offset)
        self.receipt.update(build_seconds=time.perf_counter()-began,
            distinct_anchor_ancestors=len({p.ancestor for p in self.particles}),
            final_actor_root_key=factory.root.key.hex())
        ancestry_weights = {}
        for particle, weight in zip(self.particles, self.receipt['final_normalized_weights']):
            ancestry_weights[particle.ancestor] = ancestry_weights.get(particle.ancestor, 0.) + weight
        self.receipt.update(final_ancestry_weights=ancestry_weights,
            final_ancestry_ess=1/math.fsum(w*w for w in ancestry_weights.values()),
            final_particle_ess=1/math.fsum(w*w for w in self.receipt['final_normalized_weights']))

    def draw(self, rng, evidence):
        if not self.particles:
            try:
                self._build(rng)
            except BaseException as error:
                if getattr(error, 'sampling_diagnostic', None) is None:
                    error.sampling_diagnostic = dict(
                        schema='pokezero.bootstrap-history-conditioning-failure.v1',
                        bootstrap_population=copy.deepcopy(self.receipt))
                raise
        factory = self.factory
        factory.check_sampling_deadline()
        weights = self.receipt['final_normalized_weights']
        ordinal = (rng.randrange(len(self.particles)) if len(set(weights)) == 1
                   else rng.choices(range(len(self.particles)), weights=weights, k=1)[0])
        particle = self.particles[ordinal]
        factory.env.restore_search_snapshot(particle.snapshot)
        subject = factory.state.player_id
        if (decision_state(factory.env.observe(subject), player=subject) != factory.root
                or public_history(factory.env.public_materialization_state(subject)) != public_history(factory.state)):
            raise ReferenceRefusal('cached hypothetical population changed the exact public/actor root')
        factory.check_sampling_deadline()
        self.draws += 1
        from .paper_reference_initial_ledger import initial_ledger_draw_view
        population_receipt=initial_ledger_draw_view(self.receipt,self.draws)
        if self.draws>1:
            # Preserve the full bank diagnostic once, and every selected
            # witness. Repeating every rejected chance pool per trajectory
            # would create transport overhead without new evidence.
            population_receipt.pop('chance_pools',None)
            population_receipt.pop('seed_banks',None)
            population_receipt.pop('chance_pilot_traces',None)
            population_receipt['chance_pools_retained_at_empirical_draw']=1
        evidence.update(status='ROOT_VALIDATED',
            substitute_particle_conditioning=dict(population_receipt, selected_particle=ordinal,
                selected_anchor_ancestor=particle.ancestor, empirical_draw=self.draws,
                anchor=particle.anchor_receipt, steps=particle.steps,
                sampled_substitute_hp=particle.substitute_hp))
        return ShowdownTrajectoryWorld(factory.env, subject=subject, evaluator=factory.evaluator,
            release=lambda: factory._release(evidence))
