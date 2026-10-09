"""Spawn-safe public-only champion/Showdown runtime for reference workers."""
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Any

from .paper_reference import ReferenceRefusal
from .paper_reference_parallel import PreparedDecision
from .paper_reference_factory import PublicRootWorldFactory
from .paper_reference_showdown import ChampionEvaluator, decision_state
from .paper_reference_pending import PendingPolicyTransition
from ..local_showdown import PublicBattleMaterializationState
from ..public_decision_corpus import PublicObservation, _public_belief_view


def validated_conditioning_batch_size(value, *, staged):
    """Reject unregistered/meaningless batching before a worker is spawned."""
    if type(staged) is not bool or type(value) is not int or not 1 <= value <= 16:
        raise ReferenceRefusal('conditioning batch size requires an integer 1..16 and boolean staged gate')
    if value != 1 and not staged:
        raise ReferenceRefusal('conditioning batching requires explicit staged conditioning')
    return value


@dataclass(frozen=True)
class PublicRootRequest:
    state: PublicBattleMaterializationState
    observation: Any
    set_source_hash: str
    pending_transition: PendingPolicyTransition | None = None

    @classmethod
    def capture(cls, state, observation, *, pending_transition=None):
        if (not isinstance(state, PublicBattleMaterializationState) or state.replay.requests
                or state.belief_engine.set_source is None):
            raise ReferenceRefusal("reference transport requires public-only, source-bound state")
        # The immutable catalog is independently pinned/warm in every worker.
        # Strip that redundant object, NOT any public evidence or candidate pins.
        engine = state.belief_engine.clone()
        source_hash = engine.set_source.metadata.source_hash
        engine.set_source = None
        sanitized = PublicObservation.from_observation(observation).to_observation(
            belief_view=_public_belief_view(observation.metadata))
        return cls(replace(state, belief_engine=engine), sanitized, source_hash, pending_transition)


@dataclass(frozen=True)
class ShowdownWorkerFactory:
    checkpoint: str
    checkpoint_sha256: str
    showdown_root: str
    set_source_hash: str
    # Explicit bounded completion adaptation; defaults preserve strict forcing.
    allow_earlier_compatible_template: bool = False
    max_known_set_draws: int = 10
    staged_substitute_conditioning: bool = False
    conditioning_batch_size: int = 1
    history_particles: int = 0
    guide_history_actions: bool = False
    history_chance_pool: int = 1
    batch_history_chance: bool = False
    guide_history_chance: bool = False
    guide_history_accuracy: bool = False
    membership_first: bool = False
    native_membership_batch_size: int = 0
    public_anchor_constraints: bool = False
    guide_anchor_genders: bool = False
    early_encore_potential: bool = False
    collect_phase_timing: bool = False
    history_prefix_reuse: bool = False
    history_prefix_public_refresh: bool = False

    def __post_init__(self):
        if type(self.collect_phase_timing) is not bool:
            raise ReferenceRefusal('phase timing requires an explicit boolean opt-in')
        from .paper_reference_particles import validate_particle_count
        from .paper_reference_native_membership import validate_native_membership
        validate_native_membership(self.native_membership_batch_size, membership_first=self.membership_first)
        validate_particle_count(self.history_particles)
        if (type(self.history_prefix_reuse) is not bool
                or self.history_prefix_reuse and not self.history_particles):
            raise ReferenceRefusal('history prefix reuse requires explicit particles and boolean opt-in')
        if (type(self.history_prefix_public_refresh) is not bool
                or self.history_prefix_public_refresh and not self.history_prefix_reuse):
            raise ReferenceRefusal('public prefix refresh requires explicit prefix reuse and boolean opt-in')
        from .paper_reference_chance_pool import validate_chance_pool_count
        validate_chance_pool_count(self.history_chance_pool,particles=self.history_particles)
        if type(self.batch_history_chance) is not bool or self.batch_history_chance and not self.history_particles:
            raise ReferenceRefusal('batched historical chance requires explicit particles and boolean opt-in')
        if type(self.guide_history_chance) is not bool or self.guide_history_chance and (not self.history_particles or self.batch_history_chance):
            raise ReferenceRefusal('guided historical seeds require particles and no batched chance allocation')
        if type(self.guide_history_accuracy) is not bool or self.guide_history_accuracy and not self.guide_history_chance:
            raise ReferenceRefusal('accuracy guidance requires explicit guided historical seeds')
        if type(self.membership_first) is not bool or self.membership_first and not self.history_particles:
            raise ReferenceRefusal('membership-first conditioning requires explicit particles and boolean opt-in')
        if type(self.public_anchor_constraints) is not bool or self.public_anchor_constraints and not self.history_particles:
            raise ReferenceRefusal('public anchor constraints require explicit particles and boolean opt-in')
        if type(self.guide_anchor_genders) is not bool or self.guide_anchor_genders and not self.public_anchor_constraints:
            raise ReferenceRefusal('anchor gender guidance requires explicit public anchor constraints')
        if (type(self.early_encore_potential) is not bool
                or self.early_encore_potential and (not self.history_particles or not self.public_anchor_constraints)):
            raise ReferenceRefusal('early Encore potential requires explicit particles and public anchor constraints')
        if (type(self.guide_history_actions) is not bool
                or self.guide_history_actions and not self.history_particles):
            raise ReferenceRefusal('historical action guidance requires explicit particles and boolean opt-in')
        validated_conditioning_batch_size(self.conditioning_batch_size,
            staged=self.staged_substitute_conditioning)

    def __call__(self, index):
        return _ShowdownRuntime(self, index)


class _ShowdownRuntime:
    def __init__(self, binding, index):
        import torch
        from ..collection import env_config_with_policy_spec_masks
        from ..local_showdown import LocalShowdownConfig, LocalShowdownEnv
        from ..neural_policy import load_transformer_policy
        from ..randbat import load_gen3_randbat_source_cached

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        checkpoint = Path(binding.checkpoint)
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != binding.checkpoint_sha256:
            raise ReferenceRefusal("worker champion checkpoint digest drift")
        source = load_gen3_randbat_source_cached(binding.showdown_root)
        if source.metadata.source_hash != binding.set_source_hash:
            raise ReferenceRefusal("worker Showdown/set-source provenance drift")
        policy = load_transformer_policy(checkpoint, device="cpu", deterministic=True,
            exploration_epsilon=0., sampling_temperature=1., family_gated_selection=False)
        self.evaluator = ChampionEvaluator(policy)
        config = env_config_with_policy_spec_masks(LocalShowdownConfig(
            showdown_root=Path(binding.showdown_root), set_belief_source=True),
            [f"neural:{checkpoint}"], context="opt-in twenty-worker paper reference")
        self.env = LocalShowdownEnv(config)
        self.source = source
        self.allow_earlier_compatible_template = binding.allow_earlier_compatible_template
        self.max_known_set_draws = binding.max_known_set_draws
        self.staged_substitute_conditioning = binding.staged_substitute_conditioning
        self.conditioning_batch_size = binding.conditioning_batch_size
        self.history_particles = binding.history_particles
        self.guide_history_actions = binding.guide_history_actions
        self.history_chance_pool = binding.history_chance_pool
        self.batch_history_chance = binding.batch_history_chance
        self.guide_history_chance = binding.guide_history_chance
        self.guide_history_accuracy = binding.guide_history_accuracy
        self.membership_first = binding.membership_first
        self.native_membership_batch_size = binding.native_membership_batch_size
        self.public_anchor_constraints = binding.public_anchor_constraints
        self.guide_anchor_genders = binding.guide_anchor_genders
        self.early_encore_potential = binding.early_encore_potential
        self.collect_phase_timing = binding.collect_phase_timing
        self.history_prefix_reuse = binding.history_prefix_reuse
        self.history_prefix_public_refresh = binding.history_prefix_public_refresh
        self.prepared_factory = None
        try:
            # Warm bridge startup belongs to the separately measured pool startup.
            # This synthetic startup battle contributes NO trajectory/statistics.
            self.env.reset(seed=2026100400 + index)
        except BaseException:
            self.env.close()
            raise

    def prepare(self, public_request):
        previous_factory = getattr(self, 'prepared_factory', None)
        if previous_factory is not None and not getattr(self, 'history_prefix_reuse', False):
            previous_factory.close()
            self.prepared_factory = None
            previous_factory = None
        if (not isinstance(public_request, PublicRootRequest)
                or public_request.set_source_hash != self.source.metadata.source_hash
                or public_request.state.replay.requests
                or public_request.state.belief_engine.set_source is not None):
            raise ReferenceRefusal("worker received private, malformed or unbound root transport")
        sanitized = PublicObservation.from_observation(public_request.observation).to_observation(
            belief_view=_public_belief_view(public_request.observation.metadata))
        if sanitized.metadata != public_request.observation.metadata:
            raise ReferenceRefusal("worker observation contains nonpublic transport metadata")
        engine = public_request.state.belief_engine.clone()
        engine.set_source = self.source
        state = replace(public_request.state, belief_engine=engine)
        observation = public_request.observation
        pending = public_request.pending_transition
        def bind_pending(pending, depth=0):
            if pending is None:
                return None
            if depth > 200:
                raise ReferenceRefusal('pending transport exceeds bounded ancestry')
            if (not isinstance(pending, PendingPolicyTransition) or pending.before_state.replay.requests
                    or pending.before_state.belief_engine.set_source is not None
                    or pending.set_source_hash != self.source.metadata.source_hash):
                raise ReferenceRefusal('worker received private or unbound pending transport')
            previous_engine = pending.before_state.belief_engine.clone()
            previous_engine.set_source = self.source
            return replace(pending, before_state=replace(pending.before_state, belief_engine=previous_engine),
                           prior_transition=bind_pending(pending.prior_transition, depth+1))
        pending = bind_pending(pending)
        root = decision_state(observation, player=state.player_id)
        factory = PublicRootWorldFactory(env=self.env, state=state, observation=observation,
            evaluator=self.evaluator, set_source=self.source,
            allow_earlier_compatible_template=self.allow_earlier_compatible_template,
            max_known_set_draws=self.max_known_set_draws, pending_transition=pending,
            staged_substitute_conditioning=getattr(self, 'staged_substitute_conditioning', False),
            history_particles=getattr(self, 'history_particles', 0),
            guide_history_actions=getattr(self, 'guide_history_actions', False),
            history_chance_pool=getattr(self, 'history_chance_pool', 1),
            batch_history_chance=getattr(self, 'batch_history_chance', False),
            guide_history_chance=getattr(self, 'guide_history_chance', False),
            guide_history_accuracy=getattr(self, 'guide_history_accuracy', False),
            membership_first=getattr(self, 'membership_first', False),
            native_membership_batch_size=getattr(self, 'native_membership_batch_size', 0),
            public_anchor_constraints=getattr(self, 'public_anchor_constraints', False),
            guide_anchor_genders=getattr(self, 'guide_anchor_genders', False),
            early_encore_potential=getattr(self, 'early_encore_potential', False),
            collect_phase_timing=getattr(self, 'collect_phase_timing', False))
        lifecycle = None
        if getattr(self, 'history_prefix_reuse', False):
            lifecycle = dict(schema='pokezero.conditioning-prefix-lifecycle.v1',
                mode='COLD_NO_COMPATIBLE_BANK', chosen_before_world_sampling=True,
                refresh_enabled=getattr(self, 'history_prefix_public_refresh', False))
        if previous_factory is not None:
            from .paper_reference_particles import HypotheticalHistoryPopulation
            from .paper_reference_prefix_frontier import (compatible_complete_prefix,
                public_refresh_certificate)
            population = HypotheticalHistoryPopulation(factory)
            old_population = previous_factory.history_population
            refresh = None
            if (getattr(self, 'history_prefix_public_refresh', False)
                    and compatible_complete_prefix(old_population, factory)):
                refresh = public_refresh_certificate(previous_factory.state, factory.state)
                lifecycle['public_certificate'] = refresh
            if refresh is not None and refresh['refresh_required']:
                lifecycle['mode'] = 'COLD_PUBLIC_CONSTRAINT_REFRESH'
            elif population.adopt_complete_prefix(old_population):
                factory.history_population = population
                lifecycle['mode'] = 'RETAIN_COMPLETE_PREFIX'
            previous_factory.close()
        self.prepared_factory = factory
        factory.conditioning_batch_size = validated_conditioning_batch_size(
            getattr(self, 'conditioning_batch_size', 1),
            staged=getattr(self, 'staged_substitute_conditioning', False))
        receipt_index, forward_index, timing_index = 0, self.evaluator.forwards, 0

        def evidence():
            nonlocal receipt_index, forward_index, timing_index
            result = {"draws": factory.receipts[receipt_index:],
                      "neural_forwards": self.evaluator.forwards - forward_index}
            if lifecycle is not None:
                result['conditioning_prefix_lifecycle'] = lifecycle
            if factory.collect_phase_timing:
                timings = getattr(factory, 'search_phase_timing', [])
                result['search_phase_timing'] = timings[timing_index:]
                timing_index = len(timings)
            receipt_index, forward_index = len(factory.receipts), self.evaluator.forwards
            return result

        def evaluate_root(expected):
            if expected != root:
                raise ReferenceRefusal("worker root evaluation identity changed")
            return self.evaluator(observation)[1]

        return PreparedDecision(root, evaluate_root, factory, evidence, factory.bind_sampling_deadline)

    def close(self):
        try:
            if getattr(self, 'prepared_factory', None) is not None:
                self.prepared_factory.close()
                self.prepared_factory = None
        finally:
            self.env.close()
