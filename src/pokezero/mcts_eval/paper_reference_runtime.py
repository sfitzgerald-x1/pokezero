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
        try:
            # Warm bridge startup belongs to the separately measured pool startup.
            # This synthetic startup battle contributes NO trajectory/statistics.
            self.env.reset(seed=2026100400 + index)
        except BaseException:
            self.env.close()
            raise

    def prepare(self, public_request):
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
            max_known_set_draws=self.max_known_set_draws, pending_transition=pending)
        receipt_index, forward_index = 0, self.evaluator.forwards

        def evidence():
            nonlocal receipt_index, forward_index
            result = {"draws": factory.receipts[receipt_index:],
                      "neural_forwards": self.evaluator.forwards - forward_index}
            receipt_index, forward_index = len(factory.receipts), self.evaluator.forwards
            return result

        def evaluate_root(expected):
            if expected != root:
                raise ReferenceRefusal("worker root evaluation identity changed")
            return self.evaluator(observation)[1]

        return PreparedDecision(root, evaluate_root, factory, evidence)

    def close(self):
        self.env.close()
