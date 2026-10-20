"""Spawn-safe public-only champion/Showdown runtime for reference workers."""
from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path
from typing import Any

from .paper_reference import ReferenceRefusal
from .paper_reference_parallel import PreparedDecision
from .paper_reference_factory import PublicRootWorldFactory
from .paper_reference_showdown import ChampionEvaluator, decision_state
from ..local_showdown import PublicBattleMaterializationState
from ..public_decision_corpus import PublicObservation, _public_belief_view


@dataclass(frozen=True)
class PublicRootRequest:
    state: PublicBattleMaterializationState
    observation: Any
    set_source_hash: str

    @classmethod
    def capture(cls, state, observation):
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
        return cls(replace(state, belief_engine=engine), sanitized, source_hash)


@dataclass(frozen=True)
class ShowdownWorkerFactory:
    checkpoint: str
    checkpoint_sha256: str
    showdown_root: str
    set_source_hash: str
    # Explicit bounded completion adaptation; defaults preserve strict forcing.
    allow_earlier_compatible_template: bool = False
    node_binary: str = "node"

    def __call__(self, index):
        return _ShowdownRuntime(self, index)


class _ShowdownRuntime:
    def __init__(self, binding, index):
        from ..collection import env_config_with_policy_spec_masks
        from ..local_showdown import LocalShowdownConfig, LocalShowdownEnv
        from ..neural_policy import load_transformer_policy, require_torch
        from ..randbat import load_gen3_randbat_source_cached

        # The shared neural initializer owns Torch's one-shot inter-op setter.
        # Preserve this worker's single-thread defaults without calling it twice
        # when the spawned process inherits the registered thread environment.
        for name in ("POKEZERO_TORCH_NUM_THREADS", "POKEZERO_TORCH_NUM_INTEROP_THREADS"):
            if not os.environ.get(name):
                os.environ[name] = "1"
        torch = require_torch()
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
            raise ReferenceRefusal("reference worker requires single-thread Torch pools")
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
            showdown_root=Path(binding.showdown_root), node_binary=binding.node_binary,
            set_belief_source=True),
            [f"neural:{checkpoint}"], context="opt-in twenty-worker paper reference")
        self.env = LocalShowdownEnv(config)
        self.source = source
        self.allow_earlier_compatible_template = binding.allow_earlier_compatible_template
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
        root = decision_state(observation, player=state.player_id)
        factory = PublicRootWorldFactory(env=self.env, state=state, observation=observation,
            evaluator=self.evaluator, set_source=self.source,
            allow_earlier_compatible_template=self.allow_earlier_compatible_template)
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
