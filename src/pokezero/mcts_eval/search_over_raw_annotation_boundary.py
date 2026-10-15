"""Explicit acting-player annotation handoff; never reconstruct tracker history.

The retained state is the production public/player-known encoder substrate,
not a simulator snapshot. Its annotations may not be re-assessed at a later
boundary. This transport does not implement annotated opponent histories.
"""
from copy import deepcopy
from dataclasses import dataclass, field, InitVar, fields
import hashlib
from pathlib import Path

from .search_over_raw import digest, require
from .search_over_raw_world_inputs import _state_payload, public_request_witness

SCHEMA = "pokezero.search-over-raw.actor-annotation-boundary.v1"
_CAPTURE_ORIGIN = object()


@dataclass(frozen=True)
class ActorAnnotationBoundary:
    state: object
    observation: object
    public_request_sha256: str
    residuals_active: bool
    _capture_origin: InitVar[object] = None
    _original_sha: str = field(init=False, repr=False)

    def __post_init__(self, _capture_origin):
        from ..showdown import PlayerRelativeBattleState
        from ..public_decision_corpus import PublicObservation
        require(_capture_origin is _CAPTURE_ORIGIN and type(self.state) is PlayerRelativeBattleState
            and type(self.observation) is PublicObservation
            and type(self.residuals_active) is bool
            and type(self.public_request_sha256) is str
            and len(self.public_request_sha256) == 64
            and all(c in "0123456789abcdef" for c in self.public_request_sha256),
            "typed original actor annotation boundary required")
        object.__setattr__(self, "_original_sha", self._inputs())

    def _inputs(self):
        return digest(dict(schema=SCHEMA, state=_state_payload(self.state),
            observation=_state_payload(self.observation), request=self.public_request_sha256,
            residuals=self.residuals_active))

    def _check(self):
        require(self._inputs() == self._original_sha, "original actor annotation boundary changed")

    @classmethod
    def capture(cls, env, context, *, pending_transition=None):
        from ..local_showdown import LocalShowdownEnv
        from ..public_decision_corpus import PublicObservation
        from .paper_reference_runtime import PublicRootRequest
        require(type(env) is LocalShowdownEnv, "actual original annotation environment required")
        request = PublicRootRequest.capture(context.public_materialization_state,
            context.observation, pending_transition=pending_transition)
        expected_sha = public_request_witness(request)["public_request_sha256"]
        def original_boundary():
            actual = PublicRootRequest.capture(env.public_materialization_state(context.player_id),
                context.observation, pending_transition=pending_transition)
            require(public_request_witness(actual)["public_request_sha256"] == expected_sha,
                "annotation source replay/request/catalog boundary changed")
        original_boundary()
        state = env._state_for_player(context.player_id)
        observed = PublicObservation.from_observation(env.observe(context.player_id))
        require(observed == PublicObservation.from_observation(context.observation),
            "annotation capture differs from original actor observation")
        original_boundary()
        return cls(deepcopy(state), deepcopy(observed), expected_sha,
            bool(env.tier2_residuals_active()), _capture_origin=_CAPTURE_ORIGIN)

    def validate(self, request, checkpoint_contract):
        """Re-encode with the original checkpoint law, not a fixture's masks."""
        from ..dex import load_showdown_dex_cached
        from ..neural_policy import (load_transformer_model_config,
            feature_masks_from_model_config, observation_spec_from_model_config,
            category_vocab_from_model_config)
        from ..showdown import observation_from_player_state
        from ..public_decision_corpus import PublicObservation
        from .paper_reference_runtime import PublicRootRequest
        from .resolver import CheckpointContract
        require(type(request) is PublicRootRequest and type(checkpoint_contract) is CheckpointContract,
            "original typed request/checkpoint contract required")
        self._check()
        originals = [(owner, tuple((f.name, getattr(owner, f.name)) for f in fields(owner)))
            for owner in (request, checkpoint_contract)]
        checkpoint_sha = digest(_state_payload(checkpoint_contract))
        require(public_request_witness(request)["public_request_sha256"] == self.public_request_sha256
            and self.state.player_id == request.state.player_id
            and self.state.battle_id == request.state.replay.battle_id
            and self.state.request == request.state.self_request
            and self.observation == PublicObservation.from_observation(request.observation),
            "annotation boundary differs from original actor/request")
        path = Path(checkpoint_contract.checkpoint_path)
        require(hashlib.sha256(path.read_bytes()).hexdigest() == checkpoint_contract.checkpoint_sha256,
            "annotation checkpoint changed")
        config = load_transformer_model_config(path)
        masks = feature_masks_from_model_config(config)
        require(self.residuals_active == bool(masks.tier2_residuals),
            "annotation activity differs from checkpoint feature law")
        encoded = observation_from_player_state(self.state,
            category_vocab=category_vocab_from_model_config(config, checkpoint_contract.showdown_root),
            spec=observation_spec_from_model_config(config),
            dex=load_showdown_dex_cached(checkpoint_contract.showdown_root), feature_masks=masks)
        require(PublicObservation.from_observation(encoded) == self.observation,
            "retained annotations do not reproduce original production observation")
        require(all(type(getattr(owner, name)) is type(value) and
            (getattr(owner, name) == value if type(value) in (str, int, float, bool, type(None))
                else getattr(owner, name) is value) for owner, entries in originals for name, value in entries)
            and digest(_state_payload(checkpoint_contract)) == checkpoint_sha
            and hashlib.sha256(path.read_bytes()).hexdigest() == checkpoint_contract.checkpoint_sha256
            and public_request_witness(request)["public_request_sha256"] == self.public_request_sha256,
            "original request/checkpoint changed during annotation validation")
        self._check()

    def active(self):
        self._check()
        return self.residuals_active

    def boundary_state(self, player_id):
        self._check()
        require(player_id == self.state.player_id, "annotation transport is acting-player only")
        return self.state

    def overlay_for(self, player_id):
        state = self.boundary_state(player_id)
        return {index:(token.residual, token.residual_valid, token.cb_bit, token.investment)
            for index, token in enumerate(state.transition_tokens)
            if token.residual is not None or token.residual_valid or token.cb_bit or token.investment}
