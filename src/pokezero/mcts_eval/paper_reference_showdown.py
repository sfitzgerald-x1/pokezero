"""Explicitly owned hypothetical Showdown-world adapter for the reference kernel.

This does not sample teams, launch experiments, or claim paper qualification.
The caller must create a NEW determinization for every trajectory through the
paper's exact-generator rejection sampler, validate public root equivalence,
and bind the champion. A real live hidden-state environment is refused.
Current champion inputs/information keys differ from the thesis's Gen 4 vector;
that controlled representation deviation remains visible in comparison plans.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any, Callable

from .paper_reference import DecisionState, Evaluation, Frame, ReferenceRefusal, Terminal
from ..public_decision_corpus import PublicObservation, _public_belief_view


def decision_state(observation: Any, *, player: str) -> DecisionState:
    """Use only the champion observation plus its sanitized actor/public fields.

    Never hashes a Showdown simulator serialization or the opponent request.
    Legal mask is included so coarse model features cannot silently alias legal
    support. Public belief is included because it changes future sampling.
    The adapter's action indices belong to THIS actor's known party, not a
    numeric slot recovered from another hidden replay world.
    """
    if player not in ("p1", "p2"):
        raise ReferenceRefusal("reference subject must be p1 or p2")
    public = PublicObservation.from_observation(observation).to_dict()
    belief = _public_belief_view(observation.metadata)
    if not isinstance(belief, dict) or belief.get("self_slot") != player:
        raise ReferenceRefusal("reference information state has no actor-relative public belief")
    for side in ("self_team", "opponent_team"):
        if not isinstance(public["acting_player_state"].get(side), list):
            raise ReferenceRefusal("reference information state has no public faint ledger")
    faint_count = sum(row.get("fainted") is True
        for side in ("self_team", "opponent_team") for row in public["acting_player_state"][side])
    payload = {"player": player, "observation": public, "public_belief": belief}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return DecisionState(hashlib.sha256(raw).digest(),
        tuple(f"action:{i}" for i, legal in enumerate(public["legal_action_mask"]) if legal), faint_count)


class ChampionEvaluator:
    """One-snapshot own-policy/critic inference through the canonical model path.

    Requires tanh reward-scale value, no calibration or exploration overrides.
    Evaluation is exactly one real forward, not a forced-choice synthetic row.
    No behavior action is chosen: opponent choices are sampled by the reference
    kernel from the full legal distribution, in a separate RNG domain.
    """

    def __init__(self, policy: Any) -> None:
        config = policy.result.model_config
        if (config.window_size != 1 or config.value_activation != "tanh"
                or getattr(policy.result, "value_calibration_transform", None) is not None
                or not policy.deterministic or policy.exploration_epsilon != 0
                or policy.sampling_temperature != 1 or policy.family_gated_selection
                or policy.history_mask_k is not None or policy.forward_fn != policy._default_forward):
            raise ReferenceRefusal("reference champion must be the unchanged one-snapshot signed actor-critic")
        self.policy = policy
        self.forwards = 0

    def __call__(self, observation: Any) -> tuple[tuple[int, ...], Evaluation]:
        from ..neural_policy import (
            _masked_action_probabilities, observation_spec_from_model_config,
            observation_window_to_torch, require_torch,
        )

        observation.validate(observation_spec_from_model_config(self.policy.result.model_config))
        tensors = observation_window_to_torch((observation,), window_size=1, device=self.policy.device)
        self.policy.model.eval()
        with require_torch().no_grad():
            output = self.policy._default_forward(tensors)
            probabilities = _masked_action_probabilities(output.policy_logits[0],
                tensors["legal_action_mask"][0], temperature=1.)
            values = tuple(float(p.detach().cpu().item()) for p in probabilities)
            signed_value = float(output.value[0].detach().cpu().item())
        self.forwards += 1
        mask = tuple(bool(v) for v in observation.legal_action_mask)
        if (not isinstance(values, (tuple, list)) or len(values) != len(mask)
                or any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 for p in values)
                or any(values[i] != 0 for i, legal in enumerate(mask) if not legal)):
            raise ReferenceRefusal("champion prior has invalid or illegal policy mass")
        legal = tuple(i for i, enabled in enumerate(mask) if enabled)
        return legal, Evaluation(tuple(float(values[i]) for i in legal), signed_value)


class ShowdownTrajectoryWorld:
    """Owns only one already-materialized sampled world; releases it on close.

    A warm environment may be reused after release by the world factory, but
    every trajectory must be rematerialized with a fresh hidden draw. It may
    not restore one fixed sampled root for the whole search.
    """

    def __init__(self, env: Any, *, subject: str, evaluator: ChampionEvaluator,
                 release: Callable[[], None]) -> None:
        if subject not in ("p1", "p2") or getattr(env, "_search_snapshot_permitted", False) is not True:
            raise ReferenceRefusal("reference requires an owned, explicitly sampled hypothetical world")
        if not callable(release):
            raise ReferenceRefusal("reference world must have an explicit release owner")
        self.env, self.subject, self.evaluator, self.release = env, subject, evaluator, release
        self.opponent = "p2" if subject == "p1" else "p1"
        self.closed = False
        self._observations: dict[bytes, Any] = {}

    def frame(self) -> Frame:
        if self.closed or self.env.terminal() is not None:
            raise ReferenceRefusal("reference world has no live request boundary")
        requested = tuple(self.env.requested_players())
        if not requested or len(set(requested)) != len(requested) or set(requested) - {"p1", "p2"}:
            raise ReferenceRefusal("reference world has an invalid requested-player set")
        state = None
        if self.subject in requested:
            observation = self.env.observe(self.subject)
            state = decision_state(observation, player=self.subject)
            self._observations[state.key] = observation
        opponent_actions: tuple[str, ...] = ()
        opponent_priors: tuple[float, ...] = ()
        if self.opponent in requested:
            indices, evaluated = self.evaluator(self.env.observe(self.opponent))
            opponent_actions = tuple(f"action:{i}" for i in indices)
            opponent_priors = evaluated.priors
        return Frame(state, opponent_actions, opponent_priors)

    def evaluate(self, state: DecisionState) -> Evaluation:
        observation = self._observations.get(state.key)
        if observation is None or decision_state(observation, player=self.subject) != state:
            raise ReferenceRefusal("reference leaf observation is not owned by the current subject state")
        indices, evaluated = self.evaluator(observation)
        if tuple(f"action:{i}" for i in indices) != state.actions:
            raise ReferenceRefusal("reference leaf policy and legal identity disagree")
        return evaluated

    def advance(self, subject_action: str | None, opponent_action: str | None,
                chance_rng: random.Random) -> Frame | Terminal:
        if self.closed:
            raise ReferenceRefusal("reference world was already released")
        requested = tuple(self.env.requested_players())
        actions = {}
        for player, action in ((self.subject, subject_action), (self.opponent, opponent_action)):
            if player in requested:
                if (not isinstance(action, str) or not action.startswith("action:")
                        or not action[7:].isdigit()):
                    raise ReferenceRefusal("requested seat has no owned legal action")
                index = int(action[7:])
                mask = self.env.legal_actions(player)
                if not 0 <= index < len(mask) or not mask[index]:
                    raise ReferenceRefusal("reference chose an illegal sampled-world action")
                actions[player] = index
            elif action is not None:
                raise ReferenceRefusal("reference fabricated an action for a nonrequested seat")
        self.env.reseed_simulator_rng(chance_rng.getrandbits(64))
        result = self.env.step(actions)
        self._observations.clear()
        if result.terminal is not None:
            if result.terminal.capped:
                raise ReferenceRefusal("simulator cap is not an actual terminal result")
            winner = result.terminal.winner
            if winner not in (None, self.subject, self.opponent):
                raise ReferenceRefusal("simulator winner is not a valid player")
            return Terminal(0 if winner is None else 1 if winner == self.subject else -1)
        return self.frame()

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._observations.clear()
            self.release()
