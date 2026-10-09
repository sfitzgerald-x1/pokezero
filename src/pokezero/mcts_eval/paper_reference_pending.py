"""Public-only certificates for an interrupted Baton Pass decision.

The previous actor-known root and its OWN already-played action may cross the
worker boundary. No live opponent request, action, priors, or snapshot may do so.
A fresh hypothetical previous world supplies champion opponent priors; jointly
sample team/action/chance and reject until the observed public transition and
current actor information state match. This is conditioning, not argmax and
not copying the live opponent's private commitment.
"""
from dataclasses import dataclass
import re
from typing import Any

from .paper_reference import ReferenceRefusal
from .paper_reference_showdown import decision_state
from ..local_showdown import PublicBattleMaterializationState
from ..public_decision_corpus import PublicObservation, _public_belief_view


@dataclass(frozen=True)
class PendingPolicyTransition:
    before_state: PublicBattleMaterializationState
    before_observation: Any
    own_action: int
    set_source_hash: str
    prior_transition: Any = None

    @classmethod
    def capture(cls, public_request, own_action):
        from .paper_reference_substitute import SubstituteHistoryTransition
        if (public_request.pending_transition is not None
                and not isinstance(public_request.pending_transition, SubstituteHistoryTransition)):
            raise ReferenceRefusal('pending certificate cannot recursively reuse a pending root')
        return cls(public_request.state, public_request.observation, own_action,
                   public_request.set_source_hash, public_request.pending_transition)


@dataclass(frozen=True)
class FaintReplacementTransition(PendingPolicyTransition):
    """Replay a public faint interruption, retaining its actual residual queue."""


def requires_faint_encore_replay(state):
    return (bool(state.self_request.get('forceSwitch'))
            and state.deferred_opponent_action_player is None
            and any('encore' in state.replay.volatiles.get(side, ()) for side in ('p1', 'p2')))


def requires_baton_interruption_replay(state):
    # Even a slow Baton Pass retains the native residual queue. Merely clearing
    # the already-spent opponent action and directly rebuilding a boundary
    # would silently skip that queue, so both fast and slow cases need replay.
    forced=state.self_request.get('forceSwitch')
    return (isinstance(forced, (list,tuple)) and any(forced)
            and state.player_id in getattr(state.replay, 'pending_baton_pass', ()))


def public_history(state):
    return tuple(event.raw_line for event in state.replay.public_events
        if event.raw_line not in ('', '|')
        and event.raw_line != "|message|The battle's RNG was reset.")


def validate_transition(transition, current, observation, set_source_hash):
    if not isinstance(transition, PendingPolicyTransition):
        raise ReferenceRefusal('pending committed opponent action needs a sampled-policy certificate')
    before = transition.before_state
    if (not isinstance(before, PublicBattleMaterializationState) or before.replay.requests
            or before.player_id != current.player_id
            or before.observation_format_id != current.observation_format_id
            or before.deferred_opponent_action_player is not None
            or before.self_request.get('forceSwitch')
            or transition.set_source_hash != set_source_hash):
        raise ReferenceRefusal('pending certificate source/player/request drift')
    sanitized = PublicObservation.from_observation(transition.before_observation).to_observation(
        belief_view=_public_belief_view(transition.before_observation.metadata))
    if sanitized.metadata != transition.before_observation.metadata:
        raise ReferenceRefusal('pending certificate contains nonpublic observation metadata')
    action = transition.own_action
    old, now = public_history(before), public_history(current)
    suffix = now[len(old):]
    if isinstance(transition, FaintReplacementTransition):
        active_party = [row for row in current.self_request.get('side', {}).get('pokemon', ())
                        if row.get('active')]
        if (not requires_faint_encore_replay(current) or len(active_party) != 1
                or str(active_party[0].get('condition', '')).split() != ['0', 'fnt']
                or type(action) is not int or not 0 <= action < len(transition.before_observation.legal_action_mask)
                or not transition.before_observation.legal_action_mask[action]
                or not old or now[:len(old)] != old or not suffix
                or any(line.startswith(('|turn|', '|upkeep')) for line in suffix)
                or not any(line == '|faint|' + active_party[0]['ident'].replace(':', 'a:', 1)
                           for line in suffix)):
            raise ReferenceRefusal('faint certificate does not bind an exact pre-upkeep replacement')
        return decision_state(observation, player=current.player_id)
    active = before.self_request.get('active', [])
    moves = active[0].get('moves', []) if len(active) == 1 else []
    if (type(action) is not int or not 0 <= action < min(4, len(moves))
            or not transition.before_observation.legal_action_mask[action]
            or re.sub('[^a-z0-9]', '', str(moves[action].get('id', '')).lower()) != 'batonpass'):
        raise ReferenceRefusal('pending certificate must bind the actor-known prior Baton Pass action')
    baton_lines = [line for line in suffix if line.startswith('|move|' + current.player_id + 'a:')
        and len(line.split('|')) >= 4 and line.split('|')[3] == 'Baton Pass']
    if (not old or now[:len(old)] != old or not suffix
            or any(line.startswith('|turn|') for line in suffix)
            or len(baton_lines) != 1 or not requires_baton_interruption_replay(current)):
        raise ReferenceRefusal('pending certificate does not match the exact public interrupted transition')
    from ..public_action_capture import public_action_identifiers_from_protocol_lines
    opponent = 'p2' if current.player_id == 'p1' else 'p1'
    before_baton = suffix[:suffix.index(baton_lines[0])]
    public_completion = public_action_identifiers_from_protocol_lines(
        before_baton, cancellation_players=(opponent,)).get(opponent)
    if current.deferred_opponent_action_player is None:
        if public_completion is None:
            raise ReferenceRefusal('slow Baton Pass lacks public opponent action completion')
    elif (current.deferred_opponent_action_player != opponent or public_completion is not None
            or len([line for line in suffix if line.startswith('|move|')]) != 1):
        raise ReferenceRefusal('fast Baton Pass has inconsistent public opponent action timing')
    # Force materialization to validate both roots against the same information
    # representation, not merely a coarse team or legality match.
    return decision_state(observation, player=current.player_id)


def condition_pending_world(factory, hidden_rng, evidence, *, max_attempts=128):
    from .paper_reference_factory import PublicRootWorldFactory
    transition = factory.pending_transition
    # This path still uses ORIGINAL joint rejection, not weighted particles.
    # In particular a nested Substitute prior must remain its original kernel:
    # silently forwarding empirical-particle flags would alter this path's
    # approximation and require separate composition/ownership qualification.
    # The prepared-root deadline is forwarded below, never reset per attempt.
    prior = PublicRootWorldFactory(env=factory.env, state=transition.before_state,
        observation=transition.before_observation, evaluator=factory.evaluator,
        set_source=factory.set_source,
        allow_earlier_compatible_template=factory.allow_earlier_compatible_template,
        max_known_set_draws=factory.max_known_set_draws,
        **({'pending_transition': transition.prior_transition}
           if getattr(transition, 'prior_transition', None) is not None else {}))
    if getattr(factory, 'sampling_deadline_at', None) is not None:
        prior.bind_sampling_deadline(factory.sampling_deadline_at)
    check_deadline = getattr(factory, 'check_sampling_deadline', lambda: None)
    slow_baton = (factory.state.deferred_opponent_action_player is None
                  and not isinstance(transition, FaintReplacementTransition))
    # A slow interruption can reveal the opponent's just-resolved move. Draw
    # from the PRIOR information state and condition on the full observed
    # suffix; do not force that later move into the earlier hidden-team prior.
    # Exact current actor/public matching below remains mandatory.
    if prior.known != factory.known and not slow_baton:
        raise ReferenceRefusal('pending transition introduces unconditioned opponent-team evidence')
    subject = factory.state.player_id
    opponent = 'p2' if subject == 'p1' else 'p1'
    rejected = []
    for attempt in range(max_attempts):
        check_deadline()
        world = prior(hidden_rng)
        accepted = False
        try:
            check_deadline()
            if set(factory.env.requested_players()) != {subject, opponent}:
                raise ReferenceRefusal('pending certificate prior root is not simultaneous')
            legal, evaluated = factory.evaluator(factory.env.observe(opponent))
            if not legal or len(legal) != len(evaluated.priors) or sum(evaluated.priors) <= 0:
                raise ReferenceRefusal('pending champion policy has no legal sampling support')
            action = hidden_rng.choices(legal, weights=evaluated.priors, k=1)[0]
            chance_seed = hidden_rng.getrandbits(64)
            factory.env.reseed_simulator_rng(chance_seed)
            factory.env.step({subject: transition.own_action, opponent: action})
            if factory.env.terminal() is not None or subject not in factory.env.requested_players():
                rejected.append(dict(attempt=attempt, reason='different public request/terminal'))
                continue
            actual = factory.env.public_materialization_state(subject)
            check_deadline()
            observed = factory.env.observe(subject)
            if (decision_state(observed, player=subject) != factory.root
                    or public_history(actual) != public_history(factory.state)
                    or actual.deferred_opponent_action_player != factory.state.deferred_opponent_action_player):
                rejected.append(dict(attempt=attempt, reason='different actor/public transition'))
                continue
            evidence.update({k: v for k, v in prior.receipts[-1].items()
                             if k not in ('ordinal', 'status', 'released')})
            evidence.update(status='ROOT_VALIDATED', pending_policy_conditioning=dict(
                algorithm='joint hidden-team/champion-policy/chance rejection on exact public transition',
                attempts=attempt+1, rejected=rejected, opponent=opponent,
                sampled_action=action, legal=list(legal), priors=list(evaluated.priors),
                chance_seed=chance_seed, own_prior_action=transition.own_action,
                prior_actor_root_key=prior.root.key.hex(), current_actor_root_key=factory.root.key.hex(),
                live_opponent_action_used=False,
                prior_conditioning_kernel='original full-support joint rejection; particle options not propagated',
                later_known_traits_conditioned_by_exact_replay=bool(slow_baton and prior.known != factory.known)))
            if isinstance(transition, FaintReplacementTransition):
                evidence['pending_policy_conditioning']['transition_kind'] = 'pre-upkeep-faint-replacement'
            elif factory.state.deferred_opponent_action_player is None:
                evidence['pending_policy_conditioning']['transition_kind'] = 'slow-baton-pass-native-residual-queue'
            prior_release = world.release
            def release():
                prior_release()
                factory._release(evidence)
            world.release = release
            accepted = True
            return world
        finally:
            if not accepted:
                world.close()
    raise ReferenceRefusal('pending public-transition conditioning exhausted its explicit rejection cap')
