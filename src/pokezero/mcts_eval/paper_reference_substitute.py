"""Condition hidden Substitute HP on an exact actor/public transition history.

This is bounded joint rejection, not uniform HP, true-state extraction, or a
replay of the live opponent's choices/chance seeds. The anchor has reconstructible
Substitute HP; every continuation stores ONLY the actor's known played action.
"""
from collections import Counter
from dataclasses import dataclass, replace

from .paper_reference import ReferenceRefusal
from .paper_reference_pending import PendingPolicyTransition, public_history
from .paper_reference_showdown import decision_state
from ..local_showdown import PublicBattleMaterializationState
from ..public_decision_corpus import PublicObservation, _public_belief_view


# Continue the original proposal stream, rather than retrying/redrawing after a
# 128-attempt exhaustion. Exact public-root matching and the conditional law are
# unchanged; this is still a finite, fail-closed operational bound.
MAX_SUBSTITUTE_REJECTION_ATTEMPTS = 2048


def requires_substitute_replay(state):
    return any(getattr(state.replay, 'substitute_health_state', {}).get(side) == 'unknown'
               and 'substitute' in getattr(state.replay, 'volatiles', {}).get(side, ())
               for side in ('p1', 'p2'))


@dataclass(frozen=True)
class SubstituteHistoryTransition(PendingPolicyTransition):
    continuation_actions: tuple[int | None, ...] = ()

    @classmethod
    def capture(cls, request, own_action):
        if requires_substitute_replay(request.state):
            raise ReferenceRefusal('Substitute anchor must have reconstructible public HP')
        return cls(request.state, request.observation, own_action, request.set_source_hash,
                   request.pending_transition)

    def append(self, own_action):
        if own_action is not None and type(own_action) is not int:
            raise ReferenceRefusal('Substitute history must contain only known own action indices')
        if len(self.continuation_actions) >= 199:
            raise ReferenceRefusal('Substitute history exceeds the registered 200 boundaries')
        return replace(self, continuation_actions=(*self.continuation_actions, own_action))


class SubstituteHistoryTracker:
    """Keep an anchor and known own choices; never accepts opponent choices."""
    def __init__(self):
        self.transition = None

    def certificate(self, state):
        if not requires_substitute_replay(state):
            return None
        if self.transition is None:
            raise ReferenceRefusal('unknown Substitute HP lacks its public reconstructible anchor')
        return self.transition

    def accept_own(self, request, action):
        if requires_substitute_replay(request.state):
            self.transition = self.certificate(request.state).append(action)
        else:
            self.transition = SubstituteHistoryTransition.capture(request, action)

    def accept_opponent_only(self):
        if self.transition is not None:
            self.transition = self.transition.append(None)


def validate_substitute_transition(transition, current, observation, source_hash):
    if not isinstance(transition, SubstituteHistoryTransition):
        raise ReferenceRefusal('unknown Substitute HP needs a public history certificate')
    before = transition.before_state
    if (not isinstance(before, PublicBattleMaterializationState) or before.replay.requests
            or before.player_id != current.player_id
            or before.observation_format_id != current.observation_format_id
            or transition.set_source_hash != source_hash or requires_substitute_replay(before)
            or not requires_substitute_replay(current)):
        raise ReferenceRefusal('Substitute certificate source/player/public-HP drift')
    sanitized = PublicObservation.from_observation(transition.before_observation).to_observation(
        belief_view=_public_belief_view(transition.before_observation.metadata))
    if sanitized.metadata != transition.before_observation.metadata:
        raise ReferenceRefusal('Substitute certificate contains nonpublic observation metadata')
    action = transition.own_action
    if (type(action) is not int or not 0 <= action < len(transition.before_observation.legal_action_mask)
            or not transition.before_observation.legal_action_mask[action]
            or type(transition.continuation_actions) is not tuple
            or len(transition.continuation_actions) > 199
            or any(a is not None and (type(a) is not int or a < 0)
                   for a in transition.continuation_actions)):
        raise ReferenceRefusal('Substitute certificate has invalid known own actions')
    old, now = public_history(before), public_history(current)
    suffix = now[len(old):]
    if (not old or now[:len(old)] != old or not suffix
            or not any(line.endswith('|Substitute|[damage]') for line in suffix)):
        raise ReferenceRefusal('Substitute certificate does not bind observed nonbreaking damage')
    return decision_state(observation, player=current.player_id)


def condition_substitute_world(factory, hidden_rng, evidence, *, max_attempts=MAX_SUBSTITUTE_REJECTION_ATTEMPTS):
    from .paper_reference_factory import PublicRootWorldFactory
    if type(max_attempts) is not int or not 1 <= max_attempts <= MAX_SUBSTITUTE_REJECTION_ATTEMPTS:
        raise ReferenceRefusal('invalid Substitute rejection limit')
    transition = factory.pending_transition
    prior = PublicRootWorldFactory(env=factory.env, state=transition.before_state,
        observation=transition.before_observation, evaluator=factory.evaluator,
        set_source=factory.set_source,
        allow_earlier_compatible_template=factory.allow_earlier_compatible_template,
        max_known_set_draws=factory.max_known_set_draws,
        pending_transition=transition.prior_transition)
    if getattr(factory, 'sampling_deadline_at', None) is not None:
        prior.bind_sampling_deadline(factory.sampling_deadline_at)
    check_deadline = getattr(factory, 'check_sampling_deadline', lambda: None)
    subject = factory.state.player_id
    opponent = 'p2' if subject == 'p1' else 'p1'
    expected_history = public_history(factory.state)
    rejected = []
    # Public-only counters diagnose exhaustion without changing the proposal
    # stream, acceptance rule, accepted witness, or sampled-world ownership.
    prefix_counts, mismatches = Counter(), Counter()
    if getattr(factory, 'staged_substitute_conditioning', False):
        from .paper_reference_staged_chance import sample_staged_path
        plan = factory.constant_chance_plan
        if plan is not None:
            return sample_staged_path(factory, prior, plan, hidden_rng, evidence, max_attempts=max_attempts)
        # Explicitly disclosed original-kernel dispatch, not a raw-policy or
        # unconditioned-world fallback. No positions or positive-likelihood
        # hidden teams are dropped to make an optimization eligible.
        evidence['constant_chance_dispatch'] = 'unsupported public program; original joint conditioning'
    for attempt in range(max_attempts):
        check_deadline()
        world, accepted = prior(hidden_rng), False
        simulated = []
        try:
            valid = True
            for own_action in (transition.own_action, *transition.continuation_actions):
                check_deadline()
                requested = set(factory.env.requested_players())
                actions = {}
                if (subject in requested) != (own_action is not None):
                    valid = False
                    break
                if own_action is not None:
                    mask = factory.env.observe(subject).legal_action_mask
                    if not 0 <= own_action < len(mask) or not mask[own_action]:
                        valid = False
                        break
                    actions[subject] = own_action
                legal, probabilities, opponent_action = (), (), None
                if opponent in requested:
                    legal, evaluated = factory.evaluator(factory.env.observe(opponent))
                    probabilities = tuple(evaluated.priors)
                    if not legal or len(legal) != len(probabilities) or sum(probabilities) <= 0:
                        raise ReferenceRefusal('Substitute champion has no legal policy support')
                    opponent_action = hidden_rng.choices(legal, weights=probabilities, k=1)[0]
                    actions[opponent] = opponent_action
                if not requested or set(actions) != requested:
                    valid = False
                    break
                chance_seed = hidden_rng.getrandbits(64)
                factory.env.reseed_simulator_rng(chance_seed)
                factory.env.step(actions)
                simulated.append(dict(own_action=own_action, opponent_action=opponent_action,
                    opponent_legal=list(legal), opponent_priors=list(probabilities), chance_seed=chance_seed))
                if factory.env.terminal() is not None:
                    valid = False
                    break
                actual_history = public_history(factory.env.public_materialization_state(subject))
                prefix = next((i for i, (left, right) in enumerate(zip(expected_history, actual_history))
                               if left != right), min(len(expected_history), len(actual_history)))
                prefix_counts[prefix] += 1
                if prefix < min(len(expected_history), len(actual_history)):
                    mismatches[(prefix, expected_history[prefix], actual_history[prefix])] += 1
                if expected_history[:len(actual_history)] != actual_history:
                    valid = False
                    break
            if not valid or subject not in factory.env.requested_players():
                rejected.append(dict(attempt=attempt, reason='different public transition/request'))
                continue
            actual = factory.env.public_materialization_state(subject)
            check_deadline()
            if (decision_state(factory.env.observe(subject), player=subject) != factory.root
                    or public_history(actual) != expected_history
                    or actual.deferred_opponent_action_player != factory.state.deferred_opponent_action_player):
                rejected.append(dict(attempt=attempt, reason='different exact actor/public root'))
                continue
            # Only this independently drawn hypothetical world's serialization is read.
            battle = factory.env.snapshot().bridge_snapshot['battle']
            hp = {side: battle['sides'][i]['pokemon'][0]['volatiles']['substitute']['hp']
                  for i, side in enumerate(('p1', 'p2'))
                  if 'substitute' in battle['sides'][i]['pokemon'][0]['volatiles']}
            if not hp or any(type(value) is not int or value <= 0 for value in hp.values()):
                raise ReferenceRefusal('conditioned hypothetical Substitute HP is invalid')
            evidence.update({k: v for k, v in prior.receipts[-1].items()
                             if k not in ('ordinal', 'status', 'released')})
            evidence.update(status='ROOT_VALIDATED', substitute_policy_conditioning=dict(
                algorithm='joint hidden-team/champion-policy/chance rejection on exact public history',
                attempts=attempt+1, max_attempts=max_attempts, rejected=rejected,
                steps=simulated, sampled_substitute_hp=hp,
                prior_actor_root_key=prior.root.key.hex(), current_actor_root_key=factory.root.key.hex(),
                live_opponent_action_used=False, live_hidden_hp_used=False))
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
    error = ReferenceRefusal('Substitute public-history conditioning exhausted its explicit rejection cap')
    error.sampling_diagnostic = dict(
        schema='pokezero.substitute-rejection-diagnostic.v1',
        attempts=max_attempts, max_attempts=max_attempts,
        expected_public_history_lines=len(expected_history),
        rejection_reasons=dict(Counter(row['reason'] for row in rejected)),
        public_prefix_checks=[dict(matched_lines=k, count=v) for k, v in sorted(prefix_counts.items())],
        first_public_mismatches=[dict(index=k[0], expected=k[1], hypothetical=k[2], count=v)
                                 for k, v in mismatches.most_common()],
        # Prefix counters measure intermediate checks, NOT complete-world draws.
        prefix_counts_are_intermediate_checks=True,
        accepted_worlds=0, live_opponent_action_used=False, live_hidden_hp_used=False,
    )
    raise error
