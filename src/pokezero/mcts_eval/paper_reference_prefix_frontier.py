"""Conservative identity gate for sequential hypothetical history filtering."""
from .paper_reference_pending import public_history
from .paper_reference_showdown import decision_state
from .paper_reference_substitute import SubstituteHistoryTransition
from .paper_reference import ReferenceRefusal
import hashlib


OPTIONS = ('history_particles', 'guide_history_actions', 'history_chance_pool',
    'batch_history_chance', 'guide_history_chance', 'guide_history_accuracy', 'membership_first',
    'native_membership_batch_size', 'public_anchor_constraints', 'guide_anchor_genders',
    'early_encore_potential', 'allow_earlier_compatible_template', 'max_known_set_draws')


def public_refresh_certificate(previous, current):
    """Predetermined public-information schedule, NEVER a failed-bank retry.

    Reinitializing the configured cold SMC at newly revealed constraints avoids
    pretending a finite old genealogy retains all prior support. The refresh
    is still approximate, may fail, and does not modify hypothetical native
    outcomes or condition on live opponent choices/chance seeds. Inspect only
    public lines; no particle, private request, ESS or acceptance counter.
    """
    old, now = public_history(previous), public_history(current)
    if previous.player_id != current.player_id or now[:len(old)] != old:
        raise ReferenceRefusal('public refresh schedule requires the same actor and exact history prefix')
    opponent = 'p2' if current.player_id == 'p1' else 'p1'
    def facts(lines):
        result = set()
        for line in lines:
            parts = line.split('|')
            if len(parts) < 4 or not parts[2].startswith(opponent):
                continue
            kind = parts[1]
            if kind in ('switch', 'drag', 'replace'):
                result.add(('opponent_details', parts[2], parts[3]))
            elif kind == 'move':
                result.add(('opponent_move', parts[2], parts[3]))
            elif kind in ('-ability', '-item', '-enditem'):
                result.add(('opponent_trait', parts[2], parts[3]))
        return result
    observed = facts(old)
    reasons = [dict(kind=kind, actor=actor, value=value)
               for kind, actor, value in sorted(facts(now[len(old):]) - observed)]
    for line in now[len(old):]:
        parts = line.split('|')
        if (len(parts) >= 4 and parts[1] == '-end' and parts[3] == 'Encore'
                and parts[2][:2] in ('p1', 'p2')):
            reasons.append(dict(kind='observed_encore_end', actor=parts[2], value='Encore'))
    return dict(schema='pokezero.public-constraint-prefix-refresh.v1',
        policy='revealed-opponent-details-moves-traits-or-encore-end.v1',
        refresh_required=bool(reasons), reasons=reasons,
        previous_public_sha256=hashlib.sha256('\n'.join(old).encode()).hexdigest(),
        current_public_sha256=hashlib.sha256('\n'.join(now).encode()).hexdigest(),
        bank_outcome_used=False, private_state_used=False, retry_after_failure=False)


def compatible_complete_prefix(population, factory):
    """Return false for another root/anchor/kernel, partial bank or new battle.

    A miss continues the explicitly configured ORIGINAL cold conditioning path,
    never a raw-action/unconditioned-world fallback. Newly observed species or
    effects must still pass native transitions and the exact public/root checks;
    zero survivors continue to refuse instead of redrawing until success.
    """
    if (population is None or not population.particles
            or not population.receipt.get('complete')
            or population.receipt.get('partial_stage_used') is not False):
        return False
    old = population.factory
    a, b = old.pending_transition, factory.pending_transition
    if (old.active or old.env is not factory.env or old.evaluator is not factory.evaluator
            or any(getattr(old, name) != getattr(factory, name) for name in OPTIONS)
            or not isinstance(a, SubstituteHistoryTransition)
            or not isinstance(b, SubstituteHistoryTransition)
            or a.prior_transition is not None or b.prior_transition is not None
            or a.set_source_hash != b.set_source_hash):
        return False
    before, after = a.before_state, b.before_state
    if (old.state.player_id != factory.state.player_id
            or before.replay.requests or after.replay.requests
            or before.replay != after.replay
            or any(getattr(before, field) != getattr(after, field) for field in (
                'player_id', 'format_id', 'observation_format_id', 'self_request',
                'self_initial_request', 'self_move_states', 'self_recharge_pp_charge'))
            or before.belief_engine.snapshot() != after.belief_engine.snapshot()
            or decision_state(a.before_observation, player=before.player_id)
                != decision_state(b.before_observation, player=after.player_id)):
        return False
    previous_actions = (a.own_action, *a.continuation_actions)
    new_actions = (b.own_action, *b.continuation_actions)
    old_history, new_history = public_history(old.state), public_history(factory.state)
    if (len(new_actions) <= len(previous_actions)
            or new_actions[:len(previous_actions)] != previous_actions
            or len(new_history) <= len(old_history)
            or new_history[:len(old_history)] != old_history
            or population.receipt.get('final_actor_root_key') != old.root.key.hex()
            or any(len(p.steps) != len(previous_actions)
                   or p.initial_importance_weight != 1. for p in population.particles)):
        return False
    return True
