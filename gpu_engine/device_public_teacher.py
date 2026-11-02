"""Opt-in direct-device public-world depth1 pricing through original D math.

The actor callbacks receive ONLY the SAME public29 encoder inputs. Opponent
reply policy observes its source-sampled own party; learner value observes its
own hypothetical party plus root-cloned/actually advanced PUBLIC knowledge.
No private critic, protocol/fold, capture certificate or survivor-only pricing
exists here. Make one Executor per worker; its pure JIT functions are retained.
"""
from dataclasses import dataclass

import jax
import jax.numpy as j
import numpy as np

from pokezero import v5_gumbel as original
from . import device_public_worlds as source_plans
from . import device_public_world_state as world_state
from . import device_public_leaf as leaf
from . import engine_device_observation as observation
from . import transition
from .benchmark_device_selfplay import request_seats


# Explicit parent/user choice: these are UNKNOWN neural history inputs, not
# exact zeros and not missing legal-reply probability. They never enter the
# physical state. Genuine sampled/root-private physical state must still exist.
REPLY_HISTORY_APPROXIMATION = {
    'scope': 'SAMPLED_OPPONENT_OWN_REPLY_POLICY_INPUT_ONLY',
    'treatment': 'DECLARED_UNKNOWN_ZERO',
    'columns': ('own.LAST_DAMAGE_TAKEN', 'foe.LAST_DAMAGE_DEALT', 'field.SELF_HAZARD_CREDIT'),
    'reason': 'Prior exact sampled-opponent victim-HP history is not public. '
        'Actual opponent-owned damage supplements are redacted, never borrowed.',
    'exact_known_zero': False,
    'physical_parity_claim': False,
}


@dataclass(frozen=True)
class Executor:
    legal: object
    observe: object
    dispatch: object
    sample: object
    advance: object


def make_executor(engine_tables, observation_tables, *, enable_device_teacher=False):
    """Retain original numerical/public/encoder functions once per worker.

    Construction defines cached functions only; no world, model or transition
    is evaluated. Callers supply already loaded/compiled original actor heads
    separately. BF16 policy/value selection remains those actors' responsibility.
    """
    if enable_device_teacher is not True:
        raise ValueError('explicit direct-device teacher enablement required')
    return Executor(
        jax.jit(lambda state: transition.action_mask(state, engine_tables) &
            request_seats(state)[..., None]),
        jax.jit(lambda state, public, owner, legal: leaf.from_public(state, public,
            observation_tables, observer_sides=owner, legal_action_mask=legal,
            enable_device_leaf=True)),
        jax.jit(lambda encoded, state, owner, action: observation.map_actions(
            encoded, state, owner, action)),
        jax.jit(transition.sample_chance_inputs),
        jax.jit(lambda state, public, actions, chance: transition.step_with_public(
            state, actions, engine_tables, public,
            belief_tables=observation_tables.belief, chance_inputs=chance)))


@dataclass(frozen=True)
class Result:
    status: str
    reduction: object | None
    world_reports: tuple
    worlds: tuple
    missing_worlds: int
    reason: str | None
    actual_teacher_calls: object
    reply_input_approximation: object
    # Source-level working consumer is not a measured training/runtime result.
    measured_runtime_ready: bool = False


def _host(value):
    return np.asarray(jax.block_until_ready(value))


def _view(executor, state, public, owner):
    legal = executor.legal(state)
    if legal.shape != (1, 2, 9) or legal.dtype != j.bool_:
        raise ValueError('actual physical legal-mask geometry differs')
    result = executor.observe(state, public, j.asarray([owner], j.int32), legal[:, owner])
    if np.any(_host(result.observation.error)):
        raise ValueError('current direct-encoder input error')
    return result


def _prediction(callback, inputs):
    # Exact original v5 model Output field/shape contract, no fake aux heads.
    result = jax.block_until_ready(callback(inputs))
    logits, value, valid = map(np.asarray, (result.policy_logits, result.value, result.valid))
    if (logits.shape != (1, 9) or value.shape != (1,) or valid.shape != (1,)
            or valid.dtype != np.bool_ or not valid[0] or
            not np.isfinite(logits).all() or not np.isfinite(value).all() or abs(value[0]) > 1):
        raise ValueError('current public actor output refused/nonfinite')
    return tuple(map(float, logits[0])), float(value[0])


def _same_inputs(actual, expected):
    if set(actual) != set(expected):
        raise ValueError('actual learner root input fields differ')
    for key in actual:
        if not np.array_equal(_host(actual[key]), _host(expected[key])):
            raise ValueError('actual learner root input differs: '+key)


def price(plans, materializer, executor, recipe, *, root_logits, root_legal,
          root_inputs, opponent_forward, learner_forward, enable_device_teacher=False):
    """Price ALL legal root/reply pairs in BOTH declared source worlds.

    One original chance block is sampled per world and reused by every joint
    branch, exactly as the original depth1 provider. No output chooses an action
    in a real game. All source/materialization/callback failures remain unknown;
    neither world nor positive reply mass is renormalized away.
    """
    if (enable_device_teacher is not True or type(plans) is not source_plans.Plans
            or type(materializer) is not world_state.Tables or type(executor) is not Executor
            or type(recipe) is not original.Recipe):
        raise ValueError('explicit actual direct-device teacher inputs required')
    root = plans.root
    if (plans.source_hash != materializer.source.source_hash or
            plans.source_rows_sha256 != materializer.source.source_rows_sha256 or
            root.source_binding_sha256 != materializer.source.binding_sha256 or
            plans.selected != recipe.selected(root.address)):
        raise ValueError('same root/source/original recipe selection required')
    counts = dict(reply_policy=0, public_leaf=0, chance=0, transitions=0)
    if not plans.selected:
        return Result('UNSELECTED', None, (), (), 0, None, counts, None)
    if (len(plans.starts) != 2 or tuple(start.world for start in plans.starts) != (0, 1)):
        raise ValueError('both original world slots required in declared order')
    # Validate original arithmetic inputs without inventing a reply denominator.
    root_legal = original._mask(root_legal)
    root_logits = tuple(root_logits)
    original._probabilities(root_logits, root_legal)
    order = recipe.pricing_order(root.address, root_logits, root_legal)
    worlds, reports, prices = [], [], []
    approximations = []
    for start in plans.starts:
        world = world_state.materialize(root, start, materializer, recipe)
        worlds.append(world)
        report = dict(world=start.world, posterior_seed=start.seed, kernel_seed=world.kernel_seed,
            status='REFUSED', reason=None, required_pairs=None, completed_pairs=0,
            unknown_pairs=None, joint_values=None)
        reports.append(report)
        if world.state is None:
            report['reason'] = 'physical-materialization:'+','.join(world.refusals)
            continue
        try:
            own = _view(executor, world.state, world.public, root.address.owner)
            own_legal = tuple(map(bool, _host(own.observation.legal_action_mask)[0]))
            if own_legal != root_legal:
                raise ValueError('same actual root legal-action denominator required')
            _same_inputs(own.inputs, root_inputs)
            peer = _view(executor, world.state, world.public, 1-root.address.owner)
            peer_legal = tuple(map(bool, _host(peer.observation.legal_action_mask)[0]))
            requested = any(peer_legal)
            if requested:
                counts['reply_policy'] += 1
                peer_logits, _ = _prediction(opponent_forward, peer.inputs)
            else:
                peer_logits = (0.,)*9
            if not world.reply_private_history_available:
                approximations.append(dict(world=start.world, **REPLY_HISTORY_APPROXIMATION))
            replies = tuple(i for i, use in enumerate(peer_legal) if use) if requested else (0,)
            grid = [[None]*9 for _ in range(9)]
            chance = executor.sample(world.state)
            counts['chance'] += 1
            required = len(order)*len(replies)
            report.update(required_pairs=required, opponent_requested=requested,
                reply_legal=peer_legal, reply_logits=peer_logits)
            errors = []
            for action in order:
                for reply in replies:
                    try:
                        actions = j.full((1, 2), -1, j.int32)
                        chosen = executor.dispatch(own.observation, world.state,
                            own.observer_sides, j.asarray([action], j.int32))
                        actions = actions.at[:, root.address.owner].set(chosen)
                        if requested:
                            chosen = executor.dispatch(peer.observation, world.state,
                                peer.observer_sides, j.asarray([reply], j.int32))
                            actions = actions.at[:, 1-root.address.owner].set(chosen)
                        counts['transitions'] += 1
                        after = jax.block_until_ready(executor.advance(world.state,
                            world.public, actions, chance))
                        if bool(_host(after.step.invalid_actions)[0]):
                            raise ValueError('hypothetical original action route invalid')
                        if bool(_host(after.step.state.done)[0]):
                            winner = int(_host(after.step.state.winner)[0])
                            if winner not in (0, 1, 2):
                                raise ValueError('hypothetical terminal winner unavailable')
                            value = 0. if winner == 2 else (1. if winner == root.address.owner else -1.)
                        else:
                            view = _view(executor, after.step.state, after.trackers, root.address.owner)
                            counts['public_leaf'] += 1
                            _, value = _prediction(learner_forward, view.inputs)
                        grid[action][reply] = value
                        report['completed_pairs'] += 1
                    except (ValueError, RuntimeError, TypeError, KeyError, IndexError) as error:
                        errors.append((action, reply, type(error).__name__+': '+str(error)))
            report.update(status='COMPLETE' if not errors else 'REFUSED',
                unknown_pairs=required-report['completed_pairs'], joint_values=grid,
                reason=None if not errors else 'unknown-required-public-leaf', errors=tuple(errors))
            prices.append(original.WorldPrices(requested, peer_legal, peer_logits, grid))
        except (ValueError, RuntimeError, TypeError, KeyError, IndexError) as error:
            report['reason'] = type(error).__name__+': '+str(error)
    missing = 2-len(prices)
    if missing:
        return Result('REFUSED', None, tuple(reports), tuple(worlds), missing,
            'unknown-required-world-or-reply-denominator', counts, tuple(approximations))
    reduced = original.reduce_completed_q(root_logits=root_logits, root_legal=root_legal, worlds=prices)
    return Result(reduced.status, reduced, tuple(reports), tuple(worlds), 0,
        reduced.reason, counts, tuple(approximations))
