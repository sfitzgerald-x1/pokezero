"""Deterministic inputs at an owned historical snapshot; no RNG/transition caching."""
from dataclasses import dataclass

from .paper_reference_pending import public_history
from ..actions import ACTION_COUNT
from ..local_showdown import (LocalShowdownEnv, LocalShowdownSnapshot,
                              requested_players_from_requests)


@dataclass(frozen=True)
class HistoricalStageInputs:
    requested: frozenset
    opponent_observation: object
    legal: tuple
    probabilities: tuple
    history: tuple


def build_stage_inputs(*, env, particle, subject, opponent, own, evaluator, need_history):
    """Project one paired snapshot. Only canonical deterministic callers may memoize.

    Canonical projection and memo hits need not restore the search shell: every subsequent pilot
    and actual chance transition atomically restores this same retained snapshot.
    Use the cached public history, never history from a previous trial's shell.
    """
    snapshot = particle.snapshot
    # Only canonical resident snapshots carry choices computed at the paired
    # boundary after both requested players' annotations. Its isolated projector
    # clones that exact parser/tracker state. Do not re-encode the subject merely to
    # read legality, or clone an entire materialization merely to read history.
    # Other transports retain their existing projection/error paths.
    requested = (frozenset(requested_players_from_requests(snapshot.latest_requests))
                 if isinstance(snapshot, LocalShowdownSnapshot) else frozenset())
    paired = (type(env) is LocalShowdownEnv and isinstance(snapshot, LocalShowdownSnapshot)
              and isinstance(snapshot.bridge_snapshot.get('snapshot_id'), str)
              and bool(snapshot.bridge_snapshot['snapshot_id'])
              and snapshot.annotation_cache is not None
              and all(player in snapshot.search_choice_cache for player in requested))
    if not paired:
        env.restore_search_snapshot(snapshot)
        requested = frozenset(env.requested_players())
    elif (env._battle_token is None or not env._search_snapshot_permitted
          or snapshot.format_id != env._format_id
          or snapshot.observation_format_id != env._observation_format_id):
        # Preserve the canonical sampled-world/format guard even when this
        # boundary has no opponent inference or has an invalid own action.
        env.restore_search_snapshot(snapshot)
    if (subject in requested) != (own is not None):
        return None
    if own is not None:
        mask = (tuple(action in snapshot.search_choice_cache[subject] for action in range(ACTION_COUNT))
                if paired else env.observe(subject).legal_action_mask)
        if not 0 <= own < len(mask) or not mask[own]:
            return None
    observation = None
    legal, probabilities = (), ()
    if opponent in requested:
        observation = (env.observe_search_snapshot(snapshot, opponent) if paired
                       else env.observe(opponent))
        legal, evaluated = evaluator(observation)
        legal, probabilities = tuple(legal), tuple(evaluated.priors)
    history = (public_history(snapshot if paired else env.public_materialization_state(subject))
               if need_history else ())
    return HistoricalStageInputs(requested, observation, legal, probabilities, history)
