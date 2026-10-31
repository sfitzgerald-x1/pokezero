"""Literal owning history -> ORIGINAL public materialization/sampling inputs.

This is the public half of a future device branch constructor, not that
constructor or a ready teacher. It never reads a live Session, peer request,
pool, engine state, model, or native simulator. Hypothetical starts below are
sampled by the original public-belief planner, once per declared world, with no
retry/refill. They are not current JAX states or scientific evaluation roots.
"""
from dataclasses import dataclass, replace
import hashlib
import random

from . import v5_contract as c
from .determinization import _gen3_randbat_belief_start_override_result
from .live_protocol_actor import ActorContract, PythonBackend, _packet
from .local_showdown import _public_materialization_payload
from .policy import PolicyContext
from .trajectory import BattleTrajectory
from .v5_gumbel import Recipe


@dataclass(frozen=True)
class PublicRoot:
    address: c.DecisionAddress
    owner_lines_sha256: str
    context: PolicyContext
    materialization_payload: dict
    unsupported_counts: tuple[tuple[str, int], ...]
    # Capturing this payload does not synchronize ANY device provenance.
    device_branch_ready: bool = False


def _owning_lines(lines, owner):
    lines = tuple(lines)
    if (not lines or any(type(line) is not str or not line.startswith('|')
            or '\n' in line or '\r' in line for line in lines)
            or lines.count('|start') != 1 or '|gen|3' not in lines or '|gametype|singles' not in lines):
        raise ValueError('complete literal Gen3 singles owning prefix from start required')
    peer = 'p2' if owner == 'p1' else 'p1'
    requests = []
    for line in lines:
        if line.startswith('|request|'):
            _packet(line, owner)  # Reject a foreign request rather than filtering it.
            requests.append(line)
        parts = line.split('|')
        if parts[1] == 'split':raise ValueError('owning wire required, no omniscient split stream')
        health = []
        if len(parts) > 4 and parts[1] in ('switch', 'drag', 'replace'):
            health.append((parts[2], parts[4]))
        elif len(parts) > 3 and parts[1] in ('-damage', '-heal', '-sethp'):
            health.append((parts[2], parts[3]))
            if parts[1] == '-sethp' and len(parts) > 5:health.append((parts[4], parts[5]))
        for ident, condition in health:
            if ident.startswith(peer) and '/' in condition and condition.split()[0].split('/', 1)[1] != '100':
                raise ValueError('opponent exact HP cannot enter owning public materialization')
    if not requests or not lines[-1].startswith('|request|'):
        raise ValueError('current original owning request boundary required')
    return lines


def capture_root(*, config, contract, lines, address, battle_id, seed):
    """Fold synthetic/captured owner wire in the original process-free backend.

    Initial request-known team stats are retained separately from current own
    state; the original materializer retains current/initial requests and PP
    history itself. No opposing numeric action index is invented. The sampler
    uses a new hypothetical party order, not replay of historical slot actions.
    """
    if type(contract) is not ActorContract or type(address) is not c.DecisionAddress:
        raise ValueError('explicit original contract and decision address required')
    if type(seed) is not int or not 0 <= seed < 2**32 or type(battle_id) is not str or not battle_id:
        raise ValueError('explicit original battle identity required')
    owner = ('p1', 'p2')[address.owner]
    lines = _owning_lines(lines, owner)
    backend = PythonBackend(contract, config=config, battle_id=battle_id, owner=owner, stream_id=address.stream)
    try:
        first_observation = None
        for line in lines:
            backend.ingest((line,))
            if first_observation is None and line.startswith('|request|'):
                first_observation = backend.env.observe(owner)
        observation = backend.env.observe(owner)
        materialization = backend.env.public_materialization_state(owner)
        if materialization.replay.requests or materialization.player_id != owner:
            raise ValueError('materialization must strip all replay requests')
        # No artificial trajectory choices: original direct materialization
        # consumes the complete public replay independently of action indexes.
        observation = replace(observation, metadata={**dict(observation.metadata),
            'self_team': first_observation.metadata['self_team']})
        context = PolicyContext(owner, address.frame, battle_id, materialization.format_id, seed, observation,
            (owner,), BattleTrajectory(battle_id, materialization.format_id, seed),
            {owner: tuple(observation.legal_action_mask)}, {owner: observation}, materialization)
        payload = _public_materialization_payload(materialization)
        counts = {}
        for side in payload['sides'].values():
            for reason in side['materializationBlockers']:counts[reason] = counts.get(reason, 0) + 1
        if payload['pendingBatonPassSides']:counts['pending-baton-phase-needs-device-constructor'] = 1
        prefix = hashlib.sha256(''.join(line+'\n' for line in lines).encode()).hexdigest()
        return PublicRoot(address, prefix, context, payload, tuple(sorted(counts.items())))
    finally:
        backend.close()


@dataclass(frozen=True)
class StartPlans:
    starts: tuple  # Both declared world slots retained, even on refusal.
    refusals: tuple[str | None, ...]
    required_worlds: int = 2
    device_branch_ready: bool = False


def plan_starts(root, recipe, *, set_source):
    """Use the existing canonical public sampler; no branch or actor is run."""
    if type(root) is not PublicRoot or type(recipe) is not Recipe:
        raise ValueError('typed public root and bounded recipe required')
    if set_source.metadata.source_hash != root.context.public_materialization_state.belief_engine.set_source.metadata.source_hash:
        raise ValueError('public root and posterior sampler source differ')
    starts, refusals = [], []
    for index in (0, 1):
        try:
            start, reason = _gen3_randbat_belief_start_override_result(context=root.context,
                set_source=set_source, rng=random.Random(recipe.world_seed(root.address, index)),
                team_size=6, canonicalize_revealed_variants=True)
        except ValueError as error:
            start, reason = None, 'public-sampler: '+str(error)
        starts.append(start)
        refusals.append(None if start is not None else (reason or 'public-sampler-unavailable'))
    return StartPlans(tuple(starts), tuple(refusals))
