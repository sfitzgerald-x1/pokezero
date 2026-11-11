"""Process-free production Python encoding of one literal captured owner history."""
import hashlib

from pokezero.live_protocol_actor import ActorContract, OwnerSnapshot, PythonBackend, _packet
from pokezero.recorded_encoder_parity import SHAPES


def _lines(lines, owner):
    if owner not in ('p1', 'p2') or type(lines) not in (list, tuple) or not lines:
        raise ValueError('literal nonempty captured owner history required')
    peer = 'p2' if owner == 'p1' else 'p1'
    requests = []
    for line in lines:
        if type(line) is not str or not line.startswith('|') or '\n' in line or '\r' in line:
            raise ValueError('literal single owner lines required')
        if line.startswith('|request|'):
            _packet(line, owner)  # Reject a peer's private request, never filter it.
            requests.append(line)
        parts = line.split('|')
        health = []
        if len(parts) > 4 and parts[1] in ('switch', 'drag', 'replace'):
            health.append((parts[2], parts[4]))
        elif len(parts) > 3 and parts[1] in ('-damage', '-heal', '-sethp'):
            health.append((parts[2], parts[3]))
            if parts[1] == '-sethp' and len(parts) > 5:
                health.append((parts[4], parts[5]))
        for ident, condition in health:
            if ident.startswith(peer) and '/' in condition:
                denominator = condition.split()[0].split('/', 1)[1]
                if denominator != '100':
                    raise ValueError('opponent exact HP is not captured Gen3 percentage owner wire')
    if not requests:
        raise ValueError('actual owning request required for Python column reference')
    return tuple(lines), requests


def observe_owner(config, lines, *, owner, battle_id, stream_id, seed):
    """Use the existing owning backend, without policy/model or simulator setup."""
    lines, requests = _lines(lines, owner)
    contract = ActorContract.champion()
    backend = PythonBackend(contract, config=config, battle_id=battle_id, owner=owner, stream_id=stream_id)
    prefix = hashlib.sha256(''.join(line+'\n' for line in lines).encode()).hexdigest()
    try:
        backend.ingest(lines)
        snapshot = OwnerSnapshot(contract.namespace, battle_id, owner, seed, stream_id, 0,
            lines, prefix, len(requests), requests[-1])
        observation, _ = backend.encode(snapshot)
        return {name:getattr(observation, name) for name in SHAPES}, dict(
            scope='ACTUAL_CAPTURED_OWNING_LINES_PRODUCTION_PYTHON_NO_MODEL', owner=owner,
            owner_lines_sha256=prefix, own_request_count=len(requests),
            namespace=contract.namespace, belief_source=contract.belief_set_source_hash,
            hp_visibility=dict(backend.env._parser.hp_visibility), simulator_started=False)
    finally:
        backend.close()
