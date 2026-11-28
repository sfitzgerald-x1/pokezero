"""Content disjointness for canary constructor pools; no cluster paths or seeds.

The reference inventory maps r1/r2/training/holdout to nonempty lists of
{path, sha256} records. Entire banks are checked, conservatively including
unused rows. Team identity ignores seed, member/move/type ordering and numeric
vocabulary IDs. Gender is assigned separately by the bound gender policy.
"""
import hashlib
import io
import json
from pathlib import Path

from .b4_fidelity_canary import encoded, require


VOCABULARIES = dict(species='species', types='types', moves='moves', item='items',
                    ability='abilities', nature='natures', hidden_power_type='types')
REFERENCE_ROLES = frozenset(('r1', 'r2', 'training', 'holdout'))


def team_hashes(record):
    """Yield canonical team hashes from one verified immutable pool snapshot."""
    import numpy as np
    from gpu_engine import team_pool
    # The engine's constructor loader drops IVs; compare the full stored team
    # definition from the exact digest-bound byte snapshot instead.
    payload = Path(record['path']).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == record['sha256'], 'team bank digest differs')
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        vocabulary = json.loads(archive['metadata_utf8'].tobytes())['vocabulary']
        arrays = {name: archive[name] for name in team_pool.FIELDS}
    del payload
    count = len(arrays['species'])
    require(count > 0, 'empty team bank')
    for name, (shape, _) in team_pool.FIELDS.items():
        require(arrays[name].dtype.kind in 'iu' and arrays[name].shape == (count, *shape),
                'team bank integer shape: ' + name)
    for i in range(count):
        members = []
        for slot in range(6):
            member = {}
            for name in team_pool.FIELDS:
                value = arrays[name][i, slot].tolist()
                if name in VOCABULARIES:
                    vocab = vocabulary[VOCABULARIES[name]]
                    def decode(index):
                        require(type(index) is int and 0 <= index < len(vocab), 'team vocabulary index')
                        require(type(vocab[index]) is str, 'team vocabulary entry')
                        return vocab[index]
                    value = [decode(x) for x in value] if isinstance(value, list) else decode(value)
                member[name] = value
            member['moves'] = sorted(zip(member['moves'], member.pop('pp')))
            member['types'] = sorted(member['types'])
            members.append(encoded(member))
        yield hashlib.sha256(b'[' + b','.join(sorted(members)) + b']').hexdigest()


def validate_team_banks(plan, pool_path, references):
    """Reject missing references, digest mismatches, reuse, or any overlap."""
    require(isinstance(references, dict) and set(references) == REFERENCE_ROLES,
            'team references require r1, r2, training and holdout')
    for role, records in references.items():
        require(isinstance(records, list) and records, 'empty team references: ' + role)
        require(all(isinstance(r, dict) and set(r) == {'path', 'sha256'} for r in records),
                'team reference records require path and sha256')
    candidate = set()
    for digest in team_hashes(dict(path=pool_path, sha256=plan.pool_sha256)):
        require(digest not in candidate, 'team content reuse within canary bank')
        candidate.add(digest)
    indices = [index for iteration in plan.iterations for batch in iteration for game in batch.games for index in game.team_indices]
    require(all(index < len(candidate) for index in indices), 'declared team index outside bank')
    counts = {}
    for role in sorted(references):
        counts[role] = 0
        for record in references[role]:
            for digest in team_hashes(record):
                require(digest not in candidate, 'team content overlap: ' + role)
                counts[role] += 1
    return dict(status='DISJOINT', candidate_teams=len(candidate), reference_teams=counts)
