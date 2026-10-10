"""Defensive historical-action proposals with explicit importance correction.

Public action matching is guidance, NOT an eligibility certificate. Ten percent
of the proposal is always the original policy, preserving all its support even
when a called/forced/interrupted action makes the public identifier misleading.
The original chance kernel and exact public-history potential remain unchanged.
"""
from collections.abc import Mapping, Sequence
import math

from .paper_reference import ReferenceRefusal
from ..public_action_capture import public_action_identifiers_from_protocol_lines


def normalized_policy(legal, weights):
    legal, weights = tuple(legal), tuple(weights)
    if (not legal or len(set(legal)) != len(legal) or len(legal) != len(weights)
            or any(type(a) is not int for a in legal)
            or any(not math.isfinite(p) or p < 0 for p in weights)):
        raise ReferenceRefusal('historical proposal requires a finite valid full policy')
    total = math.fsum(weights)
    if not math.isfinite(total) or total <= 0:
        raise ReferenceRefusal('historical proposal has no original policy support')
    return legal, tuple(p / total for p in weights)


def defensive_action_proposal(legal, weights, compatible, *, guided_fraction=.9):
    """Return q and pi, including all ambiguous matches; never force one action.

    q(a) = (1-g)*pi(a) + g*pi(a)*1[a in C]/pi(C). C need not
    contain every possible explanation because the first term has full support.
    Empty/zero-mass guidance means q=pi, not impossibility or a raw-policy fallback.
    """
    if (type(guided_fraction) not in (int, float) or not math.isfinite(guided_fraction)
            or not 0 <= guided_fraction < 1):
        raise ReferenceRefusal('defensive proposal must preserve positive original support')
    legal, prior = normalized_policy(legal, weights)
    compatible = frozenset(compatible)
    if any(type(a) is not int or a not in legal for a in compatible):
        raise ReferenceRefusal('historical guidance contains a nonlegal action')
    mass = math.fsum(p for a, p in zip(legal, prior) if a in compatible)
    fraction = guided_fraction if mass > 0 else 0.
    proposal = tuple((1-fraction)*p + (fraction*p/mass if a in compatible and mass else 0.)
                     for a, p in zip(legal, prior))
    if any(p > 0 and q <= 0 for p, q in zip(prior, proposal)):
        raise ReferenceRefusal('historical guidance lost original policy support')
    return prior, proposal, dict(guided_fraction=fraction, compatible_probability_mass=mass,
                               compatible_actions=sorted(compatible), full_policy_support=True)


def public_guidance_actions(observation, legal, current_history, expected_history, opponent):
    """Suggest next-round public-ID matches using only hypothetical/public data.

    Round boundaries can be ambiguous at interruptions; the defensive mixture
    makes this heuristic safe without claiming its matches are exhaustive.
    Cancellations/cant/called moves and unavailable metadata get no guidance.
    """
    if expected_history[:len(current_history)] != current_history:
        raise ReferenceRefusal('guidance input is not the registered public prefix')
    suffix = []
    for line in expected_history[len(current_history):]:
        if line.startswith(('|turn|', '|upkeep')):
            break
        suffix.append(line)
    identifier = public_action_identifiers_from_protocol_lines(
        suffix, cancellation_players=(opponent,)).get(opponent)
    if identifier is None or identifier.kind not in ('move', 'switch'):
        return (), dict(reason='no unambiguous move/switch guidance', identifier=None)
    candidates = getattr(observation, 'metadata', {}).get('action_candidates')
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return (), dict(reason='missing hypothetical action metadata', identifier=identifier.to_dict())
    normalize = lambda value: ''.join(c for c in str(value).lower() if c.isalnum())
    matches = set()
    for row in candidates:
        if not isinstance(row, Mapping) or type(row.get('action_index')) is not int:
            continue
        action = row['action_index']
        if action not in legal or row.get('kind') != identifier.kind:
            continue
        if identifier.kind == 'move':
            value, expected = row.get('move_id'), identifier.move_id
        else:
            pokemon = row.get('pokemon')
            value = row.get('switched_species')
            if value is None and isinstance(pokemon, Mapping):
                value = pokemon.get('species')
            expected = identifier.switched_species
        if value is not None and normalize(value) == normalize(expected):
            matches.add(action)
    return tuple(sorted(matches)), dict(reason='public next-round heuristic, not eligibility',
                                       identifier=identifier.to_dict())
