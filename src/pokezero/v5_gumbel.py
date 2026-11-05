"""Bounded D target arithmetic, NOT a sampled-world/engine teacher provider.

The two worlds must be priced by the future public-root branch constructor.
This reducer cannot certify their provenance or manufacture leaf observations.
No actor action, outcome, GAE, model, sampler, or live session enters this API.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real

from . import v5_contract as c

SCHEMA = 'pokezero.v5.depth1_completed_q.v1'
DECISIONS = (
    'Address-domain SHA256 modulo4 selects one quarter without consuming actor RNG.',
    'Two equally weighted posterior draws are shared across every legal root action.',
    'Every legal reply is priced using its own-seat public policy; unknown leaves refuse the label.',
    'Full coverage uses Gumbel root pricing order, no halving; noise does not change the completed-Q target.',
    'The target is softmax(raw root logits + Q), fixed Qscale1; root value is its Q expectation.',
    'Literal outcome/root-value equal mixing occurs once downstream in V5Labels.columns only.',
)


def _real(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError('finite real scalar required')
    return float(value)


def _mask(mask):
    mask = tuple(mask)
    if len(mask) != 9 or any(type(v) is not bool for v in mask):
        raise ValueError('literal nine-action bool mask required')
    return mask


def _probabilities(logits, legal):
    logits = tuple(_real(v) for v in logits)
    if len(logits) != 9 or not any(legal):
        raise ValueError('finite nine logits and a legal action required')
    maximum = max(v for v, active in zip(logits, legal) if active)
    weights = tuple(math.exp(v - maximum) if active else 0. for v, active in zip(logits, legal))
    total = math.fsum(weights)
    return tuple(v / total for v in weights)


@dataclass(frozen=True)
class Recipe:
    seed: int

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError('explicit unsigned64 teacher seed required')

    def _digest(self, address, domain, index=0):
        if type(address) is not c.DecisionAddress:
            raise ValueError('original typed decision address required')
        payload = (SCHEMA, domain, self.seed, address.stream, address.frame, address.owner, index)
        return hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).digest()

    def selected(self, address):
        return self._digest(address, 'selection')[0] % 4 == 0

    def world_seed(self, address, world):
        if type(world) is not int or world not in (0, 1):
            raise ValueError('the declared two world indices are0,1')
        return int.from_bytes(self._digest(address, 'posterior-world', world), 'big')

    def pricing_order(self, address, logits, legal):
        legal = _mask(legal)
        logits = tuple(logits)
        _probabilities(logits, legal)
        scores = []
        for action in range(9):
            if legal[action]:
                # 48 bits keep u strictly inside(0,1) after float conversion.
                u = (int.from_bytes(self._digest(address, 'gumbel', action)[:6], 'big') + .5) / 2**48
                scores.append((float(logits[action]) - math.log(-math.log(u)), action))
        return tuple(action for _, action in sorted(scores, reverse=True))


@dataclass(frozen=True)
class WorldPrices:
    """Untrusted numeric pricing input; ownership must be established upstream.

    joint_values[root action][reply action] is our public leaf value after that
    actual branch. None means unknown/refused, never a zero replacement. A
    waiting opponent has no legal actions and one unit-mass no-action column0.
    """
    opponent_requested: bool
    opponent_legal: tuple[bool, ...]
    opponent_logits: tuple[float, ...]
    joint_values: tuple[tuple[float | None, ...], ...]

    def __post_init__(self):
        object.__setattr__(self, 'opponent_legal', _mask(self.opponent_legal))
        object.__setattr__(self, 'opponent_logits', tuple(_real(v) for v in self.opponent_logits))
        object.__setattr__(self, 'joint_values', tuple(tuple(row) for row in self.joint_values))
        if (type(self.opponent_requested) is not bool or len(self.opponent_logits) != 9
                or any(self.opponent_legal) != self.opponent_requested
                or len(self.joint_values) != 9 or any(len(row) != 9 for row in self.joint_values)):
            raise ValueError('exact requested/waiting reply surface and9x9 pricing required')
        for row in self.joint_values:
            for value in row:
                if value is not None and not -1 <= _real(value) <= 1:
                    raise ValueError('public leaf value outside[-1,1]')
        object.__setattr__(self, 'joint_values', tuple(tuple(None if v is None else float(v)
            for v in row) for row in self.joint_values))


@dataclass(frozen=True)
class Reduction:
    status: str
    improvement: c.RootImprovement | None
    required_pairs: int
    unknown_pairs: int
    # One full unit of reply mass for EACH legal root action, across worlds.
    unknown_reply_mass: float
    reason: str | None


def reduce_completed_q(*, root_logits, root_legal, worlds):
    """Retain the full legal denominator; never normalize surviving branches.

    Every legal branch is required, including an exp-underflowed reply. This
    stricter numerical rule prevents mathematically positive mass disappearing.
    Illegal slots are ignored and cannot be used to fill a missing legal slot.
    """
    legal = _mask(root_legal)
    root_logits = tuple(root_logits)
    _probabilities(root_logits, legal)
    worlds = tuple(worlds)
    if len(worlds) != 2 or any(type(world) is not WorldPrices for world in worlds):
        raise ValueError('both declared posterior worlds required; no survivor-only reduction')
    required = unknown = 0
    missing_mass = []
    weighted = [[] for _ in range(9)]
    for world in worlds:
        replies = tuple(i for i, active in enumerate(world.opponent_legal) if active) if world.opponent_requested else (0,)
        probabilities = _probabilities(world.opponent_logits, world.opponent_legal) if world.opponent_requested else (1.,) + (0.,) * 8
        for action, active in enumerate(legal):
            if not active:
                continue
            required += len(replies)
            for reply in replies:
                value = world.joint_values[action][reply]
                if value is None:
                    unknown += 1
                    missing_mass.append(probabilities[reply] / 2)
                else:
                    weighted[action].append(probabilities[reply] * float(value) / 2)
    if unknown:
        return Reduction('REFUSED', None, required, unknown, math.fsum(missing_mass), 'unknown-required-public-leaf')
    q = tuple(math.fsum(weighted[i]) if active else None for i, active in enumerate(legal))
    target = _probabilities(tuple(float(logit) + (q[i] or 0.) for i, logit in enumerate(root_logits)), legal)
    value = math.fsum(p * v for p, v in zip(target, q) if v is not None)
    result = c.RootImprovement(target, value, q, worlds=2)
    result.validate(legal)
    return Reduction('LABELED', result, required, 0, 0., None)
