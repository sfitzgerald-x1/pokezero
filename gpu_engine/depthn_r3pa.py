"""R3pa: the depth-N search's root decision with each world's ROOT opponent reply prior replaced by the cross-world average of the
worlds' priors, mapped by lane label (mc_ladder_backline's R3pa rung, as a pure core for the live root operator). Host code only:
the standard library and search_hedge / v5_gumbel (no JAX; `import pokezero` itself loads torch, as search_hedge does); no engine,
model or device, and nothing from depthn_live_search.

World w (a v5_gumbel.WorldPrices, as depthn_backups builds them) has a grid G_w[a][b] (our action a x opponent reply lane b) and a
root reply prior pi_w(b) (v5_gumbel._probabilities of opponent_logits over opponent_legal, its kept lanes; whatever logits the
prices carry, so after a reply-model substitution where there is one). Averaging plays argmax_a q(a), q(a) = (1/W) sum_w sum_b
pi_w(b) G_w[a][b] (search_hedge.averaging); NOISE_GATED_K2 plays that choice only when it differs from greedy (the root logits'
argmax over the legal actions) and the mean per-world gap Q_w(choice) - Q_w(greedy) exceeds k = 2 standard errors
(search_hedge.noise_gated on per_world_q), else greedy (depthn_live_search.noise_gate's rule).

Lane labels (mc_ladder_backline.labels): per world nine entries, move:<id>, switch:<species> for a species seen in every world,
switch:? (UNSEEN) for a switch into an unseen slot, None for a lane without one. root_priors: pi_bar = sum_w p_w pi_w by label over
the requested worlds (p_w = 1/W over all W worlds, waiting ones included); each requested world's kept lane takes pi_bar of its label,
UNSEEN mass split over the world's own unseen-switch lanes in proportion to its own prior over them (equally where that prior puts
nothing), a label the world lacks drops, then renormalized over its kept lanes; with nothing landing the world keeps its own prior
(unreachable with uniform p_w: a world's own mass always lands on itself). Waiting worlds (opponent not requested) are untouched
(None). This is mc_ladder_backline.prior_average(prices, labels, [1/W] * W) operation for operation (same accumulation order), so the
priors agree bit for bit. A requested world with an unlabelled kept lane raises LaneUnlabelled (mc_ladder_backline.check_labels'
lane_unlabelled). The caller also refuses a world whose kept lanes are not its peer view's legal lanes (check_labels'
kept_matches_peer_legal), which a WorldPrices cannot show.

substitute: the priors enter the search as WorldPrices (a block substitution: decide / noise_gate run unchanged on the result;
substitute_blocks keeps depthn_live_search's pair structure, root_priors taking the flat world and label lists in block order).
Each requested world with a prior gets opponent_logits log p(b) on its legal lanes, ZERO_MASS_LOGIT where p(b) = 0 (a finite value
v5_gumbel._probabilities turns into exactly 0 mass), its illegal lanes' logits as they were; every other field is the world's own.
The softmax of log p reproduces p to rounding (relative error ~1e-16 |log p|), not bit for bit: q can differ from a direct prior
override in the last bits, so an exact near-tie may resolve differently.

Modes (choose), every one through search_hedge's own functions:
- off: today's live NOISE_GATED_K2: averaging on the given prices (top_action), then the gate on them. Labels are not read.
- gated (R3pag): substitute(prices, root_priors), then averaging, then the same gate, both on the substituted prices.
- replace_gate (R3pa): substitute(prices, root_priors), then averaging's choice; no gate.
Decision.q is the mode's averaging q. Refusals: None when a priced cell is unknown (averaging's None). ValueError (LaneUnlabelled is
one) on invalid input, on an invalid averaging result (the live decide's AVERAGING_INVALID_RESULT: an illegal choice or a missing or
non-finite q on a legal action) and on a gate that cannot be evaluated (fewer than two worlds, a non-finite gap: the live
noise_gate's NOISE_GATE_ERROR). A gated mode never plays its ungated choice on an error.
"""
import collections
from dataclasses import dataclass, replace
import math
import statistics

from pokezero import v5_gumbel as original
from . import search_hedge as hedge

MODES = ('off', 'gated', 'replace_gate')
UNSEEN = 'switch:?'   # mc_ladder_backline.UNSEEN
NOISE_GATE_K = 2      # depthn_live_search.NOISE_GATE_K
FALLBACK = 'prior_fallback_worlds'   # mc_ladder_backline.prior_average's counter
# A legal lane without prior mass. _probabilities computes exp(logit - max) over the legal lanes with max >= log(5e-324) > -745
# (the largest lane's log p), so exp(-1e4 - max) underflows to exactly 0.0 for any positive prior; and it is finite (_real).
ZERO_MASS_LOGIT = -1e4


class LaneUnlabelled(ValueError):
    """A requested world's kept reply lane has no label (mc_ladder_backline.check_labels' lane_unlabelled refusal)."""
    reason = 'lane_unlabelled'


@dataclass(frozen=True)
class Gate:
    """NOISE_GATED_K2's outcome (depthn_live_search.noise_gate's record): mean_gap and stderr of Q_w(candidate) - Q_w(greedy), None
    when they are equal."""
    played: int
    candidate: int
    greedy: int
    passed: bool
    mean_gap: float | None
    stderr: float | None


@dataclass(frozen=True)
class Decision:
    """choose()'s result: the played action; averaging's choice (candidate) and q on the mode's prices; the root logits' greedy
    action; the gate (None in replace_gate); the root priors substituted (None in off) and the requested worlds that kept their
    own (nothing landed)."""
    mode: str
    action: int
    candidate: int
    greedy: int
    q: tuple
    gate: Gate | None
    priors: tuple | None
    fallback_worlds: int


def kept(p):
    """A world's kept reply lanes (none when its opponent waits)."""
    return [b for b in range(9) if p.opponent_requested and p.opponent_legal[b]]


def own_prior(p):
    """{lane: probability} over a requested world's kept lanes: averaging's own pi_w (v5_gumbel._probabilities)."""
    probs = original._probabilities(p.opponent_logits, p.opponent_legal)
    return {b: probs[b] for b in kept(p)}


def _label_lists(world_prices, lane_labels):
    prices = tuple(world_prices)
    if not prices:raise ValueError('one or more worlds required')
    if lane_labels is None or len(lane_labels) != len(prices):raise ValueError('one label list per world required')
    labs = [(None,) * 9 if lab is None else tuple(lab) for lab in lane_labels]
    if any(len(lab) != 9 or any(x is not None and type(x) is not str for x in lab) for lab in labs):
        raise ValueError('nine lane labels (str or None) per world required')
    return prices, labs


def check_labels(world_prices, lane_labels):
    """Raise LaneUnlabelled for a requested world with a kept lane without a label (waiting worlds are not read)."""
    for w, (p, lab) in enumerate(zip(world_prices, lane_labels)):
        missing = [b for b in kept(p) if lab[b] is None]
        if missing:raise LaneUnlabelled(f'lane_unlabelled: world {w} kept lanes {missing} have no label')


def label_mass(p, lab):
    """{label: the world's own prior mass on its lanes with that label} (lane order)."""
    mass = collections.defaultdict(float)
    for b, v in own_prior(p).items():mass[lab[b]] += v
    return mass


def spread(mass, target, target_labels):
    """Label mass onto a requested target world's kept lanes (module docstring), renormalized; None when nothing lands."""
    lanes = kept(target);own = own_prior(target);unseen = [b for b in lanes if target_labels[b] == UNSEEN]
    own_unseen = math.fsum(own[b] for b in unseen)
    raw = {}
    for b in lanes:
        if target_labels[b] == UNSEEN:raw[b] = mass.get(UNSEEN, 0.) * (own[b] / own_unseen if own_unseen > 0 else 1. / len(unseen))
        else:raw[b] = mass.get(target_labels[b], 0.)
    total = math.fsum(raw.values())
    return None if not total > 0 else {b: v / total for b, v in raw.items()}


def root_priors(world_prices, lane_labels, *, weights=None, counts=None):
    """Per world, the label-mapped cross-world average prior ({lane: probability} over its kept lanes), or None for a waiting
    world (module docstring). weights: p_w per world (default uniform 1/W, R3pa); counts: a Counter that gains
    prior_fallback_worlds. Raises LaneUnlabelled on an unlabelled kept lane, ValueError on other invalid input."""
    prices, labs = _label_lists(world_prices, lane_labels);check_labels(prices, labs)
    if weights is None:weights = [1. / len(prices)] * len(prices)
    elif len(weights) != len(prices) or any(isinstance(x, bool) or not math.isfinite(x) or x < 0 for x in weights):
        raise ValueError('one finite nonnegative weight per world required')
    mass = collections.defaultdict(float)
    for p, lab, w in zip(prices, labs, weights):
        if not p.opponent_requested:continue
        for label, v in label_mass(p, lab).items():mass[label] += w * v
    out = []
    for p, lab in zip(prices, labs):
        if not p.opponent_requested:out.append(None);continue
        mapped = spread(mass, p, lab)
        if mapped is None and counts is not None:counts[FALLBACK] += 1
        out.append(mapped or own_prior(p))
    return out


def substitute(world_prices, priors):
    """The worlds in order, each requested world with a prior ({lane: probability}, positive mass only on its legal lanes) given
    opponent_logits log p on its legal lanes (ZERO_MASS_LOGIT where p is 0) and its illegal lanes' logits unchanged (module
    docstring); a waiting world (its prior must be None) or a None prior passes through as the same object. ValueError on a bad
    prior."""
    prices = tuple(world_prices);priors = tuple(priors)
    if len(priors) != len(prices):raise ValueError('one prior (or None) per world required')
    out = []
    for w, (p, pr) in enumerate(zip(prices, priors)):
        if not p.opponent_requested and pr is not None:raise ValueError(f'world {w}: a waiting world takes no prior (None)')
        if pr is None:out.append(p);continue
        if any(type(b) is not int or not 0 <= b < 9 for b in pr):raise ValueError(f'world {w}: prior lanes 0..8 required')
        if any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in pr.values()) or not math.fsum(pr.values()) > 0:
            raise ValueError(f'world {w}: a finite nonnegative prior with positive mass required')
        if any(v > 0 and not p.opponent_legal[b] for b, v in pr.items()):raise ValueError(f'world {w}: prior mass on a lane that is not legal')
        logits = tuple((math.log(pr[b]) if pr.get(b, 0.) > 0 else ZERO_MASS_LOGIT) if p.opponent_legal[b] else p.opponent_logits[b]
                       for b in range(9))
        new = replace(p, opponent_logits=logits)
        probs = original._probabilities(new.opponent_logits, new.opponent_legal)
        if any(probs[b] != 0. for b in range(9) if p.opponent_legal[b] and not pr.get(b, 0.) > 0):   # cannot happen (ZERO_MASS_LOGIT)
            raise ValueError(f'world {w}: a zero-mass lane kept mass')
        out.append(new)
    return out


def substitute_blocks(blocks, priors):
    """substitute on depthn_live_search's blocks (a list of world pairs; their worlds in block order are averaging's worlds) with
    priors the flat per-world list in that order (pair 0's two worlds, then pair 1's, ...): the same pair structure, a list of
    tuples."""
    blocks = [tuple(pair) for pair in blocks];flat = substitute([w for pair in blocks for w in pair], priors);out = [];i = 0
    for pair in blocks:out.append(tuple(flat[i:i + len(pair)]));i += len(pair)
    return out


def _checked(choice, legal):
    """Averaging's played action (top_action) after the live decide's validity check (AVERAGING_INVALID_RESULT)."""
    action = hedge.top_action(choice.strategy);q = choice.q
    if not legal[action] or any(legal[a] != (q[a] is not None) or legal[a] and not math.isfinite(q[a]) for a in range(9)):
        raise ValueError('averaging_invalid_result')
    return action


def _gate(prices, legal, root_logits, candidate, k):
    """depthn_live_search.noise_gate's rule: noise_gated(per_world_q of the priced worlds, candidate, logits_greedy, k); ValueError
    when it cannot be evaluated."""
    gs = hedge.grids(prices, legal)
    if gs is None or len(gs) < 2:raise ValueError('two or more priced worlds required')
    qw = hedge.per_world_q(gs);greedy = hedge.logits_greedy(root_logits, legal);played = hedge.noise_gated(qw, candidate, greedy, k)
    gaps = None if candidate == greedy else [q[candidate] - q[greedy] for q in qw]
    if not legal[played] or gaps is not None and not all(math.isfinite(g) for g in gaps):raise ValueError('a non-finite gap or illegal action')
    mean_gap, stderr = (None, None) if gaps is None else (statistics.fmean(gaps), statistics.stdev(gaps) / math.sqrt(len(gaps)))
    return Gate(played, candidate, greedy, played == candidate, mean_gap, stderr)


def choose(world_prices, root_legal, root_logits, lane_labels, mode, *, k=NOISE_GATE_K):
    """One root's Decision under `mode` (module docstring), or None when a priced cell is unknown; raises ValueError (LaneUnlabelled
    among them). lane_labels is not read in 'off' (None allowed)."""
    if mode not in MODES:raise ValueError(f'mode {mode!r} not in {MODES}')
    prices = tuple(world_prices);legal = tuple(root_legal);priors = None;counts = collections.Counter()
    if mode != 'off':priors = tuple(root_priors(prices, lane_labels, counts=counts));prices = tuple(substitute(prices, priors))
    choice = hedge.averaging(prices, legal)
    if choice is None:return None
    candidate = _checked(choice, legal)
    if mode == 'replace_gate':
        return Decision(mode, candidate, candidate, hedge.logits_greedy(root_logits, legal), choice.q, None, priors, counts[FALLBACK])
    gate = _gate(prices, legal, root_logits, candidate, k)
    return Decision(mode, gate.played, candidate, gate.greedy, choice.q, gate, priors, counts[FALLBACK])
