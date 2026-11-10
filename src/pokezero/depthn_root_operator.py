"""DepthN live root operators: which legal action a priced root plays (pure Python, no JAX).

One module for every reader, so the live search, the panel's declaration, the RR scorer and the replay checks share one
definition (the RR scorer, scripts/depthn_rr_score.py, imports argmax, target_argmax, per_world_q and noise_gated from here):
- ARGMAX_MEAN_COMPLETED_Q (the default; every screen through screen-r3): gpu_engine.search_hedge.averaging's argmax over the
  legal actions of q(a), the equal-weight mean over worlds of each world's completed Q (lowest index on ties).
- TARGET_ARGMAX (screen-r4b): `target_argmax`, the RR scorer's operator of that name: the argmax over the legal actions of the
  root's policy_target (constructor_worlds' composition, the improved target softmax(root_logits + completed Q) the live
  response reports), lowest index on ties; when that target is missing or not finite on a legal action, the argmax of
  root_logits + q (the same target's logits), q averaging's per-action mean over worlds.
- NOISE_GATED_K2 (screen-r4b): `noise_gated` with k = 2, the RR scorer's noise_gated:k=2. With a* averaging's choice and g the
  greedy action: a* when a* != g and mean_w gap_w > 2 sd_w(gap_w) / sqrt(W), else g; gap_w = Q_w(a*) - Q_w(g),
  Q_w(a) = sum_b pi_w(b) G_w[a][b] (`per_world_q`: every world of the root, W of them; pi_w the world's reply prior
  search_hedge.grid builds, softmax of its root_peer_logits over its kept replies, or the waiting column 0 at mass 1), sd the
  sample standard deviation (statistics.stdev), the comparison strict (a mean exactly at the threshold, and sd = 0 with a zero
  mean gap, play g). g, `logits_greedy`: the argmax of root_logits over the legal actions, lowest index on ties (the RR
  scorer's greedy_argmax; its greedy_check rule). The RR scorer's own greedy is the live actor's greedy_action_index (the
  champion's CPU FP32 policy, argmax of its masked softmax), which the search never sees; where the two differ the scorer's
  choice can differ from the live one (scripts/depthn_noise_gated_replay_check.py and
  scripts/depthn_noise_gated_override_rate.py count it).
RESPONSE_KEYS: the keys a response under each operator adds (the default none; TARGET_ARGMAX target_fallback; NOISE_GATED_K2
noise_gate = dict(passed: the gate passed and a* != g was played, greedy: g, candidate: a*)).
"""
import math
import statistics

ARGMAX_MEAN_COMPLETED_Q = 'ARGMAX_MEAN_COMPLETED_Q'
TARGET_ARGMAX = 'TARGET_ARGMAX'
NOISE_GATED_K2 = 'NOISE_GATED_K2'
ROOT_OPERATORS = (ARGMAX_MEAN_COMPLETED_Q, TARGET_ARGMAX, NOISE_GATED_K2)   # the declarable live root operators; the first is the default
IMPLS = {ARGMAX_MEAN_COMPLETED_Q: 'search_hedge.averaging', TARGET_ARGMAX: 'depthn_root_operator.target_argmax',
         NOISE_GATED_K2: 'depthn_root_operator.noise_gated'}
RESPONSE_KEYS = {ARGMAX_MEAN_COMPLETED_Q: (), TARGET_ARGMAX: ('target_fallback',), NOISE_GATED_K2: ('noise_gate',)}
NOISE_K2 = 2   # NOISE_GATED_K2's k: the RR scorer's noise_gated:k=2 (its NOISE_K value, an int)


def argmax(values, actions):
    """Highest value over `actions`, lowest index on ties."""
    return max(actions, key=lambda a: (values[a], -a))


def target_argmax(target, root_legal, root_logits, q):
    """(the live policy_target's argmax over the legal actions, False), or (argmax of root_logits + q, True) without it."""
    actions = [a for a in range(9) if root_legal[a]]
    if (type(target) is list and len(target) == 9
            and all(type(target[a]) in (int, float) and math.isfinite(target[a]) for a in actions)):return argmax(target, actions), False
    return argmax({a: float(root_logits[a]) + q[a] for a in actions}, actions), True


def per_world_q(gs):
    """Q_w(a) = sum_b pi_w(b) G_w[a][b] per world: the grid's reply prior over kept replies, or the waiting column 0."""
    return [{a: math.fsum(g.prior[k] * g.values[i, k] for k in range(len(g.replies))) for i, a in enumerate(g.actions)} for g in gs]


def noise_gated(qw, chosen, greedy, k):
    """Averaging's choice only when it differs from greedy and the per-world gap's mean exceeds k standard errors."""
    if chosen == greedy:return greedy
    gaps = [q[chosen] - q[greedy] for q in qw]
    return chosen if statistics.fmean(gaps) > k * statistics.stdev(gaps) / math.sqrt(len(gaps)) else greedy


def logits_greedy(root_logits, root_legal):
    """NOISE_GATED_K2's greedy: the argmax of root_logits over the legal actions, lowest index on ties (the RR scorer's
    greedy_argmax)."""
    return argmax([float(x) for x in root_logits], [a for a in range(9) if root_legal[a]])


def noise_gated_k2(gs, chosen, root_logits, root_legal):
    """NOISE_GATED_K2 on one root: (played, noise_gate). `gs` the root's search_hedge grids (every world, block order),
    `chosen` averaging's action a*; played = noise_gated(per_world_q(gs), a*, logits_greedy(root_logits, root_legal), 2);
    noise_gate = dict(passed=played != greedy, greedy=greedy, candidate=a*)."""
    greedy = logits_greedy(root_logits, root_legal)
    played = noise_gated(per_world_q(gs), chosen, greedy, NOISE_K2)
    return played, dict(passed=played != greedy, greedy=greedy, candidate=chosen)
