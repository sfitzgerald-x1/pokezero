"""DepthN live root operators: which legal action a priced root plays (pure Python, no JAX).

One module for every reader, so the live search, the panel's declaration, the RR scorer and the TARGET_ARGMAX replay check
share one definition (the RR scorer, scripts/depthn_rr_score.py, imports argmax and target_argmax from here):
- ARGMAX_MEAN_COMPLETED_Q (the default; every screen through screen-r3): gpu_engine.search_hedge.averaging's argmax over the
  legal actions of q(a), the equal-weight mean over worlds of each world's completed Q (lowest index on ties).
- TARGET_ARGMAX (screen-r4b): `target_argmax`, the RR scorer's operator of that name: the argmax over the legal actions of the
  root's policy_target (constructor_worlds' composition, the improved target softmax(root_logits + completed Q) the live
  response reports), lowest index on ties; when that target is missing or not finite on a legal action, the argmax of
  root_logits + q (the same target's logits), q averaging's per-action mean over worlds.
"""
import math

ARGMAX_MEAN_COMPLETED_Q = 'ARGMAX_MEAN_COMPLETED_Q'
TARGET_ARGMAX = 'TARGET_ARGMAX'
ROOT_OPERATORS = (ARGMAX_MEAN_COMPLETED_Q, TARGET_ARGMAX)   # the declarable live root operators; the first is the default
IMPLS = {ARGMAX_MEAN_COMPLETED_Q: 'search_hedge.averaging', TARGET_ARGMAX: 'depthn_root_operator.target_argmax'}


def argmax(values, actions):
    """Highest value over `actions`, lowest index on ties."""
    return max(actions, key=lambda a: (values[a], -a))


def target_argmax(target, root_legal, root_logits, q):
    """(the live policy_target's argmax over the legal actions, False), or (argmax of root_logits + q, True) without it."""
    actions = [a for a in range(9) if root_legal[a]]
    if (type(target) is list and len(target) == 9
            and all(type(target[a]) in (int, float) and math.isfinite(target[a]) for a in actions)):return argmax(target, actions), False
    return argmax({a: float(root_logits[a]) + q[a] for a in actions}, actions), True
