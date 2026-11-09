"""The live br_fp reply prior (depthn_live_server --reply-archive): a reply model's logits in place of the champion's
root_peer_logits, so the live search's averaging becomes br_fp (averaging with each world's reply prior from the reply model).

Per priced root, each world's depth-0 peer-view inputs are rebuilt exactly as the RR re-pricer's self-check rebuilds them
(scripts/depthn_rr_reprice.py PeerInputs.capture, bit-exact with the pricer's own root_peer_logits): the world's state and
public carry are fetched to the host, device_depth2_pricer._make_views(executor) builds both seats' observations eagerly, and
the peer's encoder inputs of the root's worlds, stacked in world order, go through the reply model's own adapter under
jax.jit as one batch. A requested world's opponent logits become the reply model's; a waiting world keeps column 0. The
scorer's br_fp (scripts/depthn_rr_score.py) is the same averaging on the same logits, so a replay of the RR rows with this
prior plays the scorer's br_fp choice."""
import time

import numpy as np

from pokezero import v5_gumbel as original


class ReplyPrior:
    def __init__(self, executor, evaluator, params=None, *, views=None):
        """`evaluator`: the reply model (a champion_evaluator.ChampionEvaluator of the reply archive, or any callable
        (params, inputs) -> an object with policy_logits); `params` default evaluator.params."""
        import jax
        from . import device_depth2_pricer as depth2
        self.views = depth2._make_views(executor) if views is None else views
        self.evaluator, self.params = evaluator, (evaluator.params if params is None else params)
        self.program = jax.jit(lambda p, x: evaluator(p, x))

    def inputs(self, worlds, owner):
        """The peer encoder inputs of `worlds` (each with .state and .public), stacked in order: {key: array [W, ...]}."""
        import jax
        batch = []
        for w in worlds:
            if w.state is None:raise ValueError('a priced root with an unconstructed world')
            state, public = jax.device_get((w.state, w.public))   # as constructor_worlds.build and the re-pricer read a world
            _, _, _, inputs = self.views(state, public, owner)
            batch.append({k: np.asarray(v) for k, v in inputs.items()})
        return {k: np.concatenate([b[k] for b in batch]) for k in batch[0]}

    def logits(self, worlds, owner):
        """The reply model's policy logits [W, 9] float32 for the root's worlds."""
        import jax
        return np.asarray(jax.device_get(self.program(self.params, self.inputs(worlds, owner))).policy_logits, np.float32)

    def apply(self, blocks, worlds, owner):
        """(blocks with each requested world's opponent_logits replaced by the reply model's, seconds). `blocks`: the root's
        backed-up worlds, pairs in block order, which is `worlds`' order."""
        started = time.perf_counter()
        flat = [w for pair in blocks for w in pair]
        if len(flat) != len(worlds):raise ValueError(f'{len(flat)} priced worlds for {len(worlds)} constructed')
        reply = self.logits(worlds, owner);out, i = [], 0
        for pair in blocks:
            row = []
            for w in pair:   # a world without a surface (None) stays None: averaging refuses the root as it would anyway
                if w is not None:
                    logits = tuple(float(x) for x in reply[i]) if w.opponent_requested else w.opponent_logits
                    w = original.WorldPrices(w.opponent_requested, w.opponent_legal, logits, w.joint_values)
                row.append(w);i += 1
            out.append(tuple(row))
        return out, time.perf_counter() - started
