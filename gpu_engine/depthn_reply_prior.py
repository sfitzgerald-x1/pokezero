"""The live br_fp reply prior (depthn_live_server --reply-archive): a reply model's logits in place of the champion's
root_peer_logits, so the live search's averaging becomes br_fp (averaging with each world's reply prior from the reply model).

Views (--reply-views):
- eager (default): per priced root, each world's depth-0 peer-view inputs are rebuilt exactly as the RR re-pricer's self-check
  rebuilds them (scripts/depthn_rr_reprice.py PeerInputs.capture, bit-exact with the pricer's own root_peer_logits): the
  world's state and public carry are fetched to the host, device_depth2_pricer._make_views(executor) builds both seats'
  observations eagerly, and the peer's encoder inputs of the root's worlds, stacked in world order, go through the reply
  model's own adapter under jax.jit as one batch. The scorer's br_fp (scripts/depthn_rr_score.py) is the same averaging on
  the same logits, so a replay of the RR rows with this prior plays the scorer's br_fp choice.
- jit: one jitted program per owner seat builds every world's peer inputs (the same views, vmapped over the root's worlds on
  the device) and runs the reply model on them; faster, but its float rounding may differ from the eager path's.
- check: plays eager and also computes the jitted logits, reported beside (the max |logit difference| and the jitted blocks,
  whose averaging the search reports): the validation of jit on the RR rows.
A requested world's opponent logits become the reply model's; a waiting world keeps column 0."""
import time

import numpy as np

from pokezero import v5_gumbel as original

VIEWS = ('eager', 'jit', 'check')


class ReplyPrior:
    def __init__(self, executor, evaluator, params=None, *, views=None, mode='eager'):
        """`evaluator`: the reply model (a champion_evaluator.ChampionEvaluator of the reply archive, or any callable
        (params, inputs) -> an object with policy_logits); `params` default evaluator.params; `mode` in VIEWS."""
        import jax
        from . import device_depth2_pricer as depth2
        if mode not in VIEWS:raise ValueError(f'reply views mode in {VIEWS} required')
        self.views = depth2._make_views(executor) if views is None else views
        self.evaluator, self.params, self.mode = evaluator, (evaluator.params if params is None else params), mode
        self.program = jax.jit(lambda p, x: evaluator(p, x));self._jitted = {}

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
        """The reply model's policy logits [W, 9] float32 for the root's worlds (the eager views)."""
        import jax
        return np.asarray(jax.device_get(self.program(self.params, self.inputs(worlds, owner))).policy_logits, np.float32)

    def jit_logits(self, worlds, owner):
        """The same logits from one jitted program: the views vmapped over the stacked worlds, then the reply model."""
        import jax
        import jax.numpy as jnp
        if any(w.state is None for w in worlds):raise ValueError('a priced root with an unconstructed world')
        if owner not in self._jitted:
            views, evaluator = self.views, self.evaluator
            def program(params, states, publics):
                inputs = jax.vmap(lambda s, p: views(s, p, owner)[3])(states, publics)   # [W, 1, ...] per key: one world each
                inputs = jax.tree.map(lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:]), inputs)
                return evaluator(params, inputs).policy_logits
            self._jitted[owner] = jax.jit(program)
        stack = lambda xs: jax.tree.map(lambda *leaves: jnp.stack(leaves), *xs)
        out = self._jitted[owner](self.params, stack([w.state for w in worlds]), stack([w.public for w in worlds]))
        return np.asarray(jax.device_get(out), np.float32)

    def warm(self, worlds):
        """Compile what a live root of these worlds' shapes runs, for both owner seats, before READY (a seat first compiled on
        its first live root cost a ~7 s outlier): eager and check build both seats' eager views and run the reply program;
        jit and check also compile each seat's jitted program. Returns {owner: seconds}."""
        import jax
        seconds = {}
        for owner in (0, 1):
            started = time.perf_counter()
            if self.mode in ('eager', 'check'):self.logits(worlds, owner)
            if self.mode in ('jit', 'check'):jax.block_until_ready(self.jit_logits(worlds, owner))
            seconds[owner] = time.perf_counter() - started
        return seconds

    @staticmethod
    def substitute(blocks, reply):
        """`blocks` with each requested world's opponent_logits replaced by reply[i] (i in block order)."""
        out, i = [], 0
        for pair in blocks:
            row = []
            for w in pair:   # a world without a surface (None) stays None: averaging refuses the root as it would anyway
                if w is not None:
                    logits = tuple(float(x) for x in reply[i]) if w.opponent_requested else w.opponent_logits
                    w = original.WorldPrices(w.opponent_requested, w.opponent_legal, logits, w.joint_values)
                row.append(w);i += 1
            out.append(tuple(row))
        return out

    def apply(self, blocks, worlds, owner):
        """(blocks with each requested world's opponent_logits replaced by the reply model's, seconds, check or None).
        `blocks`: the root's backed-up worlds, pairs in block order, which is `worlds`' order. check (mode 'check' only):
        dict(blocks=the jitted logits' blocks, max_abs_logit_diff over the requested worlds)."""
        started = time.perf_counter()
        flat = [w for pair in blocks for w in pair]
        if len(flat) != len(worlds):raise ValueError(f'{len(flat)} priced worlds for {len(worlds)} constructed')
        reply = self.jit_logits(worlds, owner) if self.mode == 'jit' else self.logits(worlds, owner)
        out = self.substitute(blocks, reply);seconds = time.perf_counter() - started;check = None
        if self.mode == 'check':   # eager plays: a failure of the jitted side is recorded, never a refusal of the root
            try:
                other = self.jit_logits(worlds, owner)
                asked = [i for i, w in enumerate(flat) if w is not None and w.opponent_requested]
                diff = float(np.max(np.abs(other[asked].astype(np.float64) - reply[asked]))) if asked else 0.
                check = dict(blocks=self.substitute(blocks, other), max_abs_logit_diff=diff)
            except Exception as error:
                check = dict(error=f'{type(error).__name__}: {str(error)[:200]}')
        return out, seconds, check
