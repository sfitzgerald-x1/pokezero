"""Action-conditional Monte Carlo labels on real Showdown, for the search value function (design: search-value-function-design,
step M0 and the labelling step).

A root is a real battle state at a decision of our seat:
- 'foulplay': a Foul Play collection game (gpu_engine.fp_reply_data.load_game) replayed in LocalShowdownEnv to one of our seat's
  decisions, both seats playing their recorded actions;
- 'self': champion self-play (both seats sampling at temperature 1) from a fresh seed, played to one of our seat's decisions.
Per legal action a of our seat, K rollouts:
- restore the root snapshot and reseed the simulator with s_k: rollout k uses the same seed for every sibling action
  (common random numbers);
- the opponent's root reply b_k (when it is requested) is drawn once per k from its model and shared by the siblings;
- then both seats sample from their policies at temperature 1 to the end, every draw from a generator seeded by
  (root, k, ply, seat), matched across siblings. The opponent model is the champion for 'self' roots and pi_FP for 'foulplay'
  roots, with its declared view-b and peer-view-v2 preprocessing.
Outcome from our seat: +1 win, -1 loss, 0 draw, CAPPED for a capped game or a rollout past MAX_DECISIONS.
Modes: 'unbatched' (one env, one forward per decision) and 'batched' (envs stepped in lockstep, one forward per policy per step
over every requested observation). Timing: wall seconds and the CPU seconds of this process and its Node children (/proc).
The mc_m0 runner freezes the roster before any rollout, records failures and timing, and emits explicit censor statuses.
"""
import argparse
import dataclasses
import hashlib
import glob
import json
import os
import random
import time
from pathlib import Path

import numpy as np

MAX_DECISIONS = 600
ACTIONS = 9


def process_cpu_seconds(pid=None):
    """utime + stime (seconds) of `pid` (default this process) and every descendant, from /proc."""
    tick = os.sysconf('SC_CLK_TCK');total = 0.;pending = [pid or os.getpid()];seen = set()
    while pending:
        p = pending.pop()
        if p in seen:continue
        seen.add(p)
        try:
            stat = Path(f'/proc/{p}/stat').read_text();fields = stat[stat.rindex(')') + 2:].split()
            total += (int(fields[11]) + int(fields[12])) / tick
            for task in Path(f'/proc/{p}/task').iterdir():
                children = (task / 'children').read_text().split();pending.extend(int(c) for c in children)
        except (OSError, ValueError, IndexError):continue
    return total


class Model:
    """A champion-format checkpoint for batched sampling: probabilities [n, 9] over the legal mask at temperature 1."""

    def __init__(self, path, torch, *, opponent=False):
        from pokezero.neural_policy import load_transformer_checkpoint
        self.torch = torch;self.model, self.result = load_transformer_checkpoint(path, map_location='cpu');self.model.eval()
        self.window = int(self.result.model_config.window_size)
        if self.window != 1:raise ValueError('M0 requires one-frame models')
        self.view = None
        if opponent:
            metadata = torch.load(path, map_location='cpu', weights_only=True).get('fp_reply', {})
            if metadata.get('view') != 'b' or metadata.get('redaction') != 'peer-view-v2':
                raise ValueError('declared red3-full view b / peer-view-v2 required')
            self.view = metadata
        self.views = None

    def reset(self):pass

    def select_action(self, observation, *, rng):
        from types import SimpleNamespace
        p = self.probabilities([observation])[0]
        return SimpleNamespace(action_index=int(p.argmax()), metadata=dict(policy_distribution=p), value_estimate=None)

    def observe(self, env, player, seed):
        observation = env.observe(player)
        if self.view is None:return observation
        from gpu_engine import fp_reply_data as fpd, fp_reply_train as fpt
        peer = fpd.builder_views(env, player, env._latest_requests[player], seed, self.views)['b']
        for name in ('legal_action_mask', 'attention_mask', 'token_type_ids'):
            if not np.array_equal(getattr(peer, name), getattr(observation, name)):
                raise ValueError('view b differs on '+name)
        numeric = np.array([peer.numeric_features], dtype=np.float32)
        victim = 'p2' if player == 'p1' else 'p1'
        credit = np.array([fpd.fp_view_opp_hazard_credit(env.protocol_lines, victim)], dtype=np.float32)
        fpt.redact_peer_view(numeric, 'peer-view-v2', credit)
        return dataclasses.replace(peer, numeric_features=tuple(tuple(float(x) for x in row) for row in numeric[0]))

    def probabilities(self, observations):
        from pokezero.neural_policy import observation_windows_to_torch
        t = observation_windows_to_torch([[o] for o in observations], window_size=self.window, device='cpu')
        with self.torch.no_grad():
            out = self.model(categorical_ids=t['categorical_ids'], numeric_features=t['numeric_features'], token_type_ids=t['token_type_ids'],
                             attention_mask=t['attention_mask'], history_mask=t['history_mask'])
            legal = self.torch.tensor([o.legal_action_mask for o in observations], dtype=self.torch.bool)
            logits = out.policy_logits.masked_fill(~legal, -1e9)
            return self.torch.softmax(logits, dim=-1).numpy().astype(np.float64)


def sample(probs, legal, u):
    """The inverse-CDF draw of `u` in [0, 1) over the legal actions' probabilities."""
    if not 0 <= u < 1:raise ValueError('uniform draw must lie in [0,1)')
    legal = np.asarray(legal, bool);probs = np.asarray(probs, float)
    if legal.shape != (ACTIONS,) or probs.shape != (ACTIONS,) or not legal.any():raise ValueError('legal action shape')
    if not np.isfinite(probs).all() or (probs < 0).any():raise ValueError('invalid probabilities')
    indices = np.flatnonzero(legal);p = probs[indices]
    if not p.sum() > 0:raise ValueError('no legal probability mass')
    c = np.cumsum(p / p.sum());return int(indices[min(np.searchsorted(c, u, side='right'), len(indices) - 1)])


def draw(root_seed, k, ply, seat):
    return float(np.random.default_rng([root_seed, k, ply, seat]).random())


def outcome(env, ours):
    t = env.terminal()
    if t is None:return None, 'decision_cap'
    if getattr(t, 'capped', False):return None, 'simulator_cap'
    return (0. if t.winner is None else (1. if t.winner == ours else -1.)), 'terminal'


def make_env(base, result, peer_result=None):
    from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv, env_config_from_checkpoint_provenance
    from pokezero.neural_policy import category_vocab_from_model_config, feature_masks_from_model_config, observation_spec_from_model_config
    configs = [result.model_config] + ([] if peer_result is None else [peer_result.model_config])
    config = env_config_from_checkpoint_provenance(
        LocalShowdownConfig(showdown_root=base['showdown_root'],node_binary=base['node_binary'],set_belief_source=bool(base['belief_set_source'])),
        [feature_masks_from_model_config(c) for c in configs], context='M0 continuation pair',
        required_specs=[observation_spec_from_model_config(c) for c in configs],
        required_vocabs=[category_vocab_from_model_config(c,base['showdown_root']) for c in configs])
    return LocalShowdownEnv(config)


def legal_of(observation):return np.asarray(observation.legal_action_mask, bool)


def self_root(env, champion, seed, ours, target):
    """Champion self-play from `seed` (both seats sampling) to our seat's `target`-th decision; True at a root as above."""
    env.reset(seed=seed);count = 0;ply = 0
    while env.terminal() is None:
        requested = list(env.requested_players())
        if ours in requested and count == target:return legal_of(env.observe(ours)).sum() >= 2
        obs = [env.observe(p) for p in requested];probs = champion.probabilities(obs)
        actions = {p: sample(probs[i], legal_of(o), draw(seed, 10**6, ply, i)) for i, (p, o) in enumerate(zip(requested, obs))}
        count += ours in requested;env.step(actions);ply += 1
    return False


class Rollout:
    """One (action, k) rollout's state in an env: started from the root snapshot."""

    def __init__(self, env, snapshot, a, k, root_seed, ours, opp, opp_reply):
        self.env, self.a, self.k, self.root_seed, self.ours, self.opp = env, a, k, root_seed, ours, opp
        self.decisions = 0;self.ply = 0;self.done = False;self.value = None
        env.restore(snapshot);env.reseed_simulator_rng(int(np.random.default_rng([root_seed, k]).integers(1, 2**31 - 1)))
        first = {ours: a}
        if opp_reply is not None:first[opp] = opp_reply
        env.step(first);self.decisions = len(first);self.ply = 1;self.check()

    def check(self):
        if self.env.terminal() is not None or self.decisions >= MAX_DECISIONS:
            self.done = True;self.value, self.status = outcome(self.env, self.ours)

    def pending(self):return [] if self.done else list(self.env.requested_players())

    def step(self, actions):self.env.step(actions);self.decisions += len(actions);self.ply += 1;self.check()


def run_root(envs, models, root, snapshot, K, batched):
    """Every legal action x K rollouts of one root on `envs` (one env unbatched; several in lockstep batched). Returns outcomes
    [len(actions)][K] and per-rollout decisions."""
    ours, opp, actions, root_seed = root['ours'], root['opp'], root['actions'], root['root_seed']
    tasks = [(i, a, k) for k in range(K) for i, a in enumerate(actions)]   # k-major: each k's siblings run close together
    outcomes = np.full((len(actions), K), np.nan);decisions = np.zeros((len(actions), K), np.int32)
    statuses = [['pending'] * K for _ in actions]
    model_of = {ours: models['ours'], opp: models['opp']}
    replies = root['opp_replies']
    active = []
    while tasks or active:
        while tasks and len(active) < (len(envs) if batched else 1):
            i, a, k = tasks.pop(0);env = next(e for e in envs if all(r.env is not e for _, r in active))
            active.append((i, Rollout(env, snapshot, a, k, root_seed, ours, opp, None if replies is None else replies[k])))
        for _, r in active:
            if not r.done and not r.pending():raise RuntimeError('no requested players before terminal')
        requests = [(j, p, model_of[p].observe(r.env, p, root['seed'])) for j, (i, r) in enumerate(active) for p in r.pending()]
        chosen = {}
        for side in (ours, opp):
            group = [(j, p, o) for j, p, o in requests if p == side]
            if not group:continue
            probs = model_of[side].probabilities([o for _, _, o in group])
            for (j, p, o), pr in zip(group, probs):
                r = active[j][1];chosen.setdefault(j, {})[p] = sample(pr, legal_of(o), draw(root_seed, r.k, r.ply, 0 if p == ours else 1))
        for j, acts in chosen.items():active[j][1].step(acts)
        still = []
        for i, r in active:
            if r.done:
                outcomes[i, r.k] = np.nan if r.value is None else r.value
                statuses[i][r.k] = r.status;decisions[i, r.k] = r.decisions
            else:still.append((i, r))
        active = still
    return outcomes, decisions, statuses


def main(argv=None):
    from gpu_engine.mc_m0 import main as run
    return run(argv)


if __name__ == '__main__':
    raise SystemExit(main())
