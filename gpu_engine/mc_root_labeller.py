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
  roots (M0: pi_FP on the env view, without the peer-view redaction).
Outcome from our seat: +1 win, -1 loss, 0 draw, CAPPED for a capped game or a rollout past MAX_DECISIONS.
Modes: 'unbatched' (one env, one forward per decision) and 'batched' (envs stepped in lockstep, one forward per policy per step
over every requested observation). Timing: wall seconds and the CPU seconds of this process and its Node children (/proc).
Output: one JSON line per root (identity, legal actions, outcomes[action][k], decisions, timings).
"""
import argparse
import glob
import json
import os
import random
import time
from pathlib import Path

import numpy as np

CAPPED = -0.25
MAX_DECISIONS = 600
ACTIONS = 9


def process_cpu_seconds(pid=None):
    """dict(self=, children=): utime + stime (seconds) of `pid` (default this process) and of all its descendants (the Node
    bridges), from /proc."""
    tick = os.sysconf('SC_CLK_TCK');root = pid or os.getpid();out = dict(self=0., children=0.);pending = [root];seen = set()
    while pending:
        p = pending.pop()
        if p in seen:continue
        seen.add(p)
        try:
            stat = Path(f'/proc/{p}/stat').read_text();fields = stat[stat.rindex(')') + 2:].split()
            out['self' if p == root else 'children'] += (int(fields[11]) + int(fields[12])) / tick
            for task in Path(f'/proc/{p}/task').iterdir():
                children = (task / 'children').read_text().split();pending.extend(int(c) for c in children)
        except (OSError, ValueError, IndexError):continue
    return out


class Model:
    """A champion-format checkpoint for batched sampling: probabilities [n, 9] over the legal mask at temperature 1."""

    def __init__(self, path, torch):
        from pokezero.neural_policy import load_transformer_checkpoint
        self.torch = torch;self.model, self.result = load_transformer_checkpoint(path, map_location='cpu');self.model.eval()
        self.window = int(self.result.model_config.window_size)

    def probabilities(self, observations):
        from pokezero.neural_policy import observation_windows_to_torch
        t = observation_windows_to_torch([[o] for o in observations], window_size=self.window, device='cpu')
        with self.torch.no_grad():
            out = self.model(categorical_ids=t['categorical_ids'], numeric_features=t['numeric_features'], token_type_ids=t['token_type_ids'],
                             attention_mask=t['attention_mask'], history_mask=t['history_mask'])
            legal = self.torch.tensor(np.array([np.asarray(o.legal_action_mask, bool) for o in observations]))
            logits = out.policy_logits.masked_fill(~legal, -1e9)
            return self.torch.softmax(logits, dim=-1).numpy().astype(np.float64)


def sample(probs, legal, u):
    """The inverse-CDF draw of `u` in [0, 1) over the legal actions' probabilities."""
    p = np.where(legal, probs, 0.);c = np.cumsum(p / p.sum());return int(min(np.searchsorted(c, u, side='right'), ACTIONS - 1))


def draw(root_seed, k, ply, seat):
    return float(np.random.default_rng([root_seed, k, ply, seat]).random())


def outcome(env, ours):
    t = env.terminal()
    if t is None:return CAPPED
    if getattr(t, 'capped', False):return CAPPED
    return 0. if t.winner is None else (1. if t.winner == ours else -1.)


def make_env(base, result):
    from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
    from pokezero.neural_policy import category_vocab_from_model_config, feature_masks_from_model_config, observation_spec_from_model_config
    return LocalShowdownEnv(LocalShowdownConfig(showdown_root=base['showdown_root'], node_binary=base['node_binary'],
        observation_spec=observation_spec_from_model_config(result.model_config),
        category_vocab=category_vocab_from_model_config(result.model_config, base['showdown_root']),
        feature_masks=feature_masks_from_model_config(result.model_config), set_belief_source=bool(base['belief_set_source'])))


def legal_of(observation):return np.asarray(observation.legal_action_mask, bool)


def foulplay_root(env, game, target):
    """Replay `game` to our seat's `target`-th decision (0-based) with both seats' recorded actions; True at the root (our seat
    requested with at least two legal actions), False when the game ends first or the root has fewer."""
    env.reset(seed=game['seed']);ours = 0;round_ = 0
    while env.terminal() is None:
        requested = env.requested_players();actions = {}
        if game['ours'] in requested:
            if ours == target:return legal_of(env.observe(game['ours'])).sum() >= 2
            actions[game['ours']] = int(game['our_actions'][ours]);ours += 1
        if game['fp'] in requested:actions[game['fp']] = int(game['journal'][round_]['action'])
        env.step(actions);round_ += 1
    return False


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
        if self.env.terminal() is not None or self.decisions >= MAX_DECISIONS:self.done = True;self.value = outcome(self.env, self.ours)

    def pending(self):return [] if self.done else list(self.env.requested_players())

    def step(self, actions):self.env.step(actions);self.decisions += len(actions);self.ply += 1;self.check()


def run_root(envs, models, root, snapshot, K, batched):
    """Every legal action x K rollouts of one root on `envs` (one env unbatched; several in lockstep batched). Returns outcomes
    [len(actions)][K] and per-rollout decisions."""
    ours, opp, actions, root_seed = root['ours'], root['opp'], root['actions'], root['root_seed']
    tasks = [(i, a, k) for k in range(K) for i, a in enumerate(actions)]   # k-major: each k's siblings run close together
    outcomes = np.full((len(actions), K), np.nan);decisions = np.zeros((len(actions), K), np.int32)
    model_of = {ours: models['ours'], opp: models['opp']}
    replies = root['opp_replies']
    active = []
    while tasks or active:
        while tasks and len(active) < (len(envs) if batched else 1):
            i, a, k = tasks.pop(0);env = next(e for e in envs if all(r.env is not e for _, r in active))
            active.append((i, Rollout(env, snapshot, a, k, root_seed, ours, opp, None if replies is None else replies[k])))
        requests = [(j, p, r.env.observe(p)) for j, (i, r) in enumerate(active) for p in r.pending()]
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
            if r.done:outcomes[i, r.k] = r.value;decisions[i, r.k] = r.decisions
            else:still.append((i, r))
        active = still
    return outcomes, decisions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--collection', required=True, help='a Foul Play collection directory (results/, telemetry/, requests/)')
    parser.add_argument('--checkpoint', required=True, help="the champion's .pt");parser.add_argument('--pi-fp', required=True, help="pi_FP's .pt")
    parser.add_argument('--out', required=True, help='JSONL of roots');parser.add_argument('--roots', type=int, default=10)
    parser.add_argument('--kind', choices=('foulplay', 'self'), required=True);parser.add_argument('--K', type=int, default=64)
    parser.add_argument('--mode', choices=('batched', 'unbatched'), default='batched');parser.add_argument('--envs', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0);parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--worker', type=int, default=0);parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(args.threads)
    from gpu_engine import fp_reply_data as fpd
    champion = Model(args.checkpoint, torch);pi_fp = Model(args.pi_fp, torch)
    collection = Path(args.collection);rng = random.Random(f'{args.seed}-{args.kind}-{args.worker}')
    paths = sorted(glob.glob(str(collection / 'results' / '*.json')), key=lambda p: int(Path(p).stem))
    base = json.loads((collection / 'requests' / f'{Path(paths[0]).stem}.json').read_bytes())['base_config']
    envs = [make_env(base, champion.result) for _ in range(args.envs if args.mode == 'batched' else 1)]
    try:return label(args, envs, champion, pi_fp, collection, rng, paths, fpd)
    finally:
        for e in envs:e.close()


def label(args, envs, champion, pi_fp, collection, rng, paths, fpd):
    out = open(args.out, 'a');made = 0;attempts = 0;skipped = {}
    while made < args.roots and attempts < 50 * args.roots:
        attempts += 1;started = time.perf_counter()
        if args.kind == 'foulplay':
            path = rng.choice(paths);game, why = fpd.load_game(collection, path)
            if game is None or len(game['our_actions']) < 4:continue
            target = rng.randrange(1, len(game['our_actions']) - 1)
            try:ok = foulplay_root(envs[0], game, target)
            except Exception as error:   # a game the replay cannot rebuild (counted as an attempt and by cause)
                why = type(error).__name__;skipped[why] = skipped.get(why, 0) + 1;continue
            ours, identity = game['ours'], dict(ordinal=game['ordinal'], seed=game['seed'], target=target)
        else:
            seed = rng.randrange(1, 2**31 - 1);ours = rng.choice(('p1', 'p2'));target = rng.randrange(1, 40)
            ok = self_root(envs[0], champion, seed, ours, target);identity = dict(seed=seed, target=target)
        if not ok:continue
        env = envs[0];opp = {'p1': 'p2', 'p2': 'p1'}[ours];snapshot = env.snapshot()
        legal = legal_of(env.observe(ours));actions = [int(a) for a in np.flatnonzero(legal)]
        root_seed = rng.randrange(1, 2**31 - 1);opp_model = pi_fp if args.kind == 'foulplay' else champion
        replies = None
        if opp in env.requested_players():
            o = env.observe(opp);pr = opp_model.probabilities([o])[0]
            replies = [sample(pr, legal_of(o), draw(root_seed, k, 0, 1)) for k in range(args.K)]
        root = dict(ours=ours, opp=opp, actions=actions, root_seed=root_seed, opp_replies=replies)
        for e in envs[1:]:e.reset(seed=identity['seed'])   # a battle shell for the snapshot (restore needs a reset env)
        prepared = time.perf_counter();cpu0 = process_cpu_seconds()
        outcomes, decisions = run_root(envs, dict(ours=champion, opp=opp_model), root, snapshot, args.K, args.mode == 'batched')
        wall = time.perf_counter() - prepared;cpu1 = process_cpu_seconds();cpu = {k: cpu1[k] - cpu0[k] for k in cpu1}
        row = dict(schema='mc-root-labels.v1', kind=args.kind, mode=args.mode, envs=len(envs), K=args.K, threads=args.threads, worker=args.worker,
                   ours=ours, actions=actions, opp_requested=replies is not None, root_seed=root_seed, **identity,
                   outcomes=outcomes.tolist(), decisions=decisions.tolist(), rollouts=int(outcomes.size),
                   wall_seconds=round(wall, 3), cpu_seconds=round(cpu['self'] + cpu['children'], 3), cpu_python_seconds=round(cpu['self'], 3),
                   cpu_node_seconds=round(cpu['children'], 3), prepare_seconds=round(prepared - started, 3))
        out.write(json.dumps(row) + '\n');out.flush();made += 1
        print(json.dumps({k: row[k] for k in ('kind', 'mode', 'actions', 'rollouts', 'wall_seconds', 'cpu_python_seconds', 'cpu_node_seconds')}), flush=True)
    print(json.dumps(dict(event='DONE', roots=made, attempts=attempts, skipped=skipped)), flush=True)
    return 0 if made else 1


if __name__ == '__main__':
    raise SystemExit(main())
