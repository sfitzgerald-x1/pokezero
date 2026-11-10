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

    def evaluate(self, observations):
        """(probabilities [n, 9] over each legal mask, values [n]: the raw tanh value head)."""
        from pokezero.neural_policy import observation_windows_to_torch
        t = observation_windows_to_torch([[o] for o in observations], window_size=self.window, device='cpu')
        with self.torch.no_grad():
            out = self.model(categorical_ids=t['categorical_ids'], numeric_features=t['numeric_features'], token_type_ids=t['token_type_ids'],
                             attention_mask=t['attention_mask'], history_mask=t['history_mask'])
            legal = self.torch.tensor(np.array([np.asarray(o.legal_action_mask, bool) for o in observations]))
            logits = out.policy_logits.masked_fill(~legal, -1e9)
            return self.torch.softmax(logits, dim=-1).numpy().astype(np.float64), out.value.reshape(-1).numpy().astype(np.float64)

    def probabilities(self, observations):return self.evaluate(observations)[0]


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


def caching_env_class():
    """LocalShowdownEnv with each player's normalized state cached per battle position: env.step's own observations normalize
    every next-requested player, and the next step's _choices_for_actions normalized them again (about 7 ms a decision). The
    cache key is a generation bumped on every state change (step, reset, restore, reseed); a root-PUCT branch observation is
    never cached. The labeller's only env; mc_root_labeller.parity checks it against the plain env."""
    from pokezero.local_showdown import LocalShowdownEnv

    class CachingEnv(LocalShowdownEnv):
        def _bump(self):self._generation = getattr(self, '_generation', 0) + 1;self._state_cache = {}

        def reset(self, *args, **kwargs):
            self._bump();out = super().reset(*args, **kwargs);self._bump();return out

        def restore(self, *args, **kwargs):
            self._bump();out = super().restore(*args, **kwargs);self._bump();return out

        def reseed_simulator_rng(self, *args, **kwargs):
            self._bump();out = super().reseed_simulator_rng(*args, **kwargs);self._bump();return out

        def _submit_step_choices(self, *args, **kwargs):
            self._bump()   # the battle is about to move: what the step observes afterwards belongs to the new position
            return super()._submit_step_choices(*args, **kwargs)

        def _state_for_player(self, player, *, root_puct_branch_observation=False):
            if root_puct_branch_observation:return super()._state_for_player(player, root_puct_branch_observation=True)
            cache = getattr(self, '_state_cache', None)
            if cache is None:self._bump();cache = self._state_cache
            key = (player, self._generation)
            if key not in cache:cache[key] = super()._state_for_player(player)
            return cache[key]

    return CachingEnv


def make_env(base, result, caching=True):
    from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
    from pokezero.neural_policy import category_vocab_from_model_config, feature_masks_from_model_config, observation_spec_from_model_config
    cls = caching_env_class() if caching else LocalShowdownEnv
    return cls(LocalShowdownConfig(showdown_root=base['showdown_root'], node_binary=base['node_binary'],
        observation_spec=observation_spec_from_model_config(result.model_config),
        category_vocab=category_vocab_from_model_config(result.model_config, base['showdown_root']),
        feature_masks=feature_masks_from_model_config(result.model_config), set_belief_source=bool(base['belief_set_source'])))


def legal_of(observation):return np.asarray(observation.legal_action_mask, bool)


def completed_q(child_values, opp_policy, actions, replies):
    """Depth-1 completed-Q: q[a] = sum_b pi_opp(b) * mean over draws of child_values[(a, b)], the replies' weights renormalized
    over `replies` (replies [None]: the opponent waits, weight 1). child_values: {(a, b): [values]}; returns q per action."""
    out = []
    for a in actions:
        if replies == [None]:out.append(float(np.mean(child_values[(a, None)])));continue
        w = np.array([opp_policy[b] for b in replies], float);w = w / w.sum()
        out.append(float(sum(wi * np.mean(child_values[(a, b)]) for wi, b in zip(w, replies))))
    return out


def depth1_baseline(env, snapshot, root, models, draws):
    """The current value head's depth-1 completed-Q at the root, on the true state (the real-Showdown U1 baseline). For every
    legal action a of our seat, every opponent reply b with pi_opp(b) > 0.001 (no reply when the opponent waits), and `draws`
    chance draws: restore, reseed with the draw's seed, step (a, b), then sample both seats' policies until our seat is requested
    again, and take the champion's value from our seat there (a terminal state: its outcome). The draws' seeds are disjoint from
    the rollouts' (an independent evaluation bank). Returns dict(completed_q, replies, child_values)."""
    ours, opp, actions, root_seed = root['ours'], root['opp'], root['actions'], root['root_seed']
    pi = root['opp_policy'];replies = [None] if pi is None else [b for b in range(ACTIONS) if pi[b] > .001]
    model_of = {ours: models['ours'], opp: models['opp']};values = {}
    for a in actions:
        for b in replies:
            for c in range(draws):
                env.restore(snapshot);env.reseed_simulator_rng(int(np.random.default_rng([root_seed, 7_000_000 + c]).integers(1, 2**31 - 1)))
                first = {ours: a}
                if b is not None:first[opp] = b
                obs = dict(env.step(first).observations);ply = 1;v = None
                while v is None:
                    if env.terminal() is not None:v = outcome(env, ours);break
                    requested = list(env.requested_players())
                    if ours in requested:v = float(models['ours'].evaluate([obs[ours]])[1][0]);break
                    acts = {}
                    for pl in requested:
                        pr = model_of[pl].probabilities([obs[pl]])[0];acts[pl] = sample(pr, legal_of(obs[pl]), draw(root_seed, 7_000_000 + c, ply, 1))
                    obs = dict(env.step(acts).observations);ply += 1
                values.setdefault((a, b), []).append(v)
    return dict(completed_q=completed_q(values, pi, actions, replies), replies=replies,
                child_values={f'{a},{b}': vs for (a, b), vs in values.items()})


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
        self.obs = dict(env.step(first).observations);self.decisions = len(first);self.ply = 1;self.check()

    def check(self):
        if self.env.terminal() is not None or self.decisions >= MAX_DECISIONS:self.done = True;self.value = outcome(self.env, self.ours)

    def pending(self):
        """The requested players with their observations: the step's own (env.step observes every next-requested player), so
        no second observe."""
        return [] if self.done else [(p, self.obs[p]) for p in self.env.requested_players()]

    def step(self, actions):
        self.obs = dict(self.env.step(actions).observations);self.decisions += len(actions);self.ply += 1;self.check()


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
        requests = [(j, p, o) for j, (i, r) in enumerate(active) for p, o in r.pending()]
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


def parity(collection, checkpoint, games=2, seed=0):
    """The caching env against the plain one: each of `games` Foul Play games replayed to a root, then the same seeded rollouts
    of every legal action in both envs; every observation's arrays, every choice and every outcome must be identical. Returns
    dict(decisions=, mismatches=)."""
    import torch
    from gpu_engine import fp_reply_data as fpd
    champion = Model(checkpoint, torch);collection = Path(collection);rng = random.Random(seed)
    paths = sorted(glob.glob(str(collection / 'results' / '*.json')), key=lambda p: int(Path(p).stem))
    base = json.loads((collection / 'requests' / f'{Path(paths[0]).stem}.json').read_bytes())['base_config']
    envs = [make_env(base, champion.result, caching=c) for c in (False, True)];decisions = mismatches = 0
    try:
        made = 0
        while made < games:
            game, _ = fpd.load_game(collection, rng.choice(paths))
            if game is None or len(game['our_actions']) < 4:continue
            target = rng.randrange(1, len(game['our_actions']) - 1)
            if not all(foulplay_root(e, game, target) for e in envs):continue
            made += 1;ours = game['ours'];opp = game['fp'];snaps = [e.snapshot() for e in envs]
            actions = [int(a) for a in np.flatnonzero(legal_of(envs[0].observe(ours)))]
            reply = None
            if opp in envs[0].requested_players():
                o = envs[0].observe(opp);reply = sample(champion.probabilities([o])[0], legal_of(o), .5)
            for a in actions:
                trace = []
                for e, snap in zip(envs, snaps):
                    r = Rollout(e, snap, a, 0, 12345, ours, opp, reply);steps = []
                    while not r.done:
                        pending = r.pending();acts = {}
                        for p, o in pending:
                            steps.append((p, tuple(np.asarray(o.numeric_features).ravel().round(9)), tuple(np.asarray(o.categorical_ids).ravel()),
                                          tuple(legal_of(o))))
                            pr = champion.probabilities([o])[0];acts[p] = sample(pr, legal_of(o), draw(12345, 0, r.ply, 0 if p == ours else 1))
                        steps.append(('acts', tuple(sorted(acts.items()))));r.step(acts)
                    trace.append((steps, r.value))
                decisions += sum(1 for x in trace[0][0] if x[0] != 'acts');mismatches += trace[0] != trace[1]
    finally:
        for e in envs:e.close()
    return dict(decisions=decisions, mismatches=mismatches)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--collection', required=True, help='a Foul Play collection directory (results/, telemetry/, requests/)')
    parser.add_argument('--checkpoint', required=True, help="the champion's .pt");parser.add_argument('--pi-fp', required=True, help="pi_FP's .pt")
    parser.add_argument('--out', required=True, help='JSONL of roots');parser.add_argument('--roots', type=int, default=10)
    parser.add_argument('--kind', choices=('foulplay', 'self'), required=True);parser.add_argument('--K', type=int, default=64)
    parser.add_argument('--mode', choices=('batched', 'unbatched'), default='batched');parser.add_argument('--envs', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0);parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--worker', type=int, default=0);parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--plain-env', action='store_true', help='the plain LocalShowdownEnv instead of CachingEnv (timing comparisons)')
    parser.add_argument('--ordinals', help="foulplay roots: a JSON list of the collection's game ordinals to draw from (e.g. held-out games); "
                        "worker w takes ordinals[w::workers]")
    parser.add_argument('--seed-base', type=int, default=None, help='self roots: battle seeds from [seed-base, seed-base + seed-span) (a reserved range)')
    parser.add_argument('--seed-span', type=int, default=10**7)
    parser.add_argument('--baseline-draws', type=int, default=0, help="depth-1 baseline: chance draws per (action, reply) (0: none)")
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(args.threads)
    from gpu_engine import fp_reply_data as fpd
    champion = Model(args.checkpoint, torch);pi_fp = Model(args.pi_fp, torch)
    collection = Path(args.collection);rng = random.Random(f'{args.seed}-{args.kind}-{args.worker}')
    paths = sorted(glob.glob(str(collection / 'results' / '*.json')), key=lambda p: int(Path(p).stem))
    base = json.loads((collection / 'requests' / f'{Path(paths[0]).stem}.json').read_bytes())['base_config']
    if args.ordinals:   # a fixed game list (held-out or training), partitioned across workers
        paths = [str(collection / 'results' / f'{o}.json') for o in json.loads(Path(args.ordinals).read_text())[args.worker::args.workers]]
    envs = [make_env(base, champion.result, caching=not args.plain_env) for _ in range(args.envs if args.mode == 'batched' else 1)]
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
            seed = (args.seed_base + rng.randrange(args.seed_span)) if args.seed_base is not None else rng.randrange(1, 2**31 - 1)
            ours = rng.choice(('p1', 'p2'));target = rng.randrange(1, 40)
            ok = self_root(envs[0], champion, seed, ours, target);identity = dict(seed=seed, target=target)
        if not ok:continue
        env = envs[0];opp = {'p1': 'p2', 'p2': 'p1'}[ours];snapshot = env.snapshot()
        legal = legal_of(env.observe(ours));actions = [int(a) for a in np.flatnonzero(legal)]
        root_seed = rng.randrange(1, 2**31 - 1);opp_model = pi_fp if args.kind == 'foulplay' else champion
        replies = None;opp_policy = None
        if opp in env.requested_players():
            o = env.observe(opp);pr = opp_model.probabilities([o])[0];opp_policy = pr
            replies = [sample(pr, legal_of(o), draw(root_seed, k, 0, 1)) for k in range(args.K)]
        our_probs, our_value = champion.evaluate([env.observe(ours)])
        root = dict(ours=ours, opp=opp, actions=actions, root_seed=root_seed, opp_replies=replies, opp_policy=opp_policy)
        baseline = None
        if args.baseline_draws:
            started_b = time.perf_counter()
            baseline = depth1_baseline(env, snapshot, root, dict(ours=champion, opp=opp_model), args.baseline_draws)
            baseline['seconds'] = round(time.perf_counter() - started_b, 3)
        for e in envs[1:]:e.reset(seed=identity['seed'])   # a battle shell for the snapshot (restore needs a reset env)
        prepared = time.perf_counter();cpu0 = process_cpu_seconds()
        outcomes, decisions = run_root(envs, dict(ours=champion, opp=opp_model), root, snapshot, args.K, args.mode == 'batched')
        wall = time.perf_counter() - prepared;cpu1 = process_cpu_seconds();cpu = {k: cpu1[k] - cpu0[k] for k in cpu1}
        row = dict(schema='mc-root-labels.v1', kind=args.kind, mode=args.mode, envs=len(envs), K=args.K, threads=args.threads, worker=args.worker,
                   ours=ours, actions=actions, opp_requested=replies is not None, root_seed=root_seed, **identity,
                   policy=[round(float(x), 6) for x in our_probs[0]], root_value=round(float(our_value[0]), 6),
                   opp_policy=None if opp_policy is None else [round(float(x), 6) for x in opp_policy], baseline=baseline,
                   outcomes=outcomes.tolist(), decisions=decisions.tolist(), rollouts=int(outcomes.size),
                   wall_seconds=round(wall, 3), cpu_seconds=round(cpu['self'] + cpu['children'], 3), cpu_python_seconds=round(cpu['self'], 3),
                   cpu_node_seconds=round(cpu['children'], 3), prepare_seconds=round(prepared - started, 3))
        out.write(json.dumps(row) + '\n');out.flush();made += 1
        print(json.dumps({k: row[k] for k in ('kind', 'mode', 'actions', 'rollouts', 'wall_seconds', 'cpu_python_seconds', 'cpu_node_seconds')}), flush=True)
    print(json.dumps(dict(event='DONE', roots=made, attempts=attempts, skipped=skipped)), flush=True)
    return 0 if made else 1


if __name__ == '__main__':
    raise SystemExit(main())
