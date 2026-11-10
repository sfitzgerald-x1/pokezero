"""Action-conditional Monte Carlo labels on real Showdown, for the search value function (design: search-value-function-design,
step M0 and the labelling step).

A root is a real battle state at a decision of our seat:
- 'foulplay': a Foul Play collection game (gpu_engine.fp_reply_data.load_game) replayed in LocalShowdownEnv to one of our seat's
  decisions, both seats playing their recorded actions;
- 'self': champion self-play (both seats sampling at temperature 1) from a fresh seed, played to one of our seat's decisions.
Per legal action a of our seat, K rollouts:
- restore the root snapshot and reseed the simulator with s_k: rollout k uses the same seed for every sibling action
  (common random numbers);
- the opponent's root reply b_k (when it is requested) is drawn once per k from the opponent's root model and shared by the
  siblings: pi_FP for 'foulplay' roots, on its trained inputs (the builder's view b of the opponent's seat with the peer-view-v2
  redaction), the champion for 'self' roots;
- then the champion plays BOTH seats at temperature 1 to the end (the lead's definition: on a 'foulplay' root the label is "our
  move against Foul Play's modelled first reply, then champion play"), every draw from a generator seeded by (root, k, ply,
  seat), matched across siblings. (M0's rows, before this, continued the opponent with pi_FP on the env view.)
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
        if self.window != 1:raise ValueError(f'{path}: window {self.window}; the labeller feeds single observations (window 1 only)')

    def evaluate(self, observations, redact=None):
        """(probabilities [n, 9] over each legal mask, values [n]: the raw tanh value head). `redact`: (name, opp_hazard_credit
        [n]) applies fp_reply_train.redact_peer_view to the numeric features first (pi_FP's trained inputs)."""
        from pokezero.neural_policy import observation_windows_to_torch
        t = observation_windows_to_torch([[o] for o in observations], window_size=self.window, device='cpu')
        if redact is not None:
            from gpu_engine.fp_reply_train import redact_peer_view
            numeric = t['numeric_features'].numpy().copy()   # [n, window, tokens, features]
            redact_peer_view(numeric, redact[0], np.asarray(redact[1], numeric.dtype).reshape(numeric.shape[:-2]))
            t = dict(t, numeric_features=self.torch.from_numpy(numeric))
        with self.torch.no_grad():
            out = self.model(categorical_ids=t['categorical_ids'], numeric_features=t['numeric_features'], token_type_ids=t['token_type_ids'],
                             attention_mask=t['attention_mask'], history_mask=t['history_mask'])
            legal = self.torch.tensor(np.array([np.asarray(o.legal_action_mask, bool) for o in observations]))
            logits = out.policy_logits.masked_fill(~legal, -1e9)
            return self.torch.softmax(logits, dim=-1).numpy().astype(np.float64), out.value.reshape(-1).numpy().astype(np.float64)

    def probabilities(self, observations):return self.evaluate(observations)[0]


def sample(probs, legal, u):
    """The inverse-CDF draw of `u` in [0, 1) over the legal actions' probabilities (always a legal action)."""
    idx = np.flatnonzero(legal);p = np.asarray(probs, float)[idx];c = np.cumsum(p / p.sum())
    return int(idx[min(int(np.searchsorted(c, u, side='right')), len(idx) - 1)])


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

        def _reset(self, *args, **kwargs):   # every reset path (reset, reset_with_start_override) goes through _reset
            self._bump();out = super()._reset(*args, **kwargs);self._bump();return out

        def _restore_local_snapshot_state(self, *args, **kwargs):   # every restore path restores the local state here
            self._bump();out = super()._restore_local_snapshot_state(*args, **kwargs);self._bump();return out

        def reseed_simulator_rng(self, *args, **kwargs):
            self._bump();out = super().reseed_simulator_rng(*args, **kwargs);self._bump();return out

        def _submit_step_choices(self, *args, **kwargs):
            self._bump()   # the battle is about to move: what the step observes afterwards belongs to the new position
            return super()._submit_step_choices(*args, **kwargs)

        def _refuse(self, *args, **kwargs):
            raise NotImplementedError('CachingEnv (the Monte Carlo labeller) supports reset, observe, step, snapshot, restore and reseed only')

        materialize_public_world = materialize_scenario_state = snapshot_actionable_boundary = _refuse
        snapshot_for_search = restore_search_snapshot = step_from_search_snapshot = reset_with_start_override = _refuse

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


def depth1_baseline(env, snapshot, root, models, draws, post_turn=False):
    """The current value head's depth-1 completed-Q at the root, on the true state (the real-Showdown U1 baseline). For every
    legal action a of our seat, every opponent reply b with pi_opp(b) > 0.001 (no reply when the opponent waits), and `draws`
    chance draws: restore, reseed with the draw's seed, step (a, b), then sample both seats' policies until our seat is requested
    again, and take the champion's value from our seat there (a terminal state: its outcome). The draws' seeds are disjoint from
    the rollouts' (an independent evaluation bank). Returns dict(completed_q, replies, child_values).
    post_turn (the ladder's R0′, the engine rungs' estimator): each child is valued by the champion's value head on our view of
    the state right after the joint turn (env.observe(ours)), requested or not, never sampled further; a terminal state keeps its
    outcome. Everything else (actions, replies, draws, their seeds, the weights) is the same."""
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
                if post_turn:v = outcome(env, ours) if env.terminal() is not None else float(models['ours'].evaluate([env.observe(ours)])[1][0])
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


def foulplay_root(env, game, target, fpd=None):
    """Replay `game` to our seat's `target`-th decision (0-based) with both seats' recorded actions, under fp_reply_data.replay's
    checks: our request equals the captured one, Foul Play has a journal entry exactly when requested, and its journaled action is
    legal and decodes from its choice string. True at the root (our seat requested with at least two legal actions), False when
    the game ends first or the root has fewer; fp_reply_data.Mismatch when the replay leaves the recording."""
    if fpd is None:from gpu_engine import fp_reply_data as fpd
    env.reset(seed=game['seed']);ours = 0;round_ = 0;journal = dict(game['journal'])
    while env.terminal() is None:
        requested = env.requested_players();actions = {}
        if game['ours'] in requested:
            captured = game['our_requests'][ours] if ours < len(game['our_requests']) else None
            if captured is None or fpd._request_body(captured) != fpd._request_body(json.dumps(env._latest_requests.get(game['ours']))):
                raise fpd.Mismatch(f'our request differs from the capture at decision {ours}')
            if ours == target:return legal_of(env.observe(game['ours'])).sum() >= 2
            actions[game['ours']] = int(game['our_actions'][ours]);ours += 1
        if game['fp'] in requested:
            entry = journal.pop(round_, None)
            if entry is None:raise fpd.Mismatch(f'Foul Play requested in round {round_} without a journal entry')
            action = int(entry['action'])
            if not 0 <= action < ACTIONS or not legal_of(env.observe(game['fp']))[action]:raise fpd.Mismatch(f'Foul Play action {action} illegal, round {round_}')
            if fpd.choice_action(env, game['fp'], entry.get('choice')) != action:raise fpd.Mismatch(f'Foul Play choice does not decode to {action}, round {round_}')
            actions[game['fp']] = action
        elif round_ in journal:raise fpd.Mismatch(f'a journal entry in round {round_} where Foul Play was never requested')
        env.step(actions);round_ += 1
    return False


def pi_fp_root_policy(env, opp, ours, seed, views, pi_fp):
    """pi_FP's reply probabilities at a 'foulplay' root on its trained inputs: the builder's view b of the opponent's (Foul Play's)
    seat (fp_reply_data.builder_views) with the peer-view-v2 redaction and the hazard credit Foul Play sees
    (fp_view_opp_hazard_credit), as π_FP-red3-full was trained."""
    from gpu_engine import fp_reply_data as fpd
    view = fpd.builder_views(env, opp, env._latest_requests.get(opp), seed, views)['b']
    if not np.array_equal(legal_of(view), legal_of(env.observe(opp))):   # as fp_reply_data's BuilderMismatch: the views' masks agree
        raise ValueError('the builder view b legal mask differs from env.observe')
    credit = fpd.fp_view_opp_hazard_credit(env.protocol_lines, ours)
    return pi_fp.evaluate([view], redact=('peer-view-v2', [credit]))[0][0]


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
    views = None
    if args.kind == 'foulplay':   # the builder-view inputs of pi_FP's root reply (fp_reply_data --views' ingredients)
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.neural_policy import category_vocab_from_model_config, feature_masks_from_model_config, observation_spec_from_model_config
        from pokezero.randbat import load_gen3_randbat_source_cached
        views = dict(source=load_gen3_randbat_source_cached(base['showdown_root']), dex=load_showdown_dex_cached(base['showdown_root']),
                     spec=observation_spec_from_model_config(pi_fp.result.model_config), masks=feature_masks_from_model_config(pi_fp.result.model_config),
                     vocab=category_vocab_from_model_config(pi_fp.result.model_config, base['showdown_root']))
    envs = [make_env(base, champion.result, caching=not args.plain_env) for _ in range(args.envs if args.mode == 'batched' else 1)]
    try:return label(args, envs, champion, pi_fp, collection, rng, paths, fpd, views)
    finally:
        for e in envs:e.close()


def candidates(args, rng, paths, fpd, collection, counts):
    """Root candidates without replacement: (kind, game or seed, our seat, target). 'foulplay': passes over the games in a shuffled
    order, a not-yet-used target per game each pass (a game is dropped once every target is used or it cannot be loaded); 'self':
    battle seeds from the reserved range, each at most once."""
    if args.kind == 'self':   # worker w draws from its own slice of the range, so a seed is used at most once over all workers
        used = set();span = args.seed_span // args.workers;low = (args.seed_base or 0) + args.worker * span
        while len(used) < span if args.seed_base is not None else True:
            seed = (low + rng.randrange(span)) if args.seed_base is not None else rng.randrange(1, 2**31 - 1)
            if seed in used:continue
            used.add(seed);yield None, seed, rng.choice(('p1', 'p2')), rng.randrange(1, 40)
        return
    games, used = {}, {}
    pool = list(paths)
    while pool:
        rng.shuffle(pool);keep = []
        for path in pool:
            if path not in games:
                try:game, why = fpd.load_game(collection, path)
                except Exception as error:game, why = None, f'load_error:{type(error).__name__}'
                if game is None:counts[f'game_{why}'] += 1;continue
                if len(game['our_actions']) < 4:counts['game_short'] += 1;continue
                games[path] = game
            game = games[path];free = [t for t in range(1, len(game['our_actions']) - 1) if t not in used.setdefault(path, set())]
            if not free:continue
            target = rng.choice(free);used[path].add(target);keep.append(path)
            yield game, game['seed'], game['ours'], target
        pool = keep


def label(args, envs, champion, pi_fp, collection, rng, paths, fpd, views=None):
    """Label up to args.roots roots; every candidate's fate is counted (roots, not_a_root, replay mismatches, errors by type) and
    written beside the output as <out>.accounting-<kind>-<worker>.json."""
    import collections
    counts = collections.Counter();made = 0
    if Path(args.out).exists() and Path(args.out).stat().st_size:   # a restarted worker would append the same roots again
        raise SystemExit(f'{args.out} already holds roots: write a new output (the candidate stream is deterministic)')
    with open(args.out, 'a') as out:
        for game, seed, ours, target in candidates(args, rng, paths, fpd, collection, counts):
            if made >= args.roots or counts['candidates'] >= 50 * args.roots:break   # a cap on candidates: never an endless search
            counts['candidates'] += 1;started = time.perf_counter()
            try:
                if args.kind == 'foulplay':
                    ok = foulplay_root(envs[0], game, target, fpd);identity = dict(ordinal=game['ordinal'], seed=seed, target=target)
                else:
                    ok = self_root(envs[0], champion, seed, ours, target);identity = dict(seed=seed, target=target)
            except fpd.Mismatch:counts['replay_mismatch'] += 1;continue
            except Exception as error:counts[f'root_error:{type(error).__name__}'] += 1;continue
            if not ok:counts['not_a_root'] += 1;continue
            try:row = label_root(args, envs, champion, pi_fp, views, ours, identity, rng, started)
            except Exception as error:counts[f'label_error:{type(error).__name__}'] += 1;continue
            out.write(json.dumps(row) + '\n');out.flush();made += 1;counts['roots'] += 1
            print(json.dumps({k: row[k] for k in ('kind', 'mode', 'actions', 'rollouts', 'wall_seconds', 'cpu_python_seconds', 'cpu_node_seconds')}), flush=True)
    counts['exhausted'] = int(made < args.roots)
    Path(f'{args.out}.accounting-{args.kind}-{args.worker}.json').write_text(json.dumps(dict(kind=args.kind, worker=args.worker, workers=args.workers, **counts)) + '\n')
    print(json.dumps(dict(event='DONE', **counts)), flush=True)
    return 0 if made else 1


def label_root(args, envs, champion, pi_fp, views, ours, identity, rng, started):
    """One root's row: our policy and value, the opponent's root reply bank (pi_FP on its trained inputs for 'foulplay' roots, the
    champion for 'self'), the optional depth-1 baseline, then every legal action x K rollouts with the champion playing both
    seats after the root."""
    env = envs[0];opp = {'p1': 'p2', 'p2': 'p1'}[ours];snapshot = env.snapshot()
    legal = legal_of(env.observe(ours));actions = [int(a) for a in np.flatnonzero(legal)]
    # the root's rollout seed from its identity, not from the candidate stream: the same root always gets the same bank
    root_seed = int(np.random.default_rng([args.seed, 1 if args.kind == 'self' else 2, identity.get('ordinal', 0), identity['seed'],
                                           identity['target']]).integers(1, 2**31 - 1));replies = None;opp_policy = None
    if opp in env.requested_players():
        o = env.observe(opp)
        opp_policy = pi_fp_root_policy(env, opp, ours, identity['seed'], views, pi_fp) if args.kind == 'foulplay' else champion.probabilities([o])[0]
        replies = [sample(opp_policy, legal_of(o), draw(root_seed, k, 0, 1)) for k in range(args.K)]
    our_probs, our_value = champion.evaluate([env.observe(ours)])
    root = dict(ours=ours, opp=opp, actions=actions, root_seed=root_seed, opp_replies=replies, opp_policy=opp_policy)
    continuation = dict(ours=champion, opp=champion)   # the lead's definition: the champion plays both seats after the root
    baseline = None
    if args.baseline_draws:
        started_b = time.perf_counter();baseline = depth1_baseline(env, snapshot, root, continuation, args.baseline_draws)
        baseline['seconds'] = round(time.perf_counter() - started_b, 3)
    for e in envs[1:]:e.reset(seed=identity['seed'])   # a battle shell for the snapshot (restore needs a reset env)
    prepared = time.perf_counter();cpu0 = process_cpu_seconds()
    outcomes, decisions = run_root(envs, continuation, root, snapshot, args.K, args.mode == 'batched')
    wall = time.perf_counter() - prepared;cpu1 = process_cpu_seconds();cpu = {k: cpu1[k] - cpu0[k] for k in cpu1}
    return dict(schema='mc-root-labels.v2', kind=args.kind, mode=args.mode, envs=len(envs), K=args.K, threads=args.threads, worker=args.worker,
                ours=ours, actions=actions, opp_requested=replies is not None, root_seed=root_seed, **identity,
                opp_root_model='pi_fp_view_b_peer_view_v2' if args.kind == 'foulplay' else 'champion', continuation='champion_both_seats',
                policy=[round(float(x), 6) for x in our_probs[0]], root_value=round(float(our_value[0]), 6),
                opp_policy=None if opp_policy is None else [round(float(x), 6) for x in opp_policy], baseline=baseline,
                outcomes=outcomes.tolist(), decisions=decisions.tolist(), rollouts=int(outcomes.size),
                wall_seconds=round(wall, 3), cpu_seconds=round(cpu['self'] + cpu['children'], 3), cpu_python_seconds=round(cpu['self'], 3),
                cpu_node_seconds=round(cpu['children'], 3), prepare_seconds=round(prepared - started, 3))


if __name__ == '__main__':
    raise SystemExit(main())
