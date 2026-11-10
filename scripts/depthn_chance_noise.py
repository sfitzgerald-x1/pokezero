#!/usr/bin/env python3
"""How noisy is the DepthN search's single engine chance draw per world? The winner's curse of one crit or miss. CPU only.

Input: the re-pricer's chance-<shard>.jsonl (scripts/depthn_rr_reprice.py --chance-seeds 0,1,...,8): per searched root the
W constructed worlds priced once per chance seed, the worlds and everything but the engine's chance keys fixed (seed 0 is
the live draw, the control). Each seed's grids G_w^s[a][b] and the per-world reply priors (opponent_requested, kept,
root_peer_logits: the same under every seed) give that seed's WorldPrices.

Definitions (every operator is the live code's own function on WorldPrices):
- averaging, single draw: search_hedge.averaging over seed s's W worlds: q_s(a) = sum_w 1/W sum_b pi_w(b) G_w^s[a][b], played
  A_s = argmax q_s (lowest index on ties).
- averaging over a seed set T: the same on the per-world mean grid G_w^T[a][b] = fsum_s G_w^s[a][b] / |T| (pi_w unchanged), so
  q_T = the mean of q_s; at |T| = 1 it is the single draw bit for bit.
- target_argmax over T: the live policy target recomputed from the mean grid exactly as the server reports it:
  d3_search_latency.compose over the K blocks' reduce_block(worlds 2k, 2k+1; root_logits, root_legal) (the equal-weight mean
  of the blocks' completed-Q, then softmax(root_logits + q) over the legal actions), and its argmax (lowest index on ties);
  when the composition refuses (reduce_completed_q's range edge) the RR scorer's fallback argmax(root_logits + q_T). At |T| = 1
  this is the live definition: the analysis checks it reproduces each seed's own policy_target (and averaging's action and q
  each seed's own action_index and q_values) exactly, `consistency`.
- greedy g: the live actor's greedy action (the re-pricer's greedy_action_index: the live telemetry's, else argmax of
  root_logits over root_legal).
- seeds: the analysis seeds (--seeds; default the run's non-zero chance seeds, 8 for 1..8), n of them, in order; a root missing
  one is excluded.
Reported on two populations: every analysable root (all analysis seeds COMPLETE, the same priced cells under every seed,
greedy legal), choices raw; and the roots the live guard would not trip (the replay's effect guard record present with (b)
false, the RR scorer's primary population), each choice through the RR scorer's guard_choices (greedy where the own-recharge
guard or (c) on that choice trips; the live record, seed 0's draw).
1. flip rate: per root the share of seed pairs (s, t) whose single-draw choices differ, averaged over roots; also the share
   of roots with any flip.
2. override rate against greedy: single draw (per root the share of seeds whose choice is not g, averaged) and on the mean
   grid of all n seeds.
3. convergence: for S = 1, 2, 4 (every S < n dividing n) the fixed design of the n seeds cut into n/S consecutive disjoint
   groups: agreement of each group's choice with the all-n reference (overlapping it, so an upper bound on agreement with
   the infinite-draw choice), agreement between every pair of disjoint groups (split-group reliability), and the override
   rate at S.
4. 1-3 for target_argmax.
5. the Q gap. Reference best a* = averaging's all-n choice; per seed gap_s = q_s(a*) - q_s(g); its seed-to-seed SD sd (sample,
   n - 1); z = mean_s gap_s / sd. Over the roots where a* != g: the distribution of z and of the gap, how many sit within 1
   (and 2) SD (z < 1, z < 2), and those with sd = 0 (chance never moves this gap). Per single-draw override (root and seed s
   with A_s != g): z_s = (q_s(A_s) - q_s(g)) / sd_t(q_t(A_s) - q_t(g)) over the n seeds t, and the share within 1 SD.
6. double estimator (van Hasselt): a1 = averaging's choice on the first half of the seeds' mean grid; override only when
   a1 != g and q_H2(a1) > q_H2(g) on the second half's mean grid, else g (and swapped: choose on the second half, check on
   the first). Its override rate and agreement with the all-n references, next to averaging and target_argmax at a single
   draw, at the half and at all n.
7. per root (roots.jsonl) the seed standard error of the gap: in sample (a* and its gap from all n: se = sd / sqrt(n)) and
   cross-fitted (a1 from the first half, its gap's mean and se = sd_H2 / sqrt(n/2) on the second half), and the override rate
   of the LCB rule "override only if gap > k * se" at k = 0, 1, 2, 3 for both.
Also: per-seed (b) of the effect guard against the live record's (guard stability), and the exclusions by reason.
Output (--out, a new directory): summary.json, roots.jsonl (one row per analysable root) and summary.md.

  python scripts/depthn_chance_noise.py --chance OUT/chance-*.jsonl --out NEW_DIRECTORY [--seeds 1,2,3,4,5,6,7,8]
"""
import argparse
import collections
from itertools import combinations
import json
import math
from pathlib import Path
import statistics
import sys

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT), str(_ROOT / 'src')):
    if _path not in sys.path:sys.path.insert(0, _path)

from gpu_engine import search_hedge as hedge  # noqa: E402
from gpu_engine.d3_search_latency import compose, reduce_block  # noqa: E402
from pokezero import v5_gumbel as original  # noqa: E402
from scripts import depthn_rr_score as score  # noqa: E402   (the RR scorer: guard_choices, target_argmax)

SCHEMA = 'pokezero.depthn-chance-noise.v1'
OPERATORS = ('averaging', 'target_argmax')
POPULATIONS = ('all', 'unguarded')
LCB_K = (0, 1, 2, 3)
QUANTILES = (.1, .25, .5, .75, .9)


class Excluded(Exception):
    def __init__(self, reason):super().__init__(reason);self.reason = reason


# ---- one root ----

def worlds_of(line, entry):
    """Seed `entry`'s WorldPrices in block order (the line's priors, or the entry's own when the priors varied by seed)."""
    priors = entry.get('priors') or line.get('priors');grids = entry.get('grids')
    if not priors or not grids or len(priors) != len(grids) or any(p is None for p in priors) or any(g is None for g in grids):raise Excluded('world_unpriced')
    try:return tuple(original.WorldPrices(p['opponent_requested'], tuple(p['kept']), tuple(p['root_peer_logits']), g) for p, g in zip(priors, grids))
    except (ValueError, TypeError, KeyError):raise Excluded('world_invalid') from None


def pattern(worlds):
    """Which grid cells are priced, per world."""
    return tuple(tuple(tuple(v is not None for v in row) for row in w.joint_values) for w in worlds)


def mean_worlds(by_seed, seeds):
    """The per-world mean grid over `seeds` (fsum / |seeds|; the priced cells are the same under every seed), priors unchanged."""
    first = by_seed[seeds[0]];n = len(seeds);out = []
    for w, world in enumerate(first):
        grid = tuple(tuple(None if world.joint_values[a][b] is None else math.fsum(by_seed[s][w].joint_values[a][b] for s in seeds) / n for b in range(9))
            for a in range(9))
        out.append(original.WorldPrices(world.opponent_requested, world.opponent_legal, world.opponent_logits, grid))
    return tuple(out)


def averaging(worlds, legal):
    """(averaging's played action, its q [9]) on these worlds; Excluded when it cannot price them."""
    choice = hedge.averaging(worlds, legal)
    if choice is None:raise Excluded('averaging_unknown_price')
    return hedge.top_action(choice.strategy), list(choice.q)


def target(worlds, logits, legal, q):
    """(target_argmax's action, the recomputed policy target or None, fallback used): compose over the K blocks as the server
    reports it, else the RR scorer's fallback argmax(root_logits + q)."""
    try:
        composed = compose([reduce_block(worlds[2 * k:2 * k + 2], logits, legal, original) for k in range(len(worlds) // 2)], logits, legal, original)
        policy = list(composed['policy_target']) if composed['status'] == 'LABELED' else None
    except ValueError:policy = None   # the range edge of reduce_completed_q: the scorer's fallback
    action, fallback = score.target_argmax(policy, legal, logits, q)
    return action, policy, fallback


def exact(x, y):return json.loads(json.dumps(x)) == json.loads(json.dumps(y))


def choose(by_seed, seeds, logits, legal):
    """dict(averaging, target_argmax, q, fallback) on the mean grid of `seeds`."""
    worlds = by_seed[seeds[0]] if len(seeds) == 1 else mean_worlds(by_seed, seeds)
    a, q = averaging(worlds, legal);t, policy, fallback = target(worlds, logits, legal, q)
    return dict(averaging=a, target_argmax=t, q=q, policy=policy, fallback=fallback)


def groups(seeds, size):
    """The fixed design: `seeds` cut into consecutive disjoint groups of `size`."""
    return [tuple(seeds[i:i + size]) for i in range(0, len(seeds), size)]


def sizes(n):return [s for s in (1, 2, 4, 8, 16) if s < n and n % s == 0]


def sample_sd(values):return statistics.stdev(values) if len(values) >= 2 else None


def ratio(gap, sd):
    """gap / sd, with sd = 0: +inf for a positive gap, None for none."""
    if sd is None:return None
    if sd > 0:return gap / sd
    return math.inf if gap > 0 else None


def analyse(line, seeds):
    """One chance line -> the root's analysis dict; Excluded with a reason when it is not analysable."""
    entries = {e['seed']: e for e in line.get('seeds') or ()}
    missing = [s for s in seeds if s not in entries]
    if missing:raise Excluded('seed_missing')
    if any(entries[s]['status'] != 'COMPLETE' for s in seeds):raise Excluded('seed_not_complete')
    root = line.get('root') or {};legal, logits = root.get('root_legal'), root.get('root_logits')
    if type(legal) is not list or len(legal) != 9 or not any(legal) or type(logits) is not list or len(logits) != 9:raise Excluded('root_invalid')
    legal = tuple(bool(x) for x in legal);greedy = line.get('greedy_action_index')
    if type(greedy) is not int or not 0 <= greedy < 9 or not legal[greedy]:raise Excluded('greedy_missing')
    by_seed = {s: worlds_of(line, entries[s]) for s in seeds}
    if len({pattern(w) for w in by_seed.values()}) != 1:raise Excluded('priced_cells_differ')
    if any(hedge.grids(w, legal) is None for w in by_seed.values()):raise Excluded('unpriced_cell')
    single = {s: choose(by_seed, (s,), logits, legal) for s in seeds}
    consistency = dict(   # the reduction to the live definition at one draw: each seed's own decision, recomputed
        averaging=all(single[s]['averaging'] == entries[s].get('action_index') and exact(single[s]['q'], entries[s].get('q_values')) for s in seeds),
        policy_target=all(single[s]['policy'] is not None and exact(single[s]['policy'], entries[s].get('policy_target')) or
            single[s]['policy'] is None and entries[s].get('policy_target') is None for s in seeds))
    n = len(seeds);reference = choose(by_seed, tuple(seeds), logits, legal);half = n // 2
    first, second = tuple(seeds[:half]), tuple(seeds[half:])
    h1, h2 = choose(by_seed, first, logits, legal), choose(by_seed, second, logits, legal)
    known = {(s,): single[s] for s in seeds};known.update({first: h1, second: h2})   # groups already chosen: the single draws, the halves
    design = {size: [known[g] if g in known else choose(by_seed, g, logits, legal) for g in groups(seeds, size)] for size in sizes(n)}
    # Q gap of the reference best against greedy, per seed
    best = reference['averaging'];gaps = [single[s]['q'][best] - single[s]['q'][greedy] for s in seeds]
    gap = math.fsum(gaps) / n;sd = sample_sd(gaps)
    draws = []
    for s in seeds:
        a = single[s]['averaging']
        if a == greedy:continue
        spread = sample_sd([single[t]['q'][a] - single[t]['q'][greedy] for t in seeds]);own = single[s]['q'][a] - single[s]['q'][greedy]
        draws.append(dict(seed=s, action=a, gap=own, sd=spread, z=ratio(own, spread)))
    # cross-fitted: choose on the first half, measure on the second (and swapped)
    def cross(choose_on, measure_on, chosen):
        a = chosen['averaging'];values = [single[s]['q'][a] - single[s]['q'][greedy] for s in measure_on]
        m = math.fsum(values) / len(values);spread = sample_sd(values)
        return dict(action=a, gap=m, sd=spread, se=None if spread is None else spread / math.sqrt(len(values)))
    fit, swapped = cross(first, second, h1), cross(second, first, h2)
    def double(chosen, check):
        a = chosen['averaging']
        return a if a != greedy and check['q'][a] > check['q'][greedy] else greedy
    se = None if sd is None else sd / math.sqrt(n)
    lcb = {f'k={k}': dict(in_sample=best != greedy and se is not None and gap > k * se,
        cross_fit=fit['action'] != greedy and fit['se'] is not None and fit['gap'] > k * fit['se']) for k in LCB_K}
    guard = line.get('effect_guard') if isinstance(line.get('effect_guard'), dict) else None
    seed_b = {s: (entries[s].get('effect_guard') or {}).get('b') for s in seeds}
    return dict(ordinal=line.get('ordinal'), decision_index=line.get('decision_index'), battle_id=line.get('battle_id'), turn=line.get('turn'),
        request_kind=line.get('request_kind'), greedy=greedy, greedy_source=line.get('greedy_source'), legal_count=sum(legal), seeds=list(seeds),
        single={op: [single[s][op] for s in seeds] for op in OPERATORS}, reference={op: reference[op] for op in OPERATORS},
        halves={op: [h1[op], h2[op]] for op in OPERATORS},
        design={str(size): {op: [c[op] for c in chosen] for op in OPERATORS} for size, chosen in design.items()},
        double_estimator=double(h1, h2), double_estimator_swapped=double(h2, h1),
        gap=dict(best=best, gap=gap, sd=sd, se=se, z=ratio(gap, sd), gaps=gaps), single_draw_overrides=draws, cross_fit=fit, cross_fit_swapped=swapped,
        lcb=lcb, target_fallbacks=sum(single[s]['fallback'] for s in seeds) + reference['fallback'], consistency=consistency,
        guard=None if guard is None else dict(b=bool(guard.get('b')), own_guarded_lanes=guard.get('own_guarded_lanes')),
        guard_seed_b=seed_b, guard_b_stable=None if guard is None or any(v is None for v in seed_b.values()) else all(v == bool(guard.get('b')) for v in seed_b.values()),
        own_moves=root.get('own_moves'), record=guard)


# ---- guards ----

def guarded(root, choice):
    """`choice` as the live actor plays it at this root (the RR scorer's guard_choices with the live record)."""
    played, _ = score.guard_choices({'x': choice}, root['greedy'], root['own_moves'], root['record'])
    return played['x']


def view(root, population):
    """The root's choices as a population sees them: raw for 'all'; through the live guards for 'unguarded'."""
    if population == 'all':return root
    g = lambda c: guarded(root, c)
    out = dict(root, single={op: [g(c) for c in cs] for op, cs in root['single'].items()}, reference={op: g(c) for op, c in root['reference'].items()},
        halves={op: [g(c) for c in cs] for op, cs in root['halves'].items()},
        design={k: {op: [g(c) for c in cs] for op, cs in v.items()} for k, v in root['design'].items()},
        double_estimator=g(root['double_estimator']), double_estimator_swapped=g(root['double_estimator_swapped']))
    out['lcb'] = {k: dict(in_sample=v['in_sample'] and g(root['gap']['best']) != root['greedy'],
        cross_fit=v['cross_fit'] and g(root['cross_fit']['action']) != root['greedy']) for k, v in root['lcb'].items()}
    return out


def members(roots, population):
    if population == 'all':return list(roots)
    return [r for r in roots if r['record'] is not None and not r['record'].get('b')]


# ---- statistics ----

def mean(values):
    values = [v for v in values if v is not None]
    return math.fsum(values) / len(values) if values else None


def quantiles(values):
    xs = sorted(v for v in values if v is not None and math.isfinite(v))
    if not xs:return None
    pick = lambda q: xs[min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))]
    return dict(n=len(xs), **{f'p{int(100 * q)}': pick(q) for q in QUANTILES}, min=xs[0], max=xs[-1])


def flip_rate(choices):
    pairs = list(combinations(choices, 2))
    return math.fsum(a != b for a, b in pairs) / len(pairs) if pairs else None


def operator_metrics(roots, op):
    """1-3 for one operator over one population's roots."""
    if not roots:return None
    out = dict(roots=len(roots), flip_rate=mean(flip_rate(r['single'][op]) for r in roots),
        roots_with_any_flip=mean(float(len(set(r['single'][op])) > 1) for r in roots),
        override_single=mean(mean(float(c != r['greedy']) for c in r['single'][op]) for r in roots),
        override_reference=mean(float(r['reference'][op] != r['greedy']) for r in roots),
        single_agrees_reference=mean(mean(float(c == r['reference'][op]) for c in r['single'][op]) for r in roots))
    convergence = {}
    for size in sorted({int(k) for r in roots for k in r['design']}):
        rows = [r for r in roots if str(size) in r['design']]
        chosen = lambda r: r['design'][str(size)][op]
        convergence[str(size)] = dict(groups=len(chosen(rows[0])) if rows else None,
            agree_reference=mean(mean(float(c == r['reference'][op]) for c in chosen(r)) for r in rows),
            agree_disjoint=mean(mean(float(a == b) for a, b in combinations(chosen(r), 2)) for r in rows),
            override=mean(mean(float(c != r['greedy']) for c in chosen(r)) for r in rows))
    n = len(roots[0]['seeds']);convergence[str(n)] = dict(groups=1, agree_reference=1., agree_disjoint=None, override=out['override_reference'])
    out['convergence'] = convergence
    return out


def gap_metrics(roots):
    """5: over the roots whose all-n averaging choice overrides greedy, and over every single-draw override."""
    overrides = [r for r in roots if r['gap']['best'] != r['greedy']]
    z = [r['gap']['z'] for r in overrides];draws = [d for r in roots for d in r['single_draw_overrides']]
    zd = [d['z'] for d in draws]
    count = lambda xs, k: sum(1 for x in xs if x is not None and x < k)
    return dict(override_roots=len(overrides), z=quantiles(z), gap=quantiles(r['gap']['gap'] for r in overrides),
        sd=quantiles(r['gap']['sd'] for r in overrides), within_1sd=count(z, 1), within_2sd=count(z, 2),
        zero_sd=sum(1 for r in overrides if r['gap']['sd'] == 0), share_within_1sd=count(z, 1) / len(overrides) if overrides else None,
        single_draw=dict(overrides=len(draws), z=quantiles(zd), within_1sd=count(zd, 1), within_2sd=count(zd, 2),
            zero_sd=sum(1 for d in draws if d['sd'] == 0), share_within_1sd=count(zd, 1) / len(draws) if draws else None))


def rules(roots):
    """6: override rate and agreement with both all-n references of every rule."""
    if not roots:return None
    table = {}
    def add(name, played):
        choices = [played(r) for r in roots]
        table[name] = dict(override=mean(float(c != r['greedy']) for c, r in zip(choices, roots)),
            agree_averaging_reference=mean(float(c == r['reference']['averaging']) for c, r in zip(choices, roots)),
            agree_target_reference=mean(float(c == r['reference']['target_argmax']) for c, r in zip(choices, roots)))
    for op in OPERATORS:   # a single draw: averaged over the seeds
        choices = lambda r, op=op: r['single'][op]
        table[f'{op}:single_draw'] = dict(override=mean(mean(float(c != r['greedy']) for c in choices(r)) for r in roots),
            agree_averaging_reference=mean(mean(float(c == r['reference']['averaging']) for c in choices(r)) for r in roots),
            agree_target_reference=mean(mean(float(c == r['reference']['target_argmax']) for c in choices(r)) for r in roots))
        add(f'{op}:first_half', lambda r, op=op: r['halves'][op][0])
        add(f'{op}:all_seeds', lambda r, op=op: r['reference'][op])
    add('double_estimator', lambda r: r['double_estimator'])
    add('double_estimator_swapped', lambda r: r['double_estimator_swapped'])
    return table


def lcb_metrics(roots):
    """7: the LCB rule's override rate at each k, in sample and cross-fitted."""
    if not roots:return None
    return {k: dict(in_sample=mean(float(r['lcb'][k]['in_sample']) for r in roots), cross_fit=mean(float(r['lcb'][k]['cross_fit']) for r in roots))
        for k in (f'k={k}' for k in LCB_K)}


def summarize(roots, excluded, lines, seeds):
    out = dict(schema=SCHEMA, seeds=list(seeds), lines=lines, analysed=len(roots), excluded=dict(excluded),
        consistency=dict(averaging=sum(r['consistency']['averaging'] for r in roots), policy_target=sum(r['consistency']['policy_target'] for r in roots),
            roots=len(roots), note='each seed recomputed from its grids reproduces its own action_index, q_values and policy_target exactly'),
        target_fallbacks=sum(r['target_fallbacks'] for r in roots),
        guard=dict(with_record=sum(r['record'] is not None for r in roots), b=sum(bool(r['record'] and r['record'].get('b')) for r in roots),
            seed_b_unstable=sum(r['guard_b_stable'] is False for r in roots), seed_b_unknown=sum(r['guard_b_stable'] is None for r in roots)),
        populations={})
    for population in POPULATIONS:
        members_ = [view(r, population) for r in members(roots, population)]
        out['populations'][population] = dict(roots=len(members_), battles=len({r['ordinal'] for r in members_}),
            operators={op: operator_metrics(members_, op) for op in OPERATORS}, gap=gap_metrics(members_), rules=rules(members_), lcb=lcb_metrics(members_))
    return out


# ---- markdown ----

def _pct(x):return '-' if x is None else f'{100 * x:.1f}%'


def _f(x, digits=4):return '-' if x is None else ('inf' if x == math.inf else f'{x:.{digits}f}')


def markdown(summary):
    s = summary;n = len(s['seeds'])
    lines = ['# DepthN chance noise: one engine chance draw per world', '',
        f"- seeds {s['seeds']} (n = {n}); lines {s['lines']}, analysed {s['analysed']}, excluded {s['excluded']}",
        f"- consistency (each seed recomputed reproduces its own decision): averaging {s['consistency']['averaging']}/{s['consistency']['roots']}, "
        f"policy_target {s['consistency']['policy_target']}/{s['consistency']['roots']}; target fallbacks {s['target_fallbacks']}",
        f"- effect guard: record on {s['guard']['with_record']} roots, (b) on {s['guard']['b']}; a seed's (b) differs from the live record's on "
        f"{s['guard']['seed_b_unstable']} (unknown {s['guard']['seed_b_unknown']})", '']
    for population, block in s['populations'].items():
        lines += [f"## {population} ({block['roots']} roots, {block['battles']} battles)" + ('' if population == 'all' else
            ': the live guard\'s (b) false, every choice through the RR scorer\'s guards'), '']
        if not block['roots']:lines += ['no roots', ''];continue
        lines += ['| operator | flip rate (seed pairs) | roots with any flip | override, one draw | override, all seeds | one draw agrees with all seeds |',
            '|---|---|---|---|---|---|']
        for op, m in block['operators'].items():
            lines.append(f"| {op} | {_pct(m['flip_rate'])} | {_pct(m['roots_with_any_flip'])} | {_pct(m['override_single'])} | {_pct(m['override_reference'])} "
                f"| {_pct(m['single_agrees_reference'])} |")
        lines += ['', '| operator | S | groups | agrees with all-seed reference | disjoint groups agree | override |', '|---|---|---|---|---|---|']
        for op, m in block['operators'].items():
            for size, c in m['convergence'].items():
                lines.append(f"| {op} | {size} | {c['groups']} | {_pct(c['agree_reference'])} | {_pct(c['agree_disjoint'])} | {_pct(c['override'])} |")
        g = block['gap'];z = g['z'] or {};zd = g['single_draw']['z'] or {}
        lines += ['', f"Q gap Q(best) - Q(greedy) of the all-seed averaging override, in units of its seed-to-seed SD: {g['override_roots']} override roots; "
            f"within 1 SD {g['within_1sd']} ({_pct(g['share_within_1sd'])}), within 2 SD {g['within_2sd']}, SD 0 {g['zero_sd']}; z quantiles "
            + ', '.join(f"{k} {_f(z.get(k), 2)}" for k in ('p10', 'p25', 'p50', 'p75', 'p90')),
            f"Single-draw overrides: {g['single_draw']['overrides']} (root, seed) pairs; within 1 SD of their own gap's spread "
            f"{g['single_draw']['within_1sd']} ({_pct(g['single_draw']['share_within_1sd'])}), within 2 SD {g['single_draw']['within_2sd']}; z quantiles "
            + ', '.join(f"{k} {_f(zd.get(k), 2)}" for k in ('p10', 'p25', 'p50', 'p75', 'p90')), '',
            '| rule | override | agrees with averaging (all seeds) | agrees with target_argmax (all seeds) |', '|---|---|---|---|']
        for name, m in block['rules'].items():
            lines.append(f"| {name} | {_pct(m['override'])} | {_pct(m['agree_averaging_reference'])} | {_pct(m['agree_target_reference'])} |")
        lines += ['', '| LCB rule: override only if gap > k * se | in sample (a* and gap from all seeds) | cross-fitted (a* first half, gap second half) |',
            '|---|---|---|']
        for k, m in block['lcb'].items():lines.append(f"| {k} | {_pct(m['in_sample'])} | {_pct(m['cross_fit'])} |")
        lines.append('')
    return '\n'.join(lines) + '\n'


# ---- run ----

def load(paths):
    for path in paths:
        with open(path, 'rb') as stream:
            for raw in stream:
                if raw.strip():yield json.loads(raw)


def default_seeds(lines):
    """The run's non-zero chance seeds, ascending (seed 0 is the control): the seeds the lines declare (chance_seeds, the
    re-pricer's --chance-seeds; the most common list), else every seed any line priced. A line missing one is excluded
    (seed_missing), never the seed."""
    declared = collections.Counter(tuple(line['chance_seeds']) for line in lines if isinstance(line.get('chance_seeds'), list))
    seen = set(declared.most_common(1)[0][0]) if declared else {e['seed'] for line in lines for e in line.get('seeds') or ()}
    return tuple(sorted(s for s in seen if s != 0))


def run(paths, out, seeds=None):
    lines = list(load(paths));seeds = tuple(seeds) if seeds else default_seeds(lines)
    if len(seeds) < 2 or len(seeds) % 2:raise ValueError(f'an even number (>= 2) of analysis seeds required, got {list(seeds)}')
    out = Path(out);out.mkdir(parents=True, exist_ok=False)
    roots, excluded, seen = [], collections.Counter(), set()
    with (out / 'roots.jsonl').open('x') as stream:
        for line in lines:
            key = (line.get('ordinal'), line.get('decision_index'))
            if key in seen:excluded['duplicate'] += 1;continue
            seen.add(key)
            try:root = analyse(line, seeds)
            except Excluded as error:excluded[error.reason] += 1;continue
            roots.append(root);stream.write(json.dumps({k: v for k, v in root.items() if k != 'record'}, allow_nan=True) + '\n')
    summary = summarize(roots, excluded, len(lines), seeds)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    (out / 'summary.md').write_text(markdown(summary))
    return summary


def seeds_arg(text):
    try:values = tuple(int(x) for x in text.split(','))
    except ValueError:raise argparse.ArgumentTypeError('comma-separated integer seeds required') from None
    if len(set(values)) != len(values):raise argparse.ArgumentTypeError('distinct seeds required')
    return values


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    p.add_argument('--chance', nargs='+', required=True, help="the re-pricer's chance-*.jsonl")
    p.add_argument('--out', required=True, help='a new output directory')
    p.add_argument('--seeds', type=seeds_arg, help='the analysis seeds, in order (default: every non-zero seed of the lines)')
    args = p.parse_args(argv)
    summary = run(args.chance, args.out, args.seeds)
    block = summary['populations']['all']
    print(json.dumps(dict(out=str(Path(args.out).resolve()), analysed=summary['analysed'], excluded=summary['excluded'],
        flip_rate={op: (block['operators'][op] or {}).get('flip_rate') for op in OPERATORS})))
    return 0


if __name__ == '__main__':sys.exit(main())
