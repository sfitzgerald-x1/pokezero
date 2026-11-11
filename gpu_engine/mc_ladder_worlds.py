"""Monte Carlo-truth ladder, the worlds lever: depth-1 world rules on the re-pricer's rows, and how much of each root's sampled
worlds falls on the opponent's true sets. CPU only: host arithmetic through mc_ladder_rungs' and depthn_rr_score's own operators;
no engine, model or device.

Inputs: rows of scripts/depthn_rr_reprice.py --targets replays of the gate bank's roots, one root kind per call.
- --sampled: a sampled-world replay: run C (Foul Play roots, --reply-archive pi_FP and --peer-view-b) or run E (self roots, the
  live configuration), or a wider replay of the same roots and configuration (--rung DN2W32 / DN2W64) for the world-count rungs.
- --true (optional): the same roots' true-world replay, run A or B (--true-worlds, --true-kind true_private). World i of a true row
  is the true state materialized with sampled world i's kernel seed (checked per root: kernel_seed_mismatch otherwise), so true
  world 0 is sampled world 0 with the opponent's true sets in place of the sampled ones, under the same chance draw.
- --weights / --true-weights (optional): gpu_engine.fp_reply_evidence weights rows ({ordinal, decision_index, world, loglik})
  joined to --sampled's / --true's rows: the opponent model's log-likelihood of the opponent's earlier public choices given each
  world's sets (pi_FP on Foul Play roots; the champion, gpu_engine.mc_ladder_evidence, on self roots).
- --manifest: the self-capture manifest (self roots only), as mc_ladder_rungs takes it.

Rungs, in mc_ladder_score's rung format exactly as mc_ladder_rungs writes them. The reply prior of every rung is R3's: the reply
model's logits (reply_peer_logits) on Foul Play roots and the champion's root_peer_logits on self roots, at depth 1.
- R3: averaging over the sampled worlds (mc_ladder_rungs.average): run C's / E's own R3, recomputed (summary r3_check counts any
  disagreement with --check-r3, that run's rung-R3.jsonl).
- R3true: the true world in sampled world 0's place: worlds T0, S1, ..., S(W-1), averaged.
- R3true_ins: the true world added: S0, ..., S(W-1), T0 (W + 1 worlds), averaged.
- R3rw_b<beta>: the sampled worlds weighted p_w = softmax_w(beta * loglik_w) (depthn_rr_score.world_weights) instead of 1/W,
  q(a) = sum_w p_w sum_b pi_w(b) G_w[a][b] (depthn_rr_score.weighted_q), argmax lowest index on ties. At beta = 0 and a
  power-of-two W this is averaging's q bit for bit.
- R3true_rw_b<beta>: R3true's worlds weighted the same way (T0's loglik: --true-weights' world 0).
- --prefixes N ...: every rung on the first N sampled worlds only, named <rung>_W<N> (a DN2W64 replay's first 16 worlds are a
  DN2W16 replay's: phase_d_protocol_world.sample seeds world i independently of the world count). A prefix's depth-1 blocks must
  be unrefused; later blocks do not matter to it.
Guards: the live actor's on the sampled row for every rung (mc_ladder_rungs.guard on the rung's own choice: the row's effect_guard
record, whose (b) the sampled worlds decided, and the own-recharge guard). An undecided root's reason is mc_ladder_rungs' (depth1_missing[:cause],
refused_at_depth1, ...) or: true_row_missing, true_worlds_failed / true_worlds_not_sampled (mc_ladder_rungs.true_worlds_check),
true_world_unpriced, kernel_seed_mismatch, root_prior_differs (the two rows' root logits or legal mask differ),
world_weights_missing / world_weights_mismatch, true_weights_missing, prefix_too_long.

Diagnostics (diag.jsonl, one row per root and prefix with a usable true row; summary.json's diagnostics aggregate them). A world's
set for a Pokemon is fp_reply_evidence.set_key (move keys, base ability, item at the root) of the re-pricer's opponent party; the
true sets are true world 0's party. Seen Pokemon: species in the true party and in every sampled world (the revealed ones; a
hidden backline is drawn afresh per world). Per root, for the uniform weights (raw) and each beta's weights (rw_b<beta>):
- joint: the weight on worlds holding every seen Pokemon's true set (P(true set)); log_joint its log (None at 0);
- per_mon: the weight on worlds holding that Pokemon's true set; active: the opponent's active Pokemon's (true world 0's
  opponent.active_species; None when not seen);
- ess: 1 / sum_w p_w^2.
And: covered (some world holds every true set), mons_covered (per seen Pokemon: some world holds its true set); with both weight
files, true_loglik_minus_max (T0's loglik minus the sampled worlds' best), true_loglik_rank (the share of sampled worlds below
T0's loglik, ties half) and true_weight_b<beta> (T0's weight in R3true_rw_b<beta>).

usage:
  python -m gpu_engine.mc_ladder_worlds rungs --kind foulplay|self --sampled ROWS.jsonl ... [--true ROWS.jsonl ...]
      [--weights W.jsonl] [--true-weights W.jsonl] [--manifest MANIFEST.jsonl] [--prefixes N ...] [--check-r3 rung-R3.jsonl]
      --out NEW_DIR
  python -m gpu_engine.mc_ladder_worlds pairs BANK_DIR --rung NAME=PATH ... [--base NAME ...] [--guards on|policy|off] --out OUT.json
rungs writes NEW_DIR/rung-<NAME>.jsonl per rung, diag.jsonl and summary.json (rows read, identity mismatches, per rung decided /
undecided by reason / guard trips, the R3 check and the diagnostics). Exit 1 when the R3 check disagrees anywhere, else 0.
pairs scores every rung (and the bank's own R0) with mc_ladder_score.score, and every rung against each --base on the roots both
decided: D = Q(rung) - Q(base), the mean with a battle-clustered 90% interval (mc_root_report.cluster_ci), per root kind.
"""
import argparse
import collections
import json
import math
from pathlib import Path

from scripts import depthn_rr_score as rr   # first: it puts src/ on sys.path for pokezero
from gpu_engine import fp_reply_evidence as ev
from gpu_engine import mc_ladder_rungs as L
from gpu_engine import search_hedge as hedge

SCHEMA = 'mc-ladder-worlds.v1'
BETAS = (0.5, 1.0)
SAMPLED_RUN = dict(foulplay='C', self='E')   # mc_ladder_rungs' run of the same kind and reply prior (identity and true-row checks)
TRUE_RUN = dict(foulplay='A', self='B')
REPLY = dict(foulplay=True, self=False)       # R3's reply prior: the reply model's logits on Foul Play roots, the champion's on self
EXAMPLES = 20


def beta_name(beta):
    return f'{beta:g}'


def rung_names(weighted, with_true, true_weighted):
    names = ['R3']
    if with_true:names += ['R3true', 'R3true_ins']
    if weighted:names += [f'R3rw_b{beta_name(b)}' for b in BETAS]
    if with_true and weighted and true_weighted:names += [f'R3true_rw_b{beta_name(b)}' for b in BETAS]
    return names


def load(paths, kind, manifest, run, counts):
    """{root index: (key, raw row)} of one run's rows (mc_ladder_rungs.identity and true_worlds_check); mismatches counted."""
    out = {}
    for raw in L.read_rows(paths):
        counts['rows'] += 1;key, problem = L.identity(raw, kind, manifest);truth = None
        if key is not None:problem, truth = L.true_worlds_check(raw, run)
        if problem is not None:counts[f'identity_{problem}'] += 1;continue
        index = tuple(key[f] for f in L.KEY_FIELDS[kind])
        if index in out:raise ValueError(f'two rows for root {index}')
        out[index] = (key, raw, truth)
    return out


def depth1_worlds(raw, count=None):
    """The row's world dicts (the first `count`), after mc_ladder_rungs.averaged's depth-1 checks on the blocks that hold them;
    raises rr.Unscorable."""
    info = raw.get('depth1')
    if type(info) is not dict:raise rr.Unscorable('depth1_missing')
    if 'refused_at_depth1' not in info:raise rr.Unscorable('row_schema_old')
    worlds = raw.get('worlds') if type(raw.get('worlds')) is list else []
    refused = list(info['refused_at_depth1'])
    if count is None:count = len(worlds)
    if count > len(worlds):raise rr.Unscorable('prefix_too_long')
    per = max(1, len(worlds) // max(1, len(refused)))   # block k holds worlds per*k .. per*k + per - 1
    if any(refused[:-(-count // per)]):raise rr.Unscorable('refused_at_depth1')
    return worlds[:count]


def prices_of(worlds, reply):
    """WorldPrices of raw world dicts at depth 1; with `reply`, each requested world's prior is its reply_peer_logits."""
    prices = rr.world_prices(rr.normalize(dict(worlds=[w.get('depth1') if type(w) is dict else None for w in worlds])))
    if reply:prices = rr.model_worlds(prices, rr.reply_logits_of(rr.normalize(dict(worlds=worlds)), prices, None))
    return prices


def weighted_choice(prices, legal, weights):
    """argmax of depthn_rr_score.weighted_q (lowest index on ties); raises rr.Unscorable."""
    if len(weights) != len(prices):raise rr.Unscorable('world_weights_mismatch')
    try:gs = hedge.grids(prices, legal)
    except ValueError:raise rr.Unscorable('averaging_error') from None
    if gs is None:raise rr.Unscorable('unpriced_cell')
    q = rr.weighted_q(gs, weights)
    if any(not math.isfinite(v) for v in q.values()):raise rr.Unscorable('nonfinite')
    return rr.argmax(q, sorted(q))


def choice(worlds, legal, reply, weights=None):
    prices = prices_of(worlds, reply)
    return L.average(prices, legal)[0] if weights is None else weighted_choice(prices, legal, weights)


def true_world(true_entry, sampled_world):
    """True world 0 of a usable true row, or raises rr.Unscorable."""
    if true_entry is None:raise rr.Unscorable('true_row_missing')
    _, raw, truth = true_entry
    if truth is not None:raise rr.Unscorable(truth)
    worlds = depth1_worlds(raw, 1)
    if not worlds or type(worlds[0]) is not dict or type(worlds[0].get('depth1')) is not dict:raise rr.Unscorable('true_world_unpriced')
    if worlds[0].get('kernel_seed') != (sampled_world or {}).get('kernel_seed'):raise rr.Unscorable('kernel_seed_mismatch')
    return worlds[0]


def same_root(a, b):
    """The two rows' root prior (root_logits, root_legal) is the same."""
    x, y = rr.normalize(a), rr.normalize(b)
    return x['root_logits'] == y['root_logits'] and x['root_legal'] == y['root_legal']


def logliks(table, key, count, reason='world_weights_missing'):
    """[loglik_w] of a root's first `count` worlds from a load_world_weights table (None: no table)."""
    if table is None:return None
    entry = table.get(key)
    if entry is None or any(w not in entry for w in range(count)):raise rr.Unscorable(reason)
    return [entry[w] for w in range(count)]


def decide(sampled, true_entry, weights, true_weights, kind, count, with_true=True):
    """{rung: (action or None, reason or None)} for one root on its first `count` sampled worlds (the true-world rungs only
    `with_true`)."""
    raw = sampled[1];d = rr.normalize(raw);reply = REPLY[kind];out = {}
    depth1 = raw.get('depth1') if type(raw.get('depth1')) is dict else {}
    if d['root_logits'] is None and d['root_legal'] is None and depth1.get('root_logits') is not None:   # as mc_ladder_rungs.choose
        d = dict(d, root_logits=depth1['root_logits'], root_legal=depth1['root_legal'])
    names = rung_names(weights is not None, with_true, true_weights is not None)

    def undecided(reason):
        cause = d['refusal_cause']
        if cause and reason == 'depth1_missing':reason += f':{cause}'
        return None, reason

    try:
        legal, _ = L.root_prior(d);worlds = depth1_worlds(raw, count)
    except rr.Unscorable as error:return {name: undecided(error.reason) for name in names}, d
    key = (raw['ordinal'], raw['decision_index'])

    def run(name, fn):
        try:out[name] = fn(), None
        except rr.Unscorable as error:out[name] = undecided(error.reason)

    run('R3', lambda: choice(worlds, legal, reply))
    mixed = added = problem = None
    try:
        if not with_true:raise rr.Unscorable('no_true_rows')
        truth = true_world(true_entry, worlds[0] if worlds else None)
        if not same_root(raw, true_entry[1]):raise rr.Unscorable('root_prior_differs')
        mixed, added = [truth] + worlds[1:], worlds + [truth]
    except rr.Unscorable as error:problem = error.reason
    for name in ('R3true', 'R3true_ins') if with_true else ():
        if problem:out[name] = undecided(problem);continue
        run(name, lambda m=(mixed if name == 'R3true' else added): choice(m, legal, reply))
    if weights is not None:
        try:sampled_ll = logliks(weights, key, len(raw['worlds']))[:count];ll_problem = None
        except rr.Unscorable as error:sampled_ll, ll_problem = None, error.reason
        for beta in BETAS:
            name = f'R3rw_b{beta_name(beta)}'
            if ll_problem:out[name] = undecided(ll_problem);continue
            run(name, lambda b=beta: choice(worlds, legal, reply, rr.world_weights(sampled_ll, b)))
        if with_true and true_weights is not None:
            for beta in BETAS:
                name = f'R3true_rw_b{beta_name(beta)}'
                if problem or ll_problem:out[name] = undecided(problem or ll_problem);continue
                try:t0 = logliks(true_weights, key, 1, 'true_weights_missing')[0]
                except rr.Unscorable as error:out[name] = undecided(error.reason);continue
                run(name, lambda b=beta: choice(mixed, legal, reply, rr.world_weights([t0] + sampled_ll[1:], b)))
    return out, d


def weighting(count, ll, beta):
    return [1. / count] * count if beta is None else rr.world_weights(ll, beta)


def diagnose(sampled, true_entry, weights, true_weights, count):
    """The root's P(true set) diagnostics on its first `count` sampled worlds (module docstring), or (None, reason)."""
    raw = sampled[1];worlds = raw.get('worlds') if type(raw.get('worlds')) is list else []
    if count > len(worlds) or not count:return None, 'prefix_too_long' if worlds else 'no_worlds'
    worlds = worlds[:count]
    if true_entry is None:return None, 'true_row_missing'
    if true_entry[2] is not None:return None, true_entry[2]
    twin = (true_entry[1].get('worlds') or [None])[0]
    if type(twin) is not dict:return None, 'true_world_missing'
    truth = ev.world_sets(twin);sets = [ev.world_sets(w) if type(w) is dict else None for w in worlds]
    if truth is None or any(s is None for s in sets):return None, 'world_sets_missing'
    seen = sorted(m for m in truth if all(m in s for s in sets))
    if not seen:return None, 'nothing_seen'
    active = (twin.get('opponent') or {}).get('active_species');active = ev.species_key(active) if active else None
    hits = [[s[m] == truth[m] for m in seen] for s in sets];joint = [all(h) for h in hits]
    key = (raw['ordinal'], raw['decision_index'])
    try:ll = logliks(weights, key, len(raw['worlds']))
    except rr.Unscorable:ll = None
    ll = None if ll is None else ll[:count]
    out = dict(worlds=count, seen=seen, active=active if active in seen else None, covered=any(joint),
               mons_covered={m: any(h[i] for h in hits) for i, m in enumerate(seen)}, weighted=ll is not None)
    for label, beta in [('raw', None)] + ([(f'rw_b{beta_name(b)}', b) for b in BETAS] if ll is not None else []):
        p = weighting(count, ll, beta);j = math.fsum(x for x, ok in zip(p, joint) if ok)
        per = {m: math.fsum(x for x, h in zip(p, hits) if h[i]) for i, m in enumerate(seen)}
        out[label] = dict(joint=j, log_joint=math.log(j) if j > 0 else None, per_mon=per, active=per.get(active),
                          ess=1. / math.fsum(x * x for x in p))
    if ll is not None and true_weights is not None and key in true_weights and 0 in true_weights[key]:
        t0 = true_weights[key][0];out['true_loglik_minus_max'] = t0 - max(ll)
        out['true_loglik_rank'] = (sum(x < t0 for x in ll) + .5 * sum(x == t0 for x in ll)) / count
        for b in BETAS:out[f'true_weight_b{beta_name(b)}'] = rr.world_weights([t0] + ll[1:], b)[0]
    return out, None


def mean(xs):
    xs = [x for x in xs if x is not None];return math.fsum(xs) / len(xs) if xs else None


def summarize_diag(rows):
    """Aggregates of one prefix's diag rows (means over roots; per-Pokemon over (root, Pokemon) pairs), and the paired
    reweighted-minus-raw differences with battle-clustered 90% intervals."""
    from gpu_engine.mc_root_report import cluster_ci
    if not rows:return dict(roots=0)
    labels = ['raw'] + [f'rw_b{beta_name(b)}' for b in BETAS]
    out = dict(roots=len(rows), weighted=sum(r['weighted'] for r in rows), covered=mean([float(r['covered']) for r in rows]),
               mons=sum(len(r['seen']) for r in rows), mons_covered=mean([float(v) for r in rows for v in r['mons_covered'].values()]),
               active_roots=sum(r['active'] is not None for r in rows),
               active_covered=mean([float(r['mons_covered'][r['active']]) for r in rows if r['active'] is not None]),
               seen_per_root=mean([len(r['seen']) for r in rows]))
    for label in labels:
        have = [r for r in rows if label in r]
        if not have:continue
        out[label] = dict(roots=len(have), joint=mean([r[label]['joint'] for r in have]),
                          log_joint_covered=mean([r[label]['log_joint'] for r in have]),
                          per_mon=mean([v for r in have for v in r[label]['per_mon'].values()]),
                          active=mean([r[label]['active'] for r in have]), ess=mean([r[label]['ess'] for r in have]))
    for label in labels[1:]:
        have = [r for r in rows if label in r]
        if not have:continue
        seeds = [r['seed'] for r in have]
        both = [r for r in have if r['raw']['log_joint'] is not None and r[label]['log_joint'] is not None]
        act = [r for r in have if r['active'] is not None]
        out[f'{label}_minus_raw'] = dict(
            joint=cluster_ci([r[label]['joint'] - r['raw']['joint'] for r in have], seeds),
            log_joint_covered=cluster_ci([r[label]['log_joint'] - r['raw']['log_joint'] for r in both], [r['seed'] for r in both]),
            active=cluster_ci([r[label]['active'] - r['raw']['active'] for r in act], [r['seed'] for r in act]))
    ranked = [r for r in rows if 'true_loglik_rank' in r]
    if ranked:
        out['true_world'] = dict(roots=len(ranked), loglik_rank=mean([r['true_loglik_rank'] for r in ranked]),
                                 loglik_minus_max=mean([r['true_loglik_minus_max'] for r in ranked]),
                                 above_every_sampled=mean([float(r['true_loglik_minus_max'] > 0) for r in ranked]),
                                 **{f'weight_b{beta_name(b)}': mean([r[f'true_weight_b{beta_name(b)}'] for r in ranked]) for b in BETAS})
    return out


def build(sampled_rows, kind, true_rows=None, weights=None, true_weights=None, manifest=None, prefixes=None, check_r3=None):
    """({rung: [rung rows]}, [diag rows], summary)."""
    manifest = manifest or {};counts = collections.Counter();true_counts = collections.Counter()
    sampled = load(sampled_rows, kind, manifest, SAMPLED_RUN[kind], counts)
    with_true = true_rows is not None
    truths = load(true_rows, kind, manifest, TRUE_RUN[kind], true_counts) if with_true else {}
    full = max((len(raw.get('worlds') or ()) for _, raw, _ in sampled.values()), default=0)
    sizes = sorted(set(prefixes)) if prefixes else [None]
    rungs = collections.defaultdict(list);diag = [];stats = {};diag_reasons = collections.Counter()
    for index in sorted(sampled):
        entry = sampled[index];key, raw, _ = entry;true_entry = truths.get(index)
        for size in sizes:
            count = len(raw.get('worlds') or ()) if size is None else size
            chosen, d = decide(entry, true_entry, weights, true_weights, kind, count, with_true)
            try:legal, greedy = L.root_prior(d)
            except rr.Unscorable:legal = greedy = None
            for name, (action, reason) in chosen.items():
                name = name if size is None else f'{name}_W{size}'
                row = dict(key, action=action, greedy_action=greedy, guard_tripped=False,
                           actions=None if legal is None else [a for a in range(9) if legal[a]])
                s = stats.setdefault(name, dict(rows=0, decided=0, undecided=collections.Counter(), guard_tripped=collections.Counter()))
                s['rows'] += 1
                if action is None:row['reason'] = reason;s['undecided'][reason] += 1
                else:
                    s['decided'] += 1;tripped, why = L.guard(d, action, greedy, REPLY[kind])
                    if tripped:row.update(guard_tripped=True, guard_reason=why);s['guard_tripped'][why] += 1
                rungs[name].append(row)
            if with_true:
                found, reason = diagnose(entry, true_entry, weights, true_weights, count)
                if found is None:diag_reasons[(size, reason)] += 1
                else:diag.append(dict(key, prefix=size, seed=raw.get('seed') if kind == 'foulplay' else key['seed'], **found))
    check = None
    if check_r3:
        mine = {tuple(r[f] for f in L.KEY_FIELDS[kind]): r['action'] for r in rungs.get('R3' if sizes == [None] else f'R3_W{full}', ())}
        check = dict(compared=0, disagreements=0, examples=[])
        for r in check_r3:
            index = tuple(r[f] for f in L.KEY_FIELDS[kind])
            if index not in mine:continue
            check['compared'] += 1
            if mine[index] != r['action']:
                check['disagreements'] += 1
                if len(check['examples']) < EXAMPLES:check['examples'].append(dict(root=list(index), mine=mine[index], theirs=r['action']))
    plain = lambda c: dict(sorted(c.items()))
    summary = dict(schema=SCHEMA, kind=kind, sampled=plain(counts), true=plain(true_counts) if with_true else None, roots=len(sampled),
                   true_roots=sum(1 for i in sampled if i in truths), worlds=full, prefixes=prefixes,
                   weighted=weights is not None, true_weighted=true_weights is not None,
                   rungs={name: dict(s, undecided=plain(s['undecided']), guard_tripped=plain(s['guard_tripped'])) for name, s in stats.items()},
                   r3_check=check,
                   diagnostics={str(size or full): summarize_diag([r for r in diag if r['prefix'] == size]) for size in sizes} if with_true else None,
                   diagnostics_skipped={f'{size or full}:{reason}': n for (size, reason), n in sorted(diag_reasons.items(), key=str)})
    return dict(rungs), diag, summary


def read_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def run_rungs(args):
    if (args.kind == 'self') != (args.manifest is not None):raise SystemExit('--manifest is required for self roots and refused for Foul Play roots')
    if args.true_weights and not args.true:raise SystemExit('--true-weights needs --true')
    manifest = L.load_manifest(args.manifest) if args.manifest else None
    weights = rr.load_world_weights(args.weights) if args.weights else None
    true_weights = rr.load_world_weights(args.true_weights) if args.true_weights else None
    rungs, diag, summary = build(L.read_rows(args.sampled), args.kind, L.read_rows(args.true) if args.true else None, weights, true_weights,
                                 manifest, args.prefixes, read_jsonl(args.check_r3) if args.check_r3 else None)
    directory = Path(args.out);directory.mkdir(parents=True, exist_ok=False)
    for name, rows in rungs.items():(directory / f'rung-{name}.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
    (directory / 'diag.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in diag))
    (directory / 'summary.json').write_text(json.dumps(dict(summary, arguments=vars(args)), indent=1) + '\n')
    print(json.dumps({k: summary[k] for k in ('kind', 'roots', 'true_roots', 'worlds', 'r3_check')}
                     | {'decided': {n: s['decided'] for n, s in summary['rungs'].items()}}, sort_keys=True))
    return 1 if summary['r3_check'] and summary['r3_check']['disagreements'] else 0


def pairs(bank_dir, rungs, bases, guards='on'):
    """mc_ladder_score per rung and kind, and each rung minus each base on the roots both decided (module docstring)."""
    from gpu_engine import mc_ladder_score as S
    bank = S.load_bank(bank_dir);r0 = S.r0_choices(bank);records = {'R0': S.as_records(r0)}
    for name, path in rungs:
        if name in records:raise ValueError(f'rung {name} given twice')
        records[name] = S.load_rung(path, guards)
    kinds = sorted({k[0] for k in bank})

    def q_of(key, record):
        if record is None or record['action'] is None:return None
        row = bank[key];q = S.truth(row);a = S.policy_action(row) if record['action'] == 'policy' else record['action']
        return q.get(a) if a in q and not math.isnan(q[a]) else None

    report = dict(schema=SCHEMA + ':pairs', bank=str(bank_dir), guards=guards, rungs={}, pairs={})
    for name, rec in records.items():report['rungs'][name] = {kind: S.score(bank, rec, r0, kind) for kind in kinds}
    for base in bases:
        if base not in records:raise ValueError(f'base {base} is not a rung')
        for name in records:
            if name == base:continue
            for kind in kinds:
                d, cl = [], []
                for key in (k for k in bank if k[0] == kind):
                    qx, qy = q_of(key, records[base].get(key)), q_of(key, records[name].get(key))
                    if qx is None or qy is None:continue
                    d.append(qy - qx);cl.append(bank[key]['seed'])
                report['pairs'][f'{name} - {base}/{kind}'] = S.cluster_ci(d, cl)
    return report


def run_pairs(args):
    specs = []
    for spec in args.rung:
        name, _, path = spec.partition('=')
        if not name or not path:raise SystemExit(f'--rung NAME=PATH required: {spec!r}')
        specs.append((name, path))
    report = pairs(args.bank, specs, args.base, args.guards)
    Path(args.out).write_text(json.dumps(report, indent=1) + '\n')
    for k, v in report['pairs'].items():print(k, v['roots'], v['mean'], v['ci90'])
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='mode', required=True)
    r = sub.add_parser('rungs');r.add_argument('--kind', choices=sorted(REPLY), required=True)
    r.add_argument('--sampled', nargs='+', required=True);r.add_argument('--true', nargs='+')
    r.add_argument('--weights');r.add_argument('--true-weights');r.add_argument('--manifest')
    r.add_argument('--prefixes', nargs='+', type=int);r.add_argument('--check-r3');r.add_argument('--out', required=True)
    p = sub.add_parser('pairs');p.add_argument('bank');p.add_argument('--rung', action='append', default=[])
    p.add_argument('--base', action='append', default=[]);p.add_argument('--guards', choices=('on', 'policy', 'off'), default='on')
    p.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    return run_rungs(args) if args.mode == 'rungs' else run_pairs(args)


if __name__ == '__main__':
    raise SystemExit(main())
