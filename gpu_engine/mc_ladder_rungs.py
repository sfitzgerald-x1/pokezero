"""Monte Carlo-truth ladder: each rung's choice at the gate bank's roots, from the re-pricer's rows, as mc_ladder_score rung files.
CPU only: host arithmetic on the rows through depthn_rr_score's and search_hedge's own operators; no engine, model or device.

Input: the rows of scripts/depthn_rr_reprice.py runs (--targets on the bank's roots; one row per sent decision) carrying each
world's depth-1 prices (worlds[i].depth1, depth1.refused_at_depth1) and, under --reply-archive, each requested world's
reply-model logits (worlds[i].reply_peer_logits). Five runs, one rung set each:
- A: Foul Play roots, --true-worlds (true_private), --reply-archive pi_FP and --peer-view-b -> R1, R2;
- B: self roots, --true-worlds (true_private), the live configuration -> R1, R2;
- C: Foul Play roots, --reply-archive pi_FP and --peer-view-b, sampled worlds -> R3, R3prime_ng, R4b;
- D: Foul Play roots, the live configuration -> R4, R5, R5_ng;
- E: self roots, the live configuration -> R3, R3prime_ng, R4, R5, R5_ng.

Rungs. Greedy is the argmax of root_logits over root_legal, lower index on ties (depthn_rr_score.argmax).
- R1 (runs A, B): averaging over the depth-1 grids of the W true worlds (the true state W times, each world its own chance
  draws), with the reply model's logits (run A) or the champion's root_peer_logits (run B) as each world's reply prior.
- R2 (runs A, B): R1 on world 0 alone, one chance draw of the true state.
- R3: averaging over the worlds' depth-1 grids with the reply model's logits as each requested world's reply prior
  (depthn_rr_score.model_worlds then search_hedge.averaging: br_fp at depth 1; a waiting world keeps column 0). The bank's self
  roots use the champion as the opponent model, so on self roots (run E) R3 is the champion's depth-1 averaging: the same
  choice as R4, written under both names.
- R3prime_ng: depthn_rr_score.noise_gated with k = 2 on R3's per-world depth-1 q: R3's choice only when it differs from greedy
  and the mean per-world gap Q_w(a*) - Q_w(greedy) exceeds 2 standard errors; otherwise greedy. On self roots it gates the
  champion's depth-1 choice.
- R4 (runs D, E) and R4b (run C): averaging over the depth-1 grids with the champion's root_peer_logits (R4b: the champion on
  view b, the same worlds as R3).
- R5: the replay's own action_index (the live code's DN2W16 averaging at depth 2). The chooser recomputes averaging on the
  rows' depth-2 grids (depthn_rr_score.recompute) and counts every COMPLETE root where its action or q differs from the replay's.
- R5_ng: noise_gated k = 2 on the depth-2 per-world q of that recomputation. Greedy is the root logits' argmax, as the r4b
  screen's NOISE_GATED_K2 operator takes it (depthn_root_operator.noise_gated_k2), not the live actor's greedy_action_index that
  depthn_rr_score's own noise_gated:k=2 reads; they differ only where those two greedies differ.

Identity. Foul Play rows (runs A, C, D): key (ordinal, target = decision_index), and the row's cell seed is echoed for the
scorer's check. Self rows (runs B, E) map through the self-capture manifest (--manifest: rows {ordinal, seed, ours, target}): the
key is (seed, ours, target), and the row's decision_index must equal target, its owner ours and its seed (when present) the
manifest's seed, and the manifest row's status must be ok (manifest_<status> otherwise, a missing status included: a capture
whose replayed root is not the bank's). A row whose target_kind names the other kind, or that fails those checks, is counted
by reason and written to no rung. Runs A and B take only rows of a --true-worlds --true-kind true_private replay (the row's
true_worlds record with requested_kind true_private; not_true_worlds otherwise), and runs C, D and E refuse a row that carries
one (true_worlds_row); both are counted as identity mismatches. Two rows with one key are an error.

Each rung row: the key, action (None when the rung cannot decide the root, with `reason`), greedy_action, guard_tripped
(with guard_reason when true) and actions (the root's legal action indices, echoed for the scorer's identity check).
Guards: depthn_rr_score.guard_choices on the rung's own choice, the live actor's rule: the own-recharge guard, or the row's
effect_guard (b), or (c) on this choice. Reply-model rungs (run C's R3 and R3prime_ng) keep a union record's reply arm, as
the live br_fp actor does; champion-prior rungs drop it (the record's own b is the champion arm's). For R5 this is the live
fallback exactly (effect_guard.tripped or the recharge guard). A decided root without a guard record (a replay that did not
end COMPLETE, whose depth-1 rungs still price) fails closed: guard_tripped true, guard_reason no_guard_record. An undecided
row never claims a trip, so mc_ladder_score counts it not_decided under every --guards mode.

Reasons for action None:
- depth1_missing[:cause]: no depth-1 frame (the root was refused before pricing; cause is the row's root.refusal_cause);
- refused_at_depth1: D3 itself refuses a block (depth1.refused_at_depth1);
- row_schema_old: a row written before the depth-1 refusal flags (depth1.refused_at_depth1);
- nonfinite: averaging's or a world's q is not finite (a backstop: WorldPrices already refuses a NaN cell as world_invalid);
- root_prior_missing / root_prior_invalid: no usable root_logits and root_legal (the row's root, or for a root decide never saw,
  the priced batch entry's own under depth1.root_logits / root_legal);
- world_missing, world_unpriced, world_invalid, unpriced_cell, averaging_error: a world or a needed cell has no price;
- reply_prior_missing: a requested world has no reply_peer_logits (only roots that reached the reply model's decide have them);
- replay_<status>[:cause]: R5 and R5_ng on a replay that did not end COMPLETE; replay_action_invalid;
- one_world: noise gating needs at least two worlds;
- true_worlds_failed[:cause], true_worlds_not_sampled[:cause] (runs A and B): the true worlds failed closed (a problem or error,
  fewer built than requested, or a kind other than true_private or its waiting-opponent fallback true), or were never sampled.
R1 and R2 can decide different roots (R2 needs only world 0's prices): compare them on the roots both decided.

usage: python -m gpu_engine.mc_ladder_rungs --rows ROWS.jsonl [ROWS.jsonl ...] --run {A,B,C,D,E} [--manifest MANIFEST.jsonl] --out NEW_DIR
--manifest is required for runs B and E (self roots) and refused for A, C and D. Writes NEW_DIR/rung-<NAME>.jsonl per rung and NEW_DIR/summary.json
(rows read, identity mismatches by reason, per rung: decided, undecided by reason, guard trips by reason; R5 against the
replay). Exit 1 when R5's recomputation disagrees with any COMPLETE replay or fails on one (recompute_failed), else 0.
"""
import argparse
import collections
import json
import math
from pathlib import Path

from scripts import depthn_rr_score as rr   # first: it puts src/ on sys.path for pokezero
from gpu_engine import search_hedge as hedge

SCHEMA = 'mc-ladder-rungs.v1'
NOISE_K = 2
# run: (root kind, {rung: (depth, reply model's logits, noise gated, worlds: None all or n the first n)}). Runs A and B are the
# true-world replays (--true-worlds, --true-kind true_private): R1 averages the W true worlds (W chance draws of the true
# state) at depth 1; R2 is world 0 alone (one draw).
RUNS = dict(
    A=('foulplay', dict(R1=(1, True, False, None), R2=(1, True, False, 1))),
    B=('self', dict(R1=(1, False, False, None), R2=(1, False, False, 1))),
    C=('foulplay', dict(R3=(1, True, False, None), R3prime_ng=(1, True, True, None), R4b=(1, False, False, None))),
    D=('foulplay', dict(R4=(1, False, False, None), R5=(2, False, False, None), R5_ng=(2, False, True, None))),
    E=('self', dict(R3=(1, False, False, None), R3prime_ng=(1, False, True, None), R4=(1, False, False, None), R5=(2, False, False, None),
                    R5_ng=(2, False, True, None))))
TRUE_RUNS = frozenset('AB')   # the true-world runs: rows of a --true-worlds --true-kind true_private replay only
TRUE_KIND = 'true_private'
KEY_FIELDS = dict(foulplay=('kind', 'ordinal', 'target'), self=('kind', 'seed', 'ours', 'target'))
EXAMPLES = 20   # disagreeing roots listed in the summary


def load_manifest(path):
    """{ordinal: row} from the self-capture manifest (mc_ladder_captures: rows {ordinal, seed, ours, target[, status]})."""
    out = {}
    for number, line in enumerate(open(path), 1):
        if not line.strip():continue
        row = json.loads(line)
        if (type(row.get('ordinal')) is not int or type(row.get('seed')) is not int or type(row.get('target')) is not int
                or row.get('ours') not in ('p1', 'p2')):
            raise ValueError(f'{path}:{number}: integer ordinal, seed and target, and ours p1 or p2, required')
        if row['ordinal'] in out:raise ValueError(f'{path}:{number}: ordinal {row["ordinal"]} repeated')
        out[row['ordinal']] = row
    return out


def identity(raw, kind, manifest):
    """(the rung row's key fields, None), or (None, the mismatch reason)."""
    if raw.get('target_kind') not in (None, kind):return None, 'kind_mismatch'
    ordinal, index = raw.get('ordinal'), raw.get('decision_index')
    if type(ordinal) is not int or type(index) is not int:return None, 'address_missing'
    if kind == 'foulplay':
        seed = raw.get('seed')
        return dict(kind=kind, ordinal=ordinal, target=index, **({'seed': seed} if type(seed) is int else {})), None
    entry = manifest.get(ordinal)
    if entry is None:return None, 'not_in_manifest'
    if entry.get('status') != 'ok':return None, f'manifest_{entry.get("status")}'   # a missing status too (manifest_None)   # a capture whose replayed root is not the bank's
    if index != entry['target']:return None, 'target_mismatch'
    if raw.get('owner') not in (None, entry['ours']):return None, 'seat_mismatch'
    if raw.get('seed') not in (None, entry['seed']):return None, 'seed_mismatch'
    return dict(kind=kind, seed=entry['seed'], ours=entry['ours'], target=entry['target']), None


def true_worlds_check(raw, run):
    """(identity mismatch or None, the root's undecided reason or None): runs A and B take only rows of a --true-kind true_private
    replay and decide a root only when its true worlds were all built; the other runs refuse a true-world row (module docstring)."""
    record = raw.get('true_worlds')
    if run not in TRUE_RUNS:return ('true_worlds_row' if record else None), None
    if type(record) is not dict or record.get('requested_kind') != TRUE_KIND:return 'not_true_worlds', None
    calls, worlds = record.get('sample_calls'), record.get('worlds')
    if type(calls) is not int or calls < 1:return None, 'true_worlds_not_sampled'
    played = record.get('kind') == TRUE_KIND or (record.get('kind') == 'true' and bool(record.get('fallback')))   # the opponent waits: true
    if (record.get('problem') is not None or record.get('error') is not None or not played or type(worlds) is not int or worlds < 1
            or record.get('built') != worlds):
        return None, 'true_worlds_failed'
    return None, None


def root_prior(d):
    """(root_legal, greedy) of a normalized row; raises Unscorable when the row has no usable root prior."""
    legal, logits = d['root_legal'], d['root_logits']
    if legal is None or logits is None:raise rr.Unscorable('root_prior_missing')
    try:
        if len(legal) != 9 or any(type(x) is not bool for x in legal) or not any(legal):raise ValueError
        rr.original._probabilities(logits, tuple(legal))   # the live prior's own validation: nine finite logits
    except (TypeError, ValueError):raise rr.Unscorable('root_prior_invalid') from None
    return tuple(legal), rr.argmax(logits, [a for a in range(9) if legal[a]])


def average(worlds, legal):
    """(averaging's choice a*, per-world q, averaging's q) on WorldPrices; raises Unscorable (nonfinite: as decide's
    AVERAGING_INVALID_RESULT, a non-finite q never chooses)."""
    try:
        gs = hedge.grids(worlds, legal)
        if gs is None:raise rr.Unscorable('unpriced_cell')
        choice = hedge.averaging(worlds, legal)
    except ValueError:raise rr.Unscorable('averaging_error') from None
    q = rr.per_world_q(gs)
    if any(v is not None and not math.isfinite(float(v)) for v in choice.q) or any(
            v is not None and not math.isfinite(float(v)) for world in q for v in (world.values() if type(world) is dict else world)):
        raise rr.Unscorable('nonfinite')
    return hedge.top_action(choice.strategy), q, choice.q


def averaged(raw, d, depth, reply, first=None):
    """(a*, per-world q, q) of averaging on the row's depth-1 or depth-2 worlds, the reply model's logits standing in for each
    requested world's reply prior when `reply`; raises Unscorable with the reason (module docstring)."""
    if depth == 2:
        if d['replay_status'] == 'COMPLETE':root_prior(d)   # a COMPLETE replay without a usable root prior: the depth-1 rungs' reason
        try:base = rr.recompute(d)   # the replay's own worlds and root, as depthn_rr_score's parity recomputes them
        except ValueError:raise rr.Unscorable('averaging_error') from None
        return hedge.top_action(base.average.strategy), rr.per_world_q(base.grids), base.average.q
    info = raw.get('depth1')
    if type(info) is not dict:raise rr.Unscorable('depth1_missing')
    if 'refused_at_depth1' not in info:raise rr.Unscorable('row_schema_old')   # a row written before the depth-1 refusal flags
    if any(info['refused_at_depth1']):raise rr.Unscorable('refused_at_depth1')
    legal, _ = root_prior(d)
    worlds = raw.get('worlds') if type(raw.get('worlds')) is list else []
    worlds = worlds if first is None else worlds[:first]   # R2: the first world (one chance draw of the true state)
    prices = rr.world_prices(rr.normalize(dict(worlds=[w.get('depth1') if type(w) is dict else None for w in worlds])))
    if reply:prices = rr.model_worlds(prices, rr.reply_logits_of(d, prices, None))
    return average(prices, legal)


def guard(d, action, greedy, reply):
    """(guard_tripped, guard_reason) of the live actor's guards on a rung's own choice (module docstring)."""
    record = d['effect_guard']
    if type(record) is not dict or 'b' not in record:return True, 'no_guard_record'
    if not reply:record = {k: v for k, v in record.items() if k != 'union'}   # the champion arm alone
    causes = rr.guard_choices({'rung': action}, greedy, d['own_moves'], record)[1]['rung']
    return (True, ','.join(causes)) if causes else (False, None)


def choose(raw, run):
    """(the normalized row, {rung: rung row without the key}, and for runs with R5 the depth-2 recomputation: ((a*, per-world q,
    q), None) or (None, the failure reason))."""
    d = rr.normalize(raw);cause = d['refusal_cause'];ops = {};prior_reason = None
    depth1 = raw.get('depth1') if type(raw.get('depth1')) is dict else {}
    if d['root_logits'] is None and d['root_legal'] is None and depth1.get('root_logits') is not None:   # a root decide never saw
        d = dict(d, root_logits=depth1['root_logits'], root_legal=depth1['root_legal'])   # the priced batch entry's own root prior
    try:legal, greedy = root_prior(d)
    except rr.Unscorable as error:legal = greedy = None;prior_reason = error.reason

    def op(depth, reply, first=None):
        if (depth, reply, first) not in ops:
            try:ops[depth, reply, first] = averaged(raw, d, depth, reply, first), None
            except rr.Unscorable as error:
                reason = error.reason
                if cause and (reason == 'depth1_missing' or reason.startswith('replay_')):reason += f':{cause}'
                ops[depth, reply, first] = None, reason
        return ops[depth, reply, first]

    out = {}
    for name, (depth, reply, gated, first) in RUNS[run][1].items():
        if depth == 2 and not gated:   # R5: the replay's own action
            action, reason = d['replay_action'], None
            if d['replay_status'] != 'COMPLETE':action, reason = None, f"replay_{d['replay_status']}" + (f':{cause}' if cause else '')
            elif legal is None:action, reason = None, prior_reason
            elif type(action) is not int or not 0 <= action < 9 or not legal[action]:action, reason = None, 'replay_action_invalid'
        else:
            result, reason = op(depth, reply, first);action = None
            if result is not None:
                chosen, qw, _ = result
                if not gated:action = chosen
                elif len(qw) < 2:reason = 'one_world'
                else:action = rr.noise_gated(qw, chosen, greedy, NOISE_K)
        row = dict(action=action, greedy_action=greedy, guard_tripped=False, actions=None if legal is None else [a for a in range(9) if legal[a]])
        if action is None:row['reason'] = reason
        else:
            tripped, why = guard(d, action, greedy, reply)
            if tripped:row.update(guard_tripped=True, guard_reason=why)
        out[name] = row
    recomputed = op(2, False) if 'R5' in RUNS[run][1] else None
    return d, out, recomputed


def build(rows, run, manifest=None):
    """({rung: [rung rows]}, summary) over re-pricer rows (module docstring)."""
    kind, rungs = RUNS[run];manifest = manifest or {}
    out = {name: [] for name in rungs};seen = set();mismatches = collections.Counter();statuses = collections.Counter();count = 0
    stats = {name: dict(rows=0, decided=0, undecided=collections.Counter(), guard_tripped=collections.Counter()) for name in rungs}
    parity = dict(compared=0, action_disagreements=0, q_disagreements=0, recompute_failed=collections.Counter(), examples=[])
    for raw in rows:
        count += 1;key, problem = identity(raw, kind, manifest);truth = None
        if key is not None:problem, truth = true_worlds_check(raw, run)
        if problem is not None:mismatches[problem] += 1;continue
        index = tuple(key[f] for f in KEY_FIELDS[kind])
        if index in seen:raise ValueError(f'two rows for root {index}')
        seen.add(index)
        if kind == 'self':statuses[str(manifest[raw['ordinal']].get('status'))] += 1
        d, chosen, recomputed = choose(raw, run)
        if truth is not None:   # a true-world root whose worlds were not all built: no rung decides it
            truth += f":{d['refusal_cause']}" if d['refusal_cause'] else ''
            chosen = {name: dict({k: v for k, v in row.items() if k not in ('reason', 'guard_reason')}, action=None, guard_tripped=False,
                                 reason=truth) for name, row in chosen.items()}
        for name, row in chosen.items():
            out[name].append(dict(key, **row));s = stats[name];s['rows'] += 1
            if row['action'] is None:s['undecided'][row['reason']] += 1
            else:
                s['decided'] += 1
                if row['guard_tripped']:s['guard_tripped'][row['guard_reason']] += 1
        if recomputed is not None and d['replay_status'] == 'COMPLETE':
            parity['compared'] += 1;result, reason = recomputed;wrong = []
            if result is None:parity['recompute_failed'][reason] += 1;wrong.append(reason)
            else:
                if result[0] != d['replay_action']:parity['action_disagreements'] += 1;wrong.append('action')
                if not rr.q_json_equal(result[2], d['replay_q']):parity['q_disagreements'] += 1;wrong.append('q_values')
            if wrong and len(parity['examples']) < EXAMPLES:parity['examples'].append(dict(key, disagreement=wrong))
    plain = lambda c: dict(sorted(c.items()))
    summary = dict(schema=SCHEMA, run=run, kind=kind, rows=count, mapped=len(seen), identity_mismatches=plain(mismatches),
                   rungs={name: dict(s, undecided=plain(s['undecided']), guard_tripped=plain(s['guard_tripped'])) for name, s in stats.items()},
                   r5_vs_replay=dict(parity, recompute_failed=plain(parity['recompute_failed'])) if 'R5' in rungs else None)
    if kind == 'self':summary['manifest_status'] = plain(statuses)
    return out, summary


def read_rows(paths):
    for path in paths:
        with open(path, 'rb') as stream:
            for raw in stream:
                if raw.strip():yield json.loads(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rows', nargs='+', required=True, help="the re-pricer's rows-<shard>.jsonl files of one run")
    parser.add_argument('--run', choices=sorted(RUNS), required=True)
    parser.add_argument('--manifest', help='runs B and E: the self-capture manifest.jsonl')
    parser.add_argument('--out', required=True, help='a new directory')
    args = parser.parse_args(argv)
    if (RUNS[args.run][0] == 'self') != (args.manifest is not None):parser.error('--manifest is required for runs B and E (self roots) and refused for A, C and D')
    out, summary = build(read_rows(args.rows), args.run, load_manifest(args.manifest) if args.manifest else None)
    directory = Path(args.out);directory.mkdir(parents=True, exist_ok=False)
    for name, rows in out.items():(directory / f'rung-{name}.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
    (directory / 'summary.json').write_text(json.dumps(dict(summary, arguments=vars(args)), indent=1) + '\n')
    print(json.dumps(summary, sort_keys=True))
    parity = summary['r5_vs_replay']
    return 1 if parity and (parity['action_disagreements'] or parity['q_disagreements'] or parity['recompute_failed']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
