"""Champion evidence on the MC ladder's self roots: the opponent model's likelihood of the opponent's earlier public choices under
each candidate set of its Pokemon, for the reweighted-worlds rungs (gpu_engine.mc_ladder_worlds R3rw on self roots).

usage: python -m gpu_engine.mc_ladder_evidence BANK_DIR OUT --manifest SELF_MANIFEST.jsonl --collection DIR --checkpoint CHAMPION.pt
           [--extra-sets ROWS.jsonl ...] [--worker W --workers N] [--limit N]

On self roots the champion is the true opponent model (mc_root_labeller.self_root: champion self-play, both seats sampling). Each
of the manifest's ok roots (ordinal, seed, ours, target; mc_ladder_captures) is replayed exactly as self_root plays it -- the
same env configuration (the collection's base_config, as mc_ladder_captures takes it), the champion's probabilities on
env.observe of every requested seat, the same deterministic draws -- to our target-th decision. At each opponent decision that is
not a forced switch, fp_reply_reweight.measure scores the opponent's sampled choice under every candidate set of its active
Pokemon (our seat's belief engine's candidates, plus --extra-sets: every set a decided root's world drew in the re-pricer's rows,
as fp_reply_evidence export scores them), with the champion in place of pi_FP: log q(a_t | v) is the log-softmax of the
champion's policy logits over the legal mask (mc_root_labeller.Model's masking) on env.observe(opponent) through the substituted
request, -inf for an illegal or unmapped label. Evidence counts only the choices made public that step
(fp_reply_reweight.public_choice), summed per Pokemon over its earlier rounds, as fp_reply_reweight.replay_game sums it.

At the root, the replayed state must be the bank row's (mc_ladder_captures.root_ok: the same legal actions, the champion's policy
within 1e-6 and its value within 1e-5). A root that passes writes one row {ordinal, decision_index: target, round, mons}
(fp_reply_evidence.snapshot: only Pokemon with evidence), fp_reply_evidence's evidence schema, so `fp_reply_evidence weights`
joins it to the re-pricer's self rows (keyed by the capture ordinal and decision index) unchanged. A root that fails, ends before
its target or raises writes no evidence row (its worlds then fail the join as no_evidence_row) and is counted in
shard-<worker>.json by status.
"""
import argparse
import collections
import glob
import json
import time
from pathlib import Path

import numpy as np

from gpu_engine import fp_reply_evidence as ev
from gpu_engine import fp_reply_reweight as rw

SCHEMA = 'mc-ladder-self-evidence.v1'


def champion_scorer(champion, torch):
    """scorer(observations, labels) -> log q per observation under the champion (mc_root_labeller.Model) for measure()."""
    from pokezero.neural_policy import observation_windows_to_torch

    def scorer(observations, labels):
        t = observation_windows_to_torch([[o] for o in observations], window_size=champion.window, device='cpu')
        legal = np.array([np.asarray(o.legal_action_mask, bool) for o in observations])
        with torch.no_grad():
            out = champion.model(categorical_ids=t['categorical_ids'], numeric_features=t['numeric_features'],
                                 token_type_ids=t['token_type_ids'], attention_mask=t['attention_mask'], history_mask=t['history_mask'])
            logits = out.policy_logits.masked_fill(~torch.as_tensor(legal), -1e9)
            logq = torch.log_softmax(logits, dim=-1).numpy().astype(float)
        rows = np.arange(len(observations));label = np.array([0 if y is None else int(y) for y in labels])
        result = logq[rows, label]
        result[np.array([y is None or not legal[i, y] for i, y in enumerate(labels)], bool)] = -np.inf
        return result
    return scorer


def replay_root(env, champion, scorer, entry, bank_row, dex, tables, extra):
    """(evidence row or None, root check or None, status, check counts) for one self root (module docstring)."""
    from gpu_engine.mc_root_labeller import draw, legal_of, sample
    seed, ours, target = int(entry['seed']), entry['ours'], int(entry['target']);opp = 'p2' if ours == 'p1' else 'p1'
    game = dict(fp=opp, ours=ours, seed=seed, ordinal=int(entry['ordinal']))
    env.reset(seed=seed);count = ply = 0;checks = collections.Counter();misaligned = set();first = None
    evidence = collections.defaultdict(dict);evidence_n = collections.Counter();mons, decisions, calib = [], [], []
    while env.terminal() is None:
        requested = list(env.requested_players())
        if ours in requested and count == target:
            own = env.observe(ours);probs, values = champion.evaluate([own])
            root = dict(actions_equal=[int(a) for a in np.flatnonzero(legal_of(own))] == bank_row['actions'],
                        policy_max_abs=float(np.max(np.abs(np.asarray(probs[0]) - np.asarray(bank_row['policy'])))),
                        value_abs=None if bank_row.get('root_value') is None else abs(float(values[0]) - float(bank_row['root_value'])))
            from gpu_engine.mc_ladder_captures import root_ok
            if not root_ok(root):return None, root, 'root_mismatch', checks
            row = dict(ordinal=game['ordinal'], decision_index=target, round=ply, mons=ev.snapshot(evidence, evidence_n, misaligned))
            return row, root, 'ok', checks
        obs = [env.observe(p) for p in requested];probs = champion.probabilities(obs)
        actions = {p: sample(probs[i], legal_of(o), draw(seed, 10**6, ply, i)) for i, (p, o) in enumerate(zip(requested, obs))}
        pending = None
        if opp in requested:
            request = env._latest_requests.get(opp)
            if first is None:first = env._first_requests.get(opp) or request
            if request.get('forceSwitch'):checks['forced_skipped'] += 1
            else:
                pending = rw.measure(env, None, game, dex, None, checks, request, first, actions[opp], ply, evidence, evidence_n, mons,
                                     decisions, calib, tables, misaligned, None, extra, scorer)
        count += ours in requested;before = len(env._lines);env.step(actions);ply += 1
        if pending is not None:   # fp_reply_reweight.replay_game's accumulation
            mon, keys, logq, act, move, aligned, skipped = pending
            observed = rw.public_choice(env._lines[before:], opp, act, move);decisions[-1]['observed'] = observed
            if observed:
                evidence_n[mon] += 1
                if not aligned:misaligned.add(mon)
                for k, lq in zip(keys, logq):evidence[mon][k] = evidence[mon].get(k, 0.) + lq
                for k in skipped:evidence[mon][k] = float('nan')
    return None, None, 'ended_before_root', checks


def load_bank(bank_dir):
    """{(seed, ours, target): bank row} of the bank's self roots."""
    out = {}
    for path in sorted(glob.glob(str(Path(bank_dir) / 'roots-self-*.jsonl'))):
        for line in open(path):
            if line.strip():r = json.loads(line);out[int(r['seed']), r['ours'], int(r['target'])] = r
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bank');parser.add_argument('out');parser.add_argument('--manifest', required=True)
    parser.add_argument('--collection', required=True);parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--extra-sets', nargs='+', help="the re-pricer's self rows: score their worlds' sets too")
    parser.add_argument('--worker', type=int, default=0);parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args(argv)
    if not 0 <= args.worker < args.workers:raise SystemExit('--worker W --workers N with 0 <= W < N required')
    import torch
    torch.set_num_threads(1)   # mc_ladder_captures' torch settings exactly: the replayed root's policy matches the bank's within 1e-6
    from pokezero.dex import load_showdown_dex_cached
    from pokezero.neural_policy import checkpoint_file_sha256
    from gpu_engine.mc_root_labeller import Model, make_env
    started = time.time();out = Path(args.out);out.mkdir(parents=True, exist_ok=True)
    rows_path, summary_path = out / f'evidence-{args.worker}.jsonl', out / f'shard-{args.worker}.json'
    if rows_path.exists() or summary_path.exists():raise SystemExit(f'{rows_path} or {summary_path} exists')
    bank = load_bank(args.bank)
    entries = [json.loads(l) for l in open(args.manifest) if l.strip()]
    entries = [e for e in entries if e.get('status') == 'ok' and int(e['ordinal']) % args.workers == args.worker]
    if args.limit:entries = entries[:args.limit]
    collection = Path(args.collection);first = sorted(glob.glob(str(collection / 'requests' / '*.json')))[0]
    base = json.loads(Path(first).read_bytes())['base_config']
    champion = Model(args.checkpoint, torch);scorer = champion_scorer(champion, torch)
    dex = load_showdown_dex_cached(base['showdown_root']);tables = rw.generator_tables(base['showdown_root'], base['node_binary'])
    extras = ev.extra_sets(ev._jsonl(sorted(args.extra_sets))) if args.extra_sets else {}
    env = make_env(base, champion.result, caching=False)
    statuses, checks, examples, written = collections.Counter(), collections.Counter(), {}, 0
    stream = open(rows_path, 'x')   # rows as they come (flushed), so an interrupted worker keeps its finished roots
    try:
        for n, entry in enumerate(entries, 1):
            key = (int(entry['seed']), entry['ours'], int(entry['target']))
            sets_of = extras.get(int(entry['ordinal']), {})
            extra = (lambda mon, sets_of=sets_of: list(sets_of.get(ev.species_key(mon), {}).values())) if args.extra_sets else None
            try:
                if key not in bank:raise KeyError(f'no bank row for {key}')
                row, root, status, game_checks = replay_root(env, champion, scorer, entry, bank[key], dex, tables, extra)
            except Exception as error:   # counted with an example; the env is rebuilt
                row, root, status, game_checks = None, None, f'error {type(error).__name__}', collections.Counter()
                examples.setdefault(status, dict(ordinal=entry['ordinal'], detail=str(error)[:500]))
                env.close();env = make_env(base, champion.result, caching=False)
            statuses[status] += 1;checks.update(game_checks)
            if status == 'root_mismatch':examples.setdefault(status, dict(ordinal=entry['ordinal'], root=root))
            if row is not None:stream.write(json.dumps(row) + '\n');stream.flush();written += 1
            if n % 10 == 0:print(json.dumps(dict(event='PROGRESS', worker=args.worker, roots=n, ok=statuses['ok'],
                                                 seconds=round(time.time() - started, 1))), flush=True)
    finally:
        env.close();stream.close()
    summary = dict(schema=SCHEMA, bank=str(args.bank), manifest=str(args.manifest), checkpoint=str(args.checkpoint),
                   checkpoint_sha256=checkpoint_file_sha256(args.checkpoint), worker=args.worker, workers=args.workers,
                   roots=len(entries), rows=written, statuses=dict(statuses), checks=dict(checks), examples=examples,
                   extra_sets=sorted(str(Path(p).resolve()) for p in args.extra_sets or ()), seconds=round(time.time() - started, 1))
    summary_path.write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(dict(event='SHARD_DONE', **{k: summary[k] for k in ('worker', 'roots', 'rows', 'statuses', 'seconds')})), flush=True)
    return 0 if statuses['ok'] == len(entries) else 1


if __name__ == '__main__':
    raise SystemExit(main())
