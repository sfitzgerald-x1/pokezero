#!/usr/bin/env python3
"""The private-reveal redaction's effect on the peer's reply prior and on averaging's play: CPU only, a measurement.

Inputs: two re-pricer runs (scripts/depthn_rr_reprice.py rows-*.jsonl) of the same selection (cells or sample, seeds, tree):
--baseline without --redact-private-reveals, --redacted with it. Rows join on (ordinal, decision_index), their worlds on
`world` (the constructor's world index, block order 2k + j); a joined pair whose battle_id or owner differs is not
compared (counted and listed). A root's redaction is applied when the redacted row's private_reveals.redacted is not empty
(constructor_worlds.private_reveal_record); every other joined root is the control.

Per world (both sides priced, the same opponent_requested and kept; a waiting world, opponent_requested false on both, has
the single reply 0 on both and is only counted): the turn-1 reply prior pi = softmax(root_peer_logits) over kept
(v5_gumbel._probabilities, averaging's own prior), KL(baseline || redacted) in nats and whether its top reply (argmax,
lowest index on ties) flips.
Per root (both replays COMPLETE): averaging's action recomputed with the BASELINE's G grids and the redacted run's turn-1
reply priors (search_hedge.averaging over WorldPrices(requested, kept, redacted root_peer_logits, baseline grid): the
scorer's averaging, summed as the live server sums it), a_flag, against the baseline's own averaging, a_base. The PLAYED
action of each side applies the full live guard (scripts/depthn_rr_score.guard_choices, the live actor's functions): greedy
(the live run's greedy action) where the own-recharge guard (an override INTO Hyper Beam, Blast Burn, Hydro Cannon or
Frenzy Plant that greedy did not choose) or UNREPRESENTED_EFFECT ((b) at the root, (c) on the choice) trips; the
baseline's on a_base with the baseline record, the redacted run's on a_flag with the redacted run's record. The headline is
the played-action flip rate over the roots the baseline would search (no guard trips on a_base); beside it, on the same
roots, the raw averaging flip (a_base != a_flag). How many roots' (b) differs between the runs is reported too.
Each, overall and split applied / control. The control (no redaction at the root) must not move at all; each of these is a
determinism failure, listed and printed loudly on stderr: a world whose logits differ by a bit, whose KL is nonzero or
whose top reply flips; a world not compared on both sides alike (priced on one side only, requested or kept differing, a
world missing on either side, logits missing); a root whose replays disagree (one COMPLETE, or different refusals), whose
averaging cannot be recomputed on a decided root (any status but scored), whose (b) changes, or whose averaging or played
action flips.
Output: a summary on stdout (and the full report as JSON with --out, a new file). Exit 0, or 3 on a determinism failure.

  python scripts/depthn_private_reveal_effect.py --baseline BASE/rows-*.jsonl --redacted FLAG/rows-*.jsonl [--out REPORT.json]
"""
import argparse
import collections
import json
import math
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT), str(_ROOT / 'src')):
    if _path not in sys.path:sys.path.insert(0, _path)

from scripts import depthn_rr_score as score  # noqa: E402
from gpu_engine import search_hedge as hedge  # noqa: E402
from pokezero import depthn_effect_guard as effect_guard, v5_gumbel as original  # noqa: E402

SCHEMA = 'pokezero.depthn-private-reveal-effect.v1'
EXAMPLES = 20
GROUPS = ('all', 'applied', 'control')


def read_jsonl(paths):
    for path in paths:
        with open(path, 'rb') as stream:
            for raw in stream:
                if raw.strip():yield json.loads(raw)


def quantile(xs, q):
    """The q-quantile of xs, linear between order statistics (numpy's default); None for no value."""
    xs = sorted(xs)
    if not xs:return None
    at = q * (len(xs) - 1);lo = math.floor(at);hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (at - lo)


def prior(logits, kept):
    return original._probabilities(tuple(float(x) for x in logits), tuple(bool(x) for x in kept))


def kl(p, q):
    """KL(p || q) in nats (fsum); inf where q has no mass on a reply p has."""
    terms = []
    for a, b in zip(p, q):
        if a <= 0:continue
        if b <= 0:return math.inf
        terms.append(a * (math.log(a) - math.log(b)))
    return max(0., math.fsum(terms))


def top(p, kept):
    return max((b for b in range(len(p)) if kept[b]), key=lambda b: (p[b], -b))


def bits_equal(a, b):
    return len(a) == len(b) and all(float(x).hex() == float(y).hex() for x, y in zip(a, b))


def applied(row):
    return bool(((row.get('private_reveals') or {}).get('redacted')))


def worlds_by_index(row):
    out = {}
    for i, w in enumerate(row.get('worlds') or ()):
        if type(w) is dict:out[w.get('world', i)] = w
    return out


def world_effect(base, flag):
    """(status, details) of one joined world: 'compared' with kl, top flip and bit equality; 'waiting' (on both sides);
    'unpriced' (on both sides); or why the two sides cannot be compared (a determinism failure on a control root)."""
    priced = bool(base.get('priced')), bool(flag.get('priced'))
    if priced != (True, True):return ('unpriced' if priced == (False, False) else 'priced_mismatch'), None
    if bool(base.get('opponent_requested')) != bool(flag.get('opponent_requested')):return 'requested_mismatch', None
    if not base.get('opponent_requested'):return 'waiting', None
    kept, other = base.get('kept'), flag.get('kept')
    if type(kept) is not list or kept != other or len(kept) != 9 or not any(kept):return 'kept_mismatch', None
    a, b = base.get('root_peer_logits'), flag.get('root_peer_logits')
    if type(a) is not list or type(b) is not list or len(a) != 9 or len(b) != 9:return 'logits_missing', None
    try:p, q = prior(a, kept), prior(b, kept)
    except ValueError:return 'logits_invalid', None
    return 'compared', dict(kl=kl(p, q), flip=top(p, kept) != top(q, kept), bits_equal=bits_equal(a, b))


WORLD_OK = ('compared', 'waiting', 'unpriced')   # a control world in any other status is a determinism failure


def played(choice, greedy, own_moves, record):
    """(played action, guarded) of one side: the live guards (depthn_rr_score.guard_choices) on averaging's `choice`."""
    out, causes = score.guard_choices({'averaging': choice}, greedy, own_moves, record if type(record) is dict else None)
    return out['averaging'], causes['averaging'] is not None


def averaging_effect(base_row, flag_row):
    """(status, details) of one joined root: averaging's action on the baseline worlds (a_base) against the same with the
    redacted run's reply priors (a_flag), each side's played action under the live guards, whether the baseline would
    search the root (no guard trips on a_base) and whether the two records' (b) differ. 'undecided' when neither replay is
    COMPLETE and both refused alike; 'decision_mismatch' when the replays disagree; else why it cannot be recomputed."""
    base, flag = score.normalize(base_row), score.normalize(flag_row)
    done = base['replay_status'] == 'COMPLETE', flag['replay_status'] == 'COMPLETE'
    if done != (True, True):
        same = done == (False, False) and (base['replay_status'], base['replay_cause']) == (flag['replay_status'], flag['replay_cause'])
        return ('undecided' if same else 'decision_mismatch'), None
    legal, logits = base['root_legal'], base['root_logits']
    if type(legal) is not list or len(legal) != 9 or not any(legal) or type(logits) is not list:return 'root_missing', None
    try:worlds = score.world_prices(base)
    except score.Unscorable as error:return f'baseline_{error.reason}', None
    others = flag['worlds']
    if type(others) is not list or len(others) != len(worlds) or any(w is None for w in others):return 'redacted_worlds_missing', None
    mixed = []
    for w, o in zip(worlds, others):
        if bool(o['requested']) != bool(w.opponent_requested) or tuple(o['kept'] or ()) != tuple(w.opponent_legal) or type(o['logits']) is not list:
            return 'world_mismatch', None
        mixed.append(original.WorldPrices(w.opponent_requested, w.opponent_legal, tuple(o['logits']), w.joint_values))
    legal = tuple(legal);mine, theirs = hedge.averaging(worlds, legal), hedge.averaging(tuple(mixed), legal)
    if mine is None or theirs is None:return 'unpriced_cell', None
    a, b = hedge.top_action(mine.strategy), hedge.top_action(theirs.strategy)
    greedy, moves = base['greedy_action'], base['own_moves']
    if type(greedy) is not int or not 0 <= greedy < 9 or not legal[greedy]:return 'greedy_missing', None
    record, other = base['effect_guard'], flag['effect_guard']
    if type(record) is not dict or type(other) is not dict:return 'no_guard_record', None
    played_base, guarded = played(a, greedy, moves, record);played_flag, _ = played(b, greedy, moves, other)
    return 'scored', dict(baseline=a, mixed=b, flip=a != b, played_baseline=played_base, played_redacted=played_flag,
                          played_flip=played_base != played_flag, searched=not guarded, b_changed=bool(other.get('b')) != bool(record.get('b')))


def compare(baseline, redacted):
    """The report (module docstring) over the two runs' rows (iterables of dicts)."""
    mine = {};counts = collections.Counter()
    for row in baseline:
        counts['baseline_rows'] += 1;key = (row.get('ordinal'), row.get('decision_index'))
        if key in mine:counts['baseline_duplicates'] += 1;continue
        mine[key] = row
    seen = set();kls = {g: [] for g in GROUPS};flips = {g: collections.Counter() for g in GROUPS}
    plays = {g: collections.Counter() for g in GROUPS};statuses = collections.Counter();roots = collections.Counter()
    species = collections.Counter();mismatches = [];determinism = [];record_mismatch = []
    for row in redacted:
        counts['redacted_rows'] += 1;key = (row.get('ordinal'), row.get('decision_index'))
        base = mine.get(key)
        if base is None:counts['redacted_without_baseline'] += 1;continue
        if key in seen:counts['redacted_duplicates'] += 1;continue
        seen.add(key);counts['joined'] += 1
        if base.get('battle_id') != row.get('battle_id') or base.get('owner') != row.get('owner'):
            mismatches.append(dict(ordinal=key[0], decision_index=key[1], battle_id=[base.get('battle_id'), row.get('battle_id')],
                                   owner=[base.get('owner'), row.get('owner')]));continue
        group = 'applied' if applied(row) else 'control';roots[group] += 1
        for x in (row.get('private_reveals') or {}).get('redacted') or ():species[str(x.get('species'))] += 1
        expected = [(x.get('species'), x.get('slot')) for x in ((base.get('private_reveals') or {}).get('applicable') or ())]
        got = [(x.get('species'), x.get('slot')) for x in ((row.get('private_reveals') or {}).get('redacted') or ())]
        if expected != got:record_mismatch.append(dict(ordinal=key[0], decision_index=key[1], baseline_applicable=expected, redacted=got))
        theirs, ours = worlds_by_index(row), worlds_by_index(base)
        for index in sorted(set(ours) | set(theirs), key=str):
            w, other = ours.get(index), theirs.get(index)
            if w is None or other is None:status, d = ('baseline_world_missing' if w is None else 'redacted_world_missing'), None
            else:status, d = world_effect(w, other)
            statuses[status] += 1
            if group == 'control' and status not in WORLD_OK:
                determinism.append(dict(ordinal=key[0], decision_index=key[1], world=index, status=status))
            if status != 'compared':continue
            for g in ('all', group):
                kls[g].append(d['kl']);flips[g]['worlds'] += 1;flips[g]['flips'] += d['flip'];flips[g]['bits_differ'] += not d['bits_equal']
            if group == 'control' and (not d['bits_equal'] or d['kl'] != 0 or d['flip']):
                determinism.append(dict(ordinal=key[0], decision_index=key[1], world=index, kl=d['kl'], top_flip=d['flip'], bits_equal=d['bits_equal']))
        status, d = averaging_effect(base, row);counts[f'averaging_{status}'] += 1
        if group == 'control' and status not in ('scored', 'undecided'):
            determinism.append(dict(ordinal=key[0], decision_index=key[1], world=None, averaging_status=status))
        if status != 'scored':continue
        for g in ('all', group):
            p = plays[g];p['roots'] += 1;p['b_changed'] += d['b_changed'];p['raw_flips_all'] += d['flip']
            if d['searched']:p['searched'] += 1;p['flips'] += d['flip'];p['played_flips'] += d['played_flip']
        if group == 'control' and (d['flip'] or d['played_flip'] or d['b_changed']):
            determinism.append(dict(ordinal=key[0], decision_index=key[1], world=None, averaging=[d['baseline'], d['mixed']],
                                    played=[d['played_baseline'], d['played_redacted']], b_changed=d['b_changed'], searched=d['searched']))
    counts['baseline_without_redacted'] = len(set(mine) - seen)
    share = lambda a, b: a / b if b else None
    groups = {}
    for g in GROUPS:
        f, p = flips[g], plays[g]
        groups[g] = dict(roots=roots[g] if g != 'all' else roots['applied'] + roots['control'],
            worlds=f['worlds'], kl_median=quantile(kls[g], .5), kl_p95=quantile(kls[g], .95), kl_max=max(kls[g]) if kls[g] else None,
            kl_nonzero=sum(x > 0 for x in kls[g]), reply_top1_flips=f['flips'], reply_top1_flip_rate=share(f['flips'], f['worlds']),
            logits_bits_differ=f['bits_differ'], averaging=dict(scored_roots=p['roots'], searched_roots=p['searched'],
                played_flips=p['played_flips'], played_flip_rate=share(p['played_flips'], p['searched']),
                raw_flips=p['flips'], raw_flip_rate=share(p['flips'], p['searched']), raw_flips_all_scored=p['raw_flips_all'],
                b_changed=p['b_changed']))
    return dict(schema=SCHEMA, counts=dict(counts), worlds=dict(statuses), groups=groups, redacted_species=dict(species),
        identity_mismatches=dict(count=len(mismatches), examples=mismatches[:EXAMPLES]),
        record_mismatches=dict(count=len(record_mismatch), examples=record_mismatch[:EXAMPLES],
            note="the baseline's applicable members against the redacted run's redacted ones: the same selection must agree"),
        determinism=dict(passed=not determinism, failures=len(determinism), rows=determinism,
            rule='the control (no redaction at the root) must not move (module docstring): every world compared alike with bit-equal '
                 'logits, KL 0 and no reply flip; replays alike; averaging recomputed; (b), averaging and the played action unchanged'))


def summary(report):
    lines = [f"private-reveal effect ({report['schema']})"]
    c = report['counts']
    lines.append(f"rows: baseline {c.get('baseline_rows', 0)}, redacted {c.get('redacted_rows', 0)}, joined {c.get('joined', 0)}; "
                 f"identity mismatches {report['identity_mismatches']['count']}; record mismatches {report['record_mismatches']['count']}")
    lines.append('worlds: ' + ', '.join(f'{k} {v}' for k, v in sorted(report['worlds'].items())))
    fmt = lambda x: 'n/a' if x is None else f'{x:.3g}'
    for g in GROUPS:
        r = report['groups'][g];a = r['averaging']
        lines.append(f"{g}: roots {r['roots']}, worlds {r['worlds']}; KL median {fmt(r['kl_median'])}, p95 {fmt(r['kl_p95'])}, max {fmt(r['kl_max'])}; "
                     f"reply top-1 flips {r['reply_top1_flips']} ({fmt(r['reply_top1_flip_rate'])}); over {a['searched_roots']} roots the baseline "
                     f"searches: PLAYED flips {a['played_flips']} ({fmt(a['played_flip_rate'])}), raw averaging flips {a['raw_flips']} "
                     f"({fmt(a['raw_flip_rate'])}); guard b changed on {a['b_changed']}")
    if report['redacted_species']:lines.append('redacted members: ' + ', '.join(f'{k} {v}' for k, v in sorted(report['redacted_species'].items())))
    d = report['determinism']
    lines.append('determinism: ' + ('ok (the control did not move)' if d['passed'] else f"FAILED: {d['failures']} control world(s) / root(s) moved"))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument('--baseline', nargs='+', required=True, help='rows-*.jsonl of the run without --redact-private-reveals')
    parser.add_argument('--redacted', nargs='+', required=True, help='rows-*.jsonl of the run with --redact-private-reveals')
    parser.add_argument('--out', help='also write the full report here (a new file)')
    args = parser.parse_args(argv)
    report = compare(read_jsonl(args.baseline), read_jsonl(args.redacted))
    if args.out:
        with open(args.out, 'x') as stream:stream.write(json.dumps(report, indent=1, sort_keys=True) + '\n')
    print(summary(report))
    for name in ('identity_mismatches', 'record_mismatches'):
        if report[name]['count']:print(f"WARNING: {report[name]['count']} {name.replace('_', ' ')}; first: {report[name]['examples'][:3]}", file=sys.stderr)
    if not report['determinism']['passed']:
        rows = report['determinism']['rows']
        print(f"DETERMINISM FAILURE: {len(rows)} control world(s) / root(s) moved without a redaction:", file=sys.stderr)
        for x in rows:print(f'  {json.dumps(x, sort_keys=True)}', file=sys.stderr)
        return 3
    return 0


if __name__ == '__main__':sys.exit(main())
