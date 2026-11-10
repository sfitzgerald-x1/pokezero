#!/usr/bin/env python3
"""NOISE_GATED_K2's override rate on existing re-priced rows, without a GPU: the RR scorer's noise_gated:k=2 recomputed from each
row's own re-priced fields. CPU only.

Input: --rows, re-pricer rows (scripts/depthn_rr_reprice.py rows-*.jsonl) of any root operator (the noise gate's inputs are the
same under every one: the worlds, averaging's choice, the root logits and the effect guard record's root-level (b) and
own_guarded_lanes; a row's own replay choice is never read). Per row whose replay is COMPLETE and that the scorer scores
(scripts/depthn_rr_score.py: normalize, recompute; a legal live greedy action):
- a* = averaging's choice over the row's re-priced worlds (recompute(row).average; its q is held to replay.q_values after a JSON
  round trip, the scorer's parity rule: q_mismatches), qw = per_world_q(its grids);
- two greedies: the RR scorer's (live.greedy_action_index, the live actor's own greedy: the RR's rule) and the live operator's
  (the argmax of the row's re-priced root.root_logits over root.root_legal, lowest index on ties: what the live search has);
- per greedy g: raw = noise_gated(qw, a*, g, 2) (pokezero.depthn_root_operator's, the scorer's own), played = guard_choices'
  choice (the live actor's guards, against the live greedy: the own-recharge guard on root.own_moves, UNREPRESENTED_EFFECT on
  the row's re-priced effect_guard record, (b) or (c) on raw).
Nothing is read from the live telemetry but the live greedy action (the RR's own greedy); policy_target, the live search's
action and q are never read.
Populations: every scored root ('all'), the roots carrying the guard record ('with_record'), and the unguarded roots (the
record's (b) false: the definitive RR's primary population, whose table gives the lead's ~34%). Per population and greedy:
roots, battles, the raw and played override rates against the live greedy (the RR table's override: played != greedy) and the
gate pass rate (raw != g); and the roots where the two greedies differ and where the two raw / played choices do. The RR's
'dropped' scoring keeps only the decisions whose Foul Play reply mapped (the run directory's captures); this script has no
captures, so its populations are the pi_filled scoring's (every scored root). --drop-battles (the definitive run's file, JSON
[{ordinal, reason}]) leaves those battles out, as the RR does.
Output: the JSON report on stdout (and --out, a new file); a one-line summary on stderr. Exit 0 when any root was scored, 2 on
an input problem, 3 when none was.

  python scripts/depthn_noise_gated_override_rate.py --rows RUN/shard-*/rows-*.jsonl [--drop-battles DROP.json] [--out REPORT.json]
"""
import argparse
import collections
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT), str(_ROOT / 'src')):
    if _path not in sys.path:sys.path.insert(0, _path)

from scripts import depthn_rr_score as scorer  # noqa: E402
from scripts.depthn_target_argmax_replay_check import ROW_SCHEMA, InputProblem, read_rows  # noqa: E402
from pokezero.depthn_root_operator import ARGMAX_MEAN_COMPLETED_Q, NOISE_K2, logits_greedy  # noqa: E402

SCHEMA = 'pokezero.depthn-noise-gated-override-rate.v1'
OPERATOR = f'noise_gated:k={NOISE_K2}'   # the scorer's SETTINGS name
GREEDIES = ('rr', 'live_operator')   # the RR scorer's greedy (the live actor's), the live search's (the root logits' argmax)
POPULATIONS = ('all', 'with_record', 'unguarded')


def _int(x):return type(x) is int


def root_choices(row):
    """dict(live_greedy, greedies={rr, live_operator}, chosen, raw={...}, played={...}, record, b, q_equal) of one COMPLETE row;
    scorer.Unscorable when the scorer leaves it out."""
    decision = scorer.normalize(row);base = scorer.recompute(decision);legal = base.legal
    live_greedy = decision['greedy_action']
    if not _int(live_greedy) or not 0 <= live_greedy < 9 or not legal[live_greedy]:raise scorer.Unscorable('greedy_missing')
    record = decision['effect_guard'] if type(decision['effect_guard']) is dict else None
    chosen = scorer.hedge.top_action(base.average.strategy);qw = scorer.per_world_q(base.grids)
    greedies = dict(rr=live_greedy, live_operator=logits_greedy(decision['root_logits'], legal))
    raw = {k: scorer.noise_gated(qw, chosen, g, NOISE_K2) for k, g in greedies.items()}
    played = {k: scorer.guard_choices({OPERATOR: raw[k]}, live_greedy, decision['own_moves'], record)[0][OPERATOR] for k in GREEDIES}
    return dict(live_greedy=live_greedy, greedies=greedies, chosen=chosen, raw=raw, played=played, record=record is not None,
                b=None if record is None else bool(record['b']), q_equal=scorer.q_json_equal(base.average.q, decision['replay_q']))


class Population:
    def __init__(self):self.n = 0;self.battles = set();self.c = collections.Counter()

    def add(self, ordinal, r):
        self.n += 1;self.battles.add(ordinal);g = r['live_greedy']
        for k in GREEDIES:
            self.c[f'{k}.raw_vs_greedy'] += r['raw'][k] != g;self.c[f'{k}.played_vs_greedy'] += r['played'][k] != g
            self.c[f'{k}.gate_passed'] += r['raw'][k] != r['greedies'][k]
        self.c['greedies_differ'] += r['greedies']['rr'] != r['greedies']['live_operator']
        self.c['raw_differs'] += r['raw']['rr'] != r['raw']['live_operator'];self.c['played_differs'] += r['played']['rr'] != r['played']['live_operator']
        self.c['averaging_vs_greedy'] += r['chosen'] != g

    def report(self):
        rate = lambda k: self.c[k] / self.n if self.n else None
        out = dict(roots=self.n, battles=len(self.battles), counts=dict(self.c), averaging_vs_greedy=rate('averaging_vs_greedy'),
                   greedies_differ=rate('greedies_differ'), raw_differs=rate('raw_differs'), played_differs=rate('played_differs'))
        for k in GREEDIES:
            out[k] = dict(raw_vs_greedy=rate(f'{k}.raw_vs_greedy'), played_vs_greedy=rate(f'{k}.played_vs_greedy'), gate_pass=rate(f'{k}.gate_passed'))
        return out


def load_dropped(path):
    """{ordinal: reason} from the RR's --drop-battles JSON [{ordinal, reason}]."""
    try:value = json.loads(Path(path).read_bytes())
    except (OSError, ValueError) as error:raise InputProblem(f'{path}: {error}') from None
    if type(value) is not list or any(type(x) is not dict or not _int(x.get('ordinal')) or type(x.get('reason')) is not str for x in value):
        raise InputProblem(f'{path}: a JSON list of {{ordinal, reason}} required')
    return {x['ordinal']: x['reason'] for x in value}


def rate(rows, *, dropped=None):
    """The report over (path, line, row) triples (module docstring); InputProblem on an input problem."""
    dropped = dropped or {};seen = set();battles = set();operators = collections.Counter();statuses = collections.Counter()
    unscored = collections.Counter();dropped_rows = collections.Counter();q_mismatches = [];q_bad = 0
    populations = {name: Population() for name in POPULATIONS}
    for path, number, row in rows:
        where = f'{path}:{number}'
        if row.get('schema') != ROW_SCHEMA:raise InputProblem(f'{where}: a re-pricer row ({ROW_SCHEMA}) required, got schema {row.get("schema")!r}')
        key = (row.get('ordinal'), row.get('decision_index'))
        if not all(map(_int, key)):raise InputProblem(f'{where}: integer ordinal and decision_index required')
        if key in seen:raise InputProblem(f'{where}: duplicated (ordinal, decision_index) {key}')
        seen.add(key);battles.add(key[0])
        replay = row.get('replay')
        if type(replay) is not dict:raise InputProblem(f'{where}: no replay')
        operators[str(replay.get('root_operator', ARGMAX_MEAN_COMPLETED_Q))] += 1;statuses[str(replay.get('status'))] += 1
        if key[0] in dropped:dropped_rows[dropped[key[0]]] += 1;continue
        if replay.get('status') != 'COMPLETE':continue
        try:r = root_choices(row)
        except scorer.Unscorable as error:unscored[error.reason] += 1;continue
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
            raise InputProblem(f'{where}: malformed row ({type(error).__name__}: {error})') from None
        q_bad += not r['q_equal']
        if not r['q_equal'] and len(q_mismatches) < 20:q_mismatches.append(dict(ordinal=key[0], decision_index=key[1]))
        populations['all'].add(key[0], r)
        if r['record']:populations['with_record'].add(key[0], r)
        if r['record'] and not r['b']:populations['unguarded'].add(key[0], r)
    return dict(schema=SCHEMA, operator=OPERATOR, rows=len(seen), battles=len(battles), row_operators=dict(operators), statuses=dict(statuses),
                dropped_battles=len(dropped), dropped_rows=dict(dropped_rows), unscored=dict(unscored), q_mismatches=q_bad, q_mismatch_examples=q_mismatches,
                populations={name: p.report() for name, p in populations.items()},
                note='override = played (or raw) != the live greedy, as the RR table counts it; gate_pass = raw != that row\'s own g; '
                     'rr: g = live.greedy_action_index (the RR scorer\'s rule); live_operator: g = argmax of root.root_logits (the live '
                     'search\'s rule); populations are the pi_filled scoring\'s (no captures here)')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument('--rows', nargs='+', required=True, help='re-pricer rows-*.jsonl (any root operator)')
    parser.add_argument('--drop-battles', help='the RR\'s --drop-battles file (JSON [{ordinal, reason}]): those battles are left out')
    parser.add_argument('--out', help='also write the report here (a new file)')
    args = parser.parse_args(argv)
    try:report = rate(read_rows(args.rows), dropped=None if args.drop_battles is None else load_dropped(args.drop_battles))
    except InputProblem as error:
        print(json.dumps(dict(schema=SCHEMA, error='input', detail=str(error)), indent=1, sort_keys=True))
        print(f'INPUT PROBLEM: {error}', file=sys.stderr);return 2
    code = 0 if report['populations']['all']['roots'] else 3;report['exit_code'] = code
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        with open(args.out, 'x') as stream:stream.write(text + '\n')
    print(text)
    u = report['populations']['unguarded']
    if u['roots']:
        print(f"{OPERATOR} on {u['roots']} unguarded roots ({u['battles']} battles): override played {u['rr']['played_vs_greedy']:.4f} "
              f"(raw {u['rr']['raw_vs_greedy']:.4f}) under the RR's greedy; {u['live_operator']['played_vs_greedy']:.4f} "
              f"(raw {u['live_operator']['raw_vs_greedy']:.4f}) under the live operator's; greedies differ on {u['counts'].get('greedies_differ', 0)}, "
              f"played choices on {u['counts'].get('played_differs', 0)}", file=sys.stderr)
    else:print('no unguarded root scored', file=sys.stderr)
    return code


if __name__ == '__main__':sys.exit(main())
