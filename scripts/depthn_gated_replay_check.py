"""Replay check for the live NOISE_GATED_K2 root operator (screen-r4a variant B): re-priced rows of a replay served with
--root-operator NOISE_GATED_K2 (and, for br_fp, --reply-archive) held to the scorer's own rule, recomputed from the rows.

For every replay-COMPLETE row: the worlds are recomputed from the captured depth-2 grids (depthn_rr_score.recompute); under a
reply model (the replay's reply_prior record present, or --reply) each requested world's prior is the row's reply_peer_logits
(depthn_rr_score.model_worlds, br_fp's rule). Expected:
- candidate: search_hedge.averaging's top action on those worlds; q: its q (the replay's q_values exactly, after a JSON round
  trip);
- greedy: the root logits' argmax over root_legal (search_hedge.logits_greedy);
- played: search_hedge.noise_gated(per_world_q(grids), candidate, greedy, 2), the replay's action_index;
- the replay's noise_gate record: candidate, greedy and passed (played == candidate) equal, its root_operator NOISE_GATED_K2.
The mc ladder's R5fp_ng (gpu_engine.mc_ladder_rungs) is the same rule on a br_fp replay's rows.

Output (stdout, JSON): rows, compared, mismatches by field, unscorable (replay-COMPLETE rows the rule cannot be recomputed on:
a healthy replay has none, as the MC ladder's recomputation has none), gate_refusals (rows the live gate refused, NOISE_GATE_ERROR),
gate pass and override rates (played != greedy over compared), and up to 20 example addresses. Exit 0 when every replay-COMPLETE
row was compared and matches and at least one was; 1 on any mismatch or unscorable row; 2 when nothing was compared.

usage: python scripts/depthn_gated_replay_check.py ROWS.jsonl [ROWS.jsonl ...] [--reply]
"""
import argparse
import collections
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import depthn_rr_score as rr   # noqa: E402   (first: it puts src/ on sys.path)
from gpu_engine import search_hedge as hedge   # noqa: E402

OPERATOR = 'NOISE_GATED_K2'
K = 2
EXAMPLES = 20


def expected(raw, d, reply):
    """(candidate, greedy, played, q) of the scorer's rule on one replay-COMPLETE row; raises rr.Unscorable."""
    base = rr.recompute(d)
    worlds = rr.model_worlds(base.worlds, rr.reply_logits_of(d, base.worlds, None)) if reply else base.worlds
    gs, choice = hedge.grids(worlds, base.legal), hedge.averaging(worlds, base.legal)
    if gs is None or choice is None:raise rr.Unscorable('unpriced_cell')
    candidate = hedge.top_action(choice.strategy);greedy = hedge.logits_greedy(d['root_logits'], base.legal)
    return candidate, greedy, hedge.noise_gated(hedge.per_world_q(gs), candidate, greedy, K), choice.q


def check(rows, reply=False):
    """The report (module docstring) over raw re-priced rows."""
    out = dict(rows=0, compared=0, mismatches=collections.Counter(), unscorable=collections.Counter(), gate_refusals=0, passed=0, overrides=0,
               examples=[])
    for raw in rows:
        out['rows'] += 1;d = rr.normalize(raw)
        if d['replay_status'] != 'COMPLETE':
            replay = raw.get('replay') if type(raw.get('replay')) is dict else {}
            out['gate_refusals'] += str(replay.get('refusal') or replay.get('search_refusal') or '').startswith('NOISE_GATE_ERROR');continue
        record = (raw.get('replay') or {}) if type(raw.get('replay')) is dict else {}
        served = reply or type(record.get('reply_prior')) is dict
        try:candidate, greedy, played, q = expected(raw, d, served)
        except (rr.Unscorable, ValueError) as error:
            out['unscorable'][getattr(error, 'reason', type(error).__name__)] += 1
            if len(out['examples']) < EXAMPLES:out['examples'].append(dict(ordinal=raw.get('ordinal'), decision_index=raw.get('decision_index'),
                                                                           wrong=['unscorable']))
            continue
        out['compared'] += 1;gate = record.get('noise_gate') if type(record.get('noise_gate')) is dict else {};wrong = []
        if d['replay_action'] != played:wrong.append('action')
        if not rr.q_json_equal(q, d['replay_q']):wrong.append('q_values')
        if record.get('root_operator') != OPERATOR:wrong.append('root_operator')
        if (gate.get('candidate'), gate.get('greedy'), gate.get('passed')) != (candidate, greedy, played == candidate):wrong.append('noise_gate')
        for field in wrong:out['mismatches'][field] += 1
        if wrong and len(out['examples']) < EXAMPLES:
            out['examples'].append(dict(ordinal=raw.get('ordinal'), decision_index=raw.get('decision_index'), wrong=wrong))
        out['passed'] += played == candidate and candidate != greedy;out['overrides'] += played != greedy
    n = out['compared']
    return dict(out, mismatches=dict(sorted(out['mismatches'].items())), unscorable=dict(sorted(out['unscorable'].items())),
                override_rate=None if not n else round(out['overrides'] / n, 4), reply=reply)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('rows', nargs='+');parser.add_argument('--reply', action='store_true', help='every row was served a reply model (br_fp)')
    args = parser.parse_args(argv)
    rows = [json.loads(line) for path in args.rows for line in open(path) if line.strip()]
    report = check(rows, args.reply);print(json.dumps(report, sort_keys=True))
    return 2 if not report['compared'] else 1 if report['mismatches'] or report['unscorable'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
