"""scripts/depthn_target_argmax_replay_check.py on synthetic re-pricer rows (CPU): a match, mismatches of every kind, zero rows,
input and coverage problems, and the exit codes. tests/test_depthn_rr_reprice runs it on rows the re-pricer itself wrote."""
import copy
from contextlib import redirect_stderr
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gpu_engine import search_hedge as hedge
from pokezero import depthn_effect_guard as G, depthn_root_operator as root_ops, v5_gumbel as original
from scripts import depthn_target_argmax_replay_check as check

LEGAL = [True, True, False, True] + [False] * 5
KEPT = [True, False, True] + [False] * 6
MOVES = ['tackle', 'counter', 'rest', 'protect']   # Counter (tier A) on lane 1
LOGITS = [.4, -30., 0., .2] + [0.] * 5   # Counter's root prior far under THRESHOLD: (b) never trips here, (c) only on lane 1
LINE = '|request|' + json.dumps(dict(active=[dict(moves=[dict(id=m) for m in MOVES])], side=dict(id='p1', pokemon=[])))


def worlds(seed):
    """Four priced worlds (opponent requested, KEPT replies) with values in (-.5, .5)."""
    out = []
    for w in range(4):
        grid = [[(((a * 9 + b) * 7 + w * 3 + seed) % 23) / 23 - .5 if LEGAL[a] and KEPT[b] else None for b in range(9)] for a in range(9)]
        out.append(dict(priced=True, opponent_requested=True, kept=list(KEPT), root_peer_logits=[.1 * b - .2 for b in range(9)], grid=grid))
    return out


def favour(action):return [(.7 if a == action else .1) if LEGAL[a] else 0. for a in range(9)]


def row(ordinal, index, *, target, greedy=0, seed=0, status='COMPLETE', operator=root_ops.TARGET_ARGMAX):
    """A re-pricer row as a TARGET_ARGMAX live search would have written it: averaging's q over the row's worlds, the played
    action target_argmax's, the effect guard record (depthn_effect_guard.evaluate over our lanes) evaluated on it."""
    ws = worlds(seed)
    q = list(hedge.averaging(tuple(original.WorldPrices(True, tuple(KEPT), tuple(w['root_peer_logits']), w['grid']) for w in ws), tuple(LEGAL)).q)
    complete = status == 'COMPLETE'
    action, fallback = root_ops.target_argmax(target, LEGAL, LOGITS, q) if complete else (None, None)
    record = G.evaluate(G.conditions([LINE], 0, LINE), own_turn1=dict(lanes=list(MOVES), logits=list(LOGITS), legal=list(LEGAL)), peer_turn1=[],
                        turn2=[], choice=action) if complete else None
    value = dict(schema=check.ROW_SCHEMA, shard=0, ordinal=ordinal, battle_id=f'b{ordinal}', owner='p1', decision_index=index,
                 replay=dict(status=status, cause=None if complete else 'depthn_refused', action_index=action, q_values=q if complete else None,
                             policy_target=target if complete else None, root_operator=operator, root_operator_impl=root_ops.IMPLS[operator],
                             target_fallback=fallback),
                 live=dict(source='search', greedy_action_index=greedy, search_action_index=None, search_q_values=None, request_kind='move', turn=1),
                 root=dict(decided=complete, root_logits=list(LOGITS) if complete else None, root_legal=list(LEGAL) if complete else None,
                           own_moves=list(MOVES), refusal_cause=None if complete else 'depthn_refused'),
                 worlds=ws if complete else [], effect_guard=record)
    return json.loads(json.dumps(value))   # as the rows file holds it


def rows():
    """Two battles: a plain root, an override, a missing target (the fallback), a target tie (lowest index), a root whose
    target plays Counter ((c) trips: greedy), and a refused root."""
    tie = [.45 if a in (0, 3) else (.1 if LEGAL[a] else 0.) for a in range(9)]
    return [row(1, 0, target=favour(0)), row(1, 1, target=favour(3), seed=5), row(1, 2, target=None, seed=9),
            row(2, 0, target=tie, seed=2), row(2, 1, target=favour(1), seed=4), row(2, 2, target=favour(0), status='REFUSED')]


class ReplayCheckTests(unittest.TestCase):
    def run_check(self, values, *extra, raw=None):
        """(exit code, report) of main over one rows file holding `values` (or the raw text)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rows-0.jsonl'
            path.write_text(raw if raw is not None else ''.join(json.dumps(v) + '\n' for v in values))
            out = io.StringIO()
            with patch('sys.stdout', out), redirect_stderr(io.StringIO()):code = check.main(['--rows', str(path), *extra])
        return code, json.loads(out.getvalue())

    def test_a_match_reports_every_count_and_exits_0(self):
        values = rows();code, report = self.run_check(values, '--expect-battles', '2')
        self.assertEqual((code, report['exit_code'], report['roots_compared'], report['mismatches'], report['mismatch_examples']), (0, 0, 5, 0, []))
        self.assertEqual((report['rows'], report['battles'], report['statuses'], report['not_complete_causes']),
                         (6, 2, dict(COMPLETE=5, REFUSED=1), dict(depthn_refused=1)))
        self.assertEqual(report['fallback'], dict(scorer=1, live=1))
        self.assertEqual({k: report['guard_trips'][k] for k in ('own_recharge', 'unrepresented_effect', 'b', 'c', 'c_without_b', 'any', 'record_missing')},
                         dict(own_recharge=0, unrepresented_effect=1, b=0, c=1, c_without_b=1, any=1, record_missing=0))
        self.assertEqual(report['guard_trips']['arms'], dict(choice=1))
        raws = [v['replay']['action_index'] for v in values[:5]];self.assertEqual(raws[:2] + raws[3:], [0, 3, 0, 1])   # the tie: lowest index
        played = raws[:4] + [0]   # Counter guarded: greedy (0) plays
        self.assertEqual(report['override']['raw_vs_greedy'], sum(r != 0 for r in raws) / 5)
        self.assertEqual(report['override']['played_vs_greedy'], sum(p != 0 for p in played) / 5)
        self.assertEqual(report['override']['guarded_share'], 1 / 5)
        self.assertEqual(report['coverage_problems'], [])

    def test_mismatches_of_every_kind_exit_1_with_examples(self):
        base = rows()
        def changed(i, change):
            values = copy.deepcopy(base);change(values[i]);return values
        def other_action(v):v['replay']['action_index'] = next(a for a in range(9) if LEGAL[a] and a != v['replay']['action_index'])
        def flip_fallback(v):v['replay']['target_fallback'] = not v['replay']['target_fallback']
        def nudge_q(v):v['replay']['q_values'][3] = math.nextafter(v['replay']['q_values'][3], 1.)
        def trip(v):v['effect_guard']['tripped'] = True   # the actor plays greedy; the scorer's rule ((b) or (c) on the choice) does not
        def stale_choice(v):v['effect_guard']['choice'] = 2   # the record evaluated on another action than the one played
        cases = ((1, other_action, {'raw', 'guard_choice', 'played'}), (0, flip_fallback, {'fallback'}), (0, nudge_q, {'q'}),
                 (1, trip, {'played', 'guard_causes'}), (0, stale_choice, {'guard_choice'}))
        for i, change, kinds in cases:
            with self.subTest(change.__name__):
                code, report = self.run_check(changed(i, change))
                self.assertEqual((code, report['mismatches']), (1, 1))
                example, = report['mismatch_examples'];self.assertEqual(set(example['kinds']), kinds)
                self.assertEqual((example['ordinal'], example['decision_index']), (base[i]['ordinal'], base[i]['decision_index']))
        values = copy.deepcopy(base)
        for v in values[:5]:flip_fallback(v)
        code, report = self.run_check(values, '--examples', '2')
        self.assertEqual((code, report['mismatches'], len(report['mismatch_examples']), report['mismatch_kinds']), (1, 5, 2, dict(fallback=5)))
        values = copy.deepcopy(base);other_action(values[1]);values[1]['live']['greedy_action_index'] = None
        self.assertEqual(self.run_check(values)[0], 3)   # a mismatch would be 1; this root is left uncompared: coverage

    def test_zero_rows_or_no_compared_root_is_a_coverage_problem(self):
        code, report = self.run_check([], raw='')
        self.assertEqual((code, report['roots_compared'], report['coverage_problems']), (3, 0, ['no root compared']))
        code, report = self.run_check([row(1, 0, target=favour(0), status='REFUSED')])
        self.assertEqual((code, report['statuses'], report['coverage_problems']), (3, dict(REFUSED=1), ['no root compared']))

    def test_coverage_problems_exit_3(self):
        code, report = self.run_check(rows(), '--expect-battles', '160')
        self.assertEqual((code, report['coverage_problems'], report['mismatches']), (3, ['2 battles, 160 expected'], 0))
        values = rows();values[0]['live']['greedy_action_index'] = None   # no live join: the scorer leaves it out
        code, report = self.run_check(values)
        self.assertEqual((code, report['uncompared'], report['roots_compared']), (3, dict(greedy_missing=1), 4))
        values = rows();values[0]['worlds'][0]['grid'] = None   # an unpriced world: the scorer's recompute refuses
        self.assertEqual(self.run_check(values)[1]['uncompared'], dict(world_unpriced=1))

    def test_input_problems_exit_2(self):
        default = rows();default[0]['replay']['root_operator'] = 'ARGMAX_MEAN_COMPLETED_Q'
        missing = rows();missing[2]['replay'].pop('root_operator')
        schema = rows();schema[0]['schema'] = 'other'
        for name, values, raw in (('default operator', default, None), ('no operator', missing, None), ('duplicate', rows() + rows()[:1], None),
                                  ('schema', schema, None), ('not json', [], '{"schema": \n'), ('not an object', [], '[1, 2]\n')):
            with self.subTest(name):
                code, report = self.run_check(values, raw=raw)
                self.assertEqual((code, report['error']), (2, 'input'))
        with redirect_stderr(io.StringIO()), patch('sys.stdout', io.StringIO()):self.assertEqual(check.main(['--rows', '/nonexistent/rows-0.jsonl']), 2)

    def test_complete_payload_rejected_by_live_client_cannot_pass_gate(self):
        # This root overrides greedy. Omitting the guard used to let both
        # checker paths choose 3, although the live client would refuse it.
        for record in (None, {}, {'tripped': 0}, {'tripped': 'false'}):
            with self.subTest(record=record):
                value = row(1, 0, target=favour(3))
                value['effect_guard'] = record
                code, report = self.run_check([value])
                self.assertEqual((code, report['error']), (2, 'input'))
                self.assertIn('COMPLETE response carries the effect guard', report['detail'])
        value = row(1, 0, target=favour(3))
        value['replay']['cause'] = 'server_error'
        self.assertEqual(self.run_check([value])[0], 2)

    def test_the_check_uses_the_scorers_function_and_the_actors_rule(self):
        from scripts import depthn_rr_score as scorer
        from pokezero import depthn_live_actor as actor
        self.assertIs(check.scorer.target_argmax, root_ops.target_argmax);self.assertIs(check.scorer, scorer)
        self.assertIs(check.actor.guarded_action, actor.guarded_action)


if __name__ == '__main__':unittest.main()
