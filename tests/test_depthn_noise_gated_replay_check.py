"""scripts/depthn_noise_gated_replay_check.py and scripts/depthn_noise_gated_override_rate.py on synthetic re-pricer rows (CPU): a
match in every regime of the gate, mismatches of every kind, the RR scorer's own greedy reported apart, input and coverage
problems and the exit codes; the override rate from the rows' own re-priced fields. tests/test_depthn_rr_reprice runs the replay
check on rows the re-pricer itself wrote."""
import collections
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
from scripts import depthn_noise_gated_override_rate as rate
from scripts import depthn_noise_gated_replay_check as check
from scripts import depthn_rr_score as scorer

LEGAL = [True, True, False, True] + [False] * 5
KEPT = [True, False, True] + [False] * 6
MOVES = ['tackle', 'counter', 'rest', 'protect']   # Counter (tier A) on lane 1
LOGITS = [.4, -30., 0., .2] + [0.] * 5   # the logits' greedy is lane 0; Counter's root prior far under THRESHOLD: (b) never trips
LINE = '|request|' + json.dumps(dict(active=[dict(moves=[dict(id=m) for m in MOVES])], side=dict(id='p1', pokemon=[])))
W = 8
FLAT = {0: .1, 1: -.5, 3: .4}                                          # a* = 3, gap .3 against 0 in every world: the gate passes
NOISY = {0: 0., 1: -.5, 3: lambda w: .5 if w % 2 else -.45}            # a* = 3, gap mean .025 under 2 SE: the gate fails
AGREE = {0: .4, 1: -.5, 3: .1}                                         # a* = 0 = g
COUNTER = {0: .1, 1: .4, 3: -.5}                                       # a* = 1 (Counter), gap .3 constant: passes, (c) trips


def worlds(values, seed=0):
    """W priced worlds (opponent requested, KEPT replies): our action a is worth values[a] (a number, or a function of the world)
    against every reply, plus a small reply term."""
    out = []
    for w in range(W):
        v = {a: x(w) if callable(x) else x for a, x in values.items()}
        grid = [[v[a] + ((b + seed) % 3) * 1e-3 if LEGAL[a] and KEPT[b] else None for b in range(9)] for a in range(9)]
        out.append(dict(priced=True, opponent_requested=True, kept=list(KEPT), root_peer_logits=[.1 * b - .2 for b in range(9)], grid=grid))
    return out


def row(ordinal, index, values, *, greedy=0, logits=LOGITS, seed=0, status='COMPLETE', operator=root_ops.NOISE_GATED_K2):
    """A re-pricer row as a NOISE_GATED_K2 live search would have written it: averaging over the row's worlds, the played action
    noise_gated_k2's, the effect guard record (depthn_effect_guard.evaluate over our lanes) evaluated on it; `greedy` the live
    actor's greedy (the RR scorer's)."""
    ws = worlds(values, seed);complete = status == 'COMPLETE'
    prices = tuple(original.WorldPrices(True, tuple(KEPT), tuple(w['root_peer_logits']), w['grid']) for w in ws)
    average = hedge.averaging(prices, tuple(LEGAL));q = list(average.q)
    chosen = hedge.top_action(average.strategy)
    action, gate = root_ops.noise_gated_k2(hedge.grids(prices, tuple(LEGAL)), chosen, logits, LEGAL) if complete else (None, None)
    record = G.evaluate(G.conditions([LINE], 0, LINE), own_turn1=dict(lanes=list(MOVES), logits=list(logits), legal=list(LEGAL)), peer_turn1=[],
                        turn2=[], choice=action) if complete else None
    extra = dict(noise_gate=gate) if operator == root_ops.NOISE_GATED_K2 else dict(target_fallback=False) if operator == root_ops.TARGET_ARGMAX else {}
    value = dict(schema=check.ROW_SCHEMA, shard=0, ordinal=ordinal, battle_id=f'b{ordinal}', owner='p1', decision_index=index,
                 replay=dict(status=status, cause=None if complete else 'depthn_refused', action_index=action, q_values=q if complete else None,
                             policy_target=None, **({} if operator == root_ops.ARGMAX_MEAN_COMPLETED_Q else
                                                    dict(root_operator=operator, root_operator_impl=root_ops.IMPLS[operator], **extra))),
                 live=dict(source='search', greedy_action_index=greedy, search_action_index=None, search_q_values=None, request_kind='move', turn=1),
                 root=dict(decided=complete, root_logits=list(logits) if complete else None, root_legal=list(LEGAL) if complete else None,
                           own_moves=list(MOVES), refusal_cause=None if complete else 'depthn_refused'),
                 worlds=ws if complete else [], effect_guard=record)
    return json.loads(json.dumps(value))   # as the rows file holds it


def rows():
    """Two battles: the gate passing, failing, a* == g, Counter played ((c) trips: greedy), a refused root, and a root whose live
    greedy (3) differs from the logits' (0)."""
    return [row(1, 0, FLAT), row(1, 1, NOISY, seed=1), row(1, 2, AGREE, seed=2), row(2, 0, COUNTER), row(2, 1, FLAT, status='REFUSED'),
            row(2, 2, NOISY, greedy=3, seed=3)]


def run_main(main, values, *extra, raw=None):
    """(exit code, report) of a script's main over one rows file holding `values` (or the raw text)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'rows-0.jsonl'
        path.write_text(raw if raw is not None else ''.join(json.dumps(v) + '\n' for v in values))
        out = io.StringIO()
        with patch('sys.stdout', out), redirect_stderr(io.StringIO()):code = main(['--rows', str(path), *[x.replace('TMP', tmp) for x in extra]])
    return code, json.loads(out.getvalue())


class ReplayCheckTests(unittest.TestCase):
    def run_check(self, values, *extra, raw=None):return run_main(check.main, values, *extra, raw=raw)

    def test_the_rows_cover_every_regime(self):
        values = rows()
        self.assertEqual([v['replay']['action_index'] for v in values], [3, 0, 0, 1, None, 0])
        self.assertEqual([None if v['replay']['noise_gate'] is None else v['replay']['noise_gate']['passed'] for v in values],
                         [True, False, False, True, None, False])
        self.assertEqual([v['effect_guard'] and v['effect_guard']['tripped'] for v in values], [False, False, False, True, None, False])

    def test_a_match_reports_every_count_and_exits_0(self):
        code, report = self.run_check(rows(), '--expect-battles', '2')
        self.assertEqual((code, report['exit_code'], report['roots_compared'], report['mismatches'], report['mismatch_examples']), (0, 0, 5, 0, []))
        self.assertEqual((report['rows'], report['battles'], report['statuses'], report['not_complete_causes']),
                         (6, 2, dict(COMPLETE=5, REFUSED=1), dict(depthn_refused=1)))
        self.assertEqual({k: report['gate'][k] for k in ('passed', 'failed', 'candidate_is_greedy', 'live_passed', 'pass_rate')},
                         dict(passed=2, failed=2, candidate_is_greedy=1, live_passed=2, pass_rate=2 / 5))
        self.assertEqual({k: report['guard_trips'][k] for k in ('own_recharge', 'unrepresented_effect', 'b', 'c', 'c_without_b', 'any', 'record_missing')},
                         dict(own_recharge=0, unrepresented_effect=1, b=0, c=1, c_without_b=1, any=1, record_missing=0))
        o = report['override']
        # raw against the live greedy, per root: 3 vs 0, 0 vs 0, 0 vs 0, 1 (Counter) vs 0, 0 vs 3 (the live greedy is 3 there);
        # played: Counter's root plays greedy
        self.assertEqual((o['all']['roots'], o['all']['raw_vs_greedy'], o['all']['played_vs_greedy'], o['all']['gate_pass']), (5, 3 / 5, 2 / 5, 2 / 5))
        self.assertEqual((o['unguarded']['roots'], o['unguarded']['played_vs_greedy']), (5, 2 / 5))   # (b) never trips here
        self.assertEqual((o['not_tripped']['roots'], o['not_tripped']['played_vs_greedy']), (4, 2 / 4))   # Counter's root tripped (c)
        rr = report['rr_greedy']
        self.assertEqual((rr['greedy_differs'], rr['raw_differs'], rr['played_differs'], rr['unguarded_played_differs']), (1, 1, 1, 1))
        example, = rr['examples'];self.assertEqual((example['ordinal'], example['decision_index'], example['rr']['raw'], example['live']['raw']), (2, 2, 3, 0))
        self.assertEqual(report['coverage_problems'], [])

    def test_the_scorer_side_is_the_scorers_own_noise_gated_k2(self):
        """compare_root's scorer side is depthn_rr_score.operators' noise_gated:k=2 on the row, with the logits' greedy (and, for
        rr_greedy, with the live greedy)."""
        for value in rows():
            if value['replay']['status'] != 'COMPLETE':continue
            with self.subTest(ordinal=value['ordinal'], index=value['decision_index']):
                result = check.compare_root(value);decision = scorer.normalize(value);base = scorer.recompute(decision)
                for greedy, side in ((root_ops.logits_greedy(decision['root_logits'], base.legal), result['scorer']['raw']),
                                     (decision['greedy_action'], result['rr']['raw'])):
                    ops, _ = scorer.operators(base.worlds, base.grids, base.average, greedy, base.logits, base.legal, None, collections.Counter())
                    self.assertEqual(ops['noise_gated:k=2'], side)

    def test_mismatches_of_every_kind_exit_1_with_examples(self):
        base = rows()
        def changed(i, change):
            values = copy.deepcopy(base);change(values[i]);return values
        def other_action(v):v['replay']['action_index'] = next(a for a in range(9) if LEGAL[a] and a != v['replay']['action_index'] and a != 1)
        def flip_gate(v):v['replay']['noise_gate']['passed'] = not v['replay']['noise_gate']['passed']
        def other_greedy(v):v['replay']['noise_gate']['greedy'] = 3
        def nudge_q(v):v['replay']['q_values'][3] = math.nextafter(v['replay']['q_values'][3], 1.)
        def trip(v):v['effect_guard']['tripped'] = True   # the actor plays greedy; the scorer's rule ((b) or (c) on the choice) does not
        def stale_choice(v):v['effect_guard']['choice'] = 2   # the record evaluated on another action than the one played
        cases = ((0, other_action, {'raw', 'guard_choice', 'played'}), (0, flip_gate, {'gate'}), (1, other_greedy, {'gate'}), (0, nudge_q, {'q'}),
                 (0, trip, {'played', 'guard_causes'}), (2, stale_choice, {'guard_choice'}))
        for i, change, kinds in cases:
            with self.subTest(change.__name__):
                code, report = self.run_check(changed(i, change))
                self.assertEqual((code, report['mismatches']), (1, 1))
                example, = report['mismatch_examples'];self.assertEqual(set(example['kinds']), kinds)
                self.assertEqual((example['ordinal'], example['decision_index']), (base[i]['ordinal'], base[i]['decision_index']))
        values = copy.deepcopy(base)
        for v in values:
            if v['replay']['noise_gate']:flip_gate(v)
        code, report = self.run_check(values, '--examples', '2')
        self.assertEqual((code, report['mismatches'], len(report['mismatch_examples']), report['mismatch_kinds']), (1, 5, 2, dict(gate=5)))

    def test_the_rr_greedy_divergence_is_reported_never_a_mismatch(self):
        values = rows()
        for v in values:v['live']['greedy_action_index'] = 3   # the live actor's greedy is lane 3 everywhere, the logits' lane 0
        code, report = self.run_check(values)
        self.assertEqual((code, report['mismatches'], report['rr_greedy']['greedy_differs']), (0, 0, 5))
        self.assertGreater(report['rr_greedy']['played_differs'], 0)

    def test_zero_rows_or_no_compared_root_is_a_coverage_problem(self):
        code, report = self.run_check([], raw='')
        self.assertEqual((code, report['roots_compared'], report['coverage_problems']), (3, 0, ['no root compared']))
        code, report = self.run_check([row(1, 0, FLAT, status='REFUSED')])
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
        default = rows();default[0] = row(1, 0, FLAT, operator=root_ops.ARGMAX_MEAN_COMPLETED_Q)
        target = rows();target[0] = row(1, 0, FLAT, operator=root_ops.TARGET_ARGMAX)
        missing = rows();missing[2]['replay'].pop('root_operator')
        schema = rows();schema[0]['schema'] = 'other'
        guardless = rows();guardless[0]['effect_guard'] = None   # a COMPLETE payload the live client refuses
        for name, values, raw in (('default operator', default, None), ('target_argmax', target, None), ('no operator', missing, None),
                                  ('duplicate', rows() + rows()[:1], None), ('schema', schema, None), ('guardless', guardless, None),
                                  ('not json', [], '{"schema": \n'), ('not an object', [], '[1, 2]\n')):
            with self.subTest(name):
                code, report = self.run_check(values, raw=raw)
                self.assertEqual((code, report['error']), (2, 'input'))
        with redirect_stderr(io.StringIO()), patch('sys.stdout', io.StringIO()):self.assertEqual(check.main(['--rows', '/nonexistent/rows-0.jsonl']), 2)

    def test_out_writes_a_new_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'report.json'
            code, report = run_main(check.main, rows(), '--out', str(out))
            self.assertEqual((code, json.loads(out.read_text())), (0, report))
            with self.assertRaises(FileExistsError), redirect_stderr(io.StringIO()), patch('sys.stdout', io.StringIO()):
                check.main(['--rows', '/dev/null', '--out', str(out)])


class OverrideRateTests(unittest.TestCase):
    def run_rate(self, values, *extra, raw=None):return run_main(rate.main, values, *extra, raw=raw)

    def test_rates_from_the_rows_own_fields_under_both_greedies(self):
        """TARGET_ARGMAX rows (the existing re-price) of the same roots: the override rates are the noise gate's on the rows' own
        re-priced worlds, root logits and records, whatever the row's replay played."""
        values = [row(r['ordinal'], r['decision_index'], spec, greedy=r['live']['greedy_action_index'], seed=seed, status=r['replay']['status'],
                      operator=root_ops.TARGET_ARGMAX)
                  for r, spec, seed in zip(rows(), (FLAT, NOISY, AGREE, COUNTER, FLAT, NOISY), (0, 1, 2, 0, 0, 3))]
        code, report = self.run_rate(values)
        self.assertEqual((code, report['rows'], report['battles'], report['row_operators'], report['q_mismatches'], report['unscored']),
                         (0, 6, 2, dict(TARGET_ARGMAX=6), 0, {}))
        u = report['populations']['unguarded']
        self.assertEqual((u['roots'], u['battles']), (5, 2))
        # live operator (g = lane 0): raw 3, 0, 0, 1, 0; played (Counter guarded) 3, 0, 0, 0, 0; against the live greedy 0, 0, 0, 0, 3
        self.assertEqual(u['live_operator'], dict(raw_vs_greedy=3 / 5, played_vs_greedy=2 / 5, gate_pass=2 / 5))
        # the RR's greedy: the last root's g is 3 = a*: raw 3 (no gate), no override
        self.assertEqual(u['rr'], dict(raw_vs_greedy=2 / 5, played_vs_greedy=1 / 5, gate_pass=2 / 5))
        self.assertEqual((u['counts']['greedies_differ'], u['counts']['raw_differs'], u['counts']['played_differs']), (1, 1, 1))
        self.assertEqual(report['populations']['all']['roots'], 5)

    def test_live_telemetry_and_the_replays_own_choice_are_never_read(self):
        base = self.run_rate(rows())[1]
        values = rows()
        for v in values:
            v['live'].update(search_action_index=1, search_q_values=[.9] * 9, policy_target=[1.] + [0.] * 8, played_action_index=1)
            v['replay']['policy_target'] = [0.] * 8 + [1.]
            if v['replay']['status'] == 'COMPLETE':v['replay']['action_index'] = 1
        self.assertEqual(self.run_rate(values)[1]['populations'], base['populations'])

    def test_guarded_roots_leave_the_unguarded_population(self):
        """A root whose record's (b) trips (Counter's root prior over the threshold) is scored but not unguarded: every operator
        plays greedy there."""
        hot = list(LOGITS);hot[1] = .3
        values = rows() + [row(3, 0, FLAT, logits=hot)]
        self.assertTrue(values[-1]['effect_guard']['b'])
        code, report = self.run_rate(values)
        self.assertEqual((report['populations']['all']['roots'], report['populations']['with_record']['roots'], report['populations']['unguarded']['roots']),
                         (6, 6, 5))

    def test_dropped_battles_q_mismatches_and_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            drop = Path(tmp) / 'drop.json';drop.write_text(json.dumps([dict(ordinal=2, reason='journal_rebuild')]))
            code, report = run_main(rate.main, rows(), '--drop-battles', str(drop))
            self.assertEqual((code, report['dropped_rows'], report['populations']['all']['roots']), (0, dict(journal_rebuild=3), 3))
            drop.write_text('{"ordinal": 2}')
            self.assertEqual(run_main(rate.main, rows(), '--drop-battles', str(drop))[0], 2)
        values = rows();values[0]['replay']['q_values'][3] = math.nextafter(values[0]['replay']['q_values'][3], 1.)
        report = self.run_rate(values)[1];self.assertEqual((report['q_mismatches'], report['q_mismatch_examples']), (1, [dict(ordinal=1, decision_index=0)]))
        values = rows();values[0]['live']['greedy_action_index'] = None
        self.assertEqual(self.run_rate(values)[1]['unscored'], dict(greedy_missing=1))
        self.assertEqual(self.run_rate([], raw='')[0], 3)
        self.assertEqual(self.run_rate([row(1, 0, FLAT, status='REFUSED')])[0], 3)
        schema = rows();schema[0]['schema'] = 'other'
        code, report = self.run_rate(schema);self.assertEqual((code, report['schema'], report['error']), (2, rate.SCHEMA, 'input'))
        self.assertEqual(self.run_rate(rows() + rows()[:1])[0], 2)


if __name__ == '__main__':unittest.main()
