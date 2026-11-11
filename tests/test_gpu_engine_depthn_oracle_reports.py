"""Depth-N gate choices, declared bounds, N2 reverdicts and default-path reference runs on the inert fake engine (CPU only).

Split from test_gpu_engine_depthn_oracle (its fixtures and helpers stay there) so each module fits the light tier.
"""
import collections
import contextlib
import copy
import hashlib
import importlib.util
import io
import itertools
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch
import jax.numpy as j
import numpy as np
from gpu_engine import d3_depthn_gate as gate, depthn_oracle as reference, depth2_backups
from gpu_engine import device_public_teacher_batch as batch
from tests.test_gpu_engine_depth2_oracle import PARAMS

from tests import test_gpu_engine_depthn_oracle as N
from tests.test_gpu_engine_depthn_oracle import (
    CHAMPION_FLAGS,
    DECLARED_ARCHIVE,
    DECLARED_CHECKPOINT,
    FAKE_BACKUPS_SOURCE,
    FAKE_MODULE,
    FLOOR,
    FakeBackupsN,
    FixtureN,
    HAVE_PRICER,
    HEAD_FLAGS,
    N2_CHAMPION,
    OracleNPricer,
    ROOT_SITES,
    SITE_KEYS,
    SPREAD,
    TURN1_SITES,
    bounds,
    committed,
    declaration,
    declared,
    depth2_pass_priors,
    failing,
    fake_as_depth2,
    git,
    load,
    measured_declaration,
    n2_reverdicted,
    n2_saved,
    reference_runs,
    reverdicted,
    saved_choices,
    saved_gate,
    site,
    world,
    written)


class ChoicesTests(FixtureN, unittest.TestCase):
    """(f) on the stand-in surface (the oracle's own values: the value bound is 0 unless a synthetic floor is set), the gate's independent
    recursion against the backups, the per-child grid gate (misreads inside the backup, the grid captured as passed, any rule), the row-q
    clause, rule 1's per-node exemption and the bound's edge, one-sided pricing, sampled runs, --reverdict and the header; then the ten
    mutants on the fake backups' source at the run's value bound and both sweeps' survivor accounting."""
    _d = N.GateNTests._d

    def prior(self):
        return self.cached('prior', lambda: self.call(self.pricers()))

    def frame_data(self, depth=3, k=2):
        """The stand-in's depth-N frame (k_own = k_opp = k) and the gate's oracle data over it."""
        def make():
            with self.engine():
                frame = OracleNPricer(self.ex, object(), depth=depth, k_own=k, k_opp=k).price(self.requests, self.tables, self.recipe, PARAMS)
                stack = reference.stack(self.requests, batch._numpy_worlds(self.requests, self.tables, self.recipe))
                return frame, gate.oracle_levels(oracle=self.oracle, stack=stack, requests=self.requests, recipe=self.recipe, params=PARAMS, frame=frame,
                    plan=gate.plan_of(depth, k, k))
        return self.cached(('frame_data', depth, k), make)

    def judge(self, frame, data, *, backups=FakeBackupsN, floor=None):
        """(result, status, (f)) of every check on (frame, data) through summarize_run; floor: a synthetic depth-1 floor (value bound 2 x floor)
        with the evaluator actors 1 (GUARD_DEPTHN_RULE certifies past turn 1)."""
        ev = gate.evaluate_n(frame, self._d(), self.requests, data, context={}, backups=backups, decision=True, choices=True)
        if floor is not None:ev['c']['depth1_noise_floor'] = gate.g2._numeric([[0]], [floor], [0.]);ev['spreads'] = [SPREAD] * 10;ev['actors'] = 1
        result, (status, _) = self.judged(ev);return result, status, result['checks']['f']

    def test_choices_pass_on_the_clean_surface(self):
        evaluation, summary = self.prior();result, (status, reasons) = self.judged(evaluation);f = result['checks']['f']
        self.assertEqual(status, 'PASS', (failing(result), reasons));self.assertEqual(list(result['checks'])[-2:], ['e', 'f'])
        self.assertEqual((f['passed'], f['status'], f['non_exempt_mismatches'], f['one_sided'], f['problems'], f['not_comparable']), (True, 'COMPARED', 0, 0, 0, 0))
        self.assertEqual(set(f['nodes_by_ply']), {'1', '2'});self.assertEqual(f['nodes'], f['learner_comparisons']);self.assertGreater(f['opponent_comparisons'], f['nodes'])
        self.assertEqual((f['max_abs_delta_row_q'], f['max_abs_delta_child_value'], f['value_bound'], f['row_q_within_bound']), (0., 0., 0., True))
        self.assertGreater(f['near_ties'], 0);self.assertEqual(f['near_tie_flips'], 0)   # exact ties among the fake's integer values (bound 0), none flips
        self.assertEqual((result['tolerance']['f_near_ties'], summary['choices']['comparisons']), (f['near_ties'], f['comparisons']))
        for depth, selection in ((4, 'prior'), (3, 'searched')):
            with self.subTest(depth=depth, selection=selection):
                ev, _ = self.call(self.pricers(depth=depth, selection=selection), decision=False);result, (status, _) = self.judged(ev)
                self.assertEqual(status, 'PASS', failing(result));self.assertEqual(set(result['checks']['f']['nodes_by_ply']), {str(p) for p in range(1, depth)})

    def test_the_oracle_recursion_equals_the_backups_and_reads_no_pricer_value(self):
        """OracleTree's values equal node_value's on every interior node (depths 3 and 4) and its choices equal pricer_record's; built over a
        frame whose values and statuses are scrambled (logits kept), it returns the same: it reads only the pricer's priors."""
        for depth in (3, 4):
            with self.subTest(depth=depth):
                frame, data = self.frame_data(depth);tree = gate.OracleTree(frame, data);nodes = gate.interior_nodes(frame, data);priced = 0
                scrambled = copy.deepcopy(frame);scrambled.leaf_value[...] = 7.;scrambled.leaf_status[...] = 3;scrambled.t1_status[...] = 0
                for level in scrambled.nodes:level.value[...] = -7.;level.status[...] = 0
                blind = gate.OracleTree(scrambled, data)
                for q, w, ply, node in nodes:
                    mine, (value, record) = FakeBackupsN.node_value(frame, q, w, ply, node), tree.solve(q, w, ply, node)
                    self.assertEqual(mine, value, (q, w, ply, node));self.assertEqual(blind.solve(q, w, ply, node)[0], value);priced += mine is not None
                    if record is not None:
                        pricer = gate.pricer_record(frame, FakeBackupsN, q, w, ply, node)
                        self.assertEqual([pricer[k] for k in ('own', 'opp', 'learner', 'argmin', 'q', 'gap', 'row_gaps', 'value')],
                            [record[k] for k in ('own', 'opp', 'learner', 'argmin', 'q', 'gap', 'row_gaps')] + [value])
                self.assertGreater(priced, 200);self.assertEqual({k[2] for k in nodes}, set(range(1, depth)))

    def plant(self, oracle_rows, pricer_rows):
        """(frame, data, P): copies with the four leaves of the first fully head-valued two-by-two ply-2 node P set to base + row offset, the
        oracle's (head values) and the pricer's (leaf_value) separately; digests untouched."""
        frame, data = self.frame_data();level = frame.nodes[0];status = np.asarray(frame.leaf_status)
        P = next((q, w, n) for q, w, n in np.argwhere(np.asarray(level.status) == 2).tolist() if (status[q, w, n] == 4).all()
            and (np.asarray(level.own_actions[q, w, n]) >= 0).all() and (np.asarray(level.opp_actions[q, w, n]) >= 0).all())
        frame, data = copy.deepcopy(frame), copy.deepcopy(data);base = float(frame.leaf_value[P + (0, 0)]);leaves = data['leaves']
        for i, key in enumerate(leaves['index'].tolist()):
            if tuple(key[:3]) == P:leaves['head_value'][i] = leaves['value'][i] = np.float32(base + oracle_rows[key[3]])
        for own in (0, 1):frame.leaf_value[P + (own,)] = np.float32(base + pricer_rows[own])
        return frame, data, P

    def test_near_tie_flips_are_exempt_and_reported_and_clear_flips_fail(self):
        """P's rows tie in the oracle and the pricer's row 1 leads by FLOOR (inside the value bound 2 x FLOOR): the learner flip is a near tie,
        reported with both gaps, and the run passes. Rows .05 apart the other way on each side: both gaps exceed the bound, (f) fails."""
        frame, data, P = self.plant((0., 0.), (0., FLOOR));result, status, f = self.judge(frame, data, floor=FLOOR)
        self.assertEqual(status, 'PASS', failing(result));self.assertEqual((f['passed'], f['non_exempt_mismatches']), (True, 0));self.assertGreaterEqual(f['near_tie_flips'], 1)
        flip = next(r for r in f['near_tie_flip_examples'] if (r['root'], r['world'], r['node']) == P and r['ply'] == 2);self.assertEqual((flip['kind'], flip['gap_oracle']), ('learner', 0.))
        self.assertAlmostEqual(flip['gap_pricer'], FLOOR, places=6);self.assertEqual(flip['actions'][::-1], [int(a) for a in frame.nodes[0].own_actions[P][:2]])
        header = gate.header_block(result['checks'])['near_ties']['f'];self.assertEqual((header['near_ties'], header['near_tie_flips']), (f['near_ties'], f['near_tie_flips']))
        frame, data, P = self.plant((.05, 0.), (0., .05));result, status, f = self.judge(frame, data, floor=FLOOR)
        self.assertEqual(status, 'FAIL');self.assertIn('f', failing(result));self.assertFalse(f['passed']);self.assertGreaterEqual(f['non_exempt_mismatches'], 1)
        first = next(r for r in f['first_mismatches'] if (r['root'], r['world'], r['node']) == P and r['ply'] == 2)
        self.assertEqual(first['kind'], 'learner');self.assertTrue(min(first['gap_pricer'], first['gap_oracle']) > f['value_bound'])
        self.assertTrue(any('flip with a gap above the value bound in both frames' in r for r in f['reasons']), f['reasons']);self.assertFalse(f['grid_within_bound'])

    def test_a_node_priced_on_one_side_only_fails(self):
        frame, data = self.frame_data()
        key = next(k for k in gate.interior_nodes(frame, data) if k[2] == 2 and FakeBackupsN.node_value(frame, *k) is not None)
        def node_value(f, q, w, ply, node, rule=FakeBackupsN.learner_max):
            return None if (q, w, ply, node) == key else FakeBackupsN.node_value(f, q, w, ply, node, rule)
        backups = SimpleNamespace(node_value=node_value, learner_max=FakeBackupsN.learner_max, learner_max_root=FakeBackupsN.learner_max_root)
        result, status, f = self.judge(frame, data, backups=backups)
        self.assertEqual(status, 'FAIL');self.assertEqual(set(failing(result)), {'f'});self.assertEqual(f['one_sided'], 1)
        self.assertEqual({k: f['one_sided_examples'][0][k] for k in ('root', 'world', 'ply', 'node', 'priced_by')}, dict(root=key[0], world=key[1], ply=2, node=key[3], priced_by='oracle'))
        self.assertIn('priced on one side only', ' '.join(f['reasons']))
        record = gate.pricer_record(frame, FakeBackupsN, *key);self.assertFalse(gate.node_fails(record, gate.OracleTree(frame, data).solve(*key)[1], 0.))
        self.assertEqual(gate.compare_records(dict(record, finite=False), record)[:2], ([], ['non-finite row q or child value (pricer)']))   # never a NaN gap
        self.assertTrue(gate.node_fails(dict(record, finite=False), record, 1.) and gate.node_fails(record, None, 1.) and not gate.node_fails(None, None, 0.))
        self.assertIn('own actions differ', gate.compare_records(dict(record, own=[8] * len(record['own'])), record)[1][0])
        with patch.dict(sys.modules, {'gpu_engine.depthn_backups': None}):self.assertIn('fail closed', gate.choice_check_n(frame, data, self.requests)['reason'])
        self.assertEqual(gate.merge_choices([dict(passed=False, reason='x')], value_bound=0.)['status'], 'NOT_RUN')

    def test_sampled_runs_compare_only_fully_checked_nodes(self):
        full = self.judged(self.prior()[0])[0]['checks']['f']
        ev, _ = self.call(self.pricers(), sample=1, decision=False);result = gate.summarize_run([ev], (2187,));f = result['checks']['f']
        self.assertEqual(gate.gate_status_n(result, sampled=True)[0], 'PASS_SAMPLED', failing(result));self.assertTrue(f['sampled']);self.assertGreater(f['not_comparable'], 0)
        self.assertLess(f['comparisons'], full['comparisons']);self.assertEqual(f['status'], 'COMPARED' if f['comparisons'] else 'SKIPPED_SAMPLED')
        ev, _ = self.call(self.pricers(), sample=10 ** 6, decision=False);result = gate.summarize_run([ev], (2187,));f = result['checks']['f']   # every leaf drawn
        self.assertEqual((gate.gate_status_n(result, sampled=True)[0], f['comparisons'], f['not_comparable']), ('PASS_SAMPLED', full['comparisons'], 0))
        empty = gate.choice_verdict(dict(saved_choices(), comparisons=0, nodes=0, not_comparable=4), .008, sampled=True)
        self.assertEqual((empty['status'], empty['passed'], gate.gate_status_n(dict(checks=dict(f=empty)), sampled=True)[0]), ('SKIPPED_SAMPLED', True, 'PASS_SAMPLED'))
        vacuous = gate.choice_verdict(dict(saved_choices(), comparisons=0, nodes=0), .008)
        self.assertEqual(gate.gate_status_n(dict(checks=dict(f=vacuous))), ('INCOMPLETE', ['vacuous: (f) compared no interior node']))
        self.assertFalse(gate.choice_verdict(dict(saved_choices(), not_comparable=3), .008)['passed'])   # unsampled: the oracle must value every node

    def test_reverdict_carries_f_through_the_same_verdict(self):
        code, out, lines, _ = reverdicted(saved_gate());f = out['checks']['f']
        self.assertEqual((code, out['status'], f['passed'], f['near_ties'], f['non_exempt_mismatches']), (0, 'PASS', True, 2, 0))
        self.assertEqual(out['header']['near_ties']['f']['near_ties'], 2);self.assertEqual(next(x for x in lines if x['message'] == 'REVERDICT')['header'], out['header'])
        self.assertEqual(list(out)[:2], ['evaluator', 'header'])
        report = saved_gate();report['checks']['f'] = saved_choices([(.001, .02)])   # a near-tie flip: reported, not gated
        code, out, _, _ = reverdicted(report);self.assertEqual((code, out['checks']['f']['near_tie_flips'], out['checks']['f']['passed']), (0, 1, True))
        report = saved_gate();report['checks']['f'] = saved_choices([(.02, .03)])   # saved as passed: the verdict function fails it
        code, out, _, _ = reverdicted(report);f = out['checks']['f']
        self.assertEqual((code, out['status'], f['passed'], f['non_exempt_mismatches']), (1, 'FAIL', False, 1));self.assertIn('f', out['reverdict']['failing_checks'])
        self.assertIn(dict(quantity='f', passed_before=True, passed=False), out['reverdict']['changed_quantities'])
        report = saved_gate();report['checks']['f']['read_identity'].update(passed=False, mismatched_nodes=2)   # exact: whatever the bound
        code, out, _, _ = reverdicted(report);self.assertEqual((code, out['checks']['f']['passed']), (1, False));self.assertIn('read identity', out['checks']['f']['reasons'][0])
        report = saved_gate();report['checks']['f'] = dict(saved_choices([(.02, .03)]), differing_complete=False, non_exempt_mismatches=1, passed=False)
        code, out, _, _ = reverdicted(report);self.assertEqual((code, out['checks']['f']['non_exempt_mismatches']), (1, 1))   # truncated list: kept under the same bound

    def test_header_block(self):
        e, f = dict(near_tie_roots=3, near_tie_flips=1, q_noise_bound=.008), dict(near_ties=7, near_tie_flips=2, status='COMPARED', value_bound=.008)
        block = gate.header_block(dict(e=dict(e, passed=True), f=dict(f, passed=True)));self.assertEqual(block['near_ties'], dict(e=e, f=f))
        rules = block['rules_changed_after_failure'];self.assertEqual(len(rules), 4)
        self.assertEqual([r['triggering_run'] for r in rules], ['d2-gate40-b0-r1', None, 'dn-gate40-prior3-r3', 'dn-gate40-searched3-r3'])
        self.assertTrue(all(set(r) == {'name', 'summary', 'triggering_run', 'trigger', 'approval'} for r in rules))
        self.assertEqual([r['approval'] for r in rules], ['confirmed by the lead 2026-11-06'] * 2 + ['per-child grid gate replaces the guard at plies >= 2: '
            'signed off by the lead 2026-11-06', 'confirmed by the lead 2026-11-06'])
        self.assertIn('not a run', rules[1]['trigger']);self.assertIn('per-child grid gate', rules[2]['summary']);self.assertIn('read identity', rules[2]['summary'])
        self.assertIn('review findings, not a run', rules[2]['trigger']);self.assertIn('kill-test sweep', rules[2]['trigger'])
        self.assertTrue(all(s in rules[0]['trigger'] for s in ('d2-gate40-b0-r1', '60a52442', 'root 15', '1.2e-3 apart in the pricer frame', '2.3e-5 apart in the oracle-heads frame', '4f88727e')))
        self.assertIn('GUARD_DEPTHN_RULE', rules[2]['name']);self.assertIn('DEEPER_LOGIT_RULE', rules[3]['name']);self.assertEqual(gate.header_block()['near_ties'], dict(e=None, f=None))
        self.assertEqual(gate.header_block(configs=dict(top2=dict(e=e)))['near_ties'], dict(top2=dict(e=e)));self.assertTrue(all(x in gate.GUARD_DEPTHN_RULE for x in ('read identity', 'per-child value gate', 'any rule')))

    # ---- the per-child grid gate, the row-q clause and the bound's edge (the review's findings 1, 2 and 4)

    def find(self, k, want):
        """(frame, data, P, record, (row slot, x, y)) for the first ply-2 node P of the k x k stand-in and a pair of opponent slots x, y in one row
        that want(record, row, x, y) accepts (row: the record's row index; x, y: its column indices)."""
        frame, data = self.frame_data(k=k)
        for key in gate._parents(frame):
            record = gate.pricer_record(frame, FakeBackupsN, *key)
            if record is None or 'shape' in record:continue
            for row in range(len(record['own'])):
                for x, y in itertools.combinations(range(len(record['opp'])), 2):
                    if want(record, row, x, y):return frame, data, key, record, (row, x, y)
        self.fail('no such node on the stand-in')

    def swap_in_row(self, key, record, row, x, y):
        """mutant_backups (the fake's source) whose node_value at P reads row `row`'s opponent slots x and y swapped: a misread INSIDE the backup."""
        i, sx, sy = record['own_slots'][row], record['opp_slots'][x], record['opp_slots'][y]
        return gate.mutant_backups([(key + (i, sx), key + (i, sy))], source=FAKE_BACKUPS_SOURCE)

    def test_a_non_pivotal_misread_fails_on_the_per_child_grid_gate_alone(self):
        """Two opponent columns of one row with EQUAL priors, neither the row's argmin, values more than the bound apart, read swapped inside the
        backup: no row q moves (the same products, fsum exact), no choice moves, the frame is clean; only the per-child grid gate fails."""
        floor = 1e-4;bound = 2 * float(np.float32(floor))
        def want(r, row, x, y):
            g = r['grid'][row];low = r['argmin'][row]
            return r['pi_opp'][x] == r['pi_opp'][y] and low not in (x, y) and abs(g[x] - g[y]) > 2 * bound
        frame, data, key, record, (row, x, y) = self.find(3, want)
        result, status, f = self.judge(frame, data, backups=self.swap_in_row(key, record, row, x, y), floor=floor)
        self.assertEqual(status, 'FAIL');self.assertEqual(set(failing(result)), {'f'}, failing(result))
        self.assertEqual((f['grid_within_bound'], f['row_q_within_bound'], f['max_abs_delta_row_q'], f['non_exempt_mismatches'], f['near_tie_flips']), (False, True, 0., 0, 0))
        self.assertEqual(f['nodes_grid_beyond_bound'], 1);self.assertAlmostEqual(f['max_abs_delta_child_value'], abs(record['grid'][row][x] - record['grid'][row][y]))
        self.assertEqual({k: f['first_grid_mismatches'][0][k] for k in ('root', 'world', 'ply', 'node')}, dict(root=key[0], world=key[1], ply=2, node=key[3]))
        self.assertTrue(any('per-child grid gate' in r for r in f['reasons']))
        self.assertTrue(self.judge(frame, data, floor=floor)[2]['passed'])   # the clean backup passes

    def test_a_row_q_misread_without_a_flip_fails_on_the_row_q_clause(self):
        """A non-learner row's two non-argmin columns with unequal priors read swapped: that row's q moves beyond the bound, no choice flips;
        the row-q clause fails (beside the grid gate), on a row other than the learner's."""
        floor = 1e-4;bound = 2 * float(np.float32(floor))
        def want(r, row, x, y):
            g = r['grid'][row];dq = abs((r['pi_opp'][x] - r['pi_opp'][y]) * (g[y] - g[x]));moved = r['q'][row] + (r['pi_opp'][x] - r['pi_opp'][y]) * (g[y] - g[x])
            return row != r['learner'] and r['argmin'][row] not in (x, y) and dq > 2 * bound and moved < r['q'][r['learner']] - 2 * bound and r['gap'] > 2 * bound
        frame, data, key, record, (row, x, y) = self.find(3, want)
        result, status, f = self.judge(frame, data, backups=self.swap_in_row(key, record, row, x, y), floor=floor)
        self.assertEqual(status, 'FAIL');self.assertEqual(set(failing(result)), {'f'}, failing(result))
        self.assertEqual((f['row_q_within_bound'], f['non_exempt_mismatches'], f['near_tie_flips']), (False, 0, 0));self.assertEqual(f['nodes_row_q_beyond_bound'], 1)
        self.assertTrue(any(r.startswith('max |row q diff|') for r in f['reasons']));self.assertGreater(f['max_abs_delta_row_q'], bound)

    def test_the_grid_is_captured_as_passed_to_the_rule(self):
        """A backup that permutes P's grid only where it hands it to the rule (every leaf read and the frame correct): the recorder holds the grid
        as passed, so the per-child grid gate fails at P; a gate re-gathering P's leaves from the frame would see nothing."""
        frame, data, key, record, _ = self.find(2, lambda r, row, x, y: r['grid'][0][0] != r['grid'][0][1])
        module = gate.mutant_backups([], source=FAKE_BACKUPS_SOURCE)
        module._AT_RULE = lambda grid, root, world, ply, node: [row[::-1] for row in grid] if (root, world, ply, node) == key else grid
        result, status, f = self.judge(frame, data, backups=module)
        self.assertEqual(status, 'FAIL');self.assertEqual(set(failing(result)), {'f'});self.assertFalse(f['grid_within_bound'])
        self.assertIn(key, [(r['root'], r['world'], r['ply'], r['node']) for r in f['first_grid_mismatches']])
        np.testing.assert_array_equal(np.asarray(frame.leaf_value)[key[:2] + key[3:]][:len(record['own']), :len(record['opp'])], np.asarray(record['grid'], np.float32))

    def test_any_rule_passes_through_the_recorder(self):
        """A stub min-max rule (not learner-max) on both sides: the recorder returns its value unchanged, the clean grids certify, and a misread
        inside the backup fails the per-child grid gate under it too (the hedge plugs in the same way)."""
        minmax = lambda grid, pi_own, pi_opp: max(min(row) for row in grid)
        frame, data = self.frame_data(k=3);tree = gate.OracleTree(frame, data, rule=minmax)
        clean = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=FakeBackupsN, rule=minmax, tree=tree), 0.)
        self.assertTrue(clean['passed'], clean['reasons']);self.assertGreater(clean['nodes'], 100);self.assertEqual(clean['max_abs_delta_child_value'], 0.)
        moved = [k for k in gate.interior_nodes(frame, data) if FakeBackupsN.node_value(frame, *k) not in (None, FakeBackupsN.node_value(frame, *k, rule=minmax))]
        self.assertTrue(moved);key = moved[0]
        self.assertEqual(gate.pricer_record(frame, FakeBackupsN, *key, rule=minmax)['value'], FakeBackupsN.node_value(frame, *key, rule=minmax))
        self.assertEqual(tree.solve(*key)[0], FakeBackupsN.node_value(frame, *key, rule=minmax))
        _, _, P, record, (row, x, y) = self.find(3, lambda r, row, x, y: abs(r['grid'][row][x] - r['grid'][row][y]) > 0)
        bad = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=self.swap_in_row(P, record, row, x, y), rule=minmax, tree=tree), 0.)
        self.assertFalse(bad['grid_within_bound']);self.assertFalse(bad['passed'])

    def test_a_gap_or_difference_at_the_bound_is_within_it(self):
        """Exactly at the bound (and a rounding's REL_EPS above) a gap is a near tie and a row q or grid difference passes; 1e-9 above, not."""
        for scale, within in ((1., True), (1. + 1e-13, True), (1. + 1e-9, False)):
            with self.subTest(scale=scale):
                at = .008 * scale;f = gate.choice_verdict(saved_choices([(at, .03)]), .008)
                self.assertEqual((f['passed'], f['near_tie_flips'], f['non_exempt_mismatches']), (within, int(within), int(not within)))
                for key in ('max_abs_delta_row_q', 'max_abs_delta_child_value'):
                    self.assertEqual(gate.choice_verdict(dict(saved_choices(), **{key: at}), .008)['passed'], within, key)
                self.assertEqual(gate._within(at, .008), within)
        self.assertFalse(gate._within(1e-300, 0.));self.assertTrue(gate._within(0., 0.))

    def test_tie_breaks_keep_the_lower_slot(self):
        solved = gate._solve([[.5, .5, .2], [.3, .3, .3]], [.25, .25, .5], [0, 2], [0, 1, 2], [4, 6], [1, 2, 3])
        self.assertEqual((solved['argmin'], solved['learner'], solved['row_gaps']), ([2, 0], 0, [.3, 0.]))
        self.assertEqual(gate._solve([[.3, .3], [.3, .3]], [.5, .5], [0, 1], [0, 1], [4, 6], [1, 2])['learner'], 0)

    def test_an_errored_descendant_decides_none_over_a_sampled_out_leaf(self):
        frame, data = self.frame_data();leaves = data['leaves'];status = np.asarray(frame.leaf_status)
        key = next(k for k in gate._parents(frame) if (status[k[:2] + k[3:]] == 4).sum() >= 2 and gate.OracleTree(frame, data).solve(*k)[0] is not None)
        rows = [i for i, index in enumerate(leaves['index'].tolist()) if (index[0], index[1], index[2]) == (key[0], key[1], key[3])]
        def tree(drop, error):
            changed = copy.deepcopy(data);mask = np.ones(len(leaves['specs']), bool);mask[drop] = False
            if error is not None:changed['leaves']['status'][error] = 3
            changed['leaves'] = {k: (np.asarray(v)[mask] if k in ('specs', 'index', 'status', 'value', 'head_value') else v) for k, v in changed['leaves'].items()}
            return gate.OracleTree(frame, changed)
        self.assertIs(tree([rows[0]], None).solve(*key)[0], gate.MISSING);self.assertIs(tree([rows[0]], rows[1]).solve(*key)[0], None)
        self.assertIs(tree([rows[0]], rows[1]).solve(*gate._up(frame, key))[0], None)
        layout = gate.tagged_frame(frame)[1];identity = lambda drop, error: gate.IdentityTree(tree(drop, error), layout, gate.Intern())   # read identity: the same order
        self.assertIs(identity([rows[0]], None).solve(*key)[0], gate.MISSING);self.assertIs(identity([rows[0]], rows[1]).solve(*key)[0], None)
        self.assertIs(identity([rows[0]], rows[1]).solve(*gate._up(frame, key))[0], None)
        sampled_out = tree([rows[0]], None)   # a leaf the oracle did not compute: unsampled, read identity fails closed; sampled, it skips the node
        hidden = gate.read_identity_n(frame, data, self.requests, backups=FakeBackupsN, tree=sampled_out)
        self.assertEqual((hidden['passed'], hidden['mismatched_nodes']), (False, 0));self.assertGreater(hidden['not_comparable'], 0)
        self.assertTrue(any('could not name' in r for r in hidden['reasons']))
        self.assertTrue(gate.read_identity_n(frame, data, self.requests, backups=FakeBackupsN, tree=sampled_out, sampled=True)['passed'])
        clean = gate.read_identity_n(frame, data, self.requests, backups=FakeBackupsN);self.assertEqual((clean['passed'], clean['not_comparable']), (True, 0))

    # ---- read identity (the re-review's C2), rules that edit their grid or use the priors, matching by action

    def interior_pair(self, depth, k, ply, want):
        """(frame, data, parent key, slot pair s, slot pair t) for the first priced parent at `ply` of the depth/k stand-in with two EXPANDED
        children (interior: ply + 1 < depth) that want(frame, key, a, b) accepts (a, b: the children's node ids at ply + 1)."""
        frame, data = self.frame_data(depth=depth, k=k);keys = gate._parents(frame) if ply >= 2 else [(q, w, 1, a1 * 9 + r1) for q, w, a1, r1 in
            np.argwhere(np.asarray(frame.t1_status) == 2).tolist()]
        k_own, k_opp = int(frame.k_own[ply - 1]), int(frame.k_opp[ply - 1]);level = frame.nodes[ply - 1]
        for key in keys:
            if key[2] != ply:continue
            record = gate.pricer_record(frame, FakeBackupsN, *key)
            if record is None or 'shape' in record:continue
            for s, t in itertools.combinations([(i, x) for i in record['own_slots'] for x in record['opp_slots']], 2):
                a, b = (key[3] * k_own * k_opp + u[0] * k_opp + u[1] for u in (s, t))
                if all(int(level.status[key[0], key[1], n]) == 2 for n in (a, b)) and want(frame, key, a, b):return frame, data, key, s, t
        self.fail('no such pair on the stand-in')

    def test_read_identity_passes_clean_frames_at_depths_3_and_4(self):
        for depth, k in ((3, 2), (3, 3), (4, 2)):
            with self.subTest(depth=depth, k=k):
                frame, data = self.frame_data(depth=depth, k=k);ri = gate.read_identity_n(frame, data, self.requests, backups=FakeBackupsN)
                self.assertEqual((ri['passed'], ri['mismatched_nodes'], ri['one_sided']), (True, 0, 0));self.assertGreater(ri['nodes'], 1000)
                self.assertEqual(ri['tags']['interned'], ri['nodes']);self.assertGreater(ri['entries'], 3 * ri['nodes'])
        f = self.judged(self.prior()[0])[0]['checks']['f'];self.assertTrue(f['read_identity']['passed']);self.assertGreater(f['read_identity']['nodes'], 1000)

    def test_read_identity_catches_an_interior_misread_the_value_gates_pass(self):
        """The re-review's test_2: depth 4, k = 3, a ply-2 parent reading two interior (ply-3) children swapped, picked where their learner-max
        values are closest relative to a min-max rule's. At a bound equal to the learner-max difference every value gate passes; read identity
        fails at the parent under the live learner-max, naming the two subtrees each entry read."""
        minmax = lambda grid, pi_own, pi_opp: max(min(row) for row in grid);B = FakeBackupsN;best = []
        def want(frame, key, a, b):
            la, lb, ma, mb = (B.node_value(frame, key[0], key[1], 3, n, **kw) for kw in ({}, {'rule': minmax}) for n in (a, b))
            if None in (la, lb, ma, mb) or ma == mb:return False
            best.append((abs(ma - mb) / max(abs(la - lb), 1e-12), key, a, b, abs(la - lb)));return False
        with self.assertRaises(AssertionError):self.interior_pair(4, 3, 2, want)   # scores every pair, accepts none
        _, key, a, b, bound = max(best, key=lambda r: r[0]);frame, data = self.frame_data(depth=4, k=3);K = int(frame.k_own[1]) * int(frame.k_opp[1])
        s, t = (((n - key[3] * K) // int(frame.k_opp[1]), (n - key[3] * K) % int(frame.k_opp[1])) for n in (a, b))
        f = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=gate.mutant_backups([(key + s, key + t)], source=FAKE_BACKUPS_SOURCE)), bound)
        self.assertEqual((f['grid_within_bound'], f['row_q_within_bound'], f['non_exempt_mismatches'], f['read_identity']['passed'], f['passed']), (True, True, 0, False, False))
        first = f['read_identity']['first_mismatches'][0];self.assertEqual((first['root'], first['world'], first['ply'], first['node']), key)
        reads = {tuple(e['actions']): (e['read'], e['expected']) for e in first['entries']};self.assertEqual(len(reads), 2)
        (r1, e1), (r2, e2) = reads.values();self.assertEqual((r1, r2), (e2, e1))   # each entry read the other child's subtree
        self.assertTrue(all(r[0][:2] == ['node', key[0]] and r[0][3] == 3 and r[1][0] == 'node' for r in reads.values()))   # interned tags named as oracle nodes
        self.assertTrue(first['origin']);self.assertTrue(any('read identity' in r for r in f['reasons']))
        self.assertEqual((f['read_identity']['injective'], f['read_identity']['interned_collisions']), (True, 0))   # a swap repeats no read

    def test_read_identity_catches_a_turn1_misread_under_min_max(self):
        """Depth 3: a turn-1 cell reading two interior (ply-2) children swapped whose min-max values are EQUAL (subtrees differ): under the
        min-max rule every value gate passes; read identity fails at the cell."""
        minmax = lambda grid, pi_own, pi_opp: max(min(row) for row in grid);B = FakeBackupsN
        frame, data, key, s, t = self.interior_pair(3, 3, 1, lambda frame, key, a, b: B.node_value(frame, key[0], key[1], 2, a, rule=minmax) is not None
            and B.node_value(frame, key[0], key[1], 2, a, rule=minmax) == B.node_value(frame, key[0], key[1], 2, b, rule=minmax))
        module = gate.mutant_backups([(key + s, key + t)], source=FAKE_BACKUPS_SOURCE)
        f = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=module, rule=minmax, tree=gate.OracleTree(frame, data, rule=minmax)), 0.)
        self.assertEqual((f['grid_within_bound'], f['row_q_within_bound'], f['read_identity']['passed'], f['passed']), (True, True, False, False))
        self.assertIn(key, [(r['root'], r['world'], r['ply'], r['node']) for r in f['read_identity']['first_mismatches']])

    def test_read_identity_guards_injectivity_and_one_to_one(self):
        """INJECTIVITY: a tagged copy (wrongly) cast to float32 collapses the tags: the distinct-count guard fails, gated, not a pass. INTERIOR
        IDENTITY: a backup handing a second node the first node's grid gives two nodes one interned tag: the one-to-one guard fails."""
        frame, data = self.frame_data();clean = gate.read_identity_n(frame, data, self.requests, backups=FakeBackupsN)
        self.assertEqual((clean['injective'], clean['interned_collisions'], clean['distinct_captured_tags']), (True, 0, clean['distinct_named_reads']))
        real = gate.tagged_frame
        def float32(frame, **kw):
            tagged, layout = real(frame, **kw)
            return gate._copy(tagged, leaf_value=np.asarray(tagged.leaf_value, np.float32), nodes=tuple(gate._copy(level, value=np.asarray(level.value, np.float32))
                for level in tagged.nodes)), layout
        with patch.object(gate, 'tagged_frame', float32):ri = gate.read_identity_n(frame, data, self.requests, backups=FakeBackupsN)
        self.assertEqual((ri['passed'], ri['injective']), (False, False));self.assertLess(ri['distinct_captured_tags'], ri['distinct_named_reads'])
        self.assertTrue(any('injectivity' in r for r in ri['reasons']))
        f = gate.choice_verdict(dict(saved_choices(), read_identity=ri), .008);self.assertFalse(f['passed']);self.assertTrue(any('injectivity' in r for r in f['reasons']))
        first, second = [k for k in gate._parents(frame) if gate.pricer_record(frame, FakeBackupsN, *k) is not None][:2];held = {}
        def hand_over(grid, root, world, ply, node):
            if (root, world, ply, node) == first:held['grid'] = grid
            return held['grid'] if (root, world, ply, node) == second and 'grid' in held else grid
        module = gate.mutant_backups([], source=FAKE_BACKUPS_SOURCE);module._AT_RULE = hand_over
        ri = gate.read_identity_n(frame, data, self.requests, backups=module)
        self.assertFalse(ri['passed']);self.assertGreaterEqual(ri['interned_collisions'], 1);self.assertIn([list(first), list(second)], [c['nodes'] for c in ri['first_collisions']])
        self.assertTrue(any('one-to-one' in r for r in ri['reasons']))


    def test_tags_are_exact_and_overflow_fails_closed(self):
        from fractions import Fraction
        for k in (0, 1, 12345, gate.TAG_CAPACITY - 1):
            tag = gate.position_tag(k);self.assertEqual(Fraction(tag), Fraction(-1) + Fraction(k + 1, 2 ** 42));self.assertTrue(-1. < tag < 0.)
            tag = gate.interned_tag(k);self.assertEqual(Fraction(tag), Fraction(k + 1, 2 ** 42));self.assertTrue(0. < tag < 1.)
        frame, data = self.frame_data(depth=4, k=2);tagged, layout = gate.tagged_frame(frame)
        arrays = [np.asarray(tagged.leaf_value)] + [np.asarray(level.value) for level in tagged.nodes]
        self.assertTrue(all(a.dtype == np.float64 for a in arrays));flat = np.concatenate([a.reshape(-1) for a in arrays])
        self.assertEqual(len(np.unique(flat)), len(flat));self.assertTrue(((flat > -1.) & (flat < 0.)).all())
        self.assertEqual(gate.decode_tag(float(tagged.leaf_value[3, 1, 5, 1, 0]), layout), ['leaf', 3, 1, 5, 1, 0])
        self.assertEqual(gate.decode_tag(float(tagged.nodes[1].value[2, 0, 17]), layout), ['ply3', 2, 0, 17]);self.assertEqual(gate.decode_tag(gate.interned_tag(4), layout), ['grid', 4])
        self.assertIs(tagged.leaf_status, frame.leaf_status);self.assertIs(tagged.nodes[0].own_actions, frame.nodes[0].own_actions)   # statuses and maps untouched
        positions = len(flat);ri = gate.read_identity_n(frame, data, self.requests, backups=FakeBackupsN, capacity=positions - 1)
        self.assertEqual(ri['passed'], False);self.assertIn('fails closed', ri['reason']);self.assertTrue(gate.read_identity_n(frame, data, self.requests, backups=FakeBackupsN,
            capacity=positions)['passed'])
        intern = gate.Intern(capacity=2);intern.tag([[.1]]);intern.tag([[.2]]);self.assertEqual(intern.tag([[.1]]), gate.interned_tag(0))
        with self.assertRaises(gate.TagOverflow):intern.tag([[.3]])
        f = gate.choice_verdict(dict(saved_choices(), read_identity=dict(ri)), .008);self.assertFalse(f['passed']);self.assertIn('fails closed', f['reasons'][0])

    def test_an_in_place_sorting_rule_neither_false_fails_nor_blinds(self):
        """The re-review's rule that sorts its grid's rows IN PLACE before solving: the clean backup passes (the recorder snapshots before the rule,
        the oracle hands the rule its own copy) and a within-row misread fails."""
        def sorting(grid, pi_own, pi_opp):
            for row in grid:row.sort()
            return max(sum(row) / len(row) for row in grid)
        frame, data = self.frame_data(k=3);tree = gate.OracleTree(frame, data, rule=sorting)
        clean = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=FakeBackupsN, rule=sorting, tree=tree), 0.)
        self.assertTrue(clean['passed'], clean['reasons']);self.assertEqual(clean['max_abs_delta_child_value'], 0.)
        _, _, key, record, (row, x, y) = self.find(3, lambda r, row, x, y: r['grid'][row][x] != r['grid'][row][y])
        bad = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=self.swap_in_row(key, record, row, x, y), rule=sorting, tree=tree), 0.)
        self.assertFalse(bad['passed']);self.assertFalse(bad['grid_within_bound']);self.assertFalse(bad['read_identity']['passed'])

    def test_a_prior_using_rule_solves_alike_on_both_sides(self):
        """A rule that weighs rows by pi_own and columns by pi_opp (depthn_backups' diagnostic average): both sides hand it the same priors in the
        same roles, so the clean grids certify; the priors' roles matter on this surface (swapping them moves some node's value)."""
        average = lambda grid, pi_own, pi_opp: math.fsum(p * math.fsum(q * v for q, v in zip(pi_opp, row)) for p, row in zip(pi_own, grid))
        swapped = lambda grid, pi_own, pi_opp: average(grid, pi_opp, pi_own)
        frame, data = self.frame_data(k=3)
        clean = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=FakeBackupsN, rule=average, tree=gate.OracleTree(frame, data, rule=average)), 0.)
        self.assertTrue(clean['passed'], clean['reasons']);self.assertGreater(clean['nodes'], 1000)
        self.assertTrue(any(FakeBackupsN.node_value(frame, *k, rule=average) != FakeBackupsN.node_value(frame, *k, rule=swapped)
            for k in gate.interior_nodes(frame, data) if FakeBackupsN.node_value(frame, *k) is not None))

    def test_records_are_matched_by_action_not_slot(self):
        """The oracle's record with its opponent columns in the reverse slot order (the same actions): no grid, row q or choice difference."""
        frame, data = self.frame_data(k=3);key = next(k for k in gate._parents(frame) if (lambda r: r is not None and len(r['opp']) == 3 and len(set(r['grid'][0])) == 3)(
            gate.pricer_record(frame, FakeBackupsN, *k)))
        mine = gate.pricer_record(frame, FakeBackupsN, *key)
        theirs = gate._solve([row[::-1] for row in mine['grid']], mine['pi_opp'][::-1], mine['own_slots'], mine['opp_slots'][::-1], mine['own'], mine['opp'][::-1])
        comparisons, problems, dq, dv = gate.compare_records(mine, theirs)
        self.assertEqual((problems, dq, dv), ([], 0., 0.));self.assertTrue(all(got == want for _, _, got, want, _, _ in comparisons))
        theirs['pi_own'] = mine['pi_own'];self.assertEqual(gate.compare_priors(mine, theirs), [])   # priors by action too (the opponent's reversed)
        self.assertNotEqual(gate.compare_priors(mine, dict(theirs, pi_opp=mine['pi_opp'])), [])

    # ---- the exact prior check (the re-review of 27adf5a7), merged gating and diagnostics

    def wrong_priors(self, new):
        """The fake backups' source with node_value's pi_own replaced by `new` (the re-review's probes)."""
        line = 'pi_own=_softmax(own_logits,[int(own_actions[i])for i in own_slots])';self.assertEqual(FAKE_BACKUPS_SOURCE.count(line), 1)
        module = types.ModuleType('wrong_priors');exec(compile(FAKE_BACKUPS_SOURCE.replace(line, new), '<wrong priors>', 'exec'), module.__dict__);return module

    def test_the_priors_handed_to_the_rule_are_exact(self):
        """Clean frames pass the exact prior check at depths 2 to 4 (k 1 to 3: single-slot maps; the fixture's waiting seats); a backup handing its
        rule a uniform pi_own, or one built from the opponent's logits, fails it alone (grid, row q and read identity pass: learner-max ignores
        pi_own), and under depthn_backups' diagnostic average rule it moves values."""
        for depth, k in ((2, 3), (3, 1), (3, 3), (4, 2)):
            with self.subTest(depth=depth, k=k):
                ev, _ = self.call(self.pricers(depth=depth, k_own=k, k_opp=k), decision=False);result = self.judged(ev)[0];f = result['checks']['f']
                self.assertEqual(self.judged(ev)[1][0], 'PASS', failing(result));self.assertEqual((f['priors']['passed'], f['priors']['mismatched_nodes']), (True, 0))
                self.assertGreater(f['priors']['nodes'], 100)
        frame, data = self.frame_data(k=3);records = [gate.pricer_record(frame, FakeBackupsN, *key) for key in gate.interior_nodes(frame, data)]
        self.assertTrue(any(r and 'shape' not in r and len(r['own']) == 1 for r in records));self.assertTrue(any(r and 'shape' not in r and len(r['opp']) == 1 for r in records))
        average = lambda grid, pi_own, pi_opp: math.fsum(p * math.fsum(q * v for q, v in zip(pi_opp, row)) for p, row in zip(pi_own, grid))
        for name, new in (('uniform', 'pi_own=[1./len(own_slots)]*len(own_slots)'), ('opponent logits', 'pi_own=_softmax(opp_logits,[int(own_actions[i])for i in own_slots])')):
            with self.subTest(name):
                module = self.wrong_priors(new);f = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=module), 0.)
                self.assertEqual((f['passed'], f['priors']['passed'], f['grid_within_bound'], f['row_q_within_bound'], f['read_identity']['passed']), (False, False, True, True, True))
                self.assertGreater(f['priors']['mismatched_nodes'], 100);self.assertEqual(f['priors']['first_mismatches'][0]['seats'][0]['seat'], 'own')
                avg = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=module, rule=average, tree=gate.OracleTree(frame, data, rule=average)), 0.)
                self.assertFalse(avg['grid_within_bound']);self.assertFalse(avg['priors']['passed'])
        mine = next(r for r in records if r and 'shape' not in r and len(r['own']) > 1)
        self.assertEqual(gate.compare_priors(dict(mine, pi_own=None), mine)[0]['pricer'], None)   # a missing prior is a mismatch
        self.assertEqual(gate.compare_priors(mine, dict(mine)), [])
        line = 'pi_opp=_softmax(opp_logits,[int(opp_actions[x])for x in opp_slots])'   # the opponent's seat is checked as exactly (the value gates aside)
        self.assertEqual(FAKE_BACKUPS_SOURCE.count(line), 1);module = types.ModuleType('wrong_opp')
        exec(compile(FAKE_BACKUPS_SOURCE.replace(line, 'pi_opp=[1./len(opp_slots)]*len(opp_slots)'), '<wrong pi_opp>', 'exec'), module.__dict__)
        f = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=module), 0.)
        self.assertFalse(f['priors']['passed']);self.assertEqual({x['seat'] for m in f['priors']['first_mismatches'] for x in m['seats']}, {'opp'})
        ulp = self.wrong_priors('pi_own=[math.nextafter(p,2.)for p in _softmax(own_logits,[int(own_actions[i])for i in own_slots])]')   # one ulp: exact, no tolerance
        f = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=ulp), 0.);self.assertFalse(f['priors']['passed'])
        self.assertEqual((f['grid_within_bound'], f['read_identity']['passed']), (True, True))

    def test_priors_are_checked_at_turn1_cells_and_at_leaf_parents(self):
        """A backup handing a uniform pi_own ONLY at the turn-1 cells, or ONLY at the parents of leaves (depthn_backups' _PRIORS hook): learner-max
        values, the grid and read identity are unchanged, and the exact prior check fails at exactly those plies."""
        frame, data = self.frame_data(k=3);uniform = lambda pi: [1. / len(pi)] * len(pi)
        for name, at in (('turn-1 cells', lambda ply: ply == 1), ('leaf parents', lambda ply: ply + 1 == int(frame.depth))):
            with self.subTest(name):
                module = gate.mutant_backups([], source=FAKE_BACKUPS_SOURCE);module._PRIORS = lambda pi, root, world, ply, node, at=at: uniform(pi) if at(ply) else pi
                f = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=module), 0.)
                self.assertEqual((f['passed'], f['priors']['passed'], f['grid_within_bound'], f['row_q_within_bound'], f['read_identity']['passed']), (False, False, True, True, True))
                self.assertGreater(f['priors']['mismatched_nodes'], 100);self.assertTrue(f['priors']['first_mismatches'])
                self.assertTrue(all(at(m['ply']) for m in f['priors']['first_mismatches']), [m['ply'] for m in f['priors']['first_mismatches']])

    def test_the_expected_prior_is_the_gates_own_softmax(self):
        """A gpu_engine.depthn_backups whose _softmax is wrong (twice the logit scale) installed where the gate imports the backups: the backup
        built from it hands wrong priors and the exact prior check FAILS, the gate's expectation being its own prior_softmax, never depthn_backups';
        the correct backup passed beside it still passes."""
        line = 'weights=[math.exp(float(logits[a])-top)for a in actions]';self.assertEqual(FAKE_BACKUPS_SOURCE.count(line), 1)
        wrong = types.ModuleType('gpu_engine.depthn_backups')
        exec(compile(FAKE_BACKUPS_SOURCE.replace(line, 'weights=[math.exp(2.*(float(logits[a])-top))for a in actions]'), '<wrong softmax>', 'exec'), wrong.__dict__)
        frame, data = self.frame_data(k=3)
        with patch.dict(sys.modules, {'gpu_engine.depthn_backups': wrong}):
            bad = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests), 0.)   # backups=None: the installed module
            good = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=FakeBackupsN), 0.)
        self.assertEqual((bad['passed'], bad['priors']['passed']), (False, False));self.assertGreater(bad['priors']['mismatched_nodes'], 100)
        self.assertTrue(good['passed'], good['reasons']);self.assertEqual(good['priors']['mismatched_nodes'], 0)
        self.assertEqual(gate.prior_softmax([0., .5, 1.], [0, 2, 1]), FakeBackupsN._softmax([0., .5, 1.], [0, 2, 1]))
        self.assertNotEqual(gate.prior_softmax([0., .5, 1.], [0, 2, 1]), wrong._softmax([0., .5, 1.], [0, 2, 1]))

    def test_a_rule_editing_its_priors_in_place_changes_nothing_recorded(self):
        """A rule that edits pi_own and pi_opp in place after solving (reversed, zeroed): the recorder copied both before the rule ran and the
        oracle hands the rule copies, so the correct backup passes the prior check and every gate, and a misread still fails."""
        def editing(grid, pi_own, pi_opp):
            value = max(math.fsum(p * v for p, v in zip(pi_opp, row)) for row in grid)   # learner-max on the priors as handed
            pi_own.reverse();pi_opp[:] = [0.] * len(pi_opp);return value
        frame, data = self.frame_data(k=3);tree = gate.OracleTree(frame, data, rule=editing)
        clean = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=FakeBackupsN, rule=editing, tree=tree), 0.)
        self.assertTrue(clean['passed'], clean['reasons']);self.assertEqual((clean['priors']['passed'], clean['priors']['mismatched_nodes']), (True, 0))
        self.assertGreater(clean['priors']['nodes'], 1000)
        _, _, key, record, (row, x, y) = self.find(3, lambda r, row, x, y: r['grid'][row][x] != r['grid'][row][y])
        bad = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=self.swap_in_row(key, record, row, x, y), rule=editing, tree=tree), 0.)
        self.assertFalse(bad['passed']);self.assertTrue(bad['priors']['passed'])   # a misread, not a prior: the priors handed stay right


    def test_read_identity_and_priors_are_gated_on_their_counts_too(self):
        """Defence in depth: a merged read identity ANDs its calls; (f) fails on a missing or null read identity or priors (a saved null fails
        --reverdict), and on mismatched_nodes, interned_collisions or a false injective flag even when `passed` says otherwise."""
        ok, bad = dict(saved_choices()['read_identity']), dict(saved_choices()['read_identity'], passed=False, mismatched_nodes=1)
        self.assertFalse(gate.merge_read_identity([ok, bad])['passed']);self.assertTrue(gate.merge_read_identity([ok, ok])['passed'])
        two = [dict(saved_choices(), read_identity=ok, _gaps=np.zeros(0)), dict(saved_choices(), read_identity=bad, _gaps=np.zeros(0))]
        self.assertFalse(gate.merge_choices(two, value_bound=.008)['passed'])
        self.assertFalse(gate.merge_choices([dict(two[0], priors=dict(two[0]['priors'], passed=False)), two[0]], value_bound=.008)['passed'])
        for name, change in (('missing read identity', dict(read_identity=None)), ('read identity a list', dict(read_identity=[])), ('missing priors', dict(priors=None)),
                ('mismatched nodes', dict(read_identity=dict(ok, mismatched_nodes=3))), ('collisions', dict(read_identity=dict(ok, interned_collisions=1))),
                ('not injective', dict(read_identity=dict(ok, injective=False))), ('injective unrecorded', dict(read_identity={k: v for k, v in ok.items() if k != 'injective'})),
                ('prior mismatches', dict(priors=dict(saved_choices()['priors'], mismatched_nodes=2))), ('prior one-sided', dict(priors=dict(saved_choices()['priors'], one_sided=1)))):
            with self.subTest(name):self.assertFalse(gate.choice_verdict(dict(saved_choices(), **change), .008)['passed'])
        self.assertIn('no read identity recorded', gate.choice_verdict(dict(saved_choices(), read_identity=None), .008)['reasons'])   # named, not only failed
        for key in ('read_identity', 'priors'):
            with self.subTest(saved_null=key):
                report = saved_gate();report['checks']['f'][key] = None;code, out, _, _ = reverdicted(report)
                self.assertEqual((code, out['status'], out['checks']['f']['passed']), (1, 'FAIL', False))

    def test_read_identity_names_nodes_and_marks_the_origin(self):
        """A ply-2 parent under a priced cell reads two leaves swapped: the parent is the origin (its leaves named), the cell above it is not (its
        entry read a grid no oracle node has, where the parent was expected)."""
        frame, data = self.frame_data();tree = gate.OracleTree(frame, data)
        key = next(k for k in gate._parents(frame) if gate.pricer_record(frame, FakeBackupsN, *k) is not None and tree.solve(*gate._up(frame, k))[0] is not None
            and len(set(map(tuple, np.asarray(frame.leaf_value)[k[:2] + k[3:]].tolist()))) > 1)
        module = gate.mutant_backups([(key + (0, 0), key + (1, 0))], source=FAKE_BACKUPS_SOURCE)
        ri = gate.read_identity_n(frame, data, self.requests, backups=module);first = ri['first_mismatches']
        self.assertEqual((ri['mismatched_nodes'], ri['origins']), (2, 1))
        self.assertEqual([(m['origin'], (m['root'], m['world'], m['ply'], m['node'])) for m in first], [(True, key), (False, gate._up(frame, key))])
        self.assertTrue(all(e['read'][0] == 'leaf' and e['expected'][0] == 'leaf' for e in first[0]['entries']))
        (cell,) = first[1]['entries'];self.assertEqual((cell['read'][0], cell['read'][-1], cell['expected']), ('grid', 'no oracle node', ['node'] + list(key)))


    # ---- the mutation kill test

    def mutated(self, value_bound=None):
        """The prior k = 3 call with --mutation-test (the fake backups' source) judged at value_bound (default: the call's own run bound), cached:
        (evaluation, mutation report, log lines)."""
        def make():
            pricers = self.pricers(k_own=3, k_opp=3);lines = []
            depth2_pricer = gate.depth2_pass_pricer(FAKE_MODULE, self.ex, object(), capacity=32, actor_capacity=3, digest=True)
            with self.engine():
                ev, _ = gate.run_call_n(requests=self.requests, recipe=self.recipe, materializer=self.tables, params=PARAMS, pricers=pricers, depth2_pricer=depth2_pricer,
                    frame_pricer=self.frame_pricer(), oracle=self.oracle, sample=None, block=0, context=dict(block=0, call=0), backups=FakeBackupsN,
                    mutation=dict(source=FAKE_BACKUPS_SOURCE))
            bound = gate.summarize_run([ev], (2187,))['tolerance']['value_bound'] if value_bound is None else value_bound
            (report,) = gate.mutation_tests([ev], value_bound=bound, backups=FakeBackupsN, progress=lambda message, **k: lines.append(dict(k, message=message)))
            return ev, report, lines
        return self.cached(('mutated', value_bound), make)

    def test_every_mutant_is_killed_on_the_fakes(self):
        ev, m, lines = self.mutated()
        self.assertTrue(m['passed'], {n: (r.get('caught_by'), r.get('reason')) for n, r in m['mutants'].items()});self.assertNotIn(gate.MUTATION_INPUTS, ev)
        self.assertEqual((m['clean']['status'], m['clean_unchanged'], m['survivors'], m['material_survivors']), ('PASS', True, [], dict(scatter=0, backup=0)))
        self.assertEqual(self.judged(ev)[1][0], 'PASS');self.assertEqual(list(m['mutants']), [f'{p}-{k}-{v}' for p, k, v in gate.MUTANTS])
        self.assertEqual(len(m['mutants']), 10);self.assertEqual(m['mutants']['backup-turn1-systematic']['caught']['f']['read_identity']['passed'], False)
        priors = m['mutants']['backup-priors-systematic']['caught']['f']   # under learner-max no value moves: the exact prior check alone
        self.assertEqual((priors['priors']['passed'], priors['grid_within_bound'], priors['row_q_within_bound'], priors['read_identity']['passed']), (False, True, True, True))
        self.assertGreater(priors['priors']['mismatched_nodes'], 100)
        for name, r in m['mutants'].items():
            with self.subTest(name):
                self.assertEqual((r['exercised'], r['killed'], r['status'], r['guard_only'], r['value_bound']), (True, True, 'FAIL', [], m['value_bound']), r)
                if r['path'] == 'backup' and r['kind'] != 'priors':   # the frame is clean: only (f) sees a misread, by read identity and the grid gate
                    self.assertEqual(r['caught_by'], ['f']);self.assertFalse(r['caught']['f']['grid_within_bound']);self.assertFalse(r['caught']['f']['read_identity']['passed'])
                elif r['path'] == 'backup':self.assertEqual(r['caught_by'], ['f'])
                else:   # the reads are right (read identity passes); the values moved: (c), the guard report-only
                    self.assertIn('c', r['caught_by']);self.assertEqual(set(r['guard_modes_past_turn1'].values()), {'digest_certified'})
                    self.assertTrue((r['caught'].get('f') or {}).get('read_identity', dict(passed=True))['passed'])
                if 'f' in r['caught']:self.assertEqual(r['caught']['f']['read_identity']['interned_collisions'], 0)
                if r['variant'] == 'single':
                    pair = r['pair'];self.assertTrue(pair['choice_reversed']);self.assertGreater(pair['abs_value_difference'], 2 * m['value_bound'])
                    self.assertGreater(min(pair['clean_gap_pricer'], pair['clean_gap_oracle']), m['value_bound']);self.assertGreater(r['caught']['f']['non_exempt_mismatches'], 0)
                    self.assertEqual(r['pairs'], 1);self.assertIn('reverses', pair['why'])
                else:self.assertGreater(r['pairs'], 10) if r['kind'] != 'priors' else self.assertEqual((r['pairs'], r['exercised']), (0, True))
        cousin = m['mutants']['backup-cousin-single']['pair'];self.assertEqual((cousin['same_root_world'], cousin['fell_back_to_another_root_world']), (True, False))
        self.assertEqual(cousin['parent2'][:2], cousin['parent'][:2]);self.assertNotEqual(cousin['parent2'], cousin['parent'])
        for kind, variant in itertools.product(('sibling', 'cousin'), ('single', 'systematic')):   # both paths misplace the same pairs: (f) sees the same
            b, s = (m['mutants'][f'{path}-{kind}-{variant}']['caught']['f'] for path in ('backup', 'scatter'))
            self.assertEqual((b['non_exempt_mismatches'], b['max_abs_delta_row_q'], b['max_abs_delta_child_value']), (s['non_exempt_mismatches'], s['max_abs_delta_row_q'], s['max_abs_delta_child_value']))
        for path in gate.SWEEP_PATHS:
            sweep = m['sweep'][path];self.assertEqual((sweep['material_survivors'], sweep['claim_holds']), (0, True), sweep['first_material_survivors'][:2])
            self.assertEqual(sum(sweep['swaps'].values()), sum(m['sweep']['pairs']['drawn'].values()));self.assertGreater(sum(sweep['swaps'].values()), 5000)
        self.assertEqual(m['sweep']['scatter']['swaps'], m['sweep']['backup']['swaps'])
        messages = [x['message'] for x in lines];self.assertEqual(messages.count('MUTATION_TEST_MUTANT'), 10);self.assertEqual(messages.count('MUTATION_TEST_SWEEP'), 2)
        self.assertIn('MUTATION_TEST', messages);self.assertGreater(m['seconds'], 0.)

    def test_mutants_are_judged_at_the_runs_value_bound(self):
        """A run whose value bound (.004 here, another call's floor) exceeds this call's own (0): the clean call and every mutant are judged at it."""
        ev, m, _ = self.mutated(value_bound=.004)
        self.assertEqual((m['value_bound'], m['clean']['status']), (.004, 'PASS'));self.assertIn('the run', m['value_bound_source'])
        self.assertEqual({r['value_bound'] for r in m['mutants'].values()}, {.004});self.assertTrue(all(r['killed'] for r in m['mutants'].values()), m['survivors'])
        self.assertEqual(m['sweep']['pairs']['value_bound'], .004);self.assertEqual(m['material_survivors'], dict(scatter=0, backup=0))
        verdict, result = gate.mutant_verdict(dict(ev, actors=1), lanes=(2187,), value_bound=.004);self.assertEqual(result['tolerance']['value_bound'], .004)
        self.assertIs(gate.at_bound(ev, None), ev)

    def test_an_unpatchable_source_leaves_the_backup_mutants_unexercised(self):
        frame, data = self.frame_data()
        clean = gate.evaluate_n(frame, self._d(), self.requests, data, context={}, backups=FakeBackupsN, decision=True, choices=True)
        out = gate.mutation_test_call(frame, data, self.requests, self._d(), clean, backups=FakeBackupsN, source='def node_value():pass\n', sweep=20)
        self.assertFalse(out['passed']);self.assertEqual(sorted(out['survivors']), sorted(n for n in out['mutants'] if n.startswith('backup')))
        self.assertIn('occurs 0 times', out['mutants']['backup-sibling-single']['reason']);self.assertTrue(all(r['killed'] for n, r in out['mutants'].items() if n.startswith('scatter')))
        with self.assertRaises(gate.MutantPatchError):gate.mutant_backups([], source=FAKE_BACKUPS_SOURCE.replace('            if leaf:\n', '            if leaf :\n'))

    def test_sweep_survivor_accounting(self):
        """Bound 0: no material survivor on either path; scatter survivors are the equal-valued pairs, backup survivors those or pairs under refused
        parents. A bound above some differences, every sibling pair drawn: scatter survivors are exactly the pairs within it, backup survivors those
        plus the pairs beyond it under a refused parent, none material. The oracle holding a pair swapped (a broken clean state the swap repairs):
        a material survivor, the claim fails."""
        frame, data = self.frame_data();tree = gate.OracleTree(frame, data);count = dict(sibling=10 ** 7, cousin=2000)
        out = gate.swap_sweep(frame, data, FakeBackupsN, tree, 0., count=count, source=FAKE_BACKUPS_SOURCE)
        self.assertEqual(out['pairs']['drawn'], dict(sibling=out['pairs']['population']['sibling'], cousin=2000))
        for path in gate.SWEEP_PATHS:
            s = out[path]
            with self.subTest(path):
                self.assertEqual(sum(s['caught'].values()) + s['survivors'], sum(s['swaps'].values()));self.assertEqual(sum(s['survived'].values()), s['survivors'])
                self.assertEqual(len(s['survivor_rows']), s['survivors']);self.assertEqual((s['material_survivors'], s['claim_holds']), (0, True))
                self.assertTrue(all(r['abs_value_difference'] == 0. or not r['parent_priced'] for r in s['survivor_rows']))
        self.assertEqual(out['scatter']['survivors_parents_refused'] + sum(r['abs_value_difference'] == 0. for r in out['scatter']['survivor_rows'] if r['parent_priced']), out['scatter']['survivors'])
        self.assertGreater(out['backup']['survivors_parents_refused'], 0)
        self.assertEqual(gate.swap_sweep(frame, data, FakeBackupsN, tree, 0., count=count, source=FAKE_BACKUPS_SOURCE)['backup']['caught'], out['backup']['caught'])
        leaves = data['leaves'];index = leaves['index'].tolist();values = np.asarray(frame.leaf_value, np.float64)[tuple(leaves['index'].T)]
        scored = (np.asarray(leaves['status']) == 4) & (np.asarray(frame.leaf_status)[tuple(leaves['index'].T)] == 4);bound = .01
        groups = collections.defaultdict(list)
        for i in np.flatnonzero(scored).tolist():groups[tuple(index[i][:3])].append(values[i])
        priced = {key: FakeBackupsN.node_value(frame, key[0], key[1], 2, key[2]) is not None for key in groups}
        within = sum(abs(a - b) <= bound for g in groups.values() for a, b in itertools.combinations(g, 2))
        refused_beyond = sum(abs(a - b) > bound for key, g in groups.items() if not priced[key] for a, b in itertools.combinations(g, 2))
        wide = gate.swap_sweep(frame, data, FakeBackupsN, tree, bound, count=dict(sibling=10 ** 7), source=FAKE_BACKUPS_SOURCE)
        self.assertEqual(wide['scatter']['survived'].get('sibling', 0), within);self.assertGreater(within, out['scatter']['survived'].get('sibling', 0))
        self.assertEqual(wide['backup']['survived'].get('sibling', 0), within + refused_beyond)
        for path in gate.SWEEP_PATHS:
            self.assertTrue(all(r['abs_value_difference'] <= bound for r in wide[path]['survivor_rows'] if r['parent_priced']));self.assertEqual(wide[path]['material_survivors'], 0)
        planted = copy.deepcopy(data);head = planted['leaves']['head_value']
        i, j = next((i, j) for g in [[k for k in np.flatnonzero(scored).tolist() if tuple(index[k][:3]) == key] for key in groups if priced[key]]
            for i, j in itertools.combinations(g, 2) if head[i] != head[j])
        head[i], head[j] = head[j], head[i];d = abs(float(head[i]) - float(head[j]))
        for planted_bound in (0., d / 3):   # material at 2 x bound: d / 3 puts the pair between 2 x and 4 x the bound
            bad = gate.swap_sweep(frame, planted, FakeBackupsN, gate.OracleTree(frame, planted), planted_bound, count=dict(sibling=10 ** 7), paths=('scatter',))['scatter']
            self.assertFalse(bad['claim_holds']);self.assertIn(sorted([index[i], index[j]]), [sorted([r['a'], r['b']]) for r in bad['first_material_survivors']])


class DeclaredBoundsTests(FixtureN, unittest.TestCase):
    """--declared-bounds: the file fails closed; the verdict takes the declared numbers (a run beyond B1's bounds passes looser declared ones
    and one inside them fails tighter ones; declaring the frozen numbers reproduces the frozen verdict); the near-tie Q bound is (e)'s alone;
    the flags are champion only (the command-line round trip runs on the champion: test_gpu_engine_champion_evaluator.ChampionCommandLineTests);
    without the flag nothing changes (the head override's roots here stand for B1's path)."""
    _frame, _d, _data, _n2_frame = N.GateNTests._frame, N.GateNTests._d, N.GateNTests._data, N.GateNTests._n2_frame
    main = N.CommandLineRunTests.main

    def roots(self):
        """The CLI fixture's roots under the head override's evaluator header (kind v2, DECLARED_ARCHIVE): what load_roots resolves for HEAD_FLAGS."""
        roots = N.CommandLineRunTests.roots(self);roots.evaluator = dict(override=True, kind='v2', identity=['v2'], precision='bfloat16',
            archive_sha256=DECLARED_ARCHIVE, checkpoint_sha256=DECLARED_CHECKPOINT, arm_checked=True, dropped_roots=[]);return roots

    def evaluated(self, *, leaf=0., t1_logits=0., ply_logits=0., root_logits=0., twin=False):
        """One call of the stand-in depth-3 surface with `leaf` added to head-valued leaves, `t1_logits` / `ply_logits` to the turn-1 cells' /
        ply-2 nodes' own priors, `root_logits` to every lane of the root opponent's prior, (e) and (f) through the fake backups, a synthetic
        depth-1 floor FLOOR (frozen value bound .008) and one actor. twin: the unshifted frame stands in for the full-width depth-2 pass
        (depth2_pass_priors) and for a block-invariance twin, so every prior site compares the shifts."""
        base = self.cached('frame', self._frame);data = self.cached('data', lambda: self._data(base));frame = copy.deepcopy(base);level = frame.nodes[0]
        frame.leaf_value[np.asarray(frame.leaf_status) == 4] += np.float32(leaf);frame.t1_own_logits[np.asarray(frame.t1_status) != 0] += np.float32(t1_logits)
        level.own_logits[np.asarray(level.status) != 0] += np.float32(ply_logits)
        if root_logits:frame.root_peer_logits[...] += np.float32(root_logits)
        ev = gate.evaluate_n(frame, self._d(), self.requests, data, context={}, backups=FakeBackupsN, decision=True, choices=True)
        ev['c']['depth1_noise_floor'] = gate.g2._numeric([[0]], [FLOOR], [0.]);ev['spreads'] = [SPREAD] * 10;ev['actors'] = 1
        if twin:ev['c'].update(depth2_pass_priors(base, frame));ev['invariance'] = [gate.g2.block_invariance(gate.frame_arrays(base), gate.frame_arrays(frame))]
        return ev

    def judge(self, ev, declared=None):
        result = gate.summarize_run([copy.deepcopy(ev)], (2187,), declared=declared);return result, gate.gate_status_n(result)[0], result['checks']['c']['quantities']

    def test_the_declaration_parses_and_fails_closed(self):
        d = declared();self.assertEqual((d['content'], d['path']), (declaration(), d['path']));self.assertEqual(len(d['sha256']), 64)
        self.assertEqual(gate.declared_numbers(d), dict(value_bound=.008, turn1_logit_bound=.1, deeper_logit_bound=None, deeper_logit_factor=2., near_tie_q_bound=None))
        self.assertEqual(gate.declared_numbers(declared(bounds=bounds(deeper_logit_rule=None, deeper_logit_bound=.3, near_tie_q_bound=.02)))['deeper_logit_bound'], .3)
        refused = {f'missing {k}': (declaration(**{k: None}), f'declaration.{k}: missing') for k in gate.DECLARED_FIELDS}
        refused.update(
            unknown_key=(dict(declaration(), comment='x'), 'declaration.comment: unknown key'), misspelt_bound=(declaration(bounds=dict(bounds(), value_bnd=1)), 'unknown key'),
            no_value_bound=(declaration(bounds=bounds(value_bound=None)), 'bounds.value_bound: missing'),
            no_turn1_logit_bound=(declaration(bounds=bounds(turn1_logit_bound=None)), 'bounds.turn1_logit_bound: missing'),
            no_deeper=(declaration(bounds=bounds(deeper_logit_rule=None)), 'exactly one'), both_deeper=(declaration(bounds=bounds(deeper_logit_bound=.2)), 'exactly one'),
            zero=(declaration(bounds=bounds(value_bound=0)), 'finite number > 0'), boolean=(declaration(bounds=bounds(turn1_logit_bound=True)), 'finite number > 0'),
            negative_factor=(declaration(bounds=bounds(deeper_logit_rule=dict(factor=-2))), 'factor: a finite number > 0'),
            text_bound=(declaration(bounds=bounds(near_tie_q_bound='.01')), 'near_tie_q_bound: a finite number > 0'),
            schema=(declaration(schema='v0'), 'schema'), kind=(declaration(evaluator=dict(declaration()['evaluator'], kind='v2')), 'evaluator.kind'),
            alias_precision=(declaration(evaluator=dict(declaration()['evaluator'], precision='bf16')), 'evaluator.precision'),
            archive=(declaration(evaluator=dict(declaration()['evaluator'], archive_sha256='7' * 63)), 'evaluator.archive_sha256'),
            naive_time=(declaration(declared_at='2026-11-01T00:00:00'), 'declared_at'), other_zone=(declaration(declared_at='2026-11-01T00:00:00+01:00'), 'declared_at'),
            empty_derivation=(declaration(derivation='  '), 'derivation: non-empty'), empty_author=(declaration(declared_by=''), 'declared_by: non-empty'),
            measurement=(declaration(source_measurement=dict(path='/shared/m.json')), 'source_measurement.sha256: missing'),
            measurement_sha=(declaration(source_measurement=dict(path='/shared/m.json', sha256='x')), 'source_measurement.sha256: a 64-hex'),
            future=(declaration(declared_at='2999-01-01T00:00:00Z'), 'after the run\'s start'), not_an_object=([1], 'a JSON object'))
        for case, (content, message) in refused.items():
            with self.subTest(case):
                with self.assertRaises(gate.DeclaredBoundsError) as raised:load(committed(content))
                self.assertIn(message, str(raised.exception))
        record = committed(declaration());text = json.dumps(declaration())
        for case, (call, message) in dict(sha_mismatch=(lambda: load(record, sha256='0' * 64), 'differs from --declared-bounds-sha256'),
                absent=(lambda: load(record, path=record.path.with_name('absent.json')), 'cannot read'),
                duplicate=(lambda: load(committed(raw=text.replace('{', '{"derivation": "a", ', 1).encode())), 'duplicate'),
                nan=(lambda: load(committed(raw=text.replace('0.008', 'NaN').encode())), 'NaN is not a JSON number'),
                not_json=(lambda: load(committed(raw=b'{"schema": ')), 'not a JSON declaration'),
                not_utf8=(lambda: load(committed(raw=b'\xff\xfe{}')), 'not UTF-8')).items():
            with self.subTest(case):
                with self.assertRaises(gate.DeclaredBoundsError) as raised:call()
                self.assertIn(message, str(raised.exception))

    def test_the_declaration_is_a_committed_record(self):
        """The file must lie under DECLARED_SUBDIR of the checkout; its git blob sha1 (git hash-object) and the commit are recorded and, in a git
        work tree, the commit must exist (a made-up one is refused), hold exactly this blob there and be an ancestor of HEAD, with the file tracked
        and unmodified at HEAD; "not verifiable here" only without git, without a work tree of the checkout, or in a shallow clone lacking the
        commit. Every case of the re-review's prereg.py, then the refusals beyond it, then the source measurement."""
        record = committed(declaration());d = load(record)
        self.assertEqual((d['repo_path'], d['commit'], d['git_verification']['status']), (gate.DECLARED_SUBDIR + '/champion.json', record.commit, 'verified'))
        self.assertIn('ancestor of HEAD', d['git_verification']['detail']);self.assertIn('tracked and unmodified at HEAD', d['git_verification']['detail'])
        self.assertEqual(d['git_blob_sha1'], git(record.root, 'hash-object', str(record.path)));self.assertEqual(d['source_measurement_verification']['status'], 'not verifiable here')
        raw = json.dumps(declaration(), indent=1).encode();sha = hashlib.sha256(raw).hexdigest();made_up = 'e' * 40
        edited = json.dumps(declaration(bounds=bounds(value_bound=.5)), indent=1).encode()
        def repo():
            root = Path(tempfile.mkdtemp(prefix='depthn-declared-repo-'));git(root, 'init', '-q');(root / gate.DECLARED_SUBDIR).mkdir(parents=True)
            (root / 'README').write_text('x');git(root, 'add', '.');git(root, 'commit', '-q', '-m', 'base');return root
        def place(root, data=raw, name='champion.json'):
            path = root / gate.DECLARED_SUBDIR / name;path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(data);return path
        def commit(root, message='declare'):
            git(root, 'add', '-A');git(root, 'commit', '-q', '-m', message);return git(root, 'rev-parse', 'HEAD')
        r1 = repo();p1 = place(r1)   # prereg 1, 2, 7: never committed, in a full repository
        r3 = repo();p3 = place(r3);c3 = commit(r3);p3.write_bytes(edited)   # prereg 3, 3b: committed, then edited in the work tree
        r4 = repo();git(r4, 'checkout', '-q', '-b', 'after-the-failure');p4 = place(r4);c4 = commit(r4)   # prereg 4: on a side branch, HEAD there
        r5 = Path(tempfile.mkdtemp(prefix='depthn-nogit-'));(r5 / gate.DECLARED_SUBDIR).mkdir(parents=True);p5 = place(r5)   # prereg 5: no git on PATH
        r6 = repo();outside = Path(tempfile.mkdtemp(prefix='depthn-outside-')) / 'champion.json';outside.write_bytes(raw)
        link = r6 / gate.DECLARED_SUBDIR / 'link.json';link.symlink_to(outside)   # prereg 6: a symlink under the directory to outside the checkout
        outer = repo();inner = outer / 'checkout';(inner / gate.DECLARED_SUBDIR).mkdir(parents=True);p8 = place(inner);c8 = commit(outer)   # prereg 8: nested
        rm = repo();pm = place(rm);cm = commit(rm);pm.write_bytes(edited);commit(rm, 'revise');pm.write_bytes(raw)   # the commit's bytes, HEAD's another revision
        ru = repo();pu = place(ru);cu = commit(ru);git(ru, 'rm', '-q', '--cached', str(pu.relative_to(ru)));git(ru, 'commit', '-q', '-m', 'untrack')   # untracked at HEAD, left in the work tree
        rs = repo();trunk = git(rs, 'rev-parse', '--abbrev-ref', 'HEAD');git(rs, 'checkout', '-q', '-b', 'side');ps = place(rs);cs = commit(rs, 'side')
        git(rs, 'checkout', '-q', trunk);place(rs);commit(rs, 'trunk')   # the same file on HEAD's line, the given commit off it
        origin = repo();po = place(origin);co = commit(origin);(origin / 'later').write_text('y');commit(origin, 'later')
        clone = Path(tempfile.mkdtemp(prefix='depthn-shallow-')) / 'clone';git(origin, 'clone', '-q', '--depth', '1', f'file://{origin}', str(clone))   # lacks co
        refused = dict(
            prereg_1_untracked_made_up_commit=(lambda: gate.load_declared(p1, sha, made_up, root=r1), 'absent: commit ' + made_up),
            prereg_2_untracked_real_head=(lambda: gate.load_declared(p1, sha, git(r1, 'rev-parse', 'HEAD'), root=r1), 'absent: commit'),
            prereg_3_edited_real_commit=(lambda: gate.load_declared(p3, hashlib.sha256(edited).hexdigest(), c3, root=r3), 'mismatch'),
            prereg_3b_edited_made_up_commit=(lambda: gate.load_declared(p3, hashlib.sha256(edited).hexdigest(), made_up, root=r3), 'absent: commit ' + made_up),
            prereg_6_symlink_to_outside=(lambda: gate.load_declared(link, sha, made_up, root=r6), f'committed under {gate.DECLARED_SUBDIR}/'),
            prereg_7_dotdot_made_up_commit=(lambda: gate.load_declared(r1 / gate.DECLARED_SUBDIR / '..' / 'declared_bounds' / 'champion.json', sha, made_up, root=r1),
                'absent: commit ' + made_up),
            made_up_commit_for_a_committed_file=(lambda: load(record, commit=made_up), 'absent: commit ' + made_up),
            modified_against_head=(lambda: gate.load_declared(pm, sha, cm, root=rm), 'modified'),
            untracked_at_head=(lambda: gate.load_declared(pu, sha, cu, root=ru), 'untracked'),
            commit_off_heads_line=(lambda: gate.load_declared(ps, sha, cs, root=rs), 'not an ancestor of HEAD'),
            short_commit=(lambda: load(record, commit=record.commit[:12]), 'full lowercase hex commit id'),
            outside=(lambda: gate.load_declared(*written(declaration()), record.commit, root=record.root), f'committed under {gate.DECLARED_SUBDIR}/'))
        for case, (call, message) in refused.items():
            with self.subTest(case):
                with self.assertRaises(gate.DeclaredBoundsError) as raised:call()
                self.assertIn(message, str(raised.exception))
        with patch.dict(os.environ, {'PATH': '/nonexistent'}):no_git = gate.load_declared(p5, sha, made_up, root=r5)
        accepted = dict(prereg_4_side_branch_head=(gate.load_declared(p4, sha, c4, root=r4), 'verified', 'ancestor of HEAD'),   # the integration branch is the audit's
            prereg_5_no_git_on_path=(no_git, 'not verifiable here', 'git is not available'),
            prereg_8_nested_checkout=(gate.load_declared(p8, sha, c8, root=inner), 'not verifiable here', 'is not the top of a git work tree'),
            no_repository=(gate.load_declared(p5, sha, made_up, root=r5), 'not verifiable here', 'is not the top of a git work tree'),
            shallow_clone_lacking_the_commit=(gate.load_declared(clone / gate.DECLARED_SUBDIR / 'champion.json', sha, co, root=clone), 'not verifiable here', 'shallow'))
        for case, (result, status, detail) in accepted.items():
            with self.subTest(case):self.assertEqual(result['git_verification']['status'], status);self.assertIn(detail, result['git_verification']['detail'])
        self.assertTrue(gate.provisional_reasons(no_git['git_verification'], no_git['source_measurement_verification'])[0].startswith('provisional: git_verification not verifiable here'))
        elsewhere = record.root / 'gpu_engine' / 'champion.json';elsewhere.write_bytes(record.path.read_bytes());inside = commit(record.root, 'the same file outside the directory')
        with self.assertRaises(gate.DeclaredBoundsError) as raised:load(record, path=elsewhere, commit=inside)
        self.assertIn(f'committed under {gate.DECLARED_SUBDIR}/', str(raised.exception))
        measurement = Path(tempfile.mkdtemp(prefix='depthn-measure-')) / 'spread.json';measurement.write_bytes(b'{"spread": 0.003}\n')
        good = dict(path=str(measurement), sha256=hashlib.sha256(measurement.read_bytes()).hexdigest())
        self.assertEqual(declared(source_measurement=good)['source_measurement_verification']['status'], 'verified')
        with self.assertRaises(gate.DeclaredBoundsError) as raised:declared(source_measurement=dict(good, sha256='9' * 64))
        self.assertIn('source_measurement', str(raised.exception))

    def test_the_declaration_binds_one_evaluator(self):
        run = dict(override=True, kind='champion', archive_sha256=DECLARED_ARCHIVE, precision='bfloat16', checkpoint_sha256=DECLARED_CHECKPOINT)
        self.assertEqual(gate.evaluator_mismatches(declaration(), run), [])
        for key, value in (('kind', 'v2'), ('archive_sha256', '6' * 64), ('precision', 'float32')):
            with self.subTest(key):
                mismatched = gate.evaluator_mismatches(declaration(), dict(run, **{key: value}))
                self.assertEqual(len(mismatched), 1);self.assertIn(f'evaluator.{key}', mismatched[0])
        pinned = declaration(evaluator=dict(declaration()['evaluator'], checkpoint_sha256=DECLARED_CHECKPOINT))
        self.assertEqual(gate.evaluator_mismatches(pinned, run), []);self.assertIn('checkpoint_sha256', gate.evaluator_mismatches(pinned, dict(run, checkpoint_sha256='5' * 64))[0])
        self.assertIn('no champion evaluator', gate.evaluator_mismatches(declaration(), dict(run, override=False))[0])
        base = ['--out', 'o', '--d3-root', 'b', '--chunk', '0', '--declared-bounds', 'd.json', '--declared-bounds-sha256', 'a' * 64, '--declared-bounds-commit', 'b' * 40]
        self.assertEqual(gate.args_identity(gate.parse(base + list(CHAMPION_FLAGS))[1]), run)
        self.assertEqual(gate.args_identity(gate.parse(base + list(CHAMPION_FLAGS) + ['--champion-precision', 'f32'])[1]), dict(run, precision='float32'))
        for flags in ((), HEAD_FLAGS):self.assertEqual(gate.args_identity(gate.parse(base[:6] + list(flags))[1])['override'], False)   # no champion: never applies

    def test_flags_are_refused_off_the_champion_and_in_incompatible_modes(self):
        out = str(Path(tempfile.gettempdir()) / f'depthn-gate-never-created-{uuid.uuid4().hex}')
        flags = ['--declared-bounds', 'd.json', '--declared-bounds-sha256', 'a' * 64, '--declared-bounds-commit', 'b' * 40]
        live = ['--out', out, '--d3-root', 'b', '--chunk', '0']
        cases = dict(alone=(live + flags[:2] + list(CHAMPION_FLAGS), 'go together'), sha_alone=(live + flags[2:4] + list(CHAMPION_FLAGS), 'go together'),
            no_commit=(live + flags[:4] + list(CHAMPION_FLAGS), 'go together'), bad_commit=(live + flags[:5] + ['B' * 40] + list(CHAMPION_FLAGS), 'full lowercase hex'),
            bad_sha=(live + flags[:3] + ['A' * 64] + flags[4:] + list(CHAMPION_FLAGS), '64-hex'), bank_actors=(live + flags, 'needs --champion-archive'),
            head_override=(live + flags + list(HEAD_FLAGS), 'champion only'),
            diagnostics=(live + flags + list(CHAMPION_FLAGS) + ['--depth3-diagnostics'], '--depth3-diagnostics'),
            diagnostics_from=(['--out', out, '--diagnostics-from', 'x'] + flags + list(CHAMPION_FLAGS), 'live gate run'),
            mutation_from=(['--out', out, '--mutation-from', 'x'] + flags, 'live gate run'), reverdict=(['--reverdict', 'gate.json'] + flags, 'takes no other option'),
            reverdict_bad_commit=(['--reverdict', 'gate.json', '--declared-bounds-commit', 'abc'], 'full lowercase hex'))
        for case, (argv, message) in cases.items():
            with self.subTest(case):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:gate.main(argv)
                self.assertEqual(raised.exception.code, 2);self.assertIn(message, stderr.getvalue())
        for extra in ((), ('--n2-equivalence',), ('--mutation-test',), ('--n2-equivalence', '--block-invariance')):
            self.assertEqual(gate.parse(live + flags + list(CHAMPION_FLAGS) + list(extra))[1].declared_bounds, 'd.json')
        self.assertEqual(gate.parse(['--reverdict', 'gate.json', '--declared-bounds-commit', 'b' * 40, '--declared-bounds-sha256', 'a' * 64])[1].reverdict, 'gate.json')
        record = committed(declaration())   # main refuses before anything is created or loaded: the file's sha256, its commit, then the flags' evaluator
        for argv, message in (([record.sha256[:-1] + '0', record.commit], 'differs from --declared-bounds-sha256'), ([record.sha256, record.commit], 'evaluator.archive_sha256')):
            head = list(CHAMPION_FLAGS);head[3] = '6' * 64
            with patch.object(gate, 'load_roots', side_effect=AssertionError('loaded')), patch.object(gate, 'log', lambda *a, **k: None), patch.object(gate, 'REPO_ROOT', record.root), \
                    self.assertRaises(SystemExit) as raised:gate.main(live + ['--declared-bounds', str(record.path), '--declared-bounds-sha256', argv[0], '--declared-bounds-commit', argv[1]] + head)
            self.assertIn(message, str(raised.exception.code));self.assertIn('fail closed', str(raised.exception.code));self.assertFalse(Path(out).exists())

    def test_the_verdict_takes_the_declared_bounds(self):
        """Turn-1 priors .12 off fail B1's 0.1 and pass a declared .15; leaves .006 off pass B1's .008 and fail a declared .005; ply-2 priors .12 off
        (turn 1 .07) pass the frozen deeper .14 and fail a declared .11 or a declared rule of factor 1.5 (.105), and pass factor 2 (.14). The
        measured floors and the frozen rule's numbers are reported beside the declared ones."""
        cases = (('turn1_logits', dict(t1_logits=.12), bounds(turn1_logit_bound=.15), ('FAIL', 'PASS'), 't1_own_logits', .15, 'declared_turn1_logit_bound'),
            ('leaf_values', dict(leaf=.006), bounds(value_bound=.005), ('PASS', 'FAIL'), 'leaf_value', .005, 'declared_value_bound'),
            ('deeper_bound', dict(t1_logits=.07, ply_logits=.12), bounds(deeper_logit_rule=None, deeper_logit_bound=.11), ('PASS', 'FAIL'), 'ply2_own_logits', .11,
                'declared_deeper_logit_bound'),
            ('deeper_rule_tighter', dict(t1_logits=.07, ply_logits=.12), bounds(deeper_logit_rule=dict(factor=1.5)), ('PASS', 'FAIL'), 'ply2_own_logits', None,
                'declared_deeper_logit_bound'),
            ('deeper_rule_frozen_factor', dict(t1_logits=.07, ply_logits=.12), bounds(), ('PASS', 'PASS'), 'ply2_own_logits', None, 'declared_deeper_logit_bound'))
        for case, noise, numbers, (frozen, wanted), name, bound, label in cases:
            with self.subTest(case):
                ev = self.evaluated(**noise);d = declared(bounds=numbers);result, status, c = self.judge(ev)
                self.assertEqual(status, frozen, failing(result))
                result, status, c = self.judge(ev, d);m = c[name];tolerance = result['tolerance']
                self.assertEqual(status, wanted, failing(result));self.assertEqual((m['bound_rule'], m['passed']), (label, wanted == 'PASS'))
                if wanted == 'FAIL':self.assertTrue('c' in failing(result) and set(failing(result)) <= {'c', 'e', 'f'}, failing(result))   # (e)/(f) share the value bound
                if bound is not None:self.assertAlmostEqual(m['bound'], bound, places=12)
                self.assertEqual((tolerance['value_bound'], tolerance['logit_bound']), (numbers['value_bound'], numbers['turn1_logit_bound']))
                section = tolerance['declared_bounds'];self.assertEqual(section['sha256'], d['sha256'])
                self.assertEqual(section['frozen_rule_would_give']['value_bound'], 2 * float(np.float32(FLOOR)));self.assertEqual(section['frozen_rule_would_give']['logit_bound'], .1)
                self.assertEqual(section['measured']['depth1_noise_floor'], float(np.float32(FLOOR)));self.assertEqual(section['measured']['turn1_logit_floor'], tolerance['turn1_logit_floor'])
                self.assertEqual(tolerance['frozen_rule']['rule'], gate.g2.TOLERANCE_RULE);self.assertEqual(tolerance['rule'], gate.DECLARED_TOLERANCE_RULE)
                if case == 'deeper_rule_tighter':self.assertAlmostEqual(m['bound'], 1.5 * tolerance['turn1_logit_floor']);self.assertIn('1.5 x turn1_logit_floor', tolerance['deeper_logit_bound_source'])
                for other, q in c.items():
                    if other not in gate.g2.FLOORS:self.assertTrue(q['bound_rule'].startswith('declared_'), other)
                self.assertEqual(result['checks']['f']['value_bound'], numbers['value_bound'])
                self.assertEqual(gate.tolerance_line_n(tolerance)['declared_bounds_sha256'], d['sha256'])

    def test_declaring_the_frozen_numbers_reproduces_the_frozen_verdict(self):
        """value_bound = 2 x the run's floor, turn1_logit_bound = 0.1, deeper_logit_rule factor 2: every check's verdict, every (c) quantity's
        bound and counts and (e)'s and (f)'s numbers are the frozen rule's: only the numbers are declared, and the one rule a declaration brings,
        GUARD_DECLARED_RULE, changes no verdict here, only the turn-1 values' guard modes (gated -> digest_certified: the stand-in's digests match
        and it prices with one parameter set)."""
        d = declared(bounds=bounds(value_bound=2 * float(np.float32(FLOOR))))
        keep = ('passed', 'bound', 'within_bound', 'beyond_bound', 'ratio_to_bound', 'within_guard', 'beyond_guard', 'first_beyond')
        for noise in (dict(), dict(leaf=.006), dict(leaf=.009), dict(t1_logits=.07, ply_logits=.12), dict(t1_logits=.12), dict(t1_logits=.07, ply_logits=.15)):
            with self.subTest(**noise):
                ev = self.evaluated(**noise);(frozen, status, c), (mine, status2, c2) = self.judge(ev), self.judge(ev, d)
                self.assertEqual(status2, status);self.assertEqual({k: v['passed'] for k, v in mine['checks'].items()}, {k: v['passed'] for k, v in frozen['checks'].items()})
                self.assertEqual({n: {k: m.get(k) for k in keep} for n, m in c2.items()}, {n: {k: m.get(k) for k in keep} for n, m in c.items()})
                self.assertEqual({n: (c[n].get('guard_mode'), m.get('guard_mode')) for n, m in c2.items() if m.get('guard_mode') != c[n].get('guard_mode')},
                    {'t1_value_head': ('gated', 'digest_certified')})
                for key in ('value_bound', 'logit_bound', 'guard_bound', 'q_noise_bound', 'deeper_logit_bound', 'turn1_logit_floor', 'depth1_noise_floor'):
                    self.assertEqual(mine['tolerance'][key], frozen['tolerance'][key], key)
                for check, keys in (('e', ('q_noise_bound', 'near_tie_roots', 'near_tie_flips', 'max_abs_delta_q')), ('f', ('value_bound', 'near_ties', 'near_tie_flips',
                        'nodes_grid_beyond_bound', 'nodes_row_q_beyond_bound'))):
                    self.assertEqual({k: mine['checks'][check][k] for k in keys}, {k: frozen['checks'][check][k] for k in keys}, check)

    def test_the_near_tie_q_bound_is_e_alone(self):
        """Leaves .006 off: (e)'s max |dQ| lies above a declared near-tie Q bound .004 (e fails), while (f) keeps the value bound .008 (passes);
        undeclared, the near-tie Q bound is the value bound."""
        ev = self.evaluated(leaf=.006)
        result, status, _ = self.judge(ev, declared(bounds=bounds(near_tie_q_bound=.004)));e, f = result['checks']['e'], result['checks']['f']
        self.assertEqual((status, set(failing(result))), ('FAIL', {'e'}));self.assertGreater(e['max_abs_delta_q'], .004)
        self.assertEqual((e['q_noise_bound'], result['tolerance']['q_noise_bound'], f['value_bound'], f['passed']), (.004, .004, .008, True))
        self.assertEqual(result['tolerance']['declared_bounds']['applied']['q_noise_bound'], .004);self.assertIn('near_tie_q_bound', result['tolerance']['q_noise_rule'])
        result, status, _ = self.judge(ev, declared());self.assertEqual((status, result['checks']['e']['q_noise_bound']), ('PASS', .008))
        self.assertIn('(= the value bound)', result['tolerance']['q_noise_rule'])
        self.assertEqual(result['tolerance']['declared_bounds']['applied']['q_noise_bound'], .008)

    def test_n2_equivalence_takes_the_declared_bounds(self):
        """summarize_n2 under a declaration: priors .12 off fail 0.1 and pass a declared .15; the near-tie Q bound is (e)'s q_noise_bound."""
        frame = fake_as_depth2(self.cached('n2', self._n2_frame));other = copy.deepcopy(frame);other.t1_own_logits[np.asarray(other.t1_status) != 0] += np.float32(.12)
        found, floats = gate.n2_compare(frame, other);e = gate.decide_roots(frame, frame, self.requests, depth2_backups)
        call = dict(a=found, floats=floats, c=dict(depth1_noise_floor=gate.g2._numeric([[0]], [FLOOR], [0.])), e=e, spreads=[SPREAD] * 10)
        self.assertEqual(gate.summarize_n2({'x': [call]})['x']['floats']['t1_own_logits']['passed'], False)
        report = gate.summarize_n2({'x': [copy.deepcopy(call)]}, declared=declared(bounds=bounds(turn1_logit_bound=.15, near_tie_q_bound=.01)))['x']
        self.assertEqual((report['floats']['t1_own_logits']['bound'], report['floats']['t1_own_logits']['passed']), (.15, True))
        self.assertEqual((report['tolerance']['value_bound'], report['tolerance']['q_noise_bound'], report['e']['q_noise_bound']), (.008, .01, .01))
        self.assertEqual(report['tolerance']['frozen_rule']['logit_bound'], .1);self.assertIn('declared_bounds', report['tolerance'])

    def test_the_root_logit_bound_parses_and_fails_closed(self):
        """bounds.root_logit_bound is optional: declared, declared_numbers carries it; absent, declared_numbers has no such key (the root keeps
        turn1_logit_bound). 0, a negative, NaN, inf, a bool and a text are refused as every bound is; NaN / Infinity in the file are not JSON
        numbers; an unknown key is still refused."""
        self.assertEqual(gate.declared_numbers(declared(bounds=bounds(root_logit_bound=.05)))['root_logit_bound'], .05)
        self.assertNotIn('root_logit_bound', gate.declared_numbers(declared()))
        for value in (.05, 1, 1e-9):self.assertEqual(gate.declared_problems(declaration(bounds=bounds(root_logit_bound=value))), [], value)
        for case, value in dict(zero=0, zero_float=0., negative=-.05, nan=float('nan'), inf=float('inf'), boolean=True, false=False, text='.05').items():
            with self.subTest(case):
                self.assertEqual(gate.declared_problems(declaration(bounds=bounds(root_logit_bound=value))), ['bounds.root_logit_bound: a finite number > 0'])
                if case not in ('nan', 'inf'):   # json.dumps would write them as NaN / Infinity: the file case below
                    with self.assertRaises(gate.DeclaredBoundsError) as raised:load(committed(declaration(bounds=bounds(root_logit_bound=value))))
                    self.assertIn('bounds.root_logit_bound: a finite number > 0', str(raised.exception))
        for literal in ('NaN', 'Infinity', '-Infinity'):
            raw = json.dumps(declaration(bounds=bounds(root_logit_bound=.05))).replace('0.05', literal).encode()
            with self.subTest(literal), self.assertRaises(gate.DeclaredBoundsError) as raised:load(committed(raw=raw))
            self.assertIn(f'{literal} is not a JSON number', str(raised.exception))
        self.assertIn('bounds.root_bound: unknown key', gate.declared_problems(declaration(bounds=bounds(root_bound=.05))))

    def test_the_root_logit_bound_reaches_every_root_site_and_no_turn1_site(self):
        """Turn-1 own priors and the root opponent's prior .1 off at every prior site ((b) against D3, (c) against the oracle, depth2_pass and
        block invariance): turn1_logit_bound .15 passes them all, .05 fails them all. A declared root_logit_bound .05 beside turn1_logit_bound
        .15 fails every root site and nothing else: each root record is the one turn1_logit_bound .05 gives (bound_rule
        declared_root_logit_bound aside), every other record, (e) and (f) are the ones without it. Swapped (turn1_logit_bound .05,
        root_logit_bound .15): every turn-1 site fails as under .05 and every root site passes as under .15. The tolerance, its applied
        section, its log line and the table record root_logit_bound; without one nothing names it."""
        ev = self.evaluated(t1_logits=.1, root_logits=.1, twin=True)
        judged = lambda **numbers: self.judge(ev, declared(bounds=bounds(**numbers)))[:2]
        (loose, _), (tight, _) = judged(turn1_logit_bound=.15), judged(turn1_logit_bound=.05)
        (root_tight, root_tight_status), (root_loose, root_loose_status) = judged(turn1_logit_bound=.15, root_logit_bound=.05), judged(turn1_logit_bound=.05, root_logit_bound=.15)
        plain = lambda m: {k: v for k, v in m.items() if k != 'bound_rule'};shifted = ROOT_SITES + tuple(s for s in TURN1_SITES if 'own' in s[1])
        for check, name in ROOT_SITES + TURN1_SITES:   # the shifted sites compare the shift and fail .05; the turn-1 peer prior is not shifted
            m = site(loose, check, name);self.assertGreater(m['compared'], 0, (check, name))
            if (check, name) in shifted:self.assertAlmostEqual(m['max_abs_difference'], .1, delta=1e-3, msg=(check, name))
            self.assertEqual((m['passed'], site(tight, check, name)['passed']), (True, (check, name) not in shifted), (check, name))
        self.assertEqual(failing(loose), {});self.assertEqual(set(failing(tight)), {'b', 'c', 'block_invariance'})
        for result, wanted, other, status in ((root_tight, .05, loose, root_tight_status), (root_loose, .15, tight, root_loose_status)):
            for check, name in ROOT_SITES:   # the root at root_logit_bound, exactly as the turn-1 bound of that number gives it
                m = site(result, check, name);self.assertEqual((m['bound'], m['bound_rule'], m['passed']), (wanted, 'declared_root_logit_bound', wanted > .1), (check, name))
                self.assertEqual(plain(m), plain(site(tight if wanted == .05 else loose, check, name)), (check, name))
            for check, name in TURN1_SITES:   # every turn-1 prior as without root_logit_bound
                self.assertEqual(site(result, check, name), site(other, check, name), (check, name))
            for check in ('b', 'c', 'block_invariance'):   # every other quantity untouched
                self.assertEqual({n: m for n, m in result['checks'][check][SITE_KEYS[check]].items() if (check, n) not in ROOT_SITES},
                    {n: m for n, m in other['checks'][check][SITE_KEYS[check]].items() if (check, n) not in ROOT_SITES}, check)
            self.assertEqual((result['checks']['e'], result['checks']['f']), (other['checks']['e'], other['checks']['f']))
            tolerance = result['tolerance'];self.assertEqual((tolerance['root_logit_bound'], tolerance['declared_bounds']['applied']['root_logit_bound']), (wanted, wanted))
            self.assertEqual(gate.tolerance_line_n(tolerance)['root_logit_bound'], wanted);self.assertEqual(tolerance['logit_bound'], .15 if wanted == .05 else .05)
            rows = {(r['check'], r['quantity']): r for r in tolerance['table']}
            for key in ROOT_SITES:self.assertEqual((rows[key]['bound'], rows[key]['passed']), (wanted, wanted > .1), key)
        self.assertEqual((root_tight_status, set(failing(root_tight))), ('FAIL', {'b', 'c', 'block_invariance'}))   # the root sites alone fail
        self.assertEqual((root_loose_status, set(failing(root_loose))), ('FAIL', {'c', 'block_invariance'}))   # (b) compares no turn-1 prior
        for result in (loose, tight):
            self.assertNotIn('root_logit_bound', result['tolerance']);self.assertNotIn('root_logit_bound', result['tolerance']['declared_bounds']['applied'])
            self.assertNotIn('root_logit_bound', gate.tolerance_line_n(result['tolerance']));self.assertNotIn('declared_root_logit_bound', json.dumps(result, default=str))
            for check, name in ROOT_SITES:self.assertEqual(site(result, check, name)['bound'], result['tolerance']['logit_bound'], (check, name))

    def test_a_root_logit_bound_equal_to_turn1_logit_bound_is_the_declaration_without_it(self):
        """root_logit_bound declared equal to turn1_logit_bound: the summarized result is the one without it (every check, record, row and
        number) once its own records are removed: tolerance.root_logit_bound and applied.root_logit_bound, and the declared_root_logit_bound
        label (declared_turn1_logit_bound on (c), absent on (b) and block invariance)."""
        def unrooted(value, where=()):
            if isinstance(value, dict):
                out = {k: unrooted(v, where + (k,)) for k, v in value.items() if k != 'root_logit_bound'}
                if out.get('bound_rule') == 'declared_root_logit_bound':
                    if 'c' in where:out['bound_rule'] = 'declared_turn1_logit_bound'
                    else:out.pop('bound_rule')
                return out
            return [unrooted(v, where) for v in value] if isinstance(value, list) else value
        for noise in (dict(t1_logits=.1, root_logits=.1, twin=True), dict(root_logits=.12, twin=True), dict(leaf=.006, root_logits=.05, twin=True)):
            with self.subTest(**noise):
                ev = self.evaluated(**noise);(without, status, _), (rooted, status2, _) = self.judge(ev, declared(bounds=bounds(turn1_logit_bound=.1))), \
                    self.judge(ev, declared(bounds=bounds(turn1_logit_bound=.1, root_logit_bound=.1)))
                self.assertEqual(status2, status);self.assertEqual(rooted['tolerance']['root_logit_bound'], .1)
                self.assertNotEqual(rooted['tolerance']['declared_bounds']['sha256'], without['tolerance']['declared_bounds']['sha256'])
                for result in (without, rooted):[result['tolerance']['declared_bounds'].pop(k) for k in ('sha256', 'path')];result['tolerance'].pop('value_bound_source')   # the files differ
                self.assertEqual(unrooted(rooted), without);self.assertEqual(unrooted(gate.tolerance_line_n(rooted['tolerance'])), gate.tolerance_line_n(without['tolerance']))

    def test_n2_equivalence_takes_the_declared_root_logit_bound(self):
        """summarize_n2: turn-1 own priors and the root prior .1 off (floats and child-block invariance). root_logit_bound .05 beside
        turn1_logit_bound .15 fails the root alone; swapped, the turn-1 prior alone fails; each record as the turn-1 bound of its number gives it."""
        frame = fake_as_depth2(self.cached('n2', self._n2_frame));other = copy.deepcopy(frame)
        other.t1_own_logits[np.asarray(other.t1_status) != 0] += np.float32(.1);other.root_peer_logits[...] += np.float32(.1)
        found, floats = gate.n2_compare(frame, other);e = gate.decide_roots(frame, frame, self.requests, depth2_backups)
        call = dict(a=found, floats=floats, c=dict(depth1_noise_floor=gate.g2._numeric([[0]], [FLOOR], [0.])), e=e, spreads=[SPREAD] * 10,
            invariance=[gate.g2.block_invariance(frame, other)])
        judged = lambda **numbers: gate.summarize_n2({'x': [copy.deepcopy(call)]}, declared=declared(bounds=bounds(**numbers)))['x']
        loose, tight = judged(turn1_logit_bound=.15), judged(turn1_logit_bound=.05)
        self.assertEqual((loose['floats']['root_peer_logits']['passed'], loose['floats']['t1_own_logits']['passed'], loose['block_invariance']['passed']), (True, True, True))
        self.assertGreater(loose['floats']['root_peer_logits']['compared'], 0);self.assertNotIn('declared_root_logit_bound', json.dumps(loose, default=str))
        plain = lambda m: {k: v for k, v in m.items() if k != 'bound_rule'}
        for numbers, wanted, other_report in ((dict(turn1_logit_bound=.15, root_logit_bound=.05), .05, loose), (dict(turn1_logit_bound=.05, root_logit_bound=.15), .15, tight)):
            report = judged(**numbers)
            for section in ('floats', 'block_invariance'):
                part = report[section] if section == 'floats' else report[section]['floats'];was = other_report[section] if section == 'floats' else other_report[section]['floats']
                m = part['root_peer_logits'];self.assertEqual((m['bound'], m['bound_rule'], m['passed']), (wanted, 'declared_root_logit_bound', wanted > .1), section)
                self.assertEqual(plain(m), plain((tight if wanted == .05 else loose)[section]['root_peer_logits'] if section == 'floats' else
                    (tight if wanted == .05 else loose)[section]['floats']['root_peer_logits']), section)
                self.assertEqual({n: x for n, x in part.items() if n != 'root_peer_logits'}, {n: x for n, x in was.items() if n != 'root_peer_logits'}, section)
            self.assertEqual((report['status'], report['passed'], report['block_invariance']['passed']), ('FAIL', False, False))
            self.assertEqual((report['tolerance']['root_logit_bound'], report['tolerance']['logit_bound']), (wanted, numbers['turn1_logit_bound']))

    def test_a_root_only_shift_passes_under_its_root_logit_bound(self):
        """Only the root opponent's prior .1 off. turn1_logit_bound .05 alone fails every root site and nothing else (gate: (b), (c),
        depth2_pass, block invariance; --n2-equivalence: the floats and child-block invariance). root_logit_bound .15 beside it passes: (b)'s
        and block invariance's passed are recomputed from their records, not kept from the turn-1 verdict g2.summarize / merge_invariance gave
        them, in the gate's verdict, in mutant_verdict and in summarize_n2. n2's turn-1 heads against the oracle (reported, not gated) bound
        the root at root_logit_bound and the turn-1 priors at turn1_logit_bound."""
        tight, rooted = declared(bounds=bounds(turn1_logit_bound=.05)), declared(bounds=bounds(turn1_logit_bound=.05, root_logit_bound=.15))
        ev = self.evaluated(root_logits=.1, twin=True)
        result, status, _ = self.judge(ev, tight)
        self.assertEqual((status, set(failing(result))), ('FAIL', {'b', 'c', 'block_invariance'}))
        for check, name in ROOT_SITES:self.assertEqual((site(result, check, name)['bound'], site(result, check, name)['passed']), (.05, False), (check, name))
        self.assertEqual({(check, n) for check in SITE_KEYS for n, m in result['checks'][check][SITE_KEYS[check]].items() if m.get('passed') is False}, set(ROOT_SITES))
        result, status, _ = self.judge(ev, rooted)
        self.assertEqual((status, failing(result), result['passed']), ('PASS', {}, True))
        self.assertEqual((result['checks']['b']['passed'], result['checks']['block_invariance']['passed']), (True, True))
        for check, name in ROOT_SITES:self.assertEqual((site(result, check, name)['bound'], site(result, check, name)['passed']), (.15, True), (check, name))
        verdict, judged = gate.mutant_verdict(copy.deepcopy(ev), lanes=(2187,), declared=rooted)
        self.assertEqual((verdict['status'], verdict['killed'], verdict['caught_by'], verdict['root_logit_bound']), ('PASS', False, [], .15))
        self.assertEqual((judged['checks']['b']['passed'], judged['checks']['block_invariance']['passed']), (True, True))
        verdict, _ = gate.mutant_verdict(copy.deepcopy(ev), lanes=(2187,), declared=tight)
        self.assertEqual((verdict['status'], verdict['caught_by']), ('FAIL', ['b', 'block_invariance', 'c']));self.assertNotIn('root_logit_bound', verdict)
        frame = fake_as_depth2(self.cached('n2', self._n2_frame));other = copy.deepcopy(frame);other.root_peer_logits[...] += np.float32(.1)
        found, floats = gate.n2_compare(frame, other);e = gate.decide_roots(frame, frame, self.requests, depth2_backups)
        heads = dict(depth1_noise_floor=gate.g2._numeric([[0]], [FLOOR], [0.]), **copy.deepcopy({k: floats[k] for k in ('root_peer_logits', 't1_own_logits', 't1_peer_logits')}))
        call = dict(a=found, floats=floats, c=heads, e=e, spreads=[SPREAD] * 10, invariance=[gate.g2.block_invariance(frame, other)])   # heads: check_heads' stand-ins
        self.assertGreater(floats['root_peer_logits']['compared'], 0);self.assertAlmostEqual(floats['root_peer_logits']['max_abs_difference'], .1, delta=1e-3)
        report = gate.summarize_n2({'x': [copy.deepcopy(call)]}, declared=tight)['x']
        self.assertEqual((report['passed'], report['floats']['root_peer_logits']['passed'], report['block_invariance']['passed']), (False, False, False))
        self.assertEqual([n for n, m in report['floats'].items() if not m['passed']] + [n for n, m in report['block_invariance']['floats'].items() if not m['passed']],
            ['root_peer_logits', 'root_peer_logits'])
        self.assertEqual(report['turn1_heads_vs_oracle']['quantities']['root_peer_logits']['bound'], .05)
        report = gate.summarize_n2({'x': [copy.deepcopy(call)]}, declared=rooted)['x']
        self.assertEqual((report['status'], report['passed'], report['block_invariance']['passed'], report['block_invariance']['discrete_equal']), ('PASS', True, True, True))
        self.assertEqual({k: report['floats']['root_peer_logits'][k] for k in ('bound', 'bound_rule', 'passed')}, dict(bound=.15, bound_rule='declared_root_logit_bound', passed=True))
        quantities = report['turn1_heads_vs_oracle']['quantities']
        self.assertEqual({k: quantities['root_peer_logits'][k] for k in ('bound', 'bound_rule', 'within_bound')}, dict(bound=.15, bound_rule='declared_root_logit_bound', within_bound=True))
        for name in ('t1_own_logits', 't1_peer_logits'):self.assertEqual((quantities[name]['bound'], 'bound_rule' in quantities[name]), (.05, False), name)

    def depth2_evaluated(self, *, values=0., terminal=0., twin=False, corrupt_leaf_digest=False, actors=1, twin_terminal=0., twin_outside_digest=False):
        """One call of the stand-in at depth 2 (k 2): `values` added to the head-valued leaves and the expanded turn-1 cells (so (b)'s and (c)'s
        turn-1 values and the leaf values move together), `terminal` to the terminal turn-1 cells (outside every digest mask), the synthetic
        floor FLOOR and spreads SPREAD (value bound .008, guard .005) and `actors` parameter sets. corrupt_leaf_digest: the pricer's digest of one
        scored leaf one bit off. twin: the unshifted frame as the full-width depth-2 pass (its depth2_pass_t1_value, which no digest certifies) and
        as a block-invariance twin (with its invariance_cover, as run_call_n records it); twin_terminal is added to that twin's terminal turn-1
        values and twin_outside_digest sets one word of its leaf digest at a leaf no one enumerated (outside the digest mask) one bit off."""
        def priced():
            with self.engine():return OracleNPricer(self.ex, object(), depth=2, k_own=2, k_opp=2).price(self.requests, self.tables, self.recipe, PARAMS)
        def enumerated():
            with self.engine():
                stack = reference.stack(self.requests, batch._numpy_worlds(self.requests, self.tables, self.recipe))
                return gate.oracle_levels(oracle=self.oracle, stack=stack, requests=self.requests, recipe=self.recipe, params=PARAMS, frame=base, plan=gate.plan_of(2, 2, 2))
        base = self.cached('frame2', priced);data = self.cached('data2', enumerated);frame = copy.deepcopy(base);t1s = np.asarray(frame.t1_status)
        frame.leaf_value[np.asarray(frame.leaf_status) == 4] += np.float32(values);frame.t1_value[t1s == 2] += np.float32(values)
        if terminal:frame.t1_value[t1s == 1] += np.float32(terminal)
        if corrupt_leaf_digest:
            index = data['leaves']['index'][np.flatnonzero(gate._leaf_scored(frame, data['leaves']))[0]];frame.leaf_digest[tuple(index)][0] ^= np.uint32(1)
        ev = gate.evaluate_n(frame, self._d(), self.requests, data, context={}, backups=FakeBackupsN, decision=True, choices=True)
        ev['c']['depth1_noise_floor'] = gate.g2._numeric([[0]], [FLOOR], [0.]);ev['spreads'] = [SPREAD] * 10;ev['actors'] = actors
        if twin:
            valued = (t1s == 2) | ((t1s == 3) & np.isin(np.asarray(frame.t1_error), (5, 6)))   # depth2_pass_check's t1_value cells
            ev['c']['depth2_pass_t1_value'] = gate.g2._numeric(np.argwhere(valued), np.asarray(base.t1_value)[valued], np.asarray(frame.t1_value)[valued])
            other = copy.deepcopy(base) if twin_terminal or twin_outside_digest else base
            if twin_terminal:other.t1_value[np.asarray(other.t1_status) == 1] += np.float32(twin_terminal)
            if twin_outside_digest:other.leaf_digest[tuple(np.argwhere(np.asarray(other.leaf_status) == 0)[0])][0] ^= np.uint32(1)
            ev['invariance'] = [gate.g2.block_invariance(gate.frame_arrays(other), gate.frame_arrays(frame))]
            ev['invariance_cover'] = [gate.invariance_cover(gate.frame_arrays(other), gate.frame_arrays(frame))]
        return ev

    def test_the_declared_guard_is_digest_certified_at_depth_2(self):
        """GUARD_DECLARED_RULE at depth 2: (b)'s and (c)'s turn-1 values and the leaf values .006 off, beyond the guard (.005) and within the value
        bound (.008), every digest matching, one parameter set. Frozen: FAIL on the guard ((b), (c)). Declared: PASS, each guarded quantity
        digest_certified with its digest evidence ((b)'s terminal cells outside t1_digest_own's mask, bit-equal), the guard still computed and
        reported (within_guard False, ratio_to_guard > 1), the table, the TOLERANCE line and tolerance.guard_declared recording the modes;
        mutant_verdict judges under the same rule. Two parameter sets, or terminal cells (outside every digest mask) that are not bit-equal, keep
        the guard gated ((a) fails those exactly too)."""
        d = declared();ev = self.depth2_evaluated(values=.006)
        frozen, status, c = self.judge(ev)
        self.assertEqual((status, set(failing(frozen))), ('FAIL', {'b', 'c'}))
        for m in (frozen['checks']['b']['values']['t1_value'], c['t1_value_head'], c['leaf_value']):
            self.assertEqual((m['within_bound'], m['within_guard'], m['passed'], m.get('guard_mode', 'gated'), m.get('guard_evidence')), (True, False, False, 'gated', None))
        self.assertNotIn('guard_declared', frozen['tolerance']);self.assertNotIn('guard_mode', frozen['checks']['b']['values']['t1_value'])
        result, status, c = self.judge(ev, d);b = result['checks']['b']['values']['t1_value']
        self.assertEqual((status, failing(result)), ('PASS', {}))
        for name, m, digest in (('b.t1_value', b, 't1_digest_own'), ('c.t1_value_head', c['t1_value_head'], 't1_digest_own'), ('c.leaf_value', c['leaf_value'], 'leaf_digest')):
            e = m['guard_evidence']
            self.assertEqual((m['guard_mode'], m['passed'], m['within_bound'], m['within_guard'], m['guard_bound'], e['digest'], e['digest_mismatched'], e['evaluator_actors']),
                ('digest_certified', True, True, False, .005, digest, 0, 1), name)
            self.assertGreater(m['ratio_to_guard'], 1, name);self.assertGreaterEqual(e['digest_compared'], m['compared'] - (e['value_positions_outside_digest'] or 0), name)
        self.assertGreater(b['guard_evidence']['value_positions_outside_digest'], 0);self.assertEqual(b['guard_evidence']['value_positions_outside_digest_inexact'], 0)
        self.assertEqual(c['leaf_value']['guard_evidence']['subset'], gate.LEAF_CONSTRUCTION);self.assertGreater(c['leaf_value']['beyond_guard'], 0)
        record = result['tolerance']['guard_declared'];self.assertEqual(record['rule'], gate.GUARD_DECLARED_RULE)
        self.assertEqual((record['modes'], record['gated_no_certifying_digest']), (dict.fromkeys(('b.t1_value', 'c.t1_value_head', 'c.leaf_value'), 'digest_certified'), []))
        self.assertIn('before any champion result', gate.GUARD_DECLARED_RULE);self.assertIn('GUARD_DECLARED_RULE', gate.DECLARED_RULE + gate.DECLARED_TOLERANCE_RULE)
        rows = {(r['check'], r['quantity']): r for r in result['tolerance']['table']}
        self.assertEqual((rows[('b', 't1_value')]['guard_mode'], rows[('b', 't1_value')]['passed']), ('digest_certified', True))
        line = {(x['check'], x['quantity']): x for x in gate.tolerance_line_n(result['tolerance'])['checks']}
        self.assertEqual({k: line[k]['guard_mode'] for k in (('b', 't1_value'), ('c', 't1_value_head'), ('c', 'leaf_value'))}, dict.fromkeys((('b', 't1_value'), ('c', 't1_value_head'), ('c', 'leaf_value')), 'digest_certified'))
        self.assertEqual(gate.mutant_verdict(copy.deepcopy(ev), lanes=(2187,), declared=d)[0]['status'], 'PASS')   # the mutation test's verdicts, same rule
        self.assertEqual(gate.mutant_verdict(copy.deepcopy(ev), lanes=(2187,))[0]['status'], 'FAIL')
        result, status, c = self.judge(self.depth2_evaluated(values=.006, actors=2), d)   # a digest certifies inputs, not which parameters priced them
        self.assertEqual((status, set(failing(result))), ('FAIL', {'b', 'c'}));self.assertEqual(set(result['tolerance']['guard_declared']['modes'].values()), {'gated'})
        self.assertEqual(c['leaf_value']['guard_evidence']['evaluator_actors'], 2)
        result, status, _ = self.judge(self.depth2_evaluated(values=.006, terminal=1e-6), d);b = result['checks']['b']['values']['t1_value']   # outside the mask, not exact
        self.assertEqual((status, set(failing(result)), b['guard_mode'], b['within_bound']), ('FAIL', {'a', 'b'}, 'gated', True))   # (a): terminal values exact
        self.assertEqual({k for k, v in result['checks']['a']['mismatched'].items() if v}, {'t1_value_terminal'})
        self.assertEqual(b['guard_evidence']['value_positions_outside_digest_inexact'], b['guard_evidence']['value_positions_outside_digest'])
        recorded, derived = self.rederived(result, d);self.assertEqual((derived, recorded['b.t1_value']), (recorded, 'gated'))   # --reverdict agrees

    def rederived(self, result, d):
        """(the guard modes a summarized declared depth-2 result records, those --reverdict re-derives once it is written as gate.json is):
        reverdict's own steps, declared_guards from the saved evidence and apply_rules_depthn without covers."""
        saved = json.loads(json.dumps(result, allow_nan=False, default=str));recorded = gate.guard_record(saved['checks'])['modes']
        actors = saved['tolerance']['evaluator_actors'];gate.declared_guards(saved, None, actors);gate.apply_rules_depthn(saved, 2, actors=actors, declared=d)
        return recorded, saved['tolerance']['guard_declared']['modes']

    def test_a_twin_terminal_value_that_is_not_exact_gates_block_invariance(self):
        """Block invariance's t1_value with the other child block's terminal turn-1 values .006 off: terminal cells lie outside t1_digest_own's
        mask, so they must be bit-equal; they are not, the guard is gated and block invariance FAILs on the guard alone (within the value
        bound), while leaf_value stays certified; --reverdict's re-derivation from the written report agrees."""
        d = declared();result, status, _ = self.judge(self.depth2_evaluated(twin=True, twin_terminal=.006), d)
        inv = result['checks']['block_invariance'];m = inv['floats']['t1_value'];e = m['guard_evidence']
        self.assertEqual((status, set(failing(result)), inv['discrete_equal']), ('FAIL', {'block_invariance'}, True))
        self.assertEqual((m['guard_mode'], m['within_bound'], m['within_guard'], m['passed'], e['digest_mismatched']), ('gated', True, False, False, 0))
        self.assertGreater(e['value_positions_outside_digest_inexact'], 0);self.assertEqual(inv['floats']['leaf_value']['guard_mode'], 'digest_certified')
        recorded, derived = self.rederived(result, d);self.assertEqual(derived, recorded);self.assertEqual(recorded['block_invariance.t1_value'], 'gated')

    def test_a_digest_differing_only_outside_the_mask_gates_the_guard(self):
        """Block invariance with the twin's leaf digest one bit off at a leaf no one enumerated (outside the mask) and the leaf values .006 apart:
        the certifying check, block_invariance's comparison of the whole digest array, has a mismatch, so leaf_value's guard is gated, live and as
        --reverdict re-derives it from the written report (its differing count); t1_value, whose digests all match, stays certified."""
        d = declared();result, status, _ = self.judge(self.depth2_evaluated(values=.006, twin=True, twin_outside_digest=True), d)
        inv = result['checks']['block_invariance'];m = inv['floats']['leaf_value']
        self.assertEqual((status, inv['passed'], inv['discrete_equal'], inv['differing']['leaf_digest']['count']), ('FAIL', False, False, 1))
        self.assertEqual((m['guard_mode'], m['guard_evidence']['digest_mismatched'], m['within_bound'], m['passed']), ('gated', 1, True, False))
        self.assertEqual(inv['floats']['t1_value']['guard_mode'], 'digest_certified')
        recorded, derived = self.rederived(result, d);self.assertEqual(derived, recorded);self.assertEqual(recorded['block_invariance.leaf_value'], 'gated')

    def test_a_corrupted_leaf_digest_gates_the_leaf_guard(self):
        """The same declared depth-2 call with one scored leaf's digest one bit off: (a) fails on it, the leaf values' guard is gated there
        (leaf_value fails on the guard alone), the turn-1 values stay digest-certified, and the run FAILs."""
        result, status, c = self.judge(self.depth2_evaluated(values=.006, corrupt_leaf_digest=True), declared())
        self.assertEqual((status, set(failing(result)), result['checks']['a']['mismatched']['leaf_digest']), ('FAIL', {'a', 'c'}, 1))
        m = c['leaf_value'];self.assertEqual((m['guard_mode'], m['guard_evidence']['digest_mismatched'], m['within_bound'], m['passed']), ('gated', 1, True, False))
        self.assertEqual((c['t1_value_head']['guard_mode'], result['checks']['b']['values']['t1_value']['guard_mode']), ('digest_certified', 'digest_certified'))
        self.assertEqual(result['tolerance']['guard_declared']['modes']['c.leaf_value'], 'gated')

    def test_a_guarded_value_without_a_certifying_digest_stays_gated(self):
        """With the full-width depth-2 pass and a block-invariance twin: block invariance's t1_value and leaf_value are digest-certified by the
        two frames' digest arrays on (a)'s masks (every value outside them bit-equal), while depth2_pass_t1_value, which no digest check
        certifies, stays gated (guard_evidence.reason, tolerance.guard_declared.gated_no_certifying_digest) and fails (c) on the guard alone."""
        result, status, c = self.judge(self.depth2_evaluated(values=.006, twin=True), declared())
        m = c['depth2_pass_t1_value'];self.assertEqual((status, set(failing(result))), ('FAIL', {'c'}))
        self.assertEqual((m['guard_mode'], m['guard_evidence']['reason'], m['guard_evidence']['digest'], m['within_bound'], m['within_guard'], m['passed']),
            ('gated', gate.GUARD_NO_DIGEST, None, True, False, False))
        self.assertEqual([n for n, q in c.items() if n not in gate.g2.FLOORS and not q['passed']], ['depth2_pass_t1_value'])
        self.assertEqual(result['tolerance']['guard_declared']['gated_no_certifying_digest'], ['c.depth2_pass_t1_value'])
        inv = result['checks']['block_invariance'];self.assertTrue(inv['passed'])
        for name, digest in (('t1_value', 't1_digest_own'), ('leaf_value', 'leaf_digest')):
            q = inv['floats'][name];e = q['guard_evidence']
            self.assertEqual((q['guard_mode'], q['within_guard'], e['digest'], e['digest_mismatched'], e['value_positions_outside_digest_inexact']), ('digest_certified', False, digest, 0, 0), name)
            self.assertEqual(result['tolerance']['guard_declared']['modes'][f'block_invariance.{name}'], 'digest_certified')
        frozen, status, _ = self.judge(self.depth2_evaluated(values=.006, twin=True))
        self.assertEqual((status, frozen['checks']['block_invariance']['passed']), ('FAIL', False))

    def test_n2_equivalence_takes_the_declared_guard(self):
        """summarize_n2 under a declaration: the Depth2Pricer frame's expanded turn-1 values and head-valued leaves .006 off the DepthNPricer's
        (beyond the guard, within the value bound): the floats and child-block invariance are digest-certified by n2_compare's t1_digest_own /
        t2_digest and the frames' digest arrays (PASS; frozen: FAIL on the guard); one head-valued leaf's t2_digest one bit off gates t2_value;
        two parameter sets gate every guard."""
        frame = fake_as_depth2(self.cached('n2', self._n2_frame))
        def call(corrupt=False, actors=1):
            other = copy.deepcopy(frame);t1s, t2s = np.asarray(other.t1_status), np.asarray(other.t2_status)
            other.t1_value[t1s == 2] += np.float32(.006);other.t2_value[t2s == 4] += np.float32(.006)
            if corrupt:other.t2_digest[tuple(np.argwhere(t2s == 4)[0])][0] ^= np.uint32(1)
            found, floats = gate.n2_compare(frame, other);e = gate.decide_roots(frame, frame, self.requests, depth2_backups)
            return dict(a=found, floats=floats, c=dict(depth1_noise_floor=gate.g2._numeric([[0]], [FLOOR], [0.])), e=e, spreads=[SPREAD] * 10, cover=gate.n2_cover(frame, other),
                actors=actors, invariance=[gate.g2.block_invariance(frame, other)], invariance_cover=[gate.invariance_cover(frame, other)])
        frozen = gate.summarize_n2({'x': [call()]})['x']
        self.assertEqual((frozen['status'], frozen['floats']['t1_value']['passed'], frozen['floats']['t2_value']['passed'], frozen['block_invariance']['passed']), ('FAIL', False, False, False))
        self.assertNotIn('guard_declared', frozen['tolerance']);self.assertNotIn('guard_mode', frozen['floats']['t1_value'])
        report = gate.summarize_n2({'x': [call()]}, declared=declared())['x'];record = report['tolerance']['guard_declared']
        self.assertEqual((report['status'], report['passed'], report['block_invariance']['passed']), ('PASS', True, True))
        self.assertEqual(record['modes'], dict.fromkeys(('floats.t1_value', 'floats.t2_value', 'block_invariance.t1_value', 'block_invariance.t2_value'), 'digest_certified'))
        for name, digest in (('t1_value', 't1_digest_own'), ('t2_value', 't2_digest')):
            m = report['floats'][name];self.assertEqual((m['within_guard'], m['guard_evidence']['digest'], m['guard_evidence']['value_positions_outside_digest']), (False, digest, 0), name)
        report = gate.summarize_n2({'x': [call(corrupt=True)]}, declared=declared())['x']
        self.assertEqual((report['status'], report['discrete']['mismatched']['t2_digest'], report['floats']['t2_value']['guard_mode'], report['floats']['t1_value']['guard_mode']),
            ('FAIL', 1, 'gated', 'digest_certified'))
        self.assertEqual((report['block_invariance']['floats']['t2_value']['guard_mode'], report['block_invariance']['floats']['t2_value']['guard_evidence']['digest_mismatched']), ('gated', 1))
        report = gate.summarize_n2({'x': [call(actors=2)]}, declared=declared())['x']   # two parameter sets: a digest certifies inputs, not which parameters priced them
        self.assertEqual((report['status'], set(report['tolerance']['guard_declared']['modes'].values()), report['floats']['t1_value']['guard_evidence']['evaluator_actors']),
            ('FAIL', {'gated'}, 2))

    def test_the_topk_root_gap_count_takes_the_root_logit_bound(self):
        """merge_topk_n: the root selection's gaps are counted below a given root_logit_bound (recorded beside the plies), every other ply's
        below logit_bound; root_logit_bound None is the call without it."""
        reports = [{'root': {'opp': dict(gaps=[.05, .12, .3], oracle_prior_set_differs=1)}, 'ply2': {'own': dict(gaps=[.05, .12, .3], oracle_prior_set_differs=0)}}]
        plain = gate.merge_topk_n(reports, logit_bound=.15)
        self.assertEqual((plain['root']['opp']['gap_below_logit_bound'], plain['ply2']['own']['gap_below_logit_bound']), (2, 2));self.assertNotIn('root_logit_bound', plain)
        self.assertEqual(gate.merge_topk_n(reports, logit_bound=.15, root_logit_bound=None), plain)
        rooted = gate.merge_topk_n(reports, logit_bound=.15, root_logit_bound=.1)
        self.assertEqual((rooted['root']['opp']['gap_below_logit_bound'], rooted['ply2'], rooted['root_logit_bound']), (1, plain['ply2'], .1))
        self.assertEqual({k: v for k, v in rooted.items() if k not in ('root', 'root_logit_bound')}, {k: v for k, v in plain.items() if k != 'root'})

    def test_reverdict_refuses_traces_of_a_declaration_and_numbers_off_the_frozen_rule(self):
        """A report without declared_bounds sections is re-judged only if nothing in it shows a declaration (the declared tolerance texts and
        sources, a frozen_rule record, a declared bound_rule, a reverdict of one) and its numbers are the frozen rule's: tolerance.rule, logit_bound
        LOGIT_BOUND, value_bound 2 x depth1_noise_floor (1e-4 without a floor), and every kept (c) and (b) bound its class's; a pre-registration
        record given for it is refused. The unchanged synthetic report re-judges as before."""
        def edited(change):
            report = saved_gate();change(report);return report
        quantity = lambda name, **values: lambda r: r['checks']['c']['quantities'][name].update(values)
        cases = dict(
            declared_rule_text=(edited(lambda r: r['tolerance'].__setitem__('rule', gate.DECLARED_TOLERANCE_RULE)), 'left traces'),
            frozen_rule_record=(edited(lambda r: r['tolerance'].__setitem__('frozen_rule', dict(value_bound=.008))), 'left traces'),
            declared_source=(edited(lambda r: r['tolerance'].__setitem__('value_bound_source', 'declared value_bound (--declared-bounds sha256 ...)')), 'left traces'),
            declared_bound_rule=(edited(quantity('leaf_value', bound_rule='declared_value_bound')), 'left traces'),
            reverdict_of_one=(edited(lambda r: r.__setitem__('reverdict', dict(declared_bounds_sha256='a' * 64))), 'left traces'),
            other_rule_text=(edited(lambda r: r['tolerance'].__setitem__('rule', 'values: something else')), 'not the frozen TOLERANCE_RULE'),
            logit_bound=(edited(lambda r: r['tolerance'].__setitem__('logit_bound', .15)), 'is not LOGIT_BOUND'),
            value_bound=(edited(lambda r: r['tolerance'].update(value_bound=.05)), "not the frozen rule's"),
            no_floor=(edited(lambda r: r['tolerance'].__setitem__('depth1_noise_floor', None)), "not the frozen rule's 0.0001"),
            value_quantity=(edited(quantity('leaf_value', bound=.05)), 'checks.c.leaf_value.bound 0.05'),
            turn1_prior=(edited(quantity('t1_own_logits', bound=.15)), 'checks.c.t1_own_logits.bound 0.15'),
            b_value=(edited(lambda r: r['checks'].__setitem__('b', dict(passed=True, values=dict(t1_value=dict(bound=.05))))), 'checks.b.t1_value.bound 0.05'))
        for case, (report, message) in cases.items():
            with self.subTest(case):
                directory = Path(tempfile.mkdtemp(prefix='depthn-reverdict-'));(directory / 'gate.json').write_text(json.dumps(report))
                with patch.object(gate, 'log', lambda *a, **k: None), self.assertRaises(SystemExit) as raised:gate.main(['--reverdict', str(directory / 'gate.json')])
                self.assertIn(message, str(raised.exception.code));self.assertIn('fail closed', str(raised.exception.code));self.assertFalse((directory / gate.REVERDICT_FILE).exists())
        directory = Path(tempfile.mkdtemp(prefix='depthn-reverdict-'));(directory / 'gate.json').write_text(json.dumps(saved_gate()))
        with patch.object(gate, 'log', lambda *a, **k: None), self.assertRaises(SystemExit) as raised:
            gate.main(['--reverdict', str(directory / 'gate.json'), '--declared-bounds-commit', 'b' * 40])
        self.assertIn('records no declaration', str(raised.exception.code))
        code, out, _, _ = reverdicted(saved_gate());self.assertEqual((code, out['status']), (0, 'PASS'));self.assertNotIn('declared_bounds_sha256', out['reverdict'])

    def test_reverdict_refuses_saved_verdicts_that_disagree_with_their_numbers(self):
        """--reverdict recomputes each (c) quantity's within_bound (no non-finite difference, max |diff| <= bound) and within_guard (max |diff| <
        guard) from its saved numbers, the live comparison, and refuses any disagreement: a max |diff| raised past the bound, a non-finite count,
        a verdict flipped either way, a guard verdict flipped. A consistent report re-judges as before."""
        def edited(name, **values):
            report = saved_gate();report['checks']['c']['quantities'][name].update(values);return report
        cases = dict(raised_max=(edited('leaf_value', max_abs_difference=1.), 'leaf_value: within_bound True'),
            nonfinite=(edited('t1_value_head', nonfinite=1), 't1_value_head: within_bound True'),
            flipped_false=(edited('ply2_value_head', within_bound=False), 'ply2_value_head: within_bound False'),
            guard_flipped=(edited('leaf_value', within_guard=True), 'leaf_value: within_guard True'))
        for case, (report, message) in cases.items():
            with self.subTest(case):
                directory = Path(tempfile.mkdtemp(prefix='depthn-reverdict-'));(directory / 'gate.json').write_text(json.dumps(report))
                with patch.object(gate, 'log', lambda *a, **k: None), self.assertRaises(SystemExit) as raised:gate.main(['--reverdict', str(directory / 'gate.json')])
                self.assertIn(message, str(raised.exception.code));self.assertIn('disagree with their own numbers', str(raised.exception.code))
                self.assertFalse((directory / gate.REVERDICT_FILE).exists())
        self.assertEqual(gate.verdict_inconsistencies(saved_gate()['checks']['c']['quantities']), [])

    def test_off_the_champion_path_depth2_and_n2_are_unchanged(self):
        """B1's path (the head override here): a depth-2 full-width gate gates only the depth-N leaf walk's path in oracle consistency and
        carries no merged_depth2_parity; --n2-equivalence --block-invariance stays refused off the champion path (the champion takes it)."""
        code, out = self.main('--depth', '2', '--k-own', '9', '--k-opp', '9', *HEAD_FLAGS);report = json.loads((out / 'gate.json').read_text())
        self.assertEqual((code, report['status']), (0, 'PASS'), report.get('status_reasons'))
        self.assertEqual(set(report['checks']['oracle_consistency']['compared']), {'leaf_path'});self.assertNotIn('merged_depth2_parity', report['header'])
        live = ['--out', 'o', '--d3-root', 'b', '--chunk', '0', '--n2-equivalence', '--block-invariance']
        champion = ['--champion-archive', 'c.npz', '--champion-archive-sha256', 'e' * 64, '--champion-checkpoint-sha256', 'c' * 64]
        for argv in (live, live + list(HEAD_FLAGS), live[:-2] + ['--depth3-diagnostics', '--block-invariance'] + champion):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:gate.parse(argv)
            self.assertEqual(raised.exception.code, 2);self.assertIn('--block-invariance belongs to the gate', stderr.getvalue())
        self.assertTrue(gate.parse(live + champion)[1].block_invariance)

    def test_without_the_flag_nothing_changes(self):
        """No declaration: no declared key in any report section, log line or saved frame; header_block keeps its three keys; declared_rule and
        declared_kwargs are the identity and the empty call. (The byte-for-byte check against the pre-change gate is the probe in the report.)"""
        lines = [];code, out = self.main('--save-arrays', lines=lines);text = (out / 'gate.json').read_text();report = json.loads(text)
        self.assertEqual((code, report['status']), (0, 'PASS'));self.assertNotIn('declared', text);self.assertNotIn('frozen_rule', report['tolerance'])
        self.assertEqual(list(report['header']), ['near_ties', 'rules_changed_after_failure', 'frozen_rule_set'])
        self.assertFalse(any('declared' in json.dumps(x, default=str) for x in lines));self.assertNotIn('DECLARED_BOUNDS', [x['message'] for x in lines])
        _, _, meta = gate.load_frame(out / 'frames-block0-call0-prior-k2.npz');self.assertNotIn('declared_bounds_sha256', meta['evaluator'])
        rule = gate.g2.tolerance_rule(None, None);self.assertIs(gate.declared_rule(rule, None), rule);self.assertEqual(gate.declared_kwargs(None), {})
        self.assertIsNone(gate.parse(['--out', 'o', '--d3-root', 'b', '--chunk', '0'])[1].declared_bounds)


class N2ReverdictTests(FixtureN, unittest.TestCase):
    """--reverdict on a saved n2-equivalence.json (reverdict_n2): the audit of a declared report written where git could not verify its record
    (provisional) clears it on the work tree holding the commit; every saved verdict is re-judged from its own numbers and a disagreement, a
    bound off the declaration or the live path's, or a guard mode the saved evidence does not support is refused; the saved statuses are
    re-derived (a FAIL stays FAIL, an inconsistent one is corrected and recorded, as the gate path re-derives its status); a report without a
    declaration is judged under the frozen rule from each config's own floor. The gate path's reverdict tests above are unchanged."""
    _n2_frame = N.GateNTests._n2_frame
    main, roots = N.CommandLineRunTests.main, N.CommandLineRunTests.roots

    def call(self, *, values=.006, invariance=True, actors=1):
        """One --n2-equivalence call's results as n2_equivalence_call records them: the stand-in N=2 frame against a copy whose expanded turn-1
        values and head-valued leaves are `values` off (.006: beyond the guard .005, within the value bound .008), (e) and the merged (e)
        through depth2_backups, the synthetic floor FLOOR and spreads SPREAD, `actors` parameter sets, the per-position covers and child-block
        invariance against the copy."""
        frame = fake_as_depth2(self.cached('n2', self._n2_frame));other = copy.deepcopy(frame);t1s, t2s = np.asarray(other.t1_status), np.asarray(other.t2_status)
        other.t1_value[t1s == 2] += np.float32(values);other.t2_value[t2s == 4] += np.float32(values)
        found, floats = gate.n2_compare(frame, other);e = gate.decide_roots(frame, frame, self.requests, depth2_backups)
        return dict(a=found, floats=floats, c=dict(depth1_noise_floor=gate.g2._numeric([[0]], [FLOOR], [0.])), e=e, e_merged=copy.deepcopy(e), spreads=[SPREAD] * 10,
            cover=gate.n2_cover(frame, other), actors=actors, **(dict(invariance=[gate.g2.block_invariance(frame, other)], invariance_cover=[gate.invariance_cover(frame, other)])
                if invariance else {}))

    def declared_report(self, *, numbers=None, **call):
        """(report, record, audit arguments) of a declared run on the champion as a pod without git writes it: the declaration (default: value
        .008, turn-1 logits .1, near-tie Q .009, root logits .05) committed in a scratch repository with a readable source measurement, and
        git_verification 'not verifiable here' in the run's record (provisional)."""
        record = committed(measured_declaration(bounds=numbers or bounds(near_tie_q_bound=.009, root_logit_bound=.05)))
        d = dict(load(record), git_verification=dict(status='not verifiable here', detail='/work/gate is not the top of a git work tree'))
        report = n2_saved({'full_width': [self.call(**call)], 'top2': [self.call(**call)]}, declared=d, evaluator=N2_CHAMPION)
        return report, record, ('--declared-bounds-commit', record.commit, '--declared-bounds-sha256', record.sha256)

    def test_a_declared_report_audits_on_the_work_tree_holding_its_commit(self):
        """The provisional PASS (guards digest-certified, the root prior at its declared bound, (e)'s q_noise_bound the near-tie Q bound): the
        audit re-judges it PASS with git and the source measurement verified and no reason left, every config and the header as saved, written
        beside it and never overwritten; on a checkout without git --reverdict alone keeps the reason and the audit refuses."""
        report, record, audit = self.declared_report();top = report['configs']['top2']
        self.assertEqual((report['status'], [c['status'] for c in report['configs'].values()]), ('PASS', ['PASS', 'PASS']))
        self.assertEqual(len(report['status_reasons']), 1);self.assertTrue(report['status_reasons'][0].startswith('provisional: git_verification not verifiable here'))
        self.assertEqual(top['tolerance']['guard_declared']['modes'], dict.fromkeys(('floats.t1_value', 'floats.t2_value', 'block_invariance.t1_value',
            'block_invariance.t2_value'), 'digest_certified'))
        self.assertEqual((top['floats']['root_peer_logits']['bound'], top['floats']['root_peer_logits']['bound_rule'], top['e']['q_noise_bound'], top['e_merged_depth2']['q_noise_bound']),
            (.05, 'declared_root_logit_bound', .009, .009))
        code, out, lines, path = n2_reverdicted(report, *audit, repo=record.root);rv = out['reverdict']
        self.assertEqual((code, out['status'], out['passed'], out['status_reasons']), (0, 'PASS', True, []))
        self.assertEqual((rv['source_status'], rv['source_status_reasons'], rv['configs'], rv['changed']), ('PASS', report['status_reasons'], dict(full_width='PASS', top2='PASS'), []))
        self.assertEqual((rv['declared_bounds_sha256'], rv['declared_bounds_commit'], rv['declared_bounds_reverified']['git']['status'],
            rv['declared_bounds_reverified']['source_measurement']['status']), (record.sha256, record.commit, 'verified', 'verified'))
        self.assertIn('nothing was re-priced', rv['note']);self.assertEqual(rv['rule'], gate.N2_REVERDICT_RULE)
        self.assertEqual((out['configs'], out['header']), (report['configs'], report['header']))   # every saved verdict re-derived as saved
        self.assertEqual(list(out)[:4], ['evaluator', 'value_scale', 'source', 'header'])
        line = next(x for x in lines if x['message'] == 'REVERDICT')
        self.assertEqual((line['old_status'], line['new_status'], line['status_reasons'], line['changed'], line['out']), ('PASS', 'PASS', [], [], str(path.with_name(gate.N2_REVERDICT_FILE))))
        before = path.with_name(gate.N2_REVERDICT_FILE).read_bytes()
        with patch.object(gate, 'log', lambda *a, **k: None), patch.object(gate, 'REPO_ROOT', record.root), self.assertRaises(SystemExit) as raised:
            gate.main(['--reverdict', str(path), *audit])
        self.assertIn('never overwritten', str(raised.exception.code));self.assertEqual(path.with_name(gate.N2_REVERDICT_FILE).read_bytes(), before)
        exported = Path(tempfile.mkdtemp(prefix='depthn-exported-'))   # a checkout that is not a git work tree
        code, out, _, _ = n2_reverdicted(report, repo=exported)
        self.assertEqual((code, out['status'], len(out['status_reasons'])), (0, 'PASS', 1));self.assertTrue(out['status_reasons'][0].startswith('provisional: git_verification'))
        text, out, _, path = n2_reverdicted(report, *audit, repo=exported)
        self.assertIsNone(out);self.assertIn('requires git_verification verified', text);self.assertFalse(path.with_name(gate.N2_REVERDICT_FILE).exists())

    def test_the_audit_refuses_what_the_saved_numbers_or_the_record_do_not_support(self):
        """Each edit of the declared report (or a commit / sha256 given that is not the recorded one) is refused and nothing is written: the
        declaration's text and sections, a config's tolerance off the declaration, a bound off the live path's, a verdict off its own numbers,
        a guard mode the saved digest findings or per-position evidence do not support, an (e) whose passed its numbers contradict; a field the
        verdict needs, missing."""
        report, record, audit = self.declared_report();h = lambda r: r['header']['declared_bounds'];top = lambda r: r['configs']['top2']
        def stripped(r):
            r['header'].pop('declared_bounds')
            for c in r['configs'].values():c['tolerance'].pop('declared_bounds')
        def evidence(section, **values):
            return lambda r: [m['guard_evidence'].update(values) for m in top(r)[section].values() if 'guard_evidence' in m]
        cases = dict(
            other_commit=(None, ('--declared-bounds-commit', 'b' * 40), 'is not the recorded'), other_sha=(None, ('--declared-bounds-sha256', '0' * 64), 'is not the recorded'),
            content_text=(lambda r: h(r).__setitem__('content_text', h(r)['content_text'].replace('0.008', '0.5')), audit, 'does not hash'),
            header=(lambda r: r['header'].pop('declared_bounds'), audit, 'configs.top2.tolerance.declared_bounds without header.declared_bounds'),
            stripped=(stripped, (), 'left traces'), evaluator=(lambda r: r['evaluator'].__setitem__('archive_sha256', '6' * 64), audit, 'evaluator.archive_sha256'),
            config_value_bound=(lambda r: top(r)['tolerance'].__setitem__('value_bound', .5), audit, 'configs.top2.tolerance value_bound / logit_bound [0.5, 0.1]'),
            config_root_bound=(lambda r: top(r)['tolerance'].pop('root_logit_bound'), audit, 'configs.top2.tolerance.root_logit_bound None'),
            config_sha=(lambda r: top(r)['tolerance']['declared_bounds'].__setitem__('sha256', '0' * 64), audit, 'configs.top2.tolerance.declared_bounds does not name'),
            q_noise_bound=(lambda r: top(r)['tolerance'].__setitem__('q_noise_bound', .008), audit, 'configs.top2.tolerance.q_noise_bound 0.008 is not 0.009'),
            e_q_noise_bound=(lambda r: top(r)['e_merged_depth2'].__setitem__('q_noise_bound', .008), audit, 'configs.top2.e_merged_depth2.q_noise_bound'),
            float_bound=(lambda r: top(r)['floats']['t1_own_logits'].__setitem__('bound', .2), audit, "configs.top2.floats.t1_own_logits: bound, guard_bound, bound_rule"),
            root_rule=(lambda r: top(r)['block_invariance']['floats']['root_peer_logits'].pop('bound_rule'), audit, "are not the live path's [0.05, None, 'declared_root_logit_bound']"),
            guard_bound=(lambda r: top(r)['floats']['t2_value'].__setitem__('guard_bound', .007), audit, 'configs.top2.floats.t2_value: bound, guard_bound'),
            within_bound=(lambda r: top(r)['floats']['t2_value'].__setitem__('within_bound', False), audit, 'disagree with their own numbers'),
            within_guard=(lambda r: top(r)['floats']['t1_value'].__setitem__('within_guard', True), audit, 'disagree with their own numbers'),
            raised_max=(lambda r: top(r)['block_invariance']['floats']['t2_value'].__setitem__('max_abs_difference', .5), audit, 'disagree with their own numbers'),
            float_passed=(lambda r: top(r)['floats']['t1_own_logits'].__setitem__('passed', False), audit, 'configs.top2.floats.t1_own_logits: passed False, re-derived True'),
            digest_mismatch=(lambda r: (top(r)['discrete']['mismatched'].__setitem__('t2_digest', 1), top(r)['discrete']['first_mismatches'].append(dict(field='t2_digest',
                cell=[0, 0, 0, 0, 0, 0], pricer=[1, 0], oracle=[0, 0]))), audit, "configs.top2.floats.t2_value: saved 'digest_certified', re-derived 'gated'"),
            inexact_outside=(lambda r: top(r)['floats']['t1_value']['guard_evidence'].__setitem__('value_positions_outside_digest_inexact', 1), audit, 'saved guard modes differ'),
            invariance_digest=(lambda r: top(r)['block_invariance']['floats']['t2_value']['guard_evidence'].__setitem__('digest_mismatched', 1), audit, 'saved guard modes differ'),
            two_actors=(evidence('floats', evaluator_actors=2), audit, 'saved guard modes differ'), mode_gated=(lambda r: top(r)['floats']['t2_value'].__setitem__('guard_mode', 'gated'), audit,
                'saved guard modes differ'),
            record_gated=(lambda r: top(r)['tolerance']['guard_declared']['modes'].__setitem__('floats.t2_value', 'gated'), audit, 'configs.top2.tolerance.guard_declared.modes'),
            e_passed=(lambda r: top(r)['e'].__setitem__('max_abs_delta_q', .02), audit, 'configs.top2.e: passed True, q_noise_within_bound True; re-derived False, False'),
            merged_passed=(lambda r: top(r)['e_merged_depth2'].__setitem__('argmax_q_differs_non_tie', 1), audit, 'configs.top2.e_merged_depth2: passed True'),
            # decision_passed clause by clause: each count that fails (e), with passed left true; a record without gating; the within flag alone
            unmatched=(lambda r: top(r)['e_merged_depth2']['replaced'].__setitem__('unmatched_head_leaves', 1), audit, 'configs.top2.e_merged_depth2: passed True, '
                'q_noise_within_bound True; re-derived False, True'),
            status_differs=(lambda r: top(r)['e'].__setitem__('status_differs', 1), audit, 'configs.top2.e: passed True, q_noise_within_bound True; re-derived False, True'),
            nonfinite=(lambda r: top(r)['e_merged_depth2'].__setitem__('nonfinite_roots', 1), audit, 'configs.top2.e_merged_depth2: passed True, q_noise_within_bound True; '
                're-derived False, True'),
            target_flip=(lambda r: top(r)['e'].__setitem__('argmax_target_differs_non_tie', 1), audit, 'configs.top2.e: passed True, q_noise_within_bound True; re-derived False'),
            no_gating=(lambda r: top(r)['e'].pop('gating'), audit, 'configs.top2.e: passed True, q_noise_within_bound True; re-derived False, None'),
            within_flag=(lambda r: top(r)['e'].__setitem__('q_noise_within_bound', False), audit, 'configs.top2.e: passed True, q_noise_within_bound False; re-derived True, True'),
            # the aggregates the saved records determine
            listed_mismatch=(lambda r: top(r)['discrete']['first_mismatches'].append(dict(field='t2_status', cell=[0], pricer=1, oracle=2)), audit,
                "configs.top2.discrete: first_mismatches list ['t2_status'] against mismatched counts 0"),
            discrete_equal=(lambda r: top(r)['block_invariance'].__setitem__('discrete_equal', False), audit, 'configs.top2.block_invariance.discrete_equal False but differing records []'),
            differing_hidden=(lambda r: top(r)['block_invariance'].update(differing=dict(t2_status=dict(count=3, size=100)), first_differences=[dict(field='t2_status', cell=[0])]),
                audit, "configs.top2.block_invariance.discrete_equal True but differing records ['t2_status']"),
            differing_unlisted=(lambda r: top(r)['block_invariance'].update(differing=dict(t2_status=dict(count=3, size=100)), discrete_equal=False, passed=False), audit,
                'configs.top2.block_invariance: first_differences against differing counts'),
            world_flag=(lambda r: top(r).__setitem__('world_reports_first_difference', dict(root=0, depth2=None, depthn={'reason': 'x'})), audit,
                'configs.top2.world_reports_equal True with world_reports_first_difference recorded'),
            world_converse=(lambda r: top(r).__setitem__('world_reports_equal', False), audit, 'configs.top2.world_reports_equal False with world_reports_first_difference null'),
            # the guard number: GUARD_FACTOR x the saved p10 (g2.tolerance_rule, guard_note), and the p10 None iff no spread is non-zero
            guard_number=(lambda r: top(r)['tolerance'].__setitem__('guard_bound', .05), audit, 'configs.top2.tolerance guard_bound, guard [0.05, \'applied\'] are not the live path'),
            p10=(lambda r: top(r)['tolerance'].__setitem__('p10_cell_a2_spread', 1.), audit, 'configs.top2.tolerance guard_bound, guard'),
            p10_nulled=(lambda r: top(r)['tolerance'].update(p10_cell_a2_spread=None, guard_bound=None, guard=gate.guard_note(dict(guard_bound=None), top(r)['tolerance']['spread'])['guard']),
                audit, 'configs.top2.tolerance.p10_cell_a2_spread None with spread.nonzero 10'))
        for case, (change, extra, message) in cases.items():
            with self.subTest(case):
                r = copy.deepcopy(report)
                if change is not None:change(r)
                text, out, _, path = n2_reverdicted(r, *extra, repo=record.root)
                self.assertIsNone(out, case);self.assertIn(message, text);self.assertIn('fail closed', text);self.assertFalse(path.with_name(gate.N2_REVERDICT_FILE).exists())
        def drop(*keys):
            def change(r):
                node = r
                for key in keys[:-1]:node = node[key]
                node.pop(keys[-1])
            return change
        missing = dict(within_bound=(drop('configs', 'top2', 'floats', 't2_value', 'within_bound'), 'configs.top2.floats.t2_value.within_bound'),
            within_guard=(drop('configs', 'top2', 'block_invariance', 'floats', 't1_value', 'within_guard'), 'configs.top2.block_invariance.floats.t1_value.within_guard'),
            max_dq=(drop('configs', 'full_width', 'e', 'max_abs_delta_q'), 'configs.full_width.e.max_abs_delta_q (a number)'),
            q_noise_bound=(drop('configs', 'top2', 'tolerance', 'q_noise_bound'), 'configs.top2.tolerance.q_noise_bound (a number)'),
            mismatched=(drop('configs', 'top2', 'discrete', 'mismatched'), 'configs.top2.discrete.mismatched (a JSON object)'),
            discrete_equal=(drop('configs', 'top2', 'block_invariance', 'discrete_equal'), 'configs.top2.block_invariance.discrete_equal'),
            world_reports=(lambda r: top(r).__setitem__('world_reports_equal', 'yes'), 'configs.top2.world_reports_equal (true or false)'),
            digest_count=(lambda r: top(r)['discrete']['compared'].__setitem__('t1_digest_own', '9'), 'configs.top2.discrete.compared (counts)'),
            evidence_count=(lambda r: top(r)['floats']['t1_value']['guard_evidence'].__setitem__('value_positions_outside_digest', 'none'),
                'configs.top2.floats.t1_value.guard_evidence.value_positions_outside_digest (an integer or null)'),
            differing=(lambda r: top(r)['block_invariance'].__setitem__('differing', dict(t2_digest=dict(count='1'))), 'configs.top2.block_invariance.differing'),
            not_an_object=(lambda r: r['configs'].__setitem__('top2', [1]), 'configs.top2 (a JSON object)'),
            unmatched_key=(drop('configs', 'top2', 'e', 'replaced', 'unmatched_head_leaves'), 'configs.top2.e.replaced.unmatched_head_leaves (an integer)'),
            merged_section=(drop('configs', 'full_width', 'e_merged_depth2'), 'configs.full_width.e_merged_depth2 (the merged (e) ran: header.merged_depth2_parity)'),
            merged_by_options=(lambda r: (drop('configs', 'full_width', 'e_merged_depth2')(r), r['header'].pop('merged_depth2_parity')), 'configs.full_width.e_merged_depth2'),
            champion_all_three=(lambda r: (r['header'].pop('merged_depth2_parity'), [(c.pop('e_merged_depth2'), r['options'][n]['depth2_options'].pop('evaluator'))
                for n, c in r['configs'].items()]), 'header.merged_depth2_parity (a champion report: run_n2 ran the merged (e))'),
            champion_parity=(drop('header', 'merged_depth2_parity'), 'header.merged_depth2_parity (a champion report'),
            champion_option=(drop('options', 'top2', 'depth2_options', 'evaluator'), 'options.top2.depth2_options.evaluator (a champion report'),
            invariance_section=(drop('configs', 'top2', 'block_invariance'), 'configs.top2.block_invariance (child-block invariance ran: options.top2.child_blocks)'),
            n2_float=(drop('configs', 'top2', 'floats', 't1_own_logits'), 'configs.top2.floats.t1_own_logits (n2_compare compares it'),
            invariance_float=(drop('configs', 'top2', 'block_invariance', 'floats', 'root_peer_logits'), 'configs.top2.block_invariance.floats.root_peer_logits (block invariance'),
            first_difference=(drop('configs', 'top2', 'world_reports_first_difference'), 'configs.top2.world_reports_first_difference'),
            spread=(drop('configs', 'top2', 'tolerance', 'spread'), 'configs.top2.tolerance.spread (a JSON object)'),
            options=(drop('options'), 'options (per config: a JSON object)'),
            configs=(drop('configs'), 'configs'), schema=(lambda r: r.__setitem__('schema', 'v0'), 'schema'))
        for case, (change, message) in missing.items():
            with self.subTest(case):
                r = copy.deepcopy(report);change(r);text, out, _, path = n2_reverdicted(r, *audit, repo=record.root)
                self.assertIsNone(out, case);self.assertIn('lacks what the verdict needs', text);self.assertIn(message, text)
                self.assertFalse(path.with_name(gate.N2_REVERDICT_FILE).exists())

    def test_the_saved_statuses_are_rederived(self):
        """A saved FAIL (values .012 off, beyond the value bound) stays FAIL with nothing changed. A saved status that its own parts contradict is
        re-derived from them (n2_status, n2_run_status) as the gate path re-derives its status, the saved one kept as source_status and each
        differing field listed in `changed`: a config saved FAIL whose parts all pass is PASS, a mismatch count under a saved PASS fails the
        config (discrete.passed re-derived), an (e) that compared no root makes it INCOMPLETE."""
        failing, record, audit = self.declared_report(values=.012);top = failing['configs']['top2']
        self.assertEqual((failing['status'], top['status'], top['floats']['t2_value']['within_bound'], top['floats']['t2_value']['guard_mode']), ('FAIL', 'FAIL', False, 'digest_certified'))
        code, out, _, _ = n2_reverdicted(failing, *audit, repo=record.root)
        self.assertEqual((code, out['status'], out['reverdict']['configs'], out['reverdict']['changed'], out['status_reasons']), (1, 'FAIL', dict(full_width='FAIL', top2='FAIL'), [], []))
        self.assertEqual(out['configs'], failing['configs'])
        report, record, audit = self.declared_report()
        def edited(change):
            r = copy.deepcopy(report);change(r);return n2_reverdicted(r, *audit, repo=record.root)
        code, out, lines, _ = edited(lambda r: (r['configs']['top2'].update(status='FAIL', passed=False), r.update(status='FAIL', passed=False)))
        self.assertEqual((code, out['status'], out['configs']['top2']['status'], out['reverdict']['source_status'], out['reverdict']['source_configs']['top2']),
            (0, 'PASS', 'PASS', 'FAIL', 'FAIL'))
        self.assertEqual(out['reverdict']['changed'], [dict(config='top2', field='status', before='FAIL', after='PASS'), dict(config='top2', field='passed', before=False, after=True)])
        self.assertEqual(next(x for x in lines if x['message'] == 'REVERDICT')['changed'], out['reverdict']['changed'])
        code, out, _, _ = edited(lambda r: (r['configs']['full_width']['discrete']['mismatched'].__setitem__('t1_status', 1),
            r['configs']['full_width']['discrete']['first_mismatches'].append(dict(field='t1_status', cell=[0, 0, 0, 0], pricer=1, oracle=2))))
        self.assertEqual((code, out['status'], out['reverdict']['configs'], out['configs']['full_width']['discrete']['passed']), (1, 'FAIL', dict(full_width='FAIL', top2='PASS'), False))
        self.assertEqual([(c['field'], c['before'], c['after']) for c in out['reverdict']['changed']],
            [('status', 'PASS', 'FAIL'), ('passed', True, False), ('discrete.passed', True, False)])
        code, out, _, _ = edited(lambda r: r['configs']['top2']['e'].__setitem__('roots_compared', 0))
        self.assertEqual((code, out['status'], out['configs']['top2']['status'], out['configs']['top2']['status_reasons']), (1, 'INCOMPLETE', 'INCOMPLETE', ['(e) compared no root']))

    def test_a_report_without_a_declaration_is_judged_under_the_frozen_rule(self):
        """No declaration: each config's numbers must be the frozen rule's from its own floor (value bound 2 x FLOOR, LOGIT_BOUND, q_noise_bound
        the value bound, the floats' bounds their class's) and no trace of a declaration may remain; then it is re-judged as saved (PASS within
        the guard, FAIL beyond it: the frozen rule gates the guard); a pre-registration record given for it is refused."""
        report = n2_saved({'full_width': [self.call(values=.002)], 'top2': [self.call(values=.002)]})
        self.assertEqual((report['status'], report['configs']['top2']['tolerance']['value_bound'], 'status_reasons' in report), ('PASS', 2 * float(np.float32(FLOOR)), False))
        code, out, _, _ = n2_reverdicted(report)
        self.assertEqual((code, out['status'], out['status_reasons'], out['reverdict']['changed'], out['configs']), (0, 'PASS', [], [], report['configs']))
        self.assertFalse(any(k.startswith('declared') for k in out['reverdict']));self.assertNotIn('declared_bounds', out['header'])
        frozen = n2_saved({'full_width': [self.call()], 'top2': [self.call()]});code, out, _, _ = n2_reverdicted(frozen)   # .006: beyond the guard, which the frozen rule gates
        self.assertEqual((code, out['status'], {q for q, m in out['configs']['top2']['floats'].items() if not m['passed']}), (1, 'FAIL', {'t1_value', 't2_value'}))
        text, out, _, _ = n2_reverdicted(report, '--declared-bounds-commit', 'b' * 40);self.assertIsNone(out);self.assertIn('records no declaration', text)
        top = lambda r: r['configs']['top2']
        cases = dict(value_bound=(lambda r: top(r)['tolerance'].__setitem__('value_bound', .05), "configs.top2.tolerance.value_bound 0.05 is not the frozen rule's"),
            no_floor=(lambda r: top(r)['tolerance'].__setitem__('depth1_noise_floor', None), "not the frozen rule's 0.0001"),
            logit_bound=(lambda r: top(r)['tolerance'].__setitem__('logit_bound', .15), 'configs.top2.tolerance.logit_bound 0.15 is not LOGIT_BOUND'),
            rule_text=(lambda r: top(r)['tolerance'].__setitem__('rule', gate.DECLARED_TOLERANCE_RULE), 'left traces'),
            frozen_record=(lambda r: top(r)['tolerance'].__setitem__('frozen_rule', {}), 'configs.top2.tolerance.frozen_rule'),
            bound_rule=(lambda r: top(r)['floats']['root_peer_logits'].__setitem__('bound_rule', 'declared_root_logit_bound'), 'configs.top2.floats.root_peer_logits.bound_rule'),
            guard_mode=(lambda r: top(r)['block_invariance']['floats']['t1_value'].__setitem__('guard_mode', 'digest_certified'), 'block_invariance.floats.t1_value.guard_mode'),
            reverdict_of_one=(lambda r: r.__setitem__('reverdict', dict(declared_bounds_sha256='a' * 64)), 'reverdict.declared_bounds_sha256'),
            q_noise_bound=(lambda r: top(r)['tolerance'].__setitem__('q_noise_bound', .009), 'configs.top2.tolerance.q_noise_bound 0.009 is not'),
            float_bound=(lambda r: top(r)['floats']['t1_own_logits'].__setitem__('bound', .15), "live path's [0.1, None, None]"),
            invariance_bound=(lambda r: top(r)['block_invariance']['floats']['t2_value'].__setitem__('bound', .05), 'configs.top2.block_invariance.floats.t2_value'))
        for case, (change, message) in cases.items():
            with self.subTest(case):
                r = copy.deepcopy(report);change(r);text, out, _, path = n2_reverdicted(r)
                self.assertIsNone(out, case);self.assertIn(message, text);self.assertIn('fail closed', text);self.assertFalse(path.with_name(gate.N2_REVERDICT_FILE).exists())

    def test_a_failure_cannot_be_hidden_by_dropping_or_editing_what_records_it(self):
        """Each genuine failure (the merged (e), child-block invariance's discrete arrays, the world reports) re-judges FAIL with nothing changed.
        Dropping the section that records it, or editing one record against another, with every status reset to PASS, is refused and writes
        nothing: the merged (e) (header.merged_depth2_parity), block invariance (options' child_blocks), discrete_equal against differing,
        world_reports_equal against its first difference; so is dropping a float, a config (with or without its option), renaming one or adding a
        float n2_compare never writes. A float absent where a :shape mismatch is recorded is no omission: the config FAILs on the mismatch."""
        report, record, audit = self.declared_report()
        def failed(kind):
            def change(r):
                c = r['configs']['full_width' if kind == 'merged' else 'top2']
                if kind == 'merged':c['e_merged_depth2'].update(argmax_q_differs_non_tie=1, argmax_q_differs=1, passed=False)
                if kind == 'invariance':c['block_invariance'].update(differing=dict(t2_status=dict(count=3, size=100)), first_differences=[dict(field='t2_status', cell=[0])],
                    discrete_equal=False, passed=False)
                if kind == 'world':c.update(world_reports_equal=False, world_reports_first_difference=dict(root=5, depth2={'a': 1}, depthn={'a': 2}))
                c.update(status='FAIL', passed=False);r.update(status='FAIL', passed=False)
            return change
        def unpassed(r):
            for c in r['configs'].values():c.update(status='PASS', passed=True)
            r.update(status='PASS', passed=True)
        for kind in ('merged', 'invariance', 'world'):
            with self.subTest(genuine=kind):
                r = copy.deepcopy(report);failed(kind)(r);code, out, _, _ = n2_reverdicted(r, *audit, repo=record.root)
                self.assertEqual((code, out['status'], out['reverdict']['changed']), (1, 'FAIL', []))
        top = lambda r: r['configs']['top2']
        forged = dict(
            merged_dropped=(lambda r: (failed('merged')(r), r['configs']['full_width'].pop('e_merged_depth2'), unpassed(r)), 'lacks what the verdict needs: configs.full_width.e_merged_depth2'),
            invariance_dropped=(lambda r: (failed('invariance')(r), top(r).pop('block_invariance'), unpassed(r)), 'lacks what the verdict needs: configs.top2.block_invariance'),
            discrete_equal_flipped=(lambda r: (failed('invariance')(r), top(r)['block_invariance'].update(discrete_equal=True), unpassed(r)), 'discrete_equal True but differing'),
            world_flag_flipped=(lambda r: (failed('world')(r), top(r).update(world_reports_equal=True), unpassed(r)), 'world_reports_equal True with world_reports_first_difference recorded'),
            float_dropped=(lambda r: top(r)['floats'].pop('t1_own_logits'), 'lacks what the verdict needs: configs.top2.floats.t1_own_logits'),
            config_dropped=(lambda r: r['configs'].pop('top2'), "as run_n2 writes it (fail closed): configs ['full_width'] are not the configs options names"),
            config_and_option_dropped=(lambda r: (r['configs'].pop('top2'), r['options'].pop('top2')), 'run_n2 writes full_width (options k 9) and top{k}'),
            renamed=(lambda r: (r['configs'].__setitem__('top3', r['configs'].pop('top2')), r['options'].__setitem__('top3', r['options'].pop('top2'))), 'run_n2 writes full_width'),
            unknown_float=(lambda r: top(r)['floats'].__setitem__('t9_value', dict(top(r)['floats']['t1_own_logits'])), "configs.top2.floats.t9_value: not one of n2_compare's floats"))
        for case, (change, message) in forged.items():
            with self.subTest(case):
                r = copy.deepcopy(report);change(r);text, out, _, path = n2_reverdicted(r, *audit, repo=record.root)
                self.assertIsNone(out, case);self.assertIn(message, text);self.assertFalse(path.with_name(gate.N2_REVERDICT_FILE).exists())
        shaped = n2_saved({'full_width': [self.call(values=.002)], 'top2': [self.call(values=.002)]});c = shaped['configs']['top2']   # undeclared
        c['discrete']['mismatched']['t1_status:shape'] = 1;c['discrete']['first_mismatches'].append(dict(field='t1_status:shape', cell=[0], pricer=[1], oracle=[2]))
        for q in ('t1_own_logits', 't1_peer_logits'):c['floats'].pop(q)
        c['world_reports_equal'] = False   # n2_compare returns before comparing the world reports: neither record is written
        code, out, _, _ = n2_reverdicted(shaped)
        self.assertEqual((code, out['status'], out['configs']['top2']['discrete']['passed'], [x['field'] for x in out['reverdict']['changed']]),
            (1, 'FAIL', False, ['status', 'passed', 'discrete.passed']))

    def test_a_forged_guard_number_is_refused(self):
        """The guard number is GUARD_FACTOR x the saved p10 (g2.tolerance_rule, guard_note), None iff the p10 is (none iff no spread is non-zero).
        A report FAILing on the guard, under the frozen rule and under a declaration with two parameter sets (every mode gated), whose guard is
        raised or removed, with every float and status made to pass, is refused and writes nothing; so is the p10 nulled beside non-zero spreads
        with the skipped guard's own text."""
        frozen = n2_saved({'full_width': [self.call()], 'top2': [self.call()]});declared, record, audit = self.declared_report(actors=2)
        self.assertEqual((frozen['status'], declared['status'], set(declared['configs']['top2']['tolerance']['guard_declared']['modes'].values())), ('FAIL', 'FAIL', {'gated'}))
        def guard(value):
            def change(r):
                for c in r['configs'].values():
                    c['tolerance']['guard_bound'] = value
                    for _, floats in gate.n2_float_sections(c).values():
                        for m in floats.values():
                            if 'guard_bound' not in m:continue
                            if value is None:
                                for k in ('guard_bound', 'within_guard', 'ratio_to_guard'):m.pop(k)
                            else:m.update(guard_bound=value, within_guard=True)
                            m['passed'] = True
                    c['block_invariance']['passed'] = True;c.update(status='PASS', passed=True)
                r.update(status='PASS', passed=True)
            return change
        def nulled(r):
            guard(None)(r)
            for c in r['configs'].values():c['tolerance'].update(p10_cell_a2_spread=None, guard=gate.guard_note(dict(guard_bound=None), c['tolerance']['spread'])['guard'])
        for name, report, extra, repo in (('frozen', frozen, (), None), ('declared', declared, audit, record.root)):
            for case, change, message in (('raised', guard(.05), "guard_bound, guard [0.05, 'applied'] are not the live path's [0.005, 'applied']"),
                    ('removed', guard(None), "guard_bound, guard [None, 'applied'] are not the live path's [0.005, 'applied']"),
                    ('p10_nulled', nulled, 'p10_cell_a2_spread None with spread.nonzero 10')):
                with self.subTest(name=name, case=case):
                    r = copy.deepcopy(report);change(r);text, out, _, path = n2_reverdicted(r, *extra, repo=repo)
                    self.assertIsNone(out);self.assertIn(message, text);self.assertIn('fail closed', text);self.assertFalse(path.with_name(gate.N2_REVERDICT_FILE).exists())

    def test_live_n2_reports_reverdict_as_written(self):
        """run_n2's own n2-equivalence.json through main() without a declaration (the stand-in, which carries no world_reports, FAILs):
        --reverdict reproduces every config."""
        code, out = self.main('--n2-equivalence', '--n2-k', '2');report = json.loads((out / gate.N2_FILE).read_text())
        self.assertEqual((code, report['status']), (1, 'FAIL'))
        with patch.object(gate, 'log', lambda *a, **k: None):code = gate.main(['--reverdict', str(out / gate.N2_FILE)])
        again = json.loads((out / gate.N2_REVERDICT_FILE).read_text())
        self.assertEqual((code, again['status'], again['reverdict']['changed'], again['configs']), (1, 'FAIL', [], report['configs']))

    @unittest.skipUnless(HAVE_PRICER, 'gpu_engine.device_depthn_pricer / depthn_backups are not on this tree')
    def test_a_live_champion_report_audits_and_keeps_its_merged_decision(self):
        """run_n2 on the champion path, as only it writes a champion report: the champion tests' fixture (an exact stand-in champion as the run's
        forward, as load_roots resolves --champion-archive) under a declaration written where git could not verify it (a checkout without git).
        The report carries header.merged_depth2_parity, every config's e_merged_depth2 and each Depth2Pricer's evaluator and PASSes
        provisionally; the audit on the work tree holding the commit re-judges it PASS with nothing changed and no reason left; with all three
        records deleted it is refused and nothing is written."""
        champion = importlib.import_module('tests.test_gpu_engine_champion_evaluator');cli = champion.ChampionCommandLineTests
        live = type('LiveChampionN2', (champion.ChampionFixture, unittest.TestCase), dict(main=cli.main, roots=cli.roots, declared=cli.declared, FLAGS=cli.FLAGS,
            runTest=lambda self: None))
        live.setUpClass()
        try:
            run = live();_, flags, sha = run.declared();record = run.record
            exported = Path(tempfile.mkdtemp(prefix='depthn-exported-'));target = exported / gate.DECLARED_SUBDIR / 'champion.json'
            target.parent.mkdir(parents=True);target.write_bytes(record.path.read_bytes());run.repo = exported   # the run's checkout: not a git work tree
            code, out = run.main('--n2-equivalence', '--n2-k', '2', '--declared-bounds', str(target), *flags[2:])
        finally:live.tearDownClass()
        report = json.loads((out / gate.N2_FILE).read_text())
        self.assertEqual((code, report['status'], report['evaluator']['kind'], report['header']['declared_bounds']['git_verification']['status']),
            (0, 'PASS', 'champion', 'not verifiable here'))
        self.assertTrue(report['status_reasons'][0].startswith('provisional: git_verification'));self.assertIn('merged_depth2_parity', report['header'])
        self.assertEqual({n: ('e_merged_depth2' in c, 'evaluator' in report['options'][n]['depth2_options']) for n, c in report['configs'].items()},
            dict(full_width=(True, True), top2=(True, True)))
        audit = ('--declared-bounds-commit', record.commit, '--declared-bounds-sha256', sha)
        code, again, _, _ = n2_reverdicted(report, *audit, repo=record.root)
        self.assertEqual((code, again['status'], again['status_reasons'], again['reverdict']['changed'], again['configs']), (0, 'PASS', [], [], report['configs']))
        self.assertEqual(again['reverdict']['declared_bounds_reverified']['git']['status'], 'verified')
        r = copy.deepcopy(report);r['header'].pop('merged_depth2_parity')
        for name, c in r['configs'].items():c.pop('e_merged_depth2');r['options'][name]['depth2_options'].pop('evaluator')
        text, again, _, path = n2_reverdicted(r, *audit, repo=record.root)
        self.assertIsNone(again);self.assertIn('lacks what the verdict needs: header.merged_depth2_parity (a champion report', text)
        self.assertFalse(path.with_name(gate.N2_REVERDICT_FILE).exists())


class DefaultPathReferenceTests(FixtureN, unittest.TestCase):
    """The frozen default paths byte for byte: reference_runs' digests (reports, mutation-test.json, gate-reverdict.json and every log line,
    timing and run directories aside) must equal REFERENCE, recorded by the same harness on 1a3b2dd3, the commit before --declared-bounds, on
    the gpu_engine of that commit. The oracle's heads are offset so the floors are non-zero (a doubled q_noise default, an extra field or text
    on the default path changes a digest). A deliberate change to a default path updates REFERENCE in its own reviewed commit."""
    roots = N.CommandLineRunTests.roots   # the bank actors (no override): the default path
    REFERENCE = {   # reference_runs on 1a3b2dd3 (the gpu_engine of that commit, this harness)
        'depthn_standin_prior3_mutation': {'log.json': '526ce50e796a2085301ea23ca6f81d781f6fff5d535d4079bc8ae8709acaff26',
            'gate.json': '9b4071a272a1f2a637786551d7b7e232aa606b9f4b188bc9a15f5532a704c52b',
            'mutation-test.json': 'c69beb3648e6bc8f41c2e7831f4e3805aa7a98dc6376eefc5b26d5a778c350f5'},
        'depthn_real_depth2_full_invariance': {'log.json': '8f3d68285b062c1e219e018e573347ab556d95d23ffd897ad9799664019f6664',
            'gate.json': '566c88b1dfb818d39103f2a2cf5138e137b18e7930f91d7e06bb22432b9c44ce'},
        'depthn_real_n2': {'log.json': '28e426902c025f562907e0c6cd1223c3a2cb1be09a906cde50317004c66a450a',
            'n2-equivalence.json': '0e6c7c3aa4887cd36a30f54d2ace860389a2063e65f88de76d987ee93e7e7493'},
        'depthn_reverdict': {'log.json': 'eeafbaf75f8320e18af0a324ba0540d960cb9717a8710fd0eb72e6e9d8351efa',
            'gate-reverdict.json': 'daaf0ac7f4b689d58a1f2e4827fedc1664bff8cbb44d2b07fde515ca2e5d4e38'},
        'depth2_full_invariance': {'log.json': '9197a81e17c9537e4c9452ae4a23c799aa64ac45f6429f5ac9934e0e884007c1',
            'gate.json': '5807bd64727ba7490eae9adcd3e3b468f73bd177d54a928ac1e56721e43e3a19'},
        'depth2_top3': {'log.json': 'db15ece5fcce98e5decab35a5a4b42972671f7eb0d6ae89e2dacfe125688ed06',
            'gate.json': '1ff2a5298a1e3809a25d17cd22e85b05e11564c56cabf680a095baf43a03cc2b'},
    }

    def test_default_paths_match_the_reference(self):
        root = Path(tempfile.mkdtemp(prefix='gate-default-reference-'));got = reference_runs(self, root)
        for name in sorted(set(got) | set(self.REFERENCE)):
            with self.subTest(name):self.assertEqual(got.get(name), self.REFERENCE.get(name), f'normalized outputs under {root / "normalized" / name}')
        tolerance = lambda run, f='gate.json': json.loads((root / 'normalized' / run / f).read_text())['tolerance']   # the floors it rests on are non-zero
        self.assertGreater(tolerance('depthn_standin_prior3_mutation')['depth1_noise_floor'], 0.);self.assertGreater(tolerance('depth2_full_invariance')['q_noise_bound'], 0.)
        self.assertGreater(tolerance('depthn_real_depth2_full_invariance')['turn1_logit_floor'], 0.)

    def test_the_topk_gap_count_is_strict(self):
        """merge_topk_n counts gaps strictly below the logit bound: a gap of exactly LOGIT_BOUND is not below it."""
        topk = gate.merge_topk_n([{'ply2': {'own': dict(gaps=[gate.g2.LOGIT_BOUND, .05, .2], oracle_prior_set_differs=0)}}])
        self.assertEqual(topk['ply2']['own']['gap_below_logit_bound'], 1)


if __name__ == '__main__':unittest.main()
