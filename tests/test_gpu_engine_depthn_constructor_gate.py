"""The depth-N gate's constructor-world mode (--constructor-roster, CONSTRUCTOR_ROOTS, R1-C) on CPU fakes.

Pure parts on fakes: CONSTRUCTOR_RULE's band selection and class top-up, the roster's capture reading and pins, the carry-signature grouping,
the bind check, the averaging_argmax verdict, the flags and declaration A-C. The run itself on ChampionFixture (the fake engine, the exact
stand-in champion, the real DepthNPricer, oracle and backups): the fixture's bound requests priced as constructor roots of K = 2 blocks under
ONE PrebuiltWorlds, with no D3 FramePricer. constructor_roots' wiring on fakes of every collaborator it loads.
"""
import collections
import contextlib
from dataclasses import replace
import hashlib
import importlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pokezero import v5_gumbel as original
from gpu_engine import d3_depthn_gate as gate, depthn_oracle, device_public_teacher_batch as batch, device_public_world_state as mapping
from tests import test_gpu_engine_champion_evaluator as C, test_gpu_engine_depthn_oracle as N
from _showdown_root import requires_showdown, showdown_root

TURNS = (2,) * 50 + (5,) * 50 + (12,) * 50 + (25,) * 50   # band-major, 50 per turn band, as the roster
FLAGS = ('--constructor-roster', 'roster.json', 'a' * 64, '--showdown-root', 'showdown', '--node-binary', 'node', '--constructor-metadata', 'metadata.json',
    'b' * 64, '--constructor-vocabulary', 'vocabulary.json', 'c' * 64)
RUNTIME = ('--saved-roots', 'saved', '--index-sha256', 'd' * 64)
HARNESS = ('--fixed-shapes', '--block-invariance')
COVERED = ('--depth', '2', '--k-own', '9', '--k-opp', '9')   # what declaration A-C covers (k1 9/9 and W 8 by default)
AC = gate.REPO_ROOT / gate.CONSTRUCTOR_DECLARATION['repo_path']
DECLARED = ('--declared-bounds', str(AC), '--declared-bounds-sha256', gate.CONSTRUCTOR_DECLARATION['sha256'], '--declared-bounds-commit', '4' * 40)
CHAMPION = C.ChampionCommandLineTests.FLAGS
MODE = (*RUNTIME, *CHAMPION, *FLAGS, *HARNESS, *COVERED, *DECLARED)   # a complete, valid CONSTRUCTOR_ROOTS command line


def dn():
    return importlib.import_module('gpu_engine.device_depthn_pricer')


def roster(turns=TURNS):
    return tuple(dict(turn=t, capture_row=i) for i, t in enumerate(turns))


class Prepare:
    """A fake prepare: every ordinal fully constructs but `refused`; counts its calls per ordinal."""
    def __init__(self, refused=()):self.refused, self.calls = set(refused), collections.Counter()

    def __call__(self, i):
        self.calls[i] += 1;full = i not in self.refused
        return dict(full=full, constructed=8 if full else 3, kind=None if full else f'WORLD_CONSTRUCTOR_REFUSED:k{i}')


class Classify:
    """A fake classifier: `classes` per ordinal (else none: other); counts its calls."""
    def __init__(self, classes, prepare):self.classes, self.prepare, self.calls = classes, prepare, collections.Counter()

    def __call__(self, i):
        self.calls[i] += 1;found = self.classes.get(i, [])
        return (None, 'UNCLASSIFIED: test') if found is None else (sorted(found), 'protocol_facts')


class SelectionTests(unittest.TestCase):
    def test_bands_take_the_first_fully_constructing_roots_and_the_top_up_adds_absent_classes(self):
        prepare = Prepare(refused={0, 2, 14, 55, 160})
        classify = Classify({1: ['wish'], 12: ['sleep'], 13: ['forced_switch', 'sleep'], 14: ['fresh_substitute'], 61: ['fresh_substitute']}, prepare)
        out = gate.select_constructor_roots(roster(), prepare, classify, per_band=10)
        self.assertEqual(out['bands'], {0: [1, 3, 4, 5, 6, 7, 8, 9, 10, 11], 1: [50, 51, 52, 53, 54, 56, 57, 58, 59, 60], 2: list(range(100, 110)),
            3: list(range(150, 160))})
        # wish is present among the 40; forced_switch's first fully constructing root is 13 (12 is sleep only), sleep's 12, fresh_substitute's
        # 61 (14 is refused); in TOP_UP order, at most one root per class.
        self.assertEqual(out['top_up'], dict(forced_switch=13, sleep=12, fresh_substitute=61))
        self.assertEqual(out['selected'], sorted({i for b in out['bands'].values() for i in b} | {12, 13, 61}))
        self.assertEqual(set(prepare.calls.values()), {1});self.assertEqual(set(classify.calls.values()), {1})   # prepare and classify once per root
        # the bands prepare up to their 10th; the top-up prepares only candidates of the class it seeks (12, 13, 14, 61), never 15-49
        self.assertEqual(sorted(prepare.calls), list(range(15)) + list(range(50, 62)) + list(range(100, 110)) + list(range(150, 160)))
        self.assertIn(20, classify.calls)
        self.assertEqual(out['refused'], [dict(ordinal=i, band=b, kind=f'WORLD_CONSTRUCTOR_REFUSED:k{i}', constructed=3) for i, b in ((0, 0), (2, 0), (14, 0), (55, 1))])
        self.assertEqual(set(out['classes']), set(out['selected']));self.assertEqual(out['classes'][13], (['forced_switch', 'sleep'], 'protocol_facts'))

    def test_a_class_the_roster_lacks_stays_absent_and_one_root_can_be_two_classes_first(self):
        prepare = Prepare();classify = Classify({20: ['sleep', 'wish']}, prepare);out = gate.select_constructor_roots(roster(), prepare, classify, per_band=10)
        self.assertEqual(out['top_up'], dict(forced_switch=None, sleep=20, fresh_substitute=None, wish=20));self.assertEqual(len(out['selected']), 41)
        # the absent classes' scans classified every root, each once, and prepared none but 20
        self.assertEqual(sorted(prepare.calls), list(range(10)) + [20] + list(range(50, 60)) + list(range(100, 110)) + list(range(150, 160)))
        self.assertEqual((sorted(classify.calls), set(prepare.calls.values()), set(classify.calls.values())), (list(range(200)), {1}, {1}))

    def test_a_band_short_of_roots_fails_closed(self):
        prepare = Prepare(refused=set(range(150, 191)))
        with self.assertRaisesRegex(ValueError, r'turn band 3 .*9 of the 10 roots'):gate.select_constructor_roots(roster(), prepare, Classify({}, prepare), per_band=10)

    def test_class_counts_per_class_and_band(self):
        roots = [dict(band=0, classes=['sleep', 'wish'], root_class='wish'), dict(band=0, classes=[], root_class='other'), dict(band=3, classes=None, root_class=None)]
        counts = gate.class_counts(roots)
        self.assertEqual((counts['roots'], counts['member'], counts['primary']), (3, dict(other=1, sleep=1, unclassified=1, wish=1), dict(other=1, unclassified=1, wish=1)))
        self.assertEqual(counts['by_band']['0'], dict(roots=2, member=dict(other=1, sleep=1, wish=1), primary=dict(other=1, wish=1)))


def capture_file(path, *, owner='p1', roots=2):
    """An append-only owning capture read_capture accepts: BEGIN, then per decision a WIRE delta and its ROOT record, chained."""
    from pokezero import phase_d_owning_capture as capture
    rows, prior, lines, prefix = [], '0' * 64, [], hashlib.sha256()
    def put(row):
        nonlocal prior
        raw = capture.encoded(dict(row, schema=capture.SCHEMA, sequence=len(rows), previous_sha256=prior));rows.append(raw);prior = hashlib.sha256(raw).hexdigest()
    put(dict(event='BEGIN', owner=owner, cell=dict(ordinal=6), declaration_sha256='d' * 64, battle_id='battle-1', seed=5, stream_id=9, namespace='ns'))
    for d in range(roots):
        delta = [f'|turn|{d + 1}', '|request|' + json.dumps(dict(side=dict(id=owner), rqid=d))];start = len(lines);lines += delta
        for line in delta:prefix.update(line.encode() + b'\n')
        put(dict(event='WIRE', lines=delta, start=start, end=len(lines), generation=d + 1, prefix_sha256=prefix.hexdigest()))
        put(dict(event='ROOT', generation=d + 1, prefix_sha256=prefix.hexdigest(), own_request_line=delta[-1], request_count=d + 1, decision_index=d, turn=d + 1))
    path.write_bytes(b''.join(r + b'\n' for r in rows));return rows


class RosterTests(unittest.TestCase):
    def test_the_request_is_the_coverage_scripts_and_the_pin_is_the_chain_head(self):
        from pokezero import phase_d_owning_capture as capture
        path = Path(tempfile.mkdtemp(prefix='r1c-capture-')) / 'wire.jsonl';rows = capture_file(path)
        found = list(capture.read_capture(path))[1];meta = {k: v for k, v in found.items() if k != 'lines'}
        request, pin = gate.roster_request(meta, 7, roster_sha256='a' * 64, worlds=8)
        self.assertEqual(set(request), {'schema', 'request_id'} | set(gate.REQUEST_KEYS));self.assertEqual(request['request_id'], 'a' * 64 + ':CONSTRUCTOR_ROOTS:8:7')
        self.assertEqual({k: request[k] for k in gate.REQUEST_KEYS}, {k: found[k] for k in gate.REQUEST_KEYS})
        self.assertEqual(pin, dict(capture_path=str(path), capture_row=found['capture_row'], capture_declaration_sha256='d' * 64,
            capture_chain_sha256=hashlib.sha256(rows[found['capture_row']]).hexdigest(), prefix_sha256=found['prefix_sha256']))
        with self.assertRaisesRegex(ValueError, 'unavailable'):gate.roster_request(dict(meta, prefix_sha256='0' * 64), 7, roster_sha256='a' * 64, worlds=8)
        path.write_bytes(path.read_bytes().replace(b'battle-1', b'battle-2'))   # the chain no longer holds
        with self.assertRaisesRegex(ValueError, 'chain'):gate.roster_request(meta, 7, roster_sha256='a' * 64, worlds=8)

    def test_the_roster_is_read_bound_at_its_sha256_with_200_roots(self):
        directory = Path(tempfile.mkdtemp(prefix='r1c-roster-'))
        for count, error in ((200, None), (199, 'frozen200')):
            raw = json.dumps(dict(declaration=dict(seed=1), selected=[dict(turn=2)] * count)).encode();path = directory / f'{count}.json';path.write_bytes(raw)
            sha = hashlib.sha256(raw).hexdigest()
            if error is None:self.assertEqual(len(gate.read_roster(path, sha)[1]), 200)
            else:
                with self.assertRaisesRegex(ValueError, error):gate.read_roster(path, sha)
            with self.assertRaisesRegex(ValueError, 'differs'):gate.read_roster(path, '0' * 64)


def fake_world(index, width=2):
    return mapping.World(index, dict(x=np.zeros(width, np.int32)), dict(y=np.zeros(1, np.float32)), (), 7, False)


def fake_ctor(widths, *, blocks=2):
    """Roots with `blocks` requests each; root n's worlds have state width widths[n] (a different carry signature per width)."""
    requests, pairs, roots = [], [], []
    for n, width in enumerate(widths):
        rows = tuple(range(len(requests), len(requests) + blocks))
        for b in range(blocks):requests.append(f'root{n}-block{b}');pairs.append((fake_world(2 * b, width), fake_world(2 * b + 1, width)))
        roots.append(dict(index=n, ordinal=10 + n, rows=rows))
    binding = dn().WorldBinding('s', 'r', 'b', 'm', ());worlds = dn().PrebuiltWorlds('test', binding, tuple(pairs))
    for root, signature in zip(roots, gate.root_signatures(worlds.pairs, roots)):root['carry_signature'] = signature
    return SimpleNamespace(requests=tuple(requests), worlds=worlds, binding=binding, roots=roots)


class SignatureTests(unittest.TestCase):
    def test_one_signature_is_one_call_with_one_worlds_object(self):
        ctor = fake_ctor([2, 2, 2]);calls = gate.constructor_calls(ctor, 512)
        self.assertEqual(len(calls), 1);call = calls[0]
        self.assertEqual((call['requests'], call['worlds'].pairs), (ctor.requests, ctor.worlds.pairs))
        self.assertEqual((call['worlds'].source, call['worlds'].binding), (ctor.worlds.source, ctor.worlds.binding))
        self.assertEqual([r['rows'] for r in call['roots']], [(0, 1), (2, 3), (4, 5)]);self.assertEqual(call['signature'], dn().carry_signature(ctor.worlds.pairs))

    def test_differing_signatures_price_one_call_each_never_mixed(self):
        ctor = fake_ctor([2, 3, 2, 3, 2]);calls = gate.constructor_calls(ctor, 512)
        self.assertEqual([[r['ordinal'] for r in c['roots']] for c in calls], [[10, 12, 14], [11, 13]])
        for call in calls:
            self.assertEqual(dn().carry_signature(call['worlds'].pairs), call['signature'])   # raises if a frame mixed two structures
            self.assertEqual(call['requests'], tuple(f'root{r["ordinal"] - 10}-block{b}' for r in call['roots'] for b in range(2)))
            self.assertEqual([r['rows'] for r in call['roots']], [(2 * i, 2 * i + 1) for i in range(len(call['roots']))])
        self.assertNotEqual(calls[0]['signature'], calls[1]['signature'])
        with self.assertRaisesRegex(ValueError, 'one world source per frame'):dn().carry_signature(ctor.worlds.pairs)

    def test_calls_split_by_whole_roots_and_a_root_with_mixed_worlds_fails_closed(self):
        calls = gate.constructor_calls(fake_ctor([2] * 5), 4)
        self.assertEqual([[r['ordinal'] for r in c['roots']] for c in calls], [[10, 11], [12, 13], [14]]);self.assertEqual([len(c['requests']) for c in calls], [4, 4, 2])
        ctor = fake_ctor([2, 2]);pairs = list(ctor.worlds.pairs);pairs[3] = (fake_world(2), fake_world(3, width=5))
        with self.assertRaisesRegex(ValueError, 'root 11: its own worlds differ'):gate.root_signatures(tuple(pairs), ctor.roots)


def waiting_world(values):
    """WorldPrices with a waiting opponent: joint_values[a][0] = values[a] (None elsewhere)."""
    return original.WorldPrices(False, (False,) * 9, (0.,) * 9, tuple(tuple(values[a] if r == 0 and a < len(values) else None for r in range(9)) for a in range(9)))


def blocks(per_world):
    """K blocks of two worlds from per-world action values (world 2k+j is block k's world j)."""
    worlds = [waiting_world(v) for v in per_world];return [tuple(worlds[i:i + 2]) for i in range(0, len(worlds), 2)]


LEGAL = (True, True, True) + (False,) * 6
LOGITS = (0., .5, .25) + (0.,) * 6


class AveragingTests(unittest.TestCase):
    def test_agreement_and_the_bound_premise(self):
        """averaging's q is the mean of the blocks' completed-Q, so max |dq| <= the largest per-block |dQ|."""
        mine = blocks([(.5, .1, -.2), (.4, .3, 0.), (.2, .2, .2), (.6, 0., .1)] * 2);theirs = blocks([(.5, .1, -.2), (.41, .3, 0.), (.2, .2, .2), (.6, 0., .1)] * 2)
        r = gate.averaging_root(mine, theirs, LOGITS, LEGAL)
        self.assertEqual((r['status'], r['compared'], r['agree'], r['action']), (['COMPLETE'] * 2, True, True, [0, 0]))
        # worlds 1 and 5 differ by .01: averaging's q by 2 x .01 / 8, blocks 0 and 2 each by .01 / 2
        self.assertAlmostEqual(r['max_abs_dq'], .02 / 8);self.assertAlmostEqual(r['max_abs_block_dq'], .01 / 2);self.assertLessEqual(r['max_abs_dq'], r['max_abs_block_dq'])
        q = [original.reduce_completed_q(root_logits=LOGITS, root_legal=LEGAL, worlds=pair).improvement.completed_q for pair in mine]
        for a in range(3):self.assertAlmostEqual(r['q_pricer'][a], sum(x[a] for x in q) / len(q), places=15)
        verdict = gate.averaging_verdict([dict(roots=[r], replaced={})], .032)
        self.assertEqual((verdict['passed'], verdict['agreement_rate'], verdict['bound'], verdict['roots_compared']), (True, 1., .064, 1))

    def test_a_disagreement_within_the_bound_is_reported_and_one_beyond_it_fails(self):
        near = gate.averaging_root(blocks([(.50, .49, 0.)] * 8), blocks([(.49, .50, 0.)] * 8), LOGITS, LEGAL)
        far = gate.averaging_root(blocks([(.5, .3, 0.)] * 8), blocks([(.3, .5, 0.)] * 8), LOGITS, LEGAL)
        self.assertEqual((near['agree'], near['action']), (False, [0, 1]));self.assertAlmostEqual(near['margin_oracle'], .01)
        self.assertAlmostEqual(far['margin_oracle'], .2);self.assertAlmostEqual(far['margin_pair'], .2)
        within = gate.averaging_verdict([dict(roots=[near], replaced={})], .032)
        self.assertEqual((within['passed'], within['disagree_within_bound'], within['disagree_beyond_bound'], within['agreement_rate']), (True, 1, 0, 0.))
        self.assertEqual(within['disagreements'][0]['action'], [0, 1])
        beyond = gate.averaging_verdict([dict(roots=[near, far], replaced={})], .032)
        self.assertEqual((beyond['passed'], beyond['disagree_within_bound'], beyond['disagree_beyond_bound']), (False, 1, 1));self.assertIn('contradicts (e)', beyond['reasons'][0])
        self.assertTrue(gate.averaging_verdict([dict(roots=[far], replaced={})], .1)['passed'])   # .2 <= 2 x .1

    def test_the_gated_margin_is_the_oracle_sides_own_action_against_the_pricer_sides(self):
        """The pricer plays the oracle side's third-best action: the oracle's top-two margin (.01) is within the bound, margin_pair (.20) is not."""
        third = gate.averaging_root(blocks([(.30, .29, .50)] * 8), blocks([(.50, .49, .30)] * 8), LOGITS, LEGAL)
        self.assertEqual((third['agree'], third['action']), (False, [2, 0]))
        self.assertAlmostEqual(third['margin_oracle'], .01);self.assertAlmostEqual(third['margin_pair'], .20)   # the pricer side's own gap is .20 too
        verdict = gate.averaging_verdict([dict(roots=[third], replaced={})], .032)
        self.assertEqual((verdict['passed'], verdict['disagree_beyond_bound'], verdict['disagree_within_bound']), (False, 1, 0))
        self.assertAlmostEqual(verdict['disagreements'][0]['margin_oracle'], .01);self.assertAlmostEqual(verdict['disagreements'][0]['margin_pair'], .20)
        self.assertIn('margin_pair', verdict['reasons'][0])

    def test_a_premise_mismatch_fails(self):
        r = gate.averaging_root(blocks([(.5, .3, 0.)] * 8), blocks([(.5, .3, 0.)] * 8), LOGITS, LEGAL)
        self.assertEqual((r['premise_gap'], r['premise_holds']), ([None, None], None))   # averaging_call fills them
        verdict = gate.averaging_verdict([dict(roots=[dict(r, premise_gap=[0., 1e-3], premise_holds=False)], replaced={})], .032)
        self.assertEqual((verdict['passed'], verdict['premise_failures'], verdict['max_premise_gap']), (False, 1, 1e-3));self.assertIn('premise', verdict['reasons'][0])

    def test_no_compared_root_is_incomplete(self):
        result = dict(checks=dict(a=dict(passed=True, compared=dict(leaf_status=4))), tolerance=dict(value_bound=.1), passed=True)
        gate.constructor_checks(result, [])   # no call ran the check: it is added, with nothing compared
        self.assertEqual((result['checks']['averaging_argmax']['roots_compared'], result['checks']['averaging_argmax']['bound']), (0, .2))
        self.assertEqual(gate.gate_status_n(result), ('INCOMPLETE', ['vacuous: averaging_argmax compared no root (every root refused on both sides, or (e) did not run)']))
        refused = gate.averaging_root('BLOCK_REFUSED: pre', 'BLOCK_REFUSED: pre', LOGITS, LEGAL)
        result['checks']['averaging_argmax'] = gate.averaging_verdict([dict(roots=[refused], replaced={})], .032)
        self.assertEqual(gate.gate_status_n(result)[0], 'INCOMPLETE');result['checks'].pop('averaging_argmax')
        self.assertEqual(gate.gate_status_n(result), ('PASS', []))   # without the check (every other mode) nothing changes

    def test_one_sided_decisions_fail_and_refusals_on_both_sides_are_counted(self):
        unknown = blocks([(.5, .3, None)] + [(.5, .3, 0.)] * 7)
        one = gate.averaging_root(blocks([(.5, .3, 0.)] * 8), unknown, LOGITS, LEGAL)
        self.assertEqual((one['status'], one['refusal'][1], one['compared']), (['COMPLETE', 'REFUSED'], 'AVERAGING_UNKNOWN_PRICE', False))
        both = gate.averaging_root('BLOCK_REFUSED: turn2', 'BLOCK_REFUSED: turn2', LOGITS, LEGAL)
        error = gate.averaging_root(ValueError('bad'), ValueError('bad'), LOGITS, LEGAL);self.assertEqual(error['refusal'], ['AVERAGING_ERROR: bad'] * 2)
        verdict = gate.averaging_verdict([dict(roots=[one, both, error], replaced={})], .032)
        self.assertEqual((verdict['passed'], verdict['one_sided'], verdict['both_refused'], verdict['roots_compared']), (False, 1, 2, 0))
        self.assertFalse(gate.averaging_verdict([dict(roots=[both], replaced=dict(unmatched_head_leaves=1))], .032)['passed'])


class FlagTests(unittest.TestCase):
    def parse(self, *argv):
        return gate.parse(['--out', 'o', *MODE, *argv])[1]

    def refused(self, *argv, base=None):
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as raised:
            gate.parse(['--out', 'o', *(base if base is not None else MODE), *argv])
        self.assertEqual(raised.exception.code, 2, argv);return err.getvalue()

    def test_defaults_and_pairs(self):
        args = self.parse()
        self.assertEqual((args.constructor_worlds, args.constructor_seed, args.constructor_per_band), (8, 20261103, 10))
        self.assertEqual((args.constructor_roster, args.constructor_metadata, args.constructor_vocabulary), (['roster.json', 'a' * 64], ['metadata.json', 'b' * 64], ['vocabulary.json', 'c' * 64]))
        self.assertEqual((self.parse('--constructor-worlds', '8', '--constructor-per-band', '10', '--k1-own', '9').depth, args.declared_bounds), (2, str(AC)))

    def test_refusals(self):
        for argv in (('--n2-equivalence',), ('--max-roots', '4'), ('--worlds-blocks', '2'), ('--mutation-test', '--depth', '3'), ('--save-oracle',),
                ('--depth3-diagnostics',), ('--capacity', '256'), ('--block-lanes', '729'), ('--oracle-chunk', '64'), ('--constructor-worlds', '4'),
                ('--constructor-per-band', '0'), ('--constructor-seed', '-1'), ('--sample-leaves', '2')):
            with self.subTest(argv=argv):self.refused(*argv)
        for argv in (('--constructor-worlds', '16'), ('--constructor-per-band', '3'), ('--depth', '3'), ('--k-own', '3'), ('--k-opp', '9,9'), ('--k1-opp', '3'),
                ('--depth', '3', '--selection', 'searched')):   # only what A-C covers
            with self.subTest(argv=argv):self.assertIn('declaration A-C covers', self.refused(*argv))
        self.assertIn('keeps declaration A', self.refused(base=(*RUNTIME, *CHAMPION, *FLAGS, *COVERED, *DECLARED, '--fixed-shapes')))
        self.assertIn('needs --champion-archive', self.refused(base=(*RUNTIME, *FLAGS, *HARNESS, *COVERED)))
        self.assertIn('refused with --d3-root', self.refused(base=('--d3-root', 'r', '--chunk', '0', *CHAMPION, *FLAGS, *HARNESS, *COVERED, *DECLARED)))
        self.assertIn('needs --constructor-vocabulary', self.refused(base=(*RUNTIME, *CHAMPION, *FLAGS[:-3], *HARNESS, *COVERED, *DECLARED)))
        self.assertIn('a 64-hex sha256', self.refused(base=(*RUNTIME, *CHAMPION, *FLAGS[:-1], 'x', *HARNESS, *COVERED, *DECLARED)))
        self.assertIn('need --constructor-roster', self.refused(base=(*RUNTIME, *CHAMPION, '--constructor-worlds', '8')))
        self.assertIn('need --constructor-roster', self.refused(base=(*RUNTIME, *CHAMPION, '--showdown-root', 's')))
        self.assertIn('refused with --diagnostics-from', self.refused(base=('--diagnostics-from', 'x', *FLAGS)))
        self.assertIn('takes no other option', self.refused(base=('--reverdict', 'gate.json', '--constructor-worlds', '8')))

    def test_the_mode_runs_under_declaration_a_c_only(self):
        a = gate.REPO_ROOT / gate.DECLARED_SUBDIR / 'champion-9375-bf16-r1r3-20261107.json';a_sha = hashlib.sha256(a.read_bytes()).hexdigest()
        base = (*RUNTIME, *CHAMPION, *FLAGS, *HARNESS, *COVERED)
        for declared in ((), ('--declared-bounds', str(a), '--declared-bounds-sha256', a_sha, '--declared-bounds-commit', '4' * 40),   # declaration A itself
                ('--declared-bounds', str(AC), '--declared-bounds-sha256', a_sha, '--declared-bounds-commit', '4' * 40),   # A-C's path, another sha256
                ('--declared-bounds', str(a), *DECLARED[2:])):   # A's path, A-C's sha256
            with self.subTest(declared=declared[1:2]):self.assertIn('under declaration A-C only', self.refused(base=(*base, *declared)))
        self.assertEqual(gate.repo_path_of(AC), gate.CONSTRUCTOR_DECLARATION['repo_path']);self.assertEqual(hashlib.sha256(AC.read_bytes()).hexdigest(), gate.CONSTRUCTOR_DECLARATION['sha256'])

    def test_the_mode_refuses_a_head_and_sampled_leaves_itself(self):
        """check_constructor_arguments' own refusals, beside the champion's and the declaration's (which refuse a head first on the command line)."""
        parser, args = gate.parse(['--out', 'o', *MODE])
        for name, value in (('head_archive', 'head.npz'), ('sample_leaves', 2)):
            with self.subTest(name=name), contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
                gate.check_constructor_arguments(parser, SimpleNamespace(**dict(vars(args), **{name: value})))
            self.assertIn('refused with --' + name.replace('_', '-'), err.getvalue())


class DeclarationACTests(unittest.TestCase):
    PATH = gate.REPO_ROOT / gate.DECLARED_SUBDIR / 'champion-9375-bf16-r1c-20261108.json'
    A = gate.REPO_ROOT / gate.DECLARED_SUBDIR / 'champion-9375-bf16-r1r3-20261107.json'

    def test_a_c_is_declaration_a_with_a_new_scope_and_passes_the_gate(self):
        raw = self.PATH.read_bytes();content = json.loads(raw);a = json.loads(self.A.read_bytes())
        self.assertEqual(gate.declared_problems(content), [])
        loaded = gate.declared_from_bytes(raw, hashlib.sha256(raw).hexdigest(), path=self.PATH)
        self.assertEqual(gate.declared_numbers(loaded), gate.declared_numbers(dict(content=a)))
        for key in ('schema', 'evaluator', 'bounds', 'source_measurement', 'declared_by'):self.assertEqual(content[key], a[key], key)
        self.assertEqual(gate.repo_path_of(self.PATH), 'gpu_engine/declared_bounds/champion-9375-bf16-r1c-20261108.json')
        self.assertTrue(content['derivation'].startswith('SCOPE: declaration A-C'));self.assertNotEqual(content['derivation'], a['derivation'])
        for text in ('R1-C', '--constructor-roster', 'R1.b.*', 'NOT_APPLICABLE', 'no new float site', 'JAX_DEFAULT_MATMUL_PRECISION=highest'):
            self.assertIn(text.lower(), content['derivation'].lower(), text)
        self.assertGreaterEqual(gate._utc(content['declared_at']), gate._utc(a['declared_at']))
        self.assertEqual(gate.evaluator_mismatches(content, gate.args_identity(gate.parse(['--out', 'o', *RUNTIME, '--champion-archive', 'x',
            '--champion-archive-sha256', a['evaluator']['archive_sha256'], '--champion-checkpoint-sha256', a['evaluator']['checkpoint_sha256']])[1])), [])


@unittest.skipUnless(N.HAVE_PRICER, 'gpu_engine.device_depthn_pricer / depthn_backups are not on this tree')
class ConstructorRunTests(C.ChampionFixture, unittest.TestCase):
    """ChampionFixture's bound requests as constructor roots: root n is request ROOTS[n] priced as K = 2 blocks of its own materialized world
    pair, all under ONE PrebuiltWorlds with the fixture source's binding (the real DepthNPricer validates it); no FramePricer."""
    ROOTS = (0, 1, 3, 7, 16, 17, 25, 26)   # plain; chance re-keyed; a missing world; pre-refused (shifted root inputs); terminal leaves; waiting; interaction

    def ctor(self, signatures=None):
        requests, pairs, roots = [], [], []
        for n, q in enumerate(self.ROOTS):
            r = self.requests[q];pair = tuple(self.materialize(r.plans.root, s, self.tables, self.recipe) for s in r.plans.starts)
            rows = (len(requests), len(requests) + 1);requests += [r, r];pairs += [pair, pair]
            roots.append(dict(index=n, ordinal=100 + q, band=n % 4, turn=2, selected_by='band', top_up_for=[], classes=[], root_class='other', class_source='test',
                rows=rows, carry_signature=(signatures or {}).get(n, 'sig'), address=[r.plans.root.address.stream, r.plans.root.address.frame, r.plans.root.address.owner],
                capture=dict(capture_path='c', capture_row=q), b2_view=dict(equal=True)))
        source = self.tables.source;binding = dn().WorldBinding(source.source_hash, source.source_rows_sha256, source.binding_sha256, 'm' * 64, ())
        return SimpleNamespace(requests=tuple(requests), worlds=dn().PrebuiltWorlds('test-constructor', binding, tuple(pairs)), binding=binding, roots=roots,
            executor=replace(self.ex),   # the live executor: the fake engine, a distinct object from any other
            selection=dict(bands={}, top_up={}, refused=[], attempted=len(roots), selected=len(roots)), pins=dict(mode=gate.CONSTRUCTOR_MODE, worlds=4, seed=1),
            source=dict(constructor_roster='roster.json', runtime=dict(fixture=True)))

    def run_worlds(self, ctor, *, lanes=gate.INVARIANCE_LANES, mutation=None, d3=False):
        """One call: ctor's first constructor call, as run_gate makes it (d3: the same requests through D3's own worlds and FramePricer)."""
        pricers = self.champion_pricers(depth=2, k=9, lanes=lanes);call = gate.constructor_calls(ctor, 32)[0]
        depth2_pricer = gate.depth2_pass_pricer(pricers.module, self.ex, object(), capacity=32, actor_capacity=3, digest=True, forward=self.champion)
        seam = dict(materializer=self.tables, frame_pricer=self.frame_pricer()) if d3 else dict(materializer=ctor.binding, frame_pricer=None,
            worlds=call['worlds'], averaging_roots=call['roots'])
        with self.engine():
            evaluation, summary = gate.run_call_n(requests=call['requests'], recipe=self.recipe, params=self.params, pricers=pricers, depth2_pricer=depth2_pricer,
                oracle=self.oracle, sample=None, block=0, context=dict(block=0, call=0), mutation=mutation, d2_consistency=True, **seam)
        return evaluation, summary, call

    def test_one_worlds_object_reaches_every_consumer_and_b_is_absent(self):
        module = dn();real_price, real_stack = module.DepthNPricer.price, depthn_oracle.stack;priced, stacked = [], []
        def price(pricer, requests, materializer, recipe, params, **kw):
            priced.append((kw.get('worlds'), materializer));return real_price(pricer, requests, materializer, recipe, params, **kw)
        def stack(requests, worlds, **kw):
            stacked.append(worlds);return real_stack(requests, worlds, **kw)
        ctor = self.ctor()
        with patch.object(module.DepthNPricer, 'price', price), patch.object(depthn_oracle, 'stack', stack), \
                patch.object(batch, '_numpy_worlds', side_effect=AssertionError('D3 worlds materialized')), \
                patch.object(batch.FramePricer, 'price', side_effect=AssertionError('D3 priced')):
            evaluation, summary, call = self.run_worlds(ctor)
        self.assertEqual(len(priced), 3)   # the depth-2 pass, the 729-lane twin, the checked 2187-lane pricer
        self.assertTrue(all(w is call['worlds'] and m is ctor.binding for w, m in priced));self.assertEqual(len(stacked), 1);self.assertIs(stacked[0], call['worlds'].pairs)
        self.assertNotIn('b', evaluation);self.assertNotIn('b_discrete', summary['passed']);self.assertEqual(evaluation['declared_cover']['b'], {})
        self.assertNotIn('depth1', summary['refusals']);self.assertNotIn('frame_pricer', summary['costs'])
        result, (status, reasons) = self.judged(evaluation);gate.constructor_checks(result, [evaluation]);status, reasons = gate.gate_status_n(result)
        self.assertEqual(status, 'PASS', (reasons, N.failing(result)))
        self.assertEqual(set(result['checks']), {'a', 'c', 'd', 'oracle_consistency', 'e', 'f', 'averaging_argmax', 'block_invariance'})
        self.assertNotIn('depth1_noise_floor', result['checks']['c']['quantities']);self.assertIsNone(result['tolerance']['depth1_noise_floor'])
        averaging = result['checks']['averaging_argmax']
        self.assertEqual((averaging['roots'], averaging['agreement_rate'], averaging['one_sided']), (len(self.ROOTS), 1., 0))
        self.assertEqual(averaging['roots_compared'] + averaging['both_refused'], len(self.ROOTS));self.assertGreaterEqual(averaging['both_refused'], 2)   # 3 and 7 refuse
        self.assertEqual(averaging['roots_compared'], result['checks']['e']['roots_compared'] // 2)   # (e) compares each of a root's two blocks
        self.assertEqual({r['ordinal']: r['refusal'] for r in averaging['per_root'] if not any(s == 'COMPLETE' for s in r['status'])},
            {103: ['BLOCK_REFUSED: materialization'] * 2, 107: ['BLOCK_REFUSED: pre'] * 2})   # the live search's block rule, before deciding
        self.assertEqual((averaging['premise_failures'], [r['premise_holds'] for r in averaging['per_root'] if r['status'] == ['COMPLETE'] * 2]),
            (0, [True] * averaging['roots_compared']));self.assertLessEqual(averaging['max_premise_gap'], gate.REL_EPS)
        self.assertEqual(summary['averaging_argmax'], dict(roots=len(self.ROOTS), compared=averaging['roots_compared'], agree=averaging['roots_compared']))

    def test_the_worlds_path_prices_the_d3_path_bit_for_bit(self):
        """The same requests through D3's materialized worlds and through the pre-built ones: every frame array bit-equal, (a), (c)-(f) the same."""
        ctor = self.ctor();mine, _, call = self.run_worlds(ctor, lanes=(2187,), mutation={});d3, _, _ = self.run_worlds(ctor, lanes=(2187,), mutation={}, d3=True)
        a, b = (gate.frame_arrays(e[gate.MUTATION_INPUTS]['frame']) for e in (mine, d3))
        self.assertEqual(set(a), set(b))
        for name in a:
            with self.subTest(name=name):self.assertTrue(np.array_equal(a[name], b[name], equal_nan=a[name].dtype.kind == 'f'))
        self.assertIn('b', d3);self.assertTrue(d3['b']['passed'])
        for key in ('a', 'd', 'oracle'):self.assertEqual(mine[key]['compared'], d3[key]['compared'], key)
        floors = gate.g2.FLOORS
        self.assertEqual({k: v['max_abs_difference'] for k, v in mine['c'].items()}, {k: v['max_abs_difference'] for k, v in d3['c'].items() if k not in floors})
        for key in ('roots_compared', 'max_abs_delta_q', 'argmax_q_differs'):self.assertEqual(mine['e'][key], d3['e'][key], key)
        self.assertEqual(mine['f']['comparisons'], d3['f']['comparisons'])

    def test_the_averaging_check_fails_on_oracle_heads_that_contradict_e(self):
        """oracle_heads_frame_n's copy with every leaf value negated: the W-world argmax flips by margins far beyond 2 x q_noise_bound (FAIL), as
        (e) does."""
        real = gate.oracle_heads_frame_n
        def negated(frame, data):
            swapped, counts = real(frame, data);return gate._copy(swapped, leaf_value=-np.asarray(swapped.leaf_value)), counts
        with patch.object(gate, 'oracle_heads_frame_n', negated):evaluation, _, _ = self.run_worlds(self.ctor(), lanes=(2187,))
        result, _ = self.judged(evaluation);gate.constructor_checks(result, [evaluation]);averaging = result['checks']['averaging_argmax']
        self.assertFalse(averaging['passed']);self.assertGreater(averaging['disagree_beyond_bound'], 0);self.assertFalse(result['checks']['e']['passed'])
        self.assertEqual(sum(r['margin_oracle'] > averaging['bound'] for r in averaging['disagreements']), averaging['disagree_beyond_bound'])
        self.assertEqual(gate.gate_status_n(result)[0], 'FAIL')

    def test_the_premise_is_checked_against_es_own_reduction(self):
        """averaging's input backed up under another rule (depthn_backups.average for learner_max) still decides, but its q is no longer the mean of
        (e)'s per-block completed-Q (learner_max_root): the premise fails and so does the check."""
        evaluation, _, call = self.run_worlds(self.ctor(), lanes=(2187,), mutation={});held = evaluation[gate.MUTATION_INPUTS]
        real = importlib.import_module('gpu_engine.depthn_backups')
        proxy = SimpleNamespace(backup=real.backup, learner_max=real.average, average=real.average, learner_max_root=real.learner_max_root,
            refusal_summary=real.refusal_summary)
        report = gate.averaging_call(held['frame'], held['data'], call['requests'], call['roots'], backups=proxy)
        verdict = gate.averaging_verdict([report], 1e-4);decided = [r for r in report['roots'] if r['compared']]
        self.assertTrue(decided);self.assertFalse(verdict['passed']);self.assertGreater(verdict['premise_failures'], 0);self.assertGreater(verdict['max_premise_gap'], 1e-3)
        clean = gate.averaging_call(held['frame'], held['data'], call['requests'], call['roots'], backups=real)
        self.assertEqual([r['premise_holds'] for r in clean['roots'] if r['compared']], [True] * len(decided))

    def test_bind_constructor_runs_bind_roots_and_fails_closed(self):
        requests = self.ctor().requests;g2 = gate.g2
        with patch.object(batch.v5_model, 'forward', side_effect=C.no_v2):
            bound, values, checked = gate.bind_constructor(requests, self.params, None, self.champion)
            self.assertEqual((checked, len(bound), len(values)), (True, len(requests), len(requests)))
            self.assertEqual([(b.root_logits, b.learner_actor, b.opponent_actor) for b in bound], [(r.root_logits, r.learner_actor, r.opponent_actor) for r in requests])
            moved = list(requests);moved[3] = replace(moved[3], root_logits=tuple(x + 1. for x in moved[3].root_logits))
            with self.assertRaisesRegex(ValueError, r'other root logits or actors than build at request\(s\) \[3\]'):gate.bind_constructor(moved, self.params, None, self.champion)
            actors = list(requests);actors[0] = replace(actors[0], learner_actor=1)
            with self.assertRaisesRegex(ValueError, r'request\(s\) \[0\]'):gate.bind_constructor(actors, self.params, None, self.champion)
        dropping = lambda groups, params, plan, evaluator: ([groups[0][1:]], [], [[1, 2, 3]], True)
        with patch.object(g2, 'bind_roots', dropping), self.assertRaisesRegex(ValueError, 'dropped 1 constructor request'):
            gate.bind_constructor(requests, self.params, None, self.champion)
        unchecked = lambda groups, params, plan, evaluator: (groups, [], [], False)
        with patch.object(g2, 'bind_roots', unchecked), self.assertRaisesRegex(ValueError, 'no evaluator check'):gate.bind_constructor(requests, self.params, None, self.champion)

    def declaration(self, **bounds):
        """A declaration for this champion committed in a scratch repository (main() takes it as the checkout and as the pinned A-C)."""
        measurement = Path(tempfile.mkdtemp(prefix='r1c-measure-')) / 'm.json';measurement.write_bytes(b'{}\n')
        content = N.declaration(evaluator=dict(kind='champion', archive_sha256='f' * 64, precision='bfloat16', checkpoint_sha256=C.CHECKPOINT),
            bounds=N.bounds(**dict(dict(value_bound=1e-6, turn1_logit_bound=.05, root_logit_bound=.04, near_tie_q_bound=2e-6), **bounds)),
            source_measurement=dict(path=str(measurement), sha256=hashlib.sha256(measurement.read_bytes()).hexdigest()))
        return N.committed(content)

    def main(self, ctor, *argv, out=None, lines=None, record=None):
        """main() under --constructor-roster with the fixture's harness and `record` (default: self.declaration()) as declaration A-C."""
        record = record or self.declaration();self.record = record
        out = out or Path(tempfile.mkdtemp(prefix='r1c-cli-')) / 'out';log = (lambda m, **k: lines.append(dict(k, message=m))) if lines is not None else (lambda *a, **k: None)
        loaded = SimpleNamespace(provider=SimpleNamespace(executor=None, materializer=self.tables, recipe=self.recipe, source=self.tables.source),   # the bank's
            plan=SimpleNamespace(model=object(), historical_window=2), source=dict(fixture=True))                                         # executor never prices
        self.seen = seen = dict(pricer=[], oracle=[]);pricer_init, oracle_init = dn().DepthNPricer.__init__, depthn_oracle.DepthNOracle.__init__
        def record_pricer(pricer, executor, *a, **k):seen['pricer'].append(executor);pricer_init(pricer, executor, *a, **k)
        def record_oracle(oracle, executor, *a, **k):seen['oracle'].append(executor);oracle_init(oracle, executor, *a, **k)
        roots = SimpleNamespace(loaded=loaded, head=None, params=self.params, groups=None, root_values=[.1, .2], forward=self.champion, actor_capacity=1,
            evaluator=dict(override=True, **self.champion.header(), arm_checked=True, dropped_roots=[]), constructor=ctor)
        with self.engine(), patch.object(gate, 'constructor_roots', lambda args: roots), patch.object(gate, 'load_roots', side_effect=AssertionError('D3 roots')), \
                patch.object(gate, 'log', log), patch.object(gate, 'CONSTRUCTOR_HARNESS', dict(capacity=32, block_lanes=2187, oracle_chunk=64)), \
                patch.object(batch.FramePricer, 'price', side_effect=AssertionError('D3 priced')), patch.object(gate, 'REPO_ROOT', record.root), \
                patch.object(gate, 'CONSTRUCTOR_DECLARATION', dict(repo_path=record.path.relative_to(record.root).as_posix(), sha256=record.sha256)), \
                patch.object(dn().DepthNPricer, '__init__', record_pricer), patch.object(depthn_oracle.DepthNOracle, '__init__', record_oracle):
            code = gate.main(['--out', str(out), *RUNTIME, *CHAMPION, *FLAGS, *HARNESS, *COVERED, '--capacity', '32', '--oracle-chunk', '64',
                '--declared-bounds', str(record.path), '--declared-bounds-sha256', record.sha256, '--declared-bounds-commit', record.commit, *argv])
        return code, out

    def reverdicted(self, out):
        with patch.object(gate, 'log', lambda *a, **k: None), patch.object(gate, 'REPO_ROOT', self.record.root):
            return gate.main(['--reverdict', str(out / 'gate.json')])

    def test_the_command_line_run_reports_the_mode_and_reverdicts(self):
        """main() under --constructor-roster: two carry signatures price as two calls; gate.json is CONSTRUCTOR_ROOTS with (b) NOT_APPLICABLE, the
        parity header's (b) entry, the constructor_roots header with the matmul precision, and averaging_argmax; --reverdict accepts it."""
        lines = [];ctor = self.ctor(signatures={1: 'other', 4: 'other'});code, out = self.main(ctor, lines=lines);report = json.loads((out / 'gate.json').read_text())
        self.assertEqual((code, report['status'], report['mode']), (0, 'PASS', gate.CONSTRUCTOR_MODE), report.get('status_reasons'))
        self.assertEqual(set(report['checks']), {'a', 'c', 'd', 'oracle_consistency', 'e', 'f', 'averaging_argmax', 'block_invariance'})
        self.assertEqual(report['not_applicable'], dict(b=dict(status='NOT_APPLICABLE', reason=gate.NO_B)))
        self.assertEqual([c['roots'] for c in report['constructor']['calls']], [[100, 103, 107, 117, 125, 126], [101, 116]])
        self.assertEqual([c['carry_signature'] for c in report['constructor']['calls']], ['sig', 'other']);self.assertEqual(len(report['calls']), 2)
        self.assertEqual(sum(1 for x in lines if x['message'] == 'CALL_CHECKED'), 2);self.assertEqual(next(x for x in lines if x['message'] == 'GATE_START')['mode'], gate.CONSTRUCTOR_MODE)
        # LIVE_EXECUTOR: the depth-2 pass, the checked pricer, its block-invariance twin and the oracle all hold the constructor's executor
        self.assertEqual((len(self.seen['pricer']), len(self.seen['oracle'])), (3, 1))
        self.assertTrue(all(e is ctor.executor for e in self.seen['pricer'] + self.seen['oracle']), self.seen)
        header = report['header'];parity = header['merged_depth2_parity']['remaining_differences']
        self.assertEqual(parity[-1], gate.CONSTRUCTOR_B_DIFFERENCE);self.assertEqual(len(parity), len(gate.MERGED_DEPTH2_DIFFERENCES) + 1)
        matched = header['merged_depth2_parity']['matched'];self.assertEqual(matched, gate.MERGED_DEPTH2_MATCHED_CONSTRUCTOR)
        self.assertNotIn(gate.B_MATCHED, matched);self.assertIn(gate.B_MATCHED, gate.MERGED_DEPTH2_MATCHED);self.assertIn('(b) NOT made', matched)
        self.assertEqual(gate.merged_depth2_parity('gate', oracle_consistency=True)['matched'], gate.MERGED_DEPTH2_MATCHED)   # every other mode as before
        import jax
        self.assertEqual(header['constructor_roots']['matmul_precision']['jax_default_matmul_precision'], jax.config.jax_default_matmul_precision)
        self.assertEqual(header['constructor_roots']['mode'], gate.CONSTRUCTOR_MODE);self.assertEqual(report['source'], ctor.source)
        averaging = report['checks']['averaging_argmax'];self.assertEqual((averaging['agreement_rate'], averaging['bound']), (1., 2 * report['tolerance']['q_noise_bound']))
        self.assertEqual(report['refusals']['dropped_roots']['count'], 0);self.assertNotIn('depth1', report['refusals'])
        self.assertEqual(report['constructor']['class_counts']['primary'], dict(other=len(self.ROOTS)));self.assertEqual(report['constructor']['b2_view']['equal'], len(self.ROOTS))
        self.assertEqual(report['header']['declared_bounds']['sha256'], self.record.sha256)
        self.assertEqual(self.reverdicted(out), 0);again = json.loads((out / gate.REVERDICT_FILE).read_text())
        self.assertEqual((again['status'], again['mode'], again['reverdict']['changed_quantities']), ('PASS', gate.CONSTRUCTOR_MODE, []))
        self.assertEqual(again['header']['constructor_roots'], header['constructor_roots']);self.assertEqual(again['checks']['averaging_argmax'], averaging)
        self.assertEqual(gate.reverdict_missing(dict(report, mode='OTHER'))[0], f'mode SAVED_ROOTS|BANK_CHUNK|{gate.CONSTRUCTOR_MODE} (got \'OTHER\')')

    def test_the_declared_numbers_reach_every_site_without_b(self):
        """Under its declaration the constructor run PASSes with no (b) site; averaging_argmax takes 2 x the declared near_tie_q_bound."""
        code, out = self.main(self.ctor(), record=self.declaration(near_tie_q_bound=3e-6));report = json.loads((out / 'gate.json').read_text())
        self.assertEqual((code, report['status']), (0, 'PASS'), report.get('status_reasons'));self.assertNotIn('b', report['checks'])
        self.assertEqual((report['tolerance']['value_bound'], report['tolerance']['q_noise_bound'], report['checks']['averaging_argmax']['bound']), (1e-6, 3e-6, 6e-6))
        self.assertFalse(any(k.startswith('b.') for k in report['tolerance']['guard_declared']['modes']));self.assertEqual(self.reverdicted(out), 0)


class Leaf:
    pass


def nested(**changes):
    """A compiled-table-like structure: a NamedTuple holding arrays (jax and numpy), None, a dataclass, a dict and a list."""
    import dataclasses, typing
    import jax.numpy as jnp
    Inner = dataclasses.make_dataclass('Inner', [('weights', object), ('flag', object)], frozen=True)
    Outer = typing.NamedTuple('Outer', [('power', object), ('chart', object), ('effects', object), ('inner', object), ('lookup', object), ('rows', object)])
    parts = dict(power=jnp.asarray([0, 40, 90], jnp.int32), chart=np.ones((2, 2), np.float32), effects=None,
        inner=Inner(np.arange(3, dtype=np.int16), True), lookup={'b': np.zeros(1, np.uint8), 'a': 7}, rows=[np.float32(.5), 'x'])
    parts.update(changes);return Outer(**parts)


class TableIdentityTests(unittest.TestCase):
    """constructor_worlds' compiled-table walk, digest and differences on a synthetic structure."""
    def test_equal_structures_digest_alike_and_compare_equal(self):
        from gpu_engine import constructor_worlds as cw
        import jax.numpy as jnp
        a, b = nested(), nested(power=np.asarray([0, 40, 90], np.int32), lookup={'a': 7, 'b': jnp.zeros(1, jnp.uint8)})   # numpy for jax, keys reordered
        self.assertEqual(cw.tables_differences(a, b), []);self.assertEqual(cw.tables_digest(a), cw.tables_digest(b))
        paths = [x[0] for x in cw.table_leaves(a)]
        self.assertEqual(paths, ['tables', 'tables.power', 'tables.chart', 'tables.effects', 'tables.inner', 'tables.inner.weights', 'tables.inner.flag',
            'tables.lookup', "tables.lookup['a']", "tables.lookup['b']", 'tables.rows', 'tables.rows[0]', 'tables.rows[1]'])
        self.assertEqual(len(cw.tables_digest(a)), 64);self.assertEqual(cw.engine_tables_digest((a, b)), cw.tables_digest(dict(engine=a, observation=b)))

    def test_every_kind_of_difference_is_named_by_path(self):
        from gpu_engine import constructor_worlds as cw
        base = nested()
        for change, text in ((dict(power=np.asarray([0, 41, 90], np.int32)), 'tables.power: bytes differ (int32 [3]; first differing byte 4)'),
                (dict(power=np.asarray([0, 40, 90], np.int64)), 'tables.power: array'), (dict(chart=np.ones((2, 3), np.float32)), 'tables.chart: array'),
                (dict(effects=np.zeros(1)), 'tables.effects: scalar'), (dict(lookup={'a': 8, 'b': np.zeros(1, np.uint8)}), "tables.lookup['a']: scalar"),
                (dict(lookup={'a': 7}), "tables.lookup: node"), (dict(rows=[np.float32(.5)]), 'tables.rows: node')):
            with self.subTest(text=text):
                found = cw.tables_differences(base, nested(**change));self.assertTrue(found and found[0].startswith(text), found)
                self.assertNotEqual(cw.tables_digest(base), cw.tables_digest(nested(**change)))
        with self.assertRaisesRegex(TypeError, r'tables.effects: unsupported compiled-table member'):cw.table_leaves(nested(effects=Leaf()))
        import jax.numpy as jnp
        brain = cw.table_leaves(nested(chart=jnp.ones((2, 2), jnp.bfloat16)))[2];half = cw.table_leaves(nested(chart=np.ones((2, 2), np.float16)))[2]
        self.assertEqual((brain[2], half[2]), (('<V2', 'bfloat16', (2, 2)), ('<f2', 'float16', (2, 2))))   # bfloat16 is named, not only <V2
        self.assertTrue(cw.tables_differences(nested(chart=jnp.ones((2, 2), jnp.bfloat16)), nested(chart=np.zeros((2, 2), np.dtype('V2'))))[0].startswith('tables.chart: array'))


@requires_showdown('the tables bridge compiles the real Gen 3 engine tables')
@unittest.skipUnless(shutil.which('node'), 'node is needed to read the generator\'s static metadata')
class TablesBridgeTests(unittest.TestCase):
    """TABLES_BRIDGE through the real code paths (transition.tables_from_metadata, engine_device_observation.compile_tables) on the generator's
    own static metadata and the real encoder catalog: documents differing only in the team-sample statistics compile to identical tables; a
    perturbed moves entry or type chart value fails, naming the differing path."""
    STATISTICS = ('move_frequency', 'move_slots', 'seed_start', 'target85_move_count')   # the four keys the bank and Codex's metadata differ in

    @classmethod
    def setUpClass(cls):
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.randbat import load_gen3_randbat_source_cached
        from pokezero.randbat_vocab import gen3_category_vocabulary
        from pokezero.recorded_mirror_roots import CHAMPION_CONTRACT
        from gpu_engine import public_belief, v4_encoder
        root = showdown_root()
        script = (r"""const fs=require('fs'),vm=require('vm'),s=fs.readFileSync(process.argv[1],'utf8'),start=s.indexOf('async function generate('),"""
            r"""body=s.slice(s.indexOf('{',start)+1,s.indexOf('  async function emit(',start)).split('\n').filter(x=>!x.includes("const {Teams}")&&"""
            r"""!x.includes('const rngPackage')).join('\n');const code=s.slice(0,start)+'\nfunction metadata(root){'+body+"return {format:'gen3randombattle',"""
            r"""compiled_tree:compiledTree(path.join(root,'dist')),vocabulary,moves,type_chart:chart,stats_order:STATS};}\nJSON.stringify(metadata(ROOT))";"""
            r"""process.stdout.write(vm.runInNewContext(code,{require,ROOT:process.argv[2]}));""")
        cls.metadata = json.loads(subprocess.check_output(['node', '-e', script, str(gate.REPO_ROOT / 'gpu_engine/generate_teams.cjs'), str(root)], text=True))
        vocabulary = gen3_category_vocabulary(root, include_feature_pack_v4=True)
        cls.catalog = v4_encoder.compile_encoder(public_belief.compile_source(load_gen3_randbat_source_cached(root)), load_showdown_dex_cached(root), vocabulary,
            checkpoint_sha256='a' * 64, feature_masks=CHAMPION_CONTRACT['feature_masks'])

    def documents(self):
        """(bank, constructor) metadata: the generator's static metadata with the four statistics keys at different values on each side."""
        bank = dict(json.loads(json.dumps(self.metadata)), move_frequency={'tackle': 3}, move_slots=[[1, 2]], seed_start=1, target85_move_count=85)
        live = dict(json.loads(json.dumps(self.metadata)), move_frequency={'tackle': 4, 'surf': 1}, move_slots=[[2, 1], [3]], seed_start=9, target85_move_count=86)
        return bank, live

    def test_the_statistics_keys_compile_to_identical_tables(self):
        from gpu_engine import constructor_worlds as cw
        bank, live = self.documents();reference = cw.engine_tables(bank, self.catalog)
        tables, record = gate.tables_bridge(bank, self.catalog, live, self.catalog, reference=reference)
        self.assertEqual(record['metadata_keys_differing'], sorted(self.STATISTICS));self.assertTrue(record['equal'] and record['bank_executor_reproduced'])
        self.assertNotEqual(record['bank_metadata_canonical_sha256'], record['constructor_metadata_canonical_sha256'])
        self.assertEqual(record['digest'], cw.engine_tables_digest(reference));self.assertEqual(record['digest'], cw.engine_tables_digest(tables))
        self.assertEqual(record['digest_function'], 'gpu_engine.constructor_worlds.engine_tables_digest');self.assertGreater(record['entries'], 200)

    def test_a_perturbed_moves_entry_or_type_chart_value_fails_naming_the_path(self):
        bank, live = self.documents();at = next(i for i, row in enumerate(live['moves']) if row and type(row.get('base_power')) is int and row['base_power'] > 0)
        moved = json.loads(json.dumps(live));moved['moves'][at]['base_power'] += 10
        with self.assertRaisesRegex(ValueError, r"compiled tables differ \(TABLES_BRIDGE\): tables\['engine'\]\.power: bytes differ"):
            gate.tables_bridge(bank, self.catalog, moved, self.catalog)
        chart = json.loads(json.dumps(live));row, col = next((r, c) for r, line in enumerate(chart['type_chart']) for c, x in enumerate(line) if x == 1)
        chart['type_chart'][row][col] = 2
        with self.assertRaisesRegex(ValueError, r"tables\['engine'\]\.type_num: bytes differ"):gate.tables_bridge(bank, self.catalog, chart, self.catalog)

    def test_a_catalog_that_changes_only_the_observation_tables_fails_naming_them(self):
        """The same metadata with a catalog whose stats_block mask is flipped: the engine tables are equal, the observation tables are not."""
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.randbat import load_gen3_randbat_source_cached
        from pokezero.randbat_vocab import gen3_category_vocabulary
        from pokezero.recorded_mirror_roots import CHAMPION_CONTRACT
        from gpu_engine import constructor_worlds as cw, public_belief, v4_encoder
        root = showdown_root();masks = dict(CHAMPION_CONTRACT['feature_masks'], stats_block=not CHAMPION_CONTRACT['feature_masks']['stats_block'])
        flipped = v4_encoder.compile_encoder(public_belief.compile_source(load_gen3_randbat_source_cached(root)), load_showdown_dex_cached(root),
            gen3_category_vocabulary(root, include_feature_pack_v4=True), checkpoint_sha256='a' * 64, feature_masks=masks)
        bank, live = self.documents();mine, theirs = cw.engine_tables(live, self.catalog), cw.engine_tables(live, flipped)
        self.assertEqual(cw.tables_differences(dict(engine=mine[0]), dict(engine=theirs[0])), [])   # the engine tables do not read the catalog
        with self.assertRaisesRegex(ValueError, r"compiled tables differ \(TABLES_BRIDGE\): tables\['observation'\]\."):
            gate.tables_bridge(bank, self.catalog, live, flipped)

    def test_the_bank_executors_own_tables_are_held_to_their_recompile(self):
        from gpu_engine import constructor_worlds as cw
        bank, live = self.documents();moved = json.loads(json.dumps(bank));moved['type_chart'][0][0] = 2 if moved['type_chart'][0][0] != 2 else 1
        with self.assertRaisesRegex(ValueError, "the bank executor's tables and their recompile from its metadata differ"):
            gate.tables_bridge(bank, self.catalog, live, self.catalog, reference=cw.engine_tables(moved, self.catalog))


class WiringTests(unittest.TestCase):
    """constructor_roots with every collaborator it loads faked (the D3 runtime, the champion, Codex's constructor inputs, the roster, prepare,
    the classifier, build and the bind): the selection, the pins, the refusal list and the fail-closed paths are wired as documented."""
    ARGS = SimpleNamespace(constructor_roster=('roster.json', 'a' * 64), constructor_metadata=('metadata.json', 'b' * 64),
        constructor_vocabulary=('vocabulary.json', 'c' * 64), showdown_root='showdown', node_binary='node', constructor_worlds=4, constructor_seed=3,
        constructor_per_band=2)

    def run_roots(self, *, refuse_build=False, refused=(1, 6), unavailable=(5,), turns=(2,) * 4 + (5,) * 4 + (12,) * 4 + (25,) * 4, bridge_error=None,
            bank_binding=dict(metadata_sha256='bank-m')):
        from gpu_engine import constructor_worlds as cw, depthn_live_search as live, phase_d_protocol_constructor as pc
        made, self.calls = [], collections.defaultdict(list)
        provider = SimpleNamespace(executor='EX', source='SRC', prepared=SimpleNamespace(metadata_json='{"m": 1, "move_frequency": 2}', catalog='CAT',
            engine_tables='BANK-E', observation_tables='BANK-O'), recipe=SimpleNamespace(seed=11))
        def bridge(bank_metadata, bank_catalog, metadata, catalog, *, reference=None):
            self.calls['bridge'].append((bank_metadata, bank_catalog, metadata, catalog, reference))
            if bridge_error:raise bridge_error if isinstance(bridge_error, Exception) else ValueError(bridge_error)
            return ('E', 'O'), dict(digest='9' * 64, equal=True)
        def executor(metadata, catalog, *, tables=None):self.calls['executor'].append((metadata, catalog, tables));return 'LIVE'
        loaded = SimpleNamespace(provider=provider, plan='PLAN', source=dict(saved_roots='saved'))
        binding = dn().WorldBinding('s', 'r', 'b', 'm', (('checkpoint_sha256', 'e' * 64),));bank = replace(binding, **bank_binding)
        def bind(source, metadata, catalog):   # the bank's inputs give the bank's binding, the constructor's the live one
            self.calls['binding'].append((source, metadata, catalog));return bank if source == 'SRC' else binding
        class Constructor:
            def __init__(self, **kw):made.append(kw)
        def request(meta, i, *, roster_sha256, worlds):
            if i in unavailable:raise ValueError('same original genuine captured root unavailable')
            return dict(request_id=f'{roster_sha256[:4]}:{worlds}:{i}', i=i), dict(capture_path='wire.jsonl', capture_row=i, capture_declaration_sha256='d' * 64,
                capture_chain_sha256=str(i) * 64, prefix_sha256='f' * 64)
        def prepare(constructor, requests, *, start_index, device):
            (r,) = requests;i = r['i'];self.calls['prepare'].append(i)
            if i in refused:return (cw.RootRefusal(r['request_id'], 'SEARCH_PREFIX_OR_REQUEST_MISMATCH', '', .5, 'B'),)
            partial = i == 2;worlds = [SimpleNamespace(state=None if partial and w == 3 else 1, refusals=('MAPPER',) if partial and w == 3 else ()) for w in range(4)]
            return (cw.Prepared(r['request_id'], 'detached', SimpleNamespace(fully_constructed=not partial, worlds=worlds), (True,) * 9, .25, 'B'),)
        def build(constructor, prepared, pricer, params, *, binding, recipe, only_complete):
            self.assertEqual((pricer.executor, params, recipe, only_complete), ('LIVE', ('P',), provider.recipe, True))   # root views: the live executor
            self.calls['build_binding'].append(binding)
            roots, requests, pairs = [], [], []
            for n, item in enumerate(prepared):
                rows = (len(requests), len(requests) + 1);requests += [f'{item.request_id}-0', f'{item.request_id}-1']
                pairs += [(fake_world(0), fake_world(1)), (fake_world(2), fake_world(3))]
                roots.append(dict(index=n, request_id=item.request_id, status='REFUSED' if refuse_build and n == 0 else 'PRICED', reason='ROOT_FORWARD_REFUSED',
                    rows=rows, address=(n, 4, 0), root_value=-.5, root_legal=(True,) * 9, b2_view=dict(equal=True)))
            return SimpleNamespace(requests=tuple(requests), worlds=dn().PrebuiltWorlds(cw.SOURCE, binding, tuple(pairs)), roots=tuple(roots))
        lines = []
        with patch.object(gate.g2, 'saved_root_requests', lambda args: loaded), patch.object(gate.g2, 'load_champion', lambda args, l: SimpleNamespace(params='P')), \
                patch.object(gate.g2, 'evaluator_header', lambda args, l, head, champion=None: dict(kind='champion')), \
                patch.object(pc, 'load_inputs', lambda a: ('catalog', 'source', dict(m=1), 'config')), patch.object(pc, 'ProtocolConstructor', Constructor), \
                patch.object(cw, 'binding', bind), patch.object(gate, 'read_roster', lambda path, sha: (dict(declaration=dict(seed=1), capture_prefix=dict(games=64),
                    selected=[]), roster(turns))), patch.object(gate, 'roster_request', request), patch.object(cw, 'prepare_roots', prepare), \
                patch.object(live, 'classify', lambda constructor, r: (['sleep'], 'protocol_facts') if r['i'] == 9 else ([], 'protocol_facts')), \
                patch.object(cw, 'build', build), patch.object(gate, 'bind_constructor', lambda requests, params, plan, ev: (tuple(requests), [.5] * len(requests), True)), \
                patch.object(gate, 'log', lambda m, **k: lines.append(dict(k, message=m))), patch.object(gate, 'tables_bridge', bridge), \
                patch.object(cw, 'live_executor', executor):
            out = gate.constructor_roots(self.ARGS)
        return out, made, lines

    def test_the_selection_pins_and_refusals(self):
        out, made, lines = self.run_roots();ctor = out.constructor
        self.assertEqual(made, [dict(catalog='catalog', source='source', metadata=dict(m=1), owning_config='config', count=4, seed=3, saved_root_admission=True)])
        # band 0 = ordinals 0-3: 1 refused, 2 partial -> 0, 3; band 1 = 4-7: 5 unavailable, 6 refused -> 4, 7; then 8, 9 and 12, 13; no class
        # among them but sleep (9), so the top-up scans for the other three find nothing.
        self.assertEqual([r['ordinal'] for r in ctor.roots], [0, 3, 4, 7, 8, 9, 12, 13]);self.assertEqual(ctor.selection['top_up'], dict(forced_switch=None, fresh_substitute=None, wish=None))
        self.assertEqual([(r['ordinal'], r['kind'].split(':')[0], r['constructed']) for r in ctor.selection['refused']],
            [(1, 'constructor-root', 0), (2, 'WORLD_CONSTRUCTOR_REFUSED', 3), (5, 'CAPTURED_ROOT_UNAVAILABLE', 0), (6, 'constructor-root', 0)])
        self.assertEqual((ctor.selection['attempted'], ctor.selection['classified']), (12, 15))   # the scans classify (5 has no request), prepare none
        self.assertEqual(sum(x['message'] == 'CONSTRUCTOR_ROOT' for x in lines), 12)
        root = ctor.roots[5];self.assertEqual((root['classes'], root['root_class'], root['selected_by'], root['band']), (['sleep'], 'sleep', 'band', 2))
        self.assertEqual(root['capture']['capture_chain_sha256'], '9' * 64);self.assertEqual(root['rows'], (10, 11))
        self.assertEqual(len({r['carry_signature'] for r in ctor.roots}), 1);self.assertEqual(len(ctor.requests), 16)
        pins = ctor.pins;from pokezero.live_protocol_actor import ActorContract
        self.assertEqual((pins['roster']['sha256'], pins['showdown']['revision'], pins['metadata']['sha256'], pins['vocabulary']['sha256'], pins['seed'], pins['worlds']),
            ('a' * 64, ActorContract.champion().showdown_revision, 'b' * 64, 'c' * 64, 3, 4))
        self.assertEqual(pins['metadata']['canonical_sha256'], hashlib.sha256(b'{"m":1}').hexdigest());self.assertEqual(pins['binding']['source_hash'], 's')
        self.assertEqual([p['ordinal'] for p in pins['roots']], [0, 3, 4, 7, 8, 9, 12, 13]);self.assertEqual(pins['b']['status'], 'NOT_APPLICABLE')
        self.assertEqual((out.params, out.actor_capacity, out.groups, out.evaluator['dropped_roots']), (('P',), 1, None, []))
        json.dumps(dict(pins=pins, section=gate.constructor_section(ctor, gate.constructor_calls(ctor, 512))), allow_nan=False, default=str)
        # TABLES_BRIDGE: the bank's metadata and catalog against the constructor's, the bank executor's own tables as the reference; the live
        # executor from the bridge's tables, and the constructor's own binding
        self.assertEqual(self.calls['bridge'], [(dict(m=1, move_frequency=2), 'CAT', dict(m=1), 'catalog', ('BANK-E', 'BANK-O'))])
        self.assertEqual(self.calls['executor'], [(dict(m=1), 'catalog', ('E', 'O'))]);self.assertEqual(ctor.executor, 'LIVE')
        self.assertEqual(pins['tables'], dict(digest='9' * 64, equal=True, bank_binding_differs=['metadata_sha256']));self.assertEqual(pins['executor'], gate.LIVE_EXECUTOR)
        # the live binding is built from the constructor's own inputs (the bank's only for the comparison) and is the one build and the run use
        self.assertEqual(sorted(self.calls['binding'], key=repr), [('SRC', dict(m=1, move_frequency=2), 'CAT'), ('source', dict(m=1), 'catalog')])   # once each
        self.assertEqual(self.calls['build_binding'][0].metadata_sha256, 'm');self.assertIs(ctor.binding, self.calls['build_binding'][0])
        self.assertEqual(pins['binding']['metadata_sha256'], 'm')
        self.assertEqual(next(x for x in lines if x['message'] == 'TABLES_BRIDGE')['digest'], '9' * 64)

    def test_a_tables_difference_fails_before_any_root_is_prepared(self):
        with self.assertRaisesRegex(SystemExit, r"fail closed\): .*tables\['engine'\].power"):
            self.run_roots(bridge_error="the bank's and the constructor's compiled tables differ (TABLES_BRIDGE): tables['engine'].power: bytes differ")
        self.assertEqual((len(self.calls['bridge']), self.calls['prepare'], self.calls['executor']), (1, [], []))

    def test_the_source_binding_stays_gated(self):
        """Only metadata_sha256 may differ between the bank's binding and the live one (TABLES_BRIDGE stands in for it); any other field fails
        closed before any root is prepared."""
        for differs in (dict(metadata_sha256='bank-m', source_binding_sha256='x'), dict(source_hash='x'), dict(encoder=(('checkpoint_sha256', 'f' * 64),))):
            with self.subTest(differs=sorted(differs)), self.assertRaisesRegex(SystemExit, r'binding differs from the live one in .*' + sorted(set(differs) - {'metadata_sha256'})[0]):
                self.run_roots(bank_binding=differs)
            self.assertEqual(self.calls['prepare'], [])
        out, _, _ = self.run_roots(bank_binding={});self.assertEqual(out.constructor.pins['tables']['bank_binding_differs'], [])   # equal bindings pass

    def test_an_unknown_table_member_fails_closed_with_a_message(self):
        with self.assertRaisesRegex(SystemExit, r'fail closed\): tables\[.engine.\]\.public_port: unsupported compiled-table member'):
            self.run_roots(bridge_error=TypeError("tables['engine'].public_port: unsupported compiled-table member builtins.object"))
        self.assertEqual(self.calls['prepare'], [])

    def test_the_live_executor_is_the_live_servers(self):
        """constructor_worlds.live_executor: device_public_teacher.make_executor on the given (or compiled) tables with the device teacher and
        batched joint steps, as scripts/depthn_live_server.build has always built it."""
        from gpu_engine import constructor_worlds as cw, device_public_teacher as teacher
        made = []
        with patch.object(teacher, 'make_executor', lambda *a, **k: made.append((a, k)) or 'executor'), patch.object(cw, 'engine_tables', lambda m, c: (('E', m), ('O', c))):
            self.assertEqual((cw.live_executor('meta', 'cat', tables=('e', 'o')), cw.live_executor('meta', 'cat')), ('executor', 'executor'))
        flags = dict(enable_device_teacher=True, batch_joint=True)
        self.assertEqual(made, [(('e', 'o'), flags), ((('E', 'meta'), ('O', 'cat')), flags)])

    def test_fail_closed(self):
        with self.assertRaisesRegex(SystemExit, 'build refused selected roots'):self.run_roots(refuse_build=True)
        with self.assertRaisesRegex(SystemExit, 'turn band 1'):self.run_roots(refused=(1, 6, 7))


if __name__ == '__main__':unittest.main()
