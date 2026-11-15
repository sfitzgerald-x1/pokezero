"""depthn_r3pa: R3pa's pure core against its references (search_hedge.averaging / noise_gated, mc_ladder_backline.prior_average,
the offline gate's direct prior override, the live depthn_live_search.decide / noise_gate on substituted blocks) on hand-built and
seeded random WorldPrices, and the parity tool (scripts/depthn_r3pa_parity.py) on hand-built re-pricer rows. CPU only; "bit for
bit" compares float.hex."""
import collections
from contextlib import redirect_stdout
import dataclasses
import io
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from _subproc_env import subproc_env
from gpu_engine import depthn_r3pa as R
from gpu_engine import mc_ladder_backline as B
from gpu_engine import search_hedge as hedge
from pokezero import v5_gumbel as original
from scripts import depthn_r3pa_parity as P
from tests.test_gpu_engine_mc_ladder_backline import BLISSEY, CHANSEY, LANE0, LANE1, SKARM, SWITCHY, make

ROOT = Path(__file__).resolve().parents[1]
UNSEEN = R.UNSEEN
MOVES = tuple(f'move:m{i}' for i in range(7))
SEEN = ('switch:skarmory', 'switch:blissey', 'switch:snorlax')
TWO = (0, 1)   # the hand-built roots' legal own actions
TIE = 1e-9     # a top-two q gap or gate margin this small may resolve either way between substitution and a direct override


def mask(*xs):return tuple(x in xs for x in range(9))


def bits(x):
    """Floats as float.hex, recursively (None, ints and strings kept): equal bits, not just equal values."""
    if x is None or type(x) in (int, bool, str):return x
    if isinstance(x, dict):return {k: bits(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):return [bits(v) for v in x]
    return float(x).hex()


def world(values, kept=(), logits=None):
    """WorldPrices: values {a: {b: G[a][b]}}; kept, the opponent's kept lanes (empty: waiting, column 0 only); logits {lane: x}."""
    jv = [[None] * 9 for _ in range(9)]
    for a, row in values.items():
        for b, v in row.items():jv[a][b] = v
    return original.WorldPrices(bool(kept), mask(*kept), tuple((logits or {}).get(b, 0.) for b in range(9)), jv)


def hand(kept, logits=None, actions=TWO):
    """A requested world on `actions` with fixed values sin(3a + b + 1) * .9 against each kept lane."""
    return world({a: {b: math.sin(3 * a + b + 1) * .9 for b in kept} for a in actions}, kept, logits)


def waiting(actions=TWO, value=.1):
    return world({a: {0: value * (a + 1)} for a in actions})


def lab(**lanes):
    """Nine labels from lane=label keywords (l0..l8), None elsewhere."""
    return [lanes.get(f'l{b}') for b in range(9)]


def pairs(prices):
    """depthn_live_search's blocks: the worlds in consecutive pairs."""
    return [tuple(prices[i:i + 2]) for i in range(0, len(prices), 2)]


def random_root(rng, count=None, waiting_share=.15, zero_unseen=.2):
    """(prices, root_legal, root_logits, labels): 2..16 worlds; a requested world has its own kept lanes, labels (four distinct
    moves; switch lanes seen species or unseen; an unkept lane's label sometimes None) and logits (its illegal lanes' logits
    random too; its kept unseen-switch lanes at -800 in some worlds: zero own mass there). Values share a per-action base in half
    the roots, so the gate passes overrides too."""
    count = count or rng.randint(2, 16);actions = sorted(rng.sample(range(9), rng.randint(2, 6)))
    noise = rng.choice((.3, 1.));base = {a: rng.uniform(-.6, .6) if noise < 1 else 0. for a in actions}
    value = lambda a: max(-1., min(1., base[a] + rng.uniform(-noise, noise)))
    prices, labels = [], []
    for _ in range(count):
        if rng.random() < waiting_share:
            prices.append(world({a: {0: value(a)} for a in actions}));labels.append([None] * 9);continue
        names = list(rng.sample(MOVES, 4)) + [UNSEEN] * 5
        for b, s in zip(rng.sample(range(4, 9), rng.randint(0, 3)), SEEN):names[b] = s
        kept = sorted(rng.sample(range(9), rng.randint(1, 9)))
        logits = {b: rng.gauss(0., 1.5) for b in range(9)}
        if rng.random() < zero_unseen and any(names[b] != UNSEEN for b in kept):
            for b in kept:
                if names[b] == UNSEEN:logits[b] = -800.
        prices.append(world({a: {b: value(a) for b in kept} for a in actions}, kept, logits))
        labels.append([x if b in kept or rng.random() < .5 else None for b, x in enumerate(names)])
    return prices, mask(*actions), [rng.gauss(0., 1.) for _ in range(9)], labels


def todays_gate(prices, legal, logits):
    """(averaging's Choice, its action, the played action) by search_hedge's own functions: the live NOISE_GATED_K2 rule."""
    choice = hedge.averaging(prices, legal);candidate = hedge.top_action(choice.strategy)
    played = hedge.noise_gated(hedge.per_world_q(hedge.grids(prices, legal)), candidate, hedge.logits_greedy(logits, legal), 2)
    return choice, candidate, played


def offline_gate(prices, legal, logits, labels):
    """R3pa / R3pag as the offline gate computed them (a direct override): prior_average's priors put into search_hedge's grid prior
    arrays (a waiting world's kept), averaging's summation, noise_gated on per_world_q. dict(q, candidate, played, q_gap, margin)."""
    mapped = B.prior_average(prices, labels, [1. / len(prices)] * len(prices))
    gs = [g if not p.opponent_requested else dataclasses.replace(g, prior=np.asarray([m.get(b, 0.) for b in g.replies], np.float64))
          for p, g, m in zip(prices, hedge.grids(prices, legal), mapped)]
    count = len(gs);actions = gs[0].actions
    q = [math.fsum(g.prior[k] * g.values[i, k] / count for g in gs for k in range(len(g.replies))) for i in range(len(actions))]
    candidate = actions[max(range(len(actions)), key=lambda i: (q[i], -i))];greedy = hedge.logits_greedy(logits, legal);qw = hedge.per_world_q(gs)
    full = [None] * 9
    for a, v in zip(actions, q):full[a] = v
    return dict(q=full, candidate=candidate, played=hedge.noise_gated(qw, candidate, greedy, 2), q_gap=P.top_two_gap(full),
                margin=P.margin(qw, candidate, greedy, 2))


class OffModeTests(unittest.TestCase):
    def test_off_is_todays_averaging_and_gate_bit_for_bit(self):
        rng = random.Random(1)
        for _ in range(300):
            prices, legal, logits, labels = random_root(rng)
            choice, candidate, played = todays_gate(prices, legal, logits)
            for labs in (labels, None):   # off never reads the labels
                got = R.choose(prices, legal, logits, labs, 'off')
                self.assertEqual((got.action, got.candidate, got.priors, got.fallback_worlds), (played, candidate, None, 0))
                self.assertEqual(bits(got.q), bits(choice.q));self.assertEqual(got.gate.played, played)


class LiveOperatorTests(unittest.TestCase):
    """The live functions, unchanged, on the original blocks (off) and on substitute_blocks' (gated, replace_gate)."""
    def test_choose_is_the_live_operator_on_substituted_blocks(self):
        from gpu_engine import depthn_live_search as live   # imports JAX: only here, never in depthn_r3pa
        rng = random.Random(2);seen = collections.Counter()
        for _ in range(40):
            prices, legal, logits, labels = random_root(rng, count=2 * rng.randint(1, 8))
            for mode, blocks in (('off', pairs(prices)), ('gated', R.substitute_blocks(pairs(prices), R.root_priors(prices, labels)))):
                decision = live.decide(blocks, logits, legal);self.assertEqual(decision['status'], 'COMPLETE')
                gated = live.noise_gate(decision, blocks, logits, legal);g = gated['noise_gate']
                got = R.choose(prices, legal, logits, labels, mode)
                self.assertEqual(got.action, gated['action_index']);self.assertEqual(bits(list(got.q)), bits(gated['q_values']))
                self.assertEqual((got.candidate, got.greedy, got.gate.passed), (g['candidate'], g['greedy'], g['passed']))
                self.assertEqual(bits([got.gate.mean_gap, got.gate.stderr]), bits([g['mean_gap'], g['stderr']]))
                if mode == 'gated':
                    ungated = R.choose(prices, legal, logits, labels, 'replace_gate')
                    self.assertEqual((ungated.action, bits(ungated.q)), (decision['action_index'], bits(decision['q_values'])))
                seen[mode, 'fell_back' if not g['passed'] else 'override' if g['candidate'] != g['greedy'] else 'greedy'] += 1
        self.assertTrue(all(seen[m, k] for m in ('off', 'gated') for k in ('fell_back', 'override', 'greedy')), seen)


class SubstituteTests(unittest.TestCase):
    def test_substitute_reproduces_the_root_priors(self):
        rng = random.Random(3);worst = 0.
        for _ in range(400):
            prices, _, _, labels = random_root(rng);priors = R.root_priors(prices, labels);out = R.substitute(prices, priors)
            self.assertEqual(len(out), len(prices))
            for p, pr, s in zip(prices, priors, out):
                if pr is None:self.assertIs(s, p);continue
                self.assertEqual((s.opponent_requested, s.opponent_legal, s.joint_values), (p.opponent_requested, p.opponent_legal, p.joint_values))
                self.assertTrue(all(math.isfinite(x) for x in s.opponent_logits) and len(s.opponent_logits) == 9)
                self.assertEqual(bits([x for x, k in zip(s.opponent_logits, p.opponent_legal) if not k]),
                                 bits([x for x, k in zip(p.opponent_logits, p.opponent_legal) if not k]))   # illegal lanes untouched
                probs = original._probabilities(s.opponent_logits, s.opponent_legal)
                for b in range(9):
                    want = pr.get(b, 0.) if p.opponent_legal[b] else 0.
                    if want == 0.:self.assertEqual(probs[b], 0.)
                    else:self.assertLessEqual(abs(probs[b] - want), 1e-12);worst = max(worst, abs(probs[b] - want))
        self.assertLess(worst, 1e-12)

    def test_a_zero_mass_lane_gets_exactly_zero(self):
        p = hand((0, 1, 2), {0: .5, 5: 3.25, 8: -7.})   # lanes 5 and 8 illegal, with their own logits
        for prior in ({0: .7, 1: .3}, {0: .6, 1: 0., 2: .4}, {0: 1.}, {0: 1e-300, 1: 1.}, {0: 5e-324}, {2: 5e-324, 1: 5e-324}):
            s = R.substitute([p], [prior])[0];probs = original._probabilities(s.opponent_logits, s.opponent_legal)
            self.assertEqual((s.opponent_logits[5], s.opponent_logits[8]), (3.25, -7.))
            for b in (0, 1, 2):
                if prior.get(b, 0.) > 0:self.assertGreater(probs[b], 0.);self.assertEqual(s.opponent_logits[b], math.log(prior[b]))
                else:self.assertEqual((s.opponent_logits[b], probs[b]), (R.ZERO_MASS_LOGIT, 0.))
            self.assertAlmostEqual(math.fsum(probs), 1., places=15)

    def test_bad_priors_raise(self):
        p = hand((0, 1));legal = mask(*TWO)
        for prior, why in (({0: .5, 2: .5}, 'not legal'), ({0: -.1, 1: 1.1}, 'nonnegative'), ({0: math.nan}, 'nonnegative'),
                           ({0: 0.}, 'positive mass'), ({9: 1.}, 'lanes 0..8'), ({True: 1.}, 'lanes 0..8')):
            with self.assertRaisesRegex(ValueError, why):R.substitute([p], [prior])
        with self.assertRaisesRegex(ValueError, 'waiting world'):R.substitute([p, waiting()], [None, {0: 1.}])
        with self.assertRaisesRegex(ValueError, 'one prior'):R.substitute([p, p], [None])
        self.assertIsNotNone(hedge.averaging(R.substitute([p, waiting()], [{0: .2, 1: .8}, None]), legal))

    def test_substitute_blocks_is_substitute_regrouped(self):
        rng = random.Random(4)
        for _ in range(100):
            prices, _, _, labels = random_root(rng, count=2 * rng.randint(1, 8));priors = R.root_priors(prices, labels)
            blocks = pairs(prices);got = R.substitute_blocks(blocks, priors);flat = R.substitute([w for pair in blocks for w in pair], priors)
            self.assertIs(type(got), list);self.assertEqual([len(pair) for pair in got], [2] * len(blocks))
            self.assertTrue(all(type(pair) is tuple for pair in got))
            self.assertEqual(got, pairs(flat));self.assertEqual([w for pair in got for w in pair], flat)
            self.assertEqual([bits(w.opponent_logits) for pair in got for w in pair], [bits(w.opponent_logits) for w in flat])
        with self.assertRaisesRegex(ValueError, 'one prior'):R.substitute_blocks(pairs(prices), priors[:-1])

    def test_own_priors_substituted_keep_averaging_to_rounding(self):
        rng = random.Random(5)
        for _ in range(200):
            prices, legal, _, _ = random_root(rng);own = [R.own_prior(p) if p.opponent_requested else None for p in prices]
            ref = hedge.averaging(prices, legal);got = hedge.averaging(R.substitute(prices, own), legal)
            for a in range(9):
                if ref.q[a] is None:self.assertIsNone(got.q[a])
                else:self.assertLessEqual(abs(got.q[a] - ref.q[a]), 1e-12)


class RootPriorTests(unittest.TestCase):
    def same(self, prices, labels, weights=None):
        """root_priors == mc_ladder_backline.prior_average bit for bit (and the fallback count); each mapped prior covers exactly
        its world's kept lanes and sums to 1."""
        mine, theirs = collections.Counter(), collections.Counter()
        got = R.root_priors(prices, labels, weights=weights, counts=mine)
        want = B.prior_average(prices, labels, weights or [1. / len(prices)] * len(prices), theirs)
        self.assertEqual(bits(got), bits(want));self.assertEqual(mine, theirs)
        for m, p in zip(got, prices):
            if m is None:self.assertFalse(p.opponent_requested);continue
            self.assertEqual(sorted(m), R.kept(p));self.assertAlmostEqual(math.fsum(m.values()), 1., places=12)
        return got, mine

    def test_moves_only(self):
        prices = [hand((0, 1, 2), {0: 1., 2: -1.}), hand((0, 1, 3), {1: .5, 3: 2.}), hand((0, 1, 2, 3), {0: -.3, 3: .7})]
        labels = [lab(l0='move:a', l1='move:b', l2='move:c'), lab(l0='move:b', l1='move:a', l3='move:d'),
                  lab(l0='move:a', l1='move:b', l2='move:c', l3='move:d')]
        self.same(prices, labels)
        twins = [hand((0, 1), {0: 1.}), hand((0, 1), {1: 2.})];labels = [lab(l0='move:a', l1='move:b')] * 2
        got, _ = self.same(twins, labels);own = [R.own_prior(p) for p in twins]   # one label set: pi_bar is the mean prior
        for m in got:
            for b in (0, 1):self.assertAlmostEqual(m[b], (own[0][b] + own[1][b]) / 2, places=15)

    def test_seen_switches_map_by_species_not_lane(self):
        prices = [hand((0, 4, 5), {4: 1.}), hand((0, 4, 5), {5: -1.})]
        labels = [lab(l0='move:a', l4='switch:skarmory', l5='switch:blissey'), lab(l0='move:a', l4='switch:blissey', l5='switch:skarmory')]
        got, _ = self.same(prices, labels)
        self.assertAlmostEqual(got[0][4], got[1][5], places=15);self.assertAlmostEqual(got[0][5], got[1][4], places=15)

    def test_unseen_switches_split_by_own_prior_or_equally_at_zero_own_mass(self):
        p0 = hand((0, 4, 5), {4: -800., 5: -800.})   # zero own mass on its two unseen switches
        p1 = hand((0, 4, 6), {4: 1., 6: -.5})        # unequal own mass on its two
        self.assertEqual((R.own_prior(p0)[4], R.own_prior(p0)[5]), (0., 0.))
        labels = [lab(l0='move:a', l4=UNSEEN, l5=UNSEEN), lab(l0='move:a', l4=UNSEEN, l6=UNSEEN)]
        got, _ = self.same([p0, p1], labels)
        self.assertGreater(got[0][4], 0.);self.assertEqual(got[0][4], got[0][5])   # equal split
        own = R.own_prior(p1);self.assertAlmostEqual(got[1][4] / got[1][6], own[4] / own[6], places=12)   # proportional split

    def test_a_label_the_world_lacks_drops(self):
        prices = [hand((0, 1), {1: 1.}), hand((0, 1), {0: .4})]
        labels = [lab(l0='move:a', l1='move:surf'), lab(l0='move:a', l1='move:b')]
        got, _ = self.same(prices, labels)
        mass = collections.Counter()
        for p, l in zip(prices, labels):
            for b, v in R.own_prior(p).items():mass[l[b]] += v / 2
        self.assertAlmostEqual(got[1][0] / got[1][1], mass['move:a'] / mass['move:b'], places=12)   # surf's mass is gone
        self.assertAlmostEqual(got[0][0] / got[0][1], mass['move:a'] / mass['move:surf'], places=12)   # world 0 lacks b

    def test_waiting_worlds_are_untouched_and_still_count_in_w(self):
        prices = [hand((0, 1), {0: 1.}), waiting(), hand((0, 2), {2: -1.})]
        labels = [lab(l0='move:a', l1='move:b'), [None] * 9, lab(l0='move:a', l2='move:c')]
        got, _ = self.same(prices, labels);self.assertIsNone(got[1])
        self.assertIsNone(R.root_priors(prices, [labels[0], None, labels[2]])[1])   # a waiting world's labels may be None
        self.assertIs(R.substitute(prices, got)[1], prices[1])

    def test_nothing_lands_keeps_the_own_prior(self):
        # Under uniform weights a world's own mass always lands on itself; a zero weight on world 1 with labels no other world has
        # leaves nothing to land there.
        prices = [hand((0, 1), {0: 1.}), hand((0, 1), {1: .3})]
        labels = [lab(l0='move:a', l1='move:b'), lab(l0='move:c', l1='move:d')]
        got, counts = self.same(prices, labels, weights=[1., 0.])
        self.assertEqual(counts[R.FALLBACK], 1);self.assertEqual(bits(got[1]), bits(R.own_prior(prices[1])))
        _, counts = self.same(prices, labels);self.assertEqual(counts[R.FALLBACK], 0)

    def test_random_roots(self):
        rng = random.Random(6);fallbacks = 0
        for _ in range(400):
            prices, _, _, labels = random_root(rng);fallbacks += self.same(prices, labels)[1][R.FALLBACK]
        self.assertEqual(fallbacks, 0)

    def test_an_unlabelled_kept_lane_raises(self):
        prices = [hand((0, 1)), hand((0, 1))];legal = mask(*TWO);logits = [1.] + [0.] * 8
        labels = [lab(l0='move:a', l1='switch:blissey'), lab(l0='move:a')]   # world 1's kept lane 1
        with self.assertRaises(R.LaneUnlabelled) as raised:R.root_priors(prices, labels)
        self.assertIsInstance(raised.exception, ValueError);self.assertIn('world 1', str(raised.exception))
        with self.assertRaises(B.rr.Unscorable) as theirs:B.check_labels([{}, {}], prices, labels)   # the reference refuses too
        self.assertEqual(theirs.exception.reason, R.LaneUnlabelled.reason)
        for mode in ('gated', 'replace_gate'):
            with self.assertRaises(R.LaneUnlabelled):R.choose(prices, legal, logits, labels, mode)
        self.assertIsNotNone(R.choose(prices, legal, logits, labels, 'off'))
        fine = [lab(l0='move:a', l1='move:b', l7=None), lab(l0='move:a', l1='move:b')]   # unkept lanes need no label
        self.assertEqual(len(R.root_priors(prices, fine)), 2)
        with self.assertRaisesRegex(ValueError, 'nine lane labels'):R.root_priors(prices, [['move:a'], fine[1]])
        with self.assertRaisesRegex(ValueError, 'one label list'):R.root_priors(prices, fine[:1])


class ModeTests(unittest.TestCase):
    def test_gated_and_replace_gate_against_the_offline_gate(self):
        rng = random.Random(7);seen = collections.Counter()
        for _ in range(400):
            prices, legal, logits, labels = random_root(rng)
            gated = R.choose(prices, legal, logits, labels, 'gated');ungated = R.choose(prices, legal, logits, labels, 'replace_gate')
            sub = R.substitute(prices, R.root_priors(prices, labels));off = R.choose(sub, legal, logits, None, 'off')
            self.assertEqual((gated.action, gated.candidate, gated.gate, bits(gated.q)), (off.action, off.candidate, off.gate, bits(off.q)))
            ref = offline_gate(prices, legal, logits, labels)   # the direct override: equal up to rounding, so up to exact near-ties
            self.assertTrue(all(abs(x - y) <= 1e-12 for x, y in zip(ungated.q, ref['q']) if x is not None))
            if ungated.action != ref['candidate']:self.assertLess(ref['q_gap'], TIE);seen['near_tie_candidate'] += 1
            if gated.action != ref['played']:
                self.assertTrue(ref['q_gap'] < TIE or ref['margin'] is not None and abs(ref['margin']) < TIE);seen['near_tie_played'] += 1
            self.assertEqual((gated.candidate, ungated.candidate, ungated.gate), (ungated.action, ungated.action, None))
            self.assertEqual(bits([gated.q, gated.priors]), bits([ungated.q, ungated.priors]))
            self.assertEqual(gated.action != ungated.action, not gated.gate.passed)   # they differ exactly when the gate falls back
            if not gated.gate.passed:self.assertEqual(gated.action, gated.greedy)
            seen['fell_back' if not gated.gate.passed else 'override' if gated.candidate != gated.greedy else 'greedy'] += 1
            seen['changed_by_r3pa'] += ungated.action != R.choose(prices, legal, logits, labels, 'off').candidate
        self.assertTrue(all(seen[k] for k in ('fell_back', 'override', 'greedy', 'changed_by_r3pa')), seen)

    def test_refusals(self):
        legal = mask(*TWO);logits = [0., 1.] + [0.] * 7;labels = [lab(l0='move:a', l1='move:b')] * 2
        one = [hand((0, 1))]
        for mode in ('off', 'gated'):
            with self.assertRaisesRegex(ValueError, 'two or more'):R.choose(one, legal, logits, labels[:1], mode)
        self.assertEqual(R.choose(one, legal, logits, labels[:1], 'replace_gate').gate, None)
        unknown = world({0: {0: 0., 1: 0.}, 1: {0: .2, 1: None}}, (0, 1))   # action 1 against reply 1: unpriced
        for mode in R.MODES:self.assertIsNone(R.choose([unknown, hand((0, 1))], legal, logits, labels, mode))
        with self.assertRaisesRegex(ValueError, 'mode'):R.choose(one * 2, legal, logits, labels, 'R3pa')
        with self.assertRaisesRegex(ValueError, 'one label list'):R.choose(one * 2, legal, logits, None, 'gated')

    def test_the_module_adds_no_jax_and_no_live_search(self):
        # `import pokezero` (through search_hedge) loads torch by itself; depthn_r3pa must add nothing beyond its own module.
        code = ("import json, sys; import gpu_engine.search_hedge; before = set(sys.modules); import gpu_engine.depthn_r3pa; "
                "print(json.dumps([sorted(set(sys.modules) - before), [m for m in ('jax', 'jaxlib', 'gpu_engine.depthn_live_search') if m in sys.modules]]))")
        out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=subproc_env(), capture_output=True, text=True, timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr);self.assertEqual(json.loads(out.stdout), [['gpu_engine.depthn_r3pa'], []])


class ParityToolTests(unittest.TestCase):
    """The parity tool on mc_ladder_backline's hand-built Foul Play rows, held to the backline's own R3pa rung (depth 1, ungated)."""
    @staticmethod
    def rows():
        lanes = [['move:tackle', 'switch:blissey'], ['move:tackle', 'switch:chansey']];parties = [[SKARM, BLISSEY], [SKARM, CHANSEY]]
        return [make(SWITCHY, lanes, parties, logits=[LANE1, LANE0]),   # pi_bar .5 / .5: q(1) = 0 exactly, a tie to action 0
                make([{0: (0., 0.), 1: (.3, -.1)}] * 2, lanes, parties, logits=[LANE1, LANE0], ordinal=8),   # q(1) = .1 > 0
                make(SWITCHY, [lanes[0], ['move:tackle']], parties, logits=[LANE1, LANE0], ordinal=9)]   # world 1 lane 1 unlabelled

    def run_tool(self, rows, expect, *extra):
        with tempfile.TemporaryDirectory() as tmp:
            rows_path, expect_path = Path(tmp) / 'rows.jsonl', Path(tmp) / 'rung.jsonl'
            rows_path.write_text(''.join(json.dumps(r) + '\n' for r in rows));expect_path.write_text(''.join(json.dumps(r) + '\n' for r in expect))
            out = io.StringIO()
            with redirect_stdout(out):
                code = P.main(['--kind', 'foulplay', '--mode', 'replace_gate', '--expect', str(expect_path), '--rows', str(rows_path), *extra])
        return code, json.loads(out.getvalue())

    def test_the_tool_reproduces_the_backline_r3pa_rung_and_flags_differences(self):
        rows = self.rows();expect = B.build(rows, 'foulplay')[0]['R3pa']
        self.assertEqual([r['action'] for r in expect][:2], [0, 1]);self.assertEqual(expect[2]['reason'], 'lane_unlabelled')
        for extra in ((), ('--depth2',)):   # these rows' top-level worlds are their depth-1 worlds
            code, report = self.run_tool(rows, expect, *extra)
            self.assertEqual(code, 0, report)
            self.assertEqual([report[k] for k in ('roots', 'compared', 'agree', 'disagreements', 'decided_mismatch', 'both_undecided')], [3, 2, 2, 0, 0, 1])
            self.assertEqual(report['undecided'], {'lane_unlabelled': 1});self.assertEqual(report['direct']['agree_with_expected'], 2)
        code, report = self.run_tool(rows, [dict(expect[0], action=1)] + expect[1:]);self.assertEqual((code, report['disagreements']), (1, 1))
        detail = report['disagreement_details'][0]
        self.assertEqual((detail['root'], detail['mine'], detail['expected'], detail['direct']), (['foulplay', 7, 3], 0, 1, 0))
        self.assertEqual((detail['q_gap_substituted'], detail['q_gap_direct'], detail['gate_margin_direct']), (0., 0., None))   # the exact tie
        code, report = self.run_tool(rows, expect[:2] + [dict(expect[2], action=0)]);self.assertEqual((code, report['decided_mismatch']), (1, 1))
        code, report = self.run_tool(rows, expect[:2]);self.assertEqual((code, report['missing_expected']), (1, 1))
        with self.assertRaises(SystemExit):self.run_tool(rows, expect, '--manifest', 'm.jsonl')


if __name__ == '__main__':
    unittest.main()
