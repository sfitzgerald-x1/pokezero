"""mc_ladder_worlds: world-rule rungs and P(true set) diagnostics from hand-built re-pricer rows (two legal actions, two kept
replies, three or four worlds), reusing mc_ladder_rungs' test rows. Pure Python: no engine, model or device."""
import io
import json
import math
import statistics
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from gpu_engine import mc_ladder_rungs as L
from gpu_engine import mc_ladder_worlds as M
from tests.test_gpu_engine_mc_ladder_rungs import D1, REPLY, TRUE, hand_q, row
from tests.test_gpu_engine_mc_ladder_score import bank_row

# A true world 0 that favours action 1 strongly against either reply; sampled worlds D1 (reply model: gaps .15, .05, .25; the
# champion's 1/2-1/2 prior: -.1, -.1, .1).
T0 = {0: (0., 0.), 1: (.9, .9)}
LOW = {0: (0., 0.), 1: (-.9, -.9)}
SEEDS = [101, 102, 103, 104]


def member(species, moves, ability='pressure', item='leftovers', slot=0):
    return dict(slot=slot, species=species, moves=moves, base_ability=ability, ability=ability, item=item)


TRUE_PARTY = [member('skarmory', ['spikes', 'toxic', 'drillpeck', 'rest']), member('snorlax', ['bodyslam', 'curse', 'rest', 'sleeptalk'],
                                                                                   'immunity', 'leftovers', 1)]
OTHER_SKARM = member('skarmory', ['spikes', 'toxic', 'drillpeck', 'whirlwind'])
OTHER_LAX = member('snorlax', ['bodyslam', 'curse', 'selfdestruct', 'earthquake'], 'immunity', 'leftovers', 1)


def dress(raw, parties, active='skarmory'):
    """Kernel seeds and opponent parties on a row's worlds."""
    for w, (world, party) in enumerate(zip(raw['worlds'], parties)):
        world['kernel_seed'] = SEEDS[w];world['opponent'] = dict(active_species=active, party=party)
    return raw


def sampled_row(depth1=D1, reply=True, parties=None, **extra):
    parties = parties or [[OTHER_SKARM, TRUE_PARTY[1]], TRUE_PARTY, [TRUE_PARTY[0], OTHER_LAX], TRUE_PARTY][:len(depth1)]
    return dress(row(depth1=depth1, depth2=depth1, reply=reply, **extra), parties)


def true_row(world0=T0, reply=True, count=3, **extra):
    raw = row(depth1=[world0] + D1[1:count], reply=reply, true_worlds=dict(TRUE, worlds=count, built=count), **extra)
    return dress(raw, [TRUE_PARTY] * count)


def weights(table):
    """A load_world_weights table {(ordinal, decision_index): {world: loglik}} for root (7, 3)."""
    return {(7, 3): dict(enumerate(table))}


def by_rung(out):
    return {name: rows[0]['action'] for name, rows in out.items()}


def reasons(out):
    return {name: rows[0].get('reason') for name, rows in out.items()}


class WorldRungTests(unittest.TestCase):
    def test_r3_is_mc_ladder_rungs_r3_and_the_true_world_rungs_follow_a_hand_computation(self):
        out, _, summary = M.build([sampled_row()], 'foulplay', [true_row()])
        got = by_rung(out);self.assertEqual(got['R3'], L.build([sampled_row()], 'C')[0]['R3'][0]['action'])
        reply = hand_q(D1, (.75, .25));true = hand_q([T0], (.75, .25))[0]
        self.assertEqual(got['R3'], 1 if statistics.fmean(reply) > 0 else 0)
        self.assertEqual(got['R3true'], 1 if statistics.fmean([true] + reply[1:]) > 0 else 0)
        self.assertEqual(got['R3true_ins'], 1 if statistics.fmean(reply + [true]) > 0 else 0)
        low, _, _ = M.build([sampled_row()], 'foulplay', [true_row(LOW)])   # the truth flips the sampled worlds' choice
        self.assertEqual((by_rung(low)['R3'], by_rung(low)['R3true']), (1, 0))
        self.assertEqual(summary['rungs']['R3true']['decided'], 1)
        self.assertEqual(got['R1'], L.build([true_row()], 'A')[0]['R1'][0]['action'])   # mc_ladder_rungs' R1 on the true row
        self.assertEqual(got['R1'], 1 if statistics.fmean([true] + reply[1:]) > 0 else 0)
        for r in out.values():self.assertEqual((r[0]['kind'], r[0]['ordinal'], r[0]['target'], r[0]['greedy_action']), ('foulplay', 7, 3, 0))

    def test_self_roots_use_the_champion_prior_and_the_manifest(self):
        manifest = {7: dict(ordinal=7, seed=11, ours='p1', target=3, status='ok')}
        out, _, _ = M.build([sampled_row(reply=False)], 'self', [true_row(reply=False)], manifest=manifest)
        champion = hand_q(D1, (.5, .5))   # -.1, -.1, .1: averaging keeps greedy 0
        self.assertEqual(by_rung(out)['R3'], L.build([sampled_row(reply=False)], 'E', manifest)[0]['R3'][0]['action'])
        self.assertEqual(by_rung(out)['R3'], 0 if statistics.fmean(champion) <= 0 else 1)
        self.assertEqual(by_rung(out)['R3true'], 1)   # T0 at (.9, .9) outweighs -.1 and .1
        self.assertEqual(out['R3'][0]['seed'], 11);self.assertEqual(out['R3'][0]['ours'], 'p1')

    def test_reweighting_follows_the_likelihood_and_beta_zero_is_averaging(self):
        heavy = weights([0., -50., 0.])   # world 1 (T0) ruled out: two LOW worlds remain
        out, _, _ = M.build([sampled_row(depth1=[LOW, T0, LOW])], 'foulplay', weights=heavy)
        self.assertEqual(by_rung(out)['R3'], 0)   # (-.9 + .9 - .9) / 3
        self.assertEqual((by_rung(out)['R3rw_b0.5'], by_rung(out)['R3rw_b1']), (0, 0))
        favour = weights([-50., 0., -50.])   # only world 1 counts: action 1
        out, _, _ = M.build([sampled_row(depth1=[LOW, T0, LOW])], 'foulplay', weights=favour)
        self.assertEqual((by_rung(out)['R3'], by_rung(out)['R3rw_b0.5'], by_rung(out)['R3rw_b1']), (0, 1, 1))
        prices = M.prices_of(sampled_row()['worlds'] + [sampled_row()['worlds'][0]], True)   # four worlds: 1/4 is a power of two
        legal = tuple(a in (0, 1) for a in range(9))
        q = M.rr.weighted_q(M.hedge.grids(prices, legal), M.rr.world_weights([0.] * 4, 1.))
        self.assertEqual([q[a] for a in (0, 1)], list(M.hedge.averaging(prices, legal).q[:2]))   # bit for bit

    def test_true_world_reweighted_rungs_take_t0s_loglik(self):
        out, _, _ = M.build([sampled_row(depth1=[LOW, LOW, LOW])], 'foulplay', [true_row()], weights([0., 0., 0.]), weights([0.] * 3))
        got = by_rung(out);self.assertEqual((got['R3true'], got['R3true_rw_b1']), (0, 0))   # (.9 - .9 - .9) / 3
        out, _, _ = M.build([sampled_row(depth1=[LOW, LOW, LOW])], 'foulplay', [true_row()], weights([0., -40., -40.]), weights([0.] * 3))
        self.assertEqual(by_rung(out)['R3true_rw_b1'], 1)   # T0 (its own loglik 0) outweighs the two LOW worlds at -40
        out, _, _ = M.build([sampled_row()], 'foulplay', [true_row()], weights([0., 0., 0.]), {})
        self.assertEqual(reasons(out)['R3true_rw_b1'], 'true_weights_missing')
        out, _, _ = M.build([sampled_row()], 'foulplay', [true_row()], weights([0., 0., 0.]), weights([0.]))   # one of three true worlds
        self.assertEqual(reasons(out)['R3true_rw_b1'], 'true_weights_missing')
        out, _, _ = M.build([sampled_row()], 'foulplay', [true_row()], weights([0., 0., 0.]), weights([0., -1., 0.]))   # sampled-like
        self.assertEqual(reasons(out)['R3true_rw_b1'], 'true_weights_inconsistent')
        out, _, _ = M.build([sampled_row()], 'foulplay', weights=weights([0., 0., 0., 0.]))   # a fourth world the row lacks
        self.assertEqual(reasons(out)['R3rw_b1'], 'world_weights_mismatch')

    def test_undecided_reasons(self):
        out, _, _ = M.build([sampled_row()], 'foulplay')
        self.assertEqual(sorted(out), ['R3'])   # no true rows, no weights: R3 alone
        out, _, _ = M.build([sampled_row()], 'foulplay', [true_row(ordinal=8)], weights({}))
        self.assertEqual(reasons(out), dict(R3=None, R1='true_row_missing', R3true='true_row_missing', R3true_ins='true_row_missing', R3or_active='true_row_missing',
                                            R3or_seen='true_row_missing',
                                            **{'R3rw_b0.5': 'world_weights_missing', 'R3rw_b1': 'world_weights_missing'}))
        bad = true_row();bad['worlds'][0]['kernel_seed'] = 999
        self.assertEqual(reasons(M.build([sampled_row()], 'foulplay', [bad])[0])['R3true'], 'kernel_seed_mismatch')
        none, unseeded = true_row(), sampled_row();none['worlds'][0]['kernel_seed'] = None;unseeded['worlds'][0]['kernel_seed'] = None
        self.assertEqual(reasons(M.build([unseeded], 'foulplay', [none])[0])['R3true'], 'kernel_seed_mismatch')   # None never matches
        shape = sampled_row(refused=[False])   # three worlds need two blocks
        self.assertEqual(set(reasons(M.build([shape], 'foulplay')[0]).values()), {'block_shape'})
        failed = true_row();failed['true_worlds'] = dict(TRUE, built=2)
        self.assertEqual(reasons(M.build([sampled_row()], 'foulplay', [failed])[0])['R3true'], 'true_worlds_failed')
        other = true_row();other['root']['root_logits'] = [0., 1.] + [0.] * 7
        self.assertEqual(reasons(M.build([sampled_row()], 'foulplay', [other])[0])['R3true'], 'root_prior_differs')
        refused = sampled_row(refused=[True, False])
        self.assertEqual(set(reasons(M.build([refused], 'foulplay', [true_row()])[0]).values()), {'refused_at_depth1'})
        _, _, summary = M.build([sampled_row(), sampled_row(true_worlds=TRUE)], 'foulplay')
        self.assertEqual(summary['sampled']['identity_true_worlds_row'], 1)

    def test_prefixes_take_the_first_worlds_and_only_their_blocks(self):
        four = sampled_row(depth1=[T0, T0, LOW, LOW], refused=[False, True])
        out, _, summary = M.build([four], 'foulplay', prefixes=[2, 4])
        self.assertEqual(by_rung(out)['R3_W2'], 1);self.assertEqual(reasons(out)['R3_W4'], 'refused_at_depth1')
        self.assertEqual(summary['prefixes'], [2, 4])
        out, _, _ = M.build([sampled_row(depth1=[T0, T0, LOW, LOW])], 'foulplay', prefixes=[2, 4, 8])
        self.assertEqual((by_rung(out)['R3_W2'], by_rung(out)['R3_W4'], reasons(out)['R3_W8']), (1, 0, 'prefix_too_long'))
        out, _, _ = M.build([sampled_row(depth1=[T0, T0, LOW, LOW])], 'foulplay', [true_row()], prefixes=[2, 4])   # three true worlds
        self.assertEqual((by_rung(out)['R1_W2'], reasons(out)['R1_W4']), (1, 'prefix_too_long'))

    def test_r3_check_counts_disagreements(self):
        theirs = [dict(kind='foulplay', ordinal=7, target=3, action=0)]   # R3 plays 1 here
        _, _, summary = M.build([sampled_row()], 'foulplay', check_r3=theirs)
        self.assertEqual((summary['r3_check']['compared'], summary['r3_check']['disagreements']), (1, 1))
        _, _, summary = M.build([sampled_row()], 'foulplay', check_r3=[dict(theirs[0], action=1)])
        self.assertEqual(summary['r3_check']['disagreements'], 0)
        four = sampled_row(depth1=[T0, T0, LOW, LOW])   # R3_W2 plays 1, R3_W4 plays 0
        _, _, summary = M.build([four], 'foulplay', prefixes=[2, 4], check_r3=[dict(theirs[0], action=1)], check_prefix=2)
        self.assertEqual((summary['r3_check']['rung'], summary['r3_check']['compared'], summary['r3_check']['disagreements']), ('R3_W2', 1, 0))
        with self.assertRaises(ValueError):M.build([four], 'foulplay', prefixes=[2, 4], check_r3=theirs)   # no R3 without the full W


class GuardTests(unittest.TestCase):
    def test_r1_takes_the_true_rows_guard_and_the_rest_the_sampled_rows(self):
        tripping = dict(tripped=True, b=True, c=False, own_guarded_lanes=[None] * 4, reasons=['b'])
        out, _, _ = M.build([sampled_row()], 'foulplay', [true_row(guard=tripping)])
        self.assertEqual((out['R1'][0]['action'], out['R1'][0]['guard_tripped']), (1, True))
        self.assertEqual((out['R3'][0]['action'], out['R3'][0]['guard_tripped']), (1, False))
        self.assertEqual(out['R3true'][0]['guard_tripped'], False)
        out, _, _ = M.build([sampled_row(guard=tripping)], 'foulplay', [true_row()])
        self.assertEqual((out['R1'][0]['guard_tripped'], out['R3'][0]['guard_tripped']), (False, True))


class TrueKindTests(unittest.TestCase):
    def test_a_public_true_replay_is_taken_only_with_its_kind(self):
        public = true_row();public['true_worlds'] = dict(public['true_worlds'], requested_kind='true', kind='true')
        out, _, summary = M.build([sampled_row()], 'foulplay', [public], true_kind='true')
        self.assertEqual((by_rung(out)['R1'], by_rung(out)['R3true'], summary['true_kind']), (1, 1, 'true'))
        out, _, summary = M.build([sampled_row()], 'foulplay', [public])   # the default wants true_private
        self.assertEqual(reasons(out)['R1'], 'true_row_missing');self.assertEqual(summary['true']['identity_not_true_worlds'], 1)
        out, _, _ = M.build([sampled_row()], 'foulplay', [true_row()], true_kind='true')
        self.assertEqual(reasons(out)['R1'], 'true_row_missing')
        failed = true_row();failed['true_worlds'] = dict(failed['true_worlds'], requested_kind='true', kind='true', built=2)
        self.assertEqual(reasons(M.build([sampled_row()], 'foulplay', [failed], true_kind='true')[0])['R1'], 'true_worlds_failed')


class OracleTests(unittest.TestCase):
    def test_oracle_rungs_keep_only_worlds_with_the_true_sets(self):
        # worlds: 0 wrong Skarmory (active), 1 both true, 2 wrong Snorlax; grids LOW, T0, LOW
        out, _, _ = M.build([sampled_row(depth1=[LOW, LOW, T0])], 'foulplay', [true_row()])
        got = by_rung(out);self.assertEqual(got['R3'], 0)
        self.assertEqual(got['R3or_active'], 0)   # worlds 1 and 2: (-.9 + .9) / 2, a tie: lowest index
        self.assertEqual(got['R3or_seen'], 0)     # world 1 alone: LOW
        out, _, _ = M.build([sampled_row(depth1=[LOW, T0, LOW])], 'foulplay', [true_row()])
        self.assertEqual((by_rung(out)['R3or_seen'], by_rung(out)['R3']), (1, 0))
        none = [[OTHER_SKARM, OTHER_LAX]] * 3
        out, _, _ = M.build([sampled_row(parties=none)], 'foulplay', [true_row()])
        self.assertEqual((reasons(out)['R3or_active'], reasons(out)['R3or_seen']), ('oracle_empty', 'oracle_empty'))
        unseen = [[member('blissey', ['softboiled'])], [member('chansey', ['softboiled'])], [member('raikou', ['thunderbolt'])]]
        out, _, _ = M.build([sampled_row(parties=unseen)], 'foulplay', [true_row()])   # no species in every world: nothing seen
        self.assertEqual((reasons(out)['R3or_active'], reasons(out)['R3or_seen']), ('oracle_empty', 'oracle_empty'))


class DiagnosticTests(unittest.TestCase):
    def test_p_true_set_raw_and_reweighted(self):
        # worlds: 0 wrong Skarmory, 1 both true, 2 wrong Snorlax; Skarmory active
        _, diag, summary = M.build([sampled_row()], 'foulplay', [true_row()], weights([0., math.log(2.), 0.]), weights([math.log(4.)] * 3))
        d = diag[0];self.assertEqual((d['seen'], d['active'], d['covered']), (['skarmory', 'snorlax'], 'skarmory', True))
        self.assertAlmostEqual(d['raw']['joint'], 1 / 3);self.assertAlmostEqual(d['raw']['per_mon']['skarmory'], 2 / 3)
        self.assertAlmostEqual(d['raw']['active'], 2 / 3);self.assertAlmostEqual(d['raw']['ess'], 3.)
        self.assertAlmostEqual(d['rw_b1']['joint'], .5);self.assertAlmostEqual(d['rw_b1']['per_mon']['snorlax'], .75)
        self.assertAlmostEqual(d['rw_b0.5']['joint'], math.sqrt(2) / (2 + math.sqrt(2)))
        self.assertAlmostEqual(d['true_loglik_minus_max'], math.log(2.));self.assertEqual(d['true_loglik_rank'], 1.)
        self.assertAlmostEqual(d['true_weight_b1'], 4 / (4 + 2 + 1))
        agg = summary['diagnostics']['3'];self.assertEqual((agg['roots'], agg['covered']), (1, 1.))
        self.assertAlmostEqual(agg['rw_b1_minus_raw']['joint']['mean'], round(.5 - 1 / 3, 5))

    def test_uncovered_mons_and_hidden_species(self):
        parties = [[OTHER_SKARM, OTHER_LAX, member('blissey', ['softboiled'])], [OTHER_SKARM, TRUE_PARTY[1]], [OTHER_SKARM, OTHER_LAX]]
        _, diag, _ = M.build([sampled_row(parties=parties)], 'foulplay', [true_row()])
        d = diag[0];self.assertEqual(d['seen'], ['skarmory', 'snorlax'])   # Blissey: one world's hidden draw
        self.assertEqual((d['covered'], d['mons_covered']), (False, dict(skarmory=False, snorlax=True)))
        self.assertEqual((d['raw']['joint'], d['raw']['log_joint'], d['raw']['active']), (0., None, 0.))
        self.assertNotIn('rw_b1', d)

    def test_diag_needs_a_usable_true_row(self):
        _, diag, summary = M.build([sampled_row()], 'foulplay', [true_row(ordinal=8)])
        self.assertEqual(diag, []);self.assertEqual(summary['diagnostics_skipped'], {'3:true_row_missing': 1})


class PairsTests(unittest.TestCase):
    def test_rungs_against_a_base_on_shared_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            rows = [bank_row('foulplay', 5, [0, 1], [0., .5], 0, 1, ordinal=7, target=3), bank_row('foulplay', 6, [0, 1], [.2, .1], 0, 0, ordinal=8, target=3)]
            (d / 'roots-foulplay-0.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            a = [dict(kind='foulplay', ordinal=7, target=3, action=1), dict(kind='foulplay', ordinal=8, target=3, action=1)]
            b = [dict(kind='foulplay', ordinal=7, target=3, action=0)]
            (d / 'a.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in a));(d / 'b.jsonl').write_text(json.dumps(b[0]) + '\n')
            with redirect_stdout(io.StringIO()):
                M.main(['pairs', str(d), '--rung', f'A={d / "a.jsonl"}', '--rung', f'B={d / "b.jsonl"}', '--base', 'B', '--out', str(d / 'p.json')])
            report = json.loads((d / 'p.json').read_text())
            self.assertEqual(report['pairs']['A - B/foulplay']['roots'], 1);self.assertEqual(report['pairs']['A - B/foulplay']['mean'], .5)
            self.assertEqual(report['rungs']['A']['foulplay']['scored'], 2)
            other = [dict(a[0], actions=[0, 2]), dict(a[1], seed=6)]   # root 7's echoed legal actions are not the bank's
            (d / 'c.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in other))
            got = M.pairs(d, [('C', d / 'c.jsonl'), ('B', d / 'b.jsonl')], ['B'])
            self.assertEqual((got['pairs']['C - B/foulplay']['roots'], got['pair_identity_mismatches']), (0, {'C': 1}))


class CliTests(unittest.TestCase):
    def test_rungs_writes_a_new_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp);(d / 's.jsonl').write_text(json.dumps(sampled_row()) + '\n');(d / 't.jsonl').write_text(json.dumps(true_row()) + '\n')
            with redirect_stdout(io.StringIO()):
                code = M.main(['rungs', '--kind', 'foulplay', '--sampled', str(d / 's.jsonl'), '--true', str(d / 't.jsonl'), '--out', str(d / 'o')])
            self.assertEqual(code, 0);self.assertEqual(sorted(p.name for p in (d / 'o').iterdir()),
                                                       ['diag.jsonl', 'rung-R1.jsonl', 'rung-R3.jsonl', 'rung-R3or_active.jsonl', 'rung-R3or_seen.jsonl', 'rung-R3true.jsonl',
                                                        'rung-R3true_ins.jsonl', 'summary.json'])
            with self.assertRaises(SystemExit):M.main(['rungs', '--kind', 'self', '--sampled', str(d / 's.jsonl'), '--out', str(d / 'p')])


if __name__ == '__main__':
    unittest.main()
