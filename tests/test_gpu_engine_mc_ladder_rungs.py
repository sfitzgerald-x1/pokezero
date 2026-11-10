"""mc_ladder_rungs: per-rung choices from hand-built re-pricer rows (two legal actions, two kept replies, three worlds)."""
import io
import json
import math
import statistics
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from gpu_engine import mc_ladder_rungs as M
from gpu_engine import search_hedge as hedge

LEGAL = [a in (0, 1) for a in range(9)]
LOGITS = [1., 0.] + [0.] * 7            # greedy: action 0
UNIFORM = [0.] * 9                      # the champion's reply logits: 1/2 on each kept reply
REPLY = [math.log(3.), 0.] + [0.] * 7   # the reply model: 3/4 on reply 0, 1/4 on reply 1
KEPT = [b in (0, 1) for b in range(9)]
MOVES = ['tackle', 'surf', 'rest', 'icebeam']
CLEAR = dict(tripped=False, b=False, c=False, own_guarded_lanes=[None] * 4)
# {own action: (value vs reply 0, value vs reply 1)} per world. Action 0 is worth 0 everywhere, so Q(1) decides.
D1 = [{0: (0., 0.), 1: (.4, -.6)}, {0: (0., 0.), 1: (.2, -.4)}, {0: (0., 0.), 1: (.4, -.2)}]
D2 = [{0: (0., 0.), 1: (.2, 0.)}, {0: (0., 0.), 1: (.2, .2)}, {0: (0., 0.), 1: (-.1, .1)}]
NG = [{0: (0., 0.), 1: (.6, 0.)}, {0: (0., 0.), 1: (-.2, .4)}, {0: (0., 0.), 1: (0., .2)}]


def grid(values):
    return [[values[a][b] if a in values and b < len(values[a]) else None for b in range(9)] for a in range(9)]


def prices(values, requested=True):
    return dict(priced=True, opponent_requested=requested, kept=KEPT if requested else [False] * 9, root_peer_logits=UNIFORM, grid=grid(values))


def row(ordinal=7, index=3, *, depth1=D1, depth2=D2, status='COMPLETE', action=1, decided=True, reply=True, guard=CLEAR, moves=MOVES,
        refused=None, waiting=(), cause=None, **extra):
    """One re-pricer row; `waiting` lists worlds whose opponent waits (column 0 only, no reply-model logits)."""
    worlds = []
    for w in range(len(depth1)):
        wait = w in waiting
        world = dict(world=w, **prices(depth2[w], not wait), depth1=prices(depth1[w], not wait))
        if reply:world['reply_peer_logits'] = None if wait else REPLY
        worlds.append(world)
    q = None
    if status == 'COMPLETE':   # the live decide's own q on the depth-2 worlds
        q = list(hedge.averaging(M.rr.world_prices(M.rr.normalize(dict(worlds=worlds))), tuple(LEGAL)).q)
    out = dict(ordinal=ordinal, decision_index=index, seed=11, owner='p1', worlds=worlds,
               replay=dict(status=status, action_index=action if status == 'COMPLETE' else None, q_values=q),
               root=dict(decided=decided, root_logits=LOGITS if decided else None, root_legal=LEGAL if decided else None, own_moves=moves,
                         refusal_cause=cause if cause else None if status == 'COMPLETE' else 'depthn_refused'),
               effect_guard=guard if status == 'COMPLETE' else None,
               depth1=dict(block_causes=[None, None], priced_worlds=len(depth1), refused_at_depth1=refused or [False, False]))
    out.update(extra)
    return out


def one(raw, run):
    """{rung: rung row} of a single row."""
    out, _ = M.build([raw], run, None if run != 'E' else {raw['ordinal']: dict(ordinal=raw['ordinal'], seed=11, ours='p1', target=raw['decision_index'])})
    return {name: rows[0] for name, rows in out.items()}


def hand_q(worlds, prior):
    """Per-world Q_w(1) - Q_w(0) by hand: Q_w(a) = sum_b prior(b) G_w[a][b] (action 0 is worth 0)."""
    return [prior[0] * w[1][0] + prior[1] * w[1][1] for w in worlds]


def gate(gaps, k=2):
    return statistics.fmean(gaps) > k * statistics.stdev(gaps) / math.sqrt(len(gaps))


class RungChoiceTests(unittest.TestCase):
    def test_run_c_rungs_against_a_hand_computation(self):
        reply, champion = hand_q(D1, (.75, .25)), hand_q(D1, (.5, .5))
        self.assertEqual([round(x, 9) for x in reply], [.15, .05, .25]);self.assertEqual([round(x, 9) for x in champion], [-.1, -.1, .1])
        self.assertTrue(gate(reply))   # mean .15 > 2 * .1 / sqrt(3)
        got = one(row(), 'C')
        self.assertEqual({n: r['action'] for n, r in got.items()}, dict(R3=1, R3prime_ng=1, R4b=0))
        for r in got.values():
            self.assertEqual((r['kind'], r['ordinal'], r['target'], r['seed']), ('foulplay', 7, 3, 11))
            self.assertEqual((r['greedy_action'], r['guard_tripped'], r['actions']), (0, False, [0, 1]));self.assertNotIn('reason', r)
        d = M.rr.normalize(row())
        a, _, q = M.averaged(row(), d, 1, True);self.assertEqual(a, 1);self.assertAlmostEqual(q[1], statistics.fmean(reply), places=12)
        a, _, q = M.averaged(row(), d, 1, False);self.assertEqual(a, 0);self.assertAlmostEqual(q[1], statistics.fmean(champion), places=12)

    def test_noise_gate_holds_back_a_noisy_depth1_override(self):
        reply = hand_q(NG, (.75, .25))
        self.assertGreater(statistics.fmean(reply), 0);self.assertFalse(gate(reply))   # mean .15 < 2 * .265 / sqrt(3)
        got = one(row(depth1=NG), 'C')
        self.assertEqual({n: r['action'] for n, r in got.items()}, dict(R3=1, R3prime_ng=0, R4b=1))

    def test_run_d_r5_is_the_replay_and_r5_ng_gates_the_depth2_q(self):
        gaps = hand_q(D2, (.5, .5))
        self.assertAlmostEqual(statistics.fmean(gaps), .1);self.assertFalse(gate(gaps))   # averaging plays 1; the gate keeps greedy
        out, summary = M.build([row()], 'D')
        self.assertEqual({n: rows[0]['action'] for n, rows in out.items()}, dict(R4=0, R5=1, R5_ng=0))
        self.assertEqual(summary['r5_vs_replay'], dict(compared=1, action_disagreements=0, q_disagreements=0, recompute_failed={}, examples=[]))
        self.assertEqual(summary['rungs']['R5'], dict(rows=1, decided=1, undecided={}, guard_tripped={}))

    def test_r5_disagreeing_with_the_replay_is_counted(self):
        bad = row(action=0)   # the replay played 0 where averaging on its own grids plays 1
        bad['replay']['q_values'][1] += 1e-9
        out, summary = M.build([bad], 'D')
        self.assertEqual(out['R5'][0]['action'], 0)   # R5 is the live code's own choice, whatever the recomputation says
        p = summary['r5_vs_replay']
        self.assertEqual((p['compared'], p['action_disagreements'], p['q_disagreements']), (1, 1, 1))
        self.assertEqual(p['examples'], [dict(kind='foulplay', ordinal=7, target=3, seed=11, disagreement=['action', 'q_values'])])

    def test_self_roots_write_r3_as_the_champion_depth1_choice(self):
        got = one(row(reply=False), 'E')
        self.assertEqual({n: r['action'] for n, r in got.items()}, dict(R3=0, R3prime_ng=0, R4=0, R5=1, R5_ng=0))
        self.assertEqual({k: got['R3'][k] for k in ('kind', 'seed', 'ours', 'target')}, dict(kind='self', seed=11, ours='p1', target=3))
        self.assertNotIn('ordinal', got['R3'])

    def test_a_waiting_world_needs_no_reply_logits(self):
        depth1 = D1[:2] + [{0: (0.,), 1: (.1,)}]   # world 2 waits: column 0 at mass 1
        got = one(row(depth1=depth1, depth2=D2[:2] + [{0: (0.,), 1: (.1,)}], waiting=(2,)), 'C')
        self.assertEqual((got['R3']['action'], got['R4b']['action']), (1, 0))   # reply mean (.15 + .05 + .1) / 3; champion (-.1 - .1 + .1) / 3


class GuardTests(unittest.TestCase):
    def test_b_trips_every_decided_rung(self):
        got = one(row(guard=dict(CLEAR, tripped=True, b=True)), 'C')
        for r in got.values():self.assertEqual((r['guard_tripped'], r['guard_reason'], r['greedy_action']), (True, 'UNREPRESENTED_EFFECT', 0))

    def test_c_and_recharge_apply_to_each_rungs_own_choice(self):
        got = one(row(guard=dict(CLEAR, own_guarded_lanes=[None, 'superpower', None, None])), 'C')
        self.assertEqual((got['R3']['guard_tripped'], got['R3']['guard_reason']), (True, 'UNREPRESENTED_EFFECT'))   # R3 plays lane 1
        self.assertFalse(got['R4b']['guard_tripped'])   # R4b plays greedy's lane 0
        got = one(row(moves=['tackle', 'hyperbeam', 'rest', 'icebeam']), 'C')
        self.assertEqual(got['R3']['guard_reason'], 'OWN_RECHARGE_UNREPRESENTED');self.assertFalse(got['R4b']['guard_tripped'])

    def test_the_union_reply_arm_guards_only_reply_model_rungs(self):
        got = one(row(guard=dict(CLEAR, tripped=True, union=dict(tripped=True, champion=dict(b=False), reply=dict(b=True)))), 'C')
        self.assertTrue(got['R3']['guard_tripped']);self.assertTrue(got['R3prime_ng']['guard_tripped'])
        self.assertFalse(got['R4b']['guard_tripped'])   # the champion arm's own b is false

    def test_a_decided_root_without_a_guard_record_fails_closed(self):
        got = one(row(status='REFUSED'), 'D')   # depth 2 refused after decide; the depth-1 frame priced it
        self.assertEqual((got['R4']['action'], got['R4']['guard_tripped'], got['R4']['guard_reason']), (0, True, 'no_guard_record'))
        for name in ('R5', 'R5_ng'):
            self.assertIsNone(got[name]['action']);self.assertEqual(got[name]['reason'], 'replay_REFUSED:depthn_refused')
            self.assertFalse(got[name]['guard_tripped'])   # an undecided row never claims a trip
        got = one(row(status='REFUSED'), 'C')   # decide ran (so the reply model's logits exist) but refused: no guard record
        self.assertEqual({n: r['action'] for n, r in got.items()}, dict(R3=1, R3prime_ng=1, R4b=0))
        for r in got.values():self.assertEqual((r['guard_tripped'], r['guard_reason']), (True, 'no_guard_record'))


class UndecidedTests(unittest.TestCase):
    def reasons(self, raw, run='C'):return {n: r.get('reason') for n, r in one(raw, run).items() if r['action'] is None}

    def test_a_root_decide_never_saw_uses_the_priced_entrys_root_prior(self):
        raw = row(status='REFUSED', decided=False, reply=False)   # refused at depth 2: no decide, so no reply-model logits
        raw['depth1'].update(root_logits=list(LOGITS), root_legal=list(LEGAL))
        got = one(raw, 'C')
        self.assertEqual(got['R3']['reason'], 'reply_prior_missing')
        self.assertIsNotNone(got['R4b']['action'])
        self.assertEqual((got['R4b']['guard_tripped'], got['R4b']['guard_reason']), (True, 'no_guard_record'))   # fail closed

    def test_reasons(self):
        self.assertEqual(self.reasons(row(status='REFUSED', decided=False)), dict.fromkeys(('R3', 'R3prime_ng', 'R4b'), 'root_prior_missing'))
        self.assertEqual(self.reasons(row(refused=[False, True])), dict.fromkeys(('R3', 'R3prime_ng', 'R4b'), 'refused_at_depth1'))
        recharge = dict(row(status='REFUSED', decided=False, cause='OPPONENT_RECHARGE_UNREPRESENTED'), depth1=None)   # refused before pricing
        self.assertEqual(self.reasons(recharge, 'D'), dict(R4='depth1_missing:OPPONENT_RECHARGE_UNREPRESENTED',
                         R5='replay_REFUSED:OPPONENT_RECHARGE_UNREPRESENTED', R5_ng='replay_REFUSED:OPPONENT_RECHARGE_UNREPRESENTED'))
        self.assertEqual(self.reasons(row(decided=False), 'D'), dict.fromkeys(('R4', 'R5', 'R5_ng'), 'root_prior_missing'))   # COMPLETE, no root
        missing = row();missing['worlds'][1]['reply_peer_logits'] = None
        self.assertEqual(self.reasons(missing), dict(R3='reply_prior_missing', R3prime_ng='reply_prior_missing'))   # R4b still decides
        unpriced = row();unpriced['worlds'][0]['depth1'].update(priced=False, grid=None)
        self.assertEqual(set(self.reasons(unpriced).values()), {'world_unpriced'})
        hole = row();hole['worlds'][2]['depth1']['grid'][1][1] = None   # a needed cell (action 1 vs kept reply 1) unpriced
        self.assertEqual(set(self.reasons(hole).values()), {'unpriced_cell'})
        undecided = one(row(status='REFUSED', decided=False), 'C')['R3']
        self.assertEqual((undecided['greedy_action'], undecided['guard_tripped'], undecided['actions']), (None, False, None))


class IdentityTests(unittest.TestCase):
    MANIFEST = {0: dict(ordinal=0, seed=900, ours='p2', target=3, status='ok'), 1: dict(ordinal=1, seed=901, ours='p1', target=5, status='ok'),
                2: dict(ordinal=2, seed=902, ours='p2', target=4, status='unexplained'), 3: dict(ordinal=3, seed=903, ours='p1', target=1)}

    def test_self_rows_map_through_the_manifest_and_mismatches_are_counted(self):
        rows = [row(0, 3, owner='p2', seed=900, reply=False),   # mapped
                row(1, 4, owner='p1', seed=901, reply=False),   # decision 4 is not the target 5
                row(2, 4, owner='p1', seed=902, reply=False),   # the capture is p1's, the bank root p2's
                row(3, 1, owner='p1', seed=7, reply=False),     # another battle's seed
                row(9, 0, reply=False),                         # not in the manifest
                row(0, 3, owner='p2', seed=900, reply=False, target_kind='foulplay')]
        out, summary = M.build(rows, 'E', self.MANIFEST)
        self.assertEqual([dict((k, r[k]) for k in ('kind', 'seed', 'ours', 'target')) for r in out['R4']], [dict(kind='self', seed=900, ours='p2', target=3)])
        self.assertEqual((summary['rows'], summary['mapped']), (6, 1))
        self.assertEqual(summary['identity_mismatches'], dict(kind_mismatch=1, not_in_manifest=1, seat_mismatch=1, seed_mismatch=1, target_mismatch=1))
        self.assertEqual(summary['manifest_status'], dict(ok=1))

    def test_foulplay_rows_key_on_ordinal_and_decision_index(self):
        out, summary = M.build([row(5, 2), row(5, 4), row(6, 2, seed=None)], 'D')
        self.assertEqual([(r['ordinal'], r['target'], r.get('seed')) for r in out['R4']], [(5, 2, 11), (5, 4, 11), (6, 2, None)])
        self.assertNotIn('manifest_status', summary)

    def test_two_rows_for_one_root_are_an_error(self):
        with self.assertRaises(ValueError):M.build([row(5, 2), row(5, 2)], 'D')


class CommandLineTests(unittest.TestCase):
    def test_writes_one_file_per_rung_and_the_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = Path(tmp) / 'rows-0.jsonl';rows.write_text(json.dumps(row()) + '\n' + json.dumps(row(8, 1, status='REFUSED', decided=False)) + '\n')
            with redirect_stdout(io.StringIO()):code = M.main(['--rows', str(rows), '--run', 'C', '--out', str(Path(tmp) / 'out')])
            self.assertEqual(code, 0)
            self.assertEqual(sorted(p.name for p in (Path(tmp) / 'out').iterdir()), ['rung-R3.jsonl', 'rung-R3prime_ng.jsonl', 'rung-R4b.jsonl', 'summary.json'])
            r3 = [json.loads(l) for l in (Path(tmp) / 'out' / 'rung-R3.jsonl').read_text().splitlines()]
            self.assertEqual([(r['ordinal'], r['action']) for r in r3], [(7, 1), (8, None)])
            summary = json.loads((Path(tmp) / 'out' / 'summary.json').read_text())
            self.assertEqual(summary['rungs']['R3'], dict(rows=2, decided=1, undecided=dict(root_prior_missing=1), guard_tripped={}))
            self.assertIsNone(summary['r5_vs_replay'])
            with redirect_stdout(io.StringIO()):code = M.main(['--rows', str(rows), '--run', 'D', '--out', str(Path(tmp) / 'd')])
            self.assertEqual(code, 0)
            bad = Path(tmp) / 'bad.jsonl';bad.write_text(json.dumps(row(action=0)) + '\n')
            with redirect_stdout(io.StringIO()):self.assertEqual(M.main(['--rows', str(bad), '--run', 'D', '--out', str(Path(tmp) / 'bad')]), 1)

    def test_the_manifest_belongs_to_run_e_only(self):
        for argv in (['--run', 'E'], ['--run', 'C', '--manifest', 'm.jsonl']):
            with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                M.main(['--rows', 'r.jsonl', '--out', 'o'] + argv)


if __name__ == '__main__':
    unittest.main()
