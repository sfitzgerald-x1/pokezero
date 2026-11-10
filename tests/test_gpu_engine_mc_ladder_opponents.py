"""CPU tests of gpu_engine/mc_ladder_opponents.py (the MC ladder's opponent sidecars): no Showdown, no torch, no JAX. The env, the
champion and the owner capture are fakes; the replays are the real ones (the OV harness's load_game and replay_rounds,
mc_root_labeller.self_root). Import closure: the standard library and numpy."""
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from gpu_engine import mc_ladder_opponents as M

TEAM = [dict(species=n, details=f'{n}, L80') for n in ('Snorlax', 'Pikachu', 'Mr. Mime', 'Gengar', 'Ho-Oh', 'Zapdos')]
OUR_TEAM = [dict(species=n, details=f'{n}, L80') for n in ('Starmie', 'Skarmory', 'Tyranitar', 'Metagross', 'Blissey', 'Salamence')]


def request(side, condition='100/100', *, wait=False, rqid=1):
    """A literal request of `side`: its active member's condition, a wait request without the active block."""
    out = dict(rqid=rqid, side=dict(id=side, name=f'name-{side}', pokemon=[dict(details=TEAM[0]['details'], condition=condition, active=True)]))
    if wait:out['wait'] = True
    else:out['active'] = [dict(moves=[dict(id='bodyslam', pp=20)])]
    return out


def observation(team=None):
    return SimpleNamespace(numeric_features=np.zeros((23, 132)), categorical_ids=np.zeros((23, 41)), token_type_ids=np.zeros(23),
        attention_mask=np.ones(23, bool), legal_action_mask=(True,) * 9, metadata=dict(self_team=team))


class ScriptEnv:
    """LocalShowdownEnv's replay surface over a script: per round, which seats are requested and each seat's literal request."""
    def __init__(self, script, teams):
        self.script, self.teams, self.closed, self.steps, self.resets = script, teams, 0, [], []

    def reset(self, *, seed):
        self.resets.append(seed);self.round = 0;self._load()

    def _load(self):
        step = self.script[self.round] if self.round < len(self.script) else {}
        self._latest_turn = self.round + 1;self._latest_requests = {p: r for p, (on, r) in step.items()}

    def terminal(self):
        return SimpleNamespace(winner='p1') if self.round >= len(self.script) else None

    def requested_players(self):
        return tuple(p for p in ('p1', 'p2') if self.script[self.round][p][0])

    def observe(self, player):
        return observation(self.teams[player])

    def step(self, actions):
        assert set(actions) == set(self.requested_players()), actions
        self.steps.append(dict(actions));self.round += 1;self._load()

    def close(self):
        self.closed += 1


# Foul Play roots ------------------------------------------------------------------------------------------------------------------
# rounds 0..3: both decide; Foul Play replaces while we wait; we replace while Foul Play waits; both decide. Our decisions 0, 1, 2
# sit in rounds 0, 2, 3 (request_count 1, 3, 4).
FP_SCRIPT = (dict(p1=(True, request('p1', rqid=1)), p2=(True, request('p2', '90/100', rqid=1))),
    dict(p1=(False, request('p1', wait=True, rqid=2)), p2=(True, request('p2', '90/100', rqid=2))),
    dict(p1=(True, request('p1', rqid=3)), p2=(False, request('p2', '40/100', wait=True, rqid=3))),
    dict(p1=(True, request('p1', rqid=4)), p2=(True, request('p2', '35/100', rqid=4))))
OUR_ROUNDS = (0, 2, 3)


def write_game(collection, ordinal=12, seed=7):
    """results/<ordinal>.json (Foul Play p2 with a journal entry per requested round) and our telemetry, as the collection holds them."""
    for d in ('results', 'telemetry', 'owning-wire'):(collection / d).mkdir(parents=True, exist_ok=True)
    journal = [dict(round=r, seat='p2', choice='move 1', action=0) for r, step in enumerate(FP_SCRIPT) if step['p2'][0]]
    (collection / 'results' / f'{ordinal}.json').write_text(json.dumps(dict(status='COMPLETE', source_game=dict(seed=seed, opponent_moves=journal),
        benchmark=dict(foulplay_player='p2'))))
    (collection / 'telemetry' / f'{ordinal}.jsonl').write_text(''.join(json.dumps(dict(owner='p1', decision_index=i, played_action_index=0)) + '\n' for i in range(3)))


def capture(rounds=OUR_ROUNDS, script=FP_SCRIPT, owner='p1'):
    """Owner capture ROOT rows: our request line per decision and its request_count (round + 1)."""
    return [dict(decision_index=i, owner=owner, request_count=r + 1, own_request_line='|request|' + json.dumps(dict(script[r][owner][1], rqid=99)))
        for i, r in enumerate(rounds)]


class FoulPlayTests(unittest.TestCase):
    def sidecar(self, targets, rows=None, **entry):
        with tempfile.TemporaryDirectory() as tmp, patch.object(M, 'capture_rows', return_value=capture() if rows is None else rows) as read:
            collection = Path(tmp);write_game(collection);env = ScriptEnv(FP_SCRIPT, dict(p1=OUR_TEAM, p2=TEAM))
            doc = M.foulplay_sidecar(env, collection, 12, dict(dict(seed=7, ours='p1', targets=targets), **entry))
            self.assertEqual(read.call_args.args[0], collection / 'owning-wire' / '12.jsonl')
        return doc, env

    def test_the_sidecar_holds_the_true_team_round_zero_and_each_roots_request(self):
        doc, env = self.sidecar([0, 1, 2, 5])
        self.assertEqual((env.resets, len(env.steps)), ([7], 4))   # the whole game, replayed once
        self.assertEqual({k: doc[k] for k in ('schema', 'kind', 'ordinal', 'seed', 'ours', 'opponent')},
            dict(schema=M.SCHEMA, kind='foulplay', ordinal=12, seed=7, ours='p1', opponent='p2'))
        self.assertEqual(doc['team_rows'], TEAM);self.assertEqual(doc['round0_request'], FP_SCRIPT[0]['p2'][1])
        self.assertEqual(list(doc['roots']), ['0', '1', '2', '5'])
        self.assertEqual(doc['roots']['0'], dict(decision_index=0, round=0, turn=1, opponent_requested=True, root_request=FP_SCRIPT[0]['p2'][1]))
        self.assertEqual(doc['roots']['1'], dict(decision_index=1, round=2, turn=3, opponent_requested=False, root_request=None))   # Foul Play waits
        self.assertEqual(doc['roots']['2']['root_request']['side']['pokemon'][0]['condition'], '35/100')
        self.assertEqual(doc['roots']['5'], dict(decision_index=5, error='no owner capture row'))
        self.assertEqual(json.loads(json.dumps(doc)), doc)   # JSON as written

    def test_a_root_whose_round_holds_another_decision_carries_an_error(self):
        rows = capture();rows[2] = dict(rows[2], request_count=3)   # round 2: decision 1's
        doc, _ = self.sidecar([1, 2], rows)
        self.assertEqual(doc['roots']['2'], dict(decision_index=2, round=2, error='the replay gives round 2 to decision 1'))
        self.assertNotIn('error', doc['roots']['1'])

    def test_a_game_that_is_not_the_banks_or_the_captures_is_refused(self):
        ov = M._ov()
        with self.assertRaises(ov.CellError) as caught:self.sidecar([0], ours='p2')
        self.assertEqual(caught.exception.kind, 'seat_differs')
        with self.assertRaises(ov.CellError) as caught:self.sidecar([0], seed=8)
        self.assertEqual(caught.exception.kind, 'seed_differs')
        rows = capture();rows[1] = dict(rows[1], own_request_line=rows[1]['own_request_line'].replace('"rqid": 99', '"rqid": 99, "x": 1'))
        with self.assertRaisesRegex(ov.Mismatch, 'differs from the capture at decision 1'):self.sidecar([0], rows)

    def test_bank_rows_group_by_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'roots-foulplay-0.jsonl', Path(tmp) / 'roots-foulplay-1.jsonl'
            a.write_text(''.join(json.dumps(dict(ordinal=o, target=t, seed=s, ours='p1', kind='foulplay')) + '\n' for o, t, s in ((12, 5, 7), (3, 1, 9), (12, 2, 7))))
            b.write_text(json.dumps(dict(ordinal=12, target=9, seed=7, ours='p1')) + '\n\n')
            self.assertEqual(M.bank_roots([a, b]), {3: dict(seed=9, ours='p1', targets=[1]), 12: dict(seed=7, ours='p1', targets=[2, 5, 9])})
            b.write_text(json.dumps(dict(ordinal=12, target=9, seed=7, ours='p2')) + '\n')
            with self.assertRaisesRegex(ValueError, 'seed or seat differs'):M.bank_roots([a, b])


# Self roots -----------------------------------------------------------------------------------------------------------------------
# We are p2. Rounds 0..2: both decide; we decide while the opponent (p1) waits; both decide. Our decisions 0, 1, 2 sit in rounds 0, 1, 2.
SELF_SCRIPT = (dict(p1=(True, request('p1', '100/100', rqid=1)), p2=(True, request('p2', rqid=1))),
    dict(p1=(False, request('p1', '70/100', wait=True, rqid=2)), p2=(True, request('p2', rqid=2))),
    dict(p1=(True, request('p1', '55/100', rqid=3)), p2=(True, request('p2', rqid=3))))


class Champion:
    """The champion's probabilities: uniform over nine lanes."""
    def __init__(self):
        self.calls = 0

    def probabilities(self, observations):
        self.calls += 1
        return np.full((len(observations), 9), 1 / 9)


class SelfTests(unittest.TestCase):
    def sidecar(self, target, *, rows=None, captured=True):
        env = ScriptEnv(SELF_SCRIPT, dict(p1=TEAM, p2=OUR_TEAM));row = dict(ordinal=4, seed=31, ours='p2', target=target, status='ok')
        with tempfile.TemporaryDirectory() as tmp, patch.object(M, 'capture_rows', return_value=capture((0, 1, 2), SELF_SCRIPT, 'p2') if rows is None else rows):
            (Path(tmp) / 'owning-wire').mkdir()
            if captured:(Path(tmp) / 'owning-wire' / '4.jsonl').write_text('')
            return M.self_sidecar(env, Champion(), row, tmp), env

    def test_the_sidecar_holds_the_opponents_round_zero_team_and_its_root_request(self):
        doc, env = self.sidecar(2)
        self.assertEqual((env.resets, len(env.steps)), ([31, 31], 2))   # round 0 read after a reset; self_root resets and plays to the root
        self.assertEqual({k: doc[k] for k in ('kind', 'ordinal', 'seed', 'ours', 'opponent')}, dict(kind='self', ordinal=4, seed=31, ours='p2', opponent='p1'))
        self.assertEqual((doc['team_rows'], doc['round0_request']), (TEAM, SELF_SCRIPT[0]['p1'][1]))
        self.assertEqual(doc['roots'], {'2': dict(decision_index=2, round=None, opponent_requested=True, root_request=SELF_SCRIPT[2]['p1'][1])})
        self.assertEqual(doc['source'], dict(manifest_status='ok', capture_checked=True))

    def test_a_waiting_opponent_has_no_root_request(self):
        doc, _ = self.sidecar(1, captured=False)
        self.assertEqual(doc['roots'], {'1': dict(decision_index=1, round=None, opponent_requested=False, root_request=None)})
        self.assertFalse(doc['source']['capture_checked'])

    def test_no_root_or_another_battle_than_the_capture_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'no root'):self.sidecar(5)
        rows = capture((0, 1, 2), SELF_SCRIPT, 'p2');rows[2] = dict(rows[2], own_request_line=rows[0]['own_request_line'].replace('"rqid": 99', '"rqid": 99, "x": 1'))
        with self.assertRaisesRegex(ValueError, 'differs from the owner capture'):self.sidecar(2, rows=rows)

    def test_manifests_are_read_in_ordinal_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'manifest-0.jsonl').write_text(json.dumps(dict(ordinal=2, seed=1, ours='p1', target=3)) + '\n')
            (Path(tmp) / 'manifest-1.jsonl').write_text(json.dumps(dict(ordinal=1, seed=5, ours='p2', target=4)) + '\n')
            self.assertEqual([r['ordinal'] for r in M.manifest_rows(tmp)], [1, 2])
            (Path(tmp) / 'manifest.jsonl').write_text(''.join(json.dumps(dict(ordinal=o, seed=1, ours='p1', target=3)) + '\n' for o in (2, 1)))
            self.assertEqual([r['ordinal'] for r in M.manifest_rows(tmp)], [1, 2])   # the merged manifest wins over the workers'
            (Path(tmp) / 'manifest.jsonl').write_text(json.dumps(dict(ordinal=1, seed=5, ours='p2', target=4)) + '\n' * 1 * 1
                                                      + json.dumps(dict(ordinal=1, seed=5, ours='p2', target=4)) + '\n')
            with self.assertRaisesRegex(ValueError, 'twice'):M.manifest_rows(tmp)   # one ordinal twice in a manifest


# Writing -------------------------------------------------------------------------------------------------------------------------
class WriteTests(unittest.TestCase):
    def test_each_cell_gets_one_file_named_by_kind_and_ordinal(self):
        envs = []

        def make_env():
            envs.append(SimpleNamespace(closed=0));envs[-1].close = lambda e=envs[-1]: setattr(e, 'closed', e.closed + 1)
            return envs[-1]

        def build(env, ordinal, item):
            if item == 'bad':raise RuntimeError('bridge died')
            roots = {0: dict(decision_index=0, opponent_requested=True, root_request={}), 3: dict(decision_index=3, opponent_requested=False, root_request=None),
                4: dict(decision_index=4, error='x')}
            return M.document('foulplay', ordinal, 7, 'p1', TEAM, dict(side={}), roots)
        with tempfile.TemporaryDirectory() as tmp, patch('sys.stdout', new_callable=io.StringIO):
            out = Path(tmp);(out / 'opponent').mkdir();(out / 'opponent' / 'foulplay-9.json').write_text('{}')
            summary = M.write_all('foulplay', [(1, 'a'), (2, 'bad'), (9, 'c'), (5, 'd')], out, make_env, build)
            written = sorted(p.name for p in (out / 'opponent').iterdir());one = json.loads((out / 'opponent' / 'foulplay-1.json').read_text())
        self.assertEqual(written, ['foulplay-1.json', 'foulplay-5.json', 'foulplay-9.json'])
        self.assertEqual((summary['written'], summary['existing'], summary['failed'], summary['roots'], summary['opponent_waits'], summary['root_errors']),
            (2, 1, 1, 6, 2, 2))
        self.assertEqual(summary['errors'], [dict(ordinal=2, error='RuntimeError: bridge died')])
        self.assertEqual([e.closed for e in envs], [1, 1])   # the env is rebuilt after a failure, and closed at the end
        self.assertEqual(list(one['roots']), ['0', '3', '4']);self.assertEqual((one['ordinal'], one['opponent'], one['team_rows']), (1, 'p2', TEAM))

    def test_a_document_needs_six_rows_and_a_round_zero_request(self):
        with self.assertRaisesRegex(ValueError, '5 self_team rows'):M.document('self', 1, 2, 'p1', TEAM[:5], {}, {})
        with self.assertRaisesRegex(ValueError, 'no round-0 request'):M.document('self', 1, 2, 'p1', TEAM, None, {})

    def test_the_command_line_pairs_each_kind_with_its_input(self):
        base = ['--collection', 'c', '--checkpoint', 'x.pt', '--out', 'o']
        for argv in (['foulplay', *base], ['self', *base], ['self', '--bank', 'b', *base], ['foulplay', '--bank', 'b', '--captures', 'd', *base],
                ['self', '--captures', 'd', '--worker', '2', '--workers', '2', *base]):
            with self.subTest(argv=argv), patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit):M.main(argv)


if __name__ == '__main__':
    unittest.main()
