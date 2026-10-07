"""Follow-through must not degrade to first-action-only or silent raw fallback."""
from types import SimpleNamespace
from pathlib import Path
import json
import runpy
import tempfile
import unittest

from pokezero.mcts_eval.followthrough import continuation_seed, play_continuation
from pokezero.mcts_eval.paper_reference import Evaluation


class Env:
    def __init__(self, capped=False):
        self.boundary = 0
        self.actions = []
        self.chance = []
        self.capped = capped
    def terminal(self):
        return None if self.boundary < 4 else SimpleNamespace(capped=self.capped, winner='p1', turn_count=4)
    def requested_players(self):
        return ('p1', 'p2') if self.boundary in (0, 3) else ('p2',) if self.boundary == 1 else ('p1',)
    def observe(self, player):
        return SimpleNamespace(player=player, legal_action_mask=(True, True, False))
    def reseed_simulator_rng(self, seed):
        self.chance.append(seed)
    def step(self, actions):
        self.actions.append(actions)
        self.boundary += 1


def opponent(obs):
    assert obs.player == 'p2'
    return (0, 1), Evaluation((.25, .75), 0.)


class FollowThroughTests(unittest.TestCase):
    def test_resume_does_not_reapply_first_action_or_relabel_rng_boundaries(self):
        env=Env();env.boundary=2
        rows=[]
        result=play_continuation(env,subject='p1',first_action=0,decision_id='root',replicate=0,
            subject_selector=lambda *args:(1,{}),opponent_evaluator=opponent,emit=rows.append,
            start_boundary=2,prior_selections=0)
        self.assertEqual([r['boundary'] for r in rows],[2,3])
        self.assertTrue(result['first_action_applied'])
        self.assertEqual(result['followthrough_decisions'],2)
        self.assertEqual(env.actions[0]['p1'],1)
        self.assertEqual(rows[0]['chance_seed'],continuation_seed('root',0,2,'chance'))

    def test_fixed_first_action_then_search_on_every_own_request_only(self):
        calls, rows = [], []
        def search(obs, boundary, seed):
            self.assertEqual(obs.player, 'p1')
            calls.append((boundary, seed))
            return 1, {'searched': True}
        env = Env()
        result = play_continuation(env, subject='p1', first_action=0, decision_id='root',
            replicate=0, subject_selector=search, opponent_evaluator=opponent, emit=rows.append)
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertEqual(result['signed_outcome'], 1)
        self.assertEqual([b for b, _ in calls], [2, 3])
        self.assertEqual(result['followthrough_decisions'], 2)
        self.assertEqual(env.actions[0]['p1'], 0)
        self.assertTrue(all(row['evidence']['p2']['selector'] == 'champion_full_masked_policy_sample'
            for row in rows if 'p2' in row['evidence']))

    def test_pairing_domains_do_not_depend_on_arm_and_do_not_share_rng_streams(self):
        self.assertEqual(continuation_seed('root', 0, 2, 'chance'), continuation_seed('root', 0, 2, 'chance'))
        self.assertEqual(len({continuation_seed('root', 0, 2, domain)
            for domain in ('chance', 'opponent', 'search')}), 3)
        self.assertNotEqual(continuation_seed('root', 0, 2, 'chance'), continuation_seed('root', 1, 2, 'chance'))

    def test_refusal_propagates_without_raw_fallback_or_another_step(self):
        env = Env()
        def refuse(*args):
            raise RuntimeError('search refused')
        with self.assertRaisesRegex(RuntimeError, 'search refused'):
            play_continuation(env, subject='p1', first_action=0, decision_id='root', replicate=0,
                subject_selector=refuse, opponent_evaluator=opponent, emit=lambda row: None)
        self.assertEqual(len(env.actions), 2)

    def test_caps_are_not_scored_as_wins(self):
        for env, limit in ((Env(), 2), (Env(capped=True), 200)):
            result = play_continuation(env, subject='p1', first_action=0, decision_id='root', replicate=0,
                subject_selector=lambda *args: (1, {}), opponent_evaluator=opponent,
                emit=lambda row: None, max_boundaries=limit)
            self.assertEqual(result['status'], 'CAPPED')
            self.assertIsNone(result['signed_outcome'])


class RetainedContinuationTests(unittest.TestCase):
    def test_recursive_completed_cell_keeps_its_registration_hash(self):
        driver = runpy.run_path(str(Path(__file__).resolve().parents[1] /
            'scripts/search_followthrough_diagnostic.py'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefix = root/'prefix.json'; prefix.write_text('{}')
            roots = [{'ordinal':3,'decision_id':'d','first_choices':{'paper_reference':'move 1'}}]
            registration = root/'registration.json'
            registration.write_text(json.dumps(dict(roots=roots,replicates=[0,1],
                planners=['deep_incumbent','paper_reference'],source_hashes={},
                retained_input_hashes={str(prefix):driver['sha'](prefix)})))
            identity='root-03-paper_reference-search-r0'
            (root/(identity+'.json')).write_text(json.dumps(dict(identity=identity,ordinal=3,
                decision_id='d',planner='paper_reference',mode='search',replicate=0,first_choice='move 1',
                status='COMPLETE',step_hashes={},registration_sha256=driver['sha'](registration),
                retained_prefix_steps={'boundary-000.json':str(prefix)})))
            retained,_=driver['retained_cells'](root,roots,{})
            self.assertEqual(retained[identity]['registration_sha256'],driver['sha'](registration))

    def test_first_boundary_resume_has_no_invented_checkpoint_and_checks_recursive_prefix(self):
        driver = runpy.run_path(str(Path(__file__).resolve().parents[1] /
            'scripts/search_followthrough_diagnostic.py'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = 'root-08-paper_reference-search-r0'
            (root / identity).mkdir()
            step = root / identity / 'boundary-000.json'
            step.write_text(json.dumps(dict(boundary=0, actions={'p1':4, 'p2':2})))
            registration = root / 'registration.json'
            registration.write_text(json.dumps(dict(retained_input_hashes={})))
            cell = dict(status='REFUSED', planner='paper_reference', subject='p1', first_action=4,
                step_hashes={step.name:driver['sha'](step)})
            (root / (identity+'.json')).write_text(json.dumps(cell))
            resume = driver['register_resume'](root, {}, identity)[identity]
            self.assertEqual(resume['start_boundary'], 1)
            self.assertEqual(resume['prior_selections'], 0)
            self.assertIsNone(resume['checkpoint_step'])
            retained = root / 'original-boundary-000.json'
            step.rename(retained)
            cell['step_hashes'] = {}
            cell['retained_prefix_steps'] = {'boundary-000.json':str(retained)}
            (root / (identity+'.json')).write_text(json.dumps(cell))
            registration.write_text(json.dumps(dict(retained_input_hashes={str(retained):driver['sha'](retained)})))
            self.assertEqual(driver['register_resume'](root, {}, identity)[identity]['start_boundary'], 1)
            retained.write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'retained prefix drift'):
                driver['register_resume'](root, {}, identity)

    def test_statistics_checkpoint_is_json_safe_and_retains_all_q_n_m_f(self):
        from pokezero.mcts_eval.paper_reference import DecisionState
        from pokezero.mcts_eval.paper_reference_exchange import MasterSnapshot, Statistics
        driver = runpy.run_path(str(Path(__file__).resolve().parents[1] /
            'scripts/search_followthrough_diagnostic.py'))
        state = DecisionState(b'public-key', ('move:0', 'move:1'), 2)
        snapshot = MasterSnapshot('b', 3, 2, (Statistics(state, (2, 1), (-1., 1.), 3),), (('0', 1),))
        row = json.loads(json.dumps(driver['reference_statistics_checkpoint'](snapshot)))
        self.assertEqual(bytes.fromhex(row['rows'][0]['key_hex']), state.key)
        self.assertEqual(row['rows'][0]['visits'], [2, 1])
        self.assertEqual(row['rows'][0]['totals'], [-1., 1.])
        self.assertEqual((row['rows'][0]['count'], row['rows'][0]['faint_count']), (3, 2))
        self.assertEqual(row['faint_floor'], 2)

    def test_only_complete_cells_are_reused_and_refusal_stays_in_history(self):
        driver = runpy.run_path(str(Path(__file__).resolve().parents[1] /
            'scripts/search_followthrough_diagnostic.py'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roots = [{'ordinal': 3, 'decision_id': 'd', 'first_choices':
                {'deep_incumbent': 'move 1', 'paper_reference': 'move 2'}}]
            registration = root / 'registration.json'
            registration.write_text(json.dumps(dict(roots=roots, replicates=[0,1],
                planners=['deep_incumbent', 'paper_reference'], source_hashes={}, retained_input_hashes={})))
            for planner, status in [('deep_incumbent', 'COMPLETE'), ('paper_reference', 'REFUSED')]:
                identity = f'root-03-{planner}-search-r0'
                (root / f'{identity}.json').write_text(json.dumps(dict(identity=identity,
                    ordinal=3, decision_id='d', planner=planner, mode='search', replicate=0,
                    first_choice=roots[0]['first_choices'][planner], status=status, step_hashes={},
                    registration_sha256=driver['sha'](registration), error=None)))
            inputs = {}
            retained, history = driver['retained_cells'](root, roots, inputs)
            self.assertEqual(list(retained), ['root-03-deep_incumbent-search-r0'])
            self.assertEqual([row['identity'] for row in history], ['root-03-paper_reference-search-r0'])
            self.assertEqual(len(inputs), 3)
            # Same denominator/question, never a replacement position or action.
            cell = root / 'root-03-deep_incumbent-search-r0.json'
            row = json.loads(cell.read_text()); row['first_choice'] = 'move 4'
            cell.write_text(json.dumps(row))
            with self.assertRaisesRegex(RuntimeError, 'provenance drift'):
                driver['retained_cells'](root, roots, {})

    def test_retained_step_hash_mismatch_refuses(self):
        driver = runpy.run_path(str(Path(__file__).resolve().parents[1] /
            'scripts/search_followthrough_diagnostic.py'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roots = [{'ordinal': 3, 'decision_id': 'd', 'first_choices': {'deep_incumbent':'move 1'}}]
            registration = root / 'registration.json'
            registration.write_text(json.dumps(dict(roots=roots, replicates=[0,1],
                planners=['deep_incumbent', 'paper_reference'], source_hashes={}, retained_input_hashes={})))
            identity = 'root-03-deep_incumbent-search-r0'
            (root/identity).mkdir()
            (root/identity/'boundary-000.json').write_text('{}')
            (root/f'{identity}.json').write_text(json.dumps(dict(identity=identity, ordinal=3,
                decision_id='d', planner='deep_incumbent', mode='search', replicate=0,
                first_choice='move 1', status='COMPLETE', registration_sha256=driver['sha'](registration),
                step_hashes={'boundary-000.json':'wrong'})))
            with self.assertRaisesRegex(RuntimeError, 'decision evidence drift'):
                driver['retained_cells'](root, roots, {})


if __name__ == '__main__':
    unittest.main()
