"""CPU tests. All bank rows and outcomes are inert fixtures."""
import json
from pathlib import Path
import tempfile
import shutil
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pokezero import v5_progress_evaluation as progress, v5_progress_report as report, v5_run as run


class ProgressEvaluationTests(unittest.TestCase):
    def test_banks_respect_protected_inventory_and_explicit_s0_pair_permission(self):
        banks = {name: [
            dict(seed=10+i*10+k, offset=k, teams={'p1': f'{name}-{k}-A', 'p2': f'{name}-{k}-B'})
            for k in range(2)] for i, name in enumerate(progress.OPPONENTS)}
        inventory = dict(complete=True, seed_ranges=[], packed_team_sha256s=[])
        progress.validate_banks(banks, inventory, pairs_per_opponent=2)
        inventory['seed_ranges'] = [dict(start=10, stop=11, owner='training')]
        with self.assertRaisesRegex(ValueError, 'overlap'):
            progress.validate_banks(banks, inventory, pairs_per_opponent=2)
        inventory['seed_ranges'][0]['owner'] = 's0_registered_overlap'
        inventory['registered_s0_pairs'] = {'foul_play:10': run._sha(banks['foul_play'][0]['teams'])}
        with self.assertRaisesRegex(ValueError, 'overlap'):
            progress.validate_banks(banks, inventory, pairs_per_opponent=2)
        inventory['seed_ranges'] = [dict(start=20, stop=21, owner='s0_registered_overlap')]
        original = banks['champion_9375'][0]['teams']
        inventory['registered_s0_pairs'] = {'champion_9375:20': run._sha(original)}
        progress.validate_banks(banks, inventory, pairs_per_opponent=2)
        banks['champion_9375'][0]['teams'] = {'p1': original['p2'], 'p2': original['p1']}
        with self.assertRaisesRegex(ValueError, 'overlap'):
            progress.validate_banks(banks, inventory, pairs_per_opponent=2)

    def identity(self, *, seat='p1', checkpoint='c'):
        request = SimpleNamespace(snapshot=SimpleNamespace(checkpoint_sha256=checkpoint, archive_sha256='a'))
        profile = dict(simulator_sha256='s', actor_law_sha256='l', max_requests=250,
                       opponent_sha256s={'foul_play': 'fingerprint'}, opponent_law_sha256s={'foul_play': 'law'},
                       foul_play={'config': {'foulplay_random_seed': 31}})
        return progress.cell_identity(request, profile, 'foul_play',
            dict(seed=10, offset=0, teams=dict(p1='team-a', p2='team-b')), seat)

    def test_shared_registry_reuses_only_exact_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            cell = self.identity()
            play = Mock(return_value=('p1', 'terminal-witness'))
            first = progress.run_shared_cell(directory, cell, play)
            self.assertEqual(first['win'], 1)
            self.assertEqual(progress.run_shared_cell(directory, dict(cell), play), first)
            play.assert_called_once()
            other = self.identity(seat='p2')
            unknown = progress.run_shared_cell(directory, other, Mock(side_effect=RuntimeError('lost engine')))
            self.assertEqual(unknown['status'], 'UNKNOWN')
            self.assertEqual(progress.run_shared_cell(directory, other, play), unknown)
            play.assert_called_once()
            progress.run_shared_cell(directory, self.identity(checkpoint='different'), play)
            self.assertEqual(play.call_count, 2)

    def test_interrupted_cell_remains_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(KeyboardInterrupt):
                progress.run_shared_cell(directory, self.identity(), Mock(side_effect=KeyboardInterrupt))
            play = Mock()
            row = progress.run_shared_cell(directory, self.identity(), play)
            self.assertEqual((row['status'], row['win']), ('UNKNOWN', None))
            play.assert_not_called()

    def test_interval_groups_pairs_and_preserves_missing_denominator(self):
        win = dict(status='COMPLETE', win=1)
        loss = dict(status='COMPLETE', win=0)
        unknown = dict(status='UNKNOWN', win=None)
        independent = [[win, win]]*125 + [[win, loss]]*250 + [[loss, loss]]*125
        correlated = [[win, win]]*250 + [[loss, loss]]*250
        first = progress.summarize_panel(independent)
        second = progress.summarize_panel(correlated)
        self.assertAlmostEqual(first['ci95'][1]-.5, .031, delta=.001)
        self.assertAlmostEqual(second['ci95'][1]-.5, .044, delta=.001)
        missing = progress.summarize_panel([[win, unknown], [loss, None]])
        self.assertEqual((missing['games_completed'], missing['unknown'], missing['pending']), (2, 1, 1))
        self.assertEqual(missing['identification_bounds'], [.25, .75])

    def test_shards_cover_both_seats_of_every_pair_once(self):
        offsets = [i for shard in range(16) for lane in progress.shard_offsets(500, 16, shard, 4) for i in lane]
        self.assertEqual(sorted(offsets), list(range(500)))

    def test_foul_play_keeps_packed_physical_teams_when_model_seats_swap(self):
        from pokezero.foulplay_bridge import _controlled_players
        teams = dict(p1='packed-one', p2='packed-two')
        for seat, other in (('p1', 'p2'), ('p2', 'p1')):
            cfg = SimpleNamespace(pokezero_player=seat, foulplay_player=other,
                                  pokezero_username='Current', foulplay_username='FP')
            users = _controlled_players(cfg, progress._packed(teams))
            self.assertEqual(users[seat]['name'], 'Current')
            self.assertEqual({key: row['team'] for key, row in users.items()}, teams)
            self.assertEqual(_controlled_players(cfg, None), {seat: 'Current', other: 'FP'})

    def test_native_randbats_honors_both_packed_teams_in_pinned_showdown(self):
        from pokezero.local_showdown import default_showdown_root
        node = shutil.which('node')
        source = default_showdown_root()
        sim = source/'dist'/'sim'
        if not node or not (sim/'index.js').is_file():
            self.skipTest('pinned compiled Showdown is not installed')
        revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != 'f76228a1354b5d0f307ca2d16101294ad3a2308b':
            self.skipTest('requires bound Showdown revision')
        script = """
const {Battle, Teams} = require(process.argv[1]);
const battle = new Battle({formatid:'gen3randombattle', seed:[1,2,3,4]});
battle.setPlayer('p1',{name:'A',team:Teams.pack([{species:'Magikarp',moves:['splash'],level:73}])});
battle.setPlayer('p2',{name:'B',team:Teams.pack([{species:'Caterpie',moves:['tackle'],level:41}])});
console.log(JSON.stringify(battle.sides.map(side=>side.pokemon.map(p=>[p.species.id,p.level]))));
battle.destroy();
"""
        data = json.loads(subprocess.check_output([node, '-e', script, str(sim)], timeout=15))
        self.assertEqual(data, [[['magikarp', 73]], [['caterpie', 41]]])

    def test_final_roster_and_active_start_appear_in_progress(self):
        from pokezero.v5_evaluation import Artifact
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request_path = root/'request'
            request_path.write_text(json.dumps({'published_unix': 100}))
            dispatch = Artifact(str(request_path), run._file_sha(request_path))
            request = SimpleNamespace(declared_games=1_000_000, declaration_sha256='decl')
            banks = {n: [{}]*1000 for n in progress.OPPONENTS}
            profile = {'shared_cell_registry': str(root)}
            def identity(_request, _profile, opponent, _row, seat):
                return {'opponent': opponent, 'seat': seat}
            with patch.object(progress, 'load_profile', return_value=(request, profile, banks)), \
                    patch.object(progress, 'cell_identity', side_effect=identity), \
                    patch.object(progress, 'read_shared_cell', return_value=None), \
                    patch.object(progress, 'read_shared_start', return_value={'started_unix': 117}):
                row = progress.summarize(dispatch, dispatch, root)
            self.assertTrue(row['work_started'])
            self.assertEqual(row['published_unix'], 100)
            self.assertAlmostEqual(row['queue_wait_seconds'], 17)
            self.assertFalse(row['complete'])
            self.assertEqual([row['panel'][n]['games_declared'] for n in progress.OPPONENTS], [1000, 2000, 2000])

    def test_staged_request_keeps_original_bytes_and_checks_copied_artifacts(self):
        from pokezero import v5_deferred_evaluation as deferred, v5_evaluation as evaluation
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory).resolve()
            original = staging/'source-root-not-mounted'
            root = staging/'jokla'
            (root/'artifacts').mkdir(parents=True)
            checkpoint, archive = root/'artifacts'/'train.pt', root/'artifacts'/'actor.npz'
            checkpoint.write_bytes(b'test checkpoint')
            archive.write_bytes(b'test actor')
            protocol_sha = 'b'*64
            declaration = dict(model={'fixture': True}, iterations=[[{'games':[{}]}]],
                evaluation=run.EvaluationProtocol(protocol_sha, 1, 'deferred').metadata())
            (root/'declaration.json').write_text(run._json(declaration))
            snapshot = run.Snapshot(0, 'artifacts/train.pt', run._file_sha(checkpoint),
                'artifacts/actor.npz', run._file_sha(archive), run._sha(declaration['model']))
            original_request = run.EvaluationRequest(1, run._sha(declaration), protocol_sha, snapshot,
                original, original/snapshot.checkpoint, original/snapshot.archive)
            dispatch = deferred.publish_dispatch(original_request, None, original_request.declaration_sha256)
            copied = root/dispatch.request
            copied.parent.mkdir(parents=True)
            copied.write_bytes((original/dispatch.request).read_bytes())
            reference = evaluation.Artifact(str(copied), dispatch.request_sha256)
            self.assertEqual(deferred.read_milestone(reference, staged_run_root=root).archive_path, archive)
            value = json.loads(copied.read_bytes())
            self.assertEqual(value['request']['run_directory'], str(original))
            with patch.object(deferred.time, 'time', return_value=value['published_unix']+17):
                self.assertEqual(deferred.publish_dispatch(original_request, None, original_request.declaration_sha256),
                                 dispatch)
            archive.write_bytes(b'wrong actor')
            with self.assertRaisesRegex(ValueError, 'frozen evaluation input differs'):
                deferred.read_milestone(reference, staged_run_root=root)

    def test_mixed_showdown_keeps_9375_and_max_damage_on_their_own_seats(self):
        class Environment:
            def reset_with_start_override(self, *, seed, start_override):
                self.start = start_override.player_teams
                self.ended = None
                self.owner_lines = {s: ['|player|p1|Alpha', '|player|p2|Beta', f'|private-{s}']
                                    for s in ('p1', 'p2')}
            def terminal(self):
                return self.ended
            def requested_players(self):
                return ('p1', 'p2')
            def observe(self, seat):
                return seat
            def _choices_for_actions(self, actions):
                return {seat: 'move 1' for seat in actions}
            def step(self, actions):
                for stream in self.owner_lines.values():
                    stream.append('|win|Beta')
                self.ended = SimpleNamespace(winner='p2', capped=False)
        def owner():
            actor = Mock(generation=0, prefix_sha256='sha')
            actor.decide.return_value = SimpleNamespace(action_index=0)
            actor.choice.return_value = 'move 1'
            actor.close_game.return_value = {'status': 'CLOSED'}
            return actor
        row = dict(seed=123, teams=dict(p1='literal-a', p2='literal-b'))
        for seat, other in (('p1', 'p2'), ('p2', 'p1')):
            current, original = owner(), owner()
            current_factory, original_factory = Mock(), Mock()
            current_factory.create_actor.return_value = current
            original_factory.actor.return_value = original
            env = Environment()
            runtime = object()
            result = progress._mixed_game(env, current_factory, (original_factory, runtime),
                                          row, seat, 2, 'bound-cell')
            self.assertEqual(result, ('p2', '|win|Beta'))
            original_factory.actor.assert_called_once_with(runtime)
            self.assertIn(f'|private-{seat}', current.accept_owner_stream.call_args_list[0].kwargs['lines'])
            self.assertIn(f'|private-{other}', original.accept_owner_stream.call_args_list[0].kwargs['lines'])
            self.assertEqual(env.start, row['teams'])
            bot = Mock()
            bot.select_action.return_value = SimpleNamespace(action_index=0)
            progress._mixed_game(env, current_factory, bot, row, seat, 2, 'bot-cell')
            self.assertEqual(bot.select_action.call_args.args, (other,))

    def test_report_keeps_one_row_per_milestone_and_pending_lag(self):
        with tempfile.TemporaryDirectory() as directory:
            panel = {n: dict(games_declared=1_000, games_completed=900, unknown=100, pending=0,
                             ci95=[.2, .45], win_rate=.3) for n in progress.OPPONENTS}
            summary = dict(schema=progress.SCHEMA, declared_games=1_000_000, dispatch_sha256='a',
                protocol_sha256='b', declaration_sha256='c', complete=True, observed_unix=150,
                published_unix=100, work_started=True, queue_wait_seconds=15, eval_lag_seconds=50, panel=panel)
            root = Path(directory)
            report.update_report(root, [summary], training_games=3_000_000, now=200)
            table = (root/'showdown-progress.md').read_text()
            self.assertEqual(table.count('| 1,000,000 |'), 1)
            self.assertIn('| 3,000,000 | Queued', table)
            self.assertIn('2 milestones behind', table)
            self.assertIn('900 complete, 100 unknown', table)
            self.assertIn('<polyline', (root/'showdown-progress.svg').read_text())
            with self.assertRaisesRegex(ValueError, 'changed'):
                report.update_report(root, [dict(summary, dispatch_sha256='x')], training_games=3_000_000)
            self.assertEqual(json.loads((root/'showdown-progress.json').read_bytes())['rows'], [summary])


if __name__ == '__main__':
    unittest.main()
