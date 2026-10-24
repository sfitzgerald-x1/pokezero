"""P2 layout/missingness contracts; no games or inferential rule changes."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
from pokezero.mcts_eval import fullgame_strength as reader
from phase0_paired_dry_run import checked_outcome
from p2_paired_dry_run import SHARDS, describe_panel, main, frozen_module


def entry(directory):
    return dict(schema='pokezero.p2-informed-one-ply-gameplay-entry.v1',
        information_model='informed_adversary_root_minimax', seed_start=100,
        seed_count=128, run_directory=str(directory),
        common_arguments=['--max-decision-rounds', '250'],
        shards=[dict(name=name, seat=seat, arguments=[
            '--policy-mode', policy, '--opponent-policy-mode', opponent])
            for name, seat, opponent, policy, _ in SHARDS])


def game(seed, seat, score=1, opponent_name='FoulPlayBot'):
    winner_seat = seat if score else ('p2' if seat == 'p1' else 'p1')
    players = {seat: 'PokeZeroBot', 'p2' if seat == 'p1' else 'p1': opponent_name}
    winner = players[winner_seat]
    return dict(seed=seed, battle_id=f'battle-gen3randombattle-controlled-{seed}',
        decision_rounds=20, capped=False, tied=False, pokezero_score=score,
        pokezero_won=bool(score), winner=winner,
        terminal_witness=dict(schema='pokezero.controlled-terminal-witness.v1',
            status='PUBLIC_WIN', capped=False, decision_rounds=20,
            max_decision_rounds=250, players=players, winner_player=winner_seat,
            public_terminal_line='|win|' + winner))


def write_shard(directory, name, rows):
    _, seat, opponent, policy, _ = next(s for s in SHARDS if s[0] == name)
    opponent_id = opponent if opponent == 'foul-play' else 'pokezero-' + opponent
    doc = dict(pokezero_player=seat, opponent_policy_id=opponent_id, policy_mode=policy,
               completed_games=len(rows), game_results=rows)
    (Path(directory) / (name + '.json')).write_text(json.dumps(doc))
    return doc


class P2ReadoutTests(unittest.TestCase):
    def test_empty_panel_preserves_all_768_cells_and_no_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = describe_panel(entry(tmp), reader, checked_outcome, tmp)
        self.assertEqual((out['planned_game_instances'], out['planned_primary_cells'],
                          out['planned_h2h_cells']), (768, 512, 256))
        self.assertEqual(out['result']['identification_bounds'], [-1, 1])
        self.assertIsNone(out['result']['primary_test'])
        self.assertFalse(out['result']['screening_rule_passed'])

    def test_h2h_never_enters_the_primary_and_both_seats_are_oriented(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, seat, _, _, arm in SHARDS:
                opponent_name = 'PokeZeroRawBot' if arm == 'h2h' else 'FoulPlayBot'
                write_shard(tmp, name, [game(s, seat, int(arm == 'h2h'), opponent_name)
                                       for s in range(100, 228)])
            out = describe_panel(entry(tmp), reader, checked_outcome, tmp)
        self.assertEqual((out['recorded_primary_cells'], out['recorded_h2h_cells']), (512, 256))
        self.assertEqual(out['h2h_statuses'], {'COMPLETE': 256})
        self.assertEqual(out['result']['identification_bounds'], [0, 0])
        self.assertIsNone(out['result']['primary_test'])  # Even a complete snapshot stays descriptive.

    def test_h2h_wire_labels_default_and_custom_names_in_both_seats(self):
        for name, seat in [('h2h-p1', 'p1'), ('h2h-p2', 'p2')]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                e = entry(tmp)
                write_shard(tmp, name, [game(100, seat, opponent_name='PokeZeroRawBot')])
                out = describe_panel(e, reader, checked_outcome, tmp)
                self.assertEqual(out['h2h_statuses'], {'COMPLETE': 1})
                self.assertEqual(out['result']['identification_bounds'], [-1, 1])
                e['common_arguments'].extend(['--foulplay-username', 'CustomRaw'])
                write_shard(tmp, name, [game(100, seat, 0, opponent_name='CustomRaw')])
                self.assertEqual(describe_panel(e, reader, checked_outcome, tmp)
                                 ['h2h_statuses'], {'COMPLETE': 1})
                # The old fabricated FoulPlay display name is not a valid raw terminal.
                write_shard(tmp, name, [game(100, seat)])
                self.assertEqual(describe_panel(e, reader, checked_outcome, tmp)
                                 ['h2h_statuses'], {'FAILED': 1})

    def test_caps_and_conflicting_terminals_stay_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap = game(100, 'p1');cap.update(capped=True, decision_rounds=250,
                winner=None, pokezero_won=False, pokezero_score=.5)
            cap['terminal_witness'].update(status='ROUND_CAP', capped=True,
                decision_rounds=250, winner_player=None, public_terminal_line=None)
            conflict = game(101, 'p1');conflict['terminal_witness']['public_terminal_line'] = '|tier|x'
            write_shard(tmp, 'candidate-p1', [cap, conflict])
            out = describe_panel(entry(tmp), reader, checked_outcome, tmp)
        self.assertEqual(out['snapshots']['candidate-p1']['statuses'], {'CAPPED': 1, 'FAILED': 1})
        self.assertEqual(out['result']['identification_bounds'], [-1, 1])

    def test_wrong_policy_opponent_seat_or_nonchronological_seed_is_rejected(self):
        for key, val in [('policy_mode', 'engine-mcts'), ('opponent_policy_id', 'raw'),
                         ('pokezero_player', 'p2'), ('completed_games', True)]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                d = write_shard(tmp, 'candidate-p1', [game(100, 'p1')]);d[key] = val
                (Path(tmp) / 'candidate-p1.json').write_text(json.dumps(d))
                with self.assertRaisesRegex(ValueError, 'chronological prefix'):
                    describe_panel(entry(tmp), reader, checked_outcome, tmp)
        with tempfile.TemporaryDirectory() as tmp:
            write_shard(tmp, 'candidate-p1', [game(101, 'p1')])
            with self.assertRaisesRegex(ValueError, 'chronological prefix'):
                describe_panel(entry(tmp), reader, checked_outcome, tmp)

    def test_duplicate_missing_shard_or_policy_override_is_rejected(self):
        for mutate in (lambda e:e['shards'].pop(),
                       lambda e:e['shards'].__setitem__(1, copy.deepcopy(e['shards'][0])),
                       lambda e:e['shards'][0]['arguments'].extend(['--policy-mode', 'raw'])):
            with tempfile.TemporaryDirectory() as tmp:
                e = entry(tmp);mutate(e)
                with self.assertRaises(ValueError):describe_panel(e, reader, checked_outcome, tmp)

    def test_entry_hash_is_checked_before_any_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'entry.json';path.write_text('{}')
            with patch.object(sys, 'argv', ['reader', '--entry', str(path),
                    '--expected-entry-sha256', 'wrong']), self.assertRaisesRegex(ValueError, 'entry hash'):
                main()

    def test_frozen_sources_come_from_registered_commit_and_reader_hash_is_enforced(self):
        payload = (ROOT / 'src/pokezero/mcts_eval/fullgame_strength.py').read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        with patch('p2_paired_dry_run.subprocess.check_output', return_value=payload) as git:
            module, actual = frozen_module('/repo', 'a'*40,
                'src/pokezero/mcts_eval/fullgame_strength.py', digest)
            self.assertEqual(actual, digest)
            self.assertEqual(module.PRIMARY_SEEDS, 128)
            self.assertEqual(git.call_args.args[0][-1], 'a'*40 + ':src/pokezero/mcts_eval/fullgame_strength.py')
            with self.assertRaisesRegex(ValueError, 'reader hash'):
                frozen_module('/repo', 'a'*40, 'same.py', 'wrong')
        with self.assertRaisesRegex(ValueError, 'source commit'):
            frozen_module('/repo', 'main', 'same.py')


if __name__ == '__main__':
    unittest.main()
