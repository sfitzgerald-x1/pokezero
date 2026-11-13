"""The protocol world's Yawn projection (phase_d_protocol_world._defer_yawn): pure payload and lines, no engine."""
import json
from pathlib import Path
import unittest

from gpu_engine import phase_d_protocol_world as protocol


def payload(p1=(), p2=()):
    return {'sides': {'p1': {'volatiles': list(p1), 'pokemon': []}, 'p2': {'volatiles': list(p2), 'pokemon': []}}}


class YawnProjectionTests(unittest.TestCase):
    start = ['|turn|3', '|move|p2a: Slowbro|Yawn|p1a: Snorlax', '|-start|p1a: Snorlax|move: Yawn|[of] p2a: Slowbro']

    def test_the_count_is_two_before_the_residual_and_one_after(self):
        """Showdown's Yawn: duration 2 at |-start|...|move: Yawn, -1 at each residual (its |upkeep|): 2 at a replacement
        boundary of that turn, 1 at the next decision. The volatile leaves the payload the CPU materializer sees."""
        for lines, count in ((self.start, 2), (self.start + ['|', '|upkeep', '|turn|4'], 1)):
            with self.subTest(count=count):
                projected, yawns = protocol._defer_yawn(payload(p1=('yawn', 'leechseed')), lines)
                self.assertEqual(yawns, ((0, count),))
                self.assertEqual(projected['sides']['p1']['volatiles'], ['leechseed'])

    def test_no_yawn_is_the_same_payload_and_an_unexplained_one_refuses(self):
        source = payload(p2=('leechseed',))
        projected, yawns = protocol._defer_yawn(source, self.start)
        self.assertEqual((projected, yawns), (source, ()))
        with self.assertRaisesRegex(ValueError, 'PUBLIC_YAWN_START_UNAVAILABLE'):
            protocol._defer_yawn(payload(p2=('yawn',)), self.start)   # the start line names p1's active
        with self.assertRaisesRegex(ValueError, 'PUBLIC_YAWN_COUNT_UNAVAILABLE'):
            protocol._defer_yawn(payload(p1=('yawn',)), self.start + ['|upkeep', '|turn|4', '|upkeep', '|turn|5'])


class PerishCountTests(unittest.TestCase):
    """phase_d_protocol_world._perish_counts on a real r4a3 prefix (ordinal 482, decision 5) and its mutants: Showdown sets
    duration 4 and prints a silent perish3 at the use; the residual decrements, then prints perishN before |upkeep|."""
    lines = json.loads((Path(__file__).parent / 'fixtures' / 'phase_d_held_volatiles' / 'perish_song_482_5.json').read_text())['lines']
    use = lines.index('|move|p2a: Misdreavus|Perish Song|p2a: Misdreavus')
    upkeep = lines.index('|upkeep', use)
    passed = ['|move|p1a: Blissey|Baton Pass|p1a: Blissey', '|switch|p1a: Chansey|Chansey, L80, F|300/300|[from] Baton Pass']

    def test_after_the_residual_the_shown_count(self):
        self.assertEqual(protocol._perish_counts(self.lines), (3, 3))

    def test_a_boundary_in_the_songs_own_turn_holds_four(self):
        """Mutant: the prefix cut after the use's silent perish3 lines (a mid-turn replacement before that turn's residual)."""
        cut = self.lines[:self.use + 4]
        self.assertEqual(cut[-1], '|-fieldactivate|move: Perish Song')
        self.assertEqual(protocol._perish_counts(cut), (4, 4))

    def test_a_baton_pass_carries_the_count_to_the_recipient(self):
        """Mutants: Blissey Baton Passes in the Song's own turn (the recipient holds 4) and after the residual (it holds 3);
        the side's latest line is still the passer's, as Showdown prints none for the recipient until the next residual."""
        same_turn = self.lines[:self.use + 4] + self.passed
        self.assertEqual(protocol._perish_counts(same_turn), (4, 4))
        later = self.lines[:self.upkeep + 2] + self.passed
        self.assertEqual(later[self.upkeep + 1], '|turn|6')
        self.assertEqual(protocol._perish_counts(later), (3, 3))

    def test_the_world_holds_four_in_the_songs_turn_and_refuses_an_unexplained_count(self):
        """_write_held_counts: map_spec wrote the shown 3 on both actives; in the Song's own turn the world holds 4, after the
        residual 3; Yawn's count lands on its side's active; a shown count without a perish line refuses."""
        import jax.numpy as j
        from gpu_engine.state import empty_state
        state = empty_state(1)._replace(active=j.asarray([[0, 2]], j.int32))
        state = state._replace(perish_counter=state.perish_counter.at[0, 0, 0].set(3).at[0, 1, 2].set(3))
        out = protocol._write_held_counts(state, ((1, 2),), protocol._perish_counts(self.lines[:self.use + 4]))
        self.assertEqual((int(out.perish_counter[0, 0, 0]), int(out.perish_counter[0, 1, 2])), (4, 4))
        self.assertEqual(int(out.yawn_counter[0, 1, 2]), 2)
        after = protocol._write_held_counts(state, (), protocol._perish_counts(self.lines))
        self.assertEqual((int(after.perish_counter[0, 0, 0]), int(after.perish_counter[0, 1, 2])), (3, 3))
        with self.assertRaisesRegex(ValueError, 'PUBLIC_PERISH_COUNT_UNAVAILABLE'):
            protocol._write_held_counts(state, (), protocol._perish_counts(self.lines[:self.use]))

    def test_no_perish_line_is_none_and_the_next_residual_counts_down(self):
        self.assertEqual(protocol._perish_counts(self.lines[:self.use]), (None, None))
        nxt = self.lines + ['|', '|-start|p2a: Misdreavus|perish2', '|-start|p1a: Blissey|perish2', '|upkeep', '|turn|7']
        self.assertEqual(protocol._perish_counts(nxt), (2, 2))


if __name__ == '__main__':
    unittest.main()
