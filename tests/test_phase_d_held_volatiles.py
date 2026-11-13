"""The protocol world's Yawn projection (phase_d_protocol_world._defer_yawn): pure payload and lines, no engine."""
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


if __name__ == '__main__':
    unittest.main()
