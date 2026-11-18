"""The fused production entry keeps learner acceleration options explicit."""
import unittest

from gpu_engine.v2_fused import argument_parser


class LearnerOptionsTests(unittest.TestCase):
    def parse(self, *options):
        return argument_parser().parse_args([
            '--config', 'diagnostic.json', '--reference-run', 'reference', '--out', 'output', *options])

    def test_existing_commands_leave_both_options_disabled(self):
        args = self.parse()
        self.assertFalse(args.native_embedding_backward)
        self.assertFalse(args.fallback_random)
        self.assertEqual(args.mode, 'cudagraphs')

    def test_native_backward_does_not_enable_random_fallback(self):
        args = self.parse('--native-embedding-backward')
        self.assertTrue(args.native_embedding_backward)
        self.assertFalse(args.fallback_random)

    def test_random_fallback_can_be_selected_independently_for_controls(self):
        for native in (False, True):
            with self.subTest(native=native):
                args = self.parse('--fallback-random', *(['--native-embedding-backward'] if native else []))
                self.assertEqual(args.native_embedding_backward, native)
                self.assertTrue(args.fallback_random)


if __name__ == '__main__':
    unittest.main()
