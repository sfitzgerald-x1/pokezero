"""mc_ladder_r0prime's host rules: the bank row's draw count and the worker's exit code. The replay itself needs a Showdown runtime
and runs pod-side."""
import unittest

from gpu_engine import mc_ladder_r0prime as P


def counts(**over):
    out = dict(roots=3, written=3, not_replayed=0, root_mismatch=0, reply_mismatch=0, errors=0, r0_checked=1, r0_max_abs=4e-7)
    out.update(over);return out


class R0PrimeRuleTests(unittest.TestCase):
    def test_draws_come_from_the_bank_row(self):
        self.assertEqual(P.bank_draws(dict(baseline=dict(child_values={'0,1': [.1] * 8, '1,1': [.2] * 8}))), 8)
        for values in ({'0,1': [.1] * 8, '1,1': [.2] * 4}, {}, {'0,None': []}):
            with self.assertRaises(ValueError):P.bank_draws(dict(baseline=dict(child_values=values)))

    def test_the_worker_passes_only_when_every_root_was_written_and_r0_rechecks(self):
        self.assertEqual(P.verdict(counts()), 0)
        for bad in (dict(written=2, not_replayed=1), dict(written=2, root_mismatch=1), dict(written=2, reply_mismatch=1),
                    dict(written=2, errors=1), dict(roots=0, written=0), dict(r0_max_abs=2e-5)):
            self.assertEqual(P.verdict(counts(**bad)), 1, bad)


if __name__ == '__main__':
    unittest.main()
