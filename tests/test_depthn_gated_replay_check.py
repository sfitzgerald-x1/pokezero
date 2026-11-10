"""depthn_gated_replay_check: a NOISE_GATED_K2 replay's rows held to the scorer's rule (br_fp worlds under a reply model)."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from gpu_engine import search_hedge as hedge
from scripts import depthn_gated_replay_check as C
from tests.test_gpu_engine_mc_ladder_rungs import D1, D2, NG, LEGAL, REPLY, row


def gated(depth2, *, played=None, operator='NOISE_GATED_K2', gate=True):
    """A br_fp + NOISE_GATED_K2 replay row on `depth2`'s grids: the replay's q is br_fp's, its action the gate's unless `played`."""
    raw = row(depth2=depth2);d = C.rr.normalize(raw);base = C.rr.recompute(d)
    worlds = C.rr.model_worlds(base.worlds, C.rr.reply_logits_of(d, base.worlds, None));choice = hedge.averaging(worlds, tuple(LEGAL))
    candidate = hedge.top_action(choice.strategy);greedy = 0
    rule = hedge.noise_gated(hedge.per_world_q(hedge.grids(worlds, tuple(LEGAL))), candidate, greedy, 2)
    action = rule if played is None else played
    raw['replay'].update(action_index=action, q_values=list(choice.q), root_operator=operator, reply_prior=dict(operator='br_fp'),
                         **({'noise_gate': dict(candidate=candidate, greedy=greedy, passed=action == candidate)} if gate else {}))
    return raw, candidate, rule


class GatedReplayCheckTests(unittest.TestCase):
    def test_a_replay_that_plays_the_rule_matches(self):
        for grids, want in ((D1, 1), (NG, 0), (D2, 0)):   # D1: the gap passes; NG and D2: br_fp plays 1 but the gate keeps greedy
            raw, candidate, rule = gated(grids);self.assertEqual((candidate, rule), (1, want))
            report = C.check([raw]);self.assertEqual((report['compared'], report['mismatches']), (1, {}), grids)

    def test_mismatches_are_counted_by_field(self):
        raw, _, _ = gated(NG, played=1)   # the ungated br_fp choice where the gate keeps greedy
        self.assertEqual(C.check([raw])['mismatches'], dict(action=1))
        raw, _, _ = gated(D1, operator='ARGMAX_MEAN_COMPLETED_Q');self.assertEqual(C.check([raw])['mismatches'], dict(root_operator=1))
        raw, _, _ = gated(D1, gate=False);self.assertEqual(C.check([raw])['mismatches'], dict(noise_gate=1))
        raw, _, _ = gated(D1);raw['replay']['q_values'][1] += 1e-9;self.assertEqual(C.check([raw])['mismatches'], dict(q_values=1))

    def test_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            good, bad, none = (Path(tmp) / f'{n}.jsonl' for n in ('good', 'bad', 'none'))
            good.write_text(json.dumps(gated(D1)[0]) + '\n');bad.write_text(json.dumps(gated(NG, played=1)[0]) + '\n')
            none.write_text(json.dumps(row(status='REFUSED')) + '\n')
            with redirect_stdout(io.StringIO()):
                self.assertEqual([C.main([str(p)]) for p in (good, bad, none)], [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
