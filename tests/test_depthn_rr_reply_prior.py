import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts import depthn_rr_reply_prior as R


def sidecar(path, roots, *, rng, schema=R.SIDECAR_SCHEMA):
    """roots: [(ordinal, decision_index, [(requested, checked), ...])]; root_peer_logits = the fake champion's logits."""
    cols = {k: [] for k in ('numeric_features', 'categorical_ids', 'token_type_ids', 'attention_mask', 'history_mask', 'legal', 'ordinal',
                            'decision_index', 'world', 'root_peer_logits', 'opponent_requested', 'self_check')}
    for ordinal, decision, worlds in roots:
        for w, (requested, checked) in enumerate(worlds):
            numeric = rng.standard_normal((23, 132)).astype(np.float32)
            cols['numeric_features'].append(numeric);cols['categorical_ids'].append(rng.integers(0, 50, (23, 41)).astype(np.int32))
            cols['token_type_ids'].append(np.arange(23, dtype=np.int32));cols['attention_mask'].append(np.ones(23, bool))
            cols['history_mask'].append(np.ones(1, bool));cols['legal'].append(np.ones(9, bool))
            cols['ordinal'].append(ordinal);cols['decision_index'].append(decision);cols['world'].append(w)
            cols['root_peer_logits'].append(champion_logits(numeric[None])[0]);cols['opponent_requested'].append(requested)
            cols['self_check'].append(1 if checked else -1)
    arrays = {k: np.asarray(v) for k, v in cols.items()}
    arrays['ordinal'] = arrays['ordinal'].astype(np.int64);arrays['decision_index'] = arrays['decision_index'].astype(np.int64)
    arrays['world'] = arrays['world'].astype(np.int32);arrays['self_check'] = arrays['self_check'].astype(np.int8)
    np.savez_compressed(path, header=np.asarray(json.dumps(dict(schema=schema))), **arrays)
    return arrays


def champion_logits(numeric):return (numeric[:, 0, :9] * 2).astype(np.float32)


def champion_forward(inputs):
    x = inputs['numeric_features'];assert x.shape[1:] == (1, 29, 132) and not x[:, :, 23:].any()
    return champion_logits(x[:, 0])


def reply_forward(inputs):return -inputs['numeric_features'][:, 0, 0, :9]


class ReplyPriorTests(unittest.TestCase):
    def setUp(self):self.tmp = TemporaryDirectory();self.root = Path(self.tmp.name);self.rng = np.random.default_rng(3)

    def tearDown(self):self.tmp.cleanup()

    def test_rows_for_requested_worlds_and_a_passing_self_check(self):
        a = sidecar(self.root / 'peer-inputs-0.npz', [(4, 0, [(True, True), (False, False), (True, True)]), (4, 1, [(True, True)])], rng=self.rng)
        rows, summary = R.reply_prior([R.read_sidecar(self.root / 'peer-inputs-0.npz')], champion_forward, reply_forward)
        self.assertEqual([(r['ordinal'], r['decision_index'], r['world']) for r in rows], [(4, 0, 0), (4, 0, 2), (4, 1, 0)])
        np.testing.assert_array_equal(np.asarray(rows[1]['logits'], np.float32), -a['numeric_features'][2, 0, :9])
        self.assertEqual((summary['roots'], summary['self_check']['checked'], summary['self_check']['equal'], summary['self_check']['passed']), (2, 3, 3, True))

    def test_a_self_check_failure_is_reported(self):
        sidecar(self.root / 's.npz', [(1, 0, [(True, True)])], rng=self.rng)
        off = lambda inputs: champion_forward(inputs) + np.float32(1e-6)
        _, summary = R.reply_prior([R.read_sidecar(self.root / 's.npz')], off, reply_forward)
        self.assertFalse(summary['self_check']['passed']);self.assertEqual(summary['self_check']['failures'][0]['world'], 0)

    def test_inputs_are_padded_to_the_v2_layout(self):
        a = sidecar(self.root / 's.npz', [(1, 0, [(True, True), (True, True)])], rng=self.rng)
        x = R.v2_inputs(a, np.array([1, 0]))
        self.assertEqual((x['numeric_features'].shape, x['categorical_ids'].shape, x['token_type_ids'].shape, x['history_mask'].shape),
                         ((2, 1, 29, 132), (2, 1, 29, 41), (2, 1, 29), (2, 1)))
        np.testing.assert_array_equal(x['numeric_features'][0, 0, :23], a['numeric_features'][1])

    def test_schema_and_duplicate_roots_are_refused(self):
        sidecar(self.root / 'bad.npz', [(1, 0, [(True, True)])], rng=self.rng, schema='other')
        with self.assertRaises(ValueError):R.read_sidecar(self.root / 'bad.npz')
        sidecar(self.root / 'a.npz', [(1, 0, [(True, True)])], rng=self.rng);sidecar(self.root / 'b.npz', [(1, 0, [(True, True)])], rng=self.rng)
        with self.assertRaises(ValueError):
            R.reply_prior([R.read_sidecar(self.root / 'a.npz'), R.read_sidecar(self.root / 'b.npz')], champion_forward, reply_forward)


if __name__ == '__main__':
    unittest.main()
