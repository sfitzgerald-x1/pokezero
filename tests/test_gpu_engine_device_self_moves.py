"""V2 self_moves on the device encoder: the observer's own real moves and its engine PP on self tokens 1-6, as B2 and the
Python encoder write them; a peer observer reads ITS world's own state, never ours; the static override turns a view
off; the tables refuse self_moves without exact_state. Off (the catalog without self_moves): no own move cell.

Fixture: tests/test_v2_offline_view_parity.py's synthetic seven-species source (real catalog, real encoder tables, real
hooks), with a second catalog compiled with self_moves on."""
import importlib.util
import os
import unittest
from unittest import mock

from _showdown_root import requires_showdown

SHOWDOWN = requires_showdown('the synthetic fixture compiles its catalog from a pokemon-showdown checkout')
OWN = ('Snorlax', 'Gengar', 'Starmie', 'Skarmory', 'Blissey', 'Tyranitar')
FOE = ('Jolteon', 'Snorlax', 'Gengar', 'Starmie', 'Skarmory', 'Blissey')
SORTED = ('bodyslam', 'protect', 'rest', 'surf')   # both fixture variants' moves, lexical
MAX = {'bodyslam': 24, 'protect': 16, 'rest': 16, 'surf': 24}


@SHOWDOWN
@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
class DeviceSelfMovesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_v2_offline_view_parity as fx
        from gpu_engine import v4_encoder, public_belief, engine_device_observation as encoder
        f = fx.fixture()
        cls.fx, cls.f = fx, f
        masks = dict(f.catalog.masks.__dict__)
        with mock.patch.dict(os.environ, {'OBS_SELF_MOVES': ''}):
            cls.on = v4_encoder.compile_encoder(public_belief.compile_source(f.source), f.catalog.dex, f.catalog.vocabulary,
                checkpoint_sha256='a' * 64, feature_masks=dict(masks, self_moves=True))
            cls.no_exact = v4_encoder.compile_encoder(public_belief.compile_source(f.source), f.catalog.dex, f.catalog.vocabulary,
                checkpoint_sha256='a' * 64, feature_masks=dict(masks, self_moves=True, exact_state=False))
        cls.tables = {False: f.tables, True: encoder.compile_tables(f.metadata, catalog=cls.on)}

    def world(self):
        """Side 0's Snorlax active with PP spent (Body Slam 21, Rest 15), its bench Gengar fainted; side 1's Jolteon with
        Surf 3, so a peer view can only show side 1's own PP."""
        import jax.numpy as j
        f, fx = self.f, self.fx
        state, _ = fx.teams_state(f, [[fx.team(f, OWN, (0,) * 6), fx.team(f, FOE, (1,) * 6)]])
        names = f.metadata['vocabulary']['moves']
        pp = state.pp
        for side, slot, move, value in ((0, 0, 'bodyslam', 21), (0, 0, 'rest', 15), (1, 0, 'surf', 3)):
            at = [names[int(m)] for m in state.moves[0, side, slot]].index(move)
            pp = pp.at[0, side, slot, at].set(value)
        return state._replace(pp=pp, hp=state.hp.at[0, 0, 1].set(0))

    def encode(self, on, observer=0, **kwargs):
        import jax.numpy as j
        from gpu_engine import engine_device_public_hooks as hooks, engine_device_trackers as tr, engine_device_observation as encoder
        state = self.world()
        c = hooks.initialize(state, self.tables[on].belief)
        side = j.full(1, observer, j.int32)
        return encoder.encode(state, tr.readout(c.trackers, side), self.tables[on], observer_sides=side,
                              legal_action_mask=j.ones((1, 9), bool), **kwargs)

    def cells(self, out, token):
        import numpy as np
        from pokezero import showdown as s
        num, cat = np.asarray(out.numeric_features)[0, token], np.asarray(out.categorical_ids)[0, token]
        pp = [float(num[s.v4_numeric_index(s.NUMERIC_OPP_MOVE_PP_OFFSET + i)]) for i in range(16)]
        valid = [float(num[s.v4_numeric_index(s.NUMERIC_OPP_MOVE_PP_VALID_OFFSET + i)]) for i in range(16)]
        counts = (float(num[s.v4_numeric_index(s.NUMERIC_REVEALED_MOVE_COUNT)]), float(num[s.v4_numeric_index(s.NUMERIC_POSSIBLE_MOVE_COUNT)]))
        return [int(x) for x in cat[s.CATEGORY_BELIEF_MOVE_OFFSET:s.CATEGORY_BELIEF_MOVE_OFFSET + 16]], pp, valid, counts

    def test_off_has_no_own_move_cells(self):
        import numpy as np
        from test_self_moves_encoders import move_columns
        out = self.encode(False)
        numeric, cats = move_columns()
        self.assertFalse(np.asarray(out.numeric_features)[0, 1:7][:, numeric].any())
        self.assertFalse(np.asarray(out.categorical_ids)[0, 1:7][:, cats].any())

    def test_on_writes_engine_moves_and_pp_and_nothing_else(self):
        import numpy as np
        from test_self_moves_encoders import move_columns
        off, on = self.encode(False), self.encode(True)
        numeric, cats = move_columns()
        for name, columns in (('numeric_features', numeric), ('categorical_ids', cats)):
            diff = np.argwhere(np.asarray(getattr(off, name))[0] != np.asarray(getattr(on, name))[0])
            self.assertTrue(len(diff) and all(1 <= t <= 6 and c in columns for t, c in diff), (name, diff.tolist()))
        vocab = self.on.vocabulary
        labels = [vocab.encode('belief:possible_move:' + m) for m in SORTED] + [0] * 12
        for token in range(1, 7):   # every own member, the fainted Gengar (token 2) included
            cat, pp, valid, counts = self.cells(on, token)
            self.assertEqual((cat, valid[:4], counts), (labels, [1.0] * 4, (4.0, 4.0)), token)
            want = {'bodyslam': 21, 'rest': 15} if token == 1 else {}
            self.assertEqual(pp, [float(np.float32(want.get(m, MAX[m]) / MAX[m])) for m in SORTED] + [0.0] * 12, token)

    def test_peer_view_reads_its_own_world_state_never_ours(self):
        import numpy as np
        on = self.encode(True, observer=1)
        _, pp, valid, _ = self.cells(on, 1)                 # the peer's own active Jolteon
        self.assertEqual(pp[:4], [float(np.float32(x)) for x in (1.0, 1.0, 1.0, 3 / 24)])
        self.assertNotIn(float(np.float32(21 / 24)), pp)

    def test_transformed_active_self_tokens_keep_saved_moves_and_pp(self):
        import jax.numpy as j
        import numpy as np
        from gpu_engine import engine_device_observation as encoder, engine_device_public_hooks as hooks, engine_device_trackers as tr
        for observer in (0, 1):
            for slot in (0, 2):
                with self.subTest(observer=observer, slot=slot):
                    state = self.world(); state = state._replace(active=state.active.at[0, observer].set(slot))
                    side = j.asarray([observer], j.int32)
                    public = tr.readout(hooks.initialize(state, self.tables[True].belief).trackers, side)
                    encode = lambda value: encoder.encode(value, public, self.tables[True], observer_sides=side,
                        legal_action_mask=j.ones((1, 9), bool))
                    before = encode(state)
                    copied = j.roll(state.moves[0, observer, slot], 1).at[2:].set(0)
                    transformed = state._replace(transformed=state.transformed.at[0, observer].set(True),
                        transform_moves=state.transform_moves.at[0, observer].set(state.moves[0, observer, slot]),
                        transform_pp=state.transform_pp.at[0, observer].set(state.pp[0, observer, slot]),
                        transform_stats=state.transform_stats.at[0, observer].set(state.stats[0, observer, slot, 1:]),
                        moves=state.moves.at[0, observer, slot].set(copied),
                        pp=state.pp.at[0, observer, slot].set(j.asarray([5, 3, 0, 0], j.int32)))
                    after = encode(transformed)
                    for token in range(1, 7):self.assertEqual(self.cells(after, token), self.cells(before, token))
                    # Copied action slots still describe the transformed active, not its saved set.
                    self.assertFalse(np.array_equal(np.asarray(before.categorical_ids)[:, 13:17],
                        np.asarray(after.categorical_ids)[:, 13:17]))
                    spent = transformed._replace(pp=transformed.pp.at[0, observer, slot, 0].set(1))
                    for token in range(1, 7):self.assertEqual(self.cells(encode(spent), token), self.cells(before, token))

    def test_static_override_turns_a_view_off(self):
        import numpy as np
        a, b = self.encode(True, observer=1, self_moves=False), self.encode(False, observer=1)
        self.assertTrue(np.array_equal(np.asarray(a.numeric_features), np.asarray(b.numeric_features)))
        self.assertTrue(np.array_equal(np.asarray(a.categorical_ids), np.asarray(b.categorical_ids)))

    def test_tables_refuse_self_moves_without_exact_state(self):
        from gpu_engine import engine_device_observation as encoder
        with self.assertRaisesRegex(ValueError, 'exact_state'):
            encoder.compile_tables(self.f.metadata, catalog=self.no_exact)

    def test_encoder_identity_differs_only_when_on(self):
        from gpu_engine import constructor_worlds as cw
        off, on = cw.encoder_identity(self.f.catalog), cw.encoder_identity(self.on)
        self.assertEqual(on[:-1], off)
        self.assertEqual(on[-1], ('self_moves', True))


if __name__ == '__main__':
    unittest.main()
