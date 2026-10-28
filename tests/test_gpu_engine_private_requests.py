"""Private action boundary parity against Python, independent of engine state."""
import copy
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("jax"), "requires isolated JAX")
class PrivateRequestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        from gpu_engine import private_requests as p
        cls.jax, cls.j, cls.p = jax, j, p

    def request(self, team_size=6, active=0, *, owner=0, moves=None):
        if moves is None:
            moves = [{"id": "surf", "pp": 0, "maxpp": 24}, {"id": "toxic", "disabled": True}]
        return {"rqid": 17, "side": {"id": f"p{owner + 1}", "pokemon": [
            {"ident": f"p{owner + 1}: same-name", "details": "Pikachu, L80", "active": i == active,
             "condition": "0 fnt" if i == (active + 1) % team_size and i != active else "100/200"}
            for i in range(team_size)]}, "active": [{"moves": moves}]}

    def test_batched_legality_target_and_pp_parity_across_request_corpus(self):
        from pokezero.showdown import _legal_action_mask, _move_pp_fraction
        from pokezero.actions import canonical_switch_action_map
        profiles = [[], [{"pp": 0, "maxpp": 24}], [{"disabled": True}],
                    [{"pp": 4, "maxpp": 8}, None, {"pp": 3, "maxpp": 0}, {"disabled": True}],
                    [{"pp": -2, "maxpp": 4}, {"pp": 99, "maxpp": 8}, {}, {"pp": 1, "maxpp": -2}]]
        flags = [{}, {"wait": True}, {"forceSwitch": [True]}, {"forceSwitch": [False]},
                 {"trapped": True}, {"maybeTrapped": True}, {"teamPreview": True}]
        cases, reference, targets, pp = [], [], [], []
        for size in range(1, 7):
            for active in range(size):
                for moves in profiles:
                    for mode in flags:
                        request = self.request(size, active, moves=moves)
                        if "trapped" in mode or "maybeTrapped" in mode:
                            request["active"][0].update(mode)
                        else:
                            request.update(mode)
                        before = copy.deepcopy(request)
                        cases.append(self.p.parse_action_request(request, owner_side=0))
                        self.assertEqual(request, before)
                        reference.append(_legal_action_mask(request))
                        mapped = canonical_switch_action_map(active, team_size=size) if size > 1 else ()
                        targets.append(mapped + (-1,) * (5 - len(mapped)))
                        # Writer leaves absent/invalid action rows at zero;
                        # _move_pp_fraction defaults to one only for a Mapping.
                        pp.append([_move_pp_fraction(m) if isinstance(m, dict) else 0 for m in moves] + [0] * (4-len(moves)))
        batch = self.jax.tree.map(lambda *xs: self.j.stack(xs), *cases)
        eager = self.p.action_inputs(batch, observer_side=0)
        compiled = self.jax.jit(lambda x: self.p.action_inputs(x, observer_side=0))(batch)
        self.assertEqual(len(cases), 735)
        self.assertEqual(compiled.legal.tolist(), [list(row) for row in reference])
        self.assertEqual(compiled.switch_targets.tolist(), [list(row) for row in targets])
        self.assertTrue(bool(self.j.allclose(compiled.move_pp_fraction, self.j.array(pp), atol=1e-6, rtol=0)))
        for x, y in zip(eager, compiled): self.assertTrue(bool(self.j.array_equal(x, y)))

    def test_force_switch_overrides_trap_and_zero_pp_does_not_disable(self):
        raw = self.request(); packet = self.p.parse_action_request(raw, owner_side=0)
        got = self.p.action_inputs(packet, observer_side=0)
        self.assertTrue(bool(got.legal[0]))
        self.assertEqual(float(got.move_pp_fraction[0]), 0)
        raw["active"][0]["trapped"] = True
        raw["forceSwitch"] = [True]
        got = self.p.action_inputs(self.p.parse_action_request(raw, owner_side=0), observer_side=0)
        self.assertFalse(bool(got.legal[:4].any()))
        self.assertTrue(bool(got.legal[4:].any()))

    def test_pp_matches_action_writer_for_missing_invalid_and_padded_rows(self):
        from types import SimpleNamespace
        from pokezero import showdown as s
        raw = self.request(moves=[{}, None, {"pp": 3, "maxpp": 6}])
        team = s._self_team_from_request(raw, "p1")
        state = SimpleNamespace(request=raw, self_team=team, self_active=team[0],
                                legal_action_mask=s._legal_action_mask(raw))
        numeric = [[0.0] * 200 for _ in range(23)]
        categories = [[""] * 60 for _ in range(23)]
        s._encode_action_tokens(categories, numeric, state)
        expected = [numeric[s.ACTION_CANDIDATE_TOKEN_OFFSET + i][s.NUMERIC_MOVE_PP_FRACTION] for i in range(4)]
        got = self.p.action_inputs(self.p.parse_action_request(raw, owner_side=0), observer_side=0)
        self.assertEqual(expected, [1, 0, .5, 0])
        self.assertEqual(got.move_pp_fraction.tolist(), expected)

    def test_wrong_seat_and_absent_waiting_packets_cannot_fabricate_actions(self):
        packet = self.p.parse_action_request(self.request(owner=1), owner_side=1)
        denied = self.jax.jit(lambda x: self.p.action_inputs(x, observer_side=0))(packet)
        self.assertFalse(bool(denied.ready))
        self.assertFalse(bool(denied.legal.any()))
        self.assertEqual(denied.switch_targets.tolist(), [-1] * 5)
        self.assertEqual(denied.move_pp_fraction.tolist(), [0] * 4)
        for raw in [None, {"wait": True}, {}, {"forceSwitch": [True]}, {"active": []}]:
            got = self.p.action_inputs(self.p.parse_action_request(raw, owner_side=0), observer_side=0)
            self.assertFalse(bool(got.legal.any()))
        with self.assertRaisesRegex(ValueError, "different seat"):
            self.p.parse_action_request(self.request(owner=1), owner_side=0)
        for observer in [1.9, True, self.j.array(1.9), self.j.array(True), -1, 2]:
            with self.subTest(observer=observer), self.assertRaises(ValueError):
                self.p.action_inputs(packet, observer_side=observer)
        denied = self.p.action_inputs(packet, observer_side=self.j.array(2, self.j.int32))
        self.assertFalse(bool(denied.legal.any()))
        self.assertEqual(denied.move_pp_fraction.tolist(), [0] * 4)

    def test_canonical_targets_follow_party_position_even_for_duplicate_identities(self):
        raw = self.request(active=4)
        # Every nickname/species is identical. Identity cannot substitute for a slot.
        got = self.p.action_inputs(self.p.parse_action_request(raw, owner_side=0), observer_side=0)
        self.assertEqual(got.switch_targets.tolist(), [0, 1, 2, 3, 5])
        self.assertFalse(bool(got.legal[8]))  # team slot5 is fainted in this fixture

    def test_unsupported_shapes_nonfinite_pp_and_invalid_owner_fail_admission(self):
        raw = self.request()
        bad = [dict(raw, active=[{}, {}]), dict(raw, forceSwitch=[True, False]),
               dict(raw, rqid=2**31), dict(raw, side={"pokemon": [{}] * 7}),
               dict(raw, active=[{"moves": [{}] * 5}]),
               dict(raw, active=[{"moves": [{"pp": float("nan"), "maxpp": 8}]}])]
        for request in bad:
            with self.subTest(request=request), self.assertRaises(ValueError):
                self.p.parse_action_request(request, owner_side=0)
        for owner in [-1, 2, True]:
            with self.assertRaises(ValueError): self.p.parse_action_request(raw, owner_side=owner)
        for a, b in [(5e-50, 1e-49), (5e-45, 1e-44), (10**1000, 8)]:
            with self.assertRaises(ValueError):
                self.p.parse_action_request(self.request(moves=[{"pp": a, "maxpp": b}]), owner_side=0)

    def test_unrelated_opponent_payload_does_not_enter_private_packet(self):
        raw = self.request()
        before = self.p.parse_action_request(raw, owner_side=0)
        raw["opponent"] = {"secret_items": ["Choice Band"] * 6, "stats": [999] * 6}
        after = self.p.parse_action_request(raw, owner_side=0)
        for x, y in zip(before, after): self.assertTrue(bool(self.j.array_equal(x, y)))


if __name__ == "__main__":
    unittest.main()
